#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort

from onnx_test_utils import prepare_image, set_cv2_threads


def _session_options() -> ort.SessionOptions:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    return so


def _remove_stale_artifacts(output_path: Path) -> None:
    candidates = [
        output_path,
        output_path.with_name(output_path.name + ".data"),
    ]
    for candidate in candidates:
        if candidate.exists():
            candidate.unlink()


def _save_model(model: onnx.ModelProto, output_path: Path) -> None:
    onnx.save_model(
        model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=output_path.name + ".data",
        size_threshold=1024,
        convert_attribute=False,
    )


def _file_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024.0 * 1024.0)


def _artifact_total_size_mb(model_path: Path) -> float:
    external_data = model_path.with_name(model_path.name + ".data")
    return _file_size_mb(model_path) + _file_size_mb(external_data)


def _make_sample_input(input_shape: Iterable[int], image_path: str | None, seed: int) -> np.ndarray:
    shape = list(input_shape)
    if len(shape) != 4:
        raise RuntimeError(f"Unexpected encoder input shape: {shape}")
    _, channels, height, width = shape
    if channels != 3:
        raise RuntimeError(f"Unexpected encoder channel count: {channels}")

    if image_path:
        image = cv2.imread(image_path)
        if image is None or image.size == 0:
            raise RuntimeError(f"Could not read image: {image_path}")
        tensor, _ = prepare_image(image, (height, width))
        return tensor.astype(np.float32, copy=False)

    rng = np.random.default_rng(seed)
    return rng.standard_normal((1, channels, height, width)).astype(np.float32)


def _benchmark_encoder(session: ort.InferenceSession,
                       input_name: str,
                       sample: np.ndarray,
                       warmup: int,
                       runs: int) -> float:
    for _ in range(max(warmup, 0)):
        session.run(None, {input_name: sample})

    elapsed_ms = []
    for _ in range(max(runs, 0)):
        start = time.perf_counter()
        session.run(None, {input_name: sample})
        elapsed_ms.append((time.perf_counter() - start) * 1000.0)

    if not elapsed_ms:
        return 0.0
    return float(sum(elapsed_ms) / len(elapsed_ms))


def _compare_outputs(reference: list[np.ndarray], candidate: list[np.ndarray]) -> tuple[float, float]:
    mean_abs = 0.0
    max_abs = 0.0
    counted = 0
    for ref, cand in zip(reference, candidate):
        if ref.shape != cand.shape:
            continue
        diff = np.abs(ref.astype(np.float32) - cand.astype(np.float32))
        mean_abs += float(diff.mean())
        max_abs = max(max_abs, float(diff.max()))
        counted += 1

    if counted == 0:
        return 0.0, 0.0
    return mean_abs / counted, max_abs


def _update_manifest(ckpt_dir: Path, output_path: Path) -> None:
    manifest_path = ckpt_dir / "manifest.json"
    if not manifest_path.exists():
        return

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] Could not update manifest: {exc}")
        return

    artifacts = manifest.setdefault("artifacts", {})
    output_data = output_path.with_name(output_path.name + ".data")
    artifacts["image_encoder_int8"] = {
        "path": output_path.name,
        "exists": output_path.exists(),
        "has_external_data": output_data.exists(),
    }

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _load_graph_counter(model_path: Path) -> Counter[str]:
    model = onnx.load_model(model_path, load_external_data=False)
    return Counter(node.op_type for node in model.graph.node)


def _quantized_matmul_count(counter: Counter[str]) -> int:
    return (
        counter.get("MatMulInteger", 0)
        + counter.get("QGemm", 0)
        + counter.get("DynamicQuantizeLinear", 0)
    )


def _producer_by_output(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    producer: dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        for output in node.output:
            producer[output] = node
    return producer


def _find_attr_ints(node: onnx.NodeProto, name: str) -> list[int] | None:
    for attr in node.attribute:
        if attr.name == name:
            return list(attr.ints)
    return None


def _count_foldable_weight_transposes_in_matmul_rhs(model_path: Path) -> int:
    model = onnx.load_model(model_path, load_external_data=False)
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    matmul_rhs = Counter(
        node.input[1]
        for node in model.graph.node
        if node.op_type == "MatMul" and len(node.input) >= 2
    )

    count = 0
    for node in model.graph.node:
        if node.op_type != "Transpose" or len(node.input) != 1 or len(node.output) != 1:
            continue
        if matmul_rhs.get(node.output[0], 0) <= 0:
            continue
        if node.input[0] not in initializer_names:
            continue
        count += 1
    return count


def _prune_unused_initializers(model: onnx.ModelProto) -> int:
    graph_input_names = {value.name for value in model.graph.input}
    graph_output_names = {value.name for value in model.graph.output}
    used_names = set(graph_output_names)
    for node in model.graph.node:
        for name in node.input:
            if name:
                used_names.add(name)

    kept = []
    removed = 0
    for initializer in model.graph.initializer:
        if (
            initializer.name in used_names
            or initializer.name in graph_input_names
            or initializer.name in graph_output_names
        ):
            kept.append(initializer)
        else:
            removed += 1

    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    return removed


def _canonicalize_model_for_quantization(input_path: Path, output_path: Path) -> tuple[int, int]:
    model = onnx.load_model(input_path)
    initializer_by_name = {initializer.name: initializer for initializer in model.graph.initializer}
    matmul_rhs = Counter(
        node.input[1]
        for node in model.graph.node
        if node.op_type == "MatMul" and len(node.input) >= 2
    )

    folded = 0
    new_initializers = []
    kept_nodes = []

    for node in model.graph.node:
        if node.op_type != "Transpose" or len(node.input) != 1 or len(node.output) != 1:
            kept_nodes.append(node)
            continue

        output_name = node.output[0]
        input_name = node.input[0]
        if matmul_rhs.get(output_name, 0) <= 0 or input_name not in initializer_by_name:
            kept_nodes.append(node)
            continue

        source_value = numpy_helper.to_array(initializer_by_name[input_name])
        perm = _find_attr_ints(node, "perm")
        if perm is None or len(perm) == 0:
            perm = list(reversed(range(source_value.ndim)))
        rewritten = np.ascontiguousarray(np.transpose(source_value, axes=perm))
        new_initializers.append(numpy_helper.from_array(rewritten, name=output_name))
        folded += 1

    if folded <= 0:
        return 0, 0

    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    model.graph.initializer.extend(new_initializers)
    pruned = _prune_unused_initializers(model)

    _remove_stale_artifacts(output_path)
    _save_model(model, output_path)
    return folded, pruned


def _analyze_matmul_rhs_sources(model_path: Path) -> Counter[str]:
    model = onnx.load_model(model_path, load_external_data=False)
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    producer_by_output = _producer_by_output(model)

    counts: Counter[str] = Counter()
    for node in model.graph.node:
        if node.op_type != "MatMul" or len(node.input) < 2:
            continue
        rhs = node.input[1]
        if rhs in initializer_names:
            counts["initializer"] += 1
        elif rhs in producer_by_output:
            counts[producer_by_output[rhs].op_type] += 1
        else:
            counts["other"] += 1
    return counts


def _print_graph_summary(prefix: str, counter: Counter[str]) -> None:
    print(
        f"{prefix} MatMul={counter.get('MatMul', 0)} | "
        f"MatMulInteger={counter.get('MatMulInteger', 0)} | "
        f"QGemm={counter.get('QGemm', 0)} | "
        f"DynamicQuantizeLinear={counter.get('DynamicQuantizeLinear', 0)}"
    )


def _load_ort_quantization():
    from onnxruntime.quantization import QuantType, quantize_dynamic

    return QuantType, quantize_dynamic


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize the SAM2 image encoder for CPU fallback.")
    parser.add_argument("--model_size", default="base_plus", choices=["base_plus", "large", "small", "tiny"])
    parser.add_argument("--input", default="", help="Optional explicit input encoder path.")
    parser.add_argument("--output", default="", help="Optional explicit output encoder path.")
    parser.add_argument("--image", default="", help="Optional image path for output comparison/benchmarking.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for synthetic verification input.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing quantized encoder.")
    parser.add_argument("--per_channel", action="store_true", help="Enable per-channel weight quantization.")
    parser.add_argument("--reduce_range", action="store_true", help="Enable reduced-range dynamic quantization.")
    parser.add_argument("--verify", action="store_true", help="Compare FP32 and INT8 outputs on CPU.")
    parser.add_argument("--benchmark_runs", type=int, default=0, help="Average CPU encoder latency over N runs.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs before benchmarking.")
    parser.add_argument("--weight_type", default="qint8", choices=["qint8", "quint8"])
    parser.add_argument(
        "--skip_canonicalize",
        action="store_true",
        help="Skip the weight-transpose canonicalization pass before quantization.",
    )
    args = parser.parse_args()

    set_cv2_threads(1)

    repo_root = Path(__file__).resolve().parent.parent
    ckpt_dir = repo_root / "checkpoints" / args.model_size
    input_path = Path(args.input) if args.input else ckpt_dir / "image_encoder.onnx"
    output_path = Path(args.output) if args.output else ckpt_dir / "image_encoder.int8.onnx"

    if not input_path.exists():
        sys.exit(f"ERROR: Input encoder not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    canonicalize = not args.skip_canonicalize
    foldable_weight_transposes = _count_foldable_weight_transposes_in_matmul_rhs(input_path) if canonicalize else 0

    output_existed = output_path.exists()
    should_rebuild = args.force or not output_existed
    if output_existed and not args.force:
        existing_ops = _load_graph_counter(output_path)
        if _quantized_matmul_count(existing_ops) == 0 and foldable_weight_transposes > 0:
            print(
                "[INFO] Existing quantized encoder does not contain quantized MatMul ops; "
                "regenerating with canonicalized encoder weights."
            )
            should_rebuild = True

    prepared_input_path = input_path
    prepared_model_path = output_path.with_name(output_path.stem + ".quantprep.onnx")
    folded_weight_transposes = 0
    pruned_initializers = 0

    if not should_rebuild:
        print(f"[INFO] Reusing existing quantized encoder: {output_path}")
    else:
        if output_existed and args.force:
            print(f"[INFO] Overwriting existing quantized encoder: {output_path}")
        _remove_stale_artifacts(output_path)

        if canonicalize and foldable_weight_transposes > 0:
            print("[INFO] Canonicalizing encoder weights for quantization...")
            folded_weight_transposes, pruned_initializers = _canonicalize_model_for_quantization(
                input_path,
                prepared_model_path,
            )
            if folded_weight_transposes > 0:
                prepared_input_path = prepared_model_path
                print(
                    f"[INFO] Canonicalized     : folded {folded_weight_transposes} "
                    f"weight transpose(s), pruned {pruned_initializers} dead initializer(s)"
                )

        print(f"[INFO] Quantizing {prepared_input_path}")
        print(f"[INFO] Output           : {output_path}")
        print(f"[INFO] Per-channel      : {args.per_channel}")
        print(f"[INFO] Reduce-range     : {args.reduce_range}")
        print(f"[INFO] Weight type      : {args.weight_type}")

        QuantType, quantize_dynamic = _load_ort_quantization()
        weight_type = QuantType.QInt8 if args.weight_type == "qint8" else QuantType.QUInt8
        start = time.perf_counter()
        quantize_dynamic(
            model_input=str(prepared_input_path),
            model_output=str(output_path),
            op_types_to_quantize=["MatMul", "Gemm"],
            per_channel=args.per_channel,
            reduce_range=args.reduce_range,
            weight_type=weight_type,
            use_external_data_format=True,
        )
        elapsed = time.perf_counter() - start
        print(f"[INFO] Quantization time: {elapsed:.1f} s")

    if prepared_input_path != input_path:
        _remove_stale_artifacts(prepared_model_path)

    output_data = output_path.with_name(output_path.name + ".data")
    print(f"[INFO] Size FP32={_artifact_total_size_mb(input_path):.1f} MB INT8={_artifact_total_size_mb(output_path):.1f} MB")
    print(f"[INFO] External data    : {'yes' if output_data.exists() else 'no'}")
    if canonicalize:
        print(f"[INFO] Foldable weight transposes in source: {foldable_weight_transposes}")
    if folded_weight_transposes > 0:
        print(f"[INFO] Folded weight transposes        : {folded_weight_transposes}")
        print(f"[INFO] Pruned dead initializers       : {pruned_initializers}")

    _update_manifest(ckpt_dir, output_path)

    fp32_ops = _load_graph_counter(input_path)
    int8_ops = _load_graph_counter(output_path)
    _print_graph_summary("[INFO] Graph ops FP32 :", fp32_ops)
    _print_graph_summary("[INFO] Graph ops INT8 :", int8_ops)
    if _quantized_matmul_count(int8_ops) == 0:
        rhs_sources = _analyze_matmul_rhs_sources(input_path)
        print(
            "[WARN] Dynamic quantization did not rewrite the encoder MatMul graph. "
            "Most MatMul RHS inputs come from graph ops instead of direct initializers."
        )
        print(
            f"[WARN] MatMul RHS      : initializer={rhs_sources.get('initializer', 0)} "
            f"Transpose={rhs_sources.get('Transpose', 0)} "
            f"Mul={rhs_sources.get('Mul', 0)} other={rhs_sources.get('other', 0)}"
        )

    if not args.verify and args.benchmark_runs <= 0:
        return

    providers = ["CPUExecutionProvider"]
    fp32_sess = ort.InferenceSession(str(input_path), sess_options=_session_options(), providers=providers)
    int8_sess = ort.InferenceSession(str(output_path), sess_options=_session_options(), providers=providers)
    input_name = fp32_sess.get_inputs()[0].name
    sample = _make_sample_input(fp32_sess.get_inputs()[0].shape, args.image or None, args.seed)

    if args.verify:
        print("[INFO] Verifying FP32 vs INT8 outputs on CPU...")
        fp32_outputs = fp32_sess.run(None, {input_name: sample})
        int8_outputs = int8_sess.run(None, {input_name: sample})
        mean_abs, max_abs = _compare_outputs(fp32_outputs, int8_outputs)
        print(f"[INFO] Output drift    : mean_abs={mean_abs:.6f} max_abs={max_abs:.6f}")

    if args.benchmark_runs > 0:
        print(f"[INFO] Benchmarking CPU encoder ({args.benchmark_runs} run(s))...")
        fp32_ms = _benchmark_encoder(fp32_sess, input_name, sample, args.warmup, args.benchmark_runs)
        int8_ms = _benchmark_encoder(int8_sess, input_name, sample, args.warmup, args.benchmark_runs)
        speedup = (fp32_ms / int8_ms) if int8_ms > 0 else 0.0
        print(f"[INFO] FP32 encoder    : {fp32_ms:.1f} ms")
        print(f"[INFO] INT8 encoder    : {int8_ms:.1f} ms")
        print(f"[INFO] Speedup         : {speedup:.2f}x")


if __name__ == "__main__":
    main()
