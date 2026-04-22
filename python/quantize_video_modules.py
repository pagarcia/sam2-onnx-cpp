#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import onnx

from quantize_image_encoder import (
    _artifact_total_size_mb,
    _canonicalize_model_for_quantization,
    _count_foldable_weight_transposes_in_matmul_rhs,
    _load_graph_counter,
    _print_graph_summary,
    _quantized_matmul_count,
    _remove_stale_artifacts,
)

_SUPPORTED_MODULES = (
    "video_decoder_propagate",
    "memory_attention",
    "memory_encoder",
)

_QUANTIZATION_TOOLS: dict[str, object] | None = None


def _load_ort_quantization_tools() -> dict[str, object]:
    global _QUANTIZATION_TOOLS
    if _QUANTIZATION_TOOLS is None:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        from onnxruntime.quantization.onnx_quantizer import ONNXQuantizer
        import onnxruntime.quantization.onnx_quantizer as ort_onnx_quantizer_module
        from onnxruntime.quantization.quant_utils import (
            QuantizationMode,
            add_infer_metadata,
            model_has_pre_process_metadata,
        )
        from onnxruntime.quantization.registry import IntegerOpsRegistry

        _QUANTIZATION_TOOLS = {
            "QuantType": QuantType,
            "quantize_dynamic": quantize_dynamic,
            "ONNXQuantizer": ONNXQuantizer,
            "ort_onnx_quantizer_module": ort_onnx_quantizer_module,
            "QuantizationMode": QuantizationMode,
            "add_infer_metadata": add_infer_metadata,
            "model_has_pre_process_metadata": model_has_pre_process_metadata,
            "IntegerOpsRegistry": IntegerOpsRegistry,
        }
    return _QUANTIZATION_TOOLS


def _quantize_dynamic_without_shape_infer(
    model_input: Path,
    model_output: Path,
    weight_type_name: str,
    per_channel: bool,
    reduce_range: bool,
) -> None:
    tools = _load_ort_quantization_tools()
    QuantType = tools["QuantType"]
    ONNXQuantizer = tools["ONNXQuantizer"]
    ort_onnx_quantizer_module = tools["ort_onnx_quantizer_module"]
    QuantizationMode = tools["QuantizationMode"]
    add_infer_metadata = tools["add_infer_metadata"]
    model_has_pre_process_metadata = tools["model_has_pre_process_metadata"]
    IntegerOpsRegistry = tools["IntegerOpsRegistry"]
    weight_type = QuantType.QInt8 if weight_type_name == "qint8" else QuantType.QUInt8

    model = onnx.load_model(model_input)
    if not model_has_pre_process_metadata(model):
        logging.warning(
            "Please consider to run pre-processing before quantization. Refer to example: "
            "https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/"
            "image_classification/cpu/ReadMe.md "
        )

    add_infer_metadata(model)
    extra_options = {
        "MatMulConstBOnly": True,
        "DefaultTensorType": onnx.TensorProto.FLOAT,
    }

    original_shape_infer_reload = ort_onnx_quantizer_module.save_and_reload_model_with_shape_infer

    def _skip_shape_infer_reload(current_model: onnx.ModelProto) -> onnx.ModelProto:
        add_infer_metadata(current_model)
        return current_model

    ort_onnx_quantizer_module.save_and_reload_model_with_shape_infer = _skip_shape_infer_reload
    try:
        quantizer = ONNXQuantizer(
            model,
            per_channel,
            reduce_range,
            QuantizationMode.IntegerOps,
            False,
            weight_type,
            QuantType.QUInt8,
            None,
            [],
            [],
            list(IntegerOpsRegistry.keys()),
            extra_options,
        )
        quantizer.quantize_model()
        quantizer.model.save_model_to_file(str(model_output), use_external_data_format=True)
    finally:
        ort_onnx_quantizer_module.save_and_reload_model_with_shape_infer = original_shape_infer_reload


def _update_manifest(ckpt_dir: Path, module_name: str, output_path: Path) -> None:
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
    artifacts[f"{module_name}_int8"] = {
        "path": output_path.name,
        "exists": output_path.exists(),
        "has_external_data": output_data.exists(),
    }

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _quantize_module(
    ckpt_dir: Path,
    module_name: str,
    weight_type_name: str,
    force: bool,
    per_channel: bool,
    reduce_range: bool,
    skip_canonicalize: bool,
) -> None:
    input_path = ckpt_dir / f"{module_name}.onnx"
    output_path = ckpt_dir / f"{module_name}.int8.onnx"

    if not input_path.exists():
        sys.exit(f"ERROR: Input module not found: {input_path}")

    canonicalize = not skip_canonicalize
    foldable_weight_transposes = (
        _count_foldable_weight_transposes_in_matmul_rhs(input_path)
        if canonicalize
        else 0
    )

    output_existed = output_path.exists()
    should_rebuild = force or not output_existed
    if output_existed and not force:
        existing_ops = _load_graph_counter(output_path)
        if _quantized_matmul_count(existing_ops) == 0 and foldable_weight_transposes > 0:
            print(
                f"[INFO] {module_name}: existing INT8 artifact does not contain quantized "
                "MatMul ops; regenerating."
            )
            should_rebuild = True

    prepared_input_path = input_path
    prepared_model_path = output_path.with_name(output_path.stem + ".quantprep.onnx")
    folded_weight_transposes = 0
    pruned_initializers = 0

    if not should_rebuild:
        print(f"[INFO] {module_name}: reusing existing quantized module: {output_path}")
    else:
        if output_existed and force:
            print(f"[INFO] {module_name}: overwriting existing quantized module: {output_path}")
        _remove_stale_artifacts(output_path)

        if canonicalize and foldable_weight_transposes > 0:
            print(f"[INFO] {module_name}: canonicalizing weight transposes...")
            folded_weight_transposes, pruned_initializers = _canonicalize_model_for_quantization(
                input_path,
                prepared_model_path,
            )
            if folded_weight_transposes > 0:
                prepared_input_path = prepared_model_path
                print(
                    f"[INFO] {module_name}: folded {folded_weight_transposes} weight transpose(s), "
                    f"pruned {pruned_initializers} dead initializer(s)"
                )

        print(f"[INFO] {module_name}: quantizing {prepared_input_path}")
        tools = _load_ort_quantization_tools()
        QuantType = tools["QuantType"]
        quantize_dynamic = tools["quantize_dynamic"]
        weight_type = QuantType.QInt8 if weight_type_name == "qint8" else QuantType.QUInt8
        start = time.perf_counter()
        try:
            quantize_dynamic(
                model_input=str(prepared_input_path),
                model_output=str(output_path),
                op_types_to_quantize=["MatMul", "Gemm"],
                per_channel=per_channel,
                reduce_range=reduce_range,
                weight_type=weight_type,
                use_external_data_format=True,
            )
        except Exception as exc:
            print(
                f"[WARN] {module_name}: stock dynamic quantization failed ({exc}). "
                "Retrying without the shape-inference reload step."
            )
            _remove_stale_artifacts(output_path)
            _quantize_dynamic_without_shape_infer(
                model_input=prepared_input_path,
                model_output=output_path,
                weight_type_name=weight_type_name,
                per_channel=per_channel,
                reduce_range=reduce_range,
            )
        elapsed = time.perf_counter() - start
        print(f"[INFO] {module_name}: quantization time {elapsed:.1f} s")

    if prepared_input_path != input_path:
        _remove_stale_artifacts(prepared_model_path)

    output_data = output_path.with_name(output_path.name + ".data")
    print(
        f"[INFO] {module_name}: size FP32={_artifact_total_size_mb(input_path):.1f} MB "
        f"INT8={_artifact_total_size_mb(output_path):.1f} MB"
    )
    print(f"[INFO] {module_name}: external data {'yes' if output_data.exists() else 'no'}")
    if canonicalize:
        print(f"[INFO] {module_name}: foldable weight transposes in source = {foldable_weight_transposes}")
    if folded_weight_transposes > 0:
        print(f"[INFO] {module_name}: folded weight transposes        = {folded_weight_transposes}")
        print(f"[INFO] {module_name}: pruned dead initializers       = {pruned_initializers}")

    fp32_ops = _load_graph_counter(input_path)
    int8_ops = _load_graph_counter(output_path)
    _print_graph_summary(f"[INFO] {module_name}: graph FP32 :", fp32_ops)
    _print_graph_summary(f"[INFO] {module_name}: graph INT8 :", int8_ops)

    _update_manifest(ckpt_dir, module_name, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize SAM2 video runtime modules for CPU fallback.")
    parser.add_argument("--model_size", default="base_plus", choices=["base_plus", "large", "small", "tiny"])
    parser.add_argument(
        "--modules",
        nargs="+",
        default=list(_SUPPORTED_MODULES),
        choices=_SUPPORTED_MODULES,
        help="One or more modules to quantize.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing quantized modules.")
    parser.add_argument("--per_channel", action="store_true", help="Enable per-channel weight quantization.")
    parser.add_argument("--reduce_range", action="store_true", help="Enable reduced-range dynamic quantization.")
    parser.add_argument(
        "--weight_type",
        default="qint8",
        choices=["qint8", "quint8"],
        help="Weight format used by dynamic quantization.",
    )
    parser.add_argument(
        "--skip_canonicalize",
        action="store_true",
        help="Skip the weight-transpose canonicalization pass before quantization.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    ckpt_dir = repo_root / "checkpoints" / args.model_size

    for module_name in args.modules:
        _quantize_module(
            ckpt_dir=ckpt_dir,
            module_name=module_name,
            weight_type_name=args.weight_type,
            force=args.force,
            per_channel=args.per_channel,
            reduce_range=args.reduce_range,
            skip_canonicalize=args.skip_canonicalize,
        )
        print("")


if __name__ == "__main__":
    main()
