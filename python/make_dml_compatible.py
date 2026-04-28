#!/usr/bin/env python
"""Create DirectML-compatible SAM2 ONNX wrapper files.

Some SAM2 exports contain Reshape nodes with allowzero=1. ONNX Runtime's
DirectML provider can reject otherwise valid graphs with that attribute. This
script creates sibling *.dml.onnx files with the attribute removed while keeping
the original external *.onnx.data weight files shared.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx


def strip_reshape_allowzero(model: onnx.ModelProto) -> int:
    changed = 0
    for node in model.graph.node:
        if node.op_type != "Reshape":
            continue

        kept = []
        for attr in node.attribute:
            if attr.name == "allowzero" and attr.i == 1:
                changed += 1
                continue
            kept.append(attr)

        if len(kept) != len(node.attribute):
            del node.attribute[:]
            node.attribute.extend(kept)

    return changed


def convert_file(path: Path, force: bool) -> tuple[Path, int] | None:
    if path.name.endswith(".dml.onnx"):
        return None

    out_path = path.with_name(path.stem + ".dml.onnx")
    if out_path.exists() and not force:
        return out_path, 0

    model = onnx.load(str(path), load_external_data=False)
    changed = strip_reshape_allowzero(model)
    if changed == 0:
        return None

    onnx.save_model(model, str(out_path))
    return out_path, changed


def iter_onnx_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.glob("*.onnx")))
        elif path.is_file() and path.suffix == ".onnx":
            files.append(path)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path("checkpoints/small"), Path("checkpoints/base_plus")],
        help="ONNX files or directories to convert.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing *.dml.onnx files.")
    args = parser.parse_args()

    converted = 0
    for path in iter_onnx_files(args.paths):
        result = convert_file(path, args.force)
        if result is None:
            continue
        out_path, changed = result
        if changed == 0:
            print(f"[SKIP] {out_path} already exists")
        else:
            print(f"[OK] {path} -> {out_path} ({changed} Reshape node(s))")
            converted += 1

    print(f"[INFO] Converted {converted} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
