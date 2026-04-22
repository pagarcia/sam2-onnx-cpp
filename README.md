# sam2-onnx-cpp

SAM2 ONNX inference for Python and C++, centered on a single production preset:

- Model preset: `base_plus`
- Runtime policy:
  - Use GPU when CUDA ONNX Runtime is available
  - Fall back to CPU otherwise
- CPU fallback policy:
  - Prefer INT8 companion artifacts when they exist
  - Use lean thread and video-memory defaults automatically

The repository still keeps explicit `legacy` and `specialized` paths for benchmarking and debugging, but the intended deployment flow is now:

1. Export `base_plus`
2. Generate CPU INT8 companion artifacts
3. Run the same app on every machine
4. Let the runtime choose GPU or CPU automatically


## Current Runtime Behavior

For image inference:

- GPU path: FP32 encoder and decoder
- CPU path: INT8 encoder when `image_encoder.int8.onnx` exists

For video inference in `auto` mode:

- GPU path: `hybrid-propagate`
  - init decoder: `image_decoder.onnx`
  - propagate decoder: `video_decoder_propagate.onnx`
  - memory attention: `memory_attention.onnx`
  - memory encoder: `memory_encoder.onnx`
- CPU path: `legacy`
  - decoder: `image_decoder.onnx`
  - memory attention: `memory_attention.onnx`
  - memory encoder: `memory_encoder.onnx`
  - if `.int8.onnx` companions exist, they are preferred automatically on CPU

Why this split? On the tested hardware, the newer hybrid video path helps on GPU, but the main CPU win comes from INT8 and lean memory settings rather than from the specialized propagate decoder itself.


## Repository Layout

```text
sam2-onnx-cpp/
  checkpoints/                 Exported ONNX artifacts live here
  cpp/                         C++ runtime and demo app
  export/                      ONNX export pipeline
  python/                      Python demos, benchmarks, quantization tools
  sam2/                        Sparse checkout of the SAM2 code used for export/native comparison
  fetch_sparse.bat
  fetch_sparse.sh
```


## Exported Artifacts

`export/onnx_export.py --model_size base_plus` writes the canonical SAM2 ONNX files plus the specialized video/image variants used for comparison and optimized GPU video propagation.

Common outputs in `checkpoints/base_plus/`:

- `image_encoder.onnx`
- `image_decoder.onnx`
- `memory_attention.onnx`
- `memory_encoder.onnx`
- `image_decoder_box.onnx`
- `video_decoder_init.onnx`
- `video_decoder_propagate.onnx`
- `memory_attention_objptr.onnx`
- `memory_attention_no_objptr.onnx`
- `memory_attention_no_objptr_1frame.onnx`
- `memory_encoder_lite.onnx`
- `manifest.json`

CPU quantization scripts add companion artifacts such as:

- `image_encoder.int8.onnx`
- `video_decoder_propagate.int8.onnx`
- `memory_attention.int8.onnx`
- `memory_encoder.int8.onnx`

The runtime continues to accept the canonical FP32 paths. On CPU it will automatically resolve to the INT8 companion files when present.


## Windows Quick Start

### 1. Fetch the sparse SAM2 checkout and checkpoints

From the repository root:

```powershell
.\fetch_sparse.bat
```

### 2. Create and activate the virtual environment

```powershell
python -m venv sam2_env
.\sam2_env\Scripts\Activate
```

### 3. Install Python dependencies

CPU-only:

```powershell
pip install torch onnx onnxruntime onnxscript hydra-core iopath pillow opencv-python pyqt5
```

NVIDIA GPU:

```powershell
pip install torch onnx onnxruntime-gpu onnxscript hydra-core iopath pillow opencv-python pyqt5
```

Notes:

- GPU execution through ONNX Runtime still requires matching NVIDIA runtime libraries.
- If those libraries are missing, the app can still run on CPU.
- Native SAM2 CUDA benchmarking requires a CUDA-enabled PyTorch build. ONNX GPU inference does not automatically imply that native PyTorch CUDA is available.

### 4. Export `base_plus`

```powershell
.\sam2_env\Scripts\python.exe .\export\onnx_export.py --model_size base_plus
```

### 5. Generate CPU INT8 companion artifacts

These are recommended for deployment even if some machines will use GPU, because CPU fallback will pick them up automatically.

```powershell
.\sam2_env\Scripts\python.exe .\python\quantize_image_encoder.py --model_size base_plus
.\sam2_env\Scripts\python.exe .\python\quantize_video_modules.py --model_size base_plus
```

### 6. Run the Python demos

Image:

```powershell
.\sam2_env\Scripts\python.exe .\python\onnx_test_image.py --model_size base_plus --prompt seed_points
.\sam2_env\Scripts\python.exe .\python\onnx_test_image.py --model_size base_plus --prompt bounding_box
```

Video:

```powershell
.\sam2_env\Scripts\python.exe .\python\onnx_test_video.py --model_size base_plus --prompt seed_points
.\sam2_env\Scripts\python.exe .\python\onnx_test_video.py --model_size base_plus --prompt bounding_box
```

Useful override for testing CPU fallback on a machine that has a GPU:

```powershell
$env:SAM2_ORT_RUNTIME_PROFILE = "cpu_lowcost"
```

Unset it to return to normal auto behavior:

```powershell
Remove-Item Env:SAM2_ORT_RUNTIME_PROFILE -ErrorAction SilentlyContinue
```


## Windows C++ Build

### 1. Install prerequisites

- Visual Studio 2022 with C++ tools
- CMake
- OpenCV
- ONNX Runtime

For GPU deployment, also install or provide matching NVIDIA CUDA and cuDNN runtime DLLs.

### 2. Configure and build

```powershell
cd .\cpp
cmake -S . -B build_release -G "Visual Studio 17 2022" `
  -DCMAKE_CONFIGURATION_TYPES=Release `
  -DOpenCV_DIR="C:/Program Files/OpenCV/Release" `
  -DONNXRUNTIME_DIR="C:/Program Files/onnxruntime-win-x64-gpu-1.22.1"

cmake --build .\build_release --config Release --target Segment -- /m:1
```

The post-build step already copies `onnxruntime*.dll` from `ONNXRUNTIME_DIR/lib` beside the executable.

### 3. Run the C++ app

Recommended: pass canonical artifact paths explicitly from `checkpoints/base_plus`.

```powershell
$ckpt = "$PWD\checkpoints\base_plus"

.\cpp\build_release\bin\Release\Segment.exe `
  --onnx_test_image `
  --prompt seed_points `
  --encoder "$ckpt\image_encoder.onnx" `
  --decoder "$ckpt\image_decoder.onnx"
```

```powershell
$ckpt = "$PWD\checkpoints\base_plus"

.\cpp\build_release\bin\Release\Segment.exe `
  --onnx_test_video `
  --prompt seed_points `
  --encoder "$ckpt\image_encoder.onnx" `
  --decoder "$ckpt\image_decoder.onnx" `
  --memattn "$ckpt\memory_attention.onnx" `
  --memenc "$ckpt\memory_encoder.onnx"
```

Pass the canonical FP32 paths even for CPU deployment. The runtime will resolve to `.int8.onnx` companions automatically when it runs on CPU.


## Deployment Guidance

### Simplest deployment: CPU-only

Ship:

- `Segment.exe`
- `onnxruntime.dll`
- `onnxruntime_providers_shared.dll`
- OpenCV DLLs required by your build
- `checkpoints/base_plus/` with:
  - `image_encoder.onnx`
  - `image_encoder.int8.onnx`
  - `image_decoder.onnx`
  - `memory_attention.onnx`
  - `memory_attention.int8.onnx`
  - `memory_encoder.onnx`
  - `memory_encoder.int8.onnx`

This is the easiest production target for low-cost PCs.

### Mixed deployment: GPU when available, CPU otherwise

Ship the CPU package above, plus:

- `video_decoder_propagate.onnx`
- optionally `video_decoder_propagate.int8.onnx`
- ONNX Runtime GPU provider DLLs
- matching CUDA runtime DLLs
- matching cuDNN DLLs

Behavior at runtime:

- If the machine has a usable CUDA stack, the app runs the GPU path automatically.
- If the CUDA stack is missing or incomplete, the app falls back to CPU automatically.
- If INT8 companion artifacts are present, CPU fallback uses them automatically.

Important:

- A user having an NVIDIA GPU is not enough by itself.
- For GPU execution, ONNX Runtime must be able to load its CUDA provider and the matching NVIDIA runtime libraries.
- If you do not want to package that stack, deploy the CPU-only configuration instead.


## Benchmarking

Compare legacy ONNX vs current auto/runtime choices:

```powershell
$video = "C:\path\to\video.mp4"

.\sam2_env\Scripts\python.exe .\python\benchmark_onnx_variants.py `
  --model_size base_plus `
  --video $video `
  --prompt seed_points `
  --frames 20 `
  --video_order single `
  --session_warmup 1 `
  --warmup 0
```

Force CPU comparison:

```powershell
$env:SAM2_ORT_RUNTIME_PROFILE = "cpu_lowcost"
.\sam2_env\Scripts\python.exe .\python\benchmark_onnx_variants.py `
  --model_size base_plus `
  --video $video `
  --prompt seed_points `
  --frames 20 `
  --video_order single `
  --session_warmup 1 `
  --warmup 0
```

Compare ONNX to native SAM2:

```powershell
.\sam2_env\Scripts\python.exe .\python\benchmark_onnx_variants.py `
  --model_size base_plus `
  --video $video `
  --prompt seed_points `
  --frames 5 `
  --video_order single `
  --session_warmup 0 `
  --warmup 0 `
  --native_compare `
  --native_device cpu
```

For `--native_device cuda`, the environment must contain a CUDA-enabled PyTorch build.


## macOS Notes

macOS support remains CPU-first. CoreML can be experimented with for the encoder, but it is not the recommended production path for this repository.

Basic flow:

```bash
chmod +x fetch_sparse.sh
./fetch_sparse.sh
python -m venv sam2_env
source sam2_env/bin/activate
pip install torch onnx onnxruntime onnxscript hydra-core iopath pillow opencv-python pyqt5
python export/onnx_export.py --model_size base_plus
python python/quantize_image_encoder.py --model_size base_plus
python python/quantize_video_modules.py --model_size base_plus
```


## Acknowledgements

- https://github.com/facebookresearch/sam2
- https://github.com/ryouchinsa/sam-cpp-macos
- https://github.com/Aimol-l/SAM2Export
- https://github.com/Aimol-l/OrtInference


## License

Apache License 2.0. See [LICENSE](LICENSE).
