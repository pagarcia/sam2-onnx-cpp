# sam2-onnx-cpp

**Segment Anything Model 2 C++ ONNX Wrapper**

This repository provides a C++ wrapper for the [Segment Anything Model 2 (SAM2)](https://github.com/facebookresearch/sam2) using ONNX.  

It wraps fork branch https://github.com/pagarcia/sam2/tree/feature/onnx-export, featuring particular changes in
`sam2/modeling/position_encoding.py` and `sam2/modeling/sam/transformer.py`.

## Windows Setup & Execution

### 1. Fetch the SAM2 Sparse Checkout and Download Checkpoints

Run the provided batch file in the repository root:
```batch
fetch_sparse.bat
```

### 2. Create a Python Virtual Environment

In the repository root, run:
```bash
python -m venv sam2_env
```
Then activate the virtual environment:
```bash
./sam2_env/Scripts/Activate
```

### 3. Install Dependencies

With the virtual environment activated, install the required packages:

#### 3.1 CPU-only (works on any PC)

```bash
pip install torch onnx onnxruntime onnxscript hydra-core iopath pillow opencv-python pyqt5
```

*Note: The `opencv-python` and `pyqt5` installs are needed for the Python test scripts.*

#### 3.2 NVIDIA GPU – Stable (works on Pascal/Turing/Ampere/Ada)

Pinned to a proven set for ONNX Runtime 1.22 + CUDA 12.5 + cuDNN 9.10 (last family that supports Pascal/Volta 
according to the [docs](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/release-notes.html)).

```bash
pip install torch onnx "onnxruntime-gpu==1.22" onnxscript hydra-core iopath pillow opencv-python pyqt5
pip install `
  "nvidia-cuda-runtime-cu12==12.5.82" `
  "nvidia-cuda-nvrtc-cu12==12.5.82" `
  "nvidia-cublas-cu12==12.5.3.2" `
  "nvidia-cufft-cu12==11.4.1.4" `
  "nvidia-curand-cu12==10.3.10.19" `
  "nvidia-nvjitlink-cu12==12.5.82" `
  "nvidia-cudnn-cu12==9.10.2.21"
```

#### 3.3 NVIDIA GPU – Optional Modern stack (Ampere/Ada only)

If you’re on RTX 30/40-series (SM ≥ 8.x) and want the newest CUDA/cuDNN, you can try:

```bash
pip install "onnxruntime-gpu==1.23.2"
pip install `
  "nvidia-cuda-runtime-cu12~=12.9" `
  "nvidia-cuda-nvrtc-cu12~=12.9" `
  "nvidia-cublas-cu12~=12.9" `
  "nvidia-cufft-cu12~=11.4" `
  "nvidia-curand-cu12~=10.3" `
  "nvidia-nvjitlink-cu12~=12.9" `
  "nvidia-cudnn-cu12~=9.14"
pip install torch onnx onnxscript hydra-core iopath pillow opencv-python pyqt5
```

### 4. Generate ONNX Files

Export the ONNX files by running:
```bash
python export/onnx_export.py --model_size base_plus
```
This produces four `.onnx` files in `checkpoints/base_plus/`:
- `image_encoder.onnx`
- `image_decoder.onnx`
- `memory_attention.onnx`
- `memory_encoder.onnx`

Other than `base_plus`, you can choose `tiny`, `small` and `large`.

### 5. Run Python Tests

Test the exported ONNX models by running:
```bash
python export/onnx_test_image.py --prompt seed_points --model_size base_plus
python export/onnx_test_image.py --prompt bounding_box --model_size base_plus
python export/onnx_test_video.py --prompt seed_points --model_size base_plus
python export/onnx_test_video.py --prompt bounding_box --model_size base_plus
```
*Note:*  
- The image test requires a `.jpg`/`.png` image.  
- The video test requires a short video clip (e.g. `.mkv`, `.mp4`, etc.).  
You can find short video samples at [filesamples.com](https://filesamples.com/formats/mkv) and [sample-videos.com](https://www.sample-videos.com/).

### 6. Compile & Run the C++ Wrapper

#### 6.1 Download onnxruntime  
Download and unzip the file from [onnxruntime-win-x64-gpu-1.22.1.zip](https://github.com/microsoft/onnxruntime/releases/download/v1.22.1/onnxruntime-win-x64-gpu-1.22.1.zip) 
(see [releases page](https://github.com/microsoft/onnxruntime/releases/tag/v1.22.1)).  
Extract to location `C:\Program Files\onnxruntime-win-x64-gpu-1.22.1`.

#### 6.2 OpenCV  
Install OpenCV (or point to an existing installation), e.g. located at `C:\Program Files\OpenCV\Release`.

#### 6.3 cuDNN  
Download and install cuDNN from https://developer.nvidia.com/cudnn-9-10-2-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local.
It should match your version of CUDA.

#### 6.4 CMake Configuration and Build

Inside the `cpp` folder, create a `build_release` folder, go inside and run:
```bash
cmake -G "Visual Studio 17 2022" -DCMAKE_CONFIGURATION_TYPES=Release -DOpenCV_DIR="C:/Program Files/OpenCV/Release" -DONNXRUNTIME_DIR="C:/Program Files/onnxruntime-win-x64-gpu-1.22.1" ..
cmake --build . --config Release
```

Inside `cpp/build_release/bin/Release` pleace the following files:

Your previously built ONNX files renamed as:
- `image_decoder.onnx`
- `image_encoder.onnx`
- `memory_attention.onnx`
- `memory_encoder.onnx`

From `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin`:
- `cublas64_12.dll`
- `cublasLt64_12.dll`
- `cudart64_12.dll`
- `cufft64_11.dll`

From `C:\Program Files\NVIDIA\CUDNN\v9.10\bin\12.9`:
- `cudnn_adv64_9.dll`
- `cudnn_cnn64_9.dll`
- `cudnn_engines_precompiled64_9.dll`
- `cudnn_engines_runtime_compiled64_9.dll`
- `cudnn_graph64_9.dll`
- `cudnn_heuristic64_9.dll`
- `cudnn_ops64_9.dll`
- `cudnn64_9.dll`

From `C:/Program Files/onnxruntime-win-x64-gpu-1.22.1/lib`:
- `onnxruntime.dll`
- `onnxruntime_providers_cuda.dll`
- `onnxruntime_providers_shared.dll`
- `onnxruntime_providers_tensorrt.dll`

From `C:/Program Files/OpenCV/Release/x64/vc17/bin`:
- `opencv_core4110.dll`
- `opencv_highgui4110.dll`
- `opencv_imgcodecs4110.dll`
- `opencv_imgproc4110.dll`
- `opencv_videoio4110.dll`

#### 6.5 Run the Compiled Executable

The compiled executable typically goes to `cpp/build_release/bin/Release`. For example, run:
```bash
./Segment.exe --onnx_test_image --prompt seed_points
./Segment.exe --onnx_test_image --prompt bounding_box
./Segment.exe --onnx_test_video --prompt seed_points
./Segment.exe --onnx_test_video --prompt bounding_box
```

## macOS Setup & Execution

### 1. Fetch the SAM2 Sparse Checkout and Download Checkpoints

Assuming the `fetch_sparse.sh` script is available, run:
```bash
chmod +x fetch_sparse.sh
./fetch_sparse.sh
```

### 2. Create a Python Virtual Environment

In the repository root, run:
```bash
python3 -m venv sam2_env
```
Then activate the virtual environment:
```bash
source sam2_env/bin/activate
```

### 3. Install Dependencies (CPU-only for Mac)

With the virtual environment activated, install the required packages:
```bash
pip install torch onnx onnxruntime onnxscript hydra-core iopath pillow opencv-python pyqt5
```

> **Why CPU-only on macOS?**  
> The CoreML Execution Provider can be very strict for this model:  
> - It has a **16,384 per-axis tensor limit**; our 1024×1024 encoder flattens a 256×256 map → **65,536** tokens, so large chunks can’t be offloaded.  
> - The decoder/memory path sometimes uses **0-length / dynamic dims** (e.g., empty prompts), which CoreML refuses.  
> In practice, only small subgraphs run on CoreML and overall performance can be worse than CPU, so we default to **CPU** for reliability.
>
> *Experimental:* you can try CoreML for the **encoder only** by setting:
> ```bash
> export SAM2_ORT_ACCEL=coreml
> ```
> (Decoder/memory will still run on CPU.) If you see compile failures or slowdowns, unset the variable to return to CPU.

### 4. Generate ONNX Files

Export the ONNX files by running:
```bash
python export/onnx_export.py --model_size base_plus
```
This produces four `.onnx` files in `checkpoints/base_plus/`:
- `image_encoder.onnx`
- `image_decoder.onnx`
- `memory_attention.onnx`
- `memory_encoder.onnx`

Other than `base_plus`, you can choose `tiny`, `small` and `large`.

### 5. Run Python Tests

Test the exported ONNX models by running:
```bash
python export/onnx_test_image.py --prompt seed_points --model_size base_plus
python export/onnx_test_image.py --prompt bounding_box --model_size base_plus
python export/onnx_test_video.py --prompt seed_points --model_size base_plus
python export/onnx_test_video.py --prompt bounding_box --model_size base_plus
```
*Note:*  
- The image test requires a `.jpg`/`.png` image.  
- The video test requires a short video clip (e.g. `.mkv`, `.mp4`, etc.).  
Short video samples are available at [filesamples.com](https://filesamples.com/formats/mkv) and [sample-videos.com](https://www.sample-videos.com/).

### 6. Compile & Run the C++ Wrapper

#### 6.1 Download onnxruntime  
Download and unzip [onnxruntime-win-arm64-1.22.1.zip](https://github.com/microsoft/onnxruntime/releases/download/v1.22.1/onnxruntime-win-arm64-1.22.1.zip) 
(see [releases page](https://github.com/microsoft/onnxruntime/releases/tag/v1.22.1)) into the `/opt/` directory.

#### 6.2 Install OpenCV

Install OpenCV using Homebrew:
```bash
brew install opencv
```

#### 6.3 Prepare the Models

Create a `models` folder inside the `cpp` folder. Copy the four exported `.onnx` files into it and rename them as follows:
- `image_decoder.onnx`
- `image_encoder.onnx`
- `memory_attention.onnx`
- `memory_encoder.onnx`

#### 6.4 Set Up Build Folders

Inside the `cpp` folder, create both a `build_release` folder and a `package` folder.

#### 6.5 CMake Configuration and Build

In the `cpp` folder, run:
```bash
cmake -S . -B build_release -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR="/opt/homebrew/opt/opencv" -DONNXRUNTIME_DIR="/opt/onnxruntime-osx-arm64-1.21.0"
```
Then build the project:
```bash
cmake --build build_release
```

#### 6.6 Create the Distributable App

Generate the app by running:
```bash
cmake --install build_release --prefix $HOME/Documents/sam2-onnx-cpp/cpp/package
```

#### 6.7 Run the App

To run the app, either double-click it or run from the terminal.

For the image test:
```bash
cd package/Segment.app/Contents/MacOS
./Segment --onnx_test_image
cd $HOME/Documents/sam2-onnx-cpp/cpp
```

For the video test:
```bash
cd package/Segment.app/Contents/MacOS
./Segment --onnx_test_video
cd $HOME/Documents/sam2-onnx-cpp/cpp
```

## Project Structure

```
sam2-onnx-cpp/
├── export/
│   ├── onnx_export.py      # Main ONNX export script
│   ├── onnx_test_image.py  # Python test script for an image with seed points or bounding box
│   ├── onnx_test_utils.py  # Python shared utils between test image and test video scripts
│   ├── onnx_test_video.py  # Python test script for a video with seed points or bounding box
│   └── src/                # Contains modules and utilities for ONNX export
├── cpp/
│   ├── CMakeLists.txt 
│   ├── build_release/      # Folder containing compiled binaries
│   ├── models/             # (Only for macOS) Folder containing ONNX files
│   ├── package/            # (Only for macOS) Folder containing executable package
│   └── src/                # src code for C++ wrapper and tests
├── checkpoints/            # Contains SAM2 model weights (fetched via sparse)
├── sam2/                   # Contains the SAM2 code (fetched via sparse)
├── fetch_sparse.bat        # Batch file to fetch sparse checkout and download checkpoints for Windows
├── fetch_sparse.sh         # Batch file to fetch sparse checkout and download checkpoints for macOS
├── LICENSE
└── README.md               # (this file)
```

## Additional Notes

- The `fetch_sparse.bat` or `fetch_sparse.sh` script automates fetching only the required SAM2 directories and downloading model checkpoints.  
- The `onnx_export.py` script uses Hydra configs to build the SAM2 model, then exports four ONNX modules:
  - **Image Encoder** (`image_encoder_*.onnx`)
  - **Image Decoder** (`image_decoder_*.onnx`)
  - **Memory Attention** (`memory_attention_*.onnx`)
  - **Memory Encoder** (`memory_encoder_*.onnx`)

## Acknowledgements

This project has been modeled on the following repositories. Their work is gratefully acknowledged:

- [ryouchinsa/sam-cpp-macos](https://github.com/ryouchinsa/sam-cpp-macos)
- [Aimol-l/SAM2Export](https://github.com/Aimol-l/SAM2Export)
- [Aimol-l/OrtInference](https://github.com/Aimol-l/OrtInference)

## License

This project is licensed under the [Apache License 2.0](LICENSE).