# sam2-onnx-cpp

**Segment Anything Model 2 C++ ONNX Wrapper**

This repository provides a C++ wrapper for the [Segment Anything Model 2 (SAM2)](https://github.com/facebookresearch/sam2) using ONNX.  
> **Note:** This project is currently configured for Windows and has additional instructions for macOS usage.

## Requirements

- Windows or macOS
- Python 3.x
- Git

## Setup Instructions

Follow these steps after cloning the repository:

1. **Fetch the SAM2 Sparse Checkout and Download Checkpoints**

   Run the provided batch file to:
   - Remove any existing `sam2` and `checkpoints` folders.
   - Clone only the required folders (`sam2` and `checkpoints`) from your SAM2 fork (branch `feature/onnx-export`) using sparse checkout.
   - Convert line endings for the checkpoint download script.
   - Download the model weights.

   On Windows in the repository root:
   ```batch
   fetch_sparse.bat
   ```

   On macOS (assuming `fetch_sparse.sh` is available):
   ```bash
   chmod +x fetch_sparse.sh
   ./fetch_sparse.sh
   ```

2. **Create a Python Virtual Environment**

   In the repository root, run:
   ```bash
   python -m venv sam2_env
   ```
   Then, activate the virtual environment.

   - **Windows**:
     ```bash
     sam2_env\Scripts\activate
     ```
   - **macOS**:
     ```bash
     python3 -m venv sam2_env
     source sam2_env/bin/activate
     ```

3. **Install Dependencies**

   With the virtual environment activated, install the required packages:
   ```bash
   pip install torch onnx onnxruntime hydra-core pillow tqdm iopath opencv-python pyqt5
   ```
   (The `opencv-python` and `pyqt5` install is needed for the Python test scripts.)

4. **Generate ONNX Files**

   Run the export script to generate the ONNX files. The script accepts a `--model_size` argument with valid values: `base_plus`, `large`, `small`, or `tiny`. For example, to export the **tiny** model variant:
   ```bash
   python export/onnx_export.py --model_size tiny
   ```

   This produces four `.onnx` files in `checkpoints/tiny/`:
   - `image_encoder_tiny.onnx`
   - `image_decoder_tiny.onnx`
   - `memory_attention_tiny.onnx`
   - `memory_encoder_tiny.onnx`

5. **(Optional) Simplify the ONNX Models**

   In some cases, you may experience shape inference warnings or segmentation faults (especially on macOS) due to dynamic shape paths. You can run [onnx-simplifier](https://github.com/daquexian/onnx-simplifier) to prune or unify these paths:

   ```bash
   pip install onnxsim
   ```

   Then, for each exported file (e.g. `image_encoder_tiny.onnx`), run:

   ```bash
   python -m onnxsim checkpoints/tiny/image_encoder_tiny.onnx checkpoints/tiny/image_encoder_tiny_simplified.onnx
   python -m onnxsim checkpoints/tiny/image_decoder_tiny.onnx checkpoints/tiny/image_decoder_tiny_simplified.onnx
   python -m onnxsim checkpoints/tiny/memory_encoder_tiny.onnx checkpoints/tiny/memory_encoder_tiny_simplified.onnx
   python -m onnxsim checkpoints/tiny/memory_attention_tiny.onnx checkpoints/tiny/memory_attention_tiny_simplified.onnx
   ```

   Repeat for `image_decoder_tiny.onnx`, `memory_encoder_tiny.onnx`, and `memory_attention_tiny.onnx`.  You can then use the simplified files in your downstream pipeline.  This often fixes shape warnings and can prevent segmentation faults on certain platforms.

6. **Run Python Tests**

   You can test the exported ONNX models by running the provided scripts:
   ```bash
   python export/onnx_test_image.py --model_size tiny
   python export/onnx_test_video.py --model_size tiny
   ```
   The image test requires a `.jpg/.png` image; the video test requires a short video clip (e.g. `.mkv`, `.mp4`, etc.).

7. **Compile & Run the C++ Wrapper**

   ### Windows

   1. **Download onnxruntime**  
      For example:  
      https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-win-x64-gpu-1.20.0.zip  
      Extract to a location like `C:\Program Files\onnxruntime-win-x64-gpu-1.20.0`.

   2. **OpenCV**  
      Install OpenCV (or point to an existing installation).  
      Suppose you have `C:\Program Files\OpenCV\Release`.

   3. **CMake**  
      Inside `cpp/build_release`:
      ```bash
      cmake -G "Visual Studio 17 2022" -DCMAKE_CONFIGURATION_TYPES=Release -DOpenCV_DIR="C:/Program Files/OpenCV/Release" -DONNXRUNTIME_DIR="C:/Program Files/onnxruntime-win-x64-gpu-1.20.0" ..
      cmake --build . --config Release
      ```

   4. **Run**  
      The compiled executable typically goes to `cpp/build_release/bin/Release`.  You can run, for example:
      ```bash
      ./Segment.exe --onnx_test_image
      ./Segment.exe --onnx_test_video
      ```

   ### macOS

   1. **Download onnxruntime**  
      e.g. https://github.com/microsoft/onnxruntime/releases/tag/v1.20.0  
      Unzip it into `/opt/onnxruntime-osx-arm64-1.20.0`.

   2. **Install OpenCV**  
      via Homebrew:
      ```bash
      brew install opencv
      ```

   3. **CMake**  
      In the `cpp` folder:
      ```bash
      mkdir build_release
      cmake -S . -B build_release -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR="/opt/homebrew/opt/opencv" -DONNXRUNTIME_DIR="/opt/onnxruntime-osx-arm64-1.20.0"
      cmake --build build_release
      ```

   4. **Package and Run**  
      ```bash
      cmake --install build_release --prefix ./package
      ```
      Then run `./package/Segment.app/Contents/MacOS/Segment --onnx_test_image` or `--onnx_test_video`.

## Project Structure

```
sam2-onnx-cpp/
├── export/
│   ├── onnx_export.py      # Main ONNX export script
│   ├── onnx_test_image.py  # Python test script for an image
│   ├── onnx_test_video.py  # Python test script for a video
│   └── src/                # Contains modules and utilities for ONNX export
├── cpp/
│   ├── CMakeLists.txt
│   └── src/                # src code for C++ wrapper and tests
├── checkpoints/            # Contains SAM2 model weights (fetched via sparse)
├── sam2/                   # Contains the SAM2 code (fetched via sparse)
├── fetch_sparse.bat        # Batch file to fetch sparse checkout and download checkpoints
├── LICENSE
└── README.md               # (this file)
```

## Additional Notes

- The `fetch_sparse.bat` or `fetch_sparse.sh` script automates fetching only the required SAM2 directories and downloading model checkpoints.  
- The `onnx_export.py` uses Hydra configs to build the SAM2 model, then exports four ONNX modules:
  - **Image Encoder** (`image_encoder_*.onnx`)
  - **Image Decoder** (`image_decoder_*.onnx`)
  - **Memory Attention** (`memory_attention_*.onnx`)
  - **Memory Encoder** (`memory_encoder_*.onnx`)
- If you see shape inference warnings or segmentation faults in onnxruntime (especially on macOS), **running onnx-simplifier** can remove leftover dynamic branches and fix many shape mismatch issues.
- The simplified `.onnx` files may grow in size due to constant folding, but often become more stable at runtime.

## License

This project is licensed under the [Apache License 2.0](LICENSE).