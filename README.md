# sam2-onnx-cpp

**Segment Anything Model 2 C++ ONNX Wrapper**

This repository provides a C++ wrapper for the [Segment Anything Model 2 (SAM2)](https://github.com/facebookresearch/sam2) using ONNX.  

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
sam2_env\Scripts\activate
```

### 3. Install Dependencies

With the virtual environment activated, install the required packages:
```bash
pip install torch onnx onnxruntime hydra-core pillow tqdm iopath opencv-python pyqt5
```
*Note: The `opencv-python` and `pyqt5` installs are needed for the Python test scripts.*

### 4. Generate ONNX Files

Export the ONNX files by running:
```bash
python export/onnx_export.py --model_size tiny
```
This produces four `.onnx` files in `checkpoints/tiny/`:
- `image_encoder_tiny.onnx`
- `image_decoder_tiny.onnx`
- `memory_attention_tiny.onnx`
- `memory_encoder_tiny.onnx`

### 5. (Optional) Simplify the ONNX Models

Some shape inference warnings or segmentation faults (especially on macOS) may occur due to dynamic shape paths. To mitigate this, install [onnx-simplifier](https://github.com/daquexian/onnx-simplifier):
```bash
pip install onnxsim
```
Then, for each exported file (using the **tiny** variant as an example), run:
```bash
python -m onnxsim checkpoints/tiny/image_encoder_tiny.onnx checkpoints/tiny/image_encoder_tiny_simplified.onnx
python -m onnxsim checkpoints/tiny/image_decoder_tiny.onnx checkpoints/tiny/image_decoder_tiny_simplified.onnx
python -m onnxsim checkpoints/tiny/memory_encoder_tiny.onnx checkpoints/tiny/memory_encoder_tiny_simplified.onnx
python -m onnxsim checkpoints/tiny/memory_attention_tiny.onnx checkpoints/tiny/memory_attention_tiny_simplified.onnx
```

### 6. Run Python Tests

Test the exported ONNX models by running:
```bash
python export/onnx_test_image.py --model_size tiny
python export/onnx_test_video.py --model_size tiny
```
*Note:*  
- The image test requires a `.jpg`/`.png` image.  
- The video test requires a short video clip (e.g. `.mkv`, `.mp4`, etc.).  
You can find short video samples at [filesamples.com](https://filesamples.com/formats/mkv) and [sample-videos.com](https://www.sample-videos.com/).

### 7. Compile & Run the C++ Wrapper

#### 7.1 Download onnxruntime  
Download and unzip the file from [onnxruntime-win-x64-gpu-1.20.0.zip](https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-win-x64-gpu-1.20.0.zip) (see [releases page](https://github.com/microsoft/onnxruntime/releases/tag/v1.20.0)).  
Extract to a location like `C:\Program Files\onnxruntime-win-x64-gpu-1.20.0`.

#### 7.2 OpenCV  
Install OpenCV (or point to an existing installation), e.g. located at `C:\Program Files\OpenCV\Release`.

#### 7.3 CMake Configuration and Build

Inside the `cpp` folder, create a `build_release` folder and run:
```bash
cmake -G "Visual Studio 17 2022" -DCMAKE_CONFIGURATION_TYPES=Release -DOpenCV_DIR="C:/Program Files/OpenCV/Release" -DONNXRUNTIME_DIR="C:/Program Files/onnxruntime-win-x64-gpu-1.20.0" ..
cmake --build . --config Release
```

#### 7.4 Run the Compiled Executable

The compiled executable typically goes to `cpp/build_release/bin/Release`. For example, run:
```bash
./Segment.exe --onnx_test_image
./Segment.exe --onnx_test_video
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

### 3. Install Dependencies

With the virtual environment activated, install the required packages:
```bash
pip install torch onnx onnxruntime hydra-core pillow tqdm iopath opencv-python pyqt5
```

### 4. Generate ONNX Files

Export the ONNX files by running:
```bash
python export/onnx_export.py --model_size tiny
```
This produces four `.onnx` files in `checkpoints/tiny/`:
- `image_encoder_tiny.onnx`
- `image_decoder_tiny.onnx`
- `memory_attention_tiny.onnx`
- `memory_encoder_tiny.onnx`

### 5. (Optional) Simplify the ONNX Models

To prune or unify dynamic shape paths, install onnx-simplifier:
```bash
pip install onnxsim
```
Then run the following commands for each file:
```bash
python -m onnxsim checkpoints/tiny/image_encoder_tiny.onnx checkpoints/tiny/image_encoder_tiny_simplified.onnx
python -m onnxsim checkpoints/tiny/image_decoder_tiny.onnx checkpoints/tiny/image_decoder_tiny_simplified.onnx
python -m onnxsim checkpoints/tiny/memory_encoder_tiny.onnx checkpoints/tiny/memory_encoder_tiny_simplified.onnx
python -m onnxsim checkpoints/tiny/memory_attention_tiny.onnx checkpoints/tiny/memory_attention_tiny_simplified.onnx
```
Repeat for each file as needed.

### 6. Run Python Tests

Test the exported ONNX models by running:
```bash
python export/onnx_test_image.py --model_size tiny
python export/onnx_test_video.py --model_size tiny
```
*Note:*  
- The image test requires a `.jpg`/`.png` image.  
- The video test requires a short video clip (e.g. `.mkv`, `.mp4`, etc.).  
Short video samples are available at [filesamples.com](https://filesamples.com/formats/mkv) and [sample-videos.com](https://www.sample-videos.com/).

### 7. Compile & Run the C++ Wrapper

#### 7.1 Download onnxruntime  
Download and unzip [onnxruntime-win-arm64-1.21.0.zip](https://github.com/microsoft/onnxruntime/releases/download/v1.21.0/onnxruntime-win-arm64-1.21.0.zip) (see [releases page](https://github.com/microsoft/onnxruntime/releases/tag/v1.21.0)) into the `/opt/` directory.

#### 7.2 Install OpenCV

Install OpenCV using Homebrew:
```bash
brew install opencv
```

#### 7.3 Prepare the Models

Create a `models` folder inside the `cpp` folder. Copy the four exported `.onnx` files into it and rename them as follows:
- `image_decoder.onnx`
- `image_encoder.onnx`
- `memory_attention.onnx`
- `memory_encoder.onnx`

#### 7.4 Set Up Build Folders

Inside the `cpp` folder, create both a `build_release` folder and a `package` folder.

#### 7.5 CMake Configuration and Build

In the `cpp` folder, run:
```bash
cmake -S . -B build_release -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR="/opt/homebrew/opt/opencv" -DONNXRUNTIME_DIR="/opt/onnxruntime-osx-arm64-1.21.0"
```
Then build the project:
```bash
cmake --build build_release
```

#### 7.6 Create the Distributable App

Generate the app by running:
```bash
cmake --install build_release --prefix $HOME/Documents/sam2-onnx-cpp/cpp/package
```

#### 7.7 Run the App

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
│   ├── onnx_test_image.py  # Python test script for an image
│   ├── onnx_test_video.py  # Python test script for a video
│   └── src/                # Contains modules and utilities for ONNX export
├── cpp/
│   ├── CMakeLists.txt
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