# sam2-onnx-cpp

**Segment Anything Model 2 C++ ONNX Wrapper**

This repository provides a C++ wrapper for the [Segment Anything Model 2 (SAM2)](https://github.com/facebookresearch/sam2) using ONNX.  
> **Note:** This project is currently configured for Windows.

## Requirements

- Windows
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

   Open a Command Prompt in the repository root and run:
   ```batch
   fetch_sparse.bat
   ```

2. **Create a Python Virtual Environment**

   In the repository root, run:
   ```bash
   python -m venv sam2_env
   ```
   Then, activate the virtual environment (for Windows):
   ```bash
   sam2_env\Scripts\activate
   ```

3. **Install Dependencies**

   With the virtual environment activated, install the required packages:
   ```bash
   pip install torch onnx onnxruntime hydra-core pillow tqdm iopath
   ```

4. **Generate ONNX Files**

   Run the export script to generate the ONNX files. The script accepts a `--model_size` argument with valid values: `base_plus`, `large`, `small`, or `tiny`. For example, to export the "small" model variant, run:
   ```bash
   python export/onnx_export.py --model_size small
   ```
   The ONNX files will be saved in the corresponding output directory (e.g., `checkpoints/small/`) with names like:
   - `image_encoder_small.onnx`
   - `image_decoder_small.onnx`
   - `memory_attention_small.onnx`
   - `memory_encoder_small.onnx`

4. **Run Python tests**

   To run the tests you will need to install additional libraries:
   ```bash
   pip install opencv-python
   ```
   Run the tests with 
   ```bash
   python export/onnx_test_image.py --model_size small
   python export/onnx_test_video.py --model_size small
   ```
   You can find short video samples in https://filesamples.com/formats/mkv and https://www.sample-videos.com/.

## Project Structure

```
sam2-onnx-cpp/
├── export/
│   ├── onnx_export.py      # Main ONNX export script
│   └── src/                # Contains modules and utilities for ONNX export
├── checkpoints/            # Contains SAM2 model weights (fetched via sparse bat script)
├── sam2/                   # Contains the SAM2 code (fetched via sparse checkout)
├── fetch_sparse.bat        # Batch file to fetch sparse checkout and download checkpoints
├── README.md               # This file
└── LICENSE                 # Apache License 2.0
```

## Additional Notes

- The `fetch_sparse.bat` script automates the process of fetching only the required SAM2 directories and downloading model checkpoints.
- The `onnx_export.py` script uses Hydra for configuration to build the SAM2 model and then exports four ONNX files:
  - **Image Encoder**
  - **Image Decoder**
  - **Memory Attention**
  - **Memory Encoder**
- The output file names incorporate the model size (e.g., `_small`, `_base_plus`, etc.) based on the `--model_size` parameter.

## License

This project is licensed under the [Apache License 2.0](LICENSE). 