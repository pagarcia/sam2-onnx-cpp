# sam2-onnx-cpp
Segment Anything Model 2 C++ ONNX Wrapper


Instructions after cloning repository:
1. Go to sam2/checkpoints and download model weights (using `bash download_ckpts.sh`)
2. Create a Python virtual environment:
`python -m venv sam2_env`
On Windows run:
`sam2_env\Scripts\activate`
3. Run `python export/onnx_export.py`