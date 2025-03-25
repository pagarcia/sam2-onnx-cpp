# sam2-onnx-cpp
Segment Anything Model 2 C++ ONNX Wrapper

For Windows for the moment.

Instructions after cloning repository:
1. Run `fetch_sparse.bat` file
2. Create a Python virtual environment:
`python -m venv sam2_env`
On Windows run:
`sam2_env\Scripts\activate`
3. Install dependencies:
`pip install torch onnx onnxruntime hydra-core pillow tqdm iopath`
4. Run `python export/onnx_export.py` to get the ONNX files inside checkpoints