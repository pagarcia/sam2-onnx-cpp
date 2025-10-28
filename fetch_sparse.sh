# sam2-onnx-cpp/fetch_sparse.sh
#!/usr/bin/env bash
set -e  # exit immediately on error

# ================================
# Equivalent to fetch_sparse.bat for macOS/Linux
# ================================

# Step 0: Delete existing "sam2" and "checkpoints" folders if they exist.
if [ -d "sam2" ]; then
  echo "Deleting existing sam2 folder..."
  rm -rf sam2
fi

if [ -d "checkpoints" ]; then
  echo "Deleting existing checkpoints folder..."
  rm -rf checkpoints
fi

# Step 1: Delete existing temporary folder if it exists.
if [ -d "sam2-sparse" ]; then
  echo "Deleting existing sam2-sparse folder..."
  rm -rf sam2-sparse
fi

# Step 2: Clone the repository in sparse checkout mode (no checkout).
echo "Cloning repository (sparse checkout mode)..."
git clone --no-checkout -b feature/onnx-export https://github.com/pagarcia/sam2.git sam2-sparse

# Step 3: Enter the temporary directory and configure sparse checkout.
cd sam2-sparse
echo "Initializing sparse checkout..."
git sparse-checkout init --cone
git sparse-checkout set sam2 checkpoints

# Step 4: Check out the branch.
echo "Checking out branch feature/onnx-export..."
git checkout feature/onnx-export
cd ..

# Step 5: Copy the desired folders into the current repository.
echo "Copying sam2 folder..."
cp -r sam2-sparse/sam2 ./sam2

echo "Copying checkpoints folder..."
cp -r sam2-sparse/checkpoints ./checkpoints

# Step 6: Clean up the temporary folder.
echo "Cleaning up temporary folder..."
rm -rf sam2-sparse

# Step 7: Convert line endings of download_ckpts.sh to Unix-style.
echo "Converting line endings in download_ckpts.sh..."
sed -i '' $'s/\r$//' checkpoints/download_ckpts.sh
# (On macOS, 'sed -i '' ' is needed to edit in-place without backup.
#  If you are on Linux, you can do: sed -i 's/\r$//' ...)

# Step 8: Change directory to checkpoints and run the download script.
echo "Downloading checkpoints..."
cd checkpoints
bash download_ckpts.sh

echo "Checkpoints downloaded successfully."
