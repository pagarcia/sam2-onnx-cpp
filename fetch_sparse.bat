@echo off
REM ================================
REM Sparse Checkout, Copy, and Checkpoint Download Script
REM This script:
REM 1. Deletes existing "sam2" and "checkpoints" folders (if present)
REM 2. Clones only the "sam2" and "checkpoints" folders from the 'feature/onnx-export' branch of the sam2 fork
REM 3. Copies them into the current repository
REM 4. Cleans up the temporary folder
REM 5. Converts line endings of download_ckpts.sh to Unix-style
REM 6. Changes directory to checkpoints and runs "bash download_ckpts.sh" to download weights
REM ================================

REM Step 0: Delete existing "sam2" and "checkpoints" folders if they exist.
if exist sam2 (
    echo Deleting existing sam2 folder...
    rmdir /s /q sam2
)
if exist checkpoints (
    echo Deleting existing checkpoints folder...
    rmdir /s /q checkpoints
)

REM Step 1: Delete existing temporary folder if it exists.
if exist sam2-sparse (
    echo Deleting existing sam2-sparse folder...
    rmdir /s /q sam2-sparse
)

REM Step 2: Clone the repository in sparse checkout mode (no checkout).
echo Cloning repository (sparse checkout mode)...
git clone --no-checkout -b feature/onnx-export https://github.com/pagarcia/sam2.git sam2-sparse
if errorlevel 1 (
    echo Error cloning repository.
    pause
    exit /b 1
)

REM Step 3: Enter the temporary directory.
cd sam2-sparse

REM Step 4: Initialize sparse-checkout and set the desired folders.
echo Initializing sparse checkout...
git sparse-checkout init --cone
git sparse-checkout set sam2 checkpoints

REM Step 5: Check out the branch.
echo Checking out branch feature/onnx-export...
git checkout feature/onnx-export
if errorlevel 1 (
    echo Error checking out branch.
    pause
    exit /b 1
)

REM Step 6: Return to the base directory.
cd ..

REM Step 7: Copy the desired folders into the current repository.
echo Copying sam2 folder...
xcopy /E /I /Y sam2-sparse\sam2 sam2
echo Copying checkpoints folder...
xcopy /E /I /Y sam2-sparse\checkpoints checkpoints

REM Step 8: Clean up the temporary folder.
echo Cleaning up temporary folder...
rmdir /s /q sam2-sparse

REM Step 9: Convert line endings of download_ckpts.sh to Unix-style.
echo Converting line endings in download_ckpts.sh...
bash -c "sed -i 's/\r$//' checkpoints/download_ckpts.sh"

REM Step 10: Change directory to checkpoints and run the download script.
echo Changing directory to checkpoints to download weights...
cd checkpoints
bash download_ckpts.sh
echo Checkpoints downloaded successfully.
pause
