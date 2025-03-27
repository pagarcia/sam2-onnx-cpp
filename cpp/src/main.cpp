// main.cpp
#include <iostream>
#include <string>

// Forward declarations
int runOnnxTestImage();                // existing single-image test
int runOnnxTestVideo(int argc, char** argv);  // new multi-frame video test

static void printMainUsage()
{
    std::cout << "\n"
              << "[USAGE]\n"
              << "  Segment <mode> [options]\n\n"
              << "  Modes:\n"
              << "    --onnx_test_image   => single-image interactive test\n"
              << "    --onnx_test_video   => multi-frame video test\n"
              << "\n"
              << "Example usage for image mode:\n"
              << "  Segment --onnx_test_image\n\n"
              << "Example usage for video mode:\n"
              << "  Segment --onnx_test_video --encoder image_encoder.onnx --decoder image_decoder.onnx\n"
              << "           --memattn memory_attention.onnx --memenc memory_encoder.onnx --video myvideo.mkv\n"
              << std::endl;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "[ERROR] No mode specified.\n";
        printMainUsage();
        return 1;
    }

    std::string modeArg = argv[1];
    if (modeArg == "--onnx_test_image")
    {
        // Original single-frame, interactive image segmentation
        return runOnnxTestImage();
    }
    else if (modeArg == "--onnx_test_video")
    {
        // New multi-frame video segmentation pipeline
        return runOnnxTestVideo(argc, argv);
    }
    else
    {
        std::cerr << "[ERROR] Unrecognized mode argument: " << modeArg << "\n";
        printMainUsage();
        return 1;
    }
}
