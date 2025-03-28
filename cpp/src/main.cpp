// main.cpp
#include <iostream>
#include <string>

// Forward declarations
int runOnnxTestImage(int argc, char** argv);     // updated
int runOnnxTestVideo(int argc, char** argv);

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
              << "  Segment --onnx_test_image --encoder image_encoder.onnx --decoder image_decoder.onnx\n"
              << "           --image myimage.jpg\n\n"
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
        // Updated call with full CLI
        return runOnnxTestImage(argc, argv);
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
