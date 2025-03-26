// main.cpp
#include <iostream>
#include <string>

// We only have one function for interactive segmentation:
int runOnnxTestImage();

static void printMainUsage()
{
    std::cout << "\n"
              << "[USAGE] Segment --onnx_test_image\n\n"
              << "  This opens a file dialog to choose an image, then provides\n"
              << "  an interactive window for positive/negative clicks.\n"
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
        return runOnnxTestImage();
    }
    else
    {
        std::cerr << "[ERROR] Unrecognized mode argument: " << modeArg << "\n";
        printMainUsage();
        return 1;
    }
}
