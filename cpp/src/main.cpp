// main.cpp
#include <iostream>
#include <string>

// Forward declarations
int runOnnxTestImage(int argc, char** argv);
int runOnnxTestImageBoundingBox(int argc, char** argv);
int runOnnxTestVideo(int argc, char** argv);

static void printMainUsage()
{
    std::cout << "\n"
              << "[USAGE]\n"
              << "  Segment <mode> [options]\n\n"
              << "  Modes:\n"
              << "    --onnx_test_image                => single-image interactive test\n"
              << "    --onnx_test_image_bounding_box   => single-image bounding box interactive test\n"
              << "    --onnx_test_video                => multi-frame video test\n"
              << "\n"
              << "Example usage for image mode:\n"
              << "  Segment --onnx_test_image_bounding_box --encoder image_encoder.onnx --decoder image_decoder.onnx\n"
              << "           --image myimage.jpg\n\n"
              << "Example usage for video mode:\n"
              << "  Segment --onnx_test_video --encoder image_encoder.onnx --decoder image_decoder.onnx\n"
              << "           --memattn memory_attention.onnx --memenc memory_encoder.onnx --video myvideo.mkv\n"
              << std::endl;
}

int main(int argc, char** argv)
{
    // If no mode argument is provided, default to onnx_test_image
    if (argc < 2) {
        // Create a new argv array with the program name and the --onnx_test_image mode
        char* newArgv[2];
        newArgv[0] = argv[0];                       // same program name
        newArgv[1] = (char*)"--onnx_test_image";    // default mode
        // Call runOnnxTestImage with these new arguments
        return runOnnxTestImage(2, newArgv);
    }

    // Otherwise, use the user-provided argument
    std::string modeArg = argv[1];

    if (modeArg == "--onnx_test_image")
    {
        return runOnnxTestImage(argc, argv);
    }
        else if (modeArg == "--onnx_test_image_bounding_box")
    {
        return runOnnxTestImageBoundingBox(argc, argv);
    }
    else if (modeArg == "--onnx_test_video")
    {
        return runOnnxTestVideo(argc, argv);
    }
    else
    {
        std::cerr << "[ERROR] Unrecognized mode argument: " << modeArg << "\n";
        printMainUsage();
        return 1;
    }
}
