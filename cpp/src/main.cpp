// ───────────────────────────────  main.cpp  ──────────────────────────────
#include <iostream>
#include <string>

// Forward declarations of the two entry points
int runOnnxTestImage (int argc, char** argv);
int runOnnxTestVideo (int argc, char** argv);

/* small helper ----------------------------------------------------------- */
static void printMainUsage()
{
    std::cout <<
    "\nUSAGE\n"
    "  Segment <mode> [options]\n\n"
    "Modes:\n"
    "  --onnx_test_image   • interactive image demo\n"
    "                      • accepts  --prompt seed_points|bounding_box  (default: seed_points)\n"
    "  --onnx_test_video   • interactive + full-video demo\n"
    "                      • accepts  --prompt seed_points|bounding_box  (default: seed_points)\n\n"
    "Examples\n"
    "  Segment --onnx_test_image --image lena.png                     # seed-points (default)\n"
    "  Segment --onnx_test_image --prompt bounding_box --image lena.png\n"
    "  Segment --onnx_test_video --video myclip.mp4 --prompt seed_points         \\\n"
    "          --encoder image_encoder.onnx --decoder image_decoder.onnx        \\\n"
    "          --memattn memory_attention.onnx --memenc memory_encoder.onnx\n"
    "  Segment --onnx_test_video --prompt bounding_box --video myclip.mp4        # bbox mode\n"
    << std::endl;
}

/* entry ------------------------------------------------------------------ */
int main(int argc, char** argv)
{
    /* no mode given → default to image demo (seed-points) */
    if (argc < 2)
    {
        char* fakeArgv[2] = { argv[0], (char*)"--onnx_test_image" };
        return runOnnxTestImage(2, fakeArgv);
    }

    std::string mode = argv[1];

    if      (mode == "--onnx_test_image") return runOnnxTestImage(argc, argv);
    else if (mode == "--onnx_test_video") return runOnnxTestVideo(argc, argv);

    std::cerr << "[ERROR] Unknown mode: " << mode << '\n';
    printMainUsage();
    return 1;
}
