#ifndef ARTIFACTRESOLVER_H
#define ARTIFACTRESOLVER_H

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace ArtifactResolver {

struct ImageDecoderSelection {
    std::string path;
    std::string mode;
};

struct VideoRuntimeSelection {
    std::string decoderInitPath;
    std::string decoderPropagatePath;
    std::string memoryAttentionPath;
    std::string memoryEncoderPath;
    std::string mode;
};

inline std::string lowerCopy(std::string value)
{
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

inline std::filesystem::path candidatePath(const std::string &path)
{
    std::filesystem::path candidate(path);
    if (candidate.has_parent_path()) {
        return candidate;
    }
    return std::filesystem::current_path() / candidate;
}

inline std::string normalizePath(const std::filesystem::path &path)
{
    try {
        if (std::filesystem::exists(path)) {
            return std::filesystem::absolute(path).lexically_normal().string();
        }
    } catch (...) {
    }
    return path.lexically_normal().string();
}

inline std::string preferredRuntimeProfile();
inline bool isLowCostCpuProfile();

inline std::string preferredRuntimeProfile()
{
    const char *value = std::getenv("SAM2_ORT_RUNTIME_PROFILE");
    if (!value) {
        return "";
    }
    return lowerCopy(value);
}

inline bool isLowCostCpuProfile()
{
    const std::string profile = preferredRuntimeProfile();
    return profile == "cpu_lowcost"
        || profile == "lowcost_cpu"
        || profile == "cpu-lowcost"
        || profile == "low-cost-cpu";
}

inline int preferredRuntimeThreads(int fallback, const std::string &device)
{
    const char *value = std::getenv("SAM2_ORT_CPU_THREADS");
    if (value && *value) {
        try {
            return std::max(1, std::stoi(value));
        } catch (...) {
        }
    }

    if (isLowCostCpuProfile() || device == "cpu") {
        return std::max(1, std::min(fallback, 4));
    }
    return std::max(1, fallback);
}

inline std::string preferredEncoderVariant()
{
    const char *value = std::getenv("SAM2_ORT_ENCODER_VARIANT");
    if (!value) {
        return "auto";
    }
    const std::string lowered = lowerCopy(value);
    if (lowered == "int8" || lowered == "fp32" || lowered == "auto") {
        return lowered;
    }
    return "auto";
}

inline std::string preferredVideoModuleVariant()
{
    const char *value = std::getenv("SAM2_ORT_VIDEO_MODULE_VARIANT");
    if (!value) {
        return "auto";
    }
    const std::string lowered = lowerCopy(value);
    if (lowered == "int8" || lowered == "fp32" || lowered == "auto") {
        return lowered;
    }
    return "auto";
}

inline bool useExperimentalVideoInitDecoder()
{
    const char *value = std::getenv("SAM2_ORT_EXPERIMENTAL_VIDEO_INIT_DECODER");
    if (!value) {
        return false;
    }
    const std::string lowered = lowerCopy(value);
    return lowered == "1" || lowered == "true" || lowered == "yes";
}

inline bool pathExists(const std::filesystem::path &path)
{
    try {
        return std::filesystem::exists(path);
    } catch (...) {
        return false;
    }
}

inline bool isBasename(const std::string &path, const std::string &name)
{
    return lowerCopy(candidatePath(path).filename().string()) == lowerCopy(name);
}

inline std::string preferQuantizedEncoderPath(const std::string &encoderPath,
                                              const std::string &device)
{
    if (!isBasename(encoderPath, "image_encoder.onnx")) {
        return encoderPath;
    }

    const std::filesystem::path current = candidatePath(encoderPath);
    const std::filesystem::path quantized = current.parent_path() / "image_encoder.int8.onnx";
    const std::string variant = preferredEncoderVariant();

    if (variant == "fp32") {
        return encoderPath;
    }

    if (variant == "int8") {
        if (pathExists(quantized)) {
            return normalizePath(quantized);
        }
        return encoderPath;
    }

    if (device != "cpu") {
        return encoderPath;
    }

    if (pathExists(quantized)) {
        return normalizePath(quantized);
    }

    return encoderPath;
}

inline std::string preferQuantizedRuntimeArtifactPath(const std::string &path,
                                                      const std::string &device,
                                                      const std::string &variant)
{
    const std::filesystem::path current = candidatePath(path);
    if (lowerCopy(current.extension().string()) != ".onnx") {
        return path;
    }

    const std::filesystem::path quantized =
        current.parent_path() / (current.stem().string() + ".int8.onnx");

    if (variant == "fp32") {
        return path;
    }

    if (variant == "int8") {
        if (pathExists(quantized)) {
            return normalizePath(quantized);
        }
        return path;
    }

    if (device != "cpu") {
        return path;
    }

    if (pathExists(quantized)) {
        return normalizePath(quantized);
    }

    return path;
}

inline ImageDecoderSelection resolveImageDecoderPath(const std::string &decoderPath,
                                                     const std::string &promptMode,
                                                     bool experimentalImagePointDecoder = false)
{
    if (!isBasename(decoderPath, "image_decoder.onnx")) {
        return {decoderPath, "manual"};
    }

    const std::filesystem::path current = candidatePath(decoderPath);
    const std::filesystem::path directory = current.parent_path();

    if (promptMode == "bounding_box") {
        const std::filesystem::path specialized = directory / "image_decoder_box.onnx";
        if (pathExists(specialized)) {
            return {normalizePath(specialized), "specialized"};
        }
        return {decoderPath, "legacy"};
    }

    if (promptMode == "seed_points") {
        const std::filesystem::path specialized = directory / "image_decoder_points.onnx";
        if (experimentalImagePointDecoder && pathExists(specialized)) {
            return {normalizePath(specialized), "specialized"};
        }
        return {decoderPath, experimentalImagePointDecoder ? "legacy-missing-specialized" : "legacy-safe-seed-points"};
    }

    return {decoderPath, "legacy"};
}

inline VideoRuntimeSelection resolveVideoRuntimePaths(const std::string &decoderPath,
                                                      const std::string &memoryAttentionPath,
                                                      const std::string &memoryEncoderPath,
                                                      bool experimentalOneFrameAttention = false,
                                                      const std::string &device = "cpu")
{
    const std::filesystem::path decoderCandidate = candidatePath(decoderPath);
    const std::filesystem::path decoderDirectory = decoderCandidate.parent_path();
    const std::filesystem::path manualDecoderInit = decoderDirectory / "video_decoder_init.onnx";
    const std::filesystem::path manualDecoderPropagate = decoderDirectory / "video_decoder_propagate.onnx";
    const std::string videoModuleVariant = preferredVideoModuleVariant();

    const auto applyVideoModuleVariants = [&](VideoRuntimeSelection selection) -> VideoRuntimeSelection {
        selection.decoderPropagatePath = preferQuantizedRuntimeArtifactPath(
            selection.decoderPropagatePath,
            device,
            videoModuleVariant);
        selection.memoryAttentionPath = preferQuantizedRuntimeArtifactPath(
            selection.memoryAttentionPath,
            device,
            videoModuleVariant);
        selection.memoryEncoderPath = preferQuantizedRuntimeArtifactPath(
            selection.memoryEncoderPath,
            device,
            videoModuleVariant);
        return selection;
    };

    if ((isBasename(decoderPath, "video_decoder_init.onnx")
         || isBasename(decoderPath, "video_decoder_propagate.onnx"))
        && (isBasename(memoryAttentionPath, "memory_attention_objptr.onnx")
            || isBasename(memoryAttentionPath, "memory_attention_no_objptr.onnx")
            || isBasename(memoryAttentionPath, "memory_attention_no_objptr_1frame.onnx"))
        && (isBasename(memoryEncoderPath, "memory_encoder.onnx")
            || isBasename(memoryEncoderPath, "memory_encoder_lite.onnx"))) {
        const std::string mode = (isBasename(memoryEncoderPath, "memory_encoder.onnx")
                                      ? "manual-specialized-temporal"
                                      : "manual-specialized-lite");
        return applyVideoModuleVariants({
            normalizePath(isBasename(decoderPath, "video_decoder_init.onnx") ? decoderCandidate : manualDecoderInit),
            normalizePath(isBasename(decoderPath, "video_decoder_propagate.onnx") ? decoderCandidate : manualDecoderPropagate),
            normalizePath(candidatePath(memoryAttentionPath)),
            normalizePath(candidatePath(memoryEncoderPath)),
            mode,
        });
    }

    if (!isBasename(decoderPath, "image_decoder.onnx")
        || !isBasename(memoryAttentionPath, "memory_attention.onnx")
        || !isBasename(memoryEncoderPath, "memory_encoder.onnx")) {
        return applyVideoModuleVariants({decoderPath, decoderPath, memoryAttentionPath, memoryEncoderPath, "manual"});
    }

    const std::filesystem::path directory = decoderDirectory;

    const std::filesystem::path decoderInit = directory / "video_decoder_init.onnx";
    const std::filesystem::path decoderProp = directory / "video_decoder_propagate.onnx";
    const std::filesystem::path attnObjPtr = directory / "memory_attention_objptr.onnx";
    const std::filesystem::path attn1Frame = directory / "memory_attention_no_objptr_1frame.onnx";
    const std::filesystem::path attnDynamic = directory / "memory_attention_no_objptr.onnx";
    const std::filesystem::path legacyMemEncoder = directory / "memory_encoder.onnx";
    const std::filesystem::path memEncoderLite = directory / "memory_encoder_lite.onnx";

    const bool specializedAvailable =
        pathExists(decoderInit)
        && pathExists(decoderProp)
        && (pathExists(attnObjPtr) || pathExists(attn1Frame) || pathExists(attnDynamic));
    const bool legacyAvailable = pathExists(candidatePath(decoderPath))
        && pathExists(candidatePath(memoryAttentionPath))
        && pathExists(candidatePath(memoryEncoderPath));
    const bool hybridAvailable =
        legacyAvailable
        && pathExists(decoderProp);
    const bool preferOptimized = device != "cpu";
    const auto hybridResult = [&]() -> VideoRuntimeSelection {
        return applyVideoModuleVariants({
            normalizePath(candidatePath(decoderPath)),
            normalizePath(decoderProp),
            normalizePath(candidatePath(memoryAttentionPath)),
            normalizePath(candidatePath(memoryEncoderPath)),
            "hybrid-propagate",
        });
    };

    const auto specializedResult = [&]() -> VideoRuntimeSelection {
        std::filesystem::path selectedAttention;
        std::string suffix;

        if (pathExists(attnObjPtr)) {
            selectedAttention = attnObjPtr;
            suffix = "objptr";
        } else if (experimentalOneFrameAttention && pathExists(attn1Frame)) {
            selectedAttention = attn1Frame;
            suffix = "1frame-attn";
        } else if (pathExists(attnDynamic)) {
            selectedAttention = attnDynamic;
            suffix = "dynamic-attn";
        } else {
            selectedAttention = attn1Frame;
            suffix = "1frame-attn-fallback";
        }

        const std::filesystem::path selectedMemEncoder = pathExists(legacyMemEncoder) ? legacyMemEncoder : memEncoderLite;
        const std::string modePrefix = pathExists(legacyMemEncoder) ? "specialized-temporal" : "specialized-lite";

        return applyVideoModuleVariants({
            normalizePath(decoderInit),
            normalizePath(decoderProp),
            normalizePath(selectedAttention),
            normalizePath(selectedMemEncoder),
            modePrefix + "-" + suffix,
        });
    };

    const auto optimizedResult = [&]() -> VideoRuntimeSelection {
        if (!useExperimentalVideoInitDecoder() && hybridAvailable) {
            return hybridResult();
        }
        if (specializedAvailable) {
            return specializedResult();
        }
        return hybridResult();
    };

    if (preferOptimized && (hybridAvailable || specializedAvailable)) {
        return optimizedResult();
    }

    if (legacyAvailable) {
        return applyVideoModuleVariants({decoderPath, decoderPath, memoryAttentionPath, memoryEncoderPath, "legacy"});
    }

    if (hybridAvailable || specializedAvailable) {
        return optimizedResult();
    }

    return applyVideoModuleVariants({decoderPath, decoderPath, memoryAttentionPath, memoryEncoderPath, "legacy"});
}

} // namespace ArtifactResolver

#endif // ARTIFACTRESOLVER_H
