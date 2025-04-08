#ifndef SAM2_SESSION_H_
#define SAM2_SESSION_H_

#include <string>
#include <vector>
#include <variant>
#include <memory>
#include <onnxruntime_cxx_api.h>

// SAM2_Session encapsulates creating and running ONNX Runtime sessions
class SAM2_Session {
public:
    SAM2_Session();
    ~SAM2_Session();

    // Initialization methods for different sessions/models
    bool initImageEncoder(const std::string &encoderPath, int threadsNumber, const std::string &device);
    bool initImageDecoder(const std::string &decoderPath, int threadsNumber, const std::string &device);
    bool initMemAttention(const std::string &memAttentionPath, int threadsNumber, const std::string &device);
    bool initMemEncoder(const std::string &memEncoderPath, int threadsNumber, const std::string &device);

    // Accessors to retrieve underlying sessions
    Ort::Session* getImgEncoderSession();
    Ort::Session* getImgDecoderSession();
    Ort::Session* getMemAttentionSession();
    Ort::Session* getMemEncoderSession();

    // Run a session given input/output names and tensors.
    // Returns either the vector of output Ort::Value objects or an error message.
    std::variant<std::vector<Ort::Value>, std::string> runSession(
        Ort::Session* session,
        const std::vector<const char*>& inputNames,
        const std::vector<const char*>& outputNames,
        const std::vector<Ort::Value>& inputTensors,
        const std::string &debugName
    );

    // Static helper to set up session options
    static void setupSessionOptions(Ort::SessionOptions &options, int threadsNumber,
                                    GraphOptimizationLevel optLevel, const std::string &device);

private:
    std::unique_ptr<Ort::Session> img_encoder_session_;
    std::unique_ptr<Ort::Session> img_decoder_session_;
    std::unique_ptr<Ort::Session> mem_attention_session_;
    std::unique_ptr<Ort::Session> mem_encoder_session_;
};

#endif // SAM2_SESSION_H_
