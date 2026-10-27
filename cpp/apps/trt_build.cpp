// ONNX -> TensorRT motoru (FP32 / FP16 / INT8).
//
//   trt_build --onnx m.onnx --out m_fp16.engine --precision fp16
//   trt_build --onnx m.onnx --out m_int8.engine --precision int8 --calib-list calib.txt [--calib-cache c.cache]
//
// INT8 kalibrasyonu: entropi kalibratoru, girdi canli hatla ayni CUDA on islemden gecer
// (kalibrasyon ve cikarim dagilimi birebir ayni olsun diye).

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "iha/args.hpp"
#include "iha/preprocess.hpp"

namespace {

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING) std::cerr << "[TRT] " << msg << "\n";
    }
} gLogger;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
class EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    EntropyCalibrator(std::vector<std::string> files, int w, int h, std::string cache)
        : files_(std::move(files)), w_(w), h_(h), cache_(std::move(cache)) {
        cudaMalloc(&d_input_, sizeof(float) * 3 * w_ * h_);
        cudaStreamCreate(&stream_);
    }
    ~EntropyCalibrator() override {
        cudaFree(d_input_);
        cudaFree(d_img_);
        cudaStreamDestroy(stream_);
    }
    int32_t getBatchSize() const noexcept override { return 1; }
    bool getBatch(void* bindings[], char const* /*names*/[], int32_t /*n*/) noexcept override {
        cv::Mat img;
        while (img.empty()) {  // okunamayan dosya kalibrasyonu bitirmesin, atla
            if (idx_ >= files_.size()) return false;
            img = cv::imread(files_[idx_++], cv::IMREAD_COLOR);
            if (img.empty()) std::cerr << "atlandi (okunamadi): " << files_[idx_ - 1] << "\n";
        }
        const std::size_t bytes = img.total() * 3;
        if (bytes > cap_) {
            cudaFree(d_img_);
            cudaMalloc(&d_img_, bytes);
            cap_ = bytes;
        }
        cudaMemcpyAsync(d_img_, img.data, bytes, cudaMemcpyHostToDevice, stream_);
        const auto lb = iha::Letterbox::compute(img.cols, img.rows, w_, h_);
        iha::launchLetterboxPreprocess(d_img_, img.cols, img.rows, img.cols * 3, d_input_, w_, h_, lb, stream_);
        cudaStreamSynchronize(stream_);
        bindings[0] = d_input_;
        if (idx_ % 100 == 0) std::cerr << "kalibrasyon " << idx_ << "/" << files_.size() << "\n";
        return true;
    }
    const void* readCalibrationCache(std::size_t& length) noexcept override {
        cache_data_.clear();
        std::ifstream f(cache_, std::ios::binary);
        if (f) cache_data_.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        length = cache_data_.size();
        return length ? cache_data_.data() : nullptr;
    }
    void writeCalibrationCache(const void* data, std::size_t length) noexcept override {
        std::ofstream(cache_, std::ios::binary).write(static_cast<const char*>(data), static_cast<std::streamsize>(length));
    }

private:
    std::vector<std::string> files_;
    std::size_t idx_ = 0;
    int w_, h_;
    std::string cache_;
    std::vector<char> cache_data_;
    float* d_input_ = nullptr;
    uint8_t* d_img_ = nullptr;
    std::size_t cap_ = 0;
    cudaStream_t stream_ = nullptr;
};
#pragma GCC diagnostic pop

}  // namespace

int main(int argc, char** argv) try {
    const iha::Args a(argc, argv);
    const std::string onnx = a.required("onnx"), out = a.required("out");
    const std::string precision = a.str("precision", "fp16");

    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(gLogger));
    std::unique_ptr<nvinfer1::INetworkDefinition> net(builder->createNetworkV2(0));
    std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*net, gLogger));
    if (!parser->parseFromFile(onnx.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
        throw std::runtime_error("ONNX ayristirilamadi: " + onnx);

    std::unique_ptr<nvinfer1::IBuilderConfig> cfg(builder->createBuilderConfig());
    cfg->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 2ULL << 30);
    std::unique_ptr<EntropyCalibrator> calib;
    if (precision == "fp16" || precision == "int8") cfg->setFlag(nvinfer1::BuilderFlag::kFP16);
    if (precision == "int8") {
        std::vector<std::string> files;
        std::ifstream lst(a.required("calib-list"));
        for (std::string l; std::getline(lst, l);) {
            while (!l.empty() && (l.back() == '\r' || l.back() == ' ')) l.pop_back();  // CRLF listeler
            if (!l.empty()) files.push_back(l);
        }
        const auto dims = net->getInput(0)->getDimensions();
        calib = std::make_unique<EntropyCalibrator>(files, static_cast<int>(dims.d[3]), static_cast<int>(dims.d[2]),
                                                    a.str("calib-cache", out + ".calib"));
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        cfg->setFlag(nvinfer1::BuilderFlag::kINT8);
        cfg->setInt8Calibrator(calib.get());
#pragma GCC diagnostic pop
        std::cerr << "INT8 kalibrasyon: " << files.size() << " goruntu\n";
    } else if (precision != "fp32" && precision != "fp16") {
        throw std::invalid_argument("--precision fp32|fp16|int8");
    }

    std::unique_ptr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*net, *cfg));
    if (!plan) throw std::runtime_error("motor derlenemedi");
    std::ofstream(out, std::ios::binary).write(static_cast<const char*>(plan->data()), static_cast<std::streamsize>(plan->size()));
    std::cout << "yazildi: " << out << " (" << plan->size() / (1 << 20) << " MB, " << precision << ")\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "hata: " << e.what() << "\n";
    return 1;
}
