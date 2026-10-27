#include "iha/trt_detector.hpp"

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace iha {

namespace {

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING) std::cerr << "[TRT] " << msg << "\n";
    }
};
Logger gLogger;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

}  // namespace

TrtDetector::TrtDetector(const std::string& engine_path, float conf_thr) : conf_thr_(conf_thr) {
    std::ifstream f(engine_path, std::ios::binary);
    if (!f) throw std::runtime_error("motor dosyasi acilamadi: " + engine_path);
    std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    runtime_.reset(nvinfer1::createInferRuntime(gLogger));
    engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
    if (!engine_) throw std::runtime_error("motor yuklenemedi (TensorRT surumu/GPU uyumsuz olabilir)");
    ctx_.reset(engine_->createExecutionContext());

    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        const char* name = engine_->getIOTensorName(i);
        const auto dims = engine_->getTensorShape(name);
        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            in_name_ = name;
            if (dims.nbDims != 4 || dims.d[1] != 3) throw std::runtime_error("beklenmeyen girdi boyutu");
            in_h_ = static_cast<int>(dims.d[2]);
            in_w_ = static_cast<int>(dims.d[3]);
        } else {
            out_name_ = name;
            if (dims.nbDims != 3 || dims.d[2] != 6)
                throw std::runtime_error("beklenmeyen cikti: [1,K,6] uctan uca YOLO26 bekleniyor");
            out_k_ = static_cast<int>(dims.d[1]);
            out_c_ = static_cast<int>(dims.d[2]);
        }
    }
    check(cudaStreamCreate(&stream_), "cudaStreamCreate");
    check(cudaMalloc(&d_input_, sizeof(float) * 3 * in_w_ * in_h_), "cudaMalloc girdi");
    check(cudaMalloc(&d_output_, sizeof(float) * out_k_ * out_c_), "cudaMalloc cikti");
    check(cudaMallocHost(&h_output_, sizeof(float) * out_k_ * out_c_), "cudaMallocHost cikti");
    ctx_->setTensorAddress(in_name_.c_str(), d_input_);
    ctx_->setTensorAddress(out_name_.c_str(), d_output_);
}

TrtDetector::~TrtDetector() {
    cudaFree(d_input_);
    cudaFree(d_output_);
    cudaFreeHost(h_output_);
    cudaFree(d_image_);
    cudaFreeHost(h_image_);
    if (stream_) cudaStreamDestroy(stream_);
    ctx_.reset();
    engine_.reset();
    runtime_.reset();
}

void TrtDetector::upload(const cv::Mat& bgr) {
    if (bgr.type() != CV_8UC3) throw std::runtime_error("BGR 8 bit 3 kanal goruntu bekleniyor");
    const std::size_t bytes = bgr.total() * 3;
    if (bytes > image_cap_) {
        cudaFree(d_image_);
        cudaFreeHost(h_image_);
        check(cudaMalloc(&d_image_, bytes), "cudaMalloc goruntu");
        check(cudaMallocHost(&h_image_, bytes), "cudaMallocHost goruntu");
        image_cap_ = bytes;
    }
    if (bgr.cols != img_w_ || bgr.rows != img_h_) {
        img_w_ = bgr.cols;
        img_h_ = bgr.rows;
        lb_ = Letterbox::compute(img_w_, img_h_, in_w_, in_h_);
    }
    // Sabitlenmis tampona kopya, DMA ile asenkron yukleme.
    if (bgr.isContinuous())
        std::memcpy(h_image_, bgr.data, bytes);
    else
        for (int r = 0; r < bgr.rows; ++r) std::memcpy(h_image_ + r * bgr.cols * 3, bgr.ptr(r), bgr.cols * 3);
    check(cudaMemcpyAsync(d_image_, h_image_, bytes, cudaMemcpyHostToDevice, stream_), "H2D");
}

std::vector<float> TrtDetector::preprocessOnly(const cv::Mat& bgr) {
    upload(bgr);
    launchLetterboxPreprocess(d_image_, img_w_, img_h_, img_w_ * 3, d_input_, in_w_, in_h_, lb_, stream_);
    std::vector<float> out(3 * in_w_ * in_h_);
    check(cudaMemcpyAsync(out.data(), d_input_, out.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_), "D2H");
    check(cudaStreamSynchronize(stream_), "sync");
    return out;
}

Detections TrtDetector::detect(const cv::Mat& bgr, LatencyStats* stats) {
    Stopwatch sw;
    auto mark = [&](const char* stage) {
        if (!stats) return;
        if (stage_sync_) check(cudaStreamSynchronize(stream_), "sync");
        stats->add(stage, sw.lap());
    };

    upload(bgr);
    mark("yukleme");
    launchLetterboxPreprocess(d_image_, img_w_, img_h_, img_w_ * 3, d_input_, in_w_, in_h_, lb_, stream_);
    check(cudaGetLastError(), "on isleme kerneli");
    mark("on_isleme");
    if (!ctx_->enqueueV3(stream_)) throw std::runtime_error("enqueueV3 basarisiz");
    mark("cikarim");
    check(cudaMemcpyAsync(h_output_, d_output_, sizeof(float) * out_k_ * out_c_, cudaMemcpyDeviceToHost, stream_),
          "D2H");
    check(cudaStreamSynchronize(stream_), "sync");

    Detections dets;
    for (int k = 0; k < out_k_; ++k) {
        const float* r = h_output_ + k * out_c_;
        if (r[4] < conf_thr_) continue;
        float x1 = lb_.toSrcX(r[0]), y1 = lb_.toSrcY(r[1]);
        float x2 = lb_.toSrcX(r[2]), y2 = lb_.toSrcY(r[3]);
        x1 = std::clamp(x1, 0.f, static_cast<float>(img_w_));
        x2 = std::clamp(x2, 0.f, static_cast<float>(img_w_));
        y1 = std::clamp(y1, 0.f, static_cast<float>(img_h_));
        y2 = std::clamp(y2, 0.f, static_cast<float>(img_h_));
        if (x2 <= x1 || y2 <= y1) continue;
        dets.push_back({{x1, y1, x2 - x1, y2 - y1}, r[4]});
    }
    if (stats) stats->add("son_isleme", sw.lap());
    return dets;
}

}  // namespace iha
