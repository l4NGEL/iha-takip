#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "iha/geometry.hpp"
#include "iha/latency.hpp"
#include "iha/preprocess.hpp"

namespace nvinfer1 {
class IRuntime;
class ICudaEngine;
class IExecutionContext;
}  // namespace nvinfer1

namespace iha {

// YOLO26 uctan uca (NMS'siz) TensorRT motoru: girdi [1,3,H,W] float, cikti [1,K,6]
// (x1, y1, x2, y2, skor, sinif) ag girdisi koordinatinda.
class TrtDetector {
public:
    explicit TrtDetector(const std::string& engine_path, float conf_thr = 0.01f);
    ~TrtDetector();
    TrtDetector(const TrtDetector&) = delete;
    TrtDetector& operator=(const TrtDetector&) = delete;

    // Asama sureleri stats'e eklenir (verilirse): yukleme, on_isleme, cikarim, son_isleme.
    // Olcum icin her asamadan sonra stream senkronlanir; uretimde --no-stage-sync ile kapatilir.
    Detections detect(const cv::Mat& bgr, LatencyStats* stats = nullptr);

    int inputWidth() const { return in_w_; }
    int inputHeight() const { return in_h_; }
    void setStageSync(bool on) { stage_sync_ = on; }

    // Ham on islem ciktisini (CHW float) dondurur; Python ile parite testi icin.
    std::vector<float> preprocessOnly(const cv::Mat& bgr);

private:
    void upload(const cv::Mat& bgr);

    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> ctx_;
    cudaStream_t stream_ = nullptr;
    std::string in_name_, out_name_;
    int in_w_ = 640, in_h_ = 640, out_k_ = 300, out_c_ = 6;
    float conf_thr_;
    bool stage_sync_ = true;

    // Cihaz ve sabitlenmis (pinned) ana bellek tamponlari
    float* d_input_ = nullptr;
    float* d_output_ = nullptr;
    float* h_output_ = nullptr;
    uint8_t* d_image_ = nullptr;
    uint8_t* h_image_ = nullptr;
    std::size_t image_cap_ = 0;
    int img_w_ = 0, img_h_ = 0;
    Letterbox lb_;
};

}  // namespace iha
