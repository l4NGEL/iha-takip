#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

namespace iha {

// Ultralytics LetterBox ile ayni yerlesim: oran korunur, kalan alan 114 ile doldurulur,
// goruntu ortalanir (ust/sol dolgu = round(d - 0.1)).
struct Letterbox {
    float scale = 1.f;        // kaynak -> ag girdisi
    int new_w = 0, new_h = 0; // olceklenmis kaynak boyutu
    int left = 0, top = 0;    // dolgu

    static Letterbox compute(int src_w, int src_h, int dst_w, int dst_h);
    // Ag girdisi koordinatini kaynak goruntu koordinatina cevirir.
    float toSrcX(float x) const { return (static_cast<float>(x) - static_cast<float>(left)) / scale; }
    float toSrcY(float y) const { return (static_cast<float>(y) - static_cast<float>(top)) / scale; }
};

// BGR uint8 (HxWx3, satir adimi src_pitch bayt) -> RGB float CHW [0,1], dst_w x dst_h,
// cift dogrusal ornekleme (cv::resize INTER_LINEAR ile ayni piksel merkezi kurali).
void launchLetterboxPreprocess(const uint8_t* d_src, int src_w, int src_h, int src_pitch, float* d_dst, int dst_w,
                               int dst_h, const Letterbox& lb, cudaStream_t stream);

}  // namespace iha
