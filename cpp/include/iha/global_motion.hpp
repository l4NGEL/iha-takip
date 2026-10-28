#pragma once

#include <optional>
#include <vector>

#include <opencv2/core.hpp>

#include "iha/geometry.hpp"

namespace iha {

struct GlobalMotionParams {
    int max_corners = 300;
    double quality = 0.01;      // goodFeaturesToTrack kalite orani
    double min_distance = 12;   // kose noktalari arasi en az uzaklik (piksel)
    int lk_window = 21;
    int lk_levels = 3;
    double ransac_px = 1.0;     // RANSAC ic nokta esigi (piksel)
    int min_inliers = 25;       // daha azsa kestirim yok sayilir (dokusuz gokyuzu)
    float downscale = 1.f;      // hiz icin ornegin 0.5; donusum tam cozunurluge geri olceklenir
};

struct GlobalMotion {
    Affine2D A;      // onceki kare -> simdiki kare (tam cozunurluk pikseli)
    int inliers = 0;
    int tracked = 0;
};

// Ardisik gri karelerden kamera hareketini (benzerlik donusumu) kestirir:
// kose noktalari -> piramit Lucas-Kanade -> RANSAC ile estimateAffinePartial2D.
// Hedefin kendisi kucuk oldugundan RANSAC'ta aykiri deger olarak elenir; yine de verilen
// kutular (ornegin bir onceki karenin tespitleri) kose aramasindan maskelenebilir.
class GlobalMotionEstimator {
public:
    explicit GlobalMotionEstimator(const GlobalMotionParams& p = {}) : p_(p) {}

    // Ilk cagrida (onceki kare yok) nullopt. exclude: bir onceki karede maskelenecek kutular.
    std::optional<GlobalMotion> update(const cv::Mat& gray, const std::vector<Box>& exclude = {});
    void reset() { prev_.release(); }

private:
    GlobalMotionParams p_;
    cv::Mat prev_;
};

}  // namespace iha
