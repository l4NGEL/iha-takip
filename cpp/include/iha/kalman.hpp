#pragma once

#include <Eigen/Dense>

#include "iha/geometry.hpp"

namespace iha {

// Sabit hizli kutu filtresi. Durum: [cx, cy, w, h, vcx, vcy, vw, vh] (piksel, piksel/kare).
// Gurultu standart sapmalari kutu boyutuyla olceklenir (ByteTrack/DeepSORT gibi) ama
// alt sinirlidir: 8 px'lik hedefte boyuta oranli sapma 0.4 px'e iner, bu da tek
// karelik bir sicramayi "imkansiz" sayip takibi kopartir.
struct KalmanParams {
    float std_pos = 1.f / 20.f;   // olcum ve surec, boyutun orani
    float std_vel = 1.f / 80.f;   // hiz sureci, boyutun orani
    float min_std_pos = 1.0f;     // piksel
    float min_std_vel = 0.5f;     // piksel/kare
    // Yeni takibin hizi bilinmiyor: ilk sapma genis olmali, yoksa hizli hedefin ikinci
    // karedeki tespiti kapi disinda kalir ve takip hic onaylanmaz.
    float init_std_vel = 10.f;    // piksel/kare
};

class KalmanBoxFilter {
public:
    using Vec8 = Eigen::Matrix<float, 8, 1>;
    using Mat8 = Eigen::Matrix<float, 8, 8>;
    using Vec4 = Eigen::Matrix<float, 4, 1>;
    using Mat4 = Eigen::Matrix<float, 4, 4>;

    explicit KalmanBoxFilter(const Box& init, const KalmanParams& p = {});

    void predict();
    void update(const Box& meas);
    // Kamera hareketi telafisi: durumu onceki kare -> simdiki kare afin eslemesiyle tasir.
    // Merkez ve hiz L ile, boyut ve boyut hizi izotrop olcekle donusur; kovaryans J P J^T.
    void warp(const Affine2D& A);
    void shift(float dx, float dy) { warp(Affine2D::translation(dx, dy)); }

    Box box() const;
    float vx() const { return x_(4); }
    float vy() const { return x_(5); }
    const Vec8& state() const { return x_; }
    const Mat8& covariance() const { return P_; }

    // Olcum uzayinda Mahalanobis uzakligi^2 (yalniz merkez; 2 serbestlik derecesi).
    float centerMahalanobis2(const Box& meas) const;

private:
    float scale() const;  // gurultu olcegi: max(w, h)

    KalmanParams p_;
    Vec8 x_;
    Mat8 P_;
    Mat8 F_;
};

}  // namespace iha
