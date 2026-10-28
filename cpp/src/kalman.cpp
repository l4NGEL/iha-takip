#include "iha/kalman.hpp"

#include <algorithm>

namespace iha {

namespace {
float sq(float v) { return v * v; }
}  // namespace

KalmanBoxFilter::KalmanBoxFilter(const Box& init, const KalmanParams& p) : p_(p) {
    x_.setZero();
    x_ << init.cx(), init.cy(), init.w, init.h, 0.f, 0.f, 0.f, 0.f;
    F_.setIdentity();
    for (int i = 0; i < 4; ++i) F_(i, i + 4) = 1.f;
    const float s = scale();
    const float sp = std::max(p_.min_std_pos, 2.f * p_.std_pos * s);
    const float sv = std::max(p_.init_std_vel, 10.f * p_.std_vel * s);
    P_.setZero();
    for (int i = 0; i < 4; ++i) {
        P_(i, i) = sq(sp);
        P_(i + 4, i + 4) = sq(sv);
    }
}

float KalmanBoxFilter::scale() const { return std::max(x_(2), x_(3)); }

void KalmanBoxFilter::predict() {
    const float s = scale();
    const float sp = std::max(p_.min_std_pos, p_.std_pos * s);
    const float sv = std::max(p_.min_std_vel, p_.std_vel * s);
    Mat8 Q = Mat8::Zero();
    for (int i = 0; i < 4; ++i) {
        Q(i, i) = sq(sp);
        Q(i + 4, i + 4) = sq(sv);
    }
    x_ = F_ * x_;
    P_ = F_ * P_ * F_.transpose() + Q;
    // Boyut negatif ya da sifir olmasin (kuculen hiz kestirimi tasabilir).
    x_(2) = std::max(x_(2), 1.f);
    x_(3) = std::max(x_(3), 1.f);
}

void KalmanBoxFilter::update(const Box& meas) {
    const float sp = std::max(p_.min_std_pos, p_.std_pos * scale());
    Mat4 R = Mat4::Identity() * sq(sp);
    Eigen::Matrix<float, 4, 8> H = Eigen::Matrix<float, 4, 8>::Zero();
    H.leftCols<4>().setIdentity();
    Vec4 z;
    z << meas.cx(), meas.cy(), meas.w, meas.h;
    const Vec4 y = z - H * x_;
    const Mat4 S = H * P_ * H.transpose() + R;
    const Eigen::Matrix<float, 8, 4> K = P_ * H.transpose() * S.inverse();
    x_ += K * y;
    // Joseph formu: yuvarlama hatasinda da P simetrik ve pozitif tanimli kalir.
    const Mat8 IKH = Mat8::Identity() - K * H;
    P_ = IKH * P_ * IKH.transpose() + K * R * K.transpose();
}

void KalmanBoxFilter::warp(const Affine2D& A) {
    const float s = A.scale();
    Mat8 J = Mat8::Zero();
    for (int base : {0, 4}) {  // konum ve hiz bloklari
        J(base, base) = A.a;
        J(base, base + 1) = A.b;
        J(base + 1, base) = A.c;
        J(base + 1, base + 1) = A.d;
        J(base + 2, base + 2) = s;
        J(base + 3, base + 3) = s;
    }
    x_ = J * x_;
    x_(0) += A.tx;
    x_(1) += A.ty;
    P_ = J * P_ * J.transpose();
}

Box KalmanBoxFilter::box() const { return Box::fromCenter(x_(0), x_(1), x_(2), x_(3)); }

float KalmanBoxFilter::centerMahalanobis2(const Box& meas) const {
    const float sp = std::max(p_.min_std_pos, p_.std_pos * scale());
    Eigen::Matrix2f S = P_.topLeftCorner<2, 2>() + Eigen::Matrix2f::Identity() * sq(sp);
    Eigen::Vector2f d(meas.cx() - x_(0), meas.cy() - x_(1));
    return d.dot(S.ldlt().solve(d));
}

}  // namespace iha
