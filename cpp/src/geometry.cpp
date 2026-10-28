#include "iha/geometry.hpp"

#include <algorithm>
#include <cmath>

namespace iha {

float iou(const Box& a, const Box& b) {
    const float ix = std::max(0.f, std::min(a.x + a.w, b.x + b.w) - std::max(a.x, b.x));
    const float iy = std::max(0.f, std::min(a.y + a.h, b.y + b.h) - std::max(a.y, b.y));
    const float inter = ix * iy;
    const float uni = a.area() + b.area() - inter;
    return uni > 0.f ? inter / uni : 0.f;
}

float Affine2D::scale() const { return std::sqrt(std::abs(a * d - b * c)); }

float diou(const Box& a, const Box& b) {
    const float dx = a.cx() - b.cx();
    const float dy = a.cy() - b.cy();
    const float ex = std::max(a.x + a.w, b.x + b.w) - std::min(a.x, b.x);
    const float ey = std::max(a.y + a.h, b.y + b.h) - std::min(a.y, b.y);
    const float c2 = ex * ex + ey * ey;
    if (c2 <= 0.f) return 0.f;
    return iou(a, b) - (dx * dx + dy * dy) / c2;
}

}  // namespace iha
