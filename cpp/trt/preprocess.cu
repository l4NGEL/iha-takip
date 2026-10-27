#include <cmath>

#include "iha/preprocess.hpp"

namespace iha {

Letterbox Letterbox::compute(int src_w, int src_h, int dst_w, int dst_h) {
    Letterbox lb;
    lb.scale = fminf(static_cast<float>(dst_w) / src_w, static_cast<float>(dst_h) / src_h);
    lb.new_w = static_cast<int>(lroundf(src_w * lb.scale));
    lb.new_h = static_cast<int>(lroundf(src_h * lb.scale));
    const float dw = (dst_w - lb.new_w) / 2.f, dh = (dst_h - lb.new_h) / 2.f;
    lb.left = static_cast<int>(lroundf(dw - 0.1f));
    lb.top = static_cast<int>(lroundf(dh - 0.1f));
    return lb;
}

namespace {

__global__ void letterboxKernel(const uint8_t* __restrict__ src, int src_w, int src_h, int src_pitch,
                                float* __restrict__ dst, int dst_w, int dst_h, int new_w, int new_h, int left,
                                int top) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;
    const int plane = dst_w * dst_h;
    const int o = y * dst_w + x;

    const int ix = x - left, iy = y - top;
    if (ix < 0 || iy < 0 || ix >= new_w || iy >= new_h) {
        const float pad = 114.f / 255.f;
        dst[o] = pad;
        dst[plane + o] = pad;
        dst[2 * plane + o] = pad;
        return;
    }
    // cv::resize INTER_LINEAR: kaynak = (hedef + 0.5) * (kaynak/hedef) - 0.5, kenarda kirpilir.
    const float fx = (ix + 0.5f) * (static_cast<float>(src_w) / new_w) - 0.5f;
    const float fy = (iy + 0.5f) * (static_cast<float>(src_h) / new_h) - 0.5f;
    const int x0 = max(0, min(src_w - 1, static_cast<int>(floorf(fx))));
    const int y0 = max(0, min(src_h - 1, static_cast<int>(floorf(fy))));
    const int x1 = min(src_w - 1, x0 + 1), y1 = min(src_h - 1, y0 + 1);
    const float ax = fminf(fmaxf(fx - x0, 0.f), 1.f), ay = fminf(fmaxf(fy - y0, 0.f), 1.f);

    const uint8_t* r0 = src + y0 * src_pitch;
    const uint8_t* r1 = src + y1 * src_pitch;
    for (int c = 0; c < 3; ++c) {
        const float v00 = r0[x0 * 3 + c], v01 = r0[x1 * 3 + c];
        const float v10 = r1[x0 * 3 + c], v11 = r1[x1 * 3 + c];
        const float v = (v00 * (1.f - ax) + v01 * ax) * (1.f - ay) + (v10 * (1.f - ax) + v11 * ax) * ay;
        // BGR -> RGB: kaynak kanal c, hedef duzlem 2 - c.
        dst[(2 - c) * plane + o] = v / 255.f;
    }
}

}  // namespace

void launchLetterboxPreprocess(const uint8_t* d_src, int src_w, int src_h, int src_pitch, float* d_dst, int dst_w,
                               int dst_h, const Letterbox& lb, cudaStream_t stream) {
    const dim3 block(32, 8);
    const dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);
    letterboxKernel<<<grid, block, 0, stream>>>(d_src, src_w, src_h, src_pitch, d_dst, dst_w, dst_h, lb.new_w,
                                                lb.new_h, lb.left, lb.top);
}

}  // namespace iha
