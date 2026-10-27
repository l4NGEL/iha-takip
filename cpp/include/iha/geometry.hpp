#pragma once

#include <vector>

namespace iha {

// Piksel koordinatinda eksene hizali kutu: sol ust kose + boyut.
struct Box {
    float x = 0.f, y = 0.f, w = 0.f, h = 0.f;

    float cx() const { return x + 0.5f * w; }
    float cy() const { return y + 0.5f * h; }
    float area() const { return w * h; }
    static Box fromCenter(float cx, float cy, float w, float h) { return {cx - 0.5f * w, cy - 0.5f * h, w, h}; }
};

struct Detection {
    Box box;
    float score = 0.f;
};

using Detections = std::vector<Detection>;

float iou(const Box& a, const Box& b);

// Distance-IoU: IoU - (merkez uzakligi^2 / kapsayan kutunun kosegeni^2), [-1, 1].
// 8 px'lik hizli hedefte ardisik kareler hic ortusmeyebilir; IoU 0'a duser ama
// DIoU yakin olan adayi uzak olandan hala ayirabilir.
float diou(const Box& a, const Box& b);

}  // namespace iha
