#include <gtest/gtest.h>

#include "iha/geometry.hpp"

using iha::Box;

TEST(Geometry, IouIdenticalAndDisjoint) {
    EXPECT_FLOAT_EQ(iha::iou({0, 0, 10, 10}, {0, 0, 10, 10}), 1.f);
    EXPECT_FLOAT_EQ(iha::iou({0, 0, 10, 10}, {20, 20, 5, 5}), 0.f);
}

TEST(Geometry, IouHalfOverlap) { EXPECT_NEAR(iha::iou({0, 0, 10, 10}, {5, 0, 10, 10}), 50.f / 150.f, 1e-6); }

TEST(Geometry, DiouEqualsIouWhenCentersCoincide) {
    const Box a{0, 0, 10, 10}, b{2, 2, 6, 6};
    EXPECT_NEAR(iha::diou(a, b), iha::iou(a, b), 1e-6);
}

TEST(Geometry, DiouRanksNonOverlappingByDistance) {
    // 8 px'lik hedef: ikisi de ortusmuyor (IoU = 0) ama yakindaki aday daha iyi puan alir.
    const Box track{100, 100, 8, 8};
    const Box near{110, 100, 8, 8}, far{160, 100, 8, 8};
    EXPECT_FLOAT_EQ(iha::iou(track, near), 0.f);
    EXPECT_FLOAT_EQ(iha::iou(track, far), 0.f);
    EXPECT_GT(iha::diou(track, near), iha::diou(track, far));
    EXPECT_GE(iha::diou(track, far), -1.f);
}

TEST(Geometry, FromCenterRoundTrip) {
    const Box b = Box::fromCenter(50, 40, 10, 6);
    EXPECT_FLOAT_EQ(b.cx(), 50);
    EXPECT_FLOAT_EQ(b.cy(), 40);
    EXPECT_FLOAT_EQ(b.x, 45);
    EXPECT_FLOAT_EQ(b.y, 37);
}
