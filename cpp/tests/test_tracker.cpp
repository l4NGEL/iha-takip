#include <gtest/gtest.h>

#include "iha/tracker.hpp"

using iha::Box;
using iha::ByteTracker;
using iha::Detection;
using iha::Detections;
using iha::TrackerParams;

namespace {
Detection det(float cx, float cy, float s, float size = 8.f) { return {Box::fromCenter(cx, cy, size, size), s}; }
}  // namespace

TEST(Tracker, ConfirmsAfterMinHitsAndKeepsId) {
    ByteTracker tr;
    tr.update({det(100, 100, 0.9f)});
    EXPECT_FALSE(tr.target().has_value());  // aday, henuz onaysiz
    for (int t = 1; t < 10; ++t) tr.update({det(100.f + 4.f * static_cast<float>(t), 100, 0.9f)});
    ASSERT_TRUE(tr.target().has_value());
    EXPECT_EQ(tr.target()->track_id, 1);
    EXPECT_TRUE(tr.target()->measured);
    EXPECT_NEAR(tr.target()->vx, 4.f, 0.5f);
}

TEST(Tracker, FastTinyTargetWithZeroIouStaysOneTrack) {
    // 6 px hedef karede 10 px kayiyor: ardisik kutular hic ortusmez, DIoU + Kalman ile tek takip.
    ByteTracker tr;
    for (int t = 0; t < 30; ++t) tr.update({det(50.f + 10.f * static_cast<float>(t), 200, 0.8f, 6.f)});
    ASSERT_TRUE(tr.target().has_value());
    EXPECT_EQ(tr.target()->track_id, 1);
    EXPECT_EQ(tr.tracks().size(), 1u);
}

TEST(Tracker, LowScoreDetectionKeepsConfirmedTrack) {
    TrackerParams p;
    ByteTracker tr(p);
    for (int t = 0; t < 5; ++t) tr.update({det(100.f + 2.f * static_cast<float>(t), 100, 0.9f)});
    // Hedef bulaniklasiyor: skor 0.2 (high_thr alti, low_thr ustu) -> ikinci asamada eslesir.
    tr.update({det(110, 100, 0.2f)});
    ASSERT_TRUE(tr.target().has_value());
    EXPECT_TRUE(tr.target()->measured);
    EXPECT_EQ(tr.target()->track_id, 1);
}

TEST(Tracker, LowScoreDetectionDoesNotStartTrack) {
    ByteTracker tr;
    for (int t = 0; t < 5; ++t) tr.update({det(100, 100, 0.3f)});
    EXPECT_TRUE(tr.tracks().empty());
    EXPECT_FALSE(tr.target().has_value());
}

TEST(Tracker, CoastsThroughShortGapThenDrops) {
    TrackerParams p;
    p.coast_output = 3;
    ByteTracker tr(p);
    for (int t = 0; t < 10; ++t) tr.update({det(100.f + 5.f * static_cast<float>(t), 100, 0.9f)});
    for (int k = 1; k <= 3; ++k) {
        tr.update({});
        ASSERT_TRUE(tr.target().has_value()) << "bosluk karesi " << k;
        EXPECT_FALSE(tr.target()->measured);
        EXPECT_NEAR(tr.target()->box.cx(), 100.f + 5.f * static_cast<float>(9 + k), 1.5f);
    }
    tr.update({});
    EXPECT_FALSE(tr.target().has_value());
}

TEST(Tracker, ReacquiresLostTrackWithSameId) {
    ByteTracker tr;
    for (int t = 0; t < 10; ++t) tr.update({det(100.f + 3.f * static_cast<float>(t), 100, 0.9f)});
    for (int k = 0; k < 8; ++k) tr.update({});
    tr.update({det(100.f + 3.f * 18, 100, 0.9f)});
    ASSERT_TRUE(tr.target().has_value());
    EXPECT_EQ(tr.target()->track_id, 1);
}

TEST(Tracker, LockedTargetIsNotStolenByBrighterClutter) {
    ByteTracker tr;
    for (int t = 0; t < 10; ++t) tr.update({det(100.f + 2.f * static_cast<float>(t), 100, 0.6f)});
    // Uzakta yuksek skorlu bir parazit belirir ve onaylanir; kilit degismemeli.
    for (int t = 10; t < 20; ++t) tr.update({det(100.f + 2.f * static_cast<float>(t), 100, 0.6f), det(500, 400, 0.95f)});
    ASSERT_TRUE(tr.target().has_value());
    EXPECT_EQ(tr.target()->track_id, 1);
    EXPECT_EQ(tr.tracks().size(), 2u);
}

TEST(Tracker, FarJumpIsGatedAndBecomesNewTrack) {
    ByteTracker tr;
    for (int t = 0; t < 10; ++t) tr.update({det(100, 100, 0.9f)});
    tr.update({det(400, 300, 0.9f)});
    bool has_new = false;
    for (const auto& t : tr.tracks())
        if (t.id != 1) has_new = true;
    EXPECT_TRUE(has_new);
}

TEST(Tracker, ResetClearsEverything) {
    ByteTracker tr;
    for (int t = 0; t < 5; ++t) tr.update({det(10, 10, 0.9f)});
    tr.reset();
    EXPECT_TRUE(tr.tracks().empty());
    EXPECT_FALSE(tr.target().has_value());
    tr.update({det(10, 10, 0.9f)});
    EXPECT_EQ(tr.tracks()[0].id, 1);
}

TEST(Tracker, CameraMotionCompensationKeepsLockWhileCameraPans) {
    // Hedef dunyada duruyor; kamera her kare 12 px saga donuyor -> hedef goruntude 12 px sola kayar.
    // Telafiyle Kalman hizi ~0 kalir ve kilit tek takipte surer.
    ByteTracker tr;
    for (int t = 0; t < 20; ++t) {
        if (t > 0) tr.compensateCameraMotion(-12.f, 0.f);
        tr.update({det(500.f - 12.f * static_cast<float>(t), 200, 0.8f)});
    }
    ASSERT_TRUE(tr.target().has_value());
    EXPECT_EQ(tr.target()->track_id, 1);
    EXPECT_NEAR(tr.target()->vx, 0.f, 0.5f);
    EXPECT_EQ(tr.tracks().size(), 1u);
}

TEST(Tracker, ReversingImageMotionWithoutCompensationBreaksTrack) {
    // Telafisiz hata modu: hedef 11 px/kare saga giderken kamera donmeye baslar ve goruntudeki
    // hareket aniden 8 px/kare sola doner; Kalman eski hizla ongordugu icin tespit kapi disinda kalir.
    ByteTracker tr;
    float u = 200.f;
    for (int t = 0; t < 10; ++t, u += 11.f) tr.update({det(u, 200, 0.8f)});
    for (int t = 0; t < 4; ++t, u -= 8.f) tr.update({det(u, 200, 0.8f)});
    EXPECT_GT(tr.tracks().size(), 1u);
}

TEST(Tracker, CoastingLockYieldsToMeasuredTrack) {
    // Kilit bir takipte; o takip kayboluyor, baska yerde olculen onayli bir takip var -> kilit gecer.
    TrackerParams p;
    p.coast_output = 30;
    ByteTracker tr(p);
    for (int t = 0; t < 10; ++t) tr.update({det(100, 100, 0.9f)});
    for (int t = 0; t < 3; ++t) tr.update({det(100, 100, 0.9f), det(400, 300, 0.8f)});
    ASSERT_EQ(tr.target()->track_id, 1);
    tr.update({det(400, 300, 0.8f)});
    ASSERT_TRUE(tr.target().has_value());
    EXPECT_NE(tr.target()->track_id, 1);
    EXPECT_TRUE(tr.target()->measured);

    p.lock_yields_to_measured = false;
    ByteTracker sticky(p);
    for (int t = 0; t < 10; ++t) sticky.update({det(100, 100, 0.9f)});
    for (int t = 0; t < 3; ++t) sticky.update({det(100, 100, 0.9f), det(400, 300, 0.8f)});
    sticky.update({det(400, 300, 0.8f)});
    EXPECT_EQ(sticky.target()->track_id, 1);
    EXPECT_FALSE(sticky.target()->measured);
}
