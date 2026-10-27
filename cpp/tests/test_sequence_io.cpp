#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "iha/sequence_io.hpp"

namespace fs = std::filesystem;

namespace {
fs::path tmpDir(const std::string& name) {
    auto p = fs::temp_directory_path() / ("iha_test_" + name);
    fs::remove_all(p);
    fs::create_directories(p);
    return p;
}
}  // namespace

TEST(SequenceIo, FrameFilename) {
    EXPECT_EQ(iha::frameFilename(0), "000001.jpg");
    EXPECT_EQ(iha::frameFilename(1499), "001500.jpg");
}

TEST(SequenceIo, DetectionCsvRoundTrip) {
    const auto dir = tmpDir("dets");
    const auto path = (dir / "d.csv").string();
    {
        iha::DetectionCsvWriter w(path);
        w.write(0, {{{1.5f, 2.f, 8.f, 6.f}, 0.9f}, {{100.f, 50.f, 4.f, 4.f}, 0.05f}});
        w.write(3, {{{10.f, 20.f, 30.f, 40.f}, 0.5f}});
    }
    const auto m = iha::readDetectionsCsv(path);
    ASSERT_EQ(m.size(), 2u);
    ASSERT_EQ(m.at(0).size(), 2u);
    EXPECT_FLOAT_EQ(m.at(0)[0].box.x, 1.5f);
    EXPECT_FLOAT_EQ(m.at(0)[1].score, 0.05f);
    EXPECT_FLOAT_EQ(m.at(3)[0].box.h, 40.f);
    EXPECT_EQ(m.count(1), 0u);
}

TEST(SequenceIo, CountFramesStopsAtGap) {
    const auto dir = tmpDir("frames");
    for (int i : {0, 1, 2, 4}) std::ofstream(dir / iha::frameFilename(i)) << "x";
    EXPECT_EQ(iha::countFrames(dir.string()), 3);
}

TEST(SequenceIo, ListSequenceDirsSorted) {
    const auto dir = tmpDir("split");
    fs::create_directories(dir / "b");
    fs::create_directories(dir / "a");
    std::ofstream(dir / "not_a_dir.txt") << "x";
    const auto d = iha::listSequenceDirs(dir.string());
    ASSERT_EQ(d.size(), 2u);
    EXPECT_EQ(fs::path(d[0]).filename(), "a");
}

TEST(SequenceIo, TrackCsvWritesEmptyRowWhenNoTarget) {
    const auto dir = tmpDir("trk");
    const auto path = (dir / "t.csv").string();
    {
        iha::TrackCsvWriter w(path);
        w.write(0, std::nullopt, {});
        iha::TargetOutput t;
        t.track_id = 3;
        t.box = {1, 2, 3, 4};
        t.measured = true;
        w.write(1, t, {0.01, -0.02});
    }
    std::ifstream is(path);
    std::string header, l0, l1;
    std::getline(is, header);
    std::getline(is, l0);
    std::getline(is, l1);
    EXPECT_EQ(l0, "0,0,0,-1,,,,,,,");
    EXPECT_EQ(l1.substr(0, 10), "1,1,1,3,1.");
}
