#include "iha/sequence_io.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace iha {

namespace {
constexpr double kRad2Deg = 57.29577951308232;
}

std::string frameFilename(int index) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%06d.jpg", index + 1);
    return buf;
}

std::vector<std::string> listSequenceDirs(const std::string& split_dir) {
    std::vector<std::string> out;
    for (const auto& e : fs::directory_iterator(split_dir))
        if (e.is_directory()) out.push_back(e.path().string());
    std::sort(out.begin(), out.end());
    return out;
}

int countFrames(const std::string& seq_dir) {
    // Kareler bosluksuz numaralidir; ilk eksik dosyada dur.
    int n = 0;
    while (fs::exists(fs::path(seq_dir) / frameFilename(n))) ++n;
    return n;
}

std::map<int, Detections> readDetectionsCsv(const std::string& path) {
    std::ifstream is(path);
    if (!is) throw std::runtime_error("acilamadi: " + path);
    std::map<int, Detections> out;
    std::string line;
    std::getline(is, line);  // baslik
    while (std::getline(is, line)) {
        if (line.empty()) continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream ls(line);
        int frame = 0;
        Detection d;
        if (!(ls >> frame >> d.box.x >> d.box.y >> d.box.w >> d.box.h >> d.score))
            throw std::runtime_error("bozuk satir: " + path);
        out[frame].push_back(d);
    }
    return out;
}

std::map<int, Affine2D> readMotionCsv(const std::string& path) {
    std::ifstream is(path);
    if (!is) throw std::runtime_error("acilamadi: " + path);
    std::map<int, Affine2D> out;
    std::string line;
    std::getline(is, line);
    while (std::getline(is, line)) {
        if (line.empty()) continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream ls(line);
        int frame = 0, valid = 0;
        Affine2D A;
        if (!(ls >> frame >> valid >> A.a >> A.b >> A.tx >> A.c >> A.d >> A.ty))
            throw std::runtime_error("bozuk satir: " + path);
        if (valid) out[frame] = A;
    }
    return out;
}

DetectionCsvWriter::DetectionCsvWriter(const std::string& path) : os_(path) {
    if (!os_) throw std::runtime_error("yazilamadi: " + path);
    os_ << "frame,x,y,w,h,score\n";
}

void DetectionCsvWriter::write(int frame, const Detections& dets) {
    char buf[128];
    for (const auto& d : dets) {
        std::snprintf(buf, sizeof(buf), "%d,%.2f,%.2f,%.2f,%.2f,%.4f\n", frame, d.box.x, d.box.y, d.box.w, d.box.h,
                      d.score);
        os_ << buf;
    }
}

TrackCsvWriter::TrackCsvWriter(const std::string& path) : os_(path) {
    if (!os_) throw std::runtime_error("yazilamadi: " + path);
    os_ << "frame,valid,measured,track_id,x,y,w,h,score,az_deg,el_deg\n";
}

void TrackCsvWriter::write(int frame, const std::optional<TargetOutput>& t, const AngleError& a) {
    char buf[192];
    if (t) {
        std::snprintf(buf, sizeof(buf), "%d,1,%d,%d,%.2f,%.2f,%.2f,%.2f,%.4f,%.4f,%.4f\n", frame, t->measured ? 1 : 0,
                      t->track_id, t->box.x, t->box.y, t->box.w, t->box.h, t->score, a.az * kRad2Deg, a.el * kRad2Deg);
    } else {
        std::snprintf(buf, sizeof(buf), "%d,0,0,-1,,,,,,,\n", frame);
    }
    os_ << buf;
}

}  // namespace iha
