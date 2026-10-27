#pragma once

#include <fstream>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "iha/geometry.hpp"
#include "iha/gimbal.hpp"
#include "iha/tracker.hpp"

namespace iha {

// Anti-UAV dizi klasoru: 000001.jpg, 000002.jpg, ... (0 tabanli kare no -> dosya adi).
std::string frameFilename(int index);
std::vector<std::string> listSequenceDirs(const std::string& split_dir);
int countFrames(const std::string& seq_dir);

// Tespit dokumu: "frame,x,y,w,h,score" (frame 0 tabanli). Hedefsiz kareler satir icermez.
std::map<int, Detections> readDetectionsCsv(const std::string& path);

class DetectionCsvWriter {
public:
    explicit DetectionCsvWriter(const std::string& path);
    void write(int frame, const Detections& dets);

private:
    std::ofstream os_;
};

// Takip ciktisi: kare basina tek satir; hedef yoksa valid=0 ve kutu bos.
class TrackCsvWriter {
public:
    explicit TrackCsvWriter(const std::string& path);
    void write(int frame, const std::optional<TargetOutput>& t, const AngleError& a);

private:
    std::ofstream os_;
};

}  // namespace iha
