// Kaydedilmis tespitler uzerinde takip (TensorRT gerektirmez).
// Canli hatla (iha_track) ayni ByteTracker ve ayni bayraklari kullanir; takip ayarlari
// val dizilerinde bununla secilir, test bir kez kosulur.
//
//   iha_track_offline --dets-dir results/trk/dets_val --root data/antiuav410/val
//       --out-dir results/trk/val_bytetrack [--mode bytetrack|top1] [--top1-thr 0.415] [takip bayraklari]
//
// --mode top1: takip yok, her karede esigi gecen en yuksek skorlu tespit (kiyas tabani).

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>

#include "iha/args.hpp"
#include "iha/gimbal.hpp"
#include "iha/pipeline_config.hpp"
#include "iha/sequence_io.hpp"
#include "iha/tracker.hpp"

namespace fs = std::filesystem;
using namespace iha;

int main(int argc, char** argv) try {
    const Args a(argc, argv);
    const std::string dets_dir = a.required("dets-dir");
    const std::string root = a.required("root");
    const std::string out_dir = a.required("out-dir");
    const std::string mode = a.str("mode", "bytetrack");
    const float top1_thr = a.f("top1-thr", 0.415f);
    const TrackerParams tp = trackerParamsFromArgs(a);
    const CameraModel cam = CameraModel::fromHfov(640, 512, a.f("hfov", 30.f));
    if (mode != "bytetrack" && mode != "top1") throw std::invalid_argument("--mode bytetrack|top1");

    fs::create_directories(out_dir);
    const auto t0 = std::chrono::steady_clock::now();
    long frames_total = 0;
    int n_seq = 0;
    for (const auto& seq_dir : listSequenceDirs(root)) {
        const std::string name = fs::path(seq_dir).filename().string();
        const fs::path det_path = fs::path(dets_dir) / (name + ".csv");
        if (!fs::exists(det_path)) throw std::runtime_error("tespit dosyasi yok: " + det_path.string());
        const auto dets = readDetectionsCsv(det_path.string());
        const int n = countFrames(seq_dir);

        ByteTracker tracker(tp);
        TrackCsvWriter out((fs::path(out_dir) / (name + ".csv")).string());
        static const Detections kEmpty;
        for (int f = 0; f < n; ++f) {
            auto it = dets.find(f);
            const Detections& d = it == dets.end() ? kEmpty : it->second;
            std::optional<TargetOutput> tgt;
            if (mode == "top1") {
                const Detection* best = nullptr;
                for (const auto& x : d)
                    if (x.score >= top1_thr && (!best || x.score > best->score)) best = &x;
                if (best) tgt = TargetOutput{0, best->box, best->score, true, 0.f, 0.f};
            } else {
                tracker.update(d);
                tgt = tracker.target();
            }
            const AngleError ang = tgt ? pixelToAngle(cam, tgt->box.cx(), tgt->box.cy()) : AngleError{};
            out.write(f, tgt, ang);
        }
        frames_total += n;
        ++n_seq;
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("%d dizi, %ld kare, %.2f s (%.0f kare/sn, yalniz takip)\n", n_seq, frames_total, s,
                static_cast<double>(frames_total) / s);
    return 0;
} catch (const std::exception& e) {
    std::cerr << "hata: " << e.what() << "\n";
    return 1;
}
