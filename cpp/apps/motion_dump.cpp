// Dizi basina kare arasi kamera hareketini (benzerlik donusumu) CSV'ye yazar; cevrimdisi
// takip (iha_track_offline --motion-dir) bunu kullanarak CMC'yi val'de ucuzca ayarlar.
//
//   motion_dump --root /data/antiuav410/val --out-dir results/trk/motion_val
//       [--dets-dir results/trk/dets_val_fp16 --mask-thr 0.3] [--downscale 0.5]
//
// Satir f: kare f-1 -> kare f eslemesi. --dets-dir verilirse onceki karenin skoru mask-thr
// ustundeki tespitleri kose aramasindan maskelenir (hedefin kendi hareketi arka plani bozmasin).

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>

#include <opencv2/imgcodecs.hpp>

#include "iha/args.hpp"
#include "iha/global_motion.hpp"
#include "iha/latency.hpp"
#include "iha/sequence_io.hpp"

namespace fs = std::filesystem;
using namespace iha;

int main(int argc, char** argv) try {
    const Args a(argc, argv);
    const std::string root = a.required("root"), out_dir = a.required("out-dir");
    const std::string dets_dir = a.str("dets-dir", "");
    const float mask_thr = a.f("mask-thr", 0.3f);
    GlobalMotionParams gp;
    gp.downscale = a.f("downscale", 1.f);
    gp.min_inliers = a.i("min-inliers", gp.min_inliers);
    fs::create_directories(out_dir);
    LatencyStats st;
    long frames = 0, valid = 0;
    for (const auto& seq_dir : listSequenceDirs(root)) {
        const std::string name = fs::path(seq_dir).filename().string();
        std::map<int, Detections> dets;
        if (!dets_dir.empty()) dets = readDetectionsCsv((fs::path(dets_dir) / (name + ".csv")).string());
        GlobalMotionEstimator est(gp);
        std::ofstream os(fs::path(out_dir) / (name + ".csv"));
        os << "frame,valid,a,b,tx,c,d,ty,inliers,tracked\n";
        const int n = countFrames(seq_dir);
        for (int f = 0; f < n; ++f) {
            const cv::Mat g = cv::imread((fs::path(seq_dir) / frameFilename(f)).string(), cv::IMREAD_GRAYSCALE);
            std::vector<Box> excl;
            if (auto it = dets.find(f - 1); it != dets.end())
                for (const auto& d : it->second)
                    if (d.score >= mask_thr) excl.push_back(d.box);
            Stopwatch sw;
            const auto m = est.update(g, excl);
            st.add("kuresel_hareket", sw.lap());
            char buf[256];
            if (m) {
                std::snprintf(buf, sizeof(buf), "%d,1,%.6f,%.6f,%.3f,%.6f,%.6f,%.3f,%d,%d\n", f, m->A.a, m->A.b, m->A.tx,
                              m->A.c, m->A.d, m->A.ty, m->inliers, m->tracked);
                ++valid;
            } else {
                std::snprintf(buf, sizeof(buf), "%d,0,1,0,0,0,1,0,0,0\n", f);
            }
            os << buf;
            ++frames;
        }
        std::fprintf(stderr, "%s: %d kare\n", name.c_str(), n);
    }
    std::printf("%ld kare, gecerli kestirim %.3f\n%s", frames, static_cast<double>(valid) / static_cast<double>(frames),
                st.toJson().c_str());
    return 0;
} catch (const std::exception& e) {
    std::cerr << "hata: " << e.what() << "\n";
    return 1;
}
