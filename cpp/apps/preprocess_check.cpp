// CUDA on isleme ciktisini ham float32 dosyaya yazar; scripts/check_parity.py bunu
// Ultralytics LetterBox + numpy referansiyla karsilastirir.
//
//   preprocess_check --engine m.engine --image a.jpg --out a_pre.bin

#include <fstream>
#include <iostream>

#include <opencv2/imgcodecs.hpp>

#include "iha/args.hpp"
#include "iha/trt_detector.hpp"

int main(int argc, char** argv) try {
    const iha::Args a(argc, argv);
    iha::TrtDetector det(a.required("engine"));
    const cv::Mat img = cv::imread(a.required("image"), cv::IMREAD_COLOR);
    if (img.empty()) throw std::runtime_error("goruntu okunamadi");
    const auto v = det.preprocessOnly(img);
    std::ofstream(a.required("out"), std::ios::binary)
        .write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)));
    std::cout << "yazildi " << v.size() << " float (" << det.inputHeight() << "x" << det.inputWidth() << ")\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "hata: " << e.what() << "\n";
    return 1;
}
