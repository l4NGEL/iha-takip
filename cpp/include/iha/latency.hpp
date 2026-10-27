#pragma once

#include <chrono>
#include <map>
#include <string>
#include <vector>

namespace iha {

struct StageSummary {
    std::string name;
    std::size_t n = 0;
    double mean_ms = 0, p50_ms = 0, p90_ms = 0, p99_ms = 0, max_ms = 0;
};

// Asama basina sure ornekleri; ozet eklenme sirasini korur.
class LatencyStats {
public:
    void add(const std::string& stage, double ms);
    std::vector<StageSummary> summary() const;
    std::string toJson() const;

private:
    std::vector<std::string> order_;
    std::map<std::string, std::vector<double>> samples_;
};

class Stopwatch {
public:
    Stopwatch() : t0_(std::chrono::steady_clock::now()) {}
    // Son lap'ten (ya da baslangictan) beri gecen ms ve sayaci sifirlar.
    double lap() {
        const auto t = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t - t0_).count();
        t0_ = t;
        return ms;
    }

private:
    std::chrono::steady_clock::time_point t0_;
};

// Siralanmis olmayan orneklerde dogrusal enterpolasyonlu yuzdelik (numpy varsayilani).
double percentile(std::vector<double> v, double q);

}  // namespace iha
