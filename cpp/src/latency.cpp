#include "iha/latency.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>

namespace iha {

double percentile(std::vector<double> v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double pos = q / 100.0 * static_cast<double>(v.size() - 1);
    const auto lo = static_cast<std::size_t>(std::floor(pos));
    const std::size_t hi = std::min(lo + 1, v.size() - 1);
    const double f = pos - static_cast<double>(lo);
    return v[lo] * (1.0 - f) + v[hi] * f;
}

void LatencyStats::add(const std::string& stage, double ms) {
    auto it = samples_.find(stage);
    if (it == samples_.end()) {
        order_.push_back(stage);
        it = samples_.emplace(stage, std::vector<double>{}).first;
    }
    it->second.push_back(ms);
}

std::vector<StageSummary> LatencyStats::summary() const {
    std::vector<StageSummary> out;
    for (const auto& name : order_) {
        const auto& v = samples_.at(name);
        StageSummary s;
        s.name = name;
        s.n = v.size();
        s.mean_ms = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
        s.p50_ms = percentile(v, 50);
        s.p90_ms = percentile(v, 90);
        s.p99_ms = percentile(v, 99);
        s.max_ms = *std::max_element(v.begin(), v.end());
        out.push_back(s);
    }
    return out;
}

std::string LatencyStats::toJson() const {
    std::ostringstream os;
    os.precision(4);
    os << std::fixed << "{\n";
    const auto s = summary();
    for (std::size_t i = 0; i < s.size(); ++i) {
        os << "  \"" << s[i].name << "\": {\"n\": " << s[i].n << ", \"mean_ms\": " << s[i].mean_ms
           << ", \"p50_ms\": " << s[i].p50_ms << ", \"p90_ms\": " << s[i].p90_ms << ", \"p99_ms\": " << s[i].p99_ms
           << ", \"max_ms\": " << s[i].max_ms << "}" << (i + 1 < s.size() ? "," : "") << "\n";
    }
    os << "}\n";
    return os.str();
}

}  // namespace iha
