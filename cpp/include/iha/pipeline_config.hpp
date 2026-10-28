#pragma once

#include "iha/args.hpp"
#include "iha/tracker.hpp"

namespace iha {

// Canli ve cevrimdisi uygulamalarin ayni takip ayarlarini ayni bayraklardan okumasi icin.
inline TrackerParams trackerParamsFromArgs(const Args& a) {
    TrackerParams p;
    p.high_thr = a.f("high-thr", p.high_thr);
    p.low_thr = a.f("low-thr", p.low_thr);
    p.new_track_thr = a.f("new-thr", p.new_track_thr);
    p.max_cost_high = a.f("max-cost-high", p.max_cost_high);
    p.max_cost_low = a.f("max-cost-low", p.max_cost_low);
    p.gate_maha2 = a.f("gate", p.gate_maha2);
    p.min_hits = a.i("min-hits", p.min_hits);
    p.max_lost = a.i("max-lost", p.max_lost);
    p.coast_output = a.i("coast", p.coast_output);
    p.lock_yields_to_measured = a.i("lock-yield", p.lock_yields_to_measured ? 1 : 0) != 0;
    p.output_min_score = a.f("out-score", p.output_min_score);
    return p;
}

}  // namespace iha
