#include "iha/tracker.hpp"

#include <algorithm>

#include "iha/hungarian.hpp"

namespace iha {

namespace {

// tracks[ti] x dets[di] maliyet matrisi; kapi disi ciftler max_cost alir.
std::vector<std::vector<float>> costMatrix(const std::vector<Track>& tracks, const std::vector<int>& ti,
                                           const Detections& dets, const std::vector<int>& di, float gate,
                                           float max_cost) {
    std::vector<std::vector<float>> c(ti.size(), std::vector<float>(di.size(), max_cost));
    for (size_t r = 0; r < ti.size(); ++r) {
        const Track& t = tracks[ti[r]];
        const Box pred = t.kf.box();
        for (size_t k = 0; k < di.size(); ++k) {
            const Box& b = dets[di[k]].box;
            if (t.kf.centerMahalanobis2(b) > gate) continue;
            c[r][k] = 1.f - diou(pred, b);
        }
    }
    return c;
}

}  // namespace

void ByteTracker::reset() {
    tracks_.clear();
    next_id_ = 1;
    locked_id_ = -1;
    target_.reset();
}

void ByteTracker::compensateCameraMotion(float dx, float dy) {
    for (auto& t : tracks_) {
        t.kf.shift(dx, dy);
        t.last_det.x += dx;
        t.last_det.y += dy;
    }
}

void ByteTracker::update(const Detections& dets) {
    for (auto& t : tracks_) {
        t.kf.predict();
        ++t.age;
        ++t.since_update;
    }

    std::vector<int> high, low;
    for (int i = 0; i < static_cast<int>(dets.size()); ++i) {
        if (dets[i].score >= p_.high_thr)
            high.push_back(i);
        else if (dets[i].score >= p_.low_thr)
            low.push_back(i);
    }

    std::vector<char> det_used(dets.size(), false);
    auto apply = [&](Track& t, const Detection& d) {
        t.kf.update(d.box);
        t.last_det = d.box;
        t.score = p_.score_ema * t.score + (1.f - p_.score_ema) * d.score;
        ++t.hits;
        t.since_update = 0;
        if (t.state == TrackState::Lost || (t.state == TrackState::Tentative && t.hits >= p_.min_hits))
            t.state = TrackState::Confirmed;
    };
    auto match = [&](const std::vector<int>& ti, const std::vector<int>& di, float max_cost) {
        std::vector<int> unmatched_t;
        std::vector<char> t_used(ti.size(), false);
        if (!ti.empty() && !di.empty()) {
            const auto c = costMatrix(tracks_, ti, dets, di, p_.gate_maha2, max_cost);
            for (const auto& [r, k] : solveAssignment(c, max_cost)) {
                apply(tracks_[ti[r]], dets[di[k]]);
                det_used[di[k]] = true;
                t_used[r] = true;
            }
        }
        for (size_t r = 0; r < ti.size(); ++r)
            if (!t_used[r]) unmatched_t.push_back(ti[r]);
        return unmatched_t;
    };
    auto remaining = [&](const std::vector<int>& di) {
        std::vector<int> out;
        for (int i : di)
            if (!det_used[i]) out.push_back(i);
        return out;
    };

    // 1) Onayli + kayip takipler <- yuksek skorlu tespitler
    std::vector<int> pool, tentative;
    for (int i = 0; i < static_cast<int>(tracks_.size()); ++i)
        (tracks_[i].state == TrackState::Tentative ? tentative : pool).push_back(i);
    const auto unmatched_pool = match(pool, high, p_.max_cost_high);

    // 2) Bir onceki karede eslesmis (kayip olmayan) onayli takipler <- dusuk skorlu tespitler
    std::vector<int> recent;
    for (int i : unmatched_pool)
        if (tracks_[i].state == TrackState::Confirmed) recent.push_back(i);
    match(recent, low, p_.max_cost_low);

    // 3) Aday takipler <- kalan yuksek skorlu tespitler
    match(tentative, remaining(high), p_.max_cost_high);

    // Durum gecisleri ve silme
    for (auto& t : tracks_) {
        if (t.since_update == 0) continue;
        if (t.state == TrackState::Confirmed) t.state = TrackState::Lost;
    }
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                 [&](const Track& t) {
                                     return (t.state == TrackState::Tentative && t.since_update > 0) ||
                                            t.since_update > p_.max_lost;
                                 }),
                  tracks_.end());

    // 4) Yeni takipler
    for (int i : remaining(high)) {
        if (dets[i].score < p_.new_track_thr) continue;
        Track t(next_id_++, dets[i], p_.kalman);
        if (p_.min_hits <= 1) t.state = TrackState::Confirmed;
        tracks_.push_back(std::move(t));
    }

    selectTarget();
}

void ByteTracker::selectTarget() {
    target_.reset();
    const Track* best = nullptr;
    for (const auto& t : tracks_) {
        if (t.id == locked_id_ && t.state != TrackState::Tentative && t.since_update <= p_.coast_output) {
            best = &t;
            break;
        }
    }
    if (best && best->since_update > 0 && p_.lock_yields_to_measured) {
        for (const auto& t : tracks_)
            if (t.state == TrackState::Confirmed && t.since_update == 0) {
                best = nullptr;
                break;
            }
    }
    if (!best) {
        for (const auto& t : tracks_) {
            if (t.state != TrackState::Confirmed || t.since_update != 0) continue;
            if (!best || t.score > best->score) best = &t;
        }
    }
    if (!best) {
        locked_id_ = -1;
        return;
    }
    locked_id_ = best->id;
    TargetOutput o;
    o.track_id = best->id;
    o.measured = best->since_update == 0;
    // Olculen karede tespit kutusu (Kalman boyut gecikmesi kucuk hedefte IoU'yu dusuruyor),
    // ongoru karesinde Kalman kutusu.
    o.box = o.measured ? best->last_det : best->kf.box();
    o.score = best->score;
    o.vx = best->kf.vx();
    o.vy = best->kf.vy();
    target_ = o;
}

}  // namespace iha
