#pragma once

#include <optional>
#include <vector>

#include "iha/geometry.hpp"
#include "iha/kalman.hpp"

namespace iha {

struct TrackerParams {
    // ByteTrack iki asamali eslestirme: once yuksek skorlu tespitler, sonra eslesmeyen
    // takiplere dusuk skorlu tespitler (ortulme/bulanikliktaki hedefi kaybetmemek icin).
    float high_thr = 0.4f;
    float low_thr = 0.1f;
    float new_track_thr = 0.5f;
    // Maliyet = 1 - DIoU (0..2). Esik ustu cift atanmaz.
    float max_cost_high = 1.5f;
    float max_cost_low = 1.3f;
    // Kalman ongorusune gore merkezin Mahalanobis^2 kapisi (2 sd; 13.8 ~ %99.9).
    float gate_maha2 = 13.8f;
    int min_hits = 2;        // bu kadar eslesmeden sonra takip onaylanir
    int max_lost = 30;       // eslesmeden bu kadar kare gecerse takip silinir
    int coast_output = 5;    // hedef kilidi eslesmesiz bu kadar kare ongoruyle surer
    float score_ema = 0.8f;  // takip skorunun ussel ortalamasi
    // Kilitli takip bu karede olculmediyse (ongoruyle suruyorsa) ve olculmus onayli bir
    // takip varsa kilit ona gecer. Kapaliyken kilit coast_output kare boyunca korunur;
    // kilit bir parazite takildiysa gercek hedef o sure boyunca kacirilir.
    bool lock_yields_to_measured = true;
    // Kilitli takibin skor EMA'si bunun altindaysa hedef disari verilmez (kilit korunur).
    // Hedefsiz karelerde yanlis cikisi azaltmak icin; 0 = kapi yok.
    float output_min_score = 0.f;
    KalmanParams kalman;
};

enum class TrackState { Tentative, Confirmed, Lost };

struct Track {
    int id = 0;
    TrackState state = TrackState::Tentative;
    KalmanBoxFilter kf;
    Box last_det;          // son eslesen tespitin kutusu
    float score = 0.f;     // skor EMA
    int hits = 0;
    int age = 0;
    int since_update = 0;  // son eslesmeden beri gecen kare

    Track(int id_, const Detection& d, const KalmanParams& kp) : id(id_), kf(d.box, kp), last_det(d.box), score(d.score), hits(1) {}
};

// Tek hedefe kilitlenen takipcinin kare cikisi.
struct TargetOutput {
    int track_id = -1;
    Box box;
    float score = 0.f;
    bool measured = false;  // bu karede tespitle guncellendi mi (yoksa ongoru)
    float vx = 0.f, vy = 0.f;  // piksel/kare, Kalman hizi
};

class ByteTracker {
public:
    explicit ByteTracker(const TrackerParams& p = {}) : p_(p) {}

    // Bir karelik tespitlerle guncelle; cok hedefli takip listesi icin tracks().
    void update(const Detections& dets);

    // Kamera hareketi telafisi (BoT-SORT'taki CMC gibi): bir onceki kareden bu kareye
    // sahne goruntude (dx, dy) piksel kaydi; update()'ten once cagrilir. Kaynak gimbal
    // enkoderi/IMU (dx = -fx * d_yaw, dy = +fy * d_pitch) ya da goruntuden kuresel hareket
    // kestirimi olabilir. Telafiyle Kalman hizi hedefin dunya acisal hizi olur (gimbal_sim'deki
    // "lead" kontrolcusu bunu kullanir). Telafisiz, gimbal donerken goruntudeki hiz tersine
    // doner ve takip kopabilir; gimbal_sim'de bu tek basina donguyu bozmadi, yapiskan kilitle
    // birlesince bozdu (bkz. lock_yields_to_measured).
    void compensateCameraMotion(float dx, float dy);
    // Genel hal: onceki kare -> simdiki kare afin eslemesi (goruntuden kuresel hareket kestirimi).
    void compensateCameraMotion(const Affine2D& A);
    const std::vector<Track>& tracks() const { return tracks_; }

    // Kilitli hedef: kilitli takip yasiyor ve en fazla coast_output karedir
    // eslesmesizse onu surdurur; yoksa onayli takipler arasindan bu karede
    // eslesmis ve skoru en yuksek olana gecer.
    std::optional<TargetOutput> target() const { return target_; }

    void reset();

private:
    void selectTarget();

    TrackerParams p_;
    std::vector<Track> tracks_;
    int next_id_ = 1;
    int locked_id_ = -1;
    std::optional<TargetOutput> target_;
};

}  // namespace iha
