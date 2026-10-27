#include "iha/protocol.hpp"

#include <cstring>

namespace iha {

namespace {

template <typename T>
void put(uint8_t* p, T v) {
    static_assert(sizeof(T) <= 8);
    uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof(T));
    for (std::size_t i = 0; i < sizeof(T); ++i) p[i] = static_cast<uint8_t>(bits >> (8 * i));
}

template <typename T>
T get(const uint8_t* p) {
    uint64_t bits = 0;
    for (std::size_t i = 0; i < sizeof(T); ++i) bits |= static_cast<uint64_t>(p[i]) << (8 * i);
    T v;
    std::memcpy(&v, &bits, sizeof(T));
    return v;
}

}  // namespace

uint16_t crc16Ccitt(const uint8_t* data, std::size_t len) {
    uint16_t crc = 0xFFFF;
    for (std::size_t i = 0; i < len; ++i) {
        crc = static_cast<uint16_t>(crc ^ (static_cast<uint16_t>(data[i]) << 8));
        for (int b = 0; b < 8; ++b)
            crc = (crc & 0x8000) ? static_cast<uint16_t>((crc << 1) ^ 0x1021) : static_cast<uint16_t>(crc << 1);
    }
    return crc;
}

std::array<uint8_t, AngleErrorMsg::kSize> encode(const AngleErrorMsg& m) {
    std::array<uint8_t, AngleErrorMsg::kSize> b{};
    b[0] = 0xA5;
    b[1] = 0x5A;
    b[2] = AngleErrorMsg::kVersion;
    b[3] = static_cast<uint8_t>((m.valid ? 1 : 0) | (m.measured ? 2 : 0));
    put(&b[4], m.seq);
    put(&b[8], m.stamp_us);
    put(&b[16], m.track_id);
    put(&b[20], m.az);
    put(&b[24], m.el);
    put(&b[28], m.score);
    put(&b[32], m.latency_us);
    put(&b[36], crc16Ccitt(b.data(), 36));
    return b;
}

std::optional<AngleErrorMsg> decode(const uint8_t* data, std::size_t len) {
    if (len != AngleErrorMsg::kSize || data[0] != 0xA5 || data[1] != 0x5A || data[2] != AngleErrorMsg::kVersion)
        return std::nullopt;
    if (get<uint16_t>(&data[36]) != crc16Ccitt(data, 36)) return std::nullopt;
    AngleErrorMsg m;
    m.valid = data[3] & 1;
    m.measured = data[3] & 2;
    m.seq = get<uint32_t>(&data[4]);
    m.stamp_us = get<uint64_t>(&data[8]);
    m.track_id = get<int32_t>(&data[16]);
    m.az = get<float>(&data[20]);
    m.el = get<float>(&data[24]);
    m.score = get<float>(&data[28]);
    m.latency_us = get<uint32_t>(&data[32]);
    return m;
}

}  // namespace iha
