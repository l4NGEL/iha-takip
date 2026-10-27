#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace iha {

// Gimbal/kontrol sistemine giden aci hatasi mesaji. Sabit 38 bayt, kucuk endian:
//   0  u8[2]  sihirli 0xA5 0x5A
//   2  u8     surum (1)
//   3  u8     bayraklar: bit0 hedef gecerli, bit1 bu karede olculdu (ongoru degil)
//   4  u32    sira no
//   8  u64    goruntu zaman damgasi (us)
//  16  i32    takip no (-1: hedef yok)
//  20  f32    azimut hatasi (rad, + sag)
//  24  f32    elevasyon hatasi (rad, + yukari)
//  28  f32    takip skoru
//  32  u32    goruntuden mesaja hat gecikmesi (us)
//  36  u16    CRC-16/CCITT-FALSE (bayt 0..35)
struct AngleErrorMsg {
    static constexpr std::size_t kSize = 38;
    static constexpr uint8_t kVersion = 1;

    bool valid = false;
    bool measured = false;
    uint32_t seq = 0;
    uint64_t stamp_us = 0;
    int32_t track_id = -1;
    float az = 0.f, el = 0.f;
    float score = 0.f;
    uint32_t latency_us = 0;
};

uint16_t crc16Ccitt(const uint8_t* data, std::size_t len);

std::array<uint8_t, AngleErrorMsg::kSize> encode(const AngleErrorMsg& m);
// Sihirli bayt, surum ya da CRC tutmazsa nullopt.
std::optional<AngleErrorMsg> decode(const uint8_t* data, std::size_t len);

}  // namespace iha
