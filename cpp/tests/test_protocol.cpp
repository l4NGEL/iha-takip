#include <gtest/gtest.h>

#include <cstring>

#include "iha/protocol.hpp"
#include "iha/udp.hpp"

TEST(Protocol, CrcKnownVector) {
    const char* s = "123456789";
    EXPECT_EQ(iha::crc16Ccitt(reinterpret_cast<const uint8_t*>(s), 9), 0x29B1);
}

TEST(Protocol, EncodeDecodeRoundTrip) {
    iha::AngleErrorMsg m;
    m.valid = true;
    m.measured = false;
    m.seq = 0xDEADBEEF;
    m.stamp_us = 1234567890123ULL;
    m.track_id = -7;
    m.az = -0.0123f;
    m.el = 0.25f;
    m.score = 0.875f;
    m.latency_us = 4321;
    const auto b = iha::encode(m);
    ASSERT_EQ(b.size(), 38u);
    EXPECT_EQ(b[0], 0xA5);
    EXPECT_EQ(b[4], 0xEF);  // kucuk endian
    const auto d = iha::decode(b.data(), b.size());
    ASSERT_TRUE(d.has_value());
    EXPECT_TRUE(d->valid);
    EXPECT_FALSE(d->measured);
    EXPECT_EQ(d->seq, m.seq);
    EXPECT_EQ(d->stamp_us, m.stamp_us);
    EXPECT_EQ(d->track_id, m.track_id);
    EXPECT_FLOAT_EQ(d->az, m.az);
    EXPECT_FLOAT_EQ(d->el, m.el);
    EXPECT_FLOAT_EQ(d->score, m.score);
    EXPECT_EQ(d->latency_us, m.latency_us);
}

TEST(Protocol, RejectsCorruptionWrongLengthAndMagic) {
    auto b = iha::encode({});
    auto bad = b;
    bad[21] ^= 0x01;
    EXPECT_FALSE(iha::decode(bad.data(), bad.size()).has_value());
    EXPECT_FALSE(iha::decode(b.data(), b.size() - 1).has_value());
    bad = b;
    bad[1] = 0x00;
    EXPECT_FALSE(iha::decode(bad.data(), bad.size()).has_value());
}

TEST(Udp, LoopbackDeliversDecodableMessage) {
    iha::UdpReceiver rx(0);
    iha::UdpSender tx("127.0.0.1", rx.port());
    iha::AngleErrorMsg m;
    m.valid = true;
    m.seq = 42;
    m.az = 0.01f;
    const auto b = iha::encode(m);
    ASSERT_TRUE(tx.send(b.data(), b.size()));
    const auto got = rx.receive(1000);
    ASSERT_TRUE(got.has_value());
    const auto d = iha::decode(got->data(), got->size());
    ASSERT_TRUE(d.has_value());
    EXPECT_EQ(d->seq, 42u);
    EXPECT_FLOAT_EQ(d->az, 0.01f);
}

TEST(Udp, RejectsBadAddress) { EXPECT_THROW(iha::UdpSender("not-an-ip", 1), std::runtime_error); }
