#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace iha {

// Bloklamayan UDP gonderici (POSIX). Gimbal/kontrol bilgisayarina aci hatasi yollar.
class UdpSender {
public:
    UdpSender(const std::string& host, uint16_t port);
    ~UdpSender();
    UdpSender(const UdpSender&) = delete;
    UdpSender& operator=(const UdpSender&) = delete;

    bool send(const uint8_t* data, std::size_t len);

private:
    int fd_ = -1;
    std::vector<uint8_t> addr_;  // sockaddr_in
};

// Test ve ornek alici icin: bagli porttan zaman asimli tek datagram okur.
class UdpReceiver {
public:
    explicit UdpReceiver(uint16_t port);  // 0: isletim sistemi secer
    ~UdpReceiver();
    UdpReceiver(const UdpReceiver&) = delete;
    UdpReceiver& operator=(const UdpReceiver&) = delete;

    uint16_t port() const { return port_; }
    std::optional<std::vector<uint8_t>> receive(int timeout_ms);

private:
    int fd_ = -1;
    uint16_t port_ = 0;
};

}  // namespace iha
