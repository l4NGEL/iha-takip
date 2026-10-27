#include "iha/udp.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <stdexcept>

namespace iha {

UdpSender::UdpSender(const std::string& host, uint16_t port) : addr_(sizeof(sockaddr_in)) {
    fd_ = ::socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK, 0);
    if (fd_ < 0) throw std::runtime_error("UDP soketi acilamadi");
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(port);
    if (::inet_pton(AF_INET, host.c_str(), &a.sin_addr) != 1) {
        ::close(fd_);
        throw std::runtime_error("gecersiz IPv4 adresi: " + host);
    }
    std::memcpy(addr_.data(), &a, sizeof(a));
}

UdpSender::~UdpSender() {
    if (fd_ >= 0) ::close(fd_);
}

bool UdpSender::send(const uint8_t* data, std::size_t len) {
    // Bloklamayan: alici yavassa paket duser, goruntu hatti asla beklemez.
    const auto n = ::sendto(fd_, data, len, 0, reinterpret_cast<const sockaddr*>(addr_.data()),
                            static_cast<socklen_t>(addr_.size()));
    return n == static_cast<ssize_t>(len);
}

UdpReceiver::UdpReceiver(uint16_t port) {
    fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (fd_ < 0) throw std::runtime_error("UDP soketi acilamadi");
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(port);
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::bind(fd_, reinterpret_cast<sockaddr*>(&a), sizeof(a)) != 0) {
        ::close(fd_);
        throw std::runtime_error("UDP bind basarisiz");
    }
    socklen_t len = sizeof(a);
    ::getsockname(fd_, reinterpret_cast<sockaddr*>(&a), &len);
    port_ = ntohs(a.sin_port);
}

UdpReceiver::~UdpReceiver() {
    if (fd_ >= 0) ::close(fd_);
}

std::optional<std::vector<uint8_t>> UdpReceiver::receive(int timeout_ms) {
    pollfd p{fd_, POLLIN, 0};
    if (::poll(&p, 1, timeout_ms) <= 0) return std::nullopt;
    std::vector<uint8_t> buf(2048);
    const auto n = ::recv(fd_, buf.data(), buf.size(), 0);
    if (n < 0) return std::nullopt;
    buf.resize(static_cast<std::size_t>(n));
    return buf;
}

}  // namespace iha
