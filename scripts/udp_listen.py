"""Gimbal tarafini taklit eden alici: aci hatasi mesajlarini (38 bayt, CRC-16/CCITT) cozer.

  python3 scripts/udp_listen.py --port 5600 --count 300 --json results/udp_rx.json
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import struct


def crc16_ccitt(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def decode(b: bytes) -> dict | None:
    if len(b) != 38 or b[0] != 0xA5 or b[1] != 0x5A or b[2] != 1:
        return None
    if struct.unpack_from("<H", b, 36)[0] != crc16_ccitt(b[:36]):
        return None
    seq, stamp, tid, az, el, score, lat = struct.unpack_from("<IQifffI", b, 4)
    return {"valid": bool(b[3] & 1), "measured": bool(b[3] & 2), "seq": seq, "stamp_us": stamp, "track_id": tid,
            "az_deg": math.degrees(az), "el_deg": math.degrees(el), "score": score, "latency_us": lat}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5600)
    ap.add_argument("--count", type=int, default=300)
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", a.port))
    s.settimeout(a.timeout)
    msgs, bad = [], 0
    try:
        while len(msgs) < a.count:
            m = decode(s.recv(2048))
            if m is None:
                bad += 1
            else:
                msgs.append(m)
    except socket.timeout:
        pass
    lat = sorted(m["latency_us"] for m in msgs)
    seqs = [m["seq"] for m in msgs]
    rep = {"received": len(msgs), "rejected": bad, "valid_target": sum(m["valid"] for m in msgs),
           "seq_gaps": sum(1 for x, y in zip(seqs, seqs[1:]) if y != x + 1),
           "latency_us_p50": lat[len(lat) // 2] if lat else None, "latency_us_max": lat[-1] if lat else None,
           "last": msgs[-1] if msgs else None}
    print(json.dumps(rep, indent=2))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rep, f, indent=2)


if __name__ == "__main__":
    main()
