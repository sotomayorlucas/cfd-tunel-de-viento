#include "png.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace cfd::png {
namespace {

// ---- CRC-32 (tabla constexpr) ----------------------------------------------------
constexpr std::array<u32, 256> make_crc_table() {
    std::array<u32, 256> t{};
    for (u32 n = 0; n < 256; ++n) {
        u32 c = n;
        for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
        t[n] = c;
    }
    return t;
}
constexpr auto k_crc = make_crc_table();

// ---- Huffman fijo (RFC 1951 §3.2.6), códigos ya invertidos para escritura LSB-first ----
constexpr u32 reverse_bits(u32 v, int n) {
    u32 r = 0;
    for (int i = 0; i < n; ++i) { r = (r << 1) | (v & 1); v >>= 1; }
    return r;
}
struct Code { u16 bits; u8 len; };
constexpr std::array<Code, 288> make_litlen() {
    std::array<Code, 288> t{};
    for (u32 s = 0; s < 288; ++s) {
        u32 c; int l;
        if (s < 144)      { c = 0x30 + s;          l = 8; }
        else if (s < 256) { c = 0x190 + (s - 144); l = 9; }
        else if (s < 280) { c = s - 256;           l = 7; }
        else              { c = 0xC0 + (s - 280);  l = 8; }
        t[s] = {static_cast<u16>(reverse_bits(c, l)), static_cast<u8>(l)};
    }
    return t;
}
constexpr auto k_litlen = make_litlen();

constexpr u16 k_len_base[29]  = {3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
constexpr u8  k_len_extra[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
constexpr u16 k_dist_base[30] = {1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769,
                                 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
constexpr u8  k_dist_extra[30] = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};

constexpr std::array<u8, 259> make_len_sym() {
    std::array<u8, 259> t{};
    for (int len = 3; len <= 258; ++len) {
        int s = 0;
        while (s + 1 < 29 && k_len_base[s + 1] <= len) ++s;
        t[len] = static_cast<u8>(s);
    }
    return t;
}
constexpr auto k_len_sym = make_len_sym();

// Truco de zlib: distancias ≤256 indexan directo, el resto por (d-1)>>7.
constexpr std::array<u8, 512> make_dist_sym() {
    std::array<u8, 512> t{};
    for (int c = 0; c < 30; ++c) {
        const int lo = k_dist_base[c] - 1, hi = lo + (1 << k_dist_extra[c]) - 1;
        for (int d = lo; d <= hi && d < 32768; ++d) {
            if (d < 256) t[d] = static_cast<u8>(c);
            else t[256 + (d >> 7)] = static_cast<u8>(c);
        }
    }
    return t;
}
constexpr auto k_dist_sym = make_dist_sym();

struct BitWriter {
    std::vector<u8>& out;
    u64 acc = 0;
    int nbits = 0;
    CFD_INLINE void put(u32 bits, int n) {
        acc |= static_cast<u64>(bits) << nbits;
        nbits += n;
        while (nbits >= 8) { out.push_back(static_cast<u8>(acc)); acc >>= 8; nbits -= 8; }
    }
    void flush() { if (nbits > 0) out.push_back(static_cast<u8>(acc)); acc = 0; nbits = 0; }
};

CFD_INLINE u32 hash3(const u8* p) {
    return ((static_cast<u32>(p[0]) << 16 | static_cast<u32>(p[1]) << 8 | p[2]) * 2654435761u) >> 17; // 15 bits
}

void put_be32(std::vector<u8>& v, u32 x) {
    v.push_back(static_cast<u8>(x >> 24)); v.push_back(static_cast<u8>(x >> 16));
    v.push_back(static_cast<u8>(x >> 8));  v.push_back(static_cast<u8>(x));
}
void put_chunk(std::vector<u8>& out, const char* type, const u8* data, usize n) {
    put_be32(out, static_cast<u32>(n));
    const usize start = out.size();
    out.insert(out.end(), type, type + 4);
    if (n) out.insert(out.end(), data, data + n);
    put_be32(out, crc32(out.data() + start, n + 4));
}

} // namespace

u32 crc32(const u8* data, usize n, u32 crc) {
    crc = ~crc;
    for (usize i = 0; i < n; ++i) crc = k_crc[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    return ~crc;
}

u32 adler32(const u8* data, usize n) {
    u32 a = 1, b = 0;
    while (n) {
        const usize blk = n < 5552 ? n : 5552;   // máximo sin desbordar antes del módulo
        for (usize i = 0; i < blk; ++i) { a += data[i]; b += a; }
        a %= 65521; b %= 65521;
        data += blk; n -= blk;
    }
    return (b << 16) | a;
}

std::vector<u8> zlib_compress(const u8* data, usize n) {
    std::vector<u8> out;
    out.reserve(n / 3 + 64);
    out.push_back(0x78); out.push_back(0x01);            // CMF/FLG: deflate, ventana 32K
    BitWriter bw{out};
    bw.put(1, 1);                                        // BFINAL
    bw.put(1, 2);                                        // BTYPE = 01 (Huffman fijo)
    constexpr u32 kWin = 32768, kMask = kWin - 1;
    std::vector<i32> head(1u << 15, -1), prev(kWin, -1);
    usize i = 0;
    while (i < n) {
        int best_len = 0, best_dist = 0;
        if (i + 3 <= n) {
            const u32 h = hash3(data + i);
            i32 cand = head[h];
            const int maxl = static_cast<int>(n - i < 258 ? n - i : 258);
            for (int chain = 24; cand >= 0 && i - static_cast<usize>(cand) <= kWin && chain > 0; --chain) {
                const u8* a = data + cand;
                const u8* b = data + i;
                if (a[best_len] == b[best_len]) {
                    int l = 0;
                    while (l < maxl && a[l] == b[l]) ++l;
                    if (l > best_len) { best_len = l; best_dist = static_cast<int>(i - static_cast<usize>(cand)); if (l == maxl) break; }
                }
                cand = prev[static_cast<u32>(cand) & kMask];
            }
            prev[i & kMask] = head[h];
            head[h] = static_cast<i32>(i);
        }
        if (best_len >= 3) {
            const int ls = k_len_sym[best_len];
            const Code c = k_litlen[257 + ls];
            bw.put(c.bits, c.len);
            if (k_len_extra[ls]) bw.put(static_cast<u32>(best_len - k_len_base[ls]), k_len_extra[ls]);
            const int d = best_dist - 1;
            const int ds = d < 256 ? k_dist_sym[d] : k_dist_sym[256 + (d >> 7)];
            bw.put(reverse_bits(static_cast<u32>(ds), 5), 5);
            if (k_dist_extra[ds]) bw.put(static_cast<u32>(best_dist - k_dist_base[ds]), k_dist_extra[ds]);
            for (int k = 1; k < best_len; ++k) {
                const usize j = i + static_cast<usize>(k);
                if (j + 3 <= n) { const u32 h = hash3(data + j); prev[j & kMask] = head[h]; head[h] = static_cast<i32>(j); }
            }
            i += static_cast<usize>(best_len);
        } else {
            const Code c = k_litlen[data[i]];
            bw.put(c.bits, c.len);
            ++i;
        }
    }
    const Code eob = k_litlen[256];
    bw.put(eob.bits, eob.len);
    bw.flush();
    put_be32(out, adler32(data, n));
    return out;
}

std::vector<u8> encode_argb(const u32* pixels, int w, int h, int stride, bool alpha) {
    const int bpp = alpha ? 4 : 3;
    const usize row_bytes = static_cast<usize>(w) * bpp;
    std::vector<u8> raw(static_cast<usize>(h) * (row_bytes + 1));
    std::vector<u8> cur(row_bytes), prv(row_bytes, 0), tmp(row_bytes), best(row_bytes);
    for (int y = 0; y < h; ++y) {
        const u32* src = pixels + static_cast<usize>(y) * stride;
        for (int x = 0; x < w; ++x) {
            const u32 p = src[x];
            u8* d = &cur[static_cast<usize>(x) * bpp];
            d[0] = static_cast<u8>(p >> 16); d[1] = static_cast<u8>(p >> 8); d[2] = static_cast<u8>(p);
            if (alpha) d[3] = static_cast<u8>(p >> 24);
        }
        // Filtro adaptativo: prueba los 5 y elige el de menor suma |byte con signo|.
        u64 best_score = ~0ull; u8 best_f = 0;
        for (u8 f = 0; f < 5; ++f) {
            u64 score = 0;
            for (usize i = 0; i < row_bytes; ++i) {
                const int a = i >= static_cast<usize>(bpp) ? cur[i - bpp] : 0;
                const int b = prv[i];
                const int c = i >= static_cast<usize>(bpp) ? prv[i - bpp] : 0;
                int pred = 0;
                switch (f) {
                    case 0: pred = 0; break;
                    case 1: pred = a; break;
                    case 2: pred = b; break;
                    case 3: pred = (a + b) >> 1; break;
                    default: {
                        const int p = a + b - c, pa = std::abs(p - a), pb = std::abs(p - b), pc = std::abs(p - c);
                        pred = (pa <= pb && pa <= pc) ? a : (pb <= pc ? b : c);
                    }
                }
                const u8 v = static_cast<u8>(cur[i] - pred);
                tmp[i] = v;
                score += static_cast<u64>(v < 128 ? v : 256 - v);
            }
            if (score < best_score) { best_score = score; best_f = f; best.swap(tmp); }
        }
        u8* dst = &raw[static_cast<usize>(y) * (row_bytes + 1)];
        dst[0] = best_f;
        std::memcpy(dst + 1, best.data(), row_bytes);
        prv.swap(cur);
    }
    std::vector<u8> out = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    u8 ihdr[13];
    const u32 W = static_cast<u32>(w), H = static_cast<u32>(h);
    ihdr[0] = static_cast<u8>(W >> 24); ihdr[1] = static_cast<u8>(W >> 16); ihdr[2] = static_cast<u8>(W >> 8); ihdr[3] = static_cast<u8>(W);
    ihdr[4] = static_cast<u8>(H >> 24); ihdr[5] = static_cast<u8>(H >> 16); ihdr[6] = static_cast<u8>(H >> 8); ihdr[7] = static_cast<u8>(H);
    ihdr[8] = 8; ihdr[9] = alpha ? 6 : 2; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
    put_chunk(out, "IHDR", ihdr, 13);
    const std::vector<u8> z = zlib_compress(raw.data(), raw.size());
    put_chunk(out, "IDAT", z.data(), z.size());
    put_chunk(out, "IEND", nullptr, 0);
    return out;
}

bool write_argb(const std::string& path, const u32* pixels, int w, int h, int stride, bool alpha) {
    const std::vector<u8> data = encode_argb(pixels, w, h, stride, alpha);
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const bool ok = std::fwrite(data.data(), 1, data.size(), f) == data.size();
    return std::fclose(f) == 0 && ok;
}

} // namespace cfd::png
