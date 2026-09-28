// ============================================================================
//  core/png.hpp — escritor PNG autocontenido (sin zlib).
//
//  Implementa DEFLATE propio: LZ77 con tabla hash + códigos Huffman fijos
//  (RFC 1951 §3.2.6), filtros PNG adaptativos por fila (heurística de
//  mínima suma absoluta), CRC-32 con tabla generada en tiempo de compilación
//  (constexpr) y Adler-32.
// ============================================================================
#pragma once

#include "config.hpp"
#include <string>
#include <vector>

namespace cfd::png {

// Píxeles 0xAARRGGBB (little-endian: B,G,R,A en memoria), fila a fila, `stride` en píxeles.
// alpha=false escribe RGB (más pequeño). Devuelve false si falla la E/S.
bool write_argb(const std::string& path, const u32* pixels, int w, int h, int stride, bool alpha = false);

// Codifica a memoria (útil para tests).
std::vector<u8> encode_argb(const u32* pixels, int w, int h, int stride, bool alpha = false);

// DEFLATE crudo con envoltura zlib (expuesto para tests).
std::vector<u8> zlib_compress(const u8* data, usize n);
u32 crc32(const u8* data, usize n, u32 crc = 0);
u32 adler32(const u8* data, usize n);

} // namespace cfd::png
