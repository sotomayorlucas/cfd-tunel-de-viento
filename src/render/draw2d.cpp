// ============================================================================
//  render/draw2d.cpp — implementación del dibujo 2D por software.
//
//  Trucos usados (medidos en docs/opt/ui.md):
//   * Relleno opaco: stores AVX2 de 8 píxeles + VPMASKMOVD para la cola (sin
//     bucle escalar); rellenos translúcidos: mezcla AVX2 a 16 bits por canal con
//     src·a precalculado (1 mul + 1 add + 1 shift por canal) y la misma fórmula
//     SWAR (2 canales por multiplicación de 32 bits) en escalar → bit-exacto.
//  *  Texto: LUT constexpr byte-de-fila → máscara de 8 carriles; cada fila de un
//     glifo 8×16 es UN VPMASKMOVD. Rango de filas no vacías por glifo precalculado
//     en tiempo de compilación (se saltan las filas vacías sin ramas).
//   * UTF-8: ASCII de 8 en 8 bytes con SWAR (detección de bytes < 0x20 y ≥ 0x80
//     con el truco "haszero"), conteo de glifos con popcount.
//   * Degradados verticales opacos con tramado ordenado Bayer 4×4 (sin bandas en
//     fondos oscuros) a coste cero: patrón de 4 píxeles por fila en un registro.
//   * Antialias por distancia al borde (líneas, polilíneas, círculos, triángulos,
//     esquinas redondeadas); la polilínea acumula cobertura MÁXIMA por fila.
// ============================================================================
#include "draw2d.hpp"
#include "font_data.hpp"

#include <bit>
#include <cmath>
#include <cstring>

namespace cfd::render {
namespace {

// Glifos extra extraídos de Uni2-TerminusBold16/32x16 (SIL OFL) + ✓ dibujada a mano.
// 31 glifos extra
constexpr char32_t k_extra_cp[] = {0x0394 /*Δ*/, 0x03C1 /*ρ*/, 0x03BD /*ν*/, 0x03C9 /*ω*/, 0x03B1 /*α*/, 0x03B2 /*β*/, 0x03B8 /*θ*/, 0x03C3 /*σ*/, 0x03A9 /*Ω*/, 0x03C4 /*τ*/, 0x2192 /*→*/, 0x2190 /*←*/, 0x2191 /*↑*/, 0x2193 /*↓*/, 0x2248 /*≈*/, 0x2264 /*≤*/, 0x2265 /*≥*/, 0x221E /*∞*/, 0x221A /*√*/, 0x2260 /*≠*/, 0x2026 /*…*/, 0x2013 /*–*/, 0x2014 /*—*/, 0x2022 /*•*/, 0x25B2 /*▲*/, 0x25BC /*▼*/, 0x25B8 /*▸*/, 0x25BE /*▾*/, 0x25C2 /*◂*/, 0x25B4 /*▴*/, 0x2713 /*✓*/};
alignas(64) constexpr u8 k_extra_small[][16] = {
  {0x00,0x00,0x10,0x10,0x38,0x38,0x6C,0x6C,0x6C,0xC6,0xC6,0xFE,0x00,0x00,0x00,0x00},
  {0x00,0x00,0x00,0x00,0x00,0x7C,0xC6,0xC6,0xC6,0xC6,0xC6,0xFC,0xC0,0xC0,0xC0,0x00},
  {0x00,0x00,0x00,0x00,0x00,0xC6,0xC6,0xC6,0x6C,0x6C,0x38,0x38,0x00,0x00,0x00,0x00},
  {0x00,0x00,0x00,0x00,0x00,0x44,0xC6,0xD6,0xD6,0xD6,0xFE,0x6C,0x00,0x00,0x00,0x00},
  {0x00,0x00,0x00,0x00,0x00,0x7A,0xCE,0xCC,0xCC,0xCC,0xCE,0x7A,0x00,0x00,0x00,0x00},
  {0x00,0x00,0x78,0xCC,0xCC,0xC8,0xFC,0xC6,0xC6,0xC6,0xC6,0xFC,0xC0,0xC0,0xC0,0x00},
  {0x00,0x00,0x3C,0x66,0x66,0x66,0x7E,0x66,0x66,0x66,0x66,0x3C,0x00,0x00,0x00,0x00},
  {0x00,0x00,0x00,0x00,0x00,0x3F,0x66,0x66,0x66,0x66,0x66,0x3C,0x00,0x00,0x00,0x00},
  {0x00,0x00,0x7C,0xC6,0xC6,0xC6,0xC6,0xC6,0xC6,0x6C,0x6C,0xEE,0x00,0x00,0x00,0x00},
  {0x00,0x00,0x00,0x00,0x00,0xFF,0x18,0x18,0x18,0x18,0x18,0x0E,0x00,0x00,0x00,0x00},
  {0x00,0x00,0x00,0x00,0x00,0x08,0x0C,0xFE,0xFE,0x0C,0x08,0x00,0x00,0x00,0x00,0x00},
  {0x00,0x00,0x00,0x00,0x00,0x20,0x60,0xFE,0xFE,0x60,0x20,0x00,0x00,0x00,0x00,0x00},
  {0x00,0x00,0x18,0x3C,0x7E,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x00,0x00,0x00,0x00},
  {0x00,0x00,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x7E,0x3C,0x18,0x00,0x00,0x00,0x00},
  {0x00,0x00,0x00,0x00,0x00,0x76,0xDC,0x00,0x76,0xDC,0x00,0x00,0x00,0x00,0x00,0x00},
  {0x00,0x00,0x00,0x0C,0x18,0x30,0x60,0x30,0x18,0x0C,0x00,0x7E,0x00,0x00,0x00,0x00},
  {0x00,0x00,0x00,0x30,0x18,0x0C,0x06,0x0C,0x18,0x30,0x00,0x7E,0x00,0x00,0x00,0x00},
  {0x00,0x00,0x00,0x00,0x00,0x7C,0xD6,0xD6,0xD6,0x7C,0x00,0x00,0x00,0x00,0x00,0x00},
  {0x00,0x0E,0x0C,0x0C,0x0C,0x0C,0xCC,0xCC,0xCC,0x6C,0x3C,0x1C,0x00,0x00,0x00,0x00},
  {0x00,0x00,0x00,0x00,0x06,0xFE,0x18,0x30,0xFE,0xC0,0x00,0x00,0x00,0x00,0x00,0x00},
  {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xDB,0xDB,0x00,0x00,0x00,0x00},
  {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xFE,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
  {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xFE,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
  {0x00,0x00,0x00,0x00,0x00,0x00,0x18,0x3C,0x3C,0x18,0x00,0x00,0x00,0x00,0x00,0x00},
  {0x00,0x00,0x18,0x3C,0x7E,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x00,0x00,0x00,0x00},
  {0x00,0x00,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x7E,0x3C,0x18,0x00,0x00,0x00,0x00},
  {0x00,0x00,0x00,0x00,0x00,0x08,0x0C,0xFE,0xFE,0x0C,0x08,0x00,0x00,0x00,0x00,0x00},
  {0x00,0x00,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x7E,0x3C,0x18,0x00,0x00,0x00,0x00},
  {0x00,0x00,0x00,0x00,0x00,0x20,0x60,0xFE,0xFE,0x60,0x20,0x00,0x00,0x00,0x00,0x00},
  {0x00,0x00,0x18,0x3C,0x7E,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x00,0x00,0x00,0x00},
  {0x00,0x00,0x00,0x00,0x01,0x03,0x06,0x0C,0xD8,0xF0,0x60,0x20,0x00,0x00,0x00,0x00},
};
alignas(64) constexpr u16 k_extra_large[][32] = {
  {0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0380,0x0380,0x0380,0x07C0,0x07C0,0x07C0,0x0EE0,0x0EE0,0x0EE0,0x1C70,0x1C70,0x1C70,0x3838,0x3838,0x3838,0x701C,0x701C,0x701C,0x7FFC,0x7FFC,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000},
  {0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x1FF0,0x3FF8,0x783C,0x701C,0x701C,0x701C,0x701C,0x701C,0x701C,0x701C,0x701C,0x703C,0x7FF8,0x7FF0,0x7000,0x7000,0x7000,0x7000,0x7000,0x0000},
  {0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x701C,0x701C,0x701C,0x3838,0x3838,0x3838,0x1C70,0x1C70,0x1C70,0x0EE0,0x0EE0,0x07C0,0x07C0,0x07C0,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000},
  {0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x3838,0x3838,0x701C,0x701C,0x739C,0x739C,0x739C,0x739C,0x739C,0x739C,0x739C,0x77DC,0x3FF8,0x1EF0,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000},
  {0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x1FEE,0x3FFE,0x783C,0x7038,0x7038,0x7038,0x7038,0x7038,0x7038,0x7038,0x7038,0x783C,0x3FFE,0x1FEE,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000},
  {0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x3FE0,0x7FF0,0x7078,0x7038,0x7038,0x7038,0x7038,0x7070,0x7FF0,0x7FF0,0x7038,0x701C,0x701C,0x701C,0x701C,0x701C,0x701C,0x703C,0x7FF8,0x7FF0,0x7000,0x7000,0x7000,0x7000,0x7000,0x0000},
  {0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0FE0,0x1FF0,0x3C78,0x3838,0x3838,0x3838,0x3838,0x3838,0x3838,0x3FF8,0x3FF8,0x3838,0x3838,0x3838,0x3838,0x3838,0x3838,0x3C78,0x1FF0,0x0FE0,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000},
  {0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x1FFE,0x3FFE,0x78F0,0x7078,0x703C,0x701C,0x701C,0x701C,0x701C,0x701C,0x701C,0x783C,0x3FF8,0x1FF0,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000},
  {0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x1FF0,0x3FF8,0x783C,0x701C,0x701C,0x701C,0x701C,0x701C,0x701C,0x701C,0x701C,0x701C,0x701C,0x701C,0x3838,0x1C70,0x1C70,0x1C70,0x7C7C,0x7C7C,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000},
  {0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x7FFC,0x7FFC,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x03C0,0x01F8,0x00F8,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000},
  {0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x00C0,0x00E0,0x0070,0x0038,0x001C,0x7FFE,0x7FFE,0x7FFE,0x001C,0x0038,0x0070,0x00E0,0x00C0,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000},
  {0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0300,0x0700,0x0E00,0x1C00,0x3800,0x7FFE,0x7FFE,0x7FFE,0x3800,0x1C00,0x0E00,0x0700,0x0300,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000},
  {0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0380,0x07C0,0x0FE0,0x1FF0,0x3BB8,0x739C,0x638C,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000},
  {0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x638C,0x739C,0x3BB8,0x1FF0,0x0FE0,0x07C0,0x0380,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000},
  {0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x3E1C,0x7FBC,0x7BFC,0x70F8,0x0000,0x0000,0x3E1C,0x7FBC,0x7BFC,0x70F8,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000},
  {0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0038,0x0070,0x00E0,0x01C0,0x0380,0x0700,0x0E00,0x1C00,0x1C00,0x0E00,0x0700,0x0380,0x01C0,0x00E0,0x0070,0x0038,0x0000,0x0000,0x3FFC,0x3FFC,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000},
  {0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x1C00,0x0E00,0x0700,0x0380,0x01C0,0x00E0,0x0070,0x0038,0x0038,0x0070,0x00E0,0x01C0,0x0380,0x0700,0x0E00,0x1C00,0x0000,0x0000,0x3FFC,0x3FFC,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000},
  {0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x3EF8,0x7FFC,0xE7CE,0xE38E,0xE38E,0xE38E,0xE38E,0xE7CE,0x7FFC,0x3EF8,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000},
  {0x0000,0x0000,0x0000,0x003E,0x003E,0x0038,0x0038,0x0038,0x0038,0x0038,0x0038,0x0038,0x0038,0x0038,0x0038,0x7038,0x7038,0x7038,0x7838,0x3C38,0x1E38,0x0F38,0x07B8,0x03F8,0x01F8,0x00F8,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000},
  {0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x000E,0x001C,0x7FFE,0x7FFE,0x00E0,0x01C0,0x0380,0x0700,0x7FFE,0x7FFE,0x3800,0x7000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000},
  {0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x739C,0x739C,0x739C,0x739C,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000},
  {0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x7FFC,0x7FFC,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000},
  {0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x7FFE,0x7FFE,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000},
  {0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x03C0,0x07E0,0x0FF0,0x0FF0,0x0FF0,0x0FF0,0x07E0,0x03C0,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000},
  {0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0380,0x07C0,0x0FE0,0x1FF0,0x3BB8,0x739C,0x638C,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000},
  {0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x638C,0x739C,0x3BB8,0x1FF0,0x0FE0,0x07C0,0x0380,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000},
  {0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x00C0,0x00E0,0x0070,0x0038,0x001C,0x7FFE,0x7FFE,0x7FFE,0x001C,0x0038,0x0070,0x00E0,0x00C0,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000},
  {0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x638C,0x739C,0x3BB8,0x1FF0,0x0FE0,0x07C0,0x0380,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000},
  {0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0300,0x0700,0x0E00,0x1C00,0x3800,0x7FFE,0x7FFE,0x7FFE,0x3800,0x1C00,0x0E00,0x0700,0x0300,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000},
  {0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0380,0x07C0,0x0FE0,0x1FF0,0x3BB8,0x739C,0x638C,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0380,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000},
  {0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0006,0x000F,0x000F,0x001E,0x001C,0x003C,0x0078,0x0070,0x60F0,0xF0E0,0xF1E0,0x7BC0,0x3F80,0x1F80,0x0F00,0x0F00,0x0600,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000},
};

constexpr int k_nextra = static_cast<int>(sizeof(k_extra_cp) / sizeof(k_extra_cp[0]));
static_assert(224 + k_nextra <= 256, "demasiados glifos extra");
constexpr int k_qmark = '?' - 0x20;

// Tablas combinadas de 256 glifos (Latin-1 + extras) con el rango de filas no vacías.
struct SmallTable { u8 g[256][16]; u8 r0[256], r1[256]; };
struct LargeTable { u16 g[256][32]; u8 r0[256], r1[256]; };

constexpr SmallTable make_small() {
    SmallTable t{};
    for (int i = 0; i < 256; ++i) {
        int a = -1, b = 0;
        for (int r = 0; r < 16; ++r) {
            const u8 v = i < 224 ? font::k_small_glyphs[i][r]
                                 : (i - 224 < k_nextra ? k_extra_small[i - 224][r] : font::k_small_glyphs[k_qmark][r]);
            t.g[i][r] = v;
            if (v) { if (a < 0) a = r; b = r + 1; }
        }
        t.r0[i] = static_cast<u8>(a < 0 ? 0 : a);
        t.r1[i] = static_cast<u8>(a < 0 ? 0 : b);
    }
    return t;
}
constexpr LargeTable make_large() {
    LargeTable t{};
    for (int i = 0; i < 256; ++i) {
        int a = -1, b = 0;
        for (int r = 0; r < 32; ++r) {
            const u16 v = i < 224 ? font::k_large_glyphs[i][r]
                                  : (i - 224 < k_nextra ? k_extra_large[i - 224][r] : font::k_large_glyphs[k_qmark][r]);
            t.g[i][r] = v;
            if (v) { if (a < 0) a = r; b = r + 1; }
        }
        t.r0[i] = static_cast<u8>(a < 0 ? 0 : a);
        t.r1[i] = static_cast<u8>(a < 0 ? 0 : b);
    }
    return t;
}
alignas(64) constexpr SmallTable k_small = make_small();
alignas(64) constexpr LargeTable k_large = make_large();

// LUT byte → máscara de 8 carriles de 32 bits (bit 7 = píxel izquierdo). 8 KiB, vive en L1.
struct MaskLut { u32 m[256][8]; };
constexpr MaskLut make_masks() {
    MaskLut t{};
    for (int b = 0; b < 256; ++b)
        for (int i = 0; i < 8; ++i) t.m[b][i] = ((b >> (7 - i)) & 1) ? 0xFFFFFFFFu : 0u;
    return t;
}
alignas(64) constexpr MaskLut k_mask = make_masks();

// Bayer 4×4 (umbral en 1/16 de LSB, centrado).
constexpr int k_bayer[4][4] = {{0, 8, 2, 10}, {12, 4, 14, 6}, {3, 11, 1, 9}, {15, 7, 13, 5}};

constexpr i32 k_inf = 1 << 29;

CFD_INLINE __m256i lane_idx() { return _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7); }
// Máscara de los primeros k carriles (0 ≤ k ≤ 8).
CFD_INLINE __m256i head_mask(int k) { return _mm256_cmpgt_epi32(_mm256_set1_epi32(k), lane_idx()); }

// Mezcla de un color constante sobre 8 píxeles: d' = (s·a + d·(256-a)) >> 8 por canal.
// El alfa del resultado se fuerza a 0xFF (igual que blend_px): bit-exacto con el SWAR escalar
// también cuando el destino no es opaco (p.ej. un Framebuffer recién creado, alfa 0).
struct ConstBlend {
    __m256i sa, ia, a_ff;
    CFD_INLINE ConstBlend(u32 c, u32 a256) {
        const __m256i s = _mm256_unpacklo_epi8(_mm256_set1_epi32(static_cast<int>(c | 0xFF000000u)), _mm256_setzero_si256());
        sa = _mm256_mullo_epi16(s, _mm256_set1_epi16(static_cast<short>(a256)));
        ia = _mm256_set1_epi16(static_cast<short>(256 - a256));
        a_ff = _mm256_set1_epi32(static_cast<int>(0xFF000000u));
    }
    CFD_INLINE __m256i operator()(__m256i d) const {
        const __m256i z = _mm256_setzero_si256();
        __m256i lo = _mm256_unpacklo_epi8(d, z), hi = _mm256_unpackhi_epi8(d, z);
        lo = _mm256_srli_epi16(_mm256_add_epi16(_mm256_mullo_epi16(lo, ia), sa), 8);
        hi = _mm256_srli_epi16(_mm256_add_epi16(_mm256_mullo_epi16(hi, ia), sa), 8);
        return _mm256_or_si256(_mm256_packus_epi16(lo, hi), a_ff);
    }
};

// Mezcla por píxel con el alfa de la fuente (alfa directo). op256 = opacidad global 0..256.
CFD_INLINE __m256i blend_src_alpha8(__m256i s, __m256i d, __m256i op256) {
    const __m256i z = _mm256_setzero_si256();
    __m256i a = _mm256_srli_epi32(s, 24);
    a = _mm256_add_epi32(a, _mm256_srli_epi32(a, 7));                       // 0..256
    a = _mm256_srli_epi32(_mm256_mullo_epi32(a, op256), 8);                  // × opacidad
    const __m256i a2 = _mm256_or_si256(a, _mm256_slli_epi32(a, 16));         // [a|a] por píxel
    const __m256i alo = _mm256_unpacklo_epi32(a2, a2), ahi = _mm256_unpackhi_epi32(a2, a2);
    const __m256i k256 = _mm256_set1_epi16(256);
    __m256i slo = _mm256_unpacklo_epi8(s, z), shi = _mm256_unpackhi_epi8(s, z);
    __m256i dlo = _mm256_unpacklo_epi8(d, z), dhi = _mm256_unpackhi_epi8(d, z);
    dlo = _mm256_srli_epi16(_mm256_add_epi16(_mm256_mullo_epi16(slo, alo), _mm256_mullo_epi16(dlo, _mm256_sub_epi16(k256, alo))), 8);
    dhi = _mm256_srli_epi16(_mm256_add_epi16(_mm256_mullo_epi16(shi, ahi), _mm256_mullo_epi16(dhi, _mm256_sub_epi16(k256, ahi))), 8);
    return _mm256_or_si256(_mm256_packus_epi16(dlo, dhi), _mm256_set1_epi32(static_cast<int>(0xFF000000u)));
}

// Píxeles hasta la siguiente frontera de 32 B (0..7). Las filas del framebuffer están
// alineadas a 64 B pero los tramos empiezan en cualquier x: un store de 32 B desalineado
// cruza línea de caché la mitad de las veces (2 accesos). Se "pela" la cabeza con un
// VPMASKMOVD y el resto va alineado.
CFD_INLINE int peel_px(const u32* d) { return static_cast<int>((8 - ((reinterpret_cast<uintptr_t>(d) >> 2) & 7)) & 7); }

// vh = vector para los stores que empiezan en d (cabeza pelada, o todo si n < 16);
// vb = vector para los que empiezan en la frontera de 32 B (d + h + 8k). Un patrón
// periódico (tramado) necesita la fase de cada tramo: ver gradient_v. Para un color
// constante vh == vb y el compilador lo pliega.
CFD_INLINE void store_span2(u32* CFD_RESTRICT d, int n, __m256i vh, __m256i vb) {
    int i = 0;
    if (n >= 16) {
        const int h = peel_px(d);
        if (h) _mm256_maskstore_epi32(reinterpret_cast<int*>(d), head_mask(h), vh);
        i = h;
        for (; i + 8 <= n; i += 8) _mm256_store_si256(reinterpret_cast<__m256i*>(d + i), vb);
        if (i < n) _mm256_maskstore_epi32(reinterpret_cast<int*>(d + i), head_mask(n - i), vb);
    } else {
        for (; i + 8 <= n; i += 8) _mm256_storeu_si256(reinterpret_cast<__m256i*>(d + i), vh);
        if (i < n) _mm256_maskstore_epi32(reinterpret_cast<int*>(d + i), head_mask(n - i), vh);
    }
}
CFD_INLINE void store_span(u32* CFD_RESTRICT d, int n, __m256i v) { store_span2(d, n, v, v); }
CFD_INLINE void blend_span(u32* CFD_RESTRICT d, int n, const ConstBlend& B) {
    int i = 0;
    if (n >= 16) {
        const int h = peel_px(d);
        if (h) {
            const __m256i m = head_mask(h);
            _mm256_maskstore_epi32(reinterpret_cast<int*>(d), m, B(_mm256_maskload_epi32(reinterpret_cast<const int*>(d), m)));
        }
        i = h;
        for (; i + 8 <= n; i += 8) {
            __m256i* p = reinterpret_cast<__m256i*>(d + i);
            _mm256_store_si256(p, B(_mm256_load_si256(p)));
        }
    } else {
        for (; i + 8 <= n; i += 8) {
            __m256i* p = reinterpret_cast<__m256i*>(d + i);
            _mm256_storeu_si256(p, B(_mm256_loadu_si256(p)));
        }
    }
    if (i < n) {
        const __m256i m = head_mask(n - i);
        int* p = reinterpret_cast<int*>(d + i);
        _mm256_maskstore_epi32(p, m, B(_mm256_maskload_epi32(p, m)));
    }
}

CFD_INLINE float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
CFD_INLINE bool finite4(float a, float b, float c, float d) { return std::isfinite(a) && std::isfinite(b) && std::isfinite(c) && std::isfinite(d); }
// float → int con saturación (evita UB con coordenadas absurdas).
CFD_INLINE int sat_floor(float v) { return static_cast<int>(std::floor(clampf(v, -1e8f, 1e8f))); }
CFD_INLINE int sat_ceil(float v) { return static_cast<int>(std::ceil(clampf(v, -1e8f, 1e8f))); }

// Distancia de p al segmento a-b (dx,dy = b-a, il2 = 1/|b-a|², 0 si degenerado).
CFD_INLINE float seg_dist(float px, float py, float ax, float ay, float dx, float dy, float il2) {
    const float wx = px - ax, wy = py - ay;
    const float t = clampf((wx * dx + wy * dy) * il2, 0.0f, 1.0f);
    const float ex = wx - dx * t, ey = wy - dy * t;
    return std::sqrt(ex * ex + ey * ey);
}
// Rango de x (continuo) de la fila de centro py que puede estar a ≤ R del segmento.
CFD_INLINE bool seg_row_range(float ax, float ay, float bx, float by, float py, float R, float& xl, float& xr) {
    const float dy = by - ay;
    float t0 = 0.0f, t1 = 1.0f;
    if (std::fabs(dy) < 1e-6f) {
        if (std::fabs(py - ay) > R) return false;
    } else {
        t0 = (py - R - ay) / dy; t1 = (py + R - ay) / dy;
        if (t0 > t1) { const float t = t0; t0 = t1; t1 = t; }
        t0 = t0 < 0.0f ? 0.0f : t0; t1 = t1 > 1.0f ? 1.0f : t1;
        if (t0 > t1) return false;
    }
    const float xa = ax + (bx - ax) * t0, xb = ax + (bx - ax) * t1;
    xl = (xa < xb ? xa : xb) - R;
    xr = (xa < xb ? xb : xa) + R;
    return true;
}

// Satura un rectángulo a ±2^23 (sin desbordes de int en x+w; exacto como float en DrawList).
CFD_INLINE Rect sat_rect(Rect r) {
    const int lim = 1 << 23;
    const i64 x0 = clamp_<i64>(r.x, -lim, lim), y0 = clamp_<i64>(r.y, -lim, lim);
    const i64 x1 = clamp_<i64>(static_cast<i64>(r.x) + r.w, -lim, lim), y1 = clamp_<i64>(static_cast<i64>(r.y) + r.h, -lim, lim);
    return {static_cast<int>(x0), static_cast<int>(y0), static_cast<int>(x1 - x0), static_cast<int>(y1 - y0)};
}

// Suma/sujeción en 64 bits → int en ±2^30 (coordenadas absurdas sin desbordes con signo).
CFD_INLINE int sat_i(i64 v) { return static_cast<int>(v < -(1ll << 30) ? -(1ll << 30) : (v > (1ll << 30) ? (1ll << 30) : v)); }

struct CovBuf {             // búfer de cobertura por fila (uno por hilo)
    float* p = nullptr; int cap = 0;
    ~CovBuf() { std::free(p); }
    float* get(int n) {
        if (n > cap) { std::free(p); cap = (n + 1023) & ~1023; p = static_cast<float*>(std::calloc(static_cast<usize>(cap), sizeof(float))); }
        return p;
    }
};
thread_local CovBuf t_cov;

} // namespace

// ============================================================================
//  Texto: UTF-8 → glifos
// ============================================================================
u8 glyph_index(char32_t cp) {
    if (cp >= 0x20 && cp <= 0xFF) return static_cast<u8>(cp - 0x20);
    switch (cp) {
        case U'\t': return 0;
        case 0x03BC: return static_cast<u8>(0xB5 - 0x20);                 // μ griega → µ
        case 0x2018: case 0x2019: case 0x2032: return static_cast<u8>('\'' - 0x20);
        case 0x201C: case 0x201D: case 0x2033: return static_cast<u8>('"' - 0x20);
        case 0x2212: case 0x2010: case 0x2011: return static_cast<u8>('-' - 0x20);
        case 0x2009: case 0x200A: case 0x202F: case 0x2002: case 0x2003: return 0;
        case 0x2715: case 0x2716: return static_cast<u8>(0xD7 - 0x20);     // ✕ → ×
        case 0x2714: return static_cast<u8>(224 + k_nextra - 1);          // ✔ → ✓
        case 0x25B6: case 0x25BA: return 224 + 26;                          // ▶ → ▸ (ver tabla)
        case 0x25C0: case 0x25C4: return 224 + 28;                          // ◀ → ◂
        default: break;
    }
    for (int k = 0; k < k_nextra; ++k)
        if (k_extra_cp[k] == cp) return static_cast<u8>(224 + k);
    return static_cast<u8>(k_qmark);
}

int utf8_to_glyphs(const char* s, int nbytes, u8* out, int cap) {
    if (!s) return 0;
    const usize n = nbytes < 0 ? std::strlen(s) : static_cast<usize>(nbytes);
    const u8* p = reinterpret_cast<const u8*>(s);
    const u8* e = p + n;
    int k = 0;
    while (p < e && k < cap) {
        // Ruta rápida SWAR: 8 bytes ASCII imprimibles de una vez (resta 0x20 sin préstamos).
        if (e - p >= 8 && cap - k >= 8) {
            u64 w;
            std::memcpy(&w, p, 8);
            const u64 hi = w & 0x8080808080808080ull;
            const u64 lt = (w - 0x2020202020202020ull) & ~w & 0x8080808080808080ull;   // byte < 0x20
            if ((hi | lt) == 0) {
                const u64 g = w - 0x2020202020202020ull;
                std::memcpy(out + k, &g, 8);
                k += 8; p += 8;
                continue;
            }
        }
        const u32 b0 = *p;
        char32_t cp;
        int len = 1;
        // Byte de continuación huérfano (10xxxxxx sin inicial): no produce glifo. Así el
        // nº de glifos coincide SIEMPRE con utf8_count (que cuenta los bytes que no son de
        // continuación): text_width mide lo que se dibuja y DrawList::text (que reserva
        // utf8_count glifos) no recorta la cola de la cadena.
        if ((b0 & 0xC0) == 0x80) { ++p; continue; }
        if (b0 < 0x80) cp = b0 < 0x20 ? U' ' : b0;
        else if (b0 >= 0xC2 && b0 <= 0xDF && e - p >= 2 && (p[1] & 0xC0) == 0x80) {
            cp = ((b0 & 0x1Fu) << 6) | (p[1] & 0x3Fu); len = 2;
        } else if (b0 >= 0xE0 && b0 <= 0xEF && e - p >= 3 && (p[1] & 0xC0) == 0x80 && (p[2] & 0xC0) == 0x80) {
            cp = ((b0 & 0x0Fu) << 12) | ((p[1] & 0x3Fu) << 6) | (p[2] & 0x3Fu); len = 3;
            if (cp < 0x800) cp = U'?';
        } else if (b0 >= 0xF0 && b0 <= 0xF4 && e - p >= 4 && (p[1] & 0xC0) == 0x80 && (p[2] & 0xC0) == 0x80 && (p[3] & 0xC0) == 0x80) {
            cp = U'?'; len = 4;   // fuera del plano básico: sin glifos
        } else {
            // Secuencia inválida o truncada → un solo '?' (subparte maximal): se consumen
            // el byte inicial y los de continuación que le sigan.
            cp = U'?';
            while (p + len < e && (p[len] & 0xC0) == 0x80) ++len;
        }
        out[k++] = glyph_index(cp);
        p += len;
    }
    return k;
}

int utf8_count(const char* s, int nbytes) {
    if (!s) return 0;
    const usize n = nbytes < 0 ? std::strlen(s) : static_cast<usize>(nbytes);
    const u8* p = reinterpret_cast<const u8*>(s);
    usize i = 0, cont = 0;
    for (; i + 8 <= n; i += 8) {
        u64 w;
        std::memcpy(&w, p + i, 8);
        // byte de continuación 10xxxxxx ⇔ bit7 = 1 y bit6 = 0
        cont += static_cast<usize>(std::popcount(w & ~(w << 1) & 0x8080808080808080ull));
    }
    for (; i < n; ++i) cont += (p[i] & 0xC0) == 0x80;
    return static_cast<int>(n - cont);
}

int utf8_prefix_bytes(const char* s, int nbytes, int max_glyphs) {
    if (!s || max_glyphs <= 0) return 0;
    const int n = nbytes < 0 ? static_cast<int>(std::strlen(s)) : nbytes;
    int g = 0;
    for (int i = 0; i < n; ++i) {
        if ((static_cast<u8>(s[i]) & 0xC0) != 0x80) {
            if (g == max_glyphs) return i;
            ++g;
        }
    }
    return n;
}

u32 lerp_color(u32 a, u32 b, float t) {
    t = clampf(t, 0.0f, 1.0f);
    const u32 k = static_cast<u32>(t * 256.0f + 0.5f), ik = 256 - k;
    const u32 rb = (((a & 0x00FF00FFu) * ik + (b & 0x00FF00FFu) * k) >> 8) & 0x00FF00FFu;
    const u32 ag = ((((a >> 8) & 0x00FF00FFu) * ik + ((b >> 8) & 0x00FF00FFu) * k)) & 0xFF00FF00u;
    return rb | ag;
}

// ============================================================================
//  Painter: recorte
// ============================================================================
void Painter::reset(Framebuffer& fb) {
    fb_ = &fb;
    cx0_ = 0; cy0_ = 0; cx1_ = fb.w; cy1_ = fb.h;
    depth_ = 0;
}

// Intersección en 64 bits; se sujeta ANTES de volver a int (x + w puede salirse de int:
// p.ej. x = -2^31+3, w = -352 → x1 < INT_MIN; truncarlo daba un x1 enorme positivo).
static CFD_INLINE void intersect_into(int& x0, int& y0, int& x1, int& y1, Rect r) {
    const i64 rx0 = r.x, ry0 = r.y, rx1 = static_cast<i64>(r.x) + r.w, ry1 = static_cast<i64>(r.y) + r.h;
    const i64 nx0 = max_<i64>(x0, rx0), ny0 = max_<i64>(y0, ry0);
    i64 nx1 = min_<i64>(x1, rx1), ny1 = min_<i64>(y1, ry1);
    if (nx1 < nx0) nx1 = nx0;
    if (ny1 < ny0) ny1 = ny0;
    x0 = static_cast<int>(nx0); y0 = static_cast<int>(ny0);
    x1 = static_cast<int>(nx1); y1 = static_cast<int>(ny1);
}

void Painter::push_clip(Rect r) {
    CFD_CHECK(depth_ < 32, "Painter: pila de recortes llena");
    stack_[depth_++] = clip();
    intersect_into(cx0_, cy0_, cx1_, cy1_, r);
}
void Painter::pop_clip() {
    if (depth_ <= 0) return;
    const Rect r = stack_[--depth_];
    cx0_ = r.x; cy0_ = r.y; cx1_ = r.x + r.w; cy1_ = r.y + r.h;
}
void Painter::set_clip(Rect r) {
    cx0_ = 0; cy0_ = 0; cx1_ = fb_->w; cy1_ = fb_->h;
    intersect_into(cx0_, cy0_, cx1_, cy1_, r);
}

// ============================================================================
//  Rectángulos
// ============================================================================
void Painter::span_fill(u32* d, int n, u32 c) {
    const u32 a = c >> 24;
    if (n <= 0 || a == 0) return;
    if (a == 255) store_span(d, n, _mm256_set1_epi32(static_cast<int>(c)));
    else blend_span(d, n, ConstBlend(c, alpha256(a)));
}

void Painter::fill_rect(Rect r, u32 c) {
    const u32 a = c >> 24;
    if (a == 0) return;
    int x0 = cx0_, y0 = cy0_, x1 = cx1_, y1 = cy1_;
    intersect_into(x0, y0, x1, y1, r);
    const int n = x1 - x0;
    if (n <= 0 || y0 >= y1) return;
    if (a == 255) {
        const __m256i v = _mm256_set1_epi32(static_cast<int>(c));
        for (int y = y0; y < y1; ++y) store_span(fb_->row(y) + x0, n, v);
    } else {
        const ConstBlend B(c, alpha256(a));
        for (int y = y0; y < y1; ++y) blend_span(fb_->row(y) + x0, n, B);
    }
}

void Painter::rect(Rect r, u32 c, int t) {
    r = sat_rect(r);
    t = min_(t, 1 << 24);                             // t*2 y r.y+t sin desborde con signo
    if (r.w <= 0 || r.h <= 0 || t <= 0) return;
    if (t * 2 >= r.w || t * 2 >= r.h) { fill_rect(r, c); return; }
    fill_rect({r.x, r.y, r.w, t}, c);
    fill_rect({r.x, r.y + r.h - t, r.w, t}, c);
    fill_rect({r.x, r.y + t, t, r.h - 2 * t}, c);
    fill_rect({r.x + r.w - t, r.y + t, t, r.h - 2 * t}, c);
}
void Painter::hline(int x0, int x1, int y, u32 c) { if (x1 > x0) fill_rect({x0, y, sat_i(static_cast<i64>(x1) - x0), 1}, c); }
void Painter::vline(int x, int y0, int y1, u32 c) { if (y1 > y0) fill_rect({x, y0, 1, sat_i(static_cast<i64>(y1) - y0)}, c); }

// Cobertura de un píxel de esquina: distancia del centro del píxel al centro del arco.
static CFD_INLINE float corner_cov(float px, float py, float ccx, float ccy, float rad) {
    const float dx = px - ccx, dy = py - ccy;
    return clampf(rad - std::sqrt(dx * dx + dy * dy) + 0.5f, 0.0f, 1.0f);
}

void Painter::fill_round_rect(Rect r, float rad, u32 c, u32 corners) {
    r = sat_rect(r);
    if (r.w <= 0 || r.h <= 0 || (c >> 24) == 0) return;
    if (!std::isfinite(rad)) rad = 0.0f;
    rad = clampf(rad, 0.0f, 0.5f * static_cast<float>(min_(r.w, r.h)));
    if (rad < 0.5f || (corners & 0xF) == 0) { fill_rect(r, c); return; }
    const int ri = static_cast<int>(std::ceil(rad));
    int x0 = cx0_, y0 = cy0_, x1 = cx1_, y1 = cy1_;
    intersect_into(x0, y0, x1, y1, r);
    if (x0 >= x1 || y0 >= y1) return;
    const u32 a256 = alpha256(c >> 24);
    const float lcx = static_cast<float>(r.x) + rad, rcx = static_cast<float>(r.x + r.w) - rad;
    const float tcy = static_cast<float>(r.y) + rad, bcy = static_cast<float>(r.y + r.h) - rad;
    for (int y = y0; y < y1; ++y) {
        u32* row = fb_->row(y);
        const bool top = y < r.y + ri, bot = y >= r.y + r.h - ri;
        const bool L = (top && (corners & 1)) || (bot && (corners & 4));
        const bool R = (top && (corners & 2)) || (bot && (corners & 8));
        if (!L && !R) { span_fill(row + x0, x1 - x0, c); continue; }
        // Sólo las celdas de esquina (ri×ri) calculan distancia; el tramo central siempre
        // está cubierto (el borde del rectángulo cae en frontera de píxel).
        const float py = static_cast<float>(y) + 0.5f;
        const float dy = max_(max_(tcy - py, py - bcy), 0.0f), dy2 = dy * dy;
        int ms = x0, me = x1;
        if (L) {
            const int le = min_(x1, r.x + ri);
            for (int x = x0; x < le; ++x) {
                const float dx = max_(lcx - (static_cast<float>(x) + 0.5f), 0.0f);
                const float cov = clampf(rad - std::sqrt(dx * dx + dy2) + 0.5f, 0.0f, 1.0f);
                const u32 k = static_cast<u32>(cov * static_cast<float>(a256) + 0.5f);
                if (k) row[x] = blend_px(row[x], c, k);
            }
            ms = max_(ms, le);
        }
        if (R) {
            const int rs = max_(max_(x0, r.x + r.w - ri), ms);
            for (int x = rs; x < x1; ++x) {
                const float dx = max_((static_cast<float>(x) + 0.5f) - rcx, 0.0f);
                const float cov = clampf(rad - std::sqrt(dx * dx + dy2) + 0.5f, 0.0f, 1.0f);
                const u32 k = static_cast<u32>(cov * static_cast<float>(a256) + 0.5f);
                if (k) row[x] = blend_px(row[x], c, k);
            }
            me = min_(me, rs);
        }
        if (me > ms) span_fill(row + ms, me - ms, c);
    }
}

void Painter::round_rect(Rect r, float rad, u32 c, int t, u32 corners) {
    r = sat_rect(r);
    t = min_(t, 1 << 20);
    if (r.w <= 0 || r.h <= 0 || t <= 0 || (c >> 24) == 0) return;
    if (!std::isfinite(rad)) rad = 0.0f;
    rad = clampf(rad, 0.0f, 0.5f * static_cast<float>(min_(r.w, r.h)));
    if (rad < 0.5f || (corners & 0xF) == 0) { rect(r, c, t); return; }
    // Contorno que cubre todo (2t ≥ lado menor) = rectángulo redondeado relleno. Antes las
    // "celdas de esquina" medían ri = max(r, t) > w y se salían del rectángulo (con t enorme
    // se pintaba un cuadrado de 2^20 px alrededor; con t = 3 en una caja de 4 px, 1-2 px fuera).
    if (2 * t >= min_(r.w, r.h)) { fill_round_rect(r, rad, c, corners); return; }
    const int ri = max_(static_cast<int>(std::ceil(rad)), t);
    // Las celdas izquierda/derecha (superior/inferior) se reparten el rectángulo por la mitad
    // cuando 2·ri > lado: sin solaparse (un píxel mezclado dos veces saldría más opaco).
    const int xm = r.x + r.w / 2, ym = r.y + r.h / 2;
    // Lados rectos (sin las esquinas).
    fill_rect({r.x + ri, r.y, r.w - 2 * ri, t}, c);
    fill_rect({r.x + ri, r.y + r.h - t, r.w - 2 * ri, t}, c);
    fill_rect({r.x, r.y + ri, t, r.h - 2 * ri}, c);
    fill_rect({r.x + r.w - t, r.y + ri, t, r.h - 2 * ri}, c);
    const u32 a256 = alpha256(c >> 24);
    const float tf = static_cast<float>(t);
    for (int k = 0; k < 4; ++k) {
        const bool right = k & 1, bottom = k & 2;
        const int qx = right ? r.x + r.w - ri : r.x, qy = bottom ? r.y + r.h - ri : r.y;
        const float ccx = right ? static_cast<float>(r.x + r.w) - rad : static_cast<float>(r.x) + rad;
        const float ccy = bottom ? static_cast<float>(r.y + r.h) - rad : static_cast<float>(r.y) + rad;
        const bool round = corners & (1u << k);
        int x0 = cx0_, y0 = cy0_, x1 = cx1_, y1 = cy1_;
        intersect_into(x0, y0, x1, y1, {qx, qy, ri, ri});
        // Mitad propia de la celda (partición sin solapes del rectángulo).
        intersect_into(x0, y0, x1, y1, {right ? xm : r.x, bottom ? ym : r.y, right ? r.x + r.w - xm : xm - r.x, bottom ? r.y + r.h - ym : ym - r.y});
        for (int y = y0; y < y1; ++y) {
            u32* row = fb_->row(y);
            const float py = static_cast<float>(y) + 0.5f;
            for (int x = x0; x < x1; ++x) {
                const float px = static_cast<float>(x) + 0.5f;
                float cov;
                if (round) {
                    // Distancia al arco en el cuadrante de la esquina (fuera → lado recto).
                    const float dx = right ? max_(px - ccx, 0.0f) : max_(ccx - px, 0.0f);
                    const float dy = bottom ? max_(py - ccy, 0.0f) : max_(ccy - py, 0.0f);
                    const float d = std::sqrt(dx * dx + dy * dy);
                    cov = clampf(rad - d + 0.5f, 0.0f, 1.0f) - clampf(rad - tf - d + 0.5f, 0.0f, 1.0f);
                } else {
                    const int ex = right ? (r.x + r.w - 1 - x) : (x - r.x);
                    const int ey = bottom ? (r.y + r.h - 1 - y) : (y - r.y);
                    cov = (ex < t || ey < t) ? 1.0f : 0.0f;
                }
                const u32 kk = static_cast<u32>(cov * static_cast<float>(a256) + 0.5f);
                if (kk) row[x] = blend_px(row[x], c, kk);
            }
        }
    }
}

void Painter::shadow(Rect r, float rad, float blur, u32 c, bool skip_inside) {
    r = sat_rect(r);
    if (r.w <= 0 || r.h <= 0 || (c >> 24) == 0 || !std::isfinite(blur) || !std::isfinite(rad)) return;
    blur = clampf(blur, 0.5f, 256.0f);
    const int e = static_cast<int>(std::ceil(blur)) + 1;
    int x0 = cx0_, y0 = cy0_, x1 = cx1_, y1 = cy1_;
    intersect_into(x0, y0, x1, y1, {r.x - e, r.y - e, r.w + 2 * e, r.h + 2 * e});
    if (x0 >= x1 || y0 >= y1) return;
    const float hx = 0.5f * static_cast<float>(r.w), hy = 0.5f * static_cast<float>(r.h);
    const float cx = static_cast<float>(r.x) + hx, cy = static_cast<float>(r.y) + hy;
    rad = clampf(rad, 0.0f, min_(hx, hy));
    const float a = static_cast<float>(c >> 24);
    const float inv2b = 1.0f / (2.0f * blur);
    const float m = max_(blur, rad) + 1.0f;                 // margen del interior "lleno"
    for (int y = y0; y < y1; ++y) {
        u32* row = fb_->row(y);
        const float py = static_cast<float>(y) + 0.5f;
        const float ay = std::fabs(py - cy);
        // Interior (d < -blur garantizado): se salta o se rellena de golpe.
        int sx0 = x1, sx1 = x1;
        if (ay < hy - m && hx - m > 0) {
            sx0 = max_(x0, sat_ceil(cx - (hx - m)));
            sx1 = min_(x1, sat_floor(cx + (hx - m)));
            if (sx1 <= sx0) { sx0 = x1; sx1 = x1; }   // vacío: sin salto (evita bucle infinito)
        }
        for (int x = x0; x < x1; ++x) {
            if (x == sx0) {
                if (!skip_inside) span_fill(row + sx0, sx1 - sx0, c);
                x = sx1 - 1;
                continue;
            }
            const float px = static_cast<float>(x) + 0.5f;
            const float qx = std::fabs(px - cx) - (hx - rad), qy = ay - (hy - rad);
            const float ox = max_(qx, 0.0f), oy = max_(qy, 0.0f);
            const float d = std::sqrt(ox * ox + oy * oy) + min_(max_(qx, qy), 0.0f) - rad;
            if (skip_inside && d < -blur) continue;
            float t = clampf((blur - d) * inv2b, 0.0f, 1.0f);
            t = t * t * (3.0f - 2.0f * t);
            const u32 k = static_cast<u32>(t * a + 0.5f);
            if (k) row[x] = blend_px(row[x], c, alpha256(k));
        }
    }
}

void Painter::gradient_v(Rect r, u32 top, u32 bottom) {
    r = sat_rect(r);
    int x0 = cx0_, y0 = cy0_, x1 = cx1_, y1 = cy1_;
    intersect_into(x0, y0, x1, y1, r);
    const int n = x1 - x0;
    if (n <= 0 || y0 >= y1) return;
    const float inv = r.h > 1 ? 1.0f / static_cast<float>(r.h - 1) : 0.0f;
    const bool opaque = (top >> 24) == 255 && (bottom >> 24) == 255;
    for (int y = y0; y < y1; ++y) {
        const float t = static_cast<float>(y - r.y) * inv;
        u32* d = fb_->row(y) + x0;
        if (!opaque) { span_fill(d, n, lerp_color(top, bottom, t)); continue; }
        // Canales en punto fijo 8.4 + umbral Bayer 4×4 → patrón de 4 píxeles por fila,
        // indexado por la columna ABSOLUTA (x & 3). pat[i] = patrón[i & 3] (11 entradas): el
        // vector de 8 píxeles que empieza en la columna xs es loadu(pat + (xs & 3)).
        // store_span2 escribe la cabeza pelada (empieza en x0) y el cuerpo alineado (empieza
        // en x0 + h) con su propia fase: antes el cuerpo reutilizaba la fase de x0 y el periodo
        // 4 del tramado se rompía en la unión cuando h no era múltiplo de 4.
        u32 pat[12];
        for (int k = 0; k < 4; ++k) {
            const int th = k_bayer[y & 3][k];
            u32 px = 0xFF000000u;
            for (int s = 0; s < 24; s += 8) {
                const float ca = static_cast<float>((top >> s) & 255), cb = static_cast<float>((bottom >> s) & 255);
                const int v16 = static_cast<int>((ca + (cb - ca) * t) * 16.0f) + th;   // en 1/16
                px |= static_cast<u32>(clamp_(v16 >> 4, 0, 255)) << s;
            }
            pat[k] = pat[k + 4] = pat[k + 8] = px;
        }
        const int ph_head = x0 & 3, ph_body = (x0 + peel_px(d)) & 3;
        store_span2(d, n, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pat + ph_head)),
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pat + ph_body)));
    }
}

void Painter::gradient_h(Rect r, u32 left, u32 right) {
    r = sat_rect(r);
    int x0 = cx0_, y0 = cy0_, x1 = cx1_, y1 = cy1_;
    intersect_into(x0, y0, x1, y1, r);
    const int n = x1 - x0;
    if (n <= 0 || y0 >= y1) return;
    const float inv = r.w > 1 ? 1.0f / static_cast<float>(r.w - 1) : 0.0f;
    const bool opaque = (left >> 24) == 255 && (right >> 24) == 255;
    float* tmpf = t_cov.get(n);                       // reutiliza el búfer por hilo como u32
    u32* cols = reinterpret_cast<u32*>(tmpf);
    for (int x = 0; x < n; ++x) cols[x] = lerp_color(left, right, static_cast<float>(x0 + x - r.x) * inv);
    for (int y = y0; y < y1; ++y) {
        u32* d = fb_->row(y) + x0;
        if (opaque) std::memcpy(d, cols, static_cast<usize>(n) * 4);
        else for (int x = 0; x < n; ++x) { const u32 a = cols[x] >> 24; if (a) d[x] = blend_px(d[x], cols[x], alpha256(a)); }
    }
    std::memset(tmpf, 0, static_cast<usize>(n) * sizeof(float));   // el búfer de cobertura debe quedar a 0
}

// ============================================================================
//  Líneas / polilíneas / círculos / triángulos (antialias por distancia)
// ============================================================================
void Painter::line(float ax, float ay, float bx, float by, u32 c, float width) {
    if ((c >> 24) == 0 || !finite4(ax, ay, bx, by) || !(width > 0.0f)) return;
    const float hw = 0.5f * width, R = hw + 1.0f;
    const float fy0 = max_(min_(ay, by) - R, static_cast<float>(cy0_));
    const float fy1 = min_(max_(ay, by) + R, static_cast<float>(cy1_));
    if (fy0 >= fy1) return;
    const float fx0 = max_(min_(ax, bx) - R, static_cast<float>(cx0_));
    const float fx1 = min_(max_(ax, bx) + R, static_cast<float>(cx1_));
    if (fx0 >= fx1) return;
    const int y0 = max_(cy0_, sat_floor(fy0)), y1 = min_(cy1_, sat_ceil(fy1));
    const float dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy;
    const float il2 = l2 > 1e-12f ? 1.0f / l2 : 0.0f;
    const u32 a256 = alpha256(c >> 24);
    for (int y = y0; y < y1; ++y) {
        const float py = static_cast<float>(y) + 0.5f;
        float xl, xr;
        if (!seg_row_range(ax, ay, bx, by, py, R, xl, xr)) continue;
        const int xa = max_(cx0_, sat_ceil(xl - 0.5f)), xb = min_(cx1_ - 1, sat_floor(xr - 0.5f));
        u32* row = fb_->row(y);
        for (int x = xa; x <= xb; ++x) {
            const float d = seg_dist(static_cast<float>(x) + 0.5f, py, ax, ay, dx, dy, il2);
            const float cov = clampf(hw + 0.5f - d, 0.0f, 1.0f);
            const u32 k = static_cast<u32>(cov * static_cast<float>(a256) + 0.5f);
            if (k) row[x] = blend_px(row[x], c, k);
        }
    }
}

void Painter::polyline(const Vec2* pts, int n, u32 c, float width) {
    if (n <= 0 || (c >> 24) == 0 || !(width > 0.0f)) return;
    if (n == 1) { fill_circle(pts[0].x, pts[0].y, 0.5f * width, c); return; }
    const float hw = 0.5f * width, R = hw + 1.0f;
    float bx0 = 1e30f, by0 = 1e30f, bx1 = -1e30f, by1 = -1e30f;
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(pts[i].x) || !std::isfinite(pts[i].y)) return;
        bx0 = min_(bx0, pts[i].x); bx1 = max_(bx1, pts[i].x);
        by0 = min_(by0, pts[i].y); by1 = max_(by1, pts[i].y);
    }
    const int y0 = max_(cy0_, sat_floor(by0 - R)), y1 = min_(cy1_, sat_ceil(by1 + R));
    const int x0 = max_(cx0_, sat_floor(bx0 - R)), x1 = min_(cx1_, sat_ceil(bx1 + R));
    if (y0 >= y1 || x0 >= x1) return;
    float* cov = t_cov.get(x1 - x0);                  // índice = x - x0
    const u32 a256 = alpha256(c >> 24);
    for (int y = y0; y < y1; ++y) {
        const float py = static_cast<float>(y) + 0.5f;
        int rx0 = x1, rx1 = x0 - 1;
        for (int i = 0; i + 1 < n; ++i) {
            const float ax = pts[i].x, ay = pts[i].y, bx = pts[i + 1].x, by = pts[i + 1].y;
            if (py < min_(ay, by) - R || py > max_(ay, by) + R) continue;
            float xl, xr;
            if (!seg_row_range(ax, ay, bx, by, py, R, xl, xr)) continue;
            const int xa = max_(x0, sat_ceil(xl - 0.5f)), xb = min_(x1 - 1, sat_floor(xr - 0.5f));
            if (xa > xb) continue;
            const float dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy;
            const float il2 = l2 > 1e-12f ? 1.0f / l2 : 0.0f;
            for (int x = xa; x <= xb; ++x) {
                const float d = seg_dist(static_cast<float>(x) + 0.5f, py, ax, ay, dx, dy, il2);
                const float cv = hw + 0.5f - d;
                float& cc = cov[x - x0];
                if (cv > cc) cc = cv > 1.0f ? 1.0f : cv;
            }
            rx0 = min_(rx0, xa); rx1 = max_(rx1, xb);
        }
        if (rx0 > rx1) continue;
        u32* row = fb_->row(y);
        // La curva puede cruzar la fila en puntos muy separados: se saltan de 8 en 8 los
        // tramos sin cobertura (VCMPPS + VMOVMSKPS) en vez de recorrer todo [rx0, rx1].
        const __m256 zero = _mm256_setzero_ps();
        for (int x = rx0; x <= rx1;) {
            if (x + 8 <= rx1 + 1) {
                float* cp = cov + (x - x0);
                const __m256 cv8 = _mm256_loadu_ps(cp);
                if (_mm256_movemask_ps(_mm256_cmp_ps(cv8, zero, _CMP_GT_OQ)) == 0) {
                    _mm256_storeu_ps(cp, zero);        // (puede haber negativos: se limpian)
                    x += 8;
                    continue;
                }
            }
            const float cv = cov[x - x0];
            cov[x - x0] = 0.0f;
            if (cv > 0.0f) {
                const u32 k = static_cast<u32>(cv * static_cast<float>(a256) + 0.5f);
                if (k) row[x] = blend_px(row[x], c, k);
            }
            ++x;
        }
    }
}

void Painter::fill_circle(float cx, float cy, float r, u32 c) {
    if ((c >> 24) == 0 || !finite4(cx, cy, r, 0.0f) || !(r > 0.0f)) return;
    const float ro = r + 0.5f;
    const int y0 = max_(cy0_, sat_floor(cy - ro)), y1 = min_(cy1_, sat_ceil(cy + ro));
    const u32 a = c >> 24, a256 = alpha256(a);
    const float ri = r - 0.75f;                       // radio "seguro" del tramo sólido
    for (int y = y0; y < y1; ++y) {
        const float py = static_cast<float>(y) + 0.5f, dy = py - cy, dy2 = dy * dy;
        if (dy2 >= ro * ro) continue;
        const float ho = std::sqrt(ro * ro - dy2);
        const int xa = max_(cx0_, sat_floor(cx - ho)), xb = min_(cx1_ - 1, sat_ceil(cx + ho));
        if (xa > xb) continue;
        int sa = xb + 1, sb = xb + 1;                   // tramo sólido [sa, sb)
        if (ri > 0.0f && dy2 < ri * ri) {
            const float hi = std::sqrt(ri * ri - dy2);
            sa = max_(xa, sat_ceil(cx - hi)); sb = min_(xb + 1, sat_floor(cx + hi));
            if (sb <= sa) { sa = sb = xb + 1; }
        }
        u32* row = fb_->row(y);
        for (int x = xa; x <= xb; ++x) {
            if (x == sa) { span_fill(row + sa, sb - sa, c); x = sb - 1; continue; }
            const float dx = static_cast<float>(x) + 0.5f - cx;
            const float cov = clampf(r + 0.5f - std::sqrt(dx * dx + dy2), 0.0f, 1.0f);
            const u32 k = static_cast<u32>(cov * static_cast<float>(a256) + 0.5f);
            if (k) row[x] = blend_px(row[x], c, k);
        }
    }
}

void Painter::circle(float cx, float cy, float r, u32 c, float width) {
    if ((c >> 24) == 0 || !finite4(cx, cy, r, width) || !(r > 0.0f) || !(width > 0.0f)) return;
    const float hw = 0.5f * width, ro = r + hw + 1.0f, rin = max_(r - hw - 1.0f, 0.0f);
    const int y0 = max_(cy0_, sat_floor(cy - ro)), y1 = min_(cy1_, sat_ceil(cy + ro));
    const u32 a256 = alpha256(c >> 24);
    for (int y = y0; y < y1; ++y) {
        const float py = static_cast<float>(y) + 0.5f, dy = py - cy, dy2 = dy * dy;
        if (dy2 >= ro * ro) continue;
        const float ho = std::sqrt(ro * ro - dy2);
        const float hi = dy2 < rin * rin ? std::sqrt(rin * rin - dy2) : -1.0f;   // agujero interior
        const int xa = max_(cx0_, sat_floor(cx - ho)), xb = min_(cx1_ - 1, sat_ceil(cx + ho));
        u32* row = fb_->row(y);
        for (int x = xa; x <= xb; ++x) {
            const float dx = static_cast<float>(x) + 0.5f - cx;
            if (hi > 0.0f && std::fabs(dx) < hi - 0.5f) { x = max_(x, sat_floor(cx + hi - 0.5f) - 1); continue; }
            const float d = std::sqrt(dx * dx + dy2);
            const float cov = clampf(min_(r + hw + 0.5f - d, d - (r - hw) + 0.5f), 0.0f, 1.0f);
            const u32 k = static_cast<u32>(cov * static_cast<float>(a256) + 0.5f);
            if (k) row[x] = blend_px(row[x], c, k);
        }
    }
}

void Painter::fill_triangle(Vec2 a, Vec2 b, Vec2 p, u32 c) {
    if ((c >> 24) == 0 || !finite4(a.x, a.y, b.x, b.y) || !finite4(p.x, p.y, 0.0f, 0.0f)) return;
    float area = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
    if (std::fabs(area) < 1e-6f) return;
    if (area < 0) { const Vec2 t = b; b = p; p = t; }
    // Aristas normalizadas: e(q) = distancia con signo (positiva dentro).
    const Vec2 v[3] = {a, b, p};
    float nx[3], ny[3], off[3];
    for (int i = 0; i < 3; ++i) {
        const Vec2 s = v[i], e = v[(i + 1) % 3];
        const float ex = e.x - s.x, ey = e.y - s.y, il = 1.0f / std::sqrt(ex * ex + ey * ey);
        nx[i] = -ey * il; ny[i] = ex * il;          // normal interior (orientación positiva en y-abajo)
        off[i] = -(nx[i] * s.x + ny[i] * s.y);
    }
    const float bx0 = min_(a.x, min_(b.x, p.x)) - 1, bx1 = max_(a.x, max_(b.x, p.x)) + 1;
    const float by0 = min_(a.y, min_(b.y, p.y)) - 1, by1 = max_(a.y, max_(b.y, p.y)) + 1;
    const int x0 = max_(cx0_, sat_floor(bx0)), x1 = min_(cx1_, sat_ceil(bx1));
    const int y0 = max_(cy0_, sat_floor(by0)), y1 = min_(cy1_, sat_ceil(by1));
    const u32 a256 = alpha256(c >> 24);
    for (int y = y0; y < y1; ++y) {
        const float py = static_cast<float>(y) + 0.5f;
        u32* row = fb_->row(y);
        for (int x = x0; x < x1; ++x) {
            const float px = static_cast<float>(x) + 0.5f;
            const float d0 = nx[0] * px + ny[0] * py + off[0];
            const float d1 = nx[1] * px + ny[1] * py + off[1];
            const float d2 = nx[2] * px + ny[2] * py + off[2];
            const float cov = clampf(min_(d0, min_(d1, d2)) + 0.5f, 0.0f, 1.0f);
            const u32 k = static_cast<u32>(cov * static_cast<float>(a256) + 0.5f);
            if (k) row[x] = blend_px(row[x], c, k);
        }
    }
}

void Painter::fill_area(const Vec2* pts, int n, float base_y, u32 c_top, u32 c_base) {
    if (n < 2 || !std::isfinite(base_y)) return;
    float ytop = base_y;
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(pts[i].x) || !std::isfinite(pts[i].y)) return;
        ytop = min_(ytop, pts[i].y);
    }
    const int x0 = max_(cx0_, sat_floor(pts[0].x)), x1 = min_(cx1_, sat_ceil(pts[n - 1].x));
    const int y0 = max_(cy0_, sat_floor(ytop)), y1 = min_(cy1_, sat_ceil(base_y));
    if (x0 >= x1 || y0 >= y1) return;
    // y de la curva en el centro de cada columna (búfer por hilo), recorrido monótono en x.
    float* cy = t_cov.get(x1 - x0);
    int seg = 0;
    for (int x = x0; x < x1; ++x) {
        const float px = static_cast<float>(x) + 0.5f;
        while (seg + 2 < n && pts[seg + 1].x < px) ++seg;
        const Vec2 a = pts[seg], b = pts[seg + 1];
        const float t = b.x > a.x ? clampf((px - a.x) / (b.x - a.x), 0.0f, 1.0f) : 0.0f;
        cy[x - x0] = a.y + (b.y - a.y) * t;
    }
    const float span = max_(base_y - ytop, 1.0f);
    for (int y = y0; y < y1; ++y) {
        const u32 c = lerp_color(c_top, c_base, (static_cast<float>(y) + 0.5f - ytop) / span);
        const u32 a256 = alpha256(c >> 24);
        if (!a256) continue;
        u32* row = fb_->row(y);
        const float yb = static_cast<float>(y + 1);
        for (int x = x0; x < x1; ++x) {
            const float cov = yb - cy[x - x0];
            if (cov <= 0.0f) continue;
            const u32 k = cov >= 1.0f ? a256 : static_cast<u32>(cov * static_cast<float>(a256) + 0.5f);
            if (k) row[x] = blend_px(row[x], c, k);
        }
    }
    std::memset(cy, 0, static_cast<usize>(x1 - x0) * sizeof(float));
}

// ============================================================================
//  Imágenes
// ============================================================================
void Painter::blit(const u32* src, int sw, int sh, int sstride, int dx, int dy, bool alpha, u32 opacity) {
    if (!src || sw <= 0 || sh <= 0 || opacity == 0) return;
    int x0 = cx0_, y0 = cy0_, x1 = cx1_, y1 = cy1_;
    intersect_into(x0, y0, x1, y1, {dx, dy, sw, sh});
    const int n = x1 - x0;
    if (n <= 0 || y0 >= y1) return;
    const u32 op256 = alpha256(min_(opacity, 255u));
    if (!alpha && opacity >= 255) {
        for (int y = y0; y < y1; ++y)
            std::memcpy(fb_->row(y) + x0, src + static_cast<usize>(y - dy) * static_cast<usize>(sstride) + (x0 - dx), static_cast<usize>(n) * 4);
        return;
    }
    const __m256i opv = _mm256_set1_epi32(static_cast<int>(op256));
    const __m256i force_opaque = _mm256_set1_epi32(alpha ? 0 : static_cast<int>(0xFF000000u));
    for (int y = y0; y < y1; ++y) {
        const u32* s = src + static_cast<usize>(y - dy) * static_cast<usize>(sstride) + (x0 - dx);
        u32* d = fb_->row(y) + x0;
        int i = 0;
        for (; i + 8 <= n; i += 8) {
            const __m256i sv = _mm256_or_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + i)), force_opaque);
            __m256i* dp = reinterpret_cast<__m256i*>(d + i);
            _mm256_storeu_si256(dp, blend_src_alpha8(sv, _mm256_loadu_si256(dp), opv));
        }
        for (; i < n; ++i) {
            const u32 sa = alpha ? (s[i] >> 24) : 255u;
            const u32 k = (alpha256(sa) * op256) >> 8;
            if (k) d[i] = blend_px(d[i], s[i], k);
        }
    }
}

void Painter::blit_scaled(const u32* src, int sw, int sh, int sstride, Rect dst, bool alpha) {
    dst = sat_rect(dst);
    if (!src || sw <= 0 || sh <= 0 || dst.w <= 0 || dst.h <= 0) return;
    int x0 = cx0_, y0 = cy0_, x1 = cx1_, y1 = cy1_;
    intersect_into(x0, y0, x1, y1, dst);
    if (x0 >= x1 || y0 >= y1) return;
    // Punto fijo 16.16: paso por píxel destino, muestreo en el centro.
    const u64 stepx = (static_cast<u64>(sw) << 16) / static_cast<u64>(dst.w);
    const u64 stepy = (static_cast<u64>(sh) << 16) / static_cast<u64>(dst.h);
    for (int y = y0; y < y1; ++y) {
        const u64 fy = static_cast<u64>(y - dst.y) * stepy + (stepy >> 1);
        const u32* s = src + static_cast<usize>(min_<u64>(fy >> 16, static_cast<u64>(sh - 1))) * static_cast<usize>(sstride);
        u32* d = fb_->row(y);
        u64 fx = static_cast<u64>(x0 - dst.x) * stepx + (stepx >> 1);
        for (int x = x0; x < x1; ++x, fx += stepx) {
            const u32 p = s[min_<u64>(fx >> 16, static_cast<u64>(sw - 1))];
            if (!alpha) d[x] = p | 0xFF000000u;
            else { const u32 a = p >> 24; if (a) d[x] = blend_px(d[x], p, alpha256(a)); }
        }
    }
}

// ============================================================================
//  Texto
// ============================================================================
int Painter::text_glyphs(int x, int y, const u8* gl, int n, u32 c, Font f) {
    const int gw = font_w(f), gh = font_h(f);
    const i64 xe = static_cast<i64>(x) + static_cast<i64>(n > 0 ? n : 0) * gw;
    const int x_end = static_cast<int>(clamp_<i64>(xe, -(1ll << 30), 1ll << 30));
    const u32 a = c >> 24;
    if (n <= 0 || a == 0) return x_end;
    if (y >= cy1_ || static_cast<i64>(y) + gh <= cy0_ || x >= cx1_ || xe <= cx0_) return x_end;
    const int r0 = max_(0, cy0_ - y), r1 = min_(gh, cy1_ - y);
    if (r0 >= r1) return x_end;
    const int i0 = x >= cx0_ ? 0 : static_cast<int>((static_cast<i64>(cx0_) - x) / gw);
    const int i1 = static_cast<int>(min_<i64>(n, (static_cast<i64>(cx1_) - x + gw - 1) / gw));
    const bool opaque = a == 255;
    const __m256i col = _mm256_set1_epi32(static_cast<int>(c));
    const ConstBlend B(c, alpha256(a));
    const u32 a256 = alpha256(a);
    const bool large = f == Font::Large;
    const usize stride = static_cast<usize>(fb_->stride);
    u32* base = fb_->row(y + r0);                    // fila y + r0 (válida: r0 ≥ cy0_ - y)
    for (int i = i0; i < i1; ++i) {
        const u32 g = gl[i];
        const int rr0 = max_(r0, static_cast<int>(large ? k_large.r0[g] : k_small.r0[g]));
        const int rr1 = min_(r1, static_cast<int>(large ? k_large.r1[g] : k_small.r1[g]));
        if (rr0 >= rr1) continue;                    // espacio u otro glifo vacío
        for (int part = 0; part < (large ? 2 : 1); ++part) {
            const int px = x + i * gw + part * 8;
            u32* d = base + static_cast<usize>(rr0 - r0) * stride;
            if (px >= cx0_ && px + 8 <= cx1_) {
                // Ruta rápida: los 8 píxeles dentro del recorte → 1 VPMASKMOVD por fila.
                d += px;
                for (int r = rr0; r < rr1; ++r, d += stride) {
                    const u32 bits = large ? (part ? (k_large.g[g][r] & 0xFFu) : (k_large.g[g][r] >> 8)) : k_small.g[g][r];
                    if (!bits) continue;
                    const __m256i m = _mm256_load_si256(reinterpret_cast<const __m256i*>(k_mask.m[bits]));
                    if (opaque) _mm256_maskstore_epi32(reinterpret_cast<int*>(d), m, col);
                    else _mm256_maskstore_epi32(reinterpret_cast<int*>(d), m, B(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(d))));
                }
            } else {
                // Glifo cortado por el borde del recorte: píxel a píxel (raro: ≤ 2 por línea).
                const int xa = max_(px, cx0_), xb = min_(px + 8, cx1_);
                for (int r = rr0; r < rr1; ++r, d += stride) {
                    const u32 bits = large ? (part ? (k_large.g[g][r] & 0xFFu) : (k_large.g[g][r] >> 8)) : k_small.g[g][r];
                    for (int xx = xa; xx < xb; ++xx)
                        if (bits & (0x80u >> (xx - px))) d[xx] = opaque ? c : blend_px(d[xx], c, a256);
                }
            }
        }
    }
    return x_end;
}

int Painter::text(int x, int y, const char* s, u32 c, Font f, int nbytes) {
    if (!s) return x;
    u8 buf[512];
    const usize len = nbytes < 0 ? std::strlen(s) : static_cast<usize>(nbytes);
    usize off = 0;
    while (off < len) {                              // trozos de ≤ 512 bytes (≤ 512 glifos)
        usize chunk = min_<usize>(len - off, 512);
        if (off + chunk < len)                       // no partir una secuencia UTF-8
            while (chunk > 1 && (static_cast<u8>(s[off + chunk]) & 0xC0) == 0x80) --chunk;
        const int ng = utf8_to_glyphs(s + off, static_cast<int>(chunk), buf, 512);
        x = text_glyphs(x, y, buf, ng, c, f);
        off += chunk;
    }
    return x;
}

int Painter::text_shadow(int x, int y, const char* s, u32 c, u32 sh, Font f) {
    x = sat_i(x); y = sat_i(y);
    text(x + 1, y + 1, s, sh, f);
    return text(x, y, s, c, f);
}

static CFD_INLINE void align_pos(Rect box, int tw, Font f, HAlign h, VAlign v, int& x, int& y) {
    const i64 bx = box.x, by = box.y, bw = box.w, bh = box.h;
    x = sat_i(h == HAlign::Left ? bx : (h == HAlign::Center ? bx + (bw - tw) / 2 : bx + bw - tw));
    y = sat_i(v == VAlign::Top ? by : (v == VAlign::Middle ? by + bh / 2 - font_center(f) : by + bh - font_h(f)));
}

void Painter::text_aligned(Rect box, const char* s, u32 c, HAlign h, VAlign v, Font f) {
    int x, y;
    align_pos(box, text_width(s, f), f, h, v, x, y);
    text(x, y, s, c, f);
}

// ============================================================================
//  DrawList
// ============================================================================
DrawList::DrawList() {
    cmds_.reserve(2048);
    glyphs_.reserve(32768);
    pts_.reserve(8192);
    images_.reserve(64);
    base_[0] = base_[1] = -k_inf;
    base_[2] = base_[3] = k_inf;
    std::memcpy(clip_, base_, sizeof clip_);
}

void DrawList::clear() {
    cmds_.clear(); glyphs_.clear(); pts_.clear(); images_.clear();
    depth_ = 0;
    std::memcpy(clip_, base_, sizeof clip_);
}

void DrawList::set_base_clip(Rect r) {
    base_[0] = r.x; base_[1] = r.y; base_[2] = r.x + r.w; base_[3] = r.y + r.h;
    std::memcpy(clip_, base_, sizeof clip_);
}

void DrawList::push_clip(Rect r) {
    CFD_CHECK(depth_ < 32, "DrawList: pila de recortes llena");
    std::memcpy(stack_[depth_++], clip_, sizeof clip_);
    int x0 = clip_[0], y0 = clip_[1], x1 = clip_[2], y1 = clip_[3];
    intersect_into(x0, y0, x1, y1, r);
    clip_[0] = x0; clip_[1] = y0; clip_[2] = x1; clip_[3] = y1;
}
void DrawList::pop_clip() {
    if (depth_ > 0) std::memcpy(clip_, stack_[--depth_], sizeof clip_);
}

DrawCmd& DrawList::push(CmdType t, int y0, int y1, u32 c0) {
    DrawCmd& c = cmds_.emplace_back();
    c.type = t;
    c.font = 0; c.corners = 0xF; c.flags = 0;
    c.c0 = c0; c.c1 = 0;
    std::memcpy(c.clip, clip_, sizeof clip_);
    c.y0 = max_(y0, clip_[1]);
    c.y1 = min_(y1, clip_[3]);
    c.off = c.n = 0;
    return c;
}

static CFD_INLINE void put_rect(DrawCmd& c, Rect r) {
    c.f[0] = static_cast<float>(r.x); c.f[1] = static_cast<float>(r.y);
    c.f[2] = static_cast<float>(r.w); c.f[3] = static_cast<float>(r.h);
}
static CFD_INLINE Rect get_rect(const DrawCmd& c) {
    return {static_cast<int>(c.f[0]), static_cast<int>(c.f[1]), static_cast<int>(c.f[2]), static_cast<int>(c.f[3])};
}

void DrawList::fill_rect(Rect r, u32 c) {
    r = sat_rect(r);
    if ((c >> 24) == 0 || r.w <= 0 || r.h <= 0 || culled(r.x, r.y, r.x + r.w, r.y + r.h)) return;
    put_rect(push(CmdType::FillRect, r.y, r.y + r.h, c), r);
}
void DrawList::rect(Rect r, u32 c, int t) {
    r = sat_rect(r);
    if ((c >> 24) == 0 || r.w <= 0 || r.h <= 0 || t <= 0 || culled(r.x, r.y, r.x + r.w, r.y + r.h)) return;
    DrawCmd& k = push(CmdType::Rect, r.y, r.y + r.h, c);
    // Grosor sujeto: float(INT_MAX) = 2^31 y volver a int sería UB al reproducir.
    put_rect(k, r); k.f[4] = static_cast<float>(min_(t, 1 << 24));
}
void DrawList::fill_round_rect(Rect r, float rad, u32 c, u32 corners) {
    r = sat_rect(r);
    if ((c >> 24) == 0 || r.w <= 0 || r.h <= 0 || culled(r.x, r.y, r.x + r.w, r.y + r.h)) return;
    DrawCmd& k = push(CmdType::RoundRect, r.y, r.y + r.h, c);
    put_rect(k, r); k.f[4] = rad; k.corners = static_cast<u8>(corners);
}
void DrawList::round_rect(Rect r, float rad, u32 c, int t, u32 corners) {
    r = sat_rect(r);
    if ((c >> 24) == 0 || r.w <= 0 || r.h <= 0 || culled(r.x, r.y, r.x + r.w, r.y + r.h)) return;
    DrawCmd& k = push(CmdType::RoundRectOutline, r.y, r.y + r.h, c);
    put_rect(k, r); k.f[4] = rad; k.f[5] = static_cast<float>(clamp_(t, 0, 1 << 24)); k.corners = static_cast<u8>(corners);
}
void DrawList::shadow(Rect r, float rad, float blur, u32 c, bool skip_inside) {
    r = sat_rect(r);
    const int e = static_cast<int>(std::ceil(clampf(blur, 0.5f, 256.0f))) + 1;
    if ((c >> 24) == 0 || r.w <= 0 || r.h <= 0 || culled(r.x - e, r.y - e, r.x + r.w + e, r.y + r.h + e)) return;
    DrawCmd& k = push(CmdType::Shadow, r.y - e, r.y + r.h + e, c);
    put_rect(k, r); k.f[4] = rad; k.f[5] = clampf(blur, 0.5f, 256.0f); k.flags = skip_inside ? 1 : 0;
}
void DrawList::gradient_v(Rect r, u32 top, u32 bottom) {
    r = sat_rect(r);
    if (r.w <= 0 || r.h <= 0 || culled(r.x, r.y, r.x + r.w, r.y + r.h)) return;
    DrawCmd& k = push(CmdType::GradientV, r.y, r.y + r.h, top);
    put_rect(k, r); k.c1 = bottom;
}
void DrawList::gradient_h(Rect r, u32 left, u32 right) {
    r = sat_rect(r);
    if (r.w <= 0 || r.h <= 0 || culled(r.x, r.y, r.x + r.w, r.y + r.h)) return;
    DrawCmd& k = push(CmdType::GradientH, r.y, r.y + r.h, left);
    put_rect(k, r); k.c1 = right;
}
void DrawList::line(float x0, float y0, float x1, float y1, u32 c, float width) {
    if ((c >> 24) == 0 || !finite4(x0, y0, x1, y1) || !(width > 0.0f)) return;
    const float R = 0.5f * width + 2.0f;
    const int bx0 = sat_floor(min_(x0, x1) - R), bx1 = sat_ceil(max_(x0, x1) + R);
    const int by0 = sat_floor(min_(y0, y1) - R), by1 = sat_ceil(max_(y0, y1) + R);
    if (culled(bx0, by0, bx1, by1)) return;
    DrawCmd& k = push(CmdType::Line, by0, by1, c);
    k.f[0] = x0; k.f[1] = y0; k.f[2] = x1; k.f[3] = y1; k.f[4] = width;
}
void DrawList::polyline(const Vec2* pts, int n, u32 c, float width) {
    if (n <= 0 || (c >> 24) == 0 || !(width > 0.0f)) return;
    float bx0 = 1e30f, by0 = 1e30f, bx1 = -1e30f, by1 = -1e30f;
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(pts[i].x) || !std::isfinite(pts[i].y)) return;
        bx0 = min_(bx0, pts[i].x); bx1 = max_(bx1, pts[i].x); by0 = min_(by0, pts[i].y); by1 = max_(by1, pts[i].y);
    }
    const float R = 0.5f * width + 2.0f;
    const int ix0 = sat_floor(bx0 - R), ix1 = sat_ceil(bx1 + R), iy0 = sat_floor(by0 - R), iy1 = sat_ceil(by1 + R);
    if (culled(ix0, iy0, ix1, iy1)) return;
    DrawCmd& k = push(CmdType::Polyline, iy0, iy1, c);
    k.off = static_cast<u32>(pts_.size()); k.n = static_cast<u32>(n); k.f[4] = width;
    pts_.insert(pts_.end(), pts, pts + n);
}
void DrawList::fill_circle(float cx, float cy, float r, u32 c) {
    if ((c >> 24) == 0 || !finite4(cx, cy, r, 0.0f) || !(r > 0.0f)) return;
    const int x0 = sat_floor(cx - r - 1), x1 = sat_ceil(cx + r + 1), y0 = sat_floor(cy - r - 1), y1 = sat_ceil(cy + r + 1);
    if (culled(x0, y0, x1, y1)) return;
    DrawCmd& k = push(CmdType::FillCircle, y0, y1, c);
    k.f[0] = cx; k.f[1] = cy; k.f[2] = r;
}
void DrawList::circle(float cx, float cy, float r, u32 c, float width) {
    if ((c >> 24) == 0 || !finite4(cx, cy, r, width) || !(r > 0.0f)) return;
    const float R = r + 0.5f * width + 1.5f;
    const int x0 = sat_floor(cx - R), x1 = sat_ceil(cx + R), y0 = sat_floor(cy - R), y1 = sat_ceil(cy + R);
    if (culled(x0, y0, x1, y1)) return;
    DrawCmd& k = push(CmdType::Circle, y0, y1, c);
    k.f[0] = cx; k.f[1] = cy; k.f[2] = r; k.f[3] = width;
}
void DrawList::fill_triangle(Vec2 a, Vec2 b, Vec2 p, u32 c) {
    if ((c >> 24) == 0 || !finite4(a.x, a.y, b.x, b.y) || !finite4(p.x, p.y, 0.0f, 0.0f)) return;
    const int x0 = sat_floor(min_(a.x, min_(b.x, p.x)) - 1), x1 = sat_ceil(max_(a.x, max_(b.x, p.x)) + 1);
    const int y0 = sat_floor(min_(a.y, min_(b.y, p.y)) - 1), y1 = sat_ceil(max_(a.y, max_(b.y, p.y)) + 1);
    if (culled(x0, y0, x1, y1)) return;
    DrawCmd& k = push(CmdType::Triangle, y0, y1, c);
    k.f[0] = a.x; k.f[1] = a.y; k.f[2] = b.x; k.f[3] = b.y; k.f[4] = p.x; k.f[5] = p.y;
}
void DrawList::fill_area(const Vec2* pts, int n, float base_y, u32 c_top, u32 c_base) {
    if (n < 2 || !std::isfinite(base_y)) return;
    float bx0 = 1e30f, by0 = base_y, bx1 = -1e30f, by1 = base_y;
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(pts[i].x) || !std::isfinite(pts[i].y)) return;
        bx0 = min_(bx0, pts[i].x); bx1 = max_(bx1, pts[i].x); by0 = min_(by0, pts[i].y); by1 = max_(by1, pts[i].y);
    }
    const int ix0 = sat_floor(bx0), ix1 = sat_ceil(bx1), iy0 = sat_floor(by0), iy1 = sat_ceil(by1) + 1;
    if (culled(ix0, iy0, ix1, iy1)) return;
    DrawCmd& k = push(CmdType::Area, iy0, iy1, c_top);
    k.c1 = c_base; k.f[4] = base_y;
    k.off = static_cast<u32>(pts_.size()); k.n = static_cast<u32>(n);
    pts_.insert(pts_.end(), pts, pts + n);
}
void DrawList::blit(const u32* src, int sw, int sh, int sstride, int dx, int dy, bool alpha, u32 opacity) {
    if (!src || sw <= 0 || sh <= 0) return;
    dx = sat_i(dx); dy = sat_i(dy);
    const int x1 = sat_i(static_cast<i64>(dx) + sw), y1 = sat_i(static_cast<i64>(dy) + sh);
    if (culled(dx, dy, x1, y1)) return;
    DrawCmd& k = push(CmdType::Blit, dy, y1, opacity);
    k.off = static_cast<u32>(images_.size()); k.flags = alpha ? 1 : 0;
    k.f[0] = static_cast<float>(dx); k.f[1] = static_cast<float>(dy);
    images_.push_back({src, sw, sh, sstride});
}
void DrawList::blit_scaled(const u32* src, int sw, int sh, int sstride, Rect dst, bool alpha) {
    dst = sat_rect(dst);
    if (!src || sw <= 0 || sh <= 0 || dst.w <= 0 || dst.h <= 0 || culled(dst.x, dst.y, dst.x + dst.w, dst.y + dst.h)) return;
    DrawCmd& k = push(CmdType::BlitScaled, dst.y, dst.y + dst.h, 0);
    put_rect(k, dst);
    k.off = static_cast<u32>(images_.size()); k.flags = alpha ? 1 : 0;
    images_.push_back({src, sw, sh, sstride});
}
int DrawList::text(int x, int y, const char* s, u32 c, Font f, int nbytes) {
    if (!s) return x;
    const int nb = nbytes < 0 ? static_cast<int>(std::strlen(s)) : nbytes;
    const int ng = utf8_count(s, nb);
    x = sat_i(x); y = sat_i(y);
    const int xe = sat_i(static_cast<i64>(x) + static_cast<i64>(ng) * font_w(f));
    if ((c >> 24) == 0 || ng == 0 || culled(x, y, xe, y + font_h(f))) return xe;
    DrawCmd& k = push(CmdType::Text, y, y + font_h(f), c);
    k.font = static_cast<u8>(f);
    k.f[0] = static_cast<float>(x); k.f[1] = static_cast<float>(y);
    const usize off = glyphs_.size();
    glyphs_.resize(off + static_cast<usize>(ng) + 8);            // +8: holgura del SWAR
    const int got = utf8_to_glyphs(s, nb, glyphs_.data() + off, ng);
    glyphs_.resize(off + static_cast<usize>(got));
    k.off = static_cast<u32>(off); k.n = static_cast<u32>(got);
    return xe;
}
int DrawList::text_shadow(int x, int y, const char* s, u32 c, u32 shadow, Font f) {
    const usize before = cmds_.size();
    const int r = text(x, y, s, c, f);
    if (cmds_.size() > before) {
        DrawCmd& k = cmds_.back();
        k.type = CmdType::TextShadow; k.c1 = shadow;
        k.y1 = min_(k.y1 + 1, clip_[3]);
    }
    return r;
}
void DrawList::text_aligned(Rect box, const char* s, u32 c, HAlign h, VAlign v, Font f) {
    int x, y;
    align_pos(box, text_width(s, f), f, h, v, x, y);
    text(x, y, s, c, f);
}

void DrawList::exec_one(Painter& p, const DrawCmd& c) const {
    switch (c.type) {
        case CmdType::FillRect: p.fill_rect(get_rect(c), c.c0); break;
        case CmdType::Rect: p.rect(get_rect(c), c.c0, static_cast<int>(c.f[4])); break;
        case CmdType::RoundRect: p.fill_round_rect(get_rect(c), c.f[4], c.c0, c.corners); break;
        case CmdType::RoundRectOutline: p.round_rect(get_rect(c), c.f[4], c.c0, static_cast<int>(c.f[5]), c.corners); break;
        case CmdType::Shadow: p.shadow(get_rect(c), c.f[4], c.f[5], c.c0, c.flags & 1); break;
        case CmdType::GradientV: p.gradient_v(get_rect(c), c.c0, c.c1); break;
        case CmdType::GradientH: p.gradient_h(get_rect(c), c.c0, c.c1); break;
        case CmdType::Line: p.line(c.f[0], c.f[1], c.f[2], c.f[3], c.c0, c.f[4]); break;
        case CmdType::Polyline: p.polyline(pts_.data() + c.off, static_cast<int>(c.n), c.c0, c.f[4]); break;
        case CmdType::Area: p.fill_area(pts_.data() + c.off, static_cast<int>(c.n), c.f[4], c.c0, c.c1); break;
        case CmdType::FillCircle: p.fill_circle(c.f[0], c.f[1], c.f[2], c.c0); break;
        case CmdType::Circle: p.circle(c.f[0], c.f[1], c.f[2], c.c0, c.f[3]); break;
        case CmdType::Triangle: p.fill_triangle({c.f[0], c.f[1]}, {c.f[2], c.f[3]}, {c.f[4], c.f[5]}, c.c0); break;
        case CmdType::Blit: {
            const Image& im = images_[c.off];
            p.blit(im.p, im.w, im.h, im.stride, static_cast<int>(c.f[0]), static_cast<int>(c.f[1]), c.flags & 1, c.c0);
            break;
        }
        case CmdType::BlitScaled: {
            const Image& im = images_[c.off];
            p.blit_scaled(im.p, im.w, im.h, im.stride, get_rect(c), c.flags & 1);
            break;
        }
        case CmdType::Text:
            p.text_glyphs(static_cast<int>(c.f[0]), static_cast<int>(c.f[1]), glyphs_.data() + c.off, static_cast<int>(c.n), c.c0, static_cast<Font>(c.font));
            break;
        case CmdType::TextShadow:
            p.text_glyphs(static_cast<int>(c.f[0]) + 1, static_cast<int>(c.f[1]) + 1, glyphs_.data() + c.off, static_cast<int>(c.n), c.c1, static_cast<Font>(c.font));
            p.text_glyphs(static_cast<int>(c.f[0]), static_cast<int>(c.f[1]), glyphs_.data() + c.off, static_cast<int>(c.n), c.c0, static_cast<Font>(c.font));
            break;
    }
}

void DrawList::execute(Painter& p) const {
    const Rect saved = p.clip();
    for (const DrawCmd& c : cmds_) {
        p.set_clip(saved);
        p.push_clip({c.clip[0], c.clip[1], c.clip[2] - c.clip[0], c.clip[3] - c.clip[1]});
        if (!p.clip_empty()) exec_one(p, c);
        p.pop_clip();
    }
    p.set_clip(saved);
}

void DrawList::render(Framebuffer& fb, const DrawList* const* lists, int nlists, int bands) {
    if (fb.h <= 0 || fb.w <= 0 || nlists <= 0) return;
    usize total = 0;
    for (int i = 0; i < nlists; ++i) total += lists[i]->cmds_.size();
    if (total == 0) return;
    if (bands <= 0) bands = pool().size() * 2;
    bands = clamp_(bands, 1, max_(1, fb.h / 16));
    const int band_h = (fb.h + bands - 1) / bands;
    parallel_for(0, bands, 1, [&](i64 lo, i64 hi) {
        Painter p(fb);
        for (i64 b = lo; b < hi; ++b) {
            const int y0 = static_cast<int>(b) * band_h, y1 = min_(fb.h, y0 + band_h);
            if (y0 >= y1) continue;
            for (int li = 0; li < nlists; ++li) {
                const DrawList& L = *lists[li];
                for (const DrawCmd& c : L.cmds_) {
                    if (c.y1 <= y0 || c.y0 >= y1) continue;
                    const int ty0 = max_(c.clip[1], y0), ty1 = min_(c.clip[3], y1);
                    p.set_clip({c.clip[0], ty0, c.clip[2] - c.clip[0], ty1 - ty0});
                    if (!p.clip_empty()) L.exec_one(p, c);
                }
            }
        }
    });
}

} // namespace cfd::render
