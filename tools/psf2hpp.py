#!/usr/bin/env python3
"""Conversor PSF1/PSF2 (consolefonts) -> src/render/font_data.hpp (Latin-1 0x20..0xFF).
Uso unico en desarrollo; el binario final no depende de este script ni de fuentes del sistema."""
import gzip, struct, sys

def load(path):
    d = gzip.open(path).read()
    if d[:4] == b'\x72\xb5\x4a\x86':
        _, ver, hsz, flags, n, bpg, h, w = struct.unpack('<IIIIIIII', d[:32])
        glyphs = d[hsz:hsz + n * bpg]
        m = {}
        for gi, e in enumerate(d[hsz + n * bpg:].split(b'\xff')[:n]):
            for ch in e.split(b'\xfe')[0].decode('utf-8', errors='ignore'):
                m.setdefault(ord(ch), gi)
        return w, h, bpg, glyphs, m
    mode, csz = d[2], d[3]
    n = 512 if mode & 1 else 256
    glyphs = d[4:4 + n * csz]
    tab = d[4 + n * csz:]
    m, gi = {}, 0
    if mode & 2:
        for v in struct.unpack('<%dH' % (len(tab) // 2), tab):
            if v == 0xFFFF: gi += 1; continue
            if v != 0xFFFE: m.setdefault(v, gi)
    return 8, csz, csz, glyphs, m

def main():
    out = ['// Generado por tools/psf2hpp.py a partir de fuentes Terminus (Lat15, SIL OFL). No editar.',
           '#pragma once', '#include <cstdint>', '', 'namespace font {', '']
    for name, path in [('small', '/usr/share/consolefonts/Lat15-TerminusBold16.psf.gz'),
                       ('large', '/usr/share/consolefonts/Lat15-TerminusBold32x16.psf.gz')]:
        w, h, bpg, glyphs, m = load(path)
        rb = (w + 7) // 8
        ctype = 'uint8_t' if rb == 1 else 'uint16_t'
        out += [f'// {path.split("/")[-1]}: {w}x{h}, Latin-1 (0x20..0xFF), filas de {rb*8} bits MSB-first',
                f'inline constexpr int k_{name}_w = {w};', f'inline constexpr int k_{name}_h = {h};',
                f'alignas(64) inline constexpr {ctype} k_{name}_glyphs[224][{h}] = {{']
        for cp in range(0x20, 0x100):
            gi = m.get(cp, m.get(ord('?')))
            g = glyphs[gi * bpg:(gi + 1) * bpg]
            rows = [(('0x%02X' % g[r]) if rb == 1 else ('0x%04X' % ((g[2*r] << 8) | g[2*r+1]))) for r in range(h)]
            out.append('  {' + ','.join(rows) + '},')
        out += ['};', '']
    out.append('} // namespace font')
    open(sys.argv[1] if len(sys.argv) > 1 else 'src/render/font_data.hpp', 'w').write('\n'.join(out) + '\n')

if __name__ == '__main__':
    main()
