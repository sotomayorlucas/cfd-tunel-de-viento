// ============================================================================
//  tests/test_drm.cpp — pruebas de la ruta DRM/i915 cruda (src/gpu/drm_i915.*).
//
//  1. Codificación de comandos MI_* / XY_FAST_COPY_BLT (sin GPU, siempre se ejecuta).
//  2. Si /dev/dri/renderD128 es i915 y accesible: lote con MI_STORE_DATA_IMM en cada
//     motor rcs0/bcs0/ccs0 verificado desde la CPU, varios lotes en un BO con
//     batch_start_offset, y un blit de 4 MiB verificado. Sin GPU/permiso → "SKIP" (éxito).
//
//  g++ -std=c++23 -O3 -march=native -Isrc -pthread tests/test_drm.cpp
//      src/gpu/drm_i915.cpp -o build/drm/test_drm
// ============================================================================
#include "gpu/drm_i915.hpp"

#include <cerrno>
#include <cstdio>
#include <cstring>

using namespace cfd::gpu::drm;

static int g_fail = 0, g_checks = 0;
#define CHECK(c)                                                                          \
    do {                                                                                  \
        ++g_checks;                                                                       \
        if (!(c)) { ++g_fail; std::printf("    FALLO %s:%d: %s\n", __FILE__, __LINE__, #c); } \
    } while (0)

static void test_encoding() {
    std::printf("[codificación de comandos]\n");
    u32 buf[64] = {};
    Batch b{buf, 0, 64};
    b.mi_store_dword(0x0000'1234'5678'9ABCull, 0xDEADBEEF);
    CHECK(buf[0] == 0x10000002u);   // MI_STORE_DATA_IMM, 4 dwords
    CHECK(buf[1] == 0x56789ABCu && buf[2] == 0x1234u && buf[3] == 0xDEADBEEFu);
    b.n = 0;
    b.mi_store_reg(0x2358, 0x1000);
    CHECK(buf[0] == 0x12000002u && buf[1] == 0x2358u && buf[2] == 0x1000u && buf[3] == 0);
    b.n = 0;
    b.mi_copy_mem_mem(0x2000, 0x1000);
    CHECK(buf[0] == 0x17000003u && buf[1] == 0x2000u && buf[3] == 0x1000u);
    b.n = 0;
    b.mi_flush_dw();
    CHECK(buf[0] == 0x13000002u && b.n == 4);
    b.n = 0;
    b.xy_fast_copy(0x3000, 0x4000, 4096, 1024, 16);
    CHECK(buf[0] == 0x50800008u && b.n == 10);   // cliente 2D, opcode 0x42, 10 dwords
    CHECK(buf[1] == ((3u << 24) | 4096u) && buf[3] == ((16u << 16) | 1024u));
    CHECK(buf[4] == 0x3000u && buf[7] == 4096u && buf[8] == 0x4000u);
    b.n = 0;
    b.mi_noop();
    b.end();
    CHECK(buf[1] == 0x05000000u && b.n == 2 && b.bytes() % 8 == 0);   // BBE + ya alineado
    b.n = 0;
    b.end();
    CHECK(b.n == 2 && buf[1] == 0);   // BBE + MI_NOOP de relleno
    u32 small[2];
    Batch o{small, 0, 2};
    o.mi_store_dword(0, 0);
    CHECK(o.overflow());
    CHECK(engine_mmio_base(0, 0) == 0x2000 && engine_mmio_base(1, 0) == 0x22000 && engine_mmio_base(4, 0) == 0x1a000);
}

static bool test_gpu() {
    std::printf("[GPU por ioctl crudos]\n");
    Device d;
    if (!d.open("/dev/dri/renderD128")) {
        std::printf("  SKIP: %s\n", d.err.c_str());
        return false;
    }
    std::vector<EngineInfo> all, map;
    CHECK(d.query_engines(all) == 0);
    for (u16 cls : {u16(0), u16(1), u16(4)})
        for (auto& e : all) if (e.cls == cls && e.instance == 0) map.push_back(e);
    CHECK(!map.empty());
    int freq = 0;
    CHECK(d.getparam(51 /* I915_PARAM_CS_TIMESTAMP_FREQUENCY */, &freq) == 0 && freq > 0);
    u32 vm = 0, ctx = 0;
    CHECK(d.vm_create(&vm) == 0);
    if (int r = d.context_create(map, vm, &ctx); r) {
        std::printf("  SKIP: CONTEXT_CREATE_EXT: %s\n", errstr(r).c_str());
        d.vm_destroy(vm);
        return false;
    }
    Bo res, bat;
    CHECK(res.create(d, 64 << 10) == 0 && res.mmap(MapMode::WC) == 0);
    CHECK(bat.create(d, 64 << 10) == 0 && bat.mmap(MapMode::WC) == 0);
    res.gpu_va = 0x1'0000'0000ull;
    bat.gpu_va = 0x1'0100'0000ull;
    std::memset(res.map, 0, res.size);

    // Un lote por motor: 32 SDI + SRM del timestamp.
    for (size_t e = 0; e < map.size(); ++e) {
        Batch b{bat.u32p(), 0, u32(bat.size / 4)};
        for (u32 i = 0; i < 32; ++i) b.mi_store_dword(res.gpu_va + 4 * (32 * e + i), 0xAB000000u + u32(e) * 1000 + i);
        b.mi_store_reg(engine_mmio_base(map[e].cls, 0) + kRingTimestamp, res.gpu_va + 4096 + 8 * e);
        b.end();
        ExecObj o[2] = {{&res, true}, {&bat, false}};
        const int r1 = execbuf(d, ctx, u32(e), o, 2, b.bytes());
        const int r2 = r1 ? r1 : gem_wait(d, bat.handle, 2'000'000'000);
        CHECK(r1 == 0 && r2 == 0);
        int good = 0;
        for (u32 i = 0; i < 32; ++i) good += res.u32p()[32 * e + i] == 0xAB000000u + u32(e) * 1000 + i;
        CHECK(good == 32);
        CHECK(res.u32p()[1024 + 2 * e] != 0);   // timestamp escrito
        std::printf("  %s0: %d/32 dwords, ts %u\n", engine_class_name(map[e].cls), good, res.u32p()[1024 + 2 * e]);
    }

    // Varios lotes en un BO con batch_start_offset, encadenados sin esperar.
    {
        const u32 K = 16;
        u32 len = 0;
        for (u32 i = 0; i < K; ++i) {
            Batch b{bat.u32p() + 16 * i, 0, 16};
            b.mi_store_dword(res.gpu_va + 8192 + 4 * i, 0xCD00u + i);
            b.end();
            len = b.bytes();
        }
        ExecObj o[2] = {{&res, true}, {&bat, false}};
        int err = 0;
        for (u32 i = 0; i < K; ++i) err |= execbuf(d, ctx, 0, o, 2, len, i * 64);
        CHECK(err == 0);
        CHECK(gem_wait(d, bat.handle, 2'000'000'000) == 0);
        int good = 0;
        for (u32 i = 0; i < K; ++i) good += res.u32p()[2048 + i] == 0xCD00u + i;
        CHECK(good == int(K));
    }

    // Blit de 4 MiB en bcs0.
    int bcs = -1;
    for (size_t i = 0; i < map.size(); ++i) if (map[i].cls == 1) bcs = int(i);
    if (bcs >= 0) {
        const u64 bytes = 4ull << 20;
        Bo src, dst;
        CHECK(src.create(d, bytes) == 0 && src.mmap(MapMode::WC) == 0);
        CHECK(dst.create(d, bytes) == 0 && dst.mmap(MapMode::WC) == 0);
        src.gpu_va = 0x2'0000'0000ull;
        dst.gpu_va = 0x3'0000'0000ull;
        for (u32 i = 0; i < bytes / 4; ++i) src.u32p()[i] = i * 2654435761u;
        Batch b{bat.u32p(), 0, u32(bat.size / 4)};
        b.xy_fast_copy(dst.gpu_va, src.gpu_va, 4096, 1024, u32(bytes / 4096));
        b.mi_flush_dw();
        b.end();
        ExecObj o[4] = {{&res, true}, {&src, false}, {&dst, true}, {&bat, false}};
        CHECK(execbuf(d, ctx, u32(bcs), o, 4, b.bytes()) == 0);
        CHECK(gem_wait(d, bat.handle, 2'000'000'000) == 0);
        u32 bad = 0;
        for (u32 i = 0; i < bytes / 4; i += 7) bad += dst.u32p()[i] != i * 2654435761u;
        CHECK(bad == 0);
        std::printf("  bcs0: blit de 4 MiB %s\n", bad ? "INCORRECTO" : "correcto");
    }
    d.context_destroy(ctx);
    d.vm_destroy(vm);
    return true;
}

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    test_encoding();
    const bool gpu = test_gpu();
    std::printf("\n%s: %d comprobaciones, %d fallos%s\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail,
                gpu ? "" : " (parte GPU omitida)");
    return g_fail ? 1 : 0;
}
