// ============================================================================
//  core/mem.hpp — memoria alineada, páginas enormes (THP) y arena lineal.
//
//  Buffer<T>: arreglo alineado a 64 B (o 2 MiB para buffers grandes) con
//  madvise(MADV_HUGEPAGE) → menos fallos de TLB en los recorridos del LBM.
//  Arena: asignador "bump" para temporales por cuadro (reset O(1)).
// ============================================================================
#pragma once

#include "config.hpp"
#include <cstring>
#include <new>
#include <type_traits>
#include <utility>
#include <sys/mman.h>

namespace cfd {

// Reserva alineada. Para tamaños >= 4 MiB alinea a 2 MiB y pide Transparent Huge Pages.
CFD_INLINE void* aligned_alloc_bytes(usize bytes, usize align = 64) {
    constexpr usize k_huge = 2u << 20;
    if (bytes >= (4u << 20)) align = k_huge;
    const usize rounded = (bytes + align - 1) & ~(align - 1);
    void* p = std::aligned_alloc(align, rounded ? rounded : align);
    CFD_CHECK(p != nullptr, "aligned_alloc: sin memoria");
    if (align == k_huge) ::madvise(p, rounded, MADV_HUGEPAGE);
    return p;
}
CFD_INLINE void aligned_free(void* p) { std::free(p); }

// Arreglo dinámico alineado, sólo para tipos triviales. No inicializa salvo que se pida.
template <class T>
class Buffer {
    static_assert(std::is_trivially_copyable_v<T>, "Buffer<T> sólo para tipos triviales");
public:
    Buffer() = default;
    explicit Buffer(usize n, bool zero = false) { resize(n, zero); }
    ~Buffer() { release(); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& o) noexcept : p_(std::exchange(o.p_, nullptr)), n_(std::exchange(o.n_, 0)) {}
    Buffer& operator=(Buffer&& o) noexcept {
        if (this != &o) { release(); p_ = std::exchange(o.p_, nullptr); n_ = std::exchange(o.n_, 0); }
        return *this;
    }
    void resize(usize n, bool zero = false) {
        release();
        n_ = n;
        if (n) {
            p_ = static_cast<T*>(aligned_alloc_bytes(n * sizeof(T)));
            if (zero) std::memset(static_cast<void*>(p_), 0, n * sizeof(T));
        }
    }
    void fill(const T& v) { for (usize i = 0; i < n_; ++i) p_[i] = v; }
    void zero() { if (n_) std::memset(static_cast<void*>(p_), 0, n_ * sizeof(T)); }
    void release() { if (p_) aligned_free(p_); p_ = nullptr; n_ = 0; }
    CFD_INLINE T* data() { return CFD_ASSUME_ALIGNED(p_, 64); }
    CFD_INLINE const T* data() const { return CFD_ASSUME_ALIGNED(p_, 64); }
    CFD_INLINE usize size() const { return n_; }
    CFD_INLINE bool empty() const { return n_ == 0; }
    CFD_INLINE T& operator[](usize i) { CFD_DASSERT(i < n_); return p_[i]; }
    CFD_INLINE const T& operator[](usize i) const { CFD_DASSERT(i < n_); return p_[i]; }
    T* begin() { return p_; }
    T* end() { return p_ + n_; }
    const T* begin() const { return p_; }
    const T* end() const { return p_ + n_; }
private:
    T* p_ = nullptr;
    usize n_ = 0;
};

// Arena lineal (bump allocator). reset() libera todo en O(1).
class Arena {
public:
    explicit Arena(usize capacity = 64u << 20) : buf_(capacity) {}
    template <class T>
    T* alloc(usize n) {
        static_assert(std::is_trivially_copyable_v<T>);
        const usize align = alignof(T) < 64 ? 64 : alignof(T);
        usize off = (top_ + align - 1) & ~(align - 1);
        CFD_CHECK(off + n * sizeof(T) <= buf_.size(), "Arena agotada");
        top_ = off + n * sizeof(T);
        return reinterpret_cast<T*>(buf_.data() + off);
    }
    void reset() { top_ = 0; }
    usize used() const { return top_; }
private:
    Buffer<unsigned char> buf_;
    usize top_ = 0;
};

// Valor rellenado a una línea de caché: evita "false sharing" en acumuladores por hilo.
// (alignas(64) redondea sizeof a múltiplo de 64, así que no hace falta relleno explícito)
template <class T>
struct alignas(64) Padded {
    T value{};
};

} // namespace cfd
