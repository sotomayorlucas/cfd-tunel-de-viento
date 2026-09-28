// ============================================================================
//  core/threadpool.hpp — pool de hilos persistente sin std::function,
//  afinado para CPUs híbridas Intel (Core Ultra 7 155H: 6 P + 8 E + 2 LP-E).
//
//  * Hilos creados una sola vez; se despiertan con un contador de generación
//    (spin corto con PAUSE y luego std::atomic::wait = futex en Linux).
//  * parallel_for con planificación dinámica por trozos (contador atómico):
//    imprescindible en CPUs híbridas donde un reparto estático deja a los
//    núcleos P esperando a los E.
//  * Barrera "sin rezagados": un trabajador debe ENTRAR al trabajo con un CAS
//    mientras está abierto; el llamador cierra el trabajo al terminar su parte
//    y sólo espera a los que entraron. Un hilo desplanificado por el SO (p.ej.
//    compilaciones o VMs en segundo plano) no bloquea el dispatch.
//  * Borrado de tipos con puntero a función + void* (cero asignaciones).
//  * El hilo llamador participa como trabajador 0.
//  * Llamadas anidadas desde un trabajador se ejecutan en serie (sin bloqueo).
// ============================================================================
#pragma once

#include "config.hpp"
#include <atomic>
#include <type_traits>
#include <thread>
#include <vector>

namespace cfd {

// Topología de la CPU (híbrida Intel: P-cores, E-cores, LP-E cores).
struct CpuTopology {
    std::vector<int> order;   // CPUs lógicas en orden de preferencia (ver detect_topology)
    int n_pcores = 0, n_ecores = 0, n_lpe = 0;
    int n_primary = 0;        // núcleos físicos P + E "principales" (sin HT ni LP-E)
    int recommended = 0;      // hilos recomendados por defecto
};
// Orden: 1 hilo por P-core físico → E-cores → hermanos HT de P-cores → LP-E.
// Así, pedir N hilos siempre toma las N mejores CPUs lógicas.
CpuTopology detect_topology();

class ThreadPool {
public:
    static ThreadPool& instance();

    // n = 0 → topología recomendada (todas menos LP-E). Idempotente.
    // Llamar a set_affinity() ANTES de start() para fijar hilos a CPUs.
    void start(int n = 0);
    // pin=true fija el hilo t a cpu_order[t] (o a la CPU t si la lista es corta).
    void set_affinity(bool pin, std::vector<int> cpu_order = {});
    void stop();
    int size() const { return nthreads_; }            // incluye al hilo llamador
    // Índice del hilo actual en [0, size()) (0 = principal), -1 fuera del pool.
    // Único entre hilos que ejecutan a la vez → válido para scratch por hilo.
    static int worker_index();

    // f(lo, hi) sobre [begin, end) en trozos de `grain` (≤0 → automático).
    template <class F>
    void parallel_for(i64 begin, i64 end, i64 grain, F&& f) {
        if (end <= begin) return;
        const i64 n = end - begin;
        if (grain <= 0) grain = max_i64(1, n / (static_cast<i64>(nthreads_) * 8));
        if (nthreads_ <= 1 || n <= grain || t_in_job_) { f(begin, end); return; }
        struct Ctx {
            alignas(64) std::atomic<i64> next;
            alignas(64) i64 end, grain;
            std::remove_reference_t<F>* fn;   // F puede deducirse como T& (callable lvalue)
        } ctx;
        ctx.next.store(begin, std::memory_order_relaxed);
        ctx.end = end; ctx.grain = grain; ctx.fn = &f;
        dispatch([](void* c, int) {
            Ctx& x = *static_cast<Ctx*>(c);
            for (;;) {
                const i64 lo = x.next.fetch_add(x.grain, std::memory_order_relaxed);
                if (lo >= x.end) break;
                const i64 hi = lo + x.grain < x.end ? lo + x.grain : x.end;
                (*x.fn)(lo, hi);
            }
        }, &ctx);
    }

    // f(slot, nslots) exactamente una vez por slot en [0, nslots), en paralelo.
    // nslots = size() por defecto. Un mismo hilo puede ejecutar varios slots
    // (secuencialmente): usar `slot` para indexar acumuladores parciales.
    template <class F>
    void run_slots(F&& f, int nslots = 0) {
        if (nslots <= 0) nslots = nthreads_;
        parallel_for(0, nslots, 1, [&](i64 lo, i64 hi) {
            for (i64 s = lo; s < hi; ++s) f(static_cast<int>(s), nslots);
        });
    }

    ~ThreadPool() { stop(); }

private:
    using JobFn = void (*)(void*, int);
    ThreadPool() = default;
    void dispatch(JobFn fn, void* ctx);
    void worker_main(int tid, u32 seen);
    void pin_threads();
    static constexpr i64 max_i64(i64 a, i64 b) { return a > b ? a : b; }

    static constexpr u64 k_open = 1ull << 31;
    static constexpr u64 k_count_mask = k_open - 1;

    alignas(64) std::atomic<u32> generation_{0};   // despertador (futex)
    alignas(64) std::atomic<u64> state_{0};        // [gen:32 | abierto:1 | entrados:31]
    alignas(64) std::atomic<u32> finished_{0};     // trabajadores que terminaron
    alignas(64) JobFn job_fn_ = nullptr;
    void* job_ctx_ = nullptr;
    u32 gen_counter_ = 0;                          // sólo lo escribe el llamador
    std::atomic<bool> stop_{false};
    int nthreads_ = 1;
    std::vector<std::thread> threads_;
    bool pin_ = false;
    std::vector<int> cpu_order_;
    static thread_local bool t_in_job_;
    static thread_local int t_index_;
};

// Atajos globales.
inline ThreadPool& pool() { return ThreadPool::instance(); }
template <class F> CFD_INLINE void parallel_for(i64 b, i64 e, i64 grain, F&& f) { pool().parallel_for(b, e, grain, static_cast<F&&>(f)); }

} // namespace cfd
