#include "threadpool.hpp"
#include <immintrin.h>
#include <x86intrin.h>
#include <pthread.h>
#include <sched.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace cfd {

thread_local bool ThreadPool::t_in_job_ = false;
thread_local int ThreadPool::t_index_ = -1;

ThreadPool& ThreadPool::instance() {
    static ThreadPool p;
    return p;
}

int ThreadPool::worker_index() { return t_index_; }

void ThreadPool::start(int n) {
    if (!threads_.empty()) return;
    if (n <= 0) n = detect_topology().recommended;
    if (n <= 0) n = static_cast<int>(std::thread::hardware_concurrency());
    if (n <= 0) n = 1;
    nthreads_ = n;
    t_index_ = 0;
    stop_.store(false, std::memory_order_relaxed);
    threads_.reserve(static_cast<usize>(n - 1));
    // La generación inicial se captura AQUÍ: si un hilo arrancara tarde y la leyera
    // después del primer dispatch, se saltaría ese trabajo (inofensivo con la barrera
    // de entrada, pero así el primer dispatch ya cuenta con todos).
    const u32 gen0 = generation_.load(std::memory_order_acquire);
    for (int t = 1; t < n; ++t) threads_.emplace_back([this, t, gen0] { worker_main(t, gen0); });
    if (pin_) pin_threads();
}

void ThreadPool::stop() {
    if (threads_.empty()) return;
    stop_.store(true, std::memory_order_release);
    generation_.fetch_add(1, std::memory_order_release);
    generation_.notify_all();
    for (auto& th : threads_) th.join();
    threads_.clear();
    nthreads_ = 1;
}

void ThreadPool::dispatch(JobFn fn, void* ctx) {
    job_fn_ = fn;
    job_ctx_ = ctx;
    finished_.store(0, std::memory_order_relaxed);
    const u32 g = ++gen_counter_;
    state_.store((static_cast<u64>(g) << 32) | k_open, std::memory_order_release);
    generation_.store(g, std::memory_order_release);      // publica job/estado
    generation_.notify_all();
    t_in_job_ = true;
    fn(ctx, 0);
    t_in_job_ = false;
    // Cierra el trabajo: nadie más puede entrar. Sólo esperamos a los que entraron.
    const u64 s = state_.fetch_and(~k_open, std::memory_order_acq_rel);
    const u32 entered = static_cast<u32>(s & k_count_mask);
    for (u32 spin = 0; finished_.load(std::memory_order_acquire) != entered; ++spin) {
        if (spin < 8192) _mm_pause();
        else std::this_thread::yield();
    }
}

void ThreadPool::worker_main(int tid, u32 seen) {
    t_index_ = tid;
    for (;;) {
        // Fase 1: spin con PAUSE (~140 ciclos por PAUSE en Redwood Cove / Crestmont).
        // Medido en este Core Ultra 7 155H: UMONITOR/UMWAIT (WAITPKG) despierta MÁS lento
        // que PAUSE (73 µs vs 3 µs por dispatch con 12 hilos), así que no se usa.
        u32 g;
        const u64 spin_until = __rdtsc() + 1500000;   // ~0.5 ms antes de ir al futex
        while ((g = generation_.load(std::memory_order_acquire)) == seen && __rdtsc() < spin_until) _mm_pause();
        // Fase 2: dormir en futex.
        while (g == seen) { generation_.wait(seen, std::memory_order_acquire); g = generation_.load(std::memory_order_acquire); }
        seen = g;
        if (stop_.load(std::memory_order_acquire)) return;
        // Entrada: sólo si el trabajo g sigue abierto. Si ya se cerró (llegamos tarde)
        // o hay uno más nuevo, volvemos arriba y el bucle lo detecta.
        u64 s = state_.load(std::memory_order_acquire);
        bool entered = false;
        while ((s >> 32) == g && (s & k_open)) {
            if (state_.compare_exchange_weak(s, s + 1, std::memory_order_acq_rel, std::memory_order_acquire)) { entered = true; break; }
        }
        if (!entered) continue;
        t_in_job_ = true;
        job_fn_(job_ctx_, tid);
        t_in_job_ = false;
        finished_.fetch_add(1, std::memory_order_release);
    }
}

// Lee una lista de CPUs estilo "0-11,14" de sysfs.
static std::vector<int> read_cpu_list(const char* path) {
    std::vector<int> v;
    FILE* f = std::fopen(path, "r");
    if (!f) return v;
    char buf[256] = {};
    const bool ok = std::fgets(buf, sizeof buf, f) != nullptr;
    std::fclose(f);
    if (!ok) return v;
    for (char* p = buf; *p && *p != '\n';) {
        const int a = static_cast<int>(std::strtol(p, &p, 10));
        int b = a;
        if (*p == '-') b = static_cast<int>(std::strtol(p + 1, &p, 10));
        for (int c = a; c <= b; ++c) v.push_back(c);
        if (*p == ',') ++p; else if (*p && *p != '\n') break;
    }
    return v;
}
static int read_int_file(const char* path, int fallback) {
    FILE* f = std::fopen(path, "r");
    if (!f) return fallback;
    long v = fallback;
    if (std::fscanf(f, "%ld", &v) != 1) v = fallback;
    std::fclose(f);
    return static_cast<int>(v);
}

CpuTopology detect_topology() {
    CpuTopology t;
    const int ncpu = static_cast<int>(std::thread::hardware_concurrency());
    std::vector<int> pcores = read_cpu_list("/sys/devices/cpu_core/cpus");
    std::vector<int> ecores = read_cpu_list("/sys/devices/cpu_atom/cpus");
    if (pcores.empty() && ecores.empty()) {               // CPU no híbrida: orden natural
        for (int c = 0; c < ncpu; ++c) t.order.push_back(c);
        t.n_primary = ncpu; t.recommended = ncpu;
        return t;
    }
    char path[128];
    // P-cores: primero un hilo por núcleo físico, luego los hermanos HT.
    std::vector<int> p_first, p_sibling;
    std::vector<int> seen_core;
    for (int c : pcores) {
        std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/core_id", c);
        const int core = read_int_file(path, c);
        bool dup = false;
        for (int s : seen_core) dup |= (s == core);
        (dup ? p_sibling : p_first).push_back(c);
        if (!dup) seen_core.push_back(core);
    }
    // E-cores: los de frecuencia máxima más baja son los LP-E del tile SoC (sin L3 compartida).
    int emax = 0;
    for (int c : ecores) {
        std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", c);
        emax = std::max(emax, read_int_file(path, 0));
    }
    std::vector<int> e_main, e_lp;
    for (int c : ecores) {
        std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", c);
        const int f = read_int_file(path, emax);
        (f < emax * 8 / 10 ? e_lp : e_main).push_back(c);
    }
    t.order.insert(t.order.end(), p_first.begin(), p_first.end());
    t.order.insert(t.order.end(), e_main.begin(), e_main.end());
    t.order.insert(t.order.end(), p_sibling.begin(), p_sibling.end());
    t.order.insert(t.order.end(), e_lp.begin(), e_lp.end());
    t.n_pcores = static_cast<int>(p_first.size());
    t.n_ecores = static_cast<int>(e_main.size());
    t.n_lpe = static_cast<int>(e_lp.size());
    t.n_primary = t.n_pcores + t.n_ecores;
    // Por defecto: todo menos los LP-E (medido: con ellos el dispatch pasa de ~3 µs a >30 µs).
    t.recommended = static_cast<int>(t.order.size()) - t.n_lpe;
    return t;
}

// Afinidad: hilo t → CPU lógica cpu_order[t]. En el Core Ultra 7 155H Linux numera
// primero los núcleos P (0-11, con HT), luego los E (12-19) y los LP-E del tile SoC
// (20-21, sin acceso a la L3 del tile de cómputo). Fijar hilos evita migraciones.
void ThreadPool::pin_threads() {
    auto pin = [](pthread_t th, int cpu) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        pthread_setaffinity_np(th, sizeof(set), &set);
    };
    const int ncpu = static_cast<int>(std::thread::hardware_concurrency());
    if (ncpu <= 0) return;
    pin(pthread_self(), cpu_order_.empty() ? 0 : cpu_order_[0] % ncpu);
    for (usize i = 0; i < threads_.size(); ++i) {
        const int logical = static_cast<int>(i + 1);
        const int cpu = logical < static_cast<int>(cpu_order_.size()) ? cpu_order_[static_cast<usize>(logical)] : logical;
        pin(threads_[i].native_handle(), cpu % ncpu);
    }
}

void ThreadPool::set_affinity(bool pin, std::vector<int> cpu_order) {
    pin_ = pin;
    cpu_order_ = std::move(cpu_order);
}

} // namespace cfd
