#pragma once
// Route A: Phase 6 inside the real recurrence.  K1, Phases 1-5, CHUNK=16 and
// the BF16 state rounding boundaries are unchanged.
//
// The packed layouts and PTX below are the ones measured in the standalone
// Phase-6 experiment (run 20260910T125328470967Z).  The new U staging, CuTe
// hand-off and named barrier are only validated by the full-KDA run.
#include <cstdint>
#ifndef C1_PHASE6_HMMA_CONTROL
#define C1_PHASE6_HMMA_CONTROL 0
#endif

namespace c1_phase6 {
// A[key, token], B_transposed[value, token]; every element is BF16.
__host__ __device__ constexpr int packed(int row, int k) {
    return (row / 8) * 128 + (k / 8) * 64 + (row % 8) * 8 + k % 8;
}
__device__ inline uint32_t address(void const* p) {
    return uint32_t(__cvta_generic_to_shared(p));
}
__device__ inline uint64_t descriptor(void const* p) {
    return ((address(p) & 0x3ffffu) >> 4) | (uint64_t(8) << 16)
         | (uint64_t(16) << 32) | (uint64_t(1) << 46);
}
constexpr uint32_t idesc = 0x08200490u;  // M128,N128,K16; BF16/BF16 -> FP32

template<class BF16> struct Storage {
    // U keeps the original VOLayout shape; A/B are packed for the measured
    // descriptor layout.
    alignas(128) BF16 u[16 * 128];
    alignas(128) BF16 a[128 * 16];
    alignas(128) BF16 bt[128 * 16];
    alignas(8) uint64_t barrier;
    uint32_t tmem;
};

template<class BF16, class Barrier>
__device__ inline void init(Storage<BF16>& s, Barrier& sync, int tid) {
    if constexpr (!C1_PHASE6_HMMA_CONTROL) {
        if (tid < 32) {  // alloc/relinquish is a warp collective: all 32 lanes.
            asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0],128;"
                         ::"r"(address(&s.tmem)):"memory");
            asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;":::"memory");
        }
        if (tid == 0) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"(address(&s.barrier)):"memory");
            asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
        }
    }
    // Only the 128 compute threads participate; do not use __syncthreads,
    // which would wait for all 192 threads including the I/O warps.
    sync.arrive_and_wait();
}

__device__ inline void wait_mma(uint32_t bar, int phase) {
    for (int attempt = 0;; ++attempt) {
        uint32_t done;
        asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p,[%1],%2;"
                     " selp.u32 %0,1,0,p; }":"=r"(done):"r"(bar),"r"(phase):"memory");
        if (done) return;
        if (attempt == 1000000) asm volatile("trap;");  // fail the process, never hang forever
    }
}

__device__ inline void load16(uint32_t addr, float (&out)[16]) {
    uint32_t r[16];
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},[%16];"
      :"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
       "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15])
      :"r"(addr):"memory");
    asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory");
    #pragma unroll
    for (int j = 0; j < 16; ++j) out[j] = __uint_as_float(r[j]);
}

// The control group pays the same U staging and A/B packing but keeps HMMA,
// so layout/synchronisation cost can be separated from the instruction change.
template<class BF16, class State, class Gate>
__device__ inline void hmma(Storage<BF16>& s, State state, Gate g, int tid) {
    int lane = tid % 32, warp = tid / 32;
    uint32_t b[4][2];
    #pragma unroll
    for (int nb = 0; nb < 4; ++nb) {
        uint32_t addr = address(s.bt + packed(warp * 32 + nb * 8 + lane % 8, ((lane / 8) % 2) * 8));
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2];"
                     :"=r"(b[nb][0]),"=r"(b[nb][1]):"r"(addr):"memory");
    }
    #pragma unroll 1
    for (int mt = 0; mt < 128; mt += 16) {
        uint32_t a[4];
        uint32_t addr = address(s.a + packed(mt + lane % 8 + ((lane / 8) % 2) * 8, (lane / 16) * 8));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
                     :"=r"(a[0]),"=r"(a[1]),"=r"(a[2]),"=r"(a[3]):"r"(addr):"memory");
        #pragma unroll
        for (int nb = 0; nb < 4; ++nb) {
            float d[4] = {0,0,0,0};
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                         "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
              :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
              :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[nb][0]),"r"(b[nb][1]));
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                int row = mt + lane / 4 + (p / 2) * 8;
                int col = warp * 32 + nb * 8 + (lane % 4) * 2 + p % 2;
                state(row, col) = BF16(fmaf(float(state(row, col)), g(row), d[p]));
            }
        }
    }
}

template<class BF16, class U, class KR, class State, class Gate, class Barrier>
__device__ inline void update(Storage<BF16>& s, U u, KR kr, State state,
                              Gate g, Barrier& sync, int tid, int chunk_index) {
    // The caller's four warps have just written the real U into s.u; wait
    // before reading a column that may not be complete yet.
    sync.arrive_and_wait();
    for (int i = tid; i < 128 * 16; i += 128) {
        int row = i / 16, token = i % 16;
        s.a[packed(row, token)] = kr(token, row);
        s.bt[packed(row, token)] = u(token, row);
    }
    asm volatile("fence.proxy.async.shared::cta;":::"memory");
    sync.arrive_and_wait();
    if constexpr (C1_PHASE6_HMMA_CONTROL) {
        hmma(s, state, g, tid);
    } else {
        asm volatile("tcgen05.fence::after_thread_sync;":::"memory");
        if (tid == 0) {
            asm volatile("{ .reg .pred p; setp.ne.u32 p,0,0; "
              "tcgen05.mma.cta_group::1.kind::f16 [%0],%1,%2,%3,{%4,%4,%4,%4},p; }"
              ::"r"(s.tmem),"l"(descriptor(s.a)),"l"(descriptor(s.bt)),"r"(idesc),"r"(0):"memory");
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                         ::"r"(address(&s.barrier)):"memory");
        }
        wait_mma(address(&s.barrier), chunk_index & 1);
        asm volatile("tcgen05.fence::after_thread_sync;":::"memory");
        #pragma unroll 1
        for (int col = 0; col < 128; col += 16) {
            float values[16];
            // The address is warp-uniform; the lane mapping comes from
            // tcgen05.ld, so the lane must not be added to the address.
            load16(s.tmem + (uint32_t((tid / 32) * 32) << 16) + col, values);
            #pragma unroll
            for (int j = 0; j < 16; ++j)
                state(tid, col + j) = BF16(fmaf(float(state(tid, col + j)), g(tid), values[j]));
        }
        asm volatile("tcgen05.fence::before_thread_sync;":::"memory");
    }
    // The caller's original compute_barrier synchronises the state write-back
    // before the input stage is released and the output is committed.
}

template<class BF16>
__device__ inline void finish(Storage<BF16>& s, int tid) {
    if constexpr (!C1_PHASE6_HMMA_CONTROL) {
        // The final compute_barrier has already ordered all cross-warp TMEM reads.
        asm volatile("tcgen05.fence::after_thread_sync;":::"memory");
        if (tid == 0) asm volatile("mbarrier.inval.shared::cta.b64 [%0];"::"r"(address(&s.barrier)):"memory");
        if (tid < 32) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,128;"
                                   ::"r"(s.tmem):"memory");
    }
}
}  // namespace c1_phase6
