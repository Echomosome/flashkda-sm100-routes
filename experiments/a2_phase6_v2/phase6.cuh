#pragma once
// A2：只迁移 Phase 6 的矩阵乘。C16、Phase 1–5、状态舍入边界均保留。
// 本版使用原来的 MN_INTER 共享布局，不再生成 packed A / packed B 两份副本。
// 这里的位域按 CUDA 13.0 / PTX 9.0 描述；尚需 B300 编译、SASS 和完整对拍。
#include <cstdint>
#ifndef C1_PHASE6_HMMA_CONTROL
#define C1_PHASE6_HMMA_CONTROL 0
#endif
namespace c1_phase6 {
template<class BF16> struct Storage {
    alignas(128) BF16 u[16 * 128];  // U 仍在寄存器生成，只写一次供 CTA 级 MMA 读取。
    alignas(8) uint64_t barrier;
    uint32_t tmem;
};
__device__ inline uint32_t address(void const* p) {
    return uint32_t(__cvta_generic_to_shared(p));
}
// MN-major, BF16：offset(m,k)=m%8+8*(k%8)+128*(m/8)+64*(k/8)。
// LBO=64*sizeof(BF16)=128B；SBO=128*sizeof(BF16)=256B；没有 swizzle。
// 编译后另用 layout_errors() 对 pinned CuTe 的全部 2048 个坐标核验。
__device__ inline uint64_t descriptor_mn(void const* p) {
    return ((address(p) & 0x3ffffu) >> 4) | (uint64_t(8) << 16)
         | (uint64_t(16) << 32) | (uint64_t(1) << 46);
}
constexpr uint32_t idesc = 0x08218490u;  // M128 N128 K16 BF16；A/B transpose bits 15/16=1。

template<class BF16, class Barrier>
__device__ inline void init(Storage<BF16>& s, Barrier& sync, int tid) {
    if constexpr (!C1_PHASE6_HMMA_CONTROL) {
        if (tid < 32) {  // alloc / relinquish 是整个 warp 的 collective。
            asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0],128;"
                         ::"r"(address(&s.tmem)):"memory");
            asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;":::"memory");
        }
        if (tid == 0) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"(address(&s.barrier)):"memory");
            asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
        }
    }
    sync.arrive_and_wait();  // 只等 128 个 compute 线程；LOAD/STORE warp 不参与。
}

template<class BF16, class Barrier>
__device__ inline void issue(Storage<BF16>& s, void const* kr, Barrier& sync, int tid) {
    // 同步 U 的所有写入，并保证 Phase 1–5 对旧状态的读取已经结束。
    asm volatile("fence.proxy.async.shared::cta;":::"memory");
    sync.arrive_and_wait();
    if constexpr (!C1_PHASE6_HMMA_CONTROL) {
        asm volatile("tcgen05.fence::after_thread_sync;":::"memory");
        if (tid == 0) {
            asm volatile("{ .reg .pred p; setp.ne.u32 p,0,0; "
                "tcgen05.mma.cta_group::1.kind::f16 [%0],%1,%2,%3,{%4,%4,%4,%4},p; }"
                ::"r"(s.tmem),"l"(descriptor_mn(kr)),"l"(descriptor_mn(s.u)),"r"(idesc),"r"(0):"memory");
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                         ::"r"(address(&s.barrier)):"memory");
        }
    }
}
template<class BF16>
__device__ inline void wait(Storage<BF16>& s, int chunk) {
    if constexpr (!C1_PHASE6_HMMA_CONTROL) {
        for (int attempt = 0;; ++attempt) {
            uint32_t done;
            asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p,[%1],%2;"
                         " selp.u32 %0,1,0,p; }":"=r"(done):"r"(address(&s.barrier)),"r"(chunk & 1):"memory");
            if (done) break;
            if (attempt == 1000000) asm volatile("trap;");
        }
        asm volatile("tcgen05.fence::after_thread_sync;":::"memory");
    }
}

// .16x256b.x2 恰好返回 SM80 m16n16 的 C 片段：每线程 8 个 float。
// r[a+2*b+4*d] -> (row=lane/4+8*b, col=2*(lane%4)+a+8*d)。
// warp w 仅访问 TMEM 行 [32*w,32*w+31]；不能沿用官方按 value 列分 warp 的分工。
template<class Frag>
__device__ inline void load_fragment(uint32_t addr, Frag& out) {
    static_assert(decltype(cute::size(out))::value == 8, "16x16 C fragment expected");
    float* f = &out(0);
    asm volatile("tcgen05.ld.sync.aligned.16x256b.x2.b32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8];"
                 :"=f"(f[0]),"=f"(f[1]),"=f"(f[2]),"=f"(f[3]),
                  "=f"(f[4]),"=f"(f[5]),"=f"(f[6]),"=f"(f[7]):"r"(addr):"memory");
    asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory");
}
template<class BF16>
__device__ inline void finish(Storage<BF16>& s, int tid) {
    if constexpr (!C1_PHASE6_HMMA_CONTROL) {
        // caller 已执行最后一次 compute_barrier，各 warp 的 TMEM 读取全部完成。
        asm volatile("tcgen05.fence::after_thread_sync;":::"memory");
        if (tid == 0) asm volatile("mbarrier.inval.shared::cta.b64 [%0];"::"r"(address(&s.barrier)):"memory");
        if (tid < 32) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,128;"
                                   ::"r"(s.tmem):"memory");
    }
}
}  // namespace c1_phase6
