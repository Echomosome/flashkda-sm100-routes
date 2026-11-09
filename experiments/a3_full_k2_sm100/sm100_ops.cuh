#pragma once
// A3：转置整个 K2 的矩阵乘，使 value=128 成为 M，而 token=16 成为 N。
// CHUNK、BF16 舍入和 K1 均保留。这里的 control 只用于比较新数据流的开销。
#include <cstdint>
#ifndef C1_TRANSPOSE_HMMA_CONTROL
#define C1_TRANSPOSE_HMMA_CONTROL 0
#endif
namespace c1_transpose {
// 同数据流的 HMMA 控制组：用于区分转置/落 SMEM 与新指令本身的收益。
// A 始终 K-major；只有 Phase6 的 B 是 MN-major 的 KR 转置别名。
template<int K, bool TransB, class TA, class TB, class Frag>
__device__ __forceinline__ void control_fragment(TA const& a,TB const& b,
                                                int mi,int ni,int lane,Frag& out) {
    auto tiled=make_tiled_mma(MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>{},
                             Layout<Shape<_1,_1>>{},Tile<_16,_16,_16>{});
    auto thr=tiled.get_slice(lane);
    auto ca=make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N,cutlass::bfloat16_t>{},tiled);
    auto ac=ca.get_slice(lane);
    auto cb=[] {
        if constexpr (TransB)
            return make_tiled_copy_B(Copy_Atom<SM75_U16x8_LDSM_T,cutlass::bfloat16_t>{},
              make_tiled_mma(MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>{},Layout<Shape<_1,_1>>{},Tile<_16,_16,_16>{}));
        else
            return make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N,cutlass::bfloat16_t>{},
              make_tiled_mma(MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>{},Layout<Shape<_1,_1>>{},Tile<_16,_16,_16>{}));
    }();
    auto bc=cb.get_slice(lane);
    clear(out);
    #pragma unroll
    for (int kk=0;kk<K/16;++kk) {
        Tensor ab=local_tile(a,make_shape(Int<16>{},Int<16>{}),make_coord(mi,kk));
        Tensor bb=local_tile(b,make_shape(Int<16>{},Int<16>{}),make_coord(ni,kk));
        auto af=thr.partition_fragment_A(ab); auto bf=thr.partition_fragment_B(bb);
        copy(ca,ac.partition_S(ab),ac.retile_D(af));
        copy(cb,bc.partition_S(bb),bc.retile_D(bf));
        cute::gemm(thr,af(_,_,Int<0>{}),bf(_,_,Int<0>{}),out);
    }
}
template<class BF16> struct Storage {
    // 合并 KD/QD 后一次做 M128,N32，减少一半 Phase1 MMA issue 和 state 读取。
    // 两者是同一 state 的右乘，不是拼接独立 head，也不会引入跨 head 交叉项。
    alignas(128) BF16 kq[32 * 128];
    // RHS 和 U 的生存期不重叠，共用 4 KiB，不另外打包 KR 或 state。
    alignas(128) BF16 rhs_u[128 * 16];
    alignas(8) uint64_t barrier;
    uint32_t tmem;
};
__device__ __forceinline__ uint32_t address(void const* p) {
    return uint32_t(__cvta_generic_to_shared(p));
}
// K_INTER(Rows,K)：k%8+8*(m%8)+64*(m/8)+8*Rows*(k/8)。
// 每个 K16 子块都以实际 layout 的起点构造 descriptor，不靠猜测连续行主序。
template<int Rows> __device__ __forceinline__ uint64_t desc_k(void const* p) {
    return ((address(p)&0x3ffffu)>>4) | (uint64_t(Rows)<<16)
         | (uint64_t(8)<<32) | (uint64_t(1)<<46);
}
// KR 转置后的 [key,token]：MN_INTER，和 A2 已验证的 descriptor 相同。
__device__ __forceinline__ uint64_t desc_mn(void const* p) {
    return ((address(p)&0x3ffffu)>>4) | (uint64_t(8)<<16)
         | (uint64_t(16)<<32) | (uint64_t(1)<<46);
}
template<int N, bool TransB=false>
constexpr uint32_t idesc = (8u<<24) | (uint32_t(N/8)<<17)
                        | (uint32_t(TransB)<<16) | 0x490u;

template<class BF16, class Barrier>
__device__ __forceinline__ void init(Storage<BF16>& s, Barrier& sync, int tid) {
    if constexpr (!C1_TRANSPOSE_HMMA_CONTROL) {
        // 同时容纳 Phase4 的 16 列和 Phase6 的 128 列，向上取 256 列。
        // alloc/relinquish/dealloc 必须整个 warp 一致执行，不能只让 lane0 调用。
        if (tid<32) {
            asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0],256;"
                         ::"r"(address(&s.tmem)):"memory");
            asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;":::"memory");
        }
        if (tid==0) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"(address(&s.barrier)):"memory");
            asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
        }
    }
    sync.arrive_and_wait();
}
template<class Barrier>
__device__ __forceinline__ void publish(Barrier& sync) {
    // 寄存器→SMEM 的 RHS/U 写完后，才交给 tcgen05 的 async proxy。
    if constexpr (!C1_TRANSPOSE_HMMA_CONTROL)
        asm volatile("tcgen05.fence::before_thread_sync;":::"memory");
    asm volatile("fence.proxy.async.shared::cta;":::"memory");
    sync.arrive_and_wait();
    if constexpr (!C1_TRANSPOSE_HMMA_CONTROL)
        asm volatile("tcgen05.fence::after_thread_sync;":::"memory");
}
template<int N, bool TransB=false>
__device__ __forceinline__ void mma(uint32_t dst, uint64_t a, uint64_t b, bool accumulate) {
    asm volatile("{ .reg .pred p; setp.ne.u32 p,%4,0; "
        "tcgen05.mma.cta_group::1.kind::f16 [%0],%1,%2,%3,{%5,%5,%5,%5},p; }"
        ::"r"(dst),"l"(a),"l"(b),"r"(idesc<N,TransB>),"r"(int(accumulate)),"r"(0):"memory");
}
template<class BF16> __device__ __forceinline__ void commit(Storage<BF16>& s) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                 ::"r"(address(&s.barrier)):"memory");
}
template<class BF16> __device__ __forceinline__ void wait(Storage<BF16>& s, int& phase) {
    if constexpr (!C1_TRANSPOSE_HMMA_CONTROL) {
        for (int tries=0;;++tries) {
            uint32_t done;
            asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p,[%1],%2;"
                         " selp.u32 %0,1,0,p; }":"=r"(done):"r"(address(&s.barrier)),"r"(phase):"memory");
            if (done) break;
            if (tries==1000000) asm volatile("trap;");
        }
        phase ^= 1;
        asm volatile("tcgen05.fence::after_thread_sync;":::"memory");
    }
}
template<class Frag> __device__ __forceinline__ void load16(uint32_t addr, Frag& out) {
    static_assert(decltype(cute::size(out))::value==8,"16x16 C fragment required");
    float* f=&out(0);
    asm volatile("tcgen05.ld.sync.aligned.16x256b.x2.b32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8];"
                 :"=f"(f[0]),"=f"(f[1]),"=f"(f[2]),"=f"(f[3]),
                  "=f"(f[4]),"=f"(f[5]),"=f"(f[6]),"=f"(f[7]):"r"(addr):"memory");
    asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory");
}
// load128 由生成器展开 64 个显式 PTX 输出寄存器。一次等待读取一个 16×128 条带。
__device__ __forceinline__ void load128(uint32_t addr, float (&f)[64]);

template<class BF16> __device__ __forceinline__ void finish(Storage<BF16>& s,int tid) {
    if constexpr (!C1_TRANSPOSE_HMMA_CONTROL) {
        asm volatile("tcgen05.fence::after_thread_sync;":::"memory");
        if (tid==0) asm volatile("mbarrier.inval.shared::cta.b64 [%0];"::"r"(address(&s.barrier)):"memory");
        if (tid<32) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,256;"
                                ::"r"(s.tmem):"memory");
    }
}
} // namespace c1_transpose
