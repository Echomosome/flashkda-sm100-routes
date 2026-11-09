            // A3：S 的物理布局仍为 [value,key]。整个 K2 的输出朝向统一为 value 行。
            // 四个 compute warp 各负责 32 个 value，与 TMEM 的 warp 行访问范围一致。
            {
            using UTLayout = decltype(tile_to_shape(
                GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{},
                make_shape(Int<128>{},Int<16>{}),LayoutLeft{}));
            Tensor ut = make_tensor(make_smem_ptr(shared_storage.transposed.rhs_u),UTLayout{});
            Tensor vt = make_tensor(make_smem_ptr(shared_storage.input[load_stage].v.begin()),TransposedVOLayout{});
            Tensor ot = make_tensor(make_smem_ptr(shared_storage.output[out_stage].out.begin()),TransposedVOLayout{});
            Tensor kr_t = make_tensor(make_smem_ptr(shared_storage.input[load_stage].k_restored.begin()),TransposedMMALayout{});
            auto tiled = make_tiled_mma(MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>{},
                                       Layout<Shape<_1,_1>>{},Tile<_16,_16,_16>{});
            const int wid=compute_tid/32, lane=compute_tid%32;
            auto thr=tiled.get_slice(lane);
            Tensor ref=local_tile(ut,make_shape(Int<16>{},Int<16>{}),make_coord(0,0));
            auto c_ref=thr.partition_C(ref);
            using Acc=decltype(thr.make_fragment_C(c_ref));
            using BFrag=decltype(make_fragment_like<BF16>(thr.make_fragment_C(c_ref)));
            auto coords=thr.partition_C(make_identity_tensor(make_shape(Int<16>{},Int<16>{})));
            auto ldC=make_tiled_copy_C(Copy_Atom<SM75_U32x4_LDSM_N,BF16>{},tiled);
            auto stC=make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N,BF16>{},tiled);
            auto ldCT=make_tiled_copy_C(Copy_Atom<SM75_U16x8_LDSM_T,BF16>{},tiled);
            auto stCT=make_tiled_copy_C(Copy_Atom<SM90_U16x8_STSM_T,BF16>{},tiled);
            auto l=ldC.get_slice(lane); auto s=stC.get_slice(lane);
            auto lt=ldCT.get_slice(lane); auto st=stCT.get_slice(lane);
            BFrag direct_out[2];
            auto& scratch=shared_storage.transposed;
            using KQLayout=decltype(tile_to_shape(
                GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{},
                make_shape(Int<32>{},Int<128>{}),LayoutLeft{}));
            Tensor kq=make_tensor(make_smem_ptr(scratch.kq),KQLayout{});
            // 每次复制 8 个 BF16，源/目的都保证 16B 对齐。原 TMA 输入布局不变。
            #pragma unroll
            for (int pack=compute_tid;pack<512;pack+=128) {
                int token=pack%32, key=(pack/32)*8;
                BF16 const* source=token<16 ? &k_decayed(token,key) : &q_decayed(token-16,key);
                *reinterpret_cast<uint4*>(&kq(token,key))=*reinterpret_cast<uint4 const*>(source);
            }

            // Phase1：原 KD@S^T 和 QD@S^T 改为 S@KD^T、S@QD^T。
            // 共同右操作数 [KD;QD]：M128,N32,K128，8 次 K16，无补零。
            c1_transpose::publish(compute_barrier);
            if constexpr (!C1_TRANSPOSE_HMMA_CONTROL) {
                if (compute_tid==0) {
                    #pragma unroll
                    for (int kb=0;kb<8;++kb) {
                        auto ad=c1_transpose::desc_k<128>(&s_acc(0,kb*16));
                        c1_transpose::mma<32>(scratch.tmem,ad,c1_transpose::desc_k<32>(&kq(0,kb*16)),kb!=0);
                    }
                    c1_transpose::commit(scratch);
                }
            }
            c1_transpose::wait(scratch,completion_phase);
            #pragma unroll
            for (int mi=0;mi<2;++mi) {
                const int row=wid*2+mi;
                Acc kv=thr.make_fragment_C(c_ref),qv=thr.make_fragment_C(c_ref);
                if constexpr (C1_TRANSPOSE_HMMA_CONTROL) {
                    c1_transpose::control_fragment<128,false>(s_acc,kq,row,0,lane,kv);
                    c1_transpose::control_fragment<128,false>(s_acc,kq,row,1,lane,qv);
                } else {
                    c1_transpose::load16(scratch.tmem+(uint32_t(row*16)<<16),kv);
                    c1_transpose::load16(scratch.tmem+(uint32_t(row*16)<<16)+16,qv);
                }
                BFrag vv,rhs;
                Tensor vb=local_tile(vt,make_shape(Int<16>{},Int<16>{}),make_coord(row,0));
                copy(ldCT,lt.partition_S(vb),lt.retile_D(vv));
                #pragma unroll
                for (int j=0;j<8;++j) {
                    const int token=int(get<1>(coords(j)));
                    BF16 beta=BF16(sigmoid_tanh_approx_f32(float(beta_tile(beta_smem_offset+token))));
                    // 与官方相同：先 cast GEMM，再 BF16 减法，再 BF16 乘 beta。
                    BF16 diff=vv(j)-BF16(kv(j));
                    rhs(j)=diff*beta;
                    direct_out[mi](j)=BF16(qv(j));
                }
                Tensor rb=local_tile(ut,make_shape(Int<16>{},Int<16>{}),make_coord(row,0));
                copy(stC,s.retile_S(rhs),s.partition_D(rb));
            }

            // Phase3：U^T=RHS^T@INV^T。其输入输出共用同一 4 KiB 缓冲。
            c1_transpose::publish(compute_barrier);
            if constexpr (!C1_TRANSPOSE_HMMA_CONTROL) {
                if (compute_tid==0) {
                    c1_transpose::mma<16>(scratch.tmem,c1_transpose::desc_k<128>(scratch.rhs_u),
                                         c1_transpose::desc_k<16>(&INV(0,0)),false);
                    c1_transpose::commit(scratch);
                }
            }
            c1_transpose::wait(scratch,completion_phase);
            #pragma unroll
            for (int mi=0;mi<2;++mi) {
                const int row=wid*2+mi;
                Acc acc=thr.make_fragment_C(c_ref); BFrag u;
                if constexpr (C1_TRANSPOSE_HMMA_CONTROL)
                    c1_transpose::control_fragment<16,false>(ut,INV,row,0,lane,acc);
                else c1_transpose::load16(scratch.tmem+(uint32_t(row*16)<<16),acc);
                cute::transform(acc,u,[] __device__(float x){return BF16(x);});
                Tensor ub=local_tile(ut,make_shape(Int<16>{},Int<16>{}),make_coord(row,0));
                copy(stC,s.retile_S(u),s.partition_D(ub));
            }

            // Phase4 和 Phase6 只依赖 U，互不依赖，合成一个异步完成组。
            // 0..127 列放 state update，128..143 列放输出增量；减少一次 commit/wait。
            c1_transpose::publish(compute_barrier);
            if constexpr (!C1_TRANSPOSE_HMMA_CONTROL) {
                if (compute_tid==0) {
                    auto ad=c1_transpose::desc_k<128>(scratch.rhs_u);
                    c1_transpose::mma<16>(scratch.tmem+128,ad,c1_transpose::desc_k<16>(&Mqk(0,0)),false);
                    c1_transpose::mma<128,true>(scratch.tmem,ad,c1_transpose::desc_mn(&kr_t(0,0)),false);
                    c1_transpose::commit(scratch);
                }
            }
            c1_transpose::wait(scratch,completion_phase);
            #pragma unroll
            for (int mi=0;mi<2;++mi) {
                const int row=wid*2+mi;
                Acc acc=thr.make_fragment_C(c_ref); BFrag out;
                if constexpr (C1_TRANSPOSE_HMMA_CONTROL)
                    c1_transpose::control_fragment<16,false>(ut,Mqk,row,0,lane,acc);
                else c1_transpose::load16(scratch.tmem+(uint32_t(row*16)<<16)+128,acc);
                #pragma unroll
                for (int j=0;j<8;++j) out(j)=direct_out[mi](j)+BF16(acc(j));
                Tensor ob=local_tile(ot,make_shape(Int<16>{},Int<16>{}),make_coord(row,0));
                copy(stCT,st.retile_S(out),st.partition_D(ob));
            }

            // State epilogue：保持 S[value,key] 朝向；gate 缩放的是 key 列。
            // 一次读取 16×128 的 TMEM 条带，省去逐 16×16 发 wait 的开销。
            #pragma unroll
            for (int mi=0;mi<2;++mi) {
                const int row=wid*2+mi;
                float wide[64];
                if constexpr (!C1_TRANSPOSE_HMMA_CONTROL)
                    c1_transpose::load128(scratch.tmem+(uint32_t(row*16)<<16),wide);
                #pragma unroll
                for (int ni=0;ni<8;++ni) {
                    Acc update=thr.make_fragment_C(c_ref); BFrag old;
                    if constexpr (C1_TRANSPOSE_HMMA_CONTROL)
                        c1_transpose::control_fragment<16,true>(ut,kr_t,row,ni,lane,update);
                    else {
                        #pragma unroll
                        for (int j=0;j<8;++j) update(j)=wide[ni*8+j];
                    }
                    Tensor sb=local_tile(s_acc,make_shape(Int<16>{},Int<16>{}),make_coord(row,ni));
                    copy(ldC,l.partition_S(sb),l.retile_D(old));
                    #pragma unroll
                    for (int j=0;j<8;++j) {
                        const int key=ni*16+int(get<1>(coords(j)));
                        old(j)=BF16(bf16_to_f32(old(j))*g_total(key)+update(j));
                    }
                    copy(stC,s.retile_S(old),s.partition_D(sb));
                }
            }
            if constexpr (!C1_TRANSPOSE_HMMA_CONTROL)
                asm volatile("tcgen05.fence::before_thread_sync;":::"memory");
            }
