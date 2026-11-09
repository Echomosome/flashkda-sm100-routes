            // ======== Phase 6: A2, direct MN descriptor + vectorized state ========
            Tensor p6_u = make_tensor(make_smem_ptr(shared_storage.phase6.u), VOLayout{});
            Tensor p6_ut = make_tensor(make_smem_ptr(shared_storage.phase6.u), TransposedVOLayout{});
            #pragma unroll
            for (int bi = 0; bi < 2; ++bi) {
                Tensor dst = local_tile(p6_u, make_shape(Int<16>{}, Int<16>{}),
                                         make_coord(0, warp_id * 2 + bi));
                copy(smem_tiled_store_C, smem_thr_store_C.retile_S(u_bf16[bi]),
                     smem_thr_store_C.partition_D(dst));
            }
            c1_phase6::issue(shared_storage.phase6,
                shared_storage.input[load_stage].k_restored.begin(), compute_barrier, compute_tid);

            // 两个版本都使用相同 U 落盘、warp 行分工及向量化状态 epilogue。
            // HMMA 控制组也从同一个 U 共享布局读取，不用无消费的虚假搬运充当控制。
            auto p6_copy_B_T = make_tiled_copy_B(Copy_Atom<SM75_U16x8_LDSM_T, BF16>{}, mma);
            auto p6_thr_B_T = p6_copy_B_T.get_thread_slice(lane_id);
            AFragT p6_a;
            BFragT_u p6_b;
            SFragT p6_state;

            #pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                const int row_tile = warp_id * 2 + mi;
                float g0 = g_total(row_tile * 16 + group_id);
                float g1 = g_total(row_tile * 16 + group_id + 8);
                Tensor state_first = local_tile(s_acc_T, make_shape(Int<16>{}, Int<16>{}),
                                                make_coord(row_tile, 0));
                copy(smem_tiled_load_C_T, smem_thr_load_C_T.partition_S(state_first),
                     smem_thr_load_C_T.retile_D(p6_state));
                if constexpr (C1_PHASE6_HMMA_CONTROL) {
                    Tensor kr_block = local_tile(k_restored_t, make_shape(Int<16>{}, Int<16>{}),
                                                 make_coord(row_tile, 0));
                    copy(smem_tiled_copy_A_T, smem_thr_copy_A_T.partition_S(kr_block),
                         smem_thr_copy_A_T.retile_D(p6_a));
                }
                // 第一个状态片段和 gate 的加载可与异步 MMA 重叠。
                if (mi == 0) c1_phase6::wait(shared_storage.phase6, t);
                #pragma unroll
                for (int ni = 0; ni < 8; ++ni) {
                    AccFragT update = thr_mma.make_fragment_C(tCrC_ref);
                    if constexpr (C1_PHASE6_HMMA_CONTROL) {
                        Tensor ub = local_tile(p6_ut, make_shape(Int<16>{}, Int<16>{}), make_coord(ni, 0));
                        copy(p6_copy_B_T, p6_thr_B_T.partition_S(ub), p6_thr_B_T.retile_D(p6_b));
                        clear(update);
                        gemm(thr_mma, p6_a(_,_,Int<0>{}), p6_b(_,_,Int<0>{}), update);
                    } else {
                        uint32_t addr = shared_storage.phase6.tmem
                                      + (uint32_t(row_tile * 16) << 16) + ni * 16;
                        c1_phase6::load_fragment(addr, update);
                    }
                    #pragma unroll
                    for (int a = 0; a < 2; ++a) {
                        #pragma unroll
                        for (int d = 0; d < 2; ++d) {
                            auto c0 = make_coord(make_coord(a, 0), 0, d);
                            auto c1 = make_coord(make_coord(a, 1), 0, d);
                            p6_state(c0) = BF16(bf16_to_f32(p6_state(c0)) * g0 + update(c0));
                            p6_state(c1) = BF16(bf16_to_f32(p6_state(c1)) * g1 + update(c1));
                        }
                    }
                    Tensor dst = local_tile(s_acc_T, make_shape(Int<16>{}, Int<16>{}), make_coord(row_tile, ni));
                    copy(smem_tiled_store_C_T, smem_thr_store_C_T.retile_S(p6_state),
                         smem_thr_store_C_T.partition_D(dst));
                    if (ni + 1 < 8) {
                        Tensor next = local_tile(s_acc_T, make_shape(Int<16>{}, Int<16>{}), make_coord(row_tile, ni + 1));
                        copy(smem_tiled_load_C_T, smem_thr_load_C_T.partition_S(next),
                             smem_thr_load_C_T.retile_D(p6_state));
                    }
                }
            }
            if constexpr (!C1_PHASE6_HMMA_CONTROL)
                asm volatile("tcgen05.fence::before_thread_sync;":::"memory");
