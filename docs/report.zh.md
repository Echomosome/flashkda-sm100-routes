# FlashKDA 在 B300 上的复现、设计分析与三路线挑战

**2026-09-11（三路线完整实机结果版，2026-11-09 公开整理）**

范围：固定源码的 forward，主形状 `T=8192, H=96, K=V=128`，以及 varlen 和单卡
`H=12` 对照。本文是课程报告的公开版本；原始日志、`.ncu-rep` 与 GPU 二进制
未随仓库发布，保留在实验机上，证据身份见 [evidence.md](evidence.md)。

## 摘要

本项目研究 FlashKDA 在 B300 上仍使用 SM80 风格 MMA 的设计取舍。我们完成原实现复现、六个讨论点的分析与实验，并将挑战范围扩展为三条分别实现的路线：A为真实 recurrence 的 Phase6 tcgen05 替换，B为CHUNK32/64＋相对前缀重标定，C为独立 value 列分片。

官方H96 benchmark中，FlashKDA按其FP32-state接口口径相对FLA的时间比为2.2923×、2.7174×、3.2670×；该接口不改变内部BF16 state。SASS与源码共同确认原计算路径使用HMMA。范围、低精度求逆、局部Phase6和state精度实验解释了“只看新指令峰值”不足以决定后端的原因。

三路线及控制组共六个实现，各通过38个完整KDA用例，包含T8192、变长与弱遗忘长链；420项output/state比较通过。A/A_control/C在已测输入上逐位等于baseline，B16/32/64满足预定FLA门槛。A保留C16及原Phase1—5，B使用FP32内部state和下三角求解，C仍使用HMMA。

最新完整forward CUDA Graph实验有24组比较、576条时间及96项计时后检查。**A耗时为baseline的2.36—3.38倍，B32/B64当前组合实现也没有完整性能收益。C在fixed H12从0.791984 ms降至0.685872 ms，加速1.1547×、耗时减少13.40%；H96三个用例回退13.07%—14.53%。** 独立Phase6的1.1096×局部收益没有转化为本次完整集成收益。

A的HMMA控制组本身已经显著回退，说明共同重写的数据流、布局与同步成本不能忽略。B的负结果来自FP32组合算子版本，不构成对融合SM100大CHUNK实现的最优性判断。因此，当前证据支持保留官方路径作默认并保留H12的C候选，不能证明所有SM100专版都不值得开发。

## 1. 任务、来源与证据口径

### 1.1 任务完成范围

| 题目要求 | 本项目完成的工作 | 适用边界 |
|---|---|---|
| Task 1：复现及指令确认 | 固定源码构建、代表性正确性、官方 H96 benchmark、NCU、HMMA SASS | H64 为题目附加形状，未据此扩展结论 |
| Task 2：六点“结论 + 证据” | 源码/ISA 推导，范围、求逆、Phase 6、state 精度实机实验，NCU 机制分析和 v2 决策 | standalone 实验各自保持范围，不代替完整 KDA 性能 |
| Task 3：三路线分别实现 | A真实Phase6指令替换、B大CHUNK＋rescale、C列分片，各有完整对拍与四形状性能 | A只换Phase6；B为FP32组合实现；C仍用HMMA；NCU机制证据仅覆盖历史C |
| 交付 | 代码、完整 README 证据、本报告、答辩 PPT、入门讲解 | 不把原始失败、负收益或未覆盖项删去 |

实验在 B300 上执行，本文据运行日志整理；源码阅读、代数检查、日志解析、哈希与统计复核在本地完成。公开整理没有新增 GPU 测量。

后文采用以下标识：**[实机]** 表示设备实验结果；**[离线·源码/推导]** 表示数学或实现分析；**[离线·从实机数据推算]** 表示根据真实字段计算的派生量。原始输出、错误过程、完整路径与哈希保留在 [证据说明](evidence.md)，本报告抽取支持结论所必需的证据。

### 1.2 环境与源码

| 项目 | 记录 |
|---|---|
| GPU | NVIDIA B300 SXM6 AC，compute capability 10.3，148 SM |
| CUDA 编译器 | 13.0，V13.0.88 |
| Python / PyTorch | Python 3.12，PyTorch 2.14.0+cu130 |
| NumPy / NCU | NumPy 2.5.3（后期探针），Nsight Compute 2025.3.1 |
| FlashKDA | 题目快照 commit `1ce47ea` |
| CUTLASS / FLA 参考 | 题目指定 `5c149f5` / `a3edffc` |
| 频率和缓存 | 实验未锁 GPU 频率；各实验分别记录缓存口径 |
| 候选构建 | 独立 Torch JIT 扩展，明确 `compute_103a / sm_103a`，未覆盖已安装 baseline |

基线来源是题目提供的快照。源码 hash 可与本地核对；远端 `.so`、`.ncu-rep` 等仅有路径/hash、尚未取得字节的文件，在实验记录中继续标为“未取得原始字节”，不写成已本地验证其内容。前后 GPU 进程快照只约束两个采样时刻。

## 2. KDA 计算与 FlashKDA 的分工

### 2.1 每个 head 维护一张状态矩阵

**[离线·源码/推导]** 以数学布局记状态为 $S_t\in\mathbb{R}^{K\times V}$，本项目 $K=V=128$。$q_t,k_t$ 是长度 K 的向量，$v_t$ 是长度 V 的向量。为便于公式阅读，把 $1/\sqrt K$ 缩放吸收到 $q_t$ 中。$g_t$ 在此表示**已激活的 log gate**，不能直接等同于接口传入的 raw gate。

递推可以写为：

$$
\bar S_t=\operatorname{diag}(\exp(g_t))S_{t-1},\qquad
\Delta v_t=v_t-\bar S_t^\top k_t,
$$

$$
S_t=\bar S_t+\beta_t k_t\Delta v_t^\top,\qquad
o_t=S_t^\top q_t.
$$

第一步按 key 维度衰减旧记忆；第二步计算当前 value 与旧记忆预测的差；第三步把这个差写回状态；最后用 query 读取状态。公式与 [FLA naive 参考](../fla_kda_ref/naive.py) 对齐。本文讨论每个 head 的 forward，不涉及训练、反向传播或整模型质量。

状态将历史信息压缩成固定大小矩阵，但 $S_t$ 依赖 $S_{t-1}$。因此不能把连续 token 当成互不相关的任务直接发给不同 CTA。并行空间要从序列、head、矩阵维度，或经证明等价的算法重构中寻找。

### 2.2 两个 kernel 具有不同并行空间

FlashKDA 将 token 分为长度 C=16 的 chunk，并把 chunk 内可重排的运算组织成小矩阵乘。

| 阶段 | 做什么 | 基线并行维度 | fixed H96 的 grid |
|---|---|---|---:|
| K1 `prepare` | gate/归一化、衰减变换、构造小矩阵、Neumann 求逆 | 序列 × head × chunk | $1\times96\times512=49152$ |
| K2 `recurrence` | 读取状态，逐 chunk 计算更新与输出，写回状态 | 序列 × head，CTA 内循环所有 chunk | $1\times96=96$ |

这里 CTA 即 CUDA thread block。baseline K2 每个 CTA 有 192 线程，即 6 个 warp：4 个 compute warp，另有搬入和搬出职责。由于 B300 有 148 个 SM，fixed H96 的 96 个 CTA 无法同时给全部 SM 分配一个 CTA；H12 时这个缺口更明显。

TMA 和 MMA 分别负责数据搬运与矩阵计算。源码使用 `SM90_TMA_LOAD/STORE`，同时使用 SM80 MMA，是一条可以共存的数据流。SM80 atom 也不意味着整个使用 TMA 的 binary 可以原样运行在 SM80 GPU 上。

实现与阶段映射见 [fwd_kernel1.cuh](../FlashKDA/csrc/smxx/fwd_kernel1.cuh)、[fwd_kernel2.cuh](../FlashKDA/csrc/smxx/fwd_kernel2.cuh) 以及 [固定版本设计文档](../FlashKDA/docs/20260420-flashkda-v1-deep-dive.md)。

## 3. Task 1：复现与主路径确认

### 3.1 官方 benchmark

**[实机]** 使用题目原始 `bench_fwd.py`，`warmup=30, iters=200, repeats=5`。每条路径共有 1000 次 CUDA Event 计时，原脚本汇总 mean/min/max；下表保留其 mean 口径。

| T=8192，H96，D128 | FlashKDA BF16-state / ms | FlashKDA FP32-state 接口 / ms | FLA chunk_kda / ms | FLA / Flash FP32 接口 |
|---|---:|---:|---:|---:|
| fixed，单序列 | 1.7553 | 1.7135 | 3.9279 | 2.2923× |
| varlen6 | 1.4759 | 1.4976 | 4.0696 | 2.7174× |
| varlen8 | 1.1866 | 1.2136 | 3.9648 | 3.2670× |

varlen6 长度为 `[1300,547,2048,963,271,3063]`；varlen8 为八个 1024。两者总 token 均为 8192。上表速度比沿用官方报告的 FP32-state 接口口径，不能把它称为“内部 FP32 递推”的性能。`chunk_gated_delta_rule` 的结果保留在实验记录，但它是不同算法，不作为 KDA 等价性能或正确性参照。

这些平均时间不能与 Task 3 的 Graph 中位数直接相除。后者采用不同计时方式，并统一 BF16 初态和终态。NCU 下 `warmup=0,iters=1` 的打印时间同样不用于替代正式 benchmark。

来源：实验记录第5、11节，运行 `20260909T164935Z`、`20260909T165802Z`。

### 3.2 SASS 正向确认

**[实机]** SASS 中出现 `HMMA.16816.F32.BF16` 和 `HMMA.16816.F16`，与源码中的 `SM80_16x8x16_F32BF16BF16F32_TN`、`SM80_16x8x16_F16F16F16F16_TN` 相符。由实际指令和源码共同确认当前 SM80 风格 MMA 计算路径。

旧导出中统计到 272 行 BF16/F32 HMMA、24 行 F16 HMMA。地址有重复呈现，统计的是静态文本命中，不是运行时次数。旧脚本搜索 `TCGEN05=0` 不能排除 Blackwell SASS 的 `UTCHMMA` 名称；本文只使用已有正向证据，不将旧正则包装成严格的全指令族排除。

### 3.3 初始 NCU 给出的优化方向

**[实机]** 同一 smoke 报告 `20260909T172953Z`：

| 指标 | K1 prepare | K2 recurrence |
|---|---:|---:|
| kernel duration | 273.216 μs | 749.472 μs |
| grid / block | 49152 / 256 | 96 / 192 |
| registers/thread | 32 | 73 |
| dynamic shared memory/CTA | 21248 B | 98432 B |
| theoretical occupancy | 100% | 18.75% |
| achieved occupancy | 96.64% | 9.37% |
| compute throughput | 70.03% | 20.53% |
| memory throughput | 72.20% | 39.01% |
| DRAM throughput | 58.87% | 18.65% |

K2 占这两个目标 kernel 时间约 73.3%，而计算和 DRAM 都没有接近其峰值。优先问题是任务数量和依赖链、SM 内可调度工作及资源，而不是先假设峰值 MMA 不够。varlen6 的 K2 有 576 CTA、achieved occupancy 16.64%，说明独立序列增多确实改变了执行形态。

Task 1 已满足题目核心要求。完整模型精度、全部官方用例、H64 附加形状并不在本文中冒充已完成。

## 4. Task 2：六个讨论点

### 4.1 CHUNK=16：范围、求逆与形状的共同约束

#### 指数范围先在哪里出现问题

**[离线·推导]** 若已激活 log gate 恒为 $-\alpha$，第 i 个前缀衰减为 $\exp(-i\alpha)$，其逆为 $\exp(i\alpha)$。最终表达式可能有有限的抵消结果，但分别物化这两个因子时，仍可能先出现 0 或 Inf，再形成 `0*Inf=NaN`。范围风险取决于累计 gate，而非仅由 CHUNK 长度决定。

**[实机]** B300 探针使用实际 `ex2.approx.ftz.f32` 和 BF16 RN 转换，没有 MMA，也没有 KDA 计时：

| 常量 α | 首个 decay 为0的前缀 | 首个 inverse 非有限的前缀 | 解释 |
|---|---:|---:|---|
| 1.0 | C≤64 内未观察到 | C≤64 内未观察到 | 本输入未触发范围边界 |
| 2.5 | 35 | 36 | decay 先被 FTZ 路径清零，随后逆溢出 |
| 5.0 | 18 | 18 | C16 内有限，扩大到 C32 时跨过边界 |

索引从1开始。α=2.5 的第35项 restored 值已为0，但非有限 restored 从36出现。因此“没有 NaN”也不保证表达式仍正确。该范围探针只验证因子化，不是完整大CHUNK实现。后续B路线已完成C32/C64全链对拍与性能，见第5—6节。

#### 补足 Neumann 阶数不能保证低精度正确

对严格下三角 $L\in\mathbb{R}^{C\times C}$，有 $L^C=0$。令 $Q=-L$，则精确算术下：

$$
(I+L)^{-1}=\sum_{j=0}^{C-1}Q^j
=\prod_{j=0}^{\log_2 C-1}(I+Q^{2^j}),\quad C=16,32,64.
$$

乘积分别包含4、5、6个因子。当前结构从首项初始化后，需要的 dense GEMM 数分别为6、8、10。固定总 token/head 时，求逆批次的纸面 dense-MMA 工作相对 C16 为 **1、5.3333、26.6667**：单矩阵立方成本增加，不能只看矩阵个数随 C 减少。

**[实机]** 独立 cuBLAS 低精度求逆实验中，shift 对照证明 C32/C64 的16项截断残差为1，补足完整项数后为0。但相干输入“严格下三角全0.5”的完整 C32 FP16 逆仍有 **3.13098362** 相对 infinity-norm 误差，即约313%，且所有值有限；C64 则全部非有限。此结果属于独立低精度序列，不等同原 FlashKDA 的逐指令舍入顺序。

#### 成本用独立 MMA 实现量化

在统一 shared-memory、1 warp/CTA、良态小范数输入的 standalone MMA 求逆中，固定 T=8192、H=96：

| C | 每批矩阵数 | 批次时间中位数 / ms | 相对 C16 |
|---|---:|---:|---:|
| 16 | 49152 | 0.130188799 | 1.0000× |
| 32 | 24576 | 0.965753555 | 7.4181× |
| 64 | 12288 | 6.988127900 | 53.6769× |

每种 C 为9轮，每轮 Graph 含5次完整 grid。该时间包含本实现的搬运、同步、求逆和写出；不等于原 helper 的时间，更不等于完整 KDA 的减速倍数。抽样9个不同矩阵、18次 direct/Graph 检查通过，不写成全批次逐元素验证。

C16 的 `16×16×16` 可由两个 `m16n8k16` atom 铺满；C32/C64 也能整倍数铺满，因此大 C 的成本不能全部归因于 padding。**结论：不存在所有输入统一的“先破项”。gate 决定范围压力，L 决定求逆稳定性，布局和资源决定成本。** 大 CHUNK 必须同时处理这些问题。

证据：实验记录第13、14、16、19节；范围 run `061622976790Z`、数值 run `064311694334Z`、MMA 求逆 run `092501138041Z`（均为20260910）。

### 4.2 tcgen05：形状有机会，数据流必须重做

**[离线·源码/ISA核对]** 普通单 CTA、dense BF16/FP32 的 tcgen05 模式允许 M=64/128，N 以8递增，K=16；普通最小 tile 为 `64×8×16`。`.ws` 模式允许 M=32，但要求 N≥64。M16 小矩阵无法照搬原来的 tile。新指令的 accumulator 在 TMEM，B operand 来自 shared memory，与当前寄存器 fragment、`MOVM_T` 和中间 epilogue 的连接方式不同。[NVIDIA PTX ISA 的形状约束](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-matrix-shape)

当前 K2 的 Phase 6 为 $128\times16$ 乘 $16\times128$，正好匹配 `m128n128k16`。它占 K2 **MMA FLOP** 的30.77%，这个百分比并不是耗时比例。Phase 1 的两个乘法可在纸面上探索共同 B 的合并；其余 M16 阶段继续面临形状或中间搬运问题。

因此先实现一个边界清楚的 Phase 6 探针：共用 packed SMEM 输入、相同 BF16 state 更新，分别调用 HMMA 与 tcgen05，包含初始 global staging、最终 store 和每 CTA 一次 TMEM 分配/释放。实机 SASS 确认两条路径分别出现 HMMA 与 UTCHMMA。

**[实机]** M=N=128、K=16、96 CTA，12轮、每 Graph 5次 kernel：

| 每 CTA 更新步数 | SM80 批次中位数 / μs | tcgen05 批次中位数 / μs | SM80 / tcgen05 |
|---|---:|---:|---:|
| 1 | 24.540801 | 24.944000 | 0.9838× |
| 32 | 218.307209 | 196.739197 | 1.1096× |

GEMM、direct、Graph warmup 和计时后 state 检查通过。32步用循环摊薄一次性成本，但 A/B 在步骤间复用，也没有前序阶段生成 U 的成本。这验证了特定Phase6类实现可以受益。后续A路线接入真实U物化与完整KDA后反而回退（第6节），说明局部结果不能外推；也不能从30.77% FLOP占比做Amdahl耗时推算。

首次编译失败和 SASS 解析失败均保留：显式 `-gencode=arch=compute_103a,code=sm_103a` 解决目标配置问题，识别 `@UP1 UTCHMMA` 解决统一谓词漏匹配。第二次失败属于检查器漏识别，不说明硬件缺少 tcgen05。

证据：实验记录第12.2—12.3、20—24节，成功 run `20260910T125328470967Z`。

### 4.3 有状态依赖时，并行度从哪里来

**[离线·推导]** 保持现有数值路径时，可用并行空间包括独立序列、head 和 value 维度。候选取舍如下：

| 方案 | 可能好处 | 反例或额外成本 | 本次选择 |
|---|---|---|---|
| 多 head 合入一个 CTA | 合并更大的 MMA 工作 | CTA 数反而减少，可能恶化小 head 数的 GPU 覆盖；state 资源变大 | 未实现 |
| persistent 调度 | 减少调度、动态平衡长度 | 没有新增独立工作，12个 head 不会凭调度变成148条独立递推 | 未实现 |
| `cta_group::2` / cluster | 两 CTA 共同执行较大 MMA | 两 CTA 合作的资源与同步，并不等于两个独立 GEMM | 未实现 |
| 时间切分 / 仿射 scan | 跨 chunk 组合变换，拓展时间并行 | 需传播或组合更大状态变换，增加计算通信并改变浮点结合顺序 | 未实现 |
| 独立 value 列分片 | 每个 head 增加独立 CTA，无跨分片归约 | 重复读取 q/k/gate 等数据，增加 I/O warp 和同步工作 | 实现 P2 |

将 $S=[S^{(0)}\ S^{(1)}]$，$v=[v^{(0)}\ v^{(1)}]$，则每个分片 p 满足：

$$
\Delta v_t^{(p)}=v_t^{(p)}-(\bar S_t^{(p)})^\top k_t,\quad
S_t^{(p)}=\bar S_t^{(p)}+\beta_t k_t(\Delta v_t^{(p)})^\top.
$$

右端只需要本分片状态以及共同 q/k/gate/β，所以 value 列彼此独立。key 维度的内积仍然完整，不能用同样理由直接拆 key 后忽略归约。这是 Task 3 选择列分片的数学依据。

### 4.4 compute-bound 还是 memory-bound

**结论按 kernel 和形状分别给出。** 初始 K1 的 compute/memory throughput 接近且较高；K2 两者均低，又存在小 grid、长递推与共享内存约束，用一个二选一标签描述整个 forward 会丢失主要问题。

分析至少结合：`launch__grid_size`、block 大小、寄存器/shared memory 驻留上限，理论和实际 occupancy，SM active/elapsed cycles，Tensor 管线 active/elapsed 利用率，eligible warp、issue active，以及 DRAM/L2 和 stall 字段。它们分别回答“有多少任务”“活跃 SM 内是否有工作可发射”“GPU 整体是否得到使用”“数据从哪里来”。

必须保留分母：active-cycle 归一化与 elapsed-cycle 归一化不能互换。NCU 的 occupancy 是 warp 驻留指标，不是“整张 GPU 开工百分比”。此外，`memory throughput` 的聚合值不能直接当作 HBM 饱和度。[Nsight Compute 指标定义](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-guide)

Task 3 的对照进一步表明：增加 SM 工作覆盖可以和 achieved occupancy 下降同时发生；H12 依然可以更快。具体值见第6节。

### 4.5 BF16 state：数值有限与精度达标分开判断

**[离线·方案]** 我们设置 FP64 数值参考 R64、FP32 递推对照，以及仅在 chunk 边界把 state 舍入为 BF16 的 Q16 模型，再对比原 FlashKDA。输入共同量化、初态非零，记录多个 checkpoint、各 head、末段输出、最终 state，以及非有限计数。Q16 用于隔离一种误差来源，不能模拟完整 kernel 的所有低精度运算。

**[实机]** T8192、H4、D128、C16、固定 seed，四个构造条件的 NRMSE 如下。定义 $\mathrm{NRMSE}=\mathrm{RMS}(x-r)/(\mathrm{RMS}(r)+10^{-12})$。

| 输入条件 | 最终 state | 最后128 token输出 | 全8192 token输出 |
|---|---:|---:|---:|
| random_decay | 0.609925% | 0.563926% | 0.569811% |
| random_weak | 0.672889% | 0.783428% | 0.768497% |
| coherent_near_identity | 0.462707% | 0.447743% | 0.257128% |
| random_slow_write | 3.766088% | 3.746915% | 2.195131% |

所有记录有限。慢写入 head 的 Q16 最终 state 误差为3.700467%，与 FlashKDA 随 token 增长的趋势相近；二者相互比较仍有1.477139% NRMSE，且参考分母不同。因此不能把完整误差全部归于 state BF16，也不能将两列 NRMSE 相减作为环节贡献。

同一 BF16 可精确表示初态下，FP32-state 接口结果与 BF16 接口一致，FP32 输出 state 仍落在 BF16 数值网格，符合内部 BF16 存储的源码。它不代表内部 FP32 递推已经实现。

结论是精度验证必须覆盖弱遗忘、慢写入、后段轨迹与 state，不能仅看短序列或 NaN。当前样本来自构造数据和一个 seed；没有模型任务精度阈值、真实激活样本或下游质量结论。FLA naive 的64 token CPU交叉核对只约束短前缀方程和布局。

证据：实验记录第25—26节，run `20260910T133341103264Z`；[证据说明](evidence.md)。

### 4.6 v2 是否推出架构专版

我们支持可选、受shape与设备约束的专用路径，保留官方路径作为默认和回退依据。支持继续研究的事实是局部tcgen05曾有收益、C在H12有完整forward正收益；反对直接通用替换的事实是A当前完整集成回退、B组合算子代价很大，以及C在H96回退。

“SM80路径”指计算atom，不能推导包含TMA的整个kernel能在A100原样运行。本次B300使用 `compute_103a/sm_103a`，没有测试另一型号SM100设备或binary兼容性。A只替换Phase6，其余阶段仍HMMA。

目前不改通用dispatch。C保留fixed H12候选，但单点不足以推出所有H≤12或全部varlen条件的阈值。A需继续研究实际U布局和同步，B需研究融合/减少临时量；这些是研究方向，不是本报告承诺的收益。

## 5. Task 3：三条实现与统一正确性

题目原要求任选一个切面；本项目后来将范围扩展为三条分别实现。原P2和独立Phase6实验均保留，新增实现有控制组及统一参考对拍。

### 5.1 A：保持C16算法，替换真实Phase6

A保留K1和K2 Phase1—5的源码与计算，保留原BF16 state舍入边界。Phase6每个chunk消费真实前序生成的U，执行 $KR^\top U$ 并更新state，不能在步骤间复用固定A/B输入。

1. 原compute warp把实际BF16 U写入共享内存，128个compute线程在原named barrier同步。
2. 把 $KR^\top$ 与U整理为tcgen05需要的共享内存布局，形状为128×16与16×128。
3. 执行tcgen05、等待完成、读出TMEM结果，按原gate进行FP32 FMA并舍入回BF16 state。
4. 每CTA分配/释放一次TMEM，保持原load/store warp及pipeline协作。

`A_control` 使用相同的U物化、packed布局和重写epilogue，但矩阵乘仍为HMMA。两者动态shared memory均为110848 B，比baseline增加12416 B，block仍为192线程。

full编译的14个recurrence变体中，A每个含HMMA×72和UTCHMMA×1，控制组每个含HMMA×76。它们是静态指令计数，不是动态执行量。该结果证明新指令已接入完整递推，但不表示整个KDA的所有MMA都已替换。

实现：[A 生成器](../flashkda_routes/build.py)、[phase6.cuh](../flashkda_routes/routes/a_phase6.cuh)。控制组共同改变布局/搬运/同步，不能完全隔离各项开销。

### 5.2 B：C32/C64与相对前缀重标定

B使用完整C32/C64块，没有偷偷拆为C16子块。内部state和单位下三角求解为FP32，输出和本次性能state API为BF16；当前实现为PyTorch CUDA组合算子，没有宣称其GEMM的SASS或完成了融合SM100 kernel。

令chunk内的已激活log gate前缀为 $G_i=\sum_{t\le i}g_t$。因果位置直接计算

$$E_{ij}=\exp(G_i-G_j),\quad i\ge j.$$

由于合法gate≤0，有 $G_i-G_j\le0$，无需分别产生巨大逆衰减与微小衰减。未来位置在exp之前屏蔽。给定chunk初态 $S_0$，构造

$$L_{ij}=\beta_i\sum_d k_{i,d}k_{j,d}E_{ij,d}\quad (i>j),\qquad L_{ij}=0\quad(i\le j),$$
$$R_i=\beta_i[v_i-(k_i\odot\exp G_i)^\top S_0],\qquad (I+L)U=R.$$

FP32单位下三角求解避免显式inverse和原FP16 Neumann序列。输出与末状态为

$$o_i=(q_i\odot\exp G_i)^\top S_0+\sum_{j\le i}\left(\sum_dq_{i,d}k_{j,d}E_{ij,d}\right)U_j,$$
$$S_{\rm end}=\operatorname{diag}(\exp G_C)S_0+\sum_j(k_j\odot\exp(G_C-G_j))U_j^\top.$$

实现以贡献token为基准计算相对衰减，chunk边界恢复真实坐标，再将新块前缀从零开始。B16是相同FP32 state和求解器的控制组。离线NumPy分块/逐token比较共210项通过；随后实机实际调用题目FLA参考完成对拍。

此版本显式形成 `[H,C,C,D]` 中间量。H96、D128下，一个FP32数组在C16/32/64分别为12/48/192 MiB。它是由shape计算的单数组大小，不是峰值显存测量。实现：[b_rescale.py](../flashkda_routes/routes/b_rescale.py)。

### 5.3 C：独立value列分片（原P2）

P2 将每个 head 的128个 value 输出维度分成两个64维分片，以增加独立 CTA；每个 CTA 仍顺序处理所有 chunk。本次的数学 key 维度保持128，CHUNK保持16。

| 结构 | baseline | P2 |
|---|---|---|
| K2 grid | `(N,H)` | `(N,H,2)` |
| value/CTA | 128 | 64 |
| threads/CTA | 192 | 128 |
| compute / load / store warp | 4 / 1 / 1 | 2 / 1 / 1 |
| 数学 state 分片 | 128×128 | 128×64 |
| CUDA物理 state 布局 | `[value,key]` 128×128 | `[value,key]` 64×128 |
| V/O tile | 16×128 | 16×64 |
| q/k tile | 16×128 | 16×128 |

注意“数学列分片”和“物理数组行片”描述同一改动，不能因存储转置混淆 key/value 轴。分片偏移为 `64*blockIdx.z`，同时调整 TMA descriptor、V/O 与 state 地址及非方阵 state 转换 helper。共同 q/k、gate、逆矩阵等仍由两个分片各自读取。

源代码检查确认 K1 和 utils 保持原始字节，每个 compute warp 的融合 MMA 与 epilogue 保持一致。每个 head 总 compute warp 仍为4，但 load/store warp 从2变4。因此新 grid 增大既带来独立工作，也带来明确的重复开销。

源码生成、逻辑地址归属、矩形转换覆盖与列可分代数检查均标为**离线**。这些检查能排除部分索引/公式问题，但不能代替 CuTe layout、TMA 或 GPU 执行验证。随后实机编译与对拍补上这一层。

实现入口：[C 路线生成器](../flashkda_routes/routes/c_colsplit.py)。该文件生成独立候选 patch 并保存源码 manifest，不替换 baseline 安装。


### 5.4 统一full对拍覆盖及结果

**[实机]** `20260911T013931783287Z`：六个实现各38用例，包含fixed/varlen七种state模式、batch2、尾长1/15/16/17/31/32/33/63/64/65、强/弱gate、同向和零q/k、H12/H96短grid、四个T8192形状及弱遗忘长链。

每个case的输入与seed在路线间相同。实际运行题目固定 `fla_kda_ref/naive.py:naive_recurrent_kda`，验证完整output、需要返回的final_state及只读initial_state。

| 实现 | 输出/末状态检查 | 相对baseline | 相对FLA参考的验收 |
|---|---:|---|---|
| A_control / A | 每个38 / 32 | 本次均逐位一致 | 通过预定baseline-relative envelope |
| B16 / B32 / B64 | 每个38 / 32 | 数值实现不同，不要求逐位一致 | NRMSE≤0.005且max_abs≤0.01，全部通过 |
| C | 38 / 32 | 全部逐位一致 | 通过同一FLA envelope |

共420项output/state比较、228项INITIAL_UNCHANGED记录通过，无非有限值。没有初态的case中“初态不变”按不适用处理，228并非228个不同的非空初态。

本轮NRMSE为 $\|x-r\|_2/\max(\|r\|_2,10^{-30})$。A预定门槛是相对baseline NRMSE≤0.005、max_abs≤0.01；C要求逐位相等。它们同时满足相对FLA的NRMSE上限 `min(0.05,max(0.015,1.10*baseline_NRMSE+0.001))` 和max_abs上限 `max(0.02,1.10*baseline_max_abs+0.002)`。这些是工程门槛，不是模型质量标准。

在 `weak_long` T8192/H4 中，baseline/A/C的全输出NRMSE为2.176988%、末状态3.608108%；B32/B64的全输出均约0.1660%、末状态均约0.1670%。A/C未增加差异，但保留原误差；B的FP32 state、求解及激活差异与rescale共同影响误差，不能把精度改善全部归给单一改动。

依据：[v1 结果 JSON](../results/v1_three_routes.json) 与 [结果表](../results/tables.md)。原 P2 的 17 用例验证、Ninja 失败及修复保留在实验机日志中。

## 6. 三路线完整forward性能与机制证据

### 6.1 统一测量与复核方法

**[实机]** `20260911T015639253562Z` 引用上一轮PASS/full build。A/A_control/C复用已校验二进制，B使用同一v1代码，默认baseline不变。四个shape使用full中相同输入seed、T8192、D128与BF16初态/终态。

计时包含图内归一化、gate/beta、prepare、全链递推和写出，排除Python提交、图外分配与host descriptor构造。每个Graph为1次完整forward，直接预热5次、Graph预热3次，12轮交替baseline/candidate次序。复用缓冲区，不清cache，不锁频。

每组先检查eager/Graph一致，再计时；计时后将output/state填NaN并重放，验证有限性和与预热结果逐位一致，确认initial_state不变后才输出接受的COMPARISON。24组共有576条时间和96条postcheck，全部完整且通过。

两份日志附件属于同一个 run，第二份是正文副本；恢复的734行log SHA256与RECEIPT匹配，不重复计为两次实验。中位数与比值从逐轮时间重新计算。每组使用自己的配对baseline，定义

$$speedup=t_b/t_c,\qquad \Delta t=(t_c/t_b-1)\times100\%.$$

### 6.2 A、控制组与C：保持baseline状态舍入边界

| 用例 | 实现 | 配对 baseline / ms | 候选 / ms | baseline / 候选 | 时间变化 |
|---|---|---:|---:|---:|---:|
| fixed H96 | A_control | 1.029104 | 2.722240 | 0.378036× | +164.53% |
| fixed H96 | A | 1.030224 | 2.914528 | 0.353479× | +182.90% |
| fixed H96 | C | 1.028480 | 1.162928 | 0.884388× | +13.07% |
| varlen6 H96 | A_control | 0.857072 | 1.823568 | 0.469997× | +112.77% |
| varlen6 H96 | A | 0.860224 | 2.164880 | 0.397354× | +151.66% |
| varlen6 H96 | C | 0.859712 | 0.984272 | 0.873450× | +14.49% |
| varlen8 H96 | A_control | 0.690320 | 1.347904 | 0.512143× | +95.26% |
| varlen8 H96 | A | 0.688816 | 1.626736 | 0.423434× | +136.16% |
| varlen8 H96 | C | 0.689776 | 0.790000 | 0.873134× | +14.53% |
| fixed H12 | A_control | 0.793920 | 2.490320 | 0.318802× | +213.67% |
| fixed H12 | A | 0.793392 | 2.681632 | 0.295862× | +238.00% |
| fixed H12 | C | 0.791984 | 0.685872 | 1.154711× | -13.40% |

A四个shape均回退，C只有fixed H12获得正收益。1.154711×表示相同工作速度比提高15.47%，对应耗时减少13.40%；不能将这两个百分比混用。H12只是单GPU每卡头数模拟，不包括八卡TP通信或模型服务。

### 6.3 B：FP32组合实现的性能

| 用例 | 实现 | 配对 baseline / ms | 候选 / ms | baseline / 候选 | 时间变化 |
|---|---|---:|---:|---:|---:|
| fixed H96 | B16 | 1.031152 | 99.041443 | 0.010411× | +9504.93% |
| fixed H96 | B32 | 1.033008 | 120.475056 | 0.008574× | +11562.55% |
| fixed H96 | B64 | 1.033264 | 199.144356 | 0.005189× | +19173.33% |
| varlen6 H96 | B16 | 0.863872 | 100.079697 | 0.008632× | +11485.01% |
| varlen6 H96 | B32 | 0.861104 | 120.944130 | 0.007120× | +13945.24% |
| varlen6 H96 | B64 | 0.865312 | 198.725151 | 0.004354× | +22865.72% |
| varlen8 H96 | B16 | 0.692864 | 99.401104 | 0.006970× | +14246.41% |
| varlen8 H96 | B32 | 0.692928 | 120.791981 | 0.005737× | +17332.11% |
| varlen8 H96 | B64 | 0.695536 | 197.996399 | 0.003513× | +28366.74% |
| fixed H12 | B16 | 0.791088 | 46.081697 | 0.017167× | +5725.10% |
| fixed H12 | B32 | 0.790256 | 32.832479 | 0.024069× | +4054.66% |
| fixed H12 | B64 | 0.790384 | 35.571184 | 0.022220× | +4400.49% |

B32相对baseline耗时41.55—174.32倍，B64为45.00—284.67倍。此版本验证算法可行性和实际组合实现成本，不能作为融合SM100后端的性能上界或下界。B16控制组也明显慢，说明比较中包含原实现所没有的求解器/精度/组合算子成本。

### 6.4 控制组限制了哪些归因

| 用例 | A 比 A_control 时间变化 | B32 比 B16 时间变化 | B64 比 B16 时间变化 |
|---|---:|---:|---:|
| fixed H96 | +7.06% | +21.64% | +101.07% |
| varlen6 H96 | +18.72% | +20.85% | +98.57% |
| varlen8 H96 | +20.69% | +21.52% | +99.19% |
| fixed H12 | +7.68% | -28.75% | -22.81% |

A_control本身耗时已是baseline的1.95—3.14倍，A再增加7.06%—20.69%。这说明共同重写的U物化、布局、同步和epilogue存在重大成本。该控制组没有完全隔离TMEM、寄存器压力和调度，不能把全部差值解释为tcgen05指令latency，也不能据此证明所有tcgen05集成都慢。

B32在H96比B16慢约21%，在H12却快28.75%。较大chunk减少循环次数，同时扩大平方大小的中间张量；二者权衡与shape有关。对单个 `[H,C,C,D]` 数组，整个序列的累计元素数约为 $(T/C)HC^2D=THCD$，随C线性增加。这是源码的逻辑处理量，不是实测DRAM总字节；未给A/B新增NCU，因此不替它们贴确定的compute/memory-bound标签。

### 6.5 与旧P2结果的关系

旧run `20260910T162614136038Z` 同一C二进制在H96增加14.33%—14.63%时间，fixed H12加速1.1455×。新统一输入结果为H96增加13.07%—14.53%，H12加速1.1547×，方向一致。旧实验每图20次完整forward、预热和fixture不同，所以旧/新绝对时间不直接相除。

以下NCU表是此前C路线的独立机制证据，时间与统计分母按原实验保留。

### 6.6 NCU 对照与性能计时分别解释

此前P2机制研究在 run `20260910T165005117299Z` 采集 H96/H12、baseline/P2 的8个目标 kernel，采用 kernel replay、`--cache-control all`、未锁频。它回答C路线的历史机制问题，未采集A/B指标。它与最新Graph在时间、输入/缓存和重放口径上不同，不能作为A/B的profile或与最新时间直接相加减。

原采集和导出已完成，摘要程序误将842列宽表当成长表，触发 `StopIteration`。本地仅修复解析，核对单位行、8个 kernel 身份、grid 与源CSV hash后恢复全部结果，没有重新采 GPU。真实CSV含831个 metric 字段/目标；解析器拒绝截断、重复ID和错误grid。

| NCU duration / μs | H96 baseline | H96 P2 | H12 baseline | H12 P2 |
|---|---:|---:|---:|---:|
| K1 prepare | 457.568 | 456.128 | 69.088 | 68.640 |
| K2 recurrence | 1288.320 | 1547.904 | 1285.600 | 1110.368 |

K1 变化约−0.31%/−0.65%，K2 则 H96 增加20.15%、H12减少13.63%。这定位到列分片改变的 recurrence；NCU百分比不替代第6.2节的完整Graph性能数字。

### 6.7 共享内存减少为何没有提高理论 occupancy

下表对应本次 NCU 的 **fixed、both BF16** 分支。varlen 的候选 both BF16 分支在编译时报告72 registers/thread、24 B stack、20 B spill stores及20 B spill loads；这些是编译资源记录，不是动态访存总量。本次没有对varlen做相同NCU机制对照，因此其负收益不能仅用fixed计数完整归因。

| K2资源 | baseline | P2 |
|---|---:|---:|
| dynamic shared memory/CTA | 98432 B | 68608 B |
| 加上driver的shared memory/CTA | 99456 B | 69632 B |
| registers/thread | 73 | 78 |
| shared-memory block limit/SM | 2 | 3 |
| warps/CTA | 6 | 4 |
| theoretical active warps/SM | 12 | 12 |
| theoretical occupancy | 18.75% | 18.75% |

**[离线·从实机数据推算]** dynamic shared memory 减少30.30%，但 $2\times6=3\times4=12$，理论驻留 warp 上限未变。因此不能将“最多3 CTA/SM”直接写成“occupancy提高50%”。实际 grid 和尾部调度还会进一步约束运行。

### 6.8 GPU覆盖和活跃SM效率出现不同方向

| K2 指标 | H96 baseline | H96 P2 | H12 baseline | H12 P2 |
|---|---:|---:|---:|---:|
| grid | 96 | 192 | 12 | 24 |
| achieved occupancy | 9.37% | 8.46% | 9.37% | 6.25% |
| eligible warps/scheduler | 0.35 | 0.28 | 0.35 | 0.22 |
| issue active | 32.76% | 25.69% | 33.01% | 22.01% |
| Tensor利用率，active口径 | 30.75% | 20.59% | 30.95% | 17.71% |
| Tensor利用率，elapsed口径 | 19.66% | 16.36% | 2.48% | 2.84% |
| 平均SM active/elapsed周期比（推算） | 63.92% | 79.47% | 8.02% | 16.06% |

最后一行是 `sm__cycles_active.avg / sm__cycles_elapsed.avg`，描述整个采样窗口的平均活动比例，**不是 occupancy，也不是精确的同时活跃 SM 个数**。H96 P2 的最小SM active cycles大于0，说明所有SM在本次窗口内都拿到过工作，不代表每时每刻都满载。

H12 的平均活动比例约翻倍，即使活跃SM内occupancy/issue下降，elapsed Tensor指标仍上升且时间缩短。这支持“原先任务过少，拆分改善覆盖”的解释。H96 虽覆盖增加，但 eligible/issue/Tensor active 下降明显，不能将 grid 增大本身当作优化成功。

### 6.9 访存开销与剩余归因边界

| K2 访存指标 | H96 baseline | H96 P2 | H12 baseline | H12 P2 |
|---|---:|---:|---:|---:|
| DRAM带宽 / GB/s | 832.4147 | 879.3558 | 94.8084 | 112.1303 |
| L2 hit rate | 51.36% | 55.47% | 51.24% | 61.43% |
| DRAM总量估计 / MB | 1072.4165 | 1361.1584 | 121.8857 | 124.5059 |

原CSV没有 `dram__bytes.sum` 等总字节字段。第三行按**带宽×同一kernel的duration**计算，使用十进制单位，是派生估计而非原始硬件总字节计数；H96增加约26.92%，H12约2.15%。不同缓存命中表现解释了为什么不能把源代码请求量直接当成HBM总流量。

**[离线·源码/推导]** 每head每chunk的逻辑TMA加载量由17984 B变成31872 B，比例1.7722，包含两个分片重复读取共同输入。这个请求模型与上表HBM估计1.2692/1.0215是不同口径。

stall实测也有方向一致的变化，例如H96的 long scoreboard 指标0.44变0.69、wait 1.11变1.32。它们是日志中 `per_issue_active.ratio`，不能读成百分比等待时间。观测支持重复访存和可发射warp减少影响H96，但没有独立消融，无法给每种开销分配因果百分比。

机制归纳：**P2通过分散状态工作改善全GPU覆盖，同时付出共同数据重复读取、额外I/O角色和局部并行度变化的代价。H12与H96的收益差异由这两类变化的相对重要性决定。** 这是与数据一致的解释，不是把单个NCU指标作为充分原因。

来源：[结果表](../results/tables.md) 与 [证据说明](evidence.md)。

## 7. 工程决策与局限

当前保留官方路径作为默认。C在fixed H12值得保留独立候选，但H96三个shape不启用；已测单点不能推出通用小H阈值。A的真实Phase6集成和B的C32/C64实现已完成，不再列为“只做了microbench、完整对拍未做”。它们的当前完整性能均无收益。

本实验给官方选择提供的是有条件的支持：成熟的寄存器数据流和调度，在这几条明确候选下仍有优势。它没有证明官方实现全局最优，也没有证明tcgen05不能做得更快。A_control与B16显示，指令之外的数据组织、精度和算子融合成本对结论有实质影响。

仍未覆盖：A其他MMA阶段替换、A/B精细profile及独立开销消融、B融合kernel、跨设备/多seed/全部shape、真实激活与模型质量、TP通信和服务延迟。已有代码、完整对拍及正负性能结果足以完成本次三路线课程挑战；扩大结论时才需要对应的新证据。

## 8. 复核入口与证据身份

### 8.1 阅读与复核顺序

1. 阅读本报告取得结论，再按节号查看 [证据说明](evidence.md) 的推导、失败过程和完整输出。
2. 阅读 [harness 设计](design.md)（其待测标注属于发布当时；当前完成状态见本报告第5—6节）、[A 生成器](../flashkda_routes/build.py)、[B 实现](../flashkda_routes/routes/b_rescale.py) 与 [统一对拍](../flashkda_routes/correctness.py)，保留各自精度/实现范围。
3. 查看 [完整 forward 计时](../flashkda_routes/benchmark.py)、[结果表](../results/tables.md) 和 [v1 结果 JSON](../results/v1_three_routes.json)，按第6.1节复算。
4. 查看 [结果表](../results/tables.md) 与 [证据说明](evidence.md)，区分硬件原字段与推算量。此步骤在CPU上即可完成，不需要再次运行GPU。
5. 尚在B300的binary、`.ncu-rep`及数组按实验记录路径保留；读取归档无需重跑同配置实验。

本次没有改动已测CUDA源码和脚本，也没有重新跑benchmark。正式报告和讲解是从已保存证据整理的离线产物。

### 8.2 关键运行索引

| 证据 | UTC 运行 ID | 本仓库入口 |
|---|---|---|
| 官方fixed / varlen benchmark | `20260909T164935Z` / `20260909T165802Z` | 第5、11节 |
| 初始NCU与SASS | `20260909T172953Z` | 第5.14—5.19节 |
| 详细fixed NCU | `20260909T182132Z` | 第11—12节 |
| varlen6 launch | `20260910T052602Z` | 第10—11节 |
| 指数范围 | `20260910T061622976790Z` | 第14节 |
| Neumann数值 / MMA成本 | `20260910T064311694334Z` / `20260910T092501138041Z` | 第16、19节 |
| Phase6成功对照 | `20260910T125328470967Z` | 第24节 |
| state精度 | `20260910T133341103264Z` | 第26节 |
| P2编译和对拍 | `20260910T152803026613Z` | 第31节 |
| P2完整性能 | `20260910T162614136038Z` | 第33节 |
| P2机制NCU及离线恢复 | `20260910T165005117299Z` | 第35—36节 |
| 三路线short / full对拍 | `20260911T011018603611Z` / `20260911T013931783287Z` | 第39—40节 |
| 三路线完整forward性能 | `20260911T015639253562Z` | 第41节 |

### 8.3 关键SHA-256

| 对象 | SHA-256 | 来源说明 |
|---|---|---|
| baseline `.so` | `38d0498ca1bf67aa88369bf98763cf75df05572f04d2109b46083a4441969721` | 实机回传，binary在B300 |
| P2候选 `.so` | `6a543354033d95dd879f783dac07b340c6a50c91df3f4bde1fa6e655aa5a8272` | 实机回传，binary在B300 |
| P2 `candidate.patch` | `bb29989a93bf1afa137621563c8236216e0c5fc95cca4c67257896bdcba37bed` | 生成补丁，来源核对见实验记录 |
| P2完整性能原log | `57659a591a2415c08216ce66330c9092fe963b53ccab81f245be35803aadb7b8` | 回传文本恢复后本地复核 |
| P2逐轮JSON | `85a66c84ba52b71be79d9e552853840fca0f4fa0e7d61e1bcba8c06e3b7e36a0` | 本地恢复并复核 |
| NCU宽表CSV | `6854fe7957bb104bd90d770a4b000aed4687a310eaee44ea6c751b54c9cdf09f` | 原CSV字节hash匹配 |
| A full `.so` | `a2acd05e67281733c0943ac8c94bbdb035fa7e5fb0f6f7bbcd592dfd37a546be` | 实机运行，binary 在 B300 |
| 三路线full日志 | `2a935cd84c57ea0cbe61787d418e4f407a1a41514e3d03f03a697ea229d0311c` | 本地恢复并核对 |
| 三路线完整性能日志 | `d7e8a05a4c599ae1480b90303e3a06419b455c69adea57f2089952f39dce99b9` | 本地恢复并复算 |

## 9. 结论

官方 SM80 MMA 路径的合理性来自具体的小矩阵形状、寄存器数据流、范围与精度控制，以及递推的并行约束。新硬件提供了新的实现机会，但需要改变 tile、存储和任务组织，并以完整 forward 验证。

三条候选均完成真实实现与完整对拍。A将新指令接入真实递推但当前集成回退；B重标定方案在C32/C64可通过数值验证，但当前FP32组合实现成本过高；C在H12取得1.1547×完整forward加速，在H96回退。最终决策是保留默认路径，保留明确shape下的C候选，并将A/B进一步融合和布局优化作为后续研究，而不是根据架构代际承诺通用加速。

## 参考材料

正文中的实验数字均来自本项目实机回传与相应实验记录节。以下为算法、实现和工具口径的出处：

1. 课程任务书：三层任务与六个讨论点（不随公开仓库发布）。
2. [FlashKDA固定版本设计文档](../FlashKDA/docs/20260420-flashkda-v1-deep-dive.md)、[官方benchmark](../FlashKDA/benchmarks/bench_fwd.py)：原实现结构与复现口径。
3. [FLA naive.py](../fla_kda_ref/naive.py)：KDA参考递推。
4. [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-matrix-shape)：tcgen05形状及指令约束，访问于2026-09-10。在线版本9.4不等同实机CUDA13.0工具链版本，实际支持由编译与执行证据确认。
5. [NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-guide)：active/elapsed和metric后缀定义，访问于2026-09-10。

入门讲解：[from-scratch.zh.md](from-scratch.zh.md)。答辩材料与原始日志保留在实验机，不随公开仓库发布。

## 10. 后续版本（v2 / A3，公开整理时补充）

第 1—9 节保留 v1 的完整结论与数字，未因后续版本改写。本节补充 2026-09-11
之后完成的两轮工作，状态区分实机与离线。

### 10.1 A2/B2：同一路线继续优化后仍无 H96 正收益

**[实机]** run `20260911T080847138521Z`（Slurm job 25909）在 B300 上
完成 20 组完整 forward 比较、480 条原始计时和 80 项计时后检查；正确性来自
`20260911T072454980341Z_task3_ab_v2` 的 full PASS（350 项比较）。

| 实现 | v1 / ms | v2 / ms | v1/v2 | 同轮官方 / ms | 官方/v2 |
|---|---:|---:|---:|---:|---:|
| `A_control` | 2.722240 | 1.288688 | 2.112x | 1.030800 | 0.7999x |
| `A` | 2.914528 | 1.389152 | 2.098x | 1.031824 | 0.7428x |
| `B16` | 99.041443 | 11.796192 | 8.396x | 1.031472 | 0.0874x |
| `B32` | 120.475056 | 14.639808 | 8.229x | 1.030848 | 0.0704x |
| `B64` | 199.144356 | 26.473456 | 7.522x | 1.032032 | 0.0390x |

A2 把 U 只落一次共享内存、复用 MN-major descriptor、用片段式 TMEM 读回，相对
A v1 快 2.10 倍；但同数据流的 HMMA 控制组仍比官方慢 25.02%，A2 再慢 7.80%。
因此剩余差距主要来自共同数据路径与同步，而不是单条 tcgen05 的延迟。B2 用三个
融合 Triton kernel（normalize/relative coefficients/recurrence）把 B1 提速
7.5—11.1 倍，但 C32/C64 仍随 C 变慢，H96 上离官方还有 11—26 倍。

**结论不变**：v2 证明“认真优化后差距会缩小”，但没有把任何 H96 候选推过官方。
完整 20 组表见 [results/tables.md](../results/tables.md)。

### 10.2 A3：整个 K2 转向 SM100，只有离线证据

A3 改变优化范围：K1 保持原实现，K2 的所有矩阵乘一起转置，使 value 维承担
M=128，避免把 token 从 16 补到 64。Phase 1 合并为 `128x32x128`，Phase 3/4/6
分别变成 `128x16x16`、`128x16x16`、`128x128x16`；Phase 4 与 Phase 6 共用
completion barrier，每 chunk 三组 commit/wait，TMEM 每 CTA 只分配一次 256 列。

**[离线·编译与 SASS]** nvcc/ptxas 13.0.88 + CUTLASS `5c149f5`：14 个 K2 变体
全部编译成功；A3 每个变体 11 条 `UTCHMMA`、0 条 `HMMA`；HMMA 控制组 104 条
`HMMA`、0 条 `UTCHMMA`。A3 94—96 registers/thread，控制组 56—64，均无 spill。
真实 pinned CuTe host 布局枚举 `errors=0`。

**没有实机数字。** 目标 fixed H96 完整 forward ≥2x（约 0.516 ms），但 B300 的
scratch 文件系统在 NCU run `20260911T121125285990Z` 遇到 `ENOSPC`，本版未执行
GPU 对拍或计时。因此 A3 只能作为“已生成全 K2 SM100 机器码”的离线工作，不能
写成正收益。源码与离线证据见 [experiments/a3_full_k2_sm100](../experiments/a3_full_k2_sm100/README.md)。

### 10.3 版本结论

- v1：三条路线均有完整实机对拍与性能；C 在 fixed H12 取得 1.155x。
- v2：A2/B2 有完整实机对拍与性能；自身显著改善，H96 仍无正收益。
- A3：有完整离线编译、布局与 SASS 证据；无 GPU 结果。
- 三个版本的负结果都保留，且各自的控制组限定了可归因范围。
