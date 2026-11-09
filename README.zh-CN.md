# Blackwell 上的 FlashKDA：复现与三条 SM100 路线

对 [FlashKDA](https://github.com/MoonshotAI/FlashKDA)（Kimi Delta Attention
前向 kernel）在 NVIDIA B300（SM103）上的独立研究。官方 kernel 即使在
Blackwell 上仍使用 SM80 的 `mma.sync`。本仓库复现这一事实，量化设计为何停在
这里，并实现三条替代路线，给出完整 KDA 对拍与完整前向计时。

一句话结论：**对这个形状，官方 SM80 路径很难被击败。** 仅替换 Phase 6 的
tcgen05、真正的 CHUNK=32/64 重标定、value 列分片三条路线都已实现并实测；
只有列分片在每卡小头数 `H=12` 上取得正收益（1.155x），在 `H=96` 上回退。
负结果连同控制组一起保留。

> 研究生 HPC seminar 的研究产物，与 MoonshotAI 无关。所有 GPU 数字来自同一张
> B300 SXM6 AC（148 SM，CUDA 13.0，PyTorch 2.14），不是模型质量或服务延迟结论。

## 主要结果

完整前向，`T=8192, H=96, D=128`，每个 Graph 一次完整 forward，12 轮交替取中位数。
比值大于 1 表示比同轮官方 baseline 快。

| 实现 | 改了什么 | 官方 / 候选 | 结论 |
|---|---|---:|---|
| A（v1） | Phase 6 的 MMA 换成 tcgen05，C16 与前序阶段不变 | 0.353x | 慢 2.83 倍 |
| A2（v2） | 同一路线，U 只落一次 SMEM、MN-major descriptor、片段式 epilogue | 0.743x | 比 A v1 快 2.10 倍，仍慢于官方 |
| B16/B32/B64（v1） | 真正的 C32/C64、相对前缀重标定、FP32 下三角求解、PyTorch 组合实现 | 0.0104x / 0.0086x / 0.0052x | 数值有效，但太慢 |
| B16/B32/B64（v2） | 三 kernel 融合 Triton 实现 | 0.087x / 0.070x / 0.039x | 比 v1 快 7.5-11.1 倍，仍慢 |
| C | value 维拆成两个 64 列 CTA | H96 0.884x | 回退 |
| C | 同上，`fixed H12` | **1.155x** | **唯一完整前向正收益** |

官方 kernel 在同形状下比 FLA Triton `chunk_kda` 快 2.29x / 2.72x / 3.27x
（fixed / varlen6 / varlen8），复现了公开范围。

## 六个发现

1. **`CHUNK=16` 是三个约束，不是一个。** 累计 gate 大时指数范围先破；16x16
   Neumann 逆在低精度下先失去精度；MMA tile 形状与 `m16n8k16` 匹配。“谁先破”
   取决于输入，所以放大 CHUNK 不是免费旋钮。
2. **tcgen05 形状适合 Phase 6，但不适合周围数据流。** Phase 6 是
   `128x16 @ 16x128`，合法的 `m128n128k16`，占 K2 MMA FLOP 的 30.8%。独立
   probe 在摊薄 TMEM 开销后有 1.11x；但接真实 U 后，搬运与同步成本超过指令
   收益。同数据流的 HMMA 控制组本身已比 baseline 慢 1.95-3.14 倍，因此不能把
   回退全归因于 tcgen05。
3. **value 列是干净的并行轴。** 分片 `p` 的状态更新只需要 `S^(p)` 和共享的
   `q/k/g/beta`，key 维内积保持完整，不需要跨分片归约。它让 CTA 数翻倍，但
   重复读取 q/k/gate 并增加 I/O warp；小 grid（H12）赢，大 grid（H96）输。
4. **“compute-bound 还是 memory-bound”对整个 forward 是错误问题。** K1 接近
   两条 roofline；K2 的 compute 与 DRAM 吞吐都低，受 96 个 CTA、18.75% 理论
   occupancy 和长递推限制。NCU 的 active-cycle 与 elapsed-cycle 分母回答不同
   问题，本文从不混用。
5. **BF16 state 需要对抗性测试。** Q16 模型（只在 chunk 边界舍入 BF16）在随机
   输入上 NRMSE 0.26-0.78%，在慢写入 head 上 2.20-3.77%。只查 NaN 会漏掉；
   验证覆盖弱遗忘、慢写入、后段轨迹和最终 state。
6. **架构专版只应在 shape gate 后成立。** 证据支持保留官方默认路径、保留 C 作为
   小每卡头数的候选；不支持通用 SM100 替换。

## 目录

```text
flashkda-sm100-routes/
|-- README.md / README.zh-CN.md
|-- docs/
|   |-- report.zh.md           正式报告（含 v2/A3 附录）
|   |-- from-scratch.zh.md     KDA 与 GPU 基础讲解
|   |-- design.md              harness 设计与路线机制
|   `-- evidence.md            证据模型、run ID、哈希、边界
|-- flashkda_routes/           可运行包（offline/correctness/benchmark）
|-- experiments/               A2/B2/A3 后续实现（状态分开标注）
|-- results/                   全部结果表与 JSON
|-- scripts/                   fetch / run 脚本
|-- tests/                     CPU 代数测试
|-- FlashKDA/                  gitignore 的上游 checkout（固定 commit）
`-- fla_kda_ref/               内置的 FLA KDA 参考（MIT，固定快照）
```

## 快速开始

### 环境

- Linux + NVIDIA Blackwell GPU（已发表数字使用 B300，`sm_103a`）。纯 CPU 可跑
  offline 检查。
- CUDA 13.0+（`nvcc`、`cuobjdump`）、Python 3.10+、PyTorch 2.4+（实测 2.14）、
  Ninja；CUTLASS 与 JIT 构建需要数 GB 磁盘。

### 1. 拉取并安装固定 baseline

```bash
git clone https://github.com/Echomosome/flashkda-sm100-routes.git flashkda-sm100-routes
cd flashkda-sm100-routes
bash scripts/fetch_upstream.sh          # FlashKDA @ 1ce47ea + CUTLASS @ 5c149f5
python -m venv .venv && source .venv/bin/activate
pip install torch          # 选择与 CUDA 13 匹配的 wheel
pip install -r requirements.txt
pip install -v --no-build-isolation ./FlashKDA
```

### 2. 离线检查（无需 GPU）

```bash
python -m flashkda_routes.cli offline
python -m pytest -q
```

offline 会生成 A/A_control/C 候选源码，检查每个 patch anchor 与 K1/utils 字节
不变量，验证 packed 地址映射，并用独立 NumPy token 递推对拍 C16/32/64 分块
代数（210 个 case）。它不编译 CUDA。

### 3. 完整 KDA 对拍

```bash
python -m flashkda_routes.cli correctness --routes A B C --suite full
```

每条路线独立进程。`full` 每个实现 38 个 case：7 种 state 接口、定长/变长、
batch 2、尾长 1/15/16/17/31/32/33/63/64/65、强/弱/零 gate、零 q/k、H12/H96
短 grid、四个 T8192 形状、弱遗忘长链。A 与 C 要求逐位等于 baseline；B 要求满足
预先写死的 FLA envelope。失败保留输入/输出数组并写 `FAIL`，阈值不因结果放宽。

### 4. 完整前向性能

```bash
python -m flashkda_routes.cli benchmark --routes C \
  --validated runs/build/<PASS-full-run>
```

benchmark 拒绝给没有 `PASS/full` verdict 的 build 计时；重新核对 runner 源码
哈希与二进制哈希；先做 eager/Graph 一致；计时后毒化输出并重放验证覆盖。报告
12 轮交替的中位数，原始样本保留在 `bench_C.json`。

### 5. 独立 probes

```bash
python -m flashkda_routes.probes.chunk_range --paper-only
python -m flashkda_routes.probes.neumann_probe --offline-check
python -m flashkda_routes.probes.neumann_mma_bench --offline-check
python -m flashkda_routes.probes.phase6_microbench --offline-check
python -m flashkda_routes.probes.state_precision --mode offline
```

这些是六个讨论点背后的原始测量脚本，独立于 harness，保留原始中文注释。

## 结果与证据

- [results/tables.md](results/tables.md)：全部表格，含 v1 24 组、v2 20 组、
  控制组差值和 A3 状态。
- [docs/evidence.md](docs/evidence.md)：run ID、二进制哈希、哪些是实机测量、
  哪些是离线推导，以及刻意不声称的内容。
- [docs/report.zh.md](docs/report.zh.md)：正式中文报告（含 v2/A3 附录）。
- [docs/from-scratch.zh.md](docs/from-scratch.zh.md)：从零讲 KDA、GPU 层级、
  chunk 与三条路线。

## 范围与限制

- 全部 GPU 结果来自一张 B300 SXM6 AC；没有其他 SM100 设备、没有多卡 TP
  通信、没有服务延迟。
- `H12` 是 TP8 下每卡头数（`96/8`），是单 GPU 形状，不是 8 卡部署测量。
- B 路线是 PyTorch CUDA 组合实现（v1）或三 kernel Triton 实现（v2），不是
  融合 SM100 kernel；它的成本不能给假设中的融合实现定上界。
- A 路线只替换 Phase 6。A3（整个 K2 转置）能编译并生成 `UTCHMMA` 机器码，
  但**没有 GPU 对拍与计时**，作为明确标注的离线工作收录。
- 没有模型级精度、下游质量结论，也不声称官方 kernel 全局最优。

## 许可与致谢

本仓库原创代码与文档为 MIT。内置的 `fla_kda_ref/` 快照为 MIT，版权
Songlin Yang、Yu Zhang、Zhiyuan Li；FlashKDA 许可证见
`third_party_flashkda_LICENSE.txt`。FlashKDA 本体不内置：fetch 脚本固定
`1ce47ea`，所有生成 patch 在改文件前检查记录的源码哈希。

本研究完成于 WMHPC x LCPU AI Infra seminar。感谢课程提供的固定 FlashKDA
快照与 B300 机时。
