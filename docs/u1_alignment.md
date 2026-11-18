# U1 alignment / U1 对齐记录

Reference / 参考版本:
[OpenSenseNova/SenseNova-U1, 414f82e](https://github.com/OpenSenseNova/SenseNova-U1/tree/414f82e9ff458a33190c47639a798994cc53d2a0).

This repository applies the reference's inference engineering capabilities to
Loop's text and image-understanding model. It uses Loop's own architecture and
checkpoint format. / 本仓库将参考项目的推理工程能力适配到 Loop 的文本和图像理解模型，
模型架构及 checkpoint 格式使用 Loop 自身的实现。

| Reference capability / 参考能力 | Loop implementation / 对应实现 | Verification / 验证 |
| --- | --- | --- |
| Installable `src` package, task examples, guides / 可安装源码包、任务示例与指南 | `pyproject.toml`, `src/sensenova_loop`, text/VQA examples, bilingual guides | Wheel build, CLI launch and local links / 构建、入口和本地链接 |
| Repository-owned model classes and Auto registration / 本地模型与 Auto 注册 | `models/modeling_sensenova_loop.py`, explicit `sensenova_loop.register()`; engine registers automatically | Native save/load without checkpoint Python, real-checkpoint output comparison / 无 checkpoint Python 的保存加载及真实模型输出对比 |
| Minimum package version gate / 包版本兼容检查 | `sensenova_loop_min_version` checked during config construction | Future-version rejection before model allocation / 分配权重前拒绝未来版本 |
| VQA/text generation and history / 视觉问答、文本及历史 | Processor, `LoopInference.generate`, caller-owned histories | Arithmetic, repeated requests, history, ordered image understanding / 计算、重复请求、历史与图像顺序 |
| JSONL input and result records / JSONL 输入与结果 | Sequential model reuse; prompt/question, image/images, text messages, per-record options; UTF-8 JSONL | Input validation/relative images, result field and CLI tests / 输入验证、相对路径及 CLI 结果字段 |
| Sampling controls / 采样控制 | Greedy, temperature/top-p/top-k/repetition penalty, seed, thinking; U1 option aliases | API forwarding and real sampled/thinking CLI / API 参数传递和真实采样思考测试 |
| Load/generation profiler / 加载与生成统计 | Synchronized phase latency, peak allocated CUDA GiB, end-to-end output token rate | CPU metric tests and real CUDA profiles / CPU 指标与真实 CUDA 测量 |
| GPU/CPU memory modes / 显存管理 | `full`; Accelerate GPU/CPU placement (`balanced`, GPU budget); CPU weight hooks (`low`) | Real text and image-understanding inference in both offload modes / 两种卸载模式的真实文本及视觉问答 |

The model source originates from `modeling_sensenova_loop.py` in the released
SenseNova-Loop-9B-Preview checkpoint. The numerical implementation is retained;
local package imports, registration and the version gate enable maintenance in
this repository. Empty foreign-cache admission also handles attention-only
caches without querying a nonexistent recurrent layer. Tests cover physical-layer weight sharing, independent virtual
cache slots, cached/full-forward equivalence and FP32 mHC routing.

模型源码来自发布 checkpoint 的 `modeling_sensenova_loop.py`，保留数值实现，补充本地包
导入、注册和版本检查，以便在仓库维护。空外部缓存的处理也支持无循环状态层的全注意力
缓存，避免查询不存在的状态。测试覆盖物理层权重共享、独立执行缓存、
缓存/完整前向一致性以及 mHC 路由 FP32 精度。

Loop's `balanced` and `low` modes use Accelerate dispatch rather than U1's custom
asynchronous prefetch/fast-residency implementation; their names describe memory
placement choices, not identical latency or residency algorithms. No U1 memory
or performance results are claimed for Loop. Package registration is explicit
for raw Transformers callers and automatic for the inference engine, preserving
lightweight CLI help.

Loop 的 `balanced` 和 `low` 使用 Accelerate dispatch，未复用 U1 的自定义异步预取及
fast 驻留实现；模式名称描述内存放置选择，不代表相同的延迟或驻留算法，也不套用 U1 的
性能和显存测量。直接使用 Transformers 时需显式注册；推理引擎自动注册，CLI 帮助无需
导入模型依赖。

U1 image synthesis/editing/interleaving, its LoRA/GGUF formats and its training
model implementation are not transferred to this text-output Loop checkpoint.
Loop's native MTP predictor and speculative decoder are implemented for its own
checkpoint; they are separate from U1's training-side MTP modules.
Loop's official external training and serving integrations remain documented in
the root README. Short correctness tests do not establish 256K-context quality,
throughput or concurrent-serving performance.

U1 的图像生成、编辑、交错输出、其 LoRA/GGUF 格式及训练模型实现未迁入这个文本输出的
Loop checkpoint。Loop 官方的外部训练与部署集成仍在根 README 说明。短输入正确性测试
不能证明 256K 上下文质量、吞吐或并发服务性能。

Loop 的原生 MTP 预测分支及推测解码使用其自身 checkpoint，
与 U1 训练端的 MTP 模块分开实现。
