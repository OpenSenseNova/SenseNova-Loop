# 本地推理

仓库提供可安装的 Python 包、`sensenova-loop` 命令行工具，以及可运行的
[文本](../examples/text/inference.py)和[视觉问答](../examples/vqa/inference.py)示例。
组织方式参考 [SenseNova-U1](https://github.com/OpenSenseNova/SenseNova-U1)；
Loop 接收文本和图像，输出文本。

本指南的示例及真实模型测试面向 `SenseNova-Loop-9B-Preview`。
正式版 `SenseNova-Loop-9B` 的 Transformers 加载与服务部署见
[README 入门说明](../README_CN.md#get-started)；本地包与正式版 checkpoint 的兼容性
尚未验证。

## 安装

使用 Python 3.10 或更新版本，并创建独立环境。先安装与 CUDA 环境匹配的
PyTorch；参考版本统一维护在 [`pyproject.toml`](../pyproject.toml) 中
（PyTorch 2.11、torchvision 0.26、Transformers 5.18）。已验证的 CUDA 构建为
`2.11.0+cu130`。

```bash
python -m venv .venv
source .venv/bin/activate
# 执行下一步前，先选择适合本机的 PyTorch wheel。
python -m pip install -e '.[dev]'
```

下载 Preview 权重，或直接使用已有的本地模型目录：

```bash
hf download sensenova/SenseNova-Loop-9B-Preview \
  --local-dir models/SenseNova-Loop-9B-Preview
```

模型目录需要完整保留配置、分词器、聊天模板、处理器配置、权重索引，以及该索引引用的
全部权重分片。
Loop/mHC 实现在 [`src/sensenova_loop/models`](../src/sensenova_loop/models/) 中维护。
加载器注册本地模型类并使用 `trust_remote_code=False`；正常推理无需 checkpoint 中的
Python 文件。本仓库不包含权重。

checkpoint 可在配置中声明 `sensenova_loop_min_version`。如果所需版本高于已安装包版本，
加载会在分配模型权重前失败；此时需先升级本仓库的包。

## 文本与图像

```bash
CUDA_VISIBLE_DEVICES=0 sensenova-loop \
  --model-path models/SenseNova-Loop-9B-Preview --local-files-only \
  --prompt '13 乘以 7 等于多少？只回答数字。'

CUDA_VISIBLE_DEVICES=0 python examples/vqa/inference.py \
  --model-path models/SenseNova-Loop-9B-Preview --local-files-only \
  --image assets/loop-mhc-overview.jpg \
  --prompt '描述这张架构图。' \
  --max-new-tokens 512 --output outputs/diagram.txt
```

`python -m sensenova_loop` 和 `python examples/text/inference.py` 支持与命令行工具
相同的选项。重复使用 `--image PATH` 可输入多张本地图像，输入顺序会保留。
`--system-prompt` 可设置系统指令。

默认采用贪心解码并关闭思考。`--think` 开启思考模式，此时应增大
`--max-new-tokens`，输出会保留思考内容。采样可使用
`--do-sample --temperature 0.7 --top-p 0.9 --seed 42`。
`--output` 将回复保存为 UTF-8 文本，同时仍会在标准输出中打印回复。
还可通过 `--top-k 20 --repetition-penalty 1.05` 调整解码。
全部选项可通过 `sensenova-loop --help` 查看。兼容 U1 风格别名，例如 `--model_path`、
`--question`、`--max_new_tokens`、`--top_p`、`--top_k` 和 `--vram_mode`。

## JSONL 批处理与结构化输出

```bash
CUDA_VISIBLE_DEVICES=0 sensenova-loop \
  --model-path models/SenseNova-Loop-9B-Preview --local-files-only \
  --jsonl examples/text/data/samples.jsonl \
  --output-dir outputs/text --profile

CUDA_VISIBLE_DEVICES=0 sensenova-loop \
  --model-path models/SenseNova-Loop-9B-Preview --local-files-only \
  --jsonl examples/vqa/data/samples.jsonl --output outputs/vqa.jsonl
```

批处理只加载一次模型，按记录顺序逐条推理。每行为一个对象，使用 `prompt` 或视觉问答
的 `question`，可选字段为 `image`/`images`、`id`、`system_prompt`、`seed`、
`max_new_tokens`、`think`、`do_sample`、`temperature`、`top_p`、`top_k`、
`repetition_penalty`。图像路径相对于 JSONL 文件解析。也可用 `messages` 输入纯文本
多轮历史，此时不能同时使用 `image`/`images`。记录中的 `system_prompt` 优先于 CLI 默认值；
提供的系统指令会替换 `messages` 开头的系统消息，若没有则插入一条。未提供覆盖值时，
保留历史中已有的系统消息。忽略空行；非法记录会在加载模型前报告行号。
两种任务都提供了输入样例。

结果以 UTF-8 JSONL 打印并保存，保留输入字段，增加 `answer`、`input_tokens`、
`output_tokens` 和 `metrics`。`--output-dir` 下生成 `answers.jsonl`，批处理默认保存到
`outputs/answers.jsonl`。单条请求可用 `--output-format json` 选择同样的结构化输出；
`--output` 保存当前格式。

`--profile` 同步测量模型加载，以及每条请求的预处理、生成和解码。统计包括耗时秒数、
CUDA 峰值已分配显存 GiB、生成 token 数除以请求总耗时。加载和请求汇总写到标准错误，
标准输出保留回复文本或 JSONL。显存统计包含常驻权重及进程已有分配；权重放置预算不代表
生成总显存上限。

## Python API 与多轮对话

```python
from sensenova_loop.inference import LoopInference, build_messages

engine = LoopInference("models/SenseNova-Loop-9B-Preview", local_files_only=True)
messages = build_messages("记住这个编号：7319。")
answer = engine.generate(messages, max_new_tokens=64)
messages.append({"role": "assistant", "content": answer})
messages.append({"role": "user", "content": "我给你的编号是什么？"})
print(engine.generate(messages, max_new_tokens=64))
```

模型只需加载一次，可多次调用 `generate`。对话历史由调用方维护；每次调用都会
重新处理传入历史并创建新的缓存。图像可通过
`build_messages(question, images=["first.png", "second.png"])` 输入，或直接传入
Transformers 的图像、文本内容块。API 每次接收一段对话。
工具执行、视频输入、服务部署和推测解码不在此推理入口的支持范围内；
外部训练与部署集成见根目录 README。

`engine.generate_result(messages, profile=True)` 返回 `GenerationResult`，包含 `.text`、
`.input_tokens`、`.output_tokens` 和 `.metrics`，不会通过共享状态保存上次结果。
直接使用 Transformers Auto 类时，先显式注册本地实现：

```python
import sensenova_loop
from transformers import AutoConfig

sensenova_loop.register()
config = AutoConfig.from_pretrained("models/SenseNova-Loop-9B-Preview",
                                    trust_remote_code=False)
```

## 设备、显存与排错

- `--device cuda:0` 指定 `CUDA_VISIBLE_DEVICES` 映射后的逻辑设备。
  例如 `CUDA_VISIBLE_DEVICES=7` 搭配默认设备将使用物理 7 号卡。
  加载器只使用这一张 CUDA 卡。
- `--vram-mode full`（默认）将全部权重保留在该卡。`balanced` 使用 Accelerate 在这张
  GPU 和 CPU 间自动放置权重；`--max-memory-gib 8` 指定 GPU 权重预算，默认是当前空闲
  显存的 80%。`low` 将权重保留在 CPU，通过 Accelerate hooks 在各模块执行时搬运权重。
  两种卸载模式均需要较多主机内存，并增加传输耗时；支持共享层的多次执行，mHC、`A_log`
  和 `dt_bias` 参数保持 FP32。balanced 预算使全部权重放在 CPU 时，也会安装 GPU 执行 hooks；临时执行分配
  可能超过常驻权重预算。这是 U1 显存管理能力在 Loop 上的适配，不包含 U1 的异步预取和
  fast 驻留算法。
- `low` 模式下 Transformers 可能提示模型位于 `meta`、输入位于 CUDA。
  Accelerate hooks 会在执行时装载卸载权重；输入应保留在选定 CUDA 设备上，请勿移到 `meta`。
- Preview 在 H200 上使用 BF16、SDPA 进行短文本推理时，峰值已分配显存约为 17.7 GiB。
  更长的输入、更大的图像和更高的生成预算需要更多显存，该数值不代表显存上限。
  CPU 卸载需按上文显式启用；此入口不提供张量并行。
- 默认使用 `--dtype bfloat16`，也可选择 `float32`。仓库模型实现会在加载 checkpoint 时直接
  保留 mHC 路由、`A_log` 和 `dt_bias` 参数的 FP32 精度。请勿将这些参数或已加载的模型
  手动转为低精度；将已舍入的值转回 FP32 无法恢复 checkpoint 中的原始值。
  “BF16 主干”描述默认加载策略，checkpoint 的存储类型可能不同；将原本以 FP32 存储的
  主干按 BF16 加载仍会舍入。需要保留该主干的 FP32 参数值时，使用 `--dtype float32`。
- 默认注意力后端为 SDPA，也可使用 `--attn-backend eager`。
  `causal_conv1d` 和 `flash-linear-attention` 是可选加速依赖；未安装时仍可正确
  运行，Transformers 会提示正在使用较慢的 PyTorch 实现。
- 如果出现未知架构或缺少符号的错误，请确认已激活正确环境，并按
  `pyproject.toml` 重新安装依赖。使用 `--local-files-only` 前需确保模型文件完整。
- 显存不足时，可缩小图像、减少对话历史或降低 `--max-new-tokens`。
  超过模型上下文限制的请求会在生成前被拒绝；未超过限制也可能耗尽显存。

## 测试

CPU 单元测试不会下载权重：

```bash
python -m pytest -q -m 'not integration'
```

可在空闲 GPU 上显式启用面向 Preview 的真实模型测试：

```bash
CUDA_VISIBLE_DEVICES=0 \
SENSENOVA_LOOP_MODEL=/path/to/SenseNova-Loop-9B-Preview \
  python -m pytest -q -m integration
```

对于未附 Python 模型文件的导出 checkpoint，将 `SENSENOVA_LOOP_REFERENCE_SOURCE`
设为原始 Preview checkpoint 目录。独立实现来自该目录，其配置和权重仍来自
`SENSENOVA_LOOP_MODEL`。测试会直接从这些权重恢复参考实现的 FP32 mHC、`A_log`
和 `dt_bias`，两边的主干均使用 BF16，验证该混合精度策略及受保护参数组的原始值。

测试覆盖文本回答、独立请求、多轮历史、多图顺序理解、FP32 路由和线性注意力参数、
56 个执行缓存槽；还会比较本地实现与原 checkpoint 实现，并验证两种 CPU 卸载模式的
文本和视觉问答。CPU 测试覆盖本地 Auto 类保存/加载、共享层缓存与完整前向的一致性、
mHC 精度、版本兼容检查、请求验证和 JSONL/输出行为。Preview 集成测试使用短输入，
不代表正式版 checkpoint 验证、长上下文质量或服务性能测试。
对应范围见 [U1 对齐记录](u1_alignment.md)。
