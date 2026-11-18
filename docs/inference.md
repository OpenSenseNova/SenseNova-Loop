# Local inference

The repository provides an installable Python package, a `sensenova-loop` CLI,
and runnable [text](../examples/text/inference.py) and
[VQA](../examples/vqa/inference.py) examples. The organization follows
[SenseNova-U1](https://github.com/OpenSenseNova/SenseNova-U1); Loop accepts text
and images and produces text.

The examples and real-checkpoint tests in this guide target
`SenseNova-Loop-9B-Preview`. For the `SenseNova-Loop-9B` release, follow the
[README's Transformers and deployment instructions](../README.md#get-started).
Compatibility of the release checkpoint with this local package has not yet
been verified.

## Install

Use Python 3.10 or newer in a separate environment. Install a CUDA-compatible
PyTorch build first; the reference versions are declared in
[`pyproject.toml`](../pyproject.toml) (PyTorch 2.11, torchvision 0.26,
Transformers 5.18). The checked CUDA build is `2.11.0+cu130`.

```bash
python -m venv .venv
source .venv/bin/activate
# Select the PyTorch wheel appropriate for your machine before this step.
python -m pip install -e '.[dev]'
```

Download the Preview checkpoint or use an existing local directory:

```bash
hf download sensenova/SenseNova-Loop-9B-Preview \
  --local-dir models/SenseNova-Loop-9B-Preview
```

Keep the config, tokenizer, chat template, processor config, weight index and
every weight shard referenced by that index together. The Loop/mHC model
implementation is maintained in
[`src/sensenova_loop/models`](../src/sensenova_loop/models/). The loader registers
these classes and uses `trust_remote_code=False`; no checkpoint Python file is
needed for normal inference. Model weights are not included in this repository.

A checkpoint may declare `sensenova_loop_min_version` in its config. Loading a
checkpoint requiring a newer package fails before allocating model weights;
upgrade this package to the required version.

## Text and images

```bash
CUDA_VISIBLE_DEVICES=0 sensenova-loop \
  --model-path models/SenseNova-Loop-9B-Preview --local-files-only \
  --prompt 'What is 13 times 7? Reply with the number only.'

CUDA_VISIBLE_DEVICES=0 python examples/vqa/inference.py \
  --model-path models/SenseNova-Loop-9B-Preview --local-files-only \
  --image assets/loop-mhc-overview.jpg \
  --prompt 'Describe this architecture diagram.' \
  --max-new-tokens 512 --output outputs/diagram.txt
```

`python -m sensenova_loop` and `python examples/text/inference.py` accept the
same options as the installed CLI. Repeat `--image PATH` to supply multiple
images; paths are local files and their order is preserved. `--system-prompt`
sets an optional system instruction.

The default is greedy decoding with thinking disabled. Use `--think` to enable
thinking, and allow a larger `--max-new-tokens` budget for reasoning. Thinking
text is retained in the response. Use `--do-sample --temperature 0.7 --top-p 0.9
--seed 42` for sampling. `--output` saves UTF-8 text and the response is also
printed to stdout. Add `--top-k 20 --repetition-penalty 1.05` to tune decoding.
Run `sensenova-loop --help` for all options. U1-style aliases such as `--model_path`,
`--question`, `--max_new_tokens`, `--top_p`, `--top_k` and `--vram_mode` are accepted.

## JSONL batches and structured output

```bash
CUDA_VISIBLE_DEVICES=0 sensenova-loop \
  --model-path models/SenseNova-Loop-9B-Preview --local-files-only \
  --jsonl examples/text/data/samples.jsonl \
  --output-dir outputs/text --profile

CUDA_VISIBLE_DEVICES=0 sensenova-loop \
  --model-path models/SenseNova-Loop-9B-Preview --local-files-only \
  --jsonl examples/vqa/data/samples.jsonl --output outputs/vqa.jsonl
```

Batch mode loads the model once and processes requests sequentially. Each record
is an object with `prompt` (or VQA `question`) and optional `image`/`images`, `id`,
`system_prompt`, `seed`, `max_new_tokens`, `think`, `do_sample`, `temperature`,
`top_p`, `top_k`, and `repetition_penalty`. Image paths resolve relative to the
JSONL file. Alternatively use `messages` for text conversation history, without
`image`/`images`. A record's `system_prompt` overrides the CLI default. A supplied
system instruction replaces a leading system turn in `messages`, or is prepended
if none exists. Without an override, an existing system turn is preserved.
Empty lines are ignored; invalid records report their line
number before model loading. Input examples are provided for both tasks.

Results are printed and saved as UTF-8 JSONL. Each result retains source fields
and adds `answer`, `input_tokens`, `output_tokens` and `metrics`. `--output-dir`
writes `answers.jsonl`; the default batch destination is `outputs/answers.jsonl`.
For a single request, `--output-format json` selects the same structured record.
`--output` always follows the selected format.

`--profile` measures synchronized model loading and each request's preprocessing,
generation and decoding. Metrics include elapsed seconds, peak allocated CUDA
GiB and output tokens per end-to-end request second. Load/request summaries go to
stderr; stdout remains response text or JSONL. GPU memory reports include
resident weights and allocations already present in the process; a placement
budget is not a bound on total generation memory.

## Python API and conversation history

```python
from sensenova_loop.inference import LoopInference, build_messages

engine = LoopInference("models/SenseNova-Loop-9B-Preview", local_files_only=True)
messages = build_messages("Remember this code: 7319.")
answer = engine.generate(messages, max_new_tokens=64)
messages.append({"role": "assistant", "content": answer})
messages.append({"role": "user", "content": "What code did I give you?"})
print(engine.generate(messages, max_new_tokens=64))
```

Load once and call `generate` repeatedly. The caller maintains the message
history; each call processes that history with a fresh cache. For images use
`build_messages(question, images=["first.png", "second.png"])` or pass
Transformers image/text content blocks directly. The API accepts one
conversation per call. Tool execution, video input, serving and speculative
decoding are outside this inference entry point; see the root README for
external training and serving integrations.

`engine.generate_result(messages, profile=True)` returns a `GenerationResult`
with `.text`, `.input_tokens`, `.output_tokens` and `.metrics` without mutating
shared result state. For direct Transformers use, call `sensenova_loop.register()`
before loading with Auto classes:

```python
import sensenova_loop
from transformers import AutoConfig

sensenova_loop.register()
config = AutoConfig.from_pretrained("models/SenseNova-Loop-9B-Preview",
                                    trust_remote_code=False)
```

## Devices, memory and troubleshooting

- `--device cuda:0` selects a logical device after `CUDA_VISIBLE_DEVICES` is
  applied. For example, `CUDA_VISIBLE_DEVICES=7` with the default device uses
  physical GPU 7. The selected CUDA device is the only GPU used by this loader.
- `--vram-mode full` (default) keeps all weights on that device. `balanced` uses
  Accelerate automatic placement between that GPU and CPU; `--max-memory-gib 8`
  sets the GPU weight budget (default: 80% of currently free GPU memory). `low`
  keeps weights on CPU and uses Accelerate hooks to transfer weights for each
  module execution. Both offload modes need substantial host RAM and add transfer
  time. They support shared Loop visits; mHC routing, `A_log` and `dt_bias` retain FP32.
  If the balanced budget places all weights on CPU, GPU execution hooks are
  installed just as in low mode. Transient execution allocations can exceed the
  persistent weight budget.
  These modes adapt the U1 memory-management capability to Loop and do not
  implement U1's asynchronous prefetch/fast residency algorithm.
- In `low` mode Transformers can warn that the model is on `meta` while inputs
  are on CUDA. Accelerate hooks materialize offloaded weights during execution;
  keep inputs on the selected CUDA device rather than moving them to `meta`.
- With SenseNova-Loop-9B-Preview, short text inference used about 17.7 GiB of peak
  allocated CUDA memory on an H200 with BF16 and SDPA. Longer prompts, larger
  images and generation budgets
  require more memory; this is not a maximum-memory guarantee. CPU offload is
  opt-in as described above; tensor parallelism is not provided by this entry point.
- Keep `--dtype bfloat16` unless you need `float32`. The repository model implementation
  preserves mHC routing, `A_log` and `dt_bias` parameters in FP32 directly during
  checkpoint loading. Do not manually cast those parameters or the loaded model
  to a lower precision; upcasting rounded values cannot recover the checkpoint values.
  “BF16 backbone” describes the default loading policy. Checkpoint storage can
  differ: an FP32-stored backbone is rounded when loaded in BF16. Use
  `--dtype float32` to retain an FP32-stored backbone's parameter values.
- SDPA is the default attention backend; `--attn-backend eager` is also available.
  Optional `causal_conv1d` and `flash-linear-attention` kernels are not required
  for correctness. Transformers warns when using slower PyTorch fallbacks.
- If loading fails with an unknown architecture or missing symbols, activate
  the intended environment and reinstall the versions in `pyproject.toml`.
  `--local-files-only` requires the complete checkpoint to be present locally.
- If you run out of memory, reduce image sizes, history or `--max-new-tokens`.
  Requests exceeding the checkpoint's context limit are rejected before
  generation; a request within that limit can still exhaust device memory.

## Tests

CPU unit tests do not download weights:

```bash
python -m pytest -q -m 'not integration'
```

Opt in to the Preview real-checkpoint tests on an available GPU:

```bash
CUDA_VISIBLE_DEVICES=0 \
SENSENOVA_LOOP_MODEL=/path/to/SenseNova-Loop-9B-Preview \
  python -m pytest -q -m integration
```

For an exported checkpoint without Python model files, set
`SENSENOVA_LOOP_REFERENCE_SOURCE` to the original Preview checkpoint directory.
The independent implementation comes from that directory; its configuration
and weights still come from `SENSENOVA_LOOP_MODEL`. The comparison restores the
reference's FP32 mHC, `A_log` and `dt_bias` directly from those weights and uses
BF16 for both backbones. It verifies this mixed precision policy, including
original values for the protected groups.

These cover text answers, independent requests, conversation history, ordered
multi-image understanding, FP32 routing and linear attention parameters, and 56 execution-cache slots.
They also compare the repository implementation with the original checkpoint
implementation and exercise text/VQA in both CPU offload modes. CPU tests cover
local Auto-class save/load, shared-layer cached/full-forward equivalence, mHC
precision, compatibility gates, request validation and JSONL/output behavior.
The Preview integration checks use short inputs; they do not validate the
release checkpoint, long-context quality or serving performance.
See the [U1 alignment record](u1_alignment.md) for the corresponding scope.
