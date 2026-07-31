# Java LLM 10 Engineering Examples — Benchmark Results

> Generated from `distribute.BenchmarkLlm10Examples`  
> Source mapping: `org/lance/ipc/llm.md` (Python) → `org.bytedeco.pytorch.llm.*` (Java)  
> Result: **PASS=125  FAIL=0  SKIP=0**  wall ≈ 34.5 s

## Run

```bash
CP=target/classes:$(cat target/cp.txt):target/pytorch.jar:target/pytorch-macosx-arm64.jar
# + openblas macosx-arm64 jar + java.library.path natives
javac -cp "$CP" -d target/classes src/main/java/org/bytedeco/pytorch/llm/peft/{PeftModel,LoraConfig,LoraLinear}.java \
  src/main/java/org/bytedeco/pytorch/llm/bitsandbytes/QLoRA.java \
  src/main/java/org/bytedeco/pytorch/llm/trl/loss/SFTLoss.java
javac -cp "$CP" -d target/samples-compile samples/BenchmarkLlm10Examples.java
java --enable-native-access=ALL-UNNAMED -Xmx4g -cp target/samples-compile:$CP distribute.BenchmarkLlm10Examples
```

## Python → Java API parity (exercised)

| Python (llm.md) | Java |
|---|---|
| `datasets.Dataset.from_list/map/train_test_split` | `HfDataset.fromList/map/trainTestSplit` |
| `transformers.AutoModelForCausalLM/AutoTokenizer/GenerationConfig` | same package under `llm.transformers` |
| `peft.LoraConfig / get_peft_model / print_trainable_parameters / save_pretrained / from_pretrained / merge_and_unload` | `PeftModel` + `LoraConfig` (snake+camel aliases) |
| `bitsandbytes.BitsAndBytesConfig` + QLoRA | `BitsAndBytesConfig` + `QLoRA.Session` |
| `trl` SFT/DPO + TrainingArguments-like | `SFTTrainer`/`DPOTrainer` + `SFTConfig`/`DPOConfig` |
| `accelerate.Accelerator` | `Accelerator.prepare/waitForEveryone/unwrapModel` |
| safetensors / GGUF / vLLM | `SafeTensors` / `GGUFWriter` / `vllm.LLM` |

## 10 examples

| # | Paradigm | Key metrics (this run) |
|---|---|---|
| 1 | Full-param SFT | loss 5.735→5.730, 29 tensors saved, gen_len=12, 41.7 ms / 4 steps |
| 2 | LoRA SFT + merge | adapters=8, trainable=16 384 (10.4%), merge+reload OK, 12.3 ms |
| 3 | QLoRA 4bit NF4 | quant_layers=8, loss 5.800→5.526, adapter+merged save OK |
| 4 | DPO | beta=0.1, loss≈0.58, adapters=4 |
| 5 | Continual pretrain | corpus=4, domain LM, full safetensors export |
| 6 | Multimodal VL SFT | MediaInput+MultimodalPrompt, real VL dir Qwen3-VL-2B-FP8 present |
| 7 | Accelerator LoRA | CPU prepare/unwrap, 1 process, adapter saved |
| 8 | Grad accum ×8 + ckpt | 16 micro-steps, trainable%≈4.17, Unsloth ckpt flag OK |
| 9 | LoRA→ST→GGUF→vLLM | GGUF 231 KB; tiny qwen2 OK; **real Qwen2.5-0.5B chat** reply_len=62 |
| 10 | Multi-turn + stream | stream 12 toks; real tok vocab≈151665, multi_turn_tokens=49 |

## Real models smoke

- Qwen2.5-0.5B-Instruct: load matched=290, chat non-empty (`2`)
- GPT-2: generate out_len=86
- Local snapshots also present: Llama-3.2-1B, DeepSeek-R1-1.5B, GLM-edge-1.5B, Qwen3-VL-2B-FP8

## API gaps filled this session

1. `PeftModel.get_peft_model` / `getPeftModel` (CausalLM `attachLora` + freeze base)
2. `print_trainable_parameters` / `save_pretrained` / `from_pretrained` / `merge_and_unload` (+ snake aliases)
3. `LoraConfig` snake aliases: `lora_alpha`, `lora_dropout`, `target_modules`, `task_type`, `use_rslora`
4. `SFTLoss`: force Long labels (CE dtype) + `ScalarType.intern()`
5. `LoraLinear.merge/unmerge` + adapter `copy_`: clear `requires_grad` before in-place (libtorch check_inplace)
6. `QLoRA.Session.loadAdapter` same safe copy
7. Benchmark harness: Long `input_ids`, safe post-train safetensors export, Accelerator `cpu(true)`, vLLM tiny=`qwen2`

## Raw summary

```
# LLM-10 EXAMPLES BENCHMARK SUMMARY
######################################################################
PASS=125  FAIL=0  SKIP=0  wall_ms=34466.4

Example reports:
  * D0-datasets | fromList→map(chat_template)→trainTestSplit OK rows=2 split=DatasetDict({test: 1 rows, train: 1 rows})
  * D0-parity | datasets/peft/bnb/trl config surface aligned with llm.md Python imports
  * Ex1 FullSFT | steps=4 first_loss=5.7349 last_loss=5.7304 saved=29 tensors gen_len=12
  * Ex2 LoRASFT | adapters=8 trainable=16384 total≈157440 last_loss=5.7452 merged_tensors=45
  * Ex3 QLoRA | adapters=8 trainable=16384 first=5.7999 last=5.5259 quant_layers=8
  * Ex4 DPO | beta=0.1 adapters=4 last_loss=0.5816 rows=2
  * Ex5 ContinualPretrain | corpus=4 last_loss=5.6349 adapters=8
  * Ex6 VL-weights | found real VL snapshot: Qwen__Qwen3-VL-2B-Instruct-FP8
  * Ex6 MultimodalSFT | rows=3 processor_ok=true adapters=4 last_loss=6.3791
  * Ex7 Accelerator | device=org.bytedeco.pytorch.Device[address=0x75f5b2340,position=0,limit=1,capacity=1,deallocator=org.bytedeco.javacpp.Pointer$NativeDeallocator[ownerAddress=0x75f5b2340,deallocatorAddress=0x17086b03c]] processes=1 adapters=4 last_loss=5.7440
  * Ex8 GradCheckpoint | accum=8 micro=16 adapters=4 last_loss=5.7356 trainable%=4.1739
  * Ex9 GGUF | file=model.gguf bytes=231488 tensors_meta≈8
  * Ex9 vLLM-tiny | requests=1 ok kind=qwen2
  * Ex9 vLLM-real | model=Qwen__Qwen2.5-0.5B-Instruct reply_len=62
  * Ex9 DeployChain | LoRA→merge→safetensors(37)→GGUF→vLLM OK
  * Ex10 real-tok | vocab≈151665 multi_turn_tokens=49
  * Ex10 MultiTurnStream | rows=2 turns0=4 last_loss=5.6564 stream_chars=38 split=DatasetDict({test: 1 rows, train: 1 rows})
  * D-Matrix | combos=8 (lr × accum × r)
  * Real-Qwen | matched=290 reply_len=1
  * Real-GPT2 | out_len=86

Timings:
  Ex1 FullSFT                                 total=   41.65 ms  steps=4  per_step= 10.412 ms
  Ex2 LoRASFT                                 total=   12.29 ms  steps=4  per_step=  3.072 ms
  Ex3 QLoRA                                   total=   37.60 ms  steps=4  per_step=  9.399 ms
  Ex4 DPO                                     total=   29.21 ms  steps=4  per_step=  7.302 ms
  Ex5 ContinualPretrain                       total=    8.50 ms  steps=4  per_step=  2.125 ms
  Ex6 MultimodalSFT                           total=   11.44 ms  steps=4  per_step=  2.861 ms
  Ex7 AcceleratorLoRA                         total=    8.04 ms  steps=4  per_step=  2.009 ms
  Ex8 GradAccum×8                             total=   30.60 ms  steps=16  per_step=  1.912 ms
  Ex9 train+export+vLLM                       total=16169.12 ms  steps=2  per_step=8084.559 ms
  Ex9 vLLM-infer-wall                         total=16152.56 ms  steps=1  per_step=16152.562 ms
  Ex10 MultiTurnSFT                           total=  104.31 ms  steps=4  per_step= 26.077 ms
  Ex10 stream-gen                             total=  106.51 ms  steps=12  per_step=  8.876 ms
  D-Matrix combos=8                           total=  111.35 ms  steps=16  per_step=  6.959 ms
  Real Qwen2.5 load                           total= 4523.33 ms  steps=1  per_step=4523.335 ms

API mapping (Python → Java) exercised:
  datasets.Dataset.from_list/map/train_test_split → HfDataset
  transformers.AutoModelForCausalLM/AutoTokenizer/GenerationConfig
  peft.LoraConfig/get_peft_model/print_trainable_parameters/
       save_pretrained/from_pretrained/merge_and_unload → PeftModel
  bitsandbytes.BitsAndBytesConfig + QLoRA.Session
  trl.SFTTrainer/DPOTrainer + SFTConfig/DPOConfig
  accelerate.Accelerator.prepare/wait_for_everyone/unwrap_model
  safetensors + GGUFWriter + vllm.LLM
######################################################################

```
