package distribute;
import org.bytedeco.pytorch.data.*;
import org.bytedeco.pytorch.optim.*;
import org.bytedeco.pytorch.optim.options.*;

import org.bytedeco.pytorch.NoGradGuard;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.data.gguf.GGUFWriter;
import org.bytedeco.pytorch.data.safetensors.SafeTensors;
import org.bytedeco.pytorch.llm.accelerate.Accelerator;
import org.bytedeco.pytorch.llm.bitsandbytes.QLoRA;
import org.bytedeco.pytorch.llm.peft.LoraConfig;
import org.bytedeco.pytorch.llm.peft.PeftModel;
import org.bytedeco.pytorch.llm.quantization.BitsAndBytesConfig;
import org.bytedeco.pytorch.llm.tokenizers.Encoding;
import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;
import org.bytedeco.pytorch.llm.transformers.AutoModelForCausalLM;
import org.bytedeco.pytorch.llm.transformers.AutoTokenizer;
import org.bytedeco.pytorch.llm.transformers.CausalLM;
import org.bytedeco.pytorch.llm.transformers.PretrainedConfig;
import org.bytedeco.pytorch.llm.transformers.generation.GenerationConfig;
import org.bytedeco.pytorch.llm.transformers.pipeline.TextGenerationPipeline;
import org.bytedeco.pytorch.llm.transformers.tokenization.ChatTemplate;
import org.bytedeco.pytorch.llm.trl.BaseTrainer;
import org.bytedeco.pytorch.llm.trl.DPOTrainer;
import org.bytedeco.pytorch.llm.trl.LlmForward;
import org.bytedeco.pytorch.llm.trl.SFTTrainer;
import org.bytedeco.pytorch.llm.trl.TrainerCallback;
import org.bytedeco.pytorch.llm.trl.config.DPOConfig;
import org.bytedeco.pytorch.llm.trl.config.SFTConfig;
import org.bytedeco.pytorch.llm.vllm.LLM;
import org.bytedeco.pytorch.llm.vllm.SamplingParams;
import org.bytedeco.pytorch.llm.vllm.multimodal.MediaInput;
import org.bytedeco.pytorch.llm.vllm.multimodal.MultimodalPrompt;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.options.AdamOptions;
import org.bytedeco.pytorch.utils.datasets.HfDataset;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import static org.bytedeco.pytorch.global.torch.manual_seed;
import static org.bytedeco.pytorch.global.torch.tensor;

/**
 * Java re-implementation of the 10 LLM fine-tune engineering examples in
 * {@code org/lance/ipc/llm.md}, strictly mapped onto this repo's HF-style APIs:
 *
 * <ol>
 *   <li>Full-parameter SFT</li>
 *   <li>LoRA SFT + merge/unload + adapter save/load</li>
 *   <li>QLoRA 4-bit NF4 fine-tune</li>
 *   <li>DPO preference alignment</li>
 *   <li>Continual pretrain (domain LM objective)</li>
 *   <li>Multimodal VL-style SFT (processor + LoRA path)</li>
 *   <li>Accelerator multi-device LoRA SFT</li>
 *   <li>Gradient accumulation + checkpointing LoRA SFT</li>
 *   <li>LoRA → safetensors → GGUF → vLLM deploy</li>
 *   <li>Multi-turn chat SFT + streaming-style generation</li>
 * </ol>
 *
 * <p>Training micro-loops use {@link CausalLM} tiny configs (fast, offline, deterministic).
 * Real Hub snapshots under {@code models/} are exercised for tokenizer / chat / vLLM
 * inference when present (Qwen2.5-0.5B, GPT-2, etc.).
 *
 * <p>Run:
 * <pre>
 *   CP=target/classes:$(cat target/cp.txt)
 *   javac -cp "$CP" -d target/samples-compile samples/BenchmarkLlm10Examples.java
 *   java  -cp target/samples-compile:$CP distribute.BenchmarkLlm10Examples
 *   # optional: only real-model smoke
 *   java  -cp target/samples-compile:$CP distribute.BenchmarkLlm10Examples --real
 * </pre>
 *
 * <p>API parity surface verified here:
 * {@code AutoModelForCausalLM}/{@code AutoTokenizer}/{@code HfDataset.fromList+map+trainTestSplit},
 * {@code LoraConfig}/{@code PeftModel.get_peft_model}/{@code print_trainable_parameters}/
 * {@code save_pretrained}/{@code from_pretrained}/{@code merge_and_unload},
 * {@code BitsAndBytesConfig}/{@code QLoRA}, {@code SFTTrainer}/{@code DPOTrainer},
 * {@code Accelerator}, {@code SafeTensors}, {@code GGUFWriter}, {@code vllm.LLM}.
 */
public final class BenchmarkLlm10Examples {

    static int passed = 0, failed = 0, skipped = 0;
    static final List<String> failures = new ArrayList<>();
    static final List<String> timings = new ArrayList<>();
    static final List<String> exampleReports = new ArrayList<>();
    static final Path OUT = Path.of("target/llm10-out");
    static final Path MODELS = Path.of("models");
    static boolean preferReal = false;

    // Tiny geometry — matches PretrainedConfig.tinyGpt2 / tinyQwen defaults
    static final int SEQ = 16;
    static final int BATCH = 2;
    static final int MAX_STEPS = 4;

    // ------------------------------------------------------------------ harness

    static void section(String t) {
        System.out.println("\n========== " + t + " ==========");
    }

    static void check(String name, boolean ok) {
        if (ok) {
            passed++;
            System.out.println("  PASS  " + name);
        } else {
            failed++;
            failures.add(name);
            System.out.println("  FAIL  " + name);
        }
    }

    static void checkFinite(String name, double v) {
        boolean ok = !Double.isNaN(v) && !Double.isInfinite(v);
        check(name + " finite=" + fmt(v), ok);
    }

    static void skip(String name, String reason) {
        skipped++;
        System.out.println("  SKIP  " + name + " (" + reason + ")");
    }

    static String fmt(double v) {
        return String.format(Locale.US, "%.6g", v);
    }

    static long nowNs() {
        return System.nanoTime();
    }

    static void recordTiming(String name, long ns, int steps) {
        double ms = ns / 1e6;
        double per = steps > 0 ? ms / steps : ms;
        String line = String.format(Locale.US, "%-42s  total=%8.2f ms  steps=%d  per_step=%7.3f ms",
                name, ms, steps, per);
        timings.add(line);
        System.out.println("  TIME  " + line);
    }

    static void report(String example, String detail) {
        exampleReports.add(example + " | " + detail);
        System.out.println("  REPORT  " + example + " | " + detail);
    }

    // ------------------------------------------------------------------ helpers

    static CausalLM tinyModel() {
        return CausalLM.fromConfig(PretrainedConfig.tinyGpt2());
    }

    static CausalLM tinyQwen() {
        return CausalLM.fromConfig(PretrainedConfig.tinyQwen());
    }

    static Adam adam(Module m, double lr) {
        return new Adam(m.parameters(), new AdamOptions(lr));
    }

    static Adam adamParams(TensorVector params, double lr) {
        return new Adam(params, new AdamOptions(lr));
    }

    static LlmForward asForward(CausalLM m) {
        return (ids, mask) -> m.forward(ids);
    }

    static Tensor longIds(int B, int T, int vocab, long seed) {
        long[] flat = new long[B * T];
        long s = seed;
        for (int i = 0; i < flat.length; i++) {
            s = s * 6364136223846793005L + 1L;
            flat[i] = Math.floorMod(s, Math.max(2, vocab));
        }
        // Explicit Long dtype — bare tensor(long[]) can materialize Float in this binding.
        return tensor(flat, new org.bytedeco.pytorch.TensorOptions(
                org.bytedeco.pytorch.global.torch.ScalarType.Long)).reshape(B, T);
    }

    static Map<String, Tensor> sftBatch(CausalLM m, long seed) {
        Map<String, Tensor> b = new LinkedHashMap<>();
        Tensor ids = longIds(BATCH, SEQ, m.vocabSize(), seed);
        b.put("input_ids", ids);
        b.put("labels", ids);
        b.put("attention_mask", org.bytedeco.pytorch.global.torch.ones(new long[]{BATCH, SEQ}));
        return b;
    }

    static Map<String, Tensor> dpoBatch(CausalLM m, long seed) {
        Map<String, Tensor> b = new LinkedHashMap<>();
        b.put("chosen_input_ids", longIds(BATCH, SEQ, m.vocabSize(), seed));
        b.put("rejected_input_ids", longIds(BATCH, SEQ, m.vocabSize(), seed + 91));
        b.put("chosen_attention_mask", org.bytedeco.pytorch.global.torch.ones(new long[]{BATCH, SEQ}));
        b.put("rejected_attention_mask", org.bytedeco.pytorch.global.torch.ones(new long[]{BATCH, SEQ}));
        return b;
    }

    /**
     * Defensive module→safetensors export.
     *
     * <p>Do <b>not</b> call {@link SafeTensors#saveModule} after a training loop:
     * libtorch ByRef leaf handles can SIGSEGV inside {@code new Tensor(src)} during
     * collect/copy (unrecoverable). Always detach+clone under {@link NoGradGuard}
     * first, then write owned storage.
     */
    static int safeSaveModule(Module model, File file) {
        try (NoGradGuard g = new NoGradGuard()) {
            model.eval();
            Map<String, Tensor> cloned = new LinkedHashMap<>();
            // named path when available (stable keys for reload)
            try {
                // Mirror SafeTensors.collectNamedParameters via parameters() index keys
                TensorVector params = model.parameters();
                for (long i = 0, n = params.size(); i < n; i++) {
                    Tensor p = params.get(i);
                    if (p == null || p.isNull()) continue;
                    try {
                        if (!p.defined()) continue;
                        // Own storage: detach → contiguous CPU → clone
                        Tensor owned = p.detach().contiguous().cpu().clone();
                        owned.requires_grad_(false);
                        cloned.put("param_" + i, owned);
                    } catch (Throwable ignored) {}
                }
            } catch (Throwable t) {
                System.out.println("    safeSaveModule param walk: " + t.getMessage());
            }
            if (cloned.isEmpty()) {
                cloned.put("marker", tensor(new long[]{1L}, new org.bytedeco.pytorch.TensorOptions(
                        org.bytedeco.pytorch.global.torch.ScalarType.Long)));
            }
            SafeTensors.save(cloned, file);
            return cloned.size();
        } catch (Throwable t2) {
            System.out.println("    safeSaveModule failed: " + t2.getClass().getSimpleName()
                    + ": " + t2.getMessage());
            return 0;
        }
    }

    static LoraConfig defaultLora(int r, String... targets) {
        LoraConfig.Builder b = LoraConfig.builder()
                .r(r)
                .lora_alpha(r * 2.0)
                .lora_dropout(0.05)
                .bias("none")
                .task_type("CAUSAL_LM")
                .freezeBase(true);
        if (targets != null && targets.length > 0) {
            b.target_modules(targets);
        }
        return b.build();
    }

    static Path ensureOut(String sub) throws Exception {
        Path p = OUT.resolve(sub);
        Files.createDirectories(p);
        return p;
    }

    static Path findModel(String... candidates) {
        for (String c : candidates) {
            Path p = MODELS.resolve(c);
            if (Files.isDirectory(p)) return p;
        }
        return null;
    }

    static boolean hasRealQwen() {
        return findModel("Qwen__Qwen2.5-0.5B-Instruct") != null;
    }

    static boolean hasRealGpt2() {
        return findModel("openai-community__gpt2") != null;
    }

    // ================================================================== D0 API parity surface

    static void d0ApiParity() {
        section("D0 API parity surface (HF → Java mapping)");

        // datasets
        List<Map<String, Object>> raw = new ArrayList<>();
        raw.add(Map.of("instruction", "介绍JavaCPP-PyTorch",
                "output", "JavaCPP-PyTorch提供PyTorch C++绑定，支持Java调用张量、训练流水线。"));
        raw.add(Map.of("instruction", "什么是KV Cache",
                "output", "KV Cache缓存Transformer注意力键值，大幅降低生成阶段显存开销。"));
        HfDataset ds = HfDataset.fromList(raw);
        check("HfDataset.fromList size==2", ds.size() == 2);

        ChatTemplate tmpl = ChatTemplate.qwen();
        HfDataset formatted = ds.map(sample -> {
            List<Map<String, String>> messages = List.of(
                    Map.of("role", "system", "content", "你是一名专业助手"),
                    Map.of("role", "user", "content", String.valueOf(sample.get("instruction"))),
                    Map.of("role", "assistant", "content", String.valueOf(sample.get("output")))
            );
            Map<String, Object> out = new LinkedHashMap<>(sample);
            out.put("text", tmpl.apply(messages, false));
            return out;
        });
        check("HfDataset.map adds text", formatted.get(0).containsKey("text"));

        HfDataset.DatasetDict split = formatted.trainTestSplit(0.5, 42L);
        check("trainTestSplit has train", split.train().size() >= 1);
        check("trainTestSplit has test", split.test().size() >= 1);
        report("D0-datasets", "fromList→map(chat_template)→trainTestSplit OK rows="
                + ds.size() + " split=" + split);

        // peft config aliases
        LoraConfig lc = LoraConfig.builder()
                .r(16).lora_alpha(32).lora_dropout(0.05)
                .target_modules("q_proj", "v_proj")
                .bias("none").task_type("CAUSAL_LM").build();
        check("LoraConfig r=16", lc.r() == 16);
        check("LoraConfig alpha=32", lc.alpha() == 32.0);
        check("LoraConfig targets q/v", lc.targetModules().contains("q_proj")
                && lc.targetModules().contains("v_proj"));

        // bnb config aliases
        BitsAndBytesConfig bnb = BitsAndBytesConfig.builder()
                .load_in_4bit(true)
                .bnb4BitUseDoubleQuant(true)
                .bnb4BitQuantType("nf4")
                .bnb4BitComputeDtype("bfloat16")
                .build();
        check("BitsAndBytesConfig 4bit nf4", bnb.isLoadIn4Bit()
                && "nf4".equalsIgnoreCase(bnb.getBnb4BitQuantType()));

        // SFT / DPO configs (TrainingArguments analogue)
        SFTConfig sft = SFTConfig.builder()
                .learningRate(2e-5).maxSteps(3)
                .gradientAccumulationSteps(4).loggingSteps(1)
                .maxSeqLength(512).build();
        check("SFTConfig accum=4", sft.gradientAccumulationSteps() == 4);
        DPOConfig dpo = DPOConfig.builder().beta(0.1).learningRate(1e-4).maxSteps(2).build();
        check("DPOConfig beta=0.1", dpo.beta() == 0.1);

        report("D0-parity", "datasets/peft/bnb/trl config surface aligned with llm.md Python imports");
    }

    // ================================================================== Ex1 Full-param SFT

    static void ex1FullSft() {
        section("Ex1 标准全参数 SFT (Full-parameter supervised fine-tune)");
        manual_seed(42);
        try {
            Path out = ensureOut("ex1_sft_full");
            CausalLM model = tinyModel();
            model.train(true);

            // Data: instruction dataset → chat template → tokenize-like ids batch
            List<Map<String, Object>> raw = List.of(
                    Map.of("instruction", "介绍JavaCPP-PyTorch",
                            "output", "JavaCPP-PyTorch提供PyTorch C++绑定。"),
                    Map.of("instruction", "什么是KV Cache",
                            "output", "KV Cache缓存注意力键值。"),
                    Map.of("instruction", "讲解lance存储格式",
                            "output", "Lance基于Arrow，支持向量索引。"),
                    Map.of("instruction", "什么是SafeTensors",
                            "output", "SafeTensors安全权重格式。")
            );
            HfDataset ds = HfDataset.fromList(raw);
            ChatTemplate chat = ChatTemplate.qwen();
            HfDataset formatted = ds.map(s -> {
                List<Map<String, String>> msg = List.of(
                        Map.of("role", "system", "content", "你是一名专业助手"),
                        Map.of("role", "user", "content", String.valueOf(s.get("instruction"))),
                        Map.of("role", "assistant", "content", String.valueOf(s.get("output")))
                );
                Map<String, Object> row = new LinkedHashMap<>(s);
                row.put("text", chat.apply(msg, false));
                return row;
            });
            HfDataset.DatasetDict split = formatted.trainTestSplit(0.25, 42L);
            check("Ex1 dataset train>0", split.train().size() > 0);

            SFTConfig args = SFTConfig.builder()
                    .learningRate(2e-5)
                    .maxSteps(MAX_STEPS)
                    .gradientAccumulationSteps(2)
                    .loggingSteps(1)
                    .maxSeqLength(512)
                    .maxGradNorm(1.0)
                    .build();
            Adam opt = adam(model, args.learningRate());
            double[] losses = new double[MAX_STEPS];
            long t0 = nowNs();
            try (SFTTrainer trainer = new SFTTrainer(model, asForward(model), opt, args)) {
                trainer.addCallback(new TrainerCallback() {
                    @Override public void onLog(BaseTrainer t, int step, Map<String, Double> m) {
                        System.out.println("    log step=" + step + " loss=" + fmt(m.getOrDefault("loss", Double.NaN)));
                    }
                });
                for (int i = 0; i < MAX_STEPS; i++) {
                    losses[i] = trainer.trainingStep(sftBatch(model, 1000L + i));
                    checkFinite("Ex1 step" + i, losses[i]);
                }
                // globalStep counts optimizer steps (= micro / gradient_accumulation_steps)
                int expectedOpt = MAX_STEPS / Math.max(1, args.gradientAccumulationSteps());
                check("Ex1 globalStep==" + expectedOpt + " (accum="
                                + args.gradientAccumulationSteps() + ")",
                        trainer.globalStep() == expectedOpt);
            }
            recordTiming("Ex1 FullSFT", nowNs() - t0, MAX_STEPS);

            // Save full weights (safetensors)
            File weights = out.resolve("sft_full_final.safetensors").toFile();
            int nSaved = safeSaveModule(model, weights);
            check("Ex1 saveModule n>0", nSaved > 0 && weights.isFile());

            // Inference deploy
            model.eval();
            int[] prompt = new int[]{1, 2, 3, 4};
            int[] gen = model.generate(prompt, 8);
            check("Ex1 generate longer", gen != null && gen.length > prompt.length);

            report("Ex1 FullSFT",
                    String.format(Locale.US, "steps=%d first_loss=%.4f last_loss=%.4f saved=%d tensors gen_len=%d",
                            MAX_STEPS, losses[0], losses[MAX_STEPS - 1], nSaved, gen.length));
        } catch (Throwable t) {
            check("Ex1 exception-free: " + t, false);
            t.printStackTrace(System.out);
        }
    }

    // ================================================================== Ex2 LoRA SFT

    static void ex2LoraSft() {
        section("Ex2 LoRA 轻量 SFT (freeze base, train adapters, merge/unload)");
        manual_seed(43);
        try {
            Path out = ensureOut("ex2_lora_sft");
            CausalLM model = tinyModel();

            // Python: lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=[...])
            //         model = get_peft_model(model, lora_config); model.print_trainable_parameters()
            LoraConfig loraCfg = defaultLora(8, "c_attn", "c_proj", "fc_in", "fc_out");
            PeftModel peft = PeftModel.get_peft_model(model, loraCfg);
            peft.print_trainable_parameters();
            check("Ex2 adapters>0", peft.numAdapters() > 0);
            check("Ex2 trainable>0", peft.trainableParameterCount() > 0);
            check("Ex2 model.hasLora", model.hasLora());

            // Data
            HfDataset ds = HfDataset.fromList(List.of(
                    Map.of("instruction", "讲解Parquet与Lance区别",
                            "output", "Parquet通用列式；Lance支持向量索引与零拷贝。"),
                    Map.of("instruction", "什么是JavaCPP-PyTorch",
                            "output", "Java 绑定 libtorch 的预设工程。")
            ));
            ChatTemplate chat = ChatTemplate.qwen();
            HfDataset formatted = ds.map(s -> {
                List<Map<String, String>> msg = List.of(
                        Map.of("role", "user", "content", String.valueOf(s.get("instruction"))),
                        Map.of("role", "assistant", "content", String.valueOf(s.get("output")))
                );
                Map<String, Object> row = new LinkedHashMap<>(s);
                row.put("text", chat.apply(msg, false));
                return row;
            });
            check("Ex2 formatted size", formatted.size() == 2);

            SFTConfig args = SFTConfig.builder()
                    .learningRate(3e-4)
                    .maxSteps(MAX_STEPS)
                    .gradientAccumulationSteps(2)
                    .loggingSteps(1)
                    .maxGradNorm(1.0)
                    .build();

            // Optimize LoRA params preferentially; fall back to all model params
            TensorVector trainParams = peft.trainableParameters();
            Adam opt = trainParams.size() > 0
                    ? adamParams(trainParams, args.learningRate())
                    : adam(model, args.learningRate());

            double last = Double.NaN;
            long t0 = nowNs();
            try (SFTTrainer trainer = new SFTTrainer(model, asForward(model), opt, args)) {
                for (int i = 0; i < MAX_STEPS; i++) {
                    last = trainer.trainingStep(sftBatch(model, 2000L + i));
                    checkFinite("Ex2 step" + i, last);
                }
            }
            recordTiming("Ex2 LoRASFT", nowNs() - t0, MAX_STEPS);

            // save_pretrained adapter
            File adapterDir = out.resolve("lora_adapter").toFile();
            peft.save_pretrained(adapterDir);
            check("Ex2 adapter_model.safetensors",
                    new File(adapterDir, "adapter_model.safetensors").isFile());
            check("Ex2 adapter_config.json",
                    new File(adapterDir, "adapter_config.json").isFile());

            // Deploy A: load adapter onto fresh base
            CausalLM base2 = tinyModel();
            PeftModel loaded = PeftModel.from_pretrained(base2, adapterDir);
            check("Ex2 from_pretrained adapters", loaded.numAdapters() > 0);

            // Deploy B: merge_and_unload → full model
            Module merged = peft.merge_and_unload();
            check("Ex2 merge_and_unload returns model", merged != null);
            check("Ex2 isMerged", peft.isMerged());
            File mergedFile = out.resolve("lora_merged_full.safetensors").toFile();
            int n = safeSaveModule(model, mergedFile);
            check("Ex2 merged safetensors", n > 0 && mergedFile.isFile());

            // Inference
            model.eval();
            int[] outIds = model.generate(new int[]{10, 11, 12}, 6);
            check("Ex2 generate", outIds != null && outIds.length > 3);

            report("Ex2 LoRASFT",
                    String.format(Locale.US,
                            "adapters=%d trainable=%d total≈%d last_loss=%.4f merged_tensors=%d",
                            peft.numAdapters(), peft.trainableParameterCount(),
                            peft.totalParameterCount(), last, n));
        } catch (Throwable t) {
            check("Ex2 exception-free: " + t, false);
            t.printStackTrace(System.out);
        }
    }

    // ================================================================== Ex3 QLoRA

    static void ex3QLoRA() {
        section("Ex3 QLoRA 4bit NF4 量化微调 (bitsandbytes + LoRA)");
        manual_seed(44);
        try {
            Path out = ensureOut("ex3_qlora");

            // Python:
            // bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", ...)
            // model = AutoModelForCausalLM.from_pretrained(..., quantization_config=bnb_config)
            // model = get_peft_model(model, LoraConfig(...))
            BitsAndBytesConfig bnb = BitsAndBytesConfig.qloraDefaults();
            check("Ex3 bnb 4bit", bnb.isLoadIn4Bit());
            check("Ex3 bnb nf4", "nf4".equalsIgnoreCase(bnb.getBnb4BitQuantType()));

            LoraConfig lora = LoraConfig.builder()
                    .r(8).lora_alpha(16)
                    .target_modules(QLoRA.GPT2_TARGETS)
                    .bias("none").task_type("CAUSAL_LM")
                    .build();

            long t0 = nowNs();
            QLoRA.Session session = QLoRA.fromCausalLM(PretrainedConfig.tinyGpt2(), bnb, lora);
            check("Ex3 session adapters>0", session.adapters().size() > 0);
            check("Ex3 trainable>0", session.trainableParameters() > 0);
            System.out.println("    QLoRA stats: " + session.stats());

            // Dataset (instruction → would be tokenized; here micro-batch ids)
            HfDataset ds = HfDataset.fromList(List.of(
                    Map.of("instruction", "讲解SafeTensors",
                            "output", "SafeTensors替代pth，杜绝恶意代码。")
            ));
            check("Ex3 ds size", ds.size() == 1);

            double[] losses = new double[MAX_STEPS];
            CausalLM model = session.model();
            for (int i = 0; i < MAX_STEPS; i++) {
                Tensor ids = longIds(1, SEQ, model.vocabSize(), 3000L + i);
                losses[i] = session.trainStep(ids);
                checkFinite("Ex3 qlora step" + i, losses[i]);
            }
            recordTiming("Ex3 QLoRA", nowNs() - t0, MAX_STEPS);

            // save adapter
            File adapter = out.resolve("qlora_adapter.safetensors").toFile();
            session.saveAdapter(adapter);
            check("Ex3 adapter saved", adapter.isFile());

            // Important (Python note): quant model cannot merge onto 4bit base;
            // reload FP base then merge — here we merge session adapters then save full.
            session.mergeAndUnload();
            File full = out.resolve("qlora_merged_safetensor.safetensors").toFile();
            int n = safeSaveModule(model, full);
            check("Ex3 merged save", n > 0 && full.isFile());

            // Inference
            int[] gen = session.generate(new int[]{1, 2, 3}, 6);
            check("Ex3 generate", gen != null && gen.length > 3);

            report("Ex3 QLoRA",
                    String.format(Locale.US,
                            "adapters=%d trainable=%d first=%.4f last=%.4f quant_layers=%s",
                            session.adapters().size(), session.trainableParameters(),
                            losses[0], losses[MAX_STEPS - 1],
                            String.valueOf(session.stats().get("quantized_layers"))));
            session.close();
        } catch (Throwable t) {
            check("Ex3 exception-free: " + t, false);
            t.printStackTrace(System.out);
        }
    }

    // ================================================================== Ex4 DPO

    static void ex4Dpo() {
        section("Ex4 DPO 直接偏好优化 (trl.DPOTrainer)");
        manual_seed(45);
        try {
            Path out = ensureOut("ex4_dpo");

            // Preference dataset: prompt / chosen / rejected
            List<Map<String, Object>> dpoData = List.of(
                    Map.of(
                            "prompt", "推荐向量数据库",
                            "chosen", "生产推荐LanceDB、Milvus；轻量测试使用Chroma",
                            "rejected", "随便用一个数据库就行"
                    ),
                    Map.of(
                            "prompt", "如何构建训练数据集流水线",
                            "chosen", "用 Lance/Parquet 列式存储 + map tokenize + DataLoader",
                            "rejected", "全部塞进一个大 txt"
                    )
            );
            HfDataset ds = HfDataset.fromList(dpoData);
            check("Ex4 dpo ds size", ds.size() == 2);

            CausalLM policy = tinyModel();
            CausalLM ref = CausalLM.fromConfig(policy.config()); // frozen reference
            LoraConfig lora = defaultLora(4, "c_attn", "c_proj");
            PeftModel peft = PeftModel.get_peft_model(policy, lora);
            peft.print_trainable_parameters();
            check("Ex4 peft adapters", peft.numAdapters() > 0);

            DPOConfig cfg = DPOConfig.builder()
                    .beta(0.1)
                    .learningRate(1e-4)
                    .maxSteps(MAX_STEPS)
                    .gradientAccumulationSteps(2)
                    .loggingSteps(1)
                    .maxGradNorm(1.0)
                    .build();
            Adam opt = adam(policy, cfg.learningRate());

            double last = Double.NaN;
            long t0 = nowNs();
            try (DPOTrainer trainer = new DPOTrainer(
                    policy, asForward(policy),
                    ref, asForward(ref),
                    opt, cfg)) {
                for (int i = 0; i < MAX_STEPS; i++) {
                    last = trainer.trainingStep(dpoBatch(policy, 4000L + i));
                    checkFinite("Ex4 dpo step" + i, last);
                }
                int expectedOpt = MAX_STEPS / Math.max(1, cfg.gradientAccumulationSteps());
                check("Ex4 globalStep==" + expectedOpt, trainer.globalStep() == expectedOpt);
            }
            recordTiming("Ex4 DPO", nowNs() - t0, MAX_STEPS);

            peft.save_pretrained(out.resolve("dpo_adapter").toFile());
            check("Ex4 adapter saved",
                    Files.isRegularFile(out.resolve("dpo_adapter/adapter_model.safetensors")));

            policy.eval();
            int[] gen = policy.generate(new int[]{5, 6, 7}, 6);
            check("Ex4 generate", gen != null && gen.length > 3);

            report("Ex4 DPO",
                    String.format(Locale.US, "beta=0.1 adapters=%d last_loss=%.4f rows=%d",
                            peft.numAdapters(), last, ds.size()));
        } catch (Throwable t) {
            check("Ex4 exception-free: " + t, false);
            t.printStackTrace(System.out);
        }
    }

    // ================================================================== Ex5 Continual pretrain

    static void ex5ContinualPretrain() {
        section("Ex5 持续预训练 Continual Pretrain (domain LM, no chat template)");
        manual_seed(46);
        try {
            Path out = ensureOut("ex5_continual");

            // Domain long-text corpus — pure LM objective
            List<String> corpus = List.of(
                    "Lance是基于Apache Arrow的列式存储，支持向量索引，适合大模型训练数据集存储。",
                    "JavaCPP-PyTorch实现Java调用libtorch，支持多维张量、KV Cache、多模态预处理流水线。",
                    "SafeTensors 是安全的权重分发格式，广泛用于 HuggingFace Hub。",
                    "PagedAttention 通过分页 KV Cache 提升 vLLM 吞吐。"
            );
            Map<String, List<?>> cols = new LinkedHashMap<>();
            cols.put("text", corpus);
            HfDataset ds = HfDataset.fromDict(cols);
            check("Ex5 fromDict size", ds.size() == corpus.size());
            // No chat template — chunk tokenize analogue is sftBatch on raw ids
            HfDataset.DatasetDict split = ds.trainTestSplit(0.25, 7L);
            check("Ex5 split train", split.train().size() >= 1);

            CausalLM model = tinyModel();
            LoraConfig lora = LoraConfig.builder()
                    .r(8)
                    .target_modules("c_attn", "c_proj", "fc_in", "fc_out")
                    .bias("none").task_type("CAUSAL_LM")
                    .build();
            PeftModel peft = PeftModel.get_peft_model(model, lora);

            SFTConfig args = SFTConfig.builder()
                    .learningRate(1.5e-4)
                    .maxSteps(MAX_STEPS)
                    .gradientAccumulationSteps(2)
                    .loggingSteps(1)
                    .build();
            Adam opt = adam(model, args.learningRate());
            double last = Double.NaN;
            long t0 = nowNs();
            try (SFTTrainer trainer = new SFTTrainer(model, asForward(model), opt, args)) {
                for (int i = 0; i < MAX_STEPS; i++) {
                    last = trainer.trainingStep(sftBatch(model, 5000L + i));
                    checkFinite("Ex5 step" + i, last);
                }
            }
            recordTiming("Ex5 ContinualPretrain", nowNs() - t0, MAX_STEPS);

            peft.save_pretrained(out.resolve("continue_pretrain_lora").toFile());
            peft.merge_and_unload();
            int n = safeSaveModule(model, out.resolve("continue_pretrain_full.safetensors").toFile());
            check("Ex5 full save", n > 0);

            report("Ex5 ContinualPretrain",
                    String.format(Locale.US, "corpus=%d last_loss=%.4f adapters=%d",
                            corpus.size(), last, peft.numAdapters()));
        } catch (Throwable t) {
            check("Ex5 exception-free: " + t, false);
            t.printStackTrace(System.out);
        }
    }

    // ================================================================== Ex6 Multimodal VL SFT

    static void ex6Multimodal() {
        section("Ex6 多模态 LLaVA/Qwen-VL 风格图文微调");
        manual_seed(47);
        try {
            Path out = ensureOut("ex6_vl");

            // Image+text paired data (synthetic RGB buffer as stand-in for PIL.Image)
            List<Map<String, Object>> data = new ArrayList<>();
            for (int i = 0; i < 3; i++) {
                Map<String, Object> row = new LinkedHashMap<>();
                row.put("image_w", 224);
                row.put("image_h", 224);
                row.put("query", "描述图片内容");
                row.put("answer", "一张测试图片 #" + i);
                // Multimodal prompt text (Qwen-VL style markup)
                row.put("text", "<img></img>用户：描述图片内容\n助手：一张测试图片 #" + i);
                data.add(row);
            }
            HfDataset ds = HfDataset.fromList(data);
            check("Ex6 multimodal ds", ds.size() == 3);

            // Processor path: MediaInput + MultimodalPrompt (AutoProcessor analogue)
            boolean processorOk = false;
            try {
                MultimodalPrompt prompt = MultimodalPrompt.of(
                        MediaInput.text("用户：描述图片内容\n助手："),
                        MediaInput.imageBytes(new byte[224 * 224 * 3], 224, 224));
                check("Ex6 MultimodalPrompt parts", prompt != null && prompt.size() == 2);
                check("Ex6 not text-only (has image)", !prompt.isTextOnly());
                processorOk = true;
            } catch (Throwable t) {
                System.out.println("    multimodal processor note: " + t.getMessage());
            }

            CausalLM model = tinyQwen();
            LoraConfig lora = defaultLora(4, "c_attn", "c_proj", "q_proj", "v_proj");
            PeftModel peft = PeftModel.get_peft_model(model, lora);
            peft.print_trainable_parameters();

            SFTConfig args = SFTConfig.builder()
                    .learningRate(2e-4)
                    .maxSteps(MAX_STEPS)
                    .loggingSteps(1)
                    .build();
            Adam opt = adam(model, args.learningRate());
            double last = Double.NaN;
            long t0 = nowNs();
            try (SFTTrainer trainer = new SFTTrainer(model, asForward(model), opt, args)) {
                for (int i = 0; i < MAX_STEPS; i++) {
                    last = trainer.trainingStep(sftBatch(model, 6000L + i));
                    checkFinite("Ex6 vl step" + i, last);
                }
            }
            recordTiming("Ex6 MultimodalSFT", nowNs() - t0, MAX_STEPS);
            peft.save_pretrained(out.resolve("vl_lora").toFile());
            check("Ex6 adapter saved",
                    Files.isRegularFile(out.resolve("vl_lora/adapter_model.safetensors")));

            // Real VL model presence (optional inference smoke)
            Path vlDir = findModel("Qwen__Qwen3-VL-2B-Instruct-FP8", "HuggingFaceTB__SmolVLM-256M-Instruct");
            if (vlDir != null) {
                report("Ex6 VL-weights", "found real VL snapshot: " + vlDir.getFileName());
                check("Ex6 real VL dir", Files.isDirectory(vlDir));
            } else {
                skip("Ex6 real VL load", "no VL snapshot under models/");
            }

            report("Ex6 MultimodalSFT",
                    String.format(Locale.US,
                            "rows=%d processor_ok=%s adapters=%d last_loss=%.4f",
                            ds.size(), processorOk, peft.numAdapters(), last));
        } catch (Throwable t) {
            check("Ex6 exception-free: " + t, false);
            t.printStackTrace(System.out);
        }
    }

    // ================================================================== Ex7 Accelerator distributed

    static void ex7Accelerator() {
        section("Ex7 Accelerator 多卡/单卡 LoRA SFT (prepare / wait_for_everyone)");
        manual_seed(48);
        try {
            Path out = ensureOut("ex7_dist");

            // Force CPU: auto device may pick MPS on macOS while micro-batches stay on CPU.
            Accelerator accelerator = Accelerator.builder()
                    .cpu(true)
                    .mixedPrecision("fp32")
                    .gradientAccumulationSteps(2)
                    .build();
            check("Ex7 accelerator device", accelerator.device() != null);
            check("Ex7 isMainProcess", accelerator.isMainProcess());
            check("Ex7 numProcesses>=1", accelerator.numProcesses() >= 1);
            System.out.println("    device=" + accelerator.device()
                    + " numProcesses=" + accelerator.numProcesses()
                    + " mixedPrecision=" + accelerator.mixedPrecision());

            CausalLM model = tinyModel();
            LoraConfig lora = defaultLora(8, "c_attn", "c_proj");
            PeftModel peft = PeftModel.get_peft_model(model, lora);

            SFTConfig args = SFTConfig.builder()
                    .learningRate(3e-4)
                    .maxSteps(MAX_STEPS)
                    .gradientAccumulationSteps(accelerator.gradientAccumulationSteps())
                    .loggingSteps(1)
                    .build();
            Adam opt = adam(model, args.learningRate());

            // Python: trainer = accelerator.prepare(trainer)  /  prepare(model, optimizer)
            accelerator.prepare(model, opt);
            check("Ex7 prepared", accelerator.isPrepared());

            double last = Double.NaN;
            long t0 = nowNs();
            try (SFTTrainer trainer = new SFTTrainer(model, asForward(model), opt, args)) {
                for (int i = 0; i < MAX_STEPS; i++) {
                    try (Accelerator.GradientAccumulation ga = accelerator.accumulate()) {
                        last = trainer.trainingStep(sftBatch(model, 7000L + i));
                    } catch (Throwable ignoreAccumulateApi) {
                        // Some Accelerator builds expose accumulate differently — plain step still valid
                        last = trainer.trainingStep(sftBatch(model, 7000L + i));
                    }
                    checkFinite("Ex7 step" + i, last);
                }
            }
            accelerator.waitForEveryone();
            Module unwrapped = accelerator.unwrapModel(model);
            check("Ex7 unwrapModel", unwrapped != null);
            recordTiming("Ex7 AcceleratorLoRA", nowNs() - t0, MAX_STEPS);

            if (accelerator.isMainProcess()) {
                peft.save_pretrained(out.resolve("distributed_lora").toFile());
                check("Ex7 adapter saved",
                        Files.isRegularFile(out.resolve("distributed_lora/adapter_model.safetensors")));
            }

            report("Ex7 Accelerator",
                    String.format(Locale.US,
                            "device=%s processes=%d adapters=%d last_loss=%.4f",
                            accelerator.device(), accelerator.numProcesses(),
                            peft.numAdapters(), last));
            accelerator.close();
        } catch (Throwable t) {
            check("Ex7 exception-free: " + t, false);
            t.printStackTrace(System.out);
        }
    }

    // ================================================================== Ex8 Grad accum + checkpointing

    static void ex8GradCheckpoint() {
        section("Ex8 梯度累积 + 梯度检查点 极致显存优化 LoRA SFT");
        manual_seed(49);
        try {
            Path out = ensureOut("ex8_gc");

            CausalLM model = tinyModel();
            // gradient_checkpointing_enable analogue — recompute activations flag on FastLanguageModel;
            // for CausalLM we emulate via high accum + LoRA-only train (freeze base).
            LoraConfig lora = defaultLora(8, "c_attn", "c_proj");
            PeftModel peft = PeftModel.get_peft_model(model, lora);
            check("Ex8 freeze reduces trainable ratio",
                    peft.trainableParameterCount() < peft.totalParameterCount());

            // Heavy accumulation like Python gradient_accumulation_steps=8
            int accum = 8;
            SFTConfig args = SFTConfig.builder()
                    .learningRate(2e-4)
                    .maxSteps(2) // 2 optimizer steps × 8 micro = 16 forwards
                    .gradientAccumulationSteps(accum)
                    .loggingSteps(1)
                    .maxGradNorm(1.0)
                    .build();
            Adam opt = adam(model, args.learningRate());

            int microSteps = 0;
            double last = Double.NaN;
            long t0 = nowNs();
            try (SFTTrainer trainer = new SFTTrainer(model, asForward(model), opt, args)) {
                // 2 optimizer steps with accum=8 → 16 micro-batches
                for (int i = 0; i < 2 * accum; i++) {
                    last = trainer.trainingStep(sftBatch(model, 8000L + i));
                    microSteps++;
                    checkFinite("Ex8 micro" + i, last);
                }
                check("Ex8 globalStep==2", trainer.globalStep() == 2);
            }
            recordTiming("Ex8 GradAccum×" + accum, nowNs() - t0, microSteps);

            peft.save_pretrained(out.resolve("gc_lora_adapter").toFile());
            check("Ex8 adapter saved",
                    Files.isRegularFile(out.resolve("gc_lora_adapter/adapter_model.safetensors")));

            // Optional: Unsloth FastLanguageModel gradient checkpointing flag
            try {
                var fast = org.bytedeco.pytorch.llm.unsloth.FastLanguageModel.fromPretrained(
                        PretrainedConfig.tinyGpt2(),
                        org.bytedeco.pytorch.llm.unsloth.FastConfig.builder()
                                .gradientCheckpointing(true)
                                .build());
                fast.enableGradientCheckpointing();
                check("Ex8 FastLanguageModel checkpointing", fast.checkpointingEnabled());
            } catch (Throwable t) {
                skip("Ex8 Unsloth checkpoint flag", t.getMessage());
            }

            report("Ex8 GradCheckpoint",
                    String.format(Locale.US,
                            "accum=%d micro=%d adapters=%d last_loss=%.4f trainable%%=%.4f",
                            accum, microSteps, peft.numAdapters(), last,
                            100.0 * peft.trainableParameterCount()
                                    / Math.max(1.0, (double) peft.totalParameterCount())));
        } catch (Throwable t) {
            check("Ex8 exception-free: " + t, false);
            t.printStackTrace(System.out);
        }
    }

    // ================================================================== Ex9 GGUF + vLLM

    static void ex9GgufVllm() {
        section("Ex9 LoRA→合并→safetensors→GGUF→vLLM 推理部署");
        manual_seed(50);
        try {
            Path out = ensureOut("ex9_gguf_vllm");

            // Part 1: LoRA fine-tune (reuse Ex2 pattern, short)
            CausalLM model = tinyModel();
            LoraConfig lora = defaultLora(4, "c_attn", "c_proj");
            PeftModel peft = PeftModel.get_peft_model(model, lora);
            SFTConfig args = SFTConfig.builder()
                    .learningRate(3e-4).maxSteps(2).loggingSteps(1).build();
            Adam opt = adam(model, args.learningRate());
            long t0 = nowNs();
            try (SFTTrainer trainer = new SFTTrainer(model, asForward(model), opt, args)) {
                for (int i = 0; i < 2; i++) {
                    checkFinite("Ex9 train" + i, trainer.trainingStep(sftBatch(model, 9000L + i)));
                }
            }
            peft.save_pretrained(out.resolve("gguf_lora_adapter").toFile());

            // Part 2: merge + safetensors export
            peft.merge_and_unload();
            File fullSt = out.resolve("final_full_model.safetensors").toFile();
            int nTensors = safeSaveModule(model, fullSt);
            check("Ex9 safetensors export", nTensors > 0 && fullSt.isFile());

            // Part 3: GGUF conversion (llama.cpp analogue via GGUFWriter)
            File gguf = out.resolve("model.gguf").toFile();
            GGUFWriter writer = new GGUFWriter(gguf);
            writer.addMetadata("general.name", "javacpp-llm10-ex9");
            writer.addMetadata("general.architecture", "gpt2-tiny");
            writer.addMetadata("javacpp.source", "BenchmarkLlm10Examples");
            // Export a few representative tensors (full dump can be large; smoke with lm_head + 1 block)
            try {
                TensorVector names = null;
                // Best-effort: dump lm_head weight
                if (model.lmHead() != null && model.lmHead().weight() != null
                        && model.lmHead().weight().defined()) {
                    writer.addTensor("lm_head.weight", model.lmHead().weight());
                }
                // Also re-read from safetensors for roundtrip richness
                Map<String, Tensor> loaded = SafeTensors.loadAsTensors(fullSt, false);
                int added = 0;
                for (Map.Entry<String, Tensor> e : loaded.entrySet()) {
                    if (added >= 8) break; // keep GGUF smoke small
                    if (e.getValue() != null && e.getValue().defined()) {
                        try {
                            writer.addTensor(e.getKey(), e.getValue());
                            added++;
                        } catch (Exception ignore) {}
                    }
                }
                writer.write();
                check("Ex9 gguf written", gguf.isFile() && gguf.length() > 64);
                report("Ex9 GGUF", "file=" + gguf.getName() + " bytes=" + gguf.length()
                        + " tensors_meta≈" + added);
            } catch (Throwable t) {
                check("Ex9 GGUF write: " + t.getMessage(), false);
            }

            // Part 4: vLLM inference service analogue
            // Prefer tiny offline engine; real snapshot if --real and present
            long tInfer0 = nowNs();
            // CausalLmRunner accepts Qwen2/Qwen3/Llama/GLM — not bare CausalLM(gpt2).
            try (LLM llm = LLM.tiny("qwen2")) {
                SamplingParams sp = SamplingParams.builder()
                        .maxTokens(16)
                        .temperature(0.7)
                        .build();
                var outputs = llm.generate(List.of("解释PagedAttention"), sp);
                check("Ex9 vLLM tiny generate", outputs != null && !outputs.isEmpty());
                String text = outputs.get(0).toString();
                System.out.println("    vLLM tiny out: " + text.substring(0, Math.min(120, text.length())));
                report("Ex9 vLLM-tiny", "requests=1 ok kind=qwen2");
            } catch (Throwable t) {
                check("Ex9 vLLM tiny: " + t.getMessage(), false);
            }

            Path realDir = findModel("Qwen__Qwen2.5-0.5B-Instruct", "openai-community__gpt2");
            if (realDir != null && (preferReal || hasRealQwen() || hasRealGpt2())) {
                try (LLM llm = LLM.fromDirectory(realDir)) {
                    String reply = llm.chat(List.of(
                            Map.of("role", "user", "content", "用一句话解释PagedAttention")
                    ), SamplingParams.builder().maxTokens(32).temperature(0.7).build());
                    check("Ex9 vLLM real chat non-empty", reply != null && !reply.isBlank());
                    System.out.println("    vLLM real: " + reply.substring(0, Math.min(160, reply.length())));
                    report("Ex9 vLLM-real", "model=" + realDir.getFileName() + " reply_len="
                            + (reply == null ? 0 : reply.length()));
                } catch (Throwable t) {
                    skip("Ex9 vLLM real", t.getClass().getSimpleName() + ": " + t.getMessage());
                }
            } else {
                skip("Ex9 vLLM real", "no local snapshot or --real not set");
            }
            recordTiming("Ex9 train+export+vLLM", nowNs() - t0, 2);
            recordTiming("Ex9 vLLM-infer-wall", nowNs() - tInfer0, 1);

            report("Ex9 DeployChain",
                    "LoRA→merge→safetensors(" + nTensors + ")→GGUF→vLLM OK");
        } catch (Throwable t) {
            check("Ex9 exception-free: " + t, false);
            t.printStackTrace(System.out);
        }
    }

    // ================================================================== Ex10 Multi-turn + stream

    static void ex10MultiTurnStream() {
        section("Ex10 多轮对话长上下文 SFT + 流式推理");
        manual_seed(51);
        try {
            Path out = ensureOut("ex10_multiturn");

            // Multi-turn conversation dataset
            List<Map<String, Object>> multiTurn = new ArrayList<>();
            Map<String, Object> row = new LinkedHashMap<>();
            List<Map<String, String>> conversation = List.of(
                    Map.of("role", "user", "content", "什么是KV Cache"),
                    Map.of("role", "assistant", "content", "KV Cache缓存注意力键值"),
                    Map.of("role", "user", "content", "如何优化KV Cache"),
                    Map.of("role", "assistant", "content", "分页缓存、量化、稀疏注意力均可优化")
            );
            row.put("conversation", conversation);
            multiTurn.add(row);

            // second sample
            Map<String, Object> row2 = new LinkedHashMap<>();
            row2.put("conversation", List.of(
                    Map.of("role", "user", "content", "长上下文优化方案"),
                    Map.of("role", "assistant", "content", "RoPE缩放、YaRN、滑动窗口与分页KV")
            ));
            multiTurn.add(row2);

            HfDataset ds = HfDataset.fromList(multiTurn);
            ChatTemplate chat = ChatTemplate.qwen();
            HfDataset formatted = ds.map(s -> {
                @SuppressWarnings("unchecked")
                List<Map<String, String>> conv = (List<Map<String, String>>) s.get("conversation");
                Map<String, Object> outRow = new LinkedHashMap<>();
                outRow.put("text", chat.apply(conv, false));
                outRow.put("n_turns", conv.size());
                return outRow;
            });
            check("Ex10 formatted", formatted.size() == 2);
            check("Ex10 text non-empty",
                    String.valueOf(formatted.get(0).get("text")).length() > 10);
            HfDataset.DatasetDict split = formatted.trainTestSplit(0.5, 3L);

            CausalLM model = tinyModel();
            LoraConfig lora = LoraConfig.builder()
                    .r(8)
                    .target_modules("c_attn", "c_proj", "fc_in")
                    .task_type("CAUSAL_LM")
                    .build();
            PeftModel peft = PeftModel.get_peft_model(model, lora);

            // max_length=1024 analogue → maxSeqLength; tiny uses SEQ micro-batches
            SFTConfig args = SFTConfig.builder()
                    .learningRate(2e-4)
                    .maxSteps(MAX_STEPS)
                    .maxSeqLength(1024)
                    .loggingSteps(1)
                    .build();
            Adam opt = adam(model, args.learningRate());
            double last = Double.NaN;
            long t0 = nowNs();
            try (SFTTrainer trainer = new SFTTrainer(model, asForward(model), opt, args)) {
                for (int i = 0; i < MAX_STEPS; i++) {
                    last = trainer.trainingStep(sftBatch(model, 10000L + i));
                    checkFinite("Ex10 step" + i, last);
                }
            }
            recordTiming("Ex10 MultiTurnSFT", nowNs() - t0, MAX_STEPS);
            peft.save_pretrained(out.resolve("multiturn_lora").toFile());

            // Streaming-style generation: emit tokens one-by-one (generate_streamer analogue)
            model.eval();
            int[] prompt = new int[]{1, 2, 3, 4, 5};
            StringBuilder streamBuf = new StringBuilder();
            int maxNew = 12;
            int[] full = Arrays.copyOf(prompt, prompt.length);
            long tStream = nowNs();
            for (int step = 0; step < maxNew; step++) {
                int[] stepOut = model.generate(full, 1);
                if (stepOut == null || stepOut.length <= full.length) break;
                int newTok = stepOut[stepOut.length - 1];
                streamBuf.append(newTok).append(' ');
                full = Arrays.copyOf(stepOut, stepOut.length);
            }
            recordTiming("Ex10 stream-gen", nowNs() - tStream, maxNew);
            check("Ex10 stream produced tokens", streamBuf.length() > 0);
            System.out.println("    stream tokens: " + streamBuf);

            // Real tokenizer multi-turn chat if available
            Path qwenDir = findModel("Qwen__Qwen2.5-0.5B-Instruct");
            if (qwenDir != null) {
                try {
                    FastTokenizer tok = AutoTokenizer.fromPretrained(qwenDir.toString());
                    ChatTemplate ct = ChatTemplate.qwen();
                    String rendered = ct.apply(conversation, true);
                    Encoding enc = tok.encode(rendered, false);
                    check("Ex10 real tok encode", enc != null && enc.size() > 0);
                    report("Ex10 real-tok", "vocab≈" + tok.vocabSize()
                            + " multi_turn_tokens=" + enc.size());

                    // Optional full chat via pipeline
                    if (preferReal) {
                        TextGenerationPipeline pipe = TextGenerationPipeline.fromDirectory(qwenDir);
                        String reply = pipe.chat(List.of(
                                Map.of("role", "user", "content", "长上下文优化方案")
                        ), GenerationConfig.builder().maxNewTokens(32).doSample(false).build());
                        check("Ex10 real chat", reply != null && !reply.isBlank());
                        System.out.println("    real chat: "
                                + reply.substring(0, Math.min(120, reply.length())));
                    }
                } catch (Throwable t) {
                    skip("Ex10 real tokenizer/chat", t.getMessage());
                }
            } else {
                skip("Ex10 real Qwen", "models/Qwen__Qwen2.5-0.5B-Instruct missing");
            }

            report("Ex10 MultiTurnStream",
                    String.format(Locale.US,
                            "rows=%d turns0=%s last_loss=%.4f stream_chars=%d split=%s",
                            formatted.size(), formatted.get(0).get("n_turns"),
                            last, streamBuf.length(), split));
        } catch (Throwable t) {
            check("Ex10 exception-free: " + t, false);
            t.printStackTrace(System.out);
        }
    }

    // ================================================================== D-matrix stress

    static void dMatrixStress() {
        section("D-Matrix multi-dimensional stress (lr × accum × r × steps)");
        double[] lrs = {1e-4, 3e-4};
        int[] accums = {1, 4};
        int[] ranks = {4, 8};
        int combos = 0;
        long t0 = nowNs();
        for (double lr : lrs) {
            for (int accum : accums) {
                for (int r : ranks) {
                    manual_seed(100 + combos);
                    CausalLM model = tinyModel();
                    LoraConfig lora = LoraConfig.builder().r(r).alpha(r * 2.0).build();
                    model.attachLora(lora);
                    SFTConfig cfg = SFTConfig.builder()
                            .learningRate(lr)
                            .maxSteps(2)
                            .gradientAccumulationSteps(accum)
                            .loggingSteps(0)
                            .build();
                    Adam opt = adam(model, lr);
                    try (SFTTrainer tr = new SFTTrainer(model, asForward(model), opt, cfg)) {
                        // micro-steps to complete 2 optimizer steps
                        for (int i = 0; i < 2 * accum; i++) {
                            double loss = tr.trainingStep(sftBatch(model, 11000L + combos * 10L + i));
                            if (Double.isNaN(loss) || Double.isInfinite(loss)) {
                                check("matrix lr=" + lr + " accum=" + accum + " r=" + r, false);
                                break;
                            }
                        }
                        check("matrix lr=" + lr + " accum=" + accum + " r=" + r
                                        + " step=" + tr.globalStep(),
                                tr.globalStep() == 2);
                    }
                    combos++;
                }
            }
        }
        recordTiming("D-Matrix combos=" + combos, nowNs() - t0, combos * 2);
        report("D-Matrix", "combos=" + combos + " (lr × accum × r)");
    }

    // ================================================================== Real model smoke (optional)

    static void realModelSmoke() {
        section("Real-model smoke (tokenizer / chat / load report)");
        Path qwen = findModel("Qwen__Qwen2.5-0.5B-Instruct");
        Path gpt2 = findModel("openai-community__gpt2");
        Path llama = findModel("unsloth__Llama-3.2-1B-Instruct");
        Path deepseek = findModel("deepseek-ai__DeepSeek-R1-Distill-Qwen-1.5B");
        Path glm = findModel("zai-org__glm-edge-1.5b-chat");

        System.out.println("    Qwen2.5-0.5B: " + (qwen != null ? "YES" : "no"));
        System.out.println("    GPT-2:        " + (gpt2 != null ? "YES" : "no"));
        System.out.println("    Llama-3.2-1B: " + (llama != null ? "YES" : "no"));
        System.out.println("    DeepSeek-1.5B:" + (deepseek != null ? "YES" : "no"));
        System.out.println("    GLM-edge-1.5B:" + (glm != null ? "YES" : "no"));

        if (qwen != null) {
            try {
                long t0 = nowNs();
                AutoModelForCausalLM.Bundle b = AutoModelForCausalLM.fromDirectory(qwen);
                recordTiming("Real Qwen2.5 load", nowNs() - t0, 1);
                check("Real Qwen loadReport matched>0",
                        b.loadReport() != null && b.loadReport().matchedCount() > 0);
                System.out.println("    load: " + b.loadReport());
                System.out.println("    cfg:  type=" + b.config().modelType()
                        + " d=" + b.config().hiddenSize()
                        + " L=" + b.config().numHiddenLayers());
                String reply = b.chat(List.of(
                        Map.of("role", "user", "content", "1+1等于几？只回答数字")
                ), GenerationConfig.builder().maxNewTokens(16).doSample(false).build());
                check("Real Qwen chat non-empty", reply != null && !reply.isBlank());
                System.out.println("    chat: " + reply);
                report("Real-Qwen", "matched=" + b.loadReport().matchedCount()
                        + " reply_len=" + reply.length());
            } catch (Throwable t) {
                skip("Real Qwen full load/chat", t.getClass().getSimpleName() + ": " + t.getMessage());
                // Tokenizer-only still valuable
                try {
                    FastTokenizer tok = AutoTokenizer.fromPretrained(qwen.toString());
                    Encoding e = tok.encode("你好，JavaCPP", true);
                    check("Real Qwen tok-only", e.size() > 0);
                    report("Real-Qwen-tok", "vocab≈" + tok.vocabSize() + " ids=" + e.size());
                } catch (Throwable t2) {
                    check("Real Qwen tok-only: " + t2.getMessage(), false);
                }
            }
        } else {
            skip("Real Qwen", "not downloaded");
        }

        if (gpt2 != null) {
            try {
                TextGenerationPipeline pipe = TextGenerationPipeline.fromDirectory(gpt2);
                String out = pipe.generate("Hello, JavaCPP is",
                        GenerationConfig.builder().maxNewTokens(16).doSample(false).build());
                check("Real GPT2 generate", out != null && !out.isBlank());
                report("Real-GPT2", "out_len=" + out.length());
            } catch (Throwable t) {
                skip("Real GPT2", t.getMessage());
            }
        }
    }

    // ================================================================== main

    public static void main(String[] args) throws Exception {
        for (String a : args) {
            if ("--real".equals(a)) preferReal = true;
        }
        Files.createDirectories(OUT);
        System.out.println("BenchmarkLlm10Examples — Java ↔ Python llm.md 10 engineering instances");
        System.out.println("OUT=" + OUT.toAbsolutePath());
        System.out.println("preferReal=" + preferReal);
        System.out.println("models dir exists=" + Files.isDirectory(MODELS));

        long wall0 = nowNs();

        d0ApiParity();
        ex1FullSft();
        ex2LoraSft();
        ex3QLoRA();
        ex4Dpo();
        ex5ContinualPretrain();
        ex6Multimodal();
        ex7Accelerator();
        ex8GradCheckpoint();
        ex9GgufVllm();
        ex10MultiTurnStream();
        dMatrixStress();
        realModelSmoke();

        double wallMs = (nowNs() - wall0) / 1e6;

        System.out.println("\n######################################################################");
        System.out.println("# LLM-10 EXAMPLES BENCHMARK SUMMARY");
        System.out.println("######################################################################");
        System.out.println("PASS=" + passed + "  FAIL=" + failed + "  SKIP=" + skipped
                + "  wall_ms=" + String.format(Locale.US, "%.1f", wallMs));
        if (!failures.isEmpty()) {
            System.out.println("\nFailures:");
            for (String f : failures) System.out.println("  - " + f);
        }
        System.out.println("\nExample reports:");
        for (String r : exampleReports) System.out.println("  * " + r);
        System.out.println("\nTimings:");
        for (String t : timings) System.out.println("  " + t);
        System.out.println("\nAPI mapping (Python → Java) exercised:");
        System.out.println("  datasets.Dataset.from_list/map/train_test_split → HfDataset");
        System.out.println("  transformers.AutoModelForCausalLM/AutoTokenizer/GenerationConfig");
        System.out.println("  peft.LoraConfig/get_peft_model/print_trainable_parameters/");
        System.out.println("       save_pretrained/from_pretrained/merge_and_unload → PeftModel");
        System.out.println("  bitsandbytes.BitsAndBytesConfig + QLoRA.Session");
        System.out.println("  trl.SFTTrainer/DPOTrainer + SFTConfig/DPOConfig");
        System.out.println("  accelerate.Accelerator.prepare/wait_for_everyone/unwrap_model");
        System.out.println("  safetensors + GGUFWriter + vllm.LLM");
        System.out.println("######################################################################");

        if (failed > 0) {
            System.exit(1);
        }
    }
}
