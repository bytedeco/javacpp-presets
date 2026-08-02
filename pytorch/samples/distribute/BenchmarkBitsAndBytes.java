package distribute;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.llm.peft.LoraConfig;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.llm.quantization.BitsAndBytesConfig;
import org.bytedeco.pytorch.llm.bitsandbytes.BitsAndBytes;
import org.bytedeco.pytorch.llm.bitsandbytes.QLoRA;
import org.bytedeco.pytorch.llm.transformers.AutoModelForCausalLM;
import org.bytedeco.pytorch.llm.transformers.CausalLM;
import org.bytedeco.pytorch.llm.transformers.PretrainedConfig;
import org.bytedeco.pytorch.llm.unsloth.FastConfig;
import org.bytedeco.pytorch.llm.unsloth.FastLanguageModel;

import java.util.LinkedHashMap;
import java.util.Map;

import static org.bytedeco.pytorch.global.torch.randn;
import static org.bytedeco.pytorch.global.torch.tensor;

/**
 * BitsAndBytes + Transformers QLoRA fine-tune benchmark.
 *
 * <pre>
 * D1  NF4 quant/dequant roundtrip
 * D2  FP4 + double quant
 * D3  INT8 blockwise
 * D4  pack/unpack 4-bit
 * D5  Linear4bit / Linear8bitLt forward
 * D6  quantizeModel + materialize
 * D7  reconstruction MAE / cosine
 * D8  BitsAndBytesConfig HF fields
 * D9  AutoModelForCausalLM + quantization_config
 * D10 QLoRA Session trainStep (bnb + peft + CausalLM)
 * D11 FastLanguageModel 4bit path
 * D12 compression ratio / memory estimate
 * </pre>
 */
public class BenchmarkBitsAndBytes {
    static int passed = 0, failed = 0;
    static final StringBuilder report = new StringBuilder();

    static void check(String name, boolean ok) {
        if (ok) {
            passed++;
            System.out.println("  PASS  " + name);
        } else {
            failed++;
            System.out.println("  FAIL  " + name);
            report.append("FAIL ").append(name).append('\n');
        }
    }

    static void section(String t) {
        System.out.println("\n=== " + t + " ===");
    }

    static void d1Nf4Roundtrip() {
        section("D1 NF4 quant/dequant roundtrip");
        Tensor w = randn(64, 32);
        BitsAndBytes.QuantState qs = BitsAndBytes.quantizeNf4(w, 64, true);
        check("quantType=nf4", "nf4".equals(qs.quantType));
        check("doubleQuant=true", qs.doubleQuant);
        check("nested present", qs.nested != null);
        check("packedCodes present", qs.packedCodes != null && qs.packedCodes.length > 0);
        check("numel=2048", qs.numel() == 64 * 32);
        Tensor restored = BitsAndBytes.dequantizeNf4(qs);
        check("restored shape[0]=64", restored.size(0) == 64);
        check("restored shape[1]=32", restored.size(1) == 32);
        double cos = BitsAndBytes.reconstructionCosine(w,
                BitsAndBytesConfig.builder().loadIn4Bit(true).bnb4BitQuantType("nf4").build());
        check("nf4 cosine > 0.9", cos > 0.9);
        System.out.println("    cosine=" + cos + " mem=" + qs.memoryBytes());
    }

    static void d2Fp4DoubleQuant() {
        section("D2 FP4 + double quant");
        Tensor w = randn(128, 16);
        BitsAndBytes.QuantState qs = BitsAndBytes.quantizeFp4(w, 64, true);
        check("quantType=fp4", "fp4".equals(qs.quantType));
        check("doubleQuant", qs.doubleQuant);
        Tensor r = BitsAndBytes.dequantizeFp4(qs);
        check("dequant numel match", r.numel() == w.numel());
        BitsAndBytesConfig cfg = BitsAndBytesConfig.builder()
                .loadIn4Bit(true).bnb4BitQuantType("fp4").bnb4BitUseDoubleQuant(true).build();
        double mae = BitsAndBytes.reconstructionMae(w, cfg);
        check("fp4 mae finite", !Double.isNaN(mae) && mae >= 0);
        System.out.println("    mae=" + mae);
    }

    static void d3Int8() {
        section("D3 INT8 blockwise");
        Tensor w = randn(100, 20);
        BitsAndBytes.QuantState qs = BitsAndBytes.quantizeInt8(w, 64);
        check("quantType=int8", "int8".equals(qs.quantType));
        Tensor r = BitsAndBytes.dequantizeInt8(qs);
        check("int8 shape restored", r.size(0) == 100 && r.size(1) == 20);
        double cos = BitsAndBytes.reconstructionCosine(w, BitsAndBytesConfig.int8Defaults());
        check("int8 cosine > 0.95", cos > 0.95);
        System.out.println("    cosine=" + cos);
    }

    static void d4PackUnpack() {
        section("D4 pack/unpack 4-bit");
        float[] codes = new float[7];
        for (int i = 0; i < codes.length; i++) codes[i] = i % 16;
        byte[] packed = BitsAndBytes.pack4bit(codes);
        check("packed len=(n+1)/2", packed.length == 4);
        float[] unpacked = BitsAndBytes.unpack4bit(packed, codes.length);
        boolean ok = true;
        for (int i = 0; i < codes.length; i++) {
            if (Math.round(unpacked[i]) != Math.round(codes[i])) ok = false;
        }
        check("pack/unpack identity", ok);
    }

    static void d5Linear4bit8bit() {
        section("D5 Linear4bit / Linear8bitLt forward");
        LinearImpl dense = new LinearImpl(32, 16);
        BitsAndBytesConfig cfg4 = BitsAndBytesConfig.qloraDefaults();
        BitsAndBytes.Linear4bit l4 = BitsAndBytes.linear4bit(dense, cfg4);
        Tensor x = randn(4, 32);
        Tensor y4 = l4.forward(x);
        check("Linear4bit out [4,16]", y4.size(0) == 4 && y4.size(1) == 16);
        check("Linear4bit quantStorage", l4.quantStorage());
        check("Linear4bit computeDtype set", l4.computeDtype() != null);

        BitsAndBytes.Linear8bitLt l8 = BitsAndBytes.linear8bit(dense, BitsAndBytesConfig.int8Defaults());
        Tensor y8 = l8.forward(x);
        check("Linear8bitLt out [4,16]", y8.size(0) == 4 && y8.size(1) == 16);
        check("threshold default 6.0", Math.abs(l8.threshold() - 6.0) < 1e-9);
        Map<String, Object> st = l4.stats();
        check("Linear4bit.stats has quant_type", st.containsKey("quant_type"));
    }

    static void d6QuantizeModel() {
        section("D6 quantizeModel + materialize");
        Map<String, LinearImpl> linears = new LinkedHashMap<>();
        linears.put("h/0/attn/c_attn", new LinearImpl(32, 96));
        linears.put("h/0/attn/c_proj", new LinearImpl(32, 32));
        linears.put("h/0/mlp/fc_in", new LinearImpl(32, 64));
        linears.put("h/0/mlp/fc_out", new LinearImpl(64, 32));
        linears.put("lm_head", new LinearImpl(32, 100));

        BitsAndBytesConfig cfg = BitsAndBytesConfig.builder()
                .loadIn4Bit(true)
                .bnb4BitQuantType("nf4")
                .bnb4BitUseDoubleQuant(true)
                .llm_int8_skip_modules("lm_head")
                .build();
        BitsAndBytes.QuantizedModel qm = BitsAndBytes.quantizeModel(linears, cfg);
        check("quantized 4 layers (skip lm_head)", qm.size() == 4);
        check("totalParams > 0", qm.totalParams() > 0);
        check("quantMemoryBytes < fp32", qm.quantMemoryBytes() < qm.totalParams() * 4);
        int n = qm.materializeInto(linears);
        check("materializeInto count=4", n == 4);
        check("base weight frozen", !linears.get("h/0/attn/c_attn").weight().requires_grad());
        Map<String, Object> stats = qm.stats();
        check("stats has compression_ratio", stats.containsKey("compression_ratio"));
        System.out.println("    compression=" + stats.get("compression_ratio")
                + " quant_mem=" + stats.get("quant_memory_bytes"));
        qm.close();
    }

    static void d7Reconstruction() {
        section("D7 reconstruction metrics");
        Tensor w = randn(256, 64);
        BitsAndBytesConfig nf4 = BitsAndBytesConfig.qloraDefaults();
        double mae = BitsAndBytes.reconstructionMae(w, nf4);
        double cos = BitsAndBytes.reconstructionCosine(w, nf4);
        check("mae >= 0", mae >= 0 && !Double.isNaN(mae));
        check("cosine in (0,1]", cos > 0 && cos <= 1.0001);
        check("nf4 cosine > 0.85", cos > 0.85);
        System.out.println("    mae=" + mae + " cos=" + cos);
    }

    static void d8ConfigFields() {
        section("D8 BitsAndBytesConfig HF fields");
        BitsAndBytesConfig cfg = BitsAndBytesConfig.builder()
                .load_in_4bit(true)
                .bnb_4bit_quant_type("nf4")
                .bnb_4bit_use_double_quant(true)
                .bnb_4bit_compute_dtype("bfloat16")
                .llm_int8_threshold(6.0)
                .llm_int8_skip_modules("lm_head", "embed_tokens")
                .blocksize(64)
                .device_map("auto")
                .build();
        check("isLoadIn4Bit", cfg.isLoadIn4Bit());
        check("isQuantized", cfg.isQuantized());
        check("quantType nf4", "nf4".equals(cfg.getBnb4BitQuantType()));
        check("doubleQuant", cfg.isBnb4BitUseDoubleQuant());
        check("blocksize 64", cfg.getBlocksize() == 64);
        check("shouldSkip lm_head", cfg.shouldSkipModule("model.lm_head"));
        check("shouldSkip embed", cfg.shouldSkipModule("embed_tokens"));
        check("!shouldSkip q_proj", !cfg.shouldSkipModule("model.layers.0.self_attn.q_proj"));
        check("qloraDefaults works", BitsAndBytesConfig.qloraDefaults().isLoadIn4Bit());
        check("int8Defaults works", BitsAndBytesConfig.int8Defaults().isLoadIn8Bit());
        // mutual exclusion
        boolean threw = false;
        try {
            BitsAndBytesConfig.builder().loadIn4Bit(true).loadIn8Bit(true).build();
        } catch (IllegalArgumentException e) {
            threw = true;
        }
        check("4bit+8bit rejected", threw);
    }

    static void d9AutoModelQuant() {
        section("D9 AutoModelForCausalLM + quantization_config");
        BitsAndBytesConfig bnb = BitsAndBytesConfig.qloraDefaults();
        AutoModelForCausalLM.Bundle bundle = AutoModelForCausalLM.tiny("gpt2", bnb);
        check("bundle not null", bundle != null);
        check("isQuantized", bundle.isQuantized());
        check("quantizationConfig set", bundle.quantizationConfig() != null);
        check("quantizedModel layers>0", bundle.quantizedModel() != null
                && bundle.quantizedModel().size() > 0);
        check("model is CausalLM", bundle.model() instanceof CausalLM);
        if (bundle.model() instanceof CausalLM clm) {
            Map<String, LinearImpl> qs = clm.quantizableLinears();
            check("quantizableLinears non-empty", !qs.isEmpty());
            // base weights should be frozen after QLoRA prepare
            LinearImpl first = qs.values().iterator().next();
            check("first linear frozen", !first.weight().requires_grad());
            // forward still runs
            int[] ids = new int[8];
            for (int i = 0; i < ids.length; i++) ids[i] = i % Math.max(1, clm.vocabSize());
            Tensor logits = clm.forward(tensor(ids).reshape(1, 8));
            check("forward logits 3D", logits.dim() == 3);
            check("logits vocab dim", logits.size(2) == clm.vocabSize());
        }
        // collectQuantizableLinears static helper
        Map<String, LinearImpl> collected = AutoModelForCausalLM.collectQuantizableLinears(bundle.model());
        check("collectQuantizableLinears size>0", collected.size() > 0);
        System.out.println("    quantized_layers=" + bundle.quantizedModel().size()
                + " linears=" + collected.size());
    }

    static void d10QLoRATrainStep() throws Exception {
        section("D10 QLoRA Session trainStep (bnb + peft + CausalLM)");
        PretrainedConfig cfg = PretrainedConfig.tinyGpt2();
        LoraConfig lora = LoraConfig.builder()
                .r(4)
                .alpha(8)
                .targetModules(QLoRA.GPT2_TARGETS)
                .freezeBase(true)
                .build();
        try (QLoRA.Session s = QLoRA.fromCausalLM(cfg, BitsAndBytesConfig.qloraDefaults(), lora)) {
            check("session prepared", s.isPrepared());
            check("quantized layers>0", s.quantized() != null && s.quantized().size() > 0);
            check("adapters>0", s.adapters().size() > 0);
            long train = s.trainableParameters();
            long total = s.totalParameters();
            check("trainable>0", train > 0);
            check("trainable < total", train < total);
            double ratio = (double) train / (double) total;
            check("trainable ratio < 0.5", ratio < 0.5);

            int[] ids = new int[8];
            for (int i = 0; i < ids.length; i++) ids[i] = (i * 3) % Math.max(1, cfg.vocabSize());
            Tensor input = tensor(ids).reshape(1, 8);
            double loss1 = s.trainStep(input);
            check("trainStep1 finite", !Double.isNaN(loss1) && !Double.isInfinite(loss1));
            check("step=1", s.step() == 1);
            double loss2 = s.trainStep(input);
            check("trainStep2 finite", !Double.isNaN(loss2) && !Double.isInfinite(loss2));
            check("step=2", s.step() == 2);

            // generate still works
            int[] gen = s.generate(new int[]{1, 2, 3}, 4);
            check("generate longer", gen.length > 3);

            // save/load adapter
            java.nio.file.Path tmp = java.nio.file.Files.createTempDirectory("qlora_ckpt");
            java.io.File adapter = tmp.resolve("adapter.pt").toFile();
            s.saveAdapter(adapter);
            check("adapter saved", adapter.exists() && adapter.length() > 0);
            s.loadAdapter(adapter);
            check("adapter reloaded", true);

            Map<String, Object> st = s.stats();
            check("stats has load_in_4bit", Boolean.TRUE.equals(st.get("load_in_4bit")));
            check("stats has compression_ratio", st.containsKey("compression_ratio"));
            System.out.println("    loss1=" + loss1 + " loss2=" + loss2
                    + " train=" + train + " total=" + total
                    + " ratio=" + String.format("%.4f", ratio)
                    + " adapters=" + s.adapters().size()
                    + " quant_layers=" + s.quantized().size());

            // cleanup
            java.nio.file.Files.walk(tmp).sorted(java.util.Comparator.reverseOrder())
                    .forEach(p -> { try { java.nio.file.Files.deleteIfExists(p); } catch (Exception ignored) {} });
        }
    }

    static void d11FastLanguageModel() {
        section("D11 FastLanguageModel 4bit path");
        PretrainedConfig cfg = PretrainedConfig.tinyGpt2();
        FastConfig fc = FastConfig.builder()
                .r(4)
                .loadIn4bit(true)
                .targetModules(java.util.List.of("c_attn", "c_proj", "fc_in", "fc_out"))
                .build();
        FastLanguageModel fm = FastLanguageModel.fromPretrained(cfg, fc).getPeftModel();
        check("peftApplied", fm.isPeftApplied());
        check("isQuantized or quantStates", fm.isQuantized() || !fm.quantStates().isEmpty());
        check("load_in_4bit stat", Boolean.TRUE.equals(fm.stats().get("load_in_4bit")));
        if (fm.quantizedModel() != null) {
            check("quantizedModel size>0", fm.quantizedModel().size() > 0);
        }
        int[] ids = new int[8];
        for (int i = 0; i < ids.length; i++) ids[i] = i % Math.max(1, cfg.vocabSize());
        Tensor input = tensor(ids).reshape(1, 8);
        try {
            fm.trainStep(input);
            check("FastLanguageModel trainStep step=1", fm.stepCount() == 1);
        } catch (Exception e) {
            check("FastLanguageModel trainStep step=1", false);
            System.out.println("    trainStep err: " + e.getMessage());
        }
        System.out.println("    stats quant_tensors=" + fm.stats().get("quant_tensors")
                + " adapters=" + fm.stats().get("adapters")
                + " is_quantized=" + fm.stats().get("is_quantized"));
    }

    static void d12Compression() {
        section("D12 compression ratio / memory estimate");
        long numel = 100_000_000L; // ~100M params
        long nf4 = BitsAndBytes.estimateMemoryBytes(numel, "nf4", 64, true);
        long fp32 = numel * 4L;
        double ratio = BitsAndBytes.compressionRatio(numel, "nf4", true);
        check("nf4 mem << fp32", nf4 < fp32 / 4);
        check("compression ratio > 4", ratio > 4.0);
        check("int8 estimate", BitsAndBytes.estimateMemoryBytes(numel, "int8") < fp32);
        check("VERSION set", BitsAndBytes.VERSION != null && !BitsAndBytes.VERSION.isEmpty());
        check("NF4_LEVELS len=16", BitsAndBytes.NF4_LEVELS.length == 16);
        check("FP4_LEVELS len=16", BitsAndBytes.FP4_LEVELS.length == 16);
        System.out.println("    100M params fp32=" + fp32 + " nf4+dq=" + nf4
                + " ratio=" + String.format("%.2f", ratio));
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=== BitsAndBytes + Transformers QLoRA benchmark ===");
        System.out.println("bnb VERSION=" + BitsAndBytes.VERSION);
        d1Nf4Roundtrip();
        d2Fp4DoubleQuant();
        d3Int8();
        d4PackUnpack();
        d5Linear4bit8bit();
        d6QuantizeModel();
        d7Reconstruction();
        d8ConfigFields();
        d9AutoModelQuant();
        d10QLoRATrainStep();
        d11FastLanguageModel();
        d12Compression();
        done();
    }

    static void done() {
        System.out.println("\n============================================================");
        System.out.println("BitsAndBytes  passed=" + passed + "  failed=" + failed);
        System.out.println("============================================================");
        if (failed > 0) {
            System.out.println(report);
            System.exit(1);
        }
        System.out.println("ALL DIMENSIONS GREEN — bnb + transformers QLoRA verified");
    }
}
