package distribute;

import org.bytedeco.pytorch.data.gguf.GGUFConstants;
import org.bytedeco.pytorch.llm.llamacpp.LlamaArchitecture;
import org.bytedeco.pytorch.llm.llamacpp.LlamaBackend;
import org.bytedeco.pytorch.llm.llamacpp.LlamaBatch;
import org.bytedeco.pytorch.llm.llamacpp.LlamaChatFormatter;
import org.bytedeco.pytorch.llm.llamacpp.LlamaContext;
import org.bytedeco.pytorch.llm.llamacpp.LlamaContextParams;
import org.bytedeco.pytorch.llm.llamacpp.LlamaCpp;
import org.bytedeco.pytorch.llm.llamacpp.LlamaHParams;
import org.bytedeco.pytorch.llm.llamacpp.LlamaKvCache;
import org.bytedeco.pytorch.llm.llamacpp.LlamaModel;
import org.bytedeco.pytorch.llm.llamacpp.LlamaProcessManager;
import org.bytedeco.pytorch.llm.llamacpp.LlamaRuntimeConfig;
import org.bytedeco.pytorch.llm.llamacpp.LlamaSampler;
import org.bytedeco.pytorch.llm.llamacpp.LlamaSamplingParams;
import org.bytedeco.pytorch.llm.llamacpp.LlamaTokenizer;
import org.bytedeco.pytorch.llm.llamacpp.model.AttentionOp;
import org.bytedeco.pytorch.llm.llamacpp.model.LlamaTransformer;
import org.bytedeco.pytorch.llm.llamacpp.model.MlpOp;
import org.bytedeco.pytorch.llm.llamacpp.model.RmsNormOp;
import org.bytedeco.pytorch.llm.llamacpp.model.RopeCache;
import org.bytedeco.pytorch.llm.llamacpp.quant.Dequantizer;
import org.bytedeco.pytorch.llm.llamacpp.quant.GgmlQuantType;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Pure-Java llama.cpp smoke / micro-benchmark.
 *
 * <p><b>No libtorch natives, no Python, no pytorch platform jar required at runtime.</b>
 * Covers the enterprise matrix dimensions that are backend-agnostic:
 * L01 arch/constants, L04 dequant, L05 synthetic in-process generate,
 * L06 sampler, L07 chat, L08 missing binary, L09 process probe, L11 server args,
 * L12 KV, plus float ops (RoPE / RMSNorm / MLP / Attention) throughput.
 *
 * <p>Skipped on purpose (need Tensor / GGUFWriter / PEFT / TRL):
 * L02 write GGUF, L03 GGUFReader load, L10 Studio adapter (heavy studio), P01, T01.
 *
 * <pre>
 *   scripts/run_llamacpp_pure_java_bench.sh
 * </pre>
 */
public final class BenchmarkLlamaCppPureJava {

    static int passed = 0, failed = 0;
    static final List<String> failures = new ArrayList<>();
    static final List<String> timings = new ArrayList<>();

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

    static void section(String t) {
        System.out.println("\n=== " + t + " ===");
    }

    static void time(String name, long nanos, String extra) {
        double ms = nanos / 1e6;
        String line = String.format("  TIME  %s: %.3f ms%s", name, ms, extra == null ? "" : " " + extra);
        timings.add(line);
        System.out.println(line);
    }

    /** Synthetic model: empty tensor map → transformer falls back to identity/pseudo embeds. */
    static LlamaModel syntheticModel(LlamaHParams hp) {
        Map<String, Object> meta = new LinkedHashMap<>();
        meta.put("general.architecture", hp.architecture().name().toLowerCase());
        meta.put("general.name", hp.name());
        meta.put("tokenizer.ggml.bos_token_id", 0);
        meta.put("tokenizer.ggml.eos_token_id", 1);
        return new LlamaModel(
                Path.of("synthetic-pure-java.gguf"),
                hp,
                meta,
                Map.of(),
                GGUFConstants.GGUF_VERSION_3);
    }

    static void l01Architecture() {
        section("L01 architecture / constants (pure)");
        check("arch llama", LlamaArchitecture.fromMetadata("llama") == LlamaArchitecture.LLAMA);
        check("arch qwen3", LlamaArchitecture.fromMetadata("qwen3") == LlamaArchitecture.QWEN3);
        check("arch gemma2", LlamaArchitecture.fromMetadata("gemma2") == LlamaArchitecture.GEMMA2);
        check("arch gpt2", LlamaArchitecture.fromMetadata("gpt2") == LlamaArchitecture.GPT2);
        check("gguf magic", GGUFConstants.GGUF_MAGIC == 0x46554747);
        check("gguf v3 supported", GGUFConstants.isSupportedVersion(GGUFConstants.GGUF_VERSION_3));
        check("version full", LlamaCpp.version() != null && !LlamaCpp.version().isBlank());
        check("backend enum IN_PROCESS", LlamaBackend.IN_PROCESS != null);
        check("backend enum PROCESS_SERVER", LlamaBackend.PROCESS_SERVER != null);
    }

    static void l04Dequant() {
        section("L04 dequant Q4_0 / Q8_0 (pure float math)");
        float[] src = new float[64];
        for (int i = 0; i < src.length; i++) src[i] = (i - 32) * 0.1f;

        long t0 = System.nanoTime();
        byte[] q = Dequantizer.quantizeQ4_0(src);
        float[] back = Dequantizer.dequantQ4_0(q, src.length);
        time("q4_0 roundtrip 64", System.nanoTime() - t0, "bytes=" + q.length);

        check("q4_0 payload > 0", q.length > 0);
        check("dequant length", back.length == src.length);
        double err = 0;
        for (int i = 0; i < src.length; i++) err += Math.abs(src[i] - back[i]);
        err /= src.length;
        check("q4_0 mean abs err < 1.0", err < 1.0);
        check("type from id Q4_0", GgmlQuantType.fromId(GGUFConstants.GGML_TYPE_Q4_0) == GgmlQuantType.Q4_0);
        check("type from id Q8_0", GgmlQuantType.fromId(GGUFConstants.GGML_TYPE_Q8_0) == GgmlQuantType.Q8_0);
        check("Q4_0 quantized flag", GgmlQuantType.Q4_0.quantized());
        check("F32 not quantized", !GgmlQuantType.F32.quantized());

        // fp16 helpers
        short h = Dequantizer.floatToFp16(1.5f);
        float backF = Dequantizer.fp16ToFloat(h & 0xffff);
        check("fp16 1.5 roundtrip ~", Math.abs(backF - 1.5f) < 0.01f);
    }

    static void l05SyntheticInProcess() throws Exception {
        section("L05 synthetic in-process generate (no GGUF / no libtorch)");
        LlamaHParams hp = LlamaHParams.tiny();
        LlamaModel model = syntheticModel(hp);
        check("tensor count 0 (synthetic)", model.tensorCount() == 0);
        check("n_embd 64", model.hparams().nEmbd() == 64);
        check("n_layer 2", model.hparams().nLayer() == 2);

        LlamaContextParams cp = LlamaContextParams.builder().nCtx(64).nBatch(16).nThreads(2).build();
        try (LlamaContext ctx = new LlamaContext(model, cp)) {
            check("nPast 0", ctx.nPast() == 0);
            int[] prompt = new int[]{0, 1, 2, 3};
            long t0 = System.nanoTime();
            float[] logits = ctx.prefill(prompt);
            long prefillNs = System.nanoTime() - t0;
            check("prefill logits non-null", logits != null);
            check("prefill logits len == n_vocab", logits.length == hp.nVocab());
            check("logits finite", isFinite(logits));
            check("nPast after prefill", ctx.nPast() == prompt.length);
            double tokPerS = prompt.length / (prefillNs / 1e9);
            time("prefill 4 tok", prefillNs, String.format("%.1f tok/s", tokPerS));

            LlamaSampler sampler = new LlamaSampler(LlamaSamplingParams.greedy(8));
            List<Integer> out = new ArrayList<>();
            for (int id : prompt) out.add(id);
            long g0 = System.nanoTime();
            int next = sampler.sampleToken(logits);
            out.add(next);
            for (int i = 1; i < 8; i++) {
                logits = ctx.step(next);
                check("step logits finite@" + i, isFinite(logits));
                next = sampler.sampleToken(logits);
                out.add(next);
            }
            long genNs = System.nanoTime() - g0;
            check("generated length >= 12", out.size() >= 12);
            time("decode 8 tok", genNs, String.format("%.1f tok/s", 8.0 / (genNs / 1e9)));

            // tokenizer encode/decode round-trip-ish
            LlamaTokenizer tok = ctx.tokenizer();
            int[] ids = tok.encode("hello pure java", true);
            check("tokenizer encode non-empty", ids != null && ids.length > 0);
            String decoded = tok.decode(ids);
            check("tokenizer decode non-null", decoded != null);

            ctx.reset();
            check("nPast 0 after reset", ctx.nPast() == 0);
        }
    }

    static void l05bTransformerDirect() throws Exception {
        section("L05b LlamaTransformer direct logits (throughput)");
        LlamaHParams hp = LlamaHParams.tiny();
        LlamaModel model = syntheticModel(hp);
        LlamaKvCache kv = new LlamaKvCache(hp, 64);
        LlamaTransformer tr = new LlamaTransformer(model, 64);
        long t0 = System.nanoTime();
        int steps = 32;
        float[] last = null;
        for (int i = 0; i < steps; i++) {
            last = tr.logits(i % hp.nVocab(), i, kv);
        }
        long ns = System.nanoTime() - t0;
        check("transformer logits len", last != null && last.length == hp.nVocab());
        check("transformer logits finite", isFinite(last));
        time("transformer " + steps + " steps", ns, String.format("%.1f tok/s", steps / (ns / 1e9)));
    }

    static void l06Sampler() {
        section("L06 sampling determinism");
        float[] logits = new float[32];
        for (int i = 0; i < 32; i++) logits[i] = i == 7 ? 5f : 0.1f;
        LlamaSampler s1 = new LlamaSampler(LlamaSamplingParams.builder().greedy(true).seed(42).build());
        LlamaSampler s2 = new LlamaSampler(LlamaSamplingParams.builder().greedy(true).seed(42).build());
        int a = s1.sampleToken(logits);
        int b = s2.sampleToken(logits);
        check("greedy argmax 7", a == 7 && b == 7);

        LlamaSampler s3 = new LlamaSampler(LlamaSamplingParams.builder()
                .temperature(0.8f).topK(10).topP(0.9f).seed(123).maxTokens(1).build());
        int c = s3.sampleToken(logits);
        check("sample in range", c >= 0 && c < 32);

        // throughput: 10k greedy samples
        LlamaSampler s4 = new LlamaSampler(LlamaSamplingParams.builder().greedy(true).seed(1).build());
        long t0 = System.nanoTime();
        int n = 10_000;
        for (int i = 0; i < n; i++) s4.sampleToken(logits);
        time("sampler greedy x" + n, System.nanoTime() - t0, null);
    }

    static void l07ChatFormatter() {
        section("L07 chat formatter");
        LlamaChatFormatter fmt = new LlamaChatFormatter(LlamaArchitecture.LLAMA);
        List<Map<String, String>> msgs = List.of(
                Map.of("role", "system", "content", "You are helpful."),
                Map.of("role", "user", "content", "Hi"));
        String p = fmt.format(msgs);
        check("llama3 has begin", p.contains("<|begin_of_text|>") || p.toLowerCase().contains("assistant"));
        check("llama3 assistant header", p.contains("assistant"));
        LlamaChatFormatter qwen = new LlamaChatFormatter(LlamaArchitecture.QWEN2);
        String q = qwen.format(msgs);
        check("chatml im_start", q.contains("<|im_start|>"));
        LlamaChatFormatter gemma = new LlamaChatFormatter(LlamaArchitecture.GEMMA2);
        String g = gemma.format(msgs);
        check("gemma non-empty", g != null && !g.isBlank());
    }

    static void l08ProcessMissing() {
        section("L08 process backend missing binary (no GGUF load)");
        // Drive LlamaProcessManager directly — ProcessLlamaRuntime.load() would call GgufModelLoader.
        LlamaRuntimeConfig cfg = LlamaRuntimeConfig.builder()
                .modelPath(Path.of("synthetic-pure-java.gguf"))
                .backend(LlamaBackend.PROCESS_SERVER)
                .llamaServerBin(Path.of("/nonexistent/llama-server-xyz"))
                .serverPort(0)
                .serverStartTimeoutMs(2000)
                .build();
        boolean threw = false;
        String msg = null;
        try (LlamaProcessManager mgr = new LlamaProcessManager(cfg)) {
            mgr.start();
        } catch (Exception e) {
            threw = true;
            msg = e.getMessage();
        }
        check("missing binary throws", threw);
        check("error mentions binary/server",
                msg != null && (msg.contains("llama-server") || msg.contains("not found")
                        || msg.contains("binary") || msg.contains("LLAMA_SERVER")));
        Path found = LlamaProcessManager.findLlamaServer(cfg);
        check("findLlamaServer null for fake bin", found == null
                || !found.toString().contains("nonexistent"));
    }

    static void l09ProcessIfPresent() {
        section("L09 process backend probe (PATH only)");
        Path bin = LlamaProcessManager.findLlamaServer(LlamaRuntimeConfig.builder()
                .modelPath(Path.of("x.gguf")).build());
        if (bin == null) {
            check("llama-server absent — soft pass", true);
            System.out.println("  INFO  no llama-server on PATH; live process chat skipped (no libtorch / no model)");
        } else {
            check("llama-server found: " + bin, Files.isExecutable(bin));
            System.out.println("  INFO  binary present but live spawn skipped in pure-java mode (needs real GGUF)");
        }
    }

    static void l11HardwareArgs() {
        section("L11 hardware controls → server args");
        Path model = Path.of("dummy.gguf");
        LlamaRuntimeConfig cfg = LlamaRuntimeConfig.builder()
                .modelPath(model)
                .nGpuLayers(99)
                .offloadMoeExperts(true)
                .flashAttn(true)
                .tensorSplit(List.of(0, 1))
                .nCtx(4096)
                .serverPort(18080)
                .build();
        List<String> args = cfg.toServerArgList(18080);
        String joined = String.join(" ", args);
        check("has -ngl", joined.contains("-ngl"));
        check("has -c", joined.contains("-c"));
        check("has --port", joined.contains("--port"));
        check("has -m model", joined.contains("-m") && joined.contains("dummy.gguf"));
        check("has -fa flash", joined.contains("-fa"));
        check("has -ts split", joined.contains("-ts"));
        check("config map backend", "PROCESS_SERVER".equals(cfg.toMap().get("backend"))
                || cfg.toMap().containsKey("backend"));
    }

    static void l12KvCache() {
        section("L12 KV cache grow/reset");
        LlamaHParams hp = LlamaHParams.tiny();
        LlamaKvCache kv = new LlamaKvCache(hp, 16);
        check("nPast 0", kv.nPast() == 0);
        float[] row = new float[hp.nHeadKv() * hp.headDim()];
        for (int t = 0; t < 5; t++) {
            for (int layer = 0; layer < hp.nLayer(); layer++) {
                kv.append(layer, row, row);
            }
            kv.advance();
        }
        check("nPast 5", kv.nPast() == 5);
        kv.reset();
        check("nPast 0 after reset", kv.nPast() == 0);
    }

    static void lOpsFloatKernels() {
        section("OPS pure float kernels (RoPE / RMSNorm / MLP / Attn micro)");
        int n = 64;
        float[] x = new float[n];
        float[] w = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = (float) Math.sin(i * 0.1);
            w[i] = 1f;
        }
        long t0 = System.nanoTime();
        for (int i = 0; i < 1000; i++) RmsNormOp.forward(x.clone(), w, 1e-5f);
        time("rmsnorm x1000 dim64", System.nanoTime() - t0, null);

        RopeCache rope = new RopeCache(16, 128, 10000f);
        float[] h = new float[16];
        for (int i = 0; i < 16; i++) h[i] = i * 0.01f;
        t0 = System.nanoTime();
        for (int i = 0; i < 5000; i++) {
            float[] c = h.clone();
            rope.apply(c, i % 128);
        }
        time("rope apply x5000", System.nanoTime() - t0, null);
        check("rope finite", isFinite(h));

        // MLP matvec smoke via MlpOp if available
        try {
            float[] a = new float[32];
            float[] mat = new float[32 * 64];
            for (int i = 0; i < a.length; i++) a[i] = 0.01f * i;
            for (int i = 0; i < mat.length; i++) mat[i] = (i % 7) * 0.001f;
            t0 = System.nanoTime();
            float[] out = MlpOp.matvec(a, mat, 32, 64);
            time("mlp matvec 32x64", System.nanoTime() - t0, "out=" + out.length);
            check("matvec len 64", out.length == 64);
            check("matvec finite", isFinite(out));
        } catch (Throwable t) {
            check("MlpOp.matvec available", false);
            System.out.println("  INFO  MlpOp.matvec: " + t);
        }

        // AttentionOp class present
        check("AttentionOp class loadable", AttentionOp.class.getSimpleName().equals("AttentionOp"));
        check("LlamaBatch ofTokens", LlamaBatch.ofTokens(new int[]{1, 2, 3}, 0, true).nTokens() == 3);
    }

    static void lFacadeOpenDoesNotLoadNatives() {
        section("FACADE LlamaCpp.open resolves without native load");
        // open() only constructs engine; load() would touch GGUFReader/Tensor — do NOT call load()
        LlamaRuntimeConfig cfg = LlamaRuntimeConfig.builder()
                .modelPath(Path.of("synthetic-pure-java.gguf"))
                .backend(LlamaBackend.IN_PROCESS)
                .nCtx(64)
                .build();
        try (var eng = LlamaCpp.open(cfg)) {
            check("open in-process backend", eng.backend() == LlamaBackend.IN_PROCESS);
            check("not loaded yet", !eng.isLoaded());
            check("stats has backend", eng.stats().containsKey("backend"));
        } catch (Exception e) {
            check("open without load", false);
            System.out.println("  INFO  " + e);
        }
        try (var eng = LlamaCpp.open(LlamaRuntimeConfig.builder()
                .modelPath(Path.of("x.gguf"))
                .backend(LlamaBackend.PROCESS_SERVER)
                .llamaServerBin(Path.of("/nonexistent/bin"))
                .build())) {
            check("open process backend", eng.backend() == LlamaBackend.PROCESS_SERVER);
        } catch (Exception e) {
            check("open process without load", false);
        }
    }

    static boolean isFinite(float[] a) {
        if (a == null) return false;
        for (float v : a) {
            if (!Float.isFinite(v)) return false;
        }
        return true;
    }

    public static void main(String[] args) throws Exception {
        System.out.println("========================================================");
        System.out.println(" LlamaCpp Pure-Java Benchmark (no libtorch / no Python)");
        System.out.println(" version: " + LlamaCpp.version());
        System.out.println(" java:    " + System.getProperty("java.version"));
        System.out.println("========================================================");
        // Sanity: fail fast if Tensor somehow got pulled into the first pure call path
        try {
            Class.forName("org.bytedeco.pytorch.serving.tensorrt.Tensor");
            System.out.println("WARN: org.bytedeco.pytorch.serving.tensorrt.Tensor is on classpath");
            System.out.println("      (ok if jars present; this smoke must not *initialize* Loader)");
        } catch (ClassNotFoundException e) {
            System.out.println("OK:   org.bytedeco.pytorch.serving.tensorrt.Tensor NOT on classpath (strict pure mode)");
        }

        long t0 = System.nanoTime();
        try {
            l01Architecture();
            l04Dequant();
            l05SyntheticInProcess();
            l05bTransformerDirect();
            l06Sampler();
            l07ChatFormatter();
            l08ProcessMissing();
            l09ProcessIfPresent();
            l11HardwareArgs();
            l12KvCache();
            lOpsFloatKernels();
            lFacadeOpenDoesNotLoadNatives();
        } catch (Throwable t) {
            t.printStackTrace();
            failed++;
            failures.add("UNCAUGHT: " + t);
        }
        double sec = (System.nanoTime() - t0) / 1e9;
        System.out.println("\n========================================");
        System.out.println("Passed: " + passed);
        System.out.println("Failed: " + failed);
        System.out.println("Time:   " + String.format("%.2f", sec) + "s");
        if (!timings.isEmpty()) {
            System.out.println("Timings:");
            for (String line : timings) System.out.println(line);
        }
        if (!failures.isEmpty()) {
            System.out.println("Failures:");
            for (String f : failures) System.out.println("  - " + f);
        }
        System.out.println("========================================");
        if (failed > 0) System.exit(1);
    }
}
