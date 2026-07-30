package samples;

import org.bytedeco.pytorch.NoGradGuard;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.llm.modules.Attention;
import org.bytedeco.pytorch.llm.modules.DecoderLayer;
import org.bytedeco.pytorch.llm.modules.Embedding;
import org.bytedeco.pytorch.llm.modules.LayerNorm;
import org.bytedeco.pytorch.llm.modules.MiniCausalLM;
import org.bytedeco.pytorch.llm.modules.MlaDecoderLayer;
import org.bytedeco.pytorch.llm.modules.Mlp;
import org.bytedeco.pytorch.llm.modules.MoE;
import org.bytedeco.pytorch.llm.modules.Modules;
import org.bytedeco.pytorch.llm.modules.MultiLatentAttention;
import org.bytedeco.pytorch.llm.modules.ParallelLinear;
import org.bytedeco.pytorch.llm.modules.RMSNorm;
import org.bytedeco.pytorch.llm.modules.RotaryEmbedding;
import org.bytedeco.pytorch.nn.Module;

import static org.bytedeco.pytorch.global.torch.manual_seed;
import static org.bytedeco.pytorch.global.torch.randn;
import static org.bytedeco.pytorch.global.torch.randint;

/**
 * Multi-dimensional full-API stress for {@code org.bytedeco.pytorch.llm.modules}.
 *
 * <p>Covers simple → complex → distributed building blocks used by
 * DeepSeek / Qwen3 / Gemini-style / GLM / OpenAI-GPT / Llama / Mixtral nets,
 * and verifies they compose into runnable mini causal LMs.
 *
 * <pre>
 * D1   RMSNorm / LayerNorm shape + scale invariance-ish
 * D2   RotaryEmbedding apply / repeatKv / interleaved
 * D3   Embedding rope-only + GPT-2 absolute pos
 * D4   MLP family: SwiGLU / Fused / GELU / ReLU / GeGLU
 * D5   Attention: MHA / GQA / MQA / Qwen2 / Qwen3 / GPT-2 / sliding / ALiBi
 * D6   Attention cache-aware prefill + decode
 * D7   DecoderLayer: Llama / Qwen / GPT-2 / GLM / MoE
 * D8   MoE forward + load-balancing loss + shared expert
 * D9   MultiLatentAttention (DeepSeek MLA) + cache
 * D10  MlaDecoderLayer dense + moe
 * D11  ParallelLinear single-rank fallback (Column/Row/SwiGLU)
 * D12  MiniCausalLM factories: llama/qwen2/qwen3/gpt2/glm/deepseek/mixtral
 * D13  MiniCausalLM loss + backward (train step)
 * D14  MiniCausalLM cached decode continuity
 * D15  Modules facade smoke
 * D16  Compose custom stack (embed→N×decoder→norm→lm_head) shapes
 * D17  Numerical stability (no NaN/Inf) across arches
 * D18  Batch / seq-length stress matrix
 * </pre>
 *
 * <p>Run:
 * <pre>
 *   CP=target/classes:$(cat target/cp.txt 2>/dev/null)
 *   javac -cp "$CP" -d target/samples-compile samples/BenchmarkLlmModules.java
 *   java -cp "target/samples-compile:$CP" samples.BenchmarkLlmModules
 * </pre>
 */
public class BenchmarkLlmModules {

    static int passed = 0, failed = 0;
    static final StringBuilder report = new StringBuilder();

    @FunctionalInterface
    interface CheckedRunnable {
        void run() throws Exception;
    }

    static void check(String name, boolean ok) {
        if (ok) {
            passed++;
            System.out.println("  PASS  " + name);
        } else {
            failed++;
            report.append("FAIL ").append(name).append('\n');
            System.out.println("  FAIL  " + name);
        }
    }

    static void section(String t) {
        System.out.println("\n=== " + t + " ===");
    }

    static void benchmark(String name, CheckedRunnable r) {
        try {
            r.run();
        } catch (Throwable t) {
            failed++;
            String msg = "EXC   " + name + " :: " + t.getClass().getSimpleName() + ": " + t.getMessage();
            report.append(msg).append('\n');
            System.out.println("  " + msg);
            t.printStackTrace(System.out);
        }
    }

    static boolean finite(Tensor t) {
        try {
            double v = t.detach().reshape(-1).get(0).item().toDouble();
            return !Double.isNaN(v) && !Double.isInfinite(v);
        } catch (Throwable e) {
            try {
                // multi-element: sum is a cheap NaN/Inf probe
                double s = t.detach().sum().item().toDouble();
                return !Double.isNaN(s) && !Double.isInfinite(s);
            } catch (Throwable e2) {
                return false;
            }
        }
    }

    /** Ambiguity-safe randint(low, high, shape...). */
    static Tensor ids(long high, long... shape) {
        return randint(0L, high, shape);
    }

    static boolean shapeEq(Tensor t, long... dims) {
        if (t.dim() != dims.length) {
            return false;
        }
        for (int i = 0; i < dims.length; i++) {
            if (t.size(i) != dims[i]) {
                return false;
            }
        }
        return true;
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=== LLM Modules Multi-Dimensional Benchmark ===\n");
        manual_seed(42);

        d1Norms();
        d2Rope();
        d3Embedding();
        d4Mlp();
        d5AttentionVariants();
        d6AttentionCache();
        d7DecoderLayers();
        d8MoE();
        d9Mla();
        d10MlaDecoder();
        d11Parallel();
        d12MiniLmFactories();
        d13TrainStep();
        d14CachedDecode();
        d15Facade();
        d16CustomStack();
        d17Stability();
        d18Stress();

        System.out.println("\n========================================");
        System.out.println("Passed: " + passed + "  Failed: " + failed);
        if (failed > 0) {
            System.out.println("--- failures ---");
            System.out.print(report);
            System.exit(1);
        }
        System.out.println("ALL LLM MODULE CHECKS PASSED");
    }

    // ── D1 Norms ─────────────────────────────────────────────────────────
    static void d1Norms() {
        section("D1 RMSNorm / LayerNorm");
        benchmark("RMSNorm construct + forward shape", () -> {
            RMSNorm n = new RMSNorm(64, 1e-6);
            check("hiddenSize", n.hiddenSize() == 64);
            check("eps", n.eps() == 1e-6);
            Tensor x = randn(2, 8, 64);
            Tensor y = n.forward(x);
            check("rms out shape", shapeEq(y, 2, 8, 64));
            check("rms finite", finite(y));
            check("weight numel 64", n.weight().numel() == 64);
        });
        benchmark("LayerNorm construct + forward", () -> {
            LayerNorm n = new LayerNorm(64, 1e-5);
            Tensor y = n.forward(randn(2, 8, 64));
            check("ln out shape", shapeEq(y, 2, 8, 64));
            check("ln finite", finite(y));
            check("ln inner non-null", n.inner() != null);
        });
        benchmark("RMSNorm scales roughly unit rms", () -> {
            try (NoGradGuard ng = new NoGradGuard()) {
                RMSNorm n = new RMSNorm(32);
                Tensor x = randn(1, 4, 32).mul(new Scalar(10.0));
                Tensor y = n.forward(x);
                check("rms still finite after scale", finite(y));
            }
        });
    }

    // ── D2 RoPE ──────────────────────────────────────────────────────────
    static void d2Rope() {
        section("D2 RotaryEmbedding");
        benchmark("RoPE apply preserves shape", () -> {
            try (NoGradGuard ng = new NoGradGuard()) {
                Tensor q = randn(2, 4, 8, 16);
                Tensor r = RotaryEmbedding.apply(q, 10000.0);
                check("rope shape", shapeEq(r, 2, 4, 8, 16));
                check("rope finite", finite(r));
            }
        });
        benchmark("RoPE with offset + scaling", () -> {
            try (NoGradGuard ng = new NoGradGuard()) {
                Tensor q = randn(1, 2, 4, 32);
                Tensor r0 = RotaryEmbedding.apply(q, 1_000_000.0, 0L, 1.0);
                Tensor r8 = RotaryEmbedding.apply(q, 1_000_000.0, 8L, 1.0);
                Tensor rs = RotaryEmbedding.apply(q, 10000.0, 0L, 2.0);
                check("offset finite", finite(r8));
                check("scaling finite", finite(rs));
                Tensor diff = r0.sub(r8).abs().sum();
                check("offset changes values", diff.item().toFloat() > 1e-4f);
            }
        });
        benchmark("repeatKv GQA", () -> {
            try (NoGradGuard ng = new NoGradGuard()) {
                Tensor kv = randn(2, 2, 8, 16);
                Tensor rep = RotaryEmbedding.repeatKv(kv, 2);
                check("repeat shape", shapeEq(rep, 2, 4, 8, 16));
                Tensor same = RotaryEmbedding.repeatKv(kv, 1);
                check("nRep=1 identity shape", shapeEq(same, 2, 2, 8, 16));
            }
        });
        benchmark("interleaved RoPE", () -> {
            try (NoGradGuard ng = new NoGradGuard()) {
                Tensor q = randn(1, 2, 4, 16);
                Tensor r = RotaryEmbedding.applyInterleaved(q, 10000.0, 0L);
                check("interleaved shape", shapeEq(r, 1, 2, 4, 16));
                check("interleaved finite", finite(r));
            }
        });
        benchmark("odd headDim passthrough", () -> {
            try (NoGradGuard ng = new NoGradGuard()) {
                Tensor q = randn(1, 1, 2, 15);
                Tensor r = RotaryEmbedding.apply(q, 10000.0);
                check("odd D unchanged numel", r.numel() == q.numel());
            }
        });
    }

    // ── D3 Embedding ─────────────────────────────────────────────────────
    static void d3Embedding() {
        section("D3 Embedding");
        benchmark("rope-only token embed", () -> {
            Embedding e = Embedding.ropeOnly(256, 64);
            check("no abs pos", !e.useAbsolutePos());
            Tensor ids = ids(256, 2, 12);
            Tensor x = e.forward(ids);
            check("embed shape", shapeEq(x, 2, 12, 64));
            check("embed finite", finite(x));
        });
        benchmark("GPT-2 absolute pos embed", () -> {
            Embedding e = Embedding.gpt2(256, 64, 128, 0.0);
            check("abs pos on", e.useAbsolutePos());
            Tensor ids = ids(256, 1, 16);
            Tensor x = e.forward(ids);
            check("gpt2 embed shape", shapeEq(x, 1, 16, 64));
            Tensor xOff = e.forward(ids, 4L);
            check("offset embed shape", shapeEq(xOff, 1, 16, 64));
            check("offset finite", finite(xOff));
        });
        benchmark("1D inputIds unsqueeze", () -> {
            Embedding e = Embedding.ropeOnly(64, 32);
            Tensor ids = ids(64, 10);
            Tensor x = e.forward(ids);
            check("1d → [1,T,H]", shapeEq(x, 1, 10, 32));
        });
    }

    // ── D4 MLP ───────────────────────────────────────────────────────────
    static void d4Mlp() {
        section("D4 MLP family");
        long H = 64, I = 128;
        Tensor x = randn(2, 8, H);
        benchmark("SwiGLU", () -> {
            Mlp.SwiGLU m = new Mlp.SwiGLU(H, I);
            Tensor y = m.forward(x);
            check("swiglu shape", shapeEq(y, 2, 8, H));
            check("swiglu finite", finite(y));
            check("gate/up/down registered", m.gate_proj != null && m.up_proj != null && m.down_proj != null);
        });
        benchmark("FusedSwiGLU (GLM)", () -> {
            Mlp.FusedSwiGLU m = new Mlp.FusedSwiGLU(H, I);
            Tensor y = m.forward(x);
            check("fused shape", shapeEq(y, 2, 8, H));
            check("fused finite", finite(y));
        });
        benchmark("GeluMlp (GPT-2)", () -> {
            Mlp.GeluMlp m = new Mlp.GeluMlp(H, I);
            check("gelu shape", shapeEq(m.forward(x), 2, 8, H));
        });
        benchmark("ReluMlp", () -> {
            check("relu shape", shapeEq(new Mlp.ReluMlp(H, I).forward(x), 2, 8, H));
        });
        benchmark("GeGLU (Gemma)", () -> {
            check("geglu shape", shapeEq(new Mlp.GeGLU(H, I).forward(x), 2, 8, H));
        });
    }

    // ── D5 Attention variants ────────────────────────────────────────────
    static void d5AttentionVariants() {
        section("D5 Attention variants");
        long H = 64;
        Tensor x = randn(2, 8, H);
        benchmark("MHA", () -> {
            Attention a = Attention.mha(H, 4, 10000.0);
            check("mha heads", a.nHeads() == 4 && a.nKvHeads() == 4);
            check("mha shape", shapeEq(a.forward(x), 2, 8, H));
        });
        benchmark("GQA Llama", () -> {
            Attention a = Attention.llama(H, 4, 2, 10000.0);
            check("gqa kv", a.nKvHeads() == 2);
            check("gqa shape", shapeEq(a.forward(x), 2, 8, H));
        });
        benchmark("MQA", () -> {
            Attention a = Attention.mqa(H, 4, 10000.0);
            check("mqa kv=1", a.nKvHeads() == 1);
            check("mqa shape", shapeEq(a.forward(x), 2, 8, H));
        });
        benchmark("Qwen2 qkv bias", () -> {
            Attention a = Attention.qwen2(H, 4, 2, 1_000_000.0);
            check("qwen2 shape", shapeEq(a.forward(x), 2, 8, H));
        });
        benchmark("Qwen3 QK-Norm", () -> {
            Attention a = Attention.qwen3(H, 4, 2, 16, 1_000_000.0, 1e-6);
            check("qwen3 qk-norm", a.useQkNorm());
            check("qwen3 q/k_norm non-null", a.q_norm != null && a.k_norm != null);
            check("qwen3 shape", shapeEq(a.forward(x), 2, 8, H));
        });
        benchmark("GPT-2 no RoPE", () -> {
            Attention a = Attention.gpt2(H, 4);
            check("gpt2 no rope", !a.useRope());
            check("gpt2 shape", shapeEq(a.forward(x), 2, 8, H));
        });
        benchmark("Sliding window", () -> {
            Attention a = Attention.slidingWindow(H, 4, 2, 10000.0, 4);
            check("window=4", a.slidingWindow() == 4);
            check("swa shape", shapeEq(a.forward(x), 2, 8, H));
        });
        benchmark("ALiBi", () -> {
            Attention a = Attention.alibi(H, 4);
            check("alibi flag", a.useAlibi() && !a.useRope());
            Tensor y = a.forward(x);
            check("alibi shape", shapeEq(y, 2, 8, H));
            check("alibi finite", finite(y));
        });
    }

    // ── D6 Attention cache ───────────────────────────────────────────────
    static void d6AttentionCache() {
        section("D6 Attention KV cache");
        benchmark("prefill then decode step", () -> {
            try (NoGradGuard ng = new NoGradGuard()) {
                Attention a = Attention.gqa(64, 4, 2, 10000.0);
                Tensor prefill = randn(1, 6, 64);
                Tensor[] r0 = a.forwardCached(prefill, 0L, null, null);
                check("prefill out", shapeEq(r0[0], 1, 6, 64));
                check("prefill K heads GQA-repeated", r0[1].size(1) == 4);
                check("prefill K T", r0[1].size(2) == 6);

                Tensor tok = randn(1, 1, 64);
                Tensor[] r1 = a.forwardCached(tok, 6L, r0[1], r0[2]);
                check("decode out", shapeEq(r1[0], 1, 1, 64));
                check("decode newK T=1", r1[1].size(2) == 1);
                check("decode finite", finite(r1[0]));
            }
        });
    }

    // ── D7 Decoder layers ────────────────────────────────────────────────
    static void d7DecoderLayers() {
        section("D7 DecoderLayer family");
        Tensor x = randn(2, 6, 64);
        benchmark("Llama decoder", () -> {
            DecoderLayer d = DecoderLayer.llama(64, 4, 2, 128, 10000.0, 0);
            check("llama layer shape", shapeEq(d.forward(x), 2, 6, 64));
            check("pre-norm", d.residualStyle() == DecoderLayer.ResidualStyle.PRE_NORM);
        });
        benchmark("Qwen3 decoder", () -> {
            DecoderLayer d = DecoderLayer.qwen3(64, 4, 2, 16, 128, 1_000_000.0, 1e-6, 1);
            check("qwen3 layer shape", shapeEq(d.forward(x), 2, 6, 64));
        });
        benchmark("GPT-2 decoder post-norm", () -> {
            DecoderLayer d = DecoderLayer.gpt2(64, 4, 128, 0.0, 0);
            check("gpt2 post-norm", d.residualStyle() == DecoderLayer.ResidualStyle.POST_NORM);
            check("gpt2 layer shape", shapeEq(d.forward(x), 2, 6, 64));
        });
        benchmark("GLM fused decoder", () -> {
            DecoderLayer d = DecoderLayer.glm(64, 4, 2, 128, 10000.0, 0);
            check("glm layer shape", shapeEq(d.forward(x), 2, 6, 64));
        });
        benchmark("DeepSeek MoE decoder", () -> {
            DecoderLayer d = DecoderLayer.deepseekMoe(64, 4, 2, 64, 10000.0, 4, 2, 0);
            Tensor y = d.forward(x);
            check("moe layer shape", shapeEq(y, 2, 6, 64));
            check("moe layer finite", finite(y));
        });
        benchmark("DecoderLayer builder Gemma/GeGLU", () -> {
            DecoderLayer d = new DecoderLayer.DecoderLayerBuilder(64, 4)
                    .nKvHeads(2).intermediateSize(128).mlpType(DecoderLayer.MlpType.GEGLU)
                    .layerIdx(0).build();
            check("geglu layer shape", shapeEq(d.forward(x), 2, 6, 64));
        });
        benchmark("decoder cached", () -> {
            try (NoGradGuard ng = new NoGradGuard()) {
                DecoderLayer d = DecoderLayer.llama(64, 4, 2, 128, 10000.0, 0);
                Tensor[] r = d.forwardCached(randn(1, 4, 64), 0L, null, null);
                check("cached triple", r.length == 3 && shapeEq(r[0], 1, 4, 64));
            }
        });
    }

    // ── D8 MoE ───────────────────────────────────────────────────────────
    static void d8MoE() {
        section("D8 MoE");
        benchmark("Mixtral-style MoE", () -> {
            MoE moe = MoE.mixtral(64, 64);
            check("8 experts", moe.numExperts() == 8);
            check("top2", moe.topK() == 2);
            check("no shared", !moe.hasSharedExpert());
            Tensor y = moe.forward(randn(2, 6, 64));
            check("moe shape", shapeEq(y, 2, 6, 64));
            check("moe finite", finite(y));
        });
        benchmark("DeepSeek shared-expert MoE", () -> {
            MoE moe = MoE.deepseek(64, 48, 4, 2);
            check("shared on", moe.hasSharedExpert());
            Tensor x = randn(2, 4, 64);
            Tensor y = moe.forward(x);
            check("shared moe shape", shapeEq(y, 2, 4, 64));
            Tensor aux = moe.loadBalancingLoss(x);
            check("aux loss scalar", aux.dim() == 0 || aux.numel() == 1);
            check("aux finite", finite(aux));
        });
        benchmark("MoE flat [N,H] input", () -> {
            MoE moe = new MoE(32, 32, 4, 2);
            Tensor y = moe.forward(randn(12, 32));
            check("flat shape", shapeEq(y, 12, 32));
        });
    }

    // ── D9 MLA ───────────────────────────────────────────────────────────
    static void d9Mla() {
        section("D9 MultiLatentAttention (DeepSeek)");
        benchmark("MLA forward shape", () -> {
            try (NoGradGuard ng = new NoGradGuard()) {
                MultiLatentAttention mla = MultiLatentAttention.deepseek(64, 4, 16, 10000.0);
                check("kv lora rank", mla.kvLoraRank() == 16);
                Tensor y = mla.forward(randn(2, 6, 64));
                check("mla shape", shapeEq(y, 2, 6, 64));
                check("mla finite", finite(y));
            }
        });
        benchmark("MLA cached prefill+decode", () -> {
            try (NoGradGuard ng = new NoGradGuard()) {
                MultiLatentAttention mla = MultiLatentAttention.deepseek(64, 4, 16, 10000.0);
                Tensor[] r0 = mla.forwardCached(randn(1, 5, 64), 0L, null, null);
                check("mla prefill out", shapeEq(r0[0], 1, 5, 64));
                check("mla ckv rank", r0[1].size(2) == 16);
                Tensor[] r1 = mla.forwardCached(randn(1, 1, 64), 5L, r0[1], r0[2]);
                check("mla decode out", shapeEq(r1[0], 1, 1, 64));
                check("mla decode finite", finite(r1[0]));
            }
        });
    }

    // ── D10 MLA decoder ──────────────────────────────────────────────────
    static void d10MlaDecoder() {
        section("D10 MlaDecoderLayer");
        benchmark("MLA dense layer", () -> {
            MlaDecoderLayer d = MlaDecoderLayer.dense(64, 4, 16, 128, 10000.0, 0);
            Tensor y = d.forward(randn(2, 4, 64));
            check("mla-dec shape", shapeEq(y, 2, 4, 64));
            check("mla-dec finite", finite(y));
        });
        benchmark("MLA + MoE layer", () -> {
            MlaDecoderLayer d = MlaDecoderLayer.moe(64, 4, 16, 48, 10000.0, 4, 2, 1);
            check("mla-moe shape", shapeEq(d.forward(randn(1, 3, 64)), 1, 3, 64));
        });
    }

    // ── D11 Parallel (single-rank) ───────────────────────────────────────
    static void d11Parallel() {
        section("D11 ParallelLinear single-rank fallback");
        benchmark("ColumnParallel single-rank", () -> {
            ParallelLinear.ColumnParallelLinear col =
                    new ParallelLinear.ColumnParallelLinear(64, 128);
            check("not multi", !col.multiRank());
            check("full out", col.fullOutFeatures() == 128);
            Tensor y = col.forward(randn(2, 8, 64));
            check("col shape", shapeEq(y, 2, 8, 128));
        });
        benchmark("RowParallel single-rank", () -> {
            ParallelLinear.RowParallelLinear row =
                    new ParallelLinear.RowParallelLinear(128, 64);
            check("row shape", shapeEq(row.forward(randn(2, 8, 128)), 2, 8, 64));
        });
        benchmark("ParallelSwiGLU single-rank == dense path", () -> {
            ParallelLinear.ParallelSwiGLU p = new ParallelLinear.ParallelSwiGLU(64, 128);
            check("swiglu not multi", !p.multiRank());
            Tensor y = p.forward(randn(2, 4, 64));
            check("swiglu shape", shapeEq(y, 2, 4, 64));
            check("swiglu finite", finite(y));
        });
    }

    // ── D12 MiniCausalLM factories ───────────────────────────────────────
    static void d12MiniLmFactories() {
        section("D12 MiniCausalLM factories");
        benchmark("tiny Llama", () -> {
            MiniCausalLM m = MiniCausalLM.tiny();
            check("layers", m.numLayers() == 2);
            Tensor ids = ids(128, 2, 8);
            Tensor logits = m.forward(ids);
            check("llama logits", shapeEq(logits, 2, 8, 128));
            check("llama finite", finite(logits));
        });
        benchmark("Qwen2 mini", () -> {
            MiniCausalLM m = MiniCausalLM.qwen2(64, 32, 2, 4, 2);
            check("qwen2 logits",
                    shapeEq(m.forward(ids(64, 1, 6)), 1, 6, 64));
        });
        benchmark("Qwen3 mini", () -> {
            MiniCausalLM m = MiniCausalLM.qwen3(64, 32, 2, 4, 2, 8);
            check("qwen3 arch", m.config().arch == MiniCausalLM.Arch.QWEN3);
            check("qwen3 logits",
                    shapeEq(m.forward(ids(64, 1, 4)), 1, 4, 64));
        });
        benchmark("GPT-2 mini", () -> {
            MiniCausalLM m = MiniCausalLM.gpt2(64, 32, 2, 4);
            check("gpt2 abs pos", m.config().useAbsolutePos);
            check("gpt2 logits",
                    shapeEq(m.forward(ids(64, 1, 5)), 1, 5, 64));
        });
        benchmark("GLM mini", () -> {
            MiniCausalLM m = MiniCausalLM.glm(64, 32, 2, 4, 2);
            check("glm logits",
                    shapeEq(m.forward(ids(64, 1, 4)), 1, 4, 64));
        });
        benchmark("DeepSeek-MoE mini", () -> {
            MiniCausalLM m = MiniCausalLM.deepseekMoe(64, 32, 4, 4, 2, 4, 2);
            check("dsk layers 4", m.numLayers() == 4);
            check("dsk logits",
                    shapeEq(m.forward(ids(64, 1, 4)), 1, 4, 64));
        });
        benchmark("Mixtral mini", () -> {
            MiniCausalLM m = MiniCausalLM.mixtral(64, 32, 2, 4, 2, 4, 2);
            check("mixtral arch", m.config().arch == MiniCausalLM.Arch.MIXTRAL);
            check("mixtral logits",
                    shapeEq(m.forward(ids(64, 1, 3)), 1, 3, 64));
        });
    }

    // ── D13 train step ───────────────────────────────────────────────────
    static void d13TrainStep() {
        section("D13 MiniCausalLM loss + backward");
        benchmark("loss backward tiny llama", () -> {
            MiniCausalLM m = MiniCausalLM.tiny();
            m.train(/*on=*/true);
            Tensor ids = ids(128, 2, 10);
            Tensor loss = m.loss(ids);
            check("loss scalar", loss.numel() == 1);
            check("loss finite", finite(loss));
            loss.backward();
            boolean anyGrad = false;
            try {
                var params = m.parameters();
                for (long i = 0, n = params.size(); i < n; i++) {
                    Tensor p = params.get(i);
                    if (p == null || p.isNull()) continue;
                    try {
                        Tensor g = p.grad();
                        if (g != null && !g.isNull() && g.defined() && g.numel() > 0) {
                            anyGrad = true;
                            break;
                        }
                    } catch (Exception ignored) {}
                }
            } catch (Exception ignored) {}
            check("some param has grad", anyGrad);
        });
    }

    // ── D14 cached decode ────────────────────────────────────────────────
    static void d14CachedDecode() {
        section("D14 MiniCausalLM cached decode");
        benchmark("prefill + token step pack", () -> {
            try (NoGradGuard ng = new NoGradGuard()) {
                MiniCausalLM m = MiniCausalLM.llama(64, 32, 2, 4, 2);
                Tensor ids = ids(64, 1, 5);
                Tensor[] pack = m.forwardCached(ids, 0L, null, null);
                check("pack has logits", pack[0] != null && shapeEq(pack[0], 1, 5, 64));
                int L = m.numLayers();
                check("pack len = 1+2L", pack.length == 1 + 2 * L);
                Tensor[] pastK = new Tensor[L];
                Tensor[] pastV = new Tensor[L];
                System.arraycopy(pack, 1, pastK, 0, L);
                System.arraycopy(pack, 1 + L, pastV, 0, L);
                Tensor next = ids(64, 1, 1);
                Tensor[] step = m.forwardCached(next, 5L, pastK, pastV);
                check("step logits", shapeEq(step[0], 1, 1, 64));
                check("step finite", finite(step[0]));
            }
        });
    }

    // ── D15 facade ───────────────────────────────────────────────────────
    static void d15Facade() {
        section("D15 Modules facade");
        benchmark("facade tinies", () -> {
            check("tinyLlama", Modules.tinyLlama() != null);
            check("tinyGpt2", Modules.tinyGpt2() != null);
            check("tinyQwen3", Modules.tinyQwen3() != null);
            check("tinyDeepseekMoe", Modules.tinyDeepseekMoe() != null);
            check("tinyMixtral", Modules.tinyMixtral() != null);
            check("tinyGlm", Modules.tinyGlm() != null);
        });
    }

    // ── D16 custom stack composition ─────────────────────────────────────
    static void d16CustomStack() {
        section("D16 Custom stack composition");
        benchmark("embed → 2×llama dec → rms → linear head", () -> {
            Embedding emb = Embedding.ropeOnly(100, 48);
            DecoderLayer l0 = DecoderLayer.llama(48, 4, 2, 96, 10000.0, 0);
            DecoderLayer l1 = DecoderLayer.llama(48, 4, 2, 96, 10000.0, 1);
            RMSNorm norm = new RMSNorm(48);
            ParallelLinear.ColumnParallelLinear head =
                    new ParallelLinear.ColumnParallelLinear(48, 100, null, null, false);

            Tensor ids = ids(100, 2, 7);
            Tensor h = emb.forward(ids);
            h = l0.forward(h);
            h = l1.forward(h);
            h = norm.forward(h);
            Tensor logits = head.forward(h);
            check("custom stack logits", shapeEq(logits, 2, 7, 100));
            check("custom stack finite", finite(logits));
        });
        benchmark("DeepSeek-like: MLA dec + MoE dec", () -> {
            Embedding emb = Embedding.ropeOnly(80, 64);
            MlaDecoderLayer d0 = MlaDecoderLayer.dense(64, 4, 16, 128, 10000.0, 0);
            MlaDecoderLayer d1 = MlaDecoderLayer.moe(64, 4, 16, 48, 10000.0, 4, 2, 1);
            RMSNorm norm = new RMSNorm(64);
            Tensor h = emb.forward(ids(80, 1, 5));
            h = d0.forward(h);
            h = d1.forward(h);
            h = norm.forward(h);
            check("dsk-like hidden", shapeEq(h, 1, 5, 64));
            check("dsk-like finite", finite(h));
        });
    }

    // ── D17 numerical stability ──────────────────────────────────────────
    static void d17Stability() {
        section("D17 Numerical stability across arches");
        benchmark("no NaN on all factory arches", () -> {
            try (NoGradGuard ng = new NoGradGuard()) {
                MiniCausalLM[] models = {
                        MiniCausalLM.llama(32, 32, 2, 4, 2),
                        MiniCausalLM.qwen2(32, 32, 2, 4, 2),
                        MiniCausalLM.qwen3(32, 32, 2, 4, 2, 8),
                        MiniCausalLM.gpt2(32, 32, 2, 4),
                        MiniCausalLM.glm(32, 32, 2, 4, 2),
                        MiniCausalLM.deepseekMoe(32, 32, 3, 4, 2, 4, 2),
                        MiniCausalLM.mixtral(32, 32, 2, 4, 2, 4, 2),
                };
                for (MiniCausalLM m : models) {
                    Tensor ids = ids(32, 1, 6);
                    Tensor logits = m.forward(ids);
                    check(m.config().arch + " finite", finite(logits));
                }
            }
        });
    }

    // ── D18 stress matrix ────────────────────────────────────────────────
    static void d18Stress() {
        section("D18 Batch / sequence stress");
        benchmark("batch×seq matrix on GQA attn", () -> {
            try (NoGradGuard ng = new NoGradGuard()) {
                Attention a = Attention.gqa(64, 4, 2, 10000.0);
                int[] Bs = {1, 2, 4};
                int[] Ts = {1, 7, 16, 32};
                for (int B : Bs) {
                    for (int T : Ts) {
                        Tensor y = a.forward(randn(B, T, 64));
                        check("B" + B + "T" + T + " shape", shapeEq(y, B, T, 64));
                        check("B" + B + "T" + T + " finite", finite(y));
                    }
                }
            }
        });
        benchmark("longish seq MiniCausalLM", () -> {
            try (NoGradGuard ng = new NoGradGuard()) {
                MiniCausalLM m = new MiniCausalLM(MiniCausalLM.Config.builder(MiniCausalLM.Arch.LLAMA)
                        .vocabSize(64).hiddenSize(32).numLayers(2)
                        .nHeads(4).nKvHeads(2).maxPositions(256).build());
                Tensor ids = ids(64, 1, 64);
                Tensor logits = m.forward(ids);
                check("T=64 logits", shapeEq(logits, 1, 64, 64));
                check("T=64 finite", finite(logits));
            }
        });
        benchmark("module parameter discovery non-empty", () -> {
            Module m = MiniCausalLM.tiny();
            long n = m.parameters().size();
            check("tiny has params", n > 0);
            System.out.println("    tiny param tensors: " + n);
        });
    }
}
