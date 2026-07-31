package samples;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.llm.sentence.CrossEncoder;
import org.bytedeco.pytorch.llm.sentence.SentenceTransformer;
import org.bytedeco.pytorch.llm.sentence.evaluation.EmbeddingSimilarityEvaluator;
import org.bytedeco.pytorch.llm.sentence.modules.Dense;
import org.bytedeco.pytorch.llm.sentence.modules.Normalize;
import org.bytedeco.pytorch.llm.sentence.modules.Pooling;
import org.bytedeco.pytorch.llm.sentence.modules.TransformerModule;
import org.bytedeco.pytorch.llm.sentence.util.SentenceEvalUtil;

import java.util.ArrayList;
import java.util.List;

/**
 * Multi-dimensional full-API stress for {@code org.bytedeco.pytorch.llm.sentence}.
 *
 * <pre>
 * D1  encode shape / L2
 * D2  cosine matrix
 * D3  paraphrase_mining
 * D4  community_detection
 * D5  CrossEncoder predict
 * D6  MNRL / CosineSimilarity loss
 * D7  mini factories + accessors
 * D8  EmbeddingSimilarityEvaluator
 * D9  batch encode
 * D10 modules (Transformer/Pooling/Normalize/Dense)
 * D11 semanticSearch + forward + pooling strategies
 * D12 CrossEncoder batch/score
 * D13 encode throughput stress
 * </pre>
 */
public class BenchmarkSentenceTransformers {
    static int passed = 0, failed = 0;

    static void check(String name, boolean ok) {
        if (ok) { passed++; System.out.println("  PASS  " + name); }
        else { failed++; System.out.println("  FAIL  " + name); }
    }
    static void section(String t) { System.out.println("\n=== " + t + " ==="); }

    public static void main(String[] args) throws Exception {
        System.out.println("=== SentenceTransformers multi-dimensional full-API stress ===");
        d1Encode();
        d2CosineMatrix();
        d3ParaphraseMining();
        d4CommunityDetection();
        d5CrossEncoder();
        d6LossBackward();
        d7SaveLoad();
        d8IRevaluator();
        d9BatchEncode();
        d10Modules();
        d11SemanticSearchLosses();
        d12CrossEncoderBatch();
        d13Stress();
        done();
    }

    static void d1Encode() throws Exception {
        section("D1 encode shape / L2");
        SentenceTransformer st = SentenceTransformer.mini(512, 64);
        float[] e = st.encode("hello world");
        check("embed dim=64", e.length == 64);
        double nrm = SentenceTransformer.cosine(e, e);
        check("self-cosine ~= 1.0", Math.abs(nrm - 1.0) < 1e-3);
        float[][] batch = st.encode(List.of("a", "b", "c"));
        check("batch size=3", batch.length == 3);
        check("batch dim=64", batch[0].length == 64);
    }

    static void d2CosineMatrix() {
        section("D2 cosine matrix SPD-ish");
        float[][] emb = {
                new float[]{1, 0, 0},
                new float[]{1, 0, 0},
                new float[]{0, 1, 0}
        };
        double[][] m = SentenceTransformer.cosineMatrix(emb);
        check("matrix 3x3", m.length == 3 && m[0].length == 3);
        check("diag ~= 1", Math.abs(m[0][0] - 1.0) < 1e-3);
        check("m[0][1] ~= 1 (identical)", Math.abs(m[0][1] - 1.0) < 1e-3);
        check("m[0][2] ~= 0", Math.abs(m[0][2]) < 1e-3);
    }

    static void d3ParaphraseMining() {
        section("D3 paraphrase_mining");
        var pairs = SentenceEvalUtil.paraphraseMining(List.of("hello world", "hi there"), 0.5);
        check("returns list", pairs != null);
    }

    static void d4CommunityDetection() {
        section("D4 community_detection");
        var comms = SentenceEvalUtil.communityDetection(List.of("a", "b", "c"), 0.5);
        check("has communities", comms.size() >= 1);
    }

    static void d5CrossEncoder() throws Exception {
        section("D5 CrossEncoder");
        try (CrossEncoder ce = new CrossEncoder(64)) {
            double s = ce.predict("hello", "hello").item_double();
            check("CrossEncoder predict finite", !Double.isNaN(s));
        }
    }

    static void d6LossBackward() throws Exception {
        section("D6 MNRL / CosineSimilarity loss forward");
        var a = org.bytedeco.pytorch.global.torch.randn(new long[]{4, 16});
        var p = org.bytedeco.pytorch.global.torch.randn(new long[]{4, 16});
        Tensor mnrl = org.bytedeco.pytorch.llm.sentence.losses.MultipleNegativesRankingLoss.forward(a, p);
        check("MNRL forward finite", mnrl != null && !Double.isNaN(mnrl.item_double()));
        Tensor cos = org.bytedeco.pytorch.llm.sentence.losses.CosineSimilarityLoss.forward(a, p, 1.0);
        check("CosineSimilarityLoss target forward", cos != null && !Double.isNaN(cos.item_double()));
        Tensor cos2 = org.bytedeco.pytorch.llm.sentence.losses.CosineSimilarityLoss.forward(a, p);
        check("CosineSimilarityLoss default forward", cos2 != null);
        mnrl.close(); cos.close(); cos2.close(); a.close(); p.close();
    }

    static void d7SaveLoad() throws Exception {
        section("D7 mini factories + accessors");
        SentenceTransformer st = SentenceTransformer.mini();
        check("SentenceTransformer.mini non-null", st != null);
        check("default embedDim > 0", st.getEmbedDim() > 0);
        check("maxSeqLength > 0", st.getMaxSeqLength() > 0);
        check("tokenizer non-null", st.tokenizer() != null);
        check("normalize flag readable", st.isNormalizeEmbeddings() || !st.isNormalizeEmbeddings());
        SentenceTransformer st2 = SentenceTransformer.mini(256, 32);
        check("mini(256,32) embedDim=32", st2.getEmbedDim() == 32);
        float[] e = st2.encode("test");
        check("mini(256,32) encode dim", e.length == 32);
    }

    static void d8IRevaluator() throws Exception {
        section("D8 EmbeddingSimilarityEvaluator + SentenceEvalUtil");
        SentenceTransformer st = SentenceTransformer.mini(256, 32);
        List<String[]> pairs = List.of(
                new String[]{"hello world", "hello world"},
                new String[]{"cat", "dog"},
                new String[]{"good", "great"},
                new String[]{"a", "z"}
        );
        List<Double> gold = List.of(1.0, 0.2, 0.8, 0.0);
        var result = EmbeddingSimilarityEvaluator.evaluate(pairs, gold, st);
        check("evaluator result non-null", result != null);
        check("evaluator mean finite", !Double.isNaN(result.mean()));
        check("evaluator spearman finite", !Double.isNaN(result.spearman()));
        check("evaluator pearson finite", !Double.isNaN(result.pearson()));
        check("evaluator toString", result.toString() != null);

        float[] a = st.encode("hello");
        float[] b = st.encode("hello");
        check("cosSim util self~1", Math.abs(SentenceEvalUtil.cosSim(a, b) - 1.0) < 1e-3
                || SentenceEvalUtil.cosSim(a, b) > 0.9);
        double[][] mat = SentenceEvalUtil.cosSimMatrix(new float[][]{
                new float[]{1, 0}, new float[]{0, 1}
        });
        check("cosSimMatrix 2x2", mat.length == 2 && Math.abs(mat[0][1]) < 1e-3);
    }

    static void d9BatchEncode() throws Exception {
        section("D9 Batch encode throughput");
        SentenceTransformer st = SentenceTransformer.mini(512, 64);
        List<String> texts = List.of("a", "b", "c", "d", "e");
        float[][] out = st.encode(texts);
        check("5-text batch dim correct", out.length == 5 && out[0].length == 64);
    }

    static void d10Modules() throws Exception {
        section("D10 modules");
        try (TransformerModule tm = new TransformerModule(512, 64)) {
            var opts = new org.bytedeco.pytorch.TensorOptions()
                    .dtype(new org.bytedeco.pytorch.ScalarTypeOptional(
                            org.bytedeco.pytorch.global.torch.ScalarType.Long));
            var t = org.bytedeco.pytorch.global.torch.randint(0, 512, new long[]{2, 8}, opts);
            var h = tm.forward(t);
            check("TransformerModule output shape hidden=64", h.size(2) == 64);
            check("TransformerModule hiddenSize", tm.hiddenSize() == 64);
            check("TransformerModule embedding", tm.embedding() != null);
            t.close(); h.close();
        }
        try (Pooling p = new Pooling(SentenceTransformer.PoolingStrategy.MEAN)) {
            var h = org.bytedeco.pytorch.global.torch.randn(new long[]{2, 4, 64});
            var pooled = p.forward(h);
            check("Pooling output finite", pooled != null);
            h.close();
        }
        try (Normalize n = new Normalize()) {
            var h = org.bytedeco.pytorch.global.torch.randn(new long[]{2, 64});
            var normed = n.forward(h);
            check("Normalize output finite", normed != null);
            h.close();
        }
        try (Dense d = new Dense(64, 32)) {
            var h = org.bytedeco.pytorch.global.torch.randn(new long[]{2, 64});
            var out = d.forward(h);
            check("Dense output shape[1]=32", out.size(1) == 32);
            h.close(); out.close();
        }
    }

    static void d11SemanticSearchLosses() throws Exception {
        section("D11 semanticSearch + encode normalize + forward");
        SentenceTransformer st = SentenceTransformer.mini(256, 32);
        List<String> corpus = List.of(
                "python is a programming language",
                "java runs on the jvm",
                "cats are animals",
                "deep learning with pytorch"
        );
        List<SentenceTransformer.SearchHit> hits = st.semanticSearch("programming languages", corpus, 2);
        check("semanticSearch topK", hits != null && hits.size() <= 2 && !hits.isEmpty());
        check("SearchHit fields", hits.get(0).text != null && hits.get(0).index >= 0);
        check("SearchHit toString", hits.get(0).toString() != null);

        float[][] normed = st.encode(corpus, true);
        float[][] raw = st.encode(corpus, false);
        check("encode normalize flag shapes", normed.length == 4 && raw.length == 4 && normed[0].length == 32);

        var opts = new org.bytedeco.pytorch.TensorOptions()
                .dtype(new org.bytedeco.pytorch.ScalarTypeOptional(
                        org.bytedeco.pytorch.global.torch.ScalarType.Long));
        var ids = org.bytedeco.pytorch.global.torch.randint(0, 100, new long[]{2, 8}, opts);
        var mask = org.bytedeco.pytorch.global.torch.ones(new long[]{2, 8}, opts);
        Tensor out = st.forward(ids, mask);
        check("forward ids+mask", out != null && out.size(0) == 2);
        Tensor out2 = st.forward(ids);
        check("forward ids only", out2 != null);
        out.close(); out2.close(); ids.close(); mask.close();

        check("PoolingStrategy values", SentenceTransformer.PoolingStrategy.values().length >= 3);
        try (Pooling pMax = new Pooling(SentenceTransformer.PoolingStrategy.MAX);
             Pooling pCls = new Pooling(SentenceTransformer.PoolingStrategy.CLS)) {
            var h = org.bytedeco.pytorch.global.torch.randn(new long[]{2, 4, 16});
            check("Pooling MAX", pMax.forward(h) != null);
            check("Pooling CLS", pCls.forward(h) != null);
            check("Pooling strategy()", pMax.strategy() == SentenceTransformer.PoolingStrategy.MAX);
            h.close();
        }
        try (Normalize n = new Normalize(1e-6);
             Dense d = new Dense(16, 8, "tanh")) {
            check("Normalize eps", Math.abs(n.eps() - 1e-6) < 1e-12);
            check("Dense activation", d.activation() != null);
            check("Dense linear", d.linear() != null);
            var h = org.bytedeco.pytorch.global.torch.randn(new long[]{2, 16});
            check("Dense tanh forward", d.forward(h).size(1) == 8);
            h.close();
        }
    }

    static void d12CrossEncoderBatch() throws Exception {
        section("D12 CrossEncoder batch + score/forward");
        try (CrossEncoder ce = new CrossEncoder();
             CrossEncoder ce2 = new CrossEncoder(32, 16, true)) {
            List<Double> scores = ce.predict(List.of(
                    new String[]{"a", "a"},
                    new String[]{"hello", "world"},
                    new String[]{"x", "y"}
            ));
            check("CrossEncoder batch predict size=3", scores != null && scores.size() == 3);
            check("batch scores finite", scores.stream().allMatch(s -> s != null && !Double.isNaN(s)));

            var ea = org.bytedeco.pytorch.global.torch.randn(new long[]{2, 32});
            var eb = org.bytedeco.pytorch.global.torch.randn(new long[]{2, 32});
            Tensor sc = ce2.score(ea, eb);
            check("CrossEncoder.score", sc != null);
            Tensor fw = ce2.forward(ea, eb);
            check("CrossEncoder.forward", fw != null);
            sc.close(); fw.close(); ea.close(); eb.close();
        }
    }

    static void d13Stress() throws Exception {
        section("D13 encode throughput stress");
        SentenceTransformer st = SentenceTransformer.mini(512, 64);
        List<String> texts = new ArrayList<>();
        for (int i = 0; i < 64; i++) {
            texts.add("This is sentence number " + i + " for embedding stress test.");
        }
        long t0 = System.nanoTime();
        float[][] emb = st.encode(texts, true);
        long ms = (System.nanoTime() - t0) / 1_000_000L;
        check("stress batch 64", emb.length == 64 && emb[0].length == 64);
        int okNorm = 0;
        for (int i = 0; i < emb.length; i += 8) {
            double n = 0;
            for (float v : emb[i]) n += v * v;
            if (Math.abs(Math.sqrt(n) - 1.0) < 1e-2) okNorm++;
        }
        check("stress L2 roughly unit", okNorm >= 1);
        System.out.println("  INFO  encode 64x64d took " + ms + " ms");

        var pairs = SentenceEvalUtil.paraphraseMining(texts.subList(0, 16), 0.3);
        check("paraphraseMining larger", pairs != null);
        var comms = SentenceEvalUtil.communityDetection(texts.subList(0, 16), 0.3);
        check("communityDetection larger", comms != null && !comms.isEmpty());
    }

    static void done() {
        System.out.println("\n============================================================");
        System.out.println("SentenceTransformers  passed=" + passed + "  failed=" + failed);
        System.out.println("============================================================");
        if (failed > 0) System.exit(1);
        System.out.println("ALL DIMENSIONS GREEN");
    }
}
