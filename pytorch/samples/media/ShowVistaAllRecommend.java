package media;

import static org.bytedeco.pytorch.global.torch.randn;
import static org.bytedeco.pytorch.global.torch.randint;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.ModulePrinter;
import org.bytedeco.pytorch.recommend.basic.features.DenseFeature;
import org.bytedeco.pytorch.recommend.basic.features.SequenceFeature;
import org.bytedeco.pytorch.recommend.basic.features.SparseFeature;
import org.bytedeco.pytorch.recommend.basic.features.Feature;
import org.bytedeco.pytorch.recommend.basic.features.Features;
import org.bytedeco.pytorch.recommend.basic.layers.EmbeddingLayer;
import org.bytedeco.pytorch.recommend.models.generative.HLLM;
import org.bytedeco.pytorch.recommend.models.generative.HLLMTransformerBlock;
import org.bytedeco.pytorch.recommend.models.generative.HSTU;
import org.bytedeco.pytorch.recommend.models.generative.LLM4Rec;
import org.bytedeco.pytorch.recommend.models.generative.LLM4RecEncoderLayer;
import org.bytedeco.pytorch.recommend.models.generative.OneRec;
import org.bytedeco.pytorch.recommend.models.generative.OneRecV2;
import org.bytedeco.pytorch.recommend.models.generative.OpenOneRec;
import org.bytedeco.pytorch.recommend.models.generative.RQVAE;
import org.bytedeco.pytorch.recommend.models.generative.ResidualVectorQuantizer;
import org.bytedeco.pytorch.recommend.models.generative.TIGER;
import org.bytedeco.pytorch.recommend.models.generative.VectorQuantizer;
import org.bytedeco.pytorch.recommend.models.multi_task.AITM;
import org.bytedeco.pytorch.recommend.models.multi_task.AttentionLayer;
import org.bytedeco.pytorch.recommend.models.multi_task.CGC;
import org.bytedeco.pytorch.recommend.models.multi_task.ESMM;
import org.bytedeco.pytorch.recommend.models.multi_task.MMOE;
import org.bytedeco.pytorch.recommend.models.multi_task.MetaEmbedding;
import org.bytedeco.pytorch.recommend.models.multi_task.MetaHeac;
import org.bytedeco.pytorch.recommend.models.multi_task.MetaLinear;
import org.bytedeco.pytorch.recommend.models.multi_task.OMoE;
import org.bytedeco.pytorch.recommend.models.multi_task.PLE;
import org.bytedeco.pytorch.recommend.models.multi_task.SharedBottom;
import org.bytedeco.pytorch.recommend.models.multi_task.SingleTaskModel;
import org.bytedeco.pytorch.plot.vista.ExportFormat;
import org.bytedeco.pytorch.plot.vista.TraceGraph;
import org.bytedeco.pytorch.plot.vista.Vista;
import org.bytedeco.pytorch.plot.vista.VistaOptions;

/**
 * Batch-generate vista HTML for <b>all</b> multi_task + generative recommend models.
 *
 * <p>Models are <b>never modified</b>. Use in-page controls:
 * LR/RL/TB/BT · ＋/－ zoom · Cute/Dark/Office themes · pin-drag frames · Export SVG/PNG/JPEG/PDF.
 *
 * <pre>
 *   java ... media.ShowVistaAllRecommend [--browser] [--only=SharedBottom,OneRec,HLLM]
 * </pre>
 */
public class ShowVistaAllRecommend {

    @FunctionalInterface
    interface ModelFactory {
        Object[] create() throws Exception; // [Module, inputs]
    }

    public static void main(String[] args) throws Exception {
        boolean openBrowser = false;
        String only = null;
        for (String a : args) {
            if ("--browser".equals(a)) openBrowser = true;
            else if ("--no-browser".equals(a)) openBrowser = false;
            else if (a.startsWith("--only=")) only = a.substring("--only=".length());
        }

        Path root = Paths.get("samples/out/vista-all");
        Path mtDir = root.resolve("multi_task");
        Path genDir = root.resolve("generative");
        Files.createDirectories(mtDir);
        Files.createDirectories(genDir);

        List<Feature> feats = Arrays.asList(
                Features.sparse("user_id", 500, 8),
                Features.sparse("item_id", 1000, 8),
                Features.sparse("cate_id", 50, 8));
        // Mixed feature catalog demo (sparse + dense + sequence) for EmbeddingLayer-only showcase
        List<Feature> mixedFeats = Arrays.asList(
                Features.sparse("user_id", 500, 8),
                Features.sparse("item_id", 1000, 8),
                Features.dense("u_age", 4),
                Features.sequence("click_seq", 200, 8, "mean"));
        List<Feature> userFeats = Arrays.asList(
                Features.sparse("user_id", 500, 8),
                Features.sparse("u_age", 20, 4));
        List<Feature> itemFeats = Arrays.asList(
                Features.sparse("item_id", 1000, 8),
                Features.sparse("cate_id", 50, 4));
        List<String> twoTasks = Arrays.asList("classification", "classification");
        // MetaHeac registers critic_<taskName>_i — task names must be unique
        List<String> metaTasks = Arrays.asList("ctr", "cvr");
        Map<String, Object> smallBottom = Map.of("dims", List.of(32L), "activation", "relu", "dropout", 0f);
        List<Map<String, Object>> twoTowers = Arrays.asList(
                Map.of("dims", List.of(16L), "activation", "relu", "dropout", 0f),
                Map.of("dims", List.of(16L), "activation", "relu", "dropout", 0f));

        Map<String, ModelFactory> multiTask = new LinkedHashMap<>();
        multiTask.put("AttentionLayer", () -> {
            AttentionLayer m = new AttentionLayer(16, "cpu");
            m.eval();
            return new Object[]{m, randn(2, 2, 16)};
        });
        multiTask.put("MetaLinear", () -> {
            MetaLinear m = new MetaLinear(12, 6, "cpu");
            m.eval();
            return new Object[]{m, randn(2, 12)};
        });
        multiTask.put("MetaEmbedding", () -> {
            MetaEmbedding m = new MetaEmbedding(100, 8, "cpu");
            m.eval();
            return new Object[]{m, randint(0, 100, new long[]{2, 4})};
        });
        // Feature-type showcase: sparse + dense + sequence on EmbeddingLayer
        multiTask.put("EmbeddingLayer", () -> {
            EmbeddingLayer m =
                    new EmbeddingLayer(mixedFeats, 8, "cpu");
            m.eval();
            Map<String, Tensor> sparse = new LinkedHashMap<>();
            sparse.put("user_id", randint(0, 500, new long[]{2}));
            sparse.put("item_id", randint(0, 1000, new long[]{2}));
            sparse.put("u_age", randn(2, 4)); // dense float (catalog still marks dense)
            Map<String, Tensor> seq = new LinkedHashMap<>();
            seq.put("click_seq", randint(0, 200, new long[]{2, 8}));
            // forward(Map sparse, Map sequence, boolean squeeze)
            return new Object[]{m, new Object[]{sparse, seq, Boolean.TRUE}};
        });
        multiTask.put("SharedBottom", () -> {
            SharedBottom m = new SharedBottom(feats, twoTasks, smallBottom, List.of(), "cpu");
            m.eval();
            return new Object[]{m, featureBatch(feats, 2)};
        });
        multiTask.put("ESMM", () -> {
            ESMM m = new ESMM(userFeats, itemFeats);
            m.eval();
            Map<String, Tensor> batch = new LinkedHashMap<>();
            batch.putAll(featureBatch(userFeats, 2));
            batch.putAll(featureBatch(itemFeats, 2));
            return new Object[]{m, batch};
        });
        multiTask.put("MMOE", () -> {
            MMOE m = new MMOE(feats, twoTasks, 2, smallBottom, List.of(), "cpu");
            m.eval();
            return new Object[]{m, featureBatch(feats, 2)};
        });
        multiTask.put("OMoE", () -> {
            OMoE m = new OMoE(new ArrayList<>(feats), twoTasks);
            m.eval();
            return new Object[]{m, featureBatch(feats, 2)};
        });
        multiTask.put("PLE", () -> {
            PLE m = new PLE(new ArrayList<>(feats), twoTasks);
            m.eval();
            return new Object[]{m, featureBatch(feats, 2)};
        });
        multiTask.put("AITM", () -> {
            // towerParamsList.size MUST equal nTask
            AITM m = new AITM(new ArrayList<>(feats), 2, smallBottom, twoTowers, "cpu");
            m.eval();
            return new Object[]{m, featureBatch(feats, 2)};
        });
        multiTask.put("SingleTaskModel", () -> {
            SingleTaskModel m = new SingleTaskModel(feats, twoTasks, 8,
                    new long[]{32L}, new long[]{16L}, 0f, "cpu");
            m.eval();
            return new Object[]{m, featureBatch(feats, 2)};
        });
        multiTask.put("CGC", () -> {
            CGC m = new CGC(0, 1, 2, 1, 1, 24,
                    Map.of("dims", List.of(16L)), "cpu");
            m.eval();
            return new Object[]{m, Arrays.asList(randn(2, 24), randn(2, 24))};
        });
        multiTask.put("MetaHeac", () -> {
            // forward(Map, Tensor taskIdx) — multi-arg Object[]
            // task names must be unique (registers critic_<name>_i)
            MetaHeac m = new MetaHeac(new ArrayList<>(feats), metaTasks,
                    8, new long[]{32L}, new long[]{16L}, 1, 1, 0f, "cpu");
            m.eval();
            Map<String, Tensor> batch = featureBatch(feats, 2);
            Tensor taskIdx = randint(0, metaTasks.size(), new long[]{2});
            return new Object[]{m, new Object[]{batch, taskIdx}};
        });

        Map<String, ModelFactory> generative = new LinkedHashMap<>();
        generative.put("OneRec", () -> {
            OneRec m = new OneRec(2, 16, 64, 2, 2, 32, 0.0, true, "cpu");
            m.eval();
            int vocab = 3 + 2 * 16;
            return new Object[]{m, randint(0, Math.max(4, vocab), new long[]{2, 8})};
        });
        generative.put("OneRecV2", () -> {
            OneRecV2 m = new OneRecV2(2, 16);
            m.eval();
            return new Object[]{m, randint(0, 40, new long[]{2, 8})};
        });
        generative.put("OpenOneRec", () -> {
            OpenOneRec m = new OpenOneRec(2, 16);
            m.eval();
            return new Object[]{m, randint(0, 40, new long[]{2, 8})};
        });
        generative.put("TIGER", () -> {
            Tensor itemEmb = randn(100, 8);
            TIGER m = new TIGER(itemEmb, 8, 32, 1, 0f, "cpu");
            m.eval();
            return new Object[]{m, randint(0, 100, new long[]{2, 5})};
        });
        generative.put("HLLM", () -> {
            // small: dModel=32, nHeads=2, nLayers=1, maxSeq=16
            long vocab = 50;
            Tensor itemEmb = randn(vocab, 32);
            HLLM m = new HLLM(itemEmb, vocab, 32, 2, 1, 16, 0f,
                    false, false, 32, "sqrt", 0.07f, "cpu");
            m.eval();
            return new Object[]{m, randint(0, vocab, new long[]{2, 8})};
        });
        generative.put("HSTU", () -> {
            // vocab, dModel, nHeads, nLayers, dqk, dv, maxSeq, …
            HSTU m = new HSTU(64, 32, 2, 1, 16, 16, 16, 0f,
                    false, 32, "sqrt", 1.0f, "minutes", true, "none", 1.0f,
                    true, false, 1e-6f, "cpu");
            m.eval();
            return new Object[]{m, randint(0, 64, new long[]{2, 8})};
        });
        generative.put("LLM4Rec", () -> {
            // vocab, embedDim, numHeads, numLayers, maxSeq, mlpDims, dropout, usePos, device
            // positionEmbedding vocab = maxSeqLen+1 = 17 → positions must be in [0,16]
            LLM4Rec m = new LLM4Rec(64, 32, 2, 1, 16, new long[]{32L}, 0f, true, "cpu");
            m.eval();
            Tensor tokens = randint(0, 64, new long[]{2, 6});
            // Keep positions well within maxSeqLen so positionEmbedding never OOBs
            Tensor positions = randint(0, 6, new long[]{2, 6});
            return new Object[]{m, new Tensor[]{tokens, positions}};
        });
        generative.put("RQVAE", () -> {
            // embedDim, numCodebooks, codebookSize, latentDim, device
            RQVAE m = new RQVAE(16, 2, 8, 8, "cpu");
            m.eval();
            // forward(Tensor, boolean useSk) — boolean defaults to true in engine
            return new Object[]{m, randn(4, 16)};
        });
        generative.put("ResidualVectorQuantizer", () -> {
            ResidualVectorQuantizer m = new ResidualVectorQuantizer(new int[]{8, 8}, 16);
            m.eval();
            return new Object[]{m, randn(4, 16)};
        });
        generative.put("VectorQuantizer", () -> {
            VectorQuantizer m = new VectorQuantizer(16, 8);
            m.eval();
            return new Object[]{m, randn(4, 8)};
        });
        generative.put("HLLMTransformerBlock", () -> {
            HLLMTransformerBlock m = new HLLMTransformerBlock(32, 2, 0f, "cpu");
            m.eval();
            return new Object[]{m, randn(2, 6, 32)};
        });
        generative.put("LLM4RecEncoderLayer", () -> {
            LLM4RecEncoderLayer m = new LLM4RecEncoderLayer(32, 2, 64, 0f, "cpu");
            m.eval();
            return new Object[]{m, randn(2, 6, 32)};
        });

        System.out.println("========== multi_task (" + multiTask.size() + ") ==========");
        runSuite("multi_task", multiTask, mtDir, only, openBrowser);
        System.out.println("\n========== generative (" + generative.size() + ") ==========");
        runSuite("generative", generative, genDir, only, openBrowser);

        System.out.println("\nHTML root: " + root.toAbsolutePath());
        System.out.println("Controls: →LR ←RL ↓TB ↑BT · ＋/－ zoom · 🍡Cute 🌙Dark 💼Office · 📌 pin-drag · ⇩ Export");
        writeIndex(root, mtDir, genDir);
    }

    private static void runSuite(String suite, Map<String, ModelFactory> factories,
                                 Path outDir, String only, boolean openBrowser) {
        int ok = 0, fail = 0, skip = 0;
        boolean opened = false;
        for (Map.Entry<String, ModelFactory> e : factories.entrySet()) {
            String name = e.getKey();
            if (only != null) {
                List<String> want = Arrays.asList(only.split(","));
                if (!want.contains(name)) { skip++; continue; }
            }
            Path html = outDir.resolve(name + ".html");
            System.out.println("\n── " + suite + "/" + name + " ──");
            try {
                Object[] pair = e.getValue().create();
                Module model = (Module) pair[0];
                Object inputs = pair[1];
                try {
                    String tree = ModulePrinter.format(model);
                    System.out.println(tree.lines().limit(14).reduce((a, b) -> a + "\n" + b).orElse(""));
                } catch (Throwable ignored) {}

                TraceGraph g = Vista.trace(model, inputs, VistaOptions.defaults()
                        .evalMode(true)
                        .collapseModulesAfterDepth(1)
                        .showModuleAttrNames(true)
                        .height(900));
                boolean open = openBrowser && !opened;
                Vista.render(g, VistaOptions.defaults()
                        .exportFormat(ExportFormat.HTML)
                        .exportPath(html.toString())
                        .openBrowser(open)
                        .height(900)
                        .showModuleAttrNames(true)
                        .collapseModulesAfterDepth(1));
                if (open) opened = true;
                System.out.println(Vista.summary(g) + " → " + html.getFileName());
                int shown = 0;
                for (String n : g.adjList().keySet()) {
                    if (shown++ >= 6) {
                        System.out.println("  … +" + (g.adjList().size() - 6) + " more");
                        break;
                    }
                    var node = g.adjList().get(n);
                    System.out.println("  [" + node.nodeType().value() + "] "
                            + g.graphNodeDisplayNames().getOrDefault(n, n)
                            + " e=" + node.edges().size());
                }
                if (g.exception() != null) {
                    System.out.println("  (partial) " + g.exception().getClass().getSimpleName()
                            + ": " + String.valueOf(g.exception().getMessage()).lines().findFirst().orElse(""));
                }
                ok++;
            } catch (Throwable t) {
                fail++;
                System.err.println("  FAIL " + name + ": " + t.getClass().getSimpleName()
                        + ": " + t.getMessage());
                t.printStackTrace(System.err);
            }
        }
        System.out.println("\n" + suite + ": ok=" + ok + " fail=" + fail + " skip=" + skip);
    }

    private static Map<String, Tensor> featureBatch(List<? extends Feature> feats, int batch) {
        Map<String, Tensor> batchMap = new LinkedHashMap<>();
        for (Feature f : feats) {
            try {
                if (f instanceof DenseFeature df) {
                    // dense → float (batch, embedDim)
                    batchMap.put(f.name(), randn(batch, Math.max(1, df.embedDim())));
                    continue;
                }
                if (f instanceof SequenceFeature sf) {
                    long vocab = Math.max(2, sf.vocabSize());
                    int maxLen = 8;
                    try { maxLen = Math.max(2, (int) sf.maxLen()); } catch (Throwable ignored) {}
                    // sequence → long (batch, seq_len)
                    batchMap.put(f.name(), randint(0, vocab, new long[]{batch, maxLen}));
                    continue;
                }
                long vocab = 100;
                if (f instanceof SparseFeature sp) {
                    vocab = Math.max(2, sp.vocabSize());
                }
                batchMap.put(f.name(), randint(0, vocab, new long[]{batch}));
            } catch (Throwable t) {
                batchMap.put(f.name(), randint(0, 100, new long[]{batch}));
            }
        }
        return batchMap;
    }

    private static void writeIndex(Path root, Path mtDir, Path genDir) throws Exception {
        StringBuilder sb = new StringBuilder();
        sb.append("<!DOCTYPE html><html><head><meta charset=utf-8><title>Vista gallery</title>");
        sb.append("<style>body{font-family:system-ui;margin:24px;background:#f5f7fa;color:#1f2937}");
        sb.append("a{color:#2563eb;text-decoration:none;font-weight:600} a:hover{text-decoration:underline}");
        sb.append("h1{font-size:22px} h2{font-size:16px;margin-top:24px;color:#64748b}");
        sb.append("ul{line-height:1.9;columns:2} .card{background:#fff;border:1px solid #e5e7eb;border-radius:12px;");
        sb.append("padding:16px 20px;margin:12px 0;box-shadow:0 4px 12px rgba(15,23,42,.06)}</style></head><body>");
        sb.append("<h1>🍡 jnitorch Vista — recommend model gallery</h1>");
        sb.append("<p>Open any graph → switch <b>LR/RL/TB/BT</b>, zoom <b>＋/－</b>, themes <b>Cute/Dark/Office</b>, ");
        sb.append("drag nodes / pin frames, <b>Export</b> SVG/PNG/JPEG/PDF. Flowing dashed edges on every hop.</p>");
        sb.append("<div class=card><h2>multi_task</h2><ul>");
        try (var st = Files.list(mtDir)) {
            st.filter(p -> p.toString().endsWith(".html")).sorted().forEach(p ->
                    sb.append("<li><a href=\"multi_task/").append(p.getFileName()).append("\">")
                            .append(p.getFileName()).append("</a></li>"));
        }
        sb.append("</ul></div><div class=card><h2>generative</h2><ul>");
        try (var st = Files.list(genDir)) {
            st.filter(p -> p.toString().endsWith(".html")).sorted().forEach(p ->
                    sb.append("<li><a href=\"generative/").append(p.getFileName()).append("\">")
                            .append(p.getFileName()).append("</a></li>"));
        }
        sb.append("</ul></div></body></html>");
        Files.writeString(root.resolve("index.html"), sb.toString());
        System.out.println("Gallery index: " + root.resolve("index.html").toAbsolutePath());
    }
}
