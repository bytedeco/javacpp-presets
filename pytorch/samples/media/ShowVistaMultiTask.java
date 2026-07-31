package media;

import static org.bytedeco.pytorch.global.torch.randn;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.ModulePrinter;
import org.bytedeco.pytorch.recommend.basic.features.Feature;
import org.bytedeco.pytorch.recommend.basic.features.Features;
import org.bytedeco.pytorch.recommend.basic.layers.MLP;
import org.bytedeco.pytorch.recommend.models.multi_task.AttentionLayer;
import org.bytedeco.pytorch.recommend.models.multi_task.MetaLinear;
import org.bytedeco.pytorch.recommend.models.multi_task.SharedBottom;
import org.bytedeco.pytorch.plot.vista.ExportFormat;
import org.bytedeco.pytorch.plot.vista.TraceGraph;
import org.bytedeco.pytorch.plot.vista.Vista;
import org.bytedeco.pytorch.plot.vista.VistaOptions;

/**
 * Vista demo on recommend {@code multi_task} models — <b>without modifying
 * model source</b>. The engine expands {@code named_children} non-invasively
 * and runs child forwards itself to capture shapes.
 *
 * <pre>
 *   java --add-opens=java.base/java.nio=ALL-UNNAMED \
 *        --enable-native-access=ALL-UNNAMED \
 *        -cp "..." media.ShowVistaMultiTask [--no-browser]
 * </pre>
 *
 * <p>Outputs under {@code samples/out/vista-multitask/}.
 */
public class ShowVistaMultiTask {

    public static void main(String[] args) throws Exception {
        boolean openBrowser = true;
        for (String a : args) {
            if ("--no-browser".equals(a)) openBrowser = false;
        }

        Path out = Paths.get("samples/out/vista-multitask");
        Files.createDirectories(out);

        // ── 1. MLP (single Sequential child — full Linear/ReLU chain) ───────
        System.out.println("=== 1. MLP ===");
        System.out.println(ModulePrinter.format(new MLP(64, new long[]{128, 64}, 8, "relu", 0.1f, false, false, true, "cpu")));
        MLP mlp = new MLP(64, new long[]{128, 64}, 8, "relu", 0.1f, false, false, true, "cpu");
        mlp.eval();
        TraceGraph gMlp = Vista.traceModel(
                mlp,
                randn(4, 64),
                opts(out.resolve("01_mlp.html"), openBrowser, 1));
        summarize(gMlp);

        // ── 2. AttentionLayer (Q/K/V Linear children — model code untouched) ─
        System.out.println("\n=== 2. AttentionLayer (multi_task) ===");
        AttentionLayer attn = new AttentionLayer(32, "cpu");
        attn.eval();
        System.out.println(ModulePrinter.format(attn));
        // Input (batch, 2, dim) per AITM AttentionLayer contract
        Tensor attnIn = randn(4, 2, 32);
        TraceGraph gAttn = Vista.traceModel(
                attn,
                attnIn,
                opts(out.resolve("02_attention.html"), false, 1));
        summarize(gAttn);

        // ── 3. MetaLinear (inner Linear child) ──────────────────────────────
        System.out.println("\n=== 3. MetaLinear (multi_task) ===");
        MetaLinear meta = new MetaLinear(16, 8, "cpu");
        meta.eval();
        System.out.println(ModulePrinter.format(meta));
        TraceGraph gMeta = Vista.traceModel(
                meta,
                randn(4, 16),
                opts(out.resolve("03_meta_linear.html"), false, 1));
        summarize(gMeta);

        // ── 4. SharedBottom multi-task (Map features — non-invasive expand) ─
        System.out.println("\n=== 4. SharedBottom (multi_task) ===");
        List<Feature> feats = new ArrayList<>();
        feats.add(Features.sparse("user_id", 1000, 8));
        feats.add(Features.sparse("item_id", 5000, 8));
        feats.add(Features.sparse("cate_id", 200, 8));
        List<String> tasks = Arrays.asList("classification", "classification");
        Map<String, Object> bottom = new LinkedHashMap<>();
        bottom.put("dims", Arrays.asList(64L, 32L));
        bottom.put("activation", "relu");
        bottom.put("dropout", 0.0f);
        SharedBottom sb = new SharedBottom(feats, tasks, bottom, List.of(), "cpu");
        sb.eval();
        System.out.println(ModulePrinter.format(sb));

        Map<String, Tensor> batch = new LinkedHashMap<>();
        // EmbeddingLayer expects Long indices (batch,)
        batch.put("user_id", org.bytedeco.pytorch.global.torch.randint(0, 1000, new long[]{4}));
        batch.put("item_id", org.bytedeco.pytorch.global.torch.randint(0, 5000, new long[]{4}));
        batch.put("cate_id", org.bytedeco.pytorch.global.torch.randint(0, 200, new long[]{4}));

        TraceGraph gSb = Vista.traceModel(
                sb,
                batch,
                opts(out.resolve("04_shared_bottom.html"), openBrowser, 1)
                        .showModuleAttrNames(true)
                        .height(900));
        summarize(gSb);

        System.out.println("\nDone. Open HTML under " + out.toAbsolutePath());
        System.out.println("Models were NOT modified — expansion is engine-side only.");
    }

    private static VistaOptions opts(Path html, boolean browser, int collapse) {
        return VistaOptions.defaults()
                .height(820)
                .collapseModulesAfterDepth(collapse)
                .showModuleAttrNames(true)
                .exportFormat(ExportFormat.HTML)
                .exportPath(html.toString())
                .openBrowser(browser)
                .evalMode(true);
    }

    private static void summarize(TraceGraph g) {
        System.out.println(Vista.summary(g));
        for (String name : g.adjList().keySet()) {
            var node = g.adjList().get(name);
            String display = g.graphNodeDisplayNames().getOrDefault(name, name);
            String in = node.originalIncomingDims() == null ? "" : node.originalIncomingDims().toString();
            String out = node.originalOutgoingDims() == null ? "" : node.originalOutgoingDims().toString();
            System.out.println("  [" + node.nodeType().value() + "] " + display
                    + "  in=" + in + " out=" + out
                    + " edges=" + node.edges().size());
        }
        if (g.exception() != null) {
            System.out.println("  (partial) error: " + g.exception().getMessage());
        }
    }
}
