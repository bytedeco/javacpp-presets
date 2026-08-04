package media;

import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.*;
import static org.bytedeco.pytorch.global.torch.randn;
import static org.bytedeco.pytorch.global.torch.randint;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.LinkedHashMap;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.ModulePrinter;
import org.bytedeco.pytorch.recommend.basic.features.Feature;
import org.bytedeco.pytorch.recommend.basic.features.Features;
import org.bytedeco.pytorch.recommend.basic.features.SparseFeature;
import org.bytedeco.pytorch.recommend.models.ranking.DeepFM;
import org.bytedeco.pytorch.recommend.models.ranking.TransformerDCN;
import org.bytedeco.pytorch.plot.vista.ExportFormat;
import org.bytedeco.pytorch.plot.vista.TraceGraph;
import org.bytedeco.pytorch.plot.vista.Vista;
import org.bytedeco.pytorch.plot.vista.VistaOptions;

/**
 * Vista demo for recommend ranking models — DeepFM and TransformerDCN.
 *
 * <pre>
 *   java --add-opens=java.base/java.nio=ALL-UNNAMED \
 *        --enable-native-access=ALL-UNNAMED \
 *        -cp "..." media.ShowVistaDeepFMTransformerDCN [--no-browser]
 * </pre>
 *
 * Outputs under {@code samples/out/vista-recommend/}.
 */
public class ShowVistaDeepFMTransformerDCN {

    public static void main(String[] args) throws Exception {
        boolean openBrowser = true;
        for (String a : args) {
            if ("--no-browser".equals(a)) openBrowser = false;
        }

        Path out = Paths.get("samples/out/vista-recommend");
        Files.createDirectories(out);

        // ── 1. DeepFM ─────────────────────────────────────────────────────
        System.out.println("\n=== 1. DeepFM ===");
        List<Feature> deepFeatures = List.of(Features.sparse("user_id", 1000, 8));
        List<Feature> fmFeatures = List.of(Features.sparse("item_id", 2000, 8));

        DeepFM deepFM = new DeepFM(deepFeatures, fmFeatures);
        deepFM.eval();
        System.out.println(ModulePrinter.format(deepFM));

        Map<String, Tensor> sparseBatch = new LinkedHashMap<>();
        sparseBatch.put("user_id", randint(0, 1000, new long[]{4}));
        sparseBatch.put("item_id", randint(0, 2000, new long[]{4}));

        TraceGraph gDeepFM = Vista.traceModel(
                deepFM,
                sparseBatch,
                opts(out.resolve("01_deepfm.html"), openBrowser, 2));
        summarize(gDeepFM);

        // ── 2. TransformerDCN ──────────────────────────────────────────────
        System.out.println("\n=== 2. TransformerDCN ===");
        // Use a compact constructor that matches the full one (defaults provided)
        TransformerDCN transformerDCN = new TransformerDCN(
                1000L,           // itemVocabSize
                64L,             // embDim
                64L,             // pretrainDim
                16L,             // embDimPretrain
                200L,            // likesVocabSize
                200L,            // viewsVocabSize
                100L,            // tagsVocabSize
                1000L,           // numItems
                16,              // tagsLen
                null, null,      // pretrainedEmbFlat, itemTagsFlat
                2L,              // numHeads
                2,               // transformerLayers
                0.1,             // transformerDropout
                256L,            // dimFeedforward
                16,              // firstKCols
                true,            // concatMaxPool
                3,               // dcnCrossLayers
                new long[]{1024L, 512L, 256L},  // dcnHiddenUnits
                new long[]{64L, 32L},           // mlpHiddenUnits
                0.2,             // netDropout
                "cpu");

        transformerDCN.eval();
        System.out.println(ModulePrinter.format(transformerDCN));

        Tensor history = randint(0, 1000, new long[]{4, 8});
        Tensor target = randint(0, 1000, new long[]{4});
        Tensor mask = torch.ones(new long[]{4, 8}).to(torch.ScalarType.Float); // float mask
        Tensor likesLevel = randint(0, 200, new long[]{4});
        Tensor viewsLevel = randint(0, 200, new long[]{4});

        TraceGraph gTDCN = Vista.traceModel(
                transformerDCN,
                new Object[]{history, target, mask, likesLevel, viewsLevel},
                opts(out.resolve("02_transformer_dcn.html"), false, 2));
        summarize(gTDCN);

        System.out.println("\nDone. Open HTML under " + out.toAbsolutePath());
        System.out.println("Models were traced — no source modification.");
    }

    private static VistaOptions opts(Path html, boolean browser, int collapse) {
        return VistaOptions.defaults()
                .height(720)
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
