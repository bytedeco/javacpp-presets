package samples;
import org.bytedeco.pytorch.nn.*;
import org.bytedeco.pytorch.nn.modules.*;
import org.bytedeco.pytorch.nn.modules.container.*;

import static org.bytedeco.pytorch.global.torch.randn;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.ModulePrinter;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.ReLUImpl;
import org.bytedeco.pytorch.nn.modules.container.SequentialImpl;
import org.bytedeco.pytorch.utils.vista.ExportFormat;
import org.bytedeco.pytorch.utils.vista.TraceGraph;
import org.bytedeco.pytorch.utils.vista.Vista;
import org.bytedeco.pytorch.utils.vista.VistaOps;
import org.bytedeco.pytorch.utils.vista.VistaOptions;

/**
 * Demo: premium vista graph + free-op recognition + nested Sequential.
 *
 * <pre>
 *   java --add-opens=java.base/java.nio=ALL-UNNAMED \
 *        --enable-native-access=ALL-UNNAMED \
 *        -cp "..." samples.ShowVistaDemo [--no-browser] [model.safetensors|model.pth|...]
 * </pre>
 */
public class ShowVistaDemo {

    /** Custom block: free ops + nested modules become graph nodes via VistaOps. */
    public static final class ResidualMLP extends Module {
        final LinearImpl fc1;
        final LinearImpl fc2;

        public ResidualMLP(long dim) {
            super("ResidualMLP");
            fc1 = register_module("fc1", new LinearImpl(dim, dim));
            fc2 = register_module("fc2", new LinearImpl(dim, dim));
        }

        @Override
        public Tensor forward(Tensor x) {
            // Child modules + free ops — must go through VistaOps to appear in graph
            Tensor h = VistaOps.module(fc1, x);
            h = VistaOps.relu(h);
            h = VistaOps.module(fc2, h);
            h = VistaOps.add(h, x); // residual
            return h;
        }
    }

    public static void main(String[] args) throws Exception {
        boolean openBrowser = true;
        String filePath = null;
        for (String a : args) {
            if ("--no-browser".equals(a)) openBrowser = false;
            else if (!a.startsWith("-")) filePath = a;
        }

        Path outDir = Paths.get("samples/out/vista-demo");
        Files.createDirectories(outDir);

        // ── 1. Sequential MLP ───────────────────────────────────────────────
        SequentialImpl model = new SequentialImpl();
        model.push_back("fc1", new LinearImpl(10, 32));
        model.push_back("act", new ReLUImpl());
        model.push_back("fc2", new LinearImpl(32, 5));
        model.eval();

        System.out.println("=== ModulePrinter ===");
        System.out.println(ModulePrinter.format(model));

        Tensor inputs = randn(2, 10);
        Path html1 = outDir.resolve("mlp.html");
        TraceGraph g1 = Vista.traceModel(
                model,
                inputs,
                VistaOptions.defaults()
                        .height(720)
                        .collapseModulesAfterDepth(1)
                        .showModuleAttrNames(true)
                        .exportFormat(ExportFormat.HTML)
                        .exportPath(html1.toString())
                        .openBrowser(openBrowser));
        System.out.println(Vista.summary(g1));
        printNodes(g1);
        System.out.println("wrote " + html1.toAbsolutePath());

        // ── 2. Custom module with free ops (relu / add / nested Linear) ─────
        ResidualMLP residual = new ResidualMLP(16);
        residual.eval();
        Tensor x2 = randn(4, 16);
        Path html2 = outDir.resolve("residual_freeops.html");
        TraceGraph g2 = Vista.traceModel(
                residual,
                x2,
                VistaOptions.defaults()
                        .height(780)
                        .showModuleAttrNames(true)
                        .exportPath(html2.toString())
                        .openBrowser(false));
        System.out.println("\n=== ResidualMLP + VistaOps free ops ===");
        System.out.println(Vista.summary(g2));
        printNodes(g2);
        System.out.println("wrote " + html2.toAbsolutePath());

        // ── 3. Nested Sequential ────────────────────────────────────────────
        SequentialImpl block = new SequentialImpl();
        block.push_back("lin", new LinearImpl(16, 16));
        block.push_back("relu", new ReLUImpl());
        SequentialImpl nested = new SequentialImpl();
        nested.push_back("stem", new LinearImpl(8, 16));
        nested.push_back("block", block);
        nested.push_back("head", new LinearImpl(16, 3));
        nested.eval();

        Path html3 = outDir.resolve("nested.html");
        TraceGraph g3 = Vista.trace(
                nested,
                randn(4, 8),
                VistaOptions.defaults().collapseModulesAfterDepth(0).showModuleAttrNames(true));
        Vista.render(
                g3,
                VistaOptions.defaults()
                        .height(800)
                        .collapseModulesAfterDepth(0)
                        .showModuleAttrNames(true)
                        .exportPath(html3.toString())
                        .openBrowser(false));
        System.out.println("\n=== Nested Sequential ===");
        System.out.println(Vista.summary(g3));
        System.out.println("wrote " + html3.toAbsolutePath());

        // ── 4. Optional: model file ─────────────────────────────────────────
        if (filePath != null) {
            Path html4 = outDir.resolve("from_file.html");
            System.out.println("\n=== traceFile " + filePath + " ===");
            try {
                TraceGraph gf = Vista.traceFile(
                        filePath,
                        null, // structure-only if no inputs; pass a Tensor for live shapes
                        VistaOptions.defaults()
                                .height(860)
                                .exportPath(html4.toString())
                                .openBrowser(openBrowser));
                System.out.println(Vista.summary(gf));
                printNodes(gf);
                System.out.println("wrote " + html4.toAbsolutePath());
            } catch (Throwable t) {
                System.err.println("traceFile failed: " + t.getMessage());
                t.printStackTrace(System.err);
            }
        }

        System.out.println("\nDone. Open HTML under " + outDir.toAbsolutePath());
    }

    private static void printNodes(TraceGraph g) {
        for (String name : g.adjList().keySet()) {
            String display = g.graphNodeDisplayNames().getOrDefault(name, name);
            String type = g.adjList().get(name).nodeType().value();
            int edges = g.adjList().get(name).edges().size();
            System.out.println("  [" + type + "] " + display + " (" + name + ") edges=" + edges);
        }
    }
}
