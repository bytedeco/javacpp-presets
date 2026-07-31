package org.bytedeco.pytorch.plot.vista;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Objects;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.data.serialize.StructureSpec;
import org.bytedeco.pytorch.inductor.AOTIModelPackageLoader;
import org.bytedeco.pytorch.nn.Module;

/**
 * Interactive PyTorch forward-pass visualisation for JavaCPP / jnitorch.
 *
 * <p>Conceptual port of
 * <a href="https://github.com/sachinhosmani/torchvista">torchvista</a>
 * ({@code trace_model}) onto modules already present in this codebase.
 *
 * <h2>Quick start</h2>
 * <pre>
 * // Live Module
 * Vista.traceModel(model, inputs);
 *
 * // Free ops inside custom Module.forward — use VistaOps
 * h = VistaOps.module(fc, x);
 * h = VistaOps.relu(h);
 * h = VistaOps.add(h, x);
 *
 * // Weight / structure files
 * Vista.traceFile("model.safetensors", sampleInput);
 * Vista.traceFile("model.pth");                 // structure if no input
 * Vista.traceFile("model.structure.json");
 * Vista.traceFile("aoti_package_dir");
 * </pre>
 *
 * @see VistaOptions
 * @see VistaOps
 * @see VistaModelFiles
 * @see TraceGraph
 */
public final class Vista {
    private Vista() {}

    // =========================================================================
    // Live Module API (torchvista trace_model)
    // =========================================================================

    public static TraceGraph traceModel(Module model, Object inputs) {
        return traceModel(model, inputs, VistaOptions.defaults());
    }

    public static TraceGraph traceModel(Module model, Object inputs, VistaOptions options) {
        Objects.requireNonNull(model, "model");
        if (options == null) options = VistaOptions.defaults();

        if (options.exportFormat() == null && options.exportPath() != null) {
            options.exportFormat(ExportFormat.HTML);
        }

        boolean wasTraining = false;
        try {
            wasTraining = model.is_training();
        } catch (Throwable ignored) {}
        if (wasTraining && !options.evalMode()) {
            System.err.println("[vista] warning: model is in training mode and evalMode=false.");
        } else if (wasTraining) {
            System.err.println("[vista] note: model was training; trace ran under eval() and restored.");
        }

        TraceGraph graph = trace(model, inputs, options);
        try {
            render(graph, options);
        } catch (IOException e) {
            throw new RuntimeException("vista render failed: " + e.getMessage(), e);
        }
        if (graph.exception() != null) {
            Throwable ex = graph.exception();
            if (ex instanceof RuntimeException) throw (RuntimeException) ex;
            if (ex instanceof Error) throw (Error) ex;
            throw new RuntimeException(ex);
        }
        return graph;
    }

    public static TraceGraph traceModel(Module model, Tensor input) {
        return traceModel(model, (Object) input, VistaOptions.defaults());
    }

    public static TraceGraph traceModel(Module model, Tensor input, VistaOptions options) {
        return traceModel(model, (Object) input, options);
    }

    public static TraceGraph traceModel(Module model, Tensor[] inputs) {
        return traceModel(model, (Object) inputs, VistaOptions.defaults());
    }

    public static TraceGraph traceModel(Module model, Tensor[] inputs, VistaOptions options) {
        return traceModel(model, (Object) inputs, options);
    }

    public static TraceGraph trace(Module model, Object inputs) {
        return trace(model, inputs, VistaOptions.defaults());
    }

    public static TraceGraph trace(Module model, Object inputs, VistaOptions options) {
        Objects.requireNonNull(model, "model");
        VistaEngine engine = new VistaEngine(options == null ? VistaOptions.defaults() : options);
        return engine.process(model, inputs);
    }

    public static Path render(TraceGraph graph, VistaOptions options) throws IOException {
        return VistaRender.plotGraph(graph, options == null ? VistaOptions.defaults() : options);
    }

    public static String toHtml(TraceGraph graph, VistaOptions options) {
        return VistaRender.buildHtml(graph, options == null ? VistaOptions.defaults() : options, null);
    }

    public static String summary(TraceGraph graph) {
        return graph == null ? "null" : graph.summary();
    }

    // =========================================================================
    // File / checkpoint API
    // =========================================================================

    /**
     * Visualise a model file (safetensors / python pth / javacpp pth /
     * structure.json / AOTI package). Without sample inputs, builds a
     * structure-only graph when possible.
     */
    public static TraceGraph traceFile(String path) {
        return traceFile(path, null, VistaOptions.defaults());
    }

    public static TraceGraph traceFile(File file) {
        return traceFile(file, null, VistaOptions.defaults());
    }

    public static TraceGraph traceFile(String path, Object inputs) {
        return traceFile(path, inputs, VistaOptions.defaults());
    }

    public static TraceGraph traceFile(String path, Object inputs, VistaOptions options) {
        return traceFile(new File(path), inputs, options);
    }

    public static TraceGraph traceFile(File file, Object inputs, VistaOptions options) {
        Objects.requireNonNull(file, "file");
        if (options == null) options = VistaOptions.defaults();
        if (options.exportFormat() == null && options.exportPath() != null) {
            options.exportFormat(ExportFormat.HTML);
        }
        if (options.exportPath() == null) {
            // default next to cwd with file stem
            String stem = file.getName().replaceAll("\\.[^.]+$", "");
            options.exportPath("jnitorch_vista_" + stem + ".html");
        }

        VistaModelFiles.Loaded loaded;
        try {
            loaded = VistaModelFiles.open(file);
        } catch (IOException e) {
            throw new RuntimeException("vista open failed: " + e.getMessage(), e);
        }
        System.out.println("[vista] loaded " + loaded.kind + " · " + loaded.note);

        TraceGraph graph;
        switch (loaded.kind) {
            case AOTI_PACKAGE:
                graph = traceAoti(loaded, inputs, options);
                break;
            case STRUCTURE_JSON:
                if (loaded.hasRunnableModule() && inputs != null) {
                    graph = trace(loaded.module, inputs, options);
                } else if (loaded.hasStructure()) {
                    graph = StructureGraphBuilder.fromStructure(loaded.structure);
                } else {
                    throw new IllegalStateException("structure.json produced nothing graphable: " + file);
                }
                break;
            default:
                if (loaded.hasRunnableModule() && inputs != null) {
                    graph = trace(loaded.module, inputs, options);
                } else if (loaded.hasStructure()) {
                    System.out.println("[vista] no sample inputs — structure-only graph"
                            + (loaded.hasRunnableModule()
                            ? " (module loaded; pass inputs for live shapes)" : ""));
                    graph = StructureGraphBuilder.fromStructure(loaded.structure);
                } else if (loaded.hasRunnableModule()) {
                    // Module without structure and without inputs: dump structure from module
                    try {
                        StructureSpec spec = StructureSpec.fromModule(loaded.module);
                        graph = StructureGraphBuilder.fromStructure(spec);
                    } catch (Throwable t) {
                        throw new IllegalStateException(
                                "module loaded but cannot build structure graph and no inputs given: "
                                        + t.getMessage(), t);
                    }
                } else {
                    throw new IllegalStateException("nothing to visualise from " + file);
                }
                break;
        }

        try {
            render(graph, options);
        } catch (IOException e) {
            throw new RuntimeException("vista render failed: " + e.getMessage(), e);
        }
        if (graph.exception() != null) {
            Throwable ex = graph.exception();
            if (ex instanceof RuntimeException) throw (RuntimeException) ex;
            throw new RuntimeException(ex);
        }
        return graph;
    }

    private static TraceGraph traceAoti(VistaModelFiles.Loaded loaded, Object inputs,
                                        VistaOptions options) {
        AOTIModelPackageLoader aot = loaded.aot;
        if (aot == null || aot.isNull()) {
            throw new IllegalStateException("AOTI loader missing for " + loaded.sourcePath);
        }
        if (inputs == null) {
            return StructureGraphBuilder.fromAoti(aot, loaded.sourcePath);
        }
        // Live AOT run: single Operation node with real output dims
        TraceGraph g = StructureGraphBuilder.fromAoti(aot, loaded.sourcePath);
        try {
            TensorVector tv = toTensorVector(inputs);
            TensorVector out = VistaModelFiles.runAoti(aot, tv);
            // Annotate output edge dims if possible
            if (out != null && !out.isNull() && out.size() > 0) {
                try {
                    Tensor t0 = out.get(0);
                    if (t0 != null && !t0.isNull()) {
                        String dims = TensorUtils.formatDims(t0);
                        GraphNode run = g.adjList().get("aoti_run_1");
                        if (run != null) {
                            run.edges().clear();
                            run.addEdge(new GraphEdge("output", dims));
                        }
                        GraphNode in = g.adjList().get("input_0");
                        if (in != null && !in.edges().isEmpty()) {
                            // keep input edge; optionally set dims from first input
                            java.util.List<Tensor> ins = TensorUtils.extractTensors(inputs);
                            if (!ins.isEmpty()) {
                                String idims = TensorUtils.formatDims(ins.get(0));
                                in.edges().clear();
                                in.addEdge(new GraphEdge("aoti_run_1", idims));
                            }
                        }
                    }
                } catch (Throwable ignored) {}
            }
        } catch (Throwable e) {
            g.setException(e);
            System.err.println("[vista] AOTI run failed (showing metadata graph): " + e.getMessage());
        }
        return g;
    }

    private static TensorVector toTensorVector(Object inputs) {
        TensorVector v = new TensorVector();
        for (Tensor t : TensorUtils.extractTensors(inputs)) {
            v.push_back(t);
        }
        return v;
    }
}
