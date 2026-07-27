package org.bytedeco.pytorch.data.serialize;
import org.bytedeco.pytorch.nn.*;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.StringTensorDict;
import org.bytedeco.pytorch.StringTensorDictItem;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Pretty-print weight / Module structure for diagnostics and conversion reports.
 *
 * <p>Works on:
 * <ul>
 *   <li>{@code Map&lt;String, Tensor&gt;} state-dicts (from .pth / safetensors)</li>
 *   <li>live {@link Module} via {@code named_parameters(true)}</li>
 *   <li>inferred typed layers via {@link StateDictModuleBuilder#infer}</li>
 * </ul>
 */
public final class ModelStructure {
    private ModelStructure() {}

    public static final class TensorInfo {
        public final String name;
        public final long[] shape;
        public final String dtype;
        public final long numel;
        public final long bytes;

        public TensorInfo(String name, long[] shape, String dtype, long numel, long bytes) {
            this.name = name;
            this.shape = shape == null ? new long[0] : shape.clone();
            this.dtype = dtype;
            this.numel = numel;
            this.bytes = bytes;
        }

        public String shapeString() {
            if (shape.length == 0) return "[]";
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < shape.length; i++) {
                if (i > 0) sb.append(", ");
                sb.append(shape[i]);
            }
            return sb.append(']').toString();
        }
    }

    public static final class Report {
        public final String title;
        public final List<TensorInfo> tensors = new ArrayList<>();
        public long totalParams;
        public long totalBytes;
        public final Map<String, Long> dtypeCounts = new LinkedHashMap<>();
        public final Map<String, Integer> prefixCounts = new LinkedHashMap<>();

        public Report(String title) { this.title = title; }

        public void print() { print(System.out); }

        public void print(PrintStream out) {
            out.println("======== Model Structure: " + title + " ========");
            out.printf(Locale.ROOT, "tensors=%d  params=%s  bytes=%s (%.2f MiB)%n",
                tensors.size(),
                formatLong(totalParams),
                formatLong(totalBytes),
                totalBytes / (1024.0 * 1024.0));
            if (!dtypeCounts.isEmpty()) {
                out.print("dtypes: ");
                boolean first = true;
                for (Map.Entry<String, Long> e : dtypeCounts.entrySet()) {
                    if (!first) out.print(", ");
                    out.print(e.getKey() + "=" + e.getValue());
                    first = false;
                }
                out.println();
            }
            if (!prefixCounts.isEmpty()) {
                out.println("top-level modules / prefixes:");
                int shown = 0;
                for (Map.Entry<String, Integer> e : prefixCounts.entrySet()) {
                    out.printf(Locale.ROOT, "  %-24s  %d tensors%n", e.getKey(), e.getValue());
                    if (++shown >= 32) {
                        out.println("  ...");
                        break;
                    }
                }
            }
            out.println("---- tensors ----");
            out.printf(Locale.ROOT, "%-48s  %-16s  %-10s  %12s  %12s%n",
                "name", "shape", "dtype", "numel", "bytes");
            int i = 0;
            for (TensorInfo t : tensors) {
                out.printf(Locale.ROOT, "%-48s  %-16s  %-10s  %12s  %12s%n",
                    truncate(t.name, 48),
                    truncate(t.shapeString(), 16),
                    t.dtype,
                    formatLong(t.numel),
                    formatLong(t.bytes));
                if (++i >= 200) {
                    out.println("... (" + (tensors.size() - i) + " more)");
                    break;
                }
            }
            out.println("================================================");
        }
    }

    /** Inspect a state-dict map (keys may use {@code .} hierarchy). */
    public static Report ofStateDict(String title, Map<String, Tensor> stateDict) {
        Report r = new Report(title == null ? "state_dict" : title);
        if (stateDict == null) return r;
        for (Map.Entry<String, Tensor> e : stateDict.entrySet()) {
            addTensor(r, e.getKey(), e.getValue());
        }
        finishPrefixes(r);
        return r;
    }

    /** Inspect a live Module via named_parameters(recurse=true). */
    public static Report ofModule(String title, Module module) {
        Report r = new Report(title == null ? "module" : title);
        if (module == null) return r;
        try {
            StringTensorDict dict = module.named_parameters(/*recurse=*/true);
            if (dict == null || dict.isNull()) return r;
            long n = dict.size();
            for (long i = 0; i < n; i++) {
                StringTensorDictItem item = dict.get(i);
                if (item == null || item.isNull()) continue;
                String name = item.key() != null ? item.key().getString() : ("param_" + i);
                Tensor t = item.value();
                addTensor(r, name, t);
            }
        } catch (Throwable ex) {
            // fall through with whatever we collected
            r.tensors.add(new TensorInfo("<error:" + ex.getClass().getSimpleName() + ">",
                new long[0], "?", 0, 0));
        }
        finishPrefixes(r);
        return r;
    }

    public static void printStateDict(String title, Map<String, Tensor> stateDict) {
        ofStateDict(title, stateDict).print();
    }

    public static void printStateDict(String title, Map<String, Tensor> stateDict, PrintStream out) {
        ofStateDict(title, stateDict).print(out);
    }

    public static void printModule(String title, Module module) {
        ofModule(title, module).print();
    }

    /**
     * Infer typed layers from a state-dict and print kinds / hyperparameters.
     * This is the structure report that shows Linear / Embedding / LayerNorm / …
     * reconstructed from shapes + names (what a pure Map of tensors cannot show).
     */
    public static void printInferred(String title, Map<String, Tensor> stateDict) {
        List<StateDictModuleBuilder.LayerInfo> layers = StateDictModuleBuilder.infer(stateDict);
        StateDictModuleBuilder.printLayers(title, layers);
    }

    /**
     * Print both the tensor inventory and the inferred typed module tree.
     */
    public static void printFull(String title, Map<String, Tensor> stateDict) {
        printStateDict(title, stateDict);
        printInferred(title + " (typed)", stateDict);
    }

    /**
     * Print a live {@link WeightBagModule} like Python {@code print(model)}
     * (via {@link org.bytedeco.pytorch.nn.ModulePrinter}), then optional
     * inferred-layer summary.
     */
    public static void printWeightBag(String title, WeightBagModule bag) {
        if (bag == null) {
            System.out.println("======== Model Structure: " + title + " (null) ========");
            return;
        }
        System.out.println("======== " + title + " ========");
        System.out.println(bag.toString()); // ModulePrinter tree
        if (!bag.layerInfos().isEmpty()) {
            StateDictModuleBuilder.printLayers(title + " layers", bag.layerInfos());
        } else {
            printInferred(title + " (inferred)", bag.stateDict());
        }
    }

    private static void addTensor(Report r, String name, Tensor t) {
        if (t == null || !t.defined()) {
            r.tensors.add(new TensorInfo(name, new long[0], "undefined", 0, 0));
            return;
        }
        long[] shape = new long[(int) t.dim()];
        for (int i = 0; i < shape.length; i++) shape[i] = t.sizes().get(i);
        long numel = t.numel();
        ScalarType st;
        try {
            st = t.scalar_type().intern();
        } catch (Throwable e) {
            st = ScalarType.Float;
        }
        String dtype = st != null ? st.name() : "?";
        int esize = elemSize(st);
        long bytes = numel * (long) esize;
        r.tensors.add(new TensorInfo(name, shape, dtype, numel, bytes));
        r.totalParams += numel;
        r.totalBytes += bytes;
        r.dtypeCounts.merge(dtype, 1L, Long::sum);
        String prefix = topPrefix(name);
        r.prefixCounts.merge(prefix, 1, Integer::sum);
    }

    private static void finishPrefixes(Report r) {
        // keep insertion order from LinkedHashMap; optionally sort by count desc would lose order —
        // leave as-is for hierarchy discovery
    }

    private static String topPrefix(String name) {
        if (name == null || name.isEmpty()) return "<root>";
        int dot = name.indexOf('.');
        if (dot < 0) return name;
        return name.substring(0, dot);
    }

    private static int elemSize(ScalarType st) {
        if (st == null) return 4;
        switch (st) {
            case Byte: case Char: case Bool: case QInt8: case QUInt8: return 1;
            case Short: case Half: case BFloat16: return 2;
            case Int: case Float: case ComplexHalf: return 4;
            case Long: case Double: case ComplexFloat: return 8;
            case ComplexDouble: return 16;
            default: return 4;
        }
    }

    private static String formatLong(long v) {
        if (v < 10_000) return Long.toString(v);
        if (v < 1_000_000) return String.format(Locale.ROOT, "%.1fK", v / 1e3);
        if (v < 1_000_000_000) return String.format(Locale.ROOT, "%.2fM", v / 1e6);
        return String.format(Locale.ROOT, "%.2fB", v / 1e9);
    }

    private static String truncate(String s, int max) {
        if (s == null) return "";
        if (s.length() <= max) return s;
        return s.substring(0, Math.max(0, max - 3)) + "...";
    }
}
