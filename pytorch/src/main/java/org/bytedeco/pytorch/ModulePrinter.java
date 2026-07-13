package org.bytedeco.pytorch;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.LongPointer;

import java.util.Arrays;

/**
 * Mirrors Python PyTorch's print(model) for JavaCPP Module. The default
 * Pointer.toString() prints a memory address; we override it via the
 * @Override in Module.java (injected by strip_module_forward.sh) to
 * delegate here. Output:
 *
 *   SequentialImpl(
 *     (0): LinearImpl(in_features=12, out_features=64, bias=false)
 *     (1): ReLUImpl()
 *     (2): DropoutImpl(p=0.5, inplace=false)
 *     (3): LinearImpl(in_features=64, out_features=10, bias=true)
 *   )
 *
 *   ModuleListImpl(
 *     (0): LinearImpl(in_features=...)
 *     (1): Conv2dImpl(in_channels=3, out_channels=16, kernel_size=[3,3], ...)
 *   )
 *
 * Implementation:
 *   - getTypeName(m) returns the simple class name from
 *     Module::name().
 *   - describeAttrs(m) dispatches on the simple class name to a
 *     hard-coded attr fetcher. Most options' scalar fields are
 *     returned as LongPointer / BoolPointer (single-element
 *     ExpandingArray wrappers in libtorch); we read .get(0) for
 *     the value.
 *   - format() recurses into children() for Sequential, plus into
 *     parameters() / named_children() etc. for container modules.
 */
final class ModulePrinter {

    private ModulePrinter() {}

    static String format(Module m) {
        if (m == null) return "null";
        StringBuilder sb = new StringBuilder();
        sb.append(getTypeName(m));
        String attrs = describeAttrs(m);
        if (!attrs.isEmpty()) {
            sb.append('(').append(attrs).append(')');
        }
        // Children / parameters — for most modules this is the
        // children() of the underlying Module, but container
        // subclasses (ModuleList, ModuleDict, ParameterList,
        // ParameterDict) may need special handling. children() works
        // for Sequential / ModuleList / etc.
        Module[] kids = safeChildren(m);
        if (kids != null && kids.length > 0) {
            sb.append("(\n");
            for (int i = 0; i < kids.length; i++) {
                sb.append("  (").append(i).append("): ");
                sb.append(format(kids[i]));
                sb.append('\n');
            }
            sb.append(')');
        }
        // parameters() in this libtorch build triggers
        // named_parameters() and SIGSEGVs in some cases. The user can
        // always call m.parameters() themselves if they need the
        // parameter list. We skip it here for stability.
        return sb.toString();
    }

    /** Returns the C++ class name from Module::name() (e.g.
     *  "torch::nn::LinearImpl"). We keep the namespace prefix to match
     *  Python PyTorch's repr(model) output. */
    private static String getTypeName(Module m) {
        try {
            String s = m.name().getString();
            if (s != null && !s.isEmpty()) return s;
        } catch (Throwable ignored) {}
        return m.getClass().getSimpleName();
    }

    private static String describeAttrs(Module m) {
        // getTypeName returns e.g. "torch::nn::LinearImpl" with the
        // torch::nn:: namespace prefix. For the dispatch below we want
        // the simple class name only, so strip the prefix.
        String name = getTypeName(m);
        int colonColon = name.lastIndexOf("::");
        String simple = (colonColon >= 0) ? name.substring(colonColon + 2) : name;
        if (simple.equals("LinearImpl") || simple.equals("BilinearImpl")) {
            return linearAttrs(m);
        }
        if (simple.equals("Conv1dImpl") || simple.equals("Conv2dImpl") || simple.equals("Conv3dImpl")) {
            return convAttrs(m, 2);
        }
        if (simple.equals("ConvTranspose1dImpl") || simple.equals("ConvTranspose2dImpl")
                || simple.equals("ConvTranspose3dImpl")) {
            return convTransposeAttrs(m);
        }
        if (simple.equals("BatchNorm1dImpl") || simple.equals("BatchNorm2dImpl")
                || simple.equals("BatchNorm3dImpl")) {
            return batchNormAttrs(m);
        }
        if (simple.equals("InstanceNorm1dImpl") || simple.equals("InstanceNorm2dImpl")
                || simple.equals("InstanceNorm3dImpl")) {
            return instanceNormAttrs(m);
        }
        if (simple.equals("GroupNormImpl")) {
            return groupNormAttrs(m);
        }
        if (simple.equals("LayerNormImpl")) {
            return layerNormAttrs(m);
        }
        if (simple.equals("DropoutImpl") || simple.equals("Dropout2dImpl") || simple.equals("Dropout3dImpl")
                || simple.equals("AlphaDropoutImpl") || simple.equals("FeatureAlphaDropoutImpl")) {
            return dropoutAttrs(m);
        }
        if (simple.equals("LSTMImpl") || simple.equals("GRUImpl") || simple.equals("RNNImpl")) {
            return rnnStackedAttrs(m);
        }
        if (simple.equals("LSTMCellImpl") || simple.equals("GRUCellImpl") || simple.equals("RNNCellImpl")) {
            return rnnCellAttrs(m);
        }
        if (simple.equals("EmbeddingImpl")) {
            return embeddingAttrs(m);
        }
        if (simple.equals("MultiheadAttentionImpl")) {
            return mhaAttrs(m);
        }
        if (simple.equals("FlattenImpl") || simple.equals("UnflattenImpl")) {
            return "";
        }
        return "";
    }

    // ---- per-layer-type attribute fetchers ----

    private static String linearAttrs(Module m) {
        try {
            org.bytedeco.pytorch.LinearImpl t = m.as(org.bytedeco.pytorch.LinearImpl.class);
            if (t == null) return "";
            org.bytedeco.pytorch.LinearOptions o = t.options();
            return "in_features=" + longValue(o.in_features())
                + ", out_features=" + longValue(o.out_features())
                + ", bias=" + boolValue(o.bias());
        } catch (Throwable e) { return ""; }
    }

    /** Conv2d-style attrs. Conv2dImpl/Conv3dImpl via Base. */
    private static String convAttrs(Module m, int spatialDims) {
        try {
            org.bytedeco.pytorch.Conv2dImpl t = m.as(org.bytedeco.pytorch.Conv2dImpl.class);
            if (t == null) return "";
            org.bytedeco.pytorch.DetailConv2dOptions o = t.options();
            StringBuilder sb = new StringBuilder();
            sb.append("in_channels=").append(longValue(o.in_channels()));
            sb.append(", out_channels=").append(longValue(o.out_channels()));
            sb.append(", kernel_size=").append(expandArrayToString(o.kernel_size()));
            sb.append(", stride=").append(expandArrayToString(o.stride()));
            sb.append(", padding=").append(paddingToString(o.padding()));
            sb.append(", dilation=").append(expandArrayToString(o.dilation()));
            sb.append(", groups=").append(longValue(o.groups()));
            sb.append(", bias=").append(boolValue(o.bias()));
            return sb.toString();
        } catch (Throwable e) { return ""; }
    }

    private static String convTransposeAttrs(Module m) {
        return convAttrs(m, 2);
    }

    private static String batchNormAttrs(Module m) {
        try {
            org.bytedeco.pytorch.BatchNorm2dImplBase t = m.as(org.bytedeco.pytorch.BatchNorm2dImplBase.class);
            if (t == null) return "";
            org.bytedeco.pytorch.BatchNormOptions o = t.options();
            StringBuilder sb = new StringBuilder();
            sb.append("num_features=").append(longValue(o.num_features()));
            sb.append(", eps=").append(doubleValue(o.eps()));
            sb.append(", momentum=").append(o.momentum());
            sb.append(", affine=").append(boolValue(o.affine()));
            sb.append(", track_running_stats=").append(boolValue(o.track_running_stats()));
            return sb.toString();
        } catch (Throwable e) { return ""; }
    }

    private static String instanceNormAttrs(Module m) {
        try {
            org.bytedeco.pytorch.InstanceNorm2dImplBase t =
                m.as(org.bytedeco.pytorch.InstanceNorm2dImplBase.class);
            if (t == null) return "";
            org.bytedeco.pytorch.InstanceNormOptions o = t.options();
            StringBuilder sb = new StringBuilder();
            sb.append("num_features=").append(longValue(o.num_features()));
            sb.append(", eps=").append(doubleValue(o.eps()));
            sb.append(", momentum=").append(o.momentum());
            sb.append(", affine=").append(boolValue(o.affine()));
            sb.append(", track_running_stats=").append(boolValue(o.track_running_stats()));
            return sb.toString();
        } catch (Throwable e) { return ""; }
    }

    private static String groupNormAttrs(Module m) {
        try {
            org.bytedeco.pytorch.GroupNormImpl t = m.as(org.bytedeco.pytorch.GroupNormImpl.class);
            if (t == null) return "";
            org.bytedeco.pytorch.GroupNormOptions o = t.options();
            StringBuilder sb = new StringBuilder();
            sb.append("num_groups=").append(longValue(o.num_groups()));
            sb.append(", num_channels=").append(longValue(o.num_channels()));
            sb.append(", eps=").append(doubleValue(o.eps()));
            sb.append(", affine=").append(boolValue(o.affine()));
            return sb.toString();
        } catch (Throwable e) { return ""; }
    }

    private static String layerNormAttrs(Module m) {
        try {
            org.bytedeco.pytorch.LayerNormImpl t = m.as(org.bytedeco.pytorch.LayerNormImpl.class);
            if (t == null) return "";
            org.bytedeco.pytorch.LayerNormOptions o = t.options();
            StringBuilder sb = new StringBuilder();
            sb.append("normalized_shape=").append(longArrayToString(o.normalized_shape().get()));
            sb.append(", eps=").append(doubleValue(o.eps()));
            sb.append(", elementwise_affine=").append(boolValue(o.elementwise_affine()));
            return sb.toString();
        } catch (Throwable e) { return ""; }
    }

    private static String dropoutAttrs(Module m) {
        try {
            org.bytedeco.pytorch.DropoutImpl t = m.as(org.bytedeco.pytorch.DropoutImpl.class);
            if (t == null) return "";
            org.bytedeco.pytorch.DropoutOptions o = t.options();
            return "p=" + doubleValue(o.p()) + ", inplace=" + boolValue(o.inplace());
        } catch (Throwable e) { return ""; }
    }

    private static String rnnStackedAttrs(Module m) {
        try {
            StringBuilder sb = new StringBuilder();
            org.bytedeco.pytorch.LSTMImpl l = m.as(org.bytedeco.pytorch.LSTMImpl.class);
            if (l != null) {
                org.bytedeco.pytorch.LSTMOptions o = l.options();
                sb.append("input_size=").append(longValue(o.input_size()));
                sb.append(", hidden_size=").append(longValue(o.hidden_size()));
                sb.append(", num_layers=").append(longValue(o.num_layers()));
                sb.append(", bias=").append(boolValue(o.bias()));
                return sb.toString();
            }
            org.bytedeco.pytorch.GRUImpl g = m.as(org.bytedeco.pytorch.GRUImpl.class);
            if (g != null) {
                org.bytedeco.pytorch.GRUOptions o = g.options();
                sb.append("input_size=").append(longValue(o.input_size()));
                sb.append(", hidden_size=").append(longValue(o.hidden_size()));
                sb.append(", num_layers=").append(longValue(o.num_layers()));
                sb.append(", bias=").append(boolValue(o.bias()));
                return sb.toString();
            }
            org.bytedeco.pytorch.RNNImpl r = m.as(org.bytedeco.pytorch.RNNImpl.class);
            if (r != null) {
                org.bytedeco.pytorch.RNNOptions o = r.options();
                sb.append("input_size=").append(longValue(o.input_size()));
                sb.append(", hidden_size=").append(longValue(o.hidden_size()));
                sb.append(", num_layers=").append(longValue(o.num_layers()));
                sb.append(", bias=").append(boolValue(o.bias()));
                return sb.toString();
            }
        } catch (Throwable e) { return ""; }
        return "";
    }

    private static String rnnCellAttrs(Module m) {
        try {
            StringBuilder sb = new StringBuilder();
            org.bytedeco.pytorch.LSTMCellImpl l = m.as(org.bytedeco.pytorch.LSTMCellImpl.class);
            if (l != null) {
                org.bytedeco.pytorch.RNNCellOptionsBase o = l.options_base();
                sb.append("input_size=").append(longValue(o.input_size()));
                sb.append(", hidden_size=").append(longValue(o.hidden_size()));
                sb.append(", bias=").append(boolValue(o.bias()));
                return sb.toString();
            }
            org.bytedeco.pytorch.GRUCellImpl g = m.as(org.bytedeco.pytorch.GRUCellImpl.class);
            if (g != null) {
                org.bytedeco.pytorch.RNNCellOptionsBase o = g.options_base();
                sb.append("input_size=").append(longValue(o.input_size()));
                sb.append(", hidden_size=").append(longValue(o.hidden_size()));
                sb.append(", bias=").append(boolValue(o.bias()));
                return sb.toString();
            }
            org.bytedeco.pytorch.RNNCellImpl r = m.as(org.bytedeco.pytorch.RNNCellImpl.class);
            if (r != null) {
                org.bytedeco.pytorch.RNNCellOptionsBase o = r.options_base();
                sb.append("input_size=").append(longValue(o.input_size()));
                sb.append(", hidden_size=").append(longValue(o.hidden_size()));
                sb.append(", bias=").append(boolValue(o.bias()));
                return sb.toString();
            }
        } catch (Throwable e) { return ""; }
        return "";
    }

    private static String embeddingAttrs(Module m) {
        try {
            org.bytedeco.pytorch.EmbeddingImpl t = m.as(org.bytedeco.pytorch.EmbeddingImpl.class);
            if (t == null) return "";
            org.bytedeco.pytorch.EmbeddingOptions o = t.options();
            return "num_embeddings=" + longValue(o.num_embeddings())
                + ", embedding_dim=" + longValue(o.embedding_dim());
        } catch (Throwable e) { return ""; }
    }

    private static String mhaAttrs(Module m) {
        try {
            org.bytedeco.pytorch.MultiheadAttentionImpl t =
                m.as(org.bytedeco.pytorch.MultiheadAttentionImpl.class);
            if (t == null) return "";
            Class<?> c = t.getClass();
            long embed = ((Number) c.getMethod("embed_dim").invoke(t)).longValue();
            long heads = ((Number) c.getMethod("num_heads").invoke(t)).longValue();
            return "embed_dim=" + embed + ", num_heads=" + heads;
        } catch (Throwable e) { return ""; }
    }

    // ---- typed pointer -> Java primitive helpers ----
    // libtorch wraps most scalar option fields as LongPointer /
    // BoolPointer (ExpandingArray). The wrapper's capacity() is
    // uninitialized (0) but .get(i) reads the correct value; we
    // ignore capacity and just call .get(0) for scalars / .get(i)
    // for arrays.

    private static long longValue(LongPointer p) {
        if (p == null) return 0;
        try { return p.get(0); } catch (Throwable e) { return 0; }
    }

    private static boolean boolValue(org.bytedeco.javacpp.BoolPointer p) {
        if (p == null) return false;
        try { return p.get(0); } catch (Throwable e) { return false; }
    }

    private static double doubleValue(org.bytedeco.javacpp.DoublePointer p) {
        if (p == null) return 0.0;
        try { return p.get(0); } catch (Throwable e) { return 0.0; }
    }

    /**
     * Conv2dOptions' kernel_size/stride/dilation return a LongPointer
     * (ExpandingArray). Read up to 4 elements; tensor data is usually
     * 1D or 2D. We can't trust .capacity() so we just try a few
     * indices and stop on the first failure.
     */
    private static String expandArrayToString(LongPointer p) {
        if (p == null) return "?";
        try {
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < 4; i++) {
                long v;
                try {
                    v = p.get(i);
                } catch (Throwable e) {
                    break;
                }
                if (i > 0) sb.append(", ");
                sb.append(v);
            }
            sb.append(']');
            return sb.toString();
        } catch (Throwable e) { return "?"; }
    }

    private static String paddingToString(org.bytedeco.pytorch.Conv2dPadding p) {
        try {
            return String.valueOf(p);
        } catch (Throwable e) { return "?"; }
    }

    private static String longArrayToString(long[] a) {
        if (a == null) return "[]";
        if (a.length == 1) return Long.toString(a[0]);
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < a.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(a[i]);
        }
        sb.append(']');
        return sb.toString();
    }

    private static String formatParameter(Tensor p) {
        if (p == null) return "null";
        // parameters() returns a TensorVector; each entry is a Tensor
        // with requires_grad=True. Skip the per-element dump to keep
        // big-weight matrices from spamming the console; just show
        // shape / dtype / device / numel.
        try {
            StringBuilder sb = new StringBuilder();
            sb.append("Tensor[shape=").append(formatShape(p));
            sb.append(", dtype=").append(formatDtype(p));
            sb.append(", device=").append(formatDevice(p));
            sb.append(", numel=").append(p.numel());
            sb.append("]");
            return sb.toString();
        } catch (Throwable e) {
            return "<param-err>";
        }
    }

    private static String formatShape(Tensor t) {
        try {
            long[] sizes = org.bytedeco.pytorch.ModulePrinterShapeHelper.sizesAsArray(t.sizes());
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < sizes.length; i++) {
                if (i > 0) sb.append(", ");
                sb.append(sizes[i]);
            }
            sb.append(']');
            return sb.toString();
        } catch (Throwable e) { return "?"; }
    }

    private static String formatDtype(Tensor t) {
        try {
            String s = String.valueOf(t.scalar_type());
            int dot = s.lastIndexOf('.');
            return (dot >= 0 ? s.substring(dot + 1) : s);
        } catch (Throwable e) { return "?"; }
    }

    private static String formatDevice(Tensor t) {
        try {
            // Device's Pointer.toString looks like
            //   "org.bytedeco.pytorch.Device[type=cpu, index=0]"
            // or "...Device[cuda:0]" or the verbose
            // "org.bytedeco.pytorch.Device[address=0x...,position=...,limit=...,capacity=...,deallocator=...]"
            // form. Show the type if available; else "?" since the
            // verbose form doesn't tell us cpu vs cuda without a roundtrip.
            String s = String.valueOf(t.device());
            int lBracket = s.indexOf('[');
            int rBracket = s.lastIndexOf(']');
            if (lBracket >= 0 && rBracket > lBracket) {
                String body = s.substring(lBracket + 1, rBracket);
                if (body.startsWith("address=")) {
                    return "?";  // verbose form, can't easily tell cpu/cuda
                }
                int typeIdx = body.indexOf("type=");
                if (typeIdx >= 0) {
                    int comma = body.indexOf(',', typeIdx);
                    return body.substring(typeIdx + 5, comma >= 0 ? comma : body.length());
                }
                return body;  // e.g. "cuda:0"
            }
            return s;
        } catch (Throwable e) {
            return "?";
        }
    }

    private static Module[] safeChildren(Module m) {
        try {
            return m.children().get();
        } catch (Throwable t) {
            return null;
        }
    }

    private static TensorVector safeParameters(Module m) {
        try {
            TensorVector v = m.parameters();
            return v;
        } catch (Throwable t) {
            return null;
        }
    }
}
