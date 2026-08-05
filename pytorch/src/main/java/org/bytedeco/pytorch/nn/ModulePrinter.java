package org.bytedeco.pytorch.nn;
import org.bytedeco.pytorch.data.*;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.nn.modules.*;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.enumtype.*;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.nn.modules.container.SharedModuleVector;
import org.bytedeco.pytorch.nn.modules.container.StringSharedModuleDict;
import org.bytedeco.pytorch.nn.modules.container.StringSharedModuleDictItem;
import org.bytedeco.pytorch.plot.vista.ModuleDiscovery;

/**
 * Mirrors Python {@code print(model)} for JavaCPP {@link Module}.
 *
 * <pre>
 * torch::nn::SequentialImpl(
 *   (0): torch::nn::LinearImpl(in_features=64, out_features=128, bias=true)
 *   (1): torch::nn::ReLUImpl
 *   (2): torch::nn::DropoutImpl(p=0.1, inplace=false)
 *   (3): torch::nn::LinearImpl(in_features=128, out_features=64, bias=true)
 * )
 *
 * MmoeLikeBlock(
 *   (experts): torch::nn::ModuleListImpl(
 *     (0): torch::nn::LinearImpl(...)
 *     (1): torch::nn::LinearImpl(...)
 *   )
 *   (gate): torch::nn::SequentialImpl(
 *     (0): torch::nn::LinearImpl(...)
 *     (1): torch::nn::SoftmaxImpl
 *   )
 * )
 * </pre>
 *
 * <p>Fixes vs earlier version:
 * <ul>
 *   <li>Recursive indent so nested containers nest cleanly</li>
 *   <li>Prefer {@code named_children()} keys (branch names) over bare indices</li>
 *   <li>Demangle JavaCPP {@code JavaCPP_torch_0003a_...} names to {@code torch::nn::...}</li>
 * </ul>
 */
public final class ModulePrinter {

    private static final String INDENT = "  ";

    private ModulePrinter() {}

    public static String format(Module m) {
        return format(m, 0);
    }

    /**
     * @param depth nesting level (0 = root). Controls indent for children lines
     *              and the closing parenthesis of multi-child modules.
     */
    private static String format(Module m, int depth) {
        if (m == null) return "null";
        StringBuilder sb = new StringBuilder();
        String typeName = getTypeName(m);
        sb.append(typeName);
        String attrs = describeAttrs(m, typeName);
        if (!attrs.isEmpty()) {
            sb.append('(').append(attrs).append(')');
        }

        NamedChild[] kids = safeNamedChildren(m);
        if (kids != null && kids.length > 0) {
            sb.append("(\n");
            String childIndent = repeat(INDENT, depth + 1);
            String closeIndent = repeat(INDENT, depth);
            for (int i = 0; i < kids.length; i++) {
                NamedChild c = kids[i];
                sb.append(childIndent);
                sb.append('(').append(c.key).append("): ");
                // format child at depth+1; multi-line child body already includes
                // its own deeper indents, but the first line continues after ": ".
                String childFmt = format(c.module, depth + 1);
                // If child is multi-line, indent continuation lines to align under
                // the content after the parent prefix. Child's internal lines are
                // already absolute from depth+1; we only need to ensure that when
                // we embed them, lines after the first keep their leading spaces
                // (they already do — format returns full lines with indent).
                // However the first line of a multi-line child is the type name
                // (no leading indent in format's return — indent is only for
                // children rows). So multi-line child looks like:
                //   TypeName(\n
                //     (0): ...\n
                //   )
                // which is correct when appended after "  (key): ".
                sb.append(childFmt);
                sb.append('\n');
            }
            sb.append(closeIndent).append(')');
        }
        return sb.toString();
    }

    // ---- type name ----------------------------------------------------------

    /**
     * Prefer C++ {@code Module::name()} (e.g. {@code torch::nn::LinearImpl}).
     * Demangle JavaCPP-encoded names like
     * {@code JavaCPP_torch_0003a_0003ann_0003a_0003aSequentialImpl}
     * → {@code torch::nn::SequentialImpl}.
     */
    private static String getTypeName(Module m) {
        if (m == null) return "null";
        try {
            return ModuleDiscovery.typeName(m);
        } catch (Throwable ignored) {
            try { return m.getClass().getSimpleName(); } catch (Throwable e) { return "Module"; }
        }
    }

    /**
     * Inverse of JavaCPP {@code Generator.mangle}:
     * <ul>
     *   <li>{@code _1} → {@code _}</li>
     *   <li>{@code _2} → {@code ;}</li>
     *   <li>{@code _3} → {@code [}</li>
     *   <li>{@code _0XXXX} (4 hex digits) → char code</li>
     *   <li>lone {@code _} → {@code .} (or {@code /})</li>
     * </ul>
     * Also strips a leading {@code JavaCPP_} prefix.
     * Example: {@code JavaCPP_torch_0003a_0003ann_0003a_0003aSequentialImpl}
     * → {@code torch::nn::SequentialImpl}.
     */
    static String demangleTypeName(String s) {
        if (s == null || s.isEmpty()) return s;
        if (s.startsWith("JavaCPP_")) {
            s = s.substring("JavaCPP_".length());
        }
        // Fast path: already clean C++ / Java name
        if (s.indexOf('_') < 0) return s;
        if (s.indexOf("::") >= 0 && s.indexOf("_0") < 0 && s.indexOf("_1") < 0) return s;

        StringBuilder out = new StringBuilder(s.length());
        for (int i = 0; i < s.length(); ) {
            char c = s.charAt(i);
            if (c != '_') {
                out.append(c);
                i++;
                continue;
            }
            // Underscore escape
            if (i + 1 >= s.length()) {
                out.append('.'); // trailing lone underscore
                i++;
                continue;
            }
            char n = s.charAt(i + 1);
            if (n == '1') {
                out.append('_');
                i += 2;
            } else if (n == '2') {
                out.append(';');
                i += 2;
            } else if (n == '3') {
                out.append('[');
                i += 2;
            } else if (n == '0'
                    && i + 6 <= s.length()
                    && isHexDigit(s.charAt(i + 2))
                    && isHexDigit(s.charAt(i + 3))
                    && isHexDigit(s.charAt(i + 4))
                    && isHexDigit(s.charAt(i + 5))) {
                int code = Integer.parseInt(s.substring(i + 2, i + 6), 16);
                if (code >= 0x20 && code <= 0x7e) {
                    out.append((char) code);
                } else {
                    // keep raw if non-printable
                    out.append(s, i, i + 6);
                }
                i += 6;
            } else {
                // lone '_' stands for '.' or '/'
                out.append('.');
                i += 1;
            }
        }
        return out.toString();
    }

    private static boolean isHexDigit(char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
    }

    private static String simpleName(String typeName) {
        if (typeName == null) return "";
        int cc = typeName.lastIndexOf("::");
        if (cc >= 0) return typeName.substring(cc + 2);
        int dot = typeName.lastIndexOf('.');
        if (dot >= 0) return typeName.substring(dot + 1);
        return typeName;
    }

    // ---- children -----------------------------------------------------------

    private static final class NamedChild {
        final String key;
        final Module module;
        NamedChild(String key, Module module) {
            this.key = key;
            this.module = module;
        }
    }

    /**
     * Prefer {@link Module#named_children()} so ModuleDict / register_module
     * branches show real keys (experts, gate, stem, …). Fall back to
     * {@link Module#children()} with numeric indices.
     */
    private static NamedChild[] safeNamedChildren(Module m) {
        // Try named_children first
        try {
            StringSharedModuleDict dict = m.named_children();
            if (dict != null && !dict.isNull()) {
                long n = dict.size();
                if (n > 0) {
                    NamedChild[] out = new NamedChild[(int) Math.min(n, Integer.MAX_VALUE)];
                    int filled = 0;
                    for (int i = 0; i < out.length; i++) {
                        StringSharedModuleDictItem item = dict.get(i);
                        if (item == null || item.isNull()) continue;
                        String key;
                        try {
                            BytePointer k = item.key();
                            key = (k != null && !k.isNull()) ? k.getString() : String.valueOf(i);
                        } catch (Throwable e) {
                            key = String.valueOf(i);
                        }
                        if (key == null || key.isEmpty()) key = String.valueOf(i);
                        Module child = item.value();
                        if (child == null || child.isNull()) continue;
                        // Recover typed Java peer when possible
                        child = recover(child);
                        out[filled++] = new NamedChild(key, child);
                    }
                    if (filled == out.length) return out;
                    if (filled > 0) return java.util.Arrays.copyOf(out, filled);
                }
            }
        } catch (Throwable ignored) {}

        // Fallback: children() with indices
        try {
            SharedModuleVector v = m.children();
            if (v == null || v.isNull() || v.size() == 0) return null;
            long n = v.size();
            NamedChild[] out = new NamedChild[(int) Math.min(n, Integer.MAX_VALUE)];
            int filled = 0;
            for (int i = 0; i < out.length; i++) {
                Module child = v.get(i);
                if (child == null || child.isNull()) continue;
                child = recover(child);
                out[filled++] = new NamedChild(String.valueOf(i), child);
            }
            if (filled == 0) return null;
            if (filled == out.length) return out;
            return java.util.Arrays.copyOf(out, filled);
        } catch (Throwable t) {
            return null;
        }
    }

    private static Module recover(Module m) {
        try {
            Module r = ModuleAsHelper.recover(m);
            return r != null ? r : m;
        } catch (Throwable e) {
            return m;
        }
    }

    // ---- attributes ---------------------------------------------------------

    private static String describeAttrs(Module m, String typeName) {
        String simple = simpleName(typeName);
        // Also try Java simple class name if demangled name is still odd
        if (simple.startsWith("JavaCPP") || simple.isEmpty()) {
            try { simple = m.getClass().getSimpleName(); } catch (Throwable ignored) {}
        }
        if (simple.equals("LinearImpl") || simple.equals("BilinearImpl")) {
            return linearAttrs(m);
        }
        if (simple.equals("Conv1dImpl") || simple.equals("Conv2dImpl") || simple.equals("Conv3dImpl")) {
            return convAttrs(m);
        }
        if (simple.equals("ConvTranspose1dImpl") || simple.equals("ConvTranspose2dImpl")
                || simple.equals("ConvTranspose3dImpl")) {
            return convAttrs(m);
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
        if (simple.equals("EmbeddingBagImpl")) {
            return embeddingBagAttrs(m);
        }
        if (simple.equals("MultiheadAttentionImpl")) {
            return mhaAttrs(m);
        }
        if (simple.equals("SoftmaxImpl") || simple.equals("LogSoftmaxImpl")) {
            return softmaxAttrs(m);
        }
        // Containers and custom modules: no scalar attrs
        return "";
    }

    private static String linearAttrs(Module m) {
        try {
            org.bytedeco.pytorch.nn.modules.LinearImpl t =
                    m.as(org.bytedeco.pytorch.nn.modules.LinearImpl.class);
            if (t == null) return "";
            org.bytedeco.pytorch.nn.options.LinearOptions o = t.options();
            return "in_features=" + longValue(o.in_features())
                    + ", out_features=" + longValue(o.out_features())
                    + ", bias=" + boolValue(o.bias());
        } catch (Throwable e) { return ""; }
    }

    private static String convAttrs(Module m) {
        try {
            org.bytedeco.pytorch.nn.modules.Conv2dImpl t =
                    m.as(org.bytedeco.pytorch.nn.modules.Conv2dImpl.class);
            if (t == null) return "";
            org.bytedeco.pytorch.nn.options.DetailConv2dOptions o = t.options();
            return "in_channels=" + longValue(o.in_channels())
                    + ", out_channels=" + longValue(o.out_channels())
                    + ", kernel_size=" + expandArrayToString(o.kernel_size())
                    + ", stride=" + expandArrayToString(o.stride())
                    + ", padding=" + paddingToString(o.padding())
                    + ", dilation=" + expandArrayToString(o.dilation())
                    + ", groups=" + longValue(o.groups())
                    + ", bias=" + boolValue(o.bias());
        } catch (Throwable e) { return ""; }
    }

    private static String batchNormAttrs(Module m) {
        try {
            org.bytedeco.pytorch.nn.modules.BatchNorm2dImplBase t =
                    m.as(org.bytedeco.pytorch.nn.modules.BatchNorm2dImplBase.class);
            if (t == null) return "";
            org.bytedeco.pytorch.nn.options.BatchNormOptions o = t.options();
            return "num_features=" + longValue(o.num_features())
                    + ", eps=" + doubleValue(o.eps())
                    + ", momentum=" + o.momentum()
                    + ", affine=" + boolValue(o.affine())
                    + ", track_running_stats=" + boolValue(o.track_running_stats());
        } catch (Throwable e) { return ""; }
    }

    private static String instanceNormAttrs(Module m) {
        try {
            org.bytedeco.pytorch.nn.modules.InstanceNorm2dImplBase t =
                    m.as(org.bytedeco.pytorch.nn.modules.InstanceNorm2dImplBase.class);
            if (t == null) return "";
            org.bytedeco.pytorch.nn.options.InstanceNormOptions o = t.options();
            return "num_features=" + longValue(o.num_features())
                    + ", eps=" + doubleValue(o.eps())
                    + ", momentum=" + o.momentum()
                    + ", affine=" + boolValue(o.affine())
                    + ", track_running_stats=" + boolValue(o.track_running_stats());
        } catch (Throwable e) { return ""; }
    }

    private static String groupNormAttrs(Module m) {
        try {
            org.bytedeco.pytorch.nn.modules.GroupNormImpl t =
                    m.as(org.bytedeco.pytorch.nn.modules.GroupNormImpl.class);
            if (t == null) return "";
            org.bytedeco.pytorch.nn.options.GroupNormOptions o = t.options();
            return "num_groups=" + longValue(o.num_groups())
                    + ", num_channels=" + longValue(o.num_channels())
                    + ", eps=" + doubleValue(o.eps())
                    + ", affine=" + boolValue(o.affine());
        } catch (Throwable e) { return ""; }
    }

    private static String layerNormAttrs(Module m) {
        try {
            org.bytedeco.pytorch.nn.modules.LayerNormImpl t =
                    m.as(org.bytedeco.pytorch.nn.modules.LayerNormImpl.class);
            if (t == null) return "";
            org.bytedeco.pytorch.nn.options.LayerNormOptions o = t.options();
            return "normalized_shape=" + longArrayToString(o.normalized_shape().get())
                    + ", eps=" + doubleValue(o.eps())
                    + ", elementwise_affine=" + boolValue(o.elementwise_affine());
        } catch (Throwable e) { return ""; }
    }

    private static String dropoutAttrs(Module m) {
        try {
            // Try concrete DropoutImpl first; 2d/3d share options shape
            org.bytedeco.pytorch.nn.modules.DropoutImpl t =
                    m.as(org.bytedeco.pytorch.nn.modules.DropoutImpl.class);
            if (t != null) {
                org.bytedeco.pytorch.nn.options.DropoutOptions o = t.options();
                return "p=" + doubleValue(o.p()) + ", inplace=" + boolValue(o.inplace());
            }
            org.bytedeco.pytorch.nn.modules.Dropout2dImpl t2 =
                    m.as(org.bytedeco.pytorch.nn.modules.Dropout2dImpl.class);
            if (t2 != null) {
                org.bytedeco.pytorch.nn.options.DropoutOptions o = t2.options();
                return "p=" + doubleValue(o.p()) + ", inplace=" + boolValue(o.inplace());
            }
            org.bytedeco.pytorch.nn.modules.Dropout3dImpl t3 =
                    m.as(org.bytedeco.pytorch.nn.modules.Dropout3dImpl.class);
            if (t3 != null) {
                org.bytedeco.pytorch.nn.options.DropoutOptions o = t3.options();
                return "p=" + doubleValue(o.p()) + ", inplace=" + boolValue(o.inplace());
            }
        } catch (Throwable e) { return ""; }
        return "";
    }

    private static String rnnStackedAttrs(Module m) {
        try {
            org.bytedeco.pytorch.nn.modules.LSTMImpl l =
                    m.as(org.bytedeco.pytorch.nn.modules.LSTMImpl.class);
            if (l != null) {
                org.bytedeco.pytorch.nn.options.LSTMOptions o = l.options();
                return "input_size=" + longValue(o.input_size())
                        + ", hidden_size=" + longValue(o.hidden_size())
                        + ", num_layers=" + longValue(o.num_layers())
                        + ", bias=" + boolValue(o.bias());
            }
            org.bytedeco.pytorch.nn.modules.GRUImpl g =
                    m.as(org.bytedeco.pytorch.nn.modules.GRUImpl.class);
            if (g != null) {
                org.bytedeco.pytorch.nn.options.GRUOptions o = g.options();
                return "input_size=" + longValue(o.input_size())
                        + ", hidden_size=" + longValue(o.hidden_size())
                        + ", num_layers=" + longValue(o.num_layers())
                        + ", bias=" + boolValue(o.bias());
            }
            org.bytedeco.pytorch.nn.modules.RNNImpl r =
                    m.as(org.bytedeco.pytorch.nn.modules.RNNImpl.class);
            if (r != null) {
                org.bytedeco.pytorch.nn.options.RNNOptions o = r.options();
                return "input_size=" + longValue(o.input_size())
                        + ", hidden_size=" + longValue(o.hidden_size())
                        + ", num_layers=" + longValue(o.num_layers())
                        + ", bias=" + boolValue(o.bias());
            }
        } catch (Throwable e) { return ""; }
        return "";
    }

    private static String rnnCellAttrs(Module m) {
        try {
            org.bytedeco.pytorch.nn.modules.LSTMCellImpl l =
                    m.as(org.bytedeco.pytorch.nn.modules.LSTMCellImpl.class);
            if (l != null) {
                org.bytedeco.pytorch.nn.options.RNNCellOptionsBase o = l.options_base();
                return "input_size=" + longValue(o.input_size())
                        + ", hidden_size=" + longValue(o.hidden_size())
                        + ", bias=" + boolValue(o.bias());
            }
            org.bytedeco.pytorch.nn.modules.GRUCellImpl g =
                    m.as(org.bytedeco.pytorch.nn.modules.GRUCellImpl.class);
            if (g != null) {
                org.bytedeco.pytorch.nn.options.RNNCellOptionsBase o = g.options_base();
                return "input_size=" + longValue(o.input_size())
                        + ", hidden_size=" + longValue(o.hidden_size())
                        + ", bias=" + boolValue(o.bias());
            }
            org.bytedeco.pytorch.nn.modules.RNNCellImpl r =
                    m.as(org.bytedeco.pytorch.nn.modules.RNNCellImpl.class);
            if (r != null) {
                org.bytedeco.pytorch.nn.options.RNNCellOptionsBase o = r.options_base();
                return "input_size=" + longValue(o.input_size())
                        + ", hidden_size=" + longValue(o.hidden_size())
                        + ", bias=" + boolValue(o.bias());
            }
        } catch (Throwable e) { return ""; }
        return "";
    }

    private static String embeddingAttrs(Module m) {
        try {
            org.bytedeco.pytorch.nn.modules.EmbeddingImpl t =
                    m.as(org.bytedeco.pytorch.nn.modules.EmbeddingImpl.class);
            if (t == null) return "";
            org.bytedeco.pytorch.nn.options.EmbeddingOptions o = t.options();
            return "num_embeddings=" + longValue(o.num_embeddings())
                    + ", embedding_dim=" + longValue(o.embedding_dim());
        } catch (Throwable e) { return ""; }
    }

    private static String embeddingBagAttrs(Module m) {
        try {
            org.bytedeco.pytorch.nn.modules.EmbeddingBagImpl t =
                    m.as(org.bytedeco.pytorch.nn.modules.EmbeddingBagImpl.class);
            if (t == null) return "";
            org.bytedeco.pytorch.nn.options.EmbeddingBagOptions o = t.options();
            return "num_embeddings=" + longValue(o.num_embeddings())
                    + ", embedding_dim=" + longValue(o.embedding_dim());
        } catch (Throwable e) { return ""; }
    }

    private static String mhaAttrs(Module m) {
        try {
            org.bytedeco.pytorch.nn.modules.MultiheadAttentionImpl t =
                    m.as(org.bytedeco.pytorch.nn.modules.MultiheadAttentionImpl.class);
            if (t == null) return "";
            // Prefer options if available
            try {
                org.bytedeco.pytorch.nn.options.MultiheadAttentionOptions o = t.options();
                return "embed_dim=" + longValue(o.embed_dim())
                        + ", num_heads=" + longValue(o.num_heads());
            } catch (Throwable ignored) {}
            Class<?> c = t.getClass();
            long embed = ((Number) c.getMethod("embed_dim").invoke(t)).longValue();
            long heads = ((Number) c.getMethod("num_heads").invoke(t)).longValue();
            return "embed_dim=" + embed + ", num_heads=" + heads;
        } catch (Throwable e) { return ""; }
    }

    private static String softmaxAttrs(Module m) {
        try {
            org.bytedeco.pytorch.nn.modules.SoftmaxImpl t =
                    m.as(org.bytedeco.pytorch.nn.modules.SoftmaxImpl.class);
            if (t != null) {
                return "dim=" + longValue(t.options().dim());
            }
            org.bytedeco.pytorch.nn.modules.LogSoftmaxImpl lt =
                    m.as(org.bytedeco.pytorch.nn.modules.LogSoftmaxImpl.class);
            if (lt != null) {
                return "dim=" + longValue(lt.options().dim());
            }
        } catch (Throwable e) { return ""; }
        return "";
    }

    // ---- pointer helpers ----------------------------------------------------

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

    private static String expandArrayToString(LongPointer p) {
        if (p == null) return "?";
        try {
            StringBuilder sb = new StringBuilder("[");
            int count = 0;
            for (int i = 0; i < 4; i++) {
                long v;
                try {
                    v = p.get(i);
                } catch (Throwable e) {
                    break;
                }
                if (i > 0) sb.append(", ");
                sb.append(v);
                count++;
            }
            sb.append(']');
            return count == 0 ? "?" : sb.toString();
        } catch (Throwable e) { return "?"; }
    }

    private static String paddingToString(org.bytedeco.pytorch.enumtype.Conv2dPadding p) {
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

    private static String repeat(String s, int n) {
        if (n <= 0) return "";
        StringBuilder sb = new StringBuilder(s.length() * n);
        for (int i = 0; i < n; i++) sb.append(s);
        return sb.toString();
    }
}
