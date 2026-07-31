package org.bytedeco.pytorch.plot.vista;

import java.util.LinkedHashMap;
import java.util.Map;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.pytorch.StringTensorDict;
import org.bytedeco.pytorch.StringTensorDictItem;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.ModulePrinter;

/**
 * Collects {@link ModuleInfo} for a module node — parameters (non-recursive)
 * and scalar attributes.
 *
 * <p>Attribute extraction reuses the same option-field knowledge as
 * {@link ModulePrinter#format(Module)} (via its public format string and
 * direct option reads for common *Impl types). We deliberately do not
 * re-implement every attr path; instead we:
 * <ol>
 *   <li>Pull non-recursive {@code named_parameters(false)} shapes.</li>
 *   <li>Parse the parenthesised attr segment of {@link ModulePrinter#format}
 *       when present (e.g. {@code in_features=10, out_features=5, bias=true}).</li>
 *   <li>Fall back to empty attributes for custom modules.</li>
 * </ol>
 */
public final class ModuleInfoCollector {
    private ModuleInfoCollector() {}

    public static ModuleInfo collect(Module module) {
        if (module == null || module.isNull()) {
            return new ModuleInfo("null", null, null, null);
        }
        Module m = ModuleDiscovery.recover(module);
        String type = ModuleDiscovery.typeName(m);
        Map<String, ModuleInfo.ParamInfo> params = collectParameters(m);
        Map<String, Object> attrs = collectAttributes(m, type);
        String extra = null;
        // ModulePrinter.format already embeds attrs; keep extra_repr empty unless
        // we later expose pretty_print. Avoid double-encoding.
        return new ModuleInfo(type, params, attrs, extra);
    }

    private static Map<String, ModuleInfo.ParamInfo> collectParameters(Module m) {
        Map<String, ModuleInfo.ParamInfo> out = new LinkedHashMap<>();
        try {
            StringTensorDict dict = m.named_parameters(/*recurse=*/false);
            if (dict == null || dict.isNull()) return out;
            long n = dict.size();
            for (long i = 0; i < n; i++) {
                StringTensorDictItem item = dict.get(i);
                if (item == null || item.isNull()) continue;
                String key;
                try {
                    BytePointer k = item.key();
                    key = (k != null && !k.isNull()) ? k.getString() : ("param_" + i);
                } catch (Throwable e) {
                    key = "param_" + i;
                }
                Tensor t = item.value();
                if (t == null || t.isNull()) continue;
                // named_parameters returns @ByRef — retain before reading metadata
                try {
                    t = t.retainReference();
                } catch (Throwable ignored) {}
                long[] shape = TensorUtils.safeShape(t);
                boolean requiresGrad = false;
                try {
                    requiresGrad = t.requires_grad();
                } catch (Throwable ignored) {}
                out.put(key, new ModuleInfo.ParamInfo(shape, requiresGrad));
            }
        } catch (Throwable ignored) {}
        return out;
    }

    /**
     * Parse {@code ModulePrinter.format(m)} attr segment:
     * {@code TypeName(a=1, b=2)(...children...)} → attributes map.
     * Only the first parenthesised group that looks like {@code key=value} pairs
     * is consumed; nested children blocks are ignored.
     */
    private static Map<String, Object> collectAttributes(Module m, String typeName) {
        Map<String, Object> out = new LinkedHashMap<>();
        String formatted;
        try {
            formatted = ModulePrinter.format(m);
        } catch (Throwable e) {
            return out;
        }
        if (formatted == null || formatted.isEmpty()) return out;

        // Strip type name prefix
        String simple = ModuleDiscovery.simpleTypeName(m);
        int start = -1;
        if (formatted.startsWith(typeName + "(")) {
            start = typeName.length() + 1;
        } else if (formatted.startsWith(simple + "(")) {
            start = simple.length() + 1;
        } else {
            int paren = formatted.indexOf('(');
            if (paren > 0) start = paren + 1;
        }
        if (start < 0 || start >= formatted.length()) return out;

        // Find matching close for the attr group. If the module has children the
        // format is: Type(attrs)(\n  (0): ...\n) — attrs close before children open.
        // If no attrs: Type(\n  (0): ...) — first char inside paren is newline.
        if (formatted.charAt(start) == '\n') return out;

        int depth = 1;
        int end = start;
        for (int i = start; i < formatted.length(); i++) {
            char c = formatted.charAt(i);
            if (c == '(') depth++;
            else if (c == ')') {
                depth--;
                if (depth == 0) {
                    end = i;
                    break;
                }
            } else if (c == '\n' && depth == 1) {
                // children block starts — no scalar attrs
                return out;
            }
        }
        if (end <= start) return out;
        String body = formatted.substring(start, end).trim();
        if (body.isEmpty() || body.startsWith("(")) return out;

        // Split on commas not inside brackets
        for (String part : splitTopLevel(body, ',')) {
            String p = part.trim();
            if (p.isEmpty()) continue;
            int eq = p.indexOf('=');
            if (eq <= 0) continue;
            String key = p.substring(0, eq).trim();
            String raw = p.substring(eq + 1).trim();
            out.put(key, coerce(raw));
        }
        return out;
    }

    private static Object coerce(String raw) {
        if (raw == null) return null;
        if ("true".equalsIgnoreCase(raw)) return true;
        if ("false".equalsIgnoreCase(raw)) return false;
        try {
            if (raw.contains(".")) return Double.parseDouble(raw);
            return Long.parseLong(raw);
        } catch (NumberFormatException ignored) {}
        // strip quotes
        if (raw.length() >= 2
                && ((raw.charAt(0) == '"' && raw.charAt(raw.length() - 1) == '"')
                || (raw.charAt(0) == '\'' && raw.charAt(raw.length() - 1) == '\''))) {
            return raw.substring(1, raw.length() - 1);
        }
        return raw;
    }

    private static java.util.List<String> splitTopLevel(String s, char sep) {
        java.util.List<String> parts = new java.util.ArrayList<>();
        int depth = 0;
        int start = 0;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '(' || c == '[' || c == '{') depth++;
            else if (c == ')' || c == ']' || c == '}') depth = Math.max(0, depth - 1);
            else if (c == sep && depth == 0) {
                parts.add(s.substring(start, i));
                start = i + 1;
            }
        }
        parts.add(s.substring(start));
        return parts;
    }
}
