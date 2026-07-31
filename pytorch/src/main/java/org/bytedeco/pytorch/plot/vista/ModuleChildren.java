package org.bytedeco.pytorch.plot.vista;

import java.util.ArrayList;
import java.util.List;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.container.SharedModuleVector;
import org.bytedeco.pytorch.nn.modules.container.StringSharedModuleDict;
import org.bytedeco.pytorch.nn.modules.container.StringSharedModuleDictItem;

/**
 * Safe iteration over {@link Module#named_children()} / {@link Module#children()},
 * recovering typed Java peers via {@link ModuleAsHelper} (same pattern as
 * {@link org.bytedeco.pytorch.nn.ModulePrinter}).
 */
public final class ModuleChildren {
    private ModuleChildren() {}

    public static final class NamedChild {
        public final String key;
        public final Module module;

        public NamedChild(String key, Module module) {
            this.key = key;
            this.module = module;
        }
    }

    public static List<NamedChild> list(Module m) {
        List<NamedChild> out = new ArrayList<>();
        if (m == null || m.isNull()) return out;

        // Prefer named_children
        try {
            StringSharedModuleDict dict = m.named_children();
            if (dict != null && !dict.isNull() && dict.size() > 0) {
                long n = dict.size();
                for (long i = 0; i < n; i++) {
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
                    // named_children returns bare Module shared_ptrs — re-type to *Impl
                    // or the original Java peer (EmbeddingLayer / MLP / …).
                    child = ModuleDiscovery.concrete(child);
                    // Extra: if still bare, try the parent's Java field of the same name
                    // (register_module peers often stay reachable as typed fields).
                    if (child.getClass() == Module.class) {
                        Module fromField = typedField(m, key);
                        if (fromField != null) child = fromField;
                    }
                    out.add(new NamedChild(key, child));
                }
                if (!out.isEmpty()) return out;
            }
        } catch (Throwable ignored) {}

        // Fallback: children() with numeric indices
        try {
            SharedModuleVector v = m.children();
            if (v == null || v.isNull() || v.size() == 0) return out;
            long n = v.size();
            for (long i = 0; i < n; i++) {
                Module child = v.get(i);
                if (child == null || child.isNull()) continue;
                child = ModuleDiscovery.concrete(child);
                out.add(new NamedChild(String.valueOf(i), child));
            }
        } catch (Throwable ignored) {}
        return out;
    }

    /**
     * Look up a typed {@link Module} field on {@code parent} whose name matches
     * {@code key} or common Java field styles ({@code userEmbedding} for key
     * {@code userEmbedding}). Remembers the peer in {@link ModuleAsHelper}.
     */
    private static Module typedField(Module parent, String key) {
        if (parent == null || key == null || key.isEmpty()) return null;
        if (parent.getClass() == Module.class) return null;
        try {
            for (java.lang.reflect.Field f : parent.getClass().getDeclaredFields()) {
                if (!Module.class.isAssignableFrom(f.getType())) continue;
                String fn = f.getName();
                if (!fn.equals(key)
                        && !fn.equalsIgnoreCase(key)
                        && !fn.toLowerCase().replace("_", "").equals(key.toLowerCase().replace("_", ""))) {
                    continue;
                }
                f.setAccessible(true);
                Object v = f.get(parent);
                if (v instanceof Module) {
                    Module typed = (Module) v;
                    if (!typed.isNull() && typed.getClass() != Module.class) {
                        try {
                            org.bytedeco.pytorch.nn.ModuleAsHelper.remember(typed);
                        } catch (Throwable ignored) {}
                        return typed;
                    }
                }
            }
            // Also scan List<Module> fields (towers, predictLayers, …)
            for (java.lang.reflect.Field f : parent.getClass().getDeclaredFields()) {
                if (!java.util.List.class.isAssignableFrom(f.getType())) continue;
                f.setAccessible(true);
                Object v = f.get(parent);
                if (!(v instanceof java.util.List)) continue;
                java.util.List<?> list = (java.util.List<?>) v;
                // key tower_0 / predictLayer_1
                int us = key.lastIndexOf('_');
                if (us <= 0 || us == key.length() - 1) continue;
                String prefix = key.substring(0, us);
                int idx;
                try {
                    idx = Integer.parseInt(key.substring(us + 1));
                } catch (NumberFormatException nfe) {
                    continue;
                }
                String fn = f.getName().toLowerCase();
                if (!fn.contains(prefix.toLowerCase()) && !prefix.toLowerCase().contains(
                        fn.replace("list", "").replace("s", ""))) {
                    // loose: towers ↔ tower, predictLayers ↔ predictLayer
                    String stem = fn.endsWith("s") ? fn.substring(0, fn.length() - 1) : fn;
                    if (!key.toLowerCase().startsWith(stem) && !stem.startsWith(prefix.toLowerCase())) {
                        continue;
                    }
                }
                if (idx < 0 || idx >= list.size()) continue;
                Object elem = list.get(idx);
                if (elem instanceof Module) {
                    Module typed = (Module) elem;
                    if (!typed.isNull()) {
                        try {
                            org.bytedeco.pytorch.nn.ModuleAsHelper.remember(typed);
                        } catch (Throwable ignored) {}
                        return typed;
                    }
                }
            }
        } catch (Throwable ignored) {}
        return null;
    }

    public static boolean hasChildren(Module m) {
        return !list(m).isEmpty();
    }
}
