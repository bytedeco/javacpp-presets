/*
 * Additive-safe schema evolution checks for FeatureView versions.
 */
package org.bytedeco.pytorch.feature.lifecycle;

import org.bytedeco.pytorch.feature.core.FeatureView;
import org.bytedeco.pytorch.feature.core.Field;
import org.bytedeco.pytorch.feature.core.ValueType;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Detect additive vs breaking schema changes. */
public final class SchemaEvolution {

    public enum ChangeType {
        ADDITIVE, BREAKING, NONE
    }

    public static final class Diff {
        public final ChangeType type;
        public final List<String> added;
        public final List<String> removed;
        public final List<String> typeChanged;

        public Diff(ChangeType type, List<String> added, List<String> removed, List<String> typeChanged) {
            this.type = type;
            this.added = Collections.unmodifiableList(added);
            this.removed = Collections.unmodifiableList(removed);
            this.typeChanged = Collections.unmodifiableList(typeChanged);
        }

        public boolean breaking() {
            return type == ChangeType.BREAKING;
        }

        @Override
        public String toString() {
            return "SchemaDiff{type=" + type + ", +=" + added + ", -=" + removed + ", ~=" + typeChanged + "}";
        }
    }

    private SchemaEvolution() {}

    public static Diff diff(FeatureView from, FeatureView to) {
        Objects.requireNonNull(from, "from");
        Objects.requireNonNull(to, "to");
        Map<String, ValueType> a = new LinkedHashMap<>();
        Map<String, ValueType> b = new LinkedHashMap<>();
        for (Field f : from.schema()) a.put(f.name(), f.valueType());
        for (Field f : to.schema()) b.put(f.name(), f.valueType());
        List<String> added = new ArrayList<>();
        List<String> removed = new ArrayList<>();
        List<String> changed = new ArrayList<>();
        for (String k : b.keySet()) {
            if (!a.containsKey(k)) added.add(k);
            else if (a.get(k) != b.get(k)) changed.add(k);
        }
        for (String k : a.keySet()) {
            if (!b.containsKey(k)) removed.add(k);
        }
        ChangeType t;
        if (removed.isEmpty() && changed.isEmpty()) {
            t = added.isEmpty() ? ChangeType.NONE : ChangeType.ADDITIVE;
        } else {
            t = ChangeType.BREAKING;
        }
        return new Diff(t, added, removed, changed);
    }

    public static void requireCompatible(FeatureView from, FeatureView to) {
        Diff d = diff(from, to);
        if (d.breaking()) {
            throw new IllegalStateException("breaking schema change: " + d
                    + " — register a new FeatureView version");
        }
    }
}
