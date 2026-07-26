package org.bytedeco.pytorch.data.dataframe.vectorstore;

import java.util.Locale;
import java.util.Objects;

/**
 * Declared payload / metadata field for backends that need an explicit schema
 * (Redis RediSearch, optionally Milvus / OpenSearch mappings).
 *
 * <pre>{@code
 * RedisVectorStore.builder()
 *     .dim(768)
 *     .payloadField(PayloadField.text("title"))
 *     .payloadField(PayloadField.tag("category"))
 *     .payloadField(PayloadField.numeric("year"))
 *     .build();
 * }</pre>
 */
public final class PayloadField {

    public enum Type {
        /** Full-text (RediSearch {@code TEXT}, OpenSearch {@code text}). */
        TEXT,
        /** Exact-match tag / keyword (RediSearch {@code TAG}, OpenSearch {@code keyword}). */
        TAG,
        /** Numeric range (RediSearch {@code NUMERIC}, OpenSearch {@code double}). */
        NUMERIC,
        /** Boolean flag. */
        BOOLEAN,
        /** Opaque JSON / object blob (not indexed for filter, still stored). */
        JSON
    }

    private final String name;
    private final Type type;
    private final boolean sortable;
    private final boolean indexed;

    private PayloadField(String name, Type type, boolean sortable, boolean indexed) {
        this.name = Objects.requireNonNull(name, "name");
        if (name.isBlank()) throw new IllegalArgumentException("payload field name blank");
        this.type = type == null ? Type.TAG : type;
        this.sortable = sortable;
        this.indexed = indexed;
    }

    public static PayloadField of(String name, Type type) {
        return new PayloadField(name, type, false, true);
    }

    public static PayloadField text(String name) { return of(name, Type.TEXT); }
    public static PayloadField tag(String name) { return of(name, Type.TAG); }
    public static PayloadField numeric(String name) { return of(name, Type.NUMERIC); }
    public static PayloadField bool(String name) { return of(name, Type.BOOLEAN); }
    public static PayloadField json(String name) { return new PayloadField(name, Type.JSON, false, false); }

    /** Return a copy marked SORTABLE (RediSearch / range queries). */
    public PayloadField sortable() { return new PayloadField(name, type, true, indexed); }
    public PayloadField unindexed() { return new PayloadField(name, type, sortable, false); }

    public String name() { return name; }
    public Type type() { return type; }
    public boolean isSortable() { return sortable; }
    public boolean isIndexed() { return indexed; }

    /** RediSearch SCHEMA fragment tokens (name + type [+ SORTABLE]), empty if unindexed JSON. */
    public void appendRedisSchema(java.util.List<Object> args) {
        if (!indexed && type == Type.JSON) {
            // still declare as TAG NOINDEX so FT can RETURN it cleanly? skip — hash fields work unindexed
            return;
        }
        args.add(name);
        switch (type) {
            case TEXT -> {
                args.add("TEXT");
                if (sortable) args.add("SORTABLE");
            }
            case TAG -> {
                args.add("TAG");
                if (sortable) args.add("SORTABLE");
            }
            case NUMERIC -> {
                args.add("NUMERIC");
                if (sortable) args.add("SORTABLE");
            }
            case BOOLEAN -> {
                // RediSearch has no BOOL — store as TAG
                args.add("TAG");
                if (sortable) args.add("SORTABLE");
            }
            case JSON -> {
                args.add("TEXT");
                args.add("NOINDEX");
            }
        }
        if (!indexed && type != Type.JSON) {
            args.add("NOINDEX");
        }
    }

    /** OpenSearch property mapping fragment. */
    public java.util.Map<String, Object> openSearchProperty() {
        return switch (type) {
            case TEXT -> java.util.Map.of("type", "text");
            case TAG -> java.util.Map.of("type", "keyword");
            case NUMERIC -> java.util.Map.of("type", "double");
            case BOOLEAN -> java.util.Map.of("type", "boolean");
            case JSON -> java.util.Map.of("type", "object", "enabled", true);
        };
    }

    public static Type parseType(String s) {
        if (s == null || s.isBlank()) return Type.TAG;
        return switch (s.trim().toUpperCase(Locale.ROOT)) {
            case "TEXT", "STRING", "FULLTEXT" -> Type.TEXT;
            case "TAG", "KEYWORD", "KEYWORD_TYPE" -> Type.TAG;
            case "NUMERIC", "NUMBER", "FLOAT", "DOUBLE", "INT", "LONG" -> Type.NUMERIC;
            case "BOOL", "BOOLEAN" -> Type.BOOLEAN;
            case "JSON", "OBJECT" -> Type.JSON;
            default -> Type.TAG;
        };
    }

    @Override
    public String toString() {
        return "PayloadField{" + name + ":" + type + (sortable ? ",sortable" : "") + "}";
    }
}
