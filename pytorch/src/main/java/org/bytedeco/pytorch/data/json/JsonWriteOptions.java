package org.bytedeco.pytorch.data.json;

import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.Objects;

/**
 * Options for {@link JsonWriter}.
 */
public final class JsonWriteOptions {
    public enum NullHandling {
        /** Write JSON null. */
        WRITE_NULL,
        /** Omit null fields in objects (arrays still write null). */
        OMIT
    }

    private final Charset charset;
    private final boolean pretty;
    private final String indent;
    private final String lineSeparator;
    private final boolean escapeNonAscii;
    private final boolean escapeHtml;
    private final boolean escapeSolidus;
    private final boolean writeBom;
    private final boolean orderKeys;
    private final NullHandling nullHandling;
    private final boolean nanAsNull; // write NaN/Infinity as null instead of error
    private final int maxDepth;

    private JsonWriteOptions(Builder b) {
        this.charset = b.charset;
        this.pretty = b.pretty;
        this.indent = b.indent;
        this.lineSeparator = b.lineSeparator;
        this.escapeNonAscii = b.escapeNonAscii;
        this.escapeHtml = b.escapeHtml;
        this.escapeSolidus = b.escapeSolidus;
        this.writeBom = b.writeBom;
        this.orderKeys = b.orderKeys;
        this.nullHandling = b.nullHandling;
        this.nanAsNull = b.nanAsNull;
        this.maxDepth = b.maxDepth;
    }

    public static Builder builder() { return new Builder(); }
    public static JsonWriteOptions defaults() { return builder().build(); }
    public static JsonWriteOptions compact() { return builder().pretty(false).build(); }

    /**
     * Pretty-print preset. Named {@code prettyMode()} so it does not clash with
     * the instance getter {@link #pretty()}.
     */
    public static JsonWriteOptions prettyMode() {
        return builder().pretty(true).indent("  ").build();
    }

    public Charset charset() { return charset; }
    public boolean pretty() { return pretty; }
    public String indent() { return indent; }
    public String lineSeparator() { return lineSeparator; }
    public boolean escapeNonAscii() { return escapeNonAscii; }
    public boolean escapeHtml() { return escapeHtml; }
    public boolean escapeSolidus() { return escapeSolidus; }
    public boolean writeBom() { return writeBom; }
    public boolean orderKeys() { return orderKeys; }
    public NullHandling nullHandling() { return nullHandling; }
    public boolean nanAsNull() { return nanAsNull; }
    public int maxDepth() { return maxDepth; }

    public static final class Builder {
        private Charset charset = StandardCharsets.UTF_8;
        private boolean pretty = false;
        private String indent = "  ";
        private String lineSeparator = "\n";
        private boolean escapeNonAscii = false;
        private boolean escapeHtml = false;
        private boolean escapeSolidus = false;
        private boolean writeBom = false;
        private boolean orderKeys = false;
        private NullHandling nullHandling = NullHandling.WRITE_NULL;
        private boolean nanAsNull = false;
        private int maxDepth = 1000;

        public Builder charset(Charset v) { this.charset = Objects.requireNonNull(v); return this; }
        public Builder pretty(boolean v) { this.pretty = v; return this; }
        public Builder indent(String v) { this.indent = v == null ? "  " : v; return this; }
        public Builder indentSpaces(int n) {
            StringBuilder sb = new StringBuilder(Math.max(0, n));
            for (int i = 0; i < n; i++) sb.append(' ');
            this.indent = sb.toString();
            this.pretty = true;
            return this;
        }
        public Builder lineSeparator(String v) { this.lineSeparator = v == null ? "\n" : v; return this; }
        public Builder escapeNonAscii(boolean v) { this.escapeNonAscii = v; return this; }
        public Builder escapeHtml(boolean v) { this.escapeHtml = v; return this; }
        public Builder escapeSolidus(boolean v) { this.escapeSolidus = v; return this; }
        public Builder writeBom(boolean v) { this.writeBom = v; return this; }
        public Builder orderKeys(boolean v) { this.orderKeys = v; return this; }
        public Builder nullHandling(NullHandling v) {
            this.nullHandling = Objects.requireNonNull(v); return this;
        }
        public Builder nanAsNull(boolean v) { this.nanAsNull = v; return this; }
        public Builder maxDepth(int v) { this.maxDepth = v; return this; }

        public JsonWriteOptions build() { return new JsonWriteOptions(this); }
    }
}
