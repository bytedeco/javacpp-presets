package org.bytedeco.pytorch.data.json;

import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.Objects;
import java.util.function.BiConsumer;

/**
 * Options for {@link JsonParser}.
 *
 * <pre>
 *   JsonReadOptions opt = JsonReadOptions.builder()
 *       .allowComments(true)
 *       .allowTrailingCommas(true)
 *       .strict(false)
 *       .build();
 * </pre>
 */
public final class JsonReadOptions {
    public enum DuplicateKeyPolicy { FIRST, LAST, ERROR }

    private final Charset charset;
    private final boolean strict;
    private final boolean stripBom;
    private final boolean allowComments;
    private final boolean allowHashComments;
    private final boolean allowTrailingCommas;
    private final boolean allowSingleQuotes;
    private final boolean allowUnquotedKeys;
    private final boolean allowNanInfinity;
    private final boolean allowTrailingContent;
    private final boolean allowMultipleValues;
    private final boolean allowEmpty;
    private final boolean allowMultiLineJsonl;
    private final boolean skipBlankLines;
    private final String commentPrefix; // for JSONL full-line comments e.g. "#"
    private final int maxDepth;
    private final int maxStringLength;
    private final int maxArrayLength;
    private final int maxObjectKeys;
    private final int maxRows;   // JSONL: -1 unlimited
    private final int skipRows;  // JSONL
    private final DuplicateKeyPolicy duplicateKeyPolicy;
    private final JsonParser.JsonErrorHandler onError;

    private JsonReadOptions(Builder b) {
        this.charset = b.charset;
        this.strict = b.strict;
        this.stripBom = b.stripBom;
        this.allowComments = b.allowComments;
        this.allowHashComments = b.allowHashComments;
        this.allowTrailingCommas = b.allowTrailingCommas;
        this.allowSingleQuotes = b.allowSingleQuotes;
        this.allowUnquotedKeys = b.allowUnquotedKeys;
        this.allowNanInfinity = b.allowNanInfinity;
        this.allowTrailingContent = b.allowTrailingContent;
        this.allowMultipleValues = b.allowMultipleValues;
        this.allowEmpty = b.allowEmpty;
        this.allowMultiLineJsonl = b.allowMultiLineJsonl;
        this.skipBlankLines = b.skipBlankLines;
        this.commentPrefix = b.commentPrefix;
        this.maxDepth = b.maxDepth;
        this.maxStringLength = b.maxStringLength;
        this.maxArrayLength = b.maxArrayLength;
        this.maxObjectKeys = b.maxObjectKeys;
        this.maxRows = b.maxRows;
        this.skipRows = b.skipRows;
        this.duplicateKeyPolicy = b.duplicateKeyPolicy;
        this.onError = b.onError;
    }

    public static Builder builder() { return new Builder(); }
    public static JsonReadOptions defaults() { return builder().build(); }

    /**
     * Strict RFC 8259 preset (no comments, no trailing commas, etc.).
     * Named {@code strictMode()} so it does not clash with the instance getter {@link #strict()}.
     */
    public static JsonReadOptions strictMode() {
        return builder().strict(true)
            .allowComments(false).allowHashComments(false)
            .allowTrailingCommas(false).allowSingleQuotes(false)
            .allowUnquotedKeys(false).allowNanInfinity(false)
            .duplicateKeyPolicy(DuplicateKeyPolicy.ERROR)
            .build();
    }

    /** Lenient / JSON5-ish defaults for real-world data. */
    public static JsonReadOptions lenient() {
        return builder().strict(false)
            .allowComments(true).allowHashComments(true)
            .allowTrailingCommas(true).allowSingleQuotes(true)
            .allowUnquotedKeys(true).allowNanInfinity(true)
            .allowMultiLineJsonl(true)
            .duplicateKeyPolicy(DuplicateKeyPolicy.LAST)
            .build();
    }

    public Charset charset() { return charset; }
    public boolean strict() { return strict; }
    public boolean stripBom() { return stripBom; }
    public boolean allowComments() { return allowComments; }
    public boolean allowHashComments() { return allowHashComments; }
    public boolean allowTrailingCommas() { return allowTrailingCommas; }
    public boolean allowSingleQuotes() { return allowSingleQuotes; }
    public boolean allowUnquotedKeys() { return allowUnquotedKeys; }
    public boolean allowNanInfinity() { return allowNanInfinity; }
    public boolean allowTrailingContent() { return allowTrailingContent; }
    public boolean allowMultipleValues() { return allowMultipleValues; }
    public boolean allowEmpty() { return allowEmpty; }
    public boolean allowMultiLineJsonl() { return allowMultiLineJsonl; }
    public boolean skipBlankLines() { return skipBlankLines; }
    public String commentPrefix() { return commentPrefix; }
    public int maxDepth() { return maxDepth; }
    public int maxStringLength() { return maxStringLength; }
    public int maxArrayLength() { return maxArrayLength; }
    public int maxObjectKeys() { return maxObjectKeys; }
    public int maxRows() { return maxRows; }
    public int skipRows() { return skipRows; }
    public DuplicateKeyPolicy duplicateKeyPolicy() { return duplicateKeyPolicy; }
    public JsonParser.JsonErrorHandler onError() { return onError; }

    public static final class Builder {
        private Charset charset = StandardCharsets.UTF_8;
        private boolean strict = false;
        private boolean stripBom = true;
        private boolean allowComments = false;
        private boolean allowHashComments = false;
        private boolean allowTrailingCommas = false;
        private boolean allowSingleQuotes = false;
        private boolean allowUnquotedKeys = false;
        private boolean allowNanInfinity = false;
        private boolean allowTrailingContent = false;
        private boolean allowMultipleValues = false;
        private boolean allowEmpty = false;
        private boolean allowMultiLineJsonl = false;
        private boolean skipBlankLines = true;
        private String commentPrefix = null;
        private int maxDepth = 1000;
        private int maxStringLength = 16 * 1024 * 1024;
        private int maxArrayLength = 10_000_000;
        private int maxObjectKeys = 1_000_000;
        private int maxRows = -1;
        private int skipRows = 0;
        private DuplicateKeyPolicy duplicateKeyPolicy = DuplicateKeyPolicy.LAST;
        private JsonParser.JsonErrorHandler onError = null;

        public Builder charset(Charset v) { this.charset = Objects.requireNonNull(v); return this; }
        public Builder strict(boolean v) { this.strict = v; return this; }
        public Builder stripBom(boolean v) { this.stripBom = v; return this; }
        public Builder allowComments(boolean v) { this.allowComments = v; return this; }
        public Builder allowHashComments(boolean v) { this.allowHashComments = v; return this; }
        public Builder allowTrailingCommas(boolean v) { this.allowTrailingCommas = v; return this; }
        public Builder allowSingleQuotes(boolean v) { this.allowSingleQuotes = v; return this; }
        public Builder allowUnquotedKeys(boolean v) { this.allowUnquotedKeys = v; return this; }
        public Builder allowNanInfinity(boolean v) { this.allowNanInfinity = v; return this; }
        public Builder allowTrailingContent(boolean v) { this.allowTrailingContent = v; return this; }
        public Builder allowMultipleValues(boolean v) { this.allowMultipleValues = v; return this; }
        public Builder allowEmpty(boolean v) { this.allowEmpty = v; return this; }
        public Builder allowMultiLineJsonl(boolean v) { this.allowMultiLineJsonl = v; return this; }
        public Builder skipBlankLines(boolean v) { this.skipBlankLines = v; return this; }
        public Builder commentPrefix(String v) { this.commentPrefix = v; return this; }
        public Builder maxDepth(int v) { this.maxDepth = v; return this; }
        public Builder maxStringLength(int v) { this.maxStringLength = v; return this; }
        public Builder maxArrayLength(int v) { this.maxArrayLength = v; return this; }
        public Builder maxObjectKeys(int v) { this.maxObjectKeys = v; return this; }
        public Builder maxRows(int v) { this.maxRows = v; return this; }
        public Builder skipRows(int v) { this.skipRows = v; return this; }
        public Builder duplicateKeyPolicy(DuplicateKeyPolicy v) {
            this.duplicateKeyPolicy = Objects.requireNonNull(v); return this;
        }
        public Builder onError(JsonParser.JsonErrorHandler h) { this.onError = h; return this; }

        /** Convenience: wire a simple logger. */
        public Builder onError(BiConsumer<Long, String> logger) {
            this.onError = (line, raw, err) -> logger.accept(line, err.getMessage());
            return this;
        }

        public JsonReadOptions build() { return new JsonReadOptions(this); }
    }
}
