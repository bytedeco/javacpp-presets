package org.bytedeco.pytorch.dataframe.json;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.data.json.JsonReadOptions;
import org.bytedeco.pytorch.data.json.JsonWriteOptions;

import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * DataFrame-oriented JSON / JSONL read-write configuration.
 *
 * <pre>
 *   JsonOptions opt = JsonOptions.builder()
 *       .orient(JsonOptions.Orient.RECORDS)
 *       .flatten(true)
 *       .inferSchema(true)
 *       .build();
 *   DataFrame df = DataFrame.readJson("data.json", opt);
 * </pre>
 */
public final class JsonOptions {

    /**
     * Layout of the JSON document relative to a DataFrame.
     * <ul>
     *   <li>{@link #RECORDS} — {@code [{col:val,...}, ...]} (default, pandas-compatible)</li>
     *   <li>{@link #COLUMNS} — {@code {col:[v1,v2,...], ...}}</li>
     *   <li>{@link #VALUES} — {@code [[v11,v12], [v21,v22], ...]}</li>
     *   <li>{@link #INDEX} — {@code {idx:{col:val}, ...}}</li>
     *   <li>{@link #SPLIT} — {@code {columns:[...], index:[...], data:[[...],...]}}</li>
     *   <li>{@link #TABLE} — pandas table schema {@code {schema:{fields:[...]}, data:[...]}}</li>
     *   <li>{@link #LINES} — JSON Lines / NDJSON (one object per line)</li>
     * </ul>
     */
    public enum Orient {
        RECORDS, COLUMNS, VALUES, INDEX, SPLIT, TABLE, LINES
    }

    public enum DateFormat {
        ISO, EPOCH_MILLIS, EPOCH_SECONDS, STRING
    }

    private final Orient orient;
    private final boolean flatten;
    private final String flattenSeparator;
    private final boolean inferSchema;
    private final int inferSampleSize;
    private final boolean strict;
    private final int maxRows;
    private final int skipRows;
    private final List<String> columnNames;
    private final Map<String, Column.DType> schema;
    private final Set<String> nullValues;
    private final boolean keepNestedAsJson;
    private final boolean explodeArrays;
    private final String recordPath;       // JSONPath to array of records
    private final List<String> metaPaths;  // extra fields pulled from parent into each row
    private final boolean pretty;
    private final boolean writeNulls;
    private final DateFormat dateFormat;
    private final Charset charset;
    private final boolean stripBom;
    private final JsonReadOptions.DuplicateKeyPolicy duplicateKeyPolicy;
    private final boolean allowComments;
    private final boolean allowTrailingCommas;
    private final boolean allowMultiLineJsonl;
    private final String linesCommentPrefix;
    private final int dateUnit; // for epoch: 1=ms default

    private JsonOptions(Builder b) {
        this.orient = b.orient;
        this.flatten = b.flatten;
        this.flattenSeparator = b.flattenSeparator;
        this.inferSchema = b.inferSchema;
        this.inferSampleSize = b.inferSampleSize;
        this.strict = b.strict;
        this.maxRows = b.maxRows;
        this.skipRows = b.skipRows;
        this.columnNames = b.columnNames == null ? null
            : Collections.unmodifiableList(new ArrayList<>(b.columnNames));
        this.schema = b.schema == null ? null
            : Collections.unmodifiableMap(new LinkedHashMap<>(b.schema));
        this.nullValues = Collections.unmodifiableSet(new LinkedHashSet<>(b.nullValues));
        this.keepNestedAsJson = b.keepNestedAsJson;
        this.explodeArrays = b.explodeArrays;
        this.recordPath = b.recordPath;
        this.metaPaths = b.metaPaths == null ? Collections.emptyList()
            : Collections.unmodifiableList(new ArrayList<>(b.metaPaths));
        this.pretty = b.pretty;
        this.writeNulls = b.writeNulls;
        this.dateFormat = b.dateFormat;
        this.charset = b.charset;
        this.stripBom = b.stripBom;
        this.duplicateKeyPolicy = b.duplicateKeyPolicy;
        this.allowComments = b.allowComments;
        this.allowTrailingCommas = b.allowTrailingCommas;
        this.allowMultiLineJsonl = b.allowMultiLineJsonl;
        this.linesCommentPrefix = b.linesCommentPrefix;
        this.dateUnit = b.dateUnit;
    }

    public static Builder builder() { return new Builder(); }
    public static JsonOptions defaults() { return builder().build(); }

    public static JsonOptions lines() {
        return builder().orient(Orient.LINES).build();
    }

    public static JsonOptions prettyRecords() {
        return builder().orient(Orient.RECORDS).pretty(true).build();
    }

    public Orient orient() { return orient; }
    public boolean flatten() { return flatten; }
    public String flattenSeparator() { return flattenSeparator; }
    public boolean inferSchema() { return inferSchema; }
    public int inferSampleSize() { return inferSampleSize; }
    public boolean strict() { return strict; }
    public int maxRows() { return maxRows; }
    public int skipRows() { return skipRows; }
    public List<String> columnNames() { return columnNames; }
    public Map<String, Column.DType> schema() { return schema; }
    public Set<String> nullValues() { return nullValues; }
    public boolean keepNestedAsJson() { return keepNestedAsJson; }
    public boolean explodeArrays() { return explodeArrays; }
    public String recordPath() { return recordPath; }
    public List<String> metaPaths() { return metaPaths; }
    public boolean pretty() { return pretty; }
    public boolean writeNulls() { return writeNulls; }
    public DateFormat dateFormat() { return dateFormat; }
    public Charset charset() { return charset; }
    public boolean stripBom() { return stripBom; }
    public JsonReadOptions.DuplicateKeyPolicy duplicateKeyPolicy() { return duplicateKeyPolicy; }
    public boolean allowComments() { return allowComments; }
    public boolean allowTrailingCommas() { return allowTrailingCommas; }
    public boolean allowMultiLineJsonl() { return allowMultiLineJsonl; }
    public String linesCommentPrefix() { return linesCommentPrefix; }
    public int dateUnit() { return dateUnit; }

    public boolean isNullToken(String s) {
        if (s == null) return true;
        return nullValues.contains(s) || nullValues.contains(s.trim());
    }

    /** Convert to low-level parse options. */
    public JsonReadOptions toReadOptions() {
        return JsonReadOptions.builder()
            .charset(charset)
            .strict(strict)
            .stripBom(stripBom)
            .allowComments(allowComments)
            .allowTrailingCommas(allowTrailingCommas)
            .allowMultiLineJsonl(allowMultiLineJsonl)
            .commentPrefix(linesCommentPrefix)
            .duplicateKeyPolicy(duplicateKeyPolicy)
            .maxRows(maxRows)
            .skipRows(skipRows)
            .skipBlankLines(true)
            .build();
    }

    /** Convert to low-level write options. */
    public JsonWriteOptions toWriteOptions() {
        return JsonWriteOptions.builder()
            .charset(charset)
            .pretty(pretty)
            .nullHandling(writeNulls
                ? JsonWriteOptions.NullHandling.WRITE_NULL
                : JsonWriteOptions.NullHandling.OMIT)
            .writeBom(false)
            .build();
    }

    public static final class Builder {
        private Orient orient = Orient.RECORDS;
        private boolean flatten = false;
        private String flattenSeparator = ".";
        private boolean inferSchema = true;
        private int inferSampleSize = 1000;
        private boolean strict = false;
        private int maxRows = -1;
        private int skipRows = 0;
        private List<String> columnNames = null;
        private Map<String, Column.DType> schema = null;
        private final Set<String> nullValues = new LinkedHashSet<>(
            Arrays.asList("", "null", "Null", "NULL", "NA", "N/A", "NaN", "nan", "None"));
        private boolean keepNestedAsJson = true;
        private boolean explodeArrays = false;
        private String recordPath = null;
        private List<String> metaPaths = null;
        private boolean pretty = false;
        private boolean writeNulls = true;
        private DateFormat dateFormat = DateFormat.ISO;
        private Charset charset = StandardCharsets.UTF_8;
        private boolean stripBom = true;
        private JsonReadOptions.DuplicateKeyPolicy duplicateKeyPolicy =
            JsonReadOptions.DuplicateKeyPolicy.LAST;
        private boolean allowComments = false;
        private boolean allowTrailingCommas = false;
        private boolean allowMultiLineJsonl = true;
        private String linesCommentPrefix = null;
        private int dateUnit = 1;

        public Builder orient(Orient v) { this.orient = v; return this; }
        public Builder flatten(boolean v) { this.flatten = v; return this; }
        public Builder flattenSeparator(String v) { this.flattenSeparator = v == null ? "." : v; return this; }
        public Builder inferSchema(boolean v) { this.inferSchema = v; return this; }
        public Builder inferSampleSize(int v) { this.inferSampleSize = v; return this; }
        public Builder strict(boolean v) { this.strict = v; return this; }
        public Builder maxRows(int v) { this.maxRows = v; return this; }
        public Builder skipRows(int v) { this.skipRows = v; return this; }
        public Builder columnNames(List<String> names) { this.columnNames = names; return this; }
        public Builder columnNames(String... names) {
            this.columnNames = names == null ? null : Arrays.asList(names);
            return this;
        }
        public Builder schema(Map<String, Column.DType> s) { this.schema = s; return this; }
        public Builder nullValues(String... tokens) {
            this.nullValues.clear();
            if (tokens != null) Collections.addAll(this.nullValues, tokens);
            return this;
        }
        public Builder addNullValue(String token) { this.nullValues.add(token); return this; }
        public Builder keepNestedAsJson(boolean v) { this.keepNestedAsJson = v; return this; }
        public Builder explodeArrays(boolean v) { this.explodeArrays = v; return this; }
        public Builder recordPath(String v) { this.recordPath = v; return this; }
        public Builder metaPaths(List<String> v) { this.metaPaths = v; return this; }
        public Builder pretty(boolean v) { this.pretty = v; return this; }
        public Builder writeNulls(boolean v) { this.writeNulls = v; return this; }
        public Builder dateFormat(DateFormat v) { this.dateFormat = v; return this; }
        public Builder charset(Charset v) { this.charset = v; return this; }
        public Builder stripBom(boolean v) { this.stripBom = v; return this; }
        public Builder duplicateKeyPolicy(JsonReadOptions.DuplicateKeyPolicy v) {
            this.duplicateKeyPolicy = v; return this;
        }
        public Builder allowComments(boolean v) { this.allowComments = v; return this; }
        public Builder allowTrailingCommas(boolean v) { this.allowTrailingCommas = v; return this; }
        public Builder allowMultiLineJsonl(boolean v) { this.allowMultiLineJsonl = v; return this; }
        public Builder linesCommentPrefix(String v) { this.linesCommentPrefix = v; return this; }
        public Builder dateUnit(int v) { this.dateUnit = v; return this; }

        public JsonOptions build() { return new JsonOptions(this); }
    }
}
