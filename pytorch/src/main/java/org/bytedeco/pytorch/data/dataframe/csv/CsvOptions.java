package org.bytedeco.pytorch.data.dataframe.csv;

import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * Configuration for robust CSV read/write.
 *
 * <pre>
 *   CsvOptions opt = CsvOptions.builder()
 *       .header(true)
 *       .delimiter(',')
 *       .nullValues("", "NA", "null")
 *       .inferSchema(true)
 *       .build();
 *   DataFrame.readCsv("a.csv", opt);
 * </pre>
 */
public final class CsvOptions {
    public enum QuoteMode {
        /** Quote only fields that need it (delimiter, quote, CR/LF, leading/trailing space). */
        MINIMAL,
        /** Quote every field. */
        ALL,
        /** Quote non-numeric fields. */
        NON_NUMERIC
    }

    private final boolean header;
    private final char delimiter;
    private final char quote;
    private final char escape;          // typically same as quote for RFC 4180 doubled quotes
    private final Charset charset;
    private final Set<String> nullValues;
    private final Character comment;    // null = disabled
    private final int skipRows;
    private final int maxRows;          // -1 = unlimited
    private final boolean inferSchema;
    private final int inferSampleSize;
    private final boolean strict;
    private final boolean typeHeader;
    private final QuoteMode quoteMode;
    private final String writeNullToken; // token written for null cells
    private final boolean stripBom;
    private final List<String> columnNames; // optional override
    private final Map<String, org.bytedeco.pytorch.data.dataframe.Column.DType> schema;

    private CsvOptions(Builder b) {
        this.header = b.header;
        this.delimiter = b.delimiter;
        this.quote = b.quote;
        this.escape = b.escape;
        this.charset = b.charset;
        this.nullValues = Collections.unmodifiableSet(new LinkedHashSet<>(b.nullValues));
        this.comment = b.comment;
        this.skipRows = b.skipRows;
        this.maxRows = b.maxRows;
        this.inferSchema = b.inferSchema;
        this.inferSampleSize = b.inferSampleSize;
        this.strict = b.strict;
        this.typeHeader = b.typeHeader;
        this.quoteMode = b.quoteMode;
        this.writeNullToken = b.writeNullToken;
        this.stripBom = b.stripBom;
        this.columnNames = b.columnNames == null ? null : Collections.unmodifiableList(new ArrayList<>(b.columnNames));
        this.schema = b.schema == null ? null : Collections.unmodifiableMap(new LinkedHashMap<>(b.schema));
    }

    public static Builder builder() { return new Builder(); }

    /** Defaults: header, comma, UTF-8, infer schema, lenient, minimal quoting. */
    public static CsvOptions defaults() { return builder().build(); }

    /**
     * TSV defaults: tab delimiter, header, UTF-8, schema inference, and pandas /
     * IMDb-style null tokens including {@code \\N}.
     */
    public static CsvOptions tsv() {
        return builder()
            .delimiter('\t')
            .nullValues("", "NA", "N/A", "null", "Null", "NULL", "NaN", "nan", "\\N")
            .build();
    }

    public boolean header() { return header; }
    public char delimiter() { return delimiter; }
    public char quote() { return quote; }
    public char escape() { return escape; }
    public Charset charset() { return charset; }
    public Set<String> nullValues() { return nullValues; }
    public Character comment() { return comment; }
    public int skipRows() { return skipRows; }
    public int maxRows() { return maxRows; }
    public boolean inferSchema() { return inferSchema; }
    public int inferSampleSize() { return inferSampleSize; }
    public boolean strict() { return strict; }
    public boolean typeHeader() { return typeHeader; }
    public QuoteMode quoteMode() { return quoteMode; }
    public String writeNullToken() { return writeNullToken; }
    public boolean stripBom() { return stripBom; }
    public List<String> columnNames() { return columnNames; }
    public Map<String, org.bytedeco.pytorch.data.dataframe.Column.DType> schema() { return schema; }

    public boolean isNullToken(String s) {
        if (s == null) return true;
        return nullValues.contains(s) || nullValues.contains(s.trim());
    }

    public static final class Builder {
        private boolean header = true;
        private char delimiter = ',';
        private char quote = '"';
        private char escape = '"';
        private Charset charset = StandardCharsets.UTF_8;
        private final Set<String> nullValues = new LinkedHashSet<>(Arrays.asList("", "NA", "N/A", "null", "Null", "NULL", "NaN", "nan"));
        private Character comment = null;
        private int skipRows = 0;
        private int maxRows = -1;
        private boolean inferSchema = true;
        private int inferSampleSize = 1000;
        private boolean strict = false;
        private boolean typeHeader = false;
        private QuoteMode quoteMode = QuoteMode.MINIMAL;
        private String writeNullToken = "";
        private boolean stripBom = true;
        private List<String> columnNames = null;
        private Map<String, org.bytedeco.pytorch.data.dataframe.Column.DType> schema = null;

        public Builder header(boolean v) { this.header = v; return this; }
        public Builder delimiter(char v) { this.delimiter = v; return this; }
        public Builder quote(char v) { this.quote = v; return this; }
        public Builder escape(char v) { this.escape = v; return this; }
        public Builder charset(Charset v) { this.charset = v; return this; }
        public Builder nullValues(String... tokens) {
            this.nullValues.clear();
            if (tokens != null) Collections.addAll(this.nullValues, tokens);
            return this;
        }
        public Builder addNullValue(String token) { this.nullValues.add(token); return this; }
        public Builder comment(Character v) { this.comment = v; return this; }
        public Builder skipRows(int v) { this.skipRows = v; return this; }
        public Builder maxRows(int v) { this.maxRows = v; return this; }
        public Builder inferSchema(boolean v) { this.inferSchema = v; return this; }
        public Builder inferSampleSize(int v) { this.inferSampleSize = v; return this; }
        public Builder strict(boolean v) { this.strict = v; return this; }
        public Builder typeHeader(boolean v) { this.typeHeader = v; return this; }
        public Builder quoteMode(QuoteMode v) { this.quoteMode = v; return this; }
        public Builder writeNullToken(String v) { this.writeNullToken = v == null ? "" : v; return this; }
        public Builder stripBom(boolean v) { this.stripBom = v; return this; }
        public Builder columnNames(List<String> names) { this.columnNames = names; return this; }
        public Builder schema(Map<String, org.bytedeco.pytorch.data.dataframe.Column.DType> s) { this.schema = s; return this; }

        public CsvOptions build() { return new CsvOptions(this); }
    }
}
