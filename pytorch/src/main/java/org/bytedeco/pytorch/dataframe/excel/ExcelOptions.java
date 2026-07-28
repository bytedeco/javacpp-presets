package org.bytedeco.pytorch.dataframe.excel;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.io.IoNullTokens;

import java.util.*;

/**
 * Configuration for Excel ({@code .xlsx}/{@code .xls}) read/write.
 *
 * <pre>
 *   ExcelOptions opt = ExcelOptions.builder()
 *       .sheet("Sheet1")
 *       .header(true)
 *       .inferSchema(true)
 *       .build();
 *   DataFrame.readExcel("a.xlsx", opt);
 * </pre>
 */
public final class ExcelOptions {
    private final String sheetName;       // null = first / by index
    private final int sheetIndex;         // used when sheetName is null; -1 = all (readAll)
    private final boolean header;
    private final int skipRows;
    private final int maxRows;            // -1 unlimited
    private final boolean inferSchema;
    private final int inferSampleSize;
    private final boolean strict;
    private final boolean evaluateFormulas;
    private final boolean dateAsLocalDate; // true → LocalDate/LocalDateTime; false → Instant/epoch
    private final Set<String> nullValues;
    private final String writeNullToken;
    private final List<String> columnNames;
    private final Map<String, Column.DType> schema;
    private final boolean freezeHeader;
    private final String writeSheetName;

    private ExcelOptions(Builder b) {
        this.sheetName = b.sheetName;
        this.sheetIndex = b.sheetIndex;
        this.header = b.header;
        this.skipRows = b.skipRows;
        this.maxRows = b.maxRows;
        this.inferSchema = b.inferSchema;
        this.inferSampleSize = b.inferSampleSize;
        this.strict = b.strict;
        this.evaluateFormulas = b.evaluateFormulas;
        this.dateAsLocalDate = b.dateAsLocalDate;
        this.nullValues = Collections.unmodifiableSet(new LinkedHashSet<>(b.nullValues));
        this.writeNullToken = b.writeNullToken;
        this.columnNames = b.columnNames == null ? null : Collections.unmodifiableList(new ArrayList<>(b.columnNames));
        this.schema = b.schema == null ? null : Collections.unmodifiableMap(new LinkedHashMap<>(b.schema));
        this.freezeHeader = b.freezeHeader;
        this.writeSheetName = b.writeSheetName;
    }

    public static Builder builder() { return new Builder(); }
    public static ExcelOptions defaults() { return builder().build(); }

    public String sheetName() { return sheetName; }
    public int sheetIndex() { return sheetIndex; }
    public boolean header() { return header; }
    public int skipRows() { return skipRows; }
    public int maxRows() { return maxRows; }
    public boolean inferSchema() { return inferSchema; }
    public int inferSampleSize() { return inferSampleSize; }
    public boolean strict() { return strict; }
    public boolean evaluateFormulas() { return evaluateFormulas; }
    public boolean dateAsLocalDate() { return dateAsLocalDate; }
    public Set<String> nullValues() { return nullValues; }
    public String writeNullToken() { return writeNullToken; }
    public List<String> columnNames() { return columnNames; }
    public Map<String, Column.DType> schema() { return schema; }
    public boolean freezeHeader() { return freezeHeader; }
    public String writeSheetName() { return writeSheetName; }

    public boolean isNullToken(String s) {
        return IoNullTokens.isNull(s, nullValues);
    }

    public static final class Builder {
        private String sheetName = null;
        private int sheetIndex = 0;
        private boolean header = true;
        private int skipRows = 0;
        private int maxRows = -1;
        private boolean inferSchema = true;
        private int inferSampleSize = 1000;
        private boolean strict = false;
        private boolean evaluateFormulas = false;
        private boolean dateAsLocalDate = true;
        private final Set<String> nullValues = new LinkedHashSet<>(IoNullTokens.PANDAS_DEFAULT);
        private String writeNullToken = "";
        private List<String> columnNames = null;
        private Map<String, Column.DType> schema = null;
        private boolean freezeHeader = false;
        private String writeSheetName = "Sheet1";

        public Builder sheet(String name) { this.sheetName = name; return this; }
        public Builder sheetIndex(int idx) { this.sheetIndex = idx; return this; }
        public Builder header(boolean v) { this.header = v; return this; }
        public Builder skipRows(int v) { this.skipRows = v; return this; }
        public Builder maxRows(int v) { this.maxRows = v; return this; }
        public Builder inferSchema(boolean v) { this.inferSchema = v; return this; }
        public Builder inferSampleSize(int v) { this.inferSampleSize = v; return this; }
        public Builder strict(boolean v) { this.strict = v; return this; }
        public Builder evaluateFormulas(boolean v) { this.evaluateFormulas = v; return this; }
        public Builder dateAsLocalDate(boolean v) { this.dateAsLocalDate = v; return this; }
        public Builder nullValues(String... tokens) {
            this.nullValues.clear();
            if (tokens != null) Collections.addAll(this.nullValues, tokens);
            return this;
        }
        public Builder writeNullToken(String v) { this.writeNullToken = v == null ? "" : v; return this; }
        public Builder columnNames(List<String> names) { this.columnNames = names; return this; }
        public Builder schema(Map<String, Column.DType> s) { this.schema = s; return this; }
        public Builder freezeHeader(boolean v) { this.freezeHeader = v; return this; }
        public Builder writeSheetName(String v) { this.writeSheetName = v == null ? "Sheet1" : v; return this; }

        public ExcelOptions build() { return new ExcelOptions(this); }
    }
}
