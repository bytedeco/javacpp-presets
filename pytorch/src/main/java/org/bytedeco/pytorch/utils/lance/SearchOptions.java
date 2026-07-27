package org.bytedeco.pytorch.utils.lance;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

/**
 * Options for vector nearest-neighbor search on an official Lance dataset.
 *
 * <pre>{@code
 * DataFrame hits = ds.search("emb", query, 10,
 *     SearchOptions.cosine().ef(64).nprobes(20).filter("label = 'a'"));
 * }</pre>
 */
public final class SearchOptions {

    private String metric = "L2";
    private Integer ef;
    private Integer minimumNprobes;
    private Integer maximumNprobes;
    private Integer refineFactor;
    private Boolean useIndex = true;
    private String filter;
    private List<String> columns;
    private boolean prefilter = true;
    private Integer queryParallelism;
    private String approxMode;
    private long batchSize = -1;

    public static SearchOptions defaults() {
        return new SearchOptions();
    }

    public static SearchOptions l2() {
        return new SearchOptions().metric("L2");
    }

    public static SearchOptions cosine() {
        return new SearchOptions().metric("cosine");
    }

    public static SearchOptions dot() {
        return new SearchOptions().metric("dot");
    }

    public static SearchOptions hamming() {
        return new SearchOptions().metric("hamming");
    }

    public SearchOptions metric(String m) {
        this.metric = m == null || m.isBlank() ? "L2" : m.trim();
        return this;
    }

    /** HNSW ef search parameter (higher = more accurate, slower). */
    public SearchOptions ef(int ef) {
        this.ef = ef;
        return this;
    }

    /** IVF nprobes (single value sets both min and max). */
    public SearchOptions nprobes(int n) {
        this.minimumNprobes = n;
        this.maximumNprobes = n;
        return this;
    }

    public SearchOptions minimumNprobes(int n) {
        this.minimumNprobes = n;
        return this;
    }

    public SearchOptions maximumNprobes(int n) {
        this.maximumNprobes = n;
        return this;
    }

    public SearchOptions refineFactor(int n) {
        this.refineFactor = n;
        return this;
    }

    public SearchOptions useIndex(boolean v) {
        this.useIndex = v;
        return this;
    }

    /** SQL-like prefilter applied before / with ANN (hybrid search). */
    public SearchOptions filter(String expr) {
        this.filter = expr;
        return this;
    }

    public SearchOptions columns(String... cols) {
        if (cols == null || cols.length == 0) this.columns = null;
        else this.columns = new ArrayList<>(Arrays.asList(cols));
        return this;
    }

    public SearchOptions columns(List<String> cols) {
        this.columns = cols == null ? null : new ArrayList<>(cols);
        return this;
    }

    public SearchOptions prefilter(boolean v) {
        this.prefilter = v;
        return this;
    }

    public SearchOptions queryParallelism(int n) {
        this.queryParallelism = n;
        return this;
    }

    /** Approx mode string forwarded to Lance when supported (e.g. "default"). */
    public SearchOptions approxMode(String mode) {
        this.approxMode = mode;
        return this;
    }

    public SearchOptions batchSize(long n) {
        this.batchSize = n;
        return this;
    }

    public String metric() { return metric; }
    public Integer ef() { return ef; }
    public Integer minimumNprobes() { return minimumNprobes; }
    public Integer maximumNprobes() { return maximumNprobes; }
    public Integer refineFactor() { return refineFactor; }
    public Boolean useIndex() { return useIndex; }
    public String filter() { return filter; }
    public List<String> columns() {
        return columns == null ? null : Collections.unmodifiableList(columns);
    }
    public boolean prefilter() { return prefilter; }
    public Integer queryParallelism() { return queryParallelism; }
    public String approxMode() { return approxMode; }
    public long batchSize() { return batchSize; }

    String metricKey() {
        return metric == null ? "l2" : metric.trim().toLowerCase(Locale.ROOT);
    }
}
