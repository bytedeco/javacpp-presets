package org.bytedeco.pytorch.utils.lance;

import org.lance.index.DistanceType;
import org.lance.index.IndexParams;
import org.lance.index.IndexType;
import org.lance.index.scalar.BTreeIndexParams;
import org.lance.index.scalar.BitmapIndexParams;
import org.lance.index.scalar.InvertedIndexParams;
import org.lance.index.scalar.LabelListIndexParams;
import org.lance.index.scalar.NGramIndexParams;
import org.lance.index.scalar.ScalarIndexParams;
import org.lance.index.vector.HnswBuildParams;
import org.lance.index.vector.IvfBuildParams;
import org.lance.index.vector.PQBuildParams;
import org.lance.index.vector.SQBuildParams;
import org.lance.index.vector.VectorIndexParams;

import java.util.Locale;
import java.util.Objects;

/**
 * Factories for Lance vector / scalar / full-text index specs.
 *
 * <pre>{@code
 * ds.createVectorIndex("emb", LanceIndex.ivfHnswPq(256, 16, 200, "cosine"));
 * ds.createScalarIndex("label", LanceIndex.btree());
 * ds.createFtsIndex("text");
 * }</pre>
 */
public final class LanceIndex {

    private final IndexType indexType;
    private final IndexParams indexParams;
    private final String name; // optional preferred name

    private LanceIndex(IndexType type, IndexParams params, String name) {
        this.indexType = Objects.requireNonNull(type, "indexType");
        this.indexParams = Objects.requireNonNull(params, "indexParams");
        this.name = name;
    }

    public IndexType indexType() { return indexType; }
    public IndexParams indexParams() { return indexParams; }
    public String name() { return name; }

    public LanceIndex named(String indexName) {
        return new LanceIndex(indexType, indexParams, indexName);
    }

    // ── vector indexes ────────────────────────────────────────────────────

    /** IVF-Flat: exact residual search within nlist partitions. */
    public static LanceIndex ivfFlat(int numPartitions, String metric) {
        DistanceType dt = parseDistance(metric);
        VectorIndexParams vip = VectorIndexParams.ivfFlat(numPartitions, dt);
        return vector(IndexType.IVF_FLAT, vip);
    }

    /**
     * IVF-PQ.
     *
     * @param numPartitions IVF nlist
     * @param numSubVectors PQ m (must divide dim)
     * @param numBits       bits per sub-vector (typically 8)
     * @param metric        l2 / cosine / dot
     * @param maxIters      k-means iters (e.g. 50)
     */
    public static LanceIndex ivfPq(int numPartitions, int numSubVectors, int numBits,
                                   String metric, int maxIters) {
        DistanceType dt = parseDistance(metric);
        VectorIndexParams vip = VectorIndexParams.ivfPq(
            numPartitions, numSubVectors, numBits, dt, maxIters);
        return vector(IndexType.IVF_PQ, vip);
    }

    public static LanceIndex ivfPq(int numPartitions, int numSubVectors, String metric) {
        return ivfPq(numPartitions, numSubVectors, 8, metric, 50);
    }

    /**
     * IVF-HNSW-PQ — preferred ANN index for large multimodal datasets.
     *
     * @param numPartitions IVF nlist
     * @param hnswM         HNSW M (graph degree)
     * @param efConstruction HNSW efConstruction
     * @param metric        l2 / cosine / dot
     * @param numSubVectors PQ m
     * @param numBits       PQ bits
     */
    public static LanceIndex ivfHnswPq(int numPartitions, int hnswM, int efConstruction,
                                       String metric, int numSubVectors, int numBits) {
        DistanceType dt = parseDistance(metric);
        IvfBuildParams ivf = new IvfBuildParams.Builder()
            .setNumPartitions(numPartitions)
            .build();
        HnswBuildParams hnsw = new HnswBuildParams.Builder()
            .setM(hnswM)
            .setEfConstruction(efConstruction)
            .build();
        PQBuildParams pq = new PQBuildParams.Builder()
            .setNumSubVectors(numSubVectors)
            .setNumBits(numBits)
            .build();
        VectorIndexParams vip = VectorIndexParams.withIvfHnswPqParams(dt, ivf, hnsw, pq);
        return vector(IndexType.IVF_HNSW_PQ, vip);
    }

    /** IVF-HNSW-PQ with default PQ (m=16, bits=8). */
    public static LanceIndex ivfHnswPq(int numPartitions, int hnswM, int efConstruction,
                                       String metric) {
        return ivfHnswPq(numPartitions, hnswM, efConstruction, metric, 16, 8);
    }

    /**
     * IVF-HNSW-SQ (scalar quantization).
     */
    public static LanceIndex ivfHnswSq(int numPartitions, int hnswM, int efConstruction,
                                       String metric, short numBits) {
        DistanceType dt = parseDistance(metric);
        IvfBuildParams ivf = new IvfBuildParams.Builder()
            .setNumPartitions(numPartitions)
            .build();
        HnswBuildParams hnsw = new HnswBuildParams.Builder()
            .setM(hnswM)
            .setEfConstruction(efConstruction)
            .build();
        SQBuildParams sq = new SQBuildParams.Builder()
            .setNumBits(numBits)
            .build();
        VectorIndexParams vip = VectorIndexParams.withIvfHnswSqParams(dt, ivf, hnsw, sq);
        return vector(IndexType.IVF_HNSW_SQ, vip);
    }

    public static LanceIndex ivfHnswSq(int numPartitions, int hnswM, int efConstruction,
                                       String metric) {
        return ivfHnswSq(numPartitions, hnswM, efConstruction, metric, (short) 8);
    }

    // ── scalar / FTS indexes ──────────────────────────────────────────────

    public static LanceIndex btree() {
        ScalarIndexParams p = new BTreeIndexParams.Builder().build();
        return scalar(IndexType.BTREE, p);
    }

    public static LanceIndex bitmap() {
        ScalarIndexParams p = new BitmapIndexParams.Builder().build();
        return scalar(IndexType.BITMAP, p);
    }

    /** Full-text inverted index (default tokenizer). */
    public static LanceIndex inverted() {
        ScalarIndexParams p = InvertedIndexParams.builder().build();
        return scalar(IndexType.INVERTED, p);
    }

    public static LanceIndex inverted(String language, boolean stem, boolean removeStopWords) {
        ScalarIndexParams p = InvertedIndexParams.builder()
            .language(language)
            .stem(stem)
            .removeStopWords(removeStopWords)
            .build();
        return scalar(IndexType.INVERTED, p);
    }

    /** Alias for {@link #inverted()}. */
    public static LanceIndex fts() {
        return inverted();
    }

    public static LanceIndex ngram() {
        ScalarIndexParams p = new NGramIndexParams.Builder().build();
        return scalar(IndexType.NGRAM, p);
    }

    public static LanceIndex labelList() {
        ScalarIndexParams p = new LabelListIndexParams.Builder().build();
        return scalar(IndexType.LABEL_LIST, p);
    }

    // ── helpers ───────────────────────────────────────────────────────────

    private static LanceIndex vector(IndexType type, VectorIndexParams vip) {
        IndexParams params = IndexParams.builder()
            .setVectorIndexParams(vip)
            .build();
        return new LanceIndex(type, params, null);
    }

    private static LanceIndex scalar(IndexType type, ScalarIndexParams sip) {
        IndexParams params = IndexParams.builder()
            .setScalarIndexParams(sip)
            .build();
        return new LanceIndex(type, params, null);
    }

    static DistanceType parseDistance(String metric) {
        if (metric == null || metric.isBlank()) return DistanceType.L2;
        String m = metric.trim().toLowerCase(Locale.ROOT);
        return switch (m) {
            case "cosine", "cos" -> DistanceType.Cosine;
            case "dot", "ip", "inner_product" -> DistanceType.Dot;
            case "hamming" -> DistanceType.Hamming;
            case "l2", "euclidean", "euclid" -> DistanceType.L2;
            default -> DistanceType.L2;
        };
    }
}
