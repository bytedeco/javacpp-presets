package org.bytedeco.pytorch.data.dataframe.vectorstore;

import org.bytedeco.pytorch.data.dataframe.ann.Distance;

/**
 * Distance / similarity metric used by remote vector stores.
 * Maps cleanly onto the in-process {@link Distance} enum.
 */
public enum VectorMetric {
    /** Squared (or plain) Euclidean — lower is better. */
    L2,
    /** Inner product — higher is better on the server; we may negate for ranking. */
    IP,
    /** Cosine distance / similarity (server-specific encoding). */
    COSINE;

    public Distance toDistance() {
        return switch (this) {
            case L2 -> Distance.L2;
            case IP -> Distance.IP;
            case COSINE -> Distance.COSINE;
        };
    }

    public static VectorMetric fromDistance(Distance d) {
        if (d == null) return L2;
        return switch (d) {
            case L2 -> L2;
            case IP -> IP;
            case COSINE -> COSINE;
        };
    }

    /** Qdrant distance name. */
    public String qdrant() {
        return switch (this) {
            case L2 -> "Euclid";
            case IP -> "Dot";
            case COSINE -> "Cosine";
        };
    }

    /** Milvus / OpenSearch / common metric name. */
    public String milvus() {
        return switch (this) {
            case L2 -> "L2";
            case IP -> "IP";
            case COSINE -> "COSINE";
        };
    }

    /** pgvector operator class suffix ({@code vector_l2_ops}, …). */
    public String pgvectorOps() {
        return switch (this) {
            case L2 -> "vector_l2_ops";
            case IP -> "vector_ip_ops";
            case COSINE -> "vector_cosine_ops";
        };
    }

    /** pgvector distance operator. */
    public String pgvectorOp() {
        return switch (this) {
            case L2 -> "<->";
            case IP -> "<#>";
            case COSINE -> "<=>";
        };
    }

    /** Redis RediSearch VECTOR TYPE FLOAT32 DISTANCE_METRIC. */
    public String redis() {
        return switch (this) {
            case L2 -> "L2";
            case IP -> "IP";
            case COSINE -> "COSINE";
        };
    }

    /** OpenSearch knn space type. */
    public String openSearch() {
        return switch (this) {
            case L2 -> "l2";
            case IP -> "innerproduct";
            case COSINE -> "cosinesimil";
        };
    }

    /** MongoDB Atlas Vector Search similarity. */
    public String mongo() {
        return switch (this) {
            case L2 -> "euclidean";
            case IP -> "dotProduct";
            case COSINE -> "cosine";
        };
    }
}
