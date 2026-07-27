package org.bytedeco.pytorch.data.dataframe.vectorstore;

import org.bytedeco.pytorch.data.dataframe.Column;
import org.bytedeco.pytorch.data.dataframe.DataFrame;
import org.bytedeco.pytorch.data.dataframe.ann.VectorColumn;
import org.bytedeco.pytorch.data.dataframe.dtype.EmbeddingData;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Backend-agnostic vector collection API.
 *
 * <p>Implementations must be safe for sequential use; concurrent use is best-effort
 * and backend-dependent. Always {@link #close()} when done.
 *
 * <pre>{@code
 * try (VectorStore store = VectorStores.qdrant("http://localhost:6333", "docs", 384, VectorMetric.COSINE)) {
 *     store.ensureCollection();
 *     store.upsert(List.of(VectorRecord.of("a", emb)));
 *     VectorSearchResult r = store.search(VectorQuery.of(query, 5));
 * }
 * }</pre>
 */
public interface VectorStore extends AutoCloseable {

    /** Backend label for diagnostics ({@code "qdrant"}, {@code "redis"}, …). */
    String backend();

    /** Logical collection / index / table name. */
    String name();

    /** Declared embedding dimensionality (0 if unknown / multi-vector). */
    int dim();

    /** Distance metric configured for this collection. */
    VectorMetric metric();

    /**
     * Create the collection / index if it does not exist (idempotent when possible).
     * No-op for stores that do not require explicit schema creation.
     */
    void ensureCollection();

    /** Drop the collection / index. Idempotent when possible. */
    void dropCollection();

    /** Approximate number of vectors, or {@code -1} if unknown. */
    long count();

    /** Upsert one or more points (insert-or-replace by id). */
    void upsert(Collection<VectorRecord> records);

    default void upsert(VectorRecord... records) {
        upsert(List.of(records));
    }

    /** Delete points by string id. */
    void delete(Collection<String> ids);

    default void delete(String... ids) {
        delete(List.of(ids));
    }

    /** k-NN (or approximate) search. */
    VectorSearchResult search(VectorQuery query);

    default VectorSearchResult search(float[] vector, int topK) {
        return search(VectorQuery.of(vector, topK));
    }

    /**
     * Upsert every row of a DataFrame.
     *
     * @param df         source frame
     * @param idCol      string/int id column (nullable → synthetic {@code "0"…"n-1"})
     * @param vectorCol  VECTOR / EMBEDDING / float[] column
     * @param payloadCols extra columns stored as payload (null → all non-id/non-vector cols)
     */
    default void upsertDataFrame(DataFrame df, String idCol, String vectorCol, String... payloadCols) {
        Objects.requireNonNull(df, "df");
        Objects.requireNonNull(vectorCol, "vectorCol");
        Column vcol = df.column(vectorCol);
        Column icol = idCol == null ? null : df.column(idCol);

        // Product policy: only 1-D EMBEDDING / VECTOR / float[] are indexable.
        // Multi-dim TENSOR must be pooled/flattened into an embedding column first.
        if (vcol.dtype() == Column.DType.TENSOR) {
            throw new VectorStoreException(
                "vectorCol '" + vectorCol + "' is TENSOR (multi-dim). "
                    + "Vector stores index 1-D EMBEDDING/VECTOR/float[] only — "
                    + "materialize an embedding column first "
                    + "(e.g. col(\"" + vectorCol + "\").tensor().flatten()).",
                -1, backend());
        }
        for (int i = 0; i < Math.min(vcol.size(), 8); i++) {
            Object sample = vcol.get(i);
            if (sample == null) continue;
            if (sample instanceof org.bytedeco.pytorch.data.dataframe.dtype.TensorData
                || (sample instanceof org.bytedeco.pytorch.data.dataframe.dtype.VectorData vd
                    && vd.getShape() != null && vd.getShape().length > 1)) {
                throw new VectorStoreException(
                    "vectorCol '" + vectorCol + "' contains multi-dim cells; "
                        + "only 1-D EMBEDDING/VECTOR/float[] may be upserted to a vector store.",
                    -1, backend());
            }
            break;
        }

        List<String> payloadNames = new ArrayList<>();
        if (payloadCols == null || payloadCols.length == 0) {
            for (int c = 0; c < df.columnCount(); c++) {
                String n = df.column(c).name();
                if (!n.equals(vectorCol) && (idCol == null || !n.equals(idCol))) {
                    payloadNames.add(n);
                }
            }
        } else {
            for (String p : payloadCols) {
                if (p != null && !p.isEmpty()) payloadNames.add(p);
            }
        }

        List<VectorRecord> batch = new ArrayList<>(df.rowCount());
        for (int r = 0; r < df.rowCount(); r++) {
            float[] vec = toFloatArray(vcol.get(r));
            if (vec == null) continue;

            VectorRecord.Builder b = VectorRecord.builder().vector(vec);
            if (icol != null) {
                Object idv = icol.get(r);
                if (idv instanceof Number n) b.id(n.longValue());
                else if (idv != null) b.id(String.valueOf(idv));
                else b.id((long) r);
            } else {
                b.id((long) r);
            }
            if (!payloadNames.isEmpty()) {
                Map<String, Object> payload = new LinkedHashMap<>();
                for (String pn : payloadNames) {
                    payload.put(pn, df.get(r, pn));
                }
                b.payload(payload);
            }
            batch.add(b.build());
        }
        if (!batch.isEmpty()) upsert(batch);
    }

    /**
     * Search and join hits back onto a source DataFrame by id column
     * (keeps original columns + {@code _score}/{@code _distance}/{@code _rank}).
     */
    default DataFrame searchAsDataFrame(float[] query, int topK) {
        return search(query, topK).toDataFrame();
    }

    /**
     * Fetch points by id. Default: empty (backends should override when supported).
     * Missing ids are omitted rather than erroring.
     */
    default List<VectorRecord> fetch(Collection<String> ids) {
        return List.of();
    }

    default List<VectorRecord> fetch(String... ids) {
        return fetch(ids == null ? List.of() : List.of(ids));
    }

    /**
     * Page through stored points. Default: empty.
     *
     * @param limit  max records this page
     * @param cursor backend-specific cursor (null = start); pass the returned next cursor
     * @return page; {@link ScrollPage#nextCursor()} null means end
     */
    default ScrollPage scroll(int limit, Object cursor) {
        return ScrollPage.empty();
    }

    /**
     * Materialize (up to {@code limit}) stored points as a DataFrame with columns
     * {@code id}, {@code vector} (VECTOR), plus flattened payload keys.
     */
    default DataFrame toDataFrame(int limit) {
        List<VectorRecord> all = new ArrayList<>();
        Object cursor = null;
        int remaining = limit <= 0 ? Integer.MAX_VALUE : limit;
        while (remaining > 0) {
            int page = Math.min(256, remaining);
            ScrollPage sp = scroll(page, cursor);
            if (sp.records().isEmpty()) break;
            all.addAll(sp.records());
            remaining -= sp.records().size();
            cursor = sp.nextCursor();
            if (cursor == null || sp.records().size() < page) break;
        }
        return recordsToDataFrame(all);
    }

    default DataFrame toDataFrame() {
        return toDataFrame(100_000);
    }

    default DataFrame toDataFrame(String idCol, String vectorCol) {
        return recordsToDataFrame(scrollAll(100_000),
            idCol == null ? "id" : idCol,
            vectorCol == null ? "vector" : vectorCol);
    }

    /** Drain scroll into a list (capped). */
    default List<VectorRecord> scrollAll(int limit) {
        List<VectorRecord> all = new ArrayList<>();
        Object cursor = null;
        int remaining = limit <= 0 ? Integer.MAX_VALUE : limit;
        while (remaining > 0) {
            int page = Math.min(256, remaining);
            ScrollPage sp = scroll(page, cursor);
            if (sp.records().isEmpty()) break;
            all.addAll(sp.records());
            remaining -= sp.records().size();
            cursor = sp.nextCursor();
            if (cursor == null || sp.records().size() < page) break;
        }
        return all;
    }

    /**
     * Multi-query (batch) search. Default loops {@link #search(VectorQuery)};
     * backends may override with a native batch endpoint (Qdrant / Milvus).
     */
    default List<VectorSearchResult> searchBatch(List<VectorQuery> queries) {
        if (queries == null || queries.isEmpty()) return List.of();
        List<VectorSearchResult> out = new ArrayList<>(queries.size());
        for (VectorQuery q : queries) out.add(search(q));
        return out;
    }

    default List<VectorSearchResult> searchBatch(float[][] queries, int topK) {
        if (queries == null || queries.length == 0) return List.of();
        List<VectorSearchResult> out = new ArrayList<>(queries.length);
        for (float[] q : queries) {
            if (q == null) out.add(VectorSearchResult.empty());
            else out.add(search(q, topK));
        }
        return out;
    }

    @Override
    void close();

    // ---- helpers ----

    static DataFrame recordsToDataFrame(List<VectorRecord> records) {
        return recordsToDataFrame(records, "id", "vector");
    }

    static DataFrame recordsToDataFrame(List<VectorRecord> records, String idCol, String vectorCol) {
        DataFrame df = DataFrame.create();
        df.addColumn(idCol, Column.DType.STRING);
        df.addColumn(vectorCol, Column.DType.VECTOR);
        List<String> payloadKeys = new ArrayList<>();
        if (records != null) {
            for (VectorRecord r : records) {
                for (String k : r.payload().keySet()) {
                    if (!payloadKeys.contains(k)) payloadKeys.add(k);
                }
            }
        }
        for (String k : payloadKeys) df.addColumn(k, Column.DType.STRING);
        if (records != null) {
            for (VectorRecord r : records) {
                int row = df.addEmptyRow();
                df.set(row, idCol, r.resolvedId());
                df.set(row, vectorCol, r.vector() == null ? null : r.vector().clone());
                for (String k : payloadKeys) {
                    Object v = r.payload().get(k);
                    df.set(row, k, v == null ? null : (v instanceof String ? v : String.valueOf(v)));
                }
            }
        }
        return df;
    }

    public static float[] toFloatArray(Object cell) {
        if (cell == null) return null;
        if (cell instanceof float[] f) return f;
        if (cell instanceof double[] d) {
            float[] f = new float[d.length];
            for (int i = 0; i < d.length; i++) f[i] = (float) d[i];
            return f;
        }
        if (cell instanceof EmbeddingData e) return e.getVector();
        try {
            return VectorColumn.asFloatArray(cell);
        } catch (IllegalArgumentException | ClassCastException ex) {
            return null;
        }
    }

    /**
     * One page of {@link #scroll(int, Object)}.
     */
    final class ScrollPage {
        private final List<VectorRecord> records;
        private final Object nextCursor;

        public ScrollPage(List<VectorRecord> records, Object nextCursor) {
            this.records = records == null ? List.of() : List.copyOf(records);
            this.nextCursor = nextCursor;
        }

        public static ScrollPage empty() {
            return new ScrollPage(List.of(), null);
        }

        public List<VectorRecord> records() { return records; }
        public Object nextCursor() { return nextCursor; }
        public boolean isEmpty() { return records.isEmpty(); }
    }
}
