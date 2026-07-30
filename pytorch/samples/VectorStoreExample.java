package samples;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.vectorstore.PayloadField;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorHit;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorMetric;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorQuery;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorRecord;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorSearchResult;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorStore;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorStores;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Zero-SDK vector-store demo: bulk upsert / fetch / scroll / batch search
 * against the in-process HNSW backend (always available), plus builder
 * snippets for Redis / Qdrant / OpenSearch / Milvus / pgvector / Mongo.
 *
 * <pre>
 *   java --add-opens=java.base/java.nio=ALL-UNNAMED -cp ... samples.VectorStoreExample
 *   java ... samples.VectorStoreExample qdrant://localhost:6333/clips?dim=32&amp;metric=cosine
 * </pre>
 *
 * <p>No vendor JARs required. Remote backends talk HTTP / RESP / JDBC only.
 */
public final class VectorStoreExample {

    static int passed = 0, failed = 0;

    public static void main(String[] args) throws Exception {
        System.out.println("=== VectorStore bulk demo (zero vendor SDKs) ===\n");

        // 1) Always-on in-memory path exercises the full SPI.
        runInMemoryBulk();

        // 2) Show how to open each remote backend (construct only — no live server required).
        showBuilders();

        // 3) Optional: if user passed a URI, try a live round-trip.
        if (args.length > 0 && args[0] != null && !args[0].isBlank()) {
            liveRoundTrip(args[0]);
        } else {
            System.out.println("(pass a URI arg to hit a live backend, e.g. memory://x/demo?dim=32)");
        }

        System.out.println("\n=== summary: passed=" + passed + " failed=" + failed + " ===");
        if (failed > 0) System.exit(1);
    }

    static void runInMemoryBulk() {
        final int dim = 32;
        final int n = 500;
        System.out.println("-- InMemory bulk upsert / fetch / scroll / searchBatch (n=" + n + ", dim=" + dim + ")");

        try (VectorStore vs = VectorStores.memory("demo", dim, VectorMetric.L2)) {
            vs.ensureCollection();

            // bulk build
            List<VectorRecord> batch = new ArrayList<>(n);
            Random rnd = new Random(42);
            for (int i = 0; i < n; i++) {
                float[] v = randomUnit(rnd, dim);
                Map<String, Object> payload = new LinkedHashMap<>();
                payload.put("title", "doc-" + i);
                payload.put("category", i % 2 == 0 ? "even" : "odd");
                payload.put("year", 2000 + (i % 25));
                batch.add(VectorRecord.of("id-" + i, v, payload));
            }

            long t0 = System.nanoTime();
            vs.upsert(batch);
            long upsertMs = (System.nanoTime() - t0) / 1_000_000L;
            check("count after bulk upsert", vs.count() == n);
            System.out.println("  upsert " + n + " pts in " + upsertMs + " ms  (" +
                (n * 1000.0 / Math.max(1, upsertMs)) + " pts/s)");

            // DataFrame path
            DataFrame df = DataFrame.create();
            df.addColumn("id", Column.DType.STRING);
            df.addColumn("emb", Column.DType.VECTOR);
            df.addColumn("tag", Column.DType.STRING);
            for (int i = 0; i < 10; i++) {
                int row = df.addEmptyRow();
                df.set(row, "id", "df-" + i);
                df.set(row, "emb", randomUnit(rnd, dim));
                df.set(row, "tag", "from-df");
            }
            vs.upsertDataFrame(df, "id", "emb", "tag");
            check("count after df upsert", vs.count() == n + 10);

            // knn
            float[] query = batch.get(0).vector();
            VectorSearchResult top = vs.search(VectorQuery.builder(query, 5).includePayload(true).build());
            check("knn returned 5", top.size() == 5);
            check("knn top is self", "id-0".equals(top.get(0).id()));
            System.out.println("  top-5 for id-0: " + ids(top));

            // batch search
            float[][] qs = new float[][]{ batch.get(0).vector(), batch.get(1).vector(), batch.get(2).vector() };
            List<VectorSearchResult> multi = vs.searchBatch(qs, 3);
            check("searchBatch size", multi.size() == 3);
            check("searchBatch[0] top", "id-0".equals(multi.get(0).get(0).id()));

            // fetch
            List<VectorRecord> got = vs.fetch(List.of("id-0", "id-1", "missing", "id-2"));
            check("fetch 3 of 4", got.size() == 3);
            System.out.println("  fetch ids: " + got.stream().map(VectorRecord::resolvedId).toList());

            // scroll pages
            int seen = 0;
            Object cursor = null;
            int pages = 0;
            while (pages < 20) {
                VectorStore.ScrollPage page = vs.scroll(100, cursor);
                if (page.isEmpty()) break;
                seen += page.records().size();
                cursor = page.nextCursor();
                pages++;
                if (cursor == null) break;
            }
            check("scroll saw all", seen == vs.count());
            System.out.println("  scroll pages=" + pages + " records=" + seen);

            // round-trip to DataFrame
            DataFrame back = vs.toDataFrame(50);
            check("toDataFrame rows", back.rowCount() == 50);
            check("toDataFrame has vector", back.hasColumn("vector"));
            System.out.println("  toDataFrame(50) cols=" + back.columns().stream().map(c -> c.name()).toList());

            // delete + re-search
            vs.delete("id-0");
            VectorSearchResult afterDel = vs.search(query, 3);
            check("deleted id not in top", !"id-0".equals(afterDel.get(0).id()));

            System.out.println("  InMemory bulk SPI: OK\n");
        } catch (Throwable e) {
            failed++;
            System.out.println(" FAIL InMemory bulk: " + e);
            e.printStackTrace(System.out);
        }
    }

    static void showBuilders() {
        System.out.println("-- Builder recipes (no network I/O until ensureCollection/upsert) --");

        // Redis + multi-field SCHEMA + pipelined HSET
        try (VectorStore redis = org.bytedeco.pytorch.dataframe.vectorstore.redis.RedisVectorStore.builder()
                .host("127.0.0.1").port(6379)
                .index("idx:clips").prefix("doc:")
                .dim(768).metric(VectorMetric.COSINE)
                .payloadField(PayloadField.text("title"))
                .payloadField(PayloadField.tag("category"))
                .payloadField(PayloadField.numeric("year").sortable())
                .pipelineBatch(256)
                .build()) {
            System.out.println("  redis      backend=" + redis.backend()
                + " schema=[title:TEXT, category:TAG, year:NUMERIC] pipelined upsert");
        }

        // OpenSearch _bulk NDJSON
        try (VectorStore os = org.bytedeco.pytorch.dataframe.vectorstore.opensearch.OpenSearchVectorStore.builder(
                    "http://localhost:9200")
                .index("clips").dim(768).metric(VectorMetric.L2)
                .payloadField(PayloadField.text("title"))
                .payloadField(PayloadField.tag("category"))
                .bulkBatch(500)
                .build()) {
            System.out.println("  opensearch backend=" + os.backend() + " uses _bulk NDJSON");
        }

        // Qdrant batch search
        try (VectorStore q = VectorStores.qdrant("http://localhost:6333", "clips", 768, VectorMetric.COSINE)) {
            System.out.println("  qdrant    backend=" + q.backend() + " bulk points + /search/batch");
        }

        // Milvus REST multi-vector search
        try (VectorStore m = VectorStores.milvus("http://localhost:9091", "clips", 768, VectorMetric.L2, "root:Milvus")) {
            System.out.println("  milvus    backend=" + m.backend() + " entities/upsert + multi search");
        }

        // pgvector JDBC batch
        try (VectorStore p = VectorStores.pgvector(
                "jdbc:postgresql://localhost:5432/postgres", "postgres", "postgres",
                "clips", 768, VectorMetric.COSINE)) {
            System.out.println("  pgvector  backend=" + p.backend() + " JDBC batch upsert (driver on app CP)");
        }

        // Mongo Atlas Data API
        try (VectorStore mo = VectorStores.mongoAtlas(
                "https://data.mongodb-api.com/app/APP/endpoint/data/v1", "API_KEY",
                "Cluster0", "rag", "clips", 768, VectorMetric.COSINE)) {
            System.out.println("  mongo     backend=" + mo.backend() + " Data API find/aggregate/updateOne");
        }

        // URI form
        System.out.println("  URI form  VectorStores.open(\"qdrant://localhost:6333/clips?dim=768&metric=cosine\")");
        System.out.println();
    }

    static void liveRoundTrip(String uri) {
        System.out.println("-- Live round-trip: " + uri);
        try (VectorStore vs = VectorStores.open(uri)) {
            int dim = vs.dim() > 0 ? vs.dim() : 32;
            // if dim unknown, reopen memory-style only works when dim in query
            if (vs.dim() <= 0 && !"memory".equals(vs.backend()) && !"hnsw".equals(vs.backend())) {
                System.out.println("  skip live write: dim not set in URI (add ?dim=N)");
                return;
            }
            vs.ensureCollection();
            Random rnd = new Random(7);
            List<VectorRecord> batch = new ArrayList<>();
            for (int i = 0; i < 20; i++) {
                batch.add(VectorRecord.of("live-" + i, randomUnit(rnd, dim),
                    Map.of("i", i, "label", "live")));
            }
            vs.upsert(batch);
            System.out.println("  count=" + vs.count());
            VectorSearchResult r = vs.search(batch.get(0).vector(), 3);
            System.out.println("  search: " + ids(r));
            List<VectorRecord> fetched = vs.fetch(List.of("live-0", "live-1"));
            System.out.println("  fetch:  " + fetched.size() + " docs");
            VectorStore.ScrollPage page = vs.scroll(10, null);
            System.out.println("  scroll: " + page.records().size() + " docs, next=" + page.nextCursor());
            check("live search non-empty", r.size() > 0);
            System.out.println("  live OK\n");
        } catch (Throwable e) {
            failed++;
            System.out.println(" FAIL live " + uri + ": " + e.getMessage());
        }
    }

    static float[] randomUnit(Random rnd, int dim) {
        float[] v = new float[dim];
        double sum = 0;
        for (int i = 0; i < dim; i++) {
            v[i] = rnd.nextFloat() * 2 - 1;
            sum += v[i] * v[i];
        }
        float inv = sum > 0 ? (float) (1.0 / Math.sqrt(sum)) : 1f;
        for (int i = 0; i < dim; i++) v[i] *= inv;
        return v;
    }

    static List<String> ids(VectorSearchResult r) {
        List<String> out = new ArrayList<>();
        for (VectorHit h : r.hits()) out.add(h.id() + ":" + String.format("%.4f", h.score()));
        return out;
    }

    static void check(String name, boolean ok) {
        if (ok) {
            passed++;
            System.out.println("  OK  " + name);
        } else {
            failed++;
            System.out.println(" FAIL " + name);
        }
    }
}
