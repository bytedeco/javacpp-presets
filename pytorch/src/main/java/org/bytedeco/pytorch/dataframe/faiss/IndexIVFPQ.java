package org.bytedeco.pytorch.dataframe.faiss;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

/**
 * IVF + Product Quantization index — mirrors {@code faiss.IndexIVFPQ}.
 *
 * <pre>
 *   IndexFlatIP quantizer = new IndexFlatIP(d);
 *   IndexIVFPQ index = new IndexIVFPQ(quantizer, d, nlist, m, nbits);
 *   index.metric_type = MetricType.METRIC_INNER_PRODUCT;
 *   index.train(base);
 *   index.add(base);
 *   index.nprobe = 32;
 *   SearchResult r = index.search(queries, k);
 * </pre>
 *
 * <p>Coarse quantizer is trained via mini-batch k-means on the training set
 * (or uses an already-trained quantizer). PQ codebooks are trained per sub-vector
 * with k-means (k = 2^{nbits}). Search: probe {@code nprobe} lists, ADC table lookup.
 */
public class IndexIVFPQ extends Index {
    private static final long serialVersionUID = 1L;

    /** Coarse quantizer (typically IndexFlat). */
    public final Index quantizer;
    /** Number of inverted lists (centroids). */
    public final int nlist;
    /** PQ sub-quantizers (d must be divisible by m). */
    public final int m;
    /** Bits per sub-quantizer code (usually 8 → 256 centroids). */
    public final int nbits;
    /** Number of lists to probe at search time (hot-updatable). */
    public int nprobe = 1;

    private final int ksub;       // 1 << nbits
    private final int dsub;       // d / m

    /** Coarse centroids [nlist * d]. */
    private float[] centroids;
    /** PQ codebooks [m][ksub * dsub]. */
    private float[][] codebooks;
    /** Inverted lists: codes (byte or short packed) and ids per list. */
    private List<InvList> invlists;

    /** Own ids by insertion order (positional). */
    private long nextId = 0;

    public IndexIVFPQ(Index quantizer, int d, int nlist, int m, int nbits) {
        super(d, quantizer != null ? quantizer.metric_type : MetricType.METRIC_L2);
        if (quantizer == null) throw new IllegalArgumentException("quantizer required");
        if (nlist <= 0) throw new IllegalArgumentException("nlist must be > 0");
        if (m <= 0 || d % m != 0)
            throw new IllegalArgumentException("d must be divisible by m (d=" + d + " m=" + m + ")");
        if (nbits < 1 || nbits > 16)
            throw new IllegalArgumentException("nbits must be in 1..16");
        this.quantizer = quantizer;
        this.nlist = nlist;
        this.m = m;
        this.nbits = nbits;
        this.ksub = 1 << nbits;
        this.dsub = d / m;
        this.is_trained = false;
        this.invlists = new ArrayList<>(nlist);
        for (int i = 0; i < nlist; i++) invlists.add(new InvList());
    }

    @Override
    public String indexType() {
        return "IVFPQ";
    }

    @Override
    public synchronized void train(float[] x, int n) {
        if (n <= 0) throw new IllegalArgumentException("need training vectors");
        checkDim(x, n);
        if (verbose) System.out.println("IVFPQ train n=" + n + " nlist=" + nlist + " m=" + m);

        // 1) coarse centroids via k-means
        int trainN = Math.min(n, Math.max(nlist * 40, nlist));
        float[] trainX = x;
        if (trainN < n) {
            // subsample
            trainX = new float[trainN * d];
            ThreadLocalRandom rnd = ThreadLocalRandom.current();
            boolean[] seen = new boolean[n];
            for (int i = 0; i < trainN; i++) {
                int r;
                do { r = rnd.nextInt(n); } while (seen[r]);
                seen[r] = true;
                System.arraycopy(x, r * d, trainX, i * d, d);
            }
        }
        centroids = kmeans(trainX, trainN, d, nlist, 25);

        // Load centroids into quantizer
        quantizer.reset();
        quantizer.add(centroids, nlist);

        // 2) assign training points, compute residuals, train PQ on residuals
        int[] assign = assignToCentroids(trainX, trainN);
        float[] residuals = new float[trainN * d];
        for (int i = 0; i < trainN; i++) {
            int c = assign[i];
            int src = i * d, cst = c * d;
            for (int j = 0; j < d; j++) residuals[src + j] = trainX[src + j] - centroids[cst + j];
        }

        codebooks = new float[m][];
        for (int sub = 0; sub < m; sub++) {
            // extract sub-vectors
            float[] subX = new float[trainN * dsub];
            for (int i = 0; i < trainN; i++) {
                System.arraycopy(residuals, i * d + sub * dsub, subX, i * dsub, dsub);
            }
            codebooks[sub] = kmeans(subX, trainN, dsub, ksub, 15);
        }

        is_trained = true;
        if (verbose) System.out.println("IVFPQ train done");
    }

    @Override
    public synchronized void add(float[] x, int n) {
        requireTrained();
        if (n <= 0) return;
        checkDim(x, n);
        int[] assign = assignToCentroids(x, n);
        for (int i = 0; i < n; i++) {
            int list = assign[i];
            // residual
            float[] residual = new float[d];
            int src = i * d, cst = list * d;
            for (int j = 0; j < d; j++) residual[j] = x[src + j] - centroids[cst + j];
            // encode PQ
            byte[] code = encode(residual);
            long id = nextId++;
            invlists.get(list).add(id, code);
        }
        ntotal += n;
    }

    @Override
    public synchronized void add_with_ids(float[] x, int n, long[] ids) {
        requireTrained();
        if (n <= 0) return;
        checkDim(x, n);
        if (ids == null || ids.length < n) throw new IllegalArgumentException("ids");
        int[] assign = assignToCentroids(x, n);
        for (int i = 0; i < n; i++) {
            int list = assign[i];
            float[] residual = new float[d];
            int src = i * d, cst = list * d;
            for (int j = 0; j < d; j++) residual[j] = x[src + j] - centroids[cst + j];
            byte[] code = encode(residual);
            invlists.get(list).add(ids[i], code);
            nextId = Math.max(nextId, ids[i] + 1);
        }
        ntotal += n;
    }

    @Override
    public SearchResult search(float[] xq, int nq, int k) {
        requireTrained();
        if (nq <= 0 || k <= 0 || ntotal == 0) return empty(nq, k);
        checkDim(xq, nq);
        k = (int) Math.min(k, ntotal);
        int probe = Math.min(Math.max(nprobe, 1), nlist);
        boolean lower = metric_type.lowerIsBetter();
        boolean l2 = metric_type == MetricType.METRIC_L2;

        float[][] D = new float[nq][k];
        long[][] I = new long[nq][k];

        // probe lists via quantizer
        SearchResult lists = quantizer.search(xq, nq, probe);

        for (int q = 0; q < nq; q++) {
            // precompute ADC tables for this query residual vs each probed list's...
            // Simpler correct approach: for each probed list, decode residual approx and compare.
            // Faster: precompute dist tables query_sub vs codebook once, then for each list
            // adjust by residual = query - centroid → table on residual subs.

            TopK heap = new TopK(k, lower);
            float[] query = new float[d];
            System.arraycopy(xq, q * d, query, 0, d);

            for (int p = 0; p < probe; p++) {
                long listId = lists.I[q][p];
                if (listId < 0 || listId >= nlist) continue;
                int list = (int) listId;
                // residual query relative to this centroid
                float[] qres = new float[d];
                int cst = list * d;
                for (int j = 0; j < d; j++) qres[j] = query[j] - centroids[cst + j];

                // ADC tables: for each subquantizer, dist(qres_sub, codebook_entry)
                float[][] tables = new float[m][ksub];
                for (int sub = 0; sub < m; sub++) {
                    float[] cb = codebooks[sub];
                    int qOff = sub * dsub;
                    for (int c = 0; c < ksub; c++) {
                        if (l2) {
                            tables[sub][c] = DistanceKernel.l2Row(qres, qOff, cb, c * dsub, dsub);
                        } else {
                            // IP on residual: sum qres·code  (approx of q·x via residual trick is imperfect for IP;
                            // for IP we store codes of residual but score raw approx reconstruct)
                            tables[sub][c] = DistanceKernel.ipRow(qres, qOff, cb, c * dsub, dsub);
                        }
                    }
                }

                InvList inv = invlists.get(list);
                for (int e = 0; e < inv.size; e++) {
                    byte[] code = inv.codes.get(e);
                    float dist = 0f;
                    if (l2) {
                        for (int sub = 0; sub < m; sub++) {
                            int ci = code[sub] & 0xFF;
                            if (nbits > 8) {
                                // packed differently — we use 1 byte when nbits<=8
                            }
                            dist += tables[sub][ci];
                        }
                    } else {
                        // IP: higher better — sum of sub IPs on residual is residual·approx;
                        // add centroid·query for better ranking
                        float centDot = DistanceKernel.ipRow(query, centroids, cst, d);
                        float resDot = 0f;
                        for (int sub = 0; sub < m; sub++) {
                            int ci = code[sub] & 0xFF;
                            resDot += tables[sub][ci];
                        }
                        dist = centDot + resDot;
                    }
                    heap.offer(inv.ids.get(e), dist);
                }
            }
            heap.export(D[q], I[q]);
        }
        return new SearchResult(D, I);
    }

    @Override
    public void reconstruct(long key, float[] recons) {
        // key is positional insertion id by default, or external id if add_with_ids
        // scan lists (OK for debug / small n)
        requireTrained();
        for (int list = 0; list < nlist; list++) {
            InvList inv = invlists.get(list);
            for (int e = 0; e < inv.size; e++) {
                if (inv.ids.get(e) != key) continue;
                byte[] code = inv.codes.get(e);
                // decode residual + centroid
                Arrays.fill(recons, 0, d, 0f);
                for (int sub = 0; sub < m; sub++) {
                    int ci = code[sub] & 0xFF;
                    float[] cb = codebooks[sub];
                    System.arraycopy(cb, ci * dsub, recons, sub * dsub, dsub);
                }
                int cst = list * d;
                for (int j = 0; j < d; j++) recons[j] += centroids[cst + j];
                return;
            }
        }
        throw new IllegalArgumentException("id not found: " + key);
    }

    @Override
    public synchronized void reset() {
        for (InvList inv : invlists) inv.clear();
        ntotal = 0;
        nextId = 0;
    }

    // ---- encode ----

    private byte[] encode(float[] residual) {
        // nbits <= 8 → one byte per sub-code
        if (nbits > 8)
            throw new UnsupportedOperationException("nbits > 8 not yet supported in pure-Java IVFPQ");
        byte[] code = new byte[m];
        for (int sub = 0; sub < m; sub++) {
            float[] cb = codebooks[sub];
            int best = 0;
            float bestD = Float.POSITIVE_INFINITY;
            int qOff = sub * dsub;
            for (int c = 0; c < ksub; c++) {
                float dist = DistanceKernel.l2Row(residual, qOff, cb, c * dsub, dsub);
                if (dist < bestD) { bestD = dist; best = c; }
            }
            code[sub] = (byte) best;
        }
        return code;
    }

    private int[] assignToCentroids(float[] x, int n) {
        int[] assign = new int[n];
        // brute force vs centroids (nlist is typically 1k-4k; OK)
        boolean l2 = metric_type == MetricType.METRIC_L2
            || quantizer.metric_type == MetricType.METRIC_L2;
        // Always use L2 for coarse assignment stability unless quantizer is IP-only trained
        for (int i = 0; i < n; i++) {
            int best = 0;
            float bestD = Float.POSITIVE_INFINITY;
            float bestIP = Float.NEGATIVE_INFINITY;
            int src = i * d;
            for (int c = 0; c < nlist; c++) {
                if (quantizer.metric_type == MetricType.METRIC_INNER_PRODUCT) {
                    float ip = DistanceKernel.ipRow(x, src, centroids, c * d, d);
                    if (ip > bestIP) { bestIP = ip; best = c; }
                } else {
                    float dist = DistanceKernel.l2Row(x, src, centroids, c * d, d);
                    if (dist < bestD) { bestD = dist; best = c; }
                }
            }
            assign[i] = best;
        }
        return assign;
    }

    // ---- k-means ----

    private static float[] kmeans(float[] x, int n, int dim, int k, int niter) {
        // Always emit exactly k centroids (FAISS PQ needs 2^nbits). When n < k,
        // allow duplicate seeds rather than shrinking k (encode loops use full ksub).
        float[] cents = new float[k * dim];
        ThreadLocalRandom rnd = ThreadLocalRandom.current();
        // init: prefer distinct points, fall back to with-replacement when n < k
        boolean[] used = new boolean[n];
        int distinct = 0;
        for (int c = 0; c < k; c++) {
            int r;
            if (distinct < n) {
                int guard = 0;
                do { r = rnd.nextInt(n); guard++; } while (used[r] && guard < n * 4);
                if (!used[r]) { used[r] = true; distinct++; }
            } else {
                r = rnd.nextInt(n);
            }
            System.arraycopy(x, r * dim, cents, c * dim, dim);
        }
        int[] assign = new int[n];
        float[] sum = new float[k * dim];
        int[] counts = new int[k];
        for (int it = 0; it < niter; it++) {
            Arrays.fill(sum, 0f);
            Arrays.fill(counts, 0);
            // assign
            for (int i = 0; i < n; i++) {
                int best = 0;
                float bestD = Float.POSITIVE_INFINITY;
                int src = i * dim;
                for (int c = 0; c < k; c++) {
                    float dist = DistanceKernel.l2Row(x, src, cents, c * dim, dim);
                    if (dist < bestD) { bestD = dist; best = c; }
                }
                assign[i] = best;
                counts[best]++;
                int dst = best * dim;
                for (int j = 0; j < dim; j++) sum[dst + j] += x[src + j];
            }
            // update
            for (int c = 0; c < k; c++) {
                if (counts[c] == 0) {
                    // re-seed empty centroid
                    int r = rnd.nextInt(n);
                    System.arraycopy(x, r * dim, cents, c * dim, dim);
                    continue;
                }
                int dst = c * dim;
                float inv = 1f / counts[c];
                for (int j = 0; j < dim; j++) cents[dst + j] = sum[dst + j] * inv;
            }
        }
        return cents;
    }

    private SearchResult empty(int nq, int k) {
        nq = Math.max(0, nq); k = Math.max(0, k);
        float[][] D = new float[nq][k];
        long[][] I = new long[nq][k];
        float fill = metric_type.lowerIsBetter() ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
        for (int q = 0; q < nq; q++) {
            Arrays.fill(D[q], fill);
            Arrays.fill(I[q], -1);
        }
        return new SearchResult(D, I);
    }

    private static final class InvList implements java.io.Serializable {
        private static final long serialVersionUID = 1L;
        final List<Long> ids = new ArrayList<>();
        final List<byte[]> codes = new ArrayList<>();
        int size;

        void add(long id, byte[] code) {
            ids.add(id);
            codes.add(code);
            size++;
        }

        void clear() {
            ids.clear();
            codes.clear();
            size = 0;
        }
    }

    // ---- FAISS binary IO accessors (package-private) ----

    /** Bytes per PQ code (nbits==8 → m). */
    int pqCodeSize() {
        if (nbits > 8) throw new UnsupportedOperationException("nbits>8");
        // FAISS packs nbits tightly; for nbits==8 each sub-code is 1 byte
        return m * ((nbits + 7) / 8);
    }

    /** Flattened codebooks [m * ksub * dsub] in FAISS PQ order. */
    float[] flatCodebooks() {
        if (codebooks == null) return new float[0];
        float[] flat = new float[m * ksub * dsub];
        for (int sub = 0; sub < m; sub++) {
            float[] cb = codebooks[sub];
            if (cb == null) continue;
            System.arraycopy(cb, 0, flat, sub * ksub * dsub, Math.min(cb.length, ksub * dsub));
        }
        return flat;
    }

    int listSize(int list) {
        if (list < 0 || list >= invlists.size()) return 0;
        return invlists.get(list).size;
    }

    /** Contiguous codes for one list: n * codeSize bytes. */
    byte[] listCodes(int list) {
        InvList inv = invlists.get(list);
        int cs = pqCodeSize();
        byte[] out = new byte[inv.size * cs];
        for (int e = 0; e < inv.size; e++) {
            byte[] c = inv.codes.get(e);
            System.arraycopy(c, 0, out, e * cs, Math.min(cs, c.length));
        }
        return out;
    }

    long[] listIds(int list) {
        InvList inv = invlists.get(list);
        long[] out = new long[inv.size];
        for (int e = 0; e < inv.size; e++) out[e] = inv.ids.get(e);
        return out;
    }

    void loadPqCodebooks(float[] flat, int pqD, int pqM, int pqNbits) {
        if (pqM != m || pqNbits != nbits)
            throw new IllegalArgumentException("PQ shape mismatch file vs index");
        int k = 1 << pqNbits;
        int ds = d / m;
        codebooks = new float[m][];
        int expect = m * k * ds;
        if (flat == null || flat.length < expect)
            throw new IllegalArgumentException("codebook flat too small: "
                + (flat == null ? -1 : flat.length) + " need " + expect);
        for (int sub = 0; sub < m; sub++) {
            codebooks[sub] = new float[k * ds];
            System.arraycopy(flat, sub * k * ds, codebooks[sub], 0, k * ds);
        }
        // coarse centroids live in quantizer
        if (quantizer instanceof IndexFlat qf && qf.ntotal() >= nlist) {
            centroids = Arrays.copyOf(qf.getXb(), nlist * d);
        } else if (quantizer != null && quantizer.ntotal() >= nlist) {
            centroids = new float[nlist * d];
            float[] tmp = new float[d];
            for (int i = 0; i < nlist; i++) {
                try {
                    quantizer.reconstruct(i, tmp);
                    System.arraycopy(tmp, 0, centroids, i * d, d);
                } catch (Exception ignored) {
                    // leave zeros
                }
            }
        } else {
            centroids = new float[nlist * d];
        }
        is_trained = true;
    }

    void loadList(int list, long[] ids, byte[] codes, int codeSize) {
        if (list < 0 || list >= nlist) throw new IllegalArgumentException("list " + list);
        InvList inv = invlists.get(list);
        inv.clear();
        int n = ids == null ? 0 : ids.length;
        int cs = codeSize > 0 ? codeSize : pqCodeSize();
        for (int e = 0; e < n; e++) {
            byte[] code = new byte[m]; // our encode uses m bytes when nbits<=8
            int src = e * cs;
            System.arraycopy(codes, src, code, 0, Math.min(m, cs));
            inv.add(ids[e], code);
        }
    }

    void recomputeNtotalFromLists() {
        long t = 0;
        for (InvList inv : invlists) t += inv.size;
        this.ntotal = t;
    }
}
