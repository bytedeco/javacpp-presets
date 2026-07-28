package org.bytedeco.pytorch.dataframe.faiss;

import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;
import java.util.ArrayList;
import java.util.List;

/**
 * HNSW flat index — mirrors {@code faiss.IndexHNSWFlat(d, M)}.
 *
 * <pre>
 *   IndexHNSWFlat index = new IndexHNSWFlat(DIM, 32);
 *   index.hnsw.efConstruction = 128;
 *   index.hnsw.efSearch = 64;
 *   index.add(base_vecs);
 *   SearchResult r = index.search(query_vecs, K);
 *   // hot-update:
 *   index.hnsw.efSearch = 96;
 * </pre>
 *
 * <p>Performance notes vs pure-Java {@code ann.HnswIndex}:
 * generation-stamp visited (no per-query {@code boolean[n]} alloc),
 * 8-wide distance unroll, compact neighbor arrays, parallel batch search.
 * Graph walk stays on CPU; bulk Flat comparisons may use CUDA via other indexes.
 */
public class IndexHNSWFlat extends Index {
    private static final long serialVersionUID = 1L;

    /** Public FAISS-style param object: {@code index.hnsw.efSearch = ...}. */
    public final HnswParams hnsw;

    private final double levelMult; // 1/ln(M)

    private float[] data;
    private int capacity;
    private int[][][] neighbors; // [node][level] = int[] neighbors
    private int[] levels;
    private int entryPoint = -1;
    private int maxLevel = -1;

    // generation-stamp visited: avoid boolean[] alloc per query
    private transient int[] visitStamp;
    private transient int visitGen = 1;
    private transient ThreadLocal<int[]> localVisitStamp;
    private transient ThreadLocal<Integer> localVisitGen;

    public IndexHNSWFlat(int d, int M) {
        this(d, M, MetricType.METRIC_L2);
    }

    public IndexHNSWFlat(int d, int M, MetricType metric) {
        super(d, metric);
        this.hnsw = new HnswParams(M);
        this.levelMult = 1.0 / Math.log(M);
        this.capacity = 0;
        this.data = new float[0];
        this.neighbors = new int[0][][];
        this.levels = new int[0];
        this.is_trained = true;
        initVisitState();
    }

    private void initVisitState() {
        localVisitStamp = ThreadLocal.withInitial(() -> new int[Math.max(16, capacity)]);
        localVisitGen = ThreadLocal.withInitial(() -> 1);
    }

    private void readObject(java.io.ObjectInputStream in)
            throws java.io.IOException, ClassNotFoundException {
        in.defaultReadObject();
        initVisitState();
    }

    @Override
    public String indexType() {
        return "HNSWFlat";
    }

    public int M() { return hnsw.M; }

    @Override
    public synchronized void add(float[] x, int n) {
        if (n <= 0) return;
        checkDim(x, n);
        ensureCapacity((int) ntotal + n);
        System.arraycopy(x, 0, data, (int) ntotal * d, n * d);
        for (int i = 0; i < n; i++) {
            insertNode((int) ntotal);
            ntotal++;
        }
    }

    @Override
    public synchronized void reset() {
        ntotal = 0;
        entryPoint = -1;
        maxLevel = -1;
    }

    @Override
    public void reconstruct(long key, float[] recons) {
        if (key < 0 || key >= ntotal)
            throw new IllegalArgumentException("reconstruct key out of range: " + key);
        System.arraycopy(data, (int) key * d, recons, 0, d);
    }

    @Override
    public SearchResult search(float[] xq, int nq, int k) {
        if (nq <= 0 || k <= 0) return emptyResult(nq, k);
        checkDim(xq, nq);
        if (ntotal == 0) return emptyResult(nq, k);
        k = (int) Math.min(k, ntotal);
        int ef = Math.max(hnsw.efSearch, k);

        if (nq == 1) {
            float[][] D = new float[1][k];
            long[][] I = new long[1][k];
            searchOne(xq, 0, k, ef, D[0], I[0]);
            return new SearchResult(D, I);
        }

        float[][] D = new float[nq][k];
        long[][] I = new long[nq][k];
        int parallelism = Math.max(1, ForkJoinPool.commonPool().getParallelism());
        int chunk = Math.max(1, (nq + parallelism - 1) / parallelism);
        final int kk = k;
        final int eef = ef;
        RecursiveAction root = new RecursiveAction() {
            @Override protected void compute() {
                List<RecursiveAction> tasks = new ArrayList<>();
                for (int s = 0; s < nq; s += chunk) {
                    final int start = s;
                    final int end = Math.min(nq, s + chunk);
                    tasks.add(new RecursiveAction() {
                        @Override protected void compute() {
                            for (int q = start; q < end; q++) {
                                searchOne(xq, q, kk, eef, D[q], I[q]);
                            }
                        }
                    });
                }
                invokeAll(tasks);
            }
        };
        ForkJoinPool.commonPool().invoke(root);
        return new SearchResult(D, I);
    }

    @Override
    public RangeSearchResult range_search(float[] xq, int nq, float radius) {
        // Approximate: over-fetch with HNSW then filter; also run a larger ef
        int kFetch = (int) Math.min(ntotal, Math.max(hnsw.efSearch * 4, 50));
        SearchResult sr = search(xq, nq, kFetch);
        boolean l2 = metric_type == MetricType.METRIC_L2;
        long[] lims = new long[nq + 1];
        // first pass count
        int total = 0;
        int[] counts = new int[nq];
        for (int q = 0; q < nq; q++) {
            int c = 0;
            for (int j = 0; j < sr.D[q].length; j++) {
                if (sr.I[q][j] < 0) continue;
                boolean keep = l2 ? sr.D[q][j] <= radius : sr.D[q][j] >= radius;
                if (keep) c++;
            }
            counts[q] = c;
            total += c;
            lims[q + 1] = total;
        }
        float[] D = new float[total];
        long[] I = new long[total];
        int p = 0;
        for (int q = 0; q < nq; q++) {
            for (int j = 0; j < sr.D[q].length; j++) {
                if (sr.I[q][j] < 0) continue;
                boolean keep = l2 ? sr.D[q][j] <= radius : sr.D[q][j] >= radius;
                if (!keep) continue;
                D[p] = sr.D[q][j];
                I[p] = sr.I[q][j];
                p++;
            }
        }
        return new RangeSearchResult(lims, D, I);
    }

    // ---- search one query ----

    private void searchOne(float[] xq, int q, int k, int ef, float[] outD, long[] outI) {
        int n = (int) ntotal;
        if (n == 0 || entryPoint < 0) {
            Arrays.fill(outI, -1);
            float fill = metric_type.lowerIsBetter() ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
            Arrays.fill(outD, fill);
            return;
        }
        float[] query = new float[d];
        System.arraycopy(xq, q * d, query, 0, d);

        int curr = entryPoint;
        // greedy from top layer down to 1
        for (int lc = maxLevel; lc > 0; lc--) {
            curr = greedyClosest(curr, query, lc);
        }
        // ef-beam on layer 0
        MinMaxHeap w = searchLayer(query, curr, ef, 0);

        // export top-k (best first)
        int got = Math.min(k, w.size);
        // w is max-heap by distance for L2 (farthest on top for pruning);
        // extract sorted
        int[] idxs = new int[w.size];
        float[] dists = new float[w.size];
        int sz = w.size;
        for (int i = 0; i < sz; i++) {
            idxs[i] = w.ids[i];
            dists[i] = w.dist[i];
        }
        // sort ascending for L2, descending for IP
        sortHeapResult(dists, idxs, sz, metric_type.lowerIsBetter());
        for (int i = 0; i < k; i++) {
            if (i < got) {
                outD[i] = dists[i];
                outI[i] = idxs[i];
            } else {
                outD[i] = metric_type.lowerIsBetter() ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
                outI[i] = -1;
            }
        }
    }

    private int greedyClosest(int enter, float[] query, int lc) {
        int curr = enter;
        float curDist = distQuery(query, curr);
        boolean changed = true;
        while (changed) {
            changed = false;
            int[] nbs = neighbors[curr][lc];
            if (nbs == null) break;
            for (int nb : nbs) {
                float dd = distQuery(query, nb);
                if (better(dd, curDist)) {
                    curDist = dd;
                    curr = nb;
                    changed = true;
                }
            }
        }
        return curr;
    }

    /**
     * Beam search on one layer. Returns candidate set of size &lt;= ef,
     * unordered in heap storage (caller sorts).
     */
    private MinMaxHeap searchLayer(float[] query, int enter, int ef, int lc) {
        int n = (int) ntotal;
        int[] stamp = localVisitStamp.get();
        if (stamp.length < n) {
            stamp = new int[Math.max(n, stamp.length * 2)];
            localVisitStamp.set(stamp);
        }
        int gen = localVisitGen.get();
        gen++;
        if (gen == Integer.MAX_VALUE) {
            Arrays.fill(stamp, 0);
            gen = 1;
        }
        localVisitGen.set(gen);

        // candidates: min-heap (best first to expand); w: result max-heap (worst at root for L2)
        CandidateHeap candidates = new CandidateHeap(ef * 2, true /* min-heap by dist for L2 */);
        // For IP (higher better), invert comparison via flag
        boolean lower = metric_type.lowerIsBetter();
        candidates.lowerIsBetter = lower;

        MinMaxHeap w = new MinMaxHeap(ef, lower);

        float enterDist = distQuery(query, enter);
        candidates.push(enter, enterDist);
        w.push(enter, enterDist);
        stamp[enter] = gen;

        while (!candidates.isEmpty()) {
            int c = candidates.peekId();
            float cDist = candidates.peekDist();
            candidates.pop();

            float worst = w.worstDist();
            if (w.size >= ef && !better(cDist, worst)) break;

            int[] nbs = (c < neighbors.length && neighbors[c] != null
                && lc < neighbors[c].length) ? neighbors[c][lc] : null;
            if (nbs == null) continue;
            for (int nb : nbs) {
                if (nb < 0 || nb >= n) continue;
                if (stamp[nb] == gen) continue;
                stamp[nb] = gen;
                float dd = distQuery(query, nb);
                if (w.size < ef || better(dd, w.worstDist())) {
                    candidates.push(nb, dd);
                    w.push(nb, dd);
                    if (w.size > ef) w.popWorst();
                }
            }
        }
        return w;
    }

    // ---- insert ----

    private void insertNode(int node) {
        int level = randomLevel();
        levels[node] = level;
        neighbors[node] = new int[level + 1][];
        for (int lc = 0; lc <= level; lc++) neighbors[node][lc] = new int[0];

        if (entryPoint < 0) {
            entryPoint = node;
            maxLevel = level;
            return;
        }

        int curr = entryPoint;
        for (int lc = maxLevel; lc > level; lc--) {
            curr = greedyClosestNode(curr, node, lc);
        }

        for (int lc = Math.min(level, maxLevel); lc >= 0; lc--) {
            MinMaxHeap cand = searchLayerNode(node, curr, hnsw.efConstruction, lc);
            int maxM = (lc == 0) ? hnsw.maxM0 : hnsw.M;
            int[] selected = selectNeighbors(node, cand, maxM);
            neighbors[node][lc] = selected;
            for (int nb : selected) {
                addNeighbor(nb, node, lc, maxM);
            }
            if (cand.size > 0) curr = cand.bestId();
        }

        if (level > maxLevel) {
            maxLevel = level;
            entryPoint = node;
        }
    }

    private int greedyClosestNode(int enter, int queryNode, int lc) {
        int curr = enter;
        float curDist = distNN(curr, queryNode);
        boolean changed = true;
        while (changed) {
            changed = false;
            int[] nbs = neighbors[curr][lc];
            if (nbs == null) break;
            for (int nb : nbs) {
                float dd = distNN(nb, queryNode);
                if (better(dd, curDist)) {
                    curDist = dd;
                    curr = nb;
                    changed = true;
                }
            }
        }
        return curr;
    }

    private MinMaxHeap searchLayerNode(int queryNode, int enter, int ef, int lc) {
        // reuse query-style search but distance via distNN
        int n = (int) ntotal; // may equal queryNode if inserting at end; visit size n+1
        int need = Math.max(n, queryNode + 1);
        int[] stamp = localVisitStamp.get();
        if (stamp.length < need) {
            stamp = new int[Math.max(need, stamp.length * 2)];
            localVisitStamp.set(stamp);
        }
        int gen = localVisitGen.get();
        gen++;
        if (gen == Integer.MAX_VALUE) {
            Arrays.fill(stamp, 0);
            gen = 1;
        }
        localVisitGen.set(gen);

        boolean lower = metric_type.lowerIsBetter();
        CandidateHeap candidates = new CandidateHeap(ef * 2, lower);
        MinMaxHeap w = new MinMaxHeap(ef, lower);

        float enterDist = distNN(enter, queryNode);
        candidates.push(enter, enterDist);
        w.push(enter, enterDist);
        stamp[enter] = gen;

        while (!candidates.isEmpty()) {
            int c = candidates.peekId();
            float cDist = candidates.peekDist();
            candidates.pop();
            if (w.size >= ef && !better(cDist, w.worstDist())) break;

            int[] nbs = (c < neighbors.length && neighbors[c] != null
                && lc < neighbors[c].length) ? neighbors[c][lc] : null;
            if (nbs == null) continue;
            for (int nb : nbs) {
                if (nb < 0) continue;
                if (nb < stamp.length && stamp[nb] == gen) continue;
                if (nb >= stamp.length) {
                    int[] nstamp = new int[Math.max(nb + 1, stamp.length * 2)];
                    System.arraycopy(stamp, 0, nstamp, 0, stamp.length);
                    stamp = nstamp;
                    localVisitStamp.set(stamp);
                }
                stamp[nb] = gen;
                float dd = distNN(nb, queryNode);
                if (w.size < ef || better(dd, w.worstDist())) {
                    candidates.push(nb, dd);
                    w.push(nb, dd);
                    if (w.size > ef) w.popWorst();
                }
            }
        }
        return w;
    }

    private void addNeighbor(int node, int nb, int lc, int maxM) {
        int[] cur = neighbors[node][lc];
        for (int x : cur) if (x == nb) return;
        if (cur.length < maxM) {
            int[] n2 = Arrays.copyOf(cur, cur.length + 1);
            n2[cur.length] = nb;
            neighbors[node][lc] = n2;
        } else {
            MinMaxHeap tmp = new MinMaxHeap(maxM + 1, metric_type.lowerIsBetter());
            for (int x : cur) tmp.push(x, distNN(node, x));
            tmp.push(nb, distNN(node, nb));
            neighbors[node][lc] = selectNeighbors(node, tmp, maxM);
        }
    }

    /** Heuristic neighbor selection (FAISS/HNSW paper style simplified). */
    private int[] selectNeighbors(int node, MinMaxHeap cand, int maxM) {
        if (cand.size <= maxM) {
            int[] out = new int[cand.size];
            for (int i = 0; i < cand.size; i++) out[i] = cand.ids[i];
            return out;
        }
        // sort candidates best-first
        int sz = cand.size;
        int[] ids = Arrays.copyOf(cand.ids, sz);
        float[] dist = Arrays.copyOf(cand.dist, sz);
        sortHeapResult(dist, ids, sz, metric_type.lowerIsBetter());

        int[] selected = new int[maxM];
        float[] selDist = new float[maxM];
        int s = 0;
        for (int i = 0; i < sz && s < maxM; i++) {
            int c = ids[i];
            float dc = dist[i];
            boolean ok = true;
            // prune if closer to an already-selected neighbor than to query node
            for (int j = 0; j < s; j++) {
                float dcn = distNN(c, selected[j]);
                if (betterOrEq(dcn, dc)) { ok = false; break; }
            }
            if (ok) {
                selected[s] = c;
                selDist[s] = dc;
                s++;
            }
        }
        // fill remaining with next best if heuristic too strict
        if (s < maxM) {
            for (int i = 0; i < sz && s < maxM; i++) {
                int c = ids[i];
                boolean exists = false;
                for (int j = 0; j < s; j++) if (selected[j] == c) { exists = true; break; }
                if (!exists) selected[s++] = c;
            }
        }
        return Arrays.copyOf(selected, s);
    }

    private int randomLevel() {
        double r = ThreadLocalRandom.current().nextDouble();
        return (int) Math.floor(-Math.log(r) * levelMult);
    }

    // ---- distances ----

    private float distNN(int a, int b) {
        int ba = a * d, bb = b * d;
        if (metric_type == MetricType.METRIC_INNER_PRODUCT) {
            // FAISS IP: higher better — return raw IP (TopK/heaps use lowerIsBetter flag)
            return DistanceKernel.ipRow(data, ba, data, bb, d);
        }
        return DistanceKernel.l2Row(data, ba, data, bb, d);
    }

    private float distQuery(float[] q, int node) {
        if (metric_type == MetricType.METRIC_INNER_PRODUCT) {
            return DistanceKernel.ipRow(q, data, node * d, d);
        }
        return DistanceKernel.l2Row(q, data, node * d, d);
    }

    /** true if a is strictly better than b under current metric. */
    private boolean better(float a, float b) {
        return metric_type.lowerIsBetter() ? a < b : a > b;
    }

    private boolean betterOrEq(float a, float b) {
        return metric_type.lowerIsBetter() ? a <= b : a >= b;
    }

    private void ensureCapacity(int need) {
        if (need <= capacity) return;
        int nc = Math.max(Math.max(16, capacity * 2), need);
        data = Arrays.copyOf(data, nc * d);
        neighbors = Arrays.copyOf(neighbors, nc);
        levels = Arrays.copyOf(levels, nc);
        capacity = nc;
    }

    private SearchResult emptyResult(int nq, int k) {
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

    private static void sortHeapResult(float[] dist, int[] ids, int n, boolean lowerIsBetter) {
        // insertion for small, else boxed sort
        if (n <= 64) {
            for (int i = 1; i < n; i++) {
                float key = dist[i];
                int kid = ids[i];
                int j = i - 1;
                if (lowerIsBetter) {
                    while (j >= 0 && dist[j] > key) {
                        dist[j + 1] = dist[j]; ids[j + 1] = ids[j]; j--;
                    }
                } else {
                    while (j >= 0 && dist[j] < key) {
                        dist[j + 1] = dist[j]; ids[j + 1] = ids[j]; j--;
                    }
                }
                dist[j + 1] = key; ids[j + 1] = kid;
            }
            return;
        }
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) order[i] = i;
        if (lowerIsBetter) Arrays.sort(order, (a, b) -> Float.compare(dist[a], dist[b]));
        else Arrays.sort(order, (a, b) -> Float.compare(dist[b], dist[a]));
        float[] nd = new float[n];
        int[] ni = new int[n];
        for (int i = 0; i < n; i++) { nd[i] = dist[order[i]]; ni[i] = ids[order[i]]; }
        System.arraycopy(nd, 0, dist, 0, n);
        System.arraycopy(ni, 0, ids, 0, n);
    }

    // ---- compact heaps ----

    /** Result set: keeps up to cap elements; worst is easily accessible. */
    static final class MinMaxHeap {
        final int cap;
        final boolean lowerIsBetter;
        final int[] ids;
        final float[] dist;
        int size;

        MinMaxHeap(int cap, boolean lowerIsBetter) {
            this.cap = cap;
            this.lowerIsBetter = lowerIsBetter;
            this.ids = new int[cap + 1];
            this.dist = new float[cap + 1];
            this.size = 0;
        }

        void push(int id, float d) {
            if (size < cap) {
                ids[size] = id;
                dist[size] = d;
                size++;
                return;
            }
            // replace worst if better
            int wi = worstIndex();
            if (lowerIsBetter ? d >= dist[wi] : d <= dist[wi]) return;
            ids[wi] = id;
            dist[wi] = d;
        }

        void popWorst() {
            if (size == 0) return;
            int wi = worstIndex();
            ids[wi] = ids[size - 1];
            dist[wi] = dist[size - 1];
            size--;
        }

        float worstDist() {
            if (size == 0)
                return lowerIsBetter ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
            return dist[worstIndex()];
        }

        int bestId() {
            if (size == 0) return -1;
            int bi = 0;
            for (int i = 1; i < size; i++) {
                if (lowerIsBetter ? dist[i] < dist[bi] : dist[i] > dist[bi]) bi = i;
            }
            return ids[bi];
        }

        private int worstIndex() {
            int wi = 0;
            for (int i = 1; i < size; i++) {
                if (lowerIsBetter ? dist[i] > dist[wi] : dist[i] < dist[wi]) wi = i;
            }
            return wi;
        }
    }

    /** Expansion candidates — simple binary heap. */
    static final class CandidateHeap {
        int[] ids;
        float[] dist;
        int size;
        boolean lowerIsBetter; // min-heap when lower is better (expand closest first)

        CandidateHeap(int cap, boolean lowerIsBetter) {
            this.ids = new int[Math.max(8, cap)];
            this.dist = new float[Math.max(8, cap)];
            this.size = 0;
            this.lowerIsBetter = lowerIsBetter;
        }

        boolean isEmpty() { return size == 0; }

        void push(int id, float d) {
            if (size == ids.length) grow();
            ids[size] = id;
            dist[size] = d;
            siftUp(size);
            size++;
        }

        int peekId() { return ids[0]; }
        float peekDist() { return dist[0]; }

        void pop() {
            if (size == 0) return;
            size--;
            ids[0] = ids[size];
            dist[0] = dist[size];
            if (size > 0) siftDown(0);
        }

        private void grow() {
            int n = ids.length * 2;
            ids = Arrays.copyOf(ids, n);
            dist = Arrays.copyOf(dist, n);
        }

        private void siftUp(int i) {
            while (i > 0) {
                int p = (i - 1) >> 1;
                if (!betterThan(dist[i], dist[p])) break;
                swap(i, p);
                i = p;
            }
        }

        private void siftDown(int i) {
            while (true) {
                int l = (i << 1) + 1;
                if (l >= size) break;
                int r = l + 1;
                int best = l;
                if (r < size && betterThan(dist[r], dist[l])) best = r;
                if (!betterThan(dist[best], dist[i])) break;
                swap(i, best);
                i = best;
            }
        }

        private boolean betterThan(float a, float b) {
            // "better to expand first" = closer for L2, higher for IP
            return lowerIsBetter ? a < b : a > b;
        }

        private void swap(int a, int b) {
            int ti = ids[a]; ids[a] = ids[b]; ids[b] = ti;
            float td = dist[a]; dist[a] = dist[b]; dist[b] = td;
        }
    }

    // ---- FAISS binary IO accessors (package-private) ----

    /** Row-major vectors currently stored (length may exceed ntotal*d). */
    float[] storageData() {
        int n = (int) ntotal;
        if (n <= 0) return new float[0];
        return Arrays.copyOf(data, n * d);
    }

    int levelOf(int node) {
        return (node >= 0 && node < levels.length) ? levels[node] : 0;
    }

    int[] neighborsOf(int node, int level) {
        if (node < 0 || node >= neighbors.length || neighbors[node] == null) return new int[0];
        if (level < 0 || level >= neighbors[node].length) return new int[0];
        int[] nbs = neighbors[node][level];
        return nbs == null ? new int[0] : nbs;
    }

    int entryPoint() { return entryPoint; }

    int maxLevel() { return maxLevel; }

    /**
     * Bulk-load vectors + FAISS HNSW graph (from {@link NativeFaissIO}).
     * Does <em>not</em> re-run insert; links come from the file.
     */
    synchronized void loadFromFaiss(float[] xb, int n, NativeFaissIO.HnswGraphData g) {
        if (n < 0) throw new IllegalArgumentException("n < 0");
        reset();
        if (n == 0) {
            this.entryPoint = g.entryPoint;
            this.maxLevel = g.maxLevel;
            return;
        }
        ensureCapacity(n);
        System.arraycopy(xb, 0, data, 0, n * d);
        // FAISS levels[i] stores (max_level_of_node + 1)
        for (int i = 0; i < n; i++) {
            int faissLevel = (g.levels != null && i < g.levels.length) ? g.levels[i] : 1;
            int maxLc = Math.max(0, faissLevel - 1);
            levels[i] = maxLc;
            neighbors[i] = new int[maxLc + 1][];
            for (int lc = 0; lc <= maxLc; lc++) {
                int begin = (int) (g.offsets[i] + cumNb(g.cumNNeighbor, lc));
                int end = (int) (g.offsets[i] + cumNb(g.cumNNeighbor, lc + 1));
                if (end > g.neighbors.length) end = g.neighbors.length;
                if (begin < 0) begin = 0;
                // compact non-(-1) neighbors
                int count = 0;
                for (int j = begin; j < end; j++) {
                    if (g.neighbors[j] >= 0) count++;
                }
                int[] nbs = new int[count];
                int p = 0;
                for (int j = begin; j < end; j++) {
                    if (g.neighbors[j] >= 0) nbs[p++] = g.neighbors[j];
                }
                neighbors[i][lc] = nbs;
            }
        }
        this.ntotal = n;
        this.entryPoint = g.entryPoint;
        this.maxLevel = g.maxLevel;
        if (this.entryPoint < 0 && n > 0) this.entryPoint = 0;
        if (this.maxLevel < 0 && n > 0) {
            int ml = 0;
            for (int i = 0; i < n; i++) if (levels[i] > ml) ml = levels[i];
            this.maxLevel = ml;
        }
    }

    private static int cumNb(int[] cum, int layerNo) {
        if (cum == null || cum.length == 0) return 0;
        if (layerNo < 0) return 0;
        if (layerNo >= cum.length) return cum[cum.length - 1];
        return cum[layerNo];
    }
}
