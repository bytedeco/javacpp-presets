package org.bytedeco.pytorch.data.dataframe.ann;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Hierarchical Navigable Small World (HNSW) index — FAISS {@code IndexHNSWFlat}-like API.
 *
 * <p>Pure Java implementation of Malkov &amp; Yashunin HNSW:
 * multi-layer graph, greedy search on upper layers, ef-beam search on layer 0,
 * heuristic neighbor selection.
 *
 * <pre>
 *   HnswIndex idx = HnswIndex.builder(dim)
 *       .M(16).efConstruction(200).space(Distance.L2)
 *       .build();
 *   idx.add(vectors);                 // float[n*dim] row-major, or float[][]
 *   AnnSearchResult r = idx.search(query, 10, 64);
 * </pre>
 */
public final class HnswIndex implements Serializable {
    private static final long serialVersionUID = 2L;

    private final int dim;
    private final int M;
    private final int maxM0;           // max neighbors on layer 0 (= 2*M)
    private final int efConstruction;
    private final Distance space;
    private final double levelMult;    // 1/ln(M)
    private final boolean normalize;

    // storage
    private float[] data;              // row-major [capacity * dim]
    private int size;                  // number of vectors
    private int capacity;
    private long[] ids;                // optional external ids (size-aligned); -1 if unused
    private boolean hasIds;

    // graph: neighbors[node][level] = int[] of neighbor node indices (variable length ≤ maxM)
    private int[][][] neighbors;
    private int[] levels;              // max level of each node
    private int entryPoint = -1;
    private int maxLevel = -1;

    private HnswIndex(int dim, int M, int efConstruction, Distance space, boolean normalize, int initialCap) {
        if (dim <= 0) throw new IllegalArgumentException("dim must be > 0");
        if (M < 2) throw new IllegalArgumentException("M must be >= 2");
        this.dim = dim;
        this.M = M;
        this.maxM0 = M * 2;
        this.efConstruction = Math.max(efConstruction, M);
        this.space = space == null ? Distance.L2 : space;
        this.levelMult = 1.0 / Math.log(M);
        this.normalize = normalize;
        this.capacity = Math.max(16, initialCap);
        this.data = new float[this.capacity * dim];
        this.neighbors = new int[this.capacity][][];
        this.levels = new int[this.capacity];
        this.ids = new long[this.capacity];
        Arrays.fill(this.ids, -1L);
        this.size = 0;
        this.hasIds = false;
    }

    public static Builder builder(int dim) { return new Builder(dim); }

    public int dim() { return dim; }
    public int size() { return size; }
    public int M() { return M; }
    public int efConstruction() { return efConstruction; }
    public Distance space() { return space; }

    // ---- builder ----

    public static final class Builder {
        private final int dim;
        private int M = 16;
        private int efConstruction = 200;
        private Distance space = Distance.L2;
        private boolean normalize = false;
        private int initialCap = 1024;
        private float[] matrix;
        private int n;
        private long[] ids;

        Builder(int dim) { this.dim = dim; }

        public Builder M(int m) { this.M = m; return this; }
        public Builder efConstruction(int ef) { this.efConstruction = ef; return this; }
        public Builder space(Distance d) { this.space = d; return this; }
        public Builder normalize(boolean v) { this.normalize = v; return this; }
        public Builder initialCapacity(int c) { this.initialCap = c; return this; }

        /** Preload vectors (row-major n*dim) to add on {@link #build()}. */
        public Builder vectors(float[] matrix, int n) {
            this.matrix = matrix; this.n = n; return this;
        }
        public Builder vectors(float[][] rows) {
            if (rows == null || rows.length == 0) { this.matrix = new float[0]; this.n = 0; return this; }
            int d = rows[0].length;
            float[] m = new float[rows.length * d];
            for (int i = 0; i < rows.length; i++) {
                if (rows[i] == null || rows[i].length != d)
                    throw new IllegalArgumentException("ragged vectors at " + i);
                System.arraycopy(rows[i], 0, m, i * d, d);
            }
            this.matrix = m; this.n = rows.length; return this;
        }
        public Builder ids(long[] ids) { this.ids = ids; return this; }

        public HnswIndex build() {
            HnswIndex idx = new HnswIndex(dim, M, efConstruction, space, normalize,
                matrix != null ? Math.max(initialCap, n) : initialCap);
            if (matrix != null && n > 0) {
                idx.add(matrix, n, ids);
            }
            return idx;
        }
    }

    // ---- add ----

    public synchronized void add(float[][] rows) {
        if (rows == null || rows.length == 0) return;
        float[] m = new float[rows.length * dim];
        for (int i = 0; i < rows.length; i++) {
            if (rows[i] == null || rows[i].length != dim)
                throw new IllegalArgumentException("vector dim mismatch at " + i);
            System.arraycopy(rows[i], 0, m, i * dim, dim);
        }
        add(m, rows.length, null);
    }

    public synchronized void add(float[] matrix, int n) {
        add(matrix, n, null);
    }

    public synchronized void add(float[] matrix, int n, long[] externalIds) {
        if (n <= 0) return;
        if (matrix.length < n * dim) throw new IllegalArgumentException("matrix too small");
        ensureCapacity(size + n);
        if (normalize) {
            // copy+normalize into storage
            for (int i = 0; i < n; i++) {
                int src = i * dim;
                int dst = (size + i) * dim;
                float sum = 0f;
                for (int d = 0; d < dim; d++) sum += matrix[src + d] * matrix[src + d];
                float inv = sum > 0f ? (float) (1.0 / Math.sqrt(sum)) : 1f;
                for (int d = 0; d < dim; d++) data[dst + d] = matrix[src + d] * inv;
            }
        } else {
            System.arraycopy(matrix, 0, data, size * dim, n * dim);
        }
        if (externalIds != null) {
            hasIds = true;
            for (int i = 0; i < n; i++) {
                ids[size + i] = i < externalIds.length ? externalIds[i] : (size + i);
            }
        }
        // Insert one-by-one so graph search sees already-inserted nodes (size advances).
        for (int i = 0; i < n; i++) {
            insertNode(size);
            size++;
        }
    }

    public synchronized void addOne(float[] vector) {
        addOne(vector, -1L);
    }

    public synchronized void addOne(float[] vector, long id) {
        if (vector == null || vector.length != dim)
            throw new IllegalArgumentException("vector dim mismatch");
        ensureCapacity(size + 1);
        int base = size * dim;
        if (normalize) {
            float sum = 0f;
            for (float v : vector) sum += v * v;
            float inv = sum > 0f ? (float) (1.0 / Math.sqrt(sum)) : 1f;
            for (int d = 0; d < dim; d++) data[base + d] = vector[d] * inv;
        } else {
            System.arraycopy(vector, 0, data, base, dim);
        }
        if (id >= 0) { hasIds = true; ids[size] = id; }
        insertNode(size);
        size++;
    }

    private void ensureCapacity(int need) {
        if (need <= capacity) return;
        int nc = Math.max(capacity * 2, need);
        data = Arrays.copyOf(data, nc * dim);
        neighbors = Arrays.copyOf(neighbors, nc);
        levels = Arrays.copyOf(levels, nc);
        ids = Arrays.copyOf(ids, nc);
        capacity = nc;
    }

    private int randomLevel() {
        double r = ThreadLocalRandom.current().nextDouble();
        // P(level >= l) = M^{-l}  ≈  exp(-l / levelMult) wait: levelMult = 1/ln(M)
        // level = floor(-ln(unif) * levelMult)
        return (int) Math.floor(-Math.log(r) * levelMult);
    }

    private void insertNode(int node) {
        int level = randomLevel();
        levels[node] = level;
        neighbors[node] = new int[level + 1][];
        for (int lc = 0; lc <= level; lc++) {
            neighbors[node][lc] = new int[0];
        }

        if (entryPoint < 0) {
            entryPoint = node;
            maxLevel = level;
            return;
        }

        int curr = entryPoint;
        // greedy search from top layer down to level+1
        for (int lc = maxLevel; lc > level; lc--) {
            curr = greedyClosest(curr, node, lc);
        }

        // for each layer from min(level, maxLevel) down to 0:
        for (int lc = Math.min(level, maxLevel); lc >= 0; lc--) {
            Neighbors cand = searchLayer(node, curr, efConstruction, lc, true /* queryIsNode */);
            int maxM = (lc == 0) ? maxM0 : M;
            int[] selected = selectNeighborsHeuristic(node, cand, maxM, lc);
            neighbors[node][lc] = selected;

            // add reverse edges
            for (int nb : selected) {
                addNeighbor(nb, node, lc, maxM);
            }
            // enter next lower layer from nearest candidate
            if (!cand.isEmpty()) curr = cand.nearest();
        }

        if (level > maxLevel) {
            maxLevel = level;
            entryPoint = node;
        }
    }

    private void addNeighbor(int node, int nb, int lc, int maxM) {
        int[] cur = neighbors[node][lc];
        // already connected?
        for (int x : cur) if (x == nb) return;
        if (cur.length < maxM) {
            int[] n2 = Arrays.copyOf(cur, cur.length + 1);
            n2[cur.length] = nb;
            neighbors[node][lc] = n2;
        } else {
            // shrink with heuristic including the new neighbor
            Neighbors tmp = new Neighbors(maxM + 1);
            for (int x : cur) tmp.add(x, distNN(node, x));
            tmp.add(nb, distNN(node, nb));
            neighbors[node][lc] = selectNeighborsHeuristic(node, tmp, maxM, lc);
        }
    }

    // ---- distance helpers ----

    private float distNN(int a, int b) {
        int ba = a * dim, bb = b * dim;
        switch (space) {
            case L2: {
                float s = 0f;
                for (int i = 0; i < dim; i++) {
                    float d = data[ba + i] - data[bb + i];
                    s += d * d;
                }
                return s;
            }
            case IP: {
                float s = 0f;
                for (int i = 0; i < dim; i++) s += data[ba + i] * data[bb + i];
                return -s;
            }
            case COSINE: {
                float dot = 0f, na = 0f, nb = 0f;
                for (int i = 0; i < dim; i++) {
                    float av = data[ba + i], bv = data[bb + i];
                    dot += av * bv; na += av * av; nb += bv * bv;
                }
                if (na == 0f || nb == 0f) return 1f;
                float cos = dot / (float) (Math.sqrt(na) * Math.sqrt(nb));
                if (cos > 1f) cos = 1f; if (cos < -1f) cos = -1f;
                return 1f - cos;
            }
            default: return Float.MAX_VALUE;
        }
    }

    private float distQuery(float[] q, int node) {
        return space.distance(q, data, node, dim);
    }

    // ---- search primitives ----

    private int greedyClosest(int enter, int queryNode, int lc) {
        int curr = enter;
        float curDist = distNN(curr, queryNode);
        boolean changed = true;
        while (changed) {
            changed = false;
            int[] nbs = neighbors[curr][lc];
            if (nbs == null) break;
            for (int nb : nbs) {
                float d = distNN(nb, queryNode);
                if (d < curDist) {
                    curDist = d;
                    curr = nb;
                    changed = true;
                }
            }
        }
        return curr;
    }

    private int greedyClosestQuery(int enter, float[] query, int lc) {
        int curr = enter;
        float curDist = distQuery(query, curr);
        boolean changed = true;
        while (changed) {
            changed = false;
            int[] nbs = neighbors[curr][lc];
            if (nbs == null) break;
            for (int nb : nbs) {
                float d = distQuery(query, nb);
                if (d < curDist) {
                    curDist = d;
                    curr = nb;
                    changed = true;
                }
            }
        }
        return curr;
    }

    /**
     * Beam search on a single layer.
     * @param queryNode if queryIsNode, the node index being inserted; else ignored
     * @param query if !queryIsNode, the query vector
     */
    private Neighbors searchLayer(int queryNode, float[] query, int enter, int ef, int lc, boolean queryIsNode) {
        Neighbors candidates = new Neighbors(ef * 2); // min-heap by distance (candidates)
        Neighbors w = new Neighbors(ef * 2);          // result set
        boolean[] visited = new boolean[size + 1]; // size may equal queryNode if inserting
        // enlarge if inserting beyond current size
        if (queryIsNode && queryNode >= visited.length) {
            visited = new boolean[queryNode + 1];
        }

        float enterDist = queryIsNode ? distNN(enter, queryNode) : distQuery(query, enter);
        candidates.add(enter, enterDist);
        w.add(enter, enterDist);
        if (enter < visited.length) visited[enter] = true;

        while (!candidates.isEmpty()) {
            int c = candidates.pollNearest();
            float cDist = candidates.lastPolledDist;
            float farthest = w.farthestDist();
            if (cDist > farthest && w.size() >= ef) break;

            int[] nbs = (c < neighbors.length && neighbors[c] != null && lc < neighbors[c].length)
                ? neighbors[c][lc] : null;
            if (nbs == null) continue;
            for (int nb : nbs) {
                if (nb < 0) continue;
                if (nb < visited.length && visited[nb]) continue;
                if (nb < visited.length) visited[nb] = true;
                else {
                    // expand visited
                    boolean[] v2 = new boolean[Math.max(nb + 1, visited.length * 2)];
                    System.arraycopy(visited, 0, v2, 0, visited.length);
                    visited = v2;
                    visited[nb] = true;
                }
                float d = queryIsNode ? distNN(nb, queryNode) : distQuery(query, nb);
                if (w.size() < ef || d < w.farthestDist()) {
                    candidates.add(nb, d);
                    w.add(nb, d);
                    if (w.size() > ef) w.pollFarthest();
                }
            }
        }
        return w;
    }

    private Neighbors searchLayer(int queryNode, int enter, int ef, int lc, boolean queryIsNode) {
        return searchLayer(queryNode, null, enter, ef, lc, queryIsNode);
    }

    /**
     * Heuristic neighbor selection (algorithm 4 from the HNSW paper).
     * Prefer diverse neighbors: keep candidate if farther from already selected than from query.
     */
    private int[] selectNeighborsHeuristic(int queryNode, Neighbors cand, int maxM, int lc) {
        if (cand.size() <= maxM) {
            int[] out = new int[cand.size()];
            int i = 0;
            for (int id : cand.toSortedIds()) out[i++] = id;
            return out;
        }
        // work on sorted-by-distance list
        int[] sorted = cand.toSortedIds();
        float[] sortedDist = cand.toSortedDists();
        List<Integer> selected = new ArrayList<>(maxM);
        for (int i = 0; i < sorted.length && selected.size() < maxM; i++) {
            int c = sorted[i];
            float dqc = sortedDist[i];
            boolean ok = true;
            for (int s : selected) {
                float dsc = distNN(s, c);
                if (dsc < dqc) { ok = false; break; }
            }
            if (ok) selected.add(c);
        }
        // if still short, fill with nearest remaining
        if (selected.size() < maxM) {
            Set<Integer> have = new HashSet<>(selected);
            for (int c : sorted) {
                if (selected.size() >= maxM) break;
                if (have.add(c)) selected.add(c);
            }
        }
        int[] out = new int[selected.size()];
        for (int i = 0; i < selected.size(); i++) out[i] = selected.get(i);
        return out;
    }

    // ---- public search ----

    public AnnSearchResult search(float[] query, int k) {
        return search(query, k, Math.max(efConstruction / 2, k));
    }

    public AnnSearchResult search(float[] query, int k, int efSearch) {
        if (size == 0) return new AnnSearchResult(new int[0], new float[0], new long[0]);
        if (query == null || query.length != dim)
            throw new IllegalArgumentException("query dim mismatch");
        float[] q = query;
        if (normalize) {
            q = Arrays.copyOf(query, dim);
            float sum = 0f;
            for (float v : q) sum += v * v;
            if (sum > 0f) {
                float inv = (float) (1.0 / Math.sqrt(sum));
                for (int i = 0; i < dim; i++) q[i] *= inv;
            }
        }
        int ef = Math.max(efSearch, k);
        int curr = entryPoint;
        for (int lc = maxLevel; lc > 0; lc--) {
            curr = greedyClosestQuery(curr, q, lc);
        }
        Neighbors top = searchLayer(-1, q, curr, ef, 0, false);
        // extract k nearest
        int[] idsSorted = top.toSortedIds();
        float[] distSorted = top.toSortedDists();
        int kk = Math.min(k, idsSorted.length);
        int[] indices = Arrays.copyOf(idsSorted, kk);
        float[] distances = Arrays.copyOf(distSorted, kk);
        long[] outIds = new long[kk];
        for (int i = 0; i < kk; i++) {
            outIds[i] = hasIds ? this.ids[indices[i]] : indices[i];
        }
        return new AnnSearchResult(indices, distances, outIds);
    }

    /** Batch search: one result per query row. */
    public AnnSearchResult[] searchBatch(float[] queries, int nq, int k, int efSearch) {
        AnnSearchResult[] out = new AnnSearchResult[nq];
        for (int i = 0; i < nq; i++) {
            float[] q = Arrays.copyOfRange(queries, i * dim, (i + 1) * dim);
            out[i] = search(q, k, efSearch);
        }
        return out;
    }

    /** Brute-force ground truth (for recall benchmarks). */
    public AnnSearchResult bruteForce(float[] query, int k) {
        if (size == 0) return new AnnSearchResult(new int[0], new float[0], new long[0]);
        float[] q = query;
        if (normalize) {
            q = Arrays.copyOf(query, dim);
            float sum = 0f; for (float v : q) sum += v * v;
            if (sum > 0f) {
                float inv = (float) (1.0 / Math.sqrt(sum));
                for (int i = 0; i < dim; i++) q[i] *= inv;
            }
        }
        // max-heap of size k (farthest on top) by storing as Neighbors
        Neighbors w = new Neighbors(k + 1);
        for (int i = 0; i < size; i++) {
            float d = distQuery(q, i);
            if (w.size() < k) w.add(i, d);
            else if (d < w.farthestDist()) {
                w.add(i, d);
                w.pollFarthest();
            }
        }
        int[] idsSorted = w.toSortedIds();
        float[] distSorted = w.toSortedDists();
        long[] outIds = new long[idsSorted.length];
        for (int i = 0; i < idsSorted.length; i++) {
            outIds[i] = hasIds ? this.ids[idsSorted[i]] : idsSorted[i];
        }
        return new AnnSearchResult(idsSorted, distSorted, outIds);
    }

    // ---- persist ----

    public void save(String path) throws IOException {
        save(Path.of(path));
    }

    public void save(Path path) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(
                new BufferedOutputStream(Files.newOutputStream(path)))) {
            oos.writeObject(this);
        }
    }

    public static HnswIndex load(String path) throws IOException, ClassNotFoundException {
        return load(Path.of(path));
    }

    public static HnswIndex load(Path path) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(
                new BufferedInputStream(Files.newInputStream(path)))) {
            return (HnswIndex) ois.readObject();
        }
    }

    // ---- copy vector out ----

    public float[] getVector(int index) {
        if (index < 0 || index >= size) throw new IndexOutOfBoundsException();
        return Arrays.copyOfRange(data, index * dim, (index + 1) * dim);
    }

    // ================================================================
    // Neighbor priority structures
    // ================================================================

    /**
     * Dual-purpose neighbor set: tracks (id, dist) pairs.
     * Supports nearest-poll (for candidates) and farthest-poll (for result set trimming).
     */
    static final class Neighbors {
        private final ArrayList<long[]> heap = new ArrayList<>(); // [id, distBits]
        float lastPolledDist;

        Neighbors(int capHint) { heap.ensureCapacity(capHint); }

        int size() { return heap.size(); }
        boolean isEmpty() { return heap.isEmpty(); }

        void add(int id, float dist) {
            heap.add(new long[]{id, Float.floatToIntBits(dist)});
        }

        float farthestDist() {
            if (heap.isEmpty()) return Float.POSITIVE_INFINITY;
            float max = Float.NEGATIVE_INFINITY;
            for (long[] e : heap) {
                float d = Float.intBitsToFloat((int) e[1]);
                if (d > max) max = d;
            }
            return max;
        }

        int nearest() {
            int best = -1;
            float min = Float.POSITIVE_INFINITY;
            for (long[] e : heap) {
                float d = Float.intBitsToFloat((int) e[1]);
                if (d < min) { min = d; best = (int) e[0]; }
            }
            return best;
        }

        int pollNearest() {
            int bi = -1;
            float min = Float.POSITIVE_INFINITY;
            for (int i = 0; i < heap.size(); i++) {
                float d = Float.intBitsToFloat((int) heap.get(i)[1]);
                if (d < min) { min = d; bi = i; }
            }
            long[] e = heap.remove(bi);
            lastPolledDist = min;
            return (int) e[0];
        }

        void pollFarthest() {
            int bi = -1;
            float max = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < heap.size(); i++) {
                float d = Float.intBitsToFloat((int) heap.get(i)[1]);
                if (d > max) { max = d; bi = i; }
            }
            if (bi >= 0) heap.remove(bi);
        }

        int[] toSortedIds() {
            heap.sort(Comparator.comparingDouble(e -> Float.intBitsToFloat((int) e[1])));
            int[] out = new int[heap.size()];
            for (int i = 0; i < heap.size(); i++) out[i] = (int) heap.get(i)[0];
            return out;
        }

        float[] toSortedDists() {
            heap.sort(Comparator.comparingDouble(e -> Float.intBitsToFloat((int) e[1])));
            float[] out = new float[heap.size()];
            for (int i = 0; i < heap.size(); i++) out[i] = Float.intBitsToFloat((int) heap.get(i)[1]);
            return out;
        }
    }
}
