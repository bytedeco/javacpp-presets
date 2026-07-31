package dataframe;

import org.bytedeco.pytorch.dataframe.faiss.*;

import java.nio.file.*;
import java.util.*;

/**
 * Native FAISS binary IO round-trip + optional Python cross-check.
 *
 * <pre>
 *   java ... dataframe.BenchmarkFaissNativeIO
 *   # with python faiss-cpu:
 *   java ... dataframe.BenchmarkFaissNativeIO --python
 * </pre>
 */
public class BenchmarkFaissNativeIO {
    static int passed = 0, failed = 0, skipped = 0;

    static void check(String name, boolean ok) {
        if (ok) {
            passed++;
            System.out.println("  OK  " + name);
        } else {
            failed++;
            System.out.println(" FAIL " + name);
            throw new AssertionError(name);
        }
    }

    static float[] randVecs(int n, int d, long seed) {
        Random rnd = new Random(seed);
        float[] v = new float[n * d];
        for (int i = 0; i < v.length; i++) v[i] = (float) rnd.nextGaussian();
        Faiss.normalize_L2(v, n, d);
        return v;
    }

    static void hexHead(Path p, int n) throws Exception {
        byte[] b = Files.readAllBytes(p);
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < Math.min(n, b.length); i++) {
            sb.append(String.format("%02x ", b[i]));
        }
        System.out.println("      head[" + p.getFileName() + "]: " + sb
            + " fourcc=" + NativeFaissIO.fourccString(
                (b[0] & 0xff) | ((b[1] & 0xff) << 8) | ((b[2] & 0xff) << 16) | ((b[3] & 0xff) << 24)));
    }

    public static void main(String[] args) throws Exception {
        boolean doPython = Arrays.asList(args).contains("--python");
        Path tmp = Files.createTempDirectory("faiss-native-io-");
        System.out.println("=== BenchmarkFaissNativeIO === dir=" + tmp);

        final int d = 32, n = 200, nq = 5, k = 5;
        float[] xb = randVecs(n, d, 1);
        float[] xq = randVecs(nq, d, 2);

        try {
            // ---- FlatL2 ----
            {
                IndexFlatL2 idx = new IndexFlatL2(d);
                idx.add(xb, n);
                Path p = tmp.resolve("flat_l2.faiss");
                Faiss.write_index(idx, p);
                hexHead(p, 16);
                check("flat native file", Faiss.is_native_faiss_file(p));
                Index loaded = Faiss.read_index(p);
                check("flat type", loaded instanceof IndexFlatL2 || loaded instanceof IndexFlat);
                check("flat ntotal", loaded.ntotal() == n);
                SearchResult a = idx.search(xq, nq, k);
                SearchResult b = loaded.search(xq, nq, k);
                check("flat same top1", a.I[0][0] == b.I[0][0]);
                float[] recon = loaded.reconstruct(0);
                check("flat recon dim", recon.length == d);
            }

            // ---- FlatIP + IDMap ----
            {
                IndexFlatIP raw = new IndexFlatIP(d);
                IndexIDMap map = new IndexIDMap(raw);
                long[] ids = new long[n];
                for (int i = 0; i < n; i++) ids[i] = 1000 + i;
                map.add_with_ids(xb, n, ids);
                Path p = tmp.resolve("flat_ip_idmap.faiss");
                Faiss.write_index(map, p);
                hexHead(p, 16);
                Index loaded = Faiss.read_index(p);
                check("idmap type", loaded instanceof IndexIDMap);
                SearchResult r = loaded.search(xq, 1, k);
                check("idmap id in range", r.I[0][0] >= 1000 && r.I[0][0] < 1000 + n);
            }

            // ---- HNSWFlat ----
            {
                IndexHNSWFlat h = new IndexHNSWFlat(d, 16, MetricType.METRIC_L2);
                h.hnsw.efConstruction = 64;
                h.hnsw.efSearch = 32;
                h.add(xb, n);
                Path p = tmp.resolve("hnsw.faiss");
                Faiss.write_index(h, p);
                hexHead(p, 16);
                Index loaded = Faiss.read_index(p);
                check("hnsw type", loaded instanceof IndexHNSWFlat);
                check("hnsw ntotal", loaded.ntotal() == n);
                IndexHNSWFlat hl = (IndexHNSWFlat) loaded;
                check("hnsw efSearch preserved", hl.hnsw.efSearch == 32);
                SearchResult a = h.search(xq, nq, k);
                SearchResult b = loaded.search(xq, nq, k);
                // graph reload should be exact → identical neighbors
                int agree = 0;
                for (int q = 0; q < nq; q++) {
                    if (a.I[q][0] == b.I[q][0]) agree++;
                }
                System.out.println("      hnsw top1 agree " + agree + "/" + nq);
                check("hnsw top1 agree all", agree == nq);
            }

            // ---- IVFPQ ----
            {
                int nlist = 16, m = 8;
                IndexFlatL2 quant = new IndexFlatL2(d);
                IndexIVFPQ ivf = new IndexIVFPQ(quant, d, nlist, m, 8);
                ivf.train(xb, n);
                ivf.add(xb, n);
                ivf.nprobe = 4;
                Path p = tmp.resolve("ivfpq.faiss");
                Faiss.write_index(ivf, p);
                hexHead(p, 16);
                Index loaded = Faiss.read_index(p);
                check("ivfpq type", loaded instanceof IndexIVFPQ);
                check("ivfpq ntotal", loaded.ntotal() == n);
                ((IndexIVFPQ) loaded).nprobe = 4;
                SearchResult r = loaded.search(xq, nq, k);
                check("ivfpq k", r.k() == k);
                check("ivfpq has results", r.I[0][0] >= 0);
            }

            // ---- JDF1 still works ----
            {
                IndexFlatL2 idx = new IndexFlatL2(d);
                idx.add(xb, n);
                Path p = tmp.resolve("flat.jdf1");
                Faiss.write_index_jdf1(idx, p);
                check("jdf1 not native", !Faiss.is_native_faiss_file(p));
                Index loaded = Faiss.read_index(p); // auto-detect
                check("jdf1 load ntotal", loaded.ntotal() == n);
            }

            // ---- Python cross-check ----
            if (doPython) {
                runPythonCross(tmp, d, n, xb, xq, k);
            } else {
                skipped++;
                System.out.println(" SKIP python cross (pass --python to enable)");
            }

        } finally {
            try {
                Files.walk(tmp).sorted(Comparator.reverseOrder()).forEach(p -> {
                    try { Files.deleteIfExists(p); } catch (Exception ignored) {}
                });
            } catch (Exception ignored) {}
        }

        System.out.println("passed=" + passed + " failed=" + failed + " skipped=" + skipped);
        if (failed > 0) System.exit(1);
        System.out.println("ALL OK");
    }

    static void runPythonCross(Path tmp, int d, int n, float[] xb, float[] xq, int k) throws Exception {
        // 1) Python writes FlatL2 → Java reads
        Path pyFlat = tmp.resolve("py_flat_l2.faiss");
        Path npyXb = tmp.resolve("xb.npy");
        // write xb as raw float32 for python via a tiny script using numpy from list is heavy;
        // instead embed vectors in the python script as base64 or write binary
        Path xbBin = tmp.resolve("xb.f32");
        Path xqBin = tmp.resolve("xq.f32");
        writeF32(xbBin, xb);
        writeF32(xqBin, xq);

        String pyWrite = ""
            + "import faiss, numpy as np, struct, sys\n"
            + "d, n = %d, %d\n".formatted(d, n)
            + "xb = np.fromfile('%s', dtype='<f4').reshape(n, d)\n".formatted(xbBin)
            + "index = faiss.IndexFlatL2(d)\n"
            + "index.add(xb)\n"
            + "faiss.write_index(index, '%s')\n".formatted(pyFlat)
            + "print('py wrote', index.ntotal)\n";
        Path pyScript = tmp.resolve("write_flat.py");
        Files.writeString(pyScript, pyWrite);
        runPy(pyScript);

        Index fromPy = Faiss.read_index(pyFlat);
        check("java read py flat ntotal", fromPy.ntotal() == n);
        check("java read py flat type", fromPy instanceof IndexFlat);
        SearchResult r = fromPy.search(xq, 1, k);
        check("java search py-flat works", r.I[0][0] >= 0);

        // 2) Java writes FlatL2 → Python reads
        Path jFlat = tmp.resolve("java_flat_l2.faiss");
        IndexFlatL2 jidx = new IndexFlatL2(d);
        jidx.add(xb, n);
        Faiss.write_index(jidx, jFlat);

        String pyRead = ""
            + "import faiss, numpy as np\n"
            + "d, n, k = %d, %d, %d\n".formatted(d, n, k)
            + "index = faiss.read_index('%s')\n".formatted(jFlat)
            + "assert index.d == d, index.d\n"
            + "assert index.ntotal == n, index.ntotal\n"
            + "xq = np.fromfile('%s', dtype='<f4').reshape(-1, d)\n".formatted(xqBin)
            + "D, I = index.search(xq[:1], k)\n"
            + "print('py read ok', index.ntotal, int(I[0,0]), float(D[0,0]))\n"
            + "assert I[0,0] >= 0\n";
        Path pyReadScript = tmp.resolve("read_flat.py");
        Files.writeString(pyReadScript, pyRead);
        runPy(pyReadScript);
        check("python read java flat", true);

        // 3) HNSW Java → Python
        Path jHnsw = tmp.resolve("java_hnsw.faiss");
        IndexHNSWFlat h = new IndexHNSWFlat(d, 16);
        h.hnsw.efConstruction = 40;
        h.hnsw.efSearch = 16;
        h.add(xb, n);
        Faiss.write_index(h, jHnsw);
        String pyHnsw = ""
            + "import faiss, numpy as np\n"
            + "d, n, k = %d, %d, %d\n".formatted(d, n, k)
            + "index = faiss.read_index('%s')\n".formatted(jHnsw)
            + "print(type(index), index.ntotal, index.d)\n"
            + "assert index.ntotal == n\n"
            + "xq = np.fromfile('%s', dtype='<f4').reshape(-1, d)\n".formatted(xqBin)
            + "D, I = index.search(xq[:1], k)\n"
            + "print('hnsw py', int(I[0,0]), float(D[0,0]))\n";
        Path pyHnswScript = tmp.resolve("read_hnsw.py");
        Files.writeString(pyHnswScript, pyHnsw);
        runPy(pyHnswScript);
        check("python read java hnsw", true);

        // 4) Python HNSW → Java
        Path pyHnswPath = tmp.resolve("py_hnsw.faiss");
        String pyWriteH = ""
            + "import faiss, numpy as np\n"
            + "d, n = %d, %d\n".formatted(d, n)
            + "xb = np.fromfile('%s', dtype='<f4').reshape(n, d)\n".formatted(xbBin)
            + "index = faiss.IndexHNSWFlat(d, 16)\n"
            + "index.hnsw.efConstruction = 40\n"
            + "index.add(xb)\n"
            + "faiss.write_index(index, '%s')\n".formatted(pyHnswPath)
            + "print('py hnsw wrote', index.ntotal)\n";
        Path pyWH = tmp.resolve("write_hnsw.py");
        Files.writeString(pyWH, pyWriteH);
        runPy(pyWH);
        Index jFromPyH = Faiss.read_index(pyHnswPath);
        check("java read py hnsw", jFromPyH instanceof IndexHNSWFlat && jFromPyH.ntotal() == n);
        SearchResult hr = jFromPyH.search(xq, 1, k);
        check("java search py-hnsw", hr.I[0][0] >= 0);

        // 5) IDMap
        Path jId = tmp.resolve("java_idmap.faiss");
        IndexIDMap idm = new IndexIDMap(new IndexFlatIP(d));
        long[] ids = new long[n];
        for (int i = 0; i < n; i++) ids[i] = 5000 + i;
        idm.add_with_ids(xb, n, ids);
        Faiss.write_index(idm, jId);
        String pyId = ""
            + "import faiss, numpy as np\n"
            + "index = faiss.read_index('%s')\n".formatted(jId)
            + "print(type(index), index.ntotal)\n"
            + "xq = np.fromfile('%s', dtype='<f4').reshape(-1, %d)\n".formatted(xqBin, d)
            + "D, I = index.search(xq[:1], 3)\n"
            + "print('ids', I[0].tolist())\n"
            + "assert I[0,0] >= 5000\n";
        Path pyIdS = tmp.resolve("read_idmap.py");
        Files.writeString(pyIdS, pyId);
        runPy(pyIdS);
        check("python read java idmap", true);

        System.out.println("  OK  python↔java cross-lang IO");
    }

    static void writeF32(Path p, float[] v) throws Exception {
        java.nio.ByteBuffer bb = java.nio.ByteBuffer.allocate(v.length * 4)
            .order(java.nio.ByteOrder.LITTLE_ENDIAN);
        for (float x : v) bb.putFloat(x);
        Files.write(p, bb.array());
    }

    static void runPy(Path script) throws Exception {
        ProcessBuilder pb = new ProcessBuilder("python3", script.toString());
        pb.redirectErrorStream(true);
        Process p = pb.start();
        String out = new String(p.getInputStream().readAllBytes());
        int code = p.waitFor();
        System.out.print(out);
        if (code != 0) throw new AssertionError("python failed code=" + code + "\n" + out);
    }
}
