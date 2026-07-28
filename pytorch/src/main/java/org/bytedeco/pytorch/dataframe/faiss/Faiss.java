package org.bytedeco.pytorch.dataframe.faiss;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Top-level FAISS-like façade — mirrors the {@code faiss} Python module surface
 * used in {@code org/lance/ipc/faiss.md}.
 *
 * <pre>
 *   Faiss.normalize_L2(vecs);
 *   IndexHNSWFlat index = new IndexHNSWFlat(d, 32);
 *   IndexIDMap idIndex = new IndexIDMap(index);
 *   idIndex.add_with_ids(vecs, ids);
 *   Faiss.write_index(idIndex, "idx.jfaiss");
 *   Index loaded = Faiss.read_index("idx.jfaiss");
 * </pre>
 *
 * <p>Pure Java implementation. Large Flat scans use javacpp-pytorch Tensor
 * (CUDA when {@link DeviceSelector} resolves to CUDA). Not a JNI binding to
 * libfaiss — file format is custom (see {@link IndexIO}).
 */
public final class Faiss {
    /** Convenience alias matching {@code faiss.METRIC_L2}. */
    public static final MetricType METRIC_L2 = MetricType.METRIC_L2;
    /** Convenience alias matching {@code faiss.METRIC_INNER_PRODUCT}. */
    public static final MetricType METRIC_INNER_PRODUCT = MetricType.METRIC_INNER_PRODUCT;

    private Faiss() {}

    // ---- normalize ----

    /** In-place L2-normalize rows of a row-major {@code [n * d]} matrix. */
    public static void normalize_L2(float[] x, int n, int d) {
        DistanceKernel.normalizeL2(x, n, d);
    }

    /** In-place L2-normalize 2-D rows. */
    public static void normalize_L2(float[][] rows) {
        if (rows == null) return;
        for (float[] r : rows) {
            if (r == null || r.length == 0) continue;
            float sum = 0f;
            for (float v : r) sum += v * v;
            if (sum > 0f) {
                float inv = (float) (1.0 / Math.sqrt(sum));
                for (int i = 0; i < r.length; i++) r[i] *= inv;
            }
        }
    }

    /** Infer d from first row length when packing is not needed. */
    public static void normalize_L2(float[] x) {
        // treat as single vector
        if (x == null || x.length == 0) return;
        normalize_L2(x, 1, x.length);
    }

    // ---- IO (native FAISS binary by default — interoperable with Python faiss) ----

    /**
     * Write index in <b>native FAISS binary</b> format (same as
     * {@code faiss.write_index} in Python/C++).
     * Supported: IndexFlatL2/IP, IndexHNSWFlat, IndexIVFPQ, IndexIDMap.
     * For unsupported types, falls back to custom JDF1 Java serialization.
     */
    public static void write_index(Index index, String path) throws IOException {
        write_index(index, Path.of(path));
    }

    public static void write_index(Index index, Path path) throws IOException {
        try {
            NativeFaissIO.write(index, path);
        } catch (IOException e) {
            // Unsupported type → JDF1 fallback (not Python-compatible)
            if (e.getMessage() != null && e.getMessage().contains("unsupported")) {
                IndexIO.write(index, path);
                return;
            }
            throw e;
        }
    }

    /**
     * Read index. Auto-detects native FAISS fourcc vs legacy JDF1.
     * Native files can be produced/consumed by Python {@code faiss.read_index}.
     */
    public static Index read_index(String path) throws IOException {
        return read_index(Path.of(path));
    }

    public static Index read_index(Path path) throws IOException {
        return NativeFaissIO.read(path);
    }

    /** Force custom JDF1 Java serialization (NOT compatible with Python faiss). */
    public static void write_index_jdf1(Index index, Path path) throws IOException {
        IndexIO.write(index, path);
    }

    public static Index read_index_jdf1(Path path) throws IOException, ClassNotFoundException {
        return IndexIO.read(path);
    }

    /** True if file is native FAISS (not JDF1). */
    public static boolean is_native_faiss_file(Path path) throws IOException {
        return NativeFaissIO.isNativeFaissFile(path);
    }

    // ---- factories (Python-style) ----

    public static IndexFlatL2 index_flat_l2(int d) { return new IndexFlatL2(d); }
    public static IndexFlatIP index_flat_ip(int d) { return new IndexFlatIP(d); }
    public static IndexHNSWFlat index_hnsw_flat(int d, int M) { return new IndexHNSWFlat(d, M); }
    public static IndexIDMap index_id_map(Index inner) { return new IndexIDMap(inner); }

    public static IndexIVFPQ index_ivfpq(Index quantizer, int d, int nlist, int m, int nbits) {
        return new IndexIVFPQ(quantizer, d, nlist, m, nbits);
    }

    public static IndexShards index_shards(int d) { return new IndexShards(d); }

    // ---- GPU semantics (torch device migration, not FAISS-GPU kernels) ----

    /**
     * Move index storage to CUDA if available — semantic equivalent of
     * {@code faiss.index_cpu_to_gpu}. Returns the same instance marked on-GPU.
     * HNSW graph search still runs on CPU; Flat/IVF scans may use CUDA GEMM.
     */
    public static Index index_cpu_to_gpu(StandardGpuResources res, Index cpuIndex) {
        int dev = res == null ? DeviceSelector.cudaDeviceIndex() : res.device();
        if (!DeviceSelector.isCudaAvailable()) {
            return cpuIndex; // silent no-op with CPU fallback
        }
        DeviceSelector.setPreferred(DeviceSelector.Device.CUDA);
        DeviceSelector.setCudaDeviceIndex(dev);
        cpuIndex.to_gpu_storage(dev);
        return cpuIndex;
    }

    public static Index index_cpu_to_gpu(StandardGpuResources res, int device, Index cpuIndex) {
        if (res != null) res.setDevice(device);
        DeviceSelector.setCudaDeviceIndex(device);
        return index_cpu_to_gpu(res, cpuIndex);
    }

    /**
     * Bring index back to CPU storage — required before {@link #write_index}.
     * Mirrors {@code faiss.index_gpu_to_cpu}.
     */
    public static Index index_gpu_to_cpu(Index gpuIndex) {
        if (gpuIndex == null) return null;
        gpuIndex.to_cpu_storage();
        return gpuIndex;
    }

    // ---- device helpers ----

    public static String device_describe() {
        return DeviceSelector.describe();
    }

    public static boolean cuda_available() {
        return DeviceSelector.isCudaAvailable();
    }

    public static void set_device(DeviceSelector.Device device) {
        DeviceSelector.setPreferred(device);
    }

    // ---- ID selector factory ----

    public static IDSelectorArray IDSelectorArray(long[] ids) {
        return new IDSelectorArray(ids);
    }

    public static IDSelectorArray IDSelectorArray(int[] ids) {
        return new IDSelectorArray(ids);
    }
}
