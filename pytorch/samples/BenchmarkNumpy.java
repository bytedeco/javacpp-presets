package samples;

import org.bytedeco.pytorch.data.numpy.DType;
import org.bytedeco.pytorch.data.numpy.NDArray;
import org.bytedeco.pytorch.data.numpy.NP;
import org.bytedeco.pytorch.data.numpy.NpyHeader;
import org.bytedeco.pytorch.global.torch;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * Comprehensive benchmark for NumPy module.
 * Tests all NP factory methods, file I/O, tensor conversion, elementwise ops,
 * reduction ops, shape ops, linear algebra, and activation functions.
 * Also validates Python interop: Java writes .npy/.npz that Python reads, and vice versa.
 */
public class BenchmarkNumpy {

    static int passed = 0;
    static int failed = 0;
    static StringBuilder report = new StringBuilder();

    public static void main(String[] args) throws Exception {
        System.out.println("=== NumPy Module Benchmark ===\n");
        Path tmpDir = Files.createTempDirectory("numpy_bench");
        System.out.println("Temp dir: " + tmpDir);

        try {
            // ── 1. DType enum ──────────────────────────────────────────────
            benchmark("DType enum - all values", () -> {
                for (DType dt : DType.values()) {
                    check("DType." + dt.name() + " descriptor", dt.getDescriptor() != null && !dt.getDescriptor().isEmpty());
                    check("DType." + dt.name() + " byteSize", dt.getByteSize() > 0);
                    check("DType." + dt.name() + " toTorch", dt.toTorch() != null);
                }
                check("fromDescriptor FLOAT64", DType.fromDescriptor("<f8") == DType.FLOAT64);
                check("fromDescriptor FLOAT16", DType.fromDescriptor("<f2") == DType.FLOAT16);
                check("fromDescriptor INT32", DType.fromDescriptor("<i4") == DType.INT32);
                check("fromDescriptor UINT8", DType.fromDescriptor("|u1") == DType.UINT8);
                check("fromTorch Double", DType.fromTorch(torch.ScalarType.Double) == DType.FLOAT64);
            });

            // ── 2. NDArray zeros/ones ───────────────────────────────────────
            benchmark("NDArray zeros/ones with dtype", () -> {
                for (DType dt : DType.values()) {
                    NDArray z = NP.zeros(dt, 3L, 4L);
                    check("zeros " + dt.name() + " shape", z.shape.length == 2 && z.shape[0] == 3 && z.shape[1] == 4);
                    check("zeros " + dt.name() + " numel", z.numel() == 12);
                    NDArray o = NP.ones(dt, 2L, 5L);
                    check("ones " + dt.name() + " numel", o.numel() == 10);
                }
            });

            benchmark("NDArray zeros/ones default dtype (FLOAT64)", () -> {
                NDArray zd = NP.zeros(4L, 3L);
                check("zeros default dtype FLOAT64", zd.dtype == DType.FLOAT64);
                check("zeros default shape", zd.shape.length == 2 && zd.shape[0] == 4 && zd.shape[1] == 3);
                NDArray od = NP.ones(2L, 2L);
                check("ones default dtype FLOAT64", od.dtype == DType.FLOAT64);
            });

            // ── 3. NDArray full/eye/linspace ────────────────────────────────
            benchmark("NDArray full", () -> {
                NDArray f = NP.full(DType.FLOAT32, 99.0, 3L, 3L);
                check("full FLOAT32 value", f.getDouble(0) == 99.0);
                check("full FLOAT32 numel", f.numel() == 9);
                NDArray f2 = NP.full(5.5, 4L);
                check("full default dtype", f2.dtype == DType.FLOAT64 && f2.getDouble(0) == 5.5);
            });

            benchmark("NDArray eye", () -> {
                NDArray e1 = NP.eye(3);
                check("eye(3) shape[0]=3", e1.shape[0] == 3);
                check("eye(3) shape[1]=3", e1.shape[1] == 3);
                check("eye(3) diag[0]=1", e1.getDouble(0) == 1.0);
                check("eye(3) off-diag[1]=0", e1.getDouble(1) == 0.0);
                check("eye(3) off-diag[3]=0", e1.getDouble(3) == 0.0);

                NDArray e2 = NP.eye(3, 4);
                check("eye(3,4) shape", e2.shape[0] == 3 && e2.shape[1] == 4);
                check("eye(3,4) diag[0]=1", e2.getDouble(0) == 1.0);

                NDArray e3 = NP.eye(4, 3, 1); // k=1 super-diagonal
                check("eye(4,3,k=1) diag offset", e3.getDouble(1) == 1.0);
            });

            benchmark("NDArray linspace", () -> {
                NDArray ls = NP.linspace(0.0, 10.0, 5);
                check("linspace numel=5", ls.numel() == 5);
                check("linspace start=0", ls.getDouble(0) == 0.0);
                check("linspace end=10", ls.getDouble(4) == 10.0);
                check("linspace step=2.5", Math.abs(ls.getDouble(1) - 2.5) < 1e-9);
            });

            // ── 4. NDArray arange ───────────────────────────────────────────
            benchmark("NDArray arange", () -> {
                NDArray a1 = NP.arange(10.0); // stop only
                check("arange(stop) numel=10", a1.numel() == 10);
                check("arange(stop) start=0", a1.getDouble(0) == 0.0);
                check("arange(stop) end=9", a1.getDouble(9) == 9.0);

                NDArray a2 = NP.arange(2.0, 8.0); // start, stop
                check("arange(start,stop) numel=6", a2.numel() == 6);
                check("arange(start,stop) start=2", a2.getDouble(0) == 2.0);
                check("arange(start,stop) end=7", a2.getDouble(5) == 7.0);

                NDArray a3 = NP.arange(0.0, 10.0, 2.0); // start, stop, step
                check("arange(start,stop,step) numel=5", a3.numel() == 5);
                check("arange(start,stop,step) step", a3.getDouble(2) == 4.0);

                NDArray a4 = NP.arange(-3.0, 3.0, 0.5); // fractional step
                check("arange negative start", a4.getDouble(0) == -3.0);
                check("arange negative end approx", Math.abs(a4.getDouble(11) - 2.5) < 1e-9);

                NDArray a5 = NP.arange(0.0, 10.0, 2.0, DType.FLOAT32); // with dtype
                check("arange with dtype FLOAT32", a5.dtype == DType.FLOAT32);
            });

            // ── 5. NDArray rand/randn/array ────────────────────────────────
            benchmark("NDArray rand", () -> {
                NDArray r = NP.rand(100L);
                check("rand numel=100", r.numel() == 100);
                check("rand range [0,1)", r.getDouble(0) >= 0.0 && r.getDouble(0) < 1.0);
                check("rand dtype FLOAT64", r.dtype == DType.FLOAT64);
            });

            benchmark("NDArray randn", () -> {
                NDArray rn = NP.randn(1000L);
                check("randn numel=1000", rn.numel() == 1000);
                double sum = 0;
                for (int i = 0; i < (int)rn.numel(); i++) sum += rn.getDouble(i);
                double mean = sum / rn.numel();
                check("randn mean approx 0", Math.abs(mean) < 0.3);
            });

            benchmark("NDArray array factories", () -> {
                double[] dblData = {1, 2, 3, 4, 5, 6};
                NDArray fromDbl = NP.array(dblData, 2L, 3L);
                check("array(double[],2,3) shape[0]=2", fromDbl.shape[0] == 2);
                check("array(double[],2,3) shape[1]=3", fromDbl.shape[1] == 3);
                check("array(double[]) contents[0]=1", fromDbl.getDouble(0) == 1.0);
                check("array(double[]) contents[5]=6", fromDbl.getDouble(5) == 6.0);

                float[] fltData = {(float)1.5, (float)2.5, (float)3.0};
                NDArray fromFlt = NP.array(fltData, 3L);
                check("array(float[]) dtype FLOAT32", fromFlt.dtype == DType.FLOAT32);
                check("array(float[]) contents[1]=2.5", fromFlt.getDouble(1) == 2.5);

                long[] lngData = {10, 20, 30, 40};
                NDArray fromLng = NP.array(lngData, 4L);
                check("array(long[]) dtype INT64", fromLng.dtype == DType.INT64);
                check("array(long[]) contents[2]=30", fromLng.getLong(2) == 30L);

                int[] intData = {100, 200, 300};
                NDArray fromInt = NP.array(intData, 3L);
                check("array(int[]) not null", fromInt != null);
                check("array(int[]) dtype INT32", fromInt.dtype == DType.INT32);
            });

            // ── 6. NDArray element get/set ─────────────────────────────────
            benchmark("NDArray element get/set", () -> {
                NDArray a = NP.reshape(NP.arange(1.0, 7.0, 1.0, DType.FLOAT64), 2L, 3L);
                check("getDouble flat[3]=4", a.getDouble(3) == 4.0);
                a.setDouble(2, 99.0);
                check("setDouble[2]=99", a.getDouble(2) == 99.0);

                NDArray i = NP.array(new long[]{10, 20, 30, 40}, 4L);
                check("getLong[2]=30", i.getLong(2) == 30L);
                i.setLong(3, 999L);
                check("setLong[3]=999", i.getLong(3) == 999L);

                NDArray c = NP.copy(a);
                check("copy contents same", c.getDouble(2) == 99.0);
                check("copy is different object", c != a);
            });

            // ── 7. NumPy save/load .npy ─────────────────────────────────────
            benchmark("NumPy save/load .npy - FLOAT64", () -> {
                Path npyFile = tmpDir.resolve("float64.npy");
                NDArray original = NP.reshape(NP.arange(0.0, 12.0, 1.0, DType.FLOAT64), 3L, 4L);
                NP.save(original, npyFile.toString());
                check("save .npy file exists", Files.exists(npyFile));
                check("save .npy file not empty", Files.size(npyFile) > 0);

                NDArray loaded = NP.load(npyFile.toString());
                check("load shape[0]=3", loaded.shape[0] == 3);
                check("load shape[1]=4", loaded.shape[1] == 4);
                check("load numel=12", loaded.numel() == 12);
                check("load dtype FLOAT64", loaded.dtype == DType.FLOAT64);
                check("load value[5]=5", loaded.getDouble(5) == 5.0);
            });

            benchmark("NumPy save/load .npy - all float dtypes", () -> {
                DType[] floatDts = {DType.FLOAT64, DType.FLOAT32, DType.FLOAT16};
                for (DType dt : floatDts) {
                    NDArray original = NP.full(dt, 1.5, 5L, 4L);
                    Path f = tmpDir.resolve(dt.name() + ".npy");
                    NP.save(original, f.toString());
                    NDArray loaded = NP.load(f.toString());
                    check("save/load " + dt.name() + " shape", loaded.shape[0] == 5 && loaded.shape[1] == 4);
                    check("save/load " + dt.name() + " dtype", loaded.dtype == original.dtype);
                }
            });

            benchmark("NumPy save/load .npy - integer dtypes", () -> {
                DType[] intDts = {DType.INT64, DType.INT32, DType.INT16, DType.INT8, DType.UINT8, DType.BOOL};
                for (DType dt : intDts) {
                    NDArray original = NP.full(dt, 3, 4L, 5L);
                    Path f = tmpDir.resolve(dt.name() + ".npy");
                    NP.save(original, f.toString());
                    NDArray loaded = NP.load(f.toString());
                    check("save/load " + dt.name() + " shape", loaded.shape[0] == 4 && loaded.shape[1] == 5);
                    check("save/load " + dt.name() + " dtype", loaded.dtype == original.dtype);
                }
            });

            // ── 8. NumPy savez/loadz .npz ───────────────────────────────────
            benchmark("NumPy savez/loadz .npz", () -> {
                Path npzFile = tmpDir.resolve("test.npz");

                Map<String, NDArray> arrays = new LinkedHashMap<>();
                arrays.put("weights", NP.randn(10L, 5L));
                arrays.put("bias", NP.zeros(5L));
                arrays.put("scale", NP.full(2.5, 3L));

                NP.savez(npzFile.toString(), arrays);
                check("savez file exists", Files.exists(npzFile));

                Map<String, NDArray> loaded = NP.loadz(npzFile.toString());
                check("loadz keys size=3", loaded.size() == 3);
                check("loadz weights numel=50", loaded.get("weights").numel() == 50);
                check("loadz bias numel=5", loaded.get("bias").numel() == 5);
                check("loadz scale value≈2.5", Math.abs(loaded.get("scale").getDouble(0) - 2.5) < 1e-9);
            });

            // ── 9. Tensor conversion ────────────────────────────────────────
            benchmark("Tensor toTensor/fromTensor", () -> {
                for (DType dt : DType.values()) {
                    NDArray arr = NP.full(dt, 2, 3L, 4L);
                    org.bytedeco.pytorch.Tensor t = NP.toTensor(arr);
                    check("toTensor " + dt.name() + " not null", t != null);
                    check("toTensor " + dt.name() + " dim=3", t.dim() == 3);
                    check("toTensor " + dt.name() + " size(0)=3", t.size(0) == 3);

                    NDArray back = NP.fromTensor(t);
                    check("fromTensor " + dt.name() + " not null", back != null);
                    check("fromTensor " + dt.name() + " dtype", back.dtype == arr.dtype);
                }

                org.bytedeco.pytorch.Tensor t4d = torch.randn(new long[]{2, 3, 4, 5});
                NDArray arr4d = NP.fromTensor(t4d);
                check("fromTensor 4D shape len=4", arr4d.shape.length == 4);
                check("fromTensor 4D numel=120", arr4d.numel() == 120);
            });

            // ── 10. Elementwise binary operations ───────────────────────────
            benchmark("Elementwise binary ops", () -> {
                NDArray a = NP.array(new double[]{1, 2, 3, 4}, 4L);
                NDArray b = NP.array(new double[]{10, 20, 30, 40}, 4L);

                check("add(a,b)[0]=11", NP.add(a, b).getDouble(0) == 11);
                check("add(a,b)[3]=44", NP.add(a, b).getDouble(3) == 44);
                check("add(a,10)[1]=12", NP.add(a, 10.0).getDouble(1) == 12);
                check("sub(a,b)[0]=-9", NP.sub(a, b).getDouble(0) == -9);
                check("mul(a,b)[2]=90", NP.mul(a, b).getDouble(2) == 90);
                check("div(a,b)[0]≈0.1", Math.abs(NP.div(a, b).getDouble(0) - 0.1) < 1e-9);
                check("power(a,2)[2]=9", NP.power(a, 2.0).getDouble(2) == 9.0);

                NDArray max_ab = NP.maximum(a, b);
                check("maximum(a,b)[0]=10", max_ab.getDouble(0) == 10);
                check("maximum(a,b)[3]=40", max_ab.getDouble(3) == 40);

                NDArray max_as = NP.maximum(a, 2.5);
                check("maximum(a,2.5)[0]=2.5", max_as.getDouble(0) == 2.5);
                check("maximum(a,2.5)[3]=4", max_as.getDouble(3) == 4.0);

                NDArray min_ab = NP.minimum(a, b);
                check("minimum(a,b)[0]=1", min_ab.getDouble(0) == 1);
                check("minimum(a,b)[3]=4", min_ab.getDouble(3) == 4);

                NDArray cond = NP.array(new double[]{1, 0, 1, 0}, 4L);
                NDArray x = NP.array(new double[]{10, 20, 30, 40}, 4L);
                NDArray y = NP.array(new double[]{100, 200, 300, 400}, 4L);
                NDArray where = NP.where(cond, x, y);
                check("where[0]=10", where.getDouble(0) == 10);
                check("where[1]=200", where.getDouble(1) == 200);
                check("where[2]=30", where.getDouble(2) == 30);
                check("where[3]=400", where.getDouble(3) == 400);
            });

            // ── 11. Elementwise unary operations ────────────────────────────
            benchmark("Elementwise unary ops", () -> {
                NDArray pos = NP.array(new double[]{1, 4, 9, 16, 25}, 5L);
                NDArray neg = NP.array(new double[]{-4, -2, 0, 2, 4}, 5L);

                check("abs[-4]=4", NP.abs(neg).getDouble(0) == 4.0);
                check("abs[0]=0", NP.abs(neg).getDouble(2) == 0.0);
                check("neg[0]=4", NP.neg(neg).getDouble(0) == 4.0);
                check("sqrt[4]=2", Math.abs(NP.sqrt(pos).getDouble(1) - 2.0) < 1e-9);
                check("exp[0]=1", Math.abs(NP.exp(NP.zeros(1L)).getDouble(0) - 1.0) < 1e-9);

                NDArray oneToFour = NP.arange(1.0, 5.0);
                check("log exp cancel", Math.abs(NP.log(NP.exp(oneToFour)).getDouble(2) - 3.0) < 1e-5);
                check("log2[8]=3", Math.abs(NP.log2(NP.array(new double[]{8}, 1L)).getDouble(0) - 3.0) < 1e-9);
                check("log10[100]=2", Math.abs(NP.log10(NP.array(new double[]{100}, 1L)).getDouble(0) - 2.0) < 1e-9);

                check("sin[0]=0", Math.abs(NP.sin(NP.zeros(1L)).getDouble(0)) < 1e-9);
                check("cos[0]=1", Math.abs(NP.cos(NP.zeros(1L)).getDouble(0) - 1.0) < 1e-9);

                check("floor[1.7]=1", NP.floor(NP.array(new double[]{1.7}, 1L)).getDouble(0) == 1.0);
                check("ceil[1.3]=2", NP.ceil(NP.array(new double[]{1.3}, 1L)).getDouble(0) == 2.0);
                check("sign[-5]=-1", NP.sign(NP.array(new double[]{-5}, 1L)).getDouble(0) == -1.0);
                check("reciprocal[4]=0.25", Math.abs(NP.reciprocal(NP.array(new double[]{4}, 1L)).getDouble(0) - 0.25) < 1e-9);
                check("square[3]=9", NP.square(NP.array(new double[]{3}, 1L)).getDouble(0) == 9.0);

                NDArray negArr = NP.array(new double[]{-1, 0, 2}, 3L);
                check("relu[-1]=0", NP.relu(negArr).getDouble(0) == 0.0);
                check("relu[2]=2", NP.relu(negArr).getDouble(2) == 2.0);
                check("leaky_relu[-1]≈-0.01", Math.abs(NP.leaky_relu(NP.array(new double[]{-1.0}, 1L), 0.01).getDouble(0) - (-0.01)) < 1e-9);
                check("sigmoid[0]=0.5", Math.abs(NP.sigmoid(NP.zeros(1L)).getDouble(0) - 0.5) < 1e-9);
                check("clip[10]->6", NP.clip(NP.array(new double[]{10}, 1L), -2, 6).getDouble(0) == 6.0);
            });

            // ── 12. Reduction operations ───────────────────────────────────
            benchmark("Reduction operations", () -> {
                NDArray a = NP.arange(1.0, 11.0); // 1..10
                check("sum 1..10=55", Math.abs(NP.sum(a) - 55.0) < 1e-9);
                check("mean 1..10=5.5", Math.abs(NP.mean(a) - 5.5) < 1e-9);
                check("max=10", Math.abs(NP.max(a) - 10.0) < 1e-9);
                check("min=1", Math.abs(NP.min(a) - 1.0) < 1e-9);
                check("prod 1..10=3628800", Math.abs(NP.prod(a) - 3628800.0) < 1e-9);
                check("std≈2.872", Math.abs(NP.std(a) - Math.sqrt(33.0/4.0)) < 1e-6);
                check("var=8.25", Math.abs(NP.var(a) - 8.25) < 1e-9);
                check("argmax=9", NP.argmax(a) == 9);
                check("argmin=0", NP.argmin(a) == 0);

                NDArray nanArr = NP.array(new double[]{1, Double.NaN, 3, Double.NaN, 5}, 5L);
                check("nansum=9", Math.abs(NP.nansum(nanArr) - 9.0) < 1e-9);
                check("nanmean=3", Math.abs(NP.nanmean(nanArr) - 3.0) < 1e-9);
            });

            // ── 13. Shape operations ────────────────────────────────────────
            benchmark("Shape operations", () -> {
                NDArray a = NP.reshape(NP.arange(1.0, 13.0), 3L, 4L);
                check("reshape 3x4", a.shape[0] == 3 && a.shape[1] == 4);

                NDArray flat = NP.flatten(a);
                check("flatten numel=12", flat.numel() == 12);
                check("flatten shape len=1", flat.shape.length == 1);

                NDArray t = NP.transpose(a);
                check("transpose shape 4x3", t.shape[0] == 4 && t.shape[1] == 3);

                NDArray c1 = NP.array(new double[]{1, 2}, 2L);
                NDArray c2 = NP.array(new double[]{3, 4}, 2L);
                NDArray concat = NP.concatenate(c1, c2);
                check("concatenate numel=4", concat.numel() == 4);
                check("concatenate [2]=3", concat.getDouble(2) == 3.0);

                NDArray s1 = NP.arange(1.0, 4.0);
                NDArray s2 = NP.arange(4.0, 7.0);
                NDArray stack = NP.stack(s1, s2);
                check("stack shape 2x3", stack.shape[0] == 2 && stack.shape[1] == 3);
            });

            // ── 14. Linear algebra ─────────────────────────────────────────
            benchmark("Linear algebra", () -> {
                NDArray a = NP.array(new double[]{1,2,3, 4,5,6}, 2L, 3L);
                NDArray b = NP.array(new double[]{1,2,3, 4,5,6}, 3L, 2L);
                NDArray m = NP.matmul(a, b);
                check("matmul shape 2x2", m.shape[0] == 2 && m.shape[1] == 2);
                check("matmul[0,0]=24", Math.abs(m.getDouble(0) - 24.0) < 1e-9);
                check("matmul[0,1]=30", Math.abs(m.getDouble(1) - 30.0) < 1e-9);
                check("matmul[1,0]=54", Math.abs(m.getDouble(2) - 54.0) < 1e-9);
                check("matmul[1,1]=69", Math.abs(m.getDouble(3) - 69.0) < 1e-9);

                NDArray v1 = NP.array(new double[]{1, 2, 3}, 3L);
                NDArray v2 = NP.array(new double[]{4, 5, 6}, 3L);
                NDArray d = NP.dot(v1, v2);
                check("dot 1D=32", Math.abs(d.getDouble(0) - 32.0) < 1e-9);
            });

            // ── 15. Softmax activation ─────────────────────────────────────
            benchmark("softmax", () -> {
                NDArray a = NP.arange(0.0, 5.0); // [0,1,2,3,4]
                NDArray sm = NP.softmax(a);
                double exp1 = Math.exp(1), exp2 = Math.exp(2), exp3 = Math.exp(3), exp4 = Math.exp(4);
                double total = 1 + exp1 + exp2 + exp3 + exp4;
                check("softmax[0]=1/total", Math.abs(sm.getDouble(0) - 1/total) < 1e-9);
                check("softmax sum≈1", Math.abs(NP.sum(sm) - 1.0) < 1e-9);

                NDArray a2 = NP.reshape(NP.arange(0.0, 6.0), 2L, 3L);
                NDArray sm2 = NP.softmax(a2, 1);
                check("softmax 2D axis=1 shape", sm2.shape[0] == 2 && sm2.shape[1] == 3);
            });

            // ── 16. astype conversions ─────────────────────────────────────
            benchmark("astype conversions", () -> {
                NDArray f64 = NP.arange(0.0, 5.0);
                for (DType dt : DType.values()) {
                    if (dt == DType.FLOAT64) continue;
                    NDArray converted = NP.astype(f64, dt);
                    check("astype to " + dt.name(), converted.dtype == dt);
                }
            });

            // ── 17. NpyHeader ──────────────────────────────────────────────
            benchmark("NpyHeader", () -> {
                NpyHeader h = new NpyHeader(DType.FLOAT32, false, new long[]{10, 20});
                check("NpyHeader dtype FLOAT32", h.dtype == DType.FLOAT32);
                check("NpyHeader fortranOrder=false", !h.fortranOrder);
                check("NpyHeader shape[0]=10", h.shape[0] == 10);
                check("NpyHeader shape[1]=20", h.shape[1] == 20);
                check("NpyHeader numel=200", h.numel() == 200);
                String headerStr = h.toHeaderString();
                check("toHeaderString not empty", headerStr != null && !headerStr.isEmpty());
                NpyHeader parsed = NpyHeader.parse(headerStr);
                check("parse roundtrip dtype", parsed.dtype == DType.FLOAT32);
                check("parse roundtrip numel", parsed.numel() == 200);

                NpyHeader h2 = new NpyHeader(DType.INT64, true, new int[]{3, 4, 5});
                check("int[] shape constructor", h2.shape[2] == 5);
            });

            // ── 18. Multi-dimensional arrays ────────────────────────────────
            benchmark("Multi-dimensional arrays", () -> {
                NDArray a3d = NP.reshape(NP.arange(0.0, 24.0), 2L, 3L, 4L);
                check("3D reshape 2x3x4", a3d.shape[0] == 2 && a3d.shape[1] == 3 && a3d.shape[2] == 4 && a3d.numel() == 24);

                NDArray f3d = NP.flatten(a3d);
                check("3D flatten numel=24", f3d.numel() == 24);

                org.bytedeco.pytorch.Tensor t3d = NP.toTensor(a3d);
                check("3D toTensor dim=3", t3d.dim() == 3);
                check("3D toTensor size(0)=2", t3d.size(0) == 2);

                NDArray a5d = NP.zeros(DType.FLOAT32, 2L, 3L, 4L, 5L, 6L);
                check("5D zeros numel=720", a5d.numel() == 720);
            });

            // ── 19. Python interop: Java writes → Python reads ──────────────
            benchmark("Python interop: Java .npy → Python reads", () -> {
                Path npyFile = tmpDir.resolve("java_numpy.npy");
                NDArray arr = NP.reshape(NP.arange(0.0, 12.0), 3L, 4L);
                NP.save(arr, npyFile.toString());

                String pyCheck = String.format(
                    "import numpy as np; " +
                    "arr = np.load('%s'); " +
                    "assert arr.shape == (3, 4), f'shape {arr.shape}'; " +
                    "assert arr[0,0] == 0.0 and arr[2,3] == 11.0, f'values {arr[0,0]}, {arr[2,3]}'; " +
                    "assert arr.dtype == np.float64, f'dtype {arr.dtype}'; " +
                    "print('PASS')",
                    npyFile.toAbsolutePath());

                String result = runPython(pyCheck);
                check("Java .npy Python reads", result.contains("PASS"));
            });

            benchmark("Python interop: Java .npz → Python reads", () -> {
                Path npzFile = tmpDir.resolve("java_numpy.npz");
                Map<String, NDArray> arrays = new LinkedHashMap<>();
                arrays.put("a", NP.reshape(NP.arange(0.0, 6.0), 2L, 3L));
                arrays.put("b", NP.full(3.14, 4L));
                NP.savez(npzFile.toString(), arrays);

                String pyCheck = String.format(
                    "import numpy as np; " +
                    "d = np.load('%s'); " +
                    "assert 'a' in d and 'b' in d, 'keys'; " +
                    "assert d['a'].shape == (2, 3), f'a shape {d[\"a\"].shape}'; " +
                    "assert d['b'].shape == (4,) and abs(d['b'][0]-3.14)<1e-9, f'b value {d[\"b\"][0]}'; " +
                    "print('PASS')",
                    npzFile.toAbsolutePath());

                String result = runPython(pyCheck);
                check("Java .npz Python reads", result.contains("PASS"));
            });

            // ── 20. Python interop: Python writes → Java reads ──────────────
            benchmark("Python interop: Python .npy → Java reads", () -> {
                Path npFile = tmpDir.resolve("python_numpy.npy");
                String pyWrite = String.format(
                    "import numpy as np; " +
                    "arr = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32); " +
                    "np.save('%s', arr)",
                    npFile.toAbsolutePath());
                runPython(pyWrite);

                NDArray loaded = NP.load(npFile.toString());
                check("Python .npy dtype FLOAT32", loaded.dtype == DType.FLOAT32);
                check("Python .npy shape 2x2", loaded.shape[0] == 2 && loaded.shape[1] == 2);
                check("Python .npy [0,0]=1.5", Math.abs(loaded.getDouble(0) - 1.5) < 1e-6);
                check("Python .npy [1,1]=4.5", Math.abs(loaded.getDouble(3) - 4.5) < 1e-6);
            });

            benchmark("Python interop: Python .npz → Java reads", () -> {
                Path npzFile = tmpDir.resolve("python_numpy.npz");
                String pyWrite = String.format(
                    "import numpy as np; " +
                    "arr1 = np.arange(6).reshape(2,3).astype(np.float64); " +
                    "arr2 = np.array([1.1, 2.2, 3.3]); " +
                    "np.savez('%s', x=arr1, y=arr2)",
                    npzFile.toAbsolutePath());
                runPython(pyWrite);

                Map<String, NDArray> loaded = NP.loadz(npzFile.toString());
                check("Python .npz has x", loaded.containsKey("x"));
                check("Python .npz has y", loaded.containsKey("y"));
                check("Python .npz x shape 2x3", loaded.get("x").shape[0] == 2 && loaded.get("x").shape[1] == 3);
            });

        } finally {
            try {
                Files.walk(tmpDir).sorted(java.util.Comparator.reverseOrder())
                    .map(Path::toFile).forEach(File::delete);
            } catch (Exception e) {
                System.err.println("Cleanup: " + e.getMessage());
            }
        }

        System.out.println("\n=== NumPy Benchmark Results ===");
        System.out.println("Passed: " + passed);
        System.out.println("Failed: " + failed);
        if (failed > 0) {
            System.out.println("\nFAILED TESTS:");
            System.out.println(report);
            System.exit(1);
        } else {
            System.out.println("\nAll tests PASSED!");
        }
    }

    static String runPython(String code) throws Exception {
        ProcessBuilder pb = new ProcessBuilder("python3", "-c", code);
        pb.redirectErrorStream(true);
        Process p = pb.start();
        String output = new String(p.getInputStream().readAllBytes());
        int exitCode = p.waitFor();
        if (exitCode != 0) return "ERROR: " + output;
        return output.trim();
    }

    @FunctionalInterface
    interface CheckedRunnable { void run() throws Exception; }

    static void benchmark(String name, CheckedRunnable r) {
        try {
            r.run();
            System.out.println("  ✓ " + name);
        } catch (Throwable t) {
            failed++;
            report.append("  FAIL [").append(name).append("]: ").append(t.getMessage()).append("\n");
            System.out.println("  ✗ " + name + " — " + t.getMessage());
        }
    }

    static void check(String name, boolean condition) {
        if (condition) passed++;
        else { failed++; report.append("  CHECK FAILED: ").append(name).append("\n"); }
    }
}
