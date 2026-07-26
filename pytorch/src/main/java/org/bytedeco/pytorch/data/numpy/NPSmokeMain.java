package org.bytedeco.pytorch.data.numpy;

/**
 * Standalone smoke checks for pure-Java NumPy ops (no LibTorch native required).
 * Run: {@code java org.bytedeco.pytorch.data.numpy.NPSmokeMain}
 */
public final class NPSmokeMain {
    private NPSmokeMain() {}

    public static void main(String[] args) {
        int failed = 0;
        failed += check("unary", () -> {
            NDArray a = NP.array(new double[]{-4, -1, 0, 1, 4});
            assertEq(NP.abs(a).getDouble(0), 4);
            assertEq(NP.square(a).getDouble(0), 16);
            assertEq(NP.sign(a).getDouble(0), -1);
            assertEq(NP.sqrt(NP.array(new double[]{9})).getDouble(0), 3);
            assertEq(NP.exp(NP.array(new double[]{0})).getDouble(0), 1);
            assertEq(NP.sin(NP.array(new double[]{0})).getDouble(0), 0);
        });
        failed += check("binary-broadcast", () -> {
            NDArray a = NP.array(new double[]{1, 2, 3}, 3, 1);
            NDArray b = NP.array(new double[]{10, 20, 30});
            NDArray c = NP.add(a, b); // (3,1)+(3,) → (3,3)
            assertTrue(c.shape[0] == 3 && c.shape[1] == 3);
            assertEq(c.getDouble(0), 11); // 1+10
            assertEq(c.getDouble(2), 31); // 1+30
            assertEq(c.getDouble(3), 12); // 2+10
        });
        failed += check("reduce-axis", () -> {
            NDArray a = NP.array(new double[]{1, 2, 3, 4}, 2, 2);
            assertEq(NP.sum(a), 10);
            NDArray row = NP.sum(a, 1, false);
            assertTrue(row.shape.length == 1 && row.shape[0] == 2);
            assertEq(row.getDouble(0), 3);
            assertEq(row.getDouble(1), 7);
            NDArray keep = NP.sum(a, 1, true);
            assertTrue(keep.shape[0] == 2 && keep.shape[1] == 1);
            assertEq(NP.mean(a), 2.5);
            assertTrue(NP.argmax(a) == 3);
        });
        failed += check("compare-logic", () -> {
            NDArray a = NP.array(new double[]{1, 2, 3});
            NDArray b = NP.array(new double[]{2, 2, 2});
            NDArray g = NP.greater(a, b);
            assertTrue(g.dtype == DType.BOOL);
            assertEq(g.getDouble(0), 0);
            assertEq(g.getDouble(2), 1);
            assertTrue(NP.all(NP.equal(a, a)));
            assertTrue(NP.any(NP.greater(a, b)));
        });
        failed += check("shape", () -> {
            NDArray a = NP.arange(6);
            NDArray r = NP.reshape(a, 2, 3);
            assertTrue(r.shape[0] == 2 && r.shape[1] == 3);
            NDArray t = NP.transpose(r);
            assertTrue(t.shape[0] == 3 && t.shape[1] == 2);
            assertEq(t.getDouble(1), 3); // [[0,3],[1,4],[2,5]]
            NDArray s = NP.stack(new NDArray[]{a, a}, 0);
            assertTrue(s.shape[0] == 2 && s.shape[1] == 6);
            NDArray f = NP.flip(NP.array(new double[]{1, 2, 3}), 0);
            assertEq(f.getDouble(0), 3);
        });
        failed += check("matmul-dot", () -> {
            NDArray a = NP.array(new double[]{1, 2, 3, 4}, 2, 2);
            NDArray b = NP.array(new double[]{5, 6, 7, 8}, 2, 2);
            NDArray c = NP.matmul(a, b);
            // [[1,2],[3,4]]*[[5,6],[7,8]] = [[19,22],[43,50]]
            assertEq(c.getDouble(0), 19);
            assertEq(c.getDouble(1), 22);
            assertEq(c.getDouble(2), 43);
            assertEq(c.getDouble(3), 50);
            NDArray d = NP.dot(NP.array(new double[]{1, 2, 3}), NP.array(new double[]{4, 5, 6}));
            assertEq(d.getDouble(0), 32);
        });
        failed += check("linalg", () -> {
            NDArray a = NP.array(new double[]{4, 7, 2, 6}, 2, 2);
            double det = NP.Linalg.det(a); // 4*6-7*2=10
            assertEq(det, 10);
            NDArray inv = NP.Linalg.inv(a);
            NDArray id = NP.matmul(a, inv);
            assertTrue(Math.abs(id.getDouble(0) - 1) < 1e-9);
            assertTrue(Math.abs(id.getDouble(1)) < 1e-9);
            assertTrue(Math.abs(id.getDouble(3) - 1) < 1e-9);
            NDArray x = NP.Linalg.solve(a, NP.array(new double[]{1, 0}));
            // A x = [1,0]
            NDArray ax = NP.matmul(a, NP.reshape(x, 2, 1));
            assertTrue(Math.abs(ax.getDouble(0) - 1) < 1e-9);
            assertTrue(Math.abs(ax.getDouble(1)) < 1e-9);
        });
        failed += check("fft", () -> {
            NDArray a = NP.array(new double[]{1, 0, 0, 0, 0, 0, 0, 0});
            NDArray[] y = NP.Fft.fft(a);
            // DFT of impulse ≈ all ones (real)
            for (int i = 0; i < 8; i++) {
                assertTrue(Math.abs(y[0].getDouble(i) - 1) < 1e-9);
                assertTrue(Math.abs(y[1].getDouble(i)) < 1e-9);
            }
            NDArray[] back = NP.Fft.ifft(y[0], y[1], null, -1);
            assertTrue(Math.abs(back[0].getDouble(0) - 1) < 1e-9);
            for (int i = 1; i < 8; i++) assertTrue(Math.abs(back[0].getDouble(i)) < 1e-9);
        });
        failed += check("random", () -> {
            NP.Random.seed(42);
            NDArray a = NP.Random.rand(1000);
            double mu = NP.mean(a);
            assertTrue(mu > 0.4 && mu < 0.6);
            NDArray p = NP.Random.permutation(5);
            assertTrue(p.size == 5);
            double s = 0;
            for (int i = 0; i < 5; i++) s += p.getDouble(i);
            assertEq(s, 10); // 0+1+2+3+4
        });
        failed += check("sort-unique-median", () -> {
            NDArray a = NP.array(new double[]{3, 1, 2, 1});
            NDArray s = NP.sort(a);
            assertEq(s.getDouble(0), 1);
            assertEq(s.getDouble(3), 3);
            NDArray u = NP.unique(a);
            assertTrue(u.size == 3);
            assertEq(NP.median(a), 1.5);
            assertEq(NP.percentile(a, 0), 1);
            assertEq(NP.percentile(a, 100), 3);
        });
        failed += check("clip-where-diff", () -> {
            NDArray a = NP.array(new double[]{-2, 0, 5});
            NDArray c = NP.clip(a, 0, 3);
            assertEq(c.getDouble(0), 0);
            assertEq(c.getDouble(2), 3);
            NDArray w = NP.where(NP.greater(a, NP.zeros(3)), a, NP.zeros(3));
            assertEq(w.getDouble(0), 0);
            assertEq(w.getDouble(2), 5);
            NDArray d = NP.diff(NP.array(new double[]{1, 2, 4, 7}));
            assertEq(d.getDouble(0), 1);
            assertEq(d.getDouble(2), 3);
        });
        failed += check("complex", () -> {
            NDArray z = NP.complex(NP.array(new double[]{3, 0}), NP.array(new double[]{4, 1}));
            assertTrue(z.isComplex());
            assertEq(NP.real(z).getDouble(0), 3);
            assertEq(NP.imag(z).getDouble(0), 4);
            assertEq(NP.abs(z).getDouble(0), 5);
            NDArray cj = NP.conj(z);
            assertEq(cj.getImag(0), -4);
            NDArray prod = NP.multiply(z, cj);
            assertEq(prod.getReal(0), 25);
            assertTrue(Math.abs(prod.getImag(0)) < 1e-9);
        });
        failed += check("partition-quickselect", () -> {
            NDArray a = NP.array(new double[]{9, 1, 5, 3, 7});
            NDArray p = NP.partition(a, 2);
            // kth=2 element should be 3rd smallest (=5); left side all <= 5
            assertEq(p.getDouble(2), 5);
            for (int i = 0; i < 2; i++) assertTrue(p.getDouble(i) <= 5);
            for (int i = 3; i < 5; i++) assertTrue(p.getDouble(i) >= 5);
        });
        failed += check("as_strided-ogrid", () -> {
            NDArray a = NP.arange(6);
            NDArray v = NP.as_strided(a, new long[]{4}, new long[]{1}, 1); // [1,2,3,4]
            assertEq(v.getDouble(0), 1);
            assertEq(v.getDouble(3), 4);
            assertTrue(v.isView);
            NDArray[] og = NP.ogrid(NP.array(new double[]{0, 1}), NP.array(new double[]{10, 20, 30}));
            assertTrue(og[0].shape[0] == 2 && og[0].shape[1] == 1);
            assertTrue(og[1].shape[0] == 1 && og[1].shape[1] == 3);
            NDArray[] mg = NP.mgrid(NP.array(new double[]{0, 1}), NP.array(new double[]{10, 20}));
            assertTrue(mg[0].shape[0] == 2 && mg[0].shape[1] == 2);
            assertEq(mg[0].getDouble(0), 0);
            assertEq(mg[1].getDouble(1), 20);
        });
        failed += check("poly-pca-mask-bits", () -> {
            NDArray x = NP.array(new double[]{0, 1, 2, 3});
            NDArray y = NP.array(new double[]{1, 3, 5, 7}); // 2x+1
            NDArray coef = NP.polyfit(x, y, 1);
            assertTrue(Math.abs(coef.getDouble(0) - 2) < 1e-6);
            assertTrue(Math.abs(coef.getDouble(1) - 1) < 1e-6);
            assertEq(NP.polyval(coef, 4), 9);

            NDArray X = NP.array(new double[]{1, 2, 3, 2, 4, 6, 3, 6, 9}, 3, 3);
            NPPCA.Result pca = NP.PCA.fitTransform(X, 1);
            assertTrue(pca.transformed.shape[0] == 3 && pca.transformed.shape[1] == 1);

            MaskedArray ma = NP.Ma.masked_equal(NP.array(new double[]{1, 2, 2, 3}), 2);
            assertEq(ma.count(), 2);
            assertEq(ma.mean(), 2.0); // (1+3)/2

            NDArray bits = NP.array(new int[]{1, 1, 0, 0, 1, 0, 1, 0});
            NDArray packed = NP.packbits(bits);
            assertEq(packed.getDouble(0), 0xCA); // 11001010
            NDArray unp = NP.unpackbits(packed);
            for (int i = 0; i < 8; i++) assertEq(unp.getDouble(i), bits.getDouble(i));
        });
        failed += check("plot-savefig", () -> {
            NDArray xx = NP.linspace(0, 2 * Math.PI, 50);
            NDArray yy = NP.sin(xx);
            NP.Plot.plot(xx, yy, "sin");
            NP.Plot.title("smoke");
            NP.Plot.legend(true);
            java.nio.file.Path tmp = java.nio.file.Files.createTempFile("np-plot-", ".png");
            try {
                NP.Plot.savefig(tmp.toString());
                assertTrue(java.nio.file.Files.size(tmp) > 100);
            } finally {
                java.nio.file.Files.deleteIfExists(tmp);
            }
        });

        if (failed == 0) {
            System.out.println("NPSmokeMain: ALL PASSED");
        } else {
            System.out.println("NPSmokeMain: FAILED " + failed + " checks");
            System.exit(1);
        }
    }

    private interface Body { void run() throws Exception; }

    private static int check(String name, Body body) {
        try {
            body.run();
            System.out.println("  ok  " + name);
            return 0;
        } catch (Throwable t) {
            System.out.println("  FAIL " + name + ": " + t.getMessage());
            t.printStackTrace(System.out);
            return 1;
        }
    }

    private static void assertEq(double a, double b) {
        if (Math.abs(a - b) > 1e-9) throw new AssertionError(a + " != " + b);
    }

    private static void assertTrue(boolean c) {
        if (!c) throw new AssertionError("assertTrue failed");
    }
}
