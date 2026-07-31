package dataframe;

import org.bytedeco.pytorch.data.pickle.Pickle;
import org.bytedeco.pytorch.global.torch;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * Comprehensive benchmark for the Pickle module.
 * Tests all Pickle.load/dump/loads/dumps, tensor serialization, map encoding.
 * Also validates Python interop: Java writes pickle that Python reads,
 * Python writes pickle that Java reads.
 */
public class BenchmarkPickle {

    static int passed = 0;
    static int failed = 0;
    static StringBuilder report = new StringBuilder();

    public static void main(String[] args) throws Exception {
        System.out.println("=== Pickle Module Benchmark ===\n");

        Path tmpDir = Files.createTempDirectory("pickle_bench");
        System.out.println("Temp dir: " + tmpDir);

        try {
            // ── 1. Primitives ─────────────────────────────────────────────
            benchmark("Pickle primitives - null", () -> {
                byte[] pickle = Pickle.dumps(null);
                Object obj = Pickle.loads(pickle);
                check("null roundtrip", obj == null);
            });

            benchmark("Pickle primitives - boolean", () -> {
                byte[] pTrue = Pickle.dumps(Boolean.TRUE);
                byte[] pFalse = Pickle.dumps(Boolean.FALSE);
                check("true roundtrip", Pickle.loads(pTrue) == Boolean.TRUE);
                check("false roundtrip", Pickle.loads(pFalse) == Boolean.FALSE);
            });

            benchmark("Pickle primitives - int", () -> {
                for (int val : new int[]{0, 1, -1, 127, 128, 255, 256, 32767, 65535, Integer.MAX_VALUE, Integer.MIN_VALUE}) {
                    byte[] p = Pickle.dumps(val);
                    int r = (Integer) Pickle.loads(p);
                    check("int roundtrip " + val, r == val);
                }
            });

            benchmark("Pickle primitives - long", () -> {
                for (long val : new long[]{0L, 1L, -1L, 127L, Long.MAX_VALUE, Long.MIN_VALUE, 1L << 62}) {
                    byte[] p = Pickle.dumps(val);
                    long r = (Long) Pickle.loads(p);
                    check("long roundtrip " + val, r == val);
                }
            });

            benchmark("Pickle primitives - float", () -> {
                for (float val : new float[]{0f, 1f, -1f, Float.MAX_VALUE, Float.MIN_VALUE, 1.5f, (float) Math.PI}) {
                    byte[] p = Pickle.dumps(val);
                    float r = (Float) Pickle.loads(p);
                    check("float roundtrip " + val, r == val);
                }
            });

            benchmark("Pickle primitives - double", () -> {
                for (double val : new double[]{0.0, 1.0, -1.0, Double.MAX_VALUE, Double.MIN_VALUE, Math.PI, Math.E, 1e-100, 1e100}) {
                    byte[] p = Pickle.dumps(val);
                    double r = (Double) Pickle.loads(p);
                    check("double roundtrip " + val, r == val);
                }
            });

            benchmark("Pickle primitives - String", () -> {
                for (String val : new String[]{"", "hello", "hello world", "unicode 中文 😀", "new\nline\ttab", "a".repeat(1000)}) {
                    byte[] p = Pickle.dumps(val);
                    String r = (String) Pickle.loads(p);
                    check("String '" + (val.length() > 20 ? "..." : val) + "'", r.equals(val));
                }
            });

            // ── 2. Arrays ────────────────────────────────────────────────
            benchmark("Pickle byte[]", () -> {
                byte[] original = "hello bytes".getBytes();
                byte[] p = Pickle.dumps(original);
                byte[] r = (byte[]) Pickle.loads(p);
                check("byte[] roundtrip", Arrays.equals(r, original));
            });

            benchmark("Pickle double[]", () -> {
                double[] original = {1.1, 2.2, 3.3, Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY};
                byte[] p = Pickle.dumps(original);
                double[] r = (double[]) Pickle.loads(p);
                check("double[] length", r.length == original.length);
                for (int i = 0; i < original.length; i++) {
                    boolean match = Double.compare(r[i], original[i]) == 0;
                    check("double[][" + i + "]", match);
                }
            });

            benchmark("Pickle long[]", () -> {
                long[] original = {0, 1, -1, Long.MAX_VALUE, Long.MIN_VALUE, 1L << 50};
                byte[] p = Pickle.dumps(original);
                long[] r = (long[]) Pickle.loads(p);
                check("long[] roundtrip", Arrays.equals(r, original));
            });

            // ── 3. List ──────────────────────────────────────────────────
            benchmark("Pickle List", () -> {
                List<Object> list = Arrays.asList(1, "hello", 3.14, true, null);
                byte[] p = Pickle.dumps(list);
                @SuppressWarnings("unchecked")
                List<Object> r = (List<Object>) Pickle.loads(p);
                check("List size 5", r.size() == 5);
                check("List[0]=1", r.get(0).equals(1));
                check("List[1]=hello", r.get(1).equals("hello"));
                check("List[2]=3.14", ((Double)r.get(2)).equals(3.14));
                check("List[3]=true", r.get(3).equals(true));
                check("List[4]=null", r.get(4) == null);
            });

            // ── 4. Map ───────────────────────────────────────────────────
            benchmark("Pickle Map", () -> {
                Map<String, Object> map = new LinkedHashMap<>();
                map.put("key1", 42);
                map.put("key2", "value");
                map.put("key3", 3.14);
                map.put("nested", Arrays.asList(1, 2, 3));
                byte[] p = Pickle.dumps(map);
                @SuppressWarnings("unchecked")
                Map<String, Object> r = (Map<String, Object>) Pickle.loads(p);
                check("Map size 4", r.size() == 4);
                check("Map key1=42", ((Integer)r.get("key1")) == 42);
                check("Map key2=value", r.get("key2").equals("value"));
                check("Map key3=3.14", ((Double)r.get("key3")).equals(3.14));
                @SuppressWarnings("unchecked")
                List<Object> nested = (List<Object>) r.get("nested");
                check("Map nested size 3", nested.size() == 3);
            });

            // ── 5. File I/O ─────────────────────────────────────────────
            benchmark("Pickle file I/O - dump/load File", () -> {
                Path pklFile = tmpDir.resolve("primitives.pkl");

                Pickle.dump(42, pklFile.toFile());
                Integer i = (Integer) Pickle.load(pklFile.toFile());
                check("dump/load int 42", i == 42);

                Pickle.dump("hello file", pklFile.toFile());
                String s = (String) Pickle.load(pklFile.toFile());
                check("dump/load String", s.equals("hello file"));

                double[] arr = {1.1, 2.2, 3.3};
                Pickle.dump(arr, pklFile.toFile());
                double[] rarr = (double[]) Pickle.load(pklFile.toFile());
                check("dump/load double[]", Arrays.equals(rarr, arr));
            });

            benchmark("Pickle file I/O - dump/load OutputStream/InputStream", () -> {
                ByteArrayOutputStream baos = new ByteArrayOutputStream();
                Pickle.dump(Arrays.asList(1, 2, 3), baos);
                byte[] data = baos.toByteArray();
                check("dump to OutputStream size > 0", data.length > 0);

                ByteArrayInputStream bais = new ByteArrayInputStream(data);
                @SuppressWarnings("unchecked")
                List<Object> list = (List<Object>) Pickle.load(bais);
                check("load from InputStream size=3", list.size() == 3);
            });

            // ── 6. Tensor roundtrip ──────────────────────────────────────
            benchmark("Pickle dumpTensor/loadTensor", () -> {
                Path pklFile = tmpDir.resolve("tensor.pkl");

                // Test scalar
                org.bytedeco.pytorch.Tensor scalar = torch.tensor(42.0);
                Pickle.dumpTensor(scalar, pklFile.toFile());
                org.bytedeco.pytorch.Tensor loadedScalar = Pickle.loadTensor(pklFile.toFile());
                check("loadTensor scalar value", Math.abs(loadedScalar.item_double() - 42.0) < 1e-9);

                // Test 2D
                org.bytedeco.pytorch.Tensor t2d = torch.randn(new long[]{3, 4});
                Pickle.dumpTensor(t2d, pklFile.toFile());
                org.bytedeco.pytorch.Tensor loaded2d = Pickle.loadTensor(pklFile.toFile());
                check("loadTensor 2D shape", loaded2d.dim() == 2 && loaded2d.size(0) == 3 && loaded2d.size(1) == 4);
                check("loadTensor 2D dtype", loaded2d.scalar_type() == t2d.scalar_type());

                // Test 4D
                org.bytedeco.pytorch.Tensor t4d = torch.randn(new long[]{2, 3, 4, 5});
                Pickle.dumpTensor(t4d, pklFile.toFile());
                org.bytedeco.pytorch.Tensor loaded4d = Pickle.loadTensor(pklFile.toFile());
                check("loadTensor 4D numel=120", loaded4d.numel() == 120);
            });

            // ── 7. tensorToMap / mapToTensor ─────────────────────────────
            benchmark("Pickle tensorToMap/mapToTensor", () -> {
                org.bytedeco.pytorch.Tensor t = torch.randn(new long[]{3, 4});
                Map<String, Object> map = Pickle.tensorToMap(t);
                check("tensorToMap has __torch_tensor__", map.containsKey("__torch_tensor__"));
                check("tensorToMap has shape", map.containsKey("shape"));
                check("tensorToMap has dtype", map.containsKey("dtype"));

                org.bytedeco.pytorch.Tensor back = Pickle.mapToTensor(map);
                check("mapToTensor shape 3x4", back.dim() == 2 && back.size(0) == 3 && back.size(1) == 4);
            });

            // ── 8. Python interop: Java → Python ─────────────────────────
            benchmark("Python interop: Java pickle → Python reads", () -> {
                Path pklFile = tmpDir.resolve("java.pkl");

                // Write a dict with multiple types
                Map<String, Object> data = new LinkedHashMap<>();
                data.put("int_val", 42);
                data.put("float_val", 3.14);
                data.put("str_val", "hello");
                data.put("list_val", Arrays.asList(1, 2, 3));
                Pickle.dump(data, pklFile.toFile());

                String pyCheck = String.format(
                    "import pickle; " +
                    "with open('%s', 'rb') as f: d = pickle.load(f); " +
                    "assert d['int_val'] == 42, f'int {d[\"int_val\"]}'; " +
                    "assert abs(d['float_val'] - 3.14) < 1e-9, f'float {d[\"float_val\"]}'; " +
                    "assert d['str_val'] == 'hello', f'str {d[\"str_val\"]}'; " +
                    "assert list(d['list_val']) == [1,2,3], f'list {d[\"list_val\"]}'; " +
                    "print('PASS')",
                    pklFile.toAbsolutePath());

                String result = runPython(pyCheck);
                check("Java pickle Python reads", result.contains("PASS"));
            });

            benchmark("Python interop: Java dumps → Python loads", () -> {
                byte[] data = Pickle.dumps(Arrays.asList("a", "b", "c"));
                String pyCheck = "import pickle; lst = pickle.loads(" +
                    repr(data) + "); assert lst == ['a','b','c'], f'lst {lst}'; print('PASS')";

                String result = runPython(pyCheck);
                check("Java dumps Python loads", result.contains("PASS"));
            });

            // ── 9. Python interop: Python → Java ─────────────────────────
            benchmark("Python interop: Python pickle → Java reads", () -> {
                Path pklFile = tmpDir.resolve("python.pkl");

                // Write using Python
                String pyWrite = String.format(
                    "import pickle; " +
                    "data = {'int': 99, 'float': 2.718, 'str': 'world', 'list': [4,5,6]}; " +
                    "with open('%s', 'wb') as f: pickle.dump(data, f)",
                    pklFile.toAbsolutePath());
                runPython(pyWrite);

                @SuppressWarnings("unchecked")
                Map<String, Object> loaded = (Map<String, Object>) Pickle.load(pklFile.toFile());
                check("Python pickle int=99", ((Number)loaded.get("int")).intValue() == 99);
                check("Python pickle float≈2.718", Math.abs((Double)loaded.get("float") - 2.718) < 1e-6);
                check("Python pickle str=world", loaded.get("str").equals("world"));
                @SuppressWarnings("unchecked")
                List<Object> lst = (List<Object>) loaded.get("list");
                check("Python pickle list len=3", lst.size() == 3);
            });

        } finally {
            try {
                Files.walk(tmpDir).sorted(java.util.Comparator.reverseOrder())
                    .map(Path::toFile).forEach(File::delete);
            } catch (Exception e) {
                System.err.println("Cleanup: " + e.getMessage());
            }
        }

        System.out.println("\n=== Pickle Benchmark Results ===");
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

    static String repr(byte[] data) {
        StringBuilder sb = new StringBuilder("b'");
        for (byte b : data) {
            if (b >= 32 && b < 127 && b != '\'' && b != '\\') sb.append((char)b);
            else sb.append(String.format("\\x%02x", b & 0xff));
        }
        sb.append("'");
        return sb.toString();
    }

    static String runPython(String code) throws Exception {
        ProcessBuilder pb = new ProcessBuilder("python3", "-c", code);
        pb.redirectErrorStream(true);
        Process p = pb.start();
        String output = new String(p.getInputStream().readAllBytes());
        int exitCode = p.waitFor();
        if (exitCode != 0) {
            return "ERROR: " + output;
        }
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
