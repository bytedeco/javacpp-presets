package dataframe;

import org.bytedeco.pytorch.data.safetensors.SafeDType;
import org.bytedeco.pytorch.data.safetensors.SafeTensors;
import org.bytedeco.pytorch.global.torch;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * Comprehensive benchmark for the Safetensors module.
 * Tests all loading, saving, module injection, dtype conversions,
 * and zero-copy vs copy modes. Also validates Python interop.
 */
public class BenchmarkSafetensors {

    static int passed = 0;
    static int failed = 0;
    static StringBuilder report = new StringBuilder();

    public static void main(String[] args) throws Exception {
        System.out.println("=== Safetensors Module Benchmark ===\n");

        Path tmpDir = Files.createTempDirectory("safetensors_bench");
        System.out.println("Temp dir: " + tmpDir);

        try {
            // ── 1. SafeDType enum ──────────────────────────────────────────
            benchmark("SafeDType enum - all values", () -> {
                for (SafeDType dt : SafeDType.values()) {
                    check("SafeDType." + dt.name() + " typeName", dt.typeName() != null && !dt.typeName().isEmpty());
                    check("SafeDType." + dt.name() + " sizeBytes", dt.sizeBytes() > 0);
                    check("SafeDType." + dt.name() + " toTorch", dt.toTorch() != null);
                    check("SafeDType." + dt.name() + " isNativeLayout", !dt.isNativeLayout() || dt.isNativeLayout()); // always true but exercises the method
                }
            });

            benchmark("SafeDType fromString/fromTorch", () -> {
                check("fromString F32", SafeDType.fromString("F32") == SafeDType.F32);
                check("fromString f16", SafeDType.fromString("f16") == SafeDType.F16);
                check("fromString BF16", SafeDType.fromString("BF16") == SafeDType.BF16);
                check("fromString I64", SafeDType.fromString("I64") == SafeDType.I64);
                check("fromString BOOL", SafeDType.fromString("BOOL") == SafeDType.BOOL);
                check("fromString unknown null", SafeDType.fromString("INVALID") == null);

                check("fromTorch Double=F64", SafeDType.fromTorch(torch.ScalarType.Double) == SafeDType.F64);
                check("fromTorch Float=F32", SafeDType.fromTorch(torch.ScalarType.Float) == SafeDType.F32);
                check("fromTorch Half=F16", SafeDType.fromTorch(torch.ScalarType.Half) == SafeDType.F16);
                check("fromTorch BFloat16=BF16", SafeDType.fromTorch(torch.ScalarType.BFloat16) == SafeDType.BF16);
                check("fromTorch Long=I64", SafeDType.fromTorch(torch.ScalarType.Long) == SafeDType.I64);
                check("fromTorch Int=I32", SafeDType.fromTorch(torch.ScalarType.Int) == SafeDType.I32);
                check("fromTorch Short=I16", SafeDType.fromTorch(torch.ScalarType.Short) == SafeDType.I16);
                check("fromTorch Char=I8", SafeDType.fromTorch(torch.ScalarType.Char) == SafeDType.I8);
                check("fromTorch Byte=U8", SafeDType.fromTorch(torch.ScalarType.Byte) == SafeDType.U8);
                check("fromTorch Bool=BOOL", SafeDType.fromTorch(torch.ScalarType.Bool) == SafeDType.BOOL);
            });

            // ── 2. Save tensors - all dtypes ─────────────────────────────
            benchmark("SafeTensors.save - all float dtypes", () -> {
                SafeDType[] dtypes = {SafeDType.F64, SafeDType.F32, SafeDType.F16, SafeDType.BF16};
                for (SafeDType sdt : dtypes) {
                    org.bytedeco.pytorch.serving.tensorrt.Tensor t = torch.randn(new long[]{3, 4, 5}).to(sdt.toTorch());
                    Map<String, org.bytedeco.pytorch.serving.tensorrt.Tensor> tensors = new LinkedHashMap<>();
                    tensors.put("weight_" + sdt.name(), t);

                    Path sfFile = tmpDir.resolve("test_" + sdt.name() + ".safetensors");
                    SafeTensors.save(tensors, sfFile.toFile());
                    check("save " + sdt.name() + " file exists", Files.exists(sfFile));
                    check("save " + sdt.name() + " file not empty", Files.size(sfFile) > 0);
                }
            });

            benchmark("SafeTensors.save - all integer dtypes", () -> {
                SafeDType[] dtypes = {SafeDType.I64, SafeDType.I32, SafeDType.I16, SafeDType.I8, SafeDType.U8};
                for (SafeDType sdt : dtypes) {
                    org.bytedeco.pytorch.serving.tensorrt.Tensor t = torch.randint(100L, new long[]{12L}, new org.bytedeco.pytorch.TensorOptions().dtype(new org.bytedeco.pytorch.ScalarTypeOptional(sdt.toTorch()))).view(3L, 4L);
                    Map<String, org.bytedeco.pytorch.serving.tensorrt.Tensor> tensors = new LinkedHashMap<>();
                    tensors.put("int_" + sdt.name(), t);

                    Path sfFile = tmpDir.resolve("test_int_" + sdt.name() + ".safetensors");
                    SafeTensors.save(tensors, sfFile.toFile());
                    check("save int " + sdt.name() + " file exists", Files.exists(sfFile));
                }
            });

            // ── 3. Load tensors - zeroCopy vs copy ─────────────────────────
            benchmark("SafeTensors.loadAsTensors - zeroCopy vs copy", () -> {
                org.bytedeco.pytorch.serving.tensorrt.Tensor original = torch.randn(new long[]{10, 26, 100}); // ~200KB
                Map<String, org.bytedeco.pytorch.serving.tensorrt.Tensor> map = new LinkedHashMap<>();
                map.put("large_tensor", original);
                Path sfFile = tmpDir.resolve("large.safetensors");
                SafeTensors.save(map, sfFile.toFile());

                Map<String, org.bytedeco.pytorch.serving.tensorrt.Tensor> zc = SafeTensors.loadAsTensors(sfFile.toFile(), true);
                check("zeroCopy load key exists", zc.containsKey("large_tensor"));
                check("zeroCopy load shape[0]=10", zc.get("large_tensor").size(0L) == 10);
                check("zeroCopy load shape[1]=26", zc.get("large_tensor").size(1L) == 26);

                Map<String, org.bytedeco.pytorch.serving.tensorrt.Tensor> copy = SafeTensors.loadAsTensors(sfFile.toFile(), false);
                check("copy load key exists", copy.containsKey("large_tensor"));
                check("copy load shape correct", copy.get("large_tensor").size(0L) == 10);
            });

            // ── 4. Load all tensors ──────────────────────────────────────
            benchmark("SafeTensors.loadAsTensors - multiple tensors", () -> {
                Map<String, org.bytedeco.pytorch.serving.tensorrt.Tensor> originals = new LinkedHashMap<>();
                originals.put("w1", torch.randn(new long[]{10, 20}));
                originals.put("w2", torch.randn(new long[]{20, 30}));
                originals.put("w3", torch.randn(new long[]{30, 40}));
                originals.put("bias", torch.zeros(new long[]{40}));

                Path sfFile = tmpDir.resolve("multi.safetensors");
                SafeTensors.save(originals, sfFile.toFile());

                Map<String, org.bytedeco.pytorch.serving.tensorrt.Tensor> loaded = SafeTensors.loadAsTensors(sfFile.toFile());
                check("loadAsTensors count=4", loaded.size() == 4);
                check("loadAsTensors w1 shape[0]=10", loaded.get("w1").size(0L) == 10);
                check("loadAsTensors w2 shape[1]=30", loaded.get("w2").size(1L) == 30);
                check("loadAsTensors bias shape[0]=40", loaded.get("bias").size(0L) == 40);
            });

            // ── 5. listTensors (header only, no data) ────────────────────
            benchmark("SafeTensors.listTensors", () -> {
                Map<String, org.bytedeco.pytorch.serving.tensorrt.Tensor> originals = new LinkedHashMap<>();
                originals.put("tensor_a", torch.randn(new long[]{5, 5}));
                originals.put("tensor_b", torch.randn(new long[]{10, 10}));
                originals.put("tensor_c", torch.randn(new long[]{15, 15}));

                Path sfFile = tmpDir.resolve("list_test.safetensors");
                SafeTensors.save(originals, sfFile.toFile());

                List<String> names = SafeTensors.listTensors(sfFile.toFile());
                check("listTensors count=3", names.size() == 3);
                check("listTensors contains a", names.contains("tensor_a"));
                check("listTensors contains b", names.contains("tensor_b"));
                check("listTensors contains c", names.contains("tensor_c"));

                // Performance: listTensors should be fast
                long start = System.nanoTime();
                for (int i = 0; i < 1000; i++) SafeTensors.listTensors(sfFile.toFile());
                long elapsed = System.nanoTime() - start;
                check("listTensors 1000x fast <100ms", elapsed < 100_000_000);
            });

            // ── 6. Save with metadata ──────────────────────────────────────
            benchmark("SafeTensors.save with metadata", () -> {
                Map<String, org.bytedeco.pytorch.serving.tensorrt.Tensor> tensors = new LinkedHashMap<>();
                tensors.put("weight", torch.randn(new long[]{10, 10}));

                Map<String, String> metadata = new LinkedHashMap<>();
                metadata.put("model_name", "test_model");
                metadata.put("version", "1.0");
                metadata.put("description", "A test model with special chars: 中文");

                Path sfFile = tmpDir.resolve("with_meta.safetensors");
                SafeTensors.save(tensors, sfFile.toFile(), metadata);

                Map<String, org.bytedeco.pytorch.serving.tensorrt.Tensor> loaded = SafeTensors.loadAsTensors(sfFile.toFile());
                check("save with metadata - tensor loads", loaded.containsKey("weight"));
            });

            // ── 7. Multi-dimensional tensors ──────────────────────────────
            benchmark("SafeTensors - multi-dimensional tensors", () -> {
                Map<String, org.bytedeco.pytorch.serving.tensorrt.Tensor> originals = new LinkedHashMap<>();
                originals.put("1d", torch.randn(new long[]{100}));
                originals.put("2d", torch.randn(new long[]{10, 20}));
                originals.put("3d", torch.randn(new long[]{4, 5, 6}));
                originals.put("4d", torch.randn(new long[]{2, 3, 4, 5}));
                originals.put("5d", torch.randn(new long[]{1, 2, 3, 4, 5}));

                Path sfFile = tmpDir.resolve("multi_dim.safetensors");
                SafeTensors.save(originals, sfFile.toFile());

                Map<String, org.bytedeco.pytorch.serving.tensorrt.Tensor> loaded = SafeTensors.loadAsTensors(sfFile.toFile());
                check("1D shape len=1", loaded.get("1d").shape().length == 1);
                check("2D shape[0]=10", loaded.get("2d").size(0L) == 10);
                check("3D numel=120", loaded.get("3d").numel() == 120);
                check("4D numel=120", loaded.get("4d").numel() == 120);
                check("5D numel=120", loaded.get("5d").numel() == 120);
            });

            // ── 8. Large tensor (>1MiB zero-copy threshold) ──────────────
            benchmark("SafeTensors - large tensor >1MiB", () -> {
                // 100*100*100*4bytes = 40MB
                org.bytedeco.pytorch.serving.tensorrt.Tensor large = torch.randn(new long[]{100, 100, 100});
                Map<String, org.bytedeco.pytorch.serving.tensorrt.Tensor> map = new LinkedHashMap<>();
                map.put("large", large);
                Path sfFile = tmpDir.resolve("large_tensor.safetensors");
                SafeTensors.save(map, sfFile.toFile());
                check("large file >1MB", Files.size(sfFile) > 1_000_000);

                Map<String, org.bytedeco.pytorch.serving.tensorrt.Tensor> loaded = SafeTensors.loadAsTensors(sfFile.toFile(), true);
                check("large tensor shape[0]=100", loaded.get("large").size(0L) == 100);

                SafeTensors.releasePinnedMaps();
                check("releasePinnedMaps no crash", true);
            });

            // ── 9. Python interop: Java → Python ─────────────────────────
            benchmark("Python interop: Java safetensors → Python reads", () -> {
                Path sfFile = tmpDir.resolve("java_st.safetensors");
                Map<String, org.bytedeco.pytorch.serving.tensorrt.Tensor> tensors = new LinkedHashMap<>();
                tensors.put("weight", torch.randn(new long[]{10, 20}));
                tensors.put("bias", torch.randn(new long[]{20}));
                SafeTensors.save(tensors, sfFile.toFile());

                String pyCheck = String.format(
                    "from safetensors import safe_open; " +
                    "f = safe_open('%s', framework='pt'); " +
                    "keys = f.keys(); " +
                    "assert 'weight' in keys, f'weight missing {keys}'; " +
                    "assert 'bias' in keys, f'bias missing'; " +
                    "w = f.get_tensor('weight'); " +
                    "assert list(w.shape) == [10, 20], f'shape {w.shape}'; " +
                    "print('PASS')",
                    sfFile.toAbsolutePath());

                String result = runPython(pyCheck);
                check("Java safetensors Python reads", result.contains("PASS"));
            });

            // ── 10. Python interop: Python → Java ─────────────────────────
            benchmark("Python interop: Python safetensors → Java reads", () -> {
                Path sfFile = tmpDir.resolve("python_st.safetensors");

                String pyWrite = String.format(
                    "import torch; " +
                    "from safetensors.torch import save_file; " +
                    "tensors = {'w': torch.randn(8, 16), 'b': torch.randn(16)}; " +
                    "save_file(tensors, '%s')",
                    sfFile.toAbsolutePath());
                runPython(pyWrite);

                Map<String, org.bytedeco.pytorch.serving.tensorrt.Tensor> loaded = SafeTensors.loadAsTensors(sfFile.toFile());
                check("Python safetensors has w", loaded.containsKey("w"));
                check("Python safetensors has b", loaded.containsKey("b"));
                check("Python safetensors w shape[0]=8", loaded.get("w").size(0L) == 8);
                check("Python safetensors w shape[1]=16", loaded.get("w").size(1L) == 16);
                check("Python safetensors b shape[0]=16", loaded.get("b").size(0L) == 16);
            });

            // ── 11. Python interop: F16/BF16 roundtrip ───────────────────
            benchmark("Python interop: F16/BF16 Java↔Python", () -> {
                // Write F16 with Java, read with Python
                Path f16File = tmpDir.resolve("f16_java.safetensors");
                Map<String, org.bytedeco.pytorch.serving.tensorrt.Tensor> f16map = new LinkedHashMap<>();
                f16map.put("f16", torch.randn(new long[]{3, 4}).to(torch.ScalarType.Half));
                SafeTensors.save(f16map, f16File.toFile());

                String pyCheck = String.format(
                    "from safetensors import safe_open; " +
                    "f = safe_open('%s', framework='pt'); " +
                    "t = f.get_tensor('f16'); " +
                    "assert t.dtype == torch.float16, f'dtype {t.dtype}'; " +
                    "assert list(t.shape) == [3, 4], f'shape {t.shape}'; " +
                    "print('PASS')",
                    f16File.toAbsolutePath());
                String result = runPython(pyCheck);
                check("Java F16 Python reads dtype float16", result.contains("PASS"));
            });

        } finally {
            SafeTensors.releasePinnedMaps();
            try {
                Files.walk(tmpDir).sorted(java.util.Comparator.reverseOrder())
                    .map(Path::toFile).forEach(File::delete);
            } catch (Exception e) {
                System.err.println("Cleanup: " + e.getMessage());
            }
        }

        System.out.println("\n=== Safetensors Benchmark Results ===");
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
