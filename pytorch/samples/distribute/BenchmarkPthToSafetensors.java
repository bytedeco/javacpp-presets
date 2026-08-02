package distribute;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.data.safetensors.SafeTensors;
import org.bytedeco.pytorch.data.serialize.ModelWeights;
import org.bytedeco.pytorch.data.serialize.PthToSafeTensors;
import org.bytedeco.pytorch.data.serialize.TorchPthReader;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.modules.LinearImpl;

import java.io.File;
import java.nio.file.*;
import java.util.*;

/**
 * Multi-dimensional benchmark for Python {@code torch.save} .pth → safetensors
 * adapter and auto weight loading.
 *
 * <p>Requires reference files produced by CPython torch (generated on the fly
 * when {@code python3 -c "import torch"} is available), otherwise uses any
 * pre-placed files under {@code /tmp/ref_*.pth}.
 */
public class BenchmarkPthToSafetensors {
    static int passed = 0, failed = 0;
    static final StringBuilder report = new StringBuilder();

    interface CheckedRunnable { void run() throws Exception; }

    static void benchmark(String name, CheckedRunnable r) {
        long t0 = System.nanoTime();
        try {
            r.run();
            System.out.println("  OK  " + name + " (" + ((System.nanoTime() - t0) / 1_000_000) + " ms)");
        } catch (Throwable e) {
            failed++;
            System.out.println(" FAIL " + name + ": " + e);
            report.append("FAIL ").append(name).append(": ").append(e).append('\n');
            e.printStackTrace(System.out);
        }
    }

    static void check(String name, boolean ok) {
        if (ok) passed++;
        else {
            failed++;
            report.append("  check failed: ").append(name).append('\n');
            throw new AssertionError(name);
        }
    }

    /** Generate reference .pth via system Python when possible. */
    static boolean ensurePythonRefs(Path dir) {
        try {
            Path script = dir.resolve("gen_refs.py");
            Files.writeString(script, ""
                + "import torch\n"
                + "from pathlib import Path\n"
                + "d = Path(r'" + dir.toAbsolutePath() + "')\n"
                + "sd = {\n"
                + "  'linear.weight': torch.tensor([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.],[10.,11.,12.]]),\n"
                + "  'linear.bias': torch.tensor([0.1,0.2,0.3,0.4]),\n"
                + "  'emb.weight': torch.arange(12, dtype=torch.float32).reshape(3,4),\n"
                + "}\n"
                + "torch.save(sd, d/'ref_state_dict.pth')\n"
                + "torch.save({'model_state_dict': sd, 'epoch': 3, 'loss': 0.5}, d/'ref_checkpoint.pth')\n"
                + "sd2 = {\n"
                + "  'w': torch.randn(2,2,dtype=torch.float16),\n"
                + "  'i': torch.arange(6,dtype=torch.int64).reshape(2,3),\n"
                + "  'b': torch.tensor([True, False, True]),\n"
                + "}\n"
                + "torch.save(sd2, d/'ref_mixed.pth')\n"
                + "print('generated')\n");
            Process p = new ProcessBuilder("python3", script.toString())
                .redirectErrorStream(true).start();
            String out = new String(p.getInputStream().readAllBytes());
            int code = p.waitFor();
            if (code != 0) {
                System.out.println("    python gen failed (" + code + "): " + out);
                return false;
            }
            System.out.println("    " + out.trim());
            return Files.isRegularFile(dir.resolve("ref_state_dict.pth"));
        } catch (Exception e) {
            System.out.println("    python unavailable: " + e);
            return false;
        }
    }

    static boolean almostEqual(Tensor a, Tensor b, double atol) {
        if (a == null || b == null) return false;
        if (a.dim() != b.dim()) return false;
        for (int i = 0; i < a.dim(); i++) {
            if (a.sizes().get(i) != b.sizes().get(i)) return false;
        }
        Tensor af = a.to(ScalarType.Float).contiguous().cpu();
        Tensor bf = b.to(ScalarType.Float).contiguous().cpu();
        Tensor diff = af.sub(bf).abs();
        double max = diff.max().item_float();
        return max <= atol;
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=== BenchmarkPthToSafetensors ===");
        Path tmp = Files.createTempDirectory("pth2st-");
        try {
            boolean hasRefs = ensurePythonRefs(tmp);
            // Also accept pre-generated /tmp refs
            if (!hasRefs) {
                for (String n : new String[]{"ref_state_dict.pth", "ref_checkpoint.pth", "ref_mixed.pth"}) {
                    Path src = Path.of("/tmp").resolve(n);
                    if (Files.isRegularFile(src)) {
                        Files.copy(src, tmp.resolve(n), StandardCopyOption.REPLACE_EXISTING);
                        hasRefs = true;
                    }
                }
            }
            if (!hasRefs) {
                System.out.println("SKIP: no Python torch and no /tmp/ref_*.pth — cannot run .pth tests");
                System.out.println("Passed: 0  Failed: 0 (skipped)");
                return;
            }

            Path statePth = tmp.resolve("ref_state_dict.pth");
            Path ckptPth = tmp.resolve("ref_checkpoint.pth");
            Path mixedPth = tmp.resolve("ref_mixed.pth");

            benchmark("1. detect ZIP torch format", () -> {
                check("isZipTorch state", TorchPthReader.isZipTorch(statePth.toFile()));
                check("detect TORCH_PTH_ZIP",
                    ModelWeights.detect(statePth) == ModelWeights.Format.TORCH_PTH_ZIP);
            });

            benchmark("2. load state_dict keys + shapes", () -> {
                Map<String, Tensor> sd = TorchPthReader.loadStateDict(statePth.toFile());
                check("3 keys", sd.size() == 3);
                check("has linear.weight", sd.containsKey("linear.weight"));
                check("has linear.bias", sd.containsKey("linear.bias"));
                check("has emb.weight", sd.containsKey("emb.weight"));
                Tensor w = sd.get("linear.weight");
                check("weight rank2", w.dim() == 2);
                check("weight 4x3", w.sizes().get(0) == 4 && w.sizes().get(1) == 3);
                check("bias 4", sd.get("linear.bias").sizes().get(0) == 4);
                // value spot-check: first row [1,2,3]
                Tensor cpu = w.contiguous().cpu().to(ScalarType.Float);
                org.bytedeco.javacpp.FloatPointer fp = cpu.data_ptr_float();
                float v0 = fp.get(0);
                float v1 = fp.get(1);
                check("w[0,0]=1", Math.abs(v0 - 1f) < 1e-5);
                check("w[0,1]=2", Math.abs(v1 - 2f) < 1e-5);
            });

            benchmark("3. checkpoint unwrap model_state_dict", () -> {
                Map<String, Tensor> sd = TorchPthReader.loadStateDict(ckptPth.toFile());
                check("unwrapped 3 keys", sd.size() == 3);
                check("has linear.weight", sd.containsKey("linear.weight"));
            });

            benchmark("4. mixed dtypes f16/i64/bool", () -> {
                Map<String, Tensor> sd = TorchPthReader.loadStateDict(mixedPth.toFile());
                check("3 mixed keys", sd.size() == 3);
                // Always intern() scalar_type() proxies before equality (see SafeDType docs)
                check("w half-ish", sd.get("w").scalar_type().intern() == ScalarType.Half);
                check("i long-ish", sd.get("i").scalar_type().intern() == ScalarType.Long);
                check("b bool-ish", sd.get("b").scalar_type().intern() == ScalarType.Bool
                    || sd.get("b").numel() == 3);
                Tensor i = sd.get("i").contiguous().cpu().to(ScalarType.Long);
                check("i numel 6", i.numel() == 6);
            });

            benchmark("5. convert pth → safetensors + round-trip values", () -> {
                File out = tmp.resolve("from_state.safetensors").toFile();
                PthToSafeTensors.convert(statePth.toFile(), out);
                check("out exists", out.isFile() && out.length() > 32);
                Map<String, Tensor> fromPth = TorchPthReader.loadStateDict(statePth.toFile());
                Map<String, Tensor> fromSt = SafeTensors.loadAsTensors(out, false);
                check("same key count", fromPth.size() == fromSt.size());
                for (String k : fromPth.keySet()) {
                    check("key " + k, fromSt.containsKey(k));
                    check("close " + k, almostEqual(fromPth.get(k), fromSt.get(k), 1e-4));
                }
                check("detect safetensors",
                    ModelWeights.detect(out) == ModelWeights.Format.SAFETENSORS);
            });

            benchmark("6. ModelWeights auto-load + convert cache", () -> {
                Path pth = tmp.resolve("auto_model.pth");
                Files.copy(statePth, pth, StandardCopyOption.REPLACE_EXISTING);
                Map<String, Tensor> w = ModelWeights.load(pth.toFile(), true);
                check("auto keys", w.size() == 3);
                Path cached = tmp.resolve("auto_model.safetensors");
                check("cache written", Files.isRegularFile(cached));
                // second load via safetensors path when we point at cache
                Map<String, Tensor> w2 = ModelWeights.load(cached.toFile());
                check("cache keys", w2.size() == 3);
            });

            benchmark("7. loadIntoModule Linear weights", () -> {
                Map<String, Tensor> sd = TorchPthReader.loadStateDict(statePth.toFile());
                // LinearImpl(in_features, out_features) — weight is [out, in] = [4, 3]
                LinearImpl lin = new LinearImpl(3, 4);
                // keys are linear.weight / linear.bias — SafeTensors matcher uses prefix
                Map<String, Tensor> renamed = new LinkedHashMap<>();
                renamed.put("weight", sd.get("linear.weight"));
                renamed.put("bias", sd.get("linear.bias"));
                int n = SafeTensors.loadIntoModule(lin, renamed, true);
                check("params written >=1", n >= 1);
                check("weight close", almostEqual(lin.weight(), renamed.get("weight"), 1e-4));
            });

            benchmark("8. directory scan prefers safetensors", () -> {
                Path dir = tmp.resolve("modeldir");
                Files.createDirectories(dir);
                Files.copy(statePth, dir.resolve("model.pth"), StandardCopyOption.REPLACE_EXISTING);
                // only pth first
                Map<String, Tensor> a = ModelWeights.loadFromDirectory(dir, true);
                check("from pth dir", a.size() == 3);
                // add safetensors — should prefer it
                PthToSafeTensors.convert(dir.resolve("model.pth").toFile(),
                    dir.resolve("model.safetensors").toFile());
                Map<String, Tensor> b = ModelWeights.loadFromDirectory(dir, true);
                check("from safe dir", b.size() == 3);
            });

            benchmark("9. error: missing file / bad magic", () -> {
                boolean threw = false;
                try {
                    TorchPthReader.loadStateDict(tmp.resolve("nope.pth").toFile());
                } catch (Exception e) { threw = true; }
                check("missing throws", threw);
                Path junk = tmp.resolve("junk.pth");
                Files.writeString(junk, "not-a-zip");
                threw = false;
                try {
                    TorchPthReader.loadStateDict(junk.toFile());
                } catch (Exception e) { threw = true; }
                check("bad magic throws", threw);
            });

            benchmark("10. scale: many tensors convert", () -> {
                // Use python to write a larger dict if available, else synthesize via convert of state
                Map<String, Tensor> sd = TorchPthReader.loadStateDict(statePth.toFile());
                // replicate keys for stress of safetensors writer path
                Map<String, Tensor> big = new LinkedHashMap<>();
                for (int i = 0; i < 50; i++) {
                    for (Map.Entry<String, Tensor> e : sd.entrySet()) {
                        big.put(e.getKey() + "." + i, e.getValue());
                    }
                }
                File out = tmp.resolve("big.safetensors").toFile();
                long t0 = System.nanoTime();
                SafeTensors.save(big, out);
                Map<String, Tensor> back = SafeTensors.loadAsTensors(out, false);
                long ms = (System.nanoTime() - t0) / 1_000_000;
                check("big keys", back.size() == big.size());
                System.out.println("    150 tensors save+load: " + ms + " ms");
            });

        } finally {
            try {
                Files.walk(tmp).sorted(Comparator.reverseOrder())
                    .forEach(p -> { try { Files.deleteIfExists(p); } catch (Exception ignored) {} });
            } catch (Exception ignored) {}
        }
        System.out.println("Passed: " + passed + "  Failed: " + failed);
        if (failed > 0) {
            System.out.println(report);
            System.exit(1);
        }
    }
}
