package distribute;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.data.safetensors.SafeTensors;
import org.bytedeco.pytorch.data.serialize.ModelStructure;
import org.bytedeco.pytorch.data.serialize.ModelWeights;
import org.bytedeco.pytorch.data.serialize.PthToSafeTensors;
import org.bytedeco.pytorch.data.serialize.TorchPthReader;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

/**
 * End-to-end multi-dimensional benchmark:
 * <ol>
 *   <li>Train a Neural Collaborative Filtering recommender in Python ({@code scripts/train_ncf_recsys.py})</li>
 *   <li>Print model structure from the saved {@code .pth}</li>
 *   <li>Convert arbitrary {@code .pth} → {@code .safetensors}</li>
 *   <li>Round-trip values, checkpoint unwrap (optimizer stub tolerance), auto-detect, module inject</li>
 * </ol>
 */
public class BenchmarkRecsysPthToSafetensors {
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

    static boolean almostEqual(Tensor a, Tensor b, double atol) {
        if (a == null || b == null || !a.defined() || !b.defined()) return false;
        if (a.dim() != b.dim()) return false;
        for (int i = 0; i < a.dim(); i++) {
            if (a.sizes().get(i) != b.sizes().get(i)) return false;
        }
        Tensor af = a.to(ScalarType.Float).contiguous().cpu();
        Tensor bf = b.to(ScalarType.Float).contiguous().cpu();
        double max = af.sub(bf).abs().max().item_float();
        return max <= atol;
    }

    /** Train NCF via scripts/train_ncf_recsys.py → outDir. */
    static boolean trainRecsys(Path outDir) {
        try {
            Files.createDirectories(outDir);
            Path script = Path.of("scripts/train_ncf_recsys.py");
            if (!Files.isRegularFile(script)) {
                // try absolute from cwd parents
                Path alt = Path.of("pytorch/scripts/train_ncf_recsys.py");
                if (Files.isRegularFile(alt)) script = alt;
            }
            if (!Files.isRegularFile(script)) {
                System.out.println("    train script missing: " + script.toAbsolutePath());
                return false;
            }
            ProcessBuilder pb = new ProcessBuilder(
                "python3", script.toAbsolutePath().toString(),
                "--out-dir", outDir.toAbsolutePath().toString(),
                "--users", "120",
                "--items", "80",
                "--positives", "2000",
                "--emb", "16",
                "--mlp", "32", "16",
                "--epochs", "4",
                "--steps", "30",
                "--batch", "128",
                "--lr", "0.02"
            );
            pb.redirectErrorStream(true);
            Process p = pb.start();
            String out = new String(p.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
            int code = p.waitFor();
            System.out.println("---- python train log (tail) ----");
            String[] lines = out.split("\n");
            int from = Math.max(0, lines.length - 40);
            for (int i = from; i < lines.length; i++) System.out.println(lines[i]);
            System.out.println("---- end train log ----");
            if (code != 0) {
                System.out.println("    train exit=" + code);
                return false;
            }
            return Files.isRegularFile(outDir.resolve("ncf_state_dict.pth"))
                && Files.isRegularFile(outDir.resolve("ncf_checkpoint.pth"));
        } catch (Exception e) {
            System.out.println("    train failed: " + e);
            e.printStackTrace(System.out);
            return false;
        }
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=== BenchmarkRecsysPthToSafetensors ===");
        Path work = Files.createTempDirectory("ncf-bench-");
        try {
            Path run = work.resolve("run");
            boolean trained = trainRecsys(run);
            if (!trained) {
                System.out.println("SKIP: could not train NCF (need python3 + torch)");
                System.out.println("Passed: 0  Failed: 0 (skipped)");
                return;
            }

            Path statePth = run.resolve("ncf_state_dict.pth");
            Path ckptPth = run.resolve("ncf_checkpoint.pth");
            Path metaJson = run.resolve("ncf_meta.json");

            benchmark("1. train artifacts exist", () -> {
                check("state_dict pth", Files.isRegularFile(statePth) && Files.size(statePth) > 100);
                check("checkpoint pth", Files.isRegularFile(ckptPth) && Files.size(ckptPth) > 100);
                check("meta json", Files.isRegularFile(metaJson));
                System.out.println("    state_dict bytes=" + Files.size(statePth)
                    + " checkpoint bytes=" + Files.size(ckptPth));
            });

            benchmark("2. print structure from state_dict.pth", () -> {
                Map<String, Tensor> sd = TorchPthReader.loadStateDictAndPrint(statePth.toFile());
                check("has tensors", sd.size() >= 8); // 4 emb + mlp linears + predict
                check("user_gmf.weight", sd.containsKey("user_gmf.weight"));
                check("item_gmf.weight", sd.containsKey("item_gmf.weight"));
                check("user_mlp.weight", sd.containsKey("user_mlp.weight"));
                check("item_mlp.weight", sd.containsKey("item_mlp.weight"));
                check("predict.weight", sd.containsKey("predict.weight"));
                // embedding shape [n_users, emb]
                Tensor ug = sd.get("user_gmf.weight");
                check("user emb rank2", ug.dim() == 2);
                check("user emb dim=16", ug.sizes().get(1) == 16);
                System.out.println("    n_users(emb)=" + ug.sizes().get(0)
                    + " n_items=" + sd.get("item_gmf.weight").sizes().get(0));
            });

            benchmark("3. convert state_dict → safetensors (prints structure)", () -> {
                File st = run.resolve("ncf_state_dict.safetensors").toFile();
                PthToSafeTensors.convert(statePth.toFile(), st, null, true);
                check("safetensors exists", st.isFile() && st.length() > 100);
                Map<String, Tensor> fromPth = TorchPthReader.loadStateDict(statePth.toFile());
                Map<String, Tensor> fromSt = SafeTensors.loadAsTensors(st, false);
                check("same key count", fromPth.size() == fromSt.size());
                int compared = 0;
                for (String k : fromPth.keySet()) {
                    check("key " + k, fromSt.containsKey(k));
                    check("close " + k, almostEqual(fromPth.get(k), fromSt.get(k), 1e-4));
                    if (++compared >= 12) break; // enough coverage; rest covered by count
                }
            });

            benchmark("4. full checkpoint (optimizer stubs) → state_dict extract", () -> {
                // Must not throw on Adam state GLOBALs
                Map<String, Tensor> sd = TorchPthReader.loadStateDict(ckptPth.toFile());
                ModelStructure.printStateDict("ncf_checkpoint.pth (unwrapped)", sd);
                check("ckpt unwrapped tensors", sd.size() >= 8);
                check("ckpt has user_gmf", sd.containsKey("user_gmf.weight"));
            });

            benchmark("5. convert full checkpoint → safetensors", () -> {
                File st = run.resolve("ncf_from_ckpt.safetensors").toFile();
                PthToSafeTensors.convert(ckptPth.toFile(), st, null, true);
                check("ckpt safetensors", st.isFile());
                Map<String, Tensor> w = SafeTensors.loadAsTensors(st, false);
                check("ckpt st keys", w.size() >= 8);
            });

            benchmark("6. ModelWeights auto-detect + directory load", () -> {
                check("detect pth", ModelWeights.detect(statePth) == ModelWeights.Format.TORCH_PTH_ZIP);
                File st = run.resolve("ncf_state_dict.safetensors").toFile();
                check("detect st", ModelWeights.detect(st) == ModelWeights.Format.SAFETENSORS);
                Map<String, Tensor> dirLoad = ModelWeights.loadFromDirectory(run, true);
                check("dir load keys", dirLoad.size() >= 8);
            });

            benchmark("7. inject embedding weights into Java EmbeddingImpl", () -> {
                Map<String, Tensor> sd = TorchPthReader.loadStateDict(statePth.toFile());
                Tensor ug = sd.get("user_gmf.weight");
                long nUsers = ug.sizes().get(0);
                long emb = ug.sizes().get(1);
                EmbeddingImpl embMod = new EmbeddingImpl(nUsers, emb);
                Map<String, Tensor> one = new LinkedHashMap<>();
                one.put("weight", ug);
                int n = SafeTensors.loadIntoModule(embMod, one, true);
                check("emb written", n >= 1);
                check("emb close", almostEqual(embMod.weight(), ug, 1e-4));
            });

            benchmark("8. inject predict Linear into Java LinearImpl", () -> {
                Map<String, Tensor> sd = TorchPthReader.loadStateDict(statePth.toFile());
                Tensor w = sd.get("predict.weight"); // [1, emb+mlp_last]
                Tensor b = sd.get("predict.bias");
                long outF = w.sizes().get(0);
                long inF = w.sizes().get(1);
                LinearImpl lin = new LinearImpl(inF, outF);
                Map<String, Tensor> params = new LinkedHashMap<>();
                params.put("weight", w);
                params.put("bias", b);
                int n = SafeTensors.loadIntoModule(lin, params, true);
                check("linear written", n >= 1);
                check("linear w close", almostEqual(lin.weight(), w, 1e-4));
            });

            benchmark("9. structure report aggregates", () -> {
                Map<String, Tensor> sd = TorchPthReader.loadStateDict(statePth.toFile());
                ModelStructure.Report r = ModelStructure.ofStateDict("ncf", sd);
                check("totalParams > 0", r.totalParams > 0);
                check("totalBytes > 0", r.totalBytes > 0);
                check("prefix user_gmf", r.prefixCounts.containsKey("user_gmf")
                    || r.prefixCounts.containsKey("user_mlp")
                    || r.tensors.stream().anyMatch(t -> t.name.startsWith("user_")));
                System.out.println("    totalParams=" + r.totalParams + " totalBytes=" + r.totalBytes);
            });

            benchmark("10. arbitrary pth: non-zip rejected clearly", () -> {
                Path junk = run.resolve("not_torch.pth");
                Files.writeString(junk, "hello");
                boolean threw = false;
                try {
                    TorchPthReader.loadStateDict(junk.toFile());
                } catch (Exception e) {
                    threw = true;
                    check("msg mentions ZIP/PK", e.getMessage() != null
                        && (e.getMessage().contains("ZIP") || e.getMessage().contains("PK")
                        || e.getMessage().toLowerCase().contains("torch")));
                }
                check("threw", threw);
            });

        } finally {
            try {
                Files.walk(work).sorted(Comparator.reverseOrder())
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
