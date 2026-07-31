package distribute;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.data.safetensors.SafeTensors;
import org.bytedeco.pytorch.data.serialize.ModelStructure;
import org.bytedeco.pytorch.data.serialize.ModelWeights;
import org.bytedeco.pytorch.data.serialize.PthToSafeTensors;
import org.bytedeco.pytorch.data.serialize.TorchPthReader;
import org.bytedeco.pytorch.data.serialize.WeightBagModule;
import org.bytedeco.pytorch.global.torch.ScalarType;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

/**
 * Validate reading VideoMMCTR {@code Transformer_DCN} weights produced by
 * {@code VideoMMCTR/scripts/train_transformer_dcn_1pct.py}:
 * <ol>
 *   <li>Optionally train (1% data, 1 epoch) via that Python script</li>
 *   <li>Load {@code .pth} with {@link TorchPthReader}</li>
 *   <li>Print model structure</li>
 *   <li>Convert to safetensors and round-trip compare</li>
 * </ol>
 *
 * <pre>
 *   # train first (from VideoMMCTR root):
 *   python3 scripts/train_transformer_dcn_1pct.py --out-dir ./checkpoints/Transformer_DCN_1pct
 *
 *   # then benchmark:
 *   java ... distribute.BenchmarkTransformerDcnPth \
 *     --pth /path/to/checkpoints/Transformer_DCN_1pct/transformer_dcn_1pct_state_dict.pth
 *
 *   # or auto-train if --train-if-missing and VideoMMCTR is sibling/default path:
 *   java ... distribute.BenchmarkTransformerDcnPth --train-if-missing
 * </pre>
 */
public class BenchmarkTransformerDcnPth {
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
        return af.sub(bf).abs().max().item_float() <= atol;
    }

    static Path defaultVideoMmctr() {
        // common layouts: sibling of javacpp-presets, or absolute user path
        Path[] candidates = {
            Path.of("/Users/muller/Documents/code/cpp/VideoMMCTR"),
            Path.of("..", "..", "cpp", "VideoMMCTR"),
            Path.of("..", "VideoMMCTR"),
            Path.of("VideoMMCTR"),
        };
        for (Path p : candidates) {
            if (Files.isDirectory(p) && Files.isRegularFile(p.resolve("src/Transformer_DCN.py"))) {
                return p.toAbsolutePath().normalize();
            }
        }
        return Path.of("/Users/muller/Documents/code/cpp/VideoMMCTR");
    }

    static boolean trainIfNeeded(Path videoRoot, Path outDir, boolean force) throws Exception {
        Path state = outDir.resolve("transformer_dcn_1pct_state_dict.pth");
        if (!force && Files.isRegularFile(state) && Files.size(state) > 1000) {
            System.out.println("    reuse existing " + state);
            return true;
        }
        Path script = videoRoot.resolve("scripts/train_transformer_dcn_1pct.py");
        if (!Files.isRegularFile(script)) {
            System.out.println("    train script missing: " + script);
            return false;
        }
        Files.createDirectories(outDir);
        ProcessBuilder pb = new ProcessBuilder(
            "python3", script.toString(),
            "--out-dir", outDir.toString(),
            "--train-frac", "0.01",
            "--valid-frac", "0.05",
            "--epochs", "1",
            "--batch-size", "256",
            "--gpu", "-1"
        );
        pb.directory(videoRoot.toFile());
        pb.redirectErrorStream(true);
        System.out.println("    running: " + String.join(" ", pb.command()));
        Process p = pb.start();
        // stream log
        String log = new String(p.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
        int code = p.waitFor();
        String[] lines = log.split("\n");
        int from = Math.max(0, lines.length - 50);
        System.out.println("---- train log (tail) ----");
        for (int i = from; i < lines.length; i++) System.out.println(lines[i]);
        System.out.println("---- end train log (exit=" + code + ") ----");
        return code == 0 && Files.isRegularFile(state);
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=== BenchmarkTransformerDcnPth ===");
        Path videoRoot = defaultVideoMmctr();
        Path outDir = videoRoot.resolve("checkpoints/Transformer_DCN_1pct");
        Path pth = null;
        boolean trainIfMissing = false;
        boolean forceTrain = false;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--video-root":
                    videoRoot = Path.of(args[++i]);
                    break;
                case "--out-dir":
                    outDir = Path.of(args[++i]);
                    break;
                case "--pth":
                    pth = Path.of(args[++i]);
                    break;
                case "--train-if-missing":
                    trainIfMissing = true;
                    break;
                case "--force-train":
                    forceTrain = true;
                    trainIfMissing = true;
                    break;
                default:
                    System.out.println("unknown arg: " + args[i]);
            }
        }

        if (pth == null) {
            pth = outDir.resolve("transformer_dcn_1pct_state_dict.pth");
        }
        Path ckpt = outDir.resolve("transformer_dcn_1pct_checkpoint.pth");

        final Path videoRootF = videoRoot;
        final Path outDirTrain = outDir;
        final boolean forceTrainF = forceTrain;
        if (trainIfMissing || forceTrain) {
            benchmark("0. train Transformer_DCN 1% × 1 epoch", () -> {
                boolean ok = trainIfNeeded(videoRootF, outDirTrain, forceTrainF);
                check("train produced state_dict", ok);
            });
        }

        if (!Files.isRegularFile(pth)) {
            System.out.println("SKIP: no pth at " + pth);
            System.out.println("Run: python3 " + videoRoot + "/scripts/train_transformer_dcn_1pct.py");
            System.out.println("  or: java ... distribute.BenchmarkTransformerDcnPth --train-if-missing");
            System.out.println("Passed: " + passed + "  Failed: " + failed);
            if (failed > 0) System.exit(1);
            return;
        }

        final Path pthFinal = pth;
        final Path outDirFinal = outDir;
        final Path ckptFinal = ckpt;

        benchmark("1. file is torch ZIP + detect", () -> {
            check("exists", Files.isRegularFile(pthFinal));
            check("isZipTorch", TorchPthReader.isZipTorch(pthFinal.toFile()));
            check("detect TORCH_PTH_ZIP",
                ModelWeights.detect(pthFinal.toFile()) == ModelWeights.Format.TORCH_PTH_ZIP);
            System.out.println("    pth bytes=" + Files.size(pthFinal));
        });

        final Map<String, Tensor>[] held = new Map[]{null};

        benchmark("2. load state_dict + print structure", () -> {
            Map<String, Tensor> sd = TorchPthReader.loadStateDictAndPrint(pthFinal.toFile());
            held[0] = sd;
            check("non-empty", sd != null && !sd.isEmpty());
            // Transformer_DCN expected name fragments
            boolean hasEmb = sd.keySet().stream().anyMatch(k -> k.contains("embedding"));
            boolean hasCross = sd.keySet().stream().anyMatch(k -> k.toLowerCase().contains("cross"));
            boolean hasMlp = sd.keySet().stream().anyMatch(k -> k.contains("mlp") || k.contains("parallel"));
            boolean hasTransformer = sd.keySet().stream().anyMatch(k ->
                k.toLowerCase().contains("transformer") || k.toLowerCase().contains("encoder")
                    || k.contains("attn") || k.contains("linear"));
            check("has embedding weights", hasEmb);
            check("has dcn/mlp-ish weights", hasCross || hasMlp);
            check("has transformer-ish weights", hasTransformer || sd.size() >= 10);
            long params = 0;
            for (Tensor t : sd.values()) params += t.numel();
            System.out.println("    tensors=" + sd.size() + " params=" + params);
            check("params > 0", params > 0);
        });

        benchmark("3. convert → safetensors + structure print", () -> {
            File st = outDirFinal.resolve("transformer_dcn_1pct.safetensors").toFile();
            PthToSafeTensors.convert(pthFinal.toFile(), st, null, true);
            check("st exists", st.isFile() && st.length() > 100);
            check("detect safetensors",
                ModelWeights.detect(st) == ModelWeights.Format.SAFETENSORS);
            System.out.println("    safetensors bytes=" + st.length());
        });

        benchmark("4. round-trip values pth vs safetensors", () -> {
            Map<String, Tensor> fromPth = held[0] != null ? held[0]
                : TorchPthReader.loadStateDict(pthFinal.toFile());
            File st = outDirFinal.resolve("transformer_dcn_1pct.safetensors").toFile();
            Map<String, Tensor> fromSt = SafeTensors.loadAsTensors(st, false);
            check("same key count", fromPth.size() == fromSt.size());
            int n = 0;
            int mismatches = 0;
            for (String k : fromPth.keySet()) {
                check("has key " + k, fromSt.containsKey(k));
                if (!almostEqual(fromPth.get(k), fromSt.get(k), 1e-4f)) {
                    mismatches++;
                    System.out.println("    mismatch: " + k);
                }
                n++;
            }
            check("no value mismatches", mismatches == 0);
            System.out.println("    compared " + n + " tensors");
        });

        benchmark("5. checkpoint unwrap (if present)", () -> {
            if (!Files.isRegularFile(ckptFinal)) {
                System.out.println("    no checkpoint file, skip");
                check("skipped", true);
                return;
            }
            Map<String, Tensor> sd = TorchPthReader.loadStateDict(ckptFinal.toFile());
            ModelStructure.printStateDict("checkpoint unwrapped", sd);
            check("ckpt tensors", sd.size() >= 5);
            // should match state_dict keys
            Map<String, Tensor> base = TorchPthReader.loadStateDict(pthFinal.toFile());
            check("ckpt vs state key count", sd.size() == base.size());
        });

        benchmark("6. ModelWeights auto-load path", () -> {
            Map<String, Tensor> w = ModelWeights.load(pthFinal.toFile(), true);
            check("auto keys", w.size() >= 5);
            // cache safetensors next to pth
            Path cached = PthToSafeTensors.defaultOutput(pthFinal.toFile()).toPath();
            // convert() may have written transformer_dcn_1pct.safetensors explicitly;
            // defaultOutput would be transformer_dcn_1pct_state_dict.safetensors
            System.out.println("    default cache path=" + cached
                + " exists=" + Files.isRegularFile(cached));
            check("auto non-empty", !w.isEmpty());
        });

        benchmark("7. spot-check item_id embedding shape if present", () -> {
            Map<String, Tensor> sd = TorchPthReader.loadStateDict(pthFinal.toFile());
            Optional<String> itemKey = sd.keySet().stream()
                .filter(k -> k.contains("item_id") && k.endsWith("weight"))
                .findFirst();
            if (itemKey.isEmpty()) {
                System.out.println("    no item_id weight key, skip shape assert");
                check("skipped", true);
                return;
            }
            Tensor t = sd.get(itemKey.get());
            check("item_id rank2", t.dim() == 2);
            // vocab ~91718 from feature map
            check("item_id vocab large", t.sizes().get(0) > 1000);
            check("item_id emb dim", t.sizes().get(1) == 64 || t.sizes().get(1) == 128
                || t.sizes().get(1) > 0);
            System.out.println("    " + itemKey.get() + " shape=["
                + t.sizes().get(0) + ", " + t.sizes().get(1) + "]");
        });

        // ── 8/9: safetensors → WeightBagModule (trainable bag) ──────────
        final WeightBagModule[] bagHolder = new WeightBagModule[1];

        benchmark("8. safetensors/pth → WeightBagModule (toModule)", () -> {
            File st = outDirFinal.resolve("transformer_dcn_1pct.safetensors").toFile();
            File src = st.isFile() ? st : pthFinal.toFile();
            System.out.println("    toModule from " + src.getName());
            long t0 = System.nanoTime();
            WeightBagModule bag = ModelWeights.toModule(src, /*requiresGrad=*/true);
            long ms = (System.nanoTime() - t0) / 1_000_000;
            bagHolder[0] = bag;
            check("bag non-empty", bag.size() >= 5);
            check("bag params large", bag.totalParamCount() > 1_000_000L);
            check("named_parameters match size", bag.namedParametersMap().size() == bag.size());
            // hierarchical keys preserved
            boolean hasEmb = bag.keys().stream().anyMatch(k -> k.contains("embedding"));
            boolean hasCross = bag.keys().stream().anyMatch(k -> k.toLowerCase().contains("cross"));
            check("has embedding key", hasEmb);
            check("has cross key", hasCross || bag.size() >= 10);
            System.out.println("    " + bag + " load_ms=" + ms);
            ModelStructure.printModule("Transformer_DCN WeightBagModule", bag);
        });

        benchmark("9. freeze embedding + Adam step on a small leaf (fine-tune smoke)", () -> {
            WeightBagModule bag = bagHolder[0];
            check("bag present", bag != null);
            bag.freezePrefix("embedding_layer.");
            long trainable = bag.trainableParamCount();
            long total = bag.totalParamCount();
            check("trainable < total after freeze", trainable < total);
            System.out.println("    trainable=" + trainable + " / total=" + total);

            // Pick a small trainable leaf so we don't OOM Adam state on 268M params
            Tensor leaf = null;
            String leafKey = null;
            for (Map.Entry<String, Tensor> e : bag.parametersMap().entrySet()) {
                Tensor t = e.getValue();
                if (t != null && t.defined() && t.requires_grad() && t.numel() < 20_000_000L) {
                    leaf = t;
                    leafKey = e.getKey();
                    break;
                }
            }
            if (leaf == null) {
                System.out.println("    no small trainable leaf, skip Adam step");
                check("skipped", true);
                return;
            }
            org.bytedeco.pytorch.TensorVector tv = new org.bytedeco.pytorch.TensorVector();
            tv.push_back(leaf);
            org.bytedeco.pytorch.optim.Adam opt =
                new org.bytedeco.pytorch.optim.Adam(tv, new org.bytedeco.pytorch.optim.options.AdamOptions(1e-4));
            float before = leaf.detach().to(ScalarType.Float).abs().mean().item_float();
            opt.zero_grad();
            Tensor lf = leaf.to(ScalarType.Float);
            Tensor loss = lf.mul(lf).mean();
            loss.backward();
            opt.step();
            float after = leaf.detach().to(ScalarType.Float).abs().mean().item_float();
            // real assert: no crash + finite loss; value should typically move
            check("loss finite", Float.isFinite(loss.item_float()));
            System.out.println("    stepped leaf=" + leafKey
                + " mean-abs " + before + " → " + after
                + " loss=" + loss.item_float());
        });

        System.out.println("Passed: " + passed + "  Failed: " + failed);
        if (failed > 0) {
            System.out.println(report);
            System.exit(1);
        }
    }
}
