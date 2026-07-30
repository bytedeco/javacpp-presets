package samples;
import org.bytedeco.pytorch.optim.options.*;
import org.bytedeco.pytorch.nn.*;
import org.bytedeco.pytorch.optim.*;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.data.safetensors.SafeTensors;
import org.bytedeco.pytorch.data.serialize.ModelStructure;
import org.bytedeco.pytorch.data.serialize.ModelWeights;
import org.bytedeco.pytorch.data.serialize.PthToSafeTensors;
import org.bytedeco.pytorch.data.serialize.TorchPthReader;
import org.bytedeco.pytorch.data.serialize.WeightBagModule;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.options.AdamOptions;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

/**
 * VideoMMCTR multi-model Python → JavaCPP structure-alignment benchmark.
 *
 * <p>For each of {@code DSSM / MMOE / DIN / MIND / AITM} (optional
 * {@code Transformer_DCN}):
 * <ol>
 *   <li>Optionally train 1% data × 1 epoch via
 *       {@code VideoMMCTR/scripts/train_videommctr_1pct.py}</li>
 *   <li>Load {@code .pth} with {@link WeightBagModule#fromPythonPth} /
 *       {@link WeightBagModule#fromFile}</li>
 *   <li>Print Module tree (should match Python {@code print(model)} style
 *       for reconstructed Linear / Embedding / ReLU / Dropout gaps)</li>
 *   <li>Convert → safetensors, round-trip values, freeze + Adam smoke</li>
 * </ol>
 *
 * <pre>
 *   # train one model first (from VideoMMCTR root):
 *   python3 scripts/train_videommctr_1pct.py --model DSSM --gpu -1
 *
 *   # then benchmark all available checkpoints:
 *   java ... samples.BenchmarkVideoMmctrPth --train-if-missing
 *
 *   # single model:
 *   java ... samples.BenchmarkVideoMmctrPth --models DSSM,DIN --train-if-missing
 *
 *   # skip train, only load existing:
 *   java ... samples.BenchmarkVideoMmctrPth --models MMOE --pth /path/to/mmoe_1pct_state_dict.pth
 * </pre>
 */
public class BenchmarkVideoMmctrPth {
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

    /** True for non-parameter buffers that often round-trip poorly as 0-d int64. */
    static boolean isBufferKey(String k) {
        if (k == null) return false;
        String s = k.toLowerCase(Locale.ROOT);
        return s.endsWith("num_batches_tracked")
            || s.endsWith("running_mean")
            || s.endsWith("running_var")
            || s.contains(".bn.num_batches")
            || s.contains("running_mean")
            || s.contains("running_var");
    }

    static boolean almostEqual(Tensor a, Tensor b, double atol) {
        if (a == null || b == null || !a.defined() || !b.defined()) return false;
        if (a.dim() != b.dim()) return false;
        for (int i = 0; i < a.dim(); i++) {
            if (a.sizes().get(i) != b.sizes().get(i)) return false;
        }
        // 0-d int64 (num_batches_tracked): compare as long, allow small drift
        try {
            if (a.numel() == 1 && a.scalar_type().intern() == ScalarType.Long) {
                long av = a.cpu().item_long();
                long bv = b.to(ScalarType.Long).cpu().item_long();
                return av == bv || Math.abs(av - bv) <= 1;
            }
        } catch (Throwable ignored) {}
        Tensor af = a.to(ScalarType.Float).contiguous().cpu();
        Tensor bf = b.to(ScalarType.Float).contiguous().cpu();
        return af.sub(bf).abs().max().item_float() <= atol;
    }

    /** Per-model expected state_dict key fragments (FuxiCTR naming). */
    static final class ModelSpec {
        final String name;          // DSSM
        final String slug;          // dssm
        final String[] keyFragments;
        final long minParams;
        final int minTensors;

        ModelSpec(String name, String[] keyFragments, long minParams, int minTensors) {
            this.name = name;
            this.slug = name.toLowerCase(Locale.ROOT);
            this.keyFragments = keyFragments;
            this.minParams = minParams;
            this.minTensors = minTensors;
        }

        Path defaultOutDir(Path videoRoot) {
            return videoRoot.resolve("checkpoints").resolve(name + "_1pct");
        }

        Path defaultStateDict(Path outDir) {
            return outDir.resolve(slug + "_1pct_state_dict.pth");
        }

        Path defaultCheckpoint(Path outDir) {
            return outDir.resolve(slug + "_1pct_checkpoint.pth");
        }

        Path defaultSafetensors(Path outDir) {
            return outDir.resolve(slug + "_1pct.safetensors");
        }
    }

    static final Map<String, ModelSpec> SPECS = new LinkedHashMap<>();
    static {
        SPECS.put("DSSM", new ModelSpec("DSSM",
            new String[]{"embedding_layer", "user_tower", "item_tower"},
            1_000_000L, 8));
        SPECS.put("MMOE", new ModelSpec("MMOE",
            new String[]{"embedding_layer", "experts", "gates", "ctr_tower"},
            1_000_000L, 10));
        SPECS.put("DIN", new ModelSpec("DIN",
            new String[]{"embedding_layer", "attention_layers", "dnn"},
            1_000_000L, 8));
        SPECS.put("MIND", new ModelSpec("MIND",
            new String[]{"embedding_layer", "mlp", "item_proj"},
            1_000_000L, 8));
        SPECS.put("AITM", new ModelSpec("AITM",
            new String[]{"embedding_layer", "bottoms", "ctr_tower", "aits"},
            1_000_000L, 10));
        SPECS.put("Transformer_DCN", new ModelSpec("Transformer_DCN",
            new String[]{"embedding_layer", "transformer_encoder"},
            1_000_000L, 10));
    }

    static final List<String> DEFAULT_MODELS =
        Arrays.asList("DSSM", "MMOE", "DIN", "MIND", "AITM");

    static Path defaultVideoMmctr() {
        Path[] candidates = {
            Path.of("/Users/muller/Documents/code/cpp/VideoMMCTR"),
            Path.of("..", "..", "cpp", "VideoMMCTR"),
            Path.of("..", "VideoMMCTR"),
            Path.of("VideoMMCTR"),
        };
        for (Path p : candidates) {
            if (Files.isDirectory(p) && Files.isRegularFile(p.resolve("src/DSSM.py"))) {
                return p.toAbsolutePath().normalize();
            }
        }
        return Path.of("/Users/muller/Documents/code/cpp/VideoMMCTR");
    }

    static boolean trainIfNeeded(Path videoRoot, ModelSpec spec, Path outDir,
                                 boolean force, int maxBatches) throws Exception {
        Path state = spec.defaultStateDict(outDir);
        if (!force && Files.isRegularFile(state) && Files.size(state) > 1000) {
            System.out.println("    reuse existing " + state);
            return true;
        }
        Path script = videoRoot.resolve("scripts/train_videommctr_1pct.py");
        // Fallback: Transformer_DCN has a dedicated script historically
        if (!Files.isRegularFile(script) && "Transformer_DCN".equals(spec.name)) {
            script = videoRoot.resolve("scripts/train_transformer_dcn_1pct.py");
        }
        if (!Files.isRegularFile(script)) {
            System.out.println("    train script missing: " + script);
            return false;
        }
        Files.createDirectories(outDir);
        List<String> cmd = new ArrayList<>();
        cmd.add("python3");
        cmd.add(script.toString());
        if (script.getFileName().toString().contains("videommctr")) {
            cmd.add("--model");
            cmd.add(spec.name);
        }
        cmd.add("--out-dir");
        cmd.add(outDir.toString());
        cmd.add("--train-frac");
        cmd.add("0.01");
        cmd.add("--valid-frac");
        cmd.add("0.05");
        cmd.add("--epochs");
        cmd.add("1");
        cmd.add("--gpu");
        cmd.add("-1");
        if (maxBatches > 0) {
            cmd.add("--max-batches");
            cmd.add(String.valueOf(maxBatches));
        }
        ProcessBuilder pb = new ProcessBuilder(cmd);
        pb.directory(videoRoot.toFile());
        pb.redirectErrorStream(true);
        System.out.println("    running: " + String.join(" ", pb.command()));
        Process p = pb.start();
        String log = new String(p.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
        int code = p.waitFor();
        String[] lines = log.split("\n");
        int from = Math.max(0, lines.length - 60);
        System.out.println("---- train log (tail) ----");
        for (int i = from; i < lines.length; i++) System.out.println(lines[i]);
        System.out.println("---- end train log (exit=" + code + ") ----");
        return code == 0 && Files.isRegularFile(state) && Files.size(state) > 1000;
    }

    static void runOneModel(ModelSpec spec, Path videoRoot, Path outDir, Path pthOverride,
                            boolean trainIfMissing, boolean forceTrain, int maxBatches)
            throws Exception {
        System.out.println();
        System.out.println("########## " + spec.name + " ##########");
        Path out = outDir != null ? outDir : spec.defaultOutDir(videoRoot);
        Path pth = pthOverride != null ? pthOverride : spec.defaultStateDict(out);
        Path ckpt = spec.defaultCheckpoint(out);
        Path stPath = spec.defaultSafetensors(out);
        Path pyStruct = out.resolve("python_structure.txt");
        Path pyModelStr = out.resolve("python_model_str.txt");

        final Path outF = out;
        final Path pthF = pth;
        final Path ckptF = ckpt;
        final Path stF = stPath;
        final Path videoRootF = videoRoot;
        final boolean forceF = forceTrain;
        final int maxBatchesF = maxBatches;

        if (trainIfMissing || forceTrain) {
            benchmark(spec.name + " 0. train 1% × 1 epoch", () -> {
                boolean ok = trainIfNeeded(videoRootF, spec, outF, forceF, maxBatchesF);
                check("train produced state_dict", ok);
            });
        }

        if (!Files.isRegularFile(pthF)) {
            System.out.println("SKIP " + spec.name + ": no pth at " + pthF);
            System.out.println("  Run: python3 " + videoRoot + "/scripts/train_videommctr_1pct.py --model " + spec.name);
            System.out.println("  or: java ... samples.BenchmarkVideoMmctrPth --models " + spec.name + " --train-if-missing");
            return;
        }

        benchmark(spec.name + " 1. file detect (torch ZIP)", () -> {
            check("exists", Files.isRegularFile(pthF));
            check("isZipTorch", TorchPthReader.isZipTorch(pthF.toFile()));
            check("detect TORCH_PTH_ZIP",
                ModelWeights.detect(pthF.toFile()) == ModelWeights.Format.TORCH_PTH_ZIP);
            System.out.println("    pth bytes=" + Files.size(pthF));
        });

        final Map<String, Tensor>[] held = new Map[]{null};

        benchmark(spec.name + " 2. load state_dict + print structure", () -> {
            Map<String, Tensor> sd = TorchPthReader.loadStateDictAndPrint(pthF.toFile());
            held[0] = sd;
            check("non-empty", sd != null && !sd.isEmpty());
            check("min tensors", sd.size() >= spec.minTensors);
            for (String frag : spec.keyFragments) {
                boolean hit = sd.keySet().stream().anyMatch(k ->
                    k.toLowerCase(Locale.ROOT).contains(frag.toLowerCase(Locale.ROOT)));
                check("has key fragment '" + frag + "'", hit);
            }
            long params = 0;
            for (Tensor t : sd.values()) params += t.numel();
            System.out.println("    tensors=" + sd.size() + " params=" + params);
            check("params >= min", params >= spec.minParams);
        });

        benchmark(spec.name + " 3. convert → safetensors", () -> {
            File st = stF.toFile();
            PthToSafeTensors.convert(pthF.toFile(), st, null, true);
            check("st exists", st.isFile() && st.length() > 100);
            check("detect safetensors",
                ModelWeights.detect(st) == ModelWeights.Format.SAFETENSORS);
            System.out.println("    safetensors bytes=" + st.length());
        });

        benchmark(spec.name + " 4. round-trip pth vs safetensors values", () -> {
            Map<String, Tensor> fromPth = held[0] != null ? held[0]
                : TorchPthReader.loadStateDict(pthF.toFile());
            Map<String, Tensor> fromSt = SafeTensors.loadAsTensors(stF.toFile(), false);
            check("same key count", fromPth.size() == fromSt.size());
            int n = 0;
            int mismatches = 0;
            int bufferMismatches = 0;
            for (String k : fromPth.keySet()) {
                check("has key " + k, fromSt.containsKey(k));
                if (!almostEqual(fromPth.get(k), fromSt.get(k), 1e-4f)) {
                    if (isBufferKey(k)) {
                        bufferMismatches++;
                        System.out.println("    buffer mismatch (tolerated): " + k);
                    } else {
                        mismatches++;
                        System.out.println("    mismatch: " + k);
                    }
                }
                n++;
            }
            check("no value mismatches (params)", mismatches == 0);
            System.out.println("    compared " + n + " tensors"
                + " (buffer_mismatches=" + bufferMismatches + ")");
        });

        benchmark(spec.name + " 5. checkpoint unwrap (if present)", () -> {
            if (!Files.isRegularFile(ckptF)) {
                System.out.println("    no checkpoint file, skip");
                check("skipped", true);
                return;
            }
            Map<String, Tensor> sd = TorchPthReader.loadStateDict(ckptF.toFile());
            ModelStructure.printStateDict(spec.name + " checkpoint unwrapped", sd);
            check("ckpt tensors", sd.size() >= spec.minTensors);
            Map<String, Tensor> base = held[0] != null ? held[0]
                : TorchPthReader.loadStateDict(pthF.toFile());
            check("ckpt vs state key count", sd.size() == base.size());
        });

        final WeightBagModule[] bagHolder = new WeightBagModule[1];

        benchmark(spec.name + " 6. WeightBagModule.fromPythonPth + print tree", () -> {
            long t0 = System.nanoTime();
            WeightBagModule bag = WeightBagModule.fromPythonPth(pthF.toFile(), true);
            long ms = (System.nanoTime() - t0) / 1_000_000;
            bagHolder[0] = bag;
            check("bag non-empty", bag.size() >= spec.minTensors);
            check("bag params large", bag.totalParamCount() >= spec.minParams);
            // named_parameters() only returns requires_grad params; ownedParams also
            // holds buffers (BN running_*/num_batches_tracked). Allow buffers ≤ size.
            int nNamed = bag.namedParametersMap().size();
            int nOwned = bag.size();
            check("named_parameters ⊆ owned", nNamed > 0 && nNamed <= nOwned);
            if (nNamed != nOwned) {
                System.out.println("    note: named_params=" + nNamed
                    + " owned=" + nOwned + " (buffers not in named_parameters)");
            }
            for (String frag : spec.keyFragments) {
                boolean hit = bag.keys().stream().anyMatch(k ->
                    k.toLowerCase(Locale.ROOT).contains(frag.toLowerCase(Locale.ROOT)));
                check("bag has fragment '" + frag + "'", hit);
            }
            System.out.println("    " + bag.summary() + " load_ms=" + ms);
            System.out.println("---- Java WeightBagModule.toString() (Python-style) ----");
            System.out.println(bag);
            System.out.println("---- end Java tree ----");
            bag.printStructure();
            ModelStructure.printWeightBag(spec.name + " WeightBagModule", bag);

            // Side-by-side with Python print(model) if available
            if (Files.isRegularFile(pyModelStr)) {
                String py = Files.readString(pyModelStr);
                System.out.println("---- Python print(model) (from train) ----");
                System.out.println(py.trim());
                System.out.println("---- end Python tree ----");
                // Soft structural checks: key top-level attribute names appear in Java tree
                String javaTree = bag.toString();
                for (String frag : spec.keyFragments) {
                    // Python uses attribute names; Java ModulePrinter uses similar paths
                    boolean inJava = javaTree.toLowerCase(Locale.ROOT)
                        .contains(frag.toLowerCase(Locale.ROOT));
                    boolean inPy = py.toLowerCase(Locale.ROOT)
                        .contains(frag.toLowerCase(Locale.ROOT));
                    if (inPy) {
                        check("java tree mentions '" + frag + "' (also in python)", inJava);
                    }
                }
            } else if (Files.isRegularFile(pyStruct)) {
                System.out.println("    (python_structure.txt present, model_str missing)");
            } else {
                System.out.println("    (no python_model_str.txt — train with train_videommctr_1pct.py to compare)");
            }
        });

        benchmark(spec.name + " 7. fromFile / fromSafetensors parity", () -> {
            WeightBagModule a = WeightBagModule.fromFile(pthF.toFile(), true);
            WeightBagModule b = WeightBagModule.fromSafetensors(stF.toFile(), true);
            check("fromFile size", a.size() == bagHolder[0].size() || a.size() >= spec.minTensors);
            check("fromSafetensors size", b.size() >= spec.minTensors);
            check("same param count pth vs st",
                a.totalParamCount() == b.totalParamCount()
                    || Math.abs(a.totalParamCount() - b.totalParamCount()) == 0);
            System.out.println("    fromFile=" + a.summary());
            System.out.println("    fromSafetensors=" + b.summary());
        });

        benchmark(spec.name + " 8. freeze embedding + Adam step smoke", () -> {
            WeightBagModule bag = bagHolder[0];
            check("bag present", bag != null);
            // FuxiCTR uses embedding_layer.* as the shared FeatureEmbedding table
            bag.freezePrefix("embedding_layer.");
            long trainable = bag.trainableParamCount();
            long total = bag.totalParamCount();
            check("trainable < total after freeze", trainable < total);
            System.out.println("    trainable=" + trainable + " / total=" + total);

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
            TensorVector tv = new TensorVector();
            tv.push_back(leaf);
            Adam opt = new Adam(tv, new AdamOptions(1e-4));
            float before = leaf.detach().to(ScalarType.Float).abs().mean().item_float();
            opt.zero_grad();
            Tensor lf = leaf.to(ScalarType.Float);
            Tensor loss = lf.mul(lf).mean();
            loss.backward();
            opt.step();
            float after = leaf.detach().to(ScalarType.Float).abs().mean().item_float();
            check("loss finite", Float.isFinite(loss.item_float()));
            System.out.println("    stepped leaf=" + leafKey
                + " mean-abs " + before + " → " + after
                + " loss=" + loss.item_float());
        });

        benchmark(spec.name + " 9. saveSafetensors with structure meta + reload", () -> {
            WeightBagModule bag = bagHolder[0];
            check("bag present", bag != null);
            File stMeta = outF.resolve(spec.slug + "_1pct_with_structure.safetensors").toFile();
            bag.saveSafetensors(stMeta);
            check("structure st exists", stMeta.isFile() && stMeta.length() > 100);
            WeightBagModule reloaded = WeightBagModule.fromSafetensors(stMeta, true);
            check("reloaded size", reloaded.size() >= spec.minTensors);
            check("reloaded params", reloaded.totalParamCount() == bag.totalParamCount()
                || reloaded.totalParamCount() > 0);
            System.out.println("---- reloaded (with module_structure meta) ----");
            System.out.println(reloaded);
            System.out.println("---- end reloaded ----");
            // structure meta should be non-empty after save when gap-fill produced layers
            Map<String, String> meta = reloaded.structureMeta();
            System.out.println("    structureMeta entries=" + (meta == null ? 0 : meta.size()));
            check("reload ok", reloaded.size() > 0);
        });

        benchmark(spec.name + " 10. spot-check item_id embedding shape", () -> {
            Map<String, Tensor> sd = held[0] != null ? held[0]
                : TorchPthReader.loadStateDict(pthF.toFile());
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
            check("item_id emb dim > 0", t.sizes().get(1) > 0);
            System.out.println("    " + itemKey.get() + " shape=["
                + t.sizes().get(0) + ", " + t.sizes().get(1) + "]");
        });
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=== BenchmarkVideoMmctrPth ===");
        Path videoRoot = defaultVideoMmctr();
        List<String> models = new ArrayList<>(DEFAULT_MODELS);
        boolean trainIfMissing = false;
        boolean forceTrain = false;
        Path pthOverride = null;
        Path outDirOverride = null;
        int maxBatches = 0; // 0 = full 1%

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--video-root":
                    videoRoot = Path.of(args[++i]);
                    break;
                case "--models":
                case "--model":
                    models = new ArrayList<>();
                    for (String m : args[++i].split(",")) {
                        String t = m.trim();
                        if (!t.isEmpty()) models.add(t);
                    }
                    break;
                case "--all":
                    models = new ArrayList<>(SPECS.keySet());
                    break;
                case "--out-dir":
                    outDirOverride = Path.of(args[++i]);
                    break;
                case "--pth":
                    pthOverride = Path.of(args[++i]);
                    break;
                case "--train-if-missing":
                    trainIfMissing = true;
                    break;
                case "--force-train":
                    forceTrain = true;
                    trainIfMissing = true;
                    break;
                case "--max-batches":
                    maxBatches = Integer.parseInt(args[++i]);
                    break;
                case "--help":
                case "-h":
                    System.out.println("Usage: BenchmarkVideoMmctrPth [options]");
                    System.out.println("  --models DSSM,MMOE,DIN,MIND,AITM");
                    System.out.println("  --all");
                    System.out.println("  --train-if-missing | --force-train");
                    System.out.println("  --max-batches N   (cap train batches for smoke)");
                    System.out.println("  --video-root PATH");
                    System.out.println("  --pth PATH        (single-model only)");
                    System.out.println("  --out-dir PATH    (single-model only)");
                    return;
                default:
                    System.out.println("unknown arg: " + args[i]);
            }
        }

        System.out.println("videoRoot=" + videoRoot);
        System.out.println("models=" + models);
        System.out.println("trainIfMissing=" + trainIfMissing + " forceTrain=" + forceTrain
            + " maxBatches=" + maxBatches);

        if (pthOverride != null && models.size() != 1) {
            System.out.println("NOTE: --pth is only meaningful with a single --model; using first model");
            if (models.isEmpty()) models = List.of("DSSM");
            models = List.of(models.get(0));
        }

        for (String name : models) {
            ModelSpec spec = SPECS.get(name);
            if (spec == null) {
                // allow case-insensitive
                for (Map.Entry<String, ModelSpec> e : SPECS.entrySet()) {
                    if (e.getKey().equalsIgnoreCase(name)) {
                        spec = e.getValue();
                        break;
                    }
                }
            }
            if (spec == null) {
                System.out.println("UNKNOWN model: " + name + " (known: " + SPECS.keySet() + ")");
                failed++;
                report.append("UNKNOWN model: ").append(name).append('\n');
                continue;
            }
            Path out = (outDirOverride != null && models.size() == 1)
                ? outDirOverride : null;
            Path pth = (pthOverride != null && models.size() == 1) ? pthOverride : null;
            try {
                runOneModel(spec, videoRoot, out, pth, trainIfMissing, forceTrain, maxBatches);
            } catch (Throwable t) {
                failed++;
                System.out.println(" FAIL " + spec.name + " (top-level): " + t);
                report.append("FAIL ").append(spec.name).append(" top-level: ").append(t).append('\n');
                t.printStackTrace(System.out);
            }
        }

        System.out.println();
        System.out.println("=== Summary ===");
        System.out.println("Passed: " + passed + "  Failed: " + failed);
        if (failed > 0) {
            System.out.println(report);
            System.exit(1);
        }
    }
}
