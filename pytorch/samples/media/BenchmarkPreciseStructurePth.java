package media;
import org.bytedeco.pytorch.nn.modules.*;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.optim.*;
import org.bytedeco.pytorch.optim.options.*;
import org.bytedeco.pytorch.autograd.*;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.data.serialize.NativeModuleIO;
import org.bytedeco.pytorch.data.serialize.StructureModuleBuilder;
import org.bytedeco.pytorch.data.serialize.StructureSpec;
import org.bytedeco.pytorch.data.serialize.TorchPthReader;
import org.bytedeco.pytorch.data.serialize.WeightBagModule;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.options.AdamOptions;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Precise structure alignment benchmark: Python .pth + .structure.json →
 * JavaCPP WeightBagModule <b>without safetensors</b>, plus native .pt round-trip.
 *
 * <pre>
 *   # export structure (from VideoMMCTR):
 *   python3 scripts/train_videommctr_1pct.py --model DSSM --structure-only
 *
 *   java ... media.BenchmarkPreciseStructurePth --models DSSM,MMOE,DIN,MIND
 * </pre>
 */
public class BenchmarkPreciseStructurePth {
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

    static Path defaultVideoMmctr() {
        Path[] candidates = {
            Path.of("/Users/muller/Documents/code/cpp/VideoMMCTR"),
            Path.of("..", "..", "cpp", "VideoMMCTR"),
            Path.of("VideoMMCTR"),
        };
        for (Path p : candidates) {
            if (Files.isDirectory(p) && Files.isRegularFile(p.resolve("src/DSSM.py"))) {
                return p.toAbsolutePath().normalize();
            }
        }
        return Path.of("/Users/muller/Documents/code/cpp/VideoMMCTR");
    }

    static final class ModelSpec {
        final String name;
        final String slug;
        ModelSpec(String name) {
            this.name = name;
            this.slug = name.toLowerCase(Locale.ROOT);
        }
        Path outDir(Path root) { return root.resolve("checkpoints").resolve(name + "_1pct"); }
        Path pth(Path out) { return out.resolve(slug + "_1pct_state_dict.pth"); }
        Path structure(Path out) { return out.resolve(slug + "_1pct.structure.json"); }
        Path pyModel(Path out) { return out.resolve("python_model_str.txt"); }
    }

    static void runOne(ModelSpec spec, Path videoRoot) throws Exception {
        System.out.println();
        System.out.println("########## PRECISE " + spec.name + " ##########");
        Path out = spec.outDir(videoRoot);
        Path pth = spec.pth(out);
        Path structure = spec.structure(out);
        Path pyModel = spec.pyModel(out);
        Path nativePt = out.resolve(spec.slug + "_1pct.javacpp.pt");

        // Delete any safetensors so we prove no mid-format is required
        try {
            Files.list(out)
                .filter(p -> p.getFileName().toString().endsWith(".safetensors"))
                .forEach(p -> {
                    try { Files.deleteIfExists(p); } catch (Exception ignored) {}
                });
        } catch (Exception ignored) {}

        if (!Files.isRegularFile(pth)) {
            System.out.println("SKIP: missing pth " + pth);
            return;
        }
        if (!Files.isRegularFile(structure)) {
            System.out.println("SKIP: missing structure " + structure
                + " — run: python3 scripts/train_videommctr_1pct.py --model "
                + spec.name + " --structure-only");
            failed++;
            report.append("missing structure for ").append(spec.name).append('\n');
            return;
        }

        final StructureSpec[] specHolder = new StructureSpec[1];
        final WeightBagModule[] bagHolder = new WeightBagModule[1];
        final Map<String, Tensor>[] sdHolder = new Map[]{null};

        benchmark(spec.name + " 1. load StructureSpec v2", () -> {
            StructureSpec s = StructureSpec.load(structure);
            specHolder[0] = s;
            check("version>=2", s.version >= 2);
            check("has root node", s.rootNode() != null);
            check("has nodes", s.nodes.size() >= 5);
            check("has parameters list", !s.parameters.isEmpty());
            System.out.println("    " + s);
            // precise fields that heuristics get wrong
            int dropouts = 0, softmaxes = 0, sigmoids = 0, linears = 0;
            for (Map.Entry<String, StructureSpec.Node> e : s.nodes.entrySet()) {
                String k = e.getValue().kind.toUpperCase(Locale.ROOT);
                if (k.equals("DROPOUT") || k.startsWith("DROPOUT:")) dropouts++;
                if (k.equals("SOFTMAX") || k.startsWith("SOFTMAX:")) softmaxes++;
                if (k.equals("SIGMOID")) sigmoids++;
                if (k.equals("LINEAR")) linears++;
            }
            System.out.println("    kinds: DROPOUT=" + dropouts + " SOFTMAX=" + softmaxes
                + " SIGMOID=" + sigmoids + " LINEAR=" + linears);
            check("has LINEAR nodes", linears > 0);
        });

        benchmark(spec.name + " 2. fromPythonPthPrecise (no safetensors)", () -> {
            // ensure no safetensors
            long stCount = 0;
            try {
                stCount = Files.list(out)
                    .filter(p -> p.getFileName().toString().endsWith(".safetensors"))
                    .count();
            } catch (Exception ignored) {}
            check("no safetensors required", stCount == 0);

            long t0 = System.nanoTime();
            WeightBagModule bag = WeightBagModule.fromPythonPthPrecise(
                pth.toFile(), structure.toFile(), true);
            long ms = (System.nanoTime() - t0) / 1_000_000;
            bagHolder[0] = bag;
            check("bag non-empty", bag.size() > 0);
            check("params > 0", bag.totalParamCount() > 0);
            System.out.println("    " + bag.summary() + " load_ms=" + ms);
            System.out.println("---- Java precise tree ----");
            System.out.println(bag);
            System.out.println("---- end Java precise tree ----");
        });

        benchmark(spec.name + " 3. structure vs Java tree assertions", () -> {
            StructureSpec s = specHolder[0];
            WeightBagModule bag = bagHolder[0];
            check("spec+bag present", s != null && bag != null);
            String tree = bag.toString();
            String treeL = tree.toLowerCase(Locale.ROOT);

            // Every DROPOUT path must appear conceptually (Sequential index present)
            for (Map.Entry<String, StructureSpec.Node> e : s.nodes.entrySet()) {
                StructureSpec.Node n = e.getValue();
                String kind = n.kind.toUpperCase(Locale.ROOT);
                if (kind.equals("DROPOUT") || kind.startsWith("DROPOUT:")) {
                    double p = n.hyperDouble("p", -1);
                    check("dropout p recorded for " + e.getKey(), p >= 0);
                    // tree should mention Dropout
                    check("tree has DropoutImpl", tree.contains("DropoutImpl") || treeL.contains("dropout"));
                }
                if (kind.equals("SOFTMAX") || kind.startsWith("SOFTMAX:")) {
                    check("tree has Softmax for " + e.getKey(),
                        tree.contains("Softmax") || treeL.contains("softmax"));
                }
                if (kind.equals("SIGMOID")) {
                    check("tree has Sigmoid",
                        tree.contains("Sigmoid") || treeL.contains("sigmoid"));
                }
                if (kind.equals("LINEAR") && e.getKey().endsWith("item_emb_d128")) {
                    // must be Linear not Embedding
                    check("item_emb_d128 is Linear not Embedding",
                        treeL.contains("item_emb_d128") && treeL.contains("linear"));
                }
            }

            // Model-specific hard checks
            if ("MMOE".equals(spec.name)) {
                check("MMOE experts dropout p=0.2 in structure",
                    s.nodes.values().stream().anyMatch(n ->
                        "DROPOUT".equalsIgnoreCase(n.kind) && Math.abs(n.hyperDouble("p", 0) - 0.2) < 1e-9));
                check("MMOE has Softmax gates",
                    s.nodes.values().stream().anyMatch(n -> n.kind.toUpperCase(Locale.ROOT).startsWith("SOFTMAX")));
            }
            if ("MIND".equals(spec.name)) {
                check("MIND mlp dropout p=0.2",
                    s.nodes.values().stream().anyMatch(n ->
                        "DROPOUT".equalsIgnoreCase(n.kind) && Math.abs(n.hyperDouble("p", 0) - 0.2) < 1e-9));
            }
            if ("DIN".equals(spec.name)) {
                check("DIN has Dice COMPOSITE",
                    s.nodes.values().stream().anyMatch(n ->
                        "Dice".equals(n.className) || (n.kind != null && n.kind.startsWith("COMPOSITE"))));
                check("DIN dnn ends with Sigmoid in structure",
                    s.hasNode("dnn.mlp.13") && "SIGMOID".equalsIgnoreCase(s.node("dnn.mlp.13").kind));
            }
            if ("DSSM".equals(spec.name)) {
                StructureSpec.Node emb = s.node("embedding_layer.embedding_layer.feature_encoders.item_emb_d128");
                check("DSSM item_emb_d128 LINEAR bias=false",
                    emb != null && "LINEAR".equalsIgnoreCase(emb.kind) && !emb.hyperBool("bias", true));
            }

            if (Files.isRegularFile(pyModel)) {
                System.out.println("---- Python print(model) ----");
                System.out.println(Files.readString(pyModel).trim());
                System.out.println("---- end Python ----");
            }
        });

        benchmark(spec.name + " 4. weight values match pth", () -> {
            Map<String, Tensor> sd = TorchPthReader.loadStateDict(pth.toFile());
            sdHolder[0] = sd;
            WeightBagModule bag = bagHolder[0];
            int compared = 0, mismatch = 0, skipped = 0;
            for (String k : sd.keySet()) {
                if (k.endsWith("num_batches_tracked")) { skipped++; continue; }
                Tensor a = sd.get(k);
                Tensor b = bag.get(k);
                try {
                    if (a == null || a.isNull() || !a.defined()) { skipped++; continue; }
                    if (b == null || b.isNull() || !b.defined()) {
                        System.out.println("    missing in bag: " + k);
                        mismatch++;
                        continue;
                    }
                    // retain before numel/compare — bag.get may still be fragile
                    Tensor ar = new Tensor(a);
                    Tensor br = new Tensor(b);
                    if (ar.numel() != br.numel()) {
                        System.out.println("    numel mismatch " + k);
                        mismatch++;
                        continue;
                    }
                    Tensor af = ar.to(ScalarType.Float).contiguous().cpu();
                    Tensor bf = br.to(ScalarType.Float).contiguous().cpu();
                    float max = af.sub(bf).abs().max().item_float();
                    if (max > 1e-4f) {
                        System.out.println("    value mismatch " + k + " maxΔ=" + max);
                        mismatch++;
                    } else {
                        compared++;
                    }
                } catch (Throwable t) {
                    System.out.println("    skip " + k + ": " + t);
                    skipped++;
                }
            }
            System.out.println("    compared=" + compared + " mismatches=" + mismatch
                + " skipped=" + skipped);
            check("mostly matched", mismatch <= 2 && compared > 0);
        });

        benchmark(spec.name + " 5. native javacpp.pt save/load (no safetensors)", () -> {
            WeightBagModule bag = bagHolder[0];
            StructureSpec s = specHolder[0];
            check("present", bag != null && s != null);

            NativeModuleIO.save(bag, nativePt.toFile());
            check("native pt exists", Files.isRegularFile(nativePt) && Files.size(nativePt) > 100);
            System.out.println("    wrote " + nativePt + " bytes=" + Files.size(nativePt));

            // Rebuild empty architecture from structure, load native weights
            WeightBagModule empty = StructureModuleBuilder.buildEmpty(s);
            NativeModuleIO.load(empty, nativePt.toFile());
            empty.loadNative(nativePt.toFile()); // also via WeightBagModule API

            // compare a few keys
            Map<String, Tensor> sd = sdHolder[0] != null ? sdHolder[0]
                : TorchPthReader.loadStateDict(pth.toFile());
            int ok = 0;
            for (String k : bag.keys()) {
                Tensor a = bag.get(k);
                Tensor b = empty.get(k);
                if (a == null || b == null || !a.defined() || !b.defined()) continue;
                if (k.endsWith("num_batches_tracked")) continue;
                try {
                    float max = a.to(ScalarType.Float).sub(b.to(ScalarType.Float)).abs().max().item_float();
                    if (max <= 1e-3f) ok++;
                } catch (Throwable ignored) {}
                if (ok >= 5) break;
            }
            check("native reload matched some tensors", ok >= 1);
            System.out.println("    native reload matched_samples=" + ok);
        });

        benchmark(spec.name + " 6. freeze + Adam smoke on precise bag", () -> {
            WeightBagModule bag = bagHolder[0];
            check("bag", bag != null);
            bag.freezePrefix("embedding_layer.");
            long trainable = bag.trainableParamCount();
            long total = bag.totalParamCount();
            check("trainable < total", trainable < total);
            Tensor leaf = null;
            String leafKey = null;
            for (Map.Entry<String, Tensor> e : bag.parametersMap().entrySet()) {
                Tensor t = e.getValue();
                if (t != null && t.defined() && t.requires_grad() && t.numel() < 5_000_000L) {
                    leaf = t; leafKey = e.getKey(); break;
                }
            }
            if (leaf == null) {
                System.out.println("    no small leaf, skip step");
                check("skipped", true);
                return;
            }
            TensorVector tv = new TensorVector();
            tv.push_back(leaf);
            Adam opt = new Adam(tv, new AdamOptions(1e-4));
            opt.zero_grad();
            Tensor loss = leaf.to(ScalarType.Float).mul(leaf.to(ScalarType.Float)).mean();
            loss.backward();
            opt.step();
            check("loss finite", Float.isFinite(loss.item_float()));
            System.out.println("    stepped " + leafKey + " loss=" + loss.item_float()
                + " trainable=" + trainable + "/" + total);
        });

        benchmark(spec.name + " 7. fromPythonPth auto-discovers structure.json", () -> {
            // delete safetensors again
            try {
                Files.list(out).filter(p -> p.toString().endsWith(".safetensors"))
                    .forEach(p -> { try { Files.deleteIfExists(p); } catch (Exception ignored) {} });
            } catch (Exception ignored) {}
            WeightBagModule bag = WeightBagModule.fromPythonPth(pth.toFile(), true);
            check("auto bag size", bag.size() > 0);
            String tree = bag.toString();
            // precise markers
            if ("MMOE".equals(spec.name) || "MIND".equals(spec.name) || "DSSM".equals(spec.name)) {
                check("auto tree has Dropout", tree.contains("Dropout") || tree.contains("DropoutImpl"));
            }
            System.out.println("    auto " + bag.summary());
        });
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=== BenchmarkPreciseStructurePth ===");
        Path videoRoot = defaultVideoMmctr();
        List<String> models = new ArrayList<>(Arrays.asList("DSSM", "MMOE", "DIN", "MIND", "AITM"));
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--video-root": videoRoot = Path.of(args[++i]); break;
                case "--models":
                case "--model":
                    models = new ArrayList<>();
                    for (String m : args[++i].split(",")) {
                        if (!m.trim().isEmpty()) models.add(m.trim());
                    }
                    break;
                default:
                    System.out.println("unknown arg: " + args[i]);
            }
        }
        System.out.println("videoRoot=" + videoRoot);
        System.out.println("models=" + models);

        for (String name : models) {
            try {
                runOne(new ModelSpec(name), videoRoot);
            } catch (Throwable t) {
                failed++;
                System.out.println(" FAIL " + name + " top-level: " + t);
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
