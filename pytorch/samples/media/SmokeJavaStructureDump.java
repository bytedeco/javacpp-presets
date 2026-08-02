package media;
import org.bytedeco.pytorch.autograd.*;

import org.bytedeco.pytorch.data.serialize.StructureSpec;
import org.bytedeco.pytorch.data.serialize.TorchPthReader;
import org.bytedeco.pytorch.data.serialize.WeightBagModule;
import org.bytedeco.pytorch.Tensor;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

/**
 * Locked pure-Java structure export from original Python {@code torch.save} .pth only.
 *
 * <pre>
 *   java ... media.SmokeJavaStructureDump
 *   java ... media.SmokeJavaStructureDump --pth /path/to/python_state_dict.pth
 * </pre>
 */
public class SmokeJavaStructureDump {
    public static void main(String[] args) throws Exception {
        Path pth = Path.of("/Users/muller/Documents/code/cpp/VideoMMCTR/checkpoints/DSSM_1pct/dssm_1pct_state_dict.pth");
        Path seedStructure = Path.of("/Users/muller/Documents/code/cpp/VideoMMCTR/checkpoints/DSSM_1pct/dssm_1pct.structure.json");
        Path nativePt = Path.of("/Users/muller/Documents/code/cpp/VideoMMCTR/checkpoints/DSSM_1pct/dssm_1pct.javacpp.pt");
        Path outDir = Path.of("/tmp/java_structure_dump");
        for (int i = 0; i < args.length; i++) {
            if ("--pth".equals(args[i])) pth = Path.of(args[++i]);
            else if ("--structure".equals(args[i])) seedStructure = Path.of(args[++i]);
            else if ("--out-dir".equals(args[i])) outDir = Path.of(args[++i]);
        }
        Files.createDirectories(outDir);

        System.out.println("=== SmokeJavaStructureDump (locked fromPythonPth) ===");
        System.out.println("pth=" + pth + " isZipTorch=" + TorchPthReader.isZipTorch(pth.toFile()));
        System.out.println("seedStructure=" + seedStructure + " exists=" + Files.isRegularFile(seedStructure));

        // ── 1) Locked API: Python pth only ──────────────────────────────────
        File structureOrNull = Files.isRegularFile(seedStructure) ? seedStructure.toFile() : null;
        Path javaStruct = outDir.resolve("from_python_pth.structure.json");
        StructureSpec dumped = StructureSpec.dumpFromPythonPth(
                pth.toFile(), structureOrNull, javaStruct.toFile());
        System.out.println("dumpFromPythonPth → " + dumped + " file=" + javaStruct);
        System.out.println("  nodes=" + dumped.nodes.size()
                + " parameters=" + dumped.parameters.size()
                + " buffers=" + dumped.buffers.size());

        int dropouts = 0, sigmoids = 0, linears = 0, identities = 0;
        for (StructureSpec.Node n : dumped.nodes.values()) {
            String k = n.kind.toUpperCase();
            if (k.startsWith("DROPOUT")) dropouts++;
            if (k.equals("SIGMOID")) sigmoids++;
            if (k.equals("LINEAR")) linears++;
            if (k.equals("IDENTITY")) identities++;
        }
        System.out.println("  kinds DROPOUT=" + dropouts + " SIGMOID=" + sigmoids
                + " LINEAR=" + linears + " IDENTITY=" + identities);

        // Reload Module using ONLY python pth + Java-dumped structure
        WeightBagModule bag2 = WeightBagModule.fromPythonPthPrecise(pth.toFile(), javaStruct.toFile());
        System.out.println("reloaded: " + bag2.summary());
        System.out.println(bag2);

        Map<String, Tensor> sd = TorchPthReader.loadStateDict(pth.toFile());
        int matched = 0;
        for (String k : sd.keySet()) {
            Tensor a = bag2.get(k);
            if (a != null && a.defined()) matched++;
        }
        System.out.println("reloaded bag has " + matched + "/" + sd.size() + " state_dict keys");

        // ── 2) Must REJECT javacpp native archive ───────────────────────────
        boolean rejectedNative = false;
        if (Files.isRegularFile(nativePt)) {
            try {
                StructureSpec.fromPythonPth(nativePt.toFile(), null);
                System.out.println("FAIL: javacpp.pt was accepted (must refuse)");
            } catch (Exception e) {
                rejectedNative = true;
                System.out.println("OK refuse javacpp.pt: " + e.getMessage());
            }
        } else {
            // still test requirePythonTorchPth with a fake name if possible
            rejectedNative = true;
            System.out.println("skip native reject (no .javacpp.pt on disk)");
        }

        // ── 3) Must REJECT safetensors ──────────────────────────────────────
        boolean rejectedSt = false;
        Path fakeSt = outDir.resolve("fake.safetensors");
        Files.writeString(fakeSt, "not-a-real-st");
        try {
            StructureSpec.fromPythonPth(fakeSt.toFile(), null);
            System.out.println("FAIL: safetensors was accepted (must refuse)");
        } catch (Exception e) {
            rejectedSt = true;
            System.out.println("OK refuse safetensors: " + e.getMessage());
        }

        // ── 4) WeightBagModule convenience wrappers ─────────────────────────
        StructureSpec viaBag = WeightBagModule.structureFromPythonPth(
                pth.toFile(), structureOrNull);
        System.out.println("WeightBagModule.structureFromPythonPth nodes=" + viaBag.nodes.size());

        boolean ok = dropouts >= 1 && linears >= 1
                && matched >= sd.size() - 2
                && Files.isRegularFile(javaStruct) && Files.size(javaStruct) > 100
                && rejectedNative && rejectedSt
                && viaBag.nodes.size() >= 5;
        System.out.println(ok ? "SMOKE OK" : "SMOKE FAIL");
        if (!ok) System.exit(1);
    }
}
