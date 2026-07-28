package org.bytedeco.pytorch.data.serialize;
import org.bytedeco.pytorch.optim.options.*;
import org.bytedeco.pytorch.optim.*;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.data.safetensors.SafeTensors;
import org.bytedeco.pytorch.nn.Module;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Auto-detect weight file format and load as {@code Map&lt;String, Tensor&gt;}.
 *
 * <p>Supported:
 * <ul>
 *   <li>{@code .safetensors} — native JavaCPP path ({@link SafeTensors})</li>
 *   <li>{@code .pth} / {@code .pt} — Python {@code torch.save} ZIP → {@link TorchPthReader}</li>
 *   <li>magic-byte sniff when extension is missing/wrong</li>
 * </ul>
 *
 * <p>Optional: convert Python checkpoints to safetensors next to the source so
 * subsequent loads skip pickle entirely.
 */
public final class ModelWeights {
    public enum Format { SAFETENSORS, TORCH_PTH_ZIP, UNKNOWN }

    private ModelWeights() {}

    public static Format detect(File file) throws IOException {
        if (file == null || !file.isFile()) return Format.UNKNOWN;
        String name = file.getName().toLowerCase(Locale.ROOT);
        if (name.endsWith(".safetensors")) return Format.SAFETENSORS;
        if (name.endsWith(".pth") || name.endsWith(".pt") || name.endsWith(".bin")) {
            if (TorchPthReader.isZipTorch(file)) return Format.TORCH_PTH_ZIP;
            // .bin is often raw pytorch_model.bin (also zip torch) or something else
            if (isSafetensorsMagic(file)) return Format.SAFETENSORS;
            return Format.UNKNOWN;
        }
        if (isSafetensorsMagic(file)) return Format.SAFETENSORS;
        if (TorchPthReader.isZipTorch(file)) return Format.TORCH_PTH_ZIP;
        return Format.UNKNOWN;
    }

    public static Format detect(Path path) throws IOException {
        return detect(path.toFile());
    }

    /**
     * Load tensors from a single weight file (auto format).
     */
    public static Map<String, Tensor> load(File file) throws IOException {
        return load(file, true);
    }

    /**
     * @param convertPthToSafe when true and input is a ZIP .pth, also write a
     *                         sibling {@code .safetensors} for faster reloads
     */
    public static Map<String, Tensor> load(File file, boolean convertPthToSafe) throws IOException {
        Format fmt = detect(file);
        switch (fmt) {
            case SAFETENSORS:
                return SafeTensors.loadAsTensors(file, true);
            case TORCH_PTH_ZIP: {
                Map<String, Tensor> sd = TorchPthReader.loadStateDict(file);
                if (convertPthToSafe && !sd.isEmpty()) {
                    File out = PthToSafeTensors.defaultOutput(file);
                    if (!out.exists() || out.lastModified() < file.lastModified()) {
                        try {
                            PthToSafeTensors.convert(file, out);
                        } catch (Exception ignored) {
                            // conversion is best-effort cache; load still returns tensors
                        }
                    }
                }
                return sd;
            }
            default:
                throw new IOException("Unrecognized weight format: " + file
                    + " (expected .safetensors or torch ZIP .pth/.pt)");
        }
    }

    public static Map<String, Tensor> load(Path path) throws IOException {
        return load(path.toFile());
    }

    public static Map<String, Tensor> load(String path) throws IOException {
        return load(new File(path));
    }

    /**
     * Load and inject into a module. Auto-detects format.
     *
     * @return parameters written
     */
    public static int loadIntoModule(Module module, File file, boolean strict) throws IOException {
        Map<String, Tensor> w = load(file, true);
        return SafeTensors.loadIntoModule(module, w, strict);
    }

    /**
     * Auto-detect format, load tensors, and build a trainable typed
     * {@link WeightBagModule}: nested hierarchy + real Linear/Embedding/…
     * leaves with hyperparameters inferred from shapes and names.
     *
     * <p>This is the primary path for "arbitrary safetensors / .pth → Module
     * so we can fine-tune": no architecture class required, structure and
     * layer names match the Python state-dict.
     *
     * <pre>{@code
     *   WeightBagModule bag = ModelWeights.toModule("model.safetensors");
     *   // or: ModelWeights.toModule("model.pth")
     *   bag.freezePrefix("embedding_layer.");
     *   Adam opt = new Adam(bag.parameters(), new AdamOptions(1e-3));
     *   bag.saveSafetensors(new File("finetuned.safetensors"));
     * }</pre>
     */
    public static WeightBagModule toModule(File file) throws IOException {
        return toModule(file, true);
    }

    public static WeightBagModule toModule(File file, boolean requiresGrad) throws IOException {
        // Delegate to WeightBagModule loaders (structure meta + Sequential gap-fill)
        return WeightBagModule.fromFile(file, requiresGrad);
    }

    public static WeightBagModule toModule(Path path) throws IOException {
        return toModule(path.toFile());
    }

    public static WeightBagModule toModule(String path) throws IOException {
        return toModule(new File(path));
    }

    /**
     * Scan directory for weights (prefer safetensors) and build a bag Module.
     */
    public static WeightBagModule toModuleFromDirectory(Path dir) throws IOException {
        return toModuleFromDirectory(dir, true);
    }

    public static WeightBagModule toModuleFromDirectory(Path dir, boolean requiresGrad)
            throws IOException {
        Map<String, Tensor> w = loadFromDirectory(dir, true);
        return WeightBagModule.fromTyped(w, requiresGrad);
    }

    /**
     * Scan a directory for weight files in preference order:
     * model.safetensors → *.safetensors shards → model.pth / pytorch_model.bin / *.pth.
     */
    public static Map<String, Tensor> loadFromDirectory(Path dir) throws IOException {
        return loadFromDirectory(dir, true);
    }

    public static Map<String, Tensor> loadFromDirectory(Path dir, boolean convertPthToSafe)
            throws IOException {
        if (dir == null || !Files.isDirectory(dir)) {
            throw new IOException("not a directory: " + dir);
        }
        // Prefer safetensors
        Path single = dir.resolve("model.safetensors");
        if (Files.isRegularFile(single)) {
            return SafeTensors.loadAsTensors(single.toFile(), true);
        }
        List<Path> safes = new ArrayList<>();
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(dir, "*.safetensors")) {
            for (Path p : ds) {
                String n = p.getFileName().toString();
                if (n.endsWith(".partial")) continue;
                safes.add(p);
            }
        }
        if (!safes.isEmpty()) {
            safes.sort(Path::compareTo);
            Map<String, Tensor> all = new LinkedHashMap<>();
            for (Path p : safes) {
                all.putAll(SafeTensors.loadAsTensors(p.toFile(), true));
            }
            return all;
        }
        // Fall back to .pth / .pt / pytorch_model.bin
        for (String name : new String[]{
            "model.pth", "pytorch_model.bin", "model.pt", "checkpoint.pth", "weights.pth"
        }) {
            Path p = dir.resolve(name);
            if (Files.isRegularFile(p)) {
                return load(p.toFile(), convertPthToSafe);
            }
        }
        List<Path> pths = new ArrayList<>();
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(dir)) {
            for (Path p : ds) {
                String n = p.getFileName().toString().toLowerCase(Locale.ROOT);
                if (n.endsWith(".pth") || n.endsWith(".pt") || n.endsWith(".bin")) {
                    pths.add(p);
                }
            }
        }
        if (!pths.isEmpty()) {
            pths.sort(Path::compareTo);
            // merge all (unusual but useful for multi-file dumps)
            Map<String, Tensor> all = new LinkedHashMap<>();
            for (Path p : pths) {
                try {
                    all.putAll(load(p.toFile(), convertPthToSafe));
                } catch (IOException ignored) { /* skip non-torch bins */ }
            }
            if (!all.isEmpty()) return all;
        }
        throw new IOException("No loadable weights (.safetensors / .pth / .pt) in " + dir);
    }

    private static boolean isSafetensorsMagic(File file) throws IOException {
        // safetensors starts with u64 little-endian header length — not a stable magic,
        // but files are never ZIP PK\x03\x04. Heuristic: extension or non-zip + readable header.
        if (TorchPthReader.isZipTorch(file)) return false;
        if (file.length() < 16) return false;
        try (InputStream in = Files.newInputStream(file.toPath())) {
            byte[] b = in.readNBytes(8);
            if (b.length < 8) return false;
            long headerLen = 0;
            for (int i = 0; i < 8; i++) headerLen |= ((long) (b[i] & 0xFF)) << (8 * i);
            // reasonable header: 2 bytes .. 100 MB
            return headerLen >= 2 && headerLen < 100_000_000L && headerLen + 8 < file.length();
        }
    }
}
