package org.bytedeco.pytorch.data.serialize;
import org.bytedeco.pytorch.optim.options.*;
import org.bytedeco.pytorch.optim.*;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.data.safetensors.SafeTensors;
import org.bytedeco.pytorch.nn.Module;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Adapter: Python {@code torch.save} {@code .pth}/{@code .pt} → HuggingFace
 * {@code .safetensors}, so JavaCPP / LibTorch can load weights that CPython
 * pickles cannot feed into {@code torch::load}.
 *
 * <pre>
 *   // convert once
 *   File st = PthToSafeTensors.convert(new File("model.pth"));
 *   // or load tensors directly
 *   Map&lt;String, Tensor&gt; sd = PthToSafeTensors.loadAsTensors(new File("model.pth"));
 *   SafeTensors.loadIntoModule(module, sd, true);
 * </pre>
 */
public final class PthToSafeTensors {
    private PthToSafeTensors() {}

    /**
     * Convert {@code input.pth} → sibling {@code input.safetensors}
     * (or {@code out} if provided). Prints model structure. Returns the output file.
     */
    public static File convert(File input) throws IOException {
        return convert(input, null, null, true);
    }

    public static File convert(File input, File output) throws IOException {
        return convert(input, output, null, true);
    }

    /**
     * @param metadata optional safetensors header metadata (may be null)
     */
    public static File convert(File input, File output, Map<String, String> metadata) throws IOException {
        return convert(input, output, metadata, true);
    }

    /**
     * @param printStructure when true, print {@link ModelStructure} report of extracted tensors
     */
    public static File convert(File input, File output, Map<String, String> metadata,
                               boolean printStructure) throws IOException {
        if (input == null) throw new IllegalArgumentException("input required");
        File out = output != null ? output : defaultOutput(input);
        Map<String, Tensor> sd = loadAsTensors(input);
        if (sd.isEmpty()) {
            throw new IOException("no tensors extracted from " + input);
        }
        if (printStructure) {
            ModelStructure.printStateDict(input.getName() + " → " + out.getName(), sd);
        }
        Map<String, String> meta = metadata == null ? new LinkedHashMap<>() : new LinkedHashMap<>(metadata);
        meta.putIfAbsent("format", "pt");
        meta.putIfAbsent("converted_from", input.getName());
        meta.putIfAbsent("converted_by", "org.bytedeco.pytorch.data.serialize.PthToSafeTensors");
        meta.putIfAbsent("tensor_count", Integer.toString(sd.size()));
        Path parent = out.toPath().getParent();
        if (parent != null) Files.createDirectories(parent);
        SafeTensors.save(sd, out, meta);
        System.out.println("[PthToSafeTensors] wrote " + out.getAbsolutePath()
            + " (" + out.length() + " bytes, " + sd.size() + " tensors)");
        return out;
    }

    public static File convert(String inputPath) throws IOException {
        return convert(new File(inputPath));
    }

    public static File convert(String inputPath, String outputPath) throws IOException {
        return convert(new File(inputPath), outputPath == null ? null : new File(outputPath));
    }

    /** Load state-dict tensors from a Python .pth without writing safetensors. */
    public static Map<String, Tensor> loadAsTensors(File input) throws IOException {
        return TorchPthReader.loadStateDict(input);
    }

    public static Map<String, Tensor> loadAsTensors(String path) throws IOException {
        return loadAsTensors(new File(path));
    }

    /**
     * Convert if needed then inject into module via {@link SafeTensors#loadIntoModule}.
     * When {@code pthOrSafe} is already {@code .safetensors}, loads it directly.
     *
     * @return number of parameters written
     */
    public static int loadIntoModule(Module module, File pthOrSafe, boolean strict) throws IOException {
        if (module == null) throw new IllegalArgumentException("module required");
        Map<String, Tensor> weights = ModelWeights.load(pthOrSafe);
        return SafeTensors.loadIntoModule(module, weights, strict);
    }

    /**
     * Load arbitrary {@code .pth} / {@code .safetensors} as a trainable typed
     * {@link WeightBagModule}: hierarchy + real Linear/Embedding/… leaves
     * matching the Python state-dict names and hyperparameters.
     *
     * <pre>{@code
     *   WeightBagModule bag = PthToSafeTensors.toModule("model.pth");
     *   bag.freezePrefix("embedding_layer.");
     *   Adam opt = new Adam(bag.parameters(), new AdamOptions(1e-4));
     *   bag.saveSafetensors(new File("finetuned.safetensors"));
     * }</pre>
     */
    public static WeightBagModule toModule(File pthOrSafe) throws IOException {
        return ModelWeights.toModule(pthOrSafe, true);
    }

    public static WeightBagModule toModule(File pthOrSafe, boolean requiresGrad) throws IOException {
        return ModelWeights.toModule(pthOrSafe, requiresGrad);
    }

    public static WeightBagModule toModule(String path) throws IOException {
        return toModule(new File(path));
    }

    /** Default sibling path: {@code model.pth} → {@code model.safetensors}. */
    public static File defaultOutput(File input) {
        String name = input.getName();
        String base;
        if (name.endsWith(".pth")) base = name.substring(0, name.length() - 4);
        else if (name.endsWith(".pt")) base = name.substring(0, name.length() - 3);
        else if (name.endsWith(".bin")) base = name.substring(0, name.length() - 4);
        else base = name;
        File parent = input.getParentFile();
        return new File(parent, base + ".safetensors");
    }
}
