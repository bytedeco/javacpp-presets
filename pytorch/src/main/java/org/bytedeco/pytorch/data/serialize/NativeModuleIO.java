package org.bytedeco.pytorch.data.serialize;
import org.bytedeco.pytorch.serialize.*;

import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.serialize.InputArchive;
import org.bytedeco.pytorch.serialize.OutputArchive;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Objects;

/**
 * Native JavaCPP / LibTorch {@code torch::save} / {@code torch::load} helpers
 * for a pre-built {@link Module} tree.
 *
 * <p>This is the true “JavaCPP .pt” path (flatbuffer/pickle archive via
 * {@link OutputArchive} / {@link InputArchive}), <b>not</b> Python ZIP pickle
 * and <b>not</b> safetensors.
 *
 * <pre>{@code
 *   // After precise rebuild from Python pth + structure.json:
 *   WeightBagModule bag = WeightBagModule.fromPythonPthPrecise(pth, structure);
 *   NativeModuleIO.save(bag, new File("model.javacpp.pt"));
 *
 *   // Later: rebuild empty architecture from the same structure, then load:
 *   WeightBagModule bag2 = StructureModuleBuilder.buildEmpty(spec);
 *   NativeModuleIO.load(bag2, new File("model.javacpp.pt"));
 * }</pre>
 *
 * <p>LibTorch {@code Module.load} requires a pre-built architecture that matches
 * the saved tree — structure cannot be invented from the archive alone.
 */
public final class NativeModuleIO {
    private NativeModuleIO() {}

    /** Serialize {@code module} to a native LibTorch file via {@link OutputArchive}. */
    public static void save(Module module, File file) throws IOException {
        Objects.requireNonNull(module, "module");
        Objects.requireNonNull(file, "file");
        File parent = file.getParentFile();
        if (parent != null && !parent.isDirectory() && !parent.mkdirs()) {
            throw new IOException("cannot create parent dir: " + parent);
        }
        try {
            OutputArchive archive = new OutputArchive();
            module.save(archive);
            archive.save_to(file.getAbsolutePath());
        } catch (Throwable t) {
            throw new IOException("NativeModuleIO.save failed for " + file + ": " + t, t);
        }
    }

    public static void save(Module module, Path path) throws IOException {
        save(module, path.toFile());
    }

    public static void save(Module module, String path) throws IOException {
        save(module, new File(path));
    }

    /**
     * Deserialize weights into a <b>pre-built</b> {@code module} of matching
     * architecture via {@link InputArchive}.
     */
    public static void load(Module module, File file) throws IOException {
        Objects.requireNonNull(module, "module");
        Objects.requireNonNull(file, "file");
        if (!file.isFile()) throw new IOException("not a file: " + file);
        try {
            InputArchive archive = new InputArchive();
            archive.load_from(file.getAbsolutePath());
            module.load(archive);
        } catch (Throwable t) {
            throw new IOException("NativeModuleIO.load failed for " + file + ": " + t, t);
        }
    }

    public static void load(Module module, Path path) throws IOException {
        load(module, path.toFile());
    }

    public static void load(Module module, String path) throws IOException {
        load(module, new File(path));
    }
}
