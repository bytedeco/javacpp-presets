/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.llm.llamafactory.model;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.llm.peft.LoraConfig;
import org.bytedeco.pytorch.llm.peft.PeftModel;
import org.bytedeco.pytorch.llm.transformers.CausalLM;
import org.bytedeco.pytorch.nn.Module;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.logging.Logger;

/**
 * Load / attach PEFT adapters from disk (HF PEFT layout:
 * {@code adapter_config.json} + {@code adapter_model.safetensors}).
 *
 * <p>Composes {@link PeftModel#fromPretrained}. Multi-adapter paths are loaded
 * sequentially; the last successful load is returned.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class AdapterLoader {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private static final Logger LOG = Logger.getLogger(AdapterLoader.class.getName());

    private AdapterLoader() {}

    /**
     * Load adapter(s) into {@code base}. {@code existing} may be non-null when
     * a fresh LoRA was already constructed — we still prefer disk weights.
     *
     * @param path single dir, or comma-separated multi-adapter paths
     */
    public static PeftModel loadInto(Module base, PeftModel existing, String path)
            throws IOException {
        Objects.requireNonNull(base, "base");
        if (path == null || path.isBlank()) {
            return existing;
        }
        List<String> paths = splitPaths(path);
        PeftModel current = existing;
        for (String p : paths) {
            File dir = new File(p.trim());
            if (!dir.exists()) {
                LOG.warning("Adapter path does not exist: " + dir.getAbsolutePath());
                continue;
            }
            if (dir.isFile()) {
                // allow pointing at adapter_model.safetensors directly
                dir = dir.getParentFile() == null ? dir : dir.getParentFile();
            }
            try {
                PeftModel loaded = PeftModel.fromPretrained(base, dir);
                if (base instanceof CausalLM causal && loaded != null) {
                    // ensure forward graph sees adapters
                    try {
                        LoraConfig cfg = loadedConfig(loaded);
                        if (cfg != null) {
                            causal.attachLora(cfg);
                        }
                    } catch (Throwable t) {
                        LOG.fine("attachLora after load: " + t.getMessage());
                    }
                }
                current = loaded;
                LOG.info("Loaded adapter from " + dir.getAbsolutePath());
            } catch (IOException e) {
                LOG.warning("Failed loading adapter " + dir + ": " + e.getMessage());
                throw e;
            }
        }
        return current;
    }

    public static PeftModel load(Module base, String path) throws IOException {
        return loadInto(base, null, path);
    }

    public static void save(PeftModel peft, Path dir) throws IOException {
        Objects.requireNonNull(peft, "peft");
        Objects.requireNonNull(dir, "dir");
        Files.createDirectories(dir);
        peft.savePretrained(dir.toFile());
    }

    public static boolean looksLikeAdapterDir(Path dir) {
        if (dir == null || !Files.isDirectory(dir)) return false;
        return Files.isRegularFile(dir.resolve("adapter_config.json"))
                || Files.isRegularFile(dir.resolve("adapter_model.safetensors"))
                || Files.isRegularFile(dir.resolve("adapter_model.bin"));
    }

    private static List<String> splitPaths(String path) {
        List<String> out = new ArrayList<>();
        for (String p : path.split("[,;]+")) {
            if (p != null && !p.isBlank()) out.add(p.trim());
        }
        return out;
    }

    private static LoraConfig loadedConfig(PeftModel peft) {
        try {
            var m = peft.getClass().getMethod("config");
            Object c = m.invoke(peft);
            if (c instanceof LoraConfig lc) return lc;
        } catch (ReflectiveOperationException ignored) {
        }
        return null;
    }
}
