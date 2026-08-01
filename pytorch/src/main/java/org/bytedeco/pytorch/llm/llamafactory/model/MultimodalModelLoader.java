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
import org.bytedeco.pytorch.llm.llamafactory.hparams.FactoryArgs;
import org.bytedeco.pytorch.llm.llamafactory.hparams.ModelArgs;
import org.bytedeco.pytorch.llm.llamafactory.data.collator.MultimodalCollator;
import org.bytedeco.pytorch.llm.transformers.CausalLM;
import org.bytedeco.pytorch.llm.transformers.PretrainedConfig;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;

import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.logging.Logger;

/**
 * Multimodal (LLaVA / Qwen-VL) train-path loader.
 *
 * <p>Composes a vision tower placeholder + projector + causal LM. Full vision
 * encoder weights are optional — offline smoke uses a Linear projector on
 * flattened pixel patches so {@link MultimodalCollator}
 * batches can flow through a tiny forward.
 *
 * <p>Production hosts should inject a real vision tower via
 * {@link #wrap(Module, Module, LinearImpl, PretrainedConfig)}.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class MultimodalModelLoader {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private static final Logger LOG = Logger.getLogger(MultimodalModelLoader.class.getName());

    /** Bundle: LM + optional vision + projector. */
    public static final class VlBundle implements AutoCloseable {
        private final ModelLoader.LoadedModel language;
        private final Module visionTower;   // may be null
        private final LinearImpl projector; // pixel-flat → hidden
        private final Map<String, Object> meta;

        public VlBundle(
                ModelLoader.LoadedModel language,
                Module visionTower,
                LinearImpl projector,
                Map<String, Object> meta) {
            this.language = Objects.requireNonNull(language, "language");
            this.visionTower = visionTower;
            this.projector = projector;
            this.meta = meta == null ? Map.of() : Map.copyOf(meta);
        }

        public ModelLoader.LoadedModel language() { return language; }
        public Module visionTower() { return visionTower; }
        public LinearImpl projector() { return projector; }
        public Map<String, Object> meta() { return meta; }
        public CausalLM causalLM() { return language.causalLM(); }
        public Module module() { return language.module(); }

        @Override
        public void close() {
            language.close();
        }
    }

    private MultimodalModelLoader() {}

    public static VlBundle load(FactoryArgs args) {
        Objects.requireNonNull(args, "args");
        ModelLoader.LoadedModel lang = ModelLoader.load(args);
        int hidden = lang.card().hiddenSize() > 0 ? lang.card().hiddenSize()
                : (lang.causalLM() != null ? lang.causalLM().hiddenSize() : 768);
        // Default projector: flatten 3*224*224 → hidden (smoke path)
        int flat = 3 * 224 * 224;
        LinearImpl projector = new LinearImpl(Math.min(flat, 768), hidden);
        // Use a smaller in-feature for tiny tests: mean-pool pixels externally
        // Re-create with hidden→hidden identity-ish projector for [B,H] vision tokens
        projector = new LinearImpl(hidden, hidden);

        Map<String, Object> meta = new LinkedHashMap<>();
        meta.put("vl", true);
        meta.put("template", args.data().template());
        meta.put("vision", detectVisionKind(args.model()));
        meta.putAll(lang.meta());
        LOG.info("MultimodalModelLoader: vision=" + meta.get("vision")
                + " hidden=" + hidden + " (projector Linear " + hidden + "→" + hidden + ")");
        return new VlBundle(lang, null, projector, meta);
    }

    public static VlBundle wrap(
            Module languageModule,
            Module visionTower,
            LinearImpl projector,
            PretrainedConfig cfg) {
        ModelArgs ma = ModelArgs.defaults();
        ModelLoader.LoadedModel lang = ModelLoader.wrap(languageModule, cfg, null, ma);
        Map<String, Object> meta = new LinkedHashMap<>();
        meta.put("vl", true);
        meta.put("wrapped", true);
        return new VlBundle(lang, visionTower, projector, meta);
    }

    private static String detectVisionKind(ModelArgs model) {
        String id = model.modelNameOrPath() == null ? "" : model.modelNameOrPath().toLowerCase(Locale.ROOT);
        String tpl = "";
        if (id.contains("llava")) return "llava";
        if (id.contains("qwen") && id.contains("vl")) return "qwen_vl";
        if (id.contains("internvl")) return "internvl";
        if (id.contains("minicpm")) return "minicpm_v";
        if (id.contains("deepseek") && id.contains("vl")) return "deepseek_vl";
        return "generic_vl";
    }
}
