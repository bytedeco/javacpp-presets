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
package org.bytedeco.pytorch.llm.llamafactory.data;

import org.bytedeco.pytorch.llm.llamafactory.data.collator.DataCollator;
import org.bytedeco.pytorch.llm.llamafactory.data.collator.KtoCollator;
import org.bytedeco.pytorch.llm.llamafactory.data.collator.MultimodalCollator;
import org.bytedeco.pytorch.llm.llamafactory.data.collator.PairwiseCollator;
import org.bytedeco.pytorch.llm.llamafactory.data.collator.SupervisedCollator;
import org.bytedeco.pytorch.llm.llamafactory.data.converter.AlpacaConverter;
import org.bytedeco.pytorch.llm.llamafactory.data.converter.KtoConverter;
import org.bytedeco.pytorch.llm.llamafactory.data.converter.OpenAIMessagesConverter;
import org.bytedeco.pytorch.llm.llamafactory.data.converter.PreferenceConverter;
import org.bytedeco.pytorch.llm.llamafactory.data.converter.SharegptConverter;
import org.bytedeco.pytorch.llm.llamafactory.data.packing.SequencePacker;
import org.bytedeco.pytorch.llm.llamafactory.data.template.Template;
import org.bytedeco.pytorch.llm.llamafactory.data.template.TemplateRegistry;
import org.bytedeco.pytorch.llm.llamafactory.hparams.DataArgs;
import org.bytedeco.pytorch.llm.llamafactory.hparams.Stage;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.llm.trl.BaseTrainer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Builds tokenized feature lists + collated batches from raw rows.
 *
 * <p>Composition order (mirrors LLaMA-Factory data pipeline):
 * <ol>
 *   <li>format detect → converter (alpaca / sharegpt / openai / preference / kto)</li>
 *   <li>template encode → prompt/response or pairwise texts</li>
 *   <li>{@link SimpleTokenizer} (or injected encoder) → id features</li>
 *   <li>optional {@link SequencePacker}</li>
 *   <li>stage-appropriate {@link DataCollator}</li>
 * </ol>
 *
 * <p>Host systems may supply pre-tokenized rows or a custom {@link TextEncoder}.
 */
public final class DatasetBuilder {

    /** Pluggable text → feature map encoder (production: FastTokenizer wrapper). */
    @FunctionalInterface
    public interface TextEncoder {
        Map<String, Object> encodeSupervised(String prompt, String response, int cutoff);
        default Map<String, Object> encodePretrain(String text, int cutoff) {
            return encodeSupervised("", text, cutoff);
        }
        default Map<String, Object> encodePairwise(
                String prompt, String chosen, String rejected, int cutoff) {
            Map<String, Object> c = encodeSupervised(prompt, chosen, cutoff);
            Map<String, Object> r = encodeSupervised(prompt, rejected, cutoff);
            Map<String, Object> feat = new LinkedHashMap<>();
            feat.put("chosen_input_ids", c.get("input_ids"));
            feat.put("chosen_labels", c.get("labels"));
            feat.put("chosen_attention_mask", c.get("attention_mask"));
            feat.put("rejected_input_ids", r.get("input_ids"));
            feat.put("rejected_labels", r.get("labels"));
            feat.put("rejected_attention_mask", r.get("attention_mask"));
            return feat;
        }
    }

    private final DataArgs dataArgs;
    private final Stage stage;
    private final Template template;
    private final TextEncoder encoder;
    private final long padTokenId;
    private final long ignoreIndex;

    public DatasetBuilder(DataArgs dataArgs, Stage stage, TextEncoder encoder, long padTokenId) {
        this.dataArgs = Objects.requireNonNull(dataArgs, "dataArgs");
        this.stage = stage == null ? Stage.SFT : stage;
        this.template = TemplateRegistry.getOrDefault(dataArgs.template());
        this.encoder = encoder == null ? simpleEncoder() : encoder;
        this.padTokenId = padTokenId;
        this.ignoreIndex = DataCollator.IGNORE_INDEX;
    }

    public DatasetBuilder(DataArgs dataArgs, Stage stage) {
        this(dataArgs, stage, null, SimpleTokenizer.PAD_ID);
    }

    public static DatasetBuilder from(DataArgs dataArgs, Stage stage) {
        return new DatasetBuilder(dataArgs, stage);
    }

    public DataArgs dataArgs() { return dataArgs; }
    public Stage stage() { return stage; }
    public Template template() { return template; }

    private static TextEncoder simpleEncoder() {
        SimpleTokenizer tok = SimpleTokenizer.defaults();
        return new TextEncoder() {
            @Override
            public Map<String, Object> encodeSupervised(String prompt, String response, int cutoff) {
                return tok.encodeSupervised(prompt, response, cutoff);
            }
            @Override
            public Map<String, Object> encodePretrain(String text, int cutoff) {
                return tok.encodePretrain(text, cutoff);
            }
            @Override
            public Map<String, Object> encodePairwise(
                    String prompt, String chosen, String rejected, int cutoff) {
                return tok.encodePairwise(prompt, chosen, rejected, cutoff);
            }
        };
    }

    /**
     * Convert + tokenize raw rows into feature maps ready for collate.
     */
    public List<Map<String, Object>> buildFeatures(List<Map<String, Object>> rawRows) {
        Objects.requireNonNull(rawRows, "rawRows");
        int cutoff = dataArgs.cutoffLen();
        int maxSamples = dataArgs.maxSamples();
        List<Map<String, Object>> features = new ArrayList<>();

        int limit = maxSamples > 0 ? Math.min(maxSamples, rawRows.size()) : rawRows.size();
        for (int i = 0; i < limit; i++) {
            Map<String, Object> raw = rawRows.get(i);
            if (raw == null || raw.isEmpty()) continue;
            Map<String, Object> feat = switch (stage) {
                case PT -> buildPretrain(raw, cutoff);
                case DPO, ORPO, RM -> buildPairwise(raw, cutoff);
                case KTO -> buildKto(raw, cutoff);
                default -> buildSupervised(raw, cutoff); // SFT, PPO prompt, GRPO
            };
            if (feat != null && !feat.isEmpty()) {
                // carry multimodal payloads through
                if (raw.containsKey("pixel_values")) {
                    feat.put("pixel_values", raw.get("pixel_values"));
                }
                if (raw.containsKey("images")) {
                    feat.put("images", raw.get("images"));
                }
                features.add(feat);
            }
        }

        if (dataArgs.packing() && (stage == Stage.SFT || stage == Stage.PT)) {
            SequencePacker packer = new SequencePacker(
                    cutoff, SimpleTokenizer.EOS_ID, ignoreIndex, dataArgs.neatPacking());
            features = packer.pack(features);
        }
        return features;
    }

    private Map<String, Object> buildSupervised(Map<String, Object> raw, int cutoff) {
        Map<String, Object> converted = convertSupervised(raw);
        String prompt = str(converted.get("prompt"), "");
        String response = str(converted.get("response"), "");
        if (prompt.isEmpty() && response.isEmpty()) {
            String text = str(converted.get("text"), "");
            if (text.isEmpty()) return Map.of();
            return encoder.encodePretrain(text, cutoff);
        }
        return encoder.encodeSupervised(prompt, response, cutoff);
    }

    private Map<String, Object> buildPretrain(Map<String, Object> raw, int cutoff) {
        String text = str(raw.get("text"), str(raw.get("content"), str(raw.get("output"), "")));
        if (text.isEmpty()) {
            Map<String, Object> c = convertSupervised(raw);
            text = str(c.get("text"), str(c.get("prompt"), "") + str(c.get("response"), ""));
        }
        if (text.isEmpty()) return Map.of();
        return encoder.encodePretrain(text, cutoff);
    }

    private Map<String, Object> buildPairwise(Map<String, Object> raw, int cutoff) {
        PreferenceConverter conv = new PreferenceConverter(template);
        Map<String, Object> c = conv.convert(raw);
        return encoder.encodePairwise(
                str(c.get("prompt"), ""),
                str(c.get("chosen"), ""),
                str(c.get("rejected"), ""),
                cutoff);
    }

    private Map<String, Object> buildKto(Map<String, Object> raw, int cutoff) {
        KtoConverter conv = new KtoConverter(template);
        Map<String, Object> c = conv.convert(raw);
        Map<String, Object> feat = encoder.encodeSupervised(
                str(c.get("prompt"), ""), str(c.get("response"), ""), cutoff);
        feat = new LinkedHashMap<>(feat);
        feat.put("desirable", c.get("desirable"));
        feat.put("kto_tags", c.get("kto_tags"));
        return feat;
    }

    private Map<String, Object> convertSupervised(Map<String, Object> raw) {
        if (raw.containsKey("messages")) {
            return new OpenAIMessagesConverter(template).convert(raw);
        }
        if (raw.containsKey("conversations")) {
            return new SharegptConverter(template).convert(raw);
        }
        // already prompt/response
        if (raw.containsKey("prompt") && raw.containsKey("response")) {
            return raw;
        }
        return new AlpacaConverter(template).convert(raw);
    }

    /** Stage-appropriate collator. */
    public DataCollator collator() {
        return switch (stage) {
            case DPO, ORPO, RM -> new PairwiseCollator(padTokenId, ignoreIndex, dataArgs.cutoffLen());
            case KTO -> new KtoCollator(padTokenId, ignoreIndex, dataArgs.cutoffLen());
            default -> {
                String t = dataArgs.template() == null ? "" : dataArgs.template().toLowerCase(Locale.ROOT);
                if (t.contains("llava") || t.contains("vl") || t.contains("vision")) {
                    yield new MultimodalCollator(padTokenId, ignoreIndex, dataArgs.cutoffLen(), 3, 224, 224);
                }
                yield new SupervisedCollator(
                        padTokenId, ignoreIndex, dataArgs.cutoffLen(), dataArgs.trainOnPrompt());
            }
        };
    }

    /**
     * Collate a slice of features into a tensor batch.
     */
    public Map<String, Tensor> collate(List<Map<String, Object>> features) {
        return collator().collate(features);
    }

    /**
     * Infinite / cycling batch supplier over features for {@link BaseTrainer#train}.
     *
     * @param features tokenized features
     * @param batchSize micro-batch size
     * @param maxBatches hard stop count; {@code <=0} means unlimited (caller stops via max_steps)
     */
    public BaseTrainer.BatchSupplier batchSupplier(
            List<Map<String, Object>> features, int batchSize, int maxBatches) {
        Objects.requireNonNull(features, "features");
        if (features.isEmpty()) {
            throw new IllegalArgumentException("features must be non-empty");
        }
        int bs = Math.max(1, batchSize);
        DataCollator col = collator();
        AtomicInteger cursor = new AtomicInteger(0);
        AtomicInteger emitted = new AtomicInteger(0);
        return () -> {
            if (maxBatches > 0 && emitted.get() >= maxBatches) {
                return null;
            }
            List<Map<String, Object>> slice = new ArrayList<>(bs);
            for (int i = 0; i < bs; i++) {
                int idx = Math.floorMod(cursor.getAndIncrement(), features.size());
                slice.add(features.get(idx));
            }
            emitted.incrementAndGet();
            return col.collate(slice);
        };
    }

    /** Demo alpaca rows for offline benchmarks (no hub download). */
    public static List<Map<String, Object>> demoAlpacaRows() {
        List<Map<String, Object>> rows = new ArrayList<>();
        rows.add(Map.of(
                "instruction", "What is the capital of France?",
                "input", "",
                "output", "Paris"));
        rows.add(Map.of(
                "instruction", "Translate to French",
                "input", "Hello",
                "output", "Bonjour"));
        rows.add(Map.of(
                "instruction", "Summarize",
                "input", "Cats are animals that meow.",
                "output", "Cats meow."));
        rows.add(Map.of(
                "instruction", "Compute 2+2",
                "input", "",
                "output", "4"));
        return rows;
    }

    /** Demo pairwise preference rows. */
    public static List<Map<String, Object>> demoPreferenceRows() {
        List<Map<String, Object>> rows = new ArrayList<>();
        rows.add(Map.of(
                "instruction", "Say hello",
                "chosen", "Hello! How can I help you today?",
                "rejected", "no"));
        rows.add(Map.of(
                "instruction", "What is 1+1?",
                "chosen", "2",
                "rejected", "I don't know"));
        return rows;
    }

    /** Demo KTO rows. */
    public static List<Map<String, Object>> demoKtoRows() {
        List<Map<String, Object>> rows = new ArrayList<>();
        Map<String, Object> good = new LinkedHashMap<>();
        good.put("instruction", "Greet the user");
        good.put("output", "Hello!");
        good.put("desirable", true);
        rows.add(good);
        Map<String, Object> bad = new LinkedHashMap<>();
        bad.put("instruction", "Greet the user");
        bad.put("output", "go away");
        bad.put("desirable", false);
        rows.add(bad);
        return rows;
    }

    public static List<Map<String, Object>> demoPretrainRows() {
        return List.of(
                Map.of("text", "The quick brown fox jumps over the lazy dog."),
                Map.of("text", "LLaMA-Factory pure Java port trains language models."),
                Map.of("text", "Gradient descent minimizes cross entropy loss."));
    }

    private static String str(Object o, String def) {
        if (o == null) return def;
        String s = String.valueOf(o);
        return s;
    }
}
