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
package org.bytedeco.pytorch.llm.tokenizers;

import org.bytedeco.pytorch.llm.tokenizers.models.BpeModel;
import org.bytedeco.pytorch.llm.tokenizers.models.WordPieceModel;
import org.bytedeco.pytorch.utils.json.Json;
import org.bytedeco.pytorch.llm.tokenizers.decoders.Decoder;
import org.bytedeco.pytorch.llm.tokenizers.models.Model;
import org.bytedeco.pytorch.llm.tokenizers.normalizers.Normalizer;
import org.bytedeco.pytorch.llm.tokenizers.pretokenizers.PreTokenizer;
import org.bytedeco.pytorch.llm.tokenizers.processors.PostProcessor;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/**
 * Loads a full HuggingFace {@code tokenizer.json} (tokenizers-rs schema) into a
 * {@link TokenizerPipeline}.
 */
public final class TokenizerJsonLoader {

    private TokenizerJsonLoader() {}

    public static TokenizerPipeline fromFile(Path path) throws IOException {
        String raw = Files.readString(path, StandardCharsets.UTF_8);
        return fromJson(raw);
    }

    public static TokenizerPipeline fromJson(String json) throws IOException {
        Map<String, Object> root = Json.decodeObject(json);
        return fromRoot(root);
    }

    public static TokenizerPipeline fromRoot(Map<String, Object> root) {
        if (root == null) throw new IllegalArgumentException("tokenizer.json root is null");

        // Detect minimal custom schema (our old FastTokenizer dump) vs real HF
        if (isLegacyMinimal(root)) {
            return fromLegacyMinimal(root);
        }

        AddedVocabulary added = AddedVocabulary.empty();
        List<Object> addedRaw = JsonMaps.asList(root.get("added_tokens"));
        if (addedRaw != null) added = AddedVocabulary.fromJsonList(addedRaw);

        Normalizer normalizer = Normalizer.fromJson(JsonMaps.asMap(root.get("normalizer")));
        PreTokenizer preTokenizer = PreTokenizer.fromJson(JsonMaps.asMap(root.get("pre_tokenizer")));
        Model model = Model.fromJson(JsonMaps.asMap(root.get("model")));
        PostProcessor postProcessor = PostProcessor.fromJson(JsonMaps.asMap(root.get("post_processor")));
        Decoder decoder = Decoder.fromJson(JsonMaps.asMap(root.get("decoder")));

        FastTokenizer.Padding padding = parsePadding(root.get("padding"));
        FastTokenizer.Truncation truncation = parseTruncation(root.get("truncation"));

        // Special token strings from added tokens + common names
        String unk = null, pad = null, cls = null, sep = null, bos = null, eos = null, mask = null;
        for (AddedToken t : added.tokens()) {
            String c = t.content();
            String lower = c.toLowerCase();
            if (unk == null && (lower.contains("unk") || c.equals("<unk>") || c.equals("[UNK]"))) unk = c;
            if (pad == null && (lower.contains("pad") || c.equals("<pad>") || c.equals("[PAD]"))) pad = c;
            if (cls == null && (c.equals("[CLS]") || c.equals("<s>") || c.equals("<|begin_of_text|>"))) cls = c;
            if (sep == null && (c.equals("[SEP]") || c.equals("</s>"))) sep = c;
            if (bos == null && (lower.contains("bos") || c.equals("<s>") || c.equals("<|begin_of_text|>")
                    || c.equals("<|im_start|>"))) bos = c;
            if (eos == null && (lower.contains("eos") || c.equals("</s>") || c.equals("<|endoftext|>")
                    || c.equals("<|im_end|>") || c.equals("<|eot_id|>") || c.equals("<|end_of_text|>"))) eos = c;
            if (mask == null && (c.equals("[MASK]") || lower.contains("mask"))) mask = c;
        }
        // Model-level unk
        Map<String, Object> modelMap = JsonMaps.asMap(root.get("model"));
        if (modelMap != null) {
            String mu = JsonMaps.asString(modelMap.get("unk_token"));
            if (mu != null) unk = mu;
        }

        int modelMaxLength = 0;
        // truncation.max_length often carries this
        if (truncation != null && truncation.maxLength > 0) modelMaxLength = truncation.maxLength;

        return new TokenizerPipeline(
                normalizer, preTokenizer, model, postProcessor, decoder,
                added, padding, truncation,
                unk, pad, cls, sep, bos, eos, mask,
                modelMaxLength, false
        );
    }

    private static boolean isLegacyMinimal(Map<String, Object> root) {
        // Our old dump had "backend" and no "pre_tokenizer"/"model.type"
        if (root.containsKey("backend") && !root.containsKey("pre_tokenizer")) {
            Map<String, Object> model = JsonMaps.asMap(root.get("model"));
            if (model != null && !model.containsKey("type") && model.containsKey("vocab")) {
                return true;
            }
        }
        return false;
    }

    private static TokenizerPipeline fromLegacyMinimal(Map<String, Object> root) {
        String backend = JsonMaps.asString(root.get("backend"));
        Map<String, Object> modelMap = JsonMaps.asMap(root.get("model"));
        Map<String, Integer> vocab = modelMap == null
                ? Map.of()
                : JsonMaps.asStringIntMap(modelMap.get("vocab"));

        Model model;
        PreTokenizer pre;
        Decoder dec;
        if ("GPT2".equalsIgnoreCase(backend)) {
            model = new BpeModel(
                    vocab, List.of(), JsonMaps.asString(root.get("unk_token")),
                    null, null, false, false, false);
            pre = new PreTokenizer.ByteLevelPreTokenizer(false, true, true);
            dec = Decoder.ByteLevelDecoder.INSTANCE;
        } else {
            // Treat as WordPiece
            model = new WordPieceModel(
                    vocab,
                    JsonMaps.asString(root.get("unk_token")),
                    "##", 100);
            pre = PreTokenizer.WhitespaceSplitPreTokenizer.INSTANCE;
            dec = new Decoder.WordPieceDecoder("##", true);
        }

        Integer mml = JsonMaps.asInt(root.get("model_max_length"));
        return new TokenizerPipeline(
                Normalizer.NOP, pre, model, PostProcessor.NOP, dec,
                AddedVocabulary.empty(),
                null, mml != null && mml > 0 ? FastTokenizer.Truncation.of(mml) : null,
                JsonMaps.asString(root.get("unk_token")),
                JsonMaps.asString(root.get("pad_token")),
                JsonMaps.asString(root.get("cls_token")),
                JsonMaps.asString(root.get("sep_token")),
                JsonMaps.asString(root.get("bos_token")),
                JsonMaps.asString(root.get("eos_token")),
                JsonMaps.asString(root.get("mask_token")),
                mml == null ? 512 : mml,
                false
        );
    }

    private static FastTokenizer.Padding parsePadding(Object raw) {
        Map<String, Object> m = JsonMaps.asMap(raw);
        if (m == null) return null;
        // HF: {"strategy":"BatchLongest"|int, "direction":"right", ...}
        Object strategy = m.get("strategy");
        String direction = JsonMaps.asString(m.get("direction"));
        if (direction == null) direction = "right";
        if (strategy instanceof Number n) {
            return new FastTokenizer.Padding(FastTokenizer.Padding.Strategy.MAX_LENGTH, n.intValue(), direction);
        }
        String s = JsonMaps.asString(strategy);
        if ("BatchLongest".equals(s) || "Longest".equals(s)) {
            return FastTokenizer.Padding.longest();
        }
        Integer ml = JsonMaps.asInt(m.get("max_length"));
        if (ml != null) {
            return new FastTokenizer.Padding(FastTokenizer.Padding.Strategy.MAX_LENGTH, ml, direction);
        }
        return null;
    }

    private static FastTokenizer.Truncation parseTruncation(Object raw) {
        Map<String, Object> m = JsonMaps.asMap(raw);
        if (m == null) return null;
        Integer ml = JsonMaps.asInt(m.get("max_length"));
        String direction = JsonMaps.asString(m.get("direction"));
        if (ml == null) return null;
        return new FastTokenizer.Truncation(ml, direction == null ? "right" : direction);
    }

    /**
     * Overlay {@code tokenizer_config.json} specials / model_max_length onto a pipeline.
     */
    public static TokenizerPipeline applyTokenizerConfig(TokenizerPipeline pipe, Path configPath)
            throws IOException {
        if (configPath == null || !Files.isRegularFile(configPath)) return pipe;
        Map<String, Object> cfg = Json.decodeObject(Files.readString(configPath, StandardCharsets.UTF_8));
        return applyTokenizerConfig(pipe, cfg);
    }

    public static TokenizerPipeline applyTokenizerConfig(TokenizerPipeline pipe, Map<String, Object> cfg) {
        if (pipe == null || cfg == null) return pipe;
        String unk = firstNonNull(JsonMaps.asTokenString(cfg.get("unk_token")), pipe.unkToken());
        String pad = firstNonNull(JsonMaps.asTokenString(cfg.get("pad_token")), pipe.padToken());
        String cls = firstNonNull(JsonMaps.asTokenString(cfg.get("cls_token")), pipe.clsToken());
        String sep = firstNonNull(JsonMaps.asTokenString(cfg.get("sep_token")), pipe.sepToken());
        String bos = firstNonNull(JsonMaps.asTokenString(cfg.get("bos_token")), pipe.bosToken());
        String eos = firstNonNull(JsonMaps.asTokenString(cfg.get("eos_token")), pipe.eosToken());
        String mask = firstNonNull(JsonMaps.asTokenString(cfg.get("mask_token")), pipe.maskToken());
        Integer mml = JsonMaps.asInt(cfg.get("model_max_length"));
        int modelMax = mml != null && mml > 0 ? mml : pipe.modelMaxLength();
        return pipe.withSpecials(unk, pad, cls, sep, bos, eos, mask, modelMax);
    }

    public static TokenizerPipeline applySpecialTokensMap(TokenizerPipeline pipe, Path path)
            throws IOException {
        if (path == null || !Files.isRegularFile(path)) return pipe;
        Map<String, Object> cfg = Json.decodeObject(Files.readString(path, StandardCharsets.UTF_8));
        return applyTokenizerConfig(pipe, cfg);
    }

    private static String firstNonNull(String a, String b) {
        return a != null ? a : b;
    }
}
