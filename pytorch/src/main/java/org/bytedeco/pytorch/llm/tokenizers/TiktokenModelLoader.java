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

import org.bytedeco.pytorch.utils.json.Json;
import org.bytedeco.pytorch.llm.tokenizers.decoders.Decoder;
import org.bytedeco.pytorch.llm.tokenizers.models.TiktokenBpeModel;
import org.bytedeco.pytorch.llm.tokenizers.normalizers.Normalizer;
import org.bytedeco.pytorch.llm.tokenizers.pretokenizers.PreTokenizer;
import org.bytedeco.pytorch.llm.tokenizers.processors.PostProcessor;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * Load ChatGLM4 / tiktoken-style {@code tokenizer.model} text dumps:
 * each line {@code base64(token_bytes) rank_or_id}.
 *
 * <p>Pre-tokenizer uses the cl100k/tiktoken split regex, then ByteLevel
 * ({@code use_regex=false}) so BPE sees {@code bytes_to_unicode} strings.
 */
public final class TiktokenModelLoader {

    /**
     * OpenAI cl100k_base / ChatGLM4 pattern (Java approximation of the Rust/Python regex).
     * Note: {@code (?i:'s|...)} is expanded with case-insensitive alternatives.
     */
    public static final String TIKTOKEN_PATTERN =
            "'s|'t|'re|'ve|'m|'ll|'d|'S|'T|'Re|'VE|'M|'LL|'D|"
                    + "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|"
                    + "\\p{N}{1,3}|"
                    + " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|"
                    + "\\s*[\\r\\n]+|"
                    + "\\s+(?!\\S)|"
                    + "\\s+";

    private TiktokenModelLoader() {}

    public static boolean present(Path dir) {
        if (dir == null) return false;
        Path p = dir.resolve("tokenizer.model");
        if (!Files.isRegularFile(p)) p = dir.resolve("spiece.model");
        if (!Files.isRegularFile(p)) return false;
        // Our text-format detector: first line looks like base64 + int
        try (BufferedReader br = Files.newBufferedReader(p, StandardCharsets.UTF_8)) {
            String line = br.readLine();
            if (line == null) return false;
            line = line.trim();
            // binary protobuf SPM usually starts with 0x0a or non-text
            if (line.isEmpty()) return false;
            // reject obvious binary
            for (int i = 0; i < Math.min(16, line.length()); i++) {
                char c = line.charAt(i);
                if (c == 0) return false;
            }
            String[] parts = line.split("\\s+");
            if (parts.length < 2) return false;
            try {
                Base64.getDecoder().decode(parts[0]);
                Integer.parseInt(parts[1]);
                return true;
            } catch (Exception e) {
                return false;
            }
        } catch (IOException e) {
            return false;
        }
    }

    public static TokenizerPipeline loadFromDirectory(Path dir) throws IOException {
        Path model = dir.resolve("tokenizer.model");
        if (!Files.isRegularFile(model)) model = dir.resolve("spiece.model");
        Map<String, Integer> vocab = loadRanks(model);
        TiktokenBpeModel bpe = new TiktokenBpeModel(vocab);

        // Split (tiktoken regex) then ByteLevel without regex
        PreTokenizer pretok = new PreTokenizer.SequencePreTokenizer(List.of(
                new PreTokenizer.SplitPreTokenizer(
                        Pattern.compile(TIKTOKEN_PATTERN),
                        RegexSplit.Behavior.ISOLATED,
                        false),
                new PreTokenizer.ByteLevelPreTokenizer(false, false, true)
        ));

        List<AddedToken> addedList = new ArrayList<>();
        Path cfgPath = dir.resolve("tokenizer_config.json");
        String unk = null, pad = null, bos = null, eos = null, cls = null, sep = null, mask = null;
        int modelMax = 0;
        if (Files.isRegularFile(cfgPath)) {
            Map<String, Object> cfg = Json.decodeObject(Files.readString(cfgPath, StandardCharsets.UTF_8));
            pad = JsonMaps.asTokenString(cfg.get("pad_token"));
            eos = JsonMaps.asTokenString(cfg.get("eos_token"));
            bos = JsonMaps.asTokenString(cfg.get("bos_token"));
            unk = JsonMaps.asTokenString(cfg.get("unk_token"));
            Integer mml = JsonMaps.asInt(cfg.get("model_max_length"));
            if (mml != null) modelMax = mml;

            // added_tokens_decoder: {"151329": {"content":"<|endoftext|>", "special":true, ...}, ...}
            Map<String, Object> dec = JsonMaps.asMap(cfg.get("added_tokens_decoder"));
            if (dec != null) {
                Map<String, Integer> extended = new LinkedHashMap<>(vocab);
                for (Map.Entry<String, Object> e : dec.entrySet()) {
                    Integer id = JsonMaps.asInt(e.getKey());
                    Map<String, Object> meta = JsonMaps.asMap(e.getValue());
                    if (id == null || meta == null) continue;
                    String content = JsonMaps.asString(meta.get("content"));
                    if (content == null) continue;
                    boolean special = JsonMaps.asBoolean(meta, "special", true);
                    addedList.add(new AddedToken(id, content,
                            JsonMaps.asBoolean(meta, "single_word", false),
                            JsonMaps.asBoolean(meta, "lstrip", false),
                            JsonMaps.asBoolean(meta, "rstrip", false),
                            JsonMaps.asBoolean(meta, "normalized", false),
                            special));
                    extended.put(content, id);
                }
                vocab = extended;
                bpe = new TiktokenBpeModel(vocab);
            }
            List<Object> extra = JsonMaps.asList(cfg.get("additional_special_tokens"));
            if (extra != null) {
                for (Object o : extra) {
                    String c = JsonMaps.asTokenString(o);
                    if (c != null && vocab.containsKey(c)) {
                        // already in added via decoder usually
                    }
                }
            }
        }

        // specials map overlay
        Path stm = dir.resolve("special_tokens_map.json");
        if (Files.isRegularFile(stm)) {
            Map<String, Object> sm = Json.decodeObject(Files.readString(stm, StandardCharsets.UTF_8));
            if (pad == null) pad = JsonMaps.asTokenString(sm.get("pad_token"));
            if (eos == null) eos = JsonMaps.asTokenString(sm.get("eos_token"));
            if (bos == null) bos = JsonMaps.asTokenString(sm.get("bos_token"));
            if (unk == null) unk = JsonMaps.asTokenString(sm.get("unk_token"));
        }

        // ChatGLM4: add_special_tokens prepends [gMASK] <sop>
        PostProcessor post = PostProcessor.NOP;
        if (Files.isRegularFile(cfgPath)) {
            try {
                Map<String, Object> cfg2 = Json.decodeObject(Files.readString(cfgPath, StandardCharsets.UTF_8));
                String tclass = JsonMaps.asString(cfg2.get("tokenizer_class"));
                if (tclass != null && tclass.toLowerCase(java.util.Locale.ROOT).contains("chatglm")) {
                    Integer gmask = null, sop = null;
                    for (AddedToken at : addedList) {
                        if ("[gMASK]".equals(at.content())) gmask = at.id();
                        if ("<sop>".equals(at.content())) sop = at.id();
                    }
                    if (gmask == null) gmask = vocab.get("[gMASK]");
                    if (sop == null) sop = vocab.get("<sop>");
                    if (gmask != null && sop != null) {
                        post = PostProcessor.TemplateProcessing.chatGlm4(gmask, sop);
                    }
                }
            } catch (Exception ignored) {}
        }

        AddedVocabulary added = new AddedVocabulary(addedList);
        return new TokenizerPipeline(
                Normalizer.NOP,
                pretok,
                bpe,
                post,
                Decoder.ByteLevelDecoder.INSTANCE,
                added,
                null,
                modelMax > 0 ? FastTokenizer.Truncation.of(modelMax) : null,
                unk, pad, cls, sep, bos, eos, mask,
                modelMax,
                false
        );
    }

    /**
     * Parse {@code base64 id} lines into bytes_to_unicode vocab.
     */
    public static Map<String, Integer> loadRanks(Path modelFile) throws IOException {
        Map<String, Integer> vocab = new LinkedHashMap<>();
        Base64.Decoder b64 = Base64.getDecoder();
        try (BufferedReader br = Files.newBufferedReader(modelFile, StandardCharsets.UTF_8)) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty() || line.startsWith("#")) continue;
                String[] parts = line.split("\\s+");
                if (parts.length < 2) continue;
                byte[] raw;
                try {
                    raw = b64.decode(parts[0]);
                } catch (IllegalArgumentException e) {
                    continue;
                }
                int id;
                try {
                    id = Integer.parseInt(parts[1]);
                } catch (NumberFormatException e) {
                    // some dumps use "token score" float — skip non-int for tiktoken ranks
                    continue;
                }
                // Map raw bytes through GPT-2 bytes_to_unicode
                char[] chars = new char[raw.length];
                for (int i = 0; i < raw.length; i++) {
                    chars[i] = BytesToUnicode.encodeByte(raw[i] & 0xff);
                }
                vocab.put(new String(chars), id);
            }
        }
        if (vocab.isEmpty()) {
            throw new IOException("No ranks loaded from " + modelFile);
        }
        return vocab;
    }
}
