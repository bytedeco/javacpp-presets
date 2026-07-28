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

import org.bytedeco.pytorch.llm.text.tokenizer.Tokenizer;
import org.bytedeco.pytorch.llm.tokenizers.decoders.Decoder;
import org.bytedeco.pytorch.llm.tokenizers.models.*;
import org.bytedeco.pytorch.llm.tokenizers.normalizers.Normalizer;
import org.bytedeco.pytorch.llm.tokenizers.pretokenizers.PreTokenizer;
import org.bytedeco.pytorch.llm.tokenizers.processors.PostProcessor;
import org.bytedeco.pytorch.llm.text.tokenizer.BPETokenizer;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * HuggingFace {@code tokenizers} style fast tokenizer (pure Java).
 *
 * <p>Backed by a full tokenizers-rs pipeline
 * ({@link TokenizerPipeline}: Normalizer → PreTokenizer → Model → PostProcessor → Decoder)
 * loaded from real {@code tokenizer.json} files (Qwen / Llama / DeepSeek / GLM / BERT / …).
 *
 * <pre>{@code
 * FastTokenizer tok = FastTokenizer.fromFile(Path.of("tokenizer.json"));
 * Encoding enc = tok.encode("Hello world", true);
 * String text = tok.decode(enc.ids(), true);
 * }</pre>
 */
public final class FastTokenizer {

    public enum Backend { BPE, WORDPIECE, GPT2, CHAR, WHITESPACE, UNIGRAM, PIPELINE }

    private final Backend backend;
    private final TokenizerPipeline pipeline;

    private FastTokenizer(Backend backend, TokenizerPipeline pipeline) {
        this.backend = backend == null ? Backend.PIPELINE : backend;
        this.pipeline = Objects.requireNonNull(pipeline, "pipeline");
    }

    /** Wrap an existing pipeline. */
    public static FastTokenizer of(TokenizerPipeline pipeline) {
        return new FastTokenizer(detectBackend(pipeline), pipeline);
    }

    public static Builder builder() {
        return new Builder();
    }

    /** WordPiece backend from an ordered vocab map (token → id). */
    public static Builder wordPiece(Map<String, Integer> vocab) {
        WordPieceModel model = new WordPieceModel(new LinkedHashMap<>(vocab), "[UNK]", "##", 100);
        TokenizerPipeline pipe = new TokenizerPipeline(
                Normalizer.NOP,
                PreTokenizer.WhitespaceSplitPreTokenizer.INSTANCE,
                model,
                new PostProcessor.BertProcessing("[CLS]",
                        vocab.getOrDefault("[CLS]", vocab.size()),
                        "[SEP]",
                        vocab.getOrDefault("[SEP]", vocab.size() + 1)),
                new Decoder.WordPieceDecoder("##", true),
                AddedVocabulary.empty(),
                null, null,
                "[UNK]", "[PAD]", "[CLS]", "[SEP]", null, null, "[MASK]",
                512, false);
        return builder().backend(Backend.WORDPIECE).pipeline(pipe)
                .unkToken("[UNK]").padToken("[PAD]").clsToken("[CLS]").sepToken("[SEP]")
                .maskToken("[MASK]");
    }

    /** GPT-2 byte-level BPE (minimal empty-merges seed — prefer {@link #fromFile}). */
    public static Builder gpt2() {
        Map<String, Integer> v = new LinkedHashMap<>();
        for (int i = 0; i < 256; i++) {
            v.put(String.valueOf(BytesToUnicode.encodeByte(i)), i);
        }
        v.put("<|endoftext|>", 256);
        BpeModel model = new BpeModel(v, List.of(), "<|endoftext|>", null, null, false, false, false);
        TokenizerPipeline pipe = new TokenizerPipeline(
                Normalizer.NOP,
                new PreTokenizer.ByteLevelPreTokenizer(false, true, true),
                model,
                PostProcessor.NOP,
                Decoder.ByteLevelDecoder.INSTANCE,
                AddedVocabulary.empty(),
                null, null,
                "<|endoftext|>", "<|endoftext|>", null, null,
                "<|endoftext|>", "<|endoftext|>", null,
                1024, false);
        return builder().backend(Backend.GPT2).pipeline(pipe)
                .unkToken("<|endoftext|>").padToken("<|endoftext|>")
                .bosToken("<|endoftext|>").eosToken("<|endoftext|>")
                .modelMaxLength(1024);
    }

    /** Whitespace split + small vocab (debug / smoke / tiny models). */
    public static Builder whitespace() {
        Map<String, Integer> v = new LinkedHashMap<>();
        v.put("[UNK]", 0);
        v.put("[PAD]", 1);
        v.put("[CLS]", 2);
        v.put("[SEP]", 3);
        Model model = new Model.WordLevelModel(v, "[UNK]");
        // BertProcessing so addSpecialTokens=true still wraps [CLS]/[SEP] for tiny demos
        PostProcessor post = new PostProcessor.BertProcessing("[CLS]", 2, "[SEP]", 3);
        TokenizerPipeline pipe = new TokenizerPipeline(
                Normalizer.NOP,
                PreTokenizer.WhitespaceSplitPreTokenizer.INSTANCE,
                model,
                post,
                Decoder.FUSE,
                AddedVocabulary.empty(),
                null, null,
                "[UNK]", "[PAD]", "[CLS]", "[SEP]", null, null, null,
                512, false);
        return builder().backend(Backend.WHITESPACE).pipeline(pipe)
                .unkToken("[UNK]").padToken("[PAD]").clsToken("[CLS]").sepToken("[SEP]");
    }

    /** Learn a small BPE on a corpus then wrap (torchtext-style demo). */
    public static Builder bpeFromCorpus(Iterable<String> corpus, int numMerges) {
        BPETokenizer bpe = BPETokenizer.learn(corpus, numMerges);
        Map<String, Integer> v = new LinkedHashMap<>(bpe.vocab());
        v.putIfAbsent("<unk>", v.size());
        v.putIfAbsent("<pad>", v.size());
        v.putIfAbsent("<s>", v.size());
        v.putIfAbsent("</s>", v.size());
        List<String> merges = new ArrayList<>(bpe.merges());
        BpeModel model = new BpeModel(v, merges, "<unk>", null, "</w>", false, false, false);
        TokenizerPipeline pipe = new TokenizerPipeline(
                Normalizer.LowercaseNormalizer.INSTANCE,
                PreTokenizer.WhitespaceSplitPreTokenizer.INSTANCE,
                model,
                PostProcessor.NOP,
                new Decoder.BPEDecoder("</w>"),
                AddedVocabulary.empty(),
                null, null,
                "<unk>", "<pad>", null, null, "<s>", "</s>", null,
                512, false);
        return builder().backend(Backend.BPE).pipeline(pipe)
                .unkToken("<unk>").padToken("<pad>").bosToken("<s>").eosToken("</s>");
    }

    // ---- loaders ------------------------------------------------------------

    /** Load a real HuggingFace {@code tokenizer.json} (tokenizers-rs schema). */
    public static FastTokenizer fromFile(Path tokenizerJson) throws IOException {
        TokenizerPipeline pipe = TokenizerJsonLoader.fromFile(tokenizerJson);
        // overlay sibling configs if present
        Path dir = tokenizerJson.getParent();
        if (dir != null) {
            pipe = TokenizerJsonLoader.applyTokenizerConfig(pipe, dir.resolve("tokenizer_config.json"));
            pipe = TokenizerJsonLoader.applySpecialTokensMap(pipe, dir.resolve("special_tokens_map.json"));
        }
        return new FastTokenizer(detectBackend(pipe), pipe);
    }

    public static FastTokenizer fromTokenizerJson(String json) throws IOException {
        TokenizerPipeline pipe = TokenizerJsonLoader.fromJson(json);
        return new FastTokenizer(detectBackend(pipe), pipe);
    }

    /**
     * Load from a model snapshot directory:
     * {@code tokenizer.json} → else {@code vocab.json}+{@code merges.txt} → else whitespace.
     */
    public static FastTokenizer fromDirectory(Path dir) throws IOException {
        return DirectoryTokenizerLoader.load(dir);
    }

    private static Backend detectBackend(TokenizerPipeline pipe) {
        Model m = pipe.model();
        if (m instanceof BpeModel) return Backend.BPE;
        if (m instanceof TiktokenBpeModel) return Backend.BPE;
        if (m instanceof WordPieceModel) return Backend.WORDPIECE;
        if (m instanceof UnigramModel) return Backend.UNIGRAM;
        return Backend.PIPELINE;
    }

    // ---- encode / decode ----------------------------------------------------

    public Encoding encode(String text) {
        return encode(text, false);
    }

    public Encoding encode(String text, boolean addSpecialTokens) {
        Encoding enc = pipeline.encode(text, addSpecialTokens);
        return applyConfiguredPadTruncate(enc);
    }

    public Encoding encodePair(String textA, String textB, boolean addSpecialTokens) {
        Encoding enc = pipeline.encodePair(textA, textB, addSpecialTokens);
        return applyConfiguredPadTruncate(enc);
    }

    public List<Encoding> encodeBatch(List<String> texts, boolean addSpecialTokens) {
        if (texts == null || texts.isEmpty()) return List.of();
        List<Encoding> out = new ArrayList<>(texts.size());
        int max = 0;
        for (String t : texts) {
            Encoding e = encode(t, addSpecialTokens);
            out.add(e);
            if (e.size() > max) max = e.size();
        }
        Padding pad = pipeline.padding();
        if (pad != null && pad.strategy == Padding.Strategy.LONGEST) {
            List<Encoding> padded = new ArrayList<>(out.size());
            for (Encoding e : out) {
                padded.add(e.padTo(max, padId(), 0, pad.direction));
            }
            return padded;
        }
        return out;
    }

    public String decode(int[] ids) {
        return decode(ids, true);
    }

    public String decode(int[] ids, boolean skipSpecialTokens) {
        return pipeline.decode(ids, skipSpecialTokens);
    }

    private Encoding applyConfiguredPadTruncate(Encoding enc) {
        // pipeline already applies its own padding/truncation; this is a no-op safety
        return enc;
    }

    // ---- vocab / specials ---------------------------------------------------

    public List<String> convertIdsToTokens(int[] ids) {
        if (ids == null) return List.of();
        List<String> out = new ArrayList<>(ids.length);
        for (int id : ids) {
            String t = pipeline.idToToken(id);
            out.add(t == null ? "" : t);
        }
        return out;
    }

    public int[] convertTokensToIds(List<String> tokens) {
        if (tokens == null) return new int[0];
        int[] ids = new int[tokens.size()];
        for (int i = 0; i < tokens.size(); i++) {
            ids[i] = pipeline.tokenToId(tokens.get(i));
        }
        return ids;
    }

    public int tokenToId(String token) {
        return pipeline.tokenToId(token);
    }

    public String idToToken(int id) {
        return pipeline.idToToken(id);
    }

    public int vocabSize() {
        return pipeline.vocabSize();
    }

    public Map<String, Integer> getVocab() {
        return pipeline.getVocab();
    }

    public Backend backend() {
        return backend;
    }

    public TokenizerPipeline pipeline() {
        return pipeline;
    }

    public int padId() { return pipeline.padId(); }
    public int unkId() { return pipeline.unkId(); }
    public int clsId() { return tokenToId(clsToken()); }
    public int sepId() { return tokenToId(sepToken()); }
    public int bosId() { return pipeline.bosId(); }
    public int eosId() { return pipeline.eosId(); }

    public String unkToken() { return pipeline.unkToken(); }
    public String padToken() { return pipeline.padToken(); }
    public String clsToken() { return pipeline.clsToken(); }
    public String sepToken() { return pipeline.sepToken(); }
    public String bosToken() { return pipeline.bosToken(); }
    public String eosToken() { return pipeline.eosToken(); }
    public String maskToken() { return pipeline.maskToken(); }
    public int modelMaxLength() { return pipeline.modelMaxLength(); }

    public FastTokenizer withPadding(Padding padding) {
        return new FastTokenizer(backend, pipeline.withPadding(padding));
    }

    public FastTokenizer withTruncation(Truncation truncation) {
        return new FastTokenizer(backend, pipeline.withTruncation(truncation));
    }

    // ---- serialize (minimal, for round-trip of synthetic builders) ----------

    public String toTokenizerJson() {
        // Prefer not to re-dump full HF schema; emit a compact legacy-compatible form
        // plus backend hint so fromTokenizerJson can rebuild a usable pipeline.
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        sb.append("  \"version\": \"1.0\",\n");
        sb.append("  \"backend\": \"").append(backend.name()).append("\",\n");
        sb.append("  \"model_max_length\": ").append(modelMaxLength()).append(",\n");
        sb.append("  \"unk_token\": ").append(jsonStr(unkToken())).append(",\n");
        sb.append("  \"pad_token\": ").append(jsonStr(padToken())).append(",\n");
        sb.append("  \"cls_token\": ").append(jsonStr(clsToken())).append(",\n");
        sb.append("  \"sep_token\": ").append(jsonStr(sepToken())).append(",\n");
        sb.append("  \"bos_token\": ").append(jsonStr(bosToken())).append(",\n");
        sb.append("  \"eos_token\": ").append(jsonStr(eosToken())).append(",\n");
        sb.append("  \"mask_token\": ").append(jsonStr(maskToken())).append(",\n");
        sb.append("  \"model\": {\n    \"vocab\": {\n");
        int i = 0;
        for (Map.Entry<String, Integer> e : getVocab().entrySet()) {
            if (i++ > 0) sb.append(",\n");
            sb.append("      ").append(jsonStr(e.getKey())).append(": ").append(e.getValue());
            // cap dump size for huge vocabs in debug paths
            if (i >= 50_000) break;
        }
        sb.append("\n    }\n  }\n}\n");
        return sb.toString();
    }

    public void save(Path dir) throws IOException {
        Files.createDirectories(dir);
        Files.writeString(dir.resolve("tokenizer.json"), toTokenizerJson(), StandardCharsets.UTF_8);
    }

    private static String jsonStr(String s) {
        if (s == null) return "null";
        return "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"") + "\"";
    }

    // ---- padding / truncation config ----------------------------------------

    public static final class Padding {
        public enum Strategy { LONGEST, MAX_LENGTH, DO_NOT_PAD }
        public final Strategy strategy;
        public final int maxLength;
        public final String direction; // "right" | "left"

        public Padding(Strategy strategy, int maxLength, String direction) {
            this.strategy = strategy == null ? Strategy.DO_NOT_PAD : strategy;
            this.maxLength = maxLength;
            this.direction = direction == null ? "right" : direction;
        }

        public static Padding longest() {
            return new Padding(Strategy.LONGEST, 0, "right");
        }

        public static Padding maxLength(int maxLength) {
            return new Padding(Strategy.MAX_LENGTH, maxLength, "right");
        }

        public static Padding none() {
            return new Padding(Strategy.DO_NOT_PAD, 0, "right");
        }
    }

    public static final class Truncation {
        public final int maxLength;
        public final String direction; // "right" | "left"

        public Truncation(int maxLength, String direction) {
            this.maxLength = maxLength;
            this.direction = direction == null ? "right" : direction;
        }

        public static Truncation of(int maxLength) {
            return new Truncation(maxLength, "right");
        }
    }

    // ---- builder ------------------------------------------------------------

    public static final class Builder {
        private Backend backend = Backend.PIPELINE;
        private TokenizerPipeline pipeline;
        private String unkToken;
        private String padToken;
        private String clsToken;
        private String sepToken;
        private String bosToken;
        private String eosToken;
        private String maskToken;
        private int modelMaxLength = -1;
        private Padding padding;
        private Truncation truncation;

        public Builder backend(Backend backend) {
            this.backend = backend;
            return this;
        }

        public Builder pipeline(TokenizerPipeline pipeline) {
            this.pipeline = pipeline;
            return this;
        }

        public Builder unkToken(String unkToken) { this.unkToken = unkToken; return this; }
        public Builder padToken(String padToken) { this.padToken = padToken; return this; }
        public Builder clsToken(String clsToken) { this.clsToken = clsToken; return this; }
        public Builder sepToken(String sepToken) { this.sepToken = sepToken; return this; }
        public Builder bosToken(String bosToken) { this.bosToken = bosToken; return this; }
        public Builder eosToken(String eosToken) { this.eosToken = eosToken; return this; }
        public Builder maskToken(String maskToken) { this.maskToken = maskToken; return this; }
        public Builder modelMaxLength(int modelMaxLength) { this.modelMaxLength = modelMaxLength; return this; }
        public Builder addPrefixSpace(boolean addPrefixSpace) {
            // applied at build via pipeline rebuild if needed — stored on pipeline already for factories
            return this;
        }
        public Builder padding(Padding padding) { this.padding = padding; return this; }
        public Builder truncation(Truncation truncation) { this.truncation = truncation; return this; }

        /** @deprecated engine is replaced by pipeline; kept for source compatibility no-op. */
        @Deprecated
        public Builder engine(Tokenizer engine) {
            return this;
        }

        /** @deprecated vocab set via pipeline; kept for source compatibility. */
        @Deprecated
        public Builder vocab(Map<String, Integer> vocab) {
            return this;
        }

        public FastTokenizer build() {
            if (pipeline == null) {
                throw new IllegalStateException("pipeline required — use fromFile/fromDirectory or a factory builder");
            }
            TokenizerPipeline p = pipeline;
            if (unkToken != null || padToken != null || clsToken != null || sepToken != null
                    || bosToken != null || eosToken != null || maskToken != null || modelMaxLength > 0) {
                p = p.withSpecials(
                        unkToken != null ? unkToken : p.unkToken(),
                        padToken != null ? padToken : p.padToken(),
                        clsToken != null ? clsToken : p.clsToken(),
                        sepToken != null ? sepToken : p.sepToken(),
                        bosToken != null ? bosToken : p.bosToken(),
                        eosToken != null ? eosToken : p.eosToken(),
                        maskToken != null ? maskToken : p.maskToken(),
                        modelMaxLength > 0 ? modelMaxLength : p.modelMaxLength());
            }
            if (padding != null) p = p.withPadding(padding);
            if (truncation != null) p = p.withTruncation(truncation);
            return new FastTokenizer(backend, p);
        }
    }
}
