/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or (at your option) any later version (collectively, the "License");
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
package org.bytedeco.pytorch.utils.tokenizers;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.distributed.*;

import org.bytedeco.pytorch.utils.tokenizers.decoders.Decoder;
import org.bytedeco.pytorch.utils.tokenizers.models.BpeModel;
import org.bytedeco.pytorch.utils.tokenizers.models.TiktokenBpeModel;
import org.bytedeco.pytorch.utils.tokenizers.normalizers.Normalizer;
import org.bytedeco.pytorch.utils.tokenizers.pretokenizers.PreTokenizer;
import org.bytedeco.pytorch.utils.tokenizers.processors.PostProcessor;

import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.zip.GZIPInputStream;

/**
 * OpenAI <a href="https://github.com/openai/tiktoken">tiktoken</a>-compatible BPE tokenizer — pure Java.
 *
 * <p>Built-in encodings (loaded from classpath resources under
 * {@code org/bytedeco/pytorch/utils/tokenizers/tiktoken/}):
 * <ul>
 *   <li>{@code gpt2} / {@code r50k_base} — GPT-2 / Codex r50k (50 257 vocab incl. EOT)</li>
 *   <li>{@code p50k_base} — Codex / text-davinci-002-003 (50 281)</li>
 *   <li>{@code p50k_edit} — edit models with FIM specials (50 284)</li>
 *   <li>{@code cl100k_base} — GPT-4 / ChatGPT / text-embedding-3 (100 277)</li>
 *   <li>{@code o200k_base} — GPT-4o / o1 / o3 (200 019)</li>
 *   <li>{@code o200k_harmony} — gpt-oss harmony specials on o200k ranks</li>
 * </ul>
 *
 * <p>Public surface mirrors Python tiktoken 0.13+:
 * {@code get_encoding} / {@code encoding_for_model} / {@code list_encoding_names} /
 * {@code encode} / {@code encode_ordinary} / {@code encode_batch} /
 * {@code decode} / {@code decode_bytes} / {@code encode_single_token} /
 * {@code special_tokens_set} / {@code eot_token} / {@code n_vocab}.
 *
 * <pre>{@code
 * Tiktoken enc = Tiktoken.getEncoding("cl100k_base");
 * // or
 * Tiktoken enc = Tiktoken.encodingForModel("gpt-4o");
 *
 * int[] ids = enc.encodeOrdinary("Hello world");          // [9906, 1917]
 * int[] eot = enc.encode("<|endoftext|>", "all");         // [100257]
 * String text = enc.decode(ids);                          // "Hello world"
 * byte[] raw = enc.decodeBytes(ids);
 *
 * // HF FastTokenizer adapter for transformers pipelines
 * FastTokenizer ft = enc.toFastTokenizer();
 * }</pre>
 */
public final class Tiktoken {

    // ---- Public encoding name constants ----

    public static final String GPT2           = "gpt2";
    public static final String R50K_BASE      = "r50k_base";
    public static final String P50K_BASE      = "p50k_base";
    public static final String P50K_EDIT      = "p50k_edit";
    public static final String CL100K_BASE    = "cl100k_base";
    public static final String O200K_BASE     = "o200k_base";
    public static final String O200K_HARMONY  = "o200k_harmony";

    /** Sentinel: allow every special token during {@link #encode(String, Object, Object)}. */
    public static final String SPECIAL_ALL = "all";

    private static final List<String> ENCODING_NAMES = List.of(
            GPT2, R50K_BASE, P50K_BASE, P50K_EDIT, CL100K_BASE, O200K_BASE, O200K_HARMONY);

    private static final Set<String> KNOWN = new LinkedHashSet<>(ENCODING_NAMES);

    /** Exact model → encoding (from tiktoken.model.MODEL_TO_ENCODING). */
    private static final Map<String, String> MODEL_TO_ENCODING = new LinkedHashMap<>();
    /** Longest-prefix model → encoding (from tiktoken.model.MODEL_PREFIX_TO_ENCODING). */
    private static final List<Map.Entry<String, String>> MODEL_PREFIX_TO_ENCODING = new ArrayList<>();

    static {
        // Fallback built-ins — overridden/extended by model_to_encoding.json when present.
        putModel("o1", O200K_BASE);
        putModel("o3", O200K_BASE);
        putModel("o4-mini", O200K_BASE);
        putModel("gpt-5", O200K_BASE);
        putModel("gpt-4.1", O200K_BASE);
        putModel("gpt-4o", O200K_BASE);
        putModel("gpt-4", CL100K_BASE);
        putModel("gpt-3.5-turbo", CL100K_BASE);
        putModel("gpt-3.5", CL100K_BASE);
        putModel("gpt-35-turbo", CL100K_BASE);
        putModel("davinci-002", CL100K_BASE);
        putModel("babbage-002", CL100K_BASE);
        putModel("text-embedding-ada-002", CL100K_BASE);
        putModel("text-embedding-3-small", CL100K_BASE);
        putModel("text-embedding-3-large", CL100K_BASE);
        putModel("text-davinci-003", P50K_BASE);
        putModel("text-davinci-002", P50K_BASE);
        putModel("code-davinci-002", P50K_BASE);
        putModel("code-davinci-001", P50K_BASE);
        putModel("davinci", R50K_BASE);
        putModel("curie", R50K_BASE);
        putModel("babbage", R50K_BASE);
        putModel("ada", R50K_BASE);
        putModel("gpt2", GPT2);
        putModel("gpt-2", GPT2);

        putPrefix("o1-", O200K_BASE);
        putPrefix("o3-", O200K_BASE);
        putPrefix("o4-mini-", O200K_BASE);
        putPrefix("gpt-5-", O200K_BASE);
        putPrefix("gpt-4.5-", O200K_BASE);
        putPrefix("gpt-4.1-", O200K_BASE);
        putPrefix("chatgpt-4o-", O200K_BASE);
        putPrefix("gpt-4o-", O200K_BASE);
        putPrefix("gpt-4-", CL100K_BASE);
        putPrefix("gpt-3.5-turbo-", CL100K_BASE);
        putPrefix("gpt-35-turbo-", CL100K_BASE);
        putPrefix("gpt-oss-", O200K_HARMONY);
        putPrefix("ft:gpt-4o", O200K_BASE);
        putPrefix("ft:gpt-4", CL100K_BASE);
        putPrefix("ft:gpt-3.5-turbo", CL100K_BASE);
        putPrefix("ft:davinci-002", CL100K_BASE);
        putPrefix("ft:babbage-002", CL100K_BASE);

        loadModelMapFromResource();
        // Longest prefix first for matching
        MODEL_PREFIX_TO_ENCODING.sort((a, b) -> Integer.compare(b.getKey().length(), a.getKey().length()));
    }

    private static void putModel(String model, String enc) {
        MODEL_TO_ENCODING.put(model, enc);
    }

    private static void putPrefix(String prefix, String enc) {
        // de-dupe by prefix
        for (int i = 0; i < MODEL_PREFIX_TO_ENCODING.size(); i++) {
            if (MODEL_PREFIX_TO_ENCODING.get(i).getKey().equals(prefix)) {
                MODEL_PREFIX_TO_ENCODING.set(i, Map.entry(prefix, enc));
                return;
            }
        }
        MODEL_PREFIX_TO_ENCODING.add(Map.entry(prefix, enc));
    }

    @SuppressWarnings("unchecked")
    private static void loadModelMapFromResource() {
        String path = "/org/bytedeco/pytorch/utils/tokenizers/tiktoken/model_to_encoding.json";
        try (InputStream in = Tiktoken.class.getResourceAsStream(path)) {
            if (in == null) return;
            String json = new String(in.readAllBytes(), StandardCharsets.UTF_8);
            // Minimal JSON object parse for two string→string maps (avoid heavy deps here).
            Map<String, Object> root = parseSimpleJsonObject(json);
            Object m = root.get("model_to_encoding");
            if (m instanceof Map<?, ?> mm) {
                for (Map.Entry<?, ?> e : mm.entrySet()) {
                    if (e.getKey() != null && e.getValue() != null) {
                        MODEL_TO_ENCODING.put(String.valueOf(e.getKey()), String.valueOf(e.getValue()));
                    }
                }
            }
            Object p = root.get("model_prefix_to_encoding");
            if (p instanceof Map<?, ?> pm) {
                for (Map.Entry<?, ?> e : pm.entrySet()) {
                    if (e.getKey() != null && e.getValue() != null) {
                        putPrefix(String.valueOf(e.getKey()), String.valueOf(e.getValue()));
                    }
                }
            }
        } catch (Exception ignored) {
            // Built-in fallback map is enough.
        }
    }

    // ---- Instance state ----

    private final String name;
    private final Pattern pattern;
    /** Rank table: token-bytes as Latin-1 String → rank/id. */
    private final Map<String, Integer> ranks;
    /** id → raw token bytes (as Latin-1 String). Sparse-capable via map. */
    private final Map<Integer, String> idToToken;
    private final Map<String, Integer> specialTokens;
    private final Set<String> specialTokensSet;
    private final Set<Integer> specialTokenIds;
    private final int maxTokenValue;
    private final int eotToken;
    private final int nVocab;
    private final AddedVocabulary addedVocabulary;
    private final TiktokenBpeModel bpeModel;
    /** Precompiled special-token finder (longest first). */
    private final Pattern specialPattern;
    private final List<String> specialOrdered; // longest first for greedy match

    private Tiktoken(String name,
                     Map<String, Integer> ranks,
                     Map<String, Integer> specialTokens,
                     Pattern pattern) {
        this.name = Objects.requireNonNull(name, "name");
        this.ranks = Collections.unmodifiableMap(new LinkedHashMap<>(Objects.requireNonNull(ranks)));
        this.specialTokens = Collections.unmodifiableMap(new LinkedHashMap<>(
                specialTokens == null ? Map.of() : specialTokens));
        this.specialTokensSet = Collections.unmodifiableSet(new LinkedHashSet<>(this.specialTokens.keySet()));
        this.specialTokenIds = new HashSet<>(this.specialTokens.values());
        this.pattern = Objects.requireNonNull(pattern, "pattern");

        Map<Integer, String> idMap = new HashMap<>(this.ranks.size() * 2);
        int max = -1;
        for (Map.Entry<String, Integer> e : this.ranks.entrySet()) {
            idMap.put(e.getValue(), e.getKey());
            if (e.getValue() > max) max = e.getValue();
        }
        for (Map.Entry<String, Integer> e : this.specialTokens.entrySet()) {
            idMap.put(e.getValue(), e.getKey()); // specials stored as UTF-8 text, not byte-latin1
            if (e.getValue() > max) max = e.getValue();
        }
        this.idToToken = idMap;
        this.maxTokenValue = max;
        // Python: n_vocab == max_token_value + 1 (includes gaps in id space)
        this.nVocab = max + 1;

        Integer eot = this.specialTokens.get("<|endoftext|>");
        this.eotToken = eot != null ? eot : -1;

        List<AddedToken> addedList = new ArrayList<>(this.specialTokens.size());
        for (Map.Entry<String, Integer> e : this.specialTokens.entrySet()) {
            addedList.add(AddedToken.of(e.getValue(), e.getKey(), true));
        }
        this.addedVocabulary = new AddedVocabulary(addedList);
        // TiktokenBpeModel expects bytes_to_unicode keys; our ranks are raw-byte Latin-1.
        // Keep a parallel model for FastTokenizer adapter built via BytesToUnicode mapping.
        this.bpeModel = new TiktokenBpeModel(toBytesToUnicodeVocab(this.ranks));

        List<String> ordered = new ArrayList<>(this.specialTokens.keySet());
        ordered.sort((a, b) -> Integer.compare(b.length(), a.length()));
        this.specialOrdered = List.copyOf(ordered);
        if (ordered.isEmpty()) {
            this.specialPattern = null;
        } else {
            StringBuilder sp = new StringBuilder();
            for (int i = 0; i < ordered.size(); i++) {
                if (i > 0) sp.append('|');
                sp.append(Pattern.quote(ordered.get(i)));
            }
            this.specialPattern = Pattern.compile(sp.toString());
        }
    }

    /** Convert raw-byte Latin-1 rank keys → GPT-2 bytes_to_unicode keys for TiktokenBpeModel. */
    private static Map<String, Integer> toBytesToUnicodeVocab(Map<String, Integer> rawRanks) {
        Map<String, Integer> out = new LinkedHashMap<>(rawRanks.size() * 2);
        for (Map.Entry<String, Integer> e : rawRanks.entrySet()) {
            String raw = e.getKey();
            char[] chars = new char[raw.length()];
            for (int i = 0; i < raw.length(); i++) {
                chars[i] = BytesToUnicode.encodeByte(raw.charAt(i) & 0xFF);
            }
            out.put(new String(chars), e.getValue());
        }
        return out;
    }

    // ---- Factory (Python: get_encoding / encoding_for_model / list_encoding_names) ----

    /** Python {@code tiktoken.list_encoding_names()}. */
    public static List<String> listEncodingNames() {
        return ENCODING_NAMES;
    }

    /** Python {@code tiktoken.get_encoding(name)} — cached. */
    public static synchronized Tiktoken getEncoding(String name) {
        return forEncoding(name);
    }

    /**
     * Return a cached Tiktoken instance for the given encoding name.
     *
     * @param name one of {@link #listEncodingNames()}
     */
    public static synchronized Tiktoken forEncoding(String name) {
        if (name == null) throw new IllegalArgumentException("encoding name is null");
        Tiktoken cached = LOADED.get(name);
        if (cached != null) return cached;
        if (!KNOWN.contains(name)) {
            throw new IllegalArgumentException(
                    "Unknown encoding '" + name + "'. Known: " + KNOWN);
        }
        Tiktoken enc = load(name);
        LOADED.put(name, enc);
        return enc;
    }

    /** Python {@code tiktoken.encoding_name_for_model(model)}. */
    public static String encodingNameForModel(String modelName) {
        if (modelName == null || modelName.isBlank()) {
            throw new IllegalArgumentException("modelName is blank");
        }
        String exact = MODEL_TO_ENCODING.get(modelName);
        if (exact != null) return exact;
        for (Map.Entry<String, String> e : MODEL_PREFIX_TO_ENCODING) {
            if (modelName.startsWith(e.getKey())) return e.getValue();
        }
        throw new IllegalArgumentException(
                "Could not automatically map " + modelName
                        + " to a tiktoken encoding. Use getEncoding(...) explicitly.");
    }

    /** Python {@code tiktoken.encoding_for_model(model)}. */
    public static Tiktoken encodingForModel(String modelName) {
        return getEncoding(encodingNameForModel(modelName));
    }

    /** Alias kept for older call sites. */
    public static Tiktoken forModel(String modelName) {
        return encodingForModel(modelName);
    }

    private static final Map<String, Tiktoken> LOADED = new HashMap<>();

    private static Tiktoken load(String name) {
        // Prefer portable .ranks.gz (all 7 encodings), fall back to legacy Java-serialized .tiktoken
        String ranksPath = "/org/bytedeco/pytorch/utils/tokenizers/tiktoken/" + name + ".ranks.gz";
        try (InputStream raw = Tiktoken.class.getResourceAsStream(ranksPath)) {
            if (raw != null) {
                return loadRanksGz(name, raw);
            }
        } catch (Exception e) {
            throw new RuntimeException("Failed to load tiktoken ranks.gz: " + name, e);
        }

        String legacyPath = "/org/bytedeco/pytorch/utils/tokenizers/tiktoken/" + name + ".tiktoken";
        try (InputStream raw = Tiktoken.class.getResourceAsStream(legacyPath)) {
            if (raw == null) {
                throw new FileNotFoundException(
                        "Tiktoken resource not found on classpath: " + ranksPath + " or " + legacyPath);
            }
            return loadLegacySerialized(name, raw);
        } catch (RuntimeException e) {
            throw e;
        } catch (Exception e) {
            throw new RuntimeException("Failed to load tiktoken encoding: " + name, e);
        }
    }

    /**
     * Load portable text format produced by our exporter:
     * <pre>
     * #TIKTOKEN_V1
     * PATTERN\t&lt;pat&gt;
     * SPECIAL\t&lt;token&gt;\t&lt;id&gt;
     * VOCAB
     * &lt;b64&gt;\t&lt;id&gt;
     * </pre>
     * gzip-compressed.
     */
    private static Tiktoken loadRanksGz(String name, InputStream gzBytes) throws IOException {
        try (GZIPInputStream gz = new GZIPInputStream(gzBytes);
             BufferedReader br = new BufferedReader(new InputStreamReader(gz, StandardCharsets.UTF_8))) {
            String header = br.readLine();
            if (header == null || !header.startsWith("#TIKTOKEN_V1")) {
                throw new IOException("Bad tiktoken ranks header for " + name + ": " + header);
            }
            String patStr = null;
            Map<String, Integer> specials = new LinkedHashMap<>();
            Map<String, Integer> ranks = new LinkedHashMap<>();
            boolean inVocab = false;
            Base64.Decoder b64 = Base64.getDecoder();
            String line;
            while ((line = br.readLine()) != null) {
                if (line.isEmpty()) continue;
                if (!inVocab) {
                    if (line.startsWith("PATTERN\t")) {
                        patStr = line.substring("PATTERN\t".length());
                    } else if (line.startsWith("SPECIAL\t")) {
                        String rest = line.substring("SPECIAL\t".length());
                        int tab = rest.lastIndexOf('\t');
                        if (tab <= 0) continue;
                        String tok = rest.substring(0, tab);
                        int id = Integer.parseInt(rest.substring(tab + 1));
                        specials.put(tok, id);
                    } else if (line.equals("VOCAB")) {
                        inVocab = true;
                    }
                } else {
                    int tab = line.lastIndexOf('\t');
                    if (tab <= 0) continue;
                    byte[] rawTok = b64.decode(line.substring(0, tab));
                    int id = Integer.parseInt(line.substring(tab + 1));
                    // Store as Latin-1 String (one char per byte) — matches Python bytes ranks.
                    ranks.put(new String(rawTok, StandardCharsets.ISO_8859_1), id);
                }
            }
            if (patStr == null) throw new IOException("Missing PATTERN for " + name);
            if (ranks.isEmpty()) throw new IOException("Empty ranks for " + name);
            Pattern pat = compileTiktokenPattern(patStr);
            return new Tiktoken(name, ranks, specials, pat);
        }
    }

    /**
     * Java's {@link Pattern} does not support possessive quantifiers ({@code ++}, {@code ?+})
     * the same way as the Rust regex crate used by tiktoken, nor {@code (?i:...)} inside all
     * positions identically. We rewrite possessive → greedy (semantically OK for tiktoken split
     * because the alternatives are ordered and non-overlapping in practice) and keep {@code (?i:)}.
     */
    static Pattern compileTiktokenPattern(String patStr) {
        // Replace possessive quantifiers ++ *+ ?+ with greedy equivalents.
        // Careful: only replace quantifier forms, not literal '+' inside character classes is hard;
        // tiktoken patterns use ++ / ?+ only as quantifiers outside classes.
        String java = patStr
                .replace("++", "+")
                .replace("*+", "*")
                .replace("?+", "?");
        // \p{L} etc. work in Java. Compile with UNICODE_CHARACTER_CLASS for \w-like if needed — not used.
        try {
            return Pattern.compile(java);
        } catch (Exception e) {
            // Last-resort: cl100k-like approximation used by TiktokenModelLoader
            return Pattern.compile(TiktokenModelLoader.TIKTOKEN_PATTERN);
        }
    }

    @SuppressWarnings("unchecked")
    private static Tiktoken loadLegacySerialized(String name, InputStream raw) throws Exception {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        raw.transferTo(baos);
        String b64 = baos.toString(StandardCharsets.US_ASCII).trim();
        byte[] compressed = Base64.getDecoder().decode(b64);
        try (GZIPInputStream gz = new GZIPInputStream(new ByteArrayInputStream(compressed));
             ObjectInputStream ois = new ObjectInputStream(gz)) {
            Map<String, Integer> vocab = (Map<String, Integer>) ois.readObject();
            Map<String, Integer> special = (Map<String, Integer>) ois.readObject();
            String patStr = (String) ois.readObject();
            Pattern pat = compileTiktokenPattern(patStr);
            return new Tiktoken(name, vocab, special, pat);
        }
    }

    /**
     * Load from a custom portable ranks file (gzip text format) or a ChatGLM-style
     * {@code base64 id} lines file.
     */
    public static Tiktoken fromRanksFile(String name, Path file) throws IOException {
        byte[] bytes = Files.readAllBytes(file);
        // gzip magic
        if (bytes.length >= 2 && (bytes[0] == (byte) 0x1f && bytes[1] == (byte) 0x8b)) {
            return loadRanksGz(name, new ByteArrayInputStream(bytes));
        }
        // plain text base64\tid lines (ChatGLM / tiktoken dump)
        Map<String, Integer> ranks = new LinkedHashMap<>();
        Base64.Decoder b64 = Base64.getDecoder();
        for (String line : Files.readAllLines(file, StandardCharsets.UTF_8)) {
            line = line.trim();
            if (line.isEmpty() || line.startsWith("#")) continue;
            String[] parts = line.split("\\s+");
            if (parts.length < 2) continue;
            try {
                byte[] raw = b64.decode(parts[0]);
                int id = Integer.parseInt(parts[1]);
                ranks.put(new String(raw, StandardCharsets.ISO_8859_1), id);
            } catch (Exception ignored) {
            }
        }
        if (ranks.isEmpty()) throw new IOException("No ranks in " + file);
        Pattern pat = compileTiktokenPattern(defaultPatternFor(name));
        return new Tiktoken(name, ranks, defaultSpecialsFor(name), pat);
    }

    private static String defaultPatternFor(String name) {
        if (CL100K_BASE.equals(name)) {
            return "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
        }
        if (O200K_BASE.equals(name) || O200K_HARMONY.equals(name)) {
            return "[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
        }
        // gpt2 / r50k / p50k
        return "'(?:[sdmt]|ll|ve|re)| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
    }

    private static Map<String, Integer> defaultSpecialsFor(String name) {
        Map<String, Integer> m = new LinkedHashMap<>();
        if (CL100K_BASE.equals(name)) {
            m.put("<|endoftext|>", 100257);
            m.put("<|fim_prefix|>", 100258);
            m.put("<|fim_middle|>", 100259);
            m.put("<|fim_suffix|>", 100260);
            m.put("<|endofprompt|>", 100276);
        } else if (O200K_BASE.equals(name) || O200K_HARMONY.equals(name)) {
            m.put("<|endoftext|>", 199999);
            m.put("<|endofprompt|>", 200018);
        } else if (P50K_EDIT.equals(name)) {
            m.put("<|endoftext|>", 50256);
            m.put("<|fim_prefix|>", 50281);
            m.put("<|fim_middle|>", 50282);
            m.put("<|fim_suffix|>", 50283);
        } else {
            m.put("<|endoftext|>", 50256);
        }
        return m;
    }

    // ---- Encode API (Python parity) ----

    /**
     * Python {@code encode_ordinary(text)} — never elevates special tokens; specials are
     * tokenized as ordinary UTF-8 bytes through the BPE path.
     */
    public int[] encodeOrdinary(String text) {
        if (text == null || text.isEmpty()) return new int[0];
        return encodeOrdinaryText(text);
    }

    /** Python {@code encode_ordinary_batch}. */
    public List<int[]> encodeOrdinaryBatch(List<String> texts) {
        List<int[]> out = new ArrayList<>(texts.size());
        for (String t : texts) out.add(encodeOrdinary(t));
        return out;
    }

    /**
     * Encode with default specials policy: {@code allowed_special=empty},
     * {@code disallowed_special="all"} — raises if a special token string appears in text.
     *
     * <p>Python: {@code encode(text)}.
     */
    public int[] encode(String text) {
        return encode(text, Set.of(), SPECIAL_ALL);
    }

    /**
     * Encode with {@code allowed_special} (Set of token strings, or {@link #SPECIAL_ALL}).
     * Disallowed defaults to all remaining specials.
     *
     * <p>Python: {@code encode(text, allowed_special=...)}.
     */
    public int[] encode(String text, Object allowedSpecial) {
        return encode(text, allowedSpecial, SPECIAL_ALL);
    }

    /**
     * Full Python {@code encode(text, allowed_special=..., disallowed_special=...)}.
     *
     * @param allowedSpecial    {@link #SPECIAL_ALL}, a {@code Set<String>}, or {@code null}/{@code empty}
     * @param disallowedSpecial {@link #SPECIAL_ALL}, a {@code Set<String>}, or empty to disable checks
     * @throws IllegalArgumentException if a disallowed special token string is present in {@code text}
     */
    public int[] encode(String text, Object allowedSpecial, Object disallowedSpecial) {
        if (text == null || text.isEmpty()) return new int[0];

        Set<String> allowed = resolveSpecialSet(allowedSpecial, /*allMeans*/ specialTokensSet);
        Set<String> disallowed = resolveSpecialSet(disallowedSpecial, /*allMeans*/ specialTokensSet);

        // Python: disallowed = disallowed - allowed
        if (!disallowed.isEmpty() && !allowed.isEmpty()) {
            Set<String> tmp = new HashSet<>(disallowed);
            tmp.removeAll(allowed);
            disallowed = tmp;
        }

        if (!disallowed.isEmpty()) {
            checkNoDisallowed(text, disallowed);
        }

        if (allowed.isEmpty()) {
            return encodeOrdinaryText(text);
        }
        return encodeWithAllowedSpecials(text, allowed);
    }

    /**
     * Convenience: encode returning an {@link Encoding} object (HF-style) for interop
     * with transformers pipelines. Uses ordinary encoding (no special elevation) unless
     * {@code allowedSpecials} is non-empty.
     */
    public Encoding encodeToEncoding(String text, boolean addSpecialTokens) {
        // addSpecialTokens in HF sense is not the same as tiktoken specials; for tiktoken
        // we treat it as allowed_special="all" when true, else ordinary.
        int[] ids = addSpecialTokens
                ? encode(text, SPECIAL_ALL, Set.of())
                : encodeOrdinary(text);
        return toEncoding(ids, text);
    }

    /** Backward-compatible: {@code encode(text, addSpecialTokens)} → Encoding. */
    public Encoding encode(String text, boolean addSpecialTokens) {
        return encodeToEncoding(text, addSpecialTokens);
    }

    /** Backward-compatible with explicit allowed specials set → Encoding. */
    public Encoding encode(String text, boolean addSpecialTokens, Set<String> allowedSpecials) {
        Set<String> allowed = allowedSpecials == null ? Set.of() : allowedSpecials;
        if (addSpecialTokens && allowed.isEmpty()) allowed = specialTokensSet;
        int[] ids = encode(text, allowed, Set.of()); // no raise
        return toEncoding(ids, text);
    }

    public Encoding encodePair(String a, String b, boolean addSpecialTokens) {
        Encoding ea = encodeToEncoding(a, false);
        Encoding eb = encodeToEncoding(b, false);
        int extra = 0;
        int eot = eotToken;
        if (addSpecialTokens && eot >= 0) extra = 1;
        int[] ids = new int[ea.size() + eb.size() + extra];
        int[] type = new int[ids.length];
        System.arraycopy(ea.ids(), 0, ids, 0, ea.size());
        Arrays.fill(type, 0, ea.size(), 0);
        int pos = ea.size();
        if (extra == 1) {
            ids[pos] = eot;
            type[pos] = 0;
            pos++;
        }
        System.arraycopy(eb.ids(), 0, ids, pos, eb.size());
        Arrays.fill(type, pos, ids.length, 1);
        List<String> toks = new ArrayList<>(ids.length);
        for (int id : ids) toks.add(idToTokenString(id));
        return Encoding.builder().ids(ids).typeIds(type).tokens(toks).build();
    }

    public List<Encoding> encodeBatch(List<String> texts, boolean addSpecial) {
        List<Encoding> out = new ArrayList<>(texts.size());
        for (String t : texts) out.add(encodeToEncoding(t, addSpecial));
        return out;
    }

    /** Python {@code encode_batch} → list of int[]. */
    public List<int[]> encodeBatchIds(List<String> texts) {
        List<int[]> out = new ArrayList<>(texts.size());
        for (String t : texts) out.add(encode(t));
        return out;
    }

    public List<Encoding> encodeBatchPairs(List<String> a, List<String> b, boolean addSpecial) {
        if (a.size() != b.size()) throw new IllegalArgumentException("a and b must have same size");
        List<Encoding> out = new ArrayList<>(a.size());
        for (int i = 0; i < a.size(); i++) out.add(encodePair(a.get(i), b.get(i), addSpecial));
        return out;
    }

    /**
     * Python {@code encode_single_token(piece)} — piece is a UTF-8 string or raw bytes
     * that must be <em>exactly</em> one token.
     */
    public int encodeSingleToken(String piece) {
        if (piece == null) throw new IllegalArgumentException("piece is null");
        Integer sp = specialTokens.get(piece);
        if (sp != null) return sp;
        byte[] utf8 = piece.getBytes(StandardCharsets.UTF_8);
        String key = new String(utf8, StandardCharsets.ISO_8859_1);
        Integer id = ranks.get(key);
        if (id == null) {
            throw new IllegalArgumentException(
                    "Encoding " + name + " does not contain sequence " + preview(piece));
        }
        return id;
    }

    public int encodeSingleToken(byte[] piece) {
        if (piece == null) throw new IllegalArgumentException("piece is null");
        String key = new String(piece, StandardCharsets.ISO_8859_1);
        Integer id = ranks.get(key);
        if (id != null) return id;
        // Try as UTF-8 special text
        String asText = new String(piece, StandardCharsets.UTF_8);
        Integer sp = specialTokens.get(asText);
        if (sp != null) return sp;
        throw new IllegalArgumentException(
                "Encoding " + name + " does not contain byte sequence of length " + piece.length);
    }

    // ---- Decode API ----

    /** Python {@code decode(ids)} — keeps specials as their text form. */
    public String decode(int[] ids) {
        return decode(ids, false);
    }

    /**
     * Decode token ids to String.
     *
     * @param skipSpecialTokens if true, drop special token ids (HF-style); Python tiktoken
     *                          has no skip flag — use {@link #decode(int[])} for pure parity.
     */
    public String decode(int[] ids, boolean skipSpecialTokens) {
        byte[] raw = decodeBytes(ids, skipSpecialTokens);
        return new String(raw, StandardCharsets.UTF_8);
    }

    /** Python {@code decode_bytes(ids)}. */
    public byte[] decodeBytes(int[] ids) {
        return decodeBytes(ids, false);
    }

    public byte[] decodeBytes(int[] ids, boolean skipSpecialTokens) {
        if (ids == null || ids.length == 0) return new byte[0];
        ByteArrayOutputStream bos = new ByteArrayOutputStream(ids.length * 4);
        for (int id : ids) {
            if (skipSpecialTokens && specialTokenIds.contains(id)) continue;
            String token = idToToken.get(id);
            if (token == null) continue;
            if (specialTokenIds.contains(id)) {
                // Specials are stored as Unicode text, not Latin-1 byte strings
                byte[] b = token.getBytes(StandardCharsets.UTF_8);
                bos.write(b, 0, b.length);
            } else {
                // Rank tokens: Latin-1 string == raw bytes
                for (int i = 0; i < token.length(); i++) {
                    bos.write((byte) token.charAt(i));
                }
            }
        }
        return bos.toByteArray();
    }

    /** Python {@code decode_batch}. */
    public List<String> decodeBatch(List<int[]> batch) {
        List<String> out = new ArrayList<>(batch.size());
        for (int[] ids : batch) out.add(decode(ids));
        return out;
    }

    /** Python {@code decode_bytes_batch}. */
    public List<byte[]> decodeBytesBatch(List<int[]> batch) {
        List<byte[]> out = new ArrayList<>(batch.size());
        for (int[] ids : batch) out.add(decodeBytes(ids));
        return out;
    }

    /** Python {@code decode_single_token_bytes(id)}. */
    public byte[] decodeSingleTokenBytes(int id) {
        String token = idToToken.get(id);
        if (token == null) {
            throw new IllegalArgumentException("Invalid token id " + id + " for " + name);
        }
        if (specialTokenIds.contains(id)) {
            return token.getBytes(StandardCharsets.UTF_8);
        }
        byte[] b = new byte[token.length()];
        for (int i = 0; i < token.length(); i++) b[i] = (byte) token.charAt(i);
        return b;
    }

    /** Python {@code token_byte_values()} — all mergeable rank byte sequences (no specials). */
    public List<byte[]> tokenByteValues() {
        // Stable by id order
        List<Map.Entry<Integer, String>> entries = new ArrayList<>();
        for (Map.Entry<String, Integer> e : ranks.entrySet()) {
            entries.add(Map.entry(e.getValue(), e.getKey()));
        }
        entries.sort(Comparator.comparingInt(Map.Entry::getKey));
        List<byte[]> out = new ArrayList<>(entries.size());
        for (Map.Entry<Integer, String> e : entries) {
            String t = e.getValue();
            byte[] b = new byte[t.length()];
            for (int i = 0; i < t.length(); i++) b[i] = (byte) t.charAt(i);
            out.add(b);
        }
        return out;
    }

    // ---- Internals: BPE ----

    private int[] encodeOrdinaryText(String text) {
        List<Integer> ids = new ArrayList<>();
        Matcher m = pattern.matcher(text);
        while (m.find()) {
            String chunk = m.group();
            byte[] utf8 = chunk.getBytes(StandardCharsets.UTF_8);
            bpeMergeInto(utf8, ids);
        }
        return toIntArray(ids);
    }

    private int[] encodeWithAllowedSpecials(String text, Set<String> allowed) {
        List<Integer> ids = new ArrayList<>();
        if (specialPattern == null || allowed.isEmpty()) {
            return encodeOrdinaryText(text);
        }
        // Build a matcher over only *allowed* specials (subset may be smaller)
        Pattern usePat = specialPattern;
        if (allowed.size() != specialTokensSet.size()) {
            List<String> sub = new ArrayList<>();
            for (String s : specialOrdered) {
                if (allowed.contains(s)) sub.add(s);
            }
            if (sub.isEmpty()) return encodeOrdinaryText(text);
            StringBuilder sp = new StringBuilder();
            for (int i = 0; i < sub.size(); i++) {
                if (i > 0) sp.append('|');
                sp.append(Pattern.quote(sub.get(i)));
            }
            usePat = Pattern.compile(sp.toString());
        }

        Matcher sm = usePat.matcher(text);
        int pos = 0;
        while (sm.find()) {
            if (sm.start() > pos) {
                appendOrdinary(text.substring(pos, sm.start()), ids);
            }
            String tok = sm.group();
            Integer id = specialTokens.get(tok);
            if (id != null && allowed.contains(tok)) {
                ids.add(id);
            } else {
                appendOrdinary(tok, ids);
            }
            pos = sm.end();
        }
        if (pos < text.length()) {
            appendOrdinary(text.substring(pos), ids);
        }
        return toIntArray(ids);
    }

    private void appendOrdinary(String text, List<Integer> ids) {
        if (text == null || text.isEmpty()) return;
        Matcher m = pattern.matcher(text);
        while (m.find()) {
            bpeMergeInto(m.group().getBytes(StandardCharsets.UTF_8), ids);
        }
    }

    /**
     * Greedy BPE merge on raw UTF-8 bytes. Rank = token id (lower merges first).
     * Matches openai/tiktoken byte-pair encode exactly.
     */
    private void bpeMergeInto(byte[] utf8, List<Integer> out) {
        if (utf8 == null || utf8.length == 0) return;

        // Fast path: whole sequence is one token
        String whole = new String(utf8, StandardCharsets.ISO_8859_1);
        Integer wholeId = ranks.get(whole);
        if (wholeId != null) {
            out.add(wholeId);
            return;
        }

        String[] parts = new String[utf8.length];
        for (int i = 0; i < utf8.length; i++) {
            parts[i] = String.valueOf((char) (utf8[i] & 0xFF));
        }

        while (parts.length > 1) {
            int bestIdx = -1;
            int bestRank = Integer.MAX_VALUE;
            for (int i = 0; i < parts.length - 1; i++) {
                String pair = parts[i] + parts[i + 1];
                Integer rank = ranks.get(pair);
                if (rank != null && rank < bestRank) {
                    bestRank = rank;
                    bestIdx = i;
                }
            }
            if (bestIdx < 0) break;

            String merged = parts[bestIdx] + parts[bestIdx + 1];
            String[] next = new String[parts.length - 1];
            System.arraycopy(parts, 0, next, 0, bestIdx);
            next[bestIdx] = merged;
            System.arraycopy(parts, bestIdx + 2, next, bestIdx + 1, parts.length - bestIdx - 2);
            parts = next;
        }

        for (String p : parts) {
            Integer id = ranks.get(p);
            // tiktoken guarantees single-byte coverage; missing → 0 is wrong, but shouldn't happen
            out.add(id != null ? id : 0);
        }
    }

    private void checkNoDisallowed(String text, Set<String> disallowed) {
        for (String sp : specialOrdered) {
            if (!disallowed.contains(sp)) continue;
            if (text.contains(sp)) {
                throw new IllegalArgumentException(
                        "Encountered text corresponding to disallowed special token '" + sp + "'.\n"
                                + "If you want this text to be encoded as a special token, "
                                + "pass it to `allowedSpecial`.\n"
                                + "If you want this text to be encoded as normal text, "
                                + "pass `disallowedSpecial=Set.of()`.\n"
                                + "To suppress this error, use encodeOrdinary(...).");
            }
        }
    }

    @SuppressWarnings("unchecked")
    private Set<String> resolveSpecialSet(Object spec, Set<String> allMeans) {
        if (spec == null) return Set.of();
        if (spec instanceof String s) {
            if (SPECIAL_ALL.equalsIgnoreCase(s) || "all".equalsIgnoreCase(s)) {
                return allMeans;
            }
            if (s.isEmpty()) return Set.of();
            throw new IllegalArgumentException(
                    "special spec must be \"all\", a Set<String>, or empty — got string '" + s + "'");
        }
        if (spec instanceof Set<?> set) {
            if (set.isEmpty()) return Set.of();
            Set<String> out = new HashSet<>();
            for (Object o : set) out.add(String.valueOf(o));
            return out;
        }
        if (spec instanceof Collection<?> col) {
            Set<String> out = new HashSet<>();
            for (Object o : col) out.add(String.valueOf(o));
            return out;
        }
        throw new IllegalArgumentException("Unsupported special spec type: " + spec.getClass());
    }

    private Encoding toEncoding(int[] ids, String text) {
        List<String> toks = new ArrayList<>(ids.length);
        List<Integer> offS = new ArrayList<>(ids.length);
        List<Integer> offE = new ArrayList<>(ids.length);
        int[] specMask = new int[ids.length];
        for (int i = 0; i < ids.length; i++) {
            toks.add(idToTokenString(ids[i]));
            offS.add(0);
            offE.add(text == null ? 0 : text.length());
            specMask[i] = specialTokenIds.contains(ids[i]) ? 1 : 0;
        }
        return Encoding.builder()
                .ids(ids)
                .tokens(toks)
                .offsetsStart(offS)
                .offsetsEnd(offE)
                .specialTokensMask(specMask)
                .build();
    }

    private String idToTokenString(int id) {
        String t = idToToken.get(id);
        return t == null ? "" : t;
    }

    private static int[] toIntArray(List<Integer> ids) {
        int[] a = new int[ids.size()];
        for (int i = 0; i < a.length; i++) a[i] = ids.get(i);
        return a;
    }

    private static String preview(String s) {
        if (s == null) return "null";
        return s.length() > 40 ? s.substring(0, 37) + "..." : s;
    }

    // ---- Accessors (Python property parity) ----

    public String name() { return name; }

    /** Python {@code n_vocab} — {@code max_token_value + 1}. */
    public int nVocab() { return nVocab; }

    /** Alias used by older call sites / HF-style. */
    public int vocabSize() { return nVocab; }

    public int maxTokenValue() { return maxTokenValue; }

    /** Python {@code eot_token}. */
    public int eotToken() { return eotToken; }

    public Set<String> specialTokensSet() { return specialTokensSet; }

    public Map<String, Integer> specialTokens() { return specialTokens; }

    public boolean isSpecialToken(int id) { return specialTokenIds.contains(id); }

    public int specialTokenId(String text) {
        Integer id = specialTokens.get(text);
        return id != null ? id : -1;
    }

    public int tokenToId(String token) {
        Integer sp = specialTokens.get(token);
        if (sp != null) return sp;
        // Interpret as UTF-8 text → raw bytes key
        byte[] utf8 = token.getBytes(StandardCharsets.UTF_8);
        Integer id = ranks.get(new String(utf8, StandardCharsets.ISO_8859_1));
        return id != null ? id : -1;
    }

    public String idToToken(int id) {
        return idToToken.get(id);
    }

    public Pattern pattern() { return pattern; }

    public TiktokenBpeModel bpeModel() { return bpeModel; }

    /** @deprecated use {@link #bpeModel()} — kept for source compatibility. */
    @Deprecated
    public BpeModel model() {
        // Legacy path constructed an empty-merges BpeModel; prefer TiktokenBpeModel.
        return new BpeModel(toBytesToUnicodeVocab(ranks), List.of(), null, "", "", false, false, false);
    }

    public Map<String, Integer> ranks() { return ranks; }

    // ---- FastTokenizer / transformers adapter ----

    /**
     * Wrap as a {@link FastTokenizer} for API compatibility with
     * {@code org.bytedeco.pytorch.utils.transformers} pipelines.
     *
     * <p>Pipeline: Split(tiktoken regex) → ByteLevel(use_regex=false) → TiktokenBpeModel
     * → ByteLevel decoder. No [CLS]/[SEP] template (tiktoken has none).
     */
    public FastTokenizer toFastTokenizer() {
        PreTokenizer pretok = new PreTokenizer.SequencePreTokenizer(List.of(
                new PreTokenizer.SplitPreTokenizer(pattern, RegexSplit.Behavior.ISOLATED, false),
                new PreTokenizer.ByteLevelPreTokenizer(false, false, true)
        ));
        String eot = specialTokens.containsKey("<|endoftext|>") ? "<|endoftext|>" : null;
        TokenizerPipeline pipe = new TokenizerPipeline(
                Normalizer.NOP,
                pretok,
                bpeModel,
                PostProcessor.NOP,
                Decoder.ByteLevelDecoder.INSTANCE,
                addedVocabulary,
                null, /* padding */
                null, /* truncation */
                eot,  /* unk */
                eot,  /* pad */
                null, /* cls */
                null, /* sep */
                eot,  /* bos */
                eot,  /* eos */
                null, /* mask */
                nVocab > 0 ? Math.min(nVocab, 1048576) : 16384,
                false
        );
        return FastTokenizer.of(pipe);
    }

    // ---- Minimal JSON object parser for model map (string values / nested string maps) ----

    /**
     * Parse a restricted JSON object: top-level object whose values are either strings
     * or objects of string→string. Good enough for model_to_encoding.json.
     */
    @SuppressWarnings("unchecked")
    static Map<String, Object> parseSimpleJsonObject(String json) {
        Map<String, Object> root = new LinkedHashMap<>();
        if (json == null) return root;
        String s = json.trim();
        if (s.isEmpty() || s.charAt(0) != '{') return root;
        int i = 1;
        int n = s.length();
        while (i < n) {
            i = skipWs(s, i);
            if (i < n && s.charAt(i) == '}') break;
            if (i < n && s.charAt(i) == ',') { i++; continue; }
            if (i >= n || s.charAt(i) != '"') break;
            String[] keyHold = new String[1];
            i = readJsonString(s, i, keyHold);
            i = skipWs(s, i);
            if (i >= n || s.charAt(i) != ':') break;
            i++;
            i = skipWs(s, i);
            if (i >= n) break;
            char c = s.charAt(i);
            if (c == '"') {
                String[] valHold = new String[1];
                i = readJsonString(s, i, valHold);
                root.put(keyHold[0], valHold[0]);
            } else if (c == '{') {
                // nested string→string map
                Map<String, String> nested = new LinkedHashMap<>();
                i++;
                while (i < n) {
                    i = skipWs(s, i);
                    if (i < n && s.charAt(i) == '}') { i++; break; }
                    if (i < n && s.charAt(i) == ',') { i++; continue; }
                    if (i >= n || s.charAt(i) != '"') break;
                    String[] nk = new String[1];
                    i = readJsonString(s, i, nk);
                    i = skipWs(s, i);
                    if (i >= n || s.charAt(i) != ':') break;
                    i++;
                    i = skipWs(s, i);
                    String[] nv = new String[1];
                    if (i < n && s.charAt(i) == '"') {
                        i = readJsonString(s, i, nv);
                        nested.put(nk[0], nv[0]);
                    } else {
                        break;
                    }
                }
                root.put(keyHold[0], nested);
            } else {
                break;
            }
        }
        return root;
    }

    private static int skipWs(String s, int i) {
        while (i < s.length()) {
            char c = s.charAt(i);
            if (c == ' ' || c == '\n' || c == '\r' || c == '\t') i++;
            else break;
        }
        return i;
    }

    private static int readJsonString(String s, int i, String[] out) {
        // s.charAt(i) == '"'
        StringBuilder sb = new StringBuilder();
        i++; // skip opening quote
        while (i < s.length()) {
            char c = s.charAt(i++);
            if (c == '"') break;
            if (c == '\\' && i < s.length()) {
                char n = s.charAt(i++);
                switch (n) {
                    case '"', '\\', '/' -> sb.append(n);
                    case 'b' -> sb.append('\b');
                    case 'f' -> sb.append('\f');
                    case 'n' -> sb.append('\n');
                    case 'r' -> sb.append('\r');
                    case 't' -> sb.append('\t');
                    case 'u' -> {
                        if (i + 4 <= s.length()) {
                            int cp = Integer.parseInt(s.substring(i, i + 4), 16);
                            sb.append((char) cp);
                            i += 4;
                        }
                    }
                    default -> sb.append(n);
                }
            } else {
                sb.append(c);
            }
        }
        out[0] = sb.toString();
        return i;
    }

    @Override
    public String toString() {
        return "Tiktoken{name=" + name + ", nVocab=" + nVocab
                + ", specials=" + specialTokensSet.size() + "}";
    }
}
