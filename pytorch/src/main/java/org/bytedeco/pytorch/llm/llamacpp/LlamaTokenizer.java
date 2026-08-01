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

package org.bytedeco.pytorch.llm.llamacpp;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Lightweight tokenizer for in-process GGUF paths.
 * Prefers {@code tokenizer.ggml.tokens} from metadata when present;
 * otherwise uses a stable hash/whitespace fallback suitable for tiny models.
 */
public final class LlamaTokenizer {

    private final Map<String, Integer> tokenToId = new HashMap<>();
    private final List<String> idToToken = new ArrayList<>();
    private final int vocabSize;
    private final int bosId;
    private final int eosId;
    private final int padId;
    private final boolean hasVocab;

    public LlamaTokenizer(LlamaHParams hparams, Map<String, Object> metadata) {
        Objects.requireNonNull(hparams, "hparams");
        this.vocabSize = Math.max(1, hparams.nVocab());
        Object tokensObj = metadata != null ? metadata.get("tokenizer.ggml.tokens") : null;
        if (tokensObj instanceof List<?> list && !list.isEmpty()) {
            int i = 0;
            for (Object t : list) {
                String s = String.valueOf(t);
                tokenToId.put(s, i);
                idToToken.add(s);
                i++;
                if (i >= vocabSize) break;
            }
            while (idToToken.size() < vocabSize) {
                String s = "<unk_" + idToToken.size() + ">";
                tokenToId.putIfAbsent(s, idToToken.size());
                idToToken.add(s);
            }
            hasVocab = true;
        } else {
            for (int i = 0; i < Math.min(vocabSize, 256); i++) {
                String s = "t" + i;
                tokenToId.put(s, i);
                idToToken.add(s);
            }
            while (idToToken.size() < vocabSize) {
                String s = "<id_" + idToToken.size() + ">";
                tokenToId.put(s, idToToken.size());
                idToToken.add(s);
            }
            hasVocab = false;
        }
        this.bosId = intMeta(metadata, "tokenizer.ggml.bos_token_id", 0);
        this.eosId = intMeta(metadata, "tokenizer.ggml.eos_token_id", Math.min(1, vocabSize - 1));
        this.padId = intMeta(metadata, "tokenizer.ggml.padding_token_id", eosId);
    }

    public static LlamaTokenizer tiny(LlamaHParams hp) {
        return new LlamaTokenizer(hp, Map.of());
    }

    public int vocabSize() { return vocabSize; }
    public int bosId() { return bosId; }
    public int eosId() { return eosId; }
    public int padId() { return padId; }
    public boolean hasVocab() { return hasVocab; }

    public int[] encode(String text, boolean addBos) {
        List<Integer> ids = new ArrayList<>();
        if (addBos) ids.add(bosId);
        if (text == null || text.isEmpty()) {
            return toArray(ids);
        }
        if (hasVocab) {
            // greedy longest-match over known tokens, else char fallback
            int i = 0;
            while (i < text.length()) {
                int matched = -1;
                int matchLen = 0;
                // limit scan window
                int maxLook = Math.min(text.length() - i, 32);
                for (int len = maxLook; len >= 1; len--) {
                    String sub = text.substring(i, i + len);
                    Integer id = tokenToId.get(sub);
                    if (id != null) {
                        matched = id;
                        matchLen = len;
                        break;
                    }
                }
                if (matched >= 0) {
                    ids.add(matched);
                    i += matchLen;
                } else {
                    ids.add(Math.floorMod(text.charAt(i), vocabSize));
                    i++;
                }
            }
        } else {
            String[] parts = text.trim().split("\\s+");
            for (String p : parts) {
                if (p.isEmpty()) continue;
                ids.add(Math.floorMod(stableHash(p), vocabSize));
            }
            if (ids.isEmpty() || (addBos && ids.size() == 1)) {
                // ensure at least one content token
                byte[] raw = text.getBytes(StandardCharsets.UTF_8);
                for (int i = 0; i < Math.min(raw.length, 64); i++) {
                    ids.add(Math.floorMod(raw[i] & 0xff, vocabSize));
                }
            }
        }
        return toArray(ids);
    }

    public String decode(int[] ids) {
        if (ids == null || ids.length == 0) return "";
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < ids.length; i++) {
            int id = ids[i];
            if (id < 0 || id >= idToToken.size()) {
                sb.append("<").append(id).append(">");
            } else {
                String t = idToToken.get(id);
                if (i > 0 && hasVocab && !t.startsWith(" ") && !t.startsWith("\n")
                        && sb.length() > 0 && !Character.isWhitespace(sb.charAt(sb.length() - 1))) {
                    // piece tokens often include leading space already
                }
                sb.append(t.startsWith("t") && !hasVocab ? (i > 0 ? " " : "") + t : t);
            }
        }
        return sb.toString();
    }

    public String decodeNew(int[] full, int promptLen) {
        if (full == null || full.length <= promptLen) return "";
        int[] gen = new int[full.length - promptLen];
        System.arraycopy(full, promptLen, gen, 0, gen.length);
        return decode(gen);
    }

    private static int[] toArray(List<Integer> ids) {
        int[] a = new int[ids.size()];
        for (int i = 0; i < ids.size(); i++) a[i] = ids.get(i);
        return a;
    }

    private static int stableHash(String s) {
        int h = 0;
        for (int i = 0; i < s.length(); i++) h = 31 * h + s.charAt(i);
        return h;
    }

    private static int intMeta(Map<String, Object> m, String k, int def) {
        if (m == null) return def;
        Object v = m.get(k);
        if (v instanceof Number n) return n.intValue();
        if (v != null) {
            try { return Integer.parseInt(String.valueOf(v)); } catch (Exception ignored) {}
        }
        return def;
    }
}
