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

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Minimal deterministic tokenizer for offline factory tests / packing.
 *
 * <p>Encodes each UTF-8 byte as a token id in {@code [1, 255]} (0 reserved as
 * pad). Not a production BPE — production paths should inject
 * {@code FastTokenizer} / {@code AutoTokenizer}. Sufficient for collator,
 * packing, and tiny-train smoke loops.
 */
public final class SimpleTokenizer {

    public static final long PAD_ID = 0L;
    public static final long EOS_ID = 1L; // repurpose SOH
    public static final long BOS_ID = 2L;

    private final int vocabSize;
    private final boolean addEos;

    public SimpleTokenizer(int vocabSize, boolean addEos) {
        this.vocabSize = Math.max(256, vocabSize);
        this.addEos = addEos;
    }

    public static SimpleTokenizer defaults() {
        return new SimpleTokenizer(256, true);
    }

    public int vocabSize() { return vocabSize; }
    public long padTokenId() { return PAD_ID; }
    public long eosTokenId() { return EOS_ID; }
    public long bosTokenId() { return BOS_ID; }

    public long[] encode(String text) {
        return encode(text, addEos);
    }

    public long[] encode(String text, boolean withEos) {
        if (text == null || text.isEmpty()) {
            return withEos ? new long[]{EOS_ID} : new long[0];
        }
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        long[] ids = new long[bytes.length + (withEos ? 1 : 0)];
        for (int i = 0; i < bytes.length; i++) {
            int b = bytes[i] & 0xff;
            // keep 0 as pad; map NUL byte to a non-zero sentinel
            ids[i] = b == 0 ? 3L : (long) b;
        }
        if (withEos) {
            ids[bytes.length] = EOS_ID;
        }
        return ids;
    }

    public String decode(long[] ids) {
        if (ids == null || ids.length == 0) return "";
        List<Byte> bytes = new ArrayList<>(ids.length);
        for (long id : ids) {
            if (id == PAD_ID || id == EOS_ID || id == BOS_ID) continue;
            if (id > 0 && id < 256) {
                bytes.add((byte) id);
            }
        }
        byte[] arr = new byte[bytes.size()];
        for (int i = 0; i < bytes.size(); i++) arr[i] = bytes.get(i);
        return new String(arr, StandardCharsets.UTF_8);
    }

    /**
     * Tokenize a supervised prompt/response pair with prompt-length metadata
     * for label masking.
     */
    public Map<String, Object> encodeSupervised(String prompt, String response, int cutoff) {
        Objects.requireNonNull(prompt, "prompt");
        String resp = response == null ? "" : response;
        long[] pIds = encode(prompt, false);
        long[] rIds = encode(resp, true);
        int total = pIds.length + rIds.length;
        int keep = cutoff > 0 ? Math.min(total, cutoff) : total;
        long[] inputIds = new long[keep];
        long[] labels = new long[keep];
        int promptKeep = Math.min(pIds.length, keep);
        System.arraycopy(pIds, 0, inputIds, 0, promptKeep);
        for (int i = 0; i < promptKeep; i++) {
            labels[i] = -100L; // IGNORE
        }
        int rest = keep - promptKeep;
        if (rest > 0) {
            System.arraycopy(rIds, 0, inputIds, promptKeep, rest);
            System.arraycopy(rIds, 0, labels, promptKeep, rest);
        }
        Map<String, Object> feat = new LinkedHashMap<>();
        feat.put("input_ids", inputIds);
        feat.put("labels", labels);
        feat.put("prompt_len", promptKeep);
        long[] attn = new long[keep];
        for (int i = 0; i < keep; i++) attn[i] = 1L;
        feat.put("attention_mask", attn);
        return feat;
    }

    /** Encode plain text for continuous pre-training (labels == input_ids). */
    public Map<String, Object> encodePretrain(String text, int cutoff) {
        long[] ids = encode(text == null ? "" : text, true);
        int keep = cutoff > 0 ? Math.min(ids.length, cutoff) : ids.length;
        long[] inputIds = new long[keep];
        System.arraycopy(ids, 0, inputIds, 0, keep);
        long[] labels = inputIds.clone();
        long[] attn = new long[keep];
        for (int i = 0; i < keep; i++) attn[i] = 1L;
        Map<String, Object> feat = new LinkedHashMap<>();
        feat.put("input_ids", inputIds);
        feat.put("labels", labels);
        feat.put("attention_mask", attn);
        feat.put("prompt_len", 0);
        return feat;
    }

    public Map<String, Object> encodePairwise(
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
