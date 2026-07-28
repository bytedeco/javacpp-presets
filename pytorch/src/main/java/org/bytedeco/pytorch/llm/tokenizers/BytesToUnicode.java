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

import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;

/**
 * GPT-2 {@code bytes_to_unicode} / {@code unicode_to_bytes} mapping used by
 * HuggingFace ByteLevel BPE (GPT-2, Llama-3, Qwen, DeepSeek, …).
 *
 * <p>Space byte {@code 0x20} maps to {@code Ġ} (U+0120) — never use raw
 * {@code (char) b} for byte-level tokens.
 */
public final class BytesToUnicode {

    private static final char[] BYTE_ENCODER = new char[256];
    private static final int[] BYTE_DECODER = new int[1 << 16]; // codepoint → byte, -1 if unused
    private static final Map<Character, Integer> DECODER_MAP = new HashMap<>(512);

    static {
        // Replicates OpenAI / HF tokenizers bytes_to_unicode exactly.
        // GPT-2 algorithm: start with printable ranges, then append the rest mapped to 256+
        java.util.List<Integer> bs = new java.util.ArrayList<>(256);
        for (int i = '!'; i <= '~'; i++) bs.add(i);           // 33..126
        for (int i = 0xA1; i <= 0xAC; i++) bs.add(i);       // 161..172
        for (int i = 0xAE; i <= 0xFF; i++) bs.add(i);       // 174..255
        java.util.List<Integer> cs = new java.util.ArrayList<>(bs);
        int nExtra = 0;
        java.util.HashSet<Integer> present = new java.util.HashSet<>(bs);
        for (int b = 0; b < 256; b++) {
            if (!present.contains(b)) {
                bs.add(b);
                cs.add(256 + nExtra);
                nExtra++;
            }
        }
        if (bs.size() != 256) {
            throw new IllegalStateException("bytes_to_unicode size=" + bs.size());
        }
        for (int i = 0; i < 256; i++) {
            int b = bs.get(i);
            int c = cs.get(i);
            BYTE_ENCODER[b] = (char) c;
            DECODER_MAP.put((char) c, b);
        }
        // Fill sparse decoder array for fast path on BMP
        for (int i = 0; i < BYTE_DECODER.length; i++) BYTE_DECODER[i] = -1;
        for (Map.Entry<Character, Integer> e : DECODER_MAP.entrySet()) {
            char c = e.getKey();
            if (c < BYTE_DECODER.length) {
                BYTE_DECODER[c] = e.getValue();
            }
        }
    }

    private BytesToUnicode() {}

    /** Map a single byte (0–255) to its GPT-2 unicode char. */
    public static char encodeByte(int b) {
        return BYTE_ENCODER[b & 0xff];
    }

    /** Map a GPT-2 unicode char back to a byte, or -1 if not in the map. */
    public static int decodeChar(char c) {
        if (c < BYTE_DECODER.length) {
            int v = BYTE_DECODER[c];
            if (v >= 0) return v;
        }
        Integer v = DECODER_MAP.get(c);
        return v == null ? -1 : v;
    }

    /**
     * Encode a UTF-8 string piece into the GPT-2 unicode string that BPE sees
     * (one char per input byte).
     */
    public static String byteEncode(String text) {
        if (text == null || text.isEmpty()) return "";
        byte[] raw = text.getBytes(StandardCharsets.UTF_8);
        char[] out = new char[raw.length];
        for (int i = 0; i < raw.length; i++) {
            out[i] = BYTE_ENCODER[raw[i] & 0xff];
        }
        return new String(out);
    }

    /**
     * Decode a GPT-2 unicode string (one char per byte) back to a UTF-8 Java string.
     * Unknown chars are skipped.
     */
    public static String byteDecode(String encoded) {
        if (encoded == null || encoded.isEmpty()) return "";
        byte[] raw = new byte[encoded.length()];
        int n = 0;
        for (int i = 0; i < encoded.length(); i++) {
            int b = decodeChar(encoded.charAt(i));
            if (b >= 0) raw[n++] = (byte) b;
        }
        return new String(raw, 0, n, StandardCharsets.UTF_8);
    }

    /** Decode a list of GPT-2 unicode token strings concatenated. */
    public static String byteDecodeTokens(Iterable<String> tokens) {
        if (tokens == null) return "";
        StringBuilder sb = new StringBuilder();
        for (String t : tokens) {
            if (t != null) sb.append(t);
        }
        return byteDecode(sb.toString());
    }

    /** Space maps to Ġ — handy for tests. */
    public static char spaceChar() {
        return BYTE_ENCODER[0x20];
    }
}
