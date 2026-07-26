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
package org.bytedeco.pytorch.utils.tokenizers.decoders;

import org.bytedeco.pytorch.utils.tokenizers.BytesToUnicode;
import org.bytedeco.pytorch.utils.tokenizers.JsonMaps;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Pattern;

/**
 * HuggingFace decoder stage: token strings → final text.
 */
@FunctionalInterface
public interface Decoder {

    String decode(List<String> tokens);

    Decoder FUSE = tokens -> {
        if (tokens == null || tokens.isEmpty()) return "";
        StringBuilder sb = new StringBuilder();
        for (String t : tokens) if (t != null) sb.append(t);
        return sb.toString();
    };

    static Decoder fromJson(Map<String, Object> m) {
        if (m == null) return FUSE;
        String type = JsonMaps.asString(m.get("type"));
        if (type == null) return FUSE;
        return switch (type) {
            case "Sequence" -> SequenceDecoder.fromJson(m);
            case "ByteLevel" -> ByteLevelDecoder.INSTANCE;
            case "Metaspace" -> MetaspaceDecoder.fromJson(m);
            case "WordPiece" -> WordPieceDecoder.fromJson(m);
            case "BPEDecoder" -> BPEDecoder.fromJson(m);
            case "CTC" -> CtcDecoder.fromJson(m);
            case "Fuse" -> FUSE;
            case "Strip" -> StripDecoder.fromJson(m);
            case "Replace" -> ReplaceDecoder.fromJson(m);
            case "ByteFallback" -> ByteFallbackDecoder.INSTANCE;
            default -> FUSE;
        };
    }

    final class SequenceDecoder implements Decoder {
        private final List<Decoder> decoders;

        public SequenceDecoder(List<Decoder> decoders) {
            this.decoders = List.copyOf(Objects.requireNonNull(decoders));
        }

        static SequenceDecoder fromJson(Map<String, Object> m) {
            List<Object> raw = JsonMaps.asList(m.get("decoders"));
            List<Decoder> list = new ArrayList<>();
            if (raw != null) {
                for (Object o : raw) {
                    Map<String, Object> cm = JsonMaps.asMap(o);
                    if (cm != null) list.add(Decoder.fromJson(cm));
                }
            }
            return new SequenceDecoder(list);
        }

        @Override
        public String decode(List<String> tokens) {
            // HF Sequence decoder: first decoder may transform token list conceptually;
            // practically most chains start with Fuse or ByteLevel operating on joined pieces.
            // We apply each decoder to the running string by re-tokenizing as single piece after first.
            if (decoders.isEmpty()) return FUSE.decode(tokens);
            // First decoder gets the token list
            String cur = decoders.get(0).decode(tokens);
            for (int i = 1; i < decoders.size(); i++) {
                cur = decoders.get(i).decode(List.of(cur));
            }
            return cur;
        }
    }

    final class ByteLevelDecoder implements Decoder {
        public static final ByteLevelDecoder INSTANCE = new ByteLevelDecoder();

        @Override
        public String decode(List<String> tokens) {
            return BytesToUnicode.byteDecodeTokens(tokens);
        }
    }

    final class MetaspaceDecoder implements Decoder {
        private final String replacement;
        private final boolean addPrefixSpace;

        public MetaspaceDecoder(String replacement, boolean addPrefixSpace) {
            this.replacement = replacement == null || replacement.isEmpty() ? "▁" : replacement;
            this.addPrefixSpace = addPrefixSpace;
        }

        static MetaspaceDecoder fromJson(Map<String, Object> m) {
            return new MetaspaceDecoder(
                    JsonMaps.asString(m.get("replacement")),
                    JsonMaps.asBoolean(m, "add_prefix_space", true));
        }

        @Override
        public String decode(List<String> tokens) {
            if (tokens == null || tokens.isEmpty()) return "";
            StringBuilder sb = new StringBuilder();
            for (String t : tokens) {
                if (t == null) continue;
                sb.append(t.replace(replacement, " "));
            }
            String s = sb.toString();
            if (addPrefixSpace && s.startsWith(" ")) {
                s = s.substring(1);
            }
            return s;
        }
    }

    final class WordPieceDecoder implements Decoder {
        private final String prefix;
        private final boolean cleanup;

        public WordPieceDecoder(String prefix, boolean cleanup) {
            this.prefix = prefix == null ? "##" : prefix;
            this.cleanup = cleanup;
        }

        static WordPieceDecoder fromJson(Map<String, Object> m) {
            return new WordPieceDecoder(
                    JsonMaps.asString(m.get("prefix")),
                    JsonMaps.asBoolean(m, "cleanup", true));
        }

        @Override
        public String decode(List<String> tokens) {
            if (tokens == null || tokens.isEmpty()) return "";
            StringBuilder sb = new StringBuilder();
            boolean first = true;
            for (String t : tokens) {
                if (t == null) continue;
                if (t.startsWith(prefix)) {
                    sb.append(t.substring(prefix.length()));
                } else {
                    if (!first) sb.append(' ');
                    sb.append(t);
                }
                first = false;
            }
            String s = sb.toString();
            if (cleanup) {
                // HF tokenizers WordPiece cleanup (approximate):
                // collapse space-before-punct, keep space around apostrophe pieces.
                s = s.replace(" .", ".")
                        .replace(" ?", "?")
                        .replace(" !", "!")
                        .replace(" ,", ",")
                        .replace(" n't", "n't")
                        .replace(" 'm", "'m")
                        .replace(" 's", "'s")
                        .replace(" 've", "'ve")
                        .replace(" 're", "'re")
                        .replace(" 'd", "'d")
                        .replace(" 'll", "'ll");
                // NOTE: do NOT collapse " ' " → "'" — HF keeps "i ' m"
            }
            return s;
        }
    }

    final class BPEDecoder implements Decoder {
        private final String suffix;

        public BPEDecoder(String suffix) {
            this.suffix = suffix == null ? "</w>" : suffix;
        }

        static BPEDecoder fromJson(Map<String, Object> m) {
            return new BPEDecoder(JsonMaps.asString(m.get("suffix")));
        }

        @Override
        public String decode(List<String> tokens) {
            if (tokens == null || tokens.isEmpty()) return "";
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < tokens.size(); i++) {
                String t = tokens.get(i);
                if (t == null) continue;
                if (t.endsWith(suffix)) {
                    sb.append(t, 0, t.length() - suffix.length());
                    if (i + 1 < tokens.size()) sb.append(' ');
                } else {
                    sb.append(t);
                }
            }
            return sb.toString();
        }
    }

    final class CtcDecoder implements Decoder {
        private final String padToken;
        private final String wordDelimiter;
        private final boolean cleanup;

        public CtcDecoder(String padToken, String wordDelimiter, boolean cleanup) {
            this.padToken = padToken == null ? "<pad>" : padToken;
            this.wordDelimiter = wordDelimiter == null ? "|" : wordDelimiter;
            this.cleanup = cleanup;
        }

        static CtcDecoder fromJson(Map<String, Object> m) {
            return new CtcDecoder(
                    JsonMaps.asString(m.get("pad_token")),
                    JsonMaps.asString(m.get("word_delimiter_token")),
                    JsonMaps.asBoolean(m, "cleanup", true));
        }

        @Override
        public String decode(List<String> tokens) {
            if (tokens == null || tokens.isEmpty()) return "";
            List<String> collapsed = new ArrayList<>();
            String prev = null;
            for (String t : tokens) {
                if (t == null || t.equals(padToken)) continue;
                if (t.equals(prev)) continue;
                collapsed.add(t);
                prev = t;
            }
            StringBuilder sb = new StringBuilder();
            for (String t : collapsed) {
                if (t.equals(wordDelimiter)) sb.append(' ');
                else sb.append(t);
            }
            String s = sb.toString();
            if (cleanup) s = s.replaceAll("\\s+", " ").trim();
            return s;
        }
    }

    final class StripDecoder implements Decoder {
        private final char content;
        private final int start;
        private final int stop;

        public StripDecoder(char content, int start, int stop) {
            this.content = content;
            this.start = start;
            this.stop = stop;
        }

        static StripDecoder fromJson(Map<String, Object> m) {
            String c = JsonMaps.asString(m.get("content"));
            char ch = (c == null || c.isEmpty()) ? ' ' : c.charAt(0);
            Integer s = JsonMaps.asInt(m.get("start"));
            Integer e = JsonMaps.asInt(m.get("stop"));
            return new StripDecoder(ch, s == null ? 0 : s, e == null ? 0 : e);
        }

        @Override
        public String decode(List<String> tokens) {
            String s = FUSE.decode(tokens);
            int a = 0, b = s.length();
            int stripped = 0;
            while (a < b && stripped < start && s.charAt(a) == content) { a++; stripped++; }
            stripped = 0;
            while (b > a && stripped < stop && s.charAt(b - 1) == content) { b--; stripped++; }
            return s.substring(a, b);
        }
    }

    final class ReplaceDecoder implements Decoder {
        private final Pattern pattern;
        private final String content;

        public ReplaceDecoder(String pattern, String content) {
            this.pattern = Pattern.compile(pattern == null ? "" : pattern, Pattern.LITERAL);
            this.content = content == null ? "" : content;
        }

        static Decoder fromJson(Map<String, Object> m) {
            Object patObj = m.get("pattern");
            String pat;
            if (patObj instanceof Map<?, ?> pm) {
                Object r = pm.get("String");
                if (r == null) r = pm.get("Regex");
                pat = r == null ? "" : String.valueOf(r);
                // If Regex key present, compile as regex not literal
                if (pm.get("Regex") != null) {
                    return new ReplaceDecoderRegex(pat, JsonMaps.asString(m.get("content")));
                }
            } else {
                pat = JsonMaps.asString(patObj);
            }
            return new ReplaceDecoder(pat == null ? "" : pat, JsonMaps.asString(m.get("content")));
        }

        @Override
        public String decode(List<String> tokens) {
            String s = FUSE.decode(tokens);
            return pattern.matcher(s).replaceAll(MatcherQuote(content));
        }

        private static String MatcherQuote(String c) {
            // escape replacement so $ \ are literal
            return java.util.regex.Matcher.quoteReplacement(c == null ? "" : c);
        }
    }

    final class ReplaceDecoderRegex implements Decoder {
        private final Pattern pattern;
        private final String content;

        ReplaceDecoderRegex(String pattern, String content) {
            this.pattern = Pattern.compile(pattern == null ? "" : pattern);
            this.content = content == null ? "" : content;
        }

        @Override
        public String decode(List<String> tokens) {
            String s = FUSE.decode(tokens);
            return pattern.matcher(s).replaceAll(java.util.regex.Matcher.quoteReplacement(content));
        }
    }

    /** Decode {@code <0xHH>} byte tokens back into bytes/UTF-8; pass other tokens through. */
    final class ByteFallbackDecoder implements Decoder {
        public static final ByteFallbackDecoder INSTANCE = new ByteFallbackDecoder();
        private static final Pattern HEX = Pattern.compile("<0x([0-9A-Fa-f]{2})>");

        @Override
        public String decode(List<String> tokens) {
            if (tokens == null || tokens.isEmpty()) return "";
            StringBuilder text = new StringBuilder();
            List<Byte> bytes = new ArrayList<>();
            for (String t : tokens) {
                if (t == null) continue;
                var m = HEX.matcher(t);
                if (m.matches()) {
                    bytes.add((byte) Integer.parseInt(m.group(1), 16));
                } else {
                    flushBytes(bytes, text);
                    text.append(t);
                }
            }
            flushBytes(bytes, text);
            return text.toString();
        }

        private static void flushBytes(List<Byte> bytes, StringBuilder text) {
            if (bytes.isEmpty()) return;
            byte[] raw = new byte[bytes.size()];
            for (int i = 0; i < bytes.size(); i++) raw[i] = bytes.get(i);
            bytes.clear();
            text.append(new String(raw, java.nio.charset.StandardCharsets.UTF_8));
        }
    }
}
