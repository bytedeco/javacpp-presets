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
package org.bytedeco.pytorch.utils.tokenizers.normalizers;

import org.bytedeco.pytorch.utils.tokenizers.JsonMaps;

import java.text.Normalizer.Form;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * HuggingFace normalizer stage.
 */
@FunctionalInterface
public interface Normalizer {

    String normalize(String text);

    Normalizer NOP = text -> text == null ? "" : text;

    static Normalizer fromJson(Map<String, Object> m) {
        if (m == null) return NOP;
        String type = JsonMaps.asString(m.get("type"));
        if (type == null) return NOP;
        return switch (type) {
            case "Sequence" -> SequenceNormalizer.fromJson(m);
            case "NFD" -> new UnicodeNormalizer(Form.NFD);
            case "NFC" -> new UnicodeNormalizer(Form.NFC);
            case "NFKD" -> new UnicodeNormalizer(Form.NFKD);
            case "NFKC" -> new UnicodeNormalizer(Form.NFKC);
            case "Lowercase" -> LowercaseNormalizer.INSTANCE;
            case "Strip" -> StripNormalizer.fromJson(m);
            case "StripAccents" -> StripAccentsNormalizer.INSTANCE;
            case "Replace" -> ReplaceNormalizer.fromJson(m);
            case "BertNormalizer" -> BertNormalizer.fromJson(m);
            case "Nmt" -> NmtNormalizer.INSTANCE;
            case "Precompiled" -> NOP; // precompiled charsmap not supported; pass-through
            case "ByteLevel" -> NOP; // rare as normalizer
            default -> NOP;
        };
    }

    final class SequenceNormalizer implements Normalizer {
        private final List<Normalizer> normalizers;

        public SequenceNormalizer(List<Normalizer> normalizers) {
            this.normalizers = List.copyOf(Objects.requireNonNull(normalizers));
        }

        static SequenceNormalizer fromJson(Map<String, Object> m) {
            List<Object> raw = JsonMaps.asList(m.get("normalizers"));
            List<Normalizer> list = new ArrayList<>();
            if (raw != null) {
                for (Object o : raw) {
                    Map<String, Object> cm = JsonMaps.asMap(o);
                    if (cm != null) list.add(Normalizer.fromJson(cm));
                }
            }
            return new SequenceNormalizer(list);
        }

        @Override
        public String normalize(String text) {
            String cur = text == null ? "" : text;
            for (Normalizer n : normalizers) cur = n.normalize(cur);
            return cur;
        }
    }

    final class UnicodeNormalizer implements Normalizer {
        private final Form form;

        public UnicodeNormalizer(Form form) {
            this.form = form;
        }

        @Override
        public String normalize(String text) {
            if (text == null || text.isEmpty()) return text == null ? "" : text;
            return java.text.Normalizer.normalize(text, form);
        }
    }

    final class LowercaseNormalizer implements Normalizer {
        public static final LowercaseNormalizer INSTANCE = new LowercaseNormalizer();

        @Override
        public String normalize(String text) {
            return text == null ? "" : text.toLowerCase(java.util.Locale.ROOT);
        }
    }

    final class StripNormalizer implements Normalizer {
        private final boolean left;
        private final boolean right;

        public StripNormalizer(boolean left, boolean right) {
            this.left = left;
            this.right = right;
        }

        static StripNormalizer fromJson(Map<String, Object> m) {
            return new StripNormalizer(
                    JsonMaps.asBoolean(m, "left", true),
                    JsonMaps.asBoolean(m, "right", true));
        }

        @Override
        public String normalize(String text) {
            if (text == null) return "";
            int s = 0, e = text.length();
            if (left) {
                while (s < e && Character.isWhitespace(text.charAt(s))) s++;
            }
            if (right) {
                while (e > s && Character.isWhitespace(text.charAt(e - 1))) e--;
            }
            return text.substring(s, e);
        }
    }

    final class StripAccentsNormalizer implements Normalizer {
        public static final StripAccentsNormalizer INSTANCE = new StripAccentsNormalizer();

        @Override
        public String normalize(String text) {
            if (text == null || text.isEmpty()) return text == null ? "" : text;
            String nfd = java.text.Normalizer.normalize(text, Form.NFD);
            StringBuilder sb = new StringBuilder(nfd.length());
            for (int i = 0; i < nfd.length(); i++) {
                char c = nfd.charAt(i);
                if (Character.getType(c) != Character.NON_SPACING_MARK) sb.append(c);
            }
            return sb.toString();
        }
    }

    final class ReplaceNormalizer implements Normalizer {
        private final Pattern pattern;
        private final String content;

        public ReplaceNormalizer(Pattern pattern, String content) {
            this.pattern = pattern;
            this.content = content == null ? "" : content;
        }

        static ReplaceNormalizer fromJson(Map<String, Object> m) {
            Object patObj = m.get("pattern");
            String pat;
            if (patObj instanceof Map<?, ?> pm) {
                Object r = pm.get("Regex");
                if (r == null) r = pm.get("String");
                pat = r == null ? null : String.valueOf(r);
            } else {
                pat = JsonMaps.asString(patObj);
            }
            if (pat == null) pat = "";
            String content = JsonMaps.asString(m.get("content"));
            if (content == null) content = "";
            return new ReplaceNormalizer(Pattern.compile(pat), content);
        }

        @Override
        public String normalize(String text) {
            if (text == null) return "";
            Matcher matcher = pattern.matcher(text);
            return matcher.replaceAll(content);
        }
    }

    /** Minimal NMT normalizer: map various spaces / control-ish chars toward ASCII space. */
    final class NmtNormalizer implements Normalizer {
        public static final NmtNormalizer INSTANCE = new NmtNormalizer();

        @Override
        public String normalize(String text) {
            if (text == null || text.isEmpty()) return text == null ? "" : text;
            StringBuilder sb = new StringBuilder(text.length());
            for (int i = 0; i < text.length(); ) {
                int cp = text.codePointAt(i);
                i += Character.charCount(cp);
                if (cp == 0x0001 || cp == 0x0002 || (cp >= 0x0003 && cp <= 0x0008)
                        || cp == 0x000B || (cp >= 0x000E && cp <= 0x001F)
                        || (cp >= 0x007F && cp <= 0x009F)
                        || cp == 0x00A0 || cp == 0x200B || cp == 0xFFFD) {
                    // drop or map like HF nmt roughly — map NBSP to space, drop controls
                    if (cp == 0x00A0) sb.append(' ');
                    continue;
                }
                sb.appendCodePoint(cp);
            }
            return sb.toString();
        }
    }

    final class BertNormalizer implements Normalizer {
        private final boolean cleanText;
        private final boolean handleChineseChars;
        private final boolean stripAccents;
        private final boolean lowercase;

        public BertNormalizer(boolean cleanText, boolean handleChineseChars,
                              boolean stripAccents, boolean lowercase) {
            this.cleanText = cleanText;
            this.handleChineseChars = handleChineseChars;
            this.stripAccents = stripAccents;
            this.lowercase = lowercase;
        }

        static BertNormalizer fromJson(Map<String, Object> m) {
            boolean lowercase = JsonMaps.asBoolean(m, "lowercase", true);
            // HF: strip_accents == null → default to lowercase
            boolean stripAccents;
            if (!m.containsKey("strip_accents") || m.get("strip_accents") == null) {
                stripAccents = lowercase;
            } else {
                stripAccents = JsonMaps.asBoolean(m, "strip_accents", true);
            }
            return new BertNormalizer(
                    JsonMaps.asBoolean(m, "clean_text", true),
                    JsonMaps.asBoolean(m, "handle_chinese_chars", true),
                    stripAccents,
                    lowercase);
        }

        @Override
        public String normalize(String text) {
            if (text == null) return "";
            String s = text;
            if (cleanText) s = clean(s);
            if (handleChineseChars) s = tokenizeChinese(s);
            if (stripAccents) s = StripAccentsNormalizer.INSTANCE.normalize(s);
            if (lowercase) s = s.toLowerCase(java.util.Locale.ROOT);
            return s;
        }

        private static String clean(String text) {
            StringBuilder sb = new StringBuilder(text.length());
            for (int i = 0; i < text.length(); ) {
                int cp = text.codePointAt(i);
                i += Character.charCount(cp);
                if (cp == 0 || cp == 0xfffd || isControl(cp)) continue;
                if (Character.isWhitespace(cp)) sb.append(' ');
                else sb.appendCodePoint(cp);
            }
            return sb.toString();
        }

        private static boolean isControl(int cp) {
            if (cp == '\t' || cp == '\n' || cp == '\r') return false;
            int type = Character.getType(cp);
            return type == Character.CONTROL || type == Character.FORMAT
                    || type == Character.PRIVATE_USE || type == Character.SURROGATE
                    || type == Character.UNASSIGNED;
        }

        private static String tokenizeChinese(String text) {
            StringBuilder sb = new StringBuilder(text.length() * 2);
            for (int i = 0; i < text.length(); ) {
                int cp = text.codePointAt(i);
                i += Character.charCount(cp);
                if (isChinese(cp)) {
                    sb.append(' ').appendCodePoint(cp).append(' ');
                } else {
                    sb.appendCodePoint(cp);
                }
            }
            return sb.toString();
        }

        private static boolean isChinese(int cp) {
            return (cp >= 0x4E00 && cp <= 0x9FFF)
                    || (cp >= 0x3400 && cp <= 0x4DBF)
                    || (cp >= 0x20000 && cp <= 0x2A6DF);
        }
    }
}
