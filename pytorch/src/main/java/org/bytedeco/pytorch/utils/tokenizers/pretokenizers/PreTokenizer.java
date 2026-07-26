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
package org.bytedeco.pytorch.utils.tokenizers.pretokenizers;

import org.bytedeco.pytorch.utils.tokenizers.BytesToUnicode;
import org.bytedeco.pytorch.utils.tokenizers.JsonMaps;
import org.bytedeco.pytorch.utils.tokenizers.RegexSplit;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * HuggingFace pre-tokenizer stage: text → list of {@link PreToken} spans.
 */
@FunctionalInterface
public interface PreTokenizer {

    List<PreToken> preTokenize(String text);

    PreTokenizer NOP = text -> {
        if (text == null || text.isEmpty()) return List.of();
        return List.of(new PreToken(text, 0, text.length()));
    };

    static PreTokenizer fromJson(Map<String, Object> m) {
        if (m == null) return NOP;
        String type = JsonMaps.asString(m.get("type"));
        if (type == null) return NOP;
        return switch (type) {
            case "Sequence" -> SequencePreTokenizer.fromJson(m);
            case "ByteLevel" -> ByteLevelPreTokenizer.fromJson(m);
            case "Split" -> SplitPreTokenizer.fromJson(m);
            case "Metaspace" -> MetaspacePreTokenizer.fromJson(m);
            case "Whitespace" -> WhitespacePreTokenizer.INSTANCE;
            case "WhitespaceSplit" -> WhitespaceSplitPreTokenizer.INSTANCE;
            case "Punctuation" -> PunctuationPreTokenizer.fromJson(m);
            case "Digits" -> DigitsPreTokenizer.fromJson(m);
            case "BertPreTokenizer" -> BertPreTokenizer.INSTANCE;
            case "CharDelimiterSplit" -> CharDelimiterSplitPreTokenizer.fromJson(m);
            case "Delimiter" -> CharDelimiterSplitPreTokenizer.fromJson(m);
            default -> NOP;
        };
    }

    // ---- implementations ----------------------------------------------------

    final class SequencePreTokenizer implements PreTokenizer {
        private final List<PreTokenizer> pretokens;

        public SequencePreTokenizer(List<PreTokenizer> pretokens) {
            this.pretokens = List.copyOf(Objects.requireNonNull(pretokens));
        }

        static SequencePreTokenizer fromJson(Map<String, Object> m) {
            List<Object> raw = JsonMaps.asList(m.get("pretokenizers"));
            List<PreTokenizer> list = new ArrayList<>();
            if (raw != null) {
                for (Object o : raw) {
                    Map<String, Object> cm = JsonMaps.asMap(o);
                    if (cm != null) list.add(PreTokenizer.fromJson(cm));
                }
            }
            return new SequencePreTokenizer(list);
        }

        @Override
        public List<PreToken> preTokenize(String text) {
            List<PreToken> cur = List.of(new PreToken(text == null ? "" : text, 0,
                    text == null ? 0 : text.length()));
            for (PreTokenizer pt : pretokens) {
                List<PreToken> next = new ArrayList<>();
                for (PreToken p : cur) {
                    if (p.added()) {
                        next.add(p);
                        continue;
                    }
                    // Offsets from sub-pretokenizers are relative to the piece;
                    // shift them back to the parent span.
                    List<PreToken> sub = pt.preTokenize(p.value());
                    for (PreToken s : sub) {
                        next.add(new PreToken(s.value(),
                                p.start() + s.start(),
                                p.start() + s.end(),
                                s.added(), s.addedId()));
                    }
                }
                cur = next;
            }
            return cur;
        }
    }

    /**
     * GPT-2 style ByteLevel: optional regex split, then map each UTF-8 byte through
     * {@link BytesToUnicode}.
     */
    final class ByteLevelPreTokenizer implements PreTokenizer {
        // HF default GPT-2 pattern
        private static final Pattern GPT2 = Pattern.compile(
                "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
        );

        private final boolean addPrefixSpace;
        private final boolean useRegex;
        private final boolean trimOffsets; // stored for post-processor; pretok ignores

        public ByteLevelPreTokenizer(boolean addPrefixSpace, boolean useRegex, boolean trimOffsets) {
            this.addPrefixSpace = addPrefixSpace;
            this.useRegex = useRegex;
            this.trimOffsets = trimOffsets;
        }

        static ByteLevelPreTokenizer fromJson(Map<String, Object> m) {
            return new ByteLevelPreTokenizer(
                    JsonMaps.asBoolean(m, "add_prefix_space", true),
                    JsonMaps.asBoolean(m, "use_regex", true),
                    JsonMaps.asBoolean(m, "trim_offsets", true));
        }

        public boolean addPrefixSpace() { return addPrefixSpace; }
        public boolean trimOffsets() { return trimOffsets; }

        @Override
        public List<PreToken> preTokenize(String text) {
            if (text == null) text = "";
            String src = text;
            int offsetShift = 0;
            if (addPrefixSpace && !src.isEmpty() && !src.startsWith(" ")) {
                src = " " + src;
                // offsets still refer to original; HF often keeps offsets on original —
                // for parity with encode ids we primarily care about values.
            }
            List<PreToken> out = new ArrayList<>();
            if (useRegex) {
                Matcher m = GPT2.matcher(src);
                while (m.find()) {
                    String piece = m.group();
                    String encoded = BytesToUnicode.byteEncode(piece);
                    out.add(new PreToken(encoded, m.start(), m.end()));
                }
            } else {
                // Whole input as one piece, byte-encoded (used after Split in Qwen/Llama)
                String encoded = BytesToUnicode.byteEncode(src);
                out.add(new PreToken(encoded, 0, text.length()));
            }
            return out;
        }
    }

    final class SplitPreTokenizer implements PreTokenizer {
        private final Pattern pattern;
        private final RegexSplit.Behavior behavior;
        private final boolean invert;

        public SplitPreTokenizer(Pattern pattern, RegexSplit.Behavior behavior, boolean invert) {
            this.pattern = pattern;
            this.behavior = behavior == null ? RegexSplit.Behavior.ISOLATED : behavior;
            this.invert = invert;
        }

        static SplitPreTokenizer fromJson(Map<String, Object> m) {
            Object patObj = m.get("pattern");
            String pat = null;
            if (patObj instanceof Map<?, ?> pm) {
                Object r = pm.get("Regex");
                if (r == null) r = pm.get("String");
                pat = r == null ? null : String.valueOf(r);
            } else {
                pat = JsonMaps.asString(patObj);
            }
            if (pat == null) pat = "";
            RegexSplit.Behavior behavior = RegexSplit.Behavior.fromString(JsonMaps.asString(m.get("behavior")));
            boolean invert = JsonMaps.asBoolean(m, "invert", false);
            Pattern compiled;
            try {
                compiled = Pattern.compile(pat);
            } catch (Exception e) {
                // Fallback: treat as literal
                compiled = Pattern.compile(Pattern.quote(pat));
            }
            return new SplitPreTokenizer(compiled, behavior, invert);
        }

        @Override
        public List<PreToken> preTokenize(String text) {
            if (text == null || text.isEmpty()) return List.of();
            List<RegexSplit.Span> spans = RegexSplit.split(text, pattern, behavior, invert);
            List<PreToken> out = new ArrayList<>(spans.size());
            for (RegexSplit.Span s : spans) {
                out.add(new PreToken(s.value, s.start, s.end));
            }
            return out;
        }
    }

    final class MetaspacePreTokenizer implements PreTokenizer {
        private final String replacement;
        private final boolean addPrefixSpace;
        private final String prependScheme; // always / first / never

        public MetaspacePreTokenizer(String replacement, boolean addPrefixSpace, String prependScheme) {
            this.replacement = replacement == null || replacement.isEmpty() ? "▁" : replacement;
            this.addPrefixSpace = addPrefixSpace;
            this.prependScheme = prependScheme == null ? "always" : prependScheme;
        }

        static MetaspacePreTokenizer fromJson(Map<String, Object> m) {
            return new MetaspacePreTokenizer(
                    JsonMaps.asString(m.get("replacement")),
                    JsonMaps.asBoolean(m, "add_prefix_space", true),
                    JsonMaps.asString(m.get("prepend_scheme")));
        }

        @Override
        public List<PreToken> preTokenize(String text) {
            if (text == null) text = "";
            boolean prepend = addPrefixSpace && !"never".equals(prependScheme);
            String s = text;
            if (prepend && (s.isEmpty() || s.charAt(0) != ' ')) {
                s = " " + s;
            }
            // Replace spaces with replacement and split on replacement boundaries
            // HF Metaspace: each word becomes replacement + word (spaces → replacement)
            String replaced = s.replace(" ", replacement);
            // Split keeping the replacement attached to following word
            List<PreToken> out = new ArrayList<>();
            int i = 0;
            while (i < replaced.length()) {
                int start = i;
                // consume optional leading replacement then non-replacement run
                if (replaced.startsWith(replacement, i)) {
                    i += replacement.length();
                }
                while (i < replaced.length() && !replaced.startsWith(replacement, i)) {
                    i++;
                }
                String piece = replaced.substring(start, i);
                if (!piece.isEmpty()) {
                    out.add(new PreToken(piece, start, i));
                }
            }
            return out;
        }
    }

    final class WhitespacePreTokenizer implements PreTokenizer {
        public static final WhitespacePreTokenizer INSTANCE = new WhitespacePreTokenizer();
        private static final Pattern P = Pattern.compile("\\w+|[^\\w\\s]+");

        @Override
        public List<PreToken> preTokenize(String text) {
            if (text == null || text.isEmpty()) return List.of();
            List<PreToken> out = new ArrayList<>();
            Matcher m = P.matcher(text);
            while (m.find()) {
                out.add(new PreToken(m.group(), m.start(), m.end()));
            }
            return out;
        }
    }

    final class WhitespaceSplitPreTokenizer implements PreTokenizer {
        public static final WhitespaceSplitPreTokenizer INSTANCE = new WhitespaceSplitPreTokenizer();

        @Override
        public List<PreToken> preTokenize(String text) {
            if (text == null || text.isEmpty()) return List.of();
            List<PreToken> out = new ArrayList<>();
            int i = 0;
            while (i < text.length()) {
                while (i < text.length() && Character.isWhitespace(text.charAt(i))) i++;
                if (i >= text.length()) break;
                int s = i;
                while (i < text.length() && !Character.isWhitespace(text.charAt(i))) i++;
                out.add(new PreToken(text.substring(s, i), s, i));
            }
            return out;
        }
    }

    final class PunctuationPreTokenizer implements PreTokenizer {
        private final RegexSplit.Behavior behavior;

        public PunctuationPreTokenizer(RegexSplit.Behavior behavior) {
            this.behavior = behavior == null ? RegexSplit.Behavior.ISOLATED : behavior;
        }

        static PunctuationPreTokenizer fromJson(Map<String, Object> m) {
            return new PunctuationPreTokenizer(
                    RegexSplit.Behavior.fromString(JsonMaps.asString(m.get("behavior"))));
        }

        private static final Pattern PUNCT = Pattern.compile("\\p{P}+");

        @Override
        public List<PreToken> preTokenize(String text) {
            if (text == null || text.isEmpty()) return List.of();
            List<RegexSplit.Span> spans = RegexSplit.split(text, PUNCT, behavior, false);
            List<PreToken> out = new ArrayList<>(spans.size());
            for (RegexSplit.Span s : spans) out.add(new PreToken(s.value, s.start, s.end));
            return out;
        }
    }

    final class DigitsPreTokenizer implements PreTokenizer {
        private final boolean individual;

        public DigitsPreTokenizer(boolean individual) {
            this.individual = individual;
        }

        static DigitsPreTokenizer fromJson(Map<String, Object> m) {
            return new DigitsPreTokenizer(JsonMaps.asBoolean(m, "individual_digits", false));
        }

        private static final Pattern DIGITS = Pattern.compile("\\d+");

        @Override
        public List<PreToken> preTokenize(String text) {
            if (text == null || text.isEmpty()) return List.of();
            if (!individual) {
                List<RegexSplit.Span> spans = RegexSplit.split(text, DIGITS, RegexSplit.Behavior.ISOLATED, false);
                List<PreToken> out = new ArrayList<>(spans.size());
                for (RegexSplit.Span s : spans) out.add(new PreToken(s.value, s.start, s.end));
                return out;
            }
            // individual digits: split each digit
            List<PreToken> out = new ArrayList<>();
            int i = 0;
            while (i < text.length()) {
                char c = text.charAt(i);
                if (Character.isDigit(c)) {
                    out.add(new PreToken(String.valueOf(c), i, i + 1));
                    i++;
                } else {
                    int s = i;
                    while (i < text.length() && !Character.isDigit(text.charAt(i))) i++;
                    out.add(new PreToken(text.substring(s, i), s, i));
                }
            }
            return out;
        }
    }

    /**
     * HF BertPreTokenizer: split on whitespace AND punctuation.
     * Keeps punctuation as isolated tokens; drops pure whitespace.
     */
    final class BertPreTokenizer implements PreTokenizer {
        public static final BertPreTokenizer INSTANCE = new BertPreTokenizer();
        private static final Pattern P = Pattern.compile("\\s+|\\p{P}|[^\\s\\p{P}]+");

        @Override
        public List<PreToken> preTokenize(String text) {
            if (text == null || text.isEmpty()) return List.of();
            List<PreToken> out = new ArrayList<>();
            Matcher m = P.matcher(text);
            while (m.find()) {
                String g = m.group();
                if (g.trim().isEmpty()) continue; // whitespace only
                out.add(new PreToken(g, m.start(), m.end()));
            }
            return out;
        }
    }    final class CharDelimiterSplitPreTokenizer implements PreTokenizer {
        private final char delimiter;

        public CharDelimiterSplitPreTokenizer(char delimiter) {
            this.delimiter = delimiter;
        }

        static CharDelimiterSplitPreTokenizer fromJson(Map<String, Object> m) {
            String d = JsonMaps.asString(m.get("delimiter"));
            char c = (d == null || d.isEmpty()) ? ' ' : d.charAt(0);
            return new CharDelimiterSplitPreTokenizer(c);
        }

        @Override
        public List<PreToken> preTokenize(String text) {
            if (text == null || text.isEmpty()) return List.of();
            List<PreToken> out = new ArrayList<>();
            int s = 0;
            for (int i = 0; i < text.length(); i++) {
                if (text.charAt(i) == delimiter) {
                    if (i > s) out.add(new PreToken(text.substring(s, i), s, i));
                    s = i + 1;
                }
            }
            if (s < text.length()) out.add(new PreToken(text.substring(s), s, text.length()));
            return out;
        }
    }
}
