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
package org.bytedeco.pytorch.utils.tokenizers.processors;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.pytorch.utils.tokenizers.Encoding;
import org.bytedeco.pytorch.utils.tokenizers.JsonMaps;
import org.bytedeco.pytorch.utils.tokenizers.models.Token;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;

/**
 * HuggingFace post-processor stage (TemplateProcessing, Bert, Roberta, ByteLevel, …).
 */
@FunctionalInterface
public interface PostProcessor {

    /**
     * @param tokensA sequence A tokens from the model
     * @param tokensB optional sequence B (may be null)
     * @param addSpecialTokens whether to inject template specials
     * @param idLookup resolve special token string → id
     */
    Encoding process(List<Token> tokensA, List<Token> tokensB,
                     boolean addSpecialTokens,
                     Function<String, Integer> idLookup);

    PostProcessor NOP = (a, b, add, ids) -> toEncoding(a, null);

    static PostProcessor fromJson(Map<String, Object> m) {
        if (m == null) return NOP;
        String type = JsonMaps.asString(m.get("type"));
        if (type == null) return NOP;
        return switch (type) {
            case "TemplateProcessing" -> TemplateProcessing.fromJson(m);
            case "BertProcessing" -> BertProcessing.fromJson(m);
            case "RobertaProcessing" -> RobertaProcessing.fromJson(m);
            case "ByteLevel" -> ByteLevelPostProcessor.fromJson(m);
            case "Sequence" -> SequencePostProcessor.fromJson(m);
            default -> NOP;
        };
    }

    static Encoding toEncoding(List<Token> tokens, int[] typeIdsOverride) {
        if (tokens == null || tokens.isEmpty()) {
            return Encoding.builder().build();
        }
        int n = tokens.size();
        int[] ids = new int[n];
        int[] typeIds = new int[n];
        int[] special = new int[n];
        int[] mask = new int[n];
        List<String> toks = new ArrayList<>(n);
        List<Integer> offS = new ArrayList<>(n);
        List<Integer> offE = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            Token t = tokens.get(i);
            ids[i] = t.id();
            typeIds[i] = typeIdsOverride != null && i < typeIdsOverride.length ? typeIdsOverride[i] : 0;
            special[i] = t.special() ? 1 : 0;
            mask[i] = 1;
            toks.add(t.value());
            offS.add(t.start());
            offE.add(t.end());
        }
        return Encoding.builder()
                .ids(ids)
                .typeIds(typeIds)
                .attentionMask(mask)
                .specialTokensMask(special)
                .tokens(toks)
                .offsetsStart(offS)
                .offsetsEnd(offE)
                .build();
    }

    // ---- TemplateProcessing -------------------------------------------------

    final class TemplateProcessing implements PostProcessor {
        private final List<Piece> single;
        private final List<Piece> pair;
        private final Map<String, Integer> specialTokens; // name → id (from JSON)

        public TemplateProcessing(List<Piece> single, List<Piece> pair, Map<String, Integer> specialTokens) {
            this.single = single == null ? List.of(Piece.sequence("A", 0)) : List.copyOf(single);
            this.pair = pair == null ? this.single : List.copyOf(pair);
            this.specialTokens = specialTokens == null ? Map.of() : Map.copyOf(specialTokens);
        }

        static TemplateProcessing fromJson(Map<String, Object> m) {
            List<Piece> single = parseTemplate(m.get("single"));
            List<Piece> pair = parseTemplate(m.get("pair"));
            Map<String, Integer> specials = new HashMap<>();
            Object st = m.get("special_tokens");
            if (st instanceof List<?> list) {
                for (Object o : list) {
                    Map<String, Object> sm = JsonMaps.asMap(o);
                    if (sm == null) continue;
                    // May be {"id":"...", "ids":[...], "tokens":[...]} wrapping
                    String idStr = JsonMaps.asString(sm.get("id"));
                    List<Object> ids = JsonMaps.asList(sm.get("ids"));
                    List<Object> tokens = JsonMaps.asList(sm.get("tokens"));
                    if (idStr != null && ids != null && !ids.isEmpty()) {
                        Integer id = JsonMaps.asInt(ids.get(0));
                        if (id != null) specials.put(idStr, id);
                        if (tokens != null && !tokens.isEmpty()) {
                            specials.put(String.valueOf(tokens.get(0)), id);
                        }
                    } else {
                        // flat SpecialToken style
                        String tok = JsonMaps.asString(sm.get("SpecialToken"));
                        // or nested
                        Object nested = sm.get("SpecialToken");
                        if (nested instanceof Map<?, ?>) {
                            Map<String, Object> nm = JsonMaps.asMap(nested);
                            String sid = JsonMaps.asString(nm.get("id"));
                            // id here is the token string; numeric id resolved later
                            if (sid != null) specials.putIfAbsent(sid, -1);
                        }
                    }
                }
            } else if (st instanceof Map<?, ?>) {
                Map<String, Object> sm = JsonMaps.asMap(st);
                for (Map.Entry<String, Object> e : sm.entrySet()) {
                    Map<String, Object> v = JsonMaps.asMap(e.getValue());
                    if (v != null) {
                        List<Object> ids = JsonMaps.asList(v.get("ids"));
                        if (ids != null && !ids.isEmpty()) {
                            Integer id = JsonMaps.asInt(ids.get(0));
                            if (id != null) specials.put(e.getKey(), id);
                        }
                    }
                }
            }
            return new TemplateProcessing(single, pair, specials);
        }

        @SuppressWarnings("unchecked")
        private static List<Piece> parseTemplate(Object raw) {
            List<Piece> out = new ArrayList<>();
            if (raw == null) return out;
            // May be a string like "<s>:0 $A:0 </s>:0" (older) or list of maps
            if (raw instanceof String s) {
                for (String part : s.trim().split("\\s+")) {
                    if (part.isEmpty()) continue;
                    String[] bits = part.split(":");
                    String name = bits[0];
                    int typeId = bits.length > 1 ? Integer.parseInt(bits[1]) : 0;
                    if (name.startsWith("$")) {
                        out.add(Piece.sequence(name.substring(1), typeId));
                    } else {
                        out.add(Piece.special(name, typeId));
                    }
                }
                return out;
            }
            List<Object> list = JsonMaps.asList(raw);
            if (list == null) return out;
            for (Object o : list) {
                Map<String, Object> m = JsonMaps.asMap(o);
                if (m == null) continue;
                if (m.containsKey("Sequence")) {
                    Map<String, Object> seq = JsonMaps.asMap(m.get("Sequence"));
                    String id = JsonMaps.asString(seq.get("id"));
                    Integer typeId = JsonMaps.asInt(seq.get("type_id"));
                    out.add(Piece.sequence(id == null ? "A" : id, typeId == null ? 0 : typeId));
                } else if (m.containsKey("SpecialToken")) {
                    Map<String, Object> sp = JsonMaps.asMap(m.get("SpecialToken"));
                    String id = JsonMaps.asString(sp.get("id"));
                    Integer typeId = JsonMaps.asInt(sp.get("type_id"));
                    out.add(Piece.special(id == null ? "" : id, typeId == null ? 0 : typeId));
                }
            }
            return out;
        }

        @Override
        public Encoding process(List<Token> tokensA, List<Token> tokensB,
                                boolean addSpecialTokens,
                                Function<String, Integer> idLookup) {
            List<Piece> template = (tokensB != null && !tokensB.isEmpty()) ? pair : single;
            if (!addSpecialTokens) {
                // Only sequences, no specials
                List<Token> all = new ArrayList<>();
                int[] types = null;
                List<Integer> typeList = new ArrayList<>();
                for (Piece p : template) {
                    if (p.sequence) {
                        List<Token> src = "B".equalsIgnoreCase(p.id) ? tokensB : tokensA;
                        if (src != null) {
                            for (Token t : src) {
                                all.add(t);
                                typeList.add(p.typeId);
                            }
                        }
                    }
                }
                if (all.isEmpty() && tokensA != null) {
                    // fallback: just A
                    return toEncoding(tokensA, null);
                }
                types = typeList.stream().mapToInt(Integer::intValue).toArray();
                return toEncoding(all, types);
            }

            List<Token> all = new ArrayList<>();
            List<Integer> typeList = new ArrayList<>();
            for (Piece p : template) {
                if (p.sequence) {
                    List<Token> src = "B".equalsIgnoreCase(p.id) ? tokensB : tokensA;
                    if (src != null) {
                        for (Token t : src) {
                            all.add(t);
                            typeList.add(p.typeId);
                        }
                    }
                } else {
                    Integer id = specialTokens.get(p.id);
                    if (id == null || id < 0) {
                        id = idLookup.apply(p.id);
                    }
                    if (id == null) id = -1;
                    if (id >= 0) {
                        all.add(Token.special(id, p.id));
                        typeList.add(p.typeId);
                    }
                }
            }
            int[] types = typeList.stream().mapToInt(Integer::intValue).toArray();
            return toEncoding(all, types);
        }

        public record Piece(String id, int typeId, boolean sequence) {
            public static Piece sequence(String id, int typeId) { return new Piece(id, typeId, true); }
            public static Piece special(String id, int typeId) { return new Piece(id, typeId, false); }
        }

        /** ChatGLM4 prefix: [gMASK] <sop> $A */
        public static TemplateProcessing chatGlm4(int gmaskId, int sopId) {
            Map<String, Integer> specials = new HashMap<>();
            specials.put("[gMASK]", gmaskId);
            specials.put("<sop>", sopId);
            List<Piece> single = List.of(
                    Piece.special("[gMASK]", 0),
                    Piece.special("<sop>", 0),
                    Piece.sequence("A", 0)
            );
            return new TemplateProcessing(single, single, specials);
        }
    }

    // ---- Bert / Roberta -----------------------------------------------------

    final class BertProcessing implements PostProcessor {
        private final String cls;
        private final int clsId;
        private final String sep;
        private final int sepId;

        public BertProcessing(String cls, int clsId, String sep, int sepId) {
            this.cls = cls;
            this.clsId = clsId;
            this.sep = sep;
            this.sepId = sepId;
        }

        static BertProcessing fromJson(Map<String, Object> m) {
            // {"cls": ["[CLS]", 101], "sep": ["[SEP]", 102]}
            String cls = "[CLS]";
            int clsId = 101;
            String sep = "[SEP]";
            int sepId = 102;
            Object clsObj = m.get("cls");
            if (clsObj instanceof List<?> list && list.size() >= 2) {
                cls = String.valueOf(list.get(0));
                Integer id = JsonMaps.asInt(list.get(1));
                if (id != null) clsId = id;
            }
            Object sepObj = m.get("sep");
            if (sepObj instanceof List<?> list && list.size() >= 2) {
                sep = String.valueOf(list.get(0));
                Integer id = JsonMaps.asInt(list.get(1));
                if (id != null) sepId = id;
            }
            return new BertProcessing(cls, clsId, sep, sepId);
        }

        @Override
        public Encoding process(List<Token> tokensA, List<Token> tokensB,
                                boolean addSpecialTokens,
                                Function<String, Integer> idLookup) {
            List<Token> all = new ArrayList<>();
            List<Integer> types = new ArrayList<>();
            if (addSpecialTokens) {
                all.add(Token.special(clsId, cls));
                types.add(0);
            }
            if (tokensA != null) {
                for (Token t : tokensA) { all.add(t); types.add(0); }
            }
            if (addSpecialTokens) {
                all.add(Token.special(sepId, sep));
                types.add(0);
            }
            if (tokensB != null && !tokensB.isEmpty()) {
                for (Token t : tokensB) { all.add(t); types.add(1); }
                if (addSpecialTokens) {
                    all.add(Token.special(sepId, sep));
                    types.add(1);
                }
            }
            return toEncoding(all, types.stream().mapToInt(Integer::intValue).toArray());
        }
    }

    final class RobertaProcessing implements PostProcessor {
        private final String cls;
        private final int clsId;
        private final String sep;
        private final int sepId;
        private final boolean trimOffsets;
        private final boolean addPrefixSpace;

        public RobertaProcessing(String cls, int clsId, String sep, int sepId,
                                 boolean trimOffsets, boolean addPrefixSpace) {
            this.cls = cls;
            this.clsId = clsId;
            this.sep = sep;
            this.sepId = sepId;
            this.trimOffsets = trimOffsets;
            this.addPrefixSpace = addPrefixSpace;
        }

        static RobertaProcessing fromJson(Map<String, Object> m) {
            String cls = "<s>";
            int clsId = 0;
            String sep = "</s>";
            int sepId = 2;
            Object clsObj = m.get("cls");
            if (clsObj instanceof List<?> list && list.size() >= 2) {
                cls = String.valueOf(list.get(0));
                Integer id = JsonMaps.asInt(list.get(1));
                if (id != null) clsId = id;
            }
            Object sepObj = m.get("sep");
            if (sepObj instanceof List<?> list && list.size() >= 2) {
                sep = String.valueOf(list.get(0));
                Integer id = JsonMaps.asInt(list.get(1));
                if (id != null) sepId = id;
            }
            return new RobertaProcessing(cls, clsId, sep, sepId,
                    JsonMaps.asBoolean(m, "trim_offsets", true),
                    JsonMaps.asBoolean(m, "add_prefix_space", true));
        }

        @Override
        public Encoding process(List<Token> tokensA, List<Token> tokensB,
                                boolean addSpecialTokens,
                                Function<String, Integer> idLookup) {
            // Same shape as Bert but type ids always 0 for RoBERTa typically
            List<Token> all = new ArrayList<>();
            if (addSpecialTokens) all.add(Token.special(clsId, cls));
            if (tokensA != null) all.addAll(tokensA);
            if (addSpecialTokens) all.add(Token.special(sepId, sep));
            if (tokensB != null && !tokensB.isEmpty()) {
                if (addSpecialTokens) all.add(Token.special(sepId, sep));
                all.addAll(tokensB);
                if (addSpecialTokens) all.add(Token.special(sepId, sep));
            }
            return toEncoding(all, null);
        }
    }

    final class ByteLevelPostProcessor implements PostProcessor {
        private final boolean trimOffsets;

        public ByteLevelPostProcessor(boolean trimOffsets) {
            this.trimOffsets = trimOffsets;
        }

        static ByteLevelPostProcessor fromJson(Map<String, Object> m) {
            return new ByteLevelPostProcessor(JsonMaps.asBoolean(m, "trim_offsets", true));
        }

        @Override
        public Encoding process(List<Token> tokensA, List<Token> tokensB,
                                boolean addSpecialTokens,
                                Function<String, Integer> idLookup) {
            // Primarily trims offsets; ids unchanged. Merge A (+ optional B).
            List<Token> all = new ArrayList<>();
            if (tokensA != null) all.addAll(tokensA);
            if (tokensB != null) all.addAll(tokensB);
            return toEncoding(all, null);
        }
    }

    final class SequencePostProcessor implements PostProcessor {
        private final List<PostProcessor> processors;

        public SequencePostProcessor(List<PostProcessor> processors) {
            this.processors = List.copyOf(Objects.requireNonNull(processors));
        }

        static SequencePostProcessor fromJson(Map<String, Object> m) {
            List<Object> raw = JsonMaps.asList(m.get("processors"));
            List<PostProcessor> list = new ArrayList<>();
            if (raw != null) {
                for (Object o : raw) {
                    Map<String, Object> cm = JsonMaps.asMap(o);
                    if (cm != null) list.add(PostProcessor.fromJson(cm));
                }
            }
            return new SequencePostProcessor(list);
        }

        @Override
        public Encoding process(List<Token> tokensA, List<Token> tokensB,
                                boolean addSpecialTokens,
                                Function<String, Integer> idLookup) {
            // Apply in sequence: first gets raw tokens; subsequent re-process is limited —
            // practically HF Sequence of post processors is rare; use first meaningful.
            Encoding enc = null;
            List<Token> curA = tokensA;
            for (PostProcessor p : processors) {
                enc = p.process(curA, tokensB, addSpecialTokens, idLookup);
                // Rebuild tokens from encoding for next stage
                List<Token> next = new ArrayList<>();
                int[] ids = enc.ids();
                List<String> toks = enc.tokens();
                int[] special = enc.specialTokensMask();
                for (int i = 0; i < ids.length; i++) {
                    String v = i < toks.size() ? toks.get(i) : "";
                    boolean sp = special != null && i < special.length && special[i] == 1;
                    next.add(new Token(ids[i], v, 0, 0, sp));
                }
                curA = next;
                tokensB = null; // only first sees pair
            }
            return enc == null ? toEncoding(tokensA, null) : enc;
        }
    }
}
