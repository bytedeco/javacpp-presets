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
package org.bytedeco.pytorch.llm.llamafactory.eval;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Tiny in-process multi-choice harness (MMLU-style prompt + letter extract).
 *
 * <p>Not a full MMLU download client — hosts supply items JSON or use {@link #demoItems()}.
 */
public final class MmluLikeHarness {

    private static final Pattern CHOICE = Pattern.compile(
            "(?i)\\b([ABCD])\\b|answer\\s*[:\\s]*([ABCD])|\\(([ABCD])\\)");

    private MmluLikeHarness() {}

    public static final class Item {
        public final String id;
        public final String question;
        public final String[] choices; // A,B,C,D texts
        public final String answer;    // "A".."D"

        public Item(String id, String question, String[] choices, String answer) {
            this.id = id == null ? "" : id;
            this.question = question == null ? "" : question;
            this.choices = choices == null ? new String[0] : choices.clone();
            this.answer = answer == null ? "" : answer.trim().toUpperCase(Locale.ROOT);
        }
    }

    public static List<Item> demoItems() {
        List<Item> items = new ArrayList<>();
        items.add(new Item(
                "demo-1",
                "What is the capital of France?",
                new String[]{"London", "Paris", "Berlin", "Madrid"},
                "B"));
        items.add(new Item(
                "demo-2",
                "2 + 2 = ?",
                new String[]{"3", "4", "5", "22"},
                "B"));
        items.add(new Item(
                "demo-3",
                "Which planet is known as the Red Planet?",
                new String[]{"Venus", "Jupiter", "Mars", "Saturn"},
                "C"));
        items.add(new Item(
                "demo-4",
                "The chemical symbol for water is?",
                new String[]{"O2", "H2O", "CO2", "NaCl"},
                "B"));
        return items;
    }

    public static String formatPrompt(Item item, int nShot) {
        StringBuilder sb = new StringBuilder();
        sb.append("The following are multiple choice questions (with answers).\n\n");
        if (nShot > 0) {
            // embed first nShot demo items as few-shot (excluding current if overlap)
            List<Item> shots = demoItems();
            int used = 0;
            for (Item s : shots) {
                if (used >= nShot) break;
                if (s.id.equals(item.id)) continue;
                appendQuestion(sb, s, true);
                used++;
            }
        }
        appendQuestion(sb, item, false);
        return sb.toString();
    }

    private static void appendQuestion(StringBuilder sb, Item item, boolean withAnswer) {
        sb.append("Question: ").append(item.question).append('\n');
        String[] labels = {"A", "B", "C", "D", "E", "F"};
        for (int i = 0; i < item.choices.length && i < labels.length; i++) {
            sb.append(labels[i]).append(". ").append(item.choices[i]).append('\n');
        }
        sb.append("Answer:");
        if (withAnswer) {
            sb.append(' ').append(item.answer).append("\n\n");
        } else {
            sb.append(' ');
        }
    }

    /** Extract first A/B/C/D letter from model text. */
    public static String extractChoice(String text) {
        if (text == null || text.isBlank()) return "";
        String t = text.trim();
        // prefer leading letter
        char c0 = Character.toUpperCase(t.charAt(0));
        if (c0 >= 'A' && c0 <= 'D') return String.valueOf(c0);
        Matcher m = CHOICE.matcher(t);
        if (m.find()) {
            for (int g = 1; g <= m.groupCount(); g++) {
                if (m.group(g) != null) {
                    return m.group(g).toUpperCase(Locale.ROOT);
                }
            }
        }
        // scan
        for (int i = 0; i < t.length(); i++) {
            char c = Character.toUpperCase(t.charAt(i));
            if (c >= 'A' && c <= 'D') return String.valueOf(c);
        }
        return "";
    }

    /** Deterministic offline fallback when generate is unavailable. */
    public static String heuristicChoice(Item item) {
        if (item.answer != null && !item.answer.isBlank()) {
            // still return a letter so accuracy is measurable against gold in smoke tests;
            // use length-argmax among choices for a non-constant baseline
            int best = 0;
            int bestLen = -1;
            for (int i = 0; i < item.choices.length; i++) {
                int len = item.choices[i] == null ? 0 : item.choices[i].length();
                if (len > bestLen) {
                    bestLen = len;
                    best = i;
                }
            }
            return String.valueOf((char) ('A' + Math.min(best, 3)));
        }
        return "A";
    }

    @SuppressWarnings("unchecked")
    public static List<Item> parseItems(Object decoded) {
        List<Item> out = new ArrayList<>();
        if (decoded instanceof List<?> list) {
            int i = 0;
            for (Object o : list) {
                if (o instanceof Map<?, ?> m) {
                    out.add(fromMap((Map<String, Object>) m, i++));
                }
            }
        } else if (decoded instanceof Map<?, ?> root) {
            Object items = root.get("items");
            if (items == null) items = root.get("data");
            if (items instanceof List<?> list) {
                int i = 0;
                for (Object o : list) {
                    if (o instanceof Map<?, ?> m) {
                        out.add(fromMap((Map<String, Object>) m, i++));
                    }
                }
            }
        }
        return out;
    }

    @SuppressWarnings("unchecked")
    private static Item fromMap(Map<String, Object> m, int idx) {
        String id = str(m.get("id"), "item-" + idx);
        String q = str(m.get("question"), str(m.get("q"), ""));
        String ans = str(m.get("answer"), str(m.get("gold"), str(m.get("label"), "")));
        String[] choices;
        Object ch = m.get("choices");
        if (ch instanceof List<?> list) {
            choices = new String[list.size()];
            for (int i = 0; i < list.size(); i++) {
                choices[i] = String.valueOf(list.get(i));
            }
        } else {
            List<String> tmp = new ArrayList<>();
            for (String k : List.of("A", "B", "C", "D", "E")) {
                if (m.containsKey(k)) tmp.add(String.valueOf(m.get(k)));
                else if (m.containsKey(k.toLowerCase(Locale.ROOT))) {
                    tmp.add(String.valueOf(m.get(k.toLowerCase(Locale.ROOT))));
                }
            }
            choices = tmp.toArray(new String[0]);
        }
        return new Item(id, q, choices, ans);
    }

    private static String str(Object o, String def) {
        return o == null ? def : String.valueOf(o);
    }
}
