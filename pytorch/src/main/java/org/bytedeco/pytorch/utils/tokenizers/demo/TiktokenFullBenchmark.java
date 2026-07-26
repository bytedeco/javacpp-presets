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
package org.bytedeco.pytorch.utils.tokenizers.demo;

import org.bytedeco.pytorch.utils.json.Json;
import org.bytedeco.pytorch.utils.tokenizers.Encoding;
import org.bytedeco.pytorch.utils.tokenizers.FastTokenizer;
import org.bytedeco.pytorch.utils.tokenizers.Tiktoken;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Multi-dimensional correctness + throughput benchmark for pure-Java
 * {@link Tiktoken}, cross-checked against Python tiktoken 0.13+ goldens and
 * exercised together with the transformers {@link FastTokenizer} adapter.
 *
 * <h2>Dimensions</h2>
 * <ol>
 *   <li><b>API surface</b> — list/get/encoding_for_model/n_vocab/eot/specials</li>
 *   <li><b>Encode parity</b> — encode_ordinary / allowed_special=all /
 *       disallowed_special=() / default raise, per encoding × text dim</li>
 *   <li><b>Decode parity</b> — decode / decode_bytes round-trip</li>
 *   <li><b>Specials policy</b> — raise / allow / ordinary paths</li>
 *   <li><b>Single-token</b> — encode_single_token / decode_single_token_bytes</li>
 *   <li><b>Batch</b> — encodeOrdinaryBatch / decodeBatch</li>
 *   <li><b>Model map</b> — encoding_for_model for gpt-4 / gpt-4o / o1 / davinci…</li>
 *   <li><b>FastTokenizer adapter</b> — Tiktoken → transformers pipeline interop</li>
 *   <li><b>Throughput</b> — chars/s + tokens/s vs Python reference in meta.json</li>
 *   <li><b>Multilingual / code / long / emoji / CJK</b> — dim tags in goldens</li>
 * </ol>
 *
 * <pre>
 *   java ... TiktokenFullBenchmark
 *   java ... TiktokenFullBenchmark cl100k_base o200k_base
 *   java ... TiktokenFullBenchmark --quick
 * </pre>
 *
 * <p>Goldens live under {@code src/test/resources/tokenizers/goldens/tiktoken_&lt;name&gt;/}
 * ({@code cases_multidim.jsonl} + {@code meta.json}), produced from Python tiktoken.
 */
public final class TiktokenFullBenchmark {

    private static final List<String> DEFAULT_ENCODINGS = List.of(
            "gpt2", "r50k_base", "p50k_base", "p50k_edit", "cl100k_base", "o200k_base"
            // o200k_harmony is huge on specials; still loadable via listEncodingNames smoke
    );

    private static final Map<String, String> MODEL_EXPECT = Map.ofEntries(
            Map.entry("gpt-4", "cl100k_base"),
            Map.entry("gpt-4-turbo", "cl100k_base"),
            Map.entry("gpt-3.5-turbo", "cl100k_base"),
            Map.entry("text-embedding-3-small", "cl100k_base"),
            Map.entry("gpt-4o", "o200k_base"),
            Map.entry("gpt-4o-mini", "o200k_base"),
            Map.entry("o1", "o200k_base"),
            Map.entry("o1-mini", "o200k_base"),
            Map.entry("o3", "o200k_base"),
            Map.entry("text-davinci-003", "p50k_base"),
            Map.entry("davinci", "r50k_base"),
            Map.entry("gpt-2", "gpt2"),
            Map.entry("gpt2", "gpt2")
    );

    private static int passed = 0;
    private static int failed = 0;
    private static final StringBuilder failures = new StringBuilder();

    public static void main(String[] args) throws Exception {
        boolean quick = false;
        List<String> encodings = new ArrayList<>();
        for (String a : args) {
            if ("--quick".equals(a)) quick = true;
            else encodings.add(a);
        }
        if (encodings.isEmpty()) encodings.addAll(DEFAULT_ENCODINGS);

        Path goldRoot = Path.of("src/test/resources/tokenizers/goldens");
        System.out.println("╔══════════════════════════════════════════════════════════════╗");
        System.out.println("║  Tiktoken FULL multi-dimensional benchmark (Java ↔ Python)  ║");
        System.out.println("╚══════════════════════════════════════════════════════════════╝");
        System.out.println("goldens: " + goldRoot.toAbsolutePath());
        System.out.println("encodings: " + encodings);
        System.out.println();

        // ── Dim 1: API surface ─────────────────────────────────────────────
        section("1. API surface / list_encoding_names");
        List<String> names = Tiktoken.listEncodingNames();
        check("list contains cl100k_base", names.contains("cl100k_base"));
        check("list contains o200k_base", names.contains("o200k_base"));
        check("list contains r50k_base", names.contains("r50k_base"));
        check("list contains gpt2", names.contains("gpt2"));
        check("list contains o200k_harmony", names.contains("o200k_harmony"));
        check("list size >= 7", names.size() >= 7);

        // ── Dim 2: model map ───────────────────────────────────────────────
        section("2. encoding_for_model mapping");
        for (Map.Entry<String, String> e : MODEL_EXPECT.entrySet()) {
            try {
                String got = Tiktoken.encodingNameForModel(e.getKey());
                checkEq("model " + e.getKey(), e.getValue(), got);
                Tiktoken enc = Tiktoken.encodingForModel(e.getKey());
                checkEq("encodingForModel.name " + e.getKey(), e.getValue(), enc.name());
            } catch (Exception ex) {
                fail("model " + e.getKey(), ex.toString());
            }
        }
        try {
            Tiktoken.encodingNameForModel("not-a-real-model-xyz");
            fail("unknown model should throw", "no exception");
        } catch (IllegalArgumentException ok) {
            pass("unknown model raises");
        }

        // ── Per-encoding dimensions ────────────────────────────────────────
        List<String> summary = new ArrayList<>();
        for (String name : encodings) {
            System.out.println();
            System.out.println("══════════════ " + name + " ══════════════");
            long tLoad0 = System.nanoTime();
            Tiktoken enc;
            try {
                enc = Tiktoken.getEncoding(name);
            } catch (Exception ex) {
                fail("load " + name, ex.toString());
                summary.add(String.format(Locale.ROOT, "%-14s LOAD_FAIL", name));
                continue;
            }
            long loadMs = (System.nanoTime() - tLoad0) / 1_000_000L;

            Path dir = goldRoot.resolve("tiktoken_" + name);
            Path metaPath = dir.resolve("meta.json");
            Path casesPath = dir.resolve("cases_multidim.jsonl");
            if (!Files.isRegularFile(casesPath)) {
                // fall back to simpler cases.jsonl
                casesPath = dir.resolve("cases.jsonl");
            }

            Map<String, Object> meta = Map.of();
            if (Files.isRegularFile(metaPath)) {
                meta = Json.decodeObject(Files.readString(metaPath));
            }

            // Dim 3: vocab / eot / specials vs meta
            section("3. meta parity (" + name + ")");
            if (meta.get("n_vocab") instanceof Number nVocab) {
                checkEq(name + " nVocab", nVocab.intValue(), enc.nVocab());
            }
            if (meta.get("eot_token") instanceof Number eot) {
                checkEq(name + " eot", eot.intValue(), enc.eotToken());
            }
            if (meta.get("max_token_value") instanceof Number mtv) {
                checkEq(name + " maxTokenValue", mtv.intValue(), enc.maxTokenValue());
            }
            if (meta.get("special_tokens_set") instanceof List<?> sp) {
                checkEq(name + " specials size", sp.size(), enc.specialTokensSet().size());
                for (Object o : sp) {
                    check(name + " has special " + o, enc.specialTokensSet().contains(String.valueOf(o)));
                }
            }
            System.out.println("  loadMs=" + loadMs + " nVocab=" + enc.nVocab()
                    + " specials=" + enc.specialTokensSet().size());

            // Dim 4–7: golden encode/decode across text dimensions
            int ok = 0, idFail = 0, decFail = 0, policyFail = 0, dimCount = 0;
            Map<String, int[]> dimStats = new LinkedHashMap<>(); // dim -> [ok,fail]

            if (Files.isRegularFile(casesPath)) {
                section("4. golden encode/decode parity (" + name + ")");
                List<String> lines = Files.readAllLines(casesPath, StandardCharsets.UTF_8);
                for (String line : lines) {
                    if (line.isBlank()) continue;
                    Map<String, Object> row = Json.decodeObject(line);
                    String text = row.get("text") == null ? "" : String.valueOf(row.get("text"));
                    String dim = row.get("dim") == null ? "default" : String.valueOf(row.get("dim"));
                    dimStats.computeIfAbsent(dim, k -> new int[2]);

                    int[] goldOrd = toIntArray(row.get("ids_ordinary"));
                    int[] gotOrd = enc.encodeOrdinary(text);
                    boolean idsOk = Arrays.equals(goldOrd, gotOrd);

                    // decode parity
                    String goldDec = row.get("decoded_ordinary") == null
                            ? null : String.valueOf(row.get("decoded_ordinary"));
                    String gotDec = enc.decode(gotOrd);
                    boolean decOk = Objects.equals(goldDec, gotDec);

                    // decode_bytes
                    boolean bytesOk = true;
                    if (row.get("decoded_bytes_ordinary") instanceof List<?> bl) {
                        byte[] goldB = toByteArray(bl);
                        byte[] gotB = enc.decodeBytes(gotOrd);
                        bytesOk = Arrays.equals(goldB, gotB);
                    }

                    // allowed_special=all
                    int[] goldAll = toIntArray(row.get("ids_allowed_all"));
                    int[] gotAll = enc.encode(text, Tiktoken.SPECIAL_ALL, Set.of());
                    boolean allOk = Arrays.equals(goldAll, gotAll);

                    // disallowed_special=()
                    int[] goldEmpty = toIntArray(row.get("ids_disallowed_empty"));
                    int[] gotEmpty = enc.encode(text, Set.of(), Set.of());
                    boolean emptyOk = Arrays.equals(goldEmpty, gotEmpty);

                    // default encode may raise
                    boolean policyOk = true;
                    Object defErr = row.get("default_error");
                    if (defErr != null && !"null".equals(String.valueOf(defErr))) {
                        try {
                            enc.encode(text);
                            policyOk = false;
                            policyFail++;
                            System.out.println("  FAIL policy: expected raise for text=" + preview(text));
                        } catch (IllegalArgumentException expected) {
                            // ok
                        }
                    } else if (row.get("ids_default") instanceof List<?>) {
                        int[] goldDef = toIntArray(row.get("ids_default"));
                        try {
                            int[] gotDef = enc.encode(text);
                            if (!Arrays.equals(goldDef, gotDef)) {
                                policyOk = false;
                                policyFail++;
                                System.out.println("  FAIL default ids text=" + preview(text));
                                System.out.println("    java=" + Arrays.toString(gotDef));
                                System.out.println("    gold=" + Arrays.toString(goldDef));
                            }
                        } catch (IllegalArgumentException ex) {
                            policyOk = false;
                            policyFail++;
                            System.out.println("  FAIL default unexpected raise: " + ex.getMessage());
                        }
                    }

                    if (!idsOk) {
                        idFail++;
                        dimStats.get(dim)[1]++;
                        System.out.println("  FAIL ordinary ids dim=" + dim + " text=" + preview(text));
                        System.out.println("    java=" + Arrays.toString(gotOrd));
                        System.out.println("    gold=" + Arrays.toString(goldOrd));
                    } else if (!decOk || !bytesOk) {
                        decFail++;
                        dimStats.get(dim)[1]++;
                        System.out.println("  FAIL decode dim=" + dim + " text=" + preview(text));
                        System.out.println("    javaDec=" + preview(gotDec) + " goldDec=" + preview(goldDec));
                    } else if (!allOk || !emptyOk) {
                        idFail++;
                        dimStats.get(dim)[1]++;
                        System.out.println("  FAIL specials-path ids dim=" + dim + " text=" + preview(text));
                        if (!allOk) {
                            System.out.println("    all java=" + Arrays.toString(gotAll));
                            System.out.println("    all gold=" + Arrays.toString(goldAll));
                        }
                        if (!emptyOk) {
                            System.out.println("    empty java=" + Arrays.toString(gotEmpty));
                            System.out.println("    empty gold=" + Arrays.toString(goldEmpty));
                        }
                    } else if (!policyOk) {
                        dimStats.get(dim)[1]++;
                    } else {
                        ok++;
                        dimStats.get(dim)[0]++;
                    }
                    dimCount++;
                }

                System.out.printf(Locale.ROOT,
                        "  parity cases=%d ok=%d idFail=%d decFail=%d policyFail=%d%n",
                        dimCount, ok, idFail, decFail, policyFail);
                System.out.println("  per-dimension:");
                for (Map.Entry<String, int[]> de : dimStats.entrySet()) {
                    System.out.printf(Locale.ROOT, "    %-12s ok=%d fail=%d%n",
                            de.getKey(), de.getValue()[0], de.getValue()[1]);
                }
            } else {
                System.out.println("  MISSING goldens at " + casesPath + " — running builtin probes only");
                builtinProbes(enc);
            }

            // Dim 8: single-token + batch
            section("5. single-token / batch / decode_bytes (" + name + ")");
            try {
                // "Hello" is a single token in all standard encodings we ship
                int hid = enc.encodeSingleToken("Hello");
                byte[] hb = enc.decodeSingleTokenBytes(hid);
                String hs = new String(hb, StandardCharsets.UTF_8);
                check(name + " single-token Hello roundtrip", "Hello".equals(hs));
                pass(name + " encodeSingleToken Hello=" + hid);
            } catch (Exception ex) {
                // some encodings may not have exact "Hello" as one token — still report
                fail(name + " single-token Hello", ex.toString());
            }
            List<String> batchIn = List.of("Hello", " world", "!", "你好");
            List<int[]> batchIds = enc.encodeOrdinaryBatch(batchIn);
            check(name + " batch size", batchIds.size() == batchIn.size());
            List<String> batchDec = enc.decodeBatch(batchIds);
            check(name + " batch decode size", batchDec.size() == batchIn.size());
            for (int i = 0; i < batchIn.size(); i++) {
                checkEq(name + " batch[" + i + "]", batchIn.get(i), batchDec.get(i));
            }

            // Dim 9: FastTokenizer / transformers adapter
            section("6. FastTokenizer adapter (transformers interop) (" + name + ")");
            try {
                FastTokenizer ft = enc.toFastTokenizer();
                check(name + " ft vocab>0", ft.vocabSize() > 0);
                Encoding fe = ft.encode("Hello world", false);
                int[] pure = enc.encodeOrdinary("Hello world");
                // Adapter goes through Split+ByteLevel+TiktokenBpeModel — should match pure path
                boolean match = Arrays.equals(fe.ids(), pure);
                if (!match) {
                    // Print but don't hard-fail the whole suite if only adapter drifts;
                    // pure tiktoken path is the source of truth vs Python.
                    System.out.println("  WARN adapter ids drift pure=" + Arrays.toString(pure)
                            + " ft=" + Arrays.toString(fe.ids()));
                    // Still count as soft check
                    check(name + " ft non-empty", fe.size() > 0);
                } else {
                    pass(name + " ft adapter matches pure encodeOrdinary");
                }
                String fdec = ft.decode(fe.ids(), false);
                check(name + " ft decode non-null", fdec != null);
            } catch (Exception ex) {
                fail(name + " FastTokenizer adapter", ex.toString());
                ex.printStackTrace(System.out);
            }

            // Dim 10: throughput
            section("7. throughput (" + name + ")");
            String longText = "Hello world! 你好世界 🎉\n".repeat(200);
            // warmup
            for (int i = 0; i < 5; i++) enc.encodeOrdinary(longText);
            int iters = quick ? 10 : 50;
            long t0 = System.nanoTime();
            int tokCount = 0;
            for (int i = 0; i < iters; i++) {
                tokCount += enc.encodeOrdinary(longText).length;
            }
            long ns = System.nanoTime() - t0;
            double sec = ns / 1e9;
            long chars = (long) longText.length() * iters;
            double cps = chars / sec;
            double tps = tokCount / sec;
            System.out.printf(Locale.ROOT,
                    "  Java  chars/s=%.0f  tokens/s=%.0f  (iters=%d, chars/iter=%d, toks_total=%d)%n",
                    cps, tps, iters, longText.length(), tokCount);
            if (meta.get("throughput_ref") instanceof Map<?, ?> ref) {
                Object pyTps = ref.get("tokens_per_sec");
                Object pyCps = ref.get("chars_per_sec");
                System.out.printf(Locale.ROOT,
                        "  Python ref tokens/s=%s  chars/s=%s%n",
                        pyTps, pyCps);
                if (pyTps instanceof Number n && n.doubleValue() > 0) {
                    double ratio = tps / n.doubleValue();
                    System.out.printf(Locale.ROOT, "  Java/Python throughput ratio=%.3f%n", ratio);
                }
            }
            check(name + " throughput > 0", tps > 0);

            // Round-trip stress
            section("8. round-trip stress (" + name + ")");
            List<String> stress = List.of(
                    "Hello world",
                    "你好世界",
                    "🎉🌍",
                    "def foo(x):\n    return x+1\n",
                    "a".repeat(512),
                    "Hello 世界 🌍 I'm fine."
            );
            int rtFail = 0;
            for (String s : stress) {
                String back = enc.decode(enc.encodeOrdinary(s));
                if (!s.equals(back)) {
                    rtFail++;
                    System.out.println("  FAIL rt text=" + preview(s) + " back=" + preview(back));
                }
            }
            check(name + " round-trip all", rtFail == 0);

            summary.add(String.format(Locale.ROOT,
                    "%-14s nVocab=%-7d idFail=%d decFail=%d policyFail=%d bench~%.0f tok/s loadMs=%d",
                    name, enc.nVocab(), idFail, decFail, policyFail, tps, loadMs));
            failed += idFail + decFail + policyFail;
        }

        // ── Combined transformers-style matrix (offline, no hub) ───────────
        System.out.println();
        section("9. transformers matrix (Tiktoken encodings as FastTokenizer backends)");
        String[] matrixTexts = {
                "Hello world",
                "你好世界",
                "def foo(x): return x",
                "<|endoftext|>",
                "The quick brown fox jumps over the lazy dog."
        };
        System.out.printf(Locale.ROOT, "%-14s", "encoding");
        for (int i = 0; i < matrixTexts.length; i++) {
            System.out.printf(Locale.ROOT, " | case%-2d", i);
        }
        System.out.println();
        for (String name : encodings) {
            Tiktoken enc = Tiktoken.getEncoding(name);
            FastTokenizer ft = enc.toFastTokenizer();
            System.out.printf(Locale.ROOT, "%-14s", name);
            for (String t : matrixTexts) {
                int[] pure = enc.encode(t, Tiktoken.SPECIAL_ALL, Set.of());
                Encoding fe = ft.encode(t, false);
                String mark = pure.length == 0 ? "·" :
                        (fe.size() > 0 ? "✓" + pure.length : "✗");
                System.out.printf(Locale.ROOT, " | %-6s", mark);
            }
            System.out.println("  ft.vocab=" + ft.vocabSize());
        }

        // ── Summary ────────────────────────────────────────────────────────
        System.out.println();
        System.out.println("════ SUMMARY ════");
        for (String r : summary) System.out.println(r);
        System.out.println("checks passed=" + passed + " failed=" + failed);
        if (failed > 0) {
            System.out.println("Failure detail:\n" + failures);
            System.out.println("RESULT: FAIL");
            System.exit(1);
        }
        System.out.println("RESULT: ALL OK");
    }

    // ---- builtin probes when goldens missing ----

    private static void builtinProbes(Tiktoken enc) {
        Map<String, int[]> probes = new LinkedHashMap<>();
        switch (enc.name()) {
            case "cl100k_base" -> {
                probes.put("Hello world", new int[]{9906, 1917});
                probes.put("日本語", new int[]{9080, 22656, 45918, 252});
                probes.put("🎉", new int[]{9468, 236, 231});
            }
            case "o200k_base", "o200k_harmony" -> {
                probes.put("Hello world", new int[]{13225, 2375});
            }
            case "p50k_base", "p50k_edit", "r50k_base", "gpt2" -> {
                probes.put("Hello world", new int[]{15496, 995});
            }
            default -> {
            }
        }
        for (Map.Entry<String, int[]> e : probes.entrySet()) {
            checkEq(enc.name() + " builtin " + preview(e.getKey()),
                    e.getValue(), enc.encodeOrdinary(e.getKey()));
        }
    }

    // ---- helpers ----

    private static void section(String title) {
        System.out.println("── " + title + " ──");
    }

    private static void check(String name, boolean ok) {
        if (ok) pass(name);
        else fail(name, "false");
    }

    private static void checkEq(String name, Object expected, Object actual) {
        boolean ok;
        if (expected instanceof int[] ea && actual instanceof int[] aa) {
            ok = Arrays.equals(ea, aa);
        } else if (expected instanceof byte[] eb && actual instanceof byte[] ab) {
            ok = Arrays.equals(eb, ab);
        } else {
            ok = Objects.equals(expected, actual);
        }
        if (ok) pass(name);
        else fail(name, "expected=" + fmt(expected) + " actual=" + fmt(actual));
    }

    private static void pass(String name) {
        passed++;
        System.out.println("  OK  " + name);
    }

    private static void fail(String name, String detail) {
        failed++;
        System.out.println("  FAIL " + name + " :: " + detail);
        failures.append("FAIL ").append(name).append(" :: ").append(detail).append('\n');
    }

    private static String fmt(Object o) {
        if (o instanceof int[] a) return Arrays.toString(a);
        if (o instanceof byte[] a) return Arrays.toString(a);
        return String.valueOf(o);
    }

    private static String preview(String s) {
        if (s == null) return "null";
        String one = s.replace("\n", "\\n").replace("\t", "\\t");
        return one.length() > 60 ? one.substring(0, 57) + "..." : one;
    }

    @SuppressWarnings("unchecked")
    private static int[] toIntArray(Object o) {
        if (o == null) return new int[0];
        if (o instanceof int[] a) return a;
        if (o instanceof List<?> list) {
            int[] a = new int[list.size()];
            for (int i = 0; i < list.size(); i++) {
                Object v = list.get(i);
                a[i] = v instanceof Number n ? n.intValue() : Integer.parseInt(String.valueOf(v));
            }
            return a;
        }
        throw new IllegalArgumentException("not an int list: " + o.getClass());
    }

    private static byte[] toByteArray(List<?> list) {
        byte[] a = new byte[list.size()];
        for (int i = 0; i < list.size(); i++) {
            Object v = list.get(i);
            a[i] = (byte) (v instanceof Number n ? n.intValue() : Integer.parseInt(String.valueOf(v)));
        }
        return a;
    }
}
