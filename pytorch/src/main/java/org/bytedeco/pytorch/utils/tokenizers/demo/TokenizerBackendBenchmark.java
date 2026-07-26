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

import org.bytedeco.pytorch.utils.hub.HfHub;
import org.bytedeco.pytorch.utils.json.Json;
import org.bytedeco.pytorch.utils.tokenizers.Encoding;
import org.bytedeco.pytorch.utils.tokenizers.FastTokenizer;
import org.bytedeco.pytorch.utils.transformers.AutoTokenizer;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Multi-backend tokenizer parity + throughput benchmark.
 *
 * <p>Covers algorithms used by real model dictionaries:
 * <ul>
 *   <li><b>BPE</b> — GPT-2 / Qwen / Llama / DeepSeek / GLM-4 (tokenizer.json)</li>
 *   <li><b>WordPiece</b> — BERT</li>
 *   <li><b>Unigram</b> (+ Metaspace) — T5 / XLM-R (SentencePiece exported to tokenizer.json)</li>
 *   <li><b>TiktokenBPE</b> — ChatGLM4 {@code tokenizer.model} rank table</li>
 * </ul>
 *
 * <p>For each model, loads HF goldens from
 * {@code src/test/resources/tokenizers/goldens/<name>/cases.jsonl}
 * (generated via Python transformers) and asserts <b>exact encode id parity</b>.
 * Also reports encode throughput (tokens/s).
 *
 * <pre>
 *   java ... TokenizerBackendBenchmark
 *   java ... TokenizerBackendBenchmark bert_uncased t5_small glm4_chat
 * </pre>
 */
public final class TokenizerBackendBenchmark {

    public record Spec(String goldenDir, String modelId, String algorithm) {}

    /** Default suite — one+ models per algorithm family. */
    public static final List<Spec> DEFAULT_SPECS = List.of(
            // WordPiece
            new Spec("bert_uncased", "bert-base-uncased", "WordPiece"),
            new Spec("bert_chinese", "bert-base-chinese", "WordPiece"),
            // Unigram / SentencePiece-via-JSON
            new Spec("t5_small", "t5-small", "Unigram/SP"),
            new Spec("xlm_roberta", "xlm-roberta-base", "Unigram/SP"),
            // TiktokenBPE
            new Spec("glm4_chat", "THUDM/glm-4-9b-chat", "TiktokenBPE"),
            // BPE
            new Spec("gpt2", "gpt2", "BPE"),
            new Spec("qwen2.5", "Qwen/Qwen2.5-0.5B-Instruct", "BPE"),
            new Spec("llama32", "unsloth/Llama-3.2-1B-Instruct", "BPE"),
            new Spec("deepseek", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "BPE"),
            new Spec("glm4_0414", "THUDM/GLM-4-9B-0414", "BPE")
    );

    public static void main(String[] args) throws Exception {
        Path root = Path.of("src/test/resources/tokenizers/goldens");
        List<Spec> specs = select(args);

        HfHub hub = HfHub.builder().build();
        int hardFail = 0;
        List<String> rows = new ArrayList<>();

        System.out.println("Tokenizer multi-backend parity + throughput");
        System.out.println("goldens root: " + root.toAbsolutePath());
        System.out.println();

        for (Spec spec : specs) {
            System.out.println("========== [" + spec.algorithm + "] "
                    + spec.goldenDir + " / " + spec.modelId + " ==========");
            long t0 = System.nanoTime();
            FastTokenizer tok;
            try {
                tok = AutoTokenizer.fromPretrained(spec.modelId, hub);
            } catch (Exception e) {
                System.out.println("LOAD FAIL: " + e.getMessage());
                e.printStackTrace(System.out);
                hardFail++;
                rows.add(String.format(Locale.ROOT, "%-14s %-12s LOAD_FAIL", spec.goldenDir, spec.algorithm));
                continue;
            }
            long loadMs = (System.nanoTime() - t0) / 1_000_000L;
            System.out.println("backend=" + tok.backend()
                    + " model=" + tok.pipeline().model().getClass().getSimpleName()
                    + " vocab=" + tok.vocabSize()
                    + " loadMs=" + loadMs);

            Path cases = root.resolve(spec.goldenDir).resolve("cases.jsonl");
            if (!Files.isRegularFile(cases)) {
                System.out.println("MISSING goldens: " + cases);
                hardFail++;
                continue;
            }

            int ok = 0, idFail = 0, decFail = 0;
            long tokens = 0, encNs = 0, decNs = 0;
            // throughput warmup + measured loop on golden texts
            List<Map<String, Object>> rowsIn = new ArrayList<>();
            for (String line : Files.readAllLines(cases)) {
                if (line.isBlank()) continue;
                rowsIn.add(Json.decodeObject(line));
            }

            for (Map<String, Object> m : rowsIn) {
                String text = m.get("text") instanceof String s ? s : String.valueOf(m.getOrDefault("text", ""));
                boolean add = Boolean.TRUE.equals(m.get("add_special_tokens"));
                @SuppressWarnings("unchecked")
                List<Object> idList = (List<Object>) m.get("ids");
                int[] goldIds = new int[idList.size()];
                for (int i = 0; i < idList.size(); i++) {
                    goldIds[i] = ((Number) idList.get(i)).intValue();
                }

                long e0 = System.nanoTime();
                Encoding enc = tok.encode(text, add);
                encNs += System.nanoTime() - e0;
                tokens += enc.size();

                long d0 = System.nanoTime();
                String decSkip = tok.decode(enc.ids(), true);
                decNs += System.nanoTime() - d0;

                String goldSkip = m.get("decoded_skip_special") == null
                        ? null : String.valueOf(m.get("decoded_skip_special"));

                boolean idsOk = Arrays.equals(enc.ids(), goldIds);
                boolean decOk = Objects.equals(decSkip, goldSkip);
                if (!idsOk) {
                    idFail++;
                    System.out.println("FAIL ids add=" + add + " text=" + preview(text));
                    System.out.println("  java=" + Arrays.toString(enc.ids()));
                    System.out.println("  gold=" + Arrays.toString(goldIds));
                } else if (!decOk) {
                    decFail++;
                    System.out.println("FAIL decode-skip add=" + add + " text=" + preview(text));
                    System.out.println("  java=" + preview(decSkip));
                    System.out.println("  gold=" + preview(goldSkip));
                } else {
                    ok++;
                }
            }

            // extra throughput pass: repeat encode on a fixed batch
            List<String> benchTexts = List.of(
                    "Hello world",
                    "The quick brown fox jumps over the lazy dog.",
                    "你好世界，今天天气不错。",
                    "def foo(x):\n    return x + 1\n",
                    "a".repeat(256)
            );
            int rounds = 200;
            long bTokens = 0;
            long bNs = 0;
            for (int r = 0; r < rounds; r++) {
                for (String bt : benchTexts) {
                    long s = System.nanoTime();
                    Encoding be = tok.encode(bt, false);
                    bNs += System.nanoTime() - s;
                    bTokens += be.size();
                }
            }

            double parityTps = encNs > 0 ? (tokens * 1e9 / encNs) : 0;
            double benchTps = bNs > 0 ? (bTokens * 1e9 / bNs) : 0;
            double decTps = decNs > 0 ? (tokens * 1e9 / decNs) : 0;
            System.out.printf(Locale.ROOT,
                    "parity ok=%d idFail=%d decFail=%d  parityEncode~%.0f tok/s  benchEncode~%.0f tok/s  decode~%.0f tok/s%n",
                    ok, idFail, decFail, parityTps, benchTps, decTps);

            rows.add(String.format(Locale.ROOT,
                    "%-14s %-12s backend=%-10s vocab=%-7d idFail=%d decFail=%d bench~%.0f tok/s",
                    spec.goldenDir, spec.algorithm, tok.backend(), tok.vocabSize(),
                    idFail, decFail, benchTps));
            hardFail += idFail;
            System.out.println();
        }

        System.out.println("==== SUMMARY ====");
        for (String r : rows) System.out.println(r);
        System.out.println(hardFail == 0
                ? "ALL BACKENDS ID PARITY OK"
                : "TOTAL ID FAIL=" + hardFail);
        if (hardFail != 0) System.exit(1);
    }

    private static List<Spec> select(String[] args) {
        if (args == null || args.length == 0) return DEFAULT_SPECS;
        Map<String, Spec> by = new LinkedHashMap<>();
        for (Spec s : DEFAULT_SPECS) by.put(s.goldenDir, s);
        List<Spec> out = new ArrayList<>();
        for (String a : args) {
            Spec s = by.get(a);
            if (s == null) {
                // allow model id passthrough as custom BPE-only (no golden)
                System.err.println("Unknown golden dir '" + a + "', expected one of " + by.keySet());
            } else {
                out.add(s);
            }
        }
        return out.isEmpty() ? DEFAULT_SPECS : out;
    }

    private static String preview(String s) {
        if (s == null) return "null";
        String one = s.replace("\n", "\\n");
        return one.length() > 80 ? one.substring(0, 77) + "..." : one;
    }
}
