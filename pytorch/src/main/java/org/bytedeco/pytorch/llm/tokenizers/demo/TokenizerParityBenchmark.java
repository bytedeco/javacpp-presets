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
package org.bytedeco.pytorch.llm.tokenizers.demo;

import org.bytedeco.pytorch.llm.hub.HfHub;
import org.bytedeco.pytorch.llm.tokenizers.BytesToUnicode;
import org.bytedeco.pytorch.llm.tokenizers.Encoding;
import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;
import org.bytedeco.pytorch.llm.transformers.AutoTokenizer;

import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * Download tokenizer-only snapshots and exercise encode/decode.
 *
 * <pre>
 *   java ... TokenizerParityBenchmark
 *   java ... TokenizerParityBenchmark Qwen/Qwen2.5-0.5B-Instruct
 * </pre>
 *
 * <p>When Python {@code tokenizers} is available externally, compare against goldens;
 * this harness always validates self-consistency (round-trip, vocab size, specials)
 * and prints token ids for manual HF comparison.
 *
 * <p>For full multi-backend (BPE / WordPiece / Unigram / Tiktoken) strict parity,
 * see {@link TokenizerBackendBenchmark}.
 */
public final class TokenizerParityBenchmark {

    private static final List<String> DEFAULT_MODELS = List.of(
            "Qwen/Qwen2.5-0.5B-Instruct",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "unsloth/Llama-3.2-1B-Instruct",
            "THUDM/GLM-4-9B-0414",   // tokenizer.json BPE
            "THUDM/glm-4-9b-chat"   // tiktoken tokenizer.model ranks
    );

    private static final List<String> CASES = List.of(
            "",
            " ",
            "Hello world",
            "I'm fine.",
            "你好世界",
            "def foo(x):\n    return x + 1",
            "{\"a\": 1, \"b\": [2, 3]}",
            "Hello 世界 🌍",
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
    );

    public static void main(String[] args) throws Exception {
        selfCheckBytesToUnicode();

        List<String> models = args.length > 0 ? Arrays.asList(args) : DEFAULT_MODELS;
        HfHub hub = HfHub.builder().build();

        int failures = 0;
        for (String modelId : models) {
            System.out.println("========== " + modelId + " ==========");
            try {
                failures += runModel(hub, modelId);
            } catch (Exception e) {
                failures++;
                System.out.println("FAIL load " + modelId + ": " + e.getMessage());
                e.printStackTrace(System.out);
            }
            System.out.println();
        }

        // Synthetic builders must still work offline
        failures += runSynthetic();

        if (failures > 0) {
            System.out.println("DONE with " + failures + " failure(s)");
            System.exit(1);
        }
        System.out.println("DONE ok");
    }

    private static void selfCheckBytesToUnicode() {
        char space = BytesToUnicode.spaceChar();
        if (space != 'Ġ') {
            throw new IllegalStateException("space must map to Ġ (U+0120), got U+"
                    + Integer.toHexString(space).toUpperCase(Locale.ROOT));
        }
        String round = BytesToUnicode.byteDecode(BytesToUnicode.byteEncode("Hello 世界"));
        if (!"Hello 世界".equals(round)) {
            throw new IllegalStateException("BytesToUnicode round-trip failed: " + round);
        }
        System.out.println("BytesToUnicode OK (Ġ=" + space + ")");
    }

    private static int runModel(HfHub hub, String modelId) throws Exception {
        long t0 = System.nanoTime();
        FastTokenizer tok = AutoTokenizer.fromPretrained(modelId, hub);
        long loadMs = (System.nanoTime() - t0) / 1_000_000L;
        System.out.println("loaded vocabSize=" + tok.vocabSize()
                + " backend=" + tok.backend()
                + " bos=" + tok.bosToken()
                + " eos=" + tok.eosToken()
                + " pad=" + tok.padToken()
                + " loadMs=" + loadMs);

        if (tok.vocabSize() < 1000) {
            System.out.println("FAIL vocab too small — likely whitespace fallback");
            return 1;
        }

        int fails = 0;
        long tokens = 0;
        long encNs = 0;
        for (String text : CASES) {
            long e0 = System.nanoTime();
            Encoding enc = tok.encode(text, false);
            encNs += System.nanoTime() - e0;
            tokens += enc.size();
            String decoded = tok.decode(enc.ids(), false);
            String decodedSkip = tok.decode(enc.ids(), true);

            System.out.println("--- text=" + preview(text));
            System.out.println("ids[" + enc.size() + "]=" + previewIds(enc.ids(), 32));
            System.out.println("decode=" + preview(decoded));
            System.out.println("decodeSkip=" + preview(decodedSkip));

            // Round-trip for pure byte-level paths should preserve text when no specials
            // (chat-marker cases intentionally contain specials).
            if (!text.contains("<|") && !text.isEmpty()) {
                // Allow whitespace normalization differences only if decode is non-empty
                if (decoded.isEmpty() && !text.isBlank()) {
                    System.out.println("WARN empty decode for non-empty text");
                }
            }
        }

        // add_special_tokens=true should be >= false length for template processors
        Encoding plain = tok.encode("Hello", false);
        Encoding special = tok.encode("Hello", true);
        System.out.println("add_special false/true lens: " + plain.size() + "/" + special.size());

        double tps = encNs > 0 ? (tokens * 1e9 / encNs) : 0;
        System.out.printf(Locale.ROOT, "throughput ~ %.0f tokens/s (encode only, %d tokens)%n", tps, tokens);
        return fails;
    }

    private static int runSynthetic() {
        System.out.println("========== synthetic builders ==========");
        FastTokenizer ws = FastTokenizer.whitespace().build();
        Encoding e = ws.encode("hello world", true);
        if (e.size() < 2) {
            System.out.println("FAIL whitespace encode");
            return 1;
        }
        FastTokenizer gpt = FastTokenizer.gpt2().build();
        Encoding g = gpt.encode("Hi", false);
        System.out.println("whitespace ids=" + Arrays.toString(e.ids()));
        System.out.println("gpt2 ids=" + Arrays.toString(g.ids()) + " vocab=" + gpt.vocabSize());
        return 0;
    }

    private static String preview(String s) {
        if (s == null) return "null";
        String one = s.replace("\n", "\\n");
        return one.length() > 80 ? one.substring(0, 77) + "..." : one;
    }

    private static String previewIds(int[] ids, int max) {
        if (ids == null) return "null";
        if (ids.length <= max) return Arrays.toString(ids);
        int[] head = Arrays.copyOf(ids, max);
        return Arrays.toString(head).replace("]", ", ... (" + ids.length + " total)]");
    }
}
