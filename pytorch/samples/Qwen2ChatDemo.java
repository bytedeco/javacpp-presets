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

import org.bytedeco.pytorch.llm.hub.HfHub;
import org.bytedeco.pytorch.llm.transformers.AutoModelForCausalLM;
import org.bytedeco.pytorch.llm.transformers.generation.GenerationConfig;
import org.bytedeco.pytorch.llm.transformers.pipeline.TextGenerationPipeline;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/**
 * End-to-end Qwen2 chat demo.
 *
 * <p>Usage:
 * <pre>
 *   # offline tiny (no network, random weights — structure smoke only)
 *   java Qwen2ChatDemo --tiny
 *
 *   # real Hub model (requires network + ~1GB disk; set HF_TOKEN if gated)
 *   java Qwen2ChatDemo Qwen/Qwen2-0.5B-Instruct
 *
 *   # local snapshot directory
 *   java Qwen2ChatDemo --dir /path/to/snapshot
 * </pre>
 */
public final class Qwen2ChatDemo {

    public static void main(String[] args) throws Exception {
        String mode = args.length == 0 ? "--tiny" : args[0];

        TextGenerationPipeline pipe;
        if ("--tiny".equals(mode)) {
            System.out.println("== tiny offline Qwen2 (random weights, structure smoke) ==");
            pipe = TextGenerationPipeline.tiny("qwen2");
        } else if ("--dir".equals(mode)) {
            if (args.length < 2) {
                System.err.println("Usage: Qwen2ChatDemo --dir <snapshot>");
                System.exit(2);
                return;
            }
            Path dir = Path.of(args[1]);
            System.out.println("== load from directory " + dir + " ==");
            pipe = TextGenerationPipeline.fromDirectory(dir);
        } else {
            String modelId = mode;
            String token = System.getenv("HF_TOKEN");
            HfHub hub = HfHub.create()
                    .token(token)
                    .logger(System.out::println)
                    .build();
            System.out.println("== from_pretrained " + modelId + " ==");
            pipe = TextGenerationPipeline.fromPretrained(modelId, hub);
        }

        AutoModelForCausalLM.Bundle b = pipe.bundle();
        System.out.println("config: " + b.config());
        if (b.loadReport() != null) {
            System.out.println("load:  " + b.loadReport());
        }
        System.out.println("tok:   backend=" + b.tokenizer().backend()
                + " vocab≈" + b.tokenizer().vocabSize());

        GenerationConfig gen = GenerationConfig.builder()
                .doSample(false)
                .maxNewTokens(64)
                .eosTokenId(b.config().eosTokenId())
                .build();

        // English factual smoke
        String en = pipe.chat(List.of(
                Map.of("role", "system", "content", "You are a helpful assistant."),
                Map.of("role", "user", "content", "What is 2+2? Reply with only the digit.")
        ), gen);
        System.out.println("\n[EN] " + en);

        // Chinese smoke
        String zh = pipe.chat(List.of(
                Map.of("role", "user", "content", "用一句话介绍杭州")
        ), gen);
        System.out.println("\n[ZH] " + zh);

        // Multi-turn
        String mt = pipe.chat(List.of(
                Map.of("role", "user", "content", "My name is Ada."),
                Map.of("role", "assistant", "content", "Nice to meet you, Ada!"),
                Map.of("role", "user", "content", "What is my name? Reply with only the name.")
        ), gen);
        System.out.println("\n[MT] " + mt);
    }
}
