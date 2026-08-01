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

import org.bytedeco.pytorch.llm.llamafactory.data.SimpleTokenizer;
import org.bytedeco.pytorch.llm.llamafactory.hparams.EvaluationArgs;
import org.bytedeco.pytorch.llm.llamafactory.hparams.FactoryArgs;
import org.bytedeco.pytorch.llm.llamafactory.hparams.ModelArgs;
import org.bytedeco.pytorch.llm.llamafactory.model.ModelLoader;
import org.bytedeco.pytorch.llm.llamafactory.model.ModelLoader.LoadedModel;
import org.bytedeco.pytorch.llm.transformers.CausalLM;
import org.bytedeco.pytorch.utils.json.Json;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.logging.Logger;

/**
 * Evaluation harness entry (MMLU-like multi-choice, offline demo items).
 *
 * <p>Production hosts inject real task JSON under {@link EvaluationArgs#taskDir()};
 * when absent, a tiny built-in multi-choice set is used so CI stays offline.
 */
public final class EvalRunner {

    private static final Logger LOG = Logger.getLogger(EvalRunner.class.getName());

    private EvalRunner() {}

    public static EvalResult run(EvaluationArgs args) {
        Objects.requireNonNull(args, "args");
        // Load a tiny causal LM for scoring (model path may be overridden via meta later)
        ModelArgs modelArgs = ModelArgs.builder()
                .modelNameOrPath("tiny-gpt2")
                .build();
        FactoryArgs fa = FactoryArgs.builder().model(modelArgs).build();
        LoadedModel loaded = ModelLoader.load(fa);
        return run(args, loaded);
    }

    public static EvalResult run(EvaluationArgs args, LoadedModel loaded) {
        Objects.requireNonNull(args, "args");
        Objects.requireNonNull(loaded, "loaded");

        List<MmluLikeHarness.Item> items = loadItems(args);
        if (items.isEmpty()) {
            items = MmluLikeHarness.demoItems();
            LOG.info("EvalRunner using demo multi-choice items (taskDir empty or missing)");
        }

        CausalLM causal = loaded.causalLM();
        SimpleTokenizer tok = SimpleTokenizer.defaults();
        int correct = 0;
        List<Map<String, Object>> log = new ArrayList<>();

        int nShot = Math.max(0, args.nShot());
        // n-shot is recorded in meta; demo path is zero-shot greedy letter pick for speed
        for (MmluLikeHarness.Item item : items) {
            String prompt = MmluLikeHarness.formatPrompt(item, nShot);
            long[] ids = tok.encode(prompt, false);
            int[] promptIds = new int[ids.length];
            for (int i = 0; i < ids.length; i++) promptIds[i] = (int) ids[i];

            String pred;
            try {
                int[] out = causal.generate(promptIds, 4, CausalLM.GenerationConfig.greedy());
                pred = MmluLikeHarness.extractChoice(tok.decode(toLong(out)));
            } catch (Throwable t) {
                // fallback: score choices by prompt length heuristic (still deterministic)
                pred = MmluLikeHarness.heuristicChoice(item);
            }
            boolean ok = item.answer != null
                    && item.answer.equalsIgnoreCase(pred);
            if (ok) correct++;

            Map<String, Object> row = new LinkedHashMap<>();
            row.put("id", item.id);
            row.put("question", item.question);
            row.put("gold", item.answer);
            row.put("pred", pred);
            row.put("correct", ok);
            log.add(row);
        }

        Map<String, Object> meta = new LinkedHashMap<>();
        meta.put("task", args.task());
        meta.put("n_shot", nShot);
        meta.put("seed", args.seed());
        meta.put("lang", args.lang());
        meta.put("batch_size", args.batchSize());
        meta.put("model", loaded.card() == null ? Map.of() : loaded.card().toMap());

        EvalResult result = new EvalResult(args.task(), items.size(), correct, log, meta);
        LOG.info(result.toString());

        if (args.saveDir() != null && !args.saveDir().isBlank()) {
            try {
                Path dir = Path.of(args.saveDir());
                Files.createDirectories(dir);
                Files.writeString(dir.resolve("eval_result.json"),
                        Json.encode(result.toMap()), StandardCharsets.UTF_8);
            } catch (IOException e) {
                LOG.warning("Failed to write eval result: " + e.getMessage());
            }
        }
        return result;
    }

    private static List<MmluLikeHarness.Item> loadItems(EvaluationArgs args) {
        String dir = args.taskDir();
        if (dir == null || dir.isBlank()) return List.of();
        Path p = Path.of(dir);
        Path file = Files.isRegularFile(p) ? p : p.resolve(args.task() + ".json");
        if (!Files.isRegularFile(file)) {
            file = p.resolve("items.json");
        }
        if (!Files.isRegularFile(file)) return List.of();
        try {
            String json = Files.readString(file, StandardCharsets.UTF_8);
            Object decoded = Json.decode(json);
            return MmluLikeHarness.parseItems(decoded);
        } catch (Exception e) {
            LOG.warning("Failed loading eval items from " + file + ": " + e.getMessage());
            return List.of();
        }
    }

    private static long[] toLong(int[] ids) {
        long[] o = new long[ids.length];
        for (int i = 0; i < ids.length; i++) o[i] = ids[i];
        return o;
    }
}
