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
package org.bytedeco.pytorch.llm.llamafactory.train;

import org.bytedeco.pytorch.llm.llamafactory.hparams.FactoryArgs;
import org.bytedeco.pytorch.llm.llamafactory.hparams.TrainingArgs;
import org.bytedeco.pytorch.llm.llamafactory.model.AdapterLoader;
import org.bytedeco.pytorch.llm.llamafactory.model.ModelLoader;
import org.bytedeco.pytorch.llm.peft.PeftModel;
import org.bytedeco.pytorch.utils.json.Json;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Checkpoint layout manager (LLaMA-Factory / HF Trainer subset).
 *
 * <p>Layout under {@code output_dir}:
 * <pre>
 *   checkpoint-{step}/
 *     trainer_state.json
 *     adapter_config.json + adapter_model.safetensors   (PEFT)
 *     OR full weights via host saver
 *   latest → checkpoint-{step}  (text pointer file)
 * </pre>
 */
public final class CheckpointManager {

    private static final Logger LOG = Logger.getLogger(CheckpointManager.class.getName());

    private final Path outputDir;
    private final int saveTotalLimit;
    private final TrainingArgs trainingArgs;

    public CheckpointManager(TrainingArgs trainingArgs) {
        this.trainingArgs = Objects.requireNonNull(trainingArgs, "trainingArgs");
        this.outputDir = Path.of(trainingArgs.outputDir() == null ? "saves/default" : trainingArgs.outputDir());
        this.saveTotalLimit = Math.max(0, trainingArgs.saveTotalLimit());
    }

    public static CheckpointManager from(FactoryArgs args) {
        return new CheckpointManager(args.training());
    }

    public Path outputDir() { return outputDir; }

    public Path checkpointDir(int step) {
        return outputDir.resolve("checkpoint-" + step);
    }

    /**
     * Save PEFT adapter (preferred) or write a trainer_state marker for full FT.
     *
     * @return directory written
     */
    public Path save(ModelLoader.LoadedModel loaded, int globalStep, Map<String, Double> metrics)
            throws IOException {
        Objects.requireNonNull(loaded, "loaded");
        Files.createDirectories(outputDir);
        Path dir = checkpointDir(globalStep);
        Files.createDirectories(dir);

        PeftModel peft = loaded.peft();
        if (peft != null) {
            AdapterLoader.save(peft, dir);
        } else {
            // Full / freeze: write a marker + card; host may plug weight export
            Map<String, Object> marker = new LinkedHashMap<>();
            marker.put("full_weights", true);
            marker.put("model", loaded.card().toMap());
            marker.put("note", "Full weight dump delegates to ModelExporter / host saver");
            writeJson(dir.resolve("pytorch_model_marker.json"), marker);
        }

        Map<String, Object> state = new LinkedHashMap<>();
        state.put("global_step", globalStep);
        state.put("metrics", metrics == null ? Map.of() : metrics);
        state.put("output_dir", outputDir.toString());
        state.put("finetuning", loaded.meta());
        state.put("model_card", loaded.card().toMap());
        writeJson(dir.resolve("trainer_state.json"), state);

        // latest pointer
        Files.writeString(outputDir.resolve("latest"), dir.getFileName().toString(), StandardCharsets.UTF_8);
        // also copy trainer_state to output root for quick resume discovery
        writeJson(outputDir.resolve("trainer_state.json"), state);

        pruneOldCheckpoints();
        LOG.info("Saved checkpoint to " + dir.toAbsolutePath());
        return dir;
    }

    /** Read global_step from a checkpoint dir or output_dir. */
    public int loadGlobalStep(Path dir) throws IOException {
        Path state = dir.resolve("trainer_state.json");
        if (!Files.isRegularFile(state)) {
            state = outputDir.resolve("trainer_state.json");
        }
        if (!Files.isRegularFile(state)) {
            return 0;
        }
        String json = Files.readString(state, StandardCharsets.UTF_8);
        Object parsed = tryParseJson(json);
        if (parsed instanceof Map<?, ?> m) {
            Object gs = m.get("global_step");
            if (gs instanceof Number n) return n.intValue();
            if (gs != null) {
                try { return Integer.parseInt(String.valueOf(gs)); }
                catch (NumberFormatException ignored) {}
            }
        }
        // naive fallback
        int idx = json.indexOf("\"global_step\"");
        if (idx >= 0) {
            String tail = json.substring(idx);
            String num = tail.replaceAll("[^0-9].*", "").replaceAll("^[^0-9]+", "");
            // better simple scan
            java.util.regex.Matcher mat = java.util.regex.Pattern
                    .compile("\"global_step\"\\s*:\\s*(\\d+)").matcher(json);
            if (mat.find()) return Integer.parseInt(mat.group(1));
        }
        return 0;
    }

    /** Resolve resume directory from TrainingArgs.resumeFromCheckpoint or latest. */
    public Path resolveResumeDir() {
        String resume = trainingArgs.resumeFromCheckpoint();
        if (resume != null && !resume.isBlank()) {
            Path p = Path.of(resume);
            if (Files.isDirectory(p)) return p;
            Path under = outputDir.resolve(resume);
            if (Files.isDirectory(under)) return under;
        }
        Path latestFile = outputDir.resolve("latest");
        if (Files.isRegularFile(latestFile)) {
            try {
                String name = Files.readString(latestFile, StandardCharsets.UTF_8).trim();
                Path dir = outputDir.resolve(name);
                if (Files.isDirectory(dir)) return dir;
            } catch (IOException ignored) {
            }
        }
        // highest checkpoint-*
        try (Stream<Path> s = Files.list(outputDir)) {
            return s.filter(Files::isDirectory)
                    .filter(p -> p.getFileName().toString().startsWith("checkpoint-"))
                    .max(Comparator.comparingInt(p -> {
                        String n = p.getFileName().toString().substring("checkpoint-".length());
                        try { return Integer.parseInt(n); }
                        catch (NumberFormatException e) { return -1; }
                    }))
                    .orElse(null);
        } catch (IOException e) {
            return null;
        }
    }

    public boolean shouldSave(int globalStep) {
        int every = trainingArgs.saveSteps();
        if (every <= 0) return false;
        return globalStep > 0 && globalStep % every == 0;
    }

    private void pruneOldCheckpoints() throws IOException {
        if (saveTotalLimit <= 0 || !Files.isDirectory(outputDir)) return;
        List<Path> ckpts;
        try (Stream<Path> s = Files.list(outputDir)) {
            ckpts = s.filter(Files::isDirectory)
                    .filter(p -> p.getFileName().toString().startsWith("checkpoint-"))
                    .sorted(Comparator.comparingInt(p -> {
                        try {
                            return Integer.parseInt(p.getFileName().toString()
                                    .substring("checkpoint-".length()));
                        } catch (Exception e) {
                            return 0;
                        }
                    }))
                    .collect(Collectors.toCollection(ArrayList::new));
        }
        while (ckpts.size() > saveTotalLimit) {
            Path old = ckpts.remove(0);
            deleteRecursive(old);
            LOG.info("Pruned old checkpoint " + old.getFileName());
        }
    }

    private static void deleteRecursive(Path p) throws IOException {
        if (!Files.exists(p)) return;
        if (Files.isDirectory(p)) {
            try (Stream<Path> s = Files.list(p)) {
                for (Path c : s.collect(Collectors.toList())) {
                    deleteRecursive(c);
                }
            }
        }
        Files.deleteIfExists(p);
    }

    private static void writeJson(Path path, Map<String, Object> map) throws IOException {
        String json;
        try {
            // prefer project Json util if it has stringify
            json = stringify(map);
        } catch (Throwable t) {
            json = manualJson(map);
        }
        Files.writeString(path, json, StandardCharsets.UTF_8);
    }

    private static String stringify(Map<String, Object> map) {
        try {
            var m = Json.class.getMethod("stringify", Object.class);
            Object r = m.invoke(null, map);
            if (r != null) return String.valueOf(r);
        } catch (ReflectiveOperationException ignored) {
        }
        try {
            var m = Json.class.getMethod("toJson", Object.class);
            Object r = m.invoke(null, map);
            if (r != null) return String.valueOf(r);
        } catch (ReflectiveOperationException ignored) {
        }
        return manualJson(map);
    }

    private static Object tryParseJson(String json) {
        try {
            var m = Json.class.getMethod("parse", String.class);
            return m.invoke(null, json);
        } catch (ReflectiveOperationException e) {
            try {
                var m = Json.class.getMethod("fromJson", String.class);
                return m.invoke(null, json);
            } catch (ReflectiveOperationException e2) {
                return null;
            }
        }
    }

    @SuppressWarnings("unchecked")
    private static String manualJson(Object o) {
        if (o == null) return "null";
        if (o instanceof String s) return quote(s);
        if (o instanceof Number || o instanceof Boolean) return String.valueOf(o);
        if (o instanceof Map<?, ?> map) {
            StringBuilder sb = new StringBuilder();
            sb.append('{');
            boolean first = true;
            for (Map.Entry<?, ?> e : map.entrySet()) {
                if (!first) sb.append(',');
                first = false;
                sb.append(quote(String.valueOf(e.getKey()))).append(':').append(manualJson(e.getValue()));
            }
            sb.append('}');
            return sb.toString();
        }
        if (o instanceof Iterable<?> it) {
            StringBuilder sb = new StringBuilder();
            sb.append('[');
            boolean first = true;
            for (Object x : it) {
                if (!first) sb.append(',');
                first = false;
                sb.append(manualJson(x));
            }
            sb.append(']');
            return sb.toString();
        }
        return quote(String.valueOf(o));
    }

    private static String quote(String s) {
        String esc = s.replace("\\", "\\\\").replace("\"", "\\\"")
                .replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t");
        return "\"" + esc + "\"";
    }
}
