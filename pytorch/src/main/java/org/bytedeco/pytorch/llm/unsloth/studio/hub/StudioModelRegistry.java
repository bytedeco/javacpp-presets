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

package org.bytedeco.pytorch.llm.unsloth.studio.hub;

import org.bytedeco.pytorch.llm.unsloth.studio.model.ModelCard;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/**
 * In-process model catalog + local inventory. Seeds well-known Unsloth-friendly
 * families (Llama / Qwen / Gemma / Mistral / Phi / gpt-oss) for search UX; actual
 * bytes still come from {@link StudioModelDownloader} / {@code llm.hub}.
 */
public final class StudioModelRegistry {

    private final Map<String, ModelCard> cards = new ConcurrentHashMap<>();
    private final Path modelsDir;

    public StudioModelRegistry(Path modelsDir) {
        this.modelsDir = modelsDir;
        seedCatalog();
        scanLocal();
    }

    public Optional<ModelCard> get(String id) {
        if (id == null) return Optional.empty();
        ModelCard c = cards.get(id);
        if (c != null) return Optional.of(c);
        // fuzzy: ignore org prefix case
        for (Map.Entry<String, ModelCard> e : cards.entrySet()) {
            if (e.getKey().equalsIgnoreCase(id) || e.getKey().endsWith("/" + id)
                    || e.getValue().displayName().equalsIgnoreCase(id)) {
                return Optional.of(e.getValue());
            }
        }
        return Optional.empty();
    }

    public ModelCard resolve(String idOrPath) {
        Optional<ModelCard> existing = get(idOrPath);
        if (existing.isPresent()) return existing.get();
        Path p = Path.of(idOrPath);
        if (Files.exists(p)) {
            ModelCard local = ModelCard.builder()
                    .id(idOrPath)
                    .displayName(p.getFileName().toString())
                    .localPath(p.toAbsolutePath().normalize())
                    .local(true)
                    .family(guessFamily(idOrPath))
                    .quant4bit(idOrPath.toLowerCase(Locale.ROOT).contains("bnb-4bit")
                            || idOrPath.toLowerCase(Locale.ROOT).contains("4bit"))
                    .build();
            cards.put(local.id(), local);
            return local;
        }
        // synthetic remote card
        ModelCard remote = ModelCard.builder()
                .id(idOrPath)
                .displayName(idOrPath)
                .family(guessFamily(idOrPath))
                .local(false)
                .quant4bit(idOrPath.toLowerCase(Locale.ROOT).contains("4bit"))
                .vision(idOrPath.toLowerCase(Locale.ROOT).contains("vl")
                        || idOrPath.toLowerCase(Locale.ROOT).contains("vision"))
                .moe(idOrPath.toLowerCase(Locale.ROOT).contains("a3b")
                        || idOrPath.toLowerCase(Locale.ROOT).contains("moe"))
                .build();
        cards.put(remote.id(), remote);
        return remote;
    }

    public List<ModelCard> search(String query) {
        String q = query == null ? "" : query.toLowerCase(Locale.ROOT).trim();
        List<ModelCard> out = new ArrayList<>();
        for (ModelCard c : cards.values()) {
            if (q.isEmpty()
                    || c.id().toLowerCase(Locale.ROOT).contains(q)
                    || c.displayName().toLowerCase(Locale.ROOT).contains(q)
                    || c.family().orElse("").toLowerCase(Locale.ROOT).contains(q)) {
                out.add(c);
            }
        }
        return out;
    }

    public List<ModelCard> listLocal() {
        List<ModelCard> out = new ArrayList<>();
        for (ModelCard c : cards.values()) if (c.local()) out.add(c);
        return out;
    }

    public void put(ModelCard card) {
        cards.put(card.id(), card);
    }

    public int size() { return cards.size(); }

    private void seedCatalog() {
        seed("unsloth/llama-3-8b-bnb-4bit", "llama", true, 8_000_000_000L);
        seed("unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", "llama", true, 8_000_000_000L);
        seed("unsloth/Llama-3.2-1B-Instruct", "llama", false, 1_000_000_000L);
        seed("unsloth/Llama-3.2-3B-Instruct", "llama", false, 3_000_000_000L);
        seed("unsloth/Qwen2.5-7B", "qwen", false, 7_000_000_000L);
        seed("unsloth/Qwen3-4B-Instruct-2507", "qwen", false, 4_000_000_000L);
        seed("unsloth/Qwen3-0.6B", "qwen", false, 600_000_000L);
        seed("unsloth/gemma-3-4b-it", "gemma", false, 4_000_000_000L);
        seed("unsloth/mistral-7b-instruct-v0.3-bnb-4bit", "mistral", true, 7_000_000_000L);
        seed("unsloth/Phi-4", "phi", false, 14_000_000_000L);
        seed("unsloth/gpt-oss-20b", "gpt-oss", false, 20_000_000_000L);
        seed("unsloth/tinyllama-bnb-4bit", "llama", true, 1_100_000_000L);
        seed("studio/tiny-gpt2", "gpt2", false, 124_000_000L); // always-available local synthetic
    }

    private void seed(String id, String family, boolean q4, long params) {
        cards.put(id, ModelCard.builder()
                .id(id).displayName(id).family(family).quant4bit(q4)
                .parameterCount(params)
                .ggufVariants(List.of("Q4_K_M", "Q5_K_M", "Q8_0", "F16"))
                .build());
    }

    private void scanLocal() {
        if (modelsDir == null || !Files.isDirectory(modelsDir)) return;
        try (var stream = Files.list(modelsDir)) {
            stream.filter(Files::isDirectory).forEach(dir -> {
                String id = "local/" + dir.getFileName();
                cards.put(id, ModelCard.builder()
                        .id(id)
                        .displayName(dir.getFileName().toString())
                        .localPath(dir.toAbsolutePath().normalize())
                        .local(true)
                        .family(guessFamily(dir.getFileName().toString()))
                        .build());
            });
        } catch (Exception ignored) {}
    }

    static String guessFamily(String id) {
        String s = id.toLowerCase(Locale.ROOT);
        if (s.contains("llama")) return "llama";
        if (s.contains("qwen")) return "qwen";
        if (s.contains("gemma")) return "gemma";
        if (s.contains("mistral") || s.contains("mixtral") || s.contains("pixtral")) return "mistral";
        if (s.contains("phi")) return "phi";
        if (s.contains("gpt-oss") || s.contains("gpt_oss")) return "gpt-oss";
        if (s.contains("whisper")) return "whisper";
        if (s.contains("embed") || s.contains("bge") || s.contains("minilm")) return "embedding";
        if (s.contains("gpt2") || s.contains("gpt-2")) return "gpt2";
        return "other";
    }
}
