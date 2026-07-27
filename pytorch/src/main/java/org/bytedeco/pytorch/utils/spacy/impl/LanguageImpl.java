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
package org.bytedeco.pytorch.utils.spacy.impl;

import org.bytedeco.pytorch.utils.spacy.Doc;
import org.bytedeco.pytorch.utils.spacy.Language;
import org.bytedeco.pytorch.utils.spacy.PipelineComponent;
import org.bytedeco.pytorch.utils.spacy.pipeline.Matcher;
import org.bytedeco.pytorch.utils.spacy.pipeline.Sentencizer;
import org.bytedeco.pytorch.utils.spacy.tokenizer.SimpleTokenizer;
import org.bytedeco.pytorch.utils.spacy.Example;
import org.bytedeco.pytorch.utils.spacy.vocab.Vocab;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Default {@link Language} implementation with tokenizer + ordered pipeline.
 */
public final class LanguageImpl implements Language {

    private final String lang;
    private final Vocab vocab;
    private final SimpleTokenizer tokenizer;
    private final Map<String, PipelineComponent> pipes = new LinkedHashMap<>();
    private final List<String> disabled = new ArrayList<>();
    private final Map<String, Object> meta = new HashMap<>();
    private final Map<String, Object> config = new HashMap<>();

    public LanguageImpl() {
        this("en");
    }

    public LanguageImpl(String lang) {
        this.lang = lang == null ? "en" : lang;
        this.vocab = new Vocab();
        this.tokenizer = new SimpleTokenizer(vocab);
        this.meta.put("lang", this.lang);
        this.meta.put("version", org.bytedeco.pytorch.utils.spacy.Spacy.VERSION);
        this.config.put("lang", this.lang);
    }

    public void setMeta(String key, Object value) {
        meta.put(key, value);
    }

    @Override
    public Doc call(String text) {
        Doc doc = tokenizer.tokenize(text, this);
        for (Map.Entry<String, PipelineComponent> e : pipes.entrySet()) {
            if (disabled.contains(e.getKey())) {
                continue;
            }
            doc = e.getValue().apply(doc);
        }
        return doc;
    }

    @Override
    public Doc[] process(Collection<String> texts) {
        if (texts == null) {
            return new Doc[0];
        }
        return texts.stream().map(this::call).toArray(Doc[]::new);
    }

    @Override
    public Stream<Doc> pipe(Stream<String> texts) {
        return texts == null ? Stream.empty() : texts.map(this::call);
    }

    @Override
    public Stream<Doc> pipe(Stream<String> texts, int batchSize, int nProcess) {
        return pipe(texts);
    }

    @Override
    public void addPipe(String name, PipelineComponent component) {
        pipes.put(name, component);
    }

    @Override
    public PipelineComponent removePipe(String name) {
        disabled.remove(name);
        return pipes.remove(name);
    }

    @Override
    public PipelineComponent replacePipe(String oldName, PipelineComponent newComponent) {
        return pipes.put(oldName, newComponent);
    }

    @Override
    public void renamePipe(String oldName, String newName) {
        PipelineComponent c = pipes.remove(oldName);
        if (c != null) {
            pipes.put(newName, c);
            if (disabled.remove(oldName)) {
                disabled.add(newName);
            }
        }
    }

    @Override
    public void disablePipe(String name) {
        if (!disabled.contains(name)) {
            disabled.add(name);
        }
    }

    @Override
    public void enablePipe(String name) {
        disabled.remove(name);
    }

    @Override
    public List<Map.Entry<String, PipelineComponent>> pipeline() {
        return new ArrayList<>(pipes.entrySet());
    }

    @Override
    public List<String> pipeNames() {
        return new ArrayList<>(pipes.keySet());
    }

    @Override
    public PipelineComponent getPipe(String name) {
        return pipes.get(name);
    }

    @Override
    public PipelineComponent createPipe(String name, Map<String, Object> config) {
        if (name == null) {
            return null;
        }
        return switch (name.toLowerCase()) {
            case "sentencizer", "senter" -> new Sentencizer();
            case "matcher" -> new Matcher();
            default -> null;
        };
    }

    @Override
    public void initialize(java.util.function.Supplier<Iterable<Example>> getExamples) {
        initialize();
    }

    @Override
    public void initialize() {
        // ensure string store has basic entries
        vocab.strings().add(lang);
    }

    @Override
    public void toDisk(Path path) throws Exception {
        Files.createDirectories(path);
        Files.writeString(path.resolve(".spacy-java"),
                "lang=" + lang + "\nversion=" + org.bytedeco.pytorch.utils.spacy.Spacy.VERSION + "\n");
        Files.writeString(path.resolve("meta.json"),
                "{\"lang\":\"" + lang + "\",\"pipes\":"
                        + pipeNames().stream().map(s -> "\"" + s + "\"").collect(Collectors.joining(",", "[", "]"))
                        + "}");
    }

    @Override
    public void fromDisk(Path path) throws Exception {
        Path marker = path.resolve(".spacy-java");
        if (!Files.exists(marker)) {
            throw new IllegalArgumentException("Model not found at " + path);
        }
    }

    @Override
    public Map<String, Double> update(Iterable<Example> examples, int batchSize, Map<String, Double> losses) {
        // training shell — returns zero losses
        Map<String, Double> out = losses == null ? new HashMap<>() : new HashMap<>(losses);
        out.putIfAbsent("loss", 0.0);
        return out;
    }

    @Override
    public Map<String, Object> evaluate(Iterable<Example> examples) {
        Map<String, Object> m = new HashMap<>();
        m.put("score", 0.0);
        return m;
    }

    @Override
    public Object config() {
        return config;
    }

    @Override
    public String info() {
        return "Language(lang=" + lang + ", pipes=" + pipeNames() + ")";
    }

    @Override
    public String lang() {
        return lang;
    }

    @Override
    public Vocab vocab() {
        return vocab;
    }

    @Override
    public Map<String, Object> meta() {
        return meta;
    }

    public SimpleTokenizer tokenizer() {
        return tokenizer;
    }
}
