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
package org.bytedeco.pytorch.utils.spacy;
import org.bytedeco.pytorch.data.*;

import org.bytedeco.pytorch.utils.spacy.Example;
import org.bytedeco.pytorch.utils.spacy.vocab.Vocab;

import java.nio.file.Path;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

/**
 * Language is the main NLP entrypoint (the {@code nlp} object).
 */
public interface Language {

    // ---- Core processing ----

    /** Process text → Doc (spaCy {@code nlp(text)}). */
    Doc call(String text);

    /** Alias of {@link #call(String)}. */
    default Doc apply(String text) {
        return call(text);
    }

    /** Alias of {@link #call(String)}. */
    default Doc process(String text) {
        return call(text);
    }

    Doc[] process(Collection<String> texts);

    Stream<Doc> pipe(Stream<String> texts);

    Stream<Doc> pipe(Stream<String> texts, int batchSize, int nProcess);

    // ---- Pipeline management ----

    void addPipe(String name, PipelineComponent component);

    PipelineComponent removePipe(String name);

    PipelineComponent replacePipe(String oldName, PipelineComponent newComponent);

    void renamePipe(String oldName, String newName);

    void disablePipe(String name);

    void enablePipe(String name);

    List<Map.Entry<String, PipelineComponent>> pipeline();

    List<String> pipeNames();

    PipelineComponent getPipe(String name);

    PipelineComponent createPipe(String name, Map<String, Object> config);

    // ---- Model lifecycle ----

    void initialize(java.util.function.Supplier<Iterable<Example>> getExamples);

    void initialize();

    void toDisk(Path path) throws Exception;

    void fromDisk(Path path) throws Exception;

    // ---- Training / evaluation shell ----

    Map<String, Double> update(Iterable<Example> examples, int batchSize, Map<String, Double> losses);

    Map<String, Object> evaluate(Iterable<Example> examples);

    // ---- Metadata ----

    Object config();

    String info();

    String lang();

    Vocab vocab();

    Map<String, Object> meta();
}
