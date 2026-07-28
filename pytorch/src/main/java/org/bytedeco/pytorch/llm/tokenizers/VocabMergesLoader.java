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
package org.bytedeco.pytorch.llm.tokenizers;

import org.bytedeco.pytorch.utils.json.Json;
import org.bytedeco.pytorch.llm.tokenizers.decoders.Decoder;
import org.bytedeco.pytorch.llm.tokenizers.models.BpeModel;
import org.bytedeco.pytorch.llm.tokenizers.normalizers.Normalizer;
import org.bytedeco.pytorch.llm.tokenizers.pretokenizers.PreTokenizer;
import org.bytedeco.pytorch.llm.tokenizers.processors.PostProcessor;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * GPT-2 style fallback loader: {@code vocab.json} + {@code merges.txt} → ByteLevel BPE pipeline.
 */
public final class VocabMergesLoader {

    private VocabMergesLoader() {}

    public static TokenizerPipeline load(Path vocabJson, Path mergesTxt) throws IOException {
        Map<String, Integer> vocab = loadVocab(vocabJson);
        List<String> merges = loadMerges(mergesTxt);
        return buildPipeline(vocab, merges);
    }

    public static TokenizerPipeline loadFromDirectory(Path dir) throws IOException {
        Path vocab = dir.resolve("vocab.json");
        Path merges = dir.resolve("merges.txt");
        if (!Files.isRegularFile(vocab) || !Files.isRegularFile(merges)) {
            throw new IOException("Missing vocab.json and/or merges.txt in " + dir);
        }
        return load(vocab, merges);
    }

    public static boolean present(Path dir) {
        return Files.isRegularFile(dir.resolve("vocab.json"))
                && Files.isRegularFile(dir.resolve("merges.txt"));
    }

    @SuppressWarnings("unchecked")
    public static Map<String, Integer> loadVocab(Path vocabJson) throws IOException {
        String raw = Files.readString(vocabJson, StandardCharsets.UTF_8);
        Map<String, Object> root = Json.decodeObject(raw);
        Map<String, Integer> vocab = new LinkedHashMap<>(root.size() * 2);
        for (Map.Entry<String, Object> e : root.entrySet()) {
            Integer id = JsonMaps.asInt(e.getValue());
            if (id != null) vocab.put(e.getKey(), id);
        }
        return vocab;
    }

    public static List<String> loadMerges(Path mergesTxt) throws IOException {
        List<String> merges = new ArrayList<>();
        try (BufferedReader br = Files.newBufferedReader(mergesTxt, StandardCharsets.UTF_8)) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty() || line.startsWith("#") || line.startsWith("version")) continue;
                merges.add(line);
            }
        }
        return merges;
    }

    private static TokenizerPipeline buildPipeline(Map<String, Integer> vocab, List<String> merges) {
        String unk = vocab.containsKey("<|endoftext|>") ? "<|endoftext|>"
                : vocab.containsKey("<unk>") ? "<unk>" : null;
        BpeModel model = new BpeModel(vocab, merges, unk, null, null, false, false, false);
        return new TokenizerPipeline(
                Normalizer.NOP,
                new PreTokenizer.ByteLevelPreTokenizer(false, true, true),
                model,
                PostProcessor.NOP,
                Decoder.ByteLevelDecoder.INSTANCE,
                AddedVocabulary.empty(),
                null, null,
                unk, unk, null, null, unk, unk, null,
                1024, false
        );
    }
}
