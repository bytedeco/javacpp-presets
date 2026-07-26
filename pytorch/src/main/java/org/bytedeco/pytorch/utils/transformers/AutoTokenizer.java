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
package org.bytedeco.pytorch.utils.transformers;

import org.bytedeco.pytorch.utils.hub.HfHub;
import org.bytedeco.pytorch.utils.tokenizers.DirectoryTokenizerLoader;
import org.bytedeco.pytorch.utils.tokenizers.FastTokenizer;
import org.bytedeco.pytorch.utils.tokenizers.Tiktoken;
import org.bytedeco.pytorch.utils.transformers.tokenization.ChatTemplate;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;

/**
 * HuggingFace {@code AutoTokenizer.from_pretrained} entry point.
 *
 * <p>Loads a real tokenizers-rs {@code tokenizer.json} (or {@code vocab.json}+{@code merges.txt})
 * from a Hub snapshot or local directory via {@link DirectoryTokenizerLoader}.
 * Use {@link ChatTemplate} for Instruct chat formatting.
 *
 * <p>Also resolves OpenAI-style model ids / tiktoken encoding names directly to the
 * pure-Java {@link Tiktoken} backend (no Hub download), e.g.
 * {@code AutoTokenizer.fromPretrained("gpt-4o")} or {@code "cl100k_base"}.
 */
public final class AutoTokenizer {

    /** Default Hub files sufficient to build a tokenizer (no weights). */
    public static final List<String> TOKENIZER_ONLY_FILES = List.of(
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "tokenizer.model",
            "spiece.model",
            "config.json"
    );

    private AutoTokenizer() {}

    public static FastTokenizer fromPretrained(String modelId, HfHub hub) throws IOException {
        FastTokenizer tik = tryTiktoken(modelId);
        if (tik != null) return tik;
        Path snap = hub.snapshotDownload(modelId, "main", "models", TOKENIZER_ONLY_FILES);
        return fromDirectory(snap);
    }

    /**
     * Download only tokenizer artifacts for {@code modelId} and load them.
     * OpenAI / tiktoken ids short-circuit to the bundled Java encodings.
     */
    public static FastTokenizer fromPretrainedTokenizerOnly(String modelId, HfHub hub) throws IOException {
        return fromPretrained(modelId, hub);
    }

    /**
     * Resolve without Hub: built-in tiktoken encoding name, OpenAI model id, or local directory.
     */
    public static FastTokenizer fromPretrained(String modelIdOrPath) throws IOException {
        FastTokenizer tik = tryTiktoken(modelIdOrPath);
        if (tik != null) return tik;
        Path p = Path.of(modelIdOrPath);
        if (Files.isDirectory(p)) return fromDirectory(p);
        throw new IOException("Not a local directory and not a known tiktoken encoding/model: "
                + modelIdOrPath + " (pass HfHub for Hub models)");
    }

    public static FastTokenizer fromDirectory(Path dir) throws IOException {
        if (dir == null || !Files.isDirectory(dir)) {
            throw new IOException("Not a directory: " + dir);
        }
        return DirectoryTokenizerLoader.load(dir);
    }

    public static FastTokenizer whitespace() {
        return FastTokenizer.whitespace().build();
    }

    public static FastTokenizer gpt2() {
        return FastTokenizer.gpt2().build();
    }

    /** OpenAI cl100k_base (GPT-4 / ChatGPT / embeddings). */
    public static FastTokenizer cl100kBase() {
        return Tiktoken.getEncoding(Tiktoken.CL100K_BASE).toFastTokenizer();
    }

    /** OpenAI o200k_base (GPT-4o / o1 / o3). */
    public static FastTokenizer o200kBase() {
        return Tiktoken.getEncoding(Tiktoken.O200K_BASE).toFastTokenizer();
    }

    /** Bundled pure-Java tiktoken encoding by name. */
    public static FastTokenizer tiktoken(String encodingName) {
        return Tiktoken.getEncoding(encodingName).toFastTokenizer();
    }

    /** Bundled pure-Java tiktoken encoding for an OpenAI model id. */
    public static FastTokenizer tiktokenForModel(String modelName) {
        return Tiktoken.encodingForModel(modelName).toFastTokenizer();
    }

    /** Detect chat template for a snapshot (Qwen ChatML / Llama-3 / Mistral). */
    public static ChatTemplate chatTemplate(Path dir, PretrainedConfig cfg) {
        return ChatTemplate.detect(dir, cfg);
    }

    /**
     * If {@code id} is a known tiktoken encoding name or OpenAI model id, return its
     * {@link FastTokenizer} adapter; otherwise {@code null}.
     */
    static FastTokenizer tryTiktoken(String id) {
        if (id == null || id.isBlank()) return null;
        String key = id.trim();
        // Direct encoding name
        for (String name : Tiktoken.listEncodingNames()) {
            if (name.equalsIgnoreCase(key)) {
                return Tiktoken.getEncoding(name).toFastTokenizer();
            }
        }
        // OpenAI model id / prefix
        try {
            return Tiktoken.encodingForModel(key).toFastTokenizer();
        } catch (IllegalArgumentException ignored) {
            try {
                return Tiktoken.encodingForModel(key.toLowerCase(Locale.ROOT)).toFastTokenizer();
            } catch (IllegalArgumentException ignored2) {
                return null;
            }
        }
    }
}
