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
package org.bytedeco.pytorch.utils.tokenizers;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Resolve a tokenizer from a HuggingFace snapshot directory.
 *
 * <ol>
 *   <li>{@code tokenizer.json} (full tokenizers-rs schema)</li>
 *   <li>{@code vocab.json} + {@code merges.txt} (GPT-2 style)</li>
 *   <li>whitespace fallback (smoke / tiny models only)</li>
 * </ol>
 */
public final class DirectoryTokenizerLoader {

    private DirectoryTokenizerLoader() {}

    public static FastTokenizer load(Path dir) throws IOException {
        if (dir == null || !Files.isDirectory(dir)) {
            throw new IOException("Not a directory: " + dir);
        }

        Path tj = dir.resolve("tokenizer.json");
        if (Files.isRegularFile(tj)) {
            // fromFile already overlays sibling tokenizer_config / special_tokens_map
            return FastTokenizer.fromFile(tj);
        }

        if (VocabMergesLoader.present(dir)) {
            TokenizerPipeline pipe = VocabMergesLoader.loadFromDirectory(dir);
            pipe = TokenizerJsonLoader.applyTokenizerConfig(pipe, dir.resolve("tokenizer_config.json"));
            pipe = TokenizerJsonLoader.applySpecialTokensMap(pipe, dir.resolve("special_tokens_map.json"));
            return FastTokenizer.of(pipe);
        }

        // ChatGLM4 / tiktoken text ranks (tokenizer.model base64 lines)
        if (TiktokenModelLoader.present(dir)) {
            TokenizerPipeline pipe = TiktokenModelLoader.loadFromDirectory(dir);
            return FastTokenizer.of(pipe);
        }

        // Last resort — keeps tiny offline demos working
        return FastTokenizer.whitespace().build();
    }
}
