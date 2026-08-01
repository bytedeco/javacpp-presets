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

package org.bytedeco.pytorch.llm.unsloth.studio.rag;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/** Simple retrieve-then-stuff pipeline for grounded chat. */
public final class RagPipeline {

    private final PdfReader pdfReader;
    private final WebSearchClient web;
    private final Chunker chunker;

    public RagPipeline(PdfReader pdfReader, WebSearchClient web, Chunker chunker) {
        this.pdfReader = pdfReader != null ? pdfReader : new PdfReader();
        this.web = web != null ? web : WebSearchClient.noop();
        this.chunker = chunker != null ? chunker : Chunker.defaults();
    }

    public String buildContext(String query, List<Path> localDocs, int topK) throws Exception {
        List<String> bits = new ArrayList<>();
        if (localDocs != null) {
            for (Path p : localDocs) {
                List<String> chunks = pdfReader.readChunks(p);
                for (String c : rank(query, chunks, topK)) bits.add(c);
            }
        }
        for (Map<String, String> hit : web.search(query, topK)) {
            bits.add(hit.getOrDefault("title", "") + ": " + hit.getOrDefault("snippet", ""));
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < bits.size(); i++) {
            sb.append("[").append(i + 1).append("] ").append(bits.get(i)).append("\n\n");
        }
        return sb.toString().trim();
    }

    public String augmentUserPrompt(String query, List<Path> localDocs, int topK) throws Exception {
        String ctx = buildContext(query, localDocs, topK);
        if (ctx.isBlank()) return query;
        return "Use the following context to answer.\n\nContext:\n" + ctx + "\n\nQuestion: " + query;
    }

    static List<String> rank(String query, List<String> chunks, int topK) {
        if (chunks == null || chunks.isEmpty()) return List.of();
        String q = query == null ? "" : query.toLowerCase(Locale.ROOT);
        String[] terms = q.split("\\s+");
        record Scored(String c, int s) {}
        List<Scored> scored = new ArrayList<>();
        for (String c : chunks) {
            String low = c.toLowerCase(Locale.ROOT);
            int s = 0;
            for (String t : terms) if (!t.isBlank() && low.contains(t)) s++;
            scored.add(new Scored(c, s));
        }
        scored.sort((a, b) -> Integer.compare(b.s, a.s));
        List<String> out = new ArrayList<>();
        for (int i = 0; i < Math.min(topK, scored.size()); i++) out.add(scored.get(i).c);
        return out;
    }
}
