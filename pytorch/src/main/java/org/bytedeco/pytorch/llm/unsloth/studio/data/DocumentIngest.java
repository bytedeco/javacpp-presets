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

package org.bytedeco.pytorch.llm.unsloth.studio.data;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/**
 * Ingest PDF / CSV / DOCX / plain text into training rows.
 * PDF/DOCX use lightweight pure-Java extractors (text only; no native deps).
 */
public final class DocumentIngest {

    public List<String> ingest(Path path) throws IOException {
        if (path == null || !Files.exists(path)) {
            throw new IOException("File not found: " + path);
        }
        String name = path.getFileName().toString().toLowerCase(Locale.ROOT);
        if (name.endsWith(".csv")) return readCsvLines(path);
        if (name.endsWith(".jsonl") || name.endsWith(".json")) return Files.readAllLines(path, StandardCharsets.UTF_8);
        if (name.endsWith(".txt") || name.endsWith(".md")) return List.of(Files.readString(path, StandardCharsets.UTF_8));
        if (name.endsWith(".docx")) return List.of(extractDocx(path));
        if (name.endsWith(".pdf")) return List.of(extractPdfRough(path));
        return List.of(Files.readString(path, StandardCharsets.UTF_8));
    }

    private List<String> readCsvLines(Path path) throws IOException {
        List<String> rows = new ArrayList<>();
        try (BufferedReader br = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            String header = br.readLine();
            String line;
            while ((line = br.readLine()) != null) {
                if (!line.isBlank()) rows.add(line);
            }
            if (header != null) {
                // keep header in meta row 0 optional — rows are data only
            }
        }
        return rows;
    }

    /** DOCX is a zip with word/document.xml — strip tags for plain text. */
    public String extractDocx(Path path) throws IOException {
        StringBuilder sb = new StringBuilder();
        try (ZipInputStream zis = new ZipInputStream(Files.newInputStream(path))) {
            ZipEntry e;
            while ((e = zis.getNextEntry()) != null) {
                if ("word/document.xml".equals(e.getName())) {
                    String xml = new String(zis.readAllBytes(), StandardCharsets.UTF_8);
                    String text = xml.replaceAll("<w:tab[^/]*/>", "\t")
                            .replaceAll("</w:p>", "\n")
                            .replaceAll("<[^>]+>", "")
                            .replaceAll("&amp;", "&")
                            .replaceAll("&lt;", "<")
                            .replaceAll("&gt;", ">")
                            .replaceAll("&quot;", "\"")
                            .replaceAll("&#10;", "\n");
                    sb.append(text);
                    break;
                }
            }
        }
        return sb.toString().trim();
    }

    /**
     * Very rough PDF text extraction: pull printable Latin sequences from binary.
     * Good enough for recipe demos; production hosts should inject a real PDF lib.
     */
    public String extractPdfRough(Path path) throws IOException {
        byte[] data = Files.readAllBytes(path);
        StringBuilder sb = new StringBuilder();
        StringBuilder cur = new StringBuilder();
        for (byte b : data) {
            int c = b & 0xff;
            if (c >= 32 && c < 127) {
                cur.append((char) c);
            } else {
                if (cur.length() >= 4) {
                    if (sb.length() > 0) sb.append(' ');
                    sb.append(cur);
                }
                cur.setLength(0);
            }
        }
        if (cur.length() >= 4) {
            if (sb.length() > 0) sb.append(' ');
            sb.append(cur);
        }
        return sb.toString().trim();
    }
}
