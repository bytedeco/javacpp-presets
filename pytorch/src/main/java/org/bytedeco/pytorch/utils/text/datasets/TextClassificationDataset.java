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
package org.bytedeco.pytorch.utils.text.datasets;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

/**
 * Text classification dataset loaded from CSV or a folder of label/text files.
 * <ul>
 *   <li>CSV: {@code label,text} or {@code text,label} (configurable)</li>
 *   <li>Folder: each subdirectory name is the label; files inside are documents</li>
 * </ul>
 */
public final class TextClassificationDataset {

    public static final class Sample {
        public final String text;
        public final int label;
        public final String labelName;

        public Sample(String text, int label, String labelName) {
            this.text = text;
            this.label = label;
            this.labelName = labelName;
        }

        @Override
        public String toString() {
            return "Sample{label=" + label + "(" + labelName + "), text='"
                    + (text == null ? "" : text.substring(0, Math.min(40, text.length()))) + "'}";
        }
    }

    private final List<Sample> samples;
    private final Map<String, Integer> labelToId;
    private final List<String> idToLabel;

    public TextClassificationDataset(List<Sample> samples, Map<String, Integer> labelToId) {
        this.samples = new ArrayList<>(samples == null ? List.of() : samples);
        this.labelToId = new LinkedHashMap<>(labelToId == null ? Map.of() : labelToId);
        this.idToLabel = new ArrayList<>();
        for (Map.Entry<String, Integer> e : this.labelToId.entrySet()) {
            while (idToLabel.size() <= e.getValue()) {
                idToLabel.add(null);
            }
            idToLabel.set(e.getValue(), e.getKey());
        }
    }

    public int size() {
        return samples.size();
    }

    public Sample get(int index) {
        return samples.get(index);
    }

    public List<Sample> samples() {
        return Collections.unmodifiableList(samples);
    }

    public int numClasses() {
        return labelToId.size();
    }

    public Map<String, Integer> labelToId() {
        return Collections.unmodifiableMap(labelToId);
    }

    public List<String> labels() {
        return Collections.unmodifiableList(idToLabel);
    }

    public List<String> texts() {
        List<String> t = new ArrayList<>(samples.size());
        for (Sample s : samples) {
            t.add(s.text);
        }
        return t;
    }

    public int[] labelIds() {
        int[] y = new int[samples.size()];
        for (int i = 0; i < samples.size(); i++) {
            y[i] = samples.get(i).label;
        }
        return y;
    }

    /**
     * Load CSV. Default: first column label, second column text.
     * Set {@code labelFirst=false} for text,label order.
     */
    public static TextClassificationDataset fromCsv(Path csvPath) {
        return fromCsv(csvPath, true, true, ',');
    }

    public static TextClassificationDataset fromCsv(Path csvPath, boolean labelFirst, boolean hasHeader, char delimiter) {
        List<Sample> samples = new ArrayList<>();
        Map<String, Integer> labelToId = new LinkedHashMap<>();
        try (BufferedReader br = Files.newBufferedReader(csvPath, StandardCharsets.UTF_8)) {
            String line;
            boolean first = true;
            while ((line = br.readLine()) != null) {
                if (line.isBlank()) {
                    continue;
                }
                if (first && hasHeader) {
                    first = false;
                    continue;
                }
                first = false;
                List<String> cols = splitCsv(line, delimiter);
                if (cols.size() < 2) {
                    continue;
                }
                String labelName;
                String text;
                if (labelFirst) {
                    labelName = cols.get(0).trim();
                    text = cols.get(1).trim();
                    if (cols.size() > 2) {
                        // join remaining as text
                        StringBuilder sb = new StringBuilder(text);
                        for (int i = 2; i < cols.size(); i++) {
                            sb.append(delimiter).append(cols.get(i));
                        }
                        text = sb.toString().trim();
                    }
                } else {
                    text = cols.get(0).trim();
                    labelName = cols.get(cols.size() - 1).trim();
                    if (cols.size() > 2) {
                        StringBuilder sb = new StringBuilder(cols.get(0));
                        for (int i = 1; i < cols.size() - 1; i++) {
                            sb.append(delimiter).append(cols.get(i));
                        }
                        text = sb.toString().trim();
                    }
                }
                labelName = unquote(labelName);
                text = unquote(text);
                int id = labelToId.computeIfAbsent(labelName, k -> labelToId.size());
                samples.add(new Sample(text, id, labelName));
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        return new TextClassificationDataset(samples, labelToId);
    }

    /**
     * Load from a directory tree: {@code root/label_name/*.txt}.
     */
    public static TextClassificationDataset fromFolder(Path root) {
        List<Sample> samples = new ArrayList<>();
        Map<String, Integer> labelToId = new LinkedHashMap<>();
        if (root == null || !Files.isDirectory(root)) {
            throw new IllegalArgumentException("Not a directory: " + root);
        }
        try (Stream<Path> dirs = Files.list(root)) {
            List<Path> labelDirs = dirs.filter(Files::isDirectory).sorted().toList();
            for (Path labelDir : labelDirs) {
                String labelName = labelDir.getFileName().toString();
                int id = labelToId.computeIfAbsent(labelName, k -> labelToId.size());
                try (Stream<Path> files = Files.walk(labelDir)) {
                    files.filter(Files::isRegularFile)
                            .filter(p -> {
                                String n = p.getFileName().toString().toLowerCase();
                                return n.endsWith(".txt") || n.endsWith(".text") || n.endsWith(".csv") || !n.contains(".");
                            })
                            .sorted()
                            .forEach(p -> {
                                try {
                                    String text = Files.readString(p, StandardCharsets.UTF_8).trim();
                                    if (!text.isEmpty()) {
                                        samples.add(new Sample(text, id, labelName));
                                    }
                                } catch (IOException e) {
                                    throw new UncheckedIOException(e);
                                }
                            });
                }
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        return new TextClassificationDataset(samples, labelToId);
    }

    private static List<String> splitCsv(String line, char delimiter) {
        List<String> cols = new ArrayList<>();
        StringBuilder cur = new StringBuilder();
        boolean inQuotes = false;
        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            if (c == '"') {
                if (inQuotes && i + 1 < line.length() && line.charAt(i + 1) == '"') {
                    cur.append('"');
                    i++;
                } else {
                    inQuotes = !inQuotes;
                }
            } else if (c == delimiter && !inQuotes) {
                cols.add(cur.toString());
                cur.setLength(0);
            } else {
                cur.append(c);
            }
        }
        cols.add(cur.toString());
        return cols;
    }

    private static String unquote(String s) {
        if (s == null) {
            return "";
        }
        s = s.trim();
        if (s.length() >= 2 && s.startsWith("\"") && s.endsWith("\"")) {
            return s.substring(1, s.length() - 1).replace("\"\"", "\"");
        }
        return s;
    }

    @Override
    public String toString() {
        return "TextClassificationDataset(size=" + size() + ", classes=" + numClasses() + ")";
    }
}
