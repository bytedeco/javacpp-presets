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
package org.bytedeco.pytorch.llm.spacy.io;

import org.bytedeco.pytorch.llm.spacy.Doc;
import org.bytedeco.pytorch.llm.spacy.Language;
import org.bytedeco.pytorch.llm.spacy.tokenizer.SimpleTokenizer;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Simple Doc binary/line serialization.
 * <ul>
 *   <li>{@link #toBytes()} / {@link #fromBytes(byte[], Language)} — Java serialization of texts</li>
 *   <li>{@link #toLines(Path)} / {@link #fromLines(Path, Language)} — one document text per line</li>
 * </ul>
 */
public final class DocBin {

    private final List<Doc> docs = new ArrayList<>();

    public DocBin() {}

    public DocBin(Iterable<Doc> docs) {
        if (docs != null) {
            for (Doc d : docs) {
                add(d);
            }
        }
    }

    public void add(Doc d) {
        if (d != null) {
            docs.add(d);
        }
    }

    public void addDoc(Doc d) {
        add(d);
    }

    public int size() {
        return docs.size();
    }

    public List<Doc> getDocs() {
        return Collections.unmodifiableList(docs);
    }

    public Doc get(int i) {
        return docs.get(i);
    }

    public byte[] toBytes() throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try (ObjectOutputStream oos = new ObjectOutputStream(baos)) {
            oos.writeInt(docs.size());
            for (Doc d : docs) {
                oos.writeUTF(d.getText() == null ? "" : d.getText());
            }
        }
        return baos.toByteArray();
    }

    public static DocBin fromBytes(byte[] data, Language nlp) throws IOException, ClassNotFoundException {
        DocBin bin = new DocBin();
        try (ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(data))) {
            int n = ois.readInt();
            for (int i = 0; i < n; i++) {
                String text = ois.readUTF();
                if (nlp != null) {
                    bin.add(nlp.call(text));
                } else {
                    bin.add(new SimpleTokenizer().tokenize(text));
                }
            }
        }
        return bin;
    }

    public void toDisk(Path path) throws IOException {
        Files.write(path, toBytes());
    }

    public static DocBin fromDisk(Path path, Language nlp) throws IOException, ClassNotFoundException {
        return fromBytes(Files.readAllBytes(path), nlp);
    }

    /** Write one document text per line (newlines in text escaped). */
    public void toLines(Path path) {
        try (BufferedWriter w = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            for (Doc d : docs) {
                String t = d.getText() == null ? "" : d.getText();
                w.write(t.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r"));
                w.newLine();
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    public static DocBin fromLines(Path path, Language nlp) {
        DocBin bin = new DocBin();
        SimpleTokenizer fallback = nlp == null ? new SimpleTokenizer() : null;
        try (BufferedReader br = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            String line;
            while ((line = br.readLine()) != null) {
                String text = line.replace("\\n", "\n").replace("\\r", "\r").replace("\\\\", "\\");
                if (nlp != null) {
                    bin.add(nlp.call(text));
                } else {
                    bin.add(fallback.tokenize(text));
                }
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        return bin;
    }

    @Override
    public String toString() {
        return "DocBin(size=" + size() + ")";
    }
}
