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
package org.bytedeco.pytorch.audio.datasets;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.audio.io.AudioIO;
import org.bytedeco.pytorch.audio.transforms.AudioTransform;

import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.Set;

/**
 * Audio folder dataset: {@code root/class_x/xxx.wav}.
 * Mirrors torchvision ImageFolder layout for speech/audio classification.
 */
public class AudioFolder extends AudioDataset {
    private static final Set<String> DEFAULT_EXTS = Set.of(
            "wav", "wave", "mp3", "flac", "m4a", "aac", "wma", "ogg");

    private final List<String> classes;
    private final List<Path> samples;
    private final List<Integer> targets;
    private final Set<String> extensions;
    private final int sampleRate;
    private final boolean mono;

    public AudioFolder(String root) throws IOException {
        this(Path.of(root), null, DEFAULT_EXTS, 16000, true);
    }

    public AudioFolder(Path root) throws IOException {
        this(root, null, DEFAULT_EXTS, 16000, true);
    }

    public AudioFolder(Path root, AudioTransform<?, ?> audioTransform) throws IOException {
        this(root, audioTransform, DEFAULT_EXTS, 16000, true);
    }

    public AudioFolder(Path root, AudioTransform<?, ?> audioTransform, int sampleRate, boolean mono) throws IOException {
        this(root, audioTransform, DEFAULT_EXTS, sampleRate, mono);
    }

    public AudioFolder(Path root, AudioTransform<?, ?> audioTransform, Set<String> extensions,
                       int sampleRate, boolean mono) throws IOException {
        super(Objects.requireNonNull(root, "root"));
        this.extensions = extensions == null ? DEFAULT_EXTS : extensions;
        this.sampleRate = sampleRate > 0 ? sampleRate : 16000;
        this.mono = mono;
        setTransform(audioTransform);

        List<String> classNames = new ArrayList<>();
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(root)) {
            for (Path p : ds) {
                if (Files.isDirectory(p) && !p.getFileName().toString().startsWith(".")) {
                    classNames.add(p.getFileName().toString());
                }
            }
        }
        Collections.sort(classNames);
        this.classes = Collections.unmodifiableList(classNames);

        List<Path> files = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();
        for (int ci = 0; ci < classes.size(); ci++) {
            Path classDir = root.resolve(classes.get(ci));
            try (DirectoryStream<Path> ds = Files.newDirectoryStream(classDir)) {
                List<Path> classFiles = new ArrayList<>();
                for (Path p : ds) {
                    if (Files.isRegularFile(p) && isAudio(p)) {
                        classFiles.add(p);
                    }
                }
                Collections.sort(classFiles);
                for (Path f : classFiles) {
                    files.add(f);
                    labels.add(ci);
                }
            }
        }
        this.samples = Collections.unmodifiableList(files);
        this.targets = Collections.unmodifiableList(labels);
    }

    private boolean isAudio(Path p) {
        String name = p.getFileName().toString();
        int dot = name.lastIndexOf('.');
        if (dot < 0) return false;
        String ext = name.substring(dot + 1).toLowerCase(Locale.ROOT);
        return extensions.contains(ext);
    }

    public List<String> classes() {
        return classes;
    }

    public List<Path> samples() {
        return samples;
    }

    public List<Integer> targets() {
        return targets;
    }

    public int sampleRate() {
        return sampleRate;
    }

    public int class_to_idx(String className) {
        int i = classes.indexOf(className);
        if (i < 0) throw new IllegalArgumentException("unknown class: " + className);
        return i;
    }

    public int classToIdx(String className) {
        return class_to_idx(className);
    }

    @Override
    public int size() {
        return samples.size();
    }

    @Override
    public Sample get(int index) {
        Path path = samples.get(index);
        int label = targets.get(index);
        try {
            AudioIO.AudioLoadResult loaded = AudioIO.load(path.toString(), sampleRate, mono);
            Tensor waveform = loaded.waveform();
            Object data = applyTransform(waveform);
            Object target = applyTargetTransform(label);
            return new Sample(data, target);
        } catch (Exception e) {
            throw new IllegalStateException("failed to load " + path, e);
        }
    }

    /** Generic folder dataset alias. */
    public static final class DatasetFolder extends AudioFolder {
        public DatasetFolder(String root) throws IOException {
            super(root);
        }

        public DatasetFolder(Path root, AudioTransform<?, ?> audioTransform, String... exts) throws IOException {
            super(root, audioTransform, exts == null || exts.length == 0
                    ? DEFAULT_EXTS
                    : new HashSet<>(Arrays.asList(exts)), 16000, true);
        }
    }
}
