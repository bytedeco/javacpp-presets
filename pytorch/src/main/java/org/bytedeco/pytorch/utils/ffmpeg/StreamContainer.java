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
package org.bytedeco.pytorch.utils.ffmpeg;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

/**
 * Stream collection — PyAV {@code container.streams}.
 *
 * <pre>{@code
 * VideoStream v = container.streams().video(0);
 * AudioStream a = container.streams().audio(0);
 * for (Stream s : container.streams()) { ... }
 * }</pre>
 */
public final class StreamContainer implements Iterable<Stream> {

    private final List<Stream> all;
    private final List<VideoStream> videos;
    private final List<AudioStream> audios;

    StreamContainer(List<Stream> all) {
        this.all = Collections.unmodifiableList(new ArrayList<>(all));
        List<VideoStream> v = new ArrayList<>();
        List<AudioStream> a = new ArrayList<>();
        for (Stream s : all) {
            if (s instanceof VideoStream) v.add((VideoStream) s);
            else if (s instanceof AudioStream) a.add((AudioStream) s);
        }
        this.videos = Collections.unmodifiableList(v);
        this.audios = Collections.unmodifiableList(a);
    }

    public int size() { return all.size(); }

    public Stream get(int index) {
        return all.get(index);
    }

    public List<Stream> all() { return all; }

    public List<VideoStream> video() { return videos; }

    public VideoStream video(int i) {
        if (i < 0 || i >= videos.size()) {
            throw new FFmpegException("no video stream at index " + i + " (have " + videos.size() + ")");
        }
        return videos.get(i);
    }

    public List<AudioStream> audio() { return audios; }

    public AudioStream audio(int i) {
        if (i < 0 || i >= audios.size()) {
            throw new FFmpegException("no audio stream at index " + i + " (have " + audios.size() + ")");
        }
        return audios.get(i);
    }

    @Override
    public Iterator<Stream> iterator() {
        return all.iterator();
    }

    @Override
    public String toString() {
        return "StreamContainer(n=" + all.size() + ", video=" + videos.size() + ", audio=" + audios.size() + ")";
    }
}
