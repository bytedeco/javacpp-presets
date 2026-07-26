/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or (at your option)
 * any later version (collectively, the "License");
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

import java.nio.file.Path;

/**
 * Static factory for opening FFmpeg-backed media files.
 *
 * <p>This class ensures native FFmpeg libraries are loaded before use.
 *
 * <pre>{@code
 * // Video
 * try (VideoFile vf = FFmpegLoader.openVideo("video.mp4")) {
 *     Tensor frames = vf.read();  // [N, 3, H, W]
 * }
 *
 * // Audio
 * try (AudioFile af = FFmpegLoader.openAudio("audio.flac")) {
 *     Tensor wave = af.read();    // [C, T]
 * }
 * }</pre>
 */
public final class FFmpegLoader {

    private FFmpegLoader() {}

    /**
     * Open a video file for reading frames as tensors.
     *
     * @param path path to video (mp4, avi, mkv, webm, …)
     * @return opened {@link VideoFile}
     */
    public static VideoFile openVideo(String path) {
        return VideoFile.open(path);
    }

    /** @see #openVideo(String) */
    public static VideoFile openVideo(Path path) {
        return VideoFile.open(path);
    }

    /**
     * Open an audio file for reading waveform as a tensor.
     *
     * @param path path to audio (mp3, wav, flac, aac, ogg, …)
     * @return opened {@link AudioFile}
     */
    public static AudioFile openAudio(String path) {
        return AudioFile.open(path);
    }

    /** @see #openAudio(String) */
    public static AudioFile openAudio(Path path) {
        return AudioFile.open(path);
    }

    /**
     * Decode all video frames from a file directly (convenience).
     *
     * @param filePath path to video
     * @return list of RGB tensors {@code [3, H, W]}, one per frame
     */
    public static java.util.List<org.bytedeco.pytorch.Tensor> decodeVideo(String filePath) {
        return VideoTensors.decodeAllFrames(filePath);
    }

    /**
     * Decode all audio samples from a file directly (convenience).
     *
     * @param filePath path to audio
     * @return waveform tensor {@code [channels, time]}, dtype float32
     */
    public static org.bytedeco.pytorch.Tensor decodeAudio(String filePath) {
        return AudioTensorsFFmpeg.decodeAllSamples(filePath);
    }
}
