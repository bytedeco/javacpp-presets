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

import java.nio.file.Path;
import java.util.Map;

/**
 * Static factory for opening FFmpeg-backed media.
 *
 * <p>Two layers:
 * <ul>
 *   <li><b>Convenience tensor IO</b> — {@link #openVideo}/{@link #openAudio} ({@link VideoFile}/{@link AudioFile})</li>
 *   <li><b>PyAV-parity</b> — {@link #open}/{@link Av#open} → {@link Container}</li>
 * </ul>
 *
 * <pre>{@code
 * // Simple tensor decode (torchaudio/torchvision style)
 * try (VideoFile vf = FFmpegLoader.openVideo("video.mp4")) {
 *     Tensor frames = vf.read();  // [N, 3, H, W]
 * }
 *
 * // PyAV-style container
 * try (Container c = FFmpegLoader.open("video.mp4")) {
 *     for (Frame f : c.decodeVideo(0)) {
 *         var rgb = ((VideoFrame) f).toNdarray("rgb24");
 *     }
 * }
 * }</pre>
 */
public final class FFmpegLoader {

    private FFmpegLoader() {}

    // ── PyAV-parity ───────────────────────────────────────────────────────

    /** {@link Av#open(String)} — read container. */
    public static Container open(String path) {
        return Av.open(path);
    }

    public static Container open(Path path) {
        return Av.open(path);
    }

    /** {@link Av#open(String, String)} — {@code mode} is {@code "r"} or {@code "w"}. */
    public static Container open(String path, String mode) {
        return Av.open(path, mode);
    }

    public static Container open(String path, String mode, Map<String, String> options) {
        return Av.open(path, mode, options);
    }

    // ── Convenience tensor readers (existing) ─────────────────────────────

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

    // ── Enterprise VideoOps facades (Daft / torchcodec / VLM style) ───────

    /** {@link VideoOps#probe(String)} */
    public static VideoFile.VideoMeta probeVideo(String path) {
        return VideoOps.probe(path);
    }

    /** Uniform sample {@code count} frames — LLaVA / Qwen-VL style. */
    public static java.util.List<org.bytedeco.pytorch.Tensor> extractUniform(String path, int count) {
        return VideoOps.extractUniform(path, count);
    }

    /** Uniform sample stacked to {@code [N,3,H,W]}. */
    public static org.bytedeco.pytorch.Tensor extractUniformStacked(String path, int count) {
        return VideoOps.extractUniformStacked(path, count);
    }

    /** Frame nearest to {@code seconds}. */
    public static org.bytedeco.pytorch.Tensor frameAt(String path, double seconds) {
        return VideoOps.frameAt(path, seconds);
    }

    /** First decodable frame (thumbnail / poster). */
    public static org.bytedeco.pytorch.Tensor thumbnail(String path) {
        return VideoOps.thumbnail(path);
    }

    /** Sample at target fps (capped). */
    public static java.util.List<org.bytedeco.pytorch.Tensor> extractAtFps(
            String path, double targetFps, int maxFrames) {
        return VideoOps.extractAtFps(path, targetFps, maxFrames);
    }

    /** Capability map for native libav + CLI ffmpeg. */
    public static java.util.Map<String, Object> videoCapabilities() {
        return VideoOps.capabilities();
    }
}
