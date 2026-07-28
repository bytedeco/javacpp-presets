/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet, mullerhai
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
package org.bytedeco.pytorch.dataframe.media;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.dataframe.dtype.AudioData;
import org.bytedeco.pytorch.dataframe.dtype.EmbeddingData;
import org.bytedeco.pytorch.dataframe.dtype.ImageData;
import org.bytedeco.pytorch.dataframe.dtype.TensorData;
import org.bytedeco.pytorch.dataframe.dtype.VideoData;
import org.bytedeco.pytorch.dataframe.tensor.TensorBridge;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.utils.audio.utils.AudioTensors;
import org.bytedeco.pytorch.utils.vision.utils.ImageTensors;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Central multimodal bridge among DataFrame media dtypes, PyTorch {@link Tensor},
 * OpenCV (via {@code OpenCVIO}/{@code MatToTensor}), FFmpeg (via {@code FFmpegLoader}),
 * torchvision ({@code ImageTensors}/{@code ImageIO}), and torchaudio ({@code AudioTensors}).
 *
 * <p>Design goals:
 * <ul>
 *   <li>One place for Image/Audio/Video ↔ Tensor ↔ ImageData/AudioData/VideoData conversion</li>
 *   <li>Prefer native OpenCV/FFmpeg when available; fall back to pure-Java (ImageIO / WAV) otherwise</li>
 *   <li>Batch helpers used by {@code DataFrame.readImages* / readVideo* / readAudio*}</li>
 *   <li>Frame extraction, resampling, and embedding-ready tensors for multimodal pipelines</li>
 * </ul>
 *
 * <pre>{@code
 * // Image: path → ImageData (OpenCV if available)
 * ImageData img = MediaBridge.loadImage("photo.jpg");
 * Tensor t = MediaBridge.imageToTensor(img);          // [3,H,W] float [0,1]
 * Tensor t255 = MediaBridge.imageToTensor255(img);    // [3,H,W] float [0,255] (OpenCV style)
 *
 * // Video: path → VideoData with real frames via FFmpeg
 * VideoData vid = MediaBridge.loadVideo("clip.mp4", MediaBridge.VideoOptions.defaults());
 * List&lt;ImageData&gt; frames = MediaBridge.extractFrames(vid, 2.0); // 2 fps
 *
 * // Audio: path → AudioData via FFmpeg (mp3/flac/…) or pure Java WAV
 * AudioData aud = MediaBridge.loadAudio("speech.mp3", 16000, true);
 * Tensor wave = MediaBridge.audioToTensor(aud);       // [C,T]
 * }</pre>
 */
public final class MediaBridge {

    private MediaBridge() {}

    // ── availability probes (lazy, once) ──────────────────────────────────

    private static final AtomicReference<Boolean> OPENCV_OK = new AtomicReference<>();
    private static final AtomicReference<Boolean> FFMPEG_OK = new AtomicReference<>();
    private static final AtomicBoolean OPENCV_WARNED = new AtomicBoolean(false);
    private static final AtomicBoolean FFMPEG_WARNED = new AtomicBoolean(false);

    /** Whether javacpp-opencv natives + our {@code OpenCVIO} glue appear loadable. */
    public static boolean isOpenCvAvailable() {
        Boolean cached = OPENCV_OK.get();
        if (cached != null) return cached;
        synchronized (OPENCV_OK) {
            if (OPENCV_OK.get() != null) return OPENCV_OK.get();
            boolean ok = false;
            try {
                // Our pure-Java glue must be present
                Class.forName("org.bytedeco.pytorch.utils.opencv.OpenCVIO");
                Class.forName("org.bytedeco.opencv.global.opencv_imgcodecs");
                // touch a cheap global so natives actually resolve
                Class<?> g = Class.forName("org.bytedeco.opencv.global.opencv_core");
                g.getField("CV_8U").get(null);
                ok = true;
            } catch (Throwable t) {
                ok = false;
            }
            OPENCV_OK.set(ok);
            return ok;
        }
    }

    /** Whether javacpp-ffmpeg natives + our FFmpegLoader glue appear loadable. */
    public static boolean isFFmpegAvailable() {
        Boolean cached = FFMPEG_OK.get();
        if (cached != null) return cached;
        synchronized (FFMPEG_OK) {
            if (FFMPEG_OK.get() != null) return FFMPEG_OK.get();
            boolean ok = false;
            try {
                Class.forName("org.bytedeco.pytorch.utils.ffmpeg.FFmpegLoader");
                Class.forName("org.bytedeco.ffmpeg.global.avformat");
                ok = true;
            } catch (Throwable t) {
                ok = false;
            }
            FFMPEG_OK.set(ok);
            return ok;
        }
    }

    private static void warnOpenCvOnce(String ctx, Throwable t) {
        if (OPENCV_WARNED.compareAndSet(false, true)) {
            System.err.println("[MediaBridge] OpenCV unavailable for " + ctx
                    + " — falling back to pure-Java ImageIO"
                    + (t == null ? "" : (": " + t.getClass().getSimpleName() + ": " + t.getMessage())));
        }
    }

    private static void warnFFmpegOnce(String ctx, Throwable t) {
        if (FFMPEG_WARNED.compareAndSet(false, true)) {
            System.err.println("[MediaBridge] FFmpeg unavailable for " + ctx
                    + " — falling back to pure-Java / stub"
                    + (t == null ? "" : (": " + t.getClass().getSimpleName() + ": " + t.getMessage())));
        }
    }

    // ── Backend preference ────────────────────────────────────────────────

    public enum ImageBackend {
        /** Prefer OpenCV; fall back to ImageIO. */
        AUTO,
        /** Force OpenCV (throws if unavailable). */
        OPENCV,
        /** Force JDK ImageIO / torchvision ImageTensors path. */
        IMAGEIO
    }

    public enum VideoBackend {
        AUTO, FFMPEG, STUB
    }

    public enum AudioBackend {
        AUTO, FFMPEG, JAVA
    }

    /** Options for video decode into {@link VideoData}. */
    public static final class VideoOptions {
        public final VideoBackend backend;
        /** Max frames to decode (≤0 → all). */
        public final int maxFrames;
        /** Decode every N-th frame (1 = all). */
        public final int frameStride;
        /** If &gt;0, subsample to approximately this fps while decoding. */
        public final double targetFps;
        /** Cap frame size (0 = keep original). Applied after decode via BufferedImage scale. */
        public final int maxWidth;
        public final int maxHeight;
        /** Also attempt to attach audio track from the same file. */
        public final boolean withAudio;

        public VideoOptions(VideoBackend backend, int maxFrames, int frameStride,
                            double targetFps, int maxWidth, int maxHeight, boolean withAudio) {
            this.backend = backend == null ? VideoBackend.AUTO : backend;
            this.maxFrames = maxFrames;
            this.frameStride = Math.max(1, frameStride);
            this.targetFps = targetFps;
            this.maxWidth = Math.max(0, maxWidth);
            this.maxHeight = Math.max(0, maxHeight);
            this.withAudio = withAudio;
        }

        public static VideoOptions defaults() {
            return new VideoOptions(VideoBackend.AUTO, 0, 1, 0, 0, 0, false);
        }

        public static VideoOptions keyframes(int maxFrames) {
            return new VideoOptions(VideoBackend.AUTO, maxFrames, 1, 1.0, 0, 0, false);
        }

        public VideoOptions withMaxFrames(int n) {
            return new VideoOptions(backend, n, frameStride, targetFps, maxWidth, maxHeight, withAudio);
        }

        public VideoOptions withTargetFps(double fps) {
            return new VideoOptions(backend, maxFrames, frameStride, fps, maxWidth, maxHeight, withAudio);
        }

        public VideoOptions withAudio(boolean on) {
            return new VideoOptions(backend, maxFrames, frameStride, targetFps, maxWidth, maxHeight, on);
        }

        public VideoOptions withMaxSize(int w, int h) {
            return new VideoOptions(backend, maxFrames, frameStride, targetFps, w, h, withAudio);
        }
    }

    /** Options for image load. */
    public static final class ImageOptions {
        public final ImageBackend backend;
        public final boolean asGray;
        public final int resizeW;
        public final int resizeH;

        public ImageOptions(ImageBackend backend, boolean asGray, int resizeW, int resizeH) {
            this.backend = backend == null ? ImageBackend.AUTO : backend;
            this.asGray = asGray;
            this.resizeW = Math.max(0, resizeW);
            this.resizeH = Math.max(0, resizeH);
        }

        public static ImageOptions defaults() {
            return new ImageOptions(ImageBackend.AUTO, false, 0, 0);
        }

        public ImageOptions withBackend(ImageBackend b) {
            return new ImageOptions(b, asGray, resizeW, resizeH);
        }

        public ImageOptions gray() {
            return new ImageOptions(backend, true, resizeW, resizeH);
        }

        public ImageOptions resize(int w, int h) {
            return new ImageOptions(backend, asGray, w, h);
        }
    }

    /** Options for audio load. */
    public static final class AudioOptions {
        public final AudioBackend backend;
        public final int sampleRate;
        public final boolean mono;

        public AudioOptions(AudioBackend backend, int sampleRate, boolean mono) {
            this.backend = backend == null ? AudioBackend.AUTO : backend;
            this.sampleRate = sampleRate > 0 ? sampleRate : 16000;
            this.mono = mono;
        }

        public static AudioOptions defaults() {
            return new AudioOptions(AudioBackend.AUTO, 16000, true);
        }

        public static AudioOptions of(int sr, boolean mono) {
            return new AudioOptions(AudioBackend.AUTO, sr, mono);
        }
    }

    // ── Image load / convert ──────────────────────────────────────────────

    public static ImageData loadImage(String path) throws IOException {
        return loadImage(path, ImageOptions.defaults());
    }

    public static ImageData loadImage(Path path) throws IOException {
        return loadImage(path.toString(), ImageOptions.defaults());
    }

    public static ImageData loadImage(String path, ImageOptions opts) throws IOException {
        Objects.requireNonNull(path, "path");
        if (opts == null) opts = ImageOptions.defaults();

        ImageBackend b = opts.backend;
        if (b == ImageBackend.AUTO) {
            b = isOpenCvAvailable() ? ImageBackend.OPENCV : ImageBackend.IMAGEIO;
        }

        ImageData img;
        if (b == ImageBackend.OPENCV) {
            try {
                img = loadImageOpenCv(path, opts.asGray);
            } catch (Throwable t) {
                if (opts.backend == ImageBackend.OPENCV) {
                    throw new IOException("OpenCV load failed: " + path, t);
                }
                warnOpenCvOnce("loadImage", t);
                img = ImageData.load(path);
            }
        } else {
            img = ImageData.load(path);
            if (opts.asGray && img.getImage() != null) {
                try {
                    img = img.toGrayscale();
                } catch (Exception ignored) {}
            }
        }

        if (opts.resizeW > 0 && opts.resizeH > 0 && img != null && img.getImage() != null) {
            img = img.resize(opts.resizeW, opts.resizeH);
        }
        return img;
    }

    /**
     * Load via OpenCV → Tensor [C,H,W] in [0,255] → BufferedImage/ImageData.
     * Values are scaled to [0,1] ImageTensors convention when building ImageData.
     */
    public static ImageData loadImageOpenCv(String path, boolean gray) throws Exception {
        Class<?> io = Class.forName("org.bytedeco.pytorch.utils.opencv.OpenCVIO");
        Tensor t;
        if (gray) {
            t = (Tensor) io.getMethod("readImageGray", String.class).invoke(null, path);
        } else {
            t = (Tensor) io.getMethod("readImage", String.class).invoke(null, path);
        }
        // OpenCV path returns [0,255]; ImageTensors expects roughly [0,1]
        Tensor scaled = scale255ToUnit(t);
        ImageData id = ImageTensors.toImageData(scaled);
        id.setPath(path);
        String ext = extensionOf(path);
        if (!ext.isEmpty()) id.setFormat(ext);
        return id;
    }

    /** torchvision-style: CHW float in [0,1]. */
    public static Tensor imageToTensor(ImageData image) {
        Objects.requireNonNull(image, "image");
        if (image.getImage() == null && image.getPath() != null) {
            try {
                ImageData loaded = loadImage(image.getPath());
                return ImageTensors.toTensor(loaded);
            } catch (Exception e) {
                throw new IllegalStateException("cannot materialize ImageData path=" + image.getPath(), e);
            }
        }
        return ImageTensors.toTensor(image);
    }

    /** OpenCV-style: CHW float in [0,255]. */
    public static Tensor imageToTensor255(ImageData image) {
        Tensor unit = imageToTensor(image);
        return unit.mul(new org.bytedeco.pytorch.Scalar(255.0));
    }

    public static ImageData tensorToImage(Tensor t) {
        Objects.requireNonNull(t, "tensor");
        // Accept [0,1] or [0,255]
        Tensor cpu = t.contiguous().cpu().to(ScalarType.Float);
        float max = maxAbs(cpu);
        if (max > 1.5f) {
            cpu = scale255ToUnit(cpu);
        }
        return ImageTensors.toImageData(cpu);
    }

    public static TensorData imageToTensorData(ImageData image) {
        Tensor t = imageToTensor(image);
        return TensorBridge.toTensorData(t);
    }

    public static ImageData tensorDataToImage(TensorData td) {
        Objects.requireNonNull(td, "tensorData");
        Tensor t = TensorBridge.toTensor(td);
        return tensorToImage(t);
    }

    /**
     * OpenCV Mat → ImageData (via MatToTensor → unit scale).
     * Accepts a live {@code org.bytedeco.opencv.opencv_core.Mat} by reflection so this
     * class still compiles when OpenCV is on an optional classpath profile.
     */
    public static ImageData matToImage(Object mat) throws Exception {
        Objects.requireNonNull(mat, "mat");
        Class<?> mtt = Class.forName("org.bytedeco.pytorch.utils.opencv.MatToTensor");
        Tensor t = (Tensor) mtt.getMethod("fromMat",
                Class.forName("org.bytedeco.opencv.opencv_core.Mat")).invoke(null, mat);
        return tensorToImage(scale255ToUnit(t));
    }

    /** ImageData → OpenCV Mat (BGR). */
    public static Object imageToMat(ImageData image) throws Exception {
        Tensor t255 = imageToTensor255(image);
        Class<?> mtt = Class.forName("org.bytedeco.pytorch.utils.opencv.MatToTensor");
        return mtt.getMethod("toMat", Tensor.class).invoke(null, t255);
    }

    // ── Audio load / convert ──────────────────────────────────────────────

    public static AudioData loadAudio(String path) throws IOException {
        return loadAudio(path, AudioOptions.defaults());
    }

    public static AudioData loadAudio(String path, int sampleRate, boolean mono) throws IOException {
        return loadAudio(path, new AudioOptions(AudioBackend.AUTO, sampleRate, mono));
    }

    public static AudioData loadAudio(String path, AudioOptions opts) throws IOException {
        Objects.requireNonNull(path, "path");
        if (opts == null) opts = AudioOptions.defaults();

        String ext = extensionOf(path).toLowerCase(Locale.ROOT);
        AudioBackend b = opts.backend;
        if (b == AudioBackend.AUTO) {
            // Prefer pure Java for WAV; FFmpeg for compressed formats when available
            if ("wav".equals(ext) || "wave".equals(ext)) {
                b = AudioBackend.JAVA;
            } else {
                b = isFFmpegAvailable() ? AudioBackend.FFMPEG : AudioBackend.JAVA;
            }
        }

        if (b == AudioBackend.FFMPEG) {
            try {
                return loadAudioFFmpeg(path, opts.sampleRate, opts.mono);
            } catch (Throwable t) {
                if (opts.backend == AudioBackend.FFMPEG) {
                    throw new IOException("FFmpeg audio load failed: " + path, t);
                }
                warnFFmpegOnce("loadAudio", t);
            }
        }
        // JAVA / fallback — call format helpers directly (NOT AudioData.loadFromFile),
        // because that method may re-enter MediaBridge for non-WAV and recurse.
        return loadAudioJava(path, opts.sampleRate, opts.mono);
    }

    /** Pure-Java / stub audio path used when FFmpeg is unavailable. */
    public static AudioData loadAudioJava(String path, int sampleRate, boolean mono) throws IOException {
        String ext = extensionOf(path).toLowerCase(Locale.ROOT);
        AudioData aud;
        if ("wav".equals(ext) || "wave".equals(ext)) {
            aud = AudioData.loadWav(path, sampleRate, mono);
        } else if ("mp3".equals(ext)) {
            aud = AudioData.loadMp3(path, sampleRate, mono);
        } else if ("flac".equals(ext)) {
            aud = AudioData.loadFlac(path, sampleRate, mono);
        } else if ("m4a".equals(ext)) {
            aud = AudioData.loadM4a(path, sampleRate, mono);
        } else if ("wma".equals(ext)) {
            aud = AudioData.loadWma(path, sampleRate, mono);
        } else if ("aac".equals(ext)) {
            aud = AudioData.loadAac(path, sampleRate, mono);
        } else if ("ogg".equals(ext)) {
            // no dedicated helper — lightweight tone stub sized by file length
            long bytes = 0;
            try { bytes = Files.size(Path.of(path)); } catch (Exception ignored) {}
            int n = Math.max(sampleRate / 10, (int) Math.min(sampleRate * 2L, Math.max(1, bytes)));
            float[] samples = new float[n];
            for (int i = 0; i < n; i++) samples[i] = (float) Math.sin(2 * Math.PI * 440 * i / sampleRate) * 0.1f;
            aud = new AudioData(samples, sampleRate, mono ? 1 : 2);
            aud.setDuration((double) n / sampleRate / (mono ? 1 : 2));
        } else {
            throw new IOException("unsupported audio extension: " + ext + " (" + path + ")");
        }
        aud.setPath(path);
        aud.setFormat(ext);
        return aud;
    }

    public static AudioData loadAudioFFmpeg(String path, int targetSr, boolean mono) throws Exception {
        Class<?> loader = Class.forName("org.bytedeco.pytorch.utils.ffmpeg.FFmpegLoader");
        try (AutoCloseable af = (AutoCloseable) loader.getMethod("openAudio", String.class).invoke(null, path)) {
            Class<?> afClass = af.getClass();
            int nativeSr = ((Number) afClass.getMethod("sampleRate").invoke(af)).intValue();
            int nativeCh = ((Number) afClass.getMethod("channels").invoke(af)).intValue();
            Tensor wave = (Tensor) afClass.getMethod("read").invoke(af); // [C,T]
            AudioData aud = AudioTensors.toAudioData(wave, nativeSr);
            aud.setPath(path);
            if (mono && aud.getChannels() > 1) {
                aud = toMono(aud);
            }
            if (targetSr > 0 && targetSr != aud.getSampleRate()) {
                aud = resample(aud, targetSr);
            }
            return aud;
        }
    }

    public static Tensor audioToTensor(AudioData audio) {
        return AudioTensors.toTensor(audio);
    }

    public static AudioData tensorToAudio(Tensor waveform, int sampleRate) {
        return AudioTensors.toAudioData(waveform, sampleRate);
    }

    public static AudioData toMono(AudioData audio) {
        Objects.requireNonNull(audio, "audio");
        float[] s = audio.getSamples();
        if (s == null) return audio;
        int ch = Math.max(1, audio.getChannels());
        if (ch == 1) return audio;
        int frames = s.length / ch;
        float[] mono = new float[frames];
        for (int i = 0; i < frames; i++) {
            float sum = 0f;
            for (int c = 0; c < ch; c++) sum += s[i * ch + c];
            mono[i] = sum / ch;
        }
        AudioData out = new AudioData(mono, audio.getSampleRate(), 1);
        out.setPath(audio.getPath());
        try { out.setDuration(audio.getDuration()); } catch (Exception ignored) {}
        return out;
    }

    public static AudioData resample(AudioData audio, int targetSr) {
        Objects.requireNonNull(audio, "audio");
        if (targetSr <= 0 || audio.getSampleRate() == targetSr) return audio;
        float[] src = audio.getSamples();
        if (src == null) return audio;
        int ch = Math.max(1, audio.getChannels());
        int srcFrames = src.length / ch;
        double ratio = (double) targetSr / audio.getSampleRate();
        int dstFrames = Math.max(1, (int) Math.round(srcFrames * ratio));
        float[] dst = new float[dstFrames * ch];
        for (int i = 0; i < dstFrames; i++) {
            double srcPos = i / ratio;
            int i0 = (int) Math.floor(srcPos);
            int i1 = Math.min(srcFrames - 1, i0 + 1);
            double t = srcPos - i0;
            for (int c = 0; c < ch; c++) {
                float a = src[Math.min(src.length - 1, i0 * ch + c)];
                float b = src[Math.min(src.length - 1, i1 * ch + c)];
                dst[i * ch + c] = (float) ((1 - t) * a + t * b);
            }
        }
        AudioData out = new AudioData(dst, targetSr, ch);
        out.setPath(audio.getPath());
        out.setDuration((double) dstFrames / targetSr);
        return out;
    }

    // ── Video load / convert ──────────────────────────────────────────────

    public static VideoData loadVideo(String path) throws IOException {
        return loadVideo(path, VideoOptions.defaults());
    }

    public static VideoData loadVideo(String path, VideoOptions opts) throws IOException {
        Objects.requireNonNull(path, "path");
        if (opts == null) opts = VideoOptions.defaults();
        if (!Files.isRegularFile(Path.of(path))) {
            throw new IOException("video file not found: " + path);
        }

        VideoBackend b = opts.backend;
        if (b == VideoBackend.AUTO) {
            b = isFFmpegAvailable() ? VideoBackend.FFMPEG : VideoBackend.STUB;
        }

        if (b == VideoBackend.FFMPEG) {
            try {
                return loadVideoFFmpeg(path, opts);
            } catch (Throwable t) {
                if (opts.backend == VideoBackend.FFMPEG) {
                    throw new IOException("FFmpeg video load failed: " + path, t);
                }
                warnFFmpegOnce("loadVideo", t);
            }
        }
        // STUB / fallback — lightweight offline mock (do NOT call VideoData.loadFromFile:
        // that method delegates back here and would recurse).
        return stubVideo(path);
    }

    /** Offline / no-FFmpeg placeholder video with a few tiny frames. */
    public static VideoData stubVideo(String path) {
        List<ImageData> frames = new ArrayList<>();
        for (int i = 0; i < 8; i++) {
            int shade = (i * 28) & 0xFF;
            int rgb = (shade << 16) | (shade << 8) | shade;
            java.awt.image.BufferedImage bi =
                    new java.awt.image.BufferedImage(32, 32, java.awt.image.BufferedImage.TYPE_INT_RGB);
            for (int y = 0; y < 32; y++)
                for (int x = 0; x < 32; x++)
                    bi.setRGB(x, y, rgb);
            frames.add(new ImageData(bi));
        }
        VideoData vid = new VideoData(frames, 8.0);
        vid.setPath(path);
        vid.setWidth(32);
        vid.setHeight(32);
        vid.setFormat(extensionOf(path));
        vid.setDuration(1.0);
        return vid;
    }

    public static VideoData loadVideoFFmpeg(String path, VideoOptions opts) throws Exception {
        Class<?> loader = Class.forName("org.bytedeco.pytorch.utils.ffmpeg.FFmpegLoader");
        try (AutoCloseable vf = (AutoCloseable) loader.getMethod("openVideo", String.class).invoke(null, path)) {
            Class<?> vfClass = vf.getClass();
            int width = ((Number) vfClass.getMethod("width").invoke(vf)).intValue();
            int height = ((Number) vfClass.getMethod("height").invoke(vf)).intValue();
            double fps = ((Number) vfClass.getMethod("fps").invoke(vf)).doubleValue();
            if (fps <= 0) fps = 30.0;

            int stride = opts.frameStride;
            if (opts.targetFps > 0 && opts.targetFps < fps) {
                stride = Math.max(stride, (int) Math.round(fps / opts.targetFps));
            }

            List<ImageData> frames = new ArrayList<>();
            int index = 0;
            // Prefer iterator API
            java.util.Iterator<?> it;
            try {
                it = (java.util.Iterator<?>) vfClass.getMethod("iterator").invoke(vf);
            } catch (NoSuchMethodException e) {
                it = null;
            }

            if (it != null) {
                while (it.hasNext()) {
                    Object frameObj = it.next();
                    if (!(frameObj instanceof Tensor frame)) {
                        index++;
                        continue;
                    }
                    if (index % stride == 0) {
                        ImageData id = tensorToImage(scale255ToUnit(frame));
                        if (opts.maxWidth > 0 && opts.maxHeight > 0
                                && (id.getWidth() > opts.maxWidth || id.getHeight() > opts.maxHeight)) {
                            id = id.resize(opts.maxWidth, opts.maxHeight);
                        }
                        frames.add(id);
                        if (opts.maxFrames > 0 && frames.size() >= opts.maxFrames) break;
                    }
                    index++;
                }
            } else {
                @SuppressWarnings("unchecked")
                List<Tensor> all = (List<Tensor>) vfClass.getMethod("readFrames").invoke(vf);
                for (Tensor frame : all) {
                    if (index % stride == 0) {
                        ImageData id = tensorToImage(scale255ToUnit(frame));
                        if (opts.maxWidth > 0 && opts.maxHeight > 0
                                && (id.getWidth() > opts.maxWidth || id.getHeight() > opts.maxHeight)) {
                            id = id.resize(opts.maxWidth, opts.maxHeight);
                        }
                        frames.add(id);
                        if (opts.maxFrames > 0 && frames.size() >= opts.maxFrames) break;
                    }
                    index++;
                }
            }

            double outFps = opts.targetFps > 0 ? opts.targetFps : (fps / stride);
            VideoData vid = new VideoData(frames, outFps);
            vid.setPath(path);
            vid.setWidth(frames.isEmpty() ? width : frames.get(0).getWidth());
            vid.setHeight(frames.isEmpty() ? height : frames.get(0).getHeight());
            vid.setFormat(extensionOf(path));
            if (!frames.isEmpty() && outFps > 0) {
                vid.setDuration(frames.size() / outFps);
            }

            if (opts.withAudio) {
                try {
                    AudioData aud = loadAudioFFmpeg(path, 16000, true);
                    vid.setAudioTrack(aud);
                } catch (Throwable ignored) {}
            }
            return vid;
        }
    }

    /**
     * Stack video frames into an NCHW tensor {@code [N,3,H,W]} in [0,1].
     * Frames are resized to the first frame's size if needed.
     */
    public static Tensor videoToTensor(VideoData video) {
        Objects.requireNonNull(video, "video");
        List<ImageData> frames = video.getFrames();
        if (frames == null || frames.isEmpty()) {
            throw new IllegalArgumentException("VideoData has no frames");
        }
        int h = frames.get(0).getHeight();
        int w = frames.get(0).getWidth();
        List<Tensor> ts = new ArrayList<>(frames.size());
        for (ImageData f : frames) {
            ImageData use = f;
            if (f.getHeight() != h || f.getWidth() != w) {
                use = f.resize(w, h);
            }
            ts.add(imageToTensor(use));
        }
        return torch.stack(new org.bytedeco.pytorch.TensorVector(ts.toArray(new Tensor[0])));
    }

    /** Extract frames at approximately {@code fps} samples per second. */
    public static List<ImageData> extractFrames(VideoData video, double fps) {
        Objects.requireNonNull(video, "video");
        List<ImageData> frames = video.getFrames();
        if (frames == null || frames.isEmpty()) return List.of();
        double srcFps = video.getFps() > 0 ? video.getFps() : 30.0;
        if (fps <= 0 || fps >= srcFps) return new ArrayList<>(frames);
        double step = srcFps / fps;
        List<ImageData> out = new ArrayList<>();
        for (double i = 0; i < frames.size(); i += step) {
            out.add(frames.get(Math.min(frames.size() - 1, (int) Math.round(i))));
        }
        return out;
    }

    /** Frame at time {@code second} (clamped). */
    public static ImageData frameAt(VideoData video, double second) {
        Objects.requireNonNull(video, "video");
        List<ImageData> frames = video.getFrames();
        if (frames == null || frames.isEmpty()) return null;
        double fps = video.getFps() > 0 ? video.getFps() : 30.0;
        int idx = (int) Math.round(second * fps);
        idx = Math.max(0, Math.min(frames.size() - 1, idx));
        return frames.get(idx);
    }

    // ── Batch helpers ─────────────────────────────────────────────────────

    /**
     * Decode a directory / glob of images into a list of {@link ImageData}
     * (OpenCV preferred).
     */
    public static List<ImageData> batchLoadImages(List<Path> files, ImageOptions opts) {
        List<ImageData> out = new ArrayList<>();
        if (files == null) return out;
        for (Path p : files) {
            try {
                out.add(loadImage(p.toString(), opts == null ? ImageOptions.defaults() : opts));
            } catch (Exception ignored) {}
        }
        return out;
    }

    public static List<AudioData> batchLoadAudio(List<Path> files, AudioOptions opts) {
        List<AudioData> out = new ArrayList<>();
        if (files == null) return out;
        for (Path p : files) {
            try {
                out.add(loadAudio(p.toString(), opts == null ? AudioOptions.defaults() : opts));
            } catch (Exception ignored) {}
        }
        return out;
    }

    public static List<VideoData> batchLoadVideo(List<Path> files, VideoOptions opts) {
        List<VideoData> out = new ArrayList<>();
        if (files == null) return out;
        for (Path p : files) {
            try {
                out.add(loadVideo(p.toString(), opts == null ? VideoOptions.defaults() : opts));
            } catch (Exception ignored) {}
        }
        return out;
    }

    /**
     * Stack a list of same-size images into NCHW {@code [N,C,H,W]} in [0,1].
     * Images are resized to the first image's size.
     */
    public static Tensor stackImages(List<ImageData> images) {
        Objects.requireNonNull(images, "images");
        if (images.isEmpty()) {
            throw new IllegalArgumentException("empty image list");
        }
        int h = images.get(0).getHeight();
        int w = images.get(0).getWidth();
        List<Tensor> ts = new ArrayList<>(images.size());
        for (ImageData img : images) {
            ImageData use = img;
            if (img.getHeight() != h || img.getWidth() != w) {
                use = img.resize(w, h);
            }
            ts.add(imageToTensor(use));
        }
        return torch.stack(new org.bytedeco.pytorch.TensorVector(ts.toArray(new Tensor[0])));
    }

    /**
     * Apply a torchvision-style {@code Transform} (or any function) to every image cell.
     * Transform class is accepted as {@code Object} to avoid hard coupling; if it
     * exposes {@code apply}/{@code forward}/{@code call}, it is invoked reflectively,
     * otherwise the object is treated as {@code java.util.function.Function}.
     */
    @SuppressWarnings("unchecked")
    public static ImageData applyImageTransform(ImageData image, Object transform) {
        Objects.requireNonNull(image, "image");
        if (transform == null) return image;
        try {
            if (transform instanceof java.util.function.Function<?, ?> fn) {
                Object r = ((java.util.function.Function<Object, Object>) fn).apply(image);
                return coerceImage(r, image);
            }
            for (String m : new String[]{"apply", "forward", "call", "transform"}) {
                try {
                    Object r = transform.getClass().getMethod(m, Object.class).invoke(transform, image);
                    return coerceImage(r, image);
                } catch (NoSuchMethodException ignored) {}
            }
            // try BufferedImage overload
            BufferedImage bi = image.getImage();
            if (bi != null) {
                for (String m : new String[]{"apply", "forward", "call"}) {
                    try {
                        Object r = transform.getClass().getMethod(m, BufferedImage.class).invoke(transform, bi);
                        return coerceImage(r, image);
                    } catch (NoSuchMethodException ignored) {}
                }
            }
        } catch (Exception e) {
            throw new IllegalStateException("image transform failed: " + e.getMessage(), e);
        }
        return image;
    }

    private static ImageData coerceImage(Object r, ImageData fallback) {
        if (r == null) return fallback;
        if (r instanceof ImageData id) return id;
        if (r instanceof BufferedImage bi) return new ImageData(bi);
        if (r instanceof Tensor t) return tensorToImage(t);
        return fallback;
    }

    // ── Embedding helpers ─────────────────────────────────────────────────

    /** Hash-style embedding from an image (color-aware + ImageData features). */
    public static EmbeddingData embedImage(ImageData image, int dim) {
        Objects.requireNonNull(image, "image");
        int d = Math.max(8, dim);
        float[] v = new float[d];
        // Color histogram / mean channels so solid red ≠ solid blue (grayscale path collapses them)
        BufferedImage bi = image.getImage();
        if (bi != null) {
            long rSum = 0, gSum = 0, bSum = 0, n = 0;
            int w = bi.getWidth(), h = bi.getHeight();
            int stepX = Math.max(1, w / 32), stepY = Math.max(1, h / 32);
            int[] hist = new int[24]; // 8 bins × 3 channels
            for (int y = 0; y < h; y += stepY) {
                for (int x = 0; x < w; x += stepX) {
                    int rgb = bi.getRGB(x, y);
                    int r = (rgb >> 16) & 0xFF, g = (rgb >> 8) & 0xFF, b = rgb & 0xFF;
                    rSum += r; gSum += g; bSum += b; n++;
                    hist[r >>> 5]++;
                    hist[8 + (g >>> 5)]++;
                    hist[16 + (b >>> 5)]++;
                }
            }
            if (n == 0) n = 1;
            v[0] = (float) rSum / n / 255f;
            v[1] = (float) gSum / n / 255f;
            v[2] = (float) bSum / n / 255f;
            v[3] = (float) w;
            v[4] = (float) h;
            int bins = Math.min(24, d - 5);
            double inv = 1.0 / n;
            for (int i = 0; i < bins; i++) v[5 + i] = (float) (hist[i] * inv);
        }
        // Mix in ImageData structural embedding if available
        try {
            float[] struct = image.extractEmbedding(Math.max(8, d / 2));
            if (struct != null) {
                int off = Math.min(d / 2, d - 1);
                for (int i = 0; i < struct.length && off + i < d; i++) {
                    v[off + i] += struct[i] * 0.5f;
                }
            }
        } catch (Exception ignored) {}
        // L2 normalize
        double norm = 0;
        for (float x : v) norm += x * x;
        norm = Math.sqrt(norm);
        if (norm > 1e-12) for (int i = 0; i < d; i++) v[i] /= (float) norm;
        if (d == dim) return new EmbeddingData(v, "hash-image");
        float[] out = new float[dim];
        System.arraycopy(v, 0, out, 0, Math.min(dim, d));
        return new EmbeddingData(out, "hash-image");
    }

    public static EmbeddingData embedAudio(AudioData audio, int dim) {
        Objects.requireNonNull(audio, "audio");
        float[] samples = audio.getSamples();
        float[] v = new float[dim];
        if (samples == null || samples.length == 0) return new EmbeddingData(v, "hash-audio");
        // simple statistical hash embedding
        double sum = 0, sum2 = 0, abs = 0;
        float min = Float.POSITIVE_INFINITY, max = Float.NEGATIVE_INFINITY;
        for (float s : samples) {
            sum += s; sum2 += s * s; abs += Math.abs(s);
            if (s < min) min = s;
            if (s > max) max = s;
        }
        int n = samples.length;
        v[0] = (float) (sum / n);
        if (dim > 1) v[1] = (float) Math.sqrt(Math.max(0, sum2 / n - v[0] * v[0]));
        if (dim > 2) v[2] = (float) (abs / n);
        if (dim > 3) v[3] = min;
        if (dim > 4) v[4] = max;
        // fill rest with strided sample energy
        for (int i = 5; i < dim; i++) {
            int idx = (int) ((long) (i - 5) * n / Math.max(1, dim - 5));
            v[i] = samples[Math.min(n - 1, idx)];
        }
        // L2 normalize
        double norm = 0;
        for (float x : v) norm += x * x;
        norm = Math.sqrt(norm);
        if (norm > 1e-12) for (int i = 0; i < dim; i++) v[i] /= (float) norm;
        return new EmbeddingData(v, "hash-audio");
    }

    public static EmbeddingData embedVideo(VideoData video, int dim) {
        Objects.requireNonNull(video, "video");
        List<ImageData> frames = video.getFrames();
        if (frames == null || frames.isEmpty()) {
            return new EmbeddingData(new float[dim], "hash-video");
        }
        // average a few frame embeddings
        int take = Math.min(8, frames.size());
        double step = frames.size() / (double) take;
        float[] acc = new float[dim];
        int used = 0;
        for (int i = 0; i < take; i++) {
            ImageData f = frames.get(Math.min(frames.size() - 1, (int) (i * step)));
            float[] e = f.extractEmbedding(dim);
            if (e == null) continue;
            for (int j = 0; j < dim; j++) acc[j] += e[j];
            used++;
        }
        if (used > 0) for (int j = 0; j < dim; j++) acc[j] /= used;
        return new EmbeddingData(acc, "hash-video");
    }


    /**
     * Neural image embedding via {@code EmbeddingRegistry} (torchvision ResNet/MobileNet by default).
     * Falls back to {@link #embedImage(ImageData, int)} hash path if the model cannot run.
     *
     * @param modelId e.g. {@code "resnet18"}, {@code "mobilenet_v2"}, {@code "clip-vit-base-patch32"}
     */
    public static EmbeddingData embedImageModel(ImageData image, String modelId) {
        Objects.requireNonNull(image, "image");
        try {
            Class<?> reg = Class.forName("org.bytedeco.pytorch.dataframe.ai.EmbeddingRegistry");
            Object model = reg.getMethod("get", String.class).invoke(null,
                    modelId == null || modelId.isBlank() ? "resnet18" : modelId);
            Class<?> modality = Class.forName("org.bytedeco.pytorch.dataframe.ai.Modality");
            Object imageMod = Enum.valueOf((Class<Enum>) modality.asSubclass(Enum.class), "IMAGE");
            float[] v = (float[]) model.getClass()
                    .getMethod("embed", Object.class, modality)
                    .invoke(model, image, imageMod);
            if (v != null) {
                String id = modelId == null ? "resnet18" : modelId;
                try {
                    Object spec = model.getClass().getMethod("spec").invoke(model);
                    id = String.valueOf(spec.getClass().getMethod("id").invoke(spec));
                } catch (Exception ignored) {}
                return new EmbeddingData(v, id);
            }
        } catch (Throwable t) {
            // fall through
        }
        int dim = 512;
        try {
            if (modelId != null && modelId.toLowerCase(Locale.ROOT).contains("mobilenet")) dim = 128;
        } catch (Exception ignored) {}
        return embedImage(image, dim);
    }

    /**
     * Neural audio embedding via {@code EmbeddingRegistry} (M5 / wav2letter by default).
     */
    public static EmbeddingData embedAudioModel(AudioData audio, String modelId) {
        Objects.requireNonNull(audio, "audio");
        try {
            Class<?> reg = Class.forName("org.bytedeco.pytorch.dataframe.ai.EmbeddingRegistry");
            Object model = reg.getMethod("get", String.class).invoke(null,
                    modelId == null || modelId.isBlank() ? "m5" : modelId);
            Class<?> modality = Class.forName("org.bytedeco.pytorch.dataframe.ai.Modality");
            Object audioMod = Enum.valueOf((Class<Enum>) modality.asSubclass(Enum.class), "AUDIO");
            float[] v = (float[]) model.getClass()
                    .getMethod("embed", Object.class, modality)
                    .invoke(model, audio, audioMod);
            if (v != null) {
                String id = modelId == null ? "m5" : modelId;
                try {
                    Object spec = model.getClass().getMethod("spec").invoke(model);
                    id = String.valueOf(spec.getClass().getMethod("id").invoke(spec));
                } catch (Exception ignored) {}
                return new EmbeddingData(v, id);
            }
        } catch (Throwable t) {
            // fall through
        }
        return embedAudio(audio, 256);
    }

    /** Neural video embedding (vision backbone over temporally pooled frames). */
    public static EmbeddingData embedVideoModel(VideoData video, String modelId) {
        Objects.requireNonNull(video, "video");
        try {
            Class<?> reg = Class.forName("org.bytedeco.pytorch.dataframe.ai.EmbeddingRegistry");
            Object model = reg.getMethod("get", String.class).invoke(null,
                    modelId == null || modelId.isBlank() ? "resnet18" : modelId);
            Class<?> modality = Class.forName("org.bytedeco.pytorch.dataframe.ai.Modality");
            Object videoMod = Enum.valueOf((Class<Enum>) modality.asSubclass(Enum.class), "VIDEO");
            float[] v = (float[]) model.getClass()
                    .getMethod("embed", Object.class, modality)
                    .invoke(model, video, videoMod);
            if (v != null) {
                String id = modelId == null ? "resnet18" : modelId;
                return new EmbeddingData(v, id);
            }
        } catch (Throwable t) {
            // fall through
        }
        return embedVideo(video, 512);
    }

    // ── internal utils ────────────────────────────────────────────────────

    static Tensor scale255ToUnit(Tensor t) {
        return t.div(new org.bytedeco.pytorch.Scalar(255.0));
    }

    static float maxAbs(Tensor t) {
        Tensor cpu = t.contiguous().cpu().to(ScalarType.Float);
        long n = cpu.numel();
        if (n == 0) return 0f;
        org.bytedeco.javacpp.FloatPointer fp = cpu.data_ptr_float();
        float m = 0f;
        long lim = Math.min(n, 4096); // sample for speed on huge tensors
        long step = Math.max(1, n / lim);
        for (long i = 0; i < n; i += step) {
            float v = Math.abs(fp.get(i));
            if (v > m) m = v;
        }
        return m;
    }

    static String extensionOf(String path) {
        if (path == null) return "";
        int slash = Math.max(path.lastIndexOf('/'), path.lastIndexOf('\\'));
        int dot = path.lastIndexOf('.');
        if (dot < 0 || dot < slash) return "";
        return path.substring(dot + 1);
    }
}
