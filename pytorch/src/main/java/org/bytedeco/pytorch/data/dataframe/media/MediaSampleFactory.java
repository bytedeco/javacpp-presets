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
package org.bytedeco.pytorch.data.dataframe.media;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.TimeUnit;

/**
 * Generate tiny but real media files (mp3 / mp4 / wav / png) for offline FFmpeg /
 * OpenCV interop benchmarks.
 *
 * <p>Prefers a system {@code ffmpeg} binary when present; otherwise writes pure-Java
 * WAV/PNG and leaves compressed containers to the caller.
 *
 * <pre>{@code
 * Path dir = MediaSampleFactory.createCorpus(Files.createTempDirectory("media"));
 * // dir contains: tone440.wav, tone440.mp3, clip_color.mp4, clip_tone.mp4, …
 * }</pre>
 */
public final class MediaSampleFactory {

    private MediaSampleFactory() {}

    private static final String[] FFMPEG_CANDIDATES = {
            "ffmpeg",
            "/opt/homebrew/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "/usr/bin/ffmpeg"
    };

    /** Resolve a usable ffmpeg binary, or {@code null}. */
    public static String findFFmpeg() {
        for (String c : FFMPEG_CANDIDATES) {
            try {
                Process p = new ProcessBuilder(c, "-version")
                        .redirectErrorStream(true)
                        .start();
                boolean ok = p.waitFor(5, TimeUnit.SECONDS) && p.exitValue() == 0;
                if (ok) return c;
            } catch (Exception ignored) {}
        }
        return null;
    }

    public static boolean hasFFmpeg() {
        return findFFmpeg() != null;
    }

    /**
     * Build a small multimodal corpus under {@code dir}:
     * <ul>
     *   <li>{@code tone440.wav} / {@code tone880.wav} — pure Java PCM</li>
     *   <li>{@code tone440.mp3} / {@code tone880.mp3} — via ffmpeg (if available)</li>
     *   <li>{@code solid_red.png} / {@code solid_blue.png}</li>
     *   <li>{@code clip_color.mp4} — color bars + sine (ffmpeg)</li>
     *   <li>{@code clip_gray.mp4} — grayscale ramp (ffmpeg)</li>
     * </ul>
     *
     * @return the same {@code dir}
     */
    public static Path createCorpus(Path dir) throws IOException {
        Files.createDirectories(dir);
        writeWav(dir.resolve("tone440.wav"), 16000, 0.5, 440.0);
        writeWav(dir.resolve("tone880.wav"), 16000, 0.5, 880.0);
        writePng(dir.resolve("solid_red.png"), 64, 48, 0xFF0000);
        writePng(dir.resolve("solid_blue.png"), 64, 48, 0x0000FF);
        writePng(dir.resolve("solid_green.png"), 32, 32, 0x00FF00);

        String ff = findFFmpeg();
        if (ff != null) {
            // mp3 from wav
            run(ff, "-y", "-i", dir.resolve("tone440.wav").toString(),
                    "-codec:a", "libmp3lame", "-qscale:a", "9",
                    dir.resolve("tone440.mp3").toString());
            run(ff, "-y", "-i", dir.resolve("tone880.wav").toString(),
                    "-codec:a", "libmp3lame", "-qscale:a", "9",
                    dir.resolve("tone880.mp3").toString());
            // short color video with audio (2 seconds, 10 fps, 160x120)
            run(ff, "-y",
                    "-f", "lavfi", "-i", "smptebars=size=160x120:rate=10:duration=2",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:duration=2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
                    dir.resolve("clip_color.mp4").toString());
            // grayscale ramp video (no audio) — different content for embed tests
            run(ff, "-y",
                    "-f", "lavfi", "-i", "gradients=size=160x120:rate=8:duration=1.5",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    dir.resolve("clip_gray.mp4").toString());
            // pure audio-in-mp4 container
            run(ff, "-y",
                    "-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=22050:duration=1",
                    "-c:a", "aac",
                    dir.resolve("beep.m4a").toString());
        }
        // manifest
        List<String> lines = new ArrayList<>();
        lines.add("ffmpeg=" + (ff != null ? ff : "none"));
        try (var stream = Files.list(dir)) {
            stream.sorted().forEach(p -> lines.add(p.getFileName().toString()));
        }
        Files.write(dir.resolve("MANIFEST.txt"), lines, StandardCharsets.UTF_8);
        return dir;
    }

    /** Write a mono 16-bit PCM WAV. */
    public static void writeWav(Path path, int sampleRate, double seconds, double freq)
            throws IOException {
        int n = Math.max(1, (int) (sampleRate * seconds));
        float[] samples = new float[n];
        for (int i = 0; i < n; i++) {
            samples[i] = (float) (Math.sin(2 * Math.PI * freq * i / sampleRate) * 0.4);
        }
        int dataBytes = n * 2;
        java.io.ByteArrayOutputStream bos = new java.io.ByteArrayOutputStream();
        java.io.DataOutputStream out = new java.io.DataOutputStream(bos);
        out.writeBytes("RIFF");
        out.writeInt(Integer.reverseBytes(36 + dataBytes));
        out.writeBytes("WAVE");
        out.writeBytes("fmt ");
        out.writeInt(Integer.reverseBytes(16));
        out.writeShort(Short.reverseBytes((short) 1));
        out.writeShort(Short.reverseBytes((short) 1));
        out.writeInt(Integer.reverseBytes(sampleRate));
        out.writeInt(Integer.reverseBytes(sampleRate * 2));
        out.writeShort(Short.reverseBytes((short) 2));
        out.writeShort(Short.reverseBytes((short) 16));
        out.writeBytes("data");
        out.writeInt(Integer.reverseBytes(dataBytes));
        out.flush();
        byte[] header = bos.toByteArray();
        java.nio.ByteBuffer pcm = java.nio.ByteBuffer.allocate(dataBytes)
                .order(java.nio.ByteOrder.LITTLE_ENDIAN);
        for (float s : samples) {
            float c = Math.max(-1f, Math.min(1f, s));
            pcm.putShort((short) (c * 32767));
        }
        byte[] all = new byte[header.length + dataBytes];
        System.arraycopy(header, 0, all, 0, header.length);
        System.arraycopy(pcm.array(), 0, all, header.length, dataBytes);
        Files.write(path, all);
    }

    public static void writePng(Path path, int w, int h, int rgb) throws IOException {
        java.awt.image.BufferedImage bi =
                new java.awt.image.BufferedImage(w, h, java.awt.image.BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                bi.setRGB(x, y, rgb);
        javax.imageio.ImageIO.write(bi, "png", path.toFile());
    }

    private static void run(String... cmd) throws IOException {
        ProcessBuilder pb = new ProcessBuilder(cmd);
        pb.redirectErrorStream(true);
        Process p = pb.start();
        StringBuilder log = new StringBuilder();
        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(p.getInputStream(), StandardCharsets.UTF_8))) {
            String line;
            while ((line = br.readLine()) != null) {
                if (log.length() < 4000) log.append(line).append('\n');
            }
        }
        try {
            boolean finished = p.waitFor(60, TimeUnit.SECONDS);
            if (!finished) {
                p.destroyForcibly();
                throw new IOException("ffmpeg timed out: " + String.join(" ", cmd));
            }
            if (p.exitValue() != 0) {
                throw new IOException("ffmpeg exit " + p.exitValue()
                        + " for " + String.join(" ", cmd)
                        + "\n" + log);
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("ffmpeg interrupted", e);
        }
    }

    /** True if path looks like a real (non-empty) media file. */
    public static boolean isNonEmptyFile(Path p) {
        try {
            return p != null && Files.isRegularFile(p) && Files.size(p) > 64;
        } catch (IOException e) {
            return false;
        }
    }

    public static String extension(Path p) {
        if (p == null) return "";
        String n = p.getFileName().toString().toLowerCase(Locale.ROOT);
        int dot = n.lastIndexOf('.');
        return dot < 0 ? "" : n.substring(dot + 1);
    }
}
