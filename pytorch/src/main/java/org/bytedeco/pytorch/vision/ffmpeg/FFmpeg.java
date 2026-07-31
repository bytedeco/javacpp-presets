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
package org.bytedeco.pytorch.vision.ffmpeg;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

/**
 * ffmpeg-python style process pipeline — thin glue over the system {@code ffmpeg} binary.
 *
 * <p>Not a reimplementation of libav*; builds a CLI argv and runs it. For in-process
 * Frame/Packet control use {@link Av} / {@link Container}.
 *
 * <pre>{@code
 * FFmpeg.input("in.mp4")
 *     .filter("scale", "1280", "720")
 *     .output("out.mp4", "vcodec", "libx264", "crf", "23")
 *     .overwriteOutput()
 *     .run();
 * }</pre>
 */
public final class FFmpeg {

    private static final String[] CANDIDATES = {
            "ffmpeg",
            "/opt/homebrew/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "/usr/bin/ffmpeg"
    };

    private FFmpeg() {}

    /** Resolve a usable ffmpeg binary, or {@code null}. */
    public static String findBinary() {
        String prop = System.getProperty("ffmpeg.binary");
        if (prop != null && !prop.isEmpty() && Files.isExecutable(Paths.get(prop))) return prop;
        String env = System.getenv("FFMPEG_BINARY");
        if (env != null && !env.isEmpty() && Files.isExecutable(Paths.get(env))) return env;
        for (String c : CANDIDATES) {
            try {
                Path p = Paths.get(c);
                if (p.isAbsolute()) {
                    if (Files.isExecutable(p)) return c;
                } else {
                    Process pr = new ProcessBuilder(c, "-version")
                            .redirectErrorStream(true).start();
                    boolean ok = pr.waitFor(3, TimeUnit.SECONDS) && pr.exitValue() == 0;
                    pr.destroyForcibly();
                    if (ok) return c;
                }
            } catch (Exception ignored) {}
        }
        return null;
    }

    public static boolean isAvailable() {
        return findBinary() != null;
    }

    public static FFmpegNode input(String path) {
        return input(path, Collections.emptyMap());
    }

    /** {@code ffmpeg.input(path, ss=1, t=5)} style kwargs as alternating key/value. */
    public static FFmpegNode input(String path, Object... kwargs) {
        return input(path, kwargsToMap(kwargs));
    }

    public static FFmpegNode input(String path, Map<String, String> options) {
        Objects.requireNonNull(path, "path");
        return new FFmpegNode().addInput(path, options);
    }

    /** Concat several inputs (v=1,a=1 by default via filter_complex concat). */
    public static FFmpegNode concat(FFmpegNode... inputs) {
        return concat(true, true, inputs);
    }

    public static FFmpegNode concat(boolean video, boolean audio, FFmpegNode... inputs) {
        if (inputs == null || inputs.length == 0) {
            throw new FFmpegException("concat requires at least one input");
        }
        FFmpegNode n = new FFmpegNode();
        for (FFmpegNode in : inputs) {
            n.inputs.addAll(in.inputs);
        }
        int nIn = n.inputs.size();
        StringBuilder fc = new StringBuilder();
        for (int i = 0; i < nIn; i++) {
            if (video) fc.append('[').append(i).append(":v]");
            if (audio) fc.append('[').append(i).append(":a]");
        }
        fc.append("concat=n=").append(nIn)
                .append(":v=").append(video ? 1 : 0)
                .append(":a=").append(audio ? 1 : 0);
        if (video) fc.append("[v]");
        if (audio) fc.append("[a]");
        n.filterComplex = fc.toString();
        n.concatVideo = video;
        n.concatAudio = audio;
        return n;
    }

    /** Merge multiple output specs into one run (multi-bitrate). */
    public static FFmpegRun mergeOutputs(FFmpegNode... branches) {
        if (branches == null || branches.length == 0) {
            throw new FFmpegException("mergeOutputs requires branches");
        }
        FFmpegRun run = new FFmpegRun();
        // share first branch inputs; collect all outputs
        FFmpegNode base = branches[0];
        run.inputs.addAll(base.inputs);
        for (FFmpegNode b : branches) {
            if (b.filterComplex != null) {
                if (run.filterComplex == null) run.filterComplex = b.filterComplex;
                else run.filterComplex = run.filterComplex + ";" + b.filterComplex;
            }
            for (String f : b.filters) {
                run.globalFilters.add(f);
            }
            run.outputs.addAll(b.outputs);
        }
        run.overwrite = base.overwrite;
        return run;
    }

    static Map<String, String> kwargsToMap(Object... kwargs) {
        if (kwargs == null || kwargs.length == 0) return Collections.emptyMap();
        if (kwargs.length % 2 != 0) {
            throw new IllegalArgumentException("kwargs require key/value pairs");
        }
        Map<String, String> m = new LinkedHashMap<>();
        for (int i = 0; i < kwargs.length; i += 2) {
            m.put(String.valueOf(kwargs[i]), String.valueOf(kwargs[i + 1]));
        }
        return m;
    }

    // ── node / run types ──────────────────────────────────────────────────

    /**
     * Builder node: inputs + filters + one or more outputs.
     * Terminal calls: {@link #run()}, {@link #runAsync(boolean)}, {@link #compile()}.
     */
    public static final class FFmpegNode {
        final List<InputSpec> inputs = new ArrayList<>();
        final List<String> filters = new ArrayList<>(); // simple -vf chain pieces
        final List<OutputSpec> outputs = new ArrayList<>();
        String filterComplex;
        boolean concatVideo;
        boolean concatAudio;
        boolean overwrite;
        boolean pipeStdout;

        FFmpegNode addInput(String path, Map<String, String> options) {
            inputs.add(new InputSpec(path, options != null ? new LinkedHashMap<>(options) : new LinkedHashMap<>()));
            return this;
        }

        /** Another input (e.g. watermark). */
        public FFmpegNode input(String path) {
            return addInput(path, Collections.emptyMap());
        }

        public FFmpegNode input(String path, Object... kwargs) {
            return addInput(path, kwargsToMap(kwargs));
        }

        /**
         * Append a simple video filter. Args joined with {@code :} after the name
         * (ffmpeg-python {@code .filter("scale", 1280, 720)} → {@code scale=1280:720}).
         * <p>If {@code name} already contains {@code =} or {@code [}, treated as raw expression
         * (overlay / multi-label graphs).
         */
        public FFmpegNode filter(String name, Object... args) {
            Objects.requireNonNull(name, "name");
            if (name.contains("=") || name.contains("[")) {
                filters.add(name);
                return this;
            }
            if (args == null || args.length == 0) {
                filters.add(name);
            } else {
                String joined = Arrays.stream(args).map(String::valueOf).collect(Collectors.joining(":"));
                // named kwargs style: width=1280 already in args as strings
                filters.add(name + "=" + joined);
            }
            return this;
        }

        public FFmpegNode output(String path, Object... kwargs) {
            return output(path, kwargsToMap(kwargs));
        }

        public FFmpegNode output(String path, Map<String, String> options) {
            Objects.requireNonNull(path, "path");
            OutputSpec o = new OutputSpec(path, options != null ? new LinkedHashMap<>(options) : new LinkedHashMap<>());
            // attach pending simple filters as -vf on this output if no filter_complex
            if (filterComplex == null && !filters.isEmpty()) {
                o.vf = String.join(",", filters);
            }
            outputs.add(o);
            return this;
        }

        public FFmpegNode overwriteOutput() {
            overwrite = true;
            return this;
        }

        /** Build argv without running. */
        public List<String> compile() {
            return toRun().compile();
        }

        public void run() {
            toRun().run();
        }

        public FFmpegProcess runAsync() {
            return runAsync(false);
        }

        public FFmpegProcess runAsync(boolean pipeStdout) {
            this.pipeStdout = pipeStdout;
            return toRun().runAsync(pipeStdout);
        }

        FFmpegRun toRun() {
            FFmpegRun r = new FFmpegRun();
            r.inputs.addAll(inputs);
            r.globalFilters.addAll(filters);
            r.outputs.addAll(outputs);
            r.filterComplex = filterComplex;
            r.concatVideo = concatVideo;
            r.concatAudio = concatAudio;
            r.overwrite = overwrite;
            r.pipeStdout = pipeStdout;
            // if filter_complex concat, map streams on outputs
            if (filterComplex != null && filterComplex.contains("concat=")) {
                for (OutputSpec o : r.outputs) {
                    if (concatVideo && !o.options.containsKey("map")) {
                        // use -map via special keys
                        o.maps.add("[v]");
                    }
                    if (concatAudio) o.maps.add("[a]");
                }
            }
            return r;
        }
    }

    static final class InputSpec {
        final String path;
        final Map<String, String> options;
        InputSpec(String path, Map<String, String> options) {
            this.path = path;
            this.options = options;
        }
    }

    static final class OutputSpec {
        final String path;
        final Map<String, String> options;
        String vf;
        final List<String> maps = new ArrayList<>();
        OutputSpec(String path, Map<String, String> options) {
            this.path = path;
            this.options = options;
        }
    }

    /** Compiled multi-output run. */
    public static final class FFmpegRun {
        final List<InputSpec> inputs = new ArrayList<>();
        final List<String> globalFilters = new ArrayList<>();
        final List<OutputSpec> outputs = new ArrayList<>();
        String filterComplex;
        boolean concatVideo;
        boolean concatAudio;
        boolean overwrite;
        boolean pipeStdout;

        public List<String> compile() {
            String bin = findBinary();
            if (bin == null) throw new FFmpegException("ffmpeg binary not found on PATH");
            List<String> cmd = new ArrayList<>();
            cmd.add(bin);
            cmd.add("-hide_banner");
            cmd.add("-loglevel");
            cmd.add("error");
            if (overwrite) cmd.add("-y");

            for (InputSpec in : inputs) {
                for (Map.Entry<String, String> e : in.options.entrySet()) {
                    cmd.add("-" + e.getKey());
                    cmd.add(e.getValue());
                }
                cmd.add("-i");
                cmd.add(in.path);
            }

            if (filterComplex != null && !filterComplex.isEmpty()) {
                cmd.add("-filter_complex");
                cmd.add(filterComplex);
            } else if (!globalFilters.isEmpty() && outputs.isEmpty()) {
                // no output yet — still allow compile of -vf
                cmd.add("-vf");
                cmd.add(String.join(",", globalFilters));
            }

            if (outputs.isEmpty()) {
                throw new FFmpegException("no output specified");
            }

            for (OutputSpec out : outputs) {
                for (String m : out.maps) {
                    cmd.add("-map");
                    cmd.add(m);
                }
                String vf = out.vf;
                if (vf == null && filterComplex == null && !globalFilters.isEmpty() && outputs.size() == 1) {
                    vf = String.join(",", globalFilters);
                }
                if (vf != null && !vf.isEmpty()) {
                    cmd.add("-vf");
                    cmd.add(vf);
                }
                // common option aliases
                for (Map.Entry<String, String> e : out.options.entrySet()) {
                    String k = e.getKey();
                    String v = e.getValue();
                    if ("vcodec".equals(k) || "c:v".equals(k)) {
                        cmd.add("-c:v"); cmd.add(v);
                    } else if ("acodec".equals(k) || "c:a".equals(k)) {
                        cmd.add("-c:a"); cmd.add(v);
                    } else if ("format".equals(k) || "f".equals(k)) {
                        cmd.add("-f"); cmd.add(v);
                    } else if ("pix_fmt".equals(k)) {
                        cmd.add("-pix_fmt"); cmd.add(v);
                    } else if ("map".equals(k)) {
                        cmd.add("-map"); cmd.add(v);
                    } else if (isFlagOnlyOption(k, v)) {
                        // -an / -vn / -y-style flags: no value argument
                        cmd.add("-" + k);
                    } else {
                        cmd.add("-" + k);
                        if (v != null && !v.isEmpty()) cmd.add(v);
                    }
                }
                cmd.add(out.path);
            }
            return cmd;
        }

        /** Options that are bare flags (or boolean true → flag present). */
        private static boolean isFlagOnlyOption(String k, String v) {
            if (k == null) return false;
            // classic ffmpeg boolean stream disables / muxer flags
            switch (k) {
                case "an": case "vn": case "sn": case "dn":
                case "shortest": case "nodefault": case "n": case "y":
                case "hide_banner": case "ignore_unknown":
                    return v == null || v.isEmpty()
                            || "true".equalsIgnoreCase(v)
                            || "1".equals(v)
                            || "yes".equalsIgnoreCase(v);
                default:
                    return "true".equalsIgnoreCase(v) || "yes".equalsIgnoreCase(v);
            }
        }

        public void run() {
            List<String> cmd = compile();
            try {
                ProcessBuilder pb = new ProcessBuilder(cmd);
                pb.redirectErrorStream(true);
                Process p = pb.start();
                ByteArrayOutputStream bos = new ByteArrayOutputStream();
                try (InputStream in = p.getInputStream()) {
                    in.transferTo(bos);
                }
                boolean finished = p.waitFor(600, TimeUnit.SECONDS);
                if (!finished) {
                    p.destroyForcibly();
                    throw new FFmpegException("ffmpeg timed out: " + String.join(" ", cmd));
                }
                if (p.exitValue() != 0) {
                    String err = bos.toString(StandardCharsets.UTF_8);
                    throw new FFmpegException("ffmpeg exit " + p.exitValue() + ": " + trim(err)
                            + "\ncmd: " + String.join(" ", cmd));
                }
            } catch (FFmpegException e) {
                throw e;
            } catch (Exception e) {
                throw new FFmpegException("ffmpeg run failed: " + e.getMessage(), e);
            }
        }

        public FFmpegProcess runAsync(boolean pipeStdout) {
            List<String> cmd = compile();
            try {
                ProcessBuilder pb = new ProcessBuilder(cmd);
                if (pipeStdout) {
                    // keep stderr separate so stdout stays pure rawvideo; drain stderr async
                    pb.redirectError(ProcessBuilder.Redirect.PIPE);
                } else {
                    pb.redirectErrorStream(true);
                }
                Process p = pb.start();
                FFmpegProcess fp = new FFmpegProcess(p, cmd);
                if (pipeStdout) {
                    Thread t = new Thread(() -> {
                        try (InputStream err = p.getErrorStream()) {
                            err.transferTo(new ByteArrayOutputStream());
                        } catch (Exception ignored) {}
                    }, "ffmpeg-stderr-drain");
                    t.setDaemon(true);
                    t.start();
                }
                return fp;
            } catch (Exception e) {
                throw new FFmpegException("ffmpeg runAsync failed: " + e.getMessage(), e);
            }
        }

        private static String trim(String s) {
            if (s == null) return "";
            s = s.trim();
            return s.length() > 800 ? s.substring(0, 800) + "..." : s;
        }
    }

    /** Handle for async / piped ffmpeg. */
    public static final class FFmpegProcess implements AutoCloseable {
        private final Process process;
        private final List<String> cmd;

        FFmpegProcess(Process process, List<String> cmd) {
            this.process = process;
            this.cmd = cmd;
        }

        public InputStream getStdout() { return process.getInputStream(); }
        public InputStream getStderr() { return process.getErrorStream(); }
        public Process process() { return process; }
        public List<String> command() { return Collections.unmodifiableList(cmd); }

        public int waitFor() throws InterruptedException {
            return process.waitFor();
        }

        public boolean waitFor(long timeout, TimeUnit unit) throws InterruptedException {
            return process.waitFor(timeout, unit);
        }

        @Override
        public void close() {
            process.destroyForcibly();
        }
    }
}
