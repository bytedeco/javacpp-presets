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

import org.bytedeco.ffmpeg.avfilter.AVFilter;
import org.bytedeco.ffmpeg.avfilter.AVFilterContext;
import org.bytedeco.ffmpeg.avfilter.AVFilterGraph;
import org.bytedeco.ffmpeg.avfilter.AVFilterInOut;
import org.bytedeco.ffmpeg.avutil.AVFrame;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Pointer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import static org.bytedeco.ffmpeg.global.avfilter.*;
import static org.bytedeco.ffmpeg.global.avutil.av_frame_alloc;
import static org.bytedeco.ffmpeg.global.avutil.av_frame_free;

/**
 * Filter graph glue — PyAV {@code av.filter.Graph} / {@code av.FilterGraph}.
 *
 * <p>Preferred: {@link #open(VideoStream, String)} with a libavfilter description
 * like {@code "scale=960:540,hqdn3d"}.
 *
 * <pre>{@code
 * try (FilterGraph g = FilterGraph.open(vin, "scale=80:60")) {
 *     g.push(frame);
 *     VideoFrame out;
 *     while ((out = g.pullVideo()) != null) { ...; out.close(); }
 * }
 * }</pre>
 */
public final class FilterGraph implements AutoCloseable {

    private AVFilterGraph graph;
    private FilterContext bufferSrc;
    private FilterContext bufferSink;
    private final List<FilterContext> nodes = new ArrayList<>();
    private boolean configured;
    private boolean closed;
    private Rational timeBase = new Rational(1, 1);

    public FilterGraph() {
        FFmpegNative.load();
        graph = avfilter_graph_alloc();
        if (graph == null || graph.isNull()) {
            throw new FFmpegException("avfilter_graph_alloc failed");
        }
    }

    /**
     * Build a video filter chain: {@code buffer → filters → buffersink}.
     */
    public static FilterGraph open(VideoStream template, String filters) {
        Objects.requireNonNull(template, "template");
        Objects.requireNonNull(filters, "filters");
        FilterGraph g = new FilterGraph();
        g.timeBase = template.timeBase();
        g.bufferSrc = g.addBuffer(template);
        g.bufferSink = g.add("buffersink", null, "out");

        String desc = filters.trim();
        if (!desc.contains("[")) {
            desc = "[in]" + desc + "[out]";
        }

        AVFilterInOut outputs = avfilter_inout_alloc();
        AVFilterInOut inputs = avfilter_inout_alloc();
        if (outputs == null || inputs == null) {
            g.close();
            throw new FFmpegException("avfilter_inout_alloc failed");
        }
        try {
            outputs.name(new BytePointer("in"));
            outputs.filter_ctx(g.bufferSrc.ctx);
            outputs.pad_idx(0);
            outputs.next(null);

            inputs.name(new BytePointer("out"));
            inputs.filter_ctx(g.bufferSink.ctx);
            inputs.pad_idx(0);
            inputs.next(null);

            // @ByPtrPtr AVFilterInOut — holders may be rewritten by parse
            int ret = avfilter_graph_parse_ptr(g.graph, desc, inputs, outputs, (Pointer) null);
            // free whatever remains in the holders
            try { avfilter_inout_free(inputs); } catch (Throwable ignored) {}
            try { avfilter_inout_free(outputs); } catch (Throwable ignored) {}
            inputs = null;
            outputs = null;
            FFmpegNative.check(ret, "avfilter_graph_parse_ptr");
            g.configure();
            return g;
        } catch (RuntimeException e) {
            g.close();
            throw e;
        } finally {
            if (inputs != null) {
                try { avfilter_inout_free(inputs); } catch (Throwable ignored) {}
            }
            if (outputs != null) {
                try { avfilter_inout_free(outputs); } catch (Throwable ignored) {}
            }
        }
    }

    public FilterContext addBuffer(VideoStream template) {
        ensureOpen();
        int w = template.width();
        int h = template.height();
        Rational tb = template.timeBase();
        Rational fr = template.rate();
        String pix = template.pixFmtName();
        if (pix == null) pix = "yuv420p";
        String args = String.format(
                "video_size=%dx%d:pix_fmt=%s:time_base=%d/%d:pixel_aspect=1/1",
                w, h, pix, Math.max(1, tb.num), Math.max(1, tb.den));
        if (fr.num > 0 && fr.den > 0) {
            args += String.format(":frame_rate=%d/%d", fr.num, fr.den);
        }
        FilterContext src = add("buffer", args, "in");
        this.bufferSrc = src;
        this.timeBase = tb;
        return src;
    }

    public FilterContext add(String name) {
        return add(name, null, name + nodes.size());
    }

    public FilterContext add(String name, String args) {
        return add(name, args, name + nodes.size());
    }

    public FilterContext add(String name, String args, String instanceName) {
        ensureOpen();
        AVFilter filt = avfilter_get_by_name(name);
        if (filt == null || filt.isNull()) {
            throw new FFmpegException("unknown filter: " + name);
        }
        // @ByPtrPtr AVFilterContext holder
        AVFilterContext ctx = new AVFilterContext(null);
        int ret = avfilter_graph_create_filter(ctx, filt, instanceName, args, null, graph);
        FFmpegNative.check(ret, "avfilter_graph_create_filter(" + name + ")");
        if (ctx.isNull()) throw new FFmpegException("filter context null after create: " + name);
        FilterContext fc = new FilterContext(this, ctx, name);
        nodes.add(fc);
        if ("buffersink".equals(name) || "abuffersink".equals(name)) bufferSink = fc;
        if ("buffer".equals(name) || "abuffer".equals(name)) bufferSrc = fc;
        return fc;
    }

    public void link(FilterContext src, FilterContext dst) {
        ensureOpen();
        FFmpegNative.check(avfilter_link(src.ctx, 0, dst.ctx, 0),
                "avfilter_link(" + src.name + " -> " + dst.name + ")");
    }

    public void configure() {
        ensureOpen();
        FFmpegNative.check(avfilter_graph_config(graph, null), "avfilter_graph_config");
        configured = true;
    }

    public void push(VideoFrame frame) {
        ensureConfigured();
        if (bufferSrc == null) throw new FFmpegException("no buffer source");
        AVFrame raw = frame == null ? null : frame.nativeFrame();
        int ret = av_buffersrc_add_frame(bufferSrc.ctx, raw);
        if (ret < 0 && !FFmpegNative.isEof(ret)) {
            if (frame != null || !FFmpegNative.isEagain(ret)) {
                if (frame == null && FFmpegNative.isEof(ret)) return;
                throw new FFmpegException("av_buffersrc_add_frame", ret);
            }
        }
    }

    public void push(Frame frame) {
        if (frame == null) {
            push((VideoFrame) null);
        } else if (frame instanceof VideoFrame) {
            push((VideoFrame) frame);
        } else {
            throw new FFmpegException("FilterGraph.push expects VideoFrame");
        }
    }

    /** Pull one filtered frame, or null if need more input / EOF. Caller owns result. */
    public VideoFrame pullVideo() {
        ensureConfigured();
        if (bufferSink == null) throw new FFmpegException("no buffersink");
        AVFrame fr = av_frame_alloc();
        int ret = av_buffersink_get_frame(bufferSink.ctx, fr);
        if (FFmpegNative.isEagain(ret) || FFmpegNative.isEof(ret)) {
            av_frame_free(fr);
            return null;
        }
        if (ret < 0) {
            av_frame_free(fr);
            throw new FFmpegException("av_buffersink_get_frame", ret);
        }
        return new VideoFrame(fr, timeBase);
    }

    public Frame pull() {
        return pullVideo();
    }

    private void ensureConfigured() {
        ensureOpen();
        if (!configured) configure();
    }

    private void ensureOpen() {
        if (closed || graph == null || graph.isNull()) {
            throw new FFmpegException("FilterGraph is closed");
        }
    }

    AVFilterGraph nativeGraph() { return graph; }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        if (graph != null && !graph.isNull()) {
            avfilter_graph_free(graph);
            graph = null;
        }
        nodes.clear();
        bufferSrc = null;
        bufferSink = null;
    }
}
