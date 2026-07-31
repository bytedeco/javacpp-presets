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

import org.bytedeco.ffmpeg.avfilter.AVFilterContext;

/**
 * One node in a {@link FilterGraph} — PyAV filter context sugar.
 *
 * <p>For chained construction prefer {@link FilterGraph#open(VideoStream, String)}.
 * Manual style:
 * <pre>{@code
 * FilterContext scale = graph.add("scale", "960:540");
 * graph.link(src, scale);
 * }</pre>
 */
public final class FilterContext {

    final FilterGraph graph;
    final AVFilterContext ctx;
    final String name;

    FilterContext(FilterGraph graph, AVFilterContext ctx, String name) {
        this.graph = graph;
        this.ctx = ctx;
        this.name = name;
    }

    public String name() { return name; }

    public AVFilterContext nativeContext() { return ctx; }

    /**
     * Convenience: add {@code name} filter to the same graph and link this → new.
     * Returns the new node (PyAV-ish {@code node.filter("scale", "960:540")}).
     */
    public FilterContext filter(String filterName, String args) {
        FilterContext next = graph.add(filterName, args);
        graph.link(this, next);
        return next;
    }

    public FilterContext filter(String filterName) {
        return filter(filterName, null);
    }

    /** Link this node to {@code sink}. */
    public void linkTo(FilterContext sink) {
        graph.link(this, sink);
    }

    @Override
    public String toString() {
        return "FilterContext(" + name + ")";
    }
}
