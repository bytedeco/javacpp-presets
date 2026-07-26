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
package org.bytedeco.pytorch.utils.tqdm;

import java.io.PrintStream;
import java.util.Iterator;
import java.util.Locale;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.StringJoiner;

/**
 * A single tqdm progress bar over an {@link Iterator}.
 *
 * <p>Implements {@link Iterable} so it can be used in enhanced-for loops.
 * Closing is optional; the bar auto-closes when the iterator is exhausted.
 *
 * <p>Enhanced API (storch-tqdm / Python tqdm parity):
 * {@code leave}, {@code disable}, {@code ncols}, {@code ascii}, {@code colour},
 * {@code set_postfix(Map)}, {@code set_description}, nested-friendly printing.
 */
public final class TqdmBar<T> implements Iterable<T>, Iterator<T>, AutoCloseable {
    private final Iterator<T> source;
    private int total; // -1 if unknown
    private final long startNanos;
    private PrintStream out;

    private String desc;
    private String unit;
    private String postfix = "";
    private long n;
    private long lastPrintNanos;
    private double minIntervalSec;
    private boolean closed;
    private boolean started;
    private boolean leave = true;
    private boolean disable = false;
    private boolean ascii = false;
    private boolean manual = false;
    private int barWidth = 24;
    private int ncols = 80;
    private int position = 0;
    private ProgressBarColor colour = ProgressBarColor.NONE;

    TqdmBar(Iterator<T> source, int total, String unit) {
        this.source = Objects.requireNonNull(source, "source");
        this.total = total;
        this.unit = unit != null ? unit : "it";
        this.desc = "";
        this.out = Tqdm.defaultOut();
        this.minIntervalSec = 0.05;
        this.startNanos = System.nanoTime();
        this.lastPrintNanos = 0L;
        this.n = 0L;
        this.closed = false;
        this.started = false;
    }

    // ---- fluent configuration ----

    public TqdmBar<T> setDescription(String desc) {
        this.desc = desc != null ? desc : "";
        return this;
    }

    /** Python alias. */
    public TqdmBar<T> set_description(String desc) {
        return setDescription(desc);
    }

    public TqdmBar<T> setUnit(String unit) {
        this.unit = unit != null ? unit : "it";
        return this;
    }

    public TqdmBar<T> set_unit(String unit) {
        return setUnit(unit);
    }

    public TqdmBar<T> setPostfix(String postfix) {
        this.postfix = postfix != null ? postfix : "";
        return this;
    }

    public TqdmBar<T> set_postfix(String postfix) {
        return setPostfix(postfix);
    }

    /** Format map as {@code k=v, k2=v2} postfix (Python {@code set_postfix}). */
    public TqdmBar<T> set_postfix(Map<String, ?> values) {
        if (values == null || values.isEmpty()) {
            this.postfix = "";
            return this;
        }
        StringJoiner joiner = new StringJoiner(", ");
        for (Map.Entry<String, ?> e : values.entrySet()) {
            joiner.add(e.getKey() + "=" + String.valueOf(e.getValue()));
        }
        this.postfix = joiner.toString();
        return this;
    }

    public TqdmBar<T> setPostfix(Map<String, ?> values) {
        return set_postfix(values);
    }

    /** Minimum seconds between redraws (default 0.05). */
    public TqdmBar<T> setMinInterval(double seconds) {
        this.minIntervalSec = Math.max(0.0, seconds);
        return this;
    }

    public TqdmBar<T> set_mininterval(double seconds) {
        return setMinInterval(seconds);
    }

    /** Keep final bar on screen when closed (default true). */
    public TqdmBar<T> setLeave(boolean leave) {
        this.leave = leave;
        return this;
    }

    public TqdmBar<T> leave(boolean leave) {
        return setLeave(leave);
    }

    /** Disable all rendering (still counts). */
    public TqdmBar<T> setDisable(boolean disable) {
        this.disable = disable;
        return this;
    }

    public TqdmBar<T> disable(boolean disable) {
        return setDisable(disable);
    }

    /** Use ASCII {@code #/-} instead of block characters. */
    public TqdmBar<T> setAscii(boolean ascii) {
        this.ascii = ascii;
        return this;
    }

    public TqdmBar<T> ascii(boolean ascii) {
        return setAscii(ascii);
    }

    /** Approximate terminal width used for padding (default 80). */
    public TqdmBar<T> setNcols(int ncols) {
        this.ncols = Math.max(20, ncols);
        return this;
    }

    public TqdmBar<T> ncols(int ncols) {
        return setNcols(ncols);
    }

    /** Bar body width in characters (default 24). */
    public TqdmBar<T> setBarWidth(int width) {
        this.barWidth = Math.max(4, width);
        return this;
    }

    public TqdmBar<T> setColour(ProgressBarColor colour) {
        this.colour = colour != null ? colour : ProgressBarColor.NONE;
        return this;
    }

    public TqdmBar<T> colour(String name) {
        return setColour(ProgressBarColor.fromName(name));
    }

    public TqdmBar<T> setColour(String name) {
        return colour(name);
    }

    /** Nested-bar line position hint (0 = current line). */
    public TqdmBar<T> setPosition(int position) {
        this.position = Math.max(0, position);
        return this;
    }

    public TqdmBar<T> position(int position) {
        return setPosition(position);
    }

    public TqdmBar<T> setOut(PrintStream out) {
        this.out = out != null ? out : Tqdm.defaultOut();
        return this;
    }

    public TqdmBar<T> setTotal(int total) {
        this.total = total;
        return this;
    }

    TqdmBar<T> setManual(boolean manual) {
        this.manual = manual;
        return this;
    }

    // ---- state ----

    public long n() {
        return n;
    }

    public int total() {
        return total;
    }

    public boolean isDisable() {
        return disable;
    }

    public boolean isLeave() {
        return leave;
    }

    /** Iterations per second (0 if no elapsed time yet). */
    public double rate() {
        double elapsed = elapsedSeconds();
        return elapsed > 0.0 ? n / elapsed : 0.0;
    }

    public double elapsedSeconds() {
        return (System.nanoTime() - startNanos) / 1_000_000_000.0;
    }

    /** Manual update by {@code steps} (for custom loops). */
    public void update(long steps) {
        if (closed) {
            return;
        }
        n += Math.max(0L, steps);
        maybePrint(false);
        if (manual && total > 0 && n >= total) {
            close();
        }
    }

    public void update() {
        update(1);
    }

    /** Force a redraw. */
    public void refresh() {
        maybePrint(true);
    }

    /** Reset counter (keeps config). */
    public void reset() {
        n = 0L;
        closed = false;
        started = false;
        lastPrintNanos = 0L;
    }

    /** Reset with a new total. */
    public void reset(int newTotal) {
        this.total = newTotal;
        reset();
    }

    @Override
    public Iterator<T> iterator() {
        return this;
    }

    @Override
    public boolean hasNext() {
        if (manual) {
            return !closed && (total < 0 || n < total);
        }
        boolean more = source.hasNext();
        if (!more && !closed) {
            close();
        }
        return more;
    }

    @Override
    public T next() {
        if (manual) {
            throw new NoSuchElementException("manual bar: use update()");
        }
        if (!source.hasNext()) {
            close();
            throw new NoSuchElementException();
        }
        if (!started) {
            started = true;
            maybePrint(true);
        }
        T v = source.next();
        n++;
        maybePrint(false);
        if (!source.hasNext()) {
            close();
        }
        return v;
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        if (!disable) {
            synchronized (Tqdm.writeLock()) {
                if (leave) {
                    printLine(true);
                    out.println();
                } else {
                    // clear the line
                    out.print('\r');
                    for (int i = 0; i < ncols; i++) {
                        out.print(' ');
                    }
                    out.print('\r');
                }
                out.flush();
            }
        }
    }

    private void maybePrint(boolean force) {
        if (disable) {
            return;
        }
        long now = System.nanoTime();
        double since = (now - lastPrintNanos) / 1_000_000_000.0;
        if (force || lastPrintNanos == 0L || since >= minIntervalSec) {
            synchronized (Tqdm.writeLock()) {
                printLine(false);
            }
            lastPrintNanos = now;
        }
    }

    private void printLine(boolean finalLine) {
        double elapsed = elapsedSeconds();
        double rate = elapsed > 0.0 ? n / elapsed : 0.0;
        StringBuilder sb = new StringBuilder(ncols + 16);
        sb.append('\r');
        if (position > 0) {
            // move down/up is terminal-specific; keep simple prefix
            sb.append('[').append(position).append("] ");
        }
        if (!desc.isEmpty()) {
            sb.append(desc).append(": ");
        }
        char fill = ascii ? '#' : '█';
        char empty = ascii ? '-' : ' ';
        if (total > 0) {
            double frac = Math.min(1.0, (double) n / (double) total);
            int filled = (int) Math.round(frac * barWidth);
            String barBody;
            {
                StringBuilder body = new StringBuilder(barWidth);
                for (int i = 0; i < barWidth; i++) {
                    body.append(i < filled ? fill : empty);
                }
                barBody = colour.apply(body.toString());
            }
            sb.append(String.format(Locale.ROOT, "%3.0f%%|", frac * 100.0));
            sb.append(barBody);
            sb.append("| ");
            sb.append(n).append('/').append(total);
        } else {
            sb.append(n);
        }
        sb.append(" [");
        sb.append(formatTime(elapsed));
        if (total > 0 && n > 0 && n < total) {
            double remain = (total - n) / Math.max(rate, 1e-12);
            sb.append('<').append(formatTime(remain));
        }
        sb.append(", ");
        sb.append(String.format(Locale.ROOT, "%.2f", rate)).append(unit).append("/s");
        sb.append(']');
        if (!postfix.isEmpty()) {
            sb.append(", ").append(postfix);
        }
        while (sb.length() < ncols) {
            sb.append(' ');
        }
        out.print(sb);
        if (finalLine) {
            out.flush();
        }
    }

    private static String formatTime(double seconds) {
        if (seconds < 0 || Double.isNaN(seconds) || Double.isInfinite(seconds)) {
            return "??:??";
        }
        long s = Math.round(seconds);
        long h = s / 3600;
        long m = (s % 3600) / 60;
        long sec = s % 60;
        if (h > 0) {
            return String.format(Locale.ROOT, "%d:%02d:%02d", h, m, sec);
        }
        return String.format(Locale.ROOT, "%02d:%02d", m, sec);
    }
}
