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
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

/**
 * Factory for tqdm-style progress bars (Java port inspired by Python {@code tqdm}
 * and bytedeco/storch-tqdm).
 *
 * <pre>{@code
 * for (Integer i : Tqdm.range(100)) {
 *     // work
 * }
 * for (Integer i : Tqdm.trange(50).setDescription("epoch").set_postfix(Map.of("loss", "0.1"))) {
 *     train();
 * }
 * for (Sample s : Tqdm.of(dataset)) {
 *     train(s);
 * }
 * Tqdm.write("checkpoint saved");
 * }</pre>
 */
public final class Tqdm {
    private static final Object WRITE_LOCK = new Object();

    private Tqdm() {}

    /** Iterate {@code 0 .. n-1} with a progress bar. */
    public static TqdmBar<Integer> range(int n) {
        return range(0, n, 1);
    }

    /** Alias of {@link #range(int)} matching Python {@code tqdm.trange}. */
    public static TqdmBar<Integer> trange(int n) {
        return range(n);
    }

    /** Iterate {@code start .. end-1} with step. */
    public static TqdmBar<Integer> range(int start, int end, int step) {
        if (step == 0) {
            throw new IllegalArgumentException("step must be non-zero");
        }
        final int total = step > 0
                ? Math.max(0, (end - start + step - 1) / step)
                : Math.max(0, (start - end - step - 1) / (-step));
        return new TqdmBar<>(new Iterator<Integer>() {
            int cur = start;
            int produced = 0;

            @Override
            public boolean hasNext() {
                return produced < total;
            }

            @Override
            public Integer next() {
                int v = cur;
                cur += step;
                produced++;
                return v;
            }
        }, total, "it");
    }

    /** Alias of {@link #range(int, int, int)}. */
    public static TqdmBar<Integer> trange(int start, int end, int step) {
        return range(start, end, step);
    }

    /** Wrap any iterable; total is taken from {@link Collection#size()} when possible. */
    public static <T> TqdmBar<T> of(Iterable<T> iterable) {
        Objects.requireNonNull(iterable, "iterable");
        int total = -1;
        if (iterable instanceof Collection) {
            total = ((Collection<?>) iterable).size();
        }
        return new TqdmBar<>(iterable.iterator(), total, "it");
    }

    /** Wrap an iterator with a known total. */
    public static <T> TqdmBar<T> of(Iterator<T> iterator, int total) {
        return new TqdmBar<>(Objects.requireNonNull(iterator, "iterator"), total, "it");
    }

    /** Wrap an iterator with unknown total. */
    public static <T> TqdmBar<T> of(Iterator<T> iterator) {
        return of(iterator, -1);
    }

    /** Convenience for training loops: same as {@link #of(Iterable)}. */
    public static <T> TqdmBar<T> wrap(Iterable<T> iterable) {
        return of(iterable);
    }

    /**
     * Manual progress bar with known total (no underlying iterator).
     * Call {@link TqdmBar#update()} / {@link TqdmBar#update(long)} yourself.
     */
    public static TqdmBar<Void> manual(int total) {
        TqdmBar<Void> bar = new TqdmBar<>(Collections.emptyIterator(), total, "it");
        return bar.setManual(true);
    }

    /** Manual bar with unknown total. */
    public static TqdmBar<Void> manual() {
        return manual(-1);
    }

    /** Stream view of a bar (consumes the bar). */
    public static <T> Stream<T> stream(TqdmBar<T> bar) {
        return StreamSupport.stream(bar.spliterator(), false);
    }

    /**
     * Thread-safe print above any active progress bar (Python {@code tqdm.write}).
     */
    public static void write(String msg) {
        write(msg, defaultOut());
    }

    public static void write(String msg, PrintStream out) {
        PrintStream o = out != null ? out : defaultOut();
        synchronized (WRITE_LOCK) {
            o.print('\r');
            // clear a typical line width
            for (int i = 0; i < 100; i++) {
                o.print(' ');
            }
            o.print('\r');
            o.println(msg == null ? "" : msg);
            o.flush();
        }
    }

    /** Default output stream for bars (stderr, like Python tqdm). */
    public static PrintStream defaultOut() {
        return System.err;
    }

    static Object writeLock() {
        return WRITE_LOCK;
    }
}
