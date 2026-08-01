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
package org.bytedeco.pytorch.llm.ktransformers.util;

import java.util.Locale;
import java.util.concurrent.TimeUnit;

/** Lightweight timing helpers for benchmarks and monitor hooks. */
public final class Timing {

    private final long startNs;

    private Timing(long startNs) {
        this.startNs = startNs;
    }

    public static Timing start() {
        return new Timing(System.nanoTime());
    }

    public long elapsedNs() {
        return System.nanoTime() - startNs;
    }

    public double elapsedMs() {
        return elapsedNs() / 1_000_000.0;
    }

    public double elapsedSec() {
        return elapsedNs() / 1_000_000_000.0;
    }

    public static double nsToMs(long ns) {
        return ns / 1_000_000.0;
    }

    public static String formatRate(long count, double seconds) {
        if (seconds <= 0) {
            return "n/a";
        }
        return String.format(Locale.ROOT, "%.2f/s", count / seconds);
    }

    public static void sleepQuietly(long ms) {
        try {
            TimeUnit.MILLISECONDS.sleep(ms);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}
