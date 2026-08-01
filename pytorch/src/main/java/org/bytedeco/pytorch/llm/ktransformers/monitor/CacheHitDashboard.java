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
package org.bytedeco.pytorch.llm.ktransformers.monitor;

import org.bytedeco.pytorch.llm.ktransformers.cache.PrefixHitStats;

import java.util.Locale;
import java.util.Map;

/**
 * Publishes three-tier cache hit rates into {@link KtMetrics}.
 */
public final class CacheHitDashboard {

    private final KtMetrics metrics;

    public CacheHitDashboard(KtMetrics metrics) {
        this.metrics = metrics != null ? metrics : new KtMetrics();
    }

    public void update(PrefixHitStats stats) {
        if (stats == null) return;
        Map<String, Double> m = stats.toMetricMap();
        metrics.setAll(m);
    }

    public static String summaryLine(PrefixHitStats s) {
        if (s == null) return "cache: n/a";
        return String.format(Locale.ROOT,
                "cache hit=%.1f%% L1=%.1f%% L2=%.1f%% L3=%.1f%% promote=%d demote=%d",
                100.0 * s.hitRate(),
                100.0 * s.l1HitRate(),
                100.0 * s.l2HitRate(),
                100.0 * s.l3HitRate(),
                s.promotes.sum(),
                s.demotes.sum());
    }

    public KtMetrics metrics() { return metrics; }
}
