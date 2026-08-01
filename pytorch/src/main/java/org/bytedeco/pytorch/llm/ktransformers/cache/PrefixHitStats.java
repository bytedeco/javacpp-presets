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
package org.bytedeco.pytorch.llm.ktransformers.cache;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.atomic.LongAdder;

/**
 * Hit / miss / promote / demote counters for the three-tier prefix cache.
 */
public final class PrefixHitStats {

    public final LongAdder l1Hits = new LongAdder();
    public final LongAdder l2Hits = new LongAdder();
    public final LongAdder l3Hits = new LongAdder();
    public final LongAdder misses = new LongAdder();
    public final LongAdder promotes = new LongAdder();
    public final LongAdder demotes = new LongAdder();
    public final LongAdder bytesPromoted = new LongAdder();
    public final LongAdder bytesDemoted = new LongAdder();
    public final LongAdder lookups = new LongAdder();

    public void recordHit(Tier tier) {
        lookups.increment();
        if (tier == Tier.GPU) l1Hits.increment();
        else if (tier == Tier.CPU) l2Hits.increment();
        else l3Hits.increment();
    }

    public void recordMiss() {
        lookups.increment();
        misses.increment();
    }

    public void recordPromote(long bytes) {
        promotes.increment();
        bytesPromoted.add(Math.max(0L, bytes));
    }

    public void recordDemote(long bytes) {
        demotes.increment();
        bytesDemoted.add(Math.max(0L, bytes));
    }

    public double hitRate() {
        long L = lookups.sum();
        if (L <= 0) return 0.0;
        long hits = l1Hits.sum() + l2Hits.sum() + l3Hits.sum();
        return hits / (double) L;
    }

    public double l1HitRate() {
        long L = lookups.sum();
        return L <= 0 ? 0.0 : l1Hits.sum() / (double) L;
    }

    public double l2HitRate() {
        long L = lookups.sum();
        return L <= 0 ? 0.0 : l2Hits.sum() / (double) L;
    }

    public double l3HitRate() {
        long L = lookups.sum();
        return L <= 0 ? 0.0 : l3Hits.sum() / (double) L;
    }

    public void reset() {
        l1Hits.reset();
        l2Hits.reset();
        l3Hits.reset();
        misses.reset();
        promotes.reset();
        demotes.reset();
        bytesPromoted.reset();
        bytesDemoted.reset();
        lookups.reset();
    }

    public Map<String, Double> toMetricMap() {
        Map<String, Double> m = new LinkedHashMap<>();
        m.put("kt/cache/lookups", (double) lookups.sum());
        m.put("kt/cache/l1_hits", (double) l1Hits.sum());
        m.put("kt/cache/l2_hits", (double) l2Hits.sum());
        m.put("kt/cache/l3_hits", (double) l3Hits.sum());
        m.put("kt/cache/misses", (double) misses.sum());
        m.put("kt/cache/hit_rate", hitRate());
        m.put("kt/cache/l1_hit_rate", l1HitRate());
        m.put("kt/cache/l2_hit_rate", l2HitRate());
        m.put("kt/cache/l3_hit_rate", l3HitRate());
        m.put("kt/cache/promotes", (double) promotes.sum());
        m.put("kt/cache/demotes", (double) demotes.sum());
        m.put("kt/cache/bytes_promoted", (double) bytesPromoted.sum());
        m.put("kt/cache/bytes_demoted", (double) bytesDemoted.sum());
        return m;
    }

    @Override
    public String toString() {
        return String.format(
                "PrefixHitStats{L1=%d L2=%d L3=%d miss=%d hitRate=%.3f promote=%d demote=%d}",
                l1Hits.sum(), l2Hits.sum(), l3Hits.sum(), misses.sum(),
                hitRate(), promotes.sum(), demotes.sum());
    }
}
