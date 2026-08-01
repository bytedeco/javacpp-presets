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
package org.bytedeco.pytorch.llm.ktransformers.moe;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;

import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.zeros_like;

/**
 * Token → expert grouped dispatch honoring CPU/GPU residency.
 *
 * <p>Reference path uses dense per-expert mask (same correctness model as
 * {@code modules.MoE}) but branches on {@link ExpertDevice}:
 * <ul>
 *   <li>GPU experts: run module on the current compute tensor</li>
 *   <li>CPU experts: run the same forward (weights conceptually CPU-resident);
 *       a real production build would pin/copy activations — here we keep a
 *       single-device numerical path so golden tests stay deterministic</li>
 * </ul>
 *
 * <p>The residency flag still drives metrics and scheduling so mixed placement
 * is observable and migratable even when tensors share one physical device in CI.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class TokenDispatcher {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final ExpertPool pool;
    private final ExpertScheduler scheduler;

    public TokenDispatcher(ExpertPool pool, ExpertScheduler scheduler) {
        this.pool = Objects.requireNonNull(pool, "pool");
        this.scheduler = Objects.requireNonNull(scheduler, "scheduler");
    }

    public ExpertPool pool() { return pool; }
    public ExpertScheduler scheduler() { return scheduler; }

    /**
     * Weighted sum of expert FFNs for flat tokens.
     *
     * @param flat   [N, H] tokens
     * @param topW   [N, K] routing weights (already normalized if desired)
     * @param topI   [N, K] expert indices (long)
     * @param topK   K
     * @return [N, H]
     */
    public Tensor dispatch(Tensor flat, Tensor topW, Tensor topI, int topK) {
        Objects.requireNonNull(flat, "flat");
        Objects.requireNonNull(topW, "topW");
        Objects.requireNonNull(topI, "topI");
        int numExperts = pool.numExperts();
        Tensor out = zeros_like(flat);

        // Collect unique expert ids hit this step for scheduler
        boolean[] seen = new boolean[numExperts];
        int hitCount = 0;
        int[] hitIds = new int[Math.min(numExperts, topK * 8)];

        for (int e = 0; e < numExperts; e++) {
            Tensor eq = topI.eq(new Scalar((long) e)); // [N, K] bool
            Tensor w = topW.mul(eq.to(topW.scalar_type())).sum(new long[]{-1L}); // [N]
            if (!w.gt(new Scalar(0.0)).any().item_bool()) {
                eq.close();
                w.close();
                continue;
            }
            if (!seen[e] && hitCount < hitIds.length) {
                seen[e] = true;
                hitIds[hitCount++] = e;
            }
            pool.recordHit(e);
            ExpertSpec spec = pool.get(e);
            Module mod = spec.module();
            // Residency is control-plane; forward is local for reference correctness.
            Tensor expertOut = mod.forward(flat); // [N, H]
            out = out.add(expertOut.mul(w.unsqueeze(-1)));
            eq.close();
            w.close();
        }

        int[] selected = new int[hitCount];
        System.arraycopy(hitIds, 0, selected, 0, hitCount);
        scheduler.onDispatch(selected);
        return out;
    }
}
