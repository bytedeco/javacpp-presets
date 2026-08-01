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

import org.bytedeco.pytorch.llm.ktransformers.config.KtMoEConfig;

/**
 * Logical NUMA partition for expert weight placement.
 *
 * <p>Upstream kt-kernel uses NUMA-aware memory for large MoE on multi-socket
 * hosts. This class provides a deterministic expert→node map without requiring
 * libnuma; optional process affinity is left to the host OS / launcher.
 */
public final class NumaAwarePlacement {

    private final int numaNodes;
    private final int numExperts;
    private final int[] expertToNode;
    private final boolean enabled;

    public NumaAwarePlacement(int numExperts, int numaNodes, boolean enabled) {
        if (numExperts < 1) {
            throw new IllegalArgumentException("numExperts must be >= 1");
        }
        this.numExperts = numExperts;
        this.numaNodes = Math.max(1, numaNodes);
        this.enabled = enabled && this.numaNodes > 1;
        this.expertToNode = new int[numExperts];
        for (int i = 0; i < numExperts; i++) {
            expertToNode[i] = this.enabled ? (i % this.numaNodes) : 0;
        }
    }

    public static NumaAwarePlacement from(KtMoEConfig moe) {
        return new NumaAwarePlacement(moe.numExperts(), moe.numaNodes(), moe.numaAware());
    }

    public static NumaAwarePlacement single() {
        return new NumaAwarePlacement(1, 1, false);
    }

    public int numaNodes() { return numaNodes; }
    public int numExperts() { return numExperts; }
    public boolean enabled() { return enabled; }

    public int nodeForExpert(int expertId) {
        if (expertId < 0 || expertId >= expertToNode.length) {
            return 0;
        }
        return expertToNode[expertId];
    }

    /** Experts assigned to a given logical node. */
    public int[] expertsOnNode(int node) {
        int n = Math.floorMod(node, numaNodes);
        int count = 0;
        for (int v : expertToNode) {
            if (v == n) count++;
        }
        int[] out = new int[count];
        int j = 0;
        for (int i = 0; i < expertToNode.length; i++) {
            if (expertToNode[i] == n) {
                out[j++] = i;
            }
        }
        return out;
    }

    public int[] expertToNodeMap() {
        return expertToNode.clone();
    }
}
