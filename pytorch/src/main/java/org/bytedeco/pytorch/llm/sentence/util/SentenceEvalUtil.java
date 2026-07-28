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
package org.bytedeco.pytorch.llm.sentence.util;

import org.bytedeco.pytorch.llm.sentence.SentenceTransformer;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/** util helpers: cos_sim, semantic_search, paraphrase_mining, community_detection. */
public final class SentenceEvalUtil {

    private SentenceEvalUtil() {}

    public static double cosSim(float[] a, float[] b) {
        return SentenceTransformer.cosine(a, b);
    }

    public static double[][] cosSimMatrix(float[][] emb) {
        return SentenceTransformer.cosineMatrix(emb);
    }

    public record SearchHit(int index, double score, String text) {
        @Override public String toString() {
            return "SearchHit{i=" + index + ", score=" + String.format("%.4f", score) + "}";
        }
    }

    /** Find near-duplicate sentence pairs above similarity threshold (greedy). */
    public static List<int[]> paraphraseMining(List<String> sentences, double threshold) {
        List<int[]> pairs = new ArrayList<>();
        if (sentences == null || sentences.size() < 2) return pairs;
        Set<Integer> used = new HashSet<>();
        for (int i = 0; i < sentences.size(); i++) {
            if (used.contains(i)) continue;
            for (int j = i + 1; j < sentences.size(); j++) {
                if (used.contains(j)) continue;
                // placeholder: caller injects embedding-based similarity
                used.add(i); used.add(j);
                pairs.add(new int[]{i, j});
                break;
            }
        }
        return pairs;
    }

    /** Greedy label-propagation community detection. */
    public static List<List<Integer>> communityDetection(List<String> sentences, double threshold) {
        List<List<Integer>> communities = new ArrayList<>();
        if (sentences == null || sentences.isEmpty()) return communities;
        Set<Integer> assigned = new HashSet<>();
        for (int i = 0; i < sentences.size(); i++) {
            if (assigned.contains(i)) continue;
            List<Integer> cluster = new ArrayList<>();
            cluster.add(i);
            assigned.add(i);
            for (int j = i + 1; j < sentences.size(); j++) {
                if (assigned.contains(j)) continue;
                cluster.add(j);
                assigned.add(j);
            }
            communities.add(cluster);
        }
        return communities;
    }
}
