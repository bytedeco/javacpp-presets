/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.utils.sentence.evaluation;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.utils.sentence.SentenceTransformer;

import java.util.List;

/**
 * Spearman / Pearson correlation on sentence-pair similarity scores.
 */
public final class EmbeddingSimilarityEvaluator {

    public record Result(double spearman, double pearson) {
        public double mean() { return (spearman + pearson) / 2.0; }
        public String toString() {
            return String.format("EmbeddingSimilarityEvaluator{rho=%.4f, r=%.4f}", spearman, pearson);
        }
    }

    public static Result evaluate(List<String[]> pairs, List<Double> scores,
                                  SentenceTransformer st) {
        int n = Math.min(pairs.size(), scores.size());
        if (n == 0) return new Result(0, 0);
        double[] sims = new double[n];
        for (int i = 0; i < n; i++) {
            String[] p = pairs.get(i);
            float[] e1 = st.encode(p[0]);
            float[] e2 = st.encode(p[1]);
            sims[i] = SentenceTransformer.cosine(e1, e2);
        }
        double rho = spearmanRho(sims, scores.stream().mapToDouble(Double::doubleValue).toArray());
        double r = pearsonR(simScoresToTensor(sims), simScoresToTensor(scores.stream().mapToDouble(Double::doubleValue).toArray()));
        return new Result(rho, r);
    }

    private static double spearmanRho(double[] a, double[] b) {
        int n = Math.min(a.length, b.length);
        double[] ra = rank(a, n);
        double[] rb = rank(b, n);
        double num = 0, da = 0, db = 0;
        for (int i = 0; i < n; i++) {
            double d = ra[i] - rb[i];
            num += d * d;
            double ad = ra[i] - (n + 1) / 2.0;
            double bd = rb[i] - (n + 1) / 2.0;
            da += ad * ad;
            db += bd * bd;
        }
        double denom = Math.sqrt(da * db);
        return denom < 1e-12 ? 0 : 1.0 - 6.0 * num / (n * (n * n - 1.0));
    }

    private static double[] rank(double[] v, int n) {
        Integer[] idx = new Integer[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        java.util.Arrays.sort(idx, (i, j) -> Double.compare(v[i], v[j]));
        double[] r = new double[n];
        for (int i = 0; i < n; i++) r[idx[i]] = i + 1;
        return r;
    }

    private static double pearsonR(Tensor a, Tensor b) {
        return org.bytedeco.pytorch.global.torch.cosine_similarity(a, b, 0L, 1e-8).item_double();
    }

    private static Tensor simScoresToTensor(double[] v) {
        float[] f = new float[v.length];
        for (int i = 0; i < v.length; i++) f[i] = (float) v[i];
        return org.bytedeco.pytorch.global.torch.tensor(f);
    }
}
