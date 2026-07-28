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
package org.bytedeco.pytorch.llm.sentence;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.pytorch.LongVector;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.LayerNormImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.llm.tokenizers.Encoding;
import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.ScalarType;
import static org.bytedeco.pytorch.global.torch.full;
import static org.bytedeco.pytorch.global.torch.ones_like;
import static org.bytedeco.pytorch.global.torch.tensor;

/**
 * Sentence-Transformers style text embedding model (pure Java / libtorch).
 *
 * <p>Mini encoder: Embedding → mean-pool (attention-masked) → Linear projection
 * → L2 normalize. Mirrors the common {@code SentenceTransformer.encode} API.
 *
 * <pre>{@code
 * SentenceTransformer st = SentenceTransformer.mini(256, 64);
 * float[][] emb = st.encode(List.of("hello world", "bonjour"));
 * double sim = SentenceTransformer.cosine(emb[0], emb[1]);
 * }</pre>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class SentenceTransformer extends Module {

    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private final FastTokenizer tokenizer;
    private final EmbeddingImpl embedding;
    private final LinearImpl projection;
    private final LayerNormImpl norm;
    private final int embedDim;
    private final int maxSeqLength;
    private final boolean normalizeEmbeddings;
    private final PoolingStrategy pooling;

    public enum PoolingStrategy { MEAN, CLS, MAX }

    public SentenceTransformer(FastTokenizer tokenizer, int vocabSize, int hiddenSize,
                               int embedDim, int maxSeqLength,
                               boolean normalizeEmbeddings, PoolingStrategy pooling) {
        super("SentenceTransformer");
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        this.embedDim = embedDim;
        this.maxSeqLength = maxSeqLength;
        this.normalizeEmbeddings = normalizeEmbeddings;
        this.pooling = pooling == null ? PoolingStrategy.MEAN : pooling;
        this.embedding = register_module("emb", new EmbeddingImpl(vocabSize, hiddenSize));
        // LongVector(long) is a SIZE ctor (n zeros) — put the single normalized dim.
        LongVector shape = new LongVector().put((long) hiddenSize);
        this.norm = register_module("norm", new LayerNormImpl(shape));
        this.projection = register_module("proj", new LinearImpl(hiddenSize, embedDim));
    }

    public static SentenceTransformer mini(int vocabHint, int embedDim) {
        FastTokenizer tok = FastTokenizer.whitespace()
                .modelMaxLength(64)
                .build();
        int vocab = Math.max(Math.max(vocabHint, tok.vocabSize() + 64), 1024);
        return new SentenceTransformer(tok, vocab, Math.max(embedDim, 32),
                embedDim, 64, true, PoolingStrategy.MEAN);
    }

    public static SentenceTransformer mini() {
        return mini(512, 64);
    }

    public int getEmbedDim() { return embedDim; }
    public int getMaxSeqLength() { return maxSeqLength; }
    public FastTokenizer tokenizer() { return tokenizer; }
    public boolean isNormalizeEmbeddings() { return normalizeEmbeddings; }

    public float[][] encode(List<String> sentences) {
        return encode(sentences, true);
    }

    public float[][] encode(List<String> sentences, boolean normalize) {
        Objects.requireNonNull(sentences, "sentences");
        if (sentences.isEmpty()) return new float[0][];
        this.eval();
        List<Encoding> encs = new ArrayList<>(sentences.size());
        int maxLen = 0;
        for (String s : sentences) {
            Encoding e = tokenizer.encode(s == null ? "" : s, true);
            if (e.size() > maxSeqLength) e = e.truncate(maxSeqLength);
            encs.add(e);
            maxLen = Math.max(maxLen, e.size());
        }
        maxLen = Math.max(1, maxLen);
        long B = sentences.size();
        long[] flatIds = new long[(int) (B * maxLen)];
        float[] flatMask = new float[(int) (B * maxLen)];
        Arrays.fill(flatIds, tokenizer.padId());
        long vocabN = embedding.weight().size(0);
        for (int i = 0; i < encs.size(); i++) {
            int[] ids = encs.get(i).ids();
            int[] mask = encs.get(i).attentionMask();
            for (int j = 0; j < ids.length && j < maxLen; j++) {
                long id = Math.floorMod(ids[j], (int) vocabN);
                flatIds[i * maxLen + j] = id;
                flatMask[i * maxLen + j] = j < mask.length ? mask[j] : 1f;
            }
        }
        Tensor inputIds = tensor(flatIds).reshape(B, maxLen);
        Tensor attn = tensor(flatMask).reshape(B, maxLen);
        Tensor emb = forward(inputIds, attn);
        if (normalize || normalizeEmbeddings) {
            emb = l2Normalize(emb);
        }
        return toRows(emb);
    }

    public float[] encode(String sentence) {
        float[][] b = encode(List.of(sentence));
        return b.length == 0 ? new float[embedDim] : b[0];
    }

    public Tensor forward(Tensor inputIds, Tensor attentionMask) {
        Tensor ids = inputIds;
        if (ids.dim() == 1) ids = ids.unsqueeze(0);
        Tensor mask = attentionMask;
        if (mask != null && mask.dim() == 1) mask = mask.unsqueeze(0);

        Tensor hidden = embedding.forward(ids);
        hidden = norm.forward(hidden);
        Tensor pooled = pool(hidden, mask);
        return projection.forward(pooled);
    }

    @Override
    public Tensor forward(Tensor inputIds) {
        Tensor mask = ones_like(inputIds).to(ScalarType.Float);
        return forward(inputIds, mask);
    }

    private Tensor pool(Tensor hidden, Tensor mask) {
        long T = hidden.size(1);
        if (pooling == PoolingStrategy.CLS) {
            return hidden.slice(1, new org.bytedeco.pytorch.LongOptional(0),
                    new org.bytedeco.pytorch.LongOptional(1), 1).squeeze(1);
        }
        if (pooling == PoolingStrategy.MAX) {
            if (mask == null) {
                return hidden.max(1L).get0();
            }
            Tensor m = mask.unsqueeze(-1);
            Tensor neg = full(new long[]{hidden.size(0), T, hidden.size(2)}, new Scalar(-1e9f));
            Tensor ones = full(new long[]{hidden.size(0), T, 1}, new Scalar(1.0f));
            Tensor masked = hidden.mul(m).add(neg.mul(ones.sub(m)));
            return masked.max(1L).get0();
        }
        // MEAN
        if (mask == null) {
            return hidden.mean(new long[]{1L});
        }
        Tensor m = mask.unsqueeze(-1).to(ScalarType.Float);
        Tensor summed = hidden.mul(m).sum(new long[]{1L});
        Tensor counts = m.sum(new long[]{1L}).clamp_min(new Scalar(1e-9));
        return summed.div(counts);
    }

    private static Tensor l2Normalize(Tensor x) {
        // norm over dim 1 with keepdim
        Tensor n = x.norm(
                new org.bytedeco.pytorch.ScalarOptional(new Scalar(2.0)),
                new long[]{1L},
                true);
        return x.div(n.clamp_min(new Scalar(1e-12)));
    }

    private static float[][] toRows(Tensor t) {
        Tensor f = t.to(ScalarType.Float).contiguous();
        long B = f.size(0);
        long D = f.size(1);
        float[] flat = new float[(int) (B * D)];
        FloatIndexer idx = f.reshape(-1).createIndexer();
        try {
            for (long i = 0; i < flat.length; i++) {
                flat[(int) i] = idx.get(i);
            }
        } finally {
            idx.release();
        }
        float[][] rows = new float[(int) B][(int) D];
        for (int i = 0; i < B; i++) {
            System.arraycopy(flat, i * (int) D, rows[i], 0, (int) D);
        }
        return rows;
    }

    public static double cosine(float[] a, float[] b) {
        Objects.requireNonNull(a);
        Objects.requireNonNull(b);
        int n = Math.min(a.length, b.length);
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < n; i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        double denom = Math.sqrt(na) * Math.sqrt(nb);
        return denom < 1e-12 ? 0.0 : dot / denom;
    }

    public static double[][] cosineMatrix(float[][] emb) {
        int n = emb.length;
        double[][] m = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                m[i][j] = cosine(emb[i], emb[j]);
            }
        }
        return m;
    }

    public static final class SearchHit {
        public final int index;
        public final double score;
        public final String text;
        public SearchHit(int index, double score, String text) {
            this.index = index;
            this.score = score;
            this.text = text;
        }
        @Override public String toString() {
            return "SearchHit{i=" + index + ", score=" + score + ", text='" + text + "'}";
        }
    }

    public List<SearchHit> semanticSearch(String query, List<String> corpus, int topK) {
        float[] q = encode(query);
        float[][] c = encode(corpus);
        List<SearchHit> hits = new ArrayList<>(corpus.size());
        for (int i = 0; i < c.length; i++) {
            hits.add(new SearchHit(i, cosine(q, c[i]), corpus.get(i)));
        }
        hits.sort((a, b) -> Double.compare(b.score, a.score));
        if (topK > 0 && hits.size() > topK) {
            return new ArrayList<>(hits.subList(0, topK));
        }
        return hits;
    }
}
