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
package org.bytedeco.pytorch.llm.text.models;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;

import static org.bytedeco.pytorch.global.torch.relu;

/**
 * Torchtext-style text classification models.
 *
 * <pre>{@code
 * TextModels.TextClassifier clf = TextModels.textClassifier(vocabSize, embedDim, numClasses);
 * Tensor logits = clf.forward(tokenIds); // Long tensor [B, T] or [T]
 * }</pre>
 */
public final class TextModels {

    private TextModels() {}

    /**
     * Bag-of-embeddings text classifier: Embedding → mean-pool → Linear.
     * Input: Long tensor of token ids shaped {@code [batch, seq]} or {@code [seq]}.
     */
    public static final class TextClassifier extends Module {
        final EmbeddingImpl embedding;
        final LinearImpl fc;
        final long embedDim;
        final long numClasses;
        final boolean useRelu;

        public TextClassifier(long vocabSize, long embedDim, long numClasses) {
            this(vocabSize, embedDim, numClasses, true);
        }

        public TextClassifier(long vocabSize, long embedDim, long numClasses, boolean useRelu) {
            super("TextClassifier");
            this.embedDim = embedDim;
            this.numClasses = numClasses;
            this.useRelu = useRelu;
            this.embedding = register_module("embedding", new EmbeddingImpl(vocabSize, embedDim));
            this.fc = register_module("fc", new LinearImpl(embedDim, numClasses));
        }

        @Override
        public Tensor forward(Tensor input) {
            // input: [B, T] or [T]
            Tensor ids = input;
            if (ids.dim() == 1) {
                ids = ids.unsqueeze(0);
            }
            Tensor emb = embedding.forward(ids); // [B, T, E]
            // mean pool over time
            Tensor pooled = emb.mean(1L); // [B, E]
            if (useRelu) {
                pooled = relu(pooled);
            }
            return fc.forward(pooled); // [B, C]
        }

        public EmbeddingImpl embedding() {
            return embedding;
        }

        public LinearImpl fc() {
            return fc;
        }

        public long embedDim() {
            return embedDim;
        }

        public long numClasses() {
            return numClasses;
        }
    }

    /**
     * Multi-layer bag-of-embeddings classifier with a hidden Linear.
     */
    public static final class TextClassifierMLP extends Module {
        final EmbeddingImpl embedding;
        final LinearImpl fc1;
        final LinearImpl fc2;
        final long embedDim;
        final long hidden;
        final long numClasses;

        public TextClassifierMLP(long vocabSize, long embedDim, long hidden, long numClasses) {
            super("TextClassifierMLP");
            this.embedDim = embedDim;
            this.hidden = hidden;
            this.numClasses = numClasses;
            this.embedding = register_module("embedding", new EmbeddingImpl(vocabSize, embedDim));
            this.fc1 = register_module("fc1", new LinearImpl(embedDim, hidden));
            this.fc2 = register_module("fc2", new LinearImpl(hidden, numClasses));
        }

        @Override
        public Tensor forward(Tensor input) {
            Tensor ids = input.dim() == 1 ? input.unsqueeze(0) : input;
            Tensor emb = embedding.forward(ids);
            Tensor pooled = emb.mean(1L);
            Tensor h = relu(fc1.forward(pooled));
            return fc2.forward(h);
        }
    }

    /**
     * Pure linear bag-of-counts classifier (no Embedding table).
     * Input should be a float bag/count vector {@code [batch, vocabSize]}.
     */
    public static final class BagOfWordsClassifier extends Module {
        final LinearImpl fc;
        final long vocabSize;
        final long numClasses;

        public BagOfWordsClassifier(long vocabSize, long numClasses) {
            super("BagOfWordsClassifier");
            this.vocabSize = vocabSize;
            this.numClasses = numClasses;
            this.fc = register_module("fc", new LinearImpl(vocabSize, numClasses));
        }

        @Override
        public Tensor forward(Tensor input) {
            return fc.forward(input);
        }

        /** Build a bag-of-words float vector from token ids. */
        public static Tensor bagVector(int[] tokenIds, int vocabSize) {
            float[] data = new float[vocabSize];
            if (tokenIds != null) {
                for (int id : tokenIds) {
                    if (id >= 0 && id < vocabSize) {
                        data[id] += 1f;
                    }
                }
            }
            return torch.tensor(data);
        }

        public static Tensor bagBatch(int[][] batchIds, int vocabSize) {
            int b = batchIds == null ? 0 : batchIds.length;
            float[] data = new float[b * vocabSize];
            for (int i = 0; i < b; i++) {
                int[] ids = batchIds[i];
                if (ids == null) {
                    continue;
                }
                int base = i * vocabSize;
                for (int id : ids) {
                    if (id >= 0 && id < vocabSize) {
                        data[base + id] += 1f;
                    }
                }
            }
            return torch.tensor(data).reshape(b, vocabSize);
        }
    }

    public static TextClassifier textClassifier(long vocabSize, long embedDim, long numClasses) {
        return new TextClassifier(vocabSize, embedDim, numClasses);
    }

    public static TextClassifierMLP textClassifierMLP(long vocabSize, long embedDim, long hidden, long numClasses) {
        return new TextClassifierMLP(vocabSize, embedDim, hidden, numClasses);
    }

    public static BagOfWordsClassifier bagOfWords(long vocabSize, long numClasses) {
        return new BagOfWordsClassifier(vocabSize, numClasses);
    }
}
