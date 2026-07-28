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
package org.bytedeco.pytorch.distributed.examples;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.LongVector;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.LayerNormImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.ReLUImpl;

import java.util.ArrayList;
import java.util.List;

/**
 * Lightweight Transformer-style language model for distributed training demos
 * and benchmarks (Java counterpart of the Python {@code MockLLM} in ddp.md).
 *
 * <p>Architecture: Embedding → N × (Linear + ReLU + LayerNorm residual block)
 * → Linear lm_head. Uses Linear residual blocks instead of full
 * {@code TransformerEncoderLayer} so Mac/CPU smoke tests stay fast and stable
 * while still exercising multi-parameter modules, Embedding, and CE loss.
 *
 * <pre>{@code
 * MockLLM model = MockLLM.tiny(); // hidden=128, layers=2, vocab=1024
 * Tensor logits = model.forward(inputIds); // [B, T, V]
 * }</pre>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class MockLLM extends Module {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private final long hiddenDim;
    private final int numLayers;
    private final long vocabSize;
    private final EmbeddingImpl emb;
    private final List<LinearImpl> layerIns = new ArrayList<>();
    private final List<LinearImpl> layerOuts = new ArrayList<>();
    private final List<LayerNormImpl> norms = new ArrayList<>();
    private final List<ReLUImpl> relus = new ArrayList<>();
    private final LinearImpl lmHead;

    public MockLLM(long hiddenDim, int layers, long vocabSize) {
        super("MockLLM");
        if (hiddenDim <= 0 || layers <= 0 || vocabSize <= 0) {
            throw new IllegalArgumentException("hiddenDim/layers/vocabSize must be > 0");
        }
        this.hiddenDim = hiddenDim;
        this.numLayers = layers;
        this.vocabSize = vocabSize;
        this.emb = register_module("emb", new EmbeddingImpl(vocabSize, hiddenDim));
        for (int i = 0; i < layers; i++) {
            layerIns.add(register_module("lin_in_" + i, new LinearImpl(hiddenDim, hiddenDim * 2)));
            layerOuts.add(register_module("lin_out_" + i, new LinearImpl(hiddenDim * 2, hiddenDim)));
            LongVector lnShape = new LongVector().put(hiddenDim);
            norms.add(register_module("norm_" + i, new LayerNormImpl(lnShape)));
            relus.add(register_module("relu_" + i, new ReLUImpl()));
        }
        this.lmHead = register_module("lm_head", new LinearImpl(hiddenDim, vocabSize));
    }

    /** Mac-friendly defaults: small vocab/hidden/layers for Gloo smoke. */
    public static MockLLM tiny() {
        return new MockLLM(128, 2, 1024);
    }

    public static MockLLM small() {
        return new MockLLM(256, 4, 32000);
    }

    /** Closer to ddp.md defaults (heavier; prefer GPU). */
    public static MockLLM medium() {
        return new MockLLM(1024, 8, 32000);
    }

    public long hiddenDim() { return hiddenDim; }
    public int numLayers() { return numLayers; }
    public long vocabSize() { return vocabSize; }
    public EmbeddingImpl embedding() { return emb; }
    public LinearImpl lmHead() { return lmHead; }

    /**
     * @param inputIds Long tensor of shape {@code [B, T]} (or {@code [T]}).
     * @return Float logits {@code [B, T, V]} (or {@code [T, V]}).
     */
    public Tensor forward(Tensor inputIds) {
        Tensor x = emb.forward(inputIds);
        for (int i = 0; i < numLayers; i++) {
            Tensor h = layerIns.get(i).forward(x);
            h = relus.get(i).forward(h);
            h = layerOuts.get(i).forward(h);
            x = norms.get(i).forward(x.add(h));
        }
        return lmHead.forward(x);
    }

    /** Number of registered parameters (numel sum). */
    public long totalParamNumel() {
        long n = 0;
        var params = parameters();
        for (long i = 0, s = params.size(); i < s; i++) {
            Tensor p = params.get(i);
            if (p != null && !p.isNull()) n += p.numel();
        }
        return n;
    }

    @Override
    public String toString() {
        return "MockLLM{hidden=" + hiddenDim + ", layers=" + numLayers
                + ", vocab=" + vocabSize + ", params=" + totalParamNumel() + '}';
    }
}
