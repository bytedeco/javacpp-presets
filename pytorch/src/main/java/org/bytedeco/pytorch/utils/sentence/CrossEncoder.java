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
package org.bytedeco.pytorch.utils.sentence;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.utils.tokenizers.FastTokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.matmul;
import static org.bytedeco.pytorch.global.torch.sigmoid;
import static org.bytedeco.pytorch.global.torch.ScalarType;
import static org.bytedeco.pytorch.global.torch.stack;

/**
 * Cross-Encoder: concatenates two sentence embeddings → MLP → score (regression / classification).
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class CrossEncoder extends Module {

    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private final FastTokenizer tokenizer;
    private final LinearImpl mlp1;
    private final LinearImpl mlp2;
    private final int embedDim;
    private final boolean isBinary;

    public CrossEncoder(int embedDim, int hiddenDim, boolean isBinary) {
        super("CrossEncoder");
        this.embedDim = embedDim;
        this.isBinary = isBinary;
        this.tokenizer = FastTokenizer.whitespace().modelMaxLength(64).build();
        this.mlp1 = register_module("mlp1", new LinearImpl(embedDim * 2, hiddenDim));
        this.mlp2 = register_module("mlp2", new LinearImpl(hiddenDim, isBinary ? 1 : 1));
    }

    public CrossEncoder(int embedDim) {
        this(embedDim, embedDim * 2, true);
    }

    public CrossEncoder() {
        this(64);
    }

    public Tensor score(Tensor embA, Tensor embB) {
        org.bytedeco.pytorch.TensorVector tv = new org.bytedeco.pytorch.TensorVector(2);
        tv.put(0, embA);
        tv.put(1, embB);
        Tensor cat = stack(tv, 1L).squeeze(2);
        Tensor h = mlp1.forward(cat).relu();
        return mlp2.forward(h);
    }

    public Tensor predict(String a, String b) {
        float[] e1 = encode(a);
        float[] e2 = encode(b);
        Tensor ta = tensorFromRow(e1);
        Tensor tb = tensorFromRow(e2);
        return score(ta, tb);
    }

    public List<Double> predict(List<String[]> pairs) {
        List<Double> out = new ArrayList<>(pairs.size());
        for (String[] p : pairs) {
            Tensor s = predict(p[0], p[1]);
            out.add(isBinary ? sigmoid(s).item_double() : s.item_double());
        }
        return out;
    }

    private float[] encode(String text) {
        var enc = tokenizer.encode(text == null ? "" : text, true);
        int[] ids = enc.ids();
        long vocab = 1024;
        float[] emb = new float[embedDim];
        for (int i = 0; i < Math.min(ids.length, embedDim); i++) {
            emb[i] = (float) (ids[i] % vocab) / vocab;
        }
        return emb;
    }

    private Tensor tensorFromRow(float[] row) {
        Tensor t = org.bytedeco.pytorch.global.torch.tensor(row).unsqueeze(0);
        return t.to(ScalarType.Float);
    }

    @Override
    public Tensor forward(Tensor a, Tensor b) {
        return score(a, b);
    }
}
