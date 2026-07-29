/*
 * Ported from torch-rechub-scala: torchrec/models/knowledge_tracing/layers/PositionalEmbedding.scala
 *
 * Cosine positional embedding used by AKT, SimpleKT, SparseKT, SAKT.
 */
package org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.TensorHelpers;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class CosinePositionalEmbedding extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int embedDim;
    private final int maxLen;
    private final Tensor pe;

    public CosinePositionalEmbedding(int embedDim) {
        this(embedDim, 512, DeviceSupport.backend());
    }

    public CosinePositionalEmbedding(int embedDim, int maxLen, String device) {
        super("CosinePositionalEmbedding");
        this.embedDim = embedDim;
        this.maxLen = maxLen;

        // Match Scala:
        //   position = arange(0, maxLen).unsqueeze(1)
        //   divTerm  = exp(arange(0, embedDim, 2) * -log(10000) / embedDim)
        //   pe[:, 0::2] = sin(pos * divTerm); pe[:, 1::2] = cos(pos * divTerm)
        float[] peArr = new float[maxLen * embedDim];
        for (int i = 0; i < maxLen; i++) {
            for (int j = 0; j < embedDim; j++) {
                int idx = i * embedDim + j;
                int k = j / 2;
                double arangeVal = k * 2.0; // arange(0, embedDim, 2) → 0, 2, 4, ...
                double dt = Math.exp(arangeVal * (-Math.log(10000.0)) / embedDim);
                if (j % 2 == 0) {
                    peArr[idx] = (float) Math.sin(i * dt);
                } else {
                    peArr[idx] = (float) Math.cos(i * dt);
                }
            }
        }

        Tensor peBuf = torch.zeros(
                new long[]{1L, maxLen, embedDim},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
        peBuf.copy_(TensorHelpers.tensor(peArr, new long[]{1L, maxLen, embedDim}));
        if (!"cpu".equals(device)) {
            peBuf = peBuf.to(new Device(device), ScalarType.Float);
        }
        this.pe = peBuf;
        register_buffer("pe", pe);
    }

    @Override
    public Tensor forward(Tensor x) {
        int batchSize = (int) x.size(0);
        int seqLen = (int) x.size(1);
        int actualLen = Math.min(seqLen, maxLen);
        return pe.narrow(1, 0, actualLen)
                .expand(new long[]{batchSize, actualLen, embedDim})
                .clone();
    }
}
