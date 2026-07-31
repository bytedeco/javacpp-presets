/*
 * Ported from torch-rechub-scala: torchrec/basic/layers/FFM.scala
 */
package org.bytedeco.pytorch.recommend.basic.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.recommend.DeviceSupport;

import java.util.ArrayList;
import java.util.List;

/**
 * Field-aware Factorization Machine pairwise field interactions.
 * Input x: (batch, numFields, embedDim) — field-aware embeddings already prepared.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class FFM extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int numFields;
    private final boolean reduceSum;
    private final String device;

    public FFM(int numFields) {
        this(numFields, true, DeviceSupport.backend());
    }

    public FFM(int numFields, boolean reduceSum) {
        this(numFields, reduceSum, DeviceSupport.backend());
    }

    public FFM(int numFields, boolean reduceSum, String device) {
        super("FFM");
        this.numFields = numFields;
        this.reduceSum = reduceSum;
        this.device = device;
    }

    public int numFields() {
        return numFields;
    }

    public boolean reduceSum() {
        return reduceSum;
    }

    @Override
    public Tensor forward(Tensor x) {
        List<Tensor> crossedEmbeddings = new ArrayList<>();

        for (int i = 0; i < numFields - 1; i++) {
            for (int j = i + 1; j < numFields; j++) {
                Tensor vi = x.select(1, i);
                Tensor vj = x.select(1, j);
                crossedEmbeddings.add(vi.mul(vj));
            }
        }

        TensorVector vec = new TensorVector();
        for (Tensor t : crossedEmbeddings) {
            vec.push_back(t);
        }
        // stacked: (num_interactions, batch, embed_dim)
        Tensor stacked = torch.stack(vec);
        // transpose to (batch, num_interactions, embed_dim)
        Tensor transposed = stacked.transpose(0L, 1L);

        if (reduceSum) {
            return transposed.sum(-1);
        }
        return transposed;
    }
}
