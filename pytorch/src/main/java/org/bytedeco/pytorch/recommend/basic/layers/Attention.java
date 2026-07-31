/*
 * Ported from torch-rechub-scala: torchrec/basic/layers/ActivationUnit.scala (Attention class)
 *
 * Generalized Attention layer. Reference: Alibaba DIN, KDD 2018.
 */
package org.bytedeco.pytorch.recommend.basic.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.recommend.DeviceSupport;

/**
 * Dot-product style attention between query and key-value pairs.
 * Input: query [batch, query_dim], keys [batch, seq_len, key_dim], values [batch, seq_len, val_dim]
 * Output: [batch, output_dim]
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class Attention extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int queryDim;
    private final int keyDim;
    private final int valueDim;
    private final int outputDim;
    private final LinearImpl queryProj;
    private final LinearImpl keyProj;
    private final LinearImpl valueProj;

    public Attention(int queryDim, int keyDim, int valueDim, int outputDim) {
        this(queryDim, keyDim, valueDim, outputDim, DeviceSupport.backend());
    }

    public Attention(int queryDim, int keyDim, int valueDim, int outputDim, String device) {
        super("Attention");
        this.queryDim = queryDim;
        this.keyDim = keyDim;
        this.valueDim = valueDim;
        this.outputDim = outputDim;

        this.queryProj = new LinearImpl(queryDim, outputDim);
        this.keyProj = new LinearImpl(keyDim, outputDim);
        this.valueProj = new LinearImpl(valueDim, outputDim);
        register_module("queryProj", queryProj);
        register_module("keyProj", keyProj);
        register_module("valueProj", valueProj);

        if (device != null && !"cpu".equals(device)) {
            Device dev = new Device(device);
            queryProj.to(dev, false);
            keyProj.to(dev, false);
            valueProj.to(dev, false);
        }
    }

    public Tensor forward(Tensor query, Tensor keys, Tensor values) {
        // query: [batch, query_dim]
        // keys: [batch, seq_len, key_dim]
        // values: [batch, seq_len, val_dim]
        Tensor q = queryProj.forward(query).unsqueeze(1);  // [batch, 1, output_dim]
        Tensor k = keyProj.forward(keys);                   // [batch, seq_len, output_dim]
        Tensor v = valueProj.forward(values);               // [batch, seq_len, output_dim]

        // Scaled dot-product attention
        Tensor scores = q.mul(k).sum(2).unsqueeze(2);       // [batch, seq_len, 1]
        Scalar scale = new Scalar((float) Math.sqrt(outputDim));
        Tensor scaledScores = scores.div(scale);
        Tensor attnWeights = scaledScores.softmax(1);       // [batch, seq_len, 1]

        return v.mul(attnWeights).sum(1);                   // [batch, output_dim]
    }
}
