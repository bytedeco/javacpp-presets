/*
 * Ported from torch-rechub-scala: torchrec/basic/layers/InteractingLayer.scala
 *
 * Multi-head Self-Attention based Interacting Layer, used in AutoInt.
 * Input/Output: (batch, num_fields, embed_dim)
 */
package org.bytedeco.pytorch.recommend.basic.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.recommend.DeviceSupport;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class InteractingLayer extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int embedDim;
    private final int numHeads;
    private final int headDim;
    private final float scale;
    private final boolean residual;
    private final LinearImpl wQ;
    private final LinearImpl wK;
    private final LinearImpl wV;
    private final LinearImpl wRes;
    private final DropoutImpl dropoutLayer;

    public InteractingLayer(int embedDim) {
        this(embedDim, 2, 0.0f, true, DeviceSupport.backend());
    }

    public InteractingLayer(int embedDim, int numHeads, float dropout, boolean residual, String device) {
        super("InteractingLayer");
        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException(
                    "embed_dim (" + embedDim + ") must be divisible by num_heads (" + numHeads + ")");
        }
        this.embedDim = embedDim;
        this.numHeads = numHeads;
        this.headDim = embedDim / numHeads;
        this.scale = 1.0f / (float) Math.sqrt(headDim);
        this.residual = residual;

        this.wQ = new LinearImpl(embedDim, embedDim);
        register_module("W_Q", wQ);
        wQ.to(new Device(device), false);

        this.wK = new LinearImpl(embedDim, embedDim);
        register_module("W_K", wK);
        wK.to(new Device(device), false);

        this.wV = new LinearImpl(embedDim, embedDim);
        register_module("W_V", wV);
        wV.to(new Device(device), false);

        if (residual) {
            this.wRes = new LinearImpl(embedDim, embedDim);
            register_module("W_Res", wRes);
            wRes.to(new Device(device), false);
        } else {
            this.wRes = null;
        }

        if (dropout > 0) {
            this.dropoutLayer = new DropoutImpl(dropout);
        } else {
            this.dropoutLayer = null;
        }
    }

    @Override
    public Tensor forward(Tensor x) {
        int batchSize = (int) x.size(0);
        int numFields = (int) x.size(1);

        Tensor Q = wQ.forward(x);
        Tensor K = wK.forward(x);
        Tensor V = wV.forward(x);

        // Reshape for multi-head: (batch, num_heads, num_fields, head_dim)
        Tensor qReshaped = Q.reshape(batchSize, numFields, numHeads, headDim).transpose(1, 2);
        Tensor kReshaped = K.reshape(batchSize, numFields, numHeads, headDim).transpose(1, 2);
        Tensor vReshaped = V.reshape(batchSize, numFields, numHeads, headDim).transpose(1, 2);

        Tensor attnScores = torch.matmul(qReshaped, kReshaped.transpose(-2, -1)).mul(new Scalar(scale));
        Tensor attnWeights = attnScores.softmax(-1);

        Tensor finalWeights = dropoutLayer != null ? dropoutLayer.forward(attnWeights) : attnWeights;

        Tensor attnOutput = torch.matmul(finalWeights, vReshaped);
        Tensor attnOutputFinal = attnOutput.transpose(1, 2).contiguous().reshape(batchSize, numFields, embedDim);

        Tensor withResidual = wRes != null ? attnOutputFinal.add(wRes.forward(x)) : attnOutputFinal;
        return torch.relu(withResidual);
    }
}
