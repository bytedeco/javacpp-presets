/*
 * Ported from torch-rechub-scala: torchrec/basic/layers/CINFixed.scala
 *
 * Fixed Compressed Interaction Network from xDeepFM.
 * Uses Linear layers instead of Conv1d; all layers registered in constructor.
 */
package org.bytedeco.pytorch.utils.recommend.basic.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;

import java.util.ArrayList;
import java.util.List;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class CINFixed extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int numFields;
    private final int embedDim;
    private final int[] crossLayerSizes;
    private final boolean splitHalf;
    private final List<LinearImpl> convLayers = new ArrayList<>();
    private final List<LinearImpl> projLayers = new ArrayList<>();

    public CINFixed(int numFields, int embedDim, int[] crossLayerSizes) {
        this(numFields, embedDim, crossLayerSizes, true, DeviceSupport.backend());
    }

    public CINFixed(int numFields, int embedDim, int[] crossLayerSizes,
                    boolean splitHalf, String device) {
        super("CINFixed");
        if (crossLayerSizes == null || crossLayerSizes.length == 0) {
            throw new IllegalArgumentException("crossLayerSizes cannot be empty");
        }
        if (embedDim <= 0) {
            throw new IllegalArgumentException("embedDim must be positive");
        }
        if (numFields <= 0) {
            throw new IllegalArgumentException("numFields must be positive");
        }
        this.numFields = numFields;
        this.embedDim = embedDim;
        this.crossLayerSizes = crossLayerSizes.clone();
        this.splitHalf = splitHalf;

        for (int i = 0; i < crossLayerSizes.length; i++) {
            int outDim = crossLayerSizes[i];
            int inChannels = numFields * numFields;
            LinearImpl conv = new LinearImpl(inChannels, outDim);
            register_module("conv_" + i, conv);
            conv.to(new Device(device), false);
            convLayers.add(conv);

            if (splitHalf && i < crossLayerSizes.length - 1) {
                int projInDim = Math.max(1, (int) Math.floor(outDim / 2.0));
                LinearImpl proj = new LinearImpl(projInDim, numFields * embedDim);
                register_module("proj_" + i, proj);
                proj.to(new Device(device), false);
                projLayers.add(proj);
            }
        }
    }

    @Override
    public Tensor forward(Tensor embeddings) {
        // embeddings: (batch, num_fields, embed_dim)
        long batchSize = embeddings.size(0);

        Tensor x0 = embeddings;
        Tensor hk = x0.transpose(1, 2);  // (batch, embed_dim, num_fields)

        List<Tensor> outputs = new ArrayList<>();
        int projLayerIdx = 0;

        for (int i = 0; i < crossLayerSizes.length; i++) {
            // Outer product via bmm: (batch, F, F)
            Tensor xh = torch.bmm(x0, hk);
            Tensor flat = xh.view(batchSize, (long) numFields * numFields);

            Tensor convOut = convLayers.get(i).forward(flat);
            convOut = convOut.relu();

            Tensor pooled = convOut.sum(1).unsqueeze(1);  // (batch, 1)
            outputs.add(pooled);

            if (splitHalf && i < crossLayerSizes.length - 1) {
                int actualNextDim = Math.max(1, (int) Math.floor(convOut.size(1) / 2.0));
                Tensor half = convOut.narrow(1, 0, actualNextDim);

                Tensor projOut = projLayers.get(projLayerIdx).forward(half);
                projLayerIdx++;

                hk = projOut.view(batchSize, numFields, embedDim).transpose(1, 2);
            }
        }

        TensorVector tensorVec = new TensorVector();
        for (Tensor t : outputs) tensorVec.push_back(t);
        Tensor cinOut = torch.cat(tensorVec, 1L);  // (batch, num_layers)
        return cinOut.sum(1).unsqueeze(1);  // (batch, 1)
    }
}
