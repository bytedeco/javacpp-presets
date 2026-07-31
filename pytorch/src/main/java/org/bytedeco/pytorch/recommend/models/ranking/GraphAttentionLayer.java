/*
 * Ported from torch-rechub-scala: torchrec/models/ranking/GNN.scala (GraphAttentionLayer + GAT)
 */
package org.bytedeco.pytorch.recommend.models.ranking;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.recommend.DeviceSupport;

import java.util.ArrayList;
import java.util.List;

/** Graph Attention Layer. */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class GraphAttentionLayer extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int headDim;
    private final int numHeads;
    private final List<LinearImpl> heads = new ArrayList<>();
    private final List<LinearImpl> attention = new ArrayList<>();
    private final DropoutImpl dropoutLayer;

    public GraphAttentionLayer(int inFeatures, int outFeatures) {
        this(inFeatures, outFeatures, 8, 0.5f, DeviceSupport.backend());
    }

    public GraphAttentionLayer(int inFeatures, int outFeatures, int numHeads,
                               float dropout, String device) {
        super("GraphAttentionLayer");
        this.numHeads = numHeads;
        this.headDim = outFeatures / numHeads;

        for (int i = 0; i < numHeads; i++) {
            LinearImpl w = new LinearImpl(inFeatures, headDim);
            register_module("head_" + i, w);
            heads.add(w);
        }
        for (int i = 0; i < numHeads; i++) {
            LinearImpl a = new LinearImpl(headDim * 2L, 1);
            register_module("attn_" + i, a);
            attention.add(a);
        }
        this.dropoutLayer = new DropoutImpl(dropout);

        if (device != null && !"cpu".equals(device)) {
            Device dev = new Device(device);
            for (LinearImpl h : heads) h.to(dev, false);
            for (LinearImpl a : attention) a.to(dev, false);
        }
    }

    public Tensor forward(Tensor input, Tensor adj) {
        long numNodes = input.size(0);
        List<Tensor> headOutputs = new ArrayList<>();

        for (int i = 0; i < numHeads; i++) {
            Tensor h = heads.get(i).forward(input);

            Tensor hI = h.unsqueeze(1).expand(numNodes, numNodes, headDim);
            Tensor hJ = h.unsqueeze(0).expand(numNodes, numNodes, headDim);
            TensorVector cVec = new TensorVector();
            cVec.push_back(hI);
            cVec.push_back(hJ);
            Tensor hConcat = torch.cat(cVec, 2);

            Tensor e = attention.get(i).forward(hConcat.view(numNodes * numNodes, headDim * 2L))
                    .view(numNodes, numNodes);

            Scalar negInf = new Scalar(-1e9f);
            Tensor masked = torch.where(adj.gt(new Scalar(0.5f)), e, negInf);
            Tensor alpha = torch.softmax(masked, 1);
            headOutputs.add(torch.matmul(alpha, h));
        }

        TensorVector outVec = new TensorVector();
        for (Tensor t : headOutputs) outVec.push_back(t.contiguous());
        Tensor concatenated = torch.cat(outVec, 1);
        return dropoutLayer.forward(concatenated);
    }
}
