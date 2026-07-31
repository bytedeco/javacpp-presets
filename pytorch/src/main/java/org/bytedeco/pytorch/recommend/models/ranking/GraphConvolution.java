/*
 * Ported from torch-rechub-scala: torchrec/models/ranking/GNN.scala
 *
 * Graph Neural Network models for risk control and fraud detection.
 * Includes GraphConvolution, GCN, GAT, GraphAttentionLayer, GraphSAGE,
 * SAGEAggregator, FraudGNN.
 */
package org.bytedeco.pytorch.recommend.models.ranking;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.recommend.DeviceSupport;

/** Graph Convolution Layer. */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class GraphConvolution extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int outFeatures;
    private final LinearImpl weight;

    public GraphConvolution(int inFeatures, int outFeatures) {
        this(inFeatures, outFeatures, DeviceSupport.backend());
    }

    public GraphConvolution(int inFeatures, int outFeatures, String device) {
        super("GraphConvolution");
        this.outFeatures = outFeatures;
        this.weight = new LinearImpl(inFeatures, outFeatures);
        register_module("weight", weight);
        if (device != null && !"cpu".equals(device)) {
            weight.to(new Device(device), false);
        }
    }

    public Tensor forward(Tensor input, Tensor adj) {
        Tensor support = weight.forward(input);
        Tensor output = torch.matmul(adj, support);
        Tensor biasTensor = torch.zeros(new long[]{outFeatures},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
                .to(output.device(), ScalarType.Float);
        return output.add(biasTensor);
    }
}
