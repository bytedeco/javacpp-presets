/*
 * Ported from torch-rechub-scala: torchrec/models/ranking/GNN.scala
 * (SAGEAggregator, GraphSAGE, FraudGNN)
 */
package org.bytedeco.pytorch.utils.recommend.models.ranking;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.ReLUImpl;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;

import java.util.ArrayList;
import java.util.List;

/** SAGE Aggregator (mean aggregation + residual + linear). */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class SAGEAggregator extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final LinearImpl weight;

    public SAGEAggregator(int inFeatures, int outFeatures) {
        this(inFeatures, outFeatures, DeviceSupport.backend());
    }

    public SAGEAggregator(int inFeatures, int outFeatures, String device) {
        super("SAGEAggregator");
        this.weight = new LinearImpl(inFeatures, outFeatures);
        register_module("weight", weight);
        if (device != null && !"cpu".equals(device)) {
            weight.to(new Device(device), false);
        }
    }

    public Tensor forward(Tensor input, Tensor adj) {
        Tensor degree = adj.sum(1).unsqueeze(1);
        Tensor neighborSum = torch.matmul(adj, input);
        Scalar epsilon = new Scalar(1e-9f);
        Tensor aggregated = neighborSum.div(degree.add(epsilon));
        Tensor combined = aggregated.add(input);
        return weight.forward(combined);
    }
}
