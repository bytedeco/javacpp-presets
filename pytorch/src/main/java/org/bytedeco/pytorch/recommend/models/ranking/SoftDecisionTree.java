/*
 * Ported from torch-rechub-scala: torchrec/models/ranking/XGBoostModel.scala (SoftDecisionTree)
 *
 * Soft decision tree — simplified routing MLP whose mean logit is the leaf output.
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

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class SoftDecisionTree extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final Device targetDevice;
    private final LinearImpl routeMLP;
    private final Tensor leafValues;

    public SoftDecisionTree(long inputDim, int depth, int numLeaves) {
        this(inputDim, depth, numLeaves, DeviceSupport.backend());
    }

    public SoftDecisionTree(long inputDim, int depth, int numLeaves, String device) {
        super("SoftDecisionTree");
        int numInternalNodes = numLeaves - 1;
        this.targetDevice = new Device(device);

        this.routeMLP = new LinearImpl(inputDim, numInternalNodes);
        routeMLP.to(targetDevice, false);
        register_module("route_mlp", routeMLP);

        // Leaf node parameters (registered, not used in simplified forward)
        this.leafValues = torch.zeros(new long[]{numLeaves},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
        leafValues.to(targetDevice, ScalarType.Float);
        register_parameter("leaf_values", leafValues);
    }

    @Override
    public Tensor forward(Tensor x) {
        Tensor xOnDev = !x.device().equals(targetDevice)
                ? x.to(targetDevice, ScalarType.Float) : x;

        // Routing: pure FC + mean → stable shape [batch, 1]
        Tensor routeLogits = routeMLP.forward(xOnDev);
        Tensor avgLogit = routeLogits.mean(new long[]{1L}, true, new ScalarTypeOptional());

        // Ensure output shape is always [batch, 1]
        return avgLogit.unsqueeze(1).squeeze(2);
    }
}
