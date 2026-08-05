/*
 * Ported from torch-rechub-scala: torchrec/basic/layers/RegulationModule.scala
 *
 * Regulation Module for EDCN. Reference: EDCN paper, KDD 2021.
 */
package org.bytedeco.pytorch.recommend.basic.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.T_TensorTensor_T;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.recommend.DeviceSupport;

import java.util.ArrayList;
import java.util.List;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class RegulationModule extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int numFields;
    private final int[] feaDims;
    private final float tau;
    private final boolean useRegulation;
    private final Tensor g1;
    private final Tensor g2;

    public RegulationModule(int numFields, int[] feaDims) {
        this(numFields, feaDims, 1.0f, true);
    }

    public RegulationModule(int numFields, int[] feaDims, float tau, boolean useRegulation) {
        super("RegulationModule");
        this.numFields = numFields;
        this.feaDims = feaDims != null ? feaDims.clone() : new int[0];
        this.tau = tau;
        this.useRegulation = useRegulation;

        if (useRegulation) {
            if (this.feaDims.length != numFields) {
                throw new IllegalArgumentException(
                        "feaDims size " + this.feaDims.length + " must match numFields " + numFields);
            }
        }

        TensorOptions opts = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float));
        if (useRegulation) {
            this.g1 = torch.ones(new long[]{numFields}, opts);
            this.g2 = torch.ones(new long[]{numFields}, opts);
            register_parameter("g1", g1);
            register_parameter("g2", g2);
        } else {
            this.g1 = torch.zeros(new long[]{numFields}, opts);
            this.g2 = torch.zeros(new long[]{numFields}, opts);
        }

        String device = DeviceSupport.backend();
        if (!"cpu".equals(device)) {
            this.to(new Device(device), false);
        }
    }

    /**
     * Forward pass of RegulationModule.
     *
     * @param x Input tensor (B, total_dim) where total_dim = sum(feaDims)
     * @return (out1, out2) tuple of gated tensors (B, total_dim)
     */
    public T_TensorTensor_T forwardReg(Tensor x) {
        return forwardReg(x, false);
    }

    @Override
    public T_TensorTensor_T forwardT_TensorTensor_T(Tensor x) {
        return forwardReg(x, false);
    }

    public T_TensorTensor_T forwardReg(Tensor x, boolean r) {
        if (!useRegulation) {
            return new T_TensorTensor_T(x, x);
        }

        Scalar tauTensor = new Scalar(tau);
        Tensor g1Scaled = g1.div(tauTensor).softmax(0);
        Tensor g2Scaled = g2.div(tauTensor).softmax(0);

        List<Tensor> g1List = new ArrayList<>();
        List<Tensor> g2List = new ArrayList<>();
        for (int i = 0; i < numFields; i++) {
            Tensor fieldG1 = g1Scaled.select(0, i).unsqueeze(0);
            Tensor fieldG2 = g2Scaled.select(0, i).unsqueeze(0);
            g1List.add(fieldG1.expand(1, feaDims[i]));
            g2List.add(fieldG2.expand(1, feaDims[i]));
        }

        TensorVector v1 = new TensorVector();
        TensorVector v2 = new TensorVector();
        for (Tensor t : g1List) v1.push_back(t);
        for (Tensor t : g2List) v2.push_back(t);

        Tensor g1Tensor = torch.cat(v1, 1);
        Tensor g2Tensor = torch.cat(v2, 1);

        long batchSize = x.size(0);
        Tensor g1Broadcast = g1Tensor.expand(batchSize, -1).to(x.device(), ScalarType.Float);
        Tensor g2Broadcast = g2Tensor.expand(batchSize, -1).to(x.device(), ScalarType.Float);

        Tensor out1 = g1Broadcast.mul(x);
        Tensor out2 = g2Broadcast.mul(x);
        return new T_TensorTensor_T(out1, out2);
    }
}
