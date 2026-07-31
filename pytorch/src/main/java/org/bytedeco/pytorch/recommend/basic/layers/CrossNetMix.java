/*
 * Ported from torch-rechub-scala: torchrec/basic/layers/CrossNetwork.scala (CrossNetMix)
 *
 * CrossNetMix from DCN v2 — Mixture-of-Experts cross network with low-rank
 * nonlinear projections and a softmax gating network per expert.
 */
package org.bytedeco.pytorch.recommend.basic.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.LinearOptions;
import org.bytedeco.pytorch.recommend.DeviceSupport;

import java.util.ArrayList;
import java.util.List;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class CrossNetMix extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int numLayers;
    private final int numExperts;
    private final List<Tensor> uList = new ArrayList<>();
    private final List<Tensor> vList = new ArrayList<>();
    private final List<Tensor> cList = new ArrayList<>();
    private final List<Tensor> biasList = new ArrayList<>();
    private final List<LinearImpl> gating = new ArrayList<>();

    public CrossNetMix(long inputDim) {
        this(inputDim, 3, 32, 4, DeviceSupport.backend());
    }

    public CrossNetMix(long inputDim, int numLayers, int lowRank, int numExperts, String device) {
        super("CrossNetMix");
        this.numLayers = numLayers;
        this.numExperts = numExperts;
        Device dev = new Device(device);

        for (int i = 0; i < numLayers; i++) {
            Tensor u = makeParam(new long[]{numExperts, inputDim, lowRank}, dev);
            register_parameter("U_" + i, u);
            uList.add(u);

            Tensor v = makeParam(new long[]{numExperts, inputDim, lowRank}, dev);
            register_parameter("V_" + i, v);
            vList.add(v);

            Tensor c = makeParam(new long[]{numExperts, lowRank, lowRank}, dev);
            register_parameter("C_" + i, c);
            cList.add(c);

            Tensor b = torch.zeros(new long[]{inputDim, 1L});
            b.to(dev, ScalarType.Float);
            register_parameter("bias_" + i, b);
            biasList.add(b);
        }

        for (int e = 0; e < numExperts; e++) {
            LinearImpl gate = new LinearImpl(new LinearOptions(inputDim, 1L).bias(false));
            register_module("gate_" + e, gate);
            gating.add(gate);
        }
    }

    private static Tensor makeParam(long[] shape, Device dev) {
        Tensor t = torch.empty(shape);
        t.to(dev, ScalarType.Float);
        torch.xavier_normal_(t);
        return t;
    }

    @Override
    public Tensor forward(Tensor x) {
        // x: (batch, inputDim) -> (batch, inputDim, 1)
        Tensor x0 = x.unsqueeze(2);
        Tensor xl = x0;

        for (int i = 0; i < numLayers; i++) {
            Tensor uLayer = uList.get(i);
            Tensor vLayer = vList.get(i);
            Tensor cLayer = cList.get(i);
            Tensor bLayer = biasList.get(i);

            List<Tensor> expertOuts = new ArrayList<>();
            List<Tensor> expertGates = new ArrayList<>();

            for (int e = 0; e < numExperts; e++) {
                Tensor gScore = gating.get(e).forward(xl.squeeze(2));
                expertGates.add(gScore);

                Tensor vx = torch.matmul(vLayer.select(0, e).t(), xl);
                vx = vx.tanh();
                vx = torch.matmul(cLayer.select(0, e), vx);
                vx = vx.tanh();
                Tensor uvX = torch.matmul(uLayer.select(0, e), vx);
                Tensor dot = x0.mul(uvX.add(bLayer));
                expertOuts.add(dot.squeeze(2));
            }

            TensorVector outVec = new TensorVector();
            for (Tensor t : expertOuts) outVec.push_back(t);
            Tensor outsStacked = torch.stack(outVec, 2L);

            TensorVector gateVec = new TensorVector();
            for (Tensor t : expertGates) gateVec.push_back(t);
            Tensor gatesStacked = torch.stack(gateVec, 1L);

            Tensor moeOut = torch.matmul(outsStacked, gatesStacked.softmax(1));
            xl = moeOut.add(xl);
        }

        return xl.squeeze(2);
    }
}
