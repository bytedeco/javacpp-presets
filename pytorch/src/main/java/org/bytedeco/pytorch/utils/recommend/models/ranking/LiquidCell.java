/*
 * Ported from torch-rechub-scala: torchrec/models/ranking/LiquidNetWork.scala (LiquidCell)
 */
package org.bytedeco.pytorch.utils.recommend.models.ranking;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.DeviceOptional;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class LiquidCell extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final LinearImpl tc, ip, hp, tm, gw, gh, gb;

    public LiquidCell(int inputDim, int hiddenDim, String device) {
        super("LiquidCell");
        Device dev = new Device(device);

        this.tc = new LinearImpl(1, hiddenDim);
        this.ip = new LinearImpl(inputDim, hiddenDim);
        this.hp = new LinearImpl(hiddenDim, hiddenDim);
        this.tm = new LinearImpl(1, hiddenDim);
        this.gw = new LinearImpl(inputDim, hiddenDim);
        this.gh = new LinearImpl(hiddenDim, hiddenDim);
        this.gb = new LinearImpl(1, hiddenDim);

        for (LinearImpl m : new LinearImpl[]{tc, ip, hp, tm, gw, gh, gb}) {
            m.to(dev, false);
        }

        register_module("timeConstant", tc);
        register_module("inputProj", ip);
        register_module("hiddenProj", hp);
        register_module("timeMod", tm);
        register_module("gateW", gw);
        register_module("gateH", gh);
        register_module("gateB", gb);
    }

    public Tensor forward(Tensor hidden, Tensor input, float time) {
        long batchSize = hidden.size(0);

        Tensor timeTensor = torch.ones(
                new long[]{batchSize, 1L},
                new TensorOptions()
                        .dtype(new ScalarTypeOptional(ScalarType.Float))
                        .device(new DeviceOptional(hidden.device())))
                .mul(new Scalar(time));

        Tensor ntc = tc.forward(timeTensor).relu().neg();
        Tensor rec = hp.forward(hidden);
        Tensor inp = ip.forward(input);
        Tensor tmd = tm.forward(timeTensor);

        Tensor gate = gw.forward(input).add(gh.forward(hidden)).add(gb.forward(timeTensor)).sigmoid();

        return rec.mul(gate).add(inp).add(tmd).add(hidden.mul(ntc));
    }
}
