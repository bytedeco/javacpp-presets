package org.bytedeco.pytorch.geometric.aggr;
import org.bytedeco.pytorch.data.datasets.*;
import org.bytedeco.pytorch.data.options.*;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.*;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.data.datasets.TensorDataset;
import org.bytedeco.pytorch.data.options.DataLoaderOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.options.CrossEntropyLossOptions;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.options.AdamOptions;
import org.bytedeco.pytorch.optim.SGD;
import org.bytedeco.pytorch.optim.options.SGDOptions;
import org.bytedeco.pytorch.optim.*;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.*;
import static org.bytedeco.pytorch.global.torch.arange;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;
//import org.gnn.framework.utils.org.bytedeco.pytorch.geometric.utils.AggrUtils;

/**
 * Attentional org.bytedeco.pytorch.geometric.aggr.Aggregation (Softmax org.bytedeco.pytorch.geometric.aggr.Aggregation with Gate MLP)
 */
public class AttentionalAggregation extends Aggregation {
    private SequentialImpl gateNN;

    public AttentionalAggregation(long inChannels) {
        // 简单的线性门控: Linear -> Tanh (或其他激活) -> Linear(1)
        // 这里简化为单层 Linear，输出维度必须是 1
        this.gateNN = new SequentialImpl();
        this.gateNN.push_back(new LinearImpl(inChannels, 1));
        register_module("gateNN", gateNN);
    }

    public AttentionalAggregation(SequentialImpl gateNN) {
        this.gateNN = gateNN;
        register_module("gateNN", gateNN);
    }

    @Override
    public Tensor forward(Tensor x, Tensor index, long dimSize) {
        // 1. 计算门控系数 (Logits) [N, 1]
        Tensor logits = gateNN.forward(x);

        // 2. Spatial Softmax [N, 1]
        // 这里 org.bytedeco.pytorch.geometric.utils.AggrUtils.scatter_softmax 需要支持广播
        Tensor alpha = AggrUtils.scatter_softmax(logits, index, dimSize);

        // 3. 加权求和
        Tensor weighted = x.mul(alpha); // 广播乘法
        return AggrUtils.scatter(weighted, index, dimSize, "sum");
    }
}