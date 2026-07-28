package org.bytedeco.pytorch.geometric.aggr;
import org.bytedeco.pytorch.data.datasets.*;
import org.bytedeco.pytorch.data.options.*;
import org.bytedeco.pytorch.nn.options.*;

import org.bytedeco.pytorch.*;
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
import org.bytedeco.pytorch.geometric.utils.AggrUtils;
//import org.gnn.framework.utils.org.bytedeco.pytorch.geometric.utils.AggrUtils;

/**
 * DeepSets org.bytedeco.pytorch.geometric.aggr.Aggregation
 * 理论基石：Permutation Invariant 函数的通用逼近器
 */
public class DeepSetsAggregation extends Aggregation {
    private Module localMLP;  // psi: 处理每个节点特征
    private Module globalMLP; // rho: 处理聚合后的特征

    public DeepSetsAggregation(Module localMLP, Module globalMLP) {
        this.localMLP = localMLP;
        this.globalMLP = globalMLP;

        // 如果 MLP 为 null，视为 Identity
        if (localMLP != null) register_module("localMLP", localMLP);
        if (globalMLP != null) register_module("globalMLP", globalMLP);
    }

    @Override
    public Tensor forward(Tensor x, Tensor index, long dimSize) {
        // 1. Local MLP (psi)
        Tensor xTrans = x;
        if (localMLP != null) {
            // 需要 cast 为 forward 可用的类型，这里假设传入的是 SequentialImpl
            xTrans = ((SequentialImpl) localMLP).forward(x);
        }

        // 2. Sum org.bytedeco.pytorch.geometric.aggr.Aggregation (DeepSets 标准用 Sum)
        Tensor agg = AggrUtils.scatter(xTrans, index, dimSize, "sum");

        // 3. Global MLP (rho)
        if (globalMLP != null) {
            return ((SequentialImpl) globalMLP).forward(agg);
        }
        return agg;
    }
}