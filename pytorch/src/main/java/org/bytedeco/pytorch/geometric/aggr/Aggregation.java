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
import org.bytedeco.pytorch.optim.AdamOptions;
import org.bytedeco.pytorch.optim.SGD;
import org.bytedeco.pytorch.optim.SGDOptions;
import org.bytedeco.pytorch.optim.*;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.*;
import static org.bytedeco.pytorch.global.torch.arange;
import org.bytedeco.pytorch.Tensor;

/**
 * PyG org.bytedeco.pytorch.geometric.aggr.Aggregation 基类
 * 所有聚合层必须实现 forward(x, index, dim_size)
 */
public abstract class Aggregation extends Module {

    public Aggregation() {
        super();
    }

    /**
     * @param x       节点特征 [NumNodes_in_Batch, Features] 或 边特征
     * @param index   索引 [NumNodes_in_Batch] (通常是 batch 向量或 edge_index 的 target)
     * @param dimSize 目标维度大小 (NumGraphs 或 NumNodes)
     * @return 聚合后的特征
     */
    public abstract Tensor forward(Tensor x, Tensor index, long dimSize);
}


//package org.gnn.framework.aggr;

//import org.bytedeco.pytorch.Tensor;
//import org.gnn.framework.utils.org.bytedeco.pytorch.geometric.utils.AggrUtils;

// 1. org.bytedeco.pytorch.geometric.aggr.SumAggregation
//public class org.bytedeco.pytorch.geometric.aggr.SumAggregation extends org.bytedeco.pytorch.geometric.aggr.Aggregation {
//    @Override
//    public Tensor forward(Tensor x, Tensor index, long dimSize) {
//        return org.bytedeco.pytorch.geometric.utils.AggrUtils.scatter(x, index, dimSize, "sum");
//    }
//}

// 2. org.bytedeco.pytorch.geometric.aggr.MeanAggregation
//public class org.bytedeco.pytorch.geometric.aggr.MeanAggregation extends org.bytedeco.pytorch.geometric.aggr.Aggregation {
//    @Override
//    public Tensor forward(Tensor x, Tensor index, long dimSize) {
//        return org.bytedeco.pytorch.geometric.utils.AggrUtils.scatter(x, index, dimSize, "mean");
//    }
//}

// 3. org.bytedeco.pytorch.geometric.aggr.MaxAggregation
//public class org.bytedeco.pytorch.geometric.aggr.MaxAggregation extends org.bytedeco.pytorch.geometric.aggr.Aggregation {
//    @Override
//    public Tensor forward(Tensor x, Tensor index, long dimSize) {
//        // 注意：Max聚合后如果某些节点没有邻居，结果是 -inf，通常需要用 fill_ 0 或者 handle empty
//        // PyG 的默认行为是 fill min value，这里保持 org.bytedeco.pytorch.geometric.utils.AggrUtils 的逻辑
//        return org.bytedeco.pytorch.geometric.utils.AggrUtils.scatter(x, index, dimSize, "max");
//    }
//}

// 4. org.bytedeco.pytorch.geometric.aggr.MinAggregation
//public class org.bytedeco.pytorch.geometric.aggr.MinAggregation extends org.bytedeco.pytorch.geometric.aggr.Aggregation {
//    @Override
//    public Tensor forward(Tensor x, Tensor index, long dimSize) {
//        return org.bytedeco.pytorch.geometric.utils.AggrUtils.scatter(x, index, dimSize, "min");
//    }
//}