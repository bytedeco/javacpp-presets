package org.bytedeco.pytorch.geometric.nn.model;
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
import org.bytedeco.pytorch.optim.AdamOptions;
import org.bytedeco.pytorch.optim.SGD;
import org.bytedeco.pytorch.optim.SGDOptions;
import org.bytedeco.pytorch.optim.*;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.*;
import static org.bytedeco.pytorch.global.torch.arange;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.nn.conv.PNAConv;
import org.bytedeco.pytorch.geometric.nn.norm.BatchNorm;

import static org.bytedeco.pytorch.global.torch.relu;


public class PNA extends GenericModule {
    private long inChannels;
    private long hiddenChannels;
    private long outChannels;
    private int numLayers;
    private double avgDegree;

    private ModuleListImpl convs;
    private ModuleListImpl norms;

    // 关键修复：增加 Java 侧的强类型引用数组
    private PNAConv[] convLayers;
    private BatchNorm[] normLayers;

    public PNA(long inChannels, long hiddenChannels, long outChannels, int numLayers,
               String[] aggregators, String[] scalers, double avgDegree) {
        super();
        this.inChannels = inChannels;
        this.hiddenChannels = hiddenChannels;
        this.outChannels = outChannels;
        this.numLayers = numLayers;
        this.avgDegree = avgDegree;

        this.convs = new ModuleListImpl();
        this.norms = new ModuleListImpl();
// 关键修复：初始化数组
        this.convLayers = new PNAConv[numLayers];
        this.normLayers = new BatchNorm[numLayers];

        for (int i = 0; i < numLayers; i++) {
            long inDim = (i == 0) ? inChannels : hiddenChannels;
            long outDim = (i == numLayers - 1) ? outChannels : hiddenChannels;

            // 1. 初始化 PNAConv，使用你要求的构造函数签名
            PNAConv conv = new PNAConv(inDim, outDim, aggregators, scalers, avgDegree);
            this.convLayers[i] = conv; // 存入 Java 数组
            this.convs.push_back(conv);

            ////class BatchNorm(in_channels: int, eps: float = 1e-05, momentum: Optional[float] = 0.1, affine: bool = True, track_running_stats: bool = True, allow_single_element: bool = False, 
            // 2. 初始化 BatchNorm (PyG PNA 的默认做法)
            // 这里使用我们之前实现的 PyGBatchNorm
            BatchNorm norm = new BatchNorm(outDim, 1e-05, 0.1, true, true, false);
            this.normLayers[i] = norm; // 存入 Java 数组
            this.norms.push_back(norm);
        }

        register_module("convs", convs);
        register_module("norms", norms);
    }

    /**
     * @param inputs inputs[0] 是 x (Features), inputs[1] 是 edge_index
     */
    @Override
    public Tensor forward(Tensor... inputs) {
        Tensor x = inputs[0];
        Tensor edge_index = inputs[1];

        for (int i = 0; i < numLayers; i++) {
            Tensor xIn = x; // 用于 Residual Connection (如果维度匹配)
// 关键修复：从 Java 数组中获取对象，避开 ClassCastException
            // 不再使用 convs.get(i)
            x = convLayers[i].forward(x, edge_index);
            x = normLayers[i].forward(x);
            // 1. 卷积层
//            x = ((PNAConv) convs.get(i)).forward(x, edge_index);
//            // 2. 标准化层
//            x = ((BatchNorm) norms.get(i)).forward(x);
            // 3. 激活函数 (最后一层通常不加激活，或者根据具体需求)

//            PNAConv conv = new PNAConv(convs.get(i));
//            BatchNorm norm = new BatchNorm(normLayers[i]);
//            x = conv.forward(x, edge_index);
//            x = norm.forward(x);

            if (i != numLayers - 1) {
                x = relu(x);
            }

            // 4. Skip Connection (仅在维度一致时，PyG 逻辑)
            if (xIn.size(-1) == x.size(-1)) {
                x = x.add(xIn);
            }
        }

        return x;
    }
}
//public class PNA extends Module {
//    private ModuleListImpl convs;
//    private ModuleListImpl batchNorms;
//
//    public PNA(long inChannels, long hiddenChannels, long outChannels, int numLayers,
//               double avgDeg, List<String> scalers, List<Aggregation> aggregators) {
//        this.convs = new ModuleListImpl();
//        this.batchNorms = new ModuleListImpl();
//
//        // 计算 PNA 聚合后的膨胀维度
//        int amplification = scalers.size() * aggregators.size();
//
//        for (int i = 0; i < numLayers; i++) {
//            long dimIn = (i == 0) ? inChannels : hiddenChannels;
//
//            // 构建聚合器
//            MultiAggregation multiAggr = new MultiAggregation(aggregators);
//            DegreeScalerAggregation scalerAggr = new DegreeScalerAggregation(avgDeg, scalers, multiAggr);
//
//            // PNAConv 输入维度是 dimIn，但聚合器输出是 dimIn * amplification
//            // Linear 层负责将 dimIn * amplification 映射回 hiddenChannels
//            PNAConv conv = new PNAConv(dimIn * amplification, hiddenChannels, scalerAggr);
//            convs.register_module(String.valueOf(i), conv);
//
//            batchNorms.register_module(String.valueOf(i), new BatchNorm(hiddenChannels, true));
//        }
//    }
//    // forward 略 (循环调用 conv -> bn -> relu)
//}