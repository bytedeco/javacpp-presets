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
import org.bytedeco.pytorch.optim.options.AdamOptions;
import org.bytedeco.pytorch.optim.SGD;
import org.bytedeco.pytorch.optim.options.SGDOptions;
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
     * Standard (x, edge_index) entry — required so {@code Module.forward(Tensor,Tensor)}
     * dispatches to Java via {@code ModuleAsHelper.hasForwardOverride}.
     */
    public Tensor forward(Tensor x, Tensor edge_index) {
        return forwardImpl(x, edge_index);
    }

    /**
     * @param inputs inputs[0] is x (features), inputs[1] is edge_index
     */
    @Override
    public Tensor forward(Tensor... inputs) {
        if (inputs == null || inputs.length < 2) {
            throw new IllegalArgumentException("PNA.forward expects [x, edge_index]");
        }
        return forwardImpl(inputs[0], inputs[1]);
    }

    private Tensor forwardImpl(Tensor x, Tensor edge_index) {
        for (int i = 0; i < numLayers; i++) {
            Tensor xIn = x; // residual when dims match
            // Prefer Java array of typed layers — avoids ClassCastException on ModuleList
            x = convLayers[i].forward(x, edge_index);
            x = normLayers[i].forward(x);

            if (i != numLayers - 1) {
                x = relu(x);
            }

            // Skip connection only when channel dims agree (PyG-style)
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