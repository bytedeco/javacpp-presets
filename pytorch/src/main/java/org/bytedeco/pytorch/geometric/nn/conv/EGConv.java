package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.Parameter;
import org.bytedeco.pytorch.geometric.utils.TensorToolkit;
import java.util.List;
import java.util.Map;
import java.util.HashMap;

public class EGConv extends MessagePassing {
    private int numHeads;
    private int numBases;
    private List<String> aggregators;
    private long inChannels;
    private long outChannelsPerHead;

    public Tensor basesWeights;
    public LinearImpl linCoeffs;
    private Tensor bias;
    private static final Map<String, AggregatorType> AGG_MAP = new HashMap<>();
    static {
        AGG_MAP.put("sum", AggregatorType.SUM);
        AGG_MAP.put("mean", AggregatorType.MEAN);
        AGG_MAP.put("max", AggregatorType.MAX);
    }

    private enum AggregatorType {
        SUM, MEAN, MAX
    }

    public EGConv(long inChannels, long outChannels, List<String> aggregators,
                  int numHeads, int numBases, boolean hasBias) {
        super("add");
        this.inChannels = inChannels;
        this.numHeads = numHeads;
        this.numBases = numBases;
        this.aggregators = aggregators;
        this.outChannelsPerHead = outChannels / numHeads;

        // 输入校验
        if (outChannels % numHeads != 0) {
            throw new IllegalArgumentException(
                    "outChannels (" + outChannels + ") must be divisible by numHeads (" + numHeads + ")");
        }
        for (String agg : aggregators) {
            if (!AGG_MAP.containsKey(agg)) {
                throw new IllegalArgumentException("Unsupported aggregator: " + agg +
                        ", support: sum/mean/max");
            }
        }

        // 1. 初始化基底权重（维度：[numBases, inChannels, outChannelsPerHead]）
        TensorOptions weightOptions = new TensorOptions()
                .dtype(new ScalarTypeOptional(torch.ScalarType.Float))
                .device(new DeviceOptional(new Device(torch.kCPU())))
                .requires_grad(new BoolOptional(true));
        long[] weightShape = new long[]{numBases, inChannels, outChannelsPerHead};
        this.basesWeights = torch.randn(weightShape, weightOptions);
        register_parameter("bases_weights", this.basesWeights);

        // 2. 初始化系数线性层
        long numAggs = aggregators.size();
        long outFeatures = numHeads * numBases * numAggs;
        this.linCoeffs = new LinearImpl(inChannels, outFeatures);

        // 重新创建权重/偏置，避免叶子张量操作
        Tensor newLinWeight = torch.randn(new long[]{outFeatures, inChannels}, weightOptions);
        Parameter linWeightParam = new Parameter(newLinWeight);
        this.linCoeffs.weight(linWeightParam);

        if (this.linCoeffs.bias() != null) {
            Tensor newLinBias = torch.zeros(new long[]{outFeatures}, weightOptions);
            Parameter linBiasParam = new Parameter(newLinBias);
            this.linCoeffs.bias(linBiasParam);
        }
        register_module("lin_coeffs", linCoeffs);

        // 3. 初始化偏置
        if (hasBias) {
            TensorOptions biasOptions = new TensorOptions()
                    .dtype(new ScalarTypeOptional(torch.ScalarType.Float))
                    .device(new DeviceOptional(new Device(torch.kCPU())))
                    .requires_grad(new BoolOptional(true));
            long[] biasShape = new long[]{numHeads * outChannelsPerHead};
            this.bias = torch.zeros(biasShape, biasOptions);
            register_parameter("bias", this.bias);
        }
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        // 输入校验
        long[] xShape = x.sizes().vec().get();
        if (xShape.length != 2) {
            throw new IllegalArgumentException("Input x must be 2D tensor, got " + xShape.length + "D");
        }
        if (xShape[1] != inChannels) {
            throw new IllegalArgumentException(
                    "Input x channels must be " + inChannels + ", got " + xShape[1]);
        }

        long[] edgeIndexShape = edge_index.sizes().vec().get();
        if (edgeIndexShape.length != 2 || edgeIndexShape[0] != 2) {
            throw new IllegalArgumentException(
                    "edge_index must be 2D tensor with shape [2, E], got shape " + shapeToString(edge_index.sizes()));
        }

        long N = xShape[0]; // 节点数
        long numAggs = aggregators.size(); // 聚合器数量

        // 1. 计算混合系数 [N, numHeads, numBases, numAggs]
        Tensor coeffs = linCoeffs.forward(x)
                .view(N, numHeads, numBases, numAggs);
        coeffs = torch.softmax(coeffs, 2); // 在numBases维度归一化

        // 2. 执行多聚合器计算：每个聚合器输出 [N, inChannels]
//        Tensor[] aggregatedList = new Tensor[(int) numAggs];
//        for (int s = 0; s < numAggs; s++) {
//            aggregatedList[s] = aggregateByType(x, edge_index, aggregators.get(s));
//        }
//        // 堆叠后形状：[numAggs, N, inChannels]
//        Tensor aggregated = torch.stack(aggregatedList, 0);

        TensorVector aggregatedList = new TensorVector();//Tensor[(int) numAggs];
        for (int s = 0; s < numAggs; s++) {
            aggregatedList.push_back(aggregateByType(x, edge_index, aggregators.get(s)));
        }
        Tensor aggregated = torch.stack(aggregatedList, 0);
        // ====================== 核心修复：维度变换（关键！）======================
        // 目标：基底权重 [B, In, Out] × 聚合结果 [S, N, In]^T → [B, S, N, Out]
        // 步骤1：调整聚合结果维度 → [S, In, N]
        Tensor aggTrans = aggregated.permute(0, 2, 1); // [numAggs, inChannels, N]

        // 步骤2：扩展基底权重维度 → [1, B, In, Out]
        Tensor basesExpanded = basesWeights.unsqueeze(0); // [1, numBases, inChannels, outChannelsPerHead]

        // 步骤3：扩展聚合结果维度 → [S, 1, In, N]
        Tensor aggExpanded = aggTrans.unsqueeze(1); // [numAggs, 1, inChannels, N]

        // 步骤4：矩阵乘法（核心修复！维度严格匹配）
        // [1, B, In, Out] × [S, 1, In, N] → 广播为 [S, B, In, Out] × [S, B, In, N]
        // matmul 作用于最后两维：In × Out 和 In × N → 需转置基底权重的最后两维
        Tensor basesTrans = basesExpanded.permute(0, 1, 3, 2); // [1, B, Out, In]
        // 矩阵乘法：[S, B, Out, In] × [S, B, In, N] = [S, B, Out, N]
        Tensor transformed = torch.matmul(basesTrans, aggExpanded);

        // 步骤5：调整维度为 [N, B, S, Out]
        transformed = transformed.permute(3, 1, 0, 2); // [N, numBases, numAggs, outChannelsPerHead]

        // ====================== 权重加权求和 ======================
        // 系数维度：[N, H, B, S] → 扩展为 [N, H, B, S, 1]
        Tensor coeffsExpanded = coeffs.unsqueeze(-1);
        // 变换结果扩展为 [N, 1, B, S, Out]
        transformed = transformed.unsqueeze(1);
        // 加权：[N, H, B, S, 1] × [N, 1, B, S, Out] = [N, H, B, S, Out]
        Tensor weighted = coeffsExpanded.mul(transformed);
        // 求和：在B和S维度求和 → [N, H, Out]
        long[] sumDims = new long[]{2, 3};
        Tensor out = torch.sum(weighted, sumDims, false, new ScalarTypeOptional(torch.ScalarType.Float));

        // 合并多头：[N, H*Out]
        Tensor finalOut = out.view(N, numHeads * outChannelsPerHead);
        // 加偏置
        if (bias != null) {
            finalOut = finalOut.add(bias.clone());
        }

        return finalOut;
    }

    private Tensor aggregateByType(Tensor x, Tensor edge_index, String aggType) {
        AggregatorType type = AGG_MAP.get(aggType);
        long[] xShape = x.sizes().vec().get();
        Tensor aggrResult = propagate(edge_index, x, new long[]{xShape[0], xShape[0]});

        switch (type) {
            case MEAN:
                Tensor row = edge_index.select(0, 0);
                TensorOptions floatOptions = new TensorOptions()
                        .dtype(new ScalarTypeOptional(torch.ScalarType.Float))
                        .device(new DeviceOptional(new Device(torch.kCPU())));
                Tensor ones = torch.ones(row.sizes(), floatOptions);
                Tensor deg = torch.zeros(new long[]{xShape[0]}, floatOptions);
                deg = deg.scatter_add(0, row, ones); // 非原地操作

                Scalar minVal = torch.tensor(1e-6f).item();
                deg = torch.clamp(deg, new ScalarOptional(minVal), new ScalarOptional());
                aggrResult = aggrResult.div(deg.unsqueeze(1));
                break;
            case MAX:
                aggrResult = maxAggregate(x, edge_index);
                break;
            case SUM:
                break;
        }
        return aggrResult;
    }

    private Tensor maxAggregate(Tensor x, Tensor edge_index) {
        long[] xShape = x.sizes().vec().get();
        long N = xShape[0];
        Tensor row = edge_index.select(0, 0);
        Tensor col = edge_index.select(0, 1);
        Tensor x_j = x.index_select(0, col);

        Scalar negInf = torch.tensor(-Float.MAX_VALUE).item();
        TensorOptions floatOptions = new TensorOptions()
                .dtype(new ScalarTypeOptional(torch.ScalarType.Float))
                .device(new DeviceOptional(new Device(torch.kCPU())));
        long[] maxShape = new long[]{N, xShape[1]};
        Tensor maxVals = torch.full(maxShape, negInf, floatOptions);

        for (long i = 0; i < N; i++) {
            Tensor iTensor = torch.tensor(i, new TensorOptions()
                    .dtype(new ScalarTypeOptional(torch.ScalarType.Long))
                    .device(new DeviceOptional(new Device(torch.kCPU()))));
            Tensor mask = row.eq(iTensor);
            if (torch.any(mask).item_bool()) {
                Tensor nodeFeatures = x_j.masked_select(mask.unsqueeze(1))
                        .view(-1, xShape[1]);
                Tensor maxFeat = torch.max(nodeFeatures, 0, false).get0();
                maxVals = maxVals.put(iTensor, maxFeat); // 非原地操作
            }
            iTensor.close();
        }

        // 替换负无穷为0（非原地）
        Tensor zeroTensor = torch.tensor(0.0f, floatOptions);
        Tensor maskNegInf = maxVals.eq(torch.tensor(-Float.MAX_VALUE, floatOptions));
        maxVals = maxVals.masked_fill(maskNegInf, zeroTensor.item());

        zeroTensor.close();
        maskNegInf.close();
        return maxVals;
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        return x_j;
    }

    private String shapeToString(LongArrayRef sizes) {
        long[] shapeArr = sizes.vec().get();
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < shapeArr.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(shapeArr[i]);
        }
        sb.append("]");
        return sb.toString();
    }

    public void setBasesWeights(Parameter newBasesWeights) {
        this.basesWeights = newBasesWeights;
        // 重新注册参数（可选，确保参数被模型管理）
        register_parameter("bases_weights", this.basesWeights);
    }

//    @Override
//    protected void finalize() throws Throwable {
//        try {
//            if (basesWeights != null) basesWeights.close();
//            if (linCoeffs != null) linCoeffs.close();
//            if (bias != null) bias.close();
//        } finally {
//            super.finalize();
//        }
//    }
}

//public class EGConv extends MessagePassing {
//    private int numHeads;
//    private int numBases;
//    private List<String> aggregators;
//    private long inChannels;
//    private long outChannelsPerHead;
//
//    public Tensor basesWeights;
//    public LinearImpl linCoeffs;
//    private Tensor bias;
//    private static final Map<String, AggregatorType> AGG_MAP = new HashMap<>();
//    static {
//        AGG_MAP.put("sum", AggregatorType.SUM);
//        AGG_MAP.put("mean", AggregatorType.MEAN);
//        AGG_MAP.put("max", AggregatorType.MAX);
//    }
//
//    private enum AggregatorType {
//        SUM, MEAN, MAX
//    }
//
//    public EGConv(long inChannels, long outChannels, List<String> aggregators,
//                  int numHeads, int numBases, boolean hasBias) {
//        super("add");
//        this.inChannels = inChannels;
//        this.numHeads = numHeads;
//        this.numBases = numBases;
//        this.aggregators = aggregators;
//        this.outChannelsPerHead = outChannels / numHeads;
//
//        if (outChannels % numHeads != 0) {
//            throw new IllegalArgumentException(
//                    "outChannels (" + outChannels + ") must be divisible by numHeads (" + numHeads + ")");
//        }
//
//        for (String agg : aggregators) {
//            if (!AGG_MAP.containsKey(agg)) {
//                throw new IllegalArgumentException("Unsupported aggregator: " + agg +
//                        ", support: sum/mean/max");
//            }
//        }
//
//        // 1. 初始化基底权重（修复：避免叶子张量直接操作）
//        TensorOptions weightOptions = new TensorOptions()
//                .dtype(new ScalarTypeOptional(torch.ScalarType.Float))
//                .device(new DeviceOptional(new Device(torch.kCPU())))
//                .requires_grad(new BoolOptional(true));
//        long[] weightShape = new long[]{numBases, inChannels, outChannelsPerHead};
//        this.basesWeights = torch.randn(weightShape, weightOptions).clone(); // 关键：clone避免叶子张量问题
//        register_parameter("bases_weights", this.basesWeights);
//
//        // 2. 初始化系数线性层
//        long numAggs = aggregators.size();
//        long outFeatures = numHeads * numBases * numAggs;
//        this.linCoeffs = new LinearImpl(inChannels, outFeatures);
//        // 修复：线性层权重clone，避免原地修改叶子张量
////        this.linCoeffs.weight().set_(this.linCoeffs.weight().clone());
//        this.linCoeffs.weight().set_requires_grad(true);
//        if (this.linCoeffs.bias() != null) {
////            this.linCoeffs.bias().set
////            this.linCoeffs.bias().set_(this.linCoeffs.bias().clone());
//            this.linCoeffs.bias().set_requires_grad(true);
//        }
//        register_module("lin_coeffs", linCoeffs);
//
//        // 3. 初始化偏置
//        if (hasBias) {
//            TensorOptions biasOptions = new TensorOptions()
//                    .dtype(new ScalarTypeOptional(torch.ScalarType.Float))
//                    .device(new DeviceOptional(new Device(torch.kCPU())))
//                    .requires_grad(new BoolOptional(true));
//            long[] biasShape = new long[]{numHeads * outChannelsPerHead};
//            this.bias = torch.zeros(biasShape, biasOptions).clone();
//            register_parameter("bias", this.bias);
//        }
//    }
//
//    @Override
//    public Tensor forward(Tensor x, Tensor edge_index) {
//        // 输入校验：修复维度判断逻辑
//        long[] xShape = x.sizes().vec().get();
//        if (xShape.length != 2) {
//            throw new IllegalArgumentException("Input x must be 2D tensor, got " + xShape.length + "D");
//        }
//        if (xShape[1] != inChannels) {
//            throw new IllegalArgumentException(
//                    "Input x channels must be " + inChannels + ", got " + xShape[1]);
//        }
//
//        long[] edgeIndexShape = edge_index.sizes().vec().get();
//        if (edgeIndexShape.length != 2 || edgeIndexShape[0] != 2) {
//            throw new IllegalArgumentException(
//                    "edge_index must be 2D tensor with shape [2, E], got shape " + shapeToString(edge_index.sizes()));
//        }
//
//        long N = xShape[0];
//        long numAggs = aggregators.size();
//
//        // 1. 计算混合系数
//        Tensor coeffs = linCoeffs.forward(x)
//                .view(N, numHeads, numBases, numAggs);
//        coeffs = torch.softmax(coeffs, 2);
//
//        // 2. 执行多聚合器计算
//        TensorVector aggregatedList = new TensorVector();//Tensor[(int) numAggs];
//        for (int s = 0; s < numAggs; s++) {
//            aggregatedList.push_back(aggregateByType(x, edge_index, aggregators.get(s)));
//        }
//        Tensor aggregated = torch.stack(aggregatedList, 0);
//
//        // 3. 基底权重线性组合：修复矩阵乘法维度不匹配问题
//        Tensor basesTrans = basesWeights.permute(0, 2, 1); // [B, C_per_head, In]
//        Tensor aggTrans = aggregated.permute(0, 2, 1);     // [S, In, N]
//
//        // 修复核心：调整维度扩展顺序，确保矩阵乘法维度匹配
//        // basesTrans: [B, C_per_head, In] → 扩展为 [B, C_per_head, 1, In]
//        Tensor basesExpanded = basesTrans.unsqueeze(2);
//        // aggTrans: [S, In, N] → 扩展为 [1, 1, S, In, N]
//        Tensor aggExpanded = aggTrans.unsqueeze(0).unsqueeze(0);
//
//        // 矩阵乘法：[B, C_per_head, 1, In] × [1, 1, S, In, N] → [B, C_per_head, S, N]
//        Tensor transformed = torch.matmul(
//                basesExpanded,
//                aggExpanded.narrow(3, 0, basesExpanded.size(3)) // 确保In维度严格匹配
//        );
//        // 转置为 [N, B, S, C_per_head]
//        transformed = transformed.permute(3, 0, 2, 1);
//
//        // 系数扩展：[N, H, B, S] → [N, H, B, S, 1]
//        Tensor coeffsExpanded = coeffs.unsqueeze(-1);
//        // 加权求和：确保维度匹配后求和
//        long[] sumDims = new long[]{2, 3};
//        Tensor out = torch.sum(
//                coeffsExpanded.mul(transformed.unsqueeze(1)),
//                sumDims,
//                false,
//                new ScalarTypeOptional(torch.ScalarType.Float)
//        );
//
//        // 4. 合并多头并加偏置
//        Tensor finalOut = out.view(N, numHeads * outChannelsPerHead);
//        if (bias != null) {
//            finalOut = finalOut.add(bias.clone()); // 修复：clone避免原地操作
//        }
//
//        return finalOut;
//    }
//
//    private Tensor aggregateByType(Tensor x, Tensor edge_index, String aggType) {
//        AggregatorType type = AGG_MAP.get(aggType);
//        long[] xShape = x.sizes().vec().get();
//        Tensor aggrResult = propagate(edge_index, x, new long[]{xShape[0], xShape[0]});
//
//        switch (type) {
//            case MEAN:
//                Tensor row = edge_index.select(0, 0);
//                TensorOptions floatOptions = new TensorOptions()
//                        .dtype(new ScalarTypeOptional(torch.ScalarType.Float))
//                        .device(new DeviceOptional(new Device(torch.kCPU())));
//                Tensor ones = torch.ones(row.sizes(), floatOptions);
//                Tensor deg = torch.zeros(new long[]{xShape[0]}, floatOptions);
//                deg = deg.scatter_add_(0, row, ones);
//
//                // 修复：clamp调用+避免原地修改
//                Scalar minVal = torch.tensor(1e-6f).item();
//                deg = torch.clamp(deg, new ScalarOptional(minVal), new ScalarOptional());
//                aggrResult = aggrResult.div(deg.unsqueeze(1));
//                break;
//            case MAX:
//                aggrResult = maxAggregate(x, edge_index);
//                break;
//            case SUM:
//                break;
//        }
//        return aggrResult;
//    }
//
//    private Tensor maxAggregate(Tensor x, Tensor edge_index) {
//        long[] xShape = x.sizes().vec().get();
//        long N = xShape[0];
//        Tensor row = edge_index.select(0, 0);
//        Tensor col = edge_index.select(0, 1);
//        Tensor x_j = x.index_select(0, col);
//
//        // 修复：full调用+避免原地修改
//        Scalar negInf = torch.tensor(-Float.MAX_VALUE).item();
//        TensorOptions floatOptions = new TensorOptions()
//                .dtype(new ScalarTypeOptional(torch.ScalarType.Float))
//                .device(new DeviceOptional(new Device(torch.kCPU())));
//        long[] maxShape = new long[]{N, xShape[1]};
//        Tensor maxVals = torch.full(maxShape, negInf, floatOptions, floatOptions).clone();
//
//        for (long i = 0; i < N; i++) {
//            Tensor iTensor = torch.tensor(i, new TensorOptions()
//                    .dtype(new ScalarTypeOptional(torch.ScalarType.Long))
//                    .device(new DeviceOptional(new Device(torch.kCPU()))));
//            Tensor mask = row.eq(iTensor);
//            if (torch.any(mask).item_bool()) {
//                Tensor nodeFeatures = x_j.masked_select(mask.unsqueeze(1))
//                        .view(-1, xShape[1]);
//                Tensor maxFeat = torch.max(nodeFeatures, 0, false).get0();
//                // 修复：使用put而非put_，避免原地修改
//                maxVals = maxVals.put(iTensor, maxFeat);
//            }
//            iTensor.close();
//        }
//
//        // 修复：替换负无穷为0，避免原地操作
//        Tensor zeroTensor = torch.tensor(0.0f, floatOptions);
//        Tensor maskNegInf = maxVals.eq(torch.tensor(-Float.MAX_VALUE, floatOptions));
//        maxVals = maxVals.masked_fill(maskNegInf, zeroTensor.item()); // 用masked_fill而非masked_fill_
//
//        zeroTensor.close();
//        maskNegInf.close();
//        return maxVals;
//    }
//
//    @Override
//    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
//        return x_j;
//    }
//
//    private String shapeToString(LongArrayRef sizes) {
//        long[] shapeArr = sizes.vec().get();
//        StringBuilder sb = new StringBuilder("[");
//        for (int i = 0; i < shapeArr.length; i++) {
//            if (i > 0) sb.append(", ");
//            sb.append(shapeArr[i]);
//        }
//        sb.append("]");
//        return sb.toString();
//    }
//
////    @Override
////    protected void finalize() throws Throwable {
////        try {
////            if (basesWeights != null) basesWeights.close();
////            if (linCoeffs != null) linCoeffs.close();
////            if (bias != null) bias.close();
////        } finally {
////            super.finalize();
////        }
////    }
//}

//public class EGConv extends MessagePassing {
//    private int numHeads;
//    private int numBases;
//    private List<String> aggregators;
//    private long inChannels;
//    private long outChannelsPerHead;
//
//    // 基底权重 [numBases, inChannels, outChannels/numHeads]
//    public Tensor basesWeights;
//    // 系数生成线性层
//    public LinearImpl linCoeffs;
//    // 偏置项
//    private Tensor bias;
//    // 聚合器映射（避免重复判断）
//    private static final Map<String, AggregatorType> AGG_MAP = new HashMap<>();
//    static {
//        AGG_MAP.put("sum", AggregatorType.SUM);
//        AGG_MAP.put("mean", AggregatorType.MEAN);
//        AGG_MAP.put("max", AggregatorType.MAX);
//    }
//
//    // 聚合器类型枚举
//    private enum AggregatorType {
//        SUM, MEAN, MAX
//    }
//
//    /**
//     * 构造函数
//     * @param inChannels 输入特征维度
//     * @param outChannels 输出特征维度
//     * @param aggregators 聚合器列表（支持 sum/mean/max）
//     * @param numHeads 注意力头数
//     * @param numBases 基底数量
//     * @param hasBias 是否使用偏置
//     */
//    public EGConv(long inChannels, long outChannels, List<String> aggregators,
//                  int numHeads, int numBases, boolean hasBias) {
//        super("add"); // 基础聚合模式
//        this.inChannels = inChannels;
//        this.numHeads = numHeads;
//        this.numBases = numBases;
//        this.aggregators = aggregators;
//        this.outChannelsPerHead = outChannels / numHeads;
//
//        // 校验输出维度可被头数整除
//        if (outChannels % numHeads != 0) {
//            throw new IllegalArgumentException(
//                    "outChannels (" + outChannels + ") must be divisible by numHeads (" + numHeads + ")");
//        }
//
//        // 校验聚合器类型
//        for (String agg : aggregators) {
//            if (!AGG_MAP.containsKey(agg)) {
//                throw new IllegalArgumentException("Unsupported aggregator: " + agg +
//                        ", support: sum/mean/max");
//            }
//        }
//
//        // 1. 初始化基底权重（带梯度）
//        // 正确的 TensorOptions 构造 + randn 调用
//        TensorOptions weightOptions = new TensorOptions()
//                .dtype(new ScalarTypeOptional(torch.torch.ScalarType.Float))
//                .device(new DeviceOptional(new Device(torch.kCPU())))
//                .requires_grad(new BoolOptional(true));
//        long[] weightShape = new long[]{numBases, inChannels, outChannelsPerHead};
//        this.basesWeights = torch.randn(weightShape, weightOptions);
//        register_parameter("bases_weights", basesWeights);
//
//        // 2. 初始化系数生成线性层（严格使用 LinearImpl）
//        long numAggs = aggregators.size();
//        long outFeatures = numHeads * numBases * numAggs;
//        this.linCoeffs = new LinearImpl(inChannels, outFeatures);
//        // 确保线性层权重带梯度
//        this.linCoeffs.weight().set_requires_grad(true);
//        if (this.linCoeffs.bias() != null) {
//            this.linCoeffs.bias().set_requires_grad(true);
//        }
//        register_module("lin_coeffs", linCoeffs);
//
//        // 3. 初始化偏置
//        if (hasBias) {
//            TensorOptions biasOptions = new TensorOptions()
//                    .dtype(new ScalarTypeOptional(torch.torch.ScalarType.Float))
//                    .device(new DeviceOptional(new Device(torch.kCPU())))
//                    .requires_grad(new BoolOptional(true));
//            long[] biasShape = new long[]{numHeads * outChannelsPerHead};
//            this.bias = torch.zeros(biasShape, biasOptions);
//            register_parameter("bias", bias);
//        }
//    }
//
//    @Override
//    public Tensor forward(Tensor x, Tensor edge_index) {
//        // 输入校验
//        long[] xShape = x.sizes().vec().get();
//        if (xShape.length != 2 || xShape[1] != inChannels) {
//            throw new IllegalArgumentException(
//                    "Input x must be 2D tensor with shape [N, " + inChannels + "], got: " +
//                            shapeToString(x.sizes()));
//        }
//        long[] edgeIndexShape = edge_index.sizes().vec().get();
//        if (edgeIndexShape.length != 2 || edgeIndexShape[0] != 2) {
//            throw new IllegalArgumentException(
//                    "edge_index must be 2D tensor with shape [2, E], got: " +
//                            shapeToString(edge_index.sizes()));
//        }
//
//        long N = xShape[0]; // 节点数
//        long numAggs = aggregators.size();
//
//        // 1. 计算混合系数 [N, H, B, S]
//        // Linear层输出: [N, H*B*S] -> reshape -> [N, H, B, S]
//        Tensor coeffs = linCoeffs.forward(x)
//                .view(N, numHeads, numBases, numAggs);
//        // 在基底维度（dim=2）做softmax归一化
//        coeffs = torch.softmax(coeffs, 2);
//
//        // 2. 执行多聚合器计算 [S, N, In]
//        TensorVector aggregatedList = new  TensorVector();//Tensor[(int) numAggs];
//        for (int s = 0; s < numAggs; s++) {
//            aggregatedList.push_back(aggregateByType(x, edge_index, aggregators.get(s)));
//        }
//        // 堆叠聚合结果: [S, N, In]
//        Tensor aggregated = torch.stack(aggregatedList, 0);
//
//        // 3. 基底权重线性组合（向量化计算，替代循环）
//        // 基底权重: [B, In, C_per_head] -> 转置 -> [B, C_per_head, In]
//        Tensor basesTrans = basesWeights.permute(0, 2, 1);
//        // 聚合结果: [S, N, In] -> 转置 -> [S, In, N]
//        Tensor aggTrans = aggregated.permute(0, 2, 1);
//
//        // 矩阵乘法: [B, C_per_head, In] @ [S, In, N] -> [B, C_per_head, S, N]
//        Tensor transformed = torch.matmul(basesTrans.unsqueeze(2), aggTrans.unsqueeze(0));
//        // 转置: [B, C_per_head, S, N] -> [N, B, S, C_per_head]
//        transformed = transformed.permute(3, 0, 2, 1);
//
//        // 系数: [N, H, B, S] -> 扩展维度 -> [N, H, B, S, 1]
//        Tensor coeffsExpanded = coeffs.unsqueeze(-1);
//        // 加权求和: [N, H, B, S, 1] * [N, B, S, C_per_head] -> [N, H, C_per_head]
//        long[] sumDims = new long[]{2, 3}; // 对B(基底)和S(聚合器)维度求和
//        Tensor out = torch.sum(
//                coeffsExpanded.mul(transformed.unsqueeze(1)),
//                sumDims,
//                false,
//                new ScalarTypeOptional(torch.torch.ScalarType.Float)
//        );
//
//        // 4. 合并多头并加偏置
//        Tensor finalOut = out.view(N, numHeads * outChannelsPerHead);
//        if (bias != null) {
//            finalOut = finalOut.add(bias);
//        }
//
//        return finalOut;
//    }
//
//    /**
//     * 按聚合器类型执行聚合
//     */
//    private Tensor aggregateByType(Tensor x, Tensor edge_index, String aggType) {
//        AggregatorType type = AGG_MAP.get(aggType);
//        long[] xShape = x.sizes().vec().get();
//        Tensor aggrResult = propagate(edge_index, x, new long[]{xShape[0], xShape[0]});
//
//        // 根据聚合类型调整结果
//        switch (type) {
//            case MEAN:
//                // 计算每个节点的入度
//                Tensor row = edge_index.select(0, 0);
//                TensorOptions floatOptions = new TensorOptions()
//                        .dtype(new ScalarTypeOptional(torch.torch.ScalarType.Float))
//                        .device(new DeviceOptional(new Device(torch.kCPU())));
//                Tensor ones = torch.ones(row.sizes(), floatOptions);
//                Tensor deg = torch.zeros(new long[]{x.sizes().vec().get()[0]}, floatOptions);
//                deg = deg.scatter_add_(0, row, ones);
//                // 避免除零：正确调用 clamp API
//                Scalar minVal = torch.tensor(1e-6f).item(); // 转为Scalar
//                deg = torch.clamp(deg, new ScalarOptional(minVal), new ScalarOptional(new Scalar(Float.MAX_VALUE)));
//                // 维度扩展
//                deg = deg.unsqueeze(1);
//                aggrResult = aggrResult.div(deg);
//                break;
//            case MAX:
//                // 重新实现max聚合
//                aggrResult = maxAggregate(x, edge_index);
//                break;
//            case SUM:
//                // 默认sum聚合，无需额外处理
//                break;
//        }
//        return aggrResult;
//    }
//
//    /**
//     * 手动实现max聚合（严格按API调用）
//     */
//    private Tensor maxAggregate(Tensor x, Tensor edge_index) {
//        long[] xShape = x.sizes().vec().get();
//        long N = xShape[0];
//        Tensor row = edge_index.select(0, 0);
//        Tensor col = edge_index.select(0, 1);
//        Tensor x_j = x.index_select(0, col);
//
//        // 初始化max结果为负无穷：正确调用full API
//        Scalar negInf = torch.tensor(-Float.MAX_VALUE).item();
//        TensorOptions floatOptions = new TensorOptions()
//                .dtype(new ScalarTypeOptional(torch.torch.ScalarType.Float))
//                .device(new DeviceOptional(new Device(torch.kCPU())));
//        long[] maxShape = new long[]{N, xShape[1]};
//        Tensor maxVals = torch.full(maxShape, negInf, floatOptions, floatOptions);
//
//        // 逐节点更新最大值
//        for (long i = 0; i < N; i++) {
//            // 正确创建标量tensor
//            Tensor iTensor = torch.tensor(i, new TensorOptions()
//                    .dtype(new ScalarTypeOptional(torch.torch.ScalarType.Long))
//                    .device(new DeviceOptional(new Device(torch.kCPU()))));
//            Tensor mask = row.eq(iTensor);
//            if (torch.any(mask).item_bool()) { // 正确的bool取值
//                Tensor nodeFeatures = x_j.masked_select(mask.unsqueeze(1))
//                        .view(-1, xShape[1]);
//                // 正确调用max
//                Tensor maxFeat = torch.max(nodeFeatures, 0, false).get0();
//                // 正确调用put_更新张量
//                maxVals.put_(iTensor, maxFeat);
//            }
//            iTensor.close();
//        }
//
//        // 负无穷替换为0
//        Scalar zeroScalar = torch.tensor(0.0f).item();
//        Tensor zeroTensor = torch.tensor(0.0f, floatOptions);
//        Tensor maskNegInf = maxVals.eq(torch.tensor(-Float.MAX_VALUE, floatOptions));
//        maxVals = maxVals.masked_fill_(maskNegInf, zeroScalar);
//
//        // 资源释放
//        zeroTensor.close();
//        maskNegInf.close();
//        return maxVals;
//    }
//
//    @Override
//    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
//        // 消息传递：仅传递邻居节点特征
//        return x_j;
//    }
//
//    /**
//     * 辅助方法：将张量形状转为字符串
//     */
//    private String shapeToString(LongArrayRef sizes) {
//        long[] shapeArr = sizes.vec().get();
//        StringBuilder sb = new StringBuilder("[");
//        for (int i = 0; i < shapeArr.length; i++) {
//            if (i > 0) sb.append(", ");
//            sb.append(shapeArr[i]);
//        }
//        sb.append("]");
//        return sb.toString();
//    }
//
//    // 资源释放（避免内存泄漏）
////    @Override
////    protected void finalize() throws Throwable {
////        try {
////            if (basesWeights != null) basesWeights.close();
////            if (linCoeffs != null) linCoeffs.close();
////            if (bias != null) bias.close();
////        } finally {
////            super.finalize();
////        }
////    }
//}



//package org.bytedeco.pytorch.geometric.nn.conv;
//
//import org.bytedeco.pytorch.nn.modules.LinearImpl;
//import org.bytedeco.pytorch.Tensor;
//import org.bytedeco.pytorch.global.torch;
//
//import java.util.List;
//
///**
// * 严格使用 LinearImpl 实现 torch_geometric.nn.conv.EGConv
// * 结合多聚合器融合与基底权重线性组合的高效图卷积。
// */
//public class EGConv extends MessagePassing {
//    private int numHeads;
//    private int numBases;
//    private List<String> aggregators;
//
//    // 严格使用 LinearImpl 持有基底权重
//    // 形状：[numBases, inChannels, outChannels / numHeads]
//    private Tensor basesWeights;
//
//    // 严格使用 LinearImpl 生成混合系数 (Attention-like coefficients)
//    private LinearImpl linCoeffs;
//
//    private Tensor bias;
//
//    public EGConv(long inChannels, long outChannels, List<String> aggregators, int numHeads, int numBases, boolean hasBias) {
//        super("add");
//        this.numHeads = numHeads;
//        this.numBases = numBases;
//        this.aggregators = aggregators;
//
//        if (outChannels % numHeads != 0) {
//            throw new IllegalArgumentException("outChannels must be divisible by numHeads");
//        }
//
//        // 1. 初始化基底权重 [B, In, C_out/H]
//        this.basesWeights = torch.randn(new long[]{numBases, inChannels, outChannels / numHeads});
//        register_parameter("bases_weights", basesWeights);
//
//        // 2. 初始化系数生成网络: 从输入特征映射到 [H, B, S] (S 为聚合器数量)
//        // 严格使用 LinearImpl
//        long numAggs = aggregators.size();
//        this.linCoeffs = new LinearImpl(inChannels, numHeads * numBases * numAggs);
//        register_module("lin_coeffs", linCoeffs);
//
//        if (hasBias) {
//            this.bias = torch.zeros(new long[]{outChannels});
//            register_parameter("bias", bias);
//        }
//    }
//
//    @Override
//    public Tensor forward(Tensor x, Tensor edge_index) {
//        long N = x.size(0);
//        long C_per_head = basesWeights.size(2);
//        long numAggs = aggregators.size();
//
//        // 1. 计算每个节点的混合系数: [N, H, B, S]
//        Tensor coeffs = linCoeffs.forward(x).view(N, numHeads, numBases, numAggs);
//        coeffs = torch.softmax(coeffs, -1); // 在聚合器维度或基底维度归一化
//
//        // 2. 执行多种聚合操作
//        // 这里简化为处理 "mean" 和 "sum"
//        Tensor[] aggregated = new Tensor[aggregators.size()];
//        for (int s = 0; s < aggregators.size(); s++) {
//            aggregated[s] = aggregate_by_type(x, edge_index, aggregators.get(s));
//        }
//
//        // 3. 核心线性组合逻辑: 
//        // 对每个聚合器 s, 每个基底 b, 应用权重并乘以系数
//        // 这是一个高效的张量收缩过程
//        Tensor out = torch.zeros(new long[]{N, numHeads, C_per_head}, x.options());
//
//        for (int b = 0; b < numBases; b++) {
//            Tensor weight_b = basesWeights.select(0, b); // [In, C_per_head]
//            for (int s = 0; s < numAggs; s++) {
//                // 执行变换: [N, In] @ [In, C_per_head] -> [N, C_per_head]
//                Tensor transformed = torch.matmul(aggregated[s], weight_b);
//
//                // 应用系数并累加到对应的头
//                Tensor c_hbs = coeffs.select(3, s).select(2, b); // [N, H]
//                out = out.add(transformed.unsqueeze(1).mul(c_hbs.unsqueeze(-1)));
//            }
//        }
//
//        // 4. 合并多头并加偏置
//        Tensor finalOut = out.view(N, numHeads * C_per_head);
//        if (bias != null) finalOut = finalOut.add(bias);
//
//        return finalOut;
//    }
//
//    private Tensor aggregate_by_type(Tensor x, Tensor edge_index, String type) {
//        // 内部调用不同的聚合逻辑 (mean, sum, symnorm 等)
//        return propagate(edge_index, x, new long[]{x.size(0), x.size(0)});
//    }
//
//    @Override
//    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
//
//        return x_j;
//    }
//}