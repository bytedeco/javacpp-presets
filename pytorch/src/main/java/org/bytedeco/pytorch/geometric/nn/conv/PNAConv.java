package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.geometric.utils.Scatter;

import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.pytorch.global.torch.*;
import static org.bytedeco.pytorch.geometric.utils.Scatter.*;
import static org.bytedeco.pytorch.geometric.utils.Scatter.scatter_add;

public class PNAConv extends MessagePassing {
    private long inChannels;
    private long outChannels;
    private String[] aggregators;
    private String[] scalers;
    private double avgDegree; // 对应 PyG 的 avg_degree

    private LinearImpl postLayer; // 聚合后的降维层
    private LinearImpl preLayer;  // 聚合前的变换层（可选，PyG 中通常包含）

    public PNAConv(Pointer p) {
        super(p);
    }

    public PNAConv(long inChannels, long outChannels, String[] aggregators, String[] scalers, double avgDegree) {
        super("add"); // 基础聚合使用 add，后续手动处理多聚合
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.aggregators = aggregators;
        this.scalers = scalers;
        this.avgDegree = avgDegree;

        // 计算聚合后的总维度: C * num_aggregators * num_scalers
        long totalIn = inChannels * aggregators.length * scalers.length;

        // PNA 通常在聚合后跟一个 Linear 或 MLP
        this.postLayer = new LinearImpl(totalIn, outChannels);
        register_module("post_layer", postLayer);
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        // 1. 进行消息传递（这里简单化处理，假设 message 只是节点特征）
        // PyG 的 PNA 通常在聚合时处理所有逻辑
        return propagate(edge_index, x, new long[]{x.size(0), x.size(0)});
    }

    /**
     * 1. 必须匹配基类签名：(x_j, x_i, edge_index, edge_attr)
     */
    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        // PNA 的消息通常就是邻居特征本身
        return x_j;
    }
    /**
     * 一比一还原 PyG 的多聚合逻辑
     */
    @Override
    public Tensor aggregate(Tensor message, Tensor index, long dimSize) {
        // 1. 计算每个节点的度 (Degree)
        // 注意：index 是 targetIdx [E]，dimSize 是 N
        Tensor deg = zeros(new long[]{dimSize}, message.options());
        Tensor ones = ones(new long[]{index.size(0)}, message.options());
        deg.scatter_add_(0, index, ones);
        deg = deg.clamp_min(new Scalar(1.0)).view(-1, 1); // [N, 1]

        // 2. 预计算 Scaler 因子: log(D+1) / log(avgD+1)
        double logAvg = Math.log(avgDegree + 1.0);
        Tensor logDeg = deg.add(new Scalar(1.0)).log().divide(new Scalar(logAvg));

        // 3. 执行所有聚合器
        List<Tensor> aggrOuts = new ArrayList<>();
        for (String agg : aggregators) {
            aggrOuts.add(dispatchAggregate(agg, message, index, dimSize));
        }
        // 拼接聚合结果: [N, C * num_aggr]
        Tensor combinedAggr = cat(new TensorVector(aggrOuts.toArray(new Tensor[0])), -1);

        // 4. 应用缩放器 (Scalers)
        List<Tensor> finalOuts = new ArrayList<>();
        for (String scaler : scalers) {
            finalOuts.add(applyScaler(scaler, combinedAggr, logDeg));
        }

        // 5. 最终拼接并降维
        Tensor out = cat(new TensorVector(finalOuts.toArray(new Tensor[0])), -1);
        return postLayer.forward(out);
    }
//    @Override
    public Tensor aggregate2(Tensor message, Tensor index, long dimSize) {
        List<Tensor> outs = new ArrayList<>();

        // --- 1. 执行所有的 Aggregators ---
        for (String agg : aggregators) {
            if (agg.equals("mean")) {
                outs.add(scatter_mean(message, index, 0, dimSize));
            } else if (agg.equals("sum")) {
                outs.add(scatter_add(message, index, 0, dimSize));
            } else if (agg.equals("max")) {
                outs.add(scatter_max(message, index, 0, dimSize));
            } else if (agg.equals("min")) {
                outs.add(scatter_min(message, index, 0, dimSize));
            } else if (agg.equals("std")) {
                Tensor mean = scatter_mean(message, index, 0, dimSize);
                Tensor meanSq = scatter_mean(message.pow(new Scalar(2)), index, 0, dimSize);
                outs.add(meanSq.subtract(mean.pow(new Scalar(2))).clamp_min(new Scalar(1e-5)).sqrt().add(new Scalar(1e-6)));
            }
        }

        Tensor out = cat(new TensorVector(outs.toArray(new Tensor[0])), -1);

        // --- 2. 执行所有的 Scalers ---
        // deg = scatter_add(ones) 计算每个节点的度
        Tensor deg = scatter_add(ones(new long[]{message.size(0), 1}, message.options()), index, 0, dimSize);
        deg = deg.clamp_min(new Scalar(1.0));
        Tensor logDeg = deg.log().divide(new Scalar(Math.log(avgDegree + 1e-6))); ///？？？ 不加 + 1e-6 会导致 nan

        List<Tensor> scaledOuts = new ArrayList<>();
        for (String scaler : scalers) {
            if (scaler.equals("identity")) {
                scaledOuts.add(out);
            } else if (scaler.equals("amplification")) {
                // (log(d+1) / log(avg_d))
                scaledOuts.add(out.multiply(logDeg.add(new Scalar(1))));
            } else if (scaler.equals("attenuation")) {
                // (log(avg_d) / log(d+1))
                scaledOuts.add(out.divide(logDeg.add(new Scalar(1))));
            }
        }

        out = cat(new TensorVector(scaledOuts.toArray(new Tensor[0])), -1);

        // 3. 最后通过线性层
        return postLayer.forward(out);
    }

    private Tensor dispatchAggregate(String agg, Tensor msg, Tensor index, long N) {
        switch (agg) {
            case "sum":  return Scatter.scatter(msg, index, N, "add");
            case "mean": return Scatter.scatter(msg, index, N, "mean");
            case "max":  return Scatter.scatter(msg, index, N, "max");
            case "min":  return Scatter.scatter(msg, index, N, "min");
            case "var":
            case "std":
                Tensor mean = Scatter.scatter(msg, index, N, "mean");
                Tensor sqMean = Scatter.scatter(msg.pow(new Scalar(2)), index, N, "mean");
                Tensor var = sqMean.sub(mean.pow(new Scalar(2))).clamp_min(new Scalar(1e-6));
                return agg.equals("var") ? var : var.sqrt();
            default: return Scatter.scatter(msg, index, N, "add");
        }
    }

    private Tensor applyScaler(String scaler, Tensor x, Tensor logDeg) {
        switch (scaler) {
            case "amplification": return x.mul(logDeg);
            case "attenuation":   return x.div(logDeg);
            case "identity":      return x;
            default:              return x;
        }
    }
}
//public class PNAConv extends MessagePassing {
//    private LinearImpl lin;
//
//    public PNAConv(long inChannels, long outChannels, DegreeScalerAggregation aggr) {
/// /        this(aggr); // 使用 PNA 聚合器
//        // PNA 聚合后的维度通常会膨胀 (aggregators * scalers)，需要投影回 outChannels
//        // 假设 aggr 内部已经知道膨胀倍数，或者我们在外部计算好。
//        // 这里假设输入维度已经是膨胀后的，或者简单的线性变换。
//        // PNA 论文中: Linear -> ReLU -> Linear
//        this.lin = new LinearImpl(inChannels, outChannels);
//        register_module("lin", lin);
//    }
//
//    public Tensor forward(Tensor x, Tensor edge_index) {
//        // x: [N, In]
//        // aggregate 会调用 org.bytedeco.pytorch.geometric.aggr.DegreeScalerAggregation, 输出 [N, In * Aggregators * Scalers]
//        // 这里简化逻辑，假设调用者处理好了维度匹配
//        Tensor out = propagate(edge_index, x);
//        return lin.forward(out);
//    }
//
//    @Override
//    public Tensor message(Tensor x_j) {
//        return x_j;
//    }
//}
