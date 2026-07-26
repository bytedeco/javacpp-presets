package org.bytedeco.pytorch.geometric.nn.norm;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.StringTensorDict;
import org.bytedeco.pytorch.Tensor;

import static org.bytedeco.pytorch.global.torch.*;

public class GraphNorm extends org.bytedeco.pytorch.nn.Module {
    private long inChannels;
    private double eps;

    public GraphNorm(long inChannels) {
        this(inChannels, 1e-5);
    }

    public GraphNorm(long inChannels, double eps) {
        super();
        this.inChannels = inChannels;
        this.eps = eps;

        // 注册参数，但不依赖类成员持有它们的强引用
        register_parameter("weight", ones(new long[]{inChannels}));
        register_parameter("bias", zeros(new long[]{inChannels}));
        register_parameter("mean_scale", ones(new long[]{inChannels}));
    }

    public Tensor forward(Tensor x, Tensor batch) {
        // 1. 预准备 Scalar，避免重复创建
        Scalar sOne = new Scalar(1.0);
        Scalar sTwo = new Scalar(2.0);
        Scalar sZero = new Scalar(0.0);
        Scalar sEps = new Scalar(eps);

        // 2. 从 Module 的参数库中安全提取 Tensor
        // 这样可以确保提取出的 Tensor 指针是当前有效的
        StringTensorDict params = named_parameters();
        Tensor weight_param = params.get("weight");
        Tensor bias_param = params.get("bias");
        Tensor mean_scale_param = params.get("mean_scale");

        if (batch == null) {
            batch = zeros(new long[]{x.size(0)},
                    x.options().dtype(new ScalarTypeOptional(ScalarType.Long)));
        }

        long numGraphs = batch.max().item().toLong() + 1;
        Tensor expandedIndex = batch.unsqueeze(1).expand_as(x);

        // 计算均值
        Tensor sum = zeros(new long[]{numGraphs, inChannels}, x.options());
        sum = sum.scatter_add(0, expandedIndex, x);
        Tensor count = zeros(new long[]{numGraphs, inChannels}, x.options());
        count = count.scatter_add(0, expandedIndex, ones_like(x));
        Tensor mean = sum.divide(count.clamp_min(sOne));

        // 计算方差
        Tensor meanOfSquares = zeros(new long[]{numGraphs, inChannels}, x.options());
        meanOfSquares = meanOfSquares.scatter_add(0, expandedIndex, x.pow(sTwo));
        meanOfSquares = meanOfSquares.divide(count.clamp_min(sOne));
        Tensor var = meanOfSquares.subtract(mean.pow(sTwo)).clamp_min(sZero);

        // 广播
        Tensor nodeMean = mean.index_select(0, batch);
        Tensor nodeVar = var.index_select(0, batch);

        // --- 修复计算逻辑，避免对参数执行 unsqueeze ---
        // 直接利用 multiply 的广播特性，但先将参数放到正确的维度
        // 之前 unsqueeze(0) 崩溃可能是因为 weight_param 没正确获取到

        // 我们改用更稳健的广播写法：
        Tensor alpha = mean_scale_param.view(new long[]{1, inChannels});
        Tensor out = x.subtract(nodeMean.multiply(alpha)).contiguous();

        out = out.divide(nodeVar.add(sEps).sqrt()).contiguous();

        // 最后的线性变换
        Tensor w = weight_param.view(new long[]{1, inChannels});
        Tensor b = bias_param.view(new long[]{1, inChannels});

        out = out.multiply(w).add(b);

        return out;
    }
}


//package org.bytedeco.pytorch.geometric.nn.norm;
//
//import org.bytedeco.pytorch.*;
//import static org.bytedeco.pytorch.global.torch.*;
//
//public class GraphNorm extends org.bytedeco.pytorch.nn.Module {
//    private long inChannels;
//    private double eps;
//    private Tensor weight;    // gamma
//    private Tensor bias;      // beta
//    private Tensor meanScale; // alpha
//
//    public GraphNorm(long inChannels) {
//        this(inChannels, 1e-5);
//    }
//
//    public GraphNorm(long inChannels, double eps) {
//        super();
//        this.inChannels = inChannels;
//        this.eps = eps;
//
//        // 1. 初始化可学习参数
//        this.weight = register_parameter("weight", ones(new long[]{inChannels}));
//        this.bias = register_parameter("bias", zeros(new long[]{inChannels}));
//        this.meanScale = register_parameter("mean_scale", ones(new long[]{inChannels}));
//    }
//
//    /**
//     * @param x 节点特征 [N, inChannels]
//     * @param batch 节点所属图的索引 [N]
//     */
//    public Tensor forward(Tensor x, Tensor batch) {
//        // --- 核心修复 1: 确保 batch 为 Long 类型，防止 index_select 导致 JVM Crash ---
//        if (batch == null) {
//            batch = zeros(new long[]{x.size(0)},
//                    x.options().dtype(new ScalarTypeOptional(org.bytedeco.pytorch.global.torch.ScalarType.Long)));
//        }
//
//        // 1. 获取图的数量
//        long numGraphs = batch.max().item().toLong() + 1;
//
//        // 2. 准备 index 用于 scatter (扩展到与 x 相同的维度 [N, inChannels])
//        Tensor expandedIndex = batch.unsqueeze(1).expand_as(x);
//        System.out.println("111Output (GraphNorm):\n" );
//        // 3. 计算均值 (Mean)
//        Tensor sum = zeros(new long[]{numGraphs, inChannels}, x.options());
//        sum = sum.scatter_add(0, expandedIndex, x);
//
//        Tensor count = zeros(new long[]{numGraphs, inChannels}, x.options());
//        // 使用 ones_like 确保设备和类型一致
//        count = count.scatter_add(0, expandedIndex, ones_like(x));
//        System.out.println("222Output (GraphNorm):\n" );
//        // --- 核心修复 2: 所有数值算子必须使用 new Scalar 包装 ---
//        Tensor mean = sum.divide(count.clamp_min(new Scalar(1.0)));
//
//        // 4. 计算方差 (Variance)
//        Tensor meanOfSquares = zeros(new long[]{numGraphs, inChannels}, x.options());
//        meanOfSquares = meanOfSquares.scatter_add(0, expandedIndex, x.pow(new Scalar(2.0)));
//        meanOfSquares = meanOfSquares.divide(count.clamp_min(new Scalar(1.0)));
//        System.out.println("333Output (GraphNorm):\n" );
//        // var = E[x^2] - (E[x])^2
//        Tensor var = meanOfSquares.subtract(mean.pow(new Scalar(2.0))).clamp_min(new Scalar(0.0));
//
//        // 5. 广播均值和方差到节点维度
//        // batch 必须是 Long 类型，否则此步会触发 libtorch 底层 Crash
//        Tensor nodeMean = mean.index_select(0, batch);
//        Tensor nodeVar = var.index_select(0, batch);
//
//        System.out.println("444Output (GraphNorm):\n" );
//        // 6. 标准化公式：out = (x - alpha * nodeMean) / sqrt(nodeVar + eps) * weight + bias
//        // 使用 new Scalar(eps) 包装
//        Tensor out = x.subtract(nodeMean.multiply(meanScale));
//        out = out.divide(nodeVar.add(new Scalar(eps)).sqrt());
//        System.out.println("222Output (GraphNorm):\n" + out);
//        Tensor w = this.weight.unsqueeze(0).expand_as(out);
//        Tensor b = this.bias.unsqueeze(0).expand_as(out);
//        out = out.multiply(w).add(b);
/// /        out = out.multiply(weight).add(bias);
//
//        return out;
//    }
//}

//
//package org.bytedeco.pytorch.geometric.nn.norm;
//
//import org.bytedeco.pytorch.*;
//import org.bytedeco.pytorch.global.torch;
//import org.bytedeco.pytorch.*;
//import static org.bytedeco.pytorch.global.torch.*;
//
//public class GraphNorm extends org.bytedeco.pytorch.nn.Module {
//    private long inChannels;
//    private double eps;
//    private Tensor weight; // gamma
//    private Tensor bias;   // beta
//    private Tensor meanScale; // alpha
//
//    public GraphNorm(long inChannels) {
//
//        this(inChannels, 1e-5);
//    }
//
//    public GraphNorm(long inChannels, double eps) {
//        super();
//        this.inChannels = inChannels;
//        this.eps = eps;
//
//        // 1. 初始化可学习参数，对应 Python 的 Parameter
//        this.weight = register_parameter("weight", ones(new long[]{inChannels}));
//        this.bias = register_parameter("bias", zeros(new long[]{inChannels}));
//        this.meanScale = register_parameter("mean_scale", ones(new long[]{inChannels}));
//    }
//
//    /**
//     * @param x 节点特征 [N, inChannels]
//     * @param batch 节点所属图的索引 [N], 如果为 null 则视为整张图
//     */
//    public Tensor forward(Tensor x, Tensor batch) {
//        if (batch == null) {
//            batch = zeros(new long[]{x.size(0)}, x.options().dtype(new ScalarTypeOptional(torch.ScalarType.Long)));
//        }
//
//        // --- 核心逻辑：基于 scatter 算子计算每个图的均值 ---
//
//        // 1. 计算每个图的节点数 (batch_size 是图的数量)
//        // 注意：JavaCPP 中 scatter_add 等操作通常通过 global 静态方法调用
//        long numGraphs = batch.max().item().toLong() + 1;
//
//        // 2. 计算均值 (Mean)
//        Tensor sum = zeros(new long[]{numGraphs, inChannels}, x.options());
//        sum = sum.scatter_add(0, batch.unsqueeze(1).expand_as(x), x);
//
//        Tensor count = zeros(new long[]{numGraphs, 1}, x.options());
//        count = count.scatter_add(0, batch.unsqueeze(1), ones(new long[]{x.size(0), 1}, x.options()));
//
//        Tensor mean = sum.divide(count.clamp_min(new Scalar(1.0)));
//
//        // 3. 计算方差 (Variance)
//        // var = mean(x^2) - mean(x)^2
//        Tensor meanOfSquares = zeros(new long[]{numGraphs, inChannels}, x.options());
//        meanOfSquares = meanOfSquares.scatter_add(0, batch.unsqueeze(1).expand_as(x), x.pow(new Scalar(2)));
//        meanOfSquares = meanOfSquares.divide(count.clamp_min(new Scalar(1.0)));
//
//        Tensor var = meanOfSquares.subtract(mean.pow(new Scalar(2))).clamp_min(new Scalar(0.0));
//
//        // --- 应用标准化公式 ---
//
//        // 将 mean 和 var 广播回每个节点 [N, inChannels]
//        Tensor nodeMean = mean.index_select(0, batch);
//        Tensor nodeVar = var.index_select(0, batch);
//
//        // 公式：out = (x - alpha * mean) / sqrt(var + eps) * weight + bias
//        Tensor out = x.subtract(nodeMean.multiply(meanScale));
//        out = out.divide(nodeVar.add(new Scalar(eps)).sqrt());
//        out = out.multiply(weight).add(bias);
//
//        return out;
//    }}