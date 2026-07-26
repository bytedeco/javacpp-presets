package org.bytedeco.pytorch.geometric.nn.norm;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.StringTensorDict;
import org.bytedeco.pytorch.Tensor;

import static org.bytedeco.pytorch.global.torch.*;
//import static org.bytedeco.pytorch.geometric.utils.Scatter.scatter_mean;


public class LayerNorm extends org.bytedeco.pytorch.nn.Module {
    private long inChannels;
    private double eps;
    private boolean affine;
    private String mode; // 'node' (默认) 或 'graph'

    public LayerNorm(long inChannels, double eps, boolean affine) {
        super();
        this.inChannels = inChannels;
        this.eps = eps;
        this.affine = affine;

        if (affine) {
            register_parameter("weight", ones(new long[]{inChannels}));
            register_parameter("bias", zeros(new long[]{inChannels}));
        }
    }

    public Tensor forward(Tensor x) {
        x = x.contiguous();
        long[] originalShape = x.shape();
        int dimsCount = (int) x.dim();
        long lastDim = dimsCount - 1;
        long C = originalShape[(int) lastDim];

        // 1. 构造用于 keepdim 效果的形状 [B, L, 1]
        long[] keepDimShape = new long[dimsCount];
        for (int i = 0; i < dimsCount - 1; i++) keepDimShape[i] = originalShape[i];
        keepDimShape[dimsCount - 1] = 1;

        // 2. 计算均值和方差，并强制 view 回 keepdim 形状
        long[] reduceDims = {lastDim};
        Tensor mean = x.mean(reduceDims, false, new ScalarTypeOptional()).view(keepDimShape);
        Tensor var = x.var(reduceDims, false).view(keepDimShape);

        Scalar sEps = new Scalar(eps);

        // 3. 标准化：显式使用 expand 替代 expand_as 增加稳定性
        Tensor std = var.add(sEps).sqrt();

        // 强制广播到原图形状
        Tensor out = x.subtract(mean).divide(std.expand(originalShape, true));

        // 4. Affine 变换
        if (affine) {
            StringTensorDict params = named_parameters();
            Tensor weight = params.get("weight");
            Tensor bias = params.get("bias");

            Tensor w = weight;
            Tensor b = bias;
            // 循环提升维度：[C] -> [1, 1, C]
            for (int i = 0; i < dimsCount - 1; i++) {
                w = w.unsqueeze(0);
                b = b.unsqueeze(0);
            }
            out = out.multiply(w.expand(originalShape, true)).add(b.expand(originalShape, true));
        }

        return out;
    }

    public Tensor forward3(Tensor x) {
        x = x.contiguous();
        long lastDim = x.dim() - 1;
        long C = x.size(lastDim); // 实际特征维度，例如 64

        // 1. 计算均值和方差 (针对最后一个维度)
        long[] dims = {lastDim};
        Tensor mean = x.mean(dims, true, new ScalarTypeOptional(ScalarType.Float));
        Tensor var = x.var(dims, true, false);

        Scalar sEps = new Scalar(eps);

        // 2. 标准化： (x - mean) / sqrt(var + eps)
        // 使用 expand_as 强制广播
        Tensor std = var.add(sEps).sqrt();
        Tensor out = x.subtract(mean).divide(std.expand_as(x));

        // 3. Affine 变换
        if (affine) {
            StringTensorDict params = named_parameters();
            Tensor weight = params.get("weight");
            Tensor bias = params.get("bias");

            // --- 修复点：更稳健的维度扩展方式 ---
            // 不再使用 view(long[])，改用循环 unsqueeze
            // 这样可以将 [64] 变为 [1, 1, 64] 或 [1, 64]，取决于 x 的维度
            Tensor w = weight;
            Tensor b = bias;

            for (int i = 0; i < x.dim() - 1; i++) {
                w = w.unsqueeze(0);
                b = b.unsqueeze(0);
            }

            // 此时 w/b 的形状会自动匹配 x 的前缀维度（全为1）
            // 例如 x 是 [B, L, C]，w 变成 [1, 1, C]
            out = out.multiply(w.expand_as(out)).add(b.expand_as(out));
        }

        return out;
    }

    public Tensor forward2(Tensor x) {
        // 1. 强制内存连续，防止异构图切片干扰
        x = x.contiguous();
        long N = x.size(0);
        long C = x.size(1); // 应该是 128

        // 2. 准备 Scalar
        Scalar sEps = new Scalar(eps);

        // 3. 计算均值和方差
        // 显式指定 dim=1 (特征维)，不依赖数组，直接传 long
        // 如果你的 JavaCPP 版本不支持直接传 long，请确保数组写法如下：
        long[] dims = {1};
        Tensor mean = x.mean(dims, true, new ScalarTypeOptional()).view(new long[]{N, 1});
        Tensor var = x.var(dims, true, false).view(new long[]{N, 1});

        // 4. 标准化： (x - mean) / sqrt(var + eps)
        // 此时 x 是 [N, C], mean 是 [N, 1]，相减会触发广播得到 [N, C]
        Tensor centered = x.subtract(mean);
        Tensor std = var.add(sEps).sqrt();

        // 关键点：显式广播 std 到 [N, C] 消除 LibTorch 的猜疑
        Tensor out = centered.divide(std.expand_as(centered));

        // 5. 应用可学习参数
        if (affine) {
            StringTensorDict params = named_parameters();
            Tensor weight = params.get("weight");
            Tensor bias = params.get("bias");

            // 显式 view 为 [1, C]
            Tensor w = weight.view(new long[]{1, C});
            Tensor b = bias.view(new long[]{1, C});

            out = out.multiply(w).add(b);
        }

        return out;
    }
}
//public class LayerNorm extends org.bytedeco.pytorch.nn.Module {
//    private long inChannels;
//    private double eps;
//    private boolean affine;
//    private String mode; // 'node' (默认) 或 'graph'
//
//    private Tensor weight; // gamma
//    private Tensor bias;   // beta
//
//    public LayerNorm(long inChannels) {
//        this(inChannels, 1e-5, true, "node");
//    }
//
//    public LayerNorm(long inChannels, double eps, boolean affine, String mode) {
//        super();
//        this.inChannels = inChannels;
//        this.eps = eps;
//        this.affine = affine;
//        this.mode = mode;
//
//        if (affine) {
//            // 一比一还原 nn.Parameter
//            this.weight = register_parameter("weight", ones(new long[]{inChannels}));
//            this.bias = register_parameter("bias", zeros(new long[]{inChannels}));
//        }
//    }
//
//    /**
//     * @param x 节点特征 [N, inChannels]
//     * @param batch 节点所属图的索引 [N]
//     */
//    public Tensor forward(Tensor x, Tensor batch) {
//        System.out.println("Input X shape: " + java.util.Arrays.toString(x.shape()));
//        System.out.println("Expected inChannels: " + this.inChannels);
//        if (mode.equals("graph")) {
//            // 图级标准化逻辑（类似 GraphNorm 但不带 alpha）
//            return forwardGraph(x, batch);
//        } else {
//            // 节点级标准化逻辑 (默认模式)
//            // 公式: (x - mean(x)) / sqrt(var(x) + eps)
//
//            // 1. 计算均值 (在特征维度 dim=1 上)
//            Tensor mean = x.mean(new long[]{1}, true, new ScalarTypeOptional());
//
//            // 2. 计算方差
//            Tensor var = x.var(new long[]{1}, true, false);
//
//            // 3. 标准化
//            Tensor out = x.subtract(mean).divide(var.add(new Scalar(eps)).sqrt());
//
//            // 4. Affine 变换
//            if (affine) {
//                out = out.multiply(weight).add(bias);
//            }
//            return out;
//        }
//    }
//
//    private Tensor forwardGraph(Tensor x, Tensor batch) {
//        if (batch == null) {
//            batch = zeros(new long[]{x.size(0)}, x.options().dtype(new ScalarTypeOptional(torch.ScalarType.Long)));
//        }
//
//        // 计算每个图的均值和方差 (利用我们之前实现的 ScatterUtils)
//        long numGraphs = batch.max().item().toLong() + 1;
//
//        Tensor mean =  scatter_mean(x, batch, 0, numGraphs);
//        Tensor sqMean = scatter_mean(x.pow(new Scalar(2)), batch, 0, numGraphs);
//        Tensor var = sqMean.subtract(mean.pow(new Scalar(2))).clamp_min(new Scalar(eps));
//
//        // 广播回节点
//        Tensor nodeMean = mean.index_select(0, batch);
//        Tensor nodeVar = var.index_select(0, batch);
//
//        Tensor out = x.subtract(nodeMean).divide(nodeVar.add(new Scalar(eps)).sqrt());
//
//        if (affine) {
//            out = out.multiply(weight).add(bias);
//        }
//        return out;
//    }
//
//
//    public static Tensor scatter_mean(Tensor src, Tensor index, long dim, long dimSize) {
//        // 创建与输出形状一致的 sum 张量
//        long[] outShape = src.shape();
//        outShape[(int)dim] = dimSize;
//        Tensor out = torch.zeros(outShape, src.options());
//
//        // 计算累加和
//        out = out.scatter_add(dim, index.unsqueeze(-1).expand_as(src), src);
//
//        // 计算每个 index 出现的次数 (Count)
//        Tensor count = torch.zeros(outShape, src.options());
//        Tensor ones = torch.ones_like(src);
//        count = count.scatter_add(dim, index.unsqueeze(-1).expand_as(src), ones);
//
//        // 防止除以 0
//        return out.divide(count.clamp_min(new Scalar(1.0)));
//    }
//}
