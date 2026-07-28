package org.bytedeco.pytorch.geometric.nn.norm;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Parameter;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;

import static org.bytedeco.pytorch.global.torch.*;

public class InstanceNorm extends org.bytedeco.pytorch.nn.Module {
    private long inChannels;
    private double eps;
    private double momentum;
    private boolean affine;
    private boolean trackRunningStats;

    // 参数
    private Tensor weight;
    private Tensor bias;

    public InstanceNorm(long inChannels, double eps, double momentum, boolean affine, boolean trackRunningStats) {
        super();
        this.inChannels = inChannels;
        this.eps = eps;
        this.momentum = momentum;
        this.affine = affine;
        this.trackRunningStats = trackRunningStats;

        if (affine) {
            this.weight = register_parameter("weight", new Parameter(ones(new long[]{inChannels}), true));
            this.bias = register_parameter("bias", new Parameter(zeros(new long[]{inChannels}), true));
        }

        // 注意：在图神经网络中，InstanceNorm 通常不需要跨 batch 维护全局统计量
        // 但为了对齐接口，如果 trackRunningStats 为 true，需手动维护 running_mean/var
    }

    public Tensor forward3(Tensor x, Tensor batch) {
        x = x.contiguous();
        long N = x.size(0);
        long C = x.size(1);

        if (batch == null) {
            batch = zeros(new long[]{N}, x.options().dtype(new ScalarTypeOptional(kLong())));
        }

        // 1. 计算每个图的节点数 (用于求均值)
        Tensor count = zeros(new long[]{batch.max().item().toLong() + 1, 1}, x.options());
        count.scatter_add_(0, batch.unsqueeze(1), ones(new long[]{N, 1}, x.options()));
        Tensor safeCount = count.clamp_min(new Scalar(1.0));

        // 2. 计算每个图的均值 [NumGraphs, C]
        Tensor sum = zeros(new long[]{count.size(0), C}, x.options());
        // 将 batch 扩展到 [N, C] 布局以便 scatter_add
        Tensor batchExpand = batch.unsqueeze(1).expand(new long[]{N, C}, true);
        sum.scatter_add_(0, batchExpand, x);
        Tensor mean = sum.divide(safeCount); // [NumGraphs, C]

        // 3. 计算每个图的方差
        // Var = E[X^2] - (E[X])^2
        Tensor sumSq = zeros_like(sum);
        sumSq.scatter_add_(0, batchExpand, x.pow(new Scalar(2)));
        Tensor meanSq = sumSq.divide(safeCount);
        Tensor var = meanSq.subtract(mean.pow(new Scalar(2))).clamp_min(new Scalar(eps));

        // 4. 标准化: (x - mean[batch]) / sqrt(var[batch] + eps)
        // 使用 index_select 将 [NumGraphs, C] 映射回 [N, C]
        Tensor nodeMean = mean.index_select(0, batch);
        Tensor nodeVar = var.index_select(0, batch);

        Tensor out = x.subtract(nodeMean).divide(nodeVar.add(new Scalar(eps)).sqrt());

        // 5. Affine 变换
        if (affine) {
            // 显式从模块中获取 parameter，防止引用失效
            Tensor w = named_parameters().get("weight");
            Tensor b = named_parameters().get("bias");
            // 使用 expand_as 辅助广播
            out = out.multiply(w.expand_as(out)).add(b.expand_as(out));
//            out = out.multiply(weight).add(bias);
        }

        return out;
    }

    public Tensor forward(Tensor x, Tensor batch) {
        // 关键 1: 确保输入是连续的
        x = x.contiguous();
        long N = x.size(0);
        long C = x.size(1);

        if (batch == null) {
            batch = zeros(new long[]{N}, x.options().dtype(new ScalarTypeOptional(kLong())));
        }

        // 1. 计算每个图的节点数 [NumGraphs, 1]
        long numGraphs = batch.max().item().toLong() + 1;
        Tensor count = zeros(new long[]{numGraphs, 1}, x.options());
        count.scatter_add_(0, batch.unsqueeze(1), ones(new long[]{N, 1}, x.options()));
        Tensor safeCount = count.clamp_min(new Scalar(1e-6)).contiguous();

        // 2. 计算均值 [NumGraphs, C]
        Tensor sum = zeros(new long[]{numGraphs, C}, x.options());
        Tensor batchExpand = batch.unsqueeze(1).expand(new long[]{N, C}, true);
        sum.scatter_add_(0, batchExpand, x);
        Tensor mean = sum.divide(safeCount).contiguous();

        // 3. 计算方差 Var = E[X^2] - (E[X])^2
        Tensor sumSq = zeros(new long[]{numGraphs, C}, x.options());
        sumSq.scatter_add_(0, batchExpand, x.pow(new Scalar(2)));
        Tensor meanSq = sumSq.divide(safeCount);
        Tensor var = meanSq.subtract(mean.pow(new Scalar(2))).clamp_min(new Scalar(eps)).contiguous();

        // 4. 映射回节点维度并强制 Contiguous (解决 Crash 的核心)
        // index_select 产生的视图必须 contiguous() 才能安全地进行 element-wise 运算
        Tensor nodeMean = mean.index_select(0, batch).contiguous();
        Tensor nodeVar = var.index_select(0, batch).contiguous();

        // 5. 执行标准化
        // 分步计算，每步确保引用存活
        Tensor std = nodeVar.add(new Scalar(eps)).sqrt().contiguous();
        Tensor out = x.subtract(nodeMean).divide(std).contiguous();

        // 6. Affine 变换
        if (affine) {
            // 显式从模块中获取 parameter，防止引用失效
            Tensor w = named_parameters().get("weight");
            Tensor b = named_parameters().get("bias");
            // 使用 expand_as 辅助广播
            out = out.multiply(w.expand_as(out)).add(b.expand_as(out));
        }

        return out;
    }

    public Tensor forward2(Tensor x, Tensor batch) {
        // x: [N, C]
        // batch: [N] 标识每个节点属于哪个图
        if (batch == null) {
            batch = zeros(new long[]{x.size(0)}, x.options().dtype(new ScalarTypeOptional(kLong())));
        }

        // 计算每个 Instance (图) 的均值和方差
        // 利用 AggrUtils.to_dense_batch 将稀疏转换为稠密进行计算更高效且稳健
        // 返回: {denseX [B, MaxSeq, C], mask [B, MaxSeq]}
        Tensor[] denseData = AggrUtils.to_dense_batch(x, batch, x.size(1), 0); //???
        Tensor denseX = denseData[0];
        Tensor mask = denseData[1];

        // 针对每个 Instance (dim=1) 计算均值
        // 只考虑有效节点
        Tensor maskExpand = mask.unsqueeze(2).expand_as(denseX).to(x.dtype());
        Tensor counts = mask.sum(new long[]{1}, true, new ScalarTypeOptional(kLong())).clamp_min(new Scalar(1e-5));

        Tensor mean = denseX.multiply(maskExpand).sum(new long[]{1}, true, new ScalarTypeOptional()).divide(counts);

        // 计算方差: E[X^2] - (E[X])^2
        Tensor var = denseX.pow(new Scalar(2)).multiply(maskExpand).sum(new long[]{1}, true, new ScalarTypeOptional())
                .divide(counts).subtract(mean.pow(new Scalar(2)))
                .clamp_min(new Scalar(eps));

        // 标准化
        Tensor outDense = denseX.subtract(mean).divide(var.add(new Scalar(eps)).sqrt());

        // 还原回稀疏格式 [N, C]
        Tensor out = outDense.masked_select(maskExpand.to(ScalarType.Bool)).view(new long[]{-1, inChannels});

        if (affine) {
            out = out.multiply(weight).add(bias);
        }

        return out;
    }
}