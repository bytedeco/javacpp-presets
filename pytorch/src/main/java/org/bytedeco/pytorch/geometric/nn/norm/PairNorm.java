package org.bytedeco.pytorch.geometric.nn.norm;
import org.bytedeco.pytorch.data.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.Parameter;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;

/**
 * PairNorm
 * 防止过平滑：先中心化，再缩放到固定半径。
 */
public class PairNorm extends Module {
    private double scale; // 目标缩放尺度 s
    private boolean scaleIndividually; // 是否对每个特征通道独立缩放 (PyG 参数)

    public PairNorm(double scale, boolean scaleIndividually) {
        this.scale = scale;
        this.scaleIndividually = scaleIndividually;
    }

    public PairNorm() {
        this(1.0, false);
    }

    public Tensor forward(Tensor x, Tensor batch) {
        // 1. Centering (Mean Subtraction)
        // 复用逻辑，手动写一遍以减少对象创建
        long batchSize = (batch != null) ? batch.max().item().toLong() + 1 : 1;

        Tensor xCentered;
        if (batch == null) {
            xCentered = x.sub(x.mean(new long[]{0}, true, new ScalarTypeOptional(torch.ScalarType.Float)));
        } else {
            Tensor mean = AggrUtils.scatter(x, batch, batchSize, "mean");
            xCentered = x.sub(mean.index_select(0, batch));
        }

        // 2. Normalization
        // 计算 Frobenius Norm 或 L2 Norm
        // PyG 逻辑: root_mean_square = sqrt( mean( x_c^2 ) )

        Tensor sq = xCentered.pow(new Scalar(2)); // x^2
        Tensor rootMeanSq;

        if (batch == null) {
            // Global Mean of Squares
            // 如果 scaleIndividually=false, 对所有 dim 求和后再 mean
            // 如果 true, 仅对 dim=0 mean
            if (!scaleIndividually) {
                Tensor sumSq = sq.sum(); // scalar tensor (approx) but we need mean
                // sum / N*C
                double N = x.size(0);
                double C = x.size(1);
                // 使用 mean() 简单处理
                rootMeanSq = sq.mean().sqrt();
            } else {
                rootMeanSq = sq.mean(new long[]{0}, true, new ScalarTypeOptional(torch.ScalarType.Float)).sqrt(); // [1, C]
            }
        } else {
            // Per Graph
            if (!scaleIndividually) {
                // Sum over C dim first -> [N]
                Tensor rowSumSq = sq.sum(new long[]{1}, false, new ScalarTypeOptional(torch.ScalarType.Float));
                // org.bytedeco.pytorch.geometric.utils.Scatter Mean over Batch -> [BatchSize]
                Tensor graphMeanSq = AggrUtils.scatter(rowSumSq, batch, batchSize, "mean");
                rootMeanSq = graphMeanSq.sqrt().unsqueeze(1); // [BatchSize, 1]
            } else {
                // org.bytedeco.pytorch.geometric.utils.Scatter Mean per channel -> [BatchSize, C]
                Tensor graphMeanSq = AggrUtils.scatter(sq, batch, batchSize, "mean");
                rootMeanSq = graphMeanSq.sqrt(); // [BatchSize, C]
            }

            // Broadcast back
            rootMeanSq = rootMeanSq.index_select(0, batch);
        }

        // 3. Scaling
        // x_out = x_c / (rms + eps) * scale
        return xCentered.div(rootMeanSq.add(new Scalar(1e-6))).mul(new Scalar(scale));
    }
}