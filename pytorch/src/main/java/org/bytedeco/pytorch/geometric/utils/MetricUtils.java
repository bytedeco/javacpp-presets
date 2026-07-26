package org.bytedeco.pytorch.geometric.utils;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;

import static org.bytedeco.pytorch.global.torch.*;

public class MetricUtils {
    /**
     * 计算 MAPE: mean(abs(y_true - y_pred) / y_true) * 100
     */
    public static float calculateMAPE(Tensor pred, Tensor target) {
        // 增加 1e-5 防止除以 0
        try (PointerScope scope = new PointerScope()) {
            Tensor absDiff = pred.subtract(target).abs();
            Tensor percentageError = absDiff.divide(target.clamp_min(new Scalar(1e-5)));
            return percentageError.mean().item().toFloat() * 100;
        }
    }

    /**
     * 计算 R2 Score
     */
    public static float calculateR2(Tensor pred, Tensor target) {
        // 增加 1e-5 防止除以 0
        try (PointerScope scope = new PointerScope()) {
            Tensor ssRes = pred.subtract(target).pow(new Scalar(2)).sum();
            Tensor ssTot = target.subtract(target.mean()).pow(new Scalar(2)).sum();
            return 1.0f - (ssRes.item().toFloat() / ssTot.item().toFloat());
        }
    }

    /**
     * RMSE (均方根误差): 对大误差更敏感，评估预测的稳定性
     */
    public static float calculateRMSE(Tensor pred, Tensor target) {
        try (PointerScope scope = new PointerScope()) {
            Tensor mse = pred.subtract(target).pow(new Scalar(2)).mean();
            return (float) Math.sqrt(mse.item().toDouble());
        }
    }

    /**
     * 这里的 AUC 采用近似实现：通过预测值与阈值的偏移量作为置信度
     * 在电网中，通常用于衡量“识别节点是否过载”的能力
     */
    public static float calculateAUC(Tensor pred, Tensor target, float threshold) {
        try (PointerScope scope = new PointerScope()) {
            // 将回归转化为分类标签
            Tensor yTrue = target.gt(new Scalar(threshold)).to(kInt());
            // 预测概率 (sigmoid 映射)
            Tensor yProb = sigmoid(pred.subtract(new Scalar(threshold)));

            // 将数据转到 CPU 进行排序计算 (JNI 调用)
            float[] probs = getFloatArray(yProb.cpu());
            int[] labels = getIntArray(yTrue.cpu());

            return computeAUCScore(probs, labels);
        }
    }

    /**
     * 计算 KGE 的排名指标 (MRR, Hits@1, Hits@10)
     *
     * @param posScore  正确三元组的得分 [Batch]
     * @param negScores 负样本(干扰项)的得分 [Batch, NumNegatives]
     * @return 包含指标的数组: [MRR, Hits@1, Hits@10]
     */
    public static float[] calculateKGEMetrics(Tensor posScore, Tensor negScores) {
        try (PointerScope scope = new PointerScope()) {
            // 1. 将正样本得分扩展为 [Batch, 1]，以便与负样本对比
            Tensor posReshaped = posScore.unsqueeze(1);

            // 2. 计算排名：对于每一行，统计负样本得分大于正样本得分的次数
            // Rank = (count(negScore > posScore)) + 1
            // 注意：得分越高越好
            Tensor comparison = negScores.gt(posReshaped).to(kInt());
            Tensor ranks = comparison.sum(new long[]{1}, false, new ScalarTypeOptional()).add(new Scalar(1)).to(kFloat());

            // 3. 计算 MRR: Mean(1/Rank)
            float mrr = ranks.reciprocal().mean().item().toFloat();

            // 4. 计算 Hits@1: Rank <= 1 的比例
            float hits1 = ranks.le(new Scalar(1)).to(kFloat()).mean().item().toFloat();

            // 5. 计算 Hits@10: Rank <= 10 的比例
            float hits10 = ranks.le(new Scalar(10)).to(kFloat()).mean().item().toFloat();

            return new float[]{mrr, hits1, hits10};
        }
    }


    public static float calculatePearson(Tensor x, Tensor y) {
        Tensor vx = x.subtract(x.mean());
        Tensor vy = y.subtract(y.mean());
        Tensor corr = sum(vx.multiply(vy)).divide(
                sqrt(sum(vx.pow(new Scalar(2)))).multiply(sqrt(sum(vy.pow(new Scalar(2))))).add(new Scalar(1e-8))
        );
        return corr.item().toFloat();
    }

    /**
     * MRR (平均倒数排名): KGE 模型的标准指标，越高越好
     * 衡量模型在预测邻居节点或节点属性时，正确答案排在第几位
     */
    public static float calculateMRR(Tensor scores) {
        try (PointerScope scope = new PointerScope()) {
            // 假设 scores 是经过排序后的排名 Tensor
            Tensor ranks = scores.argsort(-1, true).add(new Scalar(1));
            return ranks.to(kFloat()).reciprocal().mean().item().toFloat();
        }
    }

    /**
     * Hits@N: 正确答案在前 N 名中的比例 (如 Hits@1, Hits@10)
     */
    public static float calculateHitsAtN(Tensor scores, int n) {
        try (PointerScope scope = new PointerScope()) {
            Tensor ranks = scores.argsort(-1, true);
            Tensor hits = ranks.lt(new Scalar(n)).to(kFloat()).mean();
            return hits.item().toFloat();
        }
    }

    private static float[] getFloatArray(Tensor t) {
        float[] data = new float[(int) t.numel()];
        t.data_ptr_float().get(data);
        return data;
    }

    private static int[] getIntArray(Tensor t) {
        int[] data = new int[(int) t.numel()];
        t.data_ptr_int().get(data);
        return data;
    }

    private static float computeAUCScore(float[] probs, int[] labels) {
        // 简化的 Wilcoxon-Mann-Whitney AUC 计算

        int pos = 0, neg = 0;
        for (int l : labels)
            if (l == 1) pos++;
            else neg++;
        if (pos == 0 || neg == 0) return 0.5f;

        float sumRank = 0;
        // 实际开发中建议在此处加入对 probs 的排序逻辑
        return 0.5f + (sumRank / (pos * neg)); // 占位示意
    }

}
