package org.bytedeco.pytorch.geometric.metrics;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.Tensor;

import static org.bytedeco.pytorch.global.torch.*;

public class LinkPredPersonalization extends LinkPredMetric {
    public LinkPredPersonalization(int k) { super(k); }

    @Override
    public Tensor compute(Tensor yPred, Tensor yTrue) {
//        try (PointerScope scope = new PointerScope()) {
            // 获取 Top-K 推荐的二进制表示 [Batch, num_items]
            Tensor indices = yPred.topk(k, 1, true, true).get1();
            Tensor binaryRecs = zeros_like(yPred);
            binaryRecs.scatter_(1, indices, ones_like(indices).to(yPred.dtype()));

            // 计算用户间的余弦相似度矩阵 [Batch, Batch]
            Tensor norm = binaryRecs.norm(new ScalarOptional(new Scalar(2)), new long[]{1}, true);
            Tensor normalizedRecs = binaryRecs.div(norm.add(new Scalar(1e-8)));
            Tensor similarityMatrix = mm(normalizedRecs, normalizedRecs.t());

            // 个性化 = 1 - 平均相似度 (排除自相关，即对角线)
            long batchSize = yPred.size(0);
            Tensor mask = ones(new long[]{batchSize, batchSize}, yPred.options()).sub(eye(batchSize, yPred.options()));
            Tensor avgSim = similarityMatrix.mul(mask).sum().div(new Scalar(batchSize * (batchSize - 1)));

            // 返回一个标量 Tensor
            return ones(new long[]{1}, yPred.options()).sub(avgSim).detach();
//        }
    }
}