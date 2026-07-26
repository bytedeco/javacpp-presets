package org.bytedeco.pytorch.geometric.demo.metric;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.geometric.metrics.LinkPredMAP;
import org.bytedeco.pytorch.geometric.metrics.LinkPredMetricCollection;
import org.bytedeco.pytorch.geometric.metrics.LinkPredPrecision;
import org.bytedeco.pytorch.geometric.metrics.LinkPredRecall;

import java.util.Map;

import static org.bytedeco.pytorch.global.torch.kFloat;
import static org.bytedeco.pytorch.global.torch.tensor;

public class MetricTest {
    public static void main(String[] args) {
        System.out.println("=== 链接预测评估指标测试 ===");

        try (PointerScope scope = new PointerScope()) {
            // 模拟预测得分 (Batch=2, Items=5)
            // 样本1: [0.1, 0.4, 0.3, 0.9, 0.2] -> Top-2 索引是 [3, 1]
            // 样本2: [0.8, 0.1, 0.2, 0.3, 0.4] -> Top-2 索引是 [0, 4]
            Tensor yPred = tensor(new float[]
                    {0.1f, 0.4f, 0.3f, 0.9f, 0.2f,
                    0.8f, 0.1f, 0.2f, 0.3f, 0.4f}
            , new TensorOptions().dtype(new ScalarTypeOptional(kFloat()))).view(2,5);

            // 真实标签
            // 样本1: 索引 3 是相关的
            // 样本2: 索引 0 是相关的
            Tensor yTrue = tensor(new float[]
                    {0, 0, 0, 1, 0,
                    1, 0, 0, 0, 0}
            , new TensorOptions().dtype(new ScalarTypeOptional(kFloat()))).view(2,5);

            LinkPredMetricCollection collection = new LinkPredMetricCollection();
            collection.addMetric("P@2", new LinkPredPrecision(2));
            collection.addMetric("R@2", new LinkPredRecall(2));
            collection.addMetric("MAP@2", new LinkPredMAP(2));

            Map<String, Double> results = collection.computeAll(yPred, yTrue);

            // 验证逻辑：
            // 样本1: Top-2 (3, 1) 中 3 是相关的 -> Precision = 1/2 = 0.5
            // 样本2: Top-2 (0, 4) 中 0 是相关的 -> Precision = 1/2 = 0.5
            // 平均 Precision@2 = 0.5

            System.out.println("测试结果: " + results);

            if (Math.abs(results.get("P@2") - 0.5) < 1e-5) {
                System.out.println("✅ LinkPredPrecision 测试通过!");
            }
            if (results.get("MAP@2") > 0) {
                System.out.println("✅ LinkPredMAP 测试通过!");
            }
        }
    }
}