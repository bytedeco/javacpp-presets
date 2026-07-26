package org.bytedeco.pytorch.geometric.demo.metric;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.geometric.metrics.LinkPredAveragePopularity;
import org.bytedeco.pytorch.geometric.metrics.LinkPredDiversity;
import org.bytedeco.pytorch.geometric.metrics.LinkPredMetricCollection;
import org.bytedeco.pytorch.geometric.metrics.LinkPredPersonalization;

import java.util.Map;

import static org.bytedeco.pytorch.global.torch.*;

public class HighLevelMetricTest {
    public static void main(String[] args) {
        System.out.println("=== 启动高阶指标测试 (Diversity, Personalization, ARP) ===");

        try (PointerScope scope = new PointerScope()) {
            // 模拟数据: Batch=2, Items=4
            Tensor yPred = tensor(new float[]
                    {0.9f, 0.8f, 0.1f, 0.1f, // 用户1 推荐 [0, 1]
                    0.1f, 0.1f, 0.9f, 0.8f}  // 用户2 推荐 [2, 3]
            ,new TensorOptions().dtype(new ScalarTypeOptional(kFloat()))).view(2,4);

            // 类别映射: Item 0,1 是分类 A(0); Item 2,3 是分类 B(1)
            Tensor itemCats = tensor(new long[]{0, 0, 1, 1},new TensorOptions().dtype(new ScalarTypeOptional(kLong())));

            // 流行度映射: 索引 0,2 是热门(1.0); 1,3 是冷门(0.2)
            Tensor itemPop = tensor(new float[]{1.0f, 0.2f, 1.0f, 0.2f}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));

            LinkPredMetricCollection collection = new LinkPredMetricCollection();
            collection.addMetric("Diversity", new LinkPredDiversity(2, itemCats));
            collection.addMetric("Personalization", new LinkPredPersonalization(2));
            collection.addMetric("ARP", new LinkPredAveragePopularity(2, itemPop));

            Map<String, Double> results = collection.computeAll(yPred, zeros_like(yPred));
            System.out.println("高阶指标结果: " + results);

            // 验证逻辑:
            // 用户1推荐 [0,1]，类别都是0 -> Diversity = 1/2 = 0.5
            // 用户1推荐 [0,1]，用户2推荐 [2,3]，完全不同 -> Personalization 接近 1.0
            // 用户1推荐流行度平均 (1.0+0.2)/2 = 0.6 -> ARP = 0.6

            if (results.get("Diversity") == 0.5) {
                System.out.println("✅ Diversity 测试通过!");
            }
            if (results.get("Personalization") > 0.9) {
                System.out.println("✅ Personalization 测试通过!");
            }
            if (Math.abs(results.get("ARP") - 0.6) < 1e-5) {
                System.out.println("✅ AveragePopularity 测试通过!");
            }
        }
    }
}
