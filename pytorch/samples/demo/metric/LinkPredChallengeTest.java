package samples.demo.metric;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.geometric.metrics.*;

import java.util.Map;

import static org.bytedeco.pytorch.global.torch.kFloat;
import static org.bytedeco.pytorch.global.torch.tensor;

public class LinkPredChallengeTest {
    public static void main(String[] args) {
        System.out.println("=== 启动高级链接预测指标测试 ===");

        try (PointerScope scope = new PointerScope()) {
            // 模拟 Batch=2, Items=4
            // 样本1：推荐顺序 [3, 0, 1, 2], 真实相关是 0 (排在第2位)
            // 样本2：推荐顺序 [2, 1, 0, 3], 真实相关是 2 (排在第1位)
             float[] f1 =new float[]
                    {0.7f, 0.2f, 0.1f, 0.9f, // Top-2: [3, 0]
                            0.1f, 0.2f, 0.9f, 0.0f};
            Tensor yPred = tensor(f1 // Top-2: [2, 1]
            , new TensorOptions().dtype(new ScalarTypeOptional(kFloat()))).view(2,4);

            float[] f2 = new float[]
                    {1, 0, 0, 0, // 索引 0 相关
                            0, 0, 1, 0};
            Tensor yTrue = tensor( f2// 索引 2 相关
            ,new TensorOptions().dtype(new ScalarTypeOptional(kFloat()))).view(2,4);

            LinkPredMetricCollection collection = new LinkPredMetricCollection();
            collection.addMetric("Hit@2", new LinkPredHitRatio(2));
            collection.addMetric("MRR@2", new LinkPredMRR(2));
            collection.addMetric("NDCG@2", new LinkPredNDCG(2));
            collection.addMetric("Coverage@2", new LinkPredCoverage(2));

            Map<String, Double> results = collection.computeAll(yPred, yTrue);
            System.out.println("指标结果: " + results);

            // 验证逻辑：
            // 样本1 Hit: Yes (0在Top2里), MRR: 1/2=0.5
            // 样本2 Hit: Yes (2在Top2里), MRR: 1/1=1.0
            // 平均 Hit@2 = 1.0, MRR@2 = 0.75

            if (results.get("Hit@2") == 1.0 && results.get("MRR@2") == 0.75) {
                System.out.println("✅ HitRatio & MRR 测试通过!");
            }
            if (results.get("Coverage@2") == 0.75) { // 推荐了 {3,0,2} 共 3 个，总共 4 个
                System.out.println("✅ Coverage 测试通过!");
            }
        }
    }
}