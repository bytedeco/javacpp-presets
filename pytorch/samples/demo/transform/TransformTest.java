package samples.demo.transform;

import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.data.GraphData;
import org.bytedeco.pytorch.geometric.transforms.*;

import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.pytorch.global.torch.*;

public class TransformTest {
    public static void main(String[] args) {
        System.out.println("=== 启动 Graph Transforms 测试 ===");

        // 1. 模拟原始数据 (100个节点, 64维原始特征)
        Tensor x = randn(new long[]{100, 64});
//        Tensor edge_index = tensor(new long[][]{{0, 1}, {1, 0}});
        Tensor edge_index = tensor(new long[]{0, 1, 1, 0}).view(2,2);
        GraphData data = new GraphData(x, edge_index);

        // 2. 构建流水线
        List<BaseTransform> pipelineList = new ArrayList<>();
        pipelineList.add(new NormalizeFeatures()); // 归一化
        pipelineList.add(new SVDFeatureReduction(16)); // 降维到 16 维
        pipelineList.add(new Constant(1.0)); // 追加一列 1.0 用于偏差建模

        Compose pipeline = new Compose(pipelineList);

        // 3. 执行变换
        GraphData processed = pipeline.apply(data);

        // 4. 验证结果
        System.out.println("原始维度: [100, 64]");
        System.out.println("处理后维度: " + java.util.Arrays.toString(processed.x.sizes().vec().get()));
        // 应为 [100, 17] (16维 SVD + 1维 Constant)

        if (processed.x.size(1) == 17) {
            System.out.println("✅ Transforms 流水线验证成功！");
        }

        // 5. 测试 ToDevice
        if (torch.hasCUDA()) {
            ToDevice toGpu = new ToDevice(new Device(kCUDA()));
            processed = toGpu.apply(processed);
            System.out.println("✅ 已成功将数据迁移至 GPU: " + processed.x.device().toString());
        }
    }
}
