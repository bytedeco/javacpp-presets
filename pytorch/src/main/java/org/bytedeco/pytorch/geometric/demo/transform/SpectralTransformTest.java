package org.bytedeco.pytorch.geometric.demo.transform;

import org.bytedeco.pytorch.geometric.data.GraphData;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.transforms.SpectralAndStructuralTransforms;

import static org.bytedeco.pytorch.global.torch.*;

public class SpectralTransformTest {
    public static void main(String[] args) {
        System.out.println("=== 启动谱特征与位置编码测试 ===");

        // 1. 构造一个环形图 (4节点: 0-1-2-3-0)
        Tensor x = ones(new long[]{4, 2}); // 初始特征 2 维
        long[] edgeArray = {0, 1, 1, 2, 2, 3, 3, 0, 1, 0, 2, 1, 3, 2, 0, 3};
        Tensor edge_index = tensor(edgeArray, new TensorOptions().dtype(new ScalarTypeOptional(kLong()))).reshape(2, 8);
        GraphData data = new GraphData(x, edge_index);

        // 2. 测试 RandomWalkPE (步数 = 3)
        // 预期维度：2 (原始) + 3 (PE) = 5 维
        SpectralAndStructuralTransforms.AddRandomWalkPE rwTransform =
                new SpectralAndStructuralTransforms.AddRandomWalkPE(3);
        data = rwTransform.apply(data);
        System.out.println("RWPE 处理后维度: " + data.x.size(1));

        // 3. 测试 LaplacianLambdaMax
        SpectralAndStructuralTransforms.LaplacianLambdaMax lMax =
                new SpectralAndStructuralTransforms.LaplacianLambdaMax();
        data = lMax.apply(data);
        System.out.println("图拉普拉斯最大特征值: " + data.get("lambda_max").item_float());

        if (data.x.size(1) == 5 && data.get("lambda_max").item_float() > 0) {
            System.out.println("✅ 谱变换与位置编码验证成功！");
        }
    }
}
