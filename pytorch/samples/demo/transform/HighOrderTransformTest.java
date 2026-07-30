package samples.demo.transform;

import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.geometric.data.GraphData;
import org.bytedeco.pytorch.geometric.transforms.HighOrderTransforms;

import static org.bytedeco.pytorch.global.torch.*;

public class HighOrderTransformTest {
    public static void main(String[] args) {
        System.out.println("=== 启动全局与高阶变换测试 ===");

        // 1. 模拟一个小图: 3 节点，2 条边
        Tensor x = randn(new long[]{3, 16});
        
//        Tensor edge_index = tensor(new long[][]{{0, 1}, {1, 2}}, new TensorOptions().dtype(new ScalarTypeOptional(kLong())));

        Tensor edge_index = tensor(new long[]{0, 1, 1, 2}, new TensorOptions().dtype(new ScalarTypeOptional(kLong()))).view(2,2);

        GraphData data = new GraphData(x, edge_index);

        // 2. 测试 VirtualNode
        // 预期：节点数 3->4, 边数 2 -> 2 + (3*2) = 8
        HighOrderTransforms.VirtualNode vnTransform = new HighOrderTransforms.VirtualNode();
        data = vnTransform.apply(data);

        System.out.println("添加虚拟节点后节点数: " + data.x.size(0));
        System.out.println("添加虚拟节点后边数: " + data.edge_index.size(1));

        if (data.x.size(0) == 4 && data.edge_index.size(1) == 8) {
            System.out.println("✅ VirtualNode 全局连接验证成功！");
        }

        // 3. 验证 MetaPath 概念
        HighOrderTransforms.AddMetaPaths am = new HighOrderTransforms.AddMetaPaths(new String[]{"药", "靶点", "药"});
        am.apply(data);
    }
}
