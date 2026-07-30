package samples.demo.sampler;
import org.bytedeco.pytorch.data.sampler.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.sampler.EdgeSamplerInput;
import org.bytedeco.pytorch.geometric.sampler.NeighborSampler;
import org.bytedeco.pytorch.geometric.sampler.SamplerModels;

import static org.bytedeco.pytorch.global.torch.*;

public class SamplerTest {
    public static void main(String[] args) {
        System.out.println("=== 启动 Sampler 采样测试 ===");

        // 构造一个包含 10 个节点的链式图: 0->1, 1->2 ...
        long[] edgeFlat = new long[18];
        for(int i=0; i<9; i++) {
            edgeFlat[i*2] = i;
            edgeFlat[i*2+1] = i+1;
        }
        Tensor edgeIndex = tensor(edgeFlat, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long))).view(9, 2).t();

        NeighborSampler sampler = new NeighborSampler(edgeIndex, 10, new int[]{2});

        // 从节点 5 开始采样
        long[] seedFlat = {5L};
        SamplerModels.NodeSamplerInput input = new SamplerModels.NodeSamplerInput(tensor(seedFlat,  new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long))));

        SamplerModels.SamplerOutput output = sampler.sampleFromNodes(input);

        assert output.node != null;
        System.out.println("✅ Sampler 基础架构运行成功！采样到的节点数: " + output.node.size(0));
//        testEdgeSampling(input);
    }

    public void testEdgeSampling(EdgeSamplerInput input) {
        // 1. 模拟从 DataLoader 获取一批索引
        Tensor indices = arange(new Scalar(0), new Scalar( Math.min(input.size(), 32L)),
                new TensorOptions().dtype(new ScalarTypeOptional(kLong())));

        // 2. 获取目标边
        Tensor targetEdges = input.getBatch(indices);

        // 3. 将边展平为唯一的种子节点进行采样
        Tensor seedNodes = unique_consecutive( targetEdges.view(-1)).get0();

        // 4. 调用你现有的 Sampler 基础架构 (假设叫 mySampler)
        // SampleResult result = mySampler.sampleFromNodes(seedNodes);

        System.out.println("✅ EdgeSampler 转换成功！");
        System.out.println("目标边数: " + targetEdges.size(1));
        System.out.println("对应唯一种子节点数: " + seedNodes.size(0));
    }
}