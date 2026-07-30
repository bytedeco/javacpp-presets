package samples.demo.trainer;

import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;

import static org.bytedeco.pytorch.global.torch.*;

public class MaterialDataFactory {
    public static class MaterialGraph {
        public Tensor x;           // 原子特征 [原子数, 7] (包含电负性、价电子等)
        public Tensor edge_index;   // 化学键/晶格连接 [2, 边数]
        public Tensor edge_attr;    // 键能、键长等
        public Tensor property;    // 目标特性 (如：带隙、硬度、导电性)

        public MaterialGraph(int numAtoms, int numEdges) {
            // 模拟 108 种元素的特征知识
            // 7个特征：原子序数, 电负性, 半径, 价电子, 周期, 族, 第一电离能
//            this.x = randn(new long[]{numAtoms, 7}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));
            this.x = randint(1, 109, new long[]{numAtoms}, new TensorOptions().dtype(new ScalarTypeOptional(kLong())));
            // 强制归一化：将 1-108 映射到 0-1 左右
//            this.x.select(1, 0).divide_(new Scalar(108.0)); // 原子序数归一化
//            this.x.select(1, 1).divide_(new Scalar(4.0));
            // 构造拓扑结构 (晶格连接)

            // 让 property 与元素 ID 挂钩（比如重元素通常密度大、硬度不同）
//            float atomicSum = x.sum().item().toFloat();
//            this.property = tensor(new float[]{atomicSum / 100.0f}, kFloat);
            this.edge_index = randint(0, numAtoms, new long[]{2, numEdges}, new TensorOptions().dtype(new ScalarTypeOptional(kLong())));

            // 模拟键属性
            this.edge_attr = rand(new long[]{numEdges, 3}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));

            float atomicSum = x.sum().item().toFloat();
            float complexity = (float) numEdges / numAtoms;
            this.property = tensor(new float[]{atomicSum * complexity / 100}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));
            // 模拟物理规律：特性由成分的电负性差和结构复杂度决定 (模拟 Target)
//            float complexity = (float) numEdges / numAtoms;
//            float electroSum = x.select(1, 1).sum().item().toFloat();
//            this.property = tensor(new float[]{electroSum * complexity}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));
        }
    }
}
