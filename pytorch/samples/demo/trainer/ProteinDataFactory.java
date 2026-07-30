package samples.demo.trainer;

import org.bytedeco.pytorch.*;

import static org.bytedeco.pytorch.global.torch.*;

public class ProteinDataFactory {
    // 假设有 20 种标准氨基酸
    public static class ProteinGraph {
        public Tensor x;           // [NumAminoAcids, 16]
        public Tensor edge_index;   // [2, NumEdges]
        public Tensor y;           // [1] 蛋白质功能分类索引

        public ProteinGraph(long numNodes, long numEdges, long numClasses) {
            // 1. 生成 0-20 的氨基酸索引
            this.x = randint(0, 20, new long[]{numNodes}, new TensorOptions().dtype(new ScalarTypeOptional(kLong())));

            // 2. 随机构造边
            this.edge_index = randint(0, numNodes, new long[]{2, numEdges}, new TensorOptions().dtype(new ScalarTypeOptional(kLong())));

            // 3. 核心改进：让 y 与 x 产生关联（规律：计算氨基酸索引的均值来决定类别）
            // 这样模型就能通过学习节点特征来预测分类了
            long pseudoClass = x.to(ScalarType.Float).mean().item().toLong() % numClasses;
            this.y = tensor(new long[]{pseudoClass}, new TensorOptions().dtype(new ScalarTypeOptional(kLong())));
        }
//        public ProteinGraph(long numNodes, long numEdges, long numClasses) {

        /// /            this.x = randn(new long[]{numNodes, 16}, new TensorOptions().dtype(new ScalarTypeOptional( kFloat())));
//
//            this.x = randint(0, 20, new long[]{numNodes}, new TensorOptions().dtype(new ScalarTypeOptional(kLong())));
//            // 随机构造边
//            this.edge_index = randint(0, numNodes, new long[]{2, numEdges}, new TensorOptions().dtype(new ScalarTypeOptional( kLong())));
//            // 随机功能分类 (例如：酶、转运蛋白、结构蛋白)
//            this.y = randint(0, numClasses, new long[]{1}, new TensorOptions().dtype(new ScalarTypeOptional( kLong())));
//        }
        public void to(Device device) {
            this.x = x.to(device, TypeMeta.fromScalarType(kFloat()), false, false);
            this.edge_index = edge_index.to(device, TypeMeta.fromScalarType(kLong()), false, false);
            this.y = y.to(device, TypeMeta.fromScalarType(kLong()), false, false);
        }
    }
}