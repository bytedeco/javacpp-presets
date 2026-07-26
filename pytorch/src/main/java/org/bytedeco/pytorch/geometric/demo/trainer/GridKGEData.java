package org.bytedeco.pytorch.geometric.demo.trainer;

import org.bytedeco.pytorch.*;

import static org.bytedeco.pytorch.global.torch.*;

public class GridKGEData {
    public static class PowerSnapshot {
        public Tensor x;             // [N, 60] 5年(60月)负荷历史
        public Tensor headIndices;   // [E_kg] 知识图谱头节点
        public Tensor relIndices;    // [E_kg] 知识图谱关系
        public Tensor tailIndices;   // [E_kg] 知识图谱尾节点
        public Tensor edge_index;     // [2, E_topo] 物理拓扑连接
        public Tensor y;             // [N] 目标负载

        public PowerSnapshot(long numNodes, long numEdges, long numKG) {
            // 1. 构造5年负荷数据 (带正弦季节波动和趋势)
            Tensor time = arange(new Scalar(0), new Scalar(60)).to(kFloat()).view(1, 60);
            Tensor trend = time.multiply(new Scalar(0.05)); // 经济增长年化趋势
            Tensor cycle = time.multiply(new Scalar(2 * Math.PI / 12)).sin().multiply(new Scalar(15)); // 12个月周期
            this.x = rand(numNodes, 1).multiply(new Scalar(100)).add(trend).add(cycle).add(randn(numNodes, 60));

            // 2. 知识图谱三元组 (h, r, t)
            this.headIndices = randint(0, numNodes, new long[]{numKG}, new TensorOptions().dtype(new ScalarTypeOptional(kLong())));
            this.relIndices = randint(0, 5, new long[]{numKG}, new TensorOptions().dtype(new ScalarTypeOptional(kLong()))); // 5种关系类型
            this.tailIndices = randint(0, numNodes, new long[]{numKG}, new TensorOptions().dtype(new ScalarTypeOptional(kLong())));

            // 3. 物理拓扑
            this.edge_index = randint(0, numNodes, new long[]{2, numEdges}, new TensorOptions().dtype(new ScalarTypeOptional(kLong())));
            this.y = x.select(1, 59).add(new Scalar(5)); // 预测目标
        }

        public void to(Device device) {
            x = x.to(device, TypeMeta.fromScalarType(kFloat()), false, false);
            headIndices = headIndices.to(device, TypeMeta.fromScalarType(kLong()), false, false);
            relIndices = relIndices.to(device, TypeMeta.fromScalarType(kLong()), false, false);
            tailIndices = tailIndices.to(device, TypeMeta.fromScalarType(kLong()), false, false);
            edge_index = edge_index.to(device, TypeMeta.fromScalarType(kLong()), false, false);
            y = y.to(device, TypeMeta.fromScalarType(kFloat()), false, false);
        }

//        public void to(Device device) {
//            this.x = x.to(device, TypeMeta.fromScalarType( kFloat()), false,false);
//            this.edge_index = edge_index.to(device, TypeMeta.fromScalarType( kLong()), false, false);
//            this.y = y.to(device, TypeMeta.fromScalarType( kLong()), false, false);
//        }
    }
}
