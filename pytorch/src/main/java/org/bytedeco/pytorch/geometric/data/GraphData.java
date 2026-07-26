package org.bytedeco.pytorch.geometric.data;
import org.bytedeco.pytorch.c10.*;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorIndex;
import org.bytedeco.pytorch.TensorIndexVector;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * 模仿 PyG 的 Data 类
 * 存储图的节点特征、边索引、标签等
 */
public class GraphData {
//    public Tensor x;           // [num_nodes, num_features]
//    public Tensor edge_index;   // [2, num_edges] LongTensor
//    public Tensor edge_attr;    // [num_edges, num_edge_features]
//    public Tensor train_mask;
//    public Tensor y;           // [num_nodes, *] or [1, *]
//    public Tensor pos;
// 2. 核心：边权重（可选，但常用）
// 对应密集邻接矩阵 A[i][j] 中的值
//public Tensor edge_weight;
    public Tensor x;            // [num_nodes, num_features] 节点特征
    public Tensor edge_index;   // [2, num_edges] LongTensor 边索引
    public Tensor edge_weight;  // [num_edges] 边权重 (通常为 Float)
    public Tensor edge_attr;    // [num_edges, num_edge_features] 边多维特征
    public Tensor y;            // 标签 (节点级 [N, *] 或 图级 [1, *])
    public Tensor pos;          // [num_nodes, num_dimensions] 节点空间坐标

    public Tensor adj; // [num_nodes, num_nodes]
    // 如果边不仅有权重，还有多维特征（如距离、类型等）
//    public Tensor edge_attr;
    // 1. 核心：边索引（必须）
//    public Tensor edge_index;


    /**
     * 根据当前的 edge_index 和 edge_weight 初始化密集邻接矩阵 adj
     * 严格对应 PyG 的 to_dense_adj 逻辑
     */
    public void initDenseAdj() {
        long numNodes = numNodes();
        // 1. 创建全 0 矩阵，类型通常与 x 或 edge_weight 一致 (kFloat32)
        // 形状为 [num_nodes, num_nodes]
        this.adj = org.bytedeco.pytorch.global.torch.zeros(
                new long[]{numNodes, numNodes},
                edge_weight != null ? edge_weight.options() : x.options()
        );
        if (this.edge_index == null) {
            System.err.println("Warning: Cannot initialize adj because edge_index is null.");
            return;
        }
        // 2. 填充值
        // edge_index 格式为 [2, E]，第一行为 row, 第二行为 col
        Tensor row = edge_index.select(0, 0);
        Tensor col = edge_index.select(0, 1);

        if (edge_weight != null) {
            // 如果有权重：adj[row, col] = edge_weight
            adj.index_put_(new TensorIndexVector(new TensorIndex(row), new TensorIndex(col)), edge_weight);
        } else {
            // 如果无权重：adj[row, col] = 1.0
            Tensor ones = org.bytedeco.pytorch.global.torch.ones_like(row).to(adj.dtype());
            adj.index_put_(new TensorIndexVector(new TensorIndex(row), new TensorIndex(col)), ones);
        }
    }

    // 3. 核心：边特征（可选）
 
    // --- 2. 动态扩展属性 (Dynamic Attributes) ---
    // 对应 PyG 的 __dict__，存储 Transform 产生的额外数据
    private final Map<String, Tensor> attributes = new HashMap<>();
    
    // 允许动态扩展属性
//    private Map<String, Tensor> otherProps = new HashMap<>();

    public GraphData(Tensor x, Tensor edge_index) {
        this.x = x;
        this.edge_index = edge_index;
//        initDenseAdj();
    }
    /**
     * 设置属性。如果 key 是核心属性名，则直接赋值给成员变量。
     */
    public void put(String key, Tensor value) {
        switch (key) {
            case "x": this.x = value; break;
            case "edge_index": this.edge_index = value; break;
            case "edge_attr": this.edge_attr = value; break;
            case "edge_weight": this.edge_weight = value; break;
            case "y": this.y = value; break;
            case "pos": this.pos = value; break;
            default:
                attributes.put(key, value);
        }
    }

    public Tensor get(String key) {
        switch (key) {
            case "x": return x;
            case "edge_index": return edge_index;
            case "edge_attr": return edge_attr;
            case "edge_weight": return edge_weight;
            case "y": return y;
            case "pos": return pos;
            default:
                return attributes.get(key);
        }
    }

    public boolean hasKey(String key) {
        return get(key) != null;
    }
    public boolean contains(String key) {
        if (get(key) != null) return true;
        return attributes.containsKey(key);
    }

    // --- 4. 辅助方法 ---

//    public long numNodes() {
//        if (x != null) return x.size(0);
//        if (pos != null) return pos.size(0);
//        // 如果都没有，可能需要从 edge_index 中计算最大值，暂略
//        return 0;
//    }

    public long numNodes() {
        if (x != null) return x.size(0);
        if (pos != null) return pos.size(0);
        if (edge_index != null && edge_index.numel() > 0) {
            return edge_index.max().item_long() + 1;
        }
        return 0;
    }

    public long numEdges() {
        return edge_index != null ? edge_index.size(1) : 0;
    }

    public int numNodeFeatures() {
        return (x != null) ? (int) x.size(1) : 0;
    }

    public int numEdgeFeatures() {
        return (edge_attr != null) ? (int) edge_attr.size(1) : 0;
    }
//    public long numNodes() {
//        return x.size(0);
//    }
//
//    public long numEdges() {
//        return edge_index.size(1);
//    }

    public Set<String> keys() {
        Set<String> allKeys = new java.util.HashSet<>(attributes.keySet());
        if (x != null) allKeys.add("x");
        if (edge_index != null) allKeys.add("edge_index");
        if (edge_weight != null) allKeys.add("edge_weight");
        if (edge_attr != null) allKeys.add("edge_attr");
        if (y != null) allKeys.add("y");
        if (pos != null) allKeys.add("pos");
        return allKeys;
    }

    @Override
    public String toString() {
        return String.format("GraphData(x=%s, edge_index=%s, edge_weight=%s, other=%s)",
                x != null ? java.util.Arrays.toString(x.sizes().vec().get()) : "null",
                edge_index != null ? java.util.Arrays.toString(edge_index.sizes().vec().get()) : "null",
                edge_weight != null ? java.util.Arrays.toString(edge_weight.sizes().vec().get()) : "null",
                attributes.keySet());
    }


    // 简单的工厂方法用于测试
    public static GraphData of(Tensor x, Tensor edge_index) {
        return new GraphData(x, edge_index);
    }
}


// 模拟 Data 对象，实际应包含 x, edge_index, y 等
//class GraphData {
//    public Tensor x;
//    public Tensor edge_index;
//    public Tensor train_mask;
//    public GraphData(Tensor x, Tensor edge_index) { this.x = x; this.edge_index = edge_index; }
//}