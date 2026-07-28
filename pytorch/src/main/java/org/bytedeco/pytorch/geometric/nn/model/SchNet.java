package org.bytedeco.pytorch.geometric.nn.model;
import org.bytedeco.pytorch.data.datasets.*;
import org.bytedeco.pytorch.data.options.*;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.autograd.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.*;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.*;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.data.datasets.TensorDataset;
import org.bytedeco.pytorch.data.options.DataLoaderOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.options.CrossEntropyLossOptions;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.options.AdamOptions;
import org.bytedeco.pytorch.optim.SGD;
import org.bytedeco.pytorch.optim.options.SGDOptions;
import org.bytedeco.pytorch.optim.*;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.*;
import static org.bytedeco.pytorch.global.torch.arange;
import org.bytedeco.pytorch.global.torch;
import static org.bytedeco.pytorch.global.torch.*;

public class SchNet extends org.bytedeco.pytorch.nn.Module {
    private long hiddenChannels;
    private long numFilters;
    private int numInteractions;
    private int numGaussians;
    private double cutoff;

    private EmbeddingImpl embedding;
    private ModuleListImpl interactions;

    public SchNet(long hiddenChannels, long numFilters, int numInteractions, int numGaussians, double cutoff) {
        super();
        this.hiddenChannels = hiddenChannels;
        this.numFilters = numFilters;
        this.numInteractions = numInteractions;
        this.numGaussians = numGaussians;
        this.cutoff = cutoff;

        // 1. 原子嵌入层 (假设支持到 100 号元素)
        this.embedding = new EmbeddingImpl(100, hiddenChannels);
        register_module("embedding", embedding);

        // 2. 交互层列表
        this.interactions = new ModuleListImpl();
        for (int i = 0; i < numInteractions; i++) {
            // 这里每个 Interaction 层负责执行 CFConv 逻辑
            SchNetInteraction interact = new SchNetInteraction(hiddenChannels, numFilters, numGaussians, cutoff);
            // 确保 interact 本身不是 null
            if (interact != null) {
                this.interactions.push_back(interact);
            }
//            interactions.push_back(interact);
        }
        register_module("interactions", interactions);
    }

    public Tensor forward(Tensor z, Tensor pos, Tensor batch) {
        // 1. 获取初始原子表征
        Tensor x = embedding.forward(z);

        // 2. 计算原子间的距离矩阵 (Pairwise Distances)
        // 在电网场景下，这代表站点间的物理欧几里得距离
        Tensor row = arange(new Scalar(pos.size(0)), pos.options().dtype(new ScalarTypeOptional(ScalarType.Long))); // 简化版，实际应通过 radius_graph 获取
        Tensor col = arange(new Scalar(pos.size(0)), pos.options().dtype(new ScalarTypeOptional(ScalarType.Long)));
        Tensor edge_index = stack(new TensorVector(row, col), 0);
// 2. 构造索引 (确保 row 和 col 与 pos 在同一设备上)
        long N = pos.size(0);
        // 3. 计算距离 (这是报错的高危区)
        // 优雅的替代方案：使用 index_select 而不是复杂的 index 算子
        Tensor pos_row = pos.index_select(0, row);
        Tensor pos_col = pos.index_select(0, col);

        // 计算欧几里得距离: ||pos[i] - pos[j]||
        Tensor dist = pos_row.subtract(pos_col).norm(new ScalarOptional(new Scalar(2)), new long[]{1}, false);
        ;
//        Tensor dist = pos.index(new TensorIndexVector(row)).subtract(pos.index(new TensorIndexVector(col))).norm(new ScalarOptional(new Scalar(2)), new long[]{1}, false);

        // --- 核心修复：高斯扩张 ---
        // 构造中心点 mu (从 0 到 cutoff，均匀分布 numGaussians 个点)
        Tensor mu = torch.linspace(new Scalar(0.0), new Scalar(cutoff), numGaussians, dist.options());
        // 构造振幅 gamma (通常为 1.0 / gap^2)
        double gap = cutoff / numGaussians;
        double gamma = 1.0 / (gap * gap);

        // 扩张: e_ij = exp(-gamma * (dist - mu)^2)
        // 利用广播机制：dist 是 [E, 1], mu 是 [1, 50] -> 结果是 [E, 50]
        Tensor edgeWeight = dist.unsqueeze(1).subtract(mu.unsqueeze(0)).pow(new Scalar(2)).multiply(new Scalar(-gamma)).exp();
        // 3. 迭代交互层
        for (int i = 0; i < numInteractions; i++) {
            // 使用之前讨论过的“指针重包装”法获取子模块
            SchNetInteraction layer = new SchNetInteraction(interactions.get(i));
            x = x.add(layer.forward(x, edge_index, edgeWeight));
        }

        return x;
    }

}
//
//public class SchNet extends Module {
//    private EmbeddingImpl embedding; // 原子序数 -> 向量
//    private Tensor distances;    // RBF centers (buffer)
//    private double gamma;        // RBF width
//    private ModuleListImpl interactions;
//    private ModuleListImpl atomwise1, atomwise2;
//    private LinearImpl fcOut;
//    private long numInteractions;
/// /class SchNet(hidden_channels: int = 128, num_filters: int = 128, num_interactions: int = 6, num_gaussians: int = 50, cutoff: float = 10.0, interaction_graph: Optional[Callable] = None, max_num_neighbors: int = 32, readout: str = 'add', dipole: bool = False, mean: Optional[float] = None, std: Optional[float] = None, atomref: Optional[Tensor] = None
//    
//    //hiddenChannels, numFilters, numInteractions, numGaussians, cutoff
//    public SchNet(long numInteractions, long hiddenChannels, long numGaussians, double cutoff) {
//        this.numInteractions = numInteractions;
//
//        // 1. Embedding
//        this.embedding = new EmbeddingImpl(100, hiddenChannels); // 假设最多100种原子
//        register_module("embedding", embedding);
//
//        // 2. RBF Centers (0 to cutoff)
//        Tensor centers = torch.linspace(new Scalar(0), new Scalar(cutoff), numGaussians);
//        this.distances = centers;
//        register_buffer("distances", distances); // 注册为 buffer
//        this.gamma = -0.5 / Math.pow((cutoff / numGaussians), 2);
//
//        // 3. Interaction Blocks
//        this.interactions = new ModuleListImpl();
//        for(int i=0; i<numInteractions; i++) {
//            InteractionBlock block = new InteractionBlock(hiddenChannels, numGaussians);
//            interactions.register_module(String.valueOf(i), block);
//        }
//
//        // 4. Output Blocks (Atom-wise)
//        this.atomwise1 = new ModuleListImpl();
//        this.atomwise2 = new ModuleListImpl();
//        this.fcOut = new LinearImpl(hiddenChannels, 1); // Energy prediction
//        register_module("fcOut", fcOut);
//    }
//
//    // Gaussian RBF Expansion
//    // dist: [E, 1] -> [E, NumGaussians]
//    private Tensor expandDistances(Tensor dist) {
//        // (dist - centers)^2
//        Tensor diff = dist.unsqueeze(1).sub(distances);
//        return diff.pow(new Scalar(2)).mul(new Scalar(gamma)).exp();
//    }
//
//    public Tensor forward(Tensor z, Tensor pos, Tensor batch) {
//        // z: Atom types [N]
//        // pos: Coordinates [N, 3]
//
//        // 1. Embedding
//        Tensor h = embedding.forward(z);
//
//        // 2. Compute Distances (Edge Index & Dist)
//        // 简化：假设全连接或已提供 edge_index，这里演示 Radius Graph 计算
//        // 实际使用时通常传入 edge_index
//        // Tensor edge_index = radius_graph(pos, cutoff, batch, ...)
//        // 这里假设 edge_index 和 dist 已计算好传入，为了 API 简洁省略几何计算部分
//
//        // ... Loop interactions ...
//        // 假设 distExpansion 是 [E, NumGaussians]
//        // h = interaction.forward(h, edge_index, distExpansion)
//
//        // 3. Readout
//        // Sum( Atomwise(h) )
//        return h; // Placeholder
//    }
//
//    // 内部类 InteractionBlock
//    public static class InteractionBlock extends Module {
//        LinearImpl mlp; // Filter Gen
//        LinearImpl out;
//
//        public InteractionBlock(long hidden, long numGaussians) {
//            this.mlp = new LinearImpl(numGaussians, hidden);
//            this.out = new LinearImpl(hidden, hidden);
//            register_module("mlp", mlp);
//            register_module("out", out);
//        }
//        // forward: x = x + scatter( x_j * mlp(rbf_ij) )
//    }
//}