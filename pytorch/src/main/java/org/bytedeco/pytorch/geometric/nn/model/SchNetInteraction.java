package org.bytedeco.pytorch.geometric.nn.model;
import org.bytedeco.pytorch.data.datasets.*;
import org.bytedeco.pytorch.data.options.*;
import org.bytedeco.pytorch.nn.options.*;

import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.Tensor;
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
import static org.bytedeco.pytorch.global.torch.zeros_like;

public class SchNetInteraction extends org.bytedeco.pytorch.nn.Module {
    private LinearImpl mlp1, mlp2; // Filter Network
    private LinearImpl postLin;    // Post-interaction update
    private long hiddenChannels;

    public SchNetInteraction(long hiddenChannels, long numFilters, int numGaussians, double cutoff) {
        super();
        this.hiddenChannels = hiddenChannels;

        // 1. Filter Network: 将高斯距离特征 [numGaussians] 映射到 [numFilters]
        this.mlp1 = new LinearImpl(numGaussians, numFilters);
        this.mlp2 = new LinearImpl(numFilters, numFilters);

        // 2. Post-interaction: 交互后的特征整合 [numFilters] -> [hiddenChannels]
        this.postLin = new LinearImpl(numFilters, hiddenChannels);

        register_module("mlp1", mlp1);
        register_module("mlp2", mlp2);
        register_module("post_lin", postLin);
    }

    // 指针构造函数（用于从 ModuleList 找回对象，解决之前讨论的转换问题）
    public SchNetInteraction(Module m) {
        super(m);
        this.mlp1 = new LinearImpl(named_modules().get("mlp1"));
        this.mlp2 = new LinearImpl(named_modules().get("mlp2"));
        this.postLin = new LinearImpl(named_modules().get("post_lin"));

    }

    /**
     * @param x          节点特征 [N, hiddenChannels]
     * @param edge_index  边索引 [2, E]
     * @param edgeWeight 高斯扩张后的距离特征 [E, numGaussians]
     */
    public Tensor forward(Tensor x, Tensor edge_index, Tensor edgeWeight) {
        // 1. 生成连续滤波器 (Continuous Filter)
        Tensor W = mlp1.forward(edgeWeight); //ssp// ssp 是 Softplus 激活
        W = mlp2.forward(W);//.ssp();

        // 2. 消息传递：取源节点特征 这种方式一般会导致 jvm crash ！！！！ 因为 edge_index 是 [2, E] 而 x 是 [N, hiddenChannels]
//        Tensor row = edge_index.index( new TensorIndexVector(new TensorIndex(0)));
//        Tensor col = edge_index.index(new TensorIndexVector(new TensorIndex(1)));
//        Tensor msg = x.index(new TensorIndexVector(col)).multiply(W); // 逐元素相乘 (Continuous-filter)

        // 2. 提取边对应的源节点和目标节点索引
        // edge_index 形状为 [2, E]
        Tensor row = edge_index.select(0, 0); // 目标节点 (target)
        Tensor col = edge_index.select(0, 1); // 源节点 (source)

        // 3. 消息传递：W 是 [E, 128], x.index_select(0, col) 是 [E, 128]
        // 这里的相乘必须是逐元素相乘 (Hadamard Product)
        Tensor msg = x.index_select(0, col).multiply(W);
        // 3. 聚合：将消息加回到目标节点
        Tensor out = zeros_like(x);
        out.index_add_(0, row, msg);

        // 4. 更新与残差连接
        out = postLin.forward(out);

        return out;
    }
}