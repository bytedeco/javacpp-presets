package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import java.util.ArrayList;
import java.util.List;

/**
 * TAGConv (Topology Adaptive Graph Convolution)
 * 修复点：
 * 1. 匹配 MessagePassing 基类接口
 * 2. 增加参数校验和异常处理
 * 3. 释放原生 PyTorch 对象，避免内存泄漏
 * 4. 实现标准的参数重置方法
 * 5. 完善 propagate 调用逻辑
 */
public class TAGConv extends MessagePassing {
    private List<LinearImpl> lins;
    private int K; // 跳数
    private long inChannels;
    private long outChannels;
    private boolean isReleased = false; // 内存释放标记

    /**
     * 构造函数
     * @param inChannels 输入特征维度
     * @param outChannels 输出特征维度
     * @param K 传播跳数（K≥0）
     */
    public TAGConv(long inChannels, long outChannels, int K) {
        super("add"); // 聚合方式：求和

        // 参数合法性校验
        if (inChannels <= 0) {
            throw new IllegalArgumentException("输入通道数必须大于0: " + inChannels);
        }
        if (outChannels <= 0) {
            throw new IllegalArgumentException("输出通道数必须大于0: " + outChannels);
        }
        if (K < 0) {
            throw new IllegalArgumentException("跳数K必须≥0: " + K);
        }

        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.K = K;
        this.lins = new ArrayList<>();

        // 为每一跳创建线性层（0到K）
        for (int k = 0; k <= K; k++) {
            LinearImpl lin = new LinearImpl(inChannels, outChannels);
            this.lins.add(lin);
            register_module("lin_" + k, lin); // 注册到模块，支持参数管理
        }
    }

    /**
     * 前向传播核心逻辑
     * @param x 节点特征 [num_nodes, in_channels]
     * @param edge_index 边索引 [2, num_edges]
     * @return 输出特征 [num_nodes, out_channels]
     */
    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        checkReleased();
        validateInput(x, edge_index);

        long numNodes = x.size(0); // 节点数量
        Tensor out = lins.get(0).forward(x).clone(); // k=0: 自身特征变换

        Tensor currentX = x.clone(); // 当前传播的特征
        for (int k = 1; k <= K; k++) {
            // 传播一次：聚合邻居特征
            currentX = propagate(edge_index, currentX,numNodes); //numNodes
            // 线性变换并累加
            Tensor transformed = lins.get(k).forward(currentX);
            out = out.add(transformed);

            // 释放临时张量，避免内存泄漏
            transformed.close();
        }

        // 释放临时张量
        currentX.close();
        return out;
    }

    /**
     * 消息函数：TAGConv 的消息就是邻居特征本身
     * 严格匹配 MessagePassing 基类签名
     */
    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        checkReleased();
        return x_j.clone(); // 返回邻居特征（clone 避免原张量被意外修改）
    }

    /**
     * 重置层参数（符合 PyTorch 标准）
     */
    public void reset_parameters() {
        checkReleased();
        for (LinearImpl lin : lins) {
            lin.reset_parameters(); // 重置线性层权重和偏置
        }
    }

    /**
     * 输入合法性校验
     */
    private void validateInput(Tensor x, Tensor edge_index) {
        if (x == null || edge_index == null) {
            throw new NullPointerException("节点特征/边索引不能为空");
        }
        if (x.dim() != 2) {
            throw new IllegalArgumentException("节点特征必须是2维张量，当前维度：" + x.dim());
        }
        if (x.size(1) != inChannels) {
            throw new IllegalArgumentException(
                    "节点特征维度不匹配，期望：" + inChannels + "，实际：" + x.size(1)
            );
        }
        if (edge_index.dim() != 2 || edge_index.size(0) != 2) {
            throw new IllegalArgumentException("边索引必须是 [2, num_edges] 形状");
        }
    }

    /**
     * 检查是否已释放资源
     */
    private void checkReleased() {
        if (isReleased) {
            throw new IllegalStateException("TAGConv 已释放资源，无法继续使用");
        }
    }

    /**
     * 释放原生 PyTorch 对象（关键：避免内存泄漏）
     */
    @Override
    public void close() {
        if (!isReleased) {
            // 释放所有线性层
            for (LinearImpl lin : lins) {
                if (lin != null) {
                    lin.close();
                }
            }
            lins.clear();
            // 释放基类资源
            super.close();
            isReleased = true;
        }
    }

    public Tensor propagate(Tensor edge_index, Tensor x, long numNodes) {
        checkReleased();
        // 1. 获取边的源节点（邻居）和目标节点
        Tensor row = edge_index.select(0, 0); // 目标节点
        Tensor col = edge_index.select(0, 1); // 源节点（邻居）

        // 2. 提取邻居特征 x_j
        Tensor x_j = x.index_select(0, col);
        Tensor x_i = x.index_select(0, row); // TAGConv 不需要，但保留接口

        // 3. 生成消息（调用子类实现）
        Tensor msg = message(x_j, x_i, edge_index, null, numNodes);

        // 4. 聚合消息到目标节点
        Tensor out = aggregate(msg, row, numNodes);

        // 释放临时张量
        row.close();
        col.close();
        x_j.close();
        x_i.close();
        msg.close();

        return out;
    }

    // Getter 方法
    public int getK() { return K; }
    public long getInChannels() { return inChannels; }
    public long getOutChannels() { return outChannels; }
    public boolean isReleased() { return isReleased; }
}

//package org.bytedeco.pytorch.geometric.nn.conv;//package org.gnn.framework.layers;
//
//import org.bytedeco.pytorch.*;
////import org.gnn.framework.nn.org.bytedeco.pytorch.geometric.nn.conv.MessagePassing;
//import java.util.ArrayList;
//import java.util.List;
//
//public class TAGConv extends MessagePassing {
//    private List<LinearImpl> lins;
//    private int K;
//
//    public TAGConv(long inChannels, long outChannels, int K) {
//        super("add");
//        this.K = K;
//        this.lins = new ArrayList<>();
//
//        // 为每一跳 k 创建一个线性层
//        for (int k = 0; k <= K; k++) {
//            LinearImpl lin = new LinearImpl(inChannels, outChannels);
//            this.lins.add(lin);
//            register_module("lin_" + k, lin);
//        }
//    }
//
//    @Override
//    public Tensor forward(Tensor x, Tensor edge_index) {
//        // 归一化部分略 (假设 norm 已经在 edge_weight 中或简化处理)
//        // 此处简化：仅做特征传递
//
//        Tensor out = lins.get(0).forward(x); // k=0 path
//
//        Tensor currentX = x;
//        for (int k = 1; k <= K; k++) {
//            // 传播一次 (相当于乘一次 A)
//            currentX = propagate(edge_index, currentX);
//
//            // 变换并累加
//            Tensor transformed = lins.get(k).forward(currentX);
//            out = out.add(transformed);
//        }
//        return out;
//    }
//    /**
//     * 必须匹配基类签名：(x_j, x_i, edge_index, edge_attr)
//     * 哪怕 SAGE 只需要 x_j，参数也必须写全！
//     */
//    @Override
//    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
//        // GraphSAGE 的 message 就是邻居特征本身
//        // 如果以后要支持带权重的 SAGE，可以在这里处理 edge_attr
//        return x_j;
//    }
//}