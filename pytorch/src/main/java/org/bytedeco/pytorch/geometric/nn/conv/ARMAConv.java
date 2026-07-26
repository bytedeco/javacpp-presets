package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import java.util.ArrayList;
import java.util.List;
import static org.bytedeco.pytorch.global.torch.*;

public class ARMAConv extends MessagePassing {
    private List<LinearImpl> initLins = new ArrayList<>();
    private List<LinearImpl> rootLins = new ArrayList<>();
    private int numStacks;      // 栈数量（必须>0）
    private int numLayers;      // 每层栈的迭代层数（必须≥0）
    private long inChannels;    // 输入通道数（必须>0）
    private long outChannels;   // 输出通道数（必须>0）
    private boolean isReleased = false; // 资源释放标记

    /**
     * 构造函数：严格校验所有参数合法性
     */
    public ARMAConv(long inChannels, long outChannels, int numStacks, int numLayers) {
        super("add");

        // 1. 严格参数校验（核心：避免非法参数传入 LinearImpl）
        if (inChannels <= 0) {
            throw new IllegalArgumentException("输入通道数必须>0: " + inChannels);
        }
        if (outChannels <= 0) {
            throw new IllegalArgumentException("输出通道数必须>0: " + outChannels);
        }
        if (numStacks <= 0) {
            throw new IllegalArgumentException("栈数量numStacks必须>0: " + numStacks);
        }
        if (numLayers < 0) {
            throw new IllegalArgumentException("迭代层数numLayers必须≥0: " + numLayers);
        }

        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.numStacks = numStacks;
        this.numLayers = numLayers;

        // 2. 初始化线性层（仅当参数合法时创建）
        try {
            for (int i = 0; i < numStacks; i++) {
                LinearImpl initLin = new LinearImpl(inChannels, outChannels);
                LinearImpl rootLin = new LinearImpl(outChannels, outChannels);
                initLins.add(initLin);
                rootLins.add(rootLin);
                register_module("init_" + i, initLin);
                register_module("root_" + i, rootLin);
            }
        } catch (Exception e) {
            // 创建失败时释放已创建的线性层，避免内存泄漏
            close();
            throw new RuntimeException("初始化线性层失败：" + e.getMessage(), e);
        }
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        checkReleased();
        // 核心修复：提前校验输入维度，在调用 Linear.forward 前抛出异常
        validateInput(x, edge_index);

        Tensor globalOut = null;

        for (int s = 0; s < numStacks; s++) {
            // 初始化 X_0 = ReLU(initLin(X))
            Tensor out = initLins.get(s).forward(x).clone();
            out = relu(out);

            // 多层迭代更新
            for (int t = 0; t < numLayers; t++) {
                Tensor aggr = propagate(edge_index, out);
                Tensor root = rootLins.get(s).forward(out);

                // 计算新输出并释放旧张量
                Tensor newOut = relu(aggr.add(root));
                aggr.close();
                root.close();
                out.close();
                out = newOut;
            }

            // 累加栈输出
            if (globalOut == null) {
                globalOut = out;
            } else {
                globalOut = globalOut.add(out);
                out.close(); // 释放已累加的栈输出
            }
        }

        // 求平均并释放临时张量
        Tensor result = globalOut.div(new Scalar(numStacks));
        globalOut.close();
        return result;
    }

    /**
     * 自定义 propagate 方法：适配 ARMA 聚合逻辑
     */
    public Tensor propagate(Tensor edge_index, Tensor x) {
        long numNodes = x.size(0);
        Tensor row = edge_index.select(0, 0); // 目标节点
        Tensor col = edge_index.select(0, 1); // 源节点（邻居）

        // 提取邻居特征并生成消息
        Tensor x_j = x.index_select(0, col);
        Tensor msg = message(x_j, null, edge_index, null, numNodes);

        // 聚合消息到目标节点
        Tensor aggr = scatter_add(
                zeros(new long[]{numNodes, x.size(1)}, x.options()),
                0,
                row.unsqueeze(1).expand(msg.sizes().vec().get()),
                msg
        );

        // 释放临时张量
        row.close();
        col.close();
        x_j.close();
        msg.close();

        return aggr;
    }

    /**
     * 输入合法性校验（核心修复：提前校验，统一异常类型）
     */
    private void validateInput(Tensor x, Tensor edge_index) {
        // 1. 空值校验
        if (x == null) {
            throw new IllegalArgumentException("节点特征 x 不能为空");
        }
        if (edge_index == null) {
            throw new IllegalArgumentException("边索引 edge_index 不能为空");
        }

        // 2. 维度校验
        if (x.dim() != 2) {
            throw new IllegalArgumentException("节点特征必须是2维张量，当前维度：" + x.dim());
        }
        // 核心修复：提前校验输入通道数，避免触发底层矩阵乘法异常
        if (x.size(1) != inChannels) {
            throw new IllegalArgumentException(
                    String.format("节点特征维度不匹配！期望输入通道数：%d，实际：%d", inChannels, x.size(1))
            );
        }
        if (edge_index.dim() != 2 || edge_index.size(0) != 2) {
            throw new IllegalArgumentException(
                    String.format("边索引必须是 [2, num_edges] 形状，当前：%dx%d", edge_index.size(0), edge_index.size(1))
            );
        }
    }

    /**
     * 检查资源是否已释放
     */
    private void checkReleased() {
        if (isReleased) {
            throw new IllegalStateException("ARMAConv 已释放资源，无法继续使用");
        }
    }

    /**
     * 重置参数（符合 PyTorch 规范）
     */
    public void resetParameters() {
        checkReleased();
        for (LinearImpl lin : initLins) {
            lin.reset_parameters();
        }
        for (LinearImpl lin : rootLins) {
            lin.reset_parameters();
        }
    }

    /**
     * 释放所有原生资源
     */
    @Override
    public void close() {
        if (!isReleased) {
            // 释放线性层
            for (LinearImpl lin : initLins) {
                if (lin != null) {
                    try { lin.close(); } catch (Exception e) {}
                }
            }
            for (LinearImpl lin : rootLins) {
                if (lin != null) {
                    try { lin.close(); } catch (Exception e) {}
                }
            }
            // 清空列表
            initLins.clear();
            rootLins.clear();
            // 释放基类资源
            super.close();
            isReleased = true;
        }
    }

    /**
     * 消息函数：匹配基类签名
     */
    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        checkReleased();
        return x_j;
    }

    // Getter 方法
    public int getNumStacks() { return numStacks; }
    public int getNumLayers() { return numLayers; }
    public long getInChannels() { return inChannels; }
    public long getOutChannels() { return outChannels; }
    public boolean isReleased() { return isReleased; }
}


//public class ARMAConv extends MessagePassing {
//    private List<LinearImpl> initLins = new ArrayList<>();
//    private List<LinearImpl> rootLins = new ArrayList<>();
//    private int numStacks;  // 栈数量
//    private int numLayers;  // 每层栈的迭代层数
//    
//    private long inChannels;    // 输入通道数
//    private long outChannels;   // 输出通道数
//    private boolean isReleased = false; // 资源释放标记
//    // 典型参数: in, out, num_stacks, num_layers, shared_weights, dropout
//    public ARMAConv(long inChannels, long outChannels, int numStacks, int numLayers) {
//        super("add");
//        // 1. 严格参数校验（核心修复：避免非法参数传入 LinearImpl）
//        if (inChannels <= 0) {
//            throw new IllegalArgumentException("输入通道数必须>0: " + inChannels);
//        }
//        if (outChannels <= 0) {
//            throw new IllegalArgumentException("输出通道数必须>0: " + outChannels);
//        }
//        if (numStacks <= 0) {
//            throw new IllegalArgumentException("栈数量numStacks必须>0: " + numStacks);
//        }
//        if (numLayers < 0) {
//            throw new IllegalArgumentException("迭代层数numLayers必须≥0: " + numLayers);
//        }
//        this.numStacks = numStacks;
//        this.numLayers = numLayers;
//        this.inChannels = inChannels;
//        this.outChannels = outChannels;
//        
//
//        for (int i = 0; i < numStacks; i++) {
//            LinearImpl init = new LinearImpl(inChannels, outChannels);
//            LinearImpl root = new LinearImpl(outChannels, outChannels);
//            initLins.add(init);
//            rootLins.add(root);
//            register_module("init_" + i, init);
//            register_module("root_" + i, root);
//        }
//    }
//
//    public Tensor forward(Tensor x, Tensor edge_index) {
//        // ARMA 包含多个独立的 Stack，最后求平均
//        Tensor globalOut = null;
//
//        for (int s = 0; s < numStacks; s++) {
//            // 1. 初始化 X_0 = ReLU(lin(X))
//            Tensor out = initLins.get(s).forward(x);
//            out = relu(out);
//
//            // 2. 递归更新 X_t = ReLU(Prop(X_{t-1}) + W * X_{t-1}) (简化版)
//            for (int t = 0; t < numLayers; t++) {
//                Tensor aggr = propagate(edge_index, out);
//                Tensor root = rootLins.get(s).forward(out);
//
//                // Skip connection logic specific to ARMA (simplified)
//                out = relu(aggr.add(root));
//            }
//
//            if (globalOut == null) globalOut = out;
//            else globalOut = globalOut.add(out);
//        }
//
//        // Average over stacks
//        // 重点：new Scalar
//        return globalOut.div(new Scalar(numStacks));
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
//
//    private void checkReleased() {
//        if (isReleased) {
//            throw new IllegalStateException("ARMAConv 已释放资源，无法继续使用");
//        }
//    }
//
//    /**
//     * 重置参数（符合 PyTorch 规范）
//     */
//    public void resetParameters() {
//        checkReleased();
//        for (LinearImpl lin : initLins) {
//            lin.reset_parameters();
//        }
//        for (LinearImpl lin : rootLins) {
//            lin.reset_parameters();
//        }
//    }
//
//    /**
//     * 释放原生资源（避免内存泄漏）
//     */
//    @Override
//    public void close() {
//        if (!isReleased) {
//            // 释放所有线性层
//            for (LinearImpl lin : initLins) {
//                if (lin != null) lin.close();
//            }
//            for (LinearImpl lin : rootLins) {
//                if (lin != null) lin.close();
//            }
//            initLins.clear();
//            rootLins.clear();
//            // 释放基类资源
//            super.close();
//            isReleased = true;
//        }
//    }
//
//    public int getNumStacks() { return numStacks; }
//    public int getNumLayers() { return numLayers; }
//    public long getInChannels() { return inChannels; }
//    public long getOutChannels() { return outChannels; }
//    public boolean isReleased() { return isReleased; }
//}