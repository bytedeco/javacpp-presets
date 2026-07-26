package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.*;
import org.bytedeco.pytorch.nn.modules.container.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;

import  org.bytedeco.pytorch.global.torch;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.container.SequentialImpl;
import static org.bytedeco.pytorch.geometric.utils.AggrUtils.scatter_softmax;
import static org.bytedeco.pytorch.geometric.utils.TensorToolkit.ensure2D;

/**
 * 最终修复版 PointTransformerConv：
 * 1. 适配 pos 为 null 时的维度逻辑
 * 2. 新增 posDim 参数，确保 posNN 输入维度动态匹配
 * 3. 修复 delta 与 q_i/k_j 的维度对齐
 */
public class PointTransformerConv extends MessagePassing {
    private LinearImpl linQ, linK, linV; // Q/K/V 线性层
    private Module posNN;                // 位置编码网络
    private Module attnNN;               // 注意力映射网络
    private long inChannels;             // 特征输入维度
    private long outChannels;            // 特征输出维度
    private long posDim;                 // 位置编码维度（关键：动态适配）
    private boolean isReleased = false;

    /**
     * 构造函数：新增 posDim 参数，指定位置编码维度
     * @param inChannels 特征输入维度（x 的通道数）
     * @param outChannels 特征输出维度
     * @param posDim 位置编码维度（pos 的通道数，如点云为3）
     * @param posNN 位置编码网络（输入维度必须 = posDim）
     * @param attnNN 注意力映射网络（输入维度必须 = outChannels）
     */
    public PointTransformerConv(long inChannels, long outChannels, long posDim, Module posNN, Module attnNN) {
        super("add"); // 聚合方式：add

        // 严格参数校验
        if (inChannels <= 0) throw new IllegalArgumentException("inChannels 必须>0: " + inChannels);
        if (outChannels <= 0) throw new IllegalArgumentException("outChannels 必须>0: " + outChannels);
        if (posDim <= 0) throw new IllegalArgumentException("posDim 必须>0: " + posDim);
        if (posNN == null) throw new IllegalArgumentException("posNN 不能为空");
        if (attnNN == null) throw new IllegalArgumentException("attnNN 不能为空");

        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.posDim = posDim;
        this.posNN = posNN;
        this.attnNN = attnNN;

        // 初始化 Q/K/V 线性层（in→out）
        this.linQ = new LinearImpl(inChannels, outChannels);
        this.linK = new LinearImpl(inChannels, outChannels);
        this.linV = new LinearImpl(inChannels, outChannels);

        // 注册子模块
        register_module("lin_q", linQ);
        register_module("lin_k", linK);
        register_module("lin_v", linV);
        register_module("pos_nn", posNN);
        register_module("attn_nn", attnNN);
    }

    // ========== 标准 forward 接口 ==========
    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        checkReleased();
        // pos 为 null 时，使用 x 的前 posDim 维作为位置编码（保证维度匹配）
        Tensor pos = x.narrow(1, 0, (int) posDim).clone();
        Tensor output = forward(x, pos, edge_index);
        pos.close(); // 释放临时 pos
        return output;
    }

    /**
     * 核心 forward：x(特征), pos(位置), edge_index(边索引)
     */
    public Tensor forward(Tensor x, Tensor pos, Tensor edge_index) {
        checkReleased();
        validateInputs(x, pos, edge_index);

        long N = x.size(0);
        // 1. 线性变换得到 Q/K/V（[N, in] → [N, out]）
        Tensor xQ = linQ.forward(x);
        Tensor xK = linK.forward(x);
        Tensor xV = linV.forward(x);

        // 2. 执行消息传递并聚合
        Tensor output = propagate(edge_index, xQ, xK, xV, pos);

        // 释放临时张量
        xQ.close();
        xK.close();
        xV.close();

        return output;
    }

    // ========== 核心 propagate 实现：维度对齐修复 ==========
//    @Override
    public Tensor propagate(Tensor edge_index, Tensor... args) {
        checkReleased();
        if (args.length < 4) throw new IllegalArgumentException("需要传入 xQ, xK, xV, pos");

        Tensor xQ = args[0];
        Tensor xK = args[1];
        Tensor xV = args[2];
        Tensor pos = args[3];

        long N = xQ.size(0);
        Tensor sourceIdx = edge_index.select(0, 0);   // 源节点 j: [E]
        Tensor targetIdx = edge_index.select(0, 1);   // 目标节点 i: [E]
        long E = sourceIdx.size(0);                   // 边数

        // 提取 Q_i (目标节点)、K_j/V_j (源节点) → [E, outChannels]
        Tensor q_i = xQ.index_select(0, targetIdx);
        Tensor k_j = xK.index_select(0, sourceIdx);
        Tensor v_j = xV.index_select(0, sourceIdx);

        // 提取位置编码 → [E, posDim]
        Tensor pos_i = pos.index_select(0, targetIdx);
        Tensor pos_j = pos.index_select(0, sourceIdx);

        Tensor output = null;
        try {
            // 1. 相对位置编码：pos_i - pos_j → [E, posDim]
            Tensor rel_pos = pos_i.sub(pos_j);
            // 位置编码网络：[E, posDim] → [E, outChannels]（保证与 q_i/k_j 维度匹配）
            Tensor delta = posNN.asSequential().forward(rel_pos);

            // 2. 注意力输入：q_i - k_j + delta → [E, outChannels]（维度强制对齐）
            Tensor qkDiff = q_i.sub(k_j);
            // 安全加法：确保 delta 维度与 qkDiff 一致
            if (delta.size(1) != qkDiff.size(1)) {
                throw new IllegalArgumentException(
                        "posNN 输出维度必须 = outChannels: " + delta.size(1) + " vs " + qkDiff.size(1)
                );
            }
            Tensor attnInput = qkDiff.add(delta);

            // 3. 注意力映射：[E, outChannels] → [E, outChannels]
            Tensor attn = attnNN.asSequential().forward(attnInput);

            // 4. 局部 Softmax 归一化（按目标节点聚合）
            attn = scatter_softmax(attn, targetIdx, N);

            // 5. 消息加权：attn * (v_j + delta) → [E, outChannels]
            Tensor msg = attn.mul(v_j.add(delta));

            // 6. 聚合到目标节点（add 方式）
            output = aggregate(msg, targetIdx, N);

        } finally {
            // 释放所有临时张量
            sourceIdx.close();
            targetIdx.close();
            q_i.close();
            k_j.close();
            v_j.close();
            pos_i.close();
            pos_j.close();
        }

        return output;
    }

    // ========== 聚合逻辑：add 聚合 ==========
    public Tensor aggregate(Tensor msg, Tensor targetIdx, long numNodes) {
        // 创建输出张量：[numNodes, outChannels]
        Tensor output = torch.zeros(new long[]{numNodes, outChannels}, msg.options());
        // 按目标节点聚合
        output.index_add_(0, targetIdx, msg);
        return output;
    }

    // ========== 输入校验：维度严格校验 ==========
    private void validateInputs(Tensor x, Tensor pos, Tensor edge_index) {
        if (x == null) throw new IllegalArgumentException("节点特征 x 不能为空");
        if (edge_index == null) throw new IllegalArgumentException("边索引 edge_index 不能为空");
        if (x.dim() != 2 || x.size(1) != inChannels) {
            throw new IllegalArgumentException("x 必须是 [N, " + inChannels + "] 形状，当前：" + x.size(0) + "x" + x.size(1));
        }
        if (edge_index.dim() != 2 || edge_index.size(0) != 2) {
            throw new IllegalArgumentException("edge_index 必须是 [2, E] 形状，当前：" + edge_index.size(0) + "x" + edge_index.size(1));
        }
        if (pos != null) {
            if (pos.dim() != 2 || pos.size(1) != posDim) {
                throw new IllegalArgumentException("pos 必须是 [N, " + posDim + "] 形状，当前：" + pos.size(0) + "x" + pos.size(1));
            }
            if (pos.size(0) != x.size(0)) {
                throw new IllegalArgumentException("pos 节点数必须与 x 一致：" + pos.size(0) + " vs " + x.size(0));
            }
        }
    }

    // ========== 资源管理 ==========
    private void checkReleased() {
        if (isReleased) throw new IllegalStateException("PointTransformerConv 已释放资源");
    }

    @Override
    public void close() {
        if (!isReleased) {
            if (linQ != null) linQ.close();
            if (linK != null) linK.close();
            if (linV != null) linV.close();
            if (posNN != null) posNN.close();
            if (attnNN != null) attnNN.close();
            super.close();
            isReleased = true;
        }
    }

    // ========== 占位方法 ==========
    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        return x_j;
    }

    // Getter 方法
    public long getInChannels() { return inChannels; }
    public long getOutChannels() { return outChannels; }
    public long getPosDim() { return posDim; }
    public boolean isReleased() { return isReleased; }
}

//public class PointTransformerConv extends MessagePassing {
//    private LinearImpl linQ, linK, linV; // Q/K/V 线性层
//    private Module posNN;                // 位置编码网络
//    private Module attnNN;               // 注意力映射网络
//    private long outChannels;
//    private boolean isReleased = false;
//
//    public PointTransformerConv(long inChannels, long outChannels, Module posNN, Module attnNN) {
//        super("add"); // 聚合方式：add
//
//        // 参数校验
//        if (inChannels <= 0) throw new IllegalArgumentException("inChannels 必须>0: " + inChannels);
//        if (outChannels <= 0) throw new IllegalArgumentException("outChannels 必须>0: " + outChannels);
//        if (posNN == null) throw new IllegalArgumentException("posNN 不能为空");
//        if (attnNN == null) throw new IllegalArgumentException("attnNN 不能为空");
//
//        this.outChannels = outChannels;
//
//        // 初始化 Q/K/V 线性层
//        this.linQ = new LinearImpl(inChannels, outChannels);
//        this.linK = new LinearImpl(inChannels, outChannels);
//        this.linV = new LinearImpl(inChannels, outChannels);
//
//        // 位置编码和注意力网络
//        this.posNN = posNN;
//        this.attnNN = attnNN;
//
//        // 注册子模块（便于参数管理/释放）
//        register_module("lin_q", linQ);
//        register_module("lin_k", linK);
//        register_module("lin_v", linV);
//        register_module("pos_nn", posNN);
//        register_module("attn_nn", attnNN);
//    }
//
//    // ========== 修正方法重载：匹配标准接口 ==========
//    @Override
//    public Tensor forward(Tensor x, Tensor edge_index) {
//        checkReleased();
//        return forward(x, null, edge_index); // pos 默认为 null 时使用 x 作为位置
//    }
//
//    /**
//     * 标准 forward 接口：x(特征), pos(位置), edge_index(边索引)
//     */
//    public Tensor forward(Tensor x, Tensor pos, Tensor edge_index) {
//        checkReleased();
//        validateInputs(x, pos, edge_index);
//
//        long N = x.size(0);
//        // 若 pos 为 null，使用 x 作为位置编码
//        Tensor posInput = (pos == null) ? x.clone() : pos;
//
//        // 1. 线性变换得到 Q/K/V
//        Tensor xQ = linQ.forward(x);
//        Tensor xK = linK.forward(x);
//        Tensor xV = linV.forward(x);
//
//        // 2. 执行消息传递并聚合
//        Tensor output = propagate(edge_index, xQ, xK, xV, posInput);
//
//        // 释放临时张量
//        xQ.close();
//        xK.close();
//        xV.close();
//        if (posInput != pos) posInput.close(); // 仅释放克隆的 pos
//
//        return output;
//    }
//
//    // ========== 核心 propagate 实现：修正逻辑 + 资源管理 ==========
////    @Override
//    public Tensor propagate(Tensor edge_index, Tensor... args) {
//        checkReleased();
//        if (args.length < 4) throw new IllegalArgumentException("需要传入 xQ, xK, xV, pos");
//
//        Tensor xQ = args[0];
//        Tensor xK = args[1];
//        Tensor xV = args[2];
//        Tensor pos = args[3];
//
//        long N = xQ.size(0);
//        Tensor sourceIdx = edge_index.select(0, 0);   // 源节点 (j)
//        Tensor targetIdx = edge_index.select(0, 1);   // 目标节点 (i)
//
//        // 提取 Q_i (目标节点)、K_j/V_j (源节点)
//        Tensor q_i = xQ.index_select(0, targetIdx);
//        Tensor k_j = xK.index_select(0, sourceIdx);
//        Tensor v_j = xV.index_select(0, sourceIdx);
//
//        // 提取位置编码
//        Tensor pos_i = pos.index_select(0, targetIdx);
//        Tensor pos_j = pos.index_select(0, sourceIdx);
//
//        Tensor output = null;
//        try {
//            // 1. 相对位置编码：pos_i - pos_j
//            Tensor rel_pos = pos_i.sub(pos_j);
//            // 修正：适配任意 Module 类型（非仅 Sequential）
//            Tensor delta = posNN.asSequential().forward(rel_pos);
//
//            // 2. 注意力计算：gamma(q_i - k_j + delta)
//            Tensor attnInput = q_i.sub(k_j).add(delta);
//            Tensor attn = attnNN.asSequential().forward(attnInput);
//
//            // 3. 局部 Softmax 归一化（按目标节点聚合）
//            attn = scatter_softmax(attn, targetIdx, N);
//
//            // 4. 消息加权：attn * (v_j + delta)
//            Tensor msg = attn.mul(v_j.add(delta));
//
//            // 5. 聚合到目标节点（add 聚合）
//            output = aggregate(msg, targetIdx, N);
//
//        } finally {
//            // 释放所有临时张量
//            sourceIdx.close();
//            targetIdx.close();
//            q_i.close();
//            k_j.close();
//            v_j.close();
//            pos_i.close();
//            pos_j.close();
//        }
//
//        return output;
//    }
//
//    // ========== 聚合逻辑：补全核心实现 ==========
//    public Tensor aggregate(Tensor msg, Tensor targetIdx, long numNodes) {
//        // 创建输出张量（初始化全0）
//        Tensor output = torch.zeros(new long[]{numNodes, outChannels}, msg.options());
//        // 按目标节点聚合（add 方式）
//        output.index_add_(0, targetIdx, msg);
//        return output;
//    }
//
//    // ========== 参数校验：避免非法输入 ==========
//    private void validateInputs(Tensor x, Tensor pos, Tensor edge_index) {
//        if (x == null) throw new IllegalArgumentException("节点特征 x 不能为空");
//        if (edge_index == null) throw new IllegalArgumentException("边索引 edge_index 不能为空");
//        if (x.dim() != 2) throw new IllegalArgumentException("x 必须是 2 维张量 [N, C]");
//        if (edge_index.dim() != 2 || edge_index.size(0) != 2) {
//            throw new IllegalArgumentException("edge_index 必须是 [2, E] 形状");
//        }
//
//        // 校验 pos 维度（若不为 null）
//        if (pos != null) {
//            if (pos.dim() != 2) throw new IllegalArgumentException("pos 必须是 2 维张量 [N, D]");
//            if (pos.size(0) != x.size(0)) {
//                throw new IllegalArgumentException("pos 节点数必须与 x 一致：" + pos.size(0) + " vs " + x.size(0));
//            }
//        }
//    }
//
//    // ========== 资源管理 ==========
//    private void checkReleased() {
//        if (isReleased) throw new IllegalStateException("PointTransformerConv 已释放资源");
//    }
//
//    @Override
//    public void close() {
//        if (!isReleased) {
//            // 释放线性层
//            if (linQ != null) linQ.close();
//            if (linK != null) linK.close();
//            if (linV != null) linV.close();
//            // 释放子模块
//            if (posNN != null) posNN.close();
//            if (attnNN != null) attnNN.close();
//            super.close();
//            isReleased = true;
//        }
//    }
//
//    // ========== 占位方法：避免父类调用 ==========
//    @Override
//    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
//        return x_j; // 逻辑已在 propagate 中实现
//    }
//
//    // Getter 方法
//    public long getOutChannels() { return outChannels; }
//    public boolean isReleased() { return isReleased; }
//}


//public class PointTransformerConv extends MessagePassing {
//    private LinearImpl linQ, linK, linV; // 线性变换矩阵 (W_q, W_k, W_v)
//    private Module posNN;                // 位置编码网络 delta
//    private Module attnNN;               // 注意力映射网络 gamma
//    private long outChannels;
//
//    public PointTransformerConv(long inChannels, long outChannels, Module posNN, Module attnNN) {
//        super("add");
//        this.outChannels = outChannels;
//
//        // 1. 定义 Q, K, V 线性层 (LinearImpl)
//        this.linQ = new LinearImpl(inChannels, outChannels);
//        this.linK = new LinearImpl(inChannels, outChannels);
//        this.linV = new LinearImpl(inChannels, outChannels);
//
//        // 2. 位置编码和注意力映射网络
//        this.posNN = posNN;
//        this.attnNN = attnNN;
//
//        register_module("lin_q", linQ);
//        register_module("lin_k", linK);
//        register_module("lin_v", linV);
//        if (posNN != null) register_module("pos_nn", posNN);
//        if (attnNN != null) register_module("attn_nn", attnNN);
//    }
//
//    @Override
//    public Tensor forward(Tensor x, Tensor edge_index) {
//        return forward(x, edge_index, null);
//    }
//    
//    public Tensor forward(Tensor x, Tensor pos, Tensor edge_index) {
//        long N = x.size(0);
//
//        // 1. 线性变换得到 Q, K, V
//        Tensor xQ = linQ.forward(x);
//        Tensor xK = linK.forward(x);
//        Tensor xV = linV.forward(x);
//
//        // 2. 执行消息传递
//        return propagate(edge_index, xQ, xK, xV, pos);
//    }
//
//    /**
//     * 重写 propagate 以实现向量自注意力逻辑
//     */
//    public Tensor propagate(Tensor edge_index, Tensor xQ, Tensor xK, Tensor xV, Tensor pos) {
//        long N = xQ.size(0);
//        Tensor sourceIdx = edge_index.select(0, 0);
//        Tensor targetIdx = edge_index.select(0, 1);
//
//        // 提取 i (目标) 和 j (源) 的分量
//        Tensor q_i = xQ.index_select(0, targetIdx);
//        Tensor k_j = xK.index_select(0, sourceIdx);
//        Tensor v_j = xV.index_select(0, sourceIdx);
//
//        Tensor pos_i = pos.index_select(0, targetIdx);
//        Tensor pos_j = pos.index_select(0, sourceIdx);
//
//        // --- 3. 相对位置编码 (delta) ---
//        Tensor rel_pos = pos_i.sub(pos_j); // pos_i - pos_j
//        Tensor delta = posNN.asSequential().forward(rel_pos);
//
//        // --- 4. 向量注意力计算 (gamma) ---
//        // attn = gamma(q_i - k_j + delta)
//        Tensor attnInput = q_i.sub(k_j).add(delta);
//        Tensor attn = attnNN.asSequential().forward(attnInput);
//
//        // 局部 Softmax 归一化 (向量级别)
//        attn = scatter_softmax(attn, targetIdx, N);
//
//        // --- 5. 消息加权 ---
//        // message = attn * (v_j + delta)
//        Tensor msg = attn.mul(v_j.add(delta));
//
//        return aggregate(msg, targetIdx, N);
//    }
//
//    @Override
//    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
//        // 逻辑已在 propagate 中展开
//        return x_j;
//    }
//}
