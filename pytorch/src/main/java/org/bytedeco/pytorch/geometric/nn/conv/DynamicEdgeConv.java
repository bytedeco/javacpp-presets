package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;

/**
 * 动态边缘卷积层（DynamicEdgeConv）
 * 核心逻辑：动态构建k-NN图 + EdgeConv消息传递（(x_i, x_j-x_i) 拼接 + 非线性变换 + 聚合）
 * 支持批次隔离（不同批次节点不构建k-NN连接）
 */
public class DynamicEdgeConv extends MessagePassing {
    private Module nn;          // 非线性变换网络（支持任意Module类型：Linear/Sequential）
    private int k;              // k-NN邻居数
    private long inChannels;    // 输入通道数

    /**
     * 构造方法：初始化DynamicEdgeConv
     * @param nn 非线性变换网络（输入维度=2*inChannels，输出维度=outChannels）
     * @param k k-NN邻居数（k≥1）
     * @param aggr 聚合方式（max/mean/add）
     */
    public DynamicEdgeConv(Module nn, int k, String aggr) {
        super(aggr);
        // 输入校验
        if (k < 1) {
            throw new IllegalArgumentException("k-NN邻居数k必须≥1，当前值：" + k);
        }
        if (aggr == null || (!aggr.equals("max") && !aggr.equals("mean") && !aggr.equals("add"))) {
            throw new IllegalArgumentException("仅支持聚合方式：max/mean/add，当前值：" + aggr);
        }

        this.nn = nn;
        this.k = k;

        // 注册模块到Module树（支持参数优化/设备迁移）
        if (nn != null) {
            register_module("nn", nn);
        }
    }

    /**
     * 前向传播（核心逻辑）
     * @param x 节点特征 [N, inChannels]
     * @param batch 批次向量 [N]（可选）
     * @return 聚合后特征 [N, outChannels]
     */
    public Tensor forward(Tensor x, Tensor batch) {
        // ========== 输入校验 ==========
        if (x.dim() != 2) {
            throw new IllegalArgumentException("x 必须是2维张量 [N, inChannels]，当前维度：" + x.dim());
        }
        long N = x.size(0); // 节点总数
        long C = x.size(1); // 输入通道数
        this.inChannels = C;

        // 关键防护：节点数必须 ≥2，且k ≤ N-1（避免topk越界）
        if (N <= 1) {
            throw new IllegalArgumentException("节点数N必须≥2，当前值：" + N);
        }
        int effectiveK = Math.min(this.k, (int) N - 1); // 动态调整k
        if (effectiveK != this.k) {
            System.out.println("警告：节点数不足，自动调整k从" + this.k + "到" + effectiveK);
        }

        // 批次向量校验 + 类型转换
        if (batch != null) {
            if (batch.dim() != 1 || batch.size(0) != N) {
                throw new IllegalArgumentException("batch必须是1维张量且长度等于节点数N，当前维度：" + batch.dim() + "，长度：" + batch.size(0));
            }
            batch = batch.to(torch.ScalarType.Long);
        }

        // ========== 设备/类型对齐（修正API参数） ==========
        if (this.nn != null) {
            this.nn.to(x.device(), x.scalar_type(),false);
        }

        // ========== 1. 动态构建k-NN图（批次隔离） ==========
        Tensor dists = torch.cdist(x, x); // [N, N] 欧式距离

        // 批次隔离：不同批次节点间距离设为无穷大
        if (batch != null) {
            Tensor batchExpandI = batch.view(-1, 1).repeat(new long[]{1, N});
            Tensor batchExpandJ = batch.view(1, -1).repeat(new long[]{N, 1});
            Tensor mask = batchExpandI.ne(batchExpandJ); // 不同批次为true
            Tensor inf = torch.tensor(Double.POSITIVE_INFINITY, x.options());
            dists = dists.where(mask.logical_not(), inf);

            // 释放临时张量
            batchExpandI.close();
            batchExpandJ.close();
            mask.close();
            inf.close();
        }

        // 获取effectiveK+1个最近邻（排除自身后取effectiveK个）
        T_TensorTensor_T topk = dists.topk(effectiveK + 1, -1, false, true);
        Tensor knnValues = topk.get0();
        Tensor knnIdx = topk.get1().narrow(-1, 1, effectiveK); // 排除自身

        // ========== 2. 构造edge_index [2, E] ==========
        Tensor targetIdx = torch.arange(new Scalar(N), knnIdx.options())
                .view(-1, 1)
                .repeat(new long[]{1, effectiveK})
                .view(-1);
        Tensor sourceIdx = knnIdx.contiguous().view(-1);
        Tensor edge_index = torch.stack(new TensorVector(sourceIdx, targetIdx), 0);

        // ========== 3. EdgeConv消息传递 ==========
        Tensor out = knn_propagate(edge_index, x);

        // ========== 释放临时张量 ==========
        dists.close();
        knnValues.close();
        knnIdx.close();
        targetIdx.close();
        sourceIdx.close();
        edge_index.close();
        topk.close();

        return out;
    }

    /**
     * 适配基类的forward方法（无批次向量）
     */
//    @Override
//    public Tensor forward(Tensor x, Tensor edge_index) {
//        return forward(x, null); // edge_index参数复用为batch，兼容基类签名
//    }

    /**
     * 自定义k-NN propagate方法（核心：兼容所有Module类型）
     */
    private Tensor knn_propagate(Tensor edge_index, Tensor x) {
        long N = x.size(0);
        Tensor sourceIdx = edge_index.select(0, 0);
        Tensor targetIdx = edge_index.select(0, 1);

        // 提取x_i（目标节点）和x_j（源节点/邻居）
        Tensor xi = x.index_select(0, targetIdx);
        Tensor xj = x.index_select(0, sourceIdx);

        // EdgeConv核心：拼接 (x_i, x_j - x_i) → [E, 2*inChannels]
        Tensor msgInput = torch.cat(new TensorVector(xi, xj.sub(xi)), -1);

        // 通过非线性网络变换（兼容Linear/Sequential等所有Module）
        Tensor msg;
        if (this.nn != null) {
            if(this.nn instanceof LinearImpl) {
                msg = ((LinearImpl) this.nn).forward(msgInput);
            } else if (this.nn instanceof SequentialImpl) {
                msg = ((SequentialImpl) this.nn).forward(msgInput);
            }else{
                msg = this.nn.asSequential().forward(msgInput);
            }
            // 其他Module类型直接调用forward（假设输入输出维度正确）
//            msg = this.nn.asSequential().forward(msgInput);
        } else {
            // 无nn时返回输入的前C维
            msg = msgInput.narrow(-1, 0, x.size(1));
        }

        // 聚合（max/mean/add）
        Tensor out = aggregate(msg, targetIdx, N);

        // 释放临时张量
        sourceIdx.close();
        targetIdx.close();
        xi.close();
        xj.close();
        msgInput.close();
        msg.close();

        return out;
    }

    /**
     * 基类message方法（仅适配签名）
     */
    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        return x_j;
    }

    /**
     * 重置参数（兼容Linear/Sequential）
     */
//    @Override
    public void reset_parameters() {
        if (this.nn != null) {
            if (this.nn instanceof LinearImpl) {
                ((LinearImpl) this.nn).reset_parameters();
            } else if (this.nn instanceof SequentialImpl) {
                ((SequentialImpl) this.nn).reset();
            }
        }
    }

    /**
     * 资源释放（避免JNI内存泄漏）
     */
    @Override
    public void close() {
        if (this.nn != null) {
            this.nn.close();
            this.nn = null;
        }
        super.close();
    }

    // 辅助方法
    public int getK() { return k; }
    public long getInChannels() { return inChannels; }
    public Module getNn() { return nn; }
}
/**
 * 动态边缘卷积层（DynamicEdgeConv）
 * 核心逻辑：动态构建k-NN图 + EdgeConv消息传递（(x_i, x_j-x_i) 拼接 + 非线性变换 + 聚合）
 * 支持批次隔离（不同批次节点不构建k-NN连接）
 */
//public class DynamicEdgeConv extends MessagePassing {
//    private Module nn;          // 非线性变换网络（如 Sequential(Linear+ReLU+Linear)）
//    private int k;              // k-NN邻居数
//    private long inChannels;    // 输入通道数（推导自nn，或通过构造方法传入）
//
//    /**
//     * 构造方法：初始化DynamicEdgeConv
//     * @param nn 非线性变换网络（输入维度=2*inChannels，输出维度=outChannels）
//     * @param k k-NN邻居数（k≥1）
//     * @param aggr 聚合方式（EdgeConv论文建议用"max"，可选"mean"/"add"）
//     */
//    public DynamicEdgeConv(Module nn, int k, String aggr) {
//        super(aggr); // 初始化MessagePassing基类，指定聚合方式
//        // 输入校验
//        if (k < 1) {
//            throw new IllegalArgumentException("k-NN邻居数k必须≥1，当前值：" + k);
//        }
//        if (aggr == null || (!aggr.equals("max") && !aggr.equals("mean") && !aggr.equals("add"))) {
//            throw new IllegalArgumentException("仅支持聚合方式：max/mean/add，当前值：" + aggr);
//        }
//
//        this.nn = nn;
//        this.k = k;
//
//        // 注册非线性网络到模块树（支持参数优化）
//        if (nn != null) {
//            register_module("nn", nn);
//        }
//    }
//
//    /**
//     * 前向传播（核心逻辑）
//     * @param x 节点特征 [N, inChannels]
//     * @param batch 批次向量 [N]（可选，值为0,1,...B-1，标识节点所属批次）
//     * @return 聚合后特征 [N, outChannels]
//     */
//    @Override
//    public Tensor forward(Tensor x, Tensor batch) {
//        // ========== 输入校验 ==========
//        if (x.dim() != 2) {
//            throw new IllegalArgumentException("x 必须是2维张量 [N, inChannels]，当前维度：" + x.dim());
//        }
//        long N = x.size(0); // 节点总数
//        long C = x.size(1); // 输入通道数
//        this.inChannels = C;
//
//        // 批次向量校验：维度为1，长度=N
//        if (batch != null) {
//            if (batch.dim() != 1 || batch.size(0) != N) {
//                throw new IllegalArgumentException("batch必须是1维张量且长度等于节点数N，当前维度：" + batch.dim() + "，长度：" + batch.size(0));
//            }
//            // 批次向量转为长整型
//            batch = batch.to(torch.ScalarType.Long);
//        }
//
//        // ========== 设备/类型对齐 ==========
//        if (this.nn != null) {
//            this.nn.to(x.device(), x.scalar_type(),false);
//        }
//
//        // ========== 1. 动态构建k-NN图（核心修复：批次隔离） ==========
//        // 计算节点间欧式距离 [N, N]
//        Tensor dists = torch.cdist(x, x);
//
//        // 批次隔离：不同批次节点间距离设为无穷大（禁止跨批次连接）
//        if (batch != null) {
//            // 构建批次掩码：batch_i != batch_j 时为true
//            Tensor batchExpandI = batch.view(-1, 1).repeat(new long[]{1, N});
//            Tensor batchExpandJ = batch.view(1, -1).repeat(new long[]{N, 1});
//            Tensor mask = batchExpandI.ne(batchExpandJ); // [N,N]，不同批次为true
//            // 无穷大值（与输入类型匹配）
//            Tensor inf = torch.tensor(Double.POSITIVE_INFINITY, x.options());
//            // 不同批次节点间距离设为无穷大
//            dists = dists.where(mask.logical_not(), inf);
//
//            // 释放临时张量
//            batchExpandI.close();
//            batchExpandJ.close();
//            mask.close();
//            inf.close();
//        }
//
//        // 获取k+1个最近邻（包含自身），排除自身后取k个
//        T_TensorTensor_T topk = dists.topk(k + 1, -1, false, true); // (values, indices)
//        Tensor knnValues = topk.get0(); // [N, k+1] 距离值
//        Tensor knnIdx = topk.get1();    // [N, k+1] 邻居索引
//        // 排除自身（第0列），取后k列
//        knnIdx = knnIdx.narrow(-1, 1, k); // [N, k]
//
//        // ========== 2. 构造edge_index [2, E] ==========
//        // targetIdx: [0,0,...1,1...N-1]（每个节点重复k次）
//        Tensor targetIdx = torch.arange(new Scalar(N), knnIdx.options())
//                .view(-1, 1)
//                .repeat(new long[]{1, k})
//                .view(-1); // [N*k]
//        // sourceIdx: 邻居索引展平 [N*k]
//        Tensor sourceIdx = knnIdx.contiguous().view(-1); // [N*k]
//        // 构造edge_index: [2, N*k]（source→target）
//        Tensor edge_index = torch.stack(new TensorVector(sourceIdx, targetIdx), 0);
//
//        // ========== 3. EdgeConv消息传递（自定义propagate） ==========
//        Tensor out = knn_propagate(edge_index, x);
//
//        // ========== 释放临时张量 ==========
//        dists.close();
//        knnValues.close();
//        knnIdx.close();
//        targetIdx.close();
//        sourceIdx.close();
//        edge_index.close();
//        topk.close(); // 释放topk返回值
//
//        return out;
//    }
//
//    /**
//     * 简化forward调用（无批次向量）
//     */
////    @Override
////    public Tensor forward(Tensor x, Tensor edge_index) {
////        // DynamicEdgeConv的forward默认接收(x, batch)，此处适配基类签名
////        return forward(x, edge_index);
////    }
//
//    /**
//     * 自定义k-NN propagate方法（EdgeConv核心逻辑）
//     * @param edge_index [2, E] k-NN边索引
//     * @param x [N, inChannels] 节点特征
//     * @return 聚合后特征 [N, outChannels]
//     */
//    private Tensor knn_propagate(Tensor edge_index, Tensor x) {
//        long N = x.size(0);
//        Tensor sourceIdx = edge_index.select(0, 0); // [E] 源节点（邻居）索引
//        Tensor targetIdx = edge_index.select(0, 1); // [E] 目标节点索引
//
//        // 提取目标节点特征x_i [E, inChannels]
//        Tensor xi = x.index_select(0, targetIdx);
//        // 提取源节点特征x_j [E, inChannels]
//        Tensor xj = x.index_select(0, sourceIdx);
//
//        // EdgeConv核心：拼接 x_i 和 (x_j - x_i) → [E, 2*inChannels]
//        Tensor msgInput = torch.cat(new TensorVector(xi, xj.sub(xi)), -1);
//
//        // 通过非线性网络变换
//        Tensor msg;
//        if (this.nn != null) {
//            msg = this.nn.asSequential().forward(msgInput); // [E, outChannels]
//        } else {
//            // 无nn时直接返回拼接结果（降维到inChannels）
//            msg = msgInput.narrow(-1, 0, x.size(1));
//        }
//
//        // 调用基类聚合方法（max/mean/add）
//        Tensor out = aggregate(msg, targetIdx, N);
//
//        // ========== 释放临时张量 ==========
//        sourceIdx.close();
//        targetIdx.close();
//        xi.close();
//        xj.close();
//        msgInput.close();
//        msg.close();
//
//        return out;
//    }
//
//    /**
//     * 基类message方法（仅适配签名，逻辑在knn_propagate中实现）
//     */
//    @Override
//    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
//        return x_j;
//    }
//
//    /**
//     * 重置参数（支持训练初始化）
//     */
////    @Override
//    public void reset_parameters() {
//        if (this.nn != null && this.nn instanceof SequentialImpl) {
//           this.nn.asSequential().reset();
//        } else if (this.nn != null && this.nn instanceof LinearImpl) {
//            ((LinearImpl) this.nn).reset_parameters();
//        }
//    }
//
//    /**
//     * 资源释放：避免JNI内存泄漏
//     */
//    @Override
//    public void close() {
//        if (this.nn != null) {
//            this.nn.close();
//            this.nn = null;
//        }
//        super.close();
//    }
//
//    // 辅助方法（测试用）
//    public int getK() { return k; }
//    public long getInChannels() { return inChannels; }
//    public Module getNn() { return nn; }
//}

//package org.bytedeco.pytorch.geometric.nn.conv;
//
//import org.bytedeco.pytorch.*;
//import org.bytedeco.pytorch.nn.Module;
//import org.bytedeco.pytorch.global.torch;
//
///**
// * 实现 torch_geometric.nn.conv.DynamicEdgeConv
// * 特点：动态构建 k-NN 图 + 边缘卷积 (EdgeConv)
// */
//public class DynamicEdgeConv extends MessagePassing {
//    private Module nn; // 传入的非线性变换网络 (包含 LinearImpl)
//    private int k;
//
//    public DynamicEdgeConv(Module nn, int k, String aggr) {
//        super(aggr); // 论文通常建议使用 "max"
//        this.nn = nn;
//        this.k = k;
//
//        if (nn != null) register_module("nn", nn);
//    }
//
//    /**
//     * @param x     节点特征 [N, inChannels]
//     * @param batch 批次向量 [N] (用于防止跨图连接)
//     */
//    @Override
//    public Tensor forward(Tensor x, Tensor batch) {
//        long N = x.size(0);
//
//        // --- 1. 动态构图 (k-NN) ---
//        // 计算特征空间距离并获取 k 个最近邻
//        // 注意：大规模节点下 cdist 极其耗显存
//        Tensor dists = torch.cdist(x, x);
//
//        // topk 获取最小的距离 (largest=false)
//        // 排除自身（第 0 个），取接下来的 k 个
//        T_TensorTensor_T topk = dists.topk(k + 1, -1, false, true);
//        Tensor knn_idx = topk.get1().narrow(-1, 1, k); // [N, k]
//
//        // --- 2. 构造 edge_index ---
//        // targetIdx: [0,0...1,1...N-1]
//        Tensor targetIdx = torch.arange(new Scalar(N), knn_idx.options())
//                .view(-1, 1)
//                .repeat(new long[]{1, k})
//                .view(-1);
//        Tensor sourceIdx = knn_idx.contiguous().view(-1);
//        Tensor edge_index = torch.stack(new TensorVector(sourceIdx, targetIdx), 0);
//
//        // --- 3. 消息传递 (EdgeConv 逻辑) ---
//        return propagate(edge_index, x);
//    }
//
//    /**
//     * 重写基础 propagate 以实现 EdgeConv 的 (x_i, x_j - x_i) 逻辑
//     */
//    public Tensor propagate(Tensor edge_index, Tensor x) {
//        long N = x.size(0);
//        Tensor sourceIdx = edge_index.select(0, 0);
//        Tensor targetIdx = edge_index.select(0, 1);
//
//        Tensor xi = x.index_select(0, targetIdx);
//        Tensor xj = x.index_select(0, sourceIdx);
//
//        // EdgeConv 核心：拼接 x_i 和 (x_j - x_i)
//        // [E, 2 * inChannels]
//        Tensor msgInput = torch.cat(new TensorVector(xi, xj.sub(xi)), -1);
//
//        // 通过传入的神经网络变换
//        Tensor msg = nn.asSequential().forward(msgInput);
//
//        // 聚合 (默认 max)
//        return aggregate(msg, targetIdx, N);
//    }
//
//    @Override
//    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
//        // 逻辑已在手动重写的 propagate 中实现
//        return x_j;
//    }
//}
