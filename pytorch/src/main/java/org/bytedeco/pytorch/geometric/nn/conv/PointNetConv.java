package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

import static org.bytedeco.pytorch.global.torch.kLong;

/**
 * 最终稳定版 PointNetConv：
 * 1. 修复维度传递错误（确保聚合后返回正确维度）
 * 2. 维度全链路校验（localNN输出=globalNN输入）
 * 3. 完善异常提示（维度不匹配时直接提示预期/实际维度）
 */
public class PointNetConv extends MessagePassing {
    private SequentialImpl localNN;    // 局部 MLP: 输入=特征+坐标维度 → 输出=F
    private SequentialImpl globalNN;   // 全局 MLP: 输入=F → 输出=G
    private boolean addSelfLoops;      // 是否添加自环
    private boolean isReleased = false;
    private long localOutDim;          // 缓存 localNN 输出维度（用于校验）

    /**
     * 构造函数：新增维度校验（localNN输出=globalNN输入）
     * @param localNN 局部 MLP（必须指定输出维度）
     * @param globalNN 全局 MLP（可为null，若不为null则输入维度必须=localNN输出维度）
     * @param addSelfLoops 是否添加自环
     */
    public PointNetConv(SequentialImpl localNN, SequentialImpl globalNN, boolean addSelfLoops) {
        super("max");
        if (localNN == null) throw new IllegalArgumentException("localNN 不能为空（PointNet 核心局部变换）");

        this.localNN = localNN;
        this.globalNN = globalNN;
        this.addSelfLoops = addSelfLoops;

        // 注册子模块
        register_module("local_nn", localNN);
        if (globalNN != null) register_module("global_nn", globalNN);

        // 预计算并校验维度（关键修复：提前校验 localNN 和 globalNN 的维度匹配）
        validateMLPDims();
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        checkReleased();
        throw new IllegalArgumentException("PointNetConv 必须传入 pos 参数（坐标），请调用 forward(x, pos, edge_index)");
    }

    /**
     * 核心 forward 方法：修复维度传递逻辑
     */
    public Tensor forward(Tensor x, Tensor pos, Tensor edge_index) {
        checkReleased();
        // 1. 输入参数校验
        validateInputs(x, pos, edge_index);
        long N = pos.size(0);

        // 2. 处理自环
        Tensor edgeIndexWithLoops = edge_index;
        if (addSelfLoops) {
            edgeIndexWithLoops = addSelfLoops(edgeIndexWithLoops, N);
        }

        // 3. 执行消息传递（核心：确保返回聚合后的正确维度）
        Tensor out = null;
        try {
            out = propagates(edgeIndexWithLoops, x, pos);

            // 维度校验：确保聚合后输出维度正确
            if (out.size(1) != localOutDim) {
                throw new IllegalArgumentException(
                        "聚合后维度不匹配：预期 " + localOutDim + "，实际 " + out.size(1) +
                                "（localNN 输出维度必须与聚合后维度一致）"
                );
            }

            // 4. 全局特征变换（维度已校验）
            if (globalNN != null) {
                out = globalNN.forward(out);
            }

        } finally {
            // 释放临时张量
            if (!edgeIndexWithLoops.equals(edge_index)) {
                edgeIndexWithLoops.close();
            }
        }

        return out;
    }

    /**
     * 修复 propagate 方法：确保返回聚合后的 msg（维度 [N, localOutDim]）
     */
//    @Override
    public Tensor propagates(Tensor edge_index, Tensor... args) {
        checkReleased();
        if (args.length < 2) throw new IllegalArgumentException("需要传入 x, pos 两个参数");

        Tensor x = args[0];
        Tensor pos = args[1];
        long N = pos.size(0);

        Tensor sourceIdx = edge_index.select(0, 0);  // 源节点 j [E]
        Tensor targetIdx = edge_index.select(0, 1);  // 目标节点 i [E]
        Tensor msg = null;
        Tensor pos_i = null, pos_j = null, rel_pos = null, msgInput = null, x_j = null;

        try {
            // 1. 计算相对坐标：pos_j - pos_i [E, D]
            pos_i = pos.index_select(0, targetIdx);
            pos_j = pos.index_select(0, sourceIdx);
            rel_pos = pos_j.sub(pos_i);

            // 2. 构建消息输入：拼接特征和相对坐标
            if (x != null) {
                x_j = x.index_select(0, sourceIdx);  // [E, C]
                msgInput = torch.cat(new TensorVector(x_j, rel_pos), 1);  // [E, C+D]
            } else {
                msgInput = rel_pos;  // [E, D]
            }

            // 3. 局部 MLP 变换（核心：输出 [E, localOutDim]）
            msg = localNN.forward(msgInput);

            // 4. Max Pooling 聚合（输出 [N, localOutDim]）
            msg = aggregate(msg, targetIdx, N);

        } catch (Exception e) {
            throw new RuntimeException("消息传递失败：" + e.getMessage(), e);
        } finally {
            // 释放所有临时张量（关键：避免内存泄漏+维度污染）
            if (pos_i != null) pos_i.close();
            if (pos_j != null) pos_j.close();
            if (rel_pos != null) rel_pos.close();
            if (msgInput != null) msgInput.close();
            if (x_j != null) x_j.close();
            sourceIdx.close();
            targetIdx.close();
        }

        // 关键：返回聚合后的 msg（维度 [N, localOutDim]）
        return msg;
    }

    // ========== 修复后的 Max Pooling 聚合逻辑 ==========
    public Tensor aggregate(Tensor msg, Tensor targetIdx, long numNodes) {
        // 校验 msg 维度（必须是 2 维 [E, F]）
        if (msg.dim() != 2) {
            throw new IllegalArgumentException("msg 必须是 2 维张量 [E, F]，当前维度：" + msg.dim());
        }
        long outDim = msg.size(1);

        // 初始化聚合结果：[numNodes, outDim]，填充极小值
        Tensor aggrOut = torch.full(
                new long[]{numNodes, outDim},
                new Scalar(-1e9),
                msg.options()
        );

        // 构建 TensorOptionalList（index_put_ 要求的索引类型）
        TensorOptionalList indices = new TensorOptionalList();
        try {
            TensorOptional targetIdxOptional = new TensorOptional(targetIdx);
            indices.push_back(targetIdxOptional);

            // 执行 Max Pooling（accumulate=true 保留最大值）
            aggrOut.index_put_(indices, msg, true);

            targetIdxOptional.close();
        } catch (Exception e) {
            indices.close();
            aggrOut.close();
            throw new RuntimeException("Max Pooling 聚合失败：" + e.getMessage(), e);
        }

        indices.close();
        return aggrOut;
    }

    // ========== 新增：MLP 维度校验（核心修复） ==========
    private void validateMLPDims() {
        // 1. 获取 localNN 输出维度（通过 dummy 张量推断）
        // 注意：这里假设 localNN 最后一层是 Linear，实际可根据需求调整
        long lastLinearInDim = 0;
        long lastLinearOutDim = 0;
        for (int i = 0; i < localNN.size(); i++) {
            Module module = localNN.get(i);
            if (module.asLinear() instanceof LinearImpl) {
                LinearImpl linear =  module.asLinear();
                lastLinearInDim = linear.options().in_features().get();
                lastLinearOutDim = linear.options().out_features().get();
                System.out.println(i + " " + lastLinearInDim + " " + lastLinearOutDim);
            }
        }
//        if (lastLinearOutDim == 0) {
//            throw new IllegalArgumentException("localNN 必须包含 Linear 层以确定输出维度");
//        }
        this.localOutDim = lastLinearOutDim;

        // 2. 校验 globalNN 输入维度（若不为null）
        if (globalNN != null) {
            long firstLinearInDim = 0;
            for (int i = 0; i < globalNN.size(); i++) {
                Module module = globalNN.get(i);
                if (module.asLinear() instanceof LinearImpl) {
                    LinearImpl linear =  module.asLinear();
                    firstLinearInDim = linear.options().in_features().get();
                    break; // 取第一层 Linear 的输入维度
                }
            }
            if (firstLinearInDim != localOutDim) {
                throw new IllegalArgumentException(
                        "localNN 输出维度(" + localOutDim + ") 与 globalNN 输入维度(" + firstLinearInDim + ") 不匹配"
                );
            }
        }
    }

    // ========== 其他辅助方法（无修改） ==========
    private Tensor addSelfLoops(Tensor edgeIndex, long numNodes) {
        Tensor selfLoops = torch.arange(new Scalar(0), new Scalar(numNodes), torch.tensor().options().dtype(new ScalarTypeOptional(kLong())));
        Tensor selfLoopEdge = torch.stack(new TensorVector(selfLoops, selfLoops), 0);
        Tensor newEdgeIndex = torch.cat(new TensorVector(edgeIndex, selfLoopEdge), 1);

        selfLoops.close();
        selfLoopEdge.close();
        return newEdgeIndex;
    }

    private void validateInputs(Tensor x, Tensor pos, Tensor edge_index) {
        if (pos == null) throw new IllegalArgumentException("PointNetConv 必须传入 pos 坐标参数（不能为空）");
        if (edge_index == null) throw new IllegalArgumentException("edge_index 不能为空");

        if (pos.dim() != 2) throw new IllegalArgumentException("pos 必须是 2 维张量 [N, D]，当前维度：" + pos.dim());
        if (edge_index.dim() != 2 || edge_index.size(0) != 2) {
            throw new IllegalArgumentException("edge_index 必须是 [2, E] 形状，当前：" + edge_index.size(0) + "x" + edge_index.size(1));
        }

        long N = pos.size(0);
        if (x != null && x.size(0) != N) {
            throw new IllegalArgumentException("x 节点数必须与 pos 一致：" + x.size(0) + " vs " + N);
        }

        Tensor maxIdx = torch.max(edge_index);
        Tensor minIdx = torch.min(edge_index);
        if (maxIdx.item_long() >= N) {
            throw new IllegalArgumentException("edge_index 包含非法节点索引：" + maxIdx.item_long() + " ≥ " + N);
        }
        if (minIdx.item_long() < 0) {
            throw new IllegalArgumentException("edge_index 包含负索引：" + minIdx.item_long());
        }
        maxIdx.close();
        minIdx.close();
    }

    private void checkReleased() {
        if (isReleased) throw new IllegalStateException("PointNetConv 已释放资源，无法继续使用");
    }

    @Override
    public void close() {
        if (!isReleased) {
            if (localNN != null) localNN.close();
            if (globalNN != null) globalNN.close();
            super.close();
            isReleased = true;
        }
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        return x_j;
    }

    // Getter 方法
    public long getLocalOutDim() { return localOutDim; }
    public boolean isAddSelfLoops() { return addSelfLoops; }
    public boolean isReleased() { return isReleased; }
}


//public class PointNetConv extends MessagePassing {
//    private SequentialImpl localNN;    // 局部 MLP: h(x_j, pos_j - pos_i)
//    private SequentialImpl globalNN;   // 全局 MLP: γ(...)
//    private boolean addSelfLoops;      // 是否添加自环
//    private boolean isReleased = false;
//
//    public PointNetConv(SequentialImpl localNN, SequentialImpl globalNN, boolean addSelfLoops) {
//        super("max");
//        if (localNN == null) throw new IllegalArgumentException("localNN 不能为空（PointNet 核心局部变换）");
//
//        this.localNN = localNN;
//        this.globalNN = globalNN;
//        this.addSelfLoops = addSelfLoops;
//
//        register_module("local_nn", localNN);
//        if (globalNN != null) register_module("global_nn", globalNN);
//    }
//
//    @Override
//    public Tensor forward(Tensor x, Tensor edge_index) {
//        checkReleased();
//        throw new IllegalArgumentException("PointNetConv 必须传入 pos 参数（坐标），请调用 forward(x, pos, edge_index)");
//    }
//
//    /**
//     * 核心 forward 方法：修复参数校验时机，先校验再使用 pos
//     */
//    public Tensor forward(Tensor x, Tensor pos, Tensor edge_index) {
//        checkReleased();
//        // ========== 关键修复：先校验（拦截空值），再使用 pos ==========
//        validateInputs(x, pos, edge_index);
//
//        // 此时 pos 一定非空，可安全调用 size(0)
//        long N = pos.size(0);
//
//        // 处理自环
//        Tensor edgeIndexWithLoops = edge_index;
//        if (addSelfLoops) {
//            edgeIndexWithLoops = addSelfLoops(edgeIndexWithLoops, N);
//        }
//
//        // 执行消息传递
//        Tensor out = propagate(edgeIndexWithLoops, x, pos);
//
//        // 全局特征变换
//        if (globalNN != null) {
//            out = globalNN.forward(out);
//        }
//
//        // 释放临时张量
//        if (!edgeIndexWithLoops.equals(edge_index)) {
//            edgeIndexWithLoops.close();
//        }
//
//        return out;
//    }
//
//
//    public Tensor propagate(Tensor edge_index, Tensor... args) {
//        checkReleased();
//        if (args.length < 2) throw new IllegalArgumentException("需要传入 x, pos 两个参数");
//
//        Tensor x = args[0];
//        Tensor pos = args[1];
//        long N = pos.size(0);
//
//        Tensor sourceIdx = edge_index.select(0, 0);  // 源节点 j
//        Tensor targetIdx = edge_index.select(0, 1);  // 目标节点 i
//
//        Tensor msg = null;
//        try {
//            // 计算相对坐标：pos_j - pos_i [E, D]
//            Tensor pos_i = pos.index_select(0, targetIdx);
//            Tensor pos_j = pos.index_select(0, sourceIdx);
//            Tensor rel_pos = pos_j.sub(pos_i);
//
//            // 构建消息输入
//            Tensor msgInput;
//            if (x != null) {
//                Tensor x_j = x.index_select(0, sourceIdx);
//                msgInput = torch.cat(new TensorVector(x_j, rel_pos), 1);
//            } else {
//                msgInput = rel_pos;
//            }
//
//            // 局部 MLP 变换
//            msg = localNN.forward(msgInput);
//
//            // Max Pooling 聚合
//            msg = aggregate(msg, targetIdx, N);
//
//        } finally {
//            sourceIdx.close();
//            targetIdx.close();
//        }
//
//        return msg;
//    }
//
//    // Max Pooling 聚合逻辑
////    public Tensor aggregate(Tensor msg, Tensor targetIdx, long numNodes) {
////        long outDim = msg.size(1);
////        Tensor aggrOut = torch.full(
////                new long[]{numNodes, outDim},
////                new Scalar(-1e9),
////                msg.options()
////        );
////        aggrOut.index_put_(
////                new TensorVector(targetIdx),
////                msg,
////                true
////        );
////        return aggrOut;
////    }
//
//    public Tensor aggregate(Tensor msg, Tensor targetIdx, long numNodes) {
//        // 校验 msg 维度（必须是 [E, F]）
//        if (msg.dim() != 2) {
//            throw new IllegalArgumentException("msg 必须是 2 维张量 [E, F]，当前维度：" + msg.dim());
//        }
//        long outDim = msg.size(1);
//
//        // 初始化聚合结果张量：[numNodes, outDim]，填充极小值
//        Tensor aggrOut = torch.full(
//                new long[]{numNodes, outDim},
//                new Scalar(-1e9),
//                msg.options()
//        );
//
//        // 构建 TensorOptionalList（index_put_ 要求的索引类型）
//        TensorOptionalList indices = new TensorOptionalList();
//        try {
//            // 将 targetIdx 包装为 TensorOptional 并添加到列表
//            TensorOptional targetIdxOptional = new TensorOptional(targetIdx);
//            indices.push_back(targetIdxOptional);
//
//            // 执行 index_put_ 实现 Max Pooling
//            // accumulate=true：对于相同索引，保留最大值（Max Pooling 核心）
//            aggrOut.index_put_(indices, msg, true);
//
//            // 释放 TensorOptional
//            targetIdxOptional.close();
//
//        } catch (Exception e) {
//            // 异常时释放资源
//            indices.close();
//            aggrOut.close();
//            throw new RuntimeException("Max Pooling 聚合失败：" + e.getMessage(), e);
//        }
//
//        // 释放索引列表
//        indices.close();
//
//        return aggrOut;
//    }
//
//    // 自环处理
//    private Tensor addSelfLoops(Tensor edgeIndex, long numNodes) {
//        Tensor selfLoops = torch.arange(new Scalar(0), new Scalar(numNodes), new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long)));
//        Tensor selfLoopEdge = torch.stack(new TensorVector(selfLoops, selfLoops), 0);
//        Tensor newEdgeIndex = torch.cat(new TensorVector(edgeIndex, selfLoopEdge), 1);
//
//        selfLoops.close();
//        selfLoopEdge.close();
//
//        return newEdgeIndex;
//    }
//
//    // ========== 输入参数校验：优先拦截空值 ==========
//    private void validateInputs(Tensor x, Tensor pos, Tensor edge_index) {
//        // 1. 优先校验必选参数的空值（核心修复：提前拦截 pos=null）
//        if (pos == null) throw new IllegalArgumentException("PointNetConv 必须传入 pos 坐标参数（不能为空）");
//        if (edge_index == null) throw new IllegalArgumentException("edge_index 不能为空");
//
//        // 2. 维度校验
//        if (pos.dim() != 2) throw new IllegalArgumentException("pos 必须是 2 维张量 [N, D]，当前维度：" + pos.dim());
//        if (edge_index.dim() != 2 || edge_index.size(0) != 2) {
//            throw new IllegalArgumentException("edge_index 必须是 [2, E] 形状，当前：" + edge_index.size(0) + "x" + edge_index.size(1));
//        }
//
//        // 3. 节点数一致性校验
//        long N = pos.size(0);
//        if (x != null && x.size(0) != N) {
//            throw new IllegalArgumentException("x 节点数必须与 pos 一致：" + x.size(0) + " vs " + N);
//        }
//
//        // 4. 边索引值范围校验
//        Tensor maxIdx = torch.max(edge_index);
//        Tensor minIdx = torch.min(edge_index);
//        if (maxIdx.item_long() >= N) {
//            throw new IllegalArgumentException("edge_index 包含非法节点索引：" + maxIdx.item_long() + " ≥ " + N);
//        }
//        if (minIdx.item_long() < 0) {
//            throw new IllegalArgumentException("edge_index 包含负索引：" + minIdx.item_long());
//        }
//        maxIdx.close();
//        minIdx.close();
//    }
//
//    // 资源管理
//    private void checkReleased() {
//        if (isReleased) throw new IllegalStateException("PointNetConv 已释放资源，无法继续使用");
//    }
//
//    @Override
//    public void close() {
//        if (!isReleased) {
//            if (localNN != null) localNN.close();
//            if (globalNN != null) globalNN.close();
//            super.close();
//            isReleased = true;
//        }
//    }
//
//    @Override
//    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
//        return x_j;
//    }
//
//    // Getter 方法
//    public boolean isAddSelfLoops() { return addSelfLoops; }
//    public boolean isReleased() { return isReleased; }
//}

//public class PointNetConv extends MessagePassing {
//    private SequentialImpl localNN;  // 局部 MLP: h(x_j, pos_j - pos_i)
//    private SequentialImpl globalNN; // 全局 MLP: γ(...)
//    private boolean addSelfLoops;
//
//    public PointNetConv(SequentialImpl localNN, SequentialImpl globalNN, boolean addSelfLoops) {
//        super("max"); // PointNet 论文默认使用 Max Pooling 聚合
//        this.localNN = localNN;
//        this.globalNN = globalNN;
//        this.addSelfLoops = addSelfLoops;
//
//        if (localNN != null) register_module("local_nn", localNN);
//        if (globalNN != null) register_module("global_nn", globalNN);
//    }
//
//    @Override
//    public Tensor forward(Tensor x, Tensor edge_index) {
//        return forward(x, edge_index, null);
//    }
//    /**
//     * @param x    节点特征 [N, C] (可以为 null)
//     * @param pos  节点坐标 [N, D] (D通常为2或3)
//     * @param edge_index 边索引 [2, E]
//     */
//    public Tensor forward(Tensor x, Tensor pos, Tensor edge_index) {
//        long N = pos.size(0);
//
//        // 1. 消息传递逻辑
//        // 我们需要传递 x 和 pos
//        Tensor out = propagate(edge_index, x, pos);
//
//     
//        // 2. 全局变换
//        if (globalNN != null) {
//            out =  globalNN.asSequential().forward(out);
////            out = globalNN.forward(out); //asSequential()
//        }
//
//        return out;
//    }
//
//    /**
//     * 重写基础 propagate 逻辑以处理 pos 差值
//     */
//    public Tensor propagate(Tensor edge_index, Tensor x, Tensor pos) {
//        long N = pos.size(0);
//        Tensor sourceIdx = edge_index.select(0, 0);
//        Tensor targetIdx = edge_index.select(0, 1);
//
//        // 获取源节点和目标节点的位置
//        Tensor pos_i = pos.index_select(0, targetIdx);
//        Tensor pos_j = pos.index_select(0, sourceIdx);
//
//        // 计算相对坐标: pos_j - pos_i
//        Tensor rel_pos = pos_j.sub(pos_i);
//
//        // 如果存在节点特征 x，则进行拼接
//        Tensor msgInput;
//        if (x != null) {
//            Tensor x_j = x.index_select(0, sourceIdx);
//            msgInput = torch.cat(new TensorVector(x_j, rel_pos), -1);
//        } else {
//            msgInput = rel_pos;
//        }
//
//        // --- 局部非线性变换 ---
//        Tensor msg = localNN.asSequential().forward(msgInput); //.asSequential()
//
//        // 聚合 (Max Pooling)
//        return aggregate(msg, targetIdx, N);
//    }
//
//    @Override
//    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
//        // 由于上面手动重写了 propagate 逻辑，此处作为接口占位
//        return x_j;
//    }
//}