package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.options.BatchNormOptions;
import org.bytedeco.pytorch.geometric.nn.Parameter;
import org.bytedeco.pytorch.geometric.utils.Scatter;

import java.util.Arrays;
import java.util.List;

import static org.bytedeco.pytorch.global.torch.zeros_;

/**
 * 最终修复版 CGConv：解决 Linear 维度计算 + 输入维度二次校验
 */
public class CGConv extends MessagePassing {
    private LinearImpl linF; // 用于生成 Softplus 激活的消息
    private LinearImpl linS; // 用于生成 Sigmoid 激活的门控
    private BatchNorm1dImpl bn;
    private Parameter bias; // 改为 Parameter 类型，支持梯度计算
    private long channels;   // 输入/输出通道数
    private int edgeDim;     // 边特征维度
    private long inChannels; // Linear 层输入维度（缓存，用于二次校验）
    private boolean hasBias; // 是否使用偏置
    private boolean isReleased = false; // 资源释放标记

    // 合法的聚合方式
    private static final List<String> VALID_AGGRS = Arrays.asList("add", "mean", "max");

    /**
     * 构造函数：严格校验参数 + 完整初始化
     * @param channels 输入/输出通道数（必须>0）
     * @param edgeDim 边特征维度（≥0）
     * @param aggr 聚合方式（add/mean/max）
     * @param batchNorm 是否启用 BatchNorm
     * @param hasBias 是否使用偏置
     */
    public CGConv(long channels, int edgeDim, String aggr, boolean batchNorm, boolean hasBias) {
        super(validateAggr(aggr));

        // 1. 参数合法性校验
        if (channels <= 0) {
            throw new IllegalArgumentException("通道数channels必须>0: " + channels);
        }
        if (edgeDim < 0) {
            throw new IllegalArgumentException("边特征维度edgeDim必须≥0: " + edgeDim);
        }

        this.channels = channels;
        this.edgeDim = edgeDim;
        this.hasBias = hasBias;

        // 2. 计算 Linear 层输入维度（核心修复：精准计算）
        this.inChannels = 2 * channels + (edgeDim > 0 ? edgeDim : 0);

        // 3. 初始化线性层
        try {
            this.linF = new LinearImpl(inChannels, channels);
            this.linS = new LinearImpl(inChannels, channels);
            register_module("lin_f", linF);
            register_module("lin_s", linS);

            // 4. 初始化 BatchNorm
            if (batchNorm) {
                BatchNormOptions bnOptions = new BatchNormOptions(channels);
                bnOptions.eps().put(1e-5); // 设置默认 eps
                bnOptions.momentum().put(0.1); // 设置默认 momentum
                this.bn = new BatchNorm1dImpl(bnOptions);
                register_module("bn", bn);
            }

            // 5. 初始化偏置
            if (hasBias) {
                this.bias = new Parameter(torch.zeros(new long[]{channels}));
                register_parameter("bias", bias);
            }
        } catch (Exception e) {
            // 创建失败时释放已创建的资源
            close();
            throw new RuntimeException("CGConv 初始化失败：" + e.getMessage(), e);
        }
    }

    /**
     * 重载 forward：支持无/有边特征的调用
     */
    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        return forward(x, edge_index, (Tensor)null);
    }

    /**
     * 核心 forward 逻辑：完整实现 CGConv 标准流程
     */
    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_attr) {
        checkReleased();
        validateInput(x, edge_index, edge_attr);

        long numNodes = x.size(0);
        Tensor out = null;
        Tensor aggr = null;

        try {
            // 1. 消息传递：聚合邻居消息
            aggr = propagate(edge_index, x, edge_attr, numNodes);

            // 2. 添加偏置（核心修复：生效 bias）
            if (hasBias && bias != null) {
                out = aggr.add(bias.data());
            } else {
                out = aggr.clone();
            }

            // 3. 残差连接：聚合消息加回原始特征
            out = x.add(out);

            // 4. 批量归一化（适配 BatchNorm1d 输入维度）
            if (bn != null) {
                // BatchNorm1d 要求输入为 [N, C] 或 [N, C, *]，这里展平为 2D
                out = bn.forward(out.reshape(-1, channels)).reshape(numNodes, channels);
            }

            return out.clone(); // 返回克隆，避免原张量被释放后失效
        } finally {
            // 释放临时张量（除返回值外）
            if (aggr != null) aggr.close();
            
            if (out != null && !out.is_same_size(x)) out.close();
        }
    }

    public Tensor propagate(Tensor edge_index, Tensor x, Tensor edge_attr, long numNodes) {
        // 提取边的源节点（col）和目标节点（row）
        Tensor row = edge_index.select(0, 0); // [E]：目标节点索引
        Tensor col = edge_index.select(0, 1); // [E]：源节点索引

        // 提取中心节点特征 x_i（目标节点）和邻居节点特征 x_j（源节点）
        Tensor x_i = x.index_select(0, row); // [E, C]
        Tensor x_j = x.index_select(0, col); // [E, C]

        // 生成消息：[E, C]
        Tensor msg = message(x_j, x_i, edge_index, edge_attr, numNodes);

        // 聚合消息（根据指定的 aggr 方式）- 核心修正
        Tensor aggr = null;
        String aggrType = getAggr();
        try {
            if ("add".equals(aggrType)) {
                // 正确调用：src=msg, index=row, dim=0, dimSize=numNodes
                aggr = Scatter.scatter_add(msg, row, 0, numNodes);
            } else if ("mean".equals(aggrType)) {
                aggr = Scatter.scatter_mean(msg, row, 0, numNodes);
            } else if ("max".equals(aggrType)) {
                aggr = Scatter.scatter_max(msg, row, 0, numNodes); // 无需 .get(0)
            } else {
                throw new IllegalArgumentException("不支持的聚合方式：" + aggrType);
            }
        } catch (Exception e) {
            throw new RuntimeException("聚合操作失败：" + aggrType, e);
        } finally {
            // 释放所有临时张量（核心：避免内存泄漏）
            row.close();
            col.close();
            x_i.close();
            x_j.close();
            msg.close();
        }

        return aggr;
    }
    /**
     * 自定义 propagate 方法：适配 CGConv 的消息传递逻辑
     */
//    public Tensor propagate3(Tensor edge_index, Tensor x, Tensor edge_attr, long numNodes) {
//        // 提取边的源节点（col）和目标节点（row）
//        Tensor row = edge_index.select(0, 0);
//        Tensor col = edge_index.select(0, 1);
//
//        // 提取中心节点特征 x_i 和邻居节点特征 x_j
//        Tensor x_i = x.index_select(0, row);
//        Tensor x_j = x.index_select(0, col);
//
//        // 生成消息
//        Tensor msg = message(x_j, x_i, edge_index, edge_attr, numNodes);
//
//        // 聚合消息（根据指定的 aggr 方式）
//        Tensor aggr;
//        String aggrType = getAggr();
//        if ("add".equals(aggrType)) {
//            aggr = Scatter.scatter_add(torch.zeros(new long[]{numNodes, channels}, x.options()), 0, row.unsqueeze(1).expand(msg.sizes().vec().get()), msg);
//        } else if ("mean".equals(aggrType)) {
//            aggr = Scatter.scatter_mean(torch.zeros(new long[]{numNodes, channels}, x.options()), 0, row.unsqueeze(1).expand(msg.sizes().vec().get()), msg);
//        } else if ("max".equals(aggrType)) {
//            aggr = Scatter.scatter_max(torch.zeros(new long[]{numNodes, channels}, x.options()), 0, row.unsqueeze(1).expand(msg.sizes().vec().get()), msg).get(0);
//        } else {
//            throw new IllegalArgumentException("不支持的聚合方式：" + aggrType);
//        }
//
//        // 释放临时张量
//        row.close();
//        col.close();
//        x_i.close();
//        x_j.close();
//        msg.close();
//
//        return aggr;
//    }

    /**
     * 消息函数：实现 CGConv 的门控消息计算 + 输入维度二次校验
     */
    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        checkReleased();

        // 1. 拼接特征：x_i + x_j + edge_attr（如果有）
        Tensor z;
        if (edge_attr != null && edgeDim > 0) {
            z = torch.cat(new TensorVector(x_i, x_j, edge_attr), -1);
        } else {
            z = torch.cat(new TensorVector(x_i, x_j), -1);
        }

        // 核心修复：二次校验拼接后的维度，避免 Linear 层维度不匹配
        if (z.size(1) != inChannels) {
            z.close();
            throw new IllegalArgumentException(
                    String.format("拼接后的特征维度不匹配！Linear 层期望输入：%d，实际：%d", inChannels, z.size(1))
            );
        }

        // 2. 计算门控和消息
        Tensor linSOut = linS.forward(z);
        Tensor g = torch.sigmoid(linSOut); // 门控（Sigmoid）

        Tensor linFOut = linF.forward(z);
        Tensor f = torch.softplus(linFOut); // 消息（Softplus）

        // 3. 门控调节后的消息
        Tensor msg = g.mul(f);

        // 释放临时张量
        z.close();
        linSOut.close();
        g.close();
        linFOut.close();
        f.close();

        return msg;
    }

    /**
     * 校验聚合方式合法性
     */
    private static String validateAggr(String aggr) {
        if (aggr == null || !VALID_AGGRS.contains(aggr)) {
            throw new IllegalArgumentException("聚合方式必须是 " + VALID_AGGRS + "，当前：" + aggr);
        }
        return aggr;
    }

    /**
     * 输入合法性校验
     */
    private void validateInput(Tensor x, Tensor edge_index, Tensor edge_attr) {
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
        if (x.size(1) != channels) {
            throw new IllegalArgumentException(
                    String.format("节点特征维度不匹配！期望：%d，实际：%d", channels, x.size(1))
            );
        }
        if (edge_index.dim() != 2 || edge_index.size(0) != 2) {
            throw new IllegalArgumentException(
                    String.format("边索引必须是 [2, num_edges] 形状，当前：%dx%d", edge_index.size(0), edge_index.size(1))
            );
        }

        // 3. 边特征维度校验
        if (edge_attr != null) {
            if (edge_attr.dim() != 2) {
                throw new IllegalArgumentException("边特征必须是2维张量，当前维度：" + edge_attr.dim());
            }
            if (edge_attr.size(0) != edge_index.size(1)) {
                throw new IllegalArgumentException(
                        String.format("边特征数量与边数不匹配！期望：%d，实际：%d", edge_index.size(1), edge_attr.size(0))
                );
            }
            if (edge_attr.size(1) != edgeDim) {
                throw new IllegalArgumentException(
                        String.format("边特征维度不匹配！期望：%d，实际：%d", edgeDim, edge_attr.size(1))
                );
            }
        } else if (edgeDim > 0) {
            throw new IllegalArgumentException("指定了 edgeDim=" + edgeDim + "，但未传入边特征");
        }
    }

    /**
     * 检查资源是否已释放
     */
    private void checkReleased() {
        if (isReleased) {
            throw new IllegalStateException("CGConv 已释放资源，无法继续使用");
        }
    }

    /**
     * 重置参数（符合 PyTorch 规范）
     */
    public void resetParameters() {
        checkReleased();
        if (linF != null) linF.reset_parameters();
        if (linS != null) linS.reset_parameters();
        if (bn != null) bn.reset_parameters();
        if (hasBias && bias != null) {
            torch.zeros_(bias.data());
        }
    }

    /**
     * 释放所有原生资源（核心修复：避免内存泄漏）
     */
    @Override
    public void close() {
        if (!isReleased) {
            // 释放线性层
            if (linF != null) {
                try { linF.close(); } catch (Exception e) {}
            }
            if (linS != null) {
                try { linS.close(); } catch (Exception e) {}
            }

            // 释放 BatchNorm
            if (bn != null) {
                try { bn.close(); } catch (Exception e) {}
            }

            // 释放偏置
            if (bias != null) {
                try { bias.data().close(); } catch (Exception e) {}
                try { bias.close(); } catch (Exception e) {}
            }

            // 释放基类资源
            super.close();

            isReleased = true;
        }
    }

    // Getter 方法
    public long getChannels() { return channels; }
    public int getEdgeDim() { return edgeDim; }
    public long getInChannels() { return inChannels; }
    public boolean hasBias() { return hasBias; }
    public boolean isReleased() { return isReleased; }
}
/**
 * 实现 torch_geometric.nn.conv.CGConv
 * 基于门控机制的晶体图卷积算子。
 */
//public class CGConv extends MessagePassing {
//    private LinearImpl linF; // 用于生成经过 Softplus 激活的消息
//    private LinearImpl linS; // 用于生成经过 Sigmoid 激活的门控
//    private BatchNorm1dImpl bn;
//    private Tensor bias;
//
//    public CGConv(long channels, int edgeDim, String aggr, boolean batchNorm, boolean hasBias) {
//        super(aggr);
//
//        // 输入维度 = 目标节点特征 + 源节点特征 + 边特征
//        // 在非二分图中即 2 * channels + edgeDim
//        long inChannels = 2 * channels + edgeDim;
//
//        this.linF = new LinearImpl(inChannels, channels);
//        this.linS = new LinearImpl(inChannels, channels);
//
//        register_module("lin_f", linF);
//        register_module("lin_s", linS);
//
//        if (batchNorm) {
//            BatchNormOptions options = new BatchNormOptions(channels);
////            options.
//            this.bn = new BatchNorm1dImpl(options);
//            register_module("bn", bn);
//        }
//
//        if (hasBias) {
//            this.bias = torch.zeros(new long[]{channels});
//            register_parameter("bias", bias);
//        }
//    }
//
//    @Override
//    public Tensor forward(Tensor x, Tensor edge_index) {
//        return forward(x, edge_index, (Tensor)null);
//    }
//    
//    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_attr) {
//        long N = x.size(0);
//
//        // 1. 消息传递：在 message 方法中完成拼接、门控计算和消息生成
//        Tensor out = propagate(edge_index, x, edge_attr);
//
//        // 2. 残差连接：CGConv 论文中通常将聚合后的消息加回到原始特征 x
//        out = x.add(out);
//
//        // 3. 可选的 Batch Normalization
//        if (bn != null) {
//            out = bn.forward(out);
//        }
//
//        return out;
//    }
//
//    @Override
//    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
//        // x_j: 邻居特征 [E, C]
//        // x_i: 中心节点特征 [E, C]
//        // edge_attr: 边特征 [E, D]
//
//        // 1. 拼接拼接三者: [E, 2*C + D]
//        Tensor z;
//        if (edge_attr != null) {
//            z = torch.cat(new TensorVector(x_i, x_j, edge_attr), -1);
//        } else {
//            z = torch.cat(new TensorVector(x_i, x_j), -1);
//        }
//
//        // 2. 计算门控 (Sigmoid) 和 消息 (Softplus)
//        // g_ij = Sigmoid(z * W_s + b_s)
//        // f_ij = Softplus(z * W_f + b_f)
//        Tensor g = torch.sigmoid(linS.forward(z));
//        Tensor f = torch.softplus(linF.forward(z));
//
//        // 3. 门控调节后的消息
//        return g.mul(f);
//    }
//
//    public void resetParameters() {
////        checkReleased();
//        if (linF != null) linF.reset_parameters();
//        if (linS != null) linS.reset_parameters();
//        if (bn != null) bn.reset_parameters();
////        if (hasBias && bias != null) {
////            torch.nn.init.zeros_(bias.get());
////        }
//    }
//}