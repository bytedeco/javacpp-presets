package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.Parameter;

import java.util.Arrays;
import java.util.List;
import org.bytedeco.pytorch.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.Parameter; // 确保导入自定义的 Parameter 类
import java.util.Arrays;
import java.util.List;

/**
 * 最终适配版 ChebConv：完全匹配你的 JavaCPP/PyTorch API 环境
 * 修复：torch.zeros_() + bias.data()
 */
public class ChebConv extends MessagePassing {
    private long inChannels;
    private long outChannels;
    private int K;
    private String normalization;
    public LinearImpl[] lins;
    public Parameter bias; // 自定义 Parameter 类（重写版）
    private boolean isReleased = false;

    private static final List<String> VALID_NORMS = Arrays.asList("sym", "rw", null);

    public ChebConv(long inChannels, long outChannels, int K, String normalization, boolean hasBias) {
        super("add");

        // 参数校验
        if (inChannels <= 0) {
            throw new IllegalArgumentException("输入通道数 inChannels 必须>0: " + inChannels);
        }
        if (outChannels <= 0) {
            throw new IllegalArgumentException("输出通道数 outChannels 必须>0: " + outChannels);
        }
        if (K < 1) {
            throw new IllegalArgumentException("切比雪夫阶数 K 必须≥1: " + K);
        }
        if (!VALID_NORMS.contains(normalization)) {
            throw new IllegalArgumentException("归一化方式必须是 sym/rw/null，当前：" + normalization);
        }

        // 初始化核心参数
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.K = K;
        this.normalization = normalization;

        // 初始化 K 个线性层 + 安全初始化
        this.lins = new LinearImpl[K];
        for (int i = 0; i < K; i++) {
            this.lins[i] = new LinearImpl(inChannels, outChannels);
            register_module("lin_" + i, this.lins[i]);
            safeLinearInit(this.lins[i]);
        }

        // 初始化偏置
        if (hasBias) {
            this.bias = new Parameter(torch.zeros(new long[]{outChannels}));
            register_parameter("bias", this.bias);
        }
    }

    /**
     * 安全的线性层初始化：修复为 torch.zeros_()
     */
    private void safeLinearInit(LinearImpl linear) {
        // 修复1：使用 torch.xavier_uniform_ (根命名空间)
        torch.xavier_uniform_(linear.weight());
        if (linear.bias() != null) {
            // 修复2：使用 torch.zeros_ (根命名空间，而非 nn.init)
            torch.zeros_(linear.bias());
        }
    }

    /**
     * 核心 forward：修复 bias.data() 调用
     */
    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        return forward(x, edge_index, null, null);
    }

    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_weight, Tensor lambda_max) {
        checkReleased();
        validateInput(x, edge_index, edge_weight);

        long N = x.size(0); // 节点数
        long E = edge_index.size(1); // 边数
        Tensor out = null;
        Tensor inputEdgeWeight = edge_weight; // 保存原始边权重引用

        // ========== 1. 预处理边权重 ==========
        if (edge_weight == null) {
            edge_weight = torch.ones(new long[]{E}, x.options());
        }
        double lMax = (lambda_max != null) ? lambda_max.item_double() : 2.0;
        if (lMax <= 0) {
            throw new IllegalArgumentException("lambda_max 必须>0: " + lMax);
        }
        Scalar scale = new Scalar(2.0 / lMax);
        Tensor normedEdgeWeight = edge_weight.mul(scale);

        // ========== 2. 切比雪夫多项式递归 ==========
        Tensor T0 = x.clone();
        out = lins[0].forward(T0);

        Tensor T1 = null;
        if (K > 1) {
            T1 = manualMessagePassing(T0, edge_index, normedEdgeWeight, N);
            out = out.add(lins[1].forward(T1));

            for (int k = 2; k < K; k++) {
                Tensor Tk = manualMessagePassing(T1, edge_index, normedEdgeWeight, N);
                Tk = Tk.mul(new Scalar(2.0)).sub(T0);
                out = out.add(lins[k].forward(Tk));

                T0.close();
                T0 = T1;
                T1 = Tk;
            }
        }

        // ========== 3. 添加偏置：修复为 bias.data() ==========
        if (bias != null) {
            // 修复3：使用自定义 Parameter 类的 data() 方法
            out = out.add(bias.data());
        }

        // ========== 4. 释放临时张量 ==========
        T0.close();
        if (T1 != null) T1.close();
        normedEdgeWeight.close();
        if (inputEdgeWeight == null && edge_weight != null) {
            edge_weight.close();
        }

        return out.clone();
    }

    /**
     * 手动消息传递（无修改）
     */
    private Tensor manualMessagePassing(Tensor x, Tensor edge_index, Tensor edge_weight, long N) {
        long E = edge_index.size(1);
        long C = x.size(1);

        Tensor srcNodes = edge_index.select(0, 0);
        Tensor dstNodes = edge_index.select(0, 1);
        Tensor xSrc = x.index_select(0, srcNodes);

        Tensor weightExpanded = edge_weight.view(new long[]{E, 1})
                .expand(new long[]{E, C});
        Tensor weightedX = xSrc.mul(weightExpanded);

        Tensor result = torch.zeros(new long[]{N, C}, x.options());
        result.index_add_(0, dstNodes, weightedX);

        // 释放临时张量
        srcNodes.close();
        dstNodes.close();
        xSrc.close();
        weightExpanded.close();
        weightedX.close();

        return result;
    }

    /**
     * 重置参数：修复 bias 初始化
     */
    public void resetParameters() {
        checkReleased();
        for (int i = 0; i < K; i++) {
            if (lins[i] != null) {
                safeLinearInit(lins[i]);
            }
        }
        if (bias != null) {
            // 修复4：使用 torch.zeros_ 初始化 bias.data()
            torch.zeros_(bias.data());
        }
    }

    // ========== 其他工具方法（无修改） ==========
    private void validateInput(Tensor x, Tensor edge_index, Tensor edge_weight) {
        if (x == null) throw new IllegalArgumentException("节点特征 x 不能为空");
        if (edge_index == null) throw new IllegalArgumentException("边索引 edge_index 不能为空");

        if (x.dim() != 2) throw new IllegalArgumentException("节点特征必须是 2 维 [N, C]，当前维度：" + x.dim());
        if (x.size(1) != inChannels) {
            throw new IllegalArgumentException(
                    String.format("输入特征维度不匹配！期望 %d，实际 %d", inChannels, x.size(1))
            );
        }
        if (edge_index.dim() != 2 || edge_index.size(0) != 2) {
            throw new IllegalArgumentException("边索引必须是 [2, E] 形状，当前：" + edge_index.size(0) + "x" + edge_index.size(1));
        }

        if (edge_weight != null) {
            if (edge_weight.dim() != 1 || edge_weight.size(0) != edge_index.size(1)) {
                throw new IllegalArgumentException(
                        String.format("边权重必须是 1 维 [E]，当前维度：%d，长度：%d",
                                edge_weight.dim(), edge_weight.size(0))
                );
            }
        }
    }

    private void checkReleased() {
        if (isReleased) {
            throw new IllegalStateException("ChebConv 已释放资源，无法继续使用");
        }
    }

    @Override
    public void close() {
        if (!isReleased) {
            if (lins != null) {
                for (LinearImpl lin : lins) {
                    if (lin != null) {
                        try { lin.close(); } catch (Exception e) {}
                    }
                }
            }

            if (bias != null) {
                try { bias.data().close(); } catch (Exception e) {}
                try { bias.close(); } catch (Exception e) {}
            }

            super.close();
            isReleased = true;
        }
    }

    // Getter 方法
    public long getInChannels() { return inChannels; }
    public long getOutChannels() { return outChannels; }
    public int getK() { return K; }
    public String getNormalization() { return normalization; }
    public boolean isReleased() { return isReleased; }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        return x_j;
    }
}


/**
 * 终极修复版 ChebConv：不依赖父类 MessagePassing 的 propagate，直接实现谱卷积逻辑
 * 彻底解决维度不匹配问题
 */
//public class ChebConv extends MessagePassing {
//    private long inChannels;
//    private long outChannels;
//    private int K;
//    private String normalization;
//    private LinearImpl[] lins;
//    public Parameter bias;
//    private boolean isReleased = false;
//
//    private static final List<String> VALID_NORMS = Arrays.asList("sym", "rw", null);
//
//    public ChebConv(long inChannels, long outChannels, int K, String normalization, boolean hasBias) {
//        super("add");
//
//        // 参数校验
//        if (inChannels <= 0) {
//            throw new IllegalArgumentException("输入通道数 inChannels 必须>0: " + inChannels);
//        }
//        if (outChannels <= 0) {
//            throw new IllegalArgumentException("输出通道数 outChannels 必须>0: " + outChannels);
//        }
//        if (K < 1) {
//            throw new IllegalArgumentException("切比雪夫阶数 K 必须≥1: " + K);
//        }
//        if (!VALID_NORMS.contains(normalization)) {
//            throw new IllegalArgumentException("归一化方式必须是 sym/rw/null，当前：" + normalization);
//        }
//
//        // 初始化核心参数
//        this.inChannels = inChannels;
//        this.outChannels = outChannels;
//        this.K = K;
//        this.normalization = normalization;
//
//        // 初始化 K 个线性层 + 安全初始化
//        this.lins = new LinearImpl[K];
//        for (int i = 0; i < K; i++) {
//            this.lins[i] = new LinearImpl(inChannels, outChannels);
//            register_module("lin_" + i, this.lins[i]);
//            safeLinearInit(this.lins[i]);
//        }
//
//        // 初始化偏置
//        if (hasBias) {
//            this.bias = new Parameter(torch.zeros(new long[]{outChannels}));
//            register_parameter("bias", this.bias);
//        }
//    }
//
//    /**
//     * 安全的线性层初始化
//     */
//    private void safeLinearInit(LinearImpl linear) {
//        torch.xavier_uniform_(linear.weight());
//        if (linear.bias() != null) {
//            torch.zeros_(linear.bias());
//        }
//    }
//
//    /**
//     * 核心修复：完全重写 forward，不依赖父类 propagate
//     */
//    @Override
//    public Tensor forward(Tensor x, Tensor edge_index) {
//        return forward(x, edge_index, null, null);
//    }
//
//    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_weight, Tensor lambda_max) {
//        checkReleased();
//        validateInput(x, edge_index, edge_weight);
//
//        long N = x.size(0); // 节点数
//        long E = edge_index.size(1); // 边数
//        Tensor out = null;
//
//        // ========== 1. 预处理边权重（核心：手动处理，不依赖父类） ==========
//        // 初始化边权重（默认全1）
//        if (edge_weight == null) {
//            edge_weight = torch.ones(new long[]{E}, x.options());
//        }
//        // 计算归一化因子
//        double lMax = (lambda_max != null) ? lambda_max.item_double() : 2.0;
//        if (lMax <= 0) {
//            throw new IllegalArgumentException("lambda_max 必须>0: " + lMax);
//        }
//        Scalar scale = new Scalar(2.0 / lMax);
//        Tensor normedEdgeWeight = edge_weight.mul(scale); // 归一化后的边权重 [E]
//
//        // ========== 2. 手动实现切比雪夫多项式递归（核心） ==========
//        // T0 = x [N, inChannels]
//        Tensor T0 = x.clone();
//        out = lins[0].forward(T0); // 第0阶项
//
//        if (K > 1) {
//            // T1 = L_hat @ x = 邻接聚合（手动实现消息传递）
//            Tensor T1 = manualMessagePassing(T0, edge_index, normedEdgeWeight, N);
//            out = out.add(lins[1].forward(T1));
//
//            // 高阶项递归：Tk = 2*L_hat@Tk-1 - Tk-2
//            for (int k = 2; k < K; k++) {
//                Tensor Tk = manualMessagePassing(T1, edge_index, normedEdgeWeight, N);
//                Tk = Tk.mul(new Scalar(2.0)).sub(T0); // 递归公式
//                out = out.add(lins[k].forward(Tk));
//
//                // 释放临时张量 + 滚动更新
//                T0.close();
//                T0 = T1;
//                T1 = Tk;
//            }
//        }
//
//        // ========== 3. 添加偏置 ==========
//        if (bias != null) {
//            out = out.add(bias.data());
//        }
//
//        // ========== 4. 释放临时张量 ==========
//        T0.close();
////        if (K > 1) T1.close();
//        normedEdgeWeight.close();
//        if (edge_weight != null && edge_weight != normedEdgeWeight) edge_weight.close();
//
//        return out.clone();
//    }
//
//    /**
//     * 手动实现消息传递（核心：绕过父类 MessagePassing，直接实现邻接聚合）
//     * 功能：计算 L_hat @ x = 聚合邻居特征并乘以归一化边权重
//     * @param x 节点特征 [N, C]
//     * @param edge_index 边索引 [2, E]
//     * @param edge_weight 归一化边权重 [E]
//     * @param N 节点数
//     * @return 聚合结果 [N, C]
//     */
//    private Tensor manualMessagePassing(Tensor x, Tensor edge_index, Tensor edge_weight, long N) {
//        long E = edge_index.size(1);
//        long C = x.size(1);
//
//        // 1. 提取源节点/目标节点索引 [E]
//        Tensor srcNodes = edge_index.select(0, 0); // 源节点（边的起点）[E]
//        Tensor dstNodes = edge_index.select(1, 0); // 目标节点（边的终点）[E]
//
//        // 2. 取出所有源节点的特征 [E, C]
//        Tensor xSrc = x.index_select(0, srcNodes); // 按源节点索引取特征
//
//        // 3. 边权重广播到特征维度 [E] -> [E, 1] -> [E, C]
//        Tensor weightExpanded = edge_weight.view(new long[]{E, 1})
//                .expand(new long[]{E, C}); // 手动广播，确保形状匹配
//
//        // 4. 邻居特征乘以边权重 [E, C] * [E, C]
//        Tensor weightedX = xSrc.mul(weightExpanded);
//
//        // 5. 聚合到目标节点（add 聚合）
//        Tensor result = torch.zeros(new long[]{N, C}, x.options());
//        result.index_add_(0, dstNodes, weightedX); // 按目标节点聚合
//
//        // 6. 释放临时张量
//        srcNodes.close();
//        dstNodes.close();
//        xSrc.close();
//        weightExpanded.close();
//        weightedX.close();
//
//        return result;
//    }
//
//    /**
//     * 输入合法性校验
//     */
//    private void validateInput(Tensor x, Tensor edge_index, Tensor edge_weight) {
//        if (x == null) throw new IllegalArgumentException("节点特征 x 不能为空");
//        if (edge_index == null) throw new IllegalArgumentException("边索引 edge_index 不能为空");
//
//        if (x.dim() != 2) throw new IllegalArgumentException("节点特征必须是 2 维 [N, C]，当前维度：" + x.dim());
//        if (x.size(1) != inChannels) {
//            throw new IllegalArgumentException(
//                    String.format("输入特征维度不匹配！期望 %d，实际 %d", inChannels, x.size(1))
//            );
//        }
//        if (edge_index.dim() != 2 || edge_index.size(0) != 2) {
//            throw new IllegalArgumentException("边索引必须是 [2, E] 形状，当前：" + edge_index.size(0) + "x" + edge_index.size(1));
//        }
//
//        if (edge_weight != null) {
//            if (edge_weight.dim() != 1 || edge_weight.size(0) != edge_index.size(1)) {
//                throw new IllegalArgumentException(
//                        String.format("边权重必须是 1 维 [E]，当前维度：%d，长度：%d",
//                                edge_weight.dim(), edge_weight.size(0))
//                );
//            }
//        }
//    }
//
//    /**
//     * 检查资源是否已释放
//     */
//    private void checkReleased() {
//        if (isReleased) {
//            throw new IllegalStateException("ChebConv 已释放资源，无法继续使用");
//        }
//    }
//
//    /**
//     * 重置参数
//     */
//    public void resetParameters() {
//        checkReleased();
//        for (int i = 0; i < K; i++) {
//            if (lins[i] != null) {
//                safeLinearInit(lins[i]);
//            }
//        }
//        if (bias != null) {
//            torch.zeros_(bias.data());
//        }
//    }
//
//    /**
//     * 释放资源
//     */
//    @Override
//    public void close() {
//        if (!isReleased) {
//            if (lins != null) {
//                for (LinearImpl lin : lins) {
//                    if (lin != null) {
//                        try { lin.close(); } catch (Exception e) {}
//                    }
//                }
//            }
//
//            if (bias != null) {
//                try { bias.data().close(); } catch (Exception e) {}
//                try { bias.close(); } catch (Exception e) {}
//            }
//
//            super.close();
//            isReleased = true;
//        }
//    }
//
//    // Getter 方法
//    public long getInChannels() { return inChannels; }
//    public long getOutChannels() { return outChannels; }
//    public int getK() { return K; }
//    public String getNormalization() { return normalization; }
//    public boolean isReleased() { return isReleased; }
//
//    // 重写空的 message 方法（避免父类调用）
//    @Override
//    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
//        return x_j; // 不会被调用，仅占位
//    }
//}
/**
 * 最终修复版 ChebConv：解决维度不匹配 + 初始化越界问题
 */
//public class ChebConv extends MessagePassing {
//    private long inChannels;
//    private long outChannels;
//    private int K;
//    private String normalization;
//    private LinearImpl[] lins;
//    public Parameter bias;
//    private boolean isReleased = false;
//
//    private static final List<String> VALID_NORMS = Arrays.asList("sym", "rw", null);
//
//    public ChebConv(long inChannels, long outChannels, int K, String normalization, boolean hasBias) {
//        super("add");
//
//        // 参数校验（不变）
//        if (inChannels <= 0) {
//            throw new IllegalArgumentException("输入通道数 inChannels 必须>0: " + inChannels);
//        }
//        if (outChannels <= 0) {
//            throw new IllegalArgumentException("输出通道数 outChannels 必须>0: " + outChannels);
//        }
//        if (K < 1) {
//            throw new IllegalArgumentException("切比雪夫阶数 K 必须≥1: " + K);
//        }
//        if (!VALID_NORMS.contains(normalization)) {
//            throw new IllegalArgumentException("归一化方式必须是 sym/rw/null，当前：" + normalization);
//        }
//
//        // 初始化核心参数（不变）
//        this.inChannels = inChannels;
//        this.outChannels = outChannels;
//        this.K = K;
//        this.normalization = normalization;
//
//        // 初始化 K 个线性层 + 安全初始化（不变）
//        this.lins = new LinearImpl[K];
//        for (int i = 0; i < K; i++) {
//            this.lins[i] = new LinearImpl(inChannels, outChannels);
//            register_module("lin_" + i, this.lins[i]);
//            safeLinearInit(this.lins[i]); // 安全初始化
//        }
//
//        // 初始化偏置（不变）
//        if (hasBias) {
//            this.bias = new Parameter(torch.zeros(new long[]{outChannels}));
//            register_parameter("bias", this.bias);
//        }
//    }
//
//    /**
//     * 安全的线性层初始化（不变）
//     */
//    private void safeLinearInit(LinearImpl linear) {
//        torch.xavier_uniform_(linear.weight());
//        if (linear.bias() != null) {
//            torch.zeros_(linear.bias());
//        }
//    }
//
//    // ========== 核心修复：重写 message 函数，解决维度不匹配 ==========
//    @Override
//    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
//        checkReleased();
//        if (edge_attr != null) {
//            // 方案1：手动指定广播形状（推荐，避免 expand_as 兼容性问题）
//            // edge_attr 形状：[E] -> [E, 1] -> [E, inChannels]
//            long E = edge_attr.size(0);
//            Tensor edgeAttrExpanded = edge_attr.view(new long[]{E, 1})
//                    .expand(new long[]{E, inChannels}); // 手动指定广播到 [E, inChannels]
//
//            // 乘法：x_j [E, inChannels] * edgeAttrExpanded [E, inChannels]
//            Tensor result = x_j.mul(edgeAttrExpanded);
//
//            // 释放临时张量
//            edgeAttrExpanded.close();
//            return result;
//        }
//        return x_j; // 无边权重时直接返回
//    }
//
//    // ========== 其他方法：仅修复 propagate 调用的参数传递 ==========
//    @Override
//    public Tensor forward(Tensor x, Tensor edge_index) {
//        return forward(x, edge_index, null, null);
//    }
//
//    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_weight, Tensor lambda_max) {
//        checkReleased();
//        validateInput(x, edge_index, edge_weight);
//
//        long N = x.size(0);
//        Tensor out = null;
//        Tensor x0 = null, x1 = null, x2 = null;
//        Tensor norm = null;
//        Tensor edgeIndexScaled = edge_index.clone();
//
//        try {
//            norm = computeChebNorm(edge_index, edge_weight, lambda_max, N);
//            x0 = x.clone();
//            out = lins[0].forward(x0);
//
//            if (K > 1) {
//                // ========== 修复：propagate 传递正确的参数 ==========
//                x1 = super.propagate(edgeIndexScaled, x0, norm, N); // 直接调用父类 propagate
//                out = out.add(lins[1].forward(x1));
//
//                for (int k = 2; k < K; k++) {
//                    x2 = super.propagate(edgeIndexScaled, x1, norm, N);
//                    x2 = x2.mul(new Scalar(2.0)).sub(x0);
//                    out = out.add(lins[k].forward(x2));
//
//                    if (x0 != null && k > 2) x0.close();
//                    x0 = x1;
//                    x1 = x2;
//                }
//            }
//
//            if (bias != null) {
//                out = out.add(bias.data());
//            }
//
//            return out.clone();
//        } catch (Exception e) {
//            throw new RuntimeException("ChebConv forward 失败：" + e.getMessage(), e);
//        } finally {
//            // 释放所有临时张量
//            if (x0 != null) x0.close();
//            if (x1 != null && x1 != x0) x1.close();
//            if (x2 != null && x2 != x1) x2.close();
//            if (norm != null) norm.close();
//            if (edgeIndexScaled != null) edgeIndexScaled.close();
//            if (out != null && !out.is_same_size(torch.empty(new long[]{N, outChannels}))) {
//                out.close();
//            }
//        }
//    }
//
//    /**
//     * 计算切比雪夫归一化因子（不变）
//     */
//    private Tensor computeChebNorm(Tensor edge_index, Tensor edge_weight, Tensor lambda_max, long N) {
//        if (edge_weight == null) {
//            edge_weight = torch.ones(new long[]{edge_index.size(1)}, edge_index.options());
//        }
//
//        double lMax = (lambda_max != null) ? lambda_max.item_double() : 2.0;
//        if (lMax <= 0) {
//            throw new IllegalArgumentException("lambda_max 必须>0: " + lMax);
//        }
//
//        Scalar scale = new Scalar(2.0 / lMax);
//        Tensor norm = edge_weight.mul(scale);
//        return norm;
//    }
//
//    /**
//     * 移除自定义 propagate 方法，避免参数传递错误
//     */
//    // 删掉这行：private Tensor propagate(Tensor edgeIndex, Tensor x, Tensor norm, long numNodes)
//
//    /**
//     * 输入合法性校验（不变）
//     */
//    private void validateInput(Tensor x, Tensor edge_index, Tensor edge_weight) {
//        if (x == null) throw new IllegalArgumentException("节点特征 x 不能为空");
//        if (edge_index == null) throw new IllegalArgumentException("边索引 edge_index 不能为空");
//
//        if (x.dim() != 2) throw new IllegalArgumentException("节点特征必须是 2 维 [N, C]，当前维度：" + x.dim());
//        if (x.size(1) != inChannels) {
//            throw new IllegalArgumentException(
//                    String.format("输入特征维度不匹配！期望 %d，实际 %d", inChannels, x.size(1))
//            );
//        }
//        if (edge_index.dim() != 2 || edge_index.size(0) != 2) {
//            throw new IllegalArgumentException("边索引必须是 [2, E] 形状，当前：" + edge_index.size(0) + "x" + edge_index.size(1));
//        }
//
//        if (edge_weight != null) {
//            if (edge_weight.dim() != 1 || edge_weight.size(0) != edge_index.size(1)) {
//                throw new IllegalArgumentException(
//                        String.format("边权重必须是 1 维 [E]，当前维度：%d，长度：%d",
//                                edge_weight.dim(), edge_weight.size(0))
//                );
//            }
//        }
//    }
//
//    /**
//     * 检查资源是否已释放（不变）
//     */
//    private void checkReleased() {
//        if (isReleased) {
//            throw new IllegalStateException("ChebConv 已释放资源，无法继续使用");
//        }
//    }
//
//    /**
//     * 重置参数（不变）
//     */
//    public void resetParameters() {
//        checkReleased();
//        for (int i = 0; i < K; i++) {
//            if (lins[i] != null) {
//                safeLinearInit(lins[i]);
//            }
//        }
//        if (bias != null) {
//            torch.zeros_(bias.data());
//        }
//    }
//
//    /**
//     * 释放资源（不变）
//     */
//    @Override
//    public void close() {
//        if (!isReleased) {
//            if (lins != null) {
//                for (LinearImpl lin : lins) {
//                    if (lin != null) {
//                        try { lin.close(); } catch (Exception e) {}
//                    }
//                }
//            }
//
//            if (bias != null) {
//                try { bias.data().close(); } catch (Exception e) {}
//                try { bias.close(); } catch (Exception e) {}
//            }
//
//            super.close();
//            isReleased = true;
//        }
//    }
//
//    // Getter 方法（不变）
//    public long getInChannels() { return inChannels; }
//    public long getOutChannels() { return outChannels; }
//    public int getK() { return K; }
//    public String getNormalization() { return normalization; }
//    public boolean isReleased() { return isReleased; }
//}


//public class ChebConv extends MessagePassing {
//    private long inChannels;
//    private long outChannels;
//    private int K;
//    private String normalization;
//    private LinearImpl[] lins;
//    public Parameter bias;
//    private boolean isReleased = false;
//
//    private static final List<String> VALID_NORMS = Arrays.asList("sym", "rw", null);
//
//    public ChebConv(long inChannels, long outChannels, int K, String normalization, boolean hasBias) {
//        super("add");
//
//        // 1. 参数校验（不变）
//        if (inChannels <= 0) {
//            throw new IllegalArgumentException("输入通道数 inChannels 必须>0: " + inChannels);
//        }
//        if (outChannels <= 0) {
//            throw new IllegalArgumentException("输出通道数 outChannels 必须>0: " + outChannels);
//        }
//        if (K < 1) {
//            throw new IllegalArgumentException("切比雪夫阶数 K 必须≥1: " + K);
//        }
//        if (!VALID_NORMS.contains(normalization)) {
//            throw new IllegalArgumentException("归一化方式必须是 sym/rw/null，当前：" + normalization);
//        }
//
//        // 2. 初始化核心参数（不变）
//        this.inChannels = inChannels;
//        this.outChannels = outChannels;
//        this.K = K;
//        this.normalization = normalization;
//
//        // 3. 初始化 K 个线性层 + 安全初始化（核心修复）
//        this.lins = new LinearImpl[K];
//        for (int i = 0; i < K; i++) {
//            this.lins[i] = new LinearImpl(inChannels, outChannels);
//            register_module("lin_" + i, this.lins[i]);
//
//            // ========== 核心修复：替换默认的 Kaiming 初始化 ==========
//            // 使用 Xavier 均匀初始化（更安全，范围不会越界）
//            safeLinearInit(this.lins[i]);
//        }
//
//        // 4. 初始化偏置（不变）
//        if (hasBias) {
//            this.bias = new Parameter(torch.zeros(new long[]{outChannels}));
//            register_parameter("bias", this.bias);
//        }
//    }
//
//    /**
//     * 安全的线性层初始化：替换默认的 Kaiming 初始化，避免 float 越界
//     */
//    private void safeLinearInit(LinearImpl linear) {
//        // 方案1：Xavier 均匀初始化（推荐，范围可控）
//        torch.xavier_uniform_(linear.weight());
//
//        // 方案2（备选）：固定范围的均匀初始化（范围 [-0.1, 0.1]，绝对安全）
//        // torch.uniform_(linear.weight(), new Scalar(-0.1), new Scalar(0.1));
//
//        // 偏置初始化为 0（保持默认）
//        if (linear.bias() != null) {
//            torch.zeros_(linear.bias());
//        }
//    }
//
//    // ========== 以下方法完全不变 ==========
//    @Override
//    public Tensor forward(Tensor x, Tensor edge_index) {
//        return forward(x, edge_index, null, null);
//    }
//
//    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_weight, Tensor lambda_max) {
//        checkReleased();
//        validateInput(x, edge_index, edge_weight);
//
//        long N = x.size(0);
//        Tensor out = null;
//        Tensor x0 = null, x1 = null, x2 = null;
//        Tensor norm = null;
//        Tensor edgeIndexScaled = edge_index.clone();
//
//        try {
//            norm = computeChebNorm(edge_index, edge_weight, lambda_max, N);
//            x0 = x.clone();
//            out = lins[0].forward(x0);
//
//            if (K > 1) {
//                x1 = propagate(edgeIndexScaled, x0, norm, N);
//                out = out.add(lins[1].forward(x1));
//
//                for (int k = 2; k < K; k++) {
//                    x2 = propagate(edgeIndexScaled, x1, norm, N);
//                    x2 = x2.mul(new Scalar(2.0)).sub(x0);
//                    out = out.add(lins[k].forward(x2));
//
//                    if (x0 != null && k > 2) x0.close();
//                    x0 = x1;
//                    x1 = x2;
//                }
//            }
//
//            if (bias != null) {
//                out = out.add(bias.data());
//            }
//
//            return out.clone();
//        } catch (Exception e) {
//            throw new RuntimeException("ChebConv forward 失败：" + e.getMessage(), e);
//        } finally {
//            if (x0 != null) x0.close();
//            if (x1 != null && x1 != x0) x1.close();
//            if (x2 != null && x2 != x1) x2.close();
//            if (norm != null) norm.close();
//            if (edgeIndexScaled != null) edgeIndexScaled.close();
//            if (out != null && !out.is_same_size(torch.empty(new long[]{N, outChannels}))) {
//                out.close();
//            }
//        }
//    }
//
//    private Tensor computeChebNorm(Tensor edge_index, Tensor edge_weight, Tensor lambda_max, long N) {
//        if (edge_weight == null) {
//            edge_weight = torch.ones(new long[]{edge_index.size(1)}, edge_index.options());
//        }
//
//        double lMax = (lambda_max != null) ? lambda_max.item_double() : 2.0;
//        if (lMax <= 0) {
//            throw new IllegalArgumentException("lambda_max 必须>0: " + lMax);
//        }
//
//        Scalar scale = new Scalar(2.0 / lMax);
//        Tensor norm = edge_weight.mul(scale);
//        return norm;
//    }
//
//    @Override
//    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
//        checkReleased();
//        if (edge_attr != null) {
//            return x_j.mul(edge_attr.view(-1, 1).expand_as(x_j));
//        }
//        return x_j;
//    }
//
//    public Tensor propagate(Tensor edgeIndex, Tensor x, Tensor norm, long numNodes) {
//        return super.propagate(edgeIndex, x, norm, numNodes);
//    }
//
//    private void validateInput(Tensor x, Tensor edge_index, Tensor edge_weight) {
//        if (x == null) throw new IllegalArgumentException("节点特征 x 不能为空");
//        if (edge_index == null) throw new IllegalArgumentException("边索引 edge_index 不能为空");
//
//        if (x.dim() != 2) throw new IllegalArgumentException("节点特征必须是 2 维 [N, C]，当前维度：" + x.dim());
//        if (x.size(1) != inChannels) {
//            throw new IllegalArgumentException(
//                    String.format("输入特征维度不匹配！期望 %d，实际 %d", inChannels, x.size(1))
//            );
//        }
//        if (edge_index.dim() != 2 || edge_index.size(0) != 2) {
//            throw new IllegalArgumentException("边索引必须是 [2, E] 形状，当前：" + edge_index.size(0) + "x" + edge_index.size(1));
//        }
//
//        if (edge_weight != null) {
//            if (edge_weight.dim() != 1 || edge_weight.size(0) != edge_index.size(1)) {
//                throw new IllegalArgumentException(
//                        String.format("边权重必须是 1 维 [E]，当前维度：%d，长度：%d",
//                                edge_weight.dim(), edge_weight.size(0))
//                );
//            }
//        }
//    }
//
//    private void checkReleased() {
//        if (isReleased) {
//            throw new IllegalStateException("ChebConv 已释放资源，无法继续使用");
//        }
//    }
//
//    public void resetParameters() {
//        checkReleased();
//        for (int i = 0; i < K; i++) {
//            if (lins[i] != null) {
//                // 重置时也使用安全初始化
//                safeLinearInit(lins[i]);
//            }
//        }
//        if (bias != null) {
//            torch.zeros_(bias.data());
//        }
//    }
//
//    @Override
//    public void close() {
//        if (!isReleased) {
//            if (lins != null) {
//                for (LinearImpl lin : lins) {
//                    if (lin != null) {
//                        try { lin.close(); } catch (Exception e) {}
//                    }
//                }
//            }
//
//            if (bias != null) {
//                try { bias.data().close(); } catch (Exception e) {}
//                try { bias.close(); } catch (Exception e) {}
//            }
//
//            super.close();
//            isReleased = true;
//        }
//    }
//
//    // Getter 方法（不变）
//    public long getInChannels() { return inChannels; }
//    public long getOutChannels() { return outChannels; }
//    public int getK() { return K; }
//    public String getNormalization() { return normalization; }
//    public boolean isReleased() { return isReleased; }
//}
//public class ChebConv extends MessagePassing {
//    private long inChannels;
//    private long outChannels;
//    private int K; // 过滤器大小/阶数
//    private String normalization; // "sym", "rw", null
//    private boolean isReleased = false;
//    // 我们需要 K 个线性层，或者一个 [K, in, out] 的参数张量
//    private LinearImpl[] lins;
//    public Tensor bias;
//
//    public ChebConv(long inChannels, long outChannels, int K, String normalization, boolean hasBias) {
//        super("add");
//        this.inChannels = inChannels;
//        this.outChannels = outChannels;
//        this.K = K;
//        this.normalization = normalization;
//
//        // 初始化 K 个权重分支
//        this.lins = new LinearImpl[K];
//        for (int i = 0; i < K; i++) {
//            lins[i] = new LinearImpl(inChannels, outChannels);
//            // 显式注册每一个线性层
//            register_module("lin_" + i, lins[i]);
//        }
//
//        if (hasBias) {
//            this.bias = torch.zeros(new long[]{outChannels});
//            register_parameter("bias", bias);
//        }
//    }
//    @Override
//    public Tensor forward(Tensor x, Tensor edge_index) {
//        return forward(x, edge_index, null);
//    }
//    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_weight, Tensor lambda_max) {
//        long N = x.size(0);
//
//        // 1. 计算归一化 Laplacian (L_hat)
//        // L_hat = 2L / lambda_max - I
//        // 这里简化实现：假设 edge_weight 已经是归一化后的 L_hat 权重
//        // 在标准 PyG 中，这里会调用 get_laplacian 并缩放
//        Tensor edge_index_scaled = edge_index;
//        Tensor norm = compute_cheb_norm(edge_index, edge_weight, lambda_max, N);
//
//        // 2. 切比雪夫递归计算
//        // T_0(X) = X
//        Tensor x0 = x;
//        Tensor out = lins[0].forward(x0);
//
//        if (K > 1) {
//            // T_1(X) = L_hat @ X
//            Tensor x1 = propagate(edge_index_scaled, x0, norm);
//            out = out.add(lins[1].forward(x1));
//
//            // T_k(X) = 2 * L_hat @ T_{k-1}(X) - T_{k-2}(X)
//            for (int k = 2; k < K; k++) {
//                Tensor x2 = propagate(edge_index_scaled, x1, norm);
//                x2 = x2.mul(new Scalar(2.0)).sub(x0);
//                out = out.add(lins[k].forward(x2));
//
//                // 滚动更新
//                x0 = x1;
//                x1 = x2;
//            }
//        }
//
//        if (bias != null) {
//            out = out.add(bias);
//        }
//        return out;
//    }
//
//    /**
//     * 计算切比雪夫归一化因子
//     */
//    private Tensor compute_cheb_norm(Tensor edge_index, Tensor edge_weight, Tensor lambda_max, long N) {
//        // 简化版：如果是生产环境，需要实现 get_laplacian 并根据 lambda_max 缩放
//        // 这里的逻辑应返回满足 T_k 递归所需的边权重
//        if (edge_weight == null) {
//            edge_weight = torch.ones(new long[]{edge_index.size(1)}, edge_index.options());
//        }
//
//        double l_max = (lambda_max != null) ? lambda_max.item_double() : 2.0;
//        // 映射 L 到 [-1, 1] 区间: 2L / l_max - I
//        return edge_weight.mul(new Scalar(2.0 / l_max));
//    }
//
//    @Override
//    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
//        // 这里的 edge_attr 承载的是 Laplacian 的权重因子
//        if (edge_attr != null) {
//            return x_j.mul(edge_attr.view(-1, 1));
//        }
//        return x_j;
//    }
//
//    private void validateInput(Tensor x, Tensor edge_index, Tensor edge_weight) {
//        // 1. 空值校验
//        if (x == null) throw new IllegalArgumentException("节点特征 x 不能为空");
//        if (edge_index == null) throw new IllegalArgumentException("边索引 edge_index 不能为空");
//
//        // 2. 维度校验
//        if (x.dim() != 2) throw new IllegalArgumentException("节点特征必须是 2 维 [N, C]，当前维度：" + x.dim());
//        if (x.size(1) != inChannels) {
//            throw new IllegalArgumentException(
//                    String.format("输入特征维度不匹配！期望 %d，实际 %d", inChannels, x.size(1))
//            );
//        }
//        if (edge_index.dim() != 2 || edge_index.size(0) != 2) {
//            throw new IllegalArgumentException("边索引必须是 [2, E] 形状，当前：" + edge_index.size(0) + "x" + edge_index.size(1));
//        }
//
//        // 3. 边权重维度校验
//        if (edge_weight != null) {
//            if (edge_weight.dim() != 1 || edge_weight.size(0) != edge_index.size(1)) {
//                throw new IllegalArgumentException(
//                        String.format("边权重必须是 1 维 [E]，当前维度：%d，长度：%d",
//                                edge_weight.dim(), edge_weight.size(0))
//                );
//            }
//        }
//    }
//
//    /**
//     * 检查资源是否已释放
//     */
//    private void checkReleased() {
//        if (isReleased) {
//            throw new IllegalStateException("ChebConv 已释放资源，无法继续使用");
//        }
//    }
//
//    /**
//     * 重置参数（符合 PyTorch 规范）
//     */
//    public void resetParameters() {
//        checkReleased();
//        for (int i = 0; i < K; i++) {
//            if (lins[i] != null) lins[i].reset_parameters();
//        }
//        if (bias != null) {
//            torch.zeros_(bias);
//        }
//    }
//
//    /**
//     * 释放所有原生资源（核心：避免内存泄漏）
//     */
//    @Override
//    public void close() {
//        if (!isReleased) {
//            // 释放所有线性层
//            if (lins != null) {
//                for (LinearImpl lin : lins) {
//                    if (lin != null) {
//                        try { lin.close(); } catch (Exception e) {}
//                    }
//                }
//            }
//
//            // 释放偏置
//            if (bias != null) {
//                try { bias.close(); } catch (Exception e) {}
//                try { bias.close(); } catch (Exception e) {}
//            }
//
//            // 释放基类资源
//            super.close();
//
//            isReleased = true;
//        }
//    }
//}