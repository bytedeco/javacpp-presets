package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.Parameter;
import org.bytedeco.pytorch.geometric.utils.Scatter;

/**
 * 实现 torch_geometric.nn.conv.AGNNConv
 * 基于余弦相似度的动态注意力传播层。
 */
public class AGNNConv extends MessagePassing {
    private Tensor beta; // 可学习的缩放参数 β
    private Parameter betaParam; // 可学习参数 β（Parameter 类型支持梯度）
    private Tensor betaBuffer;   // 非学习参数 β（Buffer 类型无梯度）
    private boolean requiresGrad;
    private boolean isReleased = false; // 资源释放标记

    public AGNNConv(boolean requiresGrad) {
        super("add");

        // 初始化 beta 为 1.0
        this.beta = torch.ones(new long[]{1});

        if (requiresGrad) {
            this.betaParam = new Parameter(register_parameter("beta", beta));
        } else {
            // 如果不需要梯度，作为普通的 buffer 注册
            this.betaBuffer = register_buffer("beta", beta);
        }
    }

//    public AGNNConv(boolean requiresGrad) {
//        super("add");
//        this.requiresGrad = requiresGrad;
//
//        // 初始化 beta 为 1.0
//        Tensor betaInit = torch.ones(new long[]{1}).to(torch.float32);
//
//        if (requiresGrad) {
//            // 注册为可学习参数（Parameter 类型）
//            this.betaParam = new Parameter(betaInit);
//            register_module("beta", this.betaParam); // 注册到模块
//        } else {
//            // 注册为非学习参数（Buffer 类型）
//            this.betaBuffer = betaInit;
//        }
//    }

    /**
     * 获取 beta 张量（统一对外接口）
     */
    private Tensor getBeta() {
        return requiresGrad ? betaParam : betaBuffer;
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        long N = x.size(0);

        // 1. 对特征进行 L2 归一化，方便后续计算余弦相似度
        // x_norm = x / ||x||
        Tensor xNorm = x.div(x.norm(new ScalarOptional(new Scalar(2)),new long[]{-1}, true).clamp_min(new Scalar(1e-12)));

        // 2. 传播
        return propagate(edge_index, x, xNorm);
    }

    /**
     * 重载 propagate 逻辑
     */
    public Tensor propagate(Tensor edge_index, Tensor x, Tensor xNorm) {
        long N = x.size(0);
        Tensor sourceIdx = edge_index.select(0, 0);
        Tensor targetIdx = edge_index.select(0, 1);

        // Lift
        Tensor xj = x.index_select(0, sourceIdx);
        Tensor xNorm_j = xNorm.index_select(0, sourceIdx);
        Tensor xNorm_i = xNorm.index_select(0, targetIdx);

        // --- 计算注意力系数 alpha ---
        // 1. 计算余弦相似度: cos(i, j) = xNorm_i · xNorm_j
        Tensor cosSim = (xNorm_i.mul(xNorm_j)).sum(-1);

        // 2. 缩放并应用 Softmax: alpha = exp(beta * cosSim)
        Tensor logits = cosSim.mul(beta);
        Tensor alpha = scatter_softmax(logits, targetIdx, N);

        // --- 消息加权 ---
        Tensor msg = xj.mul(alpha.unsqueeze(-1));

        // 3. 聚合
        return aggregate(msg, targetIdx, N);
    }

    private Tensor scatter_softmax(Tensor src, Tensor index, long dimSize) {
        // 使用数值稳定的实现
        Tensor maxVal = Scatter.scatter(src, index, dimSize, "max");
        Tensor out = src.sub(maxVal.index_select(0, index)).exp();
        Tensor sum = Scatter.scatter(out, index, dimSize, "add");
        return out.div(sum.index_select(0, index).add(new Scalar(1e-16)));
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        return x_j; // 占位
    }

    public void resetParameters() {
        checkReleased();
        if (requiresGrad) {
            betaParam.fill_(new Scalar(1.0f)); // 重置为 1.0
        } else {
//            betaBuffer.fill_(new Scalar(1.0f));
        }
    }


    private void checkReleased() {
        if (isReleased) {
            throw new IllegalStateException("AGNNConv 已释放资源，无法继续使用");
        }
    }

    @Override
    public void close() {
        if (!isReleased) {
            // 释放 beta 参数/缓冲区
            if (betaParam != null) {
                betaParam.close();
            }
            if (betaBuffer != null) {
                betaBuffer.close();
            }
            // 释放基类资源
            super.close();
            isReleased = true;
        }
    }

    public boolean isRequiresGrad() { return requiresGrad; }
    public boolean isReleased() { return isReleased; }
}