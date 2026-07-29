package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;

/**
 * 严格使用 LinearImpl 规范实现 torch_geometric.nn.conv.AntiSymmetricConv
 * 借鉴 ODE 稳定性理论，通过反对称矩阵确保深层网络的数值稳定性。
 */
public class AntiSymmetricConv extends Module {
    private MessagePassing phi;     // 内部消息传递算子 (默认为 GCNConv)
    private LinearImpl lin;         // 用于构造反对称权重矩阵 W 的基础层
    private int numIters;           // 迭代次数 (离散化步数)
    private float epsilon;          // 步长
    private float gamma;            // 扩散/阻尼系数
    private Tensor bias;

    public AntiSymmetricConv(int inChannels, MessagePassing phi, int numIters,
                             float epsilon, float gamma, boolean hasBias) {
        super();
        this.numIters = numIters;
        this.epsilon = epsilon;
        this.gamma = gamma;

        // 1. 内部算子注册
        this.phi = (phi != null) ? phi : new GCNConv(inChannels, inChannels);
        register_module("phi", this.phi);

        // 2. 权重矩阵注册 (Strictly LinearImpl)
        // 用于构造 W - W^T + gamma*I
        this.lin = new LinearImpl(inChannels, inChannels);
        register_module("lin", lin);

        if (hasBias) {
            this.bias = torch.zeros(new long[]{inChannels});
            register_parameter("bias", bias);
        }
    }

    /**
     * @param x          节点特征 [N, inChannels]
     * @param edge_index 边索引 [2, E]
     */

    public Tensor forward(Tensor x, Tensor edge_index) {
        // 构造反对称权重矩阵: A = W - W^T
        Tensor W = lin.weight();
        Tensor antiW = W.sub(W.t());

        // 构造扩散控制项: -gamma * I
        Tensor identity = torch.eye(x.size(1), x.options());
        Tensor structuralParam = antiW.sub(identity.mul(new Scalar(gamma)));

        Tensor h = x;
        for (int i = 0; i < numIters; i++) {
            // 核心演化公式:
            // h_next = h + eps * (sigma(phi(h, edge_index) @ (W - W^T - gamma*I) + bias))

            // 1. 邻域传播
            Tensor out;
            if (phi instanceof GCNConv) {
                out = ((GCNConv) phi).forward(h, edge_index);
            } else {
                // 如果是其他类型，以此类推，或者使用通用的调用方式
                out = phi.asSequential().forward(h, edge_index); // 尝试匹配三个参数的基类方法
            }
//            Tensor out = phi.forward(h, edge_index);

            // 2. 乘法变换 (应用反对称约束)
            out = torch.matmul(out, structuralParam);

            if (bias != null) {
                out = out.add(bias);
            }

            // 3. 非线性激活与残差更新 (类似 Euler 积分)
            h = h.add(torch.tanh(out).mul(new Scalar(epsilon)));
        }

        return h;
    }
}