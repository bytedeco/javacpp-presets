package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;
//import org.gnn.framework.nn.org.bytedeco.pytorch.geometric.nn.conv.MessagePassing;
//import java.lang.reflect.Parameter;
//package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.Parameter; // 引入自定义 Parameter

import static org.bytedeco.pytorch.global.torch.*;

/**
 * 完全模仿 Python 版 PyTorch Geometric 的 GINConv
 * 核心逻辑：out = MLP((1+eps)·x + sum(x_j))
 * 对齐 Python 版 API 和行为
 */
public class GINConv extends MessagePassing {
    private final Module mlp;          // 外部传入的 MLP (Sequential/Linear)
    private final Parameter eps;       // 模仿 Python 的 Parameter 类型
    private final boolean trainEps;    // 是否训练 eps

    // ========== 构造函数（完全对齐 Python 版 PyG） ==========
    /**
     * Python 对应：GINConv(mlp, train_eps=True)
     * @param mlp 外部定义的 MLP (SequentialImpl/LinearImpl)
     * @param trainEps 是否将 eps 作为可训练参数
     */
    public GINConv(Module mlp, boolean trainEps) {
        super("add"); // GIN 固定使用 add 聚合（sum）
        // 参数校验（模仿 Python 的参数检查）
        if (mlp == null) {
            throw new IllegalArgumentException("mlp must not be None");
        }
        this.mlp = mlp;
        this.trainEps = trainEps;

        // 1. 注册 MLP 子模块（Python: self.mlp = mlp; self.add_module('mlp', mlp)）
        register_module("mlp", mlp);

        // 2. 初始化 eps（完全模仿 Python 版逻辑）
        // Python: self.eps = torch.nn.Parameter(torch.tensor(0.0)) if train_eps else torch.tensor(0.0)
        Tensor epsTensor = zeros(new long[]{1}, new TensorOptions().dtype(new ScalarTypeOptional(torch.kFloat())));
        if (trainEps) {
            // 可训练：创建 Parameter（requires_grad=True）
            this.eps = new Parameter(epsTensor, true);
            register_parameter("eps", this.eps); // 注册为可训练参数
        } else {
            // 不可训练：创建 Parameter（requires_grad=False）
            this.eps = new Parameter(epsTensor, false);
            register_buffer("eps", this.eps); // 注册为缓冲区（Python: self.register_buffer）
        }
    }

    /**
     * 简化构造（Python 对应：GINConv(mlp)）
     */
    public GINConv(Module mlp) {
        this(mlp, false);
    }

    // ========== 前向传播（逐行模仿 Python 版） ==========
    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        // Python 版核心逻辑：
        // x_j = propagate(edge_index, x=x, aggr='add')
        // out = (1 + self.eps) * x + x_j
        // out = self.mlp(out)
        // return out

        // 1. 聚合邻居特征: sum(x_j)（模仿 Python 的 propagate）
        Tensor neighborSum = propagate(edge_index, x);

        // 2. 计算 (1 + eps) * x + sum(x_j)（完全对齐 Python）
        Tensor factor = this.eps.add(new Scalar(1.0f)); // (1 + eps)
        Tensor selfFeat = x.mul(factor); // (1+eps)·x
        Tensor out = selfFeat.add(neighborSum); // 核心公式

        // 3. 通过 MLP 映射（模仿 Python 的 self.mlp(out)）
        return forwardMLP(out);
    }

    // ========== MessagePassing 基类方法（模仿 Python） ==========
    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        // Python: def message(self, x_j): return x_j
        return x_j;
    }

    // ========== 工具方法（模仿 Python 的 MLP 调用） ==========
    private Tensor forwardMLP(Tensor input) {
        try {
            // 模仿 Python 的 mlp.forward(input)
            if (mlp instanceof SequentialImpl) {
                return ((SequentialImpl) mlp).forward(input);
            } else if (mlp instanceof LinearImpl) {
                return ((LinearImpl) mlp).forward(input);
            } else {
                return mlp.asSequential().forward(input);
            }
        } catch (Exception e) {
            throw new RuntimeException("MLP forward failed: " + e.getMessage(), e);
        }
    }

    // ========== 模仿 Python 的属性访问 ==========
    public Module mlp() {
        return this.mlp;
    }

    public Parameter eps() {
        return this.eps;
    }

    public boolean train_eps() {
        return this.trainEps;
    }
}

//public class GINConv extends MessagePassing {
//    private Module mlp; // 传入外部定义的 MLP (SequentialImpl)
//    private Tensor eps;
//
//    public GINConv(Module mlp, boolean trainEps) {
//        super("add");
//        this.mlp = mlp;
//        register_module("mlp", mlp);
//
//        // 使用 new Scalar 初始化 Tensor
//        this.eps = torch.zeros(new long[]{1}, new TensorOptions());
//        if (trainEps) {
//            register_parameter("eps", eps);
//        }
//    }
//
//    @Override
//    public Tensor forward(Tensor x, Tensor edge_index) {
////        Tensor out = propagate(edge_index, x, null); // sum(neighbors)
////
////        // 计算 (1 + eps)
////        // 关键点：数值 1.0 必须用 new Scalar(1.0)
////        Tensor one = torch.full(new long[]{1}, new Scalar(1.0), x.options());
////        Tensor factor = one.add(eps);
////
////        // out = out + (1+eps)*x
////        Tensor self = x.mul(factor);
////        out = out.add(self);
////
////        // 注意：JavaCPP 中 Module 的 forward 通常需要转换类型，
////        // 这里假设 mlp 是 LinearImpl 或 SequentialImpl，直接调 forward 可能需要 cast
////        // 为了通用，建议 mlp.asModule().forward(out) (取决于具体版本API)
////        // 这里演示核心逻辑
////        return mlp.asSequential().forward(out);
//
//        // 1. 聚合邻居特征: sum(x_j)
//        // 得到的结果 shape: [N, in_channels]
//        Tensor neighborSum = propagate(edge_index, x, new long[]{x.size(0), x.size(0)});
//
//        // 2. 计算 (1 + eps) * x
//        // 优化点：直接利用 Tensor 的标量加法和乘法，无需创建 one Tensor
//        // out = (1 + eps) * x + neighborSum
//        Tensor out = x.mul(eps.add(new Scalar(1.0))).add(neighborSum);
//
//        // 3. 通过 MLP 映射到目标维度
//        // 注意：在 JavaCPP 中，Module 的调用通常需要明确子类类型
//        return forwardMLP(out);
//    }
//
//
//    /**
//     * 由于 Java 是强类型，直接 mlp.forward 会报错。
//     * 根据你传入的是 Sequential 还是 Linear 转换。
//     */
//    private Tensor forwardMLP(Tensor input) {
//        if (mlp instanceof SequentialImpl) {
//            return ((SequentialImpl) mlp).forward(input);
//        } else if (mlp instanceof LinearImpl) {
//            return ((LinearImpl) mlp).forward(input);
//        }
//        // 如果是自定义 ModuleImpl
//        return mlp.asSequential().forward(input);
//    }
//
//    @Override
//    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
//        // GIN 的 message 就是简单的特征传递
//        return x_j;
//    }
//}