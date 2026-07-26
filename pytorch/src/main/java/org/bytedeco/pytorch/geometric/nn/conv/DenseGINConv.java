package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.Parameter;

/**
 * 稠密图的 GIN 卷积层（继承 MessagePassing 基类）
 * 公式：MLP((1+eps)·X + A·X)
 * 适配稠密张量输入 [B, N, C]，支持可训练 eps 参数
 * 修复点：
 * 1. 必初始化 MessagePassing 基类
 * 2. 仅在构造方法中创建/注册 Parameter（避免重复）
 * 3. 严格管理张量生命周期，避免野指针
 * 4. 修正 GIN 公式计算逻辑
 */
public class DenseGINConv extends MessagePassing {
    private SequentialImpl mlp;       // 多层感知机（核心映射）
    private double eps;              // 初始 eps 值
    private boolean trainEps;        // 是否训练 eps
    private Parameter epsParam;      // 可训练的 eps 参数（仅构造时初始化）

    // 构造方法1：默认 eps=0.0，是否训练 eps 可控
    public DenseGINConv(SequentialImpl mlp, boolean trainEps) {
        this(mlp, 0.0, trainEps);
    }

    // 构造方法2：自定义初始 eps 值，核心构造
    public DenseGINConv(SequentialImpl mlp, double eps, boolean trainEps) {
//        super(false); // 必须初始化 MessagePassing 基类！否则JNI崩溃
        this.mlp = mlp;
        this.eps = eps;
        this.trainEps = trainEps;

        // 仅在构造时初始化+注册 Parameter（避免forward中重复创建）
        if (trainEps) {
//            System.out.println("forward shape DenseGINConv55");
            // 初始化 eps 参数：形状[1]，初始值=eps，类型与输入兼容
            Tensor epsTensor = torch.tensor(eps).to(torch.ScalarType.Float);
            this.epsParam = new Parameter(epsTensor,true);
//            System.out.println("forward shape DenseGINConv66");
            // 注册参数到基类（唯一注册时机）
            register_parameter("eps", this.epsParam);
//            System.out.println("forward shape DenseGINConv77");
        }
        // 注册MLP模块（仅构造时注册一次）
        register_module("mlp", this.mlp);
    }

    /**
     * 前向传播（核心逻辑，无内存泄漏/重复创建）
     * @param x 节点特征 [B, N, in_channels]
     * @param adj 邻接矩阵 [B, N, N]（稠密矩阵）
     * @return 卷积输出 [B, N, out_channels]
     */
    public Tensor forward(Tensor x, Tensor adj) {
        // 1. 设备/类型对齐：保证 epsParam 与输入 x 一致（仅在训练模式）
        Scalar scalar1PlusEps;
        if (trainEps && epsParam != null) {
            // 安全获取 eps 值：先对齐设备，再取标量，避免JNI错误
            Parameter epsAligned = new Parameter(epsParam.to(x.device(), x.scalar_type()));
            // GIN核心：1 + eps（修正之前仅取eps的错误）
            scalar1PlusEps = new Scalar(1.0 + epsAligned.data().item_double());
        } else {
            // 固定eps模式：1 + 初始eps
            scalar1PlusEps = new Scalar(1.0 + this.eps);
        }

        // 2. GIN核心计算：(1+eps)X + A·X（严格遵循公式）
        Tensor selfLoop = x.mul(scalar1PlusEps); // 自环特征
        Tensor neighborSum = adj.matmul(x);      // 邻居求和
        Tensor out = selfLoop.add(neighborSum);  // 合并

        // 3. MLP映射输出（复用注册的MLP模块）
        out = mlp.forward(out);

        return out;
    }

    /**
     * 复写 MessagePassing 基类的 message 方法（必须实现）
     * 稠密图场景下仅保持签名兼容，实际逻辑在forward中通过矩阵乘法实现
     */
    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        return x_j; // 保持基类签名兼容，无实际逻辑
    }

    // ====== 安全的参数访问方法（避免外部直接操作导致内存错误） ======
    public Parameter getEpsParam() {
        return this.epsParam;
    }

    public SequentialImpl getMlp() {
        return this.mlp;
    }

    // ====== 资源释放（严格遵循先子类后基类的顺序） ======
    @Override
    public void close() {
        // 1. 释放子类资源（先释放Tensor/Parameter，再释放Module）
        if (epsParam != null) {
            epsParam.close();
            epsParam = null; // 置空避免野指针
        }
        if (mlp != null) {
            mlp.close();
            mlp = null;
        }
        // 2. 释放基类资源（后调用super，避免基类访问已释放的子类资源）
        super.close();
    }

    // 防止GC时未手动close导致内存泄漏
//    @Override
//    protected void finalize() throws Throwable {
//        try {
//            close();
//        } finally {
//            super.finalize();
//        }
//    }
}

//package org.bytedeco.pytorch.geometric.nn.conv;
//
//import org.bytedeco.pytorch.*;
//import org.bytedeco.pytorch.global.torch;
//import org.bytedeco.pytorch.geometric.demo.layer.SimpleGAT;
//import org.bytedeco.pytorch.*;
//import org.bytedeco.pytorch.global.torch;
//import org.bytedeco.pytorch.geometric.nn.Parameter;
//
///**
// * 稠密图的 GIN 卷积层（继承 MessagePassing 基类）
// * 公式：MLP((1+eps)·X + A·X)
// * 适配稠密张量输入 [B, N, C]，支持可训练 eps 参数
// */
//public class DenseGINConv extends MessagePassing implements AutoCloseable {
//    private SequentialImpl mlp;       // 多层感知机（核心映射）
//    private double eps;              // 初始 eps 值
//    private boolean trainEps;        // 是否训练 eps
//    private Parameter epsParam;      // 可训练的 eps 参数
//
//    // 构造方法1：默认 eps=0.0，是否训练 eps 可控
//    public DenseGINConv(SequentialImpl mlp, boolean trainEps) {
//        this(mlp, 0.0, trainEps);
//    }
//
//    // 构造方法2：自定义初始 eps 值，更灵活
//    public DenseGINConv(SequentialImpl mlp, double eps, boolean trainEps) {
////        super(false); // MessagePassing 构造：false=不使用流控制（稠密图无需）
//        this.mlp = mlp;
//        this.eps = eps;
//        this.trainEps = trainEps;
//
//        // 初始化可训练的 eps 参数（必须是 Parameter 才能参与反向传播）
//        if (trainEps) {
//            this.epsParam = new Parameter(torch.tensor(new Scalar(eps)));
//            register_parameter("eps", epsParam); // 注册到 MessagePassing 基类
//        }
//        register_module("mlp", mlp); // 注册 MLP 模块到基类
//    }
//
//    /**
//     * 前向传播（核心逻辑）
//     * @param x 节点特征 [B, N, in_channels]
//     * @param adj 邻接矩阵 [B, N, N]（稠密矩阵）
//     * @return 卷积输出 [B, N, out_channels]
//     */
//    public Tensor forward(Tensor x, Tensor adj) {
//        // 1. 设备对齐：保证 epsParam 与输入 x 在同一设备
//        if (trainEps && epsParam != null) {
//            epsParam = (Parameter) epsParam.to(x.device(), x.scalar_type());
//        }
//
//        // 2. 计算 (1 + eps) 标量（0 维标量，避免广播错误）
//        Scalar scalar1PlusEps;
//        if (trainEps) {
//            this.epsParam = new Parameter(torch.zeros(new long[]{1})) ; //new Parameter(
//            register_parameter("epsParam", epsParam);
//            scalar1PlusEps = this.epsParam.item();// epsParam.data().item().add(new Scalar(1.0));
//        } else {
//            scalar1PlusEps = new Scalar(1.0 + this.eps);
//        }
//
//        // 3. 稠密图消息传递：邻居求和（A·X） + 自环特征（(1+eps)·X）
//        Tensor out = adj.matmul(x); // 邻居特征求和（对应 MessagePassing 的 propagate）
//        out = out.add(x.mul(scalar1PlusEps)); // 加上自环特征
//
//        // 4. 通过 MLP 映射最终输出
//        out = mlp.forward(out);
//
//        return out;
//    }
//
//    /**
//     * 复写 MessagePassing 基类的 message 方法（必须实现，即使稠密图用不到）
//     * 保持基类签名兼容：(x_j, x_i, edge_index, edge_attr, numNodes)
//     */
//    @Override
//    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
//        // 稠密图场景下，message 逻辑已在 forward 中通过矩阵乘法实现，此处返回 x_j 即可
//        return x_j;
//    }
//
//    // 获取 eps 参数（用于测试）
//    public Parameter getEpsParam() {
//        return epsParam;
//    }
//
//    // 获取 MLP 模块（用于测试）
//    public SequentialImpl getMlp() {
//        return mlp;
//    }
//
//    // 释放资源（覆盖 AutoCloseable）
//    @Override
//    public void close() {
//        if (mlp != null) mlp.close();
//        if (epsParam != null) epsParam.close();
//        super.close(); // 调用 MessagePassing 基类的资源释放
//    }
//}

//public class DenseGINConv extends MessagePassing {
//    private SequentialImpl mlp;
//    private double eps;
//    private boolean trainEps;
//    private Tensor  epsParam; //Parameter
//
//    public  SequentialImpl getMlp() {
//        return  mlp;
//    }
//    public Tensor getEpsParam(){
//        if (trainEps) {
//            return epsParam;
//        } else {
//            return torch.tensor(new Scalar(eps)).to(epsParam.device(), torch.ScalarType.Float);
//        }
//    }
//    public DenseGINConv(SequentialImpl mlp,double eps, boolean trainEps) {
//        this.mlp = mlp;
//        this.trainEps = trainEps;
//        if (trainEps) {
//            this.epsParam = torch.zeros(new long[]{1}); //new Parameter(
//            register_parameter("epsParam", epsParam);
//        }
//        register_module("mlp", mlp);
//    }
//
//    @Override
//    public Tensor forward(Tensor x, Tensor adj) {
//        // 1. Neighbor Sum: A @ X
//        Tensor out = adj.matmul(x);
//
//        // 2. Self: (1 + eps) * X
//        Tensor scalarEps;
//        if (trainEps) {
//            scalarEps = epsParam.add(new Scalar(1.0));
//        } else {
//            scalarEps = torch.tensor( new Scalar(1.0 + eps)).to(x.device(), torch.ScalarType.Float);
//        }
//
//        // 3. Combine
//        // out = out + x * (1+eps)
//        out = out.add(x.mul(scalarEps));
//
//        // 4. MLP
//        return mlp.forward(out);
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
