package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.Parameter;

import java.util.HashMap;
import java.util.Map;

/**
 * 修正版 ClusterGCNConv：
 * 1. 继承 MessagePassing + 重写 propagate
 * 2. 所有临时Tensor均存入父类的tensors容器
 * 3. 完善资源释放逻辑
 */
public class ClusterGCNConv extends MessagePassing {
    private LinearImpl lin; // 线性变换层
    private float diagLambda; // 对角线增强系数
    private boolean addSelfLoops; // 是否添加自环
    private Parameter bias; // 可训练偏置

    // 构造函数：调用父类构造器，指定聚合方式为add
    public ClusterGCNConv(long inChannels, long outChannels, float diagLambda, boolean addSelfLoops, boolean hasBias) {
        super("add");
        this.diagLambda = diagLambda;
        this.addSelfLoops = addSelfLoops;

        // 初始化线性层（LinearImpl：输入维度→输出维度）
        this.lin = new LinearImpl(inChannels, outChannels);
        register_module("lin", lin);

        // 初始化偏置（包装为Parameter，支持反向传播）
        if (hasBias) {
            Tensor biasTensor = torch.zeros(new long[]{outChannels}, torch.dtype(torch.ScalarType.Float));
            this.bias = new Parameter(biasTensor);
            register_parameter("bias", this.bias);
        } else {
            this.bias = null;
        }
    }

    /**
     * 前向传播：ClusterGCNConv核心逻辑
     * @param x 节点特征 [N, inChannels]
     * @param edge_index 边索引 [2, E]
     * @return 卷积输出 [N, outChannels]
     */
    public Tensor forward(Tensor x, Tensor edge_index) {
        // 1. 输入合法性校验
        if (x.dim() != 2) {
            throw new IllegalArgumentException("节点特征x必须是2维张量，当前维度：" + x.dim());
        }
        if (edge_index.dim() != 2 || edge_index.size(0) != 2) {
            throw new IllegalArgumentException("边索引edge_index必须是[2, E]维度，当前维度：" + edge_index);
        }
        long N = x.size(0); // 节点数

        // 2. 添加自环（如果开启）
        Tensor edgeIndexWithSelfLoops = edge_index;
        if (addSelfLoops) {
            // 生成自环：[[0,1,...,N-1], [0,1,...,N-1]]
            Tensor selfLoops = torch.arange(new Scalar(0), new Scalar(N)).unsqueeze(0).repeat(2, 1);
            tensors.push_back(selfLoops); // 存入tensors
            // 拼接原有边 + 自环
            edgeIndexWithSelfLoops = torch.cat(new TensorVector(edge_index, selfLoops), 1);
            tensors.push_back(edgeIndexWithSelfLoops); // 存入tensors
        }

        // 3. 计算对称归一化系数 D^(-1/2) * A * D^(-1/2)
        // 3.1 初始化边权重（全1）
        Tensor edgeWeight = torch.ones(new long[]{edgeIndexWithSelfLoops.size(1)}, x.options());
        tensors.push_back(edgeWeight);
        // 3.2 拆分边索引为row（起点）、col（终点）
        Tensor row = edgeIndexWithSelfLoops.select(0, 0);
        Tensor col = edgeIndexWithSelfLoops.select(0, 1);
        tensors.push_back(row);
        tensors.push_back(col);
        // 3.3 计算度矩阵 D = sum(A, dim=1)
        Tensor deg = torch.zeros(new long[]{N}, x.options());
        tensors.push_back(deg);
        deg.scatter_add_(0, row, edgeWeight); // 行度
        deg.scatter_add_(0, col, edgeWeight); // 列度（无向图，行/列度一致）
        // 3.4 对称归一化：D^(-1/2)，处理度为0的情况（避免inf）
        Tensor degInvSqrt = deg.pow(new Scalar(-0.5));
        tensors.push_back(degInvSqrt);
        degInvSqrt.masked_fill_(degInvSqrt.isinf(), new Scalar(0.0));
        // 3.5 计算归一化因子：norm = D^(-1/2)[row] * D^(-1/2)[col]
        Tensor norm = degInvSqrt.index_select(0, row).mul(degInvSqrt.index_select(0, col));
        tensors.push_back(norm);

        // 4. 调用重写的propagate方法，执行消息传递
        Map<String, Tensor> kwargs = new HashMap<>();
        kwargs.put("norm", norm);
        Tensor out = propagate(edgeIndexWithSelfLoops, x, kwargs);
        tensors.push_back(out);

        // 5. 对角线增强：out = out + lambda * x
        if (diagLambda != 0) {
            Tensor lambdaX = x.mul(new Scalar(diagLambda));
            tensors.push_back(lambdaX);
            out = out.add(lambdaX);
            tensors.push_back(out);
        }

        // 6. 线性变换 W
        out = lin.forward(out);
        tensors.push_back(out);

        // 7. 添加偏置
        if (bias != null) {
            out = out.add(bias.data());
            tensors.push_back(out);
        }

        return out;
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        return null;
    }

    /**
     * 核心：重写父类propagate方法，自定义消息传递流程
     *
     * @param edge_index 边索引 [2, E]
     * @param x          节点特征 [N, inChannels]
     * @param kwargs     额外参数（norm：归一化系数）
     * @return 聚合后的特征 [N, inChannels]
     */
//    @Override
    public Tensor propagate(Tensor edge_index, Tensor x, Map<String, Tensor> kwargs) {
        // 步骤1：拆分边索引为row（起点）、col（终点）
        Tensor row = edge_index.select(0, 0); //.asTensor();
        Tensor col = edge_index.select(0, 1);//.asTensor();
        tensors.push_back(row);
        tensors.push_back(col);

        // 步骤2：提取邻居节点特征 x_j = x[col]
        Tensor x_j = x.index_select(0, col);
        tensors.push_back(x_j);

        // 步骤3：调用message方法构造消息（应用归一化）
        Tensor msg = message(x_j, kwargs);
        tensors.push_back(msg);

        // 步骤4：调用父类aggregate方法聚合消息（求和）
        Tensor out = super.aggregate(msg, row, x.size(0));
        tensors.push_back(out);

        return out;
    }

    /**
     * 重写message方法：给邻居特征应用归一化系数
     *
     * @param x_j    邻居节点特征 [E, inChannels]
     * @param kwargs 额外参数（norm：归一化系数）
     * @return 归一化后的邻居特征 [E, inChannels]
     */
//    @Override
    public Tensor message(Tensor x_j, Map<String, Tensor> kwargs) {
        Tensor norm = kwargs.get("norm");
        if (norm == null) {
            throw new IllegalArgumentException("kwargs必须包含norm参数（归一化系数）");
        }
        // norm形状[E] → 扩展为[E,1]，广播到特征维度
        return x_j.mul(norm.view(-1, 1));
    }

    /**
     * 重写参数获取方法：返回所有可训练参数
     * @return ParameterDict 参数字典
     */
//    @Override
//    public ParameterDict named_parameters() {
//        ParameterDict params = new ParameterDict();
//        // 添加线性层权重
//        if (lin != null && lin.weight() != null) {
//            params.put("lin.weight", lin.weight());
//        }
//        // 添加线性层偏置（如果有）
//        if (lin != null && lin.bias() != null) {
//            params.put("lin.bias", lin.bias());
//        }
//        // 添加卷积层偏置（如果有）
//        if (bias != null) {
//            params.put("bias", bias);
//        }
//        return params;
//    }

    /**
     * 重写close方法：释放所有资源
     */
    @Override
    public void close() {
        // 1. 调用父类close，释放tensors中的临时Tensor
        super.close();
        // 2. 释放线性层
        if (lin != null) {
            lin.close();
        }
        // 3. 释放偏置参数
        if (bias != null) {
            bias.close();
        }
    }
}

//package org.bytedeco.pytorch.geometric.nn.conv;
//
//import org.bytedeco.pytorch.*;
//import org.bytedeco.pytorch.global.torch;
//import org.bytedeco.pytorch.geometric.nn.Parameter;
//
//import java.util.HashMap;
//import java.util.Map;
//
/// **
// * 修正版 ClusterGCNConv：严格遵循 PyTorch Geometric 标准实现
// * 核心：对称归一化 + 正确自环处理 + 标准 MessagePassing 调用流程
// */
//public class ClusterGCNConv extends MessagePassing {
//    private LinearImpl lin; // 节点特征线性变换层
//    private float diagLambda; // 对角线增强系数
//    private boolean addSelfLoops; // 是否添加自环
//    private Parameter bias; // 可训练偏置（包装为 Parameter）
//
//    /**
//     * 构造函数
//     * @param inChannels 输入特征维度
//     * @param outChannels 输出特征维度
//     * @param diagLambda 对角线增强系数（lambda）
//     * @param addSelfLoops 是否添加自环
//     * @param hasBias 是否使用偏置
//     */
//    public ClusterGCNConv(long inChannels, long outChannels, float diagLambda, boolean addSelfLoops, boolean hasBias) {
//        super("add"); // 聚合方式：求和
//        this.diagLambda = diagLambda;
//        this.addSelfLoops = addSelfLoops;
//
//        // 初始化线性层（严格使用 LinearImpl）
//        this.lin = new LinearImpl(inChannels, outChannels);
//        register_module("lin", lin); // 注册子模块
//
//        // 初始化偏置（正确包装为 Parameter）
//        if (hasBias) {
//            Tensor biasTensor = torch.zeros(new long[]{outChannels}, torch.dtype(torch.ScalarType.Float));
//            this.bias = new Parameter(biasTensor);
//            register_parameter("bias", this.bias);
//        } else {
//            this.bias = null;
//        }
//    }
//
//    /**
//     * 前向传播（核心逻辑）
//     * @param x 节点特征 [N, inChannels]
//     * @param edge_index 边索引 [2, E]
//     * @return 卷积输出 [N, outChannels]
//     */
//    @Override
//    public Tensor forward(Tensor x, Tensor edge_index) {
//        // 1. 输入校验
//        if (x.dim() != 2) {
//            throw new IllegalArgumentException("节点特征 x 必须是 2 维张量，当前维度：" + x.dim());
//        }
//        if (edge_index.dim() != 2 || edge_index.size(0) != 2) {
//            throw new IllegalArgumentException("边索引 edge_index 必须是 [2, E] 维度，当前维度：" + edge_index);
//        }
//        long N = x.size(0); // 节点数
//
//        // 2. 添加自环（如果开启）
//        if (addSelfLoops) {
//            Tensor selfLoops = torch.arange(new Scalar(0), new Scalar(N)).unsqueeze(0).repeat(2, 1);
//            edge_index = torch.cat(new TensorVector(edge_index, selfLoops), 1);
//        }
//
//        // 3. 计算对称归一化系数 D^(-1/2) * A * D^(-1/2)
//        Tensor edgeWeight = torch.ones(new long[]{edge_index.size(1)}, x.options()); // 边权重初始化为1
//        Tensor row = edge_index.select(0, 0); // 边的起点 [E]
//        Tensor col = edge_index.select(0, 1); // 边的终点 [E]
//
//        // 计算度矩阵 D = sum(A, dim=1)
//        Tensor deg = torch.zeros(new long[]{N}, x.options());
//        deg.scatter_add_(0, row, edgeWeight); // 行度
//        deg.scatter_add_(0, col, edgeWeight); // 列度（无向图，行/列度一致）
//
//        // 对称归一化：D^(-1/2)，处理度为0的情况（避免inf）
//        Tensor degInvSqrt = deg.pow(new Scalar(-0.5));
//        degInvSqrt.masked_fill_(degInvSqrt.isinf(), new Scalar(0.0));
//
//        // 计算归一化因子：norm = D^(-1/2)[row] * D^(-1/2)[col]
//        Tensor norm = degInvSqrt.index_select(0, row).mul(degInvSqrt.index_select(0, col));
//
//        // 4. 消息传递：聚合邻居特征 (D^(-1/2) * A * D^(-1/2)) @ x
//        // 正确传递 norm 到 message 方法（通过 Map 传递 kwargs）
//        Map<String, Tensor> kwargs = new HashMap<>();
//        kwargs.put("norm", norm);
//        Tensor out = propagate(edge_index, x, kwargs);
//
//        // 5. 对角线增强：out = out + lambda * x
//        if (diagLambda != 0) {
//            out = out.add(x.mul(new Scalar(diagLambda)));
//        }
//
//        // 6. 线性变换 W
//        out = lin.forward(out);
//
//        // 7. 添加偏置
//        if (bias != null) {
//            out = out.add(bias.data()); // Parameter 需通过 .data() 获取 Tensor
//        }
//
//        return out;
//    }
//
// 
//    
//    /**
//     * 消息构造函数（MessagePassing 核心方法）
//     * @param x_j 邻居节点特征 [E, inChannels]
//     * @param kwargs 额外参数（包含归一化系数 norm）
//     * @return 构造后的消息 [E, inChannels]
//     */
////    @Override
//    public Tensor message(Tensor x_j, Map<String, Tensor> kwargs) {
//        // 获取预计算的归一化系数
//        Tensor norm = kwargs.get("norm");
//        // 应用归一化：x_j * norm (广播到特征维度)
//        return x_j.mul(norm.view(-1, 1));
//    }
//
//    // 覆盖无用的 message 重载方法（避免编译警告）
//    @Override
//    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
//        throw new UnsupportedOperationException("请使用带 kwargs 的 message 方法");
//    }
//}

//package org.bytedeco.pytorch.geometric.nn.conv;
//
//import org.bytedeco.pytorch.*;
//import org.bytedeco.pytorch.global.torch;
//
///**
// * 严格使用 LinearImpl 实现 torch_geometric.nn.conv.ClusterGCNConv
// * 针对大规模分簇图优化的卷积算子。
// */
//public class ClusterGCNConv extends MessagePassing {
//    private LinearImpl lin; // 变换矩阵 W
//    private float diagLambda;
//    private boolean addSelfLoops;
//    private Tensor bias;
//
//    public ClusterGCNConv(long inChannels, long outChannels, float diagLambda, boolean addSelfLoops, boolean hasBias) {
//        super("add");
//        this.diagLambda = diagLambda;
//        this.addSelfLoops = addSelfLoops;
//
//        // 严格使用 LinearImpl
//        this.lin = new LinearImpl(inChannels, outChannels);
//        register_module("lin", lin);
//
//        if (hasBias) {
//            this.bias = torch.zeros(new long[]{outChannels});
//            register_parameter("bias", bias);
//        }
//    }
//
//    /**
//     * @param x          节点特征 [N, inChannels]
//     * @param edge_index 子图边索引 [2, E_sub]
//     */
//    @Override
//    public Tensor forward(Tensor x, Tensor edge_index) {
//        long N = x.size(0);
//
//        // 1. 计算对称归一化系数 (D^-1 * A)
//        // ClusterGCN 通常使用简化的随机游走归一化
//        Tensor edge_weight = torch.ones(new long[]{edge_index.size(1)}, x.options());
//        Tensor row = edge_index.select(0, 0);
//        Tensor deg = torch.zeros(new long[]{N}, x.options());
//        deg.scatter_add_(0, row, edge_weight);
//
//        // 计算归一化因子: 1 / deg
//        Tensor degInv = deg.pow(new Scalar(-1.0));
//        degInv.masked_fill_(degInv.isinf(), new Scalar(0));
//        Tensor norm = degInv.index_select(0, row);
//
//        // 2. 邻居聚合: A_hat @ x
//        Tensor out = propagate(edge_index, x, norm);
//
//        // 3. 对角线增强: (A_hat + lambda * I) @ x
//        // 即 out = out + lambda * x
//        if (diagLambda != 0) {
//            out = out.add(x.mul(new Scalar(diagLambda)));
//        }
//
//        // 4. 线性变换 W
//        out = lin.forward(out);
//
//        if (bias != null) {
//            out = out.add(bias);
//        }
//
//        return out;
//    }
//
//    @Override
//    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
//        // 应用预计算的归一化系数
//        return x_j.mul(edge_attr.view(-1, 1));
//    }
//}
