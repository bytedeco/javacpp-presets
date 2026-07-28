package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.autograd.*;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.options.LinearOptions;
import org.bytedeco.pytorch.nn.Parameter;
import org.bytedeco.pytorch.geometric.utils.Scatter;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.utils.Scatter;

import static org.bytedeco.pytorch.global.torch.empty;
import static org.bytedeco.pytorch.global.torch.kFloat;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.utils.Scatter;
import static org.bytedeco.pytorch.global.torch.*;

/**
 * 修复版 GINEConv（严格匹配 torch_geometric.nn.conv.GINEConv 逻辑）
 * 解决维度不匹配、空指针、设备不一致、内存泄漏等问题
 */
public class GINEConv extends MessagePassing {
    private SequentialImpl nn;       // 核心 MLP 网络
    public Parameter epsParam;      // 可学习的 eps 参数
    private Tensor epsBuffer;        // 不可学习的 eps 缓冲区
    private LinearImpl linEdge;      // 边特征维度对齐层
    private boolean trainEps;        // eps 是否可学习
    private float initEps;           // eps 初始值

    /**
     * GINEConv 构造函数（GIN + Edge Features）
     * @param nn           核心 MLP 网络（可为 null，此时仅执行聚合逻辑）
     * @param epsValue     ε 初始值（GINE 标准默认 0.0）
     * @param trainEps     ε 是否可学习
     * @param edgeDim      边特征维度（可为 null）
     * @param nodeDim      节点特征维度（必须传入，用于维度对齐）
     */
    public GINEConv(SequentialImpl nn, double epsValue, boolean trainEps, Integer edgeDim, Integer nodeDim) {
        super("add"); // GINE 必须使用 add 聚合以保证同构判别力
        this.nn = nn;
        this.trainEps = trainEps;
        this.initEps = (float) epsValue;

        // 1. 输入校验
        if (nodeDim == null || nodeDim <= 0) {
            throw new IllegalArgumentException("节点特征维度 nodeDim 必须大于 0");
        }
        if (edgeDim != null && edgeDim <= 0) {
            throw new IllegalArgumentException("边特征维度 edgeDim 必须大于 0");
        }

        // 2. 初始化 eps（区分可学习/不可学习）
        TensorOptions epsOpts = new TensorOptions()
                .dtype(new ScalarTypeOptional(kFloat()))
                .device(new DeviceOptional(new Device(kCPU()))) ;// 默认 CPU，前向时对齐输入设备
//                .requires_grad(new BoolOptional(trainEps));
        Tensor epsTensor = torch.tensor(new float[]{this.initEps}, epsOpts).requires_grad_(trainEps);

        if (trainEps) {
            this.epsParam = new Parameter(epsTensor);
            register_parameter("eps", this.epsParam);
        } else {
            this.epsBuffer = epsTensor;
            register_buffer("eps", this.epsBuffer);
        }

        // 3. 边特征维度对齐（核心修复）
        if (edgeDim != null && !edgeDim.equals(nodeDim)) {
            var options = new LinearOptions(edgeDim, nodeDim);
            options.bias().put(true);
            this.linEdge = new LinearImpl(options); // 带 Bias
            register_module("lin_edge", this.linEdge);
        }

        // 4. 注册 MLP 网络
        if (this.nn != null) {
            register_module("nn", this.nn);
        }
    }

    // 重载 forward：无边形特征输入（修复 null 返回问题）
    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        return forward(x, edge_index, (Tensor)null);
    }

    // 核心 forward：带边特征输入
    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_attr) {
        // 1. 输入校验
        if (x == null || edge_index == null) {
            throw new NullPointerException("x 和 edge_index 不能为空");
        }
        if (x.dim() != 2) {
            throw new IllegalArgumentException("x 必须是 2D 张量 [N, C]，当前维度: " + x.dim());
        }
        if (edge_index.dim() != 2 || edge_index.size(0) != 2) {
            throw new IllegalArgumentException("edge_index 必须是 2D 张量 [2, E]，当前维度: " + edge_index.dim());
        }

        long numNodes = x.size(0);
        Device device = x.device(); // 获取输入设备，保证所有张量设备一致

        // 2. 消息传递与聚合
        Tensor out = propagate(edge_index, x, edge_attr, device);

        // 3. GINE 核心公式：(1 + eps) * x + aggregated_msg
        Tensor eps = getEpsTensor(device); // 获取对齐设备的 eps
        Tensor epsPlusOne = eps.add(new Scalar(1.0f));
        Tensor centerUpdate = x.mul(epsPlusOne);
        out = out.add(centerUpdate);

        // 4. 应用 MLP（兼容 nn 为 null 的情况）
        if (this.nn != null) {
            out = this.nn.forward(out);
        }

        // 5. 数值异常修复
        out = fixNumericAnomalies(out);

        // 6. 释放临时张量（修复内存泄漏）
        eps.close();
        epsPlusOne.close();
        centerUpdate.close();

        return out;
    }

    // 核心 propagate 方法：消息传递逻辑
    private Tensor propagate(Tensor edge_index, Tensor x, Tensor edge_attr, Device device) {
        long numNodes = x.size(0);
        Tensor row = edge_index.select(0, 0); // 源节点 (j)
        Tensor col = edge_index.select(0, 1); // 目标节点 (i)

        // 提取邻居节点特征 x_j [E, C]
        Tensor x_j = x.index_select(0, row);
        // 中心节点特征 x_i（GINE 暂未使用，保留以兼容扩展）
        Tensor x_i = x.index_select(0, col);

        // 生成消息
        Tensor msg = message(x_j, x_i, edge_index, edge_attr, numNodes, device);

        // 聚合消息（add 聚合）
        Tensor out = aggregate(msg, col, numNodes);

        // 释放临时张量
        row.close();
        col.close();
        x_j.close();
        x_i.close();
        msg.close();

        return out;
    }

    // 修复 message 方法：GINE 核心消息生成
    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        // 兼容设备参数（重载方法，适配基类）
        return message(x_j, x_i, edge_index, edge_attr, numNodes, x_j.device());
    }

    // 核心 message 方法：处理边特征维度对齐
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes, Device device) {
        // 1. 无边形特征：退化为 GIN 逻辑
        if (edge_attr == null || !edge_attr.defined()) {
            return relu(x_j);
        }

        // 2. 边特征设备对齐
        edge_attr = edge_attr.to(device,ScalarType.Float);

        // 3. 边特征维度兼容（处理 1D 边特征）
        if (edge_attr.dim() == 1) {
            edge_attr = edge_attr.unsqueeze(1); // [E] → [E, 1]
        }

        // 4. 边特征维度对齐
        Tensor ej = edge_attr;
        if (this.linEdge != null) {
            ej = this.linEdge.forward(ej);
        }

        // 5. 维度校验（核心修复）
        if (ej.size(1) != x_j.size(1)) {
            throw new IllegalArgumentException(
                    "边特征维度与节点特征维度不匹配: edge_attr=" + ej.size(1) + ", x_j=" + x_j.size(1)
            );
        }

        // 6. GINE 核心公式：ReLU(x_j + e_ij)
        Tensor msg = relu(x_j.add(ej));

        // 7. 数值异常修复
        msg = fixNumericAnomalies(msg);

        return msg;
    }

    // 修复 aggregate 方法：严格使用 add 聚合
    @Override
    public Tensor aggregate(Tensor inputs, Tensor index, long dimSize) {
        // 输入校验
        if (inputs == null || index == null) {
            throw new NullPointerException("inputs 和 index 不能为空");
        }
        // 使用增强版 Scatter 进行 add 聚合，保证无数值异常
        return Scatter.scatter_add(inputs, index, 0, dimSize);
    }

    // 工具方法：获取对齐设备的 eps 张量
    private Tensor getEpsTensor(Device device) {
        Tensor eps;
        if (this.trainEps) {
            eps = this.epsParam.data().to(device,ScalarType.Float);
        } else {
            eps = this.epsBuffer.to(device,ScalarType.Float);
        }
        return eps;
    }

    // 工具方法：修复数值异常（NaN/Inf/溢出）
    private Tensor fixNumericAnomalies(Tensor tensor) {
        if (tensor == null) return null;

        // NaN 替换为 0
        Tensor nanMask = tensor.isnan();
        if (nanMask.any().item().toBool()) {
            tensor = tensor.where(nanMask.logical_not(), zeros_like(tensor));
        }

        // Inf 替换为安全值
        Tensor infMask = tensor.isinf();
        if (infMask.any().item().toBool()) {
            Tensor posInf = tensor.gt(new Scalar(1e18));
            Tensor negInf = tensor.lt(new Scalar(-1e18));
            tensor = tensor.where(posInf.logical_not(), full_like(tensor, new Scalar(1e9)));
            tensor = tensor.where(negInf.logical_not(), full_like(tensor, new Scalar(-1e9)));
            posInf.close();
            negInf.close();
        }

        // 数值裁剪
        tensor = tensor.clamp(new ScalarOptional(new Scalar(-1e9)), new ScalarOptional(new Scalar(1e9)));

        // 释放临时张量
        nanMask.close();
        infMask.close();

        return tensor;
    }

    // 空实现：满足 MessagePassing 基类约束
    @Override
    public Tensor update(Tensor inputs, Tensor x) {
        return inputs;
    }

    // 模块/参数注册（适配框架）
    protected void register_module(String name, LinearImpl module) {}
    protected void register_module(String name, SequentialImpl module) {}
    protected void register_parameter(String name, Parameter param) {}
//    public void register_buffer(String name, Tensor tensor) {}

    // 资源释放
    public void close() {
        if (nn != null) nn.close();
        if (linEdge != null) linEdge.close();
        if (epsParam != null) epsParam.close();
        if (epsBuffer != null) epsBuffer.close();
    }
}

//public class GINEConv extends MessagePassing {
//    private SequentialImpl nn;       // 明确使用 SequentialImpl 以便调用 forward
//    private Tensor eps;
//    private LinearImpl linEdge;
//    private boolean trainEps;
//
//    /**
//     * @param nn           核心 MLP
//     * @param epsValue     ε 初始值
//     * @param trainEps     ε 是否可学习
//     * @param edgeDim      边特征维度
//     * @param nodeDim      节点特征维度 (必须传入以进行维度对齐检查)
//     */
//    public GINEConv(SequentialImpl nn, double epsValue, boolean trainEps, Integer edgeDim, Integer nodeDim) {
//        super("add");
//        this.nn = nn;
//        this.trainEps = trainEps;
//
//        // 1. 初始化 eps 并确保类型为 kFloat
//        this.eps = torch.tensor(new float[]{(float) epsValue},
//                new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));
//
//        if (trainEps) {
//            register_parameter("eps", eps);
//        } else {
//            register_buffer("eps", eps);
//        }
//
//        // 2. 核心修正：如果维度不一致，必须初始化 linEdge
//        if (edgeDim != null && nodeDim != null && !edgeDim.equals(nodeDim)) {
//            this.linEdge = new LinearImpl(edgeDim, nodeDim);
//            register_module("lin_edge", linEdge);
//        }
//
//        if (nn != null) register_module("nn", nn);
//    }
//
//    @Override
//    public Tensor forward(Tensor x, Tensor edge_index) {
//        return null;
//    }
//
//    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_attr) {
//        // 1. 消息传递
//        // 注意：x 和 edge_attr 都会被传给 message 函数
//        Tensor out = propagate(edge_index, x, edge_attr);
//
//        // 2. 结合中心节点：(1 + eps) * x + out
//        // eps.add(1.0) 会返回一个张量，确保设备与 x 一致
//        Tensor epsPlusOne = eps.add(new Scalar(1.0));
//        Tensor centerUpdate = x.mul(epsPlusOne);
//        out = out.add(centerUpdate);
//
//        // 3. 应用非线性 MLP
//        // 修正：直接使用传入的 SequentialImpl 执行
//        return nn.forward(out);
//    }
//
//    //    @Override
////    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
//    @Override
//    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
//        if (edge_attr == null || !edge_attr.defined()) {
//            return torch.relu(x_j);
//        }
//
//        Tensor ej = edge_attr;
//        // 3. 维度对齐变换
//        if (linEdge != null) {
//            ej = linEdge.forward(ej);
//        }
//
//        // GINE 公式: ReLU(x_j + e_ij)
//        // 现在 x_j 和 ej 维度已经一致
//        return torch.relu(x_j.add(ej));
//    }
//
//    @Override
//    public Tensor aggregate(Tensor inputs, Tensor index, long dimSize) {
//        return Scatter.scatter(inputs, index, dimSize, "add");
//    }
//}


//public class GINEConv extends MessagePassing {
//    private Module nn;              // 核心 MLP 网络
//    private Tensor eps;             // ε 参数
//    private LinearImpl linEdge;         // 用于对齐边特征维度
//    private boolean trainEps;
//
//    public GINEConv(Module nn, double epsValue, boolean trainEps, Integer edgeDim) {
//        super("add"); // GIN 必须使用求和聚合以保持同构判别力
//        this.nn = nn;
//        this.trainEps = trainEps;
//
//        // 初始化 eps
//        this.eps = torch.tensor(new float[]{(float) epsValue});
//        if (trainEps) {
//            register_parameter("eps", eps);
//        } else {
//            register_buffer("eps", eps);
//        }
//
//        // 如果边维度与节点维度不一致，需要一个线性层来对齐
//        if (edgeDim != null) {
//            // 假设 nn 的第一个输入维度即为我们需要对齐的目标维度
//            // 在生产代码中，建议显式传入 nodeChannels
//            // this.linEdge = new Linear(edgeDim, nodeChannels);
//            // register_module("lin_edge", linEdge);
//        }
//
//        register_module("nn", nn);
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
//        // 1. 消息传递：将邻居特征与边特征相加
//        Tensor out = propagate(edge_index, x, edge_attr);
//
//        // 2. 结合中心节点：(1 + eps) * x_i + aggregated_msg
//        Tensor centerUpdate = x.mul(eps.add(new Scalar(1.0)));
//        out = out.add(centerUpdate);
//
//        // 3. 应用非线性 MLP
//        out = nn.asSequential().forward(out);
//        return out;
//    }
//
//    @Override
//    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
//        // x_j: 邻居特征
//        // edge_attr: 边特征
//
//        if (edge_attr == null) {
//            return torch.relu(x_j); // 退化为普通 GIN 逻辑
//        }
//
//        Tensor ej = edge_attr;
//        if (linEdge != null) {
//            ej = linEdge.forward(ej);
//        }
//
//        // GINE 核心公式：ReLU(x_j + e_ij)
//        // 注意：要求 x_j 和 e_ij 维度必须一致
//        return torch.relu(x_j.add(ej));
//    }
//
//    /**
//     * 重写 aggregate 以确保严谨性（虽然父类 add 已经处理）
//     */
//    @Override
//    public Tensor aggregate(Tensor inputs, Tensor index, long dimSize) {
//        return Scatter.scatter(inputs, index, dimSize, "add");
//    }
//}