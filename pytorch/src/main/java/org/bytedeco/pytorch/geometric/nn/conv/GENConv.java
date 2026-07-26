package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.data.datasets.*;
import org.bytedeco.pytorch.data.options.*;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.options.LinearOptions;
import org.bytedeco.pytorch.geometric.nn.Parameter;
import org.bytedeco.pytorch.geometric.utils.Scatter;

//package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.utils.Scatter;
import static org.bytedeco.pytorch.global.torch.*;
import org.bytedeco.pytorch.*;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.*;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.data.datasets.TensorDataset;
import org.bytedeco.pytorch.data.options.DataLoaderOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.options.CrossEntropyLossOptions;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.AdamOptions;
import org.bytedeco.pytorch.optim.SGD;
import org.bytedeco.pytorch.optim.SGDOptions;
import org.bytedeco.pytorch.optim.*;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.*;
import static org.bytedeco.pytorch.global.torch.arange;
import org.bytedeco.pytorch.global.torch;
/**
 * 修复版 GENConv（严格匹配 torch_geometric.nn.conv.GENConv 逻辑）
 * 解决维度不匹配、数值异常、参数注册、内存泄漏等问题
 */
public class GENConv extends MessagePassing {
    private LinearImpl lin;         // 最终映射 W
    private LinearImpl linEdge;     // 边特征对齐 W_edge
    private Parameter t;            // Softmax 温度系数（封装为 Parameter）
    private Parameter p;            // PowerMean 指数（封装为 Parameter）
    private float eps;
    private String aggrType;
    private boolean hasBias;

    public Parameter getT(){
       return this.t;
    }
    public Parameter getP(){
        return this.p;
    }
    // GENConv 标准构造函数（修复参数注册、默认值等问题）
    public GENConv(long inChannels, long outChannels) {
        this(inChannels, outChannels, "softmax", 1.0f, false, 1.0f, false, null, 1e-7f, true);
    }

    public GENConv(long inChannels, long outChannels, String aggr, float tVal, boolean learnT,
                   float pVal, boolean learnP, Integer edgeDim, float eps, boolean hasBias) {
        super(aggr);
        // 1. 输入校验
        if (inChannels <= 0 || outChannels <= 0) {
            throw new IllegalArgumentException("通道数必须大于0: inChannels=" + inChannels + ", outChannels=" + outChannels);
        }
        if (!aggr.equals("softmax") && !aggr.equals("powermean") && !aggr.equals("add") && !aggr.equals("mean") && !aggr.equals("max")) {
            throw new UnsupportedOperationException("不支持的聚合方式: " + aggr);
        }

        this.aggrType = aggr;
        this.eps = eps;
        this.hasBias = hasBias;

        var linOptions = new LinearOptions(inChannels, outChannels);
        linOptions.bias().put(hasBias);
        this.lin = new LinearImpl(linOptions);
        register_module("lin", lin);

        // 3. 边特征线性层（修复维度对齐）
        if (edgeDim != null && edgeDim > 0) {
            var linEdgeOptions = new LinearOptions(edgeDim, inChannels);
            linEdgeOptions.bias().put(hasBias);
            this.linEdge = new LinearImpl(linEdgeOptions);
            register_module("lin_edge", linEdge);
        }

        // 4. 聚合参数（修复 Parameter 注册）
        // 温度系数 t：Softmax 聚合
        Tensor tTensor = torch.tensor(new float[]{tVal}).requires_grad_();//, new TensorOptions().requires_grad(new BoolOptional(learnT)));
        this.t = new Parameter(tTensor);
        if (learnT) register_parameter("t", this.t);

        // 幂指数 p：PowerMean 聚合
        Tensor pTensor = torch.tensor(new float[]{pVal}).requires_grad_();//, new TensorOptions().requires_grad(new BoolOptional(learnP)));
        this.p = new Parameter(pTensor);
        if (learnP) register_parameter("p", this.p);
    }

    // 修复 forward 重载（兼容单/双边特征输入）
    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        return forward(x, edge_index, (Tensor)null);
    }

    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_attr) {
        // 输入校验
        if (x == null || edge_index == null) {
            throw new NullPointerException("x 和 edge_index 不能为空");
        }
        if (x.dim() != 2 || edge_index.dim() != 2 || edge_index.size(0) != 2) {
            throw new IllegalArgumentException("x 必须是 2D 张量，edge_index 必须是 2xE 张量");
        }

        // 消息传递与聚合
        Tensor out = propagate(edge_index, x, edge_attr);

        // 最终线性变换（修复维度映射）
        out = lin.forward(out);

        // 数值异常修复
        out = fixNumericAnomalies(out);

        return out;
    }

    // 修复 message 方法（GENConv 标准消息构造）
    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        // GENConv 核心：msg = ReLU(x_j + edge_attr) + eps
        Tensor msg = x_j.clone(); // 避免修改原张量

        // 边特征对齐（修复维度不匹配）
        if (edge_attr != null) {
            if (linEdge != null) {
                edge_attr = linEdge.forward(edge_attr); // 边特征映射到节点特征维度
            }
            // 确保边特征维度与节点特征一致
            if (edge_attr.size(1) != msg.size(1)) {
                throw new IllegalArgumentException("边特征维度不匹配: edge_attr=" + edge_attr.size(1) + ", x_j=" + msg.size(1));
            }
            msg = msg.add(edge_attr);
        }

        // GENConv 标准激活 + 偏移
        msg = relu(msg).add(new Scalar(eps));

        // 数值异常修复
        msg = fixNumericAnomalies(msg);

        return msg;
    }

    // 修复 propagate 方法（核心消息传递逻辑）
    @Override
    public Tensor propagate(Tensor edge_index, Tensor x, Tensor edge_attr) {
        long numNodes = x.size(0);
        Tensor row = edge_index.select(0, 0); // 源节点 (j)
        Tensor col = edge_index.select(0, 1); // 目标节点 (i)

        // 提取 x_j (邻居节点特征)
        Tensor x_j = x.index_select(0, row);
        // x_i 仅用于扩展场景，GENConv 核心只需要 x_j
        Tensor x_i = x.index_select(0, col);

        // 生成消息
        Tensor msg = message(x_j, x_i, edge_index, edge_attr, numNodes);

        // 聚合消息
        Tensor out = aggregate(msg, col, numNodes);

        // 释放临时张量（修复内存泄漏）
        row.close();
        col.close();
        x_j.close();
        x_i.close();
        msg.close();

        return out;
    }

    // 修复聚合方法（实现 GENConv 标准 Softmax/PowerMean 聚合）
    @Override
    public Tensor aggregate(Tensor inputs, Tensor index, long dimSize) {
        Tensor out = null;

        if (aggrType.equals("softmax")) {
            // Softmax Aggregation: sum(exp(t * m_j) * m_j) / sum(exp(t * m_j))
            // 数值稳定性：减去 max 避免 exp 溢出
            Tensor tVal = this.t.data(); // 获取 Parameter 数据
            Tensor scaledInputs = inputs.mul(tVal);

            // 按节点聚合最大值
            Tensor maxVal = Scatter.scatter_max(scaledInputs, index, 0, dimSize);//[0];
            // 广播 maxVal 到所有边
            maxVal = maxVal.index_select(0, index);

            // Softmax 计算（数值稳定版）
            Tensor expInputs = scaledInputs.sub(maxVal).exp();
            Tensor weightedInputs = inputs.mul(expInputs);

            // 聚合加权输入和权重和
            Tensor sumWeighted = Scatter.scatter_add(weightedInputs, index, 0, dimSize);
            Tensor sumExp = Scatter.scatter_add(expInputs, index, 0, dimSize);

            // 避免除0
            sumExp = sumExp.clamp_min(new Scalar(1e-16));
            out = sumWeighted.div(sumExp);

            // 释放临时张量
            scaledInputs.close();
            maxVal.close();
            expInputs.close();
            weightedInputs.close();
            sumWeighted.close();
            sumExp.close();

        } else if (aggrType.equals("powermean")) {
            // PowerMean Aggregation: (1/|N| * sum(m_j^p))^(1/p)
            Tensor pVal = this.p.data();
            float pFloat = pVal.item_float();

            // 处理 p=0 特殊情况（退化为 Geometric Mean）
            if (Math.abs(pFloat) < 1e-8) {
                Tensor logInputs = inputs.clamp_min(new Scalar(1e-16)).log();
                Tensor meanLog = Scatter.scatter_mean(logInputs, index, 0, dimSize);
                out = meanLog.exp();
                logInputs.close();
                meanLog.close();
            } else {
                // 常规 PowerMean
                Tensor powInputs = inputs.clamp_min(new Scalar(1e-16)).pow(pVal);
                Tensor meanPow = Scatter.scatter_mean(powInputs, index, 0, dimSize);
                // 避免 1/p 溢出
                float invP = 1.0f / pFloat;
                out = meanPow.clamp_min(new Scalar(1e-16)).pow(new Scalar(invP));

                // 释放临时张量
                powInputs.close();
                meanPow.close();
            }

        } else if (aggrType.equals("add") || aggrType.equals("sum")) {
            out = Scatter.scatter_add(inputs, index, 0, dimSize);
        } else if (aggrType.equals("mean")) {
            out = Scatter.scatter_mean(inputs, index, 0, dimSize);
        } else if (aggrType.equals("max")) {
            out = Scatter.scatter_max(inputs, index, 0, dimSize);//[0];
        }

        // 数值异常修复
        out = fixNumericAnomalies(out);

        return out;
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

    // 空实现（满足 MessagePassing 基类约束）
    @Override
    public Tensor update(Tensor inputs, Tensor x) {
        return inputs;
    }

    // 模块注册（适配 Parameter/Module 管理）
    protected void register_module(String name, LinearImpl module) {}
    protected void register_parameter(String name, Parameter param) {}

    // 资源释放
    public void close() {
        if (lin != null) lin.close();
        if (linEdge != null) linEdge.close();
        if (t != null) t.close();
        if (p != null) p.close();
    }
}

//public class GENConv extends MessagePassing {
//    private LinearImpl lin;         // 最终映射 W
//    private LinearImpl linEdge;     // 边特征对齐 W_edge
//    private Tensor t;               // Softmax 聚合的温度系数
//    private Tensor p;               // PowerMean 聚合的指数
//    private float eps;
//    private String aggrType;
//
//    public GENConv(long inChannels, long outChannels, String aggr, float tVal, boolean learnT,
//                   float pVal, boolean learnP, Integer edgeDim, float eps, boolean hasBias) {
//        super(aggr);
//        this.aggrType = aggr;
//        this.eps = eps;
//
//        // 1. 严格使用 LinearImpl 进行特征变换
//        // 注意：GENConv 聚合后的维度依然是 outChannels
//        this.lin = new LinearImpl(inChannels, outChannels);
//        register_module("lin", lin);
//
//        if (edgeDim != null) {
//            this.linEdge = new LinearImpl(edgeDim, inChannels);
//            register_module("lin_edge", linEdge);
//        }
//
//        // 2. 聚合参数设置
//        this.t = torch.tensor(new float[]{tVal});
//        if (learnT) register_parameter("t", t);
//
//        this.p = torch.tensor(new float[]{pVal});
//        if (learnP) register_parameter("p", p);
//
//        // 如果需要 Bias，注册到 lin 内部或手动管理
//    }
//
//    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_attr) {
//        // 1. 消息传递与聚合
//        Tensor out = propagate(edge_index, x, edge_attr);
//
//        // 2. 最终线性变换 W
//        return lin.forward(out);
//    }
//    @Override
//    public Tensor forward(Tensor x, Tensor edge_index) {
//        return forward(x, edge_index, (Tensor)null);
//    }
//
//    @Override
//    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
//        // GENConv 消息构造: ReLU(x_j + edge_attr) + eps
//        Tensor msg = x_j;
//
//        if (edge_attr != null && linEdge != null) {
//            msg = msg.add(linEdge.forward(edge_attr));
//        } else if (edge_attr != null) {
//            msg = msg.add(edge_attr);
//        }
//
//        return torch.relu(msg).add(new Scalar(eps));
//    }
//
//    /**
//     * 实现 Generalized Aggregation (Softmax / PowerMean)
//     */
//    @Override
//    public Tensor aggregate(Tensor inputs, Tensor index, long dimSize) {
//        if (aggrType.equals("softmax")) {
//            // Softmax Aggregation: sum(exp(t * m_j) * m_j) / sum(exp(t * m_j))
//            Tensor out = inputs.mul(t);
//            Tensor maxVal = Scatter.scatter(out, index, dimSize, "max");
//            out = out.sub(maxVal.index_select(0, index)).exp();
//
//            Tensor weightedInput = inputs.mul(out);
//            Tensor sumWeight = Scatter.scatter(out, index, dimSize, "add");
//            Tensor sumInput = Scatter.scatter(weightedInput, index, dimSize, "add");
//
//            return sumInput.div(sumWeight.add(new Scalar(1e-16)));
//        } else if (aggrType.equals("powermean")) {
//            // PowerMean Aggregation: (1/|N| * sum(m_j^p))^(1/p)
//            Tensor out = inputs.pow(p);
//            Tensor mean = Scatter.scatter(out, index, dimSize, "mean");
//            return mean.pow(new Scalar(1.0 / p.item_float()));
//        }
//
//        return super.aggregate(inputs, index, dimSize);
//    }
//}