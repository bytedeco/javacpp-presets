package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;

/**
 * 严格使用 LinearImpl 实现 torch_geometric.nn.conv.FiLMConv
 * 利用特征线性调制（FiLM）处理异构关系图。
 */
public class FiLMConv extends MessagePassing {
    public LinearImpl[] lins;      // 每个关系对应的线性变换 W_r
    public LinearImpl filmLin;     // 用于生成 gamma 和 beta 的网络
    private long outChannels;
    private int numRelations;
    private SequentialImpl act;

    public FiLMConv(long inChannels, long outChannels, int numRelations, AnyModule act) {
        super("mean");
        this.outChannels = outChannels;
        this.numRelations = numRelations;
        this.act = new SequentialImpl();
        if (act != null){
            System.out.println("Adding activation module: " + act.getClass().getSimpleName());
//            this.act.push_back("act",new AnyModule(act));
            this.act.push_back("act",act);
        }
       

        // 1. 为每个关系初始化 LinearImpl
        this.lins = new LinearImpl[numRelations];
        for (int r = 0; r < numRelations; r++) {
            lins[r] = new LinearImpl(inChannels, outChannels);
            register_module("lin_" + r, lins[r]);
        }

        // 2. 初始化 FiLM 参数生成层：从中心节点特征生成 [gamma, beta]
        // 输出维度是 2 * outChannels (一半给 gamma, 一半给 beta)
        this.filmLin = new LinearImpl(inChannels, 2 * outChannels);
        register_module("film_lin", filmLin);
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        return forward(x, edge_index, (Tensor)null);
    }
    /**
     * @param x          节点特征 [N, inChannels]
     * @param edge_index 边索引 [2, E]
     * @param edge_type  关系类型 [E] (0 到 numRelations-1)
     */
    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_type) {
        // FiLM 机制：计算目标节点的调制参数
        // [N, 2 * outChannels]
        Tensor filmParams = filmLin.forward(x);
        Tensor gamma = filmParams.narrow(-1, 0, outChannels);
        Tensor beta = filmParams.narrow(-1, outChannels, outChannels);

        // 执行消息传递，传递调制参数
        return propagate(edge_index, x, edge_type, gamma, beta);
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        // 注意：此处需要访问关系类型和调制参数
        // 假设通过参数透传，此处逻辑在 Java CPP 中常需手动处理 index_select
        return x_j;
    }

    /**
     * 手动覆盖 propagate 以实现关系特定的 FiLM 调制
     */
    public Tensor propagate(Tensor edge_index, Tensor x, Tensor edge_type, Tensor gamma, Tensor beta) {
        long N = x.size(0);
        Tensor sourceIdx = edge_index.select(0, 0);
        Tensor targetIdx = edge_index.select(0, 1);
        Tensor out = torch.zeros(new long[]{N, outChannels}, x.options());

        // 按关系循环（异构处理）
        for (int r = 0; r < numRelations; r++) {
            Tensor mask = edge_type.eq(new Scalar(r));
            if (!mask.any().item_bool()) continue;

            // 1. 提取该关系的子集
            Tensor subSource = sourceIdx.masked_select(mask);
            Tensor subTarget = targetIdx.masked_select(mask);

            // 2. 邻居变换: W_r * x_j
            Tensor x_j = lins[r].forward(x.index_select(0, subSource));

            // 3. 应用 FiLM 调制: gamma_i * (W_r * x_j) + beta_i
            Tensor g_i = gamma.index_select(0, subTarget);
            Tensor b_i = beta.index_select(0, subTarget);
            Tensor msg = x_j.mul(g_i).add(b_i);

            if (act != null && act.asSequential().size()>0 ) msg = act.asSequential().forward(msg);

            // 4. 聚合到结果中
            out.scatter_add_(0, subTarget.view(-1, 1).expand_as(msg), msg);
        }

        return out;
    }
}


/**
 * 修正版 FiLMConv（特征线性调制图卷积）
 * 核心特性：
 * 1. 异构关系图的关系特定线性变换
 * 2. 目标节点驱动的 FiLM 调制（gamma*x + beta）
 * 3. 兼容 PyTorch 梯度计算规则，无原地操作风险
 */
//public class FiLMConv extends MessagePassing {
//    private LinearImpl[] lins;      // 每个关系对应的线性变换 W_r [numRelations]
//    private LinearImpl filmLin;     // FiLM 参数生成层：in → 2*out
//    private long outChannels;
//    private int numRelations;
//    private Module act;             // 激活函数（如 ReLU）
//
//    public FiLMConv(long inChannels, long outChannels, int numRelations, Module act) {
//        super("mean"); // 聚合模式：mean
//        this.outChannels = outChannels;
//        this.numRelations = numRelations;
//        this.act = act;
//
//        // 统一参数配置：Float + CPU，确保设备/类型一致
//        TensorOptions paramOpts = new TensorOptions()
//                .dtype(new ScalarTypeOptional(torch.ScalarType.Float))
//                .device(new DeviceOptional(new Device(torch.kCPU())))
//                .requires_grad(new BoolOptional(true));
//
//        // 1. 初始化每个关系的线性层
//        this.lins = new LinearImpl[numRelations];
//        for (int r = 0; r < numRelations; r++) {
//            lins[r] = new LinearImpl(inChannels, outChannels);
//            // 初始化线性层参数（Xavier 初始化，支持梯度）
//            initLinearParams(lins[r], inChannels, outChannels, paramOpts);
//            register_module("lin_" + r, lins[r]);
//        }
//
//        // 2. 初始化 FiLM 参数生成层（输出 gamma + beta）
//        this.filmLin = new LinearImpl(inChannels, 2 * outChannels);
//        initLinearParams(filmLin, inChannels, 2 * outChannels, paramOpts);
//        register_module("film_lin", filmLin);
//    }
//
//    /**
//     * 核心前向传播逻辑
//     * @param x          节点特征 [N, inChannels]
//     * @param edge_index 边索引 [2, E]
//     * @param edge_type  关系类型 [E] (0 ~ numRelations-1)
//     * @return 输出特征 [N, outChannels]
//     */
//    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_type) {
//        // ========== 输入校验 ==========
//        long[] xShape = x.sizes().vec().get();
//        if (xShape.length != 2) {
//            throw new IllegalArgumentException("节点特征x必须是2D张量，当前维度：" + xShape.length);
//        }
//        long N = xShape[0]; // 节点数
//
//        long[] edgeIndexShape = edge_index.sizes().vec().get();
//        if (edgeIndexShape.length != 2 || edgeIndexShape[0] != 2) {
//            throw new IllegalArgumentException("边索引edge_index必须是[2, E]形状，当前：" + edgeIndexShape);
//        }
//        long E = edgeIndexShape[1]; // 边数
//
//        // 空边场景处理
//        if (E == 0) {
//            return torch.zeros(new long[]{N, outChannels}, x.options());
//        }
//
//        // 关系类型校验
//        if (edge_type == null) {
//            throw new IllegalArgumentException("edge_type 不能为空（异构图必须指定关系类型）");
//        }
//        long[] edgeTypeShape = edge_type.sizes().vec().get();
//        if (edgeTypeShape.length != 1 || edgeTypeShape[0] != E) {
//            throw new IllegalArgumentException("关系类型edge_type必须是[E]形状，当前：" + edgeTypeShape);
//        }
//
//        // ========== 1. 计算 FiLM 调制参数（gamma/beta） ==========
//        Tensor filmParams = filmLin.forward(x); // [N, 2*outChannels]
//        Tensor gamma = filmParams.narrow(-1, 0, outChannels); // [N, outChannels]
//        Tensor beta = filmParams.narrow(-1, outChannels, outChannels); // [N, outChannels]
//
//        // ========== 2. 调用自定义 propagate 完成消息传递 + FiLM 调制 ==========
//        Tensor out = propagate(edge_index, x, edge_type, gamma, beta);
//
//        // ========== 3. 应用激活函数（如果有） ==========
//        if (act != null) {
//            out = act.asSequential().forward(out);
//        }
//
//        // ========== 资源释放 ==========
//        filmParams.close();
//        gamma.close();
//        beta.close();
//
//        return out;
//    }
//
//    /**
//     * 覆写父类 forward 方法（无 edge_type 时抛出异常）
//     */
//    @Override
//    public Tensor forward(Tensor x, Tensor edge_index) {
//        throw new UnsupportedOperationException("FiLMConv 必须指定 edge_type（异构图）");
//    }
//
//    /**
//     * 自定义 propagate 方法：实现关系特定的 FiLM 调制 + 聚合
//     */
//    public Tensor propagate(Tensor edge_index, Tensor x, Tensor edge_type, Tensor gamma, Tensor beta) {
//        long N = x.size(0);
//        Tensor sourceIdx = edge_index.select(0, 0); // [E]
//        Tensor targetIdx = edge_index.select(0, 1); // [E]
//
//        // 初始化输出张量（非原地操作，避免梯度报错）
//        Tensor out = torch.zeros(new long[]{N, outChannels}, x.options());
//        // 初始化计数张量（用于 mean 聚合）
//        Tensor count = torch.zeros(new long[]{N}, x.options().dtype(new ScalarTypeOptional(torch.ScalarType.Float)));
//
//        // 按关系类型循环处理
//        for (int r = 0; r < numRelations; r++) {
//            // 1. 筛选当前关系的边掩码 [E]
//            Tensor mask = edge_type.eq(new Scalar(r)); // bool 张量
//            if (!mask.any().item_bool()) {
//                mask.close();
//                continue; // 无当前关系的边，跳过
//            }
//
//            // 2. 提取当前关系的源/目标节点索引
//            Tensor subSource = sourceIdx.masked_select(mask); // [E_r]
//            Tensor subTarget = targetIdx.masked_select(mask); // [E_r]
//            long E_r = subSource.size(0);
//            if (E_r == 0) {
//                mask.close();
//                subSource.close();
//                subTarget.close();
//                continue;
//            }
//
//            // 3. 关系特定的邻居特征变换：W_r * x_j
//            Tensor x_j = x.index_select(0, subSource); // [E_r, in]
//            Tensor x_j_trans = lins[r].forward(x_j); // [E_r, out]
//
//            // 4. FiLM 调制：gamma_i * x_j_trans + beta_i
//            Tensor g_i = gamma.index_select(0, subTarget); // [E_r, out]
//            Tensor b_i = beta.index_select(0, subTarget); // [E_r, out]
//            Tensor msg = x_j_trans.mul(g_i).add(b_i); // [E_r, out]
//
//            // 5. 聚合到目标节点（非原地 scatter_add）
//            out = out.scatter_add(0, subTarget.view(-1, 1).expand_as(msg), msg);
//            // 更新计数（用于 mean 聚合）
//            count = count.scatter_add(0, subTarget, torch.ones(new long[]{E_r}, x.options()));
//
//            // 6. 资源释放
//            mask.close();
//            subSource.close();
//            subTarget.close();
//            x_j.close();
//            x_j_trans.close();
//            g_i.close();
//            b_i.close();
//            msg.close();
//        }
//
//        // Mean 聚合：除以入边数（避免除0）
//        count = count.clamp(new Scalar(1.0f)); // 小于1的置为1
//        out = out.div(count.unsqueeze(-1).expand_as(out));
//
//        // 资源释放
//        sourceIdx.close();
//        targetIdx.close();
//        count.close();
//
//        return out;
//    }
//
//    /**
//     * 初始化线性层参数（Xavier 初始化，支持梯度）
//     */
//    private void initLinearParams(LinearImpl linear, long inDim, long outDim, TensorOptions paramOpts) {
//        // 权重初始化（非原地操作）
//        Tensor weight = torch.empty(new long[]{outDim, inDim}, paramOpts);
//        weight = torch.xavier_uniform(weight);
//        linear.weight(new Parameter(weight));
//
//        // 偏置初始化（如果有）
//        if (linear.bias() != null) {
//            Tensor bias = torch.zeros(new long[]{outDim}, paramOpts);
//            linear.bias(new Parameter(bias));
//        }
//    }
//
//    /**
//     * 覆写父类 message 方法（适配接口）
//     */
//    @Override
//    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
//        return x_j; // 核心逻辑在 propagate 中实现
//    }
//
//    /**
//     * 资源释放（避免内存泄漏）
//     */
//    @Override
//    protected void finalize() throws Throwable {
//        try {
//            if (lins != null) {
//                for (LinearImpl lin : lins) {
//                    if (lin != null) lin.close();
//                }
//            }
//            if (filmLin != null) filmLin.close();
//            if (act != null) act.close();
//        } finally {
//            super.finalize();
//        }
//    }
//
//    // ========== Getter 方法（测试用） ==========
//    public long getOutChannels() {
//        return outChannels;
//    }
//
//    public int getNumRelations() {
//        return numRelations;
//    }
//}