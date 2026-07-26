package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.utils.Scatter;
import static org.bytedeco.pytorch.global.torch.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.utils.Scatter;
import static org.bytedeco.pytorch.global.torch.*;

/**
 * 修复版 EdgeConv（解决维度不匹配问题）
 * 核心：确保特征拼接后维度为 2*inChannels，匹配 Linear 层输入
 */
public class EdgeConv extends MessagePassing {
    private SequentialImpl nn;
    private long inChannels;
    private long outChannels;

    public EdgeConv(long inChannels, long outChannels) {
        super("max");
        if (inChannels <= 0 || outChannels <= 0) {
            throw new IllegalArgumentException("通道数必须大于0");
        }
        this.inChannels = inChannels;
        this.outChannels = outChannels;

        // 关键：Linear 输入维度必须是 2*inChannels
        this.nn = new SequentialImpl();
        nn.push_back(new LinearImpl(2 * inChannels, outChannels));
        nn.push_back(new ReLUImpl());
        nn.push_back(new LinearImpl(outChannels, outChannels));

        register_module("nn", nn);
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        if (x == null || edge_index == null) {
            throw new NullPointerException("输入张量不能为空");
        }
        if (x.dim() != 2 || edge_index.dim() != 2 || edge_index.size(0) != 2) {
            throw new IllegalArgumentException("输入维度错误：x应为2维，edge_index应为2xN维");
        }
        if (x.size(1) != inChannels) {
            throw new IllegalArgumentException("输入通道数不匹配：预期" + inChannels + "，实际" + x.size(1));
        }

        return propagate(edge_index, x);
    }

    public Tensor propagate(Tensor edge_index, Tensor x) {
        long numNodes = x.size(0);
        Tensor sourceIdx = edge_index.select(0, 0);
        Tensor targetIdx = edge_index.select(0, 1);

        // 提取 x_i (中心节点) 和 x_j (邻居节点) [E, C]
        Tensor x_i = x.index_select(0, targetIdx);
        Tensor x_j = x.index_select(0, sourceIdx);

        // 打印调试维度（确认拼接前维度）
        System.out.println("🔍 拼接前维度 - x_i: " + x_i.size(0) + "x" + x_i.size(1) + ", x_j: " + x_j.size(0) + "x" + x_j.size(1));

        // 生成消息
        Tensor msg = message(x_j, x_i, edge_index, null, numNodes);

        // 聚合（max）
        Tensor out = aggregate(msg, targetIdx, numNodes);
        out = update(out, x);

        // 释放资源
        sourceIdx.close();
        targetIdx.close();
        x_i.close();
        x_j.close();
        msg.close();

        return out;
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        // 1. 计算差异 [E, C]
        Tensor diff = x_j.sub(x_i);

        // 2. 强制确保维度正确（核心修复）
        if (x_i.size(1) != inChannels || diff.size(1) != inChannels) {
            throw new RuntimeException("维度错误：x_i/diff 通道数应为" + inChannels + "，实际为" + x_i.size(1));
        }

        // 3. 拼接特征 [x_i, diff] → [E, 2*C]
        // 关键修复：明确指定 TensorVector 长度 + 验证拼接后维度
        TensorVector catVec = new TensorVector(2); // 预指定长度
        catVec.put(0, x_i);
        catVec.put(1, diff);

        // 拼接维度必须用 long 类型，且明确指定 dim=1
        Tensor catFeat = torch.cat(catVec, 1);

        // 打印拼接后维度（调试）
        System.out.println("🔍 拼接后维度 - catFeat: " + catFeat.size(0) + "x" + catFeat.size(1));

        // 验证拼接维度（核心：必须是 2*inChannels）
        if (catFeat.size(1) != 2 * inChannels) {
            throw new RuntimeException("拼接维度错误：预期" + 2*inChannels + "，实际" + catFeat.size(1));
        }

        // 4. MLP 前向传播
        Tensor msg = nn.forward(catFeat);

        // 释放临时张量
        diff.close();
        catVec.close();
        catFeat.close();

        return msg;
    }

    @Override
    public Tensor update(Tensor inputs, Tensor x) {
        return inputs;
    }

    public Tensor aggregate(Tensor msg, Tensor index, long dimSize) {
        return Scatter.scatter(msg, index, dimSize, "max");
    }

    public void close() {
        if (nn != null) nn.close();
    }
}

//public class EdgeConv extends MessagePassing {
//    private SequentialImpl nn;
//    private long inChannels;
//    private long outChannels;
//
//    // 构造函数：初始化 MLP + 输入校验
//    public EdgeConv(long inChannels, long outChannels) {
//        super("max"); // 使用 max 聚合
//        // 输入校验
//        if (inChannels <= 0 || outChannels <= 0) {
//            throw new IllegalArgumentException("通道数必须大于0");
//        }
//        this.inChannels = inChannels;
//        this.outChannels = outChannels;
//
//        // 初始化 MLP (输入: 2*in → 隐藏层 → 输出: out)
//        this.nn = new SequentialImpl();
//        var options = new LinearOptions(2 * inChannels, outChannels);
//        options.bias().put(false); // EdgeConv 通常不使用偏置
//        nn.push_back(new LinearImpl(options));
//        nn.push_back(new ReLUImpl());
//        var options1 = new LinearOptions(outChannels, outChannels);
//        options1.bias().put(false);
//        nn.push_back(new LinearImpl(options1));
//
//        // 注册模块（保证梯度传播）
//        register_module("nn", nn);
//    }
//
//    // 核心前向传播（统一入口）
//    @Override
//    public Tensor forward(Tensor x, Tensor edge_index) {
//        // 输入校验
//        if (x == null || edge_index == null) {
//            throw new NullPointerException("输入张量不能为空");
//        }
//        if (x.dim() != 2 || edge_index.dim() != 2 || edge_index.size(0) != 2) {
//            throw new IllegalArgumentException("输入维度错误：x应为2维，edge_index应为2xN维");
//        }
//        if (x.size(1) != inChannels) {
//            throw new IllegalArgumentException("输入通道数不匹配：预期" + inChannels + "，实际" + x.size(1));
//        }
//
//        // 核心传播逻辑（实现 MessagePassing 核心流程）
//        return propagate(edge_index, x);
//    }
//
//    // 实现基类核心 propagate 方法（EdgeConv 核心传播逻辑）
//    public Tensor propagate(Tensor edge_index, Tensor x) {
//        long numNodes = x.size(0);
//        Tensor sourceIdx = edge_index.select(0, 0); // 源节点（邻居）[E]
//        Tensor targetIdx = edge_index.select(0, 1); // 目标节点（中心）[E]
//
//        // 1. 提取 x_i (中心节点) 和 x_j (邻居节点) [E, C]
//        Tensor x_i = x.index_select(0, targetIdx);
//        Tensor x_j = x.index_select(0, sourceIdx);
//
//        // 2. 生成消息（调用 message 方法）
//        Tensor msg = message(x_j, x_i, edge_index, null, numNodes);
//
//        // 3. 聚合消息（max 聚合）
//        Tensor out = aggregate(msg, targetIdx, numNodes);
//
//        // 4. 更新输出（EdgeConv 无额外更新逻辑）
//        out = update(out, x);
//
//        // 5. 释放临时张量（避免内存泄漏）
//        sourceIdx.close();
//        targetIdx.close();
//        x_i.close();
//        x_j.close();
//        msg.close();
//
//        return out;
//    }
//
//    // 实现基类唯一的 message 方法（消除重载冲突）
//    @Override
//    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
//        // 1. 计算邻居与中心节点的差异: x_j - x_i [E, C]
//        Tensor diff = x_j.sub(x_i);
//
//        // 2. 拼接特征: [x_i, x_j - x_i] → [E, 2*C]
//        // 修复 TensorVector 初始化 + cat 维度（必须用 long 类型）
//        TensorVector catVec = new TensorVector();
//        catVec.put(x_i);
//        catVec.put(diff);
//        Tensor catFeat = torch.cat(catVec, 1); // dim=1 拼接
//
//        // 3. 通过 MLP 处理拼接特征
//        Tensor msg = nn.forward(catFeat);
//
//        // 4. 释放临时张量
//        diff.close();
//        catVec.close();
//        catFeat.close();
//
//        return msg;
//    }
//
//    // 实现基类 update 方法（EdgeConv 无额外更新）
//    @Override
//    public Tensor update(Tensor inputs, Tensor x) {
//        return inputs;
//    }
//
//    @Override
//    public Tensor aggregate(Tensor msg, Tensor index, long dimSize) {
//        return Scatter.scatter(msg, index, dimSize, "max");
//    }
//
//    // 释放资源
//    public void close() {
//        if (nn != null) nn.close();
//    }
//}

//public class EdgeConv extends MessagePassing {
//    private SequentialImpl nn;
//
//    public EdgeConv(long inChannels, long outChannels) {
//        super("max");
//        this.nn = new SequentialImpl();
//        // 输入维度 x2
//        nn.push_back(new LinearImpl(2 * inChannels, outChannels));
//        nn.push_back(new ReLUImpl());
//        nn.push_back(new LinearImpl(outChannels, outChannels));
//        register_module("nn", nn);
//    }
//
//    @Override
//    public Tensor forward(Tensor x, Tensor edge_index) {
//        return propagate(edge_index, x);
//    }
//
//    /**
//     * 必须匹配基类签名：(x_j, x_i, edge_index, edge_attr)
//     */
//    @Override
//    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
//        // x_i: 中心节点特征 [E, C]
//        // x_j: 邻居节点特征 [E, C]
//
//        // 1. 计算差异: x_j - x_i
//        Tensor diff = x_j.sub(x_i);
//
//        // 2. 拼接: [x_i, x_j - x_i] -> [E, 2 * C]
//        // 注意：JavaCPP 的 cat 需要传 Tensor 数组
//        Tensor catFeat = torch.cat(new TensorVector(x_i, diff), -1);
//
//        // 3. 通过 MLP (nn)
//        // 建议在这里进行类型转换或确保 nn 已定义好 forward
//        return nn.forward(catFeat);
//    }
//    @Override
//    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index) {
//        // 1. 计算差异: x_j - x_i
//        Tensor diff = x_j.sub(x_i);
//
//        // 2. 拼接: [x_i, x_j - x_i]
//        // ！！！修正点：直接在构造函数中传入张量 ！！！
//        TensorVector vec = new TensorVector(x_i, diff);
//
//        // Cat dimension 1
//        Tensor catFeat = torch.cat(vec, 1);
//
//        // 3. MLP
//        return nn.forward(catFeat);
//    }
//
//
//}


//public class org.bytedeco.pytorch.geometric.nn.conv.EdgeConv extends org.bytedeco.pytorch.geometric.nn.conv.MessagePassing {
//    private SequentialImpl nn; // MLP
//
//    public org.bytedeco.pytorch.geometric.nn.conv.EdgeConv(long inChannels, long outChannels) {
//        super("max");
//        this.nn = new SequentialImpl();
//        // 输入维度 x2
//        nn.push_back(new LinearImpl(2 * inChannels, outChannels));
//        nn.push_back(new ReLUImpl());
//        nn.push_back(new LinearImpl(outChannels, outChannels));
//        register_module("nn", nn);
//    }
//
//    public Tensor forward(Tensor x, Tensor edge_index) {
//        return propagate(edge_index, x);
//    }
//
//    @Override
//    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index) {
//        // Logic: cat([x_i, x_j - x_i], dim=1)
//        Tensor diff = x_j.sub(x_i);
//
//        // 重点：torch.cat 入参必须包裹
//        TensorVector vec = new TensorVector();
//        vec.put(x_i);
//        vec.put(diff);
//
//        Tensor catFeat = torch.cat(vec, 1); // dim 1
//        return nn.forward(catFeat);
//    }
//}