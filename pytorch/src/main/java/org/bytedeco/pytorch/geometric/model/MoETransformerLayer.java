package org.bytedeco.pytorch.geometric.model;
import org.bytedeco.pytorch.optim.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.conv.TransformerConv;
//import org.gnn.framework.layers.org.bytedeco.pytorch.geometric.nn.conv.TransformerConv; // 使用我们之前实现的类
import java.util.ArrayList;
import java.util.List;

/**
 * MoE (Mixture of Experts) Layer based on org.bytedeco.pytorch.geometric.nn.conv.TransformerConv
 * 逻辑: Output = Sum( Softmax(Gate(x)) * Expert_i(x, edge_index) )
 */
public class MoETransformerLayer extends Module {

    private List<TransformerConv> experts;
    private LinearImpl gate;
    private int numExperts;

    public MoETransformerLayer(long inChannels, long outChannels, long heads, int numExperts) {
        super();
        this.numExperts = numExperts;
        this.experts = new ArrayList<>();

        // 1. 初始化门控网络: [In -> NumExperts]
        this.gate = new LinearImpl(inChannels, numExperts);
        register_module("gate", gate);

        // 2. 初始化专家网络
        for (int i = 0; i < numExperts; i++) {
            TransformerConv expert = new TransformerConv(inChannels, outChannels, heads);
            this.experts.add(expert);
            // 必须手动注册每个 expert 以便 Optimizer 能够更新参数
            register_module("expert_" + i, expert);
        }
    }

    public Tensor forward(Tensor x, Tensor edge_index) {
        long numNodes = x.size(0);

        // 1. 计算门控权重 (Gating Weights)
        // [N, In] -> [N, NumExperts]
        Tensor gateLogits = gate.forward(x);
        // Softmax over experts dimension (dim=1) -> [N, NumExperts]
        Tensor gateWeights = torch.softmax(gateLogits, 1);

        // 2. 专家计算并加权融合
        Tensor output = null;

        for (int i = 0; i < numExperts; i++) {
            // Expert forward: [N, Heads * Out]
            Tensor expertOut = experts.get(i).forward(x, edge_index);

            // 获取当前专家对每个节点的权重
            // gateWeights[:, i] -> [N] -> [N, 1] 以便广播
            Tensor weight = gateWeights.select(1, i).unsqueeze(1);

            // 加权: ExpertOut * Weight
            Tensor weightedExpert = expertOut.mul(weight);

            if (output == null) {
                output = weightedExpert;
            } else {
                output = output.add(weightedExpert);
            }
        }

        return output;
    }
}