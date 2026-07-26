package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;

import java.util.*;

import static org.bytedeco.pytorch.geometric.utils.AggrUtils.scatter_softmax;

/**
 * 严格使用 LinearImpl 实现 torch_geometric.nn.conv.HGTConv
 * 异构图 Transformer 算子，支持节点类型和边类型的深度定制投影。
 */

public class HGTConv extends Module {
    private Map<String, LinearImpl> kLin, qLin, vLin, aLin;
    private Map<String, Tensor> relationAtt, relationMsg, relationPri;

    private long outChannels;
    private int heads;
    private double sqrtDim;
    private long dK; // 每个头的维度

    public HGTConv(Map<String, Integer> inChannelsDict, int outChannels,
                   List<String> nodeTypes, List<String[]> edgeTypes, int heads) {
        super();
        this.outChannels = outChannels;
        this.heads = heads;
        this.dK = outChannels / heads;
        this.sqrtDim = Math.sqrt(dK);

        this.kLin = new HashMap<>();
        this.qLin = new HashMap<>();
        this.vLin = new HashMap<>();
        this.aLin = new HashMap<>();

        for (String nodeType : nodeTypes) {
            int inC = inChannelsDict.get(nodeType);
            kLin.put(nodeType, register_module("k_" + nodeType, new LinearImpl(inC, outChannels)));
            qLin.put(nodeType, register_module("q_" + nodeType, new LinearImpl(inC, outChannels)));
            vLin.put(nodeType, register_module("v_" + nodeType, new LinearImpl(inC, outChannels)));
            aLin.put(nodeType, register_module("a_" + nodeType, new LinearImpl(outChannels, outChannels)));
        }

        this.relationAtt = new HashMap<>();
        this.relationMsg = new HashMap<>();
        this.relationPri = new HashMap<>();

        for (String[] edgeType : edgeTypes) {
            String edgeName = String.join("_", edgeType);
            Tensor att = torch.randn(new long[]{heads, dK, dK}).retainReference();
            Tensor msg = torch.randn(new long[]{heads, dK, dK}).retainReference();
            Tensor pri = torch.ones(new long[]{heads}).retainReference();

            relationAtt.put(edgeName, att);
            relationMsg.put(edgeName, msg);
            relationPri.put(edgeName, pri);

//            relationAtt.put(edgeName, register_parameter("r_att_" + edgeName, att));
//            relationMsg.put(edgeName, register_parameter("r_msg_" + edgeName, msg));
//            relationPri.put(edgeName, register_parameter("r_pri_" + edgeName, pri));
//            relationAtt.put(edgeName, register_parameter("r_att_" + edgeName, torch.randn(new long[]{heads, dK, dK})));
//            relationMsg.put(edgeName, register_parameter("r_msg_" + edgeName, torch.randn(new long[]{heads, dK, dK})));
//            relationPri.put(edgeName, register_parameter("r_pri_" + edgeName, torch.ones(new long[]{heads})));
        }
    }

    public Map<String, Tensor> forward(Map<String, Tensor> xDict, Map<String[], Tensor> edgeIndexDict) {
        // 1. 使用 TensorVector 来暂存聚合后的消息，避免 ArrayList 转换时的指针丢失
        Map<String, Tensor> msgSumDict = new HashMap<>();


        for (Map.Entry<String[], Tensor> entry : edgeIndexDict.entrySet()) {
//            try (PointerScope innerScope = new PointerScope()) {
            String[] eType = entry.getKey();
            Tensor edgeIndex = entry.getValue();
            String edgeName = String.join("_", eType);

            String srcType = eType[0];
            String dstType = eType[2];

            // 提取投影矩阵 (注意检查 null)
            LinearImpl kL = kLin.get(srcType);
            LinearImpl vL = vLin.get(srcType);
            LinearImpl qL = qLin.get(dstType);

            // 投影并重塑 [N, H, D/H]
            Tensor k = kL.forward(xDict.get(srcType)).view(-1, heads, dK);
            Tensor v = vL.forward(xDict.get(srcType)).view(-1, heads, dK);
            Tensor q = qL.forward(xDict.get(dstType)).view(-1, heads, dK);

            // 关系投影
            Tensor rAtt = relationAtt.get(edgeName);
            Tensor rMsg = relationMsg.get(edgeName);
            System.out.println("k before einsum:0000 " + edgeName + "  " + Arrays.toString(k.sizes().vec().get())); //k before einsum:0000 [100, 8, 16]
            System.out.println("v before einsum:1111 " + edgeName + "  " + Arrays.toString(v.sizes().vec().get())); //v before einsum:1111 [100, 8, 16]
            System.out.println("rAtt before einsum:2222 " + Arrays.toString(rAtt.sizes().vec().get())); // [8, 16, 16]
            System.out.println("rMsg before einsum:2222 " + Arrays.toString(rMsg.sizes().vec().get())); //[8, 16, 16]
            TensorVector kVec = new TensorVector();
            kVec.push_back(k);    // k 和 rAtt 必须是有效指针
            kVec.push_back(rAtt);
            TensorVector vVec = new TensorVector();
            vVec.push_back(v);
            vVec.push_back(rMsg);
            // k = k @ rAtt, v = v @ rMsg
//            k = torch.einsum("nhd, hde -> nhe", k, rAtt);
//            v = torch.einsum("nhd, hde -> nhe", v, rMsg);
            System.out.println("k before einsum: " + Arrays.toString(k.sizes().vec().get()));
//                Tensor k_trans = k.transpose(0, 1); // [H, N, D_H]
//                k = torch.matmul(k_trans, rAtt).transpose(0, 1).contiguous().retainReference();
            k = torch.einsum("nhd, hde -> nhe", kVec);
            v = torch.einsum("nhd, hde -> nhe", vVec);

            // 注意力计算
            Tensor srcIdx = edgeIndex.select(0, 0);
            Tensor dstIdx = edgeIndex.select(0, 1);

            Tensor k_j = k.index_select(0, srcIdx);
            Tensor q_i = q.index_select(0, dstIdx);

            // Score 计算
            Tensor alpha = (q_i.mul(k_j)).sum(-1).mul(relationPri.get(edgeName)).div(new Scalar(sqrtDim));
            // Softmax
            alpha = scatter_softmax(alpha, dstIdx, xDict.get(dstType).size(0));

            // 消息聚合: Message = V * Alpha
            Tensor msg = v.index_select(0, srcIdx).mul(alpha.unsqueeze(-1));

            // 将消息从 [E, H, D/H] 展平回 [E, D]
            Tensor msgFlat = msg.reshape(edgeIndex.size(1), outChannels);

            // 聚合到目标节点类型
            Tensor currentOut = msgSumDict.getOrDefault(dstType, torch.zeros_like(xDict.get(dstType)));
            currentOut.scatter_add_(0, dstIdx.unsqueeze(-1).expand_as(msgFlat), msgFlat);

            // 关键修复：确保 Tensor 在 Map 中存活
            msgSumDict.put(dstType, currentOut.retainReference());
//            }

        }

        // 2. 最终输出变换
        Map<String, Tensor> finalOut = new HashMap<>();
        for (String nodeType : xDict.keySet()) {
            Tensor out = msgSumDict.get(nodeType);
            if (out != null && out.defined()) {
                finalOut.put(nodeType, aLin.get(nodeType).forward(out).retainReference());
            } else {
                finalOut.put(nodeType, xDict.get(nodeType).retainReference());
            }
        }
        return finalOut;
    }
    public Map<String, Tensor> forward2(Map<String, Tensor> xDict, Map<String[], Tensor> edgeIndexDict) {
        // 用于存储每个目标节点类型收到的所有消息
        Map<String, List<Tensor>> msgDict = new HashMap<>();

        for (Map.Entry<String[], Tensor> entry : edgeIndexDict.entrySet()) {
            String[] eType = entry.getKey(); // [src_node, relation, dst_node]
            Tensor edgeIndex = entry.getValue();
            String edgeName = String.join("_", eType);

            String srcType = eType[0];
            String relType = eType[1];
            String dstType = eType[2];

            // 1. 节点投影与维度重塑 [N, H, D/H]
            Tensor k = kLin.get(srcType).forward(xDict.get(srcType)).view(-1, heads, dK);
            Tensor v = vLin.get(srcType).forward(xDict.get(srcType)).view(-1, heads, dK);
            Tensor q = qLin.get(dstType).forward(xDict.get(dstType)).view(-1, heads, dK);

            // 2. 关系特定的投影 (Relation-specific Projection)
            // K = K * R_att, V = V * R_msg
            Tensor rAtt = relationAtt.get(edgeName);
            Tensor rMsg = relationMsg.get(edgeName);

            // 使用 einsum 或 matmul 进行 Head-wise 乘法
            // k: [N, H, D/H], rAtt: [H, D/H, D/H] -> [N, H, D/H]
            k = torch.einsum("nhd, hde -> nhe", new TensorVector(k, rAtt));
            v = torch.einsum("nhd, hde -> nhe",new TensorVector( v, rMsg));

            // 3. 计算注意力 Score (基于 edge_index)
            Tensor srcIdx = edgeIndex.select(0, 0);
            Tensor dstIdx = edgeIndex.select(0, 1);

            Tensor k_j = k.index_select(0, srcIdx); // 源节点
            Tensor q_i = q.index_select(0, dstIdx); // 目标节点

            // Score = (q_i * k_j).sum(-1) * Pri / sqrt(dK)
            Tensor alpha = (q_i.mul(k_j)).sum(-1).mul(relationPri.get(edgeName)).div(new Scalar(sqrtDim));

            // 4. Softmax 归一化与消息聚合 (这里简化处理，实际需对 dstIdx 做 softmax)
            alpha = scatter_softmax(alpha, dstIdx, xDict.get(dstType).size(0));
            Tensor msg = v.index_select(0, srcIdx).mul(alpha.unsqueeze(-1));

            // 5. 聚合到目标节点
            Tensor out = torch.zeros_like(xDict.get(dstType));
            out.scatter_add_(0, dstIdx.unsqueeze(-1).expand_as(msg.view(-1, outChannels)), msg.view(-1, outChannels));

            msgDict.computeIfAbsent(dstType, _k -> new ArrayList<>()).add(out);
        }

        // 6. 语义级融合与残差输出
        Map<String, Tensor> finalOut = new HashMap<>();
        for (String nodeType : xDict.keySet()) {
            if (msgDict.containsKey(nodeType)) {
                // 简单求和聚合来自不同关系的消息
                Tensor aggregated = msgDict.get(nodeType).get(0);
                for (int i = 1; i < msgDict.get(nodeType).size(); i++) {
                    aggregated = aggregated.add(msgDict.get(nodeType).get(i));
                }
                // 最终线性映射
                finalOut.put(nodeType, aLin.get(nodeType).forward(aggregated));
            } else {
                finalOut.put(nodeType, xDict.get(nodeType)); // 无消息则返回原特征
            }
        }
        return finalOut;
    }
}
//public class HGTConv extends Module {
//    private Map<String, LinearImpl> kLin, qLin, vLin; // 针对节点类型的投影
//    private Map<String, LinearImpl> aLin;            // 针对节点类型的输出映射
//    private Map<String, Tensor> relationPri;         // 关系先验权重
//    private Map<String, Tensor> relationAtt;         // 关系特定注意力矩阵
//    private Map<String, Tensor> relationMsg;         // 关系特定消息矩阵
//
//    private long outChannels;
//    private int heads;
//    private double sqrtDim;
//
//    public HGTConv(Map<String, Integer> inChannelsDict, int outChannels,
//                   List<String> nodeTypes, List<String[]> edgeTypes, int heads) {
//        super();
//        this.outChannels = outChannels;
//        this.heads = heads;
//        this.sqrtDim = Math.sqrt(outChannels / (double) heads);
//
//        this.kLin = new HashMap<>();
//        this.qLin = new HashMap<>();
//        this.vLin = new HashMap<>();
//        this.aLin = new HashMap<>();
//
//        // 1. 为每种节点类型注册 LinearImpl (K, Q, V, A)
//        for (String nodeType : nodeTypes) {
//            int inC = inChannelsDict.get(nodeType);
//            kLin.put(nodeType, new LinearImpl(inC, outChannels));
//            qLin.put(nodeType, new LinearImpl(inC, outChannels));
//            vLin.put(nodeType, new LinearImpl(inC, outChannels));
//            aLin.put(nodeType, new LinearImpl(outChannels, outChannels));
//
//            register_module("k_lin_" + nodeType, kLin.get(nodeType));
//            register_module("q_lin_" + nodeType, qLin.get(nodeType));
//            register_module("v_lin_" + nodeType, vLin.get(nodeType));
//            register_module("a_lin_" + nodeType, aLin.get(nodeType));
//        }
//
//        // 2. 为每种边类型注册关系特定的变换张量
//        this.relationAtt = new HashMap<>();
//        this.relationMsg = new HashMap<>();
//        this.relationPri = new HashMap<>();
//
//        for (String[] edgeType : edgeTypes) {
//            String edgeName = String.join("_", edgeType);
//            // 关系特定的注意力矩阵 [heads, C/heads, C/heads]
//            Tensor att = torch.randn(new long[]{heads, outChannels / heads, outChannels / heads});
//            Tensor msg = torch.randn(new long[]{heads, outChannels / heads, outChannels / heads});
//            Tensor pri = torch.ones(new long[]{heads}); // 关系先验
//
//            relationAtt.put(edgeName, att);
//            relationMsg.put(edgeName, msg);
//            relationPri.put(edgeName, pri);
//
//            register_parameter("rel_att_" + edgeName, att);
//            register_parameter("rel_msg_" + edgeName, msg);
//            register_parameter("rel_pri_" + edgeName, pri);
//        }
//    }
//
//    public Map<String, Tensor> forward(Map<String, Tensor> xDict, Map<String[], Tensor> edgeIndexDict) {
//        Map<String, List<Tensor>> outDict = new HashMap<>();
//
//        // --- 核心逻辑：计算每种边类型的注意力与消息 ---
//        for (Map.Entry<String[], Tensor> entry : edgeIndexDict.entrySet()) {
//            String[] eType = entry.getKey(); // [src_type, rel_type, dst_type]
//            String edgeName = String.join("_", eType);
//            Tensor edgeIndex = entry.getValue();
//
//            Tensor k = kLin.get(eType[0]).forward(xDict.get(eType[0])).view(-1, heads, outChannels / heads);
//            Tensor v = vLin.get(eType[0]).forward(xDict.get(eType[0])).view(-1, heads, outChannels / heads);
//            Tensor q = qLin.get(eType[1]).forward(xDict.get(eType[2])).view(-1, heads, outChannels / heads);
//
//            // 1. 计算注意力 Score: (Q * R_att) * K
//            // 这里涉及关系特定的 R_att 张量乘法
//            // ... (逻辑简化)
//
//            // 2. 计算消息 Message: V * R_msg
//            // ... (逻辑简化)
//
//            // 3. 聚合消息并将结果加入对应节点类型的 List 中
//        }
//
//        // --- 最终映射 ---
//        Map<String, Tensor> finalOut = new HashMap<>();
//        for (String nodeType : xDict.keySet()) {
//            // 对所有进入该节点的消息进行聚合，最后通过 aLin.forward(x) 变换
//            // finalOut.put(nodeType, aLin.get(nodeType).forward(aggregated_x));
//        }
//        return finalOut;
//    }
//}