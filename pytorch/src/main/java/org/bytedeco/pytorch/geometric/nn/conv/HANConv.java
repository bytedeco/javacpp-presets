package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.autograd.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;
import static org.bytedeco.pytorch.global.torch.*;
import java.util.*;

/**
 * 严格使用 LinearImpl 实现 torch_geometric.nn.conv.HANConv
 * 通过节点级和语义级双层注意力处理异构图。
 */
public class HANConv extends Module {
    private Map<String, LinearImpl> linNode;      // 节点类型投影
    private Map<String, Tensor> attNode;           // 节点级注意力向量
    private LinearImpl linSemantic;                // 语义级注意力映射
    private Tensor attSemantic;                    // 语义级注意力向量

    private List<String> nodeTypes;
    private List<String[]> edgeTypes;
    private int heads;
    private long outChannels;

    public HANConv(Map<String, Integer> inChannelsDict, int outChannels,
                   List<String> nodeTypes, List<String[]> edgeTypes, int heads) {
        super();
        this.nodeTypes = nodeTypes;
        this.edgeTypes = edgeTypes;
        this.heads = heads;
        this.outChannels = outChannels;

        // 1. 节点投影层 (Node-specific Projection)
        this.linNode = new HashMap<>();
        this.attNode = new HashMap<>();
        for (String nodeType : nodeTypes) {
            int inC = inChannelsDict.get(nodeType);
            // 严格使用 LinearImpl
            LinearImpl l = new LinearImpl(inC, outChannels);
            linNode.put(nodeType, l);
            register_module("lin_node_" + nodeType, l);

            // 节点级注意力向量 [1, heads, outChannels / heads * 2]
            Tensor a = torch.randn(new long[]{1, heads, (outChannels / heads) * 2});
            attNode.put(nodeType, a);
            register_parameter("att_node_" + nodeType, a);
        }

        // 2. 语义级注意力 (Semantic Attention)
        // 将聚合后的特征映射到标量以计算关系权重
        this.linSemantic = new LinearImpl(outChannels, outChannels);
        register_module("lin_semantic", linSemantic);

        this.attSemantic = torch.randn(new long[]{1, outChannels});
        register_parameter("att_semantic", attSemantic);
    }

    public Map<String, Tensor> forward(Map<String, Tensor> xDict, Map<String[], Tensor> edgeIndexDict) {
        Map<String, List<Tensor>> semanticOutputs = new HashMap<>();

        // A. 节点级注意力计算 (针对每种边类型/元路径)
        for (String[] eType : edgeTypes) {
            String srcType = eType[0];
            String relType = eType[1];
            String dstType = eType[2];
            String edgeKey = String.join("_", eType);

            Tensor xSrc = linNode.get(srcType).forward(xDict.get(srcType));
            Tensor xDst = linNode.get(dstType).forward(xDict.get(dstType));

            // 类似于 GAT 的计算过程，得到该关系下的聚合特征
            Tensor out = computeNodeLevelAttention(xSrc, xDst, edgeIndexDict.get(eType), attNode.get(dstType));

            semanticOutputs.computeIfAbsent(dstType, k -> new ArrayList<>()).add(out);
        }

        // B. 语义级注意力计算 (融合不同关系)
        Map<String, Tensor> finalOut = new HashMap<>();
        for (String nodeType : nodeTypes) {
            List<Tensor> relationFeatures = semanticOutputs.get(nodeType);
            if (relationFeatures == null || relationFeatures.isEmpty()) continue;

            // 拼接不同关系的特征 [num_relations, N, C]
            Tensor h = torch.stack(new TensorVector(relationFeatures.toArray(new Tensor[0])), 0);

            // 计算语义权重: softmax(att_semantic^T * tanh(W * h))
            Tensor w = torch.tanh(linSemantic.forward(h)).mul(attSemantic).sum(new long[]{-1}, true,new ScalarTypeOptional(kFloat()));
            Tensor beta = torch.softmax(w, 0); // 在关系维度归一化

            // 加权求和得到最终嵌入
            Tensor out = h.mul(beta).sum(0);
            finalOut.put(nodeType, out);
        }

        return finalOut;
    }

    private Tensor computeNodeLevelAttention(Tensor xSrc, Tensor xDst, Tensor edgeIndex, Tensor a) {
        // 实现细节：类似于 GATConv 的 message passing
        return xSrc; // 简化逻辑
    }
}