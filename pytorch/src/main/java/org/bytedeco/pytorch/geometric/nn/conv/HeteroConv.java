package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.c10.*;
import org.bytedeco.pytorch.autograd.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.StringTensorDict;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;

import java.util.*;

public class HeteroConv extends Module {
    private Map<String, MessagePassing> convs;
    private String aggr;

    public HeteroConv(Map<String, MessagePassing> convsMap, String aggr) {
        super();
        this.convs = convsMap;
        this.aggr = aggr;

        // 注册子模块
        for (Map.Entry<String, MessagePassing> entry : convsMap.entrySet()) {
            register_module("conv_" + entry.getKey().replace(",", "_"), entry.getValue());
        }

    }

    public Map<String, Tensor> forward(Map<String, Tensor> xDict, Map<String, Tensor> edgeIndexDict) {
        Map<String, List<Tensor>> outDict = new HashMap<>();

        // 显式遍历 edgeIndexDict，确保只处理当前存在的边
        for (String edgeTypeStr : edgeIndexDict.keySet()) {
            // 1. 获取对应的算子
            MessagePassing conv = this.convs.get(edgeTypeStr);
            if (conv == null) {
                System.err.println("Warning: No conv defined for edge type: " + edgeTypeStr);
                continue;
            }

            String[] tokens = edgeTypeStr.split(",");

            String srcType = tokens[0];
            String dstType = tokens[2];
            Tensor xSrc = xDict.get(srcType);
            Tensor xDst = xDict.get(dstType);
            Tensor edgeIndex = edgeIndexDict.get(edgeTypeStr);

            long numSrcNodes = xSrc.size(0);
            long numDstNodes = xDst.size(0);

// 获取 edgeIndex 的最大值
// edgeIndex 形状为 [2, E]，第 0 行是 Source，第 1 行是 Target
            long maxSrcIdx = edgeIndex.select(0, 0).max().item_long();
            long maxDstIdx = edgeIndex.select(0, 1).max().item_long();


            if (maxSrcIdx >= numSrcNodes) {
                throw new RuntimeException("越界错误：边类型 " + edgeTypeStr + " 的源节点索引 " + maxSrcIdx + " 超过了节点数 " + numSrcNodes);
            }
            if (maxDstIdx >= numDstNodes) {
                throw new RuntimeException("越界错误：边类型 " + edgeTypeStr + " 的目标节点索引 " + maxDstIdx + " 超过了节点数 " + numDstNodes);
            }

            // 2. 增加断点式维度校验
            long expectedInDim = conv.named_parameters().values().get(0).size(1);//keys.get("linL.weight")
            if (expectedInDim != xSrc.size(1)) {
                throw new RuntimeException(String.format(
                        "CRITICAL ERROR: Edge [%s] expected inDim %d, but got xSrc dim %d. " +
                                "This means the wrong Conv object is being used!",
                        edgeTypeStr, expectedInDim, xSrc.size(1)));
            }

            // 3. 执行计算
            Tensor out;
            if (tokens[0].equals(tokens[2])) {
                System.out.println("Checking Edge: 222" + edgeTypeStr);
                out = conv.forward(xSrc, edgeIndex);
            } else {
                System.out.println("Checking Edge:3333 " + edgeTypeStr);
                out = conv.forward(xSrc, xDst, edgeIndex);
            }

            outDict.computeIfAbsent(tokens[2], k -> new ArrayList<>()).add(out);

        }
        // 聚合指向同一目标节点类型的不同关系
        Map<String, Tensor> finalOut = new HashMap<>();
        for (Map.Entry<String, List<Tensor>> entry : outDict.entrySet()) {
            String nodeType = entry.getKey();
            List<Tensor> tensorList = entry.getValue();

            if (tensorList.isEmpty()) continue;

            Tensor aggregated = tensorList.get(0);
            for (int i = 1; i < tensorList.size(); i++) {
                if (aggr.equals("sum") || aggr.equals("mean")) {
                    aggregated = aggregated.add(tensorList.get(i));
                } else if (aggr.equals("max")) {
                    aggregated = torch.max(aggregated, tensorList.get(i));
                }
            }

            if (aggr.equals("mean") && tensorList.size() > 1) {
                aggregated = aggregated.div(new Scalar(tensorList.size()));
            }

            finalOut.put(nodeType, aggregated);
        }

        return finalOut;
        // ... 聚合逻辑
    }
}


//            Tensor xSrc = xDict.get(tokens[0]);
//            Tensor xDst = xDict.get(tokens[2]);
//            Tensor edgeIndex = edgeIndexDict.get(edgeTypeStr);

//            System.out.println("---HeteroConv Debugging Conv Parameters ---");
//            StringTensorDict params = conv.named_parameters();
//            for (int j = 0; j < params.size(); j++) {
////                params.keys().get(j).getString();
//                System.out.println("Available Key: " + params.keys().get(j).getString());
//            }


//    public Map<String, Tensor> forward2(Map<String, Tensor> xDict, Map<String, Tensor> edgeIndexDict) {
//        // 用于存放每个目标节点类型收到的所有消息列表
//        Map<String, List<Tensor>> outDict = new HashMap<>();
//
////        for (Map.Entry<String, MessagePassing> entry : convs.entrySet()) {
////            String edgeTypeStr = entry.getKey();
////            MessagePassing conv = entry.getValue();
////
////            // 解析三元组: [src_type, rel_type, dst_type]
////            String[] tokens = edgeTypeStr.split(",");
////            String srcType = tokens[0];
////            String dstType = tokens[2];
////
////            if (!edgeIndexDict.containsKey(edgeTypeStr)) continue;
////
////            Tensor edgeIndex = edgeIndexDict.get(edgeTypeStr);
////            Tensor xSrc = xDict.get(srcType);
////            Tensor xDst = xDict.get(dstType);
////
////            Tensor out;
////            if (srcType.equals(dstType)) {
////                // 同构边情形
////                out = conv.forward(xSrc, edgeIndex);
////            } else {
////                // 异构/二部图边情形：传入源和目标特征对
////                // 这确保了 MessagePassing 内部 aggregate 后的维度匹配 xDst 的数量 (100)
////                out = conv.forward(new Tensor[]{xSrc, xDst}, edgeIndex);
////            }
////
////            outDict.computeIfAbsent(dstType, k -> new ArrayList<>()).add(out);
////        }
////
////        Map<String, List<Tensor>> outDict = new HashMap<>();
//
//        for (Map.Entry<String, MessagePassing> entry : convs.entrySet()) {
//            String edgeTypeStr = entry.getKey();
//            MessagePassing conv = entry.getValue();
//
//            String[] tokens = edgeTypeStr.split(",");
//            String srcType = tokens[0];
//            String dstType = tokens[2];
//
//            if (!edgeIndexDict.containsKey(edgeTypeStr)) continue;
//
//            Tensor edgeIndex = edgeIndexDict.get(edgeTypeStr);
//            Tensor xSrc = xDict.get(srcType);
//            Tensor xDst = xDict.get(dstType);
//
//            Tensor out;
//            if (srcType.equals(dstType)) {
//                // 同构边：直接调用
//                out = conv.forward(xSrc, edgeIndex);
//            } else {
//                // 异构边修复：
//                // 绝大多数子类算子（如 SAGEConv）的 forward 签名其实支持 (Tensor, Tensor)
//                // 如果你的子类没有定义这个签名，我们需要手动调用 propagate
//                out = conv.forward(xSrc, xDst, edgeIndex);
//            }
//
//
//           
//            StringTensorDict params = conv.named_parameters();
//            for (int j = 0; j < params.size(); j++) {
////                params.keys().get(j).getString();
//                System.out.println("Available Key: " + params.keys().get(j).getString());
//            }
//            outDict.computeIfAbsent(dstType, k -> new ArrayList<>()).add(out);
//        }
//
//        // 聚合指向同一目标节点类型的不同关系
//        Map<String, Tensor> finalOut = new HashMap<>();
//        for (Map.Entry<String, List<Tensor>> entry : outDict.entrySet()) {
//            String nodeType = entry.getKey();
//            List<Tensor> tensorList = entry.getValue();
//
//            if (tensorList.isEmpty()) continue;
//
//            Tensor aggregated = tensorList.get(0);
//            for (int i = 1; i < tensorList.size(); i++) {
//                if (aggr.equals("sum") || aggr.equals("mean")) {
//                    aggregated = aggregated.add(tensorList.get(i));
//                } else if (aggr.equals("max")) {
//                    aggregated = torch.max(aggregated, tensorList.get(i));
//                }
//            }
//
//            if (aggr.equals("mean") && tensorList.size() > 1) {
//                aggregated = aggregated.div(new Scalar(tensorList.size()));
//            }
//
//            finalOut.put(nodeType, aggregated);
//        }
//
//        return finalOut;
//    }




// System.out.println("Processing edge: " + edgeTypeStr);
//            System.out.println("xSrc shape: " + Arrays.toString(xSrc.sizes().vec().get()));
//        System.out.println("xDst shape: " + Arrays.toString(xDst.sizes().vec().get()));
//        System.out.println("edgeIndex max src index: " + edgeIndex.select(0, 0).max().item().toFloat());
//        System.out.println("edgeIndex max dst index: " + edgeIndex.select(0, 1).max().item().toFloat());
//        System.out.println("Current Edge: " + edgeTypeStr);
//            System.out.println("Src type: " + srcType + ", Dst type: " + dstType);
////            System.out.println("Output shape: " + Arrays.stream(conv.named_parameters().keys().get()).map(String::toString).collect(Collectors.toList()));
//            System.out.println("Conv weight linL.weight shape: " + Arrays.toString(conv.named_parameters().get("linL.weight").sizes().vec().get()));
//        System.out.println("Conv weight linL.bias  shape: " + Arrays.toString(conv.named_parameters().get("linL.bias").sizes().vec().get()));
//        System.out.println("Conv weight  linR.weight shape: " + Arrays.toString(conv.named_parameters().get("linR.weight").sizes().vec().get()));
//        System.out.println("Conv weight linR.bias shape: " + Arrays.toString(conv.named_parameters().get("linR.bias").sizes().vec().get()));
//
//        System.out.println("--- Debugging Conv Parameters ---");

//            System.out.println("Checking Edge: " + edgeTypeStr);
//            System.out.println("Source nodes: " + numSrcNodes + ", Max index in edgeIndex[0]: " + maxSrcIdx);
//            System.out.println("Target nodes: " + numDstNodes + ", Max index in edgeIndex[1]: " + maxDstIdx);
//            System.out.println("Conv weight linL.weight shape: " + Arrays.toString(conv.named_parameters().get("linL.weight").sizes().vec().get()));
//            System.out.println("Conv weight linL.bias  shape: " + Arrays.toString(conv.named_parameters().get("linL.bias").sizes().vec().get()));
//            System.out.println("Conv weight  linR.weight shape: " + Arrays.toString(conv.named_parameters().get("linR.weight").sizes().vec().get()));
//            System.out.println("Conv weight linR.bias shape: " + Arrays.toString(conv.named_parameters().get("linR.bias").sizes().vec().get()));

//public class HeteroConv extends Module {
//    // 存储 关系三元组 -> 卷积算子 的映射
//    private Map<String, MessagePassing> convs;
//    private String aggr;
//
//    /**
//     * @param convsMap 传入的算子映射，Key 为 "src_type,rel_type,dst_type"
//     * @param aggr     多边类型聚合方式 ("sum", "mean", "max", "cat")
//     */
//    public HeteroConv(Map<String, MessagePassing> convsMap, String aggr) {
//        super();
//        this.convs = convsMap;
//        this.aggr = aggr;
//
//        // 严格注册内部所有子模块，确保 LinearImpl 参数被收纳
//        for (Map.Entry<String, MessagePassing> entry : convsMap.entrySet()) {
//            register_module("conv_" + entry.getKey().replace(",", "_"), entry.getValue());
//        }
//    }
//
//    /**
//     * @param xDict         节点类型 -> 特征张量的字典
//     * @param edgeIndexDict 边类型三元组 -> 边索引张量的字典
//     * @return 聚合后的节点特征字典
//     */
//    public Map<String, Tensor> forward(Map<String, Tensor> xDict, Map<String, Tensor> edgeIndexDict) {
//        Map<String, List<Tensor>> outDict = new HashMap<>();
//
//        for (Map.Entry<String, MessagePassing> entry : convs.entrySet()) {
//            String edgeTypeStr = entry.getKey();
//            MessagePassing conv = entry.getValue();
//
//            String[] tokens = edgeTypeStr.split(",");
//            String srcType = tokens[0];
//            String dstType = tokens[2];
//
//            if (!edgeIndexDict.containsKey(edgeTypeStr)) continue;
//
//            Tensor edgeIndex = edgeIndexDict.get(edgeTypeStr);
//            Tensor xSrc = xDict.get(srcType);
//            Tensor xDst = xDict.get(dstType);
//
//            // 核心修正：异构卷积需要处理二部图（Bipartite Graph）
//            // 传入 (xSrc, xDst)，MessagePassing 才能知道输出行数应匹配 xDst (100) 而不是 xSrc (50)
////            Tensor out = conv.forward(new TensorVector(xSrc, xDst), edgeIndex);
//            Tensor out = conv.forward(xSrc, edgeIndex);
//            outDict.computeIfAbsent(dstType, k -> new ArrayList<>()).add(out);
//        }
//
//        Map<String, Tensor> finalOut = new HashMap<>();
//        for (Map.Entry<String, List<Tensor>> entry : outDict.entrySet()) {
//            String nodeType = entry.getKey();
//            List<Tensor> tensorList = entry.getValue();
//
//            // 执行聚合时，由于上面的修正，tensorList 里的所有张量现在行数都是一致的了
//            Tensor aggregated = tensorList.get(0);
//            for (int i = 1; i < tensorList.size(); i++) {
//                aggregated = aggregated.add(tensorList.get(i));
//            }
//            finalOut.put(nodeType, aggregated);
//        }
//        return finalOut;
//    }
//
//}


//   public Map<String, Tensor> forward2(Map<String, Tensor> xDict, Map<String, Tensor> edgeIndexDict) {
//        // 用于存放每个目标节点类型收到的所有消息
//        Map<String, List<Tensor>> outDict = new HashMap<>();
//
//        for (Map.Entry<String, MessagePassing> entry : convs.entrySet()) {
//            String edgeTypeStr = entry.getKey();
//            MessagePassing conv = entry.getValue();
//
//            // 解析三元组: [src_type, rel_type, dst_type]
//            String[] tokens = edgeTypeStr.split(",");
//            String srcType = tokens[0];
//            String dstType = tokens[2];
//
//            if (!edgeIndexDict.containsKey(edgeTypeStr)) continue;
//
//            Tensor edgeIndex = edgeIndexDict.get(edgeTypeStr);
//            Tensor xSrc = xDict.get(srcType);
//            Tensor xDst = xDict.get(dstType);
//
//            // 1. 执行特定的卷积操作 (此时 conv 内部的 LinearImpl 会被调用)
//            // 如果是二部图算子，通常传入元组 (x_src, x_dst)
//            Tensor out = conv.forward(xSrc, edgeIndex);
//
//            // 2. 将结果存入目标节点的待聚合列表
//            outDict.computeIfAbsent(dstType, k -> new ArrayList<>()).add(out);
//        }
//
//        // 3. 聚合指向同一目标节点类型的不同关系结果
//        Map<String, Tensor> finalOut = new HashMap<>();
//        for (Map.Entry<String, List<Tensor>> entry : outDict.entrySet()) {
//            String nodeType = entry.getKey();
//            List<Tensor> tensorList = entry.getValue();
//
//            if (tensorList.isEmpty()) continue;
//
//            Tensor aggregated;
//            if (aggr.equals("sum")) {
//                aggregated = tensorList.get(0);
//                for (int i = 1; i < tensorList.size(); i++) aggregated = aggregated.add(tensorList.get(i));
//            } else if (aggr.equals("mean")) {
//                Tensor sum = tensorList.get(0);
//                for (int i = 1; i < tensorList.size(); i++) sum = sum.add(tensorList.get(i));
//                aggregated = sum.div(new Scalar(tensorList.size()));
//            } else {
//                aggregated = torch.stack(new TensorVector(
//                    tensorList.toArray(new Tensor[0])
//                ), 0).sum(0); // 默认行为
//            }
//            finalOut.put(nodeType, aggregated);
//        }
//
//        return finalOut;
//    }