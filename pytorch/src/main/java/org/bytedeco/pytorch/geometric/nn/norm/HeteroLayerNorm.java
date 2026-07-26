package org.bytedeco.pytorch.geometric.nn.norm;
import org.bytedeco.pytorch.autograd.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.Tensor;

import java.util.HashMap;
import java.util.Map;

/**
 * 6. org.bytedeco.pytorch.geometric.nn.norm.HeteroLayerNorm
 * 对每种节点类型应用独立的 LayerNorm
 */

public class HeteroLayerNorm extends org.bytedeco.pytorch.nn.Module {
    private long inChannels;
    private double eps;
    private boolean affine;
    private String mode;

    // 存储每种节点类型对应的 LayerNorm 模块
    private Map<String, LayerNorm> lns = new HashMap<>();

    /**
     * @param inChannels 输入维度
     * @param nodeTypes  节点类型列表，例如 ["user", "item", "store"]
     */
    public HeteroLayerNorm(long inChannels, String[] nodeTypes) {
        this(inChannels, nodeTypes, 1e-5, true, "node");
    }

    public HeteroLayerNorm(long inChannels, String[] nodeTypes, double eps, boolean affine, String mode) {
        super();
        this.inChannels = inChannels;
        this.eps = eps;
        this.affine = affine;
        this.mode = mode;

        for (String nodeType : nodeTypes) {
            // 为每种类型创建一个独立的 LayerNorm
            LayerNorm ln = new LayerNorm(inChannels, eps, affine);

            // 必须在 Module 中注册，否则无法进行参数优化和状态保存
            // JavaCPP 中建议使用 register_module
            register_module("ln_" + nodeType, ln);
            lns.put(nodeType, ln);
        }
    }

    /**
     * 对应 Python 的 forward(x_dict, batch_dict)
     *
     * @param xDict     Map<节点类型, 特征Tensor>
     * @param batchDict Map<节点类型, Batch索引Tensor>，可为 null
     * @return 处理后的 Map
     */
    public Map<String, Tensor> forward(Map<String, Tensor> xDict, Map<String, Tensor> batchDict) {
        Map<String, Tensor> outDict = new HashMap<>();

        for (Map.Entry<String, Tensor> entry : xDict.entrySet()) {
            String nodeType = entry.getKey();
            Tensor x = entry.getValue();

            if (!lns.containsKey(nodeType)) {
                throw new RuntimeException("Node type " + nodeType + " not initialized in HeteroLayerNorm");
            }

            // 获取对应的 Batch 索引（如果存在）
            Tensor batch = (batchDict != null) ? batchDict.get(nodeType) : null;

            // 调用我们之前实现的 LayerNorm
            Tensor out = lns.get(nodeType).forward2(x);
            outDict.put(nodeType, out);
        }

        return outDict;
    }
}

//public class HeteroLayerNorm extends Module {
//    private Map<String, LayerNorm> lns;
//
//    public HeteroLayerNorm(Map<String, Long> inChannelsMap) {
//        super();
//        this.lns = new HashMap<>();
//
//        for (Map.Entry<String, Long> entry : inChannelsMap.entrySet()) {
//            String nodeType = entry.getKey();
//            Long channels = entry.getValue();
//
//            LayerNorm ln = new LayerNorm(channels);
//            lns.put(nodeType, ln);
//            register_module("ln_" + nodeType, ln);
//        }
//    }
//
//    public Map<String, Tensor> forward(Map<String, Tensor> xDict) {
//        Map<String, Tensor> outDict = new HashMap<>();
//        for (Map.Entry<String, Tensor> entry : xDict.entrySet()) {
//            String nodeType = entry.getKey();
//            Tensor x = entry.getValue();
//            if (lns.containsKey(nodeType)) {
//                outDict.put(nodeType, lns.get(nodeType).forward(x));
//            } else {
//                outDict.put(nodeType, x);
//            }
//        }
//        return outDict;
//    }
//}