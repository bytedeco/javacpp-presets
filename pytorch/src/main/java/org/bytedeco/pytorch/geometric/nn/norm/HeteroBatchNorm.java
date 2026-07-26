package org.bytedeco.pytorch.geometric.nn.norm;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.optim.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.nn.options.BatchNormOptions;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;

import static org.bytedeco.pytorch.global.torch.zeros_like;

public class HeteroBatchNorm extends org.bytedeco.pytorch.nn.Module {
    private long inChannels;
    private int numTypes;
    private double eps;
    private double momentum; //DoubleOptional
    private boolean affine;
    private boolean trackRunningStats;

    // 存储不同类型的 BatchNorm 实例
    private BatchNorm[] bns;

    //DoubleOptional
    public HeteroBatchNorm(long inChannels, int numTypes, double eps, double momentum, boolean affine, boolean trackRunningStats) {
        super();
        this.inChannels = inChannels;
        this.numTypes = numTypes;
        this.eps = eps;
        this.momentum = momentum;
        this.affine = affine;
        this.trackRunningStats = trackRunningStats;
//public BatchNorm(long inChannels,  double eps, double momentum, boolean affine, boolean track_running_stats, boolean allowSingleElement) 
        this.bns = new BatchNorm[numTypes];
        for (int i = 0; i < numTypes; i++) {
            // 配置每个类型的独立 BatchNorm
            BatchNormOptions opts = new BatchNormOptions(inChannels);
//            opts.eps().put(eps);
//            opts.momentum().put(momentum);
//            opts.affine().put(affine);
//            opts.track_running_stats().put(trackRunningStats);

            bns[i] = new BatchNorm(inChannels, eps, momentum, affine, trackRunningStats, false);
            // 注册子模块，名称通常为 "bn0", "bn1" ...
            register_module("bn" + i, bns[i]);
        }
    }

    /**
     * @param x       输入特征 [N, C]
     * @param typeIdx 类型索引 [N]，范围在 [0, numTypes-1] 之间
     */
    public Tensor forward(Tensor x, Tensor typeIdx) {
        x = x.contiguous();
        Tensor out = zeros_like(x);

        for (int i = 0; i < numTypes; i++) {
            // 找到属于当前类型的节点掩码
            Tensor mask = typeIdx.eq(new Scalar(i));

            // 检查该类型是否有节点，避免空输入报错
            if (mask.any().item().toBool()) {
                Tensor xType = x.masked_select(mask.unsqueeze(1).expand_as(x)).view(new long[]{-1, inChannels});
                Tensor outType = bns[i].forward(xType);

                // 将结果写回 (使用 masked_scatter_)
                out.masked_scatter_(mask.unsqueeze(1).expand_as(out), outType);
            }
        }
        return out;
    }
}

//public class HeteroBatchNorm extends Module {
//    private Map<String, BatchNorm> bns;
//
//    public HeteroBatchNorm(Map<String, Long> inChannelsMap) {
//        super();
//        this.bns = new HashMap<>();
//
//        for (Map.Entry<String, Long> entry : inChannelsMap.entrySet()) {
//            String nodeType = entry.getKey();
//            Long channels = entry.getValue();
//
//            BatchNorm bn = new BatchNorm(channels, true);
//            bns.put(nodeType, bn);
//
//            // 注册子模块，以便 Optimizer 识别参数
//            register_module("bn_" + nodeType, bn);
//        }
//    }
//
//    public Map<String, Tensor> forward(Map<String, Tensor> xDict) {
//        Map<String, Tensor> outDict = new HashMap<>();
//
//        for (Map.Entry<String, Tensor> entry : xDict.entrySet()) {
//            String nodeType = entry.getKey();
//            Tensor x = entry.getValue();
//
//            if (bns.containsKey(nodeType)) {
//                outDict.put(nodeType, bns.get(nodeType).forward(x));
//            } else {
//                // 如果没有对应的 BN，原样返回 (或报错，视需求而定)
//                outDict.put(nodeType, x);
//            }
//        }
//        return outDict;
//    }
//}
//
