package org.bytedeco.pytorch.geometric.nn.norm;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.pytorch.nn.modules.BatchNorm1dImpl;
import org.bytedeco.pytorch.nn.options.BatchNormOptions;
import org.bytedeco.pytorch.Tensor;

public class BatchNorm extends org.bytedeco.pytorch.nn.Module {
    public BatchNorm1dImpl innerBN;
    private boolean allowSingleElement;

    //class BatchNorm(in_channels: int, eps: float = 1e-05, momentum: Optional[float] = 0.1, affine: bool = True, track_running_stats: bool = True, allow_single_element: bool = False, 
    public BatchNorm(long inChannels, double eps, double momentum, boolean affine, boolean track_running_stats, boolean allowSingleElement) {
        super();
        this.allowSingleElement = allowSingleElement;

        // 初始化底层 BatchNorm1d
        BatchNormOptions options = new BatchNormOptions(inChannels);
        options.eps().put(eps);
        ;//1e-5);
        options.momentum().put(momentum);//0.1);
        options.affine().put(affine);//true);
        options.track_running_stats().put(track_running_stats);//true);

        this.innerBN = new BatchNorm1dImpl(options);
        register_module("module", innerBN);
    }

    public BatchNorm(Pointer p) {
        super(p);
    }

    public Tensor forward(Tensor x) {
        // x 的形状通常是 [N, C]
        long numElements = x.size(0);

        // 处理 allow_single_element 的核心逻辑
        if (numElements <= 1 && allowSingleElement) {
            // 备份原始训练状态
            boolean wasTraining = innerBN.is_training();

            // 如果只有 1 个元素且处于训练模式，BN 无法计算方差
            // 此时临时切换到 eval 模式，使用 running_mean/var 而不更新它们
            innerBN.eval();
            Tensor out = innerBN.forward(x);

            // 恢复原始状态
            if (wasTraining) innerBN.train(true);
            return out;
        }

        // 正常情况直接调用
        return innerBN.forward(x);
    }
}
//public class BatchNorm extends Module {
//
//    private BatchNorm1dImpl batchNorm;
//    private boolean allowSingleElement;
//    
//    public BatchNorm(int inChannels, boolean allowSingleElement) {
//        super();
//        this.batchNorm = new BatchNorm1dImpl(inChannels);
//        this.allowSingleElement = allowSingleElement;
//    }
//    
//    @Override
//    public Tensor forward(Tensor x) {
//        long numElements = x.size(0);
//        if (numElements == 1 && allowSingleElement) {
//            boolean wasTraining = batchNorm.is_training();
//            batchNorm.eval();
//            Tensor result = x;
//            var out =batchNorm.forward(x);
//            if(wasTraining) {
//                batchNorm.train(wasTraining);
//            }
//            return out;
//        }
//        return batchNorm.forward(x);
//    }
//}
