package org.bytedeco.pytorch.distribution;

import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import static org.bytedeco.pytorch.global.torch.*;

/**
 * Independent：reinterprets the rightmost batch dims of a base distribution as event dims
 * (mirrors torch.distributions.Independent).
 */
public class Independent extends Distribution implements AutoCloseable {
    // 基础分布（不可为空）
    private final Distribution baseDist;
    // 要重新解释的批量维度数（≥0）
    private final int reinterpretedBatchNdims;
    // 防止重复释放
    private boolean isClosed = false;

    /**
     * 构造函数
     * @param base 基础分布（不可为null）
     * @param ndims 要重新解释的批量维度数（必须≥0）
     */
    public Independent(Distribution base, int ndims) {
        if (base == null) {
            throw new IllegalArgumentException("基础分布base_dist不能为null！");
        }
        if (ndims < 0) {
            throw new IllegalArgumentException("重新解释的维度数reinterpreted_batch_ndims必须≥0，当前值：" + ndims);
        }

        this.baseDist = base;
        this.reinterpretedBatchNdims = ndims;
    }

    @Override
    public String name() {
        return "Independent(" + baseDist.name() + ")";
    }

    /**
     * Sample from the base distribution (no caching — each call is independent).
     */
    @Override
    public Tensor sample(long... s) {
        return baseDist.sample(s);
    }

    /**
     * 对数概率：对基础分布的log_prob按指定维度求和
     */
    @Override
    public Tensor log_prob(Tensor v) {
        Tensor baseLp = baseDist.log_prob(v);
        Tensor currentLp = baseLp.clone().detach();
        Tensor prevLp = null;

        for (int i = 0; i < reinterpretedBatchNdims; i++) {
            if (currentLp.dim() < 1) {
                break;
            }
            if (prevLp != null) {
                prevLp.close();
            }
            prevLp = currentLp;
            currentLp = currentLp.sum(new long[]{-1}, true, new ScalarTypeOptional());
        }

        baseLp.close();
        if (prevLp != null && prevLp != currentLp) {
            prevLp.close();
        }

        return currentLp.squeeze();
    }

    /**
     * 熵：对基础分布的熵按指定维度求和
     */
    @Override
    public Tensor entropy() {
        Tensor baseEnt = baseDist.entropy();
        Tensor currentEnt = baseEnt.clone().detach();
        Tensor prevEnt = null;

        for (int i = 0; i < reinterpretedBatchNdims; i++) {
            if (currentEnt.dim() < 1) {
                break;
            }
            if (prevEnt != null) {
                prevEnt.close();
            }
            prevEnt = currentEnt;
            currentEnt = currentEnt.sum(new long[]{-1}, true, new ScalarTypeOptional());
        }

        baseEnt.close();
        if (prevEnt != null && prevEnt != currentEnt) {
            prevEnt.close();
        }

        return currentEnt.squeeze();
    }

    /**
     * 均值：与基础分布均值完全一致
     */
    @Override
    public Tensor mean() {
        return baseDist.mean().clone().detach();
    }

    /**
     * 安全释放资源
     */
    @Override
    public void close() {
        if (!isClosed) {
            // 释放基础分布
            if (baseDist instanceof AutoCloseable) {
                try {
                    ((AutoCloseable) baseDist).close();
                } catch (Exception e) {
                    throw new RuntimeException("释放基础分布资源失败", e);
                }
            }
            isClosed = true;
        }
    }

    // Getter方法
    public Distribution getBaseDist() {
        return baseDist;
    }

    public int getReinterpretedBatchNdims() {
        return reinterpretedBatchNdims;
    }
}
