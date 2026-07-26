package org.bytedeco.pytorch.distribution;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import static org.bytedeco.pytorch.global.torch.bernoulli;
import static org.bytedeco.pytorch.global.torch.log;

public class Bernoulli extends Distribution {
    
    private final Tensor probs;
    private static final Scalar SCALAR_1 = new Scalar(1.0);
    private static final Scalar EPS = new Scalar(1e-8); // 数值稳定性极小值

    public Bernoulli(Tensor probs) {
        if (torch.any(probs.lt(new Scalar(0.0))).item().toBool() || torch.any(probs.gt(new Scalar(1.0))).item().toBool()) {
            throw new IllegalArgumentException("伯努利分布参数probs必须满足 0 ≤ probs ≤ 1！");
        }
        this.probs = probs;
    }
    
    public String name() { 
        return "Bernoulli"; 
    }

    // 使用伯努利分布采样
    public Tensor sample(long... sampleShape) {
        long[] extendedShape = getExtendedShape(probs, sampleShape);
        // 步骤2：扩展probs形状以匹配采样维度（广播）
        Tensor expandedProbs = probs.expand(extendedShape);
        // 步骤3：按扩展后的形状采样
        Tensor sample = torch.bernoulli(expandedProbs);
        return sample; //torch.bernoulli(probs); 
    }

    // 计算伯努利分布的对数概率
    public Tensor log_prob(Tensor value) {
        
        Tensor isNotZero = torch.ne(value, new Scalar(0.0)); // value != 0.0
        Tensor isNotOne = torch.ne(value, new Scalar(1.0));  // value != 1.0
        Tensor isInvalid = torch.logical_and(isNotZero, isNotOne); // (value !=0) && (value !=1)
        if (torch.any(isInvalid).item().toBool()) {
            // 释放临时张量，避免内存泄漏
            isNotZero.close();
            isNotOne.close();
            isInvalid.close();
            throw new IllegalArgumentException("log_prob的输入value必须是0或1（浮点型为0.0/1.0）！");
        }
        // 步骤1：添加极小值避免log(0)
        Tensor safeProbs = probs.add(new Scalar(EPS)).clamp(new ScalarOptional(new Scalar(0.0)), new ScalarOptional(new Scalar(1.0))); // 限制在[0,1]
        Tensor q = safeProbs.neg().add(new Scalar(1.0f)); // 1 - p
        q = q.add(new Scalar(EPS)).clamp(new ScalarOptional(new Scalar(0.0)), new ScalarOptional(new Scalar(1.0))); // 同样添加极小值

        // 步骤2：计算对数概率：v*log(p) + (1-v)*log(1-p)
        Tensor logP = log(safeProbs);
        Tensor logQ = log(q);
        Tensor term1 = value.mul(logP);
        Tensor term2 = value.neg().add(new Scalar(1.0f)).mul(logQ);
        Tensor logProb = term1.add(term2);
        return logProb;
//        return v.mul(log(probs)).add(v.neg().add(new Scalar(1)).mul(log(probs.neg().add(new Scalar(1)))));
    }

    // 计算伯努利分布的熵
    public Tensor entropy() {
        Tensor safeProbs = probs.add(new Scalar(EPS)).clamp(new ScalarOptional(new Scalar(0.0)), new ScalarOptional(new Scalar(1.0)));
        Tensor q = safeProbs.neg().add(new Scalar(new Scalar(1.0f))).add(new Scalar(EPS)).clamp(new ScalarOptional(new Scalar(0.0)), new ScalarOptional(new Scalar(1.0)));

        // 步骤2：计算熵：- [p*log(p) + (1-p)*log(1-p)]
        Tensor term1 = safeProbs.mul(log(safeProbs));
        Tensor term2 = q.mul(log(q));
        Tensor entropy = term1.add(term2).neg();

        return entropy;

    }
    
    @Override
    public Tensor mean() { 
        return probs; 
    }

    @Override
    public void close() {
        
    }
}
