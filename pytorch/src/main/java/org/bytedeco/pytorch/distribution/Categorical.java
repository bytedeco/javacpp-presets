package org.bytedeco.pytorch.distribution;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import static org.bytedeco.pytorch.global.torch.*;

public class Categorical extends Distribution implements AutoCloseable {
    private final Tensor probs;       // 归一化后的概率 [Batch..., ActionDim]
    private final int numCategories;  // 类别数（ActionDim）

    // 预定义常量
    private static final ScalarTypeOptional SCALAR_TYPE_OPT = new ScalarTypeOptional();
    private static final GeneratorOptional GENERATOR_OPT = new GeneratorOptional();

    // 构造函数：校验+归一化概率
    public Categorical(Tensor probs) {
        // 步骤1：校验输入合法性（probs≥0）
        Tensor probsLt0 = lt(probs, tensor(0.0f));
        if (any(probsLt0).item().toBool()) {
            probsLt0.close();
            throw new IllegalArgumentException("分类分布probs必须≥0！");
        }
        probsLt0.close();

        // 步骤2：数值稳定性处理（避免sum=0）
        Tensor epsTensor = tensor(1e-8, probs.options());
        Tensor safeProbs = add(probs, epsTensor);

        // 步骤3：归一化概率（按最后一维求和，保持维度）
        Tensor sumProbs = sum(safeProbs, new long[]{-1}, true, SCALAR_TYPE_OPT);
        Tensor sumProbsSafe = where(
                eq(sumProbs, tensor(0.0f)),
                ones_like(sumProbs),
                sumProbs
        );
        this.probs = div(safeProbs, sumProbsSafe);

        // 步骤4：记录类别数
        this.numCategories = (int) probs.size(-1);

        // 释放临时张量
        epsTensor.close();
        safeProbs.close();
        sumProbs.close();
        sumProbsSafe.close();
    }

    @Override
    public String name() {
        return "Categorical";
    }

    public Tensor getProbs() {
        return this.probs;
    }

    public int numCategories() {
        return this.numCategories;
    }

    @Override
    public Tensor sample(long... sampleShape) {
        // 步骤1：扩展probs到批量形状 [SampleShape, Batch, ActionDim]
        long[] extendedShape = getExtendedShape(probs, sampleShape);
        Tensor expandedProbs = probs.expand(extendedShape);

        // 步骤2：分类分布采样
        Tensor samples = multinomial(
                expandedProbs.reshape(new long[]{-1, numCategories}),
                1,
                false,
                GENERATOR_OPT
        );

        // 步骤3：恢复形状 [SampleShape, Batch]
        long[] sampleTargetShape = new long[extendedShape.length - 1];
        System.arraycopy(extendedShape, 0, sampleTargetShape, 0, extendedShape.length - 1);
        Tensor result = samples.reshape(sampleTargetShape).to(kLong());

        // 释放临时张量
        expandedProbs.close();
        samples.close();

        return result;
    }

    @Override
    public Tensor log_prob(Tensor actions) {
        // actions: integer class indices, shape = sample_shape + batch_shape
        Tensor index = actions.to(kLong()).clone();
        Tensor logProbs = log(add(probs, tensor(1e-8, probs.options())));

        // gather along last dim → unsqueeze index to [..., 1]
        Tensor indexUnsq = index.unsqueeze(-1);

        // Expand logProbs to sample_shape + batch_shape + [K]
        long[] indexShape = index.sizes().vec().get();
        long[] target = new long[indexShape.length + 1];
        System.arraycopy(indexShape, 0, target, 0, indexShape.length);
        target[indexShape.length] = numCategories;

        Tensor expandedLogProbs = logProbs.expand(target);

        // Range check 0 ≤ index < K
        Tensor indexLt0 = lt(indexUnsq, tensor(0L, indexUnsq.options()));
        Tensor indexGeK = ge(indexUnsq, tensor((long) numCategories, indexUnsq.options()));
        Tensor invalid = logical_or(indexLt0, indexGeK);
        if (any(invalid).item().toBool()) {
            indexLt0.close();
            indexGeK.close();
            invalid.close();
            throw new IllegalArgumentException("actions索引必须满足0≤index<" + numCategories + "！");
        }
        indexLt0.close();
        indexGeK.close();
        invalid.close();

        Tensor gatherResult = gather(expandedLogProbs, -1, indexUnsq);
        Tensor result = gatherResult.squeeze(-1);

        index.close();
        indexUnsq.close();
        logProbs.close();
        expandedLogProbs.close();
        gatherResult.close();

        return result;
    }

    @Override
    public Tensor entropy() {
        Tensor epsTensor = tensor(1e-8, probs.options());
        Tensor safeProbs = add(probs, epsTensor);
        Tensor logProbs = log(safeProbs);

        Tensor entropyTerms = mul(probs, logProbs);
        Tensor entropy = neg(sum(entropyTerms, -1));

        epsTensor.close();
        safeProbs.close();
        logProbs.close();
        entropyTerms.close();

        return entropy;
    }

    @Override
    public Tensor mean() {
        // 生成类别索引 [0,1,...,k-1]
        long[] shape = new long[(int) probs.dim()];
        java.util.Arrays.fill(shape, 1);
        shape[(int) probs.dim() - 1] = numCategories;

        Tensor indices = arange(
                new Scalar(0),
                new Scalar(numCategories),
                new Scalar(1),
                probs.options().dtype(new ScalarTypeOptional(kFloat()))
        ).reshape(shape);

        Tensor mean = sum(mul(indices, probs), -1);

        indices.close();
        return mean;
    }

    protected long[] getExtendedShape(Tensor baseTensor, long... sampleShape) {
        long[] baseShape = baseTensor.sizes().vec().get();
        long[] extended = new long[sampleShape.length + baseShape.length];
        System.arraycopy(sampleShape, 0, extended, 0, sampleShape.length);
        System.arraycopy(baseShape, 0, extended, sampleShape.length, baseShape.length);
        return extended;
    }

    // 辅助函数：数组转字符串（增强错误提示）
    private String arrayToString(long[] arr) {
        if (arr == null || arr.length == 0) return "[]";
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < arr.length; i++) {
            sb.append(arr[i]);
            if (i < arr.length - 1) sb.append(", ");
        }
        sb.append("]");
        return sb.toString();
    }

    @Override
    public void close() {
        if (this.probs != null && !this.probs.isNull()) {
            this.probs.close();
        }
    }
}
