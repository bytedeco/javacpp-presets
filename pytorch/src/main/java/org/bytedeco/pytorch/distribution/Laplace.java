package org.bytedeco.pytorch.distribution;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import static org.bytedeco.pytorch.global.torch.*;

public class Laplace extends Distribution implements AutoCloseable {
    private final Tensor loc;   // 位置参数μ
    private final Tensor scale; // 尺度参数b（必须>0）

    // 预定义标量（复用避免重复创建，提升性能）
    private static final Scalar SCALAR_0 = new Scalar(0.0);
    private static final Scalar SCALAR_0_5 = new Scalar(0.5);
    private static final Scalar SCALAR_1 = new Scalar(1.0);
    private static final Scalar SCALAR_2 = new Scalar(2.0);
    private static final Scalar SCALAR_E = new Scalar(Math.E);
    private static final Scalar SCALAR_EPS = new Scalar(1e-8); // 数值稳定性极小值

    // 构造函数：校验参数合法性 + 深拷贝
    public Laplace(Tensor loc, Tensor scale) {
        // 校验尺度参数scale>0（拉普拉斯分布核心约束）
        Tensor scaleLe0 = torch.lt(scale, torch.tensor(0.0f, scale.options()));
        if (torch.any(scaleLe0).item().toBool()) {
            scaleLe0.close();
            throw new IllegalArgumentException("拉普拉斯分布scale(b)必须大于0！");
        }
        scaleLe0.close();

        // 深拷贝避免外部修改内部状态
        this.loc = loc.clone();
        this.scale = scale.clone();
    }

    @Override
    public String name() {
        return "Laplace";
    }

    @Override
    public Tensor sample(long... sampleShape) {
        // 步骤1：复用父类方法扩展采样形状
        long[] extendedShape = getExtendedShape(loc, sampleShape);
        // 扩展loc/scale到批量形状（保证维度对齐）
        Tensor expandedLoc = loc.expand(extendedShape);
        Tensor expandedScale = scale.expand(extendedShape);

        // 步骤2：生成受限Uniform(ε,1-ε)随机数（避免log(0)）
        Tensor u = torch.rand(extendedShape, loc.options())
                .clamp(new ScalarOptional(new Scalar(1e-6)), new ScalarOptional(new Scalar(1.0 - 1e-6)));

        // 步骤3：拉普拉斯采样公式：μ - b * sgn(U-0.5) * log(1 - 2|U-0.5|)
        Tensor tensor05 = torch.tensor(0.5f, u.options());
        Tensor uMinus05 = torch.sub(u, tensor05); // U - 0.5
        Tensor sgnUMinus05 = torch.sgn(uMinus05); // sgn(U-0.5)

        Tensor absUMinus05 = torch.abs(uMinus05); // |U-0.5|
        Tensor twoAbsUMinus05 = torch.mul(absUMinus05, torch.tensor(2.0f, absUMinus05.options())); // 2|U-0.5|
        Tensor oneMinusTwoAbs = torch.sub(torch.tensor(1.0f, twoAbsUMinus05.options()), twoAbsUMinus05); // 1 - 2|U-0.5|

        // 数值稳定性：避免log(0)
        Tensor oneMinusTwoAbsSafe = torch.clamp(
                oneMinusTwoAbs,
                new ScalarOptional(new Scalar(1e-8)),
                new ScalarOptional(oneMinusTwoAbs.max().item())
        );
        Tensor logTerm = torch.log(oneMinusTwoAbsSafe); // log(1 - 2|U-0.5|)

        // 计算核心项：b * sgn(U-0.5) * log(1 - 2|U-0.5|)
        Tensor coreTerm = torch.mul(expandedScale, torch.mul(sgnUMinus05, logTerm));
        // 最终采样结果：μ - 核心项
        Tensor laplaceSample = torch.sub(expandedLoc, coreTerm);

        // 释放所有临时张量
        expandedLoc.close();
        expandedScale.close();
        u.close();
        tensor05.close();
        uMinus05.close();
        sgnUMinus05.close();
        absUMinus05.close();
        twoAbsUMinus05.close();
        oneMinusTwoAbs.close();
        oneMinusTwoAbsSafe.close();
        logTerm.close();
        coreTerm.close();

        return laplaceSample;
    }

    @Override
    public Tensor log_prob(Tensor v) {
        // 步骤1：扩展loc/scale到v的形状
        Tensor expandedLoc = loc.expand(v.sizes());
        Tensor expandedScale = scale.expand(v.sizes());
        // 数值稳定性：避免scale→0导致除零
        Tensor safeScale = torch.clamp(expandedScale, new ScalarOptional(new Scalar(1e-8)), new ScalarOptional(expandedScale.max().item()));

        // 步骤2：计算对数概率公式：-log(2b) - |v-μ|/b
        // term1 = -log(2b)
        Tensor twoScale = torch.mul(safeScale, torch.tensor(2.0f, safeScale.options())); // 2b
        Tensor logTwoScale = torch.log(twoScale);
        Tensor term1 = torch.neg(logTwoScale); // -log(2b)

        // term2 = -|v-μ|/b
        Tensor vMinusLoc = torch.sub(v, expandedLoc); // v-μ
        Tensor absVMinusLoc = torch.abs(vMinusLoc); // |v-μ|
        Tensor absDivScale = torch.div(absVMinusLoc, safeScale); // |v-μ|/b
        Tensor term2 = torch.neg(absDivScale); // -|v-μ|/b

        // 完整对数概率
        Tensor logProb = torch.add(term1, term2);

        // 释放临时张量
        expandedLoc.close();
        expandedScale.close();
        safeScale.close();
        twoScale.close();
        logTwoScale.close();
        term1.close();
        vMinusLoc.close();
        absVMinusLoc.close();
        absDivScale.close();
        term2.close();

        return logProb;
    }

    @Override
    public Tensor entropy() {
        // 熵公式：H = log(2be)
        Tensor tensor2 = torch.tensor(2.0f, scale.options());
        Tensor tensorE = torch.tensor(Math.E, scale.options());

        // 计算2be
        Tensor twoB = torch.mul(tensor2, scale); // 2b
        Tensor twoBe = torch.mul(twoB, tensorE); // 2be

        // 数值稳定性：避免scale→0导致log(0)
        Tensor twoBeSafe = torch.clamp(twoBe, new ScalarOptional(new Scalar(1e-8)), new ScalarOptional(twoBe.max().item()));
        Tensor entropy = torch.log(twoBeSafe);

        // 释放临时张量
        tensor2.close();
        tensorE.close();
        twoB.close();
        twoBe.close();
        twoBeSafe.close();

        return entropy;
    }

    @Override
    public Tensor mean() {
        // 均值为位置参数μ（返回拷贝避免外部修改）
        Tensor mean = loc.clone();
        return mean;
    }

    // 资源释放：实现AutoCloseable，避免内存泄漏
    @Override
    public void close() {
        loc.close();
        scale.close();
        // 释放预定义Scalar常量

    }
}
