package org.bytedeco.pytorch.geometric.nn.model;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;

public class ARGA extends GAE {
    private GenericModule discriminator;

    public ARGA(GenericModule encoder, GenericModule discriminator) {
        super(encoder);
        this.discriminator = discriminator;
        register_module("discriminator", this.discriminator);
    }

    public Tensor discriminator_loss(Tensor z) {
        // 训练判别器：真样本 vs 假样本(Z)
        Tensor real = torch.randn_like(z);
        Tensor dReal = discriminator.forward(real);
        Tensor lossReal = torch.binary_cross_entropy_with_logits(dReal, torch.ones_like(dReal)).mean();

        Tensor dFake = discriminator.forward(z.detach());
        Tensor lossFake = torch.binary_cross_entropy_with_logits(dFake, torch.zeros_like(dFake)).mean();

        return lossReal.add(lossFake);
    }

    public Tensor reg_loss(Tensor z) {
        // 训练 Encoder 欺骗 D
        Tensor dFake = discriminator.forward(z);
        return torch.binary_cross_entropy_with_logits(dFake, torch.ones_like(dFake)).mean();
    }
}

//public class ARGA extends GAE {
//    private org.bytedeco.pytorch.nn.Module discriminator;
//
//    public ARGA(org.bytedeco.pytorch.nn.Module encoder, org.bytedeco.pytorch.nn.Module discriminator) {
//        super(encoder);
//        this.discriminator = discriminator;
//        register_module("discriminator", discriminator);
//    }
//
//    /**
//     * 对应 Python 的 discriminator_loss
//     * 训练判别器 D
//     */
//    public Tensor discriminator_loss(Tensor z) {
//        // 1. 真实分布样本 (Prior: 标准正态分布)
//        Tensor real = torch.randn_like(z);
//        Tensor dReal = discriminator.as(GenericModule.class).forward(real);
//        // 使用 BCEWithLogitsLoss 逻辑
//        Tensor lossReal = torch.binary_cross_entropy_with_logits(dReal, torch.ones_like(dReal)).mean();
//
//        // 2. 伪造样本 (Encoder 生成的 Z)
//        // 注意必须 detach()，训练 D 时不更新 Encoder
//        Tensor dFake = discriminator.as(GenericModule.class).forward(z.detach());
//        Tensor lossFake = torch.binary_cross_entropy_with_logits(dFake, torch.zeros_like(dFake)).mean();
//
//        return lossReal.add(lossFake);
//    }
//
//    /**
//     * 对应 Python 的 reg_loss
//     * 训练编码器 Encoder，让其生成的 Z 欺骗判别器
//     */
//    public Tensor reg_loss(Tensor z) {
//        Tensor dFake = discriminator.as(GenericModule.class).forward(z);
//        // 目标是让 Z 看起来像 1 (Real)
//        return torch.binary_cross_entropy_with_logits(dFake, torch.ones_like(dFake)).mean();
//    }
//}