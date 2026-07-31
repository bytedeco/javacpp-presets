/*
 * Ported from torch-rechub-scala: torchrec/basic/layers/LossFunc.scala
 *
 * RegularizationLoss, HingeLoss, NCELoss, InBatchNCELoss, BPRLoss.
 */
package org.bytedeco.pytorch.recommend.basic.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.BatchNorm1dImpl;
import org.bytedeco.pytorch.nn.modules.BatchNorm2dImpl;
import org.bytedeco.pytorch.nn.modules.BatchNorm3dImpl;
import org.bytedeco.pytorch.nn.modules.EmbeddingBagImpl;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.GroupNormImpl;
import org.bytedeco.pytorch.nn.modules.InstanceNorm1dImpl;
import org.bytedeco.pytorch.nn.modules.InstanceNorm2dImpl;
import org.bytedeco.pytorch.nn.modules.InstanceNorm3dImpl;
import org.bytedeco.pytorch.nn.modules.LayerNormImpl;

import java.util.HashSet;
import java.util.Set;

/**
 * Unified L1/L2 regularization for embedding and dense parameters.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class RegularizationLoss extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final float embeddingL1;
    private final float embeddingL2;
    private final float denseL1;
    private final float denseL2;

    public RegularizationLoss() {
        this(0.0f, 0.0f, 0.0f, 0.0f);
    }

    public RegularizationLoss(float embeddingL1, float embeddingL2, float denseL1, float denseL2) {
        super("RegularizationLoss");
        this.embeddingL1 = embeddingL1;
        this.embeddingL2 = embeddingL2;
        this.denseL1 = denseL1;
        this.denseL2 = denseL2;
    }

    public Tensor apply(Module model) {
        float regLoss = 0.0f;

        Class<?>[] normModules = {
                BatchNorm1dImpl.class, BatchNorm2dImpl.class, BatchNorm3dImpl.class,
                LayerNormImpl.class, GroupNormImpl.class,
                InstanceNorm1dImpl.class, InstanceNorm2dImpl.class, InstanceNorm3dImpl.class
        };
        Class<?>[] embedModules = {EmbeddingImpl.class, EmbeddingBagImpl.class};

        Set<Integer> normParamIds = new HashSet<>();
        Set<Integer> embeddingParamIds = new HashSet<>();

        var modBegin = model.modules().begin();
        var modEnd = model.modules().end();
        while (!modBegin.equals(modEnd)) {
            Module module = modBegin.get();
            if (module != null && !module.isNull()) {
                Class<?> moduleClass = module.getClass();
                boolean isNorm = false;
                for (Class<?> c : normModules) {
                    if (c.isAssignableFrom(moduleClass)) {
                        isNorm = true;
                        break;
                    }
                }
                boolean isEmbed = false;
                for (Class<?> c : embedModules) {
                    if (c.isAssignableFrom(moduleClass)) {
                        isEmbed = true;
                        break;
                    }
                }
                if (isNorm || isEmbed) {
                    var paramBegin = module.parameters().begin();
                    var paramEnd = module.parameters().end();
                    while (!paramBegin.equals(paramEnd)) {
                        Tensor p = paramBegin.get();
                        if (p != null && !p.isNull()) {
                            int id = System.identityHashCode(p);
                            if (isNorm) normParamIds.add(id);
                            if (isEmbed) embeddingParamIds.add(id);
                        }
                        paramBegin = paramBegin.increment();
                    }
                }
            }
            modBegin = modBegin.increment();
        }

        var paramBegin = model.parameters().begin();
        var paramEnd = model.parameters().end();
        while (!paramBegin.equals(paramEnd)) {
            Tensor param = paramBegin.get();
            if (param != null && !param.isNull() && param.requires_grad()) {
                int paramId = System.identityHashCode(param);
                if (!normParamIds.contains(paramId)) {
                    if (embeddingParamIds.contains(paramId)) {
                        if (embeddingL1 > 0.0f) {
                            regLoss += embeddingL1 * torch.sum(torch.abs(param)).item().toFloat();
                        }
                        if (embeddingL2 > 0.0f) {
                            regLoss += embeddingL2 * torch.sum(param.pow(new Scalar(2.0f))).item().toFloat();
                        }
                    } else {
                        if (denseL1 > 0.0f) {
                            regLoss += denseL1 * torch.sum(torch.abs(param)).item().toFloat();
                        }
                        if (denseL2 > 0.0f) {
                            regLoss += denseL2 * torch.sum(param.pow(new Scalar(2.0f))).item().toFloat();
                        }
                    }
                }
            }
            paramBegin = paramBegin.increment();
        }

        return torch.tensor(new float[]{regLoss});
    }
}
