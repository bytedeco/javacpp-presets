/*
 * Ported from torch-rechub-scala: torchrec/models/matching/MIND.scala
 * (matching-local CapsuleNetwork — different from basic.layers.CapsuleNetwork)
 *
 * Simple attention-based multi-interest capsule projection.
 */
package org.bytedeco.pytorch.recommend.models.matching;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.recommend.DeviceSupport;

/**
 * Capsule Network for Multi-Interest Learning (MIND matching package).
 * Named MindCapsuleNetwork to avoid clash with basic.layers.CapsuleNetwork.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class MindCapsuleNetwork extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int numInterests;
    private final int capsuleDim;
    private final String device;
    private final LinearImpl W;

    public MindCapsuleNetwork(int embedDim, int numInterests, int capsuleDim) {
        this(embedDim, numInterests, capsuleDim, 3, DeviceSupport.backend());
    }

    public MindCapsuleNetwork(int embedDim, int numInterests, int capsuleDim,
                              int numRoutings, String device) {
        super("CapsuleNetwork");
        // numRoutings kept for API parity (Scala ctor has it but body uses attention pooling).
        this.numInterests = numInterests;
        this.capsuleDim = capsuleDim;
        this.device = device;

        this.W = new LinearImpl(embedDim, (long) numInterests * capsuleDim);
        W.to(new Device(device), false);
        register_module("W", W);
    }

    @Override
    public Tensor forward(Tensor x) {
        // x: (batch, seq_len, embed_dim)
        Device dev = new Device(device);
        Tensor xOn;
        try {
            xOn = x.to(dev, ScalarType.Float);
        } catch (Throwable t) {
            xOn = x;
        }
        int batchSize = (int) xOn.size(0);
        int seqLen = (int) xOn.size(1);

        Tensor uRaw = W.forward(xOn);

        Tensor u;
        try {
            long totalElements = uRaw.numel();
            long denom = (long) batchSize * seqLen * numInterests;
            if (denom == 0) {
                throw new RuntimeException("Invalid shape components: batchSize=" + batchSize
                        + " seqLen=" + seqLen + " numInterests=" + numInterests);
            }
            if (totalElements % denom != 0) {
                throw new RuntimeException("Capsule reshape failed: totalElements=" + totalElements
                        + " not divisible by " + denom);
            }
            long derivedCapsuleDim = totalElements / denom;
            u = uRaw.view(batchSize, seqLen, numInterests, derivedCapsuleDim);
        } catch (Throwable e) {
            // Fallback: pool then project
            Tensor pooled = xOn.mean(1);
            Tensor fallback = W.forward(pooled);
            int outDim = (int) fallback.size(1);
            int actualCapsuleDim = (outDim % numInterests == 0) ? outDim / numInterests : capsuleDim;
            return fallback.view(batchSize, numInterests, actualCapsuleDim);
        }

        Tensor scores = u.mean(3); // (batch, seq_len, num_interests)
        Tensor attnWeights = scores.softmax(1).unsqueeze(3); // (batch, seq_len, num_interests, 1)
        return u.mul(attnWeights).sum(1); // (batch, num_interests, capsule_dim)
    }
}
