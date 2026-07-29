/*
 * Ported from torch-rechub-scala: torchrec/basic/layers/CapsuleNetwork.scala
 *
 * Capsule network for multi-interest (MIND/Comirec).
 * Input: itemEb (B, L, D), mask (B, L, 1) → Output: (B, interest_num, D)
 */
package org.bytedeco.pytorch.utils.recommend.basic.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.BoolOptional;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.DeviceOptional;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.ReLUImpl;
import org.bytedeco.pytorch.nn.modules.container.SequentialImpl;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class CapsuleNetwork extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int embeddingDim;
    private final int seqLen;
    private final int bilinearType;
    private final int interestNum;
    private final int routingTimes;
    private final boolean reluLayer;
    private final Device dev;
    private final TensorOptions tensorOptsDev;

    private final SequentialImpl relu;
    private final LinearImpl linear;
    private final Tensor w;

    public CapsuleNetwork(int embeddingDim, int seqLen) {
        this(embeddingDim, seqLen, 2, 4, 3, false, DeviceSupport.backend());
    }

    public CapsuleNetwork(int embeddingDim, int seqLen, int bilinearType, int interestNum,
                          int routingTimes, boolean reluLayer, String device) {
        super("CapsuleNetwork");
        this.embeddingDim = embeddingDim;
        this.seqLen = seqLen;
        this.bilinearType = bilinearType;
        this.interestNum = interestNum;
        this.routingTimes = routingTimes;
        this.reluLayer = reluLayer;
        this.dev = DeviceSupport.deviceOf(device);

        TensorOptions tensorOptsCPU = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float));
        this.tensorOptsDev = tensorOptsCPU.device(new DeviceOptional(dev));

        if (reluLayer) {
            SequentialImpl seq = new SequentialImpl();
            LinearImpl lin = new LinearImpl(embeddingDim, embeddingDim);
            seq.push_back("linear", lin);
            seq.push_back("relu", new ReLUImpl());
            register_module("relu", seq);
            this.relu = seq;
        } else {
            this.relu = null;
        }

        switch (bilinearType) {
            case 0: { // MIND
                LinearImpl l = new LinearImpl(embeddingDim, embeddingDim);
                register_module("linear", l);
                this.linear = l;
                this.w = null;
                break;
            }
            case 1: {
                LinearImpl l = new LinearImpl(embeddingDim, embeddingDim * interestNum);
                register_module("linear", l);
                this.linear = l;
                this.w = null;
                break;
            }
            default: { // ComirecDR
                Tensor wt = torch.rand(
                        new long[]{1L, seqLen, (long) interestNum * embeddingDim, embeddingDim},
                        tensorOptsDev);
                register_parameter("w", wt);
                this.w = wt;
                this.linear = null;
                break;
            }
        }
    }

    public Tensor forward(Tensor itemEb, Tensor mask) {
        long batchSize = itemEb.size(0);
        if (!itemEb.device().equals(dev)) {
            throw new IllegalArgumentException(
                    "itemEb device mismatch, layer=" + dev + ", input=" + itemEb.device());
        }

        // Step 1: Bilinear transform
        Tensor itemEbHat;
        switch (bilinearType) {
            case 0: {
                Tensor out = linear.forward(itemEb);
                itemEbHat = out.repeat(new long[]{1L, 1L, interestNum});
                break;
            }
            case 1:
                itemEbHat = linear.forward(itemEb);
                break;
            default: {
                Tensor u = itemEb.unsqueeze(2);
                itemEbHat = torch.sum(w.narrow(1, 0, seqLen).mul(u), 3);
                break;
            }
        }

        // Reshape & transpose to [B, K, L, D]
        Tensor reshaped = itemEbHat.reshape(batchSize, seqLen, interestNum, embeddingDim);
        Tensor transposed = reshaped.transpose(1, 2).contiguous();
        itemEbHat = transposed.reshape(batchSize, interestNum, seqLen, embeddingDim);

        Tensor itemEbHatIter = itemEbHat.detach();

        // Initialize capsuleWeight: [B, K, L]
        Tensor capsuleWeight;
        TensorOptions noGradOpts = tensorOptsDev.requires_grad(new BoolOptional(false));
        if (bilinearType > 0) {
            capsuleWeight = torch.zeros(new long[]{batchSize, interestNum, seqLen}, noGradOpts);
        } else {
            capsuleWeight = torch.randn(new long[]{batchSize, interestNum, seqLen}, noGradOpts);
        }

        Tensor interestCapsule = torch.empty();

        for (int i = 0; i < routingTimes; i++) {
            Tensor attenMask = mask.unsqueeze(1).repeat(new long[]{1L, interestNum, 1L, 1L});
            Tensor paddings = torch.zeros_like(attenMask);

            Tensor capsuleSoftmaxWeight = torch.softmax(capsuleWeight, -1);

            Tensor capsuleSoftmaxWeight4d = capsuleSoftmaxWeight.unsqueeze(-1);
            Tensor maskedWeight4d = torch.where(
                    torch.eq(attenMask, new Scalar(0f)), paddings, capsuleSoftmaxWeight4d);

            Tensor unsqueezedWeight = maskedWeight4d.squeeze(-1).unsqueeze(2);

            Tensor targetItemEbHat = (i < routingTimes - 1) ? itemEbHatIter : itemEbHat;
            interestCapsule = torch.matmul(unsqueezedWeight, targetItemEbHat);

            Tensor rawNorm = torch.sum(torch.square(interestCapsule), -1);
            Tensor capNorm = rawNorm.unsqueeze(-1);

            Tensor scalarFactor = capNorm.div(capNorm.add(new Scalar(1f)))
                    .div(torch.sqrt(capNorm.add(new Scalar(1e-9f))));
            interestCapsule = scalarFactor.mul(interestCapsule);

            if (i < routingTimes - 1) {
                Tensor capT = interestCapsule.transpose(2, 3).contiguous();
                Tensor deltaWeight = torch.matmul(itemEbHatIter, capT);
                capsuleWeight = capsuleWeight.add(deltaWeight.squeeze(-1));
            }
        }

        Tensor result = interestCapsule.reshape(batchSize, interestNum, embeddingDim);
        if (reluLayer && relu != null) {
            return relu.forward(result);
        }
        return result;
    }
}
