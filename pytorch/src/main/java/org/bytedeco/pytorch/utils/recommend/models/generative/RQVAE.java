/*
 * Ported from torch-rechub-scala: torchrec/models/generative/RQVAE.scala
 *
 * RQVAE: Residual Quantized Variational Autoencoder.
 * encode → multi-stage residual VQ → decode.
 */
package org.bytedeco.pytorch.utils.recommend.models.generative;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.ReLUImpl;
import org.bytedeco.pytorch.nn.modules.container.SequentialImpl;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class RQVAE extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final float quantLossWeight;
    private final SequentialImpl encoder;
    private final ResidualVectorQuantizer rq;
    private final SequentialImpl decoder;

    /** Simplified constructor: embedDim, numCodebooks, codebookSize, latentDim, device. */
    public RQVAE(int embedDim, int numCodebooks, int codebookSize, int latentDim, String device) {
        this(embedDim, fillCodebookSizes(numCodebooks, codebookSize), latentDim,
                new long[]{256L, 128L}, new long[]{128L, 256L}, 0.0f, 1.0f, 0.25f, 0.003f, 100, device);
    }

    public RQVAE(int embedDim, int numCodebooks, int codebookSize, int latentDim) {
        this(embedDim, numCodebooks, codebookSize, latentDim, DeviceSupport.backend());
    }

    public RQVAE(
            int inDim,
            int[] numEmbList,
            int eDim,
            long[] encoderDims,
            long[] decoderDims,
            float dropout,
            float quantLossWeight,
            float beta,
            float skEpsilon,
            int skIters,
            String device) {
        super("RQVAE");
        this.quantLossWeight = quantLossWeight;
        Device targetDevice = new Device(device);
        int numQuantizers = numEmbList.length;

        // Encoder Sequential
        SequentialImpl enc = new SequentialImpl();
        long prevDim = inDim;
        for (long dim : encoderDims) {
            enc.push_back(new LinearImpl(prevDim, dim));
            enc.push_back(new ReLUImpl());
            if (dropout > 0) {
                enc.push_back(new DropoutImpl(dropout));
            }
            prevDim = dim;
        }
        enc.push_back(new LinearImpl(prevDim, eDim));
        this.encoder = enc;
        register_module("encoder", encoder);

        float[] skEpsList = new float[numQuantizers];
        for (int i = 0; i < numQuantizers; i++) {
            skEpsList[i] = skEpsilon;
        }
        this.rq = new ResidualVectorQuantizer(numEmbList, eDim, beta, skEpsList, skIters, device);
        register_module("rq", rq);

        // Decoder Sequential
        SequentialImpl dec = new SequentialImpl();
        prevDim = eDim;
        for (long dim : decoderDims) {
            dec.push_back(new LinearImpl(prevDim, dim));
            dec.push_back(new ReLUImpl());
            if (dropout > 0) {
                dec.push_back(new DropoutImpl(dropout));
            }
            prevDim = dim;
        }
        dec.push_back(new LinearImpl(prevDim, inDim));
        this.decoder = dec;
        register_module("decoder", decoder);

        if (!"cpu".equals(device)) {
            encoder.to(targetDevice, false);
            rq.to(targetDevice, false);
            decoder.to(targetDevice, false);
            this.to(targetDevice, false);
        }
    }

    private static int[] fillCodebookSizes(int numCodebooks, int codebookSize) {
        int[] arr = new int[numCodebooks];
        for (int i = 0; i < numCodebooks; i++) {
            arr[i] = codebookSize;
        }
        return arr;
    }

    /** Forward: encode → quantize → decode. Returns (decoded, rq_loss, indices). */
    public VectorQuantizer.Result forward(Tensor x, boolean useSk) {
        Tensor encoded = encoder.forward(x);
        VectorQuantizer.Result r = rq.forward(encoded, useSk);
        Tensor decoded = decoder.forward(r.quantized);
        return new VectorQuantizer.Result(decoded, r.loss, r.indices);
    }

    public VectorQuantizer.Result quantize(Tensor x) {
        return forward(x, true);
    }

    /** Returns (totalLoss, reconLoss). */
    public Tensor[] computeLoss(Tensor out, Tensor quantLoss, Tensor target, String lossType) {
        Tensor reconLoss;
        if ("mse".equals(lossType)) {
            reconLoss = torch.mse_loss(out, target).mean();
        } else if ("l1".equals(lossType)) {
            reconLoss = torch.l1_loss(out, target).mean();
        } else {
            throw new IllegalArgumentException("Unknown loss type: " + lossType);
        }
        Tensor totalLoss = reconLoss.add(quantLoss.mul(new Scalar((double) quantLossWeight)));
        return new Tensor[]{totalLoss, reconLoss};
    }

    public Tensor[] computeLoss(Tensor out, Tensor quantLoss, Tensor target) {
        return computeLoss(out, quantLoss, target, "mse");
    }

    public Tensor getIndices(Tensor x, boolean useSk) {
        Tensor encoded = encoder.forward(x);
        return rq.forward(encoded, useSk).indices;
    }

    public Tensor getIndices(Tensor x) {
        return getIndices(x, false);
    }
}
