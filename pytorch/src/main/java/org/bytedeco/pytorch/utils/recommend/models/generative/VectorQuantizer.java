/*
 * Ported from torch-rechub-scala: torchrec/models/generative/RQVAE.scala (VectorQuantizer)
 *
 * Single-stage vector quantization with optional Sinkhorn soft assignment.
 * Returns (quantized, loss, indices).
 */
package org.bytedeco.pytorch.utils.recommend.models.generative;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class VectorQuantizer extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    /** Result of vector quantization: quantized tensor, loss, indices. */
    public static final class Result {
        public final Tensor quantized;
        public final Tensor loss;
        public final Tensor indices;

        public Result(Tensor quantized, Tensor loss, Tensor indices) {
            this.quantized = quantized;
            this.loss = loss;
            this.indices = indices;
        }
    }

    private final int nE;
    private final int eDim;
    private final float beta;
    private final float skEpsilon;
    private final int skIters;
    private final EmbeddingImpl embedding;
    private boolean initted = true;

    public VectorQuantizer(int nE, int eDim) {
        this(nE, eDim, 0.25f, 0.003f, 100, DeviceSupport.backend());
    }

    public VectorQuantizer(
            int nE,
            int eDim,
            float beta,
            float skEpsilon,
            int skIters,
            String device) {
        super("VectorQuantizer");
        this.nE = nE;
        this.eDim = eDim;
        this.beta = beta;
        this.skEpsilon = skEpsilon;
        this.skIters = skIters;

        this.embedding = new EmbeddingImpl(new EmbeddingOptions(nE, eDim));
        register_module("embedding", embedding);

        float bound = 1.0f / nE;
        torch.uniform_(embedding.weight(), -bound, bound);
    }

    public Tensor getCodebook() {
        return embedding.weight();
    }

    public Tensor getCodebookEntry(Tensor indices, long[] shape) {
        Tensor zQ = embedding.forward(indices.toType(ScalarType.Long));
        if (shape != null && shape.length > 1) {
            if (shape.length == 2) {
                zQ = zQ.view(shape[0], shape[1]);
            } else if (shape.length == 3) {
                zQ = zQ.view(shape[0], shape[1], shape[2]);
            } else if (shape.length == 4) {
                zQ = zQ.view(shape[0], shape[1], shape[2], shape[3]);
            }
        }
        return zQ;
    }

    public Tensor getCodebookEntry(Tensor indices) {
        return getCodebookEntry(indices, null);
    }

    public void initEmb(Tensor data) {
        embedding.weight().zero_();
        initted = true;
    }

    private Tensor centerDistanceForConstraint(Tensor distances) {
        Tensor maxDistance = distances.max();
        Tensor minDistance = distances.min();
        Tensor middle = maxDistance.add(minDistance).mul(new Scalar(0.5f));
        Tensor amplitude = maxDistance.sub(middle).add(new Scalar(1e-5));
        return distances.sub(middle).div(amplitude);
    }

    private Tensor sinkhornAlgorithm(Tensor distances) {
        Tensor Q = torch.exp(distances.neg().div(new Scalar((double) skEpsilon)));
        int B = (int) Q.size(0);
        int K = (int) Q.size(1);

        Tensor sumQ = Q.sum(1L).sum(0L);
        Q = Q.div(sumQ);

        for (int i = 0; i < skIters; i++) {
            Q = Q.div(Q.sum(1L));
            Q = Q.div(new Scalar((double) B));
            Q = Q.div(Q.sum(0L));
            Q = Q.div(new Scalar((double) K));
        }
        Q = Q.mul(new Scalar((double) B));
        return Q;
    }

    public Result forward(Tensor x, boolean useSk) {
        long batchSize = x.size(0);
        long seqLen;
        Tensor flat;
        if (x.dim() == 3L) {
            seqLen = x.size(1);
            flat = x.view(-1, eDim);
        } else if (x.dim() == 2L) {
            seqLen = 1L;
            flat = x;
        } else {
            seqLen = 1L;
            flat = x.view(-1, eDim);
        }

        if (!initted && is_training()) {
            initEmb(flat);
        }

        Scalar twoScalar = new Scalar(2.0);
        Tensor latentSq = torch.pow(flat, twoScalar).sum(1).unsqueeze(1);
        Tensor codebookSq = torch.pow(embedding.weight(), twoScalar).sum(1).unsqueeze(0);
        Tensor crossTerm = torch.matmul(flat, embedding.weight().t()).mul(new Scalar(-2.0));
        Tensor d = latentSq.add(codebookSq).add(crossTerm);

        Tensor indices;
        if (!useSk || skEpsilon <= 0) {
            indices = d.argmin(new LongOptional(1L), false);
        } else {
            d = centerDistanceForConstraint(d.toType(ScalarType.Double));
            Tensor Q = sinkhornAlgorithm(d.toType(ScalarType.Double));
            if (Q.isnan().any().item().toFloat() != 0.0f || Q.isinf().any().item().toFloat() != 0.0f) {
                System.out.println("Sinkhorn returns nan/inf, falling back to hard assignment");
                indices = d.toType(ScalarType.Float).argmin(new LongOptional(1L), false);
            } else {
                indices = Q.toType(ScalarType.Float).argmax(new LongOptional(1L), false);
            }
        }

        Tensor xQ = embedding.forward(indices.toType(ScalarType.Long));
        Tensor xQReshaped;
        if (x.dim() == 3L) {
            xQReshaped = xQ.view(batchSize, seqLen, (long) eDim);
        } else if (x.dim() == 2L) {
            xQReshaped = xQ.view(batchSize, (long) eDim);
        } else {
            xQReshaped = xQ;
        }

        Tensor commitmentLoss = torch.mse_loss(xQReshaped.detach(), x);
        Tensor codebookLoss = torch.mse_loss(xQReshaped, x.detach());
        Tensor loss = codebookLoss.add(commitmentLoss.mul(new Scalar((double) beta)));

        // Straight-through estimator
        Tensor xQSt = x.add(xQReshaped.sub(x).detach());

        Tensor indicesFinal = x.dim() == 3L
                ? indices.view(batchSize, seqLen)
                : indices.view(batchSize, 1L);

        return new Result(xQSt, loss.mean(), indicesFinal);
    }

    public Result quantize(Tensor x) {
        return forward(x, true);
    }
}
