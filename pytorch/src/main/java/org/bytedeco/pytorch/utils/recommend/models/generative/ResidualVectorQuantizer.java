/*
 * Ported from torch-rechub-scala: torchrec/models/generative/RQVAE.scala (ResidualVectorQuantizer)
 *
 * Multi-stage residual vector quantization.
 * Returns (quantized, mean_loss, stacked_indices).
 */
package org.bytedeco.pytorch.utils.recommend.models.generative;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;

import java.util.ArrayList;
import java.util.List;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class ResidualVectorQuantizer extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final List<VectorQuantizer> vqRefs = new ArrayList<>();

    public ResidualVectorQuantizer(int[] nEList, int eDim) {
        this(nEList, eDim, 0.25f, null, 100, DeviceSupport.backend());
    }

    public ResidualVectorQuantizer(
            int[] nEList,
            int eDim,
            float beta,
            float[] skEpsilonList,
            int skIters,
            String device) {
        super("ResidualVectorQuantizer");
        int numQuantizers = nEList.length;
        float[] skEps = skEpsilonList != null ? skEpsilonList : defaultSkEps(numQuantizers);

        for (int i = 0; i < numQuantizers; i++) {
            VectorQuantizer vq = new VectorQuantizer(
                    nEList[i], eDim, beta, skEps[i], skIters, device);
            register_module("vq_" + i, vq);
            vqRefs.add(vq);
        }
    }

    private static float[] defaultSkEps(int n) {
        float[] eps = new float[n];
        for (int i = 0; i < n; i++) {
            eps[i] = 0.003f;
        }
        return eps;
    }

    public Tensor getCodebook() {
        List<Tensor> codebooks = new ArrayList<>();
        for (VectorQuantizer vq : vqRefs) {
            codebooks.add(vq.getCodebook());
        }
        return torch.stack(new TensorVector(codebooks.toArray(new Tensor[0])));
    }

    public VectorQuantizer.Result forward(Tensor x, boolean useSk) {
        List<Tensor> allLosses = new ArrayList<>();
        List<Tensor> allIndices = new ArrayList<>();

        Tensor xQ = torch.zeros_like(x);
        Tensor residual = x;

        for (VectorQuantizer vq : vqRefs) {
            VectorQuantizer.Result r = vq.forward(residual, useSk);
            residual = residual.sub(r.quantized);
            xQ = xQ.add(r.quantized);
            allLosses.add(r.loss);
            allIndices.add(r.indices);
        }

        Tensor meanLosses = torch.stack(new TensorVector(allLosses.toArray(new Tensor[0]))).mean();
        Tensor stackedIndices = torch.stack(new TensorVector(allIndices.toArray(new Tensor[0])), -1L);
        return new VectorQuantizer.Result(xQ, meanLosses, stackedIndices);
    }

    public VectorQuantizer.Result quantize(Tensor x) {
        return forward(x, true);
    }
}
