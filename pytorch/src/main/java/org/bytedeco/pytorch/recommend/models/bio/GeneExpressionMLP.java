/*
 * GeneExpressionMLP — tabular multi-omics / gene-expression classifier or regressor.
 *
 * Context (bioinformatics industrial & research practice):
 *   High-dimensional gene expression (RNA-seq counts / TPM, microarray) is a
 *   classic tabular problem. Production and research stacks often use:
 *     - MLP / denoising autoencoder on log-normalized expression (TCGA pan-cancer)
 *     - pathway-guided sparsity (not fully modeled here)
 *     - Cox / survival heads for clinical endpoints
 *
 *   Representative refs:
 *     - Way & Greene, "Extracting a biologically relevant latent space from
 *       cancer transcriptomes with variational autoencoders", PSB 2018
 *     - Deep learning for RNA-seq in drug response (GDSC / CTRP benchmarks)
 *
 * This module:
 *   expression [B, G] -> optional LN -> MLP -> task output
 *   Supports multi-task clinical heads (response, subtype, risk).
 */
package org.bytedeco.pytorch.recommend.models.bio;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.LongVector;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LayerNormImpl;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.basic.layers.MLP;

import java.util.ArrayList;
import java.util.List;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class GeneExpressionMLP extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final LayerNormImpl inputNorm;
    private final MLP encoder;
    private final List<MLP> taskHeads = new ArrayList<>();
    private final int numGenes;
    private final int numTasks;
    private final int repDim;
    private final boolean classification;

    public GeneExpressionMLP(int numGenes, int numTasks) {
        this(numGenes, numTasks, new long[]{512L, 256L, 128L}, true, DeviceSupport.backend());
    }

    /**
     * @param numGenes       input gene / feature dimension G
     * @param numTasks       number of prediction heads
     * @param hiddenDims     encoder MLP dims (last = representation dim)
     * @param classification if true, sigmoid outputs; else raw regression
     */
    public GeneExpressionMLP(int numGenes, int numTasks, long[] hiddenDims,
                             boolean classification, String device) {
        super("GeneExpressionMLP");
        if (numGenes < 1) throw new IllegalArgumentException("numGenes must be >= 1");
        if (numTasks < 1) throw new IllegalArgumentException("numTasks must be >= 1");
        this.numGenes = numGenes;
        this.numTasks = numTasks;
        this.classification = classification;
        this.repDim = (int) hiddenDims[hiddenDims.length - 1];

        LongVector shape = new LongVector(1);
        shape.put(0, numGenes);
        this.inputNorm = new LayerNormImpl(shape);
        register_module("input_norm", inputNorm);

        this.encoder = new MLP(numGenes, hiddenDims, repDim, "relu", 0.2f,
                false, false, true, device);
        register_module("encoder", encoder);

        for (int t = 0; t < numTasks; t++) {
            MLP head = new MLP(repDim, new long[]{64L}, 1L, "relu", 0.1f,
                    false, false, true, device);
            register_module("task_head_" + t, head);
            taskHeads.add(head);
        }
    }

    /**
     * @param expression [B, G] log1p-normalized expression recommended
     * @return [B, numTasks]
     */
    public Tensor forward(Tensor expression) {
        Tensor x = inputNorm.forward(expression);
        Tensor z = encoder.forward(x);
        TensorVector outs = new TensorVector();
        for (MLP head : taskHeads) {
            Tensor o = head.forward(z);
            if (classification) o = o.sigmoid();
            outs.push_back(o);
        }
        return torch.cat(outs, 1L);
    }

    /** Latent representation [B, repDim] for transfer / clustering. */
    public Tensor encode(Tensor expression) {
        return encoder.forward(inputNorm.forward(expression));
    }

    public int numGenes() { return numGenes; }
    public int numTasks() { return numTasks; }
    public int repDim() { return repDim; }
}
