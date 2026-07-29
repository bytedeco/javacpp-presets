/*
 * DnaSeqCnn — 1-D CNN classifier / regressor over DNA / RNA nucleotide sequences.
 *
 * Context (genomics / regulatory genomics):
 *   Classic deep models for DNA motif detection and functional sequence
 *   prediction use multi-filter Conv1d over one-hot or embedded nucleotides:
 *     - DeepBind (Alipanahi et al., Nature Biotech 2015) — binding affinity
 *     - DeepSEA (Zhou & Troyanskaya, Nature Methods 2015) — chromatin effects
 *     - DanQ (Quang & Xie, Nucleic Acids Research 2016) — CNN + BiLSTM
 *
 * This module implements the DeepBind/DeepSEA-style CNN backbone (embed or
 * expect pre-embedded channels) with multi-task heads — suitable as a
 * building block; full DeepSEA multi-thousands-of-epigenomic-targets would
 * use a larger head config.
 *
 * Nucleotide vocab convention:
 *   0=pad, 1=A, 2=C, 3=G, 4=T/U, 5=N (unknown)
 */
package org.bytedeco.pytorch.utils.recommend.models.bio;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.Conv1dImpl;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.ReLUImpl;
import org.bytedeco.pytorch.nn.options.Conv1dOptions;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.basic.layers.MLP;

import java.util.ArrayList;
import java.util.List;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class DnaSeqCnn extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    public static final int DEFAULT_NT_VOCAB = 6; // pad A C G T N

    private final EmbeddingImpl ntEmbedding;
    private final List<Conv1dImpl> convs = new ArrayList<>();
    private final ReLUImpl relu;
    private final DropoutImpl dropout;
    private final MLP head;
    private final int numFilters;
    private final int numTasks;
    private final boolean classification;

    public DnaSeqCnn(int numTasks) {
        this(DEFAULT_NT_VOCAB, 4, 16, new int[]{8, 12, 16, 20}, numTasks,
                true, DeviceSupport.backend());
    }

    /**
     * @param vocabSize   nucleotide vocab (default 6)
     * @param embedDim    embedding dim (DeepBind often uses one-hot=4; embedding is fine)
     * @param numFilters  filters per kernel
     * @param kernelSizes motif-scale kernels (DeepBind uses ~8–24)
     * @param numTasks    output tasks (e.g. TF binding, accessibility)
     */
    public DnaSeqCnn(int vocabSize, int embedDim, int numFilters, int[] kernelSizes,
                     int numTasks, boolean classification, String device) {
        super("DnaSeqCnn");
        this.numFilters = numFilters;
        this.numTasks = Math.max(numTasks, 1);
        this.classification = classification;
        this.relu = new ReLUImpl();
        this.dropout = new DropoutImpl(0.2f);
        int[] kernels = kernelSizes != null ? kernelSizes : new int[]{8, 12, 16};

        EmbeddingOptions opts = new EmbeddingOptions(Math.max(vocabSize, DEFAULT_NT_VOCAB), embedDim);
        opts.padding_idx().put(new LongOptional(0L));
        this.ntEmbedding = new EmbeddingImpl(opts);
        register_module("nt_embedding", ntEmbedding);

        for (int i = 0; i < kernels.length; i++) {
            int k = kernels[i];
            LongPointer kp = new LongPointer(new long[]{k});
            Conv1dOptions copt = new Conv1dOptions(embedDim, numFilters, kp);
            copt.padding().put(new LongPointer(new long[]{Math.max(k / 2, 0)}));
            Conv1dImpl conv = new Conv1dImpl(copt);
            register_module("conv_" + i, conv);
            convs.add(conv);
        }

        long pooledDim = (long) numFilters * kernels.length;
        this.head = new MLP(pooledDim, new long[]{128L, 64L}, this.numTasks, "relu", 0.2f,
                false, false, true, device);
        register_module("head", head);

        if (device != null && !"cpu".equals(device)) {
            Device dev = new Device(device);
            ntEmbedding.to(dev, false);
        }
    }

    /**
     * @param ntIds [B, L] nucleotide ids
     * @return [B, numTasks]
     */
    public Tensor forward(Tensor ntIds) {
        Tensor x = ntEmbedding.forward(ntIds.toType(
                org.bytedeco.pytorch.global.torch.ScalarType.Long)); // [B, L, E]
        Tensor xt = x.transpose(1, 2); // [B, E, L]
        List<Tensor> pooled = new ArrayList<>();
        for (Conv1dImpl conv : convs) {
            Tensor h = dropout.forward(relu.forward(conv.forward(xt)));
            pooled.add(torch.max(h, 2L).get0()); // global max pool (motif presence)
        }
        TensorVector vec = new TensorVector();
        for (Tensor p : pooled) vec.push_back(p);
        Tensor rep = torch.cat(vec, 1L);
        Tensor out = head.forward(rep);
        if (classification) {
            return out.sigmoid();
        }
        return out;
    }

    public int numTasks() {
        return numTasks;
    }
}
