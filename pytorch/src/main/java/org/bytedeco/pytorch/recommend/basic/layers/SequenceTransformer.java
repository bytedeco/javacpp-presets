/*
 * Ported from torchSa: torchrec/model/SequenceTransformer.scala
 * and WWW2025_MMCTR Python Transformer_DCN.Transformer.
 *
 * Sequence transformer with target conditioning for ranking.
 *
 * Inputs:
 *   targetEmb   : [B, 1, itemInfoDim] or [B, itemInfoDim]
 *   sequenceEmb : [B, S, itemInfoDim]
 *   mask        : [B, S] float, 1 = valid, 0 = padded
 *
 * Steps:
 *   concat [seq, target_expanded] along last dim -> [B, S, 2*itemInfoDim]
 *   key_padding_mask = True where padded (plus fully-padded safeguard)
 *   TransformerEncoder (ReLU activation, PyTorch default)
 *   zero-fill padded positions
 *   take last firstKCols positions, flatten
 *   if concatMaxPool: max-pool over sequence (pad = -1e9), Linear, concat
 */
package org.bytedeco.pytorch.recommend.basic.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.enumtype.TransformerActivation;
import org.bytedeco.pytorch.enumtype.kReLU;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.TransformerEncoderImpl;
import org.bytedeco.pytorch.nn.options.TransformerEncoderLayerOptions;
import org.bytedeco.pytorch.nn.options.TransformerEncoderOptions;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.TensorHelpers;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class SequenceTransformer extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final long inDim;
    private final int firstKCols;
    private final boolean concatMaxPool;
    private final TransformerEncoderImpl encoder;
    private final LinearImpl outProj; // nullable when concatMaxPool=false

    public SequenceTransformer(long inDim) {
        this(inDim, 1L, 256L, 0.2, 2, 16, true, DeviceSupport.backend());
    }

    public SequenceTransformer(
            long inDim,
            long numHeads,
            long dimFeedforward,
            double dropout,
            int numLayers,
            int firstKCols,
            boolean concatMaxPool,
            String device) {
        super("SequenceTransformer");
        this.inDim = inDim;
        this.firstKCols = firstKCols;
        this.concatMaxPool = concatMaxPool;

        // PyTorch default activation is ReLU (not GELU)
        TransformerEncoderLayerOptions layerOpts =
                new TransformerEncoderLayerOptions(inDim, numHeads)
                        .dim_feedforward(dimFeedforward)
                        .dropout(dropout)
                        .activation(new TransformerActivation(new kReLU()));

        TransformerEncoderOptions encoderOpts =
                new TransformerEncoderOptions(layerOpts, numLayers);
        this.encoder = new TransformerEncoderImpl(encoderOpts);
        register_module("encoder", encoder);

        if (concatMaxPool) {
            this.outProj = new LinearImpl(inDim, inDim);
            register_module("out_proj", outProj);
        } else {
            this.outProj = null;
        }

        if (device != null && !"cpu".equals(device)) {
            this.to(new Device(device), false);
        }
    }

    /**
     * @param targetEmb   [B, 1, itemInfoDim] or [B, itemInfoDim]
     * @param sequenceEmb [B, S, itemInfoDim]
     * @param mask        [B, S]  1=valid, 0=padded
     * @return            [B, outputDim]
     */
    public Tensor forward(Tensor targetEmb, Tensor sequenceEmb, Tensor mask) {
        long seqLen = sequenceEmb.size(1);
        long batchSize = sequenceEmb.size(0);

        // Ensure target is [B, 1, D]
        Tensor target3d = targetEmb.dim() == 2 ? targetEmb.unsqueeze(1) : targetEmb;

        // Expand target along sequence: [B, S, D]
        Tensor targetExpanded = target3d.expand(batchSize, seqLen, -1L);

        // Concat along feature dim: [B, S, 2*D] = [B, S, inDim]
        Tensor concatEmb = TensorHelpers.cat(new Tensor[]{sequenceEmb, targetExpanded}, 2);

        // key_padding_mask: True = ignore (padded). mask is 1=valid, 0=padded.
        Tensor padMask = buildKeyPaddingMask(mask); // [B, S] bool

        // JavaCPP TransformerEncoder is seq-first: [S, B, E]
        Tensor src = concatEmb.transpose(0, 1).contiguous();
        Tensor emptyAttnMask = new Tensor();
        Tensor outSeqFirst = encoder.forward(src, emptyAttnMask, padMask);
        Tensor out = outSeqFirst.transpose(0, 1).contiguous(); // [B, S, E]

        // Zero-fill padded positions
        Tensor padMask3d = padMask.unsqueeze(2); // [B, S, 1]
        Tensor outZeroed = out.masked_fill(padMask3d, new Scalar(0.0));

        // Take last firstKCols positions and flatten: [B, k * inDim]
        long k = Math.min((long) firstKCols, seqLen);
        Tensor lastK = outZeroed.narrow(1, seqLen - k, k);
        Tensor flatLastK = lastK.reshape(batchSize, k * inDim);

        if (concatMaxPool && outProj != null) {
            // Max pool with pads set to -1e9 so they never win
            Tensor forMax = out.masked_fill(padMask3d, new Scalar(-1e9));
            Tensor maxPooled = forMax.amax(new long[]{1L}, false); // [B, E]
            Tensor projected = outProj.forward(maxPooled);
            return TensorHelpers.cat(new Tensor[]{flatLastK, projected}, 1);
        }
        return flatLastK;
    }

    /**
     * Build src_key_padding_mask (True = padded/ignore).
     * If a row is fully padded, unmask the last position so Transformer
     * never sees an all-True mask (Python adjust_mask intent).
     */
    private Tensor buildKeyPaddingMask(Tensor mask) {
        // mask: 1=valid, 0=padded  -> pad = mask < 0.5
        Tensor pad = mask.lt(new Scalar(0.5)); // bool [B, S]

        Tensor padF = pad.toType(ScalarType.Float);
        double seqLen = mask.size(1);
        Tensor fullyPadded = padF.sum(new long[]{1L}).eq(new Scalar(seqLen)); // [B]

        // fully padded -> keep last col True; otherwise last col = original pad
        // newLast = lastCol AND NOT fullyPadded  => if fully padded, last becomes False (unmasked)
        Tensor lastCol = pad.select(1L, mask.size(1) - 1); // [B]
        Tensor newLast = lastCol.logical_and(fullyPadded.logical_not());
        return pad.select_scatter(newLast, 1L, mask.size(1) - 1);
    }

    /** Output feature dim of this block. */
    public long outputDim() {
        long kDim = (long) firstKCols * inDim;
        return concatMaxPool ? kDim + inDim : kDim;
    }

    public long inDim() {
        return inDim;
    }

    public int firstKCols() {
        return firstKCols;
    }

    public boolean concatMaxPool() {
        return concatMaxPool;
    }
}
