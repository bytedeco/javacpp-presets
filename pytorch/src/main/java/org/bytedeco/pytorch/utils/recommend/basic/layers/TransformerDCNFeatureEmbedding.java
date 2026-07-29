/*
 * Ported from torchSa: torchrec/model/FeatureEmbedding.scala
 * matching Python FuxiCTR / Transformer_DCN feature composition.
 *
 * Batch-level: likes || views -> [B, 2*embDim]
 * Item-level:  item_id_emb || tags_pooled || emb_d128_proj -> [..., itemInfoDim]
 *
 * Item tables (pretrained emb + tags) are frozen Embedding modules so that
 * model.to(device) moves them with the rest of the parameters.
 */
package org.bytedeco.pytorch.utils.recommend.basic.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.nn.options.LinearOptions;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.TensorHelpers;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class TransformerDCNFeatureEmbedding extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final long embDim;
    private final long embDimPretrain;
    private final long pretrainDim;
    private final long numItems;
    private final int tagsLen;
    private final long itemInfoDim;

    private final EmbeddingImpl itemEmb;
    private final EmbeddingImpl likesEmb;
    private final EmbeddingImpl viewsEmb;
    private final EmbeddingImpl tagsEmb;
    private final EmbeddingImpl pretrainedTable;
    private final EmbeddingImpl tagsTable;
    private final LinearImpl projPretrained;

    /**
     * @param itemVocabSize      vocab for learnable item id embedding
     * @param embDim             categorical embedding dim (likes/views/item_id/tags)
     * @param likesVocabSize     likes_level vocab
     * @param viewsVocabSize     views_level vocab
     * @param tagsVocabSize      tag id vocab
     * @param pretrainDim        raw pretrained emb dim (e.g. 128)
     * @param embDimPretrain     projected pretrained dim (e.g. 128)
     * @param numItems           rows in frozen item tables (item_id == row index)
     * @param tagsLen            fixed tag list length per item (e.g. 5)
     * @param pretrainedEmbFlat  [numItems * pretrainDim] float, row-major; may be null/empty
     * @param itemTagsFlat       [numItems * tagsLen] long tag ids; may be null/empty
     * @param device             target device string
     */
    public TransformerDCNFeatureEmbedding(
            long itemVocabSize,
            long embDim,
            long likesVocabSize,
            long viewsVocabSize,
            long tagsVocabSize,
            long pretrainDim,
            long embDimPretrain,
            long numItems,
            int tagsLen,
            float[] pretrainedEmbFlat,
            long[] itemTagsFlat,
            String device) {
        super("TransformerDCNFeatureEmbedding");
        this.embDim = embDim;
        this.embDimPretrain = embDimPretrain;
        this.pretrainDim = pretrainDim;
        this.numItems = numItems;
        this.tagsLen = tagsLen;
        this.itemInfoDim = embDim + embDim + embDimPretrain; // id + tags_pooled + pretrain_proj

        this.itemEmb = embWithPadding(itemVocabSize, embDim);
        register_module("item_emb", itemEmb);

        this.likesEmb = embWithPadding(likesVocabSize, embDim);
        register_module("likes_emb", likesEmb);

        this.viewsEmb = embWithPadding(viewsVocabSize, embDim);
        register_module("views_emb", viewsEmb);

        this.tagsEmb = embWithPadding(tagsVocabSize, embDim);
        register_module("tags_emb", tagsEmb);

        // Frozen pretrained emb table: Embedding(numItems, pretrainDim)
        this.pretrainedTable = buildFrozenFloatTable(
                "pretrained_table", numItems, pretrainDim, pretrainedEmbFlat);
        // Frozen tags table stored as float ids (tag vocab < 2^24 so float32 exact)
        this.tagsTable = buildFrozenTagsTable(
                "tags_table", numItems, tagsLen, itemTagsFlat);

        // Python: Linear(pretrain_dim, embedding_dim=128, bias=False)
        this.projPretrained = new LinearImpl(
                new LinearOptions(pretrainDim, embDimPretrain).bias(false));
        register_module("proj_pretrained", projPretrained);

        if (device != null && !"cpu".equals(device)) {
            this.to(new Device(device), false);
        }
    }

    /** Convenience: default device via DeviceSupport. */
    public TransformerDCNFeatureEmbedding(
            long itemVocabSize,
            long embDim,
            long likesVocabSize,
            long viewsVocabSize,
            long tagsVocabSize,
            long pretrainDim,
            long embDimPretrain,
            long numItems,
            int tagsLen,
            float[] pretrainedEmbFlat,
            long[] itemTagsFlat) {
        this(itemVocabSize, embDim, likesVocabSize, viewsVocabSize, tagsVocabSize,
                pretrainDim, embDimPretrain, numItems, tagsLen,
                pretrainedEmbFlat, itemTagsFlat, DeviceSupport.backend());
    }

    private static EmbeddingImpl embWithPadding(long num, long dim) {
        EmbeddingOptions opts = new EmbeddingOptions(num, dim);
        opts.padding_idx(new LongOptional(0L));
        return new EmbeddingImpl(opts);
    }

    private EmbeddingImpl buildFrozenFloatTable(
            String name, long rows, long cols, float[] flat) {
        // Build weight via TensorHelpers so we own a stable handle (no ByRef dangle).
        float[] data;
        int expected = (int) (rows * cols);
        if (flat != null && flat.length > 0) {
            if (flat.length == expected) {
                data = flat;
            } else {
                data = new float[expected];
                System.arraycopy(flat, 0, data, 0, Math.min(flat.length, expected));
            }
        } else {
            data = new float[expected];
        }

        EmbeddingOptions opts = new EmbeddingOptions(rows, cols);
        opts.padding_idx(new LongOptional(0L));
        EmbeddingImpl emb = new EmbeddingImpl(opts);
        // IMPORTANT: do NOT replace emb.weight() with a new Tensor handle via emb.weight(t).
        // That ByRef swap often leaves a weight that Module.to(device) cannot move to MPS
        // ("Placeholder storage has not been allocated on MPS device").
        // Copy into the Embedding-owned parameter storage, then freeze.
        // In-place on a leaf that requires_grad is rejected — clear the flag first
        // (same pattern as LoraLinear merge / HF peft).
        Tensor src = TensorHelpers.tensor(data, rows, cols).contiguous();
        try (org.bytedeco.pytorch.NoGradGuard g = new org.bytedeco.pytorch.NoGradGuard()) {
            Tensor w = emb.weight();
            w.requires_grad_(false);
            w.copy_(src);
            w.requires_grad_(false);
        }
        register_module(name, emb);
        return emb;
    }

    private EmbeddingImpl buildFrozenTagsTable(
            String name, long rows, int cols, long[] tagsFlat) {
        float[] asFloat = null;
        if (tagsFlat != null && tagsFlat.length > 0) {
            asFloat = new float[tagsFlat.length];
            for (int i = 0; i < tagsFlat.length; i++) {
                asFloat[i] = (float) tagsFlat[i];
            }
        }
        return buildFrozenFloatTable(name, rows, cols, asFloat);
    }

    /** Batch-level features: likes || views -> [B, 2*embDim]. */
    public Tensor encodeBatch(Tensor likesLevel, Tensor viewsLevel) {
        Tensor likesE = likesEmb.forward(likesLevel);
        Tensor viewsE = viewsEmb.forward(viewsLevel);
        return TensorHelpers.cat(new Tensor[]{likesE, viewsE}, 1);
    }

    /**
     * Item-level features for arbitrary item id tensor.
     * itemIds: [B, S] or [B] -> [..., itemInfoDim]
     */
    public Tensor encodeItems(Tensor itemIds) {
        int nDims = (int) itemIds.dim();
        Tensor flatIds = itemIds.reshape(-1L);

        Tensor safeIds = flatIds
                .clamp_min(new Scalar(0L))
                .clamp_max(new Scalar(numItems - 1));

        // 1) item_id categorical embedding
        Tensor idEmb = itemEmb.forward(safeIds); // [N, embDim]

        // 2) tags: frozen table lookup -> [N, tagsLen] float ids -> Long -> emb -> pool
        Tensor tagsIdxF = tagsTable.forward(safeIds); // [N, tagsLen] float
        Tensor tagsIdx = tagsIdxF.toType(ScalarType.Long);
        Tensor tagsE = tagsEmb.forward(tagsIdx); // [N, tagsLen, embDim]
        Tensor tagsPooled = maskedAvgPool(tagsE, tagsIdx);

        // 3) pretrained emb: frozen embedding + Linear(bias=false)
        Tensor pre = pretrainedTable.forward(safeIds); // [N, pretrainDim]
        Tensor preProj = projPretrained.forward(pre);  // [N, embDimPretrain]

        Tensor flatOut = TensorHelpers.cat(new Tensor[]{idEmb, tagsPooled, preProj}, 1);

        if (nDims == 1) {
            return flatOut;
        }
        long[] outShape = new long[nDims + 1];
        for (int i = 0; i < nDims; i++) {
            outShape[i] = itemIds.size(i);
        }
        outShape[nDims] = itemInfoDim;
        return flatOut.reshape(outShape);
    }

    private Tensor maskedAvgPool(Tensor tagsE, Tensor tagIds) {
        // mask: non-zero tag ids are valid
        Tensor mask = tagIds.ne(new Scalar(0L)).toType(ScalarType.Float); // [N, T]
        Tensor sumOut = tagsE.sum(new long[]{1L});
        Tensor count = mask.sum(new long[]{1L}).unsqueeze(1).add(new Scalar(1e-12));
        return sumOut.div(count);
    }

    public long itemInfoDim() {
        return itemInfoDim;
    }

    public long embDim() {
        return embDim;
    }

    public long embDimPretrain() {
        return embDimPretrain;
    }

    public long pretrainDim() {
        return pretrainDim;
    }

    public long numItems() {
        return numItems;
    }

    public int tagsLen() {
        return tagsLen;
    }
}
