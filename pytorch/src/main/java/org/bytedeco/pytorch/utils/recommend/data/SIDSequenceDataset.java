/*
 * Sequential interaction data turned into Semantic-ID token sequences for OneRec SFT.
 *
 * Supports:
 *   - MicroLens npy (train_item_seq.npy + train_item_id.npy + item emb)
 *   - Generic long[][] histories + targets
 *
 * Each sample becomes tokens: [BOS] + flatten(hist SIDs) + flatten(target SID) [+ EOS]
 * stored in Batch.tokens for {@link org.bytedeco.pytorch.utils.recommend.trainers.GenerativeTrainer}.
 */
package org.bytedeco.pytorch.utils.recommend.data;

import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.utils.recommend.TensorHelpers;
import org.bytedeco.pytorch.utils.recommend.models.generative.RQKMeans;
import org.bytedeco.pytorch.utils.recommend.models.generative.SemanticID;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class SIDSequenceDataset extends RecommendDataset {

    private final int[][] tokens; // [N][T_i] variable length — we pad on getBatch
    private final int maxSeqLen;
    private final int padId;
    private final long n;

    public SIDSequenceDataset(int[][] tokens, int maxSeqLen) {
        super();
        this.tokens = tokens;
        this.maxSeqLen = maxSeqLen;
        this.padId = SemanticID.PAD;
        this.n = tokens.length;
    }

    @Override
    public long sizeLong() {
        return n;
    }

    @Override
    public Batch getBatch(long index) {
        int[] seq = tokens[(int) index];
        int len = Math.min(seq.length, maxSeqLen);
        int[] padded = new int[maxSeqLen];
        // left content, right pad
        System.arraycopy(seq, 0, padded, 0, len);
        // rest already 0 = PAD
        Tensor t = TensorHelpers.tensor(padded, maxSeqLen).toType(ScalarType.Long);
        return new Batch(
                Collections.emptyMap(), Collections.emptyMap(), Collections.emptyMap(),
                null, t, null, null, null,
                Collections.emptyMap(), null, null);
    }

    @Override
    public List<String> sparseOrder() {
        return Collections.emptyList();
    }

    public int maxSeqLen() { return maxSeqLen; }

    // ---- builders -----------------------------------------------------------

    /**
     * Build from item-level histories.
     *
     * @param histItemIds [N][H] item ids (0 = pad in history)
     * @param targetItemIds [N] target item id
     * @param itemSids [numItems][L] residual codes (row = item id)
     * @param codebookSize K
     * @param addEos append EOS after target
     * @param maxHistItems truncate history to last maxHistItems non-zero
     */
    public static SIDSequenceDataset fromItemSequences(
            long[][] histItemIds,
            long[] targetItemIds,
            int[][] itemSids,
            int codebookSize,
            boolean addEos,
            int maxHistItems,
            int maxSeqLen) {
        int N = histItemIds.length;
        int L = itemSids[0].length;
        int[][] toks = new int[N][];
        int numItems = itemSids.length;
        for (int i = 0; i < N; i++) {
            List<int[]> histSids = new ArrayList<>();
            long[] hist = histItemIds[i];
            // take last maxHistItems non-padding
            List<Integer> ids = new ArrayList<>();
            for (long id : hist) {
                if (id > 0 && id < numItems) ids.add((int) id);
            }
            int from = Math.max(0, ids.size() - maxHistItems);
            for (int j = from; j < ids.size(); j++) {
                histSids.add(itemSids[ids.get(j)]);
            }
            int tgt = (int) targetItemIds[i];
            if (tgt < 0 || tgt >= numItems) tgt = 0;
            // sequence = hist + target as "items"
            int[][] all = new int[histSids.size() + 1][];
            for (int j = 0; j < histSids.size(); j++) all[j] = histSids.get(j);
            all[histSids.size()] = itemSids[tgt];
            toks[i] = SemanticID.buildSequence(all, codebookSize, addEos);
        }
        return new SIDSequenceDataset(toks, maxSeqLen);
    }

    /**
     * MicroLens layout under dataRoot:
     *   item_info_item_emb_d128.npy  [numItems, 128]
     *   train_item_seq.npy           [N, H]
     *   train_item_id.npy            [N]
     *   valid_item_seq.npy / valid_item_id.npy
     */
    public static final class MicroLensSplit {
        public final SIDSequenceDataset train;
        public final SIDSequenceDataset valid;
        public final int[][] itemSids;
        public final SemanticID.Trie trie;
        public final int numItems;
        public final int numLevels;
        public final int codebookSize;
        public final float[] itemEmbFlat;
        public final int embDim;

        public MicroLensSplit(SIDSequenceDataset train, SIDSequenceDataset valid,
                              int[][] itemSids, SemanticID.Trie trie,
                              int numItems, int numLevels, int codebookSize,
                              float[] itemEmbFlat, int embDim) {
            this.train = train;
            this.valid = valid;
            this.itemSids = itemSids;
            this.trie = trie;
            this.numItems = numItems;
            this.numLevels = numLevels;
            this.codebookSize = codebookSize;
            this.itemEmbFlat = itemEmbFlat;
            this.embDim = embDim;
        }
    }

    public static MicroLensSplit loadMicroLens(
            Path dataRoot,
            int numLevels,
            int codebookSize,
            int maxHistItems,
            int maxTrainRows,
            int maxValidRows,
            int rqIters,
            long seed) throws IOException {
        Path embPath = dataRoot.resolve("item_info_item_emb_d128.npy");
        if (!embPath.toFile().exists()) {
            embPath = dataRoot.resolve("item_info_emb.npy");
        }
        System.out.print("Loading item emb: " + embPath + " ... ");
        System.out.flush();
        NpyIO.Array embArr = NpyIO.load(embPath);
        int numItems = (int) embArr.shape[0];
        int embDim = (int) embArr.shape[1];
        float[] embFlat = NpyIO.loadFloat32Flat(embPath);
        System.out.printf("%,d items × %d dim%n", numItems, embDim);

        System.out.print("RQ-KMeans SID fit L=" + numLevels + " K=" + codebookSize + " ... ");
        System.out.flush();
        long t0 = System.nanoTime();
        RQKMeans rqk = new RQKMeans(numLevels, codebookSize, embDim, rqIters, seed);
        // Fit on up to 20k items for speed; predict assigns all items
        int fitMax = Math.min(numItems, 20_000);
        rqk.fit(embFlat, numItems, fitMax);
        int[][] itemSids = rqk.predict(embFlat, numItems);
        SemanticID.Trie trie = rqk.toTrie(itemSids);
        System.out.printf("done in %.1fs  trie=%d (fit on %,d/%,d)%n",
                (System.nanoTime() - t0) / 1e9, trie.size(), fitMax, numItems);

        long[][] trainSeq = NpyIO.loadInt64Matrix(dataRoot.resolve("train_item_seq.npy"));
        long[] trainTgt = NpyIO.loadInt64Flat(dataRoot.resolve("train_item_id.npy"));
        long[][] validSeq = NpyIO.loadInt64Matrix(dataRoot.resolve("valid_item_seq.npy"));
        long[] validTgt = NpyIO.loadInt64Flat(dataRoot.resolve("valid_item_id.npy"));

        if (maxTrainRows > 0 && maxTrainRows < trainSeq.length) {
            trainSeq = sliceRows(trainSeq, maxTrainRows);
            trainTgt = sliceLong(trainTgt, maxTrainRows);
        }
        if (maxValidRows > 0 && maxValidRows < validSeq.length) {
            validSeq = sliceRows(validSeq, maxValidRows);
            validTgt = sliceLong(validTgt, maxValidRows);
        }

        int maxSeqLen = 1 + maxHistItems * numLevels + numLevels + 1; // BOS+hist+tgt+EOS
        SIDSequenceDataset train = fromItemSequences(
                trainSeq, trainTgt, itemSids, codebookSize, false, maxHistItems, maxSeqLen);
        SIDSequenceDataset valid = fromItemSequences(
                validSeq, validTgt, itemSids, codebookSize, false, maxHistItems, maxSeqLen);
        System.out.printf("  train samples: %,d  valid: %,d  maxSeqLen=%d%n",
                train.sizeLong(), valid.sizeLong(), maxSeqLen);
        return new MicroLensSplit(train, valid, itemSids, trie, numItems, numLevels, codebookSize,
                embFlat, embDim);
    }

    /** Subsample train rows randomly (for quick benchmarks). */
    public static SIDSequenceDataset subsample(SIDSequenceDataset ds, int maxRows, long seed) {
        if (maxRows <= 0 || maxRows >= ds.sizeLong()) return ds;
        Random rng = new Random(seed);
        int n = (int) ds.sizeLong();
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        for (int i = n - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
        }
        int[][] toks = new int[maxRows][];
        for (int i = 0; i < maxRows; i++) {
            Batch b = ds.getBatch(idx[i]);
            // extract non-pad prefix roughly — store full padded row as tokens content
            float[] arr = TensorHelpers.toFloatArray(b.tokens.toType(ScalarType.Float).cpu());
            // find last non-pad
            int len = arr.length;
            while (len > 1 && arr[len - 1] == 0) len--;
            int[] seq = new int[len];
            for (int k = 0; k < len; k++) seq[k] = (int) arr[k];
            toks[i] = seq;
        }
        return new SIDSequenceDataset(toks, ds.maxSeqLen());
    }

    private static long[][] sliceRows(long[][] m, int rows) {
        long[][] o = new long[rows][];
        System.arraycopy(m, 0, o, 0, rows);
        return o;
    }

    private static long[] sliceLong(long[] a, int n) {
        long[] o = new long[n];
        System.arraycopy(a, 0, o, 0, n);
        return o;
    }
}
