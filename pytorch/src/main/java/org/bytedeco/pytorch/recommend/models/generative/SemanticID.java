/*
 * Semantic ID (SID) utilities for Kuaishou OneRec-style generative recommendation.
 *
 * Reference:
 *   - OneRec: Unifying Retrieve and Rank with Generative Recommender
 *     https://arxiv.org/abs/2502.18965
 *   - OneRec Technical Report https://arxiv.org/abs/2506.13695
 *   - MiniOneRec https://github.com/AkaliKong/MiniOneRec
 *   - OpenOneRec https://github.com/Kuaishou-OneRec/OpenOneRec
 *
 * Pipeline:
 *   item embedding → RQ-VAE / RQ-KMeans → L residual codes (Semantic ID)
 *   history of SIDs flattened into a token sequence for autoregressive generation.
 *
 * Token layout (shared across the generative stack):
 *   0           PAD
 *   1           BOS
 *   2           EOS
 *   3 + l*K + c  code c at level l   (0 <= c < K, 0 <= l < L)
 *
 * So total vocab size = 3 + L * K.
 */
package org.bytedeco.pytorch.recommend.models.generative;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.recommend.TensorHelpers;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public final class SemanticID {

    public static final int PAD = 0;
    public static final int BOS = 1;
    public static final int EOS = 2;
    public static final int SPECIAL = 3; // first non-special token id

    private SemanticID() {}

    /** Total vocabulary size for L levels each with codebook size K. */
    public static int vocabSize(int numLevels, int codebookSize) {
        if (numLevels <= 0 || codebookSize <= 0) {
            throw new IllegalArgumentException("numLevels and codebookSize must be positive");
        }
        return SPECIAL + numLevels * codebookSize;
    }

    /** Encode residual code (level, code) → token id. */
    public static int encode(int level, int code, int codebookSize) {
        if (level < 0 || code < 0 || code >= codebookSize) {
            throw new IllegalArgumentException(
                    "invalid level/code: level=" + level + " code=" + code + " K=" + codebookSize);
        }
        return SPECIAL + level * codebookSize + code;
    }

    /** Decode token id → [level, code], or null if special/pad. */
    public static int[] decode(int tokenId, int numLevels, int codebookSize) {
        if (tokenId < SPECIAL) return null;
        int offset = tokenId - SPECIAL;
        int level = offset / codebookSize;
        int code = offset % codebookSize;
        if (level >= numLevels) return null;
        return new int[]{level, code};
    }

    public static boolean isCodeToken(int tokenId) {
        return tokenId >= SPECIAL;
    }

    public static int levelOf(int tokenId, int codebookSize) {
        if (tokenId < SPECIAL) return -1;
        return (tokenId - SPECIAL) / codebookSize;
    }

    /**
     * Flatten per-item SIDs into a token sequence.
     *
     * @param itemSids [numItems][numLevels] residual codes (0..K-1)
     * @return flat token ids length = numItems * numLevels
     */
    public static int[] flatten(int[][] itemSids, int codebookSize) {
        if (itemSids == null || itemSids.length == 0) return new int[0];
        int L = itemSids[0].length;
        int[] out = new int[itemSids.length * L];
        int p = 0;
        for (int[] sid : itemSids) {
            if (sid.length != L) {
                throw new IllegalArgumentException("ragged SID lengths");
            }
            for (int l = 0; l < L; l++) {
                out[p++] = encode(l, sid[l], codebookSize);
            }
        }
        return out;
    }

    /**
     * Build a training sequence: [BOS] + flatten(history SIDs) + [EOS optional].
     * Target for NTP is the sequence shifted by 1.
     */
    public static int[] buildSequence(int[][] historySids, int codebookSize, boolean addEos) {
        int[] flat = flatten(historySids, codebookSize);
        int extra = addEos ? 2 : 1;
        int[] seq = new int[flat.length + extra];
        seq[0] = BOS;
        System.arraycopy(flat, 0, seq, 1, flat.length);
        if (addEos) seq[seq.length - 1] = EOS;
        return seq;
    }

    /**
     * Convert RQ-VAE indices tensor [N, L] (long) to int[][] SIDs.
     */
    public static int[][] fromIndicesTensor(Tensor indices) {
        Tensor cpu = indices.to(ScalarType.Long).contiguous().cpu();
        long[] flat = TensorHelpers.toLongArray(cpu);
        long rows = indices.size(0);
        long cols = indices.dim() >= 2 ? indices.size(1) : 1L;
        int[][] out = new int[(int) rows][(int) cols];
        int p = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                out[i][j] = (int) flat[p++];
            }
        }
        return out;
    }

    /**
     * Prefix trie over valid SID paths for constrained decoding.
     * Each path is a length-L sequence of code tokens (already encoded with level offsets).
     */
    public static final class Trie {
        private final Node root = new Node();
        private final int numLevels;
        private final int codebookSize;
        private int size;

        public Trie(int numLevels, int codebookSize) {
            this.numLevels = numLevels;
            this.codebookSize = codebookSize;
        }

        public int size() { return size; }
        public int numLevels() { return numLevels; }
        public int codebookSize() { return codebookSize; }

        /** Insert one item SID (raw codes length L). */
        public void insertCodes(int[] codes) {
            if (codes == null || codes.length != numLevels) {
                throw new IllegalArgumentException("codes length must equal numLevels");
            }
            Node cur = root;
            for (int l = 0; l < numLevels; l++) {
                int tok = encode(l, codes[l], codebookSize);
                cur = cur.children.computeIfAbsent(tok, k -> new Node());
            }
            if (!cur.terminal) {
                cur.terminal = true;
                size++;
            }
        }

        public void insertAll(int[][] allItemSids) {
            for (int[] sid : allItemSids) insertCodes(sid);
        }

        /**
         * Allowed next tokens given the SID-prefix generated so far inside the current item.
         * {@code prefixTokens} are already-encoded code tokens for the incomplete item
         * (length 0..L-1). When prefix is complete (length L), returns {EOS} or empty.
         */
        public int[] allowedNext(List<Integer> prefixTokens) {
            if (prefixTokens == null || prefixTokens.isEmpty()) {
                // first level: all children of root
                return toArray(root.children.keySet());
            }
            Node cur = root;
            for (int tok : prefixTokens) {
                cur = cur.children.get(tok);
                if (cur == null) return new int[0];
            }
            if (prefixTokens.size() >= numLevels) {
                return new int[]{EOS};
            }
            return toArray(cur.children.keySet());
        }

        /** Whether a full L-token SID path exists. */
        public boolean contains(int[] encodedPath) {
            if (encodedPath == null || encodedPath.length != numLevels) return false;
            Node cur = root;
            for (int tok : encodedPath) {
                cur = cur.children.get(tok);
                if (cur == null) return false;
            }
            return cur.terminal;
        }

        private static int[] toArray(Set<Integer> set) {
            int[] a = new int[set.size()];
            int i = 0;
            for (int v : set) a[i++] = v;
            Arrays.sort(a);
            return a;
        }

        private static final class Node {
            final Map<Integer, Node> children = new HashMap<>();
            boolean terminal;
        }
    }

    /**
     * Constrained next-token mask helper for beam / greedy decoding over SID sequences.
     *
     * <p>Tracks how many code tokens have been emitted for the current item and queries
     * the trie for legal continuations. After L codes, forces EOS (or BOS of next item
     * if multi-item session generation is desired — here we generate one item then EOS).
     */
    public static final class ConstrainedDecoder {
        private final Trie trie;
        private final int numLevels;
        private final List<Integer> currentItemPrefix = new ArrayList<>();
        private boolean finished;

        public ConstrainedDecoder(Trie trie) {
            this.trie = trie;
            this.numLevels = trie.numLevels();
        }

        public void reset() {
            currentItemPrefix.clear();
            finished = false;
        }

        public boolean finished() { return finished; }

        public List<Integer> currentPrefix() {
            return Collections.unmodifiableList(currentItemPrefix);
        }

        /** Call after sampling/selecting {@code token}. */
        public void accept(int token) {
            if (finished) return;
            if (token == EOS) {
                finished = true;
                return;
            }
            if (token == BOS || token == PAD) return;
            if (isCodeToken(token)) {
                currentItemPrefix.add(token);
                if (currentItemPrefix.size() >= numLevels) {
                    // item complete — next must be EOS (single-item generation)
                    // keep prefix for validation; finished after EOS
                }
            }
        }

        /** Legal token ids for the next step. */
        public int[] allowed() {
            if (finished) return new int[]{EOS};
            if (currentItemPrefix.size() >= numLevels) return new int[]{EOS};
            return trie.allowedNext(currentItemPrefix);
        }

        /**
         * Apply constraint in-place on a 1-D logits vector [V]: set illegal positions to -inf.
         */
        public void maskLogits(float[] logits) {
            int[] allow = allowed();
            Set<Integer> ok = new HashSet<>();
            for (int a : allow) ok.add(a);
            for (int i = 0; i < logits.length; i++) {
                if (!ok.contains(i)) logits[i] = Float.NEGATIVE_INFINITY;
            }
        }
    }
}
