/*
 * Constrained beam search over Semantic ID trie (MiniOneRec-style).
 *
 * Only expands beams along trie-legal next tokens, so every completed L-code
 * path is a real catalog item. Used by GRPO sampling and offline generation.
 *
 * Reference:
 *   - MiniOneRec LogitProcessor / constrained decoding
 *     https://github.com/AkaliKong/MiniOneRec
 *   - OneRec session generation with valid SID paths
 */
package org.bytedeco.pytorch.recommend.models.generative;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.recommend.TensorHelpers;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

public final class ConstrainedBeamSearch {

    private ConstrainedBeamSearch() {}

    public static final class Hypothesis {
        public final int[] tokens;       // generated code tokens so far (length <= L)
        public final double logProb;     // cumulative log-prob
        public final boolean finished;   // reached L codes (or EOS)

        public Hypothesis(int[] tokens, double logProb, boolean finished) {
            this.tokens = tokens;
            this.logProb = logProb;
            this.finished = finished;
        }

        public Hypothesis extend(int token, double tokenLogProb, int numLevels) {
            int[] next = Arrays.copyOf(tokens, tokens.length + 1);
            next[tokens.length] = token;
            boolean done = next.length >= numLevels;
            return new Hypothesis(next, logProb + tokenLogProb, done);
        }
    }

    public static final class Result {
        public final List<Hypothesis> beams; // sorted best-first
        public Result(List<Hypothesis> beams) { this.beams = beams; }
        public int[] bestTokens() {
            return beams.isEmpty() ? new int[0] : beams.get(0).tokens;
        }
    }

    /**
     * Functional interface: given current prefix token ids [T], return logits [V] for next step.
     * Implementations typically call model.forward on a batch-1 tensor.
     */
    @FunctionalInterface
    public interface StepLogits {
        float[] logits(int[] prefixIncludingContext);
    }

    /**
     * Beam search for one next-item SID (exactly numLevels code tokens).
     *
     * @param contextPrefix already-generated context tokens (BOS+hist...), not modified
     * @param trie          legal SID trie
     * @param step          model step function
     * @param beamSize      beam width
     * @param numLevels     L
     */
    public static Result search(
            int[] contextPrefix,
            SemanticID.Trie trie,
            StepLogits step,
            int beamSize,
            int numLevels) {
        if (beamSize <= 0) throw new IllegalArgumentException("beamSize must be positive");
        List<Hypothesis> beams = new ArrayList<>();
        beams.add(new Hypothesis(new int[0], 0.0, false));

        for (int depth = 0; depth < numLevels; depth++) {
            PriorityQueue<Hypothesis> candidates =
                    new PriorityQueue<>(Comparator.comparingDouble(h -> -h.logProb));
            for (Hypothesis hyp : beams) {
                if (hyp.finished) {
                    candidates.add(hyp);
                    continue;
                }
                // build full prefix for model: context + generated codes so far
                int[] full = concat(contextPrefix, hyp.tokens);
                float[] logits = step.logits(full);
                // constrain
                List<Integer> genPrefix = new ArrayList<>();
                for (int t : hyp.tokens) genPrefix.add(t);
                int[] allowed = trie.allowedNext(genPrefix);
                if (allowed.length == 0) continue;

                // log-softmax over full vocab then pick allowed
                double[] logp = logSoftmax(logits);
                // take top among allowed (up to beamSize)
                Integer[] order = new Integer[allowed.length];
                for (int i = 0; i < allowed.length; i++) order[i] = allowed[i];
                Arrays.sort(order, (a, b) -> Double.compare(logp[b], logp[a]));
                int take = Math.min(beamSize, order.length);
                for (int i = 0; i < take; i++) {
                    int tok = order[i];
                    candidates.add(hyp.extend(tok, logp[tok], numLevels));
                }
            }
            // prune to beamSize
            List<Hypothesis> next = new ArrayList<>(beamSize);
            while (!candidates.isEmpty() && next.size() < beamSize) {
                next.add(candidates.poll());
            }
            // sort best first
            next.sort((a, b) -> Double.compare(b.logProb, a.logProb));
            beams = next;
            if (beams.isEmpty()) break;
        }
        return new Result(beams);
    }

    /**
     * Convenience: run beam search using OneRec/OneRecV2/OpenOneRec forward on device.
     * Context tensor [1, T] long; returns best SID tokens length L.
     */
    public static int[] generateOne(
            Module model,
            Tensor contextTokens,
            SemanticID.Trie trie,
            int beamSize,
            int numLevels,
            String device) {
        long[] ctxFlat = TensorHelpers.toLongArray(
                contextTokens.reshape(-1L).to(ScalarType.Long).cpu().contiguous());
        int[] ctx = new int[ctxFlat.length];
        for (int i = 0; i < ctxFlat.length; i++) ctx[i] = (int) ctxFlat[i];

        StepLogits step = prefix -> {
            int[] full = prefix;
            Tensor t = TensorHelpers.tensor(full, 1L, (long) full.length).toType(ScalarType.Long);
            if (device != null && !"cpu".equals(device)) {
                try {
                    t = t.to(new org.bytedeco.pytorch.Device(device), ScalarType.Long);
                } catch (Throwable ignored) {}
            }
            Tensor logits;
            if (model instanceof OneRec) {
                logits = ((OneRec) model).forward(t);
            } else if (model instanceof OneRecV2) {
                logits = ((OneRecV2) model).forward(t);
            } else if (model instanceof OpenOneRec) {
                logits = ((OpenOneRec) model).forward(t);
            } else {
                throw new IllegalArgumentException("unsupported model: " + model.getClass().getName());
            }
            Tensor last = logits.select(1, logits.size(1) - 1).select(0, 0)
                    .contiguous().cpu().toType(ScalarType.Float);
            return TensorHelpers.toFloatArray(last);
        };

        Result r = search(ctx, trie, step, beamSize, numLevels);
        return r.bestTokens();
    }

    /** Sample up to {@code groupSize} diverse SID completions for GRPO. */
    public static List<int[]> sampleGroup(
            Module model,
            Tensor contextTokens,
            SemanticID.Trie trie,
            int groupSize,
            int numLevels,
            String device,
            long seed) {
        // Use beam search with beam=groupSize and return all finished hypotheses
        long[] ctxFlat = TensorHelpers.toLongArray(
                contextTokens.reshape(-1L).to(ScalarType.Long).cpu().contiguous());
        int[] ctx = new int[ctxFlat.length];
        for (int i = 0; i < ctxFlat.length; i++) ctx[i] = (int) ctxFlat[i];

        StepLogits step = prefix -> {
            Tensor t = TensorHelpers.tensor(prefix, 1L, (long) prefix.length).toType(ScalarType.Long);
            if (device != null && !"cpu".equals(device)) {
                try {
                    t = t.to(new org.bytedeco.pytorch.Device(device), ScalarType.Long);
                } catch (Throwable ignored) {}
            }
            Tensor logits;
            if (model instanceof OneRec) logits = ((OneRec) model).forward(t);
            else if (model instanceof OneRecV2) logits = ((OneRecV2) model).forward(t);
            else if (model instanceof OpenOneRec) logits = ((OpenOneRec) model).forward(t);
            else throw new IllegalArgumentException("unsupported model");
            return TensorHelpers.toFloatArray(
                    logits.select(1, logits.size(1) - 1).select(0, 0)
                            .contiguous().cpu().toType(ScalarType.Float));
        };
        Result r = search(ctx, trie, step, Math.max(groupSize, 2), numLevels);
        List<int[]> out = new ArrayList<>();
        for (Hypothesis h : r.beams) {
            if (h.tokens.length == numLevels) out.add(h.tokens);
            if (out.size() >= groupSize) break;
        }
        // if not enough unique beams, pad by repeating best (rare with large trie)
        while (!out.isEmpty() && out.size() < groupSize) {
            out.add(Arrays.copyOf(out.get(0), numLevels));
        }
        return out;
    }

    private static int[] concat(int[] a, int[] b) {
        int[] c = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, c, a.length, b.length);
        return c;
    }

    private static double[] logSoftmax(float[] logits) {
        double max = Double.NEGATIVE_INFINITY;
        for (float v : logits) if (v > max && !Float.isInfinite(v)) max = v;
        if (Double.isInfinite(max)) max = 0;
        double sum = 0;
        double[] exps = new double[logits.length];
        for (int i = 0; i < logits.length; i++) {
            double e = Math.exp(logits[i] - max);
            if (Double.isNaN(e) || Double.isInfinite(e)) e = 0;
            exps[i] = e;
            sum += e;
        }
        if (sum <= 0) sum = 1;
        double[] logp = new double[logits.length];
        double logSum = Math.log(sum);
        for (int i = 0; i < logits.length; i++) {
            if (exps[i] <= 0) logp[i] = -1e9;
            else logp[i] = Math.log(exps[i]) - logSum;
        }
        return logp;
    }
}
