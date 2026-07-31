/*
 * Trainer for generative recommenders (OneRec / HSTU / LLM4Rec-style NTP).
 *
 * Extends {@link Trainer}. Expects batches whose {@code tokens} field holds the
 * full SID / item token sequence [B, T] (BOS + history codes [+ EOS]).
 *
 * Loss is next-token prediction via the model's {@code computeLoss(tokens)} when
 * available (OneRec), otherwise CE on {@code forward(tokens[:, :-1])} vs shifted labels.
 *
 * Reference:
 *   - OneRec https://arxiv.org/abs/2502.18965
 *   - MiniOneRec SFT stage https://github.com/AkaliKong/MiniOneRec
 */
package org.bytedeco.pytorch.recommend.trainers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.recommend.models.generative.OneRecV2;
import org.bytedeco.pytorch.recommend.models.generative.OpenOneRec;
import org.bytedeco.pytorch.recommend.TensorHelpers;
import org.bytedeco.pytorch.recommend.data.Batch;
import org.bytedeco.pytorch.recommend.models.generative.OneRec;
import org.bytedeco.pytorch.recommend.models.generative.SemanticID;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class GenerativeTrainer extends Trainer<GenerativeTrainer> {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    /** Optional SID trie for constrained eval / generation. */
    private SemanticID.Trie trie;
    private int topK;
    private boolean reportTokenAccuracy;

    public GenerativeTrainer(Module model) {
        super(model);
        this.topK = 10;
        this.reportTokenAccuracy = true;
        // generative NTP usually minimises loss for early stop
        maximizeMetric(false);
    }

    public GenerativeTrainer withTrie(SemanticID.Trie trie) {
        this.trie = trie;
        return this;
    }

    public GenerativeTrainer topK(int k) {
        this.topK = Math.max(1, k);
        return this;
    }

    public GenerativeTrainer reportTokenAccuracy(boolean v) {
        this.reportTokenAccuracy = v;
        return this;
    }

    @Override
    protected String primaryMetricName() {
        return "loss";
    }

    @Override
    protected Tensor computeTrainLoss(Batch batch) {
        if (batch == null || batch.tokens == null) {
            return null;
        }
        Tensor tokens = batch.tokens.toType(ScalarType.Long);
        if (model instanceof OneRec) {
            return ((OneRec) model).computeLoss(tokens);
        }
        if (model instanceof OneRecV2) {
            return ((OneRecV2) model)
                    .computeLoss(tokens);
        }
        if (model instanceof OpenOneRec) {
            return ((OpenOneRec) model)
                    .computeLoss(tokens);
        }
        // Generic: forward on prefix, CE vs shifted targets
        Tensor input = tokens.narrow(1, 0, tokens.size(1) - 1);
        Tensor target = tokens.narrow(1, 1, tokens.size(1) - 1);
        Tensor logits = genericForward(input);
        if (logits == null) return null;
        return tokenCeLoss(logits, target);
    }

    @Override
    protected Tensor predictBatch(Batch batch) {
        if (batch == null || batch.tokens == null) return null;
        Tensor tokens = batch.tokens.toType(ScalarType.Long);
        if (model instanceof OneRec) {
            return ((OneRec) model).forward(tokens);
        }
        if (model instanceof OneRecV2) {
            return ((OneRecV2) model)
                    .forward(tokens);
        }
        if (model instanceof OpenOneRec) {
            return ((OpenOneRec) model)
                    .forward(tokens);
        }
        return genericForward(tokens);
    }

    private Tensor genericForward(Tensor tokens) {
        // Extension point for other generative models.
        return null;
    }

    /** CE over [B,T,V] logits vs [B,T] targets, ignore PAD. */
    private static Tensor tokenCeLoss(Tensor logits, Tensor target) {
        long B = logits.size(0);
        long T = logits.size(1);
        long V = logits.size(2);
        Tensor flatLogits = logits.reshape(B * T, V);
        Tensor flatTarget = target.reshape(B * T).toType(ScalarType.Long);
        Tensor logProb = torch.log_softmax(flatLogits, 1L);
        Tensor logp = logProb.gather(1, flatTarget.view(-1L, 1L)).squeeze(1).neg();
        Tensor mask = flatTarget.ne(new Scalar((long) SemanticID.PAD)).toType(ScalarType.Float);
        Tensor denom = mask.sum().clamp_min(new Scalar(1.0f));
        return logp.mul(mask).sum().div(denom);
    }

    @Override
    public Map<String, Float> evaluate(Iterable<Batch> dataLoader) {
        model.eval();
        double lossSum = 0.0;
        long lossN = 0;
        long correct = 0;
        long total = 0;

        for (Batch batch : dataLoader) {
            if (batch == null || batch.tokens == null) continue;
            try {
                Tensor tokens = batch.tokens.toType(ScalarType.Long);
                Tensor loss = computeTrainLoss(batch);
                if (loss == null) continue;
                float lv = (float) TensorHelpers.itemSafe(loss);
                if (!Float.isNaN(lv) && !Float.isInfinite(lv)) {
                    long bs = tokens.size(0);
                    lossSum += lv * bs;
                    lossN += bs;
                }

                if (reportTokenAccuracy) {
                    Tensor input = tokens.narrow(1, 0, tokens.size(1) - 1);
                    Tensor target = tokens.narrow(1, 1, tokens.size(1) - 1);
                    Tensor logits = null;
                    if (model instanceof OneRec) {
                        logits = ((OneRec) model).forward(input);
                    } else if (model instanceof OneRecV2) {
                        logits = ((OneRecV2) model)
                                .forward(input);
                    } else if (model instanceof OpenOneRec) {
                        logits = ((OpenOneRec) model)
                                .forward(input);
                    } else {
                        logits = genericForward(input);
                    }
                    if (logits != null) {
                        Tensor pred = logits.argmax(new org.bytedeco.pytorch.LongOptional(2L), false);
                        Tensor mask = target.ne(new Scalar((long) SemanticID.PAD));
                        Tensor eq = pred.eq(target).logical_and(mask);
                        correct += (long) TensorHelpers.itemSafe(eq.toType(ScalarType.Long).sum());
                        total += (long) TensorHelpers.itemSafe(mask.toType(ScalarType.Long).sum());
                    }
                }
            } catch (Throwable t) {
                // skip bad batch
            }
        }

        model.train(true);
        Map<String, Float> out = new LinkedHashMap<>();
        out.put("loss", lossN > 0 ? (float) (lossSum / lossN) : 0.0f);
        if (reportTokenAccuracy) {
            out.put("token_acc", total > 0 ? (float) correct / total : 0.0f);
        }
        return out;
    }

    /**
     * Generate next-item SIDs for each row (OneRec / OneRecV2 / OpenOneRec).
     */
    public List<int[]> generate(Iterable<Batch> dataLoader) {
        List<int[]> results = new ArrayList<>();
        model.eval();
        for (Batch batch : dataLoader) {
            if (batch == null || batch.tokens == null) continue;
            Tensor tokens = batch.tokens.toType(ScalarType.Long);
            int B = (int) tokens.size(0);
            SemanticID.ConstrainedDecoder[] decoders = null;
            if (trie != null) {
                decoders = new SemanticID.ConstrainedDecoder[B];
                for (int b = 0; b < B; b++) {
                    decoders[b] = new SemanticID.ConstrainedDecoder(trie);
                }
            }
            Tensor gen;
            int L;
            if (model instanceof OneRec) {
                OneRec m = (OneRec) model;
                gen = m.generateItem(tokens, decoders);
                L = m.numLevels();
            } else if (model instanceof OneRecV2) {
                OneRecV2 m =
                        (OneRecV2) model;
                gen = m.generateItem(tokens, decoders);
                L = m.numLevels();
            } else if (model instanceof OpenOneRec) {
                OpenOneRec m =
                        (OpenOneRec) model;
                gen = m.generateItem(tokens, decoders);
                L = m.numLevels();
            } else {
                throw new UnsupportedOperationException(
                        "generate() requires OneRec/OneRecV2/OpenOneRec");
            }
            Tensor host = gen.to(ScalarType.Long).cpu().contiguous();
            long[] flat = TensorHelpers.toLongArray(host);
            for (int b = 0; b < B; b++) {
                int[] sid = new int[L];
                for (int l = 0; l < L; l++) sid[l] = (int) flat[b * L + l];
                results.add(sid);
            }
        }
        model.train(true);
        return results;
    }
}
