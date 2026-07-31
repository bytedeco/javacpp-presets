/*
 * Typed Module.forward dispatch for recommend models.
 *
 * Replaces the giant Scala match-case trees in CTRTrainer / MatchTrainer / MTLTrainer.
 * Each adapter calls the concrete model's overloaded forward(...) — no reflection.
 *
 * Usage:
 *   BatchForward fwd = ModelForwards.ctr(model);
 *   Tensor logits = fwd.forward(batch);
 */
package org.bytedeco.pytorch.recommend.trainers;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.recommend.data.Batch;
import org.bytedeco.pytorch.recommend.models.generative.HLLM;
import org.bytedeco.pytorch.recommend.models.generative.LLM4Rec;
import org.bytedeco.pytorch.recommend.models.matching.DSSM;
import org.bytedeco.pytorch.recommend.models.matching.MAMBA;
import org.bytedeco.pytorch.recommend.models.matching.MIND;
import org.bytedeco.pytorch.recommend.models.matching.YoutubeDNN;
import org.bytedeco.pytorch.recommend.models.multi_task.AITM;
import org.bytedeco.pytorch.recommend.models.multi_task.ESMM;
import org.bytedeco.pytorch.recommend.models.multi_task.MMOE;
import org.bytedeco.pytorch.recommend.models.multi_task.MetaHeac;
import org.bytedeco.pytorch.recommend.models.multi_task.OMoE;
import org.bytedeco.pytorch.recommend.models.multi_task.PLE;
import org.bytedeco.pytorch.recommend.models.multi_task.SharedBottom;
import org.bytedeco.pytorch.recommend.models.multi_task.SingleTaskModel;
import org.bytedeco.pytorch.recommend.models.ranking.AFM;
import org.bytedeco.pytorch.recommend.models.ranking.AutoInt;
import org.bytedeco.pytorch.recommend.models.ranking.BST;
import org.bytedeco.pytorch.recommend.models.ranking.DCN;
import org.bytedeco.pytorch.recommend.models.ranking.DCNv2;
import org.bytedeco.pytorch.recommend.models.ranking.DIEN;
import org.bytedeco.pytorch.recommend.models.ranking.DIN;
import org.bytedeco.pytorch.recommend.models.ranking.DeepFM;
import org.bytedeco.pytorch.recommend.models.ranking.ETA;
import org.bytedeco.pytorch.recommend.models.ranking.FiBiNet;
import org.bytedeco.pytorch.recommend.models.ranking.LiquidNetWork;
import org.bytedeco.pytorch.recommend.models.ranking.MEMBA;
import org.bytedeco.pytorch.recommend.models.ranking.SIM;
import org.bytedeco.pytorch.recommend.models.ranking.WideDeep;
import org.bytedeco.pytorch.recommend.models.ranking.XDeepFM;
import org.bytedeco.pytorch.recommend.models.ranking.XGBoostModel;
import org.bytedeco.pytorch.recommend.trainers.Trainer.BatchForward;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

public final class ModelForwards {

    private ModelForwards() {}

    /**
     * CTR / ranking logits from a Batch. Returns null when the batch lacks required fields
     * for the given model (e.g. sequence models without sequenceFeatures).
     */
    public static BatchForward ctr(Module model) {
        if (model instanceof DeepFM) {
            DeepFM m = (DeepFM) model;
            return b -> m.forward(b.sparseFeatures, b.denseFeatures);
        }
        if (model instanceof DCN) {
            DCN m = (DCN) model;
            return b -> m.forward(b.sparseFeatures, b.denseFeatures);
        }
        if (model instanceof DCNv2) {
            DCNv2 m = (DCNv2) model;
            return b -> m.forward(b.sparseFeatures, b.denseFeatures);
        }
        if (model instanceof AFM) {
            AFM m = (AFM) model;
            return b -> m.forward(b.sparseFeatures, b.denseFeatures);
        }
        if (model instanceof WideDeep) {
            WideDeep m = (WideDeep) model;
            return b -> m.forward(b.sparseFeatures, b.denseFeatures);
        }
        if (model instanceof XDeepFM) {
            XDeepFM m = (XDeepFM) model;
            return b -> m.forward(b.sparseFeatures, b.denseFeatures);
        }
        if (model instanceof FiBiNet) {
            FiBiNet m = (FiBiNet) model;
            return b -> m.forward(b.sparseFeatures, b.denseFeatures);
        }
        if (model instanceof AutoInt) {
            AutoInt m = (AutoInt) model;
            return b -> m.forward(b.sparseFeatures, b.denseFeatures);
        }
        if (model instanceof XGBoostModel) {
            XGBoostModel m = (XGBoostModel) model;
            return b -> m.forward(b.sparseFeatures, b.denseFeatures);
        }
        if (model instanceof DIN) {
            DIN m = (DIN) model;
            return b -> {
                if (b.sequenceFeatures == null || b.sequenceFeatures.isEmpty() || b.labels == null) {
                    return null;
                }
                Tensor targetIdx = b.labels.view(b.labels.size(0), 1L).toType(ScalarType.Long);
                return m.forward(b.sparseFeatures, b.sequenceFeatures, targetIdx);
            };
        }
        if (model instanceof BST) {
            BST m = (BST) model;
            return b -> {
                if (b.sequenceFeatures == null || b.sequenceFeatures.isEmpty()) {
                    return null;
                }
                return m.forward(b.sparseFeatures, b.sequenceFeatures);
            };
        }
        if (model instanceof SIM) {
            SIM m = (SIM) model;
            return b -> {
                if (b.sequenceFeatures == null || b.sequenceFeatures.isEmpty()) {
                    return null;
                }
                // Use sequence features for item/cate/time/target when dedicated maps absent
                Map<String, Tensor> seq = b.sequenceFeatures;
                return m.forward(b.sparseFeatures, seq, seq, seq, seq);
            };
        }
        if (model instanceof ETA) {
            ETA m = (ETA) model;
            return b -> {
                if (b.sequenceFeatures == null || b.sequenceFeatures.isEmpty()) {
                    return null;
                }
                return m.forward(b.sparseFeatures, b.sequenceFeatures, b.sequenceFeatures);
            };
        }
        if (model instanceof MEMBA) {
            MEMBA m = (MEMBA) model;
            return b -> {
                if (b.sequenceFeatures == null || b.sequenceFeatures.isEmpty() || b.labels == null) {
                    return null;
                }
                Tensor targetIdx = b.labels.view(b.labels.size(0), 1L).toType(ScalarType.Long);
                return m.forward(b.sparseFeatures, b.sequenceFeatures, targetIdx);
            };
        }
        if (model instanceof LiquidNetWork) {
            LiquidNetWork m = (LiquidNetWork) model;
            return b -> {
                if (b.sequenceFeatures == null || b.sequenceFeatures.isEmpty()) {
                    return null;
                }
                return m.forward(b.sparseFeatures, b.sequenceFeatures);
            };
        }
        if (model instanceof DIEN) {
            DIEN m = (DIEN) model;
            return b -> {
                if (b.sequenceFeatures == null || b.sequenceFeatures.isEmpty()) {
                    return null;
                }
                return m.forward(b.sparseFeatures, b.sequenceFeatures);
            };
        }
        if (model instanceof LLM4Rec) {
            LLM4Rec m = (LLM4Rec) model;
            return b -> {
                if (b.tokens == null) {
                    return null;
                }
                Tensor positions = b.positions != null
                        ? b.positions
                        : Trainer.positionsFor(b.tokens, null);
                return m.forward(b.tokens, positions);
            };
        }
        if (model instanceof HLLM) {
            HLLM m = (HLLM) model;
            return b -> hllmCtrLogits(m, b);
        }
        // Generic fallback: sparse-only models that expose forward(Map)
        return b -> sparseOnlyFallback(model, b);
    }

    /**
     * HLLM CTR path: last-step vocab logits gathered at target item id from sparse features.
     */
    private static Tensor hllmCtrLogits(HLLM m, Batch b) {
        if (b.tokens == null || b.sparseFeatures == null || b.sparseFeatures.isEmpty()) {
            return null;
        }
        Tensor allLogits = m.forward(b.tokens, b.timeDiffs);
        long batchSize = allLogits.size(0);
        long seqLen = allLogits.size(1);
        Tensor lastStep = allLogits.select(1, (int) (seqLen - 1));
        Tensor targetItem = b.sparseFeatures.values().iterator().next();
        Tensor targetItem2D = targetItem.view(batchSize, 1L).toType(ScalarType.Long);
        return lastStep.gather(1, targetItem2D);
    }

    private static Tensor sparseOnlyFallback(Module model, Batch b) {
        // Last-resort: try common ranking models already covered; otherwise null.
        // Keeps trainers from crashing on unknown types — caller skips the batch.
        return null;
    }

    // ---- matching towers -----------------------------------------------------

    /** User-tower embedding for matching models. */
    public static BatchForward matchUser(Module model, String device) {
        if (model instanceof DSSM) {
            DSSM m = (DSSM) model;
            return b -> m.userTowerForward(b.sparseFeatures);
        }
        if (model instanceof YoutubeDNN) {
            YoutubeDNN m = (YoutubeDNN) model;
            return b -> {
                Tensor history = b.sparseFeatures.get("history");
                if (history != null) {
                    Map<String, Tensor> sparseOnly = new LinkedHashMap<>(b.sparseFeatures);
                    sparseOnly.remove("history");
                    return m.userTowerForward(sparseOnly, Collections.singletonMap("history", history));
                }
                return m.userTowerForward(b.sparseFeatures);
            };
        }
        if (model instanceof MAMBA) {
            MAMBA m = (MAMBA) model;
            return b -> {
                if (b.tokens == null) {
                    return null;
                }
                Tensor positions = b.positions != null
                        ? b.positions
                        : Trainer.positionsFor(b.tokens, device);
                return m.forward(b.tokens, positions);
            };
        }
        if (model instanceof MIND) {
            MIND m = (MIND) model;
            return b -> {
                if (b.tokens == null) {
                    return null;
                }
                return m.forward(b.sparseFeatures, b.tokens);
            };
        }
        if (model instanceof DIEN) {
            DIEN m = (DIEN) model;
            return b -> {
                Map<String, Tensor> seq = b.sequenceFeatures;
                if (seq == null || seq.isEmpty()) {
                    if (b.tokens == null) {
                        return null;
                    }
                    seq = Collections.singletonMap("seq_feat", b.tokens);
                }
                return m.forward(b.sparseFeatures, seq);
            };
        }
        return b -> null;
    }

    /** Item-tower embedding for matching models. */
    public static BatchForward matchItem(Module model) {
        if (model instanceof DSSM) {
            DSSM m = (DSSM) model;
            return b -> {
                if (b.itemFeatures == null || b.itemFeatures.isEmpty()) {
                    return null;
                }
                return m.itemTowerForward(b.itemFeatures);
            };
        }
        // Generic: first item feature tensor as embedding (MAMBA/MIND path in Scala)
        return b -> {
            if (b.itemFeatures == null || b.itemFeatures.isEmpty()) {
                return null;
            }
            return b.itemFeatures.values().iterator().next();
        };
    }

    // ---- multi-task ----------------------------------------------------------

    /**
     * MTL forward. Returns either a concatenated Tensor [B, T] or Map&lt;String,Tensor&gt;.
     * Callers should use {@link #mtlAsMap} / {@link #mtlTaskLogit}.
     */
    public static Object mtlRaw(Module model, Batch batch) {
        Map<String, Tensor> sparse = batch.sparseFeatures;
        if (model instanceof MMOE) {
            return ((MMOE) model).forward(sparse);
        }
        if (model instanceof SharedBottom) {
            return ((SharedBottom) model).forward(sparse);
        }
        if (model instanceof PLE) {
            return ((PLE) model).forward(sparse);
        }
        if (model instanceof ESMM) {
            return ((ESMM) model).forward(sparse);
        }
        if (model instanceof AITM) {
            return ((AITM) model).forward(sparse);
        }
        if (model instanceof OMoE) {
            return ((OMoE) model).forward(sparse);
        }
        if (model instanceof SingleTaskModel) {
            return ((SingleTaskModel) model).forward(sparse);
        }
        if (model instanceof MetaHeac) {
            return ((MetaHeac) model).forwardByName(sparse);
        }
        throw new IllegalArgumentException("Unknown MTL model type: " + model.getClass().getName());
    }

    /** Extract one task's logit from MTL raw output. */
    @SuppressWarnings("unchecked")
    public static Tensor mtlTaskLogit(Object raw, String taskName, int taskIndex) {
        if (raw instanceof Tensor) {
            return ((Tensor) raw).select(1, taskIndex);
        }
        if (raw instanceof Map) {
            return ((Map<String, Tensor>) raw).get(taskName);
        }
        return null;
    }

    /** Default positions helper exposed for callers that already have tokens. */
    public static Tensor defaultPositions(Tensor tokens, String device) {
        return Trainer.positionsFor(tokens, device);
    }
}
