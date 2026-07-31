/*
 * Ported from torch-rechub-scala: torchrec/basic/losses/Loss.scala
 *
 * BCELoss, BCEWithLogitsLoss, CrossEntropyLoss, BPRLoss, HingeLoss,
 * TripletMarginLoss, InBatchNCELoss, MaskedCrossEntropyLoss, FocalLoss.
 *
 * Note: BPRLoss/HingeLoss/InBatchNCELoss here are the losses-package variants
 * (different from basic.layers.* pair-ranking losses used by matching trainers).
 */
package org.bytedeco.pytorch.recommend.basic.losses;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class Losses {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private Losses() {}

    /** Binary cross-entropy on probabilities (manual clamp+log form). */
    public static final class BCELoss {
        private final String reduction;
        private final Tensor posWeight; // unused in Scala body; kept for API parity

        public BCELoss() {
            this("mean", null);
        }

        public BCELoss(String reduction) {
            this(reduction, null);
        }

        public BCELoss(String reduction, Tensor posWeight) {
            this.reduction = reduction != null ? reduction : "mean";
            this.posWeight = posWeight;
        }

        public Tensor apply(Tensor predictions, Tensor targets) {
            float eps = 1e-7f;
            Tensor clipped = predictions.clamp(
                    new ScalarOptional(new Scalar(eps)),
                    new ScalarOptional(new Scalar(1.0f - eps)));
            Tensor logClipped = clipped.log();
            Tensor oneMinusClipped = predictions.neg().add(new Scalar(1.0f))
                    .clamp(new ScalarOptional(new Scalar(eps)),
                            new ScalarOptional(new Scalar(1.0f - eps)))
                    .log();
            Tensor loss = targets.mul(logClipped.neg())
                    .add(targets.neg().add(new Scalar(1.0f)).mul(oneMinusClipped.neg()));

            switch (reduction) {
                case "sum":
                    return loss.sum();
                case "none":
                    return loss;
                default:
                    return loss.mean();
            }
        }
    }

    public static final class BCEWithLogitsLoss {
        private final String reduction;

        public BCEWithLogitsLoss() {
            this("mean");
        }

        public BCEWithLogitsLoss(String reduction) {
            this.reduction = reduction != null ? reduction : "mean";
        }

        public Tensor apply(Tensor predictions, Tensor targets) {
            var dev = predictions.device();
            Tensor targetsOnDev = !targets.device().equals(dev)
                    ? targets.to(dev, targets.dtype()) : targets;

            Tensor rawLoss;
            if (predictions.dim() == 1 && targetsOnDev.dim() == 2 && targetsOnDev.size(1) == 1) {
                rawLoss = torch.binary_cross_entropy_with_logits(predictions, targetsOnDev.squeeze(1));
            } else if (predictions.dim() == 2 && predictions.size(1) == 1 && targetsOnDev.dim() == 1) {
                rawLoss = torch.binary_cross_entropy_with_logits(predictions.squeeze(1), targetsOnDev);
            } else {
                rawLoss = torch.binary_cross_entropy_with_logits(predictions, targetsOnDev);
            }

            switch (reduction) {
                case "sum":
                    return rawLoss.sum();
                case "none":
                    return rawLoss;
                default:
                    return rawLoss.mean();
            }
        }
    }

    public static final class CrossEntropyLoss {
        private final String reduction;
        private final float labelSmoothing; // kept for API parity; Scala body ignores it

        public CrossEntropyLoss() {
            this("mean", 0.0f);
        }

        public CrossEntropyLoss(String reduction, float labelSmoothing) {
            this.reduction = reduction != null ? reduction : "mean";
            this.labelSmoothing = labelSmoothing;
        }

        public Tensor apply(Tensor predictions, Tensor targets) {
            var dev = predictions.device();
            Tensor targetsOnDev = !targets.device().equals(dev)
                    ? targets.to(dev, targets.dtype()) : targets;
            Tensor loss = torch.cross_entropy(predictions, targetsOnDev);
            switch (reduction) {
                case "sum":
                    return loss.sum();
                case "none":
                    return loss;
                default:
                    return loss.mean();
            }
        }
    }

    /** Pairwise BPR loss (losses package variant). */
    public static final class BPRLoss {
        private final float margin; // kept for API parity; not used in formula

        public BPRLoss() {
            this(1.0f);
        }

        public BPRLoss(float margin) {
            this.margin = margin;
        }

        public Tensor apply(Tensor posScores, Tensor negScores) {
            Tensor diff = posScores.sub(negScores);
            return diff.sigmoid().log().neg().mean();
        }

        public float apply(float posScore, float negScore) {
            float diff = posScore - negScore;
            float sigmoid = 1.0f / (1.0f + (float) Math.exp(-diff));
            return (float) (-Math.log(sigmoid + 1e-8));
        }
    }

    /** Pairwise hinge loss (losses package variant). */
    public static final class HingeLoss {
        private final float margin;

        public HingeLoss() {
            this(1.0f);
        }

        public HingeLoss(float margin) {
            this.margin = margin;
        }

        public Tensor apply(Tensor posScores, Tensor negScores) {
            Tensor diff = posScores.sub(negScores);
            var dev = diff.device();
            Tensor mTensor = torch.full(new long[]{1L}, new Scalar(margin)).to(dev, ScalarType.Float);
            Tensor zeroTensor = torch.full(new long[]{1L}, new Scalar(0.0f)).to(dev, ScalarType.Float);
            return diff.neg().add(mTensor).maximum(zeroTensor).mean();
        }
    }

    public static final class TripletMarginLoss {
        private final float margin;

        public TripletMarginLoss() {
            this(1.0f);
        }

        public TripletMarginLoss(float margin) {
            this.margin = margin;
        }

        public Tensor apply(Tensor anchor, Tensor positive, Tensor negative) {
            Tensor distPos = anchor.sub(positive).pow(new Scalar(2.0f)).sum(1);
            Tensor distNeg = anchor.sub(negative).pow(new Scalar(2.0f)).sum(1);
            var dev = distPos.device();
            Tensor mTensor = torch.full(new long[]{1L}, new Scalar(margin)).to(dev, ScalarType.Float);
            Tensor zeroTensor = torch.full(new long[]{1L}, new Scalar(0.0f)).to(dev, ScalarType.Float);
            return distPos.sub(distNeg).add(mTensor).maximum(zeroTensor).mean();
        }
    }

    /** In-batch NCE with explicit negatives (losses package variant). */
    public static final class InBatchNCELoss {
        private final float temperature;

        public InBatchNCELoss() {
            this(0.07f);
        }

        public InBatchNCELoss(float temperature) {
            this.temperature = temperature;
        }

        public Tensor apply(Tensor userEmbeds, Tensor posItemEmbeds, Tensor negItemEmbeds) {
            Tensor posScores = userEmbeds.mul(posItemEmbeds).sum(1);
            Tensor negScores = torch.bmm(negItemEmbeds, userEmbeds.unsqueeze(2)).squeeze(2);
            TensorVector vec = new TensorVector();
            vec.push_back(posScores.unsqueeze(1));
            vec.push_back(negScores);
            Tensor allScores = torch.cat(vec, 1);
            Tensor scaledScores = allScores.div(new Scalar(temperature));
            Tensor labels = torch.zeros(userEmbeds.size(0))
                    .toType(ScalarType.Long)
                    .to(userEmbeds.device(), ScalarType.Long);
            return torch.cross_entropy(scaledScores, labels);
        }
    }

    public static final class MaskedCrossEntropyLoss {
        public Tensor apply(Tensor logits, Tensor targets, Tensor mask) {
            var dev = logits.device();
            Tensor targetsOnDev = targets.to(dev, ScalarType.Long);
            Tensor loss = torch.cross_entropy(logits, targetsOnDev);
            Tensor maskOnDev = !mask.device().equals(dev) ? mask.to(dev, mask.dtype()) : mask;
            Tensor maskedLoss = loss.mul(maskOnDev);
            return maskedLoss.sum().div(maskOnDev.sum());
        }
    }

    public static final class FocalLoss {
        private final float alpha;
        private final float gamma;

        public FocalLoss() {
            this(0.25f, 2.0f);
        }

        public FocalLoss(float alpha, float gamma) {
            this.alpha = alpha;
            this.gamma = gamma;
        }

        public Tensor apply(Tensor predictions, Tensor targets) {
            var dev = predictions.device();
            Tensor targetsOnDev = !targets.device().equals(dev)
                    ? targets.to(dev, targets.dtype()) : targets;
            Tensor p = predictions.sigmoid();
            Tensor ceLoss = torch.binary_cross_entropy(predictions, targetsOnDev);
            Scalar one = new Scalar(1.0f);
            Tensor pTerm = p.mul(targets).add(p.neg().add(one).mul(targets.neg().add(one)));
            Tensor focalWeight = pTerm.neg().add(one).pow(new Scalar((double) gamma));
            Tensor alphaWeight = targets.mul(new Scalar((double) alpha))
                    .add(targets.neg().add(one).mul(new Scalar(1.0 - alpha)));
            return alphaWeight.mul(focalWeight).mul(ceLoss).mean();
        }
    }
}
