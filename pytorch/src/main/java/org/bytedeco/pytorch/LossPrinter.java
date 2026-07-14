package org.bytedeco.pytorch;

/**
 * Companion printer for Loss subclasses (MSELoss, CrossEntropyLoss,
 * NLLLoss, BCELoss, BCEWithLogitsLoss, L1Loss, SmoothL1Loss). Shows
 * the loss class name plus the key options — reduction, ignore_index,
 * label_smoothing, pos_weight, beta — depending on what's available.
 */
final class LossPrinter {

    private LossPrinter() {}

    static String format(org.bytedeco.pytorch.Module loss) {
        if (loss == null) return "null";
        StringBuilder sb = new StringBuilder();
        sb.append(loss.getClass().getSimpleName());
        sb.append("(");

        if (loss instanceof MSELossImpl) {
            appendMSE(sb, (MSELossImpl) loss);
        } else if (loss instanceof CrossEntropyLossImpl) {
            appendCrossEntropy(sb, (CrossEntropyLossImpl) loss);
        } else if (loss instanceof NLLLossImpl) {
            appendNLL(sb, (NLLLossImpl) loss);
        } else if (loss instanceof BCELossImpl) {
            appendBCE(sb, (BCELossImpl) loss);
        } else if (loss instanceof BCEWithLogitsLossImpl) {
            appendBCEWithLogits(sb, (BCEWithLogitsLossImpl) loss);
        } else if (loss instanceof L1LossImpl) {
            appendL1(sb, (L1LossImpl) loss);
        } else if (loss instanceof SmoothL1LossImpl) {
            appendSmoothL1(sb, (SmoothL1LossImpl) loss);
        } else {
            // Fallback: reflect into the loss's "options" if it exists.
            try {
                java.lang.reflect.Method optsM = loss.getClass().getMethod("options");
                Object opts = optsM.invoke(loss);
                sb.append("reduction=").append(opts);
            } catch (Throwable ignored) {}
        }
        sb.append(")");
        return sb.toString();
    }

    private static void appendMSE(StringBuilder sb, MSELossImpl l) {
        MSELossOptions o = l.options();
        sb.append("reduction=").append(o.reduction());
    }

    private static void appendCrossEntropy(StringBuilder sb, CrossEntropyLossImpl l) {
        CrossEntropyLossOptions o = l.options();
        sb.append("reduction=").append(o.reduction());
        try {
            // ignore_index() returns a LongPointer (single-element
            // ExpandingArray wrapping the c10::optional<int64_t>).
            sb.append(", ignore_index=").append(o.ignore_index().get(0));
        } catch (Throwable ignored) {}
        try {
            // label_smoothing() returns a DoublePointer.
            sb.append(", label_smoothing=").append(o.label_smoothing().get(0));
        } catch (Throwable ignored) {}
    }

    private static void appendNLL(StringBuilder sb, NLLLossImpl l) {
        NLLLossOptions o = l.options();
        sb.append("reduction=").append(o.reduction());
        try {
            sb.append(", ignore_index=").append(o.ignore_index().get(0));
        } catch (Throwable ignored) {}
    }

    private static void appendBCE(StringBuilder sb, BCELossImpl l) {
        BCELossOptions o = l.options();
        sb.append("reduction=").append(o.reduction());
        // BCEWithLogitsLoss has pos_weight (an OptionalTensor in
        // c10), but plain BCELoss only has weight. We skip weight
        // since printing a Tensor value here would be noisy.
    }

    private static void appendBCEWithLogits(StringBuilder sb, BCEWithLogitsLossImpl l) {
        BCEWithLogitsLossOptions o = l.options();
        sb.append("reduction=").append(o.reduction());
    }

    private static void appendL1(StringBuilder sb, L1LossImpl l) {
        L1LossOptions o = l.options();
        sb.append("reduction=").append(o.reduction());
    }

    private static void appendSmoothL1(StringBuilder sb, SmoothL1LossImpl l) {
        SmoothL1LossOptions o = l.options();
        sb.append("reduction=").append(o.reduction());
        try {
            // beta is a double in DoubleOptional.
            org.bytedeco.pytorch.DoubleOptional p = o.beta();
            if (p != null) {
                sb.append(", beta=").append(p.get());
            }
        } catch (Throwable ignored) {}
    }
}
