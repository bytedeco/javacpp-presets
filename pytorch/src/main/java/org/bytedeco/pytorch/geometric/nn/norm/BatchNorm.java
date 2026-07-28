package org.bytedeco.pytorch.geometric.nn.norm;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.BatchNorm1dImpl;
import org.bytedeco.pytorch.nn.options.BatchNormOptions;

/**
 * BatchNorm for node features [N, C] wrapping LibTorch {@link BatchNorm1dImpl}.
 *
 * <p>When {@code allowSingleElement=true} and N≤1 in train mode, temporarily
 * switches to eval so BN can use running statistics (variance undefined for N=1).
 */
public class BatchNorm extends Module {

    private final BatchNorm1dImpl innerBN;
    private final boolean allowSingleElement;
    private final long inChannels;

    public BatchNorm(long inChannels) {
        this(inChannels, 1e-5, 0.1, true, true, false);
    }

    public BatchNorm(long inChannels, boolean allowSingleElement) {
        this(inChannels, 1e-5, 0.1, true, true, allowSingleElement);
    }

    public BatchNorm(long inChannels, double eps, double momentum,
                     boolean affine, boolean trackRunningStats, boolean allowSingleElement) {
        super();
        if (inChannels <= 0) {
            throw new IllegalArgumentException("inChannels must be > 0");
        }
        this.inChannels = inChannels;
        this.allowSingleElement = allowSingleElement;

        BatchNormOptions options = new BatchNormOptions(inChannels);
        options.eps().put(eps);
        options.momentum().put(momentum);
        options.affine().put(affine);
        options.track_running_stats().put(trackRunningStats);

        this.innerBN = register_module("module", new BatchNorm1dImpl(options));
    }

    /** JavaCPP interop. */
    public BatchNorm(Pointer p) {
        super(p);
        this.innerBN = null;
        this.allowSingleElement = false;
        this.inChannels = 0;
    }

    /**
     * @param x [N, C] (or [N, C, *] if BN1d supports extra dims)
     */
    public Tensor forward(Tensor x) {
        if (x == null) {
            throw new NullPointerException("x must not be null");
        }
        if (innerBN == null) {
            throw new IllegalStateException("BatchNorm not fully constructed (Pointer ctor)");
        }
        long numElements = x.size(0);
        if (numElements <= 1 && allowSingleElement && innerBN.is_training()) {
            boolean wasTraining = true;
            innerBN.eval();
            try {
                return innerBN.forward(x);
            } finally {
                if (wasTraining) {
                    innerBN.train(true);
                }
            }
        }
        return innerBN.forward(x);
    }

    public BatchNorm1dImpl getInnerBN() {
        return innerBN;
    }

    public long getInChannels() {
        return inChannels;
    }

    public boolean isAllowSingleElement() {
        return allowSingleElement;
    }
}
