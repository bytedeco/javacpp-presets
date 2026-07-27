package org.bytedeco.pytorch.data.dataframe.dataset;

/**
 * Options controlling how a pure-Java {@link DataFrameDataset} is projected
 * into a native single-{@code Example} / {@code TensorExample} view.
 *
 * <p>Native {@code Example} can only hold one data tensor + one target tensor.
 * Named multi-feature batches stay on {@link DataFrameDataLoader}; these options
 * pick which feature becomes {@code Example.data()}.
 *
 * <ul>
 *   <li>{@link Mode#STACKED_SCALARS} — packed scalar features {@code [n_feat]} (default if any scalars)</li>
 *   <li>{@link Mode#FIRST_SEQUENCE} — first sequence feature</li>
 *   <li>{@link Mode#PRIMARY} — named feature via {@link #primaryFeature(String)}</li>
 *   <li>{@link Mode#AUTO} — scalars if present, else first sequence (default)</li>
 * </ul>
 */
public final class NativeViewOptions {

    public enum Mode {
        AUTO,
        STACKED_SCALARS,
        FIRST_SEQUENCE,
        PRIMARY
    }

    private Mode mode = Mode.AUTO;
    private String primaryFeature;
    private boolean emptyTargetIfMissing = true;

    public NativeViewOptions mode(Mode m) {
        this.mode = m == null ? Mode.AUTO : m;
        return this;
    }

    public NativeViewOptions primaryFeature(String name) {
        this.primaryFeature = name;
        if (name != null && !name.isEmpty()) this.mode = Mode.PRIMARY;
        return this;
    }

    public NativeViewOptions preferScalars(boolean v) {
        this.mode = v ? Mode.STACKED_SCALARS : Mode.FIRST_SEQUENCE;
        return this;
    }

    public NativeViewOptions emptyTargetIfMissing(boolean v) {
        this.emptyTargetIfMissing = v;
        return this;
    }

    public Mode mode() { return mode; }
    public String primaryFeature() { return primaryFeature; }
    public boolean emptyTargetIfMissing() { return emptyTargetIfMissing; }

    public static NativeViewOptions defaults() {
        return new NativeViewOptions();
    }

    public NativeViewOptions copy() {
        NativeViewOptions o = new NativeViewOptions();
        o.mode = this.mode;
        o.primaryFeature = this.primaryFeature;
        o.emptyTargetIfMissing = this.emptyTargetIfMissing;
        return o;
    }
}
