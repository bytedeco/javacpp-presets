package org.bytedeco.pytorch.info;

/**
 * 对应 torch.finfo
 */
public class FInfo extends TypeInfo {
    public final double min;
    public final double max;
    public final double eps;
    public final double tiny;
    public final int resolution;

    public FInfo(String type, int bits, double min, double max, double eps, double tiny, int resolution) {
        super(type, bits);
        this.min = min;
        this.max = max;
        this.eps = eps;
        this.tiny = tiny;
        this.resolution = resolution;
    }

    @Override
    public String toString() {
        return String.format("finfo(type=%s, min=%e, max=%e, eps=%e)", type, min, max, eps);
    }
}