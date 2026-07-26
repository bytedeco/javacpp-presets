package org.bytedeco.pytorch.info;

/**
 * 对应 torch.iinfo
 */
public class IInfo extends TypeInfo {
    public final long min;
    public final long max;

    public IInfo(String type, int bits, long min, long max) {
        super(type, bits);
        this.min = min;
        this.max = max;
    }

    @Override
    public String toString() {
        return String.format("iinfo(type=%s, min=%d, max=%d)", type, min, max);
    }
}
