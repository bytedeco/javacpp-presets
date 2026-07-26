package org.bytedeco.pytorch.data.numpy;

import java.util.Arrays;

/**
 * NumPy-like masked array: data + boolean mask (true = invalid / masked).
 */
public final class MaskedArray {
    public final NDArray data;
    /** true where value is masked (invalid). Same shape as data. */
    public final NDArray mask;
    public final double fillValue;

    public MaskedArray(NDArray data, NDArray mask) {
        this(data, mask, Double.NaN);
    }

    public MaskedArray(NDArray data, NDArray mask, double fillValue) {
        if (data == null) throw new IllegalArgumentException("data");
        this.data = data;
        if (mask == null) {
            this.mask = NP.zeros(DType.BOOL, data.shape);
        } else {
            if (mask.size != data.size) throw new IllegalArgumentException("mask size mismatch");
            NDArray m = mask.dtype == DType.BOOL ? mask : NP.astype(mask, DType.BOOL);
            if (!Arrays.equals(m.shape, data.shape)) m = NP.reshape(m, data.shape);
            this.mask = m;
        }
        this.fillValue = fillValue;
    }

    public static MaskedArray masked_array(NDArray data, NDArray mask) {
        return new MaskedArray(data, mask);
    }

    public static MaskedArray masked_array(NDArray data, NDArray mask, double fill_value) {
        return new MaskedArray(data, mask, fill_value);
    }

    public static MaskedArray masked_where(NDArray condition, NDArray data) {
        return new MaskedArray(data, condition);
    }

    public static MaskedArray masked_equal(NDArray data, double value) {
        NDArray m = NP.equal(data, NP.full(value, data.shape));
        return new MaskedArray(data, m);
    }

    public static MaskedArray masked_invalid(NDArray data) {
        NDArray m = NP.logical_or(NP.isnan(data), NP.isinf(data));
        return new MaskedArray(data, m);
    }

    public static MaskedArray masked_greater(NDArray data, double value) {
        return new MaskedArray(data, NP.greater(data, NP.full(value, data.shape)));
    }

    public static MaskedArray masked_less(NDArray data, double value) {
        return new MaskedArray(data, NP.less(data, NP.full(value, data.shape)));
    }

    public long count() {
        long n = 0;
        for (int i = 0; i < mask.size; i++) if (mask.getLong(i) == 0) n++;
        return n;
    }

    public long count_masked() { return mask.size - count(); }

    public NDArray filled() { return filled(fillValue); }

    public NDArray filled(double value) {
        NDArray out = NP.copy(data);
        for (int i = 0; i < out.size; i++) {
            if (mask.getLong(i) != 0) out.setDouble(i, value);
        }
        return out;
    }

    public NDArray compressed() {
        int n = (int) count();
        NDArray out = new NDArray(data.dtype, n);
        int k = 0;
        for (int i = 0; i < data.size; i++) {
            if (mask.getLong(i) == 0) out.setDouble(k++, data.getDouble(i));
        }
        return out;
    }

    public double sum() {
        double s = 0;
        for (int i = 0; i < data.size; i++) if (mask.getLong(i) == 0) s += data.getDouble(i);
        return s;
    }

    public double mean() {
        long n = count();
        return n == 0 ? Double.NaN : sum() / n;
    }

    public double min() {
        double m = Double.POSITIVE_INFINITY;
        boolean any = false;
        for (int i = 0; i < data.size; i++) {
            if (mask.getLong(i) != 0) continue;
            m = Math.min(m, data.getDouble(i));
            any = true;
        }
        return any ? m : Double.NaN;
    }

    public double max() {
        double m = Double.NEGATIVE_INFINITY;
        boolean any = false;
        for (int i = 0; i < data.size; i++) {
            if (mask.getLong(i) != 0) continue;
            m = Math.max(m, data.getDouble(i));
            any = true;
        }
        return any ? m : Double.NaN;
    }

    public MaskedArray add(MaskedArray other) {
        NDArray d = NP.add(this.data, other.data);
        NDArray m = NP.logical_or(this.mask, other.mask);
        return new MaskedArray(d, m, fillValue);
    }

    public MaskedArray multiply(MaskedArray other) {
        NDArray d = NP.multiply(this.data, other.data);
        NDArray m = NP.logical_or(this.mask, other.mask);
        return new MaskedArray(d, m, fillValue);
    }

    public MaskedArray copy() {
        return new MaskedArray(NP.copy(data), NP.copy(mask), fillValue);
    }

    @Override
    public String toString() {
        return "MaskedArray(data=" + data + ", masked=" + count_masked() + "/" + data.size
                + ", fill_value=" + fillValue + ")";
    }
}
