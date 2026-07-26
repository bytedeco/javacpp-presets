package org.bytedeco.pytorch.data.numpy;

import java.util.Arrays;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Parsed NumPy {@code .npy} header ({@code descr}, {@code fortran_order}, {@code shape}). */
public final class NpyHeader {
    public final DType dtype;
    public final boolean fortranOrder;
    public final long[] shape;

    public NpyHeader(DType dtype, boolean fortranOrder, long[] shape) {
        this.dtype = dtype != null ? dtype : DType.FLOAT64;
        this.fortranOrder = fortranOrder;
        this.shape = shape != null ? shape.clone() : new long[0];
    }

    public NpyHeader(DType dtype, boolean fortranOrder, int[] shape) {
        this(dtype, fortranOrder, toLong(shape));
    }

    public static NpyHeader parse(String headerStr) {
        String desc = extract(headerStr, "'descr'\\s*:\\s*'([^']+)'");
        if (desc.isEmpty()) desc = extract(headerStr, "\"descr\"\\s*:\\s*\"([^\"]+)\"");
        DType dtype = DType.fromDescriptor(desc);

        String fo = extract(headerStr, "'fortran_order'\\s*:\\s*(True|False)");
        if (fo.isEmpty()) fo = extract(headerStr, "\"fortran_order\"\\s*:\\s*(true|false)");
        boolean fortranOrder = "True".equalsIgnoreCase(fo) || "true".equals(fo);

        String shapeRaw = extract(headerStr, "'shape'\\s*:\\s*\\(([^)]*)\\)");
        if (shapeRaw.isEmpty()) shapeRaw = extract(headerStr, "\"shape\"\\s*:\\s*\\[([^\\]]*)\\]");
        long[] shape;
        if (shapeRaw.isEmpty()) {
            shape = new long[0];
        } else {
            shape = Arrays.stream(shapeRaw.split(","))
                    .map(String::trim)
                    .filter(s -> !s.isEmpty())
                    .mapToLong(Long::parseLong)
                    .toArray();
        }
        return new NpyHeader(dtype, fortranOrder, shape);
    }

    public String toHeaderString() {
        StringBuilder shapeStr = new StringBuilder("(");
        for (int i = 0; i < shape.length; i++) {
            if (i > 0) shapeStr.append(", ");
            shapeStr.append(shape[i]);
        }
        if (shape.length == 1) shapeStr.append(',');
        shapeStr.append(')');
        return "{'descr': '" + dtype.getDescriptor()
                + "', 'fortran_order': " + (fortranOrder ? "True" : "False")
                + ", 'shape': " + shapeStr + ", }";
    }

    private static String extract(String text, String regex) {
        Matcher m = Pattern.compile(regex).matcher(text);
        return m.find() ? m.group(1) : "";
    }

    private static long[] toLong(int[] a) {
        if (a == null) return new long[0];
        long[] o = new long[a.length];
        for (int i = 0; i < a.length; i++) o[i] = a[i];
        return o;
    }

    public long numel() {
        long n = 1;
        for (long s : shape) n *= s;
        return n;
    }
}
