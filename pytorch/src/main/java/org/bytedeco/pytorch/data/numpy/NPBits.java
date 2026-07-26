package org.bytedeco.pytorch.data.numpy;

/**
 * Bit packing utilities: {@code packbits} / {@code unpackbits}.
 */
public final class NPBits {
    private NPBits() {}

    /**
     * Packs binary values (0/1) of a uint8/bool/int array into bits of a uint8 array.
     * Along {@code axis} (null → ravel first).
     */
    public static NDArray packbits(NDArray a, Integer axis, String bitorder) {
        boolean big = bitorder == null || "big".equalsIgnoreCase(bitorder);
        NDArray src = axis == null ? NPShape.ravel(a) : a;
        int ax = axis == null ? 0 : NPArrayUtil.normalizeAxis(axis, src.shape.length);
        long len = src.shape[ax];
        long packedLen = (len + 7) / 8;
        long[] outShape = src.shape.clone();
        outShape[ax] = packedLen;
        NDArray out = new NDArray(DType.UINT8, outShape);

        long[] sSt = NPArrayUtil.stridesOf(src.shape);
        long[] oSt = NPArrayUtil.stridesOf(outShape);
        long otherN = src.size / Math.max(len, 1);
        long[] otherShape = new long[Math.max(0, src.shape.length - 1)];
        int k = 0;
        for (int d = 0; d < src.shape.length; d++) if (d != ax) otherShape[k++] = src.shape[d];
        long[] otherSt = otherShape.length == 0 ? new long[0] : NPArrayUtil.stridesOf(otherShape);
        int[] idx = new int[src.shape.length];

        for (int o = 0; o < otherN; o++) {
            int p = 0;
            for (int d = 0; d < src.shape.length; d++) {
                if (d == ax) idx[d] = 0;
                else {
                    idx[d] = otherShape.length == 0 ? 0 : (int) ((o / otherSt[p]) % otherShape[p]);
                    p++;
                }
            }
            for (int byteI = 0; byteI < packedLen; byteI++) {
                int val = 0;
                for (int bit = 0; bit < 8; bit++) {
                    long pos = byteI * 8L + bit;
                    if (pos >= len) break;
                    idx[ax] = (int) pos;
                    int bitVal = src.getLong(NPArrayUtil.ravel(idx, sSt)) != 0 ? 1 : 0;
                    if (big) val = (val << 1) | bitVal;
                    else val |= (bitVal << bit);
                }
                if (big) {
                    // if partial last byte, already shifted; ok for full; for partial need left-align:
                    long used = Math.min(8, len - byteI * 8L);
                    if (used < 8) val <<= (8 - used);
                }
                idx[ax] = byteI;
                // map to out index
                int[] oIdx = idx.clone();
                oIdx[ax] = byteI;
                out.setLong(NPArrayUtil.ravel(oIdx, oSt), val & 0xff);
            }
        }
        return out;
    }

    public static NDArray packbits(NDArray a) { return packbits(a, null, "big"); }

    public static NDArray unpackbits(NDArray a, Integer axis, Integer count, String bitorder) {
        boolean big = bitorder == null || "big".equalsIgnoreCase(bitorder);
        NDArray src = axis == null ? NPShape.ravel(a) : a;
        int ax = axis == null ? 0 : NPArrayUtil.normalizeAxis(axis, src.shape.length);
        long len = src.shape[ax];
        long unpacked = count != null ? count : len * 8;
        long[] outShape = src.shape.clone();
        outShape[ax] = unpacked;
        NDArray out = new NDArray(DType.UINT8, outShape);

        long[] sSt = NPArrayUtil.stridesOf(src.shape);
        long[] oSt = NPArrayUtil.stridesOf(outShape);
        long otherN = src.size / Math.max(len, 1);
        long[] otherShape = new long[Math.max(0, src.shape.length - 1)];
        int k = 0;
        for (int d = 0; d < src.shape.length; d++) if (d != ax) otherShape[k++] = src.shape[d];
        long[] otherSt = otherShape.length == 0 ? new long[0] : NPArrayUtil.stridesOf(otherShape);
        int[] idx = new int[src.shape.length];

        for (int o = 0; o < otherN; o++) {
            int p = 0;
            for (int d = 0; d < src.shape.length; d++) {
                if (d == ax) idx[d] = 0;
                else {
                    idx[d] = otherShape.length == 0 ? 0 : (int) ((o / otherSt[p]) % otherShape[p]);
                    p++;
                }
            }
            for (int byteI = 0; byteI < len; byteI++) {
                idx[ax] = byteI;
                int val = (int) src.getLong(NPArrayUtil.ravel(idx, sSt)) & 0xff;
                for (int bit = 0; bit < 8; bit++) {
                    long pos = byteI * 8L + bit;
                    if (pos >= unpacked) break;
                    int bitVal;
                    if (big) bitVal = (val >> (7 - bit)) & 1;
                    else bitVal = (val >> bit) & 1;
                    int[] oIdx = idx.clone();
                    oIdx[ax] = (int) pos;
                    out.setLong(NPArrayUtil.ravel(oIdx, oSt), bitVal);
                }
            }
        }
        return out;
    }

    public static NDArray unpackbits(NDArray a) { return unpackbits(a, null, null, "big"); }
}
