package org.bytedeco.pytorch.plot.tensorboard;

/**
 * Castagnoli CRC-32C used by TensorBoard / TFRecord event files.
 */
public final class Crc32C {
    private static final long POLY = 0x82f63b78L;
    private static final long[] TABLE = new long[256];

    static {
        for (int n = 0; n < 256; n++) {
            long crc = n;
            for (int i = 0; i < 8; i++) {
                crc = ((crc & 1) == 1) ? ((crc >>> 1) ^ POLY) : (crc >>> 1);
            }
            TABLE[n] = crc;
        }
    }

    private Crc32C() {}

    public static long crc32c(byte[] data) {
        long crc = 0xffffffffL;
        for (byte b : data) {
            int index = ((int) crc ^ b) & 0xff;
            crc = (crc >>> 8) ^ TABLE[index];
        }
        return crc ^ 0xffffffffL;
    }

    /** TensorBoard masked CRC32C (little-endian u32 on the wire). */
    public static int maskedCrc32c(byte[] data) {
        long crc = crc32c(data);
        long shifted = (crc >>> 15) | (crc << 17);
        return (int) (shifted + 0xa282ead8L);
    }
}
