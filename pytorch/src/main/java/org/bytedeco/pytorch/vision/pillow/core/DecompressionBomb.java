package org.bytedeco.pytorch.vision.pillow.core;

/**
 * Pillow-aligned decompression bomb guard ({@code Image.MAX_IMAGE_PIXELS}).
 */
public final class DecompressionBomb {

    /** Same default order of magnitude as Pillow (~89M pixels). */
    public static long MAX_IMAGE_PIXELS = 89_478_485L;

    private DecompressionBomb() {}

    public static void check(int width, int height) {
        check((long) width * (long) height);
    }

    public static void check(long pixels) {
        if (MAX_IMAGE_PIXELS > 0 && pixels > MAX_IMAGE_PIXELS) {
            throw new DecompressionBombError(
                    "image size (" + pixels + " pixels) exceeds limit of " + MAX_IMAGE_PIXELS);
        }
        if (MAX_IMAGE_PIXELS > 0 && pixels > MAX_IMAGE_PIXELS / 2) {
            // Soft warning path: log once via stderr to avoid silent large allocs.
            System.err.println("DecompressionBombWarning: image pixels=" + pixels
                    + " > half of MAX_IMAGE_PIXELS=" + MAX_IMAGE_PIXELS);
        }
    }

    public static void setMaxImagePixels(long max) {
        MAX_IMAGE_PIXELS = max;
    }

    public static long getMaxImagePixels() {
        return MAX_IMAGE_PIXELS;
    }

    /** Hard failure, mirrors Pillow {@code DecompressionBombError}. */
    public static final class DecompressionBombError extends RuntimeException {
        public DecompressionBombError(String message) {
            super(message);
        }
    }

    /** Soft warning marker (not thrown by default). */
    public static final class DecompressionBombWarning extends RuntimeException {
        public DecompressionBombWarning(String message) {
            super(message);
        }
    }
}
