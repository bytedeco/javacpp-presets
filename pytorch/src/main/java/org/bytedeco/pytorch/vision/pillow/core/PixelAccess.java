package org.bytedeco.pytorch.vision.pillow.core;

/**
 * Thin pixel accessor façade (Pillow {@code PixelAccess} / {@code load()} return).
 */
public final class PixelAccess {

    private final ImagingBuffer buffer;

    public PixelAccess(ImagingBuffer buffer) {
        this.buffer = buffer;
    }

    public int[] getpixel(int x, int y) {
        return buffer.getpixel(x, y);
    }

    public void putpixel(int x, int y, int[] color) {
        buffer.putpixel(x, y, color);
    }

    public void putpixel(int x, int y, int gray) {
        buffer.putpixel(x, y, new int[]{gray});
    }

    public ImagingBuffer buffer() {
        return buffer;
    }
}
