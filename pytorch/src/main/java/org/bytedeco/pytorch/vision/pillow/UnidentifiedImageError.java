package org.bytedeco.pytorch.vision.pillow;

/**
 * Raised when Pillow cannot identify / open an image (mirrors PIL.UnidentifiedImageError).
 */
public class UnidentifiedImageError extends RuntimeException {
    public UnidentifiedImageError(String message) {
        super(message);
    }

    public UnidentifiedImageError(String message, Throwable cause) {
        super(message, cause);
    }
}
