package org.bytedeco.pytorch.vision.pillow.tensor;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.dataframe.dtype.ImageData;
import org.bytedeco.pytorch.vision.pillow.Image;
import org.bytedeco.pytorch.vision.pillow.core.ImagingBuffer;
import org.bytedeco.pytorch.vision.utils.ImageTensors;

import java.awt.image.BufferedImage;
import java.util.Objects;

/**
 * Bridges Pillow {@link Image} ↔ {@link Tensor} / {@link ImageData}
 * by reusing {@link ImageTensors} (CHW float {@code [0,1]}).
 */
public final class PillowTensors {

    private PillowTensors() {}

    /** {@code [C,H,W]} float in {@code [0,1]} (alpha dropped like ImageTensors). */
    public static Tensor toTensor(Image image) {
        Objects.requireNonNull(image, "image");
        BufferedImage bi = image.toBufferedImage();
        return ImageTensors.toTensor(bi);
    }

    public static Tensor to_tensor(Image image) {
        return toTensor(image);
    }

    public static Image fromTensor(Tensor t) {
        return fromTensor(t, null);
    }

    public static Image from_tensor(Tensor t) {
        return fromTensor(t, null);
    }

    public static Image fromTensor(Tensor t, String mode) {
        Objects.requireNonNull(t, "tensor");
        BufferedImage bi = ImageTensors.toBufferedImage(t);
        Image im = Image.fromBufferedImage(bi);
        if (mode != null && !mode.equals(im.mode())) {
            return im.convert(mode);
        }
        return im;
    }

    public static ImageData toImageData(Image image) {
        Objects.requireNonNull(image, "image");
        ImageData id = new ImageData(image.toBufferedImage());
        if (image.format() != null) {
            id.setFormat(image.format().toLowerCase());
        }
        return id;
    }

    public static ImageData to_image_data(Image image) {
        return toImageData(image);
    }

    public static Image fromImageData(ImageData id) {
        Objects.requireNonNull(id, "imageData");
        BufferedImage bi = id.getImage();
        if (bi == null) {
            throw new IllegalArgumentException("ImageData has no BufferedImage loaded");
        }
        return Image.fromBufferedImage(bi);
    }

    public static Image from_image_data(ImageData id) {
        return fromImageData(id);
    }

    /** Direct path from ImagingBuffer without going through Image. */
    public static Tensor bufferToTensor(ImagingBuffer buf) {
        Objects.requireNonNull(buf, "buf");
        return ImageTensors.toTensor(buf.toBufferedImage());
    }

    public static ImagingBuffer tensorToBuffer(Tensor t, String mode) {
        Image im = fromTensor(t, mode);
        return im.getImagingBuffer().copy();
    }
}
