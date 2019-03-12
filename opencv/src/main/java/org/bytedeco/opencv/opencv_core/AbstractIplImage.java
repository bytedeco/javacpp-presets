package org.bytedeco.opencv.opencv_core;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.Properties;

import static org.bytedeco.opencv.global.opencv_core.*;

@Properties(inherit = org.bytedeco.opencv.presets.opencv_core.class)
public abstract class AbstractIplImage extends CvArr {
    protected BytePointer pointer; // a reference to prevent deallocation

    public AbstractIplImage(Pointer p) { super(p); }

    /**
     * Calls cvCreateImage(), and registers a deallocator.
     * @return IplImage created. Do not call cvReleaseImage() on it.
     */
    public static IplImage create(CvSize size, int depth, int channels) {
        IplImage i = cvCreateImage(size, depth, channels);
        if (i != null) {
            i.deallocator(new ReleaseDeallocator(i));
        }
        return i;
    }
    /**
     * Calls cvCreateImage(), and registers a deallocator.
     * @return IplImage created. Do not call cvReleaseImage() on it.
     */
    public static IplImage create(int width, int height, int depth, int channels) {
        return create(org.bytedeco.opencv.global.opencv_core.cvSize(width, height), depth, channels);
    }
    /**
     * Calls cvCreateImage(), and registers a deallocator. Also assigns {@link #origin()}.
     * @return IplImage created. Do not call cvReleaseImage() on it.
     */
    public static IplImage create(CvSize size, int depth, int channels, int origin) {
        IplImage i = create(size, depth, channels);
        if (i != null) {
            i.origin(origin);
        }
        return i;
    }
    /**
     * Calls cvCreateImage(), and registers a deallocator. Also assigns {@link #origin()}.
     * @return IplImage created. Do not call cvReleaseImage() on it.
     */
    public static IplImage create(int width, int height, int depth, int channels, int origin) {
        IplImage i = create(width, height, depth, channels);
        if (i != null) {
            i.origin(origin);
        }
        return i;
    }

    /**
     * Calls cvCreateImageHeader(), and registers a deallocator.
     * @return IplImage created. Do not call cvReleaseImageHeader() on it.
     */
    public static IplImage createHeader(CvSize size, int depth, int channels) {
        IplImage i = cvCreateImageHeader(size, depth, channels);
        if (i != null) {
            i.deallocator(new HeaderReleaseDeallocator(i));
        }
        return i;
    }
    /**
     * Calls cvCreateImageHeader(), and registers a deallocator.
     * @return IplImage created. Do not call cvReleaseImageHeader() on it.
     */
    public static IplImage createHeader(int width, int height, int depth, int channels) {
        return createHeader(org.bytedeco.opencv.global.opencv_core.cvSize(width, height), depth, channels);
    }
    /**
     * Calls cvCreateImageHeader(), and registers a deallocator. Also assigns {@link #origin()}.
     * @return IplImage created. Do not call cvReleaseImageHeader() on it.
     */
    public static IplImage createHeader(CvSize size, int depth, int channels, int origin) {
        IplImage i = createHeader(size, depth, channels);
        if (i != null) {
            i.origin(origin);
        }
        return i;
    }
    /**
     * Calls cvCreateImageHeader(), and registers a deallocator. Also assigns {@link #origin()}.
     * @return IplImage created. Do not call cvReleaseImageHeader() on it.
     */
    public static IplImage createHeader(int width, int height, int depth, int channels, int origin) {
        IplImage i = createHeader(width, height, depth, channels);
        if (i != null) {
            i.origin(origin);
        }
        return i;
    }

    /**
     * Calls createHeader(), and initializes data, keeping a reference to prevent deallocation.
     * @return IplImage created. Do not call cvReleaseImageHeader() on it.
     */
    public static IplImage create(int width, int height, int depth, int channels, Pointer data) {
        IplImage i = createHeader(width, height, depth, channels);
        i.imageData(i.pointer = new BytePointer(data));
        return i;
    }

    /**
     * Creates an IplImage based on another IplImage.
     * @return IplImage created. Do not call cvReleaseImage() on it.
     */
    public static IplImage createCompatible(IplImage template) {
        return createIfNotCompatible(null, template);
    }
    /**
     * Creates an IplImage based on another IplImage, unless the template is OK.
     * @return template or IplImage created. Do not call cvReleaseImage() on it.
     */
    public static IplImage createIfNotCompatible(IplImage image, IplImage template) {
        if (image == null || image.width() != template.width() || image.height() != template.height() ||
                image.depth() != template.depth() || image.nChannels() != template.nChannels()) {
            image = create(template.width(), template.height(),
                    template.depth(), template.nChannels(), template.origin());
        }
        image.origin(template.origin());
        return image;
    }

    /**
     * Calls cvCloneImage(), and registers a deallocator.
     * @return IplImage cloned. Do not call cvReleaseImage() on it.
     */
    @Override public IplImage clone() {
        IplImage i = cvCloneImage((IplImage)this);
        if (i != null) {
            i.deallocator(new ReleaseDeallocator(i));
        }
        return i;
    }

    /**
     * Calls the deallocator, if registered, otherwise has no effect.
     */
    public void release() {
        deallocate();
    }
    protected static class ReleaseDeallocator extends IplImage implements Deallocator {
        ReleaseDeallocator(IplImage p) { super(p); }
        @Override public void deallocate() { if (isNull()) return; cvReleaseImage(this); setNull(); }
    }
    protected static class HeaderReleaseDeallocator extends IplImage implements Deallocator {
        HeaderReleaseDeallocator(IplImage p) { super(p); }
        @Override public void deallocate() { if (isNull()) return; cvReleaseImageHeader(this); setNull(); }
    }

    public abstract int nChannels();
    public abstract int depth();
    public abstract int origin(); public abstract IplImage origin(int origin);
    public abstract int width();
    public abstract int height();
    public abstract IplROI roi();
    public abstract int imageSize();
    public abstract BytePointer imageData();
    public abstract int widthStep();

    @Override public int arrayChannels() { return nChannels(); }
    @Override public int arrayDepth() { return depth(); }
    @Override public int arrayOrigin() { return origin(); }
    @Override public void arrayOrigin(int origin) { origin(origin); }
    @Override public int arrayWidth() { return width(); }
    @Override public int arrayHeight() { return height(); }
    @Override public IplROI arrayROI() { return roi(); }
    @Override public long arraySize() { return imageSize(); }
    @Override public BytePointer arrayData() { return imageData(); }
    @Override public long arrayStep() { return widthStep(); }

    public CvMat asCvMat() {
        CvMat mat = new CvMat();
        cvGetMat(this, mat, (IntPointer)null, 0);
        return mat;
    }
}
