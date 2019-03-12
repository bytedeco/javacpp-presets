package org.bytedeco.opencv.opencv_core;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.indexer.*;

import static org.bytedeco.opencv.global.opencv_core.*;

@Properties(inherit = org.bytedeco.opencv.presets.opencv_core.class)
public abstract class AbstractMat extends AbstractArray {
    public AbstractMat(Pointer p) { super(p); }

    public abstract void create(int rows, int cols, int type);
    public abstract void release();
    public abstract int type();
    public abstract int depth();
    public abstract int channels();
    public abstract int rows();
    public abstract int cols();
    public abstract BytePointer data();
    public abstract int size(int i);
    public abstract long step(int i);
    public abstract int dims();
    public abstract long elemSize1();

    @Override public int arrayChannels() { return channels(); }
    @Override public int arrayDepth() {
        switch (depth()) {
            case CV_8U : return IPL_DEPTH_8U;
            case CV_8S : return IPL_DEPTH_8S;
            case CV_16U: return IPL_DEPTH_16U;
            case CV_16S: return IPL_DEPTH_16S;
            case CV_32S: return IPL_DEPTH_32S;
            case CV_32F: return IPL_DEPTH_32F;
            case CV_64F: return IPL_DEPTH_64F;
            default: assert (false);
        }
        return -1;
    }
    @Override public int arrayOrigin() { return 0; }
    @Override public void arrayOrigin(int origin) { }
    @Override public int arrayWidth() { return cols(); }
    @Override public int arrayHeight() { return rows(); }
    @Override public IplROI arrayROI() { return null; }
    @Override public long arraySize() { return step(0)*size(0); }
    @Override public BytePointer arrayData() { return data(); }
    @Override public long arrayStep() { return step(0); }

    public static final Mat EMPTY = null;

    @Override public <I extends Indexer> I createIndexer(boolean direct) {
        BytePointer ptr = arrayData();
        long size = arraySize();
        int dims = dims();
        int depth = depth();
        long elemSize = elemSize1();

        long[] sizes = new long[dims+1];
        long[] strides = new long[dims+1];

        for (int i=0; i<dims; i++) {
            sizes[i] = size(i);
            long step = step(i);
            if (step%elemSize != 0) {
                throw new UnsupportedOperationException("Step is not a multiple of element size");
            }
            strides[i] = step/elemSize;
        }
        sizes[dims] = arrayChannels();
        strides[dims] = 1;
        switch (depth) {
            case CV_8U:
                return (I)UByteIndexer.create(ptr.capacity(size), sizes, strides, direct).indexable(this);
            case CV_8S:
                return (I)ByteIndexer.create(ptr.capacity(size), sizes, strides, direct).indexable(this);
            case CV_16U:
                return (I)UShortIndexer.create(new ShortPointer(ptr).capacity(size/2), sizes, strides, direct).indexable(this);
            case CV_16S:
                return (I)ShortIndexer.create(new ShortPointer(ptr).capacity(size/2), sizes, strides, direct).indexable(this);
            case CV_32S:
                return (I)IntIndexer.create(new IntPointer(ptr).capacity(size/4), sizes, strides, direct).indexable(this);
            case CV_32F:
                return (I)FloatIndexer.create(new FloatPointer(ptr).capacity(size/4), sizes, strides, direct).indexable(this);
            case CV_64F:
                return (I)DoubleIndexer.create(new DoublePointer(ptr).capacity(size/8), sizes, strides, direct).indexable(this);
            default: assert false;
        }
        return null;
    }
}
