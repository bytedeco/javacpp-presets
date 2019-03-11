package org.bytedeco.opencv.opencv_core;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.ShortBuffer;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.indexer.*;

import static org.bytedeco.opencv.global.opencv_core.*;

@Properties(inherit = org.bytedeco.opencv.presets.opencv_core.class)
public abstract class AbstractArray extends Pointer implements Indexable {
    static { Loader.load(); }
    public AbstractArray(Pointer p) { super(p); }

    public abstract int arrayChannels();
    public abstract int arrayDepth();
    public abstract int arrayOrigin();
    public abstract void arrayOrigin(int origin);
    public abstract int arrayWidth();
    public abstract int arrayHeight();
    public abstract IplROI arrayROI();
    public abstract long arraySize();
    public abstract BytePointer arrayData();
    public abstract long arrayStep();

    /** @return {@code createBuffer(0)} */
    public <B extends Buffer> B createBuffer() {
        return (B)createBuffer(0);
    }
    /** @return {@link #arrayData()} wrapped in a {@link Buffer} of appropriate type starting at given index */
    public <B extends Buffer> B createBuffer(int index) {
        BytePointer ptr = arrayData();
        long size = arraySize();
        switch (arrayDepth()) {
            case IPL_DEPTH_8U:
            case IPL_DEPTH_8S:  return (B)ptr.position(index).capacity(size).asBuffer();
            case IPL_DEPTH_16U:
            case IPL_DEPTH_16S: return (B)new ShortPointer(ptr).position(index).capacity(size/2).asBuffer();
            case IPL_DEPTH_32S: return (B)new IntPointer(ptr).position(index).capacity(size/4).asBuffer();
            case IPL_DEPTH_32F: return (B)new FloatPointer(ptr).position(index).capacity(size/4).asBuffer();
            case IPL_DEPTH_64F: return (B)new DoublePointer(ptr).position(index).capacity(size/8).asBuffer();
            case IPL_DEPTH_1U:
            default: assert false;
        }
        return null;
    }

    /** @return {@code createIndexer(true)} */
    public <I extends Indexer> I createIndexer() {
        return (I)createIndexer(true);
    }
    @Override public <I extends Indexer> I createIndexer(boolean direct) {
        BytePointer ptr = arrayData();
        long size = arraySize();
        long[] sizes = { arrayHeight(), arrayWidth(), arrayChannels() };
        long[] strides = { arrayStep(), arrayChannels(), 1 };
        switch (arrayDepth()) {
            case IPL_DEPTH_8U:
                return (I)UByteIndexer.create(ptr.capacity(size), sizes, strides, direct).indexable(this);
            case IPL_DEPTH_8S:
                return (I)ByteIndexer.create(ptr.capacity(size), sizes, strides, direct).indexable(this);
            case IPL_DEPTH_16U:
                strides[0] /= 2;
                return (I)UShortIndexer.create(new ShortPointer(ptr).capacity(size/2), sizes, strides, direct).indexable(this);
            case IPL_DEPTH_16S:
                strides[0] /= 2;
                return (I)ShortIndexer.create(new ShortPointer(ptr).capacity(size/2), sizes, strides, direct).indexable(this);
            case IPL_DEPTH_32S:
                strides[0] /= 4;
                return (I)IntIndexer.create(new IntPointer(ptr).capacity(size/4), sizes, strides, direct).indexable(this);
            case IPL_DEPTH_32F:
                strides[0] /= 4;
                return (I)FloatIndexer.create(new FloatPointer(ptr).capacity(size/4), sizes, strides, direct).indexable(this);
            case IPL_DEPTH_64F:
                strides[0] /= 8;
                return (I)DoubleIndexer.create(new DoublePointer(ptr).capacity(size/8), sizes, strides, direct).indexable(this);
            case IPL_DEPTH_1U:
            default: assert false;
        }
        return null;
    }

    public double highValue() {
        double highValue = 0.0;
        switch (arrayDepth()) {
            case IPL_DEPTH_8U:  highValue = 0xFF;              break;
            case IPL_DEPTH_16U: highValue = 0xFFFF;            break;
            case IPL_DEPTH_8S:  highValue = Byte.MAX_VALUE;    break;
            case IPL_DEPTH_16S: highValue = Short.MAX_VALUE;   break;
            case IPL_DEPTH_32S: highValue = Integer.MAX_VALUE; break;
            case IPL_DEPTH_1U:
            case IPL_DEPTH_32F:
            case IPL_DEPTH_64F: highValue = 1.0; break;
            default: assert false;
        }
        return highValue;
    }

    public CvSize cvSize() { return org.bytedeco.opencv.global.opencv_core.cvSize(arrayWidth(), arrayHeight()); }

    /** @see #createBuffer(int) */
    @Deprecated public ByteBuffer   getByteBuffer  (int index) { return arrayData().position(index).capacity(arraySize()).asByteBuffer(); }
    /** @see #createBuffer(int) */
    @Deprecated public ShortBuffer  getShortBuffer (int index) { return getByteBuffer(index*2).asShortBuffer();  }
    /** @see #createBuffer(int) */
    @Deprecated public IntBuffer    getIntBuffer   (int index) { return getByteBuffer(index*4).asIntBuffer();    }
    /** @see #createBuffer(int) */
    @Deprecated public FloatBuffer  getFloatBuffer (int index) { return getByteBuffer(index*4).asFloatBuffer();  }
    /** @see #createBuffer(int) */
    @Deprecated public DoubleBuffer getDoubleBuffer(int index) { return getByteBuffer(index*8).asDoubleBuffer(); }
    /** @see #createBuffer() */
    @Deprecated public ByteBuffer   getByteBuffer()   { return getByteBuffer  (0); }
    /** @see #createBuffer() */
    @Deprecated public ShortBuffer  getShortBuffer()  { return getShortBuffer (0); }
    /** @see #createBuffer() */
    @Deprecated public IntBuffer    getIntBuffer()    { return getIntBuffer   (0); }
    /** @see #createBuffer() */
    @Deprecated public FloatBuffer  getFloatBuffer()  { return getFloatBuffer (0); }
    /** @see #createBuffer() */
    @Deprecated public DoubleBuffer getDoubleBuffer() { return getDoubleBuffer(0); }

    @Override public String toString() {
        if (isNull()) {
            return super.toString();
        } else {
            try {
                return getClass().getName() + "[width=" + arrayWidth() + ",height=" + arrayHeight()
                                            + ",depth=" + arrayDepth() + ",channels=" + arrayChannels() + "]";
            } catch (Exception e) {
                return super.toString();
            }
        }
    }
}
