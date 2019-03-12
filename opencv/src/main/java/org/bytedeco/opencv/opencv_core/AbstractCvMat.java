package org.bytedeco.opencv.opencv_core;

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.ShortBuffer;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.Properties;

import static org.bytedeco.opencv.global.opencv_core.*;

@Properties(inherit = org.bytedeco.opencv.presets.opencv_core.class)
public abstract class AbstractCvMat extends CvArr {
    protected BytePointer pointer; // a reference to prevent deallocation

    public AbstractCvMat(Pointer p) { super(p); }

    /**
     * Calls cvCreateMat(), and registers a deallocator.
     * @return CvMat created. Do not call cvReleaseMat() on it.
     */
    public static CvMat create(int rows, int cols, int type) {
        CvMat m = cvCreateMat(rows, cols, type);
        if (m != null) {
            ((AbstractCvMat)m).fullSize = m.size();
            ((AbstractCvMat)m).deallocator(new ReleaseDeallocator(m));
        }
        return m;
    }
    /**
     * Calls cvCreateMat(), and registers a deallocator.
     * @return CvMat created. Do not call cvReleaseMat() on it.
     */
    public static CvMat create(int rows, int cols, int depth, int channels) {
        return create(rows, cols, CV_MAKETYPE(depth, channels));
    }
    /**
     * Calls cvCreateMat(rows, cols, CV_64F, 1), and registers a deallocator.
     * @return CvMat created. Do not call cvReleaseMat() on it.
     */
    public static CvMat create(int rows, int cols) {
        return create(rows, cols, CV_64F, 1);
    }

    /**
     * Calls cvCreateMatHeader(), and registers a deallocator.
     * @return CvMat created. Do not call cvReleaseMat() on it.
     */
    public static CvMat createHeader(int rows, int cols, int type) {
        CvMat m = cvCreateMatHeader(rows, cols, type);
        if (m != null) {
            ((AbstractCvMat)m).fullSize = m.size();
            ((AbstractCvMat)m).deallocator(new ReleaseDeallocator(m));
        }
        return m;
    }
    /**
     * Calls cvCreateMatHeader(), and registers a deallocator.
     * @return CvMat created. Do not call cvReleaseMat() on it.
     */
    public static CvMat createHeader(int rows, int cols, int depth, int channels) {
        return createHeader(rows, cols, CV_MAKETYPE(depth, channels));
    }
    /**
     * Calls cvCreateMatHeader(rows, cols, CV_64F, 1), and registers a deallocator.
     * @return CvMat created. Do not call cvReleaseMat() on it.
     */
    public static CvMat createHeader(int rows, int cols) {
        return createHeader(rows, cols, CV_64F, 1);
    }

    /**
     * Calls createHeader(), and initializes data, keeping a reference to prevent deallocation.
     * @return CvMat created. Do not call cvReleaseMat() on it.
     */
    public static CvMat create(int rows, int cols, int depth, int channels, Pointer data) {
        CvMat m = createHeader(rows, cols, depth, channels);
        m.data_ptr(m.pointer = new BytePointer(data));
        return m;
    }

    public static ThreadLocal<CvMat> createThreadLocal(final int rows, final int cols, final int type) {
        return new ThreadLocal<CvMat>() { @Override protected CvMat initialValue() {
            return AbstractCvMat.create(rows, cols, type);
        }};
    }
    public static ThreadLocal<CvMat> createThreadLocal(int rows, int cols, int depth, int channels) {
        return createThreadLocal(rows, cols, CV_MAKETYPE(depth, channels));
    }
    public static ThreadLocal<CvMat> createThreadLocal(int rows, int cols) {
        return createThreadLocal(rows, cols, CV_64F, 1);
    }

    public static ThreadLocal<CvMat> createHeaderThreadLocal(final int rows, final int cols, final int type) {
        return new ThreadLocal<CvMat>() { @Override protected CvMat initialValue() {
            return AbstractCvMat.createHeader(rows, cols, type);
        }};
    }
    public static ThreadLocal<CvMat> createHeaderThreadLocal(int rows, int cols, int depth, int channels) {
        return createHeaderThreadLocal(rows, cols, CV_MAKETYPE(depth, channels));
    }
    public static ThreadLocal<CvMat> createHeaderThreadLocal(int rows, int cols) {
        return createHeaderThreadLocal(rows, cols, CV_64F, 1);
    }

    /**
     * Calls cvCloneMat(), and registers a deallocator.
     * @return CvMat cloned. Do not call cvReleaseMat() on it.
     */
    @Override public CvMat clone() {
        CvMat m = cvCloneMat((CvMat)this);
        if (m != null) {
            m.deallocator(new ReleaseDeallocator(m));
        }
        return m;
    }

    /**
     * Calls the deallocator, if registered, otherwise has no effect.
     */
    public void release() {
        deallocate();
    }
    protected static class ReleaseDeallocator extends CvMat implements Deallocator {
        ReleaseDeallocator(CvMat m) { super(m); }
        @Override public void deallocate() { if (isNull()) return; cvReleaseMat(this); setNull(); }
    }

    public abstract int type(); public abstract CvMat type(int type);
    public abstract int step();
    public abstract BytePointer   data_ptr();
    public abstract FloatPointer  data_fl();
    public abstract DoublePointer data_db();
    public abstract IntPointer    data_i();
    public abstract ShortPointer  data_s();
    public abstract int rows();
    public abstract int cols();

    public int matType() {
        return CV_MAT_TYPE(type());
    }
    public void type(int depth, int cn) {
        type(CV_MAKETYPE(depth, cn) | CV_MAT_MAGIC_VAL);
    }
    public int depth() {
        return CV_MAT_DEPTH(type());
    }
    public int channels() {
        return CV_MAT_CN(type());
    }
    public int nChannels() {
        return CV_MAT_CN(type());
    }
    public boolean isContinuous() {
        return CV_IS_MAT_CONT(type()) != 0;
    }
    public int elemSize() {
        switch (depth()) {
            case CV_8U:
            case CV_8S:  return 1;
            case CV_16U:
            case CV_16S: return 2;
            case CV_32S:
            case CV_32F: return 4;
            case CV_64F: return 8;
            default: assert false;
        }
        return 0;
    }
    public int length() {
        return rows()*cols();
    }
    public int total() {
        return rows()*cols();
    }
    public boolean empty() {
        return length() == 0;
    }
    public int size() {
        // step == 0 when height == 1...
        int rows = rows();
        return cols()*elemSize()*channels() + (rows > 1 ? step()*(rows-1) : 0);
    }

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
    @Override public long arraySize() { return size(); }
    @Override public BytePointer arrayData() { return data_ptr(); }
    @Override public long arrayStep() { return step(); }

    /** @see #createBuffer() */
    @Deprecated public void reset() {
        fullSize = 0;
        byteBuffer = null;
        shortBuffer = null;
        intBuffer = null;
        floatBuffer = null;
        doubleBuffer = null;
    }

    private int fullSize = 0;
    private int fullSize() { return fullSize > 0 ? fullSize : (fullSize = size()); }
    private ByteBuffer byteBuffer = null;
    private ShortBuffer shortBuffer = null;
    private IntBuffer intBuffer = null;
    private FloatBuffer floatBuffer = null;
    private DoubleBuffer doubleBuffer = null;
    @Deprecated public ByteBuffer getByteBuffer() {
        if (byteBuffer == null) {
            byteBuffer = data_ptr().capacity(fullSize()).asBuffer();
        }
        byteBuffer.position(0);
        return byteBuffer;
    }
    @Deprecated public ShortBuffer getShortBuffer() {
        if (shortBuffer == null) {
            shortBuffer = data_s().capacity(fullSize()/2).asBuffer();
        }
        shortBuffer.position(0);
        return shortBuffer;
    }
    @Deprecated public IntBuffer getIntBuffer() {
        if (intBuffer == null) {
            intBuffer = data_i().capacity(fullSize()/4).asBuffer();
        }
        intBuffer.position(0);
        return intBuffer;
    }
    @Deprecated public FloatBuffer getFloatBuffer() {
        if (floatBuffer == null) {
            floatBuffer = data_fl().capacity(fullSize()/4).asBuffer();
        }
        floatBuffer.position(0);
        return floatBuffer;
    }
    @Deprecated public DoubleBuffer getDoubleBuffer() {
        if (doubleBuffer == null) {
            doubleBuffer = data_db().capacity(fullSize()/8).asBuffer();
        }
        doubleBuffer.position(0);
        return doubleBuffer;
    }

    /** @see #createIndexer() */
    @Deprecated public double get(int i) {
        switch (depth()) {
            case CV_8U:  return getByteBuffer()  .get(i)&0xFF;
            case CV_8S:  return getByteBuffer()  .get(i);
            case CV_16U: return getShortBuffer() .get(i)&0xFFFF;
            case CV_16S: return getShortBuffer() .get(i);
            case CV_32S: return getIntBuffer()   .get(i);
            case CV_32F: return getFloatBuffer() .get(i);
            case CV_64F: return getDoubleBuffer().get(i);
            default: assert false;
        }
        return Double.NaN;
    }
    /** @see #createIndexer() */
    @Deprecated public double get(int i, int j) {
        return get(i*step()/elemSize() + j*channels());
    }

    /** @see #createIndexer() */
    @Deprecated public double get(int i, int j, int k) {
        return get(i*step()/elemSize() + j*channels() + k);
    }
    /** @see #createIndexer() */
    @Deprecated public synchronized CvMat get(int index, double[] vv, int offset, int length) {
        int d = depth();
        switch (d) {
            case CV_8U:
            case CV_8S:
                ByteBuffer bb = getByteBuffer();
                bb.position(index);
                for (int i = 0; i < length; i++) {
                    if (d == CV_8U) {
                        vv[i+offset] = bb.get(i)&0xFF;
                    } else {
                        vv[i+offset] = bb.get(i);
                    }
                }
                break;
            case CV_16U:
            case CV_16S:
                ShortBuffer sb = getShortBuffer();
                sb.position(index);
                for (int i = 0; i < length; i++) {
                    if (d == CV_16U) {
                        vv[i+offset] = sb.get()&0xFFFF;
                    } else {
                        vv[i+offset] = sb.get();
                    }
                }
                break;
            case CV_32S:
                IntBuffer ib = getIntBuffer();
                ib.position(index);
                for (int i = 0; i < length; i++) {
                    vv[i+offset] = ib.get();
                }
                break;
            case CV_32F:
                FloatBuffer fb = getFloatBuffer();
                fb.position(index);
                for (int i = 0; i < length; i++) {
                    vv[i+offset] = fb.get();
                }
                break;
            case CV_64F:
                getDoubleBuffer().position(index);
                getDoubleBuffer().get(vv, offset, length);
                break;
            default: assert false;
        }
        return (CvMat)this;
    }
    /** @see #createIndexer() */
    @Deprecated public CvMat get(int index, double[] vv) {
        return get(index, vv, 0, vv.length);
    }
    /** @see #createIndexer() */
    @Deprecated public CvMat get(double[] vv) {
        return get(0, vv);
    }
    /** @see #createIndexer() */
    @Deprecated public double[] get() {
        double[] vv = new double[fullSize()/elemSize()];
        get(vv);
        return vv;
    }

    /** @see #createIndexer() */
    @Deprecated public CvMat put(int i, double v) {
        switch (depth()) {
            case CV_8U:
            case CV_8S:  getByteBuffer()  .put(i, (byte)(int)v);  break;
            case CV_16U:
            case CV_16S: getShortBuffer() .put(i, (short)(int)v); break;
            case CV_32S: getIntBuffer()   .put(i, (int)v);        break;
            case CV_32F: getFloatBuffer() .put(i, (float)v);      break;
            case CV_64F: getDoubleBuffer().put(i, v);             break;
            default: assert false;
        }
        return (CvMat)this;
    }
    /** @see #createIndexer() */
    @Deprecated public CvMat put(int i, int j, double v) {
        return put(i*step()/elemSize() + j*channels(), v);
    }
    /** @see #createIndexer() */
    @Deprecated public CvMat put(int i, int j, int k, double v) {
        return put(i*step()/elemSize() + j*channels() + k, v);
    }
    /** @see #createIndexer() */
    @Deprecated public synchronized CvMat put(int index, double[] vv, int offset, int length) {
        switch (depth()) {
            case CV_8U:
            case CV_8S:
                ByteBuffer bb = getByteBuffer();
                bb.position(index);
                for (int i = 0; i < length; i++) {
                    bb.put((byte)(int)vv[i+offset]);
                }
                break;
            case CV_16U:
            case CV_16S:
                ShortBuffer sb = getShortBuffer();
                sb.position(index);
                for (int i = 0; i < length; i++) {
                    sb.put((short)(int)vv[i+offset]);
                }
                break;
            case CV_32S:
                IntBuffer ib = getIntBuffer();
                ib.position(index);
                for (int i = 0; i < length; i++) {
                    ib.put((int)vv[i+offset]);
                }
                break;
            case CV_32F:
                FloatBuffer fb = getFloatBuffer();
                fb.position(index);
                for (int i = 0; i < length; i++) {
                    fb.put((float)vv[i+offset]);
                }
                break;
            case CV_64F:
                DoubleBuffer db = getDoubleBuffer();
                db.position(index);
                db.put(vv, offset, length);
                break;
            default: assert false;
        }
        return (CvMat)this;
    }
    /** @see #createIndexer() */
    @Deprecated public CvMat put(int index, double ... vv) {
        return put(index, vv, 0, vv.length);
    }
    /** @see #createIndexer() */
    @Deprecated public CvMat put(double ... vv) {
        return put(0, vv);
    }

    public CvMat put(CvMat mat) {
        return put(0, 0, 0, mat, 0, 0, 0);
    }
    public synchronized CvMat put(int dsti, int dstj, int dstk,
            CvMat mat, int srci, int srcj, int srck) {
        if (rows() == mat.rows() && cols() == mat.cols() && step() == mat.step() && type() == mat.type() &&
                dsti == 0 && dstj == 0 && dstk == 0 && srci == 0 && srcj == 0 && srck == 0) {
            getByteBuffer().clear();
            mat.getByteBuffer().clear();
            getByteBuffer().put(mat.getByteBuffer());
        } else {
            int w = Math.min(rows()-dsti, mat.rows()-srci);
            int h = Math.min(cols()-dstj, mat.cols()-srcj);
            int d = Math.min(channels()-dstk, mat.channels()-srck);
            for (int i = 0; i < w; i++) {
                for (int j = 0; j < h; j++) {
                    for (int k = 0; k < d; k++) {
                        put(i+dsti, j+dstj, k+dstk, mat.get(i+srci, j+srcj, k+srck));
                    }
                }
            }
        }
        return (CvMat)this;
    }

    public IplImage asIplImage() {
        IplImage image = new IplImage();
        cvGetImage(this, image);
        return image;
    }

    @Override public String toString() {
        return toString(0);
    }
    public String toString(int indent) {
        StringBuilder s = new StringBuilder("[ ");
        int channels = channels();
        for (int i = 0; i < rows(); i++) {
            for (int j = 0; j < cols(); j++) {
                CvScalar v = cvGet2D(this, i, j);
                if (channels > 1) {
                    s.append("(");
                }
                for (int k = 0; k < channels; k++) {
                    s.append((float)v.val(k));
                    if (k < channels-1) {
                        s.append(", ");
                    }
                }
                if (channels > 1) {
                    s.append(")");
                }
                if (j < cols()-1) {
                    s.append(", ");
                }
            }
            if (i < rows()-1) {
                s.append("\n  ");
                for (int j = 0; j < indent; j++) {
                    s.append(' ');
                }
            }
        }
        s.append(" ]");
        return s.toString();
    }
}
