/*
 * Copyright (C) 2014-2016 Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.bytedeco.javacpp.helper;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.ShortBuffer;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.javacpp.ShortPointer;
import org.bytedeco.javacpp.annotation.Name;
import org.bytedeco.javacpp.annotation.Opaque;
import org.bytedeco.javacpp.annotation.ValueGetter;
import org.bytedeco.javacpp.indexer.ByteIndexer;
import org.bytedeco.javacpp.indexer.DoubleIndexer;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.javacpp.indexer.Indexable;
import org.bytedeco.javacpp.indexer.Indexer;
import org.bytedeco.javacpp.indexer.IntIndexer;
import org.bytedeco.javacpp.indexer.ShortIndexer;
import org.bytedeco.javacpp.indexer.UByteIndexer;
import org.bytedeco.javacpp.indexer.UShortIndexer;

// required by javac to resolve circular dependencies
import org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_core.CV_16S;
import static org.bytedeco.javacpp.opencv_core.CV_16U;
import static org.bytedeco.javacpp.opencv_core.CV_32F;
import static org.bytedeco.javacpp.opencv_core.CV_32S;
import static org.bytedeco.javacpp.opencv_core.CV_64F;
import static org.bytedeco.javacpp.opencv_core.CV_8S;
import static org.bytedeco.javacpp.opencv_core.CV_8U;
import static org.bytedeco.javacpp.opencv_core.CV_IS_MAT_CONT;
import static org.bytedeco.javacpp.opencv_core.CV_L2;
import static org.bytedeco.javacpp.opencv_core.CV_MAKETYPE;
import static org.bytedeco.javacpp.opencv_core.CV_MAT_CN;
import static org.bytedeco.javacpp.opencv_core.CV_MAT_DEPTH;
import static org.bytedeco.javacpp.opencv_core.CV_MAT_MAGIC_VAL;
import static org.bytedeco.javacpp.opencv_core.CV_MAT_TYPE;
import static org.bytedeco.javacpp.opencv_core.IPL_DEPTH_16S;
import static org.bytedeco.javacpp.opencv_core.IPL_DEPTH_16U;
import static org.bytedeco.javacpp.opencv_core.IPL_DEPTH_1U;
import static org.bytedeco.javacpp.opencv_core.IPL_DEPTH_32F;
import static org.bytedeco.javacpp.opencv_core.IPL_DEPTH_32S;
import static org.bytedeco.javacpp.opencv_core.IPL_DEPTH_64F;
import static org.bytedeco.javacpp.opencv_core.IPL_DEPTH_8S;
import static org.bytedeco.javacpp.opencv_core.IPL_DEPTH_8U;
import static org.bytedeco.javacpp.opencv_core.cvCloneImage;
import static org.bytedeco.javacpp.opencv_core.cvCloneMat;
import static org.bytedeco.javacpp.opencv_core.cvCloneMatND;
import static org.bytedeco.javacpp.opencv_core.cvCloneSparseMat;
import static org.bytedeco.javacpp.opencv_core.cvCreateGraph;
import static org.bytedeco.javacpp.opencv_core.cvCreateGraphScanner;
import static org.bytedeco.javacpp.opencv_core.cvCreateImage;
import static org.bytedeco.javacpp.opencv_core.cvCreateImageHeader;
import static org.bytedeco.javacpp.opencv_core.cvCreateMat;
import static org.bytedeco.javacpp.opencv_core.cvCreateMatHeader;
import static org.bytedeco.javacpp.opencv_core.cvCreateMatND;
import static org.bytedeco.javacpp.opencv_core.cvCreateMemStorage;
import static org.bytedeco.javacpp.opencv_core.cvCreateSeq;
import static org.bytedeco.javacpp.opencv_core.cvCreateSet;
import static org.bytedeco.javacpp.opencv_core.cvCreateSparseMat;
import static org.bytedeco.javacpp.opencv_core.cvGet2D;
import static org.bytedeco.javacpp.opencv_core.cvGetImage;
import static org.bytedeco.javacpp.opencv_core.cvGetMat;
import static org.bytedeco.javacpp.opencv_core.cvOpenFileStorage;
import static org.bytedeco.javacpp.opencv_core.cvReleaseFileStorage;
import static org.bytedeco.javacpp.opencv_core.cvReleaseGraphScanner;
import static org.bytedeco.javacpp.opencv_core.cvReleaseImage;
import static org.bytedeco.javacpp.opencv_core.cvReleaseImageHeader;
import static org.bytedeco.javacpp.opencv_core.cvReleaseMat;
import static org.bytedeco.javacpp.opencv_core.cvReleaseMatND;
import static org.bytedeco.javacpp.opencv_core.cvReleaseMemStorage;
import static org.bytedeco.javacpp.opencv_core.cvReleaseSparseMat;
import static org.bytedeco.javacpp.opencv_core.cvScalar;

public class opencv_core extends org.bytedeco.javacpp.presets.opencv_core {

    public static abstract class AbstractArray extends Pointer implements Indexable {
        static { Loader.load(); }
        public AbstractArray(Pointer p) { super(p); }

        public abstract int arrayChannels();
        public abstract int arrayDepth();
        public abstract int arrayOrigin();
        public abstract void arrayOrigin(int origin);
        public abstract int arrayWidth();
        public abstract int arrayHeight();
        public abstract IplROI arrayROI();
        public abstract int arraySize();
        public abstract BytePointer arrayData();
        public abstract int arrayStep();

        /** @return {@code createBuffer(0)} */
        public <B extends Buffer> B createBuffer() {
            return (B)createBuffer(0);
        }
        /** @return {@link #arrayData()} wrapped in a {@link Buffer} of appropriate type starting at given index */
        public <B extends Buffer> B createBuffer(int index) {
            BytePointer ptr = arrayData();
            int size = arraySize();
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
            int size = arraySize();
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

        public CvSize cvSize() { return org.bytedeco.javacpp.opencv_core.cvSize(arrayWidth(), arrayHeight()); }

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

    @Opaque public static class CvArr extends AbstractArray {
        public CvArr(Pointer p) { super(p); }

        @Override public int arrayChannels()          { throw new UnsupportedOperationException(); }
        @Override public int arrayDepth()             { throw new UnsupportedOperationException(); }
        @Override public int arrayOrigin()            { throw new UnsupportedOperationException(); }
        @Override public void arrayOrigin(int origin) { throw new UnsupportedOperationException(); }
        @Override public int arrayWidth()             { throw new UnsupportedOperationException(); }
        @Override public int arrayHeight()            { throw new UnsupportedOperationException(); }
        @Override public IplROI arrayROI()            { throw new UnsupportedOperationException(); }
        @Override public int arraySize()              { throw new UnsupportedOperationException(); }
        @Override public BytePointer arrayData()      { throw new UnsupportedOperationException(); }
        @Override public int arrayStep()              { throw new UnsupportedOperationException(); }
    }

    @Name("CvArr*")
    public static class CvArrArray extends PointerPointer<CvArr> {
        static { Loader.load(); }
        public CvArrArray(CvArr ... array) { this(array.length); put(array); position(0); }
        public CvArrArray(long size) { super(size); allocateArray(size); }
        public CvArrArray(Pointer p) { super(p); }
        private native void allocateArray(long size);

        @Override public CvArrArray position(long position) {
            return (CvArrArray)super.position(position);
        }

        public CvArrArray put(CvArr ... array) {
            for (int i = 0; i < array.length; i++) {
                position(i).put(array[i]);
            }
            return this;
        }

        public native CvArr get();
        public native CvArrArray put(CvArr p);
    }

    @Name("CvMat*")
    public static class CvMatArray extends CvArrArray {
        public CvMatArray(CvMat ... array) { this(array.length); put(array); position(0); }
        public CvMatArray(long size) { allocateArray(size); }
        public CvMatArray(Pointer p) { super(p); }
        private native void allocateArray(long size);

        @Override public CvMatArray position(long position) {
            return (CvMatArray)super.position(position);
        }
        @Override public CvMatArray put(CvArr ... array) {
            return (CvMatArray)super.put(array);
        }
        @Override @ValueGetter public native CvMat get();
        @Override public CvMatArray put(CvArr p) {
            if (p instanceof CvMat) {
                return (CvMatArray)super.put(p);
            } else {
                throw new ArrayStoreException(p.getClass().getName());
            }
        }
    }

    @Name("CvMatND*")
    public static class CvMatNDArray extends CvArrArray {
        public CvMatNDArray(CvMatND ... array) { this(array.length); put(array); position(0); }
        public CvMatNDArray(long size) { allocateArray(size); }
        public CvMatNDArray(Pointer p) { super(p); }
        private native void allocateArray(long size);

        @Override public CvMatNDArray position(long position) {
            return (CvMatNDArray)super.position(position);
        }
        @Override public CvMatNDArray put(CvArr ... array) {
            return (CvMatNDArray)super.put(array);
        }
        @Override @ValueGetter public native CvMatND get();
        @Override public CvMatNDArray put(CvArr p) {
            if (p instanceof CvMatND) {
                return (CvMatNDArray)super.put(p);
            } else {
                throw new ArrayStoreException(p.getClass().getName());
            }
        }
    }

    @Name("IplImage*")
    public static class IplImageArray extends CvArrArray {
        public IplImageArray(IplImage ... array) { this(array.length); put(array); position(0); }
        public IplImageArray(long size) { allocateArray(size); }
        public IplImageArray(Pointer p) { super(p); }
        private native void allocateArray(long size);

        @Override public IplImageArray position(long position) {
            return (IplImageArray)super.position(position);
        }
        @Override public IplImageArray put(CvArr ... array) {
            return (IplImageArray)super.put(array);
        }
        @Override @ValueGetter public native IplImage get();
        @Override public IplImageArray put(CvArr p) {
            if (p instanceof IplImage) {
                return (IplImageArray)super.put(p);
            } else {
                throw new ArrayStoreException(p.getClass().getName());
            }
        }
    }

    public static abstract class AbstractIplImage extends CvArr {
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
            return create(org.bytedeco.javacpp.opencv_core.cvSize(width, height), depth, channels);
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
            return createHeader(org.bytedeco.javacpp.opencv_core.cvSize(width, height), depth, channels);
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
        protected static class ReleaseDeallocator extends IplImage implements Pointer.Deallocator {
            ReleaseDeallocator(IplImage p) { super(p); }
            @Override public void deallocate() { cvReleaseImage(this); }
        }
        protected static class HeaderReleaseDeallocator extends IplImage implements Pointer.Deallocator {
            HeaderReleaseDeallocator(IplImage p) { super(p); }
            @Override public void deallocate() { cvReleaseImageHeader(this); }
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
        @Override public int arraySize() { return imageSize(); }
        @Override public BytePointer arrayData() { return imageData(); }
        @Override public int arrayStep() { return widthStep(); }

        public CvMat asCvMat() {
            CvMat mat = new CvMat();
            cvGetMat(this, mat, (IntPointer)null, 0);
            return mat;
        }
    }

    public static abstract class AbstractCvMat extends CvArr {
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
        protected static class ReleaseDeallocator extends CvMat implements Pointer.Deallocator {
            ReleaseDeallocator(CvMat m) { super(m); }
            @Override public void deallocate() { cvReleaseMat(this); }
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
        @Override public int arraySize() { return size(); }
        @Override public BytePointer arrayData() { return data_ptr(); }
        @Override public int arrayStep() { return step(); }

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

    public static abstract class AbstractCvMatND extends CvArr {
        public AbstractCvMatND(Pointer p) { super(p); }

        /**
         * Calls CvMatND(), and registers a deallocator.
         * @return CvMatND created. Do not call cvReleaseMatND() on it.
         */
        public static CvMatND create(int dims, int[] sizes, int type) {
            CvMatND m = cvCreateMatND(dims, sizes, type);
            if (m != null) {
                ((AbstractCvMatND)m).deallocator(new ReleaseDeallocator(m));
            }
            return m;
        }

        /**
         * Calls cvCloneMatND(), and registers a deallocator.
         * @return CvMatND cloned. Do not call cvReleaseMatND() on it.
         */
        @Override public CvMatND clone() {
            CvMatND m = cvCloneMatND((CvMatND)this);
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
        protected static class ReleaseDeallocator extends CvMatND implements Pointer.Deallocator {
            ReleaseDeallocator(CvMatND p) { super(p); }
            @Override public void deallocate() { cvReleaseMatND(this); }
        }
    }

    public static abstract class AbstractCvSparseMat extends CvArr {
        public AbstractCvSparseMat(Pointer p) { super(p); }

        /**
         * Calls cvCreateSparseMat(), and registers a deallocator.
         * @return CvSparseMat created. Do not call cvReleaseSparseMat() on it.
         */
        public static CvSparseMat create(int dims, int[] sizes, int type) {
            CvSparseMat m = cvCreateSparseMat(dims, sizes, type);
            if (m != null) {
                m.deallocator(new ReleaseDeallocator(m));
            }
            return m;
        }

        /**
         * Calls cvCloneSparseMat(), and registers a deallocator.
         * @return CvSparseMat cloned. Do not call cvReleaseSparseMat() on it.
         */
        @Override public CvSparseMat clone() {
            CvSparseMat m = cvCloneSparseMat((CvSparseMat)this);
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
        protected static class ReleaseDeallocator extends CvSparseMat implements Pointer.Deallocator {
            ReleaseDeallocator(CvSparseMat p) { super(p); }
            @Override public void deallocate() { cvReleaseSparseMat(this); }
        }
    }

    public static abstract class AbstractCvRect extends IntPointer {
        static { Loader.load(); }
        public AbstractCvRect(Pointer p) { super(p); }

//        public CvRect(int x, int y, int width, int height) {
//            allocate(); x(x).y(y).width(width).height(height);
//        }

        public abstract int x();
        public abstract int y();
        public abstract int width();
        public abstract int height();

        @Override public String toString() {
            if (isNull()) {
                return super.toString();
            } else {
                if (capacity() == 0) {
                    return "(" + x() + ", " + y() + "; " + width() + ", " + height() + ")";
                }
                String s = "";
                long p = position();
                for (long i = 0; i < capacity(); i++) {
                    position(i);
                    s += (i == 0 ? "(" : " (") + x() + ", " + y() + "; " + width() + ", " + height() + ")";
                }
                position(p);
                return s;
            }
        }
    }

    public static abstract class AbstractCvPoint extends IntPointer {
        static { Loader.load(); }
        public AbstractCvPoint(Pointer p) { super(p); }

//        public CvPoint(int[] pts, int offset, int length) {
//            this(length/2);
//            put(pts, offset, length);
//        }
//        public CvPoint(int ... pts) {
//            this(pts, 0, pts.length);
//        }
//        public CvPoint(byte shift, double[] pts, int offset, int length) {
//            this(length/2);
//            put(shift, pts, offset, length);
//        }
//        public CvPoint(byte shift, double ... pts) {
//            this(shift, pts, 0, pts.length);
//        }

        public abstract int x(); public abstract CvPoint x(int x);
        public abstract int y(); public abstract CvPoint y(int y);

//        public int[] get() {
//            int[] pts = new int[capacity == 0 ? 2 : 2*capacity];
//            get(pts);
//            return pts;
//        }
        public CvPoint get(int[] pts) {
            return get(pts, 0, pts.length);
        }
        public CvPoint get(int[] pts, int offset, int length) {
            for (int i = 0; i < length/2; i++) {
                position(i);
                pts[offset + i*2  ] = x();
                pts[offset + i*2+1] = y();
            }
            return (CvPoint)position(0);
        }

        public final CvPoint put(int[] pts, int offset, int length) {
            for (int i = 0; i < length/2; i++) {
                position(i); put(pts[offset + i*2], pts[offset + i*2+1]);
            }
            return (CvPoint)position(0);
        }
        public final CvPoint put(int ... pts) {
            return put(pts, 0, pts.length);
        }
        public final CvPoint put(byte shift, double[] pts, int offset, int length) {
            int[] a = new int[length];
            for (int i = 0; i < length; i++) {
                a[i] = (int)Math.round(pts[offset + i] * (1<<shift));
            }
            return put(a, 0, length);
        }
        public final CvPoint put(byte shift, double ... pts) {
            return put(shift, pts, 0, pts.length);
        }

        public CvPoint put(int x, int y) {
            return x(x).y(y);
        }
        public CvPoint put(CvPoint o) {
            return x(o.x()).y(o.y());
        }
        public CvPoint put(byte shift, CvPoint2D32f o) {
            x((int)Math.round(o.x() * (1<<shift)));
            y((int)Math.round(o.y() * (1<<shift)));
            return (CvPoint)this;
        }
        public CvPoint put(byte shift, CvPoint2D64f o) {
            x((int)Math.round(o.x() * (1<<shift)));
            y((int)Math.round(o.y() * (1<<shift)));
            return (CvPoint)this;
        }

        @Override public String toString() {
            if (isNull()) {
                return super.toString();
            } else {
                if (capacity() == 0) {
                    return "(" + x() + ", " + y() + ")";
                }
                String s = "";
                long p = position();
                for (long i = 0; i < capacity(); i++) {
                    position(i);
                    s += (i == 0 ? "(" : " (") + x() + ", " + y() + ")";
                }
                position(p);
                return s;
            }
        }

        public static final CvPoint ZERO = new CvPoint().x(0).y(0);
    }

    public static abstract class AbstractCvPoint2D32f extends FloatPointer {
        static { Loader.load(); }
        public AbstractCvPoint2D32f(Pointer p) { super(p); }

//        public CvPoint2D32f(double[] pts, int offset, int length) {
//            this(length/2);
//            put(pts, offset, length);
//        }
//        public CvPoint2D32f(double ... pts) {
//            this(pts, 0, pts.length);
//        }

        public abstract float x(); public abstract CvPoint2D32f x(float x);
        public abstract float y(); public abstract CvPoint2D32f y(float y);

//        public double[] get() {
//            double[] pts = new double[capacity == 0 ? 2 : 2*capacity];
//            get(pts);
//            return pts;
//        }
        public CvPoint2D32f get(double[] pts) {
            return get(pts, 0, pts.length);
        }
        public CvPoint2D32f get(double[] pts, int offset, int length) {
            for (int i = 0; i < length/2; i++) {
                position(i);
                pts[offset + i*2  ] = x();
                pts[offset + i*2+1] = y();
            }
            return (CvPoint2D32f)position(0);
        }

        public final CvPoint2D32f put(double[] pts, int offset, int length) {
            for (int i = 0; i < length/2; i++) {
                position(i); put(pts[offset + i*2], pts[offset + i*2+1]);
            }
            return (CvPoint2D32f)position(0);
        }
        public final CvPoint2D32f put(double ... pts) {
            return put(pts, 0, pts.length);
        }

        public CvPoint2D32f put(double x, double y) {
            return x((float)x).y((float)y);
        }
        public CvPoint2D32f put(CvPoint o) {
            return x(o.x()).y(o.y());
        }
        public CvPoint2D32f put(CvPoint2D32f o) {
            return x(o.x()).y(o.y());
        }
        public CvPoint2D32f put(CvPoint2D64f o) {
            return x((float)o.x()).y((float)o.y());
        }

        @Override public String toString() {
            if (isNull()) {
                return super.toString();
            } else {
                if (capacity() == 0) {
                    return "(" + x() + ", " + y() + ")";
                }
                String s = "";
                long p = position();
                for (long i = 0; i < capacity(); i++) {
                    position(i);
                    s += (i == 0 ? "(" : " (") + x() + ", " + y() + ")";
                }
                position(p);
                return s;
            }
        }
    }

    public static abstract class AbstractCvPoint3D32f extends FloatPointer {
        static { Loader.load(); }
        public AbstractCvPoint3D32f(Pointer p) { super(p); }

//        public CvPoint3D32f(double[] pts, int offset, int length) {
//            this(length/3);
//            put(pts, offset, length);
//        }
//        public CvPoint3D32f(double ... pts) {
//            this(pts, 0, pts.length);
//        }

        public abstract float x(); public abstract CvPoint3D32f x(float x);
        public abstract float y(); public abstract CvPoint3D32f y(float y);
        public abstract float z(); public abstract CvPoint3D32f z(float z);

//        public double[] get() {
//            double[] pts = new double[capacity == 0 ? 3 : 3*capacity];
//            get(pts);
//            return pts;
//        }
        public CvPoint3D32f get(double[] pts) {
            return get(pts, 0, pts.length);
        }
        public CvPoint3D32f get(double[] pts, int offset, int length) {
            for (int i = 0; i < length/3; i++) {
                position(i);
                pts[offset + i*3  ] = x();
                pts[offset + i*3+1] = y();
                pts[offset + i*3+2] = z();
            }
            return (CvPoint3D32f)position(0);
        }

        public final CvPoint3D32f put(double[] pts, int offset, int length) {
            for (int i = 0; i < length/3; i++) {
                position(i); put(pts[offset + i*3], pts[offset + i*3+1], pts[offset + i*3+2]);
            }
            return (CvPoint3D32f)position(0);
        }
        public final CvPoint3D32f put(double ... pts) {
            return put(pts, 0, pts.length);
        }

        public CvPoint3D32f put(double x, double y, double z) {
            return x((float)x).y((float)y).z((float)z);
        }
        public CvPoint3D32f put(CvPoint o) {
            return x(o.x()).y(o.y()).z(0);
        }
        public CvPoint3D32f put(CvPoint2D32f o) {
            return x(o.x()).y(o.y()).z(0);
        }
        public CvPoint3D32f put(CvPoint2D64f o) {
            return x((float)o.x()).y((float)o.y()).z(0);
        }

        @Override public String toString() {
            if (isNull()) {
                return super.toString();
            } else {
                if (capacity() == 0) {
                    return "(" + x() + ", " + y() + ", " + z() + ")";
                }
                String s = "";
                long p = position();
                for (long i = 0; i < capacity(); i++) {
                    position(i);
                    s += (i == 0 ? "(" : " (") + x() + ", " + y() + ", " + z() + ")";
                }
                position(p);
                return s;
            }
        }
    }

    public static abstract class AbstractCvPoint2D64f extends DoublePointer {
        static { Loader.load(); }
        public AbstractCvPoint2D64f(Pointer p) { super(p); }

//        public CvPoint2D64f(double[] pts, int offset, int length) {
//            this(length/2);
//            put(pts, offset, length);
//        }
//        public CvPoint2D64f(double ... pts) {
//            this(pts, 0, pts.length);
//        }

        public abstract double x(); public abstract CvPoint2D64f x(double x);
        public abstract double y(); public abstract CvPoint2D64f y(double y);

//        public double[] get() {
//            double[] pts = new double[capacity == 0 ? 2 : 2*capacity];
//            get(pts);
//            return pts;
//        }
        public CvPoint2D64f get(double[] pts) {
            return get(pts, 0, pts.length);
        }
        public CvPoint2D64f get(double[] pts, int offset, int length) {
            for (int i = 0; i < length/2; i++) {
                position(i);
                pts[offset + i*2  ] = x();
                pts[offset + i*2+1] = y();
            }
            return (CvPoint2D64f)position(0);
        }

        public final CvPoint2D64f put(double[] pts, int offset, int length) {
            for (int i = 0; i < length/2; i++) {
                position(i); put(pts[offset + i*2], pts[offset + i*2+1]);
            }
            return (CvPoint2D64f)position(0);
        }
        public final CvPoint2D64f put(double ... pts) {
            return put(pts, 0, pts.length);
        }

        public CvPoint2D64f put(double x, double y) {
            return x(x).y(y);
        }
        public CvPoint2D64f put(CvPoint o) {
            return x(o.x()).y(o.y());
        }
        public CvPoint2D64f put(CvPoint2D32f o) {
            return x(o.x()).y(o.y());
        }
        public CvPoint2D64f put(CvPoint2D64f o) {
            return x(o.x()).y(o.y());
        }

        @Override public String toString() {
            if (isNull()) {
                return super.toString();
            } else {
                if (capacity() == 0) {
                    return "(" + (float)x() + ", " + (float)y() + ")";
                }
                String s = "";
                long p = position();
                for (long i = 0; i < capacity(); i++) {
                    position(i);
                    s += (i == 0 ? "(" : " (") + (float)x() + ", " + (float)y() + ")";
                }
                position(p);
                return s;
            }
        }
    }

    public static abstract class AbstractCvPoint3D64f extends DoublePointer {
        static { Loader.load(); }
        public AbstractCvPoint3D64f(Pointer p) { super(p); }

//        public CvPoint3D64f(double[] pts, int offset, int length) {
//            this(length/3);
//            put(pts, offset, length);
//        }
//        public CvPoint3D64f(double ... pts) {
//            this(pts, 0, pts.length);
//        }

        public abstract double x(); public abstract CvPoint3D64f x(double x);
        public abstract double y(); public abstract CvPoint3D64f y(double y);
        public abstract double z(); public abstract CvPoint3D64f z(double z);

//        public double[] get() {
//            double[] pts = new double[capacity == 0 ? 3 : 3*capacity];
//            get(pts);
//            return pts;
//        }
        public CvPoint3D64f get(double[] pts) {
            return get(pts, 0, pts.length);
        }
        public CvPoint3D64f get(double[] pts, int offset, int length) {
            for (int i = 0; i < length/3; i++) {
                position(i);
                pts[offset + i*3  ] = x();
                pts[offset + i*3+1] = y();
                pts[offset + i*3+2] = z();
            }
            return (CvPoint3D64f)position(0);
        }

        public final CvPoint3D64f put(double[] pts, int offset, int length) {
            for (int i = 0; i < length/3; i++) {
                position(i); put(pts[offset + i*3], pts[offset + i*3+1], pts[offset + i*3+2]);
            }
            return (CvPoint3D64f)position(0);
        }
        public final CvPoint3D64f put(double ... pts) {
            return put(pts, 0, pts.length);
        }

        public CvPoint3D64f put(double x, double y, double z) {
            return x(x()).y(y()).z(z());
        }
        public CvPoint3D64f put(CvPoint o) {
            return x(o.x()).y(o.y()).z(0);
        }
        public CvPoint3D64f put(CvPoint2D32f o) {
            return x(o.x()).y(o.y()).z(0);
        }
        public CvPoint3D64f put(CvPoint2D64f o) {
            return x(o.x()).y(o.y()).z(0);
        }

        @Override public String toString() {
            if (isNull()) {
                return super.toString();
            } else {
                if (capacity() == 0) {
                    return "(" + (float)x() + ", " + (float)y() + ", " + (float)z() + ")";
                }
                String s = "";
                long p = position();
                for (long i = 0; i < capacity(); i++) {
                    position(i);
                    s += (i == 0 ? "(" : " (") + (float)x() + ", " + (float)y() + ", " + (float)z() + ")";
                }
                position(p);
                return s;
            }
        }
    }

    public static abstract class AbstractCvSize extends IntPointer {
        static { Loader.load(); }
        public AbstractCvSize(Pointer p) { super(p); }

//        public CvSize(int width, int height) {
//            allocate(); width(width).height(height);
//        }

        public abstract int width();  public abstract CvSize width(int width);
        public abstract int height(); public abstract CvSize height(int height);

        @Override public String toString() {
            if (isNull()) {
                return super.toString();
            } else {
                if (capacity() == 0) {
                    return "(" + width() + ", " + height() + ")";
                }
                String s = "";
                long p = position();
                for (long i = 0; i < capacity(); i++) {
                    position(i);
                    s += (i == 0 ? "(" : " (") + width() + ", " + height() + ")";
                }
                position(p);
                return s;
            }
        }
        public static final CvSize ZERO = new CvSize().width(0).height(0);
    }

    public static abstract class AbstractCvSize2D32f extends FloatPointer {
        static { Loader.load(); }
        public AbstractCvSize2D32f(Pointer p) { super(p); }

//        public CvSize2D32f(float width, float height) {
//            allocate(); width(width).height(height);
//        }

        public abstract float width();  public abstract CvSize2D32f width(float width);
        public abstract float height(); public abstract CvSize2D32f height(float height);

        @Override public String toString() {
            if (isNull()) {
                return super.toString();
            } else {
                if (capacity() == 0) {
                    return "(" + width() + ", " + height() + ")";
                }
                String s = "";
                long p = position();
                for (long i = 0; i < capacity(); i++) {
                    position(i);
                    s += (i == 0 ? "(" : " (") + width() + ", " + height() + ")";
                }
                position(p);
                return s;
            }
        }
    }

    public static abstract class AbstractCvBox2D extends FloatPointer {
        static { Loader.load(); }
        public AbstractCvBox2D(Pointer p) { super(p); }

//        public CvBox2D(CvPoint2D32f center, CvSize2D32f size, float angle) {
//            allocate(); center(center).size(size).angle(angle);
//        }

        public abstract CvPoint2D32f center(); public abstract CvBox2D center(CvPoint2D32f center);
        public abstract CvSize2D32f size();    public abstract CvBox2D size(CvSize2D32f size);
        public abstract float angle();         public abstract CvBox2D angle(float angle);

        @Override public String toString() {
            if (isNull()) {
                return super.toString();
            } else {
                if (capacity() == 0) {
                    return "(" + center() + ", " + size() + ", " + angle() + ")";
                }
                String s = "";
                long p = position();
                for (long i = 0; i < capacity(); i++) {
                    position(i);
                    s += (i == 0 ? "(" : " (") + center() + ", " + size() + ", " + angle() + ")";
                }
                position(p);
                return s;
            }
        }
    }

    public static abstract class AbstractCvScalar extends DoublePointer {
        static { Loader.load(); }
        public AbstractCvScalar(Pointer p) { super(p); }

//        public CvScalar(double val0, double val1, double val2, double val3) {
//            allocate(); val(0, val0).val(1, val1).val(2, val2).val(3, val3);
//        }

        public abstract double/*[4]*/ val(int i); public abstract CvScalar val(int i, double val);
        public double getVal(int i)               { return val(i);      }
        public CvScalar setVal(int i, double val) { return val(i, val); }

        public abstract DoublePointer val();
        public DoublePointer getDoublePointerVal() { return val(); }
        public LongPointer getLongPointerVal() { return new LongPointer(val()); }

        public void scale(double s) {
            for (int i = 0; i < 4; i++) {
                val(i, val(i) * s);
            }
        }

        public double red()      { return val(2); }
        public double green()    { return val(1); }
        public double blue()     { return val(0); }
        public CvScalar red  (double r) { val(2, r); return (CvScalar)this; }
        public CvScalar green(double g) { val(1, g); return (CvScalar)this; }
        public CvScalar blue (double b) { val(0, b); return (CvScalar)this; }

        public double magnitude() {
            return Math.sqrt(val(0)*val(0) + val(1)*val(1) + val(2)*val(2) + val(3)*val(3));
        }

        @Override public String toString() {
            if (isNull()) {
                return super.toString();
            } else {
                if (capacity() == 0) {
                    return "(" + (float)val(0) + ", " + (float)val(1) + ", " +
                            (float)val(2) + ", " + (float)val(3) + ")";
                }
                String s = "";
                long p = position();
                for (long i = 0; i < capacity(); i++) {
                    position(i);
                    s += (i == 0 ? "(" : " (") + (float)val(0) + ", " + (float)val(1) + ", " +
                            (float)val(2) + ", " + (float)val(3) + ")";
                }
                position(p);
                return s;
            }
        }

        public static final CvScalar
                ZERO    = new CvScalar().val(0, 0.0).val(1, 0.0).val(2, 0.0).val(3, 0.0),
                ONE     = new CvScalar().val(0, 1.0).val(1, 1.0).val(2, 1.0).val(3, 1.0),
                ONEHALF = new CvScalar().val(0, 0.5).val(1, 0.5).val(2, 0.5).val(3, 0.5),
                ALPHA1  = new CvScalar().val(0, 0.0).val(1, 0.0).val(2, 0.0).val(3, 1.0),
                ALPHA255= new CvScalar().val(0, 0.0).val(1, 0.0).val(2, 0.0).val(3, 255.0),

                WHITE   = CV_RGB(255, 255, 255),
                GRAY    = CV_RGB(128, 128, 128),
                BLACK   = CV_RGB(  0,   0,   0),
                RED     = CV_RGB(255,   0,   0),
                GREEN   = CV_RGB(  0, 255,   0),
                BLUE    = CV_RGB(  0,   0, 255),
                CYAN    = CV_RGB(  0, 255, 255),
                MAGENTA = CV_RGB(255,   0, 255),
                YELLOW  = CV_RGB(255, 255,   0);
    }
    public static CvScalar CV_RGB(double r, double g, double b) {
        return cvScalar(b, g, r, 0);
    }

    public static abstract class AbstractCvMemStorage extends Pointer {
        static { Loader.load(); }
        public AbstractCvMemStorage(Pointer p) { super(p); }

        /**
         * Calls cvCreateMemStorage(), and registers a deallocator.
         * @return CvMemStorage created. Do not call cvReleaseMemStorage() on it.
         */
        public static CvMemStorage create(int block_size) {
            CvMemStorage m = cvCreateMemStorage(block_size);
            if (m != null) {
                m.deallocator(new ReleaseDeallocator(m));
            }
            return m;
        }
        /**
         * Calls cvCreateMemStorage(0), and registers a deallocator.
         * @return CvMemStorage created. Do not call cvReleaseMemStorage() on it.
         */
        public static CvMemStorage create() {
            return create(0);
        }

        /**
         * Calls the deallocator, if registered, otherwise has no effect.
         */
        public void release() {
            deallocate();
        }
        protected static class ReleaseDeallocator extends CvMemStorage implements Deallocator {
            ReleaseDeallocator(CvMemStorage p) { super(p); }
            @Override public void deallocate() { cvReleaseMemStorage(this); }
        }
    }

    public static abstract class AbstractCvSeq extends CvArr {
        public AbstractCvSeq(Pointer p) { super(p); }

        public static CvSeq create(int seq_flags, int header_size, int elem_size, CvMemStorage storage) {
            return cvCreateSeq(seq_flags, header_size, elem_size, storage);
        }
    }

    public static abstract class AbstractCvSet extends CvSeq {
        public AbstractCvSet(Pointer p) { super(p); }

        public static CvSet create(int set_flags, int header_size, int elem_size,
                CvMemStorage storage) {
            return cvCreateSet(set_flags, header_size, elem_size, storage);
        }
    }

    public static abstract class AbstractCvGraph extends CvSet {
        public AbstractCvGraph(Pointer p) { super(p); }

        public static CvGraph create(int graph_flags, int header_size, int vtx_size,
                int edge_size, CvMemStorage storage) {
            return cvCreateGraph(graph_flags, header_size, vtx_size, edge_size, storage);
        }
    }

    public static abstract class AbstractCvFileStorage extends Pointer {
        public AbstractCvFileStorage(Pointer p) { super(p); }

        /**
         * Calls cvOpenFileStorage(), and registers a deallocator. Uses default encoding.
         * @return CvFileStorage opened. Do not call cvReleaseFileStorage() on it.
         */
        public static CvFileStorage open(String filename, CvMemStorage memstorage, int flags) {
            return open(filename, memstorage, flags, null);
        }
        /**
         * Calls cvOpenFileStorage(), and registers a deallocator.
         * @return CvFileStorage opened. Do not call cvReleaseFileStorage() on it.
         */
        public static CvFileStorage open(String filename, CvMemStorage memstorage, int flags, String encoding) {
            CvFileStorage f = cvOpenFileStorage(filename, memstorage, flags, encoding);
            if (f != null) {
                f.deallocator(new ReleaseDeallocator(f));
            }
            return f;
        }

        /**
         * Calls the deallocator, if registered, otherwise has no effect.
         */
        public void release() {
            deallocate();
        }
        protected static class ReleaseDeallocator extends CvFileStorage implements Deallocator {
            ReleaseDeallocator(CvFileStorage p) { super(p); }
            @Override public void deallocate() { cvReleaseFileStorage(this); }
        }
    }

    public static abstract class AbstractCvGraphScanner extends Pointer {
        public AbstractCvGraphScanner(Pointer p) { super(p); }

        /**
         * Calls cvCreateGraphScanner(), and registers a deallocator.
         * @return CvGraphScanner created. Do not call cvReleaseGraphScanner() on it.
         */
        public static CvGraphScanner create(CvGraph graph,
                CvGraphVtx vtx/*=null*/, int mask/*=CV_GRAPH_ALL_ITEMS*/) {
            CvGraphScanner g = cvCreateGraphScanner(graph, vtx, mask);
            if (g != null) {
                g.deallocator(new ReleaseDeallocator(g));
            }
            return g;
        }
        /**
         * Calls the deallocator, if registered, otherwise has no effect.
         */
        public void release() {
            deallocate();
        }
        protected static class ReleaseDeallocator extends CvGraphScanner implements Deallocator {
            ReleaseDeallocator(CvGraphScanner p) { super(p); }
            @Override public void deallocate() { cvReleaseGraphScanner(this); }
        }
    }

    public static int cvInitNArrayIterator(int count, CvArr[] arrs,
            CvArr mask, CvMatND stubs, CvNArrayIterator array_iterator, int flags/*=0*/) {
        return org.bytedeco.javacpp.opencv_core.cvInitNArrayIterator(count, new CvArrArray(arrs),
                mask, stubs, array_iterator, flags);
    }

    public static void cvMixChannels(CvArr[] src, int src_count,
            CvArr[] dst, int dst_count, int[] from_to, int pair_count) {
        org.bytedeco.javacpp.opencv_core.cvMixChannels(new CvArrArray(src), src_count,
                new CvArrArray(dst), dst_count, new IntPointer(from_to), pair_count);
    }

    public static void cvCalcCovarMatrix(CvArr[] vects, int count, CvArr cov_mat, CvArr avg, int flags) {
        org.bytedeco.javacpp.opencv_core.cvCalcCovarMatrix(new CvArrArray(vects), count, cov_mat, avg, flags);
    }

    public static double cvNorm(CvArr arr1, CvArr arr2) {
        return org.bytedeco.javacpp.opencv_core.cvNorm(arr1, arr2, CV_L2, null);
    }

    public static abstract class AbstractCvFont extends Pointer {
        public AbstractCvFont(Pointer p) { super(p); }

//        public AbstractCvFont(int font_face, double hscale, double vscale,
//            double shear, int thickness, int line_type) {
//            allocate();
//            cvInitFont(this, font_face, hscale, vscale, shear, thickness, line_type);
//        }
//        public AbstractCvFont(int font_face, double scale, int thickness) {
//            allocate();
//            cvInitFont(this, font_face, scale, scale, 0, thickness, CV_AA);
//        }
    }

    public static abstract class AbstractMat extends AbstractArray {
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
        public abstract int step(int i);

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
        @Override public int arraySize() { return step(0)*size(0); }
        @Override public BytePointer arrayData() { return data(); }
        @Override public int arrayStep() { return step(0); }

        public static final Mat EMPTY = null;
    }

    public static abstract class AbstractScalar extends DoublePointer {
        static { Loader.load(); }
        public AbstractScalar(Pointer p) { super(p); }

        public void scale(double s) {
            for (int i = 0; i < 4; i++) {
                put(i, get(i) * s);
            }
        }

        public double red()      { return get(2); }
        public double green()    { return get(1); }
        public double blue()     { return get(0); }
        public Scalar red  (double r) { put(2, r); return (Scalar)this; }
        public Scalar green(double g) { put(1, g); return (Scalar)this; }
        public Scalar blue (double b) { put(0, b); return (Scalar)this; }

        public double magnitude() {
            return Math.sqrt(get(0)*get(0) + get(1)*get(1) + get(2)*get(2) + get(3)*get(3));
        }

        @Override public String toString() {
            if (isNull()) {
                return super.toString();
            } else {
                if (capacity() == 0) {
                    return "(" + (float)get(0) + ", " + (float)get(1) + ", " +
                            (float)get(2) + ", " + (float)get(3) + ")";
                }
                String s = "";
                long p = position();
                for (long i = 0; i < capacity(); i++) {
                    position(i);
                    s += (i == 0 ? "(" : " (") + (float)get(0) + ", " + (float)get(1) + ", " +
                            (float)get(2) + ", " + (float)get(3) + ")";
                }
                position(p);
                return s;
            }
        }

        public static final Scalar
                ZERO    = new Scalar(0.0, 0.0, 0.0, 0.0),
                ONE     = new Scalar(1.0, 1.0, 1.0, 1.0),
                ONEHALF = new Scalar(0.5, 0.5, 0.5, 0.5),
                ALPHA1  = new Scalar(0.0, 0.0, 0.0, 1.0),
                ALPHA255= new Scalar(0.0, 0.0, 0.0, 255.0),

                WHITE   = RGB(255, 255, 255),
                GRAY    = RGB(128, 128, 128),
                BLACK   = RGB(  0,   0,   0),
                RED     = RGB(255,   0,   0),
                GREEN   = RGB(  0, 255,   0),
                BLUE    = RGB(  0,   0, 255),
                CYAN    = RGB(  0, 255, 255),
                MAGENTA = RGB(255,   0, 255),
                YELLOW  = RGB(255, 255,   0);
    }
    public static Scalar RGB(double r, double g, double b) {
        return new Scalar(b, g, r, 0);
    }
}
