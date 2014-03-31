/*
 * Copyright (C) 2014 Samuel Audet
 *
 * This file is part of JavaCPP.
 *
 * JavaCPP is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version (subject to the "Classpath" exception
 * as provided in the LICENSE.txt file that accompanied this code).
 *
 * JavaCPP is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with JavaCPP.  If not, see <http://www.gnu.org/licenses/>.
 */

package com.googlecode.javacpp.helper;

import com.googlecode.javacpp.BytePointer;
import com.googlecode.javacpp.DoublePointer;
import com.googlecode.javacpp.FloatPointer;
import com.googlecode.javacpp.IntPointer;
import com.googlecode.javacpp.Loader;
import com.googlecode.javacpp.LongPointer;
import com.googlecode.javacpp.Pointer;
import com.googlecode.javacpp.PointerPointer;
import com.googlecode.javacpp.ShortPointer;
import com.googlecode.javacpp.annotation.Name;
import com.googlecode.javacpp.annotation.Opaque;
import com.googlecode.javacpp.annotation.ValueGetter;
import java.awt.Rectangle;
import java.awt.Transparency;
import java.awt.color.ColorSpace;
import java.awt.image.BufferedImage;
import java.awt.image.ColorModel;
import java.awt.image.ComponentColorModel;
import java.awt.image.ComponentSampleModel;
import java.awt.image.DataBuffer;
import java.awt.image.DataBufferByte;
import java.awt.image.DataBufferDouble;
import java.awt.image.DataBufferFloat;
import java.awt.image.DataBufferInt;
import java.awt.image.DataBufferShort;
import java.awt.image.DataBufferUShort;
import java.awt.image.MultiPixelPackedSampleModel;
import java.awt.image.Raster;
import java.awt.image.SampleModel;
import java.awt.image.SinglePixelPackedSampleModel;
import java.awt.image.WritableRaster;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.ShortBuffer;

// required by javac to resolve circular dependencies
import com.googlecode.javacpp.opencv_core.*;
import static com.googlecode.javacpp.opencv_core.CV_16S;
import static com.googlecode.javacpp.opencv_core.CV_16U;
import static com.googlecode.javacpp.opencv_core.CV_32F;
import static com.googlecode.javacpp.opencv_core.CV_32S;
import static com.googlecode.javacpp.opencv_core.CV_64F;
import static com.googlecode.javacpp.opencv_core.CV_8S;
import static com.googlecode.javacpp.opencv_core.CV_8U;
import static com.googlecode.javacpp.opencv_core.CV_IS_MAT_CONT;
import static com.googlecode.javacpp.opencv_core.CV_L2;
import static com.googlecode.javacpp.opencv_core.CV_MAKETYPE;
import static com.googlecode.javacpp.opencv_core.CV_MAT_CN;
import static com.googlecode.javacpp.opencv_core.CV_MAT_DEPTH;
import static com.googlecode.javacpp.opencv_core.CV_MAT_MAGIC_VAL;
import static com.googlecode.javacpp.opencv_core.CV_MAT_TYPE;
import static com.googlecode.javacpp.opencv_core.IPL_DEPTH_16S;
import static com.googlecode.javacpp.opencv_core.IPL_DEPTH_16U;
import static com.googlecode.javacpp.opencv_core.IPL_DEPTH_1U;
import static com.googlecode.javacpp.opencv_core.IPL_DEPTH_32F;
import static com.googlecode.javacpp.opencv_core.IPL_DEPTH_32S;
import static com.googlecode.javacpp.opencv_core.IPL_DEPTH_64F;
import static com.googlecode.javacpp.opencv_core.IPL_DEPTH_8S;
import static com.googlecode.javacpp.opencv_core.IPL_DEPTH_8U;
import static com.googlecode.javacpp.opencv_core.IPL_ORIGIN_BL;
import static com.googlecode.javacpp.opencv_core.IPL_ORIGIN_TL;
import static com.googlecode.javacpp.opencv_core.cvCloneImage;
import static com.googlecode.javacpp.opencv_core.cvCloneMat;
import static com.googlecode.javacpp.opencv_core.cvCloneMatND;
import static com.googlecode.javacpp.opencv_core.cvCloneSparseMat;
import static com.googlecode.javacpp.opencv_core.cvCreateGraph;
import static com.googlecode.javacpp.opencv_core.cvCreateGraphScanner;
import static com.googlecode.javacpp.opencv_core.cvCreateImage;
import static com.googlecode.javacpp.opencv_core.cvCreateImageHeader;
import static com.googlecode.javacpp.opencv_core.cvCreateMat;
import static com.googlecode.javacpp.opencv_core.cvCreateMatHeader;
import static com.googlecode.javacpp.opencv_core.cvCreateMatND;
import static com.googlecode.javacpp.opencv_core.cvCreateMemStorage;
import static com.googlecode.javacpp.opencv_core.cvCreateSeq;
import static com.googlecode.javacpp.opencv_core.cvCreateSet;
import static com.googlecode.javacpp.opencv_core.cvCreateSparseMat;
import static com.googlecode.javacpp.opencv_core.cvGet2D;
import static com.googlecode.javacpp.opencv_core.cvGetImage;
import static com.googlecode.javacpp.opencv_core.cvGetMat;
import static com.googlecode.javacpp.opencv_core.cvOpenFileStorage;
import static com.googlecode.javacpp.opencv_core.cvReleaseFileStorage;
import static com.googlecode.javacpp.opencv_core.cvReleaseGraphScanner;
import static com.googlecode.javacpp.opencv_core.cvReleaseImage;
import static com.googlecode.javacpp.opencv_core.cvReleaseImageHeader;
import static com.googlecode.javacpp.opencv_core.cvReleaseMat;
import static com.googlecode.javacpp.opencv_core.cvReleaseMatND;
import static com.googlecode.javacpp.opencv_core.cvReleaseMemStorage;
import static com.googlecode.javacpp.opencv_core.cvReleaseSparseMat;
import static com.googlecode.javacpp.opencv_core.cvScalar;

public class opencv_core extends com.googlecode.javacpp.presets.opencv_core {

    public static abstract class AbstractArray extends Pointer {
        public AbstractArray() { }
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

        protected BufferedImage cloneBufferedImage() {
            if (bufferedImage == null) {
                return null;
            }
            BufferedImage bi = (BufferedImage)bufferedImage;
            int type = bi.getType();
            if (type == BufferedImage.TYPE_CUSTOM) {
                return new BufferedImage(bi.getColorModel(),
                        bi.copyData(null), bi.isAlphaPremultiplied(), null);
            } else {
                return new BufferedImage(bi.getWidth(), bi.getHeight(), type);
            }
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

        public CvSize cvSize() { return com.googlecode.javacpp.opencv_core.cvSize(arrayWidth(), arrayHeight()); }

        public ByteBuffer   getByteBuffer  (int index) { return arrayData().position(index).capacity(arraySize() - index).asByteBuffer(); }
        public ShortBuffer  getShortBuffer (int index) { return getByteBuffer(index*2).asShortBuffer();  }
        public IntBuffer    getIntBuffer   (int index) { return getByteBuffer(index*4).asIntBuffer();    }
        public FloatBuffer  getFloatBuffer (int index) { return getByteBuffer(index*4).asFloatBuffer();  }
        public DoubleBuffer getDoubleBuffer(int index) { return getByteBuffer(index*8).asDoubleBuffer(); }
        public ByteBuffer   getByteBuffer()   { return getByteBuffer  (0); }
        public ShortBuffer  getShortBuffer()  { return getShortBuffer (0); }
        public IntBuffer    getIntBuffer()    { return getIntBuffer   (0); }
        public FloatBuffer  getFloatBuffer()  { return getFloatBuffer (0); }
        public DoubleBuffer getDoubleBuffer() { return getDoubleBuffer(0); }

        public static final byte[]
                gamma22    = new byte[256],
                gamma22inv = new byte[256];
        static {
            for (int i = 0; i < 256; i++) {
                gamma22[i]    = (byte)Math.round(Math.pow(i/255.0,   2.2)*255.0);
                gamma22inv[i] = (byte)Math.round(Math.pow(i/255.0, 1/2.2)*255.0);
            }
        }
        public static int decodeGamma22(int value) {
            return gamma22[value & 0xFF] & 0xFF;
        }
        public static int encodeGamma22(int value) {
            return gamma22inv[value & 0xFF] & 0xFF;
        }
        public static void flipCopyWithGamma(ByteBuffer srcBuf, int srcStep,
                ByteBuffer dstBuf, int dstStep, boolean signed, double gamma, boolean flip, int channels) {
            assert srcBuf != dstBuf;
            int w = Math.min(srcStep, dstStep);
            int srcLine = srcBuf.position(), dstLine = dstBuf.position();
            byte[] buffer = new byte[channels];
            while (srcLine < srcBuf.capacity() && dstLine < dstBuf.capacity()) {
                if (flip) {
                    srcBuf.position(srcBuf.capacity() - srcLine - srcStep);
                } else {
                    srcBuf.position(srcLine);
                }
                dstBuf.position(dstLine);
                w = Math.min(Math.min(w, srcBuf.remaining()), dstBuf.remaining());
                if (signed) {
                    if (channels > 1) {
                        for (int x = 0; x < w; x+=channels) {
                            for (int z = 0; z < channels; z++) {
                                int in = srcBuf.get();
                                byte out;
                                if (gamma == 1.0) {
                                    out = (byte)in;
                                } else {
                                    out = (byte)Math.round(Math.pow((double)in/Byte.MAX_VALUE, gamma)*Byte.MAX_VALUE);
                                }
                                buffer[z] = out;
                            }
                            for (int z = channels-1; z >= 0; z--) {
                                dstBuf.put(buffer[z]);
                            }
                        }
                    } else {
                        for (int x = 0; x < w; x++) {
                            int in = srcBuf.get();
                            byte out;
                            if (gamma == 1.0) {
                                out = (byte)in;
                            } else {
                                out = (byte)Math.round(Math.pow((double)in/Byte.MAX_VALUE, gamma)*Byte.MAX_VALUE);
                            }
                            dstBuf.put(out);
                        }
                    }
                } else {
                    if (channels > 1) {
                        for (int x = 0; x < w; x+=channels) {
                            for (int z = 0; z < channels; z++) {
                                byte out;
                                int in = srcBuf.get() & 0xFF;
                                if (gamma == 1.0) {
                                    out = (byte)in;
                                } else if (gamma == 2.2) {
                                    out = gamma22[in];
                                } else if (gamma == 1/2.2) {
                                    out = gamma22inv[in];
                                } else {
                                    out = (byte)Math.round(Math.pow((double)in/0xFF, gamma)*0xFF);
                                }
                                buffer[z] = out;
                            }
                            for (int z = channels-1; z >= 0; z--) {
                                dstBuf.put(buffer[z]);
                            }
                        }
                    } else {
                        for (int x = 0; x < w; x++) {
                            byte out;
                            int in = srcBuf.get() & 0xFF;
                            if (gamma == 1.0) {
                                out = (byte)in;
                            } else if (gamma == 2.2) {
                                out = gamma22[in];
                            } else if (gamma == 1/2.2) {
                                out = gamma22inv[in];
                            } else {
                                out = (byte)Math.round(Math.pow((double)in/0xFF, gamma)*0xFF);
                            }
                            dstBuf.put(out);
                        }
                    }
                }
                srcLine += srcStep;
                dstLine += dstStep;
            }
        }
        public static void flipCopyWithGamma(ShortBuffer srcBuf, int srcStep,
                ShortBuffer dstBuf, int dstStep, boolean signed, double gamma, boolean flip, int channels) {
            assert srcBuf != dstBuf;
            int w = Math.min(srcStep, dstStep);
            int srcLine = srcBuf.position(), dstLine = dstBuf.position();
            short[] buffer = new short[channels];
            while (srcLine < srcBuf.capacity() && dstLine < dstBuf.capacity()) {
                if (flip) {
                    srcBuf.position(srcBuf.capacity() - srcLine - srcStep);
                } else {
                    srcBuf.position(srcLine);
                }
                dstBuf.position(dstLine);
                w = Math.min(Math.min(w, srcBuf.remaining()), dstBuf.remaining());
                if (signed) {
                    if (channels > 1) {
                        for (int x = 0; x < w; x+=channels) {
                            for (int z = 0; z < channels; z++) {
                                int in = srcBuf.get();
                                short out;
                                if (gamma == 1.0) {
                                    out = (short)in;
                                } else {
                                    out = (short)Math.round(Math.pow((double)in/Short.MAX_VALUE, gamma)*Short.MAX_VALUE);
                                }
                                buffer[z] = out;
                            }
                            for (int z = channels-1; z >= 0; z--) {
                                dstBuf.put(buffer[z]);
                            }
                        }
                    } else {
                        for (int x = 0; x < w; x++) {
                            int in = srcBuf.get();
                            short out;
                            if (gamma == 1.0) {
                                out = (short)in;
                            } else {
                                out = (short)Math.round(Math.pow((double)in/Short.MAX_VALUE, gamma)*Short.MAX_VALUE);
                            }
                            dstBuf.put(out);
                        }
                    }
                } else {
                    if (channels > 1) {
                        for (int x = 0; x < w; x+=channels) {
                            for (int z = 0; z < channels; z++) {
                                int in = srcBuf.get();
                                short out;
                                if (gamma == 1.0) {
                                    out = (short)in;
                                } else {
                                    out = (short)Math.round(Math.pow((double)in/0xFFFF, gamma)*0xFFFF);
                                }
                                buffer[z] = out;
                            }
                            for (int z = channels-1; z >= 0; z--) {
                                dstBuf.put(buffer[z]);
                            }
                        }
                    } else {
                        for (int x = 0; x < w; x++) {
                            int in = srcBuf.get() & 0xFFFF;
                            short out;
                            if (gamma == 1.0) {
                                out = (short)in;
                            } else {
                                out = (short)Math.round(Math.pow((double)in/0xFFFF, gamma)*0xFFFF);
                            }
                            dstBuf.put(out);
                        }
                    }
                }
                srcLine += srcStep;
                dstLine += dstStep;
            }
        }
        public static void flipCopyWithGamma(IntBuffer srcBuf, int srcStep,
                IntBuffer dstBuf, int dstStep, double gamma, boolean flip, int channels) {
            assert srcBuf != dstBuf;
            int w = Math.min(srcStep, dstStep);
            int srcLine = srcBuf.position(), dstLine = dstBuf.position();
            int[] buffer = new int[channels];
            while (srcLine < srcBuf.capacity() && dstLine < dstBuf.capacity()) {
                if (flip) {
                    srcBuf.position(srcBuf.capacity() - srcLine - srcStep);
                } else {
                    srcBuf.position(srcLine);
                }
                dstBuf.position(dstLine);
                w = Math.min(Math.min(w, srcBuf.remaining()), dstBuf.remaining());
                if (channels > 1) {
                    for (int x = 0; x < w; x+=channels) {
                        for (int z = 0; z < channels; z++) {
                            int in = srcBuf.get();
                            int out;
                            if (gamma == 1.0) {
                                out = (int)in;
                            } else {
                                out = (int)Math.round(Math.pow((double)in/Integer.MAX_VALUE, gamma)*Integer.MAX_VALUE);
                            }
                            buffer[z] = out;
                        }
                        for (int z = channels-1; z >= 0; z--) {
                            dstBuf.put(buffer[z]);
                        }
                    }
                } else {
                    for (int x = 0; x < w; x++) {
                        int in = srcBuf.get();
                        int out;
                        if (gamma == 1.0) {
                            out = in;
                        } else {
                            out = (int)Math.round(Math.pow((double)in/Integer.MAX_VALUE, gamma)*Integer.MAX_VALUE);
                        }
                        dstBuf.put(out);
                    }
                }
                srcLine += srcStep;
                dstLine += dstStep;
            }
        }
        public static void flipCopyWithGamma(FloatBuffer srcBuf, int srcStep,
                FloatBuffer dstBuf, int dstStep, double gamma, boolean flip, int channels) {
            assert srcBuf != dstBuf;
            int w = Math.min(srcStep, dstStep);
            int srcLine = srcBuf.position(), dstLine = dstBuf.position();
            float[] buffer = new float[channels];
            while (srcLine < srcBuf.capacity() && dstLine < dstBuf.capacity()) {
                if (flip) {
                    srcBuf.position(srcBuf.capacity() - srcLine - srcStep);
                } else {
                    srcBuf.position(srcLine);
                }
                dstBuf.position(dstLine);
                w = Math.min(Math.min(w, srcBuf.remaining()), dstBuf.remaining());
                if (channels > 1) {
                    for (int x = 0; x < w; x+=channels) {
                        for (int z = 0; z < channels; z++) {
                            float in = srcBuf.get();
                            float out;
                            if (gamma == 1.0) {
                                out = in;
                            } else {
                                out = (float)Math.pow(in, gamma);
                            }
                            buffer[z] = out;
                        }
                        for (int z = channels-1; z >= 0; z--) {
                            dstBuf.put(buffer[z]);
                        }
                    }
                } else {
                    for (int x = 0; x < w; x++) {
                        float in = srcBuf.get();
                        float out;
                        if (gamma == 1.0) {
                            out = in;
                        } else {
                            out = (float)Math.pow(in, gamma);
                        }
                        dstBuf.put(out);
                    }
                }
                srcLine += srcStep;
                dstLine += dstStep;
            }
        }
        public static void flipCopyWithGamma(DoubleBuffer srcBuf, int srcStep,
                DoubleBuffer dstBuf, int dstStep, double gamma, boolean flip, int channels) {
            assert srcBuf != dstBuf;
            int w = Math.min(srcStep, dstStep);
            int srcLine = srcBuf.position(), dstLine = dstBuf.position();
            double[] buffer = new double[channels];
            while (srcLine < srcBuf.capacity() && dstLine < dstBuf.capacity()) {
                if (flip) {
                    srcBuf.position(srcBuf.capacity() - srcLine - srcStep);
                } else {
                    srcBuf.position(srcLine);
                }
                dstBuf.position(dstLine);
                w = Math.min(Math.min(w, srcBuf.remaining()), dstBuf.remaining());
                if (channels > 1) {
                    for (int x = 0; x < w; x+=channels) {
                        for (int z = 0; z < channels; z++) {
                            double in = srcBuf.get();
                            double out;
                            if (gamma == 1.0) {
                                out = in;
                            } else {
                                out = Math.pow(in, gamma);
                            }
                            buffer[z] = out;
                        }
                        for (int z = channels-1; z >= 0; z--) {
                            dstBuf.put(buffer[z]);
                        }
                    }
                } else {
                    for (int x = 0; x < w; x++) {
                        double in = srcBuf.get();
                        double out;
                        if (gamma == 1.0) {
                            out = in;
                        } else {
                            out = Math.pow(in, gamma);
                        }
                        dstBuf.put(out);
                    }
                }
                srcLine += srcStep;
                dstLine += dstStep;
            }
        }
        public void applyGamma(double gamma) {
            if (gamma == 1.0) {
                return;
            }
            switch (arrayDepth()) {
                case IPL_DEPTH_8U:
                    flipCopyWithGamma(getByteBuffer(), arrayStep(), getByteBuffer(), arrayStep(), false, gamma, false, 0);
                    break;
                case IPL_DEPTH_8S:
                    flipCopyWithGamma(getByteBuffer(), arrayStep(), getByteBuffer(), arrayStep(), true, gamma, false, 0);
                    break;
                case IPL_DEPTH_16U:
                    flipCopyWithGamma(getShortBuffer(), arrayStep()/2, getShortBuffer(), arrayStep()/2, false, gamma, false, 0);
                    break;
                case IPL_DEPTH_16S:
                    flipCopyWithGamma(getShortBuffer(), arrayStep()/2, getShortBuffer(), arrayStep()/2, true, gamma, false, 0);
                    break;
                case IPL_DEPTH_32S:
                    flipCopyWithGamma(getFloatBuffer(), arrayStep()/4, getFloatBuffer(), arrayStep()/4, gamma, false, 0);
                    break;
                case IPL_DEPTH_32F:
                    flipCopyWithGamma(getFloatBuffer(), arrayStep()/4, getFloatBuffer(), arrayStep()/4, gamma, false, 0);
                    break;
                case IPL_DEPTH_64F:
                    flipCopyWithGamma(getDoubleBuffer(), arrayStep()/8, getDoubleBuffer(), arrayStep()/8, gamma, false, 0);
                    break;
                default:
                    assert false;
            }
        }


        public void copyTo(BufferedImage image) {
            copyTo(image, 1.0);
        }
        public void copyTo(BufferedImage image, double gamma) {
            copyTo(image, gamma, false);
        }
        public void copyTo(BufferedImage image, double gamma, boolean flipChannels) {
            Rectangle r = null;
            IplROI roi = arrayROI();
            if (roi != null) {
                r = new Rectangle(roi.xOffset(), roi.yOffset(), roi.width(), roi.height());
            }
            copyTo(image, gamma, flipChannels, r);
        }
        public void copyTo(BufferedImage image, double gamma, boolean flipChannels, Rectangle roi) {
            boolean flip = arrayOrigin() == IPL_ORIGIN_BL; // need to add support for ROI..

            ByteBuffer in  = getByteBuffer(roi == null ? 0 : roi.y*arrayStep() + roi.x*arrayChannels());
            SampleModel sm = image.getSampleModel();
            Raster r       = image.getRaster();
            DataBuffer out = r.getDataBuffer();
            int x = -r.getSampleModelTranslateX();
            int y = -r.getSampleModelTranslateY();
            int step = sm.getWidth()*sm.getNumBands();
            int channels = sm.getNumBands();
            if (sm instanceof ComponentSampleModel) {
                step = ((ComponentSampleModel)sm).getScanlineStride();
                channels = ((ComponentSampleModel)sm).getPixelStride();
            } else if (sm instanceof SinglePixelPackedSampleModel) {
                step = ((SinglePixelPackedSampleModel)sm).getScanlineStride();
                channels = 1;
            } else if (sm instanceof MultiPixelPackedSampleModel) {
                step = ((MultiPixelPackedSampleModel)sm).getScanlineStride();
                channels = ((MultiPixelPackedSampleModel)sm).getPixelBitStride()/8; // ??
            }
            int start = y*step + x*channels;

            if (out instanceof DataBufferByte) {
                byte[] a = ((DataBufferByte)out).getData();
                flipCopyWithGamma(in, arrayStep(), ByteBuffer.wrap(a, start, a.length - start), step, false, gamma, flip, flipChannels ? channels : 0);
            } else if (out instanceof DataBufferDouble) {
                double[] a = ((DataBufferDouble)out).getData();
                flipCopyWithGamma(in.asDoubleBuffer(), arrayStep()/8, DoubleBuffer.wrap(a, start, a.length - start), step, gamma, flip, flipChannels ? channels : 0);
            } else if (out instanceof DataBufferFloat) {
                float[] a = ((DataBufferFloat)out).getData();
                flipCopyWithGamma(in.asFloatBuffer(), arrayStep()/4, FloatBuffer.wrap(a, start, a.length - start), step, gamma, flip, flipChannels ? channels : 0);
            } else if (out instanceof DataBufferInt) {
                int[] a = ((DataBufferInt)out).getData();
                flipCopyWithGamma(in.asIntBuffer(), arrayStep()/4, IntBuffer.wrap(a, start, a.length - start), step, gamma, flip, flipChannels ? channels : 0);
            } else if (out instanceof DataBufferShort) {
                short[] a = ((DataBufferShort)out).getData();
                flipCopyWithGamma(in.asShortBuffer(), arrayStep()/2, ShortBuffer.wrap(a, start, a.length - start), step, true, gamma, flip, flipChannels ? channels : 0);
            } else if (out instanceof DataBufferUShort) {
                short[] a = ((DataBufferUShort)out).getData();
                flipCopyWithGamma(in.asShortBuffer(), arrayStep()/2, ShortBuffer.wrap(a, start, a.length - start), step, false, gamma, flip, flipChannels ? channels : 0);
            } else {
                assert false;
            }
        }

        public void copyFrom(BufferedImage image) {
            copyFrom(image, 1.0);
        }
        public void copyFrom(BufferedImage image, double gamma) {
            copyFrom(image, gamma, false);
        }
        public void copyFrom(BufferedImage image, double gamma, boolean flipChannels) {
            Rectangle r = null;
            IplROI roi = arrayROI();
            if (roi != null) {
                r = new Rectangle(roi.xOffset(), roi.yOffset(), roi.width(), roi.height());
            }
            copyFrom(image, gamma, flipChannels, r);
        }
        public void copyFrom(BufferedImage image, double gamma, boolean flipChannels, Rectangle roi) {
            arrayOrigin(IPL_ORIGIN_TL);

            ByteBuffer out = getByteBuffer(roi == null ? 0 : roi.y*arrayStep() + roi.x);
            SampleModel sm = image.getSampleModel();
            Raster r       = image.getRaster();
            DataBuffer in  = r.getDataBuffer();
            int x = -r.getSampleModelTranslateX();
            int y = -r.getSampleModelTranslateY();
            int step = sm.getWidth()*sm.getNumBands();
            int channels = sm.getNumBands();
            if (sm instanceof ComponentSampleModel) {
                step = ((ComponentSampleModel)sm).getScanlineStride();
                channels = ((ComponentSampleModel)sm).getPixelStride();
            } else if (sm instanceof SinglePixelPackedSampleModel) {
                step = ((SinglePixelPackedSampleModel)sm).getScanlineStride();
                channels = 1;
            } else if (sm instanceof MultiPixelPackedSampleModel) {
                step = ((MultiPixelPackedSampleModel)sm).getScanlineStride();
                channels = ((MultiPixelPackedSampleModel)sm).getPixelBitStride()/8; // ??
            }
            int start = y*step + x*channels;

            if (in instanceof DataBufferByte) {
                byte[] a = ((DataBufferByte)in).getData();
                flipCopyWithGamma(ByteBuffer.wrap(a, start, a.length - start), step, out, arrayStep(), false, gamma, false, flipChannels ? channels : 0);
            } else if (in instanceof DataBufferDouble) {
                double[] a = ((DataBufferDouble)in).getData();
                flipCopyWithGamma(DoubleBuffer.wrap(a, start, a.length - start), step, out.asDoubleBuffer(), arrayStep()/8, gamma, false, flipChannels ? channels : 0);
            } else if (in instanceof DataBufferFloat) {
                float[] a = ((DataBufferFloat)in).getData();
                flipCopyWithGamma(FloatBuffer.wrap(a, start, a.length - start), step, out.asFloatBuffer(), arrayStep()/4, gamma, false, flipChannels ? channels : 0);
            } else if (in instanceof DataBufferInt) {
                int[] a = ((DataBufferInt)in).getData();
                flipCopyWithGamma(IntBuffer.wrap(a, start, a.length - start), step, out.asIntBuffer(), arrayStep()/4, gamma, false, flipChannels ? channels : 0);
            } else if (in instanceof DataBufferShort) {
                short[] a = ((DataBufferShort)in).getData();
                flipCopyWithGamma(ShortBuffer.wrap(a, start, a.length - start), step, out.asShortBuffer(), arrayStep()/2, true, gamma, false, flipChannels ? channels : 0);
            } else if (in instanceof DataBufferUShort) {
                short[] a = ((DataBufferUShort)in).getData();
                flipCopyWithGamma(ShortBuffer.wrap(a, start, a.length - start), step, out.asShortBuffer(), arrayStep()/2, false, gamma, false, flipChannels ? channels : 0);
            } else {
                assert false;
            }
            if (bufferedImage == null && roi == null &&
                    image.getWidth() == arrayWidth() && image.getHeight() == arrayHeight()) {
                bufferedImage = image;
            }
        }
        // not declared as BufferedImage => Android friendly
        protected Object bufferedImage = null;
        public int getBufferedImageType() {
            // precanned BufferedImage types are confusing... in practice though,
            // they all use the sRGB color model when blitting:
            //     http://bugs.sun.com/bugdatabase/view_bug.do?bug_id=5051418
            // and we should use them because they are *A LOT* faster with Java 2D.
            // workaround: do gamma correction ourselves ("gamma" parameter)
            //             since we'll never use getRGB() and setRGB(), right?
            int type = BufferedImage.TYPE_CUSTOM;
            if (arrayChannels() == 1) {
                if (arrayDepth() == IPL_DEPTH_8U || arrayDepth() == IPL_DEPTH_8S) {
                    type = BufferedImage.TYPE_BYTE_GRAY;
                } else if (arrayDepth() == IPL_DEPTH_16U) {
                    type = BufferedImage.TYPE_USHORT_GRAY;
                }
            } else if (arrayChannels() == 3) {
                if (arrayDepth() == IPL_DEPTH_8U || arrayDepth() == IPL_DEPTH_8S) {
                    type = BufferedImage.TYPE_3BYTE_BGR;
                }
            } else if (arrayChannels() == 4) {
                // The channels end up reversed of what we need for OpenCL.
                // We work around this in copyTo() and copyFrom() by
                // inversing the channels to let us use RGBA in our IplImage.
                if (arrayDepth() == IPL_DEPTH_8U || arrayDepth() == IPL_DEPTH_8S) {
                    type = BufferedImage.TYPE_4BYTE_ABGR;
                }
            }
            return type;
        }
        public BufferedImage getBufferedImage() {
            return getBufferedImage(1.0);
        }
        public BufferedImage getBufferedImage(double gamma) {
            return getBufferedImage(gamma, false);
        }
        public BufferedImage getBufferedImage(double gamma, boolean flipChannels) {
            return getBufferedImage(gamma, flipChannels, null);
        }
        public BufferedImage getBufferedImage(double gamma, boolean flipChannels, ColorSpace cs) {
            int type = getBufferedImageType();

            if (bufferedImage == null && type != BufferedImage.TYPE_CUSTOM && cs == null) {
                bufferedImage = new BufferedImage(arrayWidth(), arrayHeight(), type);
            }

            if (bufferedImage == null) {
                boolean alpha = false;
                int[] offsets = null;
                if (arrayChannels() == 1) {
                    alpha = false;
                    if (cs == null) {
                        cs = ColorSpace.getInstance(ColorSpace.CS_GRAY);
                    }
                    offsets = new int[] {0};
                } else if (arrayChannels() == 3) {
                    alpha = false;
                    if (cs == null) {
                        cs = ColorSpace.getInstance(ColorSpace.CS_LINEAR_RGB);
                    }
                    // raster in "BGR" order like OpenCV..
                    offsets = new int[] {2, 1, 0};
                } else if (arrayChannels() == 4) {
                    alpha = true;
                    if (cs == null) {
                        cs = ColorSpace.getInstance(ColorSpace.CS_LINEAR_RGB);
                    }
                    // raster in "RGBA" order for OpenCL.. alpha needs to be last
                    offsets = new int[] {0, 1, 2, 3};
                } else {
                    assert false;
                }

                ColorModel cm = null;
                WritableRaster wr = null;
                if (arrayDepth() == IPL_DEPTH_8U || arrayDepth() == IPL_DEPTH_8S) {
                    cm = new ComponentColorModel(cs, alpha,
                            false, Transparency.OPAQUE, DataBuffer.TYPE_BYTE);
                    wr = Raster.createWritableRaster(new ComponentSampleModel(
                            DataBuffer.TYPE_BYTE, arrayWidth(), arrayHeight(), arrayChannels(), arrayStep(),
                            offsets), null);
                } else if (arrayDepth() == IPL_DEPTH_16U) {
                    cm = new ComponentColorModel(cs, alpha,
                            false, Transparency.OPAQUE, DataBuffer.TYPE_USHORT);
                    wr = Raster.createWritableRaster(new ComponentSampleModel(
                            DataBuffer.TYPE_USHORT, arrayWidth(), arrayHeight(), arrayChannels(), arrayStep()/2,
                            offsets), null);
                } else if (arrayDepth() == IPL_DEPTH_16S) {
                    cm = new ComponentColorModel(cs, alpha,
                            false, Transparency.OPAQUE, DataBuffer.TYPE_SHORT);
                    wr = Raster.createWritableRaster(new ComponentSampleModel(
                            DataBuffer.TYPE_SHORT, arrayWidth(), arrayHeight(), arrayChannels(), arrayStep()/2,
                            offsets), null);
                } else if (arrayDepth() == IPL_DEPTH_32S) {
                    cm = new ComponentColorModel(cs, alpha,
                            false, Transparency.OPAQUE, DataBuffer.TYPE_INT);
                    wr = Raster.createWritableRaster(new ComponentSampleModel(
                            DataBuffer.TYPE_INT, arrayWidth(), arrayHeight(), arrayChannels(), arrayStep()/4,
                            offsets), null);
                } else if (arrayDepth() == IPL_DEPTH_32F) {
                    cm = new ComponentColorModel(cs, alpha,
                            false, Transparency.OPAQUE, DataBuffer.TYPE_FLOAT);
                    wr = Raster.createWritableRaster(new ComponentSampleModel(
                            DataBuffer.TYPE_FLOAT, arrayWidth(), arrayHeight(), arrayChannels(), arrayStep()/4,
                            offsets), null);
                } else if (arrayDepth() == IPL_DEPTH_64F) {
                    cm = new ComponentColorModel(cs, alpha,
                            false, Transparency.OPAQUE, DataBuffer.TYPE_DOUBLE);
                    wr = Raster.createWritableRaster(new ComponentSampleModel(
                            DataBuffer.TYPE_DOUBLE, arrayWidth(), arrayHeight(), arrayChannels(), arrayStep()/8,
                            offsets), null);
                } else {
                    assert false;
                }

                bufferedImage = new BufferedImage(cm, wr, false, null);
            }

            if (bufferedImage != null) {
                IplROI roi = arrayROI();
                if (roi != null) {
                    copyTo(((BufferedImage)bufferedImage).getSubimage(roi.xOffset(), roi.yOffset(), roi.width(), roi.height()), gamma, flipChannels);
                } else {
                    copyTo((BufferedImage)bufferedImage, gamma, flipChannels);
                }
            }

            return (BufferedImage)bufferedImage;
        }

        @Override public String toString() {
            if (isNull()) {
                return super.toString();
            } else {
                try {
                    return "AbstractArray[width=" + arrayWidth() + ",height=" + arrayHeight()
                                      + ",depth=" + arrayDepth() + ",channels=" + arrayChannels() + "]";
                } catch (Exception e) {
                    return super.toString();
                }
            }
        }
    }

    @Opaque public static class CvArr extends AbstractArray {
        public CvArr() { }
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
        public CvArrArray(int size) { super(size); allocateArray(size); }
        public CvArrArray(Pointer p) { super(p); }
        private native void allocateArray(int size);

        @Override public CvArrArray position(int position) {
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
        public CvMatArray(int size) { allocateArray(size); }
        public CvMatArray(Pointer p) { super(p); }
        private native void allocateArray(int size);

        @Override public CvMatArray position(int position) {
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
        public CvMatNDArray(int size) { allocateArray(size); }
        public CvMatNDArray(Pointer p) { super(p); }
        private native void allocateArray(int size);

        @Override public CvMatNDArray position(int position) {
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
        public IplImageArray(int size) { allocateArray(size); }
        public IplImageArray(Pointer p) { super(p); }
        private native void allocateArray(int size);

        @Override public IplImageArray position(int position) {
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
        public AbstractIplImage() { }
        public AbstractIplImage(Pointer p) { super(p); }

        public static IplImage create(CvSize size, int depth, int channels) {
            IplImage i = cvCreateImage(size, depth, channels);
            if (i != null) {
                i.deallocator(new ReleaseDeallocator(i));
            }
            return i;
        }
        public static IplImage create(int width, int height, int depth, int channels) {
            return create(com.googlecode.javacpp.opencv_core.cvSize(width, height), depth, channels);
        }
        public static IplImage create(CvSize size, int depth, int channels, int origin) {
            IplImage i = create(size, depth, channels);
            if (i != null) {
                i.origin(origin);
            }
            return i;
        }
        public static IplImage create(int width, int height, int depth, int channels, int origin) {
            IplImage i = create(width, height, depth, channels);
            if (i != null) {
                i.origin(origin);
            }
            return i;
        }

        public static IplImage createHeader(CvSize size, int depth, int channels) {
            IplImage i = cvCreateImageHeader(size, depth, channels);
            if (i != null) {
                i.deallocator(new HeaderReleaseDeallocator(i));
            }
            return i;
        }
        public static IplImage createHeader(int width, int height, int depth, int channels) {
            return createHeader(com.googlecode.javacpp.opencv_core.cvSize(width, height), depth, channels);
        }
        public static IplImage createHeader(CvSize size, int depth, int channels, int origin) {
            IplImage i = createHeader(size, depth, channels);
            if (i != null) {
                i.origin(origin);
            }
            return i;
        }
        public static IplImage createHeader(int width, int height, int depth, int channels, int origin) {
            IplImage i = createHeader(width, height, depth, channels);
            if (i != null) {
                i.origin(origin);
            }
            return i;
        }

        public static IplImage createCompatible(IplImage template) {
            return createIfNotCompatible(null, template);
        }
        public static IplImage createIfNotCompatible(IplImage image, IplImage template) {
            if (image == null || image.width() != template.width() || image.height() != template.height() ||
                    image.depth() != template.depth() || image.nChannels() != template.nChannels()) {
                image = create(template.width(), template.height(),
                        template.depth(), template.nChannels(), template.origin());
                if (((AbstractIplImage)template).bufferedImage != null) {
                    ((AbstractIplImage)template).bufferedImage = template.cloneBufferedImage();
                }
            }
            image.origin(template.origin());
            return image;
        }

        public static IplImage createFrom(BufferedImage image) {
            return createFrom(image, 1.0);
        }
        public static IplImage createFrom(BufferedImage image, double gamma) {
            return createFrom(image, gamma, false);
        }
        public static IplImage createFrom(BufferedImage image, double gamma, boolean flipChannels) {
            if (image == null) {
                return null;
            }
            SampleModel sm = image.getSampleModel();
            int depth = 0, numChannels = sm.getNumBands();
            switch (image.getType()) {
                case BufferedImage.TYPE_INT_RGB:
                case BufferedImage.TYPE_INT_ARGB:
                case BufferedImage.TYPE_INT_ARGB_PRE:
                case BufferedImage.TYPE_INT_BGR:
                    depth = IPL_DEPTH_8U;
                    numChannels = 4;
                    break;
            }
            if (depth == 0 || numChannels == 0) {
                switch (sm.getDataType()) {
                    case DataBuffer.TYPE_BYTE:   depth = IPL_DEPTH_8U;  break;
                    case DataBuffer.TYPE_USHORT: depth = IPL_DEPTH_16U; break;
                    case DataBuffer.TYPE_SHORT:  depth = IPL_DEPTH_16S; break;
                    case DataBuffer.TYPE_INT:    depth = IPL_DEPTH_32S; break;
                    case DataBuffer.TYPE_FLOAT:  depth = IPL_DEPTH_32F; break;
                    case DataBuffer.TYPE_DOUBLE: depth = IPL_DEPTH_64F; break;
                    default: assert false;
                }
            }
            IplImage i = create(image.getWidth(), image.getHeight(), depth, numChannels);
            i.copyFrom(image, gamma, flipChannels);
            return i;
        }

        @Override public IplImage clone() {
            IplImage i = cvCloneImage((IplImage)this);
            if (i != null) {
                i.deallocator(new ReleaseDeallocator(i));
            }
            if (i != null && bufferedImage != null) {
                ((AbstractIplImage)i).bufferedImage = cloneBufferedImage();
            }
            return i;
        }

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

        @Override public String toString() {
            if (isNull()) {
                return super.toString();
            } else {
                return "IplImage[width=" + width() + ",height=" + height() +
                               ",depth=" + depth() + ",nChannels=" + nChannels() + "]";
            }
        }
    }

    public static abstract class AbstractCvMat extends CvArr {
        public AbstractCvMat() { }
        public AbstractCvMat(Pointer p) { super(p); }

        public static CvMat create(int rows, int cols, int type) {
            CvMat m = cvCreateMat(rows, cols, type);
            if (m != null) {
                ((AbstractCvMat)m).fullSize = m.size();
                ((AbstractCvMat)m).deallocator(new ReleaseDeallocator(m));
            }
            return m;
        }
        public static CvMat create(int rows, int cols, int depth, int channels) {
            return create(rows, cols, CV_MAKETYPE(depth, channels));
        }
        public static CvMat create(int rows, int cols) {
            return create(rows, cols, CV_64F, 1);
        }

        public static CvMat createHeader(int rows, int cols, int type) {
            CvMat m = cvCreateMatHeader(rows, cols, type);
            if (m != null) {
                ((AbstractCvMat)m).fullSize = m.size();
                ((AbstractCvMat)m).deallocator(new ReleaseDeallocator(m));
            }
            return m;
        }
        public static CvMat createHeader(int rows, int cols, int depth, int channels) {
            return createHeader(rows, cols, CV_MAKETYPE(depth, channels));
        }
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

        @Override public CvMat clone() {
            CvMat m = cvCloneMat((CvMat)this);
            if (m != null) {
                m.deallocator(new ReleaseDeallocator(m));
            }
            return m;
        }

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

        public void reset() {
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
        public ByteBuffer getByteBuffer() {
            if (byteBuffer == null) {
                byteBuffer = data_ptr().capacity(fullSize()).asBuffer();
            }
            byteBuffer.position(0);
            return byteBuffer;
        }
        public ShortBuffer getShortBuffer() {
            if (shortBuffer == null) {
                shortBuffer = data_s().capacity(fullSize()/2).asBuffer();
            }
            shortBuffer.position(0);
            return shortBuffer;
        }
        public IntBuffer getIntBuffer() {
            if (intBuffer == null) {
                intBuffer = data_i().capacity(fullSize()/4).asBuffer();
            }
            intBuffer.position(0);
            return intBuffer;
        }
        public FloatBuffer getFloatBuffer() {
            if (floatBuffer == null) {
                floatBuffer = data_fl().capacity(fullSize()/4).asBuffer();
            }
            floatBuffer.position(0);
            return floatBuffer;
        }
        public DoubleBuffer getDoubleBuffer() {
            if (doubleBuffer == null) {
                doubleBuffer = data_db().capacity(fullSize()/8).asBuffer();
            }
            doubleBuffer.position(0);
            return doubleBuffer;
        }

        public double get(int i) {
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
        public double get(int i, int j) {
            return get(i*step()/elemSize() + j*channels());
        }

        public double get(int i, int j, int k) {
            return get(i*step()/elemSize() + j*channels() + k);
        }
        public synchronized CvMat get(int index, double[] vv, int offset, int length) {
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
        public CvMat get(int index, double[] vv) {
            return get(index, vv, 0, vv.length);
        }
        public CvMat get(double[] vv) {
            return get(0, vv);
        }
        public double[] get() {
            double[] vv = new double[fullSize()/elemSize()];
            get(vv);
            return vv;
        }

        public CvMat put(int i, double v) {
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
        public CvMat put(int i, int j, double v) {
            return put(i*step()/elemSize() + j*channels(), v);
        }
        public CvMat put(int i, int j, int k, double v) {
            return put(i*step()/elemSize() + j*channels() + k, v);
        }
        public synchronized CvMat put(int index, double[] vv, int offset, int length) {
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
        public CvMat put(int index, double ... vv) {
            return put(index, vv, 0, vv.length);
        }
        public CvMat put(double ... vv) {
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
        public AbstractCvMatND() { }
        public AbstractCvMatND(Pointer p) { super(p); }

        public static CvMatND create(int dims, int[] sizes, int type) {
            CvMatND m = cvCreateMatND(dims, sizes, type);
            if (m != null) {
                ((AbstractCvMatND)m).deallocator(new ReleaseDeallocator(m));
            }
            return m;
        }

        @Override public CvMatND clone() {
            CvMatND m = cvCloneMatND((CvMatND)this);
            if (m != null) {
                m.deallocator(new ReleaseDeallocator(m));
            }
            return m;
        }

        public void release() {
            deallocate();
        }
        protected static class ReleaseDeallocator extends CvMatND implements Pointer.Deallocator {
            ReleaseDeallocator(CvMatND p) { super(p); }
            @Override public void deallocate() { cvReleaseMatND(this); }
        }
    }

    public static abstract class AbstractCvSparseMat extends CvArr {
        public AbstractCvSparseMat() { }
        public AbstractCvSparseMat(Pointer p) { super(p); }

        public static CvSparseMat create(int dims, int[] sizes, int type) {
            CvSparseMat m = cvCreateSparseMat(dims, sizes, type);
            if (m != null) {
                m.deallocator(new ReleaseDeallocator(m));
            }
            return m;
        }

        @Override public CvSparseMat clone() {
            CvSparseMat m = cvCloneSparseMat((CvSparseMat)this);
            if (m != null) {
                m.deallocator(new ReleaseDeallocator(m));
            }
            return m;
        }
        public void release() {
            deallocate();
        }
        protected static class ReleaseDeallocator extends CvSparseMat implements Pointer.Deallocator {
            ReleaseDeallocator(CvSparseMat p) { super(p); }
            @Override public void deallocate() { cvReleaseSparseMat(this); }
        }
    }

    public static abstract class AbstractCvRect extends Pointer {
        public AbstractCvRect() { }
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
                int p = position();
                for (int i = 0; i < capacity(); i++) {
                    position(i);
                    s += (i == 0 ? "(" : " (") + x() + ", " + y() + "; " + width() + ", " + height() + ")";
                }
                position(p);
                return s;
            }
        }
    }

    public static abstract class AbstractCvPoint extends Pointer {
        public AbstractCvPoint() { }
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

        public int[] get() {
            int[] pts = new int[capacity == 0 ? 2 : 2*capacity];
            get(pts);
            return pts;
        }
        public CvPoint get(int[] pts) {
            return get(pts, 0, pts.length);
        }
        public CvPoint get(int[] pts, int offset, int length) {
            for (int i = 0; i < length/2; i++) {
                position(i);
                pts[offset + i*2  ] = x();
                pts[offset + i*2+1] = y();
            }
            return position(0);
        }

        public final CvPoint put(int[] pts, int offset, int length) {
            for (int i = 0; i < length/2; i++) {
                position(i); put(pts[offset + i*2], pts[offset + i*2+1]);
            }
            return position(0);
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
                int p = position();
                for (int i = 0; i < capacity(); i++) {
                    position(i);
                    s += (i == 0 ? "(" : " (") + x() + ", " + y() + ")";
                }
                position(p);
                return s;
            }
        }

        public static final CvPoint ZERO = new CvPoint().x(0).y(0);
    }

    public static abstract class AbstractCvPoint2D32f extends Pointer {
        public AbstractCvPoint2D32f() { }
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

        public double[] get() {
            double[] pts = new double[capacity == 0 ? 2 : 2*capacity];
            get(pts);
            return pts;
        }
        public CvPoint2D32f get(double[] pts) {
            return get(pts, 0, pts.length);
        }
        public CvPoint2D32f get(double[] pts, int offset, int length) {
            for (int i = 0; i < length/2; i++) {
                position(i);
                pts[offset + i*2  ] = x();
                pts[offset + i*2+1] = y();
            }
            return position(0);
        }

        public final CvPoint2D32f put(double[] pts, int offset, int length) {
            for (int i = 0; i < length/2; i++) {
                position(i); put(pts[offset + i*2], pts[offset + i*2+1]);
            }
            return position(0);
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
                int p = position();
                for (int i = 0; i < capacity(); i++) {
                    position(i);
                    s += (i == 0 ? "(" : " (") + x() + ", " + y() + ")";
                }
                position(p);
                return s;
            }
        }
    }

    public static abstract class AbstractCvPoint3D32f extends Pointer {
        public AbstractCvPoint3D32f() { }
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

        public double[] get() {
            double[] pts = new double[capacity == 0 ? 3 : 3*capacity];
            get(pts);
            return pts;
        }
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
            return position(0);
        }

        public final CvPoint3D32f put(double[] pts, int offset, int length) {
            for (int i = 0; i < length/3; i++) {
                position(i); put(pts[offset + i*3], pts[offset + i*3+1], pts[offset + i*3+2]);
            }
            return position(0);
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
                int p = position();
                for (int i = 0; i < capacity(); i++) {
                    position(i);
                    s += (i == 0 ? "(" : " (") + x() + ", " + y() + ", " + z() + ")";
                }
                position(p);
                return s;
            }
        }
    }

    public static abstract class AbstractCvPoint2D64f extends Pointer {
        public AbstractCvPoint2D64f() { }
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

        public double[] get() {
            double[] pts = new double[capacity == 0 ? 2 : 2*capacity];
            get(pts);
            return pts;
        }
        public CvPoint2D64f get(double[] pts) {
            return get(pts, 0, pts.length);
        }
        public CvPoint2D64f get(double[] pts, int offset, int length) {
            for (int i = 0; i < length/2; i++) {
                position(i);
                pts[offset + i*2  ] = x();
                pts[offset + i*2+1] = y();
            }
            return position(0);
        }

        public final CvPoint2D64f put(double[] pts, int offset, int length) {
            for (int i = 0; i < length/2; i++) {
                position(i); put(pts[offset + i*2], pts[offset + i*2+1]);
            }
            return position(0);
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
                int p = position();
                for (int i = 0; i < capacity(); i++) {
                    position(i);
                    s += (i == 0 ? "(" : " (") + (float)x() + ", " + (float)y() + ")";
                }
                position(p);
                return s;
            }
        }
    }

    public static abstract class AbstractCvPoint3D64f extends Pointer {
        public AbstractCvPoint3D64f() { }
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

        public double[] get() {
            double[] pts = new double[capacity == 0 ? 3 : 3*capacity];
            get(pts);
            return pts;
        }
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
            return position(0);
        }

        public final CvPoint3D64f put(double[] pts, int offset, int length) {
            for (int i = 0; i < length/3; i++) {
                position(i); put(pts[offset + i*3], pts[offset + i*3+1], pts[offset + i*3+2]);
            }
            return position(0);
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
                int p = position();
                for (int i = 0; i < capacity(); i++) {
                    position(i);
                    s += (i == 0 ? "(" : " (") + (float)x() + ", " + (float)y() + ", " + (float)z() + ")";
                }
                position(p);
                return s;
            }
        }
    }

    public static abstract class AbstractCvSize extends Pointer {
        public AbstractCvSize() { }
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
                int p = position();
                for (int i = 0; i < capacity(); i++) {
                    position(i);
                    s += (i == 0 ? "(" : " (") + width() + ", " + height() + ")";
                }
                position(p);
                return s;
            }
        }
        public static final CvSize ZERO = new CvSize().width(0).height(0);
    }

    public static abstract class AbstractCvSize2D32f extends Pointer {
        public AbstractCvSize2D32f() { }
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
                int p = position();
                for (int i = 0; i < capacity(); i++) {
                    position(i);
                    s += (i == 0 ? "(" : " (") + width() + ", " + height() + ")";
                }
                position(p);
                return s;
            }
        }
    }

    public static abstract class AbstractCvBox2D extends Pointer {
        public AbstractCvBox2D() { }
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
                int p = position();
                for (int i = 0; i < capacity(); i++) {
                    position(i);
                    s += (i == 0 ? "(" : " (") + center() + ", " + size() + ", " + angle() + ")";
                }
                position(p);
                return s;
            }
        }
    }

    public static abstract class AbstractCvScalar extends Pointer {
        public AbstractCvScalar() { }
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
                int p = position();
                for (int i = 0; i < capacity(); i++) {
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
        public AbstractCvMemStorage() { }
        public AbstractCvMemStorage(Pointer p) { super(p); }

        public static CvMemStorage create(int block_size) {
            CvMemStorage m = cvCreateMemStorage(block_size);
            if (m != null) {
                m.deallocator(new ReleaseDeallocator(m));
            }
            return m;
        }
        public static CvMemStorage create() {
            return create(0);
        }

        public void release() {
            deallocate();
        }
        protected static class ReleaseDeallocator extends CvMemStorage implements Deallocator {
            ReleaseDeallocator(CvMemStorage p) { super(p); }
            @Override public void deallocate() { cvReleaseMemStorage(this); }
        }
    }

    public static abstract class AbstractCvSeq extends CvArr {
        public AbstractCvSeq() { }
        public AbstractCvSeq(Pointer p) { super(p); }

        public static CvSeq create(int seq_flags, int header_size, int elem_size, CvMemStorage storage) {
            return cvCreateSeq(seq_flags, header_size, elem_size, storage);
        }
    }

    public static abstract class AbstractCvSet extends CvSeq {
        public AbstractCvSet() { }
        public AbstractCvSet(Pointer p) { super(p); }

        public static CvSet create(int set_flags, int header_size, int elem_size,
                CvMemStorage storage) {
            return cvCreateSet(set_flags, header_size, elem_size, storage);
        }
    }

    public static abstract class AbstractCvGraph extends CvSet {
        public AbstractCvGraph() { }
        public AbstractCvGraph(Pointer p) { super(p); }

        public static CvGraph create(int graph_flags, int header_size, int vtx_size,
                int edge_size, CvMemStorage storage) {
            return cvCreateGraph(graph_flags, header_size, vtx_size, edge_size, storage);
        }
    }

    public static abstract class AbstractCvFileStorage extends Pointer {
        public AbstractCvFileStorage() { }
        public AbstractCvFileStorage(Pointer p) { super(p); }

        public static CvFileStorage open(String filename, CvMemStorage memstorage, int flags) {
            return open(filename, memstorage, flags, null);
        }
        public static CvFileStorage open(String filename, CvMemStorage memstorage, int flags, String encoding) {
            CvFileStorage f = cvOpenFileStorage(filename, memstorage, flags, encoding);
            if (f != null) {
                f.deallocator(new ReleaseDeallocator(f));
            }
            return f;
        }

        public void release() {
            deallocate();
        }
        protected static class ReleaseDeallocator extends CvFileStorage implements Deallocator {
            ReleaseDeallocator(CvFileStorage p) { super(p); }
            @Override public void deallocate() { cvReleaseFileStorage(this); }
        }
    }

    public static abstract class AbstractCvGraphScanner extends Pointer {
        public AbstractCvGraphScanner() { }
        public AbstractCvGraphScanner(Pointer p) { super(p); }

        public static CvGraphScanner create(CvGraph graph,
                CvGraphVtx vtx/*=null*/, int mask/*=CV_GRAPH_ALL_ITEMS*/) {
            CvGraphScanner g = cvCreateGraphScanner(graph, vtx, mask);
            if (g != null) {
                g.deallocator(new ReleaseDeallocator(g));
            }
            return g;
        }
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
        return com.googlecode.javacpp.opencv_core.cvInitNArrayIterator(count, new CvArrArray(arrs),
                mask, stubs, array_iterator, flags);
    }

    public static void cvMixChannels(CvArr[] src, int src_count,
            CvArr[] dst, int dst_count, int[] from_to, int pair_count) {
        com.googlecode.javacpp.opencv_core.cvMixChannels(new CvArrArray(src), src_count,
                new CvArrArray(dst), dst_count, new IntPointer(from_to), pair_count);
    }

    public static void cvCalcCovarMatrix(CvArr[] vects, int count, CvArr cov_mat, CvArr avg, int flags) {
        com.googlecode.javacpp.opencv_core.cvCalcCovarMatrix(new CvArrArray(vects), count, cov_mat, avg, flags);
    }

    public static double cvNorm(CvArr arr1, CvArr arr2) {
        return com.googlecode.javacpp.opencv_core.cvNorm(arr1, arr2, CV_L2, null);
    }

    public static void cvFillPoly(CvArr img, CvPoint[] pts, int[] npts,
            int contours, CvScalar color, int line_type/*=8*/, int shift/*=0*/) {
        com.googlecode.javacpp.opencv_core.cvFillPoly(img, new PointerPointer(pts),
                new IntPointer(npts), contours, color, line_type, shift);
    }

    public static void cvPolyLine(CvArr img, CvPoint[] pts,
            int[] npts, int contours, int is_closed, CvScalar color,
            int thickness/*=1*/, int line_type/*=8*/, int shift/*=0*/) {
        com.googlecode.javacpp.opencv_core.cvPolyLine(img, new PointerPointer(pts),
                new IntPointer(npts), contours, is_closed, color, thickness, line_type, shift);
    }

    public static void cvDrawPolyLine(CvArr img, CvPoint[] pts,
            int[] npts, int contours, int is_closed, CvScalar color,
            int thickness/*=1*/, int line_type/*=8*/, int shift/*=0*/) {
        cvPolyLine(img, pts, npts, contours, is_closed, color, thickness, line_type, shift);
    }

    public static abstract class AbstractCvFont extends Pointer {
        public AbstractCvFont() { }
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

    public static void cvDrawContours(CvArr img, CvSeq contour, CvScalar external_color,
            CvScalar hole_color, int max_level, int thickness/*=1*/, int line_type/*=8*/) {
        com.googlecode.javacpp.opencv_core.cvDrawContours(img, contour, external_color,
                hole_color, max_level, thickness, line_type, CvPoint.ZERO);
    }


    public static abstract class AbstractMat extends AbstractArray {
        public AbstractMat() { }
        public AbstractMat(Pointer p) { super(p); }

        public void createFrom(BufferedImage image) {
            createFrom(image, 1.0);
        }
        public void createFrom(BufferedImage image, double gamma) {
            createFrom(image, gamma, false);
        }
        public void createFrom(BufferedImage image, double gamma, boolean flipChannels) {
            if (image == null) {
                release();
                return;
            }
            SampleModel sm = image.getSampleModel();
            int depth = 0, numChannels = sm.getNumBands();
            switch (image.getType()) {
                case BufferedImage.TYPE_INT_RGB:
                case BufferedImage.TYPE_INT_ARGB:
                case BufferedImage.TYPE_INT_ARGB_PRE:
                case BufferedImage.TYPE_INT_BGR:
                    depth = CV_8U;
                    numChannels = 4;
                    break;
            }
            if (depth == 0 || numChannels == 0) {
                switch (sm.getDataType()) {
                    case DataBuffer.TYPE_BYTE:   depth = CV_8U;  break;
                    case DataBuffer.TYPE_USHORT: depth = CV_16U; break;
                    case DataBuffer.TYPE_SHORT:  depth = CV_16S; break;
                    case DataBuffer.TYPE_INT:    depth = CV_32S; break;
                    case DataBuffer.TYPE_FLOAT:  depth = CV_32F; break;
                    case DataBuffer.TYPE_DOUBLE: depth = CV_64F; break;
                    default: assert false;
                }
            }
            create(image.getWidth(), image.getHeight(), CV_MAKETYPE(depth, numChannels));
            copyFrom(image, gamma, flipChannels);
        }

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

        public static final Mat EMPTY = new Mat();
    }

}
