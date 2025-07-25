// Targeted by JavaCPP version 1.5.13-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.opencv.opencv_objdetect;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import static org.bytedeco.openblas.global.openblas_nolapack.*;
import static org.bytedeco.openblas.global.openblas.*;
import org.bytedeco.opencv.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_core.*;
import org.bytedeco.opencv.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import org.bytedeco.opencv.opencv_videoio.*;
import static org.bytedeco.opencv.global.opencv_videoio.*;
import org.bytedeco.opencv.opencv_highgui.*;
import static org.bytedeco.opencv.global.opencv_highgui.*;
import org.bytedeco.opencv.opencv_flann.*;
import static org.bytedeco.opencv.global.opencv_flann.*;
import org.bytedeco.opencv.opencv_features2d.*;
import static org.bytedeco.opencv.global.opencv_features2d.*;
import org.bytedeco.opencv.opencv_calib3d.*;
import static org.bytedeco.opencv.global.opencv_calib3d.*;
import org.bytedeco.opencv.opencv_dnn.*;
import static org.bytedeco.opencv.global.opencv_dnn.*;

import static org.bytedeco.opencv.global.opencv_objdetect.*;

/** \}
 <p>
 *  \addtogroup objdetect_qrcode
 *  \{ */

@Namespace("cv") @Properties(inherit = org.bytedeco.opencv.presets.opencv_objdetect.class)
public class QRCodeEncoder extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public QRCodeEncoder(Pointer p) { super(p); }


    /** enum cv::QRCodeEncoder::EncodeMode */
    public static final int
        MODE_AUTO              = -1,
        MODE_NUMERIC           = 1, // 0b0001
        MODE_ALPHANUMERIC      = 2, // 0b0010
        MODE_BYTE              = 4, // 0b0100
        MODE_ECI               = 7, // 0b0111
        MODE_KANJI             = 8, // 0b1000
        MODE_STRUCTURED_APPEND = 3;  // 0b0011

    /** enum cv::QRCodeEncoder::CorrectionLevel */
    public static final int
        CORRECT_LEVEL_L = 0,
        CORRECT_LEVEL_M = 1,
        CORRECT_LEVEL_Q = 2,
        CORRECT_LEVEL_H = 3;

    /** enum cv::QRCodeEncoder::ECIEncodings */
    public static final int
        ECI_SHIFT_JIS = 20,
        ECI_UTF8 = 26;

    /** \brief QR code encoder parameters. */
    @NoOffset public static class Params extends Pointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public Params(Pointer p) { super(p); }
        /** Native array allocator. Access with {@link Pointer#position(long)}. */
        public Params(long size) { super((Pointer)null); allocateArray(size); }
        private native void allocateArray(long size);
        @Override public Params position(long position) {
            return (Params)super.position(position);
        }
        @Override public Params getPointer(long i) {
            return new Params((Pointer)this).offsetAddress(i);
        }
    
        public Params() { super((Pointer)null); allocate(); }
        private native void allocate();

        /** The optional version of QR code (by default - maximum possible depending on the length of the string). */
        public native int version(); public native Params version(int setter);

        /** The optional level of error correction (by default - the lowest). */
        public native @Cast("cv::QRCodeEncoder::CorrectionLevel") int correction_level(); public native Params correction_level(int setter);

        /** The optional encoding mode - Numeric, Alphanumeric, Byte, Kanji, ECI or Structured Append. */
        public native @Cast("cv::QRCodeEncoder::EncodeMode") int mode(); public native Params mode(int setter);

        /** The optional number of QR codes to generate in Structured Append mode. */
        public native int structure_number(); public native Params structure_number(int setter);
    }

    /** \brief Constructor
    @param parameters QR code encoder parameters QRCodeEncoder::Params
    */
    public static native @Ptr QRCodeEncoder create(@Const @ByRef(nullValue = "cv::QRCodeEncoder::Params()") Params parameters);
    public static native @Ptr QRCodeEncoder create();

    /** \brief Generates QR code from input string.
     @param encoded_info Input string to encode.
     @param qrcode Generated QR code.
    */
    public native void encode(@Str BytePointer encoded_info, @ByVal Mat qrcode);
    public native void encode(@Str String encoded_info, @ByVal Mat qrcode);
    public native void encode(@Str String encoded_info, @ByVal UMat qrcode);
    public native void encode(@Str BytePointer encoded_info, @ByVal UMat qrcode);
    public native void encode(@Str BytePointer encoded_info, @ByVal GpuMat qrcode);
    public native void encode(@Str String encoded_info, @ByVal GpuMat qrcode);

    /** \brief Generates QR code from input string in Structured Append mode. The encoded message is splitting over a number of QR codes.
     @param encoded_info Input string to encode.
     @param qrcodes Vector of generated QR codes.
    */
    public native void encodeStructuredAppend(@Str BytePointer encoded_info, @ByVal MatVector qrcodes);
    public native void encodeStructuredAppend(@Str String encoded_info, @ByVal UMatVector qrcodes);
    public native void encodeStructuredAppend(@Str BytePointer encoded_info, @ByVal GpuMatVector qrcodes);
    public native void encodeStructuredAppend(@Str String encoded_info, @ByVal MatVector qrcodes);
    public native void encodeStructuredAppend(@Str BytePointer encoded_info, @ByVal UMatVector qrcodes);
    public native void encodeStructuredAppend(@Str String encoded_info, @ByVal GpuMatVector qrcodes);

}
