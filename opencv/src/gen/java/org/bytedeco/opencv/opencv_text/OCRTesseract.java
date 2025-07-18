// Targeted by JavaCPP version 1.5.13-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.opencv.opencv_text;

import org.bytedeco.javacpp.annotation.Index;
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
import org.bytedeco.opencv.opencv_dnn.*;
import static org.bytedeco.opencv.global.opencv_dnn.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import org.bytedeco.opencv.opencv_videoio.*;
import static org.bytedeco.opencv.global.opencv_videoio.*;
import org.bytedeco.opencv.opencv_highgui.*;
import static org.bytedeco.opencv.global.opencv_highgui.*;
import org.bytedeco.opencv.opencv_flann.*;
import static org.bytedeco.opencv.global.opencv_flann.*;
import org.bytedeco.opencv.opencv_features2d.*;
import static org.bytedeco.opencv.global.opencv_features2d.*;
import org.bytedeco.opencv.opencv_ml.*;
import static org.bytedeco.opencv.global.opencv_ml.*;

import static org.bytedeco.opencv.global.opencv_text.*;


/** \brief OCRTesseract class provides an interface with the tesseract-ocr API (v3.02.02) in C++.
<p>
Notice that it is compiled only when tesseract-ocr is correctly installed.
<p>
\note
   -   (C++) An example of OCRTesseract recognition combined with scene text detection can be found
        at the end_to_end_recognition demo:
        <https://github.com/opencv/opencv_contrib/blob/master/modules/text/samples/end_to_end_recognition.cpp>
    -   (C++) Another example of OCRTesseract recognition combined with scene text detection can be
        found at the webcam_demo:
        <https://github.com/opencv/opencv_contrib/blob/master/modules/text/samples/webcam_demo.cpp>
 */
@Namespace("cv::text") @Properties(inherit = org.bytedeco.opencv.presets.opencv_text.class)
public class OCRTesseract extends BaseOCR {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public OCRTesseract(Pointer p) { super(p); }

    /** \brief Recognize text using the tesseract-ocr API.
    <p>
    Takes image on input and returns recognized text in the output_text parameter. Optionally
    provides also the Rects for individual text elements found (e.g. words), and the list of those
    text elements with their confidence values.
    <p>
    @param image Input image CV_8UC1 or CV_8UC3
    @param output_text Output text of the tesseract-ocr.
    @param component_rects If provided the method will output a list of Rects for the individual
    text elements found (e.g. words or text lines).
    @param component_texts If provided the method will output a list of text strings for the
    recognition of individual text elements found (e.g. words or text lines).
    @param component_confidences If provided the method will output a list of confidence values
    for the recognition of individual text elements found (e.g. words or text lines).
    @param component_level OCR_LEVEL_WORD (by default), or OCR_LEVEL_TEXTLINE.
     */
    public native @Override void run(@ByRef Mat image, @StdString @ByRef BytePointer output_text, RectVector component_rects/*=NULL*/,
                         StringVector component_texts/*=NULL*/, FloatVector component_confidences/*=NULL*/,
                         int component_level/*=0*/);
    public native void run(@ByRef Mat image, @StdString @ByRef BytePointer output_text);

    public native @Override void run(@ByRef Mat image, @ByRef Mat mask, @StdString @ByRef BytePointer output_text, RectVector component_rects/*=NULL*/,
                         StringVector component_texts/*=NULL*/, FloatVector component_confidences/*=NULL*/,
                         int component_level/*=0*/);
    public native void run(@ByRef Mat image, @ByRef Mat mask, @StdString @ByRef BytePointer output_text);

    // aliases for scripting
    public native @Str BytePointer run(@ByVal Mat image, int min_confidence, int component_level/*=0*/);
    public native @Str BytePointer run(@ByVal Mat image, int min_confidence);
    public native @Str String run(@ByVal UMat image, int min_confidence, int component_level/*=0*/);
    public native @Str String run(@ByVal UMat image, int min_confidence);
    public native @Str BytePointer run(@ByVal GpuMat image, int min_confidence, int component_level/*=0*/);
    public native @Str BytePointer run(@ByVal GpuMat image, int min_confidence);

    public native @Str BytePointer run(@ByVal Mat image, @ByVal Mat mask, int min_confidence, int component_level/*=0*/);
    public native @Str BytePointer run(@ByVal Mat image, @ByVal Mat mask, int min_confidence);
    public native @Str String run(@ByVal UMat image, @ByVal UMat mask, int min_confidence, int component_level/*=0*/);
    public native @Str String run(@ByVal UMat image, @ByVal UMat mask, int min_confidence);
    public native @Str BytePointer run(@ByVal GpuMat image, @ByVal GpuMat mask, int min_confidence, int component_level/*=0*/);
    public native @Str BytePointer run(@ByVal GpuMat image, @ByVal GpuMat mask, int min_confidence);

    public native void setWhiteList(@Str BytePointer char_whitelist);
    public native void setWhiteList(@Str String char_whitelist);


    /** \brief Creates an instance of the OCRTesseract class. Initializes Tesseract.
    <p>
    @param datapath the name of the parent directory of tessdata ended with "/", or NULL to use the
    system's default directory.
    @param language an ISO 639-3 code or NULL will default to "eng".
    @param char_whitelist specifies the list of characters used for recognition. NULL defaults to ""
    (All characters will be used for recognition).
    @param oem tesseract-ocr offers different OCR Engine Modes (OEM), by default
    tesseract::OEM_DEFAULT is used. See the tesseract-ocr API documentation for other possible
    values.
    @param psmode tesseract-ocr offers different Page Segmentation Modes (PSM) tesseract::PSM_AUTO
    (fully automatic layout analysis) is used. See the tesseract-ocr API documentation for other
    possible values.
    <p>
    \note The char_whitelist default is changed after OpenCV 4.7.0/3.19.0 from "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" to "".
     */
    public static native @Ptr OCRTesseract create(@Cast("const char*") BytePointer datapath/*=NULL*/, @Cast("const char*") BytePointer language/*=NULL*/,
                                        @Cast("const char*") BytePointer char_whitelist/*=NULL*/, int oem/*=cv::text::OEM_DEFAULT*/, int psmode/*=cv::text::PSM_AUTO*/);
    public static native @Ptr OCRTesseract create();
    public static native @Ptr OCRTesseract create(String datapath/*=NULL*/, String language/*=NULL*/,
                                        String char_whitelist/*=NULL*/, int oem/*=cv::text::OEM_DEFAULT*/, int psmode/*=cv::text::PSM_AUTO*/);
}
