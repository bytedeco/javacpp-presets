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


/** \addtogroup objdetect_barcode
 *  \{ */

@Namespace("cv::barcode") @Properties(inherit = org.bytedeco.opencv.presets.opencv_objdetect.class)
public class BarcodeDetector extends GraphicalCodeDetector {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public BarcodeDetector(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public BarcodeDetector(long size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(long size);
    @Override public BarcodeDetector position(long position) {
        return (BarcodeDetector)super.position(position);
    }
    @Override public BarcodeDetector getPointer(long i) {
        return new BarcodeDetector((Pointer)this).offsetAddress(i);
    }

    /** \brief Initialize the BarcodeDetector.
    */
    public BarcodeDetector() { super((Pointer)null); allocate(); }
    private native void allocate();
    /** \brief Initialize the BarcodeDetector.
     *
     * Parameters allow to load _optional_ Super Resolution DNN model for better quality.
     * @param prototxt_path prototxt file path for the super resolution model
     * @param model_path model file path for the super resolution model
     */
    public BarcodeDetector(@StdString BytePointer prototxt_path, @StdString BytePointer model_path) { super((Pointer)null); allocate(prototxt_path, model_path); }
    private native void allocate(@StdString BytePointer prototxt_path, @StdString BytePointer model_path);
    public BarcodeDetector(@StdString String prototxt_path, @StdString String model_path) { super((Pointer)null); allocate(prototxt_path, model_path); }
    private native void allocate(@StdString String prototxt_path, @StdString String model_path);

    /** \brief Decodes barcode in image once it's found by the detect() method.
     *
     * @param img grayscale or color (BGR) image containing bar code.
     * @param points vector of rotated rectangle vertices found by detect() method (or some other algorithm).
     * For N detected barcodes, the dimensions of this array should be [N][4].
     * Order of four points in vector<Point2f> is bottomLeft, topLeft, topRight, bottomRight.
     * @param decoded_info UTF8-encoded output vector of string or empty vector of string if the codes cannot be decoded.
     * @param decoded_type vector strings, specifies the type of these barcodes
     * @return true if at least one valid barcode have been found
     */
    public native @Cast("bool") boolean decodeWithType(@ByVal Mat img,
                                 @ByVal Mat points,
                                 @ByRef StringVector decoded_info,
                                 @ByRef StringVector decoded_type);
    public native @Cast("bool") boolean decodeWithType(@ByVal UMat img,
                                 @ByVal UMat points,
                                 @ByRef StringVector decoded_info,
                                 @ByRef StringVector decoded_type);
    public native @Cast("bool") boolean decodeWithType(@ByVal GpuMat img,
                                 @ByVal GpuMat points,
                                 @ByRef StringVector decoded_info,
                                 @ByRef StringVector decoded_type);

    /** \brief Both detects and decodes barcode
     <p>
     * @param img grayscale or color (BGR) image containing barcode.
     * @param decoded_info UTF8-encoded output vector of string(s) or empty vector of string if the codes cannot be decoded.
     * @param decoded_type vector of strings, specifies the type of these barcodes
     * @param points optional output vector of vertices of the found  barcode rectangle. Will be empty if not found.
     * @return true if at least one valid barcode have been found
     */
    public native @Cast("bool") boolean detectAndDecodeWithType(@ByVal Mat img,
                                          @ByRef StringVector decoded_info,
                                          @ByRef StringVector decoded_type,
                                          @ByVal(nullValue = "cv::OutputArray(cv::noArray())") Mat points);
    public native @Cast("bool") boolean detectAndDecodeWithType(@ByVal Mat img,
                                          @ByRef StringVector decoded_info,
                                          @ByRef StringVector decoded_type);
    public native @Cast("bool") boolean detectAndDecodeWithType(@ByVal UMat img,
                                          @ByRef StringVector decoded_info,
                                          @ByRef StringVector decoded_type,
                                          @ByVal(nullValue = "cv::OutputArray(cv::noArray())") UMat points);
    public native @Cast("bool") boolean detectAndDecodeWithType(@ByVal UMat img,
                                          @ByRef StringVector decoded_info,
                                          @ByRef StringVector decoded_type);
    public native @Cast("bool") boolean detectAndDecodeWithType(@ByVal GpuMat img,
                                          @ByRef StringVector decoded_info,
                                          @ByRef StringVector decoded_type,
                                          @ByVal(nullValue = "cv::OutputArray(cv::noArray())") GpuMat points);
    public native @Cast("bool") boolean detectAndDecodeWithType(@ByVal GpuMat img,
                                          @ByRef StringVector decoded_info,
                                          @ByRef StringVector decoded_type);

    /** \brief Get detector downsampling threshold.
     *
     * @return detector downsampling threshold
     */
    public native double getDownsamplingThreshold();

    /** \brief Set detector downsampling threshold.
     *
     * By default, the detect method resizes the input image to this limit if the smallest image size is is greater than the threshold.
     * Increasing this value can improve detection accuracy and the number of results at the expense of performance.
     * Correlates with detector scales. Setting this to a large value will disable downsampling.
     * @param thresh downsampling limit to apply (default 512)
     * @see setDetectorScales
     */
    public native @ByRef BarcodeDetector setDownsamplingThreshold(double thresh);

    /** \brief Returns detector box filter sizes.
     *
     * @param sizes output parameter for returning the sizes.
     */
    public native void getDetectorScales(@StdVector FloatPointer sizes);
    public native void getDetectorScales(@StdVector FloatBuffer sizes);
    public native void getDetectorScales(@StdVector float[] sizes);

    /** \brief Set detector box filter sizes.
     *
     * Adjusts the value and the number of box filters used in the detect step.
     * The filter sizes directly correlate with the expected line widths for a barcode. Corresponds to expected barcode distance.
     * If the downsampling limit is increased, filter sizes need to be adjusted in an inversely proportional way.
     * @param sizes box filter sizes, relative to minimum dimension of the image (default [0.01, 0.03, 0.06, 0.08])
     */
    public native @ByRef BarcodeDetector setDetectorScales(@StdVector FloatPointer sizes);
    public native @ByRef BarcodeDetector setDetectorScales(@StdVector FloatBuffer sizes);
    public native @ByRef BarcodeDetector setDetectorScales(@StdVector float[] sizes);

    /** \brief Get detector gradient magnitude threshold.
     *
     * @return detector gradient magnitude threshold.
     */
    public native double getGradientThreshold();

    /** \brief Set detector gradient magnitude threshold.
     *
     * Sets the coherence threshold for detected bounding boxes.
     * Increasing this value will generate a closer fitted bounding box width and can reduce false-positives.
     * Values between 16 and 1024 generally work, while too high of a value will remove valid detections.
     * @param thresh gradient magnitude threshold (default 64).
     */
    public native @ByRef BarcodeDetector setGradientThreshold(double thresh);
}
