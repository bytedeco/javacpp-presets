// Targeted by JavaCPP version 1.5.11: DO NOT EDIT THIS FILE

package org.bytedeco.depthai;

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

import static org.bytedeco.depthai.global.depthai.*;


/**
 * ImageManipConfig message. Specifies image manipulation options like:
 *
 *  - Crop
 *
 *  - Resize
 *
 *  - Warp
 *
 *  - ...
 */
@Namespace("dai") @NoOffset @Properties(inherit = org.bytedeco.depthai.presets.depthai.class)
public class ImageManipConfig extends Buffer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ImageManipConfig(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public ImageManipConfig(long size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(long size);
    @Override public ImageManipConfig position(long position) {
        return (ImageManipConfig)super.position(position);
    }
    @Override public ImageManipConfig getPointer(long i) {
        return new ImageManipConfig((Pointer)this).offsetAddress(i);
    }

    // Alias

    /** Construct ImageManipConfig message */
    public ImageManipConfig() { super((Pointer)null); allocate(); }
    private native void allocate();
    public ImageManipConfig(@SharedPtr RawImageManipConfig ptr) { super((Pointer)null); allocate(ptr); }
    private native void allocate(@SharedPtr RawImageManipConfig ptr);

    // Functions to set properties
    /**
     * Specifies crop with rectangle with normalized values (0..1)
     * @param xmin Top left X coordinate of rectangle
     * @param ymin Top left Y coordinate of rectangle
     * @param xmax Bottom right X coordinate of rectangle
     * @param ymax Bottom right Y coordinate of rectangle
     */
    public native @ByRef ImageManipConfig setCropRect(float xmin, float ymin, float xmax, float ymax);

    /**
     * Specifies crop with rectangle with normalized values (0..1)
     * @param coordinates Coordinate of rectangle
     */
    public native @ByRef ImageManipConfig setCropRect(@ByVal @Cast("std::tuple<float,float,float,float>*") Pointer coordinates);

    /**
     * Specifies crop with rotated rectangle. Optionally as non normalized coordinates
     * @param rr Rotated rectangle which specifies crop
     * @param normalizedCoords If true coordinates are in normalized range (0..1) otherwise absolute
     */
    public native @ByRef ImageManipConfig setCropRotatedRect(@ByVal RotatedRect rr, @Cast("bool") boolean normalizedCoords/*=true*/);
    public native @ByRef ImageManipConfig setCropRotatedRect(@ByVal RotatedRect rr);

    /**
     * Specifies a centered crop.
     * @param ratio Ratio between input image and crop region (0..1)
     * @param whRatio Crop region aspect ratio - 1 equals to square, 1.7 equals to 16:9, ...
     */
    public native @ByRef ImageManipConfig setCenterCrop(float ratio, float whRatio/*=1.0f*/);
    public native @ByRef ImageManipConfig setCenterCrop(float ratio);

    /**
     * Specifies warp by supplying 4 points in either absolute or normalized coordinates
     * @param pt 4 points specifying warp
     * @param normalizedCoords If true pt is interpreted as normalized, absolute otherwise
     */
    public native @ByRef ImageManipConfig setWarpTransformFourPoints(@ByVal Point2fVector pt, @Cast("bool") boolean normalizedCoords);

    /**
     * Specifies warp with a 3x3 matrix
     * @param mat 3x3 matrix
     */
    public native @ByRef ImageManipConfig setWarpTransformMatrix3x3(@StdVector FloatPointer mat);
    public native @ByRef ImageManipConfig setWarpTransformMatrix3x3(@StdVector FloatBuffer mat);
    public native @ByRef ImageManipConfig setWarpTransformMatrix3x3(@StdVector float[] mat);

    /**
     * Specifies that warp replicates border pixels
     */
    public native @ByRef ImageManipConfig setWarpBorderReplicatePixels();

    /**
     * Specifies fill color for border pixels. Example:
     *
     *  - setWarpBorderFillColor(255,255,255) -> white
     *
     *  - setWarpBorderFillColor(0,0,255) -> blue
     *
     * @param red Red component
     * @param green Green component
     * @param blue Blue component
     */
    public native @ByRef ImageManipConfig setWarpBorderFillColor(int red, int green, int blue);

    /**
     * Specifies clockwise rotation in degrees
     * @param deg Rotation in degrees
     */
    public native @ByRef ImageManipConfig setRotationDegrees(float deg);

    /**
     * Specifies clockwise rotation in radians
     * @param rad Rotation in radians
     */
    public native @ByRef ImageManipConfig setRotationRadians(float rad);

    /**
     * Specifies output image size. After crop stage the image will be stretched to fit.
     * @param w Width in pixels
     * @param h Height in pixels
     */
    public native @ByRef ImageManipConfig setResize(int w, int h);

    /**
     * Specifies output image size. After crop stage the image will be stretched to fit.
     * @param size Size in pixels
     */
    public native @ByRef ImageManipConfig setResize(@ByVal @Cast("std::tuple<int,int>*") Pointer size);

    /**
     * Specifies output image size. After crop stage the image will be resized by preserving aspect ration.
     * Optionally background can be specified.
     *
     * @param w Width in pixels
     * @param h Height in pixels
     * @param bgRed Red component
     * @param bgGreen Green component
     * @param bgBlue Blue component
     */
    public native @ByRef ImageManipConfig setResizeThumbnail(int w, int h, int bgRed/*=0*/, int bgGreen/*=0*/, int bgBlue/*=0*/);
    public native @ByRef ImageManipConfig setResizeThumbnail(int w, int h);

    /**
     * Specifies output image size. After crop stage the image will be resized by preserving aspect ration.
     * Optionally background can be specified.
     *
     * @param size Size in pixels
     * @param bgRed Red component
     * @param bgGreen Green component
     * @param bgBlue Blue component
     */
    public native @ByRef ImageManipConfig setResizeThumbnail(@ByVal @Cast("std::tuple<int,int>*") Pointer size, int bgRed/*=0*/, int bgGreen/*=0*/, int bgBlue/*=0*/);
    public native @ByRef ImageManipConfig setResizeThumbnail(@ByVal @Cast("std::tuple<int,int>*") Pointer size);

    /**
     * Specify output frame type.
     * @param name Frame type
     */
    public native @ByRef ImageManipConfig setFrameType(RawImgFrame.Type name);
    public native @ByRef ImageManipConfig setFrameType(@Cast("dai::RawImgFrame::Type") int name);

    /**
     * Specify gray to color conversion map
     * @param colormap map from Colormap enum or Colormap::NONE to disable
     */
    public native @ByRef ImageManipConfig setColormap(Colormap colormap, int min, int max);
    public native @ByRef ImageManipConfig setColormap(@Cast("dai::Colormap") int colormap, int min, int max);
    public native @ByRef ImageManipConfig setColormap(Colormap colormap, float maxf);
    public native @ByRef ImageManipConfig setColormap(@Cast("dai::Colormap") int colormap, float maxf);
    public native @ByRef ImageManipConfig setColormap(Colormap colormap, int max/*=255*/);
    public native @ByRef ImageManipConfig setColormap(Colormap colormap);
    public native @ByRef ImageManipConfig setColormap(@Cast("dai::Colormap") int colormap, int max/*=255*/);
    public native @ByRef ImageManipConfig setColormap(@Cast("dai::Colormap") int colormap);

    /**
     * Specify horizontal flip
     * @param flip True to enable flip, false otherwise
     */
    public native @ByRef ImageManipConfig setHorizontalFlip(@Cast("bool") boolean flip);

    /**
     * Specify vertical flip
     * @param flip True to enable vertical flip, false otherwise
     */
    public native void setVerticalFlip(@Cast("bool") boolean flip);

    /**
     * Instruct ImageManip to not remove current image from its queue and use the same for next message.
     * @param reuse True to enable reuse, false otherwise
     */
    public native @ByRef ImageManipConfig setReusePreviousImage(@Cast("bool") boolean reuse);

    /**
     * Instructs ImageManip to skip current image and wait for next in queue.
     * @param skip True to skip current image, false otherwise
     */
    public native @ByRef ImageManipConfig setSkipCurrentImage(@Cast("bool") boolean skip);

    /**
     * Specifies to whether to keep aspect ratio or not
     */
    public native @ByRef ImageManipConfig setKeepAspectRatio(@Cast("bool") boolean keep);

    /**
     * Specify which interpolation method to use
     * @param interpolation type of interpolation
     */
    public native @ByRef ImageManipConfig setInterpolation(Interpolation interpolation);
    public native @ByRef ImageManipConfig setInterpolation(@Cast("dai::Interpolation") int interpolation);

    // Functions to retrieve properties
    /**
     * @return Top left X coordinate of crop region
     */
    public native float getCropXMin();

    /**
     * @return Top left Y coordinate of crop region
     */
    public native float getCropYMin();

    /**
     * @return Bottom right X coordinate of crop region
     */
    public native float getCropXMax();

    /**
     * @return Bottom right Y coordinate of crop region
     */
    public native float getCropYMax();

    /**
     * @return Output image width
     */
    public native int getResizeWidth();

    /**
     * @return Output image height
     */
    public native int getResizeHeight();

    /**
     * @return Crop configuration
     */
    public native @ByVal @Cast("dai::ImageManipConfig::CropConfig*") RawImageManipConfig.CropConfig getCropConfig();

    /**
     * @return Resize configuration
     */
    public native @ByVal @Cast("dai::ImageManipConfig::ResizeConfig*") RawImageManipConfig.ResizeConfig getResizeConfig();

    /**
     * @return Format configuration
     */
    public native @ByVal @Cast("dai::ImageManipConfig::FormatConfig*") RawImageManipConfig.FormatConfig getFormatConfig();

    /**
     * @return True if resize thumbnail mode is set, false otherwise
     */
    public native @Cast("bool") boolean isResizeThumbnail();

    /**
     * @return specified colormap
     */
    public native Colormap getColormap();

    /**
     * Set explicit configuration.
     * @param config Explicit configuration
     */
    public native @ByRef ImageManipConfig set(@ByVal RawImageManipConfig config);

    /**
     * Retrieve configuration data for ImageManip.
     * @return config for ImageManip
     */
    public native @ByVal RawImageManipConfig get();

    /** Retrieve which interpolation method to use */
    public native Interpolation getInterpolation();
}
