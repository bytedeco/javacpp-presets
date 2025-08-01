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


/** \brief Planar board with grid arrangement of markers
 *
 * More common type of board. All markers are placed in the same plane in a grid arrangement.
 * The board image can be drawn using generateImage() method.
 */
@Namespace("cv::aruco") @Properties(inherit = org.bytedeco.opencv.presets.opencv_objdetect.class)
public class GridBoard extends Board {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public GridBoard(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public GridBoard(long size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(long size);
    @Override public GridBoard position(long position) {
        return (GridBoard)super.position(position);
    }
    @Override public GridBoard getPointer(long i) {
        return new GridBoard((Pointer)this).offsetAddress(i);
    }

    /**
     * \brief GridBoard constructor
     *
     * @param size number of markers in x and y directions
     * @param markerLength marker side length (normally in meters)
     * @param markerSeparation separation between two markers (same unit as markerLength)
     * @param dictionary dictionary of markers indicating the type of markers
     * @param ids set of marker ids in dictionary to use on board.
     */
    public GridBoard(@Const @ByRef Size size, float markerLength, float markerSeparation,
                          @Const @ByRef Dictionary dictionary, @ByVal(nullValue = "cv::InputArray(cv::noArray())") Mat ids) { super((Pointer)null); allocate(size, markerLength, markerSeparation, dictionary, ids); }
    private native void allocate(@Const @ByRef Size size, float markerLength, float markerSeparation,
                          @Const @ByRef Dictionary dictionary, @ByVal(nullValue = "cv::InputArray(cv::noArray())") Mat ids);
    public GridBoard(@Const @ByRef Size size, float markerLength, float markerSeparation,
                          @Const @ByRef Dictionary dictionary) { super((Pointer)null); allocate(size, markerLength, markerSeparation, dictionary); }
    private native void allocate(@Const @ByRef Size size, float markerLength, float markerSeparation,
                          @Const @ByRef Dictionary dictionary);
    public GridBoard(@Const @ByRef Size size, float markerLength, float markerSeparation,
                          @Const @ByRef Dictionary dictionary, @ByVal(nullValue = "cv::InputArray(cv::noArray())") UMat ids) { super((Pointer)null); allocate(size, markerLength, markerSeparation, dictionary, ids); }
    private native void allocate(@Const @ByRef Size size, float markerLength, float markerSeparation,
                          @Const @ByRef Dictionary dictionary, @ByVal(nullValue = "cv::InputArray(cv::noArray())") UMat ids);
    public GridBoard(@Const @ByRef Size size, float markerLength, float markerSeparation,
                          @Const @ByRef Dictionary dictionary, @ByVal(nullValue = "cv::InputArray(cv::noArray())") GpuMat ids) { super((Pointer)null); allocate(size, markerLength, markerSeparation, dictionary, ids); }
    private native void allocate(@Const @ByRef Size size, float markerLength, float markerSeparation,
                          @Const @ByRef Dictionary dictionary, @ByVal(nullValue = "cv::InputArray(cv::noArray())") GpuMat ids);

    public native @ByVal Size getGridSize();
    public native float getMarkerLength();
    public native float getMarkerSeparation();

    @Deprecated public GridBoard() { super((Pointer)null); allocate(); }
    @Deprecated private native void allocate();
}
