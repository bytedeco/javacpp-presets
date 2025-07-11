// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.tensorrt.nvinfer;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import org.bytedeco.cuda.cudart.*;
import static org.bytedeco.cuda.global.cudart.*;
import org.bytedeco.cuda.cublas.*;
import static org.bytedeco.cuda.global.cublas.*;
import org.bytedeco.cuda.cudnn.*;
import static org.bytedeco.cuda.global.cudnn.*;
import org.bytedeco.cuda.nvrtc.*;
import static org.bytedeco.cuda.global.nvrtc.*;

import static org.bytedeco.tensorrt.global.nvinfer.*;
 // namespace impl

/** \class IResizeLayer
 * 
 *  \brief A resize layer in a network definition.
 * 
 *  Resize layer can be used for resizing a N-D tensor.
 * 
 *  Resize layer currently supports the following configurations:
 *      -   InterpolationMode::kNEAREST - resizes last {@code m} dimensions of N-D, where 0 < m <= min(8, N) and N > 0
 *      -   InterpolationMode::kLINEAR - resizes last {@code m} dimensions of N-D, where 0 < m <= min(3, N) and N > 0
 * 
 *  Default resize mode is InterpolationMode::kNEAREST.
 * 
 *  The coordinates in the output tensor are mapped to coordinates in the input tensor using a function set by calling
 *  setCoordinateTransformation(). The default for all InterpolationMode settings (nearest, linear, bilinear, etc.) is
 *  ResizeCoordinateTransformation::kASYMMETRIC.
 * 
 *  The resize layer provides two ways to resize tensor dimensions.
 *      -   Set output dimensions directly. It can be done for static as well as dynamic resize layer.
 *          Static resize layer requires output dimensions to be known at build-time.
 *          Dynamic resize layer requires output dimensions to be set as one of the input tensors.
 *      -   Set scales for resize. Each output dimension is calculated as floor(input dimension * scale).
 *          Only static resize layer allows setting scales where the scales are known at build-time.
 * 
 *  If executing this layer on DLA, the following combinations of parameters are supported:
 * 
 *  - In kNEAREST mode:
 *      * (ResizeCoordinateTransformation::kASYMMETRIC, ResizeSelector::kFORMULA, ResizeRoundMode::kFLOOR)
 *      * (ResizeCoordinateTransformation::kHALF_PIXEL, ResizeSelector::kFORMULA, ResizeRoundMode::kHALF_DOWN)
 *      * (ResizeCoordinateTransformation::kHALF_PIXEL, ResizeSelector::kFORMULA, ResizeRoundMode::kHALF_UP)
 * 
 *  - In kLINEAR mode:
 *      * (ResizeCoordinateTransformation::kHALF_PIXEL, ResizeSelector::kFORMULA)
 *      * (ResizeCoordinateTransformation::kHALF_PIXEL, ResizeSelector::kUPPER)
 * 
 *  \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
 *  */
@Namespace("nvinfer1") @NoOffset @Properties(inherit = org.bytedeco.tensorrt.presets.nvinfer.class)
public class IResizeLayer extends ILayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public IResizeLayer(Pointer p) { super(p); }

    /**
     *  \brief Set the output dimensions.
     * 
     *  @param dimensions The output dimensions. Number of output dimensions must be the same as the number of input
     *  dimensions.
     * 
     *  If executing this layer on DLA, setOutputDimensions() is not supported.
     * 
     *  If there is a second input, i.e. resize layer is dynamic,
     *  calling setOutputDimensions() is an error and does not update the
     *  dimensions.
     * 
     *  Output dimensions can be specified directly, or via scale factors relative to input dimensions.
     *  Scales for resize can be provided using setScales().
     * 
     *  @see setScales
     *  @see getOutputDimensions
     *  */
    
    
    //!
    //!
    //!
    public native @NoException(true) void setOutputDimensions(@Cast("const nvinfer1::Dims*") @ByRef Dims64 dimensions);

    /**
     *  \brief Get the output dimensions.
     * 
     *  @return The output dimensions.
     *  */
    
    
    //!
    //!
    //!
    //!
    //!
    //!
    //!
    //!
    public native @ByVal @Cast("nvinfer1::Dims*") @NoException(true) Dims64 getOutputDimensions();

    /**
     *  \brief Set the resize scales.
     * 
     *  @param scales An array of resize scales.
     *  @param nbScales Number of scales. Number of scales must be equal to the number of input dimensions.
     * 
     *  If executing this layer on DLA, there are three restrictions:
     *  1) nbScales has to be exactly 4.
     *  2) the first two elements in scales need to be exactly 1 (for unchanged batch and channel dimensions).
     *  3) The last two elements in scales, representing the scale values along height and width dimensions,
     *  respectively, need to be integer values in the range of [1, 32] for kNEAREST mode and [1, 4] for kLINEAR.
     *  Example of DLA-supported scales: {1, 1, 2, 2}.
     * 
     *  If there is a second input, i.e. resize layer is dynamic,
     *  calling setScales() is an error and does not update the scales.
     * 
     *  Output dimensions are calculated as follows:
     *  outputDims[i] = floor(inputDims[i] * scales[i])
     * 
     *  Output dimensions can be specified directly, or via scale factors relative to input dimensions.
     *  Output dimensions can be provided directly using setOutputDimensions().
     * 
     *  @see setOutputDimensions
     *  @see getScales
     *  */
    
    
    //!
    //!
    //!
    //!
    //!
    //!
    public native @NoException(true) void setScales(@Const FloatPointer scales, int nbScales);
    public native @NoException(true) void setScales(@Const FloatBuffer scales, int nbScales);
    public native @NoException(true) void setScales(@Const float[] scales, int nbScales);

    /**
     *  \brief Copies resize scales to scales[0, ..., nbScales-1], where nbScales is the number of scales that were set.
     * 
     *  @param size The number of scales to get. If size != nbScales, no scales will be copied.
     * 
     *  @param scales Pointer to where to copy the scales. Scales will be copied only if
     *                size == nbScales and scales != nullptr.
     * 
     *  In case the size is not known consider using size = 0 and scales = nullptr. This method will return
     *  the number of resize scales.
     * 
     *  @return The number of resize scales i.e. nbScales if scales were set.
     *          Return -1 in case no scales were set or resize layer is used in dynamic mode.
     *  */
    
    
    //!
    //!
    //!
    //!
    public native @NoException(true) int getScales(int size, FloatPointer scales);
    public native @NoException(true) int getScales(int size, FloatBuffer scales);
    public native @NoException(true) int getScales(int size, float[] scales);

    /**
     *  \brief Set resize mode for an input tensor.
     * 
     *  Supported resize modes are Nearest Neighbor and Linear.
     * 
     *  @see InterpolationMode
     *  */
    
    
    //!
    //!
    //!
    public native @NoException(true) void setResizeMode(InterpolationMode interpolationMode);
    public native @NoException(true) void setResizeMode(@Cast("nvinfer1::InterpolationMode") int interpolationMode);

    /**
     *  \brief Get resize mode for an input tensor.
     * 
     *  @return The resize mode.
     *  */
    
    
    //!
    //!
    //!
    //!
    //!
    //!
    //!
    public native @NoException(true) InterpolationMode getResizeMode();

    /**
     *  \brief Append or replace an input of this layer with a specific tensor
     * 
     *  @param index the index of the input to modify.
     *  @param tensor the new input tensor.
     * 
     *  Sets the input tensor for the given index. The index must be 0 for a static resize layer.
     *  A static resize layer is converted to a dynamic resize layer by calling setInput with an index 1.
     *  A dynamic resize layer cannot be converted back to a static resize layer.
     * 
     *  For a dynamic resize layer, the values 0 and 1 are valid.
     *  The indices in the dynamic case are as follows:
     * 
     *  - 0: Execution tensor to be resized.
     *  - 1: The output dimensions, as a 1D tensor of type Int32 or Int64.
     * 
     *  If this function is called with the value 1, then the function getNbInputs() changes
     *  from returning 1 to 2.
     *  */
    
    
    //!
    //!
    //!
    //!
    //!

    /**
     *  \brief Set coordinate transformation function.
     * 
     *  The function maps a coordinate in the output tensor to a coordinate in the input tensor.
     * 
     *  Default function is ResizeCoordinateTransformation::kASYMMETRIC.
     * 
     *  @see ResizeCoordinateTransformation
     *  */
    
    
    //!
    //!
    //!
    public native @NoException(true) void setCoordinateTransformation(ResizeCoordinateTransformation coordTransform);
    public native @NoException(true) void setCoordinateTransformation(@Cast("nvinfer1::ResizeCoordinateTransformation") int coordTransform);

    /**
     *  \brief Get coordinate transformation function.
     * 
     *  @return The coordinate transformation function.
     *  */
    
    
    //!
    //!
    //!
    //!
    //!
    public native @NoException(true) ResizeCoordinateTransformation getCoordinateTransformation();

    /**
     *  \brief Set coordinate selector function when resized to single pixel.
     * 
     *  When resize to single pixel image, use this function to decide how to map the coordinate in the original
     *  image.
     * 
     *  Default is ResizeSelector::kFORMULA.
     * 
     *  @see ResizeSelector
     *  */
    
    
    //!
    //!
    //!
    public native @NoException(true) void setSelectorForSinglePixel(ResizeSelector selector);
    public native @NoException(true) void setSelectorForSinglePixel(@Cast("nvinfer1::ResizeSelector") int selector);

    /**
     *  \brief Get the coordinate selector function when resized to single pixel.
     * 
     *  @return The selector function.
     *  */
    
    
    //!
    //!
    //!
    //!
    //!
    public native @NoException(true) ResizeSelector getSelectorForSinglePixel();

    /**
     *  \brief Set rounding mode for nearest neighbor resize.
     * 
     *  This value is used for nearest neighbor interpolation rounding. It is applied after coordinate transformation.
     * 
     *  Default is kFLOOR.
     * 
     *  @see ResizeRoundMode
     *  */
    
    
    //!
    //!
    //!
    public native @NoException(true) void setNearestRounding(ResizeRoundMode value);
    public native @NoException(true) void setNearestRounding(@Cast("nvinfer1::ResizeRoundMode") int value);

    /**
     *  \brief Get rounding mode for nearest neighbor resize.
     * 
     *  @return The rounding mode.
     *  */
    
    
    //!
    //!
    //!
    //!
    //!
    //!
    //!
    public native @NoException(true) ResizeRoundMode getNearestRounding();

    /**
     *  \brief Set the coefficient 'A' used in cubic interpolation.
     * 
     *  Cubic uses the coefficient 'A' to calculate the weight of input pixels:
     * 
     *  <pre>
     *  x := The relative distance between the sampled pixels and the input coordinates.
     * 
     *  weight(x) := for |x| <= 1, ((A + 2) * x - (A + 3)) * x * x + 1,
     *               for 1 < |x| < 2, ((A * x - 5 * A) * x + 8 * A) * x - 4 * A,
     *               others 0;
     *  </pre>
     * 
     *  This attribute is valid only if "resize mode" is "cubic".
     * 
     *  The default value is -0.75.
     *  */
    
    
    //!
    //!
    //!
    public native @NoException(true) void setCubicCoeff(float A);

    /**
     *  \brief Get the coefficient 'A' used in cubic interpolation.
     * 
     *  @see setCubicCoeff()
     *  */
    
    
    //!
    //!
    //!
    //!
    public native @NoException(true) float getCubicCoeff();

    /**
     *  \brief Set the state for excluding outside pixels.
     * 
     *  If set to true, the weight of sampling locations outside the input tensor will be set to false, and the weight
     *  will be renormalized so that their sum is 1.0.
     * 
     *  The default value is false.
     *  */
    
    
    //!
    //!
    //!
    public native @NoException(true) void setExcludeOutside(@Cast("bool") boolean excludeFlag);

    /**
     *  \brief Get the state for excluding outside pixels.
     * 
     *  @see setExcludeOutside()
     *  */
    public native @Cast("bool") @NoException(true) boolean getExcludeOutside();
}
