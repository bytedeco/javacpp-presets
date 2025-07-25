// Targeted by JavaCPP version 1.5.13-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.opencv.opencv_dnn;

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

import static org.bytedeco.opencv.global.opencv_dnn.*;


    /** LSTM recurrent layer */
    @Namespace("cv::dnn") @Properties(inherit = org.bytedeco.opencv.presets.opencv_dnn.class)
public class LSTMLayer extends Layer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public LSTMLayer(Pointer p) { super(p); }
    
        /** Creates instance of LSTM layer */
        public static native @Ptr LSTMLayer create(@Const @ByRef LayerParams params);

        /** @deprecated Use LayerParams::blobs instead.
        \brief Set trained weights for LSTM layer.
        <p>
        LSTM behavior on each step is defined by current input, previous output, previous cell state and learned weights.
        <p>
        Let {@code x_t} be current input, {@code h_t} be current output, {@code c_t} be current state.
        Than current output and current cell state is computed as follows:
        <pre>{@code \begin{eqnarray*}
        h_t &= o_t \odot tanh(c_t),               \\
        c_t &= f_t \odot c_{t-1} + i_t \odot g_t, \\
        \end{eqnarray*}}</pre>
        where {@code \odot} is per-element multiply operation and {@code i_t, f_t, o_t, g_t} is internal gates that are computed using learned weights.
        <p>
        Gates are computed as follows:
        <pre>{@code \begin{eqnarray*}
        i_t &= sigmoid&(W_{xi} x_t + W_{hi} h_{t-1} + b_i), \\
        f_t &= sigmoid&(W_{xf} x_t + W_{hf} h_{t-1} + b_f), \\
        o_t &= sigmoid&(W_{xo} x_t + W_{ho} h_{t-1} + b_o), \\
        g_t &= tanh   &(W_{xg} x_t + W_{hg} h_{t-1} + b_g), \\
        \end{eqnarray*}}</pre>
        where {@code W_{x?}}, {@code W_{h?}} and {@code b_{?}} are learned weights represented as matrices:
        {@code W_{x?} \in R^{N_h \times N_x}}, {@code W_{h?} \in R^{N_h \times N_h}}, {@code b_? \in R^{N_h}}.
        <p>
        For simplicity and performance purposes we use {@code  W_x = [W_{xi}; W_{xf}; W_{xo}, W_{xg}] }
        (i.e. {@code W_x} is vertical concatenation of {@code  W_{x?} }), {@code  W_x \in R^{4N_h \times N_x} }.
        The same for {@code  W_h = [W_{hi}; W_{hf}; W_{ho}, W_{hg}], W_h \in R^{4N_h \times N_h} }
        and for {@code  b = [b_i; b_f, b_o, b_g]}, {@code b \in R^{4N_h} }.
        <p>
        @param Wh is matrix defining how previous output is transformed to internal gates (i.e. according to above mentioned notation is {@code  W_h })
        @param Wx is matrix defining how current input is transformed to internal gates (i.e. according to above mentioned notation is {@code  W_x })
        @param b  is bias vector (i.e. according to above mentioned notation is {@code  b })
        */
        public native @Deprecated void setWeights(@Const @ByRef Mat Wh, @Const @ByRef Mat Wx, @Const @ByRef Mat b);

        /** \brief Specifies shape of output blob which will be [[{@code T}], {@code N}] + \p outTailShape.
          * \details If this parameter is empty or unset then \p outTailShape = [{@code Wh}.size(0)] will be used,
          * where {@code Wh} is parameter from setWeights().
          */
        public native void setOutShape(@Const @StdVector @ByRef(nullValue = "cv::dnn::MatShape()") IntPointer outTailShape);
        public native void setOutShape();

        /** @deprecated Use flag {@code produce_cell_output} in LayerParams.
          * \brief Specifies either interpret first dimension of input blob as timestamp dimension either as sample.
          *
          * If flag is set to true then shape of input blob will be interpreted as [{@code T}, {@code N}, {@code [data dims]}] where {@code T} specifies number of timestamps, {@code N} is number of independent streams.
          * In this case each forward() call will iterate through {@code T} timestamps and update layer's state {@code T} times.
          *
          * If flag is set to false then shape of input blob will be interpreted as [{@code N}, {@code [data dims]}].
          * In this case each forward() call will make one iteration and produce one timestamp with shape [{@code N}, {@code [out dims]}].
          */
        public native @Deprecated void setUseTimstampsDim(@Cast("bool") boolean use/*=true*/);
        public native @Deprecated void setUseTimstampsDim();

        /** @deprecated Use flag {@code use_timestamp_dim} in LayerParams.
         * \brief If this flag is set to true then layer will produce {@code  c_t } as second output.
         * \details Shape of the second output is the same as first output.
         */
        public native @Deprecated void setProduceCellOutput(@Cast("bool") boolean produce/*=false*/);
        public native @Deprecated void setProduceCellOutput();

        /* In common case it use single input with @f$x_t@f$ values to compute output(s) @f$h_t@f$ (and @f$c_t@f$).
         * @param input should contain packed values @f$x_t@f$
         * @param output contains computed outputs: @f$h_t@f$ (and @f$c_t@f$ if setProduceCellOutput() flag was set to true).
         *
         * If setUseTimstampsDim() is set to true then @p input[0] should has at least two dimensions with the following shape: [`T`, `N`, `[data dims]`],
         * where `T` specifies number of timestamps, `N` is number of independent streams (i.e. @f$ x_{t_0 + t}^{stream} @f$ is stored inside @p input[0][t, stream, ...]).
         *
         * If setUseTimstampsDim() is set to false then @p input[0] should contain single timestamp, its shape should has form [`N`, `[data dims]`] with at least one dimension.
         * (i.e. @f$ x_{t}^{stream} @f$ is stored inside @p input[0][stream, ...]).
        */

        public native @Override int inputNameToIndex(@Str BytePointer inputName);
        public native @Override int inputNameToIndex(@Str String inputName);
        public native @Override int outputNameToIndex(@Str BytePointer outputName);
        public native @Override int outputNameToIndex(@Str String outputName);
    }
