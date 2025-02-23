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

@NoOffset @Name("std::tuple<bool,float>") @Properties(inherit = org.bytedeco.depthai.presets.depthai.class)
public class BoolFloatTuple extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public BoolFloatTuple(Pointer p) { super(p); }
    public BoolFloatTuple(@Cast("bool") boolean value0, float value1) { allocate(value0, value1); }
    private native void allocate(@Cast("bool") boolean value0, float value1);
    public BoolFloatTuple()       { allocate();  }
    private native void allocate();
    public native @Name("operator =") @ByRef BoolFloatTuple put(@ByRef BoolFloatTuple x);

    public @Cast("bool") boolean get0() { return get0(this); }
    @Namespace @Name("std::get<0>") public static native @Cast("bool") boolean get0(@ByRef BoolFloatTuple container);
    public float get1() { return get1(this); }
    @Namespace @Name("std::get<1>") public static native float get1(@ByRef BoolFloatTuple container);
}

