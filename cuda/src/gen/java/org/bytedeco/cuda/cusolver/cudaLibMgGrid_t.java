// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.cuda.cusolver;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import org.bytedeco.cuda.cudart.*;
import static org.bytedeco.cuda.global.cudart.*;
import org.bytedeco.cuda.cublas.*;
import static org.bytedeco.cuda.global.cublas.*;
import org.bytedeco.cuda.cusparse.*;
import static org.bytedeco.cuda.global.cusparse.*;

import static org.bytedeco.cuda.global.cusolver.*;


  /** \brief Opaque structure of the distributed grid */
  @Namespace @Name("void") @Opaque @Properties(inherit = org.bytedeco.cuda.presets.cusolver.class)
public class cudaLibMgGrid_t extends Pointer {
      /** Empty constructor. Calls {@code super((Pointer)null)}. */
      public cudaLibMgGrid_t() { super((Pointer)null); }
      /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
      public cudaLibMgGrid_t(Pointer p) { super(p); }
  }
