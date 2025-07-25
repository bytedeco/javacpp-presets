// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.cuda.global;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import org.bytedeco.cuda.cudart.*;
import static org.bytedeco.cuda.global.cudart.*;

public class cufft extends org.bytedeco.cuda.presets.cufft {
    static { Loader.load(); }

// Parsed from <cufft.h>

 /* Copyright 2005-2021 NVIDIA Corporation.  All rights reserved.
  *
  * NOTICE TO LICENSEE:
  *
  * The source code and/or documentation ("Licensed Deliverables") are
  * subject to NVIDIA intellectual property rights under U.S. and
  * international Copyright laws.
  *
  * The Licensed Deliverables contained herein are PROPRIETARY and
  * CONFIDENTIAL to NVIDIA and are being provided under the terms and
  * conditions of a form of NVIDIA software license agreement by and
  * between NVIDIA and Licensee ("License Agreement") or electronically
  * accepted by Licensee.  Notwithstanding any terms or conditions to
  * the contrary in the License Agreement, reproduction or disclosure
  * of the Licensed Deliverables to any third party without the express
  * written consent of NVIDIA is prohibited.
  *
  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
  * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
  * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  THEY ARE
  * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
  * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
  * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
  * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
  * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
  * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
  * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
  * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
  * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
  * OF THESE LICENSED DELIVERABLES.
  *
  * U.S. Government End Users.  These Licensed Deliverables are a
  * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
  * 1995), consisting of "commercial computer software" and "commercial
  * computer software documentation" as such terms are used in 48
  * C.F.R. 12.212 (SEPT 1995) and are provided to the U.S. Government
  * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
  * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
  * U.S. Government End Users acquire the Licensed Deliverables with
  * only those rights set forth herein.
  *
  * Any use of the Licensed Deliverables in individual and commercial
  * software must include, in the user documentation and internal
  * comments to the code, the above Disclaimer and U.S. Government End
  * Users Notice.
  */

/**
* \file cufft.h
* \brief Public header file for the NVIDIA CUDA FFT library (CUFFT)
*/

// #ifndef _CUFFT_H_
// #define _CUFFT_H_


// #include "cuComplex.h"
// #include "driver_types.h"
// #include "library_types.h"

// #ifndef CUFFTAPI
// #ifdef _WIN32
// #define CUFFTAPI __stdcall
// #elif __GNUC__ >= 4
// #define CUFFTAPI __attribute__ ((visibility ("default")))
// #else
// #define CUFFTAPI
// #endif
// #endif

// #ifdef __cplusplus
// #endif

public static final int CUFFT_VER_MAJOR = 11;
public static final int CUFFT_VER_MINOR = 4;
public static final int CUFFT_VER_PATCH = 0;
public static final int CUFFT_VER_BUILD = 6;

public static final int CUFFT_VERSION = 11400;

// CUFFT API function return values
/** enum cufftResult */
public static final int
  CUFFT_SUCCESS        = 0x0,
  CUFFT_INVALID_PLAN   = 0x1,
  CUFFT_ALLOC_FAILED   = 0x2,
  CUFFT_INVALID_TYPE   = 0x3,
  CUFFT_INVALID_VALUE  = 0x4,
  CUFFT_INTERNAL_ERROR = 0x5,
  CUFFT_EXEC_FAILED    = 0x6,
  CUFFT_SETUP_FAILED   = 0x7,
  CUFFT_INVALID_SIZE   = 0x8,
  CUFFT_UNALIGNED_DATA = 0x9,
  CUFFT_INCOMPLETE_PARAMETER_LIST = 0xA,
  CUFFT_INVALID_DEVICE = 0xB,
  CUFFT_PARSE_ERROR = 0xC,
  CUFFT_NO_WORKSPACE = 0xD,
  CUFFT_NOT_IMPLEMENTED = 0xE,
  CUFFT_LICENSE_ERROR = 0x0F,
  CUFFT_NOT_SUPPORTED = 0x10;

public static final int MAX_CUFFT_ERROR = 0x11;


// CUFFT defines and supports the following data types


// cufftReal is a single-precision, floating-point real data type.
// cufftDoubleReal is a double-precision, real data type.

// cufftComplex is a single-precision, floating-point complex data type that
// consists of interleaved real and imaginary components.
// cufftDoubleComplex is the double-precision equivalent.

// CUFFT transform directions
public static final int CUFFT_FORWARD = -1; // Forward FFT
public static final int CUFFT_INVERSE =  1; // Inverse FFT

// CUFFT supports the following transform types
/** enum cufftType */
public static final int
  CUFFT_R2C = 0x2a,     // Real to Complex (interleaved)
  CUFFT_C2R = 0x2c,     // Complex (interleaved) to Real
  CUFFT_C2C = 0x29,     // Complex to Complex, interleaved
  CUFFT_D2Z = 0x6a,     // Double to Double-Complex
  CUFFT_Z2D = 0x6c,     // Double-Complex to Double
  CUFFT_Z2Z = 0x69;      // Double-Complex to Double-Complex

// CUFFT supports the following data layouts
/** enum cufftCompatibility */
public static final int
    CUFFT_COMPATIBILITY_FFTW_PADDING    = 0x01;    // The default value

public static final int CUFFT_COMPATIBILITY_DEFAULT =   CUFFT_COMPATIBILITY_FFTW_PADDING;

//
// structure definition used by the shim between old and new APIs
//
public static final int MAX_SHIM_RANK = 3;

// cufftHandle is a handle type used to store and access CUFFT plans.


public static native @Cast("cufftResult") int cufftPlan1d(@Cast("cufftHandle*") IntPointer plan,
                                 int nx,
                                 @Cast("cufftType") int type,
                                 int batch);
public static native @Cast("cufftResult") int cufftPlan1d(@Cast("cufftHandle*") IntBuffer plan,
                                 int nx,
                                 @Cast("cufftType") int type,
                                 int batch);
public static native @Cast("cufftResult") int cufftPlan1d(@Cast("cufftHandle*") int[] plan,
                                 int nx,
                                 @Cast("cufftType") int type,
                                 int batch);

public static native @Cast("cufftResult") int cufftPlan2d(@Cast("cufftHandle*") IntPointer plan,
                                 int nx, int ny,
                                 @Cast("cufftType") int type);
public static native @Cast("cufftResult") int cufftPlan2d(@Cast("cufftHandle*") IntBuffer plan,
                                 int nx, int ny,
                                 @Cast("cufftType") int type);
public static native @Cast("cufftResult") int cufftPlan2d(@Cast("cufftHandle*") int[] plan,
                                 int nx, int ny,
                                 @Cast("cufftType") int type);

public static native @Cast("cufftResult") int cufftPlan3d(@Cast("cufftHandle*") IntPointer plan,
                                 int nx, int ny, int nz,
                                 @Cast("cufftType") int type);
public static native @Cast("cufftResult") int cufftPlan3d(@Cast("cufftHandle*") IntBuffer plan,
                                 int nx, int ny, int nz,
                                 @Cast("cufftType") int type);
public static native @Cast("cufftResult") int cufftPlan3d(@Cast("cufftHandle*") int[] plan,
                                 int nx, int ny, int nz,
                                 @Cast("cufftType") int type);

public static native @Cast("cufftResult") int cufftPlanMany(@Cast("cufftHandle*") IntPointer plan,
                                   int rank,
                                   IntPointer n,
                                   IntPointer inembed, int istride, int idist,
                                   IntPointer onembed, int ostride, int odist,
                                   @Cast("cufftType") int type,
                                   int batch);
public static native @Cast("cufftResult") int cufftPlanMany(@Cast("cufftHandle*") IntBuffer plan,
                                   int rank,
                                   IntBuffer n,
                                   IntBuffer inembed, int istride, int idist,
                                   IntBuffer onembed, int ostride, int odist,
                                   @Cast("cufftType") int type,
                                   int batch);
public static native @Cast("cufftResult") int cufftPlanMany(@Cast("cufftHandle*") int[] plan,
                                   int rank,
                                   int[] n,
                                   int[] inembed, int istride, int idist,
                                   int[] onembed, int ostride, int odist,
                                   @Cast("cufftType") int type,
                                   int batch);

public static native @Cast("cufftResult") int cufftMakePlan1d(@Cast("cufftHandle") int plan,
                                     int nx,
                                     @Cast("cufftType") int type,
                                     int batch,
                                     @Cast("size_t*") SizeTPointer workSize);

public static native @Cast("cufftResult") int cufftMakePlan2d(@Cast("cufftHandle") int plan,
                                     int nx, int ny,
                                     @Cast("cufftType") int type,
                                     @Cast("size_t*") SizeTPointer workSize);

public static native @Cast("cufftResult") int cufftMakePlan3d(@Cast("cufftHandle") int plan,
                                     int nx, int ny, int nz,
                                     @Cast("cufftType") int type,
                                     @Cast("size_t*") SizeTPointer workSize);

public static native @Cast("cufftResult") int cufftMakePlanMany(@Cast("cufftHandle") int plan,
                                       int rank,
                                       IntPointer n,
                                       IntPointer inembed, int istride, int idist,
                                       IntPointer onembed, int ostride, int odist,
                                       @Cast("cufftType") int type,
                                       int batch,
                                       @Cast("size_t*") SizeTPointer workSize);
public static native @Cast("cufftResult") int cufftMakePlanMany(@Cast("cufftHandle") int plan,
                                       int rank,
                                       IntBuffer n,
                                       IntBuffer inembed, int istride, int idist,
                                       IntBuffer onembed, int ostride, int odist,
                                       @Cast("cufftType") int type,
                                       int batch,
                                       @Cast("size_t*") SizeTPointer workSize);
public static native @Cast("cufftResult") int cufftMakePlanMany(@Cast("cufftHandle") int plan,
                                       int rank,
                                       int[] n,
                                       int[] inembed, int istride, int idist,
                                       int[] onembed, int ostride, int odist,
                                       @Cast("cufftType") int type,
                                       int batch,
                                       @Cast("size_t*") SizeTPointer workSize);

public static native @Cast("cufftResult") int cufftMakePlanMany64(@Cast("cufftHandle") int plan,
                                         int rank,
                                         @Cast("long long int*") LongPointer n,
                                         @Cast("long long int*") LongPointer inembed,
                                         @Cast("long long int") long istride,
                                         @Cast("long long int") long idist,
                                         @Cast("long long int*") LongPointer onembed,
                                         @Cast("long long int") long ostride, @Cast("long long int") long odist,
                                         @Cast("cufftType") int type,
                                         @Cast("long long int") long batch,
                                         @Cast("size_t*") SizeTPointer workSize);
public static native @Cast("cufftResult") int cufftMakePlanMany64(@Cast("cufftHandle") int plan,
                                         int rank,
                                         @Cast("long long int*") LongBuffer n,
                                         @Cast("long long int*") LongBuffer inembed,
                                         @Cast("long long int") long istride,
                                         @Cast("long long int") long idist,
                                         @Cast("long long int*") LongBuffer onembed,
                                         @Cast("long long int") long ostride, @Cast("long long int") long odist,
                                         @Cast("cufftType") int type,
                                         @Cast("long long int") long batch,
                                         @Cast("size_t*") SizeTPointer workSize);
public static native @Cast("cufftResult") int cufftMakePlanMany64(@Cast("cufftHandle") int plan,
                                         int rank,
                                         @Cast("long long int*") long[] n,
                                         @Cast("long long int*") long[] inembed,
                                         @Cast("long long int") long istride,
                                         @Cast("long long int") long idist,
                                         @Cast("long long int*") long[] onembed,
                                         @Cast("long long int") long ostride, @Cast("long long int") long odist,
                                         @Cast("cufftType") int type,
                                         @Cast("long long int") long batch,
                                         @Cast("size_t*") SizeTPointer workSize);

public static native @Cast("cufftResult") int cufftGetSizeMany64(@Cast("cufftHandle") int plan,
                                        int rank,
                                        @Cast("long long int*") LongPointer n,
                                        @Cast("long long int*") LongPointer inembed,
                                        @Cast("long long int") long istride, @Cast("long long int") long idist,
                                        @Cast("long long int*") LongPointer onembed,
                                        @Cast("long long int") long ostride, @Cast("long long int") long odist,
                                        @Cast("cufftType") int type,
                                        @Cast("long long int") long batch,
                                        @Cast("size_t*") SizeTPointer workSize);
public static native @Cast("cufftResult") int cufftGetSizeMany64(@Cast("cufftHandle") int plan,
                                        int rank,
                                        @Cast("long long int*") LongBuffer n,
                                        @Cast("long long int*") LongBuffer inembed,
                                        @Cast("long long int") long istride, @Cast("long long int") long idist,
                                        @Cast("long long int*") LongBuffer onembed,
                                        @Cast("long long int") long ostride, @Cast("long long int") long odist,
                                        @Cast("cufftType") int type,
                                        @Cast("long long int") long batch,
                                        @Cast("size_t*") SizeTPointer workSize);
public static native @Cast("cufftResult") int cufftGetSizeMany64(@Cast("cufftHandle") int plan,
                                        int rank,
                                        @Cast("long long int*") long[] n,
                                        @Cast("long long int*") long[] inembed,
                                        @Cast("long long int") long istride, @Cast("long long int") long idist,
                                        @Cast("long long int*") long[] onembed,
                                        @Cast("long long int") long ostride, @Cast("long long int") long odist,
                                        @Cast("cufftType") int type,
                                        @Cast("long long int") long batch,
                                        @Cast("size_t*") SizeTPointer workSize);




public static native @Cast("cufftResult") int cufftEstimate1d(int nx,
                                     @Cast("cufftType") int type,
                                     int batch,
                                     @Cast("size_t*") SizeTPointer workSize);

public static native @Cast("cufftResult") int cufftEstimate2d(int nx, int ny,
                                     @Cast("cufftType") int type,
                                     @Cast("size_t*") SizeTPointer workSize);

public static native @Cast("cufftResult") int cufftEstimate3d(int nx, int ny, int nz,
                                     @Cast("cufftType") int type,
                                     @Cast("size_t*") SizeTPointer workSize);

public static native @Cast("cufftResult") int cufftEstimateMany(int rank,
                                       IntPointer n,
                                       IntPointer inembed, int istride, int idist,
                                       IntPointer onembed, int ostride, int odist,
                                       @Cast("cufftType") int type,
                                       int batch,
                                       @Cast("size_t*") SizeTPointer workSize);
public static native @Cast("cufftResult") int cufftEstimateMany(int rank,
                                       IntBuffer n,
                                       IntBuffer inembed, int istride, int idist,
                                       IntBuffer onembed, int ostride, int odist,
                                       @Cast("cufftType") int type,
                                       int batch,
                                       @Cast("size_t*") SizeTPointer workSize);
public static native @Cast("cufftResult") int cufftEstimateMany(int rank,
                                       int[] n,
                                       int[] inembed, int istride, int idist,
                                       int[] onembed, int ostride, int odist,
                                       @Cast("cufftType") int type,
                                       int batch,
                                       @Cast("size_t*") SizeTPointer workSize);

public static native @Cast("cufftResult") int cufftCreate(@Cast("cufftHandle*") IntPointer handle);
public static native @Cast("cufftResult") int cufftCreate(@Cast("cufftHandle*") IntBuffer handle);
public static native @Cast("cufftResult") int cufftCreate(@Cast("cufftHandle*") int[] handle);

public static native @Cast("cufftResult") int cufftGetSize1d(@Cast("cufftHandle") int handle,
                                    int nx,
                                    @Cast("cufftType") int type,
                                    int batch,
                                    @Cast("size_t*") SizeTPointer workSize );

public static native @Cast("cufftResult") int cufftGetSize2d(@Cast("cufftHandle") int handle,
                                    int nx, int ny,
                                    @Cast("cufftType") int type,
                                    @Cast("size_t*") SizeTPointer workSize);

public static native @Cast("cufftResult") int cufftGetSize3d(@Cast("cufftHandle") int handle,
                                    int nx, int ny, int nz,
                                    @Cast("cufftType") int type,
                                    @Cast("size_t*") SizeTPointer workSize);

public static native @Cast("cufftResult") int cufftGetSizeMany(@Cast("cufftHandle") int handle,
                                      int rank, IntPointer n,
                                      IntPointer inembed, int istride, int idist,
                                      IntPointer onembed, int ostride, int odist,
                                      @Cast("cufftType") int type, int batch, @Cast("size_t*") SizeTPointer workArea);
public static native @Cast("cufftResult") int cufftGetSizeMany(@Cast("cufftHandle") int handle,
                                      int rank, IntBuffer n,
                                      IntBuffer inembed, int istride, int idist,
                                      IntBuffer onembed, int ostride, int odist,
                                      @Cast("cufftType") int type, int batch, @Cast("size_t*") SizeTPointer workArea);
public static native @Cast("cufftResult") int cufftGetSizeMany(@Cast("cufftHandle") int handle,
                                      int rank, int[] n,
                                      int[] inembed, int istride, int idist,
                                      int[] onembed, int ostride, int odist,
                                      @Cast("cufftType") int type, int batch, @Cast("size_t*") SizeTPointer workArea);

public static native @Cast("cufftResult") int cufftGetSize(@Cast("cufftHandle") int handle, @Cast("size_t*") SizeTPointer workSize);

public static native @Cast("cufftResult") int cufftSetWorkArea(@Cast("cufftHandle") int plan, Pointer workArea);

public static native @Cast("cufftResult") int cufftSetAutoAllocation(@Cast("cufftHandle") int plan, int autoAllocate);

public static native @Cast("cufftResult") int cufftExecC2C(@Cast("cufftHandle") int plan,
                                  @Cast("cufftComplex*") float2 idata,
                                  @Cast("cufftComplex*") float2 odata,
                                  int direction);

public static native @Cast("cufftResult") int cufftExecR2C(@Cast("cufftHandle") int plan,
                                  @Cast("cufftReal*") FloatPointer idata,
                                  @Cast("cufftComplex*") float2 odata);
public static native @Cast("cufftResult") int cufftExecR2C(@Cast("cufftHandle") int plan,
                                  @Cast("cufftReal*") FloatBuffer idata,
                                  @Cast("cufftComplex*") float2 odata);
public static native @Cast("cufftResult") int cufftExecR2C(@Cast("cufftHandle") int plan,
                                  @Cast("cufftReal*") float[] idata,
                                  @Cast("cufftComplex*") float2 odata);

public static native @Cast("cufftResult") int cufftExecC2R(@Cast("cufftHandle") int plan,
                                  @Cast("cufftComplex*") float2 idata,
                                  @Cast("cufftReal*") FloatPointer odata);
public static native @Cast("cufftResult") int cufftExecC2R(@Cast("cufftHandle") int plan,
                                  @Cast("cufftComplex*") float2 idata,
                                  @Cast("cufftReal*") FloatBuffer odata);
public static native @Cast("cufftResult") int cufftExecC2R(@Cast("cufftHandle") int plan,
                                  @Cast("cufftComplex*") float2 idata,
                                  @Cast("cufftReal*") float[] odata);

public static native @Cast("cufftResult") int cufftExecZ2Z(@Cast("cufftHandle") int plan,
                                  @Cast("cufftDoubleComplex*") double2 idata,
                                  @Cast("cufftDoubleComplex*") double2 odata,
                                  int direction);

public static native @Cast("cufftResult") int cufftExecD2Z(@Cast("cufftHandle") int plan,
                                  @Cast("cufftDoubleReal*") DoublePointer idata,
                                  @Cast("cufftDoubleComplex*") double2 odata);
public static native @Cast("cufftResult") int cufftExecD2Z(@Cast("cufftHandle") int plan,
                                  @Cast("cufftDoubleReal*") DoubleBuffer idata,
                                  @Cast("cufftDoubleComplex*") double2 odata);
public static native @Cast("cufftResult") int cufftExecD2Z(@Cast("cufftHandle") int plan,
                                  @Cast("cufftDoubleReal*") double[] idata,
                                  @Cast("cufftDoubleComplex*") double2 odata);

public static native @Cast("cufftResult") int cufftExecZ2D(@Cast("cufftHandle") int plan,
                                  @Cast("cufftDoubleComplex*") double2 idata,
                                  @Cast("cufftDoubleReal*") DoublePointer odata);
public static native @Cast("cufftResult") int cufftExecZ2D(@Cast("cufftHandle") int plan,
                                  @Cast("cufftDoubleComplex*") double2 idata,
                                  @Cast("cufftDoubleReal*") DoubleBuffer odata);
public static native @Cast("cufftResult") int cufftExecZ2D(@Cast("cufftHandle") int plan,
                                  @Cast("cufftDoubleComplex*") double2 idata,
                                  @Cast("cufftDoubleReal*") double[] odata);


// utility functions
public static native @Cast("cufftResult") int cufftSetStream(@Cast("cufftHandle") int plan,
                                    CUstream_st stream);

public static native @Cast("cufftResult") int cufftDestroy(@Cast("cufftHandle") int plan);

public static native @Cast("cufftResult") int cufftGetVersion(IntPointer version);
public static native @Cast("cufftResult") int cufftGetVersion(IntBuffer version);
public static native @Cast("cufftResult") int cufftGetVersion(int[] version);

public static native @Cast("cufftResult") int cufftGetProperty(@Cast("libraryPropertyType") int type,
                                      IntPointer value);
public static native @Cast("cufftResult") int cufftGetProperty(@Cast("libraryPropertyType") int type,
                                      IntBuffer value);
public static native @Cast("cufftResult") int cufftGetProperty(@Cast("libraryPropertyType") int type,
                                      int[] value);

//
// Set/Get PlanProperty APIs configures per-plan behavior 
//
/** enum cufftProperty */
public static final int
    NVFFT_PLAN_PROPERTY_INT64_PATIENT_JIT = 0x1,
    NVFFT_PLAN_PROPERTY_INT64_MAX_NUM_HOST_THREADS = 0x2;

public static native @Cast("cufftResult") int cufftSetPlanPropertyInt64(@Cast("cufftHandle") int plan, 
                                               @Cast("cufftProperty") int property, 
                                               @Cast("const long long int") long inputValueInt);

public static native @Cast("cufftResult") int cufftGetPlanPropertyInt64(@Cast("cufftHandle") int plan, 
                                               @Cast("cufftProperty") int property, 
                                               @Cast("long long int*") LongPointer returnPtrValue);
public static native @Cast("cufftResult") int cufftGetPlanPropertyInt64(@Cast("cufftHandle") int plan, 
                                               @Cast("cufftProperty") int property, 
                                               @Cast("long long int*") LongBuffer returnPtrValue);
public static native @Cast("cufftResult") int cufftGetPlanPropertyInt64(@Cast("cufftHandle") int plan, 
                                               @Cast("cufftProperty") int property, 
                                               @Cast("long long int*") long[] returnPtrValue);

public static native @Cast("cufftResult") int cufftResetPlanProperty(@Cast("cufftHandle") int plan, @Cast("cufftProperty") int property);

// #ifdef __cplusplus
// #endif

// #endif /* _CUFFT_H_ */


}
