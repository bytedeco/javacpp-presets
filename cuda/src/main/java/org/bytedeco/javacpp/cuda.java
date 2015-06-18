// Targeted by JavaCPP version 1.0-SNAPSHOT

package org.bytedeco.javacpp;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

public class cuda extends org.bytedeco.javacpp.presets.cuda {
    static { Loader.load(); }

// Parsed from <host_defines.h>

/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
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
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
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

// #if !defined(__HOST_DEFINES_H__)
// #define __HOST_DEFINES_H__

// #if defined(__GNUC__) || defined(__CUDA_LIBDEVICE__)

// #define __no_return__
//         __attribute__((noreturn))
        
// #if defined(__CUDACC__) || defined(__CUDA_ARCH__)
/* gcc allows users to define attributes with underscores, 
   e.g., __attribute__((__noinline__)).
   Consider a non-CUDA source file (e.g. .cpp) that has the 
   above attribute specification, and includes this header file. In that case,
   defining __noinline__ as below  would cause a gcc compilation error.
   Hence, only define __noinline__ when the code is being processed
   by a  CUDA compiler component.
*/   
// #define __noinline__
//         __attribute__((noinline))
// #endif /* __CUDACC__  || __CUDA_ARCH__ */       
        
// #define __forceinline__
//         __inline__ __attribute__((always_inline))
// #define __align__(n)
//         __attribute__((aligned(n)))
// #define __thread__
//         __thread
// #define __import__
// #define __export__
// #define __cdecl
// #define __annotate__(a)
//         __attribute__((a))
// #define __location__(a)
//         __annotate__(a)
// #define CUDARTAPI

// #elif defined(_MSC_VER)

// #if _MSC_VER >= 1400

// #define __restrict__
//         __restrict

// #else /* _MSC_VER >= 1400 */

// #define __restrict__

// #endif /* _MSC_VER >= 1400 */

// #define __inline__
//         __inline
// #define __no_return__
//         __declspec(noreturn)
// #define __noinline__
//         __declspec(noinline)
// #define __forceinline__
//         __forceinline
// #define __align__(n)
//         __declspec(align(n))
// #define __thread__
//         __declspec(thread)
// #define __import__
//         __declspec(dllimport)
// #define __export__
//         __declspec(dllexport)
// #define __annotate__(a)
//         __declspec(a)
// #define __location__(a)
//         __annotate__(__##a##__)
// #define CUDARTAPI
//         __stdcall

// #else /* __GNUC__ || __CUDA_LIBDEVICE__ */

// #define __inline__

// #if !defined(__align__)

// #error --- !!! UNKNOWN COMPILER: please provide a CUDA compatible definition for '__align__' !!! ---

// #endif /* !__align__ */

// #if !defined(CUDARTAPI)

// #error --- !!! UNKNOWN COMPILER: please provide a CUDA compatible definition for 'CUDARTAPI' !!! ---

// #endif /* !CUDARTAPI */

// #endif /* !__GNUC__ */

// #if !defined(__GNUC__) || __GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 3 && !defined(__clang__) )

// #define __specialization_static
//         static

// #else /* !__GNUC__ || __GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 3) */

// #define __specialization_static

// #endif /* !__GNUC__ || __GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 3) */

// #if !defined(__CUDACC__) && !defined(__CUDABE__)

// #undef __annotate__
// #define __annotate__(a)

// #else /* !__CUDACC__ && !__CUDABE__ */

// #define __launch_bounds__(...)
//         __annotate__(launch_bounds(__VA_ARGS__))

// #endif /* !__CUDACC__ && !__CUDABE__ */

// #if defined(__CUDACC__) || defined(__CUDABE__) ||
//     defined(__GNUC__) || defined(_WIN64)

// #define __builtin_align__(a)
//         __align__(a)

// #else /* __CUDACC__ || __CUDABE__ || __GNUC__ || _WIN64 */

// #define __builtin_align__(a)

// #endif /* __CUDACC__ || __CUDABE__ || __GNUC__  || _WIN64 */

// #define __host__
//         __location__(host)
// #define __device__
//         __location__(device)
// #define __global__
//         __location__(global)
// #define __shared__
//         __location__(shared)
// #define __constant__
//         __location__(constant)
// #define __managed__
//         __location__(managed)
        
// #if defined(__CUDABE__) || !defined(__CUDACC__)
// #define __device_builtin__
// #define __device_builtin_texture_type__
// #define __device_builtin_surface_type__
// #define __cudart_builtin__
// #else /* __CUDABE__  || !__CUDACC__ */
// #endif /* __CUDABE__ || !__CUDACC__ */


// #endif /* !__HOST_DEFINES_H__ */


// Parsed from <device_types.h>

/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
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
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
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

// #if !defined(__DEVICE_TYPES_H__)
// #define __DEVICE_TYPES_H__

// #include "host_defines.h"

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

/** enum cudaRoundMode */
public static final int
    cudaRoundNearest = 0,
    cudaRoundZero = 1,
    cudaRoundPosInf = 2,
    cudaRoundMinInf = 3;

// #endif /* !__DEVICE_TYPES_H__ */


// Parsed from <driver_types.h>

/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
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
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
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

// #if !defined(__DRIVER_TYPES_H__)
// #define __DRIVER_TYPES_H__

// #include "host_defines.h"

/**
 * \defgroup CUDART_TYPES Data types used by CUDA Runtime
 * \ingroup CUDART
 *
 * @{
 */

/*******************************************************************************
*                                                                              *
*  TYPE DEFINITIONS USED BY RUNTIME API                                        *
*                                                                              *
*******************************************************************************/

// #if !defined(__CUDA_INTERNAL_COMPILATION__)

// #include <limits.h>
// #include <stddef.h>

/** Default page-locked allocation flag */
public static final int cudaHostAllocDefault =           0x00;
/** Pinned memory accessible by all CUDA contexts */
public static final int cudaHostAllocPortable =          0x01;
/** Map allocation into device space */
public static final int cudaHostAllocMapped =            0x02;
/** Write-combined memory */
public static final int cudaHostAllocWriteCombined =     0x04;

/** Default host memory registration flag */
public static final int cudaHostRegisterDefault =        0x00;
/** Pinned memory accessible by all CUDA contexts */
public static final int cudaHostRegisterPortable =       0x01;
/** Map registered memory into device space */
public static final int cudaHostRegisterMapped =         0x02;

/** Default peer addressing enable flag */
public static final int cudaPeerAccessDefault =          0x00;

/** Default stream flag */
public static final int cudaStreamDefault =              0x00;
/** Stream does not synchronize with stream 0 (the NULL stream) */
public static final int cudaStreamNonBlocking =          0x01;


/** Default event flag */
public static final int cudaEventDefault =               0x00;
/** Event uses blocking synchronization */
public static final int cudaEventBlockingSync =          0x01;
/** Event will not record timing data */
public static final int cudaEventDisableTiming =         0x02;
/** Event is suitable for interprocess use. cudaEventDisableTiming must be set */
public static final int cudaEventInterprocess =          0x04;

/** Device flag - Automatic scheduling */
public static final int cudaDeviceScheduleAuto =         0x00;
/** Device flag - Spin default scheduling */
public static final int cudaDeviceScheduleSpin =         0x01;
/** Device flag - Yield default scheduling */
public static final int cudaDeviceScheduleYield =        0x02;
/** Device flag - Use blocking synchronization */
public static final int cudaDeviceScheduleBlockingSync = 0x04;
/** Device flag - Use blocking synchronization 
                                               *  \deprecated This flag was deprecated as of CUDA 4.0 and
                                               *  replaced with ::cudaDeviceScheduleBlockingSync. */
public static final int cudaDeviceBlockingSync =         0x04;
/** Device schedule flags mask */
public static final int cudaDeviceScheduleMask =         0x07;
/** Device flag - Support mapped pinned allocations */
public static final int cudaDeviceMapHost =              0x08;
/** Device flag - Keep local memory allocation after launch */
public static final int cudaDeviceLmemResizeToMax =      0x10;
/** Device flags mask */
public static final int cudaDeviceMask =                 0x1f;

/** Default CUDA array allocation flag */
public static final int cudaArrayDefault =               0x00;
/** Must be set in cudaMalloc3DArray to create a layered CUDA array */
public static final int cudaArrayLayered =               0x01;
/** Must be set in cudaMallocArray or cudaMalloc3DArray in order to bind surfaces to the CUDA array */
public static final int cudaArraySurfaceLoadStore =      0x02;
/** Must be set in cudaMalloc3DArray to create a cubemap CUDA array */
public static final int cudaArrayCubemap =               0x04;
/** Must be set in cudaMallocArray or cudaMalloc3DArray in order to perform texture gather operations on the CUDA array */
public static final int cudaArrayTextureGather =         0x08;

/** Automatically enable peer access between remote devices as needed */
public static final int cudaIpcMemLazyEnablePeerAccess = 0x01;

/** Memory can be accessed by any stream on any device*/
public static final int cudaMemAttachGlobal =            0x01;
/** Memory cannot be accessed by any stream on any device */
public static final int cudaMemAttachHost =              0x02;
/** Memory can only be accessed by a single stream on the associated device */
public static final int cudaMemAttachSingle =            0x04;

// #endif /* !__CUDA_INTERNAL_COMPILATION__ */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

/**
 * CUDA error types
 */
/** enum cudaError */
public static final int
    /**
     * The API call returned with no errors. In the case of query calls, this
     * can also mean that the operation being queried is complete (see
     * ::cudaEventQuery() and ::cudaStreamQuery()).
     */
    cudaSuccess                           = 0,
  
    /**
     * The device function being invoked (usually via ::cudaLaunch()) was not
     * previously configured via the ::cudaConfigureCall() function.
     */
    cudaErrorMissingConfiguration         = 1,
  
    /**
     * The API call failed because it was unable to allocate enough memory to
     * perform the requested operation.
     */
    cudaErrorMemoryAllocation             = 2,
  
    /**
     * The API call failed because the CUDA driver and runtime could not be
     * initialized.
     */
    cudaErrorInitializationError          = 3,
  
    /**
     * An exception occurred on the device while executing a kernel. Common
     * causes include dereferencing an invalid device pointer and accessing
     * out of bounds shared memory. The device cannot be used until
     * ::cudaThreadExit() is called. All existing device memory allocations
     * are invalid and must be reconstructed if the program is to continue
     * using CUDA.
     */
    cudaErrorLaunchFailure                = 4,
  
    /**
     * This indicated that a previous kernel launch failed. This was previously
     * used for device emulation of kernel launches.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    cudaErrorPriorLaunchFailure           = 5,
  
    /**
     * This indicates that the device kernel took too long to execute. This can
     * only occur if timeouts are enabled - see the device property
     * \ref ::cudaDeviceProp::kernelExecTimeoutEnabled "kernelExecTimeoutEnabled"
     * for more information. The device cannot be used until ::cudaThreadExit()
     * is called. All existing device memory allocations are invalid and must be
     * reconstructed if the program is to continue using CUDA.
     */
    cudaErrorLaunchTimeout                = 6,
  
    /**
     * This indicates that a launch did not occur because it did not have
     * appropriate resources. Although this error is similar to
     * ::cudaErrorInvalidConfiguration, this error usually indicates that the
     * user has attempted to pass too many arguments to the device kernel, or the
     * kernel launch specifies too many threads for the kernel's register count.
     */
    cudaErrorLaunchOutOfResources         = 7,
  
    /**
     * The requested device function does not exist or is not compiled for the
     * proper device architecture.
     */
    cudaErrorInvalidDeviceFunction        = 8,
  
    /**
     * This indicates that a kernel launch is requesting resources that can
     * never be satisfied by the current device. Requesting more shared memory
     * per block than the device supports will trigger this error, as will
     * requesting too many threads or blocks. See ::cudaDeviceProp for more
     * device limitations.
     */
    cudaErrorInvalidConfiguration         = 9,
  
    /**
     * This indicates that the device ordinal supplied by the user does not
     * correspond to a valid CUDA device.
     */
    cudaErrorInvalidDevice                = 10,
  
    /**
     * This indicates that one or more of the parameters passed to the API call
     * is not within an acceptable range of values.
     */
    cudaErrorInvalidValue                 = 11,
  
    /**
     * This indicates that one or more of the pitch-related parameters passed
     * to the API call is not within the acceptable range for pitch.
     */
    cudaErrorInvalidPitchValue            = 12,
  
    /**
     * This indicates that the symbol name/identifier passed to the API call
     * is not a valid name or identifier.
     */
    cudaErrorInvalidSymbol                = 13,
  
    /**
     * This indicates that the buffer object could not be mapped.
     */
    cudaErrorMapBufferObjectFailed        = 14,
  
    /**
     * This indicates that the buffer object could not be unmapped.
     */
    cudaErrorUnmapBufferObjectFailed      = 15,
  
    /**
     * This indicates that at least one host pointer passed to the API call is
     * not a valid host pointer.
     */
    cudaErrorInvalidHostPointer           = 16,
  
    /**
     * This indicates that at least one device pointer passed to the API call is
     * not a valid device pointer.
     */
    cudaErrorInvalidDevicePointer         = 17,
  
    /**
     * This indicates that the texture passed to the API call is not a valid
     * texture.
     */
    cudaErrorInvalidTexture               = 18,
  
    /**
     * This indicates that the texture binding is not valid. This occurs if you
     * call ::cudaGetTextureAlignmentOffset() with an unbound texture.
     */
    cudaErrorInvalidTextureBinding        = 19,
  
    /**
     * This indicates that the channel descriptor passed to the API call is not
     * valid. This occurs if the format is not one of the formats specified by
     * ::cudaChannelFormatKind, or if one of the dimensions is invalid.
     */
    cudaErrorInvalidChannelDescriptor     = 20,
  
    /**
     * This indicates that the direction of the memcpy passed to the API call is
     * not one of the types specified by ::cudaMemcpyKind.
     */
    cudaErrorInvalidMemcpyDirection       = 21,
  
    /**
     * This indicated that the user has taken the address of a constant variable,
     * which was forbidden up until the CUDA 3.1 release.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Variables in constant
     * memory may now have their address taken by the runtime via
     * ::cudaGetSymbolAddress().
     */
    cudaErrorAddressOfConstant            = 22,
  
    /**
     * This indicated that a texture fetch was not able to be performed.
     * This was previously used for device emulation of texture operations.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    cudaErrorTextureFetchFailed           = 23,
  
    /**
     * This indicated that a texture was not bound for access.
     * This was previously used for device emulation of texture operations.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    cudaErrorTextureNotBound              = 24,
  
    /**
     * This indicated that a synchronization operation had failed.
     * This was previously used for some device emulation functions.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    cudaErrorSynchronizationError         = 25,
  
    /**
     * This indicates that a non-float texture was being accessed with linear
     * filtering. This is not supported by CUDA.
     */
    cudaErrorInvalidFilterSetting         = 26,
  
    /**
     * This indicates that an attempt was made to read a non-float texture as a
     * normalized float. This is not supported by CUDA.
     */
    cudaErrorInvalidNormSetting           = 27,
  
    /**
     * Mixing of device and device emulation code was not allowed.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    cudaErrorMixedDeviceExecution         = 28,
  
    /**
     * This indicates that a CUDA Runtime API call cannot be executed because
     * it is being called during process shut down, at a point in time after
     * CUDA driver has been unloaded.
     */
    cudaErrorCudartUnloading              = 29,
  
    /**
     * This indicates that an unknown internal error has occurred.
     */
    cudaErrorUnknown                      = 30,

    /**
     * This indicates that the API call is not yet implemented. Production
     * releases of CUDA will never return this error.
     * \deprecated
     * This error return is deprecated as of CUDA 4.1.
     */
    cudaErrorNotYetImplemented            = 31,
  
    /**
     * This indicated that an emulated device pointer exceeded the 32-bit address
     * range.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    cudaErrorMemoryValueTooLarge          = 32,
  
    /**
     * This indicates that a resource handle passed to the API call was not
     * valid. Resource handles are opaque types like ::cudaStream_t and
     * ::cudaEvent_t.
     */
    cudaErrorInvalidResourceHandle        = 33,
  
    /**
     * This indicates that asynchronous operations issued previously have not
     * completed yet. This result is not actually an error, but must be indicated
     * differently than ::cudaSuccess (which indicates completion). Calls that
     * may return this value include ::cudaEventQuery() and ::cudaStreamQuery().
     */
    cudaErrorNotReady                     = 34,
  
    /**
     * This indicates that the installed NVIDIA CUDA driver is older than the
     * CUDA runtime library. This is not a supported configuration. Users should
     * install an updated NVIDIA display driver to allow the application to run.
     */
    cudaErrorInsufficientDriver           = 35,
  
    /**
     * This indicates that the user has called ::cudaSetValidDevices(),
     * ::cudaSetDeviceFlags(), ::cudaD3D9SetDirect3DDevice(),
     * ::cudaD3D10SetDirect3DDevice, ::cudaD3D11SetDirect3DDevice(), or
     * ::cudaVDPAUSetVDPAUDevice() after initializing the CUDA runtime by
     * calling non-device management operations (allocating memory and
     * launching kernels are examples of non-device management operations).
     * This error can also be returned if using runtime/driver
     * interoperability and there is an existing ::CUcontext active on the
     * host thread.
     */
    cudaErrorSetOnActiveProcess           = 36,
  
    /**
     * This indicates that the surface passed to the API call is not a valid
     * surface.
     */
    cudaErrorInvalidSurface               = 37,
  
    /**
     * This indicates that no CUDA-capable devices were detected by the installed
     * CUDA driver.
     */
    cudaErrorNoDevice                     = 38,
  
    /**
     * This indicates that an uncorrectable ECC error was detected during
     * execution.
     */
    cudaErrorECCUncorrectable             = 39,
  
    /**
     * This indicates that a link to a shared object failed to resolve.
     */
    cudaErrorSharedObjectSymbolNotFound   = 40,
  
    /**
     * This indicates that initialization of a shared object failed.
     */
    cudaErrorSharedObjectInitFailed       = 41,
  
    /**
     * This indicates that the ::cudaLimit passed to the API call is not
     * supported by the active device.
     */
    cudaErrorUnsupportedLimit             = 42,
  
    /**
     * This indicates that multiple global or constant variables (across separate
     * CUDA source files in the application) share the same string name.
     */
    cudaErrorDuplicateVariableName        = 43,
  
    /**
     * This indicates that multiple textures (across separate CUDA source
     * files in the application) share the same string name.
     */
    cudaErrorDuplicateTextureName         = 44,
  
    /**
     * This indicates that multiple surfaces (across separate CUDA source
     * files in the application) share the same string name.
     */
    cudaErrorDuplicateSurfaceName         = 45,
  
    /**
     * This indicates that all CUDA devices are busy or unavailable at the current
     * time. Devices are often busy/unavailable due to use of
     * ::cudaComputeModeExclusive, ::cudaComputeModeProhibited or when long
     * running CUDA kernels have filled up the GPU and are blocking new work
     * from starting. They can also be unavailable due to memory constraints
     * on a device that already has active CUDA work being performed.
     */
    cudaErrorDevicesUnavailable           = 46,
  
    /**
     * This indicates that the device kernel image is invalid.
     */
    cudaErrorInvalidKernelImage           = 47,
  
    /**
     * This indicates that there is no kernel image available that is suitable
     * for the device. This can occur when a user specifies code generation
     * options for a particular CUDA source file that do not include the
     * corresponding device configuration.
     */
    cudaErrorNoKernelImageForDevice       = 48,
  
    /**
     * This indicates that the current context is not compatible with this
     * the CUDA Runtime. This can only occur if you are using CUDA
     * Runtime/Driver interoperability and have created an existing Driver
     * context using the driver API. The Driver context may be incompatible
     * either because the Driver context was created using an older version 
     * of the API, because the Runtime API call expects a primary driver 
     * context and the Driver context is not primary, or because the Driver 
     * context has been destroyed. Please see \ref CUDART_DRIVER "Interactions 
     * with the CUDA Driver API" for more information.
     */
    cudaErrorIncompatibleDriverContext    = 49,
      
    /**
     * This error indicates that a call to ::cudaDeviceEnablePeerAccess() is
     * trying to re-enable peer addressing on from a context which has already
     * had peer addressing enabled.
     */
    cudaErrorPeerAccessAlreadyEnabled     = 50,
    
    /**
     * This error indicates that ::cudaDeviceDisablePeerAccess() is trying to 
     * disable peer addressing which has not been enabled yet via 
     * ::cudaDeviceEnablePeerAccess().
     */
    cudaErrorPeerAccessNotEnabled         = 51,
    
    /**
     * This indicates that a call tried to access an exclusive-thread device that 
     * is already in use by a different thread.
     */
    cudaErrorDeviceAlreadyInUse           = 54,

    /**
     * This indicates profiler is not initialized for this run. This can
     * happen when the application is running with external profiling tools
     * like visual profiler.
     */
    cudaErrorProfilerDisabled             = 55,

    /**
     * \deprecated
     * This error return is deprecated as of CUDA 5.0. It is no longer an error
     * to attempt to enable/disable the profiling via ::cudaProfilerStart or
     * ::cudaProfilerStop without initialization.
     */
    cudaErrorProfilerNotInitialized       = 56,

    /**
     * \deprecated
     * This error return is deprecated as of CUDA 5.0. It is no longer an error
     * to call cudaProfilerStart() when profiling is already enabled.
     */
    cudaErrorProfilerAlreadyStarted       = 57,

    /**
     * \deprecated
     * This error return is deprecated as of CUDA 5.0. It is no longer an error
     * to call cudaProfilerStop() when profiling is already disabled.
     */
     cudaErrorProfilerAlreadyStopped       = 58,

    /**
     * An assert triggered in device code during kernel execution. The device
     * cannot be used again until ::cudaThreadExit() is called. All existing 
     * allocations are invalid and must be reconstructed if the program is to
     * continue using CUDA. 
     */
    cudaErrorAssert                        = 59,
  
    /**
     * This error indicates that the hardware resources required to enable
     * peer access have been exhausted for one or more of the devices 
     * passed to ::cudaEnablePeerAccess().
     */
    cudaErrorTooManyPeers                 = 60,
  
    /**
     * This error indicates that the memory range passed to ::cudaHostRegister()
     * has already been registered.
     */
    cudaErrorHostMemoryAlreadyRegistered  = 61,
        
    /**
     * This error indicates that the pointer passed to ::cudaHostUnregister()
     * does not correspond to any currently registered memory region.
     */
    cudaErrorHostMemoryNotRegistered      = 62,

    /**
     * This error indicates that an OS call failed.
     */
    cudaErrorOperatingSystem              = 63,

    /**
     * This error indicates that P2P access is not supported across the given
     * devices.
     */
    cudaErrorPeerAccessUnsupported        = 64,

    /**
     * This error indicates that a device runtime grid launch did not occur 
     * because the depth of the child grid would exceed the maximum supported
     * number of nested grid launches. 
     */
    cudaErrorLaunchMaxDepthExceeded       = 65,

    /**
     * This error indicates that a grid launch did not occur because the kernel 
     * uses file-scoped textures which are unsupported by the device runtime. 
     * Kernels launched via the device runtime only support textures created with 
     * the Texture Object API's.
     */
    cudaErrorLaunchFileScopedTex          = 66,

    /**
     * This error indicates that a grid launch did not occur because the kernel 
     * uses file-scoped surfaces which are unsupported by the device runtime.
     * Kernels launched via the device runtime only support surfaces created with
     * the Surface Object API's.
     */
    cudaErrorLaunchFileScopedSurf         = 67,

    /**
     * This error indicates that a call to ::cudaDeviceSynchronize made from
     * the device runtime failed because the call was made at grid depth greater
     * than than either the default (2 levels of grids) or user specified device 
     * limit ::cudaLimitDevRuntimeSyncDepth. To be able to synchronize on 
     * launched grids at a greater depth successfully, the maximum nested 
     * depth at which ::cudaDeviceSynchronize will be called must be specified 
     * with the ::cudaLimitDevRuntimeSyncDepth limit to the ::cudaDeviceSetLimit
     * api before the host-side launch of a kernel using the device runtime. 
     * Keep in mind that additional levels of sync depth require the runtime 
     * to reserve large amounts of device memory that cannot be used for 
     * user allocations.
     */
    cudaErrorSyncDepthExceeded            = 68,

    /**
     * This error indicates that a device runtime grid launch failed because
     * the launch would exceed the limit ::cudaLimitDevRuntimePendingLaunchCount.
     * For this launch to proceed successfully, ::cudaDeviceSetLimit must be
     * called to set the ::cudaLimitDevRuntimePendingLaunchCount to be higher 
     * than the upper bound of outstanding launches that can be issued to the
     * device runtime. Keep in mind that raising the limit of pending device
     * runtime launches will require the runtime to reserve device memory that
     * cannot be used for user allocations.
     */
    cudaErrorLaunchPendingCountExceeded   = 69,
    
    /**
     * This error indicates the attempted operation is not permitted.
     */
    cudaErrorNotPermitted                 = 70,

    /**
     * This error indicates the attempted operation is not supported
     * on the current system or device.
     */
    cudaErrorNotSupported                 = 71,

    /**
     * Device encountered an error in the call stack during kernel execution,
     * possibly due to stack corruption or exceeding the stack size limit.
     * The context cannot be used, so it must be destroyed (and a new one should be created).
     * All existing device memory allocations from this context are invalid
     * and must be reconstructed if the program is to continue using CUDA.
     */
    cudaErrorHardwareStackError           = 72,

    /**
     * The device encountered an illegal instruction during kernel execution
     * The context cannot be used, so it must be destroyed (and a new one should be created).
     * All existing device memory allocations from this context are invalid
     * and must be reconstructed if the program is to continue using CUDA.
     */
    cudaErrorIllegalInstruction           = 73,

    /**
     * The device encountered a load or store instruction
     * on a memory address which is not aligned.
     * The context cannot be used, so it must be destroyed (and a new one should be created).
     * All existing device memory allocations from this context are invalid
     * and must be reconstructed if the program is to continue using CUDA.
     */
    cudaErrorMisalignedAddress            = 74,

    /**
     * While executing a kernel, the device encountered an instruction
     * which can only operate on memory locations in certain address spaces
     * (global, shared, or local), but was supplied a memory address not
     * belonging to an allowed address space.
     * The context cannot be used, so it must be destroyed (and a new one should be created).
     * All existing device memory allocations from this context are invalid
     * and must be reconstructed if the program is to continue using CUDA.
     */
    cudaErrorInvalidAddressSpace          = 75,

    /**
     * The device encountered an invalid program counter.
     * The context cannot be used, so it must be destroyed (and a new one should be created).
     * All existing device memory allocations from this context are invalid
     * and must be reconstructed if the program is to continue using CUDA.
     */
    cudaErrorInvalidPc                    = 76,

    /**
     * The device encountered a load or store instruction on an invalid memory address.
     * The context cannot be used, so it must be destroyed (and a new one should be created).
     * All existing device memory allocations from this context are invalid
     * and must be reconstructed if the program is to continue using CUDA.
     */
    cudaErrorIllegalAddress               = 77,

    /**
     * A PTX compilation failed. The runtime may fall back to compiling PTX if
     * an application does not contain a suitable binary for the current device.
     */
    cudaErrorInvalidPtx                   = 78,

    /**
     * This indicates an error with the OpenGL or DirectX context.
     */
    cudaErrorInvalidGraphicsContext       = 79,


    /**
     * This indicates an internal startup failure in the CUDA runtime.
     */
    cudaErrorStartupFailure               =    0x7f,

    /**
     * Any unhandled CUDA driver error is added to this value and returned via
     * the runtime. Production releases of CUDA should not return such errors.
     * \deprecated
     * This error return is deprecated as of CUDA 4.1.
     */
    cudaErrorApiFailureBase               = 10000;

/**
 * Channel format kind
 */
/** enum cudaChannelFormatKind */
public static final int
    /** Signed channel format */
    cudaChannelFormatKindSigned           = 0,
    /** Unsigned channel format */
    cudaChannelFormatKindUnsigned         = 1,
    /** Float channel format */
    cudaChannelFormatKindFloat            = 2,
    /** No channel format */
    cudaChannelFormatKindNone             = 3;

/**
 * CUDA Channel format descriptor
 */
public static class cudaChannelFormatDesc extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public cudaChannelFormatDesc() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public cudaChannelFormatDesc(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public cudaChannelFormatDesc(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public cudaChannelFormatDesc position(int position) {
        return (cudaChannelFormatDesc)super.position(position);
    }

    /** x */
    public native int x(); public native cudaChannelFormatDesc x(int x);
    /** y */
    public native int y(); public native cudaChannelFormatDesc y(int y);
    /** z */
    public native int z(); public native cudaChannelFormatDesc z(int z);
    /** w */
    public native int w(); public native cudaChannelFormatDesc w(int w);
    /** Channel format kind */
    public native @Cast("cudaChannelFormatKind") int f(); public native cudaChannelFormatDesc f(int f);
}

/**
 * CUDA array
 */
@Opaque public static class cudaArray extends Pointer {
    /** Empty constructor. */
    public cudaArray() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public cudaArray(Pointer p) { super(p); }
}

/**
 * CUDA array (as source copy argument)
 */

/**
 * CUDA mipmapped array
 */
@Opaque public static class cudaMipmappedArray extends Pointer {
    /** Empty constructor. */
    public cudaMipmappedArray() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public cudaMipmappedArray(Pointer p) { super(p); }
}

/**
 * CUDA mipmapped array (as source argument)
 */

/**
 * CUDA memory types
 */
/** enum cudaMemoryType */
public static final int
    /** Host memory */
    cudaMemoryTypeHost   = 1,
    /** Device memory */
    cudaMemoryTypeDevice = 2;

/**
 * CUDA memory copy types
 */
/** enum cudaMemcpyKind */
public static final int
    /** Host   -> Host */
    cudaMemcpyHostToHost          = 0,
    /** Host   -> Device */
    cudaMemcpyHostToDevice        = 1,
    /** Device -> Host */
    cudaMemcpyDeviceToHost        = 2,
    /** Device -> Device */
    cudaMemcpyDeviceToDevice      = 3,
    /** Default based unified virtual address space */
    cudaMemcpyDefault             = 4;

/**
 * CUDA Pitched memory pointer
 *
 * \sa ::make_cudaPitchedPtr
 */
public static class cudaPitchedPtr extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public cudaPitchedPtr() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public cudaPitchedPtr(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public cudaPitchedPtr(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public cudaPitchedPtr position(int position) {
        return (cudaPitchedPtr)super.position(position);
    }

    /** Pointer to allocated memory */
    public native Pointer ptr(); public native cudaPitchedPtr ptr(Pointer ptr);
    /** Pitch of allocated memory in bytes */
    public native @Cast("size_t") long pitch(); public native cudaPitchedPtr pitch(long pitch);
    /** Logical width of allocation in elements */
    public native @Cast("size_t") long xsize(); public native cudaPitchedPtr xsize(long xsize);
    /** Logical height of allocation in elements */
    public native @Cast("size_t") long ysize(); public native cudaPitchedPtr ysize(long ysize);
}

/**
 * CUDA extent
 *
 * \sa ::make_cudaExtent
 */
public static class cudaExtent extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public cudaExtent() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public cudaExtent(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public cudaExtent(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public cudaExtent position(int position) {
        return (cudaExtent)super.position(position);
    }

    /** Width in elements when referring to array memory, in bytes when referring to linear memory */
    public native @Cast("size_t") long width(); public native cudaExtent width(long width);
    /** Height in elements */
    public native @Cast("size_t") long height(); public native cudaExtent height(long height);
    /** Depth in elements */
    public native @Cast("size_t") long depth(); public native cudaExtent depth(long depth);
}

/**
 * CUDA 3D position
 *
 * \sa ::make_cudaPos
 */
public static class cudaPos extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public cudaPos() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public cudaPos(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public cudaPos(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public cudaPos position(int position) {
        return (cudaPos)super.position(position);
    }

    /** x */
    public native @Cast("size_t") long x(); public native cudaPos x(long x);
    /** y */
    public native @Cast("size_t") long y(); public native cudaPos y(long y);
    /** z */
    public native @Cast("size_t") long z(); public native cudaPos z(long z);
}

/**
 * CUDA 3D memory copying parameters
 */
public static class cudaMemcpy3DParms extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public cudaMemcpy3DParms() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public cudaMemcpy3DParms(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public cudaMemcpy3DParms(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public cudaMemcpy3DParms position(int position) {
        return (cudaMemcpy3DParms)super.position(position);
    }

    /** Source memory address */
    public native cudaArray srcArray(); public native cudaMemcpy3DParms srcArray(cudaArray srcArray);
    /** Source position offset */
    public native @ByRef cudaPos srcPos(); public native cudaMemcpy3DParms srcPos(cudaPos srcPos);
    /** Pitched source memory address */
    public native @ByRef cudaPitchedPtr srcPtr(); public native cudaMemcpy3DParms srcPtr(cudaPitchedPtr srcPtr);
  
    /** Destination memory address */
    public native cudaArray dstArray(); public native cudaMemcpy3DParms dstArray(cudaArray dstArray);
    /** Destination position offset */
    public native @ByRef cudaPos dstPos(); public native cudaMemcpy3DParms dstPos(cudaPos dstPos);
    /** Pitched destination memory address */
    public native @ByRef cudaPitchedPtr dstPtr(); public native cudaMemcpy3DParms dstPtr(cudaPitchedPtr dstPtr);
  
    /** Requested memory copy size */
    public native @ByRef cudaExtent extent(); public native cudaMemcpy3DParms extent(cudaExtent extent);
    /** Type of transfer */
    public native @Cast("cudaMemcpyKind") int kind(); public native cudaMemcpy3DParms kind(int kind);
}

/**
 * CUDA 3D cross-device memory copying parameters
 */
public static class cudaMemcpy3DPeerParms extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public cudaMemcpy3DPeerParms() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public cudaMemcpy3DPeerParms(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public cudaMemcpy3DPeerParms(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public cudaMemcpy3DPeerParms position(int position) {
        return (cudaMemcpy3DPeerParms)super.position(position);
    }

    /** Source memory address */
    public native cudaArray srcArray(); public native cudaMemcpy3DPeerParms srcArray(cudaArray srcArray);
    /** Source position offset */
    public native @ByRef cudaPos srcPos(); public native cudaMemcpy3DPeerParms srcPos(cudaPos srcPos);
    /** Pitched source memory address */
    public native @ByRef cudaPitchedPtr srcPtr(); public native cudaMemcpy3DPeerParms srcPtr(cudaPitchedPtr srcPtr);
    /** Source device */
    public native int srcDevice(); public native cudaMemcpy3DPeerParms srcDevice(int srcDevice);
  
    /** Destination memory address */
    public native cudaArray dstArray(); public native cudaMemcpy3DPeerParms dstArray(cudaArray dstArray);
    /** Destination position offset */
    public native @ByRef cudaPos dstPos(); public native cudaMemcpy3DPeerParms dstPos(cudaPos dstPos);
    /** Pitched destination memory address */
    public native @ByRef cudaPitchedPtr dstPtr(); public native cudaMemcpy3DPeerParms dstPtr(cudaPitchedPtr dstPtr);
    /** Destination device */
    public native int dstDevice(); public native cudaMemcpy3DPeerParms dstDevice(int dstDevice);
  
    /** Requested memory copy size */
    public native @ByRef cudaExtent extent(); public native cudaMemcpy3DPeerParms extent(cudaExtent extent);
}

/**
 * CUDA graphics interop resource
 */
@Opaque public static class cudaGraphicsResource extends Pointer {
    /** Empty constructor. */
    public cudaGraphicsResource() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public cudaGraphicsResource(Pointer p) { super(p); }
}

/**
 * CUDA graphics interop register flags
 */
/** enum cudaGraphicsRegisterFlags */
public static final int
    /** Default */
    cudaGraphicsRegisterFlagsNone             = 0,
    /** CUDA will not write to this resource */
    cudaGraphicsRegisterFlagsReadOnly         = 1, 
    /** CUDA will only write to and will not read from this resource */
    cudaGraphicsRegisterFlagsWriteDiscard     = 2,
    /** CUDA will bind this resource to a surface reference */
    cudaGraphicsRegisterFlagsSurfaceLoadStore = 4,
    /** CUDA will perform texture gather operations on this resource */
    cudaGraphicsRegisterFlagsTextureGather    = 8;

/**
 * CUDA graphics interop map flags
 */
/** enum cudaGraphicsMapFlags */
public static final int
    /** Default; Assume resource can be read/written */
    cudaGraphicsMapFlagsNone         = 0,
    /** CUDA will not write to this resource */
    cudaGraphicsMapFlagsReadOnly     = 1,
    /** CUDA will only write to and will not read from this resource */
    cudaGraphicsMapFlagsWriteDiscard = 2;

/**
 * CUDA graphics interop array indices for cube maps
 */
/** enum cudaGraphicsCubeFace */
public static final int
    /** Positive X face of cubemap */
    cudaGraphicsCubeFacePositiveX =  0x00,
    /** Negative X face of cubemap */
    cudaGraphicsCubeFaceNegativeX =  0x01,
    /** Positive Y face of cubemap */
    cudaGraphicsCubeFacePositiveY =  0x02,
    /** Negative Y face of cubemap */
    cudaGraphicsCubeFaceNegativeY =  0x03,
    /** Positive Z face of cubemap */
    cudaGraphicsCubeFacePositiveZ =  0x04,
    /** Negative Z face of cubemap */
    cudaGraphicsCubeFaceNegativeZ =  0x05;

/**
 * CUDA resource types
 */
/** enum cudaResourceType */
public static final int
    /** Array resource */
    cudaResourceTypeArray          =  0x00,
    /** Mipmapped array resource */
    cudaResourceTypeMipmappedArray =  0x01,
    /** Linear resource */
    cudaResourceTypeLinear         =  0x02,
    /** Pitch 2D resource */
    cudaResourceTypePitch2D        =  0x03;

/**
 * CUDA texture resource view formats
 */
/** enum cudaResourceViewFormat */
public static final int
    /** No resource view format (use underlying resource format) */
    cudaResViewFormatNone                      =  0x00,
    /** 1 channel unsigned 8-bit integers */
    cudaResViewFormatUnsignedChar1             =  0x01,
    /** 2 channel unsigned 8-bit integers */
    cudaResViewFormatUnsignedChar2             =  0x02,
    /** 4 channel unsigned 8-bit integers */
    cudaResViewFormatUnsignedChar4             =  0x03,
    /** 1 channel signed 8-bit integers */
    cudaResViewFormatSignedChar1               =  0x04,
    /** 2 channel signed 8-bit integers */
    cudaResViewFormatSignedChar2               =  0x05,
    /** 4 channel signed 8-bit integers */
    cudaResViewFormatSignedChar4               =  0x06,
    /** 1 channel unsigned 16-bit integers */
    cudaResViewFormatUnsignedShort1            =  0x07,
    /** 2 channel unsigned 16-bit integers */
    cudaResViewFormatUnsignedShort2            =  0x08,
    /** 4 channel unsigned 16-bit integers */
    cudaResViewFormatUnsignedShort4            =  0x09,
    /** 1 channel signed 16-bit integers */
    cudaResViewFormatSignedShort1              =  0x0a,
    /** 2 channel signed 16-bit integers */
    cudaResViewFormatSignedShort2              =  0x0b,
    /** 4 channel signed 16-bit integers */
    cudaResViewFormatSignedShort4              =  0x0c,
    /** 1 channel unsigned 32-bit integers */
    cudaResViewFormatUnsignedInt1              =  0x0d,
    /** 2 channel unsigned 32-bit integers */
    cudaResViewFormatUnsignedInt2              =  0x0e,
    /** 4 channel unsigned 32-bit integers */
    cudaResViewFormatUnsignedInt4              =  0x0f,
    /** 1 channel signed 32-bit integers */
    cudaResViewFormatSignedInt1                =  0x10,
    /** 2 channel signed 32-bit integers */
    cudaResViewFormatSignedInt2                =  0x11,
    /** 4 channel signed 32-bit integers */
    cudaResViewFormatSignedInt4                =  0x12,
    /** 1 channel 16-bit floating point */
    cudaResViewFormatHalf1                     =  0x13,
    /** 2 channel 16-bit floating point */
    cudaResViewFormatHalf2                     =  0x14,
    /** 4 channel 16-bit floating point */
    cudaResViewFormatHalf4                     =  0x15,
    /** 1 channel 32-bit floating point */
    cudaResViewFormatFloat1                    =  0x16,
    /** 2 channel 32-bit floating point */
    cudaResViewFormatFloat2                    =  0x17,
    /** 4 channel 32-bit floating point */
    cudaResViewFormatFloat4                    =  0x18,
    /** Block compressed 1 */
    cudaResViewFormatUnsignedBlockCompressed1  =  0x19,
    /** Block compressed 2 */
    cudaResViewFormatUnsignedBlockCompressed2  =  0x1a,
    /** Block compressed 3 */
    cudaResViewFormatUnsignedBlockCompressed3  =  0x1b,
    /** Block compressed 4 unsigned */
    cudaResViewFormatUnsignedBlockCompressed4  =  0x1c,
    /** Block compressed 4 signed */
    cudaResViewFormatSignedBlockCompressed4    =  0x1d,
    /** Block compressed 5 unsigned */
    cudaResViewFormatUnsignedBlockCompressed5  =  0x1e,
    /** Block compressed 5 signed */
    cudaResViewFormatSignedBlockCompressed5    =  0x1f,
    /** Block compressed 6 unsigned half-float */
    cudaResViewFormatUnsignedBlockCompressed6H =  0x20,
    /** Block compressed 6 signed half-float */
    cudaResViewFormatSignedBlockCompressed6H   =  0x21,
    /** Block compressed 7 */
    cudaResViewFormatUnsignedBlockCompressed7  =  0x22;

/**
 * CUDA resource descriptor
 */
public static class cudaResourceDesc extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public cudaResourceDesc() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public cudaResourceDesc(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public cudaResourceDesc(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public cudaResourceDesc position(int position) {
        return (cudaResourceDesc)super.position(position);
    }

	/** Resource type */
	public native @Cast("cudaResourceType") int resType(); public native cudaResourceDesc resType(int resType);
	
			/** CUDA array */
			@Name("res.array.array") public native cudaArray res_array_array(); public native cudaResourceDesc res_array_array(cudaArray res_array_array);
            /** CUDA mipmapped array */
            @Name("res.mipmap.mipmap") public native cudaMipmappedArray res_mipmap_mipmap(); public native cudaResourceDesc res_mipmap_mipmap(cudaMipmappedArray res_mipmap_mipmap);
			/** Device pointer */
			@Name("res.linear.devPtr") public native Pointer res_linear_devPtr(); public native cudaResourceDesc res_linear_devPtr(Pointer res_linear_devPtr);
			/** Channel descriptor */
			@Name("res.linear.desc") public native @ByRef cudaChannelFormatDesc res_linear_desc(); public native cudaResourceDesc res_linear_desc(cudaChannelFormatDesc res_linear_desc);
			/** Size in bytes */
			@Name("res.linear.sizeInBytes") public native @Cast("size_t") long res_linear_sizeInBytes(); public native cudaResourceDesc res_linear_sizeInBytes(long res_linear_sizeInBytes);
			/** Device pointer */
			@Name("res.pitch2D.devPtr") public native Pointer res_pitch2D_devPtr(); public native cudaResourceDesc res_pitch2D_devPtr(Pointer res_pitch2D_devPtr);
			/** Channel descriptor */
			@Name("res.pitch2D.desc") public native @ByRef cudaChannelFormatDesc res_pitch2D_desc(); public native cudaResourceDesc res_pitch2D_desc(cudaChannelFormatDesc res_pitch2D_desc);
			/** Width of the array in elements */
			@Name("res.pitch2D.width") public native @Cast("size_t") long res_pitch2D_width(); public native cudaResourceDesc res_pitch2D_width(long res_pitch2D_width);
			/** Height of the array in elements */
			@Name("res.pitch2D.height") public native @Cast("size_t") long res_pitch2D_height(); public native cudaResourceDesc res_pitch2D_height(long res_pitch2D_height);
			/** Pitch between two rows in bytes */
			@Name("res.pitch2D.pitchInBytes") public native @Cast("size_t") long res_pitch2D_pitchInBytes(); public native cudaResourceDesc res_pitch2D_pitchInBytes(long res_pitch2D_pitchInBytes);
}

/**
 * CUDA resource view descriptor
 */
public static class cudaResourceViewDesc extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public cudaResourceViewDesc() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public cudaResourceViewDesc(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public cudaResourceViewDesc(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public cudaResourceViewDesc position(int position) {
        return (cudaResourceViewDesc)super.position(position);
    }

    /** Resource view format */
    public native @Cast("cudaResourceViewFormat") int format(); public native cudaResourceViewDesc format(int format);
    /** Width of the resource view */
    public native @Cast("size_t") long width(); public native cudaResourceViewDesc width(long width);
    /** Height of the resource view */
    public native @Cast("size_t") long height(); public native cudaResourceViewDesc height(long height);
    /** Depth of the resource view */
    public native @Cast("size_t") long depth(); public native cudaResourceViewDesc depth(long depth);
    /** First defined mipmap level */
    public native @Cast("unsigned int") int firstMipmapLevel(); public native cudaResourceViewDesc firstMipmapLevel(int firstMipmapLevel);
    /** Last defined mipmap level */
    public native @Cast("unsigned int") int lastMipmapLevel(); public native cudaResourceViewDesc lastMipmapLevel(int lastMipmapLevel);
    /** First layer index */
    public native @Cast("unsigned int") int firstLayer(); public native cudaResourceViewDesc firstLayer(int firstLayer);
    /** Last layer index */
    public native @Cast("unsigned int") int lastLayer(); public native cudaResourceViewDesc lastLayer(int lastLayer);
}

/**
 * CUDA pointer attributes
 */
public static class cudaPointerAttributes extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public cudaPointerAttributes() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public cudaPointerAttributes(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public cudaPointerAttributes(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public cudaPointerAttributes position(int position) {
        return (cudaPointerAttributes)super.position(position);
    }

    /**
     * The physical location of the memory, ::cudaMemoryTypeHost or 
     * ::cudaMemoryTypeDevice.
     */
    public native @Cast("cudaMemoryType") int memoryType(); public native cudaPointerAttributes memoryType(int memoryType);

    /** 
     * The device against which the memory was allocated or registered.
     * If the memory type is ::cudaMemoryTypeDevice then this identifies 
     * the device on which the memory referred physically resides.  If
     * the memory type is ::cudaMemoryTypeHost then this identifies the 
     * device which was current when the memory was allocated or registered
     * (and if that device is deinitialized then this allocation will vanish
     * with that device's state).
     */
    public native int device(); public native cudaPointerAttributes device(int device);

    /**
     * The address which may be dereferenced on the current device to access 
     * the memory or NULL if no such address exists.
     */
    public native Pointer devicePointer(); public native cudaPointerAttributes devicePointer(Pointer devicePointer);

    /**
     * The address which may be dereferenced on the host to access the
     * memory or NULL if no such address exists.
     */
    public native Pointer hostPointer(); public native cudaPointerAttributes hostPointer(Pointer hostPointer);

    /**
     * Indicates if this pointer points to managed memory
     */
    public native int isManaged(); public native cudaPointerAttributes isManaged(int isManaged);
}

/**
 * CUDA function attributes
 */
public static class cudaFuncAttributes extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public cudaFuncAttributes() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public cudaFuncAttributes(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public cudaFuncAttributes(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public cudaFuncAttributes position(int position) {
        return (cudaFuncAttributes)super.position(position);
    }

   /**
    * The size in bytes of statically-allocated shared memory per block
    * required by this function. This does not include dynamically-allocated
    * shared memory requested by the user at runtime.
    */
   public native @Cast("size_t") long sharedSizeBytes(); public native cudaFuncAttributes sharedSizeBytes(long sharedSizeBytes);

   /**
    * The size in bytes of user-allocated constant memory required by this
    * function.
    */
   public native @Cast("size_t") long constSizeBytes(); public native cudaFuncAttributes constSizeBytes(long constSizeBytes);

   /**
    * The size in bytes of local memory used by each thread of this function.
    */
   public native @Cast("size_t") long localSizeBytes(); public native cudaFuncAttributes localSizeBytes(long localSizeBytes);

   /**
    * The maximum number of threads per block, beyond which a launch of the
    * function would fail. This number depends on both the function and the
    * device on which the function is currently loaded.
    */
   public native int maxThreadsPerBlock(); public native cudaFuncAttributes maxThreadsPerBlock(int maxThreadsPerBlock);

   /**
    * The number of registers used by each thread of this function.
    */
   public native int numRegs(); public native cudaFuncAttributes numRegs(int numRegs);

   /**
    * The PTX virtual architecture version for which the function was
    * compiled. This value is the major PTX version * 10 + the minor PTX
    * version, so a PTX version 1.3 function would return the value 13.
    */
   public native int ptxVersion(); public native cudaFuncAttributes ptxVersion(int ptxVersion);

   /**
    * The binary architecture version for which the function was compiled.
    * This value is the major binary version * 10 + the minor binary version,
    * so a binary version 1.3 function would return the value 13.
    */
   public native int binaryVersion(); public native cudaFuncAttributes binaryVersion(int binaryVersion);

   /**
    * The attribute to indicate whether the function has been compiled with 
    * user specified option "-Xptxas --dlcm=ca" set.
    */
   public native int cacheModeCA(); public native cudaFuncAttributes cacheModeCA(int cacheModeCA);
}

/**
 * CUDA function cache configurations
 */
/** enum cudaFuncCache */
public static final int
    /** Default function cache configuration, no preference */
    cudaFuncCachePreferNone   = 0,
    /** Prefer larger shared memory and smaller L1 cache  */
    cudaFuncCachePreferShared = 1,
    /** Prefer larger L1 cache and smaller shared memory */
    cudaFuncCachePreferL1     = 2,
    /** Prefer equal size L1 cache and shared memory */
    cudaFuncCachePreferEqual  = 3;

/**
 * CUDA shared memory configuration
 */

/** enum cudaSharedMemConfig */
public static final int
    cudaSharedMemBankSizeDefault   = 0,
    cudaSharedMemBankSizeFourByte  = 1,
    cudaSharedMemBankSizeEightByte = 2;

/**
 * CUDA device compute modes
 */
/** enum cudaComputeMode */
public static final int
    /** Default compute mode (Multiple threads can use ::cudaSetDevice() with this device) */
    cudaComputeModeDefault          = 0,
    /** Compute-exclusive-thread mode (Only one thread in one process will be able to use ::cudaSetDevice() with this device) */
    cudaComputeModeExclusive        = 1,
    /** Compute-prohibited mode (No threads can use ::cudaSetDevice() with this device) */
    cudaComputeModeProhibited       = 2,
    /** Compute-exclusive-process mode (Many threads in one process will be able to use ::cudaSetDevice() with this device) */
    cudaComputeModeExclusiveProcess = 3;

/**
 * CUDA Limits
 */
/** enum cudaLimit */
public static final int
    /** GPU thread stack size */
    cudaLimitStackSize                    =  0x00,
    /** GPU printf/fprintf FIFO size */
    cudaLimitPrintfFifoSize               =  0x01,
    /** GPU malloc heap size */
    cudaLimitMallocHeapSize               =  0x02,
    /** GPU device runtime synchronize depth */
    cudaLimitDevRuntimeSyncDepth          =  0x03,
    /** GPU device runtime pending launch count */
    cudaLimitDevRuntimePendingLaunchCount =  0x04;

/**
 * CUDA Profiler Output modes
 */
/** enum cudaOutputMode */
public static final int
    /** Output mode Key-Value pair format. */
    cudaKeyValuePair    =  0x00,
    /** Output mode Comma separated values format. */
    cudaCSV             =  0x01;

/**
 * CUDA device attributes
 */
/** enum cudaDeviceAttr */
public static final int
    /** Maximum number of threads per block */
    cudaDevAttrMaxThreadsPerBlock             = 1,
    /** Maximum block dimension X */
    cudaDevAttrMaxBlockDimX                   = 2,
    /** Maximum block dimension Y */
    cudaDevAttrMaxBlockDimY                   = 3,
    /** Maximum block dimension Z */
    cudaDevAttrMaxBlockDimZ                   = 4,
    /** Maximum grid dimension X */
    cudaDevAttrMaxGridDimX                    = 5,
    /** Maximum grid dimension Y */
    cudaDevAttrMaxGridDimY                    = 6,
    /** Maximum grid dimension Z */
    cudaDevAttrMaxGridDimZ                    = 7,
    /** Maximum shared memory available per block in bytes */
    cudaDevAttrMaxSharedMemoryPerBlock        = 8,
    /** Memory available on device for __constant__ variables in a CUDA C kernel in bytes */
    cudaDevAttrTotalConstantMemory            = 9,
    /** Warp size in threads */
    cudaDevAttrWarpSize                       = 10,
    /** Maximum pitch in bytes allowed by memory copies */
    cudaDevAttrMaxPitch                       = 11,
    /** Maximum number of 32-bit registers available per block */
    cudaDevAttrMaxRegistersPerBlock           = 12,
    /** Peak clock frequency in kilohertz */
    cudaDevAttrClockRate                      = 13,
    /** Alignment requirement for textures */
    cudaDevAttrTextureAlignment               = 14,
    /** Device can possibly copy memory and execute a kernel concurrently */
    cudaDevAttrGpuOverlap                     = 15,
    /** Number of multiprocessors on device */
    cudaDevAttrMultiProcessorCount            = 16,
    /** Specifies whether there is a run time limit on kernels */
    cudaDevAttrKernelExecTimeout              = 17,
    /** Device is integrated with host memory */
    cudaDevAttrIntegrated                     = 18,
    /** Device can map host memory into CUDA address space */
    cudaDevAttrCanMapHostMemory               = 19,
    /** Compute mode (See ::cudaComputeMode for details) */
    cudaDevAttrComputeMode                    = 20,
    /** Maximum 1D texture width */
    cudaDevAttrMaxTexture1DWidth              = 21,
    /** Maximum 2D texture width */
    cudaDevAttrMaxTexture2DWidth              = 22,
    /** Maximum 2D texture height */
    cudaDevAttrMaxTexture2DHeight             = 23,
    /** Maximum 3D texture width */
    cudaDevAttrMaxTexture3DWidth              = 24,
    /** Maximum 3D texture height */
    cudaDevAttrMaxTexture3DHeight             = 25,
    /** Maximum 3D texture depth */
    cudaDevAttrMaxTexture3DDepth              = 26,
    /** Maximum 2D layered texture width */
    cudaDevAttrMaxTexture2DLayeredWidth       = 27,
    /** Maximum 2D layered texture height */
    cudaDevAttrMaxTexture2DLayeredHeight      = 28,
    /** Maximum layers in a 2D layered texture */
    cudaDevAttrMaxTexture2DLayeredLayers      = 29,
    /** Alignment requirement for surfaces */
    cudaDevAttrSurfaceAlignment               = 30,
    /** Device can possibly execute multiple kernels concurrently */
    cudaDevAttrConcurrentKernels              = 31,
    /** Device has ECC support enabled */
    cudaDevAttrEccEnabled                     = 32,
    /** PCI bus ID of the device */
    cudaDevAttrPciBusId                       = 33,
    /** PCI device ID of the device */
    cudaDevAttrPciDeviceId                    = 34,
    /** Device is using TCC driver model */
    cudaDevAttrTccDriver                      = 35,
    /** Peak memory clock frequency in kilohertz */
    cudaDevAttrMemoryClockRate                = 36,
    /** Global memory bus width in bits */
    cudaDevAttrGlobalMemoryBusWidth           = 37,
    /** Size of L2 cache in bytes */
    cudaDevAttrL2CacheSize                    = 38,
    /** Maximum resident threads per multiprocessor */
    cudaDevAttrMaxThreadsPerMultiProcessor    = 39,
    /** Number of asynchronous engines */
    cudaDevAttrAsyncEngineCount               = 40,
    /** Device shares a unified address space with the host */
    cudaDevAttrUnifiedAddressing              = 41,    
    /** Maximum 1D layered texture width */
    cudaDevAttrMaxTexture1DLayeredWidth       = 42,
    /** Maximum layers in a 1D layered texture */
    cudaDevAttrMaxTexture1DLayeredLayers      = 43,
    /** Maximum 2D texture width if cudaArrayTextureGather is set */
    cudaDevAttrMaxTexture2DGatherWidth        = 45,
    /** Maximum 2D texture height if cudaArrayTextureGather is set */
    cudaDevAttrMaxTexture2DGatherHeight       = 46,
    /** Alternate maximum 3D texture width */
    cudaDevAttrMaxTexture3DWidthAlt           = 47,
    /** Alternate maximum 3D texture height */
    cudaDevAttrMaxTexture3DHeightAlt          = 48,
    /** Alternate maximum 3D texture depth */
    cudaDevAttrMaxTexture3DDepthAlt           = 49,
    /** PCI domain ID of the device */
    cudaDevAttrPciDomainId                    = 50,
    /** Pitch alignment requirement for textures */
    cudaDevAttrTexturePitchAlignment          = 51,
    /** Maximum cubemap texture width/height */
    cudaDevAttrMaxTextureCubemapWidth         = 52,
    /** Maximum cubemap layered texture width/height */
    cudaDevAttrMaxTextureCubemapLayeredWidth  = 53,
    /** Maximum layers in a cubemap layered texture */
    cudaDevAttrMaxTextureCubemapLayeredLayers = 54,
    /** Maximum 1D surface width */
    cudaDevAttrMaxSurface1DWidth              = 55,
    /** Maximum 2D surface width */
    cudaDevAttrMaxSurface2DWidth              = 56,
    /** Maximum 2D surface height */
    cudaDevAttrMaxSurface2DHeight             = 57,
    /** Maximum 3D surface width */
    cudaDevAttrMaxSurface3DWidth              = 58,
    /** Maximum 3D surface height */
    cudaDevAttrMaxSurface3DHeight             = 59,
    /** Maximum 3D surface depth */
    cudaDevAttrMaxSurface3DDepth              = 60,
    /** Maximum 1D layered surface width */
    cudaDevAttrMaxSurface1DLayeredWidth       = 61,
    /** Maximum layers in a 1D layered surface */
    cudaDevAttrMaxSurface1DLayeredLayers      = 62,
    /** Maximum 2D layered surface width */
    cudaDevAttrMaxSurface2DLayeredWidth       = 63,
    /** Maximum 2D layered surface height */
    cudaDevAttrMaxSurface2DLayeredHeight      = 64,
    /** Maximum layers in a 2D layered surface */
    cudaDevAttrMaxSurface2DLayeredLayers      = 65,
    /** Maximum cubemap surface width */
    cudaDevAttrMaxSurfaceCubemapWidth         = 66,
    /** Maximum cubemap layered surface width */
    cudaDevAttrMaxSurfaceCubemapLayeredWidth  = 67,
    /** Maximum layers in a cubemap layered surface */
    cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68,
    /** Maximum 1D linear texture width */
    cudaDevAttrMaxTexture1DLinearWidth        = 69,
    /** Maximum 2D linear texture width */
    cudaDevAttrMaxTexture2DLinearWidth        = 70,
    /** Maximum 2D linear texture height */
    cudaDevAttrMaxTexture2DLinearHeight       = 71,
    /** Maximum 2D linear texture pitch in bytes */
    cudaDevAttrMaxTexture2DLinearPitch        = 72,
    /** Maximum mipmapped 2D texture width */
    cudaDevAttrMaxTexture2DMipmappedWidth     = 73,
    /** Maximum mipmapped 2D texture height */
    cudaDevAttrMaxTexture2DMipmappedHeight    = 74,
    /** Major compute capability version number */
    cudaDevAttrComputeCapabilityMajor         = 75, 
    /** Minor compute capability version number */
    cudaDevAttrComputeCapabilityMinor         = 76,
    /** Maximum mipmapped 1D texture width */
    cudaDevAttrMaxTexture1DMipmappedWidth     = 77,
    /** Device supports stream priorities */
    cudaDevAttrStreamPrioritiesSupported      = 78,
    /** Device supports caching globals in L1 */
    cudaDevAttrGlobalL1CacheSupported         = 79,
    /** Device supports caching locals in L1 */
    cudaDevAttrLocalL1CacheSupported          = 80,
    /** Maximum shared memory available per multiprocessor in bytes */
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
    /** Maximum number of 32-bit registers available per multiprocessor */
    cudaDevAttrMaxRegistersPerMultiprocessor  = 82,
    /** Device can allocate managed memory on this system */
    cudaDevAttrManagedMemory                  = 83,
    /** Device is on a multi-GPU board */
    cudaDevAttrIsMultiGpuBoard                = 84,
    /** Unique identifier for a group of devices on the same multi-GPU board */
    cudaDevAttrMultiGpuBoardGroupID           = 85;

/**
 * CUDA device properties
 */
public static class cudaDeviceProp extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public cudaDeviceProp() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public cudaDeviceProp(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public cudaDeviceProp(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public cudaDeviceProp position(int position) {
        return (cudaDeviceProp)super.position(position);
    }

    /** ASCII string identifying device */
    public native @Cast("char") byte name(int i); public native cudaDeviceProp name(int i, byte name);
    @MemberGetter public native @Cast("char*") BytePointer name();
    /** Global memory available on device in bytes */
    public native @Cast("size_t") long totalGlobalMem(); public native cudaDeviceProp totalGlobalMem(long totalGlobalMem);
    /** Shared memory available per block in bytes */
    public native @Cast("size_t") long sharedMemPerBlock(); public native cudaDeviceProp sharedMemPerBlock(long sharedMemPerBlock);
    /** 32-bit registers available per block */
    public native int regsPerBlock(); public native cudaDeviceProp regsPerBlock(int regsPerBlock);
    /** Warp size in threads */
    public native int warpSize(); public native cudaDeviceProp warpSize(int warpSize);
    /** Maximum pitch in bytes allowed by memory copies */
    public native @Cast("size_t") long memPitch(); public native cudaDeviceProp memPitch(long memPitch);
    /** Maximum number of threads per block */
    public native int maxThreadsPerBlock(); public native cudaDeviceProp maxThreadsPerBlock(int maxThreadsPerBlock);
    /** Maximum size of each dimension of a block */
    public native int maxThreadsDim(int i); public native cudaDeviceProp maxThreadsDim(int i, int maxThreadsDim);
    @MemberGetter public native IntPointer maxThreadsDim();
    /** Maximum size of each dimension of a grid */
    public native int maxGridSize(int i); public native cudaDeviceProp maxGridSize(int i, int maxGridSize);
    @MemberGetter public native IntPointer maxGridSize();
    /** Clock frequency in kilohertz */
    public native int clockRate(); public native cudaDeviceProp clockRate(int clockRate);
    /** Constant memory available on device in bytes */
    public native @Cast("size_t") long totalConstMem(); public native cudaDeviceProp totalConstMem(long totalConstMem);
    /** Major compute capability */
    public native int major(); public native cudaDeviceProp major(int major);
    /** Minor compute capability */
    public native int minor(); public native cudaDeviceProp minor(int minor);
    /** Alignment requirement for textures */
    public native @Cast("size_t") long textureAlignment(); public native cudaDeviceProp textureAlignment(long textureAlignment);
    /** Pitch alignment requirement for texture references bound to pitched memory */
    public native @Cast("size_t") long texturePitchAlignment(); public native cudaDeviceProp texturePitchAlignment(long texturePitchAlignment);
    /** Device can concurrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount. */
    public native int deviceOverlap(); public native cudaDeviceProp deviceOverlap(int deviceOverlap);
    /** Number of multiprocessors on device */
    public native int multiProcessorCount(); public native cudaDeviceProp multiProcessorCount(int multiProcessorCount);
    /** Specified whether there is a run time limit on kernels */
    public native int kernelExecTimeoutEnabled(); public native cudaDeviceProp kernelExecTimeoutEnabled(int kernelExecTimeoutEnabled);
    /** Device is integrated as opposed to discrete */
    public native int integrated(); public native cudaDeviceProp integrated(int integrated);
    /** Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer */
    public native int canMapHostMemory(); public native cudaDeviceProp canMapHostMemory(int canMapHostMemory);
    /** Compute mode (See ::cudaComputeMode) */
    public native int computeMode(); public native cudaDeviceProp computeMode(int computeMode);
    /** Maximum 1D texture size */
    public native int maxTexture1D(); public native cudaDeviceProp maxTexture1D(int maxTexture1D);
    /** Maximum 1D mipmapped texture size */
    public native int maxTexture1DMipmap(); public native cudaDeviceProp maxTexture1DMipmap(int maxTexture1DMipmap);
    /** Maximum size for 1D textures bound to linear memory */
    public native int maxTexture1DLinear(); public native cudaDeviceProp maxTexture1DLinear(int maxTexture1DLinear);
    /** Maximum 2D texture dimensions */
    public native int maxTexture2D(int i); public native cudaDeviceProp maxTexture2D(int i, int maxTexture2D);
    @MemberGetter public native IntPointer maxTexture2D();
    /** Maximum 2D mipmapped texture dimensions */
    public native int maxTexture2DMipmap(int i); public native cudaDeviceProp maxTexture2DMipmap(int i, int maxTexture2DMipmap);
    @MemberGetter public native IntPointer maxTexture2DMipmap();
    /** Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory */
    public native int maxTexture2DLinear(int i); public native cudaDeviceProp maxTexture2DLinear(int i, int maxTexture2DLinear);
    @MemberGetter public native IntPointer maxTexture2DLinear();
    /** Maximum 2D texture dimensions if texture gather operations have to be performed */
    public native int maxTexture2DGather(int i); public native cudaDeviceProp maxTexture2DGather(int i, int maxTexture2DGather);
    @MemberGetter public native IntPointer maxTexture2DGather();
    /** Maximum 3D texture dimensions */
    public native int maxTexture3D(int i); public native cudaDeviceProp maxTexture3D(int i, int maxTexture3D);
    @MemberGetter public native IntPointer maxTexture3D();
    /** Maximum alternate 3D texture dimensions */
    public native int maxTexture3DAlt(int i); public native cudaDeviceProp maxTexture3DAlt(int i, int maxTexture3DAlt);
    @MemberGetter public native IntPointer maxTexture3DAlt();
    /** Maximum Cubemap texture dimensions */
    public native int maxTextureCubemap(); public native cudaDeviceProp maxTextureCubemap(int maxTextureCubemap);
    /** Maximum 1D layered texture dimensions */
    public native int maxTexture1DLayered(int i); public native cudaDeviceProp maxTexture1DLayered(int i, int maxTexture1DLayered);
    @MemberGetter public native IntPointer maxTexture1DLayered();
    /** Maximum 2D layered texture dimensions */
    public native int maxTexture2DLayered(int i); public native cudaDeviceProp maxTexture2DLayered(int i, int maxTexture2DLayered);
    @MemberGetter public native IntPointer maxTexture2DLayered();
    /** Maximum Cubemap layered texture dimensions */
    public native int maxTextureCubemapLayered(int i); public native cudaDeviceProp maxTextureCubemapLayered(int i, int maxTextureCubemapLayered);
    @MemberGetter public native IntPointer maxTextureCubemapLayered();
    /** Maximum 1D surface size */
    public native int maxSurface1D(); public native cudaDeviceProp maxSurface1D(int maxSurface1D);
    /** Maximum 2D surface dimensions */
    public native int maxSurface2D(int i); public native cudaDeviceProp maxSurface2D(int i, int maxSurface2D);
    @MemberGetter public native IntPointer maxSurface2D();
    /** Maximum 3D surface dimensions */
    public native int maxSurface3D(int i); public native cudaDeviceProp maxSurface3D(int i, int maxSurface3D);
    @MemberGetter public native IntPointer maxSurface3D();
    /** Maximum 1D layered surface dimensions */
    public native int maxSurface1DLayered(int i); public native cudaDeviceProp maxSurface1DLayered(int i, int maxSurface1DLayered);
    @MemberGetter public native IntPointer maxSurface1DLayered();
    /** Maximum 2D layered surface dimensions */
    public native int maxSurface2DLayered(int i); public native cudaDeviceProp maxSurface2DLayered(int i, int maxSurface2DLayered);
    @MemberGetter public native IntPointer maxSurface2DLayered();
    /** Maximum Cubemap surface dimensions */
    public native int maxSurfaceCubemap(); public native cudaDeviceProp maxSurfaceCubemap(int maxSurfaceCubemap);
    /** Maximum Cubemap layered surface dimensions */
    public native int maxSurfaceCubemapLayered(int i); public native cudaDeviceProp maxSurfaceCubemapLayered(int i, int maxSurfaceCubemapLayered);
    @MemberGetter public native IntPointer maxSurfaceCubemapLayered();
    /** Alignment requirements for surfaces */
    public native @Cast("size_t") long surfaceAlignment(); public native cudaDeviceProp surfaceAlignment(long surfaceAlignment);
    /** Device can possibly execute multiple kernels concurrently */
    public native int concurrentKernels(); public native cudaDeviceProp concurrentKernels(int concurrentKernels);
    /** Device has ECC support enabled */
    public native int ECCEnabled(); public native cudaDeviceProp ECCEnabled(int ECCEnabled);
    /** PCI bus ID of the device */
    public native int pciBusID(); public native cudaDeviceProp pciBusID(int pciBusID);
    /** PCI device ID of the device */
    public native int pciDeviceID(); public native cudaDeviceProp pciDeviceID(int pciDeviceID);
    /** PCI domain ID of the device */
    public native int pciDomainID(); public native cudaDeviceProp pciDomainID(int pciDomainID);
    /** 1 if device is a Tesla device using TCC driver, 0 otherwise */
    public native int tccDriver(); public native cudaDeviceProp tccDriver(int tccDriver);
    /** Number of asynchronous engines */
    public native int asyncEngineCount(); public native cudaDeviceProp asyncEngineCount(int asyncEngineCount);
    /** Device shares a unified address space with the host */
    public native int unifiedAddressing(); public native cudaDeviceProp unifiedAddressing(int unifiedAddressing);
    /** Peak memory clock frequency in kilohertz */
    public native int memoryClockRate(); public native cudaDeviceProp memoryClockRate(int memoryClockRate);
    /** Global memory bus width in bits */
    public native int memoryBusWidth(); public native cudaDeviceProp memoryBusWidth(int memoryBusWidth);
    /** Size of L2 cache in bytes */
    public native int l2CacheSize(); public native cudaDeviceProp l2CacheSize(int l2CacheSize);
    /** Maximum resident threads per multiprocessor */
    public native int maxThreadsPerMultiProcessor(); public native cudaDeviceProp maxThreadsPerMultiProcessor(int maxThreadsPerMultiProcessor);
    /** Device supports stream priorities */
    public native int streamPrioritiesSupported(); public native cudaDeviceProp streamPrioritiesSupported(int streamPrioritiesSupported);
    /** Device supports caching globals in L1 */
    public native int globalL1CacheSupported(); public native cudaDeviceProp globalL1CacheSupported(int globalL1CacheSupported);
    /** Device supports caching locals in L1 */
    public native int localL1CacheSupported(); public native cudaDeviceProp localL1CacheSupported(int localL1CacheSupported);
    /** Shared memory available per multiprocessor in bytes */
    public native @Cast("size_t") long sharedMemPerMultiprocessor(); public native cudaDeviceProp sharedMemPerMultiprocessor(long sharedMemPerMultiprocessor);
    /** 32-bit registers available per multiprocessor */
    public native int regsPerMultiprocessor(); public native cudaDeviceProp regsPerMultiprocessor(int regsPerMultiprocessor);
    /** Device supports allocating managed memory on this system */
    public native int managedMemory(); public native cudaDeviceProp managedMemory(int managedMemory);
    /** Device is on a multi-GPU board */
    public native int isMultiGpuBoard(); public native cudaDeviceProp isMultiGpuBoard(int isMultiGpuBoard);
    /** Unique identifier for a group of devices on the same multi-GPU board */
    public native int multiGpuBoardGroupID(); public native cudaDeviceProp multiGpuBoardGroupID(int multiGpuBoardGroupID);
}

/** Empty device properties */
// #define cudaDevicePropDontCare
//         {
//           {'\0'},    /* char   name[256];               */
//           0,         /* size_t totalGlobalMem;          */
//           0,         /* size_t sharedMemPerBlock;       */
//           0,         /* int    regsPerBlock;            */
//           0,         /* int    warpSize;                */
//           0,         /* size_t memPitch;                */
//           0,         /* int    maxThreadsPerBlock;      */
//           {0, 0, 0}, /* int    maxThreadsDim[3];        */
//           {0, 0, 0}, /* int    maxGridSize[3];          */
//           0,         /* int    clockRate;               */
//           0,         /* size_t totalConstMem;           */
//           -1,        /* int    major;                   */
//           -1,        /* int    minor;                   */
//           0,         /* size_t textureAlignment;        */
//           0,         /* size_t texturePitchAlignment    */
//           -1,        /* int    deviceOverlap;           */
//           0,         /* int    multiProcessorCount;     */
//           0,         /* int    kernelExecTimeoutEnabled */
//           0,         /* int    integrated               */
//           0,         /* int    canMapHostMemory         */
//           0,         /* int    computeMode              */
//           0,         /* int    maxTexture1D             */
//           0,         /* int    maxTexture1DMipmap       */
//           0,         /* int    maxTexture1DLinear       */
//           {0, 0},    /* int    maxTexture2D[2]          */
//           {0, 0},    /* int    maxTexture2DMipmap[2]    */
//           {0, 0, 0}, /* int    maxTexture2DLinear[3]    */
//           {0, 0},    /* int    maxTexture2DGather[2]    */
//           {0, 0, 0}, /* int    maxTexture3D[3]          */
//           {0, 0, 0}, /* int    maxTexture3DAlt[3]       */
//           0,         /* int    maxTextureCubemap        */
//           {0, 0},    /* int    maxTexture1DLayered[2]   */
//           {0, 0, 0}, /* int    maxTexture2DLayered[3]   */
//           {0, 0},    /* int    maxTextureCubemapLayered[2] */
//           0,         /* int    maxSurface1D             */
//           {0, 0},    /* int    maxSurface2D[2]          */
//           {0, 0, 0}, /* int    maxSurface3D[3]          */
//           {0, 0},    /* int    maxSurface1DLayered[2]   */
//           {0, 0, 0}, /* int    maxSurface2DLayered[3]   */
//           0,         /* int    maxSurfaceCubemap        */
//           {0, 0},    /* int    maxSurfaceCubemapLayered[2] */
//           0,         /* size_t surfaceAlignment         */
//           0,         /* int    concurrentKernels        */
//           0,         /* int    ECCEnabled               */
//           0,         /* int    pciBusID                 */
//           0,         /* int    pciDeviceID              */
//           0,         /* int    pciDomainID              */
//           0,         /* int    tccDriver                */
//           0,         /* int    asyncEngineCount         */
//           0,         /* int    unifiedAddressing        */
//           0,         /* int    memoryClockRate          */
//           0,         /* int    memoryBusWidth           */
//           0,         /* int    l2CacheSize              */
//           0,         /* int    maxThreadsPerMultiProcessor */
//           0,         /* int    streamPrioritiesSupported */
//           0,         /* int    globalL1CacheSupported   */
//           0,         /* int    localL1CacheSupported    */
//           0,         /* size_t sharedMemPerMultiprocessor; */
//           0,         /* int    regsPerMultiprocessor;   */
//           0,         /* int    managedMemory            */
//           0,         /* int    isMultiGpuBoard          */
//           0,         /* int    multiGpuBoardGroupID     */
//         }

/**
 * CUDA IPC Handle Size
 */
public static final int CUDA_IPC_HANDLE_SIZE = 64;

/**
 * CUDA IPC event handle
 */
public static class cudaIpcEventHandle_t extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public cudaIpcEventHandle_t() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public cudaIpcEventHandle_t(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public cudaIpcEventHandle_t(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public cudaIpcEventHandle_t position(int position) {
        return (cudaIpcEventHandle_t)super.position(position);
    }

    public native @Cast("char") byte reserved(int i); public native cudaIpcEventHandle_t reserved(int i, byte reserved);
    @MemberGetter public native @Cast("char*") BytePointer reserved();
}

/**
 * CUDA IPC memory handle
 */
public static class cudaIpcMemHandle_t extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public cudaIpcMemHandle_t() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public cudaIpcMemHandle_t(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public cudaIpcMemHandle_t(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public cudaIpcMemHandle_t position(int position) {
        return (cudaIpcMemHandle_t)super.position(position);
    }

    public native @Cast("char") byte reserved(int i); public native cudaIpcMemHandle_t reserved(int i, byte reserved);
    @MemberGetter public native @Cast("char*") BytePointer reserved();
}

/*******************************************************************************
*                                                                              *
*  SHORTHAND TYPE DEFINITION USED BY RUNTIME API                               *
*                                                                              *
*******************************************************************************/

/**
 * CUDA Error types
 */

/**
 * CUDA stream
 */
@Name("CUstream_st") @Opaque public static class cudaStream extends Pointer {
    /** Empty constructor. */
    public cudaStream() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public cudaStream(Pointer p) { super(p); }
}

/**
 * CUDA event types
 */
@Name("CUevent_st") @Opaque public static class cudaEvent extends Pointer {
    /** Empty constructor. */
    public cudaEvent() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public cudaEvent(Pointer p) { super(p); }
}

/**
 * CUDA graphics resource types
 */

/**
 * CUDA UUID types
 */
@Opaque public static class cudaUUID_t extends Pointer {
    /** Empty constructor. */
    public cudaUUID_t() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public cudaUUID_t(Pointer p) { super(p); }
}

/**
 * CUDA output file modes
 */

/** @} */
/** @} */ /* END CUDART_TYPES */

// #endif /* !__DRIVER_TYPES_H__ */


// Parsed from <surface_types.h>

/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
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
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
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

// #if !defined(__SURFACE_TYPES_H__)
// #define __SURFACE_TYPES_H__

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

// #include "driver_types.h"

/**
 * \addtogroup CUDART_TYPES
 *
 * @{
 */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

public static final int cudaSurfaceType1D =              0x01;
public static final int cudaSurfaceType2D =              0x02;
public static final int cudaSurfaceType3D =              0x03;
public static final int cudaSurfaceTypeCubemap =         0x0C;
public static final int cudaSurfaceType1DLayered =       0xF1;
public static final int cudaSurfaceType2DLayered =       0xF2;
public static final int cudaSurfaceTypeCubemapLayered =  0xFC;

/**
 * CUDA Surface boundary modes
 */
/** enum cudaSurfaceBoundaryMode */
public static final int
    /** Zero boundary mode */
    cudaBoundaryModeZero  = 0,
    /** Clamp boundary mode */
    cudaBoundaryModeClamp = 1,
    /** Trap boundary mode */
    cudaBoundaryModeTrap  = 2;

/**
 * CUDA Surface format modes
 */
/** enum cudaSurfaceFormatMode */
public static final int
    /** Forced format mode */
    cudaFormatModeForced = 0,
    /** Auto format mode */
    cudaFormatModeAuto = 1;

/**
 * CUDA Surface reference
 */
public static class surfaceReference extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public surfaceReference() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public surfaceReference(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public surfaceReference(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public surfaceReference position(int position) {
        return (surfaceReference)super.position(position);
    }

    /**
     * Channel descriptor for surface reference
     */
    public native @ByRef cudaChannelFormatDesc channelDesc(); public native surfaceReference channelDesc(cudaChannelFormatDesc channelDesc);
}

/**
 * CUDA Surface object
 */

/** @} */
/** @} */ /* END CUDART_TYPES */

// #endif /* !__SURFACE_TYPES_H__ */


// Parsed from <texture_types.h>

/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
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
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
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

// #if !defined(__TEXTURE_TYPES_H__)
// #define __TEXTURE_TYPES_H__

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

// #include "driver_types.h"

/**
 * \addtogroup CUDART_TYPES
 *
 * @{
 */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

public static final int cudaTextureType1D =              0x01;
public static final int cudaTextureType2D =              0x02;
public static final int cudaTextureType3D =              0x03;
public static final int cudaTextureTypeCubemap =         0x0C;
public static final int cudaTextureType1DLayered =       0xF1;
public static final int cudaTextureType2DLayered =       0xF2;
public static final int cudaTextureTypeCubemapLayered =  0xFC;

/**
 * CUDA texture address modes
 */
/** enum cudaTextureAddressMode */
public static final int
    /** Wrapping address mode */
    cudaAddressModeWrap   = 0,
    /** Clamp to edge address mode */
    cudaAddressModeClamp  = 1,
    /** Mirror address mode */
    cudaAddressModeMirror = 2,
    /** Border address mode */
    cudaAddressModeBorder = 3;

/**
 * CUDA texture filter modes
 */
/** enum cudaTextureFilterMode */
public static final int
    /** Point filter mode */
    cudaFilterModePoint  = 0,
    /** Linear filter mode */
    cudaFilterModeLinear = 1;

/**
 * CUDA texture read modes
 */
/** enum cudaTextureReadMode */
public static final int
    /** Read texture as specified element type */
    cudaReadModeElementType     = 0,
    /** Read texture as normalized float */
    cudaReadModeNormalizedFloat = 1;

/**
 * CUDA texture reference
 */
public static class textureReference extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public textureReference() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public textureReference(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public textureReference(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public textureReference position(int position) {
        return (textureReference)super.position(position);
    }

    /**
     * Indicates whether texture reads are normalized or not
     */
    public native int normalized(); public native textureReference normalized(int normalized);
    /**
     * Texture filter mode
     */
    public native @Cast("cudaTextureFilterMode") int filterMode(); public native textureReference filterMode(int filterMode);
    /**
     * Texture address mode for up to 3 dimensions
     */
    public native @Cast("cudaTextureAddressMode") int addressMode(int i); public native textureReference addressMode(int i, int addressMode);
    @MemberGetter public native @Cast("cudaTextureAddressMode*") IntPointer addressMode();
    /**
     * Channel descriptor for the texture reference
     */
    public native @ByRef cudaChannelFormatDesc channelDesc(); public native textureReference channelDesc(cudaChannelFormatDesc channelDesc);
    /**
     * Perform sRGB->linear conversion during texture read
     */
    public native int sRGB(); public native textureReference sRGB(int sRGB);
    /**
     * Limit to the anisotropy ratio
     */
    public native @Cast("unsigned int") int maxAnisotropy(); public native textureReference maxAnisotropy(int maxAnisotropy);
    /**
     * Mipmap filter mode
     */
    public native @Cast("cudaTextureFilterMode") int mipmapFilterMode(); public native textureReference mipmapFilterMode(int mipmapFilterMode);
    /**
     * Offset applied to the supplied mipmap level
     */
    public native float mipmapLevelBias(); public native textureReference mipmapLevelBias(float mipmapLevelBias);
    /**
     * Lower end of the mipmap level range to clamp access to
     */
    public native float minMipmapLevelClamp(); public native textureReference minMipmapLevelClamp(float minMipmapLevelClamp);
    /**
     * Upper end of the mipmap level range to clamp access to
     */
    public native float maxMipmapLevelClamp(); public native textureReference maxMipmapLevelClamp(float maxMipmapLevelClamp);
    public native int __cudaReserved(int i); public native textureReference __cudaReserved(int i, int __cudaReserved);
    @MemberGetter public native IntPointer __cudaReserved();
}

/**
 * CUDA texture descriptor
 */
public static class cudaTextureDesc extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public cudaTextureDesc() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public cudaTextureDesc(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public cudaTextureDesc(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public cudaTextureDesc position(int position) {
        return (cudaTextureDesc)super.position(position);
    }

    /**
     * Texture address mode for up to 3 dimensions
     */
    public native @Cast("cudaTextureAddressMode") int addressMode(int i); public native cudaTextureDesc addressMode(int i, int addressMode);
    @MemberGetter public native @Cast("cudaTextureAddressMode*") IntPointer addressMode();
    /**
     * Texture filter mode
     */
    public native @Cast("cudaTextureFilterMode") int filterMode(); public native cudaTextureDesc filterMode(int filterMode);
    /**
     * Texture read mode
     */
    public native @Cast("cudaTextureReadMode") int readMode(); public native cudaTextureDesc readMode(int readMode);
    /**
     * Perform sRGB->linear conversion during texture read
     */
    public native int sRGB(); public native cudaTextureDesc sRGB(int sRGB);
    /**
     * Indicates whether texture reads are normalized or not
     */
    public native int normalizedCoords(); public native cudaTextureDesc normalizedCoords(int normalizedCoords);
    /**
     * Limit to the anisotropy ratio
     */
    public native @Cast("unsigned int") int maxAnisotropy(); public native cudaTextureDesc maxAnisotropy(int maxAnisotropy);
    /**
     * Mipmap filter mode
     */
    public native @Cast("cudaTextureFilterMode") int mipmapFilterMode(); public native cudaTextureDesc mipmapFilterMode(int mipmapFilterMode);
    /**
     * Offset applied to the supplied mipmap level
     */
    public native float mipmapLevelBias(); public native cudaTextureDesc mipmapLevelBias(float mipmapLevelBias);
    /**
     * Lower end of the mipmap level range to clamp access to
     */
    public native float minMipmapLevelClamp(); public native cudaTextureDesc minMipmapLevelClamp(float minMipmapLevelClamp);
    /**
     * Upper end of the mipmap level range to clamp access to
     */
    public native float maxMipmapLevelClamp(); public native cudaTextureDesc maxMipmapLevelClamp(float maxMipmapLevelClamp);
}

/**
 * CUDA texture object
 */

/** @} */
/** @} */ /* END CUDART_TYPES */

// #endif /* !__TEXTURE_TYPES_H__ */


// Parsed from <vector_types.h>

/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
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
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
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

// #if !defined(__VECTOR_TYPES_H__)
// #define __VECTOR_TYPES_H__

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

// #if !defined(__CUDA_LIBDEVICE__)
// #include "builtin_types.h"
// #endif /* !__CUDA_LIBDEVICE__ */
// #include "host_defines.h"

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

// #if !defined(__CUDACC__) && !defined(__CUDABE__) &&
//     defined(_WIN32) && !defined(_WIN64)

// #else /* !__CUDACC__ && !__CUDABE__ && _WIN32 && !_WIN64 */

// #define __cuda_builtin_vector_align8(tag, members)
// struct __device_builtin__ __align__(8) tag
// {
//     members
// }

// #endif /* !__CUDACC__ && !__CUDABE__ && _WIN32 && !_WIN64 */

public static class char1 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public char1() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public char1(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public char1(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public char1 position(int position) {
        return (char1)super.position(position);
    }

    public native byte x(); public native char1 x(byte x);
}

public static class uchar1 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public uchar1() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public uchar1(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public uchar1(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public uchar1 position(int position) {
        return (uchar1)super.position(position);
    }

    public native @Cast("unsigned char") byte x(); public native uchar1 x(byte x);
}


public static class char2 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public char2() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public char2(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public char2(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public char2 position(int position) {
        return (char2)super.position(position);
    }

    public native byte x(); public native char2 x(byte x);
    public native byte y(); public native char2 y(byte y);
}

public static class uchar2 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public uchar2() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public uchar2(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public uchar2(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public uchar2 position(int position) {
        return (uchar2)super.position(position);
    }

    public native @Cast("unsigned char") byte x(); public native uchar2 x(byte x);
    public native @Cast("unsigned char") byte y(); public native uchar2 y(byte y);
}

public static class char3 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public char3() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public char3(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public char3(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public char3 position(int position) {
        return (char3)super.position(position);
    }

    public native byte x(); public native char3 x(byte x);
    public native byte y(); public native char3 y(byte y);
    public native byte z(); public native char3 z(byte z);
}

public static class uchar3 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public uchar3() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public uchar3(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public uchar3(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public uchar3 position(int position) {
        return (uchar3)super.position(position);
    }

    public native @Cast("unsigned char") byte x(); public native uchar3 x(byte x);
    public native @Cast("unsigned char") byte y(); public native uchar3 y(byte y);
    public native @Cast("unsigned char") byte z(); public native uchar3 z(byte z);
}

public static class char4 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public char4() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public char4(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public char4(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public char4 position(int position) {
        return (char4)super.position(position);
    }

    public native byte x(); public native char4 x(byte x);
    public native byte y(); public native char4 y(byte y);
    public native byte z(); public native char4 z(byte z);
    public native byte w(); public native char4 w(byte w);
}

public static class uchar4 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public uchar4() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public uchar4(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public uchar4(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public uchar4 position(int position) {
        return (uchar4)super.position(position);
    }

    public native @Cast("unsigned char") byte x(); public native uchar4 x(byte x);
    public native @Cast("unsigned char") byte y(); public native uchar4 y(byte y);
    public native @Cast("unsigned char") byte z(); public native uchar4 z(byte z);
    public native @Cast("unsigned char") byte w(); public native uchar4 w(byte w);
}

public static class short1 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public short1() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public short1(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public short1(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public short1 position(int position) {
        return (short1)super.position(position);
    }

    public native short x(); public native short1 x(short x);
}

public static class ushort1 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public ushort1() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public ushort1(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ushort1(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public ushort1 position(int position) {
        return (ushort1)super.position(position);
    }

    public native @Cast("unsigned short") short x(); public native ushort1 x(short x);
}

public static class short2 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public short2() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public short2(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public short2(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public short2 position(int position) {
        return (short2)super.position(position);
    }

    public native short x(); public native short2 x(short x);
    public native short y(); public native short2 y(short y);
}

public static class ushort2 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public ushort2() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public ushort2(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ushort2(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public ushort2 position(int position) {
        return (ushort2)super.position(position);
    }

    public native @Cast("unsigned short") short x(); public native ushort2 x(short x);
    public native @Cast("unsigned short") short y(); public native ushort2 y(short y);
}

public static class short3 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public short3() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public short3(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public short3(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public short3 position(int position) {
        return (short3)super.position(position);
    }

    public native short x(); public native short3 x(short x);
    public native short y(); public native short3 y(short y);
    public native short z(); public native short3 z(short z);
}

public static class ushort3 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public ushort3() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public ushort3(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ushort3(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public ushort3 position(int position) {
        return (ushort3)super.position(position);
    }

    public native @Cast("unsigned short") short x(); public native ushort3 x(short x);
    public native @Cast("unsigned short") short y(); public native ushort3 y(short y);
    public native @Cast("unsigned short") short z(); public native ushort3 z(short z);
}

public static class short4 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public short4() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public short4(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public short4(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public short4 position(int position) {
        return (short4)super.position(position);
    }
 public native short x(); public native short4 x(short x); public native short y(); public native short4 y(short y); public native short z(); public native short4 z(short z); public native short w(); public native short4 w(short w);
}
public static class ushort4 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public ushort4() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public ushort4(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ushort4(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public ushort4 position(int position) {
        return (ushort4)super.position(position);
    }
 public native @Cast("unsigned short") short x(); public native ushort4 x(short x); public native @Cast("unsigned short") short y(); public native ushort4 y(short y); public native @Cast("unsigned short") short z(); public native ushort4 z(short z); public native @Cast("unsigned short") short w(); public native ushort4 w(short w);
}

public static class int1 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public int1() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public int1(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public int1(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public int1 position(int position) {
        return (int1)super.position(position);
    }

    public native int x(); public native int1 x(int x);
}

public static class uint1 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public uint1() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public uint1(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public uint1(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public uint1 position(int position) {
        return (uint1)super.position(position);
    }

    public native @Cast("unsigned int") int x(); public native uint1 x(int x);
}

public static class int2 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public int2() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public int2(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public int2(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public int2 position(int position) {
        return (int2)super.position(position);
    }
 public native int x(); public native int2 x(int x); public native int y(); public native int2 y(int y);
}
public static class uint2 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public uint2() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public uint2(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public uint2(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public uint2 position(int position) {
        return (uint2)super.position(position);
    }
 public native @Cast("unsigned int") int x(); public native uint2 x(int x); public native @Cast("unsigned int") int y(); public native uint2 y(int y);
}

public static class int3 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public int3() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public int3(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public int3(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public int3 position(int position) {
        return (int3)super.position(position);
    }

    public native int x(); public native int3 x(int x);
    public native int y(); public native int3 y(int y);
    public native int z(); public native int3 z(int z);
}

public static class uint3 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public uint3() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public uint3(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public uint3(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public uint3 position(int position) {
        return (uint3)super.position(position);
    }

    public native @Cast("unsigned int") int x(); public native uint3 x(int x);
    public native @Cast("unsigned int") int y(); public native uint3 y(int y);
    public native @Cast("unsigned int") int z(); public native uint3 z(int z);
}

public static class int4 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public int4() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public int4(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public int4(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public int4 position(int position) {
        return (int4)super.position(position);
    }

    public native int x(); public native int4 x(int x);
    public native int y(); public native int4 y(int y);
    public native int z(); public native int4 z(int z);
    public native int w(); public native int4 w(int w);
}

public static class uint4 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public uint4() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public uint4(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public uint4(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public uint4 position(int position) {
        return (uint4)super.position(position);
    }

    public native @Cast("unsigned int") int x(); public native uint4 x(int x);
    public native @Cast("unsigned int") int y(); public native uint4 y(int y);
    public native @Cast("unsigned int") int z(); public native uint4 z(int z);
    public native @Cast("unsigned int") int w(); public native uint4 w(int w);
}

public static class long1 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public long1() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public long1(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public long1(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public long1 position(int position) {
        return (long1)super.position(position);
    }

    public native long x(); public native long1 x(long x);
}

public static class ulong1 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public ulong1() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public ulong1(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ulong1(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public ulong1 position(int position) {
        return (ulong1)super.position(position);
    }

    public native @Cast("unsigned long") long x(); public native ulong1 x(long x);
}

// #if defined (_WIN32)
public static class long2 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public long2() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public long2(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public long2(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public long2 position(int position) {
        return (long2)super.position(position);
    }
 public native long x(); public native long2 x(long x); public native long y(); public native long2 y(long y);
}
public static class ulong2 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public ulong2() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public ulong2(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ulong2(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public ulong2 position(int position) {
        return (ulong2)super.position(position);
    }
 public native @Cast("unsigned long int") long x(); public native ulong2 x(long x); public native @Cast("unsigned long int") long y(); public native ulong2 y(long y);
}
// #else /* _WIN32 */

// #endif /* _WIN32 */

public static class long3 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public long3() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public long3(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public long3(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public long3 position(int position) {
        return (long3)super.position(position);
    }

    public native long x(); public native long3 x(long x);
    public native long y(); public native long3 y(long y);
    public native long z(); public native long3 z(long z);
}

public static class ulong3 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public ulong3() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public ulong3(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ulong3(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public ulong3 position(int position) {
        return (ulong3)super.position(position);
    }

    public native @Cast("unsigned long int") long x(); public native ulong3 x(long x);
    public native @Cast("unsigned long int") long y(); public native ulong3 y(long y);
    public native @Cast("unsigned long int") long z(); public native ulong3 z(long z);
}

public static class long4 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public long4() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public long4(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public long4(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public long4 position(int position) {
        return (long4)super.position(position);
    }

    public native long x(); public native long4 x(long x);
    public native long y(); public native long4 y(long y);
    public native long z(); public native long4 z(long z);
    public native long w(); public native long4 w(long w);
}

public static class ulong4 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public ulong4() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public ulong4(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ulong4(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public ulong4 position(int position) {
        return (ulong4)super.position(position);
    }

    public native @Cast("unsigned long int") long x(); public native ulong4 x(long x);
    public native @Cast("unsigned long int") long y(); public native ulong4 y(long y);
    public native @Cast("unsigned long int") long z(); public native ulong4 z(long z);
    public native @Cast("unsigned long int") long w(); public native ulong4 w(long w);
}

public static class float1 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public float1() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public float1(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public float1(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public float1 position(int position) {
        return (float1)super.position(position);
    }

    public native float x(); public native float1 x(float x);
}

// #if !defined(__CUDACC__) && !defined(__CUDABE__) && defined(__arm__) &&
//     defined(__ARM_PCS_VFP) && __GNUC__ == 4 && __GNUC_MINOR__ == 6

// #else /* !__CUDACC__ && !__CUDABE__ && __arm__ && __ARM_PCS_VFP &&
//          __GNUC__ == 4&& __GNUC_MINOR__ == 6 */

public static class float2 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public float2() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public float2(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public float2(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public float2 position(int position) {
        return (float2)super.position(position);
    }
 public native float x(); public native float2 x(float x); public native float y(); public native float2 y(float y);
}

// #endif /* !__CUDACC__ && !__CUDABE__ && __arm__ && __ARM_PCS_VFP &&
//           __GNUC__ == 4&& __GNUC_MINOR__ == 6 */

public static class float3 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public float3() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public float3(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public float3(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public float3 position(int position) {
        return (float3)super.position(position);
    }

    public native float x(); public native float3 x(float x);
    public native float y(); public native float3 y(float y);
    public native float z(); public native float3 z(float z);
}

public static class float4 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public float4() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public float4(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public float4(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public float4 position(int position) {
        return (float4)super.position(position);
    }

    public native float x(); public native float4 x(float x);
    public native float y(); public native float4 y(float y);
    public native float z(); public native float4 z(float z);
    public native float w(); public native float4 w(float w);
}

public static class longlong1 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public longlong1() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public longlong1(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public longlong1(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public longlong1 position(int position) {
        return (longlong1)super.position(position);
    }

    public native long x(); public native longlong1 x(long x);
}

public static class ulonglong1 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public ulonglong1() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public ulonglong1(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ulonglong1(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public ulonglong1 position(int position) {
        return (ulonglong1)super.position(position);
    }

    public native @Cast("unsigned long long int") long x(); public native ulonglong1 x(long x);
}

public static class longlong2 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public longlong2() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public longlong2(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public longlong2(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public longlong2 position(int position) {
        return (longlong2)super.position(position);
    }

    public native long x(); public native longlong2 x(long x);
    public native long y(); public native longlong2 y(long y);
}

public static class ulonglong2 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public ulonglong2() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public ulonglong2(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ulonglong2(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public ulonglong2 position(int position) {
        return (ulonglong2)super.position(position);
    }

    public native @Cast("unsigned long long int") long x(); public native ulonglong2 x(long x);
    public native @Cast("unsigned long long int") long y(); public native ulonglong2 y(long y);
}

public static class longlong3 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public longlong3() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public longlong3(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public longlong3(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public longlong3 position(int position) {
        return (longlong3)super.position(position);
    }

    public native long x(); public native longlong3 x(long x);
    public native long y(); public native longlong3 y(long y);
    public native long z(); public native longlong3 z(long z);
}

public static class ulonglong3 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public ulonglong3() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public ulonglong3(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ulonglong3(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public ulonglong3 position(int position) {
        return (ulonglong3)super.position(position);
    }

    public native @Cast("unsigned long long int") long x(); public native ulonglong3 x(long x);
    public native @Cast("unsigned long long int") long y(); public native ulonglong3 y(long y);
    public native @Cast("unsigned long long int") long z(); public native ulonglong3 z(long z);
}

public static class longlong4 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public longlong4() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public longlong4(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public longlong4(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public longlong4 position(int position) {
        return (longlong4)super.position(position);
    }

    public native long x(); public native longlong4 x(long x);
    public native long y(); public native longlong4 y(long y);
    public native long z(); public native longlong4 z(long z);
    public native long w(); public native longlong4 w(long w);
}

public static class ulonglong4 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public ulonglong4() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public ulonglong4(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ulonglong4(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public ulonglong4 position(int position) {
        return (ulonglong4)super.position(position);
    }

    public native @Cast("unsigned long long int") long x(); public native ulonglong4 x(long x);
    public native @Cast("unsigned long long int") long y(); public native ulonglong4 y(long y);
    public native @Cast("unsigned long long int") long z(); public native ulonglong4 z(long z);
    public native @Cast("unsigned long long int") long w(); public native ulonglong4 w(long w);
}

public static class double1 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public double1() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public double1(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public double1(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public double1 position(int position) {
        return (double1)super.position(position);
    }

    public native double x(); public native double1 x(double x);
}

public static class double2 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public double2() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public double2(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public double2(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public double2 position(int position) {
        return (double2)super.position(position);
    }

    public native double x(); public native double2 x(double x);
    public native double y(); public native double2 y(double y);
}

public static class double3 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public double3() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public double3(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public double3(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public double3 position(int position) {
        return (double3)super.position(position);
    }

    public native double x(); public native double3 x(double x);
    public native double y(); public native double3 y(double y);
    public native double z(); public native double3 z(double z);
}

public static class double4 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public double4() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public double4(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public double4(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public double4 position(int position) {
        return (double4)super.position(position);
    }

    public native double x(); public native double4 x(double x);
    public native double y(); public native double4 y(double y);
    public native double z(); public native double4 z(double z);
    public native double w(); public native double4 w(double w);
}

// #if !defined(__CUDACC__) && !defined(__CUDABE__) &&
//     defined(_WIN32) && !defined(_WIN64)

// #endif /* !__CUDACC__ && !__CUDABE__ && _WIN32 && !_WIN64 */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

@NoOffset public static class dim3 extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public dim3(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public dim3(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public dim3 position(int position) {
        return (dim3)super.position(position);
    }

    public native @Cast("unsigned int") int x(); public native dim3 x(int x);
    public native @Cast("unsigned int") int y(); public native dim3 y(int y);
    public native @Cast("unsigned int") int z(); public native dim3 z(int z);
// #if defined(__cplusplus)
    public dim3(@Cast("unsigned int") int vx/*=1*/, @Cast("unsigned int") int vy/*=1*/, @Cast("unsigned int") int vz/*=1*/) { allocate(vx, vy, vz); }
    private native void allocate(@Cast("unsigned int") int vx/*=1*/, @Cast("unsigned int") int vy/*=1*/, @Cast("unsigned int") int vz/*=1*/);
    public dim3() { allocate(); }
    private native void allocate();
    public dim3(@ByVal uint3 v) { allocate(v); }
    private native void allocate(@ByVal uint3 v);
    public native @ByVal @Name("operator uint3") uint3 asUint3();
// #endif /* __cplusplus */
}

// #undef  __cuda_builtin_vector_align8

// #endif /* !__VECTOR_TYPES_H__ */


// Parsed from <builtin_types.h>

/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
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
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
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

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

// #include "device_types.h"
// #include "driver_types.h"
// #include "surface_types.h"
// #include "texture_types.h"
// #include "vector_types.h"


// Parsed from <cuda_runtime_api.h>

/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
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
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
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

// #if !defined(__CUDA_RUNTIME_API_H__)
// #define __CUDA_RUNTIME_API_H__

/**
 * \latexonly
 * \page sync_async API synchronization behavior
 *
 * \section memcpy_sync_async_behavior Memcpy
 * The API provides memcpy/memset functions in both synchronous and asynchronous forms,
 * the latter having an \e "Async" suffix. This is a misnomer as each function
 * may exhibit synchronous or asynchronous behavior depending on the arguments
 * passed to the function. In the reference documentation, each memcpy function is
 * categorized as \e synchronous or \e asynchronous, corresponding to the definitions
 * below.
 * 
 * \subsection MemcpySynchronousBehavior Synchronous
 * 
 * <ol>
 * <li> For transfers from pageable host memory to device memory, a stream sync is performed
 * before the copy is initiated. The function will return once the pageable
 * buffer has been copied to the staging memory for DMA transfer to device memory,
 * but the DMA to final destination may not have completed.
 * 
 * <li> For transfers from pinned host memory to device memory, the function is synchronous
 * with respect to the host.
 *
 * <li> For transfers from device to either pageable or pinned host memory, the function returns
 * only once the copy has completed.
 * 
 * <li> For transfers from device memory to device memory, no host-side synchronization is
 * performed.
 *
 * <li> For transfers from any host memory to any host memory, the function is fully
 * synchronous with respect to the host.
 * </ol>
 * 
 * \subsection MemcpyAsynchronousBehavior Asynchronous
 *
 * <ol>
 * <li> For transfers from pageable host memory to device memory, host memory is
 * copied to a staging buffer immediately (no device synchronization is
 * performed). The function will return once the pageable buffer has been copied
 * to the staging memory. The DMA transfer to final destination may not have
 * completed.
 *
 * <li> For transfers between pinned host memory and device memory, the function
 * is fully asynchronous.
 * 
 * <li> For transfers from device memory to pageable host memory, the function
 * will return only once the copy has completed.
 * 
 * <li> For all other transfers, the function is fully asynchronous. If pageable
 * memory must first be staged to pinned memory, this will be handled
 * asynchronously with a worker thread.
 *
 * <li> For transfers from any host memory to any host memory, the function is fully
 * synchronous with respect to the host.
 * </ol>
 *
 * \section memset_sync_async_behavior Memset
 * The cudaMemset functions are asynchronous with respect to the host
 * except when the target memory is pinned host memory. The \e Async
 * versions are always asynchronous with respect to the host.
 *
 * \section kernel_launch_details Kernel Launches
 * Kernel launches are asynchronous with respect to the host. Details of
 * concurrent kernel execution and data transfers can be found in the CUDA
 * Programmers Guide.
 *
 * \endlatexonly
 */

/**
 * There are two levels for the runtime API.
 *
 * The C API (<i>cuda_runtime_api.h</i>) is
 * a C-style interface that does not require compiling with \p nvcc.
 *
 * The \ref CUDART_HIGHLEVEL "C++ API" (<i>cuda_runtime.h</i>) is a
 * C++-style interface built on top of the C API. It wraps some of the
 * C API routines, using overloading, references and default arguments.
 * These wrappers can be used from C++ code and can be compiled with any C++
 * compiler. The C++ API also has some CUDA-specific wrappers that wrap
 * C API routines that deal with symbols, textures, and device functions.
 * These wrappers require the use of \p nvcc because they depend on code being
 * generated by the compiler. For example, the execution configuration syntax
 * to invoke kernels is only available in source code compiled with \p nvcc.
 */

/** CUDA Runtime API Version */
public static final int CUDART_VERSION =  6050;

// #include "host_defines.h"
// #include "builtin_types.h"
// #include "cuda_device_runtime_api.h"

/** \cond impl_private */
// #if !defined(__dv)

// #if defined(__cplusplus)

// #define __dv(v)
//         = v

// #else /* __cplusplus */

// #define __dv(v)

// #endif /* __cplusplus */

// #endif /* !__dv */
/** \endcond impl_private */

// #if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 350)   /** Visible to SM>=3.5 and "__host__ __device__" only **/

// #define CUDART_DEVICE __device__ 

// #else

// #define CUDART_DEVICE

// #endif /** CUDART_DEVICE */

// #if defined(__cplusplus)
// #endif /* __cplusplus */

/**
 * \defgroup CUDART_DEVICE Device Management
 *
 * ___MANBRIEF___ device management functions of the CUDA runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the device management functions of the CUDA runtime
 * application programming interface.
 *
 * @{
 */

/**
 * \brief Destroy all allocations and reset all state on the current device
 * in the current process.
 *
 * Explicitly destroys and cleans up all resources associated with the current
 * device in the current process.  Any subsequent API call to this device will 
 * reinitialize the device.
 *
 * Note that this function will reset the device immediately.  It is the caller's
 * responsibility to ensure that the device is not being accessed by any 
 * other host threads from the process when this function is called.
 *
 * \return
 * ::cudaSuccess
 * \notefnerr
 *
 * \sa ::cudaDeviceSynchronize
 */
public static native @Cast("cudaError_t") int cudaDeviceReset();

/**
 * \brief Wait for compute device to finish
 *
 * Blocks until the device has completed all preceding requested tasks.
 * ::cudaDeviceSynchronize() returns an error if one of the preceding tasks
 * has failed. If the ::cudaDeviceScheduleBlockingSync flag was set for 
 * this device, the host thread will block until the device has finished 
 * its work.
 *
 * \return
 * ::cudaSuccess
 * \notefnerr
 *
 * \sa ::cudaDeviceReset
 */
public static native @Cast("cudaError_t") int cudaDeviceSynchronize();

/**
 * \brief Set resource limits
 *
 * Setting \p limit to \p value is a request by the application to update
 * the current limit maintained by the device.  The driver is free to
 * modify the requested value to meet h/w requirements (this could be
 * clamping to minimum or maximum values, rounding up to nearest element
 * size, etc).  The application can use ::cudaDeviceGetLimit() to find out
 * exactly what the limit has been set to.
 *
 * Setting each ::cudaLimit has its own specific restrictions, so each is
 * discussed here.
 *
 * - ::cudaLimitStackSize controls the stack size in bytes of each GPU
 *   thread. This limit is only applicable to devices of compute capability
 *   2.0 and higher.  Attempting to set this limit on devices of compute
 *   capability less than 2.0 will result in the error
 *   ::cudaErrorUnsupportedLimit being returned.
 *
 * - ::cudaLimitPrintfFifoSize controls the size in bytes of the shared FIFO
 *   used by the ::printf() and ::fprintf() device system calls. Setting
 *   ::cudaLimitPrintfFifoSize must be performed before launching any kernel
 *   that uses the ::printf() or ::fprintf() device system calls, otherwise
 *   ::cudaErrorInvalidValue will be returned. This limit is only applicable
 *   to devices of compute capability 2.0 and higher.  Attempting to set this
 *   limit on devices of compute capability less than 2.0 will result in the
 *   error ::cudaErrorUnsupportedLimit being returned.
 *
 * - ::cudaLimitMallocHeapSize controls the size in bytes of the heap used by
 *   the ::malloc() and ::free() device system calls. Setting
 *   ::cudaLimitMallocHeapSize must be performed before launching any kernel
 *   that uses the ::malloc() or ::free() device system calls, otherwise
 *   ::cudaErrorInvalidValue will be returned. This limit is only applicable
 *   to devices of compute capability 2.0 and higher. Attempting to set this
 *   limit on devices of compute capability less than 2.0 will result in the
 *   error ::cudaErrorUnsupportedLimit being returned.
 *
 * - ::cudaLimitDevRuntimeSyncDepth controls the maximum nesting depth of a
 *   grid at which a thread can safely call ::cudaDeviceSynchronize(). Setting
 *   this limit must be performed before any launch of a kernel that uses the
 *   device runtime and calls ::cudaDeviceSynchronize() above the default sync
 *   depth, two levels of grids. Calls to ::cudaDeviceSynchronize() will fail
 *   with error code ::cudaErrorSyncDepthExceeded if the limitation is
 *   violated. This limit can be set smaller than the default or up the maximum
 *   launch depth of 24. When setting this limit, keep in mind that additional
 *   levels of sync depth require the runtime to reserve large amounts of
 *   device memory which can no longer be used for user allocations. If these
 *   reservations of device memory fail, ::cudaDeviceSetLimit will return
 *   ::cudaErrorMemoryAllocation, and the limit can be reset to a lower value.
 *   This limit is only applicable to devices of compute capability 3.5 and
 *   higher. Attempting to set this limit on devices of compute capability less
 *   than 3.5 will result in the error ::cudaErrorUnsupportedLimit being
 *   returned.
 *
 * - ::cudaLimitDevRuntimePendingLaunchCount controls the maximum number of
 *   outstanding device runtime launches that can be made from the current
 *   device. A grid is outstanding from the point of launch up until the grid
 *   is known to have been completed. Device runtime launches which violate 
 *   this limitation fail and return ::cudaErrorLaunchPendingCountExceeded when
 *   ::cudaGetLastError() is called after launch. If more pending launches than
 *   the default (2048 launches) are needed for a module using the device
 *   runtime, this limit can be increased. Keep in mind that being able to
 *   sustain additional pending launches will require the runtime to reserve
 *   larger amounts of device memory upfront which can no longer be used for
 *   allocations. If these reservations fail, ::cudaDeviceSetLimit will return
 *   ::cudaErrorMemoryAllocation, and the limit can be reset to a lower value.
 *   This limit is only applicable to devices of compute capability 3.5 and
 *   higher. Attempting to set this limit on devices of compute capability less
 *   than 3.5 will result in the error ::cudaErrorUnsupportedLimit being
 *   returned. 
 *
 * \param limit - Limit to set
 * \param value - Size of limit
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorUnsupportedLimit,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorMemoryAllocation
 * \notefnerr
 *
 * \sa ::cudaDeviceGetLimit
 */
public static native @Cast("cudaError_t") int cudaDeviceSetLimit(@Cast("cudaLimit") int limit, @Cast("size_t") long value);

/**
 * \brief Returns resource limits
 *
 * Returns in \p *pValue the current size of \p limit.  The supported
 * ::cudaLimit values are:
 * - ::cudaLimitStackSize: stack size in bytes of each GPU thread;
 * - ::cudaLimitPrintfFifoSize: size in bytes of the shared FIFO used by the
 *   ::printf() and ::fprintf() device system calls.
 * - ::cudaLimitMallocHeapSize: size in bytes of the heap used by the
 *   ::malloc() and ::free() device system calls;
 * - ::cudaLimitDevRuntimeSyncDepth: maximum grid depth at which a
 *   thread can isssue the device runtime call ::cudaDeviceSynchronize()
 *   to wait on child grid launches to complete.
 * - ::cudaLimitDevRuntimePendingLaunchCount: maximum number of outstanding
 *   device runtime launches.
 *
 * \param limit  - Limit to query
 * \param pValue - Returned size of the limit
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorUnsupportedLimit,
 * ::cudaErrorInvalidValue
 * \notefnerr
 *
 * \sa ::cudaDeviceSetLimit
 */
public static native @Cast("cudaError_t") int cudaDeviceGetLimit(@Cast("size_t*") SizeTPointer pValue, @Cast("cudaLimit") int limit);

/**
 * \brief Returns the preferred cache configuration for the current device.
 *
 * On devices where the L1 cache and shared memory use the same hardware
 * resources, this returns through \p pCacheConfig the preferred cache
 * configuration for the current device. This is only a preference. The
 * runtime will use the requested configuration if possible, but it is free to
 * choose a different configuration if required to execute functions.
 *
 * This will return a \p pCacheConfig of ::cudaFuncCachePreferNone on devices
 * where the size of the L1 cache and shared memory are fixed.
 *
 * The supported cache configurations are:
 * - ::cudaFuncCachePreferNone: no preference for shared memory or L1 (default)
 * - ::cudaFuncCachePreferShared: prefer larger shared memory and smaller L1 cache
 * - ::cudaFuncCachePreferL1: prefer larger L1 cache and smaller shared memory
 * - ::cudaFuncCachePreferEqual: prefer equal size L1 cache and shared memory
 *
 * \param pCacheConfig - Returned cache configuration
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInitializationError
 * \notefnerr
 *
 * \sa cudaDeviceSetCacheConfig,
 * \ref ::cudaFuncSetCacheConfig(const void*, enum cudaFuncCache) "cudaFuncSetCacheConfig (C API)",
 * \ref ::cudaFuncSetCacheConfig(T*, enum cudaFuncCache) "cudaFuncSetCacheConfig (C++ API)"
 */
public static native @Cast("cudaError_t") int cudaDeviceGetCacheConfig(@Cast("cudaFuncCache*") IntPointer pCacheConfig);
public static native @Cast("cudaError_t") int cudaDeviceGetCacheConfig(@Cast("cudaFuncCache*") IntBuffer pCacheConfig);
public static native @Cast("cudaError_t") int cudaDeviceGetCacheConfig(@Cast("cudaFuncCache*") int[] pCacheConfig);

/**
 * \brief Returns numerical values that correspond to the least and
 * greatest stream priorities.
 *
 * Returns in \p *leastPriority and \p *greatestPriority the numerical values that correspond
 * to the least and greatest stream priorities respectively. Stream priorities
 * follow a convention where lower numbers imply greater priorities. The range of
 * meaningful stream priorities is given by [\p *greatestPriority, \p *leastPriority].
 * If the user attempts to create a stream with a priority value that is
 * outside the the meaningful range as specified by this API, the priority is
 * automatically clamped down or up to either \p *leastPriority or \p *greatestPriority
 * respectively. See ::cudaStreamCreateWithPriority for details on creating a
 * priority stream.
 * A NULL may be passed in for \p *leastPriority or \p *greatestPriority if the value
 * is not desired.
 *
 * This function will return '0' in both \p *leastPriority and \p *greatestPriority if
 * the current context's device does not support stream priorities
 * (see ::cudaDeviceGetAttribute).
 *
 * \param leastPriority    - Pointer to an int in which the numerical value for least
 *                           stream priority is returned
 * \param greatestPriority - Pointer to an int in which the numerical value for greatest
 *                           stream priority is returned
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \notefnerr
 *
 * \sa ::cudaStreamCreateWithPriority,
 * ::cudaStreamGetPriority
 */
public static native @Cast("cudaError_t") int cudaDeviceGetStreamPriorityRange(IntPointer leastPriority, IntPointer greatestPriority);
public static native @Cast("cudaError_t") int cudaDeviceGetStreamPriorityRange(IntBuffer leastPriority, IntBuffer greatestPriority);
public static native @Cast("cudaError_t") int cudaDeviceGetStreamPriorityRange(int[] leastPriority, int[] greatestPriority);

/**
 * \brief Sets the preferred cache configuration for the current device.
 *
 * On devices where the L1 cache and shared memory use the same hardware
 * resources, this sets through \p cacheConfig the preferred cache
 * configuration for the current device. This is only a preference. The
 * runtime will use the requested configuration if possible, but it is free to
 * choose a different configuration if required to execute the function. Any
 * function preference set via
 * \ref ::cudaFuncSetCacheConfig(const void*, enum cudaFuncCache) "cudaFuncSetCacheConfig (C API)"
 * or
 * \ref ::cudaFuncSetCacheConfig(T*, enum cudaFuncCache) "cudaFuncSetCacheConfig (C++ API)"
 * will be preferred over this device-wide setting. Setting the device-wide
 * cache configuration to ::cudaFuncCachePreferNone will cause subsequent
 * kernel launches to prefer to not change the cache configuration unless
 * required to launch the kernel.
 *
 * This setting does nothing on devices where the size of the L1 cache and
 * shared memory are fixed.
 *
 * Launching a kernel with a different preference than the most recent
 * preference setting may insert a device-side synchronization point.
 *
 * The supported cache configurations are:
 * - ::cudaFuncCachePreferNone: no preference for shared memory or L1 (default)
 * - ::cudaFuncCachePreferShared: prefer larger shared memory and smaller L1 cache
 * - ::cudaFuncCachePreferL1: prefer larger L1 cache and smaller shared memory
 * - ::cudaFuncCachePreferEqual: prefer equal size L1 cache and shared memory
 *
 * \param cacheConfig - Requested cache configuration
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInitializationError
 * \notefnerr
 *
 * \sa ::cudaDeviceGetCacheConfig,
 * \ref ::cudaFuncSetCacheConfig(const void*, enum cudaFuncCache) "cudaFuncSetCacheConfig (C API)",
 * \ref ::cudaFuncSetCacheConfig(T*, enum cudaFuncCache) "cudaFuncSetCacheConfig (C++ API)"
 */
public static native @Cast("cudaError_t") int cudaDeviceSetCacheConfig(@Cast("cudaFuncCache") int cacheConfig);

/**
 * \brief Returns the shared memory configuration for the current device.
 *
 * This function will return in \p pConfig the current size of shared memory banks
 * on the current device. On devices with configurable shared memory banks, 
 * ::cudaDeviceSetSharedMemConfig can be used to change this setting, so that all 
 * subsequent kernel launches will by default use the new bank size. When 
 * ::cudaDeviceGetSharedMemConfig is called on devices without configurable shared 
 * memory, it will return the fixed bank size of the hardware.
 *
 * The returned bank configurations can be either:
 * - ::cudaSharedMemBankSizeFourByte - shared memory bank width is four bytes.
 * - ::cudaSharedMemBankSizeEightByte - shared memory bank width is eight bytes.
 *
 * \param pConfig - Returned cache configuration
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInitializationError
 * \notefnerr
 *
 * \sa ::cudaDeviceSetCacheConfig,
 * ::cudaDeviceGetCacheConfig,
 * ::cudaDeviceSetSharedMemConfig,
 * ::cudaFuncSetCacheConfig
 */
public static native @Cast("cudaError_t") int cudaDeviceGetSharedMemConfig(@Cast("cudaSharedMemConfig*") IntPointer pConfig);
public static native @Cast("cudaError_t") int cudaDeviceGetSharedMemConfig(@Cast("cudaSharedMemConfig*") IntBuffer pConfig);
public static native @Cast("cudaError_t") int cudaDeviceGetSharedMemConfig(@Cast("cudaSharedMemConfig*") int[] pConfig);

/**
 * \brief Sets the shared memory configuration for the current device.
 *
 * On devices with configurable shared memory banks, this function will set
 * the shared memory bank size which is used for all subsequent kernel launches.
 * Any per-function setting of shared memory set via ::cudaFuncSetSharedMemConfig
 * will override the device wide setting.
 *
 * Changing the shared memory configuration between launches may introduce
 * a device side synchronization point.
 *
 * Changing the shared memory bank size will not increase shared memory usage
 * or affect occupancy of kernels, but may have major effects on performance. 
 * Larger bank sizes will allow for greater potential bandwidth to shared memory,
 * but will change what kinds of accesses to shared memory will result in bank 
 * conflicts.
 *
 * This function will do nothing on devices with fixed shared memory bank size.
 *
 * The supported bank configurations are:
 * - ::cudaSharedMemBankSizeDefault: set bank width the device default (currently,
 *   four bytes)
 * - ::cudaSharedMemBankSizeFourByte: set shared memory bank width to be four bytes
 *   natively.
 * - ::cudaSharedMemBankSizeEightByte: set shared memory bank width to be eight 
 *   bytes natively.
 *
 * \param config - Requested cache configuration
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInitializationError
 * \notefnerr
 *
 * \sa ::cudaDeviceSetCacheConfig,
 * ::cudaDeviceGetCacheConfig,
 * ::cudaDeviceGetSharedMemConfig,
 * ::cudaFuncSetCacheConfig
 */
public static native @Cast("cudaError_t") int cudaDeviceSetSharedMemConfig(@Cast("cudaSharedMemConfig") int config);

/**
 * \brief Returns a handle to a compute device
 *
 * Returns in \p *device a device ordinal given a PCI bus ID string.
 *
 * \param device   - Returned device ordinal
 *
 * \param pciBusId - String in one of the following forms: 
 * [domain]:[bus]:[device].[function]
 * [domain]:[bus]:[device]
 * [bus]:[device].[function]
 * where \p domain, \p bus, \p device, and \p function are all hexadecimal values
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidDevice
 * \notefnerr
 *
 * \sa ::cudaDeviceGetPCIBusId
 */
public static native @Cast("cudaError_t") int cudaDeviceGetByPCIBusId(IntPointer device, @Cast("const char*") BytePointer pciBusId);
public static native @Cast("cudaError_t") int cudaDeviceGetByPCIBusId(IntBuffer device, String pciBusId);
public static native @Cast("cudaError_t") int cudaDeviceGetByPCIBusId(int[] device, @Cast("const char*") BytePointer pciBusId);
public static native @Cast("cudaError_t") int cudaDeviceGetByPCIBusId(IntPointer device, String pciBusId);
public static native @Cast("cudaError_t") int cudaDeviceGetByPCIBusId(IntBuffer device, @Cast("const char*") BytePointer pciBusId);
public static native @Cast("cudaError_t") int cudaDeviceGetByPCIBusId(int[] device, String pciBusId);

/**
 * \brief Returns a PCI Bus Id string for the device
 *
 * Returns an ASCII string identifying the device \p dev in the NULL-terminated
 * string pointed to by \p pciBusId. \p len specifies the maximum length of the
 * string that may be returned.
 *
 * \param pciBusId - Returned identifier string for the device in the following format
 * [domain]:[bus]:[device].[function]
 * where \p domain, \p bus, \p device, and \p function are all hexadecimal values.
 * pciBusId should be large enough to store 13 characters including the NULL-terminator.
 *
 * \param len      - Maximum length of string to store in \p name
 *
 * \param device   - Device to get identifier string for
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidDevice
 * \notefnerr
 *
 * \sa ::cudaDeviceGetByPCIBusId

 */
public static native @Cast("cudaError_t") int cudaDeviceGetPCIBusId(@Cast("char*") BytePointer pciBusId, int len, int device);
public static native @Cast("cudaError_t") int cudaDeviceGetPCIBusId(@Cast("char*") ByteBuffer pciBusId, int len, int device);
public static native @Cast("cudaError_t") int cudaDeviceGetPCIBusId(@Cast("char*") byte[] pciBusId, int len, int device);

/**
 * \brief Gets an interprocess handle for a previously allocated event
 *
 * Takes as input a previously allocated event. This event must have been 
 * created with the ::cudaEventInterprocess and ::cudaEventDisableTiming
 * flags set. This opaque handle may be copied into other processes and
 * opened with ::cudaIpcOpenEventHandle to allow efficient hardware
 * synchronization between GPU work in different processes.
 *
 * After the event has been been opened in the importing process, 
 * ::cudaEventRecord, ::cudaEventSynchronize, ::cudaStreamWaitEvent and 
 * ::cudaEventQuery may be used in either process. Performing operations 
 * on the imported event after the exported event has been freed 
 * with ::cudaEventDestroy will result in undefined behavior.
 *
 * IPC functionality is restricted to devices with support for unified 
 * addressing on Linux operating systems.
 *
 * \param handle - Pointer to a user allocated cudaIpcEventHandle
 *                    in which to return the opaque event handle
 * \param event   - Event allocated with ::cudaEventInterprocess and 
 *                    ::cudaEventDisableTiming flags.
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorMemoryAllocation,
 * ::cudaErrorMapBufferObjectFailed
 *
 * \sa
 * ::cudaEventCreate,
 * ::cudaEventDestroy,
 * ::cudaEventSynchronize,
 * ::cudaEventQuery,
 * ::cudaStreamWaitEvent,
 * ::cudaIpcOpenEventHandle,
 * ::cudaIpcGetMemHandle,
 * ::cudaIpcOpenMemHandle,
 * ::cudaIpcCloseMemHandle
 */
public static native @Cast("cudaError_t") int cudaIpcGetEventHandle(cudaIpcEventHandle_t handle, cudaEvent event);

/**
 * \brief Opens an interprocess event handle for use in the current process
 *
 * Opens an interprocess event handle exported from another process with 
 * ::cudaIpcGetEventHandle. This function returns a ::cudaEvent_t that behaves like 
 * a locally created event with the ::cudaEventDisableTiming flag specified. 
 * This event must be freed with ::cudaEventDestroy.
 *
 * Performing operations on the imported event after the exported event has 
 * been freed with ::cudaEventDestroy will result in undefined behavior.
 *
 * IPC functionality is restricted to devices with support for unified 
 * addressing on Linux operating systems.
 *
 * \param event - Returns the imported event
 * \param handle  - Interprocess handle to open
 *
 * \returns
 * ::cudaSuccess,
 * ::cudaErrorMapBufferObjectFailed,
 * ::cudaErrorInvalidResourceHandle
 *
  * \sa
 * ::cudaEventCreate,
 * ::cudaEventDestroy,
 * ::cudaEventSynchronize,
 * ::cudaEventQuery,
 * ::cudaStreamWaitEvent,
 * ::cudaIpcGetEventHandle,
 * ::cudaIpcGetMemHandle,
 * ::cudaIpcOpenMemHandle,
 * ::cudaIpcCloseMemHandle
 */
public static native @Cast("cudaError_t") int cudaIpcOpenEventHandle(@ByPtrPtr cudaEvent event, @ByVal cudaIpcEventHandle_t handle);


/**
 * \brief Gets an interprocess memory handle for an existing device memory
 *          allocation
 *
 * Takes a pointer to the base of an existing device memory allocation created 
 * with ::cudaMalloc and exports it for use in another process. This is a 
 * lightweight operation and may be called multiple times on an allocation
 * without adverse effects. 
 *
 * If a region of memory is freed with ::cudaFree and a subsequent call
 * to ::cudaMalloc returns memory with the same device address,
 * ::cudaIpcGetMemHandle will return a unique handle for the
 * new memory. 
 *
 * IPC functionality is restricted to devices with support for unified 
 * addressing on Linux operating systems.
 *
 * \param handle - Pointer to user allocated ::cudaIpcMemHandle to return
 *                    the handle in.
 * \param devPtr - Base pointer to previously allocated device memory 
 *
 * \returns
 * ::cudaSuccess,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorMemoryAllocation,
 * ::cudaErrorMapBufferObjectFailed,
 *
 * \sa
 * ::cudaMalloc,
 * ::cudaFree,
 * ::cudaIpcGetEventHandle,
 * ::cudaIpcOpenEventHandle,
 * ::cudaIpcOpenMemHandle,
 * ::cudaIpcCloseMemHandle
 */
public static native @Cast("cudaError_t") int cudaIpcGetMemHandle(cudaIpcMemHandle_t handle, Pointer devPtr);

/**
 * \brief Opens an interprocess memory handle exported from another process
 *          and returns a device pointer usable in the local process.
 *
 * Maps memory exported from another process with ::cudaIpcGetMemHandle into
 * the current device address space. For contexts on different devices 
 * ::cudaIpcOpenMemHandle can attempt to enable peer access between the
 * devices as if the user called ::cudaDeviceEnablePeerAccess. This behavior is 
 * controlled by the ::cudaIpcMemLazyEnablePeerAccess flag. 
 * ::cudaDeviceCanAccessPeer can determine if a mapping is possible.
 *
 * Contexts that may open ::cudaIpcMemHandles are restricted in the following way.
 * ::cudaIpcMemHandles from each device in a given process may only be opened 
 * by one context per device per other process.
 *
 * Memory returned from ::cudaIpcOpenMemHandle must be freed with
 * ::cudaIpcCloseMemHandle.
 *
 * Calling ::cudaFree on an exported memory region before calling
 * ::cudaIpcCloseMemHandle in the importing context will result in undefined
 * behavior.
 * 
 * IPC functionality is restricted to devices with support for unified 
 * addressing on Linux operating systems.
 *
 * \param devPtr - Returned device pointer
 * \param handle - ::cudaIpcMemHandle to open
 * \param flags  - Flags for this operation. Must be specified as ::cudaIpcMemLazyEnablePeerAccess
 *
 * \returns
 * ::cudaSuccess,
 * ::cudaErrorMapBufferObjectFailed,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorTooManyPeers
 *
 * \note No guarantees are made about the address returned in \p *devPtr.  
 * In particular, multiple processes may not receive the same address for the same \p handle.
 *
 * \sa
 * ::cudaMalloc,
 * ::cudaFree,
 * ::cudaIpcGetEventHandle,
 * ::cudaIpcOpenEventHandle,
 * ::cudaIpcGetMemHandle,
 * ::cudaIpcCloseMemHandle,
 * ::cudaDeviceEnablePeerAccess,
 * ::cudaDeviceCanAccessPeer,
 */
public static native @Cast("cudaError_t") int cudaIpcOpenMemHandle(@Cast("void**") PointerPointer devPtr, @ByVal cudaIpcMemHandle_t handle, @Cast("unsigned int") int flags);
public static native @Cast("cudaError_t") int cudaIpcOpenMemHandle(@Cast("void**") @ByPtrPtr Pointer devPtr, @ByVal cudaIpcMemHandle_t handle, @Cast("unsigned int") int flags);

/**
 * \brief Close memory mapped with cudaIpcOpenMemHandle
 * 
 * Unmaps memory returnd by ::cudaIpcOpenMemHandle. The original allocation
 * in the exporting process as well as imported mappings in other processes
 * will be unaffected.
 *
 * Any resources used to enable peer access will be freed if this is the
 * last mapping using them.
 *
 * IPC functionality is restricted to devices with support for unified 
 * addressing on Linux operating systems.
 *
 * \param devPtr - Device pointer returned by ::cudaIpcOpenMemHandle
 * 
 * \returns
 * ::cudaSuccess,
 * ::cudaErrorMapBufferObjectFailed,
 * ::cudaErrorInvalidResourceHandle,
 *
 * \sa
 * ::cudaMalloc,
 * ::cudaFree,
 * ::cudaIpcGetEventHandle,
 * ::cudaIpcOpenEventHandle,
 * ::cudaIpcGetMemHandle,
 * ::cudaIpcOpenMemHandle,
 */
public static native @Cast("cudaError_t") int cudaIpcCloseMemHandle(Pointer devPtr);

/** @} */ /* END CUDART_DEVICE */

/**
 * \defgroup CUDART_THREAD_DEPRECATED Thread Management [DEPRECATED]
 *
 * ___MANBRIEF___ deprecated thread management functions of the CUDA runtime
 * API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes deprecated thread management functions of the CUDA runtime
 * application programming interface.
 *
 * @{
 */

/**
 * \brief Exit and clean up from CUDA launches
 *
 * \deprecated
 *
 * Note that this function is deprecated because its name does not 
 * reflect its behavior.  Its functionality is identical to the 
 * non-deprecated function ::cudaDeviceReset(), which should be used
 * instead.
 *
 * Explicitly destroys all cleans up all resources associated with the current
 * device in the current process.  Any subsequent API call to this device will 
 * reinitialize the device.  
 *
 * Note that this function will reset the device immediately.  It is the caller's
 * responsibility to ensure that the device is not being accessed by any 
 * other host threads from the process when this function is called.
 *
 * \return
 * ::cudaSuccess
 * \notefnerr
 *
 * \sa ::cudaDeviceReset
 */
public static native @Cast("cudaError_t") int cudaThreadExit();

/**
 * \brief Wait for compute device to finish
 *
 * \deprecated
 *
 * Note that this function is deprecated because its name does not 
 * reflect its behavior.  Its functionality is similar to the 
 * non-deprecated function ::cudaDeviceSynchronize(), which should be used
 * instead.
 *
 * Blocks until the device has completed all preceding requested tasks.
 * ::cudaThreadSynchronize() returns an error if one of the preceding tasks
 * has failed. If the ::cudaDeviceScheduleBlockingSync flag was set for 
 * this device, the host thread will block until the device has finished 
 * its work.
 *
 * \return
 * ::cudaSuccess
 * \notefnerr
 *
 * \sa ::cudaDeviceSynchronize
 */
public static native @Cast("cudaError_t") int cudaThreadSynchronize();

/**
 * \brief Set resource limits
 *
 * \deprecated
 *
 * Note that this function is deprecated because its name does not 
 * reflect its behavior.  Its functionality is identical to the 
 * non-deprecated function ::cudaDeviceSetLimit(), which should be used
 * instead.
 *
 * Setting \p limit to \p value is a request by the application to update
 * the current limit maintained by the device.  The driver is free to
 * modify the requested value to meet h/w requirements (this could be
 * clamping to minimum or maximum values, rounding up to nearest element
 * size, etc).  The application can use ::cudaThreadGetLimit() to find out
 * exactly what the limit has been set to.
 *
 * Setting each ::cudaLimit has its own specific restrictions, so each is
 * discussed here.
 *
 * - ::cudaLimitStackSize controls the stack size of each GPU thread.
 *   This limit is only applicable to devices of compute capability
 *   2.0 and higher.  Attempting to set this limit on devices of
 *   compute capability less than 2.0 will result in the error
 *   ::cudaErrorUnsupportedLimit being returned.
 *
 * - ::cudaLimitPrintfFifoSize controls the size of the shared FIFO
 *   used by the ::printf() and ::fprintf() device system calls.
 *   Setting ::cudaLimitPrintfFifoSize must be performed before
 *   launching any kernel that uses the ::printf() or ::fprintf() device
 *   system calls, otherwise ::cudaErrorInvalidValue will be returned.
 *   This limit is only applicable to devices of compute capability
 *   2.0 and higher.  Attempting to set this limit on devices of
 *   compute capability less than 2.0 will result in the error
 *   ::cudaErrorUnsupportedLimit being returned.
 *
 * - ::cudaLimitMallocHeapSize controls the size of the heap used
 *   by the ::malloc() and ::free() device system calls.  Setting
 *   ::cudaLimitMallocHeapSize must be performed before launching
 *   any kernel that uses the ::malloc() or ::free() device system calls,
 *   otherwise ::cudaErrorInvalidValue will be returned.
 *   This limit is only applicable to devices of compute capability
 *   2.0 and higher.  Attempting to set this limit on devices of
 *   compute capability less than 2.0 will result in the error
 *   ::cudaErrorUnsupportedLimit being returned.
 *
 * \param limit - Limit to set
 * \param value - Size in bytes of limit
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorUnsupportedLimit,
 * ::cudaErrorInvalidValue
 * \notefnerr
 *
 * \sa ::cudaDeviceSetLimit
 */
public static native @Cast("cudaError_t") int cudaThreadSetLimit(@Cast("cudaLimit") int limit, @Cast("size_t") long value);

/**
 * \brief Returns resource limits
 *
 * \deprecated
 *
 * Note that this function is deprecated because its name does not 
 * reflect its behavior.  Its functionality is identical to the 
 * non-deprecated function ::cudaDeviceGetLimit(), which should be used
 * instead.
 *
 * Returns in \p *pValue the current size of \p limit.  The supported
 * ::cudaLimit values are:
 * - ::cudaLimitStackSize: stack size of each GPU thread;
 * - ::cudaLimitPrintfFifoSize: size of the shared FIFO used by the
 *   ::printf() and ::fprintf() device system calls.
 * - ::cudaLimitMallocHeapSize: size of the heap used by the
 *   ::malloc() and ::free() device system calls;
 *
 * \param limit  - Limit to query
 * \param pValue - Returned size in bytes of limit
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorUnsupportedLimit,
 * ::cudaErrorInvalidValue
 * \notefnerr
 *
 * \sa ::cudaDeviceGetLimit
 */
public static native @Cast("cudaError_t") int cudaThreadGetLimit(@Cast("size_t*") SizeTPointer pValue, @Cast("cudaLimit") int limit);

/**
 * \brief Returns the preferred cache configuration for the current device.
 *
 * \deprecated
 *
 * Note that this function is deprecated because its name does not 
 * reflect its behavior.  Its functionality is identical to the 
 * non-deprecated function ::cudaDeviceGetCacheConfig(), which should be 
 * used instead.
 * 
 * On devices where the L1 cache and shared memory use the same hardware
 * resources, this returns through \p pCacheConfig the preferred cache
 * configuration for the current device. This is only a preference. The
 * runtime will use the requested configuration if possible, but it is free to
 * choose a different configuration if required to execute functions.
 *
 * This will return a \p pCacheConfig of ::cudaFuncCachePreferNone on devices
 * where the size of the L1 cache and shared memory are fixed.
 *
 * The supported cache configurations are:
 * - ::cudaFuncCachePreferNone: no preference for shared memory or L1 (default)
 * - ::cudaFuncCachePreferShared: prefer larger shared memory and smaller L1 cache
 * - ::cudaFuncCachePreferL1: prefer larger L1 cache and smaller shared memory
 *
 * \param pCacheConfig - Returned cache configuration
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInitializationError
 * \notefnerr
 *
 * \sa cudaDeviceGetCacheConfig
 */
public static native @Cast("cudaError_t") int cudaThreadGetCacheConfig(@Cast("cudaFuncCache*") IntPointer pCacheConfig);
public static native @Cast("cudaError_t") int cudaThreadGetCacheConfig(@Cast("cudaFuncCache*") IntBuffer pCacheConfig);
public static native @Cast("cudaError_t") int cudaThreadGetCacheConfig(@Cast("cudaFuncCache*") int[] pCacheConfig);

/**
 * \brief Sets the preferred cache configuration for the current device.
 *
 * \deprecated
 *
 * Note that this function is deprecated because its name does not 
 * reflect its behavior.  Its functionality is identical to the 
 * non-deprecated function ::cudaDeviceSetCacheConfig(), which should be 
 * used instead.
 * 
 * On devices where the L1 cache and shared memory use the same hardware
 * resources, this sets through \p cacheConfig the preferred cache
 * configuration for the current device. This is only a preference. The
 * runtime will use the requested configuration if possible, but it is free to
 * choose a different configuration if required to execute the function. Any
 * function preference set via
 * \ref ::cudaFuncSetCacheConfig(const void*, enum cudaFuncCache) "cudaFuncSetCacheConfig (C API)"
 * or
 * \ref ::cudaFuncSetCacheConfig(T*, enum cudaFuncCache) "cudaFuncSetCacheConfig (C++ API)"
 * will be preferred over this device-wide setting. Setting the device-wide
 * cache configuration to ::cudaFuncCachePreferNone will cause subsequent
 * kernel launches to prefer to not change the cache configuration unless
 * required to launch the kernel.
 *
 * This setting does nothing on devices where the size of the L1 cache and
 * shared memory are fixed.
 *
 * Launching a kernel with a different preference than the most recent
 * preference setting may insert a device-side synchronization point.
 *
 * The supported cache configurations are:
 * - ::cudaFuncCachePreferNone: no preference for shared memory or L1 (default)
 * - ::cudaFuncCachePreferShared: prefer larger shared memory and smaller L1 cache
 * - ::cudaFuncCachePreferL1: prefer larger L1 cache and smaller shared memory
 *
 * \param cacheConfig - Requested cache configuration
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInitializationError
 * \notefnerr
 *
 * \sa ::cudaDeviceSetCacheConfig
 */
public static native @Cast("cudaError_t") int cudaThreadSetCacheConfig(@Cast("cudaFuncCache") int cacheConfig);

/** @} */ /* END CUDART_THREAD_DEPRECATED */

/**
 * \defgroup CUDART_ERROR Error Handling
 *
 * ___MANBRIEF___ error handling functions of the CUDA runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the error handling functions of the CUDA runtime
 * application programming interface.
 *
 * @{
 */

/**
 * \brief Returns the last error from a runtime call
 *
 * Returns the last error that has been produced by any of the runtime calls
 * in the same host thread and resets it to ::cudaSuccess.
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorMissingConfiguration,
 * ::cudaErrorMemoryAllocation,
 * ::cudaErrorInitializationError,
 * ::cudaErrorLaunchFailure,
 * ::cudaErrorLaunchTimeout,
 * ::cudaErrorLaunchOutOfResources,
 * ::cudaErrorInvalidDeviceFunction,
 * ::cudaErrorInvalidConfiguration,
 * ::cudaErrorInvalidDevice,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidPitchValue,
 * ::cudaErrorInvalidSymbol,
 * ::cudaErrorUnmapBufferObjectFailed,
 * ::cudaErrorInvalidHostPointer,
 * ::cudaErrorInvalidDevicePointer,
 * ::cudaErrorInvalidTexture,
 * ::cudaErrorInvalidTextureBinding,
 * ::cudaErrorInvalidChannelDescriptor,
 * ::cudaErrorInvalidMemcpyDirection,
 * ::cudaErrorInvalidFilterSetting,
 * ::cudaErrorInvalidNormSetting,
 * ::cudaErrorUnknown,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorInsufficientDriver,
 * ::cudaErrorSetOnActiveProcess,
 * ::cudaErrorStartupFailure,
 * \notefnerr
 *
 * \sa ::cudaPeekAtLastError, ::cudaGetErrorName, ::cudaGetErrorString, ::cudaError
 */
public static native @Cast("cudaError_t") int cudaGetLastError();

/**
 * \brief Returns the last error from a runtime call
 *
 * Returns the last error that has been produced by any of the runtime calls
 * in the same host thread. Note that this call does not reset the error to
 * ::cudaSuccess like ::cudaGetLastError().
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorMissingConfiguration,
 * ::cudaErrorMemoryAllocation,
 * ::cudaErrorInitializationError,
 * ::cudaErrorLaunchFailure,
 * ::cudaErrorLaunchTimeout,
 * ::cudaErrorLaunchOutOfResources,
 * ::cudaErrorInvalidDeviceFunction,
 * ::cudaErrorInvalidConfiguration,
 * ::cudaErrorInvalidDevice,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidPitchValue,
 * ::cudaErrorInvalidSymbol,
 * ::cudaErrorUnmapBufferObjectFailed,
 * ::cudaErrorInvalidHostPointer,
 * ::cudaErrorInvalidDevicePointer,
 * ::cudaErrorInvalidTexture,
 * ::cudaErrorInvalidTextureBinding,
 * ::cudaErrorInvalidChannelDescriptor,
 * ::cudaErrorInvalidMemcpyDirection,
 * ::cudaErrorInvalidFilterSetting,
 * ::cudaErrorInvalidNormSetting,
 * ::cudaErrorUnknown,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorInsufficientDriver,
 * ::cudaErrorSetOnActiveProcess,
 * ::cudaErrorStartupFailure,
 * \notefnerr
 *
 * \sa ::cudaGetLastError, ::cudaGetErrorName, ::cudaGetErrorString, ::cudaError
 */
public static native @Cast("cudaError_t") int cudaPeekAtLastError();

/**
 * \brief Returns the string representation of an error code enum name
 *
 * Returns a string containing the name of an error code in the enum, or NULL
 * if the error code is not valid.
 *
 * \param error - Error code to convert to string
 *
 * \return
 * \p char* pointer to a NULL-terminated string, or NULL if the error code is not valid.
 *
 * \sa ::cudaGetErrorString, ::cudaGetLastError, ::cudaPeekAtLastError, ::cudaError
 */
public static native @Cast("const char*") BytePointer cudaGetErrorName(@Cast("cudaError_t") int error);

/**
 * \brief Returns the description string for an error code
 *
 * Returns the description string for an error code, or NULL if the error
 * code is not valid.
 *
 * \param error - Error code to convert to string
 *
 * \return
 * \p char* pointer to a NULL-terminated string, or NULL if the error code is not valid.
 *
 * \sa ::cudaGetErrorName, ::cudaGetLastError, ::cudaPeekAtLastError, ::cudaError
 */
public static native @Cast("const char*") BytePointer cudaGetErrorString(@Cast("cudaError_t") int error);
/** @} */ /* END CUDART_ERROR */

/**
 * \addtogroup CUDART_DEVICE 
 *
 * @{
 */

/**
 * \brief Returns the number of compute-capable devices
 *
 * Returns in \p *count the number of devices with compute capability greater
 * or equal to 1.0 that are available for execution.  If there is no such
 * device then ::cudaGetDeviceCount() will return ::cudaErrorNoDevice.
 * If no driver can be loaded to determine if any such devices exist then
 * ::cudaGetDeviceCount() will return ::cudaErrorInsufficientDriver.
 *
 * \param count - Returns the number of devices with compute capability
 * greater or equal to 1.0
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorNoDevice,
 * ::cudaErrorInsufficientDriver
 * \notefnerr
 *
 * \sa ::cudaGetDevice, ::cudaSetDevice, ::cudaGetDeviceProperties,
 * ::cudaChooseDevice
 */
public static native @Cast("cudaError_t") int cudaGetDeviceCount(IntPointer count);
public static native @Cast("cudaError_t") int cudaGetDeviceCount(IntBuffer count);
public static native @Cast("cudaError_t") int cudaGetDeviceCount(int[] count);

/**
 * \brief Returns information about the compute-device
 *
 * Returns in \p *prop the properties of device \p dev. The ::cudaDeviceProp
 * structure is defined as:
 * \code
    struct cudaDeviceProp {
        char name[256];
        size_t totalGlobalMem;
        size_t sharedMemPerBlock;
        int regsPerBlock;
        int warpSize;
        size_t memPitch;
        int maxThreadsPerBlock;
        int maxThreadsDim[3];
        int maxGridSize[3];
        int clockRate;
        size_t totalConstMem;
        int major;
        int minor;
        size_t textureAlignment;
        size_t texturePitchAlignment;
        int deviceOverlap;
        int multiProcessorCount;
        int kernelExecTimeoutEnabled;
        int integrated;
        int canMapHostMemory;
        int computeMode;
        int maxTexture1D;
        int maxTexture1DMipmap;
        int maxTexture1DLinear;
        int maxTexture2D[2];
        int maxTexture2DMipmap[2];
        int maxTexture2DLinear[3];
        int maxTexture2DGather[2];
        int maxTexture3D[3];
        int maxTexture3DAlt[3];
        int maxTextureCubemap;
        int maxTexture1DLayered[2];
        int maxTexture2DLayered[3];
        int maxTextureCubemapLayered[2];
        int maxSurface1D;
        int maxSurface2D[2];
        int maxSurface3D[3];
        int maxSurface1DLayered[2];
        int maxSurface2DLayered[3];
        int maxSurfaceCubemap;
        int maxSurfaceCubemapLayered[2];
        size_t surfaceAlignment;
        int concurrentKernels;
        int ECCEnabled;
        int pciBusID;
        int pciDeviceID;
        int pciDomainID;
        int tccDriver;
        int asyncEngineCount;
        int unifiedAddressing;
        int memoryClockRate;
        int memoryBusWidth;
        int l2CacheSize;
        int maxThreadsPerMultiProcessor;
        int streamPrioritiesSupported;
        int globalL1CacheSupported;
        int localL1CacheSupported;
        size_t sharedMemPerMultiprocessor;
        int regsPerMultiprocessor;
        int managedMemSupported;
        int isMultiGpuBoard;
        int multiGpuBoardGroupID;
    }
 \endcode
 * where:
 * - \ref ::cudaDeviceProp::name "name[256]" is an ASCII string identifying
 *   the device;
 * - \ref ::cudaDeviceProp::totalGlobalMem "totalGlobalMem" is the total
 *   amount of global memory available on the device in bytes;
 * - \ref ::cudaDeviceProp::sharedMemPerBlock "sharedMemPerBlock" is the
 *   maximum amount of shared memory available to a thread block in bytes;
 * - \ref ::cudaDeviceProp::regsPerBlock "regsPerBlock" is the maximum number
 *   of 32-bit registers available to a thread block;
 * - \ref ::cudaDeviceProp::warpSize "warpSize" is the warp size in threads;
 * - \ref ::cudaDeviceProp::memPitch "memPitch" is the maximum pitch in
 *   bytes allowed by the memory copy functions that involve memory regions
 *   allocated through ::cudaMallocPitch();
 * - \ref ::cudaDeviceProp::maxThreadsPerBlock "maxThreadsPerBlock" is the
 *   maximum number of threads per block;
 * - \ref ::cudaDeviceProp::maxThreadsDim "maxThreadsDim[3]" contains the
 *   maximum size of each dimension of a block;
 * - \ref ::cudaDeviceProp::maxGridSize "maxGridSize[3]" contains the
 *   maximum size of each dimension of a grid;
 * - \ref ::cudaDeviceProp::clockRate "clockRate" is the clock frequency in
 *   kilohertz;
 * - \ref ::cudaDeviceProp::totalConstMem "totalConstMem" is the total amount
 *   of constant memory available on the device in bytes;
 * - \ref ::cudaDeviceProp::major "major",
 *   \ref ::cudaDeviceProp::minor "minor" are the major and minor revision
 *   numbers defining the device's compute capability;
 * - \ref ::cudaDeviceProp::textureAlignment "textureAlignment" is the
 *   alignment requirement; texture base addresses that are aligned to
 *   \ref ::cudaDeviceProp::textureAlignment "textureAlignment" bytes do not
 *   need an offset applied to texture fetches;
 * - \ref ::cudaDeviceProp::texturePitchAlignment "texturePitchAlignment" is the
 *   pitch alignment requirement for 2D texture references that are bound to 
 *   pitched memory;
 * - \ref ::cudaDeviceProp::deviceOverlap "deviceOverlap" is 1 if the device
 *   can concurrently copy memory between host and device while executing a
 *   kernel, or 0 if not.  Deprecated, use instead asyncEngineCount.
 * - \ref ::cudaDeviceProp::multiProcessorCount "multiProcessorCount" is the
 *   number of multiprocessors on the device;
 * - \ref ::cudaDeviceProp::kernelExecTimeoutEnabled "kernelExecTimeoutEnabled"
 *   is 1 if there is a run time limit for kernels executed on the device, or
 *   0 if not.
 * - \ref ::cudaDeviceProp::integrated "integrated" is 1 if the device is an
 *   integrated (motherboard) GPU and 0 if it is a discrete (card) component.
 * - \ref ::cudaDeviceProp::canMapHostMemory "canMapHostMemory" is 1 if the
 *   device can map host memory into the CUDA address space for use with
 *   ::cudaHostAlloc()/::cudaHostGetDevicePointer(), or 0 if not;
 * - \ref ::cudaDeviceProp::computeMode "computeMode" is the compute mode
 *   that the device is currently in. Available modes are as follows:
 *   - cudaComputeModeDefault: Default mode - Device is not restricted and
 *     multiple threads can use ::cudaSetDevice() with this device.
 *   - cudaComputeModeExclusive: Compute-exclusive mode - Only one thread will
 *     be able to use ::cudaSetDevice() with this device.
 *   - cudaComputeModeProhibited: Compute-prohibited mode - No threads can use
 *     ::cudaSetDevice() with this device.
 *   - cudaComputeModeExclusiveProcess: Compute-exclusive-process mode - Many 
 *     threads in one process will be able to use ::cudaSetDevice() with this device.
 *   <br> If ::cudaSetDevice() is called on an already occupied \p device with 
 *   computeMode ::cudaComputeModeExclusive, ::cudaErrorDeviceAlreadyInUse
 *   will be immediately returned indicating the device cannot be used.
 *   When an occupied exclusive mode device is chosen with ::cudaSetDevice,
 *   all subsequent non-device management runtime functions will return
 *   ::cudaErrorDevicesUnavailable.
 * - \ref ::cudaDeviceProp::maxTexture1D "maxTexture1D" is the maximum 1D
 *   texture size.
 * - \ref ::cudaDeviceProp::maxTexture1DMipmap "maxTexture1DMipmap" is the maximum
 *   1D mipmapped texture texture size.
 * - \ref ::cudaDeviceProp::maxTexture1DLinear "maxTexture1DLinear" is the maximum
 *   1D texture size for textures bound to linear memory.
 * - \ref ::cudaDeviceProp::maxTexture2D "maxTexture2D[2]" contains the maximum
 *   2D texture dimensions.
 * - \ref ::cudaDeviceProp::maxTexture2DMipmap "maxTexture2DMipmap[2]" contains the
 *   maximum 2D mipmapped texture dimensions.
 * - \ref ::cudaDeviceProp::maxTexture2DLinear "maxTexture2DLinear[3]" contains the 
 *   maximum 2D texture dimensions for 2D textures bound to pitch linear memory.
 * - \ref ::cudaDeviceProp::maxTexture2DGather "maxTexture2DGather[2]" contains the 
 *   maximum 2D texture dimensions if texture gather operations have to be performed.
 * - \ref ::cudaDeviceProp::maxTexture3D "maxTexture3D[3]" contains the maximum
 *   3D texture dimensions.
 * - \ref ::cudaDeviceProp::maxTexture3DAlt "maxTexture3DAlt[3]"
 *   contains the maximum alternate 3D texture dimensions.
 * - \ref ::cudaDeviceProp::maxTextureCubemap "maxTextureCubemap" is the 
 *   maximum cubemap texture width or height.
 * - \ref ::cudaDeviceProp::maxTexture1DLayered "maxTexture1DLayered[2]" contains
 *   the maximum 1D layered texture dimensions.
 * - \ref ::cudaDeviceProp::maxTexture2DLayered "maxTexture2DLayered[3]" contains
 *   the maximum 2D layered texture dimensions.
 * - \ref ::cudaDeviceProp::maxTextureCubemapLayered "maxTextureCubemapLayered[2]"
 *   contains the maximum cubemap layered texture dimensions.
 * - \ref ::cudaDeviceProp::maxSurface1D "maxSurface1D" is the maximum 1D
 *   surface size.
 * - \ref ::cudaDeviceProp::maxSurface2D "maxSurface2D[2]" contains the maximum
 *   2D surface dimensions.
 * - \ref ::cudaDeviceProp::maxSurface3D "maxSurface3D[3]" contains the maximum
 *   3D surface dimensions.
 * - \ref ::cudaDeviceProp::maxSurface1DLayered "maxSurface1DLayered[2]" contains
 *   the maximum 1D layered surface dimensions.
 * - \ref ::cudaDeviceProp::maxSurface2DLayered "maxSurface2DLayered[3]" contains
 *   the maximum 2D layered surface dimensions.
 * - \ref ::cudaDeviceProp::maxSurfaceCubemap "maxSurfaceCubemap" is the maximum 
 *   cubemap surface width or height.
 * - \ref ::cudaDeviceProp::maxSurfaceCubemapLayered "maxSurfaceCubemapLayered[2]"
 *   contains the maximum cubemap layered surface dimensions.
 * - \ref ::cudaDeviceProp::surfaceAlignment "surfaceAlignment" specifies the
 *   alignment requirements for surfaces.
 * - \ref ::cudaDeviceProp::concurrentKernels "concurrentKernels" is 1 if the
 *   device supports executing multiple kernels within the same context
 *   simultaneously, or 0 if not. It is not guaranteed that multiple kernels
 *   will be resident on the device concurrently so this feature should not be
 *   relied upon for correctness;
 * - \ref ::cudaDeviceProp::ECCEnabled "ECCEnabled" is 1 if the device has ECC
 *   support turned on, or 0 if not.
 * - \ref ::cudaDeviceProp::pciBusID "pciBusID" is the PCI bus identifier of
 *   the device.
 * - \ref ::cudaDeviceProp::pciDeviceID "pciDeviceID" is the PCI device
 *   (sometimes called slot) identifier of the device.
 * - \ref ::cudaDeviceProp::pciDomainID "pciDomainID" is the PCI domain identifier
 *   of the device.
 * - \ref ::cudaDeviceProp::tccDriver "tccDriver" is 1 if the device is using a
 *   TCC driver or 0 if not.
 * - \ref ::cudaDeviceProp::asyncEngineCount "asyncEngineCount" is 1 when the
 *   device can concurrently copy memory between host and device while executing
 *   a kernel. It is 2 when the device can concurrently copy memory between host
 *   and device in both directions and execute a kernel at the same time. It is
 *   0 if neither of these is supported.
 * - \ref ::cudaDeviceProp::unifiedAddressing "unifiedAddressing" is 1 if the device 
 *   shares a unified address space with the host and 0 otherwise.
 * - \ref ::cudaDeviceProp::memoryClockRate "memoryClockRate" is the peak memory 
 *   clock frequency in kilohertz.
 * - \ref ::cudaDeviceProp::memoryBusWidth "memoryBusWidth" is the memory bus width  
 *   in bits.
 * - \ref ::cudaDeviceProp::l2CacheSize "l2CacheSize" is L2 cache size in bytes. 
 * - \ref ::cudaDeviceProp::maxThreadsPerMultiProcessor "maxThreadsPerMultiProcessor"  
 *   is the number of maximum resident threads per multiprocessor.
 * - \ref ::cudaDeviceProp::streamPrioritiesSupported "streamPrioritiesSupported"
 *   is 1 if the device supports stream priorities, or 0 if it is not supported.
 * - \ref ::cudaDeviceProp::globalL1CacheSupported "globalL1CacheSupported"
 *   is 1 if the device supports caching of globals in L1 cache, or 0 if it is not supported.
 * - \ref ::cudaDeviceProp::localL1CacheSupported "localL1CacheSupported"
 *   is 1 if the device supports caching of locals in L1 cache, or 0 if it is not supported.
 * - \ref ::cudaDeviceProp::sharedMemPerMultiprocessor "sharedMemPerMultiprocessor" is the
 *   maximum amount of shared memory available to a multiprocessor in bytes; this amount is
 *   shared by all thread blocks simultaneously resident on a multiprocessor;
 * - \ref ::cudaDeviceProp::regsPerMultiprocessor "regsPerMultiprocessor" is the maximum number
 *   of 32-bit registers available to a multiprocessor; this number is shared
 *   by all thread blocks simultaneously resident on a multiprocessor;
 * - \ref ::cudaDeviceProp::managedMemSupported "managedMemSupported"
 *   is 1 if the device supports allocating managed memory, or 0 if it is not supported.
 * - \ref ::cudaDeviceProp::isMultiGpuBoard "isMultiGpuBoard"
 *   is 1 if the device is on a multi-GPU board (e.g. Gemini cards), and 0 if not;
 * - \ref ::cudaDeviceProp::multiGpuBoardGroupID "multiGpuBoardGroupID" is a unique identifier
 *   for a group of devices associated with the same board.
 *   Devices on the same multi-GPU board will share the same identifier;
 *
 * \param prop   - Properties for the specified device
 * \param device - Device number to get properties for
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDevice
 *
 * \sa ::cudaGetDeviceCount, ::cudaGetDevice, ::cudaSetDevice, ::cudaChooseDevice,
 * ::cudaDeviceGetAttribute
 */
public static native @Cast("cudaError_t") int cudaGetDeviceProperties(cudaDeviceProp prop, int device);

/**
 * \brief Returns information about the device
 *
 * Returns in \p *value the integer value of the attribute \p attr on device
 * \p device. The supported attributes are:
 * - ::cudaDevAttrMaxThreadsPerBlock: Maximum number of threads per block;
 * - ::cudaDevAttrMaxBlockDimX: Maximum x-dimension of a block;
 * - ::cudaDevAttrMaxBlockDimY: Maximum y-dimension of a block;
 * - ::cudaDevAttrMaxBlockDimZ: Maximum z-dimension of a block;
 * - ::cudaDevAttrMaxGridDimX: Maximum x-dimension of a grid;
 * - ::cudaDevAttrMaxGridDimY: Maximum y-dimension of a grid;
 * - ::cudaDevAttrMaxGridDimZ: Maximum z-dimension of a grid;
 * - ::cudaDevAttrMaxSharedMemoryPerBlock: Maximum amount of shared memory
 *   available to a thread block in bytes;
 * - ::cudaDevAttrTotalConstantMemory: Memory available on device for
 *   __constant__ variables in a CUDA C kernel in bytes;
 * - ::cudaDevAttrWarpSize: Warp size in threads;
 * - ::cudaDevAttrMaxPitch: Maximum pitch in bytes allowed by the memory copy
 *   functions that involve memory regions allocated through ::cudaMallocPitch();
 * - ::cudaDevAttrMaxTexture1DWidth: Maximum 1D texture width;
 * - ::cudaDevAttrMaxTexture1DLinearWidth: Maximum width for a 1D texture bound
 *   to linear memory;
 * - ::cudaDevAttrMaxTexture1DMipmappedWidth: Maximum mipmapped 1D texture width;
 * - ::cudaDevAttrMaxTexture2DWidth: Maximum 2D texture width;
 * - ::cudaDevAttrMaxTexture2DHeight: Maximum 2D texture height;
 * - ::cudaDevAttrMaxTexture2DLinearWidth: Maximum width for a 2D texture
 *   bound to linear memory;
 * - ::cudaDevAttrMaxTexture2DLinearHeight: Maximum height for a 2D texture
 *   bound to linear memory;
 * - ::cudaDevAttrMaxTexture2DLinearPitch: Maximum pitch in bytes for a 2D
 *   texture bound to linear memory;
 * - ::cudaDevAttrMaxTexture2DMipmappedWidth: Maximum mipmapped 2D texture
 *   width;
 * - ::cudaDevAttrMaxTexture2DMipmappedHeight: Maximum mipmapped 2D texture
 *   height;
 * - ::cudaDevAttrMaxTexture3DWidth: Maximum 3D texture width;
 * - ::cudaDevAttrMaxTexture3DHeight: Maximum 3D texture height;
 * - ::cudaDevAttrMaxTexture3DDepth: Maximum 3D texture depth;
 * - ::cudaDevAttrMaxTexture3DWidthAlt: Alternate maximum 3D texture width,
 *   0 if no alternate maximum 3D texture size is supported;
 * - ::cudaDevAttrMaxTexture3DHeightAlt: Alternate maximum 3D texture height,
 *   0 if no alternate maximum 3D texture size is supported;
 * - ::cudaDevAttrMaxTexture3DDepthAlt: Alternate maximum 3D texture depth,
 *   0 if no alternate maximum 3D texture size is supported;
 * - ::cudaDevAttrMaxTextureCubemapWidth: Maximum cubemap texture width or
 *   height;
 * - ::cudaDevAttrMaxTexture1DLayeredWidth: Maximum 1D layered texture width;
 * - ::cudaDevAttrMaxTexture1DLayeredLayers: Maximum layers in a 1D layered
 *   texture;
 * - ::cudaDevAttrMaxTexture2DLayeredWidth: Maximum 2D layered texture width;
 * - ::cudaDevAttrMaxTexture2DLayeredHeight: Maximum 2D layered texture height;
 * - ::cudaDevAttrMaxTexture2DLayeredLayers: Maximum layers in a 2D layered
 *   texture;
 * - ::cudaDevAttrMaxTextureCubemapLayeredWidth: Maximum cubemap layered
 *   texture width or height;
 * - ::cudaDevAttrMaxTextureCubemapLayeredLayers: Maximum layers in a cubemap
 *   layered texture;
 * - ::cudaDevAttrMaxSurface1DWidth: Maximum 1D surface width;
 * - ::cudaDevAttrMaxSurface2DWidth: Maximum 2D surface width;
 * - ::cudaDevAttrMaxSurface2DHeight: Maximum 2D surface height;
 * - ::cudaDevAttrMaxSurface3DWidth: Maximum 3D surface width;
 * - ::cudaDevAttrMaxSurface3DHeight: Maximum 3D surface height;
 * - ::cudaDevAttrMaxSurface3DDepth: Maximum 3D surface depth;
 * - ::cudaDevAttrMaxSurface1DLayeredWidth: Maximum 1D layered surface width;
 * - ::cudaDevAttrMaxSurface1DLayeredLayers: Maximum layers in a 1D layered
 *   surface;
 * - ::cudaDevAttrMaxSurface2DLayeredWidth: Maximum 2D layered surface width;
 * - ::cudaDevAttrMaxSurface2DLayeredHeight: Maximum 2D layered surface height;
 * - ::cudaDevAttrMaxSurface2DLayeredLayers: Maximum layers in a 2D layered
 *   surface;
 * - ::cudaDevAttrMaxSurfaceCubemapWidth: Maximum cubemap surface width;
 * - ::cudaDevAttrMaxSurfaceCubemapLayeredWidth: Maximum cubemap layered
 *   surface width;
 * - ::cudaDevAttrMaxSurfaceCubemapLayeredLayers: Maximum layers in a cubemap
 *   layered surface;
 * - ::cudaDevAttrMaxRegistersPerBlock: Maximum number of 32-bit registers 
 *   available to a thread block;
 * - ::cudaDevAttrClockRate: Peak clock frequency in kilohertz;
 * - ::cudaDevAttrTextureAlignment: Alignment requirement; texture base
 *   addresses aligned to ::textureAlign bytes do not need an offset applied
 *   to texture fetches;
 * - ::cudaDevAttrTexturePitchAlignment: Pitch alignment requirement for 2D
 *   texture references bound to pitched memory;
 * - ::cudaDevAttrGpuOverlap: 1 if the device can concurrently copy memory
 *   between host and device while executing a kernel, or 0 if not;
 * - ::cudaDevAttrMultiProcessorCount: Number of multiprocessors on the device;
 * - ::cudaDevAttrKernelExecTimeout: 1 if there is a run time limit for kernels
 *   executed on the device, or 0 if not;
 * - ::cudaDevAttrIntegrated: 1 if the device is integrated with the memory
 *   subsystem, or 0 if not;
 * - ::cudaDevAttrCanMapHostMemory: 1 if the device can map host memory into
 *   the CUDA address space, or 0 if not;
 * - ::cudaDevAttrComputeMode: Compute mode is the compute mode that the device
 *   is currently in. Available modes are as follows:
 *   - ::cudaComputeModeDefault: Default mode - Device is not restricted and
 *     multiple threads can use ::cudaSetDevice() with this device.
 *   - ::cudaComputeModeExclusive: Compute-exclusive mode - Only one thread will
 *     be able to use ::cudaSetDevice() with this device.
 *   - ::cudaComputeModeProhibited: Compute-prohibited mode - No threads can use
 *     ::cudaSetDevice() with this device.
 *   - ::cudaComputeModeExclusiveProcess: Compute-exclusive-process mode - Many 
 *     threads in one process will be able to use ::cudaSetDevice() with this
 *     device.
 * - ::cudaDevAttrConcurrentKernels: 1 if the device supports executing
 *   multiple kernels within the same context simultaneously, or 0 if
 *   not. It is not guaranteed that multiple kernels will be resident on the
 *   device concurrently so this feature should not be relied upon for
 *   correctness;
 * - ::cudaDevAttrEccEnabled: 1 if error correction is enabled on the device,
 *   0 if error correction is disabled or not supported by the device;
 * - ::cudaDevAttrPciBusId: PCI bus identifier of the device;
 * - ::cudaDevAttrPciDeviceId: PCI device (also known as slot) identifier of
 *   the device;
 * - ::cudaDevAttrTccDriver: 1 if the device is using a TCC driver. TCC is only
 *   available on Tesla hardware running Windows Vista or later;
 * - ::cudaDevAttrMemoryClockRate: Peak memory clock frequency in kilohertz;
 * - ::cudaDevAttrGlobalMemoryBusWidth: Global memory bus width in bits;
 * - ::cudaDevAttrL2CacheSize: Size of L2 cache in bytes. 0 if the device
 *   doesn't have L2 cache;
 * - ::cudaDevAttrMaxThreadsPerMultiProcessor: Maximum resident threads per 
 *   multiprocessor;
 * - ::cudaDevAttrUnifiedAddressing: 1 if the device shares a unified address
 *   space with the host, or 0 if not;
 * - ::cudaDevAttrComputeCapabilityMajor: Major compute capability version
 *   number;
 * - ::cudaDevAttrComputeCapabilityMinor: Minor compute capability version
 *   number;
 * - ::cudaDevAttrStreamPrioritiesSupported: 1 if the device supports stream
 *   priorities, or 0 if not;
 * - ::cudaDevAttrGlobalL1CacheSupported: 1 if device supports caching globals 
 *    in L1 cache, 0 if not;
 * - ::cudaDevAttrGlobalL1CacheSupported: 1 if device supports caching locals 
 *    in L1 cache, 0 if not;
 * - ::cudaDevAttrMaxSharedMemoryPerMultiprocessor: Maximum amount of shared memory
 *   available to a multiprocessor in bytes; this amount is shared by all 
 *   thread blocks simultaneously resident on a multiprocessor;
 * - ::cudaDevAttrMaxRegistersPerMultiprocessor: Maximum number of 32-bit registers 
 *   available to a multiprocessor; this number is shared by all thread blocks
 *   simultaneously resident on a multiprocessor;
 * - ::cudaDevAttrManagedMemSupported: 1 if device supports allocating
 *   managed memory, 0 if not;
 * - ::cudaDevAttrIsMultiGpuBoard: 1 if device is on a multi-GPU board, 0 if not;
 * - ::cudaDevAttrMultiGpuBoardGroupID: Unique identifier for a group of devices on the
 *   same multi-GPU board;
 *
 * \param value  - Returned device attribute value
 * \param attr   - Device attribute to query
 * \param device - Device number to query 
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDevice,
 * ::cudaErrorInvalidValue
 * \notefnerr
 *
 * \sa ::cudaGetDeviceCount, ::cudaGetDevice, ::cudaSetDevice, ::cudaChooseDevice,
 * ::cudaGetDeviceProperties
 */
public static native @Cast("cudaError_t") int cudaDeviceGetAttribute(IntPointer value, @Cast("cudaDeviceAttr") int attr, int device);
public static native @Cast("cudaError_t") int cudaDeviceGetAttribute(IntBuffer value, @Cast("cudaDeviceAttr") int attr, int device);
public static native @Cast("cudaError_t") int cudaDeviceGetAttribute(int[] value, @Cast("cudaDeviceAttr") int attr, int device);

/**
 * \brief Select compute-device which best matches criteria
 *
 * Returns in \p *device the device which has properties that best match
 * \p *prop.
 *
 * \param device - Device with best match
 * \param prop   - Desired device properties
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \notefnerr
 *
 * \sa ::cudaGetDeviceCount, ::cudaGetDevice, ::cudaSetDevice,
 * ::cudaGetDeviceProperties
 */
public static native @Cast("cudaError_t") int cudaChooseDevice(IntPointer device, @Const cudaDeviceProp prop);
public static native @Cast("cudaError_t") int cudaChooseDevice(IntBuffer device, @Const cudaDeviceProp prop);
public static native @Cast("cudaError_t") int cudaChooseDevice(int[] device, @Const cudaDeviceProp prop);

/**
 * \brief Set device to be used for GPU executions
 *
 * Sets \p device as the current device for the calling host thread.
 * Valid device id's are 0 to (::cudaGetDeviceCount() - 1).
 *
 * Any device memory subsequently allocated from this host thread
 * using ::cudaMalloc(), ::cudaMallocPitch() or ::cudaMallocArray()
 * will be physically resident on \p device.  Any host memory allocated
 * from this host thread using ::cudaMallocHost() or ::cudaHostAlloc() 
 * or ::cudaHostRegister() will have its lifetime associated  with
 * \p device.  Any streams or events created from this host thread will 
 * be associated with \p device.  Any kernels launched from this host
 * thread using the <<<>>> operator or ::cudaLaunch() will be executed 
 * on \p device.
 *
 * This call may be made from any host thread, to any device, and at 
 * any time.  This function will do no synchronization with the previous 
 * or new device, and should be considered a very low overhead call.
 *
 * \param device - Device on which the active host thread should execute the
 * device code.
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDevice,
 * ::cudaErrorDeviceAlreadyInUse
 * \notefnerr
 *
 * \sa ::cudaGetDeviceCount, ::cudaGetDevice, ::cudaGetDeviceProperties,
 * ::cudaChooseDevice
 */
public static native @Cast("cudaError_t") int cudaSetDevice(int device);

/**
 * \brief Returns which device is currently being used
 *
 * Returns in \p *device the current device for the calling host thread.
 *
 * \param device - Returns the device on which the active host thread
 * executes the device code.
 *
 * \return
 * ::cudaSuccess
 * \notefnerr
 *
 * \sa ::cudaGetDeviceCount, ::cudaSetDevice, ::cudaGetDeviceProperties,
 * ::cudaChooseDevice
 */
public static native @Cast("cudaError_t") int cudaGetDevice(IntPointer device);
public static native @Cast("cudaError_t") int cudaGetDevice(IntBuffer device);
public static native @Cast("cudaError_t") int cudaGetDevice(int[] device);

/**
 * \brief Set a list of devices that can be used for CUDA
 *
 * Sets a list of devices for CUDA execution in priority order using
 * \p device_arr. The parameter \p len specifies the number of elements in the
 * list.  CUDA will try devices from the list sequentially until it finds one
 * that works.  If this function is not called, or if it is called with a \p len
 * of 0, then CUDA will go back to its default behavior of trying devices
 * sequentially from a default list containing all of the available CUDA
 * devices in the system. If a specified device ID in the list does not exist,
 * this function will return ::cudaErrorInvalidDevice. If \p len is not 0 and
 * \p device_arr is NULL or if \p len exceeds the number of devices in
 * the system, then ::cudaErrorInvalidValue is returned.
 *
 * \param device_arr - List of devices to try
 * \param len        - Number of devices in specified list
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidDevice
 * \notefnerr
 *
 * \sa ::cudaGetDeviceCount, ::cudaSetDevice, ::cudaGetDeviceProperties,
 * ::cudaSetDeviceFlags,
 * ::cudaChooseDevice
 */
public static native @Cast("cudaError_t") int cudaSetValidDevices(IntPointer device_arr, int len);
public static native @Cast("cudaError_t") int cudaSetValidDevices(IntBuffer device_arr, int len);
public static native @Cast("cudaError_t") int cudaSetValidDevices(int[] device_arr, int len);

/**
 * \brief Sets flags to be used for device executions
 *
 * Records \p flags as the flags to use when initializing the current 
 * device.  If no device has been made current to the calling thread
 * then \p flags will be applied to the initialization of any device
 * initialized by the calling host thread, unless that device has had
 * its initialization flags set explicitly by this or any host thread.
 * 
 * If the current device has been set and that device has already been 
 * initialized then this call will fail with the error 
 * ::cudaErrorSetOnActiveProcess.  In this case it is necessary 
 * to reset \p device using ::cudaDeviceReset() before the device's
 * initialization flags may be set.
 *
 * The two LSBs of the \p flags parameter can be used to control how the CPU
 * thread interacts with the OS scheduler when waiting for results from the
 * device.
 *
 * - ::cudaDeviceScheduleAuto: The default value if the \p flags parameter is
 * zero, uses a heuristic based on the number of active CUDA contexts in the
 * process \p C and the number of logical processors in the system \p P. If
 * \p C \> \p P, then CUDA will yield to other OS threads when waiting for the
 * device, otherwise CUDA will not yield while waiting for results and
 * actively spin on the processor.
 * - ::cudaDeviceScheduleSpin: Instruct CUDA to actively spin when waiting for
 * results from the device. This can decrease latency when waiting for the
 * device, but may lower the performance of CPU threads if they are performing
 * work in parallel with the CUDA thread.
 * - ::cudaDeviceScheduleYield: Instruct CUDA to yield its thread when waiting
 * for results from the device. This can increase latency when waiting for the
 * device, but can increase the performance of CPU threads performing work in
 * parallel with the device.
 * - ::cudaDeviceScheduleBlockingSync: Instruct CUDA to block the CPU thread 
 * on a synchronization primitive when waiting for the device to finish work.
 * - ::cudaDeviceBlockingSync: Instruct CUDA to block the CPU thread on a 
 * synchronization primitive when waiting for the device to finish work. <br>
 * \ref deprecated "Deprecated:" This flag was deprecated as of CUDA 4.0 and
 * replaced with ::cudaDeviceScheduleBlockingSync.
 * - ::cudaDeviceMapHost: This flag must be set in order to allocate pinned
 * host memory that is accessible to the device. If this flag is not set,
 * ::cudaHostGetDevicePointer() will always return a failure code.
 * - ::cudaDeviceLmemResizeToMax: Instruct CUDA to not reduce local memory
 * after resizing local memory for a kernel. This can prevent thrashing by
 * local memory allocations when launching many kernels with high local
 * memory usage at the cost of potentially increased memory usage.
 *
 * \param flags - Parameters for device operation
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDevice,
 * ::cudaErrorSetOnActiveProcess
 *
 * \sa ::cudaGetDeviceCount, ::cudaGetDevice, ::cudaGetDeviceProperties,
 * ::cudaSetDevice, ::cudaSetValidDevices,
 * ::cudaChooseDevice
 */
public static native @Cast("cudaError_t") int cudaSetDeviceFlags( @Cast("unsigned int") int flags );

/** @} */ /* END CUDART_DEVICE */

/**
 * \defgroup CUDART_STREAM Stream Management
 *
 * ___MANBRIEF___ stream management functions of the CUDA runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the stream management functions of the CUDA runtime
 * application programming interface.
 *
 * @{
 */

/**
 * \brief Create an asynchronous stream
 *
 * Creates a new asynchronous stream.
 *
 * \param pStream - Pointer to new stream identifier
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \notefnerr
 *
 * \sa ::cudaStreamCreateWithPriority,
 * ::cudaStreamCreateWithFlags,
 * ::cudaStreamGetPriority,
 * ::cudaStreamGetFlags,
 * ::cudaStreamQuery,
 * ::cudaStreamSynchronize,
 * ::cudaStreamWaitEvent,
 * ::cudaStreamAddCallback,
 * ::cudaStreamDestroy
 */
public static native @Cast("cudaError_t") int cudaStreamCreate(@ByPtrPtr cudaStream pStream);

/**
 * \brief Create an asynchronous stream
 *
 * Creates a new asynchronous stream.  The \p flags argument determines the 
 * behaviors of the stream.  Valid values for \p flags are
 * - ::cudaStreamDefault: Default stream creation flag.
 * - ::cudaStreamNonBlocking: Specifies that work running in the created 
 *   stream may run concurrently with work in stream 0 (the NULL stream), and that
 *   the created stream should perform no implicit synchronization with stream 0.
 *
 * \param pStream - Pointer to new stream identifier
 * \param flags   - Parameters for stream creation
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \notefnerr
 *
 * \sa ::cudaStreamCreate,
 * ::cudaStreamCreateWithPriority,
 * ::cudaStreamGetFlags,
 * ::cudaStreamQuery,
 * ::cudaStreamSynchronize,
 * ::cudaStreamWaitEvent,
 * ::cudaStreamAddCallback,
 * ::cudaStreamDestroy
 */
public static native @Cast("cudaError_t") int cudaStreamCreateWithFlags(@ByPtrPtr cudaStream pStream, @Cast("unsigned int") int flags);

/**
 * \brief Create an asynchronous stream with the specified priority
 *
 * Creates a stream with the specified priority and returns a handle in \p pStream.
 * This API alters the scheduler priority of work in the stream. Work in a higher
 * priority stream may preempt work already executing in a low priority stream.
 *
 * \p priority follows a convention where lower numbers represent higher priorities.
 * '0' represents default priority. The range of meaningful numerical priorities can
 * be queried using ::cudaDeviceGetStreamPriorityRange. If the specified priority is
 * outside the numerical range returned by ::cudaDeviceGetStreamPriorityRange,
 * it will automatically be clamped to the lowest or the highest number in the range.
 *
 * \param pStream  - Pointer to new stream identifier
 * \param flags    - Flags for stream creation. See ::cudaStreamCreateWithFlags for a list of valid flags that can be passed
 * \param priority - Priority of the stream. Lower numbers represent higher priorities.
 *                   See ::cudaDeviceGetStreamPriorityRange for more information about
 *                   the meaningful stream priorities that can be passed.
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \notefnerr
 *
 * \note Stream priorities are supported only on Quadro and Tesla GPUs
 * with compute capability 3.5 or higher.
 *
 * \note In the current implementation, only compute kernels launched in
 * priority streams are affected by the stream's priority. Stream priorities have
 * no effect on host-to-device and device-to-host memory operations.
 *
 * \sa ::cudaStreamCreate,
 * ::cudaStreamCreateWithFlags,
 * ::cudaDeviceGetStreamPriorityRange,
 * ::cudaStreamGetPriority,
 * ::cudaStreamQuery,
 * ::cudaStreamWaitEvent,
 * ::cudaStreamAddCallback,
 * ::cudaStreamSynchronize,
 * ::cudaStreamDestroy
 */
public static native @Cast("cudaError_t") int cudaStreamCreateWithPriority(@ByPtrPtr cudaStream pStream, @Cast("unsigned int") int flags, int priority);

/**
 * \brief Query the priority of a stream
 *
 * Query the priority of a stream. The priority is returned in in \p priority.
 * Note that if the stream was created with a priority outside the meaningful
 * numerical range returned by ::cudaDeviceGetStreamPriorityRange,
 * this function returns the clamped priority.
 * See ::cudaStreamCreateWithPriority for details about priority clamping.
 *
 * \param hStream    - Handle to the stream to be queried
 * \param priority   - Pointer to a signed integer in which the stream's priority is returned
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle
 * \notefnerr
 *
 * \sa ::cudaStreamCreateWithPriority,
 * ::cudaDeviceGetStreamPriorityRange,
 * ::cudaStreamGetFlags
 */
public static native @Cast("cudaError_t") int cudaStreamGetPriority(cudaStream hStream, IntPointer priority);
public static native @Cast("cudaError_t") int cudaStreamGetPriority(cudaStream hStream, IntBuffer priority);
public static native @Cast("cudaError_t") int cudaStreamGetPriority(cudaStream hStream, int[] priority);

/**
 * \brief Query the flags of a stream
 *
 * Query the flags of a stream. The flags are returned in \p flags.
 * See ::cudaStreamCreateWithFlags for a list of valid flags.
 *
 * \param hStream - Handle to the stream to be queried
 * \param flags   - Pointer to an unsigned integer in which the stream's flags are returned
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle
 * \notefnerr
 *
 * \sa ::cudaStreamCreateWithPriority,
 * ::cudaStreamCreateWithFlags,
 * ::cudaStreamGetPriority
 */
public static native @Cast("cudaError_t") int cudaStreamGetFlags(cudaStream hStream, @Cast("unsigned int*") IntPointer flags);
public static native @Cast("cudaError_t") int cudaStreamGetFlags(cudaStream hStream, @Cast("unsigned int*") IntBuffer flags);
public static native @Cast("cudaError_t") int cudaStreamGetFlags(cudaStream hStream, @Cast("unsigned int*") int[] flags);

/**
 * \brief Destroys and cleans up an asynchronous stream
 *
 * Destroys and cleans up the asynchronous stream specified by \p stream.
 *
 * In case the device is still doing work in the stream \p stream
 * when ::cudaStreamDestroy() is called, the function will return immediately 
 * and the resources associated with \p stream will be released automatically 
 * once the device has completed all work in \p stream.
 *
 * \param stream - Stream identifier
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidResourceHandle
 * \notefnerr
 *
 * \sa ::cudaStreamCreate, ::cudaStreamCreateWithFlags, ::cudaStreamQuery, ::cudaStreamWaitEvent, ::cudaStreamSynchronize, ::cudaStreamAddCallback
 */
public static native @Cast("cudaError_t") int cudaStreamDestroy(cudaStream stream);

/**
 * \brief Make a compute stream wait on an event
 *
 * Makes all future work submitted to \p stream wait until \p event reports
 * completion before beginning execution.  This synchronization will be
 * performed efficiently on the device.  The event \p event may
 * be from a different context than \p stream, in which case this function
 * will perform cross-device synchronization.
 *
 * The stream \p stream will wait only for the completion of the most recent
 * host call to ::cudaEventRecord() on \p event.  Once this call has returned,
 * any functions (including ::cudaEventRecord() and ::cudaEventDestroy()) may be
 * called on \p event again, and the subsequent calls will not have any effect
 * on \p stream.
 *
 * If ::cudaEventRecord() has not been called on \p event, this call acts as if
 * the record has already completed, and so is a functional no-op.
 *
 * \param stream - Stream to wait
 * \param event  - Event to wait on
 * \param flags  - Parameters for the operation (must be 0)
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidResourceHandle
 * \note_null_stream
 * \notefnerr
 *
 * \sa ::cudaStreamCreate, ::cudaStreamCreateWithFlags, ::cudaStreamQuery, ::cudaStreamSynchronize, ::cudaStreamAddCallback, ::cudaStreamDestroy
 */
public static native @Cast("cudaError_t") int cudaStreamWaitEvent(cudaStream stream, cudaEvent event, @Cast("unsigned int") int flags);

// #ifdef _WIN32
// #define CUDART_CB __stdcall
// #else
// #define CUDART_CB
// #endif

/**
 * Type of stream callback functions.
 * \param stream The stream as passed to ::cudaStreamAddCallback, may be NULL.
 * \param status ::cudaSuccess or any persistent error on the stream.
 * \param userData User parameter provided at registration.
 */
@Convention("CUDART_CB") public static class cudaStreamCallback_t extends FunctionPointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public    cudaStreamCallback_t(Pointer p) { super(p); }
    protected cudaStreamCallback_t() { allocate(); }
    private native void allocate();
    public native void call(cudaStream stream, @Cast("cudaError_t") int status, Pointer userData);
}

/**
 * \brief Add a callback to a compute stream
 *
 * Adds a callback to be called on the host after all currently enqueued
 * items in the stream have completed.  For each 
 * cudaStreamAddCallback call, a callback will be executed exactly once.
 * The callback will block later work in the stream until it is finished.
 *
 * The callback may be passed ::cudaSuccess or an error code.  In the event
 * of a device error, all subsequently executed callbacks will receive an
 * appropriate ::cudaError_t.
 *
 * Callbacks must not make any CUDA API calls.  Attempting to use CUDA APIs
 * will result in ::cudaErrorNotPermitted.  Callbacks must not perform any
 * synchronization that may depend on outstanding device work or other callbacks
 * that are not mandated to run earlier.  Callbacks without a mandated order
 * (in independent streams) execute in undefined order and may be serialized.
 *
 * This API requires compute capability 1.1 or greater.  See
 * ::cudaDeviceGetAttribute or ::cudaGetDeviceProperties to query compute
 * capability.  Calling this API with an earlier compute version will
 * return ::cudaErrorNotSupported.
 *
 * For the purposes of Unified Memory, callback execution makes a number of
 * guarantees:
 * <ul>
 *   <li>The callback stream is considered idle for the duration of the
 *   callback.  Thus, for example, a callback may always use memory attached
 *   to the callback stream.</li>
 *   <li>The start of execution of a callback has the same effect as
 *   synchronizing an event recorded in the same stream immediately prior to
 *   the callback.  It thus synchronizes streams which have been "joined"
 *   prior to the callback.</li>
 *   <li>Adding device work to any stream does not have the effect of making
 *   the stream active until all preceding callbacks have executed.  Thus, for
 *   example, a callback might use global attached memory even if work has
 *   been added to another stream, if it has been properly ordered with an
 *   event.</li>
 *   <li>Completion of a callback does not cause a stream to become
 *   active except as described above.  The callback stream will remain idle
 *   if no device work follows the callback, and will remain idle across
 *   consecutive callbacks without device work in between.  Thus, for example,
 *   stream synchronization can be done by signaling from a callback at the
 *   end of the stream.</li>
 * </ul>
 *
 * \param stream   - Stream to add callback to
 * \param callback - The function to call once preceding stream operations are complete
 * \param userData - User specified data to be passed to the callback function
 * \param flags    - Reserved for future use, must be 0
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorNotSupported
 * \note_null_stream
 * \notefnerr
 *
 * \sa ::cudaStreamCreate, ::cudaStreamCreateWithFlags, ::cudaStreamQuery, ::cudaStreamSynchronize, ::cudaStreamWaitEvent, ::cudaStreamDestroy, ::cudaMallocManaged, ::cudaStreamAttachMemAsync
 */
public static native @Cast("cudaError_t") int cudaStreamAddCallback(cudaStream stream,
        cudaStreamCallback_t callback, Pointer userData, @Cast("unsigned int") int flags);

/**
 * \brief Waits for stream tasks to complete
 *
 * Blocks until \p stream has completed all operations. If the
 * ::cudaDeviceScheduleBlockingSync flag was set for this device, 
 * the host thread will block until the stream is finished with 
 * all of its tasks.
 *
 * \param stream - Stream identifier
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidResourceHandle
 * \notefnerr
 *
 * \sa ::cudaStreamCreate, ::cudaStreamCreateWithFlags, ::cudaStreamQuery, ::cudaStreamWaitEvent, ::cudaStreamAddCallback, ::cudaStreamDestroy
 */
public static native @Cast("cudaError_t") int cudaStreamSynchronize(cudaStream stream);

/**
 * \brief Queries an asynchronous stream for completion status
 *
 * Returns ::cudaSuccess if all operations in \p stream have
 * completed, or ::cudaErrorNotReady if not.
 *
 * For the purposes of Unified Memory, a return value of ::cudaSuccess
 * is equivalent to having called ::cudaStreamSynchronize().
 *
 * \param stream - Stream identifier
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorNotReady,
 * ::cudaErrorInvalidResourceHandle
 * \notefnerr
 *
 * \sa ::cudaStreamCreate, ::cudaStreamCreateWithFlags, ::cudaStreamWaitEvent, ::cudaStreamSynchronize, ::cudaStreamAddCallback, ::cudaStreamDestroy
 */
public static native @Cast("cudaError_t") int cudaStreamQuery(cudaStream stream);

/**
 * \brief Attach memory to a stream asynchronously
 *
 * Enqueues an operation in \p stream to specify stream association of
 * \p length bytes of memory starting from \p devPtr. This function is a
 * stream-ordered operation, meaning that it is dependent on, and will
 * only take effect when, previous work in stream has completed. Any
 * previous association is automatically replaced.
 *
 * \p devPtr must point to an address within managed memory space declared
 * using the __managed__ keyword or allocated with ::cudaMallocManaged.
 *
 * \p length must be zero, to indicate that the entire allocation's
 * stream association is being changed.  Currently, it's not possible
 * to change stream association for a portion of an allocation.
 *
 * The stream association is specified using \p flags which must be
 * one of ::cudaMemAttachGlobal, ::cudaMemAttachHost or ::cudaMemAttachSingle.
 * If the ::cudaMemAttachGlobal flag is specified, the memory can be accessed
 * by any stream on any device.
 * If the ::cudaMemAttachHost flag is specified, the program makes a guarantee
 * that it won't access the memory on the device from any stream.
 * If the ::cudaMemAttachSingle flag is specified, the program makes a guarantee
 * that it will only access the memory on the device from \p stream. It is illegal
 * to attach singly to the NULL stream, because the NULL stream is a virtual global
 * stream and not a specific stream. An error will be returned in this case.
 *
 * When memory is associated with a single stream, the Unified Memory system will
 * allow CPU access to this memory region so long as all operations in \p stream
 * have completed, regardless of whether other streams are active. In effect,
 * this constrains exclusive ownership of the managed memory region by
 * an active GPU to per-stream activity instead of whole-GPU activity.
 *
 * Accessing memory on the device from streams that are not associated with
 * it will produce undefined results. No error checking is performed by the
 * Unified Memory system to ensure that kernels launched into other streams
 * do not access this region. 
 *
 * It is a program's responsibility to order calls to ::cudaStreamAttachMemAsync
 * via events, synchronization or other means to ensure legal access to memory
 * at all times. Data visibility and coherency will be changed appropriately
 * for all kernels which follow a stream-association change.
 *
 * If \p stream is destroyed while data is associated with it, the association is
 * removed and the association reverts to the default visibility of the allocation
 * as specified at ::cudaMallocManaged. For __managed__ variables, the default
 * association is always ::cudaMemAttachGlobal. Note that destroying a stream is an
 * asynchronous operation, and as a result, the change to default association won't
 * happen until all work in the stream has completed.
 *
 * \param stream  - Stream in which to enqueue the attach operation
 * \param devPtr  - Pointer to memory (must be a pointer to managed memory)
 * \param length  - Length of memory (must be zero)
 * \param flags   - Must be one of ::cudaMemAttachGlobal, ::cudaMemAttachHost or ::cudaMemAttachSingle
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorNotReady,
 * ::cudaErrorInvalidValue
 * ::cudaErrorInvalidResourceHandle
 * \notefnerr
 *
 * \sa ::cudaStreamCreate, ::cudaStreamCreateWithFlags, ::cudaStreamWaitEvent, ::cudaStreamSynchronize, ::cudaStreamAddCallback, ::cudaStreamDestroy, ::cudaMallocManaged
 */
public static native @Cast("cudaError_t") int cudaStreamAttachMemAsync(cudaStream stream, Pointer devPtr, @Cast("size_t") long length, @Cast("unsigned int") int flags);

/** @} */ /* END CUDART_STREAM */

/**
 * \defgroup CUDART_EVENT Event Management
 *
 * ___MANBRIEF___ event management functions of the CUDA runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the event management functions of the CUDA runtime
 * application programming interface.
 *
 * @{
 */

/**
 * \brief Creates an event object
 *
 * Creates an event object using ::cudaEventDefault.
 *
 * \param event - Newly created event
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInitializationError,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorLaunchFailure,
 * ::cudaErrorMemoryAllocation
 * \notefnerr
 *
 * \sa \ref ::cudaEventCreate(cudaEvent_t*, unsigned int) "cudaEventCreate (C++ API)",
 * ::cudaEventCreateWithFlags, ::cudaEventRecord, ::cudaEventQuery,
 * ::cudaEventSynchronize, ::cudaEventDestroy, ::cudaEventElapsedTime,
 * ::cudaStreamWaitEvent
 */
public static native @Cast("cudaError_t") int cudaEventCreate(@ByPtrPtr cudaEvent event);

/**
 * \brief Creates an event object with the specified flags
 *
 * Creates an event object with the specified flags. Valid flags include:
 * - ::cudaEventDefault: Default event creation flag.
 * - ::cudaEventBlockingSync: Specifies that event should use blocking
 *   synchronization. A host thread that uses ::cudaEventSynchronize() to wait
 *   on an event created with this flag will block until the event actually
 *   completes.
 * - ::cudaEventDisableTiming: Specifies that the created event does not need
 *   to record timing data.  Events created with this flag specified and
 *   the ::cudaEventBlockingSync flag not specified will provide the best
 *   performance when used with ::cudaStreamWaitEvent() and ::cudaEventQuery().
 * - ::cudaEventInterprocess: Specifies that the created event may be used as an
 *   interprocess event by ::cudaIpcGetEventHandle(). ::cudaEventInterprocess must
 *   be specified along with ::cudaEventDisableTiming.
 *
 * \param event - Newly created event
 * \param flags - Flags for new event
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInitializationError,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorLaunchFailure,
 * ::cudaErrorMemoryAllocation
 * \notefnerr
 *
 * \sa \ref ::cudaEventCreate(cudaEvent_t*) "cudaEventCreate (C API)",
 * ::cudaEventSynchronize, ::cudaEventDestroy, ::cudaEventElapsedTime,
 * ::cudaStreamWaitEvent
 */
public static native @Cast("cudaError_t") int cudaEventCreateWithFlags(@ByPtrPtr cudaEvent event, @Cast("unsigned int") int flags);

/**
 * \brief Records an event
 *
 * Records an event. See note about NULL stream behavior. Since operation
 * is asynchronous, ::cudaEventQuery() or ::cudaEventSynchronize() must
 * be used to determine when the event has actually been recorded.
 *
 * If ::cudaEventRecord() has previously been called on \p event, then this
 * call will overwrite any existing state in \p event.  Any subsequent calls
 * which examine the status of \p event will only examine the completion of
 * this most recent call to ::cudaEventRecord().
 *
 * \param event  - Event to record
 * \param stream - Stream in which to record event
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInitializationError,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorLaunchFailure
 * \note_null_stream
 * \notefnerr
 *
 * \sa \ref ::cudaEventCreate(cudaEvent_t*) "cudaEventCreate (C API)",
 * ::cudaEventCreateWithFlags, ::cudaEventQuery,
 * ::cudaEventSynchronize, ::cudaEventDestroy, ::cudaEventElapsedTime,
 * ::cudaStreamWaitEvent
 */
public static native @Cast("cudaError_t") int cudaEventRecord(cudaEvent event, cudaStream stream/*=0*/);
public static native @Cast("cudaError_t") int cudaEventRecord(cudaEvent event);

/**
 * \brief Queries an event's status
 *
 * Query the status of all device work preceding the most recent call to
 * ::cudaEventRecord() (in the appropriate compute streams, as specified by the
 * arguments to ::cudaEventRecord()).
 *
 * If this work has successfully been completed by the device, or if
 * ::cudaEventRecord() has not been called on \p event, then ::cudaSuccess is
 * returned. If this work has not yet been completed by the device then
 * ::cudaErrorNotReady is returned.
 *
 * For the purposes of Unified Memory, a return value of ::cudaSuccess
 * is equivalent to having called ::cudaEventSynchronize().
 *
 * \param event - Event to query
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorNotReady,
 * ::cudaErrorInitializationError,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorLaunchFailure
 * \notefnerr
 *
 * \sa \ref ::cudaEventCreate(cudaEvent_t*) "cudaEventCreate (C API)",
 * ::cudaEventCreateWithFlags, ::cudaEventRecord,
 * ::cudaEventSynchronize, ::cudaEventDestroy, ::cudaEventElapsedTime
 */
public static native @Cast("cudaError_t") int cudaEventQuery(cudaEvent event);

/**
 * \brief Waits for an event to complete
 *
 * Wait until the completion of all device work preceding the most recent
 * call to ::cudaEventRecord() (in the appropriate compute streams, as specified
 * by the arguments to ::cudaEventRecord()).
 *
 * If ::cudaEventRecord() has not been called on \p event, ::cudaSuccess is
 * returned immediately.
 *
 * Waiting for an event that was created with the ::cudaEventBlockingSync
 * flag will cause the calling CPU thread to block until the event has
 * been completed by the device.  If the ::cudaEventBlockingSync flag has
 * not been set, then the CPU thread will busy-wait until the event has
 * been completed by the device.
 *
 * \param event - Event to wait for
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInitializationError,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorLaunchFailure
 * \notefnerr
 *
 * \sa \ref ::cudaEventCreate(cudaEvent_t*) "cudaEventCreate (C API)",
 * ::cudaEventCreateWithFlags, ::cudaEventRecord,
 * ::cudaEventQuery, ::cudaEventDestroy, ::cudaEventElapsedTime
 */
public static native @Cast("cudaError_t") int cudaEventSynchronize(cudaEvent event);

/**
 * \brief Destroys an event object
 *
 * Destroys the event specified by \p event.
 *
 * In case \p event has been recorded but has not yet been completed
 * when ::cudaEventDestroy() is called, the function will return immediately and 
 * the resources associated with \p event will be released automatically once
 * the device has completed \p event.
 *
 * \param event - Event to destroy
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInitializationError,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorLaunchFailure
 * \notefnerr
 *
 * \sa \ref ::cudaEventCreate(cudaEvent_t*) "cudaEventCreate (C API)",
 * ::cudaEventCreateWithFlags, ::cudaEventQuery,
 * ::cudaEventSynchronize, ::cudaEventRecord, ::cudaEventElapsedTime
 */
public static native @Cast("cudaError_t") int cudaEventDestroy(cudaEvent event);

/**
 * \brief Computes the elapsed time between events
 *
 * Computes the elapsed time between two events (in milliseconds with a
 * resolution of around 0.5 microseconds).
 *
 * If either event was last recorded in a non-NULL stream, the resulting time
 * may be greater than expected (even if both used the same stream handle). This
 * happens because the ::cudaEventRecord() operation takes place asynchronously
 * and there is no guarantee that the measured latency is actually just between
 * the two events. Any number of other different stream operations could execute
 * in between the two measured events, thus altering the timing in a significant
 * way.
 *
 * If ::cudaEventRecord() has not been called on either event, then
 * ::cudaErrorInvalidResourceHandle is returned. If ::cudaEventRecord() has been
 * called on both events but one or both of them has not yet been completed
 * (that is, ::cudaEventQuery() would return ::cudaErrorNotReady on at least one
 * of the events), ::cudaErrorNotReady is returned. If either event was created
 * with the ::cudaEventDisableTiming flag, then this function will return
 * ::cudaErrorInvalidResourceHandle.
 *
 * \param ms    - Time between \p start and \p end in ms
 * \param start - Starting event
 * \param end   - Ending event
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorNotReady,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInitializationError,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorLaunchFailure
 * \notefnerr
 *
 * \sa \ref ::cudaEventCreate(cudaEvent_t*) "cudaEventCreate (C API)",
 * ::cudaEventCreateWithFlags, ::cudaEventQuery,
 * ::cudaEventSynchronize, ::cudaEventDestroy, ::cudaEventRecord
 */
public static native @Cast("cudaError_t") int cudaEventElapsedTime(FloatPointer ms, cudaEvent start, cudaEvent end);
public static native @Cast("cudaError_t") int cudaEventElapsedTime(FloatBuffer ms, cudaEvent start, cudaEvent end);
public static native @Cast("cudaError_t") int cudaEventElapsedTime(float[] ms, cudaEvent start, cudaEvent end);

/** @} */ /* END CUDART_EVENT */

/**
 * \defgroup CUDART_EXECUTION Execution Control
 *
 * ___MANBRIEF___ execution control functions of the CUDA runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the execution control functions of the CUDA runtime
 * application programming interface.
 *
 * Some functions have overloaded C++ API template versions documented separately in the
 * \ref CUDART_HIGHLEVEL "C++ API Routines" module.
 *
 * @{
 */

/**
 * \brief Configure a device-launch
 *
 * Specifies the grid and block dimensions for the device call to be executed
 * similar to the execution configuration syntax. ::cudaConfigureCall() is
 * stack based. Each call pushes data on top of an execution stack. This data
 * contains the dimension for the grid and thread blocks, together with any
 * arguments for the call.
 *
 * \param gridDim   - Grid dimensions
 * \param blockDim  - Block dimensions
 * \param sharedMem - Shared memory
 * \param stream    - Stream identifier
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidConfiguration
 * \note_null_stream
 * \notefnerr
 *
 * \sa
 * \ref ::cudaFuncSetCacheConfig(const void*, enum cudaFuncCache) "cudaFuncSetCacheConfig (C API)",
 * \ref ::cudaFuncGetAttributes(struct cudaFuncAttributes*, const void*) "cudaFuncGetAttributes (C API)",
 * \ref ::cudaLaunch(const void*) "cudaLaunch (C API)",
 * ::cudaSetDoubleForDevice,
 * ::cudaSetDoubleForHost,
 * \ref ::cudaSetupArgument(const void*, size_t, size_t) "cudaSetupArgument (C API)",
 */
public static native @Cast("cudaError_t") int cudaConfigureCall(@ByVal dim3 gridDim, @ByVal dim3 blockDim, @Cast("size_t") long sharedMem/*=0*/, cudaStream stream/*=0*/);
public static native @Cast("cudaError_t") int cudaConfigureCall(@ByVal dim3 gridDim, @ByVal dim3 blockDim);

/**
 * \brief Configure a device launch
 *
 * Pushes \p size bytes of the argument pointed to by \p arg at \p offset
 * bytes from the start of the parameter passing area, which starts at
 * offset 0. The arguments are stored in the top of the execution stack.
 * \ref ::cudaSetupArgument(const void*, size_t, size_t) "cudaSetupArgument()"
 * must be preceded by a call to ::cudaConfigureCall().
 *
 * \param arg    - Argument to push for a kernel launch
 * \param size   - Size of argument
 * \param offset - Offset in argument stack to push new arg
 *
 * \return
 * ::cudaSuccess
 * \notefnerr
 *
 * \sa ::cudaConfigureCall,
 * \ref ::cudaFuncSetCacheConfig(const void*, enum cudaFuncCache) "cudaFuncSetCacheConfig (C API)",
 * \ref ::cudaFuncGetAttributes(struct cudaFuncAttributes*, const void*) "cudaFuncGetAttributes (C API)",
 * \ref ::cudaLaunch(const void*) "cudaLaunch (C API)",
 * ::cudaSetDoubleForDevice,
 * ::cudaSetDoubleForHost,
 * \ref ::cudaSetupArgument(T, size_t) "cudaSetupArgument (C++ API)",
 */
public static native @Cast("cudaError_t") int cudaSetupArgument(@Const Pointer arg, @Cast("size_t") long size, @Cast("size_t") long offset);

/**
 * \brief Sets the preferred cache configuration for a device function
 *
 * On devices where the L1 cache and shared memory use the same hardware
 * resources, this sets through \p cacheConfig the preferred cache configuration
 * for the function specified via \p func. This is only a preference. The
 * runtime will use the requested configuration if possible, but it is free to
 * choose a different configuration if required to execute \p func.
 *
 * \p func is a device function symbol and must be declared as a
 * \c __global__ function. If the specified function does not exist,
 * then ::cudaErrorInvalidDeviceFunction is returned.
 *
 * This setting does nothing on devices where the size of the L1 cache and
 * shared memory are fixed.
 *
 * Launching a kernel with a different preference than the most recent
 * preference setting may insert a device-side synchronization point.
 *
 * The supported cache configurations are:
 * - ::cudaFuncCachePreferNone: no preference for shared memory or L1 (default)
 * - ::cudaFuncCachePreferShared: prefer larger shared memory and smaller L1 cache
 * - ::cudaFuncCachePreferL1: prefer larger L1 cache and smaller shared memory
 * - ::cudaFuncCachePreferEqual: prefer equal size L1 cache and shared memory
 *
 * \param func        - Device function symbol
 * \param cacheConfig - Requested cache configuration
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInitializationError,
 * ::cudaErrorInvalidDeviceFunction
 * \notefnerr
 * \note_string_api_deprecation2
 *
 * \sa ::cudaConfigureCall,
 * \ref ::cudaFuncSetCacheConfig(T*, enum cudaFuncCache) "cudaFuncSetCacheConfig (C++ API)",
 * \ref ::cudaFuncGetAttributes(struct cudaFuncAttributes*, const void*) "cudaFuncGetAttributes (C API)",
 * \ref ::cudaLaunch(const void*) "cudaLaunch (C API)",
 * ::cudaSetDoubleForDevice,
 * ::cudaSetDoubleForHost,
 * \ref ::cudaSetupArgument(const void*, size_t, size_t) "cudaSetupArgument (C API)",
 * ::cudaThreadGetCacheConfig,
 * ::cudaThreadSetCacheConfig
 */
public static native @Cast("cudaError_t") int cudaFuncSetCacheConfig(@Const Pointer func, @Cast("cudaFuncCache") int cacheConfig);

/**
 * \brief Sets the shared memory configuration for a device function
 *
 * On devices with configurable shared memory banks, this function will 
 * force all subsequent launches of the specified device function to have
 * the given shared memory bank size configuration. On any given launch of the
 * function, the shared memory configuration of the device will be temporarily
 * changed if needed to suit the function's preferred configuration. Changes in
 * shared memory configuration between subsequent launches of functions, 
 * may introduce a device side synchronization point.
 *
 * Any per-function setting of shared memory bank size set via 
 * ::cudaFuncSetSharedMemConfig will override the device wide setting set by
 * ::cudaDeviceSetSharedMemConfig.
 *
 * Changing the shared memory bank size will not increase shared memory usage
 * or affect occupancy of kernels, but may have major effects on performance. 
 * Larger bank sizes will allow for greater potential bandwidth to shared memory,
 * but will change what kinds of accesses to shared memory will result in bank 
 * conflicts.
 *
 * This function will do nothing on devices with fixed shared memory bank size.
 *
 * The supported bank configurations are:
 * - ::cudaSharedMemBankSizeDefault: use the device's shared memory configuration
 *   when launching this function.
 * - ::cudaSharedMemBankSizeFourByte: set shared memory bank width to be 
 *   four bytes natively when launching this function.
 * - ::cudaSharedMemBankSizeEightByte: set shared memory bank width to be eight 
 *   bytes natively when launching this function.
 *
 * \param func   - Device function symbol
 * \param config - Requested shared memory configuration
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInitializationError,
 * ::cudaErrorInvalidDeviceFunction,
 * ::cudaErrorInvalidValue,
 * \notefnerr
 * \note_string_api_deprecation2
 *
 * \sa ::cudaConfigureCall,
 * ::cudaDeviceSetSharedMemConfig,
 * ::cudaDeviceGetSharedMemConfig,
 * ::cudaDeviceSetCacheConfig,
 * ::cudaDeviceGetCacheConfig,
 * ::cudaFuncSetCacheConfig
 */
public static native @Cast("cudaError_t") int cudaFuncSetSharedMemConfig(@Const Pointer func, @Cast("cudaSharedMemConfig") int config);

/**
 * \brief Launches a device function
 *
 * Launches the function \p func on the device. The parameter \p func must
 * be a device function symbol. The parameter specified by \p func must be
 * declared as a \p __global__ function.
 * \ref ::cudaLaunch(const void*) "cudaLaunch()" must be preceded by a call to
 * ::cudaConfigureCall() since it pops the data that was pushed by
 * ::cudaConfigureCall() from the execution stack.
 *
 * \param func - Device function symbol
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDeviceFunction,
 * ::cudaErrorInvalidConfiguration,
 * ::cudaErrorLaunchFailure,
 * ::cudaErrorLaunchTimeout,
 * ::cudaErrorLaunchOutOfResources,
 * ::cudaErrorSharedObjectInitFailed
 * \notefnerr
 * \note_string_api_deprecation_50
 *
 * \sa ::cudaConfigureCall,
 * \ref ::cudaFuncSetCacheConfig(const void*, enum cudaFuncCache) "cudaFuncSetCacheConfig (C API)",
 * \ref ::cudaFuncGetAttributes(struct cudaFuncAttributes*, const void*) "cudaFuncGetAttributes (C API)",
 * \ref ::cudaLaunch(T*) "cudaLaunch (C++ API)",
 * ::cudaSetDoubleForDevice,
 * ::cudaSetDoubleForHost,
 * \ref ::cudaSetupArgument(const void*, size_t, size_t) "cudaSetupArgument (C API)",
 * ::cudaThreadGetCacheConfig,
 * ::cudaThreadSetCacheConfig
 */
public static native @Cast("cudaError_t") int cudaLaunch(@Const Pointer func);

/**
 * \brief Find out attributes for a given function
 *
 * This function obtains the attributes of a function specified via \p func.
 * \p func is a device function symbol and must be declared as a
 * \c __global__ function. The fetched attributes are placed in \p attr.
 * If the specified function does not exist, then
 * ::cudaErrorInvalidDeviceFunction is returned.
 *
 * Note that some function attributes such as
 * \ref ::cudaFuncAttributes::maxThreadsPerBlock "maxThreadsPerBlock"
 * may vary based on the device that is currently being used.
 *
 * \param attr - Return pointer to function's attributes
 * \param func - Device function symbol
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInitializationError,
 * ::cudaErrorInvalidDeviceFunction
 * \notefnerr
 * \note_string_api_deprecation2
 *
 * \sa ::cudaConfigureCall,
 * \ref ::cudaFuncSetCacheConfig(const void*, enum cudaFuncCache) "cudaFuncSetCacheConfig (C API)",
 * \ref ::cudaFuncGetAttributes(struct cudaFuncAttributes*, T*) "cudaFuncGetAttributes (C++ API)",
 * \ref ::cudaLaunch(const void*) "cudaLaunch (C API)",
 * ::cudaSetDoubleForDevice,
 * ::cudaSetDoubleForHost,
 * \ref ::cudaSetupArgument(const void*, size_t, size_t) "cudaSetupArgument (C API)"
 */
public static native @Cast("cudaError_t") int cudaFuncGetAttributes(cudaFuncAttributes attr, @Const Pointer func);

/**
 * \brief Converts a double argument to be executed on a device
 *
 * \param d - Double to convert
 *
 * Converts the double value of \p d to an internal float representation if
 * the device does not support double arithmetic. If the device does natively
 * support doubles, then this function does nothing.
 *
 * \return
 * ::cudaSuccess
 * \notefnerr
 *
 * \sa ::cudaConfigureCall,
 * \ref ::cudaFuncSetCacheConfig(const void*, enum cudaFuncCache) "cudaFuncSetCacheConfig (C API)",
 * \ref ::cudaFuncGetAttributes(struct cudaFuncAttributes*, const void*) "cudaFuncGetAttributes (C API)",
 * \ref ::cudaLaunch(const void*) "cudaLaunch (C API)",
 * ::cudaSetDoubleForHost,
 * \ref ::cudaSetupArgument(const void*, size_t, size_t) "cudaSetupArgument (C API)"
 */
public static native @Cast("cudaError_t") int cudaSetDoubleForDevice(DoublePointer d);
public static native @Cast("cudaError_t") int cudaSetDoubleForDevice(DoubleBuffer d);
public static native @Cast("cudaError_t") int cudaSetDoubleForDevice(double[] d);

/**
 * \brief Converts a double argument after execution on a device
 *
 * Converts the double value of \p d from a potentially internal float
 * representation if the device does not support double arithmetic. If the
 * device does natively support doubles, then this function does nothing.
 *
 * \param d - Double to convert
 *
 * \return
 * ::cudaSuccess
 * \notefnerr
 *
 * \sa ::cudaConfigureCall,
 * \ref ::cudaFuncSetCacheConfig(const void*, enum cudaFuncCache) "cudaFuncSetCacheConfig (C API)",
 * \ref ::cudaFuncGetAttributes(struct cudaFuncAttributes*, const void*) "cudaFuncGetAttributes (C API)",
 * \ref ::cudaLaunch(const void*) "cudaLaunch (C API)",
 * ::cudaSetDoubleForDevice,
 * \ref ::cudaSetupArgument(const void*, size_t, size_t) "cudaSetupArgument (C API)"
 */
public static native @Cast("cudaError_t") int cudaSetDoubleForHost(DoublePointer d);
public static native @Cast("cudaError_t") int cudaSetDoubleForHost(DoubleBuffer d);
public static native @Cast("cudaError_t") int cudaSetDoubleForHost(double[] d);

/** @} */ /* END CUDART_EXECUTION */

// #if CUDART_VERSION >= 6050

/**
 * \defgroup CUDART_OCCUPANCY Occupancy
 *
 * ___MANBRIEF___ occupancy calculation functions of the CUDA runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the occupancy calculation functions of the CUDA runtime
 * application programming interface.
 *
 * Besides the occupancy calculator function
 * (\ref ::cudaOccupancyMaxActiveBlocksPerMultiprocessor), there are also C++ only
 * occupancy-based launch configuration functions documented in
 * \ref CUDART_HIGHLEVEL "C++ API Routines" module.
 *
 * See
 * \ref ::cudaOccupancyMaxPotentialBlockSize(int*, int*, T, size_t, int) "cudaOccupancyMaxPotentialBlockSize (C++ API)"
 * and
 * \ref ::cudaOccupancyMaxPotentialBlockSizeVariableSMem(int*, int*, T, UnaryFunction, int) "cudaOccupancyMaxPotentialBlockSizeVariableSMem (C++ API)"
 *
 * @{
 */

/**
 * \brief Returns occupancy for a device function
 *
 * Returns in \p *numBlocks the maximum number of active blocks per
 * streaming multiprocessor for the device function.
 *
 * \param numBlocks       - Returned occupancy
 * \param func            - Kernel function for which occupancy is calulated
 * \param blockSize       - Block size the kernel is intended to be launched with
 * \param dynamicSMemSize  - Per-block dynamic shared memory usage intended, in bytes
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorCudartUnloading,
 * ::cudaErrorInitializationError,
 * ::cudaErrorInvalidDevice,
 * ::cudaErrorInvalidDeviceFunction,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorUnknown,
 * \notefnerr
 *
 * \sa ::cudaOccupancyMaxPotentialBlockSize,
 * \ref ::cudaOccupancyMaxPotentialBlockSize(int*, int*, T, size_t, int) "cudaOccupancyMaxPotentialBlockSize (C++ API)",
 * \ref ::cudaOccupancyMaxPotentialBlockSizeVariableSMem(int*, int*, T, UnaryFunction, int) "cudaOccupancyMaxPotentialBlockSizeVariableSMem (C++ API)"
 */
public static native @Cast("cudaError_t") int cudaOccupancyMaxActiveBlocksPerMultiprocessor(IntPointer numBlocks, @Const Pointer func, int blockSize, @Cast("size_t") long dynamicSMemSize);
public static native @Cast("cudaError_t") int cudaOccupancyMaxActiveBlocksPerMultiprocessor(IntBuffer numBlocks, @Const Pointer func, int blockSize, @Cast("size_t") long dynamicSMemSize);
public static native @Cast("cudaError_t") int cudaOccupancyMaxActiveBlocksPerMultiprocessor(int[] numBlocks, @Const Pointer func, int blockSize, @Cast("size_t") long dynamicSMemSize);

/** @} */ /* END CUDA_OCCUPANCY */
// #endif /* CUDART_VERSION >= 6050 */

/**
 * \defgroup CUDART_MEMORY Memory Management
 *
 * ___MANBRIEF___ memory management functions of the CUDA runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the memory management functions of the CUDA runtime
 * application programming interface.
 *
 * Some functions have overloaded C++ API template versions documented separately in the
 * \ref CUDART_HIGHLEVEL "C++ API Routines" module.
 *
 * @{
 */

/**
 * \brief Allocates memory that will be automatically managed by the Unified Memory system
 *
 * Allocates \p size bytes of managed memory on the device and returns in
 * \p *devPtr a pointer to the allocated memory. If the device doesn't support
 * allocating managed memory, ::cudaErrorNotSupported is returned. Support
 * for managed memory can be queried using the device attribute
 * ::cudaDevAttrManagedMemory. The allocated memory is suitably
 * aligned for any kind of variable. The memory is not cleared. If \p size
 * is 0, ::cudaMallocManaged returns ::cudaErrorInvalidValue. The pointer
 * is valid on the CPU and on all GPUs in the system that support managed memory.
 * All accesses to this pointer must obey the Unified Memory programming model.
 *
 * \p flags specifies the default stream association for this allocation.
 * \p flags must be one of ::cudaMemAttachGlobal or ::cudaMemAttachHost.
 * If ::cudaMemAttachGlobal is specified, then this memory is accessible from
 * any stream on any device. If ::cudaMemAttachHost is specified, then the
 * allocation is created with initial visibility restricted to host access only;
 * an explicit call to ::cudaStreamAttachMemAsync will be required to enable access
 * on the device.
 *
 * If the association is later changed via ::cudaStreamAttachMemAsync to
 * a single stream, the default association, as specifed during ::cudaMallocManaged,
 * is restored when that stream is destroyed. For __managed__ variables, the
 * default association is always ::cudaMemAttachGlobal. Note that destroying a
 * stream is an asynchronous operation, and as a result, the change to default
 * association won't happen until all work in the stream has completed.
 *
 * Memory allocated with ::cudaMallocManaged should be released with ::cudaFree.
 *
 * On a multi-GPU system with peer-to-peer support, where multiple GPUs support
 * managed memory, the physical storage is created on the GPU which is active
 * at the time ::cudaMallocManaged is called. All other GPUs will reference the
 * data at reduced bandwidth via peer mappings over the PCIe bus. The Unified
 * Memory management system does not migrate memory between GPUs.
 *
 * On a multi-GPU system where multiple GPUs support managed memory, but not
 * all pairs of such GPUs have peer-to-peer support between them, the physical
 * storage is created in 'zero-copy' or system memory. All GPUs will reference
 * the data at reduced bandwidth over the PCIe bus. In these circumstances,
 * use of the environment variable, CUDA_VISIBLE_DEVICES, is recommended to
 * restrict CUDA to only use those GPUs that have peer-to-peer support.
 * Alternatively, users can also set CUDA_MANAGED_FORCE_DEVICE_ALLOC to a non-zero
 * value to force the driver to always use device memory for physical storage.
 * When this environment variable is set to a non-zero value, all devices used in
 * that process that support managed memory have to be peer-to-peer compatible
 * with each other. The error ::cudaErrorInvalidDevice will be returned if a device
 * that supports managed memory is used and it is not peer-to-peer compatible with
 * any of the other managed memory supporting devices that were previously used in
 * that process, even if ::cudaDeviceReset has been called on those devices. These
 * environment variables are described in the CUDA programming guide under the
 * "CUDA environment variables" section.
 *
 * \param devPtr - Pointer to allocated device memory
 * \param size   - Requested allocation size in bytes
 * \param flags  - Must be either ::cudaMemAttachGlobal or ::cudaMemAttachHost
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorMemoryAllocation
 * ::cudaErrorNotSupported
 * ::cudaErrorInvalidValue
 *
 * \sa ::cudaMallocPitch, ::cudaFree, ::cudaMallocArray, ::cudaFreeArray,
 * ::cudaMalloc3D, ::cudaMalloc3DArray,
 * \ref ::cudaMallocHost(void**, size_t) "cudaMallocHost (C API)",
 * ::cudaFreeHost, ::cudaHostAlloc, ::cudaDeviceGetAttribute, ::cudaStreamAttachMemAsync
 */
public static native @Cast("cudaError_t") int cudaMallocManaged(@Cast("void**") PointerPointer devPtr, @Cast("size_t") long size, @Cast("unsigned int") int flags);
public static native @Cast("cudaError_t") int cudaMallocManaged(@Cast("void**") @ByPtrPtr Pointer devPtr, @Cast("size_t") long size, @Cast("unsigned int") int flags);


/**
 * \brief Allocate memory on the device
 *
 * Allocates \p size bytes of linear memory on the device and returns in
 * \p *devPtr a pointer to the allocated memory. The allocated memory is
 * suitably aligned for any kind of variable. The memory is not cleared.
 * ::cudaMalloc() returns ::cudaErrorMemoryAllocation in case of failure.
 *
 * \param devPtr - Pointer to allocated device memory
 * \param size   - Requested allocation size in bytes
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorMemoryAllocation
 *
 * \sa ::cudaMallocPitch, ::cudaFree, ::cudaMallocArray, ::cudaFreeArray,
 * ::cudaMalloc3D, ::cudaMalloc3DArray,
 * \ref ::cudaMallocHost(void**, size_t) "cudaMallocHost (C API)",
 * ::cudaFreeHost, ::cudaHostAlloc
 */
public static native @Cast("cudaError_t") int cudaMalloc(@Cast("void**") PointerPointer devPtr, @Cast("size_t") long size);
public static native @Cast("cudaError_t") int cudaMalloc(@Cast("void**") @ByPtrPtr Pointer devPtr, @Cast("size_t") long size);

/**
 * \brief Allocates page-locked memory on the host
 *
 * Allocates \p size bytes of host memory that is page-locked and accessible
 * to the device. The driver tracks the virtual memory ranges allocated with
 * this function and automatically accelerates calls to functions such as
 * ::cudaMemcpy*(). Since the memory can be accessed directly by the device,
 * it can be read or written with much higher bandwidth than pageable memory
 * obtained with functions such as ::malloc(). Allocating excessive amounts of
 * memory with ::cudaMallocHost() may degrade system performance, since it
 * reduces the amount of memory available to the system for paging. As a
 * result, this function is best used sparingly to allocate staging areas for
 * data exchange between host and device.
 *
 * \param ptr  - Pointer to allocated host memory
 * \param size - Requested allocation size in bytes
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorMemoryAllocation
 * \notefnerr
 *
 * \sa ::cudaMalloc, ::cudaMallocPitch, ::cudaMallocArray, ::cudaMalloc3D,
 * ::cudaMalloc3DArray, ::cudaHostAlloc, ::cudaFree, ::cudaFreeArray,
 * \ref ::cudaMallocHost(void**, size_t, unsigned int) "cudaMallocHost (C++ API)",
 * ::cudaFreeHost, ::cudaHostAlloc
 */
public static native @Cast("cudaError_t") int cudaMallocHost(@Cast("void**") PointerPointer ptr, @Cast("size_t") long size);
public static native @Cast("cudaError_t") int cudaMallocHost(@Cast("void**") @ByPtrPtr Pointer ptr, @Cast("size_t") long size);

/**
 * \brief Allocates pitched memory on the device
 *
 * Allocates at least \p width (in bytes) * \p height bytes of linear memory
 * on the device and returns in \p *devPtr a pointer to the allocated memory.
 * The function may pad the allocation to ensure that corresponding pointers
 * in any given row will continue to meet the alignment requirements for
 * coalescing as the address is updated from row to row. The pitch returned in
 * \p *pitch by ::cudaMallocPitch() is the width in bytes of the allocation.
 * The intended usage of \p pitch is as a separate parameter of the allocation,
 * used to compute addresses within the 2D array. Given the row and column of
 * an array element of type \p T, the address is computed as:
 * \code
    T* pElement = (T*)((char*)BaseAddress + Row * pitch) + Column;
   \endcode
 *
 * For allocations of 2D arrays, it is recommended that programmers consider
 * performing pitch allocations using ::cudaMallocPitch(). Due to pitch
 * alignment restrictions in the hardware, this is especially true if the
 * application will be performing 2D memory copies between different regions
 * of device memory (whether linear memory or CUDA arrays).
 *
 * \param devPtr - Pointer to allocated pitched device memory
 * \param pitch  - Pitch for allocation
 * \param width  - Requested pitched allocation width (in bytes)
 * \param height - Requested pitched allocation height
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorMemoryAllocation
 * \notefnerr
 *
 * \sa ::cudaMalloc, ::cudaFree, ::cudaMallocArray, ::cudaFreeArray,
 * \ref ::cudaMallocHost(void**, size_t) "cudaMallocHost (C API)",
 * ::cudaFreeHost, ::cudaMalloc3D, ::cudaMalloc3DArray,
 * ::cudaHostAlloc
 */
public static native @Cast("cudaError_t") int cudaMallocPitch(@Cast("void**") PointerPointer devPtr, @Cast("size_t*") SizeTPointer pitch, @Cast("size_t") long width, @Cast("size_t") long height);
public static native @Cast("cudaError_t") int cudaMallocPitch(@Cast("void**") @ByPtrPtr Pointer devPtr, @Cast("size_t*") SizeTPointer pitch, @Cast("size_t") long width, @Cast("size_t") long height);

/**
 * \brief Allocate an array on the device
 *
 * Allocates a CUDA array according to the ::cudaChannelFormatDesc structure
 * \p desc and returns a handle to the new CUDA array in \p *array.
 *
 * The ::cudaChannelFormatDesc is defined as:
 * \code
    struct cudaChannelFormatDesc {
        int x, y, z, w;
    enum cudaChannelFormatKind f;
    };
    \endcode
 * where ::cudaChannelFormatKind is one of ::cudaChannelFormatKindSigned,
 * ::cudaChannelFormatKindUnsigned, or ::cudaChannelFormatKindFloat.
 *
 * The \p flags parameter enables different options to be specified that affect
 * the allocation, as follows.
 * - ::cudaArrayDefault: This flag's value is defined to be 0 and provides default array allocation
 * - ::cudaArraySurfaceLoadStore: Allocates an array that can be read from or written to using a surface reference
 * - ::cudaArrayTextureGather: This flag indicates that texture gather operations will be performed on the array.
 *
 * \p width and \p height must meet certain size requirements. See ::cudaMalloc3DArray() for more details.
 *
 * \param array  - Pointer to allocated array in device memory
 * \param desc   - Requested channel format
 * \param width  - Requested array allocation width
 * \param height - Requested array allocation height
 * \param flags  - Requested properties of allocated array
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorMemoryAllocation
 * \notefnerr
 *
 * \sa ::cudaMalloc, ::cudaMallocPitch, ::cudaFree, ::cudaFreeArray,
 * \ref ::cudaMallocHost(void**, size_t) "cudaMallocHost (C API)",
 * ::cudaFreeHost, ::cudaMalloc3D, ::cudaMalloc3DArray,
 * ::cudaHostAlloc
 */
public static native @Cast("cudaError_t") int cudaMallocArray(@ByPtrPtr cudaArray array, @Const cudaChannelFormatDesc desc, @Cast("size_t") long width, @Cast("size_t") long height/*=0*/, @Cast("unsigned int") int flags/*=0*/);
public static native @Cast("cudaError_t") int cudaMallocArray(@ByPtrPtr cudaArray array, @Const cudaChannelFormatDesc desc, @Cast("size_t") long width);

/**
 * \brief Frees memory on the device
 *
 * Frees the memory space pointed to by \p devPtr, which must have been
 * returned by a previous call to ::cudaMalloc() or ::cudaMallocPitch().
 * Otherwise, or if ::cudaFree(\p devPtr) has already been called before,
 * an error is returned. If \p devPtr is 0, no operation is performed.
 * ::cudaFree() returns ::cudaErrorInvalidDevicePointer in case of failure.
 *
 * \param devPtr - Device pointer to memory to free
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDevicePointer,
 * ::cudaErrorInitializationError
 * \notefnerr
 *
 * \sa ::cudaMalloc, ::cudaMallocPitch, ::cudaMallocArray, ::cudaFreeArray,
 * \ref ::cudaMallocHost(void**, size_t) "cudaMallocHost (C API)",
 * ::cudaFreeHost, ::cudaMalloc3D, ::cudaMalloc3DArray,
 * ::cudaHostAlloc
 */
public static native @Cast("cudaError_t") int cudaFree(Pointer devPtr);

/**
 * \brief Frees page-locked memory
 *
 * Frees the memory space pointed to by \p hostPtr, which must have been
 * returned by a previous call to ::cudaMallocHost() or ::cudaHostAlloc().
 *
 * \param ptr - Pointer to memory to free
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInitializationError
 * \notefnerr
 *
 * \sa ::cudaMalloc, ::cudaMallocPitch, ::cudaFree, ::cudaMallocArray,
 * ::cudaFreeArray,
 * \ref ::cudaMallocHost(void**, size_t) "cudaMallocHost (C API)",
 * ::cudaMalloc3D, ::cudaMalloc3DArray, ::cudaHostAlloc
 */
public static native @Cast("cudaError_t") int cudaFreeHost(Pointer ptr);

/**
 * \brief Frees an array on the device
 *
 * Frees the CUDA array \p array, which must have been * returned by a
 * previous call to ::cudaMallocArray(). If ::cudaFreeArray(\p array) has
 * already been called before, ::cudaErrorInvalidValue is returned. If
 * \p devPtr is 0, no operation is performed.
 *
 * \param array - Pointer to array to free
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInitializationError
 * \notefnerr
 *
 * \sa ::cudaMalloc, ::cudaMallocPitch, ::cudaFree, ::cudaMallocArray,
 * \ref ::cudaMallocHost(void**, size_t) "cudaMallocHost (C API)",
 * ::cudaFreeHost, ::cudaHostAlloc
 */
public static native @Cast("cudaError_t") int cudaFreeArray(cudaArray array);

/**
 * \brief Frees a mipmapped array on the device
 *
 * Frees the CUDA mipmapped array \p mipmappedArray, which must have been 
 * returned by a previous call to ::cudaMallocMipmappedArray(). 
 * If ::cudaFreeMipmappedArray(\p mipmappedArray) has already been called before,
 * ::cudaErrorInvalidValue is returned.
 *
 * \param mipmappedArray - Pointer to mipmapped array to free
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInitializationError
 * \notefnerr
 *
 * \sa ::cudaMalloc, ::cudaMallocPitch, ::cudaFree, ::cudaMallocArray,
 * \ref ::cudaMallocHost(void**, size_t) "cudaMallocHost (C API)",
 * ::cudaFreeHost, ::cudaHostAlloc
 */
public static native @Cast("cudaError_t") int cudaFreeMipmappedArray(cudaMipmappedArray mipmappedArray);


/**
 * \brief Allocates page-locked memory on the host
 *
 * Allocates \p size bytes of host memory that is page-locked and accessible
 * to the device. The driver tracks the virtual memory ranges allocated with
 * this function and automatically accelerates calls to functions such as
 * ::cudaMemcpy(). Since the memory can be accessed directly by the device, it
 * can be read or written with much higher bandwidth than pageable memory
 * obtained with functions such as ::malloc(). Allocating excessive amounts of
 * pinned memory may degrade system performance, since it reduces the amount
 * of memory available to the system for paging. As a result, this function is
 * best used sparingly to allocate staging areas for data exchange between host
 * and device.
 *
 * The \p flags parameter enables different options to be specified that affect
 * the allocation, as follows.
 * - ::cudaHostAllocDefault: This flag's value is defined to be 0 and causes
 * ::cudaHostAlloc() to emulate ::cudaMallocHost().
 * - ::cudaHostAllocPortable: The memory returned by this call will be
 * considered as pinned memory by all CUDA contexts, not just the one that
 * performed the allocation.
 * - ::cudaHostAllocMapped: Maps the allocation into the CUDA address space.
 * The device pointer to the memory may be obtained by calling
 * ::cudaHostGetDevicePointer().
 * - ::cudaHostAllocWriteCombined: Allocates the memory as write-combined (WC).
 * WC memory can be transferred across the PCI Express bus more quickly on some
 * system configurations, but cannot be read efficiently by most CPUs.  WC
 * memory is a good option for buffers that will be written by the CPU and read
 * by the device via mapped pinned memory or host->device transfers.
 *
 * All of these flags are orthogonal to one another: a developer may allocate
 * memory that is portable, mapped and/or write-combined with no restrictions.
 *
 * ::cudaSetDeviceFlags() must have been called with the ::cudaDeviceMapHost
 * flag in order for the ::cudaHostAllocMapped flag to have any effect.
 *
 * The ::cudaHostAllocMapped flag may be specified on CUDA contexts for devices
 * that do not support mapped pinned memory. The failure is deferred to
 * ::cudaHostGetDevicePointer() because the memory may be mapped into other
 * CUDA contexts via the ::cudaHostAllocPortable flag.
 *
 * Memory allocated by this function must be freed with ::cudaFreeHost().
 *
 * \param pHost - Device pointer to allocated memory
 * \param size  - Requested allocation size in bytes
 * \param flags - Requested properties of allocated memory
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorMemoryAllocation
 * \notefnerr
 *
 * \sa ::cudaSetDeviceFlags,
 * \ref ::cudaMallocHost(void**, size_t) "cudaMallocHost (C API)",
 * ::cudaFreeHost
 */
public static native @Cast("cudaError_t") int cudaHostAlloc(@Cast("void**") PointerPointer pHost, @Cast("size_t") long size, @Cast("unsigned int") int flags);
public static native @Cast("cudaError_t") int cudaHostAlloc(@Cast("void**") @ByPtrPtr Pointer pHost, @Cast("size_t") long size, @Cast("unsigned int") int flags);

/**
 * \brief Registers an existing host memory range for use by CUDA
 *
 * Page-locks the memory range specified by \p ptr and \p size and maps it
 * for the device(s) as specified by \p flags. This memory range also is added
 * to the same tracking mechanism as ::cudaHostAlloc() to automatically accelerate
 * calls to functions such as ::cudaMemcpy(). Since the memory can be accessed 
 * directly by the device, it can be read or written with much higher bandwidth 
 * than pageable memory that has not been registered.  Page-locking excessive
 * amounts of memory may degrade system performance, since it reduces the amount
 * of memory available to the system for paging. As a result, this function is
 * best used sparingly to register staging areas for data exchange between
 * host and device.
 *
 * The \p flags parameter enables different options to be specified that
 * affect the allocation, as follows.
 *
 * - ::cudaHostRegisterPortable: The memory returned by this call will be
 *   considered as pinned memory by all CUDA contexts, not just the one that
 *   performed the allocation.
 *
 * - ::cudaHostRegisterMapped: Maps the allocation into the CUDA address
 *   space. The device pointer to the memory may be obtained by calling
 *   ::cudaHostGetDevicePointer(). This feature is available only on GPUs
 *   with compute capability greater than or equal to 1.1.
 *
 * All of these flags are orthogonal to one another: a developer may page-lock
 * memory that is portable or mapped with no restrictions.
 *
 * The CUDA context must have been created with the ::cudaMapHost flag in
 * order for the ::cudaHostRegisterMapped flag to have any effect.
 *
 * The ::cudaHostRegisterMapped flag may be specified on CUDA contexts for
 * devices that do not support mapped pinned memory. The failure is deferred
 * to ::cudaHostGetDevicePointer() because the memory may be mapped into
 * other CUDA contexts via the ::cudaHostRegisterPortable flag.
 *
 * The memory page-locked by this function must be unregistered with ::cudaHostUnregister().
 *
 * \param ptr   - Host pointer to memory to page-lock
 * \param size  - Size in bytes of the address range to page-lock in bytes
 * \param flags - Flags for allocation request
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorMemoryAllocation
 * \notefnerr
 *
 * \sa ::cudaHostUnregister, ::cudaHostGetFlags, ::cudaHostGetDevicePointer
 */
public static native @Cast("cudaError_t") int cudaHostRegister(Pointer ptr, @Cast("size_t") long size, @Cast("unsigned int") int flags);

/**
 * \brief Unregisters a memory range that was registered with cudaHostRegister
 *
 * Unmaps the memory range whose base address is specified by \p ptr, and makes
 * it pageable again.
 *
 * The base address must be the same one specified to ::cudaHostRegister().
 *
 * \param ptr - Host pointer to memory to unregister
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \notefnerr
 *
 * \sa ::cudaHostUnregister
 */
public static native @Cast("cudaError_t") int cudaHostUnregister(Pointer ptr);

/**
 * \brief Passes back device pointer of mapped host memory allocated by
 * cudaHostAlloc or registered by cudaHostRegister
 *
 * Passes back the device pointer corresponding to the mapped, pinned host
 * buffer allocated by ::cudaHostAlloc() or registered by ::cudaHostRegister().
 *
 * ::cudaHostGetDevicePointer() will fail if the ::cudaDeviceMapHost flag was
 * not specified before deferred context creation occurred, or if called on a
 * device that does not support mapped, pinned memory.
 *
 * \p flags provides for future releases.  For now, it must be set to 0.
 *
 * \param pDevice - Returned device pointer for mapped memory
 * \param pHost   - Requested host pointer mapping
 * \param flags   - Flags for extensions (must be 0 for now)
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorMemoryAllocation
 * \notefnerr
 *
 * \sa ::cudaSetDeviceFlags, ::cudaHostAlloc
 */
public static native @Cast("cudaError_t") int cudaHostGetDevicePointer(@Cast("void**") PointerPointer pDevice, Pointer pHost, @Cast("unsigned int") int flags);
public static native @Cast("cudaError_t") int cudaHostGetDevicePointer(@Cast("void**") @ByPtrPtr Pointer pDevice, Pointer pHost, @Cast("unsigned int") int flags);

/**
 * \brief Passes back flags used to allocate pinned host memory allocated by
 * cudaHostAlloc
 *
 * ::cudaHostGetFlags() will fail if the input pointer does not
 * reside in an address range allocated by ::cudaHostAlloc().
 *
 * \param pFlags - Returned flags word
 * \param pHost - Host pointer
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \notefnerr
 *
 * \sa ::cudaHostAlloc
 */
public static native @Cast("cudaError_t") int cudaHostGetFlags(@Cast("unsigned int*") IntPointer pFlags, Pointer pHost);
public static native @Cast("cudaError_t") int cudaHostGetFlags(@Cast("unsigned int*") IntBuffer pFlags, Pointer pHost);
public static native @Cast("cudaError_t") int cudaHostGetFlags(@Cast("unsigned int*") int[] pFlags, Pointer pHost);

/**
 * \brief Allocates logical 1D, 2D, or 3D memory objects on the device
 *
 * Allocates at least \p width * \p height * \p depth bytes of linear memory
 * on the device and returns a ::cudaPitchedPtr in which \p ptr is a pointer
 * to the allocated memory. The function may pad the allocation to ensure
 * hardware alignment requirements are met. The pitch returned in the \p pitch
 * field of \p pitchedDevPtr is the width in bytes of the allocation.
 *
 * The returned ::cudaPitchedPtr contains additional fields \p xsize and
 * \p ysize, the logical width and height of the allocation, which are
 * equivalent to the \p width and \p height \p extent parameters provided by
 * the programmer during allocation.
 *
 * For allocations of 2D and 3D objects, it is highly recommended that
 * programmers perform allocations using ::cudaMalloc3D() or
 * ::cudaMallocPitch(). Due to alignment restrictions in the hardware, this is
 * especially true if the application will be performing memory copies
 * involving 2D or 3D objects (whether linear memory or CUDA arrays).
 *
 * \param pitchedDevPtr  - Pointer to allocated pitched device memory
 * \param extent         - Requested allocation size (\p width field in bytes)
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorMemoryAllocation
 * \notefnerr
 *
 * \sa ::cudaMallocPitch, ::cudaFree, ::cudaMemcpy3D, ::cudaMemset3D,
 * ::cudaMalloc3DArray, ::cudaMallocArray, ::cudaFreeArray,
 * \ref ::cudaMallocHost(void**, size_t) "cudaMallocHost (C API)",
 * ::cudaFreeHost, ::cudaHostAlloc, ::make_cudaPitchedPtr, ::make_cudaExtent
 */
public static native @Cast("cudaError_t") int cudaMalloc3D(cudaPitchedPtr pitchedDevPtr, @ByVal cudaExtent extent);

/**
 * \brief Allocate an array on the device
 *
 * Allocates a CUDA array according to the ::cudaChannelFormatDesc structure
 * \p desc and returns a handle to the new CUDA array in \p *array.
 *
 * The ::cudaChannelFormatDesc is defined as:
 * \code
    struct cudaChannelFormatDesc {
        int x, y, z, w;
        enum cudaChannelFormatKind f;
    };
    \endcode
 * where ::cudaChannelFormatKind is one of ::cudaChannelFormatKindSigned,
 * ::cudaChannelFormatKindUnsigned, or ::cudaChannelFormatKindFloat.
 *
 * ::cudaMalloc3DArray() can allocate the following:
 *
 * - A 1D array is allocated if the height and depth extents are both zero.
 * - A 2D array is allocated if only the depth extent is zero.
 * - A 3D array is allocated if all three extents are non-zero.
 * - A 1D layered CUDA array is allocated if only the height extent is zero and
 * the cudaArrayLayered flag is set. Each layer is a 1D array. The number of layers is 
 * determined by the depth extent.
 * - A 2D layered CUDA array is allocated if all three extents are non-zero and 
 * the cudaArrayLayered flag is set. Each layer is a 2D array. The number of layers is 
 * determined by the depth extent.
 * - A cubemap CUDA array is allocated if all three extents are non-zero and the
 * cudaArrayCubemap flag is set. Width must be equal to height, and depth must be six. A cubemap is
 * a special type of 2D layered CUDA array, where the six layers represent the six faces of a cube. 
 * The order of the six layers in memory is the same as that listed in ::cudaGraphicsCubeFace.
 * - A cubemap layered CUDA array is allocated if all three extents are non-zero, and both,
 * cudaArrayCubemap and cudaArrayLayered flags are set. Width must be equal to height, and depth must be 
 * a multiple of six. A cubemap layered CUDA array is a special type of 2D layered CUDA array that consists 
 * of a collection of cubemaps. The first six layers represent the first cubemap, the next six layers form 
 * the second cubemap, and so on.
 *
 *
 * The \p flags parameter enables different options to be specified that affect
 * the allocation, as follows.
 * - ::cudaArrayDefault: This flag's value is defined to be 0 and provides default array allocation
 * - ::cudaArrayLayered: Allocates a layered CUDA array, with the depth extent indicating the number of layers
 * - ::cudaArrayCubemap: Allocates a cubemap CUDA array. Width must be equal to height, and depth must be six.
 *   If the cudaArrayLayered flag is also set, depth must be a multiple of six.
 * - ::cudaArraySurfaceLoadStore: Allocates a CUDA array that could be read from or written to using a surface
 *   reference.
 * - ::cudaArrayTextureGather: This flag indicates that texture gather operations will be performed on the CUDA 
 *   array. Texture gather can only be performed on 2D CUDA arrays.
 *
 * The width, height and depth extents must meet certain size requirements as listed in the following table.
 * All values are specified in elements.
 *
 * Note that 2D CUDA arrays have different size requirements if the ::cudaArrayTextureGather flag is set. In that
 * case, the valid range for (width, height, depth) is ((1,maxTexture2DGather[0]), (1,maxTexture2DGather[1]), 0).
 *
 * \xmlonly
 * <table outputclass="xmlonly">
 * <tgroup cols="3" colsep="1" rowsep="1">
 * <colspec colname="c1" colwidth="1.0*"/>
 * <colspec colname="c2" colwidth="3.0*"/>
 * <colspec colname="c3" colwidth="3.0*"/>
 * <thead>
 * <row>
 * <entry>CUDA array type</entry>
 * <entry>Valid extents that must always be met {(width range in elements),
 * (height range), (depth range)}</entry>
 * <entry>Valid extents with cudaArraySurfaceLoadStore set {(width range in
 * elements), (height range), (depth range)}</entry>
 * </row>
 * </thead>
 * <tbody>
 * <row>
 * <entry>1D</entry>
 * <entry>{ (1,maxTexture1D), 0, 0 }</entry>
 * <entry>{ (1,maxSurface1D), 0, 0 }</entry>
 * </row>
 * <row>
 * <entry>2D</entry>
 * <entry>{ (1,maxTexture2D[0]), (1,maxTexture2D[1]), 0 }</entry>
 * <entry>{ (1,maxSurface2D[0]), (1,maxSurface2D[1]), 0 }</entry>
 * </row>
 * <row>
 * <entry>3D</entry>
 * <entry>{ (1,maxTexture3D[0]), (1,maxTexture3D[1]), (1,maxTexture3D[2]) }
 * OR { (1,maxTexture3DAlt[0]), (1,maxTexture3DAlt[1]),
 * (1,maxTexture3DAlt[2]) }</entry>
 * <entry>{ (1,maxSurface3D[0]), (1,maxSurface3D[1]), (1,maxSurface3D[2]) }</entry>
 * </row>
 * <row>
 * <entry>1D Layered</entry>
 * <entry>{ (1,maxTexture1DLayered[0]), 0, (1,maxTexture1DLayered[1]) }</entry>
 * <entry>{ (1,maxSurface1DLayered[0]), 0, (1,maxSurface1DLayered[1]) }</entry>
 * </row>
 * <row>
 * <entry>2D Layered</entry>
 * <entry>{ (1,maxTexture2DLayered[0]), (1,maxTexture2DLayered[1]),
 * (1,maxTexture2DLayered[2]) }</entry>
 * <entry>{ (1,maxSurface2DLayered[0]), (1,maxSurface2DLayered[1]),
 * (1,maxSurface2DLayered[2]) }</entry>
 * </row>
 * <row>
 * <entry>Cubemap</entry>
 * <entry>{ (1,maxTextureCubemap), (1,maxTextureCubemap), 6 }</entry>
 * <entry>{ (1,maxSurfaceCubemap), (1,maxSurfaceCubemap), 6 }</entry>
 * </row>
 * <row>
 * <entry>Cubemap Layered</entry>
 * <entry>{ (1,maxTextureCubemapLayered[0]), (1,maxTextureCubemapLayered[0]),
 * (1,maxTextureCubemapLayered[1]) }</entry>
 * <entry>{ (1,maxSurfaceCubemapLayered[0]), (1,maxSurfaceCubemapLayered[0]),
 * (1,maxSurfaceCubemapLayered[1]) }</entry>
 * </row>
 * </tbody>
 * </tgroup>
 * </table>
 * \endxmlonly
 *
 * \param array  - Pointer to allocated array in device memory
 * \param desc   - Requested channel format
 * \param extent - Requested allocation size (\p width field in elements)
 * \param flags  - Flags for extensions
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorMemoryAllocation
 * \notefnerr
 *
 * \sa ::cudaMalloc3D, ::cudaMalloc, ::cudaMallocPitch, ::cudaFree,
 * ::cudaFreeArray,
 * \ref ::cudaMallocHost(void**, size_t) "cudaMallocHost (C API)",
 * ::cudaFreeHost, ::cudaHostAlloc,
 * ::make_cudaExtent
 */
public static native @Cast("cudaError_t") int cudaMalloc3DArray(@ByPtrPtr cudaArray array, @Const cudaChannelFormatDesc desc, @ByVal cudaExtent extent, @Cast("unsigned int") int flags/*=0*/);
public static native @Cast("cudaError_t") int cudaMalloc3DArray(@ByPtrPtr cudaArray array, @Const cudaChannelFormatDesc desc, @ByVal cudaExtent extent);

/**
 * \brief Allocate a mipmapped array on the device
 *
 * Allocates a CUDA mipmapped array according to the ::cudaChannelFormatDesc structure
 * \p desc and returns a handle to the new CUDA mipmapped array in \p *mipmappedArray.
 * \p numLevels specifies the number of mipmap levels to be allocated. This value is
 * clamped to the range [1, 1 + floor(log2(max(width, height, depth)))].
 *
 * The ::cudaChannelFormatDesc is defined as:
 * \code
    struct cudaChannelFormatDesc {
        int x, y, z, w;
        enum cudaChannelFormatKind f;
    };
    \endcode
 * where ::cudaChannelFormatKind is one of ::cudaChannelFormatKindSigned,
 * ::cudaChannelFormatKindUnsigned, or ::cudaChannelFormatKindFloat.
 *
 * ::cudaMallocMipmappedArray() can allocate the following:
 *
 * - A 1D mipmapped array is allocated if the height and depth extents are both zero.
 * - A 2D mipmapped array is allocated if only the depth extent is zero.
 * - A 3D mipmapped array is allocated if all three extents are non-zero.
 * - A 1D layered CUDA mipmapped array is allocated if only the height extent is zero and
 * the cudaArrayLayered flag is set. Each layer is a 1D mipmapped array. The number of layers is 
 * determined by the depth extent.
 * - A 2D layered CUDA mipmapped array is allocated if all three extents are non-zero and 
 * the cudaArrayLayered flag is set. Each layer is a 2D mipmapped array. The number of layers is 
 * determined by the depth extent.
 * - A cubemap CUDA mipmapped array is allocated if all three extents are non-zero and the
 * cudaArrayCubemap flag is set. Width must be equal to height, and depth must be six.
 * The order of the six layers in memory is the same as that listed in ::cudaGraphicsCubeFace.
 * - A cubemap layered CUDA mipmapped array is allocated if all three extents are non-zero, and both,
 * cudaArrayCubemap and cudaArrayLayered flags are set. Width must be equal to height, and depth must be 
 * a multiple of six. A cubemap layered CUDA mipmapped array is a special type of 2D layered CUDA mipmapped
 * array that consists of a collection of cubemap mipmapped arrays. The first six layers represent the 
 * first cubemap mipmapped array, the next six layers form the second cubemap mipmapped array, and so on.
 *
 *
 * The \p flags parameter enables different options to be specified that affect
 * the allocation, as follows.
 * - ::cudaArrayDefault: This flag's value is defined to be 0 and provides default mipmapped array allocation
 * - ::cudaArrayLayered: Allocates a layered CUDA mipmapped array, with the depth extent indicating the number of layers
 * - ::cudaArrayCubemap: Allocates a cubemap CUDA mipmapped array. Width must be equal to height, and depth must be six.
 *   If the cudaArrayLayered flag is also set, depth must be a multiple of six.
 * - ::cudaArraySurfaceLoadStore: This flag indicates that individual mipmap levels of the CUDA mipmapped array 
 *   will be read from or written to using a surface reference.
 * - ::cudaArrayTextureGather: This flag indicates that texture gather operations will be performed on the CUDA 
 *   array. Texture gather can only be performed on 2D CUDA mipmapped arrays, and the gather operations are
 *   performed only on the most detailed mipmap level.
 *
 * The width, height and depth extents must meet certain size requirements as listed in the following table.
 * All values are specified in elements.
 *
 * \xmlonly
 * <table outputclass="xmlonly">
 * <tgroup cols="2" colsep="1" rowsep="1">
 * <colspec colname="c1" colwidth="1.0*"/>
 * <colspec colname="c2" colwidth="3.0*"/>
 * <thead>
 * <row>
 * <entry>CUDA array type</entry>
 * <entry>Valid extents {(width range in elements), (height range), (depth
 * range)}</entry>
 * </row>
 * </thead>
 * <tbody>
 * <row>
 * <entry>1D</entry>
 * <entry>{ (1,maxTexture1DMipmap), 0, 0 }</entry>
 * </row>
 * <row>
 * <entry>2D</entry>
 * <entry>{ (1,maxTexture2DMipmap[0]), (1,maxTexture2DMipmap[1]), 0 }</entry>
 * </row>
 * <row>
 * <entry>3D</entry>
 * <entry>{ (1,maxTexture3D[0]), (1,maxTexture3D[1]), (1,maxTexture3D[2]) }</entry>
 * </row>
 * <row>
 * <entry>1D Layered</entry>
 * <entry>{ (1,maxTexture1DLayered[0]), 0, (1,maxTexture1DLayered[1]) }</entry>
 * </row>
 * <row>
 * <entry>2D Layered</entry>
 * <entry>{ (1,maxTexture2DLayered[0]), (1,maxTexture2DLayered[1]),
 * (1,maxTexture2DLayered[2]) }</entry>
 * </row>
 * <row>
 * <entry>Cubemap</entry>
 * <entry>{ (1,maxTextureCubemap), (1,maxTextureCubemap), 6 }</entry>
 * </row>
 * <row>
 * <entry>Cubemap Layered</entry>
 * <entry>{ (1,maxTextureCubemapLayered[0]), (1,maxTextureCubemapLayered[0]),
 * (1,maxTextureCubemapLayered[1]) }</entry>
 * </row>
 * </tbody>
 * </tgroup>
 * </table>
 * \endxmlonly
 *
 * \param mipmappedArray  - Pointer to allocated mipmapped array in device memory
 * \param desc            - Requested channel format
 * \param extent          - Requested allocation size (\p width field in elements)
 * \param numLevels       - Number of mipmap levels to allocate
 * \param flags           - Flags for extensions
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorMemoryAllocation
 * \notefnerr
 *
 * \sa ::cudaMalloc3D, ::cudaMalloc, ::cudaMallocPitch, ::cudaFree,
 * ::cudaFreeArray,
 * \ref ::cudaMallocHost(void**, size_t) "cudaMallocHost (C API)",
 * ::cudaFreeHost, ::cudaHostAlloc,
 * ::make_cudaExtent
 */
public static native @Cast("cudaError_t") int cudaMallocMipmappedArray(@ByPtrPtr cudaMipmappedArray mipmappedArray, @Const cudaChannelFormatDesc desc, @ByVal cudaExtent extent, @Cast("unsigned int") int numLevels, @Cast("unsigned int") int flags/*=0*/);
public static native @Cast("cudaError_t") int cudaMallocMipmappedArray(@ByPtrPtr cudaMipmappedArray mipmappedArray, @Const cudaChannelFormatDesc desc, @ByVal cudaExtent extent, @Cast("unsigned int") int numLevels);

/**
 * \brief Gets a mipmap level of a CUDA mipmapped array
 *
 * Returns in \p *levelArray a CUDA array that represents a single mipmap level
 * of the CUDA mipmapped array \p mipmappedArray.
 *
 * If \p level is greater than the maximum number of levels in this mipmapped array,
 * ::cudaErrorInvalidValue is returned.
 *
 * \param levelArray     - Returned mipmap level CUDA array
 * \param mipmappedArray - CUDA mipmapped array
 * \param level          - Mipmap level
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \notefnerr
 *
 * \sa ::cudaMalloc3D, ::cudaMalloc, ::cudaMallocPitch, ::cudaFree,
 * ::cudaFreeArray,
 * \ref ::cudaMallocHost(void**, size_t) "cudaMallocHost (C API)",
 * ::cudaFreeHost, ::cudaHostAlloc,
 * ::make_cudaExtent
 */
public static native @Cast("cudaError_t") int cudaGetMipmappedArrayLevel(@ByPtrPtr cudaArray levelArray, cudaMipmappedArray mipmappedArray, @Cast("unsigned int") int level);

/**
 * \brief Copies data between 3D objects
 *
\code
struct cudaExtent {
  size_t width;
  size_t height;
  size_t depth;
};
struct cudaExtent make_cudaExtent(size_t w, size_t h, size_t d);

struct cudaPos {
  size_t x;
  size_t y;
  size_t z;
};
struct cudaPos make_cudaPos(size_t x, size_t y, size_t z);

struct cudaMemcpy3DParms {
  cudaArray_t           srcArray;
  struct cudaPos        srcPos;
  struct cudaPitchedPtr srcPtr;
  cudaArray_t           dstArray;
  struct cudaPos        dstPos;
  struct cudaPitchedPtr dstPtr;
  struct cudaExtent     extent;
  enum cudaMemcpyKind   kind;
};
\endcode
 *
 * ::cudaMemcpy3D() copies data betwen two 3D objects. The source and
 * destination objects may be in either host memory, device memory, or a CUDA
 * array. The source, destination, extent, and kind of copy performed is
 * specified by the ::cudaMemcpy3DParms struct which should be initialized to
 * zero before use:
\code
cudaMemcpy3DParms myParms = {0};
\endcode
 *
 * The struct passed to ::cudaMemcpy3D() must specify one of \p srcArray or
 * \p srcPtr and one of \p dstArray or \p dstPtr. Passing more than one
 * non-zero source or destination will cause ::cudaMemcpy3D() to return an
 * error.
 *
 * The \p srcPos and \p dstPos fields are optional offsets into the source and
 * destination objects and are defined in units of each object's elements. The
 * element for a host or device pointer is assumed to be <b>unsigned char</b>.
 * For CUDA arrays, positions must be in the range [0, 2048) for any
 * dimension.
 *
 * The \p extent field defines the dimensions of the transferred area in
 * elements. If a CUDA array is participating in the copy, the extent is
 * defined in terms of that array's elements. If no CUDA array is
 * participating in the copy then the extents are defined in elements of
 * <b>unsigned char</b>.
 *
 * The \p kind field defines the direction of the copy. It must be one of
 * ::cudaMemcpyHostToHost, ::cudaMemcpyHostToDevice, ::cudaMemcpyDeviceToHost,
 * or ::cudaMemcpyDeviceToDevice.
 *
 * If the source and destination are both arrays, ::cudaMemcpy3D() will return
 * an error if they do not have the same element size.
 *
 * The source and destination object may not overlap. If overlapping source
 * and destination objects are specified, undefined behavior will result.
 *
 * The source object must lie entirely within the region defined by \p srcPos
 * and \p extent. The destination object must lie entirely within the region
 * defined by \p dstPos and \p extent.
 *
 * ::cudaMemcpy3D() returns an error if the pitch of \p srcPtr or \p dstPtr
 * exceeds the maximum allowed. The pitch of a ::cudaPitchedPtr allocated
 * with ::cudaMalloc3D() will always be valid.
 *
 * \param p - 3D memory copy parameters
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidDevicePointer,
 * ::cudaErrorInvalidPitchValue,
 * ::cudaErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_sync
 *
 * \sa ::cudaMalloc3D, ::cudaMalloc3DArray, ::cudaMemset3D, ::cudaMemcpy3DAsync,
 * ::cudaMemcpy, ::cudaMemcpy2D, ::cudaMemcpyToArray,
 * ::cudaMemcpy2DToArray, ::cudaMemcpyFromArray, ::cudaMemcpy2DFromArray,
 * ::cudaMemcpyArrayToArray, ::cudaMemcpy2DArrayToArray, ::cudaMemcpyToSymbol,
 * ::cudaMemcpyFromSymbol, ::cudaMemcpyAsync, ::cudaMemcpy2DAsync,
 * ::cudaMemcpyToArrayAsync, ::cudaMemcpy2DToArrayAsync,
 * ::cudaMemcpyFromArrayAsync, ::cudaMemcpy2DFromArrayAsync,
 * ::cudaMemcpyToSymbolAsync, ::cudaMemcpyFromSymbolAsync,
 * ::make_cudaExtent, ::make_cudaPos
 */
public static native @Cast("cudaError_t") int cudaMemcpy3D(@Const cudaMemcpy3DParms p);

/**
 * \brief Copies memory between devices
 *
 * Perform a 3D memory copy according to the parameters specified in
 * \p p.  See the definition of the ::cudaMemcpy3DPeerParms structure
 * for documentation of its parameters.
 *
 * Note that this function is synchronous with respect to the host only if
 * the source or destination of the transfer is host memory.  Note also 
 * that this copy is serialized with respect to all pending and future 
 * asynchronous work in to the current device, the copy's source device,
 * and the copy's destination device (use ::cudaMemcpy3DPeerAsync to avoid 
 * this synchronization).
 *
 * \param p - Parameters for the memory copy
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidDevice
 * \notefnerr
 * \note_sync
 *
 * \sa ::cudaMemcpy, ::cudaMemcpyPeer, ::cudaMemcpyAsync, ::cudaMemcpyPeerAsync,
 * ::cudaMemcpy3DPeerAsync
 */
public static native @Cast("cudaError_t") int cudaMemcpy3DPeer(@Const cudaMemcpy3DPeerParms p);

/**
 * \brief Copies data between 3D objects
 *
\code
struct cudaExtent {
  size_t width;
  size_t height;
  size_t depth;
};
struct cudaExtent make_cudaExtent(size_t w, size_t h, size_t d);

struct cudaPos {
  size_t x;
  size_t y;
  size_t z;
};
struct cudaPos make_cudaPos(size_t x, size_t y, size_t z);

struct cudaMemcpy3DParms {
  cudaArray_t           srcArray;
  struct cudaPos        srcPos;
  struct cudaPitchedPtr srcPtr;
  cudaArray_t           dstArray;
  struct cudaPos        dstPos;
  struct cudaPitchedPtr dstPtr;
  struct cudaExtent     extent;
  enum cudaMemcpyKind   kind;
};
\endcode
 *
 * ::cudaMemcpy3DAsync() copies data betwen two 3D objects. The source and
 * destination objects may be in either host memory, device memory, or a CUDA
 * array. The source, destination, extent, and kind of copy performed is
 * specified by the ::cudaMemcpy3DParms struct which should be initialized to
 * zero before use:
\code
cudaMemcpy3DParms myParms = {0};
\endcode
 *
 * The struct passed to ::cudaMemcpy3DAsync() must specify one of \p srcArray
 * or \p srcPtr and one of \p dstArray or \p dstPtr. Passing more than one
 * non-zero source or destination will cause ::cudaMemcpy3DAsync() to return an
 * error.
 *
 * The \p srcPos and \p dstPos fields are optional offsets into the source and
 * destination objects and are defined in units of each object's elements. The
 * element for a host or device pointer is assumed to be <b>unsigned char</b>.
 * For CUDA arrays, positions must be in the range [0, 2048) for any
 * dimension.
 *
 * The \p extent field defines the dimensions of the transferred area in
 * elements. If a CUDA array is participating in the copy, the extent is
 * defined in terms of that array's elements. If no CUDA array is
 * participating in the copy then the extents are defined in elements of
 * <b>unsigned char</b>.
 *
 * The \p kind field defines the direction of the copy. It must be one of
 * ::cudaMemcpyHostToHost, ::cudaMemcpyHostToDevice, ::cudaMemcpyDeviceToHost,
 * or ::cudaMemcpyDeviceToDevice.
 *
 * If the source and destination are both arrays, ::cudaMemcpy3DAsync() will
 * return an error if they do not have the same element size.
 *
 * The source and destination object may not overlap. If overlapping source
 * and destination objects are specified, undefined behavior will result.
 *
 * The source object must lie entirely within the region defined by \p srcPos
 * and \p extent. The destination object must lie entirely within the region
 * defined by \p dstPos and \p extent.
 *
 * ::cudaMemcpy3DAsync() returns an error if the pitch of \p srcPtr or
 * \p dstPtr exceeds the maximum allowed. The pitch of a
 * ::cudaPitchedPtr allocated with ::cudaMalloc3D() will always be valid.
 *
 * ::cudaMemcpy3DAsync() is asynchronous with respect to the host, so
 * the call may return before the copy is complete. The copy can optionally
 * be associated to a stream by passing a non-zero \p stream argument. If
 * \p kind is ::cudaMemcpyHostToDevice or ::cudaMemcpyDeviceToHost and \p stream
 * is non-zero, the copy may overlap with operations in other streams.
 *
 * \param p      - 3D memory copy parameters
 * \param stream - Stream identifier
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidDevicePointer,
 * ::cudaErrorInvalidPitchValue,
 * ::cudaErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_async
 * \note_null_stream
 *
 * \sa ::cudaMalloc3D, ::cudaMalloc3DArray, ::cudaMemset3D, ::cudaMemcpy3D,
 * ::cudaMemcpy, ::cudaMemcpy2D, ::cudaMemcpyToArray,
 * ::cudaMemcpy2DToArray, ::cudaMemcpyFromArray, ::cudaMemcpy2DFromArray,
 * ::cudaMemcpyArrayToArray, ::cudaMemcpy2DArrayToArray, ::cudaMemcpyToSymbol,
 * ::cudaMemcpyFromSymbol, ::cudaMemcpyAsync, ::cudaMemcpy2DAsync,
 * ::cudaMemcpyToArrayAsync, ::cudaMemcpy2DToArrayAsync,
 * ::cudaMemcpyFromArrayAsync, ::cudaMemcpy2DFromArrayAsync,
 * ::cudaMemcpyToSymbolAsync, ::cudaMemcpyFromSymbolAsync,
 * ::make_cudaExtent, ::make_cudaPos
 */
public static native @Cast("cudaError_t") int cudaMemcpy3DAsync(@Const cudaMemcpy3DParms p, cudaStream stream/*=0*/);
public static native @Cast("cudaError_t") int cudaMemcpy3DAsync(@Const cudaMemcpy3DParms p);

/**
 * \brief Copies memory between devices asynchronously.
 *
 * Perform a 3D memory copy according to the parameters specified in
 * \p p.  See the definition of the ::cudaMemcpy3DPeerParms structure
 * for documentation of its parameters.
 *
 * \param p      - Parameters for the memory copy
 * \param stream - Stream identifier
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidDevice
 * \notefnerr
 * \note_async
 * \note_null_stream
 *
 * \sa ::cudaMemcpy, ::cudaMemcpyPeer, ::cudaMemcpyAsync, ::cudaMemcpyPeerAsync,
 * ::cudaMemcpy3DPeerAsync
 */
public static native @Cast("cudaError_t") int cudaMemcpy3DPeerAsync(@Const cudaMemcpy3DPeerParms p, cudaStream stream/*=0*/);
public static native @Cast("cudaError_t") int cudaMemcpy3DPeerAsync(@Const cudaMemcpy3DPeerParms p);

/**
 * \brief Gets free and total device memory
 *
 * Returns in \p *free and \p *total respectively, the free and total amount of
 * memory available for allocation by the device in bytes.
 *
 * \param free  - Returned free memory in bytes
 * \param total - Returned total memory in bytes
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInitializationError,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorLaunchFailure
 * \notefnerr
 *
 */
public static native @Cast("cudaError_t") int cudaMemGetInfo(@Cast("size_t*") SizeTPointer free, @Cast("size_t*") SizeTPointer total);

/**
 * \brief Gets info about the specified cudaArray
 * 
 * Returns in \p *desc, \p *extent and \p *flags respectively, the type, shape 
 * and flags of \p array.
 *
 * Any of \p *desc, \p *extent and \p *flags may be specified as NULL.
 *
 * \param desc   - Returned array type
 * \param extent - Returned array shape. 2D arrays will have depth of zero
 * \param flags  - Returned array flags
 * \param array  - The ::cudaArray to get info for
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \notefnerr
 *
 */
public static native @Cast("cudaError_t") int cudaArrayGetInfo(cudaChannelFormatDesc desc, cudaExtent extent, @Cast("unsigned int*") IntPointer flags, cudaArray array);
public static native @Cast("cudaError_t") int cudaArrayGetInfo(cudaChannelFormatDesc desc, cudaExtent extent, @Cast("unsigned int*") IntBuffer flags, cudaArray array);
public static native @Cast("cudaError_t") int cudaArrayGetInfo(cudaChannelFormatDesc desc, cudaExtent extent, @Cast("unsigned int*") int[] flags, cudaArray array);

/**
 * \brief Copies data between host and device
 *
 * Copies \p count bytes from the memory area pointed to by \p src to the
 * memory area pointed to by \p dst, where \p kind is one of
 * ::cudaMemcpyHostToHost, ::cudaMemcpyHostToDevice, ::cudaMemcpyDeviceToHost,
 * or ::cudaMemcpyDeviceToDevice, and specifies the direction of the copy. The
 * memory areas may not overlap. Calling ::cudaMemcpy() with \p dst and \p src
 * pointers that do not match the direction of the copy results in an
 * undefined behavior.
 *
 * \param dst   - Destination memory address
 * \param src   - Source memory address
 * \param count - Size in bytes to copy
 * \param kind  - Type of transfer
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidDevicePointer,
 * ::cudaErrorInvalidMemcpyDirection
 * \notefnerr
 *
 * \note_sync
 *
 * \sa ::cudaMemcpy2D, ::cudaMemcpyToArray,
 * ::cudaMemcpy2DToArray, ::cudaMemcpyFromArray, ::cudaMemcpy2DFromArray,
 * ::cudaMemcpyArrayToArray, ::cudaMemcpy2DArrayToArray, ::cudaMemcpyToSymbol,
 * ::cudaMemcpyFromSymbol, ::cudaMemcpyAsync, ::cudaMemcpy2DAsync,
 * ::cudaMemcpyToArrayAsync, ::cudaMemcpy2DToArrayAsync,
 * ::cudaMemcpyFromArrayAsync, ::cudaMemcpy2DFromArrayAsync,
 * ::cudaMemcpyToSymbolAsync, ::cudaMemcpyFromSymbolAsync
 */
public static native @Cast("cudaError_t") int cudaMemcpy(Pointer dst, @Const Pointer src, @Cast("size_t") long count, @Cast("cudaMemcpyKind") int kind);

/**
 * \brief Copies memory between two devices
 *
 * Copies memory from one device to memory on another device.  \p dst is the 
 * base device pointer of the destination memory and \p dstDevice is the 
 * destination device.  \p src is the base device pointer of the source memory 
 * and \p srcDevice is the source device.  \p count specifies the number of bytes 
 * to copy.
 *
 * Note that this function is asynchronous with respect to the host, but 
 * serialized with respect all pending and future asynchronous work in to the 
 * current device, \p srcDevice, and \p dstDevice (use ::cudaMemcpyPeerAsync 
 * to avoid this synchronization).
 *
 * \param dst       - Destination device pointer
 * \param dstDevice - Destination device
 * \param src       - Source device pointer
 * \param srcDevice - Source device
 * \param count     - Size of memory copy in bytes
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidDevice
 * \notefnerr
 * \note_sync
 *
 * \sa ::cudaMemcpy, ::cudaMemcpyAsync, ::cudaMemcpyPeerAsync,
 * ::cudaMemcpy3DPeerAsync
 */
public static native @Cast("cudaError_t") int cudaMemcpyPeer(Pointer dst, int dstDevice, @Const Pointer src, int srcDevice, @Cast("size_t") long count);

/**
 * \brief Copies data between host and device
 *
 * Copies \p count bytes from the memory area pointed to by \p src to the
 * CUDA array \p dst starting at the upper left corner
 * (\p wOffset, \p hOffset), where \p kind is one of ::cudaMemcpyHostToHost,
 * ::cudaMemcpyHostToDevice, ::cudaMemcpyDeviceToHost, or
 * ::cudaMemcpyDeviceToDevice, and specifies the direction of the copy.
 *
 * \param dst     - Destination memory address
 * \param wOffset - Destination starting X offset
 * \param hOffset - Destination starting Y offset
 * \param src     - Source memory address
 * \param count   - Size in bytes to copy
 * \param kind    - Type of transfer
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidDevicePointer,
 * ::cudaErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_sync
 *
 * \sa ::cudaMemcpy, ::cudaMemcpy2D,
 * ::cudaMemcpy2DToArray, ::cudaMemcpyFromArray, ::cudaMemcpy2DFromArray,
 * ::cudaMemcpyArrayToArray, ::cudaMemcpy2DArrayToArray, ::cudaMemcpyToSymbol,
 * ::cudaMemcpyFromSymbol, ::cudaMemcpyAsync, ::cudaMemcpy2DAsync,
 * ::cudaMemcpyToArrayAsync, ::cudaMemcpy2DToArrayAsync,
 * ::cudaMemcpyFromArrayAsync, ::cudaMemcpy2DFromArrayAsync,
 * ::cudaMemcpyToSymbolAsync, ::cudaMemcpyFromSymbolAsync
 */
public static native @Cast("cudaError_t") int cudaMemcpyToArray(cudaArray dst, @Cast("size_t") long wOffset, @Cast("size_t") long hOffset, @Const Pointer src, @Cast("size_t") long count, @Cast("cudaMemcpyKind") int kind);

/**
 * \brief Copies data between host and device
 *
 * Copies \p count bytes from the CUDA array \p src starting at the upper
 * left corner (\p wOffset, hOffset) to the memory area pointed to by \p dst,
 * where \p kind is one of ::cudaMemcpyHostToHost, ::cudaMemcpyHostToDevice,
 * ::cudaMemcpyDeviceToHost, or ::cudaMemcpyDeviceToDevice, and specifies the
 * direction of the copy.
 *
 * \param dst     - Destination memory address
 * \param src     - Source memory address
 * \param wOffset - Source starting X offset
 * \param hOffset - Source starting Y offset
 * \param count   - Size in bytes to copy
 * \param kind    - Type of transfer
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidDevicePointer,
 * ::cudaErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_sync
 *
 * \sa ::cudaMemcpy, ::cudaMemcpy2D, ::cudaMemcpyToArray,
 * ::cudaMemcpy2DToArray, ::cudaMemcpy2DFromArray,
 * ::cudaMemcpyArrayToArray, ::cudaMemcpy2DArrayToArray, ::cudaMemcpyToSymbol,
 * ::cudaMemcpyFromSymbol, ::cudaMemcpyAsync, ::cudaMemcpy2DAsync,
 * ::cudaMemcpyToArrayAsync, ::cudaMemcpy2DToArrayAsync,
 * ::cudaMemcpyFromArrayAsync, ::cudaMemcpy2DFromArrayAsync,
 * ::cudaMemcpyToSymbolAsync, ::cudaMemcpyFromSymbolAsync
 */
public static native @Cast("cudaError_t") int cudaMemcpyFromArray(Pointer dst, cudaArray src, @Cast("size_t") long wOffset, @Cast("size_t") long hOffset, @Cast("size_t") long count, @Cast("cudaMemcpyKind") int kind);

/**
 * \brief Copies data between host and device
 *
 * Copies \p count bytes from the CUDA array \p src starting at the upper
 * left corner (\p wOffsetSrc, \p hOffsetSrc) to the CUDA array \p dst
 * starting at the upper left corner (\p wOffsetDst, \p hOffsetDst) where
 * \p kind is one of ::cudaMemcpyHostToHost, ::cudaMemcpyHostToDevice,
 * ::cudaMemcpyDeviceToHost, or ::cudaMemcpyDeviceToDevice, and specifies the
 * direction of the copy.
 *
 * \param dst        - Destination memory address
 * \param wOffsetDst - Destination starting X offset
 * \param hOffsetDst - Destination starting Y offset
 * \param src        - Source memory address
 * \param wOffsetSrc - Source starting X offset
 * \param hOffsetSrc - Source starting Y offset
 * \param count      - Size in bytes to copy
 * \param kind       - Type of transfer
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidMemcpyDirection
 * \notefnerr
 *
 * \sa ::cudaMemcpy, ::cudaMemcpy2D, ::cudaMemcpyToArray,
 * ::cudaMemcpy2DToArray, ::cudaMemcpyFromArray, ::cudaMemcpy2DFromArray,
 * ::cudaMemcpy2DArrayToArray, ::cudaMemcpyToSymbol,
 * ::cudaMemcpyFromSymbol, ::cudaMemcpyAsync, ::cudaMemcpy2DAsync,
 * ::cudaMemcpyToArrayAsync, ::cudaMemcpy2DToArrayAsync,
 * ::cudaMemcpyFromArrayAsync, ::cudaMemcpy2DFromArrayAsync,
 * ::cudaMemcpyToSymbolAsync, ::cudaMemcpyFromSymbolAsync
 */
public static native @Cast("cudaError_t") int cudaMemcpyArrayToArray(cudaArray dst, @Cast("size_t") long wOffsetDst, @Cast("size_t") long hOffsetDst, cudaArray src, @Cast("size_t") long wOffsetSrc, @Cast("size_t") long hOffsetSrc, @Cast("size_t") long count, @Cast("cudaMemcpyKind") int kind/*=cudaMemcpyDeviceToDevice*/);
public static native @Cast("cudaError_t") int cudaMemcpyArrayToArray(cudaArray dst, @Cast("size_t") long wOffsetDst, @Cast("size_t") long hOffsetDst, cudaArray src, @Cast("size_t") long wOffsetSrc, @Cast("size_t") long hOffsetSrc, @Cast("size_t") long count);

/**
 * \brief Copies data between host and device
 *
 * Copies a matrix (\p height rows of \p width bytes each) from the memory
 * area pointed to by \p src to the memory area pointed to by \p dst, where
 * \p kind is one of ::cudaMemcpyHostToHost, ::cudaMemcpyHostToDevice,
 * ::cudaMemcpyDeviceToHost, or ::cudaMemcpyDeviceToDevice, and specifies the
 * direction of the copy. \p dpitch and \p spitch are the widths in memory in
 * bytes of the 2D arrays pointed to by \p dst and \p src, including any
 * padding added to the end of each row. The memory areas may not overlap.
 * \p width must not exceed either \p dpitch or \p spitch.
 * Calling ::cudaMemcpy2D() with \p dst and \p src pointers that do not match
 * the direction of the copy results in an undefined behavior.
 * ::cudaMemcpy2D() returns an error if \p dpitch or \p spitch exceeds
 * the maximum allowed.
 *
 * \param dst    - Destination memory address
 * \param dpitch - Pitch of destination memory
 * \param src    - Source memory address
 * \param spitch - Pitch of source memory
 * \param width  - Width of matrix transfer (columns in bytes)
 * \param height - Height of matrix transfer (rows)
 * \param kind   - Type of transfer
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidPitchValue,
 * ::cudaErrorInvalidDevicePointer,
 * ::cudaErrorInvalidMemcpyDirection
 * \notefnerr
 *
 * \sa ::cudaMemcpy, ::cudaMemcpyToArray,
 * ::cudaMemcpy2DToArray, ::cudaMemcpyFromArray, ::cudaMemcpy2DFromArray,
 * ::cudaMemcpyArrayToArray, ::cudaMemcpy2DArrayToArray, ::cudaMemcpyToSymbol,
 * ::cudaMemcpyFromSymbol, ::cudaMemcpyAsync, ::cudaMemcpy2DAsync,
 * ::cudaMemcpyToArrayAsync, ::cudaMemcpy2DToArrayAsync,
 * ::cudaMemcpyFromArrayAsync, ::cudaMemcpy2DFromArrayAsync,
 * ::cudaMemcpyToSymbolAsync, ::cudaMemcpyFromSymbolAsync
 */
public static native @Cast("cudaError_t") int cudaMemcpy2D(Pointer dst, @Cast("size_t") long dpitch, @Const Pointer src, @Cast("size_t") long spitch, @Cast("size_t") long width, @Cast("size_t") long height, @Cast("cudaMemcpyKind") int kind);

/**
 * \brief Copies data between host and device
 *
 * Copies a matrix (\p height rows of \p width bytes each) from the memory
 * area pointed to by \p src to the CUDA array \p dst starting at the
 * upper left corner (\p wOffset, \p hOffset) where \p kind is one of
 * ::cudaMemcpyHostToHost, ::cudaMemcpyHostToDevice, ::cudaMemcpyDeviceToHost,
 * or ::cudaMemcpyDeviceToDevice, and specifies the direction of the copy.
 * \p spitch is the width in memory in bytes of the 2D array pointed to by
 * \p src, including any padding added to the end of each row. \p wOffset +
 * \p width must not exceed the width of the CUDA array \p dst. \p width must
 * not exceed \p spitch. ::cudaMemcpy2DToArray() returns an error if \p spitch
 * exceeds the maximum allowed.
 *
 * \param dst     - Destination memory address
 * \param wOffset - Destination starting X offset
 * \param hOffset - Destination starting Y offset
 * \param src     - Source memory address
 * \param spitch  - Pitch of source memory
 * \param width   - Width of matrix transfer (columns in bytes)
 * \param height  - Height of matrix transfer (rows)
 * \param kind    - Type of transfer
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidDevicePointer,
 * ::cudaErrorInvalidPitchValue,
 * ::cudaErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_sync
 *
 * \sa ::cudaMemcpy, ::cudaMemcpy2D, ::cudaMemcpyToArray,
 * ::cudaMemcpyFromArray, ::cudaMemcpy2DFromArray,
 * ::cudaMemcpyArrayToArray, ::cudaMemcpy2DArrayToArray, ::cudaMemcpyToSymbol,
 * ::cudaMemcpyFromSymbol, ::cudaMemcpyAsync, ::cudaMemcpy2DAsync,
 * ::cudaMemcpyToArrayAsync, ::cudaMemcpy2DToArrayAsync,
 * ::cudaMemcpyFromArrayAsync, ::cudaMemcpy2DFromArrayAsync,
 * ::cudaMemcpyToSymbolAsync, ::cudaMemcpyFromSymbolAsync
 */
public static native @Cast("cudaError_t") int cudaMemcpy2DToArray(cudaArray dst, @Cast("size_t") long wOffset, @Cast("size_t") long hOffset, @Const Pointer src, @Cast("size_t") long spitch, @Cast("size_t") long width, @Cast("size_t") long height, @Cast("cudaMemcpyKind") int kind);

/**
 * \brief Copies data between host and device
 *
 * Copies a matrix (\p height rows of \p width bytes each) from the CUDA
 * array \p srcArray starting at the upper left corner
 * (\p wOffset, \p hOffset) to the memory area pointed to by \p dst, where
 * \p kind is one of ::cudaMemcpyHostToHost, ::cudaMemcpyHostToDevice,
 * ::cudaMemcpyDeviceToHost, or ::cudaMemcpyDeviceToDevice, and specifies the
 * direction of the copy. \p dpitch is the width in memory in bytes of the 2D
 * array pointed to by \p dst, including any padding added to the end of each
 * row. \p wOffset + \p width must not exceed the width of the CUDA array
 * \p src. \p width must not exceed \p dpitch. ::cudaMemcpy2DFromArray()
 * returns an error if \p dpitch exceeds the maximum allowed.
 *
 * \param dst     - Destination memory address
 * \param dpitch  - Pitch of destination memory
 * \param src     - Source memory address
 * \param wOffset - Source starting X offset
 * \param hOffset - Source starting Y offset
 * \param width   - Width of matrix transfer (columns in bytes)
 * \param height  - Height of matrix transfer (rows)
 * \param kind    - Type of transfer
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidDevicePointer,
 * ::cudaErrorInvalidPitchValue,
 * ::cudaErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_sync
 *
 * \sa ::cudaMemcpy, ::cudaMemcpy2D, ::cudaMemcpyToArray,
 * ::cudaMemcpy2DToArray, ::cudaMemcpyFromArray,
 * ::cudaMemcpyArrayToArray, ::cudaMemcpy2DArrayToArray, ::cudaMemcpyToSymbol,
 * ::cudaMemcpyFromSymbol, ::cudaMemcpyAsync, ::cudaMemcpy2DAsync,
 * ::cudaMemcpyToArrayAsync, ::cudaMemcpy2DToArrayAsync,
 * ::cudaMemcpyFromArrayAsync, ::cudaMemcpy2DFromArrayAsync,
 * ::cudaMemcpyToSymbolAsync, ::cudaMemcpyFromSymbolAsync
 */
public static native @Cast("cudaError_t") int cudaMemcpy2DFromArray(Pointer dst, @Cast("size_t") long dpitch, cudaArray src, @Cast("size_t") long wOffset, @Cast("size_t") long hOffset, @Cast("size_t") long width, @Cast("size_t") long height, @Cast("cudaMemcpyKind") int kind);

/**
 * \brief Copies data between host and device
 *
 * Copies a matrix (\p height rows of \p width bytes each) from the CUDA
 * array \p srcArray starting at the upper left corner
 * (\p wOffsetSrc, \p hOffsetSrc) to the CUDA array \p dst starting at
 * the upper left corner (\p wOffsetDst, \p hOffsetDst), where \p kind is one
 * of ::cudaMemcpyHostToHost, ::cudaMemcpyHostToDevice,
 * ::cudaMemcpyDeviceToHost, or ::cudaMemcpyDeviceToDevice, and specifies the
 * direction of the copy. \p wOffsetDst + \p width must not exceed the width
 * of the CUDA array \p dst. \p wOffsetSrc + \p width must not exceed the width
 * of the CUDA array \p src.
 *
 * \param dst        - Destination memory address
 * \param wOffsetDst - Destination starting X offset
 * \param hOffsetDst - Destination starting Y offset
 * \param src        - Source memory address
 * \param wOffsetSrc - Source starting X offset
 * \param hOffsetSrc - Source starting Y offset
 * \param width      - Width of matrix transfer (columns in bytes)
 * \param height     - Height of matrix transfer (rows)
 * \param kind       - Type of transfer
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_sync
 *
 * \sa ::cudaMemcpy, ::cudaMemcpy2D, ::cudaMemcpyToArray,
 * ::cudaMemcpy2DToArray, ::cudaMemcpyFromArray, ::cudaMemcpy2DFromArray,
 * ::cudaMemcpyArrayToArray, ::cudaMemcpyToSymbol,
 * ::cudaMemcpyFromSymbol, ::cudaMemcpyAsync, ::cudaMemcpy2DAsync,
 * ::cudaMemcpyToArrayAsync, ::cudaMemcpy2DToArrayAsync,
 * ::cudaMemcpyFromArrayAsync, ::cudaMemcpy2DFromArrayAsync,
 * ::cudaMemcpyToSymbolAsync, ::cudaMemcpyFromSymbolAsync
 */
public static native @Cast("cudaError_t") int cudaMemcpy2DArrayToArray(cudaArray dst, @Cast("size_t") long wOffsetDst, @Cast("size_t") long hOffsetDst, cudaArray src, @Cast("size_t") long wOffsetSrc, @Cast("size_t") long hOffsetSrc, @Cast("size_t") long width, @Cast("size_t") long height, @Cast("cudaMemcpyKind") int kind/*=cudaMemcpyDeviceToDevice*/);
public static native @Cast("cudaError_t") int cudaMemcpy2DArrayToArray(cudaArray dst, @Cast("size_t") long wOffsetDst, @Cast("size_t") long hOffsetDst, cudaArray src, @Cast("size_t") long wOffsetSrc, @Cast("size_t") long hOffsetSrc, @Cast("size_t") long width, @Cast("size_t") long height);

/**
 * \brief Copies data to the given symbol on the device
 *
 * Copies \p count bytes from the memory area pointed to by \p src
 * to the memory area pointed to by \p offset bytes from the start of symbol
 * \p symbol. The memory areas may not overlap. \p symbol is a variable that
 * resides in global or constant memory space. \p kind can be either
 * ::cudaMemcpyHostToDevice or ::cudaMemcpyDeviceToDevice.
 *
 * \param symbol - Device symbol address
 * \param src    - Source memory address
 * \param count  - Size in bytes to copy
 * \param offset - Offset from start of symbol in bytes
 * \param kind   - Type of transfer
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidSymbol,
 * ::cudaErrorInvalidDevicePointer,
 * ::cudaErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_sync
 * \note_string_api_deprecation
 *
 * \sa ::cudaMemcpy, ::cudaMemcpy2D, ::cudaMemcpyToArray,
 * ::cudaMemcpy2DToArray, ::cudaMemcpyFromArray, ::cudaMemcpy2DFromArray,
 * ::cudaMemcpyArrayToArray, ::cudaMemcpy2DArrayToArray,
 * ::cudaMemcpyFromSymbol, ::cudaMemcpyAsync, ::cudaMemcpy2DAsync,
 * ::cudaMemcpyToArrayAsync, ::cudaMemcpy2DToArrayAsync,
 * ::cudaMemcpyFromArrayAsync, ::cudaMemcpy2DFromArrayAsync,
 * ::cudaMemcpyToSymbolAsync, ::cudaMemcpyFromSymbolAsync
 */
public static native @Cast("cudaError_t") int cudaMemcpyToSymbol(@Const Pointer symbol, @Const Pointer src, @Cast("size_t") long count, @Cast("size_t") long offset/*=0*/, @Cast("cudaMemcpyKind") int kind/*=cudaMemcpyHostToDevice*/);
public static native @Cast("cudaError_t") int cudaMemcpyToSymbol(@Const Pointer symbol, @Const Pointer src, @Cast("size_t") long count);

/**
 * \brief Copies data from the given symbol on the device
 *
 * Copies \p count bytes from the memory area pointed to by \p offset bytes
 * from the start of symbol \p symbol to the memory area pointed to by \p dst.
 * The memory areas may not overlap. \p symbol is a variable that
 * resides in global or constant memory space. \p kind can be either
 * ::cudaMemcpyDeviceToHost or ::cudaMemcpyDeviceToDevice.
 *
 * \param dst    - Destination memory address
 * \param symbol - Device symbol address
 * \param count  - Size in bytes to copy
 * \param offset - Offset from start of symbol in bytes
 * \param kind   - Type of transfer
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidSymbol,
 * ::cudaErrorInvalidDevicePointer,
 * ::cudaErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_sync
 * \note_string_api_deprecation
 *
 * \sa ::cudaMemcpy, ::cudaMemcpy2D, ::cudaMemcpyToArray,
 * ::cudaMemcpy2DToArray, ::cudaMemcpyFromArray, ::cudaMemcpy2DFromArray,
 * ::cudaMemcpyArrayToArray, ::cudaMemcpy2DArrayToArray, ::cudaMemcpyToSymbol,
 * ::cudaMemcpyAsync, ::cudaMemcpy2DAsync,
 * ::cudaMemcpyToArrayAsync, ::cudaMemcpy2DToArrayAsync,
 * ::cudaMemcpyFromArrayAsync, ::cudaMemcpy2DFromArrayAsync,
 * ::cudaMemcpyToSymbolAsync, ::cudaMemcpyFromSymbolAsync
 */
public static native @Cast("cudaError_t") int cudaMemcpyFromSymbol(Pointer dst, @Const Pointer symbol, @Cast("size_t") long count, @Cast("size_t") long offset/*=0*/, @Cast("cudaMemcpyKind") int kind/*=cudaMemcpyDeviceToHost*/);
public static native @Cast("cudaError_t") int cudaMemcpyFromSymbol(Pointer dst, @Const Pointer symbol, @Cast("size_t") long count);


/**
 * \brief Copies data between host and device
 *
 * Copies \p count bytes from the memory area pointed to by \p src to the
 * memory area pointed to by \p dst, where \p kind is one of
 * ::cudaMemcpyHostToHost, ::cudaMemcpyHostToDevice, ::cudaMemcpyDeviceToHost,
 * or ::cudaMemcpyDeviceToDevice, and specifies the direction of the copy. The
 * memory areas may not overlap. Calling ::cudaMemcpyAsync() with \p dst and
 * \p src pointers that do not match the direction of the copy results in an
 * undefined behavior.
 *
 * ::cudaMemcpyAsync() is asynchronous with respect to the host, so the call
 * may return before the copy is complete. The copy can optionally be
 * associated to a stream by passing a non-zero \p stream argument. If \p kind
 * is ::cudaMemcpyHostToDevice or ::cudaMemcpyDeviceToHost and the \p stream is
 * non-zero, the copy may overlap with operations in other streams.
 *
 * \param dst    - Destination memory address
 * \param src    - Source memory address
 * \param count  - Size in bytes to copy
 * \param kind   - Type of transfer
 * \param stream - Stream identifier
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidDevicePointer,
 * ::cudaErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_async
 * \note_null_stream
 *
 * \sa ::cudaMemcpy, ::cudaMemcpy2D, ::cudaMemcpyToArray,
 * ::cudaMemcpy2DToArray, ::cudaMemcpyFromArray, ::cudaMemcpy2DFromArray,
 * ::cudaMemcpyArrayToArray, ::cudaMemcpy2DArrayToArray, ::cudaMemcpyToSymbol,
 * ::cudaMemcpyFromSymbol, ::cudaMemcpy2DAsync,
 * ::cudaMemcpyToArrayAsync, ::cudaMemcpy2DToArrayAsync,
 * ::cudaMemcpyFromArrayAsync, ::cudaMemcpy2DFromArrayAsync,
 * ::cudaMemcpyToSymbolAsync, ::cudaMemcpyFromSymbolAsync
 */
public static native @Cast("cudaError_t") int cudaMemcpyAsync(Pointer dst, @Const Pointer src, @Cast("size_t") long count, @Cast("cudaMemcpyKind") int kind, cudaStream stream/*=0*/);
public static native @Cast("cudaError_t") int cudaMemcpyAsync(Pointer dst, @Const Pointer src, @Cast("size_t") long count, @Cast("cudaMemcpyKind") int kind);

/**
 * \brief Copies memory between two devices asynchronously.
 *
 * Copies memory from one device to memory on another device.  \p dst is the 
 * base device pointer of the destination memory and \p dstDevice is the 
 * destination device.  \p src is the base device pointer of the source memory 
 * and \p srcDevice is the source device.  \p count specifies the number of bytes 
 * to copy.
 *
 * Note that this function is asynchronous with respect to the host and all work
 * on other devices.
 *
 * \param dst       - Destination device pointer
 * \param dstDevice - Destination device
 * \param src       - Source device pointer
 * \param srcDevice - Source device
 * \param count     - Size of memory copy in bytes
 * \param stream    - Stream identifier
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidDevice
 * \notefnerr
 * \note_async
 * \note_null_stream
 *
 * \sa ::cudaMemcpy, ::cudaMemcpyPeer, ::cudaMemcpyAsync,
 * ::cudaMemcpy3DPeerAsync
 */
public static native @Cast("cudaError_t") int cudaMemcpyPeerAsync(Pointer dst, int dstDevice, @Const Pointer src, int srcDevice, @Cast("size_t") long count, cudaStream stream/*=0*/);
public static native @Cast("cudaError_t") int cudaMemcpyPeerAsync(Pointer dst, int dstDevice, @Const Pointer src, int srcDevice, @Cast("size_t") long count);

/**
 * \brief Copies data between host and device
 *
 * Copies \p count bytes from the memory area pointed to by \p src to the
 * CUDA array \p dst starting at the upper left corner
 * (\p wOffset, \p hOffset), where \p kind is one of ::cudaMemcpyHostToHost,
 * ::cudaMemcpyHostToDevice, ::cudaMemcpyDeviceToHost, or
 * ::cudaMemcpyDeviceToDevice, and specifies the direction of the copy.
 *
 * ::cudaMemcpyToArrayAsync() is asynchronous with respect to the host, so
 * the call may return before the copy is complete. The copy can optionally
 * be associated to a stream by passing a non-zero \p stream argument. If \p
 * kind is ::cudaMemcpyHostToDevice or ::cudaMemcpyDeviceToHost and \p stream
 * is non-zero, the copy may overlap with operations in other streams.
 *
 * \param dst     - Destination memory address
 * \param wOffset - Destination starting X offset
 * \param hOffset - Destination starting Y offset
 * \param src     - Source memory address
 * \param count   - Size in bytes to copy
 * \param kind    - Type of transfer
 * \param stream  - Stream identifier
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidDevicePointer,
 * ::cudaErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_async
 * \note_null_stream
 *
 * \sa ::cudaMemcpy, ::cudaMemcpy2D, ::cudaMemcpyToArray,
 * ::cudaMemcpy2DToArray, ::cudaMemcpyFromArray, ::cudaMemcpy2DFromArray,
 * ::cudaMemcpyArrayToArray, ::cudaMemcpy2DArrayToArray, ::cudaMemcpyToSymbol,
 * ::cudaMemcpyFromSymbol, ::cudaMemcpyAsync, ::cudaMemcpy2DAsync,
 * ::cudaMemcpy2DToArrayAsync,
 * ::cudaMemcpyFromArrayAsync, ::cudaMemcpy2DFromArrayAsync,
 * ::cudaMemcpyToSymbolAsync, ::cudaMemcpyFromSymbolAsync
 */
public static native @Cast("cudaError_t") int cudaMemcpyToArrayAsync(cudaArray dst, @Cast("size_t") long wOffset, @Cast("size_t") long hOffset, @Const Pointer src, @Cast("size_t") long count, @Cast("cudaMemcpyKind") int kind, cudaStream stream/*=0*/);
public static native @Cast("cudaError_t") int cudaMemcpyToArrayAsync(cudaArray dst, @Cast("size_t") long wOffset, @Cast("size_t") long hOffset, @Const Pointer src, @Cast("size_t") long count, @Cast("cudaMemcpyKind") int kind);

/**
 * \brief Copies data between host and device
 *
 * Copies \p count bytes from the CUDA array \p src starting at the upper
 * left corner (\p wOffset, hOffset) to the memory area pointed to by \p dst,
 * where \p kind is one of ::cudaMemcpyHostToHost, ::cudaMemcpyHostToDevice,
 * ::cudaMemcpyDeviceToHost, or ::cudaMemcpyDeviceToDevice, and specifies the
 * direction of the copy.
 *
 * ::cudaMemcpyFromArrayAsync() is asynchronous with respect to the host, so
 * the call may return before the copy is complete. The copy can optionally
 * be associated to a stream by passing a non-zero \p stream argument. If \p
 * kind is ::cudaMemcpyHostToDevice or ::cudaMemcpyDeviceToHost and \p stream
 * is non-zero, the copy may overlap with operations in other streams.
 *
 * \param dst     - Destination memory address
 * \param src     - Source memory address
 * \param wOffset - Source starting X offset
 * \param hOffset - Source starting Y offset
 * \param count   - Size in bytes to copy
 * \param kind    - Type of transfer
 * \param stream  - Stream identifier
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidDevicePointer,
 * ::cudaErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_async
 * \note_null_stream
 *
 * \sa ::cudaMemcpy, ::cudaMemcpy2D, ::cudaMemcpyToArray,
 * ::cudaMemcpy2DToArray, ::cudaMemcpyFromArray, ::cudaMemcpy2DFromArray,
 * ::cudaMemcpyArrayToArray, ::cudaMemcpy2DArrayToArray, ::cudaMemcpyToSymbol,
 * ::cudaMemcpyFromSymbol, ::cudaMemcpyAsync, ::cudaMemcpy2DAsync,
 * ::cudaMemcpyToArrayAsync, ::cudaMemcpy2DToArrayAsync,
 * ::cudaMemcpy2DFromArrayAsync,
 * ::cudaMemcpyToSymbolAsync, ::cudaMemcpyFromSymbolAsync
 */
public static native @Cast("cudaError_t") int cudaMemcpyFromArrayAsync(Pointer dst, cudaArray src, @Cast("size_t") long wOffset, @Cast("size_t") long hOffset, @Cast("size_t") long count, @Cast("cudaMemcpyKind") int kind, cudaStream stream/*=0*/);
public static native @Cast("cudaError_t") int cudaMemcpyFromArrayAsync(Pointer dst, cudaArray src, @Cast("size_t") long wOffset, @Cast("size_t") long hOffset, @Cast("size_t") long count, @Cast("cudaMemcpyKind") int kind);

/**
 * \brief Copies data between host and device
 *
 * Copies a matrix (\p height rows of \p width bytes each) from the memory
 * area pointed to by \p src to the memory area pointed to by \p dst, where
 * \p kind is one of ::cudaMemcpyHostToHost, ::cudaMemcpyHostToDevice,
 * ::cudaMemcpyDeviceToHost, or ::cudaMemcpyDeviceToDevice, and specifies the
 * direction of the copy. \p dpitch and \p spitch are the widths in memory in
 * bytes of the 2D arrays pointed to by \p dst and \p src, including any
 * padding added to the end of each row. The memory areas may not overlap.
 * \p width must not exceed either \p dpitch or \p spitch.
 * Calling ::cudaMemcpy2DAsync() with \p dst and \p src pointers that do not
 * match the direction of the copy results in an undefined behavior.
 * ::cudaMemcpy2DAsync() returns an error if \p dpitch or \p spitch is greater
 * than the maximum allowed.
 *
 * ::cudaMemcpy2DAsync() is asynchronous with respect to the host, so
 * the call may return before the copy is complete. The copy can optionally
 * be associated to a stream by passing a non-zero \p stream argument. If
 * \p kind is ::cudaMemcpyHostToDevice or ::cudaMemcpyDeviceToHost and
 * \p stream is non-zero, the copy may overlap with operations in other
 * streams.
 *
 * \param dst    - Destination memory address
 * \param dpitch - Pitch of destination memory
 * \param src    - Source memory address
 * \param spitch - Pitch of source memory
 * \param width  - Width of matrix transfer (columns in bytes)
 * \param height - Height of matrix transfer (rows)
 * \param kind   - Type of transfer
 * \param stream - Stream identifier
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidPitchValue,
 * ::cudaErrorInvalidDevicePointer,
 * ::cudaErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_async
 * \note_null_stream
 *
 * \sa ::cudaMemcpy, ::cudaMemcpy2D, ::cudaMemcpyToArray,
 * ::cudaMemcpy2DToArray, ::cudaMemcpyFromArray, ::cudaMemcpy2DFromArray,
 * ::cudaMemcpyArrayToArray, ::cudaMemcpy2DArrayToArray, ::cudaMemcpyToSymbol,
 * ::cudaMemcpyFromSymbol, ::cudaMemcpyAsync,
 * ::cudaMemcpyToArrayAsync, ::cudaMemcpy2DToArrayAsync,
 * ::cudaMemcpyFromArrayAsync, ::cudaMemcpy2DFromArrayAsync,
 * ::cudaMemcpyToSymbolAsync, ::cudaMemcpyFromSymbolAsync
 */
public static native @Cast("cudaError_t") int cudaMemcpy2DAsync(Pointer dst, @Cast("size_t") long dpitch, @Const Pointer src, @Cast("size_t") long spitch, @Cast("size_t") long width, @Cast("size_t") long height, @Cast("cudaMemcpyKind") int kind, cudaStream stream/*=0*/);
public static native @Cast("cudaError_t") int cudaMemcpy2DAsync(Pointer dst, @Cast("size_t") long dpitch, @Const Pointer src, @Cast("size_t") long spitch, @Cast("size_t") long width, @Cast("size_t") long height, @Cast("cudaMemcpyKind") int kind);

/**
 * \brief Copies data between host and device
 *
 * Copies a matrix (\p height rows of \p width bytes each) from the memory
 * area pointed to by \p src to the CUDA array \p dst starting at the
 * upper left corner (\p wOffset, \p hOffset) where \p kind is one of
 * ::cudaMemcpyHostToHost, ::cudaMemcpyHostToDevice, ::cudaMemcpyDeviceToHost,
 * or ::cudaMemcpyDeviceToDevice, and specifies the direction of the copy.
 * \p spitch is the width in memory in bytes of the 2D array pointed to by
 * \p src, including any padding added to the end of each row. \p wOffset +
 * \p width must not exceed the width of the CUDA array \p dst. \p width must
 * not exceed \p spitch. ::cudaMemcpy2DToArrayAsync() returns an error if
 * \p spitch exceeds the maximum allowed.
 *
 * ::cudaMemcpy2DToArrayAsync() is asynchronous with respect to the host, so
 * the call may return before the copy is complete. The copy can optionally
 * be associated to a stream by passing a non-zero \p stream argument. If
 * \p kind is ::cudaMemcpyHostToDevice or ::cudaMemcpyDeviceToHost and
 * \p stream is non-zero, the copy may overlap with operations in other
 * streams.
 *
 * \param dst     - Destination memory address
 * \param wOffset - Destination starting X offset
 * \param hOffset - Destination starting Y offset
 * \param src     - Source memory address
 * \param spitch  - Pitch of source memory
 * \param width   - Width of matrix transfer (columns in bytes)
 * \param height  - Height of matrix transfer (rows)
 * \param kind    - Type of transfer
 * \param stream  - Stream identifier
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidDevicePointer,
 * ::cudaErrorInvalidPitchValue,
 * ::cudaErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_async
 * \note_null_stream
 *
 * \sa ::cudaMemcpy, ::cudaMemcpy2D, ::cudaMemcpyToArray,
 * ::cudaMemcpy2DToArray, ::cudaMemcpyFromArray, ::cudaMemcpy2DFromArray,
 * ::cudaMemcpyArrayToArray, ::cudaMemcpy2DArrayToArray, ::cudaMemcpyToSymbol,
 * ::cudaMemcpyFromSymbol, ::cudaMemcpyAsync, ::cudaMemcpy2DAsync,
 * ::cudaMemcpyToArrayAsync,
 * ::cudaMemcpyFromArrayAsync, ::cudaMemcpy2DFromArrayAsync,
 * ::cudaMemcpyToSymbolAsync, ::cudaMemcpyFromSymbolAsync
 */
public static native @Cast("cudaError_t") int cudaMemcpy2DToArrayAsync(cudaArray dst, @Cast("size_t") long wOffset, @Cast("size_t") long hOffset, @Const Pointer src, @Cast("size_t") long spitch, @Cast("size_t") long width, @Cast("size_t") long height, @Cast("cudaMemcpyKind") int kind, cudaStream stream/*=0*/);
public static native @Cast("cudaError_t") int cudaMemcpy2DToArrayAsync(cudaArray dst, @Cast("size_t") long wOffset, @Cast("size_t") long hOffset, @Const Pointer src, @Cast("size_t") long spitch, @Cast("size_t") long width, @Cast("size_t") long height, @Cast("cudaMemcpyKind") int kind);

/**
 * \brief Copies data between host and device
 *
 * Copies a matrix (\p height rows of \p width bytes each) from the CUDA
 * array \p srcArray starting at the upper left corner
 * (\p wOffset, \p hOffset) to the memory area pointed to by \p dst, where
 * \p kind is one of ::cudaMemcpyHostToHost, ::cudaMemcpyHostToDevice,
 * ::cudaMemcpyDeviceToHost, or ::cudaMemcpyDeviceToDevice, and specifies the
 * direction of the copy. \p dpitch is the width in memory in bytes of the 2D
 * array pointed to by \p dst, including any padding added to the end of each
 * row. \p wOffset + \p width must not exceed the width of the CUDA array
 * \p src. \p width must not exceed \p dpitch. ::cudaMemcpy2DFromArrayAsync()
 * returns an error if \p dpitch exceeds the maximum allowed.
 *
 * ::cudaMemcpy2DFromArrayAsync() is asynchronous with respect to the host, so
 * the call may return before the copy is complete. The copy can optionally be
 * associated to a stream by passing a non-zero \p stream argument. If \p kind
 * is ::cudaMemcpyHostToDevice or ::cudaMemcpyDeviceToHost and \p stream is
 * non-zero, the copy may overlap with operations in other streams.
 *
 * \param dst     - Destination memory address
 * \param dpitch  - Pitch of destination memory
 * \param src     - Source memory address
 * \param wOffset - Source starting X offset
 * \param hOffset - Source starting Y offset
 * \param width   - Width of matrix transfer (columns in bytes)
 * \param height  - Height of matrix transfer (rows)
 * \param kind    - Type of transfer
 * \param stream  - Stream identifier
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidDevicePointer,
 * ::cudaErrorInvalidPitchValue,
 * ::cudaErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_async
 * \note_null_stream
 *
 * \sa ::cudaMemcpy, ::cudaMemcpy2D, ::cudaMemcpyToArray,
 * ::cudaMemcpy2DToArray, ::cudaMemcpyFromArray, ::cudaMemcpy2DFromArray,
 * ::cudaMemcpyArrayToArray, ::cudaMemcpy2DArrayToArray, ::cudaMemcpyToSymbol,
 * ::cudaMemcpyFromSymbol, ::cudaMemcpyAsync, ::cudaMemcpy2DAsync,
 * ::cudaMemcpyToArrayAsync, ::cudaMemcpy2DToArrayAsync,
 * ::cudaMemcpyFromArrayAsync,
 * ::cudaMemcpyToSymbolAsync, ::cudaMemcpyFromSymbolAsync
 */
public static native @Cast("cudaError_t") int cudaMemcpy2DFromArrayAsync(Pointer dst, @Cast("size_t") long dpitch, cudaArray src, @Cast("size_t") long wOffset, @Cast("size_t") long hOffset, @Cast("size_t") long width, @Cast("size_t") long height, @Cast("cudaMemcpyKind") int kind, cudaStream stream/*=0*/);
public static native @Cast("cudaError_t") int cudaMemcpy2DFromArrayAsync(Pointer dst, @Cast("size_t") long dpitch, cudaArray src, @Cast("size_t") long wOffset, @Cast("size_t") long hOffset, @Cast("size_t") long width, @Cast("size_t") long height, @Cast("cudaMemcpyKind") int kind);

/**
 * \brief Copies data to the given symbol on the device
 *
 * Copies \p count bytes from the memory area pointed to by \p src
 * to the memory area pointed to by \p offset bytes from the start of symbol
 * \p symbol. The memory areas may not overlap. \p symbol is a variable that
 * resides in global or constant memory space. \p kind can be either
 * ::cudaMemcpyHostToDevice or ::cudaMemcpyDeviceToDevice.
 *
 * ::cudaMemcpyToSymbolAsync() is asynchronous with respect to the host, so
 * the call may return before the copy is complete. The copy can optionally
 * be associated to a stream by passing a non-zero \p stream argument. If
 * \p kind is ::cudaMemcpyHostToDevice and \p stream is non-zero, the copy
 * may overlap with operations in other streams.
 *
 * \param symbol - Device symbol address
 * \param src    - Source memory address
 * \param count  - Size in bytes to copy
 * \param offset - Offset from start of symbol in bytes
 * \param kind   - Type of transfer
 * \param stream - Stream identifier
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidSymbol,
 * ::cudaErrorInvalidDevicePointer,
 * ::cudaErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_async
 * \note_null_stream
 * \note_string_api_deprecation
 *
 * \sa ::cudaMemcpy, ::cudaMemcpy2D, ::cudaMemcpyToArray,
 * ::cudaMemcpy2DToArray, ::cudaMemcpyFromArray, ::cudaMemcpy2DFromArray,
 * ::cudaMemcpyArrayToArray, ::cudaMemcpy2DArrayToArray, ::cudaMemcpyToSymbol,
 * ::cudaMemcpyFromSymbol, ::cudaMemcpyAsync, ::cudaMemcpy2DAsync,
 * ::cudaMemcpyToArrayAsync, ::cudaMemcpy2DToArrayAsync,
 * ::cudaMemcpyFromArrayAsync, ::cudaMemcpy2DFromArrayAsync,
 * ::cudaMemcpyFromSymbolAsync
 */
public static native @Cast("cudaError_t") int cudaMemcpyToSymbolAsync(@Const Pointer symbol, @Const Pointer src, @Cast("size_t") long count, @Cast("size_t") long offset, @Cast("cudaMemcpyKind") int kind, cudaStream stream/*=0*/);
public static native @Cast("cudaError_t") int cudaMemcpyToSymbolAsync(@Const Pointer symbol, @Const Pointer src, @Cast("size_t") long count, @Cast("size_t") long offset, @Cast("cudaMemcpyKind") int kind);

/**
 * \brief Copies data from the given symbol on the device
 *
 * Copies \p count bytes from the memory area pointed to by \p offset bytes
 * from the start of symbol \p symbol to the memory area pointed to by \p dst.
 * The memory areas may not overlap. \p symbol is a variable that resides in
 * global or constant memory space. \p kind can be either
 * ::cudaMemcpyDeviceToHost or ::cudaMemcpyDeviceToDevice.
 *
 * ::cudaMemcpyFromSymbolAsync() is asynchronous with respect to the host, so
 * the call may return before the copy is complete. The copy can optionally be
 * associated to a stream by passing a non-zero \p stream argument. If \p kind
 * is ::cudaMemcpyDeviceToHost and \p stream is non-zero, the copy may overlap
 * with operations in other streams.
 *
 * \param dst    - Destination memory address
 * \param symbol - Device symbol address
 * \param count  - Size in bytes to copy
 * \param offset - Offset from start of symbol in bytes
 * \param kind   - Type of transfer
 * \param stream - Stream identifier
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidSymbol,
 * ::cudaErrorInvalidDevicePointer,
 * ::cudaErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_async
 * \note_null_stream
 * \note_string_api_deprecation
 *
 * \sa ::cudaMemcpy, ::cudaMemcpy2D, ::cudaMemcpyToArray,
 * ::cudaMemcpy2DToArray, ::cudaMemcpyFromArray, ::cudaMemcpy2DFromArray,
 * ::cudaMemcpyArrayToArray, ::cudaMemcpy2DArrayToArray, ::cudaMemcpyToSymbol,
 * ::cudaMemcpyFromSymbol, ::cudaMemcpyAsync, ::cudaMemcpy2DAsync,
 * ::cudaMemcpyToArrayAsync, ::cudaMemcpy2DToArrayAsync,
 * ::cudaMemcpyFromArrayAsync, ::cudaMemcpy2DFromArrayAsync,
 * ::cudaMemcpyToSymbolAsync
 */
public static native @Cast("cudaError_t") int cudaMemcpyFromSymbolAsync(Pointer dst, @Const Pointer symbol, @Cast("size_t") long count, @Cast("size_t") long offset, @Cast("cudaMemcpyKind") int kind, cudaStream stream/*=0*/);
public static native @Cast("cudaError_t") int cudaMemcpyFromSymbolAsync(Pointer dst, @Const Pointer symbol, @Cast("size_t") long count, @Cast("size_t") long offset, @Cast("cudaMemcpyKind") int kind);


/**
 * \brief Initializes or sets device memory to a value
 *
 * Fills the first \p count bytes of the memory area pointed to by \p devPtr
 * with the constant byte value \p value.
 *
 * Note that this function is asynchronous with respect to the host unless
 * \p devPtr refers to pinned host memory.
 *
 * \param devPtr - Pointer to device memory
 * \param value  - Value to set for each byte of specified memory
 * \param count  - Size in bytes to set
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidDevicePointer
 * \notefnerr
 * \note_memset
 *
 * \sa ::cudaMemset2D, ::cudaMemset3D, ::cudaMemsetAsync,
 * ::cudaMemset2DAsync, ::cudaMemset3DAsync
 */
public static native @Cast("cudaError_t") int cudaMemset(Pointer devPtr, int value, @Cast("size_t") long count);

/**
 * \brief Initializes or sets device memory to a value
 *
 * Sets to the specified value \p value a matrix (\p height rows of \p width
 * bytes each) pointed to by \p dstPtr. \p pitch is the width in bytes of the
 * 2D array pointed to by \p dstPtr, including any padding added to the end
 * of each row. This function performs fastest when the pitch is one that has
 * been passed back by ::cudaMallocPitch().
 *
 * Note that this function is asynchronous with respect to the host unless
 * \p devPtr refers to pinned host memory.
 *
 * \param devPtr - Pointer to 2D device memory
 * \param pitch  - Pitch in bytes of 2D device memory
 * \param value  - Value to set for each byte of specified memory
 * \param width  - Width of matrix set (columns in bytes)
 * \param height - Height of matrix set (rows)
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidDevicePointer
 * \notefnerr
 * \note_memset
 *
 * \sa ::cudaMemset, ::cudaMemset3D, ::cudaMemsetAsync,
 * ::cudaMemset2DAsync, ::cudaMemset3DAsync
 */
public static native @Cast("cudaError_t") int cudaMemset2D(Pointer devPtr, @Cast("size_t") long pitch, int value, @Cast("size_t") long width, @Cast("size_t") long height);

/**
 * \brief Initializes or sets device memory to a value
 *
 * Initializes each element of a 3D array to the specified value \p value.
 * The object to initialize is defined by \p pitchedDevPtr. The \p pitch field
 * of \p pitchedDevPtr is the width in memory in bytes of the 3D array pointed
 * to by \p pitchedDevPtr, including any padding added to the end of each row.
 * The \p xsize field specifies the logical width of each row in bytes, while
 * the \p ysize field specifies the height of each 2D slice in rows.
 *
 * The extents of the initialized region are specified as a \p width in bytes,
 * a \p height in rows, and a \p depth in slices.
 *
 * Extents with \p width greater than or equal to the \p xsize of
 * \p pitchedDevPtr may perform significantly faster than extents narrower
 * than the \p xsize. Secondarily, extents with \p height equal to the
 * \p ysize of \p pitchedDevPtr will perform faster than when the \p height is
 * shorter than the \p ysize.
 *
 * This function performs fastest when the \p pitchedDevPtr has been allocated
 * by ::cudaMalloc3D().
 *
 * Note that this function is asynchronous with respect to the host unless
 * \p pitchedDevPtr refers to pinned host memory.
 *
 * \param pitchedDevPtr - Pointer to pitched device memory
 * \param value         - Value to set for each byte of specified memory
 * \param extent        - Size parameters for where to set device memory (\p width field in bytes)
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidDevicePointer
 * \notefnerr
 * \note_memset
 *
 * \sa ::cudaMemset, ::cudaMemset2D,
 * ::cudaMemsetAsync, ::cudaMemset2DAsync, ::cudaMemset3DAsync,
 * ::cudaMalloc3D, ::make_cudaPitchedPtr,
 * ::make_cudaExtent
 */
public static native @Cast("cudaError_t") int cudaMemset3D(@ByVal cudaPitchedPtr pitchedDevPtr, int value, @ByVal cudaExtent extent);

/**
 * \brief Initializes or sets device memory to a value
 *
 * Fills the first \p count bytes of the memory area pointed to by \p devPtr
 * with the constant byte value \p value.
 *
 * ::cudaMemsetAsync() is asynchronous with respect to the host, so
 * the call may return before the memset is complete. The operation can optionally
 * be associated to a stream by passing a non-zero \p stream argument.
 * If \p stream is non-zero, the operation may overlap with operations in other streams.
 *
 * \param devPtr - Pointer to device memory
 * \param value  - Value to set for each byte of specified memory
 * \param count  - Size in bytes to set
 * \param stream - Stream identifier
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidDevicePointer
 * \notefnerr
 * \note_memset
 * \note_null_stream
 *
 * \sa ::cudaMemset, ::cudaMemset2D, ::cudaMemset3D,
 * ::cudaMemset2DAsync, ::cudaMemset3DAsync
 */
public static native @Cast("cudaError_t") int cudaMemsetAsync(Pointer devPtr, int value, @Cast("size_t") long count, cudaStream stream/*=0*/);
public static native @Cast("cudaError_t") int cudaMemsetAsync(Pointer devPtr, int value, @Cast("size_t") long count);

/**
 * \brief Initializes or sets device memory to a value
 *
 * Sets to the specified value \p value a matrix (\p height rows of \p width
 * bytes each) pointed to by \p dstPtr. \p pitch is the width in bytes of the
 * 2D array pointed to by \p dstPtr, including any padding added to the end
 * of each row. This function performs fastest when the pitch is one that has
 * been passed back by ::cudaMallocPitch().
 *
 * ::cudaMemset2DAsync() is asynchronous with respect to the host, so
 * the call may return before the memset is complete. The operation can optionally
 * be associated to a stream by passing a non-zero \p stream argument.
 * If \p stream is non-zero, the operation may overlap with operations in other streams.
 *
 * \param devPtr - Pointer to 2D device memory
 * \param pitch  - Pitch in bytes of 2D device memory
 * \param value  - Value to set for each byte of specified memory
 * \param width  - Width of matrix set (columns in bytes)
 * \param height - Height of matrix set (rows)
 * \param stream - Stream identifier
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidDevicePointer
 * \notefnerr
 * \note_memset
 * \note_null_stream
 *
 * \sa ::cudaMemset, ::cudaMemset2D, ::cudaMemset3D,
 * ::cudaMemsetAsync, ::cudaMemset3DAsync
 */
public static native @Cast("cudaError_t") int cudaMemset2DAsync(Pointer devPtr, @Cast("size_t") long pitch, int value, @Cast("size_t") long width, @Cast("size_t") long height, cudaStream stream/*=0*/);
public static native @Cast("cudaError_t") int cudaMemset2DAsync(Pointer devPtr, @Cast("size_t") long pitch, int value, @Cast("size_t") long width, @Cast("size_t") long height);

/**
 * \brief Initializes or sets device memory to a value
 *
 * Initializes each element of a 3D array to the specified value \p value.
 * The object to initialize is defined by \p pitchedDevPtr. The \p pitch field
 * of \p pitchedDevPtr is the width in memory in bytes of the 3D array pointed
 * to by \p pitchedDevPtr, including any padding added to the end of each row.
 * The \p xsize field specifies the logical width of each row in bytes, while
 * the \p ysize field specifies the height of each 2D slice in rows.
 *
 * The extents of the initialized region are specified as a \p width in bytes,
 * a \p height in rows, and a \p depth in slices.
 *
 * Extents with \p width greater than or equal to the \p xsize of
 * \p pitchedDevPtr may perform significantly faster than extents narrower
 * than the \p xsize. Secondarily, extents with \p height equal to the
 * \p ysize of \p pitchedDevPtr will perform faster than when the \p height is
 * shorter than the \p ysize.
 *
 * This function performs fastest when the \p pitchedDevPtr has been allocated
 * by ::cudaMalloc3D().
 *
 * ::cudaMemset3DAsync() is asynchronous with respect to the host, so
 * the call may return before the memset is complete. The operation can optionally
 * be associated to a stream by passing a non-zero \p stream argument.
 * If \p stream is non-zero, the operation may overlap with operations in other streams.
 *
 * \param pitchedDevPtr - Pointer to pitched device memory
 * \param value         - Value to set for each byte of specified memory
 * \param extent        - Size parameters for where to set device memory (\p width field in bytes)
 * \param stream - Stream identifier
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidDevicePointer
 * \notefnerr
 * \note_memset
 * \note_null_stream
 *
 * \sa ::cudaMemset, ::cudaMemset2D, ::cudaMemset3D,
 * ::cudaMemsetAsync, ::cudaMemset2DAsync,
 * ::cudaMalloc3D, ::make_cudaPitchedPtr,
 * ::make_cudaExtent
 */
public static native @Cast("cudaError_t") int cudaMemset3DAsync(@ByVal cudaPitchedPtr pitchedDevPtr, int value, @ByVal cudaExtent extent, cudaStream stream/*=0*/);
public static native @Cast("cudaError_t") int cudaMemset3DAsync(@ByVal cudaPitchedPtr pitchedDevPtr, int value, @ByVal cudaExtent extent);

/**
 * \brief Finds the address associated with a CUDA symbol
 *
 * Returns in \p *devPtr the address of symbol \p symbol on the device.
 * \p symbol is a variable that resides in global or constant memory space.
 * If \p symbol cannot be found, or if \p symbol is not declared in the
 * global or constant memory space, \p *devPtr is unchanged and the error
 * ::cudaErrorInvalidSymbol is returned.
 *
 * \param devPtr - Return device pointer associated with symbol
 * \param symbol - Device symbol address
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidSymbol
 * \notefnerr
 * \note_string_api_deprecation
 *
 * \sa \ref ::cudaGetSymbolAddress(void**, const T&) "cudaGetSymbolAddress (C++ API)",
 * \ref ::cudaGetSymbolSize(size_t*, const void*) "cudaGetSymbolSize (C API)"
 */
public static native @Cast("cudaError_t") int cudaGetSymbolAddress(@Cast("void**") PointerPointer devPtr, @Const Pointer symbol);
public static native @Cast("cudaError_t") int cudaGetSymbolAddress(@Cast("void**") @ByPtrPtr Pointer devPtr, @Const Pointer symbol);

/**
 * \brief Finds the size of the object associated with a CUDA symbol
 *
 * Returns in \p *size the size of symbol \p symbol. \p symbol is a variable that
 * resides in global or constant memory space. If \p symbol cannot be found, or
 * if \p symbol is not declared in global or constant memory space, \p *size is
 * unchanged and the error ::cudaErrorInvalidSymbol is returned.
 *
 * \param size   - Size of object associated with symbol
 * \param symbol - Device symbol address
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidSymbol
 * \notefnerr
 * \note_string_api_deprecation
 *
 * \sa \ref ::cudaGetSymbolAddress(void**, const void*) "cudaGetSymbolAddress (C API)",
 * \ref ::cudaGetSymbolSize(size_t*, const T&) "cudaGetSymbolSize (C++ API)"
 */
public static native @Cast("cudaError_t") int cudaGetSymbolSize(@Cast("size_t*") SizeTPointer size, @Const Pointer symbol);

/** @} */ /* END CUDART_MEMORY */

/**
 * \defgroup CUDART_UNIFIED Unified Addressing
 *
 * ___MANBRIEF___ unified addressing functions of the CUDA runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the unified addressing functions of the CUDA 
 * runtime application programming interface.
 *
 * @{
 *
 * \section CUDART_UNIFIED_overview Overview
 *
 * CUDA devices can share a unified address space with the host.  
 * For these devices there is no distinction between a device
 * pointer and a host pointer -- the same pointer value may be 
 * used to access memory from the host program and from a kernel 
 * running on the device (with exceptions enumerated below).
 *
 * \section CUDART_UNIFIED_support Supported Platforms
 * 
 * Whether or not a device supports unified addressing may be 
 * queried by calling ::cudaGetDeviceProperties() with the device 
 * property ::cudaDeviceProp::unifiedAddressing.
 *
 * Unified addressing is automatically enabled in 64-bit processes 
 * on devices with compute capability greater than or equal to 2.0.
 *
 * Unified addressing is not yet supported on Windows Vista or
 * Windows 7 for devices that do not use the TCC driver model.
 *
 * \section CUDART_UNIFIED_lookup Looking Up Information from Pointer Values
 *
 * It is possible to look up information about the memory which backs a 
 * pointer value.  For instance, one may want to know if a pointer points
 * to host or device memory.  As another example, in the case of device 
 * memory, one may want to know on which CUDA device the memory 
 * resides.  These properties may be queried using the function 
 * ::cudaPointerGetAttributes()
 *
 * Since pointers are unique, it is not necessary to specify information
 * about the pointers specified to ::cudaMemcpy() and other copy functions.  
 * The copy direction ::cudaMemcpyDefault may be used to specify that the 
 * CUDA runtime should infer the location of the pointer from its value.
 *
 * \section CUDART_UNIFIED_automaphost Automatic Mapping of Host Allocated Host Memory
 *
 * All host memory allocated through all devices using ::cudaMallocHost() and
 * ::cudaHostAlloc() is always directly accessible from all devices that 
 * support unified addressing.  This is the case regardless of whether or 
 * not the flags ::cudaHostAllocPortable and ::cudaHostAllocMapped are 
 * specified.
 *
 * The pointer value through which allocated host memory may be accessed 
 * in kernels on all devices that support unified addressing is the same 
 * as the pointer value through which that memory is accessed on the host.
 * It is not necessary to call ::cudaHostGetDevicePointer() to get the device 
 * pointer for these allocations.  
 *
 * Note that this is not the case for memory allocated using the flag
 * ::cudaHostAllocWriteCombined, as discussed below.
 *
 * \section CUDART_UNIFIED_autopeerregister Direct Access of Peer Memory
 
 * Upon enabling direct access from a device that supports unified addressing 
 * to another peer device that supports unified addressing using 
 * ::cudaDeviceEnablePeerAccess() all memory allocated in the peer device using 
 * ::cudaMalloc() and ::cudaMallocPitch() will immediately be accessible 
 * by the current device.  The device pointer value through 
 * which any peer's memory may be accessed in the current device 
 * is the same pointer value through which that memory may be 
 * accessed from the peer device. 
 *
 * \section CUDART_UNIFIED_exceptions Exceptions, Disjoint Addressing
 * 
 * Not all memory may be accessed on devices through the same pointer
 * value through which they are accessed on the host.  These exceptions
 * are host memory registered using ::cudaHostRegister() and host memory
 * allocated using the flag ::cudaHostAllocWriteCombined.  For these 
 * exceptions, there exists a distinct host and device address for the
 * memory.  The device address is guaranteed to not overlap any valid host
 * pointer range and is guaranteed to have the same value across all devices
 * that support unified addressing.  
 * 
 * This device address may be queried using ::cudaHostGetDevicePointer() 
 * when a device using unified addressing is current.  Either the host 
 * or the unified device pointer value may be used to refer to this memory 
 * in ::cudaMemcpy() and similar functions using the ::cudaMemcpyDefault 
 * memory direction.
 *
 */

/**
 * \brief Returns attributes about a specified pointer
 *
 * Returns in \p *attributes the attributes of the pointer \p ptr.
 * If pointer was not allocated in, mapped by or registered with context
 * supporting unified addressing ::cudaErrorInvalidValue is returned.
 *
 * The ::cudaPointerAttributes structure is defined as:
 * \code
    struct cudaPointerAttributes {
        enum cudaMemoryType memoryType;
        int device;
        void *devicePointer;
        void *hostPointer;
        int isManaged;
    }
    \endcode
 * In this structure, the individual fields mean
 *
 * - \ref ::cudaPointerAttributes::memoryType "memoryType" identifies the physical 
 *   location of the memory associated with pointer \p ptr.  It can be
 *   ::cudaMemoryTypeHost for host memory or ::cudaMemoryTypeDevice for device
 *   memory.
 *
 * - \ref ::cudaPointerAttributes::device "device" is the device against which
 *   \p ptr was allocated.  If \p ptr has memory type ::cudaMemoryTypeDevice
 *   then this identifies the device on which the memory referred to by \p ptr
 *   physically resides.  If \p ptr has memory type ::cudaMemoryTypeHost then this
 *   identifies the device which was current when the allocation was made
 *   (and if that device is deinitialized then this allocation will vanish
 *   with that device's state).
 *
 * - \ref ::cudaPointerAttributes::devicePointer "devicePointer" is
 *   the device pointer alias through which the memory referred to by \p ptr
 *   may be accessed on the current device.
 *   If the memory referred to by \p ptr cannot be accessed directly by the 
 *   current device then this is NULL.  
 *
 * - \ref ::cudaPointerAttributes::hostPointer "hostPointer" is
 *   the host pointer alias through which the memory referred to by \p ptr
 *   may be accessed on the host.
 *   If the memory referred to by \p ptr cannot be accessed directly by the
 *   host then this is NULL.
 *
 * - \ref ::cudaPointerAttributes::isManaged "isManaged" indicates if
 *   the pointer \p ptr points to managed memory or not.
 *
 * \param attributes - Attributes for the specified pointer
 * \param ptr        - Pointer to get attributes for
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDevice
 * ::cudaErrorInvalidValue
 *
 * \sa ::cudaGetDeviceCount, ::cudaGetDevice, ::cudaSetDevice,
 * ::cudaChooseDevice
 */
public static native @Cast("cudaError_t") int cudaPointerGetAttributes(cudaPointerAttributes attributes, @Const Pointer ptr);

/** @} */ /* END CUDART_UNIFIED */

/**
 * \defgroup CUDART_PEER Peer Device Memory Access
 *
 * ___MANBRIEF___ peer device memory access functions of the CUDA runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the peer device memory access functions of the CUDA runtime
 * application programming interface.
 *
 * @{
 */

/**
 * \brief Queries if a device may directly access a peer device's memory.
 *
 * Returns in \p *canAccessPeer a value of 1 if device \p device is capable of
 * directly accessing memory from \p peerDevice and 0 otherwise.  If direct
 * access of \p peerDevice from \p device is possible, then access may be
 * enabled by calling ::cudaDeviceEnablePeerAccess().
 *
 * \param canAccessPeer - Returned access capability
 * \param device        - Device from which allocations on \p peerDevice are to
 *                        be directly accessed.
 * \param peerDevice    - Device on which the allocations to be directly accessed 
 *                        by \p device reside.
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDevice
 * \notefnerr
 *
 * \sa ::cudaDeviceEnablePeerAccess,
 * ::cudaDeviceDisablePeerAccess
 */
public static native @Cast("cudaError_t") int cudaDeviceCanAccessPeer(IntPointer canAccessPeer, int device, int peerDevice);
public static native @Cast("cudaError_t") int cudaDeviceCanAccessPeer(IntBuffer canAccessPeer, int device, int peerDevice);
public static native @Cast("cudaError_t") int cudaDeviceCanAccessPeer(int[] canAccessPeer, int device, int peerDevice);

/**
 * \brief Enables direct access to memory allocations on a peer device.
 *
 * On success, all allocations from \p peerDevice will immediately be accessible by
 * the current device.  They will remain accessible until access is explicitly
 * disabled using ::cudaDeviceDisablePeerAccess() or either device is reset using
 * ::cudaDeviceReset().
 *
 * Note that access granted by this call is unidirectional and that in order to access
 * memory on the current device from \p peerDevice, a separate symmetric call 
 * to ::cudaDeviceEnablePeerAccess() is required.
 *
 * Peer access is not supported in 32 bit applications.
 *
 * Returns ::cudaErrorInvalidDevice if ::cudaDeviceCanAccessPeer() indicates
 * that the current device cannot directly access memory from \p peerDevice.
 *
 * Returns ::cudaErrorPeerAccessAlreadyEnabled if direct access of
 * \p peerDevice from the current device has already been enabled.
 *
 * Returns ::cudaErrorInvalidValue if \p flags is not 0.
 *
 * \param peerDevice  - Peer device to enable direct access to from the current device
 * \param flags       - Reserved for future use and must be set to 0
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDevice,
 * ::cudaErrorPeerAccessAlreadyEnabled,
 * ::cudaErrorInvalidValue
 * \notefnerr
 *
 * \sa ::cudaDeviceCanAccessPeer,
 * ::cudaDeviceDisablePeerAccess
 */
public static native @Cast("cudaError_t") int cudaDeviceEnablePeerAccess(int peerDevice, @Cast("unsigned int") int flags);

/**
 * \brief Disables direct access to memory allocations on a peer device.
 *
 * Returns ::cudaErrorPeerAccessNotEnabled if direct access to memory on
 * \p peerDevice has not yet been enabled from the current device.
 *
 * \param peerDevice - Peer device to disable direct access to
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorPeerAccessNotEnabled,
 * ::cudaErrorInvalidDevice
 * \notefnerr
 *
 * \sa ::cudaDeviceCanAccessPeer,
 * ::cudaDeviceEnablePeerAccess
 */
public static native @Cast("cudaError_t") int cudaDeviceDisablePeerAccess(int peerDevice);

/** @} */ /* END CUDART_PEER */

/** \defgroup CUDART_OPENGL OpenGL Interoperability */

/** \defgroup CUDART_OPENGL_DEPRECATED OpenGL Interoperability [DEPRECATED] */

/** \defgroup CUDART_D3D9 Direct3D 9 Interoperability */

/** \defgroup CUDART_D3D9_DEPRECATED Direct3D 9 Interoperability [DEPRECATED] */

/** \defgroup CUDART_D3D10 Direct3D 10 Interoperability */

/** \defgroup CUDART_D3D10_DEPRECATED Direct3D 10 Interoperability [DEPRECATED] */

/** \defgroup CUDART_D3D11 Direct3D 11 Interoperability */

/** \defgroup CUDART_D3D11_DEPRECATED Direct3D 11 Interoperability [DEPRECATED] */

/** \defgroup CUDART_VDPAU VDPAU Interoperability */

/**
 * \defgroup CUDART_INTEROP Graphics Interoperability
 *
 * ___MANBRIEF___ graphics interoperability functions of the CUDA runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the graphics interoperability functions of the CUDA
 * runtime application programming interface.
 *
 * @{
 */

/**
 * \brief Unregisters a graphics resource for access by CUDA
 *
 * Unregisters the graphics resource \p resource so it is not accessible by
 * CUDA unless registered again.
 *
 * If \p resource is invalid then ::cudaErrorInvalidResourceHandle is
 * returned.
 *
 * \param resource - Resource to unregister
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorUnknown
 * \notefnerr
 *
 * \sa
 * ::cudaGraphicsD3D9RegisterResource,
 * ::cudaGraphicsD3D10RegisterResource,
 * ::cudaGraphicsD3D11RegisterResource,
 * ::cudaGraphicsGLRegisterBuffer,
 * ::cudaGraphicsGLRegisterImage
 */
public static native @Cast("cudaError_t") int cudaGraphicsUnregisterResource(cudaGraphicsResource resource);

/**
 * \brief Set usage flags for mapping a graphics resource
 *
 * Set \p flags for mapping the graphics resource \p resource.
 *
 * Changes to \p flags will take effect the next time \p resource is mapped.
 * The \p flags argument may be any of the following:
 * - ::cudaGraphicsMapFlagsNone: Specifies no hints about how \p resource will
 *     be used. It is therefore assumed that CUDA may read from or write to \p resource.
 * - ::cudaGraphicsMapFlagsReadOnly: Specifies that CUDA will not write to \p resource.
 * - ::cudaGraphicsMapFlagsWriteDiscard: Specifies CUDA will not read from \p resource and will
 *   write over the entire contents of \p resource, so none of the data
 *   previously stored in \p resource will be preserved.
 *
 * If \p resource is presently mapped for access by CUDA then ::cudaErrorUnknown is returned.
 * If \p flags is not one of the above values then ::cudaErrorInvalidValue is returned.
 *
 * \param resource - Registered resource to set flags for
 * \param flags    - Parameters for resource mapping
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorUnknown,
 * \notefnerr
 *
 * \sa
 * ::cudaGraphicsMapResources
 */
public static native @Cast("cudaError_t") int cudaGraphicsResourceSetMapFlags(cudaGraphicsResource resource, @Cast("unsigned int") int flags);

/**
 * \brief Map graphics resources for access by CUDA
 *
 * Maps the \p count graphics resources in \p resources for access by CUDA.
 *
 * The resources in \p resources may be accessed by CUDA until they
 * are unmapped. The graphics API from which \p resources were registered
 * should not access any resources while they are mapped by CUDA. If an
 * application does so, the results are undefined.
 *
 * This function provides the synchronization guarantee that any graphics calls
 * issued before ::cudaGraphicsMapResources() will complete before any subsequent CUDA
 * work issued in \p stream begins.
 *
 * If \p resources contains any duplicate entries then ::cudaErrorInvalidResourceHandle
 * is returned. If any of \p resources are presently mapped for access by
 * CUDA then ::cudaErrorUnknown is returned.
 *
 * \param count     - Number of resources to map
 * \param resources - Resources to map for CUDA
 * \param stream    - Stream for synchronization
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorUnknown
 * \note_null_stream
 * \notefnerr
 *
 * \sa
 * ::cudaGraphicsResourceGetMappedPointer,
 * ::cudaGraphicsSubResourceGetMappedArray,
 * ::cudaGraphicsUnmapResources
 */
public static native @Cast("cudaError_t") int cudaGraphicsMapResources(int count, @ByPtrPtr cudaGraphicsResource resources, cudaStream stream/*=0*/);
public static native @Cast("cudaError_t") int cudaGraphicsMapResources(int count, @ByPtrPtr cudaGraphicsResource resources);

/**
 * \brief Unmap graphics resources.
 *
 * Unmaps the \p count graphics resources in \p resources.
 *
 * Once unmapped, the resources in \p resources may not be accessed by CUDA
 * until they are mapped again.
 *
 * This function provides the synchronization guarantee that any CUDA work issued
 * in \p stream before ::cudaGraphicsUnmapResources() will complete before any
 * subsequently issued graphics work begins.
 *
 * If \p resources contains any duplicate entries then ::cudaErrorInvalidResourceHandle
 * is returned. If any of \p resources are not presently mapped for access by
 * CUDA then ::cudaErrorUnknown is returned.
 *
 * \param count     - Number of resources to unmap
 * \param resources - Resources to unmap
 * \param stream    - Stream for synchronization
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorUnknown
 * \note_null_stream
 * \notefnerr
 *
 * \sa
 * ::cudaGraphicsMapResources
 */
public static native @Cast("cudaError_t") int cudaGraphicsUnmapResources(int count, @ByPtrPtr cudaGraphicsResource resources, cudaStream stream/*=0*/);
public static native @Cast("cudaError_t") int cudaGraphicsUnmapResources(int count, @ByPtrPtr cudaGraphicsResource resources);

/**
 * \brief Get an device pointer through which to access a mapped graphics resource.
 *
 * Returns in \p *devPtr a pointer through which the mapped graphics resource
 * \p resource may be accessed.
 * Returns in \p *size the size of the memory in bytes which may be accessed from that pointer.
 * The value set in \p devPtr may change every time that \p resource is mapped.
 *
 * If \p resource is not a buffer then it cannot be accessed via a pointer and
 * ::cudaErrorUnknown is returned.
 * If \p resource is not mapped then ::cudaErrorUnknown is returned.
 * *
 * \param devPtr     - Returned pointer through which \p resource may be accessed
 * \param size       - Returned size of the buffer accessible starting at \p *devPtr
 * \param resource   - Mapped resource to access
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorUnknown
 * \notefnerr
 *
 * \sa
 * ::cudaGraphicsMapResources,
 * ::cudaGraphicsSubResourceGetMappedArray
 */
public static native @Cast("cudaError_t") int cudaGraphicsResourceGetMappedPointer(@Cast("void**") PointerPointer devPtr, @Cast("size_t*") SizeTPointer size, cudaGraphicsResource resource);
public static native @Cast("cudaError_t") int cudaGraphicsResourceGetMappedPointer(@Cast("void**") @ByPtrPtr Pointer devPtr, @Cast("size_t*") SizeTPointer size, cudaGraphicsResource resource);

/**
 * \brief Get an array through which to access a subresource of a mapped graphics resource.
 *
 * Returns in \p *array an array through which the subresource of the mapped
 * graphics resource \p resource which corresponds to array index \p arrayIndex
 * and mipmap level \p mipLevel may be accessed.  The value set in \p array may
 * change every time that \p resource is mapped.
 *
 * If \p resource is not a texture then it cannot be accessed via an array and
 * ::cudaErrorUnknown is returned.
 * If \p arrayIndex is not a valid array index for \p resource then
 * ::cudaErrorInvalidValue is returned.
 * If \p mipLevel is not a valid mipmap level for \p resource then
 * ::cudaErrorInvalidValue is returned.
 * If \p resource is not mapped then ::cudaErrorUnknown is returned.
 *
 * \param array       - Returned array through which a subresource of \p resource may be accessed
 * \param resource    - Mapped resource to access
 * \param arrayIndex  - Array index for array textures or cubemap face
 *                      index as defined by ::cudaGraphicsCubeFace for
 *                      cubemap textures for the subresource to access
 * \param mipLevel    - Mipmap level for the subresource to access
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorUnknown
 * \notefnerr
 *
 * \sa ::cudaGraphicsResourceGetMappedPointer
 */
public static native @Cast("cudaError_t") int cudaGraphicsSubResourceGetMappedArray(@ByPtrPtr cudaArray array, cudaGraphicsResource resource, @Cast("unsigned int") int arrayIndex, @Cast("unsigned int") int mipLevel);

/**
 * \brief Get a mipmapped array through which to access a mapped graphics resource.
 *
 * Returns in \p *mipmappedArray a mipmapped array through which the mapped
 * graphics resource \p resource may be accessed. The value set in \p mipmappedArray may
 * change every time that \p resource is mapped.
 *
 * If \p resource is not a texture then it cannot be accessed via an array and
 * ::cudaErrorUnknown is returned.
 * If \p resource is not mapped then ::cudaErrorUnknown is returned.
 *
 * \param mipmappedArray - Returned mipmapped array through which \p resource may be accessed
 * \param resource       - Mapped resource to access
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorUnknown
 * \notefnerr
 *
 * \sa ::cudaGraphicsResourceGetMappedPointer
 */
public static native @Cast("cudaError_t") int cudaGraphicsResourceGetMappedMipmappedArray(@ByPtrPtr cudaMipmappedArray mipmappedArray, cudaGraphicsResource resource);

/** @} */ /* END CUDART_INTEROP */

/**
 * \defgroup CUDART_TEXTURE Texture Reference Management
 *
 * ___MANBRIEF___ texture reference management functions of the CUDA runtime
 * API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the low level texture reference management functions
 * of the CUDA runtime application programming interface.
 *
 * Some functions have overloaded C++ API template versions documented separately in the
 * \ref CUDART_HIGHLEVEL "C++ API Routines" module.
 *
 * @{
 */

/**
 * \brief Get the channel descriptor of an array
 *
 * Returns in \p *desc the channel descriptor of the CUDA array \p array.
 *
 * \param desc  - Channel format
 * \param array - Memory array on device
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \notefnerr
 *
 * \sa \ref ::cudaCreateChannelDesc(int, int, int, int, cudaChannelFormatKind) "cudaCreateChannelDesc (C API)",
 * ::cudaGetTextureReference,
 * \ref ::cudaBindTexture(size_t*, const struct textureReference*, const void*, const struct cudaChannelFormatDesc*, size_t) "cudaBindTexture (C API)",
 * \ref ::cudaBindTexture2D(size_t*, const struct textureReference*, const void*, const struct cudaChannelFormatDesc*, size_t, size_t, size_t) "cudaBindTexture2D (C API)",
 * \ref ::cudaBindTextureToArray(const struct textureReference*, cudaArray_const_t, const struct cudaChannelFormatDesc*) "cudaBindTextureToArray (C API)",
 * \ref ::cudaUnbindTexture(const struct textureReference*) "cudaUnbindTexture (C API)",
 * \ref ::cudaGetTextureAlignmentOffset(size_t*, const struct textureReference*) "cudaGetTextureAlignmentOffset (C API)"
 */
public static native @Cast("cudaError_t") int cudaGetChannelDesc(cudaChannelFormatDesc desc, cudaArray array);

/**
 * \brief Returns a channel descriptor using the specified format
 *
 * Returns a channel descriptor with format \p f and number of bits of each
 * component \p x, \p y, \p z, and \p w.  The ::cudaChannelFormatDesc is
 * defined as:
 * \code
  struct cudaChannelFormatDesc {
    int x, y, z, w;
    enum cudaChannelFormatKind f;
  };
 * \endcode
 *
 * where ::cudaChannelFormatKind is one of ::cudaChannelFormatKindSigned,
 * ::cudaChannelFormatKindUnsigned, or ::cudaChannelFormatKindFloat.
 *
 * \param x - X component
 * \param y - Y component
 * \param z - Z component
 * \param w - W component
 * \param f - Channel format
 *
 * \return
 * Channel descriptor with format \p f
 *
 * \sa \ref ::cudaCreateChannelDesc(void) "cudaCreateChannelDesc (C++ API)",
 * ::cudaGetChannelDesc, ::cudaGetTextureReference,
 * \ref ::cudaBindTexture(size_t*, const struct textureReference*, const void*, const struct cudaChannelFormatDesc*, size_t) "cudaBindTexture (C API)",
 * \ref ::cudaBindTexture2D(size_t*, const struct textureReference*, const void*, const struct cudaChannelFormatDesc*, size_t, size_t, size_t) "cudaBindTexture2D (C API)",
 * \ref ::cudaBindTextureToArray(const struct textureReference*, cudaArray_const_t, const struct cudaChannelFormatDesc*) "cudaBindTextureToArray (C API)",
 * \ref ::cudaUnbindTexture(const struct textureReference*) "cudaUnbindTexture (C API)",
 * \ref ::cudaGetTextureAlignmentOffset(size_t*, const struct textureReference*) "cudaGetTextureAlignmentOffset (C API)"
 */
public static native @ByVal cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w, @Cast("cudaChannelFormatKind") int f);


/**
 * \brief Binds a memory area to a texture
 *
 * Binds \p size bytes of the memory area pointed to by \p devPtr to the
 * texture reference \p texref. \p desc describes how the memory is interpreted
 * when fetching values from the texture. Any memory previously bound to
 * \p texref is unbound.
 *
 * Since the hardware enforces an alignment requirement on texture base
 * addresses,
 * \ref ::cudaBindTexture(size_t*, const struct textureReference*, const void*, const struct cudaChannelFormatDesc*, size_t) "cudaBindTexture()"
 * returns in \p *offset a byte offset that
 * must be applied to texture fetches in order to read from the desired memory.
 * This offset must be divided by the texel size and passed to kernels that
 * read from the texture so they can be applied to the ::tex1Dfetch() function.
 * If the device memory pointer was returned from ::cudaMalloc(), the offset is
 * guaranteed to be 0 and NULL may be passed as the \p offset parameter.
 *
 * The total number of elements (or texels) in the linear address range
 * cannot exceed ::cudaDeviceProp::maxTexture1DLinear[0].
 * The number of elements is computed as (\p size / elementSize),
 * where elementSize is determined from \p desc.
 *
 * \param offset - Offset in bytes
 * \param texref - Texture to bind
 * \param devPtr - Memory area on device
 * \param desc   - Channel format
 * \param size   - Size of the memory area pointed to by devPtr
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidDevicePointer,
 * ::cudaErrorInvalidTexture
 * \notefnerr
 *
 * \sa \ref ::cudaCreateChannelDesc(int, int, int, int, cudaChannelFormatKind) "cudaCreateChannelDesc (C API)",
 * ::cudaGetChannelDesc, ::cudaGetTextureReference,
 * \ref ::cudaBindTexture(size_t*, const struct texture< T, dim, readMode>&, const void*, const struct cudaChannelFormatDesc&, size_t) "cudaBindTexture (C++ API)",
 * \ref ::cudaBindTexture2D(size_t*, const struct textureReference*, const void*, const struct cudaChannelFormatDesc*, size_t, size_t, size_t) "cudaBindTexture2D (C API)",
 * \ref ::cudaBindTextureToArray(const struct textureReference*, cudaArray_const_t, const struct cudaChannelFormatDesc*) "cudaBindTextureToArray (C API)",
 * \ref ::cudaUnbindTexture(const struct textureReference*) "cudaUnbindTexture (C API)",
 * \ref ::cudaGetTextureAlignmentOffset(size_t*, const struct textureReference*) "cudaGetTextureAlignmentOffset (C API)"
 */
public static native @Cast("cudaError_t") int cudaBindTexture(@Cast("size_t*") SizeTPointer offset, @Const textureReference texref, @Const Pointer devPtr, @Const cudaChannelFormatDesc desc, @Cast("size_t") long size/*=UINT_MAX*/);
public static native @Cast("cudaError_t") int cudaBindTexture(@Cast("size_t*") SizeTPointer offset, @Const textureReference texref, @Const Pointer devPtr, @Const cudaChannelFormatDesc desc);

/**
 * \brief Binds a 2D memory area to a texture
 *
 * Binds the 2D memory area pointed to by \p devPtr to the
 * texture reference \p texref. The size of the area is constrained by
 * \p width in texel units, \p height in texel units, and \p pitch in byte
 * units. \p desc describes how the memory is interpreted when fetching values
 * from the texture. Any memory previously bound to \p texref is unbound.
 *
 * Since the hardware enforces an alignment requirement on texture base
 * addresses, ::cudaBindTexture2D() returns in \p *offset a byte offset that
 * must be applied to texture fetches in order to read from the desired memory.
 * This offset must be divided by the texel size and passed to kernels that
 * read from the texture so they can be applied to the ::tex2D() function.
 * If the device memory pointer was returned from ::cudaMalloc(), the offset is
 * guaranteed to be 0 and NULL may be passed as the \p offset parameter.
 *
 * \p width and \p height, which are specified in elements (or texels), cannot
 * exceed ::cudaDeviceProp::maxTexture2DLinear[0] and ::cudaDeviceProp::maxTexture2DLinear[1]
 * respectively. \p pitch, which is specified in bytes, cannot exceed
 * ::cudaDeviceProp::maxTexture2DLinear[2].
 *
 * The driver returns ::cudaErrorInvalidValue if \p pitch is not a multiple of
 * ::cudaDeviceProp::texturePitchAlignment.
 *
 * \param offset - Offset in bytes
 * \param texref - Texture reference to bind
 * \param devPtr - 2D memory area on device
 * \param desc   - Channel format
 * \param width  - Width in texel units
 * \param height - Height in texel units
 * \param pitch  - Pitch in bytes
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidDevicePointer,
 * ::cudaErrorInvalidTexture
 * \notefnerr
 *
 * \sa \ref ::cudaCreateChannelDesc(int, int, int, int, cudaChannelFormatKind) "cudaCreateChannelDesc (C API)",
 * ::cudaGetChannelDesc, ::cudaGetTextureReference,
 * \ref ::cudaBindTexture(size_t*, const struct textureReference*, const void*, const struct cudaChannelFormatDesc*, size_t) "cudaBindTexture (C API)",
 * \ref ::cudaBindTexture2D(size_t*, const struct texture< T, dim, readMode>&, const void*, const struct cudaChannelFormatDesc&, size_t, size_t, size_t) "cudaBindTexture2D (C++ API)",
 * \ref ::cudaBindTexture2D(size_t*, const struct texture<T, dim, readMode>&, const void*, size_t, size_t, size_t) "cudaBindTexture2D (C++ API, inherited channel descriptor)",
 * \ref ::cudaBindTextureToArray(const struct textureReference*, cudaArray_const_t, const struct cudaChannelFormatDesc*) "cudaBindTextureToArray (C API)",
 * \ref ::cudaUnbindTexture(const struct textureReference*) "cudaBindTextureToArray (C API)",
 * \ref ::cudaGetTextureAlignmentOffset(size_t*, const struct textureReference*) "cudaGetTextureAlignmentOffset (C API)"
 */
public static native @Cast("cudaError_t") int cudaBindTexture2D(@Cast("size_t*") SizeTPointer offset, @Const textureReference texref, @Const Pointer devPtr, @Const cudaChannelFormatDesc desc, @Cast("size_t") long width, @Cast("size_t") long height, @Cast("size_t") long pitch);

/**
 * \brief Binds an array to a texture
 *
 * Binds the CUDA array \p array to the texture reference \p texref.
 * \p desc describes how the memory is interpreted when fetching values from
 * the texture. Any CUDA array previously bound to \p texref is unbound.
 *
 * \param texref - Texture to bind
 * \param array  - Memory array on device
 * \param desc   - Channel format
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidDevicePointer,
 * ::cudaErrorInvalidTexture
 * \notefnerr
 *
 * \sa \ref ::cudaCreateChannelDesc(int, int, int, int, cudaChannelFormatKind) "cudaCreateChannelDesc (C API)",
 * ::cudaGetChannelDesc, ::cudaGetTextureReference,
 * \ref ::cudaBindTexture(size_t*, const struct textureReference*, const void*, const struct cudaChannelFormatDesc*, size_t) "cudaBindTexture (C API)",
 * \ref ::cudaBindTexture2D(size_t*, const struct textureReference*, const void*, const struct cudaChannelFormatDesc*, size_t, size_t, size_t) "cudaBindTexture2D (C API)",
 * \ref ::cudaBindTextureToArray(const struct texture< T, dim, readMode>&, cudaArray_const_t, const struct cudaChannelFormatDesc&) "cudaBindTextureToArray (C++ API)",
 * \ref ::cudaUnbindTexture(const struct textureReference*) "cudaUnbindTexture (C API)",
 * \ref ::cudaGetTextureAlignmentOffset(size_t*, const struct textureReference*) "cudaGetTextureAlignmentOffset (C API)"
 */
public static native @Cast("cudaError_t") int cudaBindTextureToArray(@Const textureReference texref, cudaArray array, @Const cudaChannelFormatDesc desc);

/**
 * \brief Binds a mipmapped array to a texture
 *
 * Binds the CUDA mipmapped array \p mipmappedArray to the texture reference \p texref.
 * \p desc describes how the memory is interpreted when fetching values from
 * the texture. Any CUDA mipmapped array previously bound to \p texref is unbound.
 *
 * \param texref         - Texture to bind
 * \param mipmappedArray - Memory mipmapped array on device
 * \param desc           - Channel format
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidDevicePointer,
 * ::cudaErrorInvalidTexture
 * \notefnerr
 *
 * \sa \ref ::cudaCreateChannelDesc(int, int, int, int, cudaChannelFormatKind) "cudaCreateChannelDesc (C API)",
 * ::cudaGetChannelDesc, ::cudaGetTextureReference,
 * \ref ::cudaBindTexture(size_t*, const struct textureReference*, const void*, const struct cudaChannelFormatDesc*, size_t) "cudaBindTexture (C API)",
 * \ref ::cudaBindTexture2D(size_t*, const struct textureReference*, const void*, const struct cudaChannelFormatDesc*, size_t, size_t, size_t) "cudaBindTexture2D (C API)",
 * \ref ::cudaBindTextureToArray(const struct texture< T, dim, readMode>&, cudaArray_const_t, const struct cudaChannelFormatDesc&) "cudaBindTextureToArray (C++ API)",
 * \ref ::cudaUnbindTexture(const struct textureReference*) "cudaUnbindTexture (C API)",
 * \ref ::cudaGetTextureAlignmentOffset(size_t*, const struct textureReference*) "cudaGetTextureAlignmentOffset (C API)"
 */
public static native @Cast("cudaError_t") int cudaBindTextureToMipmappedArray(@Const textureReference texref, cudaMipmappedArray mipmappedArray, @Const cudaChannelFormatDesc desc);

/**
 * \brief Unbinds a texture
 *
 * Unbinds the texture bound to \p texref.
 *
 * \param texref - Texture to unbind
 *
 * \return
 * ::cudaSuccess
 * \notefnerr
 *
 * \sa \ref ::cudaCreateChannelDesc(int, int, int, int, cudaChannelFormatKind) "cudaCreateChannelDesc (C API)",
 * ::cudaGetChannelDesc, ::cudaGetTextureReference,
 * \ref ::cudaBindTexture(size_t*, const struct textureReference*, const void*, const struct cudaChannelFormatDesc*, size_t) "cudaBindTexture (C API)",
 * \ref ::cudaBindTexture2D(size_t*, const struct textureReference*, const void*, const struct cudaChannelFormatDesc*, size_t, size_t, size_t) "cudaBindTexture2D (C API)",
 * \ref ::cudaBindTextureToArray(const struct textureReference*, cudaArray_const_t, const struct cudaChannelFormatDesc*) "cudaBindTextureToArray (C API)",
 * \ref ::cudaUnbindTexture(const struct texture< T, dim, readMode>&) "cudaUnbindTexture (C++ API)",
 * \ref ::cudaGetTextureAlignmentOffset(size_t*, const struct textureReference*) "cudaGetTextureAlignmentOffset (C API)"
 */
public static native @Cast("cudaError_t") int cudaUnbindTexture(@Const textureReference texref);

/**
 * \brief Get the alignment offset of a texture
 *
 * Returns in \p *offset the offset that was returned when texture reference
 * \p texref was bound.
 *
 * \param offset - Offset of texture reference in bytes
 * \param texref - Texture to get offset of
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidTexture,
 * ::cudaErrorInvalidTextureBinding
 * \notefnerr
 *
 * \sa \ref ::cudaCreateChannelDesc(int, int, int, int, cudaChannelFormatKind) "cudaCreateChannelDesc (C API)",
 * ::cudaGetChannelDesc, ::cudaGetTextureReference,
 * \ref ::cudaBindTexture(size_t*, const struct textureReference*, const void*, const struct cudaChannelFormatDesc*, size_t) "cudaBindTexture (C API)",
 * \ref ::cudaBindTexture2D(size_t*, const struct textureReference*, const void*, const struct cudaChannelFormatDesc*, size_t, size_t, size_t) "cudaBindTexture2D (C API)",
 * \ref ::cudaBindTextureToArray(const struct textureReference*, cudaArray_const_t, const struct cudaChannelFormatDesc*) "cudaBindTextureToArray (C API)",
 * \ref ::cudaUnbindTexture(const struct textureReference*) "cudaUnbindTexture (C API)",
 * \ref ::cudaGetTextureAlignmentOffset(size_t*, const struct texture< T, dim, readMode>&) "cudaGetTextureAlignmentOffset (C++ API)"
 */
public static native @Cast("cudaError_t") int cudaGetTextureAlignmentOffset(@Cast("size_t*") SizeTPointer offset, @Const textureReference texref);

/**
 * \brief Get the texture reference associated with a symbol
 *
 * Returns in \p *texref the structure associated to the texture reference
 * defined by symbol \p symbol.
 *
 * \param texref - Texture reference associated with symbol
 * \param symbol - Texture to get reference for
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidTexture
 * \notefnerr
 * \note_string_api_deprecation_50
 *
 * \sa \ref ::cudaCreateChannelDesc(int, int, int, int, cudaChannelFormatKind) "cudaCreateChannelDesc (C API)",
 * ::cudaGetChannelDesc,
 * \ref ::cudaGetTextureAlignmentOffset(size_t*, const struct textureReference*) "cudaGetTextureAlignmentOffset (C API)",
 * \ref ::cudaBindTexture(size_t*, const struct textureReference*, const void*, const struct cudaChannelFormatDesc*, size_t) "cudaBindTexture (C API)",
 * \ref ::cudaBindTexture2D(size_t*, const struct textureReference*, const void*, const struct cudaChannelFormatDesc*, size_t, size_t, size_t) "cudaBindTexture2D (C API)",
 * \ref ::cudaBindTextureToArray(const struct textureReference*, cudaArray_const_t, const struct cudaChannelFormatDesc*) "cudaBindTextureToArray (C API)",
 * \ref ::cudaUnbindTexture(const struct textureReference*) "cudaUnbindTexture (C API)"
 */
public static native @Cast("cudaError_t") int cudaGetTextureReference(@Cast("const textureReference**") PointerPointer texref, @Const Pointer symbol);
public static native @Cast("cudaError_t") int cudaGetTextureReference(@Const @ByPtrPtr textureReference texref, @Const Pointer symbol);

/** @} */ /* END CUDART_TEXTURE */

/**
 * \defgroup CUDART_SURFACE Surface Reference Management
 *
 * ___MANBRIEF___ surface reference management functions of the CUDA runtime
 * API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the low level surface reference management functions
 * of the CUDA runtime application programming interface.
 *
 * Some functions have overloaded C++ API template versions documented separately in the
 * \ref CUDART_HIGHLEVEL "C++ API Routines" module.
 *
 * @{
 */

/**
 * \brief Binds an array to a surface
 *
 * Binds the CUDA array \p array to the surface reference \p surfref.
 * \p desc describes how the memory is interpreted when fetching values from
 * the surface. Any CUDA array previously bound to \p surfref is unbound.
 *
 * \param surfref - Surface to bind
 * \param array  - Memory array on device
 * \param desc   - Channel format
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidSurface
 * \notefnerr
 *
 * \sa \ref ::cudaBindSurfaceToArray(const struct surface< T, dim>&, cudaArray_const_t, const struct cudaChannelFormatDesc&) "cudaBindSurfaceToArray (C++ API)",
 * \ref ::cudaBindSurfaceToArray(const struct surface< T, dim>&, cudaArray_const_t) "cudaBindSurfaceToArray (C++ API, inherited channel descriptor)",
 * ::cudaGetSurfaceReference
 */
public static native @Cast("cudaError_t") int cudaBindSurfaceToArray(@Const surfaceReference surfref, cudaArray array, @Const cudaChannelFormatDesc desc);

/**
 * \brief Get the surface reference associated with a symbol
 *
 * Returns in \p *surfref the structure associated to the surface reference
 * defined by symbol \p symbol.
 *
 * \param surfref - Surface reference associated with symbol
 * \param symbol - Surface to get reference for
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidSurface
 * \notefnerr
 * \note_string_api_deprecation_50
 *
 * \sa \ref ::cudaBindSurfaceToArray(const struct surfaceReference*, cudaArray_const_t, const struct cudaChannelFormatDesc*) "cudaBindSurfaceToArray (C API)"
 */
public static native @Cast("cudaError_t") int cudaGetSurfaceReference(@Cast("const surfaceReference**") PointerPointer surfref, @Const Pointer symbol);
public static native @Cast("cudaError_t") int cudaGetSurfaceReference(@Const @ByPtrPtr surfaceReference surfref, @Const Pointer symbol);

/** @} */ /* END CUDART_SURFACE */

/**
 * \defgroup CUDART_TEXTURE_OBJECT Texture Object Management
 *
 * ___MANBRIEF___ texture object management functions of the CUDA runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the low level texture object management functions
 * of the CUDA runtime application programming interface. The texture
 * object API is only supported on devices of compute capability 3.0 or higher.
 *
 * @{
 */

/**
 * \brief Creates a texture object
 *
 * Creates a texture object and returns it in \p pTexObject. \p pResDesc describes
 * the data to texture from. \p pTexDesc describes how the data should be sampled.
 * \p pResViewDesc is an optional argument that specifies an alternate format for
 * the data described by \p pResDesc, and also describes the subresource region
 * to restrict access to when texturing. \p pResViewDesc can only be specified if
 * the type of resource is a CUDA array or a CUDA mipmapped array.
 *
 * Texture objects are only supported on devices of compute capability 3.0 or higher.
 *
 * The ::cudaResourceDesc structure is defined as:
 * \code
        struct cudaResourceDesc {
	        enum cudaResourceType resType;
        	
	        union {
		        struct {
			        cudaArray_t array;
		        } array;
                struct {
                    cudaMipmappedArray_t mipmap;
                } mipmap;
		        struct {
			        void *devPtr;
			        struct cudaChannelFormatDesc desc;
			        size_t sizeInBytes;
		        } linear;
		        struct {
			        void *devPtr;
			        struct cudaChannelFormatDesc desc;
			        size_t width;
			        size_t height;
			        size_t pitchInBytes;
		        } pitch2D;
	        } res;
        };
 * \endcode
 * where:
 * - ::cudaResourceDesc::resType specifies the type of resource to texture from.
 * CUresourceType is defined as:
 * \code
        enum cudaResourceType {
            cudaResourceTypeArray          = 0x00,
            cudaResourceTypeMipmappedArray = 0x01,
            cudaResourceTypeLinear         = 0x02,
            cudaResourceTypePitch2D        = 0x03
        };
 * \endcode
 *
 * \par
 * If ::cudaResourceDesc::resType is set to ::cudaResourceTypeArray, ::cudaResourceDesc::res::array::array
 * must be set to a valid CUDA array handle.
 *
 * \par
 * If ::cudaResourceDesc::resType is set to ::cudaResourceTypeMipmappedArray, ::cudaResourceDesc::res::mipmap::mipmap
 * must be set to a valid CUDA mipmapped array handle.
 *
 * \par
 * If ::cudaResourceDesc::resType is set to ::cudaResourceTypeLinear, ::cudaResourceDesc::res::linear::devPtr
 * must be set to a valid device pointer, that is aligned to ::cudaDeviceProp::textureAlignment.
 * ::cudaResourceDesc::res::linear::desc describes the format and the number of components per array element. ::cudaResourceDesc::res::linear::sizeInBytes
 * specifies the size of the array in bytes. The total number of elements in the linear address range cannot exceed 
 * ::cudaDeviceProp::maxTexture1DLinear. The number of elements is computed as (sizeInBytes / sizeof(desc)).
 *
 * \par
 * If ::cudaResourceDesc::resType is set to ::cudaResourceTypePitch2D, ::cudaResourceDesc::res::pitch2D::devPtr
 * must be set to a valid device pointer, that is aligned to ::cudaDeviceProp::textureAlignment.
 * ::cudaResourceDesc::res::pitch2D::desc describes the format and the number of components per array element. ::cudaResourceDesc::res::pitch2D::width
 * and ::cudaResourceDesc::res::pitch2D::height specify the width and height of the array in elements, and cannot exceed
 * ::cudaDeviceProp::maxTexture2DLinear[0] and ::cudaDeviceProp::maxTexture2DLinear[1] respectively.
 * ::cudaResourceDesc::res::pitch2D::pitchInBytes specifies the pitch between two rows in bytes and has to be aligned to 
 * ::cudaDeviceProp::texturePitchAlignment. Pitch cannot exceed ::cudaDeviceProp::maxTexture2DLinear[2].
 *
 *
 * The ::cudaTextureDesc struct is defined as
 * \code
        struct cudaTextureDesc {
            enum cudaTextureAddressMode addressMode[3];
            enum cudaTextureFilterMode  filterMode;
            enum cudaTextureReadMode    readMode;
            int                         sRGB;
            int                         normalizedCoords;
            unsigned int                maxAnisotropy;
            enum cudaTextureFilterMode  mipmapFilterMode;
            float                       mipmapLevelBias;
            float                       minMipmapLevelClamp;
            float                       maxMipmapLevelClamp;
        };
 * \endcode
 * where
 * - ::cudaTextureDesc::addressMode specifies the addressing mode for each dimension of the texture data. ::cudaTextureAddressMode is defined as:
 *   \code
        enum cudaTextureAddressMode {
            cudaAddressModeWrap   = 0,
            cudaAddressModeClamp  = 1,
            cudaAddressModeMirror = 2,
            cudaAddressModeBorder = 3
        };
 *   \endcode
 *   This is ignored if ::cudaResourceDesc::resType is ::cudaResourceTypeLinear. Also, if ::cudaTextureDesc::normalizedCoords
 *   is set to zero, ::cudaAddressModeWrap and ::cudaAddressModeMirror won't be supported and will be switched to ::cudaAddressModeClamp.
 *
 * - ::cudaTextureDesc::filterMode specifies the filtering mode to be used when fetching from the texture. ::cudaTextureFilterMode is defined as:
 *   \code
        enum cudaTextureFilterMode {
            cudaFilterModePoint  = 0,
            cudaFilterModeLinear = 1
        };
 *   \endcode
 *   This is ignored if ::cudaResourceDesc::resType is ::cudaResourceTypeLinear.
 *
 * - ::cudaTextureDesc::readMode specifies whether integer data should be converted to floating point or not. ::cudaTextureReadMode is defined as:
 *   \code
        enum cudaTextureReadMode {
            cudaReadModeElementType     = 0,
            cudaReadModeNormalizedFloat = 1
        };
 *   \endcode
 *   Note that this applies only to 8-bit and 16-bit integer formats. 32-bit integer format would not be promoted, regardless of 
 *   whether or not this ::cudaTextureDesc::readMode is set ::cudaReadModeNormalizedFloat is specified.
 *
 * - ::cudaTextureDesc::sRGB specifies whether sRGB to linear conversion should be performed during texture fetch.
 *
 * - ::cudaTextureDesc::normalizedCoords specifies whether the texture coordinates will be normalized or not.
 *
 * - ::cudaTextureDesc::maxAnisotropy specifies the maximum anistropy ratio to be used when doing anisotropic filtering. This value will be
 *   clamped to the range [1,16].
 *
 * - ::cudaTextureDesc::mipmapFilterMode specifies the filter mode when the calculated mipmap level lies between two defined mipmap levels.
 *
 * - ::cudaTextureDesc::mipmapLevelBias specifies the offset to be applied to the calculated mipmap level.
 *
 * - ::cudaTextureDesc::minMipmapLevelClamp specifies the lower end of the mipmap level range to clamp access to.
 *
 * - ::cudaTextureDesc::maxMipmapLevelClamp specifies the upper end of the mipmap level range to clamp access to.
 *
 *
 * The ::cudaResourceViewDesc struct is defined as
 * \code
        struct cudaResourceViewDesc {
            enum cudaResourceViewFormat format;
            size_t                      width;
            size_t                      height;
            size_t                      depth;
            unsigned int                firstMipmapLevel;
            unsigned int                lastMipmapLevel;
            unsigned int                firstLayer;
            unsigned int                lastLayer;
        };
 * \endcode
 * where:
 * - ::cudaResourceViewDesc::format specifies how the data contained in the CUDA array or CUDA mipmapped array should
 *   be interpreted. Note that this can incur a change in size of the texture data. If the resource view format is a block
 *   compressed format, then the underlying CUDA array or CUDA mipmapped array has to have a 32-bit unsigned integer format
 *   with 2 or 4 channels, depending on the block compressed format. For ex., BC1 and BC4 require the underlying CUDA array to have
 *   a 32-bit unsigned int with 2 channels. The other BC formats require the underlying resource to have the same 32-bit unsigned int
 *   format but with 4 channels.
 *
 * - ::cudaResourceViewDesc::width specifies the new width of the texture data. If the resource view format is a block
 *   compressed format, this value has to be 4 times the original width of the resource. For non block compressed formats,
 *   this value has to be equal to that of the original resource.
 *
 * - ::cudaResourceViewDesc::height specifies the new height of the texture data. If the resource view format is a block
 *   compressed format, this value has to be 4 times the original height of the resource. For non block compressed formats,
 *   this value has to be equal to that of the original resource.
 *
 * - ::cudaResourceViewDesc::depth specifies the new depth of the texture data. This value has to be equal to that of the
 *   original resource.
 *
 * - ::cudaResourceViewDesc::firstMipmapLevel specifies the most detailed mipmap level. This will be the new mipmap level zero.
 *   For non-mipmapped resources, this value has to be zero.::cudaTextureDesc::minMipmapLevelClamp and ::cudaTextureDesc::maxMipmapLevelClamp
 *   will be relative to this value. For ex., if the firstMipmapLevel is set to 2, and a minMipmapLevelClamp of 1.2 is specified,
 *   then the actual minimum mipmap level clamp will be 3.2.
 *
 * - ::cudaResourceViewDesc::lastMipmapLevel specifies the least detailed mipmap level. For non-mipmapped resources, this value
 *   has to be zero.
 *
 * - ::cudaResourceViewDesc::firstLayer specifies the first layer index for layered textures. This will be the new layer zero.
 *   For non-layered resources, this value has to be zero.
 *
 * - ::cudaResourceViewDesc::lastLayer specifies the last layer index for layered textures. For non-layered resources, 
 *   this value has to be zero.
 *
 *
 * \param pTexObject   - Texture object to create
 * \param pResDesc     - Resource descriptor
 * \param pTexDesc     - Texture descriptor
 * \param pResViewDesc - Resource view descriptor
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 *
 * \sa ::cudaDestroyTextureObject
 */

public static native @Cast("cudaError_t") int cudaCreateTextureObject(@Cast("cudaTextureObject_t*") LongPointer pTexObject, @Const cudaResourceDesc pResDesc, @Const cudaTextureDesc pTexDesc, @Const cudaResourceViewDesc pResViewDesc);
public static native @Cast("cudaError_t") int cudaCreateTextureObject(@Cast("cudaTextureObject_t*") LongBuffer pTexObject, @Const cudaResourceDesc pResDesc, @Const cudaTextureDesc pTexDesc, @Const cudaResourceViewDesc pResViewDesc);
public static native @Cast("cudaError_t") int cudaCreateTextureObject(@Cast("cudaTextureObject_t*") long[] pTexObject, @Const cudaResourceDesc pResDesc, @Const cudaTextureDesc pTexDesc, @Const cudaResourceViewDesc pResViewDesc);

/**
 * \brief Destroys a texture object
 *
 * Destroys the texture object specified by \p texObject.
 *
 * \param texObject - Texture object to destroy
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 *
 * \sa ::cudaCreateTextureObject
 */
public static native @Cast("cudaError_t") int cudaDestroyTextureObject(@Cast("cudaTextureObject_t") long texObject);

/**
 * \brief Returns a texture object's resource descriptor
 *
 * Returns the resource descriptor for the texture object specified by \p texObject.
 *
 * \param pResDesc  - Resource descriptor
 * \param texObject - Texture object
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 *
 * \sa ::cudaCreateTextureObject
 */
public static native @Cast("cudaError_t") int cudaGetTextureObjectResourceDesc(cudaResourceDesc pResDesc, @Cast("cudaTextureObject_t") long texObject);

/**
 * \brief Returns a texture object's texture descriptor
 *
 * Returns the texture descriptor for the texture object specified by \p texObject.
 *
 * \param pTexDesc  - Texture descriptor
 * \param texObject - Texture object
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 *
 * \sa ::cudaCreateTextureObject
 */
public static native @Cast("cudaError_t") int cudaGetTextureObjectTextureDesc(cudaTextureDesc pTexDesc, @Cast("cudaTextureObject_t") long texObject);

/**
 * \brief Returns a texture object's resource view descriptor
 *
 * Returns the resource view descriptor for the texture object specified by \p texObject.
 * If no resource view was specified, ::cudaErrorInvalidValue is returned.
 *
 * \param pResViewDesc - Resource view descriptor
 * \param texObject    - Texture object
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 *
 * \sa ::cudaCreateTextureObject
 */
public static native @Cast("cudaError_t") int cudaGetTextureObjectResourceViewDesc(cudaResourceViewDesc pResViewDesc, @Cast("cudaTextureObject_t") long texObject);

/** @} */ /* END CUDART_TEXTURE_OBJECT */

/**
 * \defgroup CUDART_SURFACE_OBJECT Surface Object Management
 *
 * ___MANBRIEF___ surface object management functions of the CUDA runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the low level texture object management functions
 * of the CUDA runtime application programming interface. The surface object 
 * API is only supported on devices of compute capability 3.0 or higher.
 *
 * @{
 */

/**
 * \brief Creates a surface object
 *
 * Creates a surface object and returns it in \p pSurfObject. \p pResDesc describes
 * the data to perform surface load/stores on. ::cudaResourceDesc::resType must be 
 * ::cudaResourceTypeArray and  ::cudaResourceDesc::res::array::array
 * must be set to a valid CUDA array handle.
 *
 * Surface objects are only supported on devices of compute capability 3.0 or higher.
 *
 * \param pSurfObject - Surface object to create
 * \param pResDesc    - Resource descriptor
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 *
 * \sa ::cudaDestroySurfaceObject
 */

public static native @Cast("cudaError_t") int cudaCreateSurfaceObject(@Cast("cudaSurfaceObject_t*") LongPointer pSurfObject, @Const cudaResourceDesc pResDesc);
public static native @Cast("cudaError_t") int cudaCreateSurfaceObject(@Cast("cudaSurfaceObject_t*") LongBuffer pSurfObject, @Const cudaResourceDesc pResDesc);
public static native @Cast("cudaError_t") int cudaCreateSurfaceObject(@Cast("cudaSurfaceObject_t*") long[] pSurfObject, @Const cudaResourceDesc pResDesc);

/**
 * \brief Destroys a surface object
 *
 * Destroys the surface object specified by \p surfObject.
 *
 * \param surfObject - Surface object to destroy
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 *
 * \sa ::cudaCreateSurfaceObject
 */
public static native @Cast("cudaError_t") int cudaDestroySurfaceObject(@Cast("cudaSurfaceObject_t") long surfObject);

/**
 * \brief Returns a surface object's resource descriptor
 * Returns the resource descriptor for the surface object specified by \p surfObject.
 *
 * \param pResDesc   - Resource descriptor
 * \param surfObject - Surface object
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 *
 * \sa ::cudaCreateSurfaceObject
 */
public static native @Cast("cudaError_t") int cudaGetSurfaceObjectResourceDesc(cudaResourceDesc pResDesc, @Cast("cudaSurfaceObject_t") long surfObject);

/** @} */ /* END CUDART_SURFACE_OBJECT */

/**
 * \defgroup CUDART__VERSION Version Management
 *
 * @{
 */

/**
 * \brief Returns the CUDA driver version
 *
 * Returns in \p *driverVersion the version number of the installed CUDA
 * driver. If no driver is installed, then 0 is returned as the driver
 * version (via \p driverVersion). This function automatically returns
 * ::cudaErrorInvalidValue if the \p driverVersion argument is NULL.
 *
 * \param driverVersion - Returns the CUDA driver version.
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \notefnerr
 *
 * \sa ::cudaRuntimeGetVersion
 */
public static native @Cast("cudaError_t") int cudaDriverGetVersion(IntPointer driverVersion);
public static native @Cast("cudaError_t") int cudaDriverGetVersion(IntBuffer driverVersion);
public static native @Cast("cudaError_t") int cudaDriverGetVersion(int[] driverVersion);

/**
 * \brief Returns the CUDA Runtime version
 *
 * Returns in \p *runtimeVersion the version number of the installed CUDA
 * Runtime. This function automatically returns ::cudaErrorInvalidValue if
 * the \p runtimeVersion argument is NULL.
 *
 * \param runtimeVersion - Returns the CUDA Runtime version.
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 *
 * \sa ::cudaDriverGetVersion
 */
public static native @Cast("cudaError_t") int cudaRuntimeGetVersion(IntPointer runtimeVersion);
public static native @Cast("cudaError_t") int cudaRuntimeGetVersion(IntBuffer runtimeVersion);
public static native @Cast("cudaError_t") int cudaRuntimeGetVersion(int[] runtimeVersion);

/** @} */ /* END CUDART__VERSION */

/** \cond impl_private */
public static native @Cast("cudaError_t") int cudaGetExportTable(@Cast("const void**") PointerPointer ppExportTable, @Const cudaUUID_t pExportTableId);
public static native @Cast("cudaError_t") int cudaGetExportTable(@Cast("const void**") @ByPtrPtr Pointer ppExportTable, @Const cudaUUID_t pExportTableId);
/** \endcond impl_private */

/**
 * \defgroup CUDART_HIGHLEVEL C++ API Routines
 *
 * ___MANBRIEF___ C++ high level API functions of the CUDA runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the C++ high level API functions of the CUDA runtime
 * application programming interface. To use these functions, your
 * application needs to be compiled with the \p nvcc compiler.
 *
 * \brief C++-style interface built on top of CUDA runtime API
 */

/**
 * \defgroup CUDART_DRIVER Interactions with the CUDA Driver API
 *
 * ___MANBRIEF___ interactions between CUDA Driver API and CUDA Runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the interactions between the CUDA Driver API and the CUDA Runtime API
 *
 * @{
 *
 * \section CUDART_CUDA_primary Primary Contexts
 *
 * There exists a one to one relationship between CUDA devices in the CUDA Runtime
 * API and ::CUcontext s in the CUDA Driver API within a process.  The specific
 * context which the CUDA Runtime API uses for a device is called the device's
 * primary context.  From the perspective of the CUDA Runtime API, a device and 
 * its primary context are synonymous.
 *
 * \section CUDART_CUDA_init Initialization and Tear-Down
 *
 * CUDA Runtime API calls operate on the CUDA Driver API ::CUcontext which is current to
 * to the calling host thread.  
 *
 * The function ::cudaSetDevice() makes the primary context for the
 * specified device current to the calling thread by calling ::cuCtxSetCurrent().
 *
 * The CUDA Runtime API will automatically initialize the primary context for
 * a device at the first CUDA Runtime API call which requires an active context.
 * If no ::CUcontext is current to the calling thread when a CUDA Runtime API call 
 * which requires an active context is made, then the primary context for a device 
 * will be selected, made current to the calling thread, and initialized.
 *
 * The context which the CUDA Runtime API initializes will be initialized using 
 * the parameters specified by the CUDA Runtime API functions
 * ::cudaSetDeviceFlags(), 
 * ::cudaD3D9SetDirect3DDevice(), 
 * ::cudaD3D10SetDirect3DDevice(), 
 * ::cudaD3D11SetDirect3DDevice(), 
 * ::cudaGLSetGLDevice(), and
 * ::cudaVDPAUSetVDPAUDevice().
 * Note that these functions will fail with ::cudaErrorSetOnActiveProcess if they are 
 * called when the primary context for the specified device has already been initialized.
 * (or if the current device has already been initialized, in the case of 
 * ::cudaSetDeviceFlags()). 
 *
 * Primary contexts will remain active until they are explicitly deinitialized 
 * using ::cudaDeviceReset().  The function ::cudaDeviceReset() will deinitialize the 
 * primary context for the calling thread's current device immediately.  The context 
 * will remain current to all of the threads that it was current to.  The next CUDA 
 * Runtime API call on any thread which requires an active context will trigger the 
 * reinitialization of that device's primary context.
 *
 * Note that there is no reference counting of the primary context's lifetime.  It is
 * recommended that the primary context not be deinitialized except just before exit
 * or to recover from an unspecified launch failure.
 * 
 * \section CUDART_CUDA_context Context Interoperability
 *
 * Note that the use of multiple ::CUcontext s per device within a single process 
 * will substantially degrade performance and is strongly discouraged.  Instead,
 * it is highly recommended that the implicit one-to-one device-to-context mapping
 * for the process provided by the CUDA Runtime API be used.
 *
 * If a non-primary ::CUcontext created by the CUDA Driver API is current to a
 * thread then the CUDA Runtime API calls to that thread will operate on that 
 * ::CUcontext, with some exceptions listed below.  Interoperability between data
 * types is discussed in the following sections.
 *
 * The function ::cudaPointerGetAttributes() will return the error 
 * ::cudaErrorIncompatibleDriverContext if the pointer being queried was allocated by a 
 * non-primary context.  The function ::cudaDeviceEnablePeerAccess() and the rest of 
 * the peer access API may not be called when a non-primary ::CUcontext is current.  
 * To use the pointer query and peer access APIs with a context created using the 
 * CUDA Driver API, it is necessary that the CUDA Driver API be used to access
 * these features.
 *
 * All CUDA Runtime API state (e.g, global variables' addresses and values) travels
 * with its underlying ::CUcontext.  In particular, if a ::CUcontext is moved from one 
 * thread to another then all CUDA Runtime API state will move to that thread as well.
 *
 * Please note that attaching to legacy contexts (those with a version of 3010 as returned
 * by ::cuCtxGetApiVersion()) is not possible. The CUDA Runtime will return
 * ::cudaErrorIncompatibleDriverContext in such cases.
 *
 * \section CUDART_CUDA_stream Interactions between CUstream and cudaStream_t
 *
 * The types ::CUstream and ::cudaStream_t are identical and may be used interchangeably.
 *
 * \section CUDART_CUDA_event Interactions between CUevent and cudaEvent_t
 *
 * The types ::CUevent and ::cudaEvent_t are identical and may be used interchangeably.
 *
 * \section CUDART_CUDA_array Interactions between CUarray and cudaArray_t 
 *
 * The types ::CUarray and struct ::cudaArray * represent the same data type and may be used
 * interchangeably by casting the two types between each other.
 *
 * In order to use a ::CUarray in a CUDA Runtime API function which takes a struct ::cudaArray *,
 * it is necessary to explicitly cast the ::CUarray to a struct ::cudaArray *.
 *
 * In order to use a struct ::cudaArray * in a CUDA Driver API function which takes a ::CUarray,
 * it is necessary to explicitly cast the struct ::cudaArray * to a ::CUarray .
 *
 * \section CUDART_CUDA_graphicsResource Interactions between CUgraphicsResource and cudaGraphicsResource_t
 *
 * The types ::CUgraphicsResource and ::cudaGraphicsResource_t represent the same data type and may be used
 * interchangeably by casting the two types between each other.
 *
 * In order to use a ::CUgraphicsResource in a CUDA Runtime API function which takes a 
 * ::cudaGraphicsResource_t, it is necessary to explicitly cast the ::CUgraphicsResource 
 * to a ::cudaGraphicsResource_t.
 *
 * In order to use a ::cudaGraphicsResource_t in a CUDA Driver API function which takes a
 * ::CUgraphicsResource, it is necessary to explicitly cast the ::cudaGraphicsResource_t 
 * to a ::CUgraphicsResource.
 *
 * @}
 */

// #if defined(__cplusplus)

// #endif /* __cplusplus */

// #undef __dv

// #endif /* !__CUDA_RUNTIME_API_H__ */


// Parsed from <driver_functions.h>

/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
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
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
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

// #if !defined(__DRIVER_FUNCTIONS_H__)
// #define __DRIVER_FUNCTIONS_H__

// #include "builtin_types.h"
// #include "host_defines.h"
// #include "driver_types.h"

/**
 * \addtogroup CUDART_MEMORY
 *
 * @{
 */

/**
 * \brief Returns a cudaPitchedPtr based on input parameters
 *
 * Returns a ::cudaPitchedPtr based on the specified input parameters \p d,
 * \p p, \p xsz, and \p ysz.
 *
 * \param d   - Pointer to allocated memory
 * \param p   - Pitch of allocated memory in bytes
 * \param xsz - Logical width of allocation in elements
 * \param ysz - Logical height of allocation in elements
 *
 * \return
 * ::cudaPitchedPtr specified by \p d, \p p, \p xsz, and \p ysz
 *
 * \sa make_cudaExtent, make_cudaPos
 */
public static native @ByVal cudaPitchedPtr make_cudaPitchedPtr(Pointer d, @Cast("size_t") long p, @Cast("size_t") long xsz, @Cast("size_t") long ysz);

/**
 * \brief Returns a cudaPos based on input parameters
 *
 * Returns a ::cudaPos based on the specified input parameters \p x,
 * \p y, and \p z.
 *
 * \param x - X position
 * \param y - Y position
 * \param z - Z position
 *
 * \return
 * ::cudaPos specified by \p x, \p y, and \p z
 *
 * \sa make_cudaExtent, make_cudaPitchedPtr
 */
public static native @ByVal cudaPos make_cudaPos(@Cast("size_t") long x, @Cast("size_t") long y, @Cast("size_t") long z);

/**
 * \brief Returns a cudaExtent based on input parameters
 *
 * Returns a ::cudaExtent based on the specified input parameters \p w,
 * \p h, and \p d.
 *
 * \param w - Width in bytes
 * \param h - Height in elements
 * \param d - Depth in elements
 *
 * \return
 * ::cudaExtent specified by \p w, \p h, and \p d
 *
 * \sa make_cudaPitchedPtr, make_cudaPos
 */
public static native @ByVal cudaExtent make_cudaExtent(@Cast("size_t") long w, @Cast("size_t") long h, @Cast("size_t") long d);

/** @} */ /* END CUDART_MEMORY */

// #endif /* !__DRIVER_FUNCTIONS_H__ */


// Parsed from <vector_functions.h>

/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
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
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
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

// #if !defined(__VECTOR_FUNCTIONS_H__)
// #define __VECTOR_FUNCTIONS_H__

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

// #include "builtin_types.h"
// #include "host_defines.h"
// #include "vector_types.h"

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

public static native @ByVal char1 make_char1(byte x);

public static native @ByVal uchar1 make_uchar1(@Cast("unsigned char") byte x);

public static native @ByVal char2 make_char2(byte x, byte y);

public static native @ByVal uchar2 make_uchar2(@Cast("unsigned char") byte x, @Cast("unsigned char") byte y);

public static native @ByVal char3 make_char3(byte x, byte y, byte z);

public static native @ByVal uchar3 make_uchar3(@Cast("unsigned char") byte x, @Cast("unsigned char") byte y, @Cast("unsigned char") byte z);

public static native @ByVal char4 make_char4(byte x, byte y, byte z, byte w);

public static native @ByVal uchar4 make_uchar4(@Cast("unsigned char") byte x, @Cast("unsigned char") byte y, @Cast("unsigned char") byte z, @Cast("unsigned char") byte w);

public static native @ByVal short1 make_short1(short x);

public static native @ByVal ushort1 make_ushort1(@Cast("unsigned short") short x);

public static native @ByVal short2 make_short2(short x, short y);

public static native @ByVal ushort2 make_ushort2(@Cast("unsigned short") short x, @Cast("unsigned short") short y);

public static native @ByVal short3 make_short3(short x,short y, short z);

public static native @ByVal ushort3 make_ushort3(@Cast("unsigned short") short x, @Cast("unsigned short") short y, @Cast("unsigned short") short z);

public static native @ByVal short4 make_short4(short x, short y, short z, short w);

public static native @ByVal ushort4 make_ushort4(@Cast("unsigned short") short x, @Cast("unsigned short") short y, @Cast("unsigned short") short z, @Cast("unsigned short") short w);

public static native @ByVal int1 make_int1(int x);

public static native @ByVal uint1 make_uint1(@Cast("unsigned int") int x);

public static native @ByVal int2 make_int2(int x, int y);

public static native @ByVal uint2 make_uint2(@Cast("unsigned int") int x, @Cast("unsigned int") int y);

public static native @ByVal int3 make_int3(int x, int y, int z);

public static native @ByVal uint3 make_uint3(@Cast("unsigned int") int x, @Cast("unsigned int") int y, @Cast("unsigned int") int z);

public static native @ByVal int4 make_int4(int x, int y, int z, int w);

public static native @ByVal uint4 make_uint4(@Cast("unsigned int") int x, @Cast("unsigned int") int y, @Cast("unsigned int") int z, @Cast("unsigned int") int w);

public static native @ByVal long1 make_long1(long x);

public static native @ByVal ulong1 make_ulong1(@Cast("unsigned long int") long x);

public static native @ByVal long2 make_long2(long x, long y);

public static native @ByVal ulong2 make_ulong2(@Cast("unsigned long int") long x, @Cast("unsigned long int") long y);

public static native @ByVal long3 make_long3(long x, long y, long z);

public static native @ByVal ulong3 make_ulong3(@Cast("unsigned long int") long x, @Cast("unsigned long int") long y, @Cast("unsigned long int") long z);

public static native @ByVal long4 make_long4(long x, long y, long z, long w);

public static native @ByVal ulong4 make_ulong4(@Cast("unsigned long int") long x, @Cast("unsigned long int") long y, @Cast("unsigned long int") long z, @Cast("unsigned long int") long w);

public static native @ByVal float1 make_float1(float x);

public static native @ByVal float2 make_float2(float x, float y);

public static native @ByVal float3 make_float3(float x, float y, float z);

public static native @ByVal float4 make_float4(float x, float y, float z, float w);

public static native @ByVal longlong1 make_longlong1(long x);

public static native @ByVal ulonglong1 make_ulonglong1(@Cast("unsigned long long int") long x);

public static native @ByVal longlong2 make_longlong2(long x, long y);

public static native @ByVal ulonglong2 make_ulonglong2(@Cast("unsigned long long int") long x, @Cast("unsigned long long int") long y);

public static native @ByVal longlong3 make_longlong3(long x, long y, long z);

public static native @ByVal ulonglong3 make_ulonglong3(@Cast("unsigned long long int") long x, @Cast("unsigned long long int") long y, @Cast("unsigned long long int") long z);

public static native @ByVal longlong4 make_longlong4(long x, long y, long z, long w);

public static native @ByVal ulonglong4 make_ulonglong4(@Cast("unsigned long long int") long x, @Cast("unsigned long long int") long y, @Cast("unsigned long long int") long z, @Cast("unsigned long long int") long w);

public static native @ByVal double1 make_double1(double x);

public static native @ByVal double2 make_double2(double x, double y);

public static native @ByVal double3 make_double3(double x, double y, double z);

public static native @ByVal double4 make_double4(double x, double y, double z, double w);

// #endif /* !__VECTOR_FUNCTIONS_H__ */


// Parsed from <cuComplex.h>

/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
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
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
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

// #if !defined(CU_COMPLEX_H_)
// #define CU_COMPLEX_H_

// #if defined(__cplusplus)
// #endif /* __cplusplus */

// #include <math.h>       /* import fabsf, sqrt */
// #include "vector_types.h"

public static native float cuCrealf(@ByVal @Cast("cuFloatComplex*") float2 x);

public static native float cuCimagf(@ByVal @Cast("cuFloatComplex*") float2 x);

public static native @ByVal @Cast("cuFloatComplex*") float2 make_cuFloatComplex(float r, float i);

public static native @ByVal @Cast("cuFloatComplex*") float2 cuConjf(@ByVal @Cast("cuFloatComplex*") float2 x);
public static native @ByVal @Cast("cuFloatComplex*") float2 cuCaddf(@ByVal @Cast("cuFloatComplex*") float2 x,
                                                              @ByVal @Cast("cuFloatComplex*") float2 y);

public static native @ByVal @Cast("cuFloatComplex*") float2 cuCsubf(@ByVal @Cast("cuFloatComplex*") float2 x,
                                                              @ByVal @Cast("cuFloatComplex*") float2 y);

/* This implementation could suffer from intermediate overflow even though
 * the final result would be in range. However, various implementations do
 * not guard against this (presumably to avoid losing performance), so we 
 * don't do it either to stay competitive.
 */
public static native @ByVal @Cast("cuFloatComplex*") float2 cuCmulf(@ByVal @Cast("cuFloatComplex*") float2 x,
                                                              @ByVal @Cast("cuFloatComplex*") float2 y);

/* This implementation guards against intermediate underflow and overflow
 * by scaling. Such guarded implementations are usually the default for
 * complex library implementations, with some also offering an unguarded,
 * faster version.
 */
public static native @ByVal @Cast("cuFloatComplex*") float2 cuCdivf(@ByVal @Cast("cuFloatComplex*") float2 x,
                                                              @ByVal @Cast("cuFloatComplex*") float2 y);

/* 
 * We would like to call hypotf(), but it's not available on all platforms.
 * This discrete implementation guards against intermediate underflow and 
 * overflow by scaling. Otherwise we would lose half the exponent range. 
 * There are various ways of doing guarded computation. For now chose the 
 * simplest and fastest solution, however this may suffer from inaccuracies 
 * if sqrt and division are not IEEE compliant. 
 */
public static native float cuCabsf(@ByVal @Cast("cuFloatComplex*") float2 x);

/* Double precision */

public static native double cuCreal(@ByVal @Cast("cuDoubleComplex*") double2 x);

public static native double cuCimag(@ByVal @Cast("cuDoubleComplex*") double2 x);

public static native @ByVal @Cast("cuDoubleComplex*") double2 make_cuDoubleComplex(double r, double i);

public static native @ByVal @Cast("cuDoubleComplex*") double2 cuConj(@ByVal @Cast("cuDoubleComplex*") double2 x);

public static native @ByVal @Cast("cuDoubleComplex*") double2 cuCadd(@ByVal @Cast("cuDoubleComplex*") double2 x,
                                                             @ByVal @Cast("cuDoubleComplex*") double2 y);

public static native @ByVal @Cast("cuDoubleComplex*") double2 cuCsub(@ByVal @Cast("cuDoubleComplex*") double2 x,
                                                             @ByVal @Cast("cuDoubleComplex*") double2 y);

/* This implementation could suffer from intermediate overflow even though
 * the final result would be in range. However, various implementations do
 * not guard against this (presumably to avoid losing performance), so we 
 * don't do it either to stay competitive.
 */
public static native @ByVal @Cast("cuDoubleComplex*") double2 cuCmul(@ByVal @Cast("cuDoubleComplex*") double2 x,
                                                             @ByVal @Cast("cuDoubleComplex*") double2 y);

/* This implementation guards against intermediate underflow and overflow
 * by scaling. Such guarded implementations are usually the default for
 * complex library implementations, with some also offering an unguarded,
 * faster version.
 */
public static native @ByVal @Cast("cuDoubleComplex*") double2 cuCdiv(@ByVal @Cast("cuDoubleComplex*") double2 x,
                                                             @ByVal @Cast("cuDoubleComplex*") double2 y);

/* This implementation guards against intermediate underflow and overflow
 * by scaling. Otherwise we would lose half the exponent range. There are
 * various ways of doing guarded computation. For now chose the simplest
 * and fastest solution, however this may suffer from inaccuracies if sqrt
 * and division are not IEEE compliant.
 */
public static native double cuCabs(@ByVal @Cast("cuDoubleComplex*") double2 x);

// #if defined(__cplusplus)
// #endif /* __cplusplus */

/* aliases */
public static native @ByVal @Cast("cuComplex*") float2 make_cuComplex(float x, 
                                                                float y);

/* float-to-double promotion */
public static native @ByVal @Cast("cuDoubleComplex*") double2 cuComplexFloatToDouble(@ByVal @Cast("cuFloatComplex*") float2 c);

public static native @ByVal @Cast("cuFloatComplex*") float2 cuComplexDoubleToFloat(@ByVal @Cast("cuDoubleComplex*") double2 c);


public static native @ByVal @Cast("cuComplex*") float2 cuCfmaf( @ByVal @Cast("cuComplex*") float2 x, @ByVal @Cast("cuComplex*") float2 y, @ByVal @Cast("cuComplex*") float2 d);

public static native @ByVal @Cast("cuDoubleComplex*") double2 cuCfma( @ByVal @Cast("cuDoubleComplex*") double2 x, @ByVal @Cast("cuDoubleComplex*") double2 y, @ByVal @Cast("cuDoubleComplex*") double2 d);

// #endif /* !defined(CU_COMPLEX_H_) */


}
