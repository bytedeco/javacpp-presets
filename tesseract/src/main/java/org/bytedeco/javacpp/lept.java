// Targeted by JavaCPP version 1.1-SNAPSHOT

package org.bytedeco.javacpp;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

public class lept extends org.bytedeco.javacpp.presets.lept {
    static { Loader.load(); }

// Parsed from leptonica/alltypes.h

/*====================================================================*
 -  Copyright (C) 2001 Leptonica.  All rights reserved.
 -
 -  Redistribution and use in source and binary forms, with or without
 -  modification, are permitted provided that the following conditions
 -  are met:
 -  1. Redistributions of source code must retain the above copyright
 -     notice, this list of conditions and the following disclaimer.
 -  2. Redistributions in binary form must reproduce the above
 -     copyright notice, this list of conditions and the following
 -     disclaimer in the documentation and/or other materials
 -     provided with the distribution.
 -
 -  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 -  ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 -  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 -  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL ANY
 -  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 -  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 -  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 -  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 -  OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 -  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 -  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *====================================================================*/

// #ifndef  LEPTONICA_ALLTYPES_H
// #define  LEPTONICA_ALLTYPES_H

    /* Standard */
// #include <stdio.h>
// #include <stdlib.h>
// #include <stdarg.h>

    /* General and configuration defs */
// #include "environ.h"

    /* Generic and non-image-specific containers */
// #include "array.h"
// #include "bbuffer.h"
// #include "heap.h"
// #include "list.h"
// #include "ptra.h"
// #include "queue.h"
// #include "stack.h"

    /* Imaging */
// #include "arrayaccess.h"
// #include "bmf.h"
// #include "ccbord.h"
// #include "dewarp.h"
// #include "gplot.h"
// #include "imageio.h"
// #include "jbclass.h"
// #include "morph.h"
// #include "pix.h"
// #include "recog.h"
// #include "regutils.h"
// #include "stringcode.h"
// #include "sudoku.h"
// #include "watershed.h"


// #endif /* LEPTONICA_ALLTYPES_H */


// Parsed from leptonica/environ.h

/*====================================================================*
 -  Copyright (C) 2001 Leptonica.  All rights reserved.
 -
 -  Redistribution and use in source and binary forms, with or without
 -  modification, are permitted provided that the following conditions
 -  are met:
 -  1. Redistributions of source code must retain the above copyright
 -     notice, this list of conditions and the following disclaimer.
 -  2. Redistributions in binary form must reproduce the above
 -     copyright notice, this list of conditions and the following
 -     disclaimer in the documentation and/or other materials
 -     provided with the distribution.
 -
 -  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 -  ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 -  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 -  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL ANY
 -  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 -  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 -  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 -  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 -  OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 -  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 -  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *====================================================================*/

// #ifndef  LEPTONICA_ENVIRON_H
// #define  LEPTONICA_ENVIRON_H

/*------------------------------------------------------------------------*
 *  Defines and includes differ for Unix and Windows.  Also for Windows,  *
 *  differentiate between conditionals based on platform and compiler.    *
 *      For platforms:                                                    *
 *          _WIN32       =>     Windows, 32- or 64-bit                    *
 *          _WIN64       =>     Windows, 64-bit only                      *
 *          __CYGWIN__   =>     Cygwin                                    *
 *      For compilers:                                                    *
 *          __GNUC__     =>     gcc                                       *
 *          _MSC_VER     =>     msvc                                      *
 *------------------------------------------------------------------------*/

/* MS VC++ does not provide stdint.h, so define the missing types here */


// #ifndef _MSC_VER
// #include <stdint.h>

// #else
/* Note that _WIN32 is defined for both 32 and 64 bit applications,
   whereas _WIN64 is defined only for the latter */

// #ifdef _WIN64
// #else
// #endif

/* VC++6 doesn't seem to have powf, expf. */
// #if (_MSC_VER < 1400)
// #define powf(x, y) (float)pow((double)(x), (double)(y))
// #define expf(x) (float)exp((double)(x))
// #endif

// #endif /* _MSC_VER */

/* Windows specifics */
// #ifdef _WIN32
  /* DLL EXPORTS and IMPORTS */
//   #if defined(LIBLEPT_EXPORTS)
//     #define LEPT_DLL __declspec(dllexport)
//   #elif defined(LIBLEPT_IMPORTS)
//     #define LEPT_DLL __declspec(dllimport)
//   #else
//     #define LEPT_DLL
//   #endif
// #else  /* non-Windows specifics */
//   #include <stdint.h>
//   #define LEPT_DLL
// #endif  /* _WIN32 */


/*--------------------------------------------------------------------*
 * !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*
 *                          USER CONFIGURABLE                         *
 * !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*
 *               Environment variables with I/O libraries             *
 *               Manual Configuration Only: NOT AUTO_CONF             *
 *--------------------------------------------------------------------*/
/*
 *  Leptonica provides interfaces to link to several external image
 *  I/O libraries, plus zlib.  Setting any of these to 0 here causes
 *  non-functioning stubs to be linked.
 */
// #if !defined(HAVE_CONFIG_H) && !defined(ANDROID_BUILD)
public static final int HAVE_LIBJPEG =     1;
public static final int HAVE_LIBTIFF =     1;
public static final int HAVE_LIBPNG =      1;
public static final int HAVE_LIBZ =        1;
public static final int HAVE_LIBGIF =      0;
public static final int HAVE_LIBUNGIF =    0;
public static final int HAVE_LIBWEBP =     0;
public static final int HAVE_LIBJP2K =     0;

    /* Leptonica supports both OpenJPEG 2.0 and 2.1.  If you have a
     * version of openjpeg (HAVE_LIBJP2K) that is not 2.1, set the
     * path to the openjpeg.h header in angle brackets here. */
// #define  LIBJP2K_HEADER   <openjpeg-2.1/openjpeg.h>
// #endif  /* ! HAVE_CONFIG_H etc. */

/*
 * On linux systems, you can do I/O between Pix and memory.  Specifically,
 * you can compress (write compressed data to memory from a Pix) and
 * uncompress (read from compressed data in memory to a Pix).
 * For jpeg, png, jp2k, gif, pnm and bmp, these use the non-posix GNU
 * functions fmemopen() and open_memstream().  These functions are not
 * available on other systems.
 * To use these functions in linux, you must define HAVE_FMEMOPEN to 1.
 * To use them on MacOS, which does not support these functions, set it to 0.
 */
// #if !defined(HAVE_CONFIG_H) && !defined(ANDROID_BUILD) && !defined(_MSC_VER)
public static final int HAVE_FMEMOPEN =    1;
// #endif  /* ! HAVE_CONFIG_H etc. */


/*--------------------------------------------------------------------*
 * !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*
 *                          USER CONFIGURABLE                         *
 * !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*
 *     Environ variables for image I/O without external libraries     *
 *--------------------------------------------------------------------*/
/*
 *  Leptonica supplies I/O support without using external libraries for:
 *     * image read/write for bmp, pnm
 *     * header read for jp2k
 *     * image wrapping write for pdf and ps.
 *  Setting any of these to 0 causes non-functioning stubs to be linked.
 */
public static final int USE_BMPIO =        1;
public static final int USE_PNMIO =        1;
public static final int USE_JP2KHEADER =   1;
public static final int USE_PDFIO =        1;
public static final int USE_PSIO =         1;


/*--------------------------------------------------------------------*
 * !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*
 *                          USER CONFIGURABLE                         *
 * !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*
 *     Optional subdirectory translation for read/write to /tmp       *
 *--------------------------------------------------------------------*/
/*
 * It is desirable on Windows to have all temp files written to the same
 * subdirectory of the Windows <Temp> directory, because files under <Temp>
 * persist after reboot, and the regression tests write a lot of files.
 * Consequently, all temp files on Windows are written to <Temp>/leptonica/
 * or subdirectories of it, with the translation:
 *        /tmp/xxx  -->   <Temp>/leptonica/xxx
 *
 * This is not the case for Unix, but we provide an option for reading
 * and writing on Unix with this translation:
 *        /tmp/xxx  -->   /tmp/leptonica/xxx
 * By default, leptonica is distributed for Unix without this translation
 * (except on Cygwin, which runs on Windows).
 */
// #if defined (__CYGWIN__)
  public static final int ADD_LEPTONICA_SUBDIR =    1;
// #else
// #endif


/*--------------------------------------------------------------------*
 *                          Built-in types                            *
 *--------------------------------------------------------------------*/
// #ifdef COMPILER_MSVC
// #else
// #endif  /* COMPILER_MSVC */


/*------------------------------------------------------------------------*
 *                            Standard macros                             *
 *------------------------------------------------------------------------*/
// #ifndef L_MIN
// #define L_MIN(x,y)   (((x) < (y)) ? (x) : (y))
// #endif

// #ifndef L_MAX
// #define L_MAX(x,y)   (((x) > (y)) ? (x) : (y))
// #endif

// #ifndef L_ABS
// #define L_ABS(x)     (((x) < 0) ? (-1 * (x)) : (x))
// #endif

// #ifndef L_SIGN
// #define L_SIGN(x)    (((x) < 0) ? -1 : 1)
// #endif

// #ifndef UNDEF
public static final int UNDEF =        -1;
// #endif

// #ifndef NULL
public static final int NULL =          0;
// #endif

// #ifndef TRUE
public static final int TRUE =          1;
// #endif

// #ifndef FALSE
public static final int FALSE =         0;
// #endif


/*--------------------------------------------------------------------*
 *            Environment variables for endian dependence             *
 *--------------------------------------------------------------------*/
/*
 *  To control conditional compilation, one of two variables
 *
 *       L_LITTLE_ENDIAN  (e.g., for Intel X86)
 *       L_BIG_ENDIAN     (e.g., for Sun SPARC, Mac Power PC)
 *
 *  is defined when the GCC compiler is invoked.
 *  All code should compile properly for both hardware architectures.
 */


/*------------------------------------------------------------------------*
 *                    Simple search state variables                       *
 *------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_NOT_FOUND = 0,
    L_FOUND = 1;


/*------------------------------------------------------------------------*
 *                     Path separator conversion                          *
 *------------------------------------------------------------------------*/
/** enum  */
public static final int
    UNIX_PATH_SEPCHAR = 0,
    WIN_PATH_SEPCHAR = 1;


/*------------------------------------------------------------------------*
 *                          Timing structs                                *
 *------------------------------------------------------------------------*/


/*------------------------------------------------------------------------*
 *                      Standard memory allocation                        *
 *                                                                        *
 *  These specify the memory management functions that are used           *
 *  on all heap data except for Pix.  Memory management for Pix           *
 *  also defaults to malloc and free.  See pix1.c for details.            *
 *------------------------------------------------------------------------*/
// #define MALLOC(blocksize)           malloc(blocksize)
// #define CALLOC(numelem, elemsize)   calloc(numelem, elemsize)
// #define REALLOC(ptr, blocksize)     realloc(ptr, blocksize)
// #define FREE(ptr)                   free(ptr)


/*------------------------------------------------------------------------*
 *         Control printing of error, warning, and info messages          *
 *                                                                        *
 *  To omit all messages to stderr, simply define NO_CONSOLE_IO on the    *
 *  command line.  For finer grained control, we have a mechanism         *
 *  based on the message severity level.  The following assumes that      *
 *  NO_CONSOLE_IO is not defined.                                         *
 *                                                                        *
 *  Messages are printed if the message severity is greater than or equal *
 *  to the current severity threshold.  The current severity threshold    *
 *  is the greater of the compile-time severity, which is the minimum     *
 *  severity that can be reported, and the run-time severity, which is    *
 *  the severity threshold at the moment.                                 *
 *                                                                        *
 *  The compile-time threshold determines which messages are compiled     *
 *  into the library for potential printing.  Messages below the          *
 *  compile-time threshold are omitted and can never be printed.  The     *
 *  default compile-time threshold is L_SEVERITY_INFO, but this may be    *
 *  overridden by defining MINIMUM_SEVERITY to the desired enumeration    *
 *  identifier on the compiler command line.  Defining NO_CONSOLE_IO on   *
 *  the command line is the same as setting MINIMUM_SEVERITY to           *
 *  L_SEVERITY_NONE.                                                      *
 *                                                                        *
 *  The run-time threshold determines which messages are printed during   *
 *  library execution.  It defaults to the compile-time threshold but     *
 *  may be changed either statically by defining DEFAULT_SEVERITY to      *
 *  the desired enumeration identifier on the compiler command line, or   *
 *  dynamically by calling setMsgSeverity() to specify a new threshold.   *
 *  The run-time threshold may also be set from the value of the          *
 *  environment variable LEPT_MSG_SEVERITY by calling setMsgSeverity()   *
 *  and specifying L_SEVERITY_EXTERNAL.                                   *
 *                                                                        *
 *  In effect, the compile-time threshold setting says, "Generate code    *
 *  to permit messages of equal or greater severity than this to be       *
 *  printed, if desired," whereas the run-time threshold setting says,    *
 *  "Print messages that have an equal or greater severity than this."    *
 *------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_SEVERITY_EXTERNAL = 0,   /* Get the severity from the environment   */
    L_SEVERITY_ALL      = 1,   /* Lowest severity: print all messages     */
    L_SEVERITY_DEBUG    = 2,   /* Print debugging and higher messages     */
    L_SEVERITY_INFO     = 3,   /* Print informational and higher messages */
    L_SEVERITY_WARNING  = 4,   /* Print warning and higher messages       */
    L_SEVERITY_ERROR    = 5,   /* Print error and higher messages         */
    L_SEVERITY_NONE     = 6;    /* Highest severity: print no messages     */

/*  No message less than the compile-time threshold will ever be
 *  reported, regardless of the current run-time threshold.  This allows
 *  selection of the set of messages to include in the library.  For
 *  example, setting the threshold to L_SEVERITY_WARNING eliminates all
 *  informational messages from the library.  With that setting, both
 *  warning and error messages would be printed unless setMsgSeverity()
 *  was called, or DEFAULT_SEVERITY was redefined, to set the run-time
 *  severity to L_SEVERITY_ERROR.  In that case, only error messages
 *  would be printed.
 *
 *  This mechanism makes the library smaller and faster, by eliminating
 *  undesired message reporting and the associated run-time overhead for
 *  message threshold checking, because code for messages whose severity
 *  is lower than MINIMUM_SEVERITY won't be generated.
 *
 *  A production library might typically permit WARNING and higher
 *  messages to be generated, and a development library might permit
 *  DEBUG and higher.  The actual messages printed (as opposed to
 *  generated) would depend on the current run-time severity threshold.
 */

// #ifdef  NO_CONSOLE_IO
//   #undef MINIMUM_SEVERITY
//   #undef DEFAULT_SEVERITY

  public static final int MINIMUM_SEVERITY =      L_SEVERITY_NONE;
  public static final int DEFAULT_SEVERITY =      L_SEVERITY_NONE;

// #else
//   #ifndef MINIMUM_SEVERITY    /* Compile-time default */
//   #endif

//   #ifndef DEFAULT_SEVERITY   /* Run-time default */
//   #endif
// #endif


/*  The run-time message severity threshold is defined in utils.c.  */
public static native @Cast("l_int32") int LeptMsgSeverity(); public static native void LeptMsgSeverity(int LeptMsgSeverity);

/*
 *  Usage
 *  =====
 *  Messages are of two types.
 *
 *  (1) The messages
 *      ERROR_INT(a,b,c)       : returns l_int32
 *      ERROR_FLOAT(a,b,c)     : returns l_float32
 *      ERROR_PTR(a,b,c)       : returns void*
 *  are used to return from functions and take a fixed set of parameters:
 *      a : <message string>
 *      b : procName
 *      c : <return value from function>
 *  where procName is the name of the local variable naming the function.
 *
 *  (2) The purely informational L_* messages
 *      L_ERROR(a,...)
 *      L_WARNING(a,...)
 *      L_INFO(a,...)
 *  do not take a return value, but they take at least two parameters:
 *      a  :  <message string> with optional format conversions
 *      v1 : procName    (this must be included as the first vararg)
 *      v2, ... :  optional varargs to match format converters in the message
 *
 *  To return an error from a function that returns void, use:
 *      L_ERROR(<message string>, procName, [...])
 *      return;
 *
 *  Implementation details
 *  ======================
 *  Messages are defined with the IF_SEV macro.  The first parameter is
 *  the message severity, the second is the function to call if the
 *  message is to be printed, and the third is the return value if the
 *  message is to be suppressed.  For example, we might have an
 *  informational message defined as:
 *
 *    IF_SEV(L_SEVERITY_INFO, fprintf(.......), 0)
 *
 *  The macro expands into a conditional.  Because the first comparison
 *  is between two constants, an optimizing compiler will remove either
 *  the comparison (if it's true) or the entire macro expansion (if it
 *  is false).  This means that there is no run-time overhead for
 *  messages whose severity falls below the minimum specified at compile
 *  time, and for others the overhead is one (not two) comparisons.
 *
 *  The L_nnn() macros below do not return a value, but because the
 *  conditional operator requires one for the false condition, we
 *  specify a void expression.
 */

// #ifdef  NO_CONSOLE_IO

//   #define PROCNAME(name)
//   #define ERROR_INT(a,b,c)            ((l_int32)(c))
//   #define ERROR_FLOAT(a,b,c)          ((l_float32)(c))
//   #define ERROR_PTR(a,b,c)            ((void *)(c))
//   #define L_ERROR(a,...)
//   #define L_WARNING(a,...)
//   #define L_INFO(a,...)

// #else

//   #define PROCNAME(name)              static const char procName[] = name
//   #define IF_SEV(l,t,f)
//       ((l) >= MINIMUM_SEVERITY && (l) >= LeptMsgSeverity ? (t) : (f))

//   #define ERROR_INT(a,b,c)
//       IF_SEV(L_SEVERITY_ERROR, returnErrorInt((a),(b),(c)), (l_int32)(c))
//   #define ERROR_FLOAT(a,b,c)
//       IF_SEV(L_SEVERITY_ERROR, returnErrorFloat((a),(b),(c)), (l_float32)(c))
//   #define ERROR_PTR(a,b,c)
//       IF_SEV(L_SEVERITY_ERROR, returnErrorPtr((a),(b),(c)), (void *)(c))

//   #define L_ERROR(a,...)
//       IF_SEV(L_SEVERITY_ERROR,
//              (void)fprintf(stderr, "Error in %s: " a, __VA_ARGS__),
//              (void)0)
//   #define L_WARNING(a,...)
//       IF_SEV(L_SEVERITY_WARNING,
//              (void)fprintf(stderr, "Warning in %s: " a, __VA_ARGS__),
//              (void)0)
//   #define L_INFO(a,...)
//       IF_SEV(L_SEVERITY_INFO,
//              (void)fprintf(stderr, "Info in %s: " a, __VA_ARGS__),
//              (void)0)

// #if 0  /* Alternative method for controlling L_* message output */
// #endif

// #endif  /* NO_CONSOLE_IO */


/*------------------------------------------------------------------------*
 *                        snprintf() renamed in MSVC                      *
 *------------------------------------------------------------------------*/
// #ifdef _MSC_VER
// #define snprintf(buf, size, ...)  _snprintf_s(buf, size, _TRUNCATE, __VA_ARGS__)
// #endif


// #endif /* LEPTONICA_ENVIRON_H */


// Parsed from leptonica/array.h

/*====================================================================*
 -  Copyright (C) 2001 Leptonica.  All rights reserved.
 -
 -  Redistribution and use in source and binary forms, with or without
 -  modification, are permitted provided that the following conditions
 -  are met:
 -  1. Redistributions of source code must retain the above copyright
 -     notice, this list of conditions and the following disclaimer.
 -  2. Redistributions in binary form must reproduce the above
 -     copyright notice, this list of conditions and the following
 -     disclaimer in the documentation and/or other materials
 -     provided with the distribution.
 -
 -  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 -  ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 -  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 -  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL ANY
 -  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 -  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 -  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 -  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 -  OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 -  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 -  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *====================================================================*/

// #ifndef  LEPTONICA_ARRAY_H
// #define  LEPTONICA_ARRAY_H

/*
 *  Contains the following structs:
 *      struct Numa
 *      struct Numaa
 *      struct Numa2d
 *      struct NumaHash
 *      struct L_Dna
 *      struct L_Dnaa
 *      struct Sarray
 *      struct L_Bytea
 *
 *  Contains definitions for:
 *      Numa interpolation flags
 *      Numa and FPix border flags
 *      Numa data type conversion to string
 */


/*------------------------------------------------------------------------*
 *                             Array Structs                              *
 *------------------------------------------------------------------------*/

public static final int NUMA_VERSION_NUMBER =     1;

    /* Number array: an array of floats */
@Name("Numa") public static class NUMA extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public NUMA() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public NUMA(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public NUMA(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public NUMA position(int position) {
        return (NUMA)super.position(position);
    }

    public native @Cast("l_int32") int nalloc(); public native NUMA nalloc(int nalloc);    /* size of allocated number array      */
    public native @Cast("l_int32") int n(); public native NUMA n(int n);         /* number of numbers saved             */
    public native @Cast("l_int32") int refcount(); public native NUMA refcount(int refcount);  /* reference count (1 if no clones)    */
    public native @Cast("l_float32") float startx(); public native NUMA startx(float startx);    /* x value assigned to array[0]        */
    public native @Cast("l_float32") float delx(); public native NUMA delx(float delx);      /* change in x value as i --> i + 1    */
    public native @Cast("l_float32*") FloatPointer array(); public native NUMA array(FloatPointer array);     /* number array                        */
}


    /* Array of number arrays */


    /* Sparse 2-dimensional array of number arrays */


    /* A hash table of Numas */


public static final int DNA_VERSION_NUMBER =     1;

    /* Double number array: an array of doubles */


    /* Array of double number arrays */


public static final int SARRAY_VERSION_NUMBER =     1;

    /* String array: an array of C strings */


    /* Byte array (analogous to C++ "string") */


/*------------------------------------------------------------------------*
 *                              Array flags                               *
 *------------------------------------------------------------------------*/
    /* Flags for interpolation in Numa */
/** enum  */
public static final int
    L_LINEAR_INTERP = 1,        /* linear     */
    L_QUADRATIC_INTERP = 2;      /* quadratic  */

    /* Flags for added borders in Numa and Fpix */
/** enum  */
public static final int
    L_CONTINUED_BORDER = 1,     /* extended with same value                  */
    L_SLOPE_BORDER = 2,         /* extended with constant normal derivative  */
    L_MIRRORED_BORDER = 3;       /* mirrored                                  */

    /* Flags for data type converted from Numa */
/** enum  */
public static final int
    L_INTEGER_VALUE = 1,        /* convert to integer  */
    L_FLOAT_VALUE = 2;           /* convert to float    */


// #endif  /* LEPTONICA_ARRAY_H */


// Parsed from leptonica/stack.h

/*====================================================================*
 -  Copyright (C) 2001 Leptonica.  All rights reserved.
 -
 -  Redistribution and use in source and binary forms, with or without
 -  modification, are permitted provided that the following conditions
 -  are met:
 -  1. Redistributions of source code must retain the above copyright
 -     notice, this list of conditions and the following disclaimer.
 -  2. Redistributions in binary form must reproduce the above
 -     copyright notice, this list of conditions and the following
 -     disclaimer in the documentation and/or other materials
 -     provided with the distribution.
 - 
 -  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 -  ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 -  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 -  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL ANY
 -  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 -  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 -  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 -  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 -  OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 -  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 -  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *====================================================================*/

// #ifndef  LEPTONICA_STACK_H
// #define  LEPTONICA_STACK_H

/*
 *  stack.h
 *
 *       Expandable pointer stack for arbitrary void* data.
 *
 *       The L_Stack is an array of void * ptrs, onto which arbitrary
 *       objects can be stored.  At any time, the number of
 *       stored objects is stack->n.  The object at the bottom
 *       of the stack is at array[0]; the object at the top of
 *       the stack is at array[n-1].  New objects are added
 *       to the top of the stack, at the first available location,
 *       which is array[n].  Objects are removed from the top of the
 *       stack.  When an attempt is made to remove an object from an
 *       empty stack, the result is null.   When the stack becomes
 *       filled, so that n = nalloc, the size is doubled.
 *
 *       The auxiliary stack can be used to store and remove
 *       objects for re-use.  It must be created by a separate
 *       call to pstackCreate().  [Just imagine the chaos if
 *       pstackCreate() created the auxiliary stack!]   
 *       pstackDestroy() checks for the auxiliary stack and removes it.
 */


    /* Note that array[n] is the first null ptr in the array */
@Name("L_Stack") public static class L_STACK extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public L_STACK() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public L_STACK(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public L_STACK(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public L_STACK position(int position) {
        return (L_STACK)super.position(position);
    }

    public native @Cast("l_int32") int nalloc(); public native L_STACK nalloc(int nalloc);       /* size of ptr array              */
    public native @Cast("l_int32") int n(); public native L_STACK n(int n);            /* number of stored elements      */
    public native Pointer array(int i); public native L_STACK array(int i, Pointer array);
    @MemberGetter public native @Cast("void**") PointerPointer array();        /* ptr array                      */
    public native L_STACK auxstack(); public native L_STACK auxstack(L_STACK auxstack);     /* auxiliary stack                */
}


// #endif /*  LEPTONICA_STACK_H */



// Parsed from leptonica/imageio.h

/*====================================================================*
 -  Copyright (C) 2001 Leptonica.  All rights reserved.
 -
 -  Redistribution and use in source and binary forms, with or without
 -  modification, are permitted provided that the following conditions
 -  are met:
 -  1. Redistributions of source code must retain the above copyright
 -     notice, this list of conditions and the following disclaimer.
 -  2. Redistributions in binary form must reproduce the above
 -     copyright notice, this list of conditions and the following
 -     disclaimer in the documentation and/or other materials
 -     provided with the distribution.
 -
 -  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 -  ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 -  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 -  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL ANY
 -  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 -  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 -  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 -  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 -  OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 -  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 -  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *====================================================================*/

/*
 *  General features of image I/O in leptonica
 *
 *  At present, there are 9 file formats for images that can be read
 *  and written:
 *      png (requires libpng, libz)
 *      jpeg (requires libjpeg)
 *      tiff (requires libtiff, libz)
 *      gif (requires libgif)
 *      webp (requires libwebp)
 *      jp2 (requires libopenjp2)
 *      bmp (no library required)
 *      pnm (no library required)
 *      spix (no library required)
 *  Additionally, there are two file formats for writing (only) images:
 *      PostScript (requires libpng, libz, libjpeg, libtiff)
 *      pdf (requires libpng, libz, libjpeg, libtiff)
 *
 *  For all 9 read/write formats, leptonica provides interconversion
 *  between pix (with raster data) and formatted image data:
 *      Conversion from pix (typically compression):
 *          pixWrite():        pix --> file
 *          pixWriteStream():  pix --> filestream (aka FILE*)
 *          pixWriteMem():     pix --> memory buffer
 *      Conversion to pix (typically decompression):
 *          pixRead():         file --> pix
 *          pixReadStream():   filestream --> pix
 *          pixReadMem():      memory buffer --> pix
 *
 *  Conversions for which the image data is not compressed are:
 *     * uncompressed tiff   (IFF_TIFF)
 *     * bmp
 *     * pnm
 *     * spix (fast serialization that copies the pix raster data)
 *
 *  The image header (metadata) information can be read from either
 *  the compressed file or a memory buffer, for all 9 formats.
 */

// #ifndef  LEPTONICA_IMAGEIO_H
// #define  LEPTONICA_IMAGEIO_H

/* ------------------ Image file format types -------------- */
/*
 *  The IFF_DEFAULT flag is used to write the file out in the
 *  same (input) file format that the pix was read from.  If the pix
 *  was not read from file, the input format field will be
 *  IFF_UNKNOWN and the output file format will be chosen to
 *  be compressed and lossless; namely, IFF_TIFF_G4 for d = 1
 *  and IFF_PNG for everything else.   IFF_JP2 is for jpeg2000, which
 *  is not supported in leptonica.
 *
 *  In the future, new format types that have defined extensions
 *  will be added before IFF_DEFAULT, and will be kept in sync with
 *  the file format extensions in writefile.c.  The positions of
 *  file formats before IFF_DEFAULT will remain invariant.
 */
/** enum  */
public static final int
    IFF_UNKNOWN        = 0,
    IFF_BMP            = 1,
    IFF_JFIF_JPEG      = 2,
    IFF_PNG            = 3,
    IFF_TIFF           = 4,
    IFF_TIFF_PACKBITS  = 5,
    IFF_TIFF_RLE       = 6,
    IFF_TIFF_G3        = 7,
    IFF_TIFF_G4        = 8,
    IFF_TIFF_LZW       = 9,
    IFF_TIFF_ZIP       = 10,
    IFF_PNM            = 11,
    IFF_PS             = 12,
    IFF_GIF            = 13,
    IFF_JP2            = 14,
    IFF_WEBP           = 15,
    IFF_LPDF           = 16,
    IFF_DEFAULT        = 17,
    IFF_SPIX           = 18;


/* ---------------------- Format header ids --------------------- */
/** enum  */
public static final int
    BMP_ID             =  0x4d42,
    TIFF_BIGEND_ID     =  0x4d4d,     /* MM - for 'motorola' */
    TIFF_LITTLEEND_ID  =  0x4949;      /* II - for 'intel'    */


/* ------------- Hinting bit flags in jpeg reader --------------- */
/** enum  */
public static final int
    L_JPEG_READ_LUMINANCE = 1,  /* only want luminance data; no chroma */
    L_JPEG_FAIL_ON_BAD_DATA = 2;  /* don't return possibly damaged pix */


/* ------------------ Pdf formated encoding types --------------- */
/** enum  */
public static final int
    L_JPEG_ENCODE   = 1,    /* use dct encoding: 8 and 32 bpp, no cmap     */
    L_G4_ENCODE     = 2,    /* use ccitt g4 fax encoding: 1 bpp            */
    L_FLATE_ENCODE  = 3,    /* use flate encoding: any depth, cmap ok      */
    L_JP2K_ENCODE  = 4;      /* use jp2k encoding: 8 and 32 bpp, no cmap    */


/* ------------------ Compressed image data --------------------- */
/*
 *  In use, either datacomp or data85 will be produced, depending
 *  on whether the data needs to be ascii85 encoded.  PostScript
 *  requires ascii85 encoding; pdf does not.
 *
 *  For the colormap (flate compression only), PostScript uses ascii85
 *  encoding and pdf uses a bracketed array of space-separated
 *  hex-encoded rgb triples.  Only tiff g4 (type == L_G4_ENCODE) uses
 *  the minisblack field.
 */
@Name("L_Compressed_Data") public static class L_COMP_DATA extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public L_COMP_DATA() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public L_COMP_DATA(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public L_COMP_DATA(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public L_COMP_DATA position(int position) {
        return (L_COMP_DATA)super.position(position);
    }

    public native @Cast("l_int32") int type(); public native L_COMP_DATA type(int type);         /* encoding type: L_JPEG_ENCODE, etc  */
    public native @Cast("l_uint8*") BytePointer datacomp(); public native L_COMP_DATA datacomp(BytePointer datacomp);     /* gzipped raster data                 */
    public native @Cast("size_t") long nbytescomp(); public native L_COMP_DATA nbytescomp(long nbytescomp);   /* number of compressed bytes          */
    public native @Cast("char*") BytePointer data85(); public native L_COMP_DATA data85(BytePointer data85);       /* ascii85-encoded gzipped raster data */
    public native @Cast("size_t") long nbytes85(); public native L_COMP_DATA nbytes85(long nbytes85);     /* number of ascii85 encoded bytes     */
    public native @Cast("char*") BytePointer cmapdata85(); public native L_COMP_DATA cmapdata85(BytePointer cmapdata85);   /* ascii85-encoded uncompressed cmap   */
    public native @Cast("char*") BytePointer cmapdatahex(); public native L_COMP_DATA cmapdatahex(BytePointer cmapdatahex);  /* hex pdf array for the cmap          */
    public native @Cast("l_int32") int ncolors(); public native L_COMP_DATA ncolors(int ncolors);      /* number of colors in cmap            */
    public native @Cast("l_int32") int w(); public native L_COMP_DATA w(int w);            /* image width                         */
    public native @Cast("l_int32") int h(); public native L_COMP_DATA h(int h);            /* image height                        */
    public native @Cast("l_int32") int bps(); public native L_COMP_DATA bps(int bps);          /* bits/sample; typ. 1, 2, 4 or 8      */
    public native @Cast("l_int32") int spp(); public native L_COMP_DATA spp(int spp);          /* samples/pixel; typ. 1 or 3          */
    public native @Cast("l_int32") int minisblack(); public native L_COMP_DATA minisblack(int minisblack);   /* tiff g4 photometry                  */
    public native @Cast("l_int32") int predictor(); public native L_COMP_DATA predictor(int predictor);    /* flate data has PNG predictors       */
    public native @Cast("size_t") long nbytes(); public native L_COMP_DATA nbytes(long nbytes);       /* number of uncompressed raster bytes */
    public native @Cast("l_int32") int res(); public native L_COMP_DATA res(int res);          /* resolution (ppi)                    */
}


/* ------------------------ Pdf multi-image flags ------------------------ */
/** enum  */
public static final int
    L_FIRST_IMAGE   = 1,    /* first image to be used                      */
    L_NEXT_IMAGE    = 2,    /* intermediate image; not first or last       */
    L_LAST_IMAGE    = 3;     /* last image to be used                       */


/* ------------------ Intermediate pdf generation data -------------------- */
/*
 *  This accumulates data for generating a pdf of a single page consisting
 *  of an arbitrary number of images.
 *
 *  None of the strings have a trailing newline.
 */


// #endif  /* LEPTONICA_IMAGEIO_H */


// Parsed from leptonica/morph.h

/*====================================================================*
 -  Copyright (C) 2001 Leptonica.  All rights reserved.
 -
 -  Redistribution and use in source and binary forms, with or without
 -  modification, are permitted provided that the following conditions
 -  are met:
 -  1. Redistributions of source code must retain the above copyright
 -     notice, this list of conditions and the following disclaimer.
 -  2. Redistributions in binary form must reproduce the above
 -     copyright notice, this list of conditions and the following
 -     disclaimer in the documentation and/or other materials
 -     provided with the distribution.
 - 
 -  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 -  ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 -  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 -  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL ANY
 -  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 -  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 -  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 -  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 -  OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 -  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 -  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *====================================================================*/

// #ifndef  LEPTONICA_MORPH_H
// #define  LEPTONICA_MORPH_H

/* 
 *  morph.h
 *
 *  Contains the following structs:
 *      struct Sel
 *      struct Sela
 *      struct Kernel
 *
 *  Contains definitions for:
 *      morphological b.c. flags
 *      structuring element types
 *      runlength flags for granulometry
 *      direction flags for grayscale morphology
 *      morphological operation flags
 *      standard border size
 *      grayscale intensity scaling flags
 *      morphological tophat flags
 *      arithmetic and logical operator flags
 *      grayscale morphology selection flags
 *      distance function b.c. flags
 *      image comparison flags
 *      color content flags
 */

/*-------------------------------------------------------------------------*
 *                             Sel and Sel array                           *
 *-------------------------------------------------------------------------*/
public static final int SEL_VERSION_NUMBER =    1;

@Name("Sel") public static class SEL extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public SEL() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public SEL(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public SEL(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public SEL position(int position) {
        return (SEL)super.position(position);
    }

    public native @Cast("l_int32") int sy(); public native SEL sy(int sy);          /* sel height                               */
    public native @Cast("l_int32") int sx(); public native SEL sx(int sx);          /* sel width                                */
    public native @Cast("l_int32") int cy(); public native SEL cy(int cy);          /* y location of sel origin                 */
    public native @Cast("l_int32") int cx(); public native SEL cx(int cx);          /* x location of sel origin                 */
    public native @Cast("l_int32*") IntPointer data(int i); public native SEL data(int i, IntPointer data);
    @MemberGetter public native @Cast("l_int32**") PointerPointer data();        /* {0,1,2}; data[i][j] in [row][col] order  */
    public native @Cast("char*") BytePointer name(); public native SEL name(BytePointer name);        /* used to find sel by name                 */
}


/*-------------------------------------------------------------------------*
 *                                 Kernel                                  *
 *-------------------------------------------------------------------------*/
public static final int KERNEL_VERSION_NUMBER =    2;


/*-------------------------------------------------------------------------*
 *                 Morphological boundary condition flags                  *
 *
 *  Two types of boundary condition for erosion.
 *  The global variable MORPH_BC takes on one of these two values.
 *  See notes in morph.c for usage.
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    SYMMETRIC_MORPH_BC = 0,
    ASYMMETRIC_MORPH_BC = 1;


/*-------------------------------------------------------------------------*
 *                        Structuring element types                        *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    SEL_DONT_CARE  = 0,
    SEL_HIT        = 1,
    SEL_MISS       = 2;


/*-------------------------------------------------------------------------*
 *                  Runlength flags for granulometry                       *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_RUN_OFF = 0,
    L_RUN_ON  = 1;


/*-------------------------------------------------------------------------*
 *         Direction flags for grayscale morphology, granulometry,         *
 *                 composable Sels, convolution, etc.                      *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_HORIZ            = 1,
    L_VERT             = 2,
    L_BOTH_DIRECTIONS  = 3;


/*-------------------------------------------------------------------------*
 *                   Morphological operation flags                         *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_MORPH_DILATE    = 1,
    L_MORPH_ERODE     = 2,
    L_MORPH_OPEN      = 3,
    L_MORPH_CLOSE     = 4,
    L_MORPH_HMT       = 5;


/*-------------------------------------------------------------------------*
 *                    Grayscale intensity scaling flags                    *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_LINEAR_SCALE  = 1,
    L_LOG_SCALE     = 2;


/*-------------------------------------------------------------------------*
 *                      Morphological tophat flags                         *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_TOPHAT_WHITE = 0,
    L_TOPHAT_BLACK = 1;


/*-------------------------------------------------------------------------*
 *                Arithmetic and logical operator flags                    *
 *                 (use on grayscale images and Numas)                     *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_ARITH_ADD       = 1,
    L_ARITH_SUBTRACT  = 2,
    L_ARITH_MULTIPLY  = 3,   /* on numas only */
    L_ARITH_DIVIDE    = 4,   /* on numas only */
    L_UNION           = 5,   /* on numas only */
    L_INTERSECTION    = 6,   /* on numas only */
    L_SUBTRACTION     = 7,   /* on numas only */
    L_EXCLUSIVE_OR    = 8;    /* on numas only */


/*-------------------------------------------------------------------------*
 *                        Min/max selection flags                          *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_CHOOSE_MIN = 1,           /* useful in a downscaling "erosion"  */
    L_CHOOSE_MAX = 2,           /* useful in a downscaling "dilation" */
    L_CHOOSE_MAX_MIN_DIFF = 3;   /* useful in a downscaling contrast   */


/*-------------------------------------------------------------------------*
 *                    Distance function b.c. flags                         *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_BOUNDARY_BG = 1,  /* assume bg outside image */
    L_BOUNDARY_FG = 2;   /* assume fg outside image */


/*-------------------------------------------------------------------------*
 *                         Image comparison flags                          *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_COMPARE_XOR = 1,
    L_COMPARE_SUBTRACT = 2,
    L_COMPARE_ABS_DIFF = 3;


/*-------------------------------------------------------------------------*
 *                          Color content flags                            *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_MAX_DIFF_FROM_AVERAGE_2 = 1,
    L_MAX_MIN_DIFF_FROM_2 = 2,
    L_MAX_DIFF = 3;


/*-------------------------------------------------------------------------*
 *    Standard size of border added around images for special processing   *
 *-------------------------------------------------------------------------*/
@MemberGetter public static native @Cast("const l_int32") int ADDED_BORDER();   /* pixels, not bits */


// #endif  /* LEPTONICA_MORPH_H */


// Parsed from leptonica/pix.h

/*====================================================================*
 -  Copyright (C) 2001 Leptonica.  All rights reserved.
 -
 -  Redistribution and use in source and binary forms, with or without
 -  modification, are permitted provided that the following conditions
 -  are met:
 -  1. Redistributions of source code must retain the above copyright
 -     notice, this list of conditions and the following disclaimer.
 -  2. Redistributions in binary form must reproduce the above
 -     copyright notice, this list of conditions and the following
 -     disclaimer in the documentation and/or other materials
 -     provided with the distribution.
 -
 -  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 -  ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 -  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 -  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL ANY
 -  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 -  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 -  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 -  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 -  OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 -  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 -  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *====================================================================*/

// #ifndef  LEPTONICA_PIX_H
// #define  LEPTONICA_PIX_H

/*
 *   pix.h
 *
 *   Valid image types in leptonica:
 *       Pix: 1 bpp, with and without colormap
 *       Pix: 2 bpp, with and without colormap
 *       Pix: 4 bpp, with and without colormap
 *       Pix: 8 bpp, with and without colormap
 *       Pix: 16 bpp (1 spp)
 *       Pix: 32 bpp (rgb, 3 spp)
 *       Pix: 32 bpp (rgba, 4 spp)
 *       FPix: 32 bpp float
 *       DPix: 64 bpp double
 *       Notes:
 *          (1) The only valid Pix image type with alpha is rgba.
 *              In particular, the alpha component is not used in
 *              cmapped images.
 *          (2) PixComp can hold any Pix with IFF_PNG encoding.
 *
 *   This file defines most of the image-related structs used in leptonica:
 *       struct Pix
 *       struct PixColormap
 *       struct RGBA_Quad
 *       struct Pixa
 *       struct Pixaa
 *       struct Box
 *       struct Boxa
 *       struct Boxaa
 *       struct Pta
 *       struct Ptaa
 *       struct Pixacc
 *       struct PixTiling
 *       struct FPix
 *       struct FPixa
 *       struct DPix
 *       struct PixComp
 *       struct PixaComp
 *
 *   This file has definitions for:
 *       Colors for RGB
 *       Perceptual color weights
 *       Colormap conversion flags
 *       Rasterop bit flags
 *       Structure access flags (for insert, copy, clone, copy-clone)
 *       Sorting flags (by type and direction)
 *       Blending flags
 *       Graphics pixel setting flags
 *       Size filtering flags
 *       Color component selection flags
 *       16-bit conversion flags
 *       Rotation and shear flags
 *       Affine transform order flags
 *       Grayscale filling flags
 *       Flags for setting to white or black
 *       Flags for getting white or black pixel value
 *       Flags for 8 and 16 bit pixel sums
 *       Dithering flags
 *       Distance flags
 *       Statistical measures
 *       Set selection flags
 *       Text orientation flags
 *       Edge orientation flags
 *       Line orientation flags
 *       Scan direction flags
 *       Box size adjustment flags
 *       Flags for selecting box boundaries from two choices
 *       Handling overlapping bounding boxes in boxa
 *       Flags for replacing invalid boxes
 *       Horizontal warp
 *       Pixel selection for resampling
 *       Thinning flags
 *       Runlength flags
 *       Edge filter flags
 *       Subpixel color component ordering in LCD display
 *       HSV histogram flags
 *       Region flags (inclusion, exclusion)
 *       Flags for adding text to a pix
 *       Flags for plotting on a pix
 *       Flags for selecting display program
 *       Flags in the 'special' pix field for non-default operations
 *       Handling negative values in conversion to unsigned int
 *       Relative to zero flags
 *       Flags for adding or removing traling slash from string                *
 */


/*-------------------------------------------------------------------------*
 *                              Basic Pix                                  *
 *-------------------------------------------------------------------------*/
    /* The 'special' field is by default 0, but it can hold integers
     * that direct non-default actions, e.g., in png and jpeg I/O. */
@Name("Pix") public static class PIX extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public PIX() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public PIX(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public PIX(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public PIX position(int position) {
        return (PIX)super.position(position);
    }

    public native @Cast("l_uint32") int w(); public native PIX w(int w);           /* width in pixels                   */
    public native @Cast("l_uint32") int h(); public native PIX h(int h);           /* height in pixels                  */
    public native @Cast("l_uint32") int d(); public native PIX d(int d);           /* depth in bits (bpp)               */
    public native @Cast("l_uint32") int spp(); public native PIX spp(int spp);         /* number of samples per pixel       */
    public native @Cast("l_uint32") int wpl(); public native PIX wpl(int wpl);         /* 32-bit words/line                 */
    public native @Cast("l_uint32") int refcount(); public native PIX refcount(int refcount);    /* reference count (1 if no clones)  */
    public native @Cast("l_int32") int xres(); public native PIX xres(int xres);        /* image res (ppi) in x direction    */
                                      /* (use 0 if unknown)                */
    public native @Cast("l_int32") int yres(); public native PIX yres(int yres);        /* image res (ppi) in y direction    */
                                      /* (use 0 if unknown)                */
    public native @Cast("l_int32") int informat(); public native PIX informat(int informat);    /* input file format, IFF_*          */
    public native @Cast("l_int32") int special(); public native PIX special(int special);     /* special instructions for I/O, etc */
    public native @Cast("char*") BytePointer text(); public native PIX text(BytePointer text);        /* text string associated with pix   */
    public native PIXCMAP colormap(); public native PIX colormap(PIXCMAP colormap);    /* colormap (may be null)            */
    public native @Cast("l_uint32*") IntPointer data(); public native PIX data(IntPointer data);        /* the image data                    */
}


@Name("PixColormap") public static class PIXCMAP extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public PIXCMAP() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public PIXCMAP(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public PIXCMAP(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public PIXCMAP position(int position) {
        return (PIXCMAP)super.position(position);
    }

    public native Pointer array(); public native PIXCMAP array(Pointer array);     /* colormap table (array of RGBA_QUAD)     */
    public native @Cast("l_int32") int depth(); public native PIXCMAP depth(int depth);     /* of pix (1, 2, 4 or 8 bpp)               */
    public native @Cast("l_int32") int nalloc(); public native PIXCMAP nalloc(int nalloc);    /* number of color entries allocated       */
    public native @Cast("l_int32") int n(); public native PIXCMAP n(int n);         /* number of color entries used            */
}


    /* Colormap table entry (after the BMP version).
     * Note that the BMP format stores the colormap table exactly
     * as it appears here, with color samples being stored sequentially,
     * in the order (b,g,r,a). */



/*-------------------------------------------------------------------------*
 *                             Colors for 32 bpp                           *
 *-------------------------------------------------------------------------*/
/*  Notes:
 *      (1) These are the byte indices for colors in 32 bpp images.
 *          They are used through the GET/SET_DATA_BYTE accessors.
 *          The 4th byte, typically known as the "alpha channel" and used
 *          for blending, is used to a small extent in leptonica.
 *      (2) Do not change these values!  If you redefine them, functions
 *          that have the shifts hardcoded for efficiency and conciseness
 *          (instead of using the constants below) will break.  These
 *          functions are labelled with "***"  next to their names at
 *          the top of the files in which they are defined.
 *      (3) The shifts to extract the red, green, blue and alpha components
 *          from a 32 bit pixel are defined here.
 */
/** enum  */
public static final int
    COLOR_RED = 0,
    COLOR_GREEN = 1,
    COLOR_BLUE = 2,
    L_ALPHA_CHANNEL = 3;

@MemberGetter public static native @Cast("const l_int32") int L_RED_SHIFT();           /* 24 */
@MemberGetter public static native @Cast("const l_int32") int L_GREEN_SHIFT();         /* 16 */
@MemberGetter public static native @Cast("const l_int32") int L_BLUE_SHIFT();          /*  8 */
@MemberGetter public static native @Cast("const l_int32") int L_ALPHA_SHIFT();     /*  0 */


/*-------------------------------------------------------------------------*
 *                       Perceptual color weights                          *
 *-------------------------------------------------------------------------*/
/*  Notes:
 *      (1) These numbers are ad-hoc, but they do add up to 1.
 *          Unlike, for example, the weighting factor for conversion
 *          of RGB to luminance, or more specifically to Y in the
 *          YUV colorspace.  Those numbers come from the
 *          International Telecommunications Union, via ITU-R.
 */
@MemberGetter public static native @Cast("const l_float32") float L_RED_WEIGHT();
@MemberGetter public static native @Cast("const l_float32") float L_GREEN_WEIGHT();
@MemberGetter public static native @Cast("const l_float32") float L_BLUE_WEIGHT();


/*-------------------------------------------------------------------------*
 *                        Flags for colormap conversion                    *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    REMOVE_CMAP_TO_BINARY = 0,
    REMOVE_CMAP_TO_GRAYSCALE = 1,
    REMOVE_CMAP_TO_FULL_COLOR = 2,
    REMOVE_CMAP_WITH_ALPHA = 3,
    REMOVE_CMAP_BASED_ON_SRC = 4;


/*-------------------------------------------------------------------------*
 *
 * The following operation bit flags have been modified from
 * Sun's pixrect.h.
 *
 * The 'op' in 'rasterop' is represented by an integer
 * composed with Boolean functions using the set of five integers
 * given below.  The integers, and the op codes resulting from
 * boolean expressions on them, need only be in the range from 0 to 15.
 * The function is applied on a per-pixel basis.
 *
 * Examples: the op code representing ORing the src and dest
 * is computed using the bit OR, as PIX_SRC | PIX_DST;  the op
 * code representing XORing src and dest is found from
 * PIX_SRC ^ PIX_DST;  the op code representing ANDing src and dest
 * is found from PIX_SRC & PIX_DST.  Note that
 * PIX_NOT(PIX_CLR) = PIX_SET, and v.v., as they must be.
 *
 * We would like to use the following set of definitions:
 *
 *      #define   PIX_SRC      0xc
 *      #define   PIX_DST      0xa
 *      #define   PIX_NOT(op)  ((op) ^ 0xf)
 *      #define   PIX_CLR      0x0
 *      #define   PIX_SET      0xf
 *
 * Now, these definitions differ from Sun's, in that Sun
 * left-shifted each value by 1 pixel, and used the least
 * significant bit as a flag for the "pseudo-operation" of
 * clipping.  We don't need this bit, because it is both
 * efficient and safe ALWAYS to clip the rectangles to the src
 * and dest images, which is what we do.  See the notes in rop.h
 * on the general choice of these bit flags.
 *
 * However, if you include Sun's xview package, you will get their
 * definitions, and because I like using these flags, we will
 * adopt the original Sun definitions to avoid redefinition conflicts.
 *
 * Then we have, for reference, the following 16 unique op flags:
 *
 *      PIX_CLR                           00000             0x0
 *      PIX_SET                           11110             0x1e
 *      PIX_SRC                           11000             0x18
 *      PIX_DST                           10100             0x14
 *      PIX_NOT(PIX_SRC)                  00110             0x06
 *      PIX_NOT(PIX_DST)                  01010             0x0a
 *      PIX_SRC | PIX_DST                 11100             0x1c
 *      PIX_SRC & PIX_DST                 10000             0x10
 *      PIX_SRC ^ PIX_DST                 01100             0x0c
 *      PIX_NOT(PIX_SRC) | PIX_DST        10110             0x16
 *      PIX_NOT(PIX_SRC) & PIX_DST        00100             0x04
 *      PIX_SRC | PIX_NOT(PIX_DST)        11010             0x1a
 *      PIX_SRC & PIX_NOT(PIX_DST)        01000             0x08
 *      PIX_NOT(PIX_SRC | PIX_DST)        00010             0x02
 *      PIX_NOT(PIX_SRC & PIX_DST)        01110             0x0e
 *      PIX_NOT(PIX_SRC ^ PIX_DST)        10010             0x12
 *
 *-------------------------------------------------------------------------*/
public static final int PIX_SRC =      (0xc << 1);
public static final int PIX_DST =      (0xa << 1);
// #define   PIX_NOT(op)  ((op) ^ 0x1e)
public static final int PIX_CLR =      (0x0 << 1);
public static final int PIX_SET =      (0xf << 1);

public static final int PIX_PAINT =    (PIX_SRC | PIX_DST);
public static final int PIX_MASK =     (PIX_SRC & PIX_DST);
public static native @MemberGetter int PIX_SUBTRACT();
public static final int PIX_SUBTRACT = PIX_SUBTRACT();
public static final int PIX_XOR =      (PIX_SRC ^ PIX_DST);


/*-------------------------------------------------------------------------*
 *
 *   Important Notes:
 *
 *       (1) The image data is stored in a single contiguous
 *           array of l_uint32, into which the pixels are packed.
 *           By "packed" we mean that there are no unused bits
 *           between pixels, except for end-of-line padding to
 *           satisfy item (2) below.
 *
 *       (2) Every image raster line begins on a 32-bit word
 *           boundary within this array.
 *
 *       (3) Pix image data is stored in 32-bit units, with the
 *           pixels ordered from left to right in the image being
 *           stored in order from the MSB to LSB within the word,
 *           for both big-endian and little-endian machines.
 *           This is the natural ordering for big-endian machines,
 *           as successive bytes are stored and fetched progressively
 *           to the right.  However, for little-endians, when storing
 *           we re-order the bytes from this byte stream order, and
 *           reshuffle again for byte access on 32-bit entities.
 *           So if the bytes come in sequence from left to right, we
 *           store them on little-endians in byte order:
 *                3 2 1 0 7 6 5 4 ...
 *           This MSB to LSB ordering allows left and right shift
 *           operations on 32 bit words to move the pixels properly.
 *
 *       (4) We use 32 bit pixels for both RGB and RGBA color images.
 *           The A (alpha) byte is ignored in most leptonica functions
 *           operating on color images.  Within each 4 byte pixel, the
 *           colors are ordered from MSB to LSB, as follows:
 *
 *                |  MSB  |  2nd MSB  |  3rd MSB  |  LSB  |
 *                   red      green       blue      alpha
 *                    0         1           2         3   (big-endian)
 *                    3         2           1         0   (little-endian)
 *
 *           Because we use MSB to LSB ordering within the 32-bit word,
 *           the individual 8-bit samples can be accessed with
 *           GET_DATA_BYTE and SET_DATA_BYTE macros, using the
 *           (implicitly big-ending) ordering
 *                 red:    byte 0  (MSB)
 *                 green:  byte 1  (2nd MSB)
 *                 blue:   byte 2  (3rd MSB)
 *                 alpha:  byte 3  (LSB)
 *
 *           The specific color assignment is made in this file,
 *           through the definitions of COLOR_RED, etc.  Then the R, G
 *           B and A sample values can be retrieved using
 *                 redval = GET_DATA_BYTE(&pixel, COLOR_RED);
 *                 greenval = GET_DATA_BYTE(&pixel, COLOR_GREEN);
 *                 blueval = GET_DATA_BYTE(&pixel, COLOR_BLUE);
 *                 alphaval = GET_DATA_BYTE(&pixel, L_ALPHA_CHANNEL);
 *           and they can be set with
 *                 SET_DATA_BYTE(&pixel, COLOR_RED, redval);
 *                 SET_DATA_BYTE(&pixel, COLOR_GREEN, greenval);
 *                 SET_DATA_BYTE(&pixel, COLOR_BLUE, blueval);
 *                 SET_DATA_BYTE(&pixel, L_ALPHA_CHANNEL, alphaval);
 *
 *           For extra speed we extract these components directly
 *           by shifting and masking, explicitly using the values in
 *           L_RED_SHIFT, etc.:
 *                 (pixel32 >> L_RED_SHIFT) & 0xff;         (red)
 *                 (pixel32 >> L_GREEN_SHIFT) & 0xff;       (green)
 *                 (pixel32 >> L_BLUE_SHIFT) & 0xff;        (blue)
 *                 (pixel32 >> L_ALPHA_SHIFT) & 0xff;       (alpha)
 *           All these operations work properly on both big- and little-endians.
 *
 *           For a few situations, these color shift values are hard-coded.
 *           Changing the RGB color component ordering through the assignments
 *           in this file will cause functions marked with "***" to fail.
 *
 *       (5) A reference count is held within each pix, giving the
 *           number of ptrs to the pix.  When a pixClone() call
 *           is made, the ref count is increased by 1, and
 *           when a pixDestroy() call is made, the reference count
 *           of the pix is decremented.  The pix is only destroyed
 *           when the reference count goes to zero.
 *
 *       (6) The version numbers (below) are used in the serialization
 *           of these data structures.  They are placed in the files,
 *           and rarely (if ever) change.  Provision is currently made for
 *           backward compatibility in reading from boxaa version 2.
 *
 *       (7) The serialization dependencies are as follows:
 *               pixaa  :  pixa  :  boxa
 *               boxaa  :  boxa
 *           So, for example, pixaa and boxaa can be changed without
 *           forcing a change in pixa or boxa.  However, if pixa is
 *           changed, it forces a change in pixaa, and if boxa is
 *           changed, if forces a change in the other three.
 *           We define four version numbers:
 *               PIXAA_VERSION_NUMBER
 *               PIXA_VERSION_NUMBER
 *               BOXAA_VERSION_NUMBER
 *               BOXA_VERSION_NUMBER
 *
 *-------------------------------------------------------------------------*/



/*-------------------------------------------------------------------------*
 *                              Array of pix                               *
 *-------------------------------------------------------------------------*/

    /*  Serialization for primary data structures */
public static final int PIXAA_VERSION_NUMBER =      2;
public static final int PIXA_VERSION_NUMBER =       2;
public static final int BOXA_VERSION_NUMBER =       2;
public static final int BOXAA_VERSION_NUMBER =      3;


@Name("Pixa") public static class PIXA extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public PIXA() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public PIXA(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public PIXA(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public PIXA position(int position) {
        return (PIXA)super.position(position);
    }

    public native @Cast("l_int32") int n(); public native PIXA n(int n);            /* number of Pix in ptr array        */
    public native @Cast("l_int32") int nalloc(); public native PIXA nalloc(int nalloc);       /* number of Pix ptrs allocated      */
    public native @Cast("l_uint32") int refcount(); public native PIXA refcount(int refcount);     /* reference count (1 if no clones)  */
    public native PIX pix(int i); public native PIXA pix(int i, PIX pix);
    @MemberGetter public native @Cast("Pix**") PointerPointer pix();          /* the array of ptrs to pix          */
    public native BOXA boxa(); public native PIXA boxa(BOXA boxa);         /* array of boxes                    */
}


@Name("Pixaa") public static class PIXAA extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public PIXAA() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public PIXAA(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public PIXAA(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public PIXAA position(int position) {
        return (PIXAA)super.position(position);
    }

    public native @Cast("l_int32") int n(); public native PIXAA n(int n);            /* number of Pixa in ptr array       */
    public native @Cast("l_int32") int nalloc(); public native PIXAA nalloc(int nalloc);       /* number of Pixa ptrs allocated     */
    public native PIXA pixa(int i); public native PIXAA pixa(int i, PIXA pixa);
    @MemberGetter public native @Cast("Pixa**") PointerPointer pixa();         /* array of ptrs to pixa             */
    public native BOXA boxa(); public native PIXAA boxa(BOXA boxa);         /* array of boxes                    */
}


/*-------------------------------------------------------------------------*
 *                    Basic rectangle and rectangle arrays                 *
 *-------------------------------------------------------------------------*/
@Name("Box") public static class BOX extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public BOX() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public BOX(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public BOX(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public BOX position(int position) {
        return (BOX)super.position(position);
    }

    public native @Cast("l_int32") int x(); public native BOX x(int x);
    public native @Cast("l_int32") int y(); public native BOX y(int y);
    public native @Cast("l_int32") int w(); public native BOX w(int w);
    public native @Cast("l_int32") int h(); public native BOX h(int h);
    public native @Cast("l_uint32") int refcount(); public native BOX refcount(int refcount);      /* reference count (1 if no clones)  */

}

@Name("Boxa") public static class BOXA extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public BOXA() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public BOXA(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public BOXA(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public BOXA position(int position) {
        return (BOXA)super.position(position);
    }

    public native @Cast("l_int32") int n(); public native BOXA n(int n);             /* number of box in ptr array        */
    public native @Cast("l_int32") int nalloc(); public native BOXA nalloc(int nalloc);        /* number of box ptrs allocated      */
    public native @Cast("l_uint32") int refcount(); public native BOXA refcount(int refcount);      /* reference count (1 if no clones)  */
    public native BOX box(int i); public native BOXA box(int i, BOX box);
    @MemberGetter public native @Cast("Box**") PointerPointer box();           /* box ptr array                     */
}

@Name("Boxaa") public static class BOXAA extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public BOXAA() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public BOXAA(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public BOXAA(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public BOXAA position(int position) {
        return (BOXAA)super.position(position);
    }

    public native @Cast("l_int32") int n(); public native BOXAA n(int n);             /* number of boxa in ptr array       */
    public native @Cast("l_int32") int nalloc(); public native BOXAA nalloc(int nalloc);        /* number of boxa ptrs allocated     */
    public native BOXA boxa(int i); public native BOXAA boxa(int i, BOXA boxa);
    @MemberGetter public native @Cast("Boxa**") PointerPointer boxa();          /* boxa ptr array                    */
}


/*-------------------------------------------------------------------------*
 *                               Array of points                           *
 *-------------------------------------------------------------------------*/
public static final int PTA_VERSION_NUMBER =      1;

@Name("Pta") public static class PTA extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public PTA() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public PTA(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public PTA(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public PTA position(int position) {
        return (PTA)super.position(position);
    }

    public native @Cast("l_int32") int n(); public native PTA n(int n);             /* actual number of pts              */
    public native @Cast("l_int32") int nalloc(); public native PTA nalloc(int nalloc);        /* size of allocated arrays          */
    public native @Cast("l_uint32") int refcount(); public native PTA refcount(int refcount);      /* reference count (1 if no clones)  */
    public native @Cast("l_float32*") FloatPointer x(); public native PTA x(FloatPointer x);
    public native @Cast("l_float32*") FloatPointer y(); public native PTA y(FloatPointer y);         /* arrays of floats                  */
}


/*-------------------------------------------------------------------------*
 *                              Array of Pta                               *
 *-------------------------------------------------------------------------*/


/*-------------------------------------------------------------------------*
 *                       Pix accumulator container                         *
 *-------------------------------------------------------------------------*/


/*-------------------------------------------------------------------------*
 *                              Pix tiling                                 *
 *-------------------------------------------------------------------------*/


/*-------------------------------------------------------------------------*
 *                       FPix: pix with float array                        *
 *-------------------------------------------------------------------------*/
public static final int FPIX_VERSION_NUMBER =      2;


/*-------------------------------------------------------------------------*
 *                       DPix: pix with double array                       *
 *-------------------------------------------------------------------------*/
public static final int DPIX_VERSION_NUMBER =      2;


/*-------------------------------------------------------------------------*
 *                        PixComp: compressed pix                          *
 *-------------------------------------------------------------------------*/


/*-------------------------------------------------------------------------*
 *                     PixaComp: array of compressed pix                   *
 *-------------------------------------------------------------------------*/
public static final int PIXACOMP_VERSION_NUMBER =      2;


/*-------------------------------------------------------------------------*
 *                         Access and storage flags                        *
 *-------------------------------------------------------------------------*/
/*
 *  For Pix, Box, Pta and Numa, there are 3 standard methods for handling
 *  the retrieval or insertion of a struct:
 *     (1) direct insertion (Don't do this if there is another handle
 *                           somewhere to this same struct!)
 *     (2) copy (Always safe, sets up a refcount of 1 on the new object.
 *               Can be undesirable if very large, such as an image or
 *               an array of images.)
 *     (3) clone (Makes another handle to the same struct, and bumps the
 *                refcount up by 1.  Safe to do unless you're changing
 *                data through one of the handles but don't want those
 *                changes to be seen by the other handle.)
 *
 *  For Pixa and Boxa, which are structs that hold an array of clonable
 *  structs, there is an additional method:
 *     (4) copy-clone (Makes a new higher-level struct with a refcount
 *                     of 1, but clones all the structs in the array.)
 *
 *  Unlike the other structs, when retrieving a string from an Sarray,
 *  you are allowed to get a handle without a copy or clone (i.e., that
 *  you don't own!).  You must not free or insert such a string!
 *  Specifically, for an Sarray, the copyflag for retrieval is either:
 *         TRUE (or 1 or L_COPY)
 *  or
 *         FALSE (or 0 or L_NOCOPY)
 *  For insertion, the copyflag is either:
 *         TRUE (or 1 or L_COPY)
 *  or
 *         FALSE (or 0 or L_INSERT)
 *  Note that L_COPY is always 1, and L_INSERT and L_NOCOPY are always 0.
 */
/** enum  */
public static final int
    L_INSERT = 0,     /* stuff it in; no copy, clone or copy-clone    */
    L_COPY = 1,       /* make/use a copy of the object                */
    L_CLONE = 2,      /* make/use clone (ref count) of the object     */
    L_COPY_CLONE = 3;  /* make a new object and fill with with clones  */
                      /* of each object in the array(s)               */
@MemberGetter public static native @Cast("const l_int32") int L_NOCOPY();  /* copyflag value in sarrayGetString() */


/*--------------------------------------------------------------------------*
 *                              Sort flags                                  *
 *--------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_SHELL_SORT = 1,             /* use shell sort                         */
    L_BIN_SORT = 2;                /* use bin sort                           */

/** enum  */
public static final int
    L_SORT_INCREASING = 1,        /* sort in increasing order               */
    L_SORT_DECREASING = 2;         /* sort in decreasing order               */

/** enum  */
public static final int
    L_SORT_BY_X = 1,              /* sort box or c.c. by left edge location  */
    L_SORT_BY_Y = 2,              /* sort box or c.c. by top edge location   */
    L_SORT_BY_RIGHT = 3,          /* sort box or c.c. by right edge location */
    L_SORT_BY_BOT = 4,            /* sort box or c.c. by bot edge location   */
    L_SORT_BY_WIDTH = 5,          /* sort box or c.c. by width               */
    L_SORT_BY_HEIGHT = 6,         /* sort box or c.c. by height              */
    L_SORT_BY_MIN_DIMENSION = 7,  /* sort box or c.c. by min dimension       */
    L_SORT_BY_MAX_DIMENSION = 8,  /* sort box or c.c. by max dimension       */
    L_SORT_BY_PERIMETER = 9,      /* sort box or c.c. by perimeter           */
    L_SORT_BY_AREA = 10,          /* sort box or c.c. by area                */
    L_SORT_BY_ASPECT_RATIO = 11;   /* sort box or c.c. by width/height ratio  */


/*-------------------------------------------------------------------------*
 *                             Blend flags                                 *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_BLEND_WITH_INVERSE = 1,     /* add some of src inverse to itself     */
    L_BLEND_TO_WHITE = 2,         /* shift src colors towards white        */
    L_BLEND_TO_BLACK = 3,         /* shift src colors towards black        */
    L_BLEND_GRAY = 4,             /* blend src directly with blender       */
    L_BLEND_GRAY_WITH_INVERSE = 5; /* add amount of src inverse to itself,  */
                                  /* based on blender pix value            */

/** enum  */
public static final int
    L_PAINT_LIGHT = 1,            /* colorize non-black pixels             */
    L_PAINT_DARK = 2;              /* colorize non-white pixels             */


/*-------------------------------------------------------------------------*
 *                        Graphics pixel setting                           *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_SET_PIXELS = 1,             /* set all bits in each pixel to 1       */
    L_CLEAR_PIXELS = 2,           /* set all bits in each pixel to 0       */
    L_FLIP_PIXELS = 3;             /* flip all bits in each pixel           */


/*-------------------------------------------------------------------------*
 *                           Size filter flags                             *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_SELECT_WIDTH = 1,           /* width must satisfy constraint         */
    L_SELECT_HEIGHT = 2,          /* height must satisfy constraint        */
    L_SELECT_IF_EITHER = 3,       /* either width or height can satisfy    */
    L_SELECT_IF_BOTH = 4;          /* both width and height must satisfy    */

/** enum  */
public static final int
    L_SELECT_IF_LT = 1,           /* save if value is less than threshold  */
    L_SELECT_IF_GT = 2,           /* save if value is more than threshold  */
    L_SELECT_IF_LTE = 3,          /* save if value is <= to the threshold  */
    L_SELECT_IF_GTE = 4;           /* save if value is >= to the threshold  */


/*-------------------------------------------------------------------------*
 *                     Color component selection flags                     *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_SELECT_RED = 1,             /* use red component                     */
    L_SELECT_GREEN = 2,           /* use green component                   */
    L_SELECT_BLUE = 3,            /* use blue component                    */
    L_SELECT_MIN = 4,             /* use min color component               */
    L_SELECT_MAX = 5,             /* use max color component               */
    L_SELECT_AVERAGE = 6;          /* use average of color components       */


/*-------------------------------------------------------------------------*
 *                         16-bit conversion flags                         *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_LS_BYTE = 0,                /* use LSB                               */
    L_MS_BYTE = 1,                /* use MSB                               */
    L_CLIP_TO_255 = 2;             /* use max(val, 255)                     */


/*-------------------------------------------------------------------------*
 *                        Rotate and shear flags                           *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_ROTATE_AREA_MAP = 1,       /* use area map rotation, if possible     */
    L_ROTATE_SHEAR = 2,          /* use shear rotation                     */
    L_ROTATE_SAMPLING = 3;        /* use sampling                           */

/** enum  */
public static final int
    L_BRING_IN_WHITE = 1,        /* bring in white pixels from the outside */
    L_BRING_IN_BLACK = 2;         /* bring in black pixels from the outside */

/** enum  */
public static final int
    L_SHEAR_ABOUT_CORNER = 1,    /* shear image about UL corner            */
    L_SHEAR_ABOUT_CENTER = 2;     /* shear image about center               */


/*-------------------------------------------------------------------------*
 *                     Affine transform order flags                        *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_TR_SC_RO = 1,              /* translate, scale, rotate               */
    L_SC_RO_TR = 2,              /* scale, rotate, translate               */
    L_RO_TR_SC = 3,              /* rotate, translate, scale               */
    L_TR_RO_SC = 4,              /* translate, rotate, scale               */
    L_RO_SC_TR = 5,              /* rotate, scale, translate               */
    L_SC_TR_RO = 6;               /* scale, translate, rotate               */


/*-------------------------------------------------------------------------*
 *                       Grayscale filling flags                           *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_FILL_WHITE = 1,           /* fill white pixels (e.g, in fg map)      */
    L_FILL_BLACK = 2;            /* fill black pixels (e.g., in bg map)     */


/*-------------------------------------------------------------------------*
 *                   Flags for setting to white or black                   *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_SET_WHITE = 1,           /* set pixels to white                      */
    L_SET_BLACK = 2;            /* set pixels to black                      */


/*-------------------------------------------------------------------------*
 *                  Flags for getting white or black value                 *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_GET_WHITE_VAL = 1,       /* get white pixel value                    */
    L_GET_BLACK_VAL = 2;        /* get black pixel value                    */


/*-------------------------------------------------------------------------*
 *                  Flags for 8 bit and 16 bit pixel sums                  *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_WHITE_IS_MAX = 1,   /* white pixels are 0xff or 0xffff; black are 0  */
    L_BLACK_IS_MAX = 2;    /* black pixels are 0xff or 0xffff; white are 0  */


/*-------------------------------------------------------------------------*
 *                           Dither parameters                             *
 *         If within this grayscale distance from black or white,          *
 *         do not propagate excess or deficit to neighboring pixels.       *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    DEFAULT_CLIP_LOWER_1 = 10,   /* dist to black with no prop; 1 bpp      */
    DEFAULT_CLIP_UPPER_1 = 10,   /* dist to black with no prop; 1 bpp      */
    DEFAULT_CLIP_LOWER_2 = 5,    /* dist to black with no prop; 2 bpp      */
    DEFAULT_CLIP_UPPER_2 = 5;     /* dist to black with no prop; 2 bpp      */


/*-------------------------------------------------------------------------*
 *                             Distance flags                              *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_MANHATTAN_DISTANCE = 1,    /* L1 distance (e.g., in color space)     */
    L_EUCLIDEAN_DISTANCE = 2;     /* L2 distance                            */


/*-------------------------------------------------------------------------*
 *                         Statistical measures                            *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_MEAN_ABSVAL = 1,           /* average of abs values                  */
    L_MEDIAN_VAL = 2,            /* median value of set                    */
    L_MODE_VAL = 3,              /* mode value of set                      */
    L_MODE_COUNT = 4,            /* mode count of set                      */
    L_ROOT_MEAN_SQUARE = 5,      /* rms of values                          */
    L_STANDARD_DEVIATION = 6,    /* standard deviation from mean           */
    L_VARIANCE = 7;               /* variance of values                     */


/*-------------------------------------------------------------------------*
 *                          Set selection flags                            *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_CHOOSE_CONSECUTIVE = 1,    /* select 'n' consecutive                 */
    L_CHOOSE_SKIP_BY = 2;         /* select at intervals of 'n'             */


/*-------------------------------------------------------------------------*
 *                         Text orientation flags                          *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_TEXT_ORIENT_UNKNOWN = 0,   /* low confidence on text orientation     */
    L_TEXT_ORIENT_UP = 1,        /* portrait, text rightside-up            */
    L_TEXT_ORIENT_LEFT = 2,      /* landscape, text up to left             */
    L_TEXT_ORIENT_DOWN = 3,      /* portrait, text upside-down             */
    L_TEXT_ORIENT_RIGHT = 4;      /* landscape, text up to right            */


/*-------------------------------------------------------------------------*
 *                         Edge orientation flags                          *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_HORIZONTAL_EDGES = 0,     /* filters for horizontal edges            */
    L_VERTICAL_EDGES = 1,       /* filters for vertical edges              */
    L_ALL_EDGES = 2;             /* filters for all edges                   */


/*-------------------------------------------------------------------------*
 *                         Line orientation flags                          *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_HORIZONTAL_LINE = 0,     /* horizontal line                          */
    L_POS_SLOPE_LINE = 1,      /* 45 degree line with positive slope       */
    L_VERTICAL_LINE = 2,       /* vertical line                            */
    L_NEG_SLOPE_LINE = 3,      /* 45 degree line with negative slope       */
    L_OBLIQUE_LINE = 4;         /* neither horizontal nor vertical */


/*-------------------------------------------------------------------------*
 *                           Scan direction flags                          *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_FROM_LEFT = 0,           /* scan from left                           */
    L_FROM_RIGHT = 1,          /* scan from right                          */
    L_FROM_TOP = 2,            /* scan from top                            */
    L_FROM_BOT = 3,            /* scan from bottom                         */
    L_SCAN_NEGATIVE = 4,       /* scan in negative direction               */
    L_SCAN_POSITIVE = 5,       /* scan in positive direction               */
    L_SCAN_BOTH = 6,           /* scan in both directions                  */
    L_SCAN_HORIZONTAL = 7,     /* horizontal scan (direction unimportant)  */
    L_SCAN_VERTICAL = 8;        /* vertical scan (direction unimportant)    */


/*-------------------------------------------------------------------------*
 *                Box size adjustment and location flags                   *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_ADJUST_SKIP = 0,             /* do not adjust                        */
    L_ADJUST_LEFT = 1,             /* adjust left edge                     */
    L_ADJUST_RIGHT = 2,            /* adjust right edge                    */
    L_ADJUST_LEFT_AND_RIGHT = 3,   /* adjust both left and right edges     */
    L_ADJUST_TOP = 4,              /* adjust top edge                      */
    L_ADJUST_BOT = 5,              /* adjust bottom edge                   */
    L_ADJUST_TOP_AND_BOT = 6,      /* adjust both top and bottom edges     */
    L_ADJUST_CHOOSE_MIN = 7,       /* choose the min median value          */
    L_ADJUST_CHOOSE_MAX = 8,       /* choose the max median value          */
    L_SET_LEFT = 9,                /* set left side to a given value       */
    L_SET_RIGHT = 10,              /* set right side to a given value      */
    L_SET_TOP = 11,                /* set top side to a given value        */
    L_SET_BOT = 12,                /* set bottom side to a given value     */
    L_GET_LEFT = 13,               /* get left side location               */
    L_GET_RIGHT = 14,              /* get right side location              */
    L_GET_TOP = 15,                /* get top side location                */
    L_GET_BOT = 16;                 /* get bottom side location             */


/*-------------------------------------------------------------------------*
 *          Flags for selecting box boundaries from two choices            *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_USE_MINSIZE = 1,             /* use boundaries giving min size       */
    L_USE_MAXSIZE = 2,             /* use boundaries giving max size       */
    L_SUB_ON_BIG_DIFF = 3,         /* substitute boundary if big abs diff  */
    L_USE_CAPPED_MIN = 4,          /* substitute boundary with capped min  */
    L_USE_CAPPED_MAX = 5;           /* substitute boundary with capped max  */

/*-------------------------------------------------------------------------*
 *              Handling overlapping bounding boxes in boxa                *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_COMBINE = 1,           /* resize to bounding region; remove smaller  */
    L_REMOVE_SMALL = 2;       /* only remove smaller                        */

/*-------------------------------------------------------------------------*
 *                    Flags for replacing invalid boxes                    *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_USE_ALL_BOXES = 1,         /* consider all boxes in the sequence     */
    L_USE_SAME_PARITY_BOXES = 2;  /* consider boxes with the same parity    */

/*-------------------------------------------------------------------------*
 *                            Horizontal warp                              *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_WARP_TO_LEFT = 1,      /* increasing stretch or contraction to left  */
    L_WARP_TO_RIGHT = 2;      /* increasing stretch or contraction to right */

/** enum  */
public static final int
    L_LINEAR_WARP = 1,       /* stretch or contraction grows linearly      */
    L_QUADRATIC_WARP = 2;     /* stretch or contraction grows quadratically */


/*-------------------------------------------------------------------------*
 *                      Pixel selection for resampling                     *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_INTERPOLATED = 1,      /* linear interpolation from src pixels       */
    L_SAMPLED = 2;            /* nearest src pixel sampling only            */


/*-------------------------------------------------------------------------*
 *                             Thinning flags                              *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_THIN_FG = 1,               /* thin foreground of 1 bpp image         */
    L_THIN_BG = 2;                /* thin background of 1 bpp image         */


/*-------------------------------------------------------------------------*
 *                            Runlength flags                              *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_HORIZONTAL_RUNS = 0,     /* determine runlengths of horizontal runs  */
    L_VERTICAL_RUNS = 1;        /* determine runlengths of vertical runs    */


/*-------------------------------------------------------------------------*
 *                          Edge filter flags                              *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_SOBEL_EDGE = 1,          /* Sobel edge filter                        */
    L_TWO_SIDED_EDGE = 2;       /* Two-sided edge filter                    */


/*-------------------------------------------------------------------------*
 *             Subpixel color component ordering in LCD display            *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_SUBPIXEL_ORDER_RGB = 1,   /* sensor order left-to-right RGB          */
    L_SUBPIXEL_ORDER_BGR = 2,   /* sensor order left-to-right BGR          */
    L_SUBPIXEL_ORDER_VRGB = 3,  /* sensor order top-to-bottom RGB          */
    L_SUBPIXEL_ORDER_VBGR = 4;   /* sensor order top-to-bottom BGR          */


/*-------------------------------------------------------------------------*
 *                          HSV histogram flags                            *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_HS_HISTO = 1,            /* Use hue-saturation histogram             */
    L_HV_HISTO = 2,            /* Use hue-value histogram                  */
    L_SV_HISTO = 3;             /* Use saturation-value histogram           */


/*-------------------------------------------------------------------------*
 *                    Region flags (inclusion, exclusion)                  *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_INCLUDE_REGION = 1,      /* Use hue-saturation histogram             */
    L_EXCLUDE_REGION = 2;       /* Use hue-value histogram                  */


/*-------------------------------------------------------------------------*
 *                    Flags for adding text to a pix                       *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_ADD_ABOVE = 1,           /* Add text above the image                 */
    L_ADD_BELOW = 2,           /* Add text below the image                 */
    L_ADD_LEFT = 3,            /* Add text to the left of the image        */
    L_ADD_RIGHT = 4,           /* Add text to the right of the image       */
    L_ADD_AT_TOP = 5,          /* Add text over the top of the image       */
    L_ADD_AT_BOT = 6,          /* Add text over the bottom of the image    */
    L_ADD_AT_LEFT = 7,         /* Add text over left side of the image     */
    L_ADD_AT_RIGHT = 8;         /* Add text over right side of the image    */


/*-------------------------------------------------------------------------*
 *                       Flags for plotting on a pix                       *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_PLOT_AT_TOP = 1,         /* Plot horizontally at top                 */
    L_PLOT_AT_MID_HORIZ = 2,   /* Plot horizontally at middle              */
    L_PLOT_AT_BOT = 3,         /* Plot horizontally at bottom              */
    L_PLOT_AT_LEFT = 4,        /* Plot vertically at left                  */
    L_PLOT_AT_MID_VERT = 5,    /* Plot vertically at middle                */
    L_PLOT_AT_RIGHT = 6;        /* Plot vertically at right                 */


/*-------------------------------------------------------------------------*
 *                   Flags for selecting display program                   *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_DISPLAY_WITH_XZGV = 1,    /* Use xzgv with pixDisplay()              */
    L_DISPLAY_WITH_XLI = 2,     /* Use xli with pixDisplay()               */
    L_DISPLAY_WITH_XV = 3,      /* Use xv with pixDisplay()                */
    L_DISPLAY_WITH_IV = 4,      /* Use irfvanview (win) with pixDisplay()  */
    L_DISPLAY_WITH_OPEN = 5;     /* Use open (apple) with pixDisplay()      */

/*-------------------------------------------------------------------------*
 *    Flag(s) used in the 'special' pix field for non-default operations   *
 *      - 0 is default                                                     *
 *      - 10-19 are reserved for zlib compression in png write             *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_NO_CHROMA_SAMPLING_JPEG = 1;     /* Write full resolution chroma      */


/*-------------------------------------------------------------------------*
 *          Handling negative values in conversion to unsigned int         *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_CLIP_TO_ZERO = 1,        /* Clip negative values to 0                */
    L_TAKE_ABSVAL = 2;          /* Convert to positive using L_ABS()        */


/*-------------------------------------------------------------------------*
 *                        Relative to zero flags                           *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_LESS_THAN_ZERO = 1,      /* Choose values less than zero             */
    L_EQUAL_TO_ZERO = 2,       /* Choose values equal to zero              */
    L_GREATER_THAN_ZERO = 3;    /* Choose values greater than zero          */


/*-------------------------------------------------------------------------*
 *         Flags for adding or removing traling slash from string          *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_ADD_TRAIL_SLASH = 1,     /* Add trailing slash to string             */
    L_REMOVE_TRAIL_SLASH = 2;   /* Remove trailing slash from string        */


// #endif  /* LEPTONICA_PIX_H */


// Parsed from leptonica/allheaders_min.h

/*====================================================================*
 -  Copyright (C) 2001 Leptonica.  All rights reserved.
 -
 -  Redistribution and use in source and binary forms, with or without
 -  modification, are permitted provided that the following conditions
 -  are met:
 -  1. Redistributions of source code must retain the above copyright
 -     notice, this list of conditions and the following disclaimer.
 -  2. Redistributions in binary form must reproduce the above
 -     copyright notice, this list of conditions and the following
 -     disclaimer in the documentation and/or other materials
 -     provided with the distribution.
 -
 -  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 -  ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 -  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 -  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL ANY
 -  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 -  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 -  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 -  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 -  OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 -  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 -  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *====================================================================*/

// #ifndef  LEPTONICA_ALLHEADERS_H
// #define  LEPTONICA_ALLHEADERS_H


public static final int LIBLEPT_MAJOR_VERSION =   1;
public static final int LIBLEPT_MINOR_VERSION =   72;

// #include "alltypes.h"

// #ifndef NO_PROTOS
/*
 *  These prototypes were autogen'd by xtractprotos, v. 1.5
 */
// #ifdef __cplusplus
// #endif  /* __cplusplus */

public static native BOXA boxaRotate( BOXA boxas, @Cast("l_float32") float xc, @Cast("l_float32") float yc, @Cast("l_float32") float angle );
public static native PIX pixReduceRankBinaryCascade( PIX pixs, @Cast("l_int32") int level1, @Cast("l_int32") int level2, @Cast("l_int32") int level3, @Cast("l_int32") int level4 );
public static native BOX boxCreate( @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int w, @Cast("l_int32") int h );
public static native void boxDestroy( @Cast("BOX**") PointerPointer pbox );
public static native void boxDestroy( @ByPtrPtr BOX pbox );
public static native @Cast("l_int32") int boxGetGeometry( BOX box, @Cast("l_int32*") IntPointer px, @Cast("l_int32*") IntPointer py, @Cast("l_int32*") IntPointer pw, @Cast("l_int32*") IntPointer ph );
public static native @Cast("l_int32") int boxGetGeometry( BOX box, @Cast("l_int32*") IntBuffer px, @Cast("l_int32*") IntBuffer py, @Cast("l_int32*") IntBuffer pw, @Cast("l_int32*") IntBuffer ph );
public static native @Cast("l_int32") int boxGetGeometry( BOX box, @Cast("l_int32*") int[] px, @Cast("l_int32*") int[] py, @Cast("l_int32*") int[] pw, @Cast("l_int32*") int[] ph );
public static native BOXA boxaCreate( @Cast("l_int32") int n );
public static native void boxaDestroy( @Cast("BOXA**") PointerPointer pboxa );
public static native void boxaDestroy( @ByPtrPtr BOXA pboxa );
public static native @Cast("l_int32") int boxaAddBox( BOXA boxa, BOX box, @Cast("l_int32") int copyflag );
public static native @Cast("l_int32") int boxaGetCount( BOXA boxa );
public static native BOX boxaGetBox( BOXA boxa, @Cast("l_int32") int index, @Cast("l_int32") int accessflag );
public static native @Cast("l_int32") int boxaGetBoxGeometry( BOXA boxa, @Cast("l_int32") int index, @Cast("l_int32*") IntPointer px, @Cast("l_int32*") IntPointer py, @Cast("l_int32*") IntPointer pw, @Cast("l_int32*") IntPointer ph );
public static native @Cast("l_int32") int boxaGetBoxGeometry( BOXA boxa, @Cast("l_int32") int index, @Cast("l_int32*") IntBuffer px, @Cast("l_int32*") IntBuffer py, @Cast("l_int32*") IntBuffer pw, @Cast("l_int32*") IntBuffer ph );
public static native @Cast("l_int32") int boxaGetBoxGeometry( BOXA boxa, @Cast("l_int32") int index, @Cast("l_int32*") int[] px, @Cast("l_int32*") int[] py, @Cast("l_int32*") int[] pw, @Cast("l_int32*") int[] ph );
public static native @Cast("l_int32") int boxaReplaceBox( BOXA boxa, @Cast("l_int32") int index, BOX box );
public static native BOXAA boxaaCreate( @Cast("l_int32") int n );
public static native void boxaaDestroy( @Cast("BOXAA**") PointerPointer pbaa );
public static native void boxaaDestroy( @ByPtrPtr BOXAA pbaa );
public static native @Cast("l_int32") int boxaaAddBoxa( BOXAA baa, BOXA ba, @Cast("l_int32") int copyflag );
public static native @Cast("l_int32") int boxaaRemoveBoxa( BOXAA baa, @Cast("l_int32") int index );
public static native @Cast("l_int32") int boxaaAddBox( BOXAA baa, @Cast("l_int32") int index, BOX box, @Cast("l_int32") int accessflag );
public static native BOX boxBoundingRegion( BOX box1, BOX box2 );
public static native PIX pixDrawBoxa( PIX pixs, BOXA boxa, @Cast("l_int32") int width, @Cast("l_uint32") int val );
public static native @Cast("l_int32") int boxaGetExtent( BOXA boxa, @Cast("l_int32*") IntPointer pw, @Cast("l_int32*") IntPointer ph, @Cast("BOX**") PointerPointer pbox );
public static native @Cast("l_int32") int boxaGetExtent( BOXA boxa, @Cast("l_int32*") IntPointer pw, @Cast("l_int32*") IntPointer ph, @ByPtrPtr BOX pbox );
public static native @Cast("l_int32") int boxaGetExtent( BOXA boxa, @Cast("l_int32*") IntBuffer pw, @Cast("l_int32*") IntBuffer ph, @ByPtrPtr BOX pbox );
public static native @Cast("l_int32") int boxaGetExtent( BOXA boxa, @Cast("l_int32*") int[] pw, @Cast("l_int32*") int[] ph, @ByPtrPtr BOX pbox );
public static native PIXCMAP pixcmapCreate( @Cast("l_int32") int depth );
public static native @Cast("l_int32") int pixcmapAddColor( PIXCMAP cmap, @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval );
public static native BOXA pixConnComp( PIX pixs, @Cast("PIXA**") PointerPointer ppixa, @Cast("l_int32") int connectivity );
public static native BOXA pixConnComp( PIX pixs, @ByPtrPtr PIXA ppixa, @Cast("l_int32") int connectivity );
public static native @Cast("l_int32") int pixCountConnComp( PIX pixs, @Cast("l_int32") int connectivity, @Cast("l_int32*") IntPointer pcount );
public static native @Cast("l_int32") int pixCountConnComp( PIX pixs, @Cast("l_int32") int connectivity, @Cast("l_int32*") IntBuffer pcount );
public static native @Cast("l_int32") int pixCountConnComp( PIX pixs, @Cast("l_int32") int connectivity, @Cast("l_int32*") int[] pcount );
public static native @Cast("l_int32") int pixSeedfill( PIX pixs, L_STACK stack, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int connectivity );
public static native PIX pixBlockconv( PIX pix, @Cast("l_int32") int wc, @Cast("l_int32") int hc );
public static native @Cast("l_int32") int pixRenderBox( PIX pix, BOX box, @Cast("l_int32") int width, @Cast("l_int32") int op );
public static native @Cast("l_int32") int pixRenderBoxArb( PIX pix, BOX box, @Cast("l_int32") int width, @Cast("l_uint8") byte rval, @Cast("l_uint8") byte gval, @Cast("l_uint8") byte bval );
public static native @Cast("l_int32") int pixRenderPolyline( PIX pix, PTA ptas, @Cast("l_int32") int width, @Cast("l_int32") int op, @Cast("l_int32") int closeflag );
public static native @Cast("l_int32") int pixRenderPolylineArb( PIX pix, PTA ptas, @Cast("l_int32") int width, @Cast("l_uint8") byte rval, @Cast("l_uint8") byte gval, @Cast("l_uint8") byte bval, @Cast("l_int32") int closeflag );
public static native PIX pixErodeGray( PIX pixs, @Cast("l_int32") int hsize, @Cast("l_int32") int vsize );
public static native PIX pixThresholdToBinary( PIX pixs, @Cast("l_int32") int thresh );
public static native @Cast("l_int32") int pixWriteJpeg( @Cast("const char*") BytePointer filename, PIX pix, @Cast("l_int32") int quality, @Cast("l_int32") int progressive );
public static native @Cast("l_int32") int pixWriteJpeg( String filename, PIX pix, @Cast("l_int32") int quality, @Cast("l_int32") int progressive );
public static native @Cast("char*") BytePointer getImagelibVersions(  );
public static native PIX pixDilate( PIX pixd, PIX pixs, SEL sel );
public static native PIX pixErode( PIX pixd, PIX pixs, SEL sel );
public static native PIX pixOpen( PIX pixd, PIX pixs, SEL sel );
public static native PIX pixClose( PIX pixd, PIX pixs, SEL sel );
public static native PIX pixDilateBrick( PIX pixd, PIX pixs, @Cast("l_int32") int hsize, @Cast("l_int32") int vsize );
public static native PIX pixErodeBrick( PIX pixd, PIX pixs, @Cast("l_int32") int hsize, @Cast("l_int32") int vsize );
public static native PIX pixOpenBrick( PIX pixd, PIX pixs, @Cast("l_int32") int hsize, @Cast("l_int32") int vsize );
public static native PIX pixCloseBrick( PIX pixd, PIX pixs, @Cast("l_int32") int hsize, @Cast("l_int32") int vsize );
public static native PIXA pixaMorphSequenceByRegion( PIX pixs, PIXA pixam, @Cast("const char*") BytePointer sequence, @Cast("l_int32") int minw, @Cast("l_int32") int minh );
public static native PIXA pixaMorphSequenceByRegion( PIX pixs, PIXA pixam, String sequence, @Cast("l_int32") int minw, @Cast("l_int32") int minh );
public static native PIX pixMorphCompSequence( PIX pixs, @Cast("const char*") BytePointer sequence, @Cast("l_int32") int dispsep );
public static native PIX pixMorphCompSequence( PIX pixs, String sequence, @Cast("l_int32") int dispsep );
public static native void numaDestroy( @Cast("NUMA**") PointerPointer pna );
public static native void numaDestroy( @ByPtrPtr NUMA pna );
public static native @Cast("l_int32") int numaGetCount( NUMA na );
public static native @Cast("l_int32") int numaGetIValue( NUMA na, @Cast("l_int32") int index, @Cast("l_int32*") IntPointer pival );
public static native @Cast("l_int32") int numaGetIValue( NUMA na, @Cast("l_int32") int index, @Cast("l_int32*") IntBuffer pival );
public static native @Cast("l_int32") int numaGetIValue( NUMA na, @Cast("l_int32") int index, @Cast("l_int32*") int[] pival );
public static native PIX pixGenHalftoneMask( PIX pixs, @Cast("PIX**") PointerPointer ppixtext, @Cast("l_int32*") IntPointer phtfound, @Cast("l_int32") int debug );
public static native PIX pixGenHalftoneMask( PIX pixs, @ByPtrPtr PIX ppixtext, @Cast("l_int32*") IntPointer phtfound, @Cast("l_int32") int debug );
public static native PIX pixGenHalftoneMask( PIX pixs, @ByPtrPtr PIX ppixtext, @Cast("l_int32*") IntBuffer phtfound, @Cast("l_int32") int debug );
public static native PIX pixGenHalftoneMask( PIX pixs, @ByPtrPtr PIX ppixtext, @Cast("l_int32*") int[] phtfound, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int pixaConvertToPdf( PIXA pixa, @Cast("l_int32") int res, @Cast("l_float32") float scalefactor, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("const char*") BytePointer title, @Cast("const char*") BytePointer fileout );
public static native @Cast("l_int32") int pixaConvertToPdf( PIXA pixa, @Cast("l_int32") int res, @Cast("l_float32") float scalefactor, @Cast("l_int32") int type, @Cast("l_int32") int quality, String title, String fileout );
public static native @Cast("l_int32") int l_generateCIDataForPdf( @Cast("const char*") BytePointer fname, PIX pix, @Cast("l_int32") int quality, @Cast("L_COMP_DATA**") PointerPointer pcid );
public static native @Cast("l_int32") int l_generateCIDataForPdf( @Cast("const char*") BytePointer fname, PIX pix, @Cast("l_int32") int quality, @ByPtrPtr L_COMP_DATA pcid );
public static native @Cast("l_int32") int l_generateCIDataForPdf( String fname, PIX pix, @Cast("l_int32") int quality, @ByPtrPtr L_COMP_DATA pcid );
public static native @Cast("l_int32") int l_generateCIData( @Cast("const char*") BytePointer fname, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("l_int32") int ascii85, @Cast("L_COMP_DATA**") PointerPointer pcid );
public static native @Cast("l_int32") int l_generateCIData( @Cast("const char*") BytePointer fname, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("l_int32") int ascii85, @ByPtrPtr L_COMP_DATA pcid );
public static native @Cast("l_int32") int l_generateCIData( String fname, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("l_int32") int ascii85, @ByPtrPtr L_COMP_DATA pcid );
public static native @Cast("l_int32") int pixGenerateCIData( PIX pixs, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("l_int32") int ascii85, @Cast("L_COMP_DATA**") PointerPointer pcid );
public static native @Cast("l_int32") int pixGenerateCIData( PIX pixs, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("l_int32") int ascii85, @ByPtrPtr L_COMP_DATA pcid );
public static native void l_CIDataDestroy( @Cast("L_COMP_DATA**") PointerPointer pcid );
public static native void l_CIDataDestroy( @ByPtrPtr L_COMP_DATA pcid );
public static native PIX pixCreate( @Cast("l_int32") int width, @Cast("l_int32") int height, @Cast("l_int32") int depth );
public static native PIX pixCreateTemplate( PIX pixs );
public static native PIX pixCreateHeader( @Cast("l_int32") int width, @Cast("l_int32") int height, @Cast("l_int32") int depth );
public static native PIX pixClone( PIX pixs );
public static native void pixDestroy( @Cast("PIX**") PointerPointer ppix );
public static native void pixDestroy( @ByPtrPtr PIX ppix );
public static native PIX pixCopy( PIX pixd, PIX pixs );
public static native @Cast("l_int32") int pixSizesEqual( PIX pix1, PIX pix2 );
public static native @Cast("l_int32") int pixGetWidth( PIX pix );
public static native @Cast("l_int32") int pixGetHeight( PIX pix );
public static native @Cast("l_int32") int pixGetDepth( PIX pix );
public static native @Cast("l_int32") int pixGetDimensions( PIX pix, @Cast("l_int32*") IntPointer pw, @Cast("l_int32*") IntPointer ph, @Cast("l_int32*") IntPointer pd );
public static native @Cast("l_int32") int pixGetDimensions( PIX pix, @Cast("l_int32*") IntBuffer pw, @Cast("l_int32*") IntBuffer ph, @Cast("l_int32*") IntBuffer pd );
public static native @Cast("l_int32") int pixGetDimensions( PIX pix, @Cast("l_int32*") int[] pw, @Cast("l_int32*") int[] ph, @Cast("l_int32*") int[] pd );
public static native @Cast("l_int32") int pixGetSpp( PIX pix );
public static native @Cast("l_int32") int pixSetSpp( PIX pix, @Cast("l_int32") int spp );
public static native @Cast("l_int32") int pixGetWpl( PIX pix );
public static native @Cast("l_int32") int pixGetXRes( PIX pix );
public static native @Cast("l_int32") int pixSetXRes( PIX pix, @Cast("l_int32") int res );
public static native @Cast("l_int32") int pixGetYRes( PIX pix );
public static native @Cast("l_int32") int pixSetYRes( PIX pix, @Cast("l_int32") int res );
public static native @Cast("l_int32") int pixSetInputFormat( PIX pix, @Cast("l_int32") int informat );
public static native @Cast("l_int32") int pixSetText( PIX pix, @Cast("const char*") BytePointer textstring );
public static native @Cast("l_int32") int pixSetText( PIX pix, String textstring );
public static native PIXCMAP pixGetColormap( PIX pix );
public static native @Cast("l_int32") int pixSetColormap( PIX pix, PIXCMAP colormap );
public static native @Cast("l_uint32*") IntPointer pixGetData( PIX pix );
public static native @Cast("l_int32") int pixSetData( PIX pix, @Cast("l_uint32*") IntPointer data );
public static native @Cast("l_int32") int pixSetData( PIX pix, @Cast("l_uint32*") IntBuffer data );
public static native @Cast("l_int32") int pixSetData( PIX pix, @Cast("l_uint32*") int[] data );
public static native @Cast("l_int32") int pixGetPixel( PIX pix, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_uint32*") IntPointer pval );
public static native @Cast("l_int32") int pixGetPixel( PIX pix, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_uint32*") IntBuffer pval );
public static native @Cast("l_int32") int pixGetPixel( PIX pix, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_uint32*") int[] pval );
public static native @Cast("l_int32") int pixSetPixel( PIX pix, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_uint32") int val );
public static native @Cast("l_int32") int pixSetAll( PIX pix );
public static native @Cast("l_int32") int pixSetAllArbitrary( PIX pix, @Cast("l_uint32") int val );
public static native @Cast("l_int32") int pixClearInRect( PIX pix, BOX box );
public static native @Cast("l_int32") int pixSetInRect( PIX pix, BOX box );
public static native @Cast("l_int32") int pixSetInRectArbitrary( PIX pix, BOX box, @Cast("l_uint32") int val );
public static native @Cast("l_int32") int pixSetOrClearBorder( PIX pixs, @Cast("l_int32") int left, @Cast("l_int32") int right, @Cast("l_int32") int top, @Cast("l_int32") int bot, @Cast("l_int32") int op );
public static native PIX pixAddBorder( PIX pixs, @Cast("l_int32") int npix, @Cast("l_uint32") int val );
public static native @Cast("l_int32") int composeRGBPixel( @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_uint32*") IntPointer ppixel );
public static native @Cast("l_int32") int composeRGBPixel( @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_uint32*") IntBuffer ppixel );
public static native @Cast("l_int32") int composeRGBPixel( @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_uint32*") int[] ppixel );
public static native @Cast("l_int32") int pixEndianByteSwap( PIX pixs );
public static native @Cast("l_int32") int pixEndianTwoByteSwap( PIX pixs );
public static native @Cast("l_int32") int pixSetMasked( PIX pixd, PIX pixm, @Cast("l_uint32") int val );
public static native PIX pixInvert( PIX pixd, PIX pixs );
public static native PIX pixOr( PIX pixd, PIX pixs1, PIX pixs2 );
public static native PIX pixAnd( PIX pixd, PIX pixs1, PIX pixs2 );
public static native PIX pixXor( PIX pixd, PIX pixs1, PIX pixs2 );
public static native PIX pixSubtract( PIX pixd, PIX pixs1, PIX pixs2 );
public static native @Cast("l_int32") int pixZero( PIX pix, @Cast("l_int32*") IntPointer pempty );
public static native @Cast("l_int32") int pixZero( PIX pix, @Cast("l_int32*") IntBuffer pempty );
public static native @Cast("l_int32") int pixZero( PIX pix, @Cast("l_int32*") int[] pempty );
public static native @Cast("l_int32") int pixForegroundFraction( PIX pix, @Cast("l_float32*") FloatPointer pfract );
public static native @Cast("l_int32") int pixForegroundFraction( PIX pix, @Cast("l_float32*") FloatBuffer pfract );
public static native @Cast("l_int32") int pixForegroundFraction( PIX pix, @Cast("l_float32*") float[] pfract );
public static native @Cast("l_int32") int pixCountPixels( PIX pix, @Cast("l_int32*") IntPointer pcount, @Cast("l_int32*") IntPointer tab8 );
public static native @Cast("l_int32") int pixCountPixels( PIX pix, @Cast("l_int32*") IntBuffer pcount, @Cast("l_int32*") IntBuffer tab8 );
public static native @Cast("l_int32") int pixCountPixels( PIX pix, @Cast("l_int32*") int[] pcount, @Cast("l_int32*") int[] tab8 );
public static native NUMA pixCountPixelsByRow( PIX pix, @Cast("l_int32*") IntPointer tab8 );
public static native NUMA pixCountPixelsByRow( PIX pix, @Cast("l_int32*") IntBuffer tab8 );
public static native NUMA pixCountPixelsByRow( PIX pix, @Cast("l_int32*") int[] tab8 );
public static native @Cast("l_int32") int pixCountPixelsInRow( PIX pix, @Cast("l_int32") int row, @Cast("l_int32*") IntPointer pcount, @Cast("l_int32*") IntPointer tab8 );
public static native @Cast("l_int32") int pixCountPixelsInRow( PIX pix, @Cast("l_int32") int row, @Cast("l_int32*") IntBuffer pcount, @Cast("l_int32*") IntBuffer tab8 );
public static native @Cast("l_int32") int pixCountPixelsInRow( PIX pix, @Cast("l_int32") int row, @Cast("l_int32*") int[] pcount, @Cast("l_int32*") int[] tab8 );
public static native PIX pixClipRectangle( PIX pixs, BOX box, @Cast("BOX**") PointerPointer pboxc );
public static native PIX pixClipRectangle( PIX pixs, BOX box, @ByPtrPtr BOX pboxc );
public static native @Cast("l_int32") int pixClipBoxToForeground( PIX pixs, BOX boxs, @Cast("PIX**") PointerPointer ppixd, @Cast("BOX**") PointerPointer pboxd );
public static native @Cast("l_int32") int pixClipBoxToForeground( PIX pixs, BOX boxs, @ByPtrPtr PIX ppixd, @ByPtrPtr BOX pboxd );
public static native PIXA pixaCreate( @Cast("l_int32") int n );
public static native void pixaDestroy( @Cast("PIXA**") PointerPointer ppixa );
public static native void pixaDestroy( @ByPtrPtr PIXA ppixa );
public static native @Cast("l_int32") int pixaAddPix( PIXA pixa, PIX pix, @Cast("l_int32") int copyflag );
public static native @Cast("l_int32") int pixaAddBox( PIXA pixa, BOX box, @Cast("l_int32") int copyflag );
public static native @Cast("l_int32") int pixaGetCount( PIXA pixa );
public static native PIX pixaGetPix( PIXA pixa, @Cast("l_int32") int index, @Cast("l_int32") int accesstype );
public static native @Cast("l_int32") int pixaReplacePix( PIXA pixa, @Cast("l_int32") int index, PIX pix, BOX box );
public static native @Cast("l_int32") int pixaInsertPix( PIXA pixa, @Cast("l_int32") int index, PIX pixs, BOX box );
public static native @Cast("l_int32") int pixaRemovePix( PIXA pixa, @Cast("l_int32") int index );
public static native PIXAA pixaaCreate( @Cast("l_int32") int n );
public static native void pixaaDestroy( @Cast("PIXAA**") PointerPointer ppaa );
public static native void pixaaDestroy( @ByPtrPtr PIXAA ppaa );
public static native @Cast("l_int32") int pixaaAddPixa( PIXAA paa, PIXA pixa, @Cast("l_int32") int copyflag );
public static native @Cast("l_int32") int pixaaAddPix( PIXAA paa, @Cast("l_int32") int index, PIX pix, BOX box, @Cast("l_int32") int copyflag );
public static native @Cast("l_int32") int pixaaAddBox( PIXAA paa, BOX box, @Cast("l_int32") int copyflag );
public static native PIX pixaDisplay( PIXA pixa, @Cast("l_int32") int w, @Cast("l_int32") int h );
public static native PIX pixaDisplayTiled( PIXA pixa, @Cast("l_int32") int maxwidth, @Cast("l_int32") int background, @Cast("l_int32") int spacing );
public static native PIX pixaDisplayTiledInRows( PIXA pixa, @Cast("l_int32") int outdepth, @Cast("l_int32") int maxwidth, @Cast("l_float32") float scalefactor, @Cast("l_int32") int background, @Cast("l_int32") int spacing, @Cast("l_int32") int border );
public static native PIX pixRemoveColormap( PIX pixs, @Cast("l_int32") int type );
public static native PIX pixConvertRGBToLuminance( PIX pixs );
public static native PIX pixConvertTo8( PIX pixs, @Cast("l_int32") int cmapflag );
public static native PIX pixConvertTo32( PIX pixs );
public static native @Cast("l_int32") int pixWriteStreamPng( @Cast("FILE*") Pointer fp, PIX pix, @Cast("l_float32") float gamma );
public static native PTA ptaCreate( @Cast("l_int32") int n );
public static native void ptaDestroy( @Cast("PTA**") PointerPointer ppta );
public static native void ptaDestroy( @ByPtrPtr PTA ppta );
public static native @Cast("l_int32") int ptaAddPt( PTA pta, @Cast("l_float32") float x, @Cast("l_float32") float y );
public static native PIX pixRead( @Cast("const char*") BytePointer filename );
public static native PIX pixRead( String filename );
public static native PIX pixReadStream( @Cast("FILE*") Pointer fp, @Cast("l_int32") int hint );
public static native @Cast("l_int32") int findFileFormat( @Cast("const char*") BytePointer filename, @Cast("l_int32*") IntPointer pformat );
public static native @Cast("l_int32") int findFileFormat( String filename, @Cast("l_int32*") IntBuffer pformat );
public static native @Cast("l_int32") int findFileFormat( @Cast("const char*") BytePointer filename, @Cast("l_int32*") int[] pformat );
public static native @Cast("l_int32") int findFileFormat( String filename, @Cast("l_int32*") IntPointer pformat );
public static native @Cast("l_int32") int findFileFormat( @Cast("const char*") BytePointer filename, @Cast("l_int32*") IntBuffer pformat );
public static native @Cast("l_int32") int findFileFormat( String filename, @Cast("l_int32*") int[] pformat );
public static native @Cast("l_int32") int findFileFormatBuffer( @Cast("const l_uint8*") BytePointer buf, @Cast("l_int32*") IntPointer pformat );
public static native @Cast("l_int32") int findFileFormatBuffer( @Cast("const l_uint8*") ByteBuffer buf, @Cast("l_int32*") IntBuffer pformat );
public static native @Cast("l_int32") int findFileFormatBuffer( @Cast("const l_uint8*") byte[] buf, @Cast("l_int32*") int[] pformat );
public static native PIX pixReadMem( @Cast("const l_uint8*") BytePointer data, @Cast("size_t") long size );
public static native PIX pixReadMem( @Cast("const l_uint8*") ByteBuffer data, @Cast("size_t") long size );
public static native PIX pixReadMem( @Cast("const l_uint8*") byte[] data, @Cast("size_t") long size );
public static native @Cast("l_int32") int pixRasterop( PIX pixd, @Cast("l_int32") int dx, @Cast("l_int32") int dy, @Cast("l_int32") int dw, @Cast("l_int32") int dh, @Cast("l_int32") int op, PIX pixs, @Cast("l_int32") int sx, @Cast("l_int32") int sy );
public static native PIX pixRotate( PIX pixs, @Cast("l_float32") float angle, @Cast("l_int32") int type, @Cast("l_int32") int incolor, @Cast("l_int32") int width, @Cast("l_int32") int height );
public static native PIX pixRotate90( PIX pixs, @Cast("l_int32") int direction );
public static native PIX pixFlipLR( PIX pixd, PIX pixs );
public static native PIX pixFlipTB( PIX pixd, PIX pixs );
public static native PIX pixScale( PIX pixs, @Cast("l_float32") float scalex, @Cast("l_float32") float scaley );
public static native PIX pixScaleToSize( PIX pixs, @Cast("l_int32") int wd, @Cast("l_int32") int hd );
public static native PIX pixExpandReplicate( PIX pixs, @Cast("l_int32") int factor );
public static native PIX pixSeedfillBinary( PIX pixd, PIX pixs, PIX pixm, @Cast("l_int32") int connectivity );
public static native PIX pixDistanceFunction( PIX pixs, @Cast("l_int32") int connectivity, @Cast("l_int32") int outdepth, @Cast("l_int32") int boundcond );
public static native SEL selCreate( @Cast("l_int32") int height, @Cast("l_int32") int width, @Cast("const char*") BytePointer name );
public static native SEL selCreate( @Cast("l_int32") int height, @Cast("l_int32") int width, String name );
public static native SEL selCreateBrick( @Cast("l_int32") int h, @Cast("l_int32") int w, @Cast("l_int32") int cy, @Cast("l_int32") int cx, @Cast("l_int32") int type );
public static native @Cast("l_int32") int selFindMaxTranslations( SEL sel, @Cast("l_int32*") IntPointer pxp, @Cast("l_int32*") IntPointer pyp, @Cast("l_int32*") IntPointer pxn, @Cast("l_int32*") IntPointer pyn );
public static native @Cast("l_int32") int selFindMaxTranslations( SEL sel, @Cast("l_int32*") IntBuffer pxp, @Cast("l_int32*") IntBuffer pyp, @Cast("l_int32*") IntBuffer pxn, @Cast("l_int32*") IntBuffer pyn );
public static native @Cast("l_int32") int selFindMaxTranslations( SEL sel, @Cast("l_int32*") int[] pxp, @Cast("l_int32*") int[] pyp, @Cast("l_int32*") int[] pxn, @Cast("l_int32*") int[] pyn );
public static native PIX pixDeskew( PIX pixs, @Cast("l_int32") int redsearch );
public static native PIX pixReadTiff( @Cast("const char*") BytePointer filename, @Cast("l_int32") int n );
public static native PIX pixReadTiff( String filename, @Cast("l_int32") int n );
public static native PIX pixReadStreamTiff( @Cast("FILE*") Pointer fp, @Cast("l_int32") int n );
public static native @Cast("l_int32") int pixWriteTiff( @Cast("const char*") BytePointer filename, PIX pix, @Cast("l_int32") int comptype, @Cast("const char*") BytePointer modestring );
public static native @Cast("l_int32") int pixWriteTiff( String filename, PIX pix, @Cast("l_int32") int comptype, String modestring );
public static native PIX pixReadMemTiff( @Cast("const l_uint8*") BytePointer cdata, @Cast("size_t") long size, @Cast("l_int32") int n );
public static native PIX pixReadMemTiff( @Cast("const l_uint8*") ByteBuffer cdata, @Cast("size_t") long size, @Cast("l_int32") int n );
public static native PIX pixReadMemTiff( @Cast("const l_uint8*") byte[] cdata, @Cast("size_t") long size, @Cast("l_int32") int n );
public static native @Cast("l_int32") int stringLength( @Cast("const char*") BytePointer src, @Cast("size_t") long size );
public static native @Cast("l_int32") int stringLength( String src, @Cast("size_t") long size );
public static native Pointer reallocNew( @Cast("void**") PointerPointer pindata, @Cast("l_int32") int oldsize, @Cast("l_int32") int newsize );
public static native Pointer reallocNew( @Cast("void**") @ByPtrPtr Pointer pindata, @Cast("l_int32") int oldsize, @Cast("l_int32") int newsize );
public static native @Cast("FILE*") Pointer fopenReadStream( @Cast("const char*") BytePointer filename );
public static native @Cast("FILE*") Pointer fopenReadStream( String filename );
public static native void lept_free( Pointer ptr );
public static native @Cast("char*") BytePointer getLeptonicaVersion(  );
public static native void l_getCurrentTime( @Cast("l_int32*") IntPointer sec, @Cast("l_int32*") IntPointer usec );
public static native void l_getCurrentTime( @Cast("l_int32*") IntBuffer sec, @Cast("l_int32*") IntBuffer usec );
public static native void l_getCurrentTime( @Cast("l_int32*") int[] sec, @Cast("l_int32*") int[] usec );
public static native @Cast("char*") BytePointer l_getFormattedDate(  );
public static native @Cast("l_int32") int pixWrite( @Cast("const char*") BytePointer filename, PIX pix, @Cast("l_int32") int format );
public static native @Cast("l_int32") int pixWrite( String filename, PIX pix, @Cast("l_int32") int format );
public static native @Cast("l_int32") int pixWriteStream( @Cast("FILE*") Pointer fp, PIX pix, @Cast("l_int32") int format );
public static native @Cast("l_int32") int pixWriteMem( @Cast("l_uint8**") PointerPointer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix, @Cast("l_int32") int format );
public static native @Cast("l_int32") int pixWriteMem( @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix, @Cast("l_int32") int format );
public static native @Cast("l_int32") int pixWriteMem( @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix, @Cast("l_int32") int format );
public static native @Cast("l_int32") int pixWriteMem( @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer psize, PIX pix, @Cast("l_int32") int format );
public static native @Cast("l_int32") int pixDisplay( PIX pixs, @Cast("l_int32") int x, @Cast("l_int32") int y );
public static native @Cast("l_int32") int pixDisplayWrite( PIX pixs, @Cast("l_int32") int reduction );
public static native @Cast("l_uint8*") BytePointer zlibCompress( @Cast("l_uint8*") BytePointer datain, @Cast("size_t") long nin, @Cast("size_t*") SizeTPointer pnout );
public static native @Cast("l_uint8*") ByteBuffer zlibCompress( @Cast("l_uint8*") ByteBuffer datain, @Cast("size_t") long nin, @Cast("size_t*") SizeTPointer pnout );
public static native @Cast("l_uint8*") byte[] zlibCompress( @Cast("l_uint8*") byte[] datain, @Cast("size_t") long nin, @Cast("size_t*") SizeTPointer pnout );
public static native PIX pixReadStreamBmp( @Cast("FILE*") Pointer fp );
public static native @Cast("l_int32") int pixWriteStreamBmp( @Cast("FILE*") Pointer fp, PIX pix );
public static native PIX pixReadMemBmp( @Cast("const l_uint8*") BytePointer cdata, @Cast("size_t") long size );
public static native PIX pixReadMemBmp( @Cast("const l_uint8*") ByteBuffer cdata, @Cast("size_t") long size );
public static native PIX pixReadMemBmp( @Cast("const l_uint8*") byte[] cdata, @Cast("size_t") long size );
public static native @Cast("l_int32") int pixWriteMemBmp( @Cast("l_uint8**") PointerPointer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix );
public static native @Cast("l_int32") int pixWriteMemBmp( @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix );
public static native @Cast("l_int32") int pixWriteMemBmp( @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix );
public static native @Cast("l_int32") int pixWriteMemBmp( @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer psize, PIX pix );
public static native PIX pixReadStreamGif( @Cast("FILE*") Pointer fp );
public static native @Cast("l_int32") int pixWriteStreamGif( @Cast("FILE*") Pointer fp, PIX pix );
public static native PIX pixReadMemGif( @Cast("const l_uint8*") BytePointer cdata, @Cast("size_t") long size );
public static native PIX pixReadMemGif( @Cast("const l_uint8*") ByteBuffer cdata, @Cast("size_t") long size );
public static native PIX pixReadMemGif( @Cast("const l_uint8*") byte[] cdata, @Cast("size_t") long size );
public static native @Cast("l_int32") int pixWriteMemGif( @Cast("l_uint8**") PointerPointer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix );
public static native @Cast("l_int32") int pixWriteMemGif( @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix );
public static native @Cast("l_int32") int pixWriteMemGif( @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix );
public static native @Cast("l_int32") int pixWriteMemGif( @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer psize, PIX pix );
public static native PIX pixReadJpeg( @Cast("const char*") BytePointer filename, @Cast("l_int32") int cmapflag, @Cast("l_int32") int reduction, @Cast("l_int32*") IntPointer pnwarn, @Cast("l_int32") int hint );
public static native PIX pixReadJpeg( String filename, @Cast("l_int32") int cmapflag, @Cast("l_int32") int reduction, @Cast("l_int32*") IntBuffer pnwarn, @Cast("l_int32") int hint );
public static native PIX pixReadJpeg( @Cast("const char*") BytePointer filename, @Cast("l_int32") int cmapflag, @Cast("l_int32") int reduction, @Cast("l_int32*") int[] pnwarn, @Cast("l_int32") int hint );
public static native PIX pixReadJpeg( String filename, @Cast("l_int32") int cmapflag, @Cast("l_int32") int reduction, @Cast("l_int32*") IntPointer pnwarn, @Cast("l_int32") int hint );
public static native PIX pixReadJpeg( @Cast("const char*") BytePointer filename, @Cast("l_int32") int cmapflag, @Cast("l_int32") int reduction, @Cast("l_int32*") IntBuffer pnwarn, @Cast("l_int32") int hint );
public static native PIX pixReadJpeg( String filename, @Cast("l_int32") int cmapflag, @Cast("l_int32") int reduction, @Cast("l_int32*") int[] pnwarn, @Cast("l_int32") int hint );
public static native @Cast("l_int32") int pixWriteStreamJpeg( @Cast("FILE*") Pointer fp, PIX pixs, @Cast("l_int32") int quality, @Cast("l_int32") int progressive );
public static native PIX pixReadMemJpeg( @Cast("const l_uint8*") BytePointer data, @Cast("size_t") long size, @Cast("l_int32") int cmflag, @Cast("l_int32") int reduction, @Cast("l_int32*") IntPointer pnwarn, @Cast("l_int32") int hint );
public static native PIX pixReadMemJpeg( @Cast("const l_uint8*") ByteBuffer data, @Cast("size_t") long size, @Cast("l_int32") int cmflag, @Cast("l_int32") int reduction, @Cast("l_int32*") IntBuffer pnwarn, @Cast("l_int32") int hint );
public static native PIX pixReadMemJpeg( @Cast("const l_uint8*") byte[] data, @Cast("size_t") long size, @Cast("l_int32") int cmflag, @Cast("l_int32") int reduction, @Cast("l_int32*") int[] pnwarn, @Cast("l_int32") int hint );
public static native PIX pixReadStreamPng( @Cast("FILE*") Pointer fp );
public static native PIX pixReadMemPng( @Cast("const l_uint8*") BytePointer cdata, @Cast("size_t") long size );
public static native PIX pixReadMemPng( @Cast("const l_uint8*") ByteBuffer cdata, @Cast("size_t") long size );
public static native PIX pixReadMemPng( @Cast("const l_uint8*") byte[] cdata, @Cast("size_t") long size );
public static native @Cast("l_int32") int pixWriteMemPng( @Cast("l_uint8**") PointerPointer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix, @Cast("l_float32") float gamma );
public static native @Cast("l_int32") int pixWriteMemPng( @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix, @Cast("l_float32") float gamma );
public static native @Cast("l_int32") int pixWriteMemPng( @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix, @Cast("l_float32") float gamma );
public static native @Cast("l_int32") int pixWriteMemPng( @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer psize, PIX pix, @Cast("l_float32") float gamma );
public static native @Cast("l_int32") int pixWritePng( @Cast("const char*") BytePointer filename, PIX pix, @Cast("l_float32") float gamma );
public static native @Cast("l_int32") int pixWritePng( String filename, PIX pix, @Cast("l_float32") float gamma );
public static native PIX pixReadWithHint( @Cast("const char*") BytePointer filename, @Cast("l_int32") int hint );
public static native PIX pixReadWithHint( String filename, @Cast("l_int32") int hint );
public static native @Cast("l_int32") int pixWriteMemTiff( @Cast("l_uint8**") PointerPointer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix, @Cast("l_int32") int comptype );
public static native @Cast("l_int32") int pixWriteMemTiff( @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix, @Cast("l_int32") int comptype );
public static native @Cast("l_int32") int pixWriteMemTiff( @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix, @Cast("l_int32") int comptype );
public static native @Cast("l_int32") int pixWriteMemTiff( @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer psize, PIX pix, @Cast("l_int32") int comptype );
public static native PIX pixReadStreamWebP( @Cast("FILE*") Pointer fp );
public static native PIX pixReadMemWebP( @Cast("const l_uint8*") BytePointer filedata, @Cast("size_t") long filesize );
public static native PIX pixReadMemWebP( @Cast("const l_uint8*") ByteBuffer filedata, @Cast("size_t") long filesize );
public static native PIX pixReadMemWebP( @Cast("const l_uint8*") byte[] filedata, @Cast("size_t") long filesize );
public static native @Cast("l_int32") int pixWriteWebP( @Cast("const char*") BytePointer filename, PIX pixs, @Cast("l_int32") int quality, @Cast("l_int32") int lossless );
public static native @Cast("l_int32") int pixWriteWebP( String filename, PIX pixs, @Cast("l_int32") int quality, @Cast("l_int32") int lossless );
public static native @Cast("l_int32") int pixWriteStreamWebP( @Cast("FILE*") Pointer fp, PIX pixs, @Cast("l_int32") int quality, @Cast("l_int32") int lossless );
public static native @Cast("l_int32") int pixWriteMemWebP( @Cast("l_uint8**") PointerPointer pencdata, @Cast("size_t*") SizeTPointer pencsize, PIX pixs, @Cast("l_int32") int quality, @Cast("l_int32") int lossless );
public static native @Cast("l_int32") int pixWriteMemWebP( @Cast("l_uint8**") @ByPtrPtr BytePointer pencdata, @Cast("size_t*") SizeTPointer pencsize, PIX pixs, @Cast("l_int32") int quality, @Cast("l_int32") int lossless );
public static native @Cast("l_int32") int pixWriteMemWebP( @Cast("l_uint8**") @ByPtrPtr ByteBuffer pencdata, @Cast("size_t*") SizeTPointer pencsize, PIX pixs, @Cast("l_int32") int quality, @Cast("l_int32") int lossless );
public static native @Cast("l_int32") int pixWriteMemWebP( @Cast("l_uint8**") @ByPtrPtr byte[] pencdata, @Cast("size_t*") SizeTPointer pencsize, PIX pixs, @Cast("l_int32") int quality, @Cast("l_int32") int lossless );

// #ifdef __cplusplus
// #endif  /* __cplusplus */
// #endif /* NO_PROTOS */


// #endif /* LEPTONICA_ALLHEADERS_H */



}
