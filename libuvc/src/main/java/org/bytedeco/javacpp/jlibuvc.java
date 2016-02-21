// Targeted by JavaCPP version 1.2-SNAPSHOT

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
@Namespace @Name("void") @Opaque public static class L_TIMER extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public L_TIMER() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public L_TIMER(Pointer p) { super(p); }
}
@Name("L_WallTimer") public static class L_WALLTIMER extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public L_WALLTIMER() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public L_WALLTIMER(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public L_WALLTIMER(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public L_WALLTIMER position(int position) {
        return (L_WALLTIMER)super.position(position);
    }

    public native @Cast("l_int32") int start_sec(); public native L_WALLTIMER start_sec(int start_sec);
    public native @Cast("l_int32") int start_usec(); public native L_WALLTIMER start_usec(int start_usec);
    public native @Cast("l_int32") int stop_sec(); public native L_WALLTIMER stop_sec(int stop_sec);
    public native @Cast("l_int32") int stop_usec(); public native L_WALLTIMER stop_usec(int stop_usec);
}


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
    public NUMA() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public NUMA(int size) { super((Pointer)null); allocateArray(size); }
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
@Name("Numaa") public static class NUMAA extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public NUMAA() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public NUMAA(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public NUMAA(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public NUMAA position(int position) {
        return (NUMAA)super.position(position);
    }

    public native @Cast("l_int32") int nalloc(); public native NUMAA nalloc(int nalloc);    /* size of allocated ptr array          */
    public native @Cast("l_int32") int n(); public native NUMAA n(int n);         /* number of Numa saved                 */
    public native NUMA numa(int i); public native NUMAA numa(int i, NUMA numa);
    @MemberGetter public native @Cast("Numa**") PointerPointer numa();      /* array of Numa                        */
}


    /* Sparse 2-dimensional array of number arrays */
@Name("Numa2d") public static class NUMA2D extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public NUMA2D() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public NUMA2D(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public NUMA2D(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public NUMA2D position(int position) {
        return (NUMA2D)super.position(position);
    }

    public native @Cast("l_int32") int nrows(); public native NUMA2D nrows(int nrows);      /* number of rows allocated for ptr array  */
    public native @Cast("l_int32") int ncols(); public native NUMA2D ncols(int ncols);      /* number of cols allocated for ptr array  */
    public native @Cast("l_int32") int initsize(); public native NUMA2D initsize(int initsize);   /* initial size of each numa that is made  */
    public native @Cast("Numa**") PointerPointer numa(int i); public native NUMA2D numa(int i, PointerPointer numa);
    @MemberGetter public native @Cast("Numa***") PointerPointer numa();       /* 2D array of Numa                        */
}


    /* A hash table of Numas */
@Name("NumaHash") public static class NUMAHASH extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public NUMAHASH() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public NUMAHASH(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public NUMAHASH(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public NUMAHASH position(int position) {
        return (NUMAHASH)super.position(position);
    }

    public native @Cast("l_int32") int nbuckets(); public native NUMAHASH nbuckets(int nbuckets);
    public native @Cast("l_int32") int initsize(); public native NUMAHASH initsize(int initsize);   /* initial size of each numa that is made  */
    public native NUMA numa(int i); public native NUMAHASH numa(int i, NUMA numa);
    @MemberGetter public native @Cast("Numa**") PointerPointer numa();
}


public static final int DNA_VERSION_NUMBER =     1;

    /* Double number array: an array of doubles */
@Name("L_Dna") public static class L_DNA extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public L_DNA() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public L_DNA(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public L_DNA(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public L_DNA position(int position) {
        return (L_DNA)super.position(position);
    }

    public native @Cast("l_int32") int nalloc(); public native L_DNA nalloc(int nalloc);    /* size of allocated number array      */
    public native @Cast("l_int32") int n(); public native L_DNA n(int n);         /* number of numbers saved             */
    public native @Cast("l_int32") int refcount(); public native L_DNA refcount(int refcount);  /* reference count (1 if no clones)    */
    public native @Cast("l_float64") double startx(); public native L_DNA startx(double startx);    /* x value assigned to array[0]        */
    public native @Cast("l_float64") double delx(); public native L_DNA delx(double delx);      /* change in x value as i --> i + 1    */
    public native @Cast("l_float64*") DoublePointer array(); public native L_DNA array(DoublePointer array);     /* number array                        */
}


    /* Array of double number arrays */
@Name("L_Dnaa") public static class L_DNAA extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public L_DNAA() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public L_DNAA(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public L_DNAA(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public L_DNAA position(int position) {
        return (L_DNAA)super.position(position);
    }

    public native @Cast("l_int32") int nalloc(); public native L_DNAA nalloc(int nalloc);    /* size of allocated ptr array          */
    public native @Cast("l_int32") int n(); public native L_DNAA n(int n);         /* number of L_Dna saved                */
    public native L_DNA dna(int i); public native L_DNAA dna(int i, L_DNA dna);
    @MemberGetter public native @Cast("L_Dna**") PointerPointer dna();       /* array of L_Dna                       */
}


public static final int SARRAY_VERSION_NUMBER =     1;

    /* String array: an array of C strings */
@Name("Sarray") public static class SARRAY extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public SARRAY() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public SARRAY(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public SARRAY(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public SARRAY position(int position) {
        return (SARRAY)super.position(position);
    }

    public native @Cast("l_int32") int nalloc(); public native SARRAY nalloc(int nalloc);    /* size of allocated ptr array         */
    public native @Cast("l_int32") int n(); public native SARRAY n(int n);         /* number of strings allocated         */
    public native @Cast("l_int32") int refcount(); public native SARRAY refcount(int refcount);  /* reference count (1 if no clones)    */
    public native @Cast("char*") BytePointer array(int i); public native SARRAY array(int i, BytePointer array);
    @MemberGetter public native @Cast("char**") PointerPointer array();     /* string array                        */
}


    /* Byte array (analogous to C++ "string") */
@Name("L_Bytea") public static class L_BYTEA extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public L_BYTEA() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public L_BYTEA(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public L_BYTEA(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public L_BYTEA position(int position) {
        return (L_BYTEA)super.position(position);
    }

    public native @Cast("size_t") long nalloc(); public native L_BYTEA nalloc(long nalloc);    /* number of bytes allocated in data array  */
    public native @Cast("size_t") long size(); public native L_BYTEA size(long size);      /* number of bytes presently used           */
    public native @Cast("l_int32") int refcount(); public native L_BYTEA refcount(int refcount);  /* reference count (1 if no clones)         */
    public native @Cast("l_uint8*") BytePointer data(); public native L_BYTEA data(BytePointer data);      /* data array                               */
}


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


// Parsed from leptonica/bbuffer.h

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

// #ifndef  LEPTONICA_BBUFFER_H
// #define  LEPTONICA_BBUFFER_H

/*
 *  bbuffer.h
 *
 *      Expandable byte buffer for reading data in from memory and
 *      writing data out to other memory.
 *
 *      This implements a queue of bytes, so data read in is put
 *      on the "back" of the queue (i.e., the end of the byte array)
 *      and data written out is taken from the "front" of the queue
 *      (i.e., from an index marker "nwritten" that is initially set at
 *      the beginning of the array.)  As usual with expandable
 *      arrays, we keep the size of the allocated array and the
 *      number of bytes that have been read into the array.
 *
 *      For implementation details, see bbuffer.c.
 */

@Name("ByteBuffer") public static class BBUFFER extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public BBUFFER() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public BBUFFER(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public BBUFFER(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public BBUFFER position(int position) {
        return (BBUFFER)super.position(position);
    }

    public native @Cast("l_int32") int nalloc(); public native BBUFFER nalloc(int nalloc);       /* size of allocated byte array            */
    public native @Cast("l_int32") int n(); public native BBUFFER n(int n);            /* number of bytes read into to the array  */
    public native @Cast("l_int32") int nwritten(); public native BBUFFER nwritten(int nwritten);     /* number of bytes written from the array  */
    public native @Cast("l_uint8*") BytePointer array(); public native BBUFFER array(BytePointer array);        /* byte array                              */
}


// #endif  /* LEPTONICA_BBUFFER_H */


// Parsed from leptonica/heap.h

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

// #ifndef  LEPTONICA_HEAP_H
// #define  LEPTONICA_HEAP_H

/*
 *  heap.h
 *
 *      Expandable priority queue configured as a heap for arbitrary void* data
 *
 *      The L_Heap is used to implement a priority queue.  The elements
 *      in the heap are ordered in either increasing or decreasing key value.
 *      The key is a float field 'keyval' that is required to be
 *      contained in the elements of the queue.
 * 
 *      The heap is a simple binary tree with the following constraints:
 *         - the key of each node is >= the keys of the two children
 *         - the tree is complete, meaning that each level (1, 2, 4, ...)
 *           is filled and the last level is filled from left to right
 *
 *      The tree structure is implicit in the queue array, with the
 *      array elements numbered as a breadth-first search of the tree
 *      from left to right.  It is thus guaranteed that the largest
 *      (or smallest) key belongs to the first element in the array.
 *
 *      Heap sort is used to sort the array.  Once an array has been
 *      sorted as a heap, it is convenient to use it as a priority queue,
 *      because the min (or max) elements are always at the root of
 *      the tree (element 0), and once removed, the heap can be
 *      resorted in not more than log[n] steps, where n is the number
 *      of elements on the heap.  Likewise, if an arbitrary element is
 *      added to the end of the array A, the sorted heap can be restored
 *      in not more than log[n] steps.
 *
 *      A L_Heap differs from a L_Queue in that the elements in the former
 *      are sorted by a key.  Internally, the array is maintained
 *      as a queue, with a pointer to the end of the array.  The
 *      head of the array always remains at array[0].  The array is
 *      maintained (sorted) as a heap.  When an item is removed from
 *      the head, the last item takes its place (thus reducing the
 *      array length by 1), and this is followed by array element
 *      swaps to restore the heap property.   When an item is added,
 *      it goes at the end of the array, and is swapped up to restore
 *      the heap.  If the ptr array is full, adding another item causes
 *      the ptr array size to double.
 *      
 *      For further implementation details, see heap.c.
 */

@Name("L_Heap") public static class L_HEAP extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public L_HEAP() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public L_HEAP(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public L_HEAP(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public L_HEAP position(int position) {
        return (L_HEAP)super.position(position);
    }

    public native @Cast("l_int32") int nalloc(); public native L_HEAP nalloc(int nalloc);      /* size of allocated ptr array                 */
    public native @Cast("l_int32") int n(); public native L_HEAP n(int n);           /* number of elements stored in the heap       */
    public native Pointer array(int i); public native L_HEAP array(int i, Pointer array);
    @MemberGetter public native @Cast("void**") PointerPointer array();       /* ptr array                                   */
    public native @Cast("l_int32") int direction(); public native L_HEAP direction(int direction);   /* L_SORT_INCREASING or L_SORT_DECREASING      */
}


// #endif  /* LEPTONICA_HEAP_H */


// Parsed from leptonica/list.h

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


// #ifndef  LEPTONICA_LIST_H
// #define  LEPTONICA_LIST_H

/*
 *   list.h
 *
 *       Cell for double-linked lists
 *
 *       This allows composition of a list of cells with 
 *           prev, next and data pointers.  Generic data
 *           structures hang on the list cell data pointers.
 *
 *       The list is not circular because that would add much
 *           complexity in traversing the list under general
 *           conditions where list cells can be added and removed.
 *           The only disadvantage of not having the head point to
 *           the last cell is that the list must be traversed to
 *           find its tail.  However, this traversal is fast, and
 *           the listRemoveFromTail() function updates the tail
 *           so there is no searching overhead with repeated use.
 *
 *       The list macros are used to run through a list, and their
 *       use is encouraged.  They are invoked, e.g., as
 *
 *             DLLIST  *head, *elem;
 *             ...
 *             L_BEGIN_LIST_FORWARD(head, elem)
 *                 <do something with elem and/or elem->data >
 *             L_END_LIST
 *
 */

@Name("DoubleLinkedList") public static class DLLIST extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public DLLIST() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public DLLIST(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DLLIST(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public DLLIST position(int position) {
        return (DLLIST)super.position(position);
    }

    public native DLLIST prev(); public native DLLIST prev(DLLIST prev);
    public native DLLIST next(); public native DLLIST next(DLLIST next);
    public native Pointer data(); public native DLLIST data(Pointer data);
}


    /*  Simple list traverse macros */
// #define L_BEGIN_LIST_FORWARD(head, element)
//         {
//         DLLIST   *_leptvar_nextelem_;
//         for ((element) = (head); (element); (element) = _leptvar_nextelem_) {
//             _leptvar_nextelem_ = (element)->next;


// #define L_BEGIN_LIST_REVERSE(tail, element)
//         {
//         DLLIST   *_leptvar_prevelem_;
//         for ((element) = (tail); (element); (element) = _leptvar_prevelem_) {
//             _leptvar_prevelem_ = (element)->prev;


// #define L_END_LIST    }}


// #endif  /* LEPTONICA_LIST_H */


// Parsed from leptonica/ptra.h

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

// #ifndef  LEPTONICA_PTRA_H
// #define  LEPTONICA_PTRA_H

/*
 *  Contains the following structs:
 *      struct L_Ptra
 *      struct L_Ptraa
 *
 *  Contains definitions for:
 *      L_Ptra compaction flags for removal
 *      L_Ptra shifting flags for insert
 *      L_Ptraa accessor flags
 */


/*------------------------------------------------------------------------* 
 *                     Generic Ptr Array Structs                          *
 *------------------------------------------------------------------------*/

    /* Generic pointer array */
@Name("L_Ptra") public static class L_PTRA extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public L_PTRA() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public L_PTRA(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public L_PTRA(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public L_PTRA position(int position) {
        return (L_PTRA)super.position(position);
    }

    public native @Cast("l_int32") int nalloc(); public native L_PTRA nalloc(int nalloc);    /* size of allocated ptr array         */
    public native @Cast("l_int32") int imax(); public native L_PTRA imax(int imax);      /* greatest valid index                */
    public native @Cast("l_int32") int nactual(); public native L_PTRA nactual(int nactual);   /* actual number of stored elements    */
    public native Pointer array(int i); public native L_PTRA array(int i, Pointer array);
    @MemberGetter public native @Cast("void**") PointerPointer array();     /* ptr array                           */
}


    /* Array of generic pointer arrays */
@Name("L_Ptraa") public static class L_PTRAA extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public L_PTRAA() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public L_PTRAA(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public L_PTRAA(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public L_PTRAA position(int position) {
        return (L_PTRAA)super.position(position);
    }

    public native @Cast("l_int32") int nalloc(); public native L_PTRAA nalloc(int nalloc);    /* size of allocated ptr array         */
    public native L_PTRA ptra(int i); public native L_PTRAA ptra(int i, L_PTRA ptra);
    @MemberGetter public native @Cast("L_Ptra**") PointerPointer ptra();      /* array of ptra                       */
}



/*------------------------------------------------------------------------* 
 *                              Array flags                               *
 *------------------------------------------------------------------------*/

    /* Flags for removal from L_Ptra */
/** enum  */
public static final int
    L_NO_COMPACTION = 1,        /* null the pointer only  */
    L_COMPACTION = 2;            /* compact the array      */

    /* Flags for insertion into L_Ptra */
/** enum  */
public static final int
    L_AUTO_DOWNSHIFT = 0,       /* choose based on number of holes        */
    L_MIN_DOWNSHIFT = 1,        /* downshifts min # of ptrs below insert  */
    L_FULL_DOWNSHIFT = 2;        /* downshifts all ptrs below insert       */

    /* Accessor flags for L_Ptraa */
/** enum  */
public static final int
    L_HANDLE_ONLY = 0,          /* ptr to L_Ptra; caller can inspect only    */
    L_REMOVE = 1;                /* caller owns; destroy or save in L_Ptraa   */


// #endif  /* LEPTONICA_PTRA_H */


// Parsed from leptonica/queue.h

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

// #ifndef  LEPTONICA_QUEUE_H
// #define  LEPTONICA_QUEUE_H

/*
 *  queue.h
 *
 *      Expandable pointer queue for arbitrary void* data.
 *
 *      The L_Queue is a fifo that implements a queue of void* pointers.
 *      It can be used to hold a queue of any type of struct.
 *
 *      Internally, it maintains two counters:
 *          nhead:  location of head (in ptrs) from the beginning
 *                  of the array.
 *          nelem:  number of ptr elements stored in the queue.
 *
 *      The element at the head of the queue, which is the next to
 *      be removed, is array[nhead].  The location at the tail of the
 *      queue to which the next element will be added is
 *      array[nhead + nelem].
 *               
 *      As items are added to the queue, nelem increases.
 *      As items are removed, nhead increases and nelem decreases.
 *      Any time the tail reaches the end of the allocated array,
 *      all the pointers are shifted to the left, so that the head
 *      is at the beginning of the array.
 *      If the array becomes more than 3/4 full, it doubles in size.
 *
 *      The auxiliary stack can be used in a wrapper for re-using
 *      items popped from the queue.  It is not made by default.
 *
 *      For further implementation details, see queue.c.
 */

@Name("L_Queue") public static class L_QUEUE extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public L_QUEUE() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public L_QUEUE(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public L_QUEUE(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public L_QUEUE position(int position) {
        return (L_QUEUE)super.position(position);
    }

    public native @Cast("l_int32") int nalloc(); public native L_QUEUE nalloc(int nalloc);     /* size of allocated ptr array            */
    public native @Cast("l_int32") int nhead(); public native L_QUEUE nhead(int nhead);      /* location of head (in ptrs) from the    */
                                 /* beginning of the array                 */
    public native @Cast("l_int32") int nelem(); public native L_QUEUE nelem(int nelem);      /* number of elements stored in the queue */
    public native Pointer array(int i); public native L_QUEUE array(int i, Pointer array);
    @MemberGetter public native @Cast("void**") PointerPointer array();      /* ptr array                              */
    public native L_STACK stack(); public native L_QUEUE stack(L_STACK stack);      /* auxiliary stack                        */

}


// #endif  /* LEPTONICA_QUEUE_H */


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
    public L_STACK() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public L_STACK(int size) { super((Pointer)null); allocateArray(size); }
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



// Parsed from leptonica/arrayaccess.h

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

// #ifndef  LEPTONICA_ARRAY_ACCESS_H
// #define  LEPTONICA_ARRAY_ACCESS_H

/*
 *  arrayaccess.h
 *
 *  1, 2, 4, 8, 16 and 32 bit data access within an array of 32-bit words
 *
 *  This is used primarily to access 1, 2, 4, 8, 16 and 32 bit pixels
 *  in a line of image data, represented as an array of 32-bit words.
 *
 *     pdata:  pointer to first 32-bit word in the array
 *     n:      index of the pixel in the array
 *
 *  Function calls for these accessors are defined in arrayaccess.c.
 *
 *  However, for efficiency we use the inline macros for all accesses.
 *  Even though the 2 and 4 bit set* accessors are more complicated,
 *  they are about 10% faster than the function calls.
 *
 *  The 32 bit access is just a cast and ptr arithmetic.  We include
 *  it so that the input ptr can be void*.
 *
 *  At the end of this file is code for invoking the function calls
 *  instead of inlining.
 *
 *  The macro SET_DATA_BIT_VAL(pdata, n, val) is a bit slower than
 *      if (val == 0)
 *          CLEAR_DATA_BIT(pdata, n);
 *      else
 *          SET_DATA_BIT(pdata, n);
 */


    /* Use the inline accessors (except with _MSC_VER), because they
     * are faster.  */
public static final int USE_INLINE_ACCESSORS =    1;

// #if USE_INLINE_ACCESSORS
// #ifndef _MSC_VER

    /*--------------------------------------------------*
     *                     1 bit access                 *
     *--------------------------------------------------*/
// #define  GET_DATA_BIT(pdata, n)
//     ((*((l_uint32 *)(pdata) + ((n) >> 5)) >> (31 - ((n) & 31))) & 1)

// #define  SET_DATA_BIT(pdata, n)
//     (*((l_uint32 *)(pdata) + ((n) >> 5)) |= (0x80000000 >> ((n) & 31)))

// #define  CLEAR_DATA_BIT(pdata, n)
//     (*((l_uint32 *)(pdata) + ((n) >> 5)) &= ~(0x80000000 >> ((n) & 31)))

// #define  SET_DATA_BIT_VAL(pdata, n, val)
//     ({l_uint32 *_TEMP_WORD_PTR_;
//      _TEMP_WORD_PTR_ = (l_uint32 *)(pdata) + ((n) >> 5);
//      *_TEMP_WORD_PTR_ &= ~(0x80000000 >> ((n) & 31));
//      *_TEMP_WORD_PTR_ |= ((val) << (31 - ((n) & 31)));
//     })


    /*--------------------------------------------------*
     *                     2 bit access                 *
     *--------------------------------------------------*/
// #define  GET_DATA_DIBIT(pdata, n)
//     ((*((l_uint32 *)(pdata) + ((n) >> 4)) >> (2 * (15 - ((n) & 15)))) & 3)

// #define  SET_DATA_DIBIT(pdata, n, val)
//     ({l_uint32 *_TEMP_WORD_PTR_;
//      _TEMP_WORD_PTR_ = (l_uint32 *)(pdata) + ((n) >> 4);
//      *_TEMP_WORD_PTR_ &= ~(0xc0000000 >> (2 * ((n) & 15)));
//      *_TEMP_WORD_PTR_ |= (((val) & 3) << (30 - 2 * ((n) & 15)));
//     })

// #define  CLEAR_DATA_DIBIT(pdata, n)
//     (*((l_uint32 *)(pdata) + ((n) >> 4)) &= ~(0xc0000000 >> (2 * ((n) & 15))))


    /*--------------------------------------------------*
     *                     4 bit access                 *
     *--------------------------------------------------*/
// #define  GET_DATA_QBIT(pdata, n)
//      ((*((l_uint32 *)(pdata) + ((n) >> 3)) >> (4 * (7 - ((n) & 7)))) & 0xf)

// #define  SET_DATA_QBIT(pdata, n, val)
//     ({l_uint32 *_TEMP_WORD_PTR_;
//      _TEMP_WORD_PTR_ = (l_uint32 *)(pdata) + ((n) >> 3);
//      *_TEMP_WORD_PTR_ &= ~(0xf0000000 >> (4 * ((n) & 7)));
//      *_TEMP_WORD_PTR_ |= (((val) & 15) << (28 - 4 * ((n) & 7)));
//     })

// #define  CLEAR_DATA_QBIT(pdata, n)
//     (*((l_uint32 *)(pdata) + ((n) >> 3)) &= ~(0xf0000000 >> (4 * ((n) & 7))))


    /*--------------------------------------------------*
     *                     8 bit access                 *
     *--------------------------------------------------*/
// #ifdef  L_BIG_ENDIAN
// #define  GET_DATA_BYTE(pdata, n)
//              (*((l_uint8 *)(pdata) + (n)))
// #else  /* L_LITTLE_ENDIAN */
// #define  GET_DATA_BYTE(pdata, n)
//              (*(l_uint8 *)((l_uintptr_t)((l_uint8 *)(pdata) + (n)) ^ 3))
// #endif  /* L_BIG_ENDIAN */

// #ifdef  L_BIG_ENDIAN
// #define  SET_DATA_BYTE(pdata, n, val)
//              (*((l_uint8 *)(pdata) + (n)) = (val))
// #else  /* L_LITTLE_ENDIAN */
// #define  SET_DATA_BYTE(pdata, n, val)
//              (*(l_uint8 *)((l_uintptr_t)((l_uint8 *)(pdata) + (n)) ^ 3) = (val))
// #endif  /* L_BIG_ENDIAN */


    /*--------------------------------------------------*
     *                    16 bit access                 *
     *--------------------------------------------------*/
// #ifdef  L_BIG_ENDIAN
// #define  GET_DATA_TWO_BYTES(pdata, n)
//              (*((l_uint16 *)(pdata) + (n)))
// #else  /* L_LITTLE_ENDIAN */
// #define  GET_DATA_TWO_BYTES(pdata, n)
//              (*(l_uint16 *)((l_uintptr_t)((l_uint16 *)(pdata) + (n)) ^ 2))
// #endif  /* L_BIG_ENDIAN */

// #ifdef  L_BIG_ENDIAN
// #define  SET_DATA_TWO_BYTES(pdata, n, val)
//              (*((l_uint16 *)(pdata) + (n)) = (val))
// #else  /* L_LITTLE_ENDIAN */
// #define  SET_DATA_TWO_BYTES(pdata, n, val)
//              (*(l_uint16 *)((l_uintptr_t)((l_uint16 *)(pdata) + (n)) ^ 2) = (val))
// #endif  /* L_BIG_ENDIAN */


    /*--------------------------------------------------*
     *                    32 bit access                 *
     *--------------------------------------------------*/
// #define  GET_DATA_FOUR_BYTES(pdata, n)
//              (*((l_uint32 *)(pdata) + (n)))

// #define  SET_DATA_FOUR_BYTES(pdata, n, val)
//              (*((l_uint32 *)(pdata) + (n)) = (val))


// #endif  /* ! _MSC_VER */
// #endif  /* USE_INLINE_ACCESSORS */



    /*--------------------------------------------------*
     *  Slower, using function calls for all accessors  *
     *--------------------------------------------------*/
// #if !USE_INLINE_ACCESSORS || defined(_MSC_VER)
// #define  GET_DATA_BIT(pdata, n)               l_getDataBit(pdata, n)
// #define  SET_DATA_BIT(pdata, n)               l_setDataBit(pdata, n)
// #define  CLEAR_DATA_BIT(pdata, n)             l_clearDataBit(pdata, n)
// #define  SET_DATA_BIT_VAL(pdata, n, val)      l_setDataBitVal(pdata, n, val)

// #define  GET_DATA_DIBIT(pdata, n)             l_getDataDibit(pdata, n)
// #define  SET_DATA_DIBIT(pdata, n, val)        l_setDataDibit(pdata, n, val)
// #define  CLEAR_DATA_DIBIT(pdata, n)           l_clearDataDibit(pdata, n)

// #define  GET_DATA_QBIT(pdata, n)              l_getDataQbit(pdata, n)
// #define  SET_DATA_QBIT(pdata, n, val)         l_setDataQbit(pdata, n, val)
// #define  CLEAR_DATA_QBIT(pdata, n)            l_clearDataQbit(pdata, n)

// #define  GET_DATA_BYTE(pdata, n)              l_getDataByte(pdata, n)
// #define  SET_DATA_BYTE(pdata, n, val)         l_setDataByte(pdata, n, val)

// #define  GET_DATA_TWO_BYTES(pdata, n)         l_getDataTwoBytes(pdata, n)
// #define  SET_DATA_TWO_BYTES(pdata, n, val)    l_setDataTwoBytes(pdata, n, val)

// #define  GET_DATA_FOUR_BYTES(pdata, n)         l_getDataFourBytes(pdata, n)
// #define  SET_DATA_FOUR_BYTES(pdata, n, val)    l_setDataFourBytes(pdata, n, val)
// #endif  /* !USE_INLINE_ACCESSORS || _MSC_VER */


// #endif /* LEPTONICA_ARRAY_ACCESS_H */


// Parsed from leptonica/bmf.h

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

// #ifndef  LEPTONICA_BMF_H
// #define  LEPTONICA_BMF_H

/*
 *  bmf.h
 *
 *     Simple data structure to hold bitmap fonts and related data
 */

    /* Constants for deciding when text block is divided into paragraphs */
/** enum  */
public static final int
    SPLIT_ON_LEADING_WHITE = 1,    /* tab or space at beginning of line   */
    SPLIT_ON_BLANK_LINE    = 2,    /* newline with optional white space   */
    SPLIT_ON_BOTH          = 3;     /* leading white space or newline      */


@Name("L_Bmf") public static class L_BMF extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public L_BMF() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public L_BMF(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public L_BMF(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public L_BMF position(int position) {
        return (L_BMF)super.position(position);
    }

    public native PIXA pixa(); public native L_BMF pixa(PIXA pixa);        /* pixa of bitmaps for 93 characters        */
    public native @Cast("l_int32") int size(); public native L_BMF size(int size);        /* font size (in points at 300 ppi)         */
    public native @Cast("char*") BytePointer directory(); public native L_BMF directory(BytePointer directory);   /* directory containing font bitmaps        */
    public native @Cast("l_int32") int baseline1(); public native L_BMF baseline1(int baseline1);   /* baseline offset for ascii 33 - 57        */
    public native @Cast("l_int32") int baseline2(); public native L_BMF baseline2(int baseline2);   /* baseline offset for ascii 58 - 91        */
    public native @Cast("l_int32") int baseline3(); public native L_BMF baseline3(int baseline3);   /* baseline offset for ascii 93 - 126       */
    public native @Cast("l_int32") int lineheight(); public native L_BMF lineheight(int lineheight);  /* max height of line of chars              */
    public native @Cast("l_int32") int kernwidth(); public native L_BMF kernwidth(int kernwidth);   /* pixel dist between char bitmaps          */
    public native @Cast("l_int32") int spacewidth(); public native L_BMF spacewidth(int spacewidth);  /* pixel dist between word bitmaps          */
    public native @Cast("l_int32") int vertlinesep(); public native L_BMF vertlinesep(int vertlinesep); /* extra vertical space between text lines  */
    public native @Cast("l_int32*") IntPointer fonttab(); public native L_BMF fonttab(IntPointer fonttab);     /* table mapping ascii --> font index       */
    public native @Cast("l_int32*") IntPointer baselinetab(); public native L_BMF baselinetab(IntPointer baselinetab); /* table mapping ascii --> baseline offset  */
    public native @Cast("l_int32*") IntPointer widthtab(); public native L_BMF widthtab(IntPointer widthtab);    /* table mapping ascii --> char width       */
}

// #endif  /* LEPTONICA_BMF_H */


// Parsed from leptonica/ccbord.h

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

// #ifndef  LEPTONICA_CCBORD_H
// #define  LEPTONICA_CCBORD_H

/*
 *  ccbord.h
 *
 *           CCBord:   represents a single connected component
 *           CCBorda:  an array of CCBord
 */

    /* Use in ccbaStepChainsToPixCoords() */
/** enum  */
public static final int
      CCB_LOCAL_COORDS = 1,
      CCB_GLOBAL_COORDS = 2;

    /* Use in ccbaGenerateSPGlobalLocs() */
/** enum  */
public static final int
      CCB_SAVE_ALL_PTS = 1,
      CCB_SAVE_TURNING_PTS = 2;


    /* CCBord contains:
     *
     *    (1) a minimally-clipped bitmap of the component (pix),
     *    (2) a boxa consisting of:
     *          for the primary component:
     *                (xul, yul) pixel location in global coords
     *                (w, h) of the bitmap
     *          for the hole components:
     *                (x, y) in relative coordinates in primary component
     *                (w, h) of the hole border (which is 2 pixels
     *                       larger in each direction than the hole itself)
     *    (3) a pta ('start') of the initial border pixel location for each
     *        closed curve, all in relative coordinates of the primary
     *        component.  This is given for the primary component,
     *        followed by the hole components, if any.
     *    (4) a refcount of the ccbord; used internally when a ccbord
     *        is accessed from a ccborda (array of ccbord)
     *    (5) a ptaa for the chain code for the border in relative
     *        coordinates, where the first pta is the exterior border
     *        and all other pta are for interior borders (holes)
     *    (6) a ptaa for the global pixel loc rendition of the border,
     *        where the first pta is the exterior border and all other
     *        pta are for interior borders (holes).
     *        This is derived from the local or step chain code.
     *    (7) a numaa for the chain code for the border as orientation
     *        directions between successive border pixels, where
     *        the first numa is the exterior border and all other
     *        numa are for interior borders (holes).  This is derived
     *        from the local chain code.  The 8 directions are 0 - 7.
     *    (8) a pta for a single chain for each c.c., comprised of outer
     *        and hole borders, plus cut paths between them, all in
     *        local coords.
     *    (9) a pta for a single chain for each c.c., comprised of outer
     *        and hole borders, plus cut paths between them, all in
     *        global coords.
     */
@Name("CCBord") public static class CCBORD extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CCBORD() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CCBORD(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CCBORD(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CCBORD position(int position) {
        return (CCBORD)super.position(position);
    }

    public native PIX pix(); public native CCBORD pix(PIX pix);            /* component bitmap (min size)      */
    public native BOXA boxa(); public native CCBORD boxa(BOXA boxa);           /* regions of each closed curve     */
    public native PTA start(); public native CCBORD start(PTA start);          /* initial border pixel locations   */
    public native @Cast("l_int32") int refcount(); public native CCBORD refcount(int refcount);       /* number of handles; start at 1    */
    public native PTAA local(); public native CCBORD local(PTAA local);          /* ptaa of chain pixels (local)     */
    public native PTAA global(); public native CCBORD global(PTAA global);         /* ptaa of chain pixels (global)    */
    public native NUMAA step(); public native CCBORD step(NUMAA step);           /* numaa of chain code (step dir)   */
    public native PTA splocal(); public native CCBORD splocal(PTA splocal);        /* pta of single chain (local)      */
    public native PTA spglobal(); public native CCBORD spglobal(PTA spglobal);       /* pta of single chain (global)     */
}


@Name("CCBorda") public static class CCBORDA extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CCBORDA() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CCBORDA(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CCBORDA(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CCBORDA position(int position) {
        return (CCBORDA)super.position(position);
    }

    public native PIX pix(); public native CCBORDA pix(PIX pix);            /* input pix (may be null)          */
    public native @Cast("l_int32") int w(); public native CCBORDA w(int w);              /* width of pix                     */
    public native @Cast("l_int32") int h(); public native CCBORDA h(int h);              /* height of pix                    */
    public native @Cast("l_int32") int n(); public native CCBORDA n(int n);              /* number of ccbord in ptr array    */
    public native @Cast("l_int32") int nalloc(); public native CCBORDA nalloc(int nalloc);         /* number of ccbord ptrs allocated  */
    public native CCBORD ccb(int i); public native CCBORDA ccb(int i, CCBORD ccb);
    @MemberGetter public native @Cast("CCBord**") PointerPointer ccb();            /* ccb ptr array                    */
}


// #endif  /* LEPTONICA_CCBORD_H */



// Parsed from leptonica/dewarp.h

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

// #ifndef  LEPTONICA_DEWARP_H
// #define  LEPTONICA_DEWARP_H

/*
 *  dewarp.h
 *
 *     Data structure to hold arrays and results for generating
 *     horizontal and vertical disparity arrays based on textlines.
 *     Each disparity array is two-dimensional.  The vertical disparity
 *     array gives a vertical displacement, relative to the lowest point
 *     in the textlines.  The horizontal disparty array gives a horizontal
 *     displacement, relative to the minimum values (for even pages)
 *     or maximum values (for odd pages) of the left and right ends of
 *     full textlines.  Horizontal alignment always involves translations
 *     away from the book gutter.
 *
 *     We have intentionally separated the process of building models
 *     from the rendering process that uses the models.  For any page,
 *     the building operation either creates an actual model (that is,
 *     a model with at least the vertical disparity being computed, and
 *     for which the 'success' flag is set) or fails to create a model.
 *     However, at rendering time, a page can have one of two different
 *     types of models.
 *     (1) A valid model is an actual model that meets the rendering
 *         constraints, which are limits on model curvature parameters.
 *         See dewarpaTestForValidModel() for details.
 *         Valid models are identified by dewarpaInsertRefModels(),
 *         which sets the 'vvalid' and 'hvalid' fields.  Only valid
 *         models are used for rendering.
 *     (2) A reference model is used by a page that doesn't have
 *         a valid model, but has a nearby valid model of the same
 *         parity (even/odd page) that it can use.  The range in pages
 *         to search for a valid model is given by the 'maxdist' field.
 *
 *     If a valid vertical disparity model (VDM) is not available,
 *     just use the input image.  Otherwise, assuming the VDM is available:
 *       (a) with useboth == 0, we use only the VDM.
 *       (b) with useboth == 1, we require using the VDM and, if a valid
 *           horizontal disparity model (HDM) is available, we also use it.
 *
 *     The 'maxdist' parameter is input when the dewarpa is created.
 *     The other rendering parameters have default values given in dewarp.c.
 *     All parameters used by rendering can be set (or reset) using accessors.
 *
 *     After dewarping, use of the VDM will cause all points on each
 *     altered curve to have a y-value equal to the minimum.  Use of
 *     the HDA will cause the left and right edges of the textlines
 *     to be vertically aligned if they had been typeset flush-left
 *     and flush-right, respectively.
 *
 *     The sampled disparity arrays are expanded to full resolution,
 *     using linear interpolation, and this is further expanded
 *     by slope continuation to the right and below if the image
 *     is larger than the full resolution disparity arrays.  Then
 *     the disparity correction can be applied to the input image.
 *     If the input pix are 2x reduced, the expansion from sampled
 *     to full res uses the product of (sampling) * (redfactor).
 *
 *     The most accurate results are produced at full resolution, and
 *     this is generally recommended.
 */

    /* Note on versioning of the serialization of this data structure:
     * The dewarping utility and the stored data can be expected to change.
     * In most situations, the serialized version is ephemeral -- it is
     * not needed after being used.  No functions will be provided to
     * convert between different versions. */
public static final int DEWARP_VERSION_NUMBER =      4;

@Name("L_Dewarpa") public static class L_DEWARPA extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public L_DEWARPA() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public L_DEWARPA(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public L_DEWARPA(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public L_DEWARPA position(int position) {
        return (L_DEWARPA)super.position(position);
    }

    public native @Cast("l_int32") int nalloc(); public native L_DEWARPA nalloc(int nalloc);        /* size of dewarp ptr array            */
    public native @Cast("l_int32") int maxpage(); public native L_DEWARPA maxpage(int maxpage);       /* maximum page number in array        */
    public native L_DEWARP dewarp(int i); public native L_DEWARPA dewarp(int i, L_DEWARP dewarp);
    @MemberGetter public native @Cast("L_Dewarp**") PointerPointer dewarp();        /* array of ptrs to page dewarp        */
    public native L_DEWARP dewarpcache(int i); public native L_DEWARPA dewarpcache(int i, L_DEWARP dewarpcache);
    @MemberGetter public native @Cast("L_Dewarp**") PointerPointer dewarpcache();   /* array of ptrs to cached dewarps     */
    public native NUMA namodels(); public native L_DEWARPA namodels(NUMA namodels);      /* list of page numbers for pages      */
                                      /* with page models                    */
    public native NUMA napages(); public native L_DEWARPA napages(NUMA napages);       /* list of page numbers with either    */
                                      /* page models or ref page models      */
    public native @Cast("l_int32") int redfactor(); public native L_DEWARPA redfactor(int redfactor);     /* reduction factor of input: 1 or 2   */
    public native @Cast("l_int32") int sampling(); public native L_DEWARPA sampling(int sampling);      /* disparity arrays sampling factor    */
    public native @Cast("l_int32") int minlines(); public native L_DEWARPA minlines(int minlines);      /* min number of long lines required   */
    public native @Cast("l_int32") int maxdist(); public native L_DEWARPA maxdist(int maxdist);       /* max distance for getting ref pages  */
    public native @Cast("l_int32") int max_linecurv(); public native L_DEWARPA max_linecurv(int max_linecurv);  /* maximum abs line curvature,         */
                                      /* in micro-units                      */
    public native @Cast("l_int32") int min_diff_linecurv(); public native L_DEWARPA min_diff_linecurv(int min_diff_linecurv); /* minimum abs diff line curvature */
                                      /* in micro-units                      */
    public native @Cast("l_int32") int max_diff_linecurv(); public native L_DEWARPA max_diff_linecurv(int max_diff_linecurv); /* maximum abs diff line curvature */
                                      /* in micro-units                      */
    public native @Cast("l_int32") int max_edgeslope(); public native L_DEWARPA max_edgeslope(int max_edgeslope); /* maximum abs left or right edge      */
                                      /* slope, in milli-units               */
    public native @Cast("l_int32") int max_edgecurv(); public native L_DEWARPA max_edgecurv(int max_edgecurv);  /* maximum abs left or right edge      */
                                      /* curvature, in micro-units           */
    public native @Cast("l_int32") int max_diff_edgecurv(); public native L_DEWARPA max_diff_edgecurv(int max_diff_edgecurv); /* maximum abs diff left-right     */
                                      /* edge curvature, in micro-units      */
    public native @Cast("l_int32") int useboth(); public native L_DEWARPA useboth(int useboth);       /* use both disparity arrays if        */
                                      /* available; just vertical otherwise  */
    public native @Cast("l_int32") int modelsready(); public native L_DEWARPA modelsready(int modelsready);   /* invalid models have been removed    */
                                      /* and refs built against valid set    */
}


@Name("L_Dewarp") public static class L_DEWARP extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public L_DEWARP() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public L_DEWARP(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public L_DEWARP(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public L_DEWARP position(int position) {
        return (L_DEWARP)super.position(position);
    }

    public native L_DEWARPA dewa(); public native L_DEWARP dewa(L_DEWARPA dewa);         /* ptr to parent (not owned)            */
    public native PIX pixs(); public native L_DEWARP pixs(PIX pixs);         /* source pix, 1 bpp                    */
    public native FPIX sampvdispar(); public native L_DEWARP sampvdispar(FPIX sampvdispar);  /* sampled vert disparity array         */
    public native FPIX samphdispar(); public native L_DEWARP samphdispar(FPIX samphdispar);  /* sampled horiz disparity array        */
    public native FPIX fullvdispar(); public native L_DEWARP fullvdispar(FPIX fullvdispar);  /* full vert disparity array            */
    public native FPIX fullhdispar(); public native L_DEWARP fullhdispar(FPIX fullhdispar);  /* full horiz disparity array           */
    public native NUMA namidys(); public native L_DEWARP namidys(NUMA namidys);      /* sorted y val of midpoint each line   */
    public native NUMA nacurves(); public native L_DEWARP nacurves(NUMA nacurves);     /* sorted curvature of each line        */
    public native @Cast("l_int32") int w(); public native L_DEWARP w(int w);            /* width of source image                */
    public native @Cast("l_int32") int h(); public native L_DEWARP h(int h);            /* height of source image               */
    public native @Cast("l_int32") int pageno(); public native L_DEWARP pageno(int pageno);       /* page number; important for reuse     */
    public native @Cast("l_int32") int sampling(); public native L_DEWARP sampling(int sampling);     /* sampling factor of disparity arrays  */
    public native @Cast("l_int32") int redfactor(); public native L_DEWARP redfactor(int redfactor);    /* reduction factor of pixs: 1 or 2     */
    public native @Cast("l_int32") int minlines(); public native L_DEWARP minlines(int minlines);     /* min number of long lines required    */
    public native @Cast("l_int32") int nlines(); public native L_DEWARP nlines(int nlines);       /* number of long lines found           */
    public native @Cast("l_int32") int mincurv(); public native L_DEWARP mincurv(int mincurv);      /* min line curvature in micro-units    */
    public native @Cast("l_int32") int maxcurv(); public native L_DEWARP maxcurv(int maxcurv);      /* max line curvature in micro-units    */
    public native @Cast("l_int32") int leftslope(); public native L_DEWARP leftslope(int leftslope);    /* left edge slope in milli-units       */
    public native @Cast("l_int32") int rightslope(); public native L_DEWARP rightslope(int rightslope);   /* right edge slope in milli-units      */
    public native @Cast("l_int32") int leftcurv(); public native L_DEWARP leftcurv(int leftcurv);     /* left edge curvature in micro-units   */
    public native @Cast("l_int32") int rightcurv(); public native L_DEWARP rightcurv(int rightcurv);    /* right edge curvature in micro-units  */
    public native @Cast("l_int32") int nx(); public native L_DEWARP nx(int nx);           /* number of sampling pts in x-dir      */
    public native @Cast("l_int32") int ny(); public native L_DEWARP ny(int ny);           /* number of sampling pts in y-dir      */
    public native @Cast("l_int32") int hasref(); public native L_DEWARP hasref(int hasref);       /* 0 if normal; 1 if has a refpage      */
    public native @Cast("l_int32") int refpage(); public native L_DEWARP refpage(int refpage);      /* page with disparity model to use     */
    public native @Cast("l_int32") int vsuccess(); public native L_DEWARP vsuccess(int vsuccess);     /* sets to 1 if vert disparity builds   */
    public native @Cast("l_int32") int hsuccess(); public native L_DEWARP hsuccess(int hsuccess);     /* sets to 1 if horiz disparity builds  */
    public native @Cast("l_int32") int vvalid(); public native L_DEWARP vvalid(int vvalid);       /* sets to 1 if valid vert disparity    */
    public native @Cast("l_int32") int hvalid(); public native L_DEWARP hvalid(int hvalid);       /* sets to 1 if valid horiz disparity   */
    public native @Cast("l_int32") int debug(); public native L_DEWARP debug(int debug);        /* sets to 1 if debug output requested  */
}

// #endif  /* LEPTONICA_DEWARP_H */


// Parsed from leptonica/gplot.h

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

// #ifndef  LEPTONICA_GPLOT_H
// #define  LEPTONICA_GPLOT_H

/*
 *   gplot.h
 *
 *       Data structures and parameters for generating gnuplot files
 */

public static final int GPLOT_VERSION_NUMBER =    1;

public static final int NUM_GPLOT_STYLES =      5;
/** enum GPLOT_STYLE */
public static final int
    GPLOT_LINES       = 0,
    GPLOT_POINTS      = 1,
    GPLOT_IMPULSES    = 2,
    GPLOT_LINESPOINTS = 3,
    GPLOT_DOTS        = 4;

public static final int NUM_GPLOT_OUTPUTS =     6;
/** enum GPLOT_OUTPUT */
public static final int
    GPLOT_NONE  = 0,
    GPLOT_PNG   = 1,
    GPLOT_PS    = 2,
    GPLOT_EPS   = 3,
    GPLOT_X11   = 4,
    GPLOT_LATEX = 5;

/** enum GPLOT_SCALING */
public static final int
    GPLOT_LINEAR_SCALE  = 0,   /* default */
    GPLOT_LOG_SCALE_X   = 1,
    GPLOT_LOG_SCALE_Y   = 2,
    GPLOT_LOG_SCALE_X_Y = 3;

@MemberGetter public static native @Cast("const char*") BytePointer gplotstylenames(int i);
@MemberGetter public static native @Cast("const char**") PointerPointer gplotstylenames();  /* used in gnuplot cmd file */
@MemberGetter public static native @Cast("const char*") BytePointer gplotfilestyles(int i);
@MemberGetter public static native @Cast("const char**") PointerPointer gplotfilestyles();  /* used in simple file input */
@MemberGetter public static native @Cast("const char*") BytePointer gplotfileoutputs(int i);
@MemberGetter public static native @Cast("const char**") PointerPointer gplotfileoutputs(); /* used in simple file input */

@Name("GPlot") public static class GPLOT extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public GPLOT() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public GPLOT(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public GPLOT(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public GPLOT position(int position) {
        return (GPLOT)super.position(position);
    }

    public native @Cast("char*") BytePointer rootname(); public native GPLOT rootname(BytePointer rootname);   /* for cmd, data, output            */
    public native @Cast("char*") BytePointer cmdname(); public native GPLOT cmdname(BytePointer cmdname);    /* command file name                */
    public native SARRAY cmddata(); public native GPLOT cmddata(SARRAY cmddata);    /* command file contents            */
    public native SARRAY datanames(); public native GPLOT datanames(SARRAY datanames);  /* data file names                  */
    public native SARRAY plotdata(); public native GPLOT plotdata(SARRAY plotdata);   /* plot data (1 string/file)        */
    public native SARRAY plottitles(); public native GPLOT plottitles(SARRAY plottitles); /* title for each individual plot   */
    public native NUMA plotstyles(); public native GPLOT plotstyles(NUMA plotstyles); /* plot style for individual plots  */
    public native @Cast("l_int32") int nplots(); public native GPLOT nplots(int nplots);     /* current number of plots          */
    public native @Cast("char*") BytePointer outname(); public native GPLOT outname(BytePointer outname);    /* output file name                 */
    public native @Cast("l_int32") int outformat(); public native GPLOT outformat(int outformat);  /* GPLOT_OUTPUT values              */
    public native @Cast("l_int32") int scaling(); public native GPLOT scaling(int scaling);    /* GPLOT_SCALING values             */
    public native @Cast("char*") BytePointer title(); public native GPLOT title(BytePointer title);      /* optional                         */
    public native @Cast("char*") BytePointer xlabel(); public native GPLOT xlabel(BytePointer xlabel);     /* optional x axis label            */
    public native @Cast("char*") BytePointer ylabel(); public native GPLOT ylabel(BytePointer ylabel);     /* optional y axis label            */
}


// #endif /* LEPTONICA_GPLOT_H */


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
    public L_COMP_DATA() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public L_COMP_DATA(int size) { super((Pointer)null); allocateArray(size); }
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
@Name("L_Pdf_Data") public static class L_PDF_DATA extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public L_PDF_DATA() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public L_PDF_DATA(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public L_PDF_DATA(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public L_PDF_DATA position(int position) {
        return (L_PDF_DATA)super.position(position);
    }

    public native @Cast("char*") BytePointer title(); public native L_PDF_DATA title(BytePointer title);        /* optional title for pdf              */
    public native @Cast("l_int32") int n(); public native L_PDF_DATA n(int n);            /* number of images                    */
    public native @Cast("l_int32") int ncmap(); public native L_PDF_DATA ncmap(int ncmap);        /* number of colormaps                 */
    public native L_PTRA cida(); public native L_PDF_DATA cida(L_PTRA cida);         /* array of compressed image data      */
    public native @Cast("char*") BytePointer id(); public native L_PDF_DATA id(BytePointer id);           /* %PDF-1.2 id string                  */
    public native @Cast("char*") BytePointer obj1(); public native L_PDF_DATA obj1(BytePointer obj1);         /* catalog string                      */
    public native @Cast("char*") BytePointer obj2(); public native L_PDF_DATA obj2(BytePointer obj2);         /* metadata string                     */
    public native @Cast("char*") BytePointer obj3(); public native L_PDF_DATA obj3(BytePointer obj3);         /* pages string                        */
    public native @Cast("char*") BytePointer obj4(); public native L_PDF_DATA obj4(BytePointer obj4);         /* page string (variable data)         */
    public native @Cast("char*") BytePointer obj5(); public native L_PDF_DATA obj5(BytePointer obj5);         /* content string (variable data)      */
    public native @Cast("char*") BytePointer poststream(); public native L_PDF_DATA poststream(BytePointer poststream);   /* post-binary-stream string           */
    public native @Cast("char*") BytePointer trailer(); public native L_PDF_DATA trailer(BytePointer trailer);      /* trailer string (variable data)      */
    public native PTA xy(); public native L_PDF_DATA xy(PTA xy);           /* store (xpt, ypt) array              */
    public native PTA wh(); public native L_PDF_DATA wh(PTA wh);           /* store (wpt, hpt) array              */
    public native BOX mediabox(); public native L_PDF_DATA mediabox(BOX mediabox);     /* bounding region for all images      */
    public native SARRAY saprex(); public native L_PDF_DATA saprex(SARRAY saprex);       /* pre-binary-stream xobject strings   */
    public native SARRAY sacmap(); public native L_PDF_DATA sacmap(SARRAY sacmap);       /* colormap pdf object strings         */
    public native L_DNA objsize(); public native L_PDF_DATA objsize(L_DNA objsize);      /* sizes of each pdf string object     */
    public native L_DNA objloc(); public native L_PDF_DATA objloc(L_DNA objloc);       /* location of each pdf string object  */
    public native @Cast("l_int32") int xrefloc(); public native L_PDF_DATA xrefloc(int xrefloc);      /* location of xref                    */
}


// #endif  /* LEPTONICA_IMAGEIO_H */


// Parsed from leptonica/jbclass.h

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

// #ifndef  LEPTONICA_JBCLASS_H
// #define  LEPTONICA_JBCLASS_H

/*
 * jbclass.h
 *
 *       JbClasser
 *       JbData
 */


    /* The JbClasser struct holds all the data accumulated during the
     * classification process that can be used for a compressed
     * jbig2-type representation of a set of images.  This is created
     * in an initialization process and added to as the selected components
     * on each successive page are analyzed.   */
@Name("JbClasser") public static class JBCLASSER extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public JBCLASSER() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public JBCLASSER(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public JBCLASSER(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public JBCLASSER position(int position) {
        return (JBCLASSER)super.position(position);
    }

    public native SARRAY safiles(); public native JBCLASSER safiles(SARRAY safiles);      /* input page image file names            */
    public native @Cast("l_int32") int method(); public native JBCLASSER method(int method);       /* JB_RANKHAUS, JB_CORRELATION            */
    public native @Cast("l_int32") int components(); public native JBCLASSER components(int components);   /* JB_CONN_COMPS, JB_CHARACTERS or        */
                                   /* JB_WORDS                               */
    public native @Cast("l_int32") int maxwidth(); public native JBCLASSER maxwidth(int maxwidth);     /* max component width allowed            */
    public native @Cast("l_int32") int maxheight(); public native JBCLASSER maxheight(int maxheight);    /* max component height allowed           */
    public native @Cast("l_int32") int npages(); public native JBCLASSER npages(int npages);       /* number of pages already processed      */
    public native @Cast("l_int32") int baseindex(); public native JBCLASSER baseindex(int baseindex);    /* number of components already processed */
                                   /* on fully processed pages               */
    public native NUMA nacomps(); public native JBCLASSER nacomps(NUMA nacomps);      /* number of components on each page      */
    public native @Cast("l_int32") int sizehaus(); public native JBCLASSER sizehaus(int sizehaus);     /* size of square struct element for haus */
    public native @Cast("l_float32") float rankhaus(); public native JBCLASSER rankhaus(float rankhaus);     /* rank val of haus match, each way       */
    public native @Cast("l_float32") float thresh(); public native JBCLASSER thresh(float thresh);       /* thresh value for correlation score     */
    public native @Cast("l_float32") float weightfactor(); public native JBCLASSER weightfactor(float weightfactor); /* corrects thresh value for heaver       */
                                   /* components; use 0 for no correction    */
    public native NUMA naarea(); public native JBCLASSER naarea(NUMA naarea);       /* w * h of each template, without extra  */
                                   /* border pixels                          */
    public native @Cast("l_int32") int w(); public native JBCLASSER w(int w);            /* max width of original src images       */
    public native @Cast("l_int32") int h(); public native JBCLASSER h(int h);            /* max height of original src images      */
    public native @Cast("l_int32") int nclass(); public native JBCLASSER nclass(int nclass);       /* current number of classes              */
    public native @Cast("l_int32") int keep_pixaa(); public native JBCLASSER keep_pixaa(int keep_pixaa);   /* If zero, pixaa isn't filled            */
    public native PIXAA pixaa(); public native JBCLASSER pixaa(PIXAA pixaa);        /* instances for each class; unbordered   */
    public native PIXA pixat(); public native JBCLASSER pixat(PIXA pixat);        /* templates for each class; bordered     */
                                   /* and not dilated                        */
    public native PIXA pixatd(); public native JBCLASSER pixatd(PIXA pixatd);       /* templates for each class; bordered     */
                                   /* and dilated                            */
    public native NUMAHASH nahash(); public native JBCLASSER nahash(NUMAHASH nahash);       /* Hash table to find templates by size   */
    public native NUMA nafgt(); public native JBCLASSER nafgt(NUMA nafgt);        /* fg areas of undilated templates;       */
                                   /* only used for rank < 1.0               */
    public native PTA ptac(); public native JBCLASSER ptac(PTA ptac);         /* centroids of all bordered cc           */
    public native PTA ptact(); public native JBCLASSER ptact(PTA ptact);        /* centroids of all bordered template cc  */
    public native NUMA naclass(); public native JBCLASSER naclass(NUMA naclass);      /* array of class ids for each component  */
    public native NUMA napage(); public native JBCLASSER napage(NUMA napage);       /* array of page nums for each component  */
    public native PTA ptaul(); public native JBCLASSER ptaul(PTA ptaul);        /* array of UL corners at which the       */
                                   /* template is to be placed for each      */
                                   /* component                              */
    public native PTA ptall(); public native JBCLASSER ptall(PTA ptall);        /* similar to ptaul, but for LL corners   */
}


    /* The JbData struct holds all the data required for
     * the compressed jbig-type representation of a set of images.
     * The data can be written to file, read back, and used
     * to regenerate an approximate version of the original,
     * which differs in two ways from the original:
     *   (1) It uses a template image for each c.c. instead of the
     *       original instance, for each occurrence on each page.
     *   (2) It discards components with either a height or width larger
     *       than the maximuma, given here by the lattice dimensions
     *       used for storing the templates.   */
@Name("JbData") public static class JBDATA extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public JBDATA() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public JBDATA(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public JBDATA(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public JBDATA position(int position) {
        return (JBDATA)super.position(position);
    }

    public native PIX pix(); public native JBDATA pix(PIX pix);        /* template composite for all classes    */
    public native @Cast("l_int32") int npages(); public native JBDATA npages(int npages);     /* number of pages                       */
    public native @Cast("l_int32") int w(); public native JBDATA w(int w);          /* max width of original page images     */
    public native @Cast("l_int32") int h(); public native JBDATA h(int h);          /* max height of original page images    */
    public native @Cast("l_int32") int nclass(); public native JBDATA nclass(int nclass);     /* number of classes                     */
    public native @Cast("l_int32") int latticew(); public native JBDATA latticew(int latticew);   /* lattice width for template composite  */
    public native @Cast("l_int32") int latticeh(); public native JBDATA latticeh(int latticeh);   /* lattice height for template composite */
    public native NUMA naclass(); public native JBDATA naclass(NUMA naclass);    /* array of class ids for each component */
    public native NUMA napage(); public native JBDATA napage(NUMA napage);     /* array of page nums for each component */
    public native PTA ptaul(); public native JBDATA ptaul(PTA ptaul);      /* array of UL corners at which the      */
                                    /* template is to be placed for each     */
                                    /* component                             */
}


    /* Classifier methods */
/** enum  */
public static final int
   JB_RANKHAUS = 0,
   JB_CORRELATION = 1;

    /* For jbGetComponents(): type of component to extract from images */
/** enum  */
public static final int
   JB_CONN_COMPS = 0,
   JB_CHARACTERS = 1,
   JB_WORDS = 2;

    /* These parameters are used for naming the two files
     * in which the jbig2-like compressed data is stored.  */
public static final String JB_TEMPLATE_EXT =      ".templates.png";
public static final String JB_DATA_EXT =          ".data";


// #endif  /* LEPTONICA_JBCLASS_H */


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
    public SEL() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public SEL(int size) { super((Pointer)null); allocateArray(size); }
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

@Name("Sela") public static class SELA extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public SELA() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public SELA(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public SELA(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public SELA position(int position) {
        return (SELA)super.position(position);
    }

    public native @Cast("l_int32") int n(); public native SELA n(int n);         /* number of sel actually stored           */
    public native @Cast("l_int32") int nalloc(); public native SELA nalloc(int nalloc);    /* size of allocated ptr array             */
    public native SEL sel(int i); public native SELA sel(int i, SEL sel);
    @MemberGetter public native @Cast("Sel**") PointerPointer sel();       /* sel ptr array                           */
}


/*-------------------------------------------------------------------------*
 *                                 Kernel                                  *
 *-------------------------------------------------------------------------*/
public static final int KERNEL_VERSION_NUMBER =    2;

@Name("L_Kernel") public static class L_KERNEL extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public L_KERNEL() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public L_KERNEL(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public L_KERNEL(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public L_KERNEL position(int position) {
        return (L_KERNEL)super.position(position);
    }

    public native @Cast("l_int32") int sy(); public native L_KERNEL sy(int sy);          /* kernel height                            */
    public native @Cast("l_int32") int sx(); public native L_KERNEL sx(int sx);          /* kernel width                             */
    public native @Cast("l_int32") int cy(); public native L_KERNEL cy(int cy);          /* y location of kernel origin              */
    public native @Cast("l_int32") int cx(); public native L_KERNEL cx(int cx);          /* x location of kernel origin              */
    public native @Cast("l_float32*") FloatPointer data(int i); public native L_KERNEL data(int i, FloatPointer data);
    @MemberGetter public native @Cast("l_float32**") PointerPointer data();        /* data[i][j] in [row][col] order           */
}


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
@MemberGetter public static native @Cast("const l_int32") int ADDED_BORDER();
public static final int ADDED_BORDER = ADDED_BORDER();   /* pixels, not bits */


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
    public PIX() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public PIX(int size) { super((Pointer)null); allocateArray(size); }
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
    public PIXCMAP() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public PIXCMAP(int size) { super((Pointer)null); allocateArray(size); }
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
@Name("RGBA_Quad") public static class RGBA_QUAD extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public RGBA_QUAD() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public RGBA_QUAD(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public RGBA_QUAD(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public RGBA_QUAD position(int position) {
        return (RGBA_QUAD)super.position(position);
    }

    public native @Cast("l_uint8") byte blue(); public native RGBA_QUAD blue(byte blue);
    public native @Cast("l_uint8") byte green(); public native RGBA_QUAD green(byte green);
    public native @Cast("l_uint8") byte red(); public native RGBA_QUAD red(byte red);
    public native @Cast("l_uint8") byte alpha(); public native RGBA_QUAD alpha(byte alpha);
}



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

@MemberGetter public static native @Cast("const l_int32") int L_RED_SHIFT();
public static final int L_RED_SHIFT = L_RED_SHIFT();           /* 24 */
@MemberGetter public static native @Cast("const l_int32") int L_GREEN_SHIFT();
public static final int L_GREEN_SHIFT = L_GREEN_SHIFT();         /* 16 */
@MemberGetter public static native @Cast("const l_int32") int L_BLUE_SHIFT();
public static final int L_BLUE_SHIFT = L_BLUE_SHIFT();          /*  8 */
@MemberGetter public static native @Cast("const l_int32") int L_ALPHA_SHIFT();
public static final int L_ALPHA_SHIFT = L_ALPHA_SHIFT();     /*  0 */


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
public static final float L_RED_WEIGHT = L_RED_WEIGHT();
@MemberGetter public static native @Cast("const l_float32") float L_GREEN_WEIGHT();
public static final float L_GREEN_WEIGHT = L_GREEN_WEIGHT();
@MemberGetter public static native @Cast("const l_float32") float L_BLUE_WEIGHT();
public static final float L_BLUE_WEIGHT = L_BLUE_WEIGHT();


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
    public PIXA() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public PIXA(int size) { super((Pointer)null); allocateArray(size); }
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
    public PIXAA() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public PIXAA(int size) { super((Pointer)null); allocateArray(size); }
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
    public BOX() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public BOX(int size) { super((Pointer)null); allocateArray(size); }
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
    public BOXA() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public BOXA(int size) { super((Pointer)null); allocateArray(size); }
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
    public BOXAA() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public BOXAA(int size) { super((Pointer)null); allocateArray(size); }
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
    public PTA() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public PTA(int size) { super((Pointer)null); allocateArray(size); }
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
@Name("Ptaa") public static class PTAA extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public PTAA() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public PTAA(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public PTAA(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public PTAA position(int position) {
        return (PTAA)super.position(position);
    }

    public native @Cast("l_int32") int n(); public native PTAA n(int n);           /* number of pta in ptr array        */
    public native @Cast("l_int32") int nalloc(); public native PTAA nalloc(int nalloc);      /* number of pta ptrs allocated      */
    public native PTA pta(int i); public native PTAA pta(int i, PTA pta);
    @MemberGetter public native @Cast("Pta**") PointerPointer pta();         /* pta ptr array                     */
}


/*-------------------------------------------------------------------------*
 *                       Pix accumulator container                         *
 *-------------------------------------------------------------------------*/
@Name("Pixacc") public static class PIXACC extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public PIXACC() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public PIXACC(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public PIXACC(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public PIXACC position(int position) {
        return (PIXACC)super.position(position);
    }

    public native @Cast("l_int32") int w(); public native PIXACC w(int w);            /* array width                       */
    public native @Cast("l_int32") int h(); public native PIXACC h(int h);            /* array height                      */
    public native @Cast("l_int32") int offset(); public native PIXACC offset(int offset);       /* used to allow negative            */
                                      /* intermediate results              */
    public native PIX pix(); public native PIXACC pix(PIX pix);          /* the 32 bit accumulator pix        */
}


/*-------------------------------------------------------------------------*
 *                              Pix tiling                                 *
 *-------------------------------------------------------------------------*/
@Name("PixTiling") public static class PIXTILING extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public PIXTILING() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public PIXTILING(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public PIXTILING(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public PIXTILING position(int position) {
        return (PIXTILING)super.position(position);
    }

    public native PIX pix(); public native PIXTILING pix(PIX pix);         /* input pix (a clone)               */
    public native @Cast("l_int32") int nx(); public native PIXTILING nx(int nx);          /* number of tiles horizontally      */
    public native @Cast("l_int32") int ny(); public native PIXTILING ny(int ny);          /* number of tiles vertically        */
    public native @Cast("l_int32") int w(); public native PIXTILING w(int w);           /* tile width                        */
    public native @Cast("l_int32") int h(); public native PIXTILING h(int h);           /* tile height                       */
    public native @Cast("l_int32") int xoverlap(); public native PIXTILING xoverlap(int xoverlap);    /* overlap on left and right         */
    public native @Cast("l_int32") int yoverlap(); public native PIXTILING yoverlap(int yoverlap);    /* overlap on top and bottom         */
    public native @Cast("l_int32") int strip(); public native PIXTILING strip(int strip);       /* strip for paint; default is TRUE  */
}


/*-------------------------------------------------------------------------*
 *                       FPix: pix with float array                        *
 *-------------------------------------------------------------------------*/
public static final int FPIX_VERSION_NUMBER =      2;

@Name("FPix") public static class FPIX extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public FPIX() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public FPIX(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FPIX(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public FPIX position(int position) {
        return (FPIX)super.position(position);
    }

    public native @Cast("l_int32") int w(); public native FPIX w(int w);           /* width in pixels                   */
    public native @Cast("l_int32") int h(); public native FPIX h(int h);           /* height in pixels                  */
    public native @Cast("l_int32") int wpl(); public native FPIX wpl(int wpl);         /* 32-bit words/line                 */
    public native @Cast("l_uint32") int refcount(); public native FPIX refcount(int refcount);    /* reference count (1 if no clones)  */
    public native @Cast("l_int32") int xres(); public native FPIX xres(int xres);        /* image res (ppi) in x direction    */
                                      /* (use 0 if unknown)                */
    public native @Cast("l_int32") int yres(); public native FPIX yres(int yres);        /* image res (ppi) in y direction    */
                                      /* (use 0 if unknown)                */
    public native @Cast("l_float32*") FloatPointer data(); public native FPIX data(FloatPointer data);        /* the float image data              */
}


@Name("FPixa") public static class FPIXA extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public FPIXA() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public FPIXA(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FPIXA(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public FPIXA position(int position) {
        return (FPIXA)super.position(position);
    }

    public native @Cast("l_int32") int n(); public native FPIXA n(int n);            /* number of fpix in ptr array       */
    public native @Cast("l_int32") int nalloc(); public native FPIXA nalloc(int nalloc);       /* number of fpix ptrs allocated     */
    public native @Cast("l_uint32") int refcount(); public native FPIXA refcount(int refcount);     /* reference count (1 if no clones)  */
    public native FPIX fpix(int i); public native FPIXA fpix(int i, FPIX fpix);
    @MemberGetter public native @Cast("FPix**") PointerPointer fpix();         /* the array of ptrs to fpix         */
}


/*-------------------------------------------------------------------------*
 *                       DPix: pix with double array                       *
 *-------------------------------------------------------------------------*/
public static final int DPIX_VERSION_NUMBER =      2;

@Name("DPix") public static class DPIX extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public DPIX() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public DPIX(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DPIX(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public DPIX position(int position) {
        return (DPIX)super.position(position);
    }

    public native @Cast("l_int32") int w(); public native DPIX w(int w);           /* width in pixels                   */
    public native @Cast("l_int32") int h(); public native DPIX h(int h);           /* height in pixels                  */
    public native @Cast("l_int32") int wpl(); public native DPIX wpl(int wpl);         /* 32-bit words/line                 */
    public native @Cast("l_uint32") int refcount(); public native DPIX refcount(int refcount);    /* reference count (1 if no clones)  */
    public native @Cast("l_int32") int xres(); public native DPIX xres(int xres);        /* image res (ppi) in x direction    */
                                      /* (use 0 if unknown)                */
    public native @Cast("l_int32") int yres(); public native DPIX yres(int yres);        /* image res (ppi) in y direction    */
                                      /* (use 0 if unknown)                */
    public native @Cast("l_float64*") DoublePointer data(); public native DPIX data(DoublePointer data);        /* the double image data             */
}


/*-------------------------------------------------------------------------*
 *                        PixComp: compressed pix                          *
 *-------------------------------------------------------------------------*/
@Name("PixComp") public static class PIXC extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public PIXC() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public PIXC(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public PIXC(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public PIXC position(int position) {
        return (PIXC)super.position(position);
    }

    public native @Cast("l_int32") int w(); public native PIXC w(int w);           /* width in pixels                   */
    public native @Cast("l_int32") int h(); public native PIXC h(int h);           /* height in pixels                  */
    public native @Cast("l_int32") int d(); public native PIXC d(int d);           /* depth in bits                     */
    public native @Cast("l_int32") int xres(); public native PIXC xres(int xres);        /* image res (ppi) in x direction    */
                                      /*   (use 0 if unknown)              */
    public native @Cast("l_int32") int yres(); public native PIXC yres(int yres);        /* image res (ppi) in y direction    */
                                      /*   (use 0 if unknown)              */
    public native @Cast("l_int32") int comptype(); public native PIXC comptype(int comptype);    /* compressed format (IFF_TIFF_G4,   */
                                      /*   IFF_PNG, IFF_JFIF_JPEG)         */
    public native @Cast("char*") BytePointer text(); public native PIXC text(BytePointer text);        /* text string associated with pix   */
    public native @Cast("l_int32") int cmapflag(); public native PIXC cmapflag(int cmapflag);    /* flag (1 for cmap, 0 otherwise)    */
    public native @Cast("l_uint8*") BytePointer data(); public native PIXC data(BytePointer data);        /* the compressed image data         */
    public native @Cast("size_t") long size(); public native PIXC size(long size);        /* size of the data array            */
}


/*-------------------------------------------------------------------------*
 *                     PixaComp: array of compressed pix                   *
 *-------------------------------------------------------------------------*/
public static final int PIXACOMP_VERSION_NUMBER =      2;

@Name("PixaComp") public static class PIXAC extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public PIXAC() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public PIXAC(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public PIXAC(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public PIXAC position(int position) {
        return (PIXAC)super.position(position);
    }

    public native @Cast("l_int32") int n(); public native PIXAC n(int n);           /* number of PixComp in ptr array    */
    public native @Cast("l_int32") int nalloc(); public native PIXAC nalloc(int nalloc);      /* number of PixComp ptrs allocated  */
    public native @Cast("l_int32") int offset(); public native PIXAC offset(int offset);      /* indexing offset into ptr array    */
    public native PIXC pixc(int i); public native PIXAC pixc(int i, PIXC pixc);
    @MemberGetter public native @Cast("PixComp**") PointerPointer pixc();        /* the array of ptrs to PixComp      */
    public native BOXA boxa(); public native PIXAC boxa(BOXA boxa);        /* array of boxes                    */
}


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
@MemberGetter public static native @Cast("const l_int32") int L_NOCOPY();
public static final int L_NOCOPY = L_NOCOPY();  /* copyflag value in sarrayGetString() */


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


// Parsed from leptonica/recog.h

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

// #ifndef  LEPTONICA_RECOG_H
// #define  LEPTONICA_RECOG_H

/*
 *  recog.h
 *
 *     A simple utility for training and recognizing individual
 *     machine-printed text characters.  In an application, one can
 *     envision using a number of these, one for each trained set.
 *
 *     In training mode, a set of labelled bitmaps is presented, either
 *     one at a time, or in a directory, or in a pixa.  If in a directory,
 *     or a pixa, the labelling text string must be embedded in the
 *     text field of the image file.
 *
 *     Any number of recognizers (L_Recog) can be trained and then used
 *     together in an array (L_Recoga).  All these trained structures
 *     can be serialized to file and read back.  The serialized version
 *     holds all the bitmaps used for training, plus, for arbitrary
 *     character sets, the UTF8 representation and the lookup table
 *     mapping from the character representation to index.
 *
 *     There are three levels of "sets" here:
 *
 *       (1) Example set: the examples representing a character that
 *           were printed in the same way, so that they can be combined
 *           without scaling to form an "average" template for the character.
 *           In the recognition phase, we use either this aligned average,
 *           or the individual bitmaps.  All examples in the set are given
 *           the same character label.   Example: the letter 'a' in the
 *           predominant font in a book.
 *
 *       (2) Character set (represented by L_Recog, a single recognizer):
 *           The set of different characters, each of which is described
 *           by (1).  Each element of the set has a different character
 *           label.  Example: the digits '0' through '9' that are used for
 *           page numbering in a book.
 *
 *       (3) Recognizer set (represented by L_Recoga, an array of recogs):
 *           A set of recognizers, each of which is described by (2).
 *           In general, we do not want to combine the character sets
 *           with the same labels within different recognizer sets,
 *           because the bitmaps can differ in font type, style or size.
 *           Example 1: the letter 'a' can be printed in two very different
 *           ways (either with a large loop or with a smaller loop in
 *           the lower half); both share the same label but need to be
 *           distinguished so that they are not mixed when averaging.
 *           Example 2: a recognizer trained for a book may be missing
 *           some characters, so we need to supplement it with another
 *           "generic" or "bootstrap" recognizer that has the additional
 *           characters from a variety of sources.  Bootstrap recognizers
 *           must be run in a mode where all characters are scaled.
 *
 *     In the recognition process, for each component in an input image,
 *     each recognizer (L_Recog) records the best match (highest
 *     correlation score).  If there is more than one recognizer, these
 *     results are aggregated to find the best match for each character
 *     for all the recognizers, and this is stored in L_Recoga.
 */

public static final int RECOG_VERSION_NUMBER =      1;

@Name("L_Recoga") public static class L_RECOGA extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public L_RECOGA() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public L_RECOGA(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public L_RECOGA(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public L_RECOGA position(int position) {
        return (L_RECOGA)super.position(position);
    }

    public native @Cast("l_int32") int n(); public native L_RECOGA n(int n);      /* number of recogs                         */
    public native @Cast("l_int32") int nalloc(); public native L_RECOGA nalloc(int nalloc); /* number of recog ptrs allocated           */
    public native L_RECOG recog(int i); public native L_RECOGA recog(int i, L_RECOG recog);
    @MemberGetter public native @Cast("L_Recog**") PointerPointer recog();  /* recog ptr array                          */
    public native L_RCHA rcha(); public native L_RECOGA rcha(L_RCHA rcha);   /* stores the array of best chars           */
}


@Name("L_Recog") public static class L_RECOG extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public L_RECOG() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public L_RECOG(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public L_RECOG(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public L_RECOG position(int position) {
        return (L_RECOG)super.position(position);
    }

    public native @Cast("l_int32") int scalew(); public native L_RECOG scalew(int scalew);       /* scale all examples to this width;        */
                                 /* use 0 prevent horizontal scaling         */
    public native @Cast("l_int32") int scaleh(); public native L_RECOG scaleh(int scaleh);       /* scale all examples to this height;       */
                                 /* use 0 prevent vertical scaling           */
    public native @Cast("l_int32") int templ_type(); public native L_RECOG templ_type(int templ_type);   /* template type: either an average of      */
                                 /* examples (L_USE_AVERAGE) or the set      */
                                 /* of all examples (L_USE_ALL)              */
    public native @Cast("l_int32") int maxarraysize(); public native L_RECOG maxarraysize(int maxarraysize); /* initialize container arrays to this      */
    public native @Cast("l_int32") int setsize(); public native L_RECOG setsize(int setsize);      /* size of character set                    */
    public native @Cast("l_int32") int threshold(); public native L_RECOG threshold(int threshold);    /* for binarizing if depth > 1              */
    public native @Cast("l_int32") int maxyshift(); public native L_RECOG maxyshift(int maxyshift);    /* vertical jiggle on nominal centroid      */
                                 /* alignment; typically 0 or 1              */
    public native @Cast("l_float32") float asperity_fr(); public native L_RECOG asperity_fr(float asperity_fr);  /* +- allowed fractional asperity ratio     */
    public native @Cast("l_int32") int charset_type(); public native L_RECOG charset_type(int charset_type); /* one of L_ARABIC_NUMERALS, etc.           */
    public native @Cast("l_int32") int charset_size(); public native L_RECOG charset_size(int charset_size); /* expected number of classes in charset    */
    public native @Cast("char*") BytePointer bootdir(); public native L_RECOG bootdir(BytePointer bootdir);      /* dir with bootstrap pixa charsets         */
    public native @Cast("char*") BytePointer bootpattern(); public native L_RECOG bootpattern(BytePointer bootpattern);  /* file pattern for bootstrap pixa charsets */
    public native @Cast("char*") BytePointer bootpath(); public native L_RECOG bootpath(BytePointer bootpath);     /* path for single bootstrap pixa charset   */
    public native @Cast("l_int32") int min_nopad(); public native L_RECOG min_nopad(int min_nopad);    /* min number of samples without padding    */
    public native @Cast("l_int32") int max_afterpad(); public native L_RECOG max_afterpad(int max_afterpad); /* max number of samples after padding      */
    public native @Cast("l_int32") int samplenum(); public native L_RECOG samplenum(int samplenum);    /* keep track of number of training samples */
    public native @Cast("l_int32") int minwidth_u(); public native L_RECOG minwidth_u(int minwidth_u);   /* min width of averaged unscaled templates */
    public native @Cast("l_int32") int maxwidth_u(); public native L_RECOG maxwidth_u(int maxwidth_u);   /* max width of averaged unscaled templates */
    public native @Cast("l_int32") int minheight_u(); public native L_RECOG minheight_u(int minheight_u);  /* min height of averaged unscaled templates */
    public native @Cast("l_int32") int maxheight_u(); public native L_RECOG maxheight_u(int maxheight_u);  /* max height of averaged unscaled templates */
    public native @Cast("l_int32") int minwidth(); public native L_RECOG minwidth(int minwidth);     /* min width of averaged scaled templates   */
    public native @Cast("l_int32") int maxwidth(); public native L_RECOG maxwidth(int maxwidth);     /* max width of averaged scaled templates   */
    public native @Cast("l_int32") int ave_done(); public native L_RECOG ave_done(int ave_done);     /* set to 1 when averaged bitmaps are made  */
    public native @Cast("l_int32") int train_done(); public native L_RECOG train_done(int train_done);   /* set to 1 when training is complete or    */
                                 /* identification has started               */
    public native @Cast("l_int32") int min_splitw(); public native L_RECOG min_splitw(int min_splitw);   /* min component width kept in splitting    */
    public native @Cast("l_int32") int min_splith(); public native L_RECOG min_splith(int min_splith);   /* min component height kept in splitting   */
    public native @Cast("l_int32") int max_splith(); public native L_RECOG max_splith(int max_splith);   /* max component height kept in splitting   */
    public native SARRAY sa_text(); public native L_RECOG sa_text(SARRAY sa_text);      /* text array for arbitrary char set        */
    public native L_DNA dna_tochar(); public native L_RECOG dna_tochar(L_DNA dna_tochar);   /* index-to-char lut for arbitrary char set */
    public native @Cast("l_int32*") IntPointer centtab(); public native L_RECOG centtab(IntPointer centtab);      /* table for finding centroids              */
    public native @Cast("l_int32*") IntPointer sumtab(); public native L_RECOG sumtab(IntPointer sumtab);       /* table for finding pixel sums             */
    public native @Cast("char*") BytePointer fname(); public native L_RECOG fname(BytePointer fname);        /* serialized filename (if read)            */
    public native PIXAA pixaa_u(); public native L_RECOG pixaa_u(PIXAA pixaa_u);      /* all unscaled bitmaps for each class      */
    public native PIXA pixa_u(); public native L_RECOG pixa_u(PIXA pixa_u);       /* averaged unscaled bitmaps for each class */
    public native PTAA ptaa_u(); public native L_RECOG ptaa_u(PTAA ptaa_u);       /* centroids of all unscaled bitmaps        */
    public native PTA pta_u(); public native L_RECOG pta_u(PTA pta_u);        /* centroids of unscaled averaged bitmaps   */
    public native NUMAA naasum_u(); public native L_RECOG naasum_u(NUMAA naasum_u);     /* area of all unscaled bitmap examples     */
    public native NUMA nasum_u(); public native L_RECOG nasum_u(NUMA nasum_u);      /* area of unscaled averaged bitmaps        */
    public native PIXAA pixaa(); public native L_RECOG pixaa(PIXAA pixaa);        /* all bitmap examples for each class       */
    public native PIXA pixa(); public native L_RECOG pixa(PIXA pixa);         /* averaged bitmaps for each class          */
    public native PTAA ptaa(); public native L_RECOG ptaa(PTAA ptaa);         /* centroids of all bitmap examples         */
    public native PTA pta(); public native L_RECOG pta(PTA pta);          /* centroids of averaged bitmaps            */
    public native NUMAA naasum(); public native L_RECOG naasum(NUMAA naasum);       /* area of all bitmap examples              */
    public native NUMA nasum(); public native L_RECOG nasum(NUMA nasum);        /* area of averaged bitmaps                 */
    public native PIXA pixa_tr(); public native L_RECOG pixa_tr(PIXA pixa_tr);      /* input training images                    */
    public native PIXA pixadb_ave(); public native L_RECOG pixadb_ave(PIXA pixadb_ave);   /* unscaled and scaled averaged bitmaps     */
    public native PIXA pixa_id(); public native L_RECOG pixa_id(PIXA pixa_id);      /* input images for identifying             */
    public native PIX pixdb_ave(); public native L_RECOG pixdb_ave(PIX pixdb_ave);    /* debug: best match of input against ave.  */
    public native PIX pixdb_range(); public native L_RECOG pixdb_range(PIX pixdb_range);  /* debug: best matches within range         */
    public native PIXA pixadb_boot(); public native L_RECOG pixadb_boot(PIXA pixadb_boot);  /* debug: bootstrap training results        */
    public native PIXA pixadb_split(); public native L_RECOG pixadb_split(PIXA pixadb_split); /* debug: splitting results                 */
    public native L_BMF bmf(); public native L_RECOG bmf(L_BMF bmf);          /* bmf fonts                                */
    public native @Cast("l_int32") int bmf_size(); public native L_RECOG bmf_size(int bmf_size);     /* font size of bmf; default is 6 pt        */
    public native L_RDID did(); public native L_RECOG did(L_RDID did);          /* temp data used for image decoding        */
    public native L_RCH rch(); public native L_RECOG rch(L_RCH rch);          /* temp data used for holding best char     */
    public native L_RCHA rcha(); public native L_RECOG rcha(L_RCHA rcha);         /* temp data used for array of best chars   */
    public native @Cast("l_int32") int bootrecog(); public native L_RECOG bootrecog(int bootrecog);    /* 1 if using bootstrap samples; else 0     */
    public native @Cast("l_int32") int index(); public native L_RECOG index(int index);        /* recog index in recoga; -1 if no parent   */
    public native L_RECOGA parent(); public native L_RECOG parent(L_RECOGA parent);    /* ptr to parent array; can be null         */

}

/*
 *  Data returned from correlation matching on a single character
 */
@Name("L_Rch") public static class L_RCH extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public L_RCH() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public L_RCH(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public L_RCH(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public L_RCH position(int position) {
        return (L_RCH)super.position(position);
    }

    public native @Cast("l_int32") int index(); public native L_RCH index(int index);        /* index of best template                   */
    public native @Cast("l_float32") float score(); public native L_RCH score(float score);        /* correlation score of best template       */
    public native @Cast("char*") BytePointer text(); public native L_RCH text(BytePointer text);         /* character string of best template        */
    public native @Cast("l_int32") int sample(); public native L_RCH sample(int sample);       /* index of best sample (within the best    */
                                 /* template class, if all samples are used) */
    public native @Cast("l_int32") int xloc(); public native L_RCH xloc(int xloc);         /* x-location of template (delx + shiftx)   */
    public native @Cast("l_int32") int yloc(); public native L_RCH yloc(int yloc);         /* y-location of template (dely + shifty)   */
    public native @Cast("l_int32") int width(); public native L_RCH width(int width);        /* width of best template                   */
}

/*
 *  Data returned from correlation matching on an array of characters
 */
@Name("L_Rcha") public static class L_RCHA extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public L_RCHA() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public L_RCHA(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public L_RCHA(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public L_RCHA position(int position) {
        return (L_RCHA)super.position(position);
    }

    public native NUMA naindex(); public native L_RCHA naindex(NUMA naindex);      /* indices of best templates                */
    public native NUMA nascore(); public native L_RCHA nascore(NUMA nascore);      /* correlation scores of best templates     */
    public native SARRAY satext(); public native L_RCHA satext(SARRAY satext);       /* character strings of best templates      */
    public native NUMA nasample(); public native L_RCHA nasample(NUMA nasample);     /* indices of best samples                  */
    public native NUMA naxloc(); public native L_RCHA naxloc(NUMA naxloc);       /* x-locations of templates (delx + shiftx) */
    public native NUMA nayloc(); public native L_RCHA nayloc(NUMA nayloc);       /* y-locations of templates (dely + shifty) */
    public native NUMA nawidth(); public native L_RCHA nawidth(NUMA nawidth);      /* widths of best templates                 */
}

/*
 *  Data used for decoding a line of characters.
 */
@Name("L_Rdid") public static class L_RDID extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public L_RDID() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public L_RDID(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public L_RDID(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public L_RDID position(int position) {
        return (L_RDID)super.position(position);
    }

    public native PIX pixs(); public native L_RDID pixs(PIX pixs);         /* clone of pix to be decoded               */
    public native @Cast("l_int32*") IntPointer counta(int i); public native L_RDID counta(int i, IntPointer counta);
    @MemberGetter public native @Cast("l_int32**") PointerPointer counta();       /* count array for each averaged template   */
    public native @Cast("l_int32*") IntPointer delya(int i); public native L_RDID delya(int i, IntPointer delya);
    @MemberGetter public native @Cast("l_int32**") PointerPointer delya();        /* best y-shift array per averaged template */
    public native @Cast("l_int32") int narray(); public native L_RDID narray(int narray);       /* number of averaged templates             */
    public native @Cast("l_int32") int size(); public native L_RDID size(int size);         /* size of count array (width of pixs)      */
    public native @Cast("l_int32*") IntPointer setwidth(); public native L_RDID setwidth(IntPointer setwidth);     /* setwidths for each template              */
    public native NUMA nasum(); public native L_RDID nasum(NUMA nasum);        /* pixel count in pixs by column            */
    public native NUMA namoment(); public native L_RDID namoment(NUMA namoment);     /* first moment of pixels in pixs by column */
    public native @Cast("l_int32") int fullarrays(); public native L_RDID fullarrays(int fullarrays);   /* 1 if full arrays are made; 0 otherwise   */
    public native @Cast("l_float32*") FloatPointer beta(); public native L_RDID beta(FloatPointer beta);         /* channel coeffs for template fg term      */
    public native @Cast("l_float32*") FloatPointer gamma(); public native L_RDID gamma(FloatPointer gamma);        /* channel coeffs for bit-and term          */
    public native @Cast("l_float32*") FloatPointer trellisscore(); public native L_RDID trellisscore(FloatPointer trellisscore); /* score on trellis                         */
    public native @Cast("l_int32*") IntPointer trellistempl(); public native L_RDID trellistempl(IntPointer trellistempl); /* template on trellis (for backtrack)      */
    public native NUMA natempl(); public native L_RDID natempl(NUMA natempl);      /* indices of best path templates           */
    public native NUMA naxloc(); public native L_RDID naxloc(NUMA naxloc);       /* x locations of best path templates       */
    public native NUMA nadely(); public native L_RDID nadely(NUMA nadely);       /* y locations of best path templates       */
    public native NUMA nawidth(); public native L_RDID nawidth(NUMA nawidth);      /* widths of best path templates            */
    public native NUMA nascore(); public native L_RDID nascore(NUMA nascore);      /* correlation scores: best path templates  */
    public native NUMA natempl_r(); public native L_RDID natempl_r(NUMA natempl_r);    /* indices of best rescored templates       */
    public native NUMA naxloc_r(); public native L_RDID naxloc_r(NUMA naxloc_r);     /* x locations of best rescoredtemplates    */
    public native NUMA nadely_r(); public native L_RDID nadely_r(NUMA nadely_r);     /* y locations of best rescoredtemplates    */
    public native NUMA nawidth_r(); public native L_RDID nawidth_r(NUMA nawidth_r);    /* widths of best rescoredtemplates         */
    public native NUMA nascore_r(); public native L_RDID nascore_r(NUMA nascore_r);    /* correlation scores: rescored templates   */
}


/*-------------------------------------------------------------------------*
 *                    Flags for selecting processing                       *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_SELECT_UNSCALED = 0,       /* select the unscaled bitmaps            */
    L_SELECT_SCALED = 1,         /* select the scaled bitmaps              */
    L_SELECT_BOTH = 2;            /* select both unscaled and scaled        */

/*-------------------------------------------------------------------------*
 *                Flags for determining what to test against               *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_USE_AVERAGE = 0,         /* form template from class average         */
    L_USE_ALL = 1;              /* match against all elements of each class */

/*-------------------------------------------------------------------------*
 *             Flags for describing limited character sets                 *
 *-------------------------------------------------------------------------*/
/** enum  */
public static final int
    L_UNKNOWN = 0,             /* character set type is not specified      */
    L_ARABIC_NUMERALS = 1,     /* 10 digits                                */
    L_LC_ROMAN_NUMERALS = 2,   /* 7 lower-case letters (i,v,x,l,c,d,m)     */
    L_UC_ROMAN_NUMERALS = 3,   /* 7 upper-case letters (I,V,X,L,C,D,M)     */
    L_LC_ALPHA = 4,            /* 26 lower-case letters                    */
    L_UC_ALPHA = 5;             /* 26 upper-case letters                    */

// #endif  /* LEPTONICA_RECOG_H */


// Parsed from leptonica/regutils.h

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

// #ifndef  LEPTONICA_REGUTILS_H
// #define  LEPTONICA_REGUTILS_H

/*
 *   regutils.h
 *
 *   Contains this regression test parameter packaging struct
 *       struct L_RegParams
 *
 *   The regression test utility allows you to write regression tests
 *   that compare results with existing "golden files" and with
 *   compiled in data.
 *
 *   Regression tests can be called in three ways.
 *   For example, for distance_reg:
 *
 *       Case 1: distance_reg [generate]
 *           This generates golden files in /tmp for the reg test.
 *
 *       Case 2: distance_reg compare
 *           This runs the test against the set of golden files.  It
 *           appends to 'outfile.txt' either "SUCCESS" or "FAILURE",
 *           as well as the details of any parts of the test that failed.
 *           It writes to a temporary file stream (fp)
 *
 *       Case 3: distance_reg display
 *           This runs the test but makes no comparison of the output
 *           against the set of golden files.  In addition, this displays
 *           images and plots that are specified in the test under
 *           control of the display variable.  Display is enabled only
 *           for this case.  Using 'display' on the command line is optional.
 *
 *   Regression tests follow the pattern given below.  Tests are
 *   automatically numbered sequentially, and it is convenient to
 *   comment each with a number to keep track (for comparison tests
 *   and for debugging).  In an actual case, comparisons of pix and
 *   of files can occur in any order.  We give a specific order here
 *   for clarity.
 *
 *       L_REGPARAMS  *rp;  // holds data required by the test functions
 *
 *       // Setup variables; optionally open stream
 *       if (regTestSetup(argc, argv, &rp))
 *           return 1;
 *
 *       // Test pairs of generated pix for identity.  This compares
 *       // two pix; no golden file is generated.
 *       regTestComparePix(rp, pix1, pix2);  // 0
 *
 *       // Test pairs of generated pix for similarity.  This compares
 *       // two pix; no golden file is generated.  The last arg determines
 *       // if stats are to be written to stderr.
 *       regTestCompareSimilarPix(rp, pix1, pix2, 15, 0.001, 0);  // 1
 *
 *       // Generation of <newfile*> outputs and testing for identity
 *       // These files can be anything, of course.
 *       regTestCheckFile(rp, <newfile0>);  // 2
 *       regTestCheckFile(rp, <newfile1>);  // 3
 *
 *       // Test pairs of output golden files for identity.  Here we
 *       // are comparing golden files 2 and 3.
 *       regTestCompareFiles(rp, 2, 3);  // 4
 *
 *       // "Write and check".  This writes a pix using a canonical
 *       // formulation for the local filename and either:
 *       //     case 1: generates a golden file
 *       //     case 2: compares the local file with a golden file
 *       //     case 3: generates local files and displays
 *       // Here we write the pix compressed with png and jpeg, respectively;
 *       // Then check against the golden file.  The internal @index
 *       // is incremented; it is embedded in the local filename and,
 *       // if generating, in the golden file as well.
 *       regTestWritePixAndCheck(rp, pix1, IFF_PNG);  // 5
 *       regTestWritePixAndCheck(rp, pix2, IFF_JFIF_JPEG);  // 6
 *
 *       // Display if reg test was called in 'display' mode
 *       pixDisplayWithTitle(pix1, 100, 100, NULL, rp->display);
 *
 *       // Clean up and output result
 *       regTestCleanup(rp);
 */

/*-------------------------------------------------------------------------*
 *                     Regression test parameter packer                    *
 *-------------------------------------------------------------------------*/
@Name("L_RegParams") public static class L_REGPARAMS extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public L_REGPARAMS() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public L_REGPARAMS(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public L_REGPARAMS(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public L_REGPARAMS position(int position) {
        return (L_REGPARAMS)super.position(position);
    }

    public native @Cast("FILE*") Pointer fp(); public native L_REGPARAMS fp(Pointer fp);        /* stream to temporary output file for compare mode */
    public native @Cast("char*") BytePointer testname(); public native L_REGPARAMS testname(BytePointer testname);  /* name of test, without '_reg'                     */
    public native @Cast("char*") BytePointer tempfile(); public native L_REGPARAMS tempfile(BytePointer tempfile);  /* name of temp file for compare mode output        */
    public native @Cast("l_int32") int mode(); public native L_REGPARAMS mode(int mode);      /* generate, compare or display                     */
    public native @Cast("l_int32") int index(); public native L_REGPARAMS index(int index);     /* index into saved files for this test; 0-based    */
    public native @Cast("l_int32") int success(); public native L_REGPARAMS success(int success);   /* overall result of the test                       */
    public native @Cast("l_int32") int display(); public native L_REGPARAMS display(int display);   /* 1 if in display mode; 0 otherwise                */
    public native L_TIMER tstart(); public native L_REGPARAMS tstart(L_TIMER tstart);    /* marks beginning of the reg test                  */
}


    /* Running modes for the test */
/** enum  */
public static final int
    L_REG_GENERATE = 0,
    L_REG_COMPARE = 1,
    L_REG_DISPLAY = 2;


// #endif  /* LEPTONICA_REGUTILS_H */



// Parsed from leptonica/stringcode.h

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

// #ifndef  LEPTONICA_STRINGCODE_H
// #define  LEPTONICA_STRINGCODE_H

/*
 *  stringcode.h
 *
 *     Data structure to hold accumulating generated code for storing
 *     and extracing serializable leptonica objects (e.g., pixa, recog).
 */

@Name("L_StrCode") public static class L_STRCODE extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public L_STRCODE() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public L_STRCODE(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public L_STRCODE(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public L_STRCODE position(int position) {
        return (L_STRCODE)super.position(position);
    }

    public native @Cast("l_int32") int fileno(); public native L_STRCODE fileno(int fileno);      /* index for function and output file names   */
    public native @Cast("l_int32") int ifunc(); public native L_STRCODE ifunc(int ifunc);       /* index into struct currently being stored   */
    public native SARRAY function(); public native L_STRCODE function(SARRAY function);    /* store case code for extraction             */
    public native SARRAY data(); public native L_STRCODE data(SARRAY data);        /* store base64 encoded data as strings       */
    public native SARRAY descr(); public native L_STRCODE descr(SARRAY descr);       /* store line in description table            */
    public native @Cast("l_int32") int n(); public native L_STRCODE n(int n);           /* number of data strings                     */
}

// #endif  /* LEPTONICA_STRINGCODE_H */


// Parsed from leptonica/sudoku.h

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

// #ifndef SUDOKU_H_INCLUDED
// #define SUDOKU_H_INCLUDED

/*
 *  sudoku.h
 *
 *    The L_Sudoku holds all the information of the current state.
 *
 *    The input to sudokuCreate() is a file with any number of lines
 *    starting with '#', followed by 9 lines consisting of 9 numbers
 *    in each line.  These have the known values and use 0 for the unknowns.
 *    Blank lines are ignored.
 *
 *    The @locs array holds the indices of the unknowns, numbered
 *    left-to-right and top-to-bottom from 0 to 80.  The array size
 *    is initialized to @num.  @current is the index into the @locs
 *    array of the current guess: locs[current].
 *
 *    The @state array is used to determine the validity of each guess.
 *    It is of size 81, and is initialized by setting the unknowns to 0
 *    and the knowns to their input values.
 */
@Name("L_Sudoku") public static class L_SUDOKU extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public L_SUDOKU() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public L_SUDOKU(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public L_SUDOKU(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public L_SUDOKU position(int position) {
        return (L_SUDOKU)super.position(position);
    }

    public native @Cast("l_int32") int num(); public native L_SUDOKU num(int num);         /* number of unknowns                     */
    public native @Cast("l_int32*") IntPointer locs(); public native L_SUDOKU locs(IntPointer locs);        /* location of unknowns                   */
    public native @Cast("l_int32") int current(); public native L_SUDOKU current(int current);     /* index into @locs of current location   */
    public native @Cast("l_int32*") IntPointer init(); public native L_SUDOKU init(IntPointer init);        /* initial state, with 0 representing     */
                                /* the unknowns                           */
    public native @Cast("l_int32*") IntPointer state(); public native L_SUDOKU state(IntPointer state);       /* present state, including inits and     */
                                /* guesses of unknowns up to @current     */
    public native @Cast("l_int32") int nguess(); public native L_SUDOKU nguess(int nguess);      /* shows current number of guesses        */
    public native @Cast("l_int32") int finished(); public native L_SUDOKU finished(int finished);    /* set to 1 when solved                   */
    public native @Cast("l_int32") int failure(); public native L_SUDOKU failure(int failure);     /* set to 1 if no solution is possible    */
}


    /* For printing out array data */
/** enum  */
public static final int
    L_SUDOKU_INIT = 0,
    L_SUDOKU_STATE = 1;

// #endif /* SUDOKU_H_INCLUDED */




// Parsed from leptonica/watershed.h

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

// #ifndef  LEPTONICA_WATERSHED_H
// #define  LEPTONICA_WATERSHED_H

/*
 *  watershed.h
 *
 *     Simple data structure to hold watershed data.
 *     All data here is owned by the L_WShed and must be freed.
 */

@Name("L_WShed") public static class L_WSHED extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public L_WSHED() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public L_WSHED(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public L_WSHED(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public L_WSHED position(int position) {
        return (L_WSHED)super.position(position);
    }

    public native PIX pixs(); public native L_WSHED pixs(PIX pixs);        /* clone of input 8 bpp pixs                */
    public native PIX pixm(); public native L_WSHED pixm(PIX pixm);        /* clone of input 1 bpp seed (marker) pixm  */
    public native @Cast("l_int32") int mindepth(); public native L_WSHED mindepth(int mindepth);    /* minimum depth allowed for a watershed    */
    public native PIX pixlab(); public native L_WSHED pixlab(PIX pixlab);      /* 16 bpp label pix                         */
    public native PIX pixt(); public native L_WSHED pixt(PIX pixt);        /* scratch pix for computing wshed regions  */
    public native Pointer lines8(int i); public native L_WSHED lines8(int i, Pointer lines8);
    @MemberGetter public native @Cast("void**") PointerPointer lines8();      /* line ptrs for pixs                       */
    public native Pointer linem1(int i); public native L_WSHED linem1(int i, Pointer linem1);
    @MemberGetter public native @Cast("void**") PointerPointer linem1();      /* line ptrs for pixm                       */
    public native Pointer linelab32(int i); public native L_WSHED linelab32(int i, Pointer linelab32);
    @MemberGetter public native @Cast("void**") PointerPointer linelab32();   /* line ptrs for pixlab                     */
    public native Pointer linet1(int i); public native L_WSHED linet1(int i, Pointer linet1);
    @MemberGetter public native @Cast("void**") PointerPointer linet1();      /* line ptrs for pixt                       */
    public native PIXA pixad(); public native L_WSHED pixad(PIXA pixad);       /* result: 1 bpp pixa of watersheds         */
    public native PTA ptas(); public native L_WSHED ptas(PTA ptas);        /* pta of initial seed pixels               */
    public native NUMA nasi(); public native L_WSHED nasi(NUMA nasi);        /* numa of seed indicators; 0 if completed  */
    public native NUMA nash(); public native L_WSHED nash(NUMA nash);        /* numa of initial seed heights             */
    public native NUMA namh(); public native L_WSHED namh(NUMA namh);        /* numa of initial minima heights           */
    public native NUMA nalevels(); public native L_WSHED nalevels(NUMA nalevels);    /* result: numa of watershed levels         */
    public native @Cast("l_int32") int nseeds(); public native L_WSHED nseeds(int nseeds);      /* number of seeds (markers)                */
    public native @Cast("l_int32") int nother(); public native L_WSHED nother(int nother);      /* number of minima different from seeds    */
    public native @Cast("l_int32*") IntPointer lut(); public native L_WSHED lut(IntPointer lut);         /* lut for pixel indices                    */
    public native NUMA links(int i); public native L_WSHED links(int i, NUMA links);
    @MemberGetter public native @Cast("Numa**") PointerPointer links();       /* back-links into lut, for updates         */
    public native @Cast("l_int32") int arraysize(); public native L_WSHED arraysize(int arraysize);   /* size of links array                      */
    public native @Cast("l_int32") int debug(); public native L_WSHED debug(int debug);       /* set to 1 for debug output                */
}

// #endif  /* LEPTONICA_WATERSHED_H */


// Parsed from leptonica/allheaders.h

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

public static native PIX pixCleanBackgroundToWhite( PIX pixs, PIX pixim, PIX pixg, @Cast("l_float32") float gamma, @Cast("l_int32") int blackval, @Cast("l_int32") int whiteval );
public static native PIX pixBackgroundNormSimple( PIX pixs, PIX pixim, PIX pixg );
public static native PIX pixBackgroundNorm( PIX pixs, PIX pixim, PIX pixg, @Cast("l_int32") int sx, @Cast("l_int32") int sy, @Cast("l_int32") int thresh, @Cast("l_int32") int mincount, @Cast("l_int32") int bgval, @Cast("l_int32") int smoothx, @Cast("l_int32") int smoothy );
public static native PIX pixBackgroundNormMorph( PIX pixs, PIX pixim, @Cast("l_int32") int reduction, @Cast("l_int32") int size, @Cast("l_int32") int bgval );
public static native @Cast("l_int32") int pixBackgroundNormGrayArray( PIX pixs, PIX pixim, @Cast("l_int32") int sx, @Cast("l_int32") int sy, @Cast("l_int32") int thresh, @Cast("l_int32") int mincount, @Cast("l_int32") int bgval, @Cast("l_int32") int smoothx, @Cast("l_int32") int smoothy, @Cast("PIX**") PointerPointer ppixd );
public static native @Cast("l_int32") int pixBackgroundNormGrayArray( PIX pixs, PIX pixim, @Cast("l_int32") int sx, @Cast("l_int32") int sy, @Cast("l_int32") int thresh, @Cast("l_int32") int mincount, @Cast("l_int32") int bgval, @Cast("l_int32") int smoothx, @Cast("l_int32") int smoothy, @ByPtrPtr PIX ppixd );
public static native @Cast("l_int32") int pixBackgroundNormRGBArrays( PIX pixs, PIX pixim, PIX pixg, @Cast("l_int32") int sx, @Cast("l_int32") int sy, @Cast("l_int32") int thresh, @Cast("l_int32") int mincount, @Cast("l_int32") int bgval, @Cast("l_int32") int smoothx, @Cast("l_int32") int smoothy, @Cast("PIX**") PointerPointer ppixr, @Cast("PIX**") PointerPointer ppixg, @Cast("PIX**") PointerPointer ppixb );
public static native @Cast("l_int32") int pixBackgroundNormRGBArrays( PIX pixs, PIX pixim, PIX pixg, @Cast("l_int32") int sx, @Cast("l_int32") int sy, @Cast("l_int32") int thresh, @Cast("l_int32") int mincount, @Cast("l_int32") int bgval, @Cast("l_int32") int smoothx, @Cast("l_int32") int smoothy, @ByPtrPtr PIX ppixr, @ByPtrPtr PIX ppixg, @ByPtrPtr PIX ppixb );
public static native @Cast("l_int32") int pixBackgroundNormGrayArrayMorph( PIX pixs, PIX pixim, @Cast("l_int32") int reduction, @Cast("l_int32") int size, @Cast("l_int32") int bgval, @Cast("PIX**") PointerPointer ppixd );
public static native @Cast("l_int32") int pixBackgroundNormGrayArrayMorph( PIX pixs, PIX pixim, @Cast("l_int32") int reduction, @Cast("l_int32") int size, @Cast("l_int32") int bgval, @ByPtrPtr PIX ppixd );
public static native @Cast("l_int32") int pixBackgroundNormRGBArraysMorph( PIX pixs, PIX pixim, @Cast("l_int32") int reduction, @Cast("l_int32") int size, @Cast("l_int32") int bgval, @Cast("PIX**") PointerPointer ppixr, @Cast("PIX**") PointerPointer ppixg, @Cast("PIX**") PointerPointer ppixb );
public static native @Cast("l_int32") int pixBackgroundNormRGBArraysMorph( PIX pixs, PIX pixim, @Cast("l_int32") int reduction, @Cast("l_int32") int size, @Cast("l_int32") int bgval, @ByPtrPtr PIX ppixr, @ByPtrPtr PIX ppixg, @ByPtrPtr PIX ppixb );
public static native @Cast("l_int32") int pixGetBackgroundGrayMap( PIX pixs, PIX pixim, @Cast("l_int32") int sx, @Cast("l_int32") int sy, @Cast("l_int32") int thresh, @Cast("l_int32") int mincount, @Cast("PIX**") PointerPointer ppixd );
public static native @Cast("l_int32") int pixGetBackgroundGrayMap( PIX pixs, PIX pixim, @Cast("l_int32") int sx, @Cast("l_int32") int sy, @Cast("l_int32") int thresh, @Cast("l_int32") int mincount, @ByPtrPtr PIX ppixd );
public static native @Cast("l_int32") int pixGetBackgroundRGBMap( PIX pixs, PIX pixim, PIX pixg, @Cast("l_int32") int sx, @Cast("l_int32") int sy, @Cast("l_int32") int thresh, @Cast("l_int32") int mincount, @Cast("PIX**") PointerPointer ppixmr, @Cast("PIX**") PointerPointer ppixmg, @Cast("PIX**") PointerPointer ppixmb );
public static native @Cast("l_int32") int pixGetBackgroundRGBMap( PIX pixs, PIX pixim, PIX pixg, @Cast("l_int32") int sx, @Cast("l_int32") int sy, @Cast("l_int32") int thresh, @Cast("l_int32") int mincount, @ByPtrPtr PIX ppixmr, @ByPtrPtr PIX ppixmg, @ByPtrPtr PIX ppixmb );
public static native @Cast("l_int32") int pixGetBackgroundGrayMapMorph( PIX pixs, PIX pixim, @Cast("l_int32") int reduction, @Cast("l_int32") int size, @Cast("PIX**") PointerPointer ppixm );
public static native @Cast("l_int32") int pixGetBackgroundGrayMapMorph( PIX pixs, PIX pixim, @Cast("l_int32") int reduction, @Cast("l_int32") int size, @ByPtrPtr PIX ppixm );
public static native @Cast("l_int32") int pixGetBackgroundRGBMapMorph( PIX pixs, PIX pixim, @Cast("l_int32") int reduction, @Cast("l_int32") int size, @Cast("PIX**") PointerPointer ppixmr, @Cast("PIX**") PointerPointer ppixmg, @Cast("PIX**") PointerPointer ppixmb );
public static native @Cast("l_int32") int pixGetBackgroundRGBMapMorph( PIX pixs, PIX pixim, @Cast("l_int32") int reduction, @Cast("l_int32") int size, @ByPtrPtr PIX ppixmr, @ByPtrPtr PIX ppixmg, @ByPtrPtr PIX ppixmb );
public static native @Cast("l_int32") int pixFillMapHoles( PIX pix, @Cast("l_int32") int nx, @Cast("l_int32") int ny, @Cast("l_int32") int filltype );
public static native PIX pixExtendByReplication( PIX pixs, @Cast("l_int32") int addw, @Cast("l_int32") int addh );
public static native @Cast("l_int32") int pixSmoothConnectedRegions( PIX pixs, PIX pixm, @Cast("l_int32") int factor );
public static native PIX pixGetInvBackgroundMap( PIX pixs, @Cast("l_int32") int bgval, @Cast("l_int32") int smoothx, @Cast("l_int32") int smoothy );
public static native PIX pixApplyInvBackgroundGrayMap( PIX pixs, PIX pixm, @Cast("l_int32") int sx, @Cast("l_int32") int sy );
public static native PIX pixApplyInvBackgroundRGBMap( PIX pixs, PIX pixmr, PIX pixmg, PIX pixmb, @Cast("l_int32") int sx, @Cast("l_int32") int sy );
public static native PIX pixApplyVariableGrayMap( PIX pixs, PIX pixg, @Cast("l_int32") int target );
public static native PIX pixGlobalNormRGB( PIX pixd, PIX pixs, @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_int32") int mapval );
public static native PIX pixGlobalNormNoSatRGB( PIX pixd, PIX pixs, @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_int32") int factor, @Cast("l_float32") float rank );
public static native @Cast("l_int32") int pixThresholdSpreadNorm( PIX pixs, @Cast("l_int32") int filtertype, @Cast("l_int32") int edgethresh, @Cast("l_int32") int smoothx, @Cast("l_int32") int smoothy, @Cast("l_float32") float gamma, @Cast("l_int32") int minval, @Cast("l_int32") int maxval, @Cast("l_int32") int targetthresh, @Cast("PIX**") PointerPointer ppixth, @Cast("PIX**") PointerPointer ppixb, @Cast("PIX**") PointerPointer ppixd );
public static native @Cast("l_int32") int pixThresholdSpreadNorm( PIX pixs, @Cast("l_int32") int filtertype, @Cast("l_int32") int edgethresh, @Cast("l_int32") int smoothx, @Cast("l_int32") int smoothy, @Cast("l_float32") float gamma, @Cast("l_int32") int minval, @Cast("l_int32") int maxval, @Cast("l_int32") int targetthresh, @ByPtrPtr PIX ppixth, @ByPtrPtr PIX ppixb, @ByPtrPtr PIX ppixd );
public static native PIX pixBackgroundNormFlex( PIX pixs, @Cast("l_int32") int sx, @Cast("l_int32") int sy, @Cast("l_int32") int smoothx, @Cast("l_int32") int smoothy, @Cast("l_int32") int delta );
public static native PIX pixContrastNorm( PIX pixd, PIX pixs, @Cast("l_int32") int sx, @Cast("l_int32") int sy, @Cast("l_int32") int mindiff, @Cast("l_int32") int smoothx, @Cast("l_int32") int smoothy );
public static native @Cast("l_int32") int pixMinMaxTiles( PIX pixs, @Cast("l_int32") int sx, @Cast("l_int32") int sy, @Cast("l_int32") int mindiff, @Cast("l_int32") int smoothx, @Cast("l_int32") int smoothy, @Cast("PIX**") PointerPointer ppixmin, @Cast("PIX**") PointerPointer ppixmax );
public static native @Cast("l_int32") int pixMinMaxTiles( PIX pixs, @Cast("l_int32") int sx, @Cast("l_int32") int sy, @Cast("l_int32") int mindiff, @Cast("l_int32") int smoothx, @Cast("l_int32") int smoothy, @ByPtrPtr PIX ppixmin, @ByPtrPtr PIX ppixmax );
public static native @Cast("l_int32") int pixSetLowContrast( PIX pixs1, PIX pixs2, @Cast("l_int32") int mindiff );
public static native PIX pixLinearTRCTiled( PIX pixd, PIX pixs, @Cast("l_int32") int sx, @Cast("l_int32") int sy, PIX pixmin, PIX pixmax );
public static native PIX pixAffineSampledPta( PIX pixs, PTA ptad, PTA ptas, @Cast("l_int32") int incolor );
public static native PIX pixAffineSampled( PIX pixs, @Cast("l_float32*") FloatPointer vc, @Cast("l_int32") int incolor );
public static native PIX pixAffineSampled( PIX pixs, @Cast("l_float32*") FloatBuffer vc, @Cast("l_int32") int incolor );
public static native PIX pixAffineSampled( PIX pixs, @Cast("l_float32*") float[] vc, @Cast("l_int32") int incolor );
public static native PIX pixAffinePta( PIX pixs, PTA ptad, PTA ptas, @Cast("l_int32") int incolor );
public static native PIX pixAffine( PIX pixs, @Cast("l_float32*") FloatPointer vc, @Cast("l_int32") int incolor );
public static native PIX pixAffine( PIX pixs, @Cast("l_float32*") FloatBuffer vc, @Cast("l_int32") int incolor );
public static native PIX pixAffine( PIX pixs, @Cast("l_float32*") float[] vc, @Cast("l_int32") int incolor );
public static native PIX pixAffinePtaColor( PIX pixs, PTA ptad, PTA ptas, @Cast("l_uint32") int colorval );
public static native PIX pixAffineColor( PIX pixs, @Cast("l_float32*") FloatPointer vc, @Cast("l_uint32") int colorval );
public static native PIX pixAffineColor( PIX pixs, @Cast("l_float32*") FloatBuffer vc, @Cast("l_uint32") int colorval );
public static native PIX pixAffineColor( PIX pixs, @Cast("l_float32*") float[] vc, @Cast("l_uint32") int colorval );
public static native PIX pixAffinePtaGray( PIX pixs, PTA ptad, PTA ptas, @Cast("l_uint8") byte grayval );
public static native PIX pixAffineGray( PIX pixs, @Cast("l_float32*") FloatPointer vc, @Cast("l_uint8") byte grayval );
public static native PIX pixAffineGray( PIX pixs, @Cast("l_float32*") FloatBuffer vc, @Cast("l_uint8") byte grayval );
public static native PIX pixAffineGray( PIX pixs, @Cast("l_float32*") float[] vc, @Cast("l_uint8") byte grayval );
public static native PIX pixAffinePtaWithAlpha( PIX pixs, PTA ptad, PTA ptas, PIX pixg, @Cast("l_float32") float fract, @Cast("l_int32") int border );
public static native @Cast("l_int32") int getAffineXformCoeffs( PTA ptas, PTA ptad, @Cast("l_float32**") PointerPointer pvc );
public static native @Cast("l_int32") int getAffineXformCoeffs( PTA ptas, PTA ptad, @Cast("l_float32**") @ByPtrPtr FloatPointer pvc );
public static native @Cast("l_int32") int getAffineXformCoeffs( PTA ptas, PTA ptad, @Cast("l_float32**") @ByPtrPtr FloatBuffer pvc );
public static native @Cast("l_int32") int getAffineXformCoeffs( PTA ptas, PTA ptad, @Cast("l_float32**") @ByPtrPtr float[] pvc );
public static native @Cast("l_int32") int affineInvertXform( @Cast("l_float32*") FloatPointer vc, @Cast("l_float32**") PointerPointer pvci );
public static native @Cast("l_int32") int affineInvertXform( @Cast("l_float32*") FloatPointer vc, @Cast("l_float32**") @ByPtrPtr FloatPointer pvci );
public static native @Cast("l_int32") int affineInvertXform( @Cast("l_float32*") FloatBuffer vc, @Cast("l_float32**") @ByPtrPtr FloatBuffer pvci );
public static native @Cast("l_int32") int affineInvertXform( @Cast("l_float32*") float[] vc, @Cast("l_float32**") @ByPtrPtr float[] pvci );
public static native @Cast("l_int32") int affineXformSampledPt( @Cast("l_float32*") FloatPointer vc, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32*") IntPointer pxp, @Cast("l_int32*") IntPointer pyp );
public static native @Cast("l_int32") int affineXformSampledPt( @Cast("l_float32*") FloatBuffer vc, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32*") IntBuffer pxp, @Cast("l_int32*") IntBuffer pyp );
public static native @Cast("l_int32") int affineXformSampledPt( @Cast("l_float32*") float[] vc, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32*") int[] pxp, @Cast("l_int32*") int[] pyp );
public static native @Cast("l_int32") int affineXformPt( @Cast("l_float32*") FloatPointer vc, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_float32*") FloatPointer pxp, @Cast("l_float32*") FloatPointer pyp );
public static native @Cast("l_int32") int affineXformPt( @Cast("l_float32*") FloatBuffer vc, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_float32*") FloatBuffer pxp, @Cast("l_float32*") FloatBuffer pyp );
public static native @Cast("l_int32") int affineXformPt( @Cast("l_float32*") float[] vc, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_float32*") float[] pxp, @Cast("l_float32*") float[] pyp );
public static native @Cast("l_int32") int linearInterpolatePixelColor( @Cast("l_uint32*") IntPointer datas, @Cast("l_int32") int wpls, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_float32") float x, @Cast("l_float32") float y, @Cast("l_uint32") int colorval, @Cast("l_uint32*") IntPointer pval );
public static native @Cast("l_int32") int linearInterpolatePixelColor( @Cast("l_uint32*") IntBuffer datas, @Cast("l_int32") int wpls, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_float32") float x, @Cast("l_float32") float y, @Cast("l_uint32") int colorval, @Cast("l_uint32*") IntBuffer pval );
public static native @Cast("l_int32") int linearInterpolatePixelColor( @Cast("l_uint32*") int[] datas, @Cast("l_int32") int wpls, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_float32") float x, @Cast("l_float32") float y, @Cast("l_uint32") int colorval, @Cast("l_uint32*") int[] pval );
public static native @Cast("l_int32") int linearInterpolatePixelGray( @Cast("l_uint32*") IntPointer datas, @Cast("l_int32") int wpls, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_float32") float x, @Cast("l_float32") float y, @Cast("l_int32") int grayval, @Cast("l_int32*") IntPointer pval );
public static native @Cast("l_int32") int linearInterpolatePixelGray( @Cast("l_uint32*") IntBuffer datas, @Cast("l_int32") int wpls, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_float32") float x, @Cast("l_float32") float y, @Cast("l_int32") int grayval, @Cast("l_int32*") IntBuffer pval );
public static native @Cast("l_int32") int linearInterpolatePixelGray( @Cast("l_uint32*") int[] datas, @Cast("l_int32") int wpls, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_float32") float x, @Cast("l_float32") float y, @Cast("l_int32") int grayval, @Cast("l_int32*") int[] pval );
public static native @Cast("l_int32") int gaussjordan( @Cast("l_float32**") PointerPointer a, @Cast("l_float32*") FloatPointer b, @Cast("l_int32") int n );
public static native @Cast("l_int32") int gaussjordan( @Cast("l_float32**") @ByPtrPtr FloatPointer a, @Cast("l_float32*") FloatPointer b, @Cast("l_int32") int n );
public static native @Cast("l_int32") int gaussjordan( @Cast("l_float32**") @ByPtrPtr FloatBuffer a, @Cast("l_float32*") FloatBuffer b, @Cast("l_int32") int n );
public static native @Cast("l_int32") int gaussjordan( @Cast("l_float32**") @ByPtrPtr float[] a, @Cast("l_float32*") float[] b, @Cast("l_int32") int n );
public static native PIX pixAffineSequential( PIX pixs, PTA ptad, PTA ptas, @Cast("l_int32") int bw, @Cast("l_int32") int bh );
public static native @Cast("l_float32*") FloatPointer createMatrix2dTranslate( @Cast("l_float32") float transx, @Cast("l_float32") float transy );
public static native @Cast("l_float32*") FloatPointer createMatrix2dScale( @Cast("l_float32") float scalex, @Cast("l_float32") float scaley );
public static native @Cast("l_float32*") FloatPointer createMatrix2dRotate( @Cast("l_float32") float xc, @Cast("l_float32") float yc, @Cast("l_float32") float angle );
public static native PTA ptaTranslate( PTA ptas, @Cast("l_float32") float transx, @Cast("l_float32") float transy );
public static native PTA ptaScale( PTA ptas, @Cast("l_float32") float scalex, @Cast("l_float32") float scaley );
public static native PTA ptaRotate( PTA ptas, @Cast("l_float32") float xc, @Cast("l_float32") float yc, @Cast("l_float32") float angle );
public static native BOXA boxaTranslate( BOXA boxas, @Cast("l_float32") float transx, @Cast("l_float32") float transy );
public static native BOXA boxaScale( BOXA boxas, @Cast("l_float32") float scalex, @Cast("l_float32") float scaley );
public static native BOXA boxaRotate( BOXA boxas, @Cast("l_float32") float xc, @Cast("l_float32") float yc, @Cast("l_float32") float angle );
public static native PTA ptaAffineTransform( PTA ptas, @Cast("l_float32*") FloatPointer mat );
public static native PTA ptaAffineTransform( PTA ptas, @Cast("l_float32*") FloatBuffer mat );
public static native PTA ptaAffineTransform( PTA ptas, @Cast("l_float32*") float[] mat );
public static native BOXA boxaAffineTransform( BOXA boxas, @Cast("l_float32*") FloatPointer mat );
public static native BOXA boxaAffineTransform( BOXA boxas, @Cast("l_float32*") FloatBuffer mat );
public static native BOXA boxaAffineTransform( BOXA boxas, @Cast("l_float32*") float[] mat );
public static native @Cast("l_int32") int l_productMatVec( @Cast("l_float32*") FloatPointer mat, @Cast("l_float32*") FloatPointer vecs, @Cast("l_float32*") FloatPointer vecd, @Cast("l_int32") int size );
public static native @Cast("l_int32") int l_productMatVec( @Cast("l_float32*") FloatBuffer mat, @Cast("l_float32*") FloatBuffer vecs, @Cast("l_float32*") FloatBuffer vecd, @Cast("l_int32") int size );
public static native @Cast("l_int32") int l_productMatVec( @Cast("l_float32*") float[] mat, @Cast("l_float32*") float[] vecs, @Cast("l_float32*") float[] vecd, @Cast("l_int32") int size );
public static native @Cast("l_int32") int l_productMat2( @Cast("l_float32*") FloatPointer mat1, @Cast("l_float32*") FloatPointer mat2, @Cast("l_float32*") FloatPointer matd, @Cast("l_int32") int size );
public static native @Cast("l_int32") int l_productMat2( @Cast("l_float32*") FloatBuffer mat1, @Cast("l_float32*") FloatBuffer mat2, @Cast("l_float32*") FloatBuffer matd, @Cast("l_int32") int size );
public static native @Cast("l_int32") int l_productMat2( @Cast("l_float32*") float[] mat1, @Cast("l_float32*") float[] mat2, @Cast("l_float32*") float[] matd, @Cast("l_int32") int size );
public static native @Cast("l_int32") int l_productMat3( @Cast("l_float32*") FloatPointer mat1, @Cast("l_float32*") FloatPointer mat2, @Cast("l_float32*") FloatPointer mat3, @Cast("l_float32*") FloatPointer matd, @Cast("l_int32") int size );
public static native @Cast("l_int32") int l_productMat3( @Cast("l_float32*") FloatBuffer mat1, @Cast("l_float32*") FloatBuffer mat2, @Cast("l_float32*") FloatBuffer mat3, @Cast("l_float32*") FloatBuffer matd, @Cast("l_int32") int size );
public static native @Cast("l_int32") int l_productMat3( @Cast("l_float32*") float[] mat1, @Cast("l_float32*") float[] mat2, @Cast("l_float32*") float[] mat3, @Cast("l_float32*") float[] matd, @Cast("l_int32") int size );
public static native @Cast("l_int32") int l_productMat4( @Cast("l_float32*") FloatPointer mat1, @Cast("l_float32*") FloatPointer mat2, @Cast("l_float32*") FloatPointer mat3, @Cast("l_float32*") FloatPointer mat4, @Cast("l_float32*") FloatPointer matd, @Cast("l_int32") int size );
public static native @Cast("l_int32") int l_productMat4( @Cast("l_float32*") FloatBuffer mat1, @Cast("l_float32*") FloatBuffer mat2, @Cast("l_float32*") FloatBuffer mat3, @Cast("l_float32*") FloatBuffer mat4, @Cast("l_float32*") FloatBuffer matd, @Cast("l_int32") int size );
public static native @Cast("l_int32") int l_productMat4( @Cast("l_float32*") float[] mat1, @Cast("l_float32*") float[] mat2, @Cast("l_float32*") float[] mat3, @Cast("l_float32*") float[] mat4, @Cast("l_float32*") float[] matd, @Cast("l_int32") int size );
public static native @Cast("l_int32") int l_getDataBit( Pointer line, @Cast("l_int32") int n );
public static native void l_setDataBit( Pointer line, @Cast("l_int32") int n );
public static native void l_clearDataBit( Pointer line, @Cast("l_int32") int n );
public static native void l_setDataBitVal( Pointer line, @Cast("l_int32") int n, @Cast("l_int32") int val );
public static native @Cast("l_int32") int l_getDataDibit( Pointer line, @Cast("l_int32") int n );
public static native void l_setDataDibit( Pointer line, @Cast("l_int32") int n, @Cast("l_int32") int val );
public static native void l_clearDataDibit( Pointer line, @Cast("l_int32") int n );
public static native @Cast("l_int32") int l_getDataQbit( Pointer line, @Cast("l_int32") int n );
public static native void l_setDataQbit( Pointer line, @Cast("l_int32") int n, @Cast("l_int32") int val );
public static native void l_clearDataQbit( Pointer line, @Cast("l_int32") int n );
public static native @Cast("l_int32") int l_getDataByte( Pointer line, @Cast("l_int32") int n );
public static native void l_setDataByte( Pointer line, @Cast("l_int32") int n, @Cast("l_int32") int val );
public static native @Cast("l_int32") int l_getDataTwoBytes( Pointer line, @Cast("l_int32") int n );
public static native void l_setDataTwoBytes( Pointer line, @Cast("l_int32") int n, @Cast("l_int32") int val );
public static native @Cast("l_int32") int l_getDataFourBytes( Pointer line, @Cast("l_int32") int n );
public static native void l_setDataFourBytes( Pointer line, @Cast("l_int32") int n, @Cast("l_int32") int val );
public static native @Cast("char*") BytePointer barcodeDispatchDecoder( @Cast("char*") BytePointer barstr, @Cast("l_int32") int format, @Cast("l_int32") int debugflag );
public static native @Cast("char*") ByteBuffer barcodeDispatchDecoder( @Cast("char*") ByteBuffer barstr, @Cast("l_int32") int format, @Cast("l_int32") int debugflag );
public static native @Cast("char*") byte[] barcodeDispatchDecoder( @Cast("char*") byte[] barstr, @Cast("l_int32") int format, @Cast("l_int32") int debugflag );
public static native @Cast("l_int32") int barcodeFormatIsSupported( @Cast("l_int32") int format );
public static native NUMA pixFindBaselines( PIX pixs, @Cast("PTA**") PointerPointer ppta, @Cast("l_int32") int debug );
public static native NUMA pixFindBaselines( PIX pixs, @ByPtrPtr PTA ppta, @Cast("l_int32") int debug );
public static native PIX pixDeskewLocal( PIX pixs, @Cast("l_int32") int nslices, @Cast("l_int32") int redsweep, @Cast("l_int32") int redsearch, @Cast("l_float32") float sweeprange, @Cast("l_float32") float sweepdelta, @Cast("l_float32") float minbsdelta );
public static native @Cast("l_int32") int pixGetLocalSkewTransform( PIX pixs, @Cast("l_int32") int nslices, @Cast("l_int32") int redsweep, @Cast("l_int32") int redsearch, @Cast("l_float32") float sweeprange, @Cast("l_float32") float sweepdelta, @Cast("l_float32") float minbsdelta, @Cast("PTA**") PointerPointer pptas, @Cast("PTA**") PointerPointer pptad );
public static native @Cast("l_int32") int pixGetLocalSkewTransform( PIX pixs, @Cast("l_int32") int nslices, @Cast("l_int32") int redsweep, @Cast("l_int32") int redsearch, @Cast("l_float32") float sweeprange, @Cast("l_float32") float sweepdelta, @Cast("l_float32") float minbsdelta, @ByPtrPtr PTA pptas, @ByPtrPtr PTA pptad );
public static native NUMA pixGetLocalSkewAngles( PIX pixs, @Cast("l_int32") int nslices, @Cast("l_int32") int redsweep, @Cast("l_int32") int redsearch, @Cast("l_float32") float sweeprange, @Cast("l_float32") float sweepdelta, @Cast("l_float32") float minbsdelta, @Cast("l_float32*") FloatPointer pa, @Cast("l_float32*") FloatPointer pb );
public static native NUMA pixGetLocalSkewAngles( PIX pixs, @Cast("l_int32") int nslices, @Cast("l_int32") int redsweep, @Cast("l_int32") int redsearch, @Cast("l_float32") float sweeprange, @Cast("l_float32") float sweepdelta, @Cast("l_float32") float minbsdelta, @Cast("l_float32*") FloatBuffer pa, @Cast("l_float32*") FloatBuffer pb );
public static native NUMA pixGetLocalSkewAngles( PIX pixs, @Cast("l_int32") int nslices, @Cast("l_int32") int redsweep, @Cast("l_int32") int redsearch, @Cast("l_float32") float sweeprange, @Cast("l_float32") float sweepdelta, @Cast("l_float32") float minbsdelta, @Cast("l_float32*") float[] pa, @Cast("l_float32*") float[] pb );
public static native BBUFFER bbufferCreate( @Cast("l_uint8*") BytePointer indata, @Cast("l_int32") int nalloc );
public static native BBUFFER bbufferCreate( @Cast("l_uint8*") ByteBuffer indata, @Cast("l_int32") int nalloc );
public static native BBUFFER bbufferCreate( @Cast("l_uint8*") byte[] indata, @Cast("l_int32") int nalloc );
public static native void bbufferDestroy( @Cast("BBUFFER**") PointerPointer pbb );
public static native void bbufferDestroy( @ByPtrPtr BBUFFER pbb );
public static native @Cast("l_uint8*") BytePointer bbufferDestroyAndSaveData( @Cast("BBUFFER**") PointerPointer pbb, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_uint8*") BytePointer bbufferDestroyAndSaveData( @ByPtrPtr BBUFFER pbb, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int bbufferRead( BBUFFER bb, @Cast("l_uint8*") BytePointer src, @Cast("l_int32") int nbytes );
public static native @Cast("l_int32") int bbufferRead( BBUFFER bb, @Cast("l_uint8*") ByteBuffer src, @Cast("l_int32") int nbytes );
public static native @Cast("l_int32") int bbufferRead( BBUFFER bb, @Cast("l_uint8*") byte[] src, @Cast("l_int32") int nbytes );
public static native @Cast("l_int32") int bbufferReadStream( BBUFFER bb, @Cast("FILE*") Pointer fp, @Cast("l_int32") int nbytes );
public static native @Cast("l_int32") int bbufferExtendArray( BBUFFER bb, @Cast("l_int32") int nbytes );
public static native @Cast("l_int32") int bbufferWrite( BBUFFER bb, @Cast("l_uint8*") BytePointer dest, @Cast("size_t") long nbytes, @Cast("size_t*") SizeTPointer pnout );
public static native @Cast("l_int32") int bbufferWrite( BBUFFER bb, @Cast("l_uint8*") ByteBuffer dest, @Cast("size_t") long nbytes, @Cast("size_t*") SizeTPointer pnout );
public static native @Cast("l_int32") int bbufferWrite( BBUFFER bb, @Cast("l_uint8*") byte[] dest, @Cast("size_t") long nbytes, @Cast("size_t*") SizeTPointer pnout );
public static native @Cast("l_int32") int bbufferWriteStream( BBUFFER bb, @Cast("FILE*") Pointer fp, @Cast("size_t") long nbytes, @Cast("size_t*") SizeTPointer pnout );
public static native PIX pixBilateral( PIX pixs, @Cast("l_float32") float spatial_stdev, @Cast("l_float32") float range_stdev, @Cast("l_int32") int ncomps, @Cast("l_int32") int reduction );
public static native PIX pixBilateralGray( PIX pixs, @Cast("l_float32") float spatial_stdev, @Cast("l_float32") float range_stdev, @Cast("l_int32") int ncomps, @Cast("l_int32") int reduction );
public static native PIX pixBilateralExact( PIX pixs, L_KERNEL spatial_kel, L_KERNEL range_kel );
public static native PIX pixBilateralGrayExact( PIX pixs, L_KERNEL spatial_kel, L_KERNEL range_kel );
public static native PIX pixBlockBilateralExact( PIX pixs, @Cast("l_float32") float spatial_stdev, @Cast("l_float32") float range_stdev );
public static native L_KERNEL makeRangeKernel( @Cast("l_float32") float range_stdev );
public static native PIX pixBilinearSampledPta( PIX pixs, PTA ptad, PTA ptas, @Cast("l_int32") int incolor );
public static native PIX pixBilinearSampled( PIX pixs, @Cast("l_float32*") FloatPointer vc, @Cast("l_int32") int incolor );
public static native PIX pixBilinearSampled( PIX pixs, @Cast("l_float32*") FloatBuffer vc, @Cast("l_int32") int incolor );
public static native PIX pixBilinearSampled( PIX pixs, @Cast("l_float32*") float[] vc, @Cast("l_int32") int incolor );
public static native PIX pixBilinearPta( PIX pixs, PTA ptad, PTA ptas, @Cast("l_int32") int incolor );
public static native PIX pixBilinear( PIX pixs, @Cast("l_float32*") FloatPointer vc, @Cast("l_int32") int incolor );
public static native PIX pixBilinear( PIX pixs, @Cast("l_float32*") FloatBuffer vc, @Cast("l_int32") int incolor );
public static native PIX pixBilinear( PIX pixs, @Cast("l_float32*") float[] vc, @Cast("l_int32") int incolor );
public static native PIX pixBilinearPtaColor( PIX pixs, PTA ptad, PTA ptas, @Cast("l_uint32") int colorval );
public static native PIX pixBilinearColor( PIX pixs, @Cast("l_float32*") FloatPointer vc, @Cast("l_uint32") int colorval );
public static native PIX pixBilinearColor( PIX pixs, @Cast("l_float32*") FloatBuffer vc, @Cast("l_uint32") int colorval );
public static native PIX pixBilinearColor( PIX pixs, @Cast("l_float32*") float[] vc, @Cast("l_uint32") int colorval );
public static native PIX pixBilinearPtaGray( PIX pixs, PTA ptad, PTA ptas, @Cast("l_uint8") byte grayval );
public static native PIX pixBilinearGray( PIX pixs, @Cast("l_float32*") FloatPointer vc, @Cast("l_uint8") byte grayval );
public static native PIX pixBilinearGray( PIX pixs, @Cast("l_float32*") FloatBuffer vc, @Cast("l_uint8") byte grayval );
public static native PIX pixBilinearGray( PIX pixs, @Cast("l_float32*") float[] vc, @Cast("l_uint8") byte grayval );
public static native PIX pixBilinearPtaWithAlpha( PIX pixs, PTA ptad, PTA ptas, PIX pixg, @Cast("l_float32") float fract, @Cast("l_int32") int border );
public static native @Cast("l_int32") int getBilinearXformCoeffs( PTA ptas, PTA ptad, @Cast("l_float32**") PointerPointer pvc );
public static native @Cast("l_int32") int getBilinearXformCoeffs( PTA ptas, PTA ptad, @Cast("l_float32**") @ByPtrPtr FloatPointer pvc );
public static native @Cast("l_int32") int getBilinearXformCoeffs( PTA ptas, PTA ptad, @Cast("l_float32**") @ByPtrPtr FloatBuffer pvc );
public static native @Cast("l_int32") int getBilinearXformCoeffs( PTA ptas, PTA ptad, @Cast("l_float32**") @ByPtrPtr float[] pvc );
public static native @Cast("l_int32") int bilinearXformSampledPt( @Cast("l_float32*") FloatPointer vc, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32*") IntPointer pxp, @Cast("l_int32*") IntPointer pyp );
public static native @Cast("l_int32") int bilinearXformSampledPt( @Cast("l_float32*") FloatBuffer vc, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32*") IntBuffer pxp, @Cast("l_int32*") IntBuffer pyp );
public static native @Cast("l_int32") int bilinearXformSampledPt( @Cast("l_float32*") float[] vc, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32*") int[] pxp, @Cast("l_int32*") int[] pyp );
public static native @Cast("l_int32") int bilinearXformPt( @Cast("l_float32*") FloatPointer vc, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_float32*") FloatPointer pxp, @Cast("l_float32*") FloatPointer pyp );
public static native @Cast("l_int32") int bilinearXformPt( @Cast("l_float32*") FloatBuffer vc, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_float32*") FloatBuffer pxp, @Cast("l_float32*") FloatBuffer pyp );
public static native @Cast("l_int32") int bilinearXformPt( @Cast("l_float32*") float[] vc, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_float32*") float[] pxp, @Cast("l_float32*") float[] pyp );
public static native @Cast("l_int32") int pixOtsuAdaptiveThreshold( PIX pixs, @Cast("l_int32") int sx, @Cast("l_int32") int sy, @Cast("l_int32") int smoothx, @Cast("l_int32") int smoothy, @Cast("l_float32") float scorefract, @Cast("PIX**") PointerPointer ppixth, @Cast("PIX**") PointerPointer ppixd );
public static native @Cast("l_int32") int pixOtsuAdaptiveThreshold( PIX pixs, @Cast("l_int32") int sx, @Cast("l_int32") int sy, @Cast("l_int32") int smoothx, @Cast("l_int32") int smoothy, @Cast("l_float32") float scorefract, @ByPtrPtr PIX ppixth, @ByPtrPtr PIX ppixd );
public static native PIX pixOtsuThreshOnBackgroundNorm( PIX pixs, PIX pixim, @Cast("l_int32") int sx, @Cast("l_int32") int sy, @Cast("l_int32") int thresh, @Cast("l_int32") int mincount, @Cast("l_int32") int bgval, @Cast("l_int32") int smoothx, @Cast("l_int32") int smoothy, @Cast("l_float32") float scorefract, @Cast("l_int32*") IntPointer pthresh );
public static native PIX pixOtsuThreshOnBackgroundNorm( PIX pixs, PIX pixim, @Cast("l_int32") int sx, @Cast("l_int32") int sy, @Cast("l_int32") int thresh, @Cast("l_int32") int mincount, @Cast("l_int32") int bgval, @Cast("l_int32") int smoothx, @Cast("l_int32") int smoothy, @Cast("l_float32") float scorefract, @Cast("l_int32*") IntBuffer pthresh );
public static native PIX pixOtsuThreshOnBackgroundNorm( PIX pixs, PIX pixim, @Cast("l_int32") int sx, @Cast("l_int32") int sy, @Cast("l_int32") int thresh, @Cast("l_int32") int mincount, @Cast("l_int32") int bgval, @Cast("l_int32") int smoothx, @Cast("l_int32") int smoothy, @Cast("l_float32") float scorefract, @Cast("l_int32*") int[] pthresh );
public static native PIX pixMaskedThreshOnBackgroundNorm( PIX pixs, PIX pixim, @Cast("l_int32") int sx, @Cast("l_int32") int sy, @Cast("l_int32") int thresh, @Cast("l_int32") int mincount, @Cast("l_int32") int smoothx, @Cast("l_int32") int smoothy, @Cast("l_float32") float scorefract, @Cast("l_int32*") IntPointer pthresh );
public static native PIX pixMaskedThreshOnBackgroundNorm( PIX pixs, PIX pixim, @Cast("l_int32") int sx, @Cast("l_int32") int sy, @Cast("l_int32") int thresh, @Cast("l_int32") int mincount, @Cast("l_int32") int smoothx, @Cast("l_int32") int smoothy, @Cast("l_float32") float scorefract, @Cast("l_int32*") IntBuffer pthresh );
public static native PIX pixMaskedThreshOnBackgroundNorm( PIX pixs, PIX pixim, @Cast("l_int32") int sx, @Cast("l_int32") int sy, @Cast("l_int32") int thresh, @Cast("l_int32") int mincount, @Cast("l_int32") int smoothx, @Cast("l_int32") int smoothy, @Cast("l_float32") float scorefract, @Cast("l_int32*") int[] pthresh );
public static native @Cast("l_int32") int pixSauvolaBinarizeTiled( PIX pixs, @Cast("l_int32") int whsize, @Cast("l_float32") float factor, @Cast("l_int32") int nx, @Cast("l_int32") int ny, @Cast("PIX**") PointerPointer ppixth, @Cast("PIX**") PointerPointer ppixd );
public static native @Cast("l_int32") int pixSauvolaBinarizeTiled( PIX pixs, @Cast("l_int32") int whsize, @Cast("l_float32") float factor, @Cast("l_int32") int nx, @Cast("l_int32") int ny, @ByPtrPtr PIX ppixth, @ByPtrPtr PIX ppixd );
public static native @Cast("l_int32") int pixSauvolaBinarize( PIX pixs, @Cast("l_int32") int whsize, @Cast("l_float32") float factor, @Cast("l_int32") int addborder, @Cast("PIX**") PointerPointer ppixm, @Cast("PIX**") PointerPointer ppixsd, @Cast("PIX**") PointerPointer ppixth, @Cast("PIX**") PointerPointer ppixd );
public static native @Cast("l_int32") int pixSauvolaBinarize( PIX pixs, @Cast("l_int32") int whsize, @Cast("l_float32") float factor, @Cast("l_int32") int addborder, @ByPtrPtr PIX ppixm, @ByPtrPtr PIX ppixsd, @ByPtrPtr PIX ppixth, @ByPtrPtr PIX ppixd );
public static native PIX pixSauvolaGetThreshold( PIX pixm, PIX pixms, @Cast("l_float32") float factor, @Cast("PIX**") PointerPointer ppixsd );
public static native PIX pixSauvolaGetThreshold( PIX pixm, PIX pixms, @Cast("l_float32") float factor, @ByPtrPtr PIX ppixsd );
public static native PIX pixApplyLocalThreshold( PIX pixs, PIX pixth, @Cast("l_int32") int redfactor );
public static native @Cast("l_int32") int pixThresholdByConnComp( PIX pixs, PIX pixm, @Cast("l_int32") int start, @Cast("l_int32") int end, @Cast("l_int32") int incr, @Cast("l_float32") float thresh48, @Cast("l_float32") float threshdiff, @Cast("l_int32*") IntPointer pglobthresh, @Cast("PIX**") PointerPointer ppixd, @Cast("l_int32") int debugflag );
public static native @Cast("l_int32") int pixThresholdByConnComp( PIX pixs, PIX pixm, @Cast("l_int32") int start, @Cast("l_int32") int end, @Cast("l_int32") int incr, @Cast("l_float32") float thresh48, @Cast("l_float32") float threshdiff, @Cast("l_int32*") IntPointer pglobthresh, @ByPtrPtr PIX ppixd, @Cast("l_int32") int debugflag );
public static native @Cast("l_int32") int pixThresholdByConnComp( PIX pixs, PIX pixm, @Cast("l_int32") int start, @Cast("l_int32") int end, @Cast("l_int32") int incr, @Cast("l_float32") float thresh48, @Cast("l_float32") float threshdiff, @Cast("l_int32*") IntBuffer pglobthresh, @ByPtrPtr PIX ppixd, @Cast("l_int32") int debugflag );
public static native @Cast("l_int32") int pixThresholdByConnComp( PIX pixs, PIX pixm, @Cast("l_int32") int start, @Cast("l_int32") int end, @Cast("l_int32") int incr, @Cast("l_float32") float thresh48, @Cast("l_float32") float threshdiff, @Cast("l_int32*") int[] pglobthresh, @ByPtrPtr PIX ppixd, @Cast("l_int32") int debugflag );
public static native PIX pixExpandBinaryReplicate( PIX pixs, @Cast("l_int32") int factor );
public static native PIX pixExpandBinaryPower2( PIX pixs, @Cast("l_int32") int factor );
public static native PIX pixReduceBinary2( PIX pixs, @Cast("l_uint8*") BytePointer intab );
public static native PIX pixReduceBinary2( PIX pixs, @Cast("l_uint8*") ByteBuffer intab );
public static native PIX pixReduceBinary2( PIX pixs, @Cast("l_uint8*") byte[] intab );
public static native PIX pixReduceRankBinaryCascade( PIX pixs, @Cast("l_int32") int level1, @Cast("l_int32") int level2, @Cast("l_int32") int level3, @Cast("l_int32") int level4 );
public static native PIX pixReduceRankBinary2( PIX pixs, @Cast("l_int32") int level, @Cast("l_uint8*") BytePointer intab );
public static native PIX pixReduceRankBinary2( PIX pixs, @Cast("l_int32") int level, @Cast("l_uint8*") ByteBuffer intab );
public static native PIX pixReduceRankBinary2( PIX pixs, @Cast("l_int32") int level, @Cast("l_uint8*") byte[] intab );
public static native @Cast("l_uint8*") BytePointer makeSubsampleTab2x( );
public static native PIX pixBlend( PIX pixs1, PIX pixs2, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_float32") float fract );
public static native PIX pixBlendMask( PIX pixd, PIX pixs1, PIX pixs2, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_float32") float fract, @Cast("l_int32") int type );
public static native PIX pixBlendGray( PIX pixd, PIX pixs1, PIX pixs2, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_float32") float fract, @Cast("l_int32") int type, @Cast("l_int32") int transparent, @Cast("l_uint32") int transpix );
public static native PIX pixBlendGrayInverse( PIX pixd, PIX pixs1, PIX pixs2, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_float32") float fract );
public static native PIX pixBlendColor( PIX pixd, PIX pixs1, PIX pixs2, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_float32") float fract, @Cast("l_int32") int transparent, @Cast("l_uint32") int transpix );
public static native PIX pixBlendColorByChannel( PIX pixd, PIX pixs1, PIX pixs2, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_float32") float rfract, @Cast("l_float32") float gfract, @Cast("l_float32") float bfract, @Cast("l_int32") int transparent, @Cast("l_uint32") int transpix );
public static native PIX pixBlendGrayAdapt( PIX pixd, PIX pixs1, PIX pixs2, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_float32") float fract, @Cast("l_int32") int shift );
public static native PIX pixFadeWithGray( PIX pixs, PIX pixb, @Cast("l_float32") float factor, @Cast("l_int32") int type );
public static native PIX pixBlendHardLight( PIX pixd, PIX pixs1, PIX pixs2, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_float32") float fract );
public static native @Cast("l_int32") int pixBlendCmap( PIX pixs, PIX pixb, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int sindex );
public static native PIX pixBlendWithGrayMask( PIX pixs1, PIX pixs2, PIX pixg, @Cast("l_int32") int x, @Cast("l_int32") int y );
public static native PIX pixBlendBackgroundToColor( PIX pixd, PIX pixs, BOX box, @Cast("l_uint32") int color, @Cast("l_float32") float gamma, @Cast("l_int32") int minval, @Cast("l_int32") int maxval );
public static native PIX pixMultiplyByColor( PIX pixd, PIX pixs, BOX box, @Cast("l_uint32") int color );
public static native PIX pixAlphaBlendUniform( PIX pixs, @Cast("l_uint32") int color );
public static native PIX pixAddAlphaToBlend( PIX pixs, @Cast("l_float32") float fract, @Cast("l_int32") int invert );
public static native PIX pixSetAlphaOverWhite( PIX pixs );
public static native L_BMF bmfCreate( @Cast("const char*") BytePointer dir, @Cast("l_int32") int fontsize );
public static native L_BMF bmfCreate( String dir, @Cast("l_int32") int fontsize );
public static native void bmfDestroy( @Cast("L_BMF**") PointerPointer pbmf );
public static native void bmfDestroy( @ByPtrPtr L_BMF pbmf );
public static native PIX bmfGetPix( L_BMF bmf, @Cast("char") byte chr );
public static native @Cast("l_int32") int bmfGetWidth( L_BMF bmf, @Cast("char") byte chr, @Cast("l_int32*") IntPointer pw );
public static native @Cast("l_int32") int bmfGetWidth( L_BMF bmf, @Cast("char") byte chr, @Cast("l_int32*") IntBuffer pw );
public static native @Cast("l_int32") int bmfGetWidth( L_BMF bmf, @Cast("char") byte chr, @Cast("l_int32*") int[] pw );
public static native @Cast("l_int32") int bmfGetBaseline( L_BMF bmf, @Cast("char") byte chr, @Cast("l_int32*") IntPointer pbaseline );
public static native @Cast("l_int32") int bmfGetBaseline( L_BMF bmf, @Cast("char") byte chr, @Cast("l_int32*") IntBuffer pbaseline );
public static native @Cast("l_int32") int bmfGetBaseline( L_BMF bmf, @Cast("char") byte chr, @Cast("l_int32*") int[] pbaseline );
public static native PIXA pixaGetFont( @Cast("const char*") BytePointer dir, @Cast("l_int32") int fontsize, @Cast("l_int32*") IntPointer pbl0, @Cast("l_int32*") IntPointer pbl1, @Cast("l_int32*") IntPointer pbl2 );
public static native PIXA pixaGetFont( String dir, @Cast("l_int32") int fontsize, @Cast("l_int32*") IntBuffer pbl0, @Cast("l_int32*") IntBuffer pbl1, @Cast("l_int32*") IntBuffer pbl2 );
public static native PIXA pixaGetFont( @Cast("const char*") BytePointer dir, @Cast("l_int32") int fontsize, @Cast("l_int32*") int[] pbl0, @Cast("l_int32*") int[] pbl1, @Cast("l_int32*") int[] pbl2 );
public static native PIXA pixaGetFont( String dir, @Cast("l_int32") int fontsize, @Cast("l_int32*") IntPointer pbl0, @Cast("l_int32*") IntPointer pbl1, @Cast("l_int32*") IntPointer pbl2 );
public static native PIXA pixaGetFont( @Cast("const char*") BytePointer dir, @Cast("l_int32") int fontsize, @Cast("l_int32*") IntBuffer pbl0, @Cast("l_int32*") IntBuffer pbl1, @Cast("l_int32*") IntBuffer pbl2 );
public static native PIXA pixaGetFont( String dir, @Cast("l_int32") int fontsize, @Cast("l_int32*") int[] pbl0, @Cast("l_int32*") int[] pbl1, @Cast("l_int32*") int[] pbl2 );
public static native @Cast("l_int32") int pixaSaveFont( @Cast("const char*") BytePointer indir, @Cast("const char*") BytePointer outdir, @Cast("l_int32") int fontsize );
public static native @Cast("l_int32") int pixaSaveFont( String indir, String outdir, @Cast("l_int32") int fontsize );
public static native PIXA pixaGenerateFontFromFile( @Cast("const char*") BytePointer dir, @Cast("l_int32") int fontsize, @Cast("l_int32*") IntPointer pbl0, @Cast("l_int32*") IntPointer pbl1, @Cast("l_int32*") IntPointer pbl2 );
public static native PIXA pixaGenerateFontFromFile( String dir, @Cast("l_int32") int fontsize, @Cast("l_int32*") IntBuffer pbl0, @Cast("l_int32*") IntBuffer pbl1, @Cast("l_int32*") IntBuffer pbl2 );
public static native PIXA pixaGenerateFontFromFile( @Cast("const char*") BytePointer dir, @Cast("l_int32") int fontsize, @Cast("l_int32*") int[] pbl0, @Cast("l_int32*") int[] pbl1, @Cast("l_int32*") int[] pbl2 );
public static native PIXA pixaGenerateFontFromFile( String dir, @Cast("l_int32") int fontsize, @Cast("l_int32*") IntPointer pbl0, @Cast("l_int32*") IntPointer pbl1, @Cast("l_int32*") IntPointer pbl2 );
public static native PIXA pixaGenerateFontFromFile( @Cast("const char*") BytePointer dir, @Cast("l_int32") int fontsize, @Cast("l_int32*") IntBuffer pbl0, @Cast("l_int32*") IntBuffer pbl1, @Cast("l_int32*") IntBuffer pbl2 );
public static native PIXA pixaGenerateFontFromFile( String dir, @Cast("l_int32") int fontsize, @Cast("l_int32*") int[] pbl0, @Cast("l_int32*") int[] pbl1, @Cast("l_int32*") int[] pbl2 );
public static native PIXA pixaGenerateFontFromString( @Cast("l_int32") int fontsize, @Cast("l_int32*") IntPointer pbl0, @Cast("l_int32*") IntPointer pbl1, @Cast("l_int32*") IntPointer pbl2 );
public static native PIXA pixaGenerateFontFromString( @Cast("l_int32") int fontsize, @Cast("l_int32*") IntBuffer pbl0, @Cast("l_int32*") IntBuffer pbl1, @Cast("l_int32*") IntBuffer pbl2 );
public static native PIXA pixaGenerateFontFromString( @Cast("l_int32") int fontsize, @Cast("l_int32*") int[] pbl0, @Cast("l_int32*") int[] pbl1, @Cast("l_int32*") int[] pbl2 );
public static native PIXA pixaGenerateFont( PIX pixs, @Cast("l_int32") int fontsize, @Cast("l_int32*") IntPointer pbl0, @Cast("l_int32*") IntPointer pbl1, @Cast("l_int32*") IntPointer pbl2 );
public static native PIXA pixaGenerateFont( PIX pixs, @Cast("l_int32") int fontsize, @Cast("l_int32*") IntBuffer pbl0, @Cast("l_int32*") IntBuffer pbl1, @Cast("l_int32*") IntBuffer pbl2 );
public static native PIXA pixaGenerateFont( PIX pixs, @Cast("l_int32") int fontsize, @Cast("l_int32*") int[] pbl0, @Cast("l_int32*") int[] pbl1, @Cast("l_int32*") int[] pbl2 );
public static native PIX pixReadStreamBmp( @Cast("FILE*") Pointer fp );
public static native @Cast("l_int32") int pixWriteStreamBmp( @Cast("FILE*") Pointer fp, PIX pix );
public static native PIX pixReadMemBmp( @Cast("const l_uint8*") BytePointer cdata, @Cast("size_t") long size );
public static native PIX pixReadMemBmp( @Cast("const l_uint8*") ByteBuffer cdata, @Cast("size_t") long size );
public static native PIX pixReadMemBmp( @Cast("const l_uint8*") byte[] cdata, @Cast("size_t") long size );
public static native @Cast("l_int32") int pixWriteMemBmp( @Cast("l_uint8**") PointerPointer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix );
public static native @Cast("l_int32") int pixWriteMemBmp( @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix );
public static native @Cast("l_int32") int pixWriteMemBmp( @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix );
public static native @Cast("l_int32") int pixWriteMemBmp( @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer psize, PIX pix );
public static native Pointer l_bootnum_gen(  );
public static native BOX boxCreate( @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int w, @Cast("l_int32") int h );
public static native BOX boxCreateValid( @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int w, @Cast("l_int32") int h );
public static native BOX boxCopy( BOX box );
public static native BOX boxClone( BOX box );
public static native void boxDestroy( @Cast("BOX**") PointerPointer pbox );
public static native void boxDestroy( @ByPtrPtr BOX pbox );
public static native @Cast("l_int32") int boxGetGeometry( BOX box, @Cast("l_int32*") IntPointer px, @Cast("l_int32*") IntPointer py, @Cast("l_int32*") IntPointer pw, @Cast("l_int32*") IntPointer ph );
public static native @Cast("l_int32") int boxGetGeometry( BOX box, @Cast("l_int32*") IntBuffer px, @Cast("l_int32*") IntBuffer py, @Cast("l_int32*") IntBuffer pw, @Cast("l_int32*") IntBuffer ph );
public static native @Cast("l_int32") int boxGetGeometry( BOX box, @Cast("l_int32*") int[] px, @Cast("l_int32*") int[] py, @Cast("l_int32*") int[] pw, @Cast("l_int32*") int[] ph );
public static native @Cast("l_int32") int boxSetGeometry( BOX box, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int w, @Cast("l_int32") int h );
public static native @Cast("l_int32") int boxGetSideLocation( BOX box, @Cast("l_int32") int side, @Cast("l_int32*") IntPointer ploc );
public static native @Cast("l_int32") int boxGetSideLocation( BOX box, @Cast("l_int32") int side, @Cast("l_int32*") IntBuffer ploc );
public static native @Cast("l_int32") int boxGetSideLocation( BOX box, @Cast("l_int32") int side, @Cast("l_int32*") int[] ploc );
public static native @Cast("l_int32") int boxGetRefcount( BOX box );
public static native @Cast("l_int32") int boxChangeRefcount( BOX box, @Cast("l_int32") int delta );
public static native @Cast("l_int32") int boxIsValid( BOX box, @Cast("l_int32*") IntPointer pvalid );
public static native @Cast("l_int32") int boxIsValid( BOX box, @Cast("l_int32*") IntBuffer pvalid );
public static native @Cast("l_int32") int boxIsValid( BOX box, @Cast("l_int32*") int[] pvalid );
public static native BOXA boxaCreate( @Cast("l_int32") int n );
public static native BOXA boxaCopy( BOXA boxa, @Cast("l_int32") int copyflag );
public static native void boxaDestroy( @Cast("BOXA**") PointerPointer pboxa );
public static native void boxaDestroy( @ByPtrPtr BOXA pboxa );
public static native @Cast("l_int32") int boxaAddBox( BOXA boxa, BOX box, @Cast("l_int32") int copyflag );
public static native @Cast("l_int32") int boxaExtendArray( BOXA boxa );
public static native @Cast("l_int32") int boxaExtendArrayToSize( BOXA boxa, @Cast("l_int32") int size );
public static native @Cast("l_int32") int boxaGetCount( BOXA boxa );
public static native @Cast("l_int32") int boxaGetValidCount( BOXA boxa );
public static native BOX boxaGetBox( BOXA boxa, @Cast("l_int32") int index, @Cast("l_int32") int accessflag );
public static native BOX boxaGetValidBox( BOXA boxa, @Cast("l_int32") int index, @Cast("l_int32") int accessflag );
public static native @Cast("l_int32") int boxaGetBoxGeometry( BOXA boxa, @Cast("l_int32") int index, @Cast("l_int32*") IntPointer px, @Cast("l_int32*") IntPointer py, @Cast("l_int32*") IntPointer pw, @Cast("l_int32*") IntPointer ph );
public static native @Cast("l_int32") int boxaGetBoxGeometry( BOXA boxa, @Cast("l_int32") int index, @Cast("l_int32*") IntBuffer px, @Cast("l_int32*") IntBuffer py, @Cast("l_int32*") IntBuffer pw, @Cast("l_int32*") IntBuffer ph );
public static native @Cast("l_int32") int boxaGetBoxGeometry( BOXA boxa, @Cast("l_int32") int index, @Cast("l_int32*") int[] px, @Cast("l_int32*") int[] py, @Cast("l_int32*") int[] pw, @Cast("l_int32*") int[] ph );
public static native @Cast("l_int32") int boxaIsFull( BOXA boxa, @Cast("l_int32*") IntPointer pfull );
public static native @Cast("l_int32") int boxaIsFull( BOXA boxa, @Cast("l_int32*") IntBuffer pfull );
public static native @Cast("l_int32") int boxaIsFull( BOXA boxa, @Cast("l_int32*") int[] pfull );
public static native @Cast("l_int32") int boxaReplaceBox( BOXA boxa, @Cast("l_int32") int index, BOX box );
public static native @Cast("l_int32") int boxaInsertBox( BOXA boxa, @Cast("l_int32") int index, BOX box );
public static native @Cast("l_int32") int boxaRemoveBox( BOXA boxa, @Cast("l_int32") int index );
public static native @Cast("l_int32") int boxaRemoveBoxAndSave( BOXA boxa, @Cast("l_int32") int index, @Cast("BOX**") PointerPointer pbox );
public static native @Cast("l_int32") int boxaRemoveBoxAndSave( BOXA boxa, @Cast("l_int32") int index, @ByPtrPtr BOX pbox );
public static native BOXA boxaSaveValid( BOXA boxas, @Cast("l_int32") int copyflag );
public static native @Cast("l_int32") int boxaInitFull( BOXA boxa, BOX box );
public static native @Cast("l_int32") int boxaClear( BOXA boxa );
public static native BOXAA boxaaCreate( @Cast("l_int32") int n );
public static native BOXAA boxaaCopy( BOXAA baas, @Cast("l_int32") int copyflag );
public static native void boxaaDestroy( @Cast("BOXAA**") PointerPointer pbaa );
public static native void boxaaDestroy( @ByPtrPtr BOXAA pbaa );
public static native @Cast("l_int32") int boxaaAddBoxa( BOXAA baa, BOXA ba, @Cast("l_int32") int copyflag );
public static native @Cast("l_int32") int boxaaExtendArray( BOXAA baa );
public static native @Cast("l_int32") int boxaaExtendArrayToSize( BOXAA baa, @Cast("l_int32") int size );
public static native @Cast("l_int32") int boxaaGetCount( BOXAA baa );
public static native @Cast("l_int32") int boxaaGetBoxCount( BOXAA baa );
public static native BOXA boxaaGetBoxa( BOXAA baa, @Cast("l_int32") int index, @Cast("l_int32") int accessflag );
public static native BOX boxaaGetBox( BOXAA baa, @Cast("l_int32") int iboxa, @Cast("l_int32") int ibox, @Cast("l_int32") int accessflag );
public static native @Cast("l_int32") int boxaaInitFull( BOXAA baa, BOXA boxa );
public static native @Cast("l_int32") int boxaaExtendWithInit( BOXAA baa, @Cast("l_int32") int maxindex, BOXA boxa );
public static native @Cast("l_int32") int boxaaReplaceBoxa( BOXAA baa, @Cast("l_int32") int index, BOXA boxa );
public static native @Cast("l_int32") int boxaaInsertBoxa( BOXAA baa, @Cast("l_int32") int index, BOXA boxa );
public static native @Cast("l_int32") int boxaaRemoveBoxa( BOXAA baa, @Cast("l_int32") int index );
public static native @Cast("l_int32") int boxaaAddBox( BOXAA baa, @Cast("l_int32") int index, BOX box, @Cast("l_int32") int accessflag );
public static native BOXAA boxaaReadFromFiles( @Cast("const char*") BytePointer dirname, @Cast("const char*") BytePointer substr, @Cast("l_int32") int first, @Cast("l_int32") int nfiles );
public static native BOXAA boxaaReadFromFiles( String dirname, String substr, @Cast("l_int32") int first, @Cast("l_int32") int nfiles );
public static native BOXAA boxaaRead( @Cast("const char*") BytePointer filename );
public static native BOXAA boxaaRead( String filename );
public static native BOXAA boxaaReadStream( @Cast("FILE*") Pointer fp );
public static native @Cast("l_int32") int boxaaWrite( @Cast("const char*") BytePointer filename, BOXAA baa );
public static native @Cast("l_int32") int boxaaWrite( String filename, BOXAA baa );
public static native @Cast("l_int32") int boxaaWriteStream( @Cast("FILE*") Pointer fp, BOXAA baa );
public static native BOXA boxaRead( @Cast("const char*") BytePointer filename );
public static native BOXA boxaRead( String filename );
public static native BOXA boxaReadStream( @Cast("FILE*") Pointer fp );
public static native BOXA boxaReadMem( @Cast("const l_uint8*") BytePointer data, @Cast("size_t") long size );
public static native BOXA boxaReadMem( @Cast("const l_uint8*") ByteBuffer data, @Cast("size_t") long size );
public static native BOXA boxaReadMem( @Cast("const l_uint8*") byte[] data, @Cast("size_t") long size );
public static native @Cast("l_int32") int boxaWrite( @Cast("const char*") BytePointer filename, BOXA boxa );
public static native @Cast("l_int32") int boxaWrite( String filename, BOXA boxa );
public static native @Cast("l_int32") int boxaWriteStream( @Cast("FILE*") Pointer fp, BOXA boxa );
public static native @Cast("l_int32") int boxaWriteMem( @Cast("l_uint8**") PointerPointer pdata, @Cast("size_t*") SizeTPointer psize, BOXA boxa );
public static native @Cast("l_int32") int boxaWriteMem( @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer psize, BOXA boxa );
public static native @Cast("l_int32") int boxaWriteMem( @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer psize, BOXA boxa );
public static native @Cast("l_int32") int boxaWriteMem( @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer psize, BOXA boxa );
public static native @Cast("l_int32") int boxPrintStreamInfo( @Cast("FILE*") Pointer fp, BOX box );
public static native @Cast("l_int32") int boxContains( BOX box1, BOX box2, @Cast("l_int32*") IntPointer presult );
public static native @Cast("l_int32") int boxContains( BOX box1, BOX box2, @Cast("l_int32*") IntBuffer presult );
public static native @Cast("l_int32") int boxContains( BOX box1, BOX box2, @Cast("l_int32*") int[] presult );
public static native @Cast("l_int32") int boxIntersects( BOX box1, BOX box2, @Cast("l_int32*") IntPointer presult );
public static native @Cast("l_int32") int boxIntersects( BOX box1, BOX box2, @Cast("l_int32*") IntBuffer presult );
public static native @Cast("l_int32") int boxIntersects( BOX box1, BOX box2, @Cast("l_int32*") int[] presult );
public static native BOXA boxaContainedInBox( BOXA boxas, BOX box );
public static native BOXA boxaIntersectsBox( BOXA boxas, BOX box );
public static native BOXA boxaClipToBox( BOXA boxas, BOX box );
public static native BOXA boxaCombineOverlaps( BOXA boxas );
public static native BOX boxOverlapRegion( BOX box1, BOX box2 );
public static native BOX boxBoundingRegion( BOX box1, BOX box2 );
public static native @Cast("l_int32") int boxOverlapFraction( BOX box1, BOX box2, @Cast("l_float32*") FloatPointer pfract );
public static native @Cast("l_int32") int boxOverlapFraction( BOX box1, BOX box2, @Cast("l_float32*") FloatBuffer pfract );
public static native @Cast("l_int32") int boxOverlapFraction( BOX box1, BOX box2, @Cast("l_float32*") float[] pfract );
public static native @Cast("l_int32") int boxOverlapArea( BOX box1, BOX box2, @Cast("l_int32*") IntPointer parea );
public static native @Cast("l_int32") int boxOverlapArea( BOX box1, BOX box2, @Cast("l_int32*") IntBuffer parea );
public static native @Cast("l_int32") int boxOverlapArea( BOX box1, BOX box2, @Cast("l_int32*") int[] parea );
public static native BOXA boxaHandleOverlaps( BOXA boxas, @Cast("l_int32") int op, @Cast("l_int32") int range, @Cast("l_float32") float min_overlap, @Cast("l_float32") float max_ratio, @Cast("NUMA**") PointerPointer pnamap );
public static native BOXA boxaHandleOverlaps( BOXA boxas, @Cast("l_int32") int op, @Cast("l_int32") int range, @Cast("l_float32") float min_overlap, @Cast("l_float32") float max_ratio, @ByPtrPtr NUMA pnamap );
public static native @Cast("l_int32") int boxSeparationDistance( BOX box1, BOX box2, @Cast("l_int32*") IntPointer ph_sep, @Cast("l_int32*") IntPointer pv_sep );
public static native @Cast("l_int32") int boxSeparationDistance( BOX box1, BOX box2, @Cast("l_int32*") IntBuffer ph_sep, @Cast("l_int32*") IntBuffer pv_sep );
public static native @Cast("l_int32") int boxSeparationDistance( BOX box1, BOX box2, @Cast("l_int32*") int[] ph_sep, @Cast("l_int32*") int[] pv_sep );
public static native @Cast("l_int32") int boxContainsPt( BOX box, @Cast("l_float32") float x, @Cast("l_float32") float y, @Cast("l_int32*") IntPointer pcontains );
public static native @Cast("l_int32") int boxContainsPt( BOX box, @Cast("l_float32") float x, @Cast("l_float32") float y, @Cast("l_int32*") IntBuffer pcontains );
public static native @Cast("l_int32") int boxContainsPt( BOX box, @Cast("l_float32") float x, @Cast("l_float32") float y, @Cast("l_int32*") int[] pcontains );
public static native BOX boxaGetNearestToPt( BOXA boxa, @Cast("l_int32") int x, @Cast("l_int32") int y );
public static native @Cast("l_int32") int boxGetCenter( BOX box, @Cast("l_float32*") FloatPointer pcx, @Cast("l_float32*") FloatPointer pcy );
public static native @Cast("l_int32") int boxGetCenter( BOX box, @Cast("l_float32*") FloatBuffer pcx, @Cast("l_float32*") FloatBuffer pcy );
public static native @Cast("l_int32") int boxGetCenter( BOX box, @Cast("l_float32*") float[] pcx, @Cast("l_float32*") float[] pcy );
public static native @Cast("l_int32") int boxIntersectByLine( BOX box, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_float32") float slope, @Cast("l_int32*") IntPointer px1, @Cast("l_int32*") IntPointer py1, @Cast("l_int32*") IntPointer px2, @Cast("l_int32*") IntPointer py2, @Cast("l_int32*") IntPointer pn );
public static native @Cast("l_int32") int boxIntersectByLine( BOX box, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_float32") float slope, @Cast("l_int32*") IntBuffer px1, @Cast("l_int32*") IntBuffer py1, @Cast("l_int32*") IntBuffer px2, @Cast("l_int32*") IntBuffer py2, @Cast("l_int32*") IntBuffer pn );
public static native @Cast("l_int32") int boxIntersectByLine( BOX box, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_float32") float slope, @Cast("l_int32*") int[] px1, @Cast("l_int32*") int[] py1, @Cast("l_int32*") int[] px2, @Cast("l_int32*") int[] py2, @Cast("l_int32*") int[] pn );
public static native BOX boxClipToRectangle( BOX box, @Cast("l_int32") int wi, @Cast("l_int32") int hi );
public static native @Cast("l_int32") int boxClipToRectangleParams( BOX box, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32*") IntPointer pxstart, @Cast("l_int32*") IntPointer pystart, @Cast("l_int32*") IntPointer pxend, @Cast("l_int32*") IntPointer pyend, @Cast("l_int32*") IntPointer pbw, @Cast("l_int32*") IntPointer pbh );
public static native @Cast("l_int32") int boxClipToRectangleParams( BOX box, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32*") IntBuffer pxstart, @Cast("l_int32*") IntBuffer pystart, @Cast("l_int32*") IntBuffer pxend, @Cast("l_int32*") IntBuffer pyend, @Cast("l_int32*") IntBuffer pbw, @Cast("l_int32*") IntBuffer pbh );
public static native @Cast("l_int32") int boxClipToRectangleParams( BOX box, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32*") int[] pxstart, @Cast("l_int32*") int[] pystart, @Cast("l_int32*") int[] pxend, @Cast("l_int32*") int[] pyend, @Cast("l_int32*") int[] pbw, @Cast("l_int32*") int[] pbh );
public static native BOX boxRelocateOneSide( BOX boxd, BOX boxs, @Cast("l_int32") int loc, @Cast("l_int32") int sideflag );
public static native BOX boxAdjustSides( BOX boxd, BOX boxs, @Cast("l_int32") int delleft, @Cast("l_int32") int delright, @Cast("l_int32") int deltop, @Cast("l_int32") int delbot );
public static native BOXA boxaSetSide( BOXA boxad, BOXA boxas, @Cast("l_int32") int side, @Cast("l_int32") int val, @Cast("l_int32") int thresh );
public static native BOXA boxaAdjustWidthToTarget( BOXA boxad, BOXA boxas, @Cast("l_int32") int sides, @Cast("l_int32") int target, @Cast("l_int32") int thresh );
public static native BOXA boxaAdjustHeightToTarget( BOXA boxad, BOXA boxas, @Cast("l_int32") int sides, @Cast("l_int32") int target, @Cast("l_int32") int thresh );
public static native @Cast("l_int32") int boxEqual( BOX box1, BOX box2, @Cast("l_int32*") IntPointer psame );
public static native @Cast("l_int32") int boxEqual( BOX box1, BOX box2, @Cast("l_int32*") IntBuffer psame );
public static native @Cast("l_int32") int boxEqual( BOX box1, BOX box2, @Cast("l_int32*") int[] psame );
public static native @Cast("l_int32") int boxaEqual( BOXA boxa1, BOXA boxa2, @Cast("l_int32") int maxdist, @Cast("NUMA**") PointerPointer pnaindex, @Cast("l_int32*") IntPointer psame );
public static native @Cast("l_int32") int boxaEqual( BOXA boxa1, BOXA boxa2, @Cast("l_int32") int maxdist, @ByPtrPtr NUMA pnaindex, @Cast("l_int32*") IntPointer psame );
public static native @Cast("l_int32") int boxaEqual( BOXA boxa1, BOXA boxa2, @Cast("l_int32") int maxdist, @ByPtrPtr NUMA pnaindex, @Cast("l_int32*") IntBuffer psame );
public static native @Cast("l_int32") int boxaEqual( BOXA boxa1, BOXA boxa2, @Cast("l_int32") int maxdist, @ByPtrPtr NUMA pnaindex, @Cast("l_int32*") int[] psame );
public static native @Cast("l_int32") int boxSimilar( BOX box1, BOX box2, @Cast("l_int32") int leftdiff, @Cast("l_int32") int rightdiff, @Cast("l_int32") int topdiff, @Cast("l_int32") int botdiff, @Cast("l_int32*") IntPointer psimilar );
public static native @Cast("l_int32") int boxSimilar( BOX box1, BOX box2, @Cast("l_int32") int leftdiff, @Cast("l_int32") int rightdiff, @Cast("l_int32") int topdiff, @Cast("l_int32") int botdiff, @Cast("l_int32*") IntBuffer psimilar );
public static native @Cast("l_int32") int boxSimilar( BOX box1, BOX box2, @Cast("l_int32") int leftdiff, @Cast("l_int32") int rightdiff, @Cast("l_int32") int topdiff, @Cast("l_int32") int botdiff, @Cast("l_int32*") int[] psimilar );
public static native @Cast("l_int32") int boxaSimilar( BOXA boxa1, BOXA boxa2, @Cast("l_int32") int leftdiff, @Cast("l_int32") int rightdiff, @Cast("l_int32") int topdiff, @Cast("l_int32") int botdiff, @Cast("l_int32") int debugflag, @Cast("l_int32*") IntPointer psimilar );
public static native @Cast("l_int32") int boxaSimilar( BOXA boxa1, BOXA boxa2, @Cast("l_int32") int leftdiff, @Cast("l_int32") int rightdiff, @Cast("l_int32") int topdiff, @Cast("l_int32") int botdiff, @Cast("l_int32") int debugflag, @Cast("l_int32*") IntBuffer psimilar );
public static native @Cast("l_int32") int boxaSimilar( BOXA boxa1, BOXA boxa2, @Cast("l_int32") int leftdiff, @Cast("l_int32") int rightdiff, @Cast("l_int32") int topdiff, @Cast("l_int32") int botdiff, @Cast("l_int32") int debugflag, @Cast("l_int32*") int[] psimilar );
public static native @Cast("l_int32") int boxaJoin( BOXA boxad, BOXA boxas, @Cast("l_int32") int istart, @Cast("l_int32") int iend );
public static native @Cast("l_int32") int boxaaJoin( BOXAA baad, BOXAA baas, @Cast("l_int32") int istart, @Cast("l_int32") int iend );
public static native @Cast("l_int32") int boxaSplitEvenOdd( BOXA boxa, @Cast("l_int32") int fillflag, @Cast("BOXA**") PointerPointer pboxae, @Cast("BOXA**") PointerPointer pboxao );
public static native @Cast("l_int32") int boxaSplitEvenOdd( BOXA boxa, @Cast("l_int32") int fillflag, @ByPtrPtr BOXA pboxae, @ByPtrPtr BOXA pboxao );
public static native BOXA boxaMergeEvenOdd( BOXA boxae, BOXA boxao, @Cast("l_int32") int fillflag );
public static native BOXA boxaTransform( BOXA boxas, @Cast("l_int32") int shiftx, @Cast("l_int32") int shifty, @Cast("l_float32") float scalex, @Cast("l_float32") float scaley );
public static native BOX boxTransform( BOX box, @Cast("l_int32") int shiftx, @Cast("l_int32") int shifty, @Cast("l_float32") float scalex, @Cast("l_float32") float scaley );
public static native BOXA boxaTransformOrdered( BOXA boxas, @Cast("l_int32") int shiftx, @Cast("l_int32") int shifty, @Cast("l_float32") float scalex, @Cast("l_float32") float scaley, @Cast("l_int32") int xcen, @Cast("l_int32") int ycen, @Cast("l_float32") float angle, @Cast("l_int32") int order );
public static native BOX boxTransformOrdered( BOX boxs, @Cast("l_int32") int shiftx, @Cast("l_int32") int shifty, @Cast("l_float32") float scalex, @Cast("l_float32") float scaley, @Cast("l_int32") int xcen, @Cast("l_int32") int ycen, @Cast("l_float32") float angle, @Cast("l_int32") int order );
public static native BOXA boxaRotateOrth( BOXA boxas, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int rotation );
public static native BOX boxRotateOrth( BOX box, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int rotation );
public static native BOXA boxaSort( BOXA boxas, @Cast("l_int32") int sorttype, @Cast("l_int32") int sortorder, @Cast("NUMA**") PointerPointer pnaindex );
public static native BOXA boxaSort( BOXA boxas, @Cast("l_int32") int sorttype, @Cast("l_int32") int sortorder, @ByPtrPtr NUMA pnaindex );
public static native BOXA boxaBinSort( BOXA boxas, @Cast("l_int32") int sorttype, @Cast("l_int32") int sortorder, @Cast("NUMA**") PointerPointer pnaindex );
public static native BOXA boxaBinSort( BOXA boxas, @Cast("l_int32") int sorttype, @Cast("l_int32") int sortorder, @ByPtrPtr NUMA pnaindex );
public static native BOXA boxaSortByIndex( BOXA boxas, NUMA naindex );
public static native BOXAA boxaSort2d( BOXA boxas, @Cast("NUMAA**") PointerPointer pnaad, @Cast("l_int32") int delta1, @Cast("l_int32") int delta2, @Cast("l_int32") int minh1 );
public static native BOXAA boxaSort2d( BOXA boxas, @ByPtrPtr NUMAA pnaad, @Cast("l_int32") int delta1, @Cast("l_int32") int delta2, @Cast("l_int32") int minh1 );
public static native BOXAA boxaSort2dByIndex( BOXA boxas, NUMAA naa );
public static native @Cast("l_int32") int boxaExtractAsNuma( BOXA boxa, @Cast("NUMA**") PointerPointer pnal, @Cast("NUMA**") PointerPointer pnat, @Cast("NUMA**") PointerPointer pnar, @Cast("NUMA**") PointerPointer pnab, @Cast("NUMA**") PointerPointer pnaw, @Cast("NUMA**") PointerPointer pnah, @Cast("l_int32") int keepinvalid );
public static native @Cast("l_int32") int boxaExtractAsNuma( BOXA boxa, @ByPtrPtr NUMA pnal, @ByPtrPtr NUMA pnat, @ByPtrPtr NUMA pnar, @ByPtrPtr NUMA pnab, @ByPtrPtr NUMA pnaw, @ByPtrPtr NUMA pnah, @Cast("l_int32") int keepinvalid );
public static native @Cast("l_int32") int boxaExtractAsPta( BOXA boxa, @Cast("PTA**") PointerPointer pptal, @Cast("PTA**") PointerPointer pptat, @Cast("PTA**") PointerPointer pptar, @Cast("PTA**") PointerPointer pptab, @Cast("PTA**") PointerPointer pptaw, @Cast("PTA**") PointerPointer pptah, @Cast("l_int32") int keepinvalid );
public static native @Cast("l_int32") int boxaExtractAsPta( BOXA boxa, @ByPtrPtr PTA pptal, @ByPtrPtr PTA pptat, @ByPtrPtr PTA pptar, @ByPtrPtr PTA pptab, @ByPtrPtr PTA pptaw, @ByPtrPtr PTA pptah, @Cast("l_int32") int keepinvalid );
public static native BOX boxaGetRankSize( BOXA boxa, @Cast("l_float32") float fract );
public static native BOX boxaGetMedian( BOXA boxa );
public static native @Cast("l_int32") int boxaGetAverageSize( BOXA boxa, @Cast("l_float32*") FloatPointer pw, @Cast("l_float32*") FloatPointer ph );
public static native @Cast("l_int32") int boxaGetAverageSize( BOXA boxa, @Cast("l_float32*") FloatBuffer pw, @Cast("l_float32*") FloatBuffer ph );
public static native @Cast("l_int32") int boxaGetAverageSize( BOXA boxa, @Cast("l_float32*") float[] pw, @Cast("l_float32*") float[] ph );
public static native @Cast("l_int32") int boxaaGetExtent( BOXAA baa, @Cast("l_int32*") IntPointer pw, @Cast("l_int32*") IntPointer ph, @Cast("BOX**") PointerPointer pbox, @Cast("BOXA**") PointerPointer pboxa );
public static native @Cast("l_int32") int boxaaGetExtent( BOXAA baa, @Cast("l_int32*") IntPointer pw, @Cast("l_int32*") IntPointer ph, @ByPtrPtr BOX pbox, @ByPtrPtr BOXA pboxa );
public static native @Cast("l_int32") int boxaaGetExtent( BOXAA baa, @Cast("l_int32*") IntBuffer pw, @Cast("l_int32*") IntBuffer ph, @ByPtrPtr BOX pbox, @ByPtrPtr BOXA pboxa );
public static native @Cast("l_int32") int boxaaGetExtent( BOXAA baa, @Cast("l_int32*") int[] pw, @Cast("l_int32*") int[] ph, @ByPtrPtr BOX pbox, @ByPtrPtr BOXA pboxa );
public static native BOXA boxaaFlattenToBoxa( BOXAA baa, @Cast("NUMA**") PointerPointer pnaindex, @Cast("l_int32") int copyflag );
public static native BOXA boxaaFlattenToBoxa( BOXAA baa, @ByPtrPtr NUMA pnaindex, @Cast("l_int32") int copyflag );
public static native BOXA boxaaFlattenAligned( BOXAA baa, @Cast("l_int32") int num, BOX fillerbox, @Cast("l_int32") int copyflag );
public static native BOXAA boxaEncapsulateAligned( BOXA boxa, @Cast("l_int32") int num, @Cast("l_int32") int copyflag );
public static native @Cast("l_int32") int boxaaAlignBox( BOXAA baa, BOX box, @Cast("l_int32") int delta, @Cast("l_int32*") IntPointer pindex );
public static native @Cast("l_int32") int boxaaAlignBox( BOXAA baa, BOX box, @Cast("l_int32") int delta, @Cast("l_int32*") IntBuffer pindex );
public static native @Cast("l_int32") int boxaaAlignBox( BOXAA baa, BOX box, @Cast("l_int32") int delta, @Cast("l_int32*") int[] pindex );
public static native PIX pixMaskConnComp( PIX pixs, @Cast("l_int32") int connectivity, @Cast("BOXA**") PointerPointer pboxa );
public static native PIX pixMaskConnComp( PIX pixs, @Cast("l_int32") int connectivity, @ByPtrPtr BOXA pboxa );
public static native PIX pixMaskBoxa( PIX pixd, PIX pixs, BOXA boxa, @Cast("l_int32") int op );
public static native PIX pixPaintBoxa( PIX pixs, BOXA boxa, @Cast("l_uint32") int val );
public static native PIX pixSetBlackOrWhiteBoxa( PIX pixs, BOXA boxa, @Cast("l_int32") int op );
public static native PIX pixPaintBoxaRandom( PIX pixs, BOXA boxa );
public static native PIX pixBlendBoxaRandom( PIX pixs, BOXA boxa, @Cast("l_float32") float fract );
public static native PIX pixDrawBoxa( PIX pixs, BOXA boxa, @Cast("l_int32") int width, @Cast("l_uint32") int val );
public static native PIX pixDrawBoxaRandom( PIX pixs, BOXA boxa, @Cast("l_int32") int width );
public static native PIX boxaaDisplay( BOXAA baa, @Cast("l_int32") int linewba, @Cast("l_int32") int linewb, @Cast("l_uint32") int colorba, @Cast("l_uint32") int colorb, @Cast("l_int32") int w, @Cast("l_int32") int h );
public static native BOXA pixSplitIntoBoxa( PIX pixs, @Cast("l_int32") int minsum, @Cast("l_int32") int skipdist, @Cast("l_int32") int delta, @Cast("l_int32") int maxbg, @Cast("l_int32") int maxcomps, @Cast("l_int32") int remainder );
public static native BOXA pixSplitComponentIntoBoxa( PIX pix, BOX box, @Cast("l_int32") int minsum, @Cast("l_int32") int skipdist, @Cast("l_int32") int delta, @Cast("l_int32") int maxbg, @Cast("l_int32") int maxcomps, @Cast("l_int32") int remainder );
public static native BOXA makeMosaicStrips( @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int direction, @Cast("l_int32") int size );
public static native @Cast("l_int32") int boxaCompareRegions( BOXA boxa1, BOXA boxa2, @Cast("l_int32") int areathresh, @Cast("l_int32*") IntPointer pnsame, @Cast("l_float32*") FloatPointer pdiffarea, @Cast("l_float32*") FloatPointer pdiffxor, @Cast("PIX**") PointerPointer ppixdb );
public static native @Cast("l_int32") int boxaCompareRegions( BOXA boxa1, BOXA boxa2, @Cast("l_int32") int areathresh, @Cast("l_int32*") IntPointer pnsame, @Cast("l_float32*") FloatPointer pdiffarea, @Cast("l_float32*") FloatPointer pdiffxor, @ByPtrPtr PIX ppixdb );
public static native @Cast("l_int32") int boxaCompareRegions( BOXA boxa1, BOXA boxa2, @Cast("l_int32") int areathresh, @Cast("l_int32*") IntBuffer pnsame, @Cast("l_float32*") FloatBuffer pdiffarea, @Cast("l_float32*") FloatBuffer pdiffxor, @ByPtrPtr PIX ppixdb );
public static native @Cast("l_int32") int boxaCompareRegions( BOXA boxa1, BOXA boxa2, @Cast("l_int32") int areathresh, @Cast("l_int32*") int[] pnsame, @Cast("l_float32*") float[] pdiffarea, @Cast("l_float32*") float[] pdiffxor, @ByPtrPtr PIX ppixdb );
public static native BOXA boxaSelectRange( BOXA boxas, @Cast("l_int32") int first, @Cast("l_int32") int last, @Cast("l_int32") int copyflag );
public static native BOXAA boxaaSelectRange( BOXAA baas, @Cast("l_int32") int first, @Cast("l_int32") int last, @Cast("l_int32") int copyflag );
public static native BOXA boxaSelectBySize( BOXA boxas, @Cast("l_int32") int width, @Cast("l_int32") int height, @Cast("l_int32") int type, @Cast("l_int32") int relation, @Cast("l_int32*") IntPointer pchanged );
public static native BOXA boxaSelectBySize( BOXA boxas, @Cast("l_int32") int width, @Cast("l_int32") int height, @Cast("l_int32") int type, @Cast("l_int32") int relation, @Cast("l_int32*") IntBuffer pchanged );
public static native BOXA boxaSelectBySize( BOXA boxas, @Cast("l_int32") int width, @Cast("l_int32") int height, @Cast("l_int32") int type, @Cast("l_int32") int relation, @Cast("l_int32*") int[] pchanged );
public static native NUMA boxaMakeSizeIndicator( BOXA boxa, @Cast("l_int32") int width, @Cast("l_int32") int height, @Cast("l_int32") int type, @Cast("l_int32") int relation );
public static native BOXA boxaSelectByArea( BOXA boxas, @Cast("l_int32") int area, @Cast("l_int32") int relation, @Cast("l_int32*") IntPointer pchanged );
public static native BOXA boxaSelectByArea( BOXA boxas, @Cast("l_int32") int area, @Cast("l_int32") int relation, @Cast("l_int32*") IntBuffer pchanged );
public static native BOXA boxaSelectByArea( BOXA boxas, @Cast("l_int32") int area, @Cast("l_int32") int relation, @Cast("l_int32*") int[] pchanged );
public static native NUMA boxaMakeAreaIndicator( BOXA boxa, @Cast("l_int32") int area, @Cast("l_int32") int relation );
public static native BOXA boxaSelectWithIndicator( BOXA boxas, NUMA na, @Cast("l_int32*") IntPointer pchanged );
public static native BOXA boxaSelectWithIndicator( BOXA boxas, NUMA na, @Cast("l_int32*") IntBuffer pchanged );
public static native BOXA boxaSelectWithIndicator( BOXA boxas, NUMA na, @Cast("l_int32*") int[] pchanged );
public static native BOXA boxaPermutePseudorandom( BOXA boxas );
public static native BOXA boxaPermuteRandom( BOXA boxad, BOXA boxas );
public static native @Cast("l_int32") int boxaSwapBoxes( BOXA boxa, @Cast("l_int32") int i, @Cast("l_int32") int j );
public static native PTA boxaConvertToPta( BOXA boxa, @Cast("l_int32") int ncorners );
public static native BOXA ptaConvertToBoxa( PTA pta, @Cast("l_int32") int ncorners );
public static native PTA boxConvertToPta( BOX box, @Cast("l_int32") int ncorners );
public static native BOX ptaConvertToBox( PTA pta );
public static native BOXA boxaSmoothSequenceLS( BOXA boxas, @Cast("l_float32") float factor, @Cast("l_int32") int subflag, @Cast("l_int32") int maxdiff, @Cast("l_int32") int debug );
public static native BOXA boxaSmoothSequenceMedian( BOXA boxas, @Cast("l_int32") int halfwin, @Cast("l_int32") int subflag, @Cast("l_int32") int maxdiff, @Cast("l_int32") int debug );
public static native BOXA boxaLinearFit( BOXA boxas, @Cast("l_float32") float factor, @Cast("l_int32") int debug );
public static native BOXA boxaWindowedMedian( BOXA boxas, @Cast("l_int32") int halfwin, @Cast("l_int32") int debug );
public static native BOXA boxaModifyWithBoxa( BOXA boxas, BOXA boxam, @Cast("l_int32") int subflag, @Cast("l_int32") int maxdiff );
public static native BOXA boxaConstrainSize( BOXA boxas, @Cast("l_int32") int width, @Cast("l_int32") int widthflag, @Cast("l_int32") int height, @Cast("l_int32") int heightflag );
public static native BOXA boxaReconcileEvenOddHeight( BOXA boxas, @Cast("l_int32") int sides, @Cast("l_int32") int delh, @Cast("l_int32") int op, @Cast("l_float32") float factor );
public static native @Cast("l_int32") int boxaPlotSides( BOXA boxa, @Cast("const char*") BytePointer plotname, @Cast("NUMA**") PointerPointer pnal, @Cast("NUMA**") PointerPointer pnat, @Cast("NUMA**") PointerPointer pnar, @Cast("NUMA**") PointerPointer pnab, @Cast("l_int32") int outformat );
public static native @Cast("l_int32") int boxaPlotSides( BOXA boxa, @Cast("const char*") BytePointer plotname, @ByPtrPtr NUMA pnal, @ByPtrPtr NUMA pnat, @ByPtrPtr NUMA pnar, @ByPtrPtr NUMA pnab, @Cast("l_int32") int outformat );
public static native @Cast("l_int32") int boxaPlotSides( BOXA boxa, String plotname, @ByPtrPtr NUMA pnal, @ByPtrPtr NUMA pnat, @ByPtrPtr NUMA pnar, @ByPtrPtr NUMA pnab, @Cast("l_int32") int outformat );
public static native BOXA boxaFillSequence( BOXA boxas, @Cast("l_int32") int useflag, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int boxaGetExtent( BOXA boxa, @Cast("l_int32*") IntPointer pw, @Cast("l_int32*") IntPointer ph, @Cast("BOX**") PointerPointer pbox );
public static native @Cast("l_int32") int boxaGetExtent( BOXA boxa, @Cast("l_int32*") IntPointer pw, @Cast("l_int32*") IntPointer ph, @ByPtrPtr BOX pbox );
public static native @Cast("l_int32") int boxaGetExtent( BOXA boxa, @Cast("l_int32*") IntBuffer pw, @Cast("l_int32*") IntBuffer ph, @ByPtrPtr BOX pbox );
public static native @Cast("l_int32") int boxaGetExtent( BOXA boxa, @Cast("l_int32*") int[] pw, @Cast("l_int32*") int[] ph, @ByPtrPtr BOX pbox );
public static native @Cast("l_int32") int boxaGetCoverage( BOXA boxa, @Cast("l_int32") int wc, @Cast("l_int32") int hc, @Cast("l_int32") int exactflag, @Cast("l_float32*") FloatPointer pfract );
public static native @Cast("l_int32") int boxaGetCoverage( BOXA boxa, @Cast("l_int32") int wc, @Cast("l_int32") int hc, @Cast("l_int32") int exactflag, @Cast("l_float32*") FloatBuffer pfract );
public static native @Cast("l_int32") int boxaGetCoverage( BOXA boxa, @Cast("l_int32") int wc, @Cast("l_int32") int hc, @Cast("l_int32") int exactflag, @Cast("l_float32*") float[] pfract );
public static native @Cast("l_int32") int boxaaSizeRange( BOXAA baa, @Cast("l_int32*") IntPointer pminw, @Cast("l_int32*") IntPointer pminh, @Cast("l_int32*") IntPointer pmaxw, @Cast("l_int32*") IntPointer pmaxh );
public static native @Cast("l_int32") int boxaaSizeRange( BOXAA baa, @Cast("l_int32*") IntBuffer pminw, @Cast("l_int32*") IntBuffer pminh, @Cast("l_int32*") IntBuffer pmaxw, @Cast("l_int32*") IntBuffer pmaxh );
public static native @Cast("l_int32") int boxaaSizeRange( BOXAA baa, @Cast("l_int32*") int[] pminw, @Cast("l_int32*") int[] pminh, @Cast("l_int32*") int[] pmaxw, @Cast("l_int32*") int[] pmaxh );
public static native @Cast("l_int32") int boxaSizeRange( BOXA boxa, @Cast("l_int32*") IntPointer pminw, @Cast("l_int32*") IntPointer pminh, @Cast("l_int32*") IntPointer pmaxw, @Cast("l_int32*") IntPointer pmaxh );
public static native @Cast("l_int32") int boxaSizeRange( BOXA boxa, @Cast("l_int32*") IntBuffer pminw, @Cast("l_int32*") IntBuffer pminh, @Cast("l_int32*") IntBuffer pmaxw, @Cast("l_int32*") IntBuffer pmaxh );
public static native @Cast("l_int32") int boxaSizeRange( BOXA boxa, @Cast("l_int32*") int[] pminw, @Cast("l_int32*") int[] pminh, @Cast("l_int32*") int[] pmaxw, @Cast("l_int32*") int[] pmaxh );
public static native @Cast("l_int32") int boxaLocationRange( BOXA boxa, @Cast("l_int32*") IntPointer pminx, @Cast("l_int32*") IntPointer pminy, @Cast("l_int32*") IntPointer pmaxx, @Cast("l_int32*") IntPointer pmaxy );
public static native @Cast("l_int32") int boxaLocationRange( BOXA boxa, @Cast("l_int32*") IntBuffer pminx, @Cast("l_int32*") IntBuffer pminy, @Cast("l_int32*") IntBuffer pmaxx, @Cast("l_int32*") IntBuffer pmaxy );
public static native @Cast("l_int32") int boxaLocationRange( BOXA boxa, @Cast("l_int32*") int[] pminx, @Cast("l_int32*") int[] pminy, @Cast("l_int32*") int[] pmaxx, @Cast("l_int32*") int[] pmaxy );
public static native @Cast("l_int32") int boxaGetArea( BOXA boxa, @Cast("l_int32*") IntPointer parea );
public static native @Cast("l_int32") int boxaGetArea( BOXA boxa, @Cast("l_int32*") IntBuffer parea );
public static native @Cast("l_int32") int boxaGetArea( BOXA boxa, @Cast("l_int32*") int[] parea );
public static native PIX boxaDisplayTiled( BOXA boxas, PIXA pixa, @Cast("l_int32") int maxwidth, @Cast("l_int32") int linewidth, @Cast("l_float32") float scalefactor, @Cast("l_int32") int background, @Cast("l_int32") int spacing, @Cast("l_int32") int border, @Cast("const char*") BytePointer fontdir );
public static native PIX boxaDisplayTiled( BOXA boxas, PIXA pixa, @Cast("l_int32") int maxwidth, @Cast("l_int32") int linewidth, @Cast("l_float32") float scalefactor, @Cast("l_int32") int background, @Cast("l_int32") int spacing, @Cast("l_int32") int border, String fontdir );
public static native L_BYTEA l_byteaCreate( @Cast("size_t") long nbytes );
public static native L_BYTEA l_byteaInitFromMem( @Cast("l_uint8*") BytePointer data, @Cast("size_t") long size );
public static native L_BYTEA l_byteaInitFromMem( @Cast("l_uint8*") ByteBuffer data, @Cast("size_t") long size );
public static native L_BYTEA l_byteaInitFromMem( @Cast("l_uint8*") byte[] data, @Cast("size_t") long size );
public static native L_BYTEA l_byteaInitFromFile( @Cast("const char*") BytePointer fname );
public static native L_BYTEA l_byteaInitFromFile( String fname );
public static native L_BYTEA l_byteaInitFromStream( @Cast("FILE*") Pointer fp );
public static native L_BYTEA l_byteaCopy( L_BYTEA bas, @Cast("l_int32") int copyflag );
public static native void l_byteaDestroy( @Cast("L_BYTEA**") PointerPointer pba );
public static native void l_byteaDestroy( @ByPtrPtr L_BYTEA pba );
public static native @Cast("size_t") long l_byteaGetSize( L_BYTEA ba );
public static native @Cast("l_uint8*") BytePointer l_byteaGetData( L_BYTEA ba, @Cast("size_t*") SizeTPointer psize );
public static native @Cast("l_uint8*") BytePointer l_byteaCopyData( L_BYTEA ba, @Cast("size_t*") SizeTPointer psize );
public static native @Cast("l_int32") int l_byteaAppendData( L_BYTEA ba, @Cast("l_uint8*") BytePointer newdata, @Cast("size_t") long newbytes );
public static native @Cast("l_int32") int l_byteaAppendData( L_BYTEA ba, @Cast("l_uint8*") ByteBuffer newdata, @Cast("size_t") long newbytes );
public static native @Cast("l_int32") int l_byteaAppendData( L_BYTEA ba, @Cast("l_uint8*") byte[] newdata, @Cast("size_t") long newbytes );
public static native @Cast("l_int32") int l_byteaAppendString( L_BYTEA ba, @Cast("char*") BytePointer str );
public static native @Cast("l_int32") int l_byteaAppendString( L_BYTEA ba, @Cast("char*") ByteBuffer str );
public static native @Cast("l_int32") int l_byteaAppendString( L_BYTEA ba, @Cast("char*") byte[] str );
public static native @Cast("l_int32") int l_byteaJoin( L_BYTEA ba1, @Cast("L_BYTEA**") PointerPointer pba2 );
public static native @Cast("l_int32") int l_byteaJoin( L_BYTEA ba1, @ByPtrPtr L_BYTEA pba2 );
public static native @Cast("l_int32") int l_byteaSplit( L_BYTEA ba1, @Cast("size_t") long splitloc, @Cast("L_BYTEA**") PointerPointer pba2 );
public static native @Cast("l_int32") int l_byteaSplit( L_BYTEA ba1, @Cast("size_t") long splitloc, @ByPtrPtr L_BYTEA pba2 );
public static native @Cast("l_int32") int l_byteaFindEachSequence( L_BYTEA ba, @Cast("l_uint8*") BytePointer sequence, @Cast("l_int32") int seqlen, @Cast("L_DNA**") PointerPointer pda );
public static native @Cast("l_int32") int l_byteaFindEachSequence( L_BYTEA ba, @Cast("l_uint8*") BytePointer sequence, @Cast("l_int32") int seqlen, @ByPtrPtr L_DNA pda );
public static native @Cast("l_int32") int l_byteaFindEachSequence( L_BYTEA ba, @Cast("l_uint8*") ByteBuffer sequence, @Cast("l_int32") int seqlen, @ByPtrPtr L_DNA pda );
public static native @Cast("l_int32") int l_byteaFindEachSequence( L_BYTEA ba, @Cast("l_uint8*") byte[] sequence, @Cast("l_int32") int seqlen, @ByPtrPtr L_DNA pda );
public static native @Cast("l_int32") int l_byteaWrite( @Cast("const char*") BytePointer fname, L_BYTEA ba, @Cast("size_t") long startloc, @Cast("size_t") long endloc );
public static native @Cast("l_int32") int l_byteaWrite( String fname, L_BYTEA ba, @Cast("size_t") long startloc, @Cast("size_t") long endloc );
public static native @Cast("l_int32") int l_byteaWriteStream( @Cast("FILE*") Pointer fp, L_BYTEA ba, @Cast("size_t") long startloc, @Cast("size_t") long endloc );
public static native CCBORDA ccbaCreate( PIX pixs, @Cast("l_int32") int n );
public static native void ccbaDestroy( @Cast("CCBORDA**") PointerPointer pccba );
public static native void ccbaDestroy( @ByPtrPtr CCBORDA pccba );
public static native CCBORD ccbCreate( PIX pixs );
public static native void ccbDestroy( @Cast("CCBORD**") PointerPointer pccb );
public static native void ccbDestroy( @ByPtrPtr CCBORD pccb );
public static native @Cast("l_int32") int ccbaAddCcb( CCBORDA ccba, CCBORD ccb );
public static native @Cast("l_int32") int ccbaGetCount( CCBORDA ccba );
public static native CCBORD ccbaGetCcb( CCBORDA ccba, @Cast("l_int32") int index );
public static native CCBORDA pixGetAllCCBorders( PIX pixs );
public static native CCBORD pixGetCCBorders( PIX pixs, BOX box );
public static native PTAA pixGetOuterBordersPtaa( PIX pixs );
public static native PTA pixGetOuterBorderPta( PIX pixs, BOX box );
public static native @Cast("l_int32") int pixGetOuterBorder( CCBORD ccb, PIX pixs, BOX box );
public static native @Cast("l_int32") int pixGetHoleBorder( CCBORD ccb, PIX pixs, BOX box, @Cast("l_int32") int xs, @Cast("l_int32") int ys );
public static native @Cast("l_int32") int findNextBorderPixel( @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_uint32*") IntPointer data, @Cast("l_int32") int wpl, @Cast("l_int32") int px, @Cast("l_int32") int py, @Cast("l_int32*") IntPointer pqpos, @Cast("l_int32*") IntPointer pnpx, @Cast("l_int32*") IntPointer pnpy );
public static native @Cast("l_int32") int findNextBorderPixel( @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_uint32*") IntBuffer data, @Cast("l_int32") int wpl, @Cast("l_int32") int px, @Cast("l_int32") int py, @Cast("l_int32*") IntBuffer pqpos, @Cast("l_int32*") IntBuffer pnpx, @Cast("l_int32*") IntBuffer pnpy );
public static native @Cast("l_int32") int findNextBorderPixel( @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_uint32*") int[] data, @Cast("l_int32") int wpl, @Cast("l_int32") int px, @Cast("l_int32") int py, @Cast("l_int32*") int[] pqpos, @Cast("l_int32*") int[] pnpx, @Cast("l_int32*") int[] pnpy );
public static native void locateOutsideSeedPixel( @Cast("l_int32") int fpx, @Cast("l_int32") int fpy, @Cast("l_int32") int spx, @Cast("l_int32") int spy, @Cast("l_int32*") IntPointer pxs, @Cast("l_int32*") IntPointer pys );
public static native void locateOutsideSeedPixel( @Cast("l_int32") int fpx, @Cast("l_int32") int fpy, @Cast("l_int32") int spx, @Cast("l_int32") int spy, @Cast("l_int32*") IntBuffer pxs, @Cast("l_int32*") IntBuffer pys );
public static native void locateOutsideSeedPixel( @Cast("l_int32") int fpx, @Cast("l_int32") int fpy, @Cast("l_int32") int spx, @Cast("l_int32") int spy, @Cast("l_int32*") int[] pxs, @Cast("l_int32*") int[] pys );
public static native @Cast("l_int32") int ccbaGenerateGlobalLocs( CCBORDA ccba );
public static native @Cast("l_int32") int ccbaGenerateStepChains( CCBORDA ccba );
public static native @Cast("l_int32") int ccbaStepChainsToPixCoords( CCBORDA ccba, @Cast("l_int32") int coordtype );
public static native @Cast("l_int32") int ccbaGenerateSPGlobalLocs( CCBORDA ccba, @Cast("l_int32") int ptsflag );
public static native @Cast("l_int32") int ccbaGenerateSinglePath( CCBORDA ccba );
public static native PTA getCutPathForHole( PIX pix, PTA pta, BOX boxinner, @Cast("l_int32*") IntPointer pdir, @Cast("l_int32*") IntPointer plen );
public static native PTA getCutPathForHole( PIX pix, PTA pta, BOX boxinner, @Cast("l_int32*") IntBuffer pdir, @Cast("l_int32*") IntBuffer plen );
public static native PTA getCutPathForHole( PIX pix, PTA pta, BOX boxinner, @Cast("l_int32*") int[] pdir, @Cast("l_int32*") int[] plen );
public static native PIX ccbaDisplayBorder( CCBORDA ccba );
public static native PIX ccbaDisplaySPBorder( CCBORDA ccba );
public static native PIX ccbaDisplayImage1( CCBORDA ccba );
public static native PIX ccbaDisplayImage2( CCBORDA ccba );
public static native @Cast("l_int32") int ccbaWrite( @Cast("const char*") BytePointer filename, CCBORDA ccba );
public static native @Cast("l_int32") int ccbaWrite( String filename, CCBORDA ccba );
public static native @Cast("l_int32") int ccbaWriteStream( @Cast("FILE*") Pointer fp, CCBORDA ccba );
public static native CCBORDA ccbaRead( @Cast("const char*") BytePointer filename );
public static native CCBORDA ccbaRead( String filename );
public static native CCBORDA ccbaReadStream( @Cast("FILE*") Pointer fp );
public static native @Cast("l_int32") int ccbaWriteSVG( @Cast("const char*") BytePointer filename, CCBORDA ccba );
public static native @Cast("l_int32") int ccbaWriteSVG( String filename, CCBORDA ccba );
public static native @Cast("char*") BytePointer ccbaWriteSVGString( @Cast("const char*") BytePointer filename, CCBORDA ccba );
public static native @Cast("char*") ByteBuffer ccbaWriteSVGString( String filename, CCBORDA ccba );
public static native PIX pixThin( PIX pixs, @Cast("l_int32") int type, @Cast("l_int32") int connectivity, @Cast("l_int32") int maxiters );
public static native PIX pixThinGeneral( PIX pixs, @Cast("l_int32") int type, SELA sela, @Cast("l_int32") int maxiters );
public static native PIX pixThinExamples( PIX pixs, @Cast("l_int32") int type, @Cast("l_int32") int index, @Cast("l_int32") int maxiters, @Cast("const char*") BytePointer selfile );
public static native PIX pixThinExamples( PIX pixs, @Cast("l_int32") int type, @Cast("l_int32") int index, @Cast("l_int32") int maxiters, String selfile );
public static native @Cast("l_int32") int jbCorrelation( @Cast("const char*") BytePointer dirin, @Cast("l_float32") float thresh, @Cast("l_float32") float weight, @Cast("l_int32") int components, @Cast("const char*") BytePointer rootname, @Cast("l_int32") int firstpage, @Cast("l_int32") int npages, @Cast("l_int32") int renderflag );
public static native @Cast("l_int32") int jbCorrelation( String dirin, @Cast("l_float32") float thresh, @Cast("l_float32") float weight, @Cast("l_int32") int components, String rootname, @Cast("l_int32") int firstpage, @Cast("l_int32") int npages, @Cast("l_int32") int renderflag );
public static native @Cast("l_int32") int jbRankHaus( @Cast("const char*") BytePointer dirin, @Cast("l_int32") int size, @Cast("l_float32") float rank, @Cast("l_int32") int components, @Cast("const char*") BytePointer rootname, @Cast("l_int32") int firstpage, @Cast("l_int32") int npages, @Cast("l_int32") int renderflag );
public static native @Cast("l_int32") int jbRankHaus( String dirin, @Cast("l_int32") int size, @Cast("l_float32") float rank, @Cast("l_int32") int components, String rootname, @Cast("l_int32") int firstpage, @Cast("l_int32") int npages, @Cast("l_int32") int renderflag );
public static native JBCLASSER jbWordsInTextlines( @Cast("const char*") BytePointer dirin, @Cast("l_int32") int reduction, @Cast("l_int32") int maxwidth, @Cast("l_int32") int maxheight, @Cast("l_float32") float thresh, @Cast("l_float32") float weight, @Cast("NUMA**") PointerPointer pnatl, @Cast("l_int32") int firstpage, @Cast("l_int32") int npages );
public static native JBCLASSER jbWordsInTextlines( @Cast("const char*") BytePointer dirin, @Cast("l_int32") int reduction, @Cast("l_int32") int maxwidth, @Cast("l_int32") int maxheight, @Cast("l_float32") float thresh, @Cast("l_float32") float weight, @ByPtrPtr NUMA pnatl, @Cast("l_int32") int firstpage, @Cast("l_int32") int npages );
public static native JBCLASSER jbWordsInTextlines( String dirin, @Cast("l_int32") int reduction, @Cast("l_int32") int maxwidth, @Cast("l_int32") int maxheight, @Cast("l_float32") float thresh, @Cast("l_float32") float weight, @ByPtrPtr NUMA pnatl, @Cast("l_int32") int firstpage, @Cast("l_int32") int npages );
public static native @Cast("l_int32") int pixGetWordsInTextlines( PIX pixs, @Cast("l_int32") int reduction, @Cast("l_int32") int minwidth, @Cast("l_int32") int minheight, @Cast("l_int32") int maxwidth, @Cast("l_int32") int maxheight, @Cast("BOXA**") PointerPointer pboxad, @Cast("PIXA**") PointerPointer ppixad, @Cast("NUMA**") PointerPointer pnai );
public static native @Cast("l_int32") int pixGetWordsInTextlines( PIX pixs, @Cast("l_int32") int reduction, @Cast("l_int32") int minwidth, @Cast("l_int32") int minheight, @Cast("l_int32") int maxwidth, @Cast("l_int32") int maxheight, @ByPtrPtr BOXA pboxad, @ByPtrPtr PIXA ppixad, @ByPtrPtr NUMA pnai );
public static native @Cast("l_int32") int pixGetWordBoxesInTextlines( PIX pixs, @Cast("l_int32") int reduction, @Cast("l_int32") int minwidth, @Cast("l_int32") int minheight, @Cast("l_int32") int maxwidth, @Cast("l_int32") int maxheight, @Cast("BOXA**") PointerPointer pboxad, @Cast("NUMA**") PointerPointer pnai );
public static native @Cast("l_int32") int pixGetWordBoxesInTextlines( PIX pixs, @Cast("l_int32") int reduction, @Cast("l_int32") int minwidth, @Cast("l_int32") int minheight, @Cast("l_int32") int maxwidth, @Cast("l_int32") int maxheight, @ByPtrPtr BOXA pboxad, @ByPtrPtr NUMA pnai );
public static native NUMAA boxaExtractSortedPattern( BOXA boxa, NUMA na );
public static native @Cast("l_int32") int numaaCompareImagesByBoxes( NUMAA naa1, NUMAA naa2, @Cast("l_int32") int nperline, @Cast("l_int32") int nreq, @Cast("l_int32") int maxshiftx, @Cast("l_int32") int maxshifty, @Cast("l_int32") int delx, @Cast("l_int32") int dely, @Cast("l_int32*") IntPointer psame, @Cast("l_int32") int debugflag );
public static native @Cast("l_int32") int numaaCompareImagesByBoxes( NUMAA naa1, NUMAA naa2, @Cast("l_int32") int nperline, @Cast("l_int32") int nreq, @Cast("l_int32") int maxshiftx, @Cast("l_int32") int maxshifty, @Cast("l_int32") int delx, @Cast("l_int32") int dely, @Cast("l_int32*") IntBuffer psame, @Cast("l_int32") int debugflag );
public static native @Cast("l_int32") int numaaCompareImagesByBoxes( NUMAA naa1, NUMAA naa2, @Cast("l_int32") int nperline, @Cast("l_int32") int nreq, @Cast("l_int32") int maxshiftx, @Cast("l_int32") int maxshifty, @Cast("l_int32") int delx, @Cast("l_int32") int dely, @Cast("l_int32*") int[] psame, @Cast("l_int32") int debugflag );
public static native @Cast("l_int32") int pixColorContent( PIX pixs, @Cast("l_int32") int rwhite, @Cast("l_int32") int gwhite, @Cast("l_int32") int bwhite, @Cast("l_int32") int mingray, @Cast("PIX**") PointerPointer ppixr, @Cast("PIX**") PointerPointer ppixg, @Cast("PIX**") PointerPointer ppixb );
public static native @Cast("l_int32") int pixColorContent( PIX pixs, @Cast("l_int32") int rwhite, @Cast("l_int32") int gwhite, @Cast("l_int32") int bwhite, @Cast("l_int32") int mingray, @ByPtrPtr PIX ppixr, @ByPtrPtr PIX ppixg, @ByPtrPtr PIX ppixb );
public static native PIX pixColorMagnitude( PIX pixs, @Cast("l_int32") int rwhite, @Cast("l_int32") int gwhite, @Cast("l_int32") int bwhite, @Cast("l_int32") int type );
public static native PIX pixMaskOverColorPixels( PIX pixs, @Cast("l_int32") int threshdiff, @Cast("l_int32") int mindist );
public static native PIX pixMaskOverColorRange( PIX pixs, @Cast("l_int32") int rmin, @Cast("l_int32") int rmax, @Cast("l_int32") int gmin, @Cast("l_int32") int gmax, @Cast("l_int32") int bmin, @Cast("l_int32") int bmax );
public static native @Cast("l_int32") int pixColorFraction( PIX pixs, @Cast("l_int32") int darkthresh, @Cast("l_int32") int lightthresh, @Cast("l_int32") int diffthresh, @Cast("l_int32") int factor, @Cast("l_float32*") FloatPointer ppixfract, @Cast("l_float32*") FloatPointer pcolorfract );
public static native @Cast("l_int32") int pixColorFraction( PIX pixs, @Cast("l_int32") int darkthresh, @Cast("l_int32") int lightthresh, @Cast("l_int32") int diffthresh, @Cast("l_int32") int factor, @Cast("l_float32*") FloatBuffer ppixfract, @Cast("l_float32*") FloatBuffer pcolorfract );
public static native @Cast("l_int32") int pixColorFraction( PIX pixs, @Cast("l_int32") int darkthresh, @Cast("l_int32") int lightthresh, @Cast("l_int32") int diffthresh, @Cast("l_int32") int factor, @Cast("l_float32*") float[] ppixfract, @Cast("l_float32*") float[] pcolorfract );
public static native @Cast("l_int32") int pixNumSignificantGrayColors( PIX pixs, @Cast("l_int32") int darkthresh, @Cast("l_int32") int lightthresh, @Cast("l_float32") float minfract, @Cast("l_int32") int factor, @Cast("l_int32*") IntPointer pncolors );
public static native @Cast("l_int32") int pixNumSignificantGrayColors( PIX pixs, @Cast("l_int32") int darkthresh, @Cast("l_int32") int lightthresh, @Cast("l_float32") float minfract, @Cast("l_int32") int factor, @Cast("l_int32*") IntBuffer pncolors );
public static native @Cast("l_int32") int pixNumSignificantGrayColors( PIX pixs, @Cast("l_int32") int darkthresh, @Cast("l_int32") int lightthresh, @Cast("l_float32") float minfract, @Cast("l_int32") int factor, @Cast("l_int32*") int[] pncolors );
public static native @Cast("l_int32") int pixColorsForQuantization( PIX pixs, @Cast("l_int32") int thresh, @Cast("l_int32*") IntPointer pncolors, @Cast("l_int32*") IntPointer piscolor, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int pixColorsForQuantization( PIX pixs, @Cast("l_int32") int thresh, @Cast("l_int32*") IntBuffer pncolors, @Cast("l_int32*") IntBuffer piscolor, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int pixColorsForQuantization( PIX pixs, @Cast("l_int32") int thresh, @Cast("l_int32*") int[] pncolors, @Cast("l_int32*") int[] piscolor, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int pixNumColors( PIX pixs, @Cast("l_int32") int factor, @Cast("l_int32*") IntPointer pncolors );
public static native @Cast("l_int32") int pixNumColors( PIX pixs, @Cast("l_int32") int factor, @Cast("l_int32*") IntBuffer pncolors );
public static native @Cast("l_int32") int pixNumColors( PIX pixs, @Cast("l_int32") int factor, @Cast("l_int32*") int[] pncolors );
public static native @Cast("l_int32") int pixGetMostPopulatedColors( PIX pixs, @Cast("l_int32") int sigbits, @Cast("l_int32") int factor, @Cast("l_int32") int ncolors, @Cast("l_uint32**") PointerPointer parray, @Cast("PIXCMAP**") PointerPointer pcmap );
public static native @Cast("l_int32") int pixGetMostPopulatedColors( PIX pixs, @Cast("l_int32") int sigbits, @Cast("l_int32") int factor, @Cast("l_int32") int ncolors, @Cast("l_uint32**") @ByPtrPtr IntPointer parray, @ByPtrPtr PIXCMAP pcmap );
public static native @Cast("l_int32") int pixGetMostPopulatedColors( PIX pixs, @Cast("l_int32") int sigbits, @Cast("l_int32") int factor, @Cast("l_int32") int ncolors, @Cast("l_uint32**") @ByPtrPtr IntBuffer parray, @ByPtrPtr PIXCMAP pcmap );
public static native @Cast("l_int32") int pixGetMostPopulatedColors( PIX pixs, @Cast("l_int32") int sigbits, @Cast("l_int32") int factor, @Cast("l_int32") int ncolors, @Cast("l_uint32**") @ByPtrPtr int[] parray, @ByPtrPtr PIXCMAP pcmap );
public static native PIX pixSimpleColorQuantize( PIX pixs, @Cast("l_int32") int sigbits, @Cast("l_int32") int factor, @Cast("l_int32") int ncolors );
public static native NUMA pixGetRGBHistogram( PIX pixs, @Cast("l_int32") int sigbits, @Cast("l_int32") int factor );
public static native @Cast("l_int32") int makeRGBIndexTables( @Cast("l_uint32**") PointerPointer prtab, @Cast("l_uint32**") PointerPointer pgtab, @Cast("l_uint32**") PointerPointer pbtab, @Cast("l_int32") int sigbits );
public static native @Cast("l_int32") int makeRGBIndexTables( @Cast("l_uint32**") @ByPtrPtr IntPointer prtab, @Cast("l_uint32**") @ByPtrPtr IntPointer pgtab, @Cast("l_uint32**") @ByPtrPtr IntPointer pbtab, @Cast("l_int32") int sigbits );
public static native @Cast("l_int32") int makeRGBIndexTables( @Cast("l_uint32**") @ByPtrPtr IntBuffer prtab, @Cast("l_uint32**") @ByPtrPtr IntBuffer pgtab, @Cast("l_uint32**") @ByPtrPtr IntBuffer pbtab, @Cast("l_int32") int sigbits );
public static native @Cast("l_int32") int makeRGBIndexTables( @Cast("l_uint32**") @ByPtrPtr int[] prtab, @Cast("l_uint32**") @ByPtrPtr int[] pgtab, @Cast("l_uint32**") @ByPtrPtr int[] pbtab, @Cast("l_int32") int sigbits );
public static native @Cast("l_int32") int getRGBFromIndex( @Cast("l_uint32") int index, @Cast("l_int32") int sigbits, @Cast("l_int32*") IntPointer prval, @Cast("l_int32*") IntPointer pgval, @Cast("l_int32*") IntPointer pbval );
public static native @Cast("l_int32") int getRGBFromIndex( @Cast("l_uint32") int index, @Cast("l_int32") int sigbits, @Cast("l_int32*") IntBuffer prval, @Cast("l_int32*") IntBuffer pgval, @Cast("l_int32*") IntBuffer pbval );
public static native @Cast("l_int32") int getRGBFromIndex( @Cast("l_uint32") int index, @Cast("l_int32") int sigbits, @Cast("l_int32*") int[] prval, @Cast("l_int32*") int[] pgval, @Cast("l_int32*") int[] pbval );
public static native @Cast("l_int32") int pixHasHighlightRed( PIX pixs, @Cast("l_int32") int factor, @Cast("l_float32") float fract, @Cast("l_float32") float fthresh, @Cast("l_int32*") IntPointer phasred, @Cast("l_float32*") FloatPointer pratio, @Cast("PIX**") PointerPointer ppixdb );
public static native @Cast("l_int32") int pixHasHighlightRed( PIX pixs, @Cast("l_int32") int factor, @Cast("l_float32") float fract, @Cast("l_float32") float fthresh, @Cast("l_int32*") IntPointer phasred, @Cast("l_float32*") FloatPointer pratio, @ByPtrPtr PIX ppixdb );
public static native @Cast("l_int32") int pixHasHighlightRed( PIX pixs, @Cast("l_int32") int factor, @Cast("l_float32") float fract, @Cast("l_float32") float fthresh, @Cast("l_int32*") IntBuffer phasred, @Cast("l_float32*") FloatBuffer pratio, @ByPtrPtr PIX ppixdb );
public static native @Cast("l_int32") int pixHasHighlightRed( PIX pixs, @Cast("l_int32") int factor, @Cast("l_float32") float fract, @Cast("l_float32") float fthresh, @Cast("l_int32*") int[] phasred, @Cast("l_float32*") float[] pratio, @ByPtrPtr PIX ppixdb );
public static native PIX pixColorGrayRegions( PIX pixs, BOXA boxa, @Cast("l_int32") int type, @Cast("l_int32") int thresh, @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval );
public static native @Cast("l_int32") int pixColorGray( PIX pixs, BOX box, @Cast("l_int32") int type, @Cast("l_int32") int thresh, @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval );
public static native PIX pixColorGrayMasked( PIX pixs, PIX pixm, @Cast("l_int32") int type, @Cast("l_int32") int thresh, @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval );
public static native PIX pixSnapColor( PIX pixd, PIX pixs, @Cast("l_uint32") int srcval, @Cast("l_uint32") int dstval, @Cast("l_int32") int diff );
public static native PIX pixSnapColorCmap( PIX pixd, PIX pixs, @Cast("l_uint32") int srcval, @Cast("l_uint32") int dstval, @Cast("l_int32") int diff );
public static native PIX pixLinearMapToTargetColor( PIX pixd, PIX pixs, @Cast("l_uint32") int srcval, @Cast("l_uint32") int dstval );
public static native @Cast("l_int32") int pixelLinearMapToTargetColor( @Cast("l_uint32") int scolor, @Cast("l_uint32") int srcmap, @Cast("l_uint32") int dstmap, @Cast("l_uint32*") IntPointer pdcolor );
public static native @Cast("l_int32") int pixelLinearMapToTargetColor( @Cast("l_uint32") int scolor, @Cast("l_uint32") int srcmap, @Cast("l_uint32") int dstmap, @Cast("l_uint32*") IntBuffer pdcolor );
public static native @Cast("l_int32") int pixelLinearMapToTargetColor( @Cast("l_uint32") int scolor, @Cast("l_uint32") int srcmap, @Cast("l_uint32") int dstmap, @Cast("l_uint32*") int[] pdcolor );
public static native PIX pixShiftByComponent( PIX pixd, PIX pixs, @Cast("l_uint32") int srcval, @Cast("l_uint32") int dstval );
public static native @Cast("l_int32") int pixelShiftByComponent( @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_uint32") int srcval, @Cast("l_uint32") int dstval, @Cast("l_uint32*") IntPointer ppixel );
public static native @Cast("l_int32") int pixelShiftByComponent( @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_uint32") int srcval, @Cast("l_uint32") int dstval, @Cast("l_uint32*") IntBuffer ppixel );
public static native @Cast("l_int32") int pixelShiftByComponent( @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_uint32") int srcval, @Cast("l_uint32") int dstval, @Cast("l_uint32*") int[] ppixel );
public static native @Cast("l_int32") int pixelFractionalShift( @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_float32") float fraction, @Cast("l_uint32*") IntPointer ppixel );
public static native @Cast("l_int32") int pixelFractionalShift( @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_float32") float fraction, @Cast("l_uint32*") IntBuffer ppixel );
public static native @Cast("l_int32") int pixelFractionalShift( @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_float32") float fraction, @Cast("l_uint32*") int[] ppixel );
public static native PIXCMAP pixcmapCreate( @Cast("l_int32") int depth );
public static native PIXCMAP pixcmapCreateRandom( @Cast("l_int32") int depth, @Cast("l_int32") int hasblack, @Cast("l_int32") int haswhite );
public static native PIXCMAP pixcmapCreateLinear( @Cast("l_int32") int d, @Cast("l_int32") int nlevels );
public static native PIXCMAP pixcmapCopy( PIXCMAP cmaps );
public static native void pixcmapDestroy( @Cast("PIXCMAP**") PointerPointer pcmap );
public static native void pixcmapDestroy( @ByPtrPtr PIXCMAP pcmap );
public static native @Cast("l_int32") int pixcmapAddColor( PIXCMAP cmap, @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval );
public static native @Cast("l_int32") int pixcmapAddRGBA( PIXCMAP cmap, @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_int32") int aval );
public static native @Cast("l_int32") int pixcmapAddNewColor( PIXCMAP cmap, @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_int32*") IntPointer pindex );
public static native @Cast("l_int32") int pixcmapAddNewColor( PIXCMAP cmap, @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_int32*") IntBuffer pindex );
public static native @Cast("l_int32") int pixcmapAddNewColor( PIXCMAP cmap, @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_int32*") int[] pindex );
public static native @Cast("l_int32") int pixcmapAddNearestColor( PIXCMAP cmap, @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_int32*") IntPointer pindex );
public static native @Cast("l_int32") int pixcmapAddNearestColor( PIXCMAP cmap, @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_int32*") IntBuffer pindex );
public static native @Cast("l_int32") int pixcmapAddNearestColor( PIXCMAP cmap, @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_int32*") int[] pindex );
public static native @Cast("l_int32") int pixcmapUsableColor( PIXCMAP cmap, @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_int32*") IntPointer pusable );
public static native @Cast("l_int32") int pixcmapUsableColor( PIXCMAP cmap, @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_int32*") IntBuffer pusable );
public static native @Cast("l_int32") int pixcmapUsableColor( PIXCMAP cmap, @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_int32*") int[] pusable );
public static native @Cast("l_int32") int pixcmapAddBlackOrWhite( PIXCMAP cmap, @Cast("l_int32") int color, @Cast("l_int32*") IntPointer pindex );
public static native @Cast("l_int32") int pixcmapAddBlackOrWhite( PIXCMAP cmap, @Cast("l_int32") int color, @Cast("l_int32*") IntBuffer pindex );
public static native @Cast("l_int32") int pixcmapAddBlackOrWhite( PIXCMAP cmap, @Cast("l_int32") int color, @Cast("l_int32*") int[] pindex );
public static native @Cast("l_int32") int pixcmapSetBlackAndWhite( PIXCMAP cmap, @Cast("l_int32") int setblack, @Cast("l_int32") int setwhite );
public static native @Cast("l_int32") int pixcmapGetCount( PIXCMAP cmap );
public static native @Cast("l_int32") int pixcmapGetFreeCount( PIXCMAP cmap );
public static native @Cast("l_int32") int pixcmapGetDepth( PIXCMAP cmap );
public static native @Cast("l_int32") int pixcmapGetMinDepth( PIXCMAP cmap, @Cast("l_int32*") IntPointer pmindepth );
public static native @Cast("l_int32") int pixcmapGetMinDepth( PIXCMAP cmap, @Cast("l_int32*") IntBuffer pmindepth );
public static native @Cast("l_int32") int pixcmapGetMinDepth( PIXCMAP cmap, @Cast("l_int32*") int[] pmindepth );
public static native @Cast("l_int32") int pixcmapClear( PIXCMAP cmap );
public static native @Cast("l_int32") int pixcmapGetColor( PIXCMAP cmap, @Cast("l_int32") int index, @Cast("l_int32*") IntPointer prval, @Cast("l_int32*") IntPointer pgval, @Cast("l_int32*") IntPointer pbval );
public static native @Cast("l_int32") int pixcmapGetColor( PIXCMAP cmap, @Cast("l_int32") int index, @Cast("l_int32*") IntBuffer prval, @Cast("l_int32*") IntBuffer pgval, @Cast("l_int32*") IntBuffer pbval );
public static native @Cast("l_int32") int pixcmapGetColor( PIXCMAP cmap, @Cast("l_int32") int index, @Cast("l_int32*") int[] prval, @Cast("l_int32*") int[] pgval, @Cast("l_int32*") int[] pbval );
public static native @Cast("l_int32") int pixcmapGetColor32( PIXCMAP cmap, @Cast("l_int32") int index, @Cast("l_uint32*") IntPointer pval32 );
public static native @Cast("l_int32") int pixcmapGetColor32( PIXCMAP cmap, @Cast("l_int32") int index, @Cast("l_uint32*") IntBuffer pval32 );
public static native @Cast("l_int32") int pixcmapGetColor32( PIXCMAP cmap, @Cast("l_int32") int index, @Cast("l_uint32*") int[] pval32 );
public static native @Cast("l_int32") int pixcmapGetRGBA( PIXCMAP cmap, @Cast("l_int32") int index, @Cast("l_int32*") IntPointer prval, @Cast("l_int32*") IntPointer pgval, @Cast("l_int32*") IntPointer pbval, @Cast("l_int32*") IntPointer paval );
public static native @Cast("l_int32") int pixcmapGetRGBA( PIXCMAP cmap, @Cast("l_int32") int index, @Cast("l_int32*") IntBuffer prval, @Cast("l_int32*") IntBuffer pgval, @Cast("l_int32*") IntBuffer pbval, @Cast("l_int32*") IntBuffer paval );
public static native @Cast("l_int32") int pixcmapGetRGBA( PIXCMAP cmap, @Cast("l_int32") int index, @Cast("l_int32*") int[] prval, @Cast("l_int32*") int[] pgval, @Cast("l_int32*") int[] pbval, @Cast("l_int32*") int[] paval );
public static native @Cast("l_int32") int pixcmapGetRGBA32( PIXCMAP cmap, @Cast("l_int32") int index, @Cast("l_uint32*") IntPointer pval32 );
public static native @Cast("l_int32") int pixcmapGetRGBA32( PIXCMAP cmap, @Cast("l_int32") int index, @Cast("l_uint32*") IntBuffer pval32 );
public static native @Cast("l_int32") int pixcmapGetRGBA32( PIXCMAP cmap, @Cast("l_int32") int index, @Cast("l_uint32*") int[] pval32 );
public static native @Cast("l_int32") int pixcmapResetColor( PIXCMAP cmap, @Cast("l_int32") int index, @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval );
public static native @Cast("l_int32") int pixcmapSetAlpha( PIXCMAP cmap, @Cast("l_int32") int index, @Cast("l_int32") int aval );
public static native @Cast("l_int32") int pixcmapGetIndex( PIXCMAP cmap, @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_int32*") IntPointer pindex );
public static native @Cast("l_int32") int pixcmapGetIndex( PIXCMAP cmap, @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_int32*") IntBuffer pindex );
public static native @Cast("l_int32") int pixcmapGetIndex( PIXCMAP cmap, @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_int32*") int[] pindex );
public static native @Cast("l_int32") int pixcmapHasColor( PIXCMAP cmap, @Cast("l_int32*") IntPointer pcolor );
public static native @Cast("l_int32") int pixcmapHasColor( PIXCMAP cmap, @Cast("l_int32*") IntBuffer pcolor );
public static native @Cast("l_int32") int pixcmapHasColor( PIXCMAP cmap, @Cast("l_int32*") int[] pcolor );
public static native @Cast("l_int32") int pixcmapIsOpaque( PIXCMAP cmap, @Cast("l_int32*") IntPointer popaque );
public static native @Cast("l_int32") int pixcmapIsOpaque( PIXCMAP cmap, @Cast("l_int32*") IntBuffer popaque );
public static native @Cast("l_int32") int pixcmapIsOpaque( PIXCMAP cmap, @Cast("l_int32*") int[] popaque );
public static native @Cast("l_int32") int pixcmapIsBlackAndWhite( PIXCMAP cmap, @Cast("l_int32*") IntPointer pblackwhite );
public static native @Cast("l_int32") int pixcmapIsBlackAndWhite( PIXCMAP cmap, @Cast("l_int32*") IntBuffer pblackwhite );
public static native @Cast("l_int32") int pixcmapIsBlackAndWhite( PIXCMAP cmap, @Cast("l_int32*") int[] pblackwhite );
public static native @Cast("l_int32") int pixcmapCountGrayColors( PIXCMAP cmap, @Cast("l_int32*") IntPointer pngray );
public static native @Cast("l_int32") int pixcmapCountGrayColors( PIXCMAP cmap, @Cast("l_int32*") IntBuffer pngray );
public static native @Cast("l_int32") int pixcmapCountGrayColors( PIXCMAP cmap, @Cast("l_int32*") int[] pngray );
public static native @Cast("l_int32") int pixcmapGetRankIntensity( PIXCMAP cmap, @Cast("l_float32") float rankval, @Cast("l_int32*") IntPointer pindex );
public static native @Cast("l_int32") int pixcmapGetRankIntensity( PIXCMAP cmap, @Cast("l_float32") float rankval, @Cast("l_int32*") IntBuffer pindex );
public static native @Cast("l_int32") int pixcmapGetRankIntensity( PIXCMAP cmap, @Cast("l_float32") float rankval, @Cast("l_int32*") int[] pindex );
public static native @Cast("l_int32") int pixcmapGetNearestIndex( PIXCMAP cmap, @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_int32*") IntPointer pindex );
public static native @Cast("l_int32") int pixcmapGetNearestIndex( PIXCMAP cmap, @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_int32*") IntBuffer pindex );
public static native @Cast("l_int32") int pixcmapGetNearestIndex( PIXCMAP cmap, @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_int32*") int[] pindex );
public static native @Cast("l_int32") int pixcmapGetNearestGrayIndex( PIXCMAP cmap, @Cast("l_int32") int val, @Cast("l_int32*") IntPointer pindex );
public static native @Cast("l_int32") int pixcmapGetNearestGrayIndex( PIXCMAP cmap, @Cast("l_int32") int val, @Cast("l_int32*") IntBuffer pindex );
public static native @Cast("l_int32") int pixcmapGetNearestGrayIndex( PIXCMAP cmap, @Cast("l_int32") int val, @Cast("l_int32*") int[] pindex );
public static native @Cast("l_int32") int pixcmapGetComponentRange( PIXCMAP cmap, @Cast("l_int32") int color, @Cast("l_int32*") IntPointer pminval, @Cast("l_int32*") IntPointer pmaxval );
public static native @Cast("l_int32") int pixcmapGetComponentRange( PIXCMAP cmap, @Cast("l_int32") int color, @Cast("l_int32*") IntBuffer pminval, @Cast("l_int32*") IntBuffer pmaxval );
public static native @Cast("l_int32") int pixcmapGetComponentRange( PIXCMAP cmap, @Cast("l_int32") int color, @Cast("l_int32*") int[] pminval, @Cast("l_int32*") int[] pmaxval );
public static native @Cast("l_int32") int pixcmapGetExtremeValue( PIXCMAP cmap, @Cast("l_int32") int type, @Cast("l_int32*") IntPointer prval, @Cast("l_int32*") IntPointer pgval, @Cast("l_int32*") IntPointer pbval );
public static native @Cast("l_int32") int pixcmapGetExtremeValue( PIXCMAP cmap, @Cast("l_int32") int type, @Cast("l_int32*") IntBuffer prval, @Cast("l_int32*") IntBuffer pgval, @Cast("l_int32*") IntBuffer pbval );
public static native @Cast("l_int32") int pixcmapGetExtremeValue( PIXCMAP cmap, @Cast("l_int32") int type, @Cast("l_int32*") int[] prval, @Cast("l_int32*") int[] pgval, @Cast("l_int32*") int[] pbval );
public static native PIXCMAP pixcmapGrayToColor( @Cast("l_uint32") int color );
public static native PIXCMAP pixcmapColorToGray( PIXCMAP cmaps, @Cast("l_float32") float rwt, @Cast("l_float32") float gwt, @Cast("l_float32") float bwt );
public static native PIXCMAP pixcmapRead( @Cast("const char*") BytePointer filename );
public static native PIXCMAP pixcmapRead( String filename );
public static native PIXCMAP pixcmapReadStream( @Cast("FILE*") Pointer fp );
public static native @Cast("l_int32") int pixcmapWrite( @Cast("const char*") BytePointer filename, PIXCMAP cmap );
public static native @Cast("l_int32") int pixcmapWrite( String filename, PIXCMAP cmap );
public static native @Cast("l_int32") int pixcmapWriteStream( @Cast("FILE*") Pointer fp, PIXCMAP cmap );
public static native @Cast("l_int32") int pixcmapToArrays( PIXCMAP cmap, @Cast("l_int32**") PointerPointer prmap, @Cast("l_int32**") PointerPointer pgmap, @Cast("l_int32**") PointerPointer pbmap, @Cast("l_int32**") PointerPointer pamap );
public static native @Cast("l_int32") int pixcmapToArrays( PIXCMAP cmap, @Cast("l_int32**") @ByPtrPtr IntPointer prmap, @Cast("l_int32**") @ByPtrPtr IntPointer pgmap, @Cast("l_int32**") @ByPtrPtr IntPointer pbmap, @Cast("l_int32**") @ByPtrPtr IntPointer pamap );
public static native @Cast("l_int32") int pixcmapToArrays( PIXCMAP cmap, @Cast("l_int32**") @ByPtrPtr IntBuffer prmap, @Cast("l_int32**") @ByPtrPtr IntBuffer pgmap, @Cast("l_int32**") @ByPtrPtr IntBuffer pbmap, @Cast("l_int32**") @ByPtrPtr IntBuffer pamap );
public static native @Cast("l_int32") int pixcmapToArrays( PIXCMAP cmap, @Cast("l_int32**") @ByPtrPtr int[] prmap, @Cast("l_int32**") @ByPtrPtr int[] pgmap, @Cast("l_int32**") @ByPtrPtr int[] pbmap, @Cast("l_int32**") @ByPtrPtr int[] pamap );
public static native @Cast("l_int32") int pixcmapToRGBTable( PIXCMAP cmap, @Cast("l_uint32**") PointerPointer ptab, @Cast("l_int32*") IntPointer pncolors );
public static native @Cast("l_int32") int pixcmapToRGBTable( PIXCMAP cmap, @Cast("l_uint32**") @ByPtrPtr IntPointer ptab, @Cast("l_int32*") IntPointer pncolors );
public static native @Cast("l_int32") int pixcmapToRGBTable( PIXCMAP cmap, @Cast("l_uint32**") @ByPtrPtr IntBuffer ptab, @Cast("l_int32*") IntBuffer pncolors );
public static native @Cast("l_int32") int pixcmapToRGBTable( PIXCMAP cmap, @Cast("l_uint32**") @ByPtrPtr int[] ptab, @Cast("l_int32*") int[] pncolors );
public static native @Cast("l_int32") int pixcmapSerializeToMemory( PIXCMAP cmap, @Cast("l_int32") int cpc, @Cast("l_int32*") IntPointer pncolors, @Cast("l_uint8**") PointerPointer pdata );
public static native @Cast("l_int32") int pixcmapSerializeToMemory( PIXCMAP cmap, @Cast("l_int32") int cpc, @Cast("l_int32*") IntPointer pncolors, @Cast("l_uint8**") @ByPtrPtr BytePointer pdata );
public static native @Cast("l_int32") int pixcmapSerializeToMemory( PIXCMAP cmap, @Cast("l_int32") int cpc, @Cast("l_int32*") IntBuffer pncolors, @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata );
public static native @Cast("l_int32") int pixcmapSerializeToMemory( PIXCMAP cmap, @Cast("l_int32") int cpc, @Cast("l_int32*") int[] pncolors, @Cast("l_uint8**") @ByPtrPtr byte[] pdata );
public static native PIXCMAP pixcmapDeserializeFromMemory( @Cast("l_uint8*") BytePointer data, @Cast("l_int32") int cpc, @Cast("l_int32") int ncolors );
public static native PIXCMAP pixcmapDeserializeFromMemory( @Cast("l_uint8*") ByteBuffer data, @Cast("l_int32") int cpc, @Cast("l_int32") int ncolors );
public static native PIXCMAP pixcmapDeserializeFromMemory( @Cast("l_uint8*") byte[] data, @Cast("l_int32") int cpc, @Cast("l_int32") int ncolors );
public static native @Cast("char*") BytePointer pixcmapConvertToHex( @Cast("l_uint8*") BytePointer data, @Cast("l_int32") int ncolors );
public static native @Cast("char*") ByteBuffer pixcmapConvertToHex( @Cast("l_uint8*") ByteBuffer data, @Cast("l_int32") int ncolors );
public static native @Cast("char*") byte[] pixcmapConvertToHex( @Cast("l_uint8*") byte[] data, @Cast("l_int32") int ncolors );
public static native @Cast("l_int32") int pixcmapGammaTRC( PIXCMAP cmap, @Cast("l_float32") float gamma, @Cast("l_int32") int minval, @Cast("l_int32") int maxval );
public static native @Cast("l_int32") int pixcmapContrastTRC( PIXCMAP cmap, @Cast("l_float32") float factor );
public static native @Cast("l_int32") int pixcmapShiftIntensity( PIXCMAP cmap, @Cast("l_float32") float fraction );
public static native @Cast("l_int32") int pixcmapShiftByComponent( PIXCMAP cmap, @Cast("l_uint32") int srcval, @Cast("l_uint32") int dstval );
public static native PIX pixColorMorph( PIX pixs, @Cast("l_int32") int type, @Cast("l_int32") int hsize, @Cast("l_int32") int vsize );
public static native PIX pixOctreeColorQuant( PIX pixs, @Cast("l_int32") int colors, @Cast("l_int32") int ditherflag );
public static native PIX pixOctreeColorQuantGeneral( PIX pixs, @Cast("l_int32") int colors, @Cast("l_int32") int ditherflag, @Cast("l_float32") float validthresh, @Cast("l_float32") float colorthresh );
public static native @Cast("l_int32") int makeRGBToIndexTables( @Cast("l_uint32**") PointerPointer prtab, @Cast("l_uint32**") PointerPointer pgtab, @Cast("l_uint32**") PointerPointer pbtab, @Cast("l_int32") int cqlevels );
public static native @Cast("l_int32") int makeRGBToIndexTables( @Cast("l_uint32**") @ByPtrPtr IntPointer prtab, @Cast("l_uint32**") @ByPtrPtr IntPointer pgtab, @Cast("l_uint32**") @ByPtrPtr IntPointer pbtab, @Cast("l_int32") int cqlevels );
public static native @Cast("l_int32") int makeRGBToIndexTables( @Cast("l_uint32**") @ByPtrPtr IntBuffer prtab, @Cast("l_uint32**") @ByPtrPtr IntBuffer pgtab, @Cast("l_uint32**") @ByPtrPtr IntBuffer pbtab, @Cast("l_int32") int cqlevels );
public static native @Cast("l_int32") int makeRGBToIndexTables( @Cast("l_uint32**") @ByPtrPtr int[] prtab, @Cast("l_uint32**") @ByPtrPtr int[] pgtab, @Cast("l_uint32**") @ByPtrPtr int[] pbtab, @Cast("l_int32") int cqlevels );
public static native void getOctcubeIndexFromRGB( @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_uint32*") IntPointer rtab, @Cast("l_uint32*") IntPointer gtab, @Cast("l_uint32*") IntPointer btab, @Cast("l_uint32*") IntPointer pindex );
public static native void getOctcubeIndexFromRGB( @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_uint32*") IntBuffer rtab, @Cast("l_uint32*") IntBuffer gtab, @Cast("l_uint32*") IntBuffer btab, @Cast("l_uint32*") IntBuffer pindex );
public static native void getOctcubeIndexFromRGB( @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_uint32*") int[] rtab, @Cast("l_uint32*") int[] gtab, @Cast("l_uint32*") int[] btab, @Cast("l_uint32*") int[] pindex );
public static native PIX pixOctreeQuantByPopulation( PIX pixs, @Cast("l_int32") int level, @Cast("l_int32") int ditherflag );
public static native PIX pixOctreeQuantNumColors( PIX pixs, @Cast("l_int32") int maxcolors, @Cast("l_int32") int subsample );
public static native PIX pixOctcubeQuantMixedWithGray( PIX pixs, @Cast("l_int32") int depth, @Cast("l_int32") int graylevels, @Cast("l_int32") int delta );
public static native PIX pixFixedOctcubeQuant256( PIX pixs, @Cast("l_int32") int ditherflag );
public static native PIX pixFewColorsOctcubeQuant1( PIX pixs, @Cast("l_int32") int level );
public static native PIX pixFewColorsOctcubeQuant2( PIX pixs, @Cast("l_int32") int level, NUMA na, @Cast("l_int32") int ncolors, @Cast("l_int32*") IntPointer pnerrors );
public static native PIX pixFewColorsOctcubeQuant2( PIX pixs, @Cast("l_int32") int level, NUMA na, @Cast("l_int32") int ncolors, @Cast("l_int32*") IntBuffer pnerrors );
public static native PIX pixFewColorsOctcubeQuant2( PIX pixs, @Cast("l_int32") int level, NUMA na, @Cast("l_int32") int ncolors, @Cast("l_int32*") int[] pnerrors );
public static native PIX pixFewColorsOctcubeQuantMixed( PIX pixs, @Cast("l_int32") int level, @Cast("l_int32") int darkthresh, @Cast("l_int32") int lightthresh, @Cast("l_int32") int diffthresh, @Cast("l_float32") float minfract, @Cast("l_int32") int maxspan );
public static native PIX pixFixedOctcubeQuantGenRGB( PIX pixs, @Cast("l_int32") int level );
public static native PIX pixQuantFromCmap( PIX pixs, PIXCMAP cmap, @Cast("l_int32") int mindepth, @Cast("l_int32") int level, @Cast("l_int32") int metric );
public static native PIX pixOctcubeQuantFromCmap( PIX pixs, PIXCMAP cmap, @Cast("l_int32") int mindepth, @Cast("l_int32") int level, @Cast("l_int32") int metric );
public static native PIX pixOctcubeQuantFromCmapLUT( PIX pixs, PIXCMAP cmap, @Cast("l_int32") int mindepth, @Cast("l_int32*") IntPointer cmaptab, @Cast("l_uint32*") IntPointer rtab, @Cast("l_uint32*") IntPointer gtab, @Cast("l_uint32*") IntPointer btab );
public static native PIX pixOctcubeQuantFromCmapLUT( PIX pixs, PIXCMAP cmap, @Cast("l_int32") int mindepth, @Cast("l_int32*") IntBuffer cmaptab, @Cast("l_uint32*") IntBuffer rtab, @Cast("l_uint32*") IntBuffer gtab, @Cast("l_uint32*") IntBuffer btab );
public static native PIX pixOctcubeQuantFromCmapLUT( PIX pixs, PIXCMAP cmap, @Cast("l_int32") int mindepth, @Cast("l_int32*") int[] cmaptab, @Cast("l_uint32*") int[] rtab, @Cast("l_uint32*") int[] gtab, @Cast("l_uint32*") int[] btab );
public static native NUMA pixOctcubeHistogram( PIX pixs, @Cast("l_int32") int level, @Cast("l_int32*") IntPointer pncolors );
public static native NUMA pixOctcubeHistogram( PIX pixs, @Cast("l_int32") int level, @Cast("l_int32*") IntBuffer pncolors );
public static native NUMA pixOctcubeHistogram( PIX pixs, @Cast("l_int32") int level, @Cast("l_int32*") int[] pncolors );
public static native @Cast("l_int32*") IntPointer pixcmapToOctcubeLUT( PIXCMAP cmap, @Cast("l_int32") int level, @Cast("l_int32") int metric );
public static native @Cast("l_int32") int pixRemoveUnusedColors( PIX pixs );
public static native @Cast("l_int32") int pixNumberOccupiedOctcubes( PIX pix, @Cast("l_int32") int level, @Cast("l_int32") int mincount, @Cast("l_float32") float minfract, @Cast("l_int32*") IntPointer pncolors );
public static native @Cast("l_int32") int pixNumberOccupiedOctcubes( PIX pix, @Cast("l_int32") int level, @Cast("l_int32") int mincount, @Cast("l_float32") float minfract, @Cast("l_int32*") IntBuffer pncolors );
public static native @Cast("l_int32") int pixNumberOccupiedOctcubes( PIX pix, @Cast("l_int32") int level, @Cast("l_int32") int mincount, @Cast("l_float32") float minfract, @Cast("l_int32*") int[] pncolors );
public static native PIX pixMedianCutQuant( PIX pixs, @Cast("l_int32") int ditherflag );
public static native PIX pixMedianCutQuantGeneral( PIX pixs, @Cast("l_int32") int ditherflag, @Cast("l_int32") int outdepth, @Cast("l_int32") int maxcolors, @Cast("l_int32") int sigbits, @Cast("l_int32") int maxsub, @Cast("l_int32") int checkbw );
public static native PIX pixMedianCutQuantMixed( PIX pixs, @Cast("l_int32") int ncolor, @Cast("l_int32") int ngray, @Cast("l_int32") int darkthresh, @Cast("l_int32") int lightthresh, @Cast("l_int32") int diffthresh );
public static native PIX pixFewColorsMedianCutQuantMixed( PIX pixs, @Cast("l_int32") int ncolor, @Cast("l_int32") int ngray, @Cast("l_int32") int maxncolors, @Cast("l_int32") int darkthresh, @Cast("l_int32") int lightthresh, @Cast("l_int32") int diffthresh );
public static native @Cast("l_int32*") IntPointer pixMedianCutHisto( PIX pixs, @Cast("l_int32") int sigbits, @Cast("l_int32") int subsample );
public static native PIX pixColorSegment( PIX pixs, @Cast("l_int32") int maxdist, @Cast("l_int32") int maxcolors, @Cast("l_int32") int selsize, @Cast("l_int32") int finalcolors );
public static native PIX pixColorSegmentCluster( PIX pixs, @Cast("l_int32") int maxdist, @Cast("l_int32") int maxcolors );
public static native @Cast("l_int32") int pixAssignToNearestColor( PIX pixd, PIX pixs, PIX pixm, @Cast("l_int32") int level, @Cast("l_int32*") IntPointer countarray );
public static native @Cast("l_int32") int pixAssignToNearestColor( PIX pixd, PIX pixs, PIX pixm, @Cast("l_int32") int level, @Cast("l_int32*") IntBuffer countarray );
public static native @Cast("l_int32") int pixAssignToNearestColor( PIX pixd, PIX pixs, PIX pixm, @Cast("l_int32") int level, @Cast("l_int32*") int[] countarray );
public static native @Cast("l_int32") int pixColorSegmentClean( PIX pixs, @Cast("l_int32") int selsize, @Cast("l_int32*") IntPointer countarray );
public static native @Cast("l_int32") int pixColorSegmentClean( PIX pixs, @Cast("l_int32") int selsize, @Cast("l_int32*") IntBuffer countarray );
public static native @Cast("l_int32") int pixColorSegmentClean( PIX pixs, @Cast("l_int32") int selsize, @Cast("l_int32*") int[] countarray );
public static native @Cast("l_int32") int pixColorSegmentRemoveColors( PIX pixd, PIX pixs, @Cast("l_int32") int finalcolors );
public static native PIX pixConvertRGBToHSV( PIX pixd, PIX pixs );
public static native PIX pixConvertHSVToRGB( PIX pixd, PIX pixs );
public static native @Cast("l_int32") int convertRGBToHSV( @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_int32*") IntPointer phval, @Cast("l_int32*") IntPointer psval, @Cast("l_int32*") IntPointer pvval );
public static native @Cast("l_int32") int convertRGBToHSV( @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_int32*") IntBuffer phval, @Cast("l_int32*") IntBuffer psval, @Cast("l_int32*") IntBuffer pvval );
public static native @Cast("l_int32") int convertRGBToHSV( @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_int32*") int[] phval, @Cast("l_int32*") int[] psval, @Cast("l_int32*") int[] pvval );
public static native @Cast("l_int32") int convertHSVToRGB( @Cast("l_int32") int hval, @Cast("l_int32") int sval, @Cast("l_int32") int vval, @Cast("l_int32*") IntPointer prval, @Cast("l_int32*") IntPointer pgval, @Cast("l_int32*") IntPointer pbval );
public static native @Cast("l_int32") int convertHSVToRGB( @Cast("l_int32") int hval, @Cast("l_int32") int sval, @Cast("l_int32") int vval, @Cast("l_int32*") IntBuffer prval, @Cast("l_int32*") IntBuffer pgval, @Cast("l_int32*") IntBuffer pbval );
public static native @Cast("l_int32") int convertHSVToRGB( @Cast("l_int32") int hval, @Cast("l_int32") int sval, @Cast("l_int32") int vval, @Cast("l_int32*") int[] prval, @Cast("l_int32*") int[] pgval, @Cast("l_int32*") int[] pbval );
public static native @Cast("l_int32") int pixcmapConvertRGBToHSV( PIXCMAP cmap );
public static native @Cast("l_int32") int pixcmapConvertHSVToRGB( PIXCMAP cmap );
public static native PIX pixConvertRGBToHue( PIX pixs );
public static native PIX pixConvertRGBToSaturation( PIX pixs );
public static native PIX pixConvertRGBToValue( PIX pixs );
public static native PIX pixMakeRangeMaskHS( PIX pixs, @Cast("l_int32") int huecenter, @Cast("l_int32") int huehw, @Cast("l_int32") int satcenter, @Cast("l_int32") int sathw, @Cast("l_int32") int regionflag );
public static native PIX pixMakeRangeMaskHV( PIX pixs, @Cast("l_int32") int huecenter, @Cast("l_int32") int huehw, @Cast("l_int32") int valcenter, @Cast("l_int32") int valhw, @Cast("l_int32") int regionflag );
public static native PIX pixMakeRangeMaskSV( PIX pixs, @Cast("l_int32") int satcenter, @Cast("l_int32") int sathw, @Cast("l_int32") int valcenter, @Cast("l_int32") int valhw, @Cast("l_int32") int regionflag );
public static native PIX pixMakeHistoHS( PIX pixs, @Cast("l_int32") int factor, @Cast("NUMA**") PointerPointer pnahue, @Cast("NUMA**") PointerPointer pnasat );
public static native PIX pixMakeHistoHS( PIX pixs, @Cast("l_int32") int factor, @ByPtrPtr NUMA pnahue, @ByPtrPtr NUMA pnasat );
public static native PIX pixMakeHistoHV( PIX pixs, @Cast("l_int32") int factor, @Cast("NUMA**") PointerPointer pnahue, @Cast("NUMA**") PointerPointer pnaval );
public static native PIX pixMakeHistoHV( PIX pixs, @Cast("l_int32") int factor, @ByPtrPtr NUMA pnahue, @ByPtrPtr NUMA pnaval );
public static native PIX pixMakeHistoSV( PIX pixs, @Cast("l_int32") int factor, @Cast("NUMA**") PointerPointer pnasat, @Cast("NUMA**") PointerPointer pnaval );
public static native PIX pixMakeHistoSV( PIX pixs, @Cast("l_int32") int factor, @ByPtrPtr NUMA pnasat, @ByPtrPtr NUMA pnaval );
public static native @Cast("l_int32") int pixFindHistoPeaksHSV( PIX pixs, @Cast("l_int32") int type, @Cast("l_int32") int width, @Cast("l_int32") int height, @Cast("l_int32") int npeaks, @Cast("l_float32") float erasefactor, @Cast("PTA**") PointerPointer ppta, @Cast("NUMA**") PointerPointer pnatot, @Cast("PIXA**") PointerPointer ppixa );
public static native @Cast("l_int32") int pixFindHistoPeaksHSV( PIX pixs, @Cast("l_int32") int type, @Cast("l_int32") int width, @Cast("l_int32") int height, @Cast("l_int32") int npeaks, @Cast("l_float32") float erasefactor, @ByPtrPtr PTA ppta, @ByPtrPtr NUMA pnatot, @ByPtrPtr PIXA ppixa );
public static native PIX displayHSVColorRange( @Cast("l_int32") int hval, @Cast("l_int32") int sval, @Cast("l_int32") int vval, @Cast("l_int32") int huehw, @Cast("l_int32") int sathw, @Cast("l_int32") int nsamp, @Cast("l_int32") int factor );
public static native PIX pixConvertRGBToYUV( PIX pixd, PIX pixs );
public static native PIX pixConvertYUVToRGB( PIX pixd, PIX pixs );
public static native @Cast("l_int32") int convertRGBToYUV( @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_int32*") IntPointer pyval, @Cast("l_int32*") IntPointer puval, @Cast("l_int32*") IntPointer pvval );
public static native @Cast("l_int32") int convertRGBToYUV( @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_int32*") IntBuffer pyval, @Cast("l_int32*") IntBuffer puval, @Cast("l_int32*") IntBuffer pvval );
public static native @Cast("l_int32") int convertRGBToYUV( @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_int32*") int[] pyval, @Cast("l_int32*") int[] puval, @Cast("l_int32*") int[] pvval );
public static native @Cast("l_int32") int convertYUVToRGB( @Cast("l_int32") int yval, @Cast("l_int32") int uval, @Cast("l_int32") int vval, @Cast("l_int32*") IntPointer prval, @Cast("l_int32*") IntPointer pgval, @Cast("l_int32*") IntPointer pbval );
public static native @Cast("l_int32") int convertYUVToRGB( @Cast("l_int32") int yval, @Cast("l_int32") int uval, @Cast("l_int32") int vval, @Cast("l_int32*") IntBuffer prval, @Cast("l_int32*") IntBuffer pgval, @Cast("l_int32*") IntBuffer pbval );
public static native @Cast("l_int32") int convertYUVToRGB( @Cast("l_int32") int yval, @Cast("l_int32") int uval, @Cast("l_int32") int vval, @Cast("l_int32*") int[] prval, @Cast("l_int32*") int[] pgval, @Cast("l_int32*") int[] pbval );
public static native @Cast("l_int32") int pixcmapConvertRGBToYUV( PIXCMAP cmap );
public static native @Cast("l_int32") int pixcmapConvertYUVToRGB( PIXCMAP cmap );
public static native FPIXA pixConvertRGBToXYZ( PIX pixs );
public static native PIX fpixaConvertXYZToRGB( FPIXA fpixa );
public static native @Cast("l_int32") int convertRGBToXYZ( @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_float32*") FloatPointer pfxval, @Cast("l_float32*") FloatPointer pfyval, @Cast("l_float32*") FloatPointer pfzval );
public static native @Cast("l_int32") int convertRGBToXYZ( @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_float32*") FloatBuffer pfxval, @Cast("l_float32*") FloatBuffer pfyval, @Cast("l_float32*") FloatBuffer pfzval );
public static native @Cast("l_int32") int convertRGBToXYZ( @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_float32*") float[] pfxval, @Cast("l_float32*") float[] pfyval, @Cast("l_float32*") float[] pfzval );
public static native @Cast("l_int32") int convertXYZToRGB( @Cast("l_float32") float fxval, @Cast("l_float32") float fyval, @Cast("l_float32") float fzval, @Cast("l_int32") int blackout, @Cast("l_int32*") IntPointer prval, @Cast("l_int32*") IntPointer pgval, @Cast("l_int32*") IntPointer pbval );
public static native @Cast("l_int32") int convertXYZToRGB( @Cast("l_float32") float fxval, @Cast("l_float32") float fyval, @Cast("l_float32") float fzval, @Cast("l_int32") int blackout, @Cast("l_int32*") IntBuffer prval, @Cast("l_int32*") IntBuffer pgval, @Cast("l_int32*") IntBuffer pbval );
public static native @Cast("l_int32") int convertXYZToRGB( @Cast("l_float32") float fxval, @Cast("l_float32") float fyval, @Cast("l_float32") float fzval, @Cast("l_int32") int blackout, @Cast("l_int32*") int[] prval, @Cast("l_int32*") int[] pgval, @Cast("l_int32*") int[] pbval );
public static native FPIXA fpixaConvertXYZToLAB( FPIXA fpixas );
public static native FPIXA fpixaConvertLABToXYZ( FPIXA fpixas );
public static native @Cast("l_int32") int convertXYZToLAB( @Cast("l_float32") float xval, @Cast("l_float32") float yval, @Cast("l_float32") float zval, @Cast("l_float32*") FloatPointer plval, @Cast("l_float32*") FloatPointer paval, @Cast("l_float32*") FloatPointer pbval );
public static native @Cast("l_int32") int convertXYZToLAB( @Cast("l_float32") float xval, @Cast("l_float32") float yval, @Cast("l_float32") float zval, @Cast("l_float32*") FloatBuffer plval, @Cast("l_float32*") FloatBuffer paval, @Cast("l_float32*") FloatBuffer pbval );
public static native @Cast("l_int32") int convertXYZToLAB( @Cast("l_float32") float xval, @Cast("l_float32") float yval, @Cast("l_float32") float zval, @Cast("l_float32*") float[] plval, @Cast("l_float32*") float[] paval, @Cast("l_float32*") float[] pbval );
public static native @Cast("l_int32") int convertLABToXYZ( @Cast("l_float32") float lval, @Cast("l_float32") float aval, @Cast("l_float32") float bval, @Cast("l_float32*") FloatPointer pxval, @Cast("l_float32*") FloatPointer pyval, @Cast("l_float32*") FloatPointer pzval );
public static native @Cast("l_int32") int convertLABToXYZ( @Cast("l_float32") float lval, @Cast("l_float32") float aval, @Cast("l_float32") float bval, @Cast("l_float32*") FloatBuffer pxval, @Cast("l_float32*") FloatBuffer pyval, @Cast("l_float32*") FloatBuffer pzval );
public static native @Cast("l_int32") int convertLABToXYZ( @Cast("l_float32") float lval, @Cast("l_float32") float aval, @Cast("l_float32") float bval, @Cast("l_float32*") float[] pxval, @Cast("l_float32*") float[] pyval, @Cast("l_float32*") float[] pzval );
public static native FPIXA pixConvertRGBToLAB( PIX pixs );
public static native PIX fpixaConvertLABToRGB( FPIXA fpixa );
public static native @Cast("l_int32") int convertRGBToLAB( @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_float32*") FloatPointer pflval, @Cast("l_float32*") FloatPointer pfaval, @Cast("l_float32*") FloatPointer pfbval );
public static native @Cast("l_int32") int convertRGBToLAB( @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_float32*") FloatBuffer pflval, @Cast("l_float32*") FloatBuffer pfaval, @Cast("l_float32*") FloatBuffer pfbval );
public static native @Cast("l_int32") int convertRGBToLAB( @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_float32*") float[] pflval, @Cast("l_float32*") float[] pfaval, @Cast("l_float32*") float[] pfbval );
public static native @Cast("l_int32") int convertLABToRGB( @Cast("l_float32") float flval, @Cast("l_float32") float faval, @Cast("l_float32") float fbval, @Cast("l_int32*") IntPointer prval, @Cast("l_int32*") IntPointer pgval, @Cast("l_int32*") IntPointer pbval );
public static native @Cast("l_int32") int convertLABToRGB( @Cast("l_float32") float flval, @Cast("l_float32") float faval, @Cast("l_float32") float fbval, @Cast("l_int32*") IntBuffer prval, @Cast("l_int32*") IntBuffer pgval, @Cast("l_int32*") IntBuffer pbval );
public static native @Cast("l_int32") int convertLABToRGB( @Cast("l_float32") float flval, @Cast("l_float32") float faval, @Cast("l_float32") float fbval, @Cast("l_int32*") int[] prval, @Cast("l_int32*") int[] pgval, @Cast("l_int32*") int[] pbval );
public static native @Cast("l_int32") int pixEqual( PIX pix1, PIX pix2, @Cast("l_int32*") IntPointer psame );
public static native @Cast("l_int32") int pixEqual( PIX pix1, PIX pix2, @Cast("l_int32*") IntBuffer psame );
public static native @Cast("l_int32") int pixEqual( PIX pix1, PIX pix2, @Cast("l_int32*") int[] psame );
public static native @Cast("l_int32") int pixEqualWithAlpha( PIX pix1, PIX pix2, @Cast("l_int32") int use_alpha, @Cast("l_int32*") IntPointer psame );
public static native @Cast("l_int32") int pixEqualWithAlpha( PIX pix1, PIX pix2, @Cast("l_int32") int use_alpha, @Cast("l_int32*") IntBuffer psame );
public static native @Cast("l_int32") int pixEqualWithAlpha( PIX pix1, PIX pix2, @Cast("l_int32") int use_alpha, @Cast("l_int32*") int[] psame );
public static native @Cast("l_int32") int pixEqualWithCmap( PIX pix1, PIX pix2, @Cast("l_int32*") IntPointer psame );
public static native @Cast("l_int32") int pixEqualWithCmap( PIX pix1, PIX pix2, @Cast("l_int32*") IntBuffer psame );
public static native @Cast("l_int32") int pixEqualWithCmap( PIX pix1, PIX pix2, @Cast("l_int32*") int[] psame );
public static native @Cast("l_int32") int pixUsesCmapColor( PIX pixs, @Cast("l_int32*") IntPointer pcolor );
public static native @Cast("l_int32") int pixUsesCmapColor( PIX pixs, @Cast("l_int32*") IntBuffer pcolor );
public static native @Cast("l_int32") int pixUsesCmapColor( PIX pixs, @Cast("l_int32*") int[] pcolor );
public static native @Cast("l_int32") int pixCorrelationBinary( PIX pix1, PIX pix2, @Cast("l_float32*") FloatPointer pval );
public static native @Cast("l_int32") int pixCorrelationBinary( PIX pix1, PIX pix2, @Cast("l_float32*") FloatBuffer pval );
public static native @Cast("l_int32") int pixCorrelationBinary( PIX pix1, PIX pix2, @Cast("l_float32*") float[] pval );
public static native PIX pixDisplayDiffBinary( PIX pix1, PIX pix2 );
public static native @Cast("l_int32") int pixCompareBinary( PIX pix1, PIX pix2, @Cast("l_int32") int comptype, @Cast("l_float32*") FloatPointer pfract, @Cast("PIX**") PointerPointer ppixdiff );
public static native @Cast("l_int32") int pixCompareBinary( PIX pix1, PIX pix2, @Cast("l_int32") int comptype, @Cast("l_float32*") FloatPointer pfract, @ByPtrPtr PIX ppixdiff );
public static native @Cast("l_int32") int pixCompareBinary( PIX pix1, PIX pix2, @Cast("l_int32") int comptype, @Cast("l_float32*") FloatBuffer pfract, @ByPtrPtr PIX ppixdiff );
public static native @Cast("l_int32") int pixCompareBinary( PIX pix1, PIX pix2, @Cast("l_int32") int comptype, @Cast("l_float32*") float[] pfract, @ByPtrPtr PIX ppixdiff );
public static native @Cast("l_int32") int pixCompareGrayOrRGB( PIX pix1, PIX pix2, @Cast("l_int32") int comptype, @Cast("l_int32") int plottype, @Cast("l_int32*") IntPointer psame, @Cast("l_float32*") FloatPointer pdiff, @Cast("l_float32*") FloatPointer prmsdiff, @Cast("PIX**") PointerPointer ppixdiff );
public static native @Cast("l_int32") int pixCompareGrayOrRGB( PIX pix1, PIX pix2, @Cast("l_int32") int comptype, @Cast("l_int32") int plottype, @Cast("l_int32*") IntPointer psame, @Cast("l_float32*") FloatPointer pdiff, @Cast("l_float32*") FloatPointer prmsdiff, @ByPtrPtr PIX ppixdiff );
public static native @Cast("l_int32") int pixCompareGrayOrRGB( PIX pix1, PIX pix2, @Cast("l_int32") int comptype, @Cast("l_int32") int plottype, @Cast("l_int32*") IntBuffer psame, @Cast("l_float32*") FloatBuffer pdiff, @Cast("l_float32*") FloatBuffer prmsdiff, @ByPtrPtr PIX ppixdiff );
public static native @Cast("l_int32") int pixCompareGrayOrRGB( PIX pix1, PIX pix2, @Cast("l_int32") int comptype, @Cast("l_int32") int plottype, @Cast("l_int32*") int[] psame, @Cast("l_float32*") float[] pdiff, @Cast("l_float32*") float[] prmsdiff, @ByPtrPtr PIX ppixdiff );
public static native @Cast("l_int32") int pixCompareGray( PIX pix1, PIX pix2, @Cast("l_int32") int comptype, @Cast("l_int32") int plottype, @Cast("l_int32*") IntPointer psame, @Cast("l_float32*") FloatPointer pdiff, @Cast("l_float32*") FloatPointer prmsdiff, @Cast("PIX**") PointerPointer ppixdiff );
public static native @Cast("l_int32") int pixCompareGray( PIX pix1, PIX pix2, @Cast("l_int32") int comptype, @Cast("l_int32") int plottype, @Cast("l_int32*") IntPointer psame, @Cast("l_float32*") FloatPointer pdiff, @Cast("l_float32*") FloatPointer prmsdiff, @ByPtrPtr PIX ppixdiff );
public static native @Cast("l_int32") int pixCompareGray( PIX pix1, PIX pix2, @Cast("l_int32") int comptype, @Cast("l_int32") int plottype, @Cast("l_int32*") IntBuffer psame, @Cast("l_float32*") FloatBuffer pdiff, @Cast("l_float32*") FloatBuffer prmsdiff, @ByPtrPtr PIX ppixdiff );
public static native @Cast("l_int32") int pixCompareGray( PIX pix1, PIX pix2, @Cast("l_int32") int comptype, @Cast("l_int32") int plottype, @Cast("l_int32*") int[] psame, @Cast("l_float32*") float[] pdiff, @Cast("l_float32*") float[] prmsdiff, @ByPtrPtr PIX ppixdiff );
public static native @Cast("l_int32") int pixCompareRGB( PIX pix1, PIX pix2, @Cast("l_int32") int comptype, @Cast("l_int32") int plottype, @Cast("l_int32*") IntPointer psame, @Cast("l_float32*") FloatPointer pdiff, @Cast("l_float32*") FloatPointer prmsdiff, @Cast("PIX**") PointerPointer ppixdiff );
public static native @Cast("l_int32") int pixCompareRGB( PIX pix1, PIX pix2, @Cast("l_int32") int comptype, @Cast("l_int32") int plottype, @Cast("l_int32*") IntPointer psame, @Cast("l_float32*") FloatPointer pdiff, @Cast("l_float32*") FloatPointer prmsdiff, @ByPtrPtr PIX ppixdiff );
public static native @Cast("l_int32") int pixCompareRGB( PIX pix1, PIX pix2, @Cast("l_int32") int comptype, @Cast("l_int32") int plottype, @Cast("l_int32*") IntBuffer psame, @Cast("l_float32*") FloatBuffer pdiff, @Cast("l_float32*") FloatBuffer prmsdiff, @ByPtrPtr PIX ppixdiff );
public static native @Cast("l_int32") int pixCompareRGB( PIX pix1, PIX pix2, @Cast("l_int32") int comptype, @Cast("l_int32") int plottype, @Cast("l_int32*") int[] psame, @Cast("l_float32*") float[] pdiff, @Cast("l_float32*") float[] prmsdiff, @ByPtrPtr PIX ppixdiff );
public static native @Cast("l_int32") int pixCompareTiled( PIX pix1, PIX pix2, @Cast("l_int32") int sx, @Cast("l_int32") int sy, @Cast("l_int32") int type, @Cast("PIX**") PointerPointer ppixdiff );
public static native @Cast("l_int32") int pixCompareTiled( PIX pix1, PIX pix2, @Cast("l_int32") int sx, @Cast("l_int32") int sy, @Cast("l_int32") int type, @ByPtrPtr PIX ppixdiff );
public static native NUMA pixCompareRankDifference( PIX pix1, PIX pix2, @Cast("l_int32") int factor );
public static native @Cast("l_int32") int pixTestForSimilarity( PIX pix1, PIX pix2, @Cast("l_int32") int factor, @Cast("l_int32") int mindiff, @Cast("l_float32") float maxfract, @Cast("l_float32") float maxave, @Cast("l_int32*") IntPointer psimilar, @Cast("l_int32") int printstats );
public static native @Cast("l_int32") int pixTestForSimilarity( PIX pix1, PIX pix2, @Cast("l_int32") int factor, @Cast("l_int32") int mindiff, @Cast("l_float32") float maxfract, @Cast("l_float32") float maxave, @Cast("l_int32*") IntBuffer psimilar, @Cast("l_int32") int printstats );
public static native @Cast("l_int32") int pixTestForSimilarity( PIX pix1, PIX pix2, @Cast("l_int32") int factor, @Cast("l_int32") int mindiff, @Cast("l_float32") float maxfract, @Cast("l_float32") float maxave, @Cast("l_int32*") int[] psimilar, @Cast("l_int32") int printstats );
public static native @Cast("l_int32") int pixGetDifferenceStats( PIX pix1, PIX pix2, @Cast("l_int32") int factor, @Cast("l_int32") int mindiff, @Cast("l_float32*") FloatPointer pfractdiff, @Cast("l_float32*") FloatPointer pavediff, @Cast("l_int32") int printstats );
public static native @Cast("l_int32") int pixGetDifferenceStats( PIX pix1, PIX pix2, @Cast("l_int32") int factor, @Cast("l_int32") int mindiff, @Cast("l_float32*") FloatBuffer pfractdiff, @Cast("l_float32*") FloatBuffer pavediff, @Cast("l_int32") int printstats );
public static native @Cast("l_int32") int pixGetDifferenceStats( PIX pix1, PIX pix2, @Cast("l_int32") int factor, @Cast("l_int32") int mindiff, @Cast("l_float32*") float[] pfractdiff, @Cast("l_float32*") float[] pavediff, @Cast("l_int32") int printstats );
public static native NUMA pixGetDifferenceHistogram( PIX pix1, PIX pix2, @Cast("l_int32") int factor );
public static native @Cast("l_int32") int pixGetPerceptualDiff( PIX pixs1, PIX pixs2, @Cast("l_int32") int sampling, @Cast("l_int32") int dilation, @Cast("l_int32") int mindiff, @Cast("l_float32*") FloatPointer pfract, @Cast("PIX**") PointerPointer ppixdiff1, @Cast("PIX**") PointerPointer ppixdiff2 );
public static native @Cast("l_int32") int pixGetPerceptualDiff( PIX pixs1, PIX pixs2, @Cast("l_int32") int sampling, @Cast("l_int32") int dilation, @Cast("l_int32") int mindiff, @Cast("l_float32*") FloatPointer pfract, @ByPtrPtr PIX ppixdiff1, @ByPtrPtr PIX ppixdiff2 );
public static native @Cast("l_int32") int pixGetPerceptualDiff( PIX pixs1, PIX pixs2, @Cast("l_int32") int sampling, @Cast("l_int32") int dilation, @Cast("l_int32") int mindiff, @Cast("l_float32*") FloatBuffer pfract, @ByPtrPtr PIX ppixdiff1, @ByPtrPtr PIX ppixdiff2 );
public static native @Cast("l_int32") int pixGetPerceptualDiff( PIX pixs1, PIX pixs2, @Cast("l_int32") int sampling, @Cast("l_int32") int dilation, @Cast("l_int32") int mindiff, @Cast("l_float32*") float[] pfract, @ByPtrPtr PIX ppixdiff1, @ByPtrPtr PIX ppixdiff2 );
public static native @Cast("l_int32") int pixGetPSNR( PIX pix1, PIX pix2, @Cast("l_int32") int factor, @Cast("l_float32*") FloatPointer ppsnr );
public static native @Cast("l_int32") int pixGetPSNR( PIX pix1, PIX pix2, @Cast("l_int32") int factor, @Cast("l_float32*") FloatBuffer ppsnr );
public static native @Cast("l_int32") int pixGetPSNR( PIX pix1, PIX pix2, @Cast("l_int32") int factor, @Cast("l_float32*") float[] ppsnr );
public static native @Cast("l_int32") int pixCompareWithTranslation( PIX pix1, PIX pix2, @Cast("l_int32") int thresh, @Cast("l_int32*") IntPointer pdelx, @Cast("l_int32*") IntPointer pdely, @Cast("l_float32*") FloatPointer pscore, @Cast("l_int32") int debugflag );
public static native @Cast("l_int32") int pixCompareWithTranslation( PIX pix1, PIX pix2, @Cast("l_int32") int thresh, @Cast("l_int32*") IntBuffer pdelx, @Cast("l_int32*") IntBuffer pdely, @Cast("l_float32*") FloatBuffer pscore, @Cast("l_int32") int debugflag );
public static native @Cast("l_int32") int pixCompareWithTranslation( PIX pix1, PIX pix2, @Cast("l_int32") int thresh, @Cast("l_int32*") int[] pdelx, @Cast("l_int32*") int[] pdely, @Cast("l_float32*") float[] pscore, @Cast("l_int32") int debugflag );
public static native @Cast("l_int32") int pixBestCorrelation( PIX pix1, PIX pix2, @Cast("l_int32") int area1, @Cast("l_int32") int area2, @Cast("l_int32") int etransx, @Cast("l_int32") int etransy, @Cast("l_int32") int maxshift, @Cast("l_int32*") IntPointer tab8, @Cast("l_int32*") IntPointer pdelx, @Cast("l_int32*") IntPointer pdely, @Cast("l_float32*") FloatPointer pscore, @Cast("l_int32") int debugflag );
public static native @Cast("l_int32") int pixBestCorrelation( PIX pix1, PIX pix2, @Cast("l_int32") int area1, @Cast("l_int32") int area2, @Cast("l_int32") int etransx, @Cast("l_int32") int etransy, @Cast("l_int32") int maxshift, @Cast("l_int32*") IntBuffer tab8, @Cast("l_int32*") IntBuffer pdelx, @Cast("l_int32*") IntBuffer pdely, @Cast("l_float32*") FloatBuffer pscore, @Cast("l_int32") int debugflag );
public static native @Cast("l_int32") int pixBestCorrelation( PIX pix1, PIX pix2, @Cast("l_int32") int area1, @Cast("l_int32") int area2, @Cast("l_int32") int etransx, @Cast("l_int32") int etransy, @Cast("l_int32") int maxshift, @Cast("l_int32*") int[] tab8, @Cast("l_int32*") int[] pdelx, @Cast("l_int32*") int[] pdely, @Cast("l_float32*") float[] pscore, @Cast("l_int32") int debugflag );
public static native BOXA pixConnComp( PIX pixs, @Cast("PIXA**") PointerPointer ppixa, @Cast("l_int32") int connectivity );
public static native BOXA pixConnComp( PIX pixs, @ByPtrPtr PIXA ppixa, @Cast("l_int32") int connectivity );
public static native BOXA pixConnCompPixa( PIX pixs, @Cast("PIXA**") PointerPointer ppixa, @Cast("l_int32") int connectivity );
public static native BOXA pixConnCompPixa( PIX pixs, @ByPtrPtr PIXA ppixa, @Cast("l_int32") int connectivity );
public static native BOXA pixConnCompBB( PIX pixs, @Cast("l_int32") int connectivity );
public static native @Cast("l_int32") int pixCountConnComp( PIX pixs, @Cast("l_int32") int connectivity, @Cast("l_int32*") IntPointer pcount );
public static native @Cast("l_int32") int pixCountConnComp( PIX pixs, @Cast("l_int32") int connectivity, @Cast("l_int32*") IntBuffer pcount );
public static native @Cast("l_int32") int pixCountConnComp( PIX pixs, @Cast("l_int32") int connectivity, @Cast("l_int32*") int[] pcount );
public static native @Cast("l_int32") int nextOnPixelInRaster( PIX pixs, @Cast("l_int32") int xstart, @Cast("l_int32") int ystart, @Cast("l_int32*") IntPointer px, @Cast("l_int32*") IntPointer py );
public static native @Cast("l_int32") int nextOnPixelInRaster( PIX pixs, @Cast("l_int32") int xstart, @Cast("l_int32") int ystart, @Cast("l_int32*") IntBuffer px, @Cast("l_int32*") IntBuffer py );
public static native @Cast("l_int32") int nextOnPixelInRaster( PIX pixs, @Cast("l_int32") int xstart, @Cast("l_int32") int ystart, @Cast("l_int32*") int[] px, @Cast("l_int32*") int[] py );
public static native @Cast("l_int32") int nextOnPixelInRasterLow( @Cast("l_uint32*") IntPointer data, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpl, @Cast("l_int32") int xstart, @Cast("l_int32") int ystart, @Cast("l_int32*") IntPointer px, @Cast("l_int32*") IntPointer py );
public static native @Cast("l_int32") int nextOnPixelInRasterLow( @Cast("l_uint32*") IntBuffer data, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpl, @Cast("l_int32") int xstart, @Cast("l_int32") int ystart, @Cast("l_int32*") IntBuffer px, @Cast("l_int32*") IntBuffer py );
public static native @Cast("l_int32") int nextOnPixelInRasterLow( @Cast("l_uint32*") int[] data, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpl, @Cast("l_int32") int xstart, @Cast("l_int32") int ystart, @Cast("l_int32*") int[] px, @Cast("l_int32*") int[] py );
public static native BOX pixSeedfillBB( PIX pixs, L_STACK stack, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int connectivity );
public static native BOX pixSeedfill4BB( PIX pixs, L_STACK stack, @Cast("l_int32") int x, @Cast("l_int32") int y );
public static native BOX pixSeedfill8BB( PIX pixs, L_STACK stack, @Cast("l_int32") int x, @Cast("l_int32") int y );
public static native @Cast("l_int32") int pixSeedfill( PIX pixs, L_STACK stack, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int connectivity );
public static native @Cast("l_int32") int pixSeedfill4( PIX pixs, L_STACK stack, @Cast("l_int32") int x, @Cast("l_int32") int y );
public static native @Cast("l_int32") int pixSeedfill8( PIX pixs, L_STACK stack, @Cast("l_int32") int x, @Cast("l_int32") int y );
public static native @Cast("l_int32") int convertFilesTo1bpp( @Cast("const char*") BytePointer dirin, @Cast("const char*") BytePointer substr, @Cast("l_int32") int upscaling, @Cast("l_int32") int thresh, @Cast("l_int32") int firstpage, @Cast("l_int32") int npages, @Cast("const char*") BytePointer dirout, @Cast("l_int32") int outformat );
public static native @Cast("l_int32") int convertFilesTo1bpp( String dirin, String substr, @Cast("l_int32") int upscaling, @Cast("l_int32") int thresh, @Cast("l_int32") int firstpage, @Cast("l_int32") int npages, String dirout, @Cast("l_int32") int outformat );
public static native PIX pixBlockconv( PIX pix, @Cast("l_int32") int wc, @Cast("l_int32") int hc );
public static native PIX pixBlockconvGray( PIX pixs, PIX pixacc, @Cast("l_int32") int wc, @Cast("l_int32") int hc );
public static native PIX pixBlockconvAccum( PIX pixs );
public static native PIX pixBlockconvGrayUnnormalized( PIX pixs, @Cast("l_int32") int wc, @Cast("l_int32") int hc );
public static native PIX pixBlockconvTiled( PIX pix, @Cast("l_int32") int wc, @Cast("l_int32") int hc, @Cast("l_int32") int nx, @Cast("l_int32") int ny );
public static native PIX pixBlockconvGrayTile( PIX pixs, PIX pixacc, @Cast("l_int32") int wc, @Cast("l_int32") int hc );
public static native @Cast("l_int32") int pixWindowedStats( PIX pixs, @Cast("l_int32") int wc, @Cast("l_int32") int hc, @Cast("l_int32") int hasborder, @Cast("PIX**") PointerPointer ppixm, @Cast("PIX**") PointerPointer ppixms, @Cast("FPIX**") PointerPointer pfpixv, @Cast("FPIX**") PointerPointer pfpixrv );
public static native @Cast("l_int32") int pixWindowedStats( PIX pixs, @Cast("l_int32") int wc, @Cast("l_int32") int hc, @Cast("l_int32") int hasborder, @ByPtrPtr PIX ppixm, @ByPtrPtr PIX ppixms, @ByPtrPtr FPIX pfpixv, @ByPtrPtr FPIX pfpixrv );
public static native PIX pixWindowedMean( PIX pixs, @Cast("l_int32") int wc, @Cast("l_int32") int hc, @Cast("l_int32") int hasborder, @Cast("l_int32") int normflag );
public static native PIX pixWindowedMeanSquare( PIX pixs, @Cast("l_int32") int wc, @Cast("l_int32") int hc, @Cast("l_int32") int hasborder );
public static native @Cast("l_int32") int pixWindowedVariance( PIX pixm, PIX pixms, @Cast("FPIX**") PointerPointer pfpixv, @Cast("FPIX**") PointerPointer pfpixrv );
public static native @Cast("l_int32") int pixWindowedVariance( PIX pixm, PIX pixms, @ByPtrPtr FPIX pfpixv, @ByPtrPtr FPIX pfpixrv );
public static native DPIX pixMeanSquareAccum( PIX pixs );
public static native PIX pixBlockrank( PIX pixs, PIX pixacc, @Cast("l_int32") int wc, @Cast("l_int32") int hc, @Cast("l_float32") float rank );
public static native PIX pixBlocksum( PIX pixs, PIX pixacc, @Cast("l_int32") int wc, @Cast("l_int32") int hc );
public static native PIX pixCensusTransform( PIX pixs, @Cast("l_int32") int halfsize, PIX pixacc );
public static native PIX pixConvolve( PIX pixs, L_KERNEL kel, @Cast("l_int32") int outdepth, @Cast("l_int32") int normflag );
public static native PIX pixConvolveSep( PIX pixs, L_KERNEL kelx, L_KERNEL kely, @Cast("l_int32") int outdepth, @Cast("l_int32") int normflag );
public static native PIX pixConvolveRGB( PIX pixs, L_KERNEL kel );
public static native PIX pixConvolveRGBSep( PIX pixs, L_KERNEL kelx, L_KERNEL kely );
public static native FPIX fpixConvolve( FPIX fpixs, L_KERNEL kel, @Cast("l_int32") int normflag );
public static native FPIX fpixConvolveSep( FPIX fpixs, L_KERNEL kelx, L_KERNEL kely, @Cast("l_int32") int normflag );
public static native PIX pixConvolveWithBias( PIX pixs, L_KERNEL kel1, L_KERNEL kel2, @Cast("l_int32") int force8, @Cast("l_int32*") IntPointer pbias );
public static native PIX pixConvolveWithBias( PIX pixs, L_KERNEL kel1, L_KERNEL kel2, @Cast("l_int32") int force8, @Cast("l_int32*") IntBuffer pbias );
public static native PIX pixConvolveWithBias( PIX pixs, L_KERNEL kel1, L_KERNEL kel2, @Cast("l_int32") int force8, @Cast("l_int32*") int[] pbias );
public static native void l_setConvolveSampling( @Cast("l_int32") int xfact, @Cast("l_int32") int yfact );
public static native PIX pixAddGaussianNoise( PIX pixs, @Cast("l_float32") float stdev );
public static native @Cast("l_float32") float gaussDistribSampling(  );
public static native @Cast("l_int32") int pixCorrelationScore( PIX pix1, PIX pix2, @Cast("l_int32") int area1, @Cast("l_int32") int area2, @Cast("l_float32") float delx, @Cast("l_float32") float dely, @Cast("l_int32") int maxdiffw, @Cast("l_int32") int maxdiffh, @Cast("l_int32*") IntPointer tab, @Cast("l_float32*") FloatPointer pscore );
public static native @Cast("l_int32") int pixCorrelationScore( PIX pix1, PIX pix2, @Cast("l_int32") int area1, @Cast("l_int32") int area2, @Cast("l_float32") float delx, @Cast("l_float32") float dely, @Cast("l_int32") int maxdiffw, @Cast("l_int32") int maxdiffh, @Cast("l_int32*") IntBuffer tab, @Cast("l_float32*") FloatBuffer pscore );
public static native @Cast("l_int32") int pixCorrelationScore( PIX pix1, PIX pix2, @Cast("l_int32") int area1, @Cast("l_int32") int area2, @Cast("l_float32") float delx, @Cast("l_float32") float dely, @Cast("l_int32") int maxdiffw, @Cast("l_int32") int maxdiffh, @Cast("l_int32*") int[] tab, @Cast("l_float32*") float[] pscore );
public static native @Cast("l_int32") int pixCorrelationScoreThresholded( PIX pix1, PIX pix2, @Cast("l_int32") int area1, @Cast("l_int32") int area2, @Cast("l_float32") float delx, @Cast("l_float32") float dely, @Cast("l_int32") int maxdiffw, @Cast("l_int32") int maxdiffh, @Cast("l_int32*") IntPointer tab, @Cast("l_int32*") IntPointer downcount, @Cast("l_float32") float score_threshold );
public static native @Cast("l_int32") int pixCorrelationScoreThresholded( PIX pix1, PIX pix2, @Cast("l_int32") int area1, @Cast("l_int32") int area2, @Cast("l_float32") float delx, @Cast("l_float32") float dely, @Cast("l_int32") int maxdiffw, @Cast("l_int32") int maxdiffh, @Cast("l_int32*") IntBuffer tab, @Cast("l_int32*") IntBuffer downcount, @Cast("l_float32") float score_threshold );
public static native @Cast("l_int32") int pixCorrelationScoreThresholded( PIX pix1, PIX pix2, @Cast("l_int32") int area1, @Cast("l_int32") int area2, @Cast("l_float32") float delx, @Cast("l_float32") float dely, @Cast("l_int32") int maxdiffw, @Cast("l_int32") int maxdiffh, @Cast("l_int32*") int[] tab, @Cast("l_int32*") int[] downcount, @Cast("l_float32") float score_threshold );
public static native @Cast("l_int32") int pixCorrelationScoreSimple( PIX pix1, PIX pix2, @Cast("l_int32") int area1, @Cast("l_int32") int area2, @Cast("l_float32") float delx, @Cast("l_float32") float dely, @Cast("l_int32") int maxdiffw, @Cast("l_int32") int maxdiffh, @Cast("l_int32*") IntPointer tab, @Cast("l_float32*") FloatPointer pscore );
public static native @Cast("l_int32") int pixCorrelationScoreSimple( PIX pix1, PIX pix2, @Cast("l_int32") int area1, @Cast("l_int32") int area2, @Cast("l_float32") float delx, @Cast("l_float32") float dely, @Cast("l_int32") int maxdiffw, @Cast("l_int32") int maxdiffh, @Cast("l_int32*") IntBuffer tab, @Cast("l_float32*") FloatBuffer pscore );
public static native @Cast("l_int32") int pixCorrelationScoreSimple( PIX pix1, PIX pix2, @Cast("l_int32") int area1, @Cast("l_int32") int area2, @Cast("l_float32") float delx, @Cast("l_float32") float dely, @Cast("l_int32") int maxdiffw, @Cast("l_int32") int maxdiffh, @Cast("l_int32*") int[] tab, @Cast("l_float32*") float[] pscore );
public static native @Cast("l_int32") int pixCorrelationScoreShifted( PIX pix1, PIX pix2, @Cast("l_int32") int area1, @Cast("l_int32") int area2, @Cast("l_int32") int delx, @Cast("l_int32") int dely, @Cast("l_int32*") IntPointer tab, @Cast("l_float32*") FloatPointer pscore );
public static native @Cast("l_int32") int pixCorrelationScoreShifted( PIX pix1, PIX pix2, @Cast("l_int32") int area1, @Cast("l_int32") int area2, @Cast("l_int32") int delx, @Cast("l_int32") int dely, @Cast("l_int32*") IntBuffer tab, @Cast("l_float32*") FloatBuffer pscore );
public static native @Cast("l_int32") int pixCorrelationScoreShifted( PIX pix1, PIX pix2, @Cast("l_int32") int area1, @Cast("l_int32") int area2, @Cast("l_int32") int delx, @Cast("l_int32") int dely, @Cast("l_int32*") int[] tab, @Cast("l_float32*") float[] pscore );
public static native L_DEWARP dewarpCreate( PIX pixs, @Cast("l_int32") int pageno );
public static native L_DEWARP dewarpCreateRef( @Cast("l_int32") int pageno, @Cast("l_int32") int refpage );
public static native void dewarpDestroy( @Cast("L_DEWARP**") PointerPointer pdew );
public static native void dewarpDestroy( @ByPtrPtr L_DEWARP pdew );
public static native L_DEWARPA dewarpaCreate( @Cast("l_int32") int nptrs, @Cast("l_int32") int sampling, @Cast("l_int32") int redfactor, @Cast("l_int32") int minlines, @Cast("l_int32") int maxdist );
public static native L_DEWARPA dewarpaCreateFromPixacomp( PIXAC pixac, @Cast("l_int32") int useboth, @Cast("l_int32") int sampling, @Cast("l_int32") int minlines, @Cast("l_int32") int maxdist );
public static native void dewarpaDestroy( @Cast("L_DEWARPA**") PointerPointer pdewa );
public static native void dewarpaDestroy( @ByPtrPtr L_DEWARPA pdewa );
public static native @Cast("l_int32") int dewarpaDestroyDewarp( L_DEWARPA dewa, @Cast("l_int32") int pageno );
public static native @Cast("l_int32") int dewarpaInsertDewarp( L_DEWARPA dewa, L_DEWARP dew );
public static native L_DEWARP dewarpaGetDewarp( L_DEWARPA dewa, @Cast("l_int32") int index );
public static native @Cast("l_int32") int dewarpaSetCurvatures( L_DEWARPA dewa, @Cast("l_int32") int max_linecurv, @Cast("l_int32") int min_diff_linecurv, @Cast("l_int32") int max_diff_linecurv, @Cast("l_int32") int max_edgecurv, @Cast("l_int32") int max_diff_edgecurv, @Cast("l_int32") int max_edgeslope );
public static native @Cast("l_int32") int dewarpaUseBothArrays( L_DEWARPA dewa, @Cast("l_int32") int useboth );
public static native @Cast("l_int32") int dewarpaSetMaxDistance( L_DEWARPA dewa, @Cast("l_int32") int maxdist );
public static native L_DEWARP dewarpRead( @Cast("const char*") BytePointer filename );
public static native L_DEWARP dewarpRead( String filename );
public static native L_DEWARP dewarpReadStream( @Cast("FILE*") Pointer fp );
public static native @Cast("l_int32") int dewarpWrite( @Cast("const char*") BytePointer filename, L_DEWARP dew );
public static native @Cast("l_int32") int dewarpWrite( String filename, L_DEWARP dew );
public static native @Cast("l_int32") int dewarpWriteStream( @Cast("FILE*") Pointer fp, L_DEWARP dew );
public static native L_DEWARPA dewarpaRead( @Cast("const char*") BytePointer filename );
public static native L_DEWARPA dewarpaRead( String filename );
public static native L_DEWARPA dewarpaReadStream( @Cast("FILE*") Pointer fp );
public static native @Cast("l_int32") int dewarpaWrite( @Cast("const char*") BytePointer filename, L_DEWARPA dewa );
public static native @Cast("l_int32") int dewarpaWrite( String filename, L_DEWARPA dewa );
public static native @Cast("l_int32") int dewarpaWriteStream( @Cast("FILE*") Pointer fp, L_DEWARPA dewa );
public static native @Cast("l_int32") int dewarpBuildPageModel( L_DEWARP dew, @Cast("const char*") BytePointer debugfile );
public static native @Cast("l_int32") int dewarpBuildPageModel( L_DEWARP dew, String debugfile );
public static native @Cast("l_int32") int dewarpFindVertDisparity( L_DEWARP dew, PTAA ptaa, @Cast("l_int32") int rotflag );
public static native @Cast("l_int32") int dewarpFindHorizDisparity( L_DEWARP dew, PTAA ptaa );
public static native PTAA dewarpGetTextlineCenters( PIX pixs, @Cast("l_int32") int debugflag );
public static native PTAA dewarpRemoveShortLines( PIX pixs, PTAA ptaas, @Cast("l_float32") float fract, @Cast("l_int32") int debugflag );
public static native @Cast("l_int32") int dewarpBuildLineModel( L_DEWARP dew, @Cast("l_int32") int opensize, @Cast("const char*") BytePointer debugfile );
public static native @Cast("l_int32") int dewarpBuildLineModel( L_DEWARP dew, @Cast("l_int32") int opensize, String debugfile );
public static native @Cast("l_int32") int dewarpaModelStatus( L_DEWARPA dewa, @Cast("l_int32") int pageno, @Cast("l_int32*") IntPointer pvsuccess, @Cast("l_int32*") IntPointer phsuccess );
public static native @Cast("l_int32") int dewarpaModelStatus( L_DEWARPA dewa, @Cast("l_int32") int pageno, @Cast("l_int32*") IntBuffer pvsuccess, @Cast("l_int32*") IntBuffer phsuccess );
public static native @Cast("l_int32") int dewarpaModelStatus( L_DEWARPA dewa, @Cast("l_int32") int pageno, @Cast("l_int32*") int[] pvsuccess, @Cast("l_int32*") int[] phsuccess );
public static native @Cast("l_int32") int dewarpaApplyDisparity( L_DEWARPA dewa, @Cast("l_int32") int pageno, PIX pixs, @Cast("l_int32") int grayin, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("PIX**") PointerPointer ppixd, @Cast("const char*") BytePointer debugfile );
public static native @Cast("l_int32") int dewarpaApplyDisparity( L_DEWARPA dewa, @Cast("l_int32") int pageno, PIX pixs, @Cast("l_int32") int grayin, @Cast("l_int32") int x, @Cast("l_int32") int y, @ByPtrPtr PIX ppixd, @Cast("const char*") BytePointer debugfile );
public static native @Cast("l_int32") int dewarpaApplyDisparity( L_DEWARPA dewa, @Cast("l_int32") int pageno, PIX pixs, @Cast("l_int32") int grayin, @Cast("l_int32") int x, @Cast("l_int32") int y, @ByPtrPtr PIX ppixd, String debugfile );
public static native @Cast("l_int32") int dewarpaApplyDisparityBoxa( L_DEWARPA dewa, @Cast("l_int32") int pageno, PIX pixs, BOXA boxas, @Cast("l_int32") int mapdir, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("BOXA**") PointerPointer pboxad, @Cast("const char*") BytePointer debugfile );
public static native @Cast("l_int32") int dewarpaApplyDisparityBoxa( L_DEWARPA dewa, @Cast("l_int32") int pageno, PIX pixs, BOXA boxas, @Cast("l_int32") int mapdir, @Cast("l_int32") int x, @Cast("l_int32") int y, @ByPtrPtr BOXA pboxad, @Cast("const char*") BytePointer debugfile );
public static native @Cast("l_int32") int dewarpaApplyDisparityBoxa( L_DEWARPA dewa, @Cast("l_int32") int pageno, PIX pixs, BOXA boxas, @Cast("l_int32") int mapdir, @Cast("l_int32") int x, @Cast("l_int32") int y, @ByPtrPtr BOXA pboxad, String debugfile );
public static native @Cast("l_int32") int dewarpMinimize( L_DEWARP dew );
public static native @Cast("l_int32") int dewarpPopulateFullRes( L_DEWARP dew, PIX pix, @Cast("l_int32") int x, @Cast("l_int32") int y );
public static native @Cast("l_int32") int dewarpSinglePage( PIX pixs, @Cast("l_int32") int thresh, @Cast("l_int32") int adaptive, @Cast("l_int32") int use_both, @Cast("PIX**") PointerPointer ppixd, @Cast("L_DEWARPA**") PointerPointer pdewa, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int dewarpSinglePage( PIX pixs, @Cast("l_int32") int thresh, @Cast("l_int32") int adaptive, @Cast("l_int32") int use_both, @ByPtrPtr PIX ppixd, @ByPtrPtr L_DEWARPA pdewa, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int dewarpSinglePageInit( PIX pixs, @Cast("l_int32") int thresh, @Cast("l_int32") int adaptive, @Cast("l_int32") int use_both, @Cast("PIX**") PointerPointer ppixb, @Cast("L_DEWARPA**") PointerPointer pdewa );
public static native @Cast("l_int32") int dewarpSinglePageInit( PIX pixs, @Cast("l_int32") int thresh, @Cast("l_int32") int adaptive, @Cast("l_int32") int use_both, @ByPtrPtr PIX ppixb, @ByPtrPtr L_DEWARPA pdewa );
public static native @Cast("l_int32") int dewarpSinglePageRun( PIX pixs, PIX pixb, L_DEWARPA dewa, @Cast("PIX**") PointerPointer ppixd, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int dewarpSinglePageRun( PIX pixs, PIX pixb, L_DEWARPA dewa, @ByPtrPtr PIX ppixd, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int dewarpaListPages( L_DEWARPA dewa );
public static native @Cast("l_int32") int dewarpaSetValidModels( L_DEWARPA dewa, @Cast("l_int32") int notests, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int dewarpaInsertRefModels( L_DEWARPA dewa, @Cast("l_int32") int notests, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int dewarpaStripRefModels( L_DEWARPA dewa );
public static native @Cast("l_int32") int dewarpaRestoreModels( L_DEWARPA dewa );
public static native @Cast("l_int32") int dewarpaInfo( @Cast("FILE*") Pointer fp, L_DEWARPA dewa );
public static native @Cast("l_int32") int dewarpaModelStats( L_DEWARPA dewa, @Cast("l_int32*") IntPointer pnnone, @Cast("l_int32*") IntPointer pnvsuccess, @Cast("l_int32*") IntPointer pnvvalid, @Cast("l_int32*") IntPointer pnhsuccess, @Cast("l_int32*") IntPointer pnhvalid, @Cast("l_int32*") IntPointer pnref );
public static native @Cast("l_int32") int dewarpaModelStats( L_DEWARPA dewa, @Cast("l_int32*") IntBuffer pnnone, @Cast("l_int32*") IntBuffer pnvsuccess, @Cast("l_int32*") IntBuffer pnvvalid, @Cast("l_int32*") IntBuffer pnhsuccess, @Cast("l_int32*") IntBuffer pnhvalid, @Cast("l_int32*") IntBuffer pnref );
public static native @Cast("l_int32") int dewarpaModelStats( L_DEWARPA dewa, @Cast("l_int32*") int[] pnnone, @Cast("l_int32*") int[] pnvsuccess, @Cast("l_int32*") int[] pnvvalid, @Cast("l_int32*") int[] pnhsuccess, @Cast("l_int32*") int[] pnhvalid, @Cast("l_int32*") int[] pnref );
public static native @Cast("l_int32") int dewarpaShowArrays( L_DEWARPA dewa, @Cast("l_float32") float scalefact, @Cast("l_int32") int first, @Cast("l_int32") int last, @Cast("const char*") BytePointer fontdir );
public static native @Cast("l_int32") int dewarpaShowArrays( L_DEWARPA dewa, @Cast("l_float32") float scalefact, @Cast("l_int32") int first, @Cast("l_int32") int last, String fontdir );
public static native @Cast("l_int32") int dewarpDebug( L_DEWARP dew, @Cast("const char*") BytePointer subdir, @Cast("l_int32") int index );
public static native @Cast("l_int32") int dewarpDebug( L_DEWARP dew, String subdir, @Cast("l_int32") int index );
public static native @Cast("l_int32") int dewarpShowResults( L_DEWARPA dewa, SARRAY sa, BOXA boxa, @Cast("l_int32") int firstpage, @Cast("l_int32") int lastpage, @Cast("const char*") BytePointer fontdir, @Cast("const char*") BytePointer pdfout );
public static native @Cast("l_int32") int dewarpShowResults( L_DEWARPA dewa, SARRAY sa, BOXA boxa, @Cast("l_int32") int firstpage, @Cast("l_int32") int lastpage, String fontdir, String pdfout );
public static native L_DNA l_dnaCreate( @Cast("l_int32") int n );
public static native L_DNA l_dnaCreateFromIArray( @Cast("l_int32*") IntPointer iarray, @Cast("l_int32") int size );
public static native L_DNA l_dnaCreateFromIArray( @Cast("l_int32*") IntBuffer iarray, @Cast("l_int32") int size );
public static native L_DNA l_dnaCreateFromIArray( @Cast("l_int32*") int[] iarray, @Cast("l_int32") int size );
public static native L_DNA l_dnaCreateFromDArray( @Cast("l_float64*") DoublePointer darray, @Cast("l_int32") int size, @Cast("l_int32") int copyflag );
public static native L_DNA l_dnaCreateFromDArray( @Cast("l_float64*") DoubleBuffer darray, @Cast("l_int32") int size, @Cast("l_int32") int copyflag );
public static native L_DNA l_dnaCreateFromDArray( @Cast("l_float64*") double[] darray, @Cast("l_int32") int size, @Cast("l_int32") int copyflag );
public static native L_DNA l_dnaMakeSequence( @Cast("l_float64") double startval, @Cast("l_float64") double increment, @Cast("l_int32") int size );
public static native void l_dnaDestroy( @Cast("L_DNA**") PointerPointer pda );
public static native void l_dnaDestroy( @ByPtrPtr L_DNA pda );
public static native L_DNA l_dnaCopy( L_DNA da );
public static native L_DNA l_dnaClone( L_DNA da );
public static native @Cast("l_int32") int l_dnaEmpty( L_DNA da );
public static native @Cast("l_int32") int l_dnaAddNumber( L_DNA da, @Cast("l_float64") double val );
public static native @Cast("l_int32") int l_dnaInsertNumber( L_DNA da, @Cast("l_int32") int index, @Cast("l_float64") double val );
public static native @Cast("l_int32") int l_dnaRemoveNumber( L_DNA da, @Cast("l_int32") int index );
public static native @Cast("l_int32") int l_dnaReplaceNumber( L_DNA da, @Cast("l_int32") int index, @Cast("l_float64") double val );
public static native @Cast("l_int32") int l_dnaGetCount( L_DNA da );
public static native @Cast("l_int32") int l_dnaSetCount( L_DNA da, @Cast("l_int32") int newcount );
public static native @Cast("l_int32") int l_dnaGetDValue( L_DNA da, @Cast("l_int32") int index, @Cast("l_float64*") DoublePointer pval );
public static native @Cast("l_int32") int l_dnaGetDValue( L_DNA da, @Cast("l_int32") int index, @Cast("l_float64*") DoubleBuffer pval );
public static native @Cast("l_int32") int l_dnaGetDValue( L_DNA da, @Cast("l_int32") int index, @Cast("l_float64*") double[] pval );
public static native @Cast("l_int32") int l_dnaGetIValue( L_DNA da, @Cast("l_int32") int index, @Cast("l_int32*") IntPointer pival );
public static native @Cast("l_int32") int l_dnaGetIValue( L_DNA da, @Cast("l_int32") int index, @Cast("l_int32*") IntBuffer pival );
public static native @Cast("l_int32") int l_dnaGetIValue( L_DNA da, @Cast("l_int32") int index, @Cast("l_int32*") int[] pival );
public static native @Cast("l_int32") int l_dnaSetValue( L_DNA da, @Cast("l_int32") int index, @Cast("l_float64") double val );
public static native @Cast("l_int32") int l_dnaShiftValue( L_DNA da, @Cast("l_int32") int index, @Cast("l_float64") double diff );
public static native @Cast("l_int32*") IntPointer l_dnaGetIArray( L_DNA da );
public static native @Cast("l_float64*") DoublePointer l_dnaGetDArray( L_DNA da, @Cast("l_int32") int copyflag );
public static native @Cast("l_int32") int l_dnaGetRefcount( L_DNA da );
public static native @Cast("l_int32") int l_dnaChangeRefcount( L_DNA da, @Cast("l_int32") int delta );
public static native @Cast("l_int32") int l_dnaGetParameters( L_DNA da, @Cast("l_float64*") DoublePointer pstartx, @Cast("l_float64*") DoublePointer pdelx );
public static native @Cast("l_int32") int l_dnaGetParameters( L_DNA da, @Cast("l_float64*") DoubleBuffer pstartx, @Cast("l_float64*") DoubleBuffer pdelx );
public static native @Cast("l_int32") int l_dnaGetParameters( L_DNA da, @Cast("l_float64*") double[] pstartx, @Cast("l_float64*") double[] pdelx );
public static native @Cast("l_int32") int l_dnaSetParameters( L_DNA da, @Cast("l_float64") double startx, @Cast("l_float64") double delx );
public static native @Cast("l_int32") int l_dnaCopyParameters( L_DNA dad, L_DNA das );
public static native L_DNA l_dnaRead( @Cast("const char*") BytePointer filename );
public static native L_DNA l_dnaRead( String filename );
public static native L_DNA l_dnaReadStream( @Cast("FILE*") Pointer fp );
public static native @Cast("l_int32") int l_dnaWrite( @Cast("const char*") BytePointer filename, L_DNA da );
public static native @Cast("l_int32") int l_dnaWrite( String filename, L_DNA da );
public static native @Cast("l_int32") int l_dnaWriteStream( @Cast("FILE*") Pointer fp, L_DNA da );
public static native L_DNAA l_dnaaCreate( @Cast("l_int32") int n );
public static native void l_dnaaDestroy( @Cast("L_DNAA**") PointerPointer pdaa );
public static native void l_dnaaDestroy( @ByPtrPtr L_DNAA pdaa );
public static native @Cast("l_int32") int l_dnaaAddDna( L_DNAA daa, L_DNA da, @Cast("l_int32") int copyflag );
public static native @Cast("l_int32") int l_dnaaGetCount( L_DNAA daa );
public static native @Cast("l_int32") int l_dnaaGetDnaCount( L_DNAA daa, @Cast("l_int32") int index );
public static native @Cast("l_int32") int l_dnaaGetNumberCount( L_DNAA daa );
public static native L_DNA l_dnaaGetDna( L_DNAA daa, @Cast("l_int32") int index, @Cast("l_int32") int accessflag );
public static native @Cast("l_int32") int l_dnaaReplaceDna( L_DNAA daa, @Cast("l_int32") int index, L_DNA da );
public static native @Cast("l_int32") int l_dnaaGetValue( L_DNAA daa, @Cast("l_int32") int i, @Cast("l_int32") int j, @Cast("l_float64*") DoublePointer pval );
public static native @Cast("l_int32") int l_dnaaGetValue( L_DNAA daa, @Cast("l_int32") int i, @Cast("l_int32") int j, @Cast("l_float64*") DoubleBuffer pval );
public static native @Cast("l_int32") int l_dnaaGetValue( L_DNAA daa, @Cast("l_int32") int i, @Cast("l_int32") int j, @Cast("l_float64*") double[] pval );
public static native @Cast("l_int32") int l_dnaaAddNumber( L_DNAA daa, @Cast("l_int32") int index, @Cast("l_float64") double val );
public static native L_DNAA l_dnaaRead( @Cast("const char*") BytePointer filename );
public static native L_DNAA l_dnaaRead( String filename );
public static native L_DNAA l_dnaaReadStream( @Cast("FILE*") Pointer fp );
public static native @Cast("l_int32") int l_dnaaWrite( @Cast("const char*") BytePointer filename, L_DNAA daa );
public static native @Cast("l_int32") int l_dnaaWrite( String filename, L_DNAA daa );
public static native @Cast("l_int32") int l_dnaaWriteStream( @Cast("FILE*") Pointer fp, L_DNAA daa );
public static native L_DNA l_dnaMakeDelta( L_DNA das );
public static native NUMA l_dnaConvertToNuma( L_DNA da );
public static native L_DNA numaConvertToDna( NUMA na );
public static native @Cast("l_int32") int l_dnaJoin( L_DNA dad, L_DNA das, @Cast("l_int32") int istart, @Cast("l_int32") int iend );
public static native PIX pixMorphDwa_2( PIX pixd, PIX pixs, @Cast("l_int32") int operation, @Cast("char*") BytePointer selname );
public static native PIX pixMorphDwa_2( PIX pixd, PIX pixs, @Cast("l_int32") int operation, @Cast("char*") ByteBuffer selname );
public static native PIX pixMorphDwa_2( PIX pixd, PIX pixs, @Cast("l_int32") int operation, @Cast("char*") byte[] selname );
public static native PIX pixFMorphopGen_2( PIX pixd, PIX pixs, @Cast("l_int32") int operation, @Cast("char*") BytePointer selname );
public static native PIX pixFMorphopGen_2( PIX pixd, PIX pixs, @Cast("l_int32") int operation, @Cast("char*") ByteBuffer selname );
public static native PIX pixFMorphopGen_2( PIX pixd, PIX pixs, @Cast("l_int32") int operation, @Cast("char*") byte[] selname );
public static native @Cast("l_int32") int fmorphopgen_low_2( @Cast("l_uint32*") IntPointer datad, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntPointer datas, @Cast("l_int32") int wpls, @Cast("l_int32") int index );
public static native @Cast("l_int32") int fmorphopgen_low_2( @Cast("l_uint32*") IntBuffer datad, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntBuffer datas, @Cast("l_int32") int wpls, @Cast("l_int32") int index );
public static native @Cast("l_int32") int fmorphopgen_low_2( @Cast("l_uint32*") int[] datad, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpld, @Cast("l_uint32*") int[] datas, @Cast("l_int32") int wpls, @Cast("l_int32") int index );
public static native PIX pixSobelEdgeFilter( PIX pixs, @Cast("l_int32") int orientflag );
public static native PIX pixTwoSidedEdgeFilter( PIX pixs, @Cast("l_int32") int orientflag );
public static native @Cast("l_int32") int pixMeasureEdgeSmoothness( PIX pixs, @Cast("l_int32") int side, @Cast("l_int32") int minjump, @Cast("l_int32") int minreversal, @Cast("l_float32*") FloatPointer pjpl, @Cast("l_float32*") FloatPointer pjspl, @Cast("l_float32*") FloatPointer prpl, @Cast("const char*") BytePointer debugfile );
public static native @Cast("l_int32") int pixMeasureEdgeSmoothness( PIX pixs, @Cast("l_int32") int side, @Cast("l_int32") int minjump, @Cast("l_int32") int minreversal, @Cast("l_float32*") FloatBuffer pjpl, @Cast("l_float32*") FloatBuffer pjspl, @Cast("l_float32*") FloatBuffer prpl, String debugfile );
public static native @Cast("l_int32") int pixMeasureEdgeSmoothness( PIX pixs, @Cast("l_int32") int side, @Cast("l_int32") int minjump, @Cast("l_int32") int minreversal, @Cast("l_float32*") float[] pjpl, @Cast("l_float32*") float[] pjspl, @Cast("l_float32*") float[] prpl, @Cast("const char*") BytePointer debugfile );
public static native @Cast("l_int32") int pixMeasureEdgeSmoothness( PIX pixs, @Cast("l_int32") int side, @Cast("l_int32") int minjump, @Cast("l_int32") int minreversal, @Cast("l_float32*") FloatPointer pjpl, @Cast("l_float32*") FloatPointer pjspl, @Cast("l_float32*") FloatPointer prpl, String debugfile );
public static native @Cast("l_int32") int pixMeasureEdgeSmoothness( PIX pixs, @Cast("l_int32") int side, @Cast("l_int32") int minjump, @Cast("l_int32") int minreversal, @Cast("l_float32*") FloatBuffer pjpl, @Cast("l_float32*") FloatBuffer pjspl, @Cast("l_float32*") FloatBuffer prpl, @Cast("const char*") BytePointer debugfile );
public static native @Cast("l_int32") int pixMeasureEdgeSmoothness( PIX pixs, @Cast("l_int32") int side, @Cast("l_int32") int minjump, @Cast("l_int32") int minreversal, @Cast("l_float32*") float[] pjpl, @Cast("l_float32*") float[] pjspl, @Cast("l_float32*") float[] prpl, String debugfile );
public static native NUMA pixGetEdgeProfile( PIX pixs, @Cast("l_int32") int side, @Cast("const char*") BytePointer debugfile );
public static native NUMA pixGetEdgeProfile( PIX pixs, @Cast("l_int32") int side, String debugfile );
public static native @Cast("l_int32") int pixGetLastOffPixelInRun( PIX pixs, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int direction, @Cast("l_int32*") IntPointer ploc );
public static native @Cast("l_int32") int pixGetLastOffPixelInRun( PIX pixs, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int direction, @Cast("l_int32*") IntBuffer ploc );
public static native @Cast("l_int32") int pixGetLastOffPixelInRun( PIX pixs, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int direction, @Cast("l_int32*") int[] ploc );
public static native @Cast("l_int32") int pixGetLastOnPixelInRun( PIX pixs, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int direction, @Cast("l_int32*") IntPointer ploc );
public static native @Cast("l_int32") int pixGetLastOnPixelInRun( PIX pixs, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int direction, @Cast("l_int32*") IntBuffer ploc );
public static native @Cast("l_int32") int pixGetLastOnPixelInRun( PIX pixs, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int direction, @Cast("l_int32*") int[] ploc );
public static native @Cast("char*") BytePointer encodeBase64( @Cast("l_uint8*") BytePointer inarray, @Cast("l_int32") int insize, @Cast("l_int32*") IntPointer poutsize );
public static native @Cast("char*") ByteBuffer encodeBase64( @Cast("l_uint8*") ByteBuffer inarray, @Cast("l_int32") int insize, @Cast("l_int32*") IntBuffer poutsize );
public static native @Cast("char*") byte[] encodeBase64( @Cast("l_uint8*") byte[] inarray, @Cast("l_int32") int insize, @Cast("l_int32*") int[] poutsize );
public static native @Cast("l_uint8*") BytePointer decodeBase64( @Cast("const char*") BytePointer inarray, @Cast("l_int32") int insize, @Cast("l_int32*") IntPointer poutsize );
public static native @Cast("l_uint8*") ByteBuffer decodeBase64( String inarray, @Cast("l_int32") int insize, @Cast("l_int32*") IntBuffer poutsize );
public static native @Cast("l_uint8*") byte[] decodeBase64( @Cast("const char*") BytePointer inarray, @Cast("l_int32") int insize, @Cast("l_int32*") int[] poutsize );
public static native @Cast("l_uint8*") BytePointer decodeBase64( String inarray, @Cast("l_int32") int insize, @Cast("l_int32*") IntPointer poutsize );
public static native @Cast("l_uint8*") ByteBuffer decodeBase64( @Cast("const char*") BytePointer inarray, @Cast("l_int32") int insize, @Cast("l_int32*") IntBuffer poutsize );
public static native @Cast("l_uint8*") byte[] decodeBase64( String inarray, @Cast("l_int32") int insize, @Cast("l_int32*") int[] poutsize );
public static native @Cast("char*") BytePointer encodeAscii85( @Cast("l_uint8*") BytePointer inarray, @Cast("l_int32") int insize, @Cast("l_int32*") IntPointer poutsize );
public static native @Cast("char*") ByteBuffer encodeAscii85( @Cast("l_uint8*") ByteBuffer inarray, @Cast("l_int32") int insize, @Cast("l_int32*") IntBuffer poutsize );
public static native @Cast("char*") byte[] encodeAscii85( @Cast("l_uint8*") byte[] inarray, @Cast("l_int32") int insize, @Cast("l_int32*") int[] poutsize );
public static native @Cast("l_uint8*") BytePointer decodeAscii85( @Cast("char*") BytePointer inarray, @Cast("l_int32") int insize, @Cast("l_int32*") IntPointer poutsize );
public static native @Cast("l_uint8*") ByteBuffer decodeAscii85( @Cast("char*") ByteBuffer inarray, @Cast("l_int32") int insize, @Cast("l_int32*") IntBuffer poutsize );
public static native @Cast("l_uint8*") byte[] decodeAscii85( @Cast("char*") byte[] inarray, @Cast("l_int32") int insize, @Cast("l_int32*") int[] poutsize );
public static native @Cast("char*") BytePointer reformatPacked64( @Cast("char*") BytePointer inarray, @Cast("l_int32") int insize, @Cast("l_int32") int leadspace, @Cast("l_int32") int linechars, @Cast("l_int32") int addquotes, @Cast("l_int32*") IntPointer poutsize );
public static native @Cast("char*") ByteBuffer reformatPacked64( @Cast("char*") ByteBuffer inarray, @Cast("l_int32") int insize, @Cast("l_int32") int leadspace, @Cast("l_int32") int linechars, @Cast("l_int32") int addquotes, @Cast("l_int32*") IntBuffer poutsize );
public static native @Cast("char*") byte[] reformatPacked64( @Cast("char*") byte[] inarray, @Cast("l_int32") int insize, @Cast("l_int32") int leadspace, @Cast("l_int32") int linechars, @Cast("l_int32") int addquotes, @Cast("l_int32*") int[] poutsize );
public static native PIX pixGammaTRC( PIX pixd, PIX pixs, @Cast("l_float32") float gamma, @Cast("l_int32") int minval, @Cast("l_int32") int maxval );
public static native PIX pixGammaTRCMasked( PIX pixd, PIX pixs, PIX pixm, @Cast("l_float32") float gamma, @Cast("l_int32") int minval, @Cast("l_int32") int maxval );
public static native PIX pixGammaTRCWithAlpha( PIX pixd, PIX pixs, @Cast("l_float32") float gamma, @Cast("l_int32") int minval, @Cast("l_int32") int maxval );
public static native NUMA numaGammaTRC( @Cast("l_float32") float gamma, @Cast("l_int32") int minval, @Cast("l_int32") int maxval );
public static native PIX pixContrastTRC( PIX pixd, PIX pixs, @Cast("l_float32") float factor );
public static native PIX pixContrastTRCMasked( PIX pixd, PIX pixs, PIX pixm, @Cast("l_float32") float factor );
public static native NUMA numaContrastTRC( @Cast("l_float32") float factor );
public static native PIX pixEqualizeTRC( PIX pixd, PIX pixs, @Cast("l_float32") float fract, @Cast("l_int32") int factor );
public static native NUMA numaEqualizeTRC( PIX pix, @Cast("l_float32") float fract, @Cast("l_int32") int factor );
public static native @Cast("l_int32") int pixTRCMap( PIX pixs, PIX pixm, NUMA na );
public static native PIX pixUnsharpMasking( PIX pixs, @Cast("l_int32") int halfwidth, @Cast("l_float32") float fract );
public static native PIX pixUnsharpMaskingGray( PIX pixs, @Cast("l_int32") int halfwidth, @Cast("l_float32") float fract );
public static native PIX pixUnsharpMaskingFast( PIX pixs, @Cast("l_int32") int halfwidth, @Cast("l_float32") float fract, @Cast("l_int32") int direction );
public static native PIX pixUnsharpMaskingGrayFast( PIX pixs, @Cast("l_int32") int halfwidth, @Cast("l_float32") float fract, @Cast("l_int32") int direction );
public static native PIX pixUnsharpMaskingGray1D( PIX pixs, @Cast("l_int32") int halfwidth, @Cast("l_float32") float fract, @Cast("l_int32") int direction );
public static native PIX pixUnsharpMaskingGray2D( PIX pixs, @Cast("l_int32") int halfwidth, @Cast("l_float32") float fract );
public static native PIX pixModifyHue( PIX pixd, PIX pixs, @Cast("l_float32") float fract );
public static native PIX pixModifySaturation( PIX pixd, PIX pixs, @Cast("l_float32") float fract );
public static native @Cast("l_int32") int pixMeasureSaturation( PIX pixs, @Cast("l_int32") int factor, @Cast("l_float32*") FloatPointer psat );
public static native @Cast("l_int32") int pixMeasureSaturation( PIX pixs, @Cast("l_int32") int factor, @Cast("l_float32*") FloatBuffer psat );
public static native @Cast("l_int32") int pixMeasureSaturation( PIX pixs, @Cast("l_int32") int factor, @Cast("l_float32*") float[] psat );
public static native PIX pixModifyBrightness( PIX pixd, PIX pixs, @Cast("l_float32") float fract );
public static native PIX pixColorShiftRGB( PIX pixs, @Cast("l_float32") float rfract, @Cast("l_float32") float gfract, @Cast("l_float32") float bfract );
public static native PIX pixMultConstantColor( PIX pixs, @Cast("l_float32") float rfact, @Cast("l_float32") float gfact, @Cast("l_float32") float bfact );
public static native PIX pixMultMatrixColor( PIX pixs, L_KERNEL kel );
public static native PIX pixHalfEdgeByBandpass( PIX pixs, @Cast("l_int32") int sm1h, @Cast("l_int32") int sm1v, @Cast("l_int32") int sm2h, @Cast("l_int32") int sm2v );
public static native @Cast("l_int32") int fhmtautogen( SELA sela, @Cast("l_int32") int fileindex, @Cast("const char*") BytePointer filename );
public static native @Cast("l_int32") int fhmtautogen( SELA sela, @Cast("l_int32") int fileindex, String filename );
public static native @Cast("l_int32") int fhmtautogen1( SELA sela, @Cast("l_int32") int fileindex, @Cast("const char*") BytePointer filename );
public static native @Cast("l_int32") int fhmtautogen1( SELA sela, @Cast("l_int32") int fileindex, String filename );
public static native @Cast("l_int32") int fhmtautogen2( SELA sela, @Cast("l_int32") int fileindex, @Cast("const char*") BytePointer filename );
public static native @Cast("l_int32") int fhmtautogen2( SELA sela, @Cast("l_int32") int fileindex, String filename );
public static native PIX pixHMTDwa_1( PIX pixd, PIX pixs, @Cast("const char*") BytePointer selname );
public static native PIX pixHMTDwa_1( PIX pixd, PIX pixs, String selname );
public static native PIX pixFHMTGen_1( PIX pixd, PIX pixs, @Cast("const char*") BytePointer selname );
public static native PIX pixFHMTGen_1( PIX pixd, PIX pixs, String selname );
public static native @Cast("l_int32") int fhmtgen_low_1( @Cast("l_uint32*") IntPointer datad, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntPointer datas, @Cast("l_int32") int wpls, @Cast("l_int32") int index );
public static native @Cast("l_int32") int fhmtgen_low_1( @Cast("l_uint32*") IntBuffer datad, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntBuffer datas, @Cast("l_int32") int wpls, @Cast("l_int32") int index );
public static native @Cast("l_int32") int fhmtgen_low_1( @Cast("l_uint32*") int[] datad, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpld, @Cast("l_uint32*") int[] datas, @Cast("l_int32") int wpls, @Cast("l_int32") int index );
public static native @Cast("l_int32") int pixItalicWords( PIX pixs, BOXA boxaw, PIX pixw, @Cast("BOXA**") PointerPointer pboxa, @Cast("l_int32") int debugflag );
public static native @Cast("l_int32") int pixItalicWords( PIX pixs, BOXA boxaw, PIX pixw, @ByPtrPtr BOXA pboxa, @Cast("l_int32") int debugflag );
public static native @Cast("l_int32") int pixOrientDetect( PIX pixs, @Cast("l_float32*") FloatPointer pupconf, @Cast("l_float32*") FloatPointer pleftconf, @Cast("l_int32") int mincount, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int pixOrientDetect( PIX pixs, @Cast("l_float32*") FloatBuffer pupconf, @Cast("l_float32*") FloatBuffer pleftconf, @Cast("l_int32") int mincount, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int pixOrientDetect( PIX pixs, @Cast("l_float32*") float[] pupconf, @Cast("l_float32*") float[] pleftconf, @Cast("l_int32") int mincount, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int makeOrientDecision( @Cast("l_float32") float upconf, @Cast("l_float32") float leftconf, @Cast("l_float32") float minupconf, @Cast("l_float32") float minratio, @Cast("l_int32*") IntPointer porient, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int makeOrientDecision( @Cast("l_float32") float upconf, @Cast("l_float32") float leftconf, @Cast("l_float32") float minupconf, @Cast("l_float32") float minratio, @Cast("l_int32*") IntBuffer porient, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int makeOrientDecision( @Cast("l_float32") float upconf, @Cast("l_float32") float leftconf, @Cast("l_float32") float minupconf, @Cast("l_float32") float minratio, @Cast("l_int32*") int[] porient, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int pixUpDownDetect( PIX pixs, @Cast("l_float32*") FloatPointer pconf, @Cast("l_int32") int mincount, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int pixUpDownDetect( PIX pixs, @Cast("l_float32*") FloatBuffer pconf, @Cast("l_int32") int mincount, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int pixUpDownDetect( PIX pixs, @Cast("l_float32*") float[] pconf, @Cast("l_int32") int mincount, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int pixUpDownDetectGeneral( PIX pixs, @Cast("l_float32*") FloatPointer pconf, @Cast("l_int32") int mincount, @Cast("l_int32") int npixels, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int pixUpDownDetectGeneral( PIX pixs, @Cast("l_float32*") FloatBuffer pconf, @Cast("l_int32") int mincount, @Cast("l_int32") int npixels, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int pixUpDownDetectGeneral( PIX pixs, @Cast("l_float32*") float[] pconf, @Cast("l_int32") int mincount, @Cast("l_int32") int npixels, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int pixOrientDetectDwa( PIX pixs, @Cast("l_float32*") FloatPointer pupconf, @Cast("l_float32*") FloatPointer pleftconf, @Cast("l_int32") int mincount, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int pixOrientDetectDwa( PIX pixs, @Cast("l_float32*") FloatBuffer pupconf, @Cast("l_float32*") FloatBuffer pleftconf, @Cast("l_int32") int mincount, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int pixOrientDetectDwa( PIX pixs, @Cast("l_float32*") float[] pupconf, @Cast("l_float32*") float[] pleftconf, @Cast("l_int32") int mincount, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int pixUpDownDetectDwa( PIX pixs, @Cast("l_float32*") FloatPointer pconf, @Cast("l_int32") int mincount, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int pixUpDownDetectDwa( PIX pixs, @Cast("l_float32*") FloatBuffer pconf, @Cast("l_int32") int mincount, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int pixUpDownDetectDwa( PIX pixs, @Cast("l_float32*") float[] pconf, @Cast("l_int32") int mincount, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int pixUpDownDetectGeneralDwa( PIX pixs, @Cast("l_float32*") FloatPointer pconf, @Cast("l_int32") int mincount, @Cast("l_int32") int npixels, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int pixUpDownDetectGeneralDwa( PIX pixs, @Cast("l_float32*") FloatBuffer pconf, @Cast("l_int32") int mincount, @Cast("l_int32") int npixels, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int pixUpDownDetectGeneralDwa( PIX pixs, @Cast("l_float32*") float[] pconf, @Cast("l_int32") int mincount, @Cast("l_int32") int npixels, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int pixMirrorDetect( PIX pixs, @Cast("l_float32*") FloatPointer pconf, @Cast("l_int32") int mincount, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int pixMirrorDetect( PIX pixs, @Cast("l_float32*") FloatBuffer pconf, @Cast("l_int32") int mincount, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int pixMirrorDetect( PIX pixs, @Cast("l_float32*") float[] pconf, @Cast("l_int32") int mincount, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int pixMirrorDetectDwa( PIX pixs, @Cast("l_float32*") FloatPointer pconf, @Cast("l_int32") int mincount, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int pixMirrorDetectDwa( PIX pixs, @Cast("l_float32*") FloatBuffer pconf, @Cast("l_int32") int mincount, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int pixMirrorDetectDwa( PIX pixs, @Cast("l_float32*") float[] pconf, @Cast("l_int32") int mincount, @Cast("l_int32") int debug );
public static native PIX pixFlipFHMTGen( PIX pixd, PIX pixs, @Cast("char*") BytePointer selname );
public static native PIX pixFlipFHMTGen( PIX pixd, PIX pixs, @Cast("char*") ByteBuffer selname );
public static native PIX pixFlipFHMTGen( PIX pixd, PIX pixs, @Cast("char*") byte[] selname );
public static native @Cast("l_int32") int fmorphautogen( SELA sela, @Cast("l_int32") int fileindex, @Cast("const char*") BytePointer filename );
public static native @Cast("l_int32") int fmorphautogen( SELA sela, @Cast("l_int32") int fileindex, String filename );
public static native @Cast("l_int32") int fmorphautogen1( SELA sela, @Cast("l_int32") int fileindex, @Cast("const char*") BytePointer filename );
public static native @Cast("l_int32") int fmorphautogen1( SELA sela, @Cast("l_int32") int fileindex, String filename );
public static native @Cast("l_int32") int fmorphautogen2( SELA sela, @Cast("l_int32") int fileindex, @Cast("const char*") BytePointer filename );
public static native @Cast("l_int32") int fmorphautogen2( SELA sela, @Cast("l_int32") int fileindex, String filename );
public static native PIX pixMorphDwa_1( PIX pixd, PIX pixs, @Cast("l_int32") int operation, @Cast("char*") BytePointer selname );
public static native PIX pixMorphDwa_1( PIX pixd, PIX pixs, @Cast("l_int32") int operation, @Cast("char*") ByteBuffer selname );
public static native PIX pixMorphDwa_1( PIX pixd, PIX pixs, @Cast("l_int32") int operation, @Cast("char*") byte[] selname );
public static native PIX pixFMorphopGen_1( PIX pixd, PIX pixs, @Cast("l_int32") int operation, @Cast("char*") BytePointer selname );
public static native PIX pixFMorphopGen_1( PIX pixd, PIX pixs, @Cast("l_int32") int operation, @Cast("char*") ByteBuffer selname );
public static native PIX pixFMorphopGen_1( PIX pixd, PIX pixs, @Cast("l_int32") int operation, @Cast("char*") byte[] selname );
public static native @Cast("l_int32") int fmorphopgen_low_1( @Cast("l_uint32*") IntPointer datad, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntPointer datas, @Cast("l_int32") int wpls, @Cast("l_int32") int index );
public static native @Cast("l_int32") int fmorphopgen_low_1( @Cast("l_uint32*") IntBuffer datad, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntBuffer datas, @Cast("l_int32") int wpls, @Cast("l_int32") int index );
public static native @Cast("l_int32") int fmorphopgen_low_1( @Cast("l_uint32*") int[] datad, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpld, @Cast("l_uint32*") int[] datas, @Cast("l_int32") int wpls, @Cast("l_int32") int index );
public static native FPIX fpixCreate( @Cast("l_int32") int width, @Cast("l_int32") int height );
public static native FPIX fpixCreateTemplate( FPIX fpixs );
public static native FPIX fpixClone( FPIX fpix );
public static native FPIX fpixCopy( FPIX fpixd, FPIX fpixs );
public static native @Cast("l_int32") int fpixResizeImageData( FPIX fpixd, FPIX fpixs );
public static native void fpixDestroy( @Cast("FPIX**") PointerPointer pfpix );
public static native void fpixDestroy( @ByPtrPtr FPIX pfpix );
public static native @Cast("l_int32") int fpixGetDimensions( FPIX fpix, @Cast("l_int32*") IntPointer pw, @Cast("l_int32*") IntPointer ph );
public static native @Cast("l_int32") int fpixGetDimensions( FPIX fpix, @Cast("l_int32*") IntBuffer pw, @Cast("l_int32*") IntBuffer ph );
public static native @Cast("l_int32") int fpixGetDimensions( FPIX fpix, @Cast("l_int32*") int[] pw, @Cast("l_int32*") int[] ph );
public static native @Cast("l_int32") int fpixSetDimensions( FPIX fpix, @Cast("l_int32") int w, @Cast("l_int32") int h );
public static native @Cast("l_int32") int fpixGetWpl( FPIX fpix );
public static native @Cast("l_int32") int fpixSetWpl( FPIX fpix, @Cast("l_int32") int wpl );
public static native @Cast("l_int32") int fpixGetRefcount( FPIX fpix );
public static native @Cast("l_int32") int fpixChangeRefcount( FPIX fpix, @Cast("l_int32") int delta );
public static native @Cast("l_int32") int fpixGetResolution( FPIX fpix, @Cast("l_int32*") IntPointer pxres, @Cast("l_int32*") IntPointer pyres );
public static native @Cast("l_int32") int fpixGetResolution( FPIX fpix, @Cast("l_int32*") IntBuffer pxres, @Cast("l_int32*") IntBuffer pyres );
public static native @Cast("l_int32") int fpixGetResolution( FPIX fpix, @Cast("l_int32*") int[] pxres, @Cast("l_int32*") int[] pyres );
public static native @Cast("l_int32") int fpixSetResolution( FPIX fpix, @Cast("l_int32") int xres, @Cast("l_int32") int yres );
public static native @Cast("l_int32") int fpixCopyResolution( FPIX fpixd, FPIX fpixs );
public static native @Cast("l_float32*") FloatPointer fpixGetData( FPIX fpix );
public static native @Cast("l_int32") int fpixSetData( FPIX fpix, @Cast("l_float32*") FloatPointer data );
public static native @Cast("l_int32") int fpixSetData( FPIX fpix, @Cast("l_float32*") FloatBuffer data );
public static native @Cast("l_int32") int fpixSetData( FPIX fpix, @Cast("l_float32*") float[] data );
public static native @Cast("l_int32") int fpixGetPixel( FPIX fpix, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_float32*") FloatPointer pval );
public static native @Cast("l_int32") int fpixGetPixel( FPIX fpix, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_float32*") FloatBuffer pval );
public static native @Cast("l_int32") int fpixGetPixel( FPIX fpix, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_float32*") float[] pval );
public static native @Cast("l_int32") int fpixSetPixel( FPIX fpix, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_float32") float val );
public static native FPIXA fpixaCreate( @Cast("l_int32") int n );
public static native FPIXA fpixaCopy( FPIXA fpixa, @Cast("l_int32") int copyflag );
public static native void fpixaDestroy( @Cast("FPIXA**") PointerPointer pfpixa );
public static native void fpixaDestroy( @ByPtrPtr FPIXA pfpixa );
public static native @Cast("l_int32") int fpixaAddFPix( FPIXA fpixa, FPIX fpix, @Cast("l_int32") int copyflag );
public static native @Cast("l_int32") int fpixaGetCount( FPIXA fpixa );
public static native @Cast("l_int32") int fpixaChangeRefcount( FPIXA fpixa, @Cast("l_int32") int delta );
public static native FPIX fpixaGetFPix( FPIXA fpixa, @Cast("l_int32") int index, @Cast("l_int32") int accesstype );
public static native @Cast("l_int32") int fpixaGetFPixDimensions( FPIXA fpixa, @Cast("l_int32") int index, @Cast("l_int32*") IntPointer pw, @Cast("l_int32*") IntPointer ph );
public static native @Cast("l_int32") int fpixaGetFPixDimensions( FPIXA fpixa, @Cast("l_int32") int index, @Cast("l_int32*") IntBuffer pw, @Cast("l_int32*") IntBuffer ph );
public static native @Cast("l_int32") int fpixaGetFPixDimensions( FPIXA fpixa, @Cast("l_int32") int index, @Cast("l_int32*") int[] pw, @Cast("l_int32*") int[] ph );
public static native @Cast("l_float32*") FloatPointer fpixaGetData( FPIXA fpixa, @Cast("l_int32") int index );
public static native @Cast("l_int32") int fpixaGetPixel( FPIXA fpixa, @Cast("l_int32") int index, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_float32*") FloatPointer pval );
public static native @Cast("l_int32") int fpixaGetPixel( FPIXA fpixa, @Cast("l_int32") int index, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_float32*") FloatBuffer pval );
public static native @Cast("l_int32") int fpixaGetPixel( FPIXA fpixa, @Cast("l_int32") int index, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_float32*") float[] pval );
public static native @Cast("l_int32") int fpixaSetPixel( FPIXA fpixa, @Cast("l_int32") int index, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_float32") float val );
public static native DPIX dpixCreate( @Cast("l_int32") int width, @Cast("l_int32") int height );
public static native DPIX dpixCreateTemplate( DPIX dpixs );
public static native DPIX dpixClone( DPIX dpix );
public static native DPIX dpixCopy( DPIX dpixd, DPIX dpixs );
public static native @Cast("l_int32") int dpixResizeImageData( DPIX dpixd, DPIX dpixs );
public static native void dpixDestroy( @Cast("DPIX**") PointerPointer pdpix );
public static native void dpixDestroy( @ByPtrPtr DPIX pdpix );
public static native @Cast("l_int32") int dpixGetDimensions( DPIX dpix, @Cast("l_int32*") IntPointer pw, @Cast("l_int32*") IntPointer ph );
public static native @Cast("l_int32") int dpixGetDimensions( DPIX dpix, @Cast("l_int32*") IntBuffer pw, @Cast("l_int32*") IntBuffer ph );
public static native @Cast("l_int32") int dpixGetDimensions( DPIX dpix, @Cast("l_int32*") int[] pw, @Cast("l_int32*") int[] ph );
public static native @Cast("l_int32") int dpixSetDimensions( DPIX dpix, @Cast("l_int32") int w, @Cast("l_int32") int h );
public static native @Cast("l_int32") int dpixGetWpl( DPIX dpix );
public static native @Cast("l_int32") int dpixSetWpl( DPIX dpix, @Cast("l_int32") int wpl );
public static native @Cast("l_int32") int dpixGetRefcount( DPIX dpix );
public static native @Cast("l_int32") int dpixChangeRefcount( DPIX dpix, @Cast("l_int32") int delta );
public static native @Cast("l_int32") int dpixGetResolution( DPIX dpix, @Cast("l_int32*") IntPointer pxres, @Cast("l_int32*") IntPointer pyres );
public static native @Cast("l_int32") int dpixGetResolution( DPIX dpix, @Cast("l_int32*") IntBuffer pxres, @Cast("l_int32*") IntBuffer pyres );
public static native @Cast("l_int32") int dpixGetResolution( DPIX dpix, @Cast("l_int32*") int[] pxres, @Cast("l_int32*") int[] pyres );
public static native @Cast("l_int32") int dpixSetResolution( DPIX dpix, @Cast("l_int32") int xres, @Cast("l_int32") int yres );
public static native @Cast("l_int32") int dpixCopyResolution( DPIX dpixd, DPIX dpixs );
public static native @Cast("l_float64*") DoublePointer dpixGetData( DPIX dpix );
public static native @Cast("l_int32") int dpixSetData( DPIX dpix, @Cast("l_float64*") DoublePointer data );
public static native @Cast("l_int32") int dpixSetData( DPIX dpix, @Cast("l_float64*") DoubleBuffer data );
public static native @Cast("l_int32") int dpixSetData( DPIX dpix, @Cast("l_float64*") double[] data );
public static native @Cast("l_int32") int dpixGetPixel( DPIX dpix, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_float64*") DoublePointer pval );
public static native @Cast("l_int32") int dpixGetPixel( DPIX dpix, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_float64*") DoubleBuffer pval );
public static native @Cast("l_int32") int dpixGetPixel( DPIX dpix, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_float64*") double[] pval );
public static native @Cast("l_int32") int dpixSetPixel( DPIX dpix, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_float64") double val );
public static native FPIX fpixRead( @Cast("const char*") BytePointer filename );
public static native FPIX fpixRead( String filename );
public static native FPIX fpixReadStream( @Cast("FILE*") Pointer fp );
public static native @Cast("l_int32") int fpixWrite( @Cast("const char*") BytePointer filename, FPIX fpix );
public static native @Cast("l_int32") int fpixWrite( String filename, FPIX fpix );
public static native @Cast("l_int32") int fpixWriteStream( @Cast("FILE*") Pointer fp, FPIX fpix );
public static native FPIX fpixEndianByteSwap( FPIX fpixd, FPIX fpixs );
public static native DPIX dpixRead( @Cast("const char*") BytePointer filename );
public static native DPIX dpixRead( String filename );
public static native DPIX dpixReadStream( @Cast("FILE*") Pointer fp );
public static native @Cast("l_int32") int dpixWrite( @Cast("const char*") BytePointer filename, DPIX dpix );
public static native @Cast("l_int32") int dpixWrite( String filename, DPIX dpix );
public static native @Cast("l_int32") int dpixWriteStream( @Cast("FILE*") Pointer fp, DPIX dpix );
public static native DPIX dpixEndianByteSwap( DPIX dpixd, DPIX dpixs );
public static native @Cast("l_int32") int fpixPrintStream( @Cast("FILE*") Pointer fp, FPIX fpix, @Cast("l_int32") int factor );
public static native FPIX pixConvertToFPix( PIX pixs, @Cast("l_int32") int ncomps );
public static native DPIX pixConvertToDPix( PIX pixs, @Cast("l_int32") int ncomps );
public static native PIX fpixConvertToPix( FPIX fpixs, @Cast("l_int32") int outdepth, @Cast("l_int32") int negvals, @Cast("l_int32") int errorflag );
public static native PIX fpixDisplayMaxDynamicRange( FPIX fpixs );
public static native DPIX fpixConvertToDPix( FPIX fpix );
public static native PIX dpixConvertToPix( DPIX dpixs, @Cast("l_int32") int outdepth, @Cast("l_int32") int negvals, @Cast("l_int32") int errorflag );
public static native FPIX dpixConvertToFPix( DPIX dpix );
public static native @Cast("l_int32") int fpixGetMin( FPIX fpix, @Cast("l_float32*") FloatPointer pminval, @Cast("l_int32*") IntPointer pxminloc, @Cast("l_int32*") IntPointer pyminloc );
public static native @Cast("l_int32") int fpixGetMin( FPIX fpix, @Cast("l_float32*") FloatBuffer pminval, @Cast("l_int32*") IntBuffer pxminloc, @Cast("l_int32*") IntBuffer pyminloc );
public static native @Cast("l_int32") int fpixGetMin( FPIX fpix, @Cast("l_float32*") float[] pminval, @Cast("l_int32*") int[] pxminloc, @Cast("l_int32*") int[] pyminloc );
public static native @Cast("l_int32") int fpixGetMax( FPIX fpix, @Cast("l_float32*") FloatPointer pmaxval, @Cast("l_int32*") IntPointer pxmaxloc, @Cast("l_int32*") IntPointer pymaxloc );
public static native @Cast("l_int32") int fpixGetMax( FPIX fpix, @Cast("l_float32*") FloatBuffer pmaxval, @Cast("l_int32*") IntBuffer pxmaxloc, @Cast("l_int32*") IntBuffer pymaxloc );
public static native @Cast("l_int32") int fpixGetMax( FPIX fpix, @Cast("l_float32*") float[] pmaxval, @Cast("l_int32*") int[] pxmaxloc, @Cast("l_int32*") int[] pymaxloc );
public static native @Cast("l_int32") int dpixGetMin( DPIX dpix, @Cast("l_float64*") DoublePointer pminval, @Cast("l_int32*") IntPointer pxminloc, @Cast("l_int32*") IntPointer pyminloc );
public static native @Cast("l_int32") int dpixGetMin( DPIX dpix, @Cast("l_float64*") DoubleBuffer pminval, @Cast("l_int32*") IntBuffer pxminloc, @Cast("l_int32*") IntBuffer pyminloc );
public static native @Cast("l_int32") int dpixGetMin( DPIX dpix, @Cast("l_float64*") double[] pminval, @Cast("l_int32*") int[] pxminloc, @Cast("l_int32*") int[] pyminloc );
public static native @Cast("l_int32") int dpixGetMax( DPIX dpix, @Cast("l_float64*") DoublePointer pmaxval, @Cast("l_int32*") IntPointer pxmaxloc, @Cast("l_int32*") IntPointer pymaxloc );
public static native @Cast("l_int32") int dpixGetMax( DPIX dpix, @Cast("l_float64*") DoubleBuffer pmaxval, @Cast("l_int32*") IntBuffer pxmaxloc, @Cast("l_int32*") IntBuffer pymaxloc );
public static native @Cast("l_int32") int dpixGetMax( DPIX dpix, @Cast("l_float64*") double[] pmaxval, @Cast("l_int32*") int[] pxmaxloc, @Cast("l_int32*") int[] pymaxloc );
public static native FPIX fpixScaleByInteger( FPIX fpixs, @Cast("l_int32") int factor );
public static native DPIX dpixScaleByInteger( DPIX dpixs, @Cast("l_int32") int factor );
public static native FPIX fpixLinearCombination( FPIX fpixd, FPIX fpixs1, FPIX fpixs2, @Cast("l_float32") float a, @Cast("l_float32") float b );
public static native @Cast("l_int32") int fpixAddMultConstant( FPIX fpix, @Cast("l_float32") float addc, @Cast("l_float32") float multc );
public static native DPIX dpixLinearCombination( DPIX dpixd, DPIX dpixs1, DPIX dpixs2, @Cast("l_float32") float a, @Cast("l_float32") float b );
public static native @Cast("l_int32") int dpixAddMultConstant( DPIX dpix, @Cast("l_float64") double addc, @Cast("l_float64") double multc );
public static native @Cast("l_int32") int fpixSetAllArbitrary( FPIX fpix, @Cast("l_float32") float inval );
public static native @Cast("l_int32") int dpixSetAllArbitrary( DPIX dpix, @Cast("l_float64") double inval );
public static native FPIX fpixAddBorder( FPIX fpixs, @Cast("l_int32") int left, @Cast("l_int32") int right, @Cast("l_int32") int top, @Cast("l_int32") int bot );
public static native FPIX fpixRemoveBorder( FPIX fpixs, @Cast("l_int32") int left, @Cast("l_int32") int right, @Cast("l_int32") int top, @Cast("l_int32") int bot );
public static native FPIX fpixAddMirroredBorder( FPIX fpixs, @Cast("l_int32") int left, @Cast("l_int32") int right, @Cast("l_int32") int top, @Cast("l_int32") int bot );
public static native FPIX fpixAddContinuedBorder( FPIX fpixs, @Cast("l_int32") int left, @Cast("l_int32") int right, @Cast("l_int32") int top, @Cast("l_int32") int bot );
public static native FPIX fpixAddSlopeBorder( FPIX fpixs, @Cast("l_int32") int left, @Cast("l_int32") int right, @Cast("l_int32") int top, @Cast("l_int32") int bot );
public static native @Cast("l_int32") int fpixRasterop( FPIX fpixd, @Cast("l_int32") int dx, @Cast("l_int32") int dy, @Cast("l_int32") int dw, @Cast("l_int32") int dh, FPIX fpixs, @Cast("l_int32") int sx, @Cast("l_int32") int sy );
public static native FPIX fpixRotateOrth( FPIX fpixs, @Cast("l_int32") int quads );
public static native FPIX fpixRotate180( FPIX fpixd, FPIX fpixs );
public static native FPIX fpixRotate90( FPIX fpixs, @Cast("l_int32") int direction );
public static native FPIX fpixFlipLR( FPIX fpixd, FPIX fpixs );
public static native FPIX fpixFlipTB( FPIX fpixd, FPIX fpixs );
public static native FPIX fpixAffinePta( FPIX fpixs, PTA ptad, PTA ptas, @Cast("l_int32") int border, @Cast("l_float32") float inval );
public static native FPIX fpixAffine( FPIX fpixs, @Cast("l_float32*") FloatPointer vc, @Cast("l_float32") float inval );
public static native FPIX fpixAffine( FPIX fpixs, @Cast("l_float32*") FloatBuffer vc, @Cast("l_float32") float inval );
public static native FPIX fpixAffine( FPIX fpixs, @Cast("l_float32*") float[] vc, @Cast("l_float32") float inval );
public static native FPIX fpixProjectivePta( FPIX fpixs, PTA ptad, PTA ptas, @Cast("l_int32") int border, @Cast("l_float32") float inval );
public static native FPIX fpixProjective( FPIX fpixs, @Cast("l_float32*") FloatPointer vc, @Cast("l_float32") float inval );
public static native FPIX fpixProjective( FPIX fpixs, @Cast("l_float32*") FloatBuffer vc, @Cast("l_float32") float inval );
public static native FPIX fpixProjective( FPIX fpixs, @Cast("l_float32*") float[] vc, @Cast("l_float32") float inval );
public static native @Cast("l_int32") int linearInterpolatePixelFloat( @Cast("l_float32*") FloatPointer datas, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_float32") float x, @Cast("l_float32") float y, @Cast("l_float32") float inval, @Cast("l_float32*") FloatPointer pval );
public static native @Cast("l_int32") int linearInterpolatePixelFloat( @Cast("l_float32*") FloatBuffer datas, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_float32") float x, @Cast("l_float32") float y, @Cast("l_float32") float inval, @Cast("l_float32*") FloatBuffer pval );
public static native @Cast("l_int32") int linearInterpolatePixelFloat( @Cast("l_float32*") float[] datas, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_float32") float x, @Cast("l_float32") float y, @Cast("l_float32") float inval, @Cast("l_float32*") float[] pval );
public static native PIX fpixThresholdToPix( FPIX fpix, @Cast("l_float32") float thresh );
public static native FPIX pixComponentFunction( PIX pix, @Cast("l_float32") float rnum, @Cast("l_float32") float gnum, @Cast("l_float32") float bnum, @Cast("l_float32") float rdenom, @Cast("l_float32") float gdenom, @Cast("l_float32") float bdenom );
public static native PIX pixReadStreamGif( @Cast("FILE*") Pointer fp );
public static native @Cast("l_int32") int pixWriteStreamGif( @Cast("FILE*") Pointer fp, PIX pix );
public static native PIX pixReadMemGif( @Cast("const l_uint8*") BytePointer cdata, @Cast("size_t") long size );
public static native PIX pixReadMemGif( @Cast("const l_uint8*") ByteBuffer cdata, @Cast("size_t") long size );
public static native PIX pixReadMemGif( @Cast("const l_uint8*") byte[] cdata, @Cast("size_t") long size );
public static native @Cast("l_int32") int pixWriteMemGif( @Cast("l_uint8**") PointerPointer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix );
public static native @Cast("l_int32") int pixWriteMemGif( @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix );
public static native @Cast("l_int32") int pixWriteMemGif( @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix );
public static native @Cast("l_int32") int pixWriteMemGif( @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer psize, PIX pix );
public static native GPLOT gplotCreate( @Cast("const char*") BytePointer rootname, @Cast("l_int32") int outformat, @Cast("const char*") BytePointer title, @Cast("const char*") BytePointer xlabel, @Cast("const char*") BytePointer ylabel );
public static native GPLOT gplotCreate( String rootname, @Cast("l_int32") int outformat, String title, String xlabel, String ylabel );
public static native void gplotDestroy( @Cast("GPLOT**") PointerPointer pgplot );
public static native void gplotDestroy( @ByPtrPtr GPLOT pgplot );
public static native @Cast("l_int32") int gplotAddPlot( GPLOT gplot, NUMA nax, NUMA nay, @Cast("l_int32") int plotstyle, @Cast("const char*") BytePointer plottitle );
public static native @Cast("l_int32") int gplotAddPlot( GPLOT gplot, NUMA nax, NUMA nay, @Cast("l_int32") int plotstyle, String plottitle );
public static native @Cast("l_int32") int gplotSetScaling( GPLOT gplot, @Cast("l_int32") int scaling );
public static native @Cast("l_int32") int gplotMakeOutput( GPLOT gplot );
public static native @Cast("l_int32") int gplotGenCommandFile( GPLOT gplot );
public static native @Cast("l_int32") int gplotGenDataFiles( GPLOT gplot );
public static native @Cast("l_int32") int gplotSimple1( NUMA na, @Cast("l_int32") int outformat, @Cast("const char*") BytePointer outroot, @Cast("const char*") BytePointer title );
public static native @Cast("l_int32") int gplotSimple1( NUMA na, @Cast("l_int32") int outformat, String outroot, String title );
public static native @Cast("l_int32") int gplotSimple2( NUMA na1, NUMA na2, @Cast("l_int32") int outformat, @Cast("const char*") BytePointer outroot, @Cast("const char*") BytePointer title );
public static native @Cast("l_int32") int gplotSimple2( NUMA na1, NUMA na2, @Cast("l_int32") int outformat, String outroot, String title );
public static native @Cast("l_int32") int gplotSimpleN( NUMAA naa, @Cast("l_int32") int outformat, @Cast("const char*") BytePointer outroot, @Cast("const char*") BytePointer title );
public static native @Cast("l_int32") int gplotSimpleN( NUMAA naa, @Cast("l_int32") int outformat, String outroot, String title );
public static native @Cast("l_int32") int gplotSimpleXY1( NUMA nax, NUMA nay, @Cast("l_int32") int outformat, @Cast("const char*") BytePointer outroot, @Cast("const char*") BytePointer title );
public static native @Cast("l_int32") int gplotSimpleXY1( NUMA nax, NUMA nay, @Cast("l_int32") int outformat, String outroot, String title );
public static native @Cast("l_int32") int gplotSimpleXY2( NUMA nax, NUMA nay1, NUMA nay2, @Cast("l_int32") int outformat, @Cast("const char*") BytePointer outroot, @Cast("const char*") BytePointer title );
public static native @Cast("l_int32") int gplotSimpleXY2( NUMA nax, NUMA nay1, NUMA nay2, @Cast("l_int32") int outformat, String outroot, String title );
public static native @Cast("l_int32") int gplotSimpleXYN( NUMA nax, NUMAA naay, @Cast("l_int32") int outformat, @Cast("const char*") BytePointer outroot, @Cast("const char*") BytePointer title );
public static native @Cast("l_int32") int gplotSimpleXYN( NUMA nax, NUMAA naay, @Cast("l_int32") int outformat, String outroot, String title );
public static native GPLOT gplotRead( @Cast("const char*") BytePointer filename );
public static native GPLOT gplotRead( String filename );
public static native @Cast("l_int32") int gplotWrite( @Cast("const char*") BytePointer filename, GPLOT gplot );
public static native @Cast("l_int32") int gplotWrite( String filename, GPLOT gplot );
public static native PTA generatePtaLine( @Cast("l_int32") int x1, @Cast("l_int32") int y1, @Cast("l_int32") int x2, @Cast("l_int32") int y2 );
public static native PTA generatePtaWideLine( @Cast("l_int32") int x1, @Cast("l_int32") int y1, @Cast("l_int32") int x2, @Cast("l_int32") int y2, @Cast("l_int32") int width );
public static native PTA generatePtaBox( BOX box, @Cast("l_int32") int width );
public static native PTA generatePtaBoxa( BOXA boxa, @Cast("l_int32") int width, @Cast("l_int32") int removedups );
public static native PTA generatePtaHashBox( BOX box, @Cast("l_int32") int spacing, @Cast("l_int32") int width, @Cast("l_int32") int orient, @Cast("l_int32") int outline );
public static native PTA generatePtaHashBoxa( BOXA boxa, @Cast("l_int32") int spacing, @Cast("l_int32") int width, @Cast("l_int32") int orient, @Cast("l_int32") int outline, @Cast("l_int32") int removedups );
public static native PTAA generatePtaaBoxa( BOXA boxa );
public static native PTAA generatePtaaHashBoxa( BOXA boxa, @Cast("l_int32") int spacing, @Cast("l_int32") int width, @Cast("l_int32") int orient, @Cast("l_int32") int outline );
public static native PTA generatePtaPolyline( PTA ptas, @Cast("l_int32") int width, @Cast("l_int32") int closeflag, @Cast("l_int32") int removedups );
public static native PTA convertPtaLineTo4cc( PTA ptas );
public static native PTA generatePtaFilledCircle( @Cast("l_int32") int radius );
public static native PTA generatePtaFilledSquare( @Cast("l_int32") int side );
public static native PTA generatePtaLineFromPt( @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_float64") double length, @Cast("l_float64") double radang );
public static native @Cast("l_int32") int locatePtRadially( @Cast("l_int32") int xr, @Cast("l_int32") int yr, @Cast("l_float64") double dist, @Cast("l_float64") double radang, @Cast("l_float64*") DoublePointer px, @Cast("l_float64*") DoublePointer py );
public static native @Cast("l_int32") int locatePtRadially( @Cast("l_int32") int xr, @Cast("l_int32") int yr, @Cast("l_float64") double dist, @Cast("l_float64") double radang, @Cast("l_float64*") DoubleBuffer px, @Cast("l_float64*") DoubleBuffer py );
public static native @Cast("l_int32") int locatePtRadially( @Cast("l_int32") int xr, @Cast("l_int32") int yr, @Cast("l_float64") double dist, @Cast("l_float64") double radang, @Cast("l_float64*") double[] px, @Cast("l_float64*") double[] py );
public static native @Cast("l_int32") int pixRenderPlotFromNuma( @Cast("PIX**") PointerPointer ppix, NUMA na, @Cast("l_int32") int plotloc, @Cast("l_int32") int linewidth, @Cast("l_int32") int max, @Cast("l_uint32") int color );
public static native @Cast("l_int32") int pixRenderPlotFromNuma( @ByPtrPtr PIX ppix, NUMA na, @Cast("l_int32") int plotloc, @Cast("l_int32") int linewidth, @Cast("l_int32") int max, @Cast("l_uint32") int color );
public static native PTA makePlotPtaFromNuma( NUMA na, @Cast("l_int32") int size, @Cast("l_int32") int plotloc, @Cast("l_int32") int linewidth, @Cast("l_int32") int max );
public static native @Cast("l_int32") int pixRenderPlotFromNumaGen( @Cast("PIX**") PointerPointer ppix, NUMA na, @Cast("l_int32") int orient, @Cast("l_int32") int linewidth, @Cast("l_int32") int refpos, @Cast("l_int32") int max, @Cast("l_int32") int drawref, @Cast("l_uint32") int color );
public static native @Cast("l_int32") int pixRenderPlotFromNumaGen( @ByPtrPtr PIX ppix, NUMA na, @Cast("l_int32") int orient, @Cast("l_int32") int linewidth, @Cast("l_int32") int refpos, @Cast("l_int32") int max, @Cast("l_int32") int drawref, @Cast("l_uint32") int color );
public static native PTA makePlotPtaFromNumaGen( NUMA na, @Cast("l_int32") int orient, @Cast("l_int32") int linewidth, @Cast("l_int32") int refpos, @Cast("l_int32") int max, @Cast("l_int32") int drawref );
public static native @Cast("l_int32") int pixRenderPta( PIX pix, PTA pta, @Cast("l_int32") int op );
public static native @Cast("l_int32") int pixRenderPtaArb( PIX pix, PTA pta, @Cast("l_uint8") byte rval, @Cast("l_uint8") byte gval, @Cast("l_uint8") byte bval );
public static native @Cast("l_int32") int pixRenderPtaBlend( PIX pix, PTA pta, @Cast("l_uint8") byte rval, @Cast("l_uint8") byte gval, @Cast("l_uint8") byte bval, @Cast("l_float32") float fract );
public static native @Cast("l_int32") int pixRenderLine( PIX pix, @Cast("l_int32") int x1, @Cast("l_int32") int y1, @Cast("l_int32") int x2, @Cast("l_int32") int y2, @Cast("l_int32") int width, @Cast("l_int32") int op );
public static native @Cast("l_int32") int pixRenderLineArb( PIX pix, @Cast("l_int32") int x1, @Cast("l_int32") int y1, @Cast("l_int32") int x2, @Cast("l_int32") int y2, @Cast("l_int32") int width, @Cast("l_uint8") byte rval, @Cast("l_uint8") byte gval, @Cast("l_uint8") byte bval );
public static native @Cast("l_int32") int pixRenderLineBlend( PIX pix, @Cast("l_int32") int x1, @Cast("l_int32") int y1, @Cast("l_int32") int x2, @Cast("l_int32") int y2, @Cast("l_int32") int width, @Cast("l_uint8") byte rval, @Cast("l_uint8") byte gval, @Cast("l_uint8") byte bval, @Cast("l_float32") float fract );
public static native @Cast("l_int32") int pixRenderBox( PIX pix, BOX box, @Cast("l_int32") int width, @Cast("l_int32") int op );
public static native @Cast("l_int32") int pixRenderBoxArb( PIX pix, BOX box, @Cast("l_int32") int width, @Cast("l_uint8") byte rval, @Cast("l_uint8") byte gval, @Cast("l_uint8") byte bval );
public static native @Cast("l_int32") int pixRenderBoxBlend( PIX pix, BOX box, @Cast("l_int32") int width, @Cast("l_uint8") byte rval, @Cast("l_uint8") byte gval, @Cast("l_uint8") byte bval, @Cast("l_float32") float fract );
public static native @Cast("l_int32") int pixRenderBoxa( PIX pix, BOXA boxa, @Cast("l_int32") int width, @Cast("l_int32") int op );
public static native @Cast("l_int32") int pixRenderBoxaArb( PIX pix, BOXA boxa, @Cast("l_int32") int width, @Cast("l_uint8") byte rval, @Cast("l_uint8") byte gval, @Cast("l_uint8") byte bval );
public static native @Cast("l_int32") int pixRenderBoxaBlend( PIX pix, BOXA boxa, @Cast("l_int32") int width, @Cast("l_uint8") byte rval, @Cast("l_uint8") byte gval, @Cast("l_uint8") byte bval, @Cast("l_float32") float fract, @Cast("l_int32") int removedups );
public static native @Cast("l_int32") int pixRenderHashBox( PIX pix, BOX box, @Cast("l_int32") int spacing, @Cast("l_int32") int width, @Cast("l_int32") int orient, @Cast("l_int32") int outline, @Cast("l_int32") int op );
public static native @Cast("l_int32") int pixRenderHashBoxArb( PIX pix, BOX box, @Cast("l_int32") int spacing, @Cast("l_int32") int width, @Cast("l_int32") int orient, @Cast("l_int32") int outline, @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval );
public static native @Cast("l_int32") int pixRenderHashBoxBlend( PIX pix, BOX box, @Cast("l_int32") int spacing, @Cast("l_int32") int width, @Cast("l_int32") int orient, @Cast("l_int32") int outline, @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_float32") float fract );
public static native @Cast("l_int32") int pixRenderHashBoxa( PIX pix, BOXA boxa, @Cast("l_int32") int spacing, @Cast("l_int32") int width, @Cast("l_int32") int orient, @Cast("l_int32") int outline, @Cast("l_int32") int op );
public static native @Cast("l_int32") int pixRenderHashBoxaArb( PIX pix, BOXA boxa, @Cast("l_int32") int spacing, @Cast("l_int32") int width, @Cast("l_int32") int orient, @Cast("l_int32") int outline, @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval );
public static native @Cast("l_int32") int pixRenderHashBoxaBlend( PIX pix, BOXA boxa, @Cast("l_int32") int spacing, @Cast("l_int32") int width, @Cast("l_int32") int orient, @Cast("l_int32") int outline, @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_float32") float fract );
public static native @Cast("l_int32") int pixRenderPolyline( PIX pix, PTA ptas, @Cast("l_int32") int width, @Cast("l_int32") int op, @Cast("l_int32") int closeflag );
public static native @Cast("l_int32") int pixRenderPolylineArb( PIX pix, PTA ptas, @Cast("l_int32") int width, @Cast("l_uint8") byte rval, @Cast("l_uint8") byte gval, @Cast("l_uint8") byte bval, @Cast("l_int32") int closeflag );
public static native @Cast("l_int32") int pixRenderPolylineBlend( PIX pix, PTA ptas, @Cast("l_int32") int width, @Cast("l_uint8") byte rval, @Cast("l_uint8") byte gval, @Cast("l_uint8") byte bval, @Cast("l_float32") float fract, @Cast("l_int32") int closeflag, @Cast("l_int32") int removedups );
public static native PIX pixRenderRandomCmapPtaa( PIX pix, PTAA ptaa, @Cast("l_int32") int polyflag, @Cast("l_int32") int width, @Cast("l_int32") int closeflag );
public static native PIX pixRenderPolygon( PTA ptas, @Cast("l_int32") int width, @Cast("l_int32*") IntPointer pxmin, @Cast("l_int32*") IntPointer pymin );
public static native PIX pixRenderPolygon( PTA ptas, @Cast("l_int32") int width, @Cast("l_int32*") IntBuffer pxmin, @Cast("l_int32*") IntBuffer pymin );
public static native PIX pixRenderPolygon( PTA ptas, @Cast("l_int32") int width, @Cast("l_int32*") int[] pxmin, @Cast("l_int32*") int[] pymin );
public static native PIX pixFillPolygon( PIX pixs, PTA pta, @Cast("l_int32") int xmin, @Cast("l_int32") int ymin );
public static native PIX pixRenderContours( PIX pixs, @Cast("l_int32") int startval, @Cast("l_int32") int incr, @Cast("l_int32") int outdepth );
public static native PIX fpixAutoRenderContours( FPIX fpix, @Cast("l_int32") int ncontours );
public static native PIX fpixRenderContours( FPIX fpixs, @Cast("l_float32") float incr, @Cast("l_float32") float proxim );
public static native PIX pixErodeGray( PIX pixs, @Cast("l_int32") int hsize, @Cast("l_int32") int vsize );
public static native PIX pixDilateGray( PIX pixs, @Cast("l_int32") int hsize, @Cast("l_int32") int vsize );
public static native PIX pixOpenGray( PIX pixs, @Cast("l_int32") int hsize, @Cast("l_int32") int vsize );
public static native PIX pixCloseGray( PIX pixs, @Cast("l_int32") int hsize, @Cast("l_int32") int vsize );
public static native PIX pixErodeGray3( PIX pixs, @Cast("l_int32") int hsize, @Cast("l_int32") int vsize );
public static native PIX pixDilateGray3( PIX pixs, @Cast("l_int32") int hsize, @Cast("l_int32") int vsize );
public static native PIX pixOpenGray3( PIX pixs, @Cast("l_int32") int hsize, @Cast("l_int32") int vsize );
public static native PIX pixCloseGray3( PIX pixs, @Cast("l_int32") int hsize, @Cast("l_int32") int vsize );
public static native PIX pixDitherToBinary( PIX pixs );
public static native PIX pixDitherToBinarySpec( PIX pixs, @Cast("l_int32") int lowerclip, @Cast("l_int32") int upperclip );
public static native PIX pixThresholdToBinary( PIX pixs, @Cast("l_int32") int thresh );
public static native PIX pixVarThresholdToBinary( PIX pixs, PIX pixg );
public static native PIX pixAdaptThresholdToBinary( PIX pixs, PIX pixm, @Cast("l_float32") float gamma );
public static native PIX pixAdaptThresholdToBinaryGen( PIX pixs, PIX pixm, @Cast("l_float32") float gamma, @Cast("l_int32") int blackval, @Cast("l_int32") int whiteval, @Cast("l_int32") int thresh );
public static native PIX pixDitherToBinaryLUT( PIX pixs, @Cast("l_int32") int lowerclip, @Cast("l_int32") int upperclip );
public static native PIX pixGenerateMaskByValue( PIX pixs, @Cast("l_int32") int val, @Cast("l_int32") int usecmap );
public static native PIX pixGenerateMaskByBand( PIX pixs, @Cast("l_int32") int lower, @Cast("l_int32") int upper, @Cast("l_int32") int inband, @Cast("l_int32") int usecmap );
public static native PIX pixDitherTo2bpp( PIX pixs, @Cast("l_int32") int cmapflag );
public static native PIX pixDitherTo2bppSpec( PIX pixs, @Cast("l_int32") int lowerclip, @Cast("l_int32") int upperclip, @Cast("l_int32") int cmapflag );
public static native PIX pixThresholdTo2bpp( PIX pixs, @Cast("l_int32") int nlevels, @Cast("l_int32") int cmapflag );
public static native PIX pixThresholdTo4bpp( PIX pixs, @Cast("l_int32") int nlevels, @Cast("l_int32") int cmapflag );
public static native PIX pixThresholdOn8bpp( PIX pixs, @Cast("l_int32") int nlevels, @Cast("l_int32") int cmapflag );
public static native PIX pixThresholdGrayArb( PIX pixs, @Cast("const char*") BytePointer edgevals, @Cast("l_int32") int outdepth, @Cast("l_int32") int use_average, @Cast("l_int32") int setblack, @Cast("l_int32") int setwhite );
public static native PIX pixThresholdGrayArb( PIX pixs, String edgevals, @Cast("l_int32") int outdepth, @Cast("l_int32") int use_average, @Cast("l_int32") int setblack, @Cast("l_int32") int setwhite );
public static native @Cast("l_int32*") IntPointer makeGrayQuantIndexTable( @Cast("l_int32") int nlevels );
public static native @Cast("l_int32*") IntPointer makeGrayQuantTargetTable( @Cast("l_int32") int nlevels, @Cast("l_int32") int depth );
public static native @Cast("l_int32") int makeGrayQuantTableArb( NUMA na, @Cast("l_int32") int outdepth, @Cast("l_int32**") PointerPointer ptab, @Cast("PIXCMAP**") PointerPointer pcmap );
public static native @Cast("l_int32") int makeGrayQuantTableArb( NUMA na, @Cast("l_int32") int outdepth, @Cast("l_int32**") @ByPtrPtr IntPointer ptab, @ByPtrPtr PIXCMAP pcmap );
public static native @Cast("l_int32") int makeGrayQuantTableArb( NUMA na, @Cast("l_int32") int outdepth, @Cast("l_int32**") @ByPtrPtr IntBuffer ptab, @ByPtrPtr PIXCMAP pcmap );
public static native @Cast("l_int32") int makeGrayQuantTableArb( NUMA na, @Cast("l_int32") int outdepth, @Cast("l_int32**") @ByPtrPtr int[] ptab, @ByPtrPtr PIXCMAP pcmap );
public static native @Cast("l_int32") int makeGrayQuantColormapArb( PIX pixs, @Cast("l_int32*") IntPointer tab, @Cast("l_int32") int outdepth, @Cast("PIXCMAP**") PointerPointer pcmap );
public static native @Cast("l_int32") int makeGrayQuantColormapArb( PIX pixs, @Cast("l_int32*") IntPointer tab, @Cast("l_int32") int outdepth, @ByPtrPtr PIXCMAP pcmap );
public static native @Cast("l_int32") int makeGrayQuantColormapArb( PIX pixs, @Cast("l_int32*") IntBuffer tab, @Cast("l_int32") int outdepth, @ByPtrPtr PIXCMAP pcmap );
public static native @Cast("l_int32") int makeGrayQuantColormapArb( PIX pixs, @Cast("l_int32*") int[] tab, @Cast("l_int32") int outdepth, @ByPtrPtr PIXCMAP pcmap );
public static native PIX pixGenerateMaskByBand32( PIX pixs, @Cast("l_uint32") int refval, @Cast("l_int32") int delm, @Cast("l_int32") int delp, @Cast("l_float32") float fractm, @Cast("l_float32") float fractp );
public static native PIX pixGenerateMaskByDiscr32( PIX pixs, @Cast("l_uint32") int refval1, @Cast("l_uint32") int refval2, @Cast("l_int32") int distflag );
public static native PIX pixGrayQuantFromHisto( PIX pixd, PIX pixs, PIX pixm, @Cast("l_float32") float minfract, @Cast("l_int32") int maxsize );
public static native PIX pixGrayQuantFromCmap( PIX pixs, PIXCMAP cmap, @Cast("l_int32") int mindepth );
public static native void ditherToBinaryLow( @Cast("l_uint32*") IntPointer datad, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntPointer datas, @Cast("l_int32") int wpls, @Cast("l_uint32*") IntPointer bufs1, @Cast("l_uint32*") IntPointer bufs2, @Cast("l_int32") int lowerclip, @Cast("l_int32") int upperclip );
public static native void ditherToBinaryLow( @Cast("l_uint32*") IntBuffer datad, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntBuffer datas, @Cast("l_int32") int wpls, @Cast("l_uint32*") IntBuffer bufs1, @Cast("l_uint32*") IntBuffer bufs2, @Cast("l_int32") int lowerclip, @Cast("l_int32") int upperclip );
public static native void ditherToBinaryLow( @Cast("l_uint32*") int[] datad, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpld, @Cast("l_uint32*") int[] datas, @Cast("l_int32") int wpls, @Cast("l_uint32*") int[] bufs1, @Cast("l_uint32*") int[] bufs2, @Cast("l_int32") int lowerclip, @Cast("l_int32") int upperclip );
public static native void ditherToBinaryLineLow( @Cast("l_uint32*") IntPointer lined, @Cast("l_int32") int w, @Cast("l_uint32*") IntPointer bufs1, @Cast("l_uint32*") IntPointer bufs2, @Cast("l_int32") int lowerclip, @Cast("l_int32") int upperclip, @Cast("l_int32") int lastlineflag );
public static native void ditherToBinaryLineLow( @Cast("l_uint32*") IntBuffer lined, @Cast("l_int32") int w, @Cast("l_uint32*") IntBuffer bufs1, @Cast("l_uint32*") IntBuffer bufs2, @Cast("l_int32") int lowerclip, @Cast("l_int32") int upperclip, @Cast("l_int32") int lastlineflag );
public static native void ditherToBinaryLineLow( @Cast("l_uint32*") int[] lined, @Cast("l_int32") int w, @Cast("l_uint32*") int[] bufs1, @Cast("l_uint32*") int[] bufs2, @Cast("l_int32") int lowerclip, @Cast("l_int32") int upperclip, @Cast("l_int32") int lastlineflag );
public static native void thresholdToBinaryLow( @Cast("l_uint32*") IntPointer datad, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntPointer datas, @Cast("l_int32") int d, @Cast("l_int32") int wpls, @Cast("l_int32") int thresh );
public static native void thresholdToBinaryLow( @Cast("l_uint32*") IntBuffer datad, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntBuffer datas, @Cast("l_int32") int d, @Cast("l_int32") int wpls, @Cast("l_int32") int thresh );
public static native void thresholdToBinaryLow( @Cast("l_uint32*") int[] datad, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpld, @Cast("l_uint32*") int[] datas, @Cast("l_int32") int d, @Cast("l_int32") int wpls, @Cast("l_int32") int thresh );
public static native void thresholdToBinaryLineLow( @Cast("l_uint32*") IntPointer lined, @Cast("l_int32") int w, @Cast("l_uint32*") IntPointer lines, @Cast("l_int32") int d, @Cast("l_int32") int thresh );
public static native void thresholdToBinaryLineLow( @Cast("l_uint32*") IntBuffer lined, @Cast("l_int32") int w, @Cast("l_uint32*") IntBuffer lines, @Cast("l_int32") int d, @Cast("l_int32") int thresh );
public static native void thresholdToBinaryLineLow( @Cast("l_uint32*") int[] lined, @Cast("l_int32") int w, @Cast("l_uint32*") int[] lines, @Cast("l_int32") int d, @Cast("l_int32") int thresh );
public static native void ditherToBinaryLUTLow( @Cast("l_uint32*") IntPointer datad, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntPointer datas, @Cast("l_int32") int wpls, @Cast("l_uint32*") IntPointer bufs1, @Cast("l_uint32*") IntPointer bufs2, @Cast("l_int32*") IntPointer tabval, @Cast("l_int32*") IntPointer tab38, @Cast("l_int32*") IntPointer tab14 );
public static native void ditherToBinaryLUTLow( @Cast("l_uint32*") IntBuffer datad, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntBuffer datas, @Cast("l_int32") int wpls, @Cast("l_uint32*") IntBuffer bufs1, @Cast("l_uint32*") IntBuffer bufs2, @Cast("l_int32*") IntBuffer tabval, @Cast("l_int32*") IntBuffer tab38, @Cast("l_int32*") IntBuffer tab14 );
public static native void ditherToBinaryLUTLow( @Cast("l_uint32*") int[] datad, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpld, @Cast("l_uint32*") int[] datas, @Cast("l_int32") int wpls, @Cast("l_uint32*") int[] bufs1, @Cast("l_uint32*") int[] bufs2, @Cast("l_int32*") int[] tabval, @Cast("l_int32*") int[] tab38, @Cast("l_int32*") int[] tab14 );
public static native void ditherToBinaryLineLUTLow( @Cast("l_uint32*") IntPointer lined, @Cast("l_int32") int w, @Cast("l_uint32*") IntPointer bufs1, @Cast("l_uint32*") IntPointer bufs2, @Cast("l_int32*") IntPointer tabval, @Cast("l_int32*") IntPointer tab38, @Cast("l_int32*") IntPointer tab14, @Cast("l_int32") int lastlineflag );
public static native void ditherToBinaryLineLUTLow( @Cast("l_uint32*") IntBuffer lined, @Cast("l_int32") int w, @Cast("l_uint32*") IntBuffer bufs1, @Cast("l_uint32*") IntBuffer bufs2, @Cast("l_int32*") IntBuffer tabval, @Cast("l_int32*") IntBuffer tab38, @Cast("l_int32*") IntBuffer tab14, @Cast("l_int32") int lastlineflag );
public static native void ditherToBinaryLineLUTLow( @Cast("l_uint32*") int[] lined, @Cast("l_int32") int w, @Cast("l_uint32*") int[] bufs1, @Cast("l_uint32*") int[] bufs2, @Cast("l_int32*") int[] tabval, @Cast("l_int32*") int[] tab38, @Cast("l_int32*") int[] tab14, @Cast("l_int32") int lastlineflag );
public static native @Cast("l_int32") int make8To1DitherTables( @Cast("l_int32**") PointerPointer ptabval, @Cast("l_int32**") PointerPointer ptab38, @Cast("l_int32**") PointerPointer ptab14, @Cast("l_int32") int lowerclip, @Cast("l_int32") int upperclip );
public static native @Cast("l_int32") int make8To1DitherTables( @Cast("l_int32**") @ByPtrPtr IntPointer ptabval, @Cast("l_int32**") @ByPtrPtr IntPointer ptab38, @Cast("l_int32**") @ByPtrPtr IntPointer ptab14, @Cast("l_int32") int lowerclip, @Cast("l_int32") int upperclip );
public static native @Cast("l_int32") int make8To1DitherTables( @Cast("l_int32**") @ByPtrPtr IntBuffer ptabval, @Cast("l_int32**") @ByPtrPtr IntBuffer ptab38, @Cast("l_int32**") @ByPtrPtr IntBuffer ptab14, @Cast("l_int32") int lowerclip, @Cast("l_int32") int upperclip );
public static native @Cast("l_int32") int make8To1DitherTables( @Cast("l_int32**") @ByPtrPtr int[] ptabval, @Cast("l_int32**") @ByPtrPtr int[] ptab38, @Cast("l_int32**") @ByPtrPtr int[] ptab14, @Cast("l_int32") int lowerclip, @Cast("l_int32") int upperclip );
public static native void ditherTo2bppLow( @Cast("l_uint32*") IntPointer datad, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntPointer datas, @Cast("l_int32") int wpls, @Cast("l_uint32*") IntPointer bufs1, @Cast("l_uint32*") IntPointer bufs2, @Cast("l_int32*") IntPointer tabval, @Cast("l_int32*") IntPointer tab38, @Cast("l_int32*") IntPointer tab14 );
public static native void ditherTo2bppLow( @Cast("l_uint32*") IntBuffer datad, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntBuffer datas, @Cast("l_int32") int wpls, @Cast("l_uint32*") IntBuffer bufs1, @Cast("l_uint32*") IntBuffer bufs2, @Cast("l_int32*") IntBuffer tabval, @Cast("l_int32*") IntBuffer tab38, @Cast("l_int32*") IntBuffer tab14 );
public static native void ditherTo2bppLow( @Cast("l_uint32*") int[] datad, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpld, @Cast("l_uint32*") int[] datas, @Cast("l_int32") int wpls, @Cast("l_uint32*") int[] bufs1, @Cast("l_uint32*") int[] bufs2, @Cast("l_int32*") int[] tabval, @Cast("l_int32*") int[] tab38, @Cast("l_int32*") int[] tab14 );
public static native void ditherTo2bppLineLow( @Cast("l_uint32*") IntPointer lined, @Cast("l_int32") int w, @Cast("l_uint32*") IntPointer bufs1, @Cast("l_uint32*") IntPointer bufs2, @Cast("l_int32*") IntPointer tabval, @Cast("l_int32*") IntPointer tab38, @Cast("l_int32*") IntPointer tab14, @Cast("l_int32") int lastlineflag );
public static native void ditherTo2bppLineLow( @Cast("l_uint32*") IntBuffer lined, @Cast("l_int32") int w, @Cast("l_uint32*") IntBuffer bufs1, @Cast("l_uint32*") IntBuffer bufs2, @Cast("l_int32*") IntBuffer tabval, @Cast("l_int32*") IntBuffer tab38, @Cast("l_int32*") IntBuffer tab14, @Cast("l_int32") int lastlineflag );
public static native void ditherTo2bppLineLow( @Cast("l_uint32*") int[] lined, @Cast("l_int32") int w, @Cast("l_uint32*") int[] bufs1, @Cast("l_uint32*") int[] bufs2, @Cast("l_int32*") int[] tabval, @Cast("l_int32*") int[] tab38, @Cast("l_int32*") int[] tab14, @Cast("l_int32") int lastlineflag );
public static native @Cast("l_int32") int make8To2DitherTables( @Cast("l_int32**") PointerPointer ptabval, @Cast("l_int32**") PointerPointer ptab38, @Cast("l_int32**") PointerPointer ptab14, @Cast("l_int32") int cliptoblack, @Cast("l_int32") int cliptowhite );
public static native @Cast("l_int32") int make8To2DitherTables( @Cast("l_int32**") @ByPtrPtr IntPointer ptabval, @Cast("l_int32**") @ByPtrPtr IntPointer ptab38, @Cast("l_int32**") @ByPtrPtr IntPointer ptab14, @Cast("l_int32") int cliptoblack, @Cast("l_int32") int cliptowhite );
public static native @Cast("l_int32") int make8To2DitherTables( @Cast("l_int32**") @ByPtrPtr IntBuffer ptabval, @Cast("l_int32**") @ByPtrPtr IntBuffer ptab38, @Cast("l_int32**") @ByPtrPtr IntBuffer ptab14, @Cast("l_int32") int cliptoblack, @Cast("l_int32") int cliptowhite );
public static native @Cast("l_int32") int make8To2DitherTables( @Cast("l_int32**") @ByPtrPtr int[] ptabval, @Cast("l_int32**") @ByPtrPtr int[] ptab38, @Cast("l_int32**") @ByPtrPtr int[] ptab14, @Cast("l_int32") int cliptoblack, @Cast("l_int32") int cliptowhite );
public static native void thresholdTo2bppLow( @Cast("l_uint32*") IntPointer datad, @Cast("l_int32") int h, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntPointer datas, @Cast("l_int32") int wpls, @Cast("l_int32*") IntPointer tab );
public static native void thresholdTo2bppLow( @Cast("l_uint32*") IntBuffer datad, @Cast("l_int32") int h, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntBuffer datas, @Cast("l_int32") int wpls, @Cast("l_int32*") IntBuffer tab );
public static native void thresholdTo2bppLow( @Cast("l_uint32*") int[] datad, @Cast("l_int32") int h, @Cast("l_int32") int wpld, @Cast("l_uint32*") int[] datas, @Cast("l_int32") int wpls, @Cast("l_int32*") int[] tab );
public static native void thresholdTo4bppLow( @Cast("l_uint32*") IntPointer datad, @Cast("l_int32") int h, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntPointer datas, @Cast("l_int32") int wpls, @Cast("l_int32*") IntPointer tab );
public static native void thresholdTo4bppLow( @Cast("l_uint32*") IntBuffer datad, @Cast("l_int32") int h, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntBuffer datas, @Cast("l_int32") int wpls, @Cast("l_int32*") IntBuffer tab );
public static native void thresholdTo4bppLow( @Cast("l_uint32*") int[] datad, @Cast("l_int32") int h, @Cast("l_int32") int wpld, @Cast("l_uint32*") int[] datas, @Cast("l_int32") int wpls, @Cast("l_int32*") int[] tab );
public static native L_HEAP lheapCreate( @Cast("l_int32") int nalloc, @Cast("l_int32") int direction );
public static native void lheapDestroy( @Cast("L_HEAP**") PointerPointer plh, @Cast("l_int32") int freeflag );
public static native void lheapDestroy( @ByPtrPtr L_HEAP plh, @Cast("l_int32") int freeflag );
public static native @Cast("l_int32") int lheapAdd( L_HEAP lh, Pointer item );
public static native Pointer lheapRemove( L_HEAP lh );
public static native @Cast("l_int32") int lheapGetCount( L_HEAP lh );
public static native @Cast("l_int32") int lheapSwapUp( L_HEAP lh, @Cast("l_int32") int index );
public static native @Cast("l_int32") int lheapSwapDown( L_HEAP lh );
public static native @Cast("l_int32") int lheapSort( L_HEAP lh );
public static native @Cast("l_int32") int lheapSortStrictOrder( L_HEAP lh );
public static native @Cast("l_int32") int lheapPrint( @Cast("FILE*") Pointer fp, L_HEAP lh );
public static native JBCLASSER jbRankHausInit( @Cast("l_int32") int components, @Cast("l_int32") int maxwidth, @Cast("l_int32") int maxheight, @Cast("l_int32") int size, @Cast("l_float32") float rank );
public static native JBCLASSER jbCorrelationInit( @Cast("l_int32") int components, @Cast("l_int32") int maxwidth, @Cast("l_int32") int maxheight, @Cast("l_float32") float thresh, @Cast("l_float32") float weightfactor );
public static native JBCLASSER jbCorrelationInitWithoutComponents( @Cast("l_int32") int components, @Cast("l_int32") int maxwidth, @Cast("l_int32") int maxheight, @Cast("l_float32") float thresh, @Cast("l_float32") float weightfactor );
public static native @Cast("l_int32") int jbAddPages( JBCLASSER classer, SARRAY safiles );
public static native @Cast("l_int32") int jbAddPage( JBCLASSER classer, PIX pixs );
public static native @Cast("l_int32") int jbAddPageComponents( JBCLASSER classer, PIX pixs, BOXA boxas, PIXA pixas );
public static native @Cast("l_int32") int jbClassifyRankHaus( JBCLASSER classer, BOXA boxa, PIXA pixas );
public static native @Cast("l_int32") int pixHaustest( PIX pix1, PIX pix2, PIX pix3, PIX pix4, @Cast("l_float32") float delx, @Cast("l_float32") float dely, @Cast("l_int32") int maxdiffw, @Cast("l_int32") int maxdiffh );
public static native @Cast("l_int32") int pixRankHaustest( PIX pix1, PIX pix2, PIX pix3, PIX pix4, @Cast("l_float32") float delx, @Cast("l_float32") float dely, @Cast("l_int32") int maxdiffw, @Cast("l_int32") int maxdiffh, @Cast("l_int32") int area1, @Cast("l_int32") int area3, @Cast("l_float32") float rank, @Cast("l_int32*") IntPointer tab8 );
public static native @Cast("l_int32") int pixRankHaustest( PIX pix1, PIX pix2, PIX pix3, PIX pix4, @Cast("l_float32") float delx, @Cast("l_float32") float dely, @Cast("l_int32") int maxdiffw, @Cast("l_int32") int maxdiffh, @Cast("l_int32") int area1, @Cast("l_int32") int area3, @Cast("l_float32") float rank, @Cast("l_int32*") IntBuffer tab8 );
public static native @Cast("l_int32") int pixRankHaustest( PIX pix1, PIX pix2, PIX pix3, PIX pix4, @Cast("l_float32") float delx, @Cast("l_float32") float dely, @Cast("l_int32") int maxdiffw, @Cast("l_int32") int maxdiffh, @Cast("l_int32") int area1, @Cast("l_int32") int area3, @Cast("l_float32") float rank, @Cast("l_int32*") int[] tab8 );
public static native @Cast("l_int32") int jbClassifyCorrelation( JBCLASSER classer, BOXA boxa, PIXA pixas );
public static native @Cast("l_int32") int jbGetComponents( PIX pixs, @Cast("l_int32") int components, @Cast("l_int32") int maxwidth, @Cast("l_int32") int maxheight, @Cast("BOXA**") PointerPointer pboxad, @Cast("PIXA**") PointerPointer ppixad );
public static native @Cast("l_int32") int jbGetComponents( PIX pixs, @Cast("l_int32") int components, @Cast("l_int32") int maxwidth, @Cast("l_int32") int maxheight, @ByPtrPtr BOXA pboxad, @ByPtrPtr PIXA ppixad );
public static native @Cast("l_int32") int pixWordMaskByDilation( PIX pixs, @Cast("l_int32") int maxdil, @Cast("PIX**") PointerPointer ppixm, @Cast("l_int32*") IntPointer psize );
public static native @Cast("l_int32") int pixWordMaskByDilation( PIX pixs, @Cast("l_int32") int maxdil, @ByPtrPtr PIX ppixm, @Cast("l_int32*") IntPointer psize );
public static native @Cast("l_int32") int pixWordMaskByDilation( PIX pixs, @Cast("l_int32") int maxdil, @ByPtrPtr PIX ppixm, @Cast("l_int32*") IntBuffer psize );
public static native @Cast("l_int32") int pixWordMaskByDilation( PIX pixs, @Cast("l_int32") int maxdil, @ByPtrPtr PIX ppixm, @Cast("l_int32*") int[] psize );
public static native @Cast("l_int32") int pixWordBoxesByDilation( PIX pixs, @Cast("l_int32") int maxdil, @Cast("l_int32") int minwidth, @Cast("l_int32") int minheight, @Cast("l_int32") int maxwidth, @Cast("l_int32") int maxheight, @Cast("BOXA**") PointerPointer pboxa, @Cast("l_int32*") IntPointer psize );
public static native @Cast("l_int32") int pixWordBoxesByDilation( PIX pixs, @Cast("l_int32") int maxdil, @Cast("l_int32") int minwidth, @Cast("l_int32") int minheight, @Cast("l_int32") int maxwidth, @Cast("l_int32") int maxheight, @ByPtrPtr BOXA pboxa, @Cast("l_int32*") IntPointer psize );
public static native @Cast("l_int32") int pixWordBoxesByDilation( PIX pixs, @Cast("l_int32") int maxdil, @Cast("l_int32") int minwidth, @Cast("l_int32") int minheight, @Cast("l_int32") int maxwidth, @Cast("l_int32") int maxheight, @ByPtrPtr BOXA pboxa, @Cast("l_int32*") IntBuffer psize );
public static native @Cast("l_int32") int pixWordBoxesByDilation( PIX pixs, @Cast("l_int32") int maxdil, @Cast("l_int32") int minwidth, @Cast("l_int32") int minheight, @Cast("l_int32") int maxwidth, @Cast("l_int32") int maxheight, @ByPtrPtr BOXA pboxa, @Cast("l_int32*") int[] psize );
public static native PIXA jbAccumulateComposites( PIXAA pixaa, @Cast("NUMA**") PointerPointer pna, @Cast("PTA**") PointerPointer pptat );
public static native PIXA jbAccumulateComposites( PIXAA pixaa, @ByPtrPtr NUMA pna, @ByPtrPtr PTA pptat );
public static native PIXA jbTemplatesFromComposites( PIXA pixac, NUMA na );
public static native JBCLASSER jbClasserCreate( @Cast("l_int32") int method, @Cast("l_int32") int components );
public static native void jbClasserDestroy( @Cast("JBCLASSER**") PointerPointer pclasser );
public static native void jbClasserDestroy( @ByPtrPtr JBCLASSER pclasser );
public static native JBDATA jbDataSave( JBCLASSER classer );
public static native void jbDataDestroy( @Cast("JBDATA**") PointerPointer pdata );
public static native void jbDataDestroy( @ByPtrPtr JBDATA pdata );
public static native @Cast("l_int32") int jbDataWrite( @Cast("const char*") BytePointer rootout, JBDATA jbdata );
public static native @Cast("l_int32") int jbDataWrite( String rootout, JBDATA jbdata );
public static native JBDATA jbDataRead( @Cast("const char*") BytePointer rootname );
public static native JBDATA jbDataRead( String rootname );
public static native PIXA jbDataRender( JBDATA data, @Cast("l_int32") int debugflag );
public static native @Cast("l_int32") int jbGetULCorners( JBCLASSER classer, PIX pixs, BOXA boxa );
public static native @Cast("l_int32") int jbGetLLCorners( JBCLASSER classer );
public static native @Cast("l_int32") int readHeaderJp2k( @Cast("const char*") BytePointer filename, @Cast("l_int32*") IntPointer pw, @Cast("l_int32*") IntPointer ph, @Cast("l_int32*") IntPointer pbps, @Cast("l_int32*") IntPointer pspp );
public static native @Cast("l_int32") int readHeaderJp2k( String filename, @Cast("l_int32*") IntBuffer pw, @Cast("l_int32*") IntBuffer ph, @Cast("l_int32*") IntBuffer pbps, @Cast("l_int32*") IntBuffer pspp );
public static native @Cast("l_int32") int readHeaderJp2k( @Cast("const char*") BytePointer filename, @Cast("l_int32*") int[] pw, @Cast("l_int32*") int[] ph, @Cast("l_int32*") int[] pbps, @Cast("l_int32*") int[] pspp );
public static native @Cast("l_int32") int readHeaderJp2k( String filename, @Cast("l_int32*") IntPointer pw, @Cast("l_int32*") IntPointer ph, @Cast("l_int32*") IntPointer pbps, @Cast("l_int32*") IntPointer pspp );
public static native @Cast("l_int32") int readHeaderJp2k( @Cast("const char*") BytePointer filename, @Cast("l_int32*") IntBuffer pw, @Cast("l_int32*") IntBuffer ph, @Cast("l_int32*") IntBuffer pbps, @Cast("l_int32*") IntBuffer pspp );
public static native @Cast("l_int32") int readHeaderJp2k( String filename, @Cast("l_int32*") int[] pw, @Cast("l_int32*") int[] ph, @Cast("l_int32*") int[] pbps, @Cast("l_int32*") int[] pspp );
public static native @Cast("l_int32") int freadHeaderJp2k( @Cast("FILE*") Pointer fp, @Cast("l_int32*") IntPointer pw, @Cast("l_int32*") IntPointer ph, @Cast("l_int32*") IntPointer pbps, @Cast("l_int32*") IntPointer pspp );
public static native @Cast("l_int32") int freadHeaderJp2k( @Cast("FILE*") Pointer fp, @Cast("l_int32*") IntBuffer pw, @Cast("l_int32*") IntBuffer ph, @Cast("l_int32*") IntBuffer pbps, @Cast("l_int32*") IntBuffer pspp );
public static native @Cast("l_int32") int freadHeaderJp2k( @Cast("FILE*") Pointer fp, @Cast("l_int32*") int[] pw, @Cast("l_int32*") int[] ph, @Cast("l_int32*") int[] pbps, @Cast("l_int32*") int[] pspp );
public static native @Cast("l_int32") int readHeaderMemJp2k( @Cast("const l_uint8*") BytePointer data, @Cast("size_t") long size, @Cast("l_int32*") IntPointer pw, @Cast("l_int32*") IntPointer ph, @Cast("l_int32*") IntPointer pbps, @Cast("l_int32*") IntPointer pspp );
public static native @Cast("l_int32") int readHeaderMemJp2k( @Cast("const l_uint8*") ByteBuffer data, @Cast("size_t") long size, @Cast("l_int32*") IntBuffer pw, @Cast("l_int32*") IntBuffer ph, @Cast("l_int32*") IntBuffer pbps, @Cast("l_int32*") IntBuffer pspp );
public static native @Cast("l_int32") int readHeaderMemJp2k( @Cast("const l_uint8*") byte[] data, @Cast("size_t") long size, @Cast("l_int32*") int[] pw, @Cast("l_int32*") int[] ph, @Cast("l_int32*") int[] pbps, @Cast("l_int32*") int[] pspp );
public static native @Cast("l_int32") int fgetJp2kResolution( @Cast("FILE*") Pointer fp, @Cast("l_int32*") IntPointer pxres, @Cast("l_int32*") IntPointer pyres );
public static native @Cast("l_int32") int fgetJp2kResolution( @Cast("FILE*") Pointer fp, @Cast("l_int32*") IntBuffer pxres, @Cast("l_int32*") IntBuffer pyres );
public static native @Cast("l_int32") int fgetJp2kResolution( @Cast("FILE*") Pointer fp, @Cast("l_int32*") int[] pxres, @Cast("l_int32*") int[] pyres );
public static native PIX pixReadJp2k( @Cast("const char*") BytePointer filename, @Cast("l_uint32") int reduction, BOX box, @Cast("l_int32") int hint, @Cast("l_int32") int debug );
public static native PIX pixReadJp2k( String filename, @Cast("l_uint32") int reduction, BOX box, @Cast("l_int32") int hint, @Cast("l_int32") int debug );
public static native PIX pixReadStreamJp2k( @Cast("FILE*") Pointer fp, @Cast("l_uint32") int reduction, BOX box, @Cast("l_int32") int hint, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int pixWriteJp2k( @Cast("const char*") BytePointer filename, PIX pix, @Cast("l_int32") int quality, @Cast("l_int32") int nlevels, @Cast("l_int32") int hint, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int pixWriteJp2k( String filename, PIX pix, @Cast("l_int32") int quality, @Cast("l_int32") int nlevels, @Cast("l_int32") int hint, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int pixWriteStreamJp2k( @Cast("FILE*") Pointer fp, PIX pix, @Cast("l_int32") int quality, @Cast("l_int32") int nlevels, @Cast("l_int32") int hint, @Cast("l_int32") int debug );
public static native PIX pixReadMemJp2k( @Cast("const l_uint8*") BytePointer data, @Cast("size_t") long size, @Cast("l_uint32") int reduction, BOX box, @Cast("l_int32") int hint, @Cast("l_int32") int debug );
public static native PIX pixReadMemJp2k( @Cast("const l_uint8*") ByteBuffer data, @Cast("size_t") long size, @Cast("l_uint32") int reduction, BOX box, @Cast("l_int32") int hint, @Cast("l_int32") int debug );
public static native PIX pixReadMemJp2k( @Cast("const l_uint8*") byte[] data, @Cast("size_t") long size, @Cast("l_uint32") int reduction, BOX box, @Cast("l_int32") int hint, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int pixWriteMemJp2k( @Cast("l_uint8**") PointerPointer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix, @Cast("l_int32") int quality, @Cast("l_int32") int nlevels, @Cast("l_int32") int hint, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int pixWriteMemJp2k( @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix, @Cast("l_int32") int quality, @Cast("l_int32") int nlevels, @Cast("l_int32") int hint, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int pixWriteMemJp2k( @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix, @Cast("l_int32") int quality, @Cast("l_int32") int nlevels, @Cast("l_int32") int hint, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int pixWriteMemJp2k( @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer psize, PIX pix, @Cast("l_int32") int quality, @Cast("l_int32") int nlevels, @Cast("l_int32") int hint, @Cast("l_int32") int debug );
public static native PIX pixReadJpeg( @Cast("const char*") BytePointer filename, @Cast("l_int32") int cmapflag, @Cast("l_int32") int reduction, @Cast("l_int32*") IntPointer pnwarn, @Cast("l_int32") int hint );
public static native PIX pixReadJpeg( String filename, @Cast("l_int32") int cmapflag, @Cast("l_int32") int reduction, @Cast("l_int32*") IntBuffer pnwarn, @Cast("l_int32") int hint );
public static native PIX pixReadJpeg( @Cast("const char*") BytePointer filename, @Cast("l_int32") int cmapflag, @Cast("l_int32") int reduction, @Cast("l_int32*") int[] pnwarn, @Cast("l_int32") int hint );
public static native PIX pixReadJpeg( String filename, @Cast("l_int32") int cmapflag, @Cast("l_int32") int reduction, @Cast("l_int32*") IntPointer pnwarn, @Cast("l_int32") int hint );
public static native PIX pixReadJpeg( @Cast("const char*") BytePointer filename, @Cast("l_int32") int cmapflag, @Cast("l_int32") int reduction, @Cast("l_int32*") IntBuffer pnwarn, @Cast("l_int32") int hint );
public static native PIX pixReadJpeg( String filename, @Cast("l_int32") int cmapflag, @Cast("l_int32") int reduction, @Cast("l_int32*") int[] pnwarn, @Cast("l_int32") int hint );
public static native PIX pixReadStreamJpeg( @Cast("FILE*") Pointer fp, @Cast("l_int32") int cmapflag, @Cast("l_int32") int reduction, @Cast("l_int32*") IntPointer pnwarn, @Cast("l_int32") int hint );
public static native PIX pixReadStreamJpeg( @Cast("FILE*") Pointer fp, @Cast("l_int32") int cmapflag, @Cast("l_int32") int reduction, @Cast("l_int32*") IntBuffer pnwarn, @Cast("l_int32") int hint );
public static native PIX pixReadStreamJpeg( @Cast("FILE*") Pointer fp, @Cast("l_int32") int cmapflag, @Cast("l_int32") int reduction, @Cast("l_int32*") int[] pnwarn, @Cast("l_int32") int hint );
public static native @Cast("l_int32") int readHeaderJpeg( @Cast("const char*") BytePointer filename, @Cast("l_int32*") IntPointer pw, @Cast("l_int32*") IntPointer ph, @Cast("l_int32*") IntPointer pspp, @Cast("l_int32*") IntPointer pycck, @Cast("l_int32*") IntPointer pcmyk );
public static native @Cast("l_int32") int readHeaderJpeg( String filename, @Cast("l_int32*") IntBuffer pw, @Cast("l_int32*") IntBuffer ph, @Cast("l_int32*") IntBuffer pspp, @Cast("l_int32*") IntBuffer pycck, @Cast("l_int32*") IntBuffer pcmyk );
public static native @Cast("l_int32") int readHeaderJpeg( @Cast("const char*") BytePointer filename, @Cast("l_int32*") int[] pw, @Cast("l_int32*") int[] ph, @Cast("l_int32*") int[] pspp, @Cast("l_int32*") int[] pycck, @Cast("l_int32*") int[] pcmyk );
public static native @Cast("l_int32") int readHeaderJpeg( String filename, @Cast("l_int32*") IntPointer pw, @Cast("l_int32*") IntPointer ph, @Cast("l_int32*") IntPointer pspp, @Cast("l_int32*") IntPointer pycck, @Cast("l_int32*") IntPointer pcmyk );
public static native @Cast("l_int32") int readHeaderJpeg( @Cast("const char*") BytePointer filename, @Cast("l_int32*") IntBuffer pw, @Cast("l_int32*") IntBuffer ph, @Cast("l_int32*") IntBuffer pspp, @Cast("l_int32*") IntBuffer pycck, @Cast("l_int32*") IntBuffer pcmyk );
public static native @Cast("l_int32") int readHeaderJpeg( String filename, @Cast("l_int32*") int[] pw, @Cast("l_int32*") int[] ph, @Cast("l_int32*") int[] pspp, @Cast("l_int32*") int[] pycck, @Cast("l_int32*") int[] pcmyk );
public static native @Cast("l_int32") int freadHeaderJpeg( @Cast("FILE*") Pointer fp, @Cast("l_int32*") IntPointer pw, @Cast("l_int32*") IntPointer ph, @Cast("l_int32*") IntPointer pspp, @Cast("l_int32*") IntPointer pycck, @Cast("l_int32*") IntPointer pcmyk );
public static native @Cast("l_int32") int freadHeaderJpeg( @Cast("FILE*") Pointer fp, @Cast("l_int32*") IntBuffer pw, @Cast("l_int32*") IntBuffer ph, @Cast("l_int32*") IntBuffer pspp, @Cast("l_int32*") IntBuffer pycck, @Cast("l_int32*") IntBuffer pcmyk );
public static native @Cast("l_int32") int freadHeaderJpeg( @Cast("FILE*") Pointer fp, @Cast("l_int32*") int[] pw, @Cast("l_int32*") int[] ph, @Cast("l_int32*") int[] pspp, @Cast("l_int32*") int[] pycck, @Cast("l_int32*") int[] pcmyk );
public static native @Cast("l_int32") int fgetJpegResolution( @Cast("FILE*") Pointer fp, @Cast("l_int32*") IntPointer pxres, @Cast("l_int32*") IntPointer pyres );
public static native @Cast("l_int32") int fgetJpegResolution( @Cast("FILE*") Pointer fp, @Cast("l_int32*") IntBuffer pxres, @Cast("l_int32*") IntBuffer pyres );
public static native @Cast("l_int32") int fgetJpegResolution( @Cast("FILE*") Pointer fp, @Cast("l_int32*") int[] pxres, @Cast("l_int32*") int[] pyres );
public static native @Cast("l_int32") int fgetJpegComment( @Cast("FILE*") Pointer fp, @Cast("l_uint8**") PointerPointer pcomment );
public static native @Cast("l_int32") int fgetJpegComment( @Cast("FILE*") Pointer fp, @Cast("l_uint8**") @ByPtrPtr BytePointer pcomment );
public static native @Cast("l_int32") int fgetJpegComment( @Cast("FILE*") Pointer fp, @Cast("l_uint8**") @ByPtrPtr ByteBuffer pcomment );
public static native @Cast("l_int32") int fgetJpegComment( @Cast("FILE*") Pointer fp, @Cast("l_uint8**") @ByPtrPtr byte[] pcomment );
public static native @Cast("l_int32") int pixWriteJpeg( @Cast("const char*") BytePointer filename, PIX pix, @Cast("l_int32") int quality, @Cast("l_int32") int progressive );
public static native @Cast("l_int32") int pixWriteJpeg( String filename, PIX pix, @Cast("l_int32") int quality, @Cast("l_int32") int progressive );
public static native @Cast("l_int32") int pixWriteStreamJpeg( @Cast("FILE*") Pointer fp, PIX pixs, @Cast("l_int32") int quality, @Cast("l_int32") int progressive );
public static native PIX pixReadMemJpeg( @Cast("const l_uint8*") BytePointer data, @Cast("size_t") long size, @Cast("l_int32") int cmflag, @Cast("l_int32") int reduction, @Cast("l_int32*") IntPointer pnwarn, @Cast("l_int32") int hint );
public static native PIX pixReadMemJpeg( @Cast("const l_uint8*") ByteBuffer data, @Cast("size_t") long size, @Cast("l_int32") int cmflag, @Cast("l_int32") int reduction, @Cast("l_int32*") IntBuffer pnwarn, @Cast("l_int32") int hint );
public static native PIX pixReadMemJpeg( @Cast("const l_uint8*") byte[] data, @Cast("size_t") long size, @Cast("l_int32") int cmflag, @Cast("l_int32") int reduction, @Cast("l_int32*") int[] pnwarn, @Cast("l_int32") int hint );
public static native @Cast("l_int32") int readHeaderMemJpeg( @Cast("const l_uint8*") BytePointer data, @Cast("size_t") long size, @Cast("l_int32*") IntPointer pw, @Cast("l_int32*") IntPointer ph, @Cast("l_int32*") IntPointer pspp, @Cast("l_int32*") IntPointer pycck, @Cast("l_int32*") IntPointer pcmyk );
public static native @Cast("l_int32") int readHeaderMemJpeg( @Cast("const l_uint8*") ByteBuffer data, @Cast("size_t") long size, @Cast("l_int32*") IntBuffer pw, @Cast("l_int32*") IntBuffer ph, @Cast("l_int32*") IntBuffer pspp, @Cast("l_int32*") IntBuffer pycck, @Cast("l_int32*") IntBuffer pcmyk );
public static native @Cast("l_int32") int readHeaderMemJpeg( @Cast("const l_uint8*") byte[] data, @Cast("size_t") long size, @Cast("l_int32*") int[] pw, @Cast("l_int32*") int[] ph, @Cast("l_int32*") int[] pspp, @Cast("l_int32*") int[] pycck, @Cast("l_int32*") int[] pcmyk );
public static native @Cast("l_int32") int pixWriteMemJpeg( @Cast("l_uint8**") PointerPointer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix, @Cast("l_int32") int quality, @Cast("l_int32") int progressive );
public static native @Cast("l_int32") int pixWriteMemJpeg( @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix, @Cast("l_int32") int quality, @Cast("l_int32") int progressive );
public static native @Cast("l_int32") int pixWriteMemJpeg( @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix, @Cast("l_int32") int quality, @Cast("l_int32") int progressive );
public static native @Cast("l_int32") int pixWriteMemJpeg( @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer psize, PIX pix, @Cast("l_int32") int quality, @Cast("l_int32") int progressive );
public static native @Cast("l_int32") int pixSetChromaSampling( PIX pix, @Cast("l_int32") int sampling );
public static native L_KERNEL kernelCreate( @Cast("l_int32") int height, @Cast("l_int32") int width );
public static native void kernelDestroy( @Cast("L_KERNEL**") PointerPointer pkel );
public static native void kernelDestroy( @ByPtrPtr L_KERNEL pkel );
public static native L_KERNEL kernelCopy( L_KERNEL kels );
public static native @Cast("l_int32") int kernelGetElement( L_KERNEL kel, @Cast("l_int32") int row, @Cast("l_int32") int col, @Cast("l_float32*") FloatPointer pval );
public static native @Cast("l_int32") int kernelGetElement( L_KERNEL kel, @Cast("l_int32") int row, @Cast("l_int32") int col, @Cast("l_float32*") FloatBuffer pval );
public static native @Cast("l_int32") int kernelGetElement( L_KERNEL kel, @Cast("l_int32") int row, @Cast("l_int32") int col, @Cast("l_float32*") float[] pval );
public static native @Cast("l_int32") int kernelSetElement( L_KERNEL kel, @Cast("l_int32") int row, @Cast("l_int32") int col, @Cast("l_float32") float val );
public static native @Cast("l_int32") int kernelGetParameters( L_KERNEL kel, @Cast("l_int32*") IntPointer psy, @Cast("l_int32*") IntPointer psx, @Cast("l_int32*") IntPointer pcy, @Cast("l_int32*") IntPointer pcx );
public static native @Cast("l_int32") int kernelGetParameters( L_KERNEL kel, @Cast("l_int32*") IntBuffer psy, @Cast("l_int32*") IntBuffer psx, @Cast("l_int32*") IntBuffer pcy, @Cast("l_int32*") IntBuffer pcx );
public static native @Cast("l_int32") int kernelGetParameters( L_KERNEL kel, @Cast("l_int32*") int[] psy, @Cast("l_int32*") int[] psx, @Cast("l_int32*") int[] pcy, @Cast("l_int32*") int[] pcx );
public static native @Cast("l_int32") int kernelSetOrigin( L_KERNEL kel, @Cast("l_int32") int cy, @Cast("l_int32") int cx );
public static native @Cast("l_int32") int kernelGetSum( L_KERNEL kel, @Cast("l_float32*") FloatPointer psum );
public static native @Cast("l_int32") int kernelGetSum( L_KERNEL kel, @Cast("l_float32*") FloatBuffer psum );
public static native @Cast("l_int32") int kernelGetSum( L_KERNEL kel, @Cast("l_float32*") float[] psum );
public static native @Cast("l_int32") int kernelGetMinMax( L_KERNEL kel, @Cast("l_float32*") FloatPointer pmin, @Cast("l_float32*") FloatPointer pmax );
public static native @Cast("l_int32") int kernelGetMinMax( L_KERNEL kel, @Cast("l_float32*") FloatBuffer pmin, @Cast("l_float32*") FloatBuffer pmax );
public static native @Cast("l_int32") int kernelGetMinMax( L_KERNEL kel, @Cast("l_float32*") float[] pmin, @Cast("l_float32*") float[] pmax );
public static native L_KERNEL kernelNormalize( L_KERNEL kels, @Cast("l_float32") float normsum );
public static native L_KERNEL kernelInvert( L_KERNEL kels );
public static native @Cast("l_float32**") PointerPointer create2dFloatArray( @Cast("l_int32") int sy, @Cast("l_int32") int sx );
public static native L_KERNEL kernelRead( @Cast("const char*") BytePointer fname );
public static native L_KERNEL kernelRead( String fname );
public static native L_KERNEL kernelReadStream( @Cast("FILE*") Pointer fp );
public static native @Cast("l_int32") int kernelWrite( @Cast("const char*") BytePointer fname, L_KERNEL kel );
public static native @Cast("l_int32") int kernelWrite( String fname, L_KERNEL kel );
public static native @Cast("l_int32") int kernelWriteStream( @Cast("FILE*") Pointer fp, L_KERNEL kel );
public static native L_KERNEL kernelCreateFromString( @Cast("l_int32") int h, @Cast("l_int32") int w, @Cast("l_int32") int cy, @Cast("l_int32") int cx, @Cast("const char*") BytePointer kdata );
public static native L_KERNEL kernelCreateFromString( @Cast("l_int32") int h, @Cast("l_int32") int w, @Cast("l_int32") int cy, @Cast("l_int32") int cx, String kdata );
public static native L_KERNEL kernelCreateFromFile( @Cast("const char*") BytePointer filename );
public static native L_KERNEL kernelCreateFromFile( String filename );
public static native L_KERNEL kernelCreateFromPix( PIX pix, @Cast("l_int32") int cy, @Cast("l_int32") int cx );
public static native PIX kernelDisplayInPix( L_KERNEL kel, @Cast("l_int32") int size, @Cast("l_int32") int gthick );
public static native NUMA parseStringForNumbers( @Cast("const char*") BytePointer str, @Cast("const char*") BytePointer seps );
public static native NUMA parseStringForNumbers( String str, String seps );
public static native L_KERNEL makeFlatKernel( @Cast("l_int32") int height, @Cast("l_int32") int width, @Cast("l_int32") int cy, @Cast("l_int32") int cx );
public static native L_KERNEL makeGaussianKernel( @Cast("l_int32") int halfheight, @Cast("l_int32") int halfwidth, @Cast("l_float32") float stdev, @Cast("l_float32") float max );
public static native @Cast("l_int32") int makeGaussianKernelSep( @Cast("l_int32") int halfheight, @Cast("l_int32") int halfwidth, @Cast("l_float32") float stdev, @Cast("l_float32") float max, @Cast("L_KERNEL**") PointerPointer pkelx, @Cast("L_KERNEL**") PointerPointer pkely );
public static native @Cast("l_int32") int makeGaussianKernelSep( @Cast("l_int32") int halfheight, @Cast("l_int32") int halfwidth, @Cast("l_float32") float stdev, @Cast("l_float32") float max, @ByPtrPtr L_KERNEL pkelx, @ByPtrPtr L_KERNEL pkely );
public static native L_KERNEL makeDoGKernel( @Cast("l_int32") int halfheight, @Cast("l_int32") int halfwidth, @Cast("l_float32") float stdev, @Cast("l_float32") float ratio );
public static native @Cast("char*") BytePointer getImagelibVersions(  );
public static native void listDestroy( @Cast("DLLIST**") PointerPointer phead );
public static native void listDestroy( @ByPtrPtr DLLIST phead );
public static native @Cast("l_int32") int listAddToHead( @Cast("DLLIST**") PointerPointer phead, Pointer data );
public static native @Cast("l_int32") int listAddToHead( @ByPtrPtr DLLIST phead, Pointer data );
public static native @Cast("l_int32") int listAddToTail( @Cast("DLLIST**") PointerPointer phead, @Cast("DLLIST**") PointerPointer ptail, Pointer data );
public static native @Cast("l_int32") int listAddToTail( @ByPtrPtr DLLIST phead, @ByPtrPtr DLLIST ptail, Pointer data );
public static native @Cast("l_int32") int listInsertBefore( @Cast("DLLIST**") PointerPointer phead, DLLIST elem, Pointer data );
public static native @Cast("l_int32") int listInsertBefore( @ByPtrPtr DLLIST phead, DLLIST elem, Pointer data );
public static native @Cast("l_int32") int listInsertAfter( @Cast("DLLIST**") PointerPointer phead, DLLIST elem, Pointer data );
public static native @Cast("l_int32") int listInsertAfter( @ByPtrPtr DLLIST phead, DLLIST elem, Pointer data );
public static native Pointer listRemoveElement( @Cast("DLLIST**") PointerPointer phead, DLLIST elem );
public static native Pointer listRemoveElement( @ByPtrPtr DLLIST phead, DLLIST elem );
public static native Pointer listRemoveFromHead( @Cast("DLLIST**") PointerPointer phead );
public static native Pointer listRemoveFromHead( @ByPtrPtr DLLIST phead );
public static native Pointer listRemoveFromTail( @Cast("DLLIST**") PointerPointer phead, @Cast("DLLIST**") PointerPointer ptail );
public static native Pointer listRemoveFromTail( @ByPtrPtr DLLIST phead, @ByPtrPtr DLLIST ptail );
public static native DLLIST listFindElement( DLLIST head, Pointer data );
public static native DLLIST listFindTail( DLLIST head );
public static native @Cast("l_int32") int listGetCount( DLLIST head );
public static native @Cast("l_int32") int listReverse( @Cast("DLLIST**") PointerPointer phead );
public static native @Cast("l_int32") int listReverse( @ByPtrPtr DLLIST phead );
public static native @Cast("l_int32") int listJoin( @Cast("DLLIST**") PointerPointer phead1, @Cast("DLLIST**") PointerPointer phead2 );
public static native @Cast("l_int32") int listJoin( @ByPtrPtr DLLIST phead1, @ByPtrPtr DLLIST phead2 );
public static native PIX generateBinaryMaze( @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int xi, @Cast("l_int32") int yi, @Cast("l_float32") float wallps, @Cast("l_float32") float ranis );
public static native PTA pixSearchBinaryMaze( PIX pixs, @Cast("l_int32") int xi, @Cast("l_int32") int yi, @Cast("l_int32") int xf, @Cast("l_int32") int yf, @Cast("PIX**") PointerPointer ppixd );
public static native PTA pixSearchBinaryMaze( PIX pixs, @Cast("l_int32") int xi, @Cast("l_int32") int yi, @Cast("l_int32") int xf, @Cast("l_int32") int yf, @ByPtrPtr PIX ppixd );
public static native PTA pixSearchGrayMaze( PIX pixs, @Cast("l_int32") int xi, @Cast("l_int32") int yi, @Cast("l_int32") int xf, @Cast("l_int32") int yf, @Cast("PIX**") PointerPointer ppixd );
public static native PTA pixSearchGrayMaze( PIX pixs, @Cast("l_int32") int xi, @Cast("l_int32") int yi, @Cast("l_int32") int xf, @Cast("l_int32") int yf, @ByPtrPtr PIX ppixd );
public static native @Cast("l_int32") int pixFindLargestRectangle( PIX pixs, @Cast("l_int32") int polarity, @Cast("BOX**") PointerPointer pbox, @Cast("const char*") BytePointer debugfile );
public static native @Cast("l_int32") int pixFindLargestRectangle( PIX pixs, @Cast("l_int32") int polarity, @ByPtrPtr BOX pbox, @Cast("const char*") BytePointer debugfile );
public static native @Cast("l_int32") int pixFindLargestRectangle( PIX pixs, @Cast("l_int32") int polarity, @ByPtrPtr BOX pbox, String debugfile );
public static native PIX pixDilate( PIX pixd, PIX pixs, SEL sel );
public static native PIX pixErode( PIX pixd, PIX pixs, SEL sel );
public static native PIX pixHMT( PIX pixd, PIX pixs, SEL sel );
public static native PIX pixOpen( PIX pixd, PIX pixs, SEL sel );
public static native PIX pixClose( PIX pixd, PIX pixs, SEL sel );
public static native PIX pixCloseSafe( PIX pixd, PIX pixs, SEL sel );
public static native PIX pixOpenGeneralized( PIX pixd, PIX pixs, SEL sel );
public static native PIX pixCloseGeneralized( PIX pixd, PIX pixs, SEL sel );
public static native PIX pixDilateBrick( PIX pixd, PIX pixs, @Cast("l_int32") int hsize, @Cast("l_int32") int vsize );
public static native PIX pixErodeBrick( PIX pixd, PIX pixs, @Cast("l_int32") int hsize, @Cast("l_int32") int vsize );
public static native PIX pixOpenBrick( PIX pixd, PIX pixs, @Cast("l_int32") int hsize, @Cast("l_int32") int vsize );
public static native PIX pixCloseBrick( PIX pixd, PIX pixs, @Cast("l_int32") int hsize, @Cast("l_int32") int vsize );
public static native PIX pixCloseSafeBrick( PIX pixd, PIX pixs, @Cast("l_int32") int hsize, @Cast("l_int32") int vsize );
public static native @Cast("l_int32") int selectComposableSels( @Cast("l_int32") int size, @Cast("l_int32") int direction, @Cast("SEL**") PointerPointer psel1, @Cast("SEL**") PointerPointer psel2 );
public static native @Cast("l_int32") int selectComposableSels( @Cast("l_int32") int size, @Cast("l_int32") int direction, @ByPtrPtr SEL psel1, @ByPtrPtr SEL psel2 );
public static native @Cast("l_int32") int selectComposableSizes( @Cast("l_int32") int size, @Cast("l_int32*") IntPointer pfactor1, @Cast("l_int32*") IntPointer pfactor2 );
public static native @Cast("l_int32") int selectComposableSizes( @Cast("l_int32") int size, @Cast("l_int32*") IntBuffer pfactor1, @Cast("l_int32*") IntBuffer pfactor2 );
public static native @Cast("l_int32") int selectComposableSizes( @Cast("l_int32") int size, @Cast("l_int32*") int[] pfactor1, @Cast("l_int32*") int[] pfactor2 );
public static native PIX pixDilateCompBrick( PIX pixd, PIX pixs, @Cast("l_int32") int hsize, @Cast("l_int32") int vsize );
public static native PIX pixErodeCompBrick( PIX pixd, PIX pixs, @Cast("l_int32") int hsize, @Cast("l_int32") int vsize );
public static native PIX pixOpenCompBrick( PIX pixd, PIX pixs, @Cast("l_int32") int hsize, @Cast("l_int32") int vsize );
public static native PIX pixCloseCompBrick( PIX pixd, PIX pixs, @Cast("l_int32") int hsize, @Cast("l_int32") int vsize );
public static native PIX pixCloseSafeCompBrick( PIX pixd, PIX pixs, @Cast("l_int32") int hsize, @Cast("l_int32") int vsize );
public static native void resetMorphBoundaryCondition( @Cast("l_int32") int bc );
public static native @Cast("l_uint32") int getMorphBorderPixelColor( @Cast("l_int32") int type, @Cast("l_int32") int depth );
public static native PIX pixExtractBoundary( PIX pixs, @Cast("l_int32") int type );
public static native PIX pixMorphSequenceMasked( PIX pixs, PIX pixm, @Cast("const char*") BytePointer sequence, @Cast("l_int32") int dispsep );
public static native PIX pixMorphSequenceMasked( PIX pixs, PIX pixm, String sequence, @Cast("l_int32") int dispsep );
public static native PIX pixMorphSequenceByComponent( PIX pixs, @Cast("const char*") BytePointer sequence, @Cast("l_int32") int connectivity, @Cast("l_int32") int minw, @Cast("l_int32") int minh, @Cast("BOXA**") PointerPointer pboxa );
public static native PIX pixMorphSequenceByComponent( PIX pixs, @Cast("const char*") BytePointer sequence, @Cast("l_int32") int connectivity, @Cast("l_int32") int minw, @Cast("l_int32") int minh, @ByPtrPtr BOXA pboxa );
public static native PIX pixMorphSequenceByComponent( PIX pixs, String sequence, @Cast("l_int32") int connectivity, @Cast("l_int32") int minw, @Cast("l_int32") int minh, @ByPtrPtr BOXA pboxa );
public static native PIXA pixaMorphSequenceByComponent( PIXA pixas, @Cast("const char*") BytePointer sequence, @Cast("l_int32") int minw, @Cast("l_int32") int minh );
public static native PIXA pixaMorphSequenceByComponent( PIXA pixas, String sequence, @Cast("l_int32") int minw, @Cast("l_int32") int minh );
public static native PIX pixMorphSequenceByRegion( PIX pixs, PIX pixm, @Cast("const char*") BytePointer sequence, @Cast("l_int32") int connectivity, @Cast("l_int32") int minw, @Cast("l_int32") int minh, @Cast("BOXA**") PointerPointer pboxa );
public static native PIX pixMorphSequenceByRegion( PIX pixs, PIX pixm, @Cast("const char*") BytePointer sequence, @Cast("l_int32") int connectivity, @Cast("l_int32") int minw, @Cast("l_int32") int minh, @ByPtrPtr BOXA pboxa );
public static native PIX pixMorphSequenceByRegion( PIX pixs, PIX pixm, String sequence, @Cast("l_int32") int connectivity, @Cast("l_int32") int minw, @Cast("l_int32") int minh, @ByPtrPtr BOXA pboxa );
public static native PIXA pixaMorphSequenceByRegion( PIX pixs, PIXA pixam, @Cast("const char*") BytePointer sequence, @Cast("l_int32") int minw, @Cast("l_int32") int minh );
public static native PIXA pixaMorphSequenceByRegion( PIX pixs, PIXA pixam, String sequence, @Cast("l_int32") int minw, @Cast("l_int32") int minh );
public static native PIX pixUnionOfMorphOps( PIX pixs, SELA sela, @Cast("l_int32") int type );
public static native PIX pixIntersectionOfMorphOps( PIX pixs, SELA sela, @Cast("l_int32") int type );
public static native PIX pixSelectiveConnCompFill( PIX pixs, @Cast("l_int32") int connectivity, @Cast("l_int32") int minw, @Cast("l_int32") int minh );
public static native @Cast("l_int32") int pixRemoveMatchedPattern( PIX pixs, PIX pixp, PIX pixe, @Cast("l_int32") int x0, @Cast("l_int32") int y0, @Cast("l_int32") int dsize );
public static native PIX pixDisplayMatchedPattern( PIX pixs, PIX pixp, PIX pixe, @Cast("l_int32") int x0, @Cast("l_int32") int y0, @Cast("l_uint32") int color, @Cast("l_float32") float scale, @Cast("l_int32") int nlevels );
public static native PIX pixSeedfillMorph( PIX pixs, PIX pixm, @Cast("l_int32") int maxiters, @Cast("l_int32") int connectivity );
public static native NUMA pixRunHistogramMorph( PIX pixs, @Cast("l_int32") int runtype, @Cast("l_int32") int direction, @Cast("l_int32") int maxsize );
public static native PIX pixTophat( PIX pixs, @Cast("l_int32") int hsize, @Cast("l_int32") int vsize, @Cast("l_int32") int type );
public static native PIX pixHDome( PIX pixs, @Cast("l_int32") int height, @Cast("l_int32") int connectivity );
public static native PIX pixFastTophat( PIX pixs, @Cast("l_int32") int xsize, @Cast("l_int32") int ysize, @Cast("l_int32") int type );
public static native PIX pixMorphGradient( PIX pixs, @Cast("l_int32") int hsize, @Cast("l_int32") int vsize, @Cast("l_int32") int smoothing );
public static native PTA pixaCentroids( PIXA pixa );
public static native @Cast("l_int32") int pixCentroid( PIX pix, @Cast("l_int32*") IntPointer centtab, @Cast("l_int32*") IntPointer sumtab, @Cast("l_float32*") FloatPointer pxave, @Cast("l_float32*") FloatPointer pyave );
public static native @Cast("l_int32") int pixCentroid( PIX pix, @Cast("l_int32*") IntBuffer centtab, @Cast("l_int32*") IntBuffer sumtab, @Cast("l_float32*") FloatBuffer pxave, @Cast("l_float32*") FloatBuffer pyave );
public static native @Cast("l_int32") int pixCentroid( PIX pix, @Cast("l_int32*") int[] centtab, @Cast("l_int32*") int[] sumtab, @Cast("l_float32*") float[] pxave, @Cast("l_float32*") float[] pyave );
public static native PIX pixDilateBrickDwa( PIX pixd, PIX pixs, @Cast("l_int32") int hsize, @Cast("l_int32") int vsize );
public static native PIX pixErodeBrickDwa( PIX pixd, PIX pixs, @Cast("l_int32") int hsize, @Cast("l_int32") int vsize );
public static native PIX pixOpenBrickDwa( PIX pixd, PIX pixs, @Cast("l_int32") int hsize, @Cast("l_int32") int vsize );
public static native PIX pixCloseBrickDwa( PIX pixd, PIX pixs, @Cast("l_int32") int hsize, @Cast("l_int32") int vsize );
public static native PIX pixDilateCompBrickDwa( PIX pixd, PIX pixs, @Cast("l_int32") int hsize, @Cast("l_int32") int vsize );
public static native PIX pixErodeCompBrickDwa( PIX pixd, PIX pixs, @Cast("l_int32") int hsize, @Cast("l_int32") int vsize );
public static native PIX pixOpenCompBrickDwa( PIX pixd, PIX pixs, @Cast("l_int32") int hsize, @Cast("l_int32") int vsize );
public static native PIX pixCloseCompBrickDwa( PIX pixd, PIX pixs, @Cast("l_int32") int hsize, @Cast("l_int32") int vsize );
public static native PIX pixDilateCompBrickExtendDwa( PIX pixd, PIX pixs, @Cast("l_int32") int hsize, @Cast("l_int32") int vsize );
public static native PIX pixErodeCompBrickExtendDwa( PIX pixd, PIX pixs, @Cast("l_int32") int hsize, @Cast("l_int32") int vsize );
public static native PIX pixOpenCompBrickExtendDwa( PIX pixd, PIX pixs, @Cast("l_int32") int hsize, @Cast("l_int32") int vsize );
public static native PIX pixCloseCompBrickExtendDwa( PIX pixd, PIX pixs, @Cast("l_int32") int hsize, @Cast("l_int32") int vsize );
public static native @Cast("l_int32") int getExtendedCompositeParameters( @Cast("l_int32") int size, @Cast("l_int32*") IntPointer pn, @Cast("l_int32*") IntPointer pextra, @Cast("l_int32*") IntPointer pactualsize );
public static native @Cast("l_int32") int getExtendedCompositeParameters( @Cast("l_int32") int size, @Cast("l_int32*") IntBuffer pn, @Cast("l_int32*") IntBuffer pextra, @Cast("l_int32*") IntBuffer pactualsize );
public static native @Cast("l_int32") int getExtendedCompositeParameters( @Cast("l_int32") int size, @Cast("l_int32*") int[] pn, @Cast("l_int32*") int[] pextra, @Cast("l_int32*") int[] pactualsize );
public static native PIX pixMorphSequence( PIX pixs, @Cast("const char*") BytePointer sequence, @Cast("l_int32") int dispsep );
public static native PIX pixMorphSequence( PIX pixs, String sequence, @Cast("l_int32") int dispsep );
public static native PIX pixMorphCompSequence( PIX pixs, @Cast("const char*") BytePointer sequence, @Cast("l_int32") int dispsep );
public static native PIX pixMorphCompSequence( PIX pixs, String sequence, @Cast("l_int32") int dispsep );
public static native PIX pixMorphSequenceDwa( PIX pixs, @Cast("const char*") BytePointer sequence, @Cast("l_int32") int dispsep );
public static native PIX pixMorphSequenceDwa( PIX pixs, String sequence, @Cast("l_int32") int dispsep );
public static native PIX pixMorphCompSequenceDwa( PIX pixs, @Cast("const char*") BytePointer sequence, @Cast("l_int32") int dispsep );
public static native PIX pixMorphCompSequenceDwa( PIX pixs, String sequence, @Cast("l_int32") int dispsep );
public static native @Cast("l_int32") int morphSequenceVerify( SARRAY sa );
public static native PIX pixGrayMorphSequence( PIX pixs, @Cast("const char*") BytePointer sequence, @Cast("l_int32") int dispsep, @Cast("l_int32") int dispy );
public static native PIX pixGrayMorphSequence( PIX pixs, String sequence, @Cast("l_int32") int dispsep, @Cast("l_int32") int dispy );
public static native PIX pixColorMorphSequence( PIX pixs, @Cast("const char*") BytePointer sequence, @Cast("l_int32") int dispsep, @Cast("l_int32") int dispy );
public static native PIX pixColorMorphSequence( PIX pixs, String sequence, @Cast("l_int32") int dispsep, @Cast("l_int32") int dispy );
public static native NUMA numaCreate( @Cast("l_int32") int n );
public static native NUMA numaCreateFromIArray( @Cast("l_int32*") IntPointer iarray, @Cast("l_int32") int size );
public static native NUMA numaCreateFromIArray( @Cast("l_int32*") IntBuffer iarray, @Cast("l_int32") int size );
public static native NUMA numaCreateFromIArray( @Cast("l_int32*") int[] iarray, @Cast("l_int32") int size );
public static native NUMA numaCreateFromFArray( @Cast("l_float32*") FloatPointer farray, @Cast("l_int32") int size, @Cast("l_int32") int copyflag );
public static native NUMA numaCreateFromFArray( @Cast("l_float32*") FloatBuffer farray, @Cast("l_int32") int size, @Cast("l_int32") int copyflag );
public static native NUMA numaCreateFromFArray( @Cast("l_float32*") float[] farray, @Cast("l_int32") int size, @Cast("l_int32") int copyflag );
public static native void numaDestroy( @Cast("NUMA**") PointerPointer pna );
public static native void numaDestroy( @ByPtrPtr NUMA pna );
public static native NUMA numaCopy( NUMA na );
public static native NUMA numaClone( NUMA na );
public static native @Cast("l_int32") int numaEmpty( NUMA na );
public static native @Cast("l_int32") int numaAddNumber( NUMA na, @Cast("l_float32") float val );
public static native @Cast("l_int32") int numaInsertNumber( NUMA na, @Cast("l_int32") int index, @Cast("l_float32") float val );
public static native @Cast("l_int32") int numaRemoveNumber( NUMA na, @Cast("l_int32") int index );
public static native @Cast("l_int32") int numaReplaceNumber( NUMA na, @Cast("l_int32") int index, @Cast("l_float32") float val );
public static native @Cast("l_int32") int numaGetCount( NUMA na );
public static native @Cast("l_int32") int numaSetCount( NUMA na, @Cast("l_int32") int newcount );
public static native @Cast("l_int32") int numaGetFValue( NUMA na, @Cast("l_int32") int index, @Cast("l_float32*") FloatPointer pval );
public static native @Cast("l_int32") int numaGetFValue( NUMA na, @Cast("l_int32") int index, @Cast("l_float32*") FloatBuffer pval );
public static native @Cast("l_int32") int numaGetFValue( NUMA na, @Cast("l_int32") int index, @Cast("l_float32*") float[] pval );
public static native @Cast("l_int32") int numaGetIValue( NUMA na, @Cast("l_int32") int index, @Cast("l_int32*") IntPointer pival );
public static native @Cast("l_int32") int numaGetIValue( NUMA na, @Cast("l_int32") int index, @Cast("l_int32*") IntBuffer pival );
public static native @Cast("l_int32") int numaGetIValue( NUMA na, @Cast("l_int32") int index, @Cast("l_int32*") int[] pival );
public static native @Cast("l_int32") int numaSetValue( NUMA na, @Cast("l_int32") int index, @Cast("l_float32") float val );
public static native @Cast("l_int32") int numaShiftValue( NUMA na, @Cast("l_int32") int index, @Cast("l_float32") float diff );
public static native @Cast("l_int32*") IntPointer numaGetIArray( NUMA na );
public static native @Cast("l_float32*") FloatPointer numaGetFArray( NUMA na, @Cast("l_int32") int copyflag );
public static native @Cast("l_int32") int numaGetRefcount( NUMA na );
public static native @Cast("l_int32") int numaChangeRefcount( NUMA na, @Cast("l_int32") int delta );
public static native @Cast("l_int32") int numaGetParameters( NUMA na, @Cast("l_float32*") FloatPointer pstartx, @Cast("l_float32*") FloatPointer pdelx );
public static native @Cast("l_int32") int numaGetParameters( NUMA na, @Cast("l_float32*") FloatBuffer pstartx, @Cast("l_float32*") FloatBuffer pdelx );
public static native @Cast("l_int32") int numaGetParameters( NUMA na, @Cast("l_float32*") float[] pstartx, @Cast("l_float32*") float[] pdelx );
public static native @Cast("l_int32") int numaSetParameters( NUMA na, @Cast("l_float32") float startx, @Cast("l_float32") float delx );
public static native @Cast("l_int32") int numaCopyParameters( NUMA nad, NUMA nas );
public static native SARRAY numaConvertToSarray( NUMA na, @Cast("l_int32") int size1, @Cast("l_int32") int size2, @Cast("l_int32") int addzeros, @Cast("l_int32") int type );
public static native NUMA numaRead( @Cast("const char*") BytePointer filename );
public static native NUMA numaRead( String filename );
public static native NUMA numaReadStream( @Cast("FILE*") Pointer fp );
public static native @Cast("l_int32") int numaWrite( @Cast("const char*") BytePointer filename, NUMA na );
public static native @Cast("l_int32") int numaWrite( String filename, NUMA na );
public static native @Cast("l_int32") int numaWriteStream( @Cast("FILE*") Pointer fp, NUMA na );
public static native NUMAA numaaCreate( @Cast("l_int32") int n );
public static native NUMAA numaaCreateFull( @Cast("l_int32") int ntop, @Cast("l_int32") int n );
public static native @Cast("l_int32") int numaaTruncate( NUMAA naa );
public static native void numaaDestroy( @Cast("NUMAA**") PointerPointer pnaa );
public static native void numaaDestroy( @ByPtrPtr NUMAA pnaa );
public static native @Cast("l_int32") int numaaAddNuma( NUMAA naa, NUMA na, @Cast("l_int32") int copyflag );
public static native @Cast("l_int32") int numaaExtendArray( NUMAA naa );
public static native @Cast("l_int32") int numaaGetCount( NUMAA naa );
public static native @Cast("l_int32") int numaaGetNumaCount( NUMAA naa, @Cast("l_int32") int index );
public static native @Cast("l_int32") int numaaGetNumberCount( NUMAA naa );
public static native @Cast("NUMA**") PointerPointer numaaGetPtrArray( NUMAA naa );
public static native NUMA numaaGetNuma( NUMAA naa, @Cast("l_int32") int index, @Cast("l_int32") int accessflag );
public static native @Cast("l_int32") int numaaReplaceNuma( NUMAA naa, @Cast("l_int32") int index, NUMA na );
public static native @Cast("l_int32") int numaaGetValue( NUMAA naa, @Cast("l_int32") int i, @Cast("l_int32") int j, @Cast("l_float32*") FloatPointer pfval, @Cast("l_int32*") IntPointer pival );
public static native @Cast("l_int32") int numaaGetValue( NUMAA naa, @Cast("l_int32") int i, @Cast("l_int32") int j, @Cast("l_float32*") FloatBuffer pfval, @Cast("l_int32*") IntBuffer pival );
public static native @Cast("l_int32") int numaaGetValue( NUMAA naa, @Cast("l_int32") int i, @Cast("l_int32") int j, @Cast("l_float32*") float[] pfval, @Cast("l_int32*") int[] pival );
public static native @Cast("l_int32") int numaaAddNumber( NUMAA naa, @Cast("l_int32") int index, @Cast("l_float32") float val );
public static native NUMAA numaaRead( @Cast("const char*") BytePointer filename );
public static native NUMAA numaaRead( String filename );
public static native NUMAA numaaReadStream( @Cast("FILE*") Pointer fp );
public static native @Cast("l_int32") int numaaWrite( @Cast("const char*") BytePointer filename, NUMAA naa );
public static native @Cast("l_int32") int numaaWrite( String filename, NUMAA naa );
public static native @Cast("l_int32") int numaaWriteStream( @Cast("FILE*") Pointer fp, NUMAA naa );
public static native NUMA2D numa2dCreate( @Cast("l_int32") int nrows, @Cast("l_int32") int ncols, @Cast("l_int32") int initsize );
public static native void numa2dDestroy( @Cast("NUMA2D**") PointerPointer pna2d );
public static native void numa2dDestroy( @ByPtrPtr NUMA2D pna2d );
public static native @Cast("l_int32") int numa2dAddNumber( NUMA2D na2d, @Cast("l_int32") int row, @Cast("l_int32") int col, @Cast("l_float32") float val );
public static native @Cast("l_int32") int numa2dGetCount( NUMA2D na2d, @Cast("l_int32") int row, @Cast("l_int32") int col );
public static native NUMA numa2dGetNuma( NUMA2D na2d, @Cast("l_int32") int row, @Cast("l_int32") int col );
public static native @Cast("l_int32") int numa2dGetFValue( NUMA2D na2d, @Cast("l_int32") int row, @Cast("l_int32") int col, @Cast("l_int32") int index, @Cast("l_float32*") FloatPointer pval );
public static native @Cast("l_int32") int numa2dGetFValue( NUMA2D na2d, @Cast("l_int32") int row, @Cast("l_int32") int col, @Cast("l_int32") int index, @Cast("l_float32*") FloatBuffer pval );
public static native @Cast("l_int32") int numa2dGetFValue( NUMA2D na2d, @Cast("l_int32") int row, @Cast("l_int32") int col, @Cast("l_int32") int index, @Cast("l_float32*") float[] pval );
public static native @Cast("l_int32") int numa2dGetIValue( NUMA2D na2d, @Cast("l_int32") int row, @Cast("l_int32") int col, @Cast("l_int32") int index, @Cast("l_int32*") IntPointer pval );
public static native @Cast("l_int32") int numa2dGetIValue( NUMA2D na2d, @Cast("l_int32") int row, @Cast("l_int32") int col, @Cast("l_int32") int index, @Cast("l_int32*") IntBuffer pval );
public static native @Cast("l_int32") int numa2dGetIValue( NUMA2D na2d, @Cast("l_int32") int row, @Cast("l_int32") int col, @Cast("l_int32") int index, @Cast("l_int32*") int[] pval );
public static native NUMAHASH numaHashCreate( @Cast("l_int32") int nbuckets, @Cast("l_int32") int initsize );
public static native void numaHashDestroy( @Cast("NUMAHASH**") PointerPointer pnahash );
public static native void numaHashDestroy( @ByPtrPtr NUMAHASH pnahash );
public static native NUMA numaHashGetNuma( NUMAHASH nahash, @Cast("l_uint32") int key );
public static native @Cast("l_int32") int numaHashAdd( NUMAHASH nahash, @Cast("l_uint32") int key, @Cast("l_float32") float value );
public static native NUMA numaArithOp( NUMA nad, NUMA na1, NUMA na2, @Cast("l_int32") int op );
public static native NUMA numaLogicalOp( NUMA nad, NUMA na1, NUMA na2, @Cast("l_int32") int op );
public static native NUMA numaInvert( NUMA nad, NUMA nas );
public static native @Cast("l_int32") int numaSimilar( NUMA na1, NUMA na2, @Cast("l_float32") float maxdiff, @Cast("l_int32*") IntPointer psimilar );
public static native @Cast("l_int32") int numaSimilar( NUMA na1, NUMA na2, @Cast("l_float32") float maxdiff, @Cast("l_int32*") IntBuffer psimilar );
public static native @Cast("l_int32") int numaSimilar( NUMA na1, NUMA na2, @Cast("l_float32") float maxdiff, @Cast("l_int32*") int[] psimilar );
public static native @Cast("l_int32") int numaAddToNumber( NUMA na, @Cast("l_int32") int index, @Cast("l_float32") float val );
public static native @Cast("l_int32") int numaGetMin( NUMA na, @Cast("l_float32*") FloatPointer pminval, @Cast("l_int32*") IntPointer piminloc );
public static native @Cast("l_int32") int numaGetMin( NUMA na, @Cast("l_float32*") FloatBuffer pminval, @Cast("l_int32*") IntBuffer piminloc );
public static native @Cast("l_int32") int numaGetMin( NUMA na, @Cast("l_float32*") float[] pminval, @Cast("l_int32*") int[] piminloc );
public static native @Cast("l_int32") int numaGetMax( NUMA na, @Cast("l_float32*") FloatPointer pmaxval, @Cast("l_int32*") IntPointer pimaxloc );
public static native @Cast("l_int32") int numaGetMax( NUMA na, @Cast("l_float32*") FloatBuffer pmaxval, @Cast("l_int32*") IntBuffer pimaxloc );
public static native @Cast("l_int32") int numaGetMax( NUMA na, @Cast("l_float32*") float[] pmaxval, @Cast("l_int32*") int[] pimaxloc );
public static native @Cast("l_int32") int numaGetSum( NUMA na, @Cast("l_float32*") FloatPointer psum );
public static native @Cast("l_int32") int numaGetSum( NUMA na, @Cast("l_float32*") FloatBuffer psum );
public static native @Cast("l_int32") int numaGetSum( NUMA na, @Cast("l_float32*") float[] psum );
public static native NUMA numaGetPartialSums( NUMA na );
public static native @Cast("l_int32") int numaGetSumOnInterval( NUMA na, @Cast("l_int32") int first, @Cast("l_int32") int last, @Cast("l_float32*") FloatPointer psum );
public static native @Cast("l_int32") int numaGetSumOnInterval( NUMA na, @Cast("l_int32") int first, @Cast("l_int32") int last, @Cast("l_float32*") FloatBuffer psum );
public static native @Cast("l_int32") int numaGetSumOnInterval( NUMA na, @Cast("l_int32") int first, @Cast("l_int32") int last, @Cast("l_float32*") float[] psum );
public static native @Cast("l_int32") int numaHasOnlyIntegers( NUMA na, @Cast("l_int32") int maxsamples, @Cast("l_int32*") IntPointer pallints );
public static native @Cast("l_int32") int numaHasOnlyIntegers( NUMA na, @Cast("l_int32") int maxsamples, @Cast("l_int32*") IntBuffer pallints );
public static native @Cast("l_int32") int numaHasOnlyIntegers( NUMA na, @Cast("l_int32") int maxsamples, @Cast("l_int32*") int[] pallints );
public static native NUMA numaSubsample( NUMA nas, @Cast("l_int32") int subfactor );
public static native NUMA numaMakeDelta( NUMA nas );
public static native NUMA numaMakeSequence( @Cast("l_float32") float startval, @Cast("l_float32") float increment, @Cast("l_int32") int size );
public static native NUMA numaMakeConstant( @Cast("l_float32") float val, @Cast("l_int32") int size );
public static native NUMA numaMakeAbsValue( NUMA nad, NUMA nas );
public static native NUMA numaAddBorder( NUMA nas, @Cast("l_int32") int left, @Cast("l_int32") int right, @Cast("l_float32") float val );
public static native NUMA numaAddSpecifiedBorder( NUMA nas, @Cast("l_int32") int left, @Cast("l_int32") int right, @Cast("l_int32") int type );
public static native NUMA numaRemoveBorder( NUMA nas, @Cast("l_int32") int left, @Cast("l_int32") int right );
public static native @Cast("l_int32") int numaGetNonzeroRange( NUMA na, @Cast("l_float32") float eps, @Cast("l_int32*") IntPointer pfirst, @Cast("l_int32*") IntPointer plast );
public static native @Cast("l_int32") int numaGetNonzeroRange( NUMA na, @Cast("l_float32") float eps, @Cast("l_int32*") IntBuffer pfirst, @Cast("l_int32*") IntBuffer plast );
public static native @Cast("l_int32") int numaGetNonzeroRange( NUMA na, @Cast("l_float32") float eps, @Cast("l_int32*") int[] pfirst, @Cast("l_int32*") int[] plast );
public static native @Cast("l_int32") int numaGetCountRelativeToZero( NUMA na, @Cast("l_int32") int type, @Cast("l_int32*") IntPointer pcount );
public static native @Cast("l_int32") int numaGetCountRelativeToZero( NUMA na, @Cast("l_int32") int type, @Cast("l_int32*") IntBuffer pcount );
public static native @Cast("l_int32") int numaGetCountRelativeToZero( NUMA na, @Cast("l_int32") int type, @Cast("l_int32*") int[] pcount );
public static native NUMA numaClipToInterval( NUMA nas, @Cast("l_int32") int first, @Cast("l_int32") int last );
public static native NUMA numaMakeThresholdIndicator( NUMA nas, @Cast("l_float32") float thresh, @Cast("l_int32") int type );
public static native NUMA numaUniformSampling( NUMA nas, @Cast("l_int32") int nsamp );
public static native NUMA numaReverse( NUMA nad, NUMA nas );
public static native NUMA numaLowPassIntervals( NUMA nas, @Cast("l_float32") float thresh, @Cast("l_float32") float maxn );
public static native NUMA numaThresholdEdges( NUMA nas, @Cast("l_float32") float thresh1, @Cast("l_float32") float thresh2, @Cast("l_float32") float maxn );
public static native @Cast("l_int32") int numaGetSpanValues( NUMA na, @Cast("l_int32") int span, @Cast("l_int32*") IntPointer pstart, @Cast("l_int32*") IntPointer pend );
public static native @Cast("l_int32") int numaGetSpanValues( NUMA na, @Cast("l_int32") int span, @Cast("l_int32*") IntBuffer pstart, @Cast("l_int32*") IntBuffer pend );
public static native @Cast("l_int32") int numaGetSpanValues( NUMA na, @Cast("l_int32") int span, @Cast("l_int32*") int[] pstart, @Cast("l_int32*") int[] pend );
public static native @Cast("l_int32") int numaGetEdgeValues( NUMA na, @Cast("l_int32") int edge, @Cast("l_int32*") IntPointer pstart, @Cast("l_int32*") IntPointer pend, @Cast("l_int32*") IntPointer psign );
public static native @Cast("l_int32") int numaGetEdgeValues( NUMA na, @Cast("l_int32") int edge, @Cast("l_int32*") IntBuffer pstart, @Cast("l_int32*") IntBuffer pend, @Cast("l_int32*") IntBuffer psign );
public static native @Cast("l_int32") int numaGetEdgeValues( NUMA na, @Cast("l_int32") int edge, @Cast("l_int32*") int[] pstart, @Cast("l_int32*") int[] pend, @Cast("l_int32*") int[] psign );
public static native @Cast("l_int32") int numaInterpolateEqxVal( @Cast("l_float32") float startx, @Cast("l_float32") float deltax, NUMA nay, @Cast("l_int32") int type, @Cast("l_float32") float xval, @Cast("l_float32*") FloatPointer pyval );
public static native @Cast("l_int32") int numaInterpolateEqxVal( @Cast("l_float32") float startx, @Cast("l_float32") float deltax, NUMA nay, @Cast("l_int32") int type, @Cast("l_float32") float xval, @Cast("l_float32*") FloatBuffer pyval );
public static native @Cast("l_int32") int numaInterpolateEqxVal( @Cast("l_float32") float startx, @Cast("l_float32") float deltax, NUMA nay, @Cast("l_int32") int type, @Cast("l_float32") float xval, @Cast("l_float32*") float[] pyval );
public static native @Cast("l_int32") int numaInterpolateArbxVal( NUMA nax, NUMA nay, @Cast("l_int32") int type, @Cast("l_float32") float xval, @Cast("l_float32*") FloatPointer pyval );
public static native @Cast("l_int32") int numaInterpolateArbxVal( NUMA nax, NUMA nay, @Cast("l_int32") int type, @Cast("l_float32") float xval, @Cast("l_float32*") FloatBuffer pyval );
public static native @Cast("l_int32") int numaInterpolateArbxVal( NUMA nax, NUMA nay, @Cast("l_int32") int type, @Cast("l_float32") float xval, @Cast("l_float32*") float[] pyval );
public static native @Cast("l_int32") int numaInterpolateEqxInterval( @Cast("l_float32") float startx, @Cast("l_float32") float deltax, NUMA nasy, @Cast("l_int32") int type, @Cast("l_float32") float x0, @Cast("l_float32") float x1, @Cast("l_int32") int npts, @Cast("NUMA**") PointerPointer pnax, @Cast("NUMA**") PointerPointer pnay );
public static native @Cast("l_int32") int numaInterpolateEqxInterval( @Cast("l_float32") float startx, @Cast("l_float32") float deltax, NUMA nasy, @Cast("l_int32") int type, @Cast("l_float32") float x0, @Cast("l_float32") float x1, @Cast("l_int32") int npts, @ByPtrPtr NUMA pnax, @ByPtrPtr NUMA pnay );
public static native @Cast("l_int32") int numaInterpolateArbxInterval( NUMA nax, NUMA nay, @Cast("l_int32") int type, @Cast("l_float32") float x0, @Cast("l_float32") float x1, @Cast("l_int32") int npts, @Cast("NUMA**") PointerPointer pnadx, @Cast("NUMA**") PointerPointer pnady );
public static native @Cast("l_int32") int numaInterpolateArbxInterval( NUMA nax, NUMA nay, @Cast("l_int32") int type, @Cast("l_float32") float x0, @Cast("l_float32") float x1, @Cast("l_int32") int npts, @ByPtrPtr NUMA pnadx, @ByPtrPtr NUMA pnady );
public static native @Cast("l_int32") int numaFitMax( NUMA na, @Cast("l_float32*") FloatPointer pmaxval, NUMA naloc, @Cast("l_float32*") FloatPointer pmaxloc );
public static native @Cast("l_int32") int numaFitMax( NUMA na, @Cast("l_float32*") FloatBuffer pmaxval, NUMA naloc, @Cast("l_float32*") FloatBuffer pmaxloc );
public static native @Cast("l_int32") int numaFitMax( NUMA na, @Cast("l_float32*") float[] pmaxval, NUMA naloc, @Cast("l_float32*") float[] pmaxloc );
public static native @Cast("l_int32") int numaDifferentiateInterval( NUMA nax, NUMA nay, @Cast("l_float32") float x0, @Cast("l_float32") float x1, @Cast("l_int32") int npts, @Cast("NUMA**") PointerPointer pnadx, @Cast("NUMA**") PointerPointer pnady );
public static native @Cast("l_int32") int numaDifferentiateInterval( NUMA nax, NUMA nay, @Cast("l_float32") float x0, @Cast("l_float32") float x1, @Cast("l_int32") int npts, @ByPtrPtr NUMA pnadx, @ByPtrPtr NUMA pnady );
public static native @Cast("l_int32") int numaIntegrateInterval( NUMA nax, NUMA nay, @Cast("l_float32") float x0, @Cast("l_float32") float x1, @Cast("l_int32") int npts, @Cast("l_float32*") FloatPointer psum );
public static native @Cast("l_int32") int numaIntegrateInterval( NUMA nax, NUMA nay, @Cast("l_float32") float x0, @Cast("l_float32") float x1, @Cast("l_int32") int npts, @Cast("l_float32*") FloatBuffer psum );
public static native @Cast("l_int32") int numaIntegrateInterval( NUMA nax, NUMA nay, @Cast("l_float32") float x0, @Cast("l_float32") float x1, @Cast("l_int32") int npts, @Cast("l_float32*") float[] psum );
public static native @Cast("l_int32") int numaSortGeneral( NUMA na, @Cast("NUMA**") PointerPointer pnasort, @Cast("NUMA**") PointerPointer pnaindex, @Cast("NUMA**") PointerPointer pnainvert, @Cast("l_int32") int sortorder, @Cast("l_int32") int sorttype );
public static native @Cast("l_int32") int numaSortGeneral( NUMA na, @ByPtrPtr NUMA pnasort, @ByPtrPtr NUMA pnaindex, @ByPtrPtr NUMA pnainvert, @Cast("l_int32") int sortorder, @Cast("l_int32") int sorttype );
public static native NUMA numaSortAutoSelect( NUMA nas, @Cast("l_int32") int sortorder );
public static native NUMA numaSortIndexAutoSelect( NUMA nas, @Cast("l_int32") int sortorder );
public static native @Cast("l_int32") int numaChooseSortType( NUMA nas );
public static native NUMA numaSort( NUMA naout, NUMA nain, @Cast("l_int32") int sortorder );
public static native NUMA numaBinSort( NUMA nas, @Cast("l_int32") int sortorder );
public static native NUMA numaGetSortIndex( NUMA na, @Cast("l_int32") int sortorder );
public static native NUMA numaGetBinSortIndex( NUMA nas, @Cast("l_int32") int sortorder );
public static native NUMA numaSortByIndex( NUMA nas, NUMA naindex );
public static native @Cast("l_int32") int numaIsSorted( NUMA nas, @Cast("l_int32") int sortorder, @Cast("l_int32*") IntPointer psorted );
public static native @Cast("l_int32") int numaIsSorted( NUMA nas, @Cast("l_int32") int sortorder, @Cast("l_int32*") IntBuffer psorted );
public static native @Cast("l_int32") int numaIsSorted( NUMA nas, @Cast("l_int32") int sortorder, @Cast("l_int32*") int[] psorted );
public static native @Cast("l_int32") int numaSortPair( NUMA nax, NUMA nay, @Cast("l_int32") int sortorder, @Cast("NUMA**") PointerPointer pnasx, @Cast("NUMA**") PointerPointer pnasy );
public static native @Cast("l_int32") int numaSortPair( NUMA nax, NUMA nay, @Cast("l_int32") int sortorder, @ByPtrPtr NUMA pnasx, @ByPtrPtr NUMA pnasy );
public static native NUMA numaInvertMap( NUMA nas );
public static native NUMA numaPseudorandomSequence( @Cast("l_int32") int size, @Cast("l_int32") int seed );
public static native NUMA numaRandomPermutation( NUMA nas, @Cast("l_int32") int seed );
public static native @Cast("l_int32") int numaGetRankValue( NUMA na, @Cast("l_float32") float fract, NUMA nasort, @Cast("l_int32") int usebins, @Cast("l_float32*") FloatPointer pval );
public static native @Cast("l_int32") int numaGetRankValue( NUMA na, @Cast("l_float32") float fract, NUMA nasort, @Cast("l_int32") int usebins, @Cast("l_float32*") FloatBuffer pval );
public static native @Cast("l_int32") int numaGetRankValue( NUMA na, @Cast("l_float32") float fract, NUMA nasort, @Cast("l_int32") int usebins, @Cast("l_float32*") float[] pval );
public static native @Cast("l_int32") int numaGetMedian( NUMA na, @Cast("l_float32*") FloatPointer pval );
public static native @Cast("l_int32") int numaGetMedian( NUMA na, @Cast("l_float32*") FloatBuffer pval );
public static native @Cast("l_int32") int numaGetMedian( NUMA na, @Cast("l_float32*") float[] pval );
public static native @Cast("l_int32") int numaGetBinnedMedian( NUMA na, @Cast("l_int32*") IntPointer pval );
public static native @Cast("l_int32") int numaGetBinnedMedian( NUMA na, @Cast("l_int32*") IntBuffer pval );
public static native @Cast("l_int32") int numaGetBinnedMedian( NUMA na, @Cast("l_int32*") int[] pval );
public static native @Cast("l_int32") int numaGetMode( NUMA na, @Cast("l_float32*") FloatPointer pval, @Cast("l_int32*") IntPointer pcount );
public static native @Cast("l_int32") int numaGetMode( NUMA na, @Cast("l_float32*") FloatBuffer pval, @Cast("l_int32*") IntBuffer pcount );
public static native @Cast("l_int32") int numaGetMode( NUMA na, @Cast("l_float32*") float[] pval, @Cast("l_int32*") int[] pcount );
public static native @Cast("l_int32") int numaGetMedianVariation( NUMA na, @Cast("l_float32*") FloatPointer pmedval, @Cast("l_float32*") FloatPointer pmedvar );
public static native @Cast("l_int32") int numaGetMedianVariation( NUMA na, @Cast("l_float32*") FloatBuffer pmedval, @Cast("l_float32*") FloatBuffer pmedvar );
public static native @Cast("l_int32") int numaGetMedianVariation( NUMA na, @Cast("l_float32*") float[] pmedval, @Cast("l_float32*") float[] pmedvar );
public static native @Cast("l_int32") int numaJoin( NUMA nad, NUMA nas, @Cast("l_int32") int istart, @Cast("l_int32") int iend );
public static native @Cast("l_int32") int numaaJoin( NUMAA naad, NUMAA naas, @Cast("l_int32") int istart, @Cast("l_int32") int iend );
public static native NUMA numaaFlattenToNuma( NUMAA naa );
public static native NUMA numaErode( NUMA nas, @Cast("l_int32") int size );
public static native NUMA numaDilate( NUMA nas, @Cast("l_int32") int size );
public static native NUMA numaOpen( NUMA nas, @Cast("l_int32") int size );
public static native NUMA numaClose( NUMA nas, @Cast("l_int32") int size );
public static native NUMA numaTransform( NUMA nas, @Cast("l_float32") float shift, @Cast("l_float32") float scale );
public static native @Cast("l_int32") int numaWindowedStats( NUMA nas, @Cast("l_int32") int wc, @Cast("NUMA**") PointerPointer pnam, @Cast("NUMA**") PointerPointer pnams, @Cast("NUMA**") PointerPointer pnav, @Cast("NUMA**") PointerPointer pnarv );
public static native @Cast("l_int32") int numaWindowedStats( NUMA nas, @Cast("l_int32") int wc, @ByPtrPtr NUMA pnam, @ByPtrPtr NUMA pnams, @ByPtrPtr NUMA pnav, @ByPtrPtr NUMA pnarv );
public static native NUMA numaWindowedMean( NUMA nas, @Cast("l_int32") int wc );
public static native NUMA numaWindowedMeanSquare( NUMA nas, @Cast("l_int32") int wc );
public static native @Cast("l_int32") int numaWindowedVariance( NUMA nam, NUMA nams, @Cast("NUMA**") PointerPointer pnav, @Cast("NUMA**") PointerPointer pnarv );
public static native @Cast("l_int32") int numaWindowedVariance( NUMA nam, NUMA nams, @ByPtrPtr NUMA pnav, @ByPtrPtr NUMA pnarv );
public static native NUMA numaWindowedMedian( NUMA nas, @Cast("l_int32") int halfwin );
public static native NUMA numaConvertToInt( NUMA nas );
public static native NUMA numaMakeHistogram( NUMA na, @Cast("l_int32") int maxbins, @Cast("l_int32*") IntPointer pbinsize, @Cast("l_int32*") IntPointer pbinstart );
public static native NUMA numaMakeHistogram( NUMA na, @Cast("l_int32") int maxbins, @Cast("l_int32*") IntBuffer pbinsize, @Cast("l_int32*") IntBuffer pbinstart );
public static native NUMA numaMakeHistogram( NUMA na, @Cast("l_int32") int maxbins, @Cast("l_int32*") int[] pbinsize, @Cast("l_int32*") int[] pbinstart );
public static native NUMA numaMakeHistogramAuto( NUMA na, @Cast("l_int32") int maxbins );
public static native NUMA numaMakeHistogramClipped( NUMA na, @Cast("l_float32") float binsize, @Cast("l_float32") float maxsize );
public static native NUMA numaRebinHistogram( NUMA nas, @Cast("l_int32") int newsize );
public static native NUMA numaNormalizeHistogram( NUMA nas, @Cast("l_float32") float tsum );
public static native @Cast("l_int32") int numaGetStatsUsingHistogram( NUMA na, @Cast("l_int32") int maxbins, @Cast("l_float32*") FloatPointer pmin, @Cast("l_float32*") FloatPointer pmax, @Cast("l_float32*") FloatPointer pmean, @Cast("l_float32*") FloatPointer pvariance, @Cast("l_float32*") FloatPointer pmedian, @Cast("l_float32") float rank, @Cast("l_float32*") FloatPointer prval, @Cast("NUMA**") PointerPointer phisto );
public static native @Cast("l_int32") int numaGetStatsUsingHistogram( NUMA na, @Cast("l_int32") int maxbins, @Cast("l_float32*") FloatPointer pmin, @Cast("l_float32*") FloatPointer pmax, @Cast("l_float32*") FloatPointer pmean, @Cast("l_float32*") FloatPointer pvariance, @Cast("l_float32*") FloatPointer pmedian, @Cast("l_float32") float rank, @Cast("l_float32*") FloatPointer prval, @ByPtrPtr NUMA phisto );
public static native @Cast("l_int32") int numaGetStatsUsingHistogram( NUMA na, @Cast("l_int32") int maxbins, @Cast("l_float32*") FloatBuffer pmin, @Cast("l_float32*") FloatBuffer pmax, @Cast("l_float32*") FloatBuffer pmean, @Cast("l_float32*") FloatBuffer pvariance, @Cast("l_float32*") FloatBuffer pmedian, @Cast("l_float32") float rank, @Cast("l_float32*") FloatBuffer prval, @ByPtrPtr NUMA phisto );
public static native @Cast("l_int32") int numaGetStatsUsingHistogram( NUMA na, @Cast("l_int32") int maxbins, @Cast("l_float32*") float[] pmin, @Cast("l_float32*") float[] pmax, @Cast("l_float32*") float[] pmean, @Cast("l_float32*") float[] pvariance, @Cast("l_float32*") float[] pmedian, @Cast("l_float32") float rank, @Cast("l_float32*") float[] prval, @ByPtrPtr NUMA phisto );
public static native @Cast("l_int32") int numaGetHistogramStats( NUMA nahisto, @Cast("l_float32") float startx, @Cast("l_float32") float deltax, @Cast("l_float32*") FloatPointer pxmean, @Cast("l_float32*") FloatPointer pxmedian, @Cast("l_float32*") FloatPointer pxmode, @Cast("l_float32*") FloatPointer pxvariance );
public static native @Cast("l_int32") int numaGetHistogramStats( NUMA nahisto, @Cast("l_float32") float startx, @Cast("l_float32") float deltax, @Cast("l_float32*") FloatBuffer pxmean, @Cast("l_float32*") FloatBuffer pxmedian, @Cast("l_float32*") FloatBuffer pxmode, @Cast("l_float32*") FloatBuffer pxvariance );
public static native @Cast("l_int32") int numaGetHistogramStats( NUMA nahisto, @Cast("l_float32") float startx, @Cast("l_float32") float deltax, @Cast("l_float32*") float[] pxmean, @Cast("l_float32*") float[] pxmedian, @Cast("l_float32*") float[] pxmode, @Cast("l_float32*") float[] pxvariance );
public static native @Cast("l_int32") int numaGetHistogramStatsOnInterval( NUMA nahisto, @Cast("l_float32") float startx, @Cast("l_float32") float deltax, @Cast("l_int32") int ifirst, @Cast("l_int32") int ilast, @Cast("l_float32*") FloatPointer pxmean, @Cast("l_float32*") FloatPointer pxmedian, @Cast("l_float32*") FloatPointer pxmode, @Cast("l_float32*") FloatPointer pxvariance );
public static native @Cast("l_int32") int numaGetHistogramStatsOnInterval( NUMA nahisto, @Cast("l_float32") float startx, @Cast("l_float32") float deltax, @Cast("l_int32") int ifirst, @Cast("l_int32") int ilast, @Cast("l_float32*") FloatBuffer pxmean, @Cast("l_float32*") FloatBuffer pxmedian, @Cast("l_float32*") FloatBuffer pxmode, @Cast("l_float32*") FloatBuffer pxvariance );
public static native @Cast("l_int32") int numaGetHistogramStatsOnInterval( NUMA nahisto, @Cast("l_float32") float startx, @Cast("l_float32") float deltax, @Cast("l_int32") int ifirst, @Cast("l_int32") int ilast, @Cast("l_float32*") float[] pxmean, @Cast("l_float32*") float[] pxmedian, @Cast("l_float32*") float[] pxmode, @Cast("l_float32*") float[] pxvariance );
public static native @Cast("l_int32") int numaMakeRankFromHistogram( @Cast("l_float32") float startx, @Cast("l_float32") float deltax, NUMA nasy, @Cast("l_int32") int npts, @Cast("NUMA**") PointerPointer pnax, @Cast("NUMA**") PointerPointer pnay );
public static native @Cast("l_int32") int numaMakeRankFromHistogram( @Cast("l_float32") float startx, @Cast("l_float32") float deltax, NUMA nasy, @Cast("l_int32") int npts, @ByPtrPtr NUMA pnax, @ByPtrPtr NUMA pnay );
public static native @Cast("l_int32") int numaHistogramGetRankFromVal( NUMA na, @Cast("l_float32") float rval, @Cast("l_float32*") FloatPointer prank );
public static native @Cast("l_int32") int numaHistogramGetRankFromVal( NUMA na, @Cast("l_float32") float rval, @Cast("l_float32*") FloatBuffer prank );
public static native @Cast("l_int32") int numaHistogramGetRankFromVal( NUMA na, @Cast("l_float32") float rval, @Cast("l_float32*") float[] prank );
public static native @Cast("l_int32") int numaHistogramGetValFromRank( NUMA na, @Cast("l_float32") float rank, @Cast("l_float32*") FloatPointer prval );
public static native @Cast("l_int32") int numaHistogramGetValFromRank( NUMA na, @Cast("l_float32") float rank, @Cast("l_float32*") FloatBuffer prval );
public static native @Cast("l_int32") int numaHistogramGetValFromRank( NUMA na, @Cast("l_float32") float rank, @Cast("l_float32*") float[] prval );
public static native @Cast("l_int32") int numaDiscretizeRankAndIntensity( NUMA na, @Cast("l_int32") int nbins, @Cast("NUMA**") PointerPointer pnarbin, @Cast("NUMA**") PointerPointer pnam, @Cast("NUMA**") PointerPointer pnar, @Cast("NUMA**") PointerPointer pnabb );
public static native @Cast("l_int32") int numaDiscretizeRankAndIntensity( NUMA na, @Cast("l_int32") int nbins, @ByPtrPtr NUMA pnarbin, @ByPtrPtr NUMA pnam, @ByPtrPtr NUMA pnar, @ByPtrPtr NUMA pnabb );
public static native @Cast("l_int32") int numaGetRankBinValues( NUMA na, @Cast("l_int32") int nbins, @Cast("NUMA**") PointerPointer pnarbin, @Cast("NUMA**") PointerPointer pnam );
public static native @Cast("l_int32") int numaGetRankBinValues( NUMA na, @Cast("l_int32") int nbins, @ByPtrPtr NUMA pnarbin, @ByPtrPtr NUMA pnam );
public static native @Cast("l_int32") int numaSplitDistribution( NUMA na, @Cast("l_float32") float scorefract, @Cast("l_int32*") IntPointer psplitindex, @Cast("l_float32*") FloatPointer pave1, @Cast("l_float32*") FloatPointer pave2, @Cast("l_float32*") FloatPointer pnum1, @Cast("l_float32*") FloatPointer pnum2, @Cast("NUMA**") PointerPointer pnascore );
public static native @Cast("l_int32") int numaSplitDistribution( NUMA na, @Cast("l_float32") float scorefract, @Cast("l_int32*") IntPointer psplitindex, @Cast("l_float32*") FloatPointer pave1, @Cast("l_float32*") FloatPointer pave2, @Cast("l_float32*") FloatPointer pnum1, @Cast("l_float32*") FloatPointer pnum2, @ByPtrPtr NUMA pnascore );
public static native @Cast("l_int32") int numaSplitDistribution( NUMA na, @Cast("l_float32") float scorefract, @Cast("l_int32*") IntBuffer psplitindex, @Cast("l_float32*") FloatBuffer pave1, @Cast("l_float32*") FloatBuffer pave2, @Cast("l_float32*") FloatBuffer pnum1, @Cast("l_float32*") FloatBuffer pnum2, @ByPtrPtr NUMA pnascore );
public static native @Cast("l_int32") int numaSplitDistribution( NUMA na, @Cast("l_float32") float scorefract, @Cast("l_int32*") int[] psplitindex, @Cast("l_float32*") float[] pave1, @Cast("l_float32*") float[] pave2, @Cast("l_float32*") float[] pnum1, @Cast("l_float32*") float[] pnum2, @ByPtrPtr NUMA pnascore );
public static native @Cast("l_int32") int numaEarthMoverDistance( NUMA na1, NUMA na2, @Cast("l_float32*") FloatPointer pdist );
public static native @Cast("l_int32") int numaEarthMoverDistance( NUMA na1, NUMA na2, @Cast("l_float32*") FloatBuffer pdist );
public static native @Cast("l_int32") int numaEarthMoverDistance( NUMA na1, NUMA na2, @Cast("l_float32*") float[] pdist );
public static native NUMA numaFindPeaks( NUMA nas, @Cast("l_int32") int nmax, @Cast("l_float32") float fract1, @Cast("l_float32") float fract2 );
public static native NUMA numaFindExtrema( NUMA nas, @Cast("l_float32") float delta );
public static native @Cast("l_int32") int numaCountReversals( NUMA nas, @Cast("l_float32") float minreversal, @Cast("l_int32*") IntPointer pnr, @Cast("l_float32*") FloatPointer pnrpl );
public static native @Cast("l_int32") int numaCountReversals( NUMA nas, @Cast("l_float32") float minreversal, @Cast("l_int32*") IntBuffer pnr, @Cast("l_float32*") FloatBuffer pnrpl );
public static native @Cast("l_int32") int numaCountReversals( NUMA nas, @Cast("l_float32") float minreversal, @Cast("l_int32*") int[] pnr, @Cast("l_float32*") float[] pnrpl );
public static native @Cast("l_int32") int numaSelectCrossingThreshold( NUMA nax, NUMA nay, @Cast("l_float32") float estthresh, @Cast("l_float32*") FloatPointer pbestthresh );
public static native @Cast("l_int32") int numaSelectCrossingThreshold( NUMA nax, NUMA nay, @Cast("l_float32") float estthresh, @Cast("l_float32*") FloatBuffer pbestthresh );
public static native @Cast("l_int32") int numaSelectCrossingThreshold( NUMA nax, NUMA nay, @Cast("l_float32") float estthresh, @Cast("l_float32*") float[] pbestthresh );
public static native NUMA numaCrossingsByThreshold( NUMA nax, NUMA nay, @Cast("l_float32") float thresh );
public static native NUMA numaCrossingsByPeaks( NUMA nax, NUMA nay, @Cast("l_float32") float delta );
public static native @Cast("l_int32") int numaEvalBestHaarParameters( NUMA nas, @Cast("l_float32") float relweight, @Cast("l_int32") int nwidth, @Cast("l_int32") int nshift, @Cast("l_float32") float minwidth, @Cast("l_float32") float maxwidth, @Cast("l_float32*") FloatPointer pbestwidth, @Cast("l_float32*") FloatPointer pbestshift, @Cast("l_float32*") FloatPointer pbestscore );
public static native @Cast("l_int32") int numaEvalBestHaarParameters( NUMA nas, @Cast("l_float32") float relweight, @Cast("l_int32") int nwidth, @Cast("l_int32") int nshift, @Cast("l_float32") float minwidth, @Cast("l_float32") float maxwidth, @Cast("l_float32*") FloatBuffer pbestwidth, @Cast("l_float32*") FloatBuffer pbestshift, @Cast("l_float32*") FloatBuffer pbestscore );
public static native @Cast("l_int32") int numaEvalBestHaarParameters( NUMA nas, @Cast("l_float32") float relweight, @Cast("l_int32") int nwidth, @Cast("l_int32") int nshift, @Cast("l_float32") float minwidth, @Cast("l_float32") float maxwidth, @Cast("l_float32*") float[] pbestwidth, @Cast("l_float32*") float[] pbestshift, @Cast("l_float32*") float[] pbestscore );
public static native @Cast("l_int32") int numaEvalHaarSum( NUMA nas, @Cast("l_float32") float width, @Cast("l_float32") float shift, @Cast("l_float32") float relweight, @Cast("l_float32*") FloatPointer pscore );
public static native @Cast("l_int32") int numaEvalHaarSum( NUMA nas, @Cast("l_float32") float width, @Cast("l_float32") float shift, @Cast("l_float32") float relweight, @Cast("l_float32*") FloatBuffer pscore );
public static native @Cast("l_int32") int numaEvalHaarSum( NUMA nas, @Cast("l_float32") float width, @Cast("l_float32") float shift, @Cast("l_float32") float relweight, @Cast("l_float32*") float[] pscore );
public static native @Cast("l_int32") int pixGetRegionsBinary( PIX pixs, @Cast("PIX**") PointerPointer ppixhm, @Cast("PIX**") PointerPointer ppixtm, @Cast("PIX**") PointerPointer ppixtb, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int pixGetRegionsBinary( PIX pixs, @ByPtrPtr PIX ppixhm, @ByPtrPtr PIX ppixtm, @ByPtrPtr PIX ppixtb, @Cast("l_int32") int debug );
public static native PIX pixGenHalftoneMask( PIX pixs, @Cast("PIX**") PointerPointer ppixtext, @Cast("l_int32*") IntPointer phtfound, @Cast("l_int32") int debug );
public static native PIX pixGenHalftoneMask( PIX pixs, @ByPtrPtr PIX ppixtext, @Cast("l_int32*") IntPointer phtfound, @Cast("l_int32") int debug );
public static native PIX pixGenHalftoneMask( PIX pixs, @ByPtrPtr PIX ppixtext, @Cast("l_int32*") IntBuffer phtfound, @Cast("l_int32") int debug );
public static native PIX pixGenHalftoneMask( PIX pixs, @ByPtrPtr PIX ppixtext, @Cast("l_int32*") int[] phtfound, @Cast("l_int32") int debug );
public static native PIX pixGenTextlineMask( PIX pixs, @Cast("PIX**") PointerPointer ppixvws, @Cast("l_int32*") IntPointer ptlfound, @Cast("l_int32") int debug );
public static native PIX pixGenTextlineMask( PIX pixs, @ByPtrPtr PIX ppixvws, @Cast("l_int32*") IntPointer ptlfound, @Cast("l_int32") int debug );
public static native PIX pixGenTextlineMask( PIX pixs, @ByPtrPtr PIX ppixvws, @Cast("l_int32*") IntBuffer ptlfound, @Cast("l_int32") int debug );
public static native PIX pixGenTextlineMask( PIX pixs, @ByPtrPtr PIX ppixvws, @Cast("l_int32*") int[] ptlfound, @Cast("l_int32") int debug );
public static native PIX pixGenTextblockMask( PIX pixs, PIX pixvws, @Cast("l_int32") int debug );
public static native BOX pixFindPageForeground( PIX pixs, @Cast("l_int32") int threshold, @Cast("l_int32") int mindist, @Cast("l_int32") int erasedist, @Cast("l_int32") int pagenum, @Cast("l_int32") int showmorph, @Cast("l_int32") int display, @Cast("const char*") BytePointer pdfdir );
public static native BOX pixFindPageForeground( PIX pixs, @Cast("l_int32") int threshold, @Cast("l_int32") int mindist, @Cast("l_int32") int erasedist, @Cast("l_int32") int pagenum, @Cast("l_int32") int showmorph, @Cast("l_int32") int display, String pdfdir );
public static native @Cast("l_int32") int pixSplitIntoCharacters( PIX pixs, @Cast("l_int32") int minw, @Cast("l_int32") int minh, @Cast("BOXA**") PointerPointer pboxa, @Cast("PIXA**") PointerPointer ppixa, @Cast("PIX**") PointerPointer ppixdebug );
public static native @Cast("l_int32") int pixSplitIntoCharacters( PIX pixs, @Cast("l_int32") int minw, @Cast("l_int32") int minh, @ByPtrPtr BOXA pboxa, @ByPtrPtr PIXA ppixa, @ByPtrPtr PIX ppixdebug );
public static native BOXA pixSplitComponentWithProfile( PIX pixs, @Cast("l_int32") int delta, @Cast("l_int32") int mindel, @Cast("PIX**") PointerPointer ppixdebug );
public static native BOXA pixSplitComponentWithProfile( PIX pixs, @Cast("l_int32") int delta, @Cast("l_int32") int mindel, @ByPtrPtr PIX ppixdebug );
public static native @Cast("l_int32") int pixSetSelectCmap( PIX pixs, BOX box, @Cast("l_int32") int sindex, @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval );
public static native @Cast("l_int32") int pixColorGrayRegionsCmap( PIX pixs, BOXA boxa, @Cast("l_int32") int type, @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval );
public static native @Cast("l_int32") int pixColorGrayCmap( PIX pixs, BOX box, @Cast("l_int32") int type, @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval );
public static native @Cast("l_int32") int pixColorGrayMaskedCmap( PIX pixs, PIX pixm, @Cast("l_int32") int type, @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval );
public static native @Cast("l_int32") int addColorizedGrayToCmap( PIXCMAP cmap, @Cast("l_int32") int type, @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("NUMA**") PointerPointer pna );
public static native @Cast("l_int32") int addColorizedGrayToCmap( PIXCMAP cmap, @Cast("l_int32") int type, @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @ByPtrPtr NUMA pna );
public static native @Cast("l_int32") int pixSetSelectMaskedCmap( PIX pixs, PIX pixm, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int sindex, @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval );
public static native @Cast("l_int32") int pixSetMaskedCmap( PIX pixs, PIX pixm, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval );
public static native @Cast("char*") BytePointer parseForProtos( @Cast("const char*") BytePointer filein, @Cast("const char*") BytePointer prestring );
public static native @Cast("char*") ByteBuffer parseForProtos( String filein, String prestring );
public static native BOXA boxaGetWhiteblocks( BOXA boxas, BOX box, @Cast("l_int32") int sortflag, @Cast("l_int32") int maxboxes, @Cast("l_float32") float maxoverlap, @Cast("l_int32") int maxperim, @Cast("l_float32") float fract, @Cast("l_int32") int maxpops );
public static native BOXA boxaPruneSortedOnOverlap( BOXA boxas, @Cast("l_float32") float maxoverlap );
public static native @Cast("l_int32") int convertFilesToPdf( @Cast("const char*") BytePointer dirname, @Cast("const char*") BytePointer substr, @Cast("l_int32") int res, @Cast("l_float32") float scalefactor, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("const char*") BytePointer title, @Cast("const char*") BytePointer fileout );
public static native @Cast("l_int32") int convertFilesToPdf( String dirname, String substr, @Cast("l_int32") int res, @Cast("l_float32") float scalefactor, @Cast("l_int32") int type, @Cast("l_int32") int quality, String title, String fileout );
public static native @Cast("l_int32") int saConvertFilesToPdf( SARRAY sa, @Cast("l_int32") int res, @Cast("l_float32") float scalefactor, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("const char*") BytePointer title, @Cast("const char*") BytePointer fileout );
public static native @Cast("l_int32") int saConvertFilesToPdf( SARRAY sa, @Cast("l_int32") int res, @Cast("l_float32") float scalefactor, @Cast("l_int32") int type, @Cast("l_int32") int quality, String title, String fileout );
public static native @Cast("l_int32") int saConvertFilesToPdfData( SARRAY sa, @Cast("l_int32") int res, @Cast("l_float32") float scalefactor, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("const char*") BytePointer title, @Cast("l_uint8**") PointerPointer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int saConvertFilesToPdfData( SARRAY sa, @Cast("l_int32") int res, @Cast("l_float32") float scalefactor, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("const char*") BytePointer title, @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int saConvertFilesToPdfData( SARRAY sa, @Cast("l_int32") int res, @Cast("l_float32") float scalefactor, @Cast("l_int32") int type, @Cast("l_int32") int quality, String title, @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int saConvertFilesToPdfData( SARRAY sa, @Cast("l_int32") int res, @Cast("l_float32") float scalefactor, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("const char*") BytePointer title, @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int saConvertFilesToPdfData( SARRAY sa, @Cast("l_int32") int res, @Cast("l_float32") float scalefactor, @Cast("l_int32") int type, @Cast("l_int32") int quality, String title, @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int saConvertFilesToPdfData( SARRAY sa, @Cast("l_int32") int res, @Cast("l_float32") float scalefactor, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("const char*") BytePointer title, @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int saConvertFilesToPdfData( SARRAY sa, @Cast("l_int32") int res, @Cast("l_float32") float scalefactor, @Cast("l_int32") int type, @Cast("l_int32") int quality, String title, @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int selectDefaultPdfEncoding( PIX pix, @Cast("l_int32*") IntPointer ptype );
public static native @Cast("l_int32") int selectDefaultPdfEncoding( PIX pix, @Cast("l_int32*") IntBuffer ptype );
public static native @Cast("l_int32") int selectDefaultPdfEncoding( PIX pix, @Cast("l_int32*") int[] ptype );
public static native @Cast("l_int32") int convertUnscaledFilesToPdf( @Cast("const char*") BytePointer dirname, @Cast("const char*") BytePointer substr, @Cast("const char*") BytePointer title, @Cast("const char*") BytePointer fileout );
public static native @Cast("l_int32") int convertUnscaledFilesToPdf( String dirname, String substr, String title, String fileout );
public static native @Cast("l_int32") int saConvertUnscaledFilesToPdf( SARRAY sa, @Cast("const char*") BytePointer title, @Cast("const char*") BytePointer fileout );
public static native @Cast("l_int32") int saConvertUnscaledFilesToPdf( SARRAY sa, String title, String fileout );
public static native @Cast("l_int32") int saConvertUnscaledFilesToPdfData( SARRAY sa, @Cast("const char*") BytePointer title, @Cast("l_uint8**") PointerPointer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int saConvertUnscaledFilesToPdfData( SARRAY sa, @Cast("const char*") BytePointer title, @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int saConvertUnscaledFilesToPdfData( SARRAY sa, String title, @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int saConvertUnscaledFilesToPdfData( SARRAY sa, @Cast("const char*") BytePointer title, @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int saConvertUnscaledFilesToPdfData( SARRAY sa, String title, @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int saConvertUnscaledFilesToPdfData( SARRAY sa, @Cast("const char*") BytePointer title, @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int saConvertUnscaledFilesToPdfData( SARRAY sa, String title, @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int convertUnscaledToPdfData( @Cast("const char*") BytePointer fname, @Cast("const char*") BytePointer title, @Cast("l_uint8**") PointerPointer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int convertUnscaledToPdfData( @Cast("const char*") BytePointer fname, @Cast("const char*") BytePointer title, @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int convertUnscaledToPdfData( String fname, String title, @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int convertUnscaledToPdfData( @Cast("const char*") BytePointer fname, @Cast("const char*") BytePointer title, @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int convertUnscaledToPdfData( String fname, String title, @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int convertUnscaledToPdfData( @Cast("const char*") BytePointer fname, @Cast("const char*") BytePointer title, @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int convertUnscaledToPdfData( String fname, String title, @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int pixaConvertToPdf( PIXA pixa, @Cast("l_int32") int res, @Cast("l_float32") float scalefactor, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("const char*") BytePointer title, @Cast("const char*") BytePointer fileout );
public static native @Cast("l_int32") int pixaConvertToPdf( PIXA pixa, @Cast("l_int32") int res, @Cast("l_float32") float scalefactor, @Cast("l_int32") int type, @Cast("l_int32") int quality, String title, String fileout );
public static native @Cast("l_int32") int pixaConvertToPdfData( PIXA pixa, @Cast("l_int32") int res, @Cast("l_float32") float scalefactor, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("const char*") BytePointer title, @Cast("l_uint8**") PointerPointer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int pixaConvertToPdfData( PIXA pixa, @Cast("l_int32") int res, @Cast("l_float32") float scalefactor, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("const char*") BytePointer title, @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int pixaConvertToPdfData( PIXA pixa, @Cast("l_int32") int res, @Cast("l_float32") float scalefactor, @Cast("l_int32") int type, @Cast("l_int32") int quality, String title, @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int pixaConvertToPdfData( PIXA pixa, @Cast("l_int32") int res, @Cast("l_float32") float scalefactor, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("const char*") BytePointer title, @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int pixaConvertToPdfData( PIXA pixa, @Cast("l_int32") int res, @Cast("l_float32") float scalefactor, @Cast("l_int32") int type, @Cast("l_int32") int quality, String title, @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int pixaConvertToPdfData( PIXA pixa, @Cast("l_int32") int res, @Cast("l_float32") float scalefactor, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("const char*") BytePointer title, @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int pixaConvertToPdfData( PIXA pixa, @Cast("l_int32") int res, @Cast("l_float32") float scalefactor, @Cast("l_int32") int type, @Cast("l_int32") int quality, String title, @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int convertToPdf( @Cast("const char*") BytePointer filein, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("const char*") BytePointer fileout, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, @Cast("const char*") BytePointer title, @Cast("L_PDF_DATA**") PointerPointer plpd, @Cast("l_int32") int position );
public static native @Cast("l_int32") int convertToPdf( @Cast("const char*") BytePointer filein, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("const char*") BytePointer fileout, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, @Cast("const char*") BytePointer title, @ByPtrPtr L_PDF_DATA plpd, @Cast("l_int32") int position );
public static native @Cast("l_int32") int convertToPdf( String filein, @Cast("l_int32") int type, @Cast("l_int32") int quality, String fileout, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, String title, @ByPtrPtr L_PDF_DATA plpd, @Cast("l_int32") int position );
public static native @Cast("l_int32") int convertImageDataToPdf( @Cast("l_uint8*") BytePointer imdata, @Cast("size_t") long size, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("const char*") BytePointer fileout, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, @Cast("const char*") BytePointer title, @Cast("L_PDF_DATA**") PointerPointer plpd, @Cast("l_int32") int position );
public static native @Cast("l_int32") int convertImageDataToPdf( @Cast("l_uint8*") BytePointer imdata, @Cast("size_t") long size, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("const char*") BytePointer fileout, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, @Cast("const char*") BytePointer title, @ByPtrPtr L_PDF_DATA plpd, @Cast("l_int32") int position );
public static native @Cast("l_int32") int convertImageDataToPdf( @Cast("l_uint8*") ByteBuffer imdata, @Cast("size_t") long size, @Cast("l_int32") int type, @Cast("l_int32") int quality, String fileout, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, String title, @ByPtrPtr L_PDF_DATA plpd, @Cast("l_int32") int position );
public static native @Cast("l_int32") int convertImageDataToPdf( @Cast("l_uint8*") byte[] imdata, @Cast("size_t") long size, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("const char*") BytePointer fileout, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, @Cast("const char*") BytePointer title, @ByPtrPtr L_PDF_DATA plpd, @Cast("l_int32") int position );
public static native @Cast("l_int32") int convertImageDataToPdf( @Cast("l_uint8*") BytePointer imdata, @Cast("size_t") long size, @Cast("l_int32") int type, @Cast("l_int32") int quality, String fileout, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, String title, @ByPtrPtr L_PDF_DATA plpd, @Cast("l_int32") int position );
public static native @Cast("l_int32") int convertImageDataToPdf( @Cast("l_uint8*") ByteBuffer imdata, @Cast("size_t") long size, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("const char*") BytePointer fileout, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, @Cast("const char*") BytePointer title, @ByPtrPtr L_PDF_DATA plpd, @Cast("l_int32") int position );
public static native @Cast("l_int32") int convertImageDataToPdf( @Cast("l_uint8*") byte[] imdata, @Cast("size_t") long size, @Cast("l_int32") int type, @Cast("l_int32") int quality, String fileout, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, String title, @ByPtrPtr L_PDF_DATA plpd, @Cast("l_int32") int position );
public static native @Cast("l_int32") int convertToPdfData( @Cast("const char*") BytePointer filein, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("l_uint8**") PointerPointer pdata, @Cast("size_t*") SizeTPointer pnbytes, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, @Cast("const char*") BytePointer title, @Cast("L_PDF_DATA**") PointerPointer plpd, @Cast("l_int32") int position );
public static native @Cast("l_int32") int convertToPdfData( @Cast("const char*") BytePointer filein, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer pnbytes, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, @Cast("const char*") BytePointer title, @ByPtrPtr L_PDF_DATA plpd, @Cast("l_int32") int position );
public static native @Cast("l_int32") int convertToPdfData( String filein, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer pnbytes, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, String title, @ByPtrPtr L_PDF_DATA plpd, @Cast("l_int32") int position );
public static native @Cast("l_int32") int convertToPdfData( @Cast("const char*") BytePointer filein, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer pnbytes, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, @Cast("const char*") BytePointer title, @ByPtrPtr L_PDF_DATA plpd, @Cast("l_int32") int position );
public static native @Cast("l_int32") int convertToPdfData( String filein, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer pnbytes, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, String title, @ByPtrPtr L_PDF_DATA plpd, @Cast("l_int32") int position );
public static native @Cast("l_int32") int convertToPdfData( @Cast("const char*") BytePointer filein, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer pnbytes, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, @Cast("const char*") BytePointer title, @ByPtrPtr L_PDF_DATA plpd, @Cast("l_int32") int position );
public static native @Cast("l_int32") int convertToPdfData( String filein, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer pnbytes, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, String title, @ByPtrPtr L_PDF_DATA plpd, @Cast("l_int32") int position );
public static native @Cast("l_int32") int convertImageDataToPdfData( @Cast("l_uint8*") BytePointer imdata, @Cast("size_t") long size, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("l_uint8**") PointerPointer pdata, @Cast("size_t*") SizeTPointer pnbytes, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, @Cast("const char*") BytePointer title, @Cast("L_PDF_DATA**") PointerPointer plpd, @Cast("l_int32") int position );
public static native @Cast("l_int32") int convertImageDataToPdfData( @Cast("l_uint8*") BytePointer imdata, @Cast("size_t") long size, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer pnbytes, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, @Cast("const char*") BytePointer title, @ByPtrPtr L_PDF_DATA plpd, @Cast("l_int32") int position );
public static native @Cast("l_int32") int convertImageDataToPdfData( @Cast("l_uint8*") ByteBuffer imdata, @Cast("size_t") long size, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer pnbytes, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, String title, @ByPtrPtr L_PDF_DATA plpd, @Cast("l_int32") int position );
public static native @Cast("l_int32") int convertImageDataToPdfData( @Cast("l_uint8*") byte[] imdata, @Cast("size_t") long size, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer pnbytes, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, @Cast("const char*") BytePointer title, @ByPtrPtr L_PDF_DATA plpd, @Cast("l_int32") int position );
public static native @Cast("l_int32") int convertImageDataToPdfData( @Cast("l_uint8*") BytePointer imdata, @Cast("size_t") long size, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer pnbytes, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, String title, @ByPtrPtr L_PDF_DATA plpd, @Cast("l_int32") int position );
public static native @Cast("l_int32") int convertImageDataToPdfData( @Cast("l_uint8*") ByteBuffer imdata, @Cast("size_t") long size, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer pnbytes, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, @Cast("const char*") BytePointer title, @ByPtrPtr L_PDF_DATA plpd, @Cast("l_int32") int position );
public static native @Cast("l_int32") int convertImageDataToPdfData( @Cast("l_uint8*") byte[] imdata, @Cast("size_t") long size, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer pnbytes, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, String title, @ByPtrPtr L_PDF_DATA plpd, @Cast("l_int32") int position );
public static native @Cast("l_int32") int pixConvertToPdf( PIX pix, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("const char*") BytePointer fileout, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, @Cast("const char*") BytePointer title, @Cast("L_PDF_DATA**") PointerPointer plpd, @Cast("l_int32") int position );
public static native @Cast("l_int32") int pixConvertToPdf( PIX pix, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("const char*") BytePointer fileout, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, @Cast("const char*") BytePointer title, @ByPtrPtr L_PDF_DATA plpd, @Cast("l_int32") int position );
public static native @Cast("l_int32") int pixConvertToPdf( PIX pix, @Cast("l_int32") int type, @Cast("l_int32") int quality, String fileout, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, String title, @ByPtrPtr L_PDF_DATA plpd, @Cast("l_int32") int position );
public static native @Cast("l_int32") int pixWriteStreamPdf( @Cast("FILE*") Pointer fp, PIX pix, @Cast("l_int32") int res, @Cast("const char*") BytePointer title );
public static native @Cast("l_int32") int pixWriteStreamPdf( @Cast("FILE*") Pointer fp, PIX pix, @Cast("l_int32") int res, String title );
public static native @Cast("l_int32") int pixWriteMemPdf( @Cast("l_uint8**") PointerPointer pdata, @Cast("size_t*") SizeTPointer pnbytes, PIX pix, @Cast("l_int32") int res, @Cast("const char*") BytePointer title );
public static native @Cast("l_int32") int pixWriteMemPdf( @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer pnbytes, PIX pix, @Cast("l_int32") int res, @Cast("const char*") BytePointer title );
public static native @Cast("l_int32") int pixWriteMemPdf( @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer pnbytes, PIX pix, @Cast("l_int32") int res, String title );
public static native @Cast("l_int32") int pixWriteMemPdf( @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer pnbytes, PIX pix, @Cast("l_int32") int res, @Cast("const char*") BytePointer title );
public static native @Cast("l_int32") int pixWriteMemPdf( @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer pnbytes, PIX pix, @Cast("l_int32") int res, String title );
public static native @Cast("l_int32") int pixWriteMemPdf( @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer pnbytes, PIX pix, @Cast("l_int32") int res, @Cast("const char*") BytePointer title );
public static native @Cast("l_int32") int pixWriteMemPdf( @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer pnbytes, PIX pix, @Cast("l_int32") int res, String title );
public static native @Cast("l_int32") int convertSegmentedFilesToPdf( @Cast("const char*") BytePointer dirname, @Cast("const char*") BytePointer substr, @Cast("l_int32") int res, @Cast("l_int32") int type, @Cast("l_int32") int thresh, BOXAA baa, @Cast("l_int32") int quality, @Cast("l_float32") float scalefactor, @Cast("const char*") BytePointer title, @Cast("const char*") BytePointer fileout );
public static native @Cast("l_int32") int convertSegmentedFilesToPdf( String dirname, String substr, @Cast("l_int32") int res, @Cast("l_int32") int type, @Cast("l_int32") int thresh, BOXAA baa, @Cast("l_int32") int quality, @Cast("l_float32") float scalefactor, String title, String fileout );
public static native BOXAA convertNumberedMasksToBoxaa( @Cast("const char*") BytePointer dirname, @Cast("const char*") BytePointer substr, @Cast("l_int32") int numpre, @Cast("l_int32") int numpost );
public static native BOXAA convertNumberedMasksToBoxaa( String dirname, String substr, @Cast("l_int32") int numpre, @Cast("l_int32") int numpost );
public static native @Cast("l_int32") int convertToPdfSegmented( @Cast("const char*") BytePointer filein, @Cast("l_int32") int res, @Cast("l_int32") int type, @Cast("l_int32") int thresh, BOXA boxa, @Cast("l_int32") int quality, @Cast("l_float32") float scalefactor, @Cast("const char*") BytePointer title, @Cast("const char*") BytePointer fileout );
public static native @Cast("l_int32") int convertToPdfSegmented( String filein, @Cast("l_int32") int res, @Cast("l_int32") int type, @Cast("l_int32") int thresh, BOXA boxa, @Cast("l_int32") int quality, @Cast("l_float32") float scalefactor, String title, String fileout );
public static native @Cast("l_int32") int pixConvertToPdfSegmented( PIX pixs, @Cast("l_int32") int res, @Cast("l_int32") int type, @Cast("l_int32") int thresh, BOXA boxa, @Cast("l_int32") int quality, @Cast("l_float32") float scalefactor, @Cast("const char*") BytePointer title, @Cast("const char*") BytePointer fileout );
public static native @Cast("l_int32") int pixConvertToPdfSegmented( PIX pixs, @Cast("l_int32") int res, @Cast("l_int32") int type, @Cast("l_int32") int thresh, BOXA boxa, @Cast("l_int32") int quality, @Cast("l_float32") float scalefactor, String title, String fileout );
public static native @Cast("l_int32") int convertToPdfDataSegmented( @Cast("const char*") BytePointer filein, @Cast("l_int32") int res, @Cast("l_int32") int type, @Cast("l_int32") int thresh, BOXA boxa, @Cast("l_int32") int quality, @Cast("l_float32") float scalefactor, @Cast("const char*") BytePointer title, @Cast("l_uint8**") PointerPointer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int convertToPdfDataSegmented( @Cast("const char*") BytePointer filein, @Cast("l_int32") int res, @Cast("l_int32") int type, @Cast("l_int32") int thresh, BOXA boxa, @Cast("l_int32") int quality, @Cast("l_float32") float scalefactor, @Cast("const char*") BytePointer title, @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int convertToPdfDataSegmented( String filein, @Cast("l_int32") int res, @Cast("l_int32") int type, @Cast("l_int32") int thresh, BOXA boxa, @Cast("l_int32") int quality, @Cast("l_float32") float scalefactor, String title, @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int convertToPdfDataSegmented( @Cast("const char*") BytePointer filein, @Cast("l_int32") int res, @Cast("l_int32") int type, @Cast("l_int32") int thresh, BOXA boxa, @Cast("l_int32") int quality, @Cast("l_float32") float scalefactor, @Cast("const char*") BytePointer title, @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int convertToPdfDataSegmented( String filein, @Cast("l_int32") int res, @Cast("l_int32") int type, @Cast("l_int32") int thresh, BOXA boxa, @Cast("l_int32") int quality, @Cast("l_float32") float scalefactor, String title, @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int convertToPdfDataSegmented( @Cast("const char*") BytePointer filein, @Cast("l_int32") int res, @Cast("l_int32") int type, @Cast("l_int32") int thresh, BOXA boxa, @Cast("l_int32") int quality, @Cast("l_float32") float scalefactor, @Cast("const char*") BytePointer title, @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int convertToPdfDataSegmented( String filein, @Cast("l_int32") int res, @Cast("l_int32") int type, @Cast("l_int32") int thresh, BOXA boxa, @Cast("l_int32") int quality, @Cast("l_float32") float scalefactor, String title, @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int pixConvertToPdfDataSegmented( PIX pixs, @Cast("l_int32") int res, @Cast("l_int32") int type, @Cast("l_int32") int thresh, BOXA boxa, @Cast("l_int32") int quality, @Cast("l_float32") float scalefactor, @Cast("const char*") BytePointer title, @Cast("l_uint8**") PointerPointer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int pixConvertToPdfDataSegmented( PIX pixs, @Cast("l_int32") int res, @Cast("l_int32") int type, @Cast("l_int32") int thresh, BOXA boxa, @Cast("l_int32") int quality, @Cast("l_float32") float scalefactor, @Cast("const char*") BytePointer title, @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int pixConvertToPdfDataSegmented( PIX pixs, @Cast("l_int32") int res, @Cast("l_int32") int type, @Cast("l_int32") int thresh, BOXA boxa, @Cast("l_int32") int quality, @Cast("l_float32") float scalefactor, String title, @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int pixConvertToPdfDataSegmented( PIX pixs, @Cast("l_int32") int res, @Cast("l_int32") int type, @Cast("l_int32") int thresh, BOXA boxa, @Cast("l_int32") int quality, @Cast("l_float32") float scalefactor, @Cast("const char*") BytePointer title, @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int pixConvertToPdfDataSegmented( PIX pixs, @Cast("l_int32") int res, @Cast("l_int32") int type, @Cast("l_int32") int thresh, BOXA boxa, @Cast("l_int32") int quality, @Cast("l_float32") float scalefactor, String title, @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int pixConvertToPdfDataSegmented( PIX pixs, @Cast("l_int32") int res, @Cast("l_int32") int type, @Cast("l_int32") int thresh, BOXA boxa, @Cast("l_int32") int quality, @Cast("l_float32") float scalefactor, @Cast("const char*") BytePointer title, @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int pixConvertToPdfDataSegmented( PIX pixs, @Cast("l_int32") int res, @Cast("l_int32") int type, @Cast("l_int32") int thresh, BOXA boxa, @Cast("l_int32") int quality, @Cast("l_float32") float scalefactor, String title, @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int concatenatePdf( @Cast("const char*") BytePointer dirname, @Cast("const char*") BytePointer substr, @Cast("const char*") BytePointer fileout );
public static native @Cast("l_int32") int concatenatePdf( String dirname, String substr, String fileout );
public static native @Cast("l_int32") int saConcatenatePdf( SARRAY sa, @Cast("const char*") BytePointer fileout );
public static native @Cast("l_int32") int saConcatenatePdf( SARRAY sa, String fileout );
public static native @Cast("l_int32") int ptraConcatenatePdf( L_PTRA pa, @Cast("const char*") BytePointer fileout );
public static native @Cast("l_int32") int ptraConcatenatePdf( L_PTRA pa, String fileout );
public static native @Cast("l_int32") int concatenatePdfToData( @Cast("const char*") BytePointer dirname, @Cast("const char*") BytePointer substr, @Cast("l_uint8**") PointerPointer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int concatenatePdfToData( @Cast("const char*") BytePointer dirname, @Cast("const char*") BytePointer substr, @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int concatenatePdfToData( String dirname, String substr, @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int concatenatePdfToData( @Cast("const char*") BytePointer dirname, @Cast("const char*") BytePointer substr, @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int concatenatePdfToData( String dirname, String substr, @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int concatenatePdfToData( @Cast("const char*") BytePointer dirname, @Cast("const char*") BytePointer substr, @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int concatenatePdfToData( String dirname, String substr, @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int saConcatenatePdfToData( SARRAY sa, @Cast("l_uint8**") PointerPointer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int saConcatenatePdfToData( SARRAY sa, @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int saConcatenatePdfToData( SARRAY sa, @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int saConcatenatePdfToData( SARRAY sa, @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int pixConvertToPdfData( PIX pix, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("l_uint8**") PointerPointer pdata, @Cast("size_t*") SizeTPointer pnbytes, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, @Cast("const char*") BytePointer title, @Cast("L_PDF_DATA**") PointerPointer plpd, @Cast("l_int32") int position );
public static native @Cast("l_int32") int pixConvertToPdfData( PIX pix, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer pnbytes, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, @Cast("const char*") BytePointer title, @ByPtrPtr L_PDF_DATA plpd, @Cast("l_int32") int position );
public static native @Cast("l_int32") int pixConvertToPdfData( PIX pix, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer pnbytes, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, String title, @ByPtrPtr L_PDF_DATA plpd, @Cast("l_int32") int position );
public static native @Cast("l_int32") int pixConvertToPdfData( PIX pix, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer pnbytes, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, @Cast("const char*") BytePointer title, @ByPtrPtr L_PDF_DATA plpd, @Cast("l_int32") int position );
public static native @Cast("l_int32") int pixConvertToPdfData( PIX pix, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer pnbytes, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, String title, @ByPtrPtr L_PDF_DATA plpd, @Cast("l_int32") int position );
public static native @Cast("l_int32") int pixConvertToPdfData( PIX pix, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer pnbytes, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, @Cast("const char*") BytePointer title, @ByPtrPtr L_PDF_DATA plpd, @Cast("l_int32") int position );
public static native @Cast("l_int32") int pixConvertToPdfData( PIX pix, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer pnbytes, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, String title, @ByPtrPtr L_PDF_DATA plpd, @Cast("l_int32") int position );
public static native @Cast("l_int32") int ptraConcatenatePdfToData( L_PTRA pa_data, SARRAY sa, @Cast("l_uint8**") PointerPointer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int ptraConcatenatePdfToData( L_PTRA pa_data, SARRAY sa, @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int ptraConcatenatePdfToData( L_PTRA pa_data, SARRAY sa, @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int ptraConcatenatePdfToData( L_PTRA pa_data, SARRAY sa, @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int l_generateCIDataForPdf( @Cast("const char*") BytePointer fname, PIX pix, @Cast("l_int32") int quality, @Cast("L_COMP_DATA**") PointerPointer pcid );
public static native @Cast("l_int32") int l_generateCIDataForPdf( @Cast("const char*") BytePointer fname, PIX pix, @Cast("l_int32") int quality, @ByPtrPtr L_COMP_DATA pcid );
public static native @Cast("l_int32") int l_generateCIDataForPdf( String fname, PIX pix, @Cast("l_int32") int quality, @ByPtrPtr L_COMP_DATA pcid );
public static native L_COMP_DATA l_generateFlateDataPdf( @Cast("const char*") BytePointer fname, PIX pixs );
public static native L_COMP_DATA l_generateFlateDataPdf( String fname, PIX pixs );
public static native L_COMP_DATA l_generateJpegData( @Cast("const char*") BytePointer fname, @Cast("l_int32") int ascii85flag );
public static native L_COMP_DATA l_generateJpegData( String fname, @Cast("l_int32") int ascii85flag );
public static native @Cast("l_int32") int l_generateCIData( @Cast("const char*") BytePointer fname, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("l_int32") int ascii85, @Cast("L_COMP_DATA**") PointerPointer pcid );
public static native @Cast("l_int32") int l_generateCIData( @Cast("const char*") BytePointer fname, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("l_int32") int ascii85, @ByPtrPtr L_COMP_DATA pcid );
public static native @Cast("l_int32") int l_generateCIData( String fname, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("l_int32") int ascii85, @ByPtrPtr L_COMP_DATA pcid );
public static native @Cast("l_int32") int pixGenerateCIData( PIX pixs, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("l_int32") int ascii85, @Cast("L_COMP_DATA**") PointerPointer pcid );
public static native @Cast("l_int32") int pixGenerateCIData( PIX pixs, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("l_int32") int ascii85, @ByPtrPtr L_COMP_DATA pcid );
public static native L_COMP_DATA l_generateFlateData( @Cast("const char*") BytePointer fname, @Cast("l_int32") int ascii85flag );
public static native L_COMP_DATA l_generateFlateData( String fname, @Cast("l_int32") int ascii85flag );
public static native L_COMP_DATA l_generateG4Data( @Cast("const char*") BytePointer fname, @Cast("l_int32") int ascii85flag );
public static native L_COMP_DATA l_generateG4Data( String fname, @Cast("l_int32") int ascii85flag );
public static native @Cast("l_int32") int cidConvertToPdfData( L_COMP_DATA cid, @Cast("const char*") BytePointer title, @Cast("l_uint8**") PointerPointer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int cidConvertToPdfData( L_COMP_DATA cid, @Cast("const char*") BytePointer title, @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int cidConvertToPdfData( L_COMP_DATA cid, String title, @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int cidConvertToPdfData( L_COMP_DATA cid, @Cast("const char*") BytePointer title, @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int cidConvertToPdfData( L_COMP_DATA cid, String title, @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int cidConvertToPdfData( L_COMP_DATA cid, @Cast("const char*") BytePointer title, @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int cidConvertToPdfData( L_COMP_DATA cid, String title, @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native void l_CIDataDestroy( @Cast("L_COMP_DATA**") PointerPointer pcid );
public static native void l_CIDataDestroy( @ByPtrPtr L_COMP_DATA pcid );
public static native void l_pdfSetG4ImageMask( @Cast("l_int32") int flag );
public static native void l_pdfSetDateAndVersion( @Cast("l_int32") int flag );
public static class Allocator_long extends FunctionPointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public    Allocator_long(Pointer p) { super(p); }
    protected Allocator_long() { allocate(); }
    private native void allocate();
    public native Pointer call( @Cast("size_t") long arg0 );
}
public static class Deallocator_Pointer extends FunctionPointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public    Deallocator_Pointer(Pointer p) { super(p); }
    protected Deallocator_Pointer() { allocate(); }
    private native void allocate();
    public native void call( Pointer arg0 );
}
public static native void setPixMemoryManager( Allocator_long allocator, Deallocator_Pointer deallocator );
public static native PIX pixCreate( @Cast("l_int32") int width, @Cast("l_int32") int height, @Cast("l_int32") int depth );
public static native PIX pixCreateNoInit( @Cast("l_int32") int width, @Cast("l_int32") int height, @Cast("l_int32") int depth );
public static native PIX pixCreateTemplate( PIX pixs );
public static native PIX pixCreateTemplateNoInit( PIX pixs );
public static native PIX pixCreateHeader( @Cast("l_int32") int width, @Cast("l_int32") int height, @Cast("l_int32") int depth );
public static native PIX pixClone( PIX pixs );
public static native void pixDestroy( @Cast("PIX**") PointerPointer ppix );
public static native void pixDestroy( @ByPtrPtr PIX ppix );
public static native PIX pixCopy( PIX pixd, PIX pixs );
public static native @Cast("l_int32") int pixResizeImageData( PIX pixd, PIX pixs );
public static native @Cast("l_int32") int pixCopyColormap( PIX pixd, PIX pixs );
public static native @Cast("l_int32") int pixSizesEqual( PIX pix1, PIX pix2 );
public static native @Cast("l_int32") int pixTransferAllData( PIX pixd, @Cast("PIX**") PointerPointer ppixs, @Cast("l_int32") int copytext, @Cast("l_int32") int copyformat );
public static native @Cast("l_int32") int pixTransferAllData( PIX pixd, @ByPtrPtr PIX ppixs, @Cast("l_int32") int copytext, @Cast("l_int32") int copyformat );
public static native @Cast("l_int32") int pixSwapAndDestroy( @Cast("PIX**") PointerPointer ppixd, @Cast("PIX**") PointerPointer ppixs );
public static native @Cast("l_int32") int pixSwapAndDestroy( @ByPtrPtr PIX ppixd, @ByPtrPtr PIX ppixs );
public static native @Cast("l_int32") int pixGetWidth( PIX pix );
public static native @Cast("l_int32") int pixSetWidth( PIX pix, @Cast("l_int32") int width );
public static native @Cast("l_int32") int pixGetHeight( PIX pix );
public static native @Cast("l_int32") int pixSetHeight( PIX pix, @Cast("l_int32") int height );
public static native @Cast("l_int32") int pixGetDepth( PIX pix );
public static native @Cast("l_int32") int pixSetDepth( PIX pix, @Cast("l_int32") int depth );
public static native @Cast("l_int32") int pixGetDimensions( PIX pix, @Cast("l_int32*") IntPointer pw, @Cast("l_int32*") IntPointer ph, @Cast("l_int32*") IntPointer pd );
public static native @Cast("l_int32") int pixGetDimensions( PIX pix, @Cast("l_int32*") IntBuffer pw, @Cast("l_int32*") IntBuffer ph, @Cast("l_int32*") IntBuffer pd );
public static native @Cast("l_int32") int pixGetDimensions( PIX pix, @Cast("l_int32*") int[] pw, @Cast("l_int32*") int[] ph, @Cast("l_int32*") int[] pd );
public static native @Cast("l_int32") int pixSetDimensions( PIX pix, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int d );
public static native @Cast("l_int32") int pixCopyDimensions( PIX pixd, PIX pixs );
public static native @Cast("l_int32") int pixGetSpp( PIX pix );
public static native @Cast("l_int32") int pixSetSpp( PIX pix, @Cast("l_int32") int spp );
public static native @Cast("l_int32") int pixCopySpp( PIX pixd, PIX pixs );
public static native @Cast("l_int32") int pixGetWpl( PIX pix );
public static native @Cast("l_int32") int pixSetWpl( PIX pix, @Cast("l_int32") int wpl );
public static native @Cast("l_int32") int pixGetRefcount( PIX pix );
public static native @Cast("l_int32") int pixChangeRefcount( PIX pix, @Cast("l_int32") int delta );
public static native @Cast("l_int32") int pixGetXRes( PIX pix );
public static native @Cast("l_int32") int pixSetXRes( PIX pix, @Cast("l_int32") int res );
public static native @Cast("l_int32") int pixGetYRes( PIX pix );
public static native @Cast("l_int32") int pixSetYRes( PIX pix, @Cast("l_int32") int res );
public static native @Cast("l_int32") int pixGetResolution( PIX pix, @Cast("l_int32*") IntPointer pxres, @Cast("l_int32*") IntPointer pyres );
public static native @Cast("l_int32") int pixGetResolution( PIX pix, @Cast("l_int32*") IntBuffer pxres, @Cast("l_int32*") IntBuffer pyres );
public static native @Cast("l_int32") int pixGetResolution( PIX pix, @Cast("l_int32*") int[] pxres, @Cast("l_int32*") int[] pyres );
public static native @Cast("l_int32") int pixSetResolution( PIX pix, @Cast("l_int32") int xres, @Cast("l_int32") int yres );
public static native @Cast("l_int32") int pixCopyResolution( PIX pixd, PIX pixs );
public static native @Cast("l_int32") int pixScaleResolution( PIX pix, @Cast("l_float32") float xscale, @Cast("l_float32") float yscale );
public static native @Cast("l_int32") int pixGetInputFormat( PIX pix );
public static native @Cast("l_int32") int pixSetInputFormat( PIX pix, @Cast("l_int32") int informat );
public static native @Cast("l_int32") int pixCopyInputFormat( PIX pixd, PIX pixs );
public static native @Cast("char*") BytePointer pixGetText( PIX pix );
public static native @Cast("l_int32") int pixSetText( PIX pix, @Cast("const char*") BytePointer textstring );
public static native @Cast("l_int32") int pixSetText( PIX pix, String textstring );
public static native @Cast("l_int32") int pixAddText( PIX pix, @Cast("const char*") BytePointer textstring );
public static native @Cast("l_int32") int pixAddText( PIX pix, String textstring );
public static native @Cast("l_int32") int pixCopyText( PIX pixd, PIX pixs );
public static native PIXCMAP pixGetColormap( PIX pix );
public static native @Cast("l_int32") int pixSetColormap( PIX pix, PIXCMAP colormap );
public static native @Cast("l_int32") int pixDestroyColormap( PIX pix );
public static native @Cast("l_uint32*") IntPointer pixGetData( PIX pix );
public static native @Cast("l_int32") int pixSetData( PIX pix, @Cast("l_uint32*") IntPointer data );
public static native @Cast("l_int32") int pixSetData( PIX pix, @Cast("l_uint32*") IntBuffer data );
public static native @Cast("l_int32") int pixSetData( PIX pix, @Cast("l_uint32*") int[] data );
public static native @Cast("l_uint32*") IntPointer pixExtractData( PIX pixs );
public static native @Cast("l_int32") int pixFreeData( PIX pix );
public static native @Cast("void**") PointerPointer pixGetLinePtrs( PIX pix, @Cast("l_int32*") IntPointer psize );
public static native @Cast("void**") @ByPtrPtr Pointer pixGetLinePtrs( PIX pix, @Cast("l_int32*") IntBuffer psize );
public static native @Cast("void**") @ByPtrPtr Pointer pixGetLinePtrs( PIX pix, @Cast("l_int32*") int[] psize );
public static native @Cast("l_int32") int pixPrintStreamInfo( @Cast("FILE*") Pointer fp, PIX pix, @Cast("const char*") BytePointer text );
public static native @Cast("l_int32") int pixPrintStreamInfo( @Cast("FILE*") Pointer fp, PIX pix, String text );
public static native @Cast("l_int32") int pixGetPixel( PIX pix, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_uint32*") IntPointer pval );
public static native @Cast("l_int32") int pixGetPixel( PIX pix, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_uint32*") IntBuffer pval );
public static native @Cast("l_int32") int pixGetPixel( PIX pix, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_uint32*") int[] pval );
public static native @Cast("l_int32") int pixSetPixel( PIX pix, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_uint32") int val );
public static native @Cast("l_int32") int pixGetRGBPixel( PIX pix, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32*") IntPointer prval, @Cast("l_int32*") IntPointer pgval, @Cast("l_int32*") IntPointer pbval );
public static native @Cast("l_int32") int pixGetRGBPixel( PIX pix, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32*") IntBuffer prval, @Cast("l_int32*") IntBuffer pgval, @Cast("l_int32*") IntBuffer pbval );
public static native @Cast("l_int32") int pixGetRGBPixel( PIX pix, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32*") int[] prval, @Cast("l_int32*") int[] pgval, @Cast("l_int32*") int[] pbval );
public static native @Cast("l_int32") int pixSetRGBPixel( PIX pix, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval );
public static native @Cast("l_int32") int pixGetRandomPixel( PIX pix, @Cast("l_uint32*") IntPointer pval, @Cast("l_int32*") IntPointer px, @Cast("l_int32*") IntPointer py );
public static native @Cast("l_int32") int pixGetRandomPixel( PIX pix, @Cast("l_uint32*") IntBuffer pval, @Cast("l_int32*") IntBuffer px, @Cast("l_int32*") IntBuffer py );
public static native @Cast("l_int32") int pixGetRandomPixel( PIX pix, @Cast("l_uint32*") int[] pval, @Cast("l_int32*") int[] px, @Cast("l_int32*") int[] py );
public static native @Cast("l_int32") int pixClearPixel( PIX pix, @Cast("l_int32") int x, @Cast("l_int32") int y );
public static native @Cast("l_int32") int pixFlipPixel( PIX pix, @Cast("l_int32") int x, @Cast("l_int32") int y );
public static native void setPixelLow( @Cast("l_uint32*") IntPointer line, @Cast("l_int32") int x, @Cast("l_int32") int depth, @Cast("l_uint32") int val );
public static native void setPixelLow( @Cast("l_uint32*") IntBuffer line, @Cast("l_int32") int x, @Cast("l_int32") int depth, @Cast("l_uint32") int val );
public static native void setPixelLow( @Cast("l_uint32*") int[] line, @Cast("l_int32") int x, @Cast("l_int32") int depth, @Cast("l_uint32") int val );
public static native @Cast("l_int32") int pixGetBlackOrWhiteVal( PIX pixs, @Cast("l_int32") int op, @Cast("l_uint32*") IntPointer pval );
public static native @Cast("l_int32") int pixGetBlackOrWhiteVal( PIX pixs, @Cast("l_int32") int op, @Cast("l_uint32*") IntBuffer pval );
public static native @Cast("l_int32") int pixGetBlackOrWhiteVal( PIX pixs, @Cast("l_int32") int op, @Cast("l_uint32*") int[] pval );
public static native @Cast("l_int32") int pixClearAll( PIX pix );
public static native @Cast("l_int32") int pixSetAll( PIX pix );
public static native @Cast("l_int32") int pixSetAllGray( PIX pix, @Cast("l_int32") int grayval );
public static native @Cast("l_int32") int pixSetAllArbitrary( PIX pix, @Cast("l_uint32") int val );
public static native @Cast("l_int32") int pixSetBlackOrWhite( PIX pixs, @Cast("l_int32") int op );
public static native @Cast("l_int32") int pixSetComponentArbitrary( PIX pix, @Cast("l_int32") int comp, @Cast("l_int32") int val );
public static native @Cast("l_int32") int pixClearInRect( PIX pix, BOX box );
public static native @Cast("l_int32") int pixSetInRect( PIX pix, BOX box );
public static native @Cast("l_int32") int pixSetInRectArbitrary( PIX pix, BOX box, @Cast("l_uint32") int val );
public static native @Cast("l_int32") int pixBlendInRect( PIX pixs, BOX box, @Cast("l_uint32") int val, @Cast("l_float32") float fract );
public static native @Cast("l_int32") int pixSetPadBits( PIX pix, @Cast("l_int32") int val );
public static native @Cast("l_int32") int pixSetPadBitsBand( PIX pix, @Cast("l_int32") int by, @Cast("l_int32") int bh, @Cast("l_int32") int val );
public static native @Cast("l_int32") int pixSetOrClearBorder( PIX pixs, @Cast("l_int32") int left, @Cast("l_int32") int right, @Cast("l_int32") int top, @Cast("l_int32") int bot, @Cast("l_int32") int op );
public static native @Cast("l_int32") int pixSetBorderVal( PIX pixs, @Cast("l_int32") int left, @Cast("l_int32") int right, @Cast("l_int32") int top, @Cast("l_int32") int bot, @Cast("l_uint32") int val );
public static native @Cast("l_int32") int pixSetBorderRingVal( PIX pixs, @Cast("l_int32") int dist, @Cast("l_uint32") int val );
public static native @Cast("l_int32") int pixSetMirroredBorder( PIX pixs, @Cast("l_int32") int left, @Cast("l_int32") int right, @Cast("l_int32") int top, @Cast("l_int32") int bot );
public static native PIX pixCopyBorder( PIX pixd, PIX pixs, @Cast("l_int32") int left, @Cast("l_int32") int right, @Cast("l_int32") int top, @Cast("l_int32") int bot );
public static native PIX pixAddBorder( PIX pixs, @Cast("l_int32") int npix, @Cast("l_uint32") int val );
public static native PIX pixAddBlackOrWhiteBorder( PIX pixs, @Cast("l_int32") int left, @Cast("l_int32") int right, @Cast("l_int32") int top, @Cast("l_int32") int bot, @Cast("l_int32") int op );
public static native PIX pixAddBorderGeneral( PIX pixs, @Cast("l_int32") int left, @Cast("l_int32") int right, @Cast("l_int32") int top, @Cast("l_int32") int bot, @Cast("l_uint32") int val );
public static native PIX pixRemoveBorder( PIX pixs, @Cast("l_int32") int npix );
public static native PIX pixRemoveBorderGeneral( PIX pixs, @Cast("l_int32") int left, @Cast("l_int32") int right, @Cast("l_int32") int top, @Cast("l_int32") int bot );
public static native PIX pixRemoveBorderToSize( PIX pixs, @Cast("l_int32") int wd, @Cast("l_int32") int hd );
public static native PIX pixAddMirroredBorder( PIX pixs, @Cast("l_int32") int left, @Cast("l_int32") int right, @Cast("l_int32") int top, @Cast("l_int32") int bot );
public static native PIX pixAddRepeatedBorder( PIX pixs, @Cast("l_int32") int left, @Cast("l_int32") int right, @Cast("l_int32") int top, @Cast("l_int32") int bot );
public static native PIX pixAddMixedBorder( PIX pixs, @Cast("l_int32") int left, @Cast("l_int32") int right, @Cast("l_int32") int top, @Cast("l_int32") int bot );
public static native PIX pixAddContinuedBorder( PIX pixs, @Cast("l_int32") int left, @Cast("l_int32") int right, @Cast("l_int32") int top, @Cast("l_int32") int bot );
public static native @Cast("l_int32") int pixShiftAndTransferAlpha( PIX pixd, PIX pixs, @Cast("l_float32") float shiftx, @Cast("l_float32") float shifty );
public static native PIX pixDisplayLayersRGBA( PIX pixs, @Cast("l_uint32") int val, @Cast("l_int32") int maxw );
public static native PIX pixCreateRGBImage( PIX pixr, PIX pixg, PIX pixb );
public static native PIX pixGetRGBComponent( PIX pixs, @Cast("l_int32") int comp );
public static native @Cast("l_int32") int pixSetRGBComponent( PIX pixd, PIX pixs, @Cast("l_int32") int comp );
public static native PIX pixGetRGBComponentCmap( PIX pixs, @Cast("l_int32") int comp );
public static native @Cast("l_int32") int pixCopyRGBComponent( PIX pixd, PIX pixs, @Cast("l_int32") int comp );
public static native @Cast("l_int32") int composeRGBPixel( @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_uint32*") IntPointer ppixel );
public static native @Cast("l_int32") int composeRGBPixel( @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_uint32*") IntBuffer ppixel );
public static native @Cast("l_int32") int composeRGBPixel( @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_uint32*") int[] ppixel );
public static native @Cast("l_int32") int composeRGBAPixel( @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_int32") int aval, @Cast("l_uint32*") IntPointer ppixel );
public static native @Cast("l_int32") int composeRGBAPixel( @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_int32") int aval, @Cast("l_uint32*") IntBuffer ppixel );
public static native @Cast("l_int32") int composeRGBAPixel( @Cast("l_int32") int rval, @Cast("l_int32") int gval, @Cast("l_int32") int bval, @Cast("l_int32") int aval, @Cast("l_uint32*") int[] ppixel );
public static native void extractRGBValues( @Cast("l_uint32") int pixel, @Cast("l_int32*") IntPointer prval, @Cast("l_int32*") IntPointer pgval, @Cast("l_int32*") IntPointer pbval );
public static native void extractRGBValues( @Cast("l_uint32") int pixel, @Cast("l_int32*") IntBuffer prval, @Cast("l_int32*") IntBuffer pgval, @Cast("l_int32*") IntBuffer pbval );
public static native void extractRGBValues( @Cast("l_uint32") int pixel, @Cast("l_int32*") int[] prval, @Cast("l_int32*") int[] pgval, @Cast("l_int32*") int[] pbval );
public static native void extractRGBAValues( @Cast("l_uint32") int pixel, @Cast("l_int32*") IntPointer prval, @Cast("l_int32*") IntPointer pgval, @Cast("l_int32*") IntPointer pbval, @Cast("l_int32*") IntPointer paval );
public static native void extractRGBAValues( @Cast("l_uint32") int pixel, @Cast("l_int32*") IntBuffer prval, @Cast("l_int32*") IntBuffer pgval, @Cast("l_int32*") IntBuffer pbval, @Cast("l_int32*") IntBuffer paval );
public static native void extractRGBAValues( @Cast("l_uint32") int pixel, @Cast("l_int32*") int[] prval, @Cast("l_int32*") int[] pgval, @Cast("l_int32*") int[] pbval, @Cast("l_int32*") int[] paval );
public static native @Cast("l_int32") int extractMinMaxComponent( @Cast("l_uint32") int pixel, @Cast("l_int32") int type );
public static native @Cast("l_int32") int pixGetRGBLine( PIX pixs, @Cast("l_int32") int row, @Cast("l_uint8*") BytePointer bufr, @Cast("l_uint8*") BytePointer bufg, @Cast("l_uint8*") BytePointer bufb );
public static native @Cast("l_int32") int pixGetRGBLine( PIX pixs, @Cast("l_int32") int row, @Cast("l_uint8*") ByteBuffer bufr, @Cast("l_uint8*") ByteBuffer bufg, @Cast("l_uint8*") ByteBuffer bufb );
public static native @Cast("l_int32") int pixGetRGBLine( PIX pixs, @Cast("l_int32") int row, @Cast("l_uint8*") byte[] bufr, @Cast("l_uint8*") byte[] bufg, @Cast("l_uint8*") byte[] bufb );
public static native PIX pixEndianByteSwapNew( PIX pixs );
public static native @Cast("l_int32") int pixEndianByteSwap( PIX pixs );
public static native @Cast("l_int32") int lineEndianByteSwap( @Cast("l_uint32*") IntPointer datad, @Cast("l_uint32*") IntPointer datas, @Cast("l_int32") int wpl );
public static native @Cast("l_int32") int lineEndianByteSwap( @Cast("l_uint32*") IntBuffer datad, @Cast("l_uint32*") IntBuffer datas, @Cast("l_int32") int wpl );
public static native @Cast("l_int32") int lineEndianByteSwap( @Cast("l_uint32*") int[] datad, @Cast("l_uint32*") int[] datas, @Cast("l_int32") int wpl );
public static native PIX pixEndianTwoByteSwapNew( PIX pixs );
public static native @Cast("l_int32") int pixEndianTwoByteSwap( PIX pixs );
public static native @Cast("l_int32") int pixGetRasterData( PIX pixs, @Cast("l_uint8**") PointerPointer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int pixGetRasterData( PIX pixs, @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int pixGetRasterData( PIX pixs, @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int pixGetRasterData( PIX pixs, @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int pixAlphaIsOpaque( PIX pix, @Cast("l_int32*") IntPointer popaque );
public static native @Cast("l_int32") int pixAlphaIsOpaque( PIX pix, @Cast("l_int32*") IntBuffer popaque );
public static native @Cast("l_int32") int pixAlphaIsOpaque( PIX pix, @Cast("l_int32*") int[] popaque );
public static native @Cast("l_uint8**") PointerPointer pixSetupByteProcessing( PIX pix, @Cast("l_int32*") IntPointer pw, @Cast("l_int32*") IntPointer ph );
public static native @Cast("l_uint8**") @ByPtrPtr ByteBuffer pixSetupByteProcessing( PIX pix, @Cast("l_int32*") IntBuffer pw, @Cast("l_int32*") IntBuffer ph );
public static native @Cast("l_uint8**") @ByPtrPtr byte[] pixSetupByteProcessing( PIX pix, @Cast("l_int32*") int[] pw, @Cast("l_int32*") int[] ph );
public static native @Cast("l_int32") int pixCleanupByteProcessing( PIX pix, @Cast("l_uint8**") PointerPointer lineptrs );
public static native @Cast("l_int32") int pixCleanupByteProcessing( PIX pix, @Cast("l_uint8**") @ByPtrPtr BytePointer lineptrs );
public static native @Cast("l_int32") int pixCleanupByteProcessing( PIX pix, @Cast("l_uint8**") @ByPtrPtr ByteBuffer lineptrs );
public static native @Cast("l_int32") int pixCleanupByteProcessing( PIX pix, @Cast("l_uint8**") @ByPtrPtr byte[] lineptrs );
public static native void l_setAlphaMaskBorder( @Cast("l_float32") float val1, @Cast("l_float32") float val2 );
public static native @Cast("l_int32") int pixSetMasked( PIX pixd, PIX pixm, @Cast("l_uint32") int val );
public static native @Cast("l_int32") int pixSetMaskedGeneral( PIX pixd, PIX pixm, @Cast("l_uint32") int val, @Cast("l_int32") int x, @Cast("l_int32") int y );
public static native @Cast("l_int32") int pixCombineMasked( PIX pixd, PIX pixs, PIX pixm );
public static native @Cast("l_int32") int pixCombineMaskedGeneral( PIX pixd, PIX pixs, PIX pixm, @Cast("l_int32") int x, @Cast("l_int32") int y );
public static native @Cast("l_int32") int pixPaintThroughMask( PIX pixd, PIX pixm, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_uint32") int val );
public static native @Cast("l_int32") int pixPaintSelfThroughMask( PIX pixd, PIX pixm, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int searchdir, @Cast("l_int32") int mindist, @Cast("l_int32") int tilesize, @Cast("l_int32") int ntiles, @Cast("l_int32") int distblend );
public static native PIX pixMakeMaskFromLUT( PIX pixs, @Cast("l_int32*") IntPointer tab );
public static native PIX pixMakeMaskFromLUT( PIX pixs, @Cast("l_int32*") IntBuffer tab );
public static native PIX pixMakeMaskFromLUT( PIX pixs, @Cast("l_int32*") int[] tab );
public static native PIX pixSetUnderTransparency( PIX pixs, @Cast("l_uint32") int val, @Cast("l_int32") int debug );
public static native PIX pixMakeAlphaFromMask( PIX pixs, @Cast("l_int32") int dist, @Cast("BOX**") PointerPointer pbox );
public static native PIX pixMakeAlphaFromMask( PIX pixs, @Cast("l_int32") int dist, @ByPtrPtr BOX pbox );
public static native @Cast("l_int32") int pixGetColorNearMaskBoundary( PIX pixs, PIX pixm, BOX box, @Cast("l_int32") int dist, @Cast("l_uint32*") IntPointer pval, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int pixGetColorNearMaskBoundary( PIX pixs, PIX pixm, BOX box, @Cast("l_int32") int dist, @Cast("l_uint32*") IntBuffer pval, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int pixGetColorNearMaskBoundary( PIX pixs, PIX pixm, BOX box, @Cast("l_int32") int dist, @Cast("l_uint32*") int[] pval, @Cast("l_int32") int debug );
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
public static native NUMA pixaCountPixels( PIXA pixa );
public static native @Cast("l_int32") int pixCountPixels( PIX pix, @Cast("l_int32*") IntPointer pcount, @Cast("l_int32*") IntPointer tab8 );
public static native @Cast("l_int32") int pixCountPixels( PIX pix, @Cast("l_int32*") IntBuffer pcount, @Cast("l_int32*") IntBuffer tab8 );
public static native @Cast("l_int32") int pixCountPixels( PIX pix, @Cast("l_int32*") int[] pcount, @Cast("l_int32*") int[] tab8 );
public static native NUMA pixCountByRow( PIX pix, BOX box );
public static native NUMA pixCountByColumn( PIX pix, BOX box );
public static native NUMA pixCountPixelsByRow( PIX pix, @Cast("l_int32*") IntPointer tab8 );
public static native NUMA pixCountPixelsByRow( PIX pix, @Cast("l_int32*") IntBuffer tab8 );
public static native NUMA pixCountPixelsByRow( PIX pix, @Cast("l_int32*") int[] tab8 );
public static native NUMA pixCountPixelsByColumn( PIX pix );
public static native @Cast("l_int32") int pixCountPixelsInRow( PIX pix, @Cast("l_int32") int row, @Cast("l_int32*") IntPointer pcount, @Cast("l_int32*") IntPointer tab8 );
public static native @Cast("l_int32") int pixCountPixelsInRow( PIX pix, @Cast("l_int32") int row, @Cast("l_int32*") IntBuffer pcount, @Cast("l_int32*") IntBuffer tab8 );
public static native @Cast("l_int32") int pixCountPixelsInRow( PIX pix, @Cast("l_int32") int row, @Cast("l_int32*") int[] pcount, @Cast("l_int32*") int[] tab8 );
public static native NUMA pixGetMomentByColumn( PIX pix, @Cast("l_int32") int order );
public static native @Cast("l_int32") int pixThresholdPixelSum( PIX pix, @Cast("l_int32") int thresh, @Cast("l_int32*") IntPointer pabove, @Cast("l_int32*") IntPointer tab8 );
public static native @Cast("l_int32") int pixThresholdPixelSum( PIX pix, @Cast("l_int32") int thresh, @Cast("l_int32*") IntBuffer pabove, @Cast("l_int32*") IntBuffer tab8 );
public static native @Cast("l_int32") int pixThresholdPixelSum( PIX pix, @Cast("l_int32") int thresh, @Cast("l_int32*") int[] pabove, @Cast("l_int32*") int[] tab8 );
public static native @Cast("l_int32*") IntPointer makePixelSumTab8( );
public static native @Cast("l_int32*") IntPointer makePixelCentroidTab8( );
public static native NUMA pixAverageByRow( PIX pix, BOX box, @Cast("l_int32") int type );
public static native NUMA pixAverageByColumn( PIX pix, BOX box, @Cast("l_int32") int type );
public static native @Cast("l_int32") int pixAverageInRect( PIX pix, BOX box, @Cast("l_float32*") FloatPointer pave );
public static native @Cast("l_int32") int pixAverageInRect( PIX pix, BOX box, @Cast("l_float32*") FloatBuffer pave );
public static native @Cast("l_int32") int pixAverageInRect( PIX pix, BOX box, @Cast("l_float32*") float[] pave );
public static native NUMA pixVarianceByRow( PIX pix, BOX box );
public static native NUMA pixVarianceByColumn( PIX pix, BOX box );
public static native @Cast("l_int32") int pixVarianceInRect( PIX pix, BOX box, @Cast("l_float32*") FloatPointer prootvar );
public static native @Cast("l_int32") int pixVarianceInRect( PIX pix, BOX box, @Cast("l_float32*") FloatBuffer prootvar );
public static native @Cast("l_int32") int pixVarianceInRect( PIX pix, BOX box, @Cast("l_float32*") float[] prootvar );
public static native NUMA pixAbsDiffByRow( PIX pix, BOX box );
public static native NUMA pixAbsDiffByColumn( PIX pix, BOX box );
public static native @Cast("l_int32") int pixAbsDiffInRect( PIX pix, BOX box, @Cast("l_int32") int dir, @Cast("l_float32*") FloatPointer pabsdiff );
public static native @Cast("l_int32") int pixAbsDiffInRect( PIX pix, BOX box, @Cast("l_int32") int dir, @Cast("l_float32*") FloatBuffer pabsdiff );
public static native @Cast("l_int32") int pixAbsDiffInRect( PIX pix, BOX box, @Cast("l_int32") int dir, @Cast("l_float32*") float[] pabsdiff );
public static native @Cast("l_int32") int pixAbsDiffOnLine( PIX pix, @Cast("l_int32") int x1, @Cast("l_int32") int y1, @Cast("l_int32") int x2, @Cast("l_int32") int y2, @Cast("l_float32*") FloatPointer pabsdiff );
public static native @Cast("l_int32") int pixAbsDiffOnLine( PIX pix, @Cast("l_int32") int x1, @Cast("l_int32") int y1, @Cast("l_int32") int x2, @Cast("l_int32") int y2, @Cast("l_float32*") FloatBuffer pabsdiff );
public static native @Cast("l_int32") int pixAbsDiffOnLine( PIX pix, @Cast("l_int32") int x1, @Cast("l_int32") int y1, @Cast("l_int32") int x2, @Cast("l_int32") int y2, @Cast("l_float32*") float[] pabsdiff );
public static native @Cast("l_int32") int pixCountArbInRect( PIX pixs, BOX box, @Cast("l_int32") int val, @Cast("l_int32") int factor, @Cast("l_int32*") IntPointer pcount );
public static native @Cast("l_int32") int pixCountArbInRect( PIX pixs, BOX box, @Cast("l_int32") int val, @Cast("l_int32") int factor, @Cast("l_int32*") IntBuffer pcount );
public static native @Cast("l_int32") int pixCountArbInRect( PIX pixs, BOX box, @Cast("l_int32") int val, @Cast("l_int32") int factor, @Cast("l_int32*") int[] pcount );
public static native PIX pixMirroredTiling( PIX pixs, @Cast("l_int32") int w, @Cast("l_int32") int h );
public static native @Cast("l_int32") int pixFindRepCloseTile( PIX pixs, BOX box, @Cast("l_int32") int searchdir, @Cast("l_int32") int mindist, @Cast("l_int32") int tsize, @Cast("l_int32") int ntiles, @Cast("BOX**") PointerPointer pboxtile, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int pixFindRepCloseTile( PIX pixs, BOX box, @Cast("l_int32") int searchdir, @Cast("l_int32") int mindist, @Cast("l_int32") int tsize, @Cast("l_int32") int ntiles, @ByPtrPtr BOX pboxtile, @Cast("l_int32") int debug );
public static native NUMA pixGetGrayHistogram( PIX pixs, @Cast("l_int32") int factor );
public static native NUMA pixGetGrayHistogramMasked( PIX pixs, PIX pixm, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int factor );
public static native NUMA pixGetGrayHistogramInRect( PIX pixs, BOX box, @Cast("l_int32") int factor );
public static native @Cast("l_int32") int pixGetColorHistogram( PIX pixs, @Cast("l_int32") int factor, @Cast("NUMA**") PointerPointer pnar, @Cast("NUMA**") PointerPointer pnag, @Cast("NUMA**") PointerPointer pnab );
public static native @Cast("l_int32") int pixGetColorHistogram( PIX pixs, @Cast("l_int32") int factor, @ByPtrPtr NUMA pnar, @ByPtrPtr NUMA pnag, @ByPtrPtr NUMA pnab );
public static native @Cast("l_int32") int pixGetColorHistogramMasked( PIX pixs, PIX pixm, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int factor, @Cast("NUMA**") PointerPointer pnar, @Cast("NUMA**") PointerPointer pnag, @Cast("NUMA**") PointerPointer pnab );
public static native @Cast("l_int32") int pixGetColorHistogramMasked( PIX pixs, PIX pixm, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int factor, @ByPtrPtr NUMA pnar, @ByPtrPtr NUMA pnag, @ByPtrPtr NUMA pnab );
public static native NUMA pixGetCmapHistogram( PIX pixs, @Cast("l_int32") int factor );
public static native NUMA pixGetCmapHistogramMasked( PIX pixs, PIX pixm, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int factor );
public static native NUMA pixGetCmapHistogramInRect( PIX pixs, BOX box, @Cast("l_int32") int factor );
public static native @Cast("l_int32") int pixGetRankValue( PIX pixs, @Cast("l_int32") int factor, @Cast("l_float32") float rank, @Cast("l_uint32*") IntPointer pvalue );
public static native @Cast("l_int32") int pixGetRankValue( PIX pixs, @Cast("l_int32") int factor, @Cast("l_float32") float rank, @Cast("l_uint32*") IntBuffer pvalue );
public static native @Cast("l_int32") int pixGetRankValue( PIX pixs, @Cast("l_int32") int factor, @Cast("l_float32") float rank, @Cast("l_uint32*") int[] pvalue );
public static native @Cast("l_int32") int pixGetRankValueMaskedRGB( PIX pixs, PIX pixm, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int factor, @Cast("l_float32") float rank, @Cast("l_float32*") FloatPointer prval, @Cast("l_float32*") FloatPointer pgval, @Cast("l_float32*") FloatPointer pbval );
public static native @Cast("l_int32") int pixGetRankValueMaskedRGB( PIX pixs, PIX pixm, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int factor, @Cast("l_float32") float rank, @Cast("l_float32*") FloatBuffer prval, @Cast("l_float32*") FloatBuffer pgval, @Cast("l_float32*") FloatBuffer pbval );
public static native @Cast("l_int32") int pixGetRankValueMaskedRGB( PIX pixs, PIX pixm, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int factor, @Cast("l_float32") float rank, @Cast("l_float32*") float[] prval, @Cast("l_float32*") float[] pgval, @Cast("l_float32*") float[] pbval );
public static native @Cast("l_int32") int pixGetRankValueMasked( PIX pixs, PIX pixm, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int factor, @Cast("l_float32") float rank, @Cast("l_float32*") FloatPointer pval, @Cast("NUMA**") PointerPointer pna );
public static native @Cast("l_int32") int pixGetRankValueMasked( PIX pixs, PIX pixm, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int factor, @Cast("l_float32") float rank, @Cast("l_float32*") FloatPointer pval, @ByPtrPtr NUMA pna );
public static native @Cast("l_int32") int pixGetRankValueMasked( PIX pixs, PIX pixm, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int factor, @Cast("l_float32") float rank, @Cast("l_float32*") FloatBuffer pval, @ByPtrPtr NUMA pna );
public static native @Cast("l_int32") int pixGetRankValueMasked( PIX pixs, PIX pixm, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int factor, @Cast("l_float32") float rank, @Cast("l_float32*") float[] pval, @ByPtrPtr NUMA pna );
public static native @Cast("l_int32") int pixGetAverageValue( PIX pixs, @Cast("l_int32") int factor, @Cast("l_int32") int type, @Cast("l_uint32*") IntPointer pvalue );
public static native @Cast("l_int32") int pixGetAverageValue( PIX pixs, @Cast("l_int32") int factor, @Cast("l_int32") int type, @Cast("l_uint32*") IntBuffer pvalue );
public static native @Cast("l_int32") int pixGetAverageValue( PIX pixs, @Cast("l_int32") int factor, @Cast("l_int32") int type, @Cast("l_uint32*") int[] pvalue );
public static native @Cast("l_int32") int pixGetAverageMaskedRGB( PIX pixs, PIX pixm, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int factor, @Cast("l_int32") int type, @Cast("l_float32*") FloatPointer prval, @Cast("l_float32*") FloatPointer pgval, @Cast("l_float32*") FloatPointer pbval );
public static native @Cast("l_int32") int pixGetAverageMaskedRGB( PIX pixs, PIX pixm, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int factor, @Cast("l_int32") int type, @Cast("l_float32*") FloatBuffer prval, @Cast("l_float32*") FloatBuffer pgval, @Cast("l_float32*") FloatBuffer pbval );
public static native @Cast("l_int32") int pixGetAverageMaskedRGB( PIX pixs, PIX pixm, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int factor, @Cast("l_int32") int type, @Cast("l_float32*") float[] prval, @Cast("l_float32*") float[] pgval, @Cast("l_float32*") float[] pbval );
public static native @Cast("l_int32") int pixGetAverageMasked( PIX pixs, PIX pixm, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int factor, @Cast("l_int32") int type, @Cast("l_float32*") FloatPointer pval );
public static native @Cast("l_int32") int pixGetAverageMasked( PIX pixs, PIX pixm, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int factor, @Cast("l_int32") int type, @Cast("l_float32*") FloatBuffer pval );
public static native @Cast("l_int32") int pixGetAverageMasked( PIX pixs, PIX pixm, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int factor, @Cast("l_int32") int type, @Cast("l_float32*") float[] pval );
public static native @Cast("l_int32") int pixGetAverageTiledRGB( PIX pixs, @Cast("l_int32") int sx, @Cast("l_int32") int sy, @Cast("l_int32") int type, @Cast("PIX**") PointerPointer ppixr, @Cast("PIX**") PointerPointer ppixg, @Cast("PIX**") PointerPointer ppixb );
public static native @Cast("l_int32") int pixGetAverageTiledRGB( PIX pixs, @Cast("l_int32") int sx, @Cast("l_int32") int sy, @Cast("l_int32") int type, @ByPtrPtr PIX ppixr, @ByPtrPtr PIX ppixg, @ByPtrPtr PIX ppixb );
public static native PIX pixGetAverageTiled( PIX pixs, @Cast("l_int32") int sx, @Cast("l_int32") int sy, @Cast("l_int32") int type );
public static native @Cast("l_int32") int pixRowStats( PIX pixs, BOX box, @Cast("NUMA**") PointerPointer pnamean, @Cast("NUMA**") PointerPointer pnamedian, @Cast("NUMA**") PointerPointer pnamode, @Cast("NUMA**") PointerPointer pnamodecount, @Cast("NUMA**") PointerPointer pnavar, @Cast("NUMA**") PointerPointer pnarootvar );
public static native @Cast("l_int32") int pixRowStats( PIX pixs, BOX box, @ByPtrPtr NUMA pnamean, @ByPtrPtr NUMA pnamedian, @ByPtrPtr NUMA pnamode, @ByPtrPtr NUMA pnamodecount, @ByPtrPtr NUMA pnavar, @ByPtrPtr NUMA pnarootvar );
public static native @Cast("l_int32") int pixColumnStats( PIX pixs, BOX box, @Cast("NUMA**") PointerPointer pnamean, @Cast("NUMA**") PointerPointer pnamedian, @Cast("NUMA**") PointerPointer pnamode, @Cast("NUMA**") PointerPointer pnamodecount, @Cast("NUMA**") PointerPointer pnavar, @Cast("NUMA**") PointerPointer pnarootvar );
public static native @Cast("l_int32") int pixColumnStats( PIX pixs, BOX box, @ByPtrPtr NUMA pnamean, @ByPtrPtr NUMA pnamedian, @ByPtrPtr NUMA pnamode, @ByPtrPtr NUMA pnamodecount, @ByPtrPtr NUMA pnavar, @ByPtrPtr NUMA pnarootvar );
public static native @Cast("l_int32") int pixGetComponentRange( PIX pixs, @Cast("l_int32") int factor, @Cast("l_int32") int color, @Cast("l_int32*") IntPointer pminval, @Cast("l_int32*") IntPointer pmaxval );
public static native @Cast("l_int32") int pixGetComponentRange( PIX pixs, @Cast("l_int32") int factor, @Cast("l_int32") int color, @Cast("l_int32*") IntBuffer pminval, @Cast("l_int32*") IntBuffer pmaxval );
public static native @Cast("l_int32") int pixGetComponentRange( PIX pixs, @Cast("l_int32") int factor, @Cast("l_int32") int color, @Cast("l_int32*") int[] pminval, @Cast("l_int32*") int[] pmaxval );
public static native @Cast("l_int32") int pixGetExtremeValue( PIX pixs, @Cast("l_int32") int factor, @Cast("l_int32") int type, @Cast("l_int32*") IntPointer prval, @Cast("l_int32*") IntPointer pgval, @Cast("l_int32*") IntPointer pbval, @Cast("l_int32*") IntPointer pgrayval );
public static native @Cast("l_int32") int pixGetExtremeValue( PIX pixs, @Cast("l_int32") int factor, @Cast("l_int32") int type, @Cast("l_int32*") IntBuffer prval, @Cast("l_int32*") IntBuffer pgval, @Cast("l_int32*") IntBuffer pbval, @Cast("l_int32*") IntBuffer pgrayval );
public static native @Cast("l_int32") int pixGetExtremeValue( PIX pixs, @Cast("l_int32") int factor, @Cast("l_int32") int type, @Cast("l_int32*") int[] prval, @Cast("l_int32*") int[] pgval, @Cast("l_int32*") int[] pbval, @Cast("l_int32*") int[] pgrayval );
public static native @Cast("l_int32") int pixGetMaxValueInRect( PIX pixs, BOX box, @Cast("l_uint32*") IntPointer pmaxval, @Cast("l_int32*") IntPointer pxmax, @Cast("l_int32*") IntPointer pymax );
public static native @Cast("l_int32") int pixGetMaxValueInRect( PIX pixs, BOX box, @Cast("l_uint32*") IntBuffer pmaxval, @Cast("l_int32*") IntBuffer pxmax, @Cast("l_int32*") IntBuffer pymax );
public static native @Cast("l_int32") int pixGetMaxValueInRect( PIX pixs, BOX box, @Cast("l_uint32*") int[] pmaxval, @Cast("l_int32*") int[] pxmax, @Cast("l_int32*") int[] pymax );
public static native @Cast("l_int32") int pixGetBinnedComponentRange( PIX pixs, @Cast("l_int32") int nbins, @Cast("l_int32") int factor, @Cast("l_int32") int color, @Cast("l_int32*") IntPointer pminval, @Cast("l_int32*") IntPointer pmaxval, @Cast("l_uint32**") PointerPointer pcarray, @Cast("const char*") BytePointer fontdir );
public static native @Cast("l_int32") int pixGetBinnedComponentRange( PIX pixs, @Cast("l_int32") int nbins, @Cast("l_int32") int factor, @Cast("l_int32") int color, @Cast("l_int32*") IntPointer pminval, @Cast("l_int32*") IntPointer pmaxval, @Cast("l_uint32**") @ByPtrPtr IntPointer pcarray, @Cast("const char*") BytePointer fontdir );
public static native @Cast("l_int32") int pixGetBinnedComponentRange( PIX pixs, @Cast("l_int32") int nbins, @Cast("l_int32") int factor, @Cast("l_int32") int color, @Cast("l_int32*") IntBuffer pminval, @Cast("l_int32*") IntBuffer pmaxval, @Cast("l_uint32**") @ByPtrPtr IntBuffer pcarray, String fontdir );
public static native @Cast("l_int32") int pixGetBinnedComponentRange( PIX pixs, @Cast("l_int32") int nbins, @Cast("l_int32") int factor, @Cast("l_int32") int color, @Cast("l_int32*") int[] pminval, @Cast("l_int32*") int[] pmaxval, @Cast("l_uint32**") @ByPtrPtr int[] pcarray, @Cast("const char*") BytePointer fontdir );
public static native @Cast("l_int32") int pixGetBinnedComponentRange( PIX pixs, @Cast("l_int32") int nbins, @Cast("l_int32") int factor, @Cast("l_int32") int color, @Cast("l_int32*") IntPointer pminval, @Cast("l_int32*") IntPointer pmaxval, @Cast("l_uint32**") @ByPtrPtr IntPointer pcarray, String fontdir );
public static native @Cast("l_int32") int pixGetBinnedComponentRange( PIX pixs, @Cast("l_int32") int nbins, @Cast("l_int32") int factor, @Cast("l_int32") int color, @Cast("l_int32*") IntBuffer pminval, @Cast("l_int32*") IntBuffer pmaxval, @Cast("l_uint32**") @ByPtrPtr IntBuffer pcarray, @Cast("const char*") BytePointer fontdir );
public static native @Cast("l_int32") int pixGetBinnedComponentRange( PIX pixs, @Cast("l_int32") int nbins, @Cast("l_int32") int factor, @Cast("l_int32") int color, @Cast("l_int32*") int[] pminval, @Cast("l_int32*") int[] pmaxval, @Cast("l_uint32**") @ByPtrPtr int[] pcarray, String fontdir );
public static native @Cast("l_int32") int pixGetRankColorArray( PIX pixs, @Cast("l_int32") int nbins, @Cast("l_int32") int type, @Cast("l_int32") int factor, @Cast("l_uint32**") PointerPointer pcarray, @Cast("l_int32") int debugflag, @Cast("const char*") BytePointer fontdir );
public static native @Cast("l_int32") int pixGetRankColorArray( PIX pixs, @Cast("l_int32") int nbins, @Cast("l_int32") int type, @Cast("l_int32") int factor, @Cast("l_uint32**") @ByPtrPtr IntPointer pcarray, @Cast("l_int32") int debugflag, @Cast("const char*") BytePointer fontdir );
public static native @Cast("l_int32") int pixGetRankColorArray( PIX pixs, @Cast("l_int32") int nbins, @Cast("l_int32") int type, @Cast("l_int32") int factor, @Cast("l_uint32**") @ByPtrPtr IntBuffer pcarray, @Cast("l_int32") int debugflag, String fontdir );
public static native @Cast("l_int32") int pixGetRankColorArray( PIX pixs, @Cast("l_int32") int nbins, @Cast("l_int32") int type, @Cast("l_int32") int factor, @Cast("l_uint32**") @ByPtrPtr int[] pcarray, @Cast("l_int32") int debugflag, @Cast("const char*") BytePointer fontdir );
public static native @Cast("l_int32") int pixGetRankColorArray( PIX pixs, @Cast("l_int32") int nbins, @Cast("l_int32") int type, @Cast("l_int32") int factor, @Cast("l_uint32**") @ByPtrPtr IntPointer pcarray, @Cast("l_int32") int debugflag, String fontdir );
public static native @Cast("l_int32") int pixGetRankColorArray( PIX pixs, @Cast("l_int32") int nbins, @Cast("l_int32") int type, @Cast("l_int32") int factor, @Cast("l_uint32**") @ByPtrPtr IntBuffer pcarray, @Cast("l_int32") int debugflag, @Cast("const char*") BytePointer fontdir );
public static native @Cast("l_int32") int pixGetRankColorArray( PIX pixs, @Cast("l_int32") int nbins, @Cast("l_int32") int type, @Cast("l_int32") int factor, @Cast("l_uint32**") @ByPtrPtr int[] pcarray, @Cast("l_int32") int debugflag, String fontdir );
public static native @Cast("l_int32") int pixGetBinnedColor( PIX pixs, PIX pixg, @Cast("l_int32") int factor, @Cast("l_int32") int nbins, NUMA nalut, @Cast("l_uint32**") PointerPointer pcarray, @Cast("l_int32") int debugflag );
public static native @Cast("l_int32") int pixGetBinnedColor( PIX pixs, PIX pixg, @Cast("l_int32") int factor, @Cast("l_int32") int nbins, NUMA nalut, @Cast("l_uint32**") @ByPtrPtr IntPointer pcarray, @Cast("l_int32") int debugflag );
public static native @Cast("l_int32") int pixGetBinnedColor( PIX pixs, PIX pixg, @Cast("l_int32") int factor, @Cast("l_int32") int nbins, NUMA nalut, @Cast("l_uint32**") @ByPtrPtr IntBuffer pcarray, @Cast("l_int32") int debugflag );
public static native @Cast("l_int32") int pixGetBinnedColor( PIX pixs, PIX pixg, @Cast("l_int32") int factor, @Cast("l_int32") int nbins, NUMA nalut, @Cast("l_uint32**") @ByPtrPtr int[] pcarray, @Cast("l_int32") int debugflag );
public static native PIX pixDisplayColorArray( @Cast("l_uint32*") IntPointer carray, @Cast("l_int32") int ncolors, @Cast("l_int32") int side, @Cast("l_int32") int ncols, @Cast("const char*") BytePointer fontdir );
public static native PIX pixDisplayColorArray( @Cast("l_uint32*") IntBuffer carray, @Cast("l_int32") int ncolors, @Cast("l_int32") int side, @Cast("l_int32") int ncols, String fontdir );
public static native PIX pixDisplayColorArray( @Cast("l_uint32*") int[] carray, @Cast("l_int32") int ncolors, @Cast("l_int32") int side, @Cast("l_int32") int ncols, @Cast("const char*") BytePointer fontdir );
public static native PIX pixDisplayColorArray( @Cast("l_uint32*") IntPointer carray, @Cast("l_int32") int ncolors, @Cast("l_int32") int side, @Cast("l_int32") int ncols, String fontdir );
public static native PIX pixDisplayColorArray( @Cast("l_uint32*") IntBuffer carray, @Cast("l_int32") int ncolors, @Cast("l_int32") int side, @Cast("l_int32") int ncols, @Cast("const char*") BytePointer fontdir );
public static native PIX pixDisplayColorArray( @Cast("l_uint32*") int[] carray, @Cast("l_int32") int ncolors, @Cast("l_int32") int side, @Cast("l_int32") int ncols, String fontdir );
public static native PIX pixRankBinByStrip( PIX pixs, @Cast("l_int32") int direction, @Cast("l_int32") int size, @Cast("l_int32") int nbins, @Cast("l_int32") int type );
public static native PIX pixaGetAlignedStats( PIXA pixa, @Cast("l_int32") int type, @Cast("l_int32") int nbins, @Cast("l_int32") int thresh );
public static native @Cast("l_int32") int pixaExtractColumnFromEachPix( PIXA pixa, @Cast("l_int32") int col, PIX pixd );
public static native @Cast("l_int32") int pixGetRowStats( PIX pixs, @Cast("l_int32") int type, @Cast("l_int32") int nbins, @Cast("l_int32") int thresh, @Cast("l_float32*") FloatPointer colvect );
public static native @Cast("l_int32") int pixGetRowStats( PIX pixs, @Cast("l_int32") int type, @Cast("l_int32") int nbins, @Cast("l_int32") int thresh, @Cast("l_float32*") FloatBuffer colvect );
public static native @Cast("l_int32") int pixGetRowStats( PIX pixs, @Cast("l_int32") int type, @Cast("l_int32") int nbins, @Cast("l_int32") int thresh, @Cast("l_float32*") float[] colvect );
public static native @Cast("l_int32") int pixGetColumnStats( PIX pixs, @Cast("l_int32") int type, @Cast("l_int32") int nbins, @Cast("l_int32") int thresh, @Cast("l_float32*") FloatPointer rowvect );
public static native @Cast("l_int32") int pixGetColumnStats( PIX pixs, @Cast("l_int32") int type, @Cast("l_int32") int nbins, @Cast("l_int32") int thresh, @Cast("l_float32*") FloatBuffer rowvect );
public static native @Cast("l_int32") int pixGetColumnStats( PIX pixs, @Cast("l_int32") int type, @Cast("l_int32") int nbins, @Cast("l_int32") int thresh, @Cast("l_float32*") float[] rowvect );
public static native @Cast("l_int32") int pixSetPixelColumn( PIX pix, @Cast("l_int32") int col, @Cast("l_float32*") FloatPointer colvect );
public static native @Cast("l_int32") int pixSetPixelColumn( PIX pix, @Cast("l_int32") int col, @Cast("l_float32*") FloatBuffer colvect );
public static native @Cast("l_int32") int pixSetPixelColumn( PIX pix, @Cast("l_int32") int col, @Cast("l_float32*") float[] colvect );
public static native @Cast("l_int32") int pixThresholdForFgBg( PIX pixs, @Cast("l_int32") int factor, @Cast("l_int32") int thresh, @Cast("l_int32*") IntPointer pfgval, @Cast("l_int32*") IntPointer pbgval );
public static native @Cast("l_int32") int pixThresholdForFgBg( PIX pixs, @Cast("l_int32") int factor, @Cast("l_int32") int thresh, @Cast("l_int32*") IntBuffer pfgval, @Cast("l_int32*") IntBuffer pbgval );
public static native @Cast("l_int32") int pixThresholdForFgBg( PIX pixs, @Cast("l_int32") int factor, @Cast("l_int32") int thresh, @Cast("l_int32*") int[] pfgval, @Cast("l_int32*") int[] pbgval );
public static native @Cast("l_int32") int pixSplitDistributionFgBg( PIX pixs, @Cast("l_float32") float scorefract, @Cast("l_int32") int factor, @Cast("l_int32*") IntPointer pthresh, @Cast("l_int32*") IntPointer pfgval, @Cast("l_int32*") IntPointer pbgval, @Cast("l_int32") int debugflag );
public static native @Cast("l_int32") int pixSplitDistributionFgBg( PIX pixs, @Cast("l_float32") float scorefract, @Cast("l_int32") int factor, @Cast("l_int32*") IntBuffer pthresh, @Cast("l_int32*") IntBuffer pfgval, @Cast("l_int32*") IntBuffer pbgval, @Cast("l_int32") int debugflag );
public static native @Cast("l_int32") int pixSplitDistributionFgBg( PIX pixs, @Cast("l_float32") float scorefract, @Cast("l_int32") int factor, @Cast("l_int32*") int[] pthresh, @Cast("l_int32*") int[] pfgval, @Cast("l_int32*") int[] pbgval, @Cast("l_int32") int debugflag );
public static native @Cast("l_int32") int pixaFindDimensions( PIXA pixa, @Cast("NUMA**") PointerPointer pnaw, @Cast("NUMA**") PointerPointer pnah );
public static native @Cast("l_int32") int pixaFindDimensions( PIXA pixa, @ByPtrPtr NUMA pnaw, @ByPtrPtr NUMA pnah );
public static native @Cast("l_int32") int pixFindAreaPerimRatio( PIX pixs, @Cast("l_int32*") IntPointer tab, @Cast("l_float32*") FloatPointer pfract );
public static native @Cast("l_int32") int pixFindAreaPerimRatio( PIX pixs, @Cast("l_int32*") IntBuffer tab, @Cast("l_float32*") FloatBuffer pfract );
public static native @Cast("l_int32") int pixFindAreaPerimRatio( PIX pixs, @Cast("l_int32*") int[] tab, @Cast("l_float32*") float[] pfract );
public static native NUMA pixaFindPerimToAreaRatio( PIXA pixa );
public static native @Cast("l_int32") int pixFindPerimToAreaRatio( PIX pixs, @Cast("l_int32*") IntPointer tab, @Cast("l_float32*") FloatPointer pfract );
public static native @Cast("l_int32") int pixFindPerimToAreaRatio( PIX pixs, @Cast("l_int32*") IntBuffer tab, @Cast("l_float32*") FloatBuffer pfract );
public static native @Cast("l_int32") int pixFindPerimToAreaRatio( PIX pixs, @Cast("l_int32*") int[] tab, @Cast("l_float32*") float[] pfract );
public static native NUMA pixaFindPerimSizeRatio( PIXA pixa );
public static native @Cast("l_int32") int pixFindPerimSizeRatio( PIX pixs, @Cast("l_int32*") IntPointer tab, @Cast("l_float32*") FloatPointer pratio );
public static native @Cast("l_int32") int pixFindPerimSizeRatio( PIX pixs, @Cast("l_int32*") IntBuffer tab, @Cast("l_float32*") FloatBuffer pratio );
public static native @Cast("l_int32") int pixFindPerimSizeRatio( PIX pixs, @Cast("l_int32*") int[] tab, @Cast("l_float32*") float[] pratio );
public static native NUMA pixaFindAreaFraction( PIXA pixa );
public static native @Cast("l_int32") int pixFindAreaFraction( PIX pixs, @Cast("l_int32*") IntPointer tab, @Cast("l_float32*") FloatPointer pfract );
public static native @Cast("l_int32") int pixFindAreaFraction( PIX pixs, @Cast("l_int32*") IntBuffer tab, @Cast("l_float32*") FloatBuffer pfract );
public static native @Cast("l_int32") int pixFindAreaFraction( PIX pixs, @Cast("l_int32*") int[] tab, @Cast("l_float32*") float[] pfract );
public static native NUMA pixaFindAreaFractionMasked( PIXA pixa, PIX pixm, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int pixFindAreaFractionMasked( PIX pixs, BOX box, PIX pixm, @Cast("l_int32*") IntPointer tab, @Cast("l_float32*") FloatPointer pfract );
public static native @Cast("l_int32") int pixFindAreaFractionMasked( PIX pixs, BOX box, PIX pixm, @Cast("l_int32*") IntBuffer tab, @Cast("l_float32*") FloatBuffer pfract );
public static native @Cast("l_int32") int pixFindAreaFractionMasked( PIX pixs, BOX box, PIX pixm, @Cast("l_int32*") int[] tab, @Cast("l_float32*") float[] pfract );
public static native NUMA pixaFindWidthHeightRatio( PIXA pixa );
public static native NUMA pixaFindWidthHeightProduct( PIXA pixa );
public static native @Cast("l_int32") int pixFindOverlapFraction( PIX pixs1, PIX pixs2, @Cast("l_int32") int x2, @Cast("l_int32") int y2, @Cast("l_int32*") IntPointer tab, @Cast("l_float32*") FloatPointer pratio, @Cast("l_int32*") IntPointer pnoverlap );
public static native @Cast("l_int32") int pixFindOverlapFraction( PIX pixs1, PIX pixs2, @Cast("l_int32") int x2, @Cast("l_int32") int y2, @Cast("l_int32*") IntBuffer tab, @Cast("l_float32*") FloatBuffer pratio, @Cast("l_int32*") IntBuffer pnoverlap );
public static native @Cast("l_int32") int pixFindOverlapFraction( PIX pixs1, PIX pixs2, @Cast("l_int32") int x2, @Cast("l_int32") int y2, @Cast("l_int32*") int[] tab, @Cast("l_float32*") float[] pratio, @Cast("l_int32*") int[] pnoverlap );
public static native BOXA pixFindRectangleComps( PIX pixs, @Cast("l_int32") int dist, @Cast("l_int32") int minw, @Cast("l_int32") int minh );
public static native @Cast("l_int32") int pixConformsToRectangle( PIX pixs, BOX box, @Cast("l_int32") int dist, @Cast("l_int32*") IntPointer pconforms );
public static native @Cast("l_int32") int pixConformsToRectangle( PIX pixs, BOX box, @Cast("l_int32") int dist, @Cast("l_int32*") IntBuffer pconforms );
public static native @Cast("l_int32") int pixConformsToRectangle( PIX pixs, BOX box, @Cast("l_int32") int dist, @Cast("l_int32*") int[] pconforms );
public static native PIXA pixClipRectangles( PIX pixs, BOXA boxa );
public static native PIX pixClipRectangle( PIX pixs, BOX box, @Cast("BOX**") PointerPointer pboxc );
public static native PIX pixClipRectangle( PIX pixs, BOX box, @ByPtrPtr BOX pboxc );
public static native PIX pixClipMasked( PIX pixs, PIX pixm, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_uint32") int outval );
public static native @Cast("l_int32") int pixCropToMatch( PIX pixs1, PIX pixs2, @Cast("PIX**") PointerPointer ppixd1, @Cast("PIX**") PointerPointer ppixd2 );
public static native @Cast("l_int32") int pixCropToMatch( PIX pixs1, PIX pixs2, @ByPtrPtr PIX ppixd1, @ByPtrPtr PIX ppixd2 );
public static native PIX pixCropToSize( PIX pixs, @Cast("l_int32") int w, @Cast("l_int32") int h );
public static native PIX pixResizeToMatch( PIX pixs, PIX pixt, @Cast("l_int32") int w, @Cast("l_int32") int h );
public static native PIX pixMakeFrameMask( @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_float32") float hf1, @Cast("l_float32") float hf2, @Cast("l_float32") float vf1, @Cast("l_float32") float vf2 );
public static native @Cast("l_int32") int pixClipToForeground( PIX pixs, @Cast("PIX**") PointerPointer ppixd, @Cast("BOX**") PointerPointer pbox );
public static native @Cast("l_int32") int pixClipToForeground( PIX pixs, @ByPtrPtr PIX ppixd, @ByPtrPtr BOX pbox );
public static native @Cast("l_int32") int pixTestClipToForeground( PIX pixs, @Cast("l_int32*") IntPointer pcanclip );
public static native @Cast("l_int32") int pixTestClipToForeground( PIX pixs, @Cast("l_int32*") IntBuffer pcanclip );
public static native @Cast("l_int32") int pixTestClipToForeground( PIX pixs, @Cast("l_int32*") int[] pcanclip );
public static native @Cast("l_int32") int pixClipBoxToForeground( PIX pixs, BOX boxs, @Cast("PIX**") PointerPointer ppixd, @Cast("BOX**") PointerPointer pboxd );
public static native @Cast("l_int32") int pixClipBoxToForeground( PIX pixs, BOX boxs, @ByPtrPtr PIX ppixd, @ByPtrPtr BOX pboxd );
public static native @Cast("l_int32") int pixScanForForeground( PIX pixs, BOX box, @Cast("l_int32") int scanflag, @Cast("l_int32*") IntPointer ploc );
public static native @Cast("l_int32") int pixScanForForeground( PIX pixs, BOX box, @Cast("l_int32") int scanflag, @Cast("l_int32*") IntBuffer ploc );
public static native @Cast("l_int32") int pixScanForForeground( PIX pixs, BOX box, @Cast("l_int32") int scanflag, @Cast("l_int32*") int[] ploc );
public static native @Cast("l_int32") int pixClipBoxToEdges( PIX pixs, BOX boxs, @Cast("l_int32") int lowthresh, @Cast("l_int32") int highthresh, @Cast("l_int32") int maxwidth, @Cast("l_int32") int factor, @Cast("PIX**") PointerPointer ppixd, @Cast("BOX**") PointerPointer pboxd );
public static native @Cast("l_int32") int pixClipBoxToEdges( PIX pixs, BOX boxs, @Cast("l_int32") int lowthresh, @Cast("l_int32") int highthresh, @Cast("l_int32") int maxwidth, @Cast("l_int32") int factor, @ByPtrPtr PIX ppixd, @ByPtrPtr BOX pboxd );
public static native @Cast("l_int32") int pixScanForEdge( PIX pixs, BOX box, @Cast("l_int32") int lowthresh, @Cast("l_int32") int highthresh, @Cast("l_int32") int maxwidth, @Cast("l_int32") int factor, @Cast("l_int32") int scanflag, @Cast("l_int32*") IntPointer ploc );
public static native @Cast("l_int32") int pixScanForEdge( PIX pixs, BOX box, @Cast("l_int32") int lowthresh, @Cast("l_int32") int highthresh, @Cast("l_int32") int maxwidth, @Cast("l_int32") int factor, @Cast("l_int32") int scanflag, @Cast("l_int32*") IntBuffer ploc );
public static native @Cast("l_int32") int pixScanForEdge( PIX pixs, BOX box, @Cast("l_int32") int lowthresh, @Cast("l_int32") int highthresh, @Cast("l_int32") int maxwidth, @Cast("l_int32") int factor, @Cast("l_int32") int scanflag, @Cast("l_int32*") int[] ploc );
public static native NUMA pixExtractOnLine( PIX pixs, @Cast("l_int32") int x1, @Cast("l_int32") int y1, @Cast("l_int32") int x2, @Cast("l_int32") int y2, @Cast("l_int32") int factor );
public static native @Cast("l_float32") float pixAverageOnLine( PIX pixs, @Cast("l_int32") int x1, @Cast("l_int32") int y1, @Cast("l_int32") int x2, @Cast("l_int32") int y2, @Cast("l_int32") int factor );
public static native NUMA pixAverageIntensityProfile( PIX pixs, @Cast("l_float32") float fract, @Cast("l_int32") int dir, @Cast("l_int32") int first, @Cast("l_int32") int last, @Cast("l_int32") int factor1, @Cast("l_int32") int factor2 );
public static native NUMA pixReversalProfile( PIX pixs, @Cast("l_float32") float fract, @Cast("l_int32") int dir, @Cast("l_int32") int first, @Cast("l_int32") int last, @Cast("l_int32") int minreversal, @Cast("l_int32") int factor1, @Cast("l_int32") int factor2 );
public static native @Cast("l_int32") int pixWindowedVarianceOnLine( PIX pixs, @Cast("l_int32") int dir, @Cast("l_int32") int loc, @Cast("l_int32") int c1, @Cast("l_int32") int c2, @Cast("l_int32") int size, @Cast("NUMA**") PointerPointer pnad );
public static native @Cast("l_int32") int pixWindowedVarianceOnLine( PIX pixs, @Cast("l_int32") int dir, @Cast("l_int32") int loc, @Cast("l_int32") int c1, @Cast("l_int32") int c2, @Cast("l_int32") int size, @ByPtrPtr NUMA pnad );
public static native @Cast("l_int32") int pixMinMaxNearLine( PIX pixs, @Cast("l_int32") int x1, @Cast("l_int32") int y1, @Cast("l_int32") int x2, @Cast("l_int32") int y2, @Cast("l_int32") int dist, @Cast("l_int32") int direction, @Cast("NUMA**") PointerPointer pnamin, @Cast("NUMA**") PointerPointer pnamax, @Cast("l_float32*") FloatPointer pminave, @Cast("l_float32*") FloatPointer pmaxave );
public static native @Cast("l_int32") int pixMinMaxNearLine( PIX pixs, @Cast("l_int32") int x1, @Cast("l_int32") int y1, @Cast("l_int32") int x2, @Cast("l_int32") int y2, @Cast("l_int32") int dist, @Cast("l_int32") int direction, @ByPtrPtr NUMA pnamin, @ByPtrPtr NUMA pnamax, @Cast("l_float32*") FloatPointer pminave, @Cast("l_float32*") FloatPointer pmaxave );
public static native @Cast("l_int32") int pixMinMaxNearLine( PIX pixs, @Cast("l_int32") int x1, @Cast("l_int32") int y1, @Cast("l_int32") int x2, @Cast("l_int32") int y2, @Cast("l_int32") int dist, @Cast("l_int32") int direction, @ByPtrPtr NUMA pnamin, @ByPtrPtr NUMA pnamax, @Cast("l_float32*") FloatBuffer pminave, @Cast("l_float32*") FloatBuffer pmaxave );
public static native @Cast("l_int32") int pixMinMaxNearLine( PIX pixs, @Cast("l_int32") int x1, @Cast("l_int32") int y1, @Cast("l_int32") int x2, @Cast("l_int32") int y2, @Cast("l_int32") int dist, @Cast("l_int32") int direction, @ByPtrPtr NUMA pnamin, @ByPtrPtr NUMA pnamax, @Cast("l_float32*") float[] pminave, @Cast("l_float32*") float[] pmaxave );
public static native PIX pixRankRowTransform( PIX pixs );
public static native PIX pixRankColumnTransform( PIX pixs );
public static native PIXA pixaCreate( @Cast("l_int32") int n );
public static native PIXA pixaCreateFromPix( PIX pixs, @Cast("l_int32") int n, @Cast("l_int32") int cellw, @Cast("l_int32") int cellh );
public static native PIXA pixaCreateFromBoxa( PIX pixs, BOXA boxa, @Cast("l_int32*") IntPointer pcropwarn );
public static native PIXA pixaCreateFromBoxa( PIX pixs, BOXA boxa, @Cast("l_int32*") IntBuffer pcropwarn );
public static native PIXA pixaCreateFromBoxa( PIX pixs, BOXA boxa, @Cast("l_int32*") int[] pcropwarn );
public static native PIXA pixaSplitPix( PIX pixs, @Cast("l_int32") int nx, @Cast("l_int32") int ny, @Cast("l_int32") int borderwidth, @Cast("l_uint32") int bordercolor );
public static native void pixaDestroy( @Cast("PIXA**") PointerPointer ppixa );
public static native void pixaDestroy( @ByPtrPtr PIXA ppixa );
public static native PIXA pixaCopy( PIXA pixa, @Cast("l_int32") int copyflag );
public static native @Cast("l_int32") int pixaAddPix( PIXA pixa, PIX pix, @Cast("l_int32") int copyflag );
public static native @Cast("l_int32") int pixaAddBox( PIXA pixa, BOX box, @Cast("l_int32") int copyflag );
public static native @Cast("l_int32") int pixaExtendArrayToSize( PIXA pixa, @Cast("l_int32") int size );
public static native @Cast("l_int32") int pixaGetCount( PIXA pixa );
public static native @Cast("l_int32") int pixaChangeRefcount( PIXA pixa, @Cast("l_int32") int delta );
public static native PIX pixaGetPix( PIXA pixa, @Cast("l_int32") int index, @Cast("l_int32") int accesstype );
public static native @Cast("l_int32") int pixaGetPixDimensions( PIXA pixa, @Cast("l_int32") int index, @Cast("l_int32*") IntPointer pw, @Cast("l_int32*") IntPointer ph, @Cast("l_int32*") IntPointer pd );
public static native @Cast("l_int32") int pixaGetPixDimensions( PIXA pixa, @Cast("l_int32") int index, @Cast("l_int32*") IntBuffer pw, @Cast("l_int32*") IntBuffer ph, @Cast("l_int32*") IntBuffer pd );
public static native @Cast("l_int32") int pixaGetPixDimensions( PIXA pixa, @Cast("l_int32") int index, @Cast("l_int32*") int[] pw, @Cast("l_int32*") int[] ph, @Cast("l_int32*") int[] pd );
public static native BOXA pixaGetBoxa( PIXA pixa, @Cast("l_int32") int accesstype );
public static native @Cast("l_int32") int pixaGetBoxaCount( PIXA pixa );
public static native BOX pixaGetBox( PIXA pixa, @Cast("l_int32") int index, @Cast("l_int32") int accesstype );
public static native @Cast("l_int32") int pixaGetBoxGeometry( PIXA pixa, @Cast("l_int32") int index, @Cast("l_int32*") IntPointer px, @Cast("l_int32*") IntPointer py, @Cast("l_int32*") IntPointer pw, @Cast("l_int32*") IntPointer ph );
public static native @Cast("l_int32") int pixaGetBoxGeometry( PIXA pixa, @Cast("l_int32") int index, @Cast("l_int32*") IntBuffer px, @Cast("l_int32*") IntBuffer py, @Cast("l_int32*") IntBuffer pw, @Cast("l_int32*") IntBuffer ph );
public static native @Cast("l_int32") int pixaGetBoxGeometry( PIXA pixa, @Cast("l_int32") int index, @Cast("l_int32*") int[] px, @Cast("l_int32*") int[] py, @Cast("l_int32*") int[] pw, @Cast("l_int32*") int[] ph );
public static native @Cast("l_int32") int pixaSetBoxa( PIXA pixa, BOXA boxa, @Cast("l_int32") int accesstype );
public static native @Cast("PIX**") PointerPointer pixaGetPixArray( PIXA pixa );
public static native @Cast("l_int32") int pixaVerifyDepth( PIXA pixa, @Cast("l_int32*") IntPointer pmaxdepth );
public static native @Cast("l_int32") int pixaVerifyDepth( PIXA pixa, @Cast("l_int32*") IntBuffer pmaxdepth );
public static native @Cast("l_int32") int pixaVerifyDepth( PIXA pixa, @Cast("l_int32*") int[] pmaxdepth );
public static native @Cast("l_int32") int pixaIsFull( PIXA pixa, @Cast("l_int32*") IntPointer pfullpa, @Cast("l_int32*") IntPointer pfullba );
public static native @Cast("l_int32") int pixaIsFull( PIXA pixa, @Cast("l_int32*") IntBuffer pfullpa, @Cast("l_int32*") IntBuffer pfullba );
public static native @Cast("l_int32") int pixaIsFull( PIXA pixa, @Cast("l_int32*") int[] pfullpa, @Cast("l_int32*") int[] pfullba );
public static native @Cast("l_int32") int pixaCountText( PIXA pixa, @Cast("l_int32*") IntPointer pntext );
public static native @Cast("l_int32") int pixaCountText( PIXA pixa, @Cast("l_int32*") IntBuffer pntext );
public static native @Cast("l_int32") int pixaCountText( PIXA pixa, @Cast("l_int32*") int[] pntext );
public static native @Cast("void***") PointerPointer pixaGetLinePtrs( PIXA pixa, @Cast("l_int32*") IntPointer psize );
public static native @Cast("void***") PointerPointer pixaGetLinePtrs( PIXA pixa, @Cast("l_int32*") IntBuffer psize );
public static native @Cast("void***") PointerPointer pixaGetLinePtrs( PIXA pixa, @Cast("l_int32*") int[] psize );
public static native @Cast("l_int32") int pixaReplacePix( PIXA pixa, @Cast("l_int32") int index, PIX pix, BOX box );
public static native @Cast("l_int32") int pixaInsertPix( PIXA pixa, @Cast("l_int32") int index, PIX pixs, BOX box );
public static native @Cast("l_int32") int pixaRemovePix( PIXA pixa, @Cast("l_int32") int index );
public static native @Cast("l_int32") int pixaRemovePixAndSave( PIXA pixa, @Cast("l_int32") int index, @Cast("PIX**") PointerPointer ppix, @Cast("BOX**") PointerPointer pbox );
public static native @Cast("l_int32") int pixaRemovePixAndSave( PIXA pixa, @Cast("l_int32") int index, @ByPtrPtr PIX ppix, @ByPtrPtr BOX pbox );
public static native @Cast("l_int32") int pixaInitFull( PIXA pixa, PIX pix, BOX box );
public static native @Cast("l_int32") int pixaClear( PIXA pixa );
public static native @Cast("l_int32") int pixaJoin( PIXA pixad, PIXA pixas, @Cast("l_int32") int istart, @Cast("l_int32") int iend );
public static native @Cast("l_int32") int pixaaJoin( PIXAA paad, PIXAA paas, @Cast("l_int32") int istart, @Cast("l_int32") int iend );
public static native PIXAA pixaaCreate( @Cast("l_int32") int n );
public static native PIXAA pixaaCreateFromPixa( PIXA pixa, @Cast("l_int32") int n, @Cast("l_int32") int type, @Cast("l_int32") int copyflag );
public static native void pixaaDestroy( @Cast("PIXAA**") PointerPointer ppaa );
public static native void pixaaDestroy( @ByPtrPtr PIXAA ppaa );
public static native @Cast("l_int32") int pixaaAddPixa( PIXAA paa, PIXA pixa, @Cast("l_int32") int copyflag );
public static native @Cast("l_int32") int pixaaExtendArray( PIXAA paa );
public static native @Cast("l_int32") int pixaaAddPix( PIXAA paa, @Cast("l_int32") int index, PIX pix, BOX box, @Cast("l_int32") int copyflag );
public static native @Cast("l_int32") int pixaaAddBox( PIXAA paa, BOX box, @Cast("l_int32") int copyflag );
public static native @Cast("l_int32") int pixaaGetCount( PIXAA paa, @Cast("NUMA**") PointerPointer pna );
public static native @Cast("l_int32") int pixaaGetCount( PIXAA paa, @ByPtrPtr NUMA pna );
public static native PIXA pixaaGetPixa( PIXAA paa, @Cast("l_int32") int index, @Cast("l_int32") int accesstype );
public static native BOXA pixaaGetBoxa( PIXAA paa, @Cast("l_int32") int accesstype );
public static native PIX pixaaGetPix( PIXAA paa, @Cast("l_int32") int index, @Cast("l_int32") int ipix, @Cast("l_int32") int accessflag );
public static native @Cast("l_int32") int pixaaVerifyDepth( PIXAA paa, @Cast("l_int32*") IntPointer pmaxdepth );
public static native @Cast("l_int32") int pixaaVerifyDepth( PIXAA paa, @Cast("l_int32*") IntBuffer pmaxdepth );
public static native @Cast("l_int32") int pixaaVerifyDepth( PIXAA paa, @Cast("l_int32*") int[] pmaxdepth );
public static native @Cast("l_int32") int pixaaIsFull( PIXAA paa, @Cast("l_int32*") IntPointer pfull );
public static native @Cast("l_int32") int pixaaIsFull( PIXAA paa, @Cast("l_int32*") IntBuffer pfull );
public static native @Cast("l_int32") int pixaaIsFull( PIXAA paa, @Cast("l_int32*") int[] pfull );
public static native @Cast("l_int32") int pixaaInitFull( PIXAA paa, PIXA pixa );
public static native @Cast("l_int32") int pixaaReplacePixa( PIXAA paa, @Cast("l_int32") int index, PIXA pixa );
public static native @Cast("l_int32") int pixaaClear( PIXAA paa );
public static native @Cast("l_int32") int pixaaTruncate( PIXAA paa );
public static native PIXA pixaRead( @Cast("const char*") BytePointer filename );
public static native PIXA pixaRead( String filename );
public static native PIXA pixaReadStream( @Cast("FILE*") Pointer fp );
public static native @Cast("l_int32") int pixaWrite( @Cast("const char*") BytePointer filename, PIXA pixa );
public static native @Cast("l_int32") int pixaWrite( String filename, PIXA pixa );
public static native @Cast("l_int32") int pixaWriteStream( @Cast("FILE*") Pointer fp, PIXA pixa );
public static native PIXAA pixaaReadFromFiles( @Cast("const char*") BytePointer dirname, @Cast("const char*") BytePointer substr, @Cast("l_int32") int first, @Cast("l_int32") int nfiles );
public static native PIXAA pixaaReadFromFiles( String dirname, String substr, @Cast("l_int32") int first, @Cast("l_int32") int nfiles );
public static native PIXAA pixaaRead( @Cast("const char*") BytePointer filename );
public static native PIXAA pixaaRead( String filename );
public static native PIXAA pixaaReadStream( @Cast("FILE*") Pointer fp );
public static native @Cast("l_int32") int pixaaWrite( @Cast("const char*") BytePointer filename, PIXAA paa );
public static native @Cast("l_int32") int pixaaWrite( String filename, PIXAA paa );
public static native @Cast("l_int32") int pixaaWriteStream( @Cast("FILE*") Pointer fp, PIXAA paa );
public static native PIXACC pixaccCreate( @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int negflag );
public static native PIXACC pixaccCreateFromPix( PIX pix, @Cast("l_int32") int negflag );
public static native void pixaccDestroy( @Cast("PIXACC**") PointerPointer ppixacc );
public static native void pixaccDestroy( @ByPtrPtr PIXACC ppixacc );
public static native PIX pixaccFinal( PIXACC pixacc, @Cast("l_int32") int outdepth );
public static native PIX pixaccGetPix( PIXACC pixacc );
public static native @Cast("l_int32") int pixaccGetOffset( PIXACC pixacc );
public static native @Cast("l_int32") int pixaccAdd( PIXACC pixacc, PIX pix );
public static native @Cast("l_int32") int pixaccSubtract( PIXACC pixacc, PIX pix );
public static native @Cast("l_int32") int pixaccMultConst( PIXACC pixacc, @Cast("l_float32") float factor );
public static native @Cast("l_int32") int pixaccMultConstAccumulate( PIXACC pixacc, PIX pix, @Cast("l_float32") float factor );
public static native PIX pixSelectBySize( PIX pixs, @Cast("l_int32") int width, @Cast("l_int32") int height, @Cast("l_int32") int connectivity, @Cast("l_int32") int type, @Cast("l_int32") int relation, @Cast("l_int32*") IntPointer pchanged );
public static native PIX pixSelectBySize( PIX pixs, @Cast("l_int32") int width, @Cast("l_int32") int height, @Cast("l_int32") int connectivity, @Cast("l_int32") int type, @Cast("l_int32") int relation, @Cast("l_int32*") IntBuffer pchanged );
public static native PIX pixSelectBySize( PIX pixs, @Cast("l_int32") int width, @Cast("l_int32") int height, @Cast("l_int32") int connectivity, @Cast("l_int32") int type, @Cast("l_int32") int relation, @Cast("l_int32*") int[] pchanged );
public static native PIXA pixaSelectBySize( PIXA pixas, @Cast("l_int32") int width, @Cast("l_int32") int height, @Cast("l_int32") int type, @Cast("l_int32") int relation, @Cast("l_int32*") IntPointer pchanged );
public static native PIXA pixaSelectBySize( PIXA pixas, @Cast("l_int32") int width, @Cast("l_int32") int height, @Cast("l_int32") int type, @Cast("l_int32") int relation, @Cast("l_int32*") IntBuffer pchanged );
public static native PIXA pixaSelectBySize( PIXA pixas, @Cast("l_int32") int width, @Cast("l_int32") int height, @Cast("l_int32") int type, @Cast("l_int32") int relation, @Cast("l_int32*") int[] pchanged );
public static native NUMA pixaMakeSizeIndicator( PIXA pixa, @Cast("l_int32") int width, @Cast("l_int32") int height, @Cast("l_int32") int type, @Cast("l_int32") int relation );
public static native PIX pixSelectByPerimToAreaRatio( PIX pixs, @Cast("l_float32") float thresh, @Cast("l_int32") int connectivity, @Cast("l_int32") int type, @Cast("l_int32*") IntPointer pchanged );
public static native PIX pixSelectByPerimToAreaRatio( PIX pixs, @Cast("l_float32") float thresh, @Cast("l_int32") int connectivity, @Cast("l_int32") int type, @Cast("l_int32*") IntBuffer pchanged );
public static native PIX pixSelectByPerimToAreaRatio( PIX pixs, @Cast("l_float32") float thresh, @Cast("l_int32") int connectivity, @Cast("l_int32") int type, @Cast("l_int32*") int[] pchanged );
public static native PIXA pixaSelectByPerimToAreaRatio( PIXA pixas, @Cast("l_float32") float thresh, @Cast("l_int32") int type, @Cast("l_int32*") IntPointer pchanged );
public static native PIXA pixaSelectByPerimToAreaRatio( PIXA pixas, @Cast("l_float32") float thresh, @Cast("l_int32") int type, @Cast("l_int32*") IntBuffer pchanged );
public static native PIXA pixaSelectByPerimToAreaRatio( PIXA pixas, @Cast("l_float32") float thresh, @Cast("l_int32") int type, @Cast("l_int32*") int[] pchanged );
public static native PIX pixSelectByPerimSizeRatio( PIX pixs, @Cast("l_float32") float thresh, @Cast("l_int32") int connectivity, @Cast("l_int32") int type, @Cast("l_int32*") IntPointer pchanged );
public static native PIX pixSelectByPerimSizeRatio( PIX pixs, @Cast("l_float32") float thresh, @Cast("l_int32") int connectivity, @Cast("l_int32") int type, @Cast("l_int32*") IntBuffer pchanged );
public static native PIX pixSelectByPerimSizeRatio( PIX pixs, @Cast("l_float32") float thresh, @Cast("l_int32") int connectivity, @Cast("l_int32") int type, @Cast("l_int32*") int[] pchanged );
public static native PIXA pixaSelectByPerimSizeRatio( PIXA pixas, @Cast("l_float32") float thresh, @Cast("l_int32") int type, @Cast("l_int32*") IntPointer pchanged );
public static native PIXA pixaSelectByPerimSizeRatio( PIXA pixas, @Cast("l_float32") float thresh, @Cast("l_int32") int type, @Cast("l_int32*") IntBuffer pchanged );
public static native PIXA pixaSelectByPerimSizeRatio( PIXA pixas, @Cast("l_float32") float thresh, @Cast("l_int32") int type, @Cast("l_int32*") int[] pchanged );
public static native PIX pixSelectByAreaFraction( PIX pixs, @Cast("l_float32") float thresh, @Cast("l_int32") int connectivity, @Cast("l_int32") int type, @Cast("l_int32*") IntPointer pchanged );
public static native PIX pixSelectByAreaFraction( PIX pixs, @Cast("l_float32") float thresh, @Cast("l_int32") int connectivity, @Cast("l_int32") int type, @Cast("l_int32*") IntBuffer pchanged );
public static native PIX pixSelectByAreaFraction( PIX pixs, @Cast("l_float32") float thresh, @Cast("l_int32") int connectivity, @Cast("l_int32") int type, @Cast("l_int32*") int[] pchanged );
public static native PIXA pixaSelectByAreaFraction( PIXA pixas, @Cast("l_float32") float thresh, @Cast("l_int32") int type, @Cast("l_int32*") IntPointer pchanged );
public static native PIXA pixaSelectByAreaFraction( PIXA pixas, @Cast("l_float32") float thresh, @Cast("l_int32") int type, @Cast("l_int32*") IntBuffer pchanged );
public static native PIXA pixaSelectByAreaFraction( PIXA pixas, @Cast("l_float32") float thresh, @Cast("l_int32") int type, @Cast("l_int32*") int[] pchanged );
public static native PIX pixSelectByWidthHeightRatio( PIX pixs, @Cast("l_float32") float thresh, @Cast("l_int32") int connectivity, @Cast("l_int32") int type, @Cast("l_int32*") IntPointer pchanged );
public static native PIX pixSelectByWidthHeightRatio( PIX pixs, @Cast("l_float32") float thresh, @Cast("l_int32") int connectivity, @Cast("l_int32") int type, @Cast("l_int32*") IntBuffer pchanged );
public static native PIX pixSelectByWidthHeightRatio( PIX pixs, @Cast("l_float32") float thresh, @Cast("l_int32") int connectivity, @Cast("l_int32") int type, @Cast("l_int32*") int[] pchanged );
public static native PIXA pixaSelectByWidthHeightRatio( PIXA pixas, @Cast("l_float32") float thresh, @Cast("l_int32") int type, @Cast("l_int32*") IntPointer pchanged );
public static native PIXA pixaSelectByWidthHeightRatio( PIXA pixas, @Cast("l_float32") float thresh, @Cast("l_int32") int type, @Cast("l_int32*") IntBuffer pchanged );
public static native PIXA pixaSelectByWidthHeightRatio( PIXA pixas, @Cast("l_float32") float thresh, @Cast("l_int32") int type, @Cast("l_int32*") int[] pchanged );
public static native PIXA pixaSelectWithIndicator( PIXA pixas, NUMA na, @Cast("l_int32*") IntPointer pchanged );
public static native PIXA pixaSelectWithIndicator( PIXA pixas, NUMA na, @Cast("l_int32*") IntBuffer pchanged );
public static native PIXA pixaSelectWithIndicator( PIXA pixas, NUMA na, @Cast("l_int32*") int[] pchanged );
public static native @Cast("l_int32") int pixRemoveWithIndicator( PIX pixs, PIXA pixa, NUMA na );
public static native @Cast("l_int32") int pixAddWithIndicator( PIX pixs, PIXA pixa, NUMA na );
public static native PIX pixaRenderComponent( PIX pixs, PIXA pixa, @Cast("l_int32") int index );
public static native PIXA pixaSort( PIXA pixas, @Cast("l_int32") int sorttype, @Cast("l_int32") int sortorder, @Cast("NUMA**") PointerPointer pnaindex, @Cast("l_int32") int copyflag );
public static native PIXA pixaSort( PIXA pixas, @Cast("l_int32") int sorttype, @Cast("l_int32") int sortorder, @ByPtrPtr NUMA pnaindex, @Cast("l_int32") int copyflag );
public static native PIXA pixaBinSort( PIXA pixas, @Cast("l_int32") int sorttype, @Cast("l_int32") int sortorder, @Cast("NUMA**") PointerPointer pnaindex, @Cast("l_int32") int copyflag );
public static native PIXA pixaBinSort( PIXA pixas, @Cast("l_int32") int sorttype, @Cast("l_int32") int sortorder, @ByPtrPtr NUMA pnaindex, @Cast("l_int32") int copyflag );
public static native PIXA pixaSortByIndex( PIXA pixas, NUMA naindex, @Cast("l_int32") int copyflag );
public static native PIXAA pixaSort2dByIndex( PIXA pixas, NUMAA naa, @Cast("l_int32") int copyflag );
public static native PIXA pixaSelectRange( PIXA pixas, @Cast("l_int32") int first, @Cast("l_int32") int last, @Cast("l_int32") int copyflag );
public static native PIXAA pixaaSelectRange( PIXAA paas, @Cast("l_int32") int first, @Cast("l_int32") int last, @Cast("l_int32") int copyflag );
public static native PIXAA pixaaScaleToSize( PIXAA paas, @Cast("l_int32") int wd, @Cast("l_int32") int hd );
public static native PIXAA pixaaScaleToSizeVar( PIXAA paas, NUMA nawd, NUMA nahd );
public static native PIXA pixaScaleToSize( PIXA pixas, @Cast("l_int32") int wd, @Cast("l_int32") int hd );
public static native PIXA pixaAddBorderGeneral( PIXA pixad, PIXA pixas, @Cast("l_int32") int left, @Cast("l_int32") int right, @Cast("l_int32") int top, @Cast("l_int32") int bot, @Cast("l_uint32") int val );
public static native PIXA pixaaFlattenToPixa( PIXAA paa, @Cast("NUMA**") PointerPointer pnaindex, @Cast("l_int32") int copyflag );
public static native PIXA pixaaFlattenToPixa( PIXAA paa, @ByPtrPtr NUMA pnaindex, @Cast("l_int32") int copyflag );
public static native @Cast("l_int32") int pixaaSizeRange( PIXAA paa, @Cast("l_int32*") IntPointer pminw, @Cast("l_int32*") IntPointer pminh, @Cast("l_int32*") IntPointer pmaxw, @Cast("l_int32*") IntPointer pmaxh );
public static native @Cast("l_int32") int pixaaSizeRange( PIXAA paa, @Cast("l_int32*") IntBuffer pminw, @Cast("l_int32*") IntBuffer pminh, @Cast("l_int32*") IntBuffer pmaxw, @Cast("l_int32*") IntBuffer pmaxh );
public static native @Cast("l_int32") int pixaaSizeRange( PIXAA paa, @Cast("l_int32*") int[] pminw, @Cast("l_int32*") int[] pminh, @Cast("l_int32*") int[] pmaxw, @Cast("l_int32*") int[] pmaxh );
public static native @Cast("l_int32") int pixaSizeRange( PIXA pixa, @Cast("l_int32*") IntPointer pminw, @Cast("l_int32*") IntPointer pminh, @Cast("l_int32*") IntPointer pmaxw, @Cast("l_int32*") IntPointer pmaxh );
public static native @Cast("l_int32") int pixaSizeRange( PIXA pixa, @Cast("l_int32*") IntBuffer pminw, @Cast("l_int32*") IntBuffer pminh, @Cast("l_int32*") IntBuffer pmaxw, @Cast("l_int32*") IntBuffer pmaxh );
public static native @Cast("l_int32") int pixaSizeRange( PIXA pixa, @Cast("l_int32*") int[] pminw, @Cast("l_int32*") int[] pminh, @Cast("l_int32*") int[] pmaxw, @Cast("l_int32*") int[] pmaxh );
public static native PIXA pixaClipToPix( PIXA pixas, PIX pixs );
public static native @Cast("l_int32") int pixaGetRenderingDepth( PIXA pixa, @Cast("l_int32*") IntPointer pdepth );
public static native @Cast("l_int32") int pixaGetRenderingDepth( PIXA pixa, @Cast("l_int32*") IntBuffer pdepth );
public static native @Cast("l_int32") int pixaGetRenderingDepth( PIXA pixa, @Cast("l_int32*") int[] pdepth );
public static native @Cast("l_int32") int pixaHasColor( PIXA pixa, @Cast("l_int32*") IntPointer phascolor );
public static native @Cast("l_int32") int pixaHasColor( PIXA pixa, @Cast("l_int32*") IntBuffer phascolor );
public static native @Cast("l_int32") int pixaHasColor( PIXA pixa, @Cast("l_int32*") int[] phascolor );
public static native @Cast("l_int32") int pixaAnyColormaps( PIXA pixa, @Cast("l_int32*") IntPointer phascmap );
public static native @Cast("l_int32") int pixaAnyColormaps( PIXA pixa, @Cast("l_int32*") IntBuffer phascmap );
public static native @Cast("l_int32") int pixaAnyColormaps( PIXA pixa, @Cast("l_int32*") int[] phascmap );
public static native @Cast("l_int32") int pixaGetDepthInfo( PIXA pixa, @Cast("l_int32*") IntPointer pmaxdepth, @Cast("l_int32*") IntPointer psame );
public static native @Cast("l_int32") int pixaGetDepthInfo( PIXA pixa, @Cast("l_int32*") IntBuffer pmaxdepth, @Cast("l_int32*") IntBuffer psame );
public static native @Cast("l_int32") int pixaGetDepthInfo( PIXA pixa, @Cast("l_int32*") int[] pmaxdepth, @Cast("l_int32*") int[] psame );
public static native PIXA pixaConvertToSameDepth( PIXA pixas );
public static native @Cast("l_int32") int pixaEqual( PIXA pixa1, PIXA pixa2, @Cast("l_int32") int maxdist, @Cast("NUMA**") PointerPointer pnaindex, @Cast("l_int32*") IntPointer psame );
public static native @Cast("l_int32") int pixaEqual( PIXA pixa1, PIXA pixa2, @Cast("l_int32") int maxdist, @ByPtrPtr NUMA pnaindex, @Cast("l_int32*") IntPointer psame );
public static native @Cast("l_int32") int pixaEqual( PIXA pixa1, PIXA pixa2, @Cast("l_int32") int maxdist, @ByPtrPtr NUMA pnaindex, @Cast("l_int32*") IntBuffer psame );
public static native @Cast("l_int32") int pixaEqual( PIXA pixa1, PIXA pixa2, @Cast("l_int32") int maxdist, @ByPtrPtr NUMA pnaindex, @Cast("l_int32*") int[] psame );
public static native PIX pixaDisplay( PIXA pixa, @Cast("l_int32") int w, @Cast("l_int32") int h );
public static native PIX pixaDisplayOnColor( PIXA pixa, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_uint32") int bgcolor );
public static native PIX pixaDisplayRandomCmap( PIXA pixa, @Cast("l_int32") int w, @Cast("l_int32") int h );
public static native PIX pixaDisplayLinearly( PIXA pixas, @Cast("l_int32") int direction, @Cast("l_float32") float scalefactor, @Cast("l_int32") int background, @Cast("l_int32") int spacing, @Cast("l_int32") int border, @Cast("BOXA**") PointerPointer pboxa );
public static native PIX pixaDisplayLinearly( PIXA pixas, @Cast("l_int32") int direction, @Cast("l_float32") float scalefactor, @Cast("l_int32") int background, @Cast("l_int32") int spacing, @Cast("l_int32") int border, @ByPtrPtr BOXA pboxa );
public static native PIX pixaDisplayOnLattice( PIXA pixa, @Cast("l_int32") int cellw, @Cast("l_int32") int cellh, @Cast("l_int32*") IntPointer pncols, @Cast("BOXA**") PointerPointer pboxa );
public static native PIX pixaDisplayOnLattice( PIXA pixa, @Cast("l_int32") int cellw, @Cast("l_int32") int cellh, @Cast("l_int32*") IntPointer pncols, @ByPtrPtr BOXA pboxa );
public static native PIX pixaDisplayOnLattice( PIXA pixa, @Cast("l_int32") int cellw, @Cast("l_int32") int cellh, @Cast("l_int32*") IntBuffer pncols, @ByPtrPtr BOXA pboxa );
public static native PIX pixaDisplayOnLattice( PIXA pixa, @Cast("l_int32") int cellw, @Cast("l_int32") int cellh, @Cast("l_int32*") int[] pncols, @ByPtrPtr BOXA pboxa );
public static native PIX pixaDisplayUnsplit( PIXA pixa, @Cast("l_int32") int nx, @Cast("l_int32") int ny, @Cast("l_int32") int borderwidth, @Cast("l_uint32") int bordercolor );
public static native PIX pixaDisplayTiled( PIXA pixa, @Cast("l_int32") int maxwidth, @Cast("l_int32") int background, @Cast("l_int32") int spacing );
public static native PIX pixaDisplayTiledInRows( PIXA pixa, @Cast("l_int32") int outdepth, @Cast("l_int32") int maxwidth, @Cast("l_float32") float scalefactor, @Cast("l_int32") int background, @Cast("l_int32") int spacing, @Cast("l_int32") int border );
public static native PIX pixaDisplayTiledAndScaled( PIXA pixa, @Cast("l_int32") int outdepth, @Cast("l_int32") int tilewidth, @Cast("l_int32") int ncols, @Cast("l_int32") int background, @Cast("l_int32") int spacing, @Cast("l_int32") int border );
public static native PIX pixaaDisplay( PIXAA paa, @Cast("l_int32") int w, @Cast("l_int32") int h );
public static native PIX pixaaDisplayByPixa( PIXAA paa, @Cast("l_int32") int xspace, @Cast("l_int32") int yspace, @Cast("l_int32") int maxw );
public static native PIXA pixaaDisplayTiledAndScaled( PIXAA paa, @Cast("l_int32") int outdepth, @Cast("l_int32") int tilewidth, @Cast("l_int32") int ncols, @Cast("l_int32") int background, @Cast("l_int32") int spacing, @Cast("l_int32") int border );
public static native PIXA pixaConvertTo1( PIXA pixas, @Cast("l_int32") int thresh );
public static native PIXA pixaConvertTo8( PIXA pixas, @Cast("l_int32") int cmapflag );
public static native PIXA pixaConvertTo8Color( PIXA pixas, @Cast("l_int32") int dither );
public static native PIXA pixaConvertTo32( PIXA pixas );
public static native @Cast("l_int32") int convertToNUpFiles( @Cast("const char*") BytePointer dir, @Cast("const char*") BytePointer substr, @Cast("l_int32") int nx, @Cast("l_int32") int ny, @Cast("l_int32") int tw, @Cast("l_int32") int spacing, @Cast("l_int32") int border, @Cast("l_int32") int fontsize, @Cast("const char*") BytePointer outdir );
public static native @Cast("l_int32") int convertToNUpFiles( String dir, String substr, @Cast("l_int32") int nx, @Cast("l_int32") int ny, @Cast("l_int32") int tw, @Cast("l_int32") int spacing, @Cast("l_int32") int border, @Cast("l_int32") int fontsize, String outdir );
public static native PIXA convertToNUpPixa( @Cast("const char*") BytePointer dir, @Cast("const char*") BytePointer substr, @Cast("l_int32") int nx, @Cast("l_int32") int ny, @Cast("l_int32") int tw, @Cast("l_int32") int spacing, @Cast("l_int32") int border, @Cast("l_int32") int fontsize );
public static native PIXA convertToNUpPixa( String dir, String substr, @Cast("l_int32") int nx, @Cast("l_int32") int ny, @Cast("l_int32") int tw, @Cast("l_int32") int spacing, @Cast("l_int32") int border, @Cast("l_int32") int fontsize );
public static native @Cast("l_int32") int pmsCreate( @Cast("size_t") long minsize, @Cast("size_t") long smallest, NUMA numalloc, @Cast("const char*") BytePointer logfile );
public static native @Cast("l_int32") int pmsCreate( @Cast("size_t") long minsize, @Cast("size_t") long smallest, NUMA numalloc, String logfile );
public static native void pmsDestroy(  );
public static native Pointer pmsCustomAlloc( @Cast("size_t") long nbytes );
public static native void pmsCustomDealloc( Pointer data );
public static native Pointer pmsGetAlloc( @Cast("size_t") long nbytes );
public static native @Cast("l_int32") int pmsGetLevelForAlloc( @Cast("size_t") long nbytes, @Cast("l_int32*") IntPointer plevel );
public static native @Cast("l_int32") int pmsGetLevelForAlloc( @Cast("size_t") long nbytes, @Cast("l_int32*") IntBuffer plevel );
public static native @Cast("l_int32") int pmsGetLevelForAlloc( @Cast("size_t") long nbytes, @Cast("l_int32*") int[] plevel );
public static native @Cast("l_int32") int pmsGetLevelForDealloc( Pointer data, @Cast("l_int32*") IntPointer plevel );
public static native @Cast("l_int32") int pmsGetLevelForDealloc( Pointer data, @Cast("l_int32*") IntBuffer plevel );
public static native @Cast("l_int32") int pmsGetLevelForDealloc( Pointer data, @Cast("l_int32*") int[] plevel );
public static native void pmsLogInfo(  );
public static native @Cast("l_int32") int pixAddConstantGray( PIX pixs, @Cast("l_int32") int val );
public static native @Cast("l_int32") int pixMultConstantGray( PIX pixs, @Cast("l_float32") float val );
public static native PIX pixAddGray( PIX pixd, PIX pixs1, PIX pixs2 );
public static native PIX pixSubtractGray( PIX pixd, PIX pixs1, PIX pixs2 );
public static native PIX pixThresholdToValue( PIX pixd, PIX pixs, @Cast("l_int32") int threshval, @Cast("l_int32") int setval );
public static native PIX pixInitAccumulate( @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_uint32") int offset );
public static native PIX pixFinalAccumulate( PIX pixs, @Cast("l_uint32") int offset, @Cast("l_int32") int depth );
public static native PIX pixFinalAccumulateThreshold( PIX pixs, @Cast("l_uint32") int offset, @Cast("l_uint32") int threshold );
public static native @Cast("l_int32") int pixAccumulate( PIX pixd, PIX pixs, @Cast("l_int32") int op );
public static native @Cast("l_int32") int pixMultConstAccumulate( PIX pixs, @Cast("l_float32") float factor, @Cast("l_uint32") int offset );
public static native PIX pixAbsDifference( PIX pixs1, PIX pixs2 );
public static native PIX pixAddRGB( PIX pixs1, PIX pixs2 );
public static native PIX pixMinOrMax( PIX pixd, PIX pixs1, PIX pixs2, @Cast("l_int32") int type );
public static native PIX pixMaxDynamicRange( PIX pixs, @Cast("l_int32") int type );
public static native @Cast("l_float32*") FloatPointer makeLogBase2Tab( );
public static native @Cast("l_float32") float getLogBase2( @Cast("l_int32") int val, @Cast("l_float32*") FloatPointer logtab );
public static native @Cast("l_float32") float getLogBase2( @Cast("l_int32") int val, @Cast("l_float32*") FloatBuffer logtab );
public static native @Cast("l_float32") float getLogBase2( @Cast("l_int32") int val, @Cast("l_float32*") float[] logtab );
public static native PIXC pixcompCreateFromPix( PIX pix, @Cast("l_int32") int comptype );
public static native PIXC pixcompCreateFromString( @Cast("l_uint8*") BytePointer data, @Cast("size_t") long size, @Cast("l_int32") int copyflag );
public static native PIXC pixcompCreateFromString( @Cast("l_uint8*") ByteBuffer data, @Cast("size_t") long size, @Cast("l_int32") int copyflag );
public static native PIXC pixcompCreateFromString( @Cast("l_uint8*") byte[] data, @Cast("size_t") long size, @Cast("l_int32") int copyflag );
public static native PIXC pixcompCreateFromFile( @Cast("const char*") BytePointer filename, @Cast("l_int32") int comptype );
public static native PIXC pixcompCreateFromFile( String filename, @Cast("l_int32") int comptype );
public static native void pixcompDestroy( @Cast("PIXC**") PointerPointer ppixc );
public static native void pixcompDestroy( @ByPtrPtr PIXC ppixc );
public static native @Cast("l_int32") int pixcompGetDimensions( PIXC pixc, @Cast("l_int32*") IntPointer pw, @Cast("l_int32*") IntPointer ph, @Cast("l_int32*") IntPointer pd );
public static native @Cast("l_int32") int pixcompGetDimensions( PIXC pixc, @Cast("l_int32*") IntBuffer pw, @Cast("l_int32*") IntBuffer ph, @Cast("l_int32*") IntBuffer pd );
public static native @Cast("l_int32") int pixcompGetDimensions( PIXC pixc, @Cast("l_int32*") int[] pw, @Cast("l_int32*") int[] ph, @Cast("l_int32*") int[] pd );
public static native @Cast("l_int32") int pixcompDetermineFormat( @Cast("l_int32") int comptype, @Cast("l_int32") int d, @Cast("l_int32") int cmapflag, @Cast("l_int32*") IntPointer pformat );
public static native @Cast("l_int32") int pixcompDetermineFormat( @Cast("l_int32") int comptype, @Cast("l_int32") int d, @Cast("l_int32") int cmapflag, @Cast("l_int32*") IntBuffer pformat );
public static native @Cast("l_int32") int pixcompDetermineFormat( @Cast("l_int32") int comptype, @Cast("l_int32") int d, @Cast("l_int32") int cmapflag, @Cast("l_int32*") int[] pformat );
public static native PIX pixCreateFromPixcomp( PIXC pixc );
public static native PIXAC pixacompCreate( @Cast("l_int32") int n );
public static native PIXAC pixacompCreateWithInit( @Cast("l_int32") int n, @Cast("l_int32") int offset, PIX pix, @Cast("l_int32") int comptype );
public static native PIXAC pixacompCreateFromPixa( PIXA pixa, @Cast("l_int32") int comptype, @Cast("l_int32") int accesstype );
public static native PIXAC pixacompCreateFromFiles( @Cast("const char*") BytePointer dirname, @Cast("const char*") BytePointer substr, @Cast("l_int32") int comptype );
public static native PIXAC pixacompCreateFromFiles( String dirname, String substr, @Cast("l_int32") int comptype );
public static native PIXAC pixacompCreateFromSA( SARRAY sa, @Cast("l_int32") int comptype );
public static native void pixacompDestroy( @Cast("PIXAC**") PointerPointer ppixac );
public static native void pixacompDestroy( @ByPtrPtr PIXAC ppixac );
public static native @Cast("l_int32") int pixacompAddPix( PIXAC pixac, PIX pix, @Cast("l_int32") int comptype );
public static native @Cast("l_int32") int pixacompAddPixcomp( PIXAC pixac, PIXC pixc );
public static native @Cast("l_int32") int pixacompReplacePix( PIXAC pixac, @Cast("l_int32") int index, PIX pix, @Cast("l_int32") int comptype );
public static native @Cast("l_int32") int pixacompReplacePixcomp( PIXAC pixac, @Cast("l_int32") int index, PIXC pixc );
public static native @Cast("l_int32") int pixacompAddBox( PIXAC pixac, BOX box, @Cast("l_int32") int copyflag );
public static native @Cast("l_int32") int pixacompGetCount( PIXAC pixac );
public static native PIXC pixacompGetPixcomp( PIXAC pixac, @Cast("l_int32") int index );
public static native PIX pixacompGetPix( PIXAC pixac, @Cast("l_int32") int index );
public static native @Cast("l_int32") int pixacompGetPixDimensions( PIXAC pixac, @Cast("l_int32") int index, @Cast("l_int32*") IntPointer pw, @Cast("l_int32*") IntPointer ph, @Cast("l_int32*") IntPointer pd );
public static native @Cast("l_int32") int pixacompGetPixDimensions( PIXAC pixac, @Cast("l_int32") int index, @Cast("l_int32*") IntBuffer pw, @Cast("l_int32*") IntBuffer ph, @Cast("l_int32*") IntBuffer pd );
public static native @Cast("l_int32") int pixacompGetPixDimensions( PIXAC pixac, @Cast("l_int32") int index, @Cast("l_int32*") int[] pw, @Cast("l_int32*") int[] ph, @Cast("l_int32*") int[] pd );
public static native BOXA pixacompGetBoxa( PIXAC pixac, @Cast("l_int32") int accesstype );
public static native @Cast("l_int32") int pixacompGetBoxaCount( PIXAC pixac );
public static native BOX pixacompGetBox( PIXAC pixac, @Cast("l_int32") int index, @Cast("l_int32") int accesstype );
public static native @Cast("l_int32") int pixacompGetBoxGeometry( PIXAC pixac, @Cast("l_int32") int index, @Cast("l_int32*") IntPointer px, @Cast("l_int32*") IntPointer py, @Cast("l_int32*") IntPointer pw, @Cast("l_int32*") IntPointer ph );
public static native @Cast("l_int32") int pixacompGetBoxGeometry( PIXAC pixac, @Cast("l_int32") int index, @Cast("l_int32*") IntBuffer px, @Cast("l_int32*") IntBuffer py, @Cast("l_int32*") IntBuffer pw, @Cast("l_int32*") IntBuffer ph );
public static native @Cast("l_int32") int pixacompGetBoxGeometry( PIXAC pixac, @Cast("l_int32") int index, @Cast("l_int32*") int[] px, @Cast("l_int32*") int[] py, @Cast("l_int32*") int[] pw, @Cast("l_int32*") int[] ph );
public static native @Cast("l_int32") int pixacompGetOffset( PIXAC pixac );
public static native @Cast("l_int32") int pixacompSetOffset( PIXAC pixac, @Cast("l_int32") int offset );
public static native PIXA pixaCreateFromPixacomp( PIXAC pixac, @Cast("l_int32") int accesstype );
public static native PIXAC pixacompRead( @Cast("const char*") BytePointer filename );
public static native PIXAC pixacompRead( String filename );
public static native PIXAC pixacompReadStream( @Cast("FILE*") Pointer fp );
public static native @Cast("l_int32") int pixacompWrite( @Cast("const char*") BytePointer filename, PIXAC pixac );
public static native @Cast("l_int32") int pixacompWrite( String filename, PIXAC pixac );
public static native @Cast("l_int32") int pixacompWriteStream( @Cast("FILE*") Pointer fp, PIXAC pixac );
public static native @Cast("l_int32") int pixacompConvertToPdf( PIXAC pixac, @Cast("l_int32") int res, @Cast("l_float32") float scalefactor, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("const char*") BytePointer title, @Cast("const char*") BytePointer fileout );
public static native @Cast("l_int32") int pixacompConvertToPdf( PIXAC pixac, @Cast("l_int32") int res, @Cast("l_float32") float scalefactor, @Cast("l_int32") int type, @Cast("l_int32") int quality, String title, String fileout );
public static native @Cast("l_int32") int pixacompConvertToPdfData( PIXAC pixac, @Cast("l_int32") int res, @Cast("l_float32") float scalefactor, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("const char*") BytePointer title, @Cast("l_uint8**") PointerPointer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int pixacompConvertToPdfData( PIXAC pixac, @Cast("l_int32") int res, @Cast("l_float32") float scalefactor, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("const char*") BytePointer title, @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int pixacompConvertToPdfData( PIXAC pixac, @Cast("l_int32") int res, @Cast("l_float32") float scalefactor, @Cast("l_int32") int type, @Cast("l_int32") int quality, String title, @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int pixacompConvertToPdfData( PIXAC pixac, @Cast("l_int32") int res, @Cast("l_float32") float scalefactor, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("const char*") BytePointer title, @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int pixacompConvertToPdfData( PIXAC pixac, @Cast("l_int32") int res, @Cast("l_float32") float scalefactor, @Cast("l_int32") int type, @Cast("l_int32") int quality, String title, @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int pixacompConvertToPdfData( PIXAC pixac, @Cast("l_int32") int res, @Cast("l_float32") float scalefactor, @Cast("l_int32") int type, @Cast("l_int32") int quality, @Cast("const char*") BytePointer title, @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int pixacompConvertToPdfData( PIXAC pixac, @Cast("l_int32") int res, @Cast("l_float32") float scalefactor, @Cast("l_int32") int type, @Cast("l_int32") int quality, String title, @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int pixacompWriteStreamInfo( @Cast("FILE*") Pointer fp, PIXAC pixac, @Cast("const char*") BytePointer text );
public static native @Cast("l_int32") int pixacompWriteStreamInfo( @Cast("FILE*") Pointer fp, PIXAC pixac, String text );
public static native @Cast("l_int32") int pixcompWriteStreamInfo( @Cast("FILE*") Pointer fp, PIXC pixc, @Cast("const char*") BytePointer text );
public static native @Cast("l_int32") int pixcompWriteStreamInfo( @Cast("FILE*") Pointer fp, PIXC pixc, String text );
public static native PIX pixacompDisplayTiledAndScaled( PIXAC pixac, @Cast("l_int32") int outdepth, @Cast("l_int32") int tilewidth, @Cast("l_int32") int ncols, @Cast("l_int32") int background, @Cast("l_int32") int spacing, @Cast("l_int32") int border );
public static native PIX pixThreshold8( PIX pixs, @Cast("l_int32") int d, @Cast("l_int32") int nlevels, @Cast("l_int32") int cmapflag );
public static native PIX pixRemoveColormapGeneral( PIX pixs, @Cast("l_int32") int type, @Cast("l_int32") int ifnocmap );
public static native PIX pixRemoveColormap( PIX pixs, @Cast("l_int32") int type );
public static native @Cast("l_int32") int pixAddGrayColormap8( PIX pixs );
public static native PIX pixAddMinimalGrayColormap8( PIX pixs );
public static native PIX pixConvertRGBToLuminance( PIX pixs );
public static native PIX pixConvertRGBToGray( PIX pixs, @Cast("l_float32") float rwt, @Cast("l_float32") float gwt, @Cast("l_float32") float bwt );
public static native PIX pixConvertRGBToGrayFast( PIX pixs );
public static native PIX pixConvertRGBToGrayMinMax( PIX pixs, @Cast("l_int32") int type );
public static native PIX pixConvertRGBToGraySatBoost( PIX pixs, @Cast("l_int32") int refval );
public static native PIX pixConvertGrayToColormap( PIX pixs );
public static native PIX pixConvertGrayToColormap8( PIX pixs, @Cast("l_int32") int mindepth );
public static native PIX pixColorizeGray( PIX pixs, @Cast("l_uint32") int color, @Cast("l_int32") int cmapflag );
public static native PIX pixConvertRGBToColormap( PIX pixs, @Cast("l_int32") int ditherflag );
public static native @Cast("l_int32") int pixQuantizeIfFewColors( PIX pixs, @Cast("l_int32") int maxcolors, @Cast("l_int32") int mingraycolors, @Cast("l_int32") int octlevel, @Cast("PIX**") PointerPointer ppixd );
public static native @Cast("l_int32") int pixQuantizeIfFewColors( PIX pixs, @Cast("l_int32") int maxcolors, @Cast("l_int32") int mingraycolors, @Cast("l_int32") int octlevel, @ByPtrPtr PIX ppixd );
public static native PIX pixConvert16To8( PIX pixs, @Cast("l_int32") int type );
public static native PIX pixConvertGrayToFalseColor( PIX pixs, @Cast("l_float32") float gamma );
public static native PIX pixUnpackBinary( PIX pixs, @Cast("l_int32") int depth, @Cast("l_int32") int invert );
public static native PIX pixConvert1To16( PIX pixd, PIX pixs, @Cast("l_uint16") short val0, @Cast("l_uint16") short val1 );
public static native PIX pixConvert1To32( PIX pixd, PIX pixs, @Cast("l_uint32") int val0, @Cast("l_uint32") int val1 );
public static native PIX pixConvert1To2Cmap( PIX pixs );
public static native PIX pixConvert1To2( PIX pixd, PIX pixs, @Cast("l_int32") int val0, @Cast("l_int32") int val1 );
public static native PIX pixConvert1To4Cmap( PIX pixs );
public static native PIX pixConvert1To4( PIX pixd, PIX pixs, @Cast("l_int32") int val0, @Cast("l_int32") int val1 );
public static native PIX pixConvert1To8( PIX pixd, PIX pixs, @Cast("l_uint8") byte val0, @Cast("l_uint8") byte val1 );
public static native PIX pixConvert2To8( PIX pixs, @Cast("l_uint8") byte val0, @Cast("l_uint8") byte val1, @Cast("l_uint8") byte val2, @Cast("l_uint8") byte val3, @Cast("l_int32") int cmapflag );
public static native PIX pixConvert4To8( PIX pixs, @Cast("l_int32") int cmapflag );
public static native PIX pixConvert8To16( PIX pixs, @Cast("l_int32") int leftshift );
public static native PIX pixConvertTo1( PIX pixs, @Cast("l_int32") int threshold );
public static native PIX pixConvertTo1BySampling( PIX pixs, @Cast("l_int32") int factor, @Cast("l_int32") int threshold );
public static native PIX pixConvertTo8( PIX pixs, @Cast("l_int32") int cmapflag );
public static native PIX pixConvertTo8BySampling( PIX pixs, @Cast("l_int32") int factor, @Cast("l_int32") int cmapflag );
public static native PIX pixConvertTo8Color( PIX pixs, @Cast("l_int32") int dither );
public static native PIX pixConvertTo16( PIX pixs );
public static native PIX pixConvertTo32( PIX pixs );
public static native PIX pixConvertTo32BySampling( PIX pixs, @Cast("l_int32") int factor );
public static native PIX pixConvert8To32( PIX pixs );
public static native PIX pixConvertTo8Or32( PIX pixs, @Cast("l_int32") int copyflag, @Cast("l_int32") int warnflag );
public static native PIX pixConvert24To32( PIX pixs );
public static native PIX pixConvert32To24( PIX pixs );
public static native PIX pixRemoveAlpha( PIX pixs );
public static native PIX pixAddAlphaTo1bpp( PIX pixd, PIX pixs );
public static native PIX pixConvertLossless( PIX pixs, @Cast("l_int32") int d );
public static native PIX pixConvertForPSWrap( PIX pixs );
public static native PIX pixConvertToSubpixelRGB( PIX pixs, @Cast("l_float32") float scalex, @Cast("l_float32") float scaley, @Cast("l_int32") int order );
public static native PIX pixConvertGrayToSubpixelRGB( PIX pixs, @Cast("l_float32") float scalex, @Cast("l_float32") float scaley, @Cast("l_int32") int order );
public static native PIX pixConvertColorToSubpixelRGB( PIX pixs, @Cast("l_float32") float scalex, @Cast("l_float32") float scaley, @Cast("l_int32") int order );
public static native PIX pixConnCompTransform( PIX pixs, @Cast("l_int32") int connect, @Cast("l_int32") int depth );
public static native PIX pixConnCompAreaTransform( PIX pixs, @Cast("l_int32") int connect );
public static native PIX pixLocToColorTransform( PIX pixs );
public static native PIXTILING pixTilingCreate( PIX pixs, @Cast("l_int32") int nx, @Cast("l_int32") int ny, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int xoverlap, @Cast("l_int32") int yoverlap );
public static native void pixTilingDestroy( @Cast("PIXTILING**") PointerPointer ppt );
public static native void pixTilingDestroy( @ByPtrPtr PIXTILING ppt );
public static native @Cast("l_int32") int pixTilingGetCount( PIXTILING pt, @Cast("l_int32*") IntPointer pnx, @Cast("l_int32*") IntPointer pny );
public static native @Cast("l_int32") int pixTilingGetCount( PIXTILING pt, @Cast("l_int32*") IntBuffer pnx, @Cast("l_int32*") IntBuffer pny );
public static native @Cast("l_int32") int pixTilingGetCount( PIXTILING pt, @Cast("l_int32*") int[] pnx, @Cast("l_int32*") int[] pny );
public static native @Cast("l_int32") int pixTilingGetSize( PIXTILING pt, @Cast("l_int32*") IntPointer pw, @Cast("l_int32*") IntPointer ph );
public static native @Cast("l_int32") int pixTilingGetSize( PIXTILING pt, @Cast("l_int32*") IntBuffer pw, @Cast("l_int32*") IntBuffer ph );
public static native @Cast("l_int32") int pixTilingGetSize( PIXTILING pt, @Cast("l_int32*") int[] pw, @Cast("l_int32*") int[] ph );
public static native PIX pixTilingGetTile( PIXTILING pt, @Cast("l_int32") int i, @Cast("l_int32") int j );
public static native @Cast("l_int32") int pixTilingNoStripOnPaint( PIXTILING pt );
public static native @Cast("l_int32") int pixTilingPaintTile( PIX pixd, @Cast("l_int32") int i, @Cast("l_int32") int j, PIX pixs, PIXTILING pt );
public static native PIX pixReadStreamPng( @Cast("FILE*") Pointer fp );
public static native @Cast("l_int32") int readHeaderPng( @Cast("const char*") BytePointer filename, @Cast("l_int32*") IntPointer pw, @Cast("l_int32*") IntPointer ph, @Cast("l_int32*") IntPointer pbps, @Cast("l_int32*") IntPointer pspp, @Cast("l_int32*") IntPointer piscmap );
public static native @Cast("l_int32") int readHeaderPng( String filename, @Cast("l_int32*") IntBuffer pw, @Cast("l_int32*") IntBuffer ph, @Cast("l_int32*") IntBuffer pbps, @Cast("l_int32*") IntBuffer pspp, @Cast("l_int32*") IntBuffer piscmap );
public static native @Cast("l_int32") int readHeaderPng( @Cast("const char*") BytePointer filename, @Cast("l_int32*") int[] pw, @Cast("l_int32*") int[] ph, @Cast("l_int32*") int[] pbps, @Cast("l_int32*") int[] pspp, @Cast("l_int32*") int[] piscmap );
public static native @Cast("l_int32") int readHeaderPng( String filename, @Cast("l_int32*") IntPointer pw, @Cast("l_int32*") IntPointer ph, @Cast("l_int32*") IntPointer pbps, @Cast("l_int32*") IntPointer pspp, @Cast("l_int32*") IntPointer piscmap );
public static native @Cast("l_int32") int readHeaderPng( @Cast("const char*") BytePointer filename, @Cast("l_int32*") IntBuffer pw, @Cast("l_int32*") IntBuffer ph, @Cast("l_int32*") IntBuffer pbps, @Cast("l_int32*") IntBuffer pspp, @Cast("l_int32*") IntBuffer piscmap );
public static native @Cast("l_int32") int readHeaderPng( String filename, @Cast("l_int32*") int[] pw, @Cast("l_int32*") int[] ph, @Cast("l_int32*") int[] pbps, @Cast("l_int32*") int[] pspp, @Cast("l_int32*") int[] piscmap );
public static native @Cast("l_int32") int freadHeaderPng( @Cast("FILE*") Pointer fp, @Cast("l_int32*") IntPointer pw, @Cast("l_int32*") IntPointer ph, @Cast("l_int32*") IntPointer pbps, @Cast("l_int32*") IntPointer pspp, @Cast("l_int32*") IntPointer piscmap );
public static native @Cast("l_int32") int freadHeaderPng( @Cast("FILE*") Pointer fp, @Cast("l_int32*") IntBuffer pw, @Cast("l_int32*") IntBuffer ph, @Cast("l_int32*") IntBuffer pbps, @Cast("l_int32*") IntBuffer pspp, @Cast("l_int32*") IntBuffer piscmap );
public static native @Cast("l_int32") int freadHeaderPng( @Cast("FILE*") Pointer fp, @Cast("l_int32*") int[] pw, @Cast("l_int32*") int[] ph, @Cast("l_int32*") int[] pbps, @Cast("l_int32*") int[] pspp, @Cast("l_int32*") int[] piscmap );
public static native @Cast("l_int32") int readHeaderMemPng( @Cast("const l_uint8*") BytePointer data, @Cast("size_t") long size, @Cast("l_int32*") IntPointer pw, @Cast("l_int32*") IntPointer ph, @Cast("l_int32*") IntPointer pbps, @Cast("l_int32*") IntPointer pspp, @Cast("l_int32*") IntPointer piscmap );
public static native @Cast("l_int32") int readHeaderMemPng( @Cast("const l_uint8*") ByteBuffer data, @Cast("size_t") long size, @Cast("l_int32*") IntBuffer pw, @Cast("l_int32*") IntBuffer ph, @Cast("l_int32*") IntBuffer pbps, @Cast("l_int32*") IntBuffer pspp, @Cast("l_int32*") IntBuffer piscmap );
public static native @Cast("l_int32") int readHeaderMemPng( @Cast("const l_uint8*") byte[] data, @Cast("size_t") long size, @Cast("l_int32*") int[] pw, @Cast("l_int32*") int[] ph, @Cast("l_int32*") int[] pbps, @Cast("l_int32*") int[] pspp, @Cast("l_int32*") int[] piscmap );
public static native @Cast("l_int32") int fgetPngResolution( @Cast("FILE*") Pointer fp, @Cast("l_int32*") IntPointer pxres, @Cast("l_int32*") IntPointer pyres );
public static native @Cast("l_int32") int fgetPngResolution( @Cast("FILE*") Pointer fp, @Cast("l_int32*") IntBuffer pxres, @Cast("l_int32*") IntBuffer pyres );
public static native @Cast("l_int32") int fgetPngResolution( @Cast("FILE*") Pointer fp, @Cast("l_int32*") int[] pxres, @Cast("l_int32*") int[] pyres );
public static native @Cast("l_int32") int isPngInterlaced( @Cast("const char*") BytePointer filename, @Cast("l_int32*") IntPointer pinterlaced );
public static native @Cast("l_int32") int isPngInterlaced( String filename, @Cast("l_int32*") IntBuffer pinterlaced );
public static native @Cast("l_int32") int isPngInterlaced( @Cast("const char*") BytePointer filename, @Cast("l_int32*") int[] pinterlaced );
public static native @Cast("l_int32") int isPngInterlaced( String filename, @Cast("l_int32*") IntPointer pinterlaced );
public static native @Cast("l_int32") int isPngInterlaced( @Cast("const char*") BytePointer filename, @Cast("l_int32*") IntBuffer pinterlaced );
public static native @Cast("l_int32") int isPngInterlaced( String filename, @Cast("l_int32*") int[] pinterlaced );
public static native @Cast("l_int32") int fgetPngColormapInfo( @Cast("FILE*") Pointer fp, @Cast("PIXCMAP**") PointerPointer pcmap, @Cast("l_int32*") IntPointer ptransparency );
public static native @Cast("l_int32") int fgetPngColormapInfo( @Cast("FILE*") Pointer fp, @ByPtrPtr PIXCMAP pcmap, @Cast("l_int32*") IntPointer ptransparency );
public static native @Cast("l_int32") int fgetPngColormapInfo( @Cast("FILE*") Pointer fp, @ByPtrPtr PIXCMAP pcmap, @Cast("l_int32*") IntBuffer ptransparency );
public static native @Cast("l_int32") int fgetPngColormapInfo( @Cast("FILE*") Pointer fp, @ByPtrPtr PIXCMAP pcmap, @Cast("l_int32*") int[] ptransparency );
public static native @Cast("l_int32") int pixWritePng( @Cast("const char*") BytePointer filename, PIX pix, @Cast("l_float32") float gamma );
public static native @Cast("l_int32") int pixWritePng( String filename, PIX pix, @Cast("l_float32") float gamma );
public static native @Cast("l_int32") int pixWriteStreamPng( @Cast("FILE*") Pointer fp, PIX pix, @Cast("l_float32") float gamma );
public static native @Cast("l_int32") int pixSetZlibCompression( PIX pix, @Cast("l_int32") int compval );
public static native void l_pngSetReadStrip16To8( @Cast("l_int32") int flag );
public static native PIX pixReadMemPng( @Cast("const l_uint8*") BytePointer cdata, @Cast("size_t") long size );
public static native PIX pixReadMemPng( @Cast("const l_uint8*") ByteBuffer cdata, @Cast("size_t") long size );
public static native PIX pixReadMemPng( @Cast("const l_uint8*") byte[] cdata, @Cast("size_t") long size );
public static native @Cast("l_int32") int pixWriteMemPng( @Cast("l_uint8**") PointerPointer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix, @Cast("l_float32") float gamma );
public static native @Cast("l_int32") int pixWriteMemPng( @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix, @Cast("l_float32") float gamma );
public static native @Cast("l_int32") int pixWriteMemPng( @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix, @Cast("l_float32") float gamma );
public static native @Cast("l_int32") int pixWriteMemPng( @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer psize, PIX pix, @Cast("l_float32") float gamma );
public static native PIX pixReadStreamPnm( @Cast("FILE*") Pointer fp );
public static native @Cast("l_int32") int readHeaderPnm( @Cast("const char*") BytePointer filename, @Cast("l_int32*") IntPointer pw, @Cast("l_int32*") IntPointer ph, @Cast("l_int32*") IntPointer pd, @Cast("l_int32*") IntPointer ptype, @Cast("l_int32*") IntPointer pbps, @Cast("l_int32*") IntPointer pspp );
public static native @Cast("l_int32") int readHeaderPnm( String filename, @Cast("l_int32*") IntBuffer pw, @Cast("l_int32*") IntBuffer ph, @Cast("l_int32*") IntBuffer pd, @Cast("l_int32*") IntBuffer ptype, @Cast("l_int32*") IntBuffer pbps, @Cast("l_int32*") IntBuffer pspp );
public static native @Cast("l_int32") int readHeaderPnm( @Cast("const char*") BytePointer filename, @Cast("l_int32*") int[] pw, @Cast("l_int32*") int[] ph, @Cast("l_int32*") int[] pd, @Cast("l_int32*") int[] ptype, @Cast("l_int32*") int[] pbps, @Cast("l_int32*") int[] pspp );
public static native @Cast("l_int32") int readHeaderPnm( String filename, @Cast("l_int32*") IntPointer pw, @Cast("l_int32*") IntPointer ph, @Cast("l_int32*") IntPointer pd, @Cast("l_int32*") IntPointer ptype, @Cast("l_int32*") IntPointer pbps, @Cast("l_int32*") IntPointer pspp );
public static native @Cast("l_int32") int readHeaderPnm( @Cast("const char*") BytePointer filename, @Cast("l_int32*") IntBuffer pw, @Cast("l_int32*") IntBuffer ph, @Cast("l_int32*") IntBuffer pd, @Cast("l_int32*") IntBuffer ptype, @Cast("l_int32*") IntBuffer pbps, @Cast("l_int32*") IntBuffer pspp );
public static native @Cast("l_int32") int readHeaderPnm( String filename, @Cast("l_int32*") int[] pw, @Cast("l_int32*") int[] ph, @Cast("l_int32*") int[] pd, @Cast("l_int32*") int[] ptype, @Cast("l_int32*") int[] pbps, @Cast("l_int32*") int[] pspp );
public static native @Cast("l_int32") int freadHeaderPnm( @Cast("FILE*") Pointer fp, @Cast("l_int32*") IntPointer pw, @Cast("l_int32*") IntPointer ph, @Cast("l_int32*") IntPointer pd, @Cast("l_int32*") IntPointer ptype, @Cast("l_int32*") IntPointer pbps, @Cast("l_int32*") IntPointer pspp );
public static native @Cast("l_int32") int freadHeaderPnm( @Cast("FILE*") Pointer fp, @Cast("l_int32*") IntBuffer pw, @Cast("l_int32*") IntBuffer ph, @Cast("l_int32*") IntBuffer pd, @Cast("l_int32*") IntBuffer ptype, @Cast("l_int32*") IntBuffer pbps, @Cast("l_int32*") IntBuffer pspp );
public static native @Cast("l_int32") int freadHeaderPnm( @Cast("FILE*") Pointer fp, @Cast("l_int32*") int[] pw, @Cast("l_int32*") int[] ph, @Cast("l_int32*") int[] pd, @Cast("l_int32*") int[] ptype, @Cast("l_int32*") int[] pbps, @Cast("l_int32*") int[] pspp );
public static native @Cast("l_int32") int pixWriteStreamPnm( @Cast("FILE*") Pointer fp, PIX pix );
public static native @Cast("l_int32") int pixWriteStreamAsciiPnm( @Cast("FILE*") Pointer fp, PIX pix );
public static native PIX pixReadMemPnm( @Cast("const l_uint8*") BytePointer cdata, @Cast("size_t") long size );
public static native PIX pixReadMemPnm( @Cast("const l_uint8*") ByteBuffer cdata, @Cast("size_t") long size );
public static native PIX pixReadMemPnm( @Cast("const l_uint8*") byte[] cdata, @Cast("size_t") long size );
public static native @Cast("l_int32") int readHeaderMemPnm( @Cast("const l_uint8*") BytePointer cdata, @Cast("size_t") long size, @Cast("l_int32*") IntPointer pw, @Cast("l_int32*") IntPointer ph, @Cast("l_int32*") IntPointer pd, @Cast("l_int32*") IntPointer ptype, @Cast("l_int32*") IntPointer pbps, @Cast("l_int32*") IntPointer pspp );
public static native @Cast("l_int32") int readHeaderMemPnm( @Cast("const l_uint8*") ByteBuffer cdata, @Cast("size_t") long size, @Cast("l_int32*") IntBuffer pw, @Cast("l_int32*") IntBuffer ph, @Cast("l_int32*") IntBuffer pd, @Cast("l_int32*") IntBuffer ptype, @Cast("l_int32*") IntBuffer pbps, @Cast("l_int32*") IntBuffer pspp );
public static native @Cast("l_int32") int readHeaderMemPnm( @Cast("const l_uint8*") byte[] cdata, @Cast("size_t") long size, @Cast("l_int32*") int[] pw, @Cast("l_int32*") int[] ph, @Cast("l_int32*") int[] pd, @Cast("l_int32*") int[] ptype, @Cast("l_int32*") int[] pbps, @Cast("l_int32*") int[] pspp );
public static native @Cast("l_int32") int pixWriteMemPnm( @Cast("l_uint8**") PointerPointer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix );
public static native @Cast("l_int32") int pixWriteMemPnm( @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix );
public static native @Cast("l_int32") int pixWriteMemPnm( @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix );
public static native @Cast("l_int32") int pixWriteMemPnm( @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer psize, PIX pix );
public static native PIX pixProjectiveSampledPta( PIX pixs, PTA ptad, PTA ptas, @Cast("l_int32") int incolor );
public static native PIX pixProjectiveSampled( PIX pixs, @Cast("l_float32*") FloatPointer vc, @Cast("l_int32") int incolor );
public static native PIX pixProjectiveSampled( PIX pixs, @Cast("l_float32*") FloatBuffer vc, @Cast("l_int32") int incolor );
public static native PIX pixProjectiveSampled( PIX pixs, @Cast("l_float32*") float[] vc, @Cast("l_int32") int incolor );
public static native PIX pixProjectivePta( PIX pixs, PTA ptad, PTA ptas, @Cast("l_int32") int incolor );
public static native PIX pixProjective( PIX pixs, @Cast("l_float32*") FloatPointer vc, @Cast("l_int32") int incolor );
public static native PIX pixProjective( PIX pixs, @Cast("l_float32*") FloatBuffer vc, @Cast("l_int32") int incolor );
public static native PIX pixProjective( PIX pixs, @Cast("l_float32*") float[] vc, @Cast("l_int32") int incolor );
public static native PIX pixProjectivePtaColor( PIX pixs, PTA ptad, PTA ptas, @Cast("l_uint32") int colorval );
public static native PIX pixProjectiveColor( PIX pixs, @Cast("l_float32*") FloatPointer vc, @Cast("l_uint32") int colorval );
public static native PIX pixProjectiveColor( PIX pixs, @Cast("l_float32*") FloatBuffer vc, @Cast("l_uint32") int colorval );
public static native PIX pixProjectiveColor( PIX pixs, @Cast("l_float32*") float[] vc, @Cast("l_uint32") int colorval );
public static native PIX pixProjectivePtaGray( PIX pixs, PTA ptad, PTA ptas, @Cast("l_uint8") byte grayval );
public static native PIX pixProjectiveGray( PIX pixs, @Cast("l_float32*") FloatPointer vc, @Cast("l_uint8") byte grayval );
public static native PIX pixProjectiveGray( PIX pixs, @Cast("l_float32*") FloatBuffer vc, @Cast("l_uint8") byte grayval );
public static native PIX pixProjectiveGray( PIX pixs, @Cast("l_float32*") float[] vc, @Cast("l_uint8") byte grayval );
public static native PIX pixProjectivePtaWithAlpha( PIX pixs, PTA ptad, PTA ptas, PIX pixg, @Cast("l_float32") float fract, @Cast("l_int32") int border );
public static native @Cast("l_int32") int getProjectiveXformCoeffs( PTA ptas, PTA ptad, @Cast("l_float32**") PointerPointer pvc );
public static native @Cast("l_int32") int getProjectiveXformCoeffs( PTA ptas, PTA ptad, @Cast("l_float32**") @ByPtrPtr FloatPointer pvc );
public static native @Cast("l_int32") int getProjectiveXformCoeffs( PTA ptas, PTA ptad, @Cast("l_float32**") @ByPtrPtr FloatBuffer pvc );
public static native @Cast("l_int32") int getProjectiveXformCoeffs( PTA ptas, PTA ptad, @Cast("l_float32**") @ByPtrPtr float[] pvc );
public static native @Cast("l_int32") int projectiveXformSampledPt( @Cast("l_float32*") FloatPointer vc, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32*") IntPointer pxp, @Cast("l_int32*") IntPointer pyp );
public static native @Cast("l_int32") int projectiveXformSampledPt( @Cast("l_float32*") FloatBuffer vc, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32*") IntBuffer pxp, @Cast("l_int32*") IntBuffer pyp );
public static native @Cast("l_int32") int projectiveXformSampledPt( @Cast("l_float32*") float[] vc, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32*") int[] pxp, @Cast("l_int32*") int[] pyp );
public static native @Cast("l_int32") int projectiveXformPt( @Cast("l_float32*") FloatPointer vc, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_float32*") FloatPointer pxp, @Cast("l_float32*") FloatPointer pyp );
public static native @Cast("l_int32") int projectiveXformPt( @Cast("l_float32*") FloatBuffer vc, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_float32*") FloatBuffer pxp, @Cast("l_float32*") FloatBuffer pyp );
public static native @Cast("l_int32") int projectiveXformPt( @Cast("l_float32*") float[] vc, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_float32*") float[] pxp, @Cast("l_float32*") float[] pyp );
public static native @Cast("l_int32") int convertFilesToPS( @Cast("const char*") BytePointer dirin, @Cast("const char*") BytePointer substr, @Cast("l_int32") int res, @Cast("const char*") BytePointer fileout );
public static native @Cast("l_int32") int convertFilesToPS( String dirin, String substr, @Cast("l_int32") int res, String fileout );
public static native @Cast("l_int32") int sarrayConvertFilesToPS( SARRAY sa, @Cast("l_int32") int res, @Cast("const char*") BytePointer fileout );
public static native @Cast("l_int32") int sarrayConvertFilesToPS( SARRAY sa, @Cast("l_int32") int res, String fileout );
public static native @Cast("l_int32") int convertFilesFittedToPS( @Cast("const char*") BytePointer dirin, @Cast("const char*") BytePointer substr, @Cast("l_float32") float xpts, @Cast("l_float32") float ypts, @Cast("const char*") BytePointer fileout );
public static native @Cast("l_int32") int convertFilesFittedToPS( String dirin, String substr, @Cast("l_float32") float xpts, @Cast("l_float32") float ypts, String fileout );
public static native @Cast("l_int32") int sarrayConvertFilesFittedToPS( SARRAY sa, @Cast("l_float32") float xpts, @Cast("l_float32") float ypts, @Cast("const char*") BytePointer fileout );
public static native @Cast("l_int32") int sarrayConvertFilesFittedToPS( SARRAY sa, @Cast("l_float32") float xpts, @Cast("l_float32") float ypts, String fileout );
public static native @Cast("l_int32") int writeImageCompressedToPSFile( @Cast("const char*") BytePointer filein, @Cast("const char*") BytePointer fileout, @Cast("l_int32") int res, @Cast("l_int32*") IntPointer pfirstfile, @Cast("l_int32*") IntPointer pindex );
public static native @Cast("l_int32") int writeImageCompressedToPSFile( String filein, String fileout, @Cast("l_int32") int res, @Cast("l_int32*") IntBuffer pfirstfile, @Cast("l_int32*") IntBuffer pindex );
public static native @Cast("l_int32") int writeImageCompressedToPSFile( @Cast("const char*") BytePointer filein, @Cast("const char*") BytePointer fileout, @Cast("l_int32") int res, @Cast("l_int32*") int[] pfirstfile, @Cast("l_int32*") int[] pindex );
public static native @Cast("l_int32") int writeImageCompressedToPSFile( String filein, String fileout, @Cast("l_int32") int res, @Cast("l_int32*") IntPointer pfirstfile, @Cast("l_int32*") IntPointer pindex );
public static native @Cast("l_int32") int writeImageCompressedToPSFile( @Cast("const char*") BytePointer filein, @Cast("const char*") BytePointer fileout, @Cast("l_int32") int res, @Cast("l_int32*") IntBuffer pfirstfile, @Cast("l_int32*") IntBuffer pindex );
public static native @Cast("l_int32") int writeImageCompressedToPSFile( String filein, String fileout, @Cast("l_int32") int res, @Cast("l_int32*") int[] pfirstfile, @Cast("l_int32*") int[] pindex );
public static native @Cast("l_int32") int convertSegmentedPagesToPS( @Cast("const char*") BytePointer pagedir, @Cast("const char*") BytePointer pagestr, @Cast("l_int32") int page_numpre, @Cast("const char*") BytePointer maskdir, @Cast("const char*") BytePointer maskstr, @Cast("l_int32") int mask_numpre, @Cast("l_int32") int numpost, @Cast("l_int32") int maxnum, @Cast("l_float32") float textscale, @Cast("l_float32") float imagescale, @Cast("l_int32") int threshold, @Cast("const char*") BytePointer fileout );
public static native @Cast("l_int32") int convertSegmentedPagesToPS( String pagedir, String pagestr, @Cast("l_int32") int page_numpre, String maskdir, String maskstr, @Cast("l_int32") int mask_numpre, @Cast("l_int32") int numpost, @Cast("l_int32") int maxnum, @Cast("l_float32") float textscale, @Cast("l_float32") float imagescale, @Cast("l_int32") int threshold, String fileout );
public static native @Cast("l_int32") int pixWriteSegmentedPageToPS( PIX pixs, PIX pixm, @Cast("l_float32") float textscale, @Cast("l_float32") float imagescale, @Cast("l_int32") int threshold, @Cast("l_int32") int pageno, @Cast("const char*") BytePointer fileout );
public static native @Cast("l_int32") int pixWriteSegmentedPageToPS( PIX pixs, PIX pixm, @Cast("l_float32") float textscale, @Cast("l_float32") float imagescale, @Cast("l_int32") int threshold, @Cast("l_int32") int pageno, String fileout );
public static native @Cast("l_int32") int pixWriteMixedToPS( PIX pixb, PIX pixc, @Cast("l_float32") float scale, @Cast("l_int32") int pageno, @Cast("const char*") BytePointer fileout );
public static native @Cast("l_int32") int pixWriteMixedToPS( PIX pixb, PIX pixc, @Cast("l_float32") float scale, @Cast("l_int32") int pageno, String fileout );
public static native @Cast("l_int32") int convertToPSEmbed( @Cast("const char*") BytePointer filein, @Cast("const char*") BytePointer fileout, @Cast("l_int32") int level );
public static native @Cast("l_int32") int convertToPSEmbed( String filein, String fileout, @Cast("l_int32") int level );
public static native @Cast("l_int32") int pixaWriteCompressedToPS( PIXA pixa, @Cast("const char*") BytePointer fileout, @Cast("l_int32") int res, @Cast("l_int32") int level );
public static native @Cast("l_int32") int pixaWriteCompressedToPS( PIXA pixa, String fileout, @Cast("l_int32") int res, @Cast("l_int32") int level );
public static native @Cast("l_int32") int pixWritePSEmbed( @Cast("const char*") BytePointer filein, @Cast("const char*") BytePointer fileout );
public static native @Cast("l_int32") int pixWritePSEmbed( String filein, String fileout );
public static native @Cast("l_int32") int pixWriteStreamPS( @Cast("FILE*") Pointer fp, PIX pix, BOX box, @Cast("l_int32") int res, @Cast("l_float32") float scale );
public static native @Cast("char*") BytePointer pixWriteStringPS( PIX pixs, BOX box, @Cast("l_int32") int res, @Cast("l_float32") float scale );
public static native @Cast("char*") BytePointer generateUncompressedPS( @Cast("char*") BytePointer hexdata, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int d, @Cast("l_int32") int psbpl, @Cast("l_int32") int bps, @Cast("l_float32") float xpt, @Cast("l_float32") float ypt, @Cast("l_float32") float wpt, @Cast("l_float32") float hpt, @Cast("l_int32") int boxflag );
public static native @Cast("char*") ByteBuffer generateUncompressedPS( @Cast("char*") ByteBuffer hexdata, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int d, @Cast("l_int32") int psbpl, @Cast("l_int32") int bps, @Cast("l_float32") float xpt, @Cast("l_float32") float ypt, @Cast("l_float32") float wpt, @Cast("l_float32") float hpt, @Cast("l_int32") int boxflag );
public static native @Cast("char*") byte[] generateUncompressedPS( @Cast("char*") byte[] hexdata, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int d, @Cast("l_int32") int psbpl, @Cast("l_int32") int bps, @Cast("l_float32") float xpt, @Cast("l_float32") float ypt, @Cast("l_float32") float wpt, @Cast("l_float32") float hpt, @Cast("l_int32") int boxflag );
public static native void getScaledParametersPS( BOX box, @Cast("l_int32") int wpix, @Cast("l_int32") int hpix, @Cast("l_int32") int res, @Cast("l_float32") float scale, @Cast("l_float32*") FloatPointer pxpt, @Cast("l_float32*") FloatPointer pypt, @Cast("l_float32*") FloatPointer pwpt, @Cast("l_float32*") FloatPointer phpt );
public static native void getScaledParametersPS( BOX box, @Cast("l_int32") int wpix, @Cast("l_int32") int hpix, @Cast("l_int32") int res, @Cast("l_float32") float scale, @Cast("l_float32*") FloatBuffer pxpt, @Cast("l_float32*") FloatBuffer pypt, @Cast("l_float32*") FloatBuffer pwpt, @Cast("l_float32*") FloatBuffer phpt );
public static native void getScaledParametersPS( BOX box, @Cast("l_int32") int wpix, @Cast("l_int32") int hpix, @Cast("l_int32") int res, @Cast("l_float32") float scale, @Cast("l_float32*") float[] pxpt, @Cast("l_float32*") float[] pypt, @Cast("l_float32*") float[] pwpt, @Cast("l_float32*") float[] phpt );
public static native void convertByteToHexAscii( @Cast("l_uint8") byte byteval, @Cast("char*") BytePointer pnib1, @Cast("char*") BytePointer pnib2 );
public static native void convertByteToHexAscii( @Cast("l_uint8") byte byteval, @Cast("char*") ByteBuffer pnib1, @Cast("char*") ByteBuffer pnib2 );
public static native void convertByteToHexAscii( @Cast("l_uint8") byte byteval, @Cast("char*") byte[] pnib1, @Cast("char*") byte[] pnib2 );
public static native @Cast("l_int32") int convertJpegToPSEmbed( @Cast("const char*") BytePointer filein, @Cast("const char*") BytePointer fileout );
public static native @Cast("l_int32") int convertJpegToPSEmbed( String filein, String fileout );
public static native @Cast("l_int32") int convertJpegToPS( @Cast("const char*") BytePointer filein, @Cast("const char*") BytePointer fileout, @Cast("const char*") BytePointer operation, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, @Cast("l_float32") float scale, @Cast("l_int32") int pageno, @Cast("l_int32") int endpage );
public static native @Cast("l_int32") int convertJpegToPS( String filein, String fileout, String operation, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, @Cast("l_float32") float scale, @Cast("l_int32") int pageno, @Cast("l_int32") int endpage );
public static native @Cast("l_int32") int convertJpegToPSString( @Cast("const char*") BytePointer filein, @Cast("char**") PointerPointer poutstr, @Cast("l_int32*") IntPointer pnbytes, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, @Cast("l_float32") float scale, @Cast("l_int32") int pageno, @Cast("l_int32") int endpage );
public static native @Cast("l_int32") int convertJpegToPSString( @Cast("const char*") BytePointer filein, @Cast("char**") @ByPtrPtr BytePointer poutstr, @Cast("l_int32*") IntPointer pnbytes, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, @Cast("l_float32") float scale, @Cast("l_int32") int pageno, @Cast("l_int32") int endpage );
public static native @Cast("l_int32") int convertJpegToPSString( String filein, @Cast("char**") @ByPtrPtr ByteBuffer poutstr, @Cast("l_int32*") IntBuffer pnbytes, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, @Cast("l_float32") float scale, @Cast("l_int32") int pageno, @Cast("l_int32") int endpage );
public static native @Cast("l_int32") int convertJpegToPSString( @Cast("const char*") BytePointer filein, @Cast("char**") @ByPtrPtr byte[] poutstr, @Cast("l_int32*") int[] pnbytes, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, @Cast("l_float32") float scale, @Cast("l_int32") int pageno, @Cast("l_int32") int endpage );
public static native @Cast("l_int32") int convertJpegToPSString( String filein, @Cast("char**") @ByPtrPtr BytePointer poutstr, @Cast("l_int32*") IntPointer pnbytes, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, @Cast("l_float32") float scale, @Cast("l_int32") int pageno, @Cast("l_int32") int endpage );
public static native @Cast("l_int32") int convertJpegToPSString( @Cast("const char*") BytePointer filein, @Cast("char**") @ByPtrPtr ByteBuffer poutstr, @Cast("l_int32*") IntBuffer pnbytes, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, @Cast("l_float32") float scale, @Cast("l_int32") int pageno, @Cast("l_int32") int endpage );
public static native @Cast("l_int32") int convertJpegToPSString( String filein, @Cast("char**") @ByPtrPtr byte[] poutstr, @Cast("l_int32*") int[] pnbytes, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, @Cast("l_float32") float scale, @Cast("l_int32") int pageno, @Cast("l_int32") int endpage );
public static native @Cast("char*") BytePointer generateJpegPS( @Cast("const char*") BytePointer filein, L_COMP_DATA cid, @Cast("l_float32") float xpt, @Cast("l_float32") float ypt, @Cast("l_float32") float wpt, @Cast("l_float32") float hpt, @Cast("l_int32") int pageno, @Cast("l_int32") int endpage );
public static native @Cast("char*") ByteBuffer generateJpegPS( String filein, L_COMP_DATA cid, @Cast("l_float32") float xpt, @Cast("l_float32") float ypt, @Cast("l_float32") float wpt, @Cast("l_float32") float hpt, @Cast("l_int32") int pageno, @Cast("l_int32") int endpage );
public static native @Cast("l_int32") int convertG4ToPSEmbed( @Cast("const char*") BytePointer filein, @Cast("const char*") BytePointer fileout );
public static native @Cast("l_int32") int convertG4ToPSEmbed( String filein, String fileout );
public static native @Cast("l_int32") int convertG4ToPS( @Cast("const char*") BytePointer filein, @Cast("const char*") BytePointer fileout, @Cast("const char*") BytePointer operation, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, @Cast("l_float32") float scale, @Cast("l_int32") int pageno, @Cast("l_int32") int maskflag, @Cast("l_int32") int endpage );
public static native @Cast("l_int32") int convertG4ToPS( String filein, String fileout, String operation, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, @Cast("l_float32") float scale, @Cast("l_int32") int pageno, @Cast("l_int32") int maskflag, @Cast("l_int32") int endpage );
public static native @Cast("l_int32") int convertG4ToPSString( @Cast("const char*") BytePointer filein, @Cast("char**") PointerPointer poutstr, @Cast("l_int32*") IntPointer pnbytes, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, @Cast("l_float32") float scale, @Cast("l_int32") int pageno, @Cast("l_int32") int maskflag, @Cast("l_int32") int endpage );
public static native @Cast("l_int32") int convertG4ToPSString( @Cast("const char*") BytePointer filein, @Cast("char**") @ByPtrPtr BytePointer poutstr, @Cast("l_int32*") IntPointer pnbytes, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, @Cast("l_float32") float scale, @Cast("l_int32") int pageno, @Cast("l_int32") int maskflag, @Cast("l_int32") int endpage );
public static native @Cast("l_int32") int convertG4ToPSString( String filein, @Cast("char**") @ByPtrPtr ByteBuffer poutstr, @Cast("l_int32*") IntBuffer pnbytes, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, @Cast("l_float32") float scale, @Cast("l_int32") int pageno, @Cast("l_int32") int maskflag, @Cast("l_int32") int endpage );
public static native @Cast("l_int32") int convertG4ToPSString( @Cast("const char*") BytePointer filein, @Cast("char**") @ByPtrPtr byte[] poutstr, @Cast("l_int32*") int[] pnbytes, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, @Cast("l_float32") float scale, @Cast("l_int32") int pageno, @Cast("l_int32") int maskflag, @Cast("l_int32") int endpage );
public static native @Cast("l_int32") int convertG4ToPSString( String filein, @Cast("char**") @ByPtrPtr BytePointer poutstr, @Cast("l_int32*") IntPointer pnbytes, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, @Cast("l_float32") float scale, @Cast("l_int32") int pageno, @Cast("l_int32") int maskflag, @Cast("l_int32") int endpage );
public static native @Cast("l_int32") int convertG4ToPSString( @Cast("const char*") BytePointer filein, @Cast("char**") @ByPtrPtr ByteBuffer poutstr, @Cast("l_int32*") IntBuffer pnbytes, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, @Cast("l_float32") float scale, @Cast("l_int32") int pageno, @Cast("l_int32") int maskflag, @Cast("l_int32") int endpage );
public static native @Cast("l_int32") int convertG4ToPSString( String filein, @Cast("char**") @ByPtrPtr byte[] poutstr, @Cast("l_int32*") int[] pnbytes, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, @Cast("l_float32") float scale, @Cast("l_int32") int pageno, @Cast("l_int32") int maskflag, @Cast("l_int32") int endpage );
public static native @Cast("char*") BytePointer generateG4PS( @Cast("const char*") BytePointer filein, L_COMP_DATA cid, @Cast("l_float32") float xpt, @Cast("l_float32") float ypt, @Cast("l_float32") float wpt, @Cast("l_float32") float hpt, @Cast("l_int32") int maskflag, @Cast("l_int32") int pageno, @Cast("l_int32") int endpage );
public static native @Cast("char*") ByteBuffer generateG4PS( String filein, L_COMP_DATA cid, @Cast("l_float32") float xpt, @Cast("l_float32") float ypt, @Cast("l_float32") float wpt, @Cast("l_float32") float hpt, @Cast("l_int32") int maskflag, @Cast("l_int32") int pageno, @Cast("l_int32") int endpage );
public static native @Cast("l_int32") int convertTiffMultipageToPS( @Cast("const char*") BytePointer filein, @Cast("const char*") BytePointer fileout, @Cast("const char*") BytePointer tempfile, @Cast("l_float32") float fillfract );
public static native @Cast("l_int32") int convertTiffMultipageToPS( String filein, String fileout, String tempfile, @Cast("l_float32") float fillfract );
public static native @Cast("l_int32") int convertFlateToPSEmbed( @Cast("const char*") BytePointer filein, @Cast("const char*") BytePointer fileout );
public static native @Cast("l_int32") int convertFlateToPSEmbed( String filein, String fileout );
public static native @Cast("l_int32") int convertFlateToPS( @Cast("const char*") BytePointer filein, @Cast("const char*") BytePointer fileout, @Cast("const char*") BytePointer operation, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, @Cast("l_float32") float scale, @Cast("l_int32") int pageno, @Cast("l_int32") int endpage );
public static native @Cast("l_int32") int convertFlateToPS( String filein, String fileout, String operation, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, @Cast("l_float32") float scale, @Cast("l_int32") int pageno, @Cast("l_int32") int endpage );
public static native @Cast("l_int32") int convertFlateToPSString( @Cast("const char*") BytePointer filein, @Cast("char**") PointerPointer poutstr, @Cast("l_int32*") IntPointer pnbytes, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, @Cast("l_float32") float scale, @Cast("l_int32") int pageno, @Cast("l_int32") int endpage );
public static native @Cast("l_int32") int convertFlateToPSString( @Cast("const char*") BytePointer filein, @Cast("char**") @ByPtrPtr BytePointer poutstr, @Cast("l_int32*") IntPointer pnbytes, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, @Cast("l_float32") float scale, @Cast("l_int32") int pageno, @Cast("l_int32") int endpage );
public static native @Cast("l_int32") int convertFlateToPSString( String filein, @Cast("char**") @ByPtrPtr ByteBuffer poutstr, @Cast("l_int32*") IntBuffer pnbytes, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, @Cast("l_float32") float scale, @Cast("l_int32") int pageno, @Cast("l_int32") int endpage );
public static native @Cast("l_int32") int convertFlateToPSString( @Cast("const char*") BytePointer filein, @Cast("char**") @ByPtrPtr byte[] poutstr, @Cast("l_int32*") int[] pnbytes, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, @Cast("l_float32") float scale, @Cast("l_int32") int pageno, @Cast("l_int32") int endpage );
public static native @Cast("l_int32") int convertFlateToPSString( String filein, @Cast("char**") @ByPtrPtr BytePointer poutstr, @Cast("l_int32*") IntPointer pnbytes, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, @Cast("l_float32") float scale, @Cast("l_int32") int pageno, @Cast("l_int32") int endpage );
public static native @Cast("l_int32") int convertFlateToPSString( @Cast("const char*") BytePointer filein, @Cast("char**") @ByPtrPtr ByteBuffer poutstr, @Cast("l_int32*") IntBuffer pnbytes, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, @Cast("l_float32") float scale, @Cast("l_int32") int pageno, @Cast("l_int32") int endpage );
public static native @Cast("l_int32") int convertFlateToPSString( String filein, @Cast("char**") @ByPtrPtr byte[] poutstr, @Cast("l_int32*") int[] pnbytes, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int res, @Cast("l_float32") float scale, @Cast("l_int32") int pageno, @Cast("l_int32") int endpage );
public static native @Cast("char*") BytePointer generateFlatePS( @Cast("const char*") BytePointer filein, L_COMP_DATA cid, @Cast("l_float32") float xpt, @Cast("l_float32") float ypt, @Cast("l_float32") float wpt, @Cast("l_float32") float hpt, @Cast("l_int32") int pageno, @Cast("l_int32") int endpage );
public static native @Cast("char*") ByteBuffer generateFlatePS( String filein, L_COMP_DATA cid, @Cast("l_float32") float xpt, @Cast("l_float32") float ypt, @Cast("l_float32") float wpt, @Cast("l_float32") float hpt, @Cast("l_int32") int pageno, @Cast("l_int32") int endpage );
public static native @Cast("l_int32") int pixWriteMemPS( @Cast("l_uint8**") PointerPointer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix, BOX box, @Cast("l_int32") int res, @Cast("l_float32") float scale );
public static native @Cast("l_int32") int pixWriteMemPS( @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix, BOX box, @Cast("l_int32") int res, @Cast("l_float32") float scale );
public static native @Cast("l_int32") int pixWriteMemPS( @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix, BOX box, @Cast("l_int32") int res, @Cast("l_float32") float scale );
public static native @Cast("l_int32") int pixWriteMemPS( @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer psize, PIX pix, BOX box, @Cast("l_int32") int res, @Cast("l_float32") float scale );
public static native @Cast("l_int32") int getResLetterPage( @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_float32") float fillfract );
public static native @Cast("l_int32") int getResA4Page( @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_float32") float fillfract );
public static native void l_psWriteBoundingBox( @Cast("l_int32") int flag );
public static native PTA ptaCreate( @Cast("l_int32") int n );
public static native PTA ptaCreateFromNuma( NUMA nax, NUMA nay );
public static native void ptaDestroy( @Cast("PTA**") PointerPointer ppta );
public static native void ptaDestroy( @ByPtrPtr PTA ppta );
public static native PTA ptaCopy( PTA pta );
public static native PTA ptaCopyRange( PTA ptas, @Cast("l_int32") int istart, @Cast("l_int32") int iend );
public static native PTA ptaClone( PTA pta );
public static native @Cast("l_int32") int ptaEmpty( PTA pta );
public static native @Cast("l_int32") int ptaAddPt( PTA pta, @Cast("l_float32") float x, @Cast("l_float32") float y );
public static native @Cast("l_int32") int ptaInsertPt( PTA pta, @Cast("l_int32") int index, @Cast("l_int32") int x, @Cast("l_int32") int y );
public static native @Cast("l_int32") int ptaRemovePt( PTA pta, @Cast("l_int32") int index );
public static native @Cast("l_int32") int ptaGetRefcount( PTA pta );
public static native @Cast("l_int32") int ptaChangeRefcount( PTA pta, @Cast("l_int32") int delta );
public static native @Cast("l_int32") int ptaGetCount( PTA pta );
public static native @Cast("l_int32") int ptaGetPt( PTA pta, @Cast("l_int32") int index, @Cast("l_float32*") FloatPointer px, @Cast("l_float32*") FloatPointer py );
public static native @Cast("l_int32") int ptaGetPt( PTA pta, @Cast("l_int32") int index, @Cast("l_float32*") FloatBuffer px, @Cast("l_float32*") FloatBuffer py );
public static native @Cast("l_int32") int ptaGetPt( PTA pta, @Cast("l_int32") int index, @Cast("l_float32*") float[] px, @Cast("l_float32*") float[] py );
public static native @Cast("l_int32") int ptaGetIPt( PTA pta, @Cast("l_int32") int index, @Cast("l_int32*") IntPointer px, @Cast("l_int32*") IntPointer py );
public static native @Cast("l_int32") int ptaGetIPt( PTA pta, @Cast("l_int32") int index, @Cast("l_int32*") IntBuffer px, @Cast("l_int32*") IntBuffer py );
public static native @Cast("l_int32") int ptaGetIPt( PTA pta, @Cast("l_int32") int index, @Cast("l_int32*") int[] px, @Cast("l_int32*") int[] py );
public static native @Cast("l_int32") int ptaSetPt( PTA pta, @Cast("l_int32") int index, @Cast("l_float32") float x, @Cast("l_float32") float y );
public static native @Cast("l_int32") int ptaGetArrays( PTA pta, @Cast("NUMA**") PointerPointer pnax, @Cast("NUMA**") PointerPointer pnay );
public static native @Cast("l_int32") int ptaGetArrays( PTA pta, @ByPtrPtr NUMA pnax, @ByPtrPtr NUMA pnay );
public static native PTA ptaRead( @Cast("const char*") BytePointer filename );
public static native PTA ptaRead( String filename );
public static native PTA ptaReadStream( @Cast("FILE*") Pointer fp );
public static native @Cast("l_int32") int ptaWrite( @Cast("const char*") BytePointer filename, PTA pta, @Cast("l_int32") int type );
public static native @Cast("l_int32") int ptaWrite( String filename, PTA pta, @Cast("l_int32") int type );
public static native @Cast("l_int32") int ptaWriteStream( @Cast("FILE*") Pointer fp, PTA pta, @Cast("l_int32") int type );
public static native PTAA ptaaCreate( @Cast("l_int32") int n );
public static native void ptaaDestroy( @Cast("PTAA**") PointerPointer pptaa );
public static native void ptaaDestroy( @ByPtrPtr PTAA pptaa );
public static native @Cast("l_int32") int ptaaAddPta( PTAA ptaa, PTA pta, @Cast("l_int32") int copyflag );
public static native @Cast("l_int32") int ptaaGetCount( PTAA ptaa );
public static native PTA ptaaGetPta( PTAA ptaa, @Cast("l_int32") int index, @Cast("l_int32") int accessflag );
public static native @Cast("l_int32") int ptaaGetPt( PTAA ptaa, @Cast("l_int32") int ipta, @Cast("l_int32") int jpt, @Cast("l_float32*") FloatPointer px, @Cast("l_float32*") FloatPointer py );
public static native @Cast("l_int32") int ptaaGetPt( PTAA ptaa, @Cast("l_int32") int ipta, @Cast("l_int32") int jpt, @Cast("l_float32*") FloatBuffer px, @Cast("l_float32*") FloatBuffer py );
public static native @Cast("l_int32") int ptaaGetPt( PTAA ptaa, @Cast("l_int32") int ipta, @Cast("l_int32") int jpt, @Cast("l_float32*") float[] px, @Cast("l_float32*") float[] py );
public static native @Cast("l_int32") int ptaaInitFull( PTAA ptaa, PTA pta );
public static native @Cast("l_int32") int ptaaReplacePta( PTAA ptaa, @Cast("l_int32") int index, PTA pta );
public static native @Cast("l_int32") int ptaaAddPt( PTAA ptaa, @Cast("l_int32") int ipta, @Cast("l_float32") float x, @Cast("l_float32") float y );
public static native @Cast("l_int32") int ptaaTruncate( PTAA ptaa );
public static native PTAA ptaaRead( @Cast("const char*") BytePointer filename );
public static native PTAA ptaaRead( String filename );
public static native PTAA ptaaReadStream( @Cast("FILE*") Pointer fp );
public static native @Cast("l_int32") int ptaaWrite( @Cast("const char*") BytePointer filename, PTAA ptaa, @Cast("l_int32") int type );
public static native @Cast("l_int32") int ptaaWrite( String filename, PTAA ptaa, @Cast("l_int32") int type );
public static native @Cast("l_int32") int ptaaWriteStream( @Cast("FILE*") Pointer fp, PTAA ptaa, @Cast("l_int32") int type );
public static native PTA ptaSubsample( PTA ptas, @Cast("l_int32") int subfactor );
public static native @Cast("l_int32") int ptaJoin( PTA ptad, PTA ptas, @Cast("l_int32") int istart, @Cast("l_int32") int iend );
public static native @Cast("l_int32") int ptaaJoin( PTAA ptaad, PTAA ptaas, @Cast("l_int32") int istart, @Cast("l_int32") int iend );
public static native PTA ptaReverse( PTA ptas, @Cast("l_int32") int type );
public static native PTA ptaTranspose( PTA ptas );
public static native PTA ptaCyclicPerm( PTA ptas, @Cast("l_int32") int xs, @Cast("l_int32") int ys );
public static native PTA ptaSort( PTA ptas, @Cast("l_int32") int sorttype, @Cast("l_int32") int sortorder, @Cast("NUMA**") PointerPointer pnaindex );
public static native PTA ptaSort( PTA ptas, @Cast("l_int32") int sorttype, @Cast("l_int32") int sortorder, @ByPtrPtr NUMA pnaindex );
public static native @Cast("l_int32") int ptaGetSortIndex( PTA ptas, @Cast("l_int32") int sorttype, @Cast("l_int32") int sortorder, @Cast("NUMA**") PointerPointer pnaindex );
public static native @Cast("l_int32") int ptaGetSortIndex( PTA ptas, @Cast("l_int32") int sorttype, @Cast("l_int32") int sortorder, @ByPtrPtr NUMA pnaindex );
public static native PTA ptaSortByIndex( PTA ptas, NUMA naindex );
public static native PTA ptaRemoveDuplicates( PTA ptas, @Cast("l_uint32") int factor );
public static native PTAA ptaaSortByIndex( PTAA ptaas, NUMA naindex );
public static native BOX ptaGetBoundingRegion( PTA pta );
public static native @Cast("l_int32") int ptaGetRange( PTA pta, @Cast("l_float32*") FloatPointer pminx, @Cast("l_float32*") FloatPointer pmaxx, @Cast("l_float32*") FloatPointer pminy, @Cast("l_float32*") FloatPointer pmaxy );
public static native @Cast("l_int32") int ptaGetRange( PTA pta, @Cast("l_float32*") FloatBuffer pminx, @Cast("l_float32*") FloatBuffer pmaxx, @Cast("l_float32*") FloatBuffer pminy, @Cast("l_float32*") FloatBuffer pmaxy );
public static native @Cast("l_int32") int ptaGetRange( PTA pta, @Cast("l_float32*") float[] pminx, @Cast("l_float32*") float[] pmaxx, @Cast("l_float32*") float[] pminy, @Cast("l_float32*") float[] pmaxy );
public static native PTA ptaGetInsideBox( PTA ptas, BOX box );
public static native PTA pixFindCornerPixels( PIX pixs );
public static native @Cast("l_int32") int ptaContainsPt( PTA pta, @Cast("l_int32") int x, @Cast("l_int32") int y );
public static native @Cast("l_int32") int ptaTestIntersection( PTA pta1, PTA pta2 );
public static native PTA ptaTransform( PTA ptas, @Cast("l_int32") int shiftx, @Cast("l_int32") int shifty, @Cast("l_float32") float scalex, @Cast("l_float32") float scaley );
public static native @Cast("l_int32") int ptaPtInsidePolygon( PTA pta, @Cast("l_float32") float x, @Cast("l_float32") float y, @Cast("l_int32*") IntPointer pinside );
public static native @Cast("l_int32") int ptaPtInsidePolygon( PTA pta, @Cast("l_float32") float x, @Cast("l_float32") float y, @Cast("l_int32*") IntBuffer pinside );
public static native @Cast("l_int32") int ptaPtInsidePolygon( PTA pta, @Cast("l_float32") float x, @Cast("l_float32") float y, @Cast("l_int32*") int[] pinside );
public static native @Cast("l_float32") float l_angleBetweenVectors( @Cast("l_float32") float x1, @Cast("l_float32") float y1, @Cast("l_float32") float x2, @Cast("l_float32") float y2 );
public static native @Cast("l_int32") int ptaGetLinearLSF( PTA pta, @Cast("l_float32*") FloatPointer pa, @Cast("l_float32*") FloatPointer pb, @Cast("NUMA**") PointerPointer pnafit );
public static native @Cast("l_int32") int ptaGetLinearLSF( PTA pta, @Cast("l_float32*") FloatPointer pa, @Cast("l_float32*") FloatPointer pb, @ByPtrPtr NUMA pnafit );
public static native @Cast("l_int32") int ptaGetLinearLSF( PTA pta, @Cast("l_float32*") FloatBuffer pa, @Cast("l_float32*") FloatBuffer pb, @ByPtrPtr NUMA pnafit );
public static native @Cast("l_int32") int ptaGetLinearLSF( PTA pta, @Cast("l_float32*") float[] pa, @Cast("l_float32*") float[] pb, @ByPtrPtr NUMA pnafit );
public static native @Cast("l_int32") int ptaGetQuadraticLSF( PTA pta, @Cast("l_float32*") FloatPointer pa, @Cast("l_float32*") FloatPointer pb, @Cast("l_float32*") FloatPointer pc, @Cast("NUMA**") PointerPointer pnafit );
public static native @Cast("l_int32") int ptaGetQuadraticLSF( PTA pta, @Cast("l_float32*") FloatPointer pa, @Cast("l_float32*") FloatPointer pb, @Cast("l_float32*") FloatPointer pc, @ByPtrPtr NUMA pnafit );
public static native @Cast("l_int32") int ptaGetQuadraticLSF( PTA pta, @Cast("l_float32*") FloatBuffer pa, @Cast("l_float32*") FloatBuffer pb, @Cast("l_float32*") FloatBuffer pc, @ByPtrPtr NUMA pnafit );
public static native @Cast("l_int32") int ptaGetQuadraticLSF( PTA pta, @Cast("l_float32*") float[] pa, @Cast("l_float32*") float[] pb, @Cast("l_float32*") float[] pc, @ByPtrPtr NUMA pnafit );
public static native @Cast("l_int32") int ptaGetCubicLSF( PTA pta, @Cast("l_float32*") FloatPointer pa, @Cast("l_float32*") FloatPointer pb, @Cast("l_float32*") FloatPointer pc, @Cast("l_float32*") FloatPointer pd, @Cast("NUMA**") PointerPointer pnafit );
public static native @Cast("l_int32") int ptaGetCubicLSF( PTA pta, @Cast("l_float32*") FloatPointer pa, @Cast("l_float32*") FloatPointer pb, @Cast("l_float32*") FloatPointer pc, @Cast("l_float32*") FloatPointer pd, @ByPtrPtr NUMA pnafit );
public static native @Cast("l_int32") int ptaGetCubicLSF( PTA pta, @Cast("l_float32*") FloatBuffer pa, @Cast("l_float32*") FloatBuffer pb, @Cast("l_float32*") FloatBuffer pc, @Cast("l_float32*") FloatBuffer pd, @ByPtrPtr NUMA pnafit );
public static native @Cast("l_int32") int ptaGetCubicLSF( PTA pta, @Cast("l_float32*") float[] pa, @Cast("l_float32*") float[] pb, @Cast("l_float32*") float[] pc, @Cast("l_float32*") float[] pd, @ByPtrPtr NUMA pnafit );
public static native @Cast("l_int32") int ptaGetQuarticLSF( PTA pta, @Cast("l_float32*") FloatPointer pa, @Cast("l_float32*") FloatPointer pb, @Cast("l_float32*") FloatPointer pc, @Cast("l_float32*") FloatPointer pd, @Cast("l_float32*") FloatPointer pe, @Cast("NUMA**") PointerPointer pnafit );
public static native @Cast("l_int32") int ptaGetQuarticLSF( PTA pta, @Cast("l_float32*") FloatPointer pa, @Cast("l_float32*") FloatPointer pb, @Cast("l_float32*") FloatPointer pc, @Cast("l_float32*") FloatPointer pd, @Cast("l_float32*") FloatPointer pe, @ByPtrPtr NUMA pnafit );
public static native @Cast("l_int32") int ptaGetQuarticLSF( PTA pta, @Cast("l_float32*") FloatBuffer pa, @Cast("l_float32*") FloatBuffer pb, @Cast("l_float32*") FloatBuffer pc, @Cast("l_float32*") FloatBuffer pd, @Cast("l_float32*") FloatBuffer pe, @ByPtrPtr NUMA pnafit );
public static native @Cast("l_int32") int ptaGetQuarticLSF( PTA pta, @Cast("l_float32*") float[] pa, @Cast("l_float32*") float[] pb, @Cast("l_float32*") float[] pc, @Cast("l_float32*") float[] pd, @Cast("l_float32*") float[] pe, @ByPtrPtr NUMA pnafit );
public static native @Cast("l_int32") int ptaNoisyLinearLSF( PTA pta, @Cast("l_float32") float factor, @Cast("PTA**") PointerPointer pptad, @Cast("l_float32*") FloatPointer pa, @Cast("l_float32*") FloatPointer pb, @Cast("l_float32*") FloatPointer pmederr, @Cast("NUMA**") PointerPointer pnafit );
public static native @Cast("l_int32") int ptaNoisyLinearLSF( PTA pta, @Cast("l_float32") float factor, @ByPtrPtr PTA pptad, @Cast("l_float32*") FloatPointer pa, @Cast("l_float32*") FloatPointer pb, @Cast("l_float32*") FloatPointer pmederr, @ByPtrPtr NUMA pnafit );
public static native @Cast("l_int32") int ptaNoisyLinearLSF( PTA pta, @Cast("l_float32") float factor, @ByPtrPtr PTA pptad, @Cast("l_float32*") FloatBuffer pa, @Cast("l_float32*") FloatBuffer pb, @Cast("l_float32*") FloatBuffer pmederr, @ByPtrPtr NUMA pnafit );
public static native @Cast("l_int32") int ptaNoisyLinearLSF( PTA pta, @Cast("l_float32") float factor, @ByPtrPtr PTA pptad, @Cast("l_float32*") float[] pa, @Cast("l_float32*") float[] pb, @Cast("l_float32*") float[] pmederr, @ByPtrPtr NUMA pnafit );
public static native @Cast("l_int32") int ptaNoisyQuadraticLSF( PTA pta, @Cast("l_float32") float factor, @Cast("PTA**") PointerPointer pptad, @Cast("l_float32*") FloatPointer pa, @Cast("l_float32*") FloatPointer pb, @Cast("l_float32*") FloatPointer pc, @Cast("l_float32*") FloatPointer pmederr, @Cast("NUMA**") PointerPointer pnafit );
public static native @Cast("l_int32") int ptaNoisyQuadraticLSF( PTA pta, @Cast("l_float32") float factor, @ByPtrPtr PTA pptad, @Cast("l_float32*") FloatPointer pa, @Cast("l_float32*") FloatPointer pb, @Cast("l_float32*") FloatPointer pc, @Cast("l_float32*") FloatPointer pmederr, @ByPtrPtr NUMA pnafit );
public static native @Cast("l_int32") int ptaNoisyQuadraticLSF( PTA pta, @Cast("l_float32") float factor, @ByPtrPtr PTA pptad, @Cast("l_float32*") FloatBuffer pa, @Cast("l_float32*") FloatBuffer pb, @Cast("l_float32*") FloatBuffer pc, @Cast("l_float32*") FloatBuffer pmederr, @ByPtrPtr NUMA pnafit );
public static native @Cast("l_int32") int ptaNoisyQuadraticLSF( PTA pta, @Cast("l_float32") float factor, @ByPtrPtr PTA pptad, @Cast("l_float32*") float[] pa, @Cast("l_float32*") float[] pb, @Cast("l_float32*") float[] pc, @Cast("l_float32*") float[] pmederr, @ByPtrPtr NUMA pnafit );
public static native @Cast("l_int32") int applyLinearFit( @Cast("l_float32") float a, @Cast("l_float32") float b, @Cast("l_float32") float x, @Cast("l_float32*") FloatPointer py );
public static native @Cast("l_int32") int applyLinearFit( @Cast("l_float32") float a, @Cast("l_float32") float b, @Cast("l_float32") float x, @Cast("l_float32*") FloatBuffer py );
public static native @Cast("l_int32") int applyLinearFit( @Cast("l_float32") float a, @Cast("l_float32") float b, @Cast("l_float32") float x, @Cast("l_float32*") float[] py );
public static native @Cast("l_int32") int applyQuadraticFit( @Cast("l_float32") float a, @Cast("l_float32") float b, @Cast("l_float32") float c, @Cast("l_float32") float x, @Cast("l_float32*") FloatPointer py );
public static native @Cast("l_int32") int applyQuadraticFit( @Cast("l_float32") float a, @Cast("l_float32") float b, @Cast("l_float32") float c, @Cast("l_float32") float x, @Cast("l_float32*") FloatBuffer py );
public static native @Cast("l_int32") int applyQuadraticFit( @Cast("l_float32") float a, @Cast("l_float32") float b, @Cast("l_float32") float c, @Cast("l_float32") float x, @Cast("l_float32*") float[] py );
public static native @Cast("l_int32") int applyCubicFit( @Cast("l_float32") float a, @Cast("l_float32") float b, @Cast("l_float32") float c, @Cast("l_float32") float d, @Cast("l_float32") float x, @Cast("l_float32*") FloatPointer py );
public static native @Cast("l_int32") int applyCubicFit( @Cast("l_float32") float a, @Cast("l_float32") float b, @Cast("l_float32") float c, @Cast("l_float32") float d, @Cast("l_float32") float x, @Cast("l_float32*") FloatBuffer py );
public static native @Cast("l_int32") int applyCubicFit( @Cast("l_float32") float a, @Cast("l_float32") float b, @Cast("l_float32") float c, @Cast("l_float32") float d, @Cast("l_float32") float x, @Cast("l_float32*") float[] py );
public static native @Cast("l_int32") int applyQuarticFit( @Cast("l_float32") float a, @Cast("l_float32") float b, @Cast("l_float32") float c, @Cast("l_float32") float d, @Cast("l_float32") float e, @Cast("l_float32") float x, @Cast("l_float32*") FloatPointer py );
public static native @Cast("l_int32") int applyQuarticFit( @Cast("l_float32") float a, @Cast("l_float32") float b, @Cast("l_float32") float c, @Cast("l_float32") float d, @Cast("l_float32") float e, @Cast("l_float32") float x, @Cast("l_float32*") FloatBuffer py );
public static native @Cast("l_int32") int applyQuarticFit( @Cast("l_float32") float a, @Cast("l_float32") float b, @Cast("l_float32") float c, @Cast("l_float32") float d, @Cast("l_float32") float e, @Cast("l_float32") float x, @Cast("l_float32*") float[] py );
public static native @Cast("l_int32") int pixPlotAlongPta( PIX pixs, PTA pta, @Cast("l_int32") int outformat, @Cast("const char*") BytePointer title );
public static native @Cast("l_int32") int pixPlotAlongPta( PIX pixs, PTA pta, @Cast("l_int32") int outformat, String title );
public static native PTA ptaGetPixelsFromPix( PIX pixs, BOX box );
public static native PIX pixGenerateFromPta( PTA pta, @Cast("l_int32") int w, @Cast("l_int32") int h );
public static native PTA ptaGetBoundaryPixels( PIX pixs, @Cast("l_int32") int type );
public static native PTAA ptaaGetBoundaryPixels( PIX pixs, @Cast("l_int32") int type, @Cast("l_int32") int connectivity, @Cast("BOXA**") PointerPointer pboxa, @Cast("PIXA**") PointerPointer ppixa );
public static native PTAA ptaaGetBoundaryPixels( PIX pixs, @Cast("l_int32") int type, @Cast("l_int32") int connectivity, @ByPtrPtr BOXA pboxa, @ByPtrPtr PIXA ppixa );
public static native PIX pixDisplayPta( PIX pixd, PIX pixs, PTA pta );
public static native PIX pixDisplayPtaaPattern( PIX pixd, PIX pixs, PTAA ptaa, PIX pixp, @Cast("l_int32") int cx, @Cast("l_int32") int cy );
public static native PIX pixDisplayPtaPattern( PIX pixd, PIX pixs, PTA pta, PIX pixp, @Cast("l_int32") int cx, @Cast("l_int32") int cy, @Cast("l_uint32") int color );
public static native PTA ptaReplicatePattern( PTA ptas, PIX pixp, PTA ptap, @Cast("l_int32") int cx, @Cast("l_int32") int cy, @Cast("l_int32") int w, @Cast("l_int32") int h );
public static native PIX pixDisplayPtaa( PIX pixs, PTAA ptaa );
public static native L_PTRA ptraCreate( @Cast("l_int32") int n );
public static native void ptraDestroy( @Cast("L_PTRA**") PointerPointer ppa, @Cast("l_int32") int freeflag, @Cast("l_int32") int warnflag );
public static native void ptraDestroy( @ByPtrPtr L_PTRA ppa, @Cast("l_int32") int freeflag, @Cast("l_int32") int warnflag );
public static native @Cast("l_int32") int ptraAdd( L_PTRA pa, Pointer item );
public static native @Cast("l_int32") int ptraInsert( L_PTRA pa, @Cast("l_int32") int index, Pointer item, @Cast("l_int32") int shiftflag );
public static native Pointer ptraRemove( L_PTRA pa, @Cast("l_int32") int index, @Cast("l_int32") int flag );
public static native Pointer ptraRemoveLast( L_PTRA pa );
public static native Pointer ptraReplace( L_PTRA pa, @Cast("l_int32") int index, Pointer item, @Cast("l_int32") int freeflag );
public static native @Cast("l_int32") int ptraSwap( L_PTRA pa, @Cast("l_int32") int index1, @Cast("l_int32") int index2 );
public static native @Cast("l_int32") int ptraCompactArray( L_PTRA pa );
public static native @Cast("l_int32") int ptraReverse( L_PTRA pa );
public static native @Cast("l_int32") int ptraJoin( L_PTRA pa1, L_PTRA pa2 );
public static native @Cast("l_int32") int ptraGetMaxIndex( L_PTRA pa, @Cast("l_int32*") IntPointer pmaxindex );
public static native @Cast("l_int32") int ptraGetMaxIndex( L_PTRA pa, @Cast("l_int32*") IntBuffer pmaxindex );
public static native @Cast("l_int32") int ptraGetMaxIndex( L_PTRA pa, @Cast("l_int32*") int[] pmaxindex );
public static native @Cast("l_int32") int ptraGetActualCount( L_PTRA pa, @Cast("l_int32*") IntPointer pcount );
public static native @Cast("l_int32") int ptraGetActualCount( L_PTRA pa, @Cast("l_int32*") IntBuffer pcount );
public static native @Cast("l_int32") int ptraGetActualCount( L_PTRA pa, @Cast("l_int32*") int[] pcount );
public static native Pointer ptraGetPtrToItem( L_PTRA pa, @Cast("l_int32") int index );
public static native L_PTRAA ptraaCreate( @Cast("l_int32") int n );
public static native void ptraaDestroy( @Cast("L_PTRAA**") PointerPointer ppaa, @Cast("l_int32") int freeflag, @Cast("l_int32") int warnflag );
public static native void ptraaDestroy( @ByPtrPtr L_PTRAA ppaa, @Cast("l_int32") int freeflag, @Cast("l_int32") int warnflag );
public static native @Cast("l_int32") int ptraaGetSize( L_PTRAA paa, @Cast("l_int32*") IntPointer psize );
public static native @Cast("l_int32") int ptraaGetSize( L_PTRAA paa, @Cast("l_int32*") IntBuffer psize );
public static native @Cast("l_int32") int ptraaGetSize( L_PTRAA paa, @Cast("l_int32*") int[] psize );
public static native @Cast("l_int32") int ptraaInsertPtra( L_PTRAA paa, @Cast("l_int32") int index, L_PTRA pa );
public static native L_PTRA ptraaGetPtra( L_PTRAA paa, @Cast("l_int32") int index, @Cast("l_int32") int accessflag );
public static native L_PTRA ptraaFlattenToPtra( L_PTRAA paa );
public static native @Cast("l_int32") int pixQuadtreeMean( PIX pixs, @Cast("l_int32") int nlevels, PIX pix_ma, @Cast("FPIXA**") PointerPointer pfpixa );
public static native @Cast("l_int32") int pixQuadtreeMean( PIX pixs, @Cast("l_int32") int nlevels, PIX pix_ma, @ByPtrPtr FPIXA pfpixa );
public static native @Cast("l_int32") int pixQuadtreeVariance( PIX pixs, @Cast("l_int32") int nlevels, PIX pix_ma, DPIX dpix_msa, @Cast("FPIXA**") PointerPointer pfpixa_v, @Cast("FPIXA**") PointerPointer pfpixa_rv );
public static native @Cast("l_int32") int pixQuadtreeVariance( PIX pixs, @Cast("l_int32") int nlevels, PIX pix_ma, DPIX dpix_msa, @ByPtrPtr FPIXA pfpixa_v, @ByPtrPtr FPIXA pfpixa_rv );
public static native @Cast("l_int32") int pixMeanInRectangle( PIX pixs, BOX box, PIX pixma, @Cast("l_float32*") FloatPointer pval );
public static native @Cast("l_int32") int pixMeanInRectangle( PIX pixs, BOX box, PIX pixma, @Cast("l_float32*") FloatBuffer pval );
public static native @Cast("l_int32") int pixMeanInRectangle( PIX pixs, BOX box, PIX pixma, @Cast("l_float32*") float[] pval );
public static native @Cast("l_int32") int pixVarianceInRectangle( PIX pixs, BOX box, PIX pix_ma, DPIX dpix_msa, @Cast("l_float32*") FloatPointer pvar, @Cast("l_float32*") FloatPointer prvar );
public static native @Cast("l_int32") int pixVarianceInRectangle( PIX pixs, BOX box, PIX pix_ma, DPIX dpix_msa, @Cast("l_float32*") FloatBuffer pvar, @Cast("l_float32*") FloatBuffer prvar );
public static native @Cast("l_int32") int pixVarianceInRectangle( PIX pixs, BOX box, PIX pix_ma, DPIX dpix_msa, @Cast("l_float32*") float[] pvar, @Cast("l_float32*") float[] prvar );
public static native BOXAA boxaaQuadtreeRegions( @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int nlevels );
public static native @Cast("l_int32") int quadtreeGetParent( FPIXA fpixa, @Cast("l_int32") int level, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_float32*") FloatPointer pval );
public static native @Cast("l_int32") int quadtreeGetParent( FPIXA fpixa, @Cast("l_int32") int level, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_float32*") FloatBuffer pval );
public static native @Cast("l_int32") int quadtreeGetParent( FPIXA fpixa, @Cast("l_int32") int level, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_float32*") float[] pval );
public static native @Cast("l_int32") int quadtreeGetChildren( FPIXA fpixa, @Cast("l_int32") int level, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_float32*") FloatPointer pval00, @Cast("l_float32*") FloatPointer pval10, @Cast("l_float32*") FloatPointer pval01, @Cast("l_float32*") FloatPointer pval11 );
public static native @Cast("l_int32") int quadtreeGetChildren( FPIXA fpixa, @Cast("l_int32") int level, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_float32*") FloatBuffer pval00, @Cast("l_float32*") FloatBuffer pval10, @Cast("l_float32*") FloatBuffer pval01, @Cast("l_float32*") FloatBuffer pval11 );
public static native @Cast("l_int32") int quadtreeGetChildren( FPIXA fpixa, @Cast("l_int32") int level, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_float32*") float[] pval00, @Cast("l_float32*") float[] pval10, @Cast("l_float32*") float[] pval01, @Cast("l_float32*") float[] pval11 );
public static native @Cast("l_int32") int quadtreeMaxLevels( @Cast("l_int32") int w, @Cast("l_int32") int h );
public static native PIX fpixaDisplayQuadtree( FPIXA fpixa, @Cast("l_int32") int factor, @Cast("const char*") BytePointer fontdir );
public static native PIX fpixaDisplayQuadtree( FPIXA fpixa, @Cast("l_int32") int factor, String fontdir );
public static native L_QUEUE lqueueCreate( @Cast("l_int32") int nalloc );
public static native void lqueueDestroy( @Cast("L_QUEUE**") PointerPointer plq, @Cast("l_int32") int freeflag );
public static native void lqueueDestroy( @ByPtrPtr L_QUEUE plq, @Cast("l_int32") int freeflag );
public static native @Cast("l_int32") int lqueueAdd( L_QUEUE lq, Pointer item );
public static native Pointer lqueueRemove( L_QUEUE lq );
public static native @Cast("l_int32") int lqueueGetCount( L_QUEUE lq );
public static native @Cast("l_int32") int lqueuePrint( @Cast("FILE*") Pointer fp, L_QUEUE lq );
public static native PIX pixRankFilter( PIX pixs, @Cast("l_int32") int wf, @Cast("l_int32") int hf, @Cast("l_float32") float rank );
public static native PIX pixRankFilterRGB( PIX pixs, @Cast("l_int32") int wf, @Cast("l_int32") int hf, @Cast("l_float32") float rank );
public static native PIX pixRankFilterGray( PIX pixs, @Cast("l_int32") int wf, @Cast("l_int32") int hf, @Cast("l_float32") float rank );
public static native PIX pixMedianFilter( PIX pixs, @Cast("l_int32") int wf, @Cast("l_int32") int hf );
public static native PIX pixRankFilterWithScaling( PIX pixs, @Cast("l_int32") int wf, @Cast("l_int32") int hf, @Cast("l_float32") float rank, @Cast("l_float32") float scalefactor );
public static native SARRAY pixProcessBarcodes( PIX pixs, @Cast("l_int32") int format, @Cast("l_int32") int method, @Cast("SARRAY**") PointerPointer psaw, @Cast("l_int32") int debugflag );
public static native SARRAY pixProcessBarcodes( PIX pixs, @Cast("l_int32") int format, @Cast("l_int32") int method, @ByPtrPtr SARRAY psaw, @Cast("l_int32") int debugflag );
public static native PIXA pixExtractBarcodes( PIX pixs, @Cast("l_int32") int debugflag );
public static native SARRAY pixReadBarcodes( PIXA pixa, @Cast("l_int32") int format, @Cast("l_int32") int method, @Cast("SARRAY**") PointerPointer psaw, @Cast("l_int32") int debugflag );
public static native SARRAY pixReadBarcodes( PIXA pixa, @Cast("l_int32") int format, @Cast("l_int32") int method, @ByPtrPtr SARRAY psaw, @Cast("l_int32") int debugflag );
public static native NUMA pixReadBarcodeWidths( PIX pixs, @Cast("l_int32") int method, @Cast("l_int32") int debugflag );
public static native BOXA pixLocateBarcodes( PIX pixs, @Cast("l_int32") int thresh, @Cast("PIX**") PointerPointer ppixb, @Cast("PIX**") PointerPointer ppixm );
public static native BOXA pixLocateBarcodes( PIX pixs, @Cast("l_int32") int thresh, @ByPtrPtr PIX ppixb, @ByPtrPtr PIX ppixm );
public static native PIX pixDeskewBarcode( PIX pixs, PIX pixb, BOX box, @Cast("l_int32") int margin, @Cast("l_int32") int threshold, @Cast("l_float32*") FloatPointer pangle, @Cast("l_float32*") FloatPointer pconf );
public static native PIX pixDeskewBarcode( PIX pixs, PIX pixb, BOX box, @Cast("l_int32") int margin, @Cast("l_int32") int threshold, @Cast("l_float32*") FloatBuffer pangle, @Cast("l_float32*") FloatBuffer pconf );
public static native PIX pixDeskewBarcode( PIX pixs, PIX pixb, BOX box, @Cast("l_int32") int margin, @Cast("l_int32") int threshold, @Cast("l_float32*") float[] pangle, @Cast("l_float32*") float[] pconf );
public static native NUMA pixExtractBarcodeWidths1( PIX pixs, @Cast("l_float32") float thresh, @Cast("l_float32") float binfract, @Cast("NUMA**") PointerPointer pnaehist, @Cast("NUMA**") PointerPointer pnaohist, @Cast("l_int32") int debugflag );
public static native NUMA pixExtractBarcodeWidths1( PIX pixs, @Cast("l_float32") float thresh, @Cast("l_float32") float binfract, @ByPtrPtr NUMA pnaehist, @ByPtrPtr NUMA pnaohist, @Cast("l_int32") int debugflag );
public static native NUMA pixExtractBarcodeWidths2( PIX pixs, @Cast("l_float32") float thresh, @Cast("l_float32*") FloatPointer pwidth, @Cast("NUMA**") PointerPointer pnac, @Cast("l_int32") int debugflag );
public static native NUMA pixExtractBarcodeWidths2( PIX pixs, @Cast("l_float32") float thresh, @Cast("l_float32*") FloatPointer pwidth, @ByPtrPtr NUMA pnac, @Cast("l_int32") int debugflag );
public static native NUMA pixExtractBarcodeWidths2( PIX pixs, @Cast("l_float32") float thresh, @Cast("l_float32*") FloatBuffer pwidth, @ByPtrPtr NUMA pnac, @Cast("l_int32") int debugflag );
public static native NUMA pixExtractBarcodeWidths2( PIX pixs, @Cast("l_float32") float thresh, @Cast("l_float32*") float[] pwidth, @ByPtrPtr NUMA pnac, @Cast("l_int32") int debugflag );
public static native NUMA pixExtractBarcodeCrossings( PIX pixs, @Cast("l_float32") float thresh, @Cast("l_int32") int debugflag );
public static native NUMA numaQuantizeCrossingsByWidth( NUMA nas, @Cast("l_float32") float binfract, @Cast("NUMA**") PointerPointer pnaehist, @Cast("NUMA**") PointerPointer pnaohist, @Cast("l_int32") int debugflag );
public static native NUMA numaQuantizeCrossingsByWidth( NUMA nas, @Cast("l_float32") float binfract, @ByPtrPtr NUMA pnaehist, @ByPtrPtr NUMA pnaohist, @Cast("l_int32") int debugflag );
public static native NUMA numaQuantizeCrossingsByWindow( NUMA nas, @Cast("l_float32") float ratio, @Cast("l_float32*") FloatPointer pwidth, @Cast("l_float32*") FloatPointer pfirstloc, @Cast("NUMA**") PointerPointer pnac, @Cast("l_int32") int debugflag );
public static native NUMA numaQuantizeCrossingsByWindow( NUMA nas, @Cast("l_float32") float ratio, @Cast("l_float32*") FloatPointer pwidth, @Cast("l_float32*") FloatPointer pfirstloc, @ByPtrPtr NUMA pnac, @Cast("l_int32") int debugflag );
public static native NUMA numaQuantizeCrossingsByWindow( NUMA nas, @Cast("l_float32") float ratio, @Cast("l_float32*") FloatBuffer pwidth, @Cast("l_float32*") FloatBuffer pfirstloc, @ByPtrPtr NUMA pnac, @Cast("l_int32") int debugflag );
public static native NUMA numaQuantizeCrossingsByWindow( NUMA nas, @Cast("l_float32") float ratio, @Cast("l_float32*") float[] pwidth, @Cast("l_float32*") float[] pfirstloc, @ByPtrPtr NUMA pnac, @Cast("l_int32") int debugflag );
public static native PIXA pixaReadFiles( @Cast("const char*") BytePointer dirname, @Cast("const char*") BytePointer substr );
public static native PIXA pixaReadFiles( String dirname, String substr );
public static native PIXA pixaReadFilesSA( SARRAY sa );
public static native PIX pixRead( @Cast("const char*") BytePointer filename );
public static native PIX pixRead( String filename );
public static native PIX pixReadWithHint( @Cast("const char*") BytePointer filename, @Cast("l_int32") int hint );
public static native PIX pixReadWithHint( String filename, @Cast("l_int32") int hint );
public static native PIX pixReadIndexed( SARRAY sa, @Cast("l_int32") int index );
public static native PIX pixReadStream( @Cast("FILE*") Pointer fp, @Cast("l_int32") int hint );
public static native @Cast("l_int32") int pixReadHeader( @Cast("const char*") BytePointer filename, @Cast("l_int32*") IntPointer pformat, @Cast("l_int32*") IntPointer pw, @Cast("l_int32*") IntPointer ph, @Cast("l_int32*") IntPointer pbps, @Cast("l_int32*") IntPointer pspp, @Cast("l_int32*") IntPointer piscmap );
public static native @Cast("l_int32") int pixReadHeader( String filename, @Cast("l_int32*") IntBuffer pformat, @Cast("l_int32*") IntBuffer pw, @Cast("l_int32*") IntBuffer ph, @Cast("l_int32*") IntBuffer pbps, @Cast("l_int32*") IntBuffer pspp, @Cast("l_int32*") IntBuffer piscmap );
public static native @Cast("l_int32") int pixReadHeader( @Cast("const char*") BytePointer filename, @Cast("l_int32*") int[] pformat, @Cast("l_int32*") int[] pw, @Cast("l_int32*") int[] ph, @Cast("l_int32*") int[] pbps, @Cast("l_int32*") int[] pspp, @Cast("l_int32*") int[] piscmap );
public static native @Cast("l_int32") int pixReadHeader( String filename, @Cast("l_int32*") IntPointer pformat, @Cast("l_int32*") IntPointer pw, @Cast("l_int32*") IntPointer ph, @Cast("l_int32*") IntPointer pbps, @Cast("l_int32*") IntPointer pspp, @Cast("l_int32*") IntPointer piscmap );
public static native @Cast("l_int32") int pixReadHeader( @Cast("const char*") BytePointer filename, @Cast("l_int32*") IntBuffer pformat, @Cast("l_int32*") IntBuffer pw, @Cast("l_int32*") IntBuffer ph, @Cast("l_int32*") IntBuffer pbps, @Cast("l_int32*") IntBuffer pspp, @Cast("l_int32*") IntBuffer piscmap );
public static native @Cast("l_int32") int pixReadHeader( String filename, @Cast("l_int32*") int[] pformat, @Cast("l_int32*") int[] pw, @Cast("l_int32*") int[] ph, @Cast("l_int32*") int[] pbps, @Cast("l_int32*") int[] pspp, @Cast("l_int32*") int[] piscmap );
public static native @Cast("l_int32") int findFileFormat( @Cast("const char*") BytePointer filename, @Cast("l_int32*") IntPointer pformat );
public static native @Cast("l_int32") int findFileFormat( String filename, @Cast("l_int32*") IntBuffer pformat );
public static native @Cast("l_int32") int findFileFormat( @Cast("const char*") BytePointer filename, @Cast("l_int32*") int[] pformat );
public static native @Cast("l_int32") int findFileFormat( String filename, @Cast("l_int32*") IntPointer pformat );
public static native @Cast("l_int32") int findFileFormat( @Cast("const char*") BytePointer filename, @Cast("l_int32*") IntBuffer pformat );
public static native @Cast("l_int32") int findFileFormat( String filename, @Cast("l_int32*") int[] pformat );
public static native @Cast("l_int32") int findFileFormatStream( @Cast("FILE*") Pointer fp, @Cast("l_int32*") IntPointer pformat );
public static native @Cast("l_int32") int findFileFormatStream( @Cast("FILE*") Pointer fp, @Cast("l_int32*") IntBuffer pformat );
public static native @Cast("l_int32") int findFileFormatStream( @Cast("FILE*") Pointer fp, @Cast("l_int32*") int[] pformat );
public static native @Cast("l_int32") int findFileFormatBuffer( @Cast("const l_uint8*") BytePointer buf, @Cast("l_int32*") IntPointer pformat );
public static native @Cast("l_int32") int findFileFormatBuffer( @Cast("const l_uint8*") ByteBuffer buf, @Cast("l_int32*") IntBuffer pformat );
public static native @Cast("l_int32") int findFileFormatBuffer( @Cast("const l_uint8*") byte[] buf, @Cast("l_int32*") int[] pformat );
public static native @Cast("l_int32") int fileFormatIsTiff( @Cast("FILE*") Pointer fp );
public static native PIX pixReadMem( @Cast("const l_uint8*") BytePointer data, @Cast("size_t") long size );
public static native PIX pixReadMem( @Cast("const l_uint8*") ByteBuffer data, @Cast("size_t") long size );
public static native PIX pixReadMem( @Cast("const l_uint8*") byte[] data, @Cast("size_t") long size );
public static native @Cast("l_int32") int pixReadHeaderMem( @Cast("const l_uint8*") BytePointer data, @Cast("size_t") long size, @Cast("l_int32*") IntPointer pformat, @Cast("l_int32*") IntPointer pw, @Cast("l_int32*") IntPointer ph, @Cast("l_int32*") IntPointer pbps, @Cast("l_int32*") IntPointer pspp, @Cast("l_int32*") IntPointer piscmap );
public static native @Cast("l_int32") int pixReadHeaderMem( @Cast("const l_uint8*") ByteBuffer data, @Cast("size_t") long size, @Cast("l_int32*") IntBuffer pformat, @Cast("l_int32*") IntBuffer pw, @Cast("l_int32*") IntBuffer ph, @Cast("l_int32*") IntBuffer pbps, @Cast("l_int32*") IntBuffer pspp, @Cast("l_int32*") IntBuffer piscmap );
public static native @Cast("l_int32") int pixReadHeaderMem( @Cast("const l_uint8*") byte[] data, @Cast("size_t") long size, @Cast("l_int32*") int[] pformat, @Cast("l_int32*") int[] pw, @Cast("l_int32*") int[] ph, @Cast("l_int32*") int[] pbps, @Cast("l_int32*") int[] pspp, @Cast("l_int32*") int[] piscmap );
public static native @Cast("l_int32") int ioFormatTest( @Cast("const char*") BytePointer filename );
public static native @Cast("l_int32") int ioFormatTest( String filename );
public static native L_RECOGA recogaCreateFromRecog( L_RECOG recog );
public static native L_RECOGA recogaCreateFromPixaa( PIXAA paa, @Cast("l_int32") int scalew, @Cast("l_int32") int scaleh, @Cast("l_int32") int templ_type, @Cast("l_int32") int threshold, @Cast("l_int32") int maxyshift );
public static native L_RECOGA recogaCreate( @Cast("l_int32") int n );
public static native void recogaDestroy( @Cast("L_RECOGA**") PointerPointer precoga );
public static native void recogaDestroy( @ByPtrPtr L_RECOGA precoga );
public static native @Cast("l_int32") int recogaAddRecog( L_RECOGA recoga, L_RECOG recog );
public static native @Cast("l_int32") int recogReplaceInRecoga( @Cast("L_RECOG**") PointerPointer precog1, L_RECOG recog2 );
public static native @Cast("l_int32") int recogReplaceInRecoga( @ByPtrPtr L_RECOG precog1, L_RECOG recog2 );
public static native L_RECOG recogaGetRecog( L_RECOGA recoga, @Cast("l_int32") int index );
public static native @Cast("l_int32") int recogaGetCount( L_RECOGA recoga );
public static native @Cast("l_int32") int recogGetCount( L_RECOG recog );
public static native @Cast("l_int32") int recogGetIndex( L_RECOG recog, @Cast("l_int32*") IntPointer pindex );
public static native @Cast("l_int32") int recogGetIndex( L_RECOG recog, @Cast("l_int32*") IntBuffer pindex );
public static native @Cast("l_int32") int recogGetIndex( L_RECOG recog, @Cast("l_int32*") int[] pindex );
public static native L_RECOGA recogGetParent( L_RECOG recog );
public static native @Cast("l_int32") int recogSetBootflag( L_RECOG recog );
public static native L_RECOG recogCreateFromRecog( L_RECOG recs, @Cast("l_int32") int scalew, @Cast("l_int32") int scaleh, @Cast("l_int32") int templ_type, @Cast("l_int32") int threshold, @Cast("l_int32") int maxyshift );
public static native L_RECOG recogCreateFromPixa( PIXA pixa, @Cast("l_int32") int scalew, @Cast("l_int32") int scaleh, @Cast("l_int32") int templ_type, @Cast("l_int32") int threshold, @Cast("l_int32") int maxyshift );
public static native L_RECOG recogCreate( @Cast("l_int32") int scalew, @Cast("l_int32") int scaleh, @Cast("l_int32") int templ_type, @Cast("l_int32") int threshold, @Cast("l_int32") int maxyshift );
public static native void recogDestroy( @Cast("L_RECOG**") PointerPointer precog );
public static native void recogDestroy( @ByPtrPtr L_RECOG precog );
public static native @Cast("l_int32") int recogAppend( L_RECOG recog1, L_RECOG recog2 );
public static native @Cast("l_int32") int recogGetClassIndex( L_RECOG recog, @Cast("l_int32") int val, @Cast("char*") BytePointer text, @Cast("l_int32*") IntPointer pindex );
public static native @Cast("l_int32") int recogGetClassIndex( L_RECOG recog, @Cast("l_int32") int val, @Cast("char*") ByteBuffer text, @Cast("l_int32*") IntBuffer pindex );
public static native @Cast("l_int32") int recogGetClassIndex( L_RECOG recog, @Cast("l_int32") int val, @Cast("char*") byte[] text, @Cast("l_int32*") int[] pindex );
public static native @Cast("l_int32") int recogStringToIndex( L_RECOG recog, @Cast("char*") BytePointer text, @Cast("l_int32*") IntPointer pindex );
public static native @Cast("l_int32") int recogStringToIndex( L_RECOG recog, @Cast("char*") ByteBuffer text, @Cast("l_int32*") IntBuffer pindex );
public static native @Cast("l_int32") int recogStringToIndex( L_RECOG recog, @Cast("char*") byte[] text, @Cast("l_int32*") int[] pindex );
public static native @Cast("l_int32") int recogGetClassString( L_RECOG recog, @Cast("l_int32") int index, @Cast("char**") PointerPointer pcharstr );
public static native @Cast("l_int32") int recogGetClassString( L_RECOG recog, @Cast("l_int32") int index, @Cast("char**") @ByPtrPtr BytePointer pcharstr );
public static native @Cast("l_int32") int recogGetClassString( L_RECOG recog, @Cast("l_int32") int index, @Cast("char**") @ByPtrPtr ByteBuffer pcharstr );
public static native @Cast("l_int32") int recogGetClassString( L_RECOG recog, @Cast("l_int32") int index, @Cast("char**") @ByPtrPtr byte[] pcharstr );
public static native @Cast("l_int32") int l_convertCharstrToInt( @Cast("const char*") BytePointer str, @Cast("l_int32*") IntPointer pval );
public static native @Cast("l_int32") int l_convertCharstrToInt( String str, @Cast("l_int32*") IntBuffer pval );
public static native @Cast("l_int32") int l_convertCharstrToInt( @Cast("const char*") BytePointer str, @Cast("l_int32*") int[] pval );
public static native @Cast("l_int32") int l_convertCharstrToInt( String str, @Cast("l_int32*") IntPointer pval );
public static native @Cast("l_int32") int l_convertCharstrToInt( @Cast("const char*") BytePointer str, @Cast("l_int32*") IntBuffer pval );
public static native @Cast("l_int32") int l_convertCharstrToInt( String str, @Cast("l_int32*") int[] pval );
public static native L_RECOGA recogaRead( @Cast("const char*") BytePointer filename );
public static native L_RECOGA recogaRead( String filename );
public static native L_RECOGA recogaReadStream( @Cast("FILE*") Pointer fp );
public static native @Cast("l_int32") int recogaWrite( @Cast("const char*") BytePointer filename, L_RECOGA recoga );
public static native @Cast("l_int32") int recogaWrite( String filename, L_RECOGA recoga );
public static native @Cast("l_int32") int recogaWriteStream( @Cast("FILE*") Pointer fp, L_RECOGA recoga, @Cast("const char*") BytePointer filename );
public static native @Cast("l_int32") int recogaWriteStream( @Cast("FILE*") Pointer fp, L_RECOGA recoga, String filename );
public static native @Cast("l_int32") int recogaWritePixaa( @Cast("const char*") BytePointer filename, L_RECOGA recoga );
public static native @Cast("l_int32") int recogaWritePixaa( String filename, L_RECOGA recoga );
public static native L_RECOG recogRead( @Cast("const char*") BytePointer filename );
public static native L_RECOG recogRead( String filename );
public static native L_RECOG recogReadStream( @Cast("FILE*") Pointer fp );
public static native @Cast("l_int32") int recogWrite( @Cast("const char*") BytePointer filename, L_RECOG recog );
public static native @Cast("l_int32") int recogWrite( String filename, L_RECOG recog );
public static native @Cast("l_int32") int recogWriteStream( @Cast("FILE*") Pointer fp, L_RECOG recog, @Cast("const char*") BytePointer filename );
public static native @Cast("l_int32") int recogWriteStream( @Cast("FILE*") Pointer fp, L_RECOG recog, String filename );
public static native @Cast("l_int32") int recogWritePixa( @Cast("const char*") BytePointer filename, L_RECOG recog );
public static native @Cast("l_int32") int recogWritePixa( String filename, L_RECOG recog );
public static native @Cast("l_int32") int recogDecode( L_RECOG recog, PIX pixs, @Cast("l_int32") int nlevels, @Cast("PIX**") PointerPointer ppixdb );
public static native @Cast("l_int32") int recogDecode( L_RECOG recog, PIX pixs, @Cast("l_int32") int nlevels, @ByPtrPtr PIX ppixdb );
public static native @Cast("l_int32") int recogMakeDecodingArrays( L_RECOG recog, PIX pixs, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int recogRunViterbi( L_RECOG recog, @Cast("PIX**") PointerPointer ppixdb );
public static native @Cast("l_int32") int recogRunViterbi( L_RECOG recog, @ByPtrPtr PIX ppixdb );
public static native @Cast("l_int32") int recogCreateDid( L_RECOG recog, PIX pixs );
public static native @Cast("l_int32") int recogDestroyDid( L_RECOG recog );
public static native @Cast("l_int32") int recogDidExists( L_RECOG recog );
public static native L_RDID recogGetDid( L_RECOG recog );
public static native @Cast("l_int32") int recogSetChannelParams( L_RECOG recog, @Cast("l_int32") int nlevels );
public static native @Cast("l_int32") int recogaIdentifyMultiple( L_RECOGA recoga, PIX pixs, @Cast("l_int32") int nitems, @Cast("l_int32") int minw, @Cast("l_int32") int minh, @Cast("BOXA**") PointerPointer pboxa, @Cast("PIXA**") PointerPointer ppixa, @Cast("PIX**") PointerPointer ppixdb, @Cast("l_int32") int debugsplit );
public static native @Cast("l_int32") int recogaIdentifyMultiple( L_RECOGA recoga, PIX pixs, @Cast("l_int32") int nitems, @Cast("l_int32") int minw, @Cast("l_int32") int minh, @ByPtrPtr BOXA pboxa, @ByPtrPtr PIXA ppixa, @ByPtrPtr PIX ppixdb, @Cast("l_int32") int debugsplit );
public static native @Cast("l_int32") int recogSplitIntoCharacters( L_RECOG recog, PIX pixs, @Cast("l_int32") int minw, @Cast("l_int32") int minh, @Cast("BOXA**") PointerPointer pboxa, @Cast("PIXA**") PointerPointer ppixa, @Cast("NUMA**") PointerPointer pnaid, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int recogSplitIntoCharacters( L_RECOG recog, PIX pixs, @Cast("l_int32") int minw, @Cast("l_int32") int minh, @ByPtrPtr BOXA pboxa, @ByPtrPtr PIXA ppixa, @ByPtrPtr NUMA pnaid, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int recogCorrelationBestRow( L_RECOG recog, PIX pixs, @Cast("BOXA**") PointerPointer pboxa, @Cast("NUMA**") PointerPointer pnascore, @Cast("NUMA**") PointerPointer pnaindex, @Cast("SARRAY**") PointerPointer psachar, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int recogCorrelationBestRow( L_RECOG recog, PIX pixs, @ByPtrPtr BOXA pboxa, @ByPtrPtr NUMA pnascore, @ByPtrPtr NUMA pnaindex, @ByPtrPtr SARRAY psachar, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int recogCorrelationBestChar( L_RECOG recog, PIX pixs, @Cast("BOX**") PointerPointer pbox, @Cast("l_float32*") FloatPointer pscore, @Cast("l_int32*") IntPointer pindex, @Cast("char**") PointerPointer pcharstr, @Cast("PIX**") PointerPointer ppixdb );
public static native @Cast("l_int32") int recogCorrelationBestChar( L_RECOG recog, PIX pixs, @ByPtrPtr BOX pbox, @Cast("l_float32*") FloatPointer pscore, @Cast("l_int32*") IntPointer pindex, @Cast("char**") @ByPtrPtr BytePointer pcharstr, @ByPtrPtr PIX ppixdb );
public static native @Cast("l_int32") int recogCorrelationBestChar( L_RECOG recog, PIX pixs, @ByPtrPtr BOX pbox, @Cast("l_float32*") FloatBuffer pscore, @Cast("l_int32*") IntBuffer pindex, @Cast("char**") @ByPtrPtr ByteBuffer pcharstr, @ByPtrPtr PIX ppixdb );
public static native @Cast("l_int32") int recogCorrelationBestChar( L_RECOG recog, PIX pixs, @ByPtrPtr BOX pbox, @Cast("l_float32*") float[] pscore, @Cast("l_int32*") int[] pindex, @Cast("char**") @ByPtrPtr byte[] pcharstr, @ByPtrPtr PIX ppixdb );
public static native @Cast("l_int32") int recogaIdentifyPixa( L_RECOGA recoga, PIXA pixa, NUMA naid, @Cast("PIX**") PointerPointer ppixdb );
public static native @Cast("l_int32") int recogaIdentifyPixa( L_RECOGA recoga, PIXA pixa, NUMA naid, @ByPtrPtr PIX ppixdb );
public static native @Cast("l_int32") int recogIdentifyPixa( L_RECOG recog, PIXA pixa, NUMA naid, @Cast("PIX**") PointerPointer ppixdb );
public static native @Cast("l_int32") int recogIdentifyPixa( L_RECOG recog, PIXA pixa, NUMA naid, @ByPtrPtr PIX ppixdb );
public static native @Cast("l_int32") int recogIdentifyPix( L_RECOG recog, PIX pixs, @Cast("PIX**") PointerPointer ppixdb );
public static native @Cast("l_int32") int recogIdentifyPix( L_RECOG recog, PIX pixs, @ByPtrPtr PIX ppixdb );
public static native @Cast("l_int32") int recogSkipIdentify( L_RECOG recog );
public static native void rchaDestroy( @Cast("L_RCHA**") PointerPointer prcha );
public static native void rchaDestroy( @ByPtrPtr L_RCHA prcha );
public static native void rchDestroy( @Cast("L_RCH**") PointerPointer prch );
public static native void rchDestroy( @ByPtrPtr L_RCH prch );
public static native @Cast("l_int32") int rchaExtract( L_RCHA rcha, @Cast("NUMA**") PointerPointer pnaindex, @Cast("NUMA**") PointerPointer pnascore, @Cast("SARRAY**") PointerPointer psatext, @Cast("NUMA**") PointerPointer pnasample, @Cast("NUMA**") PointerPointer pnaxloc, @Cast("NUMA**") PointerPointer pnayloc, @Cast("NUMA**") PointerPointer pnawidth );
public static native @Cast("l_int32") int rchaExtract( L_RCHA rcha, @ByPtrPtr NUMA pnaindex, @ByPtrPtr NUMA pnascore, @ByPtrPtr SARRAY psatext, @ByPtrPtr NUMA pnasample, @ByPtrPtr NUMA pnaxloc, @ByPtrPtr NUMA pnayloc, @ByPtrPtr NUMA pnawidth );
public static native @Cast("l_int32") int rchExtract( L_RCH rch, @Cast("l_int32*") IntPointer pindex, @Cast("l_float32*") FloatPointer pscore, @Cast("char**") PointerPointer ptext, @Cast("l_int32*") IntPointer psample, @Cast("l_int32*") IntPointer pxloc, @Cast("l_int32*") IntPointer pyloc, @Cast("l_int32*") IntPointer pwidth );
public static native @Cast("l_int32") int rchExtract( L_RCH rch, @Cast("l_int32*") IntPointer pindex, @Cast("l_float32*") FloatPointer pscore, @Cast("char**") @ByPtrPtr BytePointer ptext, @Cast("l_int32*") IntPointer psample, @Cast("l_int32*") IntPointer pxloc, @Cast("l_int32*") IntPointer pyloc, @Cast("l_int32*") IntPointer pwidth );
public static native @Cast("l_int32") int rchExtract( L_RCH rch, @Cast("l_int32*") IntBuffer pindex, @Cast("l_float32*") FloatBuffer pscore, @Cast("char**") @ByPtrPtr ByteBuffer ptext, @Cast("l_int32*") IntBuffer psample, @Cast("l_int32*") IntBuffer pxloc, @Cast("l_int32*") IntBuffer pyloc, @Cast("l_int32*") IntBuffer pwidth );
public static native @Cast("l_int32") int rchExtract( L_RCH rch, @Cast("l_int32*") int[] pindex, @Cast("l_float32*") float[] pscore, @Cast("char**") @ByPtrPtr byte[] ptext, @Cast("l_int32*") int[] psample, @Cast("l_int32*") int[] pxloc, @Cast("l_int32*") int[] pyloc, @Cast("l_int32*") int[] pwidth );
public static native PIX recogProcessToIdentify( L_RECOG recog, PIX pixs, @Cast("l_int32") int pad );
public static native PIX recogPreSplittingFilter( L_RECOG recog, PIX pixs, @Cast("l_float32") float maxasp, @Cast("l_float32") float minaf, @Cast("l_float32") float maxaf, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int recogSplittingFilter( L_RECOG recog, PIX pixs, @Cast("l_float32") float maxasp, @Cast("l_float32") float minaf, @Cast("l_float32") float maxaf, @Cast("l_int32*") IntPointer premove, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int recogSplittingFilter( L_RECOG recog, PIX pixs, @Cast("l_float32") float maxasp, @Cast("l_float32") float minaf, @Cast("l_float32") float maxaf, @Cast("l_int32*") IntBuffer premove, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int recogSplittingFilter( L_RECOG recog, PIX pixs, @Cast("l_float32") float maxasp, @Cast("l_float32") float minaf, @Cast("l_float32") float maxaf, @Cast("l_int32*") int[] premove, @Cast("l_int32") int debug );
public static native SARRAY recogaExtractNumbers( L_RECOGA recoga, BOXA boxas, @Cast("l_float32") float scorethresh, @Cast("l_int32") int spacethresh, @Cast("BOXAA**") PointerPointer pbaa, @Cast("NUMAA**") PointerPointer pnaa );
public static native SARRAY recogaExtractNumbers( L_RECOGA recoga, BOXA boxas, @Cast("l_float32") float scorethresh, @Cast("l_int32") int spacethresh, @ByPtrPtr BOXAA pbaa, @ByPtrPtr NUMAA pnaa );
public static native @Cast("l_int32") int recogSetTemplateType( L_RECOG recog, @Cast("l_int32") int templ_type );
public static native @Cast("l_int32") int recogSetScaling( L_RECOG recog, @Cast("l_int32") int scalew, @Cast("l_int32") int scaleh );
public static native @Cast("l_int32") int recogTrainLabelled( L_RECOG recog, PIX pixs, BOX box, @Cast("char*") BytePointer text, @Cast("l_int32") int multflag, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int recogTrainLabelled( L_RECOG recog, PIX pixs, BOX box, @Cast("char*") ByteBuffer text, @Cast("l_int32") int multflag, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int recogTrainLabelled( L_RECOG recog, PIX pixs, BOX box, @Cast("char*") byte[] text, @Cast("l_int32") int multflag, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int recogProcessMultLabelled( L_RECOG recog, PIX pixs, BOX box, @Cast("char*") BytePointer text, @Cast("PIXA**") PointerPointer ppixa, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int recogProcessMultLabelled( L_RECOG recog, PIX pixs, BOX box, @Cast("char*") BytePointer text, @ByPtrPtr PIXA ppixa, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int recogProcessMultLabelled( L_RECOG recog, PIX pixs, BOX box, @Cast("char*") ByteBuffer text, @ByPtrPtr PIXA ppixa, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int recogProcessMultLabelled( L_RECOG recog, PIX pixs, BOX box, @Cast("char*") byte[] text, @ByPtrPtr PIXA ppixa, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int recogProcessSingleLabelled( L_RECOG recog, PIX pixs, BOX box, @Cast("char*") BytePointer text, @Cast("PIXA**") PointerPointer ppixa );
public static native @Cast("l_int32") int recogProcessSingleLabelled( L_RECOG recog, PIX pixs, BOX box, @Cast("char*") BytePointer text, @ByPtrPtr PIXA ppixa );
public static native @Cast("l_int32") int recogProcessSingleLabelled( L_RECOG recog, PIX pixs, BOX box, @Cast("char*") ByteBuffer text, @ByPtrPtr PIXA ppixa );
public static native @Cast("l_int32") int recogProcessSingleLabelled( L_RECOG recog, PIX pixs, BOX box, @Cast("char*") byte[] text, @ByPtrPtr PIXA ppixa );
public static native @Cast("l_int32") int recogAddSamples( L_RECOG recog, PIXA pixa, @Cast("l_int32") int classindex, @Cast("l_int32") int debug );
public static native PIX recogScaleCharacter( L_RECOG recog, PIX pixs );
public static native @Cast("l_int32") int recogAverageSamples( L_RECOG recog, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int pixaAccumulateSamples( PIXA pixa, PTA pta, @Cast("PIX**") PointerPointer ppixd, @Cast("l_float32*") FloatPointer px, @Cast("l_float32*") FloatPointer py );
public static native @Cast("l_int32") int pixaAccumulateSamples( PIXA pixa, PTA pta, @ByPtrPtr PIX ppixd, @Cast("l_float32*") FloatPointer px, @Cast("l_float32*") FloatPointer py );
public static native @Cast("l_int32") int pixaAccumulateSamples( PIXA pixa, PTA pta, @ByPtrPtr PIX ppixd, @Cast("l_float32*") FloatBuffer px, @Cast("l_float32*") FloatBuffer py );
public static native @Cast("l_int32") int pixaAccumulateSamples( PIXA pixa, PTA pta, @ByPtrPtr PIX ppixd, @Cast("l_float32*") float[] px, @Cast("l_float32*") float[] py );
public static native @Cast("l_int32") int recogTrainingFinished( L_RECOG recog, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int recogRemoveOutliers( L_RECOG recog, @Cast("l_float32") float targetscore, @Cast("l_float32") float minfract, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int recogaTrainingDone( L_RECOGA recoga, @Cast("l_int32*") IntPointer pdone );
public static native @Cast("l_int32") int recogaTrainingDone( L_RECOGA recoga, @Cast("l_int32*") IntBuffer pdone );
public static native @Cast("l_int32") int recogaTrainingDone( L_RECOGA recoga, @Cast("l_int32*") int[] pdone );
public static native @Cast("l_int32") int recogaFinishAveraging( L_RECOGA recoga );
public static native @Cast("l_int32") int recogTrainUnlabelled( L_RECOG recog, L_RECOG recogboot, PIX pixs, BOX box, @Cast("l_int32") int singlechar, @Cast("l_float32") float minscore, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int recogPadTrainingSet( @Cast("L_RECOG**") PointerPointer precog, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int recogPadTrainingSet( @ByPtrPtr L_RECOG precog, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int recogBestCorrelForPadding( L_RECOG recog, L_RECOGA recoga, @Cast("NUMA**") PointerPointer pnaset, @Cast("NUMA**") PointerPointer pnaindex, @Cast("NUMA**") PointerPointer pnascore, @Cast("NUMA**") PointerPointer pnasum, @Cast("PIXA**") PointerPointer ppixadb );
public static native @Cast("l_int32") int recogBestCorrelForPadding( L_RECOG recog, L_RECOGA recoga, @ByPtrPtr NUMA pnaset, @ByPtrPtr NUMA pnaindex, @ByPtrPtr NUMA pnascore, @ByPtrPtr NUMA pnasum, @ByPtrPtr PIXA ppixadb );
public static native @Cast("l_int32") int recogCorrelAverages( L_RECOG recog1, L_RECOG recog2, @Cast("NUMA**") PointerPointer pnaindex, @Cast("NUMA**") PointerPointer pnascore, @Cast("PIXA**") PointerPointer ppixadb );
public static native @Cast("l_int32") int recogCorrelAverages( L_RECOG recog1, L_RECOG recog2, @ByPtrPtr NUMA pnaindex, @ByPtrPtr NUMA pnascore, @ByPtrPtr PIXA ppixadb );
public static native @Cast("l_int32") int recogSetPadParams( L_RECOG recog, @Cast("const char*") BytePointer bootdir, @Cast("const char*") BytePointer bootpattern, @Cast("const char*") BytePointer bootpath, @Cast("l_int32") int type, @Cast("l_int32") int min_nopad, @Cast("l_int32") int max_afterpad );
public static native @Cast("l_int32") int recogSetPadParams( L_RECOG recog, String bootdir, String bootpattern, String bootpath, @Cast("l_int32") int type, @Cast("l_int32") int min_nopad, @Cast("l_int32") int max_afterpad );
public static native @Cast("l_int32") int recogaShowContent( @Cast("FILE*") Pointer fp, L_RECOGA recoga, @Cast("l_int32") int display );
public static native @Cast("l_int32") int recogShowContent( @Cast("FILE*") Pointer fp, L_RECOG recog, @Cast("l_int32") int display );
public static native @Cast("l_int32") int recogDebugAverages( L_RECOG recog, @Cast("l_int32") int debug );
public static native @Cast("l_int32") int recogShowAverageTemplates( L_RECOG recog );
public static native @Cast("l_int32") int recogShowMatchesInRange( L_RECOG recog, PIXA pixa, @Cast("l_float32") float minscore, @Cast("l_float32") float maxscore, @Cast("l_int32") int display );
public static native PIX recogShowMatch( L_RECOG recog, PIX pix1, PIX pix2, BOX box, @Cast("l_int32") int index, @Cast("l_float32") float score );
public static native @Cast("l_int32") int recogResetBmf( L_RECOG recog, @Cast("l_int32") int size );
public static native @Cast("l_int32") int regTestSetup( @Cast("l_int32") int argc, @Cast("char**") PointerPointer argv, @Cast("L_REGPARAMS**") PointerPointer prp );
public static native @Cast("l_int32") int regTestSetup( @Cast("l_int32") int argc, @Cast("char**") @ByPtrPtr BytePointer argv, @ByPtrPtr L_REGPARAMS prp );
public static native @Cast("l_int32") int regTestSetup( @Cast("l_int32") int argc, @Cast("char**") @ByPtrPtr ByteBuffer argv, @ByPtrPtr L_REGPARAMS prp );
public static native @Cast("l_int32") int regTestSetup( @Cast("l_int32") int argc, @Cast("char**") @ByPtrPtr byte[] argv, @ByPtrPtr L_REGPARAMS prp );
public static native @Cast("l_int32") int regTestCleanup( L_REGPARAMS rp );
public static native @Cast("l_int32") int regTestCompareValues( L_REGPARAMS rp, @Cast("l_float32") float val1, @Cast("l_float32") float val2, @Cast("l_float32") float delta );
public static native @Cast("l_int32") int regTestCompareStrings( L_REGPARAMS rp, @Cast("l_uint8*") BytePointer string1, @Cast("size_t") long bytes1, @Cast("l_uint8*") BytePointer string2, @Cast("size_t") long bytes2 );
public static native @Cast("l_int32") int regTestCompareStrings( L_REGPARAMS rp, @Cast("l_uint8*") ByteBuffer string1, @Cast("size_t") long bytes1, @Cast("l_uint8*") ByteBuffer string2, @Cast("size_t") long bytes2 );
public static native @Cast("l_int32") int regTestCompareStrings( L_REGPARAMS rp, @Cast("l_uint8*") byte[] string1, @Cast("size_t") long bytes1, @Cast("l_uint8*") byte[] string2, @Cast("size_t") long bytes2 );
public static native @Cast("l_int32") int regTestComparePix( L_REGPARAMS rp, PIX pix1, PIX pix2 );
public static native @Cast("l_int32") int regTestCompareSimilarPix( L_REGPARAMS rp, PIX pix1, PIX pix2, @Cast("l_int32") int mindiff, @Cast("l_float32") float maxfract, @Cast("l_int32") int printstats );
public static native @Cast("l_int32") int regTestCheckFile( L_REGPARAMS rp, @Cast("const char*") BytePointer localname );
public static native @Cast("l_int32") int regTestCheckFile( L_REGPARAMS rp, String localname );
public static native @Cast("l_int32") int regTestCompareFiles( L_REGPARAMS rp, @Cast("l_int32") int index1, @Cast("l_int32") int index2 );
public static native @Cast("l_int32") int regTestWritePixAndCheck( L_REGPARAMS rp, PIX pix, @Cast("l_int32") int format );
public static native @Cast("l_int32") int pixRasterop( PIX pixd, @Cast("l_int32") int dx, @Cast("l_int32") int dy, @Cast("l_int32") int dw, @Cast("l_int32") int dh, @Cast("l_int32") int op, PIX pixs, @Cast("l_int32") int sx, @Cast("l_int32") int sy );
public static native @Cast("l_int32") int pixRasteropVip( PIX pixd, @Cast("l_int32") int bx, @Cast("l_int32") int bw, @Cast("l_int32") int vshift, @Cast("l_int32") int incolor );
public static native @Cast("l_int32") int pixRasteropHip( PIX pixd, @Cast("l_int32") int by, @Cast("l_int32") int bh, @Cast("l_int32") int hshift, @Cast("l_int32") int incolor );
public static native PIX pixTranslate( PIX pixd, PIX pixs, @Cast("l_int32") int hshift, @Cast("l_int32") int vshift, @Cast("l_int32") int incolor );
public static native @Cast("l_int32") int pixRasteropIP( PIX pixd, @Cast("l_int32") int hshift, @Cast("l_int32") int vshift, @Cast("l_int32") int incolor );
public static native @Cast("l_int32") int pixRasteropFullImage( PIX pixd, PIX pixs, @Cast("l_int32") int op );
public static native void rasteropVipLow( @Cast("l_uint32*") IntPointer data, @Cast("l_int32") int pixw, @Cast("l_int32") int pixh, @Cast("l_int32") int depth, @Cast("l_int32") int wpl, @Cast("l_int32") int x, @Cast("l_int32") int w, @Cast("l_int32") int shift );
public static native void rasteropVipLow( @Cast("l_uint32*") IntBuffer data, @Cast("l_int32") int pixw, @Cast("l_int32") int pixh, @Cast("l_int32") int depth, @Cast("l_int32") int wpl, @Cast("l_int32") int x, @Cast("l_int32") int w, @Cast("l_int32") int shift );
public static native void rasteropVipLow( @Cast("l_uint32*") int[] data, @Cast("l_int32") int pixw, @Cast("l_int32") int pixh, @Cast("l_int32") int depth, @Cast("l_int32") int wpl, @Cast("l_int32") int x, @Cast("l_int32") int w, @Cast("l_int32") int shift );
public static native void rasteropHipLow( @Cast("l_uint32*") IntPointer data, @Cast("l_int32") int pixh, @Cast("l_int32") int depth, @Cast("l_int32") int wpl, @Cast("l_int32") int y, @Cast("l_int32") int h, @Cast("l_int32") int shift );
public static native void rasteropHipLow( @Cast("l_uint32*") IntBuffer data, @Cast("l_int32") int pixh, @Cast("l_int32") int depth, @Cast("l_int32") int wpl, @Cast("l_int32") int y, @Cast("l_int32") int h, @Cast("l_int32") int shift );
public static native void rasteropHipLow( @Cast("l_uint32*") int[] data, @Cast("l_int32") int pixh, @Cast("l_int32") int depth, @Cast("l_int32") int wpl, @Cast("l_int32") int y, @Cast("l_int32") int h, @Cast("l_int32") int shift );
public static native void shiftDataHorizontalLow( @Cast("l_uint32*") IntPointer datad, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntPointer datas, @Cast("l_int32") int wpls, @Cast("l_int32") int shift );
public static native void shiftDataHorizontalLow( @Cast("l_uint32*") IntBuffer datad, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntBuffer datas, @Cast("l_int32") int wpls, @Cast("l_int32") int shift );
public static native void shiftDataHorizontalLow( @Cast("l_uint32*") int[] datad, @Cast("l_int32") int wpld, @Cast("l_uint32*") int[] datas, @Cast("l_int32") int wpls, @Cast("l_int32") int shift );
public static native void rasteropUniLow( @Cast("l_uint32*") IntPointer datad, @Cast("l_int32") int dpixw, @Cast("l_int32") int dpixh, @Cast("l_int32") int depth, @Cast("l_int32") int dwpl, @Cast("l_int32") int dx, @Cast("l_int32") int dy, @Cast("l_int32") int dw, @Cast("l_int32") int dh, @Cast("l_int32") int op );
public static native void rasteropUniLow( @Cast("l_uint32*") IntBuffer datad, @Cast("l_int32") int dpixw, @Cast("l_int32") int dpixh, @Cast("l_int32") int depth, @Cast("l_int32") int dwpl, @Cast("l_int32") int dx, @Cast("l_int32") int dy, @Cast("l_int32") int dw, @Cast("l_int32") int dh, @Cast("l_int32") int op );
public static native void rasteropUniLow( @Cast("l_uint32*") int[] datad, @Cast("l_int32") int dpixw, @Cast("l_int32") int dpixh, @Cast("l_int32") int depth, @Cast("l_int32") int dwpl, @Cast("l_int32") int dx, @Cast("l_int32") int dy, @Cast("l_int32") int dw, @Cast("l_int32") int dh, @Cast("l_int32") int op );
public static native void rasteropLow( @Cast("l_uint32*") IntPointer datad, @Cast("l_int32") int dpixw, @Cast("l_int32") int dpixh, @Cast("l_int32") int depth, @Cast("l_int32") int dwpl, @Cast("l_int32") int dx, @Cast("l_int32") int dy, @Cast("l_int32") int dw, @Cast("l_int32") int dh, @Cast("l_int32") int op, @Cast("l_uint32*") IntPointer datas, @Cast("l_int32") int spixw, @Cast("l_int32") int spixh, @Cast("l_int32") int swpl, @Cast("l_int32") int sx, @Cast("l_int32") int sy );
public static native void rasteropLow( @Cast("l_uint32*") IntBuffer datad, @Cast("l_int32") int dpixw, @Cast("l_int32") int dpixh, @Cast("l_int32") int depth, @Cast("l_int32") int dwpl, @Cast("l_int32") int dx, @Cast("l_int32") int dy, @Cast("l_int32") int dw, @Cast("l_int32") int dh, @Cast("l_int32") int op, @Cast("l_uint32*") IntBuffer datas, @Cast("l_int32") int spixw, @Cast("l_int32") int spixh, @Cast("l_int32") int swpl, @Cast("l_int32") int sx, @Cast("l_int32") int sy );
public static native void rasteropLow( @Cast("l_uint32*") int[] datad, @Cast("l_int32") int dpixw, @Cast("l_int32") int dpixh, @Cast("l_int32") int depth, @Cast("l_int32") int dwpl, @Cast("l_int32") int dx, @Cast("l_int32") int dy, @Cast("l_int32") int dw, @Cast("l_int32") int dh, @Cast("l_int32") int op, @Cast("l_uint32*") int[] datas, @Cast("l_int32") int spixw, @Cast("l_int32") int spixh, @Cast("l_int32") int swpl, @Cast("l_int32") int sx, @Cast("l_int32") int sy );
public static native PIX pixRotate( PIX pixs, @Cast("l_float32") float angle, @Cast("l_int32") int type, @Cast("l_int32") int incolor, @Cast("l_int32") int width, @Cast("l_int32") int height );
public static native PIX pixEmbedForRotation( PIX pixs, @Cast("l_float32") float angle, @Cast("l_int32") int incolor, @Cast("l_int32") int width, @Cast("l_int32") int height );
public static native PIX pixRotateBySampling( PIX pixs, @Cast("l_int32") int xcen, @Cast("l_int32") int ycen, @Cast("l_float32") float angle, @Cast("l_int32") int incolor );
public static native PIX pixRotateBinaryNice( PIX pixs, @Cast("l_float32") float angle, @Cast("l_int32") int incolor );
public static native PIX pixRotateWithAlpha( PIX pixs, @Cast("l_float32") float angle, PIX pixg, @Cast("l_float32") float fract );
public static native PIX pixRotateAM( PIX pixs, @Cast("l_float32") float angle, @Cast("l_int32") int incolor );
public static native PIX pixRotateAMColor( PIX pixs, @Cast("l_float32") float angle, @Cast("l_uint32") int colorval );
public static native PIX pixRotateAMGray( PIX pixs, @Cast("l_float32") float angle, @Cast("l_uint8") byte grayval );
public static native PIX pixRotateAMCorner( PIX pixs, @Cast("l_float32") float angle, @Cast("l_int32") int incolor );
public static native PIX pixRotateAMColorCorner( PIX pixs, @Cast("l_float32") float angle, @Cast("l_uint32") int fillval );
public static native PIX pixRotateAMGrayCorner( PIX pixs, @Cast("l_float32") float angle, @Cast("l_uint8") byte grayval );
public static native PIX pixRotateAMColorFast( PIX pixs, @Cast("l_float32") float angle, @Cast("l_uint32") int colorval );
public static native void rotateAMColorLow( @Cast("l_uint32*") IntPointer datad, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntPointer datas, @Cast("l_int32") int wpls, @Cast("l_float32") float angle, @Cast("l_uint32") int colorval );
public static native void rotateAMColorLow( @Cast("l_uint32*") IntBuffer datad, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntBuffer datas, @Cast("l_int32") int wpls, @Cast("l_float32") float angle, @Cast("l_uint32") int colorval );
public static native void rotateAMColorLow( @Cast("l_uint32*") int[] datad, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpld, @Cast("l_uint32*") int[] datas, @Cast("l_int32") int wpls, @Cast("l_float32") float angle, @Cast("l_uint32") int colorval );
public static native void rotateAMGrayLow( @Cast("l_uint32*") IntPointer datad, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntPointer datas, @Cast("l_int32") int wpls, @Cast("l_float32") float angle, @Cast("l_uint8") byte grayval );
public static native void rotateAMGrayLow( @Cast("l_uint32*") IntBuffer datad, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntBuffer datas, @Cast("l_int32") int wpls, @Cast("l_float32") float angle, @Cast("l_uint8") byte grayval );
public static native void rotateAMGrayLow( @Cast("l_uint32*") int[] datad, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpld, @Cast("l_uint32*") int[] datas, @Cast("l_int32") int wpls, @Cast("l_float32") float angle, @Cast("l_uint8") byte grayval );
public static native void rotateAMColorCornerLow( @Cast("l_uint32*") IntPointer datad, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntPointer datas, @Cast("l_int32") int wpls, @Cast("l_float32") float angle, @Cast("l_uint32") int colorval );
public static native void rotateAMColorCornerLow( @Cast("l_uint32*") IntBuffer datad, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntBuffer datas, @Cast("l_int32") int wpls, @Cast("l_float32") float angle, @Cast("l_uint32") int colorval );
public static native void rotateAMColorCornerLow( @Cast("l_uint32*") int[] datad, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpld, @Cast("l_uint32*") int[] datas, @Cast("l_int32") int wpls, @Cast("l_float32") float angle, @Cast("l_uint32") int colorval );
public static native void rotateAMGrayCornerLow( @Cast("l_uint32*") IntPointer datad, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntPointer datas, @Cast("l_int32") int wpls, @Cast("l_float32") float angle, @Cast("l_uint8") byte grayval );
public static native void rotateAMGrayCornerLow( @Cast("l_uint32*") IntBuffer datad, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntBuffer datas, @Cast("l_int32") int wpls, @Cast("l_float32") float angle, @Cast("l_uint8") byte grayval );
public static native void rotateAMGrayCornerLow( @Cast("l_uint32*") int[] datad, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpld, @Cast("l_uint32*") int[] datas, @Cast("l_int32") int wpls, @Cast("l_float32") float angle, @Cast("l_uint8") byte grayval );
public static native void rotateAMColorFastLow( @Cast("l_uint32*") IntPointer datad, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntPointer datas, @Cast("l_int32") int wpls, @Cast("l_float32") float angle, @Cast("l_uint32") int colorval );
public static native void rotateAMColorFastLow( @Cast("l_uint32*") IntBuffer datad, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntBuffer datas, @Cast("l_int32") int wpls, @Cast("l_float32") float angle, @Cast("l_uint32") int colorval );
public static native void rotateAMColorFastLow( @Cast("l_uint32*") int[] datad, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpld, @Cast("l_uint32*") int[] datas, @Cast("l_int32") int wpls, @Cast("l_float32") float angle, @Cast("l_uint32") int colorval );
public static native PIX pixRotateOrth( PIX pixs, @Cast("l_int32") int quads );
public static native PIX pixRotate180( PIX pixd, PIX pixs );
public static native PIX pixRotate90( PIX pixs, @Cast("l_int32") int direction );
public static native PIX pixFlipLR( PIX pixd, PIX pixs );
public static native PIX pixFlipTB( PIX pixd, PIX pixs );
public static native PIX pixRotateShear( PIX pixs, @Cast("l_int32") int xcen, @Cast("l_int32") int ycen, @Cast("l_float32") float angle, @Cast("l_int32") int incolor );
public static native PIX pixRotate2Shear( PIX pixs, @Cast("l_int32") int xcen, @Cast("l_int32") int ycen, @Cast("l_float32") float angle, @Cast("l_int32") int incolor );
public static native PIX pixRotate3Shear( PIX pixs, @Cast("l_int32") int xcen, @Cast("l_int32") int ycen, @Cast("l_float32") float angle, @Cast("l_int32") int incolor );
public static native @Cast("l_int32") int pixRotateShearIP( PIX pixs, @Cast("l_int32") int xcen, @Cast("l_int32") int ycen, @Cast("l_float32") float angle, @Cast("l_int32") int incolor );
public static native PIX pixRotateShearCenter( PIX pixs, @Cast("l_float32") float angle, @Cast("l_int32") int incolor );
public static native @Cast("l_int32") int pixRotateShearCenterIP( PIX pixs, @Cast("l_float32") float angle, @Cast("l_int32") int incolor );
public static native PIX pixStrokeWidthTransform( PIX pixs, @Cast("l_int32") int color, @Cast("l_int32") int depth, @Cast("l_int32") int nangles );
public static native PIX pixRunlengthTransform( PIX pixs, @Cast("l_int32") int color, @Cast("l_int32") int direction, @Cast("l_int32") int depth );
public static native @Cast("l_int32") int pixFindHorizontalRuns( PIX pix, @Cast("l_int32") int y, @Cast("l_int32*") IntPointer xstart, @Cast("l_int32*") IntPointer xend, @Cast("l_int32*") IntPointer pn );
public static native @Cast("l_int32") int pixFindHorizontalRuns( PIX pix, @Cast("l_int32") int y, @Cast("l_int32*") IntBuffer xstart, @Cast("l_int32*") IntBuffer xend, @Cast("l_int32*") IntBuffer pn );
public static native @Cast("l_int32") int pixFindHorizontalRuns( PIX pix, @Cast("l_int32") int y, @Cast("l_int32*") int[] xstart, @Cast("l_int32*") int[] xend, @Cast("l_int32*") int[] pn );
public static native @Cast("l_int32") int pixFindVerticalRuns( PIX pix, @Cast("l_int32") int x, @Cast("l_int32*") IntPointer ystart, @Cast("l_int32*") IntPointer yend, @Cast("l_int32*") IntPointer pn );
public static native @Cast("l_int32") int pixFindVerticalRuns( PIX pix, @Cast("l_int32") int x, @Cast("l_int32*") IntBuffer ystart, @Cast("l_int32*") IntBuffer yend, @Cast("l_int32*") IntBuffer pn );
public static native @Cast("l_int32") int pixFindVerticalRuns( PIX pix, @Cast("l_int32") int x, @Cast("l_int32*") int[] ystart, @Cast("l_int32*") int[] yend, @Cast("l_int32*") int[] pn );
public static native NUMA pixFindMaxRuns( PIX pix, @Cast("l_int32") int direction, @Cast("NUMA**") PointerPointer pnastart );
public static native NUMA pixFindMaxRuns( PIX pix, @Cast("l_int32") int direction, @ByPtrPtr NUMA pnastart );
public static native @Cast("l_int32") int pixFindMaxHorizontalRunOnLine( PIX pix, @Cast("l_int32") int y, @Cast("l_int32*") IntPointer pxstart, @Cast("l_int32*") IntPointer psize );
public static native @Cast("l_int32") int pixFindMaxHorizontalRunOnLine( PIX pix, @Cast("l_int32") int y, @Cast("l_int32*") IntBuffer pxstart, @Cast("l_int32*") IntBuffer psize );
public static native @Cast("l_int32") int pixFindMaxHorizontalRunOnLine( PIX pix, @Cast("l_int32") int y, @Cast("l_int32*") int[] pxstart, @Cast("l_int32*") int[] psize );
public static native @Cast("l_int32") int pixFindMaxVerticalRunOnLine( PIX pix, @Cast("l_int32") int x, @Cast("l_int32*") IntPointer pystart, @Cast("l_int32*") IntPointer psize );
public static native @Cast("l_int32") int pixFindMaxVerticalRunOnLine( PIX pix, @Cast("l_int32") int x, @Cast("l_int32*") IntBuffer pystart, @Cast("l_int32*") IntBuffer psize );
public static native @Cast("l_int32") int pixFindMaxVerticalRunOnLine( PIX pix, @Cast("l_int32") int x, @Cast("l_int32*") int[] pystart, @Cast("l_int32*") int[] psize );
public static native @Cast("l_int32") int runlengthMembershipOnLine( @Cast("l_int32*") IntPointer buffer, @Cast("l_int32") int size, @Cast("l_int32") int depth, @Cast("l_int32*") IntPointer start, @Cast("l_int32*") IntPointer end, @Cast("l_int32") int n );
public static native @Cast("l_int32") int runlengthMembershipOnLine( @Cast("l_int32*") IntBuffer buffer, @Cast("l_int32") int size, @Cast("l_int32") int depth, @Cast("l_int32*") IntBuffer start, @Cast("l_int32*") IntBuffer end, @Cast("l_int32") int n );
public static native @Cast("l_int32") int runlengthMembershipOnLine( @Cast("l_int32*") int[] buffer, @Cast("l_int32") int size, @Cast("l_int32") int depth, @Cast("l_int32*") int[] start, @Cast("l_int32*") int[] end, @Cast("l_int32") int n );
public static native @Cast("l_int32*") IntPointer makeMSBitLocTab( @Cast("l_int32") int bitval );
public static native SARRAY sarrayCreate( @Cast("l_int32") int n );
public static native SARRAY sarrayCreateInitialized( @Cast("l_int32") int n, @Cast("char*") BytePointer initstr );
public static native SARRAY sarrayCreateInitialized( @Cast("l_int32") int n, @Cast("char*") ByteBuffer initstr );
public static native SARRAY sarrayCreateInitialized( @Cast("l_int32") int n, @Cast("char*") byte[] initstr );
public static native SARRAY sarrayCreateWordsFromString( @Cast("const char*") BytePointer string );
public static native SARRAY sarrayCreateWordsFromString( String string );
public static native SARRAY sarrayCreateLinesFromString( @Cast("char*") BytePointer string, @Cast("l_int32") int blankflag );
public static native SARRAY sarrayCreateLinesFromString( @Cast("char*") ByteBuffer string, @Cast("l_int32") int blankflag );
public static native SARRAY sarrayCreateLinesFromString( @Cast("char*") byte[] string, @Cast("l_int32") int blankflag );
public static native void sarrayDestroy( @Cast("SARRAY**") PointerPointer psa );
public static native void sarrayDestroy( @ByPtrPtr SARRAY psa );
public static native SARRAY sarrayCopy( SARRAY sa );
public static native SARRAY sarrayClone( SARRAY sa );
public static native @Cast("l_int32") int sarrayAddString( SARRAY sa, @Cast("char*") BytePointer string, @Cast("l_int32") int copyflag );
public static native @Cast("l_int32") int sarrayAddString( SARRAY sa, @Cast("char*") ByteBuffer string, @Cast("l_int32") int copyflag );
public static native @Cast("l_int32") int sarrayAddString( SARRAY sa, @Cast("char*") byte[] string, @Cast("l_int32") int copyflag );
public static native @Cast("char*") BytePointer sarrayRemoveString( SARRAY sa, @Cast("l_int32") int index );
public static native @Cast("l_int32") int sarrayReplaceString( SARRAY sa, @Cast("l_int32") int index, @Cast("char*") BytePointer newstr, @Cast("l_int32") int copyflag );
public static native @Cast("l_int32") int sarrayReplaceString( SARRAY sa, @Cast("l_int32") int index, @Cast("char*") ByteBuffer newstr, @Cast("l_int32") int copyflag );
public static native @Cast("l_int32") int sarrayReplaceString( SARRAY sa, @Cast("l_int32") int index, @Cast("char*") byte[] newstr, @Cast("l_int32") int copyflag );
public static native @Cast("l_int32") int sarrayClear( SARRAY sa );
public static native @Cast("l_int32") int sarrayGetCount( SARRAY sa );
public static native @Cast("char**") PointerPointer sarrayGetArray( SARRAY sa, @Cast("l_int32*") IntPointer pnalloc, @Cast("l_int32*") IntPointer pn );
public static native @Cast("char**") @ByPtrPtr ByteBuffer sarrayGetArray( SARRAY sa, @Cast("l_int32*") IntBuffer pnalloc, @Cast("l_int32*") IntBuffer pn );
public static native @Cast("char**") @ByPtrPtr byte[] sarrayGetArray( SARRAY sa, @Cast("l_int32*") int[] pnalloc, @Cast("l_int32*") int[] pn );
public static native @Cast("char*") BytePointer sarrayGetString( SARRAY sa, @Cast("l_int32") int index, @Cast("l_int32") int copyflag );
public static native @Cast("l_int32") int sarrayGetRefcount( SARRAY sa );
public static native @Cast("l_int32") int sarrayChangeRefcount( SARRAY sa, @Cast("l_int32") int delta );
public static native @Cast("char*") BytePointer sarrayToString( SARRAY sa, @Cast("l_int32") int addnlflag );
public static native @Cast("char*") BytePointer sarrayToStringRange( SARRAY sa, @Cast("l_int32") int first, @Cast("l_int32") int nstrings, @Cast("l_int32") int addnlflag );
public static native @Cast("l_int32") int sarrayConcatenate( SARRAY sa1, SARRAY sa2 );
public static native @Cast("l_int32") int sarrayAppendRange( SARRAY sa1, SARRAY sa2, @Cast("l_int32") int start, @Cast("l_int32") int end );
public static native @Cast("l_int32") int sarrayPadToSameSize( SARRAY sa1, SARRAY sa2, @Cast("char*") BytePointer padstring );
public static native @Cast("l_int32") int sarrayPadToSameSize( SARRAY sa1, SARRAY sa2, @Cast("char*") ByteBuffer padstring );
public static native @Cast("l_int32") int sarrayPadToSameSize( SARRAY sa1, SARRAY sa2, @Cast("char*") byte[] padstring );
public static native SARRAY sarrayConvertWordsToLines( SARRAY sa, @Cast("l_int32") int linesize );
public static native @Cast("l_int32") int sarraySplitString( SARRAY sa, @Cast("const char*") BytePointer str, @Cast("const char*") BytePointer separators );
public static native @Cast("l_int32") int sarraySplitString( SARRAY sa, String str, String separators );
public static native SARRAY sarraySelectBySubstring( SARRAY sain, @Cast("const char*") BytePointer substr );
public static native SARRAY sarraySelectBySubstring( SARRAY sain, String substr );
public static native SARRAY sarraySelectByRange( SARRAY sain, @Cast("l_int32") int first, @Cast("l_int32") int last );
public static native @Cast("l_int32") int sarrayParseRange( SARRAY sa, @Cast("l_int32") int start, @Cast("l_int32*") IntPointer pactualstart, @Cast("l_int32*") IntPointer pend, @Cast("l_int32*") IntPointer pnewstart, @Cast("const char*") BytePointer substr, @Cast("l_int32") int loc );
public static native @Cast("l_int32") int sarrayParseRange( SARRAY sa, @Cast("l_int32") int start, @Cast("l_int32*") IntBuffer pactualstart, @Cast("l_int32*") IntBuffer pend, @Cast("l_int32*") IntBuffer pnewstart, String substr, @Cast("l_int32") int loc );
public static native @Cast("l_int32") int sarrayParseRange( SARRAY sa, @Cast("l_int32") int start, @Cast("l_int32*") int[] pactualstart, @Cast("l_int32*") int[] pend, @Cast("l_int32*") int[] pnewstart, @Cast("const char*") BytePointer substr, @Cast("l_int32") int loc );
public static native @Cast("l_int32") int sarrayParseRange( SARRAY sa, @Cast("l_int32") int start, @Cast("l_int32*") IntPointer pactualstart, @Cast("l_int32*") IntPointer pend, @Cast("l_int32*") IntPointer pnewstart, String substr, @Cast("l_int32") int loc );
public static native @Cast("l_int32") int sarrayParseRange( SARRAY sa, @Cast("l_int32") int start, @Cast("l_int32*") IntBuffer pactualstart, @Cast("l_int32*") IntBuffer pend, @Cast("l_int32*") IntBuffer pnewstart, @Cast("const char*") BytePointer substr, @Cast("l_int32") int loc );
public static native @Cast("l_int32") int sarrayParseRange( SARRAY sa, @Cast("l_int32") int start, @Cast("l_int32*") int[] pactualstart, @Cast("l_int32*") int[] pend, @Cast("l_int32*") int[] pnewstart, String substr, @Cast("l_int32") int loc );
public static native SARRAY sarraySort( SARRAY saout, SARRAY sain, @Cast("l_int32") int sortorder );
public static native SARRAY sarraySortByIndex( SARRAY sain, NUMA naindex );
public static native @Cast("l_int32") int stringCompareLexical( @Cast("const char*") BytePointer str1, @Cast("const char*") BytePointer str2 );
public static native @Cast("l_int32") int stringCompareLexical( String str1, String str2 );
public static native SARRAY sarrayRead( @Cast("const char*") BytePointer filename );
public static native SARRAY sarrayRead( String filename );
public static native SARRAY sarrayReadStream( @Cast("FILE*") Pointer fp );
public static native @Cast("l_int32") int sarrayWrite( @Cast("const char*") BytePointer filename, SARRAY sa );
public static native @Cast("l_int32") int sarrayWrite( String filename, SARRAY sa );
public static native @Cast("l_int32") int sarrayWriteStream( @Cast("FILE*") Pointer fp, SARRAY sa );
public static native @Cast("l_int32") int sarrayAppend( @Cast("const char*") BytePointer filename, SARRAY sa );
public static native @Cast("l_int32") int sarrayAppend( String filename, SARRAY sa );
public static native SARRAY getNumberedPathnamesInDirectory( @Cast("const char*") BytePointer dirname, @Cast("const char*") BytePointer substr, @Cast("l_int32") int numpre, @Cast("l_int32") int numpost, @Cast("l_int32") int maxnum );
public static native SARRAY getNumberedPathnamesInDirectory( String dirname, String substr, @Cast("l_int32") int numpre, @Cast("l_int32") int numpost, @Cast("l_int32") int maxnum );
public static native SARRAY getSortedPathnamesInDirectory( @Cast("const char*") BytePointer dirname, @Cast("const char*") BytePointer substr, @Cast("l_int32") int first, @Cast("l_int32") int nfiles );
public static native SARRAY getSortedPathnamesInDirectory( String dirname, String substr, @Cast("l_int32") int first, @Cast("l_int32") int nfiles );
public static native SARRAY convertSortedToNumberedPathnames( SARRAY sa, @Cast("l_int32") int numpre, @Cast("l_int32") int numpost, @Cast("l_int32") int maxnum );
public static native SARRAY getFilenamesInDirectory( @Cast("const char*") BytePointer dirname );
public static native SARRAY getFilenamesInDirectory( String dirname );
public static native PIX pixScale( PIX pixs, @Cast("l_float32") float scalex, @Cast("l_float32") float scaley );
public static native PIX pixScaleToSize( PIX pixs, @Cast("l_int32") int wd, @Cast("l_int32") int hd );
public static native PIX pixScaleGeneral( PIX pixs, @Cast("l_float32") float scalex, @Cast("l_float32") float scaley, @Cast("l_float32") float sharpfract, @Cast("l_int32") int sharpwidth );
public static native PIX pixScaleLI( PIX pixs, @Cast("l_float32") float scalex, @Cast("l_float32") float scaley );
public static native PIX pixScaleColorLI( PIX pixs, @Cast("l_float32") float scalex, @Cast("l_float32") float scaley );
public static native PIX pixScaleColor2xLI( PIX pixs );
public static native PIX pixScaleColor4xLI( PIX pixs );
public static native PIX pixScaleGrayLI( PIX pixs, @Cast("l_float32") float scalex, @Cast("l_float32") float scaley );
public static native PIX pixScaleGray2xLI( PIX pixs );
public static native PIX pixScaleGray4xLI( PIX pixs );
public static native PIX pixScaleBySampling( PIX pixs, @Cast("l_float32") float scalex, @Cast("l_float32") float scaley );
public static native PIX pixScaleBySamplingToSize( PIX pixs, @Cast("l_int32") int wd, @Cast("l_int32") int hd );
public static native PIX pixScaleByIntSampling( PIX pixs, @Cast("l_int32") int factor );
public static native PIX pixScaleRGBToGrayFast( PIX pixs, @Cast("l_int32") int factor, @Cast("l_int32") int color );
public static native PIX pixScaleRGBToBinaryFast( PIX pixs, @Cast("l_int32") int factor, @Cast("l_int32") int thresh );
public static native PIX pixScaleGrayToBinaryFast( PIX pixs, @Cast("l_int32") int factor, @Cast("l_int32") int thresh );
public static native PIX pixScaleSmooth( PIX pix, @Cast("l_float32") float scalex, @Cast("l_float32") float scaley );
public static native PIX pixScaleRGBToGray2( PIX pixs, @Cast("l_float32") float rwt, @Cast("l_float32") float gwt, @Cast("l_float32") float bwt );
public static native PIX pixScaleAreaMap( PIX pix, @Cast("l_float32") float scalex, @Cast("l_float32") float scaley );
public static native PIX pixScaleAreaMap2( PIX pix );
public static native PIX pixScaleBinary( PIX pixs, @Cast("l_float32") float scalex, @Cast("l_float32") float scaley );
public static native PIX pixScaleToGray( PIX pixs, @Cast("l_float32") float scalefactor );
public static native PIX pixScaleToGrayFast( PIX pixs, @Cast("l_float32") float scalefactor );
public static native PIX pixScaleToGray2( PIX pixs );
public static native PIX pixScaleToGray3( PIX pixs );
public static native PIX pixScaleToGray4( PIX pixs );
public static native PIX pixScaleToGray6( PIX pixs );
public static native PIX pixScaleToGray8( PIX pixs );
public static native PIX pixScaleToGray16( PIX pixs );
public static native PIX pixScaleToGrayMipmap( PIX pixs, @Cast("l_float32") float scalefactor );
public static native PIX pixScaleMipmap( PIX pixs1, PIX pixs2, @Cast("l_float32") float scale );
public static native PIX pixExpandReplicate( PIX pixs, @Cast("l_int32") int factor );
public static native PIX pixScaleGray2xLIThresh( PIX pixs, @Cast("l_int32") int thresh );
public static native PIX pixScaleGray2xLIDither( PIX pixs );
public static native PIX pixScaleGray4xLIThresh( PIX pixs, @Cast("l_int32") int thresh );
public static native PIX pixScaleGray4xLIDither( PIX pixs );
public static native PIX pixScaleGrayMinMax( PIX pixs, @Cast("l_int32") int xfact, @Cast("l_int32") int yfact, @Cast("l_int32") int type );
public static native PIX pixScaleGrayMinMax2( PIX pixs, @Cast("l_int32") int type );
public static native PIX pixScaleGrayRankCascade( PIX pixs, @Cast("l_int32") int level1, @Cast("l_int32") int level2, @Cast("l_int32") int level3, @Cast("l_int32") int level4 );
public static native PIX pixScaleGrayRank2( PIX pixs, @Cast("l_int32") int rank );
public static native @Cast("l_int32") int pixScaleAndTransferAlpha( PIX pixd, PIX pixs, @Cast("l_float32") float scalex, @Cast("l_float32") float scaley );
public static native PIX pixScaleWithAlpha( PIX pixs, @Cast("l_float32") float scalex, @Cast("l_float32") float scaley, PIX pixg, @Cast("l_float32") float fract );
public static native void scaleColorLILow( @Cast("l_uint32*") IntPointer datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntPointer datas, @Cast("l_int32") int ws, @Cast("l_int32") int hs, @Cast("l_int32") int wpls );
public static native void scaleColorLILow( @Cast("l_uint32*") IntBuffer datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntBuffer datas, @Cast("l_int32") int ws, @Cast("l_int32") int hs, @Cast("l_int32") int wpls );
public static native void scaleColorLILow( @Cast("l_uint32*") int[] datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") int[] datas, @Cast("l_int32") int ws, @Cast("l_int32") int hs, @Cast("l_int32") int wpls );
public static native void scaleGrayLILow( @Cast("l_uint32*") IntPointer datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntPointer datas, @Cast("l_int32") int ws, @Cast("l_int32") int hs, @Cast("l_int32") int wpls );
public static native void scaleGrayLILow( @Cast("l_uint32*") IntBuffer datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntBuffer datas, @Cast("l_int32") int ws, @Cast("l_int32") int hs, @Cast("l_int32") int wpls );
public static native void scaleGrayLILow( @Cast("l_uint32*") int[] datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") int[] datas, @Cast("l_int32") int ws, @Cast("l_int32") int hs, @Cast("l_int32") int wpls );
public static native void scaleColor2xLILow( @Cast("l_uint32*") IntPointer datad, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntPointer datas, @Cast("l_int32") int ws, @Cast("l_int32") int hs, @Cast("l_int32") int wpls );
public static native void scaleColor2xLILow( @Cast("l_uint32*") IntBuffer datad, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntBuffer datas, @Cast("l_int32") int ws, @Cast("l_int32") int hs, @Cast("l_int32") int wpls );
public static native void scaleColor2xLILow( @Cast("l_uint32*") int[] datad, @Cast("l_int32") int wpld, @Cast("l_uint32*") int[] datas, @Cast("l_int32") int ws, @Cast("l_int32") int hs, @Cast("l_int32") int wpls );
public static native void scaleColor2xLILineLow( @Cast("l_uint32*") IntPointer lined, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntPointer lines, @Cast("l_int32") int ws, @Cast("l_int32") int wpls, @Cast("l_int32") int lastlineflag );
public static native void scaleColor2xLILineLow( @Cast("l_uint32*") IntBuffer lined, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntBuffer lines, @Cast("l_int32") int ws, @Cast("l_int32") int wpls, @Cast("l_int32") int lastlineflag );
public static native void scaleColor2xLILineLow( @Cast("l_uint32*") int[] lined, @Cast("l_int32") int wpld, @Cast("l_uint32*") int[] lines, @Cast("l_int32") int ws, @Cast("l_int32") int wpls, @Cast("l_int32") int lastlineflag );
public static native void scaleGray2xLILow( @Cast("l_uint32*") IntPointer datad, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntPointer datas, @Cast("l_int32") int ws, @Cast("l_int32") int hs, @Cast("l_int32") int wpls );
public static native void scaleGray2xLILow( @Cast("l_uint32*") IntBuffer datad, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntBuffer datas, @Cast("l_int32") int ws, @Cast("l_int32") int hs, @Cast("l_int32") int wpls );
public static native void scaleGray2xLILow( @Cast("l_uint32*") int[] datad, @Cast("l_int32") int wpld, @Cast("l_uint32*") int[] datas, @Cast("l_int32") int ws, @Cast("l_int32") int hs, @Cast("l_int32") int wpls );
public static native void scaleGray2xLILineLow( @Cast("l_uint32*") IntPointer lined, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntPointer lines, @Cast("l_int32") int ws, @Cast("l_int32") int wpls, @Cast("l_int32") int lastlineflag );
public static native void scaleGray2xLILineLow( @Cast("l_uint32*") IntBuffer lined, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntBuffer lines, @Cast("l_int32") int ws, @Cast("l_int32") int wpls, @Cast("l_int32") int lastlineflag );
public static native void scaleGray2xLILineLow( @Cast("l_uint32*") int[] lined, @Cast("l_int32") int wpld, @Cast("l_uint32*") int[] lines, @Cast("l_int32") int ws, @Cast("l_int32") int wpls, @Cast("l_int32") int lastlineflag );
public static native void scaleGray4xLILow( @Cast("l_uint32*") IntPointer datad, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntPointer datas, @Cast("l_int32") int ws, @Cast("l_int32") int hs, @Cast("l_int32") int wpls );
public static native void scaleGray4xLILow( @Cast("l_uint32*") IntBuffer datad, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntBuffer datas, @Cast("l_int32") int ws, @Cast("l_int32") int hs, @Cast("l_int32") int wpls );
public static native void scaleGray4xLILow( @Cast("l_uint32*") int[] datad, @Cast("l_int32") int wpld, @Cast("l_uint32*") int[] datas, @Cast("l_int32") int ws, @Cast("l_int32") int hs, @Cast("l_int32") int wpls );
public static native void scaleGray4xLILineLow( @Cast("l_uint32*") IntPointer lined, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntPointer lines, @Cast("l_int32") int ws, @Cast("l_int32") int wpls, @Cast("l_int32") int lastlineflag );
public static native void scaleGray4xLILineLow( @Cast("l_uint32*") IntBuffer lined, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntBuffer lines, @Cast("l_int32") int ws, @Cast("l_int32") int wpls, @Cast("l_int32") int lastlineflag );
public static native void scaleGray4xLILineLow( @Cast("l_uint32*") int[] lined, @Cast("l_int32") int wpld, @Cast("l_uint32*") int[] lines, @Cast("l_int32") int ws, @Cast("l_int32") int wpls, @Cast("l_int32") int lastlineflag );
public static native @Cast("l_int32") int scaleBySamplingLow( @Cast("l_uint32*") IntPointer datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntPointer datas, @Cast("l_int32") int ws, @Cast("l_int32") int hs, @Cast("l_int32") int d, @Cast("l_int32") int wpls );
public static native @Cast("l_int32") int scaleBySamplingLow( @Cast("l_uint32*") IntBuffer datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntBuffer datas, @Cast("l_int32") int ws, @Cast("l_int32") int hs, @Cast("l_int32") int d, @Cast("l_int32") int wpls );
public static native @Cast("l_int32") int scaleBySamplingLow( @Cast("l_uint32*") int[] datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") int[] datas, @Cast("l_int32") int ws, @Cast("l_int32") int hs, @Cast("l_int32") int d, @Cast("l_int32") int wpls );
public static native @Cast("l_int32") int scaleSmoothLow( @Cast("l_uint32*") IntPointer datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntPointer datas, @Cast("l_int32") int ws, @Cast("l_int32") int hs, @Cast("l_int32") int d, @Cast("l_int32") int wpls, @Cast("l_int32") int size );
public static native @Cast("l_int32") int scaleSmoothLow( @Cast("l_uint32*") IntBuffer datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntBuffer datas, @Cast("l_int32") int ws, @Cast("l_int32") int hs, @Cast("l_int32") int d, @Cast("l_int32") int wpls, @Cast("l_int32") int size );
public static native @Cast("l_int32") int scaleSmoothLow( @Cast("l_uint32*") int[] datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") int[] datas, @Cast("l_int32") int ws, @Cast("l_int32") int hs, @Cast("l_int32") int d, @Cast("l_int32") int wpls, @Cast("l_int32") int size );
public static native void scaleRGBToGray2Low( @Cast("l_uint32*") IntPointer datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntPointer datas, @Cast("l_int32") int wpls, @Cast("l_float32") float rwt, @Cast("l_float32") float gwt, @Cast("l_float32") float bwt );
public static native void scaleRGBToGray2Low( @Cast("l_uint32*") IntBuffer datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntBuffer datas, @Cast("l_int32") int wpls, @Cast("l_float32") float rwt, @Cast("l_float32") float gwt, @Cast("l_float32") float bwt );
public static native void scaleRGBToGray2Low( @Cast("l_uint32*") int[] datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") int[] datas, @Cast("l_int32") int wpls, @Cast("l_float32") float rwt, @Cast("l_float32") float gwt, @Cast("l_float32") float bwt );
public static native void scaleColorAreaMapLow( @Cast("l_uint32*") IntPointer datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntPointer datas, @Cast("l_int32") int ws, @Cast("l_int32") int hs, @Cast("l_int32") int wpls );
public static native void scaleColorAreaMapLow( @Cast("l_uint32*") IntBuffer datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntBuffer datas, @Cast("l_int32") int ws, @Cast("l_int32") int hs, @Cast("l_int32") int wpls );
public static native void scaleColorAreaMapLow( @Cast("l_uint32*") int[] datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") int[] datas, @Cast("l_int32") int ws, @Cast("l_int32") int hs, @Cast("l_int32") int wpls );
public static native void scaleGrayAreaMapLow( @Cast("l_uint32*") IntPointer datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntPointer datas, @Cast("l_int32") int ws, @Cast("l_int32") int hs, @Cast("l_int32") int wpls );
public static native void scaleGrayAreaMapLow( @Cast("l_uint32*") IntBuffer datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntBuffer datas, @Cast("l_int32") int ws, @Cast("l_int32") int hs, @Cast("l_int32") int wpls );
public static native void scaleGrayAreaMapLow( @Cast("l_uint32*") int[] datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") int[] datas, @Cast("l_int32") int ws, @Cast("l_int32") int hs, @Cast("l_int32") int wpls );
public static native void scaleAreaMapLow2( @Cast("l_uint32*") IntPointer datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntPointer datas, @Cast("l_int32") int d, @Cast("l_int32") int wpls );
public static native void scaleAreaMapLow2( @Cast("l_uint32*") IntBuffer datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntBuffer datas, @Cast("l_int32") int d, @Cast("l_int32") int wpls );
public static native void scaleAreaMapLow2( @Cast("l_uint32*") int[] datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") int[] datas, @Cast("l_int32") int d, @Cast("l_int32") int wpls );
public static native @Cast("l_int32") int scaleBinaryLow( @Cast("l_uint32*") IntPointer datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntPointer datas, @Cast("l_int32") int ws, @Cast("l_int32") int hs, @Cast("l_int32") int wpls );
public static native @Cast("l_int32") int scaleBinaryLow( @Cast("l_uint32*") IntBuffer datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntBuffer datas, @Cast("l_int32") int ws, @Cast("l_int32") int hs, @Cast("l_int32") int wpls );
public static native @Cast("l_int32") int scaleBinaryLow( @Cast("l_uint32*") int[] datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") int[] datas, @Cast("l_int32") int ws, @Cast("l_int32") int hs, @Cast("l_int32") int wpls );
public static native void scaleToGray2Low( @Cast("l_uint32*") IntPointer datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntPointer datas, @Cast("l_int32") int wpls, @Cast("l_uint32*") IntPointer sumtab, @Cast("l_uint8*") BytePointer valtab );
public static native void scaleToGray2Low( @Cast("l_uint32*") IntBuffer datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntBuffer datas, @Cast("l_int32") int wpls, @Cast("l_uint32*") IntBuffer sumtab, @Cast("l_uint8*") ByteBuffer valtab );
public static native void scaleToGray2Low( @Cast("l_uint32*") int[] datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") int[] datas, @Cast("l_int32") int wpls, @Cast("l_uint32*") int[] sumtab, @Cast("l_uint8*") byte[] valtab );
public static native @Cast("l_uint32*") IntPointer makeSumTabSG2( );
public static native @Cast("l_uint8*") BytePointer makeValTabSG2( );
public static native void scaleToGray3Low( @Cast("l_uint32*") IntPointer datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntPointer datas, @Cast("l_int32") int wpls, @Cast("l_uint32*") IntPointer sumtab, @Cast("l_uint8*") BytePointer valtab );
public static native void scaleToGray3Low( @Cast("l_uint32*") IntBuffer datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntBuffer datas, @Cast("l_int32") int wpls, @Cast("l_uint32*") IntBuffer sumtab, @Cast("l_uint8*") ByteBuffer valtab );
public static native void scaleToGray3Low( @Cast("l_uint32*") int[] datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") int[] datas, @Cast("l_int32") int wpls, @Cast("l_uint32*") int[] sumtab, @Cast("l_uint8*") byte[] valtab );
public static native @Cast("l_uint32*") IntPointer makeSumTabSG3( );
public static native @Cast("l_uint8*") BytePointer makeValTabSG3( );
public static native void scaleToGray4Low( @Cast("l_uint32*") IntPointer datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntPointer datas, @Cast("l_int32") int wpls, @Cast("l_uint32*") IntPointer sumtab, @Cast("l_uint8*") BytePointer valtab );
public static native void scaleToGray4Low( @Cast("l_uint32*") IntBuffer datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntBuffer datas, @Cast("l_int32") int wpls, @Cast("l_uint32*") IntBuffer sumtab, @Cast("l_uint8*") ByteBuffer valtab );
public static native void scaleToGray4Low( @Cast("l_uint32*") int[] datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") int[] datas, @Cast("l_int32") int wpls, @Cast("l_uint32*") int[] sumtab, @Cast("l_uint8*") byte[] valtab );
public static native @Cast("l_uint32*") IntPointer makeSumTabSG4( );
public static native @Cast("l_uint8*") BytePointer makeValTabSG4( );
public static native void scaleToGray6Low( @Cast("l_uint32*") IntPointer datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntPointer datas, @Cast("l_int32") int wpls, @Cast("l_int32*") IntPointer tab8, @Cast("l_uint8*") BytePointer valtab );
public static native void scaleToGray6Low( @Cast("l_uint32*") IntBuffer datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntBuffer datas, @Cast("l_int32") int wpls, @Cast("l_int32*") IntBuffer tab8, @Cast("l_uint8*") ByteBuffer valtab );
public static native void scaleToGray6Low( @Cast("l_uint32*") int[] datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") int[] datas, @Cast("l_int32") int wpls, @Cast("l_int32*") int[] tab8, @Cast("l_uint8*") byte[] valtab );
public static native @Cast("l_uint8*") BytePointer makeValTabSG6( );
public static native void scaleToGray8Low( @Cast("l_uint32*") IntPointer datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntPointer datas, @Cast("l_int32") int wpls, @Cast("l_int32*") IntPointer tab8, @Cast("l_uint8*") BytePointer valtab );
public static native void scaleToGray8Low( @Cast("l_uint32*") IntBuffer datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntBuffer datas, @Cast("l_int32") int wpls, @Cast("l_int32*") IntBuffer tab8, @Cast("l_uint8*") ByteBuffer valtab );
public static native void scaleToGray8Low( @Cast("l_uint32*") int[] datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") int[] datas, @Cast("l_int32") int wpls, @Cast("l_int32*") int[] tab8, @Cast("l_uint8*") byte[] valtab );
public static native @Cast("l_uint8*") BytePointer makeValTabSG8( );
public static native void scaleToGray16Low( @Cast("l_uint32*") IntPointer datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntPointer datas, @Cast("l_int32") int wpls, @Cast("l_int32*") IntPointer tab8 );
public static native void scaleToGray16Low( @Cast("l_uint32*") IntBuffer datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntBuffer datas, @Cast("l_int32") int wpls, @Cast("l_int32*") IntBuffer tab8 );
public static native void scaleToGray16Low( @Cast("l_uint32*") int[] datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") int[] datas, @Cast("l_int32") int wpls, @Cast("l_int32*") int[] tab8 );
public static native @Cast("l_int32") int scaleMipmapLow( @Cast("l_uint32*") IntPointer datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntPointer datas1, @Cast("l_int32") int wpls1, @Cast("l_uint32*") IntPointer datas2, @Cast("l_int32") int wpls2, @Cast("l_float32") float red );
public static native @Cast("l_int32") int scaleMipmapLow( @Cast("l_uint32*") IntBuffer datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntBuffer datas1, @Cast("l_int32") int wpls1, @Cast("l_uint32*") IntBuffer datas2, @Cast("l_int32") int wpls2, @Cast("l_float32") float red );
public static native @Cast("l_int32") int scaleMipmapLow( @Cast("l_uint32*") int[] datad, @Cast("l_int32") int wd, @Cast("l_int32") int hd, @Cast("l_int32") int wpld, @Cast("l_uint32*") int[] datas1, @Cast("l_int32") int wpls1, @Cast("l_uint32*") int[] datas2, @Cast("l_int32") int wpls2, @Cast("l_float32") float red );
public static native PIX pixSeedfillBinary( PIX pixd, PIX pixs, PIX pixm, @Cast("l_int32") int connectivity );
public static native PIX pixSeedfillBinaryRestricted( PIX pixd, PIX pixs, PIX pixm, @Cast("l_int32") int connectivity, @Cast("l_int32") int xmax, @Cast("l_int32") int ymax );
public static native PIX pixHolesByFilling( PIX pixs, @Cast("l_int32") int connectivity );
public static native PIX pixFillClosedBorders( PIX pixs, @Cast("l_int32") int connectivity );
public static native PIX pixExtractBorderConnComps( PIX pixs, @Cast("l_int32") int connectivity );
public static native PIX pixRemoveBorderConnComps( PIX pixs, @Cast("l_int32") int connectivity );
public static native PIX pixFillBgFromBorder( PIX pixs, @Cast("l_int32") int connectivity );
public static native PIX pixFillHolesToBoundingRect( PIX pixs, @Cast("l_int32") int minsize, @Cast("l_float32") float maxhfract, @Cast("l_float32") float minfgfract );
public static native @Cast("l_int32") int pixSeedfillGray( PIX pixs, PIX pixm, @Cast("l_int32") int connectivity );
public static native @Cast("l_int32") int pixSeedfillGrayInv( PIX pixs, PIX pixm, @Cast("l_int32") int connectivity );
public static native @Cast("l_int32") int pixSeedfillGraySimple( PIX pixs, PIX pixm, @Cast("l_int32") int connectivity );
public static native @Cast("l_int32") int pixSeedfillGrayInvSimple( PIX pixs, PIX pixm, @Cast("l_int32") int connectivity );
public static native PIX pixSeedfillGrayBasin( PIX pixb, PIX pixm, @Cast("l_int32") int delta, @Cast("l_int32") int connectivity );
public static native PIX pixDistanceFunction( PIX pixs, @Cast("l_int32") int connectivity, @Cast("l_int32") int outdepth, @Cast("l_int32") int boundcond );
public static native PIX pixSeedspread( PIX pixs, @Cast("l_int32") int connectivity );
public static native @Cast("l_int32") int pixLocalExtrema( PIX pixs, @Cast("l_int32") int maxmin, @Cast("l_int32") int minmax, @Cast("PIX**") PointerPointer ppixmin, @Cast("PIX**") PointerPointer ppixmax );
public static native @Cast("l_int32") int pixLocalExtrema( PIX pixs, @Cast("l_int32") int maxmin, @Cast("l_int32") int minmax, @ByPtrPtr PIX ppixmin, @ByPtrPtr PIX ppixmax );
public static native @Cast("l_int32") int pixSelectedLocalExtrema( PIX pixs, @Cast("l_int32") int mindist, @Cast("PIX**") PointerPointer ppixmin, @Cast("PIX**") PointerPointer ppixmax );
public static native @Cast("l_int32") int pixSelectedLocalExtrema( PIX pixs, @Cast("l_int32") int mindist, @ByPtrPtr PIX ppixmin, @ByPtrPtr PIX ppixmax );
public static native PIX pixFindEqualValues( PIX pixs1, PIX pixs2 );
public static native @Cast("l_int32") int pixSelectMinInConnComp( PIX pixs, PIX pixm, @Cast("PTA**") PointerPointer ppta, @Cast("NUMA**") PointerPointer pnav );
public static native @Cast("l_int32") int pixSelectMinInConnComp( PIX pixs, PIX pixm, @ByPtrPtr PTA ppta, @ByPtrPtr NUMA pnav );
public static native PIX pixRemoveSeededComponents( PIX pixd, PIX pixs, PIX pixm, @Cast("l_int32") int connectivity, @Cast("l_int32") int bordersize );
public static native void seedfillBinaryLow( @Cast("l_uint32*") IntPointer datas, @Cast("l_int32") int hs, @Cast("l_int32") int wpls, @Cast("l_uint32*") IntPointer datam, @Cast("l_int32") int hm, @Cast("l_int32") int wplm, @Cast("l_int32") int connectivity );
public static native void seedfillBinaryLow( @Cast("l_uint32*") IntBuffer datas, @Cast("l_int32") int hs, @Cast("l_int32") int wpls, @Cast("l_uint32*") IntBuffer datam, @Cast("l_int32") int hm, @Cast("l_int32") int wplm, @Cast("l_int32") int connectivity );
public static native void seedfillBinaryLow( @Cast("l_uint32*") int[] datas, @Cast("l_int32") int hs, @Cast("l_int32") int wpls, @Cast("l_uint32*") int[] datam, @Cast("l_int32") int hm, @Cast("l_int32") int wplm, @Cast("l_int32") int connectivity );
public static native void seedfillGrayLow( @Cast("l_uint32*") IntPointer datas, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpls, @Cast("l_uint32*") IntPointer datam, @Cast("l_int32") int wplm, @Cast("l_int32") int connectivity );
public static native void seedfillGrayLow( @Cast("l_uint32*") IntBuffer datas, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpls, @Cast("l_uint32*") IntBuffer datam, @Cast("l_int32") int wplm, @Cast("l_int32") int connectivity );
public static native void seedfillGrayLow( @Cast("l_uint32*") int[] datas, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpls, @Cast("l_uint32*") int[] datam, @Cast("l_int32") int wplm, @Cast("l_int32") int connectivity );
public static native void seedfillGrayInvLow( @Cast("l_uint32*") IntPointer datas, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpls, @Cast("l_uint32*") IntPointer datam, @Cast("l_int32") int wplm, @Cast("l_int32") int connectivity );
public static native void seedfillGrayInvLow( @Cast("l_uint32*") IntBuffer datas, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpls, @Cast("l_uint32*") IntBuffer datam, @Cast("l_int32") int wplm, @Cast("l_int32") int connectivity );
public static native void seedfillGrayInvLow( @Cast("l_uint32*") int[] datas, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpls, @Cast("l_uint32*") int[] datam, @Cast("l_int32") int wplm, @Cast("l_int32") int connectivity );
public static native void seedfillGrayLowSimple( @Cast("l_uint32*") IntPointer datas, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpls, @Cast("l_uint32*") IntPointer datam, @Cast("l_int32") int wplm, @Cast("l_int32") int connectivity );
public static native void seedfillGrayLowSimple( @Cast("l_uint32*") IntBuffer datas, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpls, @Cast("l_uint32*") IntBuffer datam, @Cast("l_int32") int wplm, @Cast("l_int32") int connectivity );
public static native void seedfillGrayLowSimple( @Cast("l_uint32*") int[] datas, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpls, @Cast("l_uint32*") int[] datam, @Cast("l_int32") int wplm, @Cast("l_int32") int connectivity );
public static native void seedfillGrayInvLowSimple( @Cast("l_uint32*") IntPointer datas, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpls, @Cast("l_uint32*") IntPointer datam, @Cast("l_int32") int wplm, @Cast("l_int32") int connectivity );
public static native void seedfillGrayInvLowSimple( @Cast("l_uint32*") IntBuffer datas, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpls, @Cast("l_uint32*") IntBuffer datam, @Cast("l_int32") int wplm, @Cast("l_int32") int connectivity );
public static native void seedfillGrayInvLowSimple( @Cast("l_uint32*") int[] datas, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpls, @Cast("l_uint32*") int[] datam, @Cast("l_int32") int wplm, @Cast("l_int32") int connectivity );
public static native void distanceFunctionLow( @Cast("l_uint32*") IntPointer datad, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int d, @Cast("l_int32") int wpld, @Cast("l_int32") int connectivity );
public static native void distanceFunctionLow( @Cast("l_uint32*") IntBuffer datad, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int d, @Cast("l_int32") int wpld, @Cast("l_int32") int connectivity );
public static native void distanceFunctionLow( @Cast("l_uint32*") int[] datad, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int d, @Cast("l_int32") int wpld, @Cast("l_int32") int connectivity );
public static native void seedspreadLow( @Cast("l_uint32*") IntPointer datad, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntPointer datat, @Cast("l_int32") int wplt, @Cast("l_int32") int connectivity );
public static native void seedspreadLow( @Cast("l_uint32*") IntBuffer datad, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpld, @Cast("l_uint32*") IntBuffer datat, @Cast("l_int32") int wplt, @Cast("l_int32") int connectivity );
public static native void seedspreadLow( @Cast("l_uint32*") int[] datad, @Cast("l_int32") int w, @Cast("l_int32") int h, @Cast("l_int32") int wpld, @Cast("l_uint32*") int[] datat, @Cast("l_int32") int wplt, @Cast("l_int32") int connectivity );
public static native SELA selaCreate( @Cast("l_int32") int n );
public static native void selaDestroy( @Cast("SELA**") PointerPointer psela );
public static native void selaDestroy( @ByPtrPtr SELA psela );
public static native SEL selCreate( @Cast("l_int32") int height, @Cast("l_int32") int width, @Cast("const char*") BytePointer name );
public static native SEL selCreate( @Cast("l_int32") int height, @Cast("l_int32") int width, String name );
public static native void selDestroy( @Cast("SEL**") PointerPointer psel );
public static native void selDestroy( @ByPtrPtr SEL psel );
public static native SEL selCopy( SEL sel );
public static native SEL selCreateBrick( @Cast("l_int32") int h, @Cast("l_int32") int w, @Cast("l_int32") int cy, @Cast("l_int32") int cx, @Cast("l_int32") int type );
public static native SEL selCreateComb( @Cast("l_int32") int factor1, @Cast("l_int32") int factor2, @Cast("l_int32") int direction );
public static native @Cast("l_int32**") PointerPointer create2dIntArray( @Cast("l_int32") int sy, @Cast("l_int32") int sx );
public static native @Cast("l_int32") int selaAddSel( SELA sela, SEL sel, @Cast("const char*") BytePointer selname, @Cast("l_int32") int copyflag );
public static native @Cast("l_int32") int selaAddSel( SELA sela, SEL sel, String selname, @Cast("l_int32") int copyflag );
public static native @Cast("l_int32") int selaGetCount( SELA sela );
public static native SEL selaGetSel( SELA sela, @Cast("l_int32") int i );
public static native @Cast("char*") BytePointer selGetName( SEL sel );
public static native @Cast("l_int32") int selSetName( SEL sel, @Cast("const char*") BytePointer name );
public static native @Cast("l_int32") int selSetName( SEL sel, String name );
public static native @Cast("l_int32") int selaFindSelByName( SELA sela, @Cast("const char*") BytePointer name, @Cast("l_int32*") IntPointer pindex, @Cast("SEL**") PointerPointer psel );
public static native @Cast("l_int32") int selaFindSelByName( SELA sela, @Cast("const char*") BytePointer name, @Cast("l_int32*") IntPointer pindex, @ByPtrPtr SEL psel );
public static native @Cast("l_int32") int selaFindSelByName( SELA sela, String name, @Cast("l_int32*") IntBuffer pindex, @ByPtrPtr SEL psel );
public static native @Cast("l_int32") int selaFindSelByName( SELA sela, @Cast("const char*") BytePointer name, @Cast("l_int32*") int[] pindex, @ByPtrPtr SEL psel );
public static native @Cast("l_int32") int selaFindSelByName( SELA sela, String name, @Cast("l_int32*") IntPointer pindex, @ByPtrPtr SEL psel );
public static native @Cast("l_int32") int selaFindSelByName( SELA sela, @Cast("const char*") BytePointer name, @Cast("l_int32*") IntBuffer pindex, @ByPtrPtr SEL psel );
public static native @Cast("l_int32") int selaFindSelByName( SELA sela, String name, @Cast("l_int32*") int[] pindex, @ByPtrPtr SEL psel );
public static native @Cast("l_int32") int selGetElement( SEL sel, @Cast("l_int32") int row, @Cast("l_int32") int col, @Cast("l_int32*") IntPointer ptype );
public static native @Cast("l_int32") int selGetElement( SEL sel, @Cast("l_int32") int row, @Cast("l_int32") int col, @Cast("l_int32*") IntBuffer ptype );
public static native @Cast("l_int32") int selGetElement( SEL sel, @Cast("l_int32") int row, @Cast("l_int32") int col, @Cast("l_int32*") int[] ptype );
public static native @Cast("l_int32") int selSetElement( SEL sel, @Cast("l_int32") int row, @Cast("l_int32") int col, @Cast("l_int32") int type );
public static native @Cast("l_int32") int selGetParameters( SEL sel, @Cast("l_int32*") IntPointer psy, @Cast("l_int32*") IntPointer psx, @Cast("l_int32*") IntPointer pcy, @Cast("l_int32*") IntPointer pcx );
public static native @Cast("l_int32") int selGetParameters( SEL sel, @Cast("l_int32*") IntBuffer psy, @Cast("l_int32*") IntBuffer psx, @Cast("l_int32*") IntBuffer pcy, @Cast("l_int32*") IntBuffer pcx );
public static native @Cast("l_int32") int selGetParameters( SEL sel, @Cast("l_int32*") int[] psy, @Cast("l_int32*") int[] psx, @Cast("l_int32*") int[] pcy, @Cast("l_int32*") int[] pcx );
public static native @Cast("l_int32") int selSetOrigin( SEL sel, @Cast("l_int32") int cy, @Cast("l_int32") int cx );
public static native @Cast("l_int32") int selGetTypeAtOrigin( SEL sel, @Cast("l_int32*") IntPointer ptype );
public static native @Cast("l_int32") int selGetTypeAtOrigin( SEL sel, @Cast("l_int32*") IntBuffer ptype );
public static native @Cast("l_int32") int selGetTypeAtOrigin( SEL sel, @Cast("l_int32*") int[] ptype );
public static native @Cast("char*") BytePointer selaGetBrickName( SELA sela, @Cast("l_int32") int hsize, @Cast("l_int32") int vsize );
public static native @Cast("char*") BytePointer selaGetCombName( SELA sela, @Cast("l_int32") int size, @Cast("l_int32") int direction );
public static native @Cast("l_int32") int getCompositeParameters( @Cast("l_int32") int size, @Cast("l_int32*") IntPointer psize1, @Cast("l_int32*") IntPointer psize2, @Cast("char**") PointerPointer pnameh1, @Cast("char**") PointerPointer pnameh2, @Cast("char**") PointerPointer pnamev1, @Cast("char**") PointerPointer pnamev2 );
public static native @Cast("l_int32") int getCompositeParameters( @Cast("l_int32") int size, @Cast("l_int32*") IntPointer psize1, @Cast("l_int32*") IntPointer psize2, @Cast("char**") @ByPtrPtr BytePointer pnameh1, @Cast("char**") @ByPtrPtr BytePointer pnameh2, @Cast("char**") @ByPtrPtr BytePointer pnamev1, @Cast("char**") @ByPtrPtr BytePointer pnamev2 );
public static native @Cast("l_int32") int getCompositeParameters( @Cast("l_int32") int size, @Cast("l_int32*") IntBuffer psize1, @Cast("l_int32*") IntBuffer psize2, @Cast("char**") @ByPtrPtr ByteBuffer pnameh1, @Cast("char**") @ByPtrPtr ByteBuffer pnameh2, @Cast("char**") @ByPtrPtr ByteBuffer pnamev1, @Cast("char**") @ByPtrPtr ByteBuffer pnamev2 );
public static native @Cast("l_int32") int getCompositeParameters( @Cast("l_int32") int size, @Cast("l_int32*") int[] psize1, @Cast("l_int32*") int[] psize2, @Cast("char**") @ByPtrPtr byte[] pnameh1, @Cast("char**") @ByPtrPtr byte[] pnameh2, @Cast("char**") @ByPtrPtr byte[] pnamev1, @Cast("char**") @ByPtrPtr byte[] pnamev2 );
public static native SARRAY selaGetSelnames( SELA sela );
public static native @Cast("l_int32") int selFindMaxTranslations( SEL sel, @Cast("l_int32*") IntPointer pxp, @Cast("l_int32*") IntPointer pyp, @Cast("l_int32*") IntPointer pxn, @Cast("l_int32*") IntPointer pyn );
public static native @Cast("l_int32") int selFindMaxTranslations( SEL sel, @Cast("l_int32*") IntBuffer pxp, @Cast("l_int32*") IntBuffer pyp, @Cast("l_int32*") IntBuffer pxn, @Cast("l_int32*") IntBuffer pyn );
public static native @Cast("l_int32") int selFindMaxTranslations( SEL sel, @Cast("l_int32*") int[] pxp, @Cast("l_int32*") int[] pyp, @Cast("l_int32*") int[] pxn, @Cast("l_int32*") int[] pyn );
public static native SEL selRotateOrth( SEL sel, @Cast("l_int32") int quads );
public static native SELA selaRead( @Cast("const char*") BytePointer fname );
public static native SELA selaRead( String fname );
public static native SELA selaReadStream( @Cast("FILE*") Pointer fp );
public static native SEL selRead( @Cast("const char*") BytePointer fname );
public static native SEL selRead( String fname );
public static native SEL selReadStream( @Cast("FILE*") Pointer fp );
public static native @Cast("l_int32") int selaWrite( @Cast("const char*") BytePointer fname, SELA sela );
public static native @Cast("l_int32") int selaWrite( String fname, SELA sela );
public static native @Cast("l_int32") int selaWriteStream( @Cast("FILE*") Pointer fp, SELA sela );
public static native @Cast("l_int32") int selWrite( @Cast("const char*") BytePointer fname, SEL sel );
public static native @Cast("l_int32") int selWrite( String fname, SEL sel );
public static native @Cast("l_int32") int selWriteStream( @Cast("FILE*") Pointer fp, SEL sel );
public static native SEL selCreateFromString( @Cast("const char*") BytePointer text, @Cast("l_int32") int h, @Cast("l_int32") int w, @Cast("const char*") BytePointer name );
public static native SEL selCreateFromString( String text, @Cast("l_int32") int h, @Cast("l_int32") int w, String name );
public static native @Cast("char*") BytePointer selPrintToString( SEL sel );
public static native SELA selaCreateFromFile( @Cast("const char*") BytePointer filename );
public static native SELA selaCreateFromFile( String filename );
public static native SEL selCreateFromPta( PTA pta, @Cast("l_int32") int cy, @Cast("l_int32") int cx, @Cast("const char*") BytePointer name );
public static native SEL selCreateFromPta( PTA pta, @Cast("l_int32") int cy, @Cast("l_int32") int cx, String name );
public static native SEL selCreateFromPix( PIX pix, @Cast("l_int32") int cy, @Cast("l_int32") int cx, @Cast("const char*") BytePointer name );
public static native SEL selCreateFromPix( PIX pix, @Cast("l_int32") int cy, @Cast("l_int32") int cx, String name );
public static native SEL selReadFromColorImage( @Cast("const char*") BytePointer pathname );
public static native SEL selReadFromColorImage( String pathname );
public static native SEL selCreateFromColorPix( PIX pixs, @Cast("char*") BytePointer selname );
public static native SEL selCreateFromColorPix( PIX pixs, @Cast("char*") ByteBuffer selname );
public static native SEL selCreateFromColorPix( PIX pixs, @Cast("char*") byte[] selname );
public static native PIX selDisplayInPix( SEL sel, @Cast("l_int32") int size, @Cast("l_int32") int gthick );
public static native PIX selaDisplayInPix( SELA sela, @Cast("l_int32") int size, @Cast("l_int32") int gthick, @Cast("l_int32") int spacing, @Cast("l_int32") int ncols );
public static native SELA selaAddBasic( SELA sela );
public static native SELA selaAddHitMiss( SELA sela );
public static native SELA selaAddDwaLinear( SELA sela );
public static native SELA selaAddDwaCombs( SELA sela );
public static native SELA selaAddCrossJunctions( SELA sela, @Cast("l_float32") float hlsize, @Cast("l_float32") float mdist, @Cast("l_int32") int norient, @Cast("l_int32") int debugflag );
public static native SELA selaAddTJunctions( SELA sela, @Cast("l_float32") float hlsize, @Cast("l_float32") float mdist, @Cast("l_int32") int norient, @Cast("l_int32") int debugflag );
public static native SEL pixGenerateSelWithRuns( PIX pixs, @Cast("l_int32") int nhlines, @Cast("l_int32") int nvlines, @Cast("l_int32") int distance, @Cast("l_int32") int minlength, @Cast("l_int32") int toppix, @Cast("l_int32") int botpix, @Cast("l_int32") int leftpix, @Cast("l_int32") int rightpix, @Cast("PIX**") PointerPointer ppixe );
public static native SEL pixGenerateSelWithRuns( PIX pixs, @Cast("l_int32") int nhlines, @Cast("l_int32") int nvlines, @Cast("l_int32") int distance, @Cast("l_int32") int minlength, @Cast("l_int32") int toppix, @Cast("l_int32") int botpix, @Cast("l_int32") int leftpix, @Cast("l_int32") int rightpix, @ByPtrPtr PIX ppixe );
public static native SEL pixGenerateSelRandom( PIX pixs, @Cast("l_float32") float hitfract, @Cast("l_float32") float missfract, @Cast("l_int32") int distance, @Cast("l_int32") int toppix, @Cast("l_int32") int botpix, @Cast("l_int32") int leftpix, @Cast("l_int32") int rightpix, @Cast("PIX**") PointerPointer ppixe );
public static native SEL pixGenerateSelRandom( PIX pixs, @Cast("l_float32") float hitfract, @Cast("l_float32") float missfract, @Cast("l_int32") int distance, @Cast("l_int32") int toppix, @Cast("l_int32") int botpix, @Cast("l_int32") int leftpix, @Cast("l_int32") int rightpix, @ByPtrPtr PIX ppixe );
public static native SEL pixGenerateSelBoundary( PIX pixs, @Cast("l_int32") int hitdist, @Cast("l_int32") int missdist, @Cast("l_int32") int hitskip, @Cast("l_int32") int missskip, @Cast("l_int32") int topflag, @Cast("l_int32") int botflag, @Cast("l_int32") int leftflag, @Cast("l_int32") int rightflag, @Cast("PIX**") PointerPointer ppixe );
public static native SEL pixGenerateSelBoundary( PIX pixs, @Cast("l_int32") int hitdist, @Cast("l_int32") int missdist, @Cast("l_int32") int hitskip, @Cast("l_int32") int missskip, @Cast("l_int32") int topflag, @Cast("l_int32") int botflag, @Cast("l_int32") int leftflag, @Cast("l_int32") int rightflag, @ByPtrPtr PIX ppixe );
public static native NUMA pixGetRunCentersOnLine( PIX pixs, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32") int minlength );
public static native NUMA pixGetRunsOnLine( PIX pixs, @Cast("l_int32") int x1, @Cast("l_int32") int y1, @Cast("l_int32") int x2, @Cast("l_int32") int y2 );
public static native PTA pixSubsampleBoundaryPixels( PIX pixs, @Cast("l_int32") int skip );
public static native @Cast("l_int32") int adjacentOnPixelInRaster( PIX pixs, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32*") IntPointer pxa, @Cast("l_int32*") IntPointer pya );
public static native @Cast("l_int32") int adjacentOnPixelInRaster( PIX pixs, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32*") IntBuffer pxa, @Cast("l_int32*") IntBuffer pya );
public static native @Cast("l_int32") int adjacentOnPixelInRaster( PIX pixs, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("l_int32*") int[] pxa, @Cast("l_int32*") int[] pya );
public static native PIX pixDisplayHitMissSel( PIX pixs, SEL sel, @Cast("l_int32") int scalefactor, @Cast("l_uint32") int hitcolor, @Cast("l_uint32") int misscolor );
public static native PIX pixHShear( PIX pixd, PIX pixs, @Cast("l_int32") int yloc, @Cast("l_float32") float radang, @Cast("l_int32") int incolor );
public static native PIX pixVShear( PIX pixd, PIX pixs, @Cast("l_int32") int xloc, @Cast("l_float32") float radang, @Cast("l_int32") int incolor );
public static native PIX pixHShearCorner( PIX pixd, PIX pixs, @Cast("l_float32") float radang, @Cast("l_int32") int incolor );
public static native PIX pixVShearCorner( PIX pixd, PIX pixs, @Cast("l_float32") float radang, @Cast("l_int32") int incolor );
public static native PIX pixHShearCenter( PIX pixd, PIX pixs, @Cast("l_float32") float radang, @Cast("l_int32") int incolor );
public static native PIX pixVShearCenter( PIX pixd, PIX pixs, @Cast("l_float32") float radang, @Cast("l_int32") int incolor );
public static native @Cast("l_int32") int pixHShearIP( PIX pixs, @Cast("l_int32") int yloc, @Cast("l_float32") float radang, @Cast("l_int32") int incolor );
public static native @Cast("l_int32") int pixVShearIP( PIX pixs, @Cast("l_int32") int xloc, @Cast("l_float32") float radang, @Cast("l_int32") int incolor );
public static native PIX pixHShearLI( PIX pixs, @Cast("l_int32") int yloc, @Cast("l_float32") float radang, @Cast("l_int32") int incolor );
public static native PIX pixVShearLI( PIX pixs, @Cast("l_int32") int xloc, @Cast("l_float32") float radang, @Cast("l_int32") int incolor );
public static native PIX pixDeskew( PIX pixs, @Cast("l_int32") int redsearch );
public static native PIX pixFindSkewAndDeskew( PIX pixs, @Cast("l_int32") int redsearch, @Cast("l_float32*") FloatPointer pangle, @Cast("l_float32*") FloatPointer pconf );
public static native PIX pixFindSkewAndDeskew( PIX pixs, @Cast("l_int32") int redsearch, @Cast("l_float32*") FloatBuffer pangle, @Cast("l_float32*") FloatBuffer pconf );
public static native PIX pixFindSkewAndDeskew( PIX pixs, @Cast("l_int32") int redsearch, @Cast("l_float32*") float[] pangle, @Cast("l_float32*") float[] pconf );
public static native PIX pixDeskewGeneral( PIX pixs, @Cast("l_int32") int redsweep, @Cast("l_float32") float sweeprange, @Cast("l_float32") float sweepdelta, @Cast("l_int32") int redsearch, @Cast("l_int32") int thresh, @Cast("l_float32*") FloatPointer pangle, @Cast("l_float32*") FloatPointer pconf );
public static native PIX pixDeskewGeneral( PIX pixs, @Cast("l_int32") int redsweep, @Cast("l_float32") float sweeprange, @Cast("l_float32") float sweepdelta, @Cast("l_int32") int redsearch, @Cast("l_int32") int thresh, @Cast("l_float32*") FloatBuffer pangle, @Cast("l_float32*") FloatBuffer pconf );
public static native PIX pixDeskewGeneral( PIX pixs, @Cast("l_int32") int redsweep, @Cast("l_float32") float sweeprange, @Cast("l_float32") float sweepdelta, @Cast("l_int32") int redsearch, @Cast("l_int32") int thresh, @Cast("l_float32*") float[] pangle, @Cast("l_float32*") float[] pconf );
public static native @Cast("l_int32") int pixFindSkew( PIX pixs, @Cast("l_float32*") FloatPointer pangle, @Cast("l_float32*") FloatPointer pconf );
public static native @Cast("l_int32") int pixFindSkew( PIX pixs, @Cast("l_float32*") FloatBuffer pangle, @Cast("l_float32*") FloatBuffer pconf );
public static native @Cast("l_int32") int pixFindSkew( PIX pixs, @Cast("l_float32*") float[] pangle, @Cast("l_float32*") float[] pconf );
public static native @Cast("l_int32") int pixFindSkewSweep( PIX pixs, @Cast("l_float32*") FloatPointer pangle, @Cast("l_int32") int reduction, @Cast("l_float32") float sweeprange, @Cast("l_float32") float sweepdelta );
public static native @Cast("l_int32") int pixFindSkewSweep( PIX pixs, @Cast("l_float32*") FloatBuffer pangle, @Cast("l_int32") int reduction, @Cast("l_float32") float sweeprange, @Cast("l_float32") float sweepdelta );
public static native @Cast("l_int32") int pixFindSkewSweep( PIX pixs, @Cast("l_float32*") float[] pangle, @Cast("l_int32") int reduction, @Cast("l_float32") float sweeprange, @Cast("l_float32") float sweepdelta );
public static native @Cast("l_int32") int pixFindSkewSweepAndSearch( PIX pixs, @Cast("l_float32*") FloatPointer pangle, @Cast("l_float32*") FloatPointer pconf, @Cast("l_int32") int redsweep, @Cast("l_int32") int redsearch, @Cast("l_float32") float sweeprange, @Cast("l_float32") float sweepdelta, @Cast("l_float32") float minbsdelta );
public static native @Cast("l_int32") int pixFindSkewSweepAndSearch( PIX pixs, @Cast("l_float32*") FloatBuffer pangle, @Cast("l_float32*") FloatBuffer pconf, @Cast("l_int32") int redsweep, @Cast("l_int32") int redsearch, @Cast("l_float32") float sweeprange, @Cast("l_float32") float sweepdelta, @Cast("l_float32") float minbsdelta );
public static native @Cast("l_int32") int pixFindSkewSweepAndSearch( PIX pixs, @Cast("l_float32*") float[] pangle, @Cast("l_float32*") float[] pconf, @Cast("l_int32") int redsweep, @Cast("l_int32") int redsearch, @Cast("l_float32") float sweeprange, @Cast("l_float32") float sweepdelta, @Cast("l_float32") float minbsdelta );
public static native @Cast("l_int32") int pixFindSkewSweepAndSearchScore( PIX pixs, @Cast("l_float32*") FloatPointer pangle, @Cast("l_float32*") FloatPointer pconf, @Cast("l_float32*") FloatPointer pendscore, @Cast("l_int32") int redsweep, @Cast("l_int32") int redsearch, @Cast("l_float32") float sweepcenter, @Cast("l_float32") float sweeprange, @Cast("l_float32") float sweepdelta, @Cast("l_float32") float minbsdelta );
public static native @Cast("l_int32") int pixFindSkewSweepAndSearchScore( PIX pixs, @Cast("l_float32*") FloatBuffer pangle, @Cast("l_float32*") FloatBuffer pconf, @Cast("l_float32*") FloatBuffer pendscore, @Cast("l_int32") int redsweep, @Cast("l_int32") int redsearch, @Cast("l_float32") float sweepcenter, @Cast("l_float32") float sweeprange, @Cast("l_float32") float sweepdelta, @Cast("l_float32") float minbsdelta );
public static native @Cast("l_int32") int pixFindSkewSweepAndSearchScore( PIX pixs, @Cast("l_float32*") float[] pangle, @Cast("l_float32*") float[] pconf, @Cast("l_float32*") float[] pendscore, @Cast("l_int32") int redsweep, @Cast("l_int32") int redsearch, @Cast("l_float32") float sweepcenter, @Cast("l_float32") float sweeprange, @Cast("l_float32") float sweepdelta, @Cast("l_float32") float minbsdelta );
public static native @Cast("l_int32") int pixFindSkewSweepAndSearchScorePivot( PIX pixs, @Cast("l_float32*") FloatPointer pangle, @Cast("l_float32*") FloatPointer pconf, @Cast("l_float32*") FloatPointer pendscore, @Cast("l_int32") int redsweep, @Cast("l_int32") int redsearch, @Cast("l_float32") float sweepcenter, @Cast("l_float32") float sweeprange, @Cast("l_float32") float sweepdelta, @Cast("l_float32") float minbsdelta, @Cast("l_int32") int pivot );
public static native @Cast("l_int32") int pixFindSkewSweepAndSearchScorePivot( PIX pixs, @Cast("l_float32*") FloatBuffer pangle, @Cast("l_float32*") FloatBuffer pconf, @Cast("l_float32*") FloatBuffer pendscore, @Cast("l_int32") int redsweep, @Cast("l_int32") int redsearch, @Cast("l_float32") float sweepcenter, @Cast("l_float32") float sweeprange, @Cast("l_float32") float sweepdelta, @Cast("l_float32") float minbsdelta, @Cast("l_int32") int pivot );
public static native @Cast("l_int32") int pixFindSkewSweepAndSearchScorePivot( PIX pixs, @Cast("l_float32*") float[] pangle, @Cast("l_float32*") float[] pconf, @Cast("l_float32*") float[] pendscore, @Cast("l_int32") int redsweep, @Cast("l_int32") int redsearch, @Cast("l_float32") float sweepcenter, @Cast("l_float32") float sweeprange, @Cast("l_float32") float sweepdelta, @Cast("l_float32") float minbsdelta, @Cast("l_int32") int pivot );
public static native @Cast("l_int32") int pixFindSkewOrthogonalRange( PIX pixs, @Cast("l_float32*") FloatPointer pangle, @Cast("l_float32*") FloatPointer pconf, @Cast("l_int32") int redsweep, @Cast("l_int32") int redsearch, @Cast("l_float32") float sweeprange, @Cast("l_float32") float sweepdelta, @Cast("l_float32") float minbsdelta, @Cast("l_float32") float confprior );
public static native @Cast("l_int32") int pixFindSkewOrthogonalRange( PIX pixs, @Cast("l_float32*") FloatBuffer pangle, @Cast("l_float32*") FloatBuffer pconf, @Cast("l_int32") int redsweep, @Cast("l_int32") int redsearch, @Cast("l_float32") float sweeprange, @Cast("l_float32") float sweepdelta, @Cast("l_float32") float minbsdelta, @Cast("l_float32") float confprior );
public static native @Cast("l_int32") int pixFindSkewOrthogonalRange( PIX pixs, @Cast("l_float32*") float[] pangle, @Cast("l_float32*") float[] pconf, @Cast("l_int32") int redsweep, @Cast("l_int32") int redsearch, @Cast("l_float32") float sweeprange, @Cast("l_float32") float sweepdelta, @Cast("l_float32") float minbsdelta, @Cast("l_float32") float confprior );
public static native @Cast("l_int32") int pixFindDifferentialSquareSum( PIX pixs, @Cast("l_float32*") FloatPointer psum );
public static native @Cast("l_int32") int pixFindDifferentialSquareSum( PIX pixs, @Cast("l_float32*") FloatBuffer psum );
public static native @Cast("l_int32") int pixFindDifferentialSquareSum( PIX pixs, @Cast("l_float32*") float[] psum );
public static native @Cast("l_int32") int pixFindNormalizedSquareSum( PIX pixs, @Cast("l_float32*") FloatPointer phratio, @Cast("l_float32*") FloatPointer pvratio, @Cast("l_float32*") FloatPointer pfract );
public static native @Cast("l_int32") int pixFindNormalizedSquareSum( PIX pixs, @Cast("l_float32*") FloatBuffer phratio, @Cast("l_float32*") FloatBuffer pvratio, @Cast("l_float32*") FloatBuffer pfract );
public static native @Cast("l_int32") int pixFindNormalizedSquareSum( PIX pixs, @Cast("l_float32*") float[] phratio, @Cast("l_float32*") float[] pvratio, @Cast("l_float32*") float[] pfract );
public static native PIX pixReadStreamSpix( @Cast("FILE*") Pointer fp );
public static native @Cast("l_int32") int readHeaderSpix( @Cast("const char*") BytePointer filename, @Cast("l_int32*") IntPointer pwidth, @Cast("l_int32*") IntPointer pheight, @Cast("l_int32*") IntPointer pbps, @Cast("l_int32*") IntPointer pspp, @Cast("l_int32*") IntPointer piscmap );
public static native @Cast("l_int32") int readHeaderSpix( String filename, @Cast("l_int32*") IntBuffer pwidth, @Cast("l_int32*") IntBuffer pheight, @Cast("l_int32*") IntBuffer pbps, @Cast("l_int32*") IntBuffer pspp, @Cast("l_int32*") IntBuffer piscmap );
public static native @Cast("l_int32") int readHeaderSpix( @Cast("const char*") BytePointer filename, @Cast("l_int32*") int[] pwidth, @Cast("l_int32*") int[] pheight, @Cast("l_int32*") int[] pbps, @Cast("l_int32*") int[] pspp, @Cast("l_int32*") int[] piscmap );
public static native @Cast("l_int32") int readHeaderSpix( String filename, @Cast("l_int32*") IntPointer pwidth, @Cast("l_int32*") IntPointer pheight, @Cast("l_int32*") IntPointer pbps, @Cast("l_int32*") IntPointer pspp, @Cast("l_int32*") IntPointer piscmap );
public static native @Cast("l_int32") int readHeaderSpix( @Cast("const char*") BytePointer filename, @Cast("l_int32*") IntBuffer pwidth, @Cast("l_int32*") IntBuffer pheight, @Cast("l_int32*") IntBuffer pbps, @Cast("l_int32*") IntBuffer pspp, @Cast("l_int32*") IntBuffer piscmap );
public static native @Cast("l_int32") int readHeaderSpix( String filename, @Cast("l_int32*") int[] pwidth, @Cast("l_int32*") int[] pheight, @Cast("l_int32*") int[] pbps, @Cast("l_int32*") int[] pspp, @Cast("l_int32*") int[] piscmap );
public static native @Cast("l_int32") int freadHeaderSpix( @Cast("FILE*") Pointer fp, @Cast("l_int32*") IntPointer pwidth, @Cast("l_int32*") IntPointer pheight, @Cast("l_int32*") IntPointer pbps, @Cast("l_int32*") IntPointer pspp, @Cast("l_int32*") IntPointer piscmap );
public static native @Cast("l_int32") int freadHeaderSpix( @Cast("FILE*") Pointer fp, @Cast("l_int32*") IntBuffer pwidth, @Cast("l_int32*") IntBuffer pheight, @Cast("l_int32*") IntBuffer pbps, @Cast("l_int32*") IntBuffer pspp, @Cast("l_int32*") IntBuffer piscmap );
public static native @Cast("l_int32") int freadHeaderSpix( @Cast("FILE*") Pointer fp, @Cast("l_int32*") int[] pwidth, @Cast("l_int32*") int[] pheight, @Cast("l_int32*") int[] pbps, @Cast("l_int32*") int[] pspp, @Cast("l_int32*") int[] piscmap );
public static native @Cast("l_int32") int sreadHeaderSpix( @Cast("const l_uint32*") IntPointer data, @Cast("l_int32*") IntPointer pwidth, @Cast("l_int32*") IntPointer pheight, @Cast("l_int32*") IntPointer pbps, @Cast("l_int32*") IntPointer pspp, @Cast("l_int32*") IntPointer piscmap );
public static native @Cast("l_int32") int sreadHeaderSpix( @Cast("const l_uint32*") IntBuffer data, @Cast("l_int32*") IntBuffer pwidth, @Cast("l_int32*") IntBuffer pheight, @Cast("l_int32*") IntBuffer pbps, @Cast("l_int32*") IntBuffer pspp, @Cast("l_int32*") IntBuffer piscmap );
public static native @Cast("l_int32") int sreadHeaderSpix( @Cast("const l_uint32*") int[] data, @Cast("l_int32*") int[] pwidth, @Cast("l_int32*") int[] pheight, @Cast("l_int32*") int[] pbps, @Cast("l_int32*") int[] pspp, @Cast("l_int32*") int[] piscmap );
public static native @Cast("l_int32") int pixWriteStreamSpix( @Cast("FILE*") Pointer fp, PIX pix );
public static native PIX pixReadMemSpix( @Cast("const l_uint8*") BytePointer data, @Cast("size_t") long size );
public static native PIX pixReadMemSpix( @Cast("const l_uint8*") ByteBuffer data, @Cast("size_t") long size );
public static native PIX pixReadMemSpix( @Cast("const l_uint8*") byte[] data, @Cast("size_t") long size );
public static native @Cast("l_int32") int pixWriteMemSpix( @Cast("l_uint8**") PointerPointer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix );
public static native @Cast("l_int32") int pixWriteMemSpix( @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix );
public static native @Cast("l_int32") int pixWriteMemSpix( @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix );
public static native @Cast("l_int32") int pixWriteMemSpix( @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer psize, PIX pix );
public static native @Cast("l_int32") int pixSerializeToMemory( PIX pixs, @Cast("l_uint32**") PointerPointer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int pixSerializeToMemory( PIX pixs, @Cast("l_uint32**") @ByPtrPtr IntPointer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int pixSerializeToMemory( PIX pixs, @Cast("l_uint32**") @ByPtrPtr IntBuffer pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_int32") int pixSerializeToMemory( PIX pixs, @Cast("l_uint32**") @ByPtrPtr int[] pdata, @Cast("size_t*") SizeTPointer pnbytes );
public static native PIX pixDeserializeFromMemory( @Cast("const l_uint32*") IntPointer data, @Cast("size_t") long nbytes );
public static native PIX pixDeserializeFromMemory( @Cast("const l_uint32*") IntBuffer data, @Cast("size_t") long nbytes );
public static native PIX pixDeserializeFromMemory( @Cast("const l_uint32*") int[] data, @Cast("size_t") long nbytes );
public static native L_STACK lstackCreate( @Cast("l_int32") int nalloc );
public static native void lstackDestroy( @Cast("L_STACK**") PointerPointer plstack, @Cast("l_int32") int freeflag );
public static native void lstackDestroy( @ByPtrPtr L_STACK plstack, @Cast("l_int32") int freeflag );
public static native @Cast("l_int32") int lstackAdd( L_STACK lstack, Pointer item );
public static native Pointer lstackRemove( L_STACK lstack );
public static native @Cast("l_int32") int lstackGetCount( L_STACK lstack );
public static native @Cast("l_int32") int lstackPrint( @Cast("FILE*") Pointer fp, L_STACK lstack );
public static native L_STRCODE strcodeCreate( @Cast("l_int32") int fileno );
public static native @Cast("l_int32") int strcodeCreateFromFile( @Cast("const char*") BytePointer filein, @Cast("l_int32") int fileno, @Cast("const char*") BytePointer outdir );
public static native @Cast("l_int32") int strcodeCreateFromFile( String filein, @Cast("l_int32") int fileno, String outdir );
public static native @Cast("l_int32") int strcodeGenerate( L_STRCODE strcode, @Cast("const char*") BytePointer filein, @Cast("const char*") BytePointer type );
public static native @Cast("l_int32") int strcodeGenerate( L_STRCODE strcode, String filein, String type );
public static native @Cast("l_int32") int strcodeFinalize( @Cast("L_STRCODE**") PointerPointer pstrcode, @Cast("const char*") BytePointer outdir );
public static native @Cast("l_int32") int strcodeFinalize( @ByPtrPtr L_STRCODE pstrcode, @Cast("const char*") BytePointer outdir );
public static native @Cast("l_int32") int strcodeFinalize( @ByPtrPtr L_STRCODE pstrcode, String outdir );
public static native @Cast("l_int32*") IntPointer sudokuReadFile( @Cast("const char*") BytePointer filename );
public static native @Cast("l_int32*") IntBuffer sudokuReadFile( String filename );
public static native @Cast("l_int32*") IntPointer sudokuReadString( @Cast("const char*") BytePointer str );
public static native @Cast("l_int32*") IntBuffer sudokuReadString( String str );
public static native L_SUDOKU sudokuCreate( @Cast("l_int32*") IntPointer array );
public static native L_SUDOKU sudokuCreate( @Cast("l_int32*") IntBuffer array );
public static native L_SUDOKU sudokuCreate( @Cast("l_int32*") int[] array );
public static native void sudokuDestroy( @Cast("L_SUDOKU**") PointerPointer psud );
public static native void sudokuDestroy( @ByPtrPtr L_SUDOKU psud );
public static native @Cast("l_int32") int sudokuSolve( L_SUDOKU sud );
public static native @Cast("l_int32") int sudokuTestUniqueness( @Cast("l_int32*") IntPointer array, @Cast("l_int32*") IntPointer punique );
public static native @Cast("l_int32") int sudokuTestUniqueness( @Cast("l_int32*") IntBuffer array, @Cast("l_int32*") IntBuffer punique );
public static native @Cast("l_int32") int sudokuTestUniqueness( @Cast("l_int32*") int[] array, @Cast("l_int32*") int[] punique );
public static native L_SUDOKU sudokuGenerate( @Cast("l_int32*") IntPointer array, @Cast("l_int32") int seed, @Cast("l_int32") int minelems, @Cast("l_int32") int maxtries );
public static native L_SUDOKU sudokuGenerate( @Cast("l_int32*") IntBuffer array, @Cast("l_int32") int seed, @Cast("l_int32") int minelems, @Cast("l_int32") int maxtries );
public static native L_SUDOKU sudokuGenerate( @Cast("l_int32*") int[] array, @Cast("l_int32") int seed, @Cast("l_int32") int minelems, @Cast("l_int32") int maxtries );
public static native @Cast("l_int32") int sudokuOutput( L_SUDOKU sud, @Cast("l_int32") int arraytype );
public static native PIX pixAddSingleTextblock( PIX pixs, L_BMF bmf, @Cast("const char*") BytePointer textstr, @Cast("l_uint32") int val, @Cast("l_int32") int location, @Cast("l_int32*") IntPointer poverflow );
public static native PIX pixAddSingleTextblock( PIX pixs, L_BMF bmf, String textstr, @Cast("l_uint32") int val, @Cast("l_int32") int location, @Cast("l_int32*") IntBuffer poverflow );
public static native PIX pixAddSingleTextblock( PIX pixs, L_BMF bmf, @Cast("const char*") BytePointer textstr, @Cast("l_uint32") int val, @Cast("l_int32") int location, @Cast("l_int32*") int[] poverflow );
public static native PIX pixAddSingleTextblock( PIX pixs, L_BMF bmf, String textstr, @Cast("l_uint32") int val, @Cast("l_int32") int location, @Cast("l_int32*") IntPointer poverflow );
public static native PIX pixAddSingleTextblock( PIX pixs, L_BMF bmf, @Cast("const char*") BytePointer textstr, @Cast("l_uint32") int val, @Cast("l_int32") int location, @Cast("l_int32*") IntBuffer poverflow );
public static native PIX pixAddSingleTextblock( PIX pixs, L_BMF bmf, String textstr, @Cast("l_uint32") int val, @Cast("l_int32") int location, @Cast("l_int32*") int[] poverflow );
public static native PIX pixAddSingleTextline( PIX pixs, L_BMF bmf, @Cast("const char*") BytePointer textstr, @Cast("l_uint32") int val, @Cast("l_int32") int location );
public static native PIX pixAddSingleTextline( PIX pixs, L_BMF bmf, String textstr, @Cast("l_uint32") int val, @Cast("l_int32") int location );
public static native @Cast("l_int32") int pixSetTextblock( PIX pixs, L_BMF bmf, @Cast("const char*") BytePointer textstr, @Cast("l_uint32") int val, @Cast("l_int32") int x0, @Cast("l_int32") int y0, @Cast("l_int32") int wtext, @Cast("l_int32") int firstindent, @Cast("l_int32*") IntPointer poverflow );
public static native @Cast("l_int32") int pixSetTextblock( PIX pixs, L_BMF bmf, String textstr, @Cast("l_uint32") int val, @Cast("l_int32") int x0, @Cast("l_int32") int y0, @Cast("l_int32") int wtext, @Cast("l_int32") int firstindent, @Cast("l_int32*") IntBuffer poverflow );
public static native @Cast("l_int32") int pixSetTextblock( PIX pixs, L_BMF bmf, @Cast("const char*") BytePointer textstr, @Cast("l_uint32") int val, @Cast("l_int32") int x0, @Cast("l_int32") int y0, @Cast("l_int32") int wtext, @Cast("l_int32") int firstindent, @Cast("l_int32*") int[] poverflow );
public static native @Cast("l_int32") int pixSetTextblock( PIX pixs, L_BMF bmf, String textstr, @Cast("l_uint32") int val, @Cast("l_int32") int x0, @Cast("l_int32") int y0, @Cast("l_int32") int wtext, @Cast("l_int32") int firstindent, @Cast("l_int32*") IntPointer poverflow );
public static native @Cast("l_int32") int pixSetTextblock( PIX pixs, L_BMF bmf, @Cast("const char*") BytePointer textstr, @Cast("l_uint32") int val, @Cast("l_int32") int x0, @Cast("l_int32") int y0, @Cast("l_int32") int wtext, @Cast("l_int32") int firstindent, @Cast("l_int32*") IntBuffer poverflow );
public static native @Cast("l_int32") int pixSetTextblock( PIX pixs, L_BMF bmf, String textstr, @Cast("l_uint32") int val, @Cast("l_int32") int x0, @Cast("l_int32") int y0, @Cast("l_int32") int wtext, @Cast("l_int32") int firstindent, @Cast("l_int32*") int[] poverflow );
public static native @Cast("l_int32") int pixSetTextline( PIX pixs, L_BMF bmf, @Cast("const char*") BytePointer textstr, @Cast("l_uint32") int val, @Cast("l_int32") int x0, @Cast("l_int32") int y0, @Cast("l_int32*") IntPointer pwidth, @Cast("l_int32*") IntPointer poverflow );
public static native @Cast("l_int32") int pixSetTextline( PIX pixs, L_BMF bmf, String textstr, @Cast("l_uint32") int val, @Cast("l_int32") int x0, @Cast("l_int32") int y0, @Cast("l_int32*") IntBuffer pwidth, @Cast("l_int32*") IntBuffer poverflow );
public static native @Cast("l_int32") int pixSetTextline( PIX pixs, L_BMF bmf, @Cast("const char*") BytePointer textstr, @Cast("l_uint32") int val, @Cast("l_int32") int x0, @Cast("l_int32") int y0, @Cast("l_int32*") int[] pwidth, @Cast("l_int32*") int[] poverflow );
public static native @Cast("l_int32") int pixSetTextline( PIX pixs, L_BMF bmf, String textstr, @Cast("l_uint32") int val, @Cast("l_int32") int x0, @Cast("l_int32") int y0, @Cast("l_int32*") IntPointer pwidth, @Cast("l_int32*") IntPointer poverflow );
public static native @Cast("l_int32") int pixSetTextline( PIX pixs, L_BMF bmf, @Cast("const char*") BytePointer textstr, @Cast("l_uint32") int val, @Cast("l_int32") int x0, @Cast("l_int32") int y0, @Cast("l_int32*") IntBuffer pwidth, @Cast("l_int32*") IntBuffer poverflow );
public static native @Cast("l_int32") int pixSetTextline( PIX pixs, L_BMF bmf, String textstr, @Cast("l_uint32") int val, @Cast("l_int32") int x0, @Cast("l_int32") int y0, @Cast("l_int32*") int[] pwidth, @Cast("l_int32*") int[] poverflow );
public static native PIXA pixaAddTextNumber( PIXA pixas, L_BMF bmf, NUMA na, @Cast("l_uint32") int val, @Cast("l_int32") int location );
public static native PIXA pixaAddTextline( PIXA pixas, L_BMF bmf, SARRAY sa, @Cast("l_uint32") int val, @Cast("l_int32") int location );
public static native SARRAY bmfGetLineStrings( L_BMF bmf, @Cast("const char*") BytePointer textstr, @Cast("l_int32") int maxw, @Cast("l_int32") int firstindent, @Cast("l_int32*") IntPointer ph );
public static native SARRAY bmfGetLineStrings( L_BMF bmf, String textstr, @Cast("l_int32") int maxw, @Cast("l_int32") int firstindent, @Cast("l_int32*") IntBuffer ph );
public static native SARRAY bmfGetLineStrings( L_BMF bmf, @Cast("const char*") BytePointer textstr, @Cast("l_int32") int maxw, @Cast("l_int32") int firstindent, @Cast("l_int32*") int[] ph );
public static native SARRAY bmfGetLineStrings( L_BMF bmf, String textstr, @Cast("l_int32") int maxw, @Cast("l_int32") int firstindent, @Cast("l_int32*") IntPointer ph );
public static native SARRAY bmfGetLineStrings( L_BMF bmf, @Cast("const char*") BytePointer textstr, @Cast("l_int32") int maxw, @Cast("l_int32") int firstindent, @Cast("l_int32*") IntBuffer ph );
public static native SARRAY bmfGetLineStrings( L_BMF bmf, String textstr, @Cast("l_int32") int maxw, @Cast("l_int32") int firstindent, @Cast("l_int32*") int[] ph );
public static native NUMA bmfGetWordWidths( L_BMF bmf, @Cast("const char*") BytePointer textstr, SARRAY sa );
public static native NUMA bmfGetWordWidths( L_BMF bmf, String textstr, SARRAY sa );
public static native @Cast("l_int32") int bmfGetStringWidth( L_BMF bmf, @Cast("const char*") BytePointer textstr, @Cast("l_int32*") IntPointer pw );
public static native @Cast("l_int32") int bmfGetStringWidth( L_BMF bmf, String textstr, @Cast("l_int32*") IntBuffer pw );
public static native @Cast("l_int32") int bmfGetStringWidth( L_BMF bmf, @Cast("const char*") BytePointer textstr, @Cast("l_int32*") int[] pw );
public static native @Cast("l_int32") int bmfGetStringWidth( L_BMF bmf, String textstr, @Cast("l_int32*") IntPointer pw );
public static native @Cast("l_int32") int bmfGetStringWidth( L_BMF bmf, @Cast("const char*") BytePointer textstr, @Cast("l_int32*") IntBuffer pw );
public static native @Cast("l_int32") int bmfGetStringWidth( L_BMF bmf, String textstr, @Cast("l_int32*") int[] pw );
public static native SARRAY splitStringToParagraphs( @Cast("char*") BytePointer textstr, @Cast("l_int32") int splitflag );
public static native SARRAY splitStringToParagraphs( @Cast("char*") ByteBuffer textstr, @Cast("l_int32") int splitflag );
public static native SARRAY splitStringToParagraphs( @Cast("char*") byte[] textstr, @Cast("l_int32") int splitflag );
public static native PIX pixReadTiff( @Cast("const char*") BytePointer filename, @Cast("l_int32") int n );
public static native PIX pixReadTiff( String filename, @Cast("l_int32") int n );
public static native PIX pixReadStreamTiff( @Cast("FILE*") Pointer fp, @Cast("l_int32") int n );
public static native @Cast("l_int32") int pixWriteTiff( @Cast("const char*") BytePointer filename, PIX pix, @Cast("l_int32") int comptype, @Cast("const char*") BytePointer modestring );
public static native @Cast("l_int32") int pixWriteTiff( String filename, PIX pix, @Cast("l_int32") int comptype, String modestring );
public static native @Cast("l_int32") int pixWriteTiffCustom( @Cast("const char*") BytePointer filename, PIX pix, @Cast("l_int32") int comptype, @Cast("const char*") BytePointer modestring, NUMA natags, SARRAY savals, SARRAY satypes, NUMA nasizes );
public static native @Cast("l_int32") int pixWriteTiffCustom( String filename, PIX pix, @Cast("l_int32") int comptype, String modestring, NUMA natags, SARRAY savals, SARRAY satypes, NUMA nasizes );
public static native @Cast("l_int32") int pixWriteStreamTiff( @Cast("FILE*") Pointer fp, PIX pix, @Cast("l_int32") int comptype );
public static native PIXA pixaReadMultipageTiff( @Cast("const char*") BytePointer filename );
public static native PIXA pixaReadMultipageTiff( String filename );
public static native @Cast("l_int32") int writeMultipageTiff( @Cast("const char*") BytePointer dirin, @Cast("const char*") BytePointer substr, @Cast("const char*") BytePointer fileout );
public static native @Cast("l_int32") int writeMultipageTiff( String dirin, String substr, String fileout );
public static native @Cast("l_int32") int writeMultipageTiffSA( SARRAY sa, @Cast("const char*") BytePointer fileout );
public static native @Cast("l_int32") int writeMultipageTiffSA( SARRAY sa, String fileout );
public static native @Cast("l_int32") int fprintTiffInfo( @Cast("FILE*") Pointer fpout, @Cast("const char*") BytePointer tiffile );
public static native @Cast("l_int32") int fprintTiffInfo( @Cast("FILE*") Pointer fpout, String tiffile );
public static native @Cast("l_int32") int tiffGetCount( @Cast("FILE*") Pointer fp, @Cast("l_int32*") IntPointer pn );
public static native @Cast("l_int32") int tiffGetCount( @Cast("FILE*") Pointer fp, @Cast("l_int32*") IntBuffer pn );
public static native @Cast("l_int32") int tiffGetCount( @Cast("FILE*") Pointer fp, @Cast("l_int32*") int[] pn );
public static native @Cast("l_int32") int getTiffResolution( @Cast("FILE*") Pointer fp, @Cast("l_int32*") IntPointer pxres, @Cast("l_int32*") IntPointer pyres );
public static native @Cast("l_int32") int getTiffResolution( @Cast("FILE*") Pointer fp, @Cast("l_int32*") IntBuffer pxres, @Cast("l_int32*") IntBuffer pyres );
public static native @Cast("l_int32") int getTiffResolution( @Cast("FILE*") Pointer fp, @Cast("l_int32*") int[] pxres, @Cast("l_int32*") int[] pyres );
public static native @Cast("l_int32") int readHeaderTiff( @Cast("const char*") BytePointer filename, @Cast("l_int32") int n, @Cast("l_int32*") IntPointer pwidth, @Cast("l_int32*") IntPointer pheight, @Cast("l_int32*") IntPointer pbps, @Cast("l_int32*") IntPointer pspp, @Cast("l_int32*") IntPointer pres, @Cast("l_int32*") IntPointer pcmap, @Cast("l_int32*") IntPointer pformat );
public static native @Cast("l_int32") int readHeaderTiff( String filename, @Cast("l_int32") int n, @Cast("l_int32*") IntBuffer pwidth, @Cast("l_int32*") IntBuffer pheight, @Cast("l_int32*") IntBuffer pbps, @Cast("l_int32*") IntBuffer pspp, @Cast("l_int32*") IntBuffer pres, @Cast("l_int32*") IntBuffer pcmap, @Cast("l_int32*") IntBuffer pformat );
public static native @Cast("l_int32") int readHeaderTiff( @Cast("const char*") BytePointer filename, @Cast("l_int32") int n, @Cast("l_int32*") int[] pwidth, @Cast("l_int32*") int[] pheight, @Cast("l_int32*") int[] pbps, @Cast("l_int32*") int[] pspp, @Cast("l_int32*") int[] pres, @Cast("l_int32*") int[] pcmap, @Cast("l_int32*") int[] pformat );
public static native @Cast("l_int32") int readHeaderTiff( String filename, @Cast("l_int32") int n, @Cast("l_int32*") IntPointer pwidth, @Cast("l_int32*") IntPointer pheight, @Cast("l_int32*") IntPointer pbps, @Cast("l_int32*") IntPointer pspp, @Cast("l_int32*") IntPointer pres, @Cast("l_int32*") IntPointer pcmap, @Cast("l_int32*") IntPointer pformat );
public static native @Cast("l_int32") int readHeaderTiff( @Cast("const char*") BytePointer filename, @Cast("l_int32") int n, @Cast("l_int32*") IntBuffer pwidth, @Cast("l_int32*") IntBuffer pheight, @Cast("l_int32*") IntBuffer pbps, @Cast("l_int32*") IntBuffer pspp, @Cast("l_int32*") IntBuffer pres, @Cast("l_int32*") IntBuffer pcmap, @Cast("l_int32*") IntBuffer pformat );
public static native @Cast("l_int32") int readHeaderTiff( String filename, @Cast("l_int32") int n, @Cast("l_int32*") int[] pwidth, @Cast("l_int32*") int[] pheight, @Cast("l_int32*") int[] pbps, @Cast("l_int32*") int[] pspp, @Cast("l_int32*") int[] pres, @Cast("l_int32*") int[] pcmap, @Cast("l_int32*") int[] pformat );
public static native @Cast("l_int32") int freadHeaderTiff( @Cast("FILE*") Pointer fp, @Cast("l_int32") int n, @Cast("l_int32*") IntPointer pwidth, @Cast("l_int32*") IntPointer pheight, @Cast("l_int32*") IntPointer pbps, @Cast("l_int32*") IntPointer pspp, @Cast("l_int32*") IntPointer pres, @Cast("l_int32*") IntPointer pcmap, @Cast("l_int32*") IntPointer pformat );
public static native @Cast("l_int32") int freadHeaderTiff( @Cast("FILE*") Pointer fp, @Cast("l_int32") int n, @Cast("l_int32*") IntBuffer pwidth, @Cast("l_int32*") IntBuffer pheight, @Cast("l_int32*") IntBuffer pbps, @Cast("l_int32*") IntBuffer pspp, @Cast("l_int32*") IntBuffer pres, @Cast("l_int32*") IntBuffer pcmap, @Cast("l_int32*") IntBuffer pformat );
public static native @Cast("l_int32") int freadHeaderTiff( @Cast("FILE*") Pointer fp, @Cast("l_int32") int n, @Cast("l_int32*") int[] pwidth, @Cast("l_int32*") int[] pheight, @Cast("l_int32*") int[] pbps, @Cast("l_int32*") int[] pspp, @Cast("l_int32*") int[] pres, @Cast("l_int32*") int[] pcmap, @Cast("l_int32*") int[] pformat );
public static native @Cast("l_int32") int readHeaderMemTiff( @Cast("const l_uint8*") BytePointer cdata, @Cast("size_t") long size, @Cast("l_int32") int n, @Cast("l_int32*") IntPointer pwidth, @Cast("l_int32*") IntPointer pheight, @Cast("l_int32*") IntPointer pbps, @Cast("l_int32*") IntPointer pspp, @Cast("l_int32*") IntPointer pres, @Cast("l_int32*") IntPointer pcmap, @Cast("l_int32*") IntPointer pformat );
public static native @Cast("l_int32") int readHeaderMemTiff( @Cast("const l_uint8*") ByteBuffer cdata, @Cast("size_t") long size, @Cast("l_int32") int n, @Cast("l_int32*") IntBuffer pwidth, @Cast("l_int32*") IntBuffer pheight, @Cast("l_int32*") IntBuffer pbps, @Cast("l_int32*") IntBuffer pspp, @Cast("l_int32*") IntBuffer pres, @Cast("l_int32*") IntBuffer pcmap, @Cast("l_int32*") IntBuffer pformat );
public static native @Cast("l_int32") int readHeaderMemTiff( @Cast("const l_uint8*") byte[] cdata, @Cast("size_t") long size, @Cast("l_int32") int n, @Cast("l_int32*") int[] pwidth, @Cast("l_int32*") int[] pheight, @Cast("l_int32*") int[] pbps, @Cast("l_int32*") int[] pspp, @Cast("l_int32*") int[] pres, @Cast("l_int32*") int[] pcmap, @Cast("l_int32*") int[] pformat );
public static native @Cast("l_int32") int findTiffCompression( @Cast("FILE*") Pointer fp, @Cast("l_int32*") IntPointer pcomptype );
public static native @Cast("l_int32") int findTiffCompression( @Cast("FILE*") Pointer fp, @Cast("l_int32*") IntBuffer pcomptype );
public static native @Cast("l_int32") int findTiffCompression( @Cast("FILE*") Pointer fp, @Cast("l_int32*") int[] pcomptype );
public static native @Cast("l_int32") int extractG4DataFromFile( @Cast("const char*") BytePointer filein, @Cast("l_uint8**") PointerPointer pdata, @Cast("size_t*") SizeTPointer pnbytes, @Cast("l_int32*") IntPointer pw, @Cast("l_int32*") IntPointer ph, @Cast("l_int32*") IntPointer pminisblack );
public static native @Cast("l_int32") int extractG4DataFromFile( @Cast("const char*") BytePointer filein, @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer pnbytes, @Cast("l_int32*") IntPointer pw, @Cast("l_int32*") IntPointer ph, @Cast("l_int32*") IntPointer pminisblack );
public static native @Cast("l_int32") int extractG4DataFromFile( String filein, @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer pnbytes, @Cast("l_int32*") IntBuffer pw, @Cast("l_int32*") IntBuffer ph, @Cast("l_int32*") IntBuffer pminisblack );
public static native @Cast("l_int32") int extractG4DataFromFile( @Cast("const char*") BytePointer filein, @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer pnbytes, @Cast("l_int32*") int[] pw, @Cast("l_int32*") int[] ph, @Cast("l_int32*") int[] pminisblack );
public static native @Cast("l_int32") int extractG4DataFromFile( String filein, @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer pnbytes, @Cast("l_int32*") IntPointer pw, @Cast("l_int32*") IntPointer ph, @Cast("l_int32*") IntPointer pminisblack );
public static native @Cast("l_int32") int extractG4DataFromFile( @Cast("const char*") BytePointer filein, @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer pnbytes, @Cast("l_int32*") IntBuffer pw, @Cast("l_int32*") IntBuffer ph, @Cast("l_int32*") IntBuffer pminisblack );
public static native @Cast("l_int32") int extractG4DataFromFile( String filein, @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer pnbytes, @Cast("l_int32*") int[] pw, @Cast("l_int32*") int[] ph, @Cast("l_int32*") int[] pminisblack );
public static native PIX pixReadMemTiff( @Cast("const l_uint8*") BytePointer cdata, @Cast("size_t") long size, @Cast("l_int32") int n );
public static native PIX pixReadMemTiff( @Cast("const l_uint8*") ByteBuffer cdata, @Cast("size_t") long size, @Cast("l_int32") int n );
public static native PIX pixReadMemTiff( @Cast("const l_uint8*") byte[] cdata, @Cast("size_t") long size, @Cast("l_int32") int n );
public static native @Cast("l_int32") int pixWriteMemTiff( @Cast("l_uint8**") PointerPointer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix, @Cast("l_int32") int comptype );
public static native @Cast("l_int32") int pixWriteMemTiff( @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix, @Cast("l_int32") int comptype );
public static native @Cast("l_int32") int pixWriteMemTiff( @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix, @Cast("l_int32") int comptype );
public static native @Cast("l_int32") int pixWriteMemTiff( @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer psize, PIX pix, @Cast("l_int32") int comptype );
public static native @Cast("l_int32") int pixWriteMemTiffCustom( @Cast("l_uint8**") PointerPointer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix, @Cast("l_int32") int comptype, NUMA natags, SARRAY savals, SARRAY satypes, NUMA nasizes );
public static native @Cast("l_int32") int pixWriteMemTiffCustom( @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix, @Cast("l_int32") int comptype, NUMA natags, SARRAY savals, SARRAY satypes, NUMA nasizes );
public static native @Cast("l_int32") int pixWriteMemTiffCustom( @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix, @Cast("l_int32") int comptype, NUMA natags, SARRAY savals, SARRAY satypes, NUMA nasizes );
public static native @Cast("l_int32") int pixWriteMemTiffCustom( @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer psize, PIX pix, @Cast("l_int32") int comptype, NUMA natags, SARRAY savals, SARRAY satypes, NUMA nasizes );
public static native @Cast("l_int32") int setMsgSeverity( @Cast("l_int32") int newsev );
public static native @Cast("l_int32") int returnErrorInt( @Cast("const char*") BytePointer msg, @Cast("const char*") BytePointer procname, @Cast("l_int32") int ival );
public static native @Cast("l_int32") int returnErrorInt( String msg, String procname, @Cast("l_int32") int ival );
public static native @Cast("l_float32") float returnErrorFloat( @Cast("const char*") BytePointer msg, @Cast("const char*") BytePointer procname, @Cast("l_float32") float fval );
public static native @Cast("l_float32") float returnErrorFloat( String msg, String procname, @Cast("l_float32") float fval );
public static native Pointer returnErrorPtr( @Cast("const char*") BytePointer msg, @Cast("const char*") BytePointer procname, Pointer pval );
public static native Pointer returnErrorPtr( String msg, String procname, Pointer pval );
public static native @Cast("char*") BytePointer stringNew( @Cast("const char*") BytePointer src );
public static native @Cast("char*") ByteBuffer stringNew( String src );
public static native @Cast("l_int32") int stringCopy( @Cast("char*") BytePointer dest, @Cast("const char*") BytePointer src, @Cast("l_int32") int n );
public static native @Cast("l_int32") int stringCopy( @Cast("char*") ByteBuffer dest, String src, @Cast("l_int32") int n );
public static native @Cast("l_int32") int stringCopy( @Cast("char*") byte[] dest, @Cast("const char*") BytePointer src, @Cast("l_int32") int n );
public static native @Cast("l_int32") int stringCopy( @Cast("char*") BytePointer dest, String src, @Cast("l_int32") int n );
public static native @Cast("l_int32") int stringCopy( @Cast("char*") ByteBuffer dest, @Cast("const char*") BytePointer src, @Cast("l_int32") int n );
public static native @Cast("l_int32") int stringCopy( @Cast("char*") byte[] dest, String src, @Cast("l_int32") int n );
public static native @Cast("l_int32") int stringReplace( @Cast("char**") PointerPointer pdest, @Cast("const char*") BytePointer src );
public static native @Cast("l_int32") int stringReplace( @Cast("char**") @ByPtrPtr BytePointer pdest, @Cast("const char*") BytePointer src );
public static native @Cast("l_int32") int stringReplace( @Cast("char**") @ByPtrPtr ByteBuffer pdest, String src );
public static native @Cast("l_int32") int stringReplace( @Cast("char**") @ByPtrPtr byte[] pdest, @Cast("const char*") BytePointer src );
public static native @Cast("l_int32") int stringReplace( @Cast("char**") @ByPtrPtr BytePointer pdest, String src );
public static native @Cast("l_int32") int stringReplace( @Cast("char**") @ByPtrPtr ByteBuffer pdest, @Cast("const char*") BytePointer src );
public static native @Cast("l_int32") int stringReplace( @Cast("char**") @ByPtrPtr byte[] pdest, String src );
public static native @Cast("l_int32") int stringLength( @Cast("const char*") BytePointer src, @Cast("size_t") long size );
public static native @Cast("l_int32") int stringLength( String src, @Cast("size_t") long size );
public static native @Cast("l_int32") int stringCat( @Cast("char*") BytePointer dest, @Cast("size_t") long size, @Cast("const char*") BytePointer src );
public static native @Cast("l_int32") int stringCat( @Cast("char*") ByteBuffer dest, @Cast("size_t") long size, String src );
public static native @Cast("l_int32") int stringCat( @Cast("char*") byte[] dest, @Cast("size_t") long size, @Cast("const char*") BytePointer src );
public static native @Cast("l_int32") int stringCat( @Cast("char*") BytePointer dest, @Cast("size_t") long size, String src );
public static native @Cast("l_int32") int stringCat( @Cast("char*") ByteBuffer dest, @Cast("size_t") long size, @Cast("const char*") BytePointer src );
public static native @Cast("l_int32") int stringCat( @Cast("char*") byte[] dest, @Cast("size_t") long size, String src );
public static native @Cast("char*") BytePointer stringConcatNew( @Cast("const char*") BytePointer first );
public static native @Cast("char*") ByteBuffer stringConcatNew( String first );
public static native @Cast("char*") BytePointer stringJoin( @Cast("const char*") BytePointer src1, @Cast("const char*") BytePointer src2 );
public static native @Cast("char*") ByteBuffer stringJoin( String src1, String src2 );
public static native @Cast("l_int32") int stringJoinIP( @Cast("char**") PointerPointer psrc1, @Cast("const char*") BytePointer src2 );
public static native @Cast("l_int32") int stringJoinIP( @Cast("char**") @ByPtrPtr BytePointer psrc1, @Cast("const char*") BytePointer src2 );
public static native @Cast("l_int32") int stringJoinIP( @Cast("char**") @ByPtrPtr ByteBuffer psrc1, String src2 );
public static native @Cast("l_int32") int stringJoinIP( @Cast("char**") @ByPtrPtr byte[] psrc1, @Cast("const char*") BytePointer src2 );
public static native @Cast("l_int32") int stringJoinIP( @Cast("char**") @ByPtrPtr BytePointer psrc1, String src2 );
public static native @Cast("l_int32") int stringJoinIP( @Cast("char**") @ByPtrPtr ByteBuffer psrc1, @Cast("const char*") BytePointer src2 );
public static native @Cast("l_int32") int stringJoinIP( @Cast("char**") @ByPtrPtr byte[] psrc1, String src2 );
public static native @Cast("char*") BytePointer stringReverse( @Cast("const char*") BytePointer src );
public static native @Cast("char*") ByteBuffer stringReverse( String src );
public static native @Cast("char*") BytePointer strtokSafe( @Cast("char*") BytePointer cstr, @Cast("const char*") BytePointer seps, @Cast("char**") PointerPointer psaveptr );
public static native @Cast("char*") BytePointer strtokSafe( @Cast("char*") BytePointer cstr, @Cast("const char*") BytePointer seps, @Cast("char**") @ByPtrPtr BytePointer psaveptr );
public static native @Cast("char*") ByteBuffer strtokSafe( @Cast("char*") ByteBuffer cstr, String seps, @Cast("char**") @ByPtrPtr ByteBuffer psaveptr );
public static native @Cast("char*") byte[] strtokSafe( @Cast("char*") byte[] cstr, @Cast("const char*") BytePointer seps, @Cast("char**") @ByPtrPtr byte[] psaveptr );
public static native @Cast("char*") BytePointer strtokSafe( @Cast("char*") BytePointer cstr, String seps, @Cast("char**") @ByPtrPtr BytePointer psaveptr );
public static native @Cast("char*") ByteBuffer strtokSafe( @Cast("char*") ByteBuffer cstr, @Cast("const char*") BytePointer seps, @Cast("char**") @ByPtrPtr ByteBuffer psaveptr );
public static native @Cast("char*") byte[] strtokSafe( @Cast("char*") byte[] cstr, String seps, @Cast("char**") @ByPtrPtr byte[] psaveptr );
public static native @Cast("l_int32") int stringSplitOnToken( @Cast("char*") BytePointer cstr, @Cast("const char*") BytePointer seps, @Cast("char**") PointerPointer phead, @Cast("char**") PointerPointer ptail );
public static native @Cast("l_int32") int stringSplitOnToken( @Cast("char*") BytePointer cstr, @Cast("const char*") BytePointer seps, @Cast("char**") @ByPtrPtr BytePointer phead, @Cast("char**") @ByPtrPtr BytePointer ptail );
public static native @Cast("l_int32") int stringSplitOnToken( @Cast("char*") ByteBuffer cstr, String seps, @Cast("char**") @ByPtrPtr ByteBuffer phead, @Cast("char**") @ByPtrPtr ByteBuffer ptail );
public static native @Cast("l_int32") int stringSplitOnToken( @Cast("char*") byte[] cstr, @Cast("const char*") BytePointer seps, @Cast("char**") @ByPtrPtr byte[] phead, @Cast("char**") @ByPtrPtr byte[] ptail );
public static native @Cast("l_int32") int stringSplitOnToken( @Cast("char*") BytePointer cstr, String seps, @Cast("char**") @ByPtrPtr BytePointer phead, @Cast("char**") @ByPtrPtr BytePointer ptail );
public static native @Cast("l_int32") int stringSplitOnToken( @Cast("char*") ByteBuffer cstr, @Cast("const char*") BytePointer seps, @Cast("char**") @ByPtrPtr ByteBuffer phead, @Cast("char**") @ByPtrPtr ByteBuffer ptail );
public static native @Cast("l_int32") int stringSplitOnToken( @Cast("char*") byte[] cstr, String seps, @Cast("char**") @ByPtrPtr byte[] phead, @Cast("char**") @ByPtrPtr byte[] ptail );
public static native @Cast("char*") BytePointer stringRemoveChars( @Cast("const char*") BytePointer src, @Cast("const char*") BytePointer remchars );
public static native @Cast("char*") ByteBuffer stringRemoveChars( String src, String remchars );
public static native @Cast("l_int32") int stringFindSubstr( @Cast("const char*") BytePointer src, @Cast("const char*") BytePointer sub, @Cast("l_int32*") IntPointer ploc );
public static native @Cast("l_int32") int stringFindSubstr( String src, String sub, @Cast("l_int32*") IntBuffer ploc );
public static native @Cast("l_int32") int stringFindSubstr( @Cast("const char*") BytePointer src, @Cast("const char*") BytePointer sub, @Cast("l_int32*") int[] ploc );
public static native @Cast("l_int32") int stringFindSubstr( String src, String sub, @Cast("l_int32*") IntPointer ploc );
public static native @Cast("l_int32") int stringFindSubstr( @Cast("const char*") BytePointer src, @Cast("const char*") BytePointer sub, @Cast("l_int32*") IntBuffer ploc );
public static native @Cast("l_int32") int stringFindSubstr( String src, String sub, @Cast("l_int32*") int[] ploc );
public static native @Cast("char*") BytePointer stringReplaceSubstr( @Cast("const char*") BytePointer src, @Cast("const char*") BytePointer sub1, @Cast("const char*") BytePointer sub2, @Cast("l_int32*") IntPointer pfound, @Cast("l_int32*") IntPointer ploc );
public static native @Cast("char*") ByteBuffer stringReplaceSubstr( String src, String sub1, String sub2, @Cast("l_int32*") IntBuffer pfound, @Cast("l_int32*") IntBuffer ploc );
public static native @Cast("char*") byte[] stringReplaceSubstr( @Cast("const char*") BytePointer src, @Cast("const char*") BytePointer sub1, @Cast("const char*") BytePointer sub2, @Cast("l_int32*") int[] pfound, @Cast("l_int32*") int[] ploc );
public static native @Cast("char*") BytePointer stringReplaceSubstr( String src, String sub1, String sub2, @Cast("l_int32*") IntPointer pfound, @Cast("l_int32*") IntPointer ploc );
public static native @Cast("char*") ByteBuffer stringReplaceSubstr( @Cast("const char*") BytePointer src, @Cast("const char*") BytePointer sub1, @Cast("const char*") BytePointer sub2, @Cast("l_int32*") IntBuffer pfound, @Cast("l_int32*") IntBuffer ploc );
public static native @Cast("char*") byte[] stringReplaceSubstr( String src, String sub1, String sub2, @Cast("l_int32*") int[] pfound, @Cast("l_int32*") int[] ploc );
public static native @Cast("char*") BytePointer stringReplaceEachSubstr( @Cast("const char*") BytePointer src, @Cast("const char*") BytePointer sub1, @Cast("const char*") BytePointer sub2, @Cast("l_int32*") IntPointer pcount );
public static native @Cast("char*") ByteBuffer stringReplaceEachSubstr( String src, String sub1, String sub2, @Cast("l_int32*") IntBuffer pcount );
public static native @Cast("char*") byte[] stringReplaceEachSubstr( @Cast("const char*") BytePointer src, @Cast("const char*") BytePointer sub1, @Cast("const char*") BytePointer sub2, @Cast("l_int32*") int[] pcount );
public static native @Cast("char*") BytePointer stringReplaceEachSubstr( String src, String sub1, String sub2, @Cast("l_int32*") IntPointer pcount );
public static native @Cast("char*") ByteBuffer stringReplaceEachSubstr( @Cast("const char*") BytePointer src, @Cast("const char*") BytePointer sub1, @Cast("const char*") BytePointer sub2, @Cast("l_int32*") IntBuffer pcount );
public static native @Cast("char*") byte[] stringReplaceEachSubstr( String src, String sub1, String sub2, @Cast("l_int32*") int[] pcount );
public static native L_DNA arrayFindEachSequence( @Cast("const l_uint8*") BytePointer data, @Cast("size_t") long datalen, @Cast("const l_uint8*") BytePointer sequence, @Cast("size_t") long seqlen );
public static native L_DNA arrayFindEachSequence( @Cast("const l_uint8*") ByteBuffer data, @Cast("size_t") long datalen, @Cast("const l_uint8*") ByteBuffer sequence, @Cast("size_t") long seqlen );
public static native L_DNA arrayFindEachSequence( @Cast("const l_uint8*") byte[] data, @Cast("size_t") long datalen, @Cast("const l_uint8*") byte[] sequence, @Cast("size_t") long seqlen );
public static native @Cast("l_int32") int arrayFindSequence( @Cast("const l_uint8*") BytePointer data, @Cast("size_t") long datalen, @Cast("const l_uint8*") BytePointer sequence, @Cast("size_t") long seqlen, @Cast("l_int32*") IntPointer poffset, @Cast("l_int32*") IntPointer pfound );
public static native @Cast("l_int32") int arrayFindSequence( @Cast("const l_uint8*") ByteBuffer data, @Cast("size_t") long datalen, @Cast("const l_uint8*") ByteBuffer sequence, @Cast("size_t") long seqlen, @Cast("l_int32*") IntBuffer poffset, @Cast("l_int32*") IntBuffer pfound );
public static native @Cast("l_int32") int arrayFindSequence( @Cast("const l_uint8*") byte[] data, @Cast("size_t") long datalen, @Cast("const l_uint8*") byte[] sequence, @Cast("size_t") long seqlen, @Cast("l_int32*") int[] poffset, @Cast("l_int32*") int[] pfound );
public static native Pointer reallocNew( @Cast("void**") PointerPointer pindata, @Cast("l_int32") int oldsize, @Cast("l_int32") int newsize );
public static native Pointer reallocNew( @Cast("void**") @ByPtrPtr Pointer pindata, @Cast("l_int32") int oldsize, @Cast("l_int32") int newsize );
public static native @Cast("l_uint8*") BytePointer l_binaryRead( @Cast("const char*") BytePointer filename, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_uint8*") ByteBuffer l_binaryRead( String filename, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_uint8*") BytePointer l_binaryReadStream( @Cast("FILE*") Pointer fp, @Cast("size_t*") SizeTPointer pnbytes );
public static native @Cast("l_uint8*") BytePointer l_binaryReadSelect( @Cast("const char*") BytePointer filename, @Cast("size_t") long start, @Cast("size_t") long nbytes, @Cast("size_t*") SizeTPointer pnread );
public static native @Cast("l_uint8*") ByteBuffer l_binaryReadSelect( String filename, @Cast("size_t") long start, @Cast("size_t") long nbytes, @Cast("size_t*") SizeTPointer pnread );
public static native @Cast("l_uint8*") BytePointer l_binaryReadSelectStream( @Cast("FILE*") Pointer fp, @Cast("size_t") long start, @Cast("size_t") long nbytes, @Cast("size_t*") SizeTPointer pnread );
public static native @Cast("l_int32") int l_binaryWrite( @Cast("const char*") BytePointer filename, @Cast("const char*") BytePointer operation, Pointer data, @Cast("size_t") long nbytes );
public static native @Cast("l_int32") int l_binaryWrite( String filename, String operation, Pointer data, @Cast("size_t") long nbytes );
public static native @Cast("size_t") long nbytesInFile( @Cast("const char*") BytePointer filename );
public static native @Cast("size_t") long nbytesInFile( String filename );
public static native @Cast("size_t") long fnbytesInFile( @Cast("FILE*") Pointer fp );
public static native @Cast("l_uint8*") BytePointer l_binaryCopy( @Cast("l_uint8*") BytePointer datas, @Cast("size_t") long size );
public static native @Cast("l_uint8*") ByteBuffer l_binaryCopy( @Cast("l_uint8*") ByteBuffer datas, @Cast("size_t") long size );
public static native @Cast("l_uint8*") byte[] l_binaryCopy( @Cast("l_uint8*") byte[] datas, @Cast("size_t") long size );
public static native @Cast("l_int32") int fileCopy( @Cast("const char*") BytePointer srcfile, @Cast("const char*") BytePointer newfile );
public static native @Cast("l_int32") int fileCopy( String srcfile, String newfile );
public static native @Cast("l_int32") int fileConcatenate( @Cast("const char*") BytePointer srcfile, @Cast("const char*") BytePointer destfile );
public static native @Cast("l_int32") int fileConcatenate( String srcfile, String destfile );
public static native @Cast("l_int32") int fileAppendString( @Cast("const char*") BytePointer filename, @Cast("const char*") BytePointer str );
public static native @Cast("l_int32") int fileAppendString( String filename, String str );
public static native @Cast("l_int32") int filesAreIdentical( @Cast("const char*") BytePointer fname1, @Cast("const char*") BytePointer fname2, @Cast("l_int32*") IntPointer psame );
public static native @Cast("l_int32") int filesAreIdentical( String fname1, String fname2, @Cast("l_int32*") IntBuffer psame );
public static native @Cast("l_int32") int filesAreIdentical( @Cast("const char*") BytePointer fname1, @Cast("const char*") BytePointer fname2, @Cast("l_int32*") int[] psame );
public static native @Cast("l_int32") int filesAreIdentical( String fname1, String fname2, @Cast("l_int32*") IntPointer psame );
public static native @Cast("l_int32") int filesAreIdentical( @Cast("const char*") BytePointer fname1, @Cast("const char*") BytePointer fname2, @Cast("l_int32*") IntBuffer psame );
public static native @Cast("l_int32") int filesAreIdentical( String fname1, String fname2, @Cast("l_int32*") int[] psame );
public static native @Cast("l_uint16") short convertOnLittleEnd16( @Cast("l_uint16") short shortin );
public static native @Cast("l_uint16") short convertOnBigEnd16( @Cast("l_uint16") short shortin );
public static native @Cast("l_uint32") int convertOnLittleEnd32( @Cast("l_uint32") int wordin );
public static native @Cast("l_uint32") int convertOnBigEnd32( @Cast("l_uint32") int wordin );
public static native @Cast("FILE*") Pointer fopenReadStream( @Cast("const char*") BytePointer filename );
public static native @Cast("FILE*") Pointer fopenReadStream( String filename );
public static native @Cast("FILE*") Pointer fopenWriteStream( @Cast("const char*") BytePointer filename, @Cast("const char*") BytePointer modestring );
public static native @Cast("FILE*") Pointer fopenWriteStream( String filename, String modestring );
public static native @Cast("FILE*") Pointer lept_fopen( @Cast("const char*") BytePointer filename, @Cast("const char*") BytePointer mode );
public static native @Cast("FILE*") Pointer lept_fopen( String filename, String mode );
public static native @Cast("l_int32") int lept_fclose( @Cast("FILE*") Pointer fp );
public static native Pointer lept_calloc( @Cast("size_t") long nmemb, @Cast("size_t") long size );
public static native void lept_free( Pointer ptr );
public static native @Cast("l_int32") int lept_mkdir( @Cast("const char*") BytePointer subdir );
public static native @Cast("l_int32") int lept_mkdir( String subdir );
public static native @Cast("l_int32") int lept_rmdir( @Cast("const char*") BytePointer subdir );
public static native @Cast("l_int32") int lept_rmdir( String subdir );
public static native void lept_direxists( @Cast("const char*") BytePointer dir, @Cast("l_int32*") IntPointer pexists );
public static native void lept_direxists( String dir, @Cast("l_int32*") IntBuffer pexists );
public static native void lept_direxists( @Cast("const char*") BytePointer dir, @Cast("l_int32*") int[] pexists );
public static native void lept_direxists( String dir, @Cast("l_int32*") IntPointer pexists );
public static native void lept_direxists( @Cast("const char*") BytePointer dir, @Cast("l_int32*") IntBuffer pexists );
public static native void lept_direxists( String dir, @Cast("l_int32*") int[] pexists );
public static native @Cast("l_int32") int lept_rm_match( @Cast("const char*") BytePointer subdir, @Cast("const char*") BytePointer substr );
public static native @Cast("l_int32") int lept_rm_match( String subdir, String substr );
public static native @Cast("l_int32") int lept_rm( @Cast("const char*") BytePointer subdir, @Cast("const char*") BytePointer tail );
public static native @Cast("l_int32") int lept_rm( String subdir, String tail );
public static native @Cast("l_int32") int lept_rmfile( @Cast("const char*") BytePointer filepath );
public static native @Cast("l_int32") int lept_rmfile( String filepath );
public static native @Cast("l_int32") int lept_mv( @Cast("const char*") BytePointer srcfile, @Cast("const char*") BytePointer newdir, @Cast("const char*") BytePointer newtail, @Cast("char**") PointerPointer pnewpath );
public static native @Cast("l_int32") int lept_mv( @Cast("const char*") BytePointer srcfile, @Cast("const char*") BytePointer newdir, @Cast("const char*") BytePointer newtail, @Cast("char**") @ByPtrPtr BytePointer pnewpath );
public static native @Cast("l_int32") int lept_mv( String srcfile, String newdir, String newtail, @Cast("char**") @ByPtrPtr ByteBuffer pnewpath );
public static native @Cast("l_int32") int lept_mv( @Cast("const char*") BytePointer srcfile, @Cast("const char*") BytePointer newdir, @Cast("const char*") BytePointer newtail, @Cast("char**") @ByPtrPtr byte[] pnewpath );
public static native @Cast("l_int32") int lept_mv( String srcfile, String newdir, String newtail, @Cast("char**") @ByPtrPtr BytePointer pnewpath );
public static native @Cast("l_int32") int lept_mv( @Cast("const char*") BytePointer srcfile, @Cast("const char*") BytePointer newdir, @Cast("const char*") BytePointer newtail, @Cast("char**") @ByPtrPtr ByteBuffer pnewpath );
public static native @Cast("l_int32") int lept_mv( String srcfile, String newdir, String newtail, @Cast("char**") @ByPtrPtr byte[] pnewpath );
public static native @Cast("l_int32") int lept_cp( @Cast("const char*") BytePointer srcfile, @Cast("const char*") BytePointer newdir, @Cast("const char*") BytePointer newtail, @Cast("char**") PointerPointer pnewpath );
public static native @Cast("l_int32") int lept_cp( @Cast("const char*") BytePointer srcfile, @Cast("const char*") BytePointer newdir, @Cast("const char*") BytePointer newtail, @Cast("char**") @ByPtrPtr BytePointer pnewpath );
public static native @Cast("l_int32") int lept_cp( String srcfile, String newdir, String newtail, @Cast("char**") @ByPtrPtr ByteBuffer pnewpath );
public static native @Cast("l_int32") int lept_cp( @Cast("const char*") BytePointer srcfile, @Cast("const char*") BytePointer newdir, @Cast("const char*") BytePointer newtail, @Cast("char**") @ByPtrPtr byte[] pnewpath );
public static native @Cast("l_int32") int lept_cp( String srcfile, String newdir, String newtail, @Cast("char**") @ByPtrPtr BytePointer pnewpath );
public static native @Cast("l_int32") int lept_cp( @Cast("const char*") BytePointer srcfile, @Cast("const char*") BytePointer newdir, @Cast("const char*") BytePointer newtail, @Cast("char**") @ByPtrPtr ByteBuffer pnewpath );
public static native @Cast("l_int32") int lept_cp( String srcfile, String newdir, String newtail, @Cast("char**") @ByPtrPtr byte[] pnewpath );
public static native @Cast("l_int32") int splitPathAtDirectory( @Cast("const char*") BytePointer pathname, @Cast("char**") PointerPointer pdir, @Cast("char**") PointerPointer ptail );
public static native @Cast("l_int32") int splitPathAtDirectory( @Cast("const char*") BytePointer pathname, @Cast("char**") @ByPtrPtr BytePointer pdir, @Cast("char**") @ByPtrPtr BytePointer ptail );
public static native @Cast("l_int32") int splitPathAtDirectory( String pathname, @Cast("char**") @ByPtrPtr ByteBuffer pdir, @Cast("char**") @ByPtrPtr ByteBuffer ptail );
public static native @Cast("l_int32") int splitPathAtDirectory( @Cast("const char*") BytePointer pathname, @Cast("char**") @ByPtrPtr byte[] pdir, @Cast("char**") @ByPtrPtr byte[] ptail );
public static native @Cast("l_int32") int splitPathAtDirectory( String pathname, @Cast("char**") @ByPtrPtr BytePointer pdir, @Cast("char**") @ByPtrPtr BytePointer ptail );
public static native @Cast("l_int32") int splitPathAtDirectory( @Cast("const char*") BytePointer pathname, @Cast("char**") @ByPtrPtr ByteBuffer pdir, @Cast("char**") @ByPtrPtr ByteBuffer ptail );
public static native @Cast("l_int32") int splitPathAtDirectory( String pathname, @Cast("char**") @ByPtrPtr byte[] pdir, @Cast("char**") @ByPtrPtr byte[] ptail );
public static native @Cast("l_int32") int splitPathAtExtension( @Cast("const char*") BytePointer pathname, @Cast("char**") PointerPointer pbasename, @Cast("char**") PointerPointer pextension );
public static native @Cast("l_int32") int splitPathAtExtension( @Cast("const char*") BytePointer pathname, @Cast("char**") @ByPtrPtr BytePointer pbasename, @Cast("char**") @ByPtrPtr BytePointer pextension );
public static native @Cast("l_int32") int splitPathAtExtension( String pathname, @Cast("char**") @ByPtrPtr ByteBuffer pbasename, @Cast("char**") @ByPtrPtr ByteBuffer pextension );
public static native @Cast("l_int32") int splitPathAtExtension( @Cast("const char*") BytePointer pathname, @Cast("char**") @ByPtrPtr byte[] pbasename, @Cast("char**") @ByPtrPtr byte[] pextension );
public static native @Cast("l_int32") int splitPathAtExtension( String pathname, @Cast("char**") @ByPtrPtr BytePointer pbasename, @Cast("char**") @ByPtrPtr BytePointer pextension );
public static native @Cast("l_int32") int splitPathAtExtension( @Cast("const char*") BytePointer pathname, @Cast("char**") @ByPtrPtr ByteBuffer pbasename, @Cast("char**") @ByPtrPtr ByteBuffer pextension );
public static native @Cast("l_int32") int splitPathAtExtension( String pathname, @Cast("char**") @ByPtrPtr byte[] pbasename, @Cast("char**") @ByPtrPtr byte[] pextension );
public static native @Cast("char*") BytePointer pathJoin( @Cast("const char*") BytePointer dir, @Cast("const char*") BytePointer fname );
public static native @Cast("char*") ByteBuffer pathJoin( String dir, String fname );
public static native @Cast("char*") BytePointer appendSubdirectory( @Cast("const char*") BytePointer dir, @Cast("const char*") BytePointer subdir );
public static native @Cast("char*") ByteBuffer appendSubdirectory( String dir, String subdir );
public static native @Cast("l_int32") int convertSepCharsInPath( @Cast("char*") BytePointer path, @Cast("l_int32") int type );
public static native @Cast("l_int32") int convertSepCharsInPath( @Cast("char*") ByteBuffer path, @Cast("l_int32") int type );
public static native @Cast("l_int32") int convertSepCharsInPath( @Cast("char*") byte[] path, @Cast("l_int32") int type );
public static native @Cast("char*") BytePointer genPathname( @Cast("const char*") BytePointer dir, @Cast("const char*") BytePointer fname );
public static native @Cast("char*") ByteBuffer genPathname( String dir, String fname );
public static native @Cast("l_int32") int makeTempDirname( @Cast("char*") BytePointer result, @Cast("size_t") long nbytes, @Cast("const char*") BytePointer subdir );
public static native @Cast("l_int32") int makeTempDirname( @Cast("char*") ByteBuffer result, @Cast("size_t") long nbytes, String subdir );
public static native @Cast("l_int32") int makeTempDirname( @Cast("char*") byte[] result, @Cast("size_t") long nbytes, @Cast("const char*") BytePointer subdir );
public static native @Cast("l_int32") int makeTempDirname( @Cast("char*") BytePointer result, @Cast("size_t") long nbytes, String subdir );
public static native @Cast("l_int32") int makeTempDirname( @Cast("char*") ByteBuffer result, @Cast("size_t") long nbytes, @Cast("const char*") BytePointer subdir );
public static native @Cast("l_int32") int makeTempDirname( @Cast("char*") byte[] result, @Cast("size_t") long nbytes, String subdir );
public static native @Cast("l_int32") int modifyTrailingSlash( @Cast("char*") BytePointer path, @Cast("size_t") long nbytes, @Cast("l_int32") int flag );
public static native @Cast("l_int32") int modifyTrailingSlash( @Cast("char*") ByteBuffer path, @Cast("size_t") long nbytes, @Cast("l_int32") int flag );
public static native @Cast("l_int32") int modifyTrailingSlash( @Cast("char*") byte[] path, @Cast("size_t") long nbytes, @Cast("l_int32") int flag );
public static native @Cast("char*") BytePointer genTempFilename( @Cast("const char*") BytePointer dir, @Cast("const char*") BytePointer tail, @Cast("l_int32") int usetime, @Cast("l_int32") int usepid );
public static native @Cast("char*") ByteBuffer genTempFilename( String dir, String tail, @Cast("l_int32") int usetime, @Cast("l_int32") int usepid );
public static native @Cast("l_int32") int extractNumberFromFilename( @Cast("const char*") BytePointer fname, @Cast("l_int32") int numpre, @Cast("l_int32") int numpost );
public static native @Cast("l_int32") int extractNumberFromFilename( String fname, @Cast("l_int32") int numpre, @Cast("l_int32") int numpost );
public static native @Cast("l_int32") int fileCorruptByDeletion( @Cast("const char*") BytePointer filein, @Cast("l_float32") float loc, @Cast("l_float32") float size, @Cast("const char*") BytePointer fileout );
public static native @Cast("l_int32") int fileCorruptByDeletion( String filein, @Cast("l_float32") float loc, @Cast("l_float32") float size, String fileout );
public static native @Cast("l_int32") int fileCorruptByMutation( @Cast("const char*") BytePointer filein, @Cast("l_float32") float loc, @Cast("l_float32") float size, @Cast("const char*") BytePointer fileout );
public static native @Cast("l_int32") int fileCorruptByMutation( String filein, @Cast("l_float32") float loc, @Cast("l_float32") float size, String fileout );
public static native @Cast("l_int32") int genRandomIntegerInRange( @Cast("l_int32") int range, @Cast("l_int32") int seed, @Cast("l_int32*") IntPointer pval );
public static native @Cast("l_int32") int genRandomIntegerInRange( @Cast("l_int32") int range, @Cast("l_int32") int seed, @Cast("l_int32*") IntBuffer pval );
public static native @Cast("l_int32") int genRandomIntegerInRange( @Cast("l_int32") int range, @Cast("l_int32") int seed, @Cast("l_int32*") int[] pval );
public static native @Cast("l_int32") int lept_roundftoi( @Cast("l_float32") float fval );
public static native @Cast("l_uint32") int convertBinaryToGrayCode( @Cast("l_uint32") int val );
public static native @Cast("l_uint32") int convertGrayCodeToBinary( @Cast("l_uint32") int val );
public static native @Cast("char*") BytePointer getLeptonicaVersion(  );
public static native void startTimer( );
public static native @Cast("l_float32") float stopTimer( );
public static native L_TIMER startTimerNested( );
public static native @Cast("l_float32") float stopTimerNested( L_TIMER rusage_start );
public static native void l_getCurrentTime( @Cast("l_int32*") IntPointer sec, @Cast("l_int32*") IntPointer usec );
public static native void l_getCurrentTime( @Cast("l_int32*") IntBuffer sec, @Cast("l_int32*") IntBuffer usec );
public static native void l_getCurrentTime( @Cast("l_int32*") int[] sec, @Cast("l_int32*") int[] usec );
public static native L_WALLTIMER startWallTimer( );
public static native @Cast("l_float32") float stopWallTimer( @Cast("L_WALLTIMER**") PointerPointer ptimer );
public static native @Cast("l_float32") float stopWallTimer( @ByPtrPtr L_WALLTIMER ptimer );
public static native @Cast("char*") BytePointer l_getFormattedDate(  );
public static native @Cast("l_int32") int pixHtmlViewer( @Cast("const char*") BytePointer dirin, @Cast("const char*") BytePointer dirout, @Cast("const char*") BytePointer rootname, @Cast("l_int32") int thumbwidth, @Cast("l_int32") int viewwidth, @Cast("l_int32") int copyorig );
public static native @Cast("l_int32") int pixHtmlViewer( String dirin, String dirout, String rootname, @Cast("l_int32") int thumbwidth, @Cast("l_int32") int viewwidth, @Cast("l_int32") int copyorig );
public static native PIX pixSimpleCaptcha( PIX pixs, @Cast("l_int32") int border, @Cast("l_int32") int nterms, @Cast("l_uint32") int seed, @Cast("l_uint32") int color, @Cast("l_int32") int cmapflag );
public static native PIX pixRandomHarmonicWarp( PIX pixs, @Cast("l_float32") float xmag, @Cast("l_float32") float ymag, @Cast("l_float32") float xfreq, @Cast("l_float32") float yfreq, @Cast("l_int32") int nx, @Cast("l_int32") int ny, @Cast("l_uint32") int seed, @Cast("l_int32") int grayval );
public static native PIX pixWarpStereoscopic( PIX pixs, @Cast("l_int32") int zbend, @Cast("l_int32") int zshiftt, @Cast("l_int32") int zshiftb, @Cast("l_int32") int ybendt, @Cast("l_int32") int ybendb, @Cast("l_int32") int redleft );
public static native PIX pixStretchHorizontal( PIX pixs, @Cast("l_int32") int dir, @Cast("l_int32") int type, @Cast("l_int32") int hmax, @Cast("l_int32") int operation, @Cast("l_int32") int incolor );
public static native PIX pixStretchHorizontalSampled( PIX pixs, @Cast("l_int32") int dir, @Cast("l_int32") int type, @Cast("l_int32") int hmax, @Cast("l_int32") int incolor );
public static native PIX pixStretchHorizontalLI( PIX pixs, @Cast("l_int32") int dir, @Cast("l_int32") int type, @Cast("l_int32") int hmax, @Cast("l_int32") int incolor );
public static native PIX pixQuadraticVShear( PIX pixs, @Cast("l_int32") int dir, @Cast("l_int32") int vmaxt, @Cast("l_int32") int vmaxb, @Cast("l_int32") int operation, @Cast("l_int32") int incolor );
public static native PIX pixQuadraticVShearSampled( PIX pixs, @Cast("l_int32") int dir, @Cast("l_int32") int vmaxt, @Cast("l_int32") int vmaxb, @Cast("l_int32") int incolor );
public static native PIX pixQuadraticVShearLI( PIX pixs, @Cast("l_int32") int dir, @Cast("l_int32") int vmaxt, @Cast("l_int32") int vmaxb, @Cast("l_int32") int incolor );
public static native PIX pixStereoFromPair( PIX pix1, PIX pix2, @Cast("l_float32") float rwt, @Cast("l_float32") float gwt, @Cast("l_float32") float bwt );
public static native L_WSHED wshedCreate( PIX pixs, PIX pixm, @Cast("l_int32") int mindepth, @Cast("l_int32") int debugflag );
public static native void wshedDestroy( @Cast("L_WSHED**") PointerPointer pwshed );
public static native void wshedDestroy( @ByPtrPtr L_WSHED pwshed );
public static native @Cast("l_int32") int wshedApply( L_WSHED wshed );
public static native @Cast("l_int32") int wshedBasins( L_WSHED wshed, @Cast("PIXA**") PointerPointer ppixa, @Cast("NUMA**") PointerPointer pnalevels );
public static native @Cast("l_int32") int wshedBasins( L_WSHED wshed, @ByPtrPtr PIXA ppixa, @ByPtrPtr NUMA pnalevels );
public static native PIX wshedRenderFill( L_WSHED wshed );
public static native PIX wshedRenderColors( L_WSHED wshed );
public static native PIX pixReadStreamWebP( @Cast("FILE*") Pointer fp );
public static native PIX pixReadMemWebP( @Cast("const l_uint8*") BytePointer filedata, @Cast("size_t") long filesize );
public static native PIX pixReadMemWebP( @Cast("const l_uint8*") ByteBuffer filedata, @Cast("size_t") long filesize );
public static native PIX pixReadMemWebP( @Cast("const l_uint8*") byte[] filedata, @Cast("size_t") long filesize );
public static native @Cast("l_int32") int readHeaderWebP( @Cast("const char*") BytePointer filename, @Cast("l_int32*") IntPointer pw, @Cast("l_int32*") IntPointer ph, @Cast("l_int32*") IntPointer pspp );
public static native @Cast("l_int32") int readHeaderWebP( String filename, @Cast("l_int32*") IntBuffer pw, @Cast("l_int32*") IntBuffer ph, @Cast("l_int32*") IntBuffer pspp );
public static native @Cast("l_int32") int readHeaderWebP( @Cast("const char*") BytePointer filename, @Cast("l_int32*") int[] pw, @Cast("l_int32*") int[] ph, @Cast("l_int32*") int[] pspp );
public static native @Cast("l_int32") int readHeaderWebP( String filename, @Cast("l_int32*") IntPointer pw, @Cast("l_int32*") IntPointer ph, @Cast("l_int32*") IntPointer pspp );
public static native @Cast("l_int32") int readHeaderWebP( @Cast("const char*") BytePointer filename, @Cast("l_int32*") IntBuffer pw, @Cast("l_int32*") IntBuffer ph, @Cast("l_int32*") IntBuffer pspp );
public static native @Cast("l_int32") int readHeaderWebP( String filename, @Cast("l_int32*") int[] pw, @Cast("l_int32*") int[] ph, @Cast("l_int32*") int[] pspp );
public static native @Cast("l_int32") int readHeaderMemWebP( @Cast("const l_uint8*") BytePointer data, @Cast("size_t") long size, @Cast("l_int32*") IntPointer pw, @Cast("l_int32*") IntPointer ph, @Cast("l_int32*") IntPointer pspp );
public static native @Cast("l_int32") int readHeaderMemWebP( @Cast("const l_uint8*") ByteBuffer data, @Cast("size_t") long size, @Cast("l_int32*") IntBuffer pw, @Cast("l_int32*") IntBuffer ph, @Cast("l_int32*") IntBuffer pspp );
public static native @Cast("l_int32") int readHeaderMemWebP( @Cast("const l_uint8*") byte[] data, @Cast("size_t") long size, @Cast("l_int32*") int[] pw, @Cast("l_int32*") int[] ph, @Cast("l_int32*") int[] pspp );
public static native @Cast("l_int32") int pixWriteWebP( @Cast("const char*") BytePointer filename, PIX pixs, @Cast("l_int32") int quality, @Cast("l_int32") int lossless );
public static native @Cast("l_int32") int pixWriteWebP( String filename, PIX pixs, @Cast("l_int32") int quality, @Cast("l_int32") int lossless );
public static native @Cast("l_int32") int pixWriteStreamWebP( @Cast("FILE*") Pointer fp, PIX pixs, @Cast("l_int32") int quality, @Cast("l_int32") int lossless );
public static native @Cast("l_int32") int pixWriteMemWebP( @Cast("l_uint8**") PointerPointer pencdata, @Cast("size_t*") SizeTPointer pencsize, PIX pixs, @Cast("l_int32") int quality, @Cast("l_int32") int lossless );
public static native @Cast("l_int32") int pixWriteMemWebP( @Cast("l_uint8**") @ByPtrPtr BytePointer pencdata, @Cast("size_t*") SizeTPointer pencsize, PIX pixs, @Cast("l_int32") int quality, @Cast("l_int32") int lossless );
public static native @Cast("l_int32") int pixWriteMemWebP( @Cast("l_uint8**") @ByPtrPtr ByteBuffer pencdata, @Cast("size_t*") SizeTPointer pencsize, PIX pixs, @Cast("l_int32") int quality, @Cast("l_int32") int lossless );
public static native @Cast("l_int32") int pixWriteMemWebP( @Cast("l_uint8**") @ByPtrPtr byte[] pencdata, @Cast("size_t*") SizeTPointer pencsize, PIX pixs, @Cast("l_int32") int quality, @Cast("l_int32") int lossless );
public static native @Cast("l_int32") int pixaWriteFiles( @Cast("const char*") BytePointer rootname, PIXA pixa, @Cast("l_int32") int format );
public static native @Cast("l_int32") int pixaWriteFiles( String rootname, PIXA pixa, @Cast("l_int32") int format );
public static native @Cast("l_int32") int pixWrite( @Cast("const char*") BytePointer filename, PIX pix, @Cast("l_int32") int format );
public static native @Cast("l_int32") int pixWrite( String filename, PIX pix, @Cast("l_int32") int format );
public static native @Cast("l_int32") int pixWriteAutoFormat( @Cast("const char*") BytePointer filename, PIX pix );
public static native @Cast("l_int32") int pixWriteAutoFormat( String filename, PIX pix );
public static native @Cast("l_int32") int pixWriteStream( @Cast("FILE*") Pointer fp, PIX pix, @Cast("l_int32") int format );
public static native @Cast("l_int32") int pixWriteImpliedFormat( @Cast("const char*") BytePointer filename, PIX pix, @Cast("l_int32") int quality, @Cast("l_int32") int progressive );
public static native @Cast("l_int32") int pixWriteImpliedFormat( String filename, PIX pix, @Cast("l_int32") int quality, @Cast("l_int32") int progressive );
public static native @Cast("l_int32") int pixWriteTempfile( @Cast("const char*") BytePointer dir, @Cast("const char*") BytePointer tail, PIX pix, @Cast("l_int32") int format, @Cast("char**") PointerPointer pfilename );
public static native @Cast("l_int32") int pixWriteTempfile( @Cast("const char*") BytePointer dir, @Cast("const char*") BytePointer tail, PIX pix, @Cast("l_int32") int format, @Cast("char**") @ByPtrPtr BytePointer pfilename );
public static native @Cast("l_int32") int pixWriteTempfile( String dir, String tail, PIX pix, @Cast("l_int32") int format, @Cast("char**") @ByPtrPtr ByteBuffer pfilename );
public static native @Cast("l_int32") int pixWriteTempfile( @Cast("const char*") BytePointer dir, @Cast("const char*") BytePointer tail, PIX pix, @Cast("l_int32") int format, @Cast("char**") @ByPtrPtr byte[] pfilename );
public static native @Cast("l_int32") int pixWriteTempfile( String dir, String tail, PIX pix, @Cast("l_int32") int format, @Cast("char**") @ByPtrPtr BytePointer pfilename );
public static native @Cast("l_int32") int pixWriteTempfile( @Cast("const char*") BytePointer dir, @Cast("const char*") BytePointer tail, PIX pix, @Cast("l_int32") int format, @Cast("char**") @ByPtrPtr ByteBuffer pfilename );
public static native @Cast("l_int32") int pixWriteTempfile( String dir, String tail, PIX pix, @Cast("l_int32") int format, @Cast("char**") @ByPtrPtr byte[] pfilename );
public static native @Cast("l_int32") int pixChooseOutputFormat( PIX pix );
public static native @Cast("l_int32") int getImpliedFileFormat( @Cast("const char*") BytePointer filename );
public static native @Cast("l_int32") int getImpliedFileFormat( String filename );
public static native @Cast("l_int32") int pixGetAutoFormat( PIX pix, @Cast("l_int32*") IntPointer pformat );
public static native @Cast("l_int32") int pixGetAutoFormat( PIX pix, @Cast("l_int32*") IntBuffer pformat );
public static native @Cast("l_int32") int pixGetAutoFormat( PIX pix, @Cast("l_int32*") int[] pformat );
public static native @Cast("const char*") BytePointer getFormatExtension( @Cast("l_int32") int format );
public static native @Cast("l_int32") int pixWriteMem( @Cast("l_uint8**") PointerPointer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix, @Cast("l_int32") int format );
public static native @Cast("l_int32") int pixWriteMem( @Cast("l_uint8**") @ByPtrPtr BytePointer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix, @Cast("l_int32") int format );
public static native @Cast("l_int32") int pixWriteMem( @Cast("l_uint8**") @ByPtrPtr ByteBuffer pdata, @Cast("size_t*") SizeTPointer psize, PIX pix, @Cast("l_int32") int format );
public static native @Cast("l_int32") int pixWriteMem( @Cast("l_uint8**") @ByPtrPtr byte[] pdata, @Cast("size_t*") SizeTPointer psize, PIX pix, @Cast("l_int32") int format );
public static native @Cast("l_int32") int pixDisplay( PIX pixs, @Cast("l_int32") int x, @Cast("l_int32") int y );
public static native @Cast("l_int32") int pixDisplayWithTitle( PIX pixs, @Cast("l_int32") int x, @Cast("l_int32") int y, @Cast("const char*") BytePointer title, @Cast("l_int32") int dispflag );
public static native @Cast("l_int32") int pixDisplayWithTitle( PIX pixs, @Cast("l_int32") int x, @Cast("l_int32") int y, String title, @Cast("l_int32") int dispflag );
public static native @Cast("l_int32") int pixDisplayMultiple( @Cast("const char*") BytePointer filepattern );
public static native @Cast("l_int32") int pixDisplayMultiple( String filepattern );
public static native @Cast("l_int32") int pixDisplayWrite( PIX pixs, @Cast("l_int32") int reduction );
public static native @Cast("l_int32") int pixDisplayWriteFormat( PIX pixs, @Cast("l_int32") int reduction, @Cast("l_int32") int format );
public static native @Cast("l_int32") int pixSaveTiled( PIX pixs, PIXA pixa, @Cast("l_float32") float scalefactor, @Cast("l_int32") int newrow, @Cast("l_int32") int space, @Cast("l_int32") int dp );
public static native @Cast("l_int32") int pixSaveTiledOutline( PIX pixs, PIXA pixa, @Cast("l_float32") float scalefactor, @Cast("l_int32") int newrow, @Cast("l_int32") int space, @Cast("l_int32") int linewidth, @Cast("l_int32") int dp );
public static native @Cast("l_int32") int pixSaveTiledWithText( PIX pixs, PIXA pixa, @Cast("l_int32") int outwidth, @Cast("l_int32") int newrow, @Cast("l_int32") int space, @Cast("l_int32") int linewidth, L_BMF bmf, @Cast("const char*") BytePointer textstr, @Cast("l_uint32") int val, @Cast("l_int32") int location );
public static native @Cast("l_int32") int pixSaveTiledWithText( PIX pixs, PIXA pixa, @Cast("l_int32") int outwidth, @Cast("l_int32") int newrow, @Cast("l_int32") int space, @Cast("l_int32") int linewidth, L_BMF bmf, String textstr, @Cast("l_uint32") int val, @Cast("l_int32") int location );
public static native void l_chooseDisplayProg( @Cast("l_int32") int selection );
public static native @Cast("l_uint8*") BytePointer zlibCompress( @Cast("l_uint8*") BytePointer datain, @Cast("size_t") long nin, @Cast("size_t*") SizeTPointer pnout );
public static native @Cast("l_uint8*") ByteBuffer zlibCompress( @Cast("l_uint8*") ByteBuffer datain, @Cast("size_t") long nin, @Cast("size_t*") SizeTPointer pnout );
public static native @Cast("l_uint8*") byte[] zlibCompress( @Cast("l_uint8*") byte[] datain, @Cast("size_t") long nin, @Cast("size_t*") SizeTPointer pnout );
public static native @Cast("l_uint8*") BytePointer zlibUncompress( @Cast("l_uint8*") BytePointer datain, @Cast("size_t") long nin, @Cast("size_t*") SizeTPointer pnout );
public static native @Cast("l_uint8*") ByteBuffer zlibUncompress( @Cast("l_uint8*") ByteBuffer datain, @Cast("size_t") long nin, @Cast("size_t*") SizeTPointer pnout );
public static native @Cast("l_uint8*") byte[] zlibUncompress( @Cast("l_uint8*") byte[] datain, @Cast("size_t") long nin, @Cast("size_t*") SizeTPointer pnout );

// #ifdef __cplusplus
// #endif  /* __cplusplus */
// #endif /* NO_PROTOS */


// #endif /* LEPTONICA_ALLHEADERS_H */



}
