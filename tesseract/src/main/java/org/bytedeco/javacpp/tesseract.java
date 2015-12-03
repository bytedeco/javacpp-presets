// Targeted by JavaCPP version 1.2-SNAPSHOT

package org.bytedeco.javacpp;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.lept.*;

public class tesseract extends org.bytedeco.javacpp.presets.tesseract {
    static { Loader.load(); }

// Parsed from tesseract/platform.h

///////////////////////////////////////////////////////////////////////
// File:        platform.h
// Description: Place holder
// Author:
// Created:
//
// (C) Copyright 2006, Google Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
///////////////////////////////////////////////////////////////////////

// #ifndef TESSERACT_CCUTIL_PLATFORM_H__
// #define TESSERACT_CCUTIL_PLATFORM_H__

// #include <string.h>

// #define DLLSYM
// #ifdef _WIN32
// #ifdef __GNUC__
// #define ultoa _ultoa
// #endif  /* __GNUC__ */
// #define SIGNED
// #if defined(_MSC_VER)
// #define snprintf _snprintf
// #if (_MSC_VER <= 1400)
// #define vsnprintf _vsnprintf
// #endif /* (_MSC_VER <= 1400) */
// #endif /* defined(_MSC_VER) */
// #else
// #define __UNIX__
// #include <limits.h>
// #ifndef PATH_MAX
public static final int MAX_PATH = 4096;
// #else
// #endif
// #define SIGNED signed
// #endif

// #ifdef _WIN32
// #ifndef M_PI
public static final double M_PI = 3.14159265358979323846;
// #endif
// #endif

// #if defined(_WIN32) || defined(__CYGWIN__)
//     #if defined(TESS_EXPORTS)
//        #define TESS_API __declspec(dllexport)
//     #elif defined(TESS_IMPORTS)
//        #define TESS_API __declspec(dllimport)
//     #else
//        #define TESS_API
//     #endif
//     #define TESS_LOCAL
// #else
//     #if __GNUC__ >= 4
//       #if defined(TESS_EXPORTS) || defined(TESS_IMPORTS)
//           #define TESS_API  __attribute__ ((visibility ("default")))
//           #define TESS_LOCAL  __attribute__ ((visibility ("hidden")))
//       #else
//           #define TESS_API
//           #define TESS_LOCAL
//       #endif
//     #else
//       #define TESS_API
//       #define TESS_LOCAL
//     #endif
// #endif

// #if defined(_WIN32) || defined(__CYGWIN__)
//     #define _TESS_FILE_BASENAME_
//       (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)
// #else   // Unices
//     #define _TESS_FILE_BASENAME_
//       (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
// #endif

// #endif  // TESSERACT_CCUTIL_PLATFORM_H__


// Parsed from tesseract/apitypes.h

///////////////////////////////////////////////////////////////////////
// File:        apitypes.h
// Description: Types used in both the API and internally
// Author:      Ray Smith
// Created:     Wed Mar 03 09:22:53 PST 2010
//
// (C) Copyright 2010, Google Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
///////////////////////////////////////////////////////////////////////

// #ifndef TESSERACT_API_APITYPES_H__
// #define TESSERACT_API_APITYPES_H__

// #include "publictypes.h"

// The types used by the API and Page/ResultIterator can be found in:
//   ccstruct/publictypes.h
//   ccmain/resultiterator.h
//   ccmain/pageiterator.h
// API interfaces and API users should be sure to include this file, rather
// than the lower-level one, and lower-level code should be sure to include
// only the lower-level file.

// #endif  // TESSERACT_API_APITYPES_H__


// Parsed from tesseract/unichar.h

///////////////////////////////////////////////////////////////////////
// File:        unichar.h
// Description: Unicode character/ligature class.
// Author:      Ray Smith
// Created:     Wed Jun 28 17:05:01 PDT 2006
//
// (C) Copyright 2006, Google Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
///////////////////////////////////////////////////////////////////////

// #ifndef TESSERACT_CCUTIL_UNICHAR_H__
// #define TESSERACT_CCUTIL_UNICHAR_H__

// #include <memory.h>
// #include <string.h>

// Maximum number of characters that can be stored in a UNICHAR. Must be
// at least 4. Must not exceed 31 without changing the coding of length.
public static final int UNICHAR_LEN = 30;

// A UNICHAR_ID is the unique id of a unichar.

// A variable to indicate an invalid or uninitialized unichar id.
@MemberGetter public static native int INVALID_UNICHAR_ID();
public static final int INVALID_UNICHAR_ID = INVALID_UNICHAR_ID();
// A special unichar that corresponds to INVALID_UNICHAR_ID.
@MemberGetter public static native byte INVALID_UNICHAR(int i);
@MemberGetter public static native @Cast("const char*") BytePointer INVALID_UNICHAR();

/** enum StrongScriptDirection */
public static final int
  DIR_NEUTRAL = 0,        // Text contains only neutral characters.
  DIR_LEFT_TO_RIGHT = 1,  // Text contains no Right-to-Left characters.
  DIR_RIGHT_TO_LEFT = 2,  // Text contains no Left-to-Right characters.
  DIR_MIX = 3;            // Text contains a mixture of left-to-right
                          // and right-to-left characters.

// The UNICHAR class holds a single classification result. This may be
// a single Unicode character (stored as between 1 and 4 utf8 bytes) or
// multple Unicode characters representing the NFKC expansion of a ligature
// such as fi, ffl etc. These are also stored as utf8.
@NoOffset public static class UNICHAR extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public UNICHAR(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public UNICHAR(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public UNICHAR position(int position) {
        return (UNICHAR)super.position(position);
    }

  public UNICHAR() { super((Pointer)null); allocate(); }
  private native void allocate();

  // Construct from a utf8 string. If len<0 then the string is null terminated.
  // If the string is too long to fit in the UNICHAR then it takes only what
  // will fit.
  

  // Construct from a single UCS4 character.
  

  // Default copy constructor and operator= are OK.

  // Get the first character as UCS-4.
  

  // Get the length of the UTF8 string.
  public native int utf8_len();

  // Get a UTF8 string, but NOT NULL terminated.
  public native @Cast("const char*") BytePointer utf8();

  // Get a terminated UTF8 string: Must delete[] it after use.
  

  // Get the number of bytes in the first character of the given utf8 string.
  

  // A class to simplify iterating over and accessing elements of a UTF8
  // string. Note that unlike the UNICHAR class, const_iterator does NOT COPY or
  // take ownership of the underlying byte array. It also does not permit
  // modification of the array (as the name suggests).
  //
  // Example:
  //   for (UNICHAR::const_iterator it = UNICHAR::begin(str, str_len);
  //        it != UNICHAR::end(str, len);
  //        ++it) {
  //     tprintf("UCS-4 symbol code = %d\n", *it);
  //     char buf[5];
  //     int char_len = it.get_utf8(buf); buf[char_len] = '\0';
  //     tprintf("Char = %s\n", buf);
  //   }
  @NoOffset public static class const_iterator extends Pointer {
      static { Loader.load(); }
      /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
      public const_iterator(Pointer p) { super(p); }
  
    // Step to the next UTF8 character.
    // If the current position is at an illegal UTF8 character, then print an
    // error message and step by one byte. If the current position is at a NULL
    // value, don't step past it.
    public native @ByRef @Name("operator ++") const_iterator increment();

    // Return the UCS-4 value at the current position.
    // If the current position is at an illegal UTF8 value, return a single
    // space character.
    public native @Name("operator *") int multiply();

    // Store the UTF-8 encoding of the current codepoint into buf, which must be
    // at least 4 bytes long. Return the number of bytes written.
    // If the current position is at an illegal UTF8 value, writes a single
    // space character and returns 1.
    // Note that this method does not null-terminate the buffer.
    public native int get_utf8(@Cast("char*") BytePointer buf);
    public native int get_utf8(@Cast("char*") ByteBuffer buf);
    public native int get_utf8(@Cast("char*") byte[] buf);
    // Returns the number of bytes of the current codepoint. Returns 1 if the
    // current position is at an illegal UTF8 value.
    public native int utf8_len();
    // Returns true if the UTF-8 encoding at the current position is legal.
    public native @Cast("bool") boolean is_legal();

    // Return the pointer into the string at the current position.
    public native @Cast("const char*") BytePointer utf8_data();

    // Iterator equality operators.
    
    
  }

  // Create a start/end iterator pointing to a string. Note that these methods
  // are static and do NOT create a copy or take ownership of the underlying
  // array.
  public static native @ByVal const_iterator begin(@Cast("const char*") BytePointer utf8_str, int byte_length);
  public static native @ByVal const_iterator begin(String utf8_str, int byte_length);
  public static native @ByVal const_iterator end(@Cast("const char*") BytePointer utf8_str, int byte_length);
  public static native @ByVal const_iterator end(String utf8_str, int byte_length);

  // Converts a utf-8 string to a vector of unicodes.
  // Returns false if the input contains invalid UTF-8, and replaces
  // the rest of the string with a single space.
  public static native @Cast("bool") boolean UTF8ToUnicode(@Cast("const char*") BytePointer utf8_str, IntGenericVector unicodes);
  public static native @Cast("bool") boolean UTF8ToUnicode(String utf8_str, IntGenericVector unicodes);
}

// #endif  // TESSERACT_CCUTIL_UNICHAR_H__


// Parsed from tesseract/host.h

/******************************************************************************
 **  Filename:       Host.h
 **  Purpose:        This is the system independent typedefs and defines
 **  Author:         MN, JG, MD
 **  Version:        5.4.1
 **  History:        11/7/94 MCD received the modification that Lennart made
 **                  to port to 32 bit world and modify this file so that it
 **                  will be shared between platform.
 **                  11/9/94 MCD Make MSW32 subset of MSW. Now MSW means
 **                  MicroSoft Window and MSW32 means the 32 bit worlds
 **                  of MicroSoft Window. Therefore you want the environment
 **                  to be MicroSoft Window and in the 32 bit world -
 **                  _WIN32 must be defined by your compiler.
 **                  11/30/94 MCD Incorporated comments received for more
 **                  readability and the missing typedef for FLOAT.
 **                  12/1/94 MCD Added PFVOID typedef
 **                  5/1/95 MCD. Made many changes based on the inputs.
 **                  Changes:
 **                  1) Rearrange the #ifdef so that there're definitions for
 **                  particular platforms.
 **                  2) Took out the #define for computer and environment
 **                  that developer can uncomment
 **                  3) Added __OLDCODE__ where the defines will be
 **                  obsoleted in the next version and advise not to use.
 **                  4) Added the definitions for the following:
 **                  FILE_HANDLE, MEMORY_HANDLE, BOOL8,
 **                  MAX_INT8, MAX_INT16, MAX_INT32, MAX_UINT8
 **                  MAX_UINT16, MAX_UINT32, MAX_FLOAT32
 **                 06/19/96 MCD. Took out MAX_FLOAT32
 **                 07/15/96 MCD. Fixed the comments error
 **                 Add back BOOL8.
 **
 **  (c) Copyright Hewlett-Packard Company, 1988-1996.
 ** Licensed under the Apache License, Version 2.0 (the "License");
 ** you may not use this file except in compliance with the License.
 ** You may obtain a copy of the License at
 ** http://www.apache.org/licenses/LICENSE-2.0
 ** Unless required by applicable law or agreed to in writing, software
 ** distributed under the License is distributed on an "AS IS" BASIS,
 ** WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 ** See the License for the specific language governing permissions and
 ** limitations under the License.
 */

// #ifndef   __HOST__
// #define   __HOST__

/******************************************************************************
 **                                IMPORTANT!!!                                                                                                                 **
 **                                                                                                                                                                              **
 ** Defines either _WIN32, __MAC__, __UNIX__, __OS2__, __PM__ to
 ** use the specified definitions indicated below in the preprocessor settings.                                                        **
 **                                                                                                                                                                              **
 ** Also define either  __FarProc__ or  __FarData__  and __MOTO__ to use the
 ** specified definitions indicated below in the preprocessor settings.                                                                        **
 **                                                                                                                                                                             **
 ** If a preprocessor settings is not allow in the compiler that is being use,
 ** then it is recommended that a "platform.h" is created with the definition
 ** of the computer and/or operating system.
 ******************************************************************************/

// #include "platform.h"
/* _WIN32 */
// #ifdef _WIN32
// #include <windows.h>
// #include <winbase.h>             // winbase.h contains windows.h
// #endif

/********************************************************/
/* __MAC__ */
// #ifdef __MAC__
// #include <Types.h>
/*----------------------------*/
/*----------------------------*/
// #endif
/********************************************************/
// #if defined(__UNIX__) || defined( __DOS__ ) || defined(__OS2__) || defined(__PM__)
/*----------------------------*/
/* FarProc and FarData */
/*----------------------------*/
/*----------------------------*/
// #endif
/*****************************************************************************
 **
 **                      Standard GHC Definitions
 **
 *****************************************************************************/

// #ifdef __MOTO__
// #define __NATIVE__   MOTO
// #else
// #define __NATIVE__   INTEL
// #endif

//typedef HANDLE FD*  PHANDLE;

// definitions of portable data types (numbers and characters)
// #if (_MSC_VER >= 1200)            //%%% vkr for VC 6.0
// #else
// #endif                           //%%% vkr for VC 6.0

public static final String INT32FORMAT = "%d";
public static final String INT64FORMAT = "%lld";

public static final int MAX_INT8 =  0x7f;
public static final int MAX_INT16 = 0x7fff;
public static final int MAX_INT32 = 0x7fffffff;
public static final int MAX_UINT8 = 0xff;
public static final int MAX_UINT16 =  0xffff;
public static final int MAX_UINT32 =  0xffffffff;
public static final double MAX_FLOAT32 = ((float)3.40282347e+38);

public static final int MIN_INT8 =  0x80;
public static final int MIN_INT16 = 0x8000;
public static final int MIN_INT32 = 0x80000000;
public static final int MIN_UINT8 = 0x00;
public static final int MIN_UINT16 =  0x0000;
public static final int MIN_UINT32 =  0x00000000;
public static final double MIN_FLOAT32 = ((float)1.17549435e-38);

// Defines
// #ifndef TRUE
public static final int TRUE =            1;
// #endif

// #ifndef FALSE
public static final int FALSE =           0;
// #endif

// #ifndef NULL
public static final long NULL =            0L;
// #endif

// Return true if x is within tolerance of y

// #endif


// Parsed from tesseract/tesscallback.h

///////////////////////////////////////////////////////////////////////
// File:        tesscallback.h
// Description: classes and functions to replace pointer-to-functions
// Author:      Samuel Charron
//
// (C) Copyright 2006, Google Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
///////////////////////////////////////////////////////////////////////

// #ifndef _TESS_CALLBACK_SPECIALIZATIONS_H
// #define _TESS_CALLBACK_SPECIALIZATIONS_H

// #include "host.h"  // For NULL.

public static class TessCallbackUtils_ extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public TessCallbackUtils_() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public TessCallbackUtils_(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TessCallbackUtils_(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public TessCallbackUtils_ position(int position) {
        return (TessCallbackUtils_)super.position(position);
    }

  
}


public static class TessClosure extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TessClosure(Pointer p) { super(p); }

  public native void Run();
}

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif



// Specified by TR1 [4.7.2] Reference modifications.

// Identity<T>::type is a typedef of T. Useful for preventing the
// compiler from inferring the type of an argument in templates.

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

@Name("TessCallback1<char>") public static class CharClearCallback extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CharClearCallback() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CharClearCallback(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CharClearCallback(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CharClearCallback position(int position) {
        return (CharClearCallback)super.position(position);
    }

  @Virtual(true) public native void Run(@Cast("char") byte arg0);
}

@Name("TessCallback1<STRING>") public static class StringClearCallback extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public StringClearCallback() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public StringClearCallback(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public StringClearCallback(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public StringClearCallback position(int position) {
        return (StringClearCallback)super.position(position);
    }

  @Virtual(true) public native void Run(@ByVal STRING arg0);
}

@Name("TessCallback1<int>") public static class IntClearCallback extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public IntClearCallback() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public IntClearCallback(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public IntClearCallback(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public IntClearCallback position(int position) {
        return (IntClearCallback)super.position(position);
    }

  @Virtual(true) public native void Run(int arg0);
}

@Name("TessResultCallback1<bool,int>") public static class DeleteCallback extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public DeleteCallback() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public DeleteCallback(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DeleteCallback(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public DeleteCallback position(int position) {
        return (DeleteCallback)super.position(position);
    }

  @Virtual(true) public native @Cast("bool") boolean Run(int arg0);
}

@Name("TessResultCallback2<bool,char const&,char const&>") public static class CharCompareCallback extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CharCompareCallback() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CharCompareCallback(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CharCompareCallback(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CharCompareCallback position(int position) {
        return (CharCompareCallback)super.position(position);
    }

  @Virtual(true) public native @Cast("bool") boolean Run(@Cast("char const*") @ByRef BytePointer arg0,@Cast("char const*") @ByRef BytePointer arg1);
}

@Name("TessResultCallback2<bool,FILE*,char const&>") public static class CharWriteCallback extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CharWriteCallback() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CharWriteCallback(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CharWriteCallback(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CharWriteCallback position(int position) {
        return (CharWriteCallback)super.position(position);
    }

  @Virtual(true) public native @Cast("bool") boolean Run(@Cast("FILE*") Pointer arg0,@Cast("char const*") @ByRef BytePointer arg1);
}

@Name("TessResultCallback2<bool,STRING const&,STRING const&>") public static class StringCompareCallback extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public StringCompareCallback() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public StringCompareCallback(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public StringCompareCallback(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public StringCompareCallback position(int position) {
        return (StringCompareCallback)super.position(position);
    }

  @Virtual(true) public native @Cast("bool") boolean Run(@Const({false, true}) @ByRef STRING arg0,@Const({false, true}) @ByRef STRING arg1);
}

@Name("TessResultCallback2<bool,FILE*,STRING const&>") public static class StringWriteCallback extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public StringWriteCallback() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public StringWriteCallback(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public StringWriteCallback(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public StringWriteCallback position(int position) {
        return (StringWriteCallback)super.position(position);
    }

  @Virtual(true) public native @Cast("bool") boolean Run(@Cast("FILE*") Pointer arg0,@Const({false, true}) @ByRef STRING arg1);
}

@Name("TessResultCallback2<bool,int const&,int const&>") public static class IntCompareCallback extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public IntCompareCallback() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public IntCompareCallback(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public IntCompareCallback(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public IntCompareCallback position(int position) {
        return (IntCompareCallback)super.position(position);
    }

  @Virtual(true) public native @Cast("bool") boolean Run(@Const({false, true}) @ByRef IntPointer arg0,@Const({false, true}) @ByRef IntPointer arg1);
}

@Name("TessResultCallback2<bool,FILE*,int const&>") public static class IntWriteCallback extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public IntWriteCallback() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public IntWriteCallback(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public IntWriteCallback(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public IntWriteCallback position(int position) {
        return (IntWriteCallback)super.position(position);
    }

  @Virtual(true) public native @Cast("bool") boolean Run(@Cast("FILE*") Pointer arg0,@Const({false, true}) @ByRef IntPointer arg1);
}

@Name("TessCallback3<const UNICHARSET&,int,PAGE_RES*>") public static class TruthCallback3 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public TruthCallback3() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public TruthCallback3(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TruthCallback3(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public TruthCallback3 position(int position) {
        return (TruthCallback3)super.position(position);
    }

  @Virtual(true) public native void Run(@Const @ByRef UNICHARSET arg0,int arg1,PAGE_RES arg2);
}

@Name("TessResultCallback3<bool,FILE*,char*,bool>") public static class CharReadCallback extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CharReadCallback() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CharReadCallback(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CharReadCallback(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CharReadCallback position(int position) {
        return (CharReadCallback)super.position(position);
    }

  @Virtual(true) public native @Cast("bool") boolean Run(@Cast("FILE*") Pointer arg0,@Cast("char*") BytePointer arg1,@Cast("bool") boolean arg2);
}

@Name("TessResultCallback3<bool,FILE*,STRING*,bool>") public static class StringReadCallback extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public StringReadCallback() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public StringReadCallback(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public StringReadCallback(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public StringReadCallback position(int position) {
        return (StringReadCallback)super.position(position);
    }

  @Virtual(true) public native @Cast("bool") boolean Run(@Cast("FILE*") Pointer arg0,STRING arg1,@Cast("bool") boolean arg2);
}

@Name("TessResultCallback3<bool,FILE*,int*,bool>") public static class IntReadCallback extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public IntReadCallback() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public IntReadCallback(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public IntReadCallback(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public IntReadCallback position(int position) {
        return (IntReadCallback)super.position(position);
    }

  @Virtual(true) public native @Cast("bool") boolean Run(@Cast("FILE*") Pointer arg0,IntPointer arg1,@Cast("bool") boolean arg2);
}

@Name("TessCallback4<const UNICHARSET&,int,tesseract::PageIterator*,Pix*>") public static class TruthCallback4 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public TruthCallback4() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public TruthCallback4(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TruthCallback4(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public TruthCallback4 position(int position) {
        return (TruthCallback4)super.position(position);
    }

  @Virtual(true) public native void Run(@Const @ByRef UNICHARSET arg0,int arg1,PageIterator arg2,PIX arg3);
}

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #ifndef SWIG
// #endif

// #endif /* _TESS_CALLBACK_SPECIALIZATIONS_H */


// Parsed from tesseract/publictypes.h

///////////////////////////////////////////////////////////////////////
// File:        publictypes.h
// Description: Types used in both the API and internally
// Author:      Ray Smith
// Created:     Wed Mar 03 09:22:53 PST 2010
//
// (C) Copyright 2010, Google Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
///////////////////////////////////////////////////////////////////////

// #ifndef TESSERACT_CCSTRUCT_PUBLICTYPES_H__
// #define TESSERACT_CCSTRUCT_PUBLICTYPES_H__

// This file contains types that are used both by the API and internally
// to Tesseract. In order to decouple the API from Tesseract and prevent cyclic
// dependencies, THIS FILE SHOULD NOT DEPEND ON ANY OTHER PART OF TESSERACT.
// Restated: It is OK for low-level Tesseract files to include publictypes.h,
// but not for the low-level tesseract code to include top-level API code.
// This file should not use other Tesseract types, as that would drag
// their includes into the API-level.
// API-level code should include apitypes.h in preference to this file.

/** Number of printers' points in an inch. The unit of the pointsize return. */
@MemberGetter public static native int kPointsPerInch();

/**
 * Possible types for a POLY_BLOCK or ColPartition.
 * Must be kept in sync with kPBColors in polyblk.cpp and PTIs*Type functions
 * below, as well as kPolyBlockNames in publictypes.cpp.
 * Used extensively by ColPartition, and POLY_BLOCK.
*/
/** enum PolyBlockType */
public static final int
  PT_UNKNOWN = 0,        // Type is not yet known. Keep as the first element.
  PT_FLOWING_TEXT = 1,   // Text that lives inside a column.
  PT_HEADING_TEXT = 2,   // Text that spans more than one column.
  PT_PULLOUT_TEXT = 3,   // Text that is in a cross-column pull-out region.
  PT_EQUATION = 4,       // Partition belonging to an equation region.
  PT_INLINE_EQUATION = 5,  // Partition has inline equation.
  PT_TABLE = 6,          // Partition belonging to a table region.
  PT_VERTICAL_TEXT = 7,  // Text-line runs vertically.
  PT_CAPTION_TEXT = 8,   // Text that belongs to an image.
  PT_FLOWING_IMAGE = 9,  // Image that lives inside a column.
  PT_HEADING_IMAGE = 10,  // Image that spans more than one column.
  PT_PULLOUT_IMAGE = 11,  // Image that is in a cross-column pull-out region.
  PT_HORZ_LINE = 12,      // Horizontal Line.
  PT_VERT_LINE = 13,      // Vertical Line.
  PT_NOISE = 14,          // Lies outside of any column.
  PT_COUNT = 15;

/** Returns true if PolyBlockType is of horizontal line type */
public static native @Cast("bool") boolean PTIsLineType(@Cast("PolyBlockType") int type);
/** Returns true if PolyBlockType is of image type */
public static native @Cast("bool") boolean PTIsImageType(@Cast("PolyBlockType") int type);
/** Returns true if PolyBlockType is of text type */
public static native @Cast("bool") boolean PTIsTextType(@Cast("PolyBlockType") int type);
// Returns true if PolyBlockType is of pullout(inter-column) type
public static native @Cast("bool") boolean PTIsPulloutType(@Cast("PolyBlockType") int type);

/** String name for each block type. Keep in sync with PolyBlockType. */

/**
 *  +------------------+  Orientation Example:
 *  | 1 Aaaa Aaaa Aaaa |  ====================
 *  | Aaa aa aaa aa    |  To left is a diagram of some (1) English and
 *  | aaaaaa A aa aaa. |  (2) Chinese text and a (3) photo credit.
 *  |                2 |
 *  |   #######  c c C |  Upright Latin characters are represented as A and a.
 *  |   #######  c c c |  '<' represents a latin character rotated
 *  | < #######  c c c |      anti-clockwise 90 degrees.
 *  | < #######  c   c |
 *  | < #######  .   c |  Upright Chinese characters are represented C and c.
 *  | 3 #######      c |
 *  +------------------+  NOTA BENE: enum values here should match goodoc.proto
 <p>
 * If you orient your head so that "up" aligns with Orientation,
 * then the characters will appear "right side up" and readable.
 *
 * In the example above, both the English and Chinese paragraphs are oriented
 * so their "up" is the top of the page (page up).  The photo credit is read
 * with one's head turned leftward ("up" is to page left).
 *
 * The values of this enum match the convention of Tesseract's osdetect.h
*/
/** enum tesseract::Orientation */
public static final int
  ORIENTATION_PAGE_UP = 0,
  ORIENTATION_PAGE_RIGHT = 1,
  ORIENTATION_PAGE_DOWN = 2,
  ORIENTATION_PAGE_LEFT = 3;

/**
 * The grapheme clusters within a line of text are laid out logically
 * in this direction, judged when looking at the text line rotated so that
 * its Orientation is "page up".
 *
 * For English text, the writing direction is left-to-right.  For the
 * Chinese text in the above example, the writing direction is top-to-bottom.
*/
/** enum tesseract::WritingDirection */
public static final int
  WRITING_DIRECTION_LEFT_TO_RIGHT = 0,
  WRITING_DIRECTION_RIGHT_TO_LEFT = 1,
  WRITING_DIRECTION_TOP_TO_BOTTOM = 2;

/**
 * The text lines are read in the given sequence.
 *
 * In English, the order is top-to-bottom.
 * In Chinese, vertical text lines are read right-to-left.  Mongolian is
 * written in vertical columns top to bottom like Chinese, but the lines
 * order left-to right.
 *
 * Note that only some combinations make sense.  For example,
 * WRITING_DIRECTION_LEFT_TO_RIGHT implies TEXTLINE_ORDER_TOP_TO_BOTTOM
*/
/** enum tesseract::TextlineOrder */
public static final int
  TEXTLINE_ORDER_LEFT_TO_RIGHT = 0,
  TEXTLINE_ORDER_RIGHT_TO_LEFT = 1,
  TEXTLINE_ORDER_TOP_TO_BOTTOM = 2;

/**
 * Possible modes for page layout analysis. These *must* be kept in order
 * of decreasing amount of layout analysis to be done, except for OSD_ONLY,
 * so that the inequality test macros below work.
*/
/** enum tesseract::PageSegMode */
public static final int
  /** Orientation and script detection only. */
  PSM_OSD_ONLY = 0,
  /** Automatic page segmentation with orientation and
 *  script detection. (OSD) */
  PSM_AUTO_OSD = 1,
  /** Automatic page segmentation, but no OSD, or OCR. */
  PSM_AUTO_ONLY = 2,
  /** Fully automatic page segmentation, but no OSD. */
  PSM_AUTO = 3,
  /** Assume a single column of text of variable sizes. */
  PSM_SINGLE_COLUMN = 4,
  /** Assume a single uniform block of vertically
 *  aligned text. */
  PSM_SINGLE_BLOCK_VERT_TEXT = 5,
  /** Assume a single uniform block of text. (Default.) */
  PSM_SINGLE_BLOCK = 6,
  /** Treat the image as a single text line. */
  PSM_SINGLE_LINE = 7,
  /** Treat the image as a single word. */
  PSM_SINGLE_WORD = 8,
  /** Treat the image as a single word in a circle. */
  PSM_CIRCLE_WORD = 9,
  /** Treat the image as a single character. */
  PSM_SINGLE_CHAR = 10,
  /** Find as much text as possible in no particular order. */
  PSM_SPARSE_TEXT = 11,
  /** Sparse text with orientation and script det. */
  PSM_SPARSE_TEXT_OSD = 12,
  /** Treat the image as a single text line, bypassing
 *  hacks that are Tesseract-specific. */
  PSM_RAW_LINE = 13,

  /** Number of enum entries. */
  PSM_COUNT = 14;

/**
 * Inline functions that act on a PageSegMode to determine whether components of
 * layout analysis are enabled.
 * *Depend critically on the order of elements of PageSegMode.*
 * NOTE that arg is an int for compatibility with INT_PARAM.
*/
@Namespace("tesseract") public static native @Cast("bool") boolean PSM_OSD_ENABLED(int pageseg_mode);
@Namespace("tesseract") public static native @Cast("bool") boolean PSM_ORIENTATION_ENABLED(int pageseg_mode);
@Namespace("tesseract") public static native @Cast("bool") boolean PSM_COL_FIND_ENABLED(int pageseg_mode);
@Namespace("tesseract") public static native @Cast("bool") boolean PSM_SPARSE(int pageseg_mode);
@Namespace("tesseract") public static native @Cast("bool") boolean PSM_BLOCK_FIND_ENABLED(int pageseg_mode);
@Namespace("tesseract") public static native @Cast("bool") boolean PSM_LINE_FIND_ENABLED(int pageseg_mode);
@Namespace("tesseract") public static native @Cast("bool") boolean PSM_WORD_FIND_ENABLED(int pageseg_mode);

/**
 * enum of the elements of the page hierarchy, used in ResultIterator
 * to provide functions that operate on each level without having to
 * have 5x as many functions.
*/
/** enum tesseract::PageIteratorLevel */
public static final int
  RIL_BLOCK = 0,     // Block of text/image/separator line.
  RIL_PARA = 1,      // Paragraph within a block.
  RIL_TEXTLINE = 2,  // Line within a paragraph.
  RIL_WORD = 3,      // Word within a textline.
  RIL_SYMBOL = 4;     // Symbol/character within a word.

/**
 * JUSTIFICATION_UNKNONW
 *   The alignment is not clearly one of the other options.  This could happen
 *   for example if there are only one or two lines of text or the text looks
 *   like source code or poetry.
 *
 * NOTA BENE: Fully justified paragraphs (text aligned to both left and right
 *    margins) are marked by Tesseract with JUSTIFICATION_LEFT if their text
 *    is written with a left-to-right script and with JUSTIFICATION_RIGHT if
 *    their text is written in a right-to-left script.
 *
 * Interpretation for text read in vertical lines:
 *   "Left" is wherever the starting reading position is.
 *
 * JUSTIFICATION_LEFT
 *   Each line, except possibly the first, is flush to the same left tab stop.
 *
 * JUSTIFICATION_CENTER
 *   The text lines of the paragraph are centered about a line going
 *   down through their middle of the text lines.
 *
 * JUSTIFICATION_RIGHT
 *   Each line, except possibly the first, is flush to the same right tab stop.
*/
/** enum tesseract::ParagraphJustification */
public static final int
  JUSTIFICATION_UNKNOWN = 0,
  JUSTIFICATION_LEFT = 1,
  JUSTIFICATION_CENTER = 2,
  JUSTIFICATION_RIGHT = 3;

/**
 * When Tesseract/Cube is initialized we can choose to instantiate/load/run
 * only the Tesseract part, only the Cube part or both along with the combiner.
 * The preference of which engine to use is stored in tessedit_ocr_engine_mode.
 *
 * ATTENTION: When modifying this enum, please make sure to make the
 * appropriate changes to all the enums mirroring it (e.g. OCREngine in
 * cityblock/workflow/detection/detection_storage.proto). Such enums will
 * mention the connection to OcrEngineMode in the comments.
*/
/** enum tesseract::OcrEngineMode */
public static final int
  OEM_TESSERACT_ONLY = 0,           // Run Tesseract only - fastest
  OEM_CUBE_ONLY = 1,                // Run Cube only - better accuracy, but slower
  OEM_TESSERACT_CUBE_COMBINED = 2,  // Run both and combine results - best accuracy
  OEM_DEFAULT = 3;                   // Specify this mode when calling init_*(),
                                // to indicate that any of the above modes
                                // should be automatically inferred from the
                                // variables in the language-specific config,
                                // command-line configs, or if not specified
                                // in any of the above should be set to the
                                // default OEM_TESSERACT_ONLY.

  // namespace tesseract.

// #endif  // TESSERACT_CCSTRUCT_PUBLICTYPES_H__


// Parsed from tesseract/thresholder.h

///////////////////////////////////////////////////////////////////////
// File:        thresholder.h
// Description: Base API for thresolding images in tesseract.
// Author:      Ray Smith
// Created:     Mon May 12 11:00:15 PDT 2008
//
// (C) Copyright 2008, Google Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
///////////////////////////////////////////////////////////////////////

// #ifndef TESSERACT_CCMAIN_THRESHOLDER_H__
// #define TESSERACT_CCMAIN_THRESHOLDER_H__

// #include "platform.h"
// #include "publictypes.h"

/** Base class for all tesseract image thresholding classes.
 *  Specific classes can add new thresholding methods by
 *  overriding ThresholdToPix.
 *  Each instance deals with a single image, but the design is intended to
 *  be useful for multiple calls to SetRectangle and ThresholdTo* if
 *  desired. */
@Namespace("tesseract") @NoOffset public static class ImageThresholder extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ImageThresholder(Pointer p) { super(p); }


  /** Destroy the Pix if there is one, freeing memory. */
  public native void Clear();

  /** Return true if no image has been set. */
  

  /** SetImage makes a copy of all the image data, so it may be deleted
   *  immediately after this call.
   *  Greyscale of 8 and color of 24 or 32 bits per pixel may be given.
   *  Palette color images will not work properly and must be converted to
   *  24 bit.
   *  Binary images of 1 bit per pixel may also be given but they must be
   *  byte packed with the MSB of the first byte being the first pixel, and a
   *  one pixel is WHITE. For binary images set bytes_per_pixel=0. */
  

  /** Store the coordinates of the rectangle to process for later use.
   *  Doesn't actually do any thresholding. */
  

  /** Get enough parameters to be able to rebuild bounding boxes in the
   *  original image (not just within the rectangle).
   *  Left and top are enough with top-down coordinates, but
   *  the height of the rectangle and the image are needed for bottom-up. */
  public native void GetImageSizes(IntPointer left, IntPointer top, IntPointer width, IntPointer height,
                               IntPointer imagewidth, IntPointer imageheight);
  public native void GetImageSizes(IntBuffer left, IntBuffer top, IntBuffer width, IntBuffer height,
                               IntBuffer imagewidth, IntBuffer imageheight);
  public native void GetImageSizes(int[] left, int[] top, int[] width, int[] height,
                               int[] imagewidth, int[] imageheight);

  /** Return true if the source image is color. */
  public native @Cast("bool") boolean IsColor();

  /** Returns true if the source image is binary. */
  public native @Cast("bool") boolean IsBinary();

  public native int GetScaleFactor();

  // Set the resolution of the source image in pixels per inch.
  // This should be called right after SetImage(), and will let us return
  // appropriate font sizes for the text.
  public native void SetSourceYResolution(int ppi);
  public native int GetSourceYResolution();
  public native int GetScaledYResolution();
  // Set the resolution of the source image in pixels per inch, as estimated
  // by the thresholder from the text size found during thresholding.
  // This value will be used to set internal size thresholds during recognition
  // and will not influence the output "point size." The default value is
  // the same as the source resolution. (yres_)
  public native void SetEstimatedResolution(int ppi);
  // Returns the estimated resolution, including any active scaling.
  // This value will be used to set internal size thresholds during recognition.
  public native int GetScaledEstimatedResolution();

  /** Pix vs raw, which to use? Pix is the preferred input for efficiency,
   *  since raw buffers are copied.
   *  SetImage for Pix clones its input, so the source pix may be pixDestroyed
   *  immediately after, but may not go away until after the Thresholder has
   *  finished with it. */
  

  /** Threshold the source image as efficiently as possible to the output Pix.
   *  Creates a Pix and sets pix to point to the resulting pointer.
   *  Caller must use pixDestroy to free the created Pix. */
  public native void ThresholdToPix(@Cast("tesseract::PageSegMode") int pageseg_mode, @Cast("Pix**") PointerPointer pix);
  public native void ThresholdToPix(@Cast("tesseract::PageSegMode") int pageseg_mode, @ByPtrPtr PIX pix);

  // Gets a pix that contains an 8 bit threshold value at each pixel. The
  // returned pix may be an integer reduction of the binary image such that
  // the scale factor may be inferred from the ratio of the sizes, even down
  // to the extreme of a 1x1 pixel thresholds image.
  // Ideally the 8 bit threshold should be the exact threshold used to generate
  // the binary image in ThresholdToPix, but this is not a hard constraint.
  // Returns NULL if the input is binary. PixDestroy after use.
  public native PIX GetPixRectThresholds();

  /** Get a clone/copy of the source image rectangle.
   *  The returned Pix must be pixDestroyed.
   *  This function will be used in the future by the page layout analysis, and
   *  the layout analysis that uses it will only be available with Leptonica,
   *  so there is no raw equivalent. */
  

  // Get a clone/copy of the source image rectangle, reduced to greyscale,
  // and at the same resolution as the output binary.
  // The returned Pix must be pixDestroyed.
  // Provided to the classifier to extract features from the greyscale image.
  
}

  // namespace tesseract.

// #endif  // TESSERACT_CCMAIN_THRESHOLDER_H__


// Parsed from tesseract/pageiterator.h

///////////////////////////////////////////////////////////////////////
// File:        pageiterator.h
// Description: Iterator for tesseract page structure that avoids using
//              tesseract internal data structures.
// Author:      Ray Smith
// Created:     Fri Feb 26 11:01:06 PST 2010
//
// (C) Copyright 2010, Google Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
///////////////////////////////////////////////////////////////////////

// #ifndef TESSERACT_CCMAIN_PAGEITERATOR_H__
// #define TESSERACT_CCMAIN_PAGEITERATOR_H__

// #include "publictypes.h"
// #include "platform.h"

@Opaque public static class BlamerBundle extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public BlamerBundle() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public BlamerBundle(Pointer p) { super(p); }
}
@Opaque public static class C_BLOB_IT extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public C_BLOB_IT() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public C_BLOB_IT(Pointer p) { super(p); }
}
@Opaque public static class PAGE_RES extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public PAGE_RES() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public PAGE_RES(Pointer p) { super(p); }
}
@Opaque public static class PAGE_RES_IT extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public PAGE_RES_IT() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public PAGE_RES_IT(Pointer p) { super(p); }
}
@Opaque public static class WERD extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public WERD() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public WERD(Pointer p) { super(p); }
}

@Namespace("tesseract") @Opaque public static class Tesseract extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public Tesseract() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Tesseract(Pointer p) { super(p); }
}

/**
 * Class to iterate over tesseract page structure, providing access to all
 * levels of the page hierarchy, without including any tesseract headers or
 * having to handle any tesseract structures.
 * WARNING! This class points to data held within the TessBaseAPI class, and
 * therefore can only be used while the TessBaseAPI class still exists and
 * has not been subjected to a call of Init, SetImage, Recognize, Clear, End
 * DetectOS, or anything else that changes the internal PAGE_RES.
 * See apitypes.h for the definition of PageIteratorLevel.
 * See also ResultIterator, derived from PageIterator, which adds in the
 * ability to access OCR output with text-specific methods.
 */

@Namespace("tesseract") @NoOffset public static class PageIterator extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public PageIterator(Pointer p) { super(p); }

  /**
   * page_res and tesseract come directly from the BaseAPI.
   * The rectangle parameters are copied indirectly from the Thresholder,
   * via the BaseAPI. They represent the coordinates of some rectangle in an
   * original image (in top-left-origin coordinates) and therefore the top-left
   * needs to be added to any output boxes in order to specify coordinates
   * in the original image. See TessBaseAPI::SetRectangle.
   * The scale and scaled_yres are in case the Thresholder scaled the image
   * rectangle prior to thresholding. Any coordinates in tesseract's image
   * must be divided by scale before adding (rect_left, rect_top).
   * The scaled_yres indicates the effective resolution of the binary image
   * that tesseract has been given by the Thresholder.
   * After the constructor, Begin has already been called.
   */
  public PageIterator(PAGE_RES page_res, Tesseract tesseract,
                 int scale, int scaled_yres,
                 int rect_left, int rect_top,
                 int rect_width, int rect_height) { super((Pointer)null); allocate(page_res, tesseract, scale, scaled_yres, rect_left, rect_top, rect_width, rect_height); }
  private native void allocate(PAGE_RES page_res, Tesseract tesseract,
                 int scale, int scaled_yres,
                 int rect_left, int rect_top,
                 int rect_width, int rect_height);

  /**
   * Page/ResultIterators may be copied! This makes it possible to iterate over
   * all the objects at a lower level, while maintaining an iterator to
   * objects at a higher level. These constructors DO NOT CALL Begin, so
   * iterations will continue from the location of src.
   */
  public PageIterator(@Const @ByRef PageIterator src) { super((Pointer)null); allocate(src); }
  private native void allocate(@Const @ByRef PageIterator src);
  public native @Const @ByRef @Name("operator =") PageIterator put(@Const @ByRef PageIterator src);

  /** Are we positioned at the same location as other? */
  public native @Cast("bool") boolean PositionedAtSameWord(@Const PAGE_RES_IT other);

  // ============= Moving around within the page ============.

  /**
   * Moves the iterator to point to the start of the page to begin an
   * iteration.
   */
  public native void Begin();

  /**
   * Moves the iterator to the beginning of the paragraph.
   * This class implements this functionality by moving it to the zero indexed
   * blob of the first (leftmost) word on the first row of the paragraph.
   */
  public native void RestartParagraph();

  /**
   * Return whether this iterator points anywhere in the first textline of a
   * paragraph.
   */
  public native @Cast("bool") boolean IsWithinFirstTextlineOfParagraph();

  /**
   * Moves the iterator to the beginning of the text line.
   * This class implements this functionality by moving it to the zero indexed
   * blob of the first (leftmost) word of the row.
   */
  public native void RestartRow();

  /**
   * Moves to the start of the next object at the given level in the
   * page hierarchy, and returns false if the end of the page was reached.
   * NOTE that RIL_SYMBOL will skip non-text blocks, but all other
   * PageIteratorLevel level values will visit each non-text block once.
   * Think of non text blocks as containing a single para, with a single line,
   * with a single imaginary word.
   * Calls to Next with different levels may be freely intermixed.
   * This function iterates words in right-to-left scripts correctly, if
   * the appropriate language has been loaded into Tesseract.
   */
  public native @Cast("bool") boolean Next(@Cast("tesseract::PageIteratorLevel") int level);

  /**
   * Returns true if the iterator is at the start of an object at the given
   * level.
   *
   * For instance, suppose an iterator it is pointed to the first symbol of the
   * first word of the third line of the second paragraph of the first block in
   * a page, then:
   *   it.IsAtBeginningOf(RIL_BLOCK) = false
   *   it.IsAtBeginningOf(RIL_PARA) = false
   *   it.IsAtBeginningOf(RIL_TEXTLINE) = true
   *   it.IsAtBeginningOf(RIL_WORD) = true
   *   it.IsAtBeginningOf(RIL_SYMBOL) = true
   */
  public native @Cast("bool") boolean IsAtBeginningOf(@Cast("tesseract::PageIteratorLevel") int level);

  /**
   * Returns whether the iterator is positioned at the last element in a
   * given level. (e.g. the last word in a line, the last line in a block)
   *
   *     Here's some two-paragraph example
   *   text.  It starts off innocuously
   *   enough but quickly turns bizarre.
   *     The author inserts a cornucopia
   *   of words to guard against confused
   *   references.
   *
   * Now take an iterator it pointed to the start of "bizarre."
   *  it.IsAtFinalElement(RIL_PARA, RIL_SYMBOL) = false
   *  it.IsAtFinalElement(RIL_PARA, RIL_WORD) = true
   *  it.IsAtFinalElement(RIL_BLOCK, RIL_WORD) = false
   */
  public native @Cast("bool") boolean IsAtFinalElement(@Cast("tesseract::PageIteratorLevel") int level,
                                  @Cast("tesseract::PageIteratorLevel") int element);

  /**
   * Returns whether this iterator is positioned
   *   before other:   -1
   *   equal to other:  0
   *   after other:     1
   */
  public native int Cmp(@Const @ByRef PageIterator other);

  // ============= Accessing data ==============.
  // Coordinate system:
  // Integer coordinates are at the cracks between the pixels.
  // The top-left corner of the top-left pixel in the image is at (0,0).
  // The bottom-right corner of the bottom-right pixel in the image is at
  // (width, height).
  // Every bounding box goes from the top-left of the top-left contained
  // pixel to the bottom-right of the bottom-right contained pixel, so
  // the bounding box of the single top-left pixel in the image is:
  // (0,0)->(1,1).
  // If an image rectangle has been set in the API, then returned coordinates
  // relate to the original (full) image, rather than the rectangle.

  /**
   * Controls what to include in a bounding box. Bounding boxes of all levels
   * between RIL_WORD and RIL_BLOCK can include or exclude potential diacritics.
   * Between layout analysis and recognition, it isn't known where all
   * diacritics belong, so this control is used to include or exclude some
   * diacritics that are above or below the main body of the word. In most cases
   * where the placement is obvious, and after recognition, it doesn't make as
   * much difference, as the diacritics will already be included in the word.
   */
  public native void SetBoundingBoxComponents(@Cast("bool") boolean include_upper_dots,
                                  @Cast("bool") boolean include_lower_dots);

  /**
   * Returns the bounding rectangle of the current object at the given level.
   * See comment on coordinate system above.
   * Returns false if there is no such object at the current position.
   * The returned bounding box is guaranteed to match the size and position
   * of the image returned by GetBinaryImage, but may clip foreground pixels
   * from a grey image. The padding argument to GetImage can be used to expand
   * the image to include more foreground pixels. See GetImage below.
   */
  public native @Cast("bool") boolean BoundingBox(@Cast("tesseract::PageIteratorLevel") int level,
                     IntPointer left, IntPointer top, IntPointer right, IntPointer bottom);
  public native @Cast("bool") boolean BoundingBox(@Cast("tesseract::PageIteratorLevel") int level,
                     IntBuffer left, IntBuffer top, IntBuffer right, IntBuffer bottom);
  public native @Cast("bool") boolean BoundingBox(@Cast("tesseract::PageIteratorLevel") int level,
                     int[] left, int[] top, int[] right, int[] bottom);
  public native @Cast("bool") boolean BoundingBox(@Cast("tesseract::PageIteratorLevel") int level, int padding,
                     IntPointer left, IntPointer top, IntPointer right, IntPointer bottom);
  public native @Cast("bool") boolean BoundingBox(@Cast("tesseract::PageIteratorLevel") int level, int padding,
                     IntBuffer left, IntBuffer top, IntBuffer right, IntBuffer bottom);
  public native @Cast("bool") boolean BoundingBox(@Cast("tesseract::PageIteratorLevel") int level, int padding,
                     int[] left, int[] top, int[] right, int[] bottom);
  /**
   * Returns the bounding rectangle of the object in a coordinate system of the
   * working image rectangle having its origin at (rect_left_, rect_top_) with
   * respect to the original image and is scaled by a factor scale_.
   */
  public native @Cast("bool") boolean BoundingBoxInternal(@Cast("tesseract::PageIteratorLevel") int level,
                             IntPointer left, IntPointer top, IntPointer right, IntPointer bottom);
  public native @Cast("bool") boolean BoundingBoxInternal(@Cast("tesseract::PageIteratorLevel") int level,
                             IntBuffer left, IntBuffer top, IntBuffer right, IntBuffer bottom);
  public native @Cast("bool") boolean BoundingBoxInternal(@Cast("tesseract::PageIteratorLevel") int level,
                             int[] left, int[] top, int[] right, int[] bottom);

  /** Returns whether there is no object of a given level. */
  public native @Cast("bool") boolean Empty(@Cast("tesseract::PageIteratorLevel") int level);

  /**
   * Returns the type of the current block. See apitypes.h for
   * PolyBlockType.
   */
  public native @Cast("PolyBlockType") int BlockType();

  /**
   * Returns the polygon outline of the current block. The returned Pta must
   * be ptaDestroy-ed after use. Note that the returned Pta lists the vertices
   * of the polygon, and the last edge is the line segment between the last
   * point and the first point. NULL will be returned if the iterator is
   * at the end of the document or layout analysis was not used.
   */
  public native PTA BlockPolygon();

  /**
   * Returns a binary image of the current object at the given level.
   * The position and size match the return from BoundingBoxInternal, and so
   * this could be upscaled with respect to the original input image.
   * Use pixDestroy to delete the image after use.
   */
  public native PIX GetBinaryImage(@Cast("tesseract::PageIteratorLevel") int level);

  /**
   * Returns an image of the current object at the given level in greyscale
   * if available in the input. To guarantee a binary image use BinaryImage.
   * NOTE that in order to give the best possible image, the bounds are
   * expanded slightly over the binary connected component, by the supplied
   * padding, so the top-left position of the returned image is returned
   * in (left,top). These will most likely not match the coordinates
   * returned by BoundingBox.
   * If you do not supply an original image, you will get a binary one.
   * Use pixDestroy to delete the image after use.
   */
  public native PIX GetImage(@Cast("tesseract::PageIteratorLevel") int level, int padding, PIX original_img,
                  IntPointer left, IntPointer top);
  public native PIX GetImage(@Cast("tesseract::PageIteratorLevel") int level, int padding, PIX original_img,
                  IntBuffer left, IntBuffer top);
  public native PIX GetImage(@Cast("tesseract::PageIteratorLevel") int level, int padding, PIX original_img,
                  int[] left, int[] top);

  /**
   * Returns the baseline of the current object at the given level.
   * The baseline is the line that passes through (x1, y1) and (x2, y2).
   * WARNING: with vertical text, baselines may be vertical!
   * Returns false if there is no baseline at the current position.
   */
  public native @Cast("bool") boolean Baseline(@Cast("tesseract::PageIteratorLevel") int level,
                  IntPointer x1, IntPointer y1, IntPointer x2, IntPointer y2);
  public native @Cast("bool") boolean Baseline(@Cast("tesseract::PageIteratorLevel") int level,
                  IntBuffer x1, IntBuffer y1, IntBuffer x2, IntBuffer y2);
  public native @Cast("bool") boolean Baseline(@Cast("tesseract::PageIteratorLevel") int level,
                  int[] x1, int[] y1, int[] x2, int[] y2);

  /**
   * Returns orientation for the block the iterator points to.
   *   orientation, writing_direction, textline_order: see publictypes.h
   *   deskew_angle: after rotating the block so the text orientation is
   *                 upright, how many radians does one have to rotate the
   *                 block anti-clockwise for it to be level?
   *                   -Pi/4 <= deskew_angle <= Pi/4
   */
  public native void Orientation(@Cast("tesseract::Orientation*") IntPointer orientation,
                     @Cast("tesseract::WritingDirection*") IntPointer writing_direction,
                     @Cast("tesseract::TextlineOrder*") IntPointer textline_order,
                     FloatPointer deskew_angle);
  public native void Orientation(@Cast("tesseract::Orientation*") IntBuffer orientation,
                     @Cast("tesseract::WritingDirection*") IntBuffer writing_direction,
                     @Cast("tesseract::TextlineOrder*") IntBuffer textline_order,
                     FloatBuffer deskew_angle);
  public native void Orientation(@Cast("tesseract::Orientation*") int[] orientation,
                     @Cast("tesseract::WritingDirection*") int[] writing_direction,
                     @Cast("tesseract::TextlineOrder*") int[] textline_order,
                     float[] deskew_angle);

  /**
   * Returns information about the current paragraph, if available.
   *
   *   justification -
   *     LEFT if ragged right, or fully justified and script is left-to-right.
   *     RIGHT if ragged left, or fully justified and script is right-to-left.
   *     unknown if it looks like source code or we have very few lines.
   *   is_list_item -
   *     true if we believe this is a member of an ordered or unordered list.
   *   is_crown -
   *     true if the first line of the paragraph is aligned with the other
   *     lines of the paragraph even though subsequent paragraphs have first
   *     line indents.  This typically indicates that this is the continuation
   *     of a previous paragraph or that it is the very first paragraph in
   *     the chapter.
   *   first_line_indent -
   *     For LEFT aligned paragraphs, the first text line of paragraphs of
   *     this kind are indented this many pixels from the left edge of the
   *     rest of the paragraph.
   *     for RIGHT aligned paragraphs, the first text line of paragraphs of
   *     this kind are indented this many pixels from the right edge of the
   *     rest of the paragraph.
   *     NOTE 1: This value may be negative.
   *     NOTE 2: if *is_crown == true, the first line of this paragraph is
   *             actually flush, and first_line_indent is set to the "common"
   *             first_line_indent for subsequent paragraphs in this block
   *             of text.
   */
  public native void ParagraphInfo(@Cast("tesseract::ParagraphJustification*") IntPointer justification,
                       @Cast("bool*") BoolPointer is_list_item,
                       @Cast("bool*") BoolPointer is_crown,
                       IntPointer first_line_indent);
  public native void ParagraphInfo(@Cast("tesseract::ParagraphJustification*") IntBuffer justification,
                       @Cast("bool*") boolean[] is_list_item,
                       @Cast("bool*") boolean[] is_crown,
                       IntBuffer first_line_indent);
  public native void ParagraphInfo(@Cast("tesseract::ParagraphJustification*") int[] justification,
                       @Cast("bool*") BoolPointer is_list_item,
                       @Cast("bool*") BoolPointer is_crown,
                       int[] first_line_indent);
  public native void ParagraphInfo(@Cast("tesseract::ParagraphJustification*") IntPointer justification,
                       @Cast("bool*") boolean[] is_list_item,
                       @Cast("bool*") boolean[] is_crown,
                       IntPointer first_line_indent);
  public native void ParagraphInfo(@Cast("tesseract::ParagraphJustification*") IntBuffer justification,
                       @Cast("bool*") BoolPointer is_list_item,
                       @Cast("bool*") BoolPointer is_crown,
                       IntBuffer first_line_indent);
  public native void ParagraphInfo(@Cast("tesseract::ParagraphJustification*") int[] justification,
                       @Cast("bool*") boolean[] is_list_item,
                       @Cast("bool*") boolean[] is_crown,
                       int[] first_line_indent);

  // If the current WERD_RES (it_->word()) is not NULL, sets the BlamerBundle
  // of the current word to the given pointer (takes ownership of the pointer)
  // and returns true.
  // Can only be used when iterating on the word level.
  public native @Cast("bool") boolean SetWordBlamerBundle(BlamerBundle blamer_bundle);
}

  // namespace tesseract.

// #endif  // TESSERACT_CCMAIN_PAGEITERATOR_H__


// Parsed from tesseract/ltrresultiterator.h

///////////////////////////////////////////////////////////////////////
// File:        ltrresultiterator.h
// Description: Iterator for tesseract results in strict left-to-right
//              order that avoids using tesseract internal data structures.
// Author:      Ray Smith
// Created:     Fri Feb 26 11:01:06 PST 2010
//
// (C) Copyright 2010, Google Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
///////////////////////////////////////////////////////////////////////

// #ifndef TESSERACT_CCMAIN_LTR_RESULT_ITERATOR_H__
// #define TESSERACT_CCMAIN_LTR_RESULT_ITERATOR_H__

// #include "platform.h"
// #include "pageiterator.h"
// #include "unichar.h"

@Opaque public static class BLOB_CHOICE_IT extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public BLOB_CHOICE_IT() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public BLOB_CHOICE_IT(Pointer p) { super(p); }
}
@Opaque public static class WERD_RES extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public WERD_RES() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public WERD_RES(Pointer p) { super(p); }
}

// Class to iterate over tesseract results, providing access to all levels
// of the page hierarchy, without including any tesseract headers or having
// to handle any tesseract structures.
// WARNING! This class points to data held within the TessBaseAPI class, and
// therefore can only be used while the TessBaseAPI class still exists and
// has not been subjected to a call of Init, SetImage, Recognize, Clear, End
// DetectOS, or anything else that changes the internal PAGE_RES.
// See apitypes.h for the definition of PageIteratorLevel.
// See also base class PageIterator, which contains the bulk of the interface.
// LTRResultIterator adds text-specific methods for access to OCR output.

@Namespace("tesseract") @NoOffset public static class LTRResultIterator extends PageIterator {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public LTRResultIterator(Pointer p) { super(p); }

  // page_res and tesseract come directly from the BaseAPI.
  // The rectangle parameters are copied indirectly from the Thresholder,
  // via the BaseAPI. They represent the coordinates of some rectangle in an
  // original image (in top-left-origin coordinates) and therefore the top-left
  // needs to be added to any output boxes in order to specify coordinates
  // in the original image. See TessBaseAPI::SetRectangle.
  // The scale and scaled_yres are in case the Thresholder scaled the image
  // rectangle prior to thresholding. Any coordinates in tesseract's image
  // must be divided by scale before adding (rect_left, rect_top).
  // The scaled_yres indicates the effective resolution of the binary image
  // that tesseract has been given by the Thresholder.
  // After the constructor, Begin has already been called.
  public LTRResultIterator(PAGE_RES page_res, Tesseract tesseract,
                      int scale, int scaled_yres,
                      int rect_left, int rect_top,
                      int rect_width, int rect_height) { super((Pointer)null); allocate(page_res, tesseract, scale, scaled_yres, rect_left, rect_top, rect_width, rect_height); }
  private native void allocate(PAGE_RES page_res, Tesseract tesseract,
                      int scale, int scaled_yres,
                      int rect_left, int rect_top,
                      int rect_width, int rect_height);

  // LTRResultIterators may be copied! This makes it possible to iterate over
  // all the objects at a lower level, while maintaining an iterator to
  // objects at a higher level. These constructors DO NOT CALL Begin, so
  // iterations will continue from the location of src.
  // TODO: For now the copy constructor and operator= only need the base class
  // versions, but if new data members are added, don't forget to add them!

  // ============= Moving around within the page ============.

  // See PageIterator.

  // ============= Accessing data ==============.

  // Returns the null terminated UTF-8 encoded text string for the current
  // object at the given level. Use delete [] to free after use.
  public native @Cast("char*") BytePointer GetUTF8Text(@Cast("tesseract::PageIteratorLevel") int level);

  // Set the string inserted at the end of each text line. "\n" by default.
  public native void SetLineSeparator(@Cast("const char*") BytePointer new_line);
  public native void SetLineSeparator(String new_line);

  // Set the string inserted at the end of each paragraph. "\n" by default.
  public native void SetParagraphSeparator(@Cast("const char*") BytePointer new_para);
  public native void SetParagraphSeparator(String new_para);

  // Returns the mean confidence of the current object at the given level.
  // The number should be interpreted as a percent probability. (0.0f-100.0f)
  public native float Confidence(@Cast("tesseract::PageIteratorLevel") int level);

  // ============= Functions that refer to words only ============.

  // Returns the font attributes of the current word. If iterating at a higher
  // level object than words, eg textlines, then this will return the
  // attributes of the first word in that textline.
  // The actual return value is a string representing a font name. It points
  // to an internal table and SHOULD NOT BE DELETED. Lifespan is the same as
  // the iterator itself, ie rendered invalid by various members of
  // TessBaseAPI, including Init, SetImage, End or deleting the TessBaseAPI.
  // Pointsize is returned in printers points (1/72 inch.)
  public native @Cast("const char*") BytePointer WordFontAttributes(@Cast("bool*") BoolPointer is_bold,
                                   @Cast("bool*") BoolPointer is_italic,
                                   @Cast("bool*") BoolPointer is_underlined,
                                   @Cast("bool*") BoolPointer is_monospace,
                                   @Cast("bool*") BoolPointer is_serif,
                                   @Cast("bool*") BoolPointer is_smallcaps,
                                   IntPointer pointsize,
                                   IntPointer font_id);
  public native String WordFontAttributes(@Cast("bool*") boolean[] is_bold,
                                   @Cast("bool*") boolean[] is_italic,
                                   @Cast("bool*") boolean[] is_underlined,
                                   @Cast("bool*") boolean[] is_monospace,
                                   @Cast("bool*") boolean[] is_serif,
                                   @Cast("bool*") boolean[] is_smallcaps,
                                   IntBuffer pointsize,
                                   IntBuffer font_id);
  public native @Cast("const char*") BytePointer WordFontAttributes(@Cast("bool*") BoolPointer is_bold,
                                   @Cast("bool*") BoolPointer is_italic,
                                   @Cast("bool*") BoolPointer is_underlined,
                                   @Cast("bool*") BoolPointer is_monospace,
                                   @Cast("bool*") BoolPointer is_serif,
                                   @Cast("bool*") BoolPointer is_smallcaps,
                                   int[] pointsize,
                                   int[] font_id);
  public native String WordFontAttributes(@Cast("bool*") boolean[] is_bold,
                                   @Cast("bool*") boolean[] is_italic,
                                   @Cast("bool*") boolean[] is_underlined,
                                   @Cast("bool*") boolean[] is_monospace,
                                   @Cast("bool*") boolean[] is_serif,
                                   @Cast("bool*") boolean[] is_smallcaps,
                                   IntPointer pointsize,
                                   IntPointer font_id);
  public native @Cast("const char*") BytePointer WordFontAttributes(@Cast("bool*") BoolPointer is_bold,
                                   @Cast("bool*") BoolPointer is_italic,
                                   @Cast("bool*") BoolPointer is_underlined,
                                   @Cast("bool*") BoolPointer is_monospace,
                                   @Cast("bool*") BoolPointer is_serif,
                                   @Cast("bool*") BoolPointer is_smallcaps,
                                   IntBuffer pointsize,
                                   IntBuffer font_id);
  public native String WordFontAttributes(@Cast("bool*") boolean[] is_bold,
                                   @Cast("bool*") boolean[] is_italic,
                                   @Cast("bool*") boolean[] is_underlined,
                                   @Cast("bool*") boolean[] is_monospace,
                                   @Cast("bool*") boolean[] is_serif,
                                   @Cast("bool*") boolean[] is_smallcaps,
                                   int[] pointsize,
                                   int[] font_id);

  // Return the name of the language used to recognize this word.
  // On error, NULL.  Do not delete this pointer.
  public native @Cast("const char*") BytePointer WordRecognitionLanguage();

  // Return the overall directionality of this word.
  public native @Cast("StrongScriptDirection") int WordDirection();

  // Returns true if the current word was found in a dictionary.
  public native @Cast("bool") boolean WordIsFromDictionary();

  // Returns true if the current word is numeric.
  public native @Cast("bool") boolean WordIsNumeric();

  // Returns true if the word contains blamer information.
  public native @Cast("bool") boolean HasBlamerInfo();

  // Returns the pointer to ParamsTrainingBundle stored in the BlamerBundle
  // of the current word.
  public native @Const Pointer GetParamsTrainingBundle();

  // Returns a pointer to the string with blamer information for this word.
  // Assumes that the word's blamer_bundle is not NULL.
  public native @Cast("const char*") BytePointer GetBlamerDebug();

  // Returns a pointer to the string with misadaption information for this word.
  // Assumes that the word's blamer_bundle is not NULL.
  public native @Cast("const char*") BytePointer GetBlamerMisadaptionDebug();

  // Returns true if a truth string was recorded for the current word.
  public native @Cast("bool") boolean HasTruthString();

  // Returns true if the given string is equivalent to the truth string for
  // the current word.
  public native @Cast("bool") boolean EquivalentToTruth(@Cast("const char*") BytePointer str);
  public native @Cast("bool") boolean EquivalentToTruth(String str);

  // Returns a null terminated UTF-8 encoded truth string for the current word.
  // Use delete [] to free after use.
  public native @Cast("char*") BytePointer WordTruthUTF8Text();

  // Returns a null terminated UTF-8 encoded normalized OCR string for the
  // current word. Use delete [] to free after use.
  public native @Cast("char*") BytePointer WordNormedUTF8Text();

  // Returns a pointer to serialized choice lattice.
  // Fills lattice_size with the number of bytes in lattice data.
  public native @Cast("const char*") BytePointer WordLattice(IntPointer lattice_size);
  public native String WordLattice(IntBuffer lattice_size);
  public native @Cast("const char*") BytePointer WordLattice(int[] lattice_size);

  // ============= Functions that refer to symbols only ============.

  // Returns true if the current symbol is a superscript.
  // If iterating at a higher level object than symbols, eg words, then
  // this will return the attributes of the first symbol in that word.
  public native @Cast("bool") boolean SymbolIsSuperscript();
  // Returns true if the current symbol is a subscript.
  // If iterating at a higher level object than symbols, eg words, then
  // this will return the attributes of the first symbol in that word.
  public native @Cast("bool") boolean SymbolIsSubscript();
  // Returns true if the current symbol is a dropcap.
  // If iterating at a higher level object than symbols, eg words, then
  // this will return the attributes of the first symbol in that word.
  public native @Cast("bool") boolean SymbolIsDropcap();
}

// Class to iterate over the classifier choices for a single RIL_SYMBOL.
@Namespace("tesseract") @NoOffset public static class ChoiceIterator extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ChoiceIterator(Pointer p) { super(p); }

  // Construction is from a LTRResultIterator that points to the symbol of
  // interest. The ChoiceIterator allows a one-shot iteration over the
  // choices for this symbol and after that is is useless.
  public ChoiceIterator(@Const @ByRef LTRResultIterator result_it) { super((Pointer)null); allocate(result_it); }
  private native void allocate(@Const @ByRef LTRResultIterator result_it);

  // Moves to the next choice for the symbol and returns false if there
  // are none left.
  public native @Cast("bool") boolean Next();

  // ============= Accessing data ==============.

  // Returns the null terminated UTF-8 encoded text string for the current
  // choice.
  // NOTE: Unlike LTRResultIterator::GetUTF8Text, the return points to an
  // internal structure and should NOT be delete[]ed to free after use.
  public native @Cast("const char*") BytePointer GetUTF8Text();

  // Returns the confidence of the current choice.
  // The number should be interpreted as a percent probability. (0.0f-100.0f)
  public native float Confidence();
}

  // namespace tesseract.

// #endif  // TESSERACT_CCMAIN_LTR_RESULT_ITERATOR_H__


// Parsed from tesseract/resultiterator.h

///////////////////////////////////////////////////////////////////////
// File:        resultiterator.h
// Description: Iterator for tesseract results that is capable of
//              iterating in proper reading order over Bi Directional
//              (e.g. mixed Hebrew and English) text.
// Author:      David Eger
// Created:     Fri May 27 13:58:06 PST 2011
//
// (C) Copyright 2011, Google Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
///////////////////////////////////////////////////////////////////////

// #ifndef TESSERACT_CCMAIN_RESULT_ITERATOR_H__
// #define TESSERACT_CCMAIN_RESULT_ITERATOR_H__

// #include "platform.h"
// #include "ltrresultiterator.h"

@Namespace("tesseract") @NoOffset public static class ResultIterator extends LTRResultIterator {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ResultIterator(Pointer p) { super(p); }

  public static native ResultIterator StartOfParagraph(@Const @ByRef LTRResultIterator resit);

  /**
   * ResultIterator is copy constructible!
   * The default copy constructor works just fine for us.
   */

  // ============= Moving around within the page ============.
  /**
   * Moves the iterator to point to the start of the page to begin
   * an iteration.
   */
  public native void Begin();

  /**
   * Moves to the start of the next object at the given level in the
   * page hierarchy in the appropriate reading order and returns false if
   * the end of the page was reached.
   * NOTE that RIL_SYMBOL will skip non-text blocks, but all other
   * PageIteratorLevel level values will visit each non-text block once.
   * Think of non text blocks as containing a single para, with a single line,
   * with a single imaginary word.
   * Calls to Next with different levels may be freely intermixed.
   * This function iterates words in right-to-left scripts correctly, if
   * the appropriate language has been loaded into Tesseract.
   */
  public native @Cast("bool") boolean Next(@Cast("tesseract::PageIteratorLevel") int level);

  /**
   * IsAtBeginningOf() returns whether we're at the logical beginning of the
   * given level.  (as opposed to ResultIterator's left-to-right top-to-bottom
   * order).  Otherwise, this acts the same as PageIterator::IsAtBeginningOf().
   * For a full description, see pageiterator.h
   */
  public native @Cast("bool") boolean IsAtBeginningOf(@Cast("tesseract::PageIteratorLevel") int level);

  /**
   * Implement PageIterator's IsAtFinalElement correctly in a BiDi context.
   * For instance, IsAtFinalElement(RIL_PARA, RIL_WORD) returns whether we
   * point at the last word in a paragraph.  See PageIterator for full comment.
  */
  public native @Cast("bool") boolean IsAtFinalElement(@Cast("tesseract::PageIteratorLevel") int level,
                                  @Cast("tesseract::PageIteratorLevel") int element);

  // ============= Accessing data ==============.

  /**
   * Returns the null terminated UTF-8 encoded text string for the current
   * object at the given level. Use delete [] to free after use.
  */
  public native @Cast("char*") BytePointer GetUTF8Text(@Cast("tesseract::PageIteratorLevel") int level);

  /**
   * Return whether the current paragraph's dominant reading direction
   * is left-to-right (as opposed to right-to-left).
  */
  public native @Cast("bool") boolean ParagraphIsLtr();

  // ============= Exposed only for testing =============.

  /**
   * Yields the reading order as a sequence of indices and (optional)
   * meta-marks for a set of words (given left-to-right).
   * The meta marks are passed as negative values:
   *   kMinorRunStart  Start of minor direction text.
   *   kMinorRunEnd    End of minor direction text.
   *   kComplexWord    The next indexed word contains both left-to-right and
   *                    right-to-left characters and was treated as neutral.
   *
   * For example, suppose we have five words in a text line,
   * indexed [0,1,2,3,4] from the leftmost side of the text line.
   * The following are all believable reading_orders:
   *
   * Left-to-Right (in ltr paragraph):
   *     { 0, 1, 2, 3, 4 }
   * Left-to-Right (in rtl paragraph):
   *     { kMinorRunStart, 0, 1, 2, 3, 4, kMinorRunEnd }
   * Right-to-Left (in rtl paragraph):
   *     { 4, 3, 2, 1, 0 }
   * Left-to-Right except for an RTL phrase in words 2, 3 in an ltr paragraph:
   *     { 0, 1, kMinorRunStart, 3, 2, kMinorRunEnd, 4 }
   */
  public static native void CalculateTextlineOrder(
        @Cast("bool") boolean paragraph_is_ltr,
        @Cast("const GenericVector<StrongScriptDirection>*") @ByRef IntGenericVector word_dirs,
        IntGenericVectorEqEq reading_order);

  
  
  
}

  // namespace tesseract.

// #endif  // TESSERACT_CCMAIN_RESULT_ITERATOR_H__


// Parsed from tesseract/strngs.h

/**********************************************************************
 * File:        strngs.h  (Formerly strings.h)
 * Description: STRING class definition.
 * Author:					Ray Smith
 * Created:					Fri Feb 15 09:15:01 GMT 1991
 *
 * (C) Copyright 1991, Hewlett-Packard Ltd.
 ** Licensed under the Apache License, Version 2.0 (the "License");
 ** you may not use this file except in compliance with the License.
 ** You may obtain a copy of the License at
 ** http://www.apache.org/licenses/LICENSE-2.0
 ** Unless required by applicable law or agreed to in writing, software
 ** distributed under the License is distributed on an "AS IS" BASIS,
 ** WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 ** See the License for the specific language governing permissions and
 ** limitations under the License.
 *
 **********************************************************************/

// #ifndef           STRNGS_H
// #define           STRNGS_H

// #include          <stdio.h>
// #include          <string.h>
// #include          "platform.h"
// #include          "memry.h"
@Namespace("tesseract") @Opaque public static class TFile extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public TFile() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TFile(Pointer p) { super(p); }
}
  // namespace tesseract.

// STRING_IS_PROTECTED means that  string[index] = X is invalid
// because you have to go through strings interface to modify it.
// This allows the string to ensure internal integrity and maintain
// its own string length. Unfortunately this is not possible because
// STRINGS are used as direct-manipulation data buffers for things
// like length arrays and many places cast away the const on string()
// to mutate the string. Turning this off means that internally we
// cannot assume we know the strlen.
public static native @MemberGetter int STRING_IS_PROTECTED();
public static final int STRING_IS_PROTECTED = STRING_IS_PROTECTED();

@NoOffset public static class STRING extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public STRING(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public STRING(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public STRING position(int position) {
        return (STRING)super.position(position);
    }

    public STRING() { super((Pointer)null); allocate(); }
    private native void allocate();
    public STRING(@Const @ByRef STRING string) { super((Pointer)null); allocate(string); }
    private native void allocate(@Const @ByRef STRING string);
    public STRING(@Cast("const char*") BytePointer string) { super((Pointer)null); allocate(string); }
    private native void allocate(@Cast("const char*") BytePointer string);
    public STRING(String string) { super((Pointer)null); allocate(string); }
    private native void allocate(String string);
    public STRING(@Cast("const char*") BytePointer data, int length) { super((Pointer)null); allocate(data, length); }
    private native void allocate(@Cast("const char*") BytePointer data, int length);
    public STRING(String data, int length) { super((Pointer)null); allocate(data, length); }
    private native void allocate(String data, int length);

    // Writes to the given file. Returns false in case of error.
    public native @Cast("bool") boolean Serialize(@Cast("FILE*") Pointer fp);
    // Reads from the given file. Returns false in case of error.
    // If swap is true, assumes a big/little-endian swap is needed.
    public native @Cast("bool") boolean DeSerialize(@Cast("bool") boolean swap, @Cast("FILE*") Pointer fp);
    // Writes to the given file. Returns false in case of error.
    public native @Cast("bool") boolean Serialize(TFile fp);
    // Reads from the given file. Returns false in case of error.
    // If swap is true, assumes a big/little-endian swap is needed.
    public native @Cast("bool") boolean DeSerialize(@Cast("bool") boolean swap, TFile fp);

    public native @Cast("BOOL8") byte contains(byte c);
    public native @Cast("inT32") int length();
    public native @Cast("inT32") int size();
    public native @Cast("const char*") BytePointer string();
    public native @Cast("const char*") BytePointer c_str();

    public native @Cast("char*") BytePointer strdup();

// #if STRING_IS_PROTECTED
// #else
    public native @Cast("char*") @ByRef @Name("operator []") BytePointer get(@Cast("inT32") int index);
// #endif
    public native void split(byte c, StringGenericVector splited);
    public native void truncate_at(@Cast("inT32") int index);

    public native @Cast("BOOL8") @Name("operator ==") byte equals(@Const @ByRef STRING string);
    public native @Cast("BOOL8") @Name("operator !=") byte notEquals(@Const @ByRef STRING string);
    public native @Cast("BOOL8") @Name("operator !=") byte notEquals(@Cast("const char*") BytePointer string);
    public native @Cast("BOOL8") @Name("operator !=") byte notEquals(String string);

    public native @ByRef @Name("operator =") STRING put(@Cast("const char*") BytePointer string);
    public native @ByRef @Name("operator =") STRING put(String string);
    public native @ByRef @Name("operator =") STRING put(@Const @ByRef STRING string);

    public native @ByVal @Name("operator +") STRING add(@Const @ByRef STRING string);
    public native @ByVal @Name("operator +") STRING add(byte ch);

    public native @ByRef @Name("operator +=") STRING addPut(@Cast("const char*") BytePointer string);
    public native @ByRef @Name("operator +=") STRING addPut(String string);
    public native @ByRef @Name("operator +=") STRING addPut(@Const @ByRef STRING string);
    public native @ByRef @Name("operator +=") STRING addPut(byte ch);

    // Assignment for strings which are not null-terminated.
    public native void assign(@Cast("const char*") BytePointer cstr, int len);
    public native void assign(String cstr, int len);

    // Appends the given string and int (as a %d) to this.
    // += cannot be used for ints as there as a char += operator that would
    // be ambiguous, and ints usually need a string before or between them
    // anyway.
    public native void add_str_int(@Cast("const char*") BytePointer str, int number);
    public native void add_str_int(String str, int number);
    // Appends the given string and double (as a %.8g) to this.
    public native void add_str_double(@Cast("const char*") BytePointer str, double number);
    public native void add_str_double(String str, double number);

    // ensure capacity but keep pointer encapsulated
    public native void ensure(@Cast("inT32") int min_capacity);
}
// #endif


// Parsed from tesseract/genericvector.h

///////////////////////////////////////////////////////////////////////
// File:        genericvector.h
// Description: Generic vector class
// Author:      Daria Antonova
// Created:     Mon Jun 23 11:26:43 PDT 2008
//
// (C) Copyright 2007, Google Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
///////////////////////////////////////////////////////////////////////
//
// #ifndef TESSERACT_CCUTIL_GENERICVECTOR_H_
// #define TESSERACT_CCUTIL_GENERICVECTOR_H_

// #include <assert.h>
// #include <stdio.h>
// #include <stdlib.h>

// #include "tesscallback.h"
// #include "errcode.h"
// #include "helpers.h"
// #include "ndminx.h"
// #include "serialis.h"
// #include "strngs.h"

// Use PointerVector<T> below in preference to GenericVector<T*>, as that
// provides automatic deletion of pointers, [De]Serialize that works, and
// sort that works.
@Name("GenericVector<char>") @NoOffset public static class CharGenericVector extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CharGenericVector(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CharGenericVector(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CharGenericVector position(int position) {
        return (CharGenericVector)super.position(position);
    }

  public CharGenericVector() { super((Pointer)null); allocate(); }
  private native void allocate();
  public CharGenericVector(int size, @Cast("char") byte init_val) { super((Pointer)null); allocate(size, init_val); }
  private native void allocate(int size, @Cast("char") byte init_val);

  // Copy
  public CharGenericVector(@Const @ByRef CharGenericVector other) { super((Pointer)null); allocate(other); }
  private native void allocate(@Const @ByRef CharGenericVector other);
  public native @ByRef @Name("operator +=") CharGenericVector addPut(@Const @ByRef CharGenericVector other);
  public native @ByRef @Name("operator =") CharGenericVector put(@Const @ByRef CharGenericVector other);

  // Reserve some memory.
  public native void reserve(int size);
  // Double the size of the internal array.
  public native void double_the_size();

  // Resizes to size and sets all values to t.
  public native void init_to_size(int size, @Cast("char") byte t);
  // Resizes to size without any initialization.
  public native void resize_no_init(int size);

  // Return the size used.
  public native int size();
  public native int size_reserved();

  public native int length();

  // Return true if empty.
  public native @Cast("bool") boolean empty();

  // Return the object from an index.
  public native @Cast("char*") @ByRef BytePointer get(int index);
  public native @Cast("char*") @ByRef BytePointer back();
  // Returns the last object and removes it.
  public native @Cast("char") byte pop_back();

  // Return the index of the T object.
  // This method NEEDS a compare_callback to be passed to
  // set_compare_callback.
  public native int get_index(@Cast("char") byte object);

  // Return true if T is in the array
  public native @Cast("bool") boolean contains(@Cast("char") byte object);

  // Return true if the index is valid
  public native @Cast("char") byte contains_index(int index);

  // Push an element in the end of the array
  public native int push_back(@Cast("char") byte object);
  public native @Name("operator +=") void addPut(@Cast("char") byte t);

  // Push an element in the end of the array if the same
  // element is not already contained in the array.
  public native int push_back_new(@Cast("char") byte object);

  // Push an element in the front of the array
  // Note: This function is O(n)
  public native int push_front(@Cast("char") byte object);

  // Set the value at the given index
  public native void set(@Cast("char") byte t, int index);

  // Insert t at the given index, push other elements to the right.
  public native void insert(@Cast("char") byte t, int index);

  // Removes an element at the given index and
  // shifts the remaining elements to the left.
  public native void remove(int index);

  // Truncates the array to the given size by removing the end.
  // If the current size is less, the array is not expanded.
  public native void truncate(int size);

  // Add a callback to be called to delete the elements when the array took
  // their ownership.
  public native void set_clear_callback(CharClearCallback cb);

  // Add a callback to be called to compare the elements when needed (contains,
  // get_id, ...)
  public native void set_compare_callback(CharCompareCallback cb);

  // Clear the array, calling the clear callback function if any.
  // All the owned callbacks are also deleted.
  // If you don't want the callbacks to be deleted, before calling clear, set
  // the callback to NULL.
  public native void clear();

  // Delete objects pointed to by data_[i]
  

  // This method clears the current object, then, does a shallow copy of
  // its argument, and finally invalidates its argument.
  // Callbacks are moved to the current object;
  public native void move(CharGenericVector from);

  // Read/Write the array to a file. This does _NOT_ read/write the callbacks.
  // The callback given must be permanent since they will be called more than
  // once. The given callback will be deleted at the end.
  // If the callbacks are NULL, then the data is simply read/written using
  // fread (and swapping)/fwrite.
  // Returns false on error or if the callback returns false.
  // DEPRECATED. Use [De]Serialize[Classes] instead.
  public native @Cast("bool") boolean write(@Cast("FILE*") Pointer f, CharWriteCallback cb);
  public native @Cast("bool") boolean read(@Cast("FILE*") Pointer f, CharReadCallback cb, @Cast("bool") boolean swap);
  // Writes a vector of simple types to the given file. Assumes that bitwise
  // read/write of T will work. Returns false in case of error.
  // TODO(rays) Change all callers to use TFile and remove deprecated methods.
  public native @Cast("bool") boolean Serialize(@Cast("FILE*") Pointer fp);
  public native @Cast("bool") boolean Serialize(TFile fp);
  // Reads a vector of simple types from the given file. Assumes that bitwise
  // read/write will work with ReverseN according to sizeof(T).
  // Returns false in case of error.
  // If swap is true, assumes a big/little-endian swap is needed.
  public native @Cast("bool") boolean DeSerialize(@Cast("bool") boolean swap, @Cast("FILE*") Pointer fp);
  public native @Cast("bool") boolean DeSerialize(@Cast("bool") boolean swap, TFile fp);
  // Writes a vector of classes to the given file. Assumes the existence of
  // bool T::Serialize(FILE* fp) const that returns false in case of error.
  // Returns false in case of error.
  
  
  // Reads a vector of classes from the given file. Assumes the existence of
  // bool T::Deserialize(bool swap, FILE* fp) that returns false in case of
  // error. Also needs T::T() and T::T(constT&), as init_to_size is used in
  // this function. Returns false in case of error.
  // If swap is true, assumes a big/little-endian swap is needed.
  
  

  // Allocates a new array of double the current_size, copies over the
  // information from data to the new location, deletes data and returns
  // the pointed to the new larger array.
  // This function uses memcpy to copy the data, instead of invoking
  // operator=() for each element like double_the_size() does.
  public static native @Cast("char*") BytePointer double_the_size_memcpy(int current_size, @Cast("char*") BytePointer data);
  public static native @Cast("char*") ByteBuffer double_the_size_memcpy(int current_size, @Cast("char*") ByteBuffer data);
  public static native @Cast("char*") byte[] double_the_size_memcpy(int current_size, @Cast("char*") byte[] data);

  // Reverses the elements of the vector.
  public native void reverse();

  // Sorts the members of this vector using the less than comparator (cmp_lt),
  // which compares the values. Useful for GenericVectors to primitive types.
  // Will not work so great for pointers (unless you just want to sort some
  // pointers). You need to provide a specialization to sort_cmp to use
  // your type.
  public native void sort();

  // Sort the array into the order defined by the qsort function comparator.
  // The comparator function is as defined by qsort, ie. it receives pointers
  // to two Ts and returns negative if the first element is to appear earlier
  // in the result and positive if it is to appear later, with 0 for equal.
  public static class Comparator_Pointer_Pointer extends FunctionPointer {
      static { Loader.load(); }
      /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
      public    Comparator_Pointer_Pointer(Pointer p) { super(p); }
      protected Comparator_Pointer_Pointer() { allocate(); }
      private native void allocate();
      public native int call(@Const Pointer arg0, @Const Pointer arg1);
  }
  public native void sort(Comparator_Pointer_Pointer comparator);

  // Searches the array (assuming sorted in ascending order, using sort()) for
  // an element equal to target and returns true if it is present.
  // Use binary_search to get the index of target, or its nearest candidate.
  public native @Cast("bool") boolean bool_binary_search(@Cast("const char") byte target);
  // Searches the array (assuming sorted in ascending order, using sort()) for
  // an element equal to target and returns the index of the best candidate.
  // The return value is conceptually the largest index i such that
  // data_[i] <= target or 0 if target < the whole vector.
  // NOTE that this function uses operator> so really the return value is
  // the largest index i such that data_[i] > target is false.
  public native int binary_search(@Cast("const char") byte target);

  // Compact the vector by deleting elements using operator!= on basic types.
  // The vector must be sorted.
  public native void compact_sorted();

  // Compact the vector by deleting elements for which delete_cb returns
  // true. delete_cb is a permanent callback and will be deleted.
  public native void compact(DeleteCallback delete_cb);

  public native @Cast("char") byte dot_product(@Const @ByRef CharGenericVector other);

  // Returns the index of what would be the target_index_th item in the array
  // if the members were sorted, without actually sorting. Members are
  // shuffled around, but it takes O(n) time.
  // NOTE: uses operator< and operator== on the members.
  public native int choose_nth_item(int target_index);

  // Swaps the elements with the given indices.
  public native void swap(int index1, int index2);
  // Returns true if all elements of *this are within the given range.
  // Only uses operator<
  public native @Cast("bool") boolean WithinBounds(@Cast("const char") byte rangemin, @Cast("const char") byte rangemax);
}
@Name("GenericVector<STRING>") @NoOffset public static class StringGenericVector extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public StringGenericVector(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public StringGenericVector(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public StringGenericVector position(int position) {
        return (StringGenericVector)super.position(position);
    }

  public StringGenericVector() { super((Pointer)null); allocate(); }
  private native void allocate();
  public StringGenericVector(int size, @ByVal STRING init_val) { super((Pointer)null); allocate(size, init_val); }
  private native void allocate(int size, @ByVal STRING init_val);

  // Copy
  public StringGenericVector(@Const @ByRef StringGenericVector other) { super((Pointer)null); allocate(other); }
  private native void allocate(@Const @ByRef StringGenericVector other);
  public native @ByRef @Name("operator +=") StringGenericVector addPut(@Const @ByRef StringGenericVector other);
  public native @ByRef @Name("operator =") StringGenericVector put(@Const @ByRef StringGenericVector other);

  // Reserve some memory.
  public native void reserve(int size);
  // Double the size of the internal array.
  public native void double_the_size();

  // Resizes to size and sets all values to t.
  public native void init_to_size(int size, @ByVal STRING t);
  // Resizes to size without any initialization.
  public native void resize_no_init(int size);

  // Return the size used.
  public native int size();
  public native int size_reserved();

  public native int length();

  // Return true if empty.
  public native @Cast("bool") boolean empty();

  // Return the object from an index.
  public native @ByRef STRING get(int index);
  public native @ByRef STRING back();
  // Returns the last object and removes it.
  public native @ByVal STRING pop_back();

  // Return the index of the T object.
  // This method NEEDS a compare_callback to be passed to
  // set_compare_callback.
  public native int get_index(@ByVal STRING object);

  // Return true if T is in the array
  public native @Cast("bool") boolean contains(@ByVal STRING object);

  // Return true if the index is valid
  

  // Push an element in the end of the array
  public native int push_back(@ByVal STRING object);
  public native @Name("operator +=") void addPut(@ByVal STRING t);

  // Push an element in the end of the array if the same
  // element is not already contained in the array.
  public native int push_back_new(@ByVal STRING object);

  // Push an element in the front of the array
  // Note: This function is O(n)
  public native int push_front(@ByVal STRING object);

  // Set the value at the given index
  public native void set(@ByVal STRING t, int index);

  // Insert t at the given index, push other elements to the right.
  public native void insert(@ByVal STRING t, int index);

  // Removes an element at the given index and
  // shifts the remaining elements to the left.
  public native void remove(int index);

  // Truncates the array to the given size by removing the end.
  // If the current size is less, the array is not expanded.
  public native void truncate(int size);

  // Add a callback to be called to delete the elements when the array took
  // their ownership.
  public native void set_clear_callback(StringClearCallback cb);

  // Add a callback to be called to compare the elements when needed (contains,
  // get_id, ...)
  public native void set_compare_callback(StringCompareCallback cb);

  // Clear the array, calling the clear callback function if any.
  // All the owned callbacks are also deleted.
  // If you don't want the callbacks to be deleted, before calling clear, set
  // the callback to NULL.
  public native void clear();

  // Delete objects pointed to by data_[i]
  

  // This method clears the current object, then, does a shallow copy of
  // its argument, and finally invalidates its argument.
  // Callbacks are moved to the current object;
  public native void move(StringGenericVector from);

  // Read/Write the array to a file. This does _NOT_ read/write the callbacks.
  // The callback given must be permanent since they will be called more than
  // once. The given callback will be deleted at the end.
  // If the callbacks are NULL, then the data is simply read/written using
  // fread (and swapping)/fwrite.
  // Returns false on error or if the callback returns false.
  // DEPRECATED. Use [De]Serialize[Classes] instead.
  public native @Cast("bool") boolean write(@Cast("FILE*") Pointer f, StringWriteCallback cb);
  public native @Cast("bool") boolean read(@Cast("FILE*") Pointer f, StringReadCallback cb, @Cast("bool") boolean swap);
  // Writes a vector of simple types to the given file. Assumes that bitwise
  // read/write of T will work. Returns false in case of error.
  // TODO(rays) Change all callers to use TFile and remove deprecated methods.
  public native @Cast("bool") boolean Serialize(@Cast("FILE*") Pointer fp);
  public native @Cast("bool") boolean Serialize(TFile fp);
  // Reads a vector of simple types from the given file. Assumes that bitwise
  // read/write will work with ReverseN according to sizeof(T).
  // Returns false in case of error.
  // If swap is true, assumes a big/little-endian swap is needed.
  public native @Cast("bool") boolean DeSerialize(@Cast("bool") boolean swap, @Cast("FILE*") Pointer fp);
  public native @Cast("bool") boolean DeSerialize(@Cast("bool") boolean swap, TFile fp);
  // Writes a vector of classes to the given file. Assumes the existence of
  // bool T::Serialize(FILE* fp) const that returns false in case of error.
  // Returns false in case of error.
  public native @Cast("bool") boolean SerializeClasses(@Cast("FILE*") Pointer fp);
  public native @Cast("bool") boolean SerializeClasses(TFile fp);
  // Reads a vector of classes from the given file. Assumes the existence of
  // bool T::Deserialize(bool swap, FILE* fp) that returns false in case of
  // error. Also needs T::T() and T::T(constT&), as init_to_size is used in
  // this function. Returns false in case of error.
  // If swap is true, assumes a big/little-endian swap is needed.
  public native @Cast("bool") boolean DeSerializeClasses(@Cast("bool") boolean swap, @Cast("FILE*") Pointer fp);
  public native @Cast("bool") boolean DeSerializeClasses(@Cast("bool") boolean swap, TFile fp);

  // Allocates a new array of double the current_size, copies over the
  // information from data to the new location, deletes data and returns
  // the pointed to the new larger array.
  // This function uses memcpy to copy the data, instead of invoking
  // operator=() for each element like double_the_size() does.
  public static native STRING double_the_size_memcpy(int current_size, STRING data);

  // Reverses the elements of the vector.
  public native void reverse();

  // Sorts the members of this vector using the less than comparator (cmp_lt),
  // which compares the values. Useful for GenericVectors to primitive types.
  // Will not work so great for pointers (unless you just want to sort some
  // pointers). You need to provide a specialization to sort_cmp to use
  // your type.
  

  // Sort the array into the order defined by the qsort function comparator.
  // The comparator function is as defined by qsort, ie. it receives pointers
  // to two Ts and returns negative if the first element is to appear earlier
  // in the result and positive if it is to appear later, with 0 for equal.
  

  // Searches the array (assuming sorted in ascending order, using sort()) for
  // an element equal to target and returns true if it is present.
  // Use binary_search to get the index of target, or its nearest candidate.
  
  // Searches the array (assuming sorted in ascending order, using sort()) for
  // an element equal to target and returns the index of the best candidate.
  // The return value is conceptually the largest index i such that
  // data_[i] <= target or 0 if target < the whole vector.
  // NOTE that this function uses operator> so really the return value is
  // the largest index i such that data_[i] > target is false.
  

  // Compact the vector by deleting elements using operator!= on basic types.
  // The vector must be sorted.
  public native void compact_sorted();

  // Compact the vector by deleting elements for which delete_cb returns
  // true. delete_cb is a permanent callback and will be deleted.
  public native void compact(DeleteCallback delete_cb);

  

  // Returns the index of what would be the target_index_th item in the array
  // if the members were sorted, without actually sorting. Members are
  // shuffled around, but it takes O(n) time.
  // NOTE: uses operator< and operator== on the members.
  

  // Swaps the elements with the given indices.
  public native void swap(int index1, int index2);
  // Returns true if all elements of *this are within the given range.
  // Only uses operator<
  
}
@Name("GenericVector<int>") @NoOffset public static class IntGenericVector extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public IntGenericVector(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public IntGenericVector(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public IntGenericVector position(int position) {
        return (IntGenericVector)super.position(position);
    }

  public IntGenericVector() { super((Pointer)null); allocate(); }
  private native void allocate();
  public IntGenericVector(int size, int init_val) { super((Pointer)null); allocate(size, init_val); }
  private native void allocate(int size, int init_val);

  // Copy
  public IntGenericVector(@Const @ByRef IntGenericVector other) { super((Pointer)null); allocate(other); }
  private native void allocate(@Const @ByRef IntGenericVector other);
  public native @ByRef @Name("operator +=") IntGenericVector addPut(@Const @ByRef IntGenericVector other);
  public native @ByRef @Name("operator =") IntGenericVector put(@Const @ByRef IntGenericVector other);

  // Reserve some memory.
  public native void reserve(int size);
  // Double the size of the internal array.
  public native void double_the_size();

  // Resizes to size and sets all values to t.
  public native void init_to_size(int size, int t);
  // Resizes to size without any initialization.
  public native void resize_no_init(int size);

  // Return the size used.
  public native int size();
  public native int size_reserved();

  public native int length();

  // Return true if empty.
  public native @Cast("bool") boolean empty();

  // Return the object from an index.
  public native @ByRef IntPointer get(int index);
  public native @ByRef IntPointer back();
  // Returns the last object and removes it.
  public native int pop_back();

  // Return the index of the T object.
  // This method NEEDS a compare_callback to be passed to
  // set_compare_callback.
  public native int get_index(int object);

  // Return true if T is in the array
  public native @Cast("bool") boolean contains(int object);

  // Return true if the index is valid
  public native int contains_index(int index);

  // Push an element in the end of the array
  public native int push_back(int object);
  public native @Name("operator +=") void addPut(int t);

  // Push an element in the end of the array if the same
  // element is not already contained in the array.
  public native int push_back_new(int object);

  // Push an element in the front of the array
  // Note: This function is O(n)
  public native int push_front(int object);

  // Set the value at the given index
  public native void set(int t, int index);

  // Insert t at the given index, push other elements to the right.
  public native void insert(int t, int index);

  // Removes an element at the given index and
  // shifts the remaining elements to the left.
  public native void remove(int index);

  // Truncates the array to the given size by removing the end.
  // If the current size is less, the array is not expanded.
  public native void truncate(int size);

  // Add a callback to be called to delete the elements when the array took
  // their ownership.
  public native void set_clear_callback(IntClearCallback cb);

  // Add a callback to be called to compare the elements when needed (contains,
  // get_id, ...)
  public native void set_compare_callback(IntCompareCallback cb);

  // Clear the array, calling the clear callback function if any.
  // All the owned callbacks are also deleted.
  // If you don't want the callbacks to be deleted, before calling clear, set
  // the callback to NULL.
  public native void clear();

  // Delete objects pointed to by data_[i]
  

  // This method clears the current object, then, does a shallow copy of
  // its argument, and finally invalidates its argument.
  // Callbacks are moved to the current object;
  public native void move(IntGenericVector from);

  // Read/Write the array to a file. This does _NOT_ read/write the callbacks.
  // The callback given must be permanent since they will be called more than
  // once. The given callback will be deleted at the end.
  // If the callbacks are NULL, then the data is simply read/written using
  // fread (and swapping)/fwrite.
  // Returns false on error or if the callback returns false.
  // DEPRECATED. Use [De]Serialize[Classes] instead.
  public native @Cast("bool") boolean write(@Cast("FILE*") Pointer f, IntWriteCallback cb);
  public native @Cast("bool") boolean read(@Cast("FILE*") Pointer f, IntReadCallback cb, @Cast("bool") boolean swap);
  // Writes a vector of simple types to the given file. Assumes that bitwise
  // read/write of T will work. Returns false in case of error.
  // TODO(rays) Change all callers to use TFile and remove deprecated methods.
  public native @Cast("bool") boolean Serialize(@Cast("FILE*") Pointer fp);
  public native @Cast("bool") boolean Serialize(TFile fp);
  // Reads a vector of simple types from the given file. Assumes that bitwise
  // read/write will work with ReverseN according to sizeof(T).
  // Returns false in case of error.
  // If swap is true, assumes a big/little-endian swap is needed.
  public native @Cast("bool") boolean DeSerialize(@Cast("bool") boolean swap, @Cast("FILE*") Pointer fp);
  public native @Cast("bool") boolean DeSerialize(@Cast("bool") boolean swap, TFile fp);
  // Writes a vector of classes to the given file. Assumes the existence of
  // bool T::Serialize(FILE* fp) const that returns false in case of error.
  // Returns false in case of error.
  
  
  // Reads a vector of classes from the given file. Assumes the existence of
  // bool T::Deserialize(bool swap, FILE* fp) that returns false in case of
  // error. Also needs T::T() and T::T(constT&), as init_to_size is used in
  // this function. Returns false in case of error.
  // If swap is true, assumes a big/little-endian swap is needed.
  
  

  // Allocates a new array of double the current_size, copies over the
  // information from data to the new location, deletes data and returns
  // the pointed to the new larger array.
  // This function uses memcpy to copy the data, instead of invoking
  // operator=() for each element like double_the_size() does.
  public static native IntPointer double_the_size_memcpy(int current_size, IntPointer data);
  public static native IntBuffer double_the_size_memcpy(int current_size, IntBuffer data);
  public static native int[] double_the_size_memcpy(int current_size, int[] data);

  // Reverses the elements of the vector.
  public native void reverse();

  // Sorts the members of this vector using the less than comparator (cmp_lt),
  // which compares the values. Useful for GenericVectors to primitive types.
  // Will not work so great for pointers (unless you just want to sort some
  // pointers). You need to provide a specialization to sort_cmp to use
  // your type.
  public native void sort();

  // Sort the array into the order defined by the qsort function comparator.
  // The comparator function is as defined by qsort, ie. it receives pointers
  // to two Ts and returns negative if the first element is to appear earlier
  // in the result and positive if it is to appear later, with 0 for equal.
  public static class Comparator_Pointer_Pointer extends FunctionPointer {
      static { Loader.load(); }
      /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
      public    Comparator_Pointer_Pointer(Pointer p) { super(p); }
      protected Comparator_Pointer_Pointer() { allocate(); }
      private native void allocate();
      public native int call(@Const Pointer arg0, @Const Pointer arg1);
  }
  public native void sort(Comparator_Pointer_Pointer comparator);

  // Searches the array (assuming sorted in ascending order, using sort()) for
  // an element equal to target and returns true if it is present.
  // Use binary_search to get the index of target, or its nearest candidate.
  public native @Cast("bool") boolean bool_binary_search(int target);
  // Searches the array (assuming sorted in ascending order, using sort()) for
  // an element equal to target and returns the index of the best candidate.
  // The return value is conceptually the largest index i such that
  // data_[i] <= target or 0 if target < the whole vector.
  // NOTE that this function uses operator> so really the return value is
  // the largest index i such that data_[i] > target is false.
  public native int binary_search(int target);

  // Compact the vector by deleting elements using operator!= on basic types.
  // The vector must be sorted.
  public native void compact_sorted();

  // Compact the vector by deleting elements for which delete_cb returns
  // true. delete_cb is a permanent callback and will be deleted.
  public native void compact(DeleteCallback delete_cb);

  public native int dot_product(@Const @ByRef IntGenericVector other);

  // Returns the index of what would be the target_index_th item in the array
  // if the members were sorted, without actually sorting. Members are
  // shuffled around, but it takes O(n) time.
  // NOTE: uses operator< and operator== on the members.
  public native int choose_nth_item(int target_index);

  // Swaps the elements with the given indices.
  public native void swap(int index1, int index2);
  // Returns true if all elements of *this are within the given range.
  // Only uses operator<
  public native @Cast("bool") boolean WithinBounds(int rangemin, int rangemax);
}

// Function to read a GenericVector<char> from a whole file.
// Returns false on failure.
public static class FileReader extends FunctionPointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public    FileReader(Pointer p) { super(p); }
    protected FileReader() { allocate(); }
    private native void allocate();
    public native @Cast("bool") boolean call(@Const @ByRef STRING filename, CharGenericVector data);
}
// Function to write a GenericVector<char> to a whole file.
// Returns false on failure.
public static class FileWriter extends FunctionPointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public    FileWriter(Pointer p) { super(p); }
    protected FileWriter() { allocate(); }
    private native void allocate();
    public native @Cast("bool") boolean call(@Const @ByRef CharGenericVector data,
                           @Const @ByRef STRING filename);
}
// The default FileReader loads the whole file into the vector of char,
// returning false on error.
@Namespace("tesseract") public static native @Cast("bool") boolean LoadDataFromFile(@Const @ByRef STRING filename,
                             CharGenericVector data);
// The default FileWriter writes the vector of char to the filename file,
// returning false on error.
@Namespace("tesseract") public static native @Cast("bool") boolean SaveDataToFile(@Const @ByRef CharGenericVector data,
                           @Const @ByRef STRING filename);

// Used by sort()
// return < 0 if t1 < t2
// return 0 if t1 == t2
// return > 0 if t1 > t2

// Used by PointerVector::sort()
// return < 0 if t1 < t2
// return 0 if t1 == t2
// return > 0 if t1 > t2

// Subclass for a vector of pointers. Use in preference to GenericVector<T*>
// as it provides automatic deletion and correct serialization, with the
// corollary that all copy operations are deep copies of the pointed-to objects.

  // namespace tesseract

// A useful vector that uses operator== to do comparisons.
@Name("GenericVectorEqEq<int>") public static class IntGenericVectorEqEq extends IntGenericVector {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public IntGenericVectorEqEq(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public IntGenericVectorEqEq(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public IntGenericVectorEqEq position(int position) {
        return (IntGenericVectorEqEq)super.position(position);
    }

  public IntGenericVectorEqEq() { super((Pointer)null); allocate(); }
  private native void allocate();
  
}





// Reserve some memory. If the internal array contains elements, they are
// copied.




// Resizes to size and sets all values to t.



// Return the object from an index.





// Returns the last object and removes it.


// Return the object from an index.


// Shifts the rest of the elements to the right to make
// space for the new elements and inserts the given element
// at the specified index.


// Removes an element at the given index and
// shifts the remaining elements to the left.


// Return true if the index is valindex


// Return the index of the T object.


// Return true if T is in the array


// Add an element in the array




// Add an element in the array (front)








// Add a callback to be called to delete the elements when the array took
// their ownership.


// Add a callback to be called to delete the elements when the array took
// their ownership.


// Clear the array, calling the callback function if any.









// Writes a vector of simple types to the given file. Assumes that bitwise
// read/write of T will work. Returns false in case of error.



// Reads a vector of simple types from the given file. Assumes that bitwise
// read/write will work with ReverseN according to sizeof(T).
// Returns false in case of error.
// If swap is true, assumes a big/little-endian swap is needed.



// Writes a vector of classes to the given file. Assumes the existence of
// bool T::Serialize(FILE* fp) const that returns false in case of error.
// Returns false in case of error.



// Reads a vector of classes from the given file. Assumes the existence of
// bool T::Deserialize(bool swap, FILE* fp) that returns false in case of
// error. Alse needs T::T() and T::T(constT&), as init_to_size is used in
// this function. Returns false in case of error.
// If swap is true, assumes a big/little-endian swap is needed.



// This method clear the current object, then, does a shallow copy of
// its argument, and finally invalidates its argument.




// Internal recursive version of choose_nth_item.
// The algorithm used comes from "Algorithms" by Sedgewick:
// http://books.google.com/books/about/Algorithms.html?id=idUdqdDXqnAC
// The principle is to choose a random pivot, and move everything less than
// the pivot to its left, and everything greater than the pivot to the end
// of the array, then recurse on the part that contains the desired index, or
// just return the answer if it is in the equal section in the middle.
// The random pivot guarantees average linear time for the same reason that
// n times vector::push_back takes linear time on average.
// target_index, start and and end are all indices into the full array.
// Seed is a seed for rand_r for thread safety purposes. Its value is
// unimportant as the random numbers do not affect the result except
// between equal answers.



// #endif  // TESSERACT_CCUTIL_GENERICVECTOR_H_


// Parsed from tesseract/baseapi.h

///////////////////////////////////////////////////////////////////////
// File:        baseapi.h
// Description: Simple API for calling tesseract.
// Author:      Ray Smith
// Created:     Fri Oct 06 15:35:01 PDT 2006
//
// (C) Copyright 2006, Google Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
///////////////////////////////////////////////////////////////////////

// #ifndef TESSERACT_API_BASEAPI_H__
// #define TESSERACT_API_BASEAPI_H__

public static final String TESSERACT_VERSION_STR = "3.04.00";
public static final int TESSERACT_VERSION = 0x030400;
// #define MAKE_VERSION(major, minor, patch) (((major) << 16) | ((minor) << 8) |
//                                             (patch))

// #include <stdio.h>
// To avoid collision with other typenames include the ABSOLUTE MINIMUM
// complexity of includes here. Use forward declarations wherever possible
// and hide includes of complex types in baseapi.cpp.
// #include "platform.h"
// #include "apitypes.h"
// #include "thresholder.h"
// #include "unichar.h"
// #include "tesscallback.h"
// #include "publictypes.h"
// #include "pageiterator.h"
// #include "resultiterator.h"
@Opaque public static class ParagraphModel extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public ParagraphModel() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ParagraphModel(Pointer p) { super(p); }
}
@Opaque public static class BLOCK_LIST extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public BLOCK_LIST() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public BLOCK_LIST(Pointer p) { super(p); }
}
@Opaque public static class DENORM extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public DENORM() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DENORM(Pointer p) { super(p); }
}
@Opaque public static class MATRIX extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public MATRIX() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public MATRIX(Pointer p) { super(p); }
}
@Opaque public static class ROW extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public ROW() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ROW(Pointer p) { super(p); }
}
@Opaque public static class ETEXT_DESC extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public ETEXT_DESC() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ETEXT_DESC(Pointer p) { super(p); }
}
@Opaque public static class OSResults extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public OSResults() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public OSResults(Pointer p) { super(p); }
}
@Opaque public static class TBOX extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public TBOX() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TBOX(Pointer p) { super(p); }
}
@Opaque public static class UNICHARSET extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public UNICHARSET() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public UNICHARSET(Pointer p) { super(p); }
}
@Opaque public static class WERD_CHOICE_LIST extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public WERD_CHOICE_LIST() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public WERD_CHOICE_LIST(Pointer p) { super(p); }
}

@Opaque public static class INT_FEATURE_STRUCT extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public INT_FEATURE_STRUCT() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public INT_FEATURE_STRUCT(Pointer p) { super(p); }
}
@Opaque public static class TBLOB extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public TBLOB() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TBLOB(Pointer p) { super(p); }
}

@Namespace("tesseract") @Opaque public static class CubeRecoContext extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public CubeRecoContext() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CubeRecoContext(Pointer p) { super(p); }
}
@Namespace("tesseract") @Opaque public static class Dawg extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public Dawg() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Dawg(Pointer p) { super(p); }
}
@Namespace("tesseract") @Opaque public static class Dict extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public Dict() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Dict(Pointer p) { super(p); }
}
@Namespace("tesseract") @Opaque public static class EquationDetect extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public EquationDetect() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public EquationDetect(Pointer p) { super(p); }
}
@Namespace("tesseract") @Opaque public static class MutableIterator extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public MutableIterator() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public MutableIterator(Pointer p) { super(p); }
}
@Namespace("tesseract") @Opaque public static class TessResultRenderer extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public TessResultRenderer() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TessResultRenderer(Pointer p) { super(p); }
}
@Namespace("tesseract") @Opaque public static class Trie extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public Trie() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Trie(Pointer p) { super(p); }
}
@Namespace("tesseract") @Opaque public static class Wordrec extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public Wordrec() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Wordrec(Pointer p) { super(p); }
}

@Namespace("tesseract::Dict") @Const public static class DictFunc extends FunctionPointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public    DictFunc(Pointer p) { super(p); }
    public native int call(Dict o, Pointer void_dawg_args,
                              @Cast("UNICHAR_ID") int unichar_id, @Cast("bool") boolean word_end);
}
@Namespace("tesseract::Dict") public static class ProbabilityInContextFunc extends FunctionPointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public    ProbabilityInContextFunc(Pointer p) { super(p); }
    public native double call(Dict o, @Cast("const char*") BytePointer lang,
                                                 @Cast("const char*") BytePointer context,
                                                 int context_bytes,
                                                 @Cast("const char*") BytePointer character,
                                                 int character_bytes);
}
@Namespace("tesseract::Dict") public static class ParamsModelClassifyFunc extends FunctionPointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public    ParamsModelClassifyFunc(Pointer p) { super(p); }
    public native float call(Dict o, 
    @Cast("const char*") BytePointer lang, Pointer path);
}
@Namespace("tesseract::Wordrec") public static class FillLatticeFunc extends FunctionPointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public    FillLatticeFunc(Pointer p) { super(p); }
    public native void call(Wordrec o, @Const @ByRef MATRIX ratings,
                                         @Const @ByRef WERD_CHOICE_LIST best_choices,
                                         @Const @ByRef UNICHARSET unicharset,
                                         BlamerBundle blamer_bundle);
}

/**
 * Base class for all tesseract APIs.
 * Specific classes can add ability to work on different inputs or produce
 * different outputs.
 * This class is mostly an interface layer on top of the Tesseract instance
 * class to hide the data types so that users of this class don't have to
 * include any other Tesseract headers.
 */
@Namespace("tesseract") @NoOffset public static class TessBaseAPI extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TessBaseAPI(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public TessBaseAPI(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public TessBaseAPI position(int position) {
        return (TessBaseAPI)super.position(position);
    }

  public TessBaseAPI() { super((Pointer)null); allocate(); }
  private native void allocate();

  /**
   * Returns the version identifier as a static string. Do not delete.
   */
  public static native @Cast("const char*") BytePointer Version();

  /**
   * If compiled with OpenCL AND an available OpenCL
   * device is deemed faster than serial code, then
   * "device" is populated with the cl_device_id
   * and returns sizeof(cl_device_id)
   * otherwise *device=NULL and returns 0.
   */
  public static native @Cast("size_t") long getOpenCLDevice(@Cast("void**") PointerPointer device);
  public static native @Cast("size_t") long getOpenCLDevice(@Cast("void**") @ByPtrPtr Pointer device);

  /**
   * Writes the thresholded image to stderr as a PBM file on receipt of a
   * SIGSEGV, SIGFPE, or SIGBUS signal. (Linux/Unix only).
   */
  public static native void CatchSignals();

  /**
   * Set the name of the input file. Needed for training and
   * reading a UNLV zone file, and for searchable PDF output.
   */
  public native void SetInputName(@Cast("const char*") BytePointer name);
  public native void SetInputName(String name);
  /**
   * These functions are required for searchable PDF output.
   * We need our hands on the input file so that we can include
   * it in the PDF without transcoding. If that is not possible,
   * we need the original image. Finally, resolution metadata
   * is stored in the PDF so we need that as well.
   */
  public native @Cast("const char*") BytePointer GetInputName();
  public native void SetInputImage(PIX pix);
  public native PIX GetInputImage();
  public native int GetSourceYResolution();
  public native @Cast("const char*") BytePointer GetDatapath();

  /** Set the name of the bonus output files. Needed only for debugging. */
  public native void SetOutputName(@Cast("const char*") BytePointer name);
  public native void SetOutputName(String name);

  /**
   * Set the value of an internal "parameter."
   * Supply the name of the parameter and the value as a string, just as
   * you would in a config file.
   * Returns false if the name lookup failed.
   * Eg SetVariable("tessedit_char_blacklist", "xyz"); to ignore x, y and z.
   * Or SetVariable("classify_bln_numeric_mode", "1"); to set numeric-only mode.
   * SetVariable may be used before Init, but settings will revert to
   * defaults on End().
   *
   * Note: Must be called after Init(). Only works for non-init variables
   * (init variables should be passed to Init()).
   */
  public native @Cast("bool") boolean SetVariable(@Cast("const char*") BytePointer name, @Cast("const char*") BytePointer value);
  public native @Cast("bool") boolean SetVariable(String name, String value);
  public native @Cast("bool") boolean SetDebugVariable(@Cast("const char*") BytePointer name, @Cast("const char*") BytePointer value);
  public native @Cast("bool") boolean SetDebugVariable(String name, String value);

  /**
   * Returns true if the parameter was found among Tesseract parameters.
   * Fills in value with the value of the parameter.
   */
  public native @Cast("bool") boolean GetIntVariable(@Cast("const char*") BytePointer name, IntPointer value);
  public native @Cast("bool") boolean GetIntVariable(String name, IntBuffer value);
  public native @Cast("bool") boolean GetIntVariable(@Cast("const char*") BytePointer name, int[] value);
  public native @Cast("bool") boolean GetIntVariable(String name, IntPointer value);
  public native @Cast("bool") boolean GetIntVariable(@Cast("const char*") BytePointer name, IntBuffer value);
  public native @Cast("bool") boolean GetIntVariable(String name, int[] value);
  public native @Cast("bool") boolean GetBoolVariable(@Cast("const char*") BytePointer name, @Cast("bool*") BoolPointer value);
  public native @Cast("bool") boolean GetBoolVariable(String name, @Cast("bool*") boolean[] value);
  public native @Cast("bool") boolean GetDoubleVariable(@Cast("const char*") BytePointer name, DoublePointer value);
  public native @Cast("bool") boolean GetDoubleVariable(String name, DoubleBuffer value);
  public native @Cast("bool") boolean GetDoubleVariable(@Cast("const char*") BytePointer name, double[] value);
  public native @Cast("bool") boolean GetDoubleVariable(String name, DoublePointer value);
  public native @Cast("bool") boolean GetDoubleVariable(@Cast("const char*") BytePointer name, DoubleBuffer value);
  public native @Cast("bool") boolean GetDoubleVariable(String name, double[] value);

  /**
   * Returns the pointer to the string that represents the value of the
   * parameter if it was found among Tesseract parameters.
   */
  public native @Cast("const char*") BytePointer GetStringVariable(@Cast("const char*") BytePointer name);
  public native String GetStringVariable(String name);

  /**
   * Print Tesseract parameters to the given file.
   */
  public native void PrintVariables(@Cast("FILE*") Pointer fp);

  /**
   * Get value of named variable as a string, if it exists.
   */
  public native @Cast("bool") boolean GetVariableAsString(@Cast("const char*") BytePointer name, STRING val);
  public native @Cast("bool") boolean GetVariableAsString(String name, STRING val);

  /**
   * Instances are now mostly thread-safe and totally independent,
   * but some global parameters remain. Basically it is safe to use multiple
   * TessBaseAPIs in different threads in parallel, UNLESS:
   * you use SetVariable on some of the Params in classify and textord.
   * If you do, then the effect will be to change it for all your instances.
   *
   * Start tesseract. Returns zero on success and -1 on failure.
   * NOTE that the only members that may be called before Init are those
   * listed above here in the class definition.
   *
   * The datapath must be the name of the parent directory of tessdata and
   * must end in / . Any name after the last / will be stripped.
   * The language is (usually) an ISO 639-3 string or NULL will default to eng.
   * It is entirely safe (and eventually will be efficient too) to call
   * Init multiple times on the same instance to change language, or just
   * to reset the classifier.
   * The language may be a string of the form [~]<lang>[+[~]<lang>]* indicating
   * that multiple languages are to be loaded. Eg hin+eng will load Hindi and
   * English. Languages may specify internally that they want to be loaded
   * with one or more other languages, so the ~ sign is available to override
   * that. Eg if hin were set to load eng by default, then hin+~eng would force
   * loading only hin. The number of loaded languages is limited only by
   * memory, with the caveat that loading additional languages will impact
   * both speed and accuracy, as there is more work to do to decide on the
   * applicable language, and there is more chance of hallucinating incorrect
   * words.
   * WARNING: On changing languages, all Tesseract parameters are reset
   * back to their default values. (Which may vary between languages.)
   * If you have a rare need to set a Variable that controls
   * initialization for a second call to Init you should explicitly
   * call End() and then use SetVariable before Init. This is only a very
   * rare use case, since there are very few uses that require any parameters
   * to be set before Init.
   *
   * If set_only_non_debug_params is true, only params that do not contain
   * "debug" in the name will be set.
   */
  public native int Init(@Cast("const char*") BytePointer datapath, @Cast("const char*") BytePointer language, @Cast("tesseract::OcrEngineMode") int mode,
             @Cast("char**") PointerPointer configs, int configs_size,
             @Const StringGenericVector vars_vec,
             @Const StringGenericVector vars_values,
             @Cast("bool") boolean set_only_non_debug_params);
  public native int Init(@Cast("const char*") BytePointer datapath, @Cast("const char*") BytePointer language, @Cast("tesseract::OcrEngineMode") int mode,
             @Cast("char**") @ByPtrPtr BytePointer configs, int configs_size,
             @Const StringGenericVector vars_vec,
             @Const StringGenericVector vars_values,
             @Cast("bool") boolean set_only_non_debug_params);
  public native int Init(String datapath, String language, @Cast("tesseract::OcrEngineMode") int mode,
             @Cast("char**") @ByPtrPtr ByteBuffer configs, int configs_size,
             @Const StringGenericVector vars_vec,
             @Const StringGenericVector vars_values,
             @Cast("bool") boolean set_only_non_debug_params);
  public native int Init(@Cast("const char*") BytePointer datapath, @Cast("const char*") BytePointer language, @Cast("tesseract::OcrEngineMode") int mode,
             @Cast("char**") @ByPtrPtr byte[] configs, int configs_size,
             @Const StringGenericVector vars_vec,
             @Const StringGenericVector vars_values,
             @Cast("bool") boolean set_only_non_debug_params);
  public native int Init(String datapath, String language, @Cast("tesseract::OcrEngineMode") int mode,
             @Cast("char**") @ByPtrPtr BytePointer configs, int configs_size,
             @Const StringGenericVector vars_vec,
             @Const StringGenericVector vars_values,
             @Cast("bool") boolean set_only_non_debug_params);
  public native int Init(@Cast("const char*") BytePointer datapath, @Cast("const char*") BytePointer language, @Cast("tesseract::OcrEngineMode") int mode,
             @Cast("char**") @ByPtrPtr ByteBuffer configs, int configs_size,
             @Const StringGenericVector vars_vec,
             @Const StringGenericVector vars_values,
             @Cast("bool") boolean set_only_non_debug_params);
  public native int Init(String datapath, String language, @Cast("tesseract::OcrEngineMode") int mode,
             @Cast("char**") @ByPtrPtr byte[] configs, int configs_size,
             @Const StringGenericVector vars_vec,
             @Const StringGenericVector vars_values,
             @Cast("bool") boolean set_only_non_debug_params);
  public native int Init(@Cast("const char*") BytePointer datapath, @Cast("const char*") BytePointer language, @Cast("tesseract::OcrEngineMode") int oem);
  public native int Init(String datapath, String language, @Cast("tesseract::OcrEngineMode") int oem);
  public native int Init(@Cast("const char*") BytePointer datapath, @Cast("const char*") BytePointer language);
  public native int Init(String datapath, String language);

  /**
   * Returns the languages string used in the last valid initialization.
   * If the last initialization specified "deu+hin" then that will be
   * returned. If hin loaded eng automatically as well, then that will
   * not be included in this list. To find the languages actually
   * loaded use GetLoadedLanguagesAsVector.
   * The returned string should NOT be deleted.
   */
  public native @Cast("const char*") BytePointer GetInitLanguagesAsString();

  /**
   * Returns the loaded languages in the vector of STRINGs.
   * Includes all languages loaded by the last Init, including those loaded
   * as dependencies of other loaded languages.
   */
  public native void GetLoadedLanguagesAsVector(StringGenericVector langs);

  /**
   * Returns the available languages in the vector of STRINGs.
   */
  public native void GetAvailableLanguagesAsVector(StringGenericVector langs);

  /**
   * Init only the lang model component of Tesseract. The only functions
   * that work after this init are SetVariable and IsValidWord.
   * WARNING: temporary! This function will be removed from here and placed
   * in a separate API at some future time.
   */
  public native int InitLangMod(@Cast("const char*") BytePointer datapath, @Cast("const char*") BytePointer language);
  public native int InitLangMod(String datapath, String language);

  /**
   * Init only for page layout analysis. Use only for calls to SetImage and
   * AnalysePage. Calls that attempt recognition will generate an error.
   */
  public native void InitForAnalysePage();

  /**
   * Read a "config" file containing a set of param, value pairs.
   * Searches the standard places: tessdata/configs, tessdata/tessconfigs
   * and also accepts a relative or absolute path name.
   * Note: only non-init params will be set (init params are set by Init()).
   */
  public native void ReadConfigFile(@Cast("const char*") BytePointer filename);
  public native void ReadConfigFile(String filename);
  /** Same as above, but only set debug params from the given config file. */
  public native void ReadDebugConfigFile(@Cast("const char*") BytePointer filename);
  public native void ReadDebugConfigFile(String filename);

  /**
   * Set the current page segmentation mode. Defaults to PSM_SINGLE_BLOCK.
   * The mode is stored as an IntParam so it can also be modified by
   * ReadConfigFile or SetVariable("tessedit_pageseg_mode", mode as string).
   */
  public native void SetPageSegMode(@Cast("tesseract::PageSegMode") int mode);

  /** Return the current page segmentation mode. */
  public native @Cast("tesseract::PageSegMode") int GetPageSegMode();

  /**
   * Recognize a rectangle from an image and return the result as a string.
   * May be called many times for a single Init.
   * Currently has no error checking.
   * Greyscale of 8 and color of 24 or 32 bits per pixel may be given.
   * Palette color images will not work properly and must be converted to
   * 24 bit.
   * Binary images of 1 bit per pixel may also be given but they must be
   * byte packed with the MSB of the first byte being the first pixel, and a
   * 1 represents WHITE. For binary images set bytes_per_pixel=0.
   * The recognized text is returned as a char* which is coded
   * as UTF8 and must be freed with the delete [] operator.
   *
   * Note that TesseractRect is the simplified convenience interface.
   * For advanced uses, use SetImage, (optionally) SetRectangle, Recognize,
   * and one or more of the Get*Text functions below.
   */
  public native @Cast("char*") BytePointer TesseractRect(@Cast("const unsigned char*") BytePointer imagedata,
                        int bytes_per_pixel, int bytes_per_line,
                        int left, int top, int width, int height);
  public native @Cast("char*") ByteBuffer TesseractRect(@Cast("const unsigned char*") ByteBuffer imagedata,
                        int bytes_per_pixel, int bytes_per_line,
                        int left, int top, int width, int height);
  public native @Cast("char*") byte[] TesseractRect(@Cast("const unsigned char*") byte[] imagedata,
                        int bytes_per_pixel, int bytes_per_line,
                        int left, int top, int width, int height);

  /**
   * Call between pages or documents etc to free up memory and forget
   * adaptive data.
   */
  public native void ClearAdaptiveClassifier();

  /**
   * \defgroup AdvancedAPI Advanced API
   * The following methods break TesseractRect into pieces, so you can
   * get hold of the thresholded image, get the text in different formats,
   * get bounding boxes, confidences etc.
   */
   /* @{ */

  /**
   * Provide an image for Tesseract to recognize. Format is as
   * TesseractRect above. Does not copy the image buffer, or take
   * ownership. The source image may be destroyed after Recognize is called,
   * either explicitly or implicitly via one of the Get*Text functions.
   * SetImage clears all recognition results, and sets the rectangle to the
   * full image, so it may be followed immediately by a GetUTF8Text, and it
   * will automatically perform recognition.
   */
  public native void SetImage(@Cast("const unsigned char*") BytePointer imagedata, int width, int height,
                  int bytes_per_pixel, int bytes_per_line);
  public native void SetImage(@Cast("const unsigned char*") ByteBuffer imagedata, int width, int height,
                  int bytes_per_pixel, int bytes_per_line);
  public native void SetImage(@Cast("const unsigned char*") byte[] imagedata, int width, int height,
                  int bytes_per_pixel, int bytes_per_line);

  /**
   * Provide an image for Tesseract to recognize. As with SetImage above,
   * Tesseract doesn't take a copy or ownership or pixDestroy the image, so
   * it must persist until after Recognize.
   * Pix vs raw, which to use?
   * Use Pix where possible. A future version of Tesseract may choose to use Pix
   * as its internal representation and discard IMAGE altogether.
   * Because of that, an implementation that sources and targets Pix may end up
   * with less copies than an implementation that does not.
   */
  public native void SetImage(PIX pix);

  /**
   * Set the resolution of the source image in pixels per inch so font size
   * information can be calculated in results.  Call this after SetImage().
   */
  public native void SetSourceResolution(int ppi);

  /**
   * Restrict recognition to a sub-rectangle of the image. Call after SetImage.
   * Each SetRectangle clears the recogntion results so multiple rectangles
   * can be recognized with the same image.
   */
  public native void SetRectangle(int left, int top, int width, int height);

  /**
   * In extreme cases only, usually with a subclass of Thresholder, it
   * is possible to provide a different Thresholder. The Thresholder may
   * be preloaded with an image, settings etc, or they may be set after.
   * Note that Tesseract takes ownership of the Thresholder and will
   * delete it when it it is replaced or the API is destructed.
   */
  public native void SetThresholder(ImageThresholder thresholder);

  /**
   * Get a copy of the internal thresholded image from Tesseract.
   * Caller takes ownership of the Pix and must pixDestroy it.
   * May be called any time after SetImage, or after TesseractRect.
   */
  public native PIX GetThresholdedImage();

  /**
   * Get the result of page layout analysis as a leptonica-style
   * Boxa, Pixa pair, in reading order.
   * Can be called before or after Recognize.
   */
  public native BOXA GetRegions(@Cast("Pixa**") PointerPointer pixa);
  public native BOXA GetRegions(@ByPtrPtr PIXA pixa);

  /**
   * Get the textlines as a leptonica-style
   * Boxa, Pixa pair, in reading order.
   * Can be called before or after Recognize.
   * If raw_image is true, then extract from the original image instead of the
   * thresholded image and pad by raw_padding pixels.
   * If blockids is not NULL, the block-id of each line is also returned as an
   * array of one element per line. delete [] after use.
   * If paraids is not NULL, the paragraph-id of each line within its block is
   * also returned as an array of one element per line. delete [] after use.
   */
  public native BOXA GetTextlines(@Cast("const bool") boolean raw_image, int raw_padding,
                       @Cast("Pixa**") PointerPointer pixa, @Cast("int**") PointerPointer blockids, @Cast("int**") PointerPointer paraids);
  public native BOXA GetTextlines(@Cast("const bool") boolean raw_image, int raw_padding,
                       @ByPtrPtr PIXA pixa, @ByPtrPtr IntPointer blockids, @ByPtrPtr IntPointer paraids);
  public native BOXA GetTextlines(@Cast("const bool") boolean raw_image, int raw_padding,
                       @ByPtrPtr PIXA pixa, @ByPtrPtr IntBuffer blockids, @ByPtrPtr IntBuffer paraids);
  public native BOXA GetTextlines(@Cast("const bool") boolean raw_image, int raw_padding,
                       @ByPtrPtr PIXA pixa, @ByPtrPtr int[] blockids, @ByPtrPtr int[] paraids);
  /*
     Helper method to extract from the thresholded image. (most common usage)
  */
  public native BOXA GetTextlines(@Cast("Pixa**") PointerPointer pixa, @Cast("int**") PointerPointer blockids);
  public native BOXA GetTextlines(@ByPtrPtr PIXA pixa, @ByPtrPtr IntPointer blockids);
  public native BOXA GetTextlines(@ByPtrPtr PIXA pixa, @ByPtrPtr IntBuffer blockids);
  public native BOXA GetTextlines(@ByPtrPtr PIXA pixa, @ByPtrPtr int[] blockids);

  /**
   * Get textlines and strips of image regions as a leptonica-style Boxa, Pixa
   * pair, in reading order. Enables downstream handling of non-rectangular
   * regions.
   * Can be called before or after Recognize.
   * If blockids is not NULL, the block-id of each line is also returned as an
   * array of one element per line. delete [] after use.
   */
  public native BOXA GetStrips(@Cast("Pixa**") PointerPointer pixa, @Cast("int**") PointerPointer blockids);
  public native BOXA GetStrips(@ByPtrPtr PIXA pixa, @ByPtrPtr IntPointer blockids);
  public native BOXA GetStrips(@ByPtrPtr PIXA pixa, @ByPtrPtr IntBuffer blockids);
  public native BOXA GetStrips(@ByPtrPtr PIXA pixa, @ByPtrPtr int[] blockids);

  /**
   * Get the words as a leptonica-style
   * Boxa, Pixa pair, in reading order.
   * Can be called before or after Recognize.
   */
  public native BOXA GetWords(@Cast("Pixa**") PointerPointer pixa);
  public native BOXA GetWords(@ByPtrPtr PIXA pixa);

  /**
   * Gets the individual connected (text) components (created
   * after pages segmentation step, but before recognition)
   * as a leptonica-style Boxa, Pixa pair, in reading order.
   * Can be called before or after Recognize.
   * Note: the caller is responsible for calling boxaDestroy()
   * on the returned Boxa array and pixaDestroy() on cc array.
   */
  public native BOXA GetConnectedComponents(@Cast("Pixa**") PointerPointer cc);
  public native BOXA GetConnectedComponents(@ByPtrPtr PIXA cc);

  /**
   * Get the given level kind of components (block, textline, word etc.) as a
   * leptonica-style Boxa, Pixa pair, in reading order.
   * Can be called before or after Recognize.
   * If blockids is not NULL, the block-id of each component is also returned
   * as an array of one element per component. delete [] after use.
   * If blockids is not NULL, the paragraph-id of each component with its block
   * is also returned as an array of one element per component. delete [] after
   * use.
   * If raw_image is true, then portions of the original image are extracted
   * instead of the thresholded image and padded with raw_padding.
   * If text_only is true, then only text components are returned.
   */
  public native BOXA GetComponentImages(@Cast("const tesseract::PageIteratorLevel") int level,
                             @Cast("const bool") boolean text_only, @Cast("const bool") boolean raw_image,
                             int raw_padding,
                             @Cast("Pixa**") PointerPointer pixa, @Cast("int**") PointerPointer blockids, @Cast("int**") PointerPointer paraids);
  public native BOXA GetComponentImages(@Cast("const tesseract::PageIteratorLevel") int level,
                             @Cast("const bool") boolean text_only, @Cast("const bool") boolean raw_image,
                             int raw_padding,
                             @ByPtrPtr PIXA pixa, @ByPtrPtr IntPointer blockids, @ByPtrPtr IntPointer paraids);
  public native BOXA GetComponentImages(@Cast("const tesseract::PageIteratorLevel") int level,
                             @Cast("const bool") boolean text_only, @Cast("const bool") boolean raw_image,
                             int raw_padding,
                             @ByPtrPtr PIXA pixa, @ByPtrPtr IntBuffer blockids, @ByPtrPtr IntBuffer paraids);
  public native BOXA GetComponentImages(@Cast("const tesseract::PageIteratorLevel") int level,
                             @Cast("const bool") boolean text_only, @Cast("const bool") boolean raw_image,
                             int raw_padding,
                             @ByPtrPtr PIXA pixa, @ByPtrPtr int[] blockids, @ByPtrPtr int[] paraids);
  // Helper function to get binary images with no padding (most common usage).
  public native BOXA GetComponentImages(@Cast("const tesseract::PageIteratorLevel") int level,
                             @Cast("const bool") boolean text_only,
                             @Cast("Pixa**") PointerPointer pixa, @Cast("int**") PointerPointer blockids);
  public native BOXA GetComponentImages(@Cast("const tesseract::PageIteratorLevel") int level,
                             @Cast("const bool") boolean text_only,
                             @ByPtrPtr PIXA pixa, @ByPtrPtr IntPointer blockids);
  public native BOXA GetComponentImages(@Cast("const tesseract::PageIteratorLevel") int level,
                             @Cast("const bool") boolean text_only,
                             @ByPtrPtr PIXA pixa, @ByPtrPtr IntBuffer blockids);
  public native BOXA GetComponentImages(@Cast("const tesseract::PageIteratorLevel") int level,
                             @Cast("const bool") boolean text_only,
                             @ByPtrPtr PIXA pixa, @ByPtrPtr int[] blockids);

  /**
   * Returns the scale factor of the thresholded image that would be returned by
   * GetThresholdedImage() and the various GetX() methods that call
   * GetComponentImages().
   * Returns 0 if no thresholder has been set.
   */
  public native int GetThresholdedImageScaleFactor();

  /**
   * Dump the internal binary image to a PGM file.
   * @deprecated Use GetThresholdedImage and write the image using pixWrite
   * instead if possible.
   */
  public native void DumpPGM(@Cast("const char*") BytePointer filename);
  public native void DumpPGM(String filename);

  /**
   * Runs page layout analysis in the mode set by SetPageSegMode.
   * May optionally be called prior to Recognize to get access to just
   * the page layout results. Returns an iterator to the results.
   * If merge_similar_words is true, words are combined where suitable for use
   * with a line recognizer. Use if you want to use AnalyseLayout to find the
   * textlines, and then want to process textline fragments with an external
   * line recognizer.
   * Returns NULL on error or an empty page.
   * The returned iterator must be deleted after use.
   * WARNING! This class points to data held within the TessBaseAPI class, and
   * therefore can only be used while the TessBaseAPI class still exists and
   * has not been subjected to a call of Init, SetImage, Recognize, Clear, End
   * DetectOS, or anything else that changes the internal PAGE_RES.
   */
  public native PageIterator AnalyseLayout();
  public native PageIterator AnalyseLayout(@Cast("bool") boolean merge_similar_words);

  /**
   * Recognize the image from SetAndThresholdImage, generating Tesseract
   * internal structures. Returns 0 on success.
   * Optional. The Get*Text functions below will call Recognize if needed.
   * After Recognize, the output is kept internally until the next SetImage.
   */
  public native int Recognize(ETEXT_DESC monitor);

  /**
   * Methods to retrieve information after SetAndThresholdImage(),
   * Recognize() or TesseractRect(). (Recognize is called implicitly if needed.)
   */

  /** Variant on Recognize used for testing chopper. */
  public native int RecognizeForChopTest(ETEXT_DESC monitor);

  /**
   * Turns images into symbolic text.
   *
   * filename can point to a single image, a multi-page TIFF,
   * or a plain text list of image filenames.
   *
   * retry_config is useful for debugging. If not NULL, you can fall
   * back to an alternate configuration if a page fails for some
   * reason.
   *
   * timeout_millisec terminates processing if any single page
   * takes too long. Set to 0 for unlimited time.
   *
   * renderer is responible for creating the output. For example,
   * use the TessTextRenderer if you want plaintext output, or
   * the TessPDFRender to produce searchable PDF.
   *
   * If tessedit_page_number is non-negative, will only process that
   * single page. Works for multi-page tiff file, or filelist.
   *
   * Returns true if successful, false on error.
   */
  public native @Cast("bool") boolean ProcessPages(@Cast("const char*") BytePointer filename, @Cast("const char*") BytePointer retry_config,
                      int timeout_millisec, TessResultRenderer renderer);
  public native @Cast("bool") boolean ProcessPages(String filename, String retry_config,
                      int timeout_millisec, TessResultRenderer renderer);
  // Does the real work of ProcessPages.
  public native @Cast("bool") boolean ProcessPagesInternal(@Cast("const char*") BytePointer filename, @Cast("const char*") BytePointer retry_config,
                              int timeout_millisec, TessResultRenderer renderer);
  public native @Cast("bool") boolean ProcessPagesInternal(String filename, String retry_config,
                              int timeout_millisec, TessResultRenderer renderer);

  /**
   * Turn a single image into symbolic text.
   *
   * The pix is the image processed. filename and page_index are
   * metadata used by side-effect processes, such as reading a box
   * file or formatting as hOCR.
   *
   * See ProcessPages for desciptions of other parameters.
   */
  public native @Cast("bool") boolean ProcessPage(PIX pix, int page_index, @Cast("const char*") BytePointer filename,
                     @Cast("const char*") BytePointer retry_config, int timeout_millisec,
                     TessResultRenderer renderer);
  public native @Cast("bool") boolean ProcessPage(PIX pix, int page_index, String filename,
                     String retry_config, int timeout_millisec,
                     TessResultRenderer renderer);

  /**
   * Get a reading-order iterator to the results of LayoutAnalysis and/or
   * Recognize. The returned iterator must be deleted after use.
   * WARNING! This class points to data held within the TessBaseAPI class, and
   * therefore can only be used while the TessBaseAPI class still exists and
   * has not been subjected to a call of Init, SetImage, Recognize, Clear, End
   * DetectOS, or anything else that changes the internal PAGE_RES.
   */
  public native ResultIterator GetIterator();

  /**
   * Get a mutable iterator to the results of LayoutAnalysis and/or Recognize.
   * The returned iterator must be deleted after use.
   * WARNING! This class points to data held within the TessBaseAPI class, and
   * therefore can only be used while the TessBaseAPI class still exists and
   * has not been subjected to a call of Init, SetImage, Recognize, Clear, End
   * DetectOS, or anything else that changes the internal PAGE_RES.
   */
  public native MutableIterator GetMutableIterator();

  /**
   * The recognized text is returned as a char* which is coded
   * as UTF8 and must be freed with the delete [] operator.
   */
  public native @Cast("char*") BytePointer GetUTF8Text();

  /**
   * Make a HTML-formatted string with hOCR markup from the internal
   * data structures.
   * page_number is 0-based but will appear in the output as 1-based.
   */
  public native @Cast("char*") BytePointer GetHOCRText(int page_number);

  /**
   * The recognized text is returned as a char* which is coded in the same
   * format as a box file used in training. Returned string must be freed with
   * the delete [] operator.
   * Constructs coordinates in the original image - not just the rectangle.
   * page_number is a 0-based page index that will appear in the box file.
   */
  public native @Cast("char*") BytePointer GetBoxText(int page_number);
  /**
   * The recognized text is returned as a char* which is coded
   * as UNLV format Latin-1 with specific reject and suspect codes
   * and must be freed with the delete [] operator.
   */
  public native @Cast("char*") BytePointer GetUNLVText();
  /** Returns the (average) confidence value between 0 and 100. */
  public native int MeanTextConf();
  /**
   * Returns all word confidences (between 0 and 100) in an array, terminated
   * by -1.  The calling function must delete [] after use.
   * The number of confidences should correspond to the number of space-
   * delimited words in GetUTF8Text.
   */
  public native IntPointer AllWordConfidences();

  /**
   * Applies the given word to the adaptive classifier if possible.
   * The word must be SPACE-DELIMITED UTF-8 - l i k e t h i s , so it can
   * tell the boundaries of the graphemes.
   * Assumes that SetImage/SetRectangle have been used to set the image
   * to the given word. The mode arg should be PSM_SINGLE_WORD or
   * PSM_CIRCLE_WORD, as that will be used to control layout analysis.
   * The currently set PageSegMode is preserved.
   * Returns false if adaption was not possible for some reason.
   */
  public native @Cast("bool") boolean AdaptToWordStr(@Cast("tesseract::PageSegMode") int mode, @Cast("const char*") BytePointer wordstr);
  public native @Cast("bool") boolean AdaptToWordStr(@Cast("tesseract::PageSegMode") int mode, String wordstr);

  /**
   * Free up recognition results and any stored image data, without actually
   * freeing any recognition data that would be time-consuming to reload.
   * Afterwards, you must call SetImage or TesseractRect before doing
   * any Recognize or Get* operation.
   */
  public native void Clear();

  /**
   * Close down tesseract and free up all memory. End() is equivalent to
   * destructing and reconstructing your TessBaseAPI.
   * Once End() has been used, none of the other API functions may be used
   * other than Init and anything declared above it in the class definition.
   */
  public native void End();

  /**
   * Clear any library-level memory caches.
   * There are a variety of expensive-to-load constant data structures (mostly
   * language dictionaries) that are cached globally -- surviving the Init()
   * and End() of individual TessBaseAPI's.  This function allows the clearing
   * of these caches.
   **/
  public static native void ClearPersistentCache();

  /**
   * Check whether a word is valid according to Tesseract's language model
   * @return 0 if the word is invalid, non-zero if valid.
   * \warning temporary! This function will be removed from here and placed
   * in a separate API at some future time.
   */
  public native int IsValidWord(@Cast("const char*") BytePointer word);
  public native int IsValidWord(String word);
  // Returns true if utf8_character is defined in the UniCharset.
  public native @Cast("bool") boolean IsValidCharacter(@Cast("const char*") BytePointer utf8_character);
  public native @Cast("bool") boolean IsValidCharacter(String utf8_character);


  public native @Cast("bool") boolean GetTextDirection(IntPointer out_offset, FloatPointer out_slope);
  public native @Cast("bool") boolean GetTextDirection(IntBuffer out_offset, FloatBuffer out_slope);
  public native @Cast("bool") boolean GetTextDirection(int[] out_offset, float[] out_slope);

  /** Sets Dict::letter_is_okay_ function to point to the given function. */
  public native void SetDictFunc(DictFunc f);

  /** Sets Dict::probability_in_context_ function to point to the given
   * function.
   */
  public native void SetProbabilityInContextFunc(ProbabilityInContextFunc f);

  /** Sets Wordrec::fill_lattice_ function to point to the given function. */
  

  /**
   * Estimates the Orientation And Script of the image.
   * @return true if the image was processed successfully.
   */
  public native @Cast("bool") boolean DetectOS(OSResults arg0);

  /** This method returns the features associated with the input image. */
  public native void GetFeaturesForBlob(TBLOB blob, INT_FEATURE_STRUCT int_features,
                            IntPointer num_features, IntPointer feature_outline_index);
  public native void GetFeaturesForBlob(TBLOB blob, INT_FEATURE_STRUCT int_features,
                            IntBuffer num_features, IntBuffer feature_outline_index);
  public native void GetFeaturesForBlob(TBLOB blob, INT_FEATURE_STRUCT int_features,
                            int[] num_features, int[] feature_outline_index);

  /**
   * This method returns the row to which a box of specified dimensions would
   * belong. If no good match is found, it returns NULL.
   */
  public static native ROW FindRowForBox(BLOCK_LIST blocks, int left, int top,
                              int right, int bottom);

  /**
   * Method to run adaptive classifier on a blob.
   * It returns at max num_max_matches results.
   */
  public native void RunAdaptiveClassifier(TBLOB blob,
                               int num_max_matches,
                               IntPointer unichar_ids,
                               FloatPointer ratings,
                               IntPointer num_matches_returned);
  public native void RunAdaptiveClassifier(TBLOB blob,
                               int num_max_matches,
                               IntBuffer unichar_ids,
                               FloatBuffer ratings,
                               IntBuffer num_matches_returned);
  public native void RunAdaptiveClassifier(TBLOB blob,
                               int num_max_matches,
                               int[] unichar_ids,
                               float[] ratings,
                               int[] num_matches_returned);

  /** This method returns the string form of the specified unichar. */
  public native @Cast("const char*") BytePointer GetUnichar(int unichar_id);

  /** Return the pointer to the i-th dawg loaded into tesseract_ object. */
  public native @Const Dawg GetDawg(int i);

  /** Return the number of dawgs loaded into tesseract_ object. */
  public native int NumDawgs();

  /** Returns a ROW object created from the input row specification. */
  public static native ROW MakeTessOCRRow(float baseline, float xheight,
                               float descender, float ascender);

  /** Returns a TBLOB corresponding to the entire input image. */
  public static native TBLOB MakeTBLOB(PIX pix);

  /**
   * This method baseline normalizes a TBLOB in-place. The input row is used
   * for normalization. The denorm is an optional parameter in which the
   * normalization-antidote is returned.
   */
  public static native void NormalizeTBLOB(TBLOB tblob, ROW row, @Cast("bool") boolean numeric_mode);

  public native Tesseract tesseract();

  public native @Cast("tesseract::OcrEngineMode const") int oem();

  public native void InitTruthCallback(@Cast("tesseract::TruthCallback*") TruthCallback4 cb);

  /** Return a pointer to underlying CubeRecoContext object if present. */
  public native CubeRecoContext GetCubeRecoContext();

  public native void set_min_orientation_margin(double margin);

  /**
   * Return text orientation of each block as determined by an earlier run
   * of layout analysis.
   */
  public native void GetBlockTextOrientations(@Cast("int**") PointerPointer block_orientation,
                                  @Cast("bool**") PointerPointer vertical_writing);
  public native void GetBlockTextOrientations(@ByPtrPtr IntPointer block_orientation,
                                  @Cast("bool**") @ByPtrPtr BoolPointer vertical_writing);
  public native void GetBlockTextOrientations(@ByPtrPtr IntBuffer block_orientation,
                                  @Cast("bool**") @ByPtrPtr boolean[] vertical_writing);
  public native void GetBlockTextOrientations(@ByPtrPtr int[] block_orientation,
                                  @Cast("bool**") @ByPtrPtr BoolPointer vertical_writing);
  public native void GetBlockTextOrientations(@ByPtrPtr IntPointer block_orientation,
                                  @Cast("bool**") @ByPtrPtr boolean[] vertical_writing);
  public native void GetBlockTextOrientations(@ByPtrPtr IntBuffer block_orientation,
                                  @Cast("bool**") @ByPtrPtr BoolPointer vertical_writing);
  public native void GetBlockTextOrientations(@ByPtrPtr int[] block_orientation,
                                  @Cast("bool**") @ByPtrPtr boolean[] vertical_writing);

  /** Find lines from the image making the BLOCK_LIST. */
  public native BLOCK_LIST FindLinesCreateBlockList();

  /**
   * Delete a block list.
   * This is to keep BLOCK_LIST pointer opaque
   * and let go of including the other headers.
   */
  public static native void DeleteBlockList(BLOCK_LIST block_list);
}  // class TessBaseAPI.

/** Escape a char string - remove &<>"' with HTML codes. */
@Namespace("tesseract") public static native @ByVal STRING HOcrEscape(@Cast("const char*") BytePointer text);
@Namespace("tesseract") public static native @ByVal STRING HOcrEscape(String text);
  // namespace tesseract.

// #endif  // TESSERACT_API_BASEAPI_H__


// Parsed from tesseract/capi.h

// #ifndef TESSERACT_API_CAPI_H__
// #define TESSERACT_API_CAPI_H__

// #ifdef TESS_CAPI_INCLUDE_BASEAPI
// #   include "baseapi.h"
// #   include "pageiterator.h"
// #   include "resultiterator.h"
// #   include "renderer.h"
// #else
// #endif

// #ifdef __cplusplus
// #endif

// #ifndef TESS_CALL
// #   if defined(WIN32)
// #       define TESS_CALL __cdecl
// #   else
// #       define TESS_CALL
// #   endif
// #endif

// #ifndef BOOL
// #endif

// #ifdef TESS_CAPI_INCLUDE_BASEAPI
// typedef tesseract::ParamsModelClassifyFunc TessParamsModelClassifyFunc;
// #else
// #endif

/* General free functions */

public static native @Cast("const char*") BytePointer TessVersion();
public static native void TessDeleteText(@Cast("char*") BytePointer text);
public static native void TessDeleteText(@Cast("char*") ByteBuffer text);
public static native void TessDeleteText(@Cast("char*") byte[] text);
public static native void TessDeleteTextArray(@Cast("char**") PointerPointer arr);
public static native void TessDeleteTextArray(@Cast("char**") @ByPtrPtr BytePointer arr);
public static native void TessDeleteTextArray(@Cast("char**") @ByPtrPtr ByteBuffer arr);
public static native void TessDeleteTextArray(@Cast("char**") @ByPtrPtr byte[] arr);
public static native void TessDeleteIntArray(IntPointer arr);
public static native void TessDeleteIntArray(IntBuffer arr);
public static native void TessDeleteIntArray(int[] arr);
// #ifdef TESS_CAPI_INCLUDE_BASEAPI
public static native void TessDeleteBlockList(BLOCK_LIST block_list);
// #endif

/* Renderer API */
public static native TessResultRenderer TessTextRendererCreate(@Cast("const char*") BytePointer outputbase);
public static native TessResultRenderer TessTextRendererCreate(String outputbase);
public static native TessResultRenderer TessHOcrRendererCreate(@Cast("const char*") BytePointer outputbase);
public static native TessResultRenderer TessHOcrRendererCreate(String outputbase);
public static native TessResultRenderer TessHOcrRendererCreate2(@Cast("const char*") BytePointer outputbase, @Cast("BOOL") boolean font_info);
public static native TessResultRenderer TessHOcrRendererCreate2(String outputbase, @Cast("BOOL") boolean font_info);
public static native TessResultRenderer TessPDFRendererCreate(@Cast("const char*") BytePointer outputbase, @Cast("const char*") BytePointer datadir);
public static native TessResultRenderer TessPDFRendererCreate(String outputbase, String datadir);
public static native TessResultRenderer TessUnlvRendererCreate(@Cast("const char*") BytePointer outputbase);
public static native TessResultRenderer TessUnlvRendererCreate(String outputbase);
public static native TessResultRenderer TessBoxTextRendererCreate(@Cast("const char*") BytePointer outputbase);
public static native TessResultRenderer TessBoxTextRendererCreate(String outputbase);

public static native void TessDeleteResultRenderer(TessResultRenderer renderer);
public static native void TessResultRendererInsert(TessResultRenderer renderer, TessResultRenderer next);
public static native TessResultRenderer TessResultRendererNext(TessResultRenderer renderer);
public static native @Cast("BOOL") boolean TessResultRendererBeginDocument(TessResultRenderer renderer, @Cast("const char*") BytePointer title);
public static native @Cast("BOOL") boolean TessResultRendererBeginDocument(TessResultRenderer renderer, String title);
public static native @Cast("BOOL") boolean TessResultRendererAddImage(TessResultRenderer renderer, TessBaseAPI api);
public static native @Cast("BOOL") boolean TessResultRendererEndDocument(TessResultRenderer renderer);

public static native @Cast("const char*") BytePointer TessResultRendererExtention(TessResultRenderer renderer);
public static native @Cast("const char*") BytePointer TessResultRendererTitle(TessResultRenderer renderer);
public static native int TessResultRendererImageNum(TessResultRenderer renderer);

/* Base API */

public static native TessBaseAPI TessBaseAPICreate();
public static native void TessBaseAPIDelete(TessBaseAPI handle);

public static native @Cast("size_t") long TessBaseAPIGetOpenCLDevice(TessBaseAPI handle, @Cast("void**") PointerPointer device);
public static native @Cast("size_t") long TessBaseAPIGetOpenCLDevice(TessBaseAPI handle, @Cast("void**") @ByPtrPtr Pointer device);

public static native void TessBaseAPISetInputName( TessBaseAPI handle, @Cast("const char*") BytePointer name);
public static native void TessBaseAPISetInputName( TessBaseAPI handle, String name);
public static native @Cast("const char*") BytePointer TessBaseAPIGetInputName(TessBaseAPI handle);

public static native void TessBaseAPISetInputImage(TessBaseAPI handle, PIX pix);
public static native PIX TessBaseAPIGetInputImage(TessBaseAPI handle);

public static native int TessBaseAPIGetSourceYResolution(TessBaseAPI handle);
public static native @Cast("const char*") BytePointer TessBaseAPIGetDatapath(TessBaseAPI handle);

public static native void TessBaseAPISetOutputName(TessBaseAPI handle, @Cast("const char*") BytePointer name);
public static native void TessBaseAPISetOutputName(TessBaseAPI handle, String name);

public static native @Cast("BOOL") boolean TessBaseAPISetVariable(TessBaseAPI handle, @Cast("const char*") BytePointer name, @Cast("const char*") BytePointer value);
public static native @Cast("BOOL") boolean TessBaseAPISetVariable(TessBaseAPI handle, String name, String value);
public static native @Cast("BOOL") boolean TessBaseAPISetDebugVariable(TessBaseAPI handle, @Cast("const char*") BytePointer name, @Cast("const char*") BytePointer value);
public static native @Cast("BOOL") boolean TessBaseAPISetDebugVariable(TessBaseAPI handle, String name, String value);

public static native @Cast("BOOL") boolean TessBaseAPIGetIntVariable(   @Const TessBaseAPI handle, @Cast("const char*") BytePointer name, IntPointer value);
public static native @Cast("BOOL") boolean TessBaseAPIGetIntVariable(   @Const TessBaseAPI handle, String name, IntBuffer value);
public static native @Cast("BOOL") boolean TessBaseAPIGetIntVariable(   @Const TessBaseAPI handle, @Cast("const char*") BytePointer name, int[] value);
public static native @Cast("BOOL") boolean TessBaseAPIGetIntVariable(   @Const TessBaseAPI handle, String name, IntPointer value);
public static native @Cast("BOOL") boolean TessBaseAPIGetIntVariable(   @Const TessBaseAPI handle, @Cast("const char*") BytePointer name, IntBuffer value);
public static native @Cast("BOOL") boolean TessBaseAPIGetIntVariable(   @Const TessBaseAPI handle, String name, int[] value);
public static native @Cast("BOOL") boolean TessBaseAPIGetBoolVariable(  @Const TessBaseAPI handle, @Cast("const char*") BytePointer name, @Cast("BOOL*") BoolPointer value);
public static native @Cast("BOOL") boolean TessBaseAPIGetBoolVariable(  @Const TessBaseAPI handle, String name, @Cast("BOOL*") BoolPointer value);
public static native @Cast("BOOL") boolean TessBaseAPIGetDoubleVariable(@Const TessBaseAPI handle, @Cast("const char*") BytePointer name, DoublePointer value);
public static native @Cast("BOOL") boolean TessBaseAPIGetDoubleVariable(@Const TessBaseAPI handle, String name, DoubleBuffer value);
public static native @Cast("BOOL") boolean TessBaseAPIGetDoubleVariable(@Const TessBaseAPI handle, @Cast("const char*") BytePointer name, double[] value);
public static native @Cast("BOOL") boolean TessBaseAPIGetDoubleVariable(@Const TessBaseAPI handle, String name, DoublePointer value);
public static native @Cast("BOOL") boolean TessBaseAPIGetDoubleVariable(@Const TessBaseAPI handle, @Cast("const char*") BytePointer name, DoubleBuffer value);
public static native @Cast("BOOL") boolean TessBaseAPIGetDoubleVariable(@Const TessBaseAPI handle, String name, double[] value);
public static native @Cast("const char*") BytePointer TessBaseAPIGetStringVariable(@Const TessBaseAPI handle, @Cast("const char*") BytePointer name);
public static native String TessBaseAPIGetStringVariable(@Const TessBaseAPI handle, String name);

public static native void TessBaseAPIPrintVariables(      @Const TessBaseAPI handle, @Cast("FILE*") Pointer fp);
public static native @Cast("BOOL") boolean TessBaseAPIPrintVariablesToFile(@Const TessBaseAPI handle, @Cast("const char*") BytePointer filename);
public static native @Cast("BOOL") boolean TessBaseAPIPrintVariablesToFile(@Const TessBaseAPI handle, String filename);
// #ifdef TESS_CAPI_INCLUDE_BASEAPI
public static native @Cast("BOOL") boolean TessBaseAPIGetVariableAsString(TessBaseAPI handle, @Cast("const char*") BytePointer name, STRING val);
public static native @Cast("BOOL") boolean TessBaseAPIGetVariableAsString(TessBaseAPI handle, String name, STRING val);
// #endif

// #ifdef TESS_CAPI_INCLUDE_BASEAPI

// #endif
public static native int TessBaseAPIInit1(TessBaseAPI handle, @Cast("const char*") BytePointer datapath, @Cast("const char*") BytePointer language, @Cast("TessOcrEngineMode") int oem,
                                          @Cast("char**") PointerPointer configs, int configs_size);
public static native int TessBaseAPIInit1(TessBaseAPI handle, @Cast("const char*") BytePointer datapath, @Cast("const char*") BytePointer language, @Cast("TessOcrEngineMode") int oem,
                                          @Cast("char**") @ByPtrPtr BytePointer configs, int configs_size);
public static native int TessBaseAPIInit1(TessBaseAPI handle, String datapath, String language, @Cast("TessOcrEngineMode") int oem,
                                          @Cast("char**") @ByPtrPtr ByteBuffer configs, int configs_size);
public static native int TessBaseAPIInit1(TessBaseAPI handle, @Cast("const char*") BytePointer datapath, @Cast("const char*") BytePointer language, @Cast("TessOcrEngineMode") int oem,
                                          @Cast("char**") @ByPtrPtr byte[] configs, int configs_size);
public static native int TessBaseAPIInit1(TessBaseAPI handle, String datapath, String language, @Cast("TessOcrEngineMode") int oem,
                                          @Cast("char**") @ByPtrPtr BytePointer configs, int configs_size);
public static native int TessBaseAPIInit1(TessBaseAPI handle, @Cast("const char*") BytePointer datapath, @Cast("const char*") BytePointer language, @Cast("TessOcrEngineMode") int oem,
                                          @Cast("char**") @ByPtrPtr ByteBuffer configs, int configs_size);
public static native int TessBaseAPIInit1(TessBaseAPI handle, String datapath, String language, @Cast("TessOcrEngineMode") int oem,
                                          @Cast("char**") @ByPtrPtr byte[] configs, int configs_size);
public static native int TessBaseAPIInit2(TessBaseAPI handle, @Cast("const char*") BytePointer datapath, @Cast("const char*") BytePointer language, @Cast("TessOcrEngineMode") int oem);
public static native int TessBaseAPIInit2(TessBaseAPI handle, String datapath, String language, @Cast("TessOcrEngineMode") int oem);
public static native int TessBaseAPIInit3(TessBaseAPI handle, @Cast("const char*") BytePointer datapath, @Cast("const char*") BytePointer language);
public static native int TessBaseAPIInit3(TessBaseAPI handle, String datapath, String language);

public static native int TessBaseAPIInit4(TessBaseAPI handle, @Cast("const char*") BytePointer datapath, @Cast("const char*") BytePointer language, @Cast("TessOcrEngineMode") int mode,
    @Cast("char**") PointerPointer configs, int configs_size,
    @Cast("char**") PointerPointer vars_vec, @Cast("char**") PointerPointer vars_values, @Cast("size_t") long vars_vec_size,
    @Cast("BOOL") boolean set_only_non_debug_params);
public static native int TessBaseAPIInit4(TessBaseAPI handle, @Cast("const char*") BytePointer datapath, @Cast("const char*") BytePointer language, @Cast("TessOcrEngineMode") int mode,
    @Cast("char**") @ByPtrPtr BytePointer configs, int configs_size,
    @Cast("char**") @ByPtrPtr BytePointer vars_vec, @Cast("char**") @ByPtrPtr BytePointer vars_values, @Cast("size_t") long vars_vec_size,
    @Cast("BOOL") boolean set_only_non_debug_params);
public static native int TessBaseAPIInit4(TessBaseAPI handle, String datapath, String language, @Cast("TessOcrEngineMode") int mode,
    @Cast("char**") @ByPtrPtr ByteBuffer configs, int configs_size,
    @Cast("char**") @ByPtrPtr ByteBuffer vars_vec, @Cast("char**") @ByPtrPtr ByteBuffer vars_values, @Cast("size_t") long vars_vec_size,
    @Cast("BOOL") boolean set_only_non_debug_params);
public static native int TessBaseAPIInit4(TessBaseAPI handle, @Cast("const char*") BytePointer datapath, @Cast("const char*") BytePointer language, @Cast("TessOcrEngineMode") int mode,
    @Cast("char**") @ByPtrPtr byte[] configs, int configs_size,
    @Cast("char**") @ByPtrPtr byte[] vars_vec, @Cast("char**") @ByPtrPtr byte[] vars_values, @Cast("size_t") long vars_vec_size,
    @Cast("BOOL") boolean set_only_non_debug_params);
public static native int TessBaseAPIInit4(TessBaseAPI handle, String datapath, String language, @Cast("TessOcrEngineMode") int mode,
    @Cast("char**") @ByPtrPtr BytePointer configs, int configs_size,
    @Cast("char**") @ByPtrPtr BytePointer vars_vec, @Cast("char**") @ByPtrPtr BytePointer vars_values, @Cast("size_t") long vars_vec_size,
    @Cast("BOOL") boolean set_only_non_debug_params);
public static native int TessBaseAPIInit4(TessBaseAPI handle, @Cast("const char*") BytePointer datapath, @Cast("const char*") BytePointer language, @Cast("TessOcrEngineMode") int mode,
    @Cast("char**") @ByPtrPtr ByteBuffer configs, int configs_size,
    @Cast("char**") @ByPtrPtr ByteBuffer vars_vec, @Cast("char**") @ByPtrPtr ByteBuffer vars_values, @Cast("size_t") long vars_vec_size,
    @Cast("BOOL") boolean set_only_non_debug_params);
public static native int TessBaseAPIInit4(TessBaseAPI handle, String datapath, String language, @Cast("TessOcrEngineMode") int mode,
    @Cast("char**") @ByPtrPtr byte[] configs, int configs_size,
    @Cast("char**") @ByPtrPtr byte[] vars_vec, @Cast("char**") @ByPtrPtr byte[] vars_values, @Cast("size_t") long vars_vec_size,
    @Cast("BOOL") boolean set_only_non_debug_params);

public static native @Cast("const char*") BytePointer TessBaseAPIGetInitLanguagesAsString(@Const TessBaseAPI handle);
public static native @Cast("char**") PointerPointer TessBaseAPIGetLoadedLanguagesAsVector(@Const TessBaseAPI handle);
public static native @Cast("char**") PointerPointer TessBaseAPIGetAvailableLanguagesAsVector(@Const TessBaseAPI handle);

public static native int TessBaseAPIInitLangMod(TessBaseAPI handle, @Cast("const char*") BytePointer datapath, @Cast("const char*") BytePointer language);
public static native int TessBaseAPIInitLangMod(TessBaseAPI handle, String datapath, String language);
public static native void TessBaseAPIInitForAnalysePage(TessBaseAPI handle);

public static native void TessBaseAPIReadConfigFile(TessBaseAPI handle, @Cast("const char*") BytePointer filename);
public static native void TessBaseAPIReadConfigFile(TessBaseAPI handle, String filename);
public static native void TessBaseAPIReadDebugConfigFile(TessBaseAPI handle, @Cast("const char*") BytePointer filename);
public static native void TessBaseAPIReadDebugConfigFile(TessBaseAPI handle, String filename);

public static native void TessBaseAPISetPageSegMode(TessBaseAPI handle, @Cast("TessPageSegMode") int mode);
public static native @Cast("TessPageSegMode") int TessBaseAPIGetPageSegMode(@Const TessBaseAPI handle);

public static native @Cast("char*") BytePointer TessBaseAPIRect(TessBaseAPI handle, @Cast("const unsigned char*") BytePointer imagedata,
                                         int bytes_per_pixel, int bytes_per_line,
                                         int left, int top, int width, int height);
public static native @Cast("char*") ByteBuffer TessBaseAPIRect(TessBaseAPI handle, @Cast("const unsigned char*") ByteBuffer imagedata,
                                         int bytes_per_pixel, int bytes_per_line,
                                         int left, int top, int width, int height);
public static native @Cast("char*") byte[] TessBaseAPIRect(TessBaseAPI handle, @Cast("const unsigned char*") byte[] imagedata,
                                         int bytes_per_pixel, int bytes_per_line,
                                         int left, int top, int width, int height);

public static native void TessBaseAPIClearAdaptiveClassifier(TessBaseAPI handle);

public static native void TessBaseAPISetImage(TessBaseAPI handle, @Cast("const unsigned char*") BytePointer imagedata, int width, int height,
                                             int bytes_per_pixel, int bytes_per_line);
public static native void TessBaseAPISetImage(TessBaseAPI handle, @Cast("const unsigned char*") ByteBuffer imagedata, int width, int height,
                                             int bytes_per_pixel, int bytes_per_line);
public static native void TessBaseAPISetImage(TessBaseAPI handle, @Cast("const unsigned char*") byte[] imagedata, int width, int height,
                                             int bytes_per_pixel, int bytes_per_line);
public static native void TessBaseAPISetImage2(TessBaseAPI handle, PIX pix);

public static native void TessBaseAPISetSourceResolution(TessBaseAPI handle, int ppi);

public static native void TessBaseAPISetRectangle(TessBaseAPI handle, int left, int top, int width, int height);

// #ifdef TESS_CAPI_INCLUDE_BASEAPI
public static native void TessBaseAPISetThresholder(TessBaseAPI handle, @Cast("TessImageThresholder*") ImageThresholder thresholder);
// #endif

public static native PIX TessBaseAPIGetThresholdedImage(   TessBaseAPI handle);
public static native BOXA TessBaseAPIGetRegions(            TessBaseAPI handle, @Cast("Pixa**") PointerPointer pixa);
public static native BOXA TessBaseAPIGetRegions(            TessBaseAPI handle, @ByPtrPtr PIXA pixa);
public static native BOXA TessBaseAPIGetTextlines(          TessBaseAPI handle, @Cast("Pixa**") PointerPointer pixa, @Cast("int**") PointerPointer blockids);
public static native BOXA TessBaseAPIGetTextlines(          TessBaseAPI handle, @ByPtrPtr PIXA pixa, @ByPtrPtr IntPointer blockids);
public static native BOXA TessBaseAPIGetTextlines(          TessBaseAPI handle, @ByPtrPtr PIXA pixa, @ByPtrPtr IntBuffer blockids);
public static native BOXA TessBaseAPIGetTextlines(          TessBaseAPI handle, @ByPtrPtr PIXA pixa, @ByPtrPtr int[] blockids);
public static native BOXA TessBaseAPIGetTextlines1(         TessBaseAPI handle, @Cast("const BOOL") boolean raw_image, int raw_padding,
                                                                                @Cast("Pixa**") PointerPointer pixa, @Cast("int**") PointerPointer blockids, @Cast("int**") PointerPointer paraids);
public static native BOXA TessBaseAPIGetTextlines1(         TessBaseAPI handle, @Cast("const BOOL") boolean raw_image, int raw_padding,
                                                                                @ByPtrPtr PIXA pixa, @ByPtrPtr IntPointer blockids, @ByPtrPtr IntPointer paraids);
public static native BOXA TessBaseAPIGetTextlines1(         TessBaseAPI handle, @Cast("const BOOL") boolean raw_image, int raw_padding,
                                                                                @ByPtrPtr PIXA pixa, @ByPtrPtr IntBuffer blockids, @ByPtrPtr IntBuffer paraids);
public static native BOXA TessBaseAPIGetTextlines1(         TessBaseAPI handle, @Cast("const BOOL") boolean raw_image, int raw_padding,
                                                                                @ByPtrPtr PIXA pixa, @ByPtrPtr int[] blockids, @ByPtrPtr int[] paraids);
public static native BOXA TessBaseAPIGetStrips(             TessBaseAPI handle, @Cast("Pixa**") PointerPointer pixa, @Cast("int**") PointerPointer blockids);
public static native BOXA TessBaseAPIGetStrips(             TessBaseAPI handle, @ByPtrPtr PIXA pixa, @ByPtrPtr IntPointer blockids);
public static native BOXA TessBaseAPIGetStrips(             TessBaseAPI handle, @ByPtrPtr PIXA pixa, @ByPtrPtr IntBuffer blockids);
public static native BOXA TessBaseAPIGetStrips(             TessBaseAPI handle, @ByPtrPtr PIXA pixa, @ByPtrPtr int[] blockids);
public static native BOXA TessBaseAPIGetWords(              TessBaseAPI handle, @Cast("Pixa**") PointerPointer pixa);
public static native BOXA TessBaseAPIGetWords(              TessBaseAPI handle, @ByPtrPtr PIXA pixa);
public static native BOXA TessBaseAPIGetConnectedComponents(TessBaseAPI handle, @Cast("Pixa**") PointerPointer cc);
public static native BOXA TessBaseAPIGetConnectedComponents(TessBaseAPI handle, @ByPtrPtr PIXA cc);
public static native BOXA TessBaseAPIGetComponentImages(    TessBaseAPI handle, @Cast("const TessPageIteratorLevel") int level, @Cast("const BOOL") boolean text_only,
                                                           @Cast("Pixa**") PointerPointer pixa, @Cast("int**") PointerPointer blockids);
public static native BOXA TessBaseAPIGetComponentImages(    TessBaseAPI handle, @Cast("const TessPageIteratorLevel") int level, @Cast("const BOOL") boolean text_only,
                                                           @ByPtrPtr PIXA pixa, @ByPtrPtr IntPointer blockids);
public static native BOXA TessBaseAPIGetComponentImages(    TessBaseAPI handle, @Cast("const TessPageIteratorLevel") int level, @Cast("const BOOL") boolean text_only,
                                                           @ByPtrPtr PIXA pixa, @ByPtrPtr IntBuffer blockids);
public static native BOXA TessBaseAPIGetComponentImages(    TessBaseAPI handle, @Cast("const TessPageIteratorLevel") int level, @Cast("const BOOL") boolean text_only,
                                                           @ByPtrPtr PIXA pixa, @ByPtrPtr int[] blockids);
public static native BOXA TessBaseAPIGetComponentImages1(   TessBaseAPI handle, @Cast("const TessPageIteratorLevel") int level, @Cast("const BOOL") boolean text_only,
                                                           @Cast("const BOOL") boolean raw_image, int raw_padding,
                                                           @Cast("Pixa**") PointerPointer pixa, @Cast("int**") PointerPointer blockids, @Cast("int**") PointerPointer paraids);
public static native BOXA TessBaseAPIGetComponentImages1(   TessBaseAPI handle, @Cast("const TessPageIteratorLevel") int level, @Cast("const BOOL") boolean text_only,
                                                           @Cast("const BOOL") boolean raw_image, int raw_padding,
                                                           @ByPtrPtr PIXA pixa, @ByPtrPtr IntPointer blockids, @ByPtrPtr IntPointer paraids);
public static native BOXA TessBaseAPIGetComponentImages1(   TessBaseAPI handle, @Cast("const TessPageIteratorLevel") int level, @Cast("const BOOL") boolean text_only,
                                                           @Cast("const BOOL") boolean raw_image, int raw_padding,
                                                           @ByPtrPtr PIXA pixa, @ByPtrPtr IntBuffer blockids, @ByPtrPtr IntBuffer paraids);
public static native BOXA TessBaseAPIGetComponentImages1(   TessBaseAPI handle, @Cast("const TessPageIteratorLevel") int level, @Cast("const BOOL") boolean text_only,
                                                           @Cast("const BOOL") boolean raw_image, int raw_padding,
                                                           @ByPtrPtr PIXA pixa, @ByPtrPtr int[] blockids, @ByPtrPtr int[] paraids);

public static native int TessBaseAPIGetThresholdedImageScaleFactor(@Const TessBaseAPI handle);

public static native void TessBaseAPIDumpPGM(TessBaseAPI handle, @Cast("const char*") BytePointer filename);
public static native void TessBaseAPIDumpPGM(TessBaseAPI handle, String filename);

public static native @Cast("TessPageIterator*") PageIterator TessBaseAPIAnalyseLayout(TessBaseAPI handle);

public static native int TessBaseAPIRecognize(TessBaseAPI handle, ETEXT_DESC monitor);
public static native int TessBaseAPIRecognizeForChopTest(TessBaseAPI handle, ETEXT_DESC monitor);
public static native @Cast("BOOL") boolean TessBaseAPIProcessPages(TessBaseAPI handle,  @Cast("const char*") BytePointer filename, @Cast("const char*") BytePointer retry_config,
                                                 int timeout_millisec, TessResultRenderer renderer);
public static native @Cast("BOOL") boolean TessBaseAPIProcessPages(TessBaseAPI handle,  String filename, String retry_config,
                                                 int timeout_millisec, TessResultRenderer renderer);
public static native @Cast("BOOL") boolean TessBaseAPIProcessPage(TessBaseAPI handle, PIX pix, int page_index, @Cast("const char*") BytePointer filename,
                                               @Cast("const char*") BytePointer retry_config, int timeout_millisec, TessResultRenderer renderer);
public static native @Cast("BOOL") boolean TessBaseAPIProcessPage(TessBaseAPI handle, PIX pix, int page_index, String filename,
                                               String retry_config, int timeout_millisec, TessResultRenderer renderer);

public static native @Cast("TessResultIterator*") ResultIterator TessBaseAPIGetIterator(TessBaseAPI handle);
public static native @Cast("TessMutableIterator*") MutableIterator TessBaseAPIGetMutableIterator(TessBaseAPI handle);

public static native @Cast("char*") BytePointer TessBaseAPIGetUTF8Text(TessBaseAPI handle);
public static native @Cast("char*") BytePointer TessBaseAPIGetHOCRText(TessBaseAPI handle, int page_number);
public static native @Cast("char*") BytePointer TessBaseAPIGetBoxText(TessBaseAPI handle, int page_number);
public static native @Cast("char*") BytePointer TessBaseAPIGetUNLVText(TessBaseAPI handle);
public static native int TessBaseAPIMeanTextConf(TessBaseAPI handle);
public static native IntPointer TessBaseAPIAllWordConfidences(TessBaseAPI handle);
public static native @Cast("BOOL") boolean TessBaseAPIAdaptToWordStr(TessBaseAPI handle, @Cast("TessPageSegMode") int mode, @Cast("const char*") BytePointer wordstr);
public static native @Cast("BOOL") boolean TessBaseAPIAdaptToWordStr(TessBaseAPI handle, @Cast("TessPageSegMode") int mode, String wordstr);

public static native void TessBaseAPIClear(TessBaseAPI handle);
public static native void TessBaseAPIEnd(TessBaseAPI handle);

public static native int TessBaseAPIIsValidWord(TessBaseAPI handle, @Cast("const char*") BytePointer word);
public static native int TessBaseAPIIsValidWord(TessBaseAPI handle, String word);
public static native @Cast("BOOL") boolean TessBaseAPIGetTextDirection(TessBaseAPI handle, IntPointer out_offset, FloatPointer out_slope);
public static native @Cast("BOOL") boolean TessBaseAPIGetTextDirection(TessBaseAPI handle, IntBuffer out_offset, FloatBuffer out_slope);
public static native @Cast("BOOL") boolean TessBaseAPIGetTextDirection(TessBaseAPI handle, int[] out_offset, float[] out_slope);

// #ifdef TESS_CAPI_INCLUDE_BASEAPI
public static native void TessBaseAPISetDictFunc(TessBaseAPI handle, @Cast("TessDictFunc") DictFunc f);
public static native void TessBaseAPIClearPersistentCache(TessBaseAPI handle);
public static native void TessBaseAPISetProbabilityInContextFunc(TessBaseAPI handle, @Cast("TessProbabilityInContextFunc") ProbabilityInContextFunc f);


public static native @Cast("BOOL") boolean TessBaseAPIDetectOS(TessBaseAPI handle, OSResults results);

public static native void TessBaseAPIGetFeaturesForBlob(TessBaseAPI handle, TBLOB blob, INT_FEATURE_STRUCT int_features,
                                                       IntPointer num_features, IntPointer FeatureOutlineIndex);
public static native void TessBaseAPIGetFeaturesForBlob(TessBaseAPI handle, TBLOB blob, INT_FEATURE_STRUCT int_features,
                                                       IntBuffer num_features, IntBuffer FeatureOutlineIndex);
public static native void TessBaseAPIGetFeaturesForBlob(TessBaseAPI handle, TBLOB blob, INT_FEATURE_STRUCT int_features,
                                                       int[] num_features, int[] FeatureOutlineIndex);

public static native ROW TessFindRowForBox(BLOCK_LIST blocks, int left, int top, int right, int bottom);
public static native void TessBaseAPIRunAdaptiveClassifier(TessBaseAPI handle, TBLOB blob, int num_max_matches,
                                                          IntPointer unichar_ids, FloatPointer ratings, IntPointer num_matches_returned);
public static native void TessBaseAPIRunAdaptiveClassifier(TessBaseAPI handle, TBLOB blob, int num_max_matches,
                                                          IntBuffer unichar_ids, FloatBuffer ratings, IntBuffer num_matches_returned);
public static native void TessBaseAPIRunAdaptiveClassifier(TessBaseAPI handle, TBLOB blob, int num_max_matches,
                                                          int[] unichar_ids, float[] ratings, int[] num_matches_returned);
// #endif

public static native @Cast("const char*") BytePointer TessBaseAPIGetUnichar(TessBaseAPI handle, int unichar_id);

// #ifdef TESS_CAPI_INCLUDE_BASEAPI
public static native @Cast("const TessDawg*") Dawg TessBaseAPIGetDawg(@Const TessBaseAPI handle, int i);
public static native int TessBaseAPINumDawgs(@Const TessBaseAPI handle);
// #endif

// #ifdef TESS_CAPI_INCLUDE_BASEAPI
public static native ROW TessMakeTessOCRRow(float baseline, float xheight, float descender, float ascender);
public static native TBLOB TessMakeTBLOB(PIX pix);
public static native void TessNormalizeTBLOB(TBLOB tblob, ROW row, @Cast("BOOL") boolean numeric_mode);

public static native @Cast("TessOcrEngineMode") int TessBaseAPIOem(@Const TessBaseAPI handle);
public static native void TessBaseAPIInitTruthCallback(TessBaseAPI handle, @Cast("TessTruthCallback*") TruthCallback4 cb);

public static native @Cast("TessCubeRecoContext*") CubeRecoContext TessBaseAPIGetCubeRecoContext(@Const TessBaseAPI handle);
// #endif

public static native void TessBaseAPISetMinOrientationMargin(TessBaseAPI handle, double margin);
// #ifdef TESS_CAPI_INCLUDE_BASEAPI


public static native BLOCK_LIST TessBaseAPIFindLinesCreateBlockList(TessBaseAPI handle);
// #endif

/* Page iterator */

public static native void TessPageIteratorDelete(@Cast("TessPageIterator*") PageIterator handle);
public static native @Cast("TessPageIterator*") PageIterator TessPageIteratorCopy(@Cast("const TessPageIterator*") PageIterator handle);

public static native void TessPageIteratorBegin(@Cast("TessPageIterator*") PageIterator handle);
public static native @Cast("BOOL") boolean TessPageIteratorNext(@Cast("TessPageIterator*") PageIterator handle, @Cast("TessPageIteratorLevel") int level);
public static native @Cast("BOOL") boolean TessPageIteratorIsAtBeginningOf(@Cast("const TessPageIterator*") PageIterator handle, @Cast("TessPageIteratorLevel") int level);
public static native @Cast("BOOL") boolean TessPageIteratorIsAtFinalElement(@Cast("const TessPageIterator*") PageIterator handle, @Cast("TessPageIteratorLevel") int level,
                                                          @Cast("TessPageIteratorLevel") int element);

public static native @Cast("BOOL") boolean TessPageIteratorBoundingBox(@Cast("const TessPageIterator*") PageIterator handle, @Cast("TessPageIteratorLevel") int level,
                                                     IntPointer left, IntPointer top, IntPointer right, IntPointer bottom);
public static native @Cast("BOOL") boolean TessPageIteratorBoundingBox(@Cast("const TessPageIterator*") PageIterator handle, @Cast("TessPageIteratorLevel") int level,
                                                     IntBuffer left, IntBuffer top, IntBuffer right, IntBuffer bottom);
public static native @Cast("BOOL") boolean TessPageIteratorBoundingBox(@Cast("const TessPageIterator*") PageIterator handle, @Cast("TessPageIteratorLevel") int level,
                                                     int[] left, int[] top, int[] right, int[] bottom);
public static native @Cast("TessPolyBlockType") int TessPageIteratorBlockType(@Cast("const TessPageIterator*") PageIterator handle);

public static native PIX TessPageIteratorGetBinaryImage(@Cast("const TessPageIterator*") PageIterator handle, @Cast("TessPageIteratorLevel") int level);
public static native PIX TessPageIteratorGetImage(@Cast("const TessPageIterator*") PageIterator handle, @Cast("TessPageIteratorLevel") int level, int padding,
                                                  PIX original_image, IntPointer left, IntPointer top);
public static native PIX TessPageIteratorGetImage(@Cast("const TessPageIterator*") PageIterator handle, @Cast("TessPageIteratorLevel") int level, int padding,
                                                  PIX original_image, IntBuffer left, IntBuffer top);
public static native PIX TessPageIteratorGetImage(@Cast("const TessPageIterator*") PageIterator handle, @Cast("TessPageIteratorLevel") int level, int padding,
                                                  PIX original_image, int[] left, int[] top);

public static native @Cast("BOOL") boolean TessPageIteratorBaseline(@Cast("const TessPageIterator*") PageIterator handle, @Cast("TessPageIteratorLevel") int level,
                                                  IntPointer x1, IntPointer y1, IntPointer x2, IntPointer y2);
public static native @Cast("BOOL") boolean TessPageIteratorBaseline(@Cast("const TessPageIterator*") PageIterator handle, @Cast("TessPageIteratorLevel") int level,
                                                  IntBuffer x1, IntBuffer y1, IntBuffer x2, IntBuffer y2);
public static native @Cast("BOOL") boolean TessPageIteratorBaseline(@Cast("const TessPageIterator*") PageIterator handle, @Cast("TessPageIteratorLevel") int level,
                                                  int[] x1, int[] y1, int[] x2, int[] y2);

public static native void TessPageIteratorOrientation(@Cast("TessPageIterator*") PageIterator handle, @Cast("TessOrientation*") IntPointer orientation,
                                                     @Cast("TessWritingDirection*") IntPointer writing_direction, @Cast("TessTextlineOrder*") IntPointer textline_order,
                                                     FloatPointer deskew_angle);
public static native void TessPageIteratorOrientation(@Cast("TessPageIterator*") PageIterator handle, @Cast("TessOrientation*") IntBuffer orientation,
                                                     @Cast("TessWritingDirection*") IntBuffer writing_direction, @Cast("TessTextlineOrder*") IntBuffer textline_order,
                                                     FloatBuffer deskew_angle);
public static native void TessPageIteratorOrientation(@Cast("TessPageIterator*") PageIterator handle, @Cast("TessOrientation*") int[] orientation,
                                                     @Cast("TessWritingDirection*") int[] writing_direction, @Cast("TessTextlineOrder*") int[] textline_order,
                                                     float[] deskew_angle);

public static native void TessPageIteratorParagraphInfo(@Cast("TessPageIterator*") PageIterator handle, @Cast("TessParagraphJustification*") IntPointer justification,
                                                       @Cast("BOOL*") BoolPointer is_list_item, @Cast("BOOL*") BoolPointer is_crown, IntPointer first_line_indent);
public static native void TessPageIteratorParagraphInfo(@Cast("TessPageIterator*") PageIterator handle, @Cast("TessParagraphJustification*") IntBuffer justification,
                                                       @Cast("BOOL*") BoolPointer is_list_item, @Cast("BOOL*") BoolPointer is_crown, IntBuffer first_line_indent);
public static native void TessPageIteratorParagraphInfo(@Cast("TessPageIterator*") PageIterator handle, @Cast("TessParagraphJustification*") int[] justification,
                                                       @Cast("BOOL*") BoolPointer is_list_item, @Cast("BOOL*") BoolPointer is_crown, int[] first_line_indent);

/* Result iterator */

public static native void TessResultIteratorDelete(@Cast("TessResultIterator*") ResultIterator handle);
public static native @Cast("TessResultIterator*") ResultIterator TessResultIteratorCopy(@Cast("const TessResultIterator*") ResultIterator handle);
public static native @Cast("TessPageIterator*") PageIterator TessResultIteratorGetPageIterator(@Cast("TessResultIterator*") ResultIterator handle);
public static native @Cast("const TessPageIterator*") PageIterator TessResultIteratorGetPageIteratorConst(@Cast("const TessResultIterator*") ResultIterator handle);
public static native @Cast("TessChoiceIterator*") ChoiceIterator TessResultIteratorGetChoiceIterator(@Cast("const TessResultIterator*") ResultIterator handle);

public static native @Cast("BOOL") boolean TessResultIteratorNext(@Cast("TessResultIterator*") ResultIterator handle, @Cast("TessPageIteratorLevel") int level);
public static native @Cast("char*") BytePointer TessResultIteratorGetUTF8Text(@Cast("const TessResultIterator*") ResultIterator handle, @Cast("TessPageIteratorLevel") int level);
public static native float TessResultIteratorConfidence(@Cast("const TessResultIterator*") ResultIterator handle, @Cast("TessPageIteratorLevel") int level);
public static native @Cast("const char*") BytePointer TessResultIteratorWordRecognitionLanguage(@Cast("const TessResultIterator*") ResultIterator handle);
public static native @Cast("const char*") BytePointer TessResultIteratorWordFontAttributes(@Cast("const TessResultIterator*") ResultIterator handle, @Cast("BOOL*") BoolPointer is_bold, @Cast("BOOL*") BoolPointer is_italic,
                                                              @Cast("BOOL*") BoolPointer is_underlined, @Cast("BOOL*") BoolPointer is_monospace, @Cast("BOOL*") BoolPointer is_serif,
                                                              @Cast("BOOL*") BoolPointer is_smallcaps, IntPointer pointsize, IntPointer font_id);
public static native String TessResultIteratorWordFontAttributes(@Cast("const TessResultIterator*") ResultIterator handle, @Cast("BOOL*") BoolPointer is_bold, @Cast("BOOL*") BoolPointer is_italic,
                                                              @Cast("BOOL*") BoolPointer is_underlined, @Cast("BOOL*") BoolPointer is_monospace, @Cast("BOOL*") BoolPointer is_serif,
                                                              @Cast("BOOL*") BoolPointer is_smallcaps, IntBuffer pointsize, IntBuffer font_id);
public static native @Cast("const char*") BytePointer TessResultIteratorWordFontAttributes(@Cast("const TessResultIterator*") ResultIterator handle, @Cast("BOOL*") BoolPointer is_bold, @Cast("BOOL*") BoolPointer is_italic,
                                                              @Cast("BOOL*") BoolPointer is_underlined, @Cast("BOOL*") BoolPointer is_monospace, @Cast("BOOL*") BoolPointer is_serif,
                                                              @Cast("BOOL*") BoolPointer is_smallcaps, int[] pointsize, int[] font_id);

public static native @Cast("BOOL") boolean TessResultIteratorWordIsFromDictionary(@Cast("const TessResultIterator*") ResultIterator handle);
public static native @Cast("BOOL") boolean TessResultIteratorWordIsNumeric(@Cast("const TessResultIterator*") ResultIterator handle);
public static native @Cast("BOOL") boolean TessResultIteratorSymbolIsSuperscript(@Cast("const TessResultIterator*") ResultIterator handle);
public static native @Cast("BOOL") boolean TessResultIteratorSymbolIsSubscript(@Cast("const TessResultIterator*") ResultIterator handle);
public static native @Cast("BOOL") boolean TessResultIteratorSymbolIsDropcap(@Cast("const TessResultIterator*") ResultIterator handle);

public static native void TessChoiceIteratorDelete(@Cast("TessChoiceIterator*") ChoiceIterator handle);
public static native @Cast("BOOL") boolean TessChoiceIteratorNext(@Cast("TessChoiceIterator*") ChoiceIterator handle);
public static native @Cast("const char*") BytePointer TessChoiceIteratorGetUTF8Text(@Cast("const TessChoiceIterator*") ChoiceIterator handle);
public static native float TessChoiceIteratorConfidence(@Cast("const TessChoiceIterator*") ChoiceIterator handle);

// #ifdef __cplusplus
// #endif

// #endif /* TESSERACT_API_CAPI_H__ */


}
