// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.tesseract;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import org.bytedeco.leptonica.*;
import static org.bytedeco.leptonica.global.leptonica.*;

import static org.bytedeco.tesseract.global.tesseract.*;


// The UNICHAR class holds a single classification result. This may be
// a single Unicode character (stored as between 1 and 4 utf8 bytes) or
// multiple Unicode characters representing the NFKC expansion of a ligature
// such as fi, ffl etc. These are also stored as utf8.
@Namespace("tesseract") @NoOffset @Properties(inherit = org.bytedeco.tesseract.presets.tesseract.class)
public class UNICHAR extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public UNICHAR(Pointer p) { super(p); }

  public UNICHAR() { super((Pointer)null); allocate(); }
  private native void allocate();

  // Construct from a utf8 string. If len<0 then the string is null terminated.
  // If the string is too long to fit in the UNICHAR then it takes only what
  // will fit.
  public UNICHAR(@Cast("const char*") BytePointer utf8_str, int len) { super((Pointer)null); allocate(utf8_str, len); }
  private native void allocate(@Cast("const char*") BytePointer utf8_str, int len);
  public UNICHAR(String utf8_str, int len) { super((Pointer)null); allocate(utf8_str, len); }
  private native void allocate(String utf8_str, int len);

  // Construct from a single UCS4 character.
  public UNICHAR(int unicode) { super((Pointer)null); allocate(unicode); }
  private native void allocate(int unicode);

  // Default copy constructor and operator= are OK.

  // Get the first character as UCS-4.
  public native int first_uni();

  // Get the length of the UTF8 string.
  public native int utf8_len();

  // Get a UTF8 string, but NOT nullptr terminated.
  public native @Cast("const char*") BytePointer utf8();

  // Get a terminated UTF8 string: Must delete[] it after use.
  public native @Cast("char*") BytePointer utf8_str();

  // Get the number of bytes in the first character of the given utf8 string.
  public static native int utf8_step(@Cast("const char*") BytePointer utf8_str);
  public static native int utf8_step(String utf8_str);

  // A class to simplify iterating over and accessing elements of a UTF8
  // string. Note that unlike the UNICHAR class, const_iterator does NOT COPY or
  // take ownership of the underlying byte array. It also does not permit
  // modification of the array (as the name suggests).
  //
  // Example:
  //   for (UNICHAR::const_iterator it = UNICHAR::begin(str, str_len);
  //        it != UNICHAR::end(str, len);
  //        ++it) {
  //     printf("UCS-4 symbol code = %d\n", *it);
  //     char buf[5];
  //     int char_len = it.get_utf8(buf); buf[char_len] = '\0';
  //     printf("Char = %s\n", buf);
  //   }
  @NoOffset public static class const_iterator extends Pointer {
      static { Loader.load(); }
      /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
      public const_iterator(Pointer p) { super(p); }
  
    // Step to the next UTF8 character.
    // If the current position is at an illegal UTF8 character, then print an
    // error message and step by one byte. If the current position is at a
    // nullptr value, don't step past it.
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
  // Returns an empty vector if the input contains invalid UTF-8.
  public static native @Cast("tesseract::char32*") @StdVector IntPointer UTF8ToUTF32(@Cast("const char*") BytePointer utf8_str);
  public static native @Cast("tesseract::char32*") @StdVector IntBuffer UTF8ToUTF32(String utf8_str);
  // Converts a vector of unicodes to a utf8 string.
  // Returns an empty string if the input contains an invalid unicode.
  public static native @StdString BytePointer UTF32ToUTF8(@Cast("tesseract::char32*") @StdVector IntPointer str32);
  public static native @StdString String UTF32ToUTF8(@Cast("tesseract::char32*") @StdVector IntBuffer str32);
  public static native @StdString BytePointer UTF32ToUTF8(@Cast("tesseract::char32*") @StdVector int[] str32);
}
