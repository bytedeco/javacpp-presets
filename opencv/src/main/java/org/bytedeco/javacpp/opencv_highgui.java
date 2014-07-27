// Targeted by JavaCPP version 0.9

package org.bytedeco.javacpp;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;

public class opencv_highgui extends org.bytedeco.javacpp.helper.opencv_highgui {
    static { Loader.load(); }

// Parsed from <opencv2/highgui/highgui_c.h>

/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

// #ifndef __OPENCV_HIGHGUI_H__
// #define __OPENCV_HIGHGUI_H__

// #include "opencv2/core/core_c.h"

// #ifdef __cplusplus
// #endif /* __cplusplus */

/****************************************************************************************\
*                                  Basic GUI functions                                   *
\****************************************************************************************/
//YV
//-----------New for Qt
/* For font */
/** enum  */
public static final int  CV_FONT_LIGHT           = 25,//QFont::Light,
        CV_FONT_NORMAL          = 50,//QFont::Normal,
        CV_FONT_DEMIBOLD        = 63,//QFont::DemiBold,
        CV_FONT_BOLD            = 75,//QFont::Bold,
        CV_FONT_BLACK           = 87; //QFont::Black

/** enum  */
public static final int  CV_STYLE_NORMAL         = 0,//QFont::StyleNormal,
        CV_STYLE_ITALIC         = 1,//QFont::StyleItalic,
        CV_STYLE_OBLIQUE        = 2; //QFont::StyleOblique
/* ---------*/

//for color cvScalar(blue_component, green_component, red\_component[, alpha_component])
//and alpha= 0 <-> 0xFF (not transparent <-> transparent)
public static native @ByVal @Platform("linux") CvFont cvFontQt(@Cast("const char*") BytePointer nameFont, int pointSize/*=-1*/, @ByVal CvScalar color/*=cvScalarAll(0)*/, int weight/*=CV_FONT_NORMAL*/,  int style/*=CV_STYLE_NORMAL*/, int spacing/*=0*/);
public static native @ByVal @Platform("linux") CvFont cvFontQt(@Cast("const char*") BytePointer nameFont);
public static native @ByVal @Platform("linux") CvFont cvFontQt(String nameFont, int pointSize/*=-1*/, @ByVal CvScalar color/*=cvScalarAll(0)*/, int weight/*=CV_FONT_NORMAL*/,  int style/*=CV_STYLE_NORMAL*/, int spacing/*=0*/);
public static native @ByVal @Platform("linux") CvFont cvFontQt(String nameFont);

public static native @Platform("linux") void cvAddText(@Const CvArr img, @Cast("const char*") BytePointer text, @ByVal CvPoint org, CvFont arg2);
public static native @Platform("linux") void cvAddText(@Const CvArr img, String text, @ByVal @Cast("CvPoint*") IntBuffer org, CvFont arg2);
public static native @Platform("linux") void cvAddText(@Const CvArr img, @Cast("const char*") BytePointer text, @ByVal @Cast("CvPoint*") int[] org, CvFont arg2);
public static native @Platform("linux") void cvAddText(@Const CvArr img, String text, @ByVal CvPoint org, CvFont arg2);
public static native @Platform("linux") void cvAddText(@Const CvArr img, @Cast("const char*") BytePointer text, @ByVal @Cast("CvPoint*") IntBuffer org, CvFont arg2);
public static native @Platform("linux") void cvAddText(@Const CvArr img, String text, @ByVal @Cast("CvPoint*") int[] org, CvFont arg2);

public static native @Platform("linux") void cvDisplayOverlay(@Cast("const char*") BytePointer name, @Cast("const char*") BytePointer text, int delayms/*=0*/);
public static native @Platform("linux") void cvDisplayOverlay(@Cast("const char*") BytePointer name, @Cast("const char*") BytePointer text);
public static native @Platform("linux") void cvDisplayOverlay(String name, String text, int delayms/*=0*/);
public static native @Platform("linux") void cvDisplayOverlay(String name, String text);
public static native @Platform("linux") void cvDisplayStatusBar(@Cast("const char*") BytePointer name, @Cast("const char*") BytePointer text, int delayms/*=0*/);
public static native @Platform("linux") void cvDisplayStatusBar(@Cast("const char*") BytePointer name, @Cast("const char*") BytePointer text);
public static native @Platform("linux") void cvDisplayStatusBar(String name, String text, int delayms/*=0*/);
public static native @Platform("linux") void cvDisplayStatusBar(String name, String text);

public static native @Platform("linux") void cvSaveWindowParameters(@Cast("const char*") BytePointer name);
public static native @Platform("linux") void cvSaveWindowParameters(String name);
public static native @Platform("linux") void cvLoadWindowParameters(@Cast("const char*") BytePointer name);
public static native @Platform("linux") void cvLoadWindowParameters(String name);
public static class Pt2Func_int_PointerPointer extends FunctionPointer {
    static { Loader.load(); }
    public    Pt2Func_int_PointerPointer(Pointer p) { super(p); }
    protected Pt2Func_int_PointerPointer() { allocate(); }
    private native void allocate();
    public native int call(int argc, @Cast("char**") PointerPointer argv);
}
public static native @Platform("linux") int cvStartLoop(Pt2Func_int_PointerPointer pt2Func, int argc, @Cast("char**") PointerPointer argv);
public static class Pt2Func_int_BytePointer extends FunctionPointer {
    static { Loader.load(); }
    public    Pt2Func_int_BytePointer(Pointer p) { super(p); }
    protected Pt2Func_int_BytePointer() { allocate(); }
    private native void allocate();
    public native int call(int argc, @Cast("char**") @ByPtrPtr BytePointer argv);
}
public static native @Platform("linux") int cvStartLoop(Pt2Func_int_BytePointer pt2Func, int argc, @Cast("char**") @ByPtrPtr BytePointer argv);
public static class Pt2Func_int_ByteBuffer extends FunctionPointer {
    static { Loader.load(); }
    public    Pt2Func_int_ByteBuffer(Pointer p) { super(p); }
    protected Pt2Func_int_ByteBuffer() { allocate(); }
    private native void allocate();
    public native int call(int argc, @Cast("char**") @ByPtrPtr ByteBuffer argv);
}
public static native @Platform("linux") int cvStartLoop(Pt2Func_int_ByteBuffer pt2Func, int argc, @Cast("char**") @ByPtrPtr ByteBuffer argv);
public static class Pt2Func_int_byte__ extends FunctionPointer {
    static { Loader.load(); }
    public    Pt2Func_int_byte__(Pointer p) { super(p); }
    protected Pt2Func_int_byte__() { allocate(); }
    private native void allocate();
    public native int call(int argc, @Cast("char**") @ByPtrPtr byte[] argv);
}
public static native @Platform("linux") int cvStartLoop(Pt2Func_int_byte__ pt2Func, int argc, @Cast("char**") @ByPtrPtr byte[] argv);
public static native @Platform("linux") void cvStopLoop( );

@Convention("CV_CDECL") public static class CvButtonCallback extends FunctionPointer {
    static { Loader.load(); }
    public    CvButtonCallback(Pointer p) { super(p); }
    protected CvButtonCallback() { allocate(); }
    private native void allocate();
    public native void call(int state, Pointer userdata);
}
/** enum  */
public static final int CV_PUSH_BUTTON = 0, CV_CHECKBOX = 1, CV_RADIOBOX = 2;
public static native @Platform("linux") int cvCreateButton( @Cast("const char*") BytePointer button_name/*=NULL*/,CvButtonCallback on_change/*=NULL*/, Pointer userdata/*=NULL*/, int button_type/*=CV_PUSH_BUTTON*/, int initial_button_state/*=0*/);
public static native @Platform("linux") int cvCreateButton();
public static native @Platform("linux") int cvCreateButton( String button_name/*=NULL*/,CvButtonCallback on_change/*=NULL*/, Pointer userdata/*=NULL*/, int button_type/*=CV_PUSH_BUTTON*/, int initial_button_state/*=0*/);
//----------------------


/* this function is used to set some external parameters in case of X Window */
public static native int cvInitSystem( int argc, @Cast("char**") PointerPointer argv );
public static native int cvInitSystem( int argc, @Cast("char**") @ByPtrPtr BytePointer argv );
public static native int cvInitSystem( int argc, @Cast("char**") @ByPtrPtr ByteBuffer argv );
public static native int cvInitSystem( int argc, @Cast("char**") @ByPtrPtr byte[] argv );

public static native int cvStartWindowThread( );

// ---------  YV ---------
/** enum  */
public static final int
    //These 3 flags are used by cvSet/GetWindowProperty
    CV_WND_PROP_FULLSCREEN = 0, //to change/get window's fullscreen property
    CV_WND_PROP_AUTOSIZE   = 1, //to change/get window's autosize property
    CV_WND_PROP_ASPECTRATIO= 2, //to change/get window's aspectratio property
    CV_WND_PROP_OPENGL     = 3, //to change/get window's opengl support

    //These 2 flags are used by cvNamedWindow and cvSet/GetWindowProperty
    CV_WINDOW_NORMAL       =  0x00000000, //the user can resize the window (no constraint)  / also use to switch a fullscreen window to a normal size
    CV_WINDOW_AUTOSIZE     =  0x00000001, //the user cannot resize the window, the size is constrainted by the image displayed
    CV_WINDOW_OPENGL       =  0x00001000, //window with opengl support

    //Those flags are only for Qt
    CV_GUI_EXPANDED         =  0x00000000, //status bar and tool bar
    CV_GUI_NORMAL           =  0x00000010, //old fashious way

    //These 3 flags are used by cvNamedWindow and cvSet/GetWindowProperty
    CV_WINDOW_FULLSCREEN   = 1,//change the window to fullscreen
    CV_WINDOW_FREERATIO    =  0x00000100,//the image expends as much as it can (no ratio constraint)
    CV_WINDOW_KEEPRATIO    =  0x00000000;//the ration image is respected.

/* create window */
public static native int cvNamedWindow( @Cast("const char*") BytePointer name, int flags/*=CV_WINDOW_AUTOSIZE*/ );
public static native int cvNamedWindow( @Cast("const char*") BytePointer name );
public static native int cvNamedWindow( String name, int flags/*=CV_WINDOW_AUTOSIZE*/ );
public static native int cvNamedWindow( String name );

/* Set and Get Property of the window */
public static native void cvSetWindowProperty(@Cast("const char*") BytePointer name, int prop_id, double prop_value);
public static native void cvSetWindowProperty(String name, int prop_id, double prop_value);
public static native double cvGetWindowProperty(@Cast("const char*") BytePointer name, int prop_id);
public static native double cvGetWindowProperty(String name, int prop_id);

/* display image within window (highgui windows remember their content) */
public static native void cvShowImage( @Cast("const char*") BytePointer name, @Const CvArr image );
public static native void cvShowImage( String name, @Const CvArr image );

/* resize/move window */
public static native void cvResizeWindow( @Cast("const char*") BytePointer name, int width, int height );
public static native void cvResizeWindow( String name, int width, int height );
public static native void cvMoveWindow( @Cast("const char*") BytePointer name, int x, int y );
public static native void cvMoveWindow( String name, int x, int y );


/* destroy window and all the trackers associated with it */
public static native void cvDestroyWindow( @Cast("const char*") BytePointer name );
public static native void cvDestroyWindow( String name );

public static native void cvDestroyAllWindows();

/* get native window handle (HWND in case of Win32 and Widget in case of X Window) */
public static native Pointer cvGetWindowHandle( @Cast("const char*") BytePointer name );
public static native Pointer cvGetWindowHandle( String name );

/* get name of highgui window given its native handle */
public static native @Cast("const char*") BytePointer cvGetWindowName( Pointer window_handle );


@Convention("CV_CDECL") public static class CvTrackbarCallback extends FunctionPointer {
    static { Loader.load(); }
    public    CvTrackbarCallback(Pointer p) { super(p); }
    protected CvTrackbarCallback() { allocate(); }
    private native void allocate();
    public native void call(int pos);
}

/* create trackbar and display it on top of given window, set callback */
public static native int cvCreateTrackbar( @Cast("const char*") BytePointer trackbar_name, @Cast("const char*") BytePointer window_name,
                             IntPointer value, int count, CvTrackbarCallback on_change/*=NULL*/);
public static native int cvCreateTrackbar( @Cast("const char*") BytePointer trackbar_name, @Cast("const char*") BytePointer window_name,
                             IntPointer value, int count);
public static native int cvCreateTrackbar( String trackbar_name, String window_name,
                             IntBuffer value, int count, CvTrackbarCallback on_change/*=NULL*/);
public static native int cvCreateTrackbar( String trackbar_name, String window_name,
                             IntBuffer value, int count);
public static native int cvCreateTrackbar( @Cast("const char*") BytePointer trackbar_name, @Cast("const char*") BytePointer window_name,
                             int[] value, int count, CvTrackbarCallback on_change/*=NULL*/);
public static native int cvCreateTrackbar( @Cast("const char*") BytePointer trackbar_name, @Cast("const char*") BytePointer window_name,
                             int[] value, int count);
public static native int cvCreateTrackbar( String trackbar_name, String window_name,
                             IntPointer value, int count, CvTrackbarCallback on_change/*=NULL*/);
public static native int cvCreateTrackbar( String trackbar_name, String window_name,
                             IntPointer value, int count);
public static native int cvCreateTrackbar( @Cast("const char*") BytePointer trackbar_name, @Cast("const char*") BytePointer window_name,
                             IntBuffer value, int count, CvTrackbarCallback on_change/*=NULL*/);
public static native int cvCreateTrackbar( @Cast("const char*") BytePointer trackbar_name, @Cast("const char*") BytePointer window_name,
                             IntBuffer value, int count);
public static native int cvCreateTrackbar( String trackbar_name, String window_name,
                             int[] value, int count, CvTrackbarCallback on_change/*=NULL*/);
public static native int cvCreateTrackbar( String trackbar_name, String window_name,
                             int[] value, int count);

@Convention("CV_CDECL") public static class CvTrackbarCallback2 extends FunctionPointer {
    static { Loader.load(); }
    public    CvTrackbarCallback2(Pointer p) { super(p); }
    protected CvTrackbarCallback2() { allocate(); }
    private native void allocate();
    public native void call(int pos, Pointer userdata);
}

public static native int cvCreateTrackbar2( @Cast("const char*") BytePointer trackbar_name, @Cast("const char*") BytePointer window_name,
                              IntPointer value, int count, CvTrackbarCallback2 on_change,
                              Pointer userdata/*=0*/);
public static native int cvCreateTrackbar2( @Cast("const char*") BytePointer trackbar_name, @Cast("const char*") BytePointer window_name,
                              IntPointer value, int count, CvTrackbarCallback2 on_change);
public static native int cvCreateTrackbar2( String trackbar_name, String window_name,
                              IntBuffer value, int count, CvTrackbarCallback2 on_change,
                              Pointer userdata/*=0*/);
public static native int cvCreateTrackbar2( String trackbar_name, String window_name,
                              IntBuffer value, int count, CvTrackbarCallback2 on_change);
public static native int cvCreateTrackbar2( @Cast("const char*") BytePointer trackbar_name, @Cast("const char*") BytePointer window_name,
                              int[] value, int count, CvTrackbarCallback2 on_change,
                              Pointer userdata/*=0*/);
public static native int cvCreateTrackbar2( @Cast("const char*") BytePointer trackbar_name, @Cast("const char*") BytePointer window_name,
                              int[] value, int count, CvTrackbarCallback2 on_change);
public static native int cvCreateTrackbar2( String trackbar_name, String window_name,
                              IntPointer value, int count, CvTrackbarCallback2 on_change,
                              Pointer userdata/*=0*/);
public static native int cvCreateTrackbar2( String trackbar_name, String window_name,
                              IntPointer value, int count, CvTrackbarCallback2 on_change);
public static native int cvCreateTrackbar2( @Cast("const char*") BytePointer trackbar_name, @Cast("const char*") BytePointer window_name,
                              IntBuffer value, int count, CvTrackbarCallback2 on_change,
                              Pointer userdata/*=0*/);
public static native int cvCreateTrackbar2( @Cast("const char*") BytePointer trackbar_name, @Cast("const char*") BytePointer window_name,
                              IntBuffer value, int count, CvTrackbarCallback2 on_change);
public static native int cvCreateTrackbar2( String trackbar_name, String window_name,
                              int[] value, int count, CvTrackbarCallback2 on_change,
                              Pointer userdata/*=0*/);
public static native int cvCreateTrackbar2( String trackbar_name, String window_name,
                              int[] value, int count, CvTrackbarCallback2 on_change);

/* retrieve or set trackbar position */
public static native int cvGetTrackbarPos( @Cast("const char*") BytePointer trackbar_name, @Cast("const char*") BytePointer window_name );
public static native int cvGetTrackbarPos( String trackbar_name, String window_name );
public static native void cvSetTrackbarPos( @Cast("const char*") BytePointer trackbar_name, @Cast("const char*") BytePointer window_name, int pos );
public static native void cvSetTrackbarPos( String trackbar_name, String window_name, int pos );

/** enum  */
public static final int
    CV_EVENT_MOUSEMOVE      = 0,
    CV_EVENT_LBUTTONDOWN    = 1,
    CV_EVENT_RBUTTONDOWN    = 2,
    CV_EVENT_MBUTTONDOWN    = 3,
    CV_EVENT_LBUTTONUP      = 4,
    CV_EVENT_RBUTTONUP      = 5,
    CV_EVENT_MBUTTONUP      = 6,
    CV_EVENT_LBUTTONDBLCLK  = 7,
    CV_EVENT_RBUTTONDBLCLK  = 8,
    CV_EVENT_MBUTTONDBLCLK  = 9;

/** enum  */
public static final int
    CV_EVENT_FLAG_LBUTTON   = 1,
    CV_EVENT_FLAG_RBUTTON   = 2,
    CV_EVENT_FLAG_MBUTTON   = 4,
    CV_EVENT_FLAG_CTRLKEY   = 8,
    CV_EVENT_FLAG_SHIFTKEY  = 16,
    CV_EVENT_FLAG_ALTKEY    = 32;

@Convention("CV_CDECL") public static class CvMouseCallback extends FunctionPointer {
    static { Loader.load(); }
    public    CvMouseCallback(Pointer p) { super(p); }
    protected CvMouseCallback() { allocate(); }
    private native void allocate();
    public native void call(int event, int x, int y, int flags, Pointer param);
}

/* assign callback for mouse events */
public static native void cvSetMouseCallback( @Cast("const char*") BytePointer window_name, CvMouseCallback on_mouse,
                                Pointer param/*=NULL*/);
public static native void cvSetMouseCallback( @Cast("const char*") BytePointer window_name, CvMouseCallback on_mouse);
public static native void cvSetMouseCallback( String window_name, CvMouseCallback on_mouse,
                                Pointer param/*=NULL*/);
public static native void cvSetMouseCallback( String window_name, CvMouseCallback on_mouse);

/** enum  */
public static final int
/* 8bit, color or not */
    CV_LOAD_IMAGE_UNCHANGED  = -1,
/* 8bit, gray */
    CV_LOAD_IMAGE_GRAYSCALE  = 0,
/* ?, color */
    CV_LOAD_IMAGE_COLOR      = 1,
/* any depth, ? */
    CV_LOAD_IMAGE_ANYDEPTH   = 2,
/* ?, any color */
    CV_LOAD_IMAGE_ANYCOLOR   = 4;

/* load image from file
  iscolor can be a combination of above flags where CV_LOAD_IMAGE_UNCHANGED
  overrides the other flags
  using CV_LOAD_IMAGE_ANYCOLOR alone is equivalent to CV_LOAD_IMAGE_UNCHANGED
  unless CV_LOAD_IMAGE_ANYDEPTH is specified images are converted to 8bit
*/
public static native IplImage cvLoadImage( @Cast("const char*") BytePointer filename, int iscolor/*=CV_LOAD_IMAGE_COLOR*/);
public static native IplImage cvLoadImage( @Cast("const char*") BytePointer filename);
public static native IplImage cvLoadImage( String filename, int iscolor/*=CV_LOAD_IMAGE_COLOR*/);
public static native IplImage cvLoadImage( String filename);
public static native CvMat cvLoadImageM( @Cast("const char*") BytePointer filename, int iscolor/*=CV_LOAD_IMAGE_COLOR*/);
public static native CvMat cvLoadImageM( @Cast("const char*") BytePointer filename);
public static native CvMat cvLoadImageM( String filename, int iscolor/*=CV_LOAD_IMAGE_COLOR*/);
public static native CvMat cvLoadImageM( String filename);

/** enum  */
public static final int
    CV_IMWRITE_JPEG_QUALITY = 1,
    CV_IMWRITE_PNG_COMPRESSION = 16,
    CV_IMWRITE_PNG_STRATEGY = 17,
    CV_IMWRITE_PNG_BILEVEL = 18,
    CV_IMWRITE_PNG_STRATEGY_DEFAULT = 0,
    CV_IMWRITE_PNG_STRATEGY_FILTERED = 1,
    CV_IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY = 2,
    CV_IMWRITE_PNG_STRATEGY_RLE = 3,
    CV_IMWRITE_PNG_STRATEGY_FIXED = 4,
    CV_IMWRITE_PXM_BINARY = 32;

/* save image to file */
public static native int cvSaveImage( @Cast("const char*") BytePointer filename, @Const CvArr image,
                        @Const IntPointer params/*=0*/ );
public static native int cvSaveImage( @Cast("const char*") BytePointer filename, @Const CvArr image );
public static native int cvSaveImage( String filename, @Const CvArr image,
                        @Const IntBuffer params/*=0*/ );
public static native int cvSaveImage( String filename, @Const CvArr image );
public static native int cvSaveImage( @Cast("const char*") BytePointer filename, @Const CvArr image,
                        @Const int[] params/*=0*/ );
public static native int cvSaveImage( String filename, @Const CvArr image,
                        @Const IntPointer params/*=0*/ );
public static native int cvSaveImage( @Cast("const char*") BytePointer filename, @Const CvArr image,
                        @Const IntBuffer params/*=0*/ );
public static native int cvSaveImage( String filename, @Const CvArr image,
                        @Const int[] params/*=0*/ );

/* decode image stored in the buffer */
public static native IplImage cvDecodeImage( @Const CvMat buf, int iscolor/*=CV_LOAD_IMAGE_COLOR*/);
public static native IplImage cvDecodeImage( @Const CvMat buf);
public static native CvMat cvDecodeImageM( @Const CvMat buf, int iscolor/*=CV_LOAD_IMAGE_COLOR*/);
public static native CvMat cvDecodeImageM( @Const CvMat buf);

/* encode image and store the result as a byte vector (single-row 8uC1 matrix) */
public static native CvMat cvEncodeImage( @Cast("const char*") BytePointer ext, @Const CvArr image,
                             @Const IntPointer params/*=0*/ );
public static native CvMat cvEncodeImage( @Cast("const char*") BytePointer ext, @Const CvArr image );
public static native CvMat cvEncodeImage( String ext, @Const CvArr image,
                             @Const IntBuffer params/*=0*/ );
public static native CvMat cvEncodeImage( String ext, @Const CvArr image );
public static native CvMat cvEncodeImage( @Cast("const char*") BytePointer ext, @Const CvArr image,
                             @Const int[] params/*=0*/ );
public static native CvMat cvEncodeImage( String ext, @Const CvArr image,
                             @Const IntPointer params/*=0*/ );
public static native CvMat cvEncodeImage( @Cast("const char*") BytePointer ext, @Const CvArr image,
                             @Const IntBuffer params/*=0*/ );
public static native CvMat cvEncodeImage( String ext, @Const CvArr image,
                             @Const int[] params/*=0*/ );

/** enum  */
public static final int
    CV_CVTIMG_FLIP      = 1,
    CV_CVTIMG_SWAP_RB   = 2;

/* utility function: convert one image to another with optional vertical flip */
public static native void cvConvertImage( @Const CvArr src, CvArr dst, int flags/*=0*/);
public static native void cvConvertImage( @Const CvArr src, CvArr dst);

/* wait for key event infinitely (delay<=0) or for "delay" milliseconds */
public static native int cvWaitKey(int delay/*=0*/);
public static native int cvWaitKey();

// OpenGL support

@Convention("CV_CDECL") public static class CvOpenGlDrawCallback extends FunctionPointer {
    static { Loader.load(); }
    public    CvOpenGlDrawCallback(Pointer p) { super(p); }
    protected CvOpenGlDrawCallback() { allocate(); }
    private native void allocate();
    public native void call(Pointer userdata);
}
public static native void cvSetOpenGlDrawCallback(@Cast("const char*") BytePointer window_name, CvOpenGlDrawCallback callback, Pointer userdata/*=NULL*/);
public static native void cvSetOpenGlDrawCallback(@Cast("const char*") BytePointer window_name, CvOpenGlDrawCallback callback);
public static native void cvSetOpenGlDrawCallback(String window_name, CvOpenGlDrawCallback callback, Pointer userdata/*=NULL*/);
public static native void cvSetOpenGlDrawCallback(String window_name, CvOpenGlDrawCallback callback);

public static native void cvSetOpenGlContext(@Cast("const char*") BytePointer window_name);
public static native void cvSetOpenGlContext(String window_name);
public static native void cvUpdateWindow(@Cast("const char*") BytePointer window_name);
public static native void cvUpdateWindow(String window_name);


/****************************************************************************************\
*                         Working with Video Files and Cameras                           *
\****************************************************************************************/

/* "black box" capture structure */
@Opaque public static class CvCapture extends Pointer {
    public CvCapture() { }
    public CvCapture(Pointer p) { super(p); }
}

/* start capturing frames from video file */
public static native CvCapture cvCreateFileCapture( @Cast("const char*") BytePointer filename );
public static native CvCapture cvCreateFileCapture( String filename );

/** enum  */
public static final int
    CV_CAP_ANY      = 0,     // autodetect

    CV_CAP_MIL      = 100,   // MIL proprietary drivers

    CV_CAP_VFW      = 200,   // platform native
    CV_CAP_V4L      = 200,
    CV_CAP_V4L2     = 200,

    CV_CAP_FIREWARE = 300,   // IEEE 1394 drivers
    CV_CAP_FIREWIRE = 300,
    CV_CAP_IEEE1394 = 300,
    CV_CAP_DC1394   = 300,
    CV_CAP_CMU1394  = 300,

    CV_CAP_STEREO   = 400,   // TYZX proprietary drivers
    CV_CAP_TYZX     = 400,
    CV_TYZX_LEFT    = 400,
    CV_TYZX_RIGHT   = 401,
    CV_TYZX_COLOR   = 402,
    CV_TYZX_Z       = 403,

    CV_CAP_QT       = 500,   // QuickTime

    CV_CAP_UNICAP   = 600,   // Unicap drivers

    CV_CAP_DSHOW    = 700,   // DirectShow (via videoInput)
    CV_CAP_MSMF     = 1400,  // Microsoft Media Foundation (via videoInput)

    CV_CAP_PVAPI    = 800,   // PvAPI, Prosilica GigE SDK

    CV_CAP_OPENNI   = 900,   // OpenNI (for Kinect)
    CV_CAP_OPENNI_ASUS = 910,   // OpenNI (for Asus Xtion)

    CV_CAP_ANDROID  = 1000,  // Android
    CV_CAP_ANDROID_BACK = CV_CAP_ANDROID+99, // Android back camera
    CV_CAP_ANDROID_FRONT = CV_CAP_ANDROID+98, // Android front camera

    CV_CAP_XIAPI    = 1100,   // XIMEA Camera API

    CV_CAP_AVFOUNDATION = 1200,  // AVFoundation framework for iOS (OS X Lion will have the same API)

    CV_CAP_GIGANETIX = 1300,  // Smartek Giganetix GigEVisionSDK

    CV_CAP_INTELPERC = 1500; // Intel Perceptual Computing SDK

/* start capturing frames from camera: index = camera_index + domain_offset (CV_CAP_*) */
public static native CvCapture cvCreateCameraCapture( int index );

/* grab a frame, return 1 on success, 0 on fail.
  this function is thought to be fast               */
public static native int cvGrabFrame( CvCapture capture );

/* get the frame grabbed with cvGrabFrame(..)
  This function may apply some frame processing like
  frame decompression, flipping etc.
  !!!DO NOT RELEASE or MODIFY the retrieved frame!!! */
public static native IplImage cvRetrieveFrame( CvCapture capture, int streamIdx/*=0*/ );
public static native IplImage cvRetrieveFrame( CvCapture capture );

/* Just a combination of cvGrabFrame and cvRetrieveFrame
   !!!DO NOT RELEASE or MODIFY the retrieved frame!!!      */
public static native IplImage cvQueryFrame( CvCapture capture );

/* stop capturing/reading and free resources */
public static native void cvReleaseCapture( @Cast("CvCapture**") PointerPointer capture );
public static native void cvReleaseCapture( @ByPtrPtr CvCapture capture );

/** enum  */
public static final int
    // modes of the controlling registers (can be: auto, manual, auto single push, absolute Latter allowed with any other mode)
    // every feature can have only one mode turned on at a time
    CV_CAP_PROP_DC1394_OFF         = -4,  //turn the feature off (not controlled manually nor automatically)
    CV_CAP_PROP_DC1394_MODE_MANUAL = -3, //set automatically when a value of the feature is set by the user
    CV_CAP_PROP_DC1394_MODE_AUTO = -2,
    CV_CAP_PROP_DC1394_MODE_ONE_PUSH_AUTO = -1,
    CV_CAP_PROP_POS_MSEC       = 0,
    CV_CAP_PROP_POS_FRAMES     = 1,
    CV_CAP_PROP_POS_AVI_RATIO  = 2,
    CV_CAP_PROP_FRAME_WIDTH    = 3,
    CV_CAP_PROP_FRAME_HEIGHT   = 4,
    CV_CAP_PROP_FPS            = 5,
    CV_CAP_PROP_FOURCC         = 6,
    CV_CAP_PROP_FRAME_COUNT    = 7,
    CV_CAP_PROP_FORMAT         = 8,
    CV_CAP_PROP_MODE           = 9,
    CV_CAP_PROP_BRIGHTNESS    = 10,
    CV_CAP_PROP_CONTRAST      = 11,
    CV_CAP_PROP_SATURATION    = 12,
    CV_CAP_PROP_HUE           = 13,
    CV_CAP_PROP_GAIN          = 14,
    CV_CAP_PROP_EXPOSURE      = 15,
    CV_CAP_PROP_CONVERT_RGB   = 16,
    CV_CAP_PROP_WHITE_BALANCE_BLUE_U = 17,
    CV_CAP_PROP_RECTIFICATION = 18,
    CV_CAP_PROP_MONOCROME     = 19,
    CV_CAP_PROP_SHARPNESS     = 20,
    CV_CAP_PROP_AUTO_EXPOSURE = 21, // exposure control done by camera,
                                   // user can adjust refernce level
                                   // using this feature
    CV_CAP_PROP_GAMMA         = 22,
    CV_CAP_PROP_TEMPERATURE   = 23,
    CV_CAP_PROP_TRIGGER       = 24,
    CV_CAP_PROP_TRIGGER_DELAY = 25,
    CV_CAP_PROP_WHITE_BALANCE_RED_V = 26,
    CV_CAP_PROP_ZOOM          = 27,
    CV_CAP_PROP_FOCUS         = 28,
    CV_CAP_PROP_GUID          = 29,
    CV_CAP_PROP_ISO_SPEED     = 30,
    CV_CAP_PROP_MAX_DC1394    = 31,
    CV_CAP_PROP_BACKLIGHT     = 32,
    CV_CAP_PROP_PAN           = 33,
    CV_CAP_PROP_TILT          = 34,
    CV_CAP_PROP_ROLL          = 35,
    CV_CAP_PROP_IRIS          = 36,
    CV_CAP_PROP_SETTINGS      = 37,

    CV_CAP_PROP_AUTOGRAB      = 1024, // property for highgui class CvCapture_Android only
    CV_CAP_PROP_SUPPORTED_PREVIEW_SIZES_STRING= 1025, // readonly, tricky property, returns cpnst char* indeed
    CV_CAP_PROP_PREVIEW_FORMAT= 1026, // readonly, tricky property, returns cpnst char* indeed

    // OpenNI map generators
    CV_CAP_OPENNI_DEPTH_GENERATOR =  1 << 31,
    CV_CAP_OPENNI_IMAGE_GENERATOR =  1 << 30,
    CV_CAP_OPENNI_GENERATORS_MASK =  CV_CAP_OPENNI_DEPTH_GENERATOR + CV_CAP_OPENNI_IMAGE_GENERATOR,

    // Properties of cameras available through OpenNI interfaces
    CV_CAP_PROP_OPENNI_OUTPUT_MODE     = 100,
    CV_CAP_PROP_OPENNI_FRAME_MAX_DEPTH = 101, // in mm
    CV_CAP_PROP_OPENNI_BASELINE        = 102, // in mm
    CV_CAP_PROP_OPENNI_FOCAL_LENGTH    = 103, // in pixels
    CV_CAP_PROP_OPENNI_REGISTRATION    = 104, // flag
    CV_CAP_PROP_OPENNI_REGISTRATION_ON =  CV_CAP_PROP_OPENNI_REGISTRATION, // flag that synchronizes the remapping depth map to image map
                                                                          // by changing depth generator's view point (if the flag is "on") or
                                                                          // sets this view point to its normal one (if the flag is "off").
    CV_CAP_PROP_OPENNI_APPROX_FRAME_SYNC = 105,
    CV_CAP_PROP_OPENNI_MAX_BUFFER_SIZE   = 106,
    CV_CAP_PROP_OPENNI_CIRCLE_BUFFER     = 107,
    CV_CAP_PROP_OPENNI_MAX_TIME_DURATION = 108,

    CV_CAP_PROP_OPENNI_GENERATOR_PRESENT = 109,

    CV_CAP_OPENNI_IMAGE_GENERATOR_PRESENT         =  CV_CAP_OPENNI_IMAGE_GENERATOR + CV_CAP_PROP_OPENNI_GENERATOR_PRESENT,
    CV_CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE     =  CV_CAP_OPENNI_IMAGE_GENERATOR + CV_CAP_PROP_OPENNI_OUTPUT_MODE,
    CV_CAP_OPENNI_DEPTH_GENERATOR_BASELINE        =  CV_CAP_OPENNI_DEPTH_GENERATOR + CV_CAP_PROP_OPENNI_BASELINE,
    CV_CAP_OPENNI_DEPTH_GENERATOR_FOCAL_LENGTH    =  CV_CAP_OPENNI_DEPTH_GENERATOR + CV_CAP_PROP_OPENNI_FOCAL_LENGTH,
    CV_CAP_OPENNI_DEPTH_GENERATOR_REGISTRATION    =  CV_CAP_OPENNI_DEPTH_GENERATOR + CV_CAP_PROP_OPENNI_REGISTRATION,
    CV_CAP_OPENNI_DEPTH_GENERATOR_REGISTRATION_ON =  CV_CAP_OPENNI_DEPTH_GENERATOR_REGISTRATION,

    // Properties of cameras available through GStreamer interface
    CV_CAP_GSTREAMER_QUEUE_LENGTH   = 200, // default is 1
    CV_CAP_PROP_PVAPI_MULTICASTIP   = 300, // ip for anable multicast master mode. 0 for disable multicast

    // Properties of cameras available through XIMEA SDK interface
    CV_CAP_PROP_XI_DOWNSAMPLING  = 400,      // Change image resolution by binning or skipping.
    CV_CAP_PROP_XI_DATA_FORMAT   = 401,       // Output data format.
    CV_CAP_PROP_XI_OFFSET_X      = 402,      // Horizontal offset from the origin to the area of interest (in pixels).
    CV_CAP_PROP_XI_OFFSET_Y      = 403,      // Vertical offset from the origin to the area of interest (in pixels).
    CV_CAP_PROP_XI_TRG_SOURCE    = 404,      // Defines source of trigger.
    CV_CAP_PROP_XI_TRG_SOFTWARE  = 405,      // Generates an internal trigger. PRM_TRG_SOURCE must be set to TRG_SOFTWARE.
    CV_CAP_PROP_XI_GPI_SELECTOR  = 406,      // Selects general purpose input
    CV_CAP_PROP_XI_GPI_MODE      = 407,      // Set general purpose input mode
    CV_CAP_PROP_XI_GPI_LEVEL     = 408,      // Get general purpose level
    CV_CAP_PROP_XI_GPO_SELECTOR  = 409,      // Selects general purpose output
    CV_CAP_PROP_XI_GPO_MODE      = 410,      // Set general purpose output mode
    CV_CAP_PROP_XI_LED_SELECTOR  = 411,      // Selects camera signalling LED
    CV_CAP_PROP_XI_LED_MODE      = 412,      // Define camera signalling LED functionality
    CV_CAP_PROP_XI_MANUAL_WB     = 413,      // Calculates White Balance(must be called during acquisition)
    CV_CAP_PROP_XI_AUTO_WB       = 414,      // Automatic white balance
    CV_CAP_PROP_XI_AEAG          = 415,      // Automatic exposure/gain
    CV_CAP_PROP_XI_EXP_PRIORITY  = 416,      // Exposure priority (0.5 - exposure 50%, gain 50%).
    CV_CAP_PROP_XI_AE_MAX_LIMIT  = 417,      // Maximum limit of exposure in AEAG procedure
    CV_CAP_PROP_XI_AG_MAX_LIMIT  = 418,      // Maximum limit of gain in AEAG procedure
    CV_CAP_PROP_XI_AEAG_LEVEL    = 419,       // Average intensity of output signal AEAG should achieve(in %)
    CV_CAP_PROP_XI_TIMEOUT       = 420,       // Image capture timeout in milliseconds

    // Properties for Android cameras
    CV_CAP_PROP_ANDROID_FLASH_MODE = 8001,
    CV_CAP_PROP_ANDROID_FOCUS_MODE = 8002,
    CV_CAP_PROP_ANDROID_WHITE_BALANCE = 8003,
    CV_CAP_PROP_ANDROID_ANTIBANDING = 8004,
    CV_CAP_PROP_ANDROID_FOCAL_LENGTH = 8005,
    CV_CAP_PROP_ANDROID_FOCUS_DISTANCE_NEAR = 8006,
    CV_CAP_PROP_ANDROID_FOCUS_DISTANCE_OPTIMAL = 8007,
    CV_CAP_PROP_ANDROID_FOCUS_DISTANCE_FAR = 8008,
    CV_CAP_PROP_ANDROID_EXPOSE_LOCK = 8009,
    CV_CAP_PROP_ANDROID_WHITEBALANCE_LOCK = 8010,

    // Properties of cameras available through AVFOUNDATION interface
    CV_CAP_PROP_IOS_DEVICE_FOCUS = 9001,
    CV_CAP_PROP_IOS_DEVICE_EXPOSURE = 9002,
    CV_CAP_PROP_IOS_DEVICE_FLASH = 9003,
    CV_CAP_PROP_IOS_DEVICE_WHITEBALANCE = 9004,
    CV_CAP_PROP_IOS_DEVICE_TORCH = 9005,

    // Properties of cameras available through Smartek Giganetix Ethernet Vision interface
    /* --- Vladimir Litvinenko (litvinenko.vladimir@gmail.com) --- */
    CV_CAP_PROP_GIGA_FRAME_OFFSET_X = 10001,
    CV_CAP_PROP_GIGA_FRAME_OFFSET_Y = 10002,
    CV_CAP_PROP_GIGA_FRAME_WIDTH_MAX = 10003,
    CV_CAP_PROP_GIGA_FRAME_HEIGH_MAX = 10004,
    CV_CAP_PROP_GIGA_FRAME_SENS_WIDTH = 10005,
    CV_CAP_PROP_GIGA_FRAME_SENS_HEIGH = 10006,

    CV_CAP_PROP_INTELPERC_PROFILE_COUNT               = 11001,
    CV_CAP_PROP_INTELPERC_PROFILE_IDX                 = 11002,
    CV_CAP_PROP_INTELPERC_DEPTH_LOW_CONFIDENCE_VALUE  = 11003,
    CV_CAP_PROP_INTELPERC_DEPTH_SATURATION_VALUE      = 11004,
    CV_CAP_PROP_INTELPERC_DEPTH_CONFIDENCE_THRESHOLD  = 11005,
    CV_CAP_PROP_INTELPERC_DEPTH_FOCAL_LENGTH_HORZ     = 11006,
    CV_CAP_PROP_INTELPERC_DEPTH_FOCAL_LENGTH_VERT     = 11007,

    // Intel PerC streams
    CV_CAP_INTELPERC_DEPTH_GENERATOR =  1 << 29,
    CV_CAP_INTELPERC_IMAGE_GENERATOR =  1 << 28,
    CV_CAP_INTELPERC_GENERATORS_MASK =  CV_CAP_INTELPERC_DEPTH_GENERATOR + CV_CAP_INTELPERC_IMAGE_GENERATOR;

/** enum  */
public static final int
    // Data given from depth generator.
    CV_CAP_OPENNI_DEPTH_MAP                 = 0, // Depth values in mm (CV_16UC1)
    CV_CAP_OPENNI_POINT_CLOUD_MAP           = 1, // XYZ in meters (CV_32FC3)
    CV_CAP_OPENNI_DISPARITY_MAP             = 2, // Disparity in pixels (CV_8UC1)
    CV_CAP_OPENNI_DISPARITY_MAP_32F         = 3, // Disparity in pixels (CV_32FC1)
    CV_CAP_OPENNI_VALID_DEPTH_MASK          = 4, // CV_8UC1

    // Data given from RGB image generator.
    CV_CAP_OPENNI_BGR_IMAGE                 = 5,
    CV_CAP_OPENNI_GRAY_IMAGE                = 6;

// Supported output modes of OpenNI image generator
/** enum  */
public static final int
    CV_CAP_OPENNI_VGA_30HZ     = 0,
    CV_CAP_OPENNI_SXGA_15HZ    = 1,
    CV_CAP_OPENNI_SXGA_30HZ    = 2,
    CV_CAP_OPENNI_QVGA_30HZ    = 3,
    CV_CAP_OPENNI_QVGA_60HZ    = 4;

//supported by Android camera output formats
/** enum  */
public static final int
    CV_CAP_ANDROID_COLOR_FRAME_BGR = 0, //BGR
    CV_CAP_ANDROID_COLOR_FRAME =  CV_CAP_ANDROID_COLOR_FRAME_BGR,
    CV_CAP_ANDROID_GREY_FRAME  = 1,  //Y
    CV_CAP_ANDROID_COLOR_FRAME_RGB = 2,
    CV_CAP_ANDROID_COLOR_FRAME_BGRA = 3,
    CV_CAP_ANDROID_COLOR_FRAME_RGBA = 4;

// supported Android camera flash modes
/** enum  */
public static final int
    CV_CAP_ANDROID_FLASH_MODE_AUTO = 0,
    CV_CAP_ANDROID_FLASH_MODE_OFF = 1,
    CV_CAP_ANDROID_FLASH_MODE_ON = 2,
    CV_CAP_ANDROID_FLASH_MODE_RED_EYE = 3,
    CV_CAP_ANDROID_FLASH_MODE_TORCH = 4;

// supported Android camera focus modes
/** enum  */
public static final int
    CV_CAP_ANDROID_FOCUS_MODE_AUTO = 0,
    CV_CAP_ANDROID_FOCUS_MODE_CONTINUOUS_PICTURE = 1,
    CV_CAP_ANDROID_FOCUS_MODE_CONTINUOUS_VIDEO = 2,
    CV_CAP_ANDROID_FOCUS_MODE_EDOF = 3,
    CV_CAP_ANDROID_FOCUS_MODE_FIXED = 4,
    CV_CAP_ANDROID_FOCUS_MODE_INFINITY = 5,
    CV_CAP_ANDROID_FOCUS_MODE_MACRO = 6;

// supported Android camera white balance modes
/** enum  */
public static final int
    CV_CAP_ANDROID_WHITE_BALANCE_AUTO = 0,
    CV_CAP_ANDROID_WHITE_BALANCE_CLOUDY_DAYLIGHT = 1,
    CV_CAP_ANDROID_WHITE_BALANCE_DAYLIGHT = 2,
    CV_CAP_ANDROID_WHITE_BALANCE_FLUORESCENT = 3,
    CV_CAP_ANDROID_WHITE_BALANCE_INCANDESCENT = 4,
    CV_CAP_ANDROID_WHITE_BALANCE_SHADE = 5,
    CV_CAP_ANDROID_WHITE_BALANCE_TWILIGHT = 6,
    CV_CAP_ANDROID_WHITE_BALANCE_WARM_FLUORESCENT = 7;

// supported Android camera antibanding modes
/** enum  */
public static final int
    CV_CAP_ANDROID_ANTIBANDING_50HZ = 0,
    CV_CAP_ANDROID_ANTIBANDING_60HZ = 1,
    CV_CAP_ANDROID_ANTIBANDING_AUTO = 2,
    CV_CAP_ANDROID_ANTIBANDING_OFF = 3;

/** enum  */
public static final int
    CV_CAP_INTELPERC_DEPTH_MAP              = 0, // Each pixel is a 16-bit integer. The value indicates the distance from an object to the camera's XY plane or the Cartesian depth.
    CV_CAP_INTELPERC_UVDEPTH_MAP            = 1, // Each pixel contains two 32-bit floating point values in the range of 0-1, representing the mapping of depth coordinates to the color coordinates.
    CV_CAP_INTELPERC_IR_MAP                 = 2, // Each pixel is a 16-bit integer. The value indicates the intensity of the reflected laser beam.
    CV_CAP_INTELPERC_IMAGE                  = 3;

/* retrieve or set capture properties */
public static native double cvGetCaptureProperty( CvCapture capture, int property_id );
public static native int cvSetCaptureProperty( CvCapture capture, int property_id, double value );

// Return the type of the capturer (eg, CV_CAP_V4W, CV_CAP_UNICAP), which is unknown if created with CV_CAP_ANY
public static native int cvGetCaptureDomain( CvCapture capture);

/* "black box" video file writer structure */
@Opaque public static class CvVideoWriter extends Pointer {
    public CvVideoWriter() { }
    public CvVideoWriter(Pointer p) { super(p); }
}

// #define CV_FOURCC_MACRO(c1, c2, c3, c4) (((c1) & 255) + (((c2) & 255) << 8) + (((c3) & 255) << 16) + (((c4) & 255) << 24))

public static native int CV_FOURCC(@Cast("char") byte c1, @Cast("char") byte c2, @Cast("char") byte c3, @Cast("char") byte c4);

public static final int CV_FOURCC_PROMPT = -1;  /* Open Codec Selection Dialog (Windows only) */
public static native @MemberGetter int CV_FOURCC_DEFAULT();
public static final int CV_FOURCC_DEFAULT = CV_FOURCC_DEFAULT(); /* Use default codec for specified filename (Linux only) */

/* initialize video file writer */
public static native CvVideoWriter cvCreateVideoWriter( @Cast("const char*") BytePointer filename, int fourcc,
                                           double fps, @ByVal CvSize frame_size,
                                           int is_color/*=1*/);
public static native CvVideoWriter cvCreateVideoWriter( @Cast("const char*") BytePointer filename, int fourcc,
                                           double fps, @ByVal CvSize frame_size);
public static native CvVideoWriter cvCreateVideoWriter( String filename, int fourcc,
                                           double fps, @ByVal CvSize frame_size,
                                           int is_color/*=1*/);
public static native CvVideoWriter cvCreateVideoWriter( String filename, int fourcc,
                                           double fps, @ByVal CvSize frame_size);

//CVAPI(CvVideoWriter*) cvCreateImageSequenceWriter( const char* filename,
//                                                   int is_color CV_DEFAULT(1));

/* write frame to video file */
public static native int cvWriteFrame( CvVideoWriter writer, @Const IplImage image );

/* close video file writer */
public static native void cvReleaseVideoWriter( @Cast("CvVideoWriter**") PointerPointer writer );
public static native void cvReleaseVideoWriter( @ByPtrPtr CvVideoWriter writer );

/****************************************************************************************\
*                              Obsolete functions/synonyms                               *
\****************************************************************************************/

public static native CvCapture cvCaptureFromFile(@Cast("const char*") BytePointer arg1);
public static native CvCapture cvCaptureFromFile(String arg1);
public static native CvCapture cvCaptureFromCAM(int arg1);
public static native CvCapture cvCaptureFromAVI(@Cast("const char*") BytePointer arg1);
public static native CvCapture cvCaptureFromAVI(String arg1);
public static native CvVideoWriter cvCreateAVIWriter(@Cast("const char*") BytePointer arg1, int arg2, double arg3, @ByVal CvSize arg4, int arg5);
public static native CvVideoWriter cvCreateAVIWriter(String arg1, int arg2, double arg3, @ByVal CvSize arg4, int arg5);
public static native int cvWriteToAVI(CvVideoWriter arg1, IplImage arg2);
public static native void cvAddSearchPath(@Cast("const char*") BytePointer path);
public static native void cvAddSearchPath(String path);
public static native int cvvInitSystem(int arg1, @Cast("char**") PointerPointer arg2);
public static native int cvvInitSystem(int arg1, @Cast("char**") @ByPtrPtr BytePointer arg2);
public static native int cvvInitSystem(int arg1, @Cast("char**") @ByPtrPtr ByteBuffer arg2);
public static native int cvvInitSystem(int arg1, @Cast("char**") @ByPtrPtr byte[] arg2);
public static native void cvvNamedWindow(@Cast("const char*") BytePointer arg1, int arg2);
public static native void cvvNamedWindow(String arg1, int arg2);
public static native void cvvShowImage(@Cast("const char*") BytePointer arg1, CvArr arg2);
public static native void cvvShowImage(String arg1, CvArr arg2);
public static native void cvvResizeWindow(@Cast("const char*") BytePointer arg1, int arg2, int arg3);
public static native void cvvResizeWindow(String arg1, int arg2, int arg3);
public static native void cvvDestroyWindow(@Cast("const char*") BytePointer arg1);
public static native void cvvDestroyWindow(String arg1);
public static native int cvvCreateTrackbar(@Cast("const char*") BytePointer arg1, @Cast("const char*") BytePointer arg2, IntPointer arg3, int arg4, CvTrackbarCallback arg5);
public static native int cvvCreateTrackbar(String arg1, String arg2, IntBuffer arg3, int arg4, CvTrackbarCallback arg5);
public static native int cvvCreateTrackbar(@Cast("const char*") BytePointer arg1, @Cast("const char*") BytePointer arg2, int[] arg3, int arg4, CvTrackbarCallback arg5);
public static native int cvvCreateTrackbar(String arg1, String arg2, IntPointer arg3, int arg4, CvTrackbarCallback arg5);
public static native int cvvCreateTrackbar(@Cast("const char*") BytePointer arg1, @Cast("const char*") BytePointer arg2, IntBuffer arg3, int arg4, CvTrackbarCallback arg5);
public static native int cvvCreateTrackbar(String arg1, String arg2, int[] arg3, int arg4, CvTrackbarCallback arg5);
public static native IplImage cvvLoadImage(@Cast("const char*") BytePointer name);
public static native IplImage cvvLoadImage(String name);
public static native int cvvSaveImage(@Cast("const char*") BytePointer arg1, CvArr arg2, IntPointer arg3);
public static native int cvvSaveImage(String arg1, CvArr arg2, IntBuffer arg3);
public static native int cvvSaveImage(@Cast("const char*") BytePointer arg1, CvArr arg2, int[] arg3);
public static native int cvvSaveImage(String arg1, CvArr arg2, IntPointer arg3);
public static native int cvvSaveImage(@Cast("const char*") BytePointer arg1, CvArr arg2, IntBuffer arg3);
public static native int cvvSaveImage(String arg1, CvArr arg2, int[] arg3);
public static native void cvvAddSearchPath(@Cast("const char*") BytePointer arg1);
public static native void cvvAddSearchPath(String arg1);
public static native int cvvWaitKey(@Cast("const char*") BytePointer name);
public static native int cvvWaitKey(String name);
public static native int cvvWaitKeyEx(@Cast("const char*") BytePointer name, int delay);
public static native int cvvWaitKeyEx(String name, int delay);
public static native void cvvConvertImage(CvArr arg1, CvArr arg2, int arg3);
public static final int HG_AUTOSIZE = CV_WINDOW_AUTOSIZE;
// #define set_preprocess_func cvSetPreprocessFuncWin32
// #define set_postprocess_func cvSetPostprocessFuncWin32

// #if defined WIN32 || defined _WIN32

// #endif

// #ifdef __cplusplus
// #endif

// #endif


// Parsed from <opencv2/highgui/highgui.hpp>

/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

// #ifndef __OPENCV_HIGHGUI_HPP__
// #define __OPENCV_HIGHGUI_HPP__

// #include "opencv2/core/core.hpp"
// #include "opencv2/highgui/highgui_c.h"

// #ifdef __cplusplus

/** enum cv:: */
public static final int
    // Flags for namedWindow
    WINDOW_NORMAL   =  CV_WINDOW_NORMAL,   // the user can resize the window (no constraint) / also use to switch a fullscreen window to a normal size
    WINDOW_AUTOSIZE =  CV_WINDOW_AUTOSIZE, // the user cannot resize the window, the size is constrainted by the image displayed
    WINDOW_OPENGL   =  CV_WINDOW_OPENGL,   // window with opengl support

    // Flags for set / getWindowProperty
    WND_PROP_FULLSCREEN   =  CV_WND_PROP_FULLSCREEN,  // fullscreen property
    WND_PROP_AUTOSIZE     =  CV_WND_PROP_AUTOSIZE,    // autosize property
    WND_PROP_ASPECT_RATIO =  CV_WND_PROP_ASPECTRATIO, // window's aspect ration
    WND_PROP_OPENGL       =  CV_WND_PROP_OPENGL;       // opengl support

@Namespace("cv") public static native void namedWindow(@StdString BytePointer winname, int flags/*=WINDOW_AUTOSIZE*/);
@Namespace("cv") public static native void namedWindow(@StdString BytePointer winname);
@Namespace("cv") public static native void namedWindow(@StdString String winname, int flags/*=WINDOW_AUTOSIZE*/);
@Namespace("cv") public static native void namedWindow(@StdString String winname);
@Namespace("cv") public static native void destroyWindow(@StdString BytePointer winname);
@Namespace("cv") public static native void destroyWindow(@StdString String winname);
@Namespace("cv") public static native void destroyAllWindows();

@Namespace("cv") public static native int startWindowThread();

@Namespace("cv") public static native int waitKey(int delay/*=0*/);
@Namespace("cv") public static native int waitKey();

@Namespace("cv") public static native void imshow(@StdString BytePointer winname, @ByVal Mat mat);
@Namespace("cv") public static native void imshow(@StdString String winname, @ByVal Mat mat);

@Namespace("cv") public static native void resizeWindow(@StdString BytePointer winname, int width, int height);
@Namespace("cv") public static native void resizeWindow(@StdString String winname, int width, int height);
@Namespace("cv") public static native void moveWindow(@StdString BytePointer winname, int x, int y);
@Namespace("cv") public static native void moveWindow(@StdString String winname, int x, int y);

@Namespace("cv") public static native void setWindowProperty(@StdString BytePointer winname, int prop_id, double prop_value);
@Namespace("cv") public static native void setWindowProperty(@StdString String winname, int prop_id, double prop_value);//YV
@Namespace("cv") public static native double getWindowProperty(@StdString BytePointer winname, int prop_id);
@Namespace("cv") public static native double getWindowProperty(@StdString String winname, int prop_id);//YV

/** enum cv:: */
public static final int
    EVENT_MOUSEMOVE      = 0,
    EVENT_LBUTTONDOWN    = 1,
    EVENT_RBUTTONDOWN    = 2,
    EVENT_MBUTTONDOWN    = 3,
    EVENT_LBUTTONUP      = 4,
    EVENT_RBUTTONUP      = 5,
    EVENT_MBUTTONUP      = 6,
    EVENT_LBUTTONDBLCLK  = 7,
    EVENT_RBUTTONDBLCLK  = 8,
    EVENT_MBUTTONDBLCLK  = 9;

/** enum cv:: */
public static final int
    EVENT_FLAG_LBUTTON   = 1,
    EVENT_FLAG_RBUTTON   = 2,
    EVENT_FLAG_MBUTTON   = 4,
    EVENT_FLAG_CTRLKEY   = 8,
    EVENT_FLAG_SHIFTKEY  = 16,
    EVENT_FLAG_ALTKEY    = 32;

public static class MouseCallback extends FunctionPointer {
    static { Loader.load(); }
    public    MouseCallback(Pointer p) { super(p); }
    protected MouseCallback() { allocate(); }
    private native void allocate();
    public native void call(int event, int x, int y, int flags, Pointer userdata);
}

/** assigns callback for mouse events */
@Namespace("cv") public static native void setMouseCallback(@StdString BytePointer winname, MouseCallback onMouse, Pointer userdata/*=0*/);
@Namespace("cv") public static native void setMouseCallback(@StdString BytePointer winname, MouseCallback onMouse);
@Namespace("cv") public static native void setMouseCallback(@StdString String winname, MouseCallback onMouse, Pointer userdata/*=0*/);
@Namespace("cv") public static native void setMouseCallback(@StdString String winname, MouseCallback onMouse);


@Convention("CV_CDECL") public static class TrackbarCallback extends FunctionPointer {
    static { Loader.load(); }
    public    TrackbarCallback(Pointer p) { super(p); }
    protected TrackbarCallback() { allocate(); }
    private native void allocate();
    public native void call(int pos, Pointer userdata);
}

@Namespace("cv") public static native int createTrackbar(@StdString BytePointer trackbarname, @StdString BytePointer winname,
                              IntPointer value, int count,
                              TrackbarCallback onChange/*=0*/,
                              Pointer userdata/*=0*/);
@Namespace("cv") public static native int createTrackbar(@StdString BytePointer trackbarname, @StdString BytePointer winname,
                              IntPointer value, int count);
@Namespace("cv") public static native int createTrackbar(@StdString String trackbarname, @StdString String winname,
                              IntBuffer value, int count,
                              TrackbarCallback onChange/*=0*/,
                              Pointer userdata/*=0*/);
@Namespace("cv") public static native int createTrackbar(@StdString String trackbarname, @StdString String winname,
                              IntBuffer value, int count);
@Namespace("cv") public static native int createTrackbar(@StdString BytePointer trackbarname, @StdString BytePointer winname,
                              int[] value, int count,
                              TrackbarCallback onChange/*=0*/,
                              Pointer userdata/*=0*/);
@Namespace("cv") public static native int createTrackbar(@StdString BytePointer trackbarname, @StdString BytePointer winname,
                              int[] value, int count);
@Namespace("cv") public static native int createTrackbar(@StdString String trackbarname, @StdString String winname,
                              IntPointer value, int count,
                              TrackbarCallback onChange/*=0*/,
                              Pointer userdata/*=0*/);
@Namespace("cv") public static native int createTrackbar(@StdString String trackbarname, @StdString String winname,
                              IntPointer value, int count);
@Namespace("cv") public static native int createTrackbar(@StdString BytePointer trackbarname, @StdString BytePointer winname,
                              IntBuffer value, int count,
                              TrackbarCallback onChange/*=0*/,
                              Pointer userdata/*=0*/);
@Namespace("cv") public static native int createTrackbar(@StdString BytePointer trackbarname, @StdString BytePointer winname,
                              IntBuffer value, int count);
@Namespace("cv") public static native int createTrackbar(@StdString String trackbarname, @StdString String winname,
                              int[] value, int count,
                              TrackbarCallback onChange/*=0*/,
                              Pointer userdata/*=0*/);
@Namespace("cv") public static native int createTrackbar(@StdString String trackbarname, @StdString String winname,
                              int[] value, int count);

@Namespace("cv") public static native int getTrackbarPos(@StdString BytePointer trackbarname, @StdString BytePointer winname);
@Namespace("cv") public static native int getTrackbarPos(@StdString String trackbarname, @StdString String winname);
@Namespace("cv") public static native void setTrackbarPos(@StdString BytePointer trackbarname, @StdString BytePointer winname, int pos);
@Namespace("cv") public static native void setTrackbarPos(@StdString String trackbarname, @StdString String winname, int pos);

// OpenGL support

public static class OpenGlDrawCallback extends FunctionPointer {
    static { Loader.load(); }
    public    OpenGlDrawCallback(Pointer p) { super(p); }
    protected OpenGlDrawCallback() { allocate(); }
    private native void allocate();
    public native void call(Pointer userdata);
}
@Namespace("cv") public static native void setOpenGlDrawCallback(@StdString BytePointer winname, OpenGlDrawCallback onOpenGlDraw, Pointer userdata/*=0*/);
@Namespace("cv") public static native void setOpenGlDrawCallback(@StdString BytePointer winname, OpenGlDrawCallback onOpenGlDraw);
@Namespace("cv") public static native void setOpenGlDrawCallback(@StdString String winname, OpenGlDrawCallback onOpenGlDraw, Pointer userdata/*=0*/);
@Namespace("cv") public static native void setOpenGlDrawCallback(@StdString String winname, OpenGlDrawCallback onOpenGlDraw);

@Namespace("cv") public static native void setOpenGlContext(@StdString BytePointer winname);
@Namespace("cv") public static native void setOpenGlContext(@StdString String winname);

@Namespace("cv") public static native void updateWindow(@StdString BytePointer winname);
@Namespace("cv") public static native void updateWindow(@StdString String winname);

// < Deperecated
@Namespace("cv") public static native void pointCloudShow(@StdString BytePointer winname, @Const @ByRef GlCamera camera, @Const @ByRef GlArrays arr);
@Namespace("cv") public static native void pointCloudShow(@StdString String winname, @Const @ByRef GlCamera camera, @Const @ByRef GlArrays arr);
@Namespace("cv") public static native void pointCloudShow(@StdString BytePointer winname, @Const @ByRef GlCamera camera, @ByVal Mat points, @ByVal Mat colors/*=noArray()*/);
@Namespace("cv") public static native void pointCloudShow(@StdString BytePointer winname, @Const @ByRef GlCamera camera, @ByVal Mat points);
@Namespace("cv") public static native void pointCloudShow(@StdString String winname, @Const @ByRef GlCamera camera, @ByVal Mat points, @ByVal Mat colors/*=noArray()*/);
@Namespace("cv") public static native void pointCloudShow(@StdString String winname, @Const @ByRef GlCamera camera, @ByVal Mat points);
// >

//Only for Qt

@Namespace("cv") public static native @ByVal CvFont fontQt(@StdString BytePointer nameFont, int pointSize/*=-1*/,
                         @ByVal Scalar color/*=Scalar::all(0)*/, int weight/*=CV_FONT_NORMAL*/,
                         int style/*=CV_STYLE_NORMAL*/, int spacing/*=0*/);
@Namespace("cv") public static native @ByVal CvFont fontQt(@StdString BytePointer nameFont);
@Namespace("cv") public static native @ByVal CvFont fontQt(@StdString String nameFont, int pointSize/*=-1*/,
                         @ByVal Scalar color/*=Scalar::all(0)*/, int weight/*=CV_FONT_NORMAL*/,
                         int style/*=CV_STYLE_NORMAL*/, int spacing/*=0*/);
@Namespace("cv") public static native @ByVal CvFont fontQt(@StdString String nameFont);
@Namespace("cv") public static native void addText( @Const @ByRef Mat img, @StdString BytePointer text, @ByVal Point org, @ByVal CvFont font);
@Namespace("cv") public static native void addText( @Const @ByRef Mat img, @StdString String text, @ByVal Point org, @ByVal CvFont font);

@Namespace("cv") public static native void displayOverlay(@StdString BytePointer winname, @StdString BytePointer text, int delayms/*=0*/);
@Namespace("cv") public static native void displayOverlay(@StdString BytePointer winname, @StdString BytePointer text);
@Namespace("cv") public static native void displayOverlay(@StdString String winname, @StdString String text, int delayms/*=0*/);
@Namespace("cv") public static native void displayOverlay(@StdString String winname, @StdString String text);
@Namespace("cv") public static native void displayStatusBar(@StdString BytePointer winname, @StdString BytePointer text, int delayms/*=0*/);
@Namespace("cv") public static native void displayStatusBar(@StdString BytePointer winname, @StdString BytePointer text);
@Namespace("cv") public static native void displayStatusBar(@StdString String winname, @StdString String text, int delayms/*=0*/);
@Namespace("cv") public static native void displayStatusBar(@StdString String winname, @StdString String text);

@Namespace("cv") public static native void saveWindowParameters(@StdString BytePointer windowName);
@Namespace("cv") public static native void saveWindowParameters(@StdString String windowName);
@Namespace("cv") public static native void loadWindowParameters(@StdString BytePointer windowName);
@Namespace("cv") public static native void loadWindowParameters(@StdString String windowName);
@Namespace("cv") public static native int startLoop(Pt2Func_int_PointerPointer pt2Func, int argc, @Cast("char**") PointerPointer argv);
@Namespace("cv") public static native int startLoop(Pt2Func_int_BytePointer pt2Func, int argc, @Cast("char**") @ByPtrPtr BytePointer argv);
@Namespace("cv") public static native int startLoop(Pt2Func_int_ByteBuffer pt2Func, int argc, @Cast("char**") @ByPtrPtr ByteBuffer argv);
@Namespace("cv") public static native int startLoop(Pt2Func_int_byte__ pt2Func, int argc, @Cast("char**") @ByPtrPtr byte[] argv);
@Namespace("cv") public static native void stopLoop();

@Convention("CV_CDECL") public static class ButtonCallback extends FunctionPointer {
    static { Loader.load(); }
    public    ButtonCallback(Pointer p) { super(p); }
    protected ButtonCallback() { allocate(); }
    private native void allocate();
    public native void call(int state, Pointer userdata);
}
@Namespace("cv") public static native int createButton( @StdString BytePointer bar_name, ButtonCallback on_change,
                             Pointer userdata/*=NULL*/, int type/*=CV_PUSH_BUTTON*/,
                             @Cast("bool") boolean initial_button_state/*=0*/);
@Namespace("cv") public static native int createButton( @StdString BytePointer bar_name, ButtonCallback on_change);
@Namespace("cv") public static native int createButton( @StdString String bar_name, ButtonCallback on_change,
                             Pointer userdata/*=NULL*/, int type/*=CV_PUSH_BUTTON*/,
                             @Cast("bool") boolean initial_button_state/*=0*/);
@Namespace("cv") public static native int createButton( @StdString String bar_name, ButtonCallback on_change);

//-------------------------

/** enum cv:: */
public static final int
    // 8bit, color or not
    IMREAD_UNCHANGED  = -1,
    // 8bit, gray
    IMREAD_GRAYSCALE  = 0,
    // ?, color
    IMREAD_COLOR      = 1,
    // any depth, ?
    IMREAD_ANYDEPTH   = 2,
    // ?, any color
    IMREAD_ANYCOLOR   = 4;

/** enum cv:: */
public static final int
    IMWRITE_JPEG_QUALITY = 1,
    IMWRITE_PNG_COMPRESSION = 16,
    IMWRITE_PNG_STRATEGY = 17,
    IMWRITE_PNG_BILEVEL = 18,
    IMWRITE_PNG_STRATEGY_DEFAULT = 0,
    IMWRITE_PNG_STRATEGY_FILTERED = 1,
    IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY = 2,
    IMWRITE_PNG_STRATEGY_RLE = 3,
    IMWRITE_PNG_STRATEGY_FIXED = 4,
    IMWRITE_PXM_BINARY = 32;

@Namespace("cv") public static native @ByVal Mat imread( @StdString BytePointer filename, int flags/*=1*/ );
@Namespace("cv") public static native @ByVal Mat imread( @StdString BytePointer filename );
@Namespace("cv") public static native @ByVal Mat imread( @StdString String filename, int flags/*=1*/ );
@Namespace("cv") public static native @ByVal Mat imread( @StdString String filename );
@Namespace("cv") public static native @Cast("bool") boolean imwrite( @StdString BytePointer filename, @ByVal Mat img,
              @StdVector IntPointer params/*=vector<int>()*/);
@Namespace("cv") public static native @Cast("bool") boolean imwrite( @StdString BytePointer filename, @ByVal Mat img);
@Namespace("cv") public static native @Cast("bool") boolean imwrite( @StdString String filename, @ByVal Mat img,
              @StdVector IntBuffer params/*=vector<int>()*/);
@Namespace("cv") public static native @Cast("bool") boolean imwrite( @StdString String filename, @ByVal Mat img);
@Namespace("cv") public static native @Cast("bool") boolean imwrite( @StdString BytePointer filename, @ByVal Mat img,
              @StdVector int[] params/*=vector<int>()*/);
@Namespace("cv") public static native @Cast("bool") boolean imwrite( @StdString String filename, @ByVal Mat img,
              @StdVector IntPointer params/*=vector<int>()*/);
@Namespace("cv") public static native @Cast("bool") boolean imwrite( @StdString BytePointer filename, @ByVal Mat img,
              @StdVector IntBuffer params/*=vector<int>()*/);
@Namespace("cv") public static native @Cast("bool") boolean imwrite( @StdString String filename, @ByVal Mat img,
              @StdVector int[] params/*=vector<int>()*/);
@Namespace("cv") public static native @ByVal Mat imdecode( @ByVal Mat buf, int flags );
@Namespace("cv") public static native @ByVal Mat imdecode( @ByVal Mat buf, int flags, Mat dst );
@Namespace("cv") public static native @Cast("bool") boolean imencode( @StdString BytePointer ext, @ByVal Mat img,
                            @Cast("uchar*") @StdVector BytePointer buf,
                            @StdVector IntPointer params/*=vector<int>()*/);
@Namespace("cv") public static native @Cast("bool") boolean imencode( @StdString BytePointer ext, @ByVal Mat img,
                            @Cast("uchar*") @StdVector BytePointer buf);
@Namespace("cv") public static native @Cast("bool") boolean imencode( @StdString String ext, @ByVal Mat img,
                            @Cast("uchar*") @StdVector ByteBuffer buf,
                            @StdVector IntBuffer params/*=vector<int>()*/);
@Namespace("cv") public static native @Cast("bool") boolean imencode( @StdString String ext, @ByVal Mat img,
                            @Cast("uchar*") @StdVector ByteBuffer buf);
@Namespace("cv") public static native @Cast("bool") boolean imencode( @StdString BytePointer ext, @ByVal Mat img,
                            @Cast("uchar*") @StdVector byte[] buf,
                            @StdVector int[] params/*=vector<int>()*/);
@Namespace("cv") public static native @Cast("bool") boolean imencode( @StdString BytePointer ext, @ByVal Mat img,
                            @Cast("uchar*") @StdVector byte[] buf);
@Namespace("cv") public static native @Cast("bool") boolean imencode( @StdString String ext, @ByVal Mat img,
                            @Cast("uchar*") @StdVector BytePointer buf,
                            @StdVector IntPointer params/*=vector<int>()*/);
@Namespace("cv") public static native @Cast("bool") boolean imencode( @StdString String ext, @ByVal Mat img,
                            @Cast("uchar*") @StdVector BytePointer buf);
@Namespace("cv") public static native @Cast("bool") boolean imencode( @StdString BytePointer ext, @ByVal Mat img,
                            @Cast("uchar*") @StdVector ByteBuffer buf,
                            @StdVector IntBuffer params/*=vector<int>()*/);
@Namespace("cv") public static native @Cast("bool") boolean imencode( @StdString BytePointer ext, @ByVal Mat img,
                            @Cast("uchar*") @StdVector ByteBuffer buf);
@Namespace("cv") public static native @Cast("bool") boolean imencode( @StdString String ext, @ByVal Mat img,
                            @Cast("uchar*") @StdVector byte[] buf,
                            @StdVector int[] params/*=vector<int>()*/);
@Namespace("cv") public static native @Cast("bool") boolean imencode( @StdString String ext, @ByVal Mat img,
                            @Cast("uchar*") @StdVector byte[] buf);

// #ifndef CV_NO_VIDEO_CAPTURE_CPP_API




@Namespace("cv") @NoOffset public static class VideoCapture extends Pointer {
    static { Loader.load(); }
    public VideoCapture(Pointer p) { super(p); }

    public VideoCapture() { allocate(); }
    private native void allocate();
    public VideoCapture(@StdString BytePointer filename) { allocate(filename); }
    private native void allocate(@StdString BytePointer filename);
    public VideoCapture(@StdString String filename) { allocate(filename); }
    private native void allocate(@StdString String filename);
    public VideoCapture(int device) { allocate(device); }
    private native void allocate(int device);
    public native @Cast("bool") boolean open(@StdString BytePointer filename);
    public native @Cast("bool") boolean open(@StdString String filename);
    public native @Cast("bool") boolean open(int device);
    public native @Cast("bool") boolean isOpened();
    public native void release();

    public native @Cast("bool") boolean grab();
    public native @Cast("bool") boolean retrieve(@ByRef Mat image, int channel/*=0*/);
    public native @Cast("bool") boolean retrieve(@ByRef Mat image);
    public native @ByRef @Name("operator>>") VideoCapture shiftRight(@ByRef Mat image);
    public native @Cast("bool") boolean read(@ByRef Mat image);

    public native @Cast("bool") boolean set(int propId, double value);
    public native double get(int propId);
}


@Namespace("cv") @NoOffset public static class VideoWriter extends Pointer {
    static { Loader.load(); }
    public VideoWriter(Pointer p) { super(p); }
    public VideoWriter(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public VideoWriter position(int position) {
        return (VideoWriter)super.position(position);
    }

    public VideoWriter() { allocate(); }
    private native void allocate();
    public VideoWriter(@StdString BytePointer filename, int fourcc, double fps,
                    @ByVal Size frameSize, @Cast("bool") boolean isColor/*=true*/) { allocate(filename, fourcc, fps, frameSize, isColor); }
    private native void allocate(@StdString BytePointer filename, int fourcc, double fps,
                    @ByVal Size frameSize, @Cast("bool") boolean isColor/*=true*/);
    public VideoWriter(@StdString BytePointer filename, int fourcc, double fps,
                    @ByVal Size frameSize) { allocate(filename, fourcc, fps, frameSize); }
    private native void allocate(@StdString BytePointer filename, int fourcc, double fps,
                    @ByVal Size frameSize);
    public VideoWriter(@StdString String filename, int fourcc, double fps,
                    @ByVal Size frameSize, @Cast("bool") boolean isColor/*=true*/) { allocate(filename, fourcc, fps, frameSize, isColor); }
    private native void allocate(@StdString String filename, int fourcc, double fps,
                    @ByVal Size frameSize, @Cast("bool") boolean isColor/*=true*/);
    public VideoWriter(@StdString String filename, int fourcc, double fps,
                    @ByVal Size frameSize) { allocate(filename, fourcc, fps, frameSize); }
    private native void allocate(@StdString String filename, int fourcc, double fps,
                    @ByVal Size frameSize);
    public native @Cast("bool") boolean open(@StdString BytePointer filename, int fourcc, double fps,
                          @ByVal Size frameSize, @Cast("bool") boolean isColor/*=true*/);
    public native @Cast("bool") boolean open(@StdString BytePointer filename, int fourcc, double fps,
                          @ByVal Size frameSize);
    public native @Cast("bool") boolean open(@StdString String filename, int fourcc, double fps,
                          @ByVal Size frameSize, @Cast("bool") boolean isColor/*=true*/);
    public native @Cast("bool") boolean open(@StdString String filename, int fourcc, double fps,
                          @ByVal Size frameSize);
    public native @Cast("bool") boolean isOpened();
    public native void release();
    public native @ByRef @Name("operator<<") VideoWriter shiftLeft(@Const @ByRef Mat image);
    public native void write(@Const @ByRef Mat image);
}

// #endif



// #endif

// #endif


}
