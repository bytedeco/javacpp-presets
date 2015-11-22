// Targeted by JavaCPP version 1.2-SNAPSHOT

package org.bytedeco.javacpp;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.*;
import static org.bytedeco.javacpp.opencv_videoio.*;

public class opencv_highgui extends org.bytedeco.javacpp.presets.opencv_highgui {
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
// #include "opencv2/imgproc/imgproc_c.h"
// #include "opencv2/imgcodecs/imgcodecs_c.h"
// #include "opencv2/videoio/videoio_c.h"

// #ifdef __cplusplus
// #endif /* __cplusplus */

/** \addtogroup highgui_c
  \{
  */

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

//for color cvScalar(blue_component, green_component, red_component[, alpha_component])
//and alpha= 0 <-> 0xFF (not transparent <-> transparent)
public static native @ByVal @Platform("linux") CvFont cvFontQt(@Cast("const char*") BytePointer nameFont, int pointSize/*=-1*/, @ByVal(nullValue = "cvScalarAll(0)") CvScalar color/*=cvScalarAll(0)*/, int weight/*=CV_FONT_NORMAL*/,  int style/*=CV_STYLE_NORMAL*/, int spacing/*=0*/);
public static native @ByVal @Platform("linux") CvFont cvFontQt(@Cast("const char*") BytePointer nameFont);
public static native @ByVal @Platform("linux") CvFont cvFontQt(String nameFont, int pointSize/*=-1*/, @ByVal(nullValue = "cvScalarAll(0)") CvScalar color/*=cvScalarAll(0)*/, int weight/*=CV_FONT_NORMAL*/,  int style/*=CV_STYLE_NORMAL*/, int spacing/*=0*/);
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
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public    Pt2Func_int_PointerPointer(Pointer p) { super(p); }
    protected Pt2Func_int_PointerPointer() { allocate(); }
    private native void allocate();
    public native int call(int argc, @Cast("char**") PointerPointer argv);
}
public static native @Platform("linux") int cvStartLoop(Pt2Func_int_PointerPointer pt2Func, int argc, @Cast("char**") PointerPointer argv);
public static class Pt2Func_int_BytePointer extends FunctionPointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public    Pt2Func_int_BytePointer(Pointer p) { super(p); }
    protected Pt2Func_int_BytePointer() { allocate(); }
    private native void allocate();
    public native int call(int argc, @Cast("char**") @ByPtrPtr BytePointer argv);
}
public static native @Platform("linux") int cvStartLoop(Pt2Func_int_BytePointer pt2Func, int argc, @Cast("char**") @ByPtrPtr BytePointer argv);
public static class Pt2Func_int_ByteBuffer extends FunctionPointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public    Pt2Func_int_ByteBuffer(Pointer p) { super(p); }
    protected Pt2Func_int_ByteBuffer() { allocate(); }
    private native void allocate();
    public native int call(int argc, @Cast("char**") @ByPtrPtr ByteBuffer argv);
}
public static native @Platform("linux") int cvStartLoop(Pt2Func_int_ByteBuffer pt2Func, int argc, @Cast("char**") @ByPtrPtr ByteBuffer argv);
public static class Pt2Func_int_byte__ extends FunctionPointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public    Pt2Func_int_byte__(Pointer p) { super(p); }
    protected Pt2Func_int_byte__() { allocate(); }
    private native void allocate();
    public native int call(int argc, @Cast("char**") @ByPtrPtr byte[] argv);
}
public static native @Platform("linux") int cvStartLoop(Pt2Func_int_byte__ pt2Func, int argc, @Cast("char**") @ByPtrPtr byte[] argv);
public static native @Platform("linux") void cvStopLoop( );

@Convention("CV_CDECL") public static class CvButtonCallback extends FunctionPointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
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
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
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
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
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
public static native void cvSetTrackbarMax(@Cast("const char*") BytePointer trackbar_name, @Cast("const char*") BytePointer window_name, int maxval);
public static native void cvSetTrackbarMax(String trackbar_name, String window_name, int maxval);

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
    CV_EVENT_MBUTTONDBLCLK  = 9,
    CV_EVENT_MOUSEWHEEL     = 10,
    CV_EVENT_MOUSEHWHEEL    = 11;

/** enum  */
public static final int
    CV_EVENT_FLAG_LBUTTON   = 1,
    CV_EVENT_FLAG_RBUTTON   = 2,
    CV_EVENT_FLAG_MBUTTON   = 4,
    CV_EVENT_FLAG_CTRLKEY   = 8,
    CV_EVENT_FLAG_SHIFTKEY  = 16,
    CV_EVENT_FLAG_ALTKEY    = 32;


// #define CV_GET_WHEEL_DELTA(flags) ((short)((flags >> 16) & 0xffff)) // upper 16 bits

@Convention("CV_CDECL") public static class CvMouseCallback extends FunctionPointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
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

/* wait for key event infinitely (delay<=0) or for "delay" milliseconds */
public static native int cvWaitKey(int delay/*=0*/);
public static native int cvWaitKey();

// OpenGL support

@Convention("CV_CDECL") public static class CvOpenGlDrawCallback extends FunctionPointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
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
<p>
*                              Obsolete functions/synonyms                               *
\****************************************************************************************/

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
public static native void cvvAddSearchPath(@Cast("const char*") BytePointer arg1);
public static native void cvvAddSearchPath(String arg1);
public static native int cvvWaitKey(@Cast("const char*") BytePointer name);
public static native int cvvWaitKey(String name);
public static native int cvvWaitKeyEx(@Cast("const char*") BytePointer name, int delay);
public static native int cvvWaitKeyEx(String name, int delay);
public static final int HG_AUTOSIZE = CV_WINDOW_AUTOSIZE;
// #define set_preprocess_func cvSetPreprocessFuncWin32
// #define set_postprocess_func cvSetPostprocessFuncWin32

// #if defined WIN32 || defined _WIN32

// #endif

/** \} highgui_c */

// #ifdef __cplusplus
// #endif

// #endif


// Parsed from <opencv2/highgui.hpp>

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

// #include "opencv2/core.hpp"
// #include "opencv2/imgcodecs.hpp"
// #include "opencv2/videoio.hpp"

/**
\defgroup highgui High-level GUI
<p>
While OpenCV was designed for use in full-scale applications and can be used within functionally
rich UI frameworks (such as Qt\*, WinForms\*, or Cocoa\*) or without any UI at all, sometimes there
it is required to try functionality quickly and visualize the results. This is what the HighGUI
module has been designed for.
<p>
It provides easy interface to:
<p>
-   Create and manipulate windows that can display images and "remember" their content (no need to
    handle repaint events from OS).
-   Add trackbars to the windows, handle simple mouse events as well as keyboard commands.
<p>
\{
    \defgroup highgui_opengl OpenGL support
    \defgroup highgui_qt Qt New Functions
    <p>
    ![image](pics/qtgui.png)
    <p>
    This figure explains new functionality implemented with Qt\* GUI. The new GUI provides a statusbar,
    a toolbar, and a control panel. The control panel can have trackbars and buttonbars attached to it.
    If you cannot see the control panel, press Ctrl+P or right-click any Qt window and select **Display
    properties window**.
    <p>
    -   To attach a trackbar, the window name parameter must be NULL.
    <p>
    -   To attach a buttonbar, a button must be created. If the last bar attached to the control panel
        is a buttonbar, the new button is added to the right of the last button. If the last bar
        attached to the control panel is a trackbar, or the control panel is empty, a new buttonbar is
        created. Then, a new button is attached to it.
    <p>
    See below the example used to generate the figure: :
    <pre>{@code
        int main(int argc, char *argv[])
            int value = 50;
            int value2 = 0;

            cvNamedWindow("main1",CV_WINDOW_NORMAL);
            cvNamedWindow("main2",CV_WINDOW_AUTOSIZE | CV_GUI_NORMAL);

            cvCreateTrackbar( "track1", "main1", &value, 255,  NULL);//OK tested
            char* nameb1 = "button1";
            char* nameb2 = "button2";
            cvCreateButton(nameb1,callbackButton,nameb1,CV_CHECKBOX,1);

            cvCreateButton(nameb2,callbackButton,nameb2,CV_CHECKBOX,0);
            cvCreateTrackbar( "track2", NULL, &value2, 255, NULL);
            cvCreateButton("button5",callbackButton1,NULL,CV_RADIOBOX,0);
            cvCreateButton("button6",callbackButton2,NULL,CV_RADIOBOX,1);

            cvSetMouseCallback( "main2",on_mouse,NULL );

            IplImage* img1 = cvLoadImage("files/flower.jpg");
            IplImage* img2 = cvCreateImage(cvGetSize(img1),8,3);
            CvCapture* video = cvCaptureFromFile("files/hockey.avi");
            IplImage* img3 = cvCreateImage(cvGetSize(cvQueryFrame(video)),8,3);

            while(cvWaitKey(33) != 27)
            {
                cvAddS(img1,cvScalarAll(value),img2);
                cvAddS(cvQueryFrame(video),cvScalarAll(value2),img3);
                cvShowImage("main1",img2);
                cvShowImage("main2",img3);
            }

            cvDestroyAllWindows();
            cvReleaseImage(&img1);
            cvReleaseImage(&img2);
            cvReleaseImage(&img3);
            cvReleaseCapture(&video);
            return 0;
        }
    }</pre>
    <p>
    \defgroup highgui_c C API
\}
*/

///////////////////////// graphical user interface //////////////////////////

/** \addtogroup highgui
 *  \{ */

// Flags for namedWindow
/** enum cv:: */
public static final int WINDOW_NORMAL     =  0x00000000, // the user can resize the window (no constraint) / also use to switch a fullscreen window to a normal size
       WINDOW_AUTOSIZE   =  0x00000001, // the user cannot resize the window, the size is constrainted by the image displayed
       WINDOW_OPENGL     =  0x00001000, // window with opengl support

       WINDOW_FULLSCREEN = 1,          // change the window to fullscreen
       WINDOW_FREERATIO  =  0x00000100, // the image expends as much as it can (no ratio constraint)
       WINDOW_KEEPRATIO  =  0x00000000;  // the ratio of the image is respected

// Flags for set / getWindowProperty
/** enum cv:: */
public static final int WND_PROP_FULLSCREEN   = 0, // fullscreen property    (can be WINDOW_NORMAL or WINDOW_FULLSCREEN)
       WND_PROP_AUTOSIZE     = 1, // autosize property      (can be WINDOW_NORMAL or WINDOW_AUTOSIZE)
       WND_PROP_ASPECT_RATIO = 2, // window's aspect ration (can be set to WINDOW_FREERATIO or WINDOW_KEEPRATIO);
       WND_PROP_OPENGL       = 3;  // opengl support

/** enum cv:: */
public static final int EVENT_MOUSEMOVE      = 0,
       EVENT_LBUTTONDOWN    = 1,
       EVENT_RBUTTONDOWN    = 2,
       EVENT_MBUTTONDOWN    = 3,
       EVENT_LBUTTONUP      = 4,
       EVENT_RBUTTONUP      = 5,
       EVENT_MBUTTONUP      = 6,
       EVENT_LBUTTONDBLCLK  = 7,
       EVENT_RBUTTONDBLCLK  = 8,
       EVENT_MBUTTONDBLCLK  = 9,
       EVENT_MOUSEWHEEL     = 10,
       EVENT_MOUSEHWHEEL    = 11;

/** enum cv:: */
public static final int EVENT_FLAG_LBUTTON   = 1,
       EVENT_FLAG_RBUTTON   = 2,
       EVENT_FLAG_MBUTTON   = 4,
       EVENT_FLAG_CTRLKEY   = 8,
       EVENT_FLAG_SHIFTKEY  = 16,
       EVENT_FLAG_ALTKEY    = 32;

// Qt font
/** enum cv:: */
public static final int  QT_FONT_LIGHT           = 25, //QFont::Light,
        QT_FONT_NORMAL          = 50, //QFont::Normal,
        QT_FONT_DEMIBOLD        = 63, //QFont::DemiBold,
        QT_FONT_BOLD            = 75, //QFont::Bold,
        QT_FONT_BLACK           = 87;  //QFont::Black

// Qt font style
/** enum cv:: */
public static final int  QT_STYLE_NORMAL         = 0, //QFont::StyleNormal,
        QT_STYLE_ITALIC         = 1, //QFont::StyleItalic,
        QT_STYLE_OBLIQUE        = 2;  //QFont::StyleOblique

// Qt "button" type
/** enum cv:: */
public static final int QT_PUSH_BUTTON = 0,
       QT_CHECKBOX    = 1,
       QT_RADIOBOX    = 2;


public static class MouseCallback extends FunctionPointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public    MouseCallback(Pointer p) { super(p); }
    protected MouseCallback() { allocate(); }
    private native void allocate();
    public native void call(int event, int x, int y, int flags, Pointer userdata);
}
public static class TrackbarCallback extends FunctionPointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public    TrackbarCallback(Pointer p) { super(p); }
    protected TrackbarCallback() { allocate(); }
    private native void allocate();
    public native void call(int pos, Pointer userdata);
}
public static class OpenGlDrawCallback extends FunctionPointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public    OpenGlDrawCallback(Pointer p) { super(p); }
    protected OpenGlDrawCallback() { allocate(); }
    private native void allocate();
    public native void call(Pointer userdata);
}
public static class ButtonCallback extends FunctionPointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public    ButtonCallback(Pointer p) { super(p); }
    protected ButtonCallback() { allocate(); }
    private native void allocate();
    public native void call(int state, Pointer userdata);
}

/** \brief Creates a window.
<p>
@param winname Name of the window in the window caption that may be used as a window identifier.
@param flags Flags of the window. The supported flags are:
> -   **WINDOW_NORMAL** If this is set, the user can resize the window (no constraint).
> -   **WINDOW_AUTOSIZE** If this is set, the window size is automatically adjusted to fit the
>     displayed image (see imshow ), and you cannot change the window size manually.
> -   **WINDOW_OPENGL** If this is set, the window will be created with OpenGL support.
<p>
The function namedWindow creates a window that can be used as a placeholder for images and
trackbars. Created windows are referred to by their names.
<p>
If a window with the same name already exists, the function does nothing.
<p>
You can call destroyWindow or destroyAllWindows to close the window and de-allocate any associated
memory usage. For a simple program, you do not really have to call these functions because all the
resources and windows of the application are closed automatically by the operating system upon exit.
<p>
\note
<p>
Qt backend supports additional flags:
 -   **CV_WINDOW_NORMAL or CV_WINDOW_AUTOSIZE:** CV_WINDOW_NORMAL enables you to resize the
     window, whereas CV_WINDOW_AUTOSIZE adjusts automatically the window size to fit the
     displayed image (see imshow ), and you cannot change the window size manually.
 -   **CV_WINDOW_FREERATIO or CV_WINDOW_KEEPRATIO:** CV_WINDOW_FREERATIO adjusts the image
     with no respect to its ratio, whereas CV_WINDOW_KEEPRATIO keeps the image ratio.
 -   **CV_GUI_NORMAL or CV_GUI_EXPANDED:** CV_GUI_NORMAL is the old way to draw the window
     without statusbar and toolbar, whereas CV_GUI_EXPANDED is a new enhanced GUI.
By default, flags == CV_WINDOW_AUTOSIZE | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED
 */
@Namespace("cv") public static native void namedWindow(@Str BytePointer winname, int flags/*=cv::WINDOW_AUTOSIZE*/);
@Namespace("cv") public static native void namedWindow(@Str BytePointer winname);
@Namespace("cv") public static native void namedWindow(@Str String winname, int flags/*=cv::WINDOW_AUTOSIZE*/);
@Namespace("cv") public static native void namedWindow(@Str String winname);

/** \brief Destroys a window.
<p>
@param winname Name of the window to be destroyed.
<p>
The function destroyWindow destroys the window with the given name.
 */
@Namespace("cv") public static native void destroyWindow(@Str BytePointer winname);
@Namespace("cv") public static native void destroyWindow(@Str String winname);

/** \brief Destroys all of the HighGUI windows.
<p>
The function destroyAllWindows destroys all of the opened HighGUI windows.
 */
@Namespace("cv") public static native void destroyAllWindows();

@Namespace("cv") public static native int startWindowThread();

/** \brief Waits for a pressed key.
<p>
@param delay Delay in milliseconds. 0 is the special value that means "forever".
<p>
The function waitKey waits for a key event infinitely (when \f$\texttt{delay}\leq 0\f$ ) or for delay
milliseconds, when it is positive. Since the OS has a minimum time between switching threads, the
function will not wait exactly delay ms, it will wait at least delay ms, depending on what else is
running on your computer at that time. It returns the code of the pressed key or -1 if no key was
pressed before the specified time had elapsed.
<p>
\note
<p>
This function is the only method in HighGUI that can fetch and handle events, so it needs to be
called periodically for normal event processing unless HighGUI is used within an environment that
takes care of event processing.
<p>
\note
<p>
The function only works if there is at least one HighGUI window created and the window is active.
If there are several HighGUI windows, any of them can be active.
 */
@Namespace("cv") public static native int waitKey(int delay/*=0*/);
@Namespace("cv") public static native int waitKey();

/** \brief Displays an image in the specified window.
<p>
@param winname Name of the window.
@param mat Image to be shown.
<p>
The function imshow displays an image in the specified window. If the window was created with the
CV_WINDOW_AUTOSIZE flag, the image is shown with its original size, however it is still limited by the screen resolution.
Otherwise, the image is scaled to fit the window. The function may scale the image, depending on its depth:
<p>
-   If the image is 8-bit unsigned, it is displayed as is.
-   If the image is 16-bit unsigned or 32-bit integer, the pixels are divided by 256. That is, the
    value range [0,255\*256] is mapped to [0,255].
-   If the image is 32-bit floating-point, the pixel values are multiplied by 255. That is, the
    value range [0,1] is mapped to [0,255].
<p>
If window was created with OpenGL support, imshow also support ogl::Buffer , ogl::Texture2D and
cuda::GpuMat as input.
<p>
If the window was not created before this function, it is assumed creating a window with CV_WINDOW_AUTOSIZE.
<p>
If you need to show an image that is bigger than the screen resolution, you will need to call namedWindow("", WINDOW_NORMAL) before the imshow.
<p>
\note This function should be followed by waitKey function which displays the image for specified
milliseconds. Otherwise, it won't display the image. For example, waitKey(0) will display the window
infinitely until any keypress (it is suitable for image display). waitKey(25) will display a frame
for 25 ms, after which display will be automatically closed. (If you put it in a loop to read
videos, it will display the video frame-by-frame)
<p>
\note
<p>
[Windows Backend Only] Pressing Ctrl+C will copy the image to the clipboard.
 <p>
 */
@Namespace("cv") public static native void imshow(@Str BytePointer winname, @ByVal Mat mat);
@Namespace("cv") public static native void imshow(@Str String winname, @ByVal Mat mat);

/** \brief Resizes window to the specified size
<p>
@param winname Window name
@param width The new window width
@param height The new window height
<p>
\note
<p>
-   The specified window size is for the image area. Toolbars are not counted.
-   Only windows created without CV_WINDOW_AUTOSIZE flag can be resized.
 */
@Namespace("cv") public static native void resizeWindow(@Str BytePointer winname, int width, int height);
@Namespace("cv") public static native void resizeWindow(@Str String winname, int width, int height);

/** \brief Moves window to the specified position
<p>
@param winname Window name
@param x The new x-coordinate of the window
@param y The new y-coordinate of the window
 */
@Namespace("cv") public static native void moveWindow(@Str BytePointer winname, int x, int y);
@Namespace("cv") public static native void moveWindow(@Str String winname, int x, int y);

/** \brief Changes parameters of a window dynamically.
<p>
@param winname Name of the window.
@param prop_id Window property to edit. The following operation flags are available:
 -   **CV_WND_PROP_FULLSCREEN** Change if the window is fullscreen ( CV_WINDOW_NORMAL or
     CV_WINDOW_FULLSCREEN ).
 -   **CV_WND_PROP_AUTOSIZE** Change if the window is resizable (CV_WINDOW_NORMAL or
     CV_WINDOW_AUTOSIZE ).
 -   **CV_WND_PROP_ASPECTRATIO** Change if the aspect ratio of the image is preserved (
     CV_WINDOW_FREERATIO or CV_WINDOW_KEEPRATIO ).
@param prop_value New value of the window property. The following operation flags are available:
 -   **CV_WINDOW_NORMAL** Change the window to normal size or make the window resizable.
 -   **CV_WINDOW_AUTOSIZE** Constrain the size by the displayed image. The window is not
     resizable.
 -   **CV_WINDOW_FULLSCREEN** Change the window to fullscreen.
 -   **CV_WINDOW_FREERATIO** Make the window resizable without any ratio constraints.
 -   **CV_WINDOW_KEEPRATIO** Make the window resizable, but preserve the proportions of the
     displayed image.
<p>
The function setWindowProperty enables changing properties of a window.
 */
@Namespace("cv") public static native void setWindowProperty(@Str BytePointer winname, int prop_id, double prop_value);
@Namespace("cv") public static native void setWindowProperty(@Str String winname, int prop_id, double prop_value);

/** \brief Updates window title
*/
@Namespace("cv") public static native void setWindowTitle(@Str BytePointer winname, @Str BytePointer title);
@Namespace("cv") public static native void setWindowTitle(@Str String winname, @Str String title);

/** \brief Provides parameters of a window.
<p>
@param winname Name of the window.
@param prop_id Window property to retrieve. The following operation flags are available:
 -   **CV_WND_PROP_FULLSCREEN** Change if the window is fullscreen ( CV_WINDOW_NORMAL or
     CV_WINDOW_FULLSCREEN ).
 -   **CV_WND_PROP_AUTOSIZE** Change if the window is resizable (CV_WINDOW_NORMAL or
     CV_WINDOW_AUTOSIZE ).
 -   **CV_WND_PROP_ASPECTRATIO** Change if the aspect ratio of the image is preserved
     (CV_WINDOW_FREERATIO or CV_WINDOW_KEEPRATIO ).
<p>
See setWindowProperty to know the meaning of the returned values.
<p>
The function getWindowProperty returns properties of a window.
 */
@Namespace("cv") public static native double getWindowProperty(@Str BytePointer winname, int prop_id);
@Namespace("cv") public static native double getWindowProperty(@Str String winname, int prop_id);

/** \brief Sets mouse handler for the specified window
<p>
@param winname Window name
@param onMouse Mouse callback. See OpenCV samples, such as
<https://github.com/Itseez/opencv/tree/master/samples/cpp/ffilldemo.cpp>, on how to specify and
use the callback.
@param userdata The optional parameter passed to the callback.
 */
@Namespace("cv") public static native void setMouseCallback(@Str BytePointer winname, MouseCallback onMouse, Pointer userdata/*=0*/);
@Namespace("cv") public static native void setMouseCallback(@Str BytePointer winname, MouseCallback onMouse);
@Namespace("cv") public static native void setMouseCallback(@Str String winname, MouseCallback onMouse, Pointer userdata/*=0*/);
@Namespace("cv") public static native void setMouseCallback(@Str String winname, MouseCallback onMouse);

/** \brief Gets the mouse-wheel motion delta, when handling mouse-wheel events EVENT_MOUSEWHEEL and
EVENT_MOUSEHWHEEL.
<p>
@param flags The mouse callback flags parameter.
<p>
For regular mice with a scroll-wheel, delta will be a multiple of 120. The value 120 corresponds to
a one notch rotation of the wheel or the threshold for action to be taken and one such action should
occur for each delta. Some high-precision mice with higher-resolution freely-rotating wheels may
generate smaller values.
<p>
For EVENT_MOUSEWHEEL positive and negative values mean forward and backward scrolling,
respectively. For EVENT_MOUSEHWHEEL, where available, positive and negative values mean right and
left scrolling, respectively.
<p>
With the C API, the macro CV_GET_WHEEL_DELTA(flags) can be used alternatively.
<p>
\note
<p>
Mouse-wheel events are currently supported only on Windows.
 */
@Namespace("cv") public static native int getMouseWheelDelta(int flags);

/** \brief Creates a trackbar and attaches it to the specified window.
<p>
@param trackbarname Name of the created trackbar.
@param winname Name of the window that will be used as a parent of the created trackbar.
@param value Optional pointer to an integer variable whose value reflects the position of the
slider. Upon creation, the slider position is defined by this variable.
@param count Maximal position of the slider. The minimal position is always 0.
@param onChange Pointer to the function to be called every time the slider changes position. This
function should be prototyped as void Foo(int,void\*); , where the first parameter is the trackbar
position and the second parameter is the user data (see the next parameter). If the callback is
the NULL pointer, no callbacks are called, but only value is updated.
@param userdata User data that is passed as is to the callback. It can be used to handle trackbar
events without using global variables.
<p>
The function createTrackbar creates a trackbar (a slider or range control) with the specified name
and range, assigns a variable value to be a position synchronized with the trackbar and specifies
the callback function onChange to be called on the trackbar position change. The created trackbar is
displayed in the specified window winname.
<p>
\note
<p>
**[Qt Backend Only]** winname can be empty (or NULL) if the trackbar should be attached to the
control panel.
<p>
Clicking the label of each trackbar enables editing the trackbar values manually.
<p>
\note
<p>
-   An example of using the trackbar functionality can be found at
    opencv_source_code/samples/cpp/connected_components.cpp
 */
@Namespace("cv") public static native int createTrackbar(@Str BytePointer trackbarname, @Str BytePointer winname,
                              IntPointer value, int count,
                              TrackbarCallback onChange/*=0*/,
                              Pointer userdata/*=0*/);
@Namespace("cv") public static native int createTrackbar(@Str BytePointer trackbarname, @Str BytePointer winname,
                              IntPointer value, int count);
@Namespace("cv") public static native int createTrackbar(@Str String trackbarname, @Str String winname,
                              IntBuffer value, int count,
                              TrackbarCallback onChange/*=0*/,
                              Pointer userdata/*=0*/);
@Namespace("cv") public static native int createTrackbar(@Str String trackbarname, @Str String winname,
                              IntBuffer value, int count);
@Namespace("cv") public static native int createTrackbar(@Str BytePointer trackbarname, @Str BytePointer winname,
                              int[] value, int count,
                              TrackbarCallback onChange/*=0*/,
                              Pointer userdata/*=0*/);
@Namespace("cv") public static native int createTrackbar(@Str BytePointer trackbarname, @Str BytePointer winname,
                              int[] value, int count);
@Namespace("cv") public static native int createTrackbar(@Str String trackbarname, @Str String winname,
                              IntPointer value, int count,
                              TrackbarCallback onChange/*=0*/,
                              Pointer userdata/*=0*/);
@Namespace("cv") public static native int createTrackbar(@Str String trackbarname, @Str String winname,
                              IntPointer value, int count);
@Namespace("cv") public static native int createTrackbar(@Str BytePointer trackbarname, @Str BytePointer winname,
                              IntBuffer value, int count,
                              TrackbarCallback onChange/*=0*/,
                              Pointer userdata/*=0*/);
@Namespace("cv") public static native int createTrackbar(@Str BytePointer trackbarname, @Str BytePointer winname,
                              IntBuffer value, int count);
@Namespace("cv") public static native int createTrackbar(@Str String trackbarname, @Str String winname,
                              int[] value, int count,
                              TrackbarCallback onChange/*=0*/,
                              Pointer userdata/*=0*/);
@Namespace("cv") public static native int createTrackbar(@Str String trackbarname, @Str String winname,
                              int[] value, int count);

/** \brief Returns the trackbar position.
<p>
@param trackbarname Name of the trackbar.
@param winname Name of the window that is the parent of the trackbar.
<p>
The function returns the current position of the specified trackbar.
<p>
\note
<p>
**[Qt Backend Only]** winname can be empty (or NULL) if the trackbar is attached to the control
panel.
 <p>
 */
@Namespace("cv") public static native int getTrackbarPos(@Str BytePointer trackbarname, @Str BytePointer winname);
@Namespace("cv") public static native int getTrackbarPos(@Str String trackbarname, @Str String winname);

/** \brief Sets the trackbar position.
<p>
@param trackbarname Name of the trackbar.
@param winname Name of the window that is the parent of trackbar.
@param pos New position.
<p>
The function sets the position of the specified trackbar in the specified window.
<p>
\note
<p>
**[Qt Backend Only]** winname can be empty (or NULL) if the trackbar is attached to the control
panel.
 */
@Namespace("cv") public static native void setTrackbarPos(@Str BytePointer trackbarname, @Str BytePointer winname, int pos);
@Namespace("cv") public static native void setTrackbarPos(@Str String trackbarname, @Str String winname, int pos);

/** \brief Sets the trackbar maximum position.
<p>
@param trackbarname Name of the trackbar.
@param winname Name of the window that is the parent of trackbar.
@param maxval New maximum position.
<p>
The function sets the maximum position of the specified trackbar in the specified window.
<p>
\note
<p>
**[Qt Backend Only]** winname can be empty (or NULL) if the trackbar is attached to the control
panel.
 */
@Namespace("cv") public static native void setTrackbarMax(@Str BytePointer trackbarname, @Str BytePointer winname, int maxval);
@Namespace("cv") public static native void setTrackbarMax(@Str String trackbarname, @Str String winname, int maxval);

/** \addtogroup highgui_opengl OpenGL support
 *  \{ */

@Namespace("cv") public static native void imshow(@Str BytePointer winname, @Const @ByRef Texture2D tex);
@Namespace("cv") public static native void imshow(@Str String winname, @Const @ByRef Texture2D tex);

/** \brief Sets a callback function to be called to draw on top of displayed image.
<p>
@param winname Name of the window.
@param onOpenGlDraw Pointer to the function to be called every frame. This function should be
prototyped as void Foo(void\*) .
@param userdata Pointer passed to the callback function. *(Optional)*
<p>
The function setOpenGlDrawCallback can be used to draw 3D data on the window. See the example of
callback function below: :
<pre>{@code
    void on_opengl(void* param)
    {
        glLoadIdentity();

        glTranslated(0.0, 0.0, -1.0);

        glRotatef( 55, 1, 0, 0 );
        glRotatef( 45, 0, 1, 0 );
        glRotatef( 0, 0, 0, 1 );

        static const int coords[6][4][3] = {
            { { +1, -1, -1 }, { -1, -1, -1 }, { -1, +1, -1 }, { +1, +1, -1 } },
            { { +1, +1, -1 }, { -1, +1, -1 }, { -1, +1, +1 }, { +1, +1, +1 } },
            { { +1, -1, +1 }, { +1, -1, -1 }, { +1, +1, -1 }, { +1, +1, +1 } },
            { { -1, -1, -1 }, { -1, -1, +1 }, { -1, +1, +1 }, { -1, +1, -1 } },
            { { +1, -1, +1 }, { -1, -1, +1 }, { -1, -1, -1 }, { +1, -1, -1 } },
            { { -1, -1, +1 }, { +1, -1, +1 }, { +1, +1, +1 }, { -1, +1, +1 } }
        };

        for (int i = 0; i < 6; ++i) {
                    glColor3ub( i*20, 100+i*10, i*42 );
                    glBegin(GL_QUADS);
                    for (int j = 0; j < 4; ++j) {
                            glVertex3d(0.2 * coords[i][j][0], 0.2 * coords[i][j][1], 0.2 * coords[i][j][2]);
                    }
                    glEnd();
        }
    }
}</pre>
 */
@Namespace("cv") public static native void setOpenGlDrawCallback(@Str BytePointer winname, OpenGlDrawCallback onOpenGlDraw, Pointer userdata/*=0*/);
@Namespace("cv") public static native void setOpenGlDrawCallback(@Str BytePointer winname, OpenGlDrawCallback onOpenGlDraw);
@Namespace("cv") public static native void setOpenGlDrawCallback(@Str String winname, OpenGlDrawCallback onOpenGlDraw, Pointer userdata/*=0*/);
@Namespace("cv") public static native void setOpenGlDrawCallback(@Str String winname, OpenGlDrawCallback onOpenGlDraw);

/** \brief Sets the specified window as current OpenGL context.
<p>
@param winname Window name
 */
@Namespace("cv") public static native void setOpenGlContext(@Str BytePointer winname);
@Namespace("cv") public static native void setOpenGlContext(@Str String winname);

/** \brief Force window to redraw its context and call draw callback ( setOpenGlDrawCallback ).
<p>
@param winname Window name
 */
@Namespace("cv") public static native void updateWindow(@Str BytePointer winname);
@Namespace("cv") public static native void updateWindow(@Str String winname);

/** \} highgui_opengl
 <p>
 *  \addtogroup highgui_qt
 *  \{ */
// Only for Qt

@Namespace("cv") public static class QtFont extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public QtFont() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public QtFont(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public QtFont(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public QtFont position(int position) {
        return (QtFont)super.position(position);
    }

    @MemberGetter public native @Cast("const char*") BytePointer nameFont();  // Qt: nameFont
    public native @ByRef Scalar color(); public native QtFont color(Scalar color);     // Qt: ColorFont -> cvScalar(blue_component, green_component, red_component[, alpha_component])
    public native int font_face(); public native QtFont font_face(int font_face); // Qt: bool italic
    @MemberGetter public native @Const IntPointer ascii();     // font data and metrics
    @MemberGetter public native @Const IntPointer greek();
    @MemberGetter public native @Const IntPointer cyrillic();
    public native float hscale(); public native QtFont hscale(float hscale);
    public native float vscale(); public native QtFont vscale(float vscale);
    public native float shear(); public native QtFont shear(float shear);     // slope coefficient: 0 - normal, >0 - italic
    public native int thickness(); public native QtFont thickness(int thickness); // Qt: weight
    public native float dx(); public native QtFont dx(float dx);        // horizontal interval between letters
    public native int line_type(); public native QtFont line_type(int line_type); // Qt: PointSize
}

/** \brief Creates the font to draw a text on an image.
<p>
@param nameFont Name of the font. The name should match the name of a system font (such as
*Times*). If the font is not found, a default one is used.
@param pointSize Size of the font. If not specified, equal zero or negative, the point size of the
font is set to a system-dependent default value. Generally, this is 12 points.
@param color Color of the font in BGRA where A = 255 is fully transparent. Use the macro CV _ RGB
for simplicity.
@param weight Font weight. The following operation flags are available:
 -   **CV_FONT_LIGHT** Weight of 25
 -   **CV_FONT_NORMAL** Weight of 50
 -   **CV_FONT_DEMIBOLD** Weight of 63
 -   **CV_FONT_BOLD** Weight of 75
 -   **CV_FONT_BLACK** Weight of 87
 <p>
 You can also specify a positive integer for better control.
@param style Font style. The following operation flags are available:
 -   **CV_STYLE_NORMAL** Normal font
 -   **CV_STYLE_ITALIC** Italic font
 -   **CV_STYLE_OBLIQUE** Oblique font
@param spacing Spacing between characters. It can be negative or positive.
<p>
The function fontQt creates a CvFont object. This CvFont is not compatible with putText .
<p>
A basic usage of this function is the following: :
<pre>{@code
    CvFont font = fontQt(''Times'');
    addText( img1, ``Hello World !'', Point(50,50), font);
}</pre>
 */
@Namespace("cv") public static native @ByVal QtFont fontQt(@Str BytePointer nameFont, int pointSize/*=-1*/,
                         @ByVal(nullValue = "cv::Scalar::all(0)") Scalar color/*=cv::Scalar::all(0)*/, int weight/*=cv::QT_FONT_NORMAL*/,
                         int style/*=cv::QT_STYLE_NORMAL*/, int spacing/*=0*/);
@Namespace("cv") public static native @ByVal QtFont fontQt(@Str BytePointer nameFont);
@Namespace("cv") public static native @ByVal QtFont fontQt(@Str String nameFont, int pointSize/*=-1*/,
                         @ByVal(nullValue = "cv::Scalar::all(0)") Scalar color/*=cv::Scalar::all(0)*/, int weight/*=cv::QT_FONT_NORMAL*/,
                         int style/*=cv::QT_STYLE_NORMAL*/, int spacing/*=0*/);
@Namespace("cv") public static native @ByVal QtFont fontQt(@Str String nameFont);

/** \brief Creates the font to draw a text on an image.
<p>
@param img 8-bit 3-channel image where the text should be drawn.
@param text Text to write on an image.
@param org Point(x,y) where the text should start on an image.
@param font Font to use to draw a text.
<p>
The function addText draws *text* on an image *img* using a specific font *font* (see example fontQt
)
 */
@Namespace("cv") public static native void addText( @Const @ByRef Mat img, @Str BytePointer text, @ByVal Point org, @Const @ByRef QtFont font);
@Namespace("cv") public static native void addText( @Const @ByRef Mat img, @Str String text, @ByVal Point org, @Const @ByRef QtFont font);

/** \brief Displays a text on a window image as an overlay for a specified duration.
<p>
@param winname Name of the window.
@param text Overlay text to write on a window image.
@param delayms The period (in milliseconds), during which the overlay text is displayed. If this
function is called before the previous overlay text timed out, the timer is restarted and the text
is updated. If this value is zero, the text never disappears.
<p>
The function displayOverlay displays useful information/tips on top of the window for a certain
amount of time *delayms*. The function does not modify the image, displayed in the window, that is,
after the specified delay the original content of the window is restored.
 */
@Namespace("cv") public static native void displayOverlay(@Str BytePointer winname, @Str BytePointer text, int delayms/*=0*/);
@Namespace("cv") public static native void displayOverlay(@Str BytePointer winname, @Str BytePointer text);
@Namespace("cv") public static native void displayOverlay(@Str String winname, @Str String text, int delayms/*=0*/);
@Namespace("cv") public static native void displayOverlay(@Str String winname, @Str String text);

/** \brief Displays a text on the window statusbar during the specified period of time.
<p>
@param winname Name of the window.
@param text Text to write on the window statusbar.
@param delayms Duration (in milliseconds) to display the text. If this function is called before
the previous text timed out, the timer is restarted and the text is updated. If this value is
zero, the text never disappears.
<p>
The function displayOverlay displays useful information/tips on top of the window for a certain
amount of time *delayms* . This information is displayed on the window statusbar (the window must be
created with the CV_GUI_EXPANDED flags).
 */
@Namespace("cv") public static native void displayStatusBar(@Str BytePointer winname, @Str BytePointer text, int delayms/*=0*/);
@Namespace("cv") public static native void displayStatusBar(@Str BytePointer winname, @Str BytePointer text);
@Namespace("cv") public static native void displayStatusBar(@Str String winname, @Str String text, int delayms/*=0*/);
@Namespace("cv") public static native void displayStatusBar(@Str String winname, @Str String text);

/** \brief Saves parameters of the specified window.
<p>
@param windowName Name of the window.
<p>
The function saveWindowParameters saves size, location, flags, trackbars value, zoom and panning
location of the window window_name .
 */
@Namespace("cv") public static native void saveWindowParameters(@Str BytePointer windowName);
@Namespace("cv") public static native void saveWindowParameters(@Str String windowName);

/** \brief Loads parameters of the specified window.
<p>
@param windowName Name of the window.
<p>
The function loadWindowParameters loads size, location, flags, trackbars value, zoom and panning
location of the window window_name .
 */
@Namespace("cv") public static native void loadWindowParameters(@Str BytePointer windowName);
@Namespace("cv") public static native void loadWindowParameters(@Str String windowName);

@Namespace("cv") public static native int startLoop(Pt2Func_int_PointerPointer pt2Func, int argc, @Cast("char**") PointerPointer argv);
@Namespace("cv") public static native int startLoop(Pt2Func_int_BytePointer pt2Func, int argc, @Cast("char**") @ByPtrPtr BytePointer argv);
@Namespace("cv") public static native int startLoop(Pt2Func_int_ByteBuffer pt2Func, int argc, @Cast("char**") @ByPtrPtr ByteBuffer argv);
@Namespace("cv") public static native int startLoop(Pt2Func_int_byte__ pt2Func, int argc, @Cast("char**") @ByPtrPtr byte[] argv);

@Namespace("cv") public static native void stopLoop();

/** \brief Attaches a button to the control panel.
<p>
@param  bar_name
   Name of the button.
@param on_change Pointer to the function to be called every time the button changes its state.
This function should be prototyped as void Foo(int state,\*void); . *state* is the current state
of the button. It could be -1 for a push button, 0 or 1 for a check/radio box button.
@param userdata Pointer passed to the callback function.
@param type Optional type of the button.
 -   **CV_PUSH_BUTTON** Push button
 -   **CV_CHECKBOX** Checkbox button
 -   **CV_RADIOBOX** Radiobox button. The radiobox on the same buttonbar (same line) are
     exclusive, that is only one can be selected at a time.
@param initial_button_state Default state of the button. Use for checkbox and radiobox. Its
value could be 0 or 1. *(Optional)*
<p>
The function createButton attaches a button to the control panel. Each button is added to a
buttonbar to the right of the last button. A new buttonbar is created if nothing was attached to the
control panel before, or if the last element attached to the control panel was a trackbar.
<p>
See below various examples of the createButton function call: :
<pre>{@code
    createButton(NULL,callbackButton);//create a push button "button 0", that will call callbackButton.
    createButton("button2",callbackButton,NULL,CV_CHECKBOX,0);
    createButton("button3",callbackButton,&value);
    createButton("button5",callbackButton1,NULL,CV_RADIOBOX);
    createButton("button6",callbackButton2,NULL,CV_PUSH_BUTTON,1);
}</pre>
*/
@Namespace("cv") public static native int createButton( @Str BytePointer bar_name, ButtonCallback on_change,
                             Pointer userdata/*=0*/, int type/*=cv::QT_PUSH_BUTTON*/,
                             @Cast("bool") boolean initial_button_state/*=false*/);
@Namespace("cv") public static native int createButton( @Str BytePointer bar_name, ButtonCallback on_change);
@Namespace("cv") public static native int createButton( @Str String bar_name, ButtonCallback on_change,
                             Pointer userdata/*=0*/, int type/*=cv::QT_PUSH_BUTTON*/,
                             @Cast("bool") boolean initial_button_state/*=false*/);
@Namespace("cv") public static native int createButton( @Str String bar_name, ButtonCallback on_change);

/** \} highgui_qt
 <p>
 *  \} highgui */

 // cv

// #ifndef DISABLE_OPENCV_24_COMPATIBILITY
// #include "opencv2/highgui/highgui_c.h"
// #endif

// #endif


}
