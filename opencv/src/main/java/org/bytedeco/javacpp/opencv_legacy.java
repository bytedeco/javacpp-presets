// Targeted by JavaCPP version 0.11

package org.bytedeco.javacpp;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_highgui.*;
import static org.bytedeco.javacpp.opencv_flann.*;
import static org.bytedeco.javacpp.opencv_features2d.*;
import static org.bytedeco.javacpp.opencv_calib3d.*;
import static org.bytedeco.javacpp.opencv_video.*;
import static org.bytedeco.javacpp.opencv_ml.*;

public class opencv_legacy extends org.bytedeco.javacpp.helper.opencv_legacy {
    static { Loader.load(); }

// Parsed from <opencv2/legacy/blobtrack.hpp>

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


// #ifndef __OPENCV_VIDEOSURVEILLANCE_H__
// #define __OPENCV_VIDEOSURVEILLANCE_H__

/* Turn off the functionality until cvaux/src/Makefile.am gets updated: */
//#if _MSC_VER >= 1200

// #include "opencv2/core/core_c.h"
// #include <stdio.h>

// #if (defined _MSC_VER && _MSC_VER >= 1200) || defined __BORLANDC__
// #define cv_stricmp stricmp
// #define cv_strnicmp strnicmp
// #if defined WINCE
// #define strdup _strdup
// #define stricmp _stricmp
// #endif
// #elif defined __GNUC__ || defined __sun
// #define cv_stricmp strcasecmp
// #define cv_strnicmp strncasecmp
// #else
// #error Do not know how to make case-insensitive string comparison on this platform
// #endif

//struct DefParam;
public static class CvDefParam extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvDefParam() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvDefParam(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvDefParam(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvDefParam position(int position) {
        return (CvDefParam)super.position(position);
    }

    public native CvDefParam next(); public native CvDefParam next(CvDefParam next);
    public native @Cast("char*") BytePointer pName(); public native CvDefParam pName(BytePointer pName);
    public native @Cast("char*") BytePointer pComment(); public native CvDefParam pComment(BytePointer pComment);
    public native DoublePointer pDouble(); public native CvDefParam pDouble(DoublePointer pDouble);
    public native double Double(); public native CvDefParam Double(double Double);
    public native FloatPointer pFloat(); public native CvDefParam pFloat(FloatPointer pFloat);
    public native float Float(); public native CvDefParam Float(float Float);
    public native IntPointer pInt(); public native CvDefParam pInt(IntPointer pInt);
    public native int Int(); public native CvDefParam Int(int Int);
    public native @Cast("char*") BytePointer pStr(int i); public native CvDefParam pStr(int i, BytePointer pStr);
    @MemberGetter public native @Cast("char**") PointerPointer pStr();
    public native @Cast("char*") BytePointer Str(); public native CvDefParam Str(BytePointer Str);
}

@NoOffset public static class CvVSModule extends Pointer {
    static { Loader.load(); }
    /** Empty constructor. */
    public CvVSModule() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvVSModule(Pointer p) { super(p); }
 /* Constructor and destructor: */ /* EXTERNAL INTERFACE */
    public native @Cast("const char*") BytePointer GetParamName(int index);
    public native @Cast("const char*") BytePointer GetParamComment(@Cast("const char*") BytePointer name);
    public native String GetParamComment(String name);
    public native double GetParam(@Cast("const char*") BytePointer name);
    public native double GetParam(String name);
    public native @Cast("const char*") BytePointer GetParamStr(@Cast("const char*") BytePointer name);
    public native String GetParamStr(String name);
    public native void SetParam(@Cast("const char*") BytePointer name, double val);
    public native void SetParam(String name, double val);
    public native void SetParamStr(@Cast("const char*") BytePointer name, @Cast("const char*") BytePointer str);
    public native void SetParamStr(String name, String str);
    public native void TransferParamsFromChild(CvVSModule pM, @Cast("const char*") BytePointer prefix/*=NULL*/);
    public native void TransferParamsFromChild(CvVSModule pM);
    public native void TransferParamsFromChild(CvVSModule pM, String prefix/*=NULL*/);
    public native void TransferParamsToChild(CvVSModule pM, @Cast("char*") BytePointer prefix/*=NULL*/);
    public native void TransferParamsToChild(CvVSModule pM);
    public native void TransferParamsToChild(CvVSModule pM, @Cast("char*") ByteBuffer prefix/*=NULL*/);
    public native void TransferParamsToChild(CvVSModule pM, @Cast("char*") byte[] prefix/*=NULL*/);
    public native void ParamUpdate();
    public native @Cast("const char*") BytePointer GetTypeName();
    public native int IsModuleTypeName(@Cast("const char*") BytePointer name);
    public native int IsModuleTypeName(String name);
    public native @Cast("char*") BytePointer GetModuleName();
    public native int IsModuleName(@Cast("const char*") BytePointer name);
    public native int IsModuleName(String name);
    public native void SetNickName(@Cast("const char*") BytePointer pStr);
    public native void SetNickName(String pStr);
    public native @Cast("const char*") BytePointer GetNickName();
    public native void SaveState(CvFileStorage arg0);
    public native void LoadState(CvFileStorage arg0, CvFileNode arg1);

    public native void Release();
}/* CvVMModule */

public static native void cvWriteStruct(CvFileStorage fs, @Cast("const char*") BytePointer name, Pointer addr, @Cast("const char*") BytePointer desc, int num/*=1*/);
public static native void cvWriteStruct(CvFileStorage fs, @Cast("const char*") BytePointer name, Pointer addr, @Cast("const char*") BytePointer desc);
public static native void cvWriteStruct(CvFileStorage fs, String name, Pointer addr, String desc, int num/*=1*/);
public static native void cvWriteStruct(CvFileStorage fs, String name, Pointer addr, String desc);
public static native void cvReadStructByName(CvFileStorage fs, CvFileNode node, @Cast("const char*") BytePointer name, Pointer addr, @Cast("const char*") BytePointer desc);
public static native void cvReadStructByName(CvFileStorage fs, CvFileNode node, String name, Pointer addr, String desc);

/* FOREGROUND DETECTOR INTERFACE */
public static class CvFGDetector extends CvVSModule {
    static { Loader.load(); }
    /** Empty constructor. */
    public CvFGDetector() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvFGDetector(Pointer p) { super(p); }

    public native IplImage GetMask();
    /* Process current image: */
    public native void Process(IplImage pImg);
    /* Release foreground detector: */
    public native void Release();
}

public static native void cvReleaseFGDetector(@Cast("CvFGDetector**") PointerPointer ppT );
public static native void cvReleaseFGDetector(@ByPtrPtr CvFGDetector ppT );
public static native CvFGDetector cvCreateFGDetectorBase(int type, Pointer param);


/* BLOB STRUCTURE*/
public static class CvBlob extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvBlob() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvBlob(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvBlob(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvBlob position(int position) {
        return (CvBlob)super.position(position);
    }

    public native float x(); public native CvBlob x(float x);
    public native float y(); public native CvBlob y(float y); /* blob position   */
    public native float w(); public native CvBlob w(float w);
    public native float h(); public native CvBlob h(float h); /* blob sizes      */
    public native int ID(); public native CvBlob ID(int ID);  /* blob ID         */
}

public static native @ByVal CvBlob cvBlob(float x,float y, float w, float h);
public static final int CV_BLOB_MINW = 5;
public static final int CV_BLOB_MINH = 5;
// #define CV_BLOB_ID(pB) (((CvBlob*)(pB))->ID)
// #define CV_BLOB_CENTER(pB) cvPoint2D32f(((CvBlob*)(pB))->x,((CvBlob*)(pB))->y)
// #define CV_BLOB_X(pB) (((CvBlob*)(pB))->x)
// #define CV_BLOB_Y(pB) (((CvBlob*)(pB))->y)
// #define CV_BLOB_WX(pB) (((CvBlob*)(pB))->w)
// #define CV_BLOB_WY(pB) (((CvBlob*)(pB))->h)
// #define CV_BLOB_RX(pB) (0.5f*CV_BLOB_WX(pB))
// #define CV_BLOB_RY(pB) (0.5f*CV_BLOB_WY(pB))
// #define CV_BLOB_RECT(pB) cvRect(cvRound(((CvBlob*)(pB))->x-CV_BLOB_RX(pB)),cvRound(((CvBlob*)(pB))->y-CV_BLOB_RY(pB)),cvRound(CV_BLOB_WX(pB)),cvRound(CV_BLOB_WY(pB)))
/* END BLOB STRUCTURE*/


/* simple BLOBLIST */
@NoOffset public static class CvBlobSeq extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvBlobSeq(Pointer p) { super(p); }

    public CvBlobSeq(int BlobSize/*=sizeof(CvBlob)*/) { allocate(BlobSize); }
    private native void allocate(int BlobSize/*=sizeof(CvBlob)*/);
    public CvBlobSeq() { allocate(); }
    private native void allocate();
    public native CvBlob GetBlob(int BlobIndex);
    public native CvBlob GetBlobByID(int BlobID);
    public native void DelBlob(int BlobIndex);
    public native void DelBlobByID(int BlobID);
    public native void Clear();
    public native void AddBlob(CvBlob pB);
    public native int GetBlobNum();
    public native void Write(CvFileStorage fs, @Cast("const char*") BytePointer name);
    public native void Write(CvFileStorage fs, String name);
    public native void Load(CvFileStorage fs, CvFileNode node);
    public native void AddFormat(@Cast("const char*") BytePointer str);
    public native void AddFormat(String str);
}
/* simple BLOBLIST */


/* simple TRACKLIST */
public static class CvBlobTrack extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvBlobTrack() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvBlobTrack(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvBlobTrack(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvBlobTrack position(int position) {
        return (CvBlobTrack)super.position(position);
    }

    public native int TrackID(); public native CvBlobTrack TrackID(int TrackID);
    public native int StartFrame(); public native CvBlobTrack StartFrame(int StartFrame);
    public native CvBlobSeq pBlobSeq(); public native CvBlobTrack pBlobSeq(CvBlobSeq pBlobSeq);
}

@NoOffset public static class CvBlobTrackSeq extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvBlobTrackSeq(Pointer p) { super(p); }

    public CvBlobTrackSeq(int TrackSize/*=sizeof(CvBlobTrack)*/) { allocate(TrackSize); }
    private native void allocate(int TrackSize/*=sizeof(CvBlobTrack)*/);
    public CvBlobTrackSeq() { allocate(); }
    private native void allocate();
    public native CvBlobTrack GetBlobTrack(int TrackIndex);
    public native CvBlobTrack GetBlobTrackByID(int TrackID);
    public native void DelBlobTrack(int TrackIndex);
    public native void DelBlobTrackByID(int TrackID);
    public native void Clear();
    public native void AddBlobTrack(int TrackID, int StartFrame/*=0*/);
    public native void AddBlobTrack(int TrackID);
    public native int GetBlobTrackNum();
}

/* simple TRACKLIST */


/* BLOB DETECTOR INTERFACE */
public static class CvBlobDetector extends CvVSModule {
    static { Loader.load(); }
    /** Empty constructor. */
    public CvBlobDetector() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvBlobDetector(Pointer p) { super(p); }

    /* Try to detect new blob entrance based on foreground mask. */
    /* pFGMask - image of foreground mask */
    /* pNewBlob - pointer to CvBlob structure which will be filled if new blob entrance detected */
    /* pOldBlobList - pointer to blob list which already exist on image */
    public native int DetectNewBlob(IplImage pImg, IplImage pImgFG, CvBlobSeq pNewBlobList, CvBlobSeq pOldBlobList);
    /* release blob detector */
    public native void Release();
}

/* Release any blob detector: */
public static native void cvReleaseBlobDetector(@Cast("CvBlobDetector**") PointerPointer ppBD);
public static native void cvReleaseBlobDetector(@ByPtrPtr CvBlobDetector ppBD);

/* Declarations of constructors of implemented modules: */
public static native CvBlobDetector cvCreateBlobDetectorSimple();
public static native CvBlobDetector cvCreateBlobDetectorCC();

@NoOffset public static class CvDetectedBlob extends CvBlob {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvDetectedBlob() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvDetectedBlob(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvDetectedBlob(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvDetectedBlob position(int position) {
        return (CvDetectedBlob)super.position(position);
    }

    public native float response(); public native CvDetectedBlob response(float response);
}

public static native @ByVal CvDetectedBlob cvDetectedBlob( float x, float y, float w, float h, int ID/*=0*/, float response/*=0.0F*/ );
public static native @ByVal CvDetectedBlob cvDetectedBlob( float x, float y, float w, float h );


@NoOffset public static class CvObjectDetector extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvObjectDetector(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvObjectDetector(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CvObjectDetector position(int position) {
        return (CvObjectDetector)super.position(position);
    }

    public CvObjectDetector( @Cast("const char*") BytePointer arg0/*=0*/ ) { allocate(arg0); }
    private native void allocate( @Cast("const char*") BytePointer arg0/*=0*/ );
    public CvObjectDetector( ) { allocate(); }
    private native void allocate( );
    public CvObjectDetector( String arg0/*=0*/ ) { allocate(arg0); }
    private native void allocate( String arg0/*=0*/ );

    /*
     * Release the current detector and load new detector from file
     * (if detector_file_name is not 0)
     * Return true on success:
     */
    public native @Cast("bool") boolean Load( @Cast("const char*") BytePointer arg0/*=0*/ );
    public native @Cast("bool") boolean Load( );
    public native @Cast("bool") boolean Load( String arg0/*=0*/ );

    /* Return min detector window size: */
    public native @ByVal CvSize GetMinWindowSize();

    /* Return max border: */
    public native int GetMaxBorderSize();

    /*
     * Detect the object on the image and push the detected
     * blobs into <detected_blob_seq> which must be the sequence of <CvDetectedBlob>s
     */
    public native void Detect( @Const CvArr arg0, CvBlobSeq arg1/*=0*/ );
    public native void Detect( @Const CvArr arg0 );
}


public static native @ByVal CvRect cvRectIntersection( @Const @ByVal CvRect r1, @Const @ByVal CvRect r2 );


/*
 * CvImageDrawer
 *
 * Draw on an image the specified ROIs from the source image and
 * given blobs as ellipses or rectangles:
 */

public static class CvDrawShape extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvDrawShape() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvDrawShape(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvDrawShape(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvDrawShape position(int position) {
        return (CvDrawShape)super.position(position);
    }

    /** enum CvDrawShape::shape */
    public static final int RECT = 0, ELLIPSE = 1;
    public native @ByRef CvScalar color(); public native CvDrawShape color(CvScalar color);
}

/*extern const CvDrawShape icv_shape[] =
{
    { CvDrawShape::ELLIPSE, CV_RGB(255,0,0) },
    { CvDrawShape::ELLIPSE, CV_RGB(0,255,0) },
    { CvDrawShape::ELLIPSE, CV_RGB(0,0,255) },
    { CvDrawShape::ELLIPSE, CV_RGB(255,255,0) },
    { CvDrawShape::ELLIPSE, CV_RGB(0,255,255) },
    { CvDrawShape::ELLIPSE, CV_RGB(255,0,255) }
};*/



/* Trajectory generation module: */
public static class CvBlobTrackGen extends CvVSModule {
    static { Loader.load(); }
    /** Empty constructor. */
    public CvBlobTrackGen() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvBlobTrackGen(Pointer p) { super(p); }

    public native void SetFileName(@Cast("char*") BytePointer pFileName);
    public native void SetFileName(@Cast("char*") ByteBuffer pFileName);
    public native void SetFileName(@Cast("char*") byte[] pFileName);
    public native void AddBlob(CvBlob pBlob);
    public native void Process(IplImage pImg/*=NULL*/, IplImage pFG/*=NULL*/);
    public native void Process();
    public native void Release();
}

public static native void cvReleaseBlobTrackGen(@Cast("CvBlobTrackGen**") PointerPointer pBTGen);
public static native void cvReleaseBlobTrackGen(@ByPtrPtr CvBlobTrackGen pBTGen);

/* Declarations of constructors of implemented modules: */
public static native CvBlobTrackGen cvCreateModuleBlobTrackGen1();
public static native CvBlobTrackGen cvCreateModuleBlobTrackGenYML();



/* BLOB TRACKER INTERFACE */
public static class CvBlobTracker extends CvVSModule {
    static { Loader.load(); }
    /** Empty constructor. */
    public CvBlobTracker() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvBlobTracker(Pointer p) { super(p); }


    /* Add new blob to track it and assign to this blob personal ID */
    /* pBlob - pointer to structure with blob parameters (ID is ignored)*/
    /* pImg - current image */
    /* pImgFG - current foreground mask */
    /* Return pointer to new added blob: */
    public native CvBlob AddBlob(CvBlob pBlob, IplImage pImg, IplImage pImgFG/*=NULL*/ );
    public native CvBlob AddBlob(CvBlob pBlob, IplImage pImg );

    /* Return number of currently tracked blobs: */
    public native int GetBlobNum();

    /* Return pointer to specified by index blob: */
    public native CvBlob GetBlob(int BlobIndex);

    /* Delete blob by its index: */
    public native void DelBlob(int BlobIndex);

    /* Process current image and track all existed blobs: */
    public native void Process(IplImage pImg, IplImage pImgFG/*=NULL*/);
    public native void Process(IplImage pImg);

    /* Release blob tracker: */
    public native void Release();


    /* Process one blob (for multi hypothesis tracing): */
    public native void ProcessBlob(int BlobIndex, CvBlob pBlob, IplImage arg2, IplImage arg3/*=NULL*/);
    public native void ProcessBlob(int BlobIndex, CvBlob pBlob, IplImage arg2);

    /* Get confidence/wieght/probability (0-1) for blob: */
    public native double GetConfidence(int arg0, CvBlob arg1, IplImage arg2, IplImage arg3/*=NULL*/);
    public native double GetConfidence(int arg0, CvBlob arg1, IplImage arg2);

    public native double GetConfidenceList(CvBlobSeq pBlobList, IplImage pImg, IplImage pImgFG/*=NULL*/);
    public native double GetConfidenceList(CvBlobSeq pBlobList, IplImage pImg);

    public native void UpdateBlob(int arg0, CvBlob arg1, IplImage arg2, IplImage arg3/*=NULL*/);
    public native void UpdateBlob(int arg0, CvBlob arg1, IplImage arg2);

    /* Update all blob models: */
    public native void Update(IplImage pImg, IplImage pImgFG/*=NULL*/);
    public native void Update(IplImage pImg);

    /* Return pointer to blob by its unique ID: */
    public native int GetBlobIndexByID(int BlobID);

    /* Return pointer to blob by its unique ID: */
    public native CvBlob GetBlobByID(int BlobID);

    /* Delete blob by its ID: */
    public native void DelBlobByID(int BlobID);

    /* Set new parameters for specified (by index) blob: */
    public native void SetBlob(int arg0, CvBlob arg1);

    /* Set new parameters for specified (by ID) blob: */
    public native void SetBlobByID(int BlobID, CvBlob pBlob);

    /*  ===============  MULTI HYPOTHESIS INTERFACE ==================  */

    /* Return number of position hyposetis of currently tracked blob: */
    public native int GetBlobHypNum(int arg0);

    /* Return pointer to specified blob hypothesis by index blob: */
    public native CvBlob GetBlobHyp(int BlobIndex, int arg1);

    /* Set new parameters for specified (by index) blob hyp
     * (can be called several times for each hyp ):
     */
    public native void SetBlobHyp(int arg0, CvBlob arg1);
}

public static native void cvReleaseBlobTracker(@Cast("CvBlobTracker**") PointerPointer ppT );
public static native void cvReleaseBlobTracker(@ByPtrPtr CvBlobTracker ppT );
/* BLOB TRACKER INTERFACE */

/*BLOB TRACKER ONE INTERFACE */
public static class CvBlobTrackerOne extends CvVSModule {
    static { Loader.load(); }
    /** Empty constructor. */
    public CvBlobTrackerOne() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvBlobTrackerOne(Pointer p) { super(p); }

    public native void Init(CvBlob pBlobInit, IplImage pImg, IplImage pImgFG/*=NULL*/);
    public native void Init(CvBlob pBlobInit, IplImage pImg);
    public native CvBlob Process(CvBlob pBlobPrev, IplImage pImg, IplImage pImgFG/*=NULL*/);
    public native CvBlob Process(CvBlob pBlobPrev, IplImage pImg);
    public native void Release();

    /* Non-required methods: */
    public native void SkipProcess(CvBlob arg0, IplImage arg1, IplImage arg2/*=NULL*/);
    public native void SkipProcess(CvBlob arg0, IplImage arg1);
    public native void Update(CvBlob arg0, IplImage arg1, IplImage arg2/*=NULL*/);
    public native void Update(CvBlob arg0, IplImage arg1);
    public native void SetCollision(int arg0); /* call in case of blob collision situation*/
    public native double GetConfidence(CvBlob arg0, IplImage arg1,
                                     IplImage arg2/*=NULL*/, IplImage arg3/*=NULL*/);
    public native double GetConfidence(CvBlob arg0, IplImage arg1);
}
public static native void cvReleaseBlobTrackerOne(@Cast("CvBlobTrackerOne**") PointerPointer ppT );
public static native void cvReleaseBlobTrackerOne(@ByPtrPtr CvBlobTrackerOne ppT );
public static class CvBlobTrackerOne_Create extends FunctionPointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public    CvBlobTrackerOne_Create(Pointer p) { super(p); }
    protected CvBlobTrackerOne_Create() { allocate(); }
    private native void allocate();
    public native CvBlobTrackerOne call();
}
public static native CvBlobTracker cvCreateBlobTrackerList(CvBlobTrackerOne_Create create);
/*BLOB TRACKER ONE INTERFACE */

/* Declarations of constructors of implemented modules: */

/* Some declarations for specific MeanShift tracker: */
public static final int PROFILE_EPANECHNIKOV =    0;
public static final int PROFILE_DOG =             1;
public static class CvBlobTrackerParamMS extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvBlobTrackerParamMS() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvBlobTrackerParamMS(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvBlobTrackerParamMS(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvBlobTrackerParamMS position(int position) {
        return (CvBlobTrackerParamMS)super.position(position);
    }

    public native int noOfSigBits(); public native CvBlobTrackerParamMS noOfSigBits(int noOfSigBits);
    public native int appearance_profile(); public native CvBlobTrackerParamMS appearance_profile(int appearance_profile);
    public native int meanshift_profile(); public native CvBlobTrackerParamMS meanshift_profile(int meanshift_profile);
    public native float sigma(); public native CvBlobTrackerParamMS sigma(float sigma);
}





/* Some declarations for specific Likelihood tracker: */
public static class CvBlobTrackerParamLH extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvBlobTrackerParamLH() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvBlobTrackerParamLH(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvBlobTrackerParamLH(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvBlobTrackerParamLH position(int position) {
        return (CvBlobTrackerParamLH)super.position(position);
    }

    public native int HistType(); public native CvBlobTrackerParamLH HistType(int HistType); /* see Prob.h */
    public native int ScaleAfter(); public native CvBlobTrackerParamLH ScaleAfter(int ScaleAfter);
}

/* Without scale optimization: */


/* With scale optimization: */


/* Simple blob tracker based on connected component tracking: */
public static native CvBlobTracker cvCreateBlobTrackerCC();

/* Connected component tracking and mean-shift particle filter collion-resolver: */
public static native CvBlobTracker cvCreateBlobTrackerCCMSPF();

/* Blob tracker that integrates meanshift and connected components: */
public static native CvBlobTracker cvCreateBlobTrackerMSFG();
public static native CvBlobTracker cvCreateBlobTrackerMSFGS();

/* Meanshift without connected-components */
public static native CvBlobTracker cvCreateBlobTrackerMS();

/* Particle filtering via Bhattacharya coefficient, which        */
/* is roughly the dot-product of two probability densities.      */
/* See: Real-Time Tracking of Non-Rigid Objects using Mean Shift */
/*      Comanicius, Ramesh, Meer, 2000, 8p                       */
/*      http://citeseer.ist.psu.edu/321441.html                  */
public static native CvBlobTracker cvCreateBlobTrackerMSPF();

/* =========== tracker integrators trackers =============*/

/* Integrator based on Particle Filtering method: */
//CV_EXPORTS CvBlobTracker* cvCreateBlobTrackerIPF();

/* Rule based integrator: */
//CV_EXPORTS CvBlobTracker* cvCreateBlobTrackerIRB();

/* Integrator based on data fusion using particle filtering: */
//CV_EXPORTS CvBlobTracker* cvCreateBlobTrackerIPFDF();




/* Trajectory postprocessing module: */
public static class CvBlobTrackPostProc extends CvVSModule {
    static { Loader.load(); }
    /** Empty constructor. */
    public CvBlobTrackPostProc() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvBlobTrackPostProc(Pointer p) { super(p); }

    public native void AddBlob(CvBlob pBlob);
    public native void Process();
    public native int GetBlobNum();
    public native CvBlob GetBlob(int index);
    public native void Release();

    /* Additional functionality: */
    public native CvBlob GetBlobByID(int BlobID);
}

public static native void cvReleaseBlobTrackPostProc(@Cast("CvBlobTrackPostProc**") PointerPointer pBTPP);
public static native void cvReleaseBlobTrackPostProc(@ByPtrPtr CvBlobTrackPostProc pBTPP);

/* Trajectory generation module: */
public static class CvBlobTrackPostProcOne extends CvVSModule {
    static { Loader.load(); }
    /** Empty constructor. */
    public CvBlobTrackPostProcOne() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvBlobTrackPostProcOne(Pointer p) { super(p); }

    public native CvBlob Process(CvBlob pBlob);
    public native void Release();
}

/* Create blob tracking post processing module based on simle module: */
public static class CvBlobTrackPostProcOne_Create extends FunctionPointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public    CvBlobTrackPostProcOne_Create(Pointer p) { super(p); }
    protected CvBlobTrackPostProcOne_Create() { allocate(); }
    private native void allocate();
    public native CvBlobTrackPostProcOne call();
}
public static native CvBlobTrackPostProc cvCreateBlobTrackPostProcList(CvBlobTrackPostProcOne_Create create);


/* Declarations of constructors of implemented modules: */
public static native CvBlobTrackPostProc cvCreateModuleBlobTrackPostProcKalman();
public static native CvBlobTrackPostProc cvCreateModuleBlobTrackPostProcTimeAverRect();
public static native CvBlobTrackPostProc cvCreateModuleBlobTrackPostProcTimeAverExp();


/* PREDICTORS */
/* blob PREDICTOR */
public static class CvBlobTrackPredictor extends CvVSModule {
    static { Loader.load(); }
    /** Empty constructor. */
    public CvBlobTrackPredictor() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvBlobTrackPredictor(Pointer p) { super(p); }

    public native CvBlob Predict();
    public native void Update(CvBlob pBlob);
    public native void Release();
}
public static native CvBlobTrackPredictor cvCreateModuleBlobTrackPredictKalman();



/* Trajectory analyser module: */
public static class CvBlobTrackAnalysis extends CvVSModule {
    static { Loader.load(); }
    /** Empty constructor. */
    public CvBlobTrackAnalysis() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvBlobTrackAnalysis(Pointer p) { super(p); }

    public native void AddBlob(CvBlob pBlob);
    public native void Process(IplImage pImg, IplImage pFG);
    public native float GetState(int BlobID);
    /* return 0 if trajectory is normal
       return >0 if trajectory abnormal */
    public native @Cast("const char*") BytePointer GetStateDesc(int arg0);
    public native void SetFileName(@Cast("char*") BytePointer arg0);
    public native void SetFileName(@Cast("char*") ByteBuffer arg0);
    public native void SetFileName(@Cast("char*") byte[] arg0);
    public native void Release();
}


public static native void cvReleaseBlobTrackAnalysis(@Cast("CvBlobTrackAnalysis**") PointerPointer pBTPP);
public static native void cvReleaseBlobTrackAnalysis(@ByPtrPtr CvBlobTrackAnalysis pBTPP);

/* Feature-vector generation module: */
public static class CvBlobTrackFVGen extends CvVSModule {
    static { Loader.load(); }
    /** Empty constructor. */
    public CvBlobTrackFVGen() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvBlobTrackFVGen(Pointer p) { super(p); }

    public native void AddBlob(CvBlob pBlob);
    public native void Process(IplImage pImg, IplImage pFG);
    public native void Release();
    public native int GetFVSize();
    public native int GetFVNum();
    public native FloatPointer GetFV(int index, IntPointer pFVID);
    public native FloatBuffer GetFV(int index, IntBuffer pFVID);
    public native float[] GetFV(int index, int[] pFVID); /* Returns pointer to FV, if return 0 then FV not created */
    public native FloatPointer GetFVVar(); /* Returns pointer to array of variation of values of FV, if returns 0 then FVVar does not exist. */
    public native FloatPointer GetFVMin(); /* Returns pointer to array of minimal values of FV, if returns 0 then FVrange does not exist */
    public native FloatPointer GetFVMax(); /* Returns pointer to array of maximal values of FV, if returns 0 then FVrange does not exist */
}


/* Trajectory Analyser module: */
public static class CvBlobTrackAnalysisOne extends Pointer {
    static { Loader.load(); }
    /** Empty constructor. */
    public CvBlobTrackAnalysisOne() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvBlobTrackAnalysisOne(Pointer p) { super(p); }

    public native int Process(CvBlob pBlob, IplImage pImg, IplImage pFG);
    /* return 0 if trajectory is normal
       return >0 if trajectory abnormal */
    public native void Release();
}

/* Create blob tracking post processing module based on simle module: */
public static class CvBlobTrackAnalysisOne_Create extends FunctionPointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public    CvBlobTrackAnalysisOne_Create(Pointer p) { super(p); }
    protected CvBlobTrackAnalysisOne_Create() { allocate(); }
    private native void allocate();
    public native CvBlobTrackAnalysisOne call();
}
public static native CvBlobTrackAnalysis cvCreateBlobTrackAnalysisList(CvBlobTrackAnalysisOne_Create create);

/* Declarations of constructors of implemented modules: */

/* Based on histogram analysis of 2D FV (x,y): */
public static native CvBlobTrackAnalysis cvCreateModuleBlobTrackAnalysisHistP();

/* Based on histogram analysis of 4D FV (x,y,vx,vy): */
public static native CvBlobTrackAnalysis cvCreateModuleBlobTrackAnalysisHistPV();

/* Based on histogram analysis of 5D FV (x,y,vx,vy,state): */
public static native CvBlobTrackAnalysis cvCreateModuleBlobTrackAnalysisHistPVS();

/* Based on histogram analysis of 4D FV (startpos,stoppos): */
public static native CvBlobTrackAnalysis cvCreateModuleBlobTrackAnalysisHistSS();



/* Based on SVM classifier analysis of 2D FV (x,y): */
//CV_EXPORTS CvBlobTrackAnalysis* cvCreateModuleBlobTrackAnalysisSVMP();

/* Based on SVM classifier analysis of 4D FV (x,y,vx,vy): */
//CV_EXPORTS CvBlobTrackAnalysis* cvCreateModuleBlobTrackAnalysisSVMPV();

/* Based on SVM classifier analysis of 5D FV (x,y,vx,vy,state): */
//CV_EXPORTS CvBlobTrackAnalysis* cvCreateModuleBlobTrackAnalysisSVMPVS();

/* Based on SVM classifier analysis of 4D FV (startpos,stoppos): */
//CV_EXPORTS CvBlobTrackAnalysis* cvCreateModuleBlobTrackAnalysisSVMSS();

/* Track analysis based on distance between tracks: */
public static native CvBlobTrackAnalysis cvCreateModuleBlobTrackAnalysisTrackDist();

/* Analyzer based on reation Road and height map: */
//CV_EXPORTS CvBlobTrackAnalysis* cvCreateModuleBlobTrackAnalysis3DRoadMap();

/* Analyzer that makes OR decision using set of analyzers: */
public static native CvBlobTrackAnalysis cvCreateModuleBlobTrackAnalysisIOR();

/* Estimator of human height: */
public static class CvBlobTrackAnalysisHeight extends CvBlobTrackAnalysis {
    static { Loader.load(); }
    /** Empty constructor. */
    public CvBlobTrackAnalysisHeight() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvBlobTrackAnalysisHeight(Pointer p) { super(p); }

    public native double GetHeight(CvBlob pB);
}
//CV_EXPORTS CvBlobTrackAnalysisHeight* cvCreateModuleBlobTrackAnalysisHeightScale();



/* AUTO BLOB TRACKER INTERFACE -- pipeline of 3 modules: */
public static class CvBlobTrackerAuto extends CvVSModule {
    static { Loader.load(); }
    /** Empty constructor. */
    public CvBlobTrackerAuto() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvBlobTrackerAuto(Pointer p) { super(p); }

    public native void Process(IplImage pImg, IplImage pMask/*=NULL*/);
    public native void Process(IplImage pImg);
    public native CvBlob GetBlob(int index);
    public native CvBlob GetBlobByID(int ID);
    public native int GetBlobNum();
    public native IplImage GetFGMask();
    public native float GetState(int BlobID);
    public native @Cast("const char*") BytePointer GetStateDesc(int BlobID);
    /* return 0 if trajectory is normal;
     * return >0 if trajectory abnormal. */
    public native void Release();
}
public static native void cvReleaseBlobTrackerAuto(@Cast("CvBlobTrackerAuto**") PointerPointer ppT);
public static native void cvReleaseBlobTrackerAuto(@ByPtrPtr CvBlobTrackerAuto ppT);
/* END AUTO BLOB TRACKER INTERFACE */


/* Constructor functions and data for specific BlobTRackerAuto modules: */

/* Parameters of blobtracker auto ver1: */
public static class CvBlobTrackerAutoParam1 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvBlobTrackerAutoParam1() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvBlobTrackerAutoParam1(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvBlobTrackerAutoParam1(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvBlobTrackerAutoParam1 position(int position) {
        return (CvBlobTrackerAutoParam1)super.position(position);
    }

    public native int FGTrainFrames(); public native CvBlobTrackerAutoParam1 FGTrainFrames(int FGTrainFrames); /* Number of frames needed for FG (foreground) detector to train.        */

    public native CvFGDetector pFG(); public native CvBlobTrackerAutoParam1 pFG(CvFGDetector pFG);           /* FGDetector module. If this field is NULL the Process FG mask is used. */

    public native CvBlobDetector pBD(); public native CvBlobTrackerAutoParam1 pBD(CvBlobDetector pBD);           /* Selected blob detector module. 					    */
                                           /* If this field is NULL default blobdetector module will be created.    */

    public native CvBlobTracker pBT(); public native CvBlobTrackerAutoParam1 pBT(CvBlobTracker pBT);           /* Selected blob tracking module.					    */
                                           /* If this field is NULL default blobtracker module will be created.     */

    public native CvBlobTrackGen pBTGen(); public native CvBlobTrackerAutoParam1 pBTGen(CvBlobTrackGen pBTGen);        /* Selected blob trajectory generator.				    */
                                           /* If this field is NULL no generator is used.                           */

    public native CvBlobTrackPostProc pBTPP(); public native CvBlobTrackerAutoParam1 pBTPP(CvBlobTrackPostProc pBTPP);         /* Selected blob trajectory postprocessing module.			    */
                                           /* If this field is NULL no postprocessing is done.                      */

    public native int UsePPData(); public native CvBlobTrackerAutoParam1 UsePPData(int UsePPData);

    public native CvBlobTrackAnalysis pBTA(); public native CvBlobTrackerAutoParam1 pBTA(CvBlobTrackAnalysis pBTA);          /* Selected blob trajectory analysis module.                             */
                                           /* If this field is NULL no track analysis is done.                      */
}

/* Create blob tracker auto ver1: */
public static native CvBlobTrackerAuto cvCreateBlobTrackerAuto1(CvBlobTrackerAutoParam1 param/*=NULL*/);
public static native CvBlobTrackerAuto cvCreateBlobTrackerAuto1();

/* Simple loader for many auto trackers by its type : */
public static native CvBlobTrackerAuto cvCreateBlobTrackerAuto(int type, Pointer param);



public static class CvTracksTimePos extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvTracksTimePos() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvTracksTimePos(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvTracksTimePos(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvTracksTimePos position(int position) {
        return (CvTracksTimePos)super.position(position);
    }

    public native int len1(); public native CvTracksTimePos len1(int len1);
    public native int len2(); public native CvTracksTimePos len2(int len2);
    public native int beg1(); public native CvTracksTimePos beg1(int beg1);
    public native int beg2(); public native CvTracksTimePos beg2(int beg2);
    public native int end1(); public native CvTracksTimePos end1(int end1);
    public native int end2(); public native CvTracksTimePos end2(int end2);
    public native int comLen(); public native CvTracksTimePos comLen(int comLen); //common length for two tracks
    public native int shift1(); public native CvTracksTimePos shift1(int shift1);
    public native int shift2(); public native CvTracksTimePos shift2(int shift2);
}

/*CV_EXPORTS int cvCompareTracks( CvBlobTrackSeq *groundTruth,
                   CvBlobTrackSeq *result,
                   FILE *file);*/


/* Constructor functions:  */






/* HIST API */
public static class CvProb extends Pointer {
    static { Loader.load(); }
    /** Empty constructor. */
    public CvProb() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvProb(Pointer p) { super(p); }


    /* Calculate probability value: */
    public native double Value(IntPointer arg0, int arg1/*=0*/, int arg2/*=0*/);
    public native double Value(IntPointer arg0);
    public native double Value(IntBuffer arg0, int arg1/*=0*/, int arg2/*=0*/);
    public native double Value(IntBuffer arg0);
    public native double Value(int[] arg0, int arg1/*=0*/, int arg2/*=0*/);
    public native double Value(int[] arg0);

    /* Update histograpp Pnew = (1-W)*Pold + W*Padd*/
    /* W weight of new added prob */
    /* comps - matrix of new fetature vectors used to update prob */
    public native void AddFeature(float W, IntPointer comps, int x/*=0*/, int y/*=0*/);
    public native void AddFeature(float W, IntPointer comps);
    public native void AddFeature(float W, IntBuffer comps, int x/*=0*/, int y/*=0*/);
    public native void AddFeature(float W, IntBuffer comps);
    public native void AddFeature(float W, int[] comps, int x/*=0*/, int y/*=0*/);
    public native void AddFeature(float W, int[] comps);
    public native void Scale(float factor/*=0*/, int x/*=-1*/, int y/*=-1*/);
    public native void Scale();
    public native void Release();
}
public static native void cvReleaseProb(@Cast("CvProb**") PointerPointer ppProb);
public static native void cvReleaseProb(@ByPtrPtr CvProb ppProb);
/* HIST API */

/* Some Prob: */





public static final int CV_BT_HIST_TYPE_S =     0;
public static final int CV_BT_HIST_TYPE_MG =    1;
public static final int CV_BT_HIST_TYPE_MG2 =   2;
public static final int CV_BT_HIST_TYPE_H =     3;




/* Noise type definitions: */
public static final int CV_NOISE_NONE =               0;
public static final int CV_NOISE_GAUSSIAN =           1;
public static final int CV_NOISE_UNIFORM =            2;
public static final int CV_NOISE_SPECKLE =            3;
public static final int CV_NOISE_SALT_AND_PEPPER =    4;

/* Add some noise to image: */
/* pImg - (input) image without noise */
/* pImg - (output) image with noise */
/* noise_type - type of added noise */
/*  CV_NOISE_GAUSSIAN - pImg += n , n - is gaussian noise with Ampl standart deviation */
/*  CV_NOISE_UNIFORM - pImg += n , n - is uniform noise with Ampl standart deviation */
/*  CV_NOISE_SPECKLE - pImg += n*pImg , n - is gaussian noise with Ampl standart deviation */
/*  CV_NOISE_SALT_AND_PAPPER - pImg = pImg with blacked and whited pixels,
            Ampl is density of brocken pixels (0-there are not broken pixels, 1 - all pixels are broken)*/
/* Ampl - "amplitude" of noise */
//CV_EXPORTS void cvAddNoise(IplImage* pImg, int noise_type, double Ampl, CvRNG* rnd_state = NULL);

/*================== GENERATOR OF TEST VIDEO SEQUENCE ===================== */
@Opaque public static class CvTestSeq extends Pointer {
    /** Empty constructor. */
    public CvTestSeq() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvTestSeq(Pointer p) { super(p); }
}

/* pConfigfile - Name of file (yml or xml) with description of test sequence */
/* videos - array of names of test videos described in "pConfigfile" file */
/* numvideos - size of "videos" array */
public static native CvTestSeq cvCreateTestSeq(@Cast("char*") BytePointer pConfigfile, @Cast("char**") PointerPointer videos, int numvideo, float Scale/*=1*/, int noise_type/*=CV_NOISE_NONE*/, double noise_ampl/*=0*/);
public static native CvTestSeq cvCreateTestSeq(@Cast("char*") BytePointer pConfigfile, @Cast("char**") @ByPtrPtr BytePointer videos, int numvideo);
public static native CvTestSeq cvCreateTestSeq(@Cast("char*") BytePointer pConfigfile, @Cast("char**") @ByPtrPtr BytePointer videos, int numvideo, float Scale/*=1*/, int noise_type/*=CV_NOISE_NONE*/, double noise_ampl/*=0*/);
public static native CvTestSeq cvCreateTestSeq(@Cast("char*") ByteBuffer pConfigfile, @Cast("char**") @ByPtrPtr ByteBuffer videos, int numvideo, float Scale/*=1*/, int noise_type/*=CV_NOISE_NONE*/, double noise_ampl/*=0*/);
public static native CvTestSeq cvCreateTestSeq(@Cast("char*") ByteBuffer pConfigfile, @Cast("char**") @ByPtrPtr ByteBuffer videos, int numvideo);
public static native CvTestSeq cvCreateTestSeq(@Cast("char*") byte[] pConfigfile, @Cast("char**") @ByPtrPtr byte[] videos, int numvideo, float Scale/*=1*/, int noise_type/*=CV_NOISE_NONE*/, double noise_ampl/*=0*/);
public static native CvTestSeq cvCreateTestSeq(@Cast("char*") byte[] pConfigfile, @Cast("char**") @ByPtrPtr byte[] videos, int numvideo);
public static native void cvReleaseTestSeq(@Cast("CvTestSeq**") PointerPointer ppTestSeq);
public static native void cvReleaseTestSeq(@ByPtrPtr CvTestSeq ppTestSeq);

/* Generate next frame from test video seq and return pointer to it: */
public static native IplImage cvTestSeqQueryFrame(CvTestSeq pTestSeq);

/* Return pointer to current foreground mask: */
public static native IplImage cvTestSeqGetFGMask(CvTestSeq pTestSeq);

/* Return pointer to current image: */
public static native IplImage cvTestSeqGetImage(CvTestSeq pTestSeq);

/* Return frame size of result test video: */
public static native @ByVal CvSize cvTestSeqGetImageSize(CvTestSeq pTestSeq);

/* Return number of frames result test video: */
public static native int cvTestSeqFrameNum(CvTestSeq pTestSeq);

/* Return number of existing objects.
 * This is general number of any objects.
 * For example number of trajectories may be equal or less than returned value:
 */
public static native int cvTestSeqGetObjectNum(CvTestSeq pTestSeq);

/* Return 0 if there is not position for current defined on current frame */
/* Return 1 if there is object position and pPos was filled */
public static native int cvTestSeqGetObjectPos(CvTestSeq pTestSeq, int ObjIndex, CvPoint2D32f pPos);
public static native int cvTestSeqGetObjectPos(CvTestSeq pTestSeq, int ObjIndex, @Cast("CvPoint2D32f*") FloatBuffer pPos);
public static native int cvTestSeqGetObjectPos(CvTestSeq pTestSeq, int ObjIndex, @Cast("CvPoint2D32f*") float[] pPos);
public static native int cvTestSeqGetObjectSize(CvTestSeq pTestSeq, int ObjIndex, CvPoint2D32f pSize);
public static native int cvTestSeqGetObjectSize(CvTestSeq pTestSeq, int ObjIndex, @Cast("CvPoint2D32f*") FloatBuffer pSize);
public static native int cvTestSeqGetObjectSize(CvTestSeq pTestSeq, int ObjIndex, @Cast("CvPoint2D32f*") float[] pSize);

/* Add noise to final image: */
public static native void cvTestSeqAddNoise(CvTestSeq pTestSeq, int noise_type/*=CV_NOISE_NONE*/, double noise_ampl/*=0*/);
public static native void cvTestSeqAddNoise(CvTestSeq pTestSeq);

/* Add Intensity variation: */
public static native void cvTestSeqAddIntensityVariation(CvTestSeq pTestSeq, float DI_per_frame, float MinI, float MaxI);
public static native void cvTestSeqSetFrame(CvTestSeq pTestSeq, int n);

// #endif

/* End of file. */


// Parsed from <opencv2/legacy/compat.hpp>

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
// Copyright( C) 2000, Intel Corporation, all rights reserved.
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
//(including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort(including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

/*
   A few macros and definitions for backward compatibility
   with the previous versions of OpenCV. They are obsolete and
   are likely to be removed in future. To check whether your code
   uses any of these, define CV_NO_BACKWARD_COMPATIBILITY before
   including cv.h.
*/

// #ifndef __OPENCV_COMPAT_HPP__
// #define __OPENCV_COMPAT_HPP__

// #include "opencv2/core/core_c.h"
// #include "opencv2/imgproc/types_c.h"

// #include <math.h>
// #include <string.h>

// #ifdef __cplusplus
// #endif

/** enum  */
public static final int
    CV_MAT32F      =  CV_32FC1,
    CV_MAT3x1_32F  =  CV_32FC1,
    CV_MAT4x1_32F  =  CV_32FC1,
    CV_MAT3x3_32F  =  CV_32FC1,
    CV_MAT4x4_32F  =  CV_32FC1,

    CV_MAT64D      =  CV_64FC1,
    CV_MAT3x1_64D  =  CV_64FC1,
    CV_MAT4x1_64D  =  CV_64FC1,
    CV_MAT3x3_64D  =  CV_64FC1,
    CV_MAT4x4_64D  =  CV_64FC1;

/** enum  */
public static final int
    IPL_GAUSSIAN_5x5 = 7;

/* allocation/deallocation macros */
public static native void cvCreateImageData(CvArr arg1);
public static native void cvReleaseImageData(CvArr arg1);
public static native void cvSetImageData(CvArr arg1, Pointer arg2, int arg3);
public static native void cvGetImageRawData(CvArr arg1, @Cast("uchar**") PointerPointer arg2, IntPointer arg3, CvSize arg4);
public static native void cvGetImageRawData(CvArr arg1, @Cast("uchar**") @ByPtrPtr BytePointer arg2, IntPointer arg3, CvSize arg4);
public static native void cvGetImageRawData(CvArr arg1, @Cast("uchar**") @ByPtrPtr ByteBuffer arg2, IntBuffer arg3, CvSize arg4);
public static native void cvGetImageRawData(CvArr arg1, @Cast("uchar**") @ByPtrPtr byte[] arg2, int[] arg3, CvSize arg4);

public static native void cvmAlloc(CvArr arg1);
public static native void cvmFree(CvArr arg1);
public static native void cvmAllocArray(CvArr arg1);
public static native void cvmFreeArray(CvArr arg1);

public static native void cvIntegralImage(CvArr arg1, CvArr arg2, CvArr arg3, CvArr arg4);
public static native double cvMatchContours(Pointer arg1, Pointer arg2, int arg3, double arg4);

public static native @ByVal CvMat cvMatArray( int rows, int cols, int type,
                            int count, Pointer data/*=0*/);
public static native @ByVal CvMat cvMatArray( int rows, int cols, int type,
                            int count);

public static native void cvUpdateMHIByTime(CvArr arg1, CvArr arg2, double arg3, double arg4);

public static native void cvAccMask(CvArr arg1, CvArr arg2, CvArr arg3);
public static native void cvSquareAccMask(CvArr arg1, CvArr arg2, CvArr arg3);
public static native void cvMultiplyAccMask(CvArr arg1, CvArr arg2, CvArr arg3, CvArr arg4);
public static native void cvRunningAvgMask(CvArr imgY, CvArr imgU, CvArr mask, double alpha);

public static native void cvSetHistThresh(CvHistogram arg1, @Cast("float**") PointerPointer arg2, int arg3);
public static native void cvSetHistThresh(CvHistogram arg1, @ByPtrPtr FloatPointer arg2, int arg3);
public static native void cvSetHistThresh(CvHistogram arg1, @ByPtrPtr FloatBuffer arg2, int arg3);
public static native void cvSetHistThresh(CvHistogram arg1, @ByPtrPtr float[] arg2, int arg3);
public static native void cvCalcHistMask(@Cast("IplImage**") PointerPointer img, CvArr mask, CvHistogram hist, int doNotClear);
public static native void cvCalcHistMask(@ByPtrPtr IplImage img, CvArr mask, CvHistogram hist, int doNotClear);

public static native double cvMean( @Const CvArr image, @Const CvArr mask/*=0*/);
public static native double cvMean( @Const CvArr image);
public static native double cvSumPixels( @Const CvArr image );
public static native void cvMean_StdDev( @Const CvArr image, DoublePointer mean, DoublePointer sdv,
                                @Const CvArr mask/*=0*/);
public static native void cvMean_StdDev( @Const CvArr image, DoublePointer mean, DoublePointer sdv);
public static native void cvMean_StdDev( @Const CvArr image, DoubleBuffer mean, DoubleBuffer sdv,
                                @Const CvArr mask/*=0*/);
public static native void cvMean_StdDev( @Const CvArr image, DoubleBuffer mean, DoubleBuffer sdv);
public static native void cvMean_StdDev( @Const CvArr image, double[] mean, double[] sdv,
                                @Const CvArr mask/*=0*/);
public static native void cvMean_StdDev( @Const CvArr image, double[] mean, double[] sdv);

public static native void cvmPerspectiveProject( @Const CvMat mat, @Const CvArr src, CvArr dst );
public static native void cvFillImage( CvArr mat, double color );

public static native void cvCvtPixToPlane(CvArr arg1, CvArr arg2, CvArr arg3, CvArr arg4, CvArr arg5);
public static native void cvCvtPlaneToPix(CvArr arg1, CvArr arg2, CvArr arg3, CvArr arg4, CvArr arg5);

public static class CvRandState extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvRandState() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvRandState(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvRandState(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvRandState position(int position) {
        return (CvRandState)super.position(position);
    }

    public native @Cast("CvRNG") long state(); public native CvRandState state(long state);    /* RNG state (the current seed and carry)*/
    public native int disttype(); public native CvRandState disttype(int disttype); /* distribution type */
    public native @ByRef CvScalar param(int i); public native CvRandState param(int i, CvScalar param);
    @MemberGetter public native CvScalar param(); /* parameters of RNG */
}

/* Changes RNG range while preserving RNG state */
public static native void cvRandSetRange( CvRandState state, double param1,
                                 double param2, int index/*=-1*/);
public static native void cvRandSetRange( CvRandState state, double param1,
                                 double param2);

public static native void cvRandInit( CvRandState state, double param1,
                             double param2, int seed,
                             int disttype/*=CV_RAND_UNI*/);
public static native void cvRandInit( CvRandState state, double param1,
                             double param2, int seed);

/* Fills array with random numbers */
public static native void cvRand( CvRandState state, CvArr arr );

// #define cvRandNext( _state ) cvRandInt( &(_state)->state )

public static native void cvbRand( CvRandState state, FloatPointer dst, int len );
public static native void cvbRand( CvRandState state, FloatBuffer dst, int len );
public static native void cvbRand( CvRandState state, float[] dst, int len );

public static native void cvbCartToPolar( @Const FloatPointer y, @Const FloatPointer x,
                                 FloatPointer magnitude, FloatPointer angle, int len );
public static native void cvbCartToPolar( @Const FloatBuffer y, @Const FloatBuffer x,
                                 FloatBuffer magnitude, FloatBuffer angle, int len );
public static native void cvbCartToPolar( @Const float[] y, @Const float[] x,
                                 float[] magnitude, float[] angle, int len );
public static native void cvbFastArctan( @Const FloatPointer y, @Const FloatPointer x, FloatPointer angle, int len );
public static native void cvbFastArctan( @Const FloatBuffer y, @Const FloatBuffer x, FloatBuffer angle, int len );
public static native void cvbFastArctan( @Const float[] y, @Const float[] x, float[] angle, int len );
public static native void cvbSqrt( @Const FloatPointer x, FloatPointer y, int len );
public static native void cvbSqrt( @Const FloatBuffer x, FloatBuffer y, int len );
public static native void cvbSqrt( @Const float[] x, float[] y, int len );
public static native void cvbInvSqrt( @Const FloatPointer x, FloatPointer y, int len );
public static native void cvbInvSqrt( @Const FloatBuffer x, FloatBuffer y, int len );
public static native void cvbInvSqrt( @Const float[] x, float[] y, int len );
public static native void cvbReciprocal( @Const FloatPointer x, FloatPointer y, int len );
public static native void cvbReciprocal( @Const FloatBuffer x, FloatBuffer y, int len );
public static native void cvbReciprocal( @Const float[] x, float[] y, int len );
public static native void cvbFastExp( @Const FloatPointer x, DoublePointer y, int len );
public static native void cvbFastExp( @Const FloatBuffer x, DoubleBuffer y, int len );
public static native void cvbFastExp( @Const float[] x, double[] y, int len );
public static native void cvbFastLog( @Const DoublePointer x, FloatPointer y, int len );
public static native void cvbFastLog( @Const DoubleBuffer x, FloatBuffer y, int len );
public static native void cvbFastLog( @Const double[] x, float[] y, int len );

public static native @ByVal CvRect cvContourBoundingRect( Pointer point_set, int update/*=0*/);
public static native @ByVal CvRect cvContourBoundingRect( Pointer point_set);

public static native double cvPseudoInverse( @Const CvArr src, CvArr dst );
public static native double cvPseudoInv(CvArr arg1, CvArr arg2);

public static native void cvContourMoments(CvArr contour, CvMoments moments);

public static native Pointer cvGetPtrAt(CvArr arg1, int arg2, int arg3);
public static native @ByVal CvScalar cvGetAt(CvArr arg1, int arg2, int arg3);
public static native void cvSetAt(CvArr arr, @ByVal CvScalar val, int y, int x);

public static native double cvMeanMask(CvArr arg1, CvArr arg2);
public static native void cvMean_StdDevMask(CvArr img, CvArr mask, DoublePointer mean, DoublePointer sdv);
public static native void cvMean_StdDevMask(CvArr img, CvArr mask, DoubleBuffer mean, DoubleBuffer sdv);
public static native void cvMean_StdDevMask(CvArr img, CvArr mask, double[] mean, double[] sdv);

public static native double cvNormMask(CvArr imgA, CvArr imgB, CvArr mask, int normType);

// #define cvMinMaxLocMask(img, mask, min_val, max_val, min_loc, max_loc)
//         cvMinMaxLoc(img, min_val, max_val, min_loc, max_loc, mask)

public static native void cvRemoveMemoryManager(CvAllocFunc arg1, CvFreeFunc arg2, Pointer arg3);

// #define cvmSetZero( mat )               cvSetZero( mat )
// #define cvmSetIdentity( mat )           cvSetIdentity( mat )
// #define cvmAdd( src1, src2, dst )       cvAdd( src1, src2, dst, 0 )
// #define cvmSub( src1, src2, dst )       cvSub( src1, src2, dst, 0 )
// #define cvmCopy( src, dst )             cvCopy( src, dst, 0 )
// #define cvmMul( src1, src2, dst )       cvMatMulAdd( src1, src2, 0, dst )
// #define cvmTranspose( src, dst )        cvT( src, dst )
// #define cvmInvert( src, dst )           cvInv( src, dst )
// #define cvmMahalanobis(vec1, vec2, mat) cvMahalanobis( vec1, vec2, mat )
// #define cvmDotProduct( vec1, vec2 )     cvDotProduct( vec1, vec2 )
// #define cvmCrossProduct(vec1, vec2,dst) cvCrossProduct( vec1, vec2, dst )
// #define cvmTrace( mat )                 (cvTrace( mat )).val[0]
// #define cvmMulTransposed( src, dst, order ) cvMulTransposed( src, dst, order )
// #define cvmEigenVV( mat, evec, eval, eps)   cvEigenVV( mat, evec, eval, eps )
// #define cvmDet( mat )                   cvDet( mat )
// #define cvmScale( src, dst, scale )     cvScale( src, dst, scale )

public static native void cvCopyImage(CvArr src, CvArr dst);
public static native void cvReleaseMatHeader(@Cast("CvMat**") PointerPointer arg1);
public static native void cvReleaseMatHeader(@ByPtrPtr CvMat arg1);

/* Calculates exact convex hull of 2d point set */
public static native void cvConvexHull( CvPoint points, int num_points,
                             CvRect bound_rect,
                             int orientation, IntPointer hull, IntPointer hullsize );
public static native void cvConvexHull( @Cast("CvPoint*") IntBuffer points, int num_points,
                             CvRect bound_rect,
                             int orientation, IntBuffer hull, IntBuffer hullsize );
public static native void cvConvexHull( @Cast("CvPoint*") int[] points, int num_points,
                             CvRect bound_rect,
                             int orientation, int[] hull, int[] hullsize );


public static native void cvMinAreaRect( CvPoint points, int n,
                              int left, int bottom,
                              int right, int top,
                              CvPoint2D32f anchor,
                              CvPoint2D32f vect1,
                              CvPoint2D32f vect2 );
public static native void cvMinAreaRect( @Cast("CvPoint*") IntBuffer points, int n,
                              int left, int bottom,
                              int right, int top,
                              @Cast("CvPoint2D32f*") FloatBuffer anchor,
                              @Cast("CvPoint2D32f*") FloatBuffer vect1,
                              @Cast("CvPoint2D32f*") FloatBuffer vect2 );
public static native void cvMinAreaRect( @Cast("CvPoint*") int[] points, int n,
                              int left, int bottom,
                              int right, int top,
                              @Cast("CvPoint2D32f*") float[] anchor,
                              @Cast("CvPoint2D32f*") float[] vect1,
                              @Cast("CvPoint2D32f*") float[] vect2 );

public static native void cvFitLine3D( CvPoint3D32f points, int count, int dist,
                    Pointer param, float reps, float aeps, FloatPointer line );
public static native void cvFitLine3D( @Cast("CvPoint3D32f*") FloatBuffer points, int count, int dist,
                    Pointer param, float reps, float aeps, FloatBuffer line );
public static native void cvFitLine3D( @Cast("CvPoint3D32f*") float[] points, int count, int dist,
                    Pointer param, float reps, float aeps, float[] line );

/* Fits a line into set of 2d points in a robust way (M-estimator technique) */
public static native void cvFitLine2D( CvPoint2D32f points, int count, int dist,
                    Pointer param, float reps, float aeps, FloatPointer line );
public static native void cvFitLine2D( @Cast("CvPoint2D32f*") FloatBuffer points, int count, int dist,
                    Pointer param, float reps, float aeps, FloatBuffer line );
public static native void cvFitLine2D( @Cast("CvPoint2D32f*") float[] points, int count, int dist,
                    Pointer param, float reps, float aeps, float[] line );

public static native void cvFitEllipse( @Const CvPoint2D32f points, int count, CvBox2D box );
public static native void cvFitEllipse( @Cast("const CvPoint2D32f*") FloatBuffer points, int count, CvBox2D box );
public static native void cvFitEllipse( @Cast("const CvPoint2D32f*") float[] points, int count, CvBox2D box );

/* Projects 2d points to one of standard coordinate planes
   (i.e. removes one of coordinates) */
public static native void cvProject3D( CvPoint3D32f points3D, int count,
                              CvPoint2D32f points2D,
                              int xIndx/*=0*/,
                              int yIndx/*=1*/);
public static native void cvProject3D( CvPoint3D32f points3D, int count,
                              CvPoint2D32f points2D);
public static native void cvProject3D( @Cast("CvPoint3D32f*") FloatBuffer points3D, int count,
                              @Cast("CvPoint2D32f*") FloatBuffer points2D,
                              int xIndx/*=0*/,
                              int yIndx/*=1*/);
public static native void cvProject3D( @Cast("CvPoint3D32f*") FloatBuffer points3D, int count,
                              @Cast("CvPoint2D32f*") FloatBuffer points2D);
public static native void cvProject3D( @Cast("CvPoint3D32f*") float[] points3D, int count,
                              @Cast("CvPoint2D32f*") float[] points2D,
                              int xIndx/*=0*/,
                              int yIndx/*=1*/);
public static native void cvProject3D( @Cast("CvPoint3D32f*") float[] points3D, int count,
                              @Cast("CvPoint2D32f*") float[] points2D);

/* Retrieves value of the particular bin
   of x-dimensional (x=1,2,3,...) histogram */
public static native float cvQueryHistValue_1D(CvHistogram hist, int idx0);
public static native float cvQueryHistValue_2D(CvHistogram hist, int idx0, int idx1);
public static native float cvQueryHistValue_3D(CvHistogram hist, int idx0, int idx1, int idx2);
public static native float cvQueryHistValue_nD(CvHistogram hist, IntPointer idx);
public static native float cvQueryHistValue_nD(CvHistogram hist, IntBuffer idx);
public static native float cvQueryHistValue_nD(CvHistogram hist, int[] idx);

/* Returns pointer to the particular bin of x-dimesional histogram.
   For sparse histogram the bin is created if it didn't exist before */
public static native Pointer cvGetHistValue_1D(CvHistogram hist, int idx0);
public static native Pointer cvGetHistValue_2D(CvHistogram hist, int idx0, int idx1);
public static native Pointer cvGetHistValue_3D(CvHistogram hist, int idx0, int idx1, int idx2);
public static native Pointer cvGetHistValue_nD(CvHistogram hist, IntPointer idx);
public static native Pointer cvGetHistValue_nD(CvHistogram hist, IntBuffer idx);
public static native Pointer cvGetHistValue_nD(CvHistogram hist, int[] idx);


public static native @Cast("bool") boolean CV_IS_SET_ELEM_EXISTS(CvSetElem arg1);


public static native int cvHoughLines( CvArr image, double rho,
                              double theta, int threshold,
                              FloatPointer lines, int linesNumber );
public static native int cvHoughLines( CvArr image, double rho,
                              double theta, int threshold,
                              FloatBuffer lines, int linesNumber );
public static native int cvHoughLines( CvArr image, double rho,
                              double theta, int threshold,
                              float[] lines, int linesNumber );

public static native int cvHoughLinesP( CvArr image, double rho,
                               double theta, int threshold,
                               int lineLength, int lineGap,
                               IntPointer lines, int linesNumber );
public static native int cvHoughLinesP( CvArr image, double rho,
                               double theta, int threshold,
                               int lineLength, int lineGap,
                               IntBuffer lines, int linesNumber );
public static native int cvHoughLinesP( CvArr image, double rho,
                               double theta, int threshold,
                               int lineLength, int lineGap,
                               int[] lines, int linesNumber );


public static native int cvHoughLinesSDiv( CvArr image, double rho, int srn,
                                  double theta, int stn, int threshold,
                                  FloatPointer lines, int linesNumber );
public static native int cvHoughLinesSDiv( CvArr image, double rho, int srn,
                                  double theta, int stn, int threshold,
                                  FloatBuffer lines, int linesNumber );
public static native int cvHoughLinesSDiv( CvArr image, double rho, int srn,
                                  double theta, int stn, int threshold,
                                  float[] lines, int linesNumber );

public static native float cvCalcEMD( @Const FloatPointer signature1, int size1,
                             @Const FloatPointer signature2, int size2,
                             int dims, int dist_type/*=CV_DIST_L2*/,
                             CvDistanceFunction dist_func/*=0*/,
                             FloatPointer lower_bound/*=0*/,
                             Pointer user_param/*=0*/);
public static native float cvCalcEMD( @Const FloatPointer signature1, int size1,
                             @Const FloatPointer signature2, int size2,
                             int dims);
public static native float cvCalcEMD( @Const FloatBuffer signature1, int size1,
                             @Const FloatBuffer signature2, int size2,
                             int dims, int dist_type/*=CV_DIST_L2*/,
                             CvDistanceFunction dist_func/*=0*/,
                             FloatBuffer lower_bound/*=0*/,
                             Pointer user_param/*=0*/);
public static native float cvCalcEMD( @Const FloatBuffer signature1, int size1,
                             @Const FloatBuffer signature2, int size2,
                             int dims);
public static native float cvCalcEMD( @Const float[] signature1, int size1,
                             @Const float[] signature2, int size2,
                             int dims, int dist_type/*=CV_DIST_L2*/,
                             CvDistanceFunction dist_func/*=0*/,
                             float[] lower_bound/*=0*/,
                             Pointer user_param/*=0*/);
public static native float cvCalcEMD( @Const float[] signature1, int size1,
                             @Const float[] signature2, int size2,
                             int dims);

public static native void cvKMeans( int num_clusters, @Cast("float**") PointerPointer samples,
                           int num_samples, int vec_size,
                           @ByVal CvTermCriteria termcrit, IntPointer cluster_idx );
public static native void cvKMeans( int num_clusters, @ByPtrPtr FloatPointer samples,
                           int num_samples, int vec_size,
                           @ByVal CvTermCriteria termcrit, IntPointer cluster_idx );
public static native void cvKMeans( int num_clusters, @ByPtrPtr FloatBuffer samples,
                           int num_samples, int vec_size,
                           @ByVal CvTermCriteria termcrit, IntBuffer cluster_idx );
public static native void cvKMeans( int num_clusters, @ByPtrPtr float[] samples,
                           int num_samples, int vec_size,
                           @ByVal CvTermCriteria termcrit, int[] cluster_idx );

public static native void cvStartScanGraph( CvGraph graph, CvGraphScanner scanner,
                                  CvGraphVtx vtx/*=NULL*/,
                                  int mask/*=CV_GRAPH_ALL_ITEMS*/);
public static native void cvStartScanGraph( CvGraph graph, CvGraphScanner scanner);

public static native void cvEndScanGraph( CvGraphScanner scanner );


/* old drawing functions */
public static native void cvLineAA( CvArr img, @ByVal CvPoint pt1, @ByVal CvPoint pt2,
                            double color, int scale/*=0*/);
public static native void cvLineAA( CvArr img, @ByVal CvPoint pt1, @ByVal CvPoint pt2,
                            double color);
public static native void cvLineAA( CvArr img, @ByVal @Cast("CvPoint*") IntBuffer pt1, @ByVal @Cast("CvPoint*") IntBuffer pt2,
                            double color, int scale/*=0*/);
public static native void cvLineAA( CvArr img, @ByVal @Cast("CvPoint*") IntBuffer pt1, @ByVal @Cast("CvPoint*") IntBuffer pt2,
                            double color);
public static native void cvLineAA( CvArr img, @ByVal @Cast("CvPoint*") int[] pt1, @ByVal @Cast("CvPoint*") int[] pt2,
                            double color, int scale/*=0*/);
public static native void cvLineAA( CvArr img, @ByVal @Cast("CvPoint*") int[] pt1, @ByVal @Cast("CvPoint*") int[] pt2,
                            double color);

public static native void cvCircleAA( CvArr img, @ByVal CvPoint center, int radius,
                            double color, int scale/*=0*/ );
public static native void cvCircleAA( CvArr img, @ByVal CvPoint center, int radius,
                            double color );
public static native void cvCircleAA( CvArr img, @ByVal @Cast("CvPoint*") IntBuffer center, int radius,
                            double color, int scale/*=0*/ );
public static native void cvCircleAA( CvArr img, @ByVal @Cast("CvPoint*") IntBuffer center, int radius,
                            double color );
public static native void cvCircleAA( CvArr img, @ByVal @Cast("CvPoint*") int[] center, int radius,
                            double color, int scale/*=0*/ );
public static native void cvCircleAA( CvArr img, @ByVal @Cast("CvPoint*") int[] center, int radius,
                            double color );

public static native void cvEllipseAA( CvArr img, @ByVal CvPoint center, @ByVal CvSize axes,
                              double angle, double start_angle,
                              double end_angle, double color,
                              int scale/*=0*/ );
public static native void cvEllipseAA( CvArr img, @ByVal CvPoint center, @ByVal CvSize axes,
                              double angle, double start_angle,
                              double end_angle, double color );
public static native void cvEllipseAA( CvArr img, @ByVal @Cast("CvPoint*") IntBuffer center, @ByVal CvSize axes,
                              double angle, double start_angle,
                              double end_angle, double color,
                              int scale/*=0*/ );
public static native void cvEllipseAA( CvArr img, @ByVal @Cast("CvPoint*") IntBuffer center, @ByVal CvSize axes,
                              double angle, double start_angle,
                              double end_angle, double color );
public static native void cvEllipseAA( CvArr img, @ByVal @Cast("CvPoint*") int[] center, @ByVal CvSize axes,
                              double angle, double start_angle,
                              double end_angle, double color,
                              int scale/*=0*/ );
public static native void cvEllipseAA( CvArr img, @ByVal @Cast("CvPoint*") int[] center, @ByVal CvSize axes,
                              double angle, double start_angle,
                              double end_angle, double color );

public static native void cvPolyLineAA( CvArr img, @Cast("CvPoint**") PointerPointer pts, IntPointer npts, int contours,
                              int is_closed, double color, int scale/*=0*/ );
public static native void cvPolyLineAA( CvArr img, @ByPtrPtr CvPoint pts, IntPointer npts, int contours,
                              int is_closed, double color );
public static native void cvPolyLineAA( CvArr img, @ByPtrPtr CvPoint pts, IntPointer npts, int contours,
                              int is_closed, double color, int scale/*=0*/ );
public static native void cvPolyLineAA( CvArr img, @Cast("CvPoint**") @ByPtrPtr IntBuffer pts, IntBuffer npts, int contours,
                              int is_closed, double color, int scale/*=0*/ );
public static native void cvPolyLineAA( CvArr img, @Cast("CvPoint**") @ByPtrPtr IntBuffer pts, IntBuffer npts, int contours,
                              int is_closed, double color );
public static native void cvPolyLineAA( CvArr img, @Cast("CvPoint**") @ByPtrPtr int[] pts, int[] npts, int contours,
                              int is_closed, double color, int scale/*=0*/ );
public static native void cvPolyLineAA( CvArr img, @Cast("CvPoint**") @ByPtrPtr int[] pts, int[] npts, int contours,
                              int is_closed, double color );

/****************************************************************************************\
*                                   Pixel Access Macros                                  *
\****************************************************************************************/

public static class CvPixelPosition8u extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvPixelPosition8u() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvPixelPosition8u(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvPixelPosition8u(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvPixelPosition8u position(int position) {
        return (CvPixelPosition8u)super.position(position);
    }

    public native @Cast("uchar*") BytePointer currline(); public native CvPixelPosition8u currline(BytePointer currline);      /* pointer to the start of the current pixel line   */
    public native @Cast("uchar*") BytePointer topline(); public native CvPixelPosition8u topline(BytePointer topline);       /* pointer to the start of the top pixel line       */
    public native @Cast("uchar*") BytePointer bottomline(); public native CvPixelPosition8u bottomline(BytePointer bottomline);    /* pointer to the start of the first line           */
                                    /* which is below the image                         */
    public native int x(); public native CvPixelPosition8u x(int x);                      /* current x coordinate ( in pixels )               */
    public native int width(); public native CvPixelPosition8u width(int width);                  /* width of the image  ( in pixels )                */
    public native int height(); public native CvPixelPosition8u height(int height);                 /* height of the image  ( in pixels )               */
    public native int step(); public native CvPixelPosition8u step(int step);                   /* distance between lines ( in elements of single   */
                                    /* plane )                                          */
    public native int step_arr(int i); public native CvPixelPosition8u step_arr(int i, int step_arr);
    @MemberGetter public native IntPointer step_arr();            /* array: ( 0, -step, step ). It is used for        */
                                    /* vertical moving                                  */
}

/* this structure differs from the above only in data type */
public static class CvPixelPosition8s extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvPixelPosition8s() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvPixelPosition8s(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvPixelPosition8s(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvPixelPosition8s position(int position) {
        return (CvPixelPosition8s)super.position(position);
    }

    public native @Cast("schar*") BytePointer currline(); public native CvPixelPosition8s currline(BytePointer currline);
    public native @Cast("schar*") BytePointer topline(); public native CvPixelPosition8s topline(BytePointer topline);
    public native @Cast("schar*") BytePointer bottomline(); public native CvPixelPosition8s bottomline(BytePointer bottomline);
    public native int x(); public native CvPixelPosition8s x(int x);
    public native int width(); public native CvPixelPosition8s width(int width);
    public native int height(); public native CvPixelPosition8s height(int height);
    public native int step(); public native CvPixelPosition8s step(int step);
    public native int step_arr(int i); public native CvPixelPosition8s step_arr(int i, int step_arr);
    @MemberGetter public native IntPointer step_arr();
}

/* this structure differs from the CvPixelPosition8u only in data type */
public static class CvPixelPosition32f extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvPixelPosition32f() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvPixelPosition32f(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvPixelPosition32f(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvPixelPosition32f position(int position) {
        return (CvPixelPosition32f)super.position(position);
    }

    public native FloatPointer currline(); public native CvPixelPosition32f currline(FloatPointer currline);
    public native FloatPointer topline(); public native CvPixelPosition32f topline(FloatPointer topline);
    public native FloatPointer bottomline(); public native CvPixelPosition32f bottomline(FloatPointer bottomline);
    public native int x(); public native CvPixelPosition32f x(int x);
    public native int width(); public native CvPixelPosition32f width(int width);
    public native int height(); public native CvPixelPosition32f height(int height);
    public native int step(); public native CvPixelPosition32f step(int step);
    public native int step_arr(int i); public native CvPixelPosition32f step_arr(int i, int step_arr);
    @MemberGetter public native IntPointer step_arr();
}


/* Initialize one of the CvPixelPosition structures.   */
/*  pos    - initialized structure                     */
/*  origin - pointer to the left-top corner of the ROI */
/*  step   - width of the whole image in bytes         */
/*  roi    - width & height of the ROI                 */
/*  x, y   - initial position                          */
// #define CV_INIT_PIXEL_POS(pos, origin, _step, roi, _x, _y, orientation)
//     (
//     (pos).step = (_step)/sizeof((pos).crrline[0]) * (orientation ? -1 : 1),
//     (pos).width = (roi).width,
//     (pos).height = (roi).height,
//     (pos).bottomline = (origin) + (pos).step*(pos).height,
//     (pos).topline = (origin) - (pos).step,
//     (pos).step_arr[0] = 0,
//     (pos).step_arr[1] = -(pos).step,
//     (pos).step_arr[2] = (pos).step,
//     (pos).x = (_x),
//     (pos).crrline = (origin) + (pos).step*(_y) )


/* Move to specified point ( absolute shift ) */
/*  pos    - position structure               */
/*  x, y   - coordinates of the new position  */
/*  cs     - number of the image channels     */
// #define CV_MOVE_TO( pos, _x, _y, cs )
// ((pos).crrline = (_y) >= 0 && (_y) < (pos).height ? (pos).topline + ((_y)+1)*(pos).step : 0,
//  (pos).x = (_x) >= 0 && (_x) < (pos).width ? (_x) : 0, (pos).crrline + (_x) * (cs) )

/* Get current coordinates                    */
/*  pos    - position structure               */
/*  x, y   - coordinates of the new position  */
/*  cs     - number of the image channels     */
// #define CV_GET_CURRENT( pos, cs )  ((pos).crrline + (pos).x * (cs))

/* Move by one pixel relatively to current position */
/*  pos    - position structure                     */
/*  cs     - number of the image channels           */

/* left */
// #define CV_MOVE_LEFT( pos, cs )
//  ( --(pos).x >= 0 ? (pos).crrline + (pos).x*(cs) : 0 )

/* right */
// #define CV_MOVE_RIGHT( pos, cs )
//  ( ++(pos).x < (pos).width ? (pos).crrline + (pos).x*(cs) : 0 )

/* up */
// #define CV_MOVE_UP( pos, cs )
//  (((pos).crrline -= (pos).step) != (pos).topline ? (pos).crrline + (pos).x*(cs) : 0 )

/* down */
// #define CV_MOVE_DOWN( pos, cs )
//  (((pos).crrline += (pos).step) != (pos).bottomline ? (pos).crrline + (pos).x*(cs) : 0 )

/* left up */
// #define CV_MOVE_LU( pos, cs ) ( CV_MOVE_LEFT(pos, cs), CV_MOVE_UP(pos, cs))

/* right up */
// #define CV_MOVE_RU( pos, cs ) ( CV_MOVE_RIGHT(pos, cs), CV_MOVE_UP(pos, cs))

/* left down */
// #define CV_MOVE_LD( pos, cs ) ( CV_MOVE_LEFT(pos, cs), CV_MOVE_DOWN(pos, cs))

/* right down */
// #define CV_MOVE_RD( pos, cs ) ( CV_MOVE_RIGHT(pos, cs), CV_MOVE_DOWN(pos, cs))



/* Move by one pixel relatively to current position with wrapping when the position     */
/* achieves image boundary                                                              */
/*  pos    - position structure                                                         */
/*  cs     - number of the image channels                                               */

/* left */
// #define CV_MOVE_LEFT_WRAP( pos, cs )
//  ((pos).crrline + ( --(pos).x >= 0 ? (pos).x : ((pos).x = (pos).width-1))*(cs))

/* right */
// #define CV_MOVE_RIGHT_WRAP( pos, cs )
//  ((pos).crrline + ( ++(pos).x < (pos).width ? (pos).x : ((pos).x = 0))*(cs) )

/* up */
// #define CV_MOVE_UP_WRAP( pos, cs )
//     ((((pos).crrline -= (pos).step) != (pos).topline ?
//     (pos).crrline : ((pos).crrline = (pos).bottomline - (pos).step)) + (pos).x*(cs) )

/* down */
// #define CV_MOVE_DOWN_WRAP( pos, cs )
//     ((((pos).crrline += (pos).step) != (pos).bottomline ?
//     (pos).crrline : ((pos).crrline = (pos).topline + (pos).step)) + (pos).x*(cs) )

/* left up */
// #define CV_MOVE_LU_WRAP( pos, cs ) ( CV_MOVE_LEFT_WRAP(pos, cs), CV_MOVE_UP_WRAP(pos, cs))
/* right up */
// #define CV_MOVE_RU_WRAP( pos, cs ) ( CV_MOVE_RIGHT_WRAP(pos, cs), CV_MOVE_UP_WRAP(pos, cs))
/* left down */
// #define CV_MOVE_LD_WRAP( pos, cs ) ( CV_MOVE_LEFT_WRAP(pos, cs), CV_MOVE_DOWN_WRAP(pos, cs))
/* right down */
// #define CV_MOVE_RD_WRAP( pos, cs ) ( CV_MOVE_RIGHT_WRAP(pos, cs), CV_MOVE_DOWN_WRAP(pos, cs))

/* Numeric constants which used for moving in arbitrary direction  */
/** enum  */
public static final int
    CV_SHIFT_NONE = 2,
    CV_SHIFT_LEFT = 1,
    CV_SHIFT_RIGHT = 3,
    CV_SHIFT_UP = 6,
    CV_SHIFT_DOWN = 10,
    CV_SHIFT_LU = 5,
    CV_SHIFT_RU = 7,
    CV_SHIFT_LD = 9,
    CV_SHIFT_RD = 11;

/* Move by one pixel in specified direction                                     */
/*  pos    - position structure                                                 */
/*  shift  - direction ( it's value must be one of the CV_SHIFT_Ö constants ) */
/*  cs     - number of the image channels                                       */
// #define CV_MOVE_PARAM( pos, shift, cs )
//     ( (pos).crrline += (pos).step_arr[(shift)>>2], (pos).x += ((shift)&3)-2,
//     ((pos).crrline != (pos).topline && (pos).crrline != (pos).bottomline &&
//     (pos).x >= 0 && (pos).x < (pos).width) ? (pos).crrline + (pos).x*(cs) : 0 )

/* Move by one pixel in specified direction with wrapping when the               */
/* position achieves image boundary                                              */
/*  pos    - position structure                                                  */
/*  shift  - direction ( it's value must be one of the CV_SHIFT_Ö constants )  */
/*  cs     - number of the image channels                                        */
// #define CV_MOVE_PARAM_WRAP( pos, shift, cs )
//     ( (pos).crrline += (pos).step_arr[(shift)>>2],
//     (pos).crrline = ((pos).crrline == (pos).topline ?
//     (pos).bottomline - (pos).step :
//     (pos).crrline == (pos).bottomline ?
//     (pos).topline + (pos).step : (pos).crrline),
// 
//     (pos).x += ((shift)&3)-2,
//     (pos).x = ((pos).x < 0 ? (pos).width-1 : (pos).x >= (pos).width ? 0 : (pos).x),
// 
//     (pos).crrline + (pos).x*(cs) )

public static native void cvUnDistortOnce( @Const CvArr src, CvArr dst,
                                @Const FloatPointer intrinsic_matrix,
                                @Const FloatPointer distortion_coeffs,
                                int interpolate );
public static native void cvUnDistortOnce( @Const CvArr src, CvArr dst,
                                @Const FloatBuffer intrinsic_matrix,
                                @Const FloatBuffer distortion_coeffs,
                                int interpolate );
public static native void cvUnDistortOnce( @Const CvArr src, CvArr dst,
                                @Const float[] intrinsic_matrix,
                                @Const float[] distortion_coeffs,
                                int interpolate );

/* the two functions below have quite hackerish implementations, use with care
   (or, which is better, switch to cvUndistortInitMap and cvRemap instead */
public static native void cvUnDistortInit( @Const CvArr src,
                                CvArr undistortion_map,
                                @Const FloatPointer A, @Const FloatPointer k,
                                int interpolate );
public static native void cvUnDistortInit( @Const CvArr src,
                                CvArr undistortion_map,
                                @Const FloatBuffer A, @Const FloatBuffer k,
                                int interpolate );
public static native void cvUnDistortInit( @Const CvArr src,
                                CvArr undistortion_map,
                                @Const float[] A, @Const float[] k,
                                int interpolate );

public static native void cvUnDistort( @Const CvArr src, CvArr dst,
                             @Const CvArr undistortion_map,
                             int interpolate );

/* Find fundamental matrix */
public static native void cvFindFundamentalMatrix( IntPointer points1, IntPointer points2,
    int numpoints, int method, FloatPointer matrix );
public static native void cvFindFundamentalMatrix( IntBuffer points1, IntBuffer points2,
    int numpoints, int method, FloatBuffer matrix );
public static native void cvFindFundamentalMatrix( int[] points1, int[] points2,
    int numpoints, int method, float[] matrix );


public static native int cvFindChessBoardCornerGuesses( @Const Pointer arr, Pointer thresharr,
                               CvMemStorage storage,
                               @ByVal CvSize pattern_size, CvPoint2D32f corners,
                               IntPointer corner_count );
public static native int cvFindChessBoardCornerGuesses( @Const Pointer arr, Pointer thresharr,
                               CvMemStorage storage,
                               @ByVal CvSize pattern_size, @Cast("CvPoint2D32f*") FloatBuffer corners,
                               IntBuffer corner_count );
public static native int cvFindChessBoardCornerGuesses( @Const Pointer arr, Pointer thresharr,
                               CvMemStorage storage,
                               @ByVal CvSize pattern_size, @Cast("CvPoint2D32f*") float[] corners,
                               int[] corner_count );

/* Calibrates camera using multiple views of calibration pattern */
public static native void cvCalibrateCamera( int image_count, IntPointer _point_counts,
    @ByVal CvSize image_size, CvPoint2D32f _image_points, CvPoint3D32f _object_points,
    FloatPointer _distortion_coeffs, FloatPointer _camera_matrix, FloatPointer _translation_vectors,
    FloatPointer _rotation_matrices, int flags );
public static native void cvCalibrateCamera( int image_count, IntBuffer _point_counts,
    @ByVal CvSize image_size, @Cast("CvPoint2D32f*") FloatBuffer _image_points, @Cast("CvPoint3D32f*") FloatBuffer _object_points,
    FloatBuffer _distortion_coeffs, FloatBuffer _camera_matrix, FloatBuffer _translation_vectors,
    FloatBuffer _rotation_matrices, int flags );
public static native void cvCalibrateCamera( int image_count, int[] _point_counts,
    @ByVal CvSize image_size, @Cast("CvPoint2D32f*") float[] _image_points, @Cast("CvPoint3D32f*") float[] _object_points,
    float[] _distortion_coeffs, float[] _camera_matrix, float[] _translation_vectors,
    float[] _rotation_matrices, int flags );


public static native void cvCalibrateCamera_64d( int image_count, IntPointer _point_counts,
    @ByVal CvSize image_size, CvPoint2D64f _image_points, CvPoint3D64f _object_points,
    DoublePointer _distortion_coeffs, DoublePointer _camera_matrix, DoublePointer _translation_vectors,
    DoublePointer _rotation_matrices, int flags );
public static native void cvCalibrateCamera_64d( int image_count, IntBuffer _point_counts,
    @ByVal CvSize image_size, @Cast("CvPoint2D64f*") DoubleBuffer _image_points, @Cast("CvPoint3D64f*") DoubleBuffer _object_points,
    DoubleBuffer _distortion_coeffs, DoubleBuffer _camera_matrix, DoubleBuffer _translation_vectors,
    DoubleBuffer _rotation_matrices, int flags );
public static native void cvCalibrateCamera_64d( int image_count, int[] _point_counts,
    @ByVal CvSize image_size, @Cast("CvPoint2D64f*") double[] _image_points, @Cast("CvPoint3D64f*") double[] _object_points,
    double[] _distortion_coeffs, double[] _camera_matrix, double[] _translation_vectors,
    double[] _rotation_matrices, int flags );


/* Find 3d position of object given intrinsic camera parameters,
   3d model of the object and projection of the object into view plane */
public static native void cvFindExtrinsicCameraParams( int point_count,
    @ByVal CvSize image_size, CvPoint2D32f _image_points,
    CvPoint3D32f _object_points, FloatPointer focal_length,
    @ByVal CvPoint2D32f principal_point, FloatPointer _distortion_coeffs,
    FloatPointer _rotation_vector, FloatPointer _translation_vector );
public static native void cvFindExtrinsicCameraParams( int point_count,
    @ByVal CvSize image_size, @Cast("CvPoint2D32f*") FloatBuffer _image_points,
    @Cast("CvPoint3D32f*") FloatBuffer _object_points, FloatBuffer focal_length,
    @ByVal @Cast("CvPoint2D32f*") FloatBuffer principal_point, FloatBuffer _distortion_coeffs,
    FloatBuffer _rotation_vector, FloatBuffer _translation_vector );
public static native void cvFindExtrinsicCameraParams( int point_count,
    @ByVal CvSize image_size, @Cast("CvPoint2D32f*") float[] _image_points,
    @Cast("CvPoint3D32f*") float[] _object_points, float[] focal_length,
    @ByVal @Cast("CvPoint2D32f*") float[] principal_point, float[] _distortion_coeffs,
    float[] _rotation_vector, float[] _translation_vector );

/* Variant of the previous function that takes double-precision parameters */
public static native void cvFindExtrinsicCameraParams_64d( int point_count,
    @ByVal CvSize image_size, CvPoint2D64f _image_points,
    CvPoint3D64f _object_points, DoublePointer focal_length,
    @ByVal CvPoint2D64f principal_point, DoublePointer _distortion_coeffs,
    DoublePointer _rotation_vector, DoublePointer _translation_vector );
public static native void cvFindExtrinsicCameraParams_64d( int point_count,
    @ByVal CvSize image_size, @Cast("CvPoint2D64f*") DoubleBuffer _image_points,
    @Cast("CvPoint3D64f*") DoubleBuffer _object_points, DoubleBuffer focal_length,
    @ByVal @Cast("CvPoint2D64f*") DoubleBuffer principal_point, DoubleBuffer _distortion_coeffs,
    DoubleBuffer _rotation_vector, DoubleBuffer _translation_vector );
public static native void cvFindExtrinsicCameraParams_64d( int point_count,
    @ByVal CvSize image_size, @Cast("CvPoint2D64f*") double[] _image_points,
    @Cast("CvPoint3D64f*") double[] _object_points, double[] focal_length,
    @ByVal @Cast("CvPoint2D64f*") double[] principal_point, double[] _distortion_coeffs,
    double[] _rotation_vector, double[] _translation_vector );

/* Rodrigues transform */
/** enum  */
public static final int
    CV_RODRIGUES_M2V = 0,
    CV_RODRIGUES_V2M = 1;

/* Converts rotation_matrix matrix to rotation_matrix vector or vice versa */
public static native void cvRodrigues( CvMat rotation_matrix, CvMat rotation_vector,
                              CvMat jacobian, int conv_type );

/* Does reprojection of 3d object points to the view plane */
public static native void cvProjectPoints( int point_count, CvPoint3D64f _object_points,
    DoublePointer _rotation_vector, DoublePointer _translation_vector,
    DoublePointer focal_length, @ByVal CvPoint2D64f principal_point,
    DoublePointer _distortion, CvPoint2D64f _image_points,
    DoublePointer _deriv_points_rotation_matrix,
    DoublePointer _deriv_points_translation_vect,
    DoublePointer _deriv_points_focal,
    DoublePointer _deriv_points_principal_point,
    DoublePointer _deriv_points_distortion_coeffs );
public static native void cvProjectPoints( int point_count, @Cast("CvPoint3D64f*") DoubleBuffer _object_points,
    DoubleBuffer _rotation_vector, DoubleBuffer _translation_vector,
    DoubleBuffer focal_length, @ByVal @Cast("CvPoint2D64f*") DoubleBuffer principal_point,
    DoubleBuffer _distortion, @Cast("CvPoint2D64f*") DoubleBuffer _image_points,
    DoubleBuffer _deriv_points_rotation_matrix,
    DoubleBuffer _deriv_points_translation_vect,
    DoubleBuffer _deriv_points_focal,
    DoubleBuffer _deriv_points_principal_point,
    DoubleBuffer _deriv_points_distortion_coeffs );
public static native void cvProjectPoints( int point_count, @Cast("CvPoint3D64f*") double[] _object_points,
    double[] _rotation_vector, double[] _translation_vector,
    double[] focal_length, @ByVal @Cast("CvPoint2D64f*") double[] principal_point,
    double[] _distortion, @Cast("CvPoint2D64f*") double[] _image_points,
    double[] _deriv_points_rotation_matrix,
    double[] _deriv_points_translation_vect,
    double[] _deriv_points_focal,
    double[] _deriv_points_principal_point,
    double[] _deriv_points_distortion_coeffs );


/* Simpler version of the previous function */
public static native void cvProjectPointsSimple( int point_count, CvPoint3D64f _object_points,
    DoublePointer _rotation_matrix, DoublePointer _translation_vector,
    DoublePointer _camera_matrix, DoublePointer _distortion, CvPoint2D64f _image_points );
public static native void cvProjectPointsSimple( int point_count, @Cast("CvPoint3D64f*") DoubleBuffer _object_points,
    DoubleBuffer _rotation_matrix, DoubleBuffer _translation_vector,
    DoubleBuffer _camera_matrix, DoubleBuffer _distortion, @Cast("CvPoint2D64f*") DoubleBuffer _image_points );
public static native void cvProjectPointsSimple( int point_count, @Cast("CvPoint3D64f*") double[] _object_points,
    double[] _rotation_matrix, double[] _translation_vector,
    double[] _camera_matrix, double[] _distortion, @Cast("CvPoint2D64f*") double[] _image_points );


public static native void cvMake2DPoints(CvMat arg1, CvMat arg2);
public static native void cvMake3DPoints(CvMat arg1, CvMat arg2);

public static native CvMat cvWarpPerspectiveQMatrix(CvPoint2D32f arg1, CvPoint2D32f arg2, CvMat arg3);
public static native CvMat cvWarpPerspectiveQMatrix(@Cast("CvPoint2D32f*") FloatBuffer arg1, @Cast("CvPoint2D32f*") FloatBuffer arg2, CvMat arg3);
public static native CvMat cvWarpPerspectiveQMatrix(@Cast("CvPoint2D32f*") float[] arg1, @Cast("CvPoint2D32f*") float[] arg2, CvMat arg3);

public static native void cvConvertPointsHomogenious(CvMat arg1, CvMat arg2);


//////////////////////////////////// feature extractors: obsolete API //////////////////////////////////

public static class CvSURFPoint extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvSURFPoint() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvSURFPoint(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvSURFPoint(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvSURFPoint position(int position) {
        return (CvSURFPoint)super.position(position);
    }

    public native @ByRef CvPoint2D32f pt(); public native CvSURFPoint pt(CvPoint2D32f pt);

    public native int laplacian(); public native CvSURFPoint laplacian(int laplacian);
    public native int size(); public native CvSURFPoint size(int size);
    public native float dir(); public native CvSURFPoint dir(float dir);
    public native float hessian(); public native CvSURFPoint hessian(float hessian);

}

public static native @ByVal CvSURFPoint cvSURFPoint( @ByVal CvPoint2D32f pt, int laplacian,
                                  int size, float dir/*=0*/,
                                  float hessian/*=0*/);
public static native @ByVal CvSURFPoint cvSURFPoint( @ByVal CvPoint2D32f pt, int laplacian,
                                  int size);
public static native @ByVal CvSURFPoint cvSURFPoint( @ByVal @Cast("CvPoint2D32f*") FloatBuffer pt, int laplacian,
                                  int size, float dir/*=0*/,
                                  float hessian/*=0*/);
public static native @ByVal CvSURFPoint cvSURFPoint( @ByVal @Cast("CvPoint2D32f*") FloatBuffer pt, int laplacian,
                                  int size);
public static native @ByVal CvSURFPoint cvSURFPoint( @ByVal @Cast("CvPoint2D32f*") float[] pt, int laplacian,
                                  int size, float dir/*=0*/,
                                  float hessian/*=0*/);
public static native @ByVal CvSURFPoint cvSURFPoint( @ByVal @Cast("CvPoint2D32f*") float[] pt, int laplacian,
                                  int size);

public static class CvSURFParams extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvSURFParams() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvSURFParams(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvSURFParams(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvSURFParams position(int position) {
        return (CvSURFParams)super.position(position);
    }

    public native int extended(); public native CvSURFParams extended(int extended);
    public native int upright(); public native CvSURFParams upright(int upright);
    public native double hessianThreshold(); public native CvSURFParams hessianThreshold(double hessianThreshold);

    public native int nOctaves(); public native CvSURFParams nOctaves(int nOctaves);
    public native int nOctaveLayers(); public native CvSURFParams nOctaveLayers(int nOctaveLayers);

}

public static native @ByVal CvSURFParams cvSURFParams( double hessianThreshold, int extended/*=0*/ );
public static native @ByVal CvSURFParams cvSURFParams( double hessianThreshold );

// If useProvidedKeyPts!=0, keypoints are not detected, but descriptors are computed
//  at the locations provided in keypoints (a CvSeq of CvSURFPoint).
public static native void cvExtractSURF( @Const CvArr img, @Const CvArr mask,
                          @Cast("CvSeq**") PointerPointer keypoints, @Cast("CvSeq**") PointerPointer descriptors,
                          CvMemStorage storage, @ByVal CvSURFParams params,
                             int useProvidedKeyPts/*=0*/  );
public static native void cvExtractSURF( @Const CvArr img, @Const CvArr mask,
                          @ByPtrPtr CvSeq keypoints, @ByPtrPtr CvSeq descriptors,
                          CvMemStorage storage, @ByVal CvSURFParams params  );
public static native void cvExtractSURF( @Const CvArr img, @Const CvArr mask,
                          @ByPtrPtr CvSeq keypoints, @ByPtrPtr CvSeq descriptors,
                          CvMemStorage storage, @ByVal CvSURFParams params,
                             int useProvidedKeyPts/*=0*/  );

/**
 Maximal Stable Regions Parameters
 */
public static class CvMSERParams extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvMSERParams() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvMSERParams(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvMSERParams(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvMSERParams position(int position) {
        return (CvMSERParams)super.position(position);
    }

    /** delta, in the code, it compares (size_{i}-size_{i-delta})/size_{i-delta} */
    public native int delta(); public native CvMSERParams delta(int delta);
    /** prune the area which bigger than maxArea */
    public native int maxArea(); public native CvMSERParams maxArea(int maxArea);
    /** prune the area which smaller than minArea */
    public native int minArea(); public native CvMSERParams minArea(int minArea);
    /** prune the area have simliar size to its children */
    public native float maxVariation(); public native CvMSERParams maxVariation(float maxVariation);
    /** trace back to cut off mser with diversity < min_diversity */
    public native float minDiversity(); public native CvMSERParams minDiversity(float minDiversity);

    /////// the next few params for MSER of color image

    /** for color image, the evolution steps */
    public native int maxEvolution(); public native CvMSERParams maxEvolution(int maxEvolution);
    /** the area threshold to cause re-initialize */
    public native double areaThreshold(); public native CvMSERParams areaThreshold(double areaThreshold);
    /** ignore too small margin */
    public native double minMargin(); public native CvMSERParams minMargin(double minMargin);
    /** the aperture size for edge blur */
    public native int edgeBlurSize(); public native CvMSERParams edgeBlurSize(int edgeBlurSize);
}



// Extracts the contours of Maximally Stable Extremal Regions



public static class CvStarKeypoint extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvStarKeypoint() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvStarKeypoint(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvStarKeypoint(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvStarKeypoint position(int position) {
        return (CvStarKeypoint)super.position(position);
    }

    public native @ByRef CvPoint pt(); public native CvStarKeypoint pt(CvPoint pt);
    public native int size(); public native CvStarKeypoint size(int size);
    public native float response(); public native CvStarKeypoint response(float response);
}

public static native @ByVal CvStarKeypoint cvStarKeypoint(@ByVal CvPoint pt, int size, float response);
public static native @ByVal CvStarKeypoint cvStarKeypoint(@ByVal @Cast("CvPoint*") IntBuffer pt, int size, float response);
public static native @ByVal CvStarKeypoint cvStarKeypoint(@ByVal @Cast("CvPoint*") int[] pt, int size, float response);

public static class CvStarDetectorParams extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvStarDetectorParams() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvStarDetectorParams(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvStarDetectorParams(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvStarDetectorParams position(int position) {
        return (CvStarDetectorParams)super.position(position);
    }

    public native int maxSize(); public native CvStarDetectorParams maxSize(int maxSize);
    public native int responseThreshold(); public native CvStarDetectorParams responseThreshold(int responseThreshold);
    public native int lineThresholdProjected(); public native CvStarDetectorParams lineThresholdProjected(int lineThresholdProjected);
    public native int lineThresholdBinarized(); public native CvStarDetectorParams lineThresholdBinarized(int lineThresholdBinarized);
    public native int suppressNonmaxSize(); public native CvStarDetectorParams suppressNonmaxSize(int suppressNonmaxSize);
}

public static native @ByVal CvStarDetectorParams cvStarDetectorParams(
                                                    int maxSize/*=45*/,
                                                    int responseThreshold/*=30*/,
                                                    int lineThresholdProjected/*=10*/,
                                                    int lineThresholdBinarized/*=8*/,
                                                    int suppressNonmaxSize/*=5*/);
public static native @ByVal CvStarDetectorParams cvStarDetectorParams();

public static native CvSeq cvGetStarKeypoints( @Const CvArr img, CvMemStorage storage,
                                 @ByVal(nullValue = "cvStarDetectorParams()") CvStarDetectorParams params/*=cvStarDetectorParams()*/);
public static native CvSeq cvGetStarKeypoints( @Const CvArr img, CvMemStorage storage);

// #ifdef __cplusplus
// #endif

// #endif


// Parsed from <opencv2/legacy/legacy.hpp>

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

// #ifndef __OPENCV_LEGACY_HPP__
// #define __OPENCV_LEGACY_HPP__

// #include "opencv2/imgproc/imgproc.hpp"
// #include "opencv2/imgproc/imgproc_c.h"
// #include "opencv2/features2d/features2d.hpp"
// #include "opencv2/calib3d/calib3d.hpp"
// #include "opencv2/ml/ml.hpp"

// #ifdef __cplusplus
// #endif

public static native CvSeq cvSegmentImage( @Const CvArr srcarr, CvArr dstarr,
                                    double canny_threshold,
                                    double ffill_threshold,
                                    CvMemStorage storage );

/****************************************************************************************\
*                                  Eigen objects                                         *
\****************************************************************************************/

@Convention("CV_CDECL") public static class CvCallback extends FunctionPointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public    CvCallback(Pointer p) { super(p); }
    protected CvCallback() { allocate(); }
    private native void allocate();
    public native int call(int index, Pointer buffer, Pointer user_data);
}
public static class CvInput extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvInput() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvInput(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvInput(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvInput position(int position) {
        return (CvInput)super.position(position);
    }

    public native CvCallback callback(); public native CvInput callback(CvCallback callback);
    public native Pointer data(); public native CvInput data(Pointer data);
}

public static final int CV_EIGOBJ_NO_CALLBACK =     0;
public static final int CV_EIGOBJ_INPUT_CALLBACK =  1;
public static final int CV_EIGOBJ_OUTPUT_CALLBACK = 2;
public static final int CV_EIGOBJ_BOTH_CALLBACK =   3;

/* Calculates covariation matrix of a set of arrays */
public static native void cvCalcCovarMatrixEx( int nObjects, Pointer input, int ioFlags,
                                  int ioBufSize, @Cast("uchar*") BytePointer buffer, Pointer userData,
                                  IplImage avg, FloatPointer covarMatrix );
public static native void cvCalcCovarMatrixEx( int nObjects, Pointer input, int ioFlags,
                                  int ioBufSize, @Cast("uchar*") ByteBuffer buffer, Pointer userData,
                                  IplImage avg, FloatBuffer covarMatrix );
public static native void cvCalcCovarMatrixEx( int nObjects, Pointer input, int ioFlags,
                                  int ioBufSize, @Cast("uchar*") byte[] buffer, Pointer userData,
                                  IplImage avg, float[] covarMatrix );

/* Calculates eigen values and vectors of covariation matrix of a set of
   arrays */
public static native void cvCalcEigenObjects( int nObjects, Pointer input, Pointer output,
                                 int ioFlags, int ioBufSize, Pointer userData,
                                 CvTermCriteria calcLimit, IplImage avg,
                                 FloatPointer eigVals );
public static native void cvCalcEigenObjects( int nObjects, Pointer input, Pointer output,
                                 int ioFlags, int ioBufSize, Pointer userData,
                                 CvTermCriteria calcLimit, IplImage avg,
                                 FloatBuffer eigVals );
public static native void cvCalcEigenObjects( int nObjects, Pointer input, Pointer output,
                                 int ioFlags, int ioBufSize, Pointer userData,
                                 CvTermCriteria calcLimit, IplImage avg,
                                 float[] eigVals );

/* Calculates dot product (obj - avg) * eigObj (i.e. projects image to eigen vector) */
public static native double cvCalcDecompCoeff( IplImage obj, IplImage eigObj, IplImage avg );

/* Projects image to eigen space (finds all decomposion coefficients */
public static native void cvEigenDecomposite( IplImage obj, int nEigObjs, Pointer eigInput,
                                 int ioFlags, Pointer userData, IplImage avg,
                                 FloatPointer coeffs );
public static native void cvEigenDecomposite( IplImage obj, int nEigObjs, Pointer eigInput,
                                 int ioFlags, Pointer userData, IplImage avg,
                                 FloatBuffer coeffs );
public static native void cvEigenDecomposite( IplImage obj, int nEigObjs, Pointer eigInput,
                                 int ioFlags, Pointer userData, IplImage avg,
                                 float[] coeffs );

/* Projects original objects used to calculate eigen space basis to that space */
public static native void cvEigenProjection( Pointer eigInput, int nEigObjs, int ioFlags,
                                Pointer userData, FloatPointer coeffs, IplImage avg,
                                IplImage proj );
public static native void cvEigenProjection( Pointer eigInput, int nEigObjs, int ioFlags,
                                Pointer userData, FloatBuffer coeffs, IplImage avg,
                                IplImage proj );
public static native void cvEigenProjection( Pointer eigInput, int nEigObjs, int ioFlags,
                                Pointer userData, float[] coeffs, IplImage avg,
                                IplImage proj );

/****************************************************************************************\
*                                       1D/2D HMM                                        *
\****************************************************************************************/

public static class CvImgObsInfo extends AbstractCvImgObsInfo {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvImgObsInfo() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvImgObsInfo(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvImgObsInfo(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvImgObsInfo position(int position) {
        return (CvImgObsInfo)super.position(position);
    }

    public native int obs_x(); public native CvImgObsInfo obs_x(int obs_x);
    public native int obs_y(); public native CvImgObsInfo obs_y(int obs_y);
    public native int obs_size(); public native CvImgObsInfo obs_size(int obs_size);
    public native FloatPointer obs(); public native CvImgObsInfo obs(FloatPointer obs);//consequtive observations

    public native IntPointer state(); public native CvImgObsInfo state(IntPointer state);/* arr of pairs superstate/state to which observation belong */
    public native IntPointer mix(); public native CvImgObsInfo mix(IntPointer mix);  /* number of mixture to which observation belong */

}/*struct for 1 image*/

public static class CvEHMMState extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvEHMMState() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvEHMMState(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvEHMMState(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvEHMMState position(int position) {
        return (CvEHMMState)super.position(position);
    }

    public native int num_mix(); public native CvEHMMState num_mix(int num_mix);        /*number of mixtures in this state*/
    public native FloatPointer mu(); public native CvEHMMState mu(FloatPointer mu);          /*mean vectors corresponding to each mixture*/
    public native FloatPointer inv_var(); public native CvEHMMState inv_var(FloatPointer inv_var);     /* square root of inversed variances corresp. to each mixture*/
    public native FloatPointer log_var_val(); public native CvEHMMState log_var_val(FloatPointer log_var_val); /* sum of 0.5 (LN2PI + ln(variance[i]) ) for i=1,n */
    public native FloatPointer weight(); public native CvEHMMState weight(FloatPointer weight);      /*array of mixture weights. Summ of all weights in state is 1. */

}

public static class CvEHMM extends AbstractCvEHMM {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvEHMM() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvEHMM(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvEHMM(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvEHMM position(int position) {
        return (CvEHMM)super.position(position);
    }

    public native int level(); public native CvEHMM level(int level); /* 0 - lowest(i.e its states are real states), ..... */
    public native int num_states(); public native CvEHMM num_states(int num_states); /* number of HMM states */
    public native FloatPointer transP(); public native CvEHMM transP(FloatPointer transP);/*transition probab. matrices for states */
    public native FloatPointer obsProb(int i); public native CvEHMM obsProb(int i, FloatPointer obsProb);
    @MemberGetter public native @Cast("float**") PointerPointer obsProb(); /* if level == 0 - array of brob matrices corresponding to hmm
                        if level == 1 - martix of matrices */
        @Name("u.state") public native CvEHMMState u_state(); public native CvEHMM u_state(CvEHMMState u_state); /* if level == 0 points to real states array,
                               if not - points to embedded hmms */
        @Name("u.ehmm") public native CvEHMM u_ehmm(); public native CvEHMM u_ehmm(CvEHMM u_ehmm); /* pointer to an embedded model or NULL, if it is a leaf */

}

/*CVAPI(int)  icvCreate1DHMM( CvEHMM** this_hmm,
                                   int state_number, int* num_mix, int obs_size );

CVAPI(int)  icvRelease1DHMM( CvEHMM** phmm );

CVAPI(int)  icvUniform1DSegm( Cv1DObsInfo* obs_info, CvEHMM* hmm );

CVAPI(int)  icvInit1DMixSegm( Cv1DObsInfo** obs_info_array, int num_img, CvEHMM* hmm);

CVAPI(int)  icvEstimate1DHMMStateParams( CvImgObsInfo** obs_info_array, int num_img, CvEHMM* hmm);

CVAPI(int)  icvEstimate1DObsProb( CvImgObsInfo* obs_info, CvEHMM* hmm );

CVAPI(int)  icvEstimate1DTransProb( Cv1DObsInfo** obs_info_array,
                                           int num_seq,
                                           CvEHMM* hmm );

CVAPI(float)  icvViterbi( Cv1DObsInfo* obs_info, CvEHMM* hmm);

CVAPI(int)  icv1DMixSegmL2( CvImgObsInfo** obs_info_array, int num_img, CvEHMM* hmm );*/

/*********************************** Embedded HMMs *************************************/

/* Creates 2D HMM */
public static native CvEHMM cvCreate2DHMM( IntPointer stateNumber, IntPointer numMix, int obsSize );
public static native CvEHMM cvCreate2DHMM( IntBuffer stateNumber, IntBuffer numMix, int obsSize );
public static native CvEHMM cvCreate2DHMM( int[] stateNumber, int[] numMix, int obsSize );

/* Releases HMM */
public static native void cvRelease2DHMM( @Cast("CvEHMM**") PointerPointer hmm );
public static native void cvRelease2DHMM( @ByPtrPtr CvEHMM hmm );

// #define CV_COUNT_OBS(roi, win, delta, numObs )
// {
//    (numObs)->width  =((roi)->width  -(win)->width  +(delta)->width)/(delta)->width;
//    (numObs)->height =((roi)->height -(win)->height +(delta)->height)/(delta)->height;
// }

/* Creates storage for observation vectors */
public static native CvImgObsInfo cvCreateObsInfo( @ByVal CvSize numObs, int obsSize );

/* Releases storage for observation vectors */
public static native void cvReleaseObsInfo( @Cast("CvImgObsInfo**") PointerPointer obs_info );
public static native void cvReleaseObsInfo( @ByPtrPtr CvImgObsInfo obs_info );


/* The function takes an image on input and and returns the sequnce of observations
   to be used with an embedded HMM; Each observation is top-left block of DCT
   coefficient matrix */
public static native void cvImgToObs_DCT( @Const CvArr arr, FloatPointer obs, @ByVal CvSize dctSize,
                             @ByVal CvSize obsSize, @ByVal CvSize delta );
public static native void cvImgToObs_DCT( @Const CvArr arr, FloatBuffer obs, @ByVal CvSize dctSize,
                             @ByVal CvSize obsSize, @ByVal CvSize delta );
public static native void cvImgToObs_DCT( @Const CvArr arr, float[] obs, @ByVal CvSize dctSize,
                             @ByVal CvSize obsSize, @ByVal CvSize delta );


/* Uniformly segments all observation vectors extracted from image */
public static native void cvUniformImgSegm( CvImgObsInfo obs_info, CvEHMM ehmm );

/* Does mixture segmentation of the states of embedded HMM */
public static native void cvInitMixSegm( @Cast("CvImgObsInfo**") PointerPointer obs_info_array,
                            int num_img, CvEHMM hmm );
public static native void cvInitMixSegm( @ByPtrPtr CvImgObsInfo obs_info_array,
                            int num_img, CvEHMM hmm );

/* Function calculates means, variances, weights of every Gaussian mixture
   of every low-level state of embedded HMM */
public static native void cvEstimateHMMStateParams( @Cast("CvImgObsInfo**") PointerPointer obs_info_array,
                                       int num_img, CvEHMM hmm );
public static native void cvEstimateHMMStateParams( @ByPtrPtr CvImgObsInfo obs_info_array,
                                       int num_img, CvEHMM hmm );

/* Function computes transition probability matrices of embedded HMM
   given observations segmentation */
public static native void cvEstimateTransProb( @Cast("CvImgObsInfo**") PointerPointer obs_info_array,
                                  int num_img, CvEHMM hmm );
public static native void cvEstimateTransProb( @ByPtrPtr CvImgObsInfo obs_info_array,
                                  int num_img, CvEHMM hmm );

/* Function computes probabilities of appearing observations at any state
   (i.e. computes P(obs|state) for every pair(obs,state)) */
public static native void cvEstimateObsProb( CvImgObsInfo obs_info,
                                CvEHMM hmm );

/* Runs Viterbi algorithm for embedded HMM */
public static native float cvEViterbi( CvImgObsInfo obs_info, CvEHMM hmm );


/* Function clusters observation vectors from several images
   given observations segmentation.
   Euclidean distance used for clustering vectors.
   Centers of clusters are given means of every mixture */
public static native void cvMixSegmL2( @Cast("CvImgObsInfo**") PointerPointer obs_info_array,
                          int num_img, CvEHMM hmm );
public static native void cvMixSegmL2( @ByPtrPtr CvImgObsInfo obs_info_array,
                          int num_img, CvEHMM hmm );

/****************************************************************************************\
*               A few functions from old stereo gesture recognition demosions            *
\****************************************************************************************/

/* Creates hand mask image given several points on the hand */
public static native void cvCreateHandMask( CvSeq hand_points,
                                   IplImage img_mask, CvRect roi);

/* Finds hand region in range image data */
public static native void cvFindHandRegion(CvPoint3D32f points, int count,
                                CvSeq indexs,
                                FloatPointer line, @ByVal CvSize2D32f size, int flag,
                                CvPoint3D32f center,
                                CvMemStorage storage, @Cast("CvSeq**") PointerPointer numbers);
public static native void cvFindHandRegion(CvPoint3D32f points, int count,
                                CvSeq indexs,
                                FloatPointer line, @ByVal CvSize2D32f size, int flag,
                                CvPoint3D32f center,
                                CvMemStorage storage, @ByPtrPtr CvSeq numbers);
public static native void cvFindHandRegion(@Cast("CvPoint3D32f*") FloatBuffer points, int count,
                                CvSeq indexs,
                                FloatBuffer line, @ByVal CvSize2D32f size, int flag,
                                @Cast("CvPoint3D32f*") FloatBuffer center,
                                CvMemStorage storage, @ByPtrPtr CvSeq numbers);
public static native void cvFindHandRegion(@Cast("CvPoint3D32f*") float[] points, int count,
                                CvSeq indexs,
                                float[] line, @ByVal CvSize2D32f size, int flag,
                                @Cast("CvPoint3D32f*") float[] center,
                                CvMemStorage storage, @ByPtrPtr CvSeq numbers);

/* Finds hand region in range image data (advanced version) */
public static native void cvFindHandRegionA( CvPoint3D32f points, int count,
                                CvSeq indexs,
                                FloatPointer line, @ByVal CvSize2D32f size, int jc,
                                CvPoint3D32f center,
                                CvMemStorage storage, @Cast("CvSeq**") PointerPointer numbers);
public static native void cvFindHandRegionA( CvPoint3D32f points, int count,
                                CvSeq indexs,
                                FloatPointer line, @ByVal CvSize2D32f size, int jc,
                                CvPoint3D32f center,
                                CvMemStorage storage, @ByPtrPtr CvSeq numbers);
public static native void cvFindHandRegionA( @Cast("CvPoint3D32f*") FloatBuffer points, int count,
                                CvSeq indexs,
                                FloatBuffer line, @ByVal CvSize2D32f size, int jc,
                                @Cast("CvPoint3D32f*") FloatBuffer center,
                                CvMemStorage storage, @ByPtrPtr CvSeq numbers);
public static native void cvFindHandRegionA( @Cast("CvPoint3D32f*") float[] points, int count,
                                CvSeq indexs,
                                float[] line, @ByVal CvSize2D32f size, int jc,
                                @Cast("CvPoint3D32f*") float[] center,
                                CvMemStorage storage, @ByPtrPtr CvSeq numbers);

/* Calculates the cooficients of the homography matrix */
public static native void cvCalcImageHomography( FloatPointer line, CvPoint3D32f center,
                                    FloatPointer intrinsic, FloatPointer homography );
public static native void cvCalcImageHomography( FloatBuffer line, @Cast("CvPoint3D32f*") FloatBuffer center,
                                    FloatBuffer intrinsic, FloatBuffer homography );
public static native void cvCalcImageHomography( float[] line, @Cast("CvPoint3D32f*") float[] center,
                                    float[] intrinsic, float[] homography );

/****************************************************************************************\
*                           More operations on sequences                                 *
\****************************************************************************************/

/*****************************************************************************************/

// #define CV_CURRENT_INT( reader ) (*((int *)(reader).ptr))
// #define CV_PREV_INT( reader ) (*((int *)(reader).prev_elem))

// #define  CV_GRAPH_WEIGHTED_VERTEX_FIELDS() CV_GRAPH_VERTEX_FIELDS()
//     float weight;

// #define  CV_GRAPH_WEIGHTED_EDGE_FIELDS() CV_GRAPH_EDGE_FIELDS()

public static class CvGraphWeightedVtx extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvGraphWeightedVtx() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvGraphWeightedVtx(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvGraphWeightedVtx(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvGraphWeightedVtx position(int position) {
        return (CvGraphWeightedVtx)super.position(position);
    }

    public native int flags(); public native CvGraphWeightedVtx flags(int flags);
    public native CvGraphEdge first(); public native CvGraphWeightedVtx first(CvGraphEdge first);
    public native float weight(); public native CvGraphWeightedVtx weight(float weight);
}

public static class CvGraphWeightedEdge extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvGraphWeightedEdge() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvGraphWeightedEdge(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvGraphWeightedEdge(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvGraphWeightedEdge position(int position) {
        return (CvGraphWeightedEdge)super.position(position);
    }

    public native int flags(); public native CvGraphWeightedEdge flags(int flags);
    public native float weight(); public native CvGraphWeightedEdge weight(float weight);
    public native CvGraphEdge next(int i); public native CvGraphWeightedEdge next(int i, CvGraphEdge next);
    @MemberGetter public native @Cast("CvGraphEdge**") PointerPointer next();
    public native CvGraphVtx vtx(int i); public native CvGraphWeightedEdge vtx(int i, CvGraphVtx vtx);
    @MemberGetter public native @Cast("CvGraphVtx**") PointerPointer vtx();
}

/** enum CvGraphWeightType */
public static final int
    CV_NOT_WEIGHTED = 0,
    CV_WEIGHTED_VTX = 1,
    CV_WEIGHTED_EDGE = 2,
    CV_WEIGHTED_ALL = 3;


/* Calculates histogram of a contour */
public static native void cvCalcPGH( @Const CvSeq contour, CvHistogram hist );

public static final int CV_DOMINANT_IPAN = 1;

/* Finds high-curvature points of the contour */
public static native CvSeq cvFindDominantPoints( CvSeq contour, CvMemStorage storage,
                                   int method/*=CV_DOMINANT_IPAN*/,
                                   double parameter1/*=0*/,
                                   double parameter2/*=0*/,
                                   double parameter3/*=0*/,
                                   double parameter4/*=0*/);
public static native CvSeq cvFindDominantPoints( CvSeq contour, CvMemStorage storage);

/*****************************************************************************************/


/*******************************Stereo correspondence*************************************/

public static class CvCliqueFinder extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvCliqueFinder() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvCliqueFinder(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvCliqueFinder(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvCliqueFinder position(int position) {
        return (CvCliqueFinder)super.position(position);
    }

    public native CvGraph graph(); public native CvCliqueFinder graph(CvGraph graph);
    public native IntPointer adj_matr(int i); public native CvCliqueFinder adj_matr(int i, IntPointer adj_matr);
    @MemberGetter public native @Cast("int**") PointerPointer adj_matr();
    public native int N(); public native CvCliqueFinder N(int N); //graph size

    // stacks, counters etc/
    public native int k(); public native CvCliqueFinder k(int k); //stack size
    public native IntPointer current_comp(); public native CvCliqueFinder current_comp(IntPointer current_comp);
    public native IntPointer All(int i); public native CvCliqueFinder All(int i, IntPointer All);
    @MemberGetter public native @Cast("int**") PointerPointer All();

    public native IntPointer ne(); public native CvCliqueFinder ne(IntPointer ne);
    public native IntPointer ce(); public native CvCliqueFinder ce(IntPointer ce);
    public native IntPointer fixp(); public native CvCliqueFinder fixp(IntPointer fixp); //node with minimal disconnections
    public native IntPointer nod(); public native CvCliqueFinder nod(IntPointer nod);
    public native IntPointer s(); public native CvCliqueFinder s(IntPointer s); //for selected candidate
    public native int status(); public native CvCliqueFinder status(int status);
    public native int best_score(); public native CvCliqueFinder best_score(int best_score);
    public native int weighted(); public native CvCliqueFinder weighted(int weighted);
    public native int weighted_edges(); public native CvCliqueFinder weighted_edges(int weighted_edges);
    public native float best_weight(); public native CvCliqueFinder best_weight(float best_weight);
    public native FloatPointer edge_weights(); public native CvCliqueFinder edge_weights(FloatPointer edge_weights);
    public native FloatPointer vertex_weights(); public native CvCliqueFinder vertex_weights(FloatPointer vertex_weights);
    public native FloatPointer cur_weight(); public native CvCliqueFinder cur_weight(FloatPointer cur_weight);
    public native FloatPointer cand_weight(); public native CvCliqueFinder cand_weight(FloatPointer cand_weight);

}

public static final int CLIQUE_TIME_OFF = 2;
public static final int CLIQUE_FOUND = 1;
public static final int CLIQUE_END =   0;

/*CVAPI(void) cvStartFindCliques( CvGraph* graph, CvCliqueFinder* finder, int reverse,
                                   int weighted CV_DEFAULT(0),  int weighted_edges CV_DEFAULT(0));
CVAPI(int) cvFindNextMaximalClique( CvCliqueFinder* finder, int* clock_rest CV_DEFAULT(0) );
CVAPI(void) cvEndFindCliques( CvCliqueFinder* finder );

CVAPI(void) cvBronKerbosch( CvGraph* graph );*/


/*F///////////////////////////////////////////////////////////////////////////////////////
//
//    Name:    cvSubgraphWeight
//    Purpose: finds weight of subgraph in a graph
//    Context:
//    Parameters:
//      graph - input graph.
//      subgraph - sequence of pairwise different ints.  These are indices of vertices of subgraph.
//      weight_type - describes the way we measure weight.
//            one of the following:
//            CV_NOT_WEIGHTED - weight of a clique is simply its size
//            CV_WEIGHTED_VTX - weight of a clique is the sum of weights of its vertices
//            CV_WEIGHTED_EDGE - the same but edges
//            CV_WEIGHTED_ALL - the same but both edges and vertices
//      weight_vtx - optional vector of floats, with size = graph->total.
//            If weight_type is either CV_WEIGHTED_VTX or CV_WEIGHTED_ALL
//            weights of vertices must be provided.  If weight_vtx not zero
//            these weights considered to be here, otherwise function assumes
//            that vertices of graph are inherited from CvGraphWeightedVtx.
//      weight_edge - optional matrix of floats, of width and height = graph->total.
//            If weight_type is either CV_WEIGHTED_EDGE or CV_WEIGHTED_ALL
//            weights of edges ought to be supplied.  If weight_edge is not zero
//            function finds them here, otherwise function expects
//            edges of graph to be inherited from CvGraphWeightedEdge.
//            If this parameter is not zero structure of the graph is determined from matrix
//            rather than from CvGraphEdge's.  In particular, elements corresponding to
//            absent edges should be zero.
//    Returns:
//      weight of subgraph.
//    Notes:
//F*/
/*CVAPI(float) cvSubgraphWeight( CvGraph *graph, CvSeq *subgraph,
                                  CvGraphWeightType weight_type CV_DEFAULT(CV_NOT_WEIGHTED),
                                  CvVect32f weight_vtx CV_DEFAULT(0),
                                  CvMatr32f weight_edge CV_DEFAULT(0) );*/


/*F///////////////////////////////////////////////////////////////////////////////////////
//
//    Name:    cvFindCliqueEx
//    Purpose: tries to find clique with maximum possible weight in a graph
//    Context:
//    Parameters:
//      graph - input graph.
//      storage - memory storage to be used by the result.
//      is_complementary - optional flag showing whether function should seek for clique
//            in complementary graph.
//      weight_type - describes our notion about weight.
//            one of the following:
//            CV_NOT_WEIGHTED - weight of a clique is simply its size
//            CV_WEIGHTED_VTX - weight of a clique is the sum of weights of its vertices
//            CV_WEIGHTED_EDGE - the same but edges
//            CV_WEIGHTED_ALL - the same but both edges and vertices
//      weight_vtx - optional vector of floats, with size = graph->total.
//            If weight_type is either CV_WEIGHTED_VTX or CV_WEIGHTED_ALL
//            weights of vertices must be provided.  If weight_vtx not zero
//            these weights considered to be here, otherwise function assumes
//            that vertices of graph are inherited from CvGraphWeightedVtx.
//      weight_edge - optional matrix of floats, of width and height = graph->total.
//            If weight_type is either CV_WEIGHTED_EDGE or CV_WEIGHTED_ALL
//            weights of edges ought to be supplied.  If weight_edge is not zero
//            function finds them here, otherwise function expects
//            edges of graph to be inherited from CvGraphWeightedEdge.
//            Note that in case of CV_WEIGHTED_EDGE or CV_WEIGHTED_ALL
//            nonzero is_complementary implies nonzero weight_edge.
//      start_clique - optional sequence of pairwise different ints.  They are indices of
//            vertices that shall be present in the output clique.
//      subgraph_of_ban - optional sequence of (maybe equal) ints.  They are indices of
//            vertices that shall not be present in the output clique.
//      clique_weight_ptr - optional output parameter.  Weight of found clique stored here.
//      num_generations - optional number of generations in evolutionary part of algorithm,
//            zero forces to return first found clique.
//      quality - optional parameter determining degree of required quality/speed tradeoff.
//            Must be in the range from 0 to 9.
//            0 is fast and dirty, 9 is slow but hopefully yields good clique.
//    Returns:
//      sequence of pairwise different ints.
//      These are indices of vertices that form found clique.
//    Notes:
//      in cases of CV_WEIGHTED_EDGE and CV_WEIGHTED_ALL weights should be nonnegative.
//      start_clique has a priority over subgraph_of_ban.
//F*/
/*CVAPI(CvSeq*) cvFindCliqueEx( CvGraph *graph, CvMemStorage *storage,
                                 int is_complementary CV_DEFAULT(0),
                                 CvGraphWeightType weight_type CV_DEFAULT(CV_NOT_WEIGHTED),
                                 CvVect32f weight_vtx CV_DEFAULT(0),
                                 CvMatr32f weight_edge CV_DEFAULT(0),
                                 CvSeq *start_clique CV_DEFAULT(0),
                                 CvSeq *subgraph_of_ban CV_DEFAULT(0),
                                 float *clique_weight_ptr CV_DEFAULT(0),
                                 int num_generations CV_DEFAULT(3),
                                 int quality CV_DEFAULT(2) );*/


public static final int CV_UNDEF_SC_PARAM =         12345; //default value of parameters

public static final int CV_IDP_BIRCHFIELD_PARAM1 =  25;
public static final int CV_IDP_BIRCHFIELD_PARAM2 =  5;
public static final int CV_IDP_BIRCHFIELD_PARAM3 =  12;
public static final int CV_IDP_BIRCHFIELD_PARAM4 =  15;
public static final int CV_IDP_BIRCHFIELD_PARAM5 =  25;


public static final int CV_DISPARITY_BIRCHFIELD =  0;


/*F///////////////////////////////////////////////////////////////////////////
//
//    Name:    cvFindStereoCorrespondence
//    Purpose: find stereo correspondence on stereo-pair
//    Context:
//    Parameters:
//      leftImage - left image of stereo-pair (format 8uC1).
//      rightImage - right image of stereo-pair (format 8uC1).
//   mode - mode of correspondence retrieval (now CV_DISPARITY_BIRCHFIELD only)
//      dispImage - destination disparity image
//      maxDisparity - maximal disparity
//      param1, param2, param3, param4, param5 - parameters of algorithm
//    Returns:
//    Notes:
//      Images must be rectified.
//      All images must have format 8uC1.
//F*/
public static native void cvFindStereoCorrespondence(
                   @Const CvArr leftImage, @Const CvArr rightImage,
                   int mode,
                   CvArr dispImage,
                   int maxDisparity,
                   double param1/*=CV_UNDEF_SC_PARAM*/,
                   double param2/*=CV_UNDEF_SC_PARAM*/,
                   double param3/*=CV_UNDEF_SC_PARAM*/,
                   double param4/*=CV_UNDEF_SC_PARAM*/,
                   double param5/*=CV_UNDEF_SC_PARAM*/ );
public static native void cvFindStereoCorrespondence(
                   @Const CvArr leftImage, @Const CvArr rightImage,
                   int mode,
                   CvArr dispImage,
                   int maxDisparity );

/*****************************************************************************************/
/************ Epiline functions *******************/



public static class CvStereoLineCoeff extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvStereoLineCoeff() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvStereoLineCoeff(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvStereoLineCoeff(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvStereoLineCoeff position(int position) {
        return (CvStereoLineCoeff)super.position(position);
    }

    public native double Xcoef(); public native CvStereoLineCoeff Xcoef(double Xcoef);
    public native double XcoefA(); public native CvStereoLineCoeff XcoefA(double XcoefA);
    public native double XcoefB(); public native CvStereoLineCoeff XcoefB(double XcoefB);
    public native double XcoefAB(); public native CvStereoLineCoeff XcoefAB(double XcoefAB);

    public native double Ycoef(); public native CvStereoLineCoeff Ycoef(double Ycoef);
    public native double YcoefA(); public native CvStereoLineCoeff YcoefA(double YcoefA);
    public native double YcoefB(); public native CvStereoLineCoeff YcoefB(double YcoefB);
    public native double YcoefAB(); public native CvStereoLineCoeff YcoefAB(double YcoefAB);

    public native double Zcoef(); public native CvStereoLineCoeff Zcoef(double Zcoef);
    public native double ZcoefA(); public native CvStereoLineCoeff ZcoefA(double ZcoefA);
    public native double ZcoefB(); public native CvStereoLineCoeff ZcoefB(double ZcoefB);
    public native double ZcoefAB(); public native CvStereoLineCoeff ZcoefAB(double ZcoefAB);
}


public static class CvCamera extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvCamera() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvCamera(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvCamera(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvCamera position(int position) {
        return (CvCamera)super.position(position);
    }

    public native float imgSize(int i); public native CvCamera imgSize(int i, float imgSize);
    @MemberGetter public native FloatPointer imgSize(); /* size of the camera view, used during calibration */
    public native float matrix(int i); public native CvCamera matrix(int i, float matrix);
    @MemberGetter public native FloatPointer matrix(); /* intinsic camera parameters:  [ fx 0 cx; 0 fy cy; 0 0 1 ] */
    public native float distortion(int i); public native CvCamera distortion(int i, float distortion);
    @MemberGetter public native FloatPointer distortion(); /* distortion coefficients - two coefficients for radial distortion
                              and another two for tangential: [ k1 k2 p1 p2 ] */
    public native float rotMatr(int i); public native CvCamera rotMatr(int i, float rotMatr);
    @MemberGetter public native FloatPointer rotMatr();
    public native float transVect(int i); public native CvCamera transVect(int i, float transVect);
    @MemberGetter public native FloatPointer transVect(); /* rotation matrix and transition vector relatively
                             to some reference point in the space. */
}

public static class CvStereoCamera extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvStereoCamera() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvStereoCamera(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvStereoCamera(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvStereoCamera position(int position) {
        return (CvStereoCamera)super.position(position);
    }

    public native CvCamera camera(int i); public native CvStereoCamera camera(int i, CvCamera camera);
    @MemberGetter public native @Cast("CvCamera**") PointerPointer camera(); /* two individual camera parameters */
    public native float fundMatr(int i); public native CvStereoCamera fundMatr(int i, float fundMatr);
    @MemberGetter public native FloatPointer fundMatr(); /* fundamental matrix */

    /* New part for stereo */
    public native @ByRef CvPoint3D32f epipole(int i); public native CvStereoCamera epipole(int i, CvPoint3D32f epipole);
    @MemberGetter public native CvPoint3D32f epipole();
    public native @ByRef CvPoint2D32f quad(int i, int j); public native CvStereoCamera quad(int i, int j, CvPoint2D32f quad);
    @MemberGetter public native @Cast("CvPoint2D32f(*)[4]") CvPoint2D32f quad(); /* coordinates of destination quadrangle after
                                epipolar geometry rectification */
    public native double coeffs(int i, int j, int k); public native CvStereoCamera coeffs(int i, int j, int k, double coeffs);
    @MemberGetter public native @Cast("double(*)[3][3]") DoublePointer coeffs();/* coefficients for transformation */
    public native @ByRef CvPoint2D32f border(int i, int j); public native CvStereoCamera border(int i, int j, CvPoint2D32f border);
    @MemberGetter public native @Cast("CvPoint2D32f(*)[4]") CvPoint2D32f border();
    public native @ByRef CvSize warpSize(); public native CvStereoCamera warpSize(CvSize warpSize);
    public native CvStereoLineCoeff lineCoeffs(); public native CvStereoCamera lineCoeffs(CvStereoLineCoeff lineCoeffs);
    public native int needSwapCameras(); public native CvStereoCamera needSwapCameras(int needSwapCameras);/* flag set to 1 if need to swap cameras for good reconstruction */
    public native float rotMatrix(int i); public native CvStereoCamera rotMatrix(int i, float rotMatrix);
    @MemberGetter public native FloatPointer rotMatrix();
    public native float transVector(int i); public native CvStereoCamera transVector(int i, float transVector);
    @MemberGetter public native FloatPointer transVector();
}


public static class CvContourOrientation extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvContourOrientation() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvContourOrientation(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvContourOrientation(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvContourOrientation position(int position) {
        return (CvContourOrientation)super.position(position);
    }

    public native float egvals(int i); public native CvContourOrientation egvals(int i, float egvals);
    @MemberGetter public native FloatPointer egvals();
    public native float egvects(int i); public native CvContourOrientation egvects(int i, float egvects);
    @MemberGetter public native FloatPointer egvects();

    public native float max(); public native CvContourOrientation max(float max);
    public native float min(); public native CvContourOrientation min(float min); // minimum and maximum projections
    public native int imax(); public native CvContourOrientation imax(int imax);
    public native int imin(); public native CvContourOrientation imin(int imin);
}

public static final int CV_CAMERA_TO_WARP = 1;
public static final int CV_WARP_TO_CAMERA = 2;

public static native int icvConvertWarpCoordinates(@Cast("double(*)[3]") DoublePointer coeffs,
                                CvPoint2D32f cameraPoint,
                                CvPoint2D32f warpPoint,
                                int direction);
public static native int icvConvertWarpCoordinates(@Cast("double(*)[3]") DoubleBuffer coeffs,
                                @Cast("CvPoint2D32f*") FloatBuffer cameraPoint,
                                @Cast("CvPoint2D32f*") FloatBuffer warpPoint,
                                int direction);
public static native int icvConvertWarpCoordinates(@Cast("double(*)[3]") double[] coeffs,
                                @Cast("CvPoint2D32f*") float[] cameraPoint,
                                @Cast("CvPoint2D32f*") float[] warpPoint,
                                int direction);

public static native int icvGetSymPoint3D(  @ByVal CvPoint3D64f pointCorner,
                            @ByVal CvPoint3D64f point1,
                            @ByVal CvPoint3D64f point2,
                            CvPoint3D64f pointSym2);
public static native int icvGetSymPoint3D(  @ByVal @Cast("CvPoint3D64f*") DoubleBuffer pointCorner,
                            @ByVal @Cast("CvPoint3D64f*") DoubleBuffer point1,
                            @ByVal @Cast("CvPoint3D64f*") DoubleBuffer point2,
                            @Cast("CvPoint3D64f*") DoubleBuffer pointSym2);
public static native int icvGetSymPoint3D(  @ByVal @Cast("CvPoint3D64f*") double[] pointCorner,
                            @ByVal @Cast("CvPoint3D64f*") double[] point1,
                            @ByVal @Cast("CvPoint3D64f*") double[] point2,
                            @Cast("CvPoint3D64f*") double[] pointSym2);

public static native void icvGetPieceLength3D(@ByVal CvPoint3D64f point1,@ByVal CvPoint3D64f point2,DoublePointer dist);
public static native void icvGetPieceLength3D(@ByVal @Cast("CvPoint3D64f*") DoubleBuffer point1,@ByVal @Cast("CvPoint3D64f*") DoubleBuffer point2,DoubleBuffer dist);
public static native void icvGetPieceLength3D(@ByVal @Cast("CvPoint3D64f*") double[] point1,@ByVal @Cast("CvPoint3D64f*") double[] point2,double[] dist);

public static native int icvCompute3DPoint(    double alpha,double betta,
                            CvStereoLineCoeff coeffs,
                            CvPoint3D64f point);
public static native int icvCompute3DPoint(    double alpha,double betta,
                            CvStereoLineCoeff coeffs,
                            @Cast("CvPoint3D64f*") DoubleBuffer point);
public static native int icvCompute3DPoint(    double alpha,double betta,
                            CvStereoLineCoeff coeffs,
                            @Cast("CvPoint3D64f*") double[] point);

public static native int icvCreateConvertMatrVect( DoublePointer rotMatr1,
                                DoublePointer transVect1,
                                DoublePointer rotMatr2,
                                DoublePointer transVect2,
                                DoublePointer convRotMatr,
                                DoublePointer convTransVect);
public static native int icvCreateConvertMatrVect( DoubleBuffer rotMatr1,
                                DoubleBuffer transVect1,
                                DoubleBuffer rotMatr2,
                                DoubleBuffer transVect2,
                                DoubleBuffer convRotMatr,
                                DoubleBuffer convTransVect);
public static native int icvCreateConvertMatrVect( double[] rotMatr1,
                                double[] transVect1,
                                double[] rotMatr2,
                                double[] transVect2,
                                double[] convRotMatr,
                                double[] convTransVect);

public static native int icvConvertPointSystem(@ByVal CvPoint3D64f M2,
                            CvPoint3D64f M1,
                            DoublePointer rotMatr,
                            DoublePointer transVect
                            );
public static native int icvConvertPointSystem(@ByVal @Cast("CvPoint3D64f*") DoubleBuffer M2,
                            @Cast("CvPoint3D64f*") DoubleBuffer M1,
                            DoubleBuffer rotMatr,
                            DoubleBuffer transVect
                            );
public static native int icvConvertPointSystem(@ByVal @Cast("CvPoint3D64f*") double[] M2,
                            @Cast("CvPoint3D64f*") double[] M1,
                            double[] rotMatr,
                            double[] transVect
                            );

public static native int icvComputeCoeffForStereo(  CvStereoCamera stereoCamera);

public static native int icvGetCrossPieceVector(@ByVal CvPoint2D32f p1_start,@ByVal CvPoint2D32f p1_end,@ByVal CvPoint2D32f v2_start,@ByVal CvPoint2D32f v2_end,CvPoint2D32f cross);
public static native int icvGetCrossPieceVector(@ByVal @Cast("CvPoint2D32f*") FloatBuffer p1_start,@ByVal @Cast("CvPoint2D32f*") FloatBuffer p1_end,@ByVal @Cast("CvPoint2D32f*") FloatBuffer v2_start,@ByVal @Cast("CvPoint2D32f*") FloatBuffer v2_end,@Cast("CvPoint2D32f*") FloatBuffer cross);
public static native int icvGetCrossPieceVector(@ByVal @Cast("CvPoint2D32f*") float[] p1_start,@ByVal @Cast("CvPoint2D32f*") float[] p1_end,@ByVal @Cast("CvPoint2D32f*") float[] v2_start,@ByVal @Cast("CvPoint2D32f*") float[] v2_end,@Cast("CvPoint2D32f*") float[] cross);
public static native int icvGetCrossLineDirect(@ByVal CvPoint2D32f p1,@ByVal CvPoint2D32f p2,float a,float b,float c,CvPoint2D32f cross);
public static native int icvGetCrossLineDirect(@ByVal @Cast("CvPoint2D32f*") FloatBuffer p1,@ByVal @Cast("CvPoint2D32f*") FloatBuffer p2,float a,float b,float c,@Cast("CvPoint2D32f*") FloatBuffer cross);
public static native int icvGetCrossLineDirect(@ByVal @Cast("CvPoint2D32f*") float[] p1,@ByVal @Cast("CvPoint2D32f*") float[] p2,float a,float b,float c,@Cast("CvPoint2D32f*") float[] cross);
public static native float icvDefinePointPosition(@ByVal CvPoint2D32f point1,@ByVal CvPoint2D32f point2,@ByVal CvPoint2D32f point);
public static native float icvDefinePointPosition(@ByVal @Cast("CvPoint2D32f*") FloatBuffer point1,@ByVal @Cast("CvPoint2D32f*") FloatBuffer point2,@ByVal @Cast("CvPoint2D32f*") FloatBuffer point);
public static native float icvDefinePointPosition(@ByVal @Cast("CvPoint2D32f*") float[] point1,@ByVal @Cast("CvPoint2D32f*") float[] point2,@ByVal @Cast("CvPoint2D32f*") float[] point);
public static native int icvStereoCalibration( int numImages,
                            IntPointer nums,
                            @ByVal CvSize imageSize,
                            CvPoint2D32f imagePoints1,
                            CvPoint2D32f imagePoints2,
                            CvPoint3D32f objectPoints,
                            CvStereoCamera stereoparams
                           );
public static native int icvStereoCalibration( int numImages,
                            IntBuffer nums,
                            @ByVal CvSize imageSize,
                            @Cast("CvPoint2D32f*") FloatBuffer imagePoints1,
                            @Cast("CvPoint2D32f*") FloatBuffer imagePoints2,
                            @Cast("CvPoint3D32f*") FloatBuffer objectPoints,
                            CvStereoCamera stereoparams
                           );
public static native int icvStereoCalibration( int numImages,
                            int[] nums,
                            @ByVal CvSize imageSize,
                            @Cast("CvPoint2D32f*") float[] imagePoints1,
                            @Cast("CvPoint2D32f*") float[] imagePoints2,
                            @Cast("CvPoint3D32f*") float[] objectPoints,
                            CvStereoCamera stereoparams
                           );


public static native int icvComputeRestStereoParams(CvStereoCamera stereoparams);

public static native void cvComputePerspectiveMap( @Cast("const double(*)[3]") DoublePointer coeffs, CvArr rectMapX, CvArr rectMapY );
public static native void cvComputePerspectiveMap( @Cast("const double(*)[3]") DoubleBuffer coeffs, CvArr rectMapX, CvArr rectMapY );
public static native void cvComputePerspectiveMap( @Cast("const double(*)[3]") double[] coeffs, CvArr rectMapX, CvArr rectMapY );

public static native int icvComCoeffForLine(   @ByVal CvPoint2D64f point1,
                            @ByVal CvPoint2D64f point2,
                            @ByVal CvPoint2D64f point3,
                            @ByVal CvPoint2D64f point4,
                            DoublePointer camMatr1,
                            DoublePointer rotMatr1,
                            DoublePointer transVect1,
                            DoublePointer camMatr2,
                            DoublePointer rotMatr2,
                            DoublePointer transVect2,
                            CvStereoLineCoeff coeffs,
                            IntPointer needSwapCameras);
public static native int icvComCoeffForLine(   @ByVal @Cast("CvPoint2D64f*") DoubleBuffer point1,
                            @ByVal @Cast("CvPoint2D64f*") DoubleBuffer point2,
                            @ByVal @Cast("CvPoint2D64f*") DoubleBuffer point3,
                            @ByVal @Cast("CvPoint2D64f*") DoubleBuffer point4,
                            DoubleBuffer camMatr1,
                            DoubleBuffer rotMatr1,
                            DoubleBuffer transVect1,
                            DoubleBuffer camMatr2,
                            DoubleBuffer rotMatr2,
                            DoubleBuffer transVect2,
                            CvStereoLineCoeff coeffs,
                            IntBuffer needSwapCameras);
public static native int icvComCoeffForLine(   @ByVal @Cast("CvPoint2D64f*") double[] point1,
                            @ByVal @Cast("CvPoint2D64f*") double[] point2,
                            @ByVal @Cast("CvPoint2D64f*") double[] point3,
                            @ByVal @Cast("CvPoint2D64f*") double[] point4,
                            double[] camMatr1,
                            double[] rotMatr1,
                            double[] transVect1,
                            double[] camMatr2,
                            double[] rotMatr2,
                            double[] transVect2,
                            CvStereoLineCoeff coeffs,
                            int[] needSwapCameras);

public static native int icvGetDirectionForPoint(  @ByVal CvPoint2D64f point,
                                DoublePointer camMatr,
                                CvPoint3D64f direct);
public static native int icvGetDirectionForPoint(  @ByVal @Cast("CvPoint2D64f*") DoubleBuffer point,
                                DoubleBuffer camMatr,
                                @Cast("CvPoint3D64f*") DoubleBuffer direct);
public static native int icvGetDirectionForPoint(  @ByVal @Cast("CvPoint2D64f*") double[] point,
                                double[] camMatr,
                                @Cast("CvPoint3D64f*") double[] direct);

public static native int icvGetCrossLines(@ByVal CvPoint3D64f point11,@ByVal CvPoint3D64f point12,
                       @ByVal CvPoint3D64f point21,@ByVal CvPoint3D64f point22,
                       CvPoint3D64f midPoint);
public static native int icvGetCrossLines(@ByVal @Cast("CvPoint3D64f*") DoubleBuffer point11,@ByVal @Cast("CvPoint3D64f*") DoubleBuffer point12,
                       @ByVal @Cast("CvPoint3D64f*") DoubleBuffer point21,@ByVal @Cast("CvPoint3D64f*") DoubleBuffer point22,
                       @Cast("CvPoint3D64f*") DoubleBuffer midPoint);
public static native int icvGetCrossLines(@ByVal @Cast("CvPoint3D64f*") double[] point11,@ByVal @Cast("CvPoint3D64f*") double[] point12,
                       @ByVal @Cast("CvPoint3D64f*") double[] point21,@ByVal @Cast("CvPoint3D64f*") double[] point22,
                       @Cast("CvPoint3D64f*") double[] midPoint);

public static native int icvComputeStereoLineCoeffs(   @ByVal CvPoint3D64f pointA,
                                    @ByVal CvPoint3D64f pointB,
                                    @ByVal CvPoint3D64f pointCam1,
                                    double gamma,
                                    CvStereoLineCoeff coeffs);
public static native int icvComputeStereoLineCoeffs(   @ByVal @Cast("CvPoint3D64f*") DoubleBuffer pointA,
                                    @ByVal @Cast("CvPoint3D64f*") DoubleBuffer pointB,
                                    @ByVal @Cast("CvPoint3D64f*") DoubleBuffer pointCam1,
                                    double gamma,
                                    CvStereoLineCoeff coeffs);
public static native int icvComputeStereoLineCoeffs(   @ByVal @Cast("CvPoint3D64f*") double[] pointA,
                                    @ByVal @Cast("CvPoint3D64f*") double[] pointB,
                                    @ByVal @Cast("CvPoint3D64f*") double[] pointCam1,
                                    double gamma,
                                    CvStereoLineCoeff coeffs);

/*CVAPI(int) icvComputeFundMatrEpipoles ( double* camMatr1,
                                    double*     rotMatr1,
                                    double*     transVect1,
                                    double*     camMatr2,
                                    double*     rotMatr2,
                                    double*     transVect2,
                                    CvPoint2D64f* epipole1,
                                    CvPoint2D64f* epipole2,
                                    double*     fundMatr);*/

public static native int icvGetAngleLine( @ByVal CvPoint2D64f startPoint, @ByVal CvSize imageSize,CvPoint2D64f point1,CvPoint2D64f point2);
public static native int icvGetAngleLine( @ByVal @Cast("CvPoint2D64f*") DoubleBuffer startPoint, @ByVal CvSize imageSize,@Cast("CvPoint2D64f*") DoubleBuffer point1,@Cast("CvPoint2D64f*") DoubleBuffer point2);
public static native int icvGetAngleLine( @ByVal @Cast("CvPoint2D64f*") double[] startPoint, @ByVal CvSize imageSize,@Cast("CvPoint2D64f*") double[] point1,@Cast("CvPoint2D64f*") double[] point2);

public static native void icvGetCoefForPiece(   @ByVal CvPoint2D64f p_start,@ByVal CvPoint2D64f p_end,
                        DoublePointer a,DoublePointer b,DoublePointer c,
                        IntPointer result);
public static native void icvGetCoefForPiece(   @ByVal @Cast("CvPoint2D64f*") DoubleBuffer p_start,@ByVal @Cast("CvPoint2D64f*") DoubleBuffer p_end,
                        DoubleBuffer a,DoubleBuffer b,DoubleBuffer c,
                        IntBuffer result);
public static native void icvGetCoefForPiece(   @ByVal @Cast("CvPoint2D64f*") double[] p_start,@ByVal @Cast("CvPoint2D64f*") double[] p_end,
                        double[] a,double[] b,double[] c,
                        int[] result);

/*CVAPI(void) icvGetCommonArea( CvSize imageSize,
                    CvPoint2D64f epipole1,CvPoint2D64f epipole2,
                    double* fundMatr,
                    double* coeff11,double* coeff12,
                    double* coeff21,double* coeff22,
                    int* result);*/

public static native void icvComputeeInfiniteProject1(DoublePointer rotMatr,
                                     DoublePointer camMatr1,
                                     DoublePointer camMatr2,
                                     @ByVal CvPoint2D32f point1,
                                     CvPoint2D32f point2);
public static native void icvComputeeInfiniteProject1(DoubleBuffer rotMatr,
                                     DoubleBuffer camMatr1,
                                     DoubleBuffer camMatr2,
                                     @ByVal @Cast("CvPoint2D32f*") FloatBuffer point1,
                                     @Cast("CvPoint2D32f*") FloatBuffer point2);
public static native void icvComputeeInfiniteProject1(double[] rotMatr,
                                     double[] camMatr1,
                                     double[] camMatr2,
                                     @ByVal @Cast("CvPoint2D32f*") float[] point1,
                                     @Cast("CvPoint2D32f*") float[] point2);

public static native void icvComputeeInfiniteProject2(DoublePointer rotMatr,
                                     DoublePointer camMatr1,
                                     DoublePointer camMatr2,
                                     CvPoint2D32f point1,
                                     @ByVal CvPoint2D32f point2);
public static native void icvComputeeInfiniteProject2(DoubleBuffer rotMatr,
                                     DoubleBuffer camMatr1,
                                     DoubleBuffer camMatr2,
                                     @Cast("CvPoint2D32f*") FloatBuffer point1,
                                     @ByVal @Cast("CvPoint2D32f*") FloatBuffer point2);
public static native void icvComputeeInfiniteProject2(double[] rotMatr,
                                     double[] camMatr1,
                                     double[] camMatr2,
                                     @Cast("CvPoint2D32f*") float[] point1,
                                     @ByVal @Cast("CvPoint2D32f*") float[] point2);

public static native void icvGetCrossDirectDirect(  DoublePointer direct1,DoublePointer direct2,
                            CvPoint2D64f cross,IntPointer result);
public static native void icvGetCrossDirectDirect(  DoubleBuffer direct1,DoubleBuffer direct2,
                            @Cast("CvPoint2D64f*") DoubleBuffer cross,IntBuffer result);
public static native void icvGetCrossDirectDirect(  double[] direct1,double[] direct2,
                            @Cast("CvPoint2D64f*") double[] cross,int[] result);

public static native void icvGetCrossPieceDirect(   @ByVal CvPoint2D64f p_start,@ByVal CvPoint2D64f p_end,
                            double a,double b,double c,
                            CvPoint2D64f cross,IntPointer result);
public static native void icvGetCrossPieceDirect(   @ByVal @Cast("CvPoint2D64f*") DoubleBuffer p_start,@ByVal @Cast("CvPoint2D64f*") DoubleBuffer p_end,
                            double a,double b,double c,
                            @Cast("CvPoint2D64f*") DoubleBuffer cross,IntBuffer result);
public static native void icvGetCrossPieceDirect(   @ByVal @Cast("CvPoint2D64f*") double[] p_start,@ByVal @Cast("CvPoint2D64f*") double[] p_end,
                            double a,double b,double c,
                            @Cast("CvPoint2D64f*") double[] cross,int[] result);

public static native void icvGetCrossPiecePiece( @ByVal CvPoint2D64f p1_start,@ByVal CvPoint2D64f p1_end,
                            @ByVal CvPoint2D64f p2_start,@ByVal CvPoint2D64f p2_end,
                            CvPoint2D64f cross,
                            IntPointer result);
public static native void icvGetCrossPiecePiece( @ByVal @Cast("CvPoint2D64f*") DoubleBuffer p1_start,@ByVal @Cast("CvPoint2D64f*") DoubleBuffer p1_end,
                            @ByVal @Cast("CvPoint2D64f*") DoubleBuffer p2_start,@ByVal @Cast("CvPoint2D64f*") DoubleBuffer p2_end,
                            @Cast("CvPoint2D64f*") DoubleBuffer cross,
                            IntBuffer result);
public static native void icvGetCrossPiecePiece( @ByVal @Cast("CvPoint2D64f*") double[] p1_start,@ByVal @Cast("CvPoint2D64f*") double[] p1_end,
                            @ByVal @Cast("CvPoint2D64f*") double[] p2_start,@ByVal @Cast("CvPoint2D64f*") double[] p2_end,
                            @Cast("CvPoint2D64f*") double[] cross,
                            int[] result);

public static native void icvGetPieceLength(@ByVal CvPoint2D64f point1,@ByVal CvPoint2D64f point2,DoublePointer dist);
public static native void icvGetPieceLength(@ByVal @Cast("CvPoint2D64f*") DoubleBuffer point1,@ByVal @Cast("CvPoint2D64f*") DoubleBuffer point2,DoubleBuffer dist);
public static native void icvGetPieceLength(@ByVal @Cast("CvPoint2D64f*") double[] point1,@ByVal @Cast("CvPoint2D64f*") double[] point2,double[] dist);

public static native void icvGetCrossRectDirect(    @ByVal CvSize imageSize,
                            double a,double b,double c,
                            CvPoint2D64f start,CvPoint2D64f end,
                            IntPointer result);
public static native void icvGetCrossRectDirect(    @ByVal CvSize imageSize,
                            double a,double b,double c,
                            @Cast("CvPoint2D64f*") DoubleBuffer start,@Cast("CvPoint2D64f*") DoubleBuffer end,
                            IntBuffer result);
public static native void icvGetCrossRectDirect(    @ByVal CvSize imageSize,
                            double a,double b,double c,
                            @Cast("CvPoint2D64f*") double[] start,@Cast("CvPoint2D64f*") double[] end,
                            int[] result);

public static native void icvProjectPointToImage(   @ByVal CvPoint3D64f point,
                            DoublePointer camMatr,DoublePointer rotMatr,DoublePointer transVect,
                            CvPoint2D64f projPoint);
public static native void icvProjectPointToImage(   @ByVal @Cast("CvPoint3D64f*") DoubleBuffer point,
                            DoubleBuffer camMatr,DoubleBuffer rotMatr,DoubleBuffer transVect,
                            @Cast("CvPoint2D64f*") DoubleBuffer projPoint);
public static native void icvProjectPointToImage(   @ByVal @Cast("CvPoint3D64f*") double[] point,
                            double[] camMatr,double[] rotMatr,double[] transVect,
                            @Cast("CvPoint2D64f*") double[] projPoint);

public static native void icvGetQuadsTransform( @ByVal CvSize imageSize,
                        DoublePointer camMatr1,
                        DoublePointer rotMatr1,
                        DoublePointer transVect1,
                        DoublePointer camMatr2,
                        DoublePointer rotMatr2,
                        DoublePointer transVect2,
                        CvSize warpSize,
                        @Cast("double(*)[2]") DoublePointer quad1,
                        @Cast("double(*)[2]") DoublePointer quad2,
                        DoublePointer fundMatr,
                        CvPoint3D64f epipole1,
                        CvPoint3D64f epipole2
                        );
public static native void icvGetQuadsTransform( @ByVal CvSize imageSize,
                        DoubleBuffer camMatr1,
                        DoubleBuffer rotMatr1,
                        DoubleBuffer transVect1,
                        DoubleBuffer camMatr2,
                        DoubleBuffer rotMatr2,
                        DoubleBuffer transVect2,
                        CvSize warpSize,
                        @Cast("double(*)[2]") DoubleBuffer quad1,
                        @Cast("double(*)[2]") DoubleBuffer quad2,
                        DoubleBuffer fundMatr,
                        @Cast("CvPoint3D64f*") DoubleBuffer epipole1,
                        @Cast("CvPoint3D64f*") DoubleBuffer epipole2
                        );
public static native void icvGetQuadsTransform( @ByVal CvSize imageSize,
                        double[] camMatr1,
                        double[] rotMatr1,
                        double[] transVect1,
                        double[] camMatr2,
                        double[] rotMatr2,
                        double[] transVect2,
                        CvSize warpSize,
                        @Cast("double(*)[2]") double[] quad1,
                        @Cast("double(*)[2]") double[] quad2,
                        double[] fundMatr,
                        @Cast("CvPoint3D64f*") double[] epipole1,
                        @Cast("CvPoint3D64f*") double[] epipole2
                        );

public static native void icvGetQuadsTransformStruct(  CvStereoCamera stereoCamera);

public static native void icvComputeStereoParamsForCameras(CvStereoCamera stereoCamera);

public static native void icvGetCutPiece(   DoublePointer areaLineCoef1,DoublePointer areaLineCoef2,
                    @ByVal CvPoint2D64f epipole,
                    @ByVal CvSize imageSize,
                    CvPoint2D64f point11,CvPoint2D64f point12,
                    CvPoint2D64f point21,CvPoint2D64f point22,
                    IntPointer result);
public static native void icvGetCutPiece(   DoubleBuffer areaLineCoef1,DoubleBuffer areaLineCoef2,
                    @ByVal @Cast("CvPoint2D64f*") DoubleBuffer epipole,
                    @ByVal CvSize imageSize,
                    @Cast("CvPoint2D64f*") DoubleBuffer point11,@Cast("CvPoint2D64f*") DoubleBuffer point12,
                    @Cast("CvPoint2D64f*") DoubleBuffer point21,@Cast("CvPoint2D64f*") DoubleBuffer point22,
                    IntBuffer result);
public static native void icvGetCutPiece(   double[] areaLineCoef1,double[] areaLineCoef2,
                    @ByVal @Cast("CvPoint2D64f*") double[] epipole,
                    @ByVal CvSize imageSize,
                    @Cast("CvPoint2D64f*") double[] point11,@Cast("CvPoint2D64f*") double[] point12,
                    @Cast("CvPoint2D64f*") double[] point21,@Cast("CvPoint2D64f*") double[] point22,
                    int[] result);

public static native void icvGetMiddleAnglePoint(   @ByVal CvPoint2D64f basePoint,
                            @ByVal CvPoint2D64f point1,@ByVal CvPoint2D64f point2,
                            CvPoint2D64f midPoint);
public static native void icvGetMiddleAnglePoint(   @ByVal @Cast("CvPoint2D64f*") DoubleBuffer basePoint,
                            @ByVal @Cast("CvPoint2D64f*") DoubleBuffer point1,@ByVal @Cast("CvPoint2D64f*") DoubleBuffer point2,
                            @Cast("CvPoint2D64f*") DoubleBuffer midPoint);
public static native void icvGetMiddleAnglePoint(   @ByVal @Cast("CvPoint2D64f*") double[] basePoint,
                            @ByVal @Cast("CvPoint2D64f*") double[] point1,@ByVal @Cast("CvPoint2D64f*") double[] point2,
                            @Cast("CvPoint2D64f*") double[] midPoint);

public static native void icvGetNormalDirect(DoublePointer direct,@ByVal CvPoint2D64f point,DoublePointer normDirect);
public static native void icvGetNormalDirect(DoubleBuffer direct,@ByVal @Cast("CvPoint2D64f*") DoubleBuffer point,DoubleBuffer normDirect);
public static native void icvGetNormalDirect(double[] direct,@ByVal @Cast("CvPoint2D64f*") double[] point,double[] normDirect);

public static native double icvGetVect(@ByVal CvPoint2D64f basePoint,@ByVal CvPoint2D64f point1,@ByVal CvPoint2D64f point2);
public static native double icvGetVect(@ByVal @Cast("CvPoint2D64f*") DoubleBuffer basePoint,@ByVal @Cast("CvPoint2D64f*") DoubleBuffer point1,@ByVal @Cast("CvPoint2D64f*") DoubleBuffer point2);
public static native double icvGetVect(@ByVal @Cast("CvPoint2D64f*") double[] basePoint,@ByVal @Cast("CvPoint2D64f*") double[] point1,@ByVal @Cast("CvPoint2D64f*") double[] point2);

public static native void icvProjectPointToDirect(  @ByVal CvPoint2D64f point,DoublePointer lineCoeff,
                            CvPoint2D64f projectPoint);
public static native void icvProjectPointToDirect(  @ByVal @Cast("CvPoint2D64f*") DoubleBuffer point,DoubleBuffer lineCoeff,
                            @Cast("CvPoint2D64f*") DoubleBuffer projectPoint);
public static native void icvProjectPointToDirect(  @ByVal @Cast("CvPoint2D64f*") double[] point,double[] lineCoeff,
                            @Cast("CvPoint2D64f*") double[] projectPoint);

public static native void icvGetDistanceFromPointToDirect( @ByVal CvPoint2D64f point,DoublePointer lineCoef,DoublePointer dist);
public static native void icvGetDistanceFromPointToDirect( @ByVal @Cast("CvPoint2D64f*") DoubleBuffer point,DoubleBuffer lineCoef,DoubleBuffer dist);
public static native void icvGetDistanceFromPointToDirect( @ByVal @Cast("CvPoint2D64f*") double[] point,double[] lineCoef,double[] dist);

public static native IplImage icvCreateIsometricImage( IplImage src, IplImage dst,
                              int desired_depth, int desired_num_channels );

public static native void cvDeInterlace( @Const CvArr frame, CvArr fieldEven, CvArr fieldOdd );

/*CVAPI(int) icvSelectBestRt(           int           numImages,
                                    int*          numPoints,
                                    CvSize        imageSize,
                                    CvPoint2D32f* imagePoints1,
                                    CvPoint2D32f* imagePoints2,
                                    CvPoint3D32f* objectPoints,

                                    CvMatr32f     cameraMatrix1,
                                    CvVect32f     distortion1,
                                    CvMatr32f     rotMatrs1,
                                    CvVect32f     transVects1,

                                    CvMatr32f     cameraMatrix2,
                                    CvVect32f     distortion2,
                                    CvMatr32f     rotMatrs2,
                                    CvVect32f     transVects2,

                                    CvMatr32f     bestRotMatr,
                                    CvVect32f     bestTransVect
                                    );*/


/****************************************************************************************\
*                                     Contour Tree                                       *
\****************************************************************************************/

/* Contour tree header */
public static class CvContourTree extends CvSeq {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvContourTree() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvContourTree(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvContourTree(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvContourTree position(int position) {
        return (CvContourTree)super.position(position);
    }

    public native int flags(); public native CvContourTree flags(int flags);             /* Miscellaneous flags.     */
    public native int header_size(); public native CvContourTree header_size(int header_size);       /* Size of sequence header. */
    public native CvSeq h_prev(); public native CvContourTree h_prev(CvSeq h_prev); /* Previous sequence.       */
    public native CvSeq h_next(); public native CvContourTree h_next(CvSeq h_next); /* Next sequence.           */
    public native CvSeq v_prev(); public native CvContourTree v_prev(CvSeq v_prev); /* 2nd previous sequence.   */
    public native CvSeq v_next(); public native CvContourTree v_next(CvSeq v_next);
    public native int total(); public native CvContourTree total(int total);          /* Total number of elements.            */
    public native int elem_size(); public native CvContourTree elem_size(int elem_size);      /* Size of sequence element in bytes.   */
    public native @Cast("schar*") BytePointer block_max(); public native CvContourTree block_max(BytePointer block_max);      /* Maximal bound of the last block.     */
    public native @Cast("schar*") BytePointer ptr(); public native CvContourTree ptr(BytePointer ptr);            /* Current write pointer.               */
    public native int delta_elems(); public native CvContourTree delta_elems(int delta_elems);    /* Grow seq this many at a time.        */
    public native CvMemStorage storage(); public native CvContourTree storage(CvMemStorage storage);    /* Where the seq is stored.             */
    public native CvSeqBlock free_blocks(); public native CvContourTree free_blocks(CvSeqBlock free_blocks);  /* Free blocks list.                    */
    public native CvSeqBlock first(); public native CvContourTree first(CvSeqBlock first);        /* Pointer to the first sequence block. */
    public native @ByRef CvPoint p1(); public native CvContourTree p1(CvPoint p1);            /* the first point of the binary tree root segment */
    public native @ByRef CvPoint p2(); public native CvContourTree p2(CvPoint p2);            /* the last point of the binary tree root segment */
}

/* Builds hierarhical representation of a contour */
public static native CvContourTree cvCreateContourTree( @Const CvSeq contour,
                                            CvMemStorage storage,
                                            double threshold );

/* Reconstruct (completelly or partially) contour a from contour tree */
public static native CvSeq cvContourFromContourTree( @Const CvContourTree tree,
                                         CvMemStorage storage,
                                         @ByVal CvTermCriteria criteria );

/* Compares two contour trees */
/** enum  */
public static final int CV_CONTOUR_TREES_MATCH_I1 = 1;

public static native double cvMatchContourTrees( @Const CvContourTree tree1,
                                    @Const CvContourTree tree2,
                                    int method, double threshold );

/****************************************************************************************\
*                                   Contour Morphing                                     *
\****************************************************************************************/

/* finds correspondence between two contours */


/* morphs contours using the pre-calculated correspondence:
   alpha=0 ~ contour1, alpha=1 ~ contour2 */



/****************************************************************************************\
*                                   Active Contours                                      *
\****************************************************************************************/

public static final int CV_VALUE =  1;
public static final int CV_ARRAY =  2;
/* Updates active contour in order to minimize its cummulative
   (internal and external) energy. */
public static native void cvSnakeImage( @Const IplImage image, CvPoint points,
                           int length, FloatPointer alpha,
                           FloatPointer beta, FloatPointer gamma,
                           int coeff_usage, @ByVal CvSize win,
                           @ByVal CvTermCriteria criteria, int calc_gradient/*=1*/);
public static native void cvSnakeImage( @Const IplImage image, CvPoint points,
                           int length, FloatPointer alpha,
                           FloatPointer beta, FloatPointer gamma,
                           int coeff_usage, @ByVal CvSize win,
                           @ByVal CvTermCriteria criteria);
public static native void cvSnakeImage( @Const IplImage image, @Cast("CvPoint*") IntBuffer points,
                           int length, FloatBuffer alpha,
                           FloatBuffer beta, FloatBuffer gamma,
                           int coeff_usage, @ByVal CvSize win,
                           @ByVal CvTermCriteria criteria, int calc_gradient/*=1*/);
public static native void cvSnakeImage( @Const IplImage image, @Cast("CvPoint*") IntBuffer points,
                           int length, FloatBuffer alpha,
                           FloatBuffer beta, FloatBuffer gamma,
                           int coeff_usage, @ByVal CvSize win,
                           @ByVal CvTermCriteria criteria);
public static native void cvSnakeImage( @Const IplImage image, @Cast("CvPoint*") int[] points,
                           int length, float[] alpha,
                           float[] beta, float[] gamma,
                           int coeff_usage, @ByVal CvSize win,
                           @ByVal CvTermCriteria criteria, int calc_gradient/*=1*/);
public static native void cvSnakeImage( @Const IplImage image, @Cast("CvPoint*") int[] points,
                           int length, float[] alpha,
                           float[] beta, float[] gamma,
                           int coeff_usage, @ByVal CvSize win,
                           @ByVal CvTermCriteria criteria);

/****************************************************************************************\
*                                    Texture Descriptors                                 *
\****************************************************************************************/

public static final int CV_GLCM_OPTIMIZATION_NONE =                   -2;
public static final int CV_GLCM_OPTIMIZATION_LUT =                    -1;
public static final int CV_GLCM_OPTIMIZATION_HISTOGRAM =              0;

public static final int CV_GLCMDESC_OPTIMIZATION_ALLOWDOUBLENEST =    10;
public static final int CV_GLCMDESC_OPTIMIZATION_ALLOWTRIPLENEST =    11;
public static final int CV_GLCMDESC_OPTIMIZATION_HISTOGRAM =          4;

public static final int CV_GLCMDESC_ENTROPY =                         0;
public static final int CV_GLCMDESC_ENERGY =                          1;
public static final int CV_GLCMDESC_HOMOGENITY =                      2;
public static final int CV_GLCMDESC_CONTRAST =                        3;
public static final int CV_GLCMDESC_CLUSTERTENDENCY =                 4;
public static final int CV_GLCMDESC_CLUSTERSHADE =                    5;
public static final int CV_GLCMDESC_CORRELATION =                     6;
public static final int CV_GLCMDESC_CORRELATIONINFO1 =                7;
public static final int CV_GLCMDESC_CORRELATIONINFO2 =                8;
public static final int CV_GLCMDESC_MAXIMUMPROBABILITY =              9;

public static final int CV_GLCM_ALL =                                 0;
public static final int CV_GLCM_GLCM =                                1;
public static final int CV_GLCM_DESC =                                2;

@Opaque public static class CvGLCM extends AbstractCvGLCM {
    /** Empty constructor. */
    public CvGLCM() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvGLCM(Pointer p) { super(p); }
}

public static native CvGLCM cvCreateGLCM( @Const IplImage srcImage,
                                int stepMagnitude,
                                @Const IntPointer stepDirections/*=0*/,
                                int numStepDirections/*=0*/,
                                int optimizationType/*=CV_GLCM_OPTIMIZATION_NONE*/);
public static native CvGLCM cvCreateGLCM( @Const IplImage srcImage,
                                int stepMagnitude);
public static native CvGLCM cvCreateGLCM( @Const IplImage srcImage,
                                int stepMagnitude,
                                @Const IntBuffer stepDirections/*=0*/,
                                int numStepDirections/*=0*/,
                                int optimizationType/*=CV_GLCM_OPTIMIZATION_NONE*/);
public static native CvGLCM cvCreateGLCM( @Const IplImage srcImage,
                                int stepMagnitude,
                                @Const int[] stepDirections/*=0*/,
                                int numStepDirections/*=0*/,
                                int optimizationType/*=CV_GLCM_OPTIMIZATION_NONE*/);

public static native void cvReleaseGLCM( @Cast("CvGLCM**") PointerPointer GLCM, int flag/*=CV_GLCM_ALL*/);
public static native void cvReleaseGLCM( @ByPtrPtr CvGLCM GLCM);
public static native void cvReleaseGLCM( @ByPtrPtr CvGLCM GLCM, int flag/*=CV_GLCM_ALL*/);

public static native void cvCreateGLCMDescriptors( CvGLCM destGLCM,
                                        int descriptorOptimizationType/*=CV_GLCMDESC_OPTIMIZATION_ALLOWDOUBLENEST*/);
public static native void cvCreateGLCMDescriptors( CvGLCM destGLCM);

public static native double cvGetGLCMDescriptor( CvGLCM GLCM, int step, int descriptor );

public static native void cvGetGLCMDescriptorStatistics( CvGLCM GLCM, int descriptor,
                                              DoublePointer average, DoublePointer standardDeviation );
public static native void cvGetGLCMDescriptorStatistics( CvGLCM GLCM, int descriptor,
                                              DoubleBuffer average, DoubleBuffer standardDeviation );
public static native void cvGetGLCMDescriptorStatistics( CvGLCM GLCM, int descriptor,
                                              double[] average, double[] standardDeviation );

public static native IplImage cvCreateGLCMImage( CvGLCM GLCM, int step );

/****************************************************************************************\
*                                  Face eyes&mouth tracking                              *
\****************************************************************************************/


@Opaque public static class CvFaceTracker extends AbstractCvFaceTracker {
    /** Empty constructor. */
    public CvFaceTracker() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvFaceTracker(Pointer p) { super(p); }
}

public static final int CV_NUM_FACE_ELEMENTS =    3;
/** enum CV_FACE_ELEMENTS */
public static final int
    CV_FACE_MOUTH = 0,
    CV_FACE_LEFT_EYE = 1,
    CV_FACE_RIGHT_EYE = 2;

public static native CvFaceTracker cvInitFaceTracker(CvFaceTracker pFaceTracking, @Const IplImage imgGray,
                                                CvRect pRects, int nRects);
public static native int cvTrackFace( CvFaceTracker pFaceTracker, IplImage imgGray,
                              CvRect pRects, int nRects,
                              CvPoint ptRotate, DoublePointer dbAngleRotate);
public static native int cvTrackFace( CvFaceTracker pFaceTracker, IplImage imgGray,
                              CvRect pRects, int nRects,
                              @Cast("CvPoint*") IntBuffer ptRotate, DoubleBuffer dbAngleRotate);
public static native int cvTrackFace( CvFaceTracker pFaceTracker, IplImage imgGray,
                              CvRect pRects, int nRects,
                              @Cast("CvPoint*") int[] ptRotate, double[] dbAngleRotate);
public static native void cvReleaseFaceTracker(@Cast("CvFaceTracker**") PointerPointer ppFaceTracker);
public static native void cvReleaseFaceTracker(@ByPtrPtr CvFaceTracker ppFaceTracker);


public static class CvFaceData extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvFaceData() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvFaceData(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvFaceData(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvFaceData position(int position) {
        return (CvFaceData)super.position(position);
    }

    public native @ByRef CvRect MouthRect(); public native CvFaceData MouthRect(CvRect MouthRect);
    public native @ByRef CvRect LeftEyeRect(); public native CvFaceData LeftEyeRect(CvRect LeftEyeRect);
    public native @ByRef CvRect RightEyeRect(); public native CvFaceData RightEyeRect(CvRect RightEyeRect);
}





/****************************************************************************************\
*                                         3D Tracker                                     *
\****************************************************************************************/

public static class Cv3dTracker2dTrackedObject extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public Cv3dTracker2dTrackedObject() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public Cv3dTracker2dTrackedObject(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Cv3dTracker2dTrackedObject(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public Cv3dTracker2dTrackedObject position(int position) {
        return (Cv3dTracker2dTrackedObject)super.position(position);
    }

    public native int id(); public native Cv3dTracker2dTrackedObject id(int id);
    public native @ByRef CvPoint2D32f p(); public native Cv3dTracker2dTrackedObject p(CvPoint2D32f p); // pgruebele: So we do not loose precision, this needs to be float
}

public static native @ByVal Cv3dTracker2dTrackedObject cv3dTracker2dTrackedObject(int id, @ByVal CvPoint2D32f p);
public static native @ByVal Cv3dTracker2dTrackedObject cv3dTracker2dTrackedObject(int id, @ByVal @Cast("CvPoint2D32f*") FloatBuffer p);
public static native @ByVal Cv3dTracker2dTrackedObject cv3dTracker2dTrackedObject(int id, @ByVal @Cast("CvPoint2D32f*") float[] p);

public static class Cv3dTrackerTrackedObject extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public Cv3dTrackerTrackedObject() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public Cv3dTrackerTrackedObject(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Cv3dTrackerTrackedObject(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public Cv3dTrackerTrackedObject position(int position) {
        return (Cv3dTrackerTrackedObject)super.position(position);
    }

    public native int id(); public native Cv3dTrackerTrackedObject id(int id);
    public native @ByRef CvPoint3D32f p(); public native Cv3dTrackerTrackedObject p(CvPoint3D32f p);             // location of the tracked object
}

public static native @ByVal Cv3dTrackerTrackedObject cv3dTrackerTrackedObject(int id, @ByVal CvPoint3D32f p);
public static native @ByVal Cv3dTrackerTrackedObject cv3dTrackerTrackedObject(int id, @ByVal @Cast("CvPoint3D32f*") FloatBuffer p);
public static native @ByVal Cv3dTrackerTrackedObject cv3dTrackerTrackedObject(int id, @ByVal @Cast("CvPoint3D32f*") float[] p);

public static class Cv3dTrackerCameraInfo extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public Cv3dTrackerCameraInfo() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public Cv3dTrackerCameraInfo(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Cv3dTrackerCameraInfo(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public Cv3dTrackerCameraInfo position(int position) {
        return (Cv3dTrackerCameraInfo)super.position(position);
    }

    public native @Cast("CvBool") byte valid(); public native Cv3dTrackerCameraInfo valid(byte valid);
    public native float mat(int i, int j); public native Cv3dTrackerCameraInfo mat(int i, int j, float mat);
    @MemberGetter public native @Cast("float(*)[4]") FloatPointer mat();              /* maps camera coordinates to world coordinates */
    public native @ByRef CvPoint2D32f principal_point(); public native Cv3dTrackerCameraInfo principal_point(CvPoint2D32f principal_point); /* copied from intrinsics so this structure */
                                  /* has all the info we need */
}

public static class Cv3dTrackerCameraIntrinsics extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public Cv3dTrackerCameraIntrinsics() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public Cv3dTrackerCameraIntrinsics(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Cv3dTrackerCameraIntrinsics(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public Cv3dTrackerCameraIntrinsics position(int position) {
        return (Cv3dTrackerCameraIntrinsics)super.position(position);
    }

    public native @ByRef CvPoint2D32f principal_point(); public native Cv3dTrackerCameraIntrinsics principal_point(CvPoint2D32f principal_point);
    public native float focal_length(int i); public native Cv3dTrackerCameraIntrinsics focal_length(int i, float focal_length);
    @MemberGetter public native FloatPointer focal_length();
    public native float distortion(int i); public native Cv3dTrackerCameraIntrinsics distortion(int i, float distortion);
    @MemberGetter public native FloatPointer distortion();
}

public static native @Cast("CvBool") byte cv3dTrackerCalibrateCameras(int num_cameras,
                     @Const Cv3dTrackerCameraIntrinsics camera_intrinsics,
                     @ByVal CvSize etalon_size,
                     float square_size,
                     @Cast("IplImage**") PointerPointer samples,
                     Cv3dTrackerCameraInfo camera_info);
public static native @Cast("CvBool") byte cv3dTrackerCalibrateCameras(int num_cameras,
                     @Const Cv3dTrackerCameraIntrinsics camera_intrinsics,
                     @ByVal CvSize etalon_size,
                     float square_size,
                     @ByPtrPtr IplImage samples,
                     Cv3dTrackerCameraInfo camera_info);                  /* size is num_cameras */

public static native int cv3dTrackerLocateObjects(int num_cameras, int num_objects,
                   @Const Cv3dTrackerCameraInfo camera_info,
                   @Const Cv3dTracker2dTrackedObject tracking_info,
                   Cv3dTrackerTrackedObject tracked_objects);      /* size is num_objects */
/****************************************************************************************
 tracking_info is a rectangular array; one row per camera, num_objects elements per row.
 The id field of any unused slots must be -1. Ids need not be ordered or consecutive. On
 completion, the return value is the number of objects located; i.e., the number of objects
 visible by more than one camera. The id field of any unused slots in tracked objects is
 set to -1.
****************************************************************************************/


/****************************************************************************************\
*                           Skeletons and Linear-Contour Models                          *
\****************************************************************************************/

/** enum CvLeeParameters */
public static final int
    CV_LEE_INT = 0,
    CV_LEE_FLOAT = 1,
    CV_LEE_DOUBLE = 2,
    CV_LEE_AUTO = -1,
    CV_LEE_ERODE = 0,
    CV_LEE_ZOOM = 1,
    CV_LEE_NON = 2;

// #define CV_NEXT_VORONOISITE2D( SITE ) ((SITE)->edge[0]->site[((SITE)->edge[0]->site[0] == (SITE))])
// #define CV_PREV_VORONOISITE2D( SITE ) ((SITE)->edge[1]->site[((SITE)->edge[1]->site[0] == (SITE))])
// #define CV_FIRST_VORONOIEDGE2D( SITE ) ((SITE)->edge[0])
// #define CV_LAST_VORONOIEDGE2D( SITE ) ((SITE)->edge[1])
// #define CV_NEXT_VORONOIEDGE2D( EDGE, SITE ) ((EDGE)->next[(EDGE)->site[0] != (SITE)])
// #define CV_PREV_VORONOIEDGE2D( EDGE, SITE ) ((EDGE)->next[2 + ((EDGE)->site[0] != (SITE))])
// #define CV_VORONOIEDGE2D_BEGINNODE( EDGE, SITE ) ((EDGE)->node[((EDGE)->site[0] != (SITE))])
// #define CV_VORONOIEDGE2D_ENDNODE( EDGE, SITE ) ((EDGE)->node[((EDGE)->site[0] == (SITE))])
// #define CV_TWIN_VORONOISITE2D( SITE, EDGE ) ( (EDGE)->site[((EDGE)->site[0] == (SITE))])

// #define CV_VORONOISITE2D_FIELDS()
//     struct CvVoronoiNode2D *node[2];
//     struct CvVoronoiEdge2D *edge[2];

public static class CvVoronoiSite2D extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvVoronoiSite2D() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvVoronoiSite2D(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvVoronoiSite2D(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvVoronoiSite2D position(int position) {
        return (CvVoronoiSite2D)super.position(position);
    }

    public native CvVoronoiNode2D node(int i); public native CvVoronoiSite2D node(int i, CvVoronoiNode2D node);
    @MemberGetter public native @Cast("CvVoronoiNode2D**") PointerPointer node();
    public native CvVoronoiEdge2D edge(int i); public native CvVoronoiSite2D edge(int i, CvVoronoiEdge2D edge);
    @MemberGetter public native @Cast("CvVoronoiEdge2D**") PointerPointer edge();
    public native CvVoronoiSite2D next(int i); public native CvVoronoiSite2D next(int i, CvVoronoiSite2D next);
    @MemberGetter public native @Cast("CvVoronoiSite2D**") PointerPointer next();
}

// #define CV_VORONOIEDGE2D_FIELDS()
//     struct CvVoronoiNode2D *node[2];
//     struct CvVoronoiSite2D *site[2];
//     struct CvVoronoiEdge2D *next[4];

public static class CvVoronoiEdge2D extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvVoronoiEdge2D() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvVoronoiEdge2D(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvVoronoiEdge2D(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvVoronoiEdge2D position(int position) {
        return (CvVoronoiEdge2D)super.position(position);
    }

    public native CvVoronoiNode2D node(int i); public native CvVoronoiEdge2D node(int i, CvVoronoiNode2D node);
    @MemberGetter public native @Cast("CvVoronoiNode2D**") PointerPointer node();
    public native CvVoronoiSite2D site(int i); public native CvVoronoiEdge2D site(int i, CvVoronoiSite2D site);
    @MemberGetter public native @Cast("CvVoronoiSite2D**") PointerPointer site();
    public native CvVoronoiEdge2D next(int i); public native CvVoronoiEdge2D next(int i, CvVoronoiEdge2D next);
    @MemberGetter public native @Cast("CvVoronoiEdge2D**") PointerPointer next();
}

// #define CV_VORONOINODE2D_FIELDS()
//     CV_SET_ELEM_FIELDS(CvVoronoiNode2D)
//     CvPoint2D32f pt;
//     float radius;

public static class CvVoronoiNode2D extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvVoronoiNode2D() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvVoronoiNode2D(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvVoronoiNode2D(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvVoronoiNode2D position(int position) {
        return (CvVoronoiNode2D)super.position(position);
    }

    public native int flags(); public native CvVoronoiNode2D flags(int flags);
    public native CvVoronoiNode2D next_free(); public native CvVoronoiNode2D next_free(CvVoronoiNode2D next_free);
    public native @ByRef CvPoint2D32f pt(); public native CvVoronoiNode2D pt(CvPoint2D32f pt);
    public native float radius(); public native CvVoronoiNode2D radius(float radius);
}

// #define CV_VORONOIDIAGRAM2D_FIELDS()
//     CV_GRAPH_FIELDS()
//     CvSet *sites;

public static class CvVoronoiDiagram2D extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvVoronoiDiagram2D() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvVoronoiDiagram2D(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvVoronoiDiagram2D(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvVoronoiDiagram2D position(int position) {
        return (CvVoronoiDiagram2D)super.position(position);
    }

    public native int flags(); public native CvVoronoiDiagram2D flags(int flags);             /* Miscellaneous flags.     */
    public native int header_size(); public native CvVoronoiDiagram2D header_size(int header_size);       /* Size of sequence header. */
    public native CvSeq h_prev(); public native CvVoronoiDiagram2D h_prev(CvSeq h_prev); /* Previous sequence.       */
    public native CvSeq h_next(); public native CvVoronoiDiagram2D h_next(CvSeq h_next); /* Next sequence.           */
    public native CvSeq v_prev(); public native CvVoronoiDiagram2D v_prev(CvSeq v_prev); /* 2nd previous sequence.   */
    public native CvSeq v_next(); public native CvVoronoiDiagram2D v_next(CvSeq v_next);
    public native int total(); public native CvVoronoiDiagram2D total(int total);          /* Total number of elements.            */
    public native int elem_size(); public native CvVoronoiDiagram2D elem_size(int elem_size);      /* Size of sequence element in bytes.   */
    public native @Cast("schar*") BytePointer block_max(); public native CvVoronoiDiagram2D block_max(BytePointer block_max);      /* Maximal bound of the last block.     */
    public native @Cast("schar*") BytePointer ptr(); public native CvVoronoiDiagram2D ptr(BytePointer ptr);            /* Current write pointer.               */
    public native int delta_elems(); public native CvVoronoiDiagram2D delta_elems(int delta_elems);    /* Grow seq this many at a time.        */
    public native CvMemStorage storage(); public native CvVoronoiDiagram2D storage(CvMemStorage storage);    /* Where the seq is stored.             */
    public native CvSeqBlock free_blocks(); public native CvVoronoiDiagram2D free_blocks(CvSeqBlock free_blocks);  /* Free blocks list.                    */
    public native CvSeqBlock first(); public native CvVoronoiDiagram2D first(CvSeqBlock first);        /* Pointer to the first sequence block. */
    public native CvSetElem free_elems(); public native CvVoronoiDiagram2D free_elems(CvSetElem free_elems);
    public native int active_count(); public native CvVoronoiDiagram2D active_count(int active_count);
    public native CvSet edges(); public native CvVoronoiDiagram2D edges(CvSet edges);
    public native CvSet sites(); public native CvVoronoiDiagram2D sites(CvSet sites);
}

/* Computes Voronoi Diagram for given polygons with holes */
public static native int cvVoronoiDiagramFromContour(CvSeq ContourSeq,
                                           @Cast("CvVoronoiDiagram2D**") PointerPointer VoronoiDiagram,
                                           CvMemStorage VoronoiStorage,
                                           @Cast("CvLeeParameters") int contour_type/*=CV_LEE_INT*/,
                                           int contour_orientation/*=-1*/,
                                           int attempt_number/*=10*/);
public static native int cvVoronoiDiagramFromContour(CvSeq ContourSeq,
                                           @ByPtrPtr CvVoronoiDiagram2D VoronoiDiagram,
                                           CvMemStorage VoronoiStorage);
public static native int cvVoronoiDiagramFromContour(CvSeq ContourSeq,
                                           @ByPtrPtr CvVoronoiDiagram2D VoronoiDiagram,
                                           CvMemStorage VoronoiStorage,
                                           @Cast("CvLeeParameters") int contour_type/*=CV_LEE_INT*/,
                                           int contour_orientation/*=-1*/,
                                           int attempt_number/*=10*/);

/* Computes Voronoi Diagram for domains in given image */
public static native int cvVoronoiDiagramFromImage(IplImage pImage,
                                         @Cast("CvSeq**") PointerPointer ContourSeq,
                                         @Cast("CvVoronoiDiagram2D**") PointerPointer VoronoiDiagram,
                                         CvMemStorage VoronoiStorage,
                                         @Cast("CvLeeParameters") int regularization_method/*=CV_LEE_NON*/,
                                         float approx_precision/*=CV_LEE_AUTO*/);
public static native int cvVoronoiDiagramFromImage(IplImage pImage,
                                         @ByPtrPtr CvSeq ContourSeq,
                                         @ByPtrPtr CvVoronoiDiagram2D VoronoiDiagram,
                                         CvMemStorage VoronoiStorage);
public static native int cvVoronoiDiagramFromImage(IplImage pImage,
                                         @ByPtrPtr CvSeq ContourSeq,
                                         @ByPtrPtr CvVoronoiDiagram2D VoronoiDiagram,
                                         CvMemStorage VoronoiStorage,
                                         @Cast("CvLeeParameters") int regularization_method/*=CV_LEE_NON*/,
                                         float approx_precision/*=CV_LEE_AUTO*/);

/* Deallocates the storage */
public static native void cvReleaseVoronoiStorage(CvVoronoiDiagram2D VoronoiDiagram,
                                          @Cast("CvMemStorage**") PointerPointer pVoronoiStorage);
public static native void cvReleaseVoronoiStorage(CvVoronoiDiagram2D VoronoiDiagram,
                                          @ByPtrPtr CvMemStorage pVoronoiStorage);

/*********************** Linear-Contour Model ****************************/

public static class CvLCMEdge extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvLCMEdge() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvLCMEdge(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvLCMEdge(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvLCMEdge position(int position) {
        return (CvLCMEdge)super.position(position);
    }

    public native int flags(); public native CvLCMEdge flags(int flags);
    public native float weight(); public native CvLCMEdge weight(float weight);
    public native CvGraphEdge next(int i); public native CvLCMEdge next(int i, CvGraphEdge next);
    @MemberGetter public native @Cast("CvGraphEdge**") PointerPointer next();
    public native CvGraphVtx vtx(int i); public native CvLCMEdge vtx(int i, CvGraphVtx vtx);
    @MemberGetter public native @Cast("CvGraphVtx**") PointerPointer vtx();
    public native CvSeq chain(); public native CvLCMEdge chain(CvSeq chain);
    public native float width(); public native CvLCMEdge width(float width);
    public native int index1(); public native CvLCMEdge index1(int index1);
    public native int index2(); public native CvLCMEdge index2(int index2);
}

public static class CvLCMNode extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvLCMNode() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvLCMNode(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvLCMNode(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvLCMNode position(int position) {
        return (CvLCMNode)super.position(position);
    }

    public native int flags(); public native CvLCMNode flags(int flags);
    public native CvGraphEdge first(); public native CvLCMNode first(CvGraphEdge first);
    public native CvContour contour(); public native CvLCMNode contour(CvContour contour);
}


/* Computes hybrid model from Voronoi Diagram */
public static native CvGraph cvLinearContorModelFromVoronoiDiagram(CvVoronoiDiagram2D VoronoiDiagram,
                                                         float maxWidth);

/* Releases hybrid model storage */
public static native int cvReleaseLinearContorModelStorage(@Cast("CvGraph**") PointerPointer Graph);
public static native int cvReleaseLinearContorModelStorage(@ByPtrPtr CvGraph Graph);


/* two stereo-related functions */

public static native void cvInitPerspectiveTransform( @ByVal CvSize size, @Const CvPoint2D32f vertex, @Cast("double(*)[3]") DoublePointer matrix,
                                              CvArr rectMap );
public static native void cvInitPerspectiveTransform( @ByVal CvSize size, @Cast("const CvPoint2D32f*") FloatBuffer vertex, @Cast("double(*)[3]") DoubleBuffer matrix,
                                              CvArr rectMap );
public static native void cvInitPerspectiveTransform( @ByVal CvSize size, @Cast("const CvPoint2D32f*") float[] vertex, @Cast("double(*)[3]") double[] matrix,
                                              CvArr rectMap );

/*CVAPI(void) cvInitStereoRectification( CvStereoCamera* params,
                                             CvArr* rectMap1, CvArr* rectMap2,
                                             int do_undistortion );*/

/*************************** View Morphing Functions ************************/

public static class CvMatrix3 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvMatrix3() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvMatrix3(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvMatrix3(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvMatrix3 position(int position) {
        return (CvMatrix3)super.position(position);
    }

    public native float m(int i, int j); public native CvMatrix3 m(int i, int j, float m);
    @MemberGetter public native @Cast("float(*)[3]") FloatPointer m();
}

/* The order of the function corresponds to the order they should appear in
   the view morphing pipeline */

/* Finds ending points of scanlines on left and right images of stereo-pair */
public static native void cvMakeScanlines( @Const CvMatrix3 matrix, @ByVal CvSize img_size,
                              IntPointer scanlines1, IntPointer scanlines2,
                              IntPointer lengths1, IntPointer lengths2,
                              IntPointer line_count );
public static native void cvMakeScanlines( @Const CvMatrix3 matrix, @ByVal CvSize img_size,
                              IntBuffer scanlines1, IntBuffer scanlines2,
                              IntBuffer lengths1, IntBuffer lengths2,
                              IntBuffer line_count );
public static native void cvMakeScanlines( @Const CvMatrix3 matrix, @ByVal CvSize img_size,
                              int[] scanlines1, int[] scanlines2,
                              int[] lengths1, int[] lengths2,
                              int[] line_count );

/* Grab pixel values from scanlines and stores them sequentially
   (some sort of perspective image transform) */
public static native void cvPreWarpImage( int line_count,
                             IplImage img,
                             @Cast("uchar*") BytePointer dst,
                             IntPointer dst_nums,
                             IntPointer scanlines);
public static native void cvPreWarpImage( int line_count,
                             IplImage img,
                             @Cast("uchar*") ByteBuffer dst,
                             IntBuffer dst_nums,
                             IntBuffer scanlines);
public static native void cvPreWarpImage( int line_count,
                             IplImage img,
                             @Cast("uchar*") byte[] dst,
                             int[] dst_nums,
                             int[] scanlines);

/* Approximate each grabbed scanline by a sequence of runs
   (lossy run-length compression) */
public static native void cvFindRuns( int line_count,
                         @Cast("uchar*") BytePointer prewarp1,
                         @Cast("uchar*") BytePointer prewarp2,
                         IntPointer line_lengths1,
                         IntPointer line_lengths2,
                         IntPointer runs1,
                         IntPointer runs2,
                         IntPointer num_runs1,
                         IntPointer num_runs2);
public static native void cvFindRuns( int line_count,
                         @Cast("uchar*") ByteBuffer prewarp1,
                         @Cast("uchar*") ByteBuffer prewarp2,
                         IntBuffer line_lengths1,
                         IntBuffer line_lengths2,
                         IntBuffer runs1,
                         IntBuffer runs2,
                         IntBuffer num_runs1,
                         IntBuffer num_runs2);
public static native void cvFindRuns( int line_count,
                         @Cast("uchar*") byte[] prewarp1,
                         @Cast("uchar*") byte[] prewarp2,
                         int[] line_lengths1,
                         int[] line_lengths2,
                         int[] runs1,
                         int[] runs2,
                         int[] num_runs1,
                         int[] num_runs2);

/* Compares two sets of compressed scanlines */
public static native void cvDynamicCorrespondMulti( int line_count,
                                       IntPointer first,
                                       IntPointer first_runs,
                                       IntPointer second,
                                       IntPointer second_runs,
                                       IntPointer first_corr,
                                       IntPointer second_corr);
public static native void cvDynamicCorrespondMulti( int line_count,
                                       IntBuffer first,
                                       IntBuffer first_runs,
                                       IntBuffer second,
                                       IntBuffer second_runs,
                                       IntBuffer first_corr,
                                       IntBuffer second_corr);
public static native void cvDynamicCorrespondMulti( int line_count,
                                       int[] first,
                                       int[] first_runs,
                                       int[] second,
                                       int[] second_runs,
                                       int[] first_corr,
                                       int[] second_corr);

/* Finds scanline ending coordinates for some intermediate "virtual" camera position */
public static native void cvMakeAlphaScanlines( IntPointer scanlines1,
                                   IntPointer scanlines2,
                                   IntPointer scanlinesA,
                                   IntPointer lengths,
                                   int line_count,
                                   float alpha);
public static native void cvMakeAlphaScanlines( IntBuffer scanlines1,
                                   IntBuffer scanlines2,
                                   IntBuffer scanlinesA,
                                   IntBuffer lengths,
                                   int line_count,
                                   float alpha);
public static native void cvMakeAlphaScanlines( int[] scanlines1,
                                   int[] scanlines2,
                                   int[] scanlinesA,
                                   int[] lengths,
                                   int line_count,
                                   float alpha);

/* Blends data of the left and right image scanlines to get
   pixel values of "virtual" image scanlines */
public static native void cvMorphEpilinesMulti( int line_count,
                                   @Cast("uchar*") BytePointer first_pix,
                                   IntPointer first_num,
                                   @Cast("uchar*") BytePointer second_pix,
                                   IntPointer second_num,
                                   @Cast("uchar*") BytePointer dst_pix,
                                   IntPointer dst_num,
                                   float alpha,
                                   IntPointer first,
                                   IntPointer first_runs,
                                   IntPointer second,
                                   IntPointer second_runs,
                                   IntPointer first_corr,
                                   IntPointer second_corr);
public static native void cvMorphEpilinesMulti( int line_count,
                                   @Cast("uchar*") ByteBuffer first_pix,
                                   IntBuffer first_num,
                                   @Cast("uchar*") ByteBuffer second_pix,
                                   IntBuffer second_num,
                                   @Cast("uchar*") ByteBuffer dst_pix,
                                   IntBuffer dst_num,
                                   float alpha,
                                   IntBuffer first,
                                   IntBuffer first_runs,
                                   IntBuffer second,
                                   IntBuffer second_runs,
                                   IntBuffer first_corr,
                                   IntBuffer second_corr);
public static native void cvMorphEpilinesMulti( int line_count,
                                   @Cast("uchar*") byte[] first_pix,
                                   int[] first_num,
                                   @Cast("uchar*") byte[] second_pix,
                                   int[] second_num,
                                   @Cast("uchar*") byte[] dst_pix,
                                   int[] dst_num,
                                   float alpha,
                                   int[] first,
                                   int[] first_runs,
                                   int[] second,
                                   int[] second_runs,
                                   int[] first_corr,
                                   int[] second_corr);

/* Does reverse warping of the morphing result to make
   it fill the destination image rectangle */
public static native void cvPostWarpImage( int line_count,
                              @Cast("uchar*") BytePointer src,
                              IntPointer src_nums,
                              IplImage img,
                              IntPointer scanlines);
public static native void cvPostWarpImage( int line_count,
                              @Cast("uchar*") ByteBuffer src,
                              IntBuffer src_nums,
                              IplImage img,
                              IntBuffer scanlines);
public static native void cvPostWarpImage( int line_count,
                              @Cast("uchar*") byte[] src,
                              int[] src_nums,
                              IplImage img,
                              int[] scanlines);

/* Deletes Moire (missed pixels that appear due to discretization) */
public static native void cvDeleteMoire( IplImage img );


public static class CvConDensation extends AbstractCvConDensation {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvConDensation() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvConDensation(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvConDensation(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvConDensation position(int position) {
        return (CvConDensation)super.position(position);
    }

    public native int MP(); public native CvConDensation MP(int MP);
    public native int DP(); public native CvConDensation DP(int DP);
    public native FloatPointer DynamMatr(); public native CvConDensation DynamMatr(FloatPointer DynamMatr);       /* Matrix of the linear Dynamics system  */
    public native FloatPointer State(); public native CvConDensation State(FloatPointer State);           /* Vector of State                       */
    public native int SamplesNum(); public native CvConDensation SamplesNum(int SamplesNum);         /* Number of the Samples                 */
    public native FloatPointer flSamples(int i); public native CvConDensation flSamples(int i, FloatPointer flSamples);
    @MemberGetter public native @Cast("float**") PointerPointer flSamples();      /* arr of the Sample Vectors             */
    public native FloatPointer flNewSamples(int i); public native CvConDensation flNewSamples(int i, FloatPointer flNewSamples);
    @MemberGetter public native @Cast("float**") PointerPointer flNewSamples();   /* temporary array of the Sample Vectors */
    public native FloatPointer flConfidence(); public native CvConDensation flConfidence(FloatPointer flConfidence);    /* Confidence for each Sample            */
    public native FloatPointer flCumulative(); public native CvConDensation flCumulative(FloatPointer flCumulative);    /* Cumulative confidence                 */
    public native FloatPointer Temp(); public native CvConDensation Temp(FloatPointer Temp);            /* Temporary vector                      */
    public native FloatPointer RandomSample(); public native CvConDensation RandomSample(FloatPointer RandomSample);    /* RandomVector to update sample set     */
    public native CvRandState RandS(); public native CvConDensation RandS(CvRandState RandS); /* Array of structures to generate random vectors */
}

/* Creates ConDensation filter state */
public static native CvConDensation cvCreateConDensation( int dynam_params,
                                             int measure_params,
                                             int sample_count );

/* Releases ConDensation filter state */
public static native void cvReleaseConDensation( @Cast("CvConDensation**") PointerPointer condens );
public static native void cvReleaseConDensation( @ByPtrPtr CvConDensation condens );

/* Updates ConDensation filter by time (predict future state of the system) */
public static native void cvConDensUpdateByTime( CvConDensation condens);

/* Initializes ConDensation filter samples  */
public static native void cvConDensInitSampleSet( CvConDensation condens, CvMat lower_bound, CvMat upper_bound );

public static native int iplWidth( @Const IplImage img );

public static native int iplHeight( @Const IplImage img );

// #ifdef __cplusplus
// #endif

// #ifdef __cplusplus

/****************************************************************************************\
*                                   Calibration engine                                   *
\****************************************************************************************/

/** enum CvCalibEtalonType */
public static final int
    CV_CALIB_ETALON_USER = -1,
    CV_CALIB_ETALON_CHESSBOARD = 0,
    CV_CALIB_ETALON_CHECKERBOARD =  CV_CALIB_ETALON_CHESSBOARD;

@NoOffset public static class CvCalibFilter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvCalibFilter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvCalibFilter(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CvCalibFilter position(int position) {
        return (CvCalibFilter)super.position(position);
    }

    /* Constructor & destructor */
    public CvCalibFilter() { allocate(); }
    private native void allocate();

    /* Sets etalon type - one for all cameras.
       etalonParams is used in case of pre-defined etalons (such as chessboard).
       Number of elements in etalonParams is determined by etalonType.
       E.g., if etalon type is CV_ETALON_TYPE_CHESSBOARD then:
         etalonParams[0] is number of squares per one side of etalon
         etalonParams[1] is number of squares per another side of etalon
         etalonParams[2] is linear size of squares in the board in arbitrary units.
       pointCount & points are used in case of
       CV_CALIB_ETALON_USER (user-defined) etalon. */
    public native @Cast("bool") boolean SetEtalon( @Cast("CvCalibEtalonType") int etalonType, DoublePointer etalonParams,
                       int pointCount/*=0*/, CvPoint2D32f points/*=0*/ );
    public native @Cast("bool") boolean SetEtalon( @Cast("CvCalibEtalonType") int etalonType, DoublePointer etalonParams );
    public native @Cast("bool") boolean SetEtalon( @Cast("CvCalibEtalonType") int etalonType, DoubleBuffer etalonParams,
                       int pointCount/*=0*/, @Cast("CvPoint2D32f*") FloatBuffer points/*=0*/ );
    public native @Cast("bool") boolean SetEtalon( @Cast("CvCalibEtalonType") int etalonType, DoubleBuffer etalonParams );
    public native @Cast("bool") boolean SetEtalon( @Cast("CvCalibEtalonType") int etalonType, double[] etalonParams,
                       int pointCount/*=0*/, @Cast("CvPoint2D32f*") float[] points/*=0*/ );
    public native @Cast("bool") boolean SetEtalon( @Cast("CvCalibEtalonType") int etalonType, double[] etalonParams );

    /* Retrieves etalon parameters/or and points */
    public native @Cast("CvCalibEtalonType") int GetEtalon( IntPointer paramCount/*=0*/, @Cast("const double**") PointerPointer etalonParams/*=0*/,
                       IntPointer pointCount/*=0*/, @Cast("const CvPoint2D32f**") PointerPointer etalonPoints/*=0*/ );
    public native @Cast("CvCalibEtalonType") int GetEtalon( );
    public native @Cast("CvCalibEtalonType") int GetEtalon( IntPointer paramCount/*=0*/, @Const @ByPtrPtr DoublePointer etalonParams/*=0*/,
                       IntPointer pointCount/*=0*/, @Const @ByPtrPtr CvPoint2D32f etalonPoints/*=0*/ );
    public native @Cast("CvCalibEtalonType") int GetEtalon( IntBuffer paramCount/*=0*/, @Const @ByPtrPtr DoubleBuffer etalonParams/*=0*/,
                       IntBuffer pointCount/*=0*/, @Cast("const CvPoint2D32f**") @ByPtrPtr FloatBuffer etalonPoints/*=0*/ );
    public native @Cast("CvCalibEtalonType") int GetEtalon( int[] paramCount/*=0*/, @Const @ByPtrPtr double[] etalonParams/*=0*/,
                       int[] pointCount/*=0*/, @Cast("const CvPoint2D32f**") @ByPtrPtr float[] etalonPoints/*=0*/ );

    /* Sets number of cameras calibrated simultaneously. It is equal to 1 initially */
    public native void SetCameraCount( int cameraCount );

    /* Retrieves number of cameras */
    public native int GetCameraCount();

    /* Starts cameras calibration */
    public native @Cast("bool") boolean SetFrames( int totalFrames );

    /* Stops cameras calibration */
    public native void Stop( @Cast("bool") boolean calibrate/*=false*/ );
    public native void Stop( );

    /* Retrieves number of cameras */
    public native @Cast("bool") boolean IsCalibrated();

    /* Feeds another serie of snapshots (one per each camera) to filter.
       Etalon points on these images are found automatically.
       If the function can't locate points, it returns false */
    public native @Cast("bool") boolean FindEtalon( @Cast("IplImage**") PointerPointer imgs );
    public native @Cast("bool") boolean FindEtalon( @ByPtrPtr IplImage imgs );

    /* The same but takes matrices */
    public native @Cast("bool") boolean FindEtalon( @ByPtrPtr CvMat imgs );

    /* Lower-level function for feeding filter with already found etalon points.
       Array of point arrays for each camera is passed. */
    public native @Cast("bool") boolean Push( @Cast("const CvPoint2D32f**") PointerPointer points/*=0*/ );
    public native @Cast("bool") boolean Push( );
    public native @Cast("bool") boolean Push( @Const @ByPtrPtr CvPoint2D32f points/*=0*/ );
    public native @Cast("bool") boolean Push( @Cast("const CvPoint2D32f**") @ByPtrPtr FloatBuffer points/*=0*/ );
    public native @Cast("bool") boolean Push( @Cast("const CvPoint2D32f**") @ByPtrPtr float[] points/*=0*/ );

    /* Returns total number of accepted frames and, optionally,
       total number of frames to collect */
    public native int GetFrameCount( IntPointer framesTotal/*=0*/ );
    public native int GetFrameCount( );
    public native int GetFrameCount( IntBuffer framesTotal/*=0*/ );
    public native int GetFrameCount( int[] framesTotal/*=0*/ );

    /* Retrieves camera parameters for specified camera.
       If camera is not calibrated the function returns 0 */
    public native @Const CvCamera GetCameraParams( int idx/*=0*/ );
    public native @Const CvCamera GetCameraParams( );

    public native @Const CvStereoCamera GetStereoParams();

    /* Sets camera parameters for all cameras */
    public native @Cast("bool") boolean SetCameraParams( CvCamera params );

    /* Saves all camera parameters to file */
    public native @Cast("bool") boolean SaveCameraParams( @Cast("const char*") BytePointer filename );
    public native @Cast("bool") boolean SaveCameraParams( String filename );

    /* Loads all camera parameters from file */
    public native @Cast("bool") boolean LoadCameraParams( @Cast("const char*") BytePointer filename );
    public native @Cast("bool") boolean LoadCameraParams( String filename );

    /* Undistorts images using camera parameters. Some of src pointers can be NULL. */
    public native @Cast("bool") boolean Undistort( @Cast("IplImage**") PointerPointer src, @Cast("IplImage**") PointerPointer dst );
    public native @Cast("bool") boolean Undistort( @ByPtrPtr IplImage src, @ByPtrPtr IplImage dst );

    /* Undistorts images using camera parameters. Some of src pointers can be NULL. */
    public native @Cast("bool") boolean Undistort( @ByPtrPtr CvMat src, @ByPtrPtr CvMat dst );

    /* Returns array of etalon points detected/partally detected
       on the latest frame for idx-th camera */
    public native @Cast("bool") boolean GetLatestPoints( int idx, @Cast("CvPoint2D32f**") PointerPointer pts,
                                                      IntPointer count, @Cast("bool*") BoolPointer found );
    public native @Cast("bool") boolean GetLatestPoints( int idx, @ByPtrPtr CvPoint2D32f pts,
                                                      IntPointer count, @Cast("bool*") BoolPointer found );
    public native @Cast("bool") boolean GetLatestPoints( int idx, @Cast("CvPoint2D32f**") @ByPtrPtr FloatBuffer pts,
                                                      IntBuffer count, @Cast("bool*") BoolPointer found );
    public native @Cast("bool") boolean GetLatestPoints( int idx, @Cast("CvPoint2D32f**") @ByPtrPtr float[] pts,
                                                      int[] count, @Cast("bool*") BoolPointer found );

    /* Draw the latest detected/partially detected etalon */
    public native void DrawPoints( @Cast("IplImage**") PointerPointer dst );
    public native void DrawPoints( @ByPtrPtr IplImage dst );

    /* Draw the latest detected/partially detected etalon */
    public native void DrawPoints( @ByPtrPtr CvMat dst );

    public native @Cast("bool") boolean Rectify( @Cast("IplImage**") PointerPointer srcarr, @Cast("IplImage**") PointerPointer dstarr );
    public native @Cast("bool") boolean Rectify( @ByPtrPtr IplImage srcarr, @ByPtrPtr IplImage dstarr );
    public native @Cast("bool") boolean Rectify( @ByPtrPtr CvMat srcarr, @ByPtrPtr CvMat dstarr );
}

// #include <iosfwd>
// #include <limits>

@NoOffset public static class CvImage extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvImage(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvImage(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CvImage position(int position) {
        return (CvImage)super.position(position);
    }

    public CvImage() { allocate(); }
    private native void allocate();
    public CvImage( @ByVal CvSize _size, int _depth, int _channels ) { allocate(_size, _depth, _channels); }
    private native void allocate( @ByVal CvSize _size, int _depth, int _channels );

    public CvImage( IplImage img ) { allocate(img); }
    private native void allocate( IplImage img );

    public CvImage( @Const @ByRef CvImage img ) { allocate(img); }
    private native void allocate( @Const @ByRef CvImage img );

    public CvImage( @Cast("const char*") BytePointer filename, @Cast("const char*") BytePointer imgname/*=0*/, int color/*=-1*/ ) { allocate(filename, imgname, color); }
    private native void allocate( @Cast("const char*") BytePointer filename, @Cast("const char*") BytePointer imgname/*=0*/, int color/*=-1*/ );
    public CvImage( @Cast("const char*") BytePointer filename ) { allocate(filename); }
    private native void allocate( @Cast("const char*") BytePointer filename );
    public CvImage( String filename, String imgname/*=0*/, int color/*=-1*/ ) { allocate(filename, imgname, color); }
    private native void allocate( String filename, String imgname/*=0*/, int color/*=-1*/ );
    public CvImage( String filename ) { allocate(filename); }
    private native void allocate( String filename );

    public CvImage( CvFileStorage fs, @Cast("const char*") BytePointer mapname, @Cast("const char*") BytePointer imgname ) { allocate(fs, mapname, imgname); }
    private native void allocate( CvFileStorage fs, @Cast("const char*") BytePointer mapname, @Cast("const char*") BytePointer imgname );
    public CvImage( CvFileStorage fs, String mapname, String imgname ) { allocate(fs, mapname, imgname); }
    private native void allocate( CvFileStorage fs, String mapname, String imgname );

    public CvImage( CvFileStorage fs, @Cast("const char*") BytePointer seqname, int idx ) { allocate(fs, seqname, idx); }
    private native void allocate( CvFileStorage fs, @Cast("const char*") BytePointer seqname, int idx );
    public CvImage( CvFileStorage fs, String seqname, int idx ) { allocate(fs, seqname, idx); }
    private native void allocate( CvFileStorage fs, String seqname, int idx );

    public native @ByVal CvImage clone();

    public native void create( @ByVal CvSize _size, int _depth, int _channels );

    public native void release();
    public native void clear();

    public native void attach( IplImage img, @Cast("bool") boolean use_refcount/*=true*/ );
    public native void attach( IplImage img );

    public native void detach();

    public native @Cast("bool") boolean load( @Cast("const char*") BytePointer filename, @Cast("const char*") BytePointer imgname/*=0*/, int color/*=-1*/ );
    public native @Cast("bool") boolean load( @Cast("const char*") BytePointer filename );
    public native @Cast("bool") boolean load( String filename, String imgname/*=0*/, int color/*=-1*/ );
    public native @Cast("bool") boolean load( String filename );
    public native @Cast("bool") boolean read( CvFileStorage fs, @Cast("const char*") BytePointer mapname, @Cast("const char*") BytePointer imgname );
    public native @Cast("bool") boolean read( CvFileStorage fs, String mapname, String imgname );
    public native @Cast("bool") boolean read( CvFileStorage fs, @Cast("const char*") BytePointer seqname, int idx );
    public native @Cast("bool") boolean read( CvFileStorage fs, String seqname, int idx );
    public native void save( @Cast("const char*") BytePointer filename, @Cast("const char*") BytePointer imgname, @Const IntPointer params/*=0*/ );
    public native void save( @Cast("const char*") BytePointer filename, @Cast("const char*") BytePointer imgname );
    public native void save( String filename, String imgname, @Const IntBuffer params/*=0*/ );
    public native void save( String filename, String imgname );
    public native void save( @Cast("const char*") BytePointer filename, @Cast("const char*") BytePointer imgname, @Const int[] params/*=0*/ );
    public native void save( String filename, String imgname, @Const IntPointer params/*=0*/ );
    public native void save( @Cast("const char*") BytePointer filename, @Cast("const char*") BytePointer imgname, @Const IntBuffer params/*=0*/ );
    public native void save( String filename, String imgname, @Const int[] params/*=0*/ );
    public native void write( CvFileStorage fs, @Cast("const char*") BytePointer imgname );
    public native void write( CvFileStorage fs, String imgname );

    public native void show( @Cast("const char*") BytePointer window_name );
    public native void show( String window_name );
    public native @Cast("bool") boolean is_valid();

    public native int width();
    public native int height();

    public native @ByVal CvSize size();

    public native @ByVal CvSize roi_size();

    public native @ByVal CvRect roi();

    public native int coi();

    public native void set_roi(@ByVal CvRect _roi);
    public native void reset_roi();
    public native void set_coi(int _coi);
    public native int depth();
    public native int channels();
    public native int pix_size();

    public native @Cast("uchar*") BytePointer data();
    public native int step();
    public native int origin();

    public native @Cast("uchar*") BytePointer roi_row(int y);
    public native @Name("operator IplImage*") IplImage asIplImage();

    public native @ByRef @Name("operator=") CvImage put(@Const @ByRef CvImage img);
}


@NoOffset public static class CvMatrix extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvMatrix(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvMatrix(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CvMatrix position(int position) {
        return (CvMatrix)super.position(position);
    }

    public CvMatrix() { allocate(); }
    private native void allocate();
    public CvMatrix( int _rows, int _cols, int _type ) { allocate(_rows, _cols, _type); }
    private native void allocate( int _rows, int _cols, int _type );

    public CvMatrix( int _rows, int _cols, int _type, CvMat hdr,
                 Pointer _data/*=0*/, int _step/*=CV_AUTOSTEP*/ ) { allocate(_rows, _cols, _type, hdr, _data, _step); }
    private native void allocate( int _rows, int _cols, int _type, CvMat hdr,
                 Pointer _data/*=0*/, int _step/*=CV_AUTOSTEP*/ );
    public CvMatrix( int _rows, int _cols, int _type, CvMat hdr ) { allocate(_rows, _cols, _type, hdr); }
    private native void allocate( int _rows, int _cols, int _type, CvMat hdr );

    public CvMatrix( int rows, int cols, int type, CvMemStorage storage, @Cast("bool") boolean alloc_data/*=true*/ ) { allocate(rows, cols, type, storage, alloc_data); }
    private native void allocate( int rows, int cols, int type, CvMemStorage storage, @Cast("bool") boolean alloc_data/*=true*/ );
    public CvMatrix( int rows, int cols, int type, CvMemStorage storage ) { allocate(rows, cols, type, storage); }
    private native void allocate( int rows, int cols, int type, CvMemStorage storage );

    public CvMatrix( int _rows, int _cols, int _type, Pointer _data, int _step/*=CV_AUTOSTEP*/ ) { allocate(_rows, _cols, _type, _data, _step); }
    private native void allocate( int _rows, int _cols, int _type, Pointer _data, int _step/*=CV_AUTOSTEP*/ );
    public CvMatrix( int _rows, int _cols, int _type, Pointer _data ) { allocate(_rows, _cols, _type, _data); }
    private native void allocate( int _rows, int _cols, int _type, Pointer _data );

    public CvMatrix( CvMat m ) { allocate(m); }
    private native void allocate( CvMat m );

    public CvMatrix( @Const @ByRef CvMatrix m ) { allocate(m); }
    private native void allocate( @Const @ByRef CvMatrix m );

    public CvMatrix( @Cast("const char*") BytePointer filename, @Cast("const char*") BytePointer matname/*=0*/, int color/*=-1*/ ) { allocate(filename, matname, color); }
    private native void allocate( @Cast("const char*") BytePointer filename, @Cast("const char*") BytePointer matname/*=0*/, int color/*=-1*/ );
    public CvMatrix( @Cast("const char*") BytePointer filename ) { allocate(filename); }
    private native void allocate( @Cast("const char*") BytePointer filename );
    public CvMatrix( String filename, String matname/*=0*/, int color/*=-1*/ ) { allocate(filename, matname, color); }
    private native void allocate( String filename, String matname/*=0*/, int color/*=-1*/ );
    public CvMatrix( String filename ) { allocate(filename); }
    private native void allocate( String filename );

    public CvMatrix( CvFileStorage fs, @Cast("const char*") BytePointer mapname, @Cast("const char*") BytePointer matname ) { allocate(fs, mapname, matname); }
    private native void allocate( CvFileStorage fs, @Cast("const char*") BytePointer mapname, @Cast("const char*") BytePointer matname );
    public CvMatrix( CvFileStorage fs, String mapname, String matname ) { allocate(fs, mapname, matname); }
    private native void allocate( CvFileStorage fs, String mapname, String matname );

    public CvMatrix( CvFileStorage fs, @Cast("const char*") BytePointer seqname, int idx ) { allocate(fs, seqname, idx); }
    private native void allocate( CvFileStorage fs, @Cast("const char*") BytePointer seqname, int idx );
    public CvMatrix( CvFileStorage fs, String seqname, int idx ) { allocate(fs, seqname, idx); }
    private native void allocate( CvFileStorage fs, String seqname, int idx );

    public native @ByVal CvMatrix clone();

    public native void set( CvMat m, @Cast("bool") boolean add_ref );

    public native void create( int _rows, int _cols, int _type );

    public native void addref();

    public native void release();

    public native void clear();

    public native @Cast("bool") boolean load( @Cast("const char*") BytePointer filename, @Cast("const char*") BytePointer matname/*=0*/, int color/*=-1*/ );
    public native @Cast("bool") boolean load( @Cast("const char*") BytePointer filename );
    public native @Cast("bool") boolean load( String filename, String matname/*=0*/, int color/*=-1*/ );
    public native @Cast("bool") boolean load( String filename );
    public native @Cast("bool") boolean read( CvFileStorage fs, @Cast("const char*") BytePointer mapname, @Cast("const char*") BytePointer matname );
    public native @Cast("bool") boolean read( CvFileStorage fs, String mapname, String matname );
    public native @Cast("bool") boolean read( CvFileStorage fs, @Cast("const char*") BytePointer seqname, int idx );
    public native @Cast("bool") boolean read( CvFileStorage fs, String seqname, int idx );
    public native void save( @Cast("const char*") BytePointer filename, @Cast("const char*") BytePointer matname, @Const IntPointer params/*=0*/ );
    public native void save( @Cast("const char*") BytePointer filename, @Cast("const char*") BytePointer matname );
    public native void save( String filename, String matname, @Const IntBuffer params/*=0*/ );
    public native void save( String filename, String matname );
    public native void save( @Cast("const char*") BytePointer filename, @Cast("const char*") BytePointer matname, @Const int[] params/*=0*/ );
    public native void save( String filename, String matname, @Const IntPointer params/*=0*/ );
    public native void save( @Cast("const char*") BytePointer filename, @Cast("const char*") BytePointer matname, @Const IntBuffer params/*=0*/ );
    public native void save( String filename, String matname, @Const int[] params/*=0*/ );
    public native void write( CvFileStorage fs, @Cast("const char*") BytePointer matname );
    public native void write( CvFileStorage fs, String matname );

    public native void show( @Cast("const char*") BytePointer window_name );
    public native void show( String window_name );

    public native @Cast("bool") boolean is_valid();

    public native int rows();
    public native int cols();

    public native @ByVal CvSize size();

    public native int type();
    public native int depth();
    public native int channels();
    public native int pix_size();

    public native @Cast("uchar*") BytePointer data();
    public native int step();

    public native void set_data( Pointer _data, int _step/*=CV_AUTOSTEP*/ );
    public native void set_data( Pointer _data );

    public native @Cast("uchar*") BytePointer row(int i);
    public native @Name("operator CvMat*") CvMat asCvMat();

    public native @ByRef @Name("operator=") CvMatrix put(@Const @ByRef CvMatrix _m);
}

/****************************************************************************************\
 *                                       CamShiftTracker                                  *
 \****************************************************************************************/

@NoOffset public static class CvCamShiftTracker extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvCamShiftTracker(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvCamShiftTracker(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CvCamShiftTracker position(int position) {
        return (CvCamShiftTracker)super.position(position);
    }


    public CvCamShiftTracker() { allocate(); }
    private native void allocate();

    /**** Characteristics of the object that are calculated by track_object method *****/
    public native float get_orientation();
    public native float get_length();
    public native float get_width();
    public native @ByVal CvPoint2D32f get_center();
    public native @ByVal CvRect get_window();

    /*********************** Tracking parameters ************************/
    public native int get_threshold();

    public native int get_hist_dims( IntPointer dims/*=0*/ );
    public native int get_hist_dims( );
    public native int get_hist_dims( IntBuffer dims/*=0*/ );
    public native int get_hist_dims( int[] dims/*=0*/ );

    public native int get_min_ch_val( int channel );

    public native int get_max_ch_val( int channel );

    // set initial object rectangle (must be called before initial calculation of the histogram)
    public native @Cast("bool") boolean set_window( @ByVal CvRect window);

    public native @Cast("bool") boolean set_threshold( int threshold );

    public native @Cast("bool") boolean set_hist_bin_range( int dim, int min_val, int max_val );

    public native @Cast("bool") boolean set_hist_dims( int c_dims, IntPointer dims );
    public native @Cast("bool") boolean set_hist_dims( int c_dims, IntBuffer dims );
    public native @Cast("bool") boolean set_hist_dims( int c_dims, int[] dims );// set the histogram parameters

    public native @Cast("bool") boolean set_min_ch_val( int channel, int val );
    public native @Cast("bool") boolean set_max_ch_val( int channel, int val );

    /************************ The processing methods *********************************/
    // update object position
    public native @Cast("bool") boolean track_object( @Const IplImage cur_frame );

    // update object histogram
    public native @Cast("bool") boolean update_histogram( @Const IplImage cur_frame );

    // reset histogram
    public native void reset_histogram();

    /************************ Retrieving internal data *******************************/
    // get back project image
    public native IplImage get_back_project();

    public native float query( IntPointer bin );
    public native float query( IntBuffer bin );
    public native float query( int[] bin );
}

/****************************************************************************************\
*                              Expectation - Maximization                                *
\****************************************************************************************/
@NoOffset public static class CvEMParams extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvEMParams(Pointer p) { super(p); }

    public CvEMParams() { allocate(); }
    private native void allocate();
    public CvEMParams( int nclusters, int cov_mat_type/*=cv::EM::COV_MAT_DIAGONAL*/,
                    int start_step/*=cv::EM::START_AUTO_STEP*/,
                    @ByVal(nullValue = "cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, FLT_EPSILON)") CvTermCriteria term_crit/*=cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, FLT_EPSILON)*/,
                    @Const CvMat probs/*=0*/, @Const CvMat weights/*=0*/, @Const CvMat means/*=0*/, @Cast("const CvMat**") PointerPointer covs/*=0*/ ) { allocate(nclusters, cov_mat_type, start_step, term_crit, probs, weights, means, covs); }
    private native void allocate( int nclusters, int cov_mat_type/*=cv::EM::COV_MAT_DIAGONAL*/,
                    int start_step/*=cv::EM::START_AUTO_STEP*/,
                    @ByVal(nullValue = "cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, FLT_EPSILON)") CvTermCriteria term_crit/*=cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, FLT_EPSILON)*/,
                    @Const CvMat probs/*=0*/, @Const CvMat weights/*=0*/, @Const CvMat means/*=0*/, @Cast("const CvMat**") PointerPointer covs/*=0*/ );
    public CvEMParams( int nclusters ) { allocate(nclusters); }
    private native void allocate( int nclusters );
    public CvEMParams( int nclusters, int cov_mat_type/*=cv::EM::COV_MAT_DIAGONAL*/,
                    int start_step/*=cv::EM::START_AUTO_STEP*/,
                    @ByVal(nullValue = "cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, FLT_EPSILON)") CvTermCriteria term_crit/*=cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, FLT_EPSILON)*/,
                    @Const CvMat probs/*=0*/, @Const CvMat weights/*=0*/, @Const CvMat means/*=0*/, @Const @ByPtrPtr CvMat covs/*=0*/ ) { allocate(nclusters, cov_mat_type, start_step, term_crit, probs, weights, means, covs); }
    private native void allocate( int nclusters, int cov_mat_type/*=cv::EM::COV_MAT_DIAGONAL*/,
                    int start_step/*=cv::EM::START_AUTO_STEP*/,
                    @ByVal(nullValue = "cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, FLT_EPSILON)") CvTermCriteria term_crit/*=cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, FLT_EPSILON)*/,
                    @Const CvMat probs/*=0*/, @Const CvMat weights/*=0*/, @Const CvMat means/*=0*/, @Const @ByPtrPtr CvMat covs/*=0*/ );

    public native int nclusters(); public native CvEMParams nclusters(int nclusters);
    public native int cov_mat_type(); public native CvEMParams cov_mat_type(int cov_mat_type);
    public native int start_step(); public native CvEMParams start_step(int start_step);
    @MemberGetter public native @Const CvMat probs();
    @MemberGetter public native @Const CvMat weights();
    @MemberGetter public native @Const CvMat means();
    @MemberGetter public native @Const CvMat covs(int i);
    @MemberGetter public native @Cast("const CvMat**") PointerPointer covs();
    public native @ByRef CvTermCriteria term_crit(); public native CvEMParams term_crit(CvTermCriteria term_crit);
}


@NoOffset public static class CvEM extends CvStatModel {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvEM(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvEM(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CvEM position(int position) {
        return (CvEM)super.position(position);
    }

    // Type of covariation matrices
    /** enum CvEM:: */
    public static final int COV_MAT_SPHERICAL=EM.COV_MAT_SPHERICAL,
           COV_MAT_DIAGONAL =EM.COV_MAT_DIAGONAL,
           COV_MAT_GENERIC  =EM.COV_MAT_GENERIC;

    // The initial step
    /** enum CvEM:: */
    public static final int START_E_STEP=EM.START_E_STEP,
           START_M_STEP=EM.START_M_STEP,
           START_AUTO_STEP=EM.START_AUTO_STEP;

    public CvEM() { allocate(); }
    private native void allocate();
    public CvEM( @Const CvMat samples, @Const CvMat sampleIdx/*=0*/,
              @ByVal(nullValue = "CvEMParams()") CvEMParams params/*=CvEMParams()*/, CvMat labels/*=0*/ ) { allocate(samples, sampleIdx, params, labels); }
    private native void allocate( @Const CvMat samples, @Const CvMat sampleIdx/*=0*/,
              @ByVal(nullValue = "CvEMParams()") CvEMParams params/*=CvEMParams()*/, CvMat labels/*=0*/ );
    public CvEM( @Const CvMat samples ) { allocate(samples); }
    private native void allocate( @Const CvMat samples );

    public native @Cast("bool") boolean train( @Const CvMat samples, @Const CvMat sampleIdx/*=0*/,
                            @ByVal(nullValue = "CvEMParams()") CvEMParams params/*=CvEMParams()*/, CvMat labels/*=0*/ );
    public native @Cast("bool") boolean train( @Const CvMat samples );

    public native float predict( @Const CvMat sample, CvMat probs );

    public CvEM( @Const @ByRef Mat samples, @Const @ByRef(nullValue = "cv::Mat()") Mat sampleIdx/*=cv::Mat()*/,
                      @ByVal(nullValue = "CvEMParams()") CvEMParams params/*=CvEMParams()*/ ) { allocate(samples, sampleIdx, params); }
    private native void allocate( @Const @ByRef Mat samples, @Const @ByRef(nullValue = "cv::Mat()") Mat sampleIdx/*=cv::Mat()*/,
                      @ByVal(nullValue = "CvEMParams()") CvEMParams params/*=CvEMParams()*/ );
    public CvEM( @Const @ByRef Mat samples ) { allocate(samples); }
    private native void allocate( @Const @ByRef Mat samples );

    public native @Cast("bool") boolean train( @Const @ByRef Mat samples,
                                    @Const @ByRef(nullValue = "cv::Mat()") Mat sampleIdx/*=cv::Mat()*/,
                                    @ByVal(nullValue = "CvEMParams()") CvEMParams params/*=CvEMParams()*/,
                                    Mat labels/*=0*/ );
    public native @Cast("bool") boolean train( @Const @ByRef Mat samples );

    public native float predict( @Const @ByRef Mat sample, Mat probs/*=0*/ );
    public native float predict( @Const @ByRef Mat sample );
    public native double calcLikelihood( @Const @ByRef Mat sample );

    public native int getNClusters();
    public native @ByVal Mat getMeans();
    public native void getCovs(@ByRef MatVector covs);
    public native @ByVal Mat getWeights();
    public native @ByVal Mat getProbs();

    public native double getLikelihood();

    public native void clear();

    public native int get_nclusters();
    public native @Const CvMat get_means();
    public native @Cast("const CvMat**") PointerPointer get_covs();
    public native @Const CvMat get_weights();
    public native @Const CvMat get_probs();

    public native double get_log_likelihood();

    public native void read( CvFileStorage fs, CvFileNode node );
    public native void write( CvFileStorage fs, @Cast("const char*") BytePointer name );
    public native void write( CvFileStorage fs, String name );
}

/**
 The Patch Generator class
 */
@Namespace("cv") @NoOffset public static class PatchGenerator extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public PatchGenerator(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public PatchGenerator(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public PatchGenerator position(int position) {
        return (PatchGenerator)super.position(position);
    }

    public PatchGenerator() { allocate(); }
    private native void allocate();
    public PatchGenerator(double _backgroundMin, double _backgroundMax,
                       double _noiseRange, @Cast("bool") boolean _randomBlur/*=true*/,
                       double _lambdaMin/*=0.6*/, double _lambdaMax/*=1.5*/,
                       double _thetaMin/*=-CV_PI*/, double _thetaMax/*=CV_PI*/,
                       double _phiMin/*=-CV_PI*/, double _phiMax/*=CV_PI*/ ) { allocate(_backgroundMin, _backgroundMax, _noiseRange, _randomBlur, _lambdaMin, _lambdaMax, _thetaMin, _thetaMax, _phiMin, _phiMax); }
    private native void allocate(double _backgroundMin, double _backgroundMax,
                       double _noiseRange, @Cast("bool") boolean _randomBlur/*=true*/,
                       double _lambdaMin/*=0.6*/, double _lambdaMax/*=1.5*/,
                       double _thetaMin/*=-CV_PI*/, double _thetaMax/*=CV_PI*/,
                       double _phiMin/*=-CV_PI*/, double _phiMax/*=CV_PI*/ );
    public PatchGenerator(double _backgroundMin, double _backgroundMax,
                       double _noiseRange ) { allocate(_backgroundMin, _backgroundMax, _noiseRange); }
    private native void allocate(double _backgroundMin, double _backgroundMax,
                       double _noiseRange );
    public native @Name("operator()") void apply(@Const @ByRef Mat image, @ByVal Point2f pt, @ByRef Mat patch, @ByVal Size patchSize, @ByRef RNG rng);
    public native @Name("operator()") void apply(@Const @ByRef Mat image, @Const @ByRef Mat transform, @ByRef Mat patch,
                        @ByVal Size patchSize, @ByRef RNG rng);
    public native void warpWholeImage(@Const @ByRef Mat image, @ByRef Mat matT, @ByRef Mat buf,
                            @ByRef Mat warped, int border, @ByRef RNG rng);
    public native void generateRandomTransform(@ByVal Point2f srcCenter, @ByVal Point2f dstCenter,
                                     @ByRef Mat transform, @ByRef RNG rng,
                                     @Cast("bool") boolean inverse/*=false*/);
    public native void generateRandomTransform(@ByVal Point2f srcCenter, @ByVal Point2f dstCenter,
                                     @ByRef Mat transform, @ByRef RNG rng);
    public native void setAffineParam(double lambda, double theta, double phi);

    public native double backgroundMin(); public native PatchGenerator backgroundMin(double backgroundMin);
    public native double backgroundMax(); public native PatchGenerator backgroundMax(double backgroundMax);
    public native double noiseRange(); public native PatchGenerator noiseRange(double noiseRange);
    public native @Cast("bool") boolean randomBlur(); public native PatchGenerator randomBlur(boolean randomBlur);
    public native double lambdaMin(); public native PatchGenerator lambdaMin(double lambdaMin);
    public native double lambdaMax(); public native PatchGenerator lambdaMax(double lambdaMax);
    public native double thetaMin(); public native PatchGenerator thetaMin(double thetaMin);
    public native double thetaMax(); public native PatchGenerator thetaMax(double thetaMax);
    public native double phiMin(); public native PatchGenerator phiMin(double phiMin);
    public native double phiMax(); public native PatchGenerator phiMax(double phiMax);
}


@Namespace("cv") @NoOffset public static class LDetector extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public LDetector(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public LDetector(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public LDetector position(int position) {
        return (LDetector)super.position(position);
    }

    public LDetector() { allocate(); }
    private native void allocate();
    public LDetector(int _radius, int _threshold, int _nOctaves,
                  int _nViews, double _baseFeatureSize, double _clusteringDistance) { allocate(_radius, _threshold, _nOctaves, _nViews, _baseFeatureSize, _clusteringDistance); }
    private native void allocate(int _radius, int _threshold, int _nOctaves,
                  int _nViews, double _baseFeatureSize, double _clusteringDistance);
    public native @Name("operator()") void apply(@Const @ByRef Mat image,
                        @StdVector KeyPoint keypoints,
                        int maxCount/*=0*/, @Cast("bool") boolean scaleCoords/*=true*/);
    public native @Name("operator()") void apply(@Const @ByRef Mat image,
                        @StdVector KeyPoint keypoints);
    public native @Name("operator()") void apply(@Const @ByRef MatVector pyr,
                        @StdVector KeyPoint keypoints,
                        int maxCount/*=0*/, @Cast("bool") boolean scaleCoords/*=true*/);
    public native @Name("operator()") void apply(@Const @ByRef MatVector pyr,
                        @StdVector KeyPoint keypoints);
    public native void getMostStable2D(@Const @ByRef Mat image, @StdVector KeyPoint keypoints,
                             int maxCount, @Const @ByRef PatchGenerator patchGenerator);
    public native void setVerbose(@Cast("bool") boolean verbose);

    public native void read(@Const @ByRef FileNode node);
    public native void write(@ByRef FileStorage fs, @StdString BytePointer name/*=cv::String()*/);
    public native void write(@ByRef FileStorage fs);
    public native void write(@ByRef FileStorage fs, @StdString String name/*=cv::String()*/);

    public native int radius(); public native LDetector radius(int radius);
    public native int threshold(); public native LDetector threshold(int threshold);
    public native int nOctaves(); public native LDetector nOctaves(int nOctaves);
    public native int nViews(); public native LDetector nViews(int nViews);
    public native @Cast("bool") boolean verbose(); public native LDetector verbose(boolean verbose);

    public native double baseFeatureSize(); public native LDetector baseFeatureSize(double baseFeatureSize);
    public native double clusteringDistance(); public native LDetector clusteringDistance(double clusteringDistance);
}

@Namespace("cv") @NoOffset public static class FernClassifier extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FernClassifier(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public FernClassifier(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public FernClassifier position(int position) {
        return (FernClassifier)super.position(position);
    }

    public FernClassifier() { allocate(); }
    private native void allocate();
    public FernClassifier(@Const @ByRef FileNode node) { allocate(node); }
    private native void allocate(@Const @ByRef FileNode node);
    public FernClassifier(@Const @ByRef Point2fVectorVector points,
                       @Const @ByRef MatVector refimgs,
                       @Const @ByRef(nullValue = "std::vector<std::vector<int> >()") IntVectorVector labels/*=std::vector<std::vector<int> >()*/,
                       int _nclasses/*=0*/, int _patchSize/*=PATCH_SIZE*/,
                       int _signatureSize/*=DEFAULT_SIGNATURE_SIZE*/,
                       int _nstructs/*=DEFAULT_STRUCTS*/,
                       int _structSize/*=DEFAULT_STRUCT_SIZE*/,
                       int _nviews/*=DEFAULT_VIEWS*/,
                       int _compressionMethod/*=COMPRESSION_NONE*/,
                       @Const @ByRef(nullValue = "cv::PatchGenerator()") PatchGenerator patchGenerator/*=cv::PatchGenerator()*/) { allocate(points, refimgs, labels, _nclasses, _patchSize, _signatureSize, _nstructs, _structSize, _nviews, _compressionMethod, patchGenerator); }
    private native void allocate(@Const @ByRef Point2fVectorVector points,
                       @Const @ByRef MatVector refimgs,
                       @Const @ByRef(nullValue = "std::vector<std::vector<int> >()") IntVectorVector labels/*=std::vector<std::vector<int> >()*/,
                       int _nclasses/*=0*/, int _patchSize/*=PATCH_SIZE*/,
                       int _signatureSize/*=DEFAULT_SIGNATURE_SIZE*/,
                       int _nstructs/*=DEFAULT_STRUCTS*/,
                       int _structSize/*=DEFAULT_STRUCT_SIZE*/,
                       int _nviews/*=DEFAULT_VIEWS*/,
                       int _compressionMethod/*=COMPRESSION_NONE*/,
                       @Const @ByRef(nullValue = "cv::PatchGenerator()") PatchGenerator patchGenerator/*=cv::PatchGenerator()*/);
    public FernClassifier(@Const @ByRef Point2fVectorVector points,
                       @Const @ByRef MatVector refimgs) { allocate(points, refimgs); }
    private native void allocate(@Const @ByRef Point2fVectorVector points,
                       @Const @ByRef MatVector refimgs);
    public native void read(@Const @ByRef FileNode n);
    public native void write(@ByRef FileStorage fs, @StdString BytePointer name/*=cv::String()*/);
    public native void write(@ByRef FileStorage fs);
    public native void write(@ByRef FileStorage fs, @StdString String name/*=cv::String()*/);
    public native void trainFromSingleView(@Const @ByRef Mat image,
                                         @StdVector KeyPoint keypoints,
                                         int _patchSize/*=PATCH_SIZE*/,
                                         int _signatureSize/*=DEFAULT_SIGNATURE_SIZE*/,
                                         int _nstructs/*=DEFAULT_STRUCTS*/,
                                         int _structSize/*=DEFAULT_STRUCT_SIZE*/,
                                         int _nviews/*=DEFAULT_VIEWS*/,
                                         int _compressionMethod/*=COMPRESSION_NONE*/,
                                         @Const @ByRef(nullValue = "cv::PatchGenerator()") PatchGenerator patchGenerator/*=cv::PatchGenerator()*/);
    public native void trainFromSingleView(@Const @ByRef Mat image,
                                         @StdVector KeyPoint keypoints);
    public native void train(@Const @ByRef Point2fVectorVector points,
                           @Const @ByRef MatVector refimgs,
                           @Const @ByRef(nullValue = "std::vector<std::vector<int> >()") IntVectorVector labels/*=std::vector<std::vector<int> >()*/,
                           int _nclasses/*=0*/, int _patchSize/*=PATCH_SIZE*/,
                           int _signatureSize/*=DEFAULT_SIGNATURE_SIZE*/,
                           int _nstructs/*=DEFAULT_STRUCTS*/,
                           int _structSize/*=DEFAULT_STRUCT_SIZE*/,
                           int _nviews/*=DEFAULT_VIEWS*/,
                           int _compressionMethod/*=COMPRESSION_NONE*/,
                           @Const @ByRef(nullValue = "cv::PatchGenerator()") PatchGenerator patchGenerator/*=cv::PatchGenerator()*/);
    public native void train(@Const @ByRef Point2fVectorVector points,
                           @Const @ByRef MatVector refimgs);
    public native @Name("operator()") int apply(@Const @ByRef Mat img, @ByVal Point2f kpt, @StdVector FloatPointer signature);
    public native @Name("operator()") int apply(@Const @ByRef Mat img, @ByVal Point2f kpt, @StdVector FloatBuffer signature);
    public native @Name("operator()") int apply(@Const @ByRef Mat img, @ByVal Point2f kpt, @StdVector float[] signature);
    public native @Name("operator()") int apply(@Const @ByRef Mat patch, @StdVector FloatPointer signature);
    public native @Name("operator()") int apply(@Const @ByRef Mat patch, @StdVector FloatBuffer signature);
    public native @Name("operator()") int apply(@Const @ByRef Mat patch, @StdVector float[] signature);
    public native void clear();
    public native @Cast("bool") boolean empty();
    public native void setVerbose(@Cast("bool") boolean verbose);

    public native int getClassCount();
    public native int getStructCount();
    public native int getStructSize();
    public native int getSignatureSize();
    public native int getCompressionMethod();
    public native @ByVal Size getPatchSize();

    @NoOffset public static class Feature extends Pointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public Feature(Pointer p) { super(p); }
        /** Native array allocator. Access with {@link Pointer#position(int)}. */
        public Feature(int size) { allocateArray(size); }
        private native void allocateArray(int size);
        @Override public Feature position(int position) {
            return (Feature)super.position(position);
        }
    
        public native @Cast("uchar") byte x1(); public native Feature x1(byte x1);
        public native @Cast("uchar") byte y1(); public native Feature y1(byte y1);
        public native @Cast("uchar") byte x2(); public native Feature x2(byte x2);
        public native @Cast("uchar") byte y2(); public native Feature y2(byte y2);
        public Feature() { allocate(); }
        private native void allocate();
        public Feature(int _x1, int _y1, int _x2, int _y2) { allocate(_x1, _y1, _x2, _y2); }
        private native void allocate(int _x1, int _y1, int _x2, int _y2);
    }

    /** enum cv::FernClassifier:: */
    public static final int
        PATCH_SIZE = 31,
        DEFAULT_STRUCTS = 50,
        DEFAULT_STRUCT_SIZE = 9,
        DEFAULT_VIEWS = 5000,
        DEFAULT_SIGNATURE_SIZE = 176,
        COMPRESSION_NONE = 0,
        COMPRESSION_RANDOM_PROJ = 1,
        COMPRESSION_PCA = 2,
        DEFAULT_COMPRESSION_METHOD =  COMPRESSION_NONE;
}


/****************************************************************************************\
 *                                 Calonder Classifier                                    *
 \****************************************************************************************/

@Namespace("cv") @NoOffset public static class BaseKeypoint extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public BaseKeypoint(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public BaseKeypoint(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public BaseKeypoint position(int position) {
        return (BaseKeypoint)super.position(position);
    }

    public native int x(); public native BaseKeypoint x(int x);
    public native int y(); public native BaseKeypoint y(int y);
    public native IplImage image(); public native BaseKeypoint image(IplImage image);

    public BaseKeypoint() { allocate(); }
    private native void allocate();

    public BaseKeypoint(int _x, int _y, IplImage _image) { allocate(_x, _y, _image); }
    private native void allocate(int _x, int _y, IplImage _image);
}

@Namespace("cv") @NoOffset public static class RandomizedTree extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public RandomizedTree(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public RandomizedTree(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public RandomizedTree position(int position) {
        return (RandomizedTree)super.position(position);
    }


    @MemberGetter public static native @Cast("const uchar") byte PATCH_SIZE();
    @MemberGetter public static native int DEFAULT_DEPTH();
    @MemberGetter public static native int DEFAULT_VIEWS();
    @MemberGetter public static native @Cast("const size_t") long DEFAULT_REDUCED_NUM_DIM();
    public static native float GET_LOWER_QUANT_PERC();
    public static native float GET_UPPER_QUANT_PERC();

    public RandomizedTree() { allocate(); }
    private native void allocate();

    public native void train(@StdVector BaseKeypoint base_set, @ByRef RNG rng,
                   int depth, int views, @Cast("size_t") long reduced_num_dim, int num_quant_bits);
    public native void train(@StdVector BaseKeypoint base_set, @ByRef RNG rng,
                   @ByRef PatchGenerator make_patch, int depth, int views, @Cast("size_t") long reduced_num_dim,
                   int num_quant_bits);

    // following two funcs are EXPERIMENTAL (do not use unless you know exactly what you do)
    public static native void quantizeVector(FloatPointer vec, int dim, int N, FloatPointer bnds, int clamp_mode/*=0*/);
    public static native void quantizeVector(FloatPointer vec, int dim, int N, FloatPointer bnds);
    public static native void quantizeVector(FloatBuffer vec, int dim, int N, FloatBuffer bnds, int clamp_mode/*=0*/);
    public static native void quantizeVector(FloatBuffer vec, int dim, int N, FloatBuffer bnds);
    public static native void quantizeVector(float[] vec, int dim, int N, float[] bnds, int clamp_mode/*=0*/);
    public static native void quantizeVector(float[] vec, int dim, int N, float[] bnds);
    public static native void quantizeVector(FloatPointer src, int dim, int N, FloatPointer bnds, @Cast("uchar*") BytePointer dst);
    public static native void quantizeVector(FloatBuffer src, int dim, int N, FloatBuffer bnds, @Cast("uchar*") ByteBuffer dst);
    public static native void quantizeVector(float[] src, int dim, int N, float[] bnds, @Cast("uchar*") byte[] dst);

    // patch_data must be a 32x32 array (no row padding)
    public native FloatPointer getPosterior(@Cast("uchar*") BytePointer patch_data);
    public native FloatBuffer getPosterior(@Cast("uchar*") ByteBuffer patch_data);
    public native float[] getPosterior(@Cast("uchar*") byte[] patch_data);
    public native @Cast("uchar*") BytePointer getPosterior2(@Cast("uchar*") BytePointer patch_data);
    public native @Cast("uchar*") ByteBuffer getPosterior2(@Cast("uchar*") ByteBuffer patch_data);
    public native @Cast("uchar*") byte[] getPosterior2(@Cast("uchar*") byte[] patch_data);

    public native void read(@Cast("const char*") BytePointer file_name, int num_quant_bits);
    public native void read(String file_name, int num_quant_bits);
    public native void read(@Cast("std::istream*") @ByRef Pointer is, int num_quant_bits);
    public native void write(@Cast("const char*") BytePointer file_name);
    public native void write(String file_name);
    public native void write(@Cast("std::ostream*") @ByRef Pointer os);

    public native int classes();
    public native int depth();

    //void setKeepFloatPosteriors(bool b) { keep_float_posteriors_ = b; }
    public native void discardFloatPosteriors();

    public native void applyQuantization(int num_quant_bits);

    // debug
    public native void savePosteriors(@StdString BytePointer url, @Cast("bool") boolean append/*=false*/);
    public native void savePosteriors(@StdString BytePointer url);
    public native void savePosteriors(@StdString String url, @Cast("bool") boolean append/*=false*/);
    public native void savePosteriors(@StdString String url);
    public native void savePosteriors2(@StdString BytePointer url, @Cast("bool") boolean append/*=false*/);
    public native void savePosteriors2(@StdString BytePointer url);
    public native void savePosteriors2(@StdString String url, @Cast("bool") boolean append/*=false*/);
    public native void savePosteriors2(@StdString String url);
}


@Namespace("cv") public static native @Cast("uchar*") BytePointer getData(IplImage image);









@Namespace("cv") @NoOffset public static class RTreeNode extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public RTreeNode(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public RTreeNode(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public RTreeNode position(int position) {
        return (RTreeNode)super.position(position);
    }

    public native short offset1(); public native RTreeNode offset1(short offset1);
    public native short offset2(); public native RTreeNode offset2(short offset2);

    public RTreeNode() { allocate(); }
    private native void allocate();
    public RTreeNode(@Cast("uchar") byte x1, @Cast("uchar") byte y1, @Cast("uchar") byte x2, @Cast("uchar") byte y2) { allocate(x1, y1, x2, y2); }
    private native void allocate(@Cast("uchar") byte x1, @Cast("uchar") byte y1, @Cast("uchar") byte x2, @Cast("uchar") byte y2);

    /** Left child on 0, right child on 1 */
    public native @Cast("bool") @Name("operator()") boolean apply(@Cast("uchar*") BytePointer patch_data);
    public native @Cast("bool") @Name("operator()") boolean apply(@Cast("uchar*") ByteBuffer patch_data);
    public native @Cast("bool") @Name("operator()") boolean apply(@Cast("uchar*") byte[] patch_data);
}

@Namespace("cv") @NoOffset public static class RTreeClassifier extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public RTreeClassifier(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public RTreeClassifier(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public RTreeClassifier position(int position) {
        return (RTreeClassifier)super.position(position);
    }

    @MemberGetter public static native int DEFAULT_TREES();
    @MemberGetter public static native @Cast("const size_t") long DEFAULT_NUM_QUANT_BITS();

    public RTreeClassifier() { allocate(); }
    private native void allocate();
    public native void train(@StdVector BaseKeypoint base_set,
                   @ByRef RNG rng,
                   int num_trees/*=cv::RTreeClassifier::DEFAULT_TREES*/,
                   int depth/*=cv::RandomizedTree::DEFAULT_DEPTH*/,
                   int views/*=cv::RandomizedTree::DEFAULT_VIEWS*/,
                   @Cast("size_t") long reduced_num_dim/*=cv::RandomizedTree::DEFAULT_REDUCED_NUM_DIM*/,
                   int num_quant_bits/*=cv::RTreeClassifier::DEFAULT_NUM_QUANT_BITS*/);
    public native void train(@StdVector BaseKeypoint base_set,
                   @ByRef RNG rng);
    public native void train(@StdVector BaseKeypoint base_set,
                   @ByRef RNG rng,
                   @ByRef PatchGenerator make_patch,
                   int num_trees/*=cv::RTreeClassifier::DEFAULT_TREES*/,
                   int depth/*=cv::RandomizedTree::DEFAULT_DEPTH*/,
                   int views/*=cv::RandomizedTree::DEFAULT_VIEWS*/,
                   @Cast("size_t") long reduced_num_dim/*=cv::RandomizedTree::DEFAULT_REDUCED_NUM_DIM*/,
                   int num_quant_bits/*=cv::RTreeClassifier::DEFAULT_NUM_QUANT_BITS*/);
    public native void train(@StdVector BaseKeypoint base_set,
                   @ByRef RNG rng,
                   @ByRef PatchGenerator make_patch);

    // sig must point to a memory block of at least classes()*sizeof(float|uchar) bytes
    public native void getSignature(IplImage patch, @Cast("uchar*") BytePointer sig);
    public native void getSignature(IplImage patch, @Cast("uchar*") ByteBuffer sig);
    public native void getSignature(IplImage patch, @Cast("uchar*") byte[] sig);
    public native void getSignature(IplImage patch, FloatPointer sig);
    public native void getSignature(IplImage patch, FloatBuffer sig);
    public native void getSignature(IplImage patch, float[] sig);
    public native void getSparseSignature(IplImage patch, FloatPointer sig, float thresh);
    public native void getSparseSignature(IplImage patch, FloatBuffer sig, float thresh);
    public native void getSparseSignature(IplImage patch, float[] sig, float thresh);
    // TODO: deprecated in favor of getSignature overload, remove
    public native void getFloatSignature(IplImage patch, FloatPointer sig);
    public native void getFloatSignature(IplImage patch, FloatBuffer sig);
    public native void getFloatSignature(IplImage patch, float[] sig);

    public static native int countNonZeroElements(FloatPointer vec, int n, double tol/*=1e-10*/);
    public static native int countNonZeroElements(FloatPointer vec, int n);
    public static native int countNonZeroElements(FloatBuffer vec, int n, double tol/*=1e-10*/);
    public static native int countNonZeroElements(FloatBuffer vec, int n);
    public static native int countNonZeroElements(float[] vec, int n, double tol/*=1e-10*/);
    public static native int countNonZeroElements(float[] vec, int n);
    
    

    public native int classes();
    public native int original_num_classes();

    public native void setQuantization(int num_quant_bits);
    public native void discardFloatPosteriors();

    public native void read(@Cast("const char*") BytePointer file_name);
    public native void read(String file_name);
    public native void read(@Cast("std::istream*") @ByRef Pointer is);
    public native void write(@Cast("const char*") BytePointer file_name);
    public native void write(String file_name);
    public native void write(@Cast("std::ostream*") @ByRef Pointer os);

    // experimental and debug
    public native void saveAllFloatPosteriors(@StdString BytePointer file_url);
    public native void saveAllFloatPosteriors(@StdString String file_url);
    public native void saveAllBytePosteriors(@StdString BytePointer file_url);
    public native void saveAllBytePosteriors(@StdString String file_url);
    public native void setFloatPosteriorsFromTextfile_176(@StdString BytePointer url);
    public native void setFloatPosteriorsFromTextfile_176(@StdString String url);
    public native float countZeroElements();

    public native @StdVector RandomizedTree trees_(); public native RTreeClassifier trees_(RandomizedTree trees_);
}

/****************************************************************************************\
*                                     One-Way Descriptor                                 *
\****************************************************************************************/

// CvAffinePose: defines a parameterized affine transformation of an image patch.
// An image patch is rotated on angle phi (in degrees), then scaled lambda1 times
// along horizontal and lambda2 times along vertical direction, and then rotated again
// on angle (theta - phi).
@Namespace("cv") public static class CvAffinePose extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvAffinePose() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvAffinePose(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvAffinePose(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvAffinePose position(int position) {
        return (CvAffinePose)super.position(position);
    }

    public native float phi(); public native CvAffinePose phi(float phi);
    public native float theta(); public native CvAffinePose theta(float theta);
    public native float lambda1(); public native CvAffinePose lambda1(float lambda1);
    public native float lambda2(); public native CvAffinePose lambda2(float lambda2);
}

@Namespace("cv") @NoOffset public static class OneWayDescriptor extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public OneWayDescriptor(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public OneWayDescriptor(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public OneWayDescriptor position(int position) {
        return (OneWayDescriptor)super.position(position);
    }

    public OneWayDescriptor() { allocate(); }
    private native void allocate();

    // allocates memory for given descriptor parameters
    public native void Allocate(int pose_count, @ByVal CvSize size, int nChannels);

    // GenerateSamples: generates affine transformed patches with averaging them over small transformation variations.
    // If external poses and transforms were specified, uses them instead of generating random ones
    // - pose_count: the number of poses to be generated
    // - frontal: the input patch (can be a roi in a larger image)
    // - norm: if nonzero, normalizes the output patch so that the sum of pixel intensities is 1
    public native void GenerateSamples(int pose_count, IplImage frontal, int norm/*=0*/);
    public native void GenerateSamples(int pose_count, IplImage frontal);

    // GenerateSamplesFast: generates affine transformed patches with averaging them over small transformation variations.
    // Uses precalculated transformed pca components.
    // - frontal: the input patch (can be a roi in a larger image)
    // - pca_hr_avg: pca average vector
    // - pca_hr_eigenvectors: pca eigenvectors
    // - pca_descriptors: an array of precomputed descriptors of pca components containing their affine transformations
    //   pca_descriptors[0] corresponds to the average, pca_descriptors[1]-pca_descriptors[pca_dim] correspond to eigenvectors
    public native void GenerateSamplesFast(IplImage frontal, CvMat pca_hr_avg,
                                 CvMat pca_hr_eigenvectors, OneWayDescriptor pca_descriptors);

    // sets the poses and corresponding transforms
    public native void SetTransforms(CvAffinePose poses, @Cast("CvMat**") PointerPointer transforms);
    public native void SetTransforms(CvAffinePose poses, @ByPtrPtr CvMat transforms);

    // Initialize: builds a descriptor.
    // - pose_count: the number of poses to build. If poses were set externally, uses them rather than generating random ones
    // - frontal: input patch. Can be a roi in a larger image
    // - feature_name: the feature name to be associated with the descriptor
    // - norm: if 1, the affine transformed patches are normalized so that their sum is 1
    public native void Initialize(int pose_count, IplImage frontal, @Cast("const char*") BytePointer feature_name/*=0*/, int norm/*=0*/);
    public native void Initialize(int pose_count, IplImage frontal);
    public native void Initialize(int pose_count, IplImage frontal, String feature_name/*=0*/, int norm/*=0*/);

    // InitializeFast: builds a descriptor using precomputed descriptors of pca components
    // - pose_count: the number of poses to build
    // - frontal: input patch. Can be a roi in a larger image
    // - feature_name: the feature name to be associated with the descriptor
    // - pca_hr_avg: average vector for PCA
    // - pca_hr_eigenvectors: PCA eigenvectors (one vector per row)
    // - pca_descriptors: precomputed descriptors of PCA components, the first descriptor for the average vector
    // followed by the descriptors for eigenvectors
    public native void InitializeFast(int pose_count, IplImage frontal, @Cast("const char*") BytePointer feature_name,
                            CvMat pca_hr_avg, CvMat pca_hr_eigenvectors, OneWayDescriptor pca_descriptors);
    public native void InitializeFast(int pose_count, IplImage frontal, String feature_name,
                            CvMat pca_hr_avg, CvMat pca_hr_eigenvectors, OneWayDescriptor pca_descriptors);

    // ProjectPCASample: unwarps an image patch into a vector and projects it into PCA space
    // - patch: input image patch
    // - avg: PCA average vector
    // - eigenvectors: PCA eigenvectors, one per row
    // - pca_coeffs: output PCA coefficients
    public native void ProjectPCASample(IplImage patch, CvMat avg, CvMat eigenvectors, CvMat pca_coeffs);

    // InitializePCACoeffs: projects all warped patches into PCA space
    // - avg: PCA average vector
    // - eigenvectors: PCA eigenvectors, one per row
    public native void InitializePCACoeffs(CvMat avg, CvMat eigenvectors);

    // EstimatePose: finds the closest match between an input patch and a set of patches with different poses
    // - patch: input image patch
    // - pose_idx: the output index of the closest pose
    // - distance: the distance to the closest pose (L2 distance)
    public native void EstimatePose(IplImage patch, @ByRef IntPointer pose_idx, @ByRef FloatPointer distance);
    public native void EstimatePose(IplImage patch, @ByRef IntBuffer pose_idx, @ByRef FloatBuffer distance);
    public native void EstimatePose(IplImage patch, @ByRef int[] pose_idx, @ByRef float[] distance);

    // EstimatePosePCA: finds the closest match between an input patch and a set of patches with different poses.
    // The distance between patches is computed in PCA space
    // - patch: input image patch
    // - pose_idx: the output index of the closest pose
    // - distance: distance to the closest pose (L2 distance in PCA space)
    // - avg: PCA average vector. If 0, matching without PCA is used
    // - eigenvectors: PCA eigenvectors, one per row
    public native void EstimatePosePCA(CvArr patch, @ByRef IntPointer pose_idx, @ByRef FloatPointer distance, CvMat avg, CvMat eigenvalues);
    public native void EstimatePosePCA(CvArr patch, @ByRef IntBuffer pose_idx, @ByRef FloatBuffer distance, CvMat avg, CvMat eigenvalues);
    public native void EstimatePosePCA(CvArr patch, @ByRef int[] pose_idx, @ByRef float[] distance, CvMat avg, CvMat eigenvalues);

    // GetPatchSize: returns the size of each image patch after warping (2 times smaller than the input patch)
    public native @ByVal CvSize GetPatchSize();

    // GetInputPatchSize: returns the required size of the patch that the descriptor is built from
    // (2 time larger than the patch after warping)
    public native @ByVal CvSize GetInputPatchSize();

    // GetPatch: returns a patch corresponding to specified pose index
    // - index: pose index
    // - return value: the patch corresponding to specified pose index
    public native IplImage GetPatch(int index);

    // GetPose: returns a pose corresponding to specified pose index
    // - index: pose index
    // - return value: the pose corresponding to specified pose index
    public native @ByVal CvAffinePose GetPose(int index);

    // Save: saves all patches with different poses to a specified path
    public native void Save(@Cast("const char*") BytePointer path);
    public native void Save(String path);

    // ReadByName: reads a descriptor from a file storage
    // - fs: file storage
    // - parent: parent node
    // - name: node name
    // - return value: 1 if succeeded, 0 otherwise
    public native int ReadByName(CvFileStorage fs, CvFileNode parent, @Cast("const char*") BytePointer name);
    public native int ReadByName(CvFileStorage fs, CvFileNode parent, String name);

    // ReadByName: reads a descriptor from a file node
    // - parent: parent node
    // - name: node name
    // - return value: 1 if succeeded, 0 otherwise
    public native int ReadByName(@Const @ByRef FileNode parent, @Cast("const char*") BytePointer name);
    public native int ReadByName(@Const @ByRef FileNode parent, String name);

    // Write: writes a descriptor into a file storage
    // - fs: file storage
    // - name: node name
    public native void Write(CvFileStorage fs, @Cast("const char*") BytePointer name);
    public native void Write(CvFileStorage fs, String name);

    // GetFeatureName: returns a name corresponding to a feature
    public native @Cast("const char*") BytePointer GetFeatureName();

    // GetCenter: returns the center of the feature
    public native @ByVal CvPoint GetCenter();

    public native void SetPCADimHigh(int pca_dim_high);
    public native void SetPCADimLow(int pca_dim_low);

    public native int GetPCADimLow();
    public native int GetPCADimHigh();

    public native @Cast("CvMat**") PointerPointer GetPCACoeffs();
}


// OneWayDescriptorBase: encapsulates functionality for training/loading a set of one way descriptors
// and finding the nearest closest descriptor to an input feature
@Namespace("cv") @NoOffset public static class OneWayDescriptorBase extends Pointer {
    static { Loader.load(); }
    /** Empty constructor. */
    public OneWayDescriptorBase() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public OneWayDescriptorBase(Pointer p) { super(p); }


    // creates an instance of OneWayDescriptor from a set of training files
    // - patch_size: size of the input (large) patch
    // - pose_count: the number of poses to generate for each descriptor
    // - train_path: path to training files
    // - pca_config: the name of the file that contains PCA for small patches (2 times smaller
    // than patch_size each dimension
    // - pca_hr_config: the name of the file that contains PCA for large patches (of patch_size size)
    // - pca_desc_config: the name of the file that contains descriptors of PCA components
    public OneWayDescriptorBase(@ByVal CvSize patch_size, int pose_count, @Cast("const char*") BytePointer train_path/*=0*/, @Cast("const char*") BytePointer pca_config/*=0*/,
                             @Cast("const char*") BytePointer pca_hr_config/*=0*/, @Cast("const char*") BytePointer pca_desc_config/*=0*/, int pyr_levels/*=1*/,
                             int pca_dim_high/*=100*/, int pca_dim_low/*=100*/) { allocate(patch_size, pose_count, train_path, pca_config, pca_hr_config, pca_desc_config, pyr_levels, pca_dim_high, pca_dim_low); }
    private native void allocate(@ByVal CvSize patch_size, int pose_count, @Cast("const char*") BytePointer train_path/*=0*/, @Cast("const char*") BytePointer pca_config/*=0*/,
                             @Cast("const char*") BytePointer pca_hr_config/*=0*/, @Cast("const char*") BytePointer pca_desc_config/*=0*/, int pyr_levels/*=1*/,
                             int pca_dim_high/*=100*/, int pca_dim_low/*=100*/);
    public OneWayDescriptorBase(@ByVal CvSize patch_size, int pose_count) { allocate(patch_size, pose_count); }
    private native void allocate(@ByVal CvSize patch_size, int pose_count);
    public OneWayDescriptorBase(@ByVal CvSize patch_size, int pose_count, String train_path/*=0*/, String pca_config/*=0*/,
                             String pca_hr_config/*=0*/, String pca_desc_config/*=0*/, int pyr_levels/*=1*/,
                             int pca_dim_high/*=100*/, int pca_dim_low/*=100*/) { allocate(patch_size, pose_count, train_path, pca_config, pca_hr_config, pca_desc_config, pyr_levels, pca_dim_high, pca_dim_low); }
    private native void allocate(@ByVal CvSize patch_size, int pose_count, String train_path/*=0*/, String pca_config/*=0*/,
                             String pca_hr_config/*=0*/, String pca_desc_config/*=0*/, int pyr_levels/*=1*/,
                             int pca_dim_high/*=100*/, int pca_dim_low/*=100*/);

    public OneWayDescriptorBase(@ByVal CvSize patch_size, int pose_count, @StdString BytePointer pca_filename, @StdString BytePointer train_path/*=std::string()*/, @StdString BytePointer images_list/*=std::string()*/,
                             float _scale_min/*=0.7f*/, float _scale_max/*=1.5f*/, float _scale_step/*=1.2f*/, int pyr_levels/*=1*/,
                             int pca_dim_high/*=100*/, int pca_dim_low/*=100*/) { allocate(patch_size, pose_count, pca_filename, train_path, images_list, _scale_min, _scale_max, _scale_step, pyr_levels, pca_dim_high, pca_dim_low); }
    private native void allocate(@ByVal CvSize patch_size, int pose_count, @StdString BytePointer pca_filename, @StdString BytePointer train_path/*=std::string()*/, @StdString BytePointer images_list/*=std::string()*/,
                             float _scale_min/*=0.7f*/, float _scale_max/*=1.5f*/, float _scale_step/*=1.2f*/, int pyr_levels/*=1*/,
                             int pca_dim_high/*=100*/, int pca_dim_low/*=100*/);
    public OneWayDescriptorBase(@ByVal CvSize patch_size, int pose_count, @StdString BytePointer pca_filename) { allocate(patch_size, pose_count, pca_filename); }
    private native void allocate(@ByVal CvSize patch_size, int pose_count, @StdString BytePointer pca_filename);
    public OneWayDescriptorBase(@ByVal CvSize patch_size, int pose_count, @StdString String pca_filename, @StdString String train_path/*=std::string()*/, @StdString String images_list/*=std::string()*/,
                             float _scale_min/*=0.7f*/, float _scale_max/*=1.5f*/, float _scale_step/*=1.2f*/, int pyr_levels/*=1*/,
                             int pca_dim_high/*=100*/, int pca_dim_low/*=100*/) { allocate(patch_size, pose_count, pca_filename, train_path, images_list, _scale_min, _scale_max, _scale_step, pyr_levels, pca_dim_high, pca_dim_low); }
    private native void allocate(@ByVal CvSize patch_size, int pose_count, @StdString String pca_filename, @StdString String train_path/*=std::string()*/, @StdString String images_list/*=std::string()*/,
                             float _scale_min/*=0.7f*/, float _scale_max/*=1.5f*/, float _scale_step/*=1.2f*/, int pyr_levels/*=1*/,
                             int pca_dim_high/*=100*/, int pca_dim_low/*=100*/);
    public OneWayDescriptorBase(@ByVal CvSize patch_size, int pose_count, @StdString String pca_filename) { allocate(patch_size, pose_count, pca_filename); }
    private native void allocate(@ByVal CvSize patch_size, int pose_count, @StdString String pca_filename);
    public native void clear();


    // Allocate: allocates memory for a given number of descriptors
    public native void Allocate(int train_feature_count);

    // AllocatePCADescriptors: allocates memory for pca descriptors
    public native void AllocatePCADescriptors();

    // returns patch size
    public native @ByVal CvSize GetPatchSize();
    // returns the number of poses for each descriptor
    public native int GetPoseCount();

    // returns the number of pyramid levels
    public native int GetPyrLevels();

    // returns the number of descriptors
    public native int GetDescriptorCount();

    // CreateDescriptorsFromImage: creates descriptors for each of the input features
    // - src: input image
    // - features: input features
    // - pyr_levels: the number of pyramid levels
    public native void CreateDescriptorsFromImage(IplImage src, @StdVector KeyPoint features);

    // CreatePCADescriptors: generates descriptors for PCA components, needed for fast generation of feature descriptors
    public native void CreatePCADescriptors();

    // returns a feature descriptor by feature index
    public native @Const OneWayDescriptor GetDescriptor(int desc_idx);

    // FindDescriptor: finds the closest descriptor
    // - patch: input image patch
    // - desc_idx: output index of the closest descriptor to the input patch
    // - pose_idx: output index of the closest pose of the closest descriptor to the input patch
    // - distance: distance from the input patch to the closest feature pose
    // - _scales: scales of the input patch for each descriptor
    // - scale_ranges: input scales variation (float[2])
    public native void FindDescriptor(IplImage patch, @ByRef IntPointer desc_idx, @ByRef IntPointer pose_idx, @ByRef FloatPointer distance, FloatPointer _scale/*=0*/, FloatPointer scale_ranges/*=0*/);
    public native void FindDescriptor(IplImage patch, @ByRef IntPointer desc_idx, @ByRef IntPointer pose_idx, @ByRef FloatPointer distance);
    public native void FindDescriptor(IplImage patch, @ByRef IntBuffer desc_idx, @ByRef IntBuffer pose_idx, @ByRef FloatBuffer distance, FloatBuffer _scale/*=0*/, FloatBuffer scale_ranges/*=0*/);
    public native void FindDescriptor(IplImage patch, @ByRef IntBuffer desc_idx, @ByRef IntBuffer pose_idx, @ByRef FloatBuffer distance);
    public native void FindDescriptor(IplImage patch, @ByRef int[] desc_idx, @ByRef int[] pose_idx, @ByRef float[] distance, float[] _scale/*=0*/, float[] scale_ranges/*=0*/);
    public native void FindDescriptor(IplImage patch, @ByRef int[] desc_idx, @ByRef int[] pose_idx, @ByRef float[] distance);

    // - patch: input image patch
    // - n: number of the closest indexes
    // - desc_idxs: output indexes of the closest descriptor to the input patch (n)
    // - pose_idx: output indexes of the closest pose of the closest descriptor to the input patch (n)
    // - distances: distance from the input patch to the closest feature pose (n)
    // - _scales: scales of the input patch
    // - scale_ranges: input scales variation (float[2])
    public native void FindDescriptor(IplImage patch, int n, @StdVector IntPointer desc_idxs, @StdVector IntPointer pose_idxs,
                            @StdVector FloatPointer distances, @StdVector FloatPointer _scales, FloatPointer scale_ranges/*=0*/);
    public native void FindDescriptor(IplImage patch, int n, @StdVector IntPointer desc_idxs, @StdVector IntPointer pose_idxs,
                            @StdVector FloatPointer distances, @StdVector FloatPointer _scales);
    public native void FindDescriptor(IplImage patch, int n, @StdVector IntBuffer desc_idxs, @StdVector IntBuffer pose_idxs,
                            @StdVector FloatBuffer distances, @StdVector FloatBuffer _scales, FloatBuffer scale_ranges/*=0*/);
    public native void FindDescriptor(IplImage patch, int n, @StdVector IntBuffer desc_idxs, @StdVector IntBuffer pose_idxs,
                            @StdVector FloatBuffer distances, @StdVector FloatBuffer _scales);
    public native void FindDescriptor(IplImage patch, int n, @StdVector int[] desc_idxs, @StdVector int[] pose_idxs,
                            @StdVector float[] distances, @StdVector float[] _scales, float[] scale_ranges/*=0*/);
    public native void FindDescriptor(IplImage patch, int n, @StdVector int[] desc_idxs, @StdVector int[] pose_idxs,
                            @StdVector float[] distances, @StdVector float[] _scales);

    // FindDescriptor: finds the closest descriptor
    // - src: input image
    // - pt: center of the feature
    // - desc_idx: output index of the closest descriptor to the input patch
    // - pose_idx: output index of the closest pose of the closest descriptor to the input patch
    // - distance: distance from the input patch to the closest feature pose
    public native void FindDescriptor(IplImage src, @ByVal Point2f pt, @ByRef IntPointer desc_idx, @ByRef IntPointer pose_idx, @ByRef FloatPointer distance);
    public native void FindDescriptor(IplImage src, @ByVal Point2f pt, @ByRef IntBuffer desc_idx, @ByRef IntBuffer pose_idx, @ByRef FloatBuffer distance);
    public native void FindDescriptor(IplImage src, @ByVal Point2f pt, @ByRef int[] desc_idx, @ByRef int[] pose_idx, @ByRef float[] distance);

    // InitializePoses: generates random poses
    public native void InitializePoses();

    // InitializeTransformsFromPoses: generates 2x3 affine matrices from poses (initializes m_transforms)
    public native void InitializeTransformsFromPoses();

    // InitializePoseTransforms: subsequently calls InitializePoses and InitializeTransformsFromPoses
    public native void InitializePoseTransforms();

    // InitializeDescriptor: initializes a descriptor
    // - desc_idx: descriptor index
    // - train_image: image patch (ROI is supported)
    // - feature_label: feature textual label
    public native void InitializeDescriptor(int desc_idx, IplImage train_image, @Cast("const char*") BytePointer feature_label);
    public native void InitializeDescriptor(int desc_idx, IplImage train_image, String feature_label);

    public native void InitializeDescriptor(int desc_idx, IplImage train_image, @Const @ByRef KeyPoint keypoint, @Cast("const char*") BytePointer feature_label);
    public native void InitializeDescriptor(int desc_idx, IplImage train_image, @Const @ByRef KeyPoint keypoint, String feature_label);

    // InitializeDescriptors: load features from an image and create descriptors for each of them
    public native void InitializeDescriptors(IplImage train_image, @StdVector KeyPoint features,
                                   @Cast("const char*") BytePointer feature_label/*=""*/, int desc_start_idx/*=0*/);
    public native void InitializeDescriptors(IplImage train_image, @StdVector KeyPoint features);
    public native void InitializeDescriptors(IplImage train_image, @StdVector KeyPoint features,
                                   String feature_label/*=""*/, int desc_start_idx/*=0*/);

    // Write: writes this object to a file storage
    // - fs: output filestorage
    public native void Write(@ByRef FileStorage fs);

    // Read: reads OneWayDescriptorBase object from a file node
    // - fn: input file node
    public native void Read(@Const @ByRef FileNode fn);

    // LoadPCADescriptors: loads PCA descriptors from a file
    // - filename: input filename
    public native int LoadPCADescriptors(@Cast("const char*") BytePointer filename);
    public native int LoadPCADescriptors(String filename);

    // LoadPCADescriptors: loads PCA descriptors from a file node
    // - fn: input file node
    public native int LoadPCADescriptors(@Const @ByRef FileNode fn);

    // SavePCADescriptors: saves PCA descriptors to a file
    // - filename: output filename
    public native void SavePCADescriptors(@Cast("const char*") BytePointer filename);
    public native void SavePCADescriptors(String filename);

    // SavePCADescriptors: saves PCA descriptors to a file storage
    // - fs: output file storage
    public native void SavePCADescriptors(CvFileStorage fs);

    // GeneratePCA: calculate and save PCA components and descriptors
    // - img_path: path to training PCA images directory
    // - images_list: filename with filenames of training PCA images
    public native void GeneratePCA(@Cast("const char*") BytePointer img_path, @Cast("const char*") BytePointer images_list, int pose_count/*=500*/);
    public native void GeneratePCA(@Cast("const char*") BytePointer img_path, @Cast("const char*") BytePointer images_list);
    public native void GeneratePCA(String img_path, String images_list, int pose_count/*=500*/);
    public native void GeneratePCA(String img_path, String images_list);

    // SetPCAHigh: sets the high resolution pca matrices (copied to internal structures)
    public native void SetPCAHigh(CvMat avg, CvMat eigenvectors);

    // SetPCALow: sets the low resolution pca matrices (copied to internal structures)
    public native void SetPCALow(CvMat avg, CvMat eigenvectors);

    public native int GetLowPCA(@Cast("CvMat**") PointerPointer avg, @Cast("CvMat**") PointerPointer eigenvectors);
    public native int GetLowPCA(@ByPtrPtr CvMat avg, @ByPtrPtr CvMat eigenvectors);

    public native int GetPCADimLow();
    public native int GetPCADimHigh();

     // Converting pca_descriptors array to KD tree

    // GetPCAFilename: get default PCA filename
    public static native @StdString BytePointer GetPCAFilename();

    public native @Cast("bool") boolean empty();
}

@Namespace("cv") @NoOffset public static class OneWayDescriptorObject extends OneWayDescriptorBase {
    static { Loader.load(); }
    /** Empty constructor. */
    public OneWayDescriptorObject() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public OneWayDescriptorObject(Pointer p) { super(p); }

    // creates an instance of OneWayDescriptorObject from a set of training files
    // - patch_size: size of the input (large) patch
    // - pose_count: the number of poses to generate for each descriptor
    // - train_path: path to training files
    // - pca_config: the name of the file that contains PCA for small patches (2 times smaller
    // than patch_size each dimension
    // - pca_hr_config: the name of the file that contains PCA for large patches (of patch_size size)
    // - pca_desc_config: the name of the file that contains descriptors of PCA components
    public OneWayDescriptorObject(@ByVal CvSize patch_size, int pose_count, @Cast("const char*") BytePointer train_path, @Cast("const char*") BytePointer pca_config,
                               @Cast("const char*") BytePointer pca_hr_config/*=0*/, @Cast("const char*") BytePointer pca_desc_config/*=0*/, int pyr_levels/*=1*/) { allocate(patch_size, pose_count, train_path, pca_config, pca_hr_config, pca_desc_config, pyr_levels); }
    private native void allocate(@ByVal CvSize patch_size, int pose_count, @Cast("const char*") BytePointer train_path, @Cast("const char*") BytePointer pca_config,
                               @Cast("const char*") BytePointer pca_hr_config/*=0*/, @Cast("const char*") BytePointer pca_desc_config/*=0*/, int pyr_levels/*=1*/);
    public OneWayDescriptorObject(@ByVal CvSize patch_size, int pose_count, @Cast("const char*") BytePointer train_path, @Cast("const char*") BytePointer pca_config) { allocate(patch_size, pose_count, train_path, pca_config); }
    private native void allocate(@ByVal CvSize patch_size, int pose_count, @Cast("const char*") BytePointer train_path, @Cast("const char*") BytePointer pca_config);
    public OneWayDescriptorObject(@ByVal CvSize patch_size, int pose_count, String train_path, String pca_config,
                               String pca_hr_config/*=0*/, String pca_desc_config/*=0*/, int pyr_levels/*=1*/) { allocate(patch_size, pose_count, train_path, pca_config, pca_hr_config, pca_desc_config, pyr_levels); }
    private native void allocate(@ByVal CvSize patch_size, int pose_count, String train_path, String pca_config,
                               String pca_hr_config/*=0*/, String pca_desc_config/*=0*/, int pyr_levels/*=1*/);
    public OneWayDescriptorObject(@ByVal CvSize patch_size, int pose_count, String train_path, String pca_config) { allocate(patch_size, pose_count, train_path, pca_config); }
    private native void allocate(@ByVal CvSize patch_size, int pose_count, String train_path, String pca_config);

    public OneWayDescriptorObject(@ByVal CvSize patch_size, int pose_count, @StdString BytePointer pca_filename,
                               @StdString BytePointer train_path/*=std::string ()*/, @StdString BytePointer images_list/*=std::string ()*/,
                               float _scale_min/*=0.7f*/, float _scale_max/*=1.5f*/, float _scale_step/*=1.2f*/, int pyr_levels/*=1*/) { allocate(patch_size, pose_count, pca_filename, train_path, images_list, _scale_min, _scale_max, _scale_step, pyr_levels); }
    private native void allocate(@ByVal CvSize patch_size, int pose_count, @StdString BytePointer pca_filename,
                               @StdString BytePointer train_path/*=std::string ()*/, @StdString BytePointer images_list/*=std::string ()*/,
                               float _scale_min/*=0.7f*/, float _scale_max/*=1.5f*/, float _scale_step/*=1.2f*/, int pyr_levels/*=1*/);
    public OneWayDescriptorObject(@ByVal CvSize patch_size, int pose_count, @StdString BytePointer pca_filename) { allocate(patch_size, pose_count, pca_filename); }
    private native void allocate(@ByVal CvSize patch_size, int pose_count, @StdString BytePointer pca_filename);
    public OneWayDescriptorObject(@ByVal CvSize patch_size, int pose_count, @StdString String pca_filename,
                               @StdString String train_path/*=std::string ()*/, @StdString String images_list/*=std::string ()*/,
                               float _scale_min/*=0.7f*/, float _scale_max/*=1.5f*/, float _scale_step/*=1.2f*/, int pyr_levels/*=1*/) { allocate(patch_size, pose_count, pca_filename, train_path, images_list, _scale_min, _scale_max, _scale_step, pyr_levels); }
    private native void allocate(@ByVal CvSize patch_size, int pose_count, @StdString String pca_filename,
                               @StdString String train_path/*=std::string ()*/, @StdString String images_list/*=std::string ()*/,
                               float _scale_min/*=0.7f*/, float _scale_max/*=1.5f*/, float _scale_step/*=1.2f*/, int pyr_levels/*=1*/);
    public OneWayDescriptorObject(@ByVal CvSize patch_size, int pose_count, @StdString String pca_filename) { allocate(patch_size, pose_count, pca_filename); }
    private native void allocate(@ByVal CvSize patch_size, int pose_count, @StdString String pca_filename);

    // Allocate: allocates memory for a given number of features
    // - train_feature_count: the total number of features
    // - object_feature_count: the number of features extracted from the object
    public native void Allocate(int train_feature_count, int object_feature_count);


    public native void SetLabeledFeatures(@StdVector KeyPoint features);
    public native @StdVector KeyPoint GetLabeledFeatures();
    public native @StdVector KeyPoint _GetLabeledFeatures();

    // IsDescriptorObject: returns 1 if descriptor with specified index is positive, otherwise 0
    public native int IsDescriptorObject(int desc_idx);

    // MatchPointToPart: returns the part number of a feature if it matches one of the object parts, otherwise -1
    public native int MatchPointToPart(@ByVal CvPoint pt);
    public native int MatchPointToPart(@ByVal @Cast("CvPoint*") IntBuffer pt);
    public native int MatchPointToPart(@ByVal @Cast("CvPoint*") int[] pt);

    // GetDescriptorPart: returns the part number of the feature corresponding to a specified descriptor
    // - desc_idx: descriptor index
    public native int GetDescriptorPart(int desc_idx);


    public native void InitializeObjectDescriptors(IplImage train_image, @StdVector KeyPoint features,
                                         @Cast("const char*") BytePointer feature_label, int desc_start_idx/*=0*/, float scale/*=1.0f*/,
                                         int is_background/*=0*/);
    public native void InitializeObjectDescriptors(IplImage train_image, @StdVector KeyPoint features,
                                         @Cast("const char*") BytePointer feature_label);
    public native void InitializeObjectDescriptors(IplImage train_image, @StdVector KeyPoint features,
                                         String feature_label, int desc_start_idx/*=0*/, float scale/*=1.0f*/,
                                         int is_background/*=0*/);
    public native void InitializeObjectDescriptors(IplImage train_image, @StdVector KeyPoint features,
                                         String feature_label);

    // GetObjectFeatureCount: returns the number of object features
    public native int GetObjectFeatureCount();

}


/*
 *  OneWayDescriptorMatcher
 */

@Namespace("cv") @NoOffset public static class OneWayDescriptorMatcher extends GenericDescriptorMatcher {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public OneWayDescriptorMatcher(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public OneWayDescriptorMatcher(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public OneWayDescriptorMatcher position(int position) {
        return (OneWayDescriptorMatcher)super.position(position);
    }

    @NoOffset public static class Params extends Pointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public Params(Pointer p) { super(p); }
        /** Native array allocator. Access with {@link Pointer#position(int)}. */
        public Params(int size) { allocateArray(size); }
        private native void allocateArray(int size);
        @Override public Params position(int position) {
            return (Params)super.position(position);
        }
    
        @MemberGetter public static native int POSE_COUNT();
        @MemberGetter public static native int PATCH_WIDTH();
        @MemberGetter public static native int PATCH_HEIGHT();
        public static native float GET_MIN_SCALE();
        public static native float GET_MAX_SCALE();
        public static native float GET_STEP_SCALE();

        public Params( int poseCount/*=cv::OneWayDescriptorMatcher::Params::POSE_COUNT*/,
                       @ByVal(nullValue = "cv::Size(cv::OneWayDescriptorMatcher::Params::PATCH_WIDTH, cv::OneWayDescriptorMatcher::Params::PATCH_HEIGHT)") Size patchSize/*=cv::Size(cv::OneWayDescriptorMatcher::Params::PATCH_WIDTH, cv::OneWayDescriptorMatcher::Params::PATCH_HEIGHT)*/,
                       @StdString BytePointer pcaFilename/*=std::string()*/,
                       @StdString BytePointer trainPath/*=std::string()*/, @StdString BytePointer trainImagesList/*=std::string()*/,
                       float minScale/*=cv::OneWayDescriptorMatcher::Params::GET_MIN_SCALE()*/, float maxScale/*=cv::OneWayDescriptorMatcher::Params::GET_MAX_SCALE()*/,
                       float stepScale/*=cv::OneWayDescriptorMatcher::Params::GET_STEP_SCALE()*/ ) { allocate(poseCount, patchSize, pcaFilename, trainPath, trainImagesList, minScale, maxScale, stepScale); }
        private native void allocate( int poseCount/*=cv::OneWayDescriptorMatcher::Params::POSE_COUNT*/,
                       @ByVal(nullValue = "cv::Size(cv::OneWayDescriptorMatcher::Params::PATCH_WIDTH, cv::OneWayDescriptorMatcher::Params::PATCH_HEIGHT)") Size patchSize/*=cv::Size(cv::OneWayDescriptorMatcher::Params::PATCH_WIDTH, cv::OneWayDescriptorMatcher::Params::PATCH_HEIGHT)*/,
                       @StdString BytePointer pcaFilename/*=std::string()*/,
                       @StdString BytePointer trainPath/*=std::string()*/, @StdString BytePointer trainImagesList/*=std::string()*/,
                       float minScale/*=cv::OneWayDescriptorMatcher::Params::GET_MIN_SCALE()*/, float maxScale/*=cv::OneWayDescriptorMatcher::Params::GET_MAX_SCALE()*/,
                       float stepScale/*=cv::OneWayDescriptorMatcher::Params::GET_STEP_SCALE()*/ );
        public Params( ) { allocate(); }
        private native void allocate( );
        public Params( int poseCount/*=cv::OneWayDescriptorMatcher::Params::POSE_COUNT*/,
                       @ByVal(nullValue = "cv::Size(cv::OneWayDescriptorMatcher::Params::PATCH_WIDTH, cv::OneWayDescriptorMatcher::Params::PATCH_HEIGHT)") Size patchSize/*=cv::Size(cv::OneWayDescriptorMatcher::Params::PATCH_WIDTH, cv::OneWayDescriptorMatcher::Params::PATCH_HEIGHT)*/,
                       @StdString String pcaFilename/*=std::string()*/,
                       @StdString String trainPath/*=std::string()*/, @StdString String trainImagesList/*=std::string()*/,
                       float minScale/*=cv::OneWayDescriptorMatcher::Params::GET_MIN_SCALE()*/, float maxScale/*=cv::OneWayDescriptorMatcher::Params::GET_MAX_SCALE()*/,
                       float stepScale/*=cv::OneWayDescriptorMatcher::Params::GET_STEP_SCALE()*/ ) { allocate(poseCount, patchSize, pcaFilename, trainPath, trainImagesList, minScale, maxScale, stepScale); }
        private native void allocate( int poseCount/*=cv::OneWayDescriptorMatcher::Params::POSE_COUNT*/,
                       @ByVal(nullValue = "cv::Size(cv::OneWayDescriptorMatcher::Params::PATCH_WIDTH, cv::OneWayDescriptorMatcher::Params::PATCH_HEIGHT)") Size patchSize/*=cv::Size(cv::OneWayDescriptorMatcher::Params::PATCH_WIDTH, cv::OneWayDescriptorMatcher::Params::PATCH_HEIGHT)*/,
                       @StdString String pcaFilename/*=std::string()*/,
                       @StdString String trainPath/*=std::string()*/, @StdString String trainImagesList/*=std::string()*/,
                       float minScale/*=cv::OneWayDescriptorMatcher::Params::GET_MIN_SCALE()*/, float maxScale/*=cv::OneWayDescriptorMatcher::Params::GET_MAX_SCALE()*/,
                       float stepScale/*=cv::OneWayDescriptorMatcher::Params::GET_STEP_SCALE()*/ );

        public native int poseCount(); public native Params poseCount(int poseCount);
        public native @ByRef Size patchSize(); public native Params patchSize(Size patchSize);
        public native @StdString BytePointer pcaFilename(); public native Params pcaFilename(BytePointer pcaFilename);
        public native @StdString BytePointer trainPath(); public native Params trainPath(BytePointer trainPath);
        public native @StdString BytePointer trainImagesList(); public native Params trainImagesList(BytePointer trainImagesList);

        public native float minScale(); public native Params minScale(float minScale);
        public native float maxScale(); public native Params maxScale(float maxScale);
        public native float stepScale(); public native Params stepScale(float stepScale);
    }

    public OneWayDescriptorMatcher( @Const @ByRef(nullValue = "cv::OneWayDescriptorMatcher::Params()") Params params/*=cv::OneWayDescriptorMatcher::Params()*/ ) { allocate(params); }
    private native void allocate( @Const @ByRef(nullValue = "cv::OneWayDescriptorMatcher::Params()") Params params/*=cv::OneWayDescriptorMatcher::Params()*/ );
    public OneWayDescriptorMatcher( ) { allocate(); }
    private native void allocate( );

    public native void initialize( @Const @ByRef Params params, @Ptr OneWayDescriptorBase base/*=cv::Ptr<cv::OneWayDescriptorBase>()*/ );
    public native void initialize( @Const @ByRef Params params );

    // Clears keypoints storing in collection and OneWayDescriptorBase
    public native void clear();

    public native void train();

    public native @Cast("bool") boolean isMaskSupported();

    public native void read( @Const @ByRef FileNode fn );
    public native void write( @ByRef FileStorage fs );

    public native @Cast("bool") boolean empty();

    public native @Ptr GenericDescriptorMatcher clone( @Cast("bool") boolean emptyTrainData/*=false*/ );
    public native @Ptr GenericDescriptorMatcher clone( );
}

/*
 *  FernDescriptorMatcher
 */

@Namespace("cv") @NoOffset public static class FernDescriptorMatcher extends GenericDescriptorMatcher {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FernDescriptorMatcher(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public FernDescriptorMatcher(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public FernDescriptorMatcher position(int position) {
        return (FernDescriptorMatcher)super.position(position);
    }

    @NoOffset public static class Params extends Pointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public Params(Pointer p) { super(p); }
        /** Native array allocator. Access with {@link Pointer#position(int)}. */
        public Params(int size) { allocateArray(size); }
        private native void allocateArray(int size);
        @Override public Params position(int position) {
            return (Params)super.position(position);
        }
    
        public Params( int nclasses/*=0*/,
                       int patchSize/*=cv::FernClassifier::PATCH_SIZE*/,
                       int signatureSize/*=cv::FernClassifier::DEFAULT_SIGNATURE_SIZE*/,
                       int nstructs/*=cv::FernClassifier::DEFAULT_STRUCTS*/,
                       int structSize/*=cv::FernClassifier::DEFAULT_STRUCT_SIZE*/,
                       int nviews/*=cv::FernClassifier::DEFAULT_VIEWS*/,
                       int compressionMethod/*=cv::FernClassifier::COMPRESSION_NONE*/,
                       @Const @ByRef(nullValue = "cv::PatchGenerator()") PatchGenerator patchGenerator/*=cv::PatchGenerator()*/ ) { allocate(nclasses, patchSize, signatureSize, nstructs, structSize, nviews, compressionMethod, patchGenerator); }
        private native void allocate( int nclasses/*=0*/,
                       int patchSize/*=cv::FernClassifier::PATCH_SIZE*/,
                       int signatureSize/*=cv::FernClassifier::DEFAULT_SIGNATURE_SIZE*/,
                       int nstructs/*=cv::FernClassifier::DEFAULT_STRUCTS*/,
                       int structSize/*=cv::FernClassifier::DEFAULT_STRUCT_SIZE*/,
                       int nviews/*=cv::FernClassifier::DEFAULT_VIEWS*/,
                       int compressionMethod/*=cv::FernClassifier::COMPRESSION_NONE*/,
                       @Const @ByRef(nullValue = "cv::PatchGenerator()") PatchGenerator patchGenerator/*=cv::PatchGenerator()*/ );
        public Params( ) { allocate(); }
        private native void allocate( );

        public Params( @StdString BytePointer filename ) { allocate(filename); }
        private native void allocate( @StdString BytePointer filename );
        public Params( @StdString String filename ) { allocate(filename); }
        private native void allocate( @StdString String filename );

        public native int nclasses(); public native Params nclasses(int nclasses);
        public native int patchSize(); public native Params patchSize(int patchSize);
        public native int signatureSize(); public native Params signatureSize(int signatureSize);
        public native int nstructs(); public native Params nstructs(int nstructs);
        public native int structSize(); public native Params structSize(int structSize);
        public native int nviews(); public native Params nviews(int nviews);
        public native int compressionMethod(); public native Params compressionMethod(int compressionMethod);
        public native @ByRef PatchGenerator patchGenerator(); public native Params patchGenerator(PatchGenerator patchGenerator);

        public native @StdString BytePointer filename(); public native Params filename(BytePointer filename);
    }

    public FernDescriptorMatcher( @Const @ByRef(nullValue = "cv::FernDescriptorMatcher::Params()") Params params/*=cv::FernDescriptorMatcher::Params()*/ ) { allocate(params); }
    private native void allocate( @Const @ByRef(nullValue = "cv::FernDescriptorMatcher::Params()") Params params/*=cv::FernDescriptorMatcher::Params()*/ );
    public FernDescriptorMatcher( ) { allocate(); }
    private native void allocate( );

    public native void clear();

    public native void train();

    public native @Cast("bool") boolean isMaskSupported();

    public native void read( @Const @ByRef FileNode fn );
    public native void write( @ByRef FileStorage fs );
    public native @Cast("bool") boolean empty();

    public native @Ptr GenericDescriptorMatcher clone( @Cast("bool") boolean emptyTrainData/*=false*/ );
    public native @Ptr GenericDescriptorMatcher clone( );
}


/*
 * CalonderDescriptorExtractor
 */












////////////////////// Brute Force Matcher //////////////////////////


/****************************************************************************************\
*                                Planar Object Detection                                 *
\****************************************************************************************/

@Namespace("cv") @NoOffset public static class PlanarObjectDetector extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public PlanarObjectDetector(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public PlanarObjectDetector(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public PlanarObjectDetector position(int position) {
        return (PlanarObjectDetector)super.position(position);
    }

    public PlanarObjectDetector() { allocate(); }
    private native void allocate();
    public PlanarObjectDetector(@Const @ByRef FileNode node) { allocate(node); }
    private native void allocate(@Const @ByRef FileNode node);
    public PlanarObjectDetector(@Const @ByRef MatVector pyr, int _npoints/*=300*/,
                             int _patchSize/*=cv::FernClassifier::PATCH_SIZE*/,
                             int _nstructs/*=cv::FernClassifier::DEFAULT_STRUCTS*/,
                             int _structSize/*=cv::FernClassifier::DEFAULT_STRUCT_SIZE*/,
                             int _nviews/*=cv::FernClassifier::DEFAULT_VIEWS*/,
                             @Const @ByRef(nullValue = "cv::LDetector()") LDetector detector/*=cv::LDetector()*/,
                             @Const @ByRef(nullValue = "cv::PatchGenerator()") PatchGenerator patchGenerator/*=cv::PatchGenerator()*/) { allocate(pyr, _npoints, _patchSize, _nstructs, _structSize, _nviews, detector, patchGenerator); }
    private native void allocate(@Const @ByRef MatVector pyr, int _npoints/*=300*/,
                             int _patchSize/*=cv::FernClassifier::PATCH_SIZE*/,
                             int _nstructs/*=cv::FernClassifier::DEFAULT_STRUCTS*/,
                             int _structSize/*=cv::FernClassifier::DEFAULT_STRUCT_SIZE*/,
                             int _nviews/*=cv::FernClassifier::DEFAULT_VIEWS*/,
                             @Const @ByRef(nullValue = "cv::LDetector()") LDetector detector/*=cv::LDetector()*/,
                             @Const @ByRef(nullValue = "cv::PatchGenerator()") PatchGenerator patchGenerator/*=cv::PatchGenerator()*/);
    public PlanarObjectDetector(@Const @ByRef MatVector pyr) { allocate(pyr); }
    private native void allocate(@Const @ByRef MatVector pyr);
    public native void train(@Const @ByRef MatVector pyr, int _npoints/*=300*/,
                           int _patchSize/*=cv::FernClassifier::PATCH_SIZE*/,
                           int _nstructs/*=cv::FernClassifier::DEFAULT_STRUCTS*/,
                           int _structSize/*=cv::FernClassifier::DEFAULT_STRUCT_SIZE*/,
                           int _nviews/*=cv::FernClassifier::DEFAULT_VIEWS*/,
                           @Const @ByRef(nullValue = "cv::LDetector()") LDetector detector/*=cv::LDetector()*/,
                           @Const @ByRef(nullValue = "cv::PatchGenerator()") PatchGenerator patchGenerator/*=cv::PatchGenerator()*/);
    public native void train(@Const @ByRef MatVector pyr);
    public native void train(@Const @ByRef MatVector pyr, @StdVector KeyPoint keypoints,
                           int _patchSize/*=cv::FernClassifier::PATCH_SIZE*/,
                           int _nstructs/*=cv::FernClassifier::DEFAULT_STRUCTS*/,
                           int _structSize/*=cv::FernClassifier::DEFAULT_STRUCT_SIZE*/,
                           int _nviews/*=cv::FernClassifier::DEFAULT_VIEWS*/,
                           @Const @ByRef(nullValue = "cv::LDetector()") LDetector detector/*=cv::LDetector()*/,
                           @Const @ByRef(nullValue = "cv::PatchGenerator()") PatchGenerator patchGenerator/*=cv::PatchGenerator()*/);
    public native void train(@Const @ByRef MatVector pyr, @StdVector KeyPoint keypoints);
    
    public native @StdVector KeyPoint getModelPoints();
    
    
    public native void setVerbose(@Cast("bool") boolean verbose);

    public native void read(@Const @ByRef FileNode node);
    public native void write(@ByRef FileStorage fs, @StdString BytePointer name/*=cv::String()*/);
    public native void write(@ByRef FileStorage fs);
    public native void write(@ByRef FileStorage fs, @StdString String name/*=cv::String()*/);
    public native @Cast("bool") @Name("operator()") boolean apply(@Const @ByRef Mat image, @ByRef Mat H, @StdVector Point2f corners);
    public native @Cast("bool") @Name("operator()") boolean apply(@Const @ByRef MatVector pyr, @StdVector KeyPoint keypoints,
                        @ByRef Mat H, @StdVector Point2f corners,
                        @StdVector IntPointer pairs/*=0*/);
    public native @Cast("bool") @Name("operator()") boolean apply(@Const @ByRef MatVector pyr, @StdVector KeyPoint keypoints,
                        @ByRef Mat H, @StdVector Point2f corners);
    public native @Cast("bool") @Name("operator()") boolean apply(@Const @ByRef MatVector pyr, @StdVector KeyPoint keypoints,
                        @ByRef Mat H, @StdVector Point2f corners,
                        @StdVector IntBuffer pairs/*=0*/);
    public native @Cast("bool") @Name("operator()") boolean apply(@Const @ByRef MatVector pyr, @StdVector KeyPoint keypoints,
                        @ByRef Mat H, @StdVector Point2f corners,
                        @StdVector int[] pairs/*=0*/);
}



// 2009-01-12, Xavier Delacour <xavier.delacour@gmail.com>

public static class lsh_hash extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public lsh_hash() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public lsh_hash(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public lsh_hash(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public lsh_hash position(int position) {
        return (lsh_hash)super.position(position);
    }

    public native int h1(); public native lsh_hash h1(int h1);
    public native int h2(); public native lsh_hash h2(int h2);
}

public static class CvLSHOperations extends Pointer {
    static { Loader.load(); }
    /** Empty constructor. */
    public CvLSHOperations() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvLSHOperations(Pointer p) { super(p); }


    public native int vector_add(@Const Pointer data);
    public native void vector_remove(int i);
    public native @Const Pointer vector_lookup(int i);
    public native void vector_reserve(int n);
    public native @Cast("unsigned int") int vector_count();

    public native void hash_insert(@ByVal lsh_hash h, int l, int i);
    public native void hash_remove(@ByVal lsh_hash h, int l, int i);
    public native int hash_lookup(@ByVal lsh_hash h, int l, IntPointer ret_i, int ret_i_max);
    public native int hash_lookup(@ByVal lsh_hash h, int l, IntBuffer ret_i, int ret_i_max);
    public native int hash_lookup(@ByVal lsh_hash h, int l, int[] ret_i, int ret_i_max);
}

// #endif

// #ifdef __cplusplus
// #endif

/* Splits color or grayscale image into multiple connected components
 of nearly the same color/brightness using modification of Burt algorithm.
 comp with contain a pointer to sequence (CvSeq)
 of connected components (CvConnectedComp) */
public static native void cvPyrSegmentation( IplImage src, IplImage dst,
                              CvMemStorage storage, @Cast("CvSeq**") PointerPointer comp,
                              int level, double threshold1,
                              double threshold2 );
public static native void cvPyrSegmentation( IplImage src, IplImage dst,
                              CvMemStorage storage, @ByPtrPtr CvSeq comp,
                              int level, double threshold1,
                              double threshold2 );

/****************************************************************************************\
*                              Planar subdivisions                                       *
\****************************************************************************************/

/* Initializes Delaunay triangulation */
public static native void cvInitSubdivDelaunay2D( CvSubdiv2D subdiv, @ByVal CvRect rect );

/* Creates new subdivision */
public static native CvSubdiv2D cvCreateSubdiv2D( int subdiv_type, int header_size,
                                     int vtx_size, int quadedge_size,
                                     CvMemStorage storage );

/************************* high-level subdivision functions ***************************/

/* Simplified Delaunay diagram creation */
public static native CvSubdiv2D cvCreateSubdivDelaunay2D( @ByVal CvRect rect, CvMemStorage storage );


/* Inserts new point to the Delaunay triangulation */
public static native CvSubdiv2DPoint cvSubdivDelaunay2DInsert( CvSubdiv2D subdiv, @ByVal CvPoint2D32f pt);
public static native CvSubdiv2DPoint cvSubdivDelaunay2DInsert( CvSubdiv2D subdiv, @ByVal @Cast("CvPoint2D32f*") FloatBuffer pt);
public static native CvSubdiv2DPoint cvSubdivDelaunay2DInsert( CvSubdiv2D subdiv, @ByVal @Cast("CvPoint2D32f*") float[] pt);

/* Locates a point within the Delaunay triangulation (finds the edge
 the point is left to or belongs to, or the triangulation point the given
 point coinsides with */
public static native @Cast("CvSubdiv2DPointLocation") int cvSubdiv2DLocate(
                                                 CvSubdiv2D subdiv, @ByVal CvPoint2D32f pt,
                                                 @Cast("CvSubdiv2DEdge*") SizeTPointer edge,
                                                 @Cast("CvSubdiv2DPoint**") PointerPointer vertex/*=NULL*/ );
public static native @Cast("CvSubdiv2DPointLocation") int cvSubdiv2DLocate(
                                                 CvSubdiv2D subdiv, @ByVal CvPoint2D32f pt,
                                                 @Cast("CvSubdiv2DEdge*") SizeTPointer edge );
public static native @Cast("CvSubdiv2DPointLocation") int cvSubdiv2DLocate(
                                                 CvSubdiv2D subdiv, @ByVal CvPoint2D32f pt,
                                                 @Cast("CvSubdiv2DEdge*") SizeTPointer edge,
                                                 @ByPtrPtr CvSubdiv2DPoint vertex/*=NULL*/ );
public static native @Cast("CvSubdiv2DPointLocation") int cvSubdiv2DLocate(
                                                 CvSubdiv2D subdiv, @ByVal @Cast("CvPoint2D32f*") FloatBuffer pt,
                                                 @Cast("CvSubdiv2DEdge*") SizeTPointer edge,
                                                 @ByPtrPtr CvSubdiv2DPoint vertex/*=NULL*/ );
public static native @Cast("CvSubdiv2DPointLocation") int cvSubdiv2DLocate(
                                                 CvSubdiv2D subdiv, @ByVal @Cast("CvPoint2D32f*") FloatBuffer pt,
                                                 @Cast("CvSubdiv2DEdge*") SizeTPointer edge );
public static native @Cast("CvSubdiv2DPointLocation") int cvSubdiv2DLocate(
                                                 CvSubdiv2D subdiv, @ByVal @Cast("CvPoint2D32f*") float[] pt,
                                                 @Cast("CvSubdiv2DEdge*") SizeTPointer edge,
                                                 @ByPtrPtr CvSubdiv2DPoint vertex/*=NULL*/ );
public static native @Cast("CvSubdiv2DPointLocation") int cvSubdiv2DLocate(
                                                 CvSubdiv2D subdiv, @ByVal @Cast("CvPoint2D32f*") float[] pt,
                                                 @Cast("CvSubdiv2DEdge*") SizeTPointer edge );

/* Calculates Voronoi tesselation (i.e. coordinates of Voronoi points) */
public static native void cvCalcSubdivVoronoi2D( CvSubdiv2D subdiv );


/* Removes all Voronoi points from the tesselation */
public static native void cvClearSubdivVoronoi2D( CvSubdiv2D subdiv );


/* Finds the nearest to the given point vertex in subdivision. */
public static native CvSubdiv2DPoint cvFindNearestPoint2D( CvSubdiv2D subdiv, @ByVal CvPoint2D32f pt );
public static native CvSubdiv2DPoint cvFindNearestPoint2D( CvSubdiv2D subdiv, @ByVal @Cast("CvPoint2D32f*") FloatBuffer pt );
public static native CvSubdiv2DPoint cvFindNearestPoint2D( CvSubdiv2D subdiv, @ByVal @Cast("CvPoint2D32f*") float[] pt );


/************ Basic quad-edge navigation and operations ************/

public static native @Cast("CvSubdiv2DEdge") long cvSubdiv2DNextEdge( @Cast("CvSubdiv2DEdge") long edge );


public static native @Cast("CvSubdiv2DEdge") long cvSubdiv2DRotateEdge( @Cast("CvSubdiv2DEdge") long edge, int rotate );

public static native @Cast("CvSubdiv2DEdge") long cvSubdiv2DSymEdge( @Cast("CvSubdiv2DEdge") long edge );

public static native @Cast("CvSubdiv2DEdge") long cvSubdiv2DGetEdge( @Cast("CvSubdiv2DEdge") long edge, @Cast("CvNextEdgeType") int type );


public static native CvSubdiv2DPoint cvSubdiv2DEdgeOrg( @Cast("CvSubdiv2DEdge") long edge );


public static native CvSubdiv2DPoint cvSubdiv2DEdgeDst( @Cast("CvSubdiv2DEdge") long edge );

/****************************************************************************************\
*                           Additional operations on Subdivisions                        *
\****************************************************************************************/

// paints voronoi diagram: just demo function
public static native void icvDrawMosaic( CvSubdiv2D subdiv, IplImage src, IplImage dst );

// checks planar subdivision for correctness. It is not an absolute check,
// but it verifies some relations between quad-edges
public static native int icvSubdiv2DCheck( CvSubdiv2D subdiv );

// returns squared distance between two 2D points with floating-point coordinates.
public static native double icvSqDist2D32f( @ByVal CvPoint2D32f pt1, @ByVal CvPoint2D32f pt2 );
public static native double icvSqDist2D32f( @ByVal @Cast("CvPoint2D32f*") FloatBuffer pt1, @ByVal @Cast("CvPoint2D32f*") FloatBuffer pt2 );
public static native double icvSqDist2D32f( @ByVal @Cast("CvPoint2D32f*") float[] pt1, @ByVal @Cast("CvPoint2D32f*") float[] pt2 );




public static native double cvTriangleArea( @ByVal CvPoint2D32f a, @ByVal CvPoint2D32f b, @ByVal CvPoint2D32f c );
public static native double cvTriangleArea( @ByVal @Cast("CvPoint2D32f*") FloatBuffer a, @ByVal @Cast("CvPoint2D32f*") FloatBuffer b, @ByVal @Cast("CvPoint2D32f*") FloatBuffer c );
public static native double cvTriangleArea( @ByVal @Cast("CvPoint2D32f*") float[] a, @ByVal @Cast("CvPoint2D32f*") float[] b, @ByVal @Cast("CvPoint2D32f*") float[] c );


/* Constructs kd-tree from set of feature descriptors */
public static native CvFeatureTree cvCreateKDTree(CvMat desc);

/* Constructs spill-tree from set of feature descriptors */
public static native CvFeatureTree cvCreateSpillTree( @Const CvMat raw_data,
                                               int naive/*=50*/,
                                               double rho/*=.7*/,
                                               double tau/*=.1*/ );
public static native CvFeatureTree cvCreateSpillTree( @Const CvMat raw_data );

/* Release feature tree */
public static native void cvReleaseFeatureTree(CvFeatureTree tr);

/* Searches feature tree for k nearest neighbors of given reference points,
 searching (in case of kd-tree/bbf) at most emax leaves. */
public static native void cvFindFeatures(CvFeatureTree tr, @Const CvMat query_points,
                           CvMat indices, CvMat dist, int k, int emax/*=20*/);
public static native void cvFindFeatures(CvFeatureTree tr, @Const CvMat query_points,
                           CvMat indices, CvMat dist, int k);

/* Search feature tree for all points that are inlier to given rect region.
 Only implemented for kd trees */
public static native int cvFindFeaturesBoxed(CvFeatureTree tr,
                               CvMat bounds_min, CvMat bounds_max,
                               CvMat out_indices);


/* Construct a Locality Sensitive Hash (LSH) table, for indexing d-dimensional vectors of
 given type. Vectors will be hashed L times with k-dimensional p-stable (p=2) functions. */
public static native CvLSH cvCreateLSH(CvLSHOperations ops, int d,
                                 int L/*=10*/, int k/*=10*/,
                                 int type/*=CV_64FC1*/, double r/*=4*/,
                                 @Cast("int64") long seed/*=-1*/);
public static native CvLSH cvCreateLSH(CvLSHOperations ops, int d);

/* Construct in-memory LSH table, with n bins. */
public static native CvLSH cvCreateMemoryLSH(int d, int n, int L/*=10*/, int k/*=10*/,
                                       int type/*=CV_64FC1*/, double r/*=4*/,
                                       @Cast("int64") long seed/*=-1*/);
public static native CvLSH cvCreateMemoryLSH(int d, int n);

/* Free the given LSH structure. */
public static native void cvReleaseLSH(@Cast("CvLSH**") PointerPointer lsh);
public static native void cvReleaseLSH(@ByPtrPtr CvLSH lsh);

/* Return the number of vectors in the LSH. */
public static native @Cast("unsigned int") int LSHSize(CvLSH lsh);

/* Add vectors to the LSH structure, optionally returning indices. */
public static native void cvLSHAdd(CvLSH lsh, @Const CvMat data, CvMat indices/*=0*/);
public static native void cvLSHAdd(CvLSH lsh, @Const CvMat data);

/* Remove vectors from LSH, as addressed by given indices. */
public static native void cvLSHRemove(CvLSH lsh, @Const CvMat indices);

/* Query the LSH n times for at most k nearest points; data is n x d,
 indices and dist are n x k. At most emax stored points will be accessed. */
public static native void cvLSHQuery(CvLSH lsh, @Const CvMat query_points,
                       CvMat indices, CvMat dist, int k, int emax);

/* Kolmogorov-Zabin stereo-correspondence algorithm (a.k.a. KZ1) */
public static native @MemberGetter int CV_STEREO_GC_OCCLUDED();
public static final int CV_STEREO_GC_OCCLUDED = CV_STEREO_GC_OCCLUDED();

public static class CvStereoGCState extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvStereoGCState() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvStereoGCState(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvStereoGCState(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvStereoGCState position(int position) {
        return (CvStereoGCState)super.position(position);
    }

    public native int Ithreshold(); public native CvStereoGCState Ithreshold(int Ithreshold);
    public native int interactionRadius(); public native CvStereoGCState interactionRadius(int interactionRadius);
    public native float K(); public native CvStereoGCState K(float K);
    public native float lambda(); public native CvStereoGCState lambda(float lambda);
    public native float lambda1(); public native CvStereoGCState lambda1(float lambda1);
    public native float lambda2(); public native CvStereoGCState lambda2(float lambda2);
    public native int occlusionCost(); public native CvStereoGCState occlusionCost(int occlusionCost);
    public native int minDisparity(); public native CvStereoGCState minDisparity(int minDisparity);
    public native int numberOfDisparities(); public native CvStereoGCState numberOfDisparities(int numberOfDisparities);
    public native int maxIters(); public native CvStereoGCState maxIters(int maxIters);

    public native CvMat left(); public native CvStereoGCState left(CvMat left);
    public native CvMat right(); public native CvStereoGCState right(CvMat right);
    public native CvMat dispLeft(); public native CvStereoGCState dispLeft(CvMat dispLeft);
    public native CvMat dispRight(); public native CvStereoGCState dispRight(CvMat dispRight);
    public native CvMat ptrLeft(); public native CvStereoGCState ptrLeft(CvMat ptrLeft);
    public native CvMat ptrRight(); public native CvStereoGCState ptrRight(CvMat ptrRight);
    public native CvMat vtxBuf(); public native CvStereoGCState vtxBuf(CvMat vtxBuf);
    public native CvMat edgeBuf(); public native CvStereoGCState edgeBuf(CvMat edgeBuf);
}

public static native CvStereoGCState cvCreateStereoGCState( int numberOfDisparities, int maxIters );
public static native void cvReleaseStereoGCState( @Cast("CvStereoGCState**") PointerPointer state );
public static native void cvReleaseStereoGCState( @ByPtrPtr CvStereoGCState state );

public static native void cvFindStereoCorrespondenceGC( @Const CvArr left, @Const CvArr right,
                                         CvArr disparityLeft, CvArr disparityRight,
                                         CvStereoGCState state,
                                         int useDisparityGuess/*=0*/ );
public static native void cvFindStereoCorrespondenceGC( @Const CvArr left, @Const CvArr right,
                                         CvArr disparityLeft, CvArr disparityRight,
                                         CvStereoGCState state );

/* Calculates optical flow for 2 images using classical Lucas & Kanade algorithm */
public static native void cvCalcOpticalFlowLK( @Const CvArr prev, @Const CvArr curr,
                                 @ByVal CvSize win_size, CvArr velx, CvArr vely );

/* Calculates optical flow for 2 images using block matching algorithm */
public static native void cvCalcOpticalFlowBM( @Const CvArr prev, @Const CvArr curr,
                                 @ByVal CvSize block_size, @ByVal CvSize shift_size,
                                 @ByVal CvSize max_range, int use_previous,
                                 CvArr velx, CvArr vely );

/* Calculates Optical flow for 2 images using Horn & Schunck algorithm */
public static native void cvCalcOpticalFlowHS( @Const CvArr prev, @Const CvArr curr,
                                 int use_previous, CvArr velx, CvArr vely,
                                 double lambda, @ByVal CvTermCriteria criteria );


/****************************************************************************************\
*                           Background/foreground segmentation                           *
\****************************************************************************************/

/* We discriminate between foreground and background pixels
 * by building and maintaining a model of the background.
 * Any pixel which does not fit this model is then deemed
 * to be foreground.
 *
 * At present we support two core background models,
 * one of which has two variations:
 *
 *  o CV_BG_MODEL_FGD: latest and greatest algorithm, described in
 *
 *	 Foreground Object Detection from Videos Containing Complex Background.
 *	 Liyuan Li, Weimin Huang, Irene Y.H. Gu, and Qi Tian.
 *	 ACM MM2003 9p
 *
 *  o CV_BG_MODEL_FGD_SIMPLE:
 *       A code comment describes this as a simplified version of the above,
 *       but the code is in fact currently identical
 *
 *  o CV_BG_MODEL_MOG: "Mixture of Gaussians", older algorithm, described in
 *
 *       Moving target classification and tracking from real-time video.
 *       A Lipton, H Fujijoshi, R Patil
 *       Proceedings IEEE Workshop on Application of Computer Vision pp 8-14 1998
 *
 *       Learning patterns of activity using real-time tracking
 *       C Stauffer and W Grimson  August 2000
 *       IEEE Transactions on Pattern Analysis and Machine Intelligence 22(8):747-757
 */


public static final int CV_BG_MODEL_FGD =		0;
public static final int CV_BG_MODEL_MOG =		1;			/* "Mixture of Gaussians".	*/
public static final int CV_BG_MODEL_FGD_SIMPLE =	2;

@Convention("CV_CDECL") public static class CvReleaseBGStatModel extends FunctionPointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public    CvReleaseBGStatModel(Pointer p) { super(p); }
    protected CvReleaseBGStatModel() { allocate(); }
    private native void allocate();
    public native void call( @ByPtrPtr CvBGStatModel bg_model );
}
@Convention("CV_CDECL") public static class CvUpdateBGStatModel extends FunctionPointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public    CvUpdateBGStatModel(Pointer p) { super(p); }
    protected CvUpdateBGStatModel() { allocate(); }
    private native void allocate();
    public native int call( IplImage curr_frame, CvBGStatModel bg_model,
                                             double learningRate );
}

// #define CV_BG_STAT_MODEL_FIELDS()
// int             type; /*type of BG model*/
// CvReleaseBGStatModel release;
// CvUpdateBGStatModel update;
// IplImage*       background;   /*8UC3 reference background image*/
// IplImage*       foreground;   /*8UC1 foreground image*/
// IplImage**      layers;       /*8UC3 reference background image, can be null */
// int             layer_count;  /* can be zero */
// CvMemStorage*   storage;      /*storage for foreground_regions*/
// CvSeq*          foreground_regions /*foreground object contours*/

public static class CvBGStatModel extends AbstractCvBGStatModel {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvBGStatModel() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvBGStatModel(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvBGStatModel(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvBGStatModel position(int position) {
        return (CvBGStatModel)super.position(position);
    }

    public native int type(); public native CvBGStatModel type(int type); /*type of BG model*/
public native CvReleaseBGStatModel release(); public native CvBGStatModel release(CvReleaseBGStatModel release);
public native CvUpdateBGStatModel update(); public native CvBGStatModel update(CvUpdateBGStatModel update);
public native IplImage background(); public native CvBGStatModel background(IplImage background);   /*8UC3 reference background image*/
public native IplImage foreground(); public native CvBGStatModel foreground(IplImage foreground);   /*8UC1 foreground image*/
public native IplImage layers(int i); public native CvBGStatModel layers(int i, IplImage layers);
@MemberGetter public native @Cast("IplImage**") PointerPointer layers();       /*8UC3 reference background image, can be null */
public native int layer_count(); public native CvBGStatModel layer_count(int layer_count);  /* can be zero */
public native CvMemStorage storage(); public native CvBGStatModel storage(CvMemStorage storage);      /*storage for foreground_regions*/
public native CvSeq foreground_regions(); public native CvBGStatModel foreground_regions(CvSeq foreground_regions);
}

//

// Releases memory used by BGStatModel
public static native void cvReleaseBGStatModel( @Cast("CvBGStatModel**") PointerPointer bg_model );
public static native void cvReleaseBGStatModel( @ByPtrPtr CvBGStatModel bg_model );

// Updates statistical model and returns number of found foreground regions
public static native int cvUpdateBGStatModel( IplImage current_frame, CvBGStatModel bg_model,
                               double learningRate/*=-1*/);
public static native int cvUpdateBGStatModel( IplImage current_frame, CvBGStatModel bg_model);

// Performs FG post-processing using segmentation
// (all pixels of a region will be classified as foreground if majority of pixels of the region are FG).
// parameters:
//      segments - pointer to result of segmentation (for example MeanShiftSegmentation)
//      bg_model - pointer to CvBGStatModel structure
public static native void cvRefineForegroundMaskBySegm( CvSeq segments, CvBGStatModel bg_model );

/* Common use change detection function */
public static native int cvChangeDetection( IplImage prev_frame,
                              IplImage curr_frame,
                              IplImage change_mask );

/*
 Interface of ACM MM2003 algorithm
 */

/* Default parameters of foreground detection algorithm: */
public static final int CV_BGFG_FGD_LC =              128;
public static final int CV_BGFG_FGD_N1C =             15;
public static final int CV_BGFG_FGD_N2C =             25;

public static final int CV_BGFG_FGD_LCC =             64;
public static final int CV_BGFG_FGD_N1CC =            25;
public static final int CV_BGFG_FGD_N2CC =            40;

/* Background reference image update parameter: */
public static final double CV_BGFG_FGD_ALPHA_1 =         0.1f;

/* stat model update parameter
 * 0.002f ~ 1K frame(~45sec), 0.005 ~ 18sec (if 25fps and absolutely static BG)
 */
public static final double CV_BGFG_FGD_ALPHA_2 =         0.005f;

/* start value for alpha parameter (to fast initiate statistic model) */
public static final double CV_BGFG_FGD_ALPHA_3 =         0.1f;

public static final int CV_BGFG_FGD_DELTA =           2;

public static final double CV_BGFG_FGD_T =               0.9f;

public static final double CV_BGFG_FGD_MINAREA =         15.f;

public static final double CV_BGFG_FGD_BG_UPDATE_TRESH = 0.5f;

/* See the above-referenced Li/Huang/Gu/Tian paper
 * for a full description of these background-model
 * tuning parameters.
 *
 * Nomenclature:  'c'  == "color", a three-component red/green/blue vector.
 *                         We use histograms of these to model the range of
 *                         colors we've seen at a given background pixel.
 *
 *                'cc' == "color co-occurrence", a six-component vector giving
 *                         RGB color for both this frame and preceding frame.
 *                             We use histograms of these to model the range of
 *                         color CHANGES we've seen at a given background pixel.
 */
public static class CvFGDStatModelParams extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvFGDStatModelParams() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvFGDStatModelParams(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvFGDStatModelParams(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvFGDStatModelParams position(int position) {
        return (CvFGDStatModelParams)super.position(position);
    }

    public native int Lc(); public native CvFGDStatModelParams Lc(int Lc);			/* Quantized levels per 'color' component. Power of two, typically 32, 64 or 128.				*/
    public native int N1c(); public native CvFGDStatModelParams N1c(int N1c);			/* Number of color vectors used to model normal background color variation at a given pixel.			*/
    public native int N2c(); public native CvFGDStatModelParams N2c(int N2c);			/* Number of color vectors retained at given pixel.  Must be > N1c, typically ~ 5/3 of N1c.			*/
    /* Used to allow the first N1c vectors to adapt over time to changing background.				*/

    public native int Lcc(); public native CvFGDStatModelParams Lcc(int Lcc);			/* Quantized levels per 'color co-occurrence' component.  Power of two, typically 16, 32 or 64.			*/
    public native int N1cc(); public native CvFGDStatModelParams N1cc(int N1cc);		/* Number of color co-occurrence vectors used to model normal background color variation at a given pixel.	*/
    public native int N2cc(); public native CvFGDStatModelParams N2cc(int N2cc);		/* Number of color co-occurrence vectors retained at given pixel.  Must be > N1cc, typically ~ 5/3 of N1cc.	*/
    /* Used to allow the first N1cc vectors to adapt over time to changing background.				*/

    public native int is_obj_without_holes(); public native CvFGDStatModelParams is_obj_without_holes(int is_obj_without_holes);/* If TRUE we ignore holes within foreground blobs. Defaults to TRUE.						*/
    public native int perform_morphing(); public native CvFGDStatModelParams perform_morphing(int perform_morphing);	/* Number of erode-dilate-erode foreground-blob cleanup iterations.						*/
    /* These erase one-pixel junk blobs and merge almost-touching blobs. Default value is 1.			*/

    public native float alpha1(); public native CvFGDStatModelParams alpha1(float alpha1);		/* How quickly we forget old background pixel values seen.  Typically set to 0.1  				*/
    public native float alpha2(); public native CvFGDStatModelParams alpha2(float alpha2);		/* "Controls speed of feature learning". Depends on T. Typical value circa 0.005. 				*/
    public native float alpha3(); public native CvFGDStatModelParams alpha3(float alpha3);		/* Alternate to alpha2, used (e.g.) for quicker initial convergence. Typical value 0.1.				*/

    public native float delta(); public native CvFGDStatModelParams delta(float delta);		/* Affects color and color co-occurrence quantization, typically set to 2.					*/
    public native float T(); public native CvFGDStatModelParams T(float T);			/* "A percentage value which determines when new features can be recognized as new background." (Typically 0.9).*/
    public native float minArea(); public native CvFGDStatModelParams minArea(float minArea);		/* Discard foreground blobs whose bounding box is smaller than this threshold.					*/
}

public static class CvBGPixelCStatTable extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvBGPixelCStatTable() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvBGPixelCStatTable(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvBGPixelCStatTable(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvBGPixelCStatTable position(int position) {
        return (CvBGPixelCStatTable)super.position(position);
    }

    public native float Pv(); public native CvBGPixelCStatTable Pv(float Pv);
    public native float Pvb(); public native CvBGPixelCStatTable Pvb(float Pvb);
    public native @Cast("uchar") byte v(int i); public native CvBGPixelCStatTable v(int i, byte v);
    @MemberGetter public native @Cast("uchar*") BytePointer v();
}

public static class CvBGPixelCCStatTable extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvBGPixelCCStatTable() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvBGPixelCCStatTable(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvBGPixelCCStatTable(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvBGPixelCCStatTable position(int position) {
        return (CvBGPixelCCStatTable)super.position(position);
    }

    public native float Pv(); public native CvBGPixelCCStatTable Pv(float Pv);
    public native float Pvb(); public native CvBGPixelCCStatTable Pvb(float Pvb);
    public native @Cast("uchar") byte v(int i); public native CvBGPixelCCStatTable v(int i, byte v);
    @MemberGetter public native @Cast("uchar*") BytePointer v();
}

public static class CvBGPixelStat extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvBGPixelStat() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvBGPixelStat(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvBGPixelStat(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvBGPixelStat position(int position) {
        return (CvBGPixelStat)super.position(position);
    }

    public native float Pbc(); public native CvBGPixelStat Pbc(float Pbc);
    public native float Pbcc(); public native CvBGPixelStat Pbcc(float Pbcc);
    public native CvBGPixelCStatTable ctable(); public native CvBGPixelStat ctable(CvBGPixelCStatTable ctable);
    public native CvBGPixelCCStatTable cctable(); public native CvBGPixelStat cctable(CvBGPixelCCStatTable cctable);
    public native @Cast("uchar") byte is_trained_st_model(); public native CvBGPixelStat is_trained_st_model(byte is_trained_st_model);
    public native @Cast("uchar") byte is_trained_dyn_model(); public native CvBGPixelStat is_trained_dyn_model(byte is_trained_dyn_model);
}


public static class CvFGDStatModel extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvFGDStatModel() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvFGDStatModel(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvFGDStatModel(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvFGDStatModel position(int position) {
        return (CvFGDStatModel)super.position(position);
    }

    public native int type(); public native CvFGDStatModel type(int type); /*type of BG model*/
public native CvReleaseBGStatModel release(); public native CvFGDStatModel release(CvReleaseBGStatModel release);
public native CvUpdateBGStatModel update(); public native CvFGDStatModel update(CvUpdateBGStatModel update);
public native IplImage background(); public native CvFGDStatModel background(IplImage background);   /*8UC3 reference background image*/
public native IplImage foreground(); public native CvFGDStatModel foreground(IplImage foreground);   /*8UC1 foreground image*/
public native IplImage layers(int i); public native CvFGDStatModel layers(int i, IplImage layers);
@MemberGetter public native @Cast("IplImage**") PointerPointer layers();       /*8UC3 reference background image, can be null */
public native int layer_count(); public native CvFGDStatModel layer_count(int layer_count);  /* can be zero */
public native CvMemStorage storage(); public native CvFGDStatModel storage(CvMemStorage storage);      /*storage for foreground_regions*/
public native CvSeq foreground_regions(); public native CvFGDStatModel foreground_regions(CvSeq foreground_regions);
    public native CvBGPixelStat pixel_stat(); public native CvFGDStatModel pixel_stat(CvBGPixelStat pixel_stat);
    public native IplImage Ftd(); public native CvFGDStatModel Ftd(IplImage Ftd);
    public native IplImage Fbd(); public native CvFGDStatModel Fbd(IplImage Fbd);
    public native IplImage prev_frame(); public native CvFGDStatModel prev_frame(IplImage prev_frame);
    public native @ByRef CvFGDStatModelParams params(); public native CvFGDStatModel params(CvFGDStatModelParams params);
}

/* Creates FGD model */
public static native CvBGStatModel cvCreateFGDStatModel( IplImage first_frame,
                                           CvFGDStatModelParams parameters/*=NULL*/);
public static native CvBGStatModel cvCreateFGDStatModel( IplImage first_frame);

/*
 Interface of Gaussian mixture algorithm

 "An improved adaptive background mixture model for real-time tracking with shadow detection"
 P. KadewTraKuPong and R. Bowden,
 Proc. 2nd European Workshp on Advanced Video-Based Surveillance Systems, 2001."
 http://personal.ee.surrey.ac.uk/Personal/R.Bowden/publications/avbs01/avbs01.pdf
 */

/* Note:  "MOG" == "Mixture Of Gaussians": */

public static final int CV_BGFG_MOG_MAX_NGAUSSIANS = 500;

/* default parameters of gaussian background detection algorithm */
public static final double CV_BGFG_MOG_BACKGROUND_THRESHOLD =     0.7;     /* threshold sum of weights for background test */
public static final double CV_BGFG_MOG_STD_THRESHOLD =            2.5;     /* lambda=2.5 is 99% */
public static final int CV_BGFG_MOG_WINDOW_SIZE =              200;     /* Learning rate; alpha = 1/CV_GBG_WINDOW_SIZE */
public static final int CV_BGFG_MOG_NGAUSSIANS =               5;       /* = K = number of Gaussians in mixture */
public static final double CV_BGFG_MOG_WEIGHT_INIT =              0.05;
public static final int CV_BGFG_MOG_SIGMA_INIT =               30;
public static final double CV_BGFG_MOG_MINAREA =                  15.f;


public static final int CV_BGFG_MOG_NCOLORS =                  3;

public static class CvGaussBGStatModelParams extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvGaussBGStatModelParams() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvGaussBGStatModelParams(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvGaussBGStatModelParams(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvGaussBGStatModelParams position(int position) {
        return (CvGaussBGStatModelParams)super.position(position);
    }

    public native int win_size(); public native CvGaussBGStatModelParams win_size(int win_size);               /* = 1/alpha */
    public native int n_gauss(); public native CvGaussBGStatModelParams n_gauss(int n_gauss);
    public native double bg_threshold(); public native CvGaussBGStatModelParams bg_threshold(double bg_threshold);
    public native double std_threshold(); public native CvGaussBGStatModelParams std_threshold(double std_threshold);
    public native double minArea(); public native CvGaussBGStatModelParams minArea(double minArea);
    public native double weight_init(); public native CvGaussBGStatModelParams weight_init(double weight_init);
    public native double variance_init(); public native CvGaussBGStatModelParams variance_init(double variance_init);
}

public static class CvGaussBGValues extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvGaussBGValues() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvGaussBGValues(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvGaussBGValues(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvGaussBGValues position(int position) {
        return (CvGaussBGValues)super.position(position);
    }

    public native int match_sum(); public native CvGaussBGValues match_sum(int match_sum);
    public native double weight(); public native CvGaussBGValues weight(double weight);
    public native double variance(int i); public native CvGaussBGValues variance(int i, double variance);
    @MemberGetter public native DoublePointer variance();
    public native double mean(int i); public native CvGaussBGValues mean(int i, double mean);
    @MemberGetter public native DoublePointer mean();
}

public static class CvGaussBGPoint extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvGaussBGPoint() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvGaussBGPoint(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvGaussBGPoint(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvGaussBGPoint position(int position) {
        return (CvGaussBGPoint)super.position(position);
    }

    public native CvGaussBGValues g_values(); public native CvGaussBGPoint g_values(CvGaussBGValues g_values);
}


public static class CvGaussBGModel extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvGaussBGModel() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvGaussBGModel(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvGaussBGModel(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvGaussBGModel position(int position) {
        return (CvGaussBGModel)super.position(position);
    }

    public native int type(); public native CvGaussBGModel type(int type); /*type of BG model*/
public native CvReleaseBGStatModel release(); public native CvGaussBGModel release(CvReleaseBGStatModel release);
public native CvUpdateBGStatModel update(); public native CvGaussBGModel update(CvUpdateBGStatModel update);
public native IplImage background(); public native CvGaussBGModel background(IplImage background);   /*8UC3 reference background image*/
public native IplImage foreground(); public native CvGaussBGModel foreground(IplImage foreground);   /*8UC1 foreground image*/
public native IplImage layers(int i); public native CvGaussBGModel layers(int i, IplImage layers);
@MemberGetter public native @Cast("IplImage**") PointerPointer layers();       /*8UC3 reference background image, can be null */
public native int layer_count(); public native CvGaussBGModel layer_count(int layer_count);  /* can be zero */
public native CvMemStorage storage(); public native CvGaussBGModel storage(CvMemStorage storage);      /*storage for foreground_regions*/
public native CvSeq foreground_regions(); public native CvGaussBGModel foreground_regions(CvSeq foreground_regions);
    public native @ByRef CvGaussBGStatModelParams params(); public native CvGaussBGModel params(CvGaussBGStatModelParams params);
    public native CvGaussBGPoint g_point(); public native CvGaussBGModel g_point(CvGaussBGPoint g_point);
    public native int countFrames(); public native CvGaussBGModel countFrames(int countFrames);
    public native Pointer mog(); public native CvGaussBGModel mog(Pointer mog);
}


/* Creates Gaussian mixture background model */
public static native CvBGStatModel cvCreateGaussianBGModel( IplImage first_frame,
                                              CvGaussBGStatModelParams parameters/*=NULL*/);
public static native CvBGStatModel cvCreateGaussianBGModel( IplImage first_frame);


public static class CvBGCodeBookElem extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvBGCodeBookElem() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvBGCodeBookElem(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvBGCodeBookElem(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvBGCodeBookElem position(int position) {
        return (CvBGCodeBookElem)super.position(position);
    }

    public native CvBGCodeBookElem next(); public native CvBGCodeBookElem next(CvBGCodeBookElem next);
    public native int tLastUpdate(); public native CvBGCodeBookElem tLastUpdate(int tLastUpdate);
    public native int stale(); public native CvBGCodeBookElem stale(int stale);
    public native @Cast("uchar") byte boxMin(int i); public native CvBGCodeBookElem boxMin(int i, byte boxMin);
    @MemberGetter public native @Cast("uchar*") BytePointer boxMin();
    public native @Cast("uchar") byte boxMax(int i); public native CvBGCodeBookElem boxMax(int i, byte boxMax);
    @MemberGetter public native @Cast("uchar*") BytePointer boxMax();
    public native @Cast("uchar") byte learnMin(int i); public native CvBGCodeBookElem learnMin(int i, byte learnMin);
    @MemberGetter public native @Cast("uchar*") BytePointer learnMin();
    public native @Cast("uchar") byte learnMax(int i); public native CvBGCodeBookElem learnMax(int i, byte learnMax);
    @MemberGetter public native @Cast("uchar*") BytePointer learnMax();
}

public static class CvBGCodeBookModel extends AbstractCvBGCodeBookModel {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvBGCodeBookModel() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CvBGCodeBookModel(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvBGCodeBookModel(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvBGCodeBookModel position(int position) {
        return (CvBGCodeBookModel)super.position(position);
    }

    public native @ByRef CvSize size(); public native CvBGCodeBookModel size(CvSize size);
    public native int t(); public native CvBGCodeBookModel t(int t);
    public native @Cast("uchar") byte cbBounds(int i); public native CvBGCodeBookModel cbBounds(int i, byte cbBounds);
    @MemberGetter public native @Cast("uchar*") BytePointer cbBounds();
    public native @Cast("uchar") byte modMin(int i); public native CvBGCodeBookModel modMin(int i, byte modMin);
    @MemberGetter public native @Cast("uchar*") BytePointer modMin();
    public native @Cast("uchar") byte modMax(int i); public native CvBGCodeBookModel modMax(int i, byte modMax);
    @MemberGetter public native @Cast("uchar*") BytePointer modMax();
    public native CvBGCodeBookElem cbmap(int i); public native CvBGCodeBookModel cbmap(int i, CvBGCodeBookElem cbmap);
    @MemberGetter public native @Cast("CvBGCodeBookElem**") PointerPointer cbmap();
    public native CvMemStorage storage(); public native CvBGCodeBookModel storage(CvMemStorage storage);
    public native CvBGCodeBookElem freeList(); public native CvBGCodeBookModel freeList(CvBGCodeBookElem freeList);
}

public static native CvBGCodeBookModel cvCreateBGCodeBookModel( );
public static native void cvReleaseBGCodeBookModel( @Cast("CvBGCodeBookModel**") PointerPointer model );
public static native void cvReleaseBGCodeBookModel( @ByPtrPtr CvBGCodeBookModel model );

public static native void cvBGCodeBookUpdate( CvBGCodeBookModel model, @Const CvArr image,
                               @ByVal(nullValue = "cvRect(0,0,0,0)") CvRect roi/*=cvRect(0,0,0,0)*/,
                               @Const CvArr mask/*=0*/ );
public static native void cvBGCodeBookUpdate( CvBGCodeBookModel model, @Const CvArr image );

public static native int cvBGCodeBookDiff( @Const CvBGCodeBookModel model, @Const CvArr image,
                            CvArr fgmask, @ByVal(nullValue = "cvRect(0,0,0,0)") CvRect roi/*=cvRect(0,0,0,0)*/ );
public static native int cvBGCodeBookDiff( @Const CvBGCodeBookModel model, @Const CvArr image,
                            CvArr fgmask );

public static native void cvBGCodeBookClearStale( CvBGCodeBookModel model, int staleThresh,
                                   @ByVal(nullValue = "cvRect(0,0,0,0)") CvRect roi/*=cvRect(0,0,0,0)*/,
                                   @Const CvArr mask/*=0*/ );
public static native void cvBGCodeBookClearStale( CvBGCodeBookModel model, int staleThresh );

public static native CvSeq cvSegmentFGMask( CvArr fgmask, int poly1Hull0/*=1*/,
                              float perimScale/*=4.f*/,
                              CvMemStorage storage/*=0*/,
                              @ByVal(nullValue = "cvPoint(0,0)") CvPoint offset/*=cvPoint(0,0)*/);
public static native CvSeq cvSegmentFGMask( CvArr fgmask);
public static native CvSeq cvSegmentFGMask( CvArr fgmask, int poly1Hull0/*=1*/,
                              float perimScale/*=4.f*/,
                              CvMemStorage storage/*=0*/,
                              @ByVal(nullValue = "cvPoint(0,0)") @Cast("CvPoint*") IntBuffer offset/*=cvPoint(0,0)*/);
public static native CvSeq cvSegmentFGMask( CvArr fgmask, int poly1Hull0/*=1*/,
                              float perimScale/*=4.f*/,
                              CvMemStorage storage/*=0*/,
                              @ByVal(nullValue = "cvPoint(0,0)") @Cast("CvPoint*") int[] offset/*=cvPoint(0,0)*/);

// #ifdef __cplusplus
// #endif

// #endif

/* End of file. */


}
