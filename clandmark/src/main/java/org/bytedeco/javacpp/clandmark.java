// Targeted by JavaCPP version 1.2-SNAPSHOT

package org.bytedeco.javacpp;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;

public class clandmark extends org.bytedeco.javacpp.presets.clandmark {
    static { Loader.load(); }

// Parsed from CLandmark.h

/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013, 2014, 2015 Michal Uricar
 * Copyright (C) 2013, 2014, 2015 Michal Uricar
 */

// #ifndef _CLANDMARK__H__
// #define _CLANDMARK__H__

// #include "CAppearanceModel.h"
// #include "CDeformationCost.h"
// #include "CMaxSumSolver.h"
// #include "CFeaturePool.h"

public static final int cimg_verbosity = 1;		// we don't need window output capabilities of CImg
public static final int cimg_display = 0;			// we don't need window output capabilities of CImg

// #include "CImg.h"
// #include "CTypes.h"

// #include <vector>

// IO functions
// #include <iostream>
// #include <iomanip>

/**
 *
 */
/** enum clandmark::EGraphType */
public static final int
	TREE= 1,
	SIMPLE_NET= 2,
	GENERAL_GRAPH= 3;

/**
 *
 */
@Namespace("clandmark") public static class Timings extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public Timings() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public Timings(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Timings(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public Timings position(long position) {
        return (Timings)super.position(position);
    }

	/** */
	public native @Cast("clandmark::fl_double_t") float overall(); public native Timings overall(float overall);
	/** */
	public native @Cast("clandmark::fl_double_t") float normalizedFrame(); public native Timings normalizedFrame(float normalizedFrame);
	/** */
	public native @Cast("clandmark::fl_double_t") float features(); public native Timings features(float features);
	/** */
	public native @Cast("clandmark::fl_double_t") float maxsum(); public native Timings maxsum(float maxsum);
}


/**
 * \brief The CLandmark class
 */
@Namespace("clandmark") @NoOffset public static class CLandmark extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CLandmark(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public CLandmark(long size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(long size);
    @Override public CLandmark position(long position) {
        return (CLandmark)super.position(position);
    }


	/**
	 * \brief CLandmark
	 * @param landmarksCount
	 * @param edgesCount
	 * @param base_window_width
	 * @param base_window_height
	 * @param base_window_margin_x
	 * @param base_window_margin_y
	 */
	public CLandmark(
			int landmarksCount,
			int edgesCount,
			int base_window_width,
			int base_window_height,
			@Cast("clandmark::fl_double_t") float base_window_margin_x,
			@Cast("clandmark::fl_double_t") float base_window_margin_y
		) { super((Pointer)null); allocate(landmarksCount, edgesCount, base_window_width, base_window_height, base_window_margin_x, base_window_margin_y); }
	private native void allocate(
			int landmarksCount,
			int edgesCount,
			int base_window_width,
			int base_window_height,
			@Cast("clandmark::fl_double_t") float base_window_margin_x,
			@Cast("clandmark::fl_double_t") float base_window_margin_y
		);

	/**
	 * \brief Default CLandmark
	 */
	public CLandmark() { super((Pointer)null); allocate(); }
	private native void allocate();

	/**
	 * \brief init
	 * @param landmarksCount
	 * @param edgesCount
	 * @param base_window_width
	 * @param base_window_height
	 * @param base_window_margin_x
	 * @param base_window_margin_y
	 */
	public native void init(int landmarksCount,
				  int edgesCount,
				  int base_window_width,
				  int base_window_height,
				  @Cast("clandmark::fl_double_t") float base_window_margin_x,
				  @Cast("clandmark::fl_double_t") float base_window_margin_y);

	/**
	 * \brief init_optimized;
	 * @param landmarksCount
	 * @param edgesCount
	 * @param base_window_width
	 * @param base_window_height
	 * @param base_window_margin_x
	 * @param base_window_margin_y
	 */
	public native void init_optimized(int landmarksCount,
							int edgesCount,
							int base_window_width,
							int base_window_height,
							@Cast("clandmark::fl_double_t") float base_window_margin_x,
							@Cast("clandmark::fl_double_t") float base_window_margin_y);

	/**
	 * \brief ~CLandmark destructor
	 */

	/**
	 * \brief Function detects landmarks within given bounding box in a given image.
	 * @param inputImage	Input image
	 * @param boundingBox	Bounding box (format: [min_x, min_y, max_x, max_y]) of object of interest (i.e. axis aligned)
	 * @param ground_truth
	 */
	public native void detect(@Cast("cimg_library::CImg<unsigned char>*") BytePointer inputImage, IntPointer boundingBox, @Cast("clandmark::fl_double_t*const") FloatPointer ground_truth/*=0*/);
	public native void detect(@Cast("cimg_library::CImg<unsigned char>*") BytePointer inputImage, IntPointer boundingBox);
	public native void detect(@Cast("cimg_library::CImg<unsigned char>*") BytePointer inputImage, IntBuffer boundingBox, @Cast("clandmark::fl_double_t*const") FloatPointer ground_truth/*=0*/);
	public native void detect(@Cast("cimg_library::CImg<unsigned char>*") BytePointer inputImage, IntBuffer boundingBox);
	public native void detect(@Cast("cimg_library::CImg<unsigned char>*") BytePointer inputImage, int[] boundingBox, @Cast("clandmark::fl_double_t*const") FloatPointer ground_truth/*=0*/);
	public native void detect(@Cast("cimg_library::CImg<unsigned char>*") BytePointer inputImage, int[] boundingBox);

	/**
	 * \brief detect_optimized
	 * @param inputImage
	 * @param boundingBox
	 * @param ground_truth
	 */
	public native void detect_optimized(@Cast("cimg_library::CImg<unsigned char>*") BytePointer inputImage, IntPointer boundingBox, @Cast("clandmark::fl_double_t*const") FloatPointer ground_truth/*=0*/);
	public native void detect_optimized(@Cast("cimg_library::CImg<unsigned char>*") BytePointer inputImage, IntPointer boundingBox);
	public native void detect_optimized(@Cast("cimg_library::CImg<unsigned char>*") BytePointer inputImage, IntBuffer boundingBox, @Cast("clandmark::fl_double_t*const") FloatPointer ground_truth/*=0*/);
	public native void detect_optimized(@Cast("cimg_library::CImg<unsigned char>*") BytePointer inputImage, IntBuffer boundingBox);
	public native void detect_optimized(@Cast("cimg_library::CImg<unsigned char>*") BytePointer inputImage, int[] boundingBox, @Cast("clandmark::fl_double_t*const") FloatPointer ground_truth/*=0*/);
	public native void detect_optimized(@Cast("cimg_library::CImg<unsigned char>*") BytePointer inputImage, int[] boundingBox);

	/**
	 * \brief detect_optimizedFromPool
	 * @param ground_truth
	 */
	public native void detect_optimizedFromPool(IntPointer boundingBox, @Cast("clandmark::fl_double_t*const") FloatPointer ground_truth/*=0*/);
	public native void detect_optimizedFromPool(IntPointer boundingBox);
	public native void detect_optimizedFromPool(IntBuffer boundingBox, @Cast("clandmark::fl_double_t*const") FloatPointer ground_truth/*=0*/);
	public native void detect_optimizedFromPool(IntBuffer boundingBox);
	public native void detect_optimizedFromPool(int[] boundingBox, @Cast("clandmark::fl_double_t*const") FloatPointer ground_truth/*=0*/);
	public native void detect_optimizedFromPool(int[] boundingBox);

	/**
	 * \brief detect_mirrored
	 * @param inputImage
	 * @param boundingBox
	 * @param ground_truth
	 */
	public native void detect_mirrored(@Cast("cimg_library::CImg<unsigned char>*") BytePointer inputImage, IntPointer boundingBox, @Cast("clandmark::fl_double_t*const") FloatPointer ground_truth/*=0*/);
	public native void detect_mirrored(@Cast("cimg_library::CImg<unsigned char>*") BytePointer inputImage, IntPointer boundingBox);
	public native void detect_mirrored(@Cast("cimg_library::CImg<unsigned char>*") BytePointer inputImage, IntBuffer boundingBox, @Cast("clandmark::fl_double_t*const") FloatPointer ground_truth/*=0*/);
	public native void detect_mirrored(@Cast("cimg_library::CImg<unsigned char>*") BytePointer inputImage, IntBuffer boundingBox);
	public native void detect_mirrored(@Cast("cimg_library::CImg<unsigned char>*") BytePointer inputImage, int[] boundingBox, @Cast("clandmark::fl_double_t*const") FloatPointer ground_truth/*=0*/);
	public native void detect_mirrored(@Cast("cimg_library::CImg<unsigned char>*") BytePointer inputImage, int[] boundingBox);

	/**
	 * \brief detect
	 * @param inputImage	normalized image frame
	 * @param ground_truth
	 */
	public native void detect_base(@Cast("cimg_library::CImg<unsigned char>*") BytePointer inputImage, IntPointer ground_truth/*=0*/);
	public native void detect_base(@Cast("cimg_library::CImg<unsigned char>*") BytePointer inputImage);
	public native void detect_base(@Cast("cimg_library::CImg<unsigned char>*") BytePointer inputImage, IntBuffer ground_truth/*=0*/);
	public native void detect_base(@Cast("cimg_library::CImg<unsigned char>*") BytePointer inputImage, int[] ground_truth/*=0*/);

	/**
	 * \brief detect_base_optimized
	 * @param nf_features_mipmap
	 * @param ground_truth
	 */
	public native void detect_base_optimized(IntPointer ground_truth/*=0*/);
	public native void detect_base_optimized();
	public native void detect_base_optimized(IntBuffer ground_truth/*=0*/);
	public native void detect_base_optimized(int[] ground_truth/*=0*/);

	/**
	 * \brief detect_base_optimized
	 * @param inputImage
	 * @param ground_truth
	 */
	public native void detect_base_optimized(@Cast("cimg_library::CImg<unsigned char>*") BytePointer inputImage, IntPointer ground_truth/*=0*/);
	public native void detect_base_optimized(@Cast("cimg_library::CImg<unsigned char>*") BytePointer inputImage);
	public native void detect_base_optimized(@Cast("cimg_library::CImg<unsigned char>*") BytePointer inputImage, IntBuffer ground_truth/*=0*/);
	public native void detect_base_optimized(@Cast("cimg_library::CImg<unsigned char>*") BytePointer inputImage, int[] ground_truth/*=0*/);

	/**
	 * \brief nodemax_base
	 * @param inputImage
	 * @param ground_truth
	 */
	public native void nodemax_base(@Cast("cimg_library::CImg<unsigned char>*") BytePointer inputImage, IntPointer ground_truth/*=0*/);
	public native void nodemax_base(@Cast("cimg_library::CImg<unsigned char>*") BytePointer inputImage);
	public native void nodemax_base(@Cast("cimg_library::CImg<unsigned char>*") BytePointer inputImage, IntBuffer ground_truth/*=0*/);
	public native void nodemax_base(@Cast("cimg_library::CImg<unsigned char>*") BytePointer inputImage, int[] ground_truth/*=0*/);

	/**
	 * \brief getFeatures
	 * @param inputImage
	 * @param boundingBox
	 * @return
	 */
	public native @Cast("clandmark::fl_double_t*") FloatPointer getFeatures(@Cast("cimg_library::CImg<unsigned char>*") BytePointer inputImage, IntPointer boundingBox, IntPointer configuration);
	public native @Cast("clandmark::fl_double_t*") FloatPointer getFeatures(@Cast("cimg_library::CImg<unsigned char>*") BytePointer inputImage, IntBuffer boundingBox, IntBuffer configuration);
	public native @Cast("clandmark::fl_double_t*") FloatPointer getFeatures(@Cast("cimg_library::CImg<unsigned char>*") BytePointer inputImage, int[] boundingBox, int[] configuration);

	/**
	 * \brief getFeatures_base
	 * @param nf
	 * @param configuration
	 * @return
	 */
	public native @Cast("clandmark::fl_double_t*") FloatPointer getFeatures_base(@Cast("cimg_library::CImg<unsigned char>*") BytePointer nf, IntPointer configuration);
	public native @Cast("clandmark::fl_double_t*") FloatPointer getFeatures_base(@Cast("cimg_library::CImg<unsigned char>*") BytePointer nf, IntBuffer configuration);
	public native @Cast("clandmark::fl_double_t*") FloatPointer getFeatures_base(@Cast("cimg_library::CImg<unsigned char>*") BytePointer nf, int[] configuration);

	/**
	 * \brief getFeatures_base_optimized
	 * @param configuration
	 * @return
	 */
	public native @Cast("clandmark::fl_double_t*") FloatPointer getFeatures_base_optimized(IntPointer configuration);
	public native @Cast("clandmark::fl_double_t*") FloatPointer getFeatures_base_optimized(IntBuffer configuration);
	public native @Cast("clandmark::fl_double_t*") FloatPointer getFeatures_base_optimized(int[] configuration);

	/**
	 * \brief getFeatures
	 * @param configuration
	 * @return
	 */
	public native @Cast("clandmark::fl_double_t*") FloatPointer getFeatures(IntPointer configuration);
	public native @Cast("clandmark::fl_double_t*") FloatPointer getFeatures(IntBuffer configuration);
	public native @Cast("clandmark::fl_double_t*") FloatPointer getFeatures(int[] configuration);

	/**
	 * \brief getPsiNodes_base
	 * @param nf
	 * @param configuration
	 * @return
	 */
	public native @Cast("clandmark::fl_double_t*") FloatPointer getPsiNodes_base(@Cast("cimg_library::CImg<unsigned char>*") BytePointer nf, IntPointer configuration);
	public native @Cast("clandmark::fl_double_t*") FloatPointer getPsiNodes_base(@Cast("cimg_library::CImg<unsigned char>*") BytePointer nf, IntBuffer configuration);
	public native @Cast("clandmark::fl_double_t*") FloatPointer getPsiNodes_base(@Cast("cimg_library::CImg<unsigned char>*") BytePointer nf, int[] configuration);

	/**
	 * \brief getPsiNodes
	 * @param configuration
	 * @return
	 */
	public native @Cast("clandmark::fl_double_t*") FloatPointer getPsiNodes(IntPointer configuration);
	public native @Cast("clandmark::fl_double_t*") FloatPointer getPsiNodes(IntBuffer configuration);
	public native @Cast("clandmark::fl_double_t*") FloatPointer getPsiNodes(int[] configuration);

	/**
	 * \brief setNormalizationFactor
	 * @param factor
	 */
	public native void setNormalizationFactor(@Cast("clandmark::fl_double_t") float factor);

	/**
	 * \brief getName
	 * @return
	 */
	public native @StdString BytePointer getName();

	/**
	 * \brief setName
	 * @param name_
	 */
	public native void setName(@StdString BytePointer name_);
	public native void setName(@StdString String name_);

	/**
	 * \brief getLandmarks
	 * @return
	 */
	public native @Cast("clandmark::fl_double_t*") FloatPointer getLandmarks();

	/**
	 * \brief getLandmarksNF
	 * @return
	 */
	public native IntPointer getLandmarksNF();

	/**
	 * \brief getLandmarksCount
	 * @return
	 */
	public native int getLandmarksCount();

	/**
	 * \brief getEdgesCount
	 * @return
	 */
	public native int getEdgesCount();

	/**
	 * \brief computeWdimension
	 */
	public native void computeWdimension();

	/**
	 * \brief getWdimension
	 * @return
	 */
	public native int getWdimension();

	/**
	 * \brief getPsiNodesDimension
	 * @return
	 */
	public native int getPsiNodesDimension();

	/**
	 * \brief getPsiEdgesDimension
	 * @return
	 */
	public native int getPsiEdgesDimension();

	/**
	 * \brief getNodesDimensions
	 * @return
	 */
	public native IntPointer getNodesDimensions();

	/**
	 * \brief getEdgesDimensions
	 * @return
	 */
	public native IntPointer getEdgesDimensions();

	/**
	 * \brief setW
	 * @param input_w
	 */
	public native void setW(@Cast("clandmark::fl_double_t*const") FloatPointer input_w);

	/**
	 * \brief getW
	 * @return joint weight vector w, allocates memory, does not care about its freeing!
	 */
	public native @Cast("clandmark::fl_double_t*") FloatPointer getW();

	/**
	 * \brief setNodesW
	 * @param input_w
	 */
	public native void setNodesW(@Cast("clandmark::fl_double_t*const") FloatPointer input_w);

	/**
	 * \brief getQvalues
	 * @return
	 */
	public native @Cast("clandmark::fl_double_t*") FloatPointer getQvalues();

	/**
	 * \brief getGvalues
	 * @return
	 */
	public native @Cast("clandmark::fl_double_t*") FloatPointer getGvalues();

	/**
	 * \brief getLossValues
	 * @return
	 */
	public native @Cast("clandmark::fl_double_t*") FloatPointer getLossValues(IntPointer position);
	public native @Cast("clandmark::fl_double_t*") FloatPointer getLossValues(IntBuffer position);
	public native @Cast("clandmark::fl_double_t*") FloatPointer getLossValues(int[] position);

	/**
	 * \brief write
	 * @param filename
	 */
	public native void write(@Cast("const char*") BytePointer filename, @Cast("bool") boolean writeW/*=true*/);
	public native void write(@Cast("const char*") BytePointer filename);
	public native void write(String filename, @Cast("bool") boolean writeW/*=true*/);
	public native void write(String filename);

	/**
	 * \brief getEdges
	 * @return
	 */
	public native IntPointer getEdges();

	/**
	* \brief getNodesSearchSpaces
	* @return
	*/
	public native IntPointer getNodesSearchSpaces();

	/**
	 * \brief getWindowSizes
	 * @return
	 */
	public native IntPointer getWindowSizes();

	/**
	 * \brief getH
	 * @return
	 */
	public native @Cast("clandmark::fl_double_t*") FloatPointer getH();

	/**
	 * \brief getHinv
	 * @return
	 */
	public native @Cast("clandmark::fl_double_t*") FloatPointer getHinv();

	/**
	 * \brief nodeHasLoss
	 * @param nodeID
	 * @return
	 */
	public native @Cast("bool") boolean nodeHasLoss(int nodeID);

	/** NEW FEATURE (NOT IMPLEMENTED YET) */
	public native void changeSearchSpace();
	/** -----------
	 <p>
	 *  ========================= SPEED UP ================================
	<p>
	/**
	 * \brief setNFfeaturesPool
	 * @param pool
	 */
	public native void setNFfeaturesPool(CFeaturePool pool);

	/** ========================= ========== ================================
	<p>
	/**
	 * \brief getIntermediateResults
	 * @param output
	 * @return
	 */
	public native @Cast("clandmark::fl_double_t**") @StdVector PointerPointer getIntermediateResults();

	/** */
	public native @ByRef Timings timings(); public native CLandmark timings(Timings timings);
}

 /* namespace clandmark */

// #endif /* _CLANDMARK__H__ */


// Parsed from Flandmark.h

/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013, 2014, 2015 Michal Uricar
 * Copyright (C) 2013, 2014, 2015 Michal Uricar
 */

// #ifndef _FLANDMARK_H__
// #define _FLANDMARK_H__

// #include "CLandmark.h"

/**
 * \brief The Flandmark class
 */
@Namespace("clandmark") @NoOffset public static class Flandmark extends CLandmark {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Flandmark(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public Flandmark(long size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(long size);
    @Override public Flandmark position(long position) {
        return (Flandmark)super.position(position);
    }


	/**
	 * \brief Flandmark
	 * @param landmarksCount
	 * @param edgesCount
	 * @param base_window_width
	 * @param base_window_height
	 * @param base_window_margin_x
	 * @param base_window_margin_y
	 */
	public Flandmark(
			int landmarksCount/*=8*/,
			int edgesCount/*=7*/,
			int base_window_width/*=40*/,
			int base_window_height/*=40*/,
			@Cast("clandmark::fl_double_t") float base_window_margin_x/*=1.2*/,
			@Cast("clandmark::fl_double_t") float base_window_margin_y/*=1.2*/
		) { super((Pointer)null); allocate(landmarksCount, edgesCount, base_window_width, base_window_height, base_window_margin_x, base_window_margin_y); }
	private native void allocate(
			int landmarksCount/*=8*/,
			int edgesCount/*=7*/,
			int base_window_width/*=40*/,
			int base_window_height/*=40*/,
			@Cast("clandmark::fl_double_t") float base_window_margin_x/*=1.2*/,
			@Cast("clandmark::fl_double_t") float base_window_margin_y/*=1.2*/
		);
	public Flandmark(
		) { super((Pointer)null); allocate(); }
	private native void allocate(
		);

	/**
	 * \brief Flandmark
	 * @param filename
	 */
	public Flandmark(@Cast("const char*") BytePointer filename, @Cast("bool") boolean train/*=false*/) { super((Pointer)null); allocate(filename, train); }
	private native void allocate(@Cast("const char*") BytePointer filename, @Cast("bool") boolean train/*=false*/);
	public Flandmark(@Cast("const char*") BytePointer filename) { super((Pointer)null); allocate(filename); }
	private native void allocate(@Cast("const char*") BytePointer filename);
	public Flandmark(String filename, @Cast("bool") boolean train/*=false*/) { super((Pointer)null); allocate(filename, train); }
	private native void allocate(String filename, @Cast("bool") boolean train/*=false*/);
	public Flandmark(String filename) { super((Pointer)null); allocate(filename); }
	private native void allocate(String filename);

	/**
	 * \brief getInstanceOf
	 * @param filename
	 * @param train
	 * @return
	 */
	//TODO - this should be protected!!!
	public static native Flandmark getInstanceOf(@Cast("const char*") BytePointer filename, @Cast("bool") boolean train/*=false*/);
	public static native Flandmark getInstanceOf(@Cast("const char*") BytePointer filename);
	public static native Flandmark getInstanceOf(String filename, @Cast("bool") boolean train/*=false*/);
	public static native Flandmark getInstanceOf(String filename);

	/** Destructor */

	/**
	 * \brief getNF
	 * @return
	 */
	public native @Cast("cimg_library::CImg<unsigned char>*") BytePointer getNF();

	/**
	 * \brief getNF
	 * @param img
	 * @param bbox
	 * @return
	 */
	public native @Cast("cimg_library::CImg<unsigned char>*") BytePointer getNF(@Cast("cimg_library::CImg<unsigned char>*") BytePointer img, IntPointer bbox, @Cast("clandmark::fl_double_t*const") FloatPointer ground_truth/*=0*/);
	public native @Cast("cimg_library::CImg<unsigned char>*") BytePointer getNF(@Cast("cimg_library::CImg<unsigned char>*") BytePointer img, IntPointer bbox);
	public native @Cast("cimg_library::CImg<unsigned char>*") BytePointer getNF(@Cast("cimg_library::CImg<unsigned char>*") BytePointer img, IntBuffer bbox, @Cast("clandmark::fl_double_t*const") FloatPointer ground_truth/*=0*/);
	public native @Cast("cimg_library::CImg<unsigned char>*") BytePointer getNF(@Cast("cimg_library::CImg<unsigned char>*") BytePointer img, IntBuffer bbox);
	public native @Cast("cimg_library::CImg<unsigned char>*") BytePointer getNF(@Cast("cimg_library::CImg<unsigned char>*") BytePointer img, int[] bbox, @Cast("clandmark::fl_double_t*const") FloatPointer ground_truth/*=0*/);
	public native @Cast("cimg_library::CImg<unsigned char>*") BytePointer getNF(@Cast("cimg_library::CImg<unsigned char>*") BytePointer img, int[] bbox);

	/**
	 * \brief getGroundTruthNF
	 * @return
	 */
	public native IntPointer getGroundTruthNF();

	/**
	 * \brief getGroundTruth
	 * @return
	 */
	public native @Cast("clandmark::fl_double_t*") FloatPointer getGroundTruth();

	/**
	 * \brief getNormalizationFactor
	 * @return
	 */
	public native @Cast("clandmark::fl_double_t") float getNormalizationFactor();

	/**
	 * \brief setLossTables
	 * @param loss_data
	 * @param landmark_id
	 */
	public native void setLossTable(@Cast("clandmark::fl_double_t*") FloatPointer loss_data, int landmark_id);

	/**
	 * \brief getVersion
	 * @return
	 */
	public native @StdString BytePointer getVersion();

	/**
	 * \brief getName
	 * @return
	 */
	public native @StdString BytePointer getName();

	/**
	 * \brief getSearchSpace
	 * @param landmark_id
	 * @return
	 */
	public native @Const IntPointer getSearchSpace(int landmark_id);

	/**
	 * \brief getBaseWindowSize
	 * @return
	 */
	public native @Const IntPointer getBaseWindowSize();

	/**
	 * \brief getScore
	 * @return
	 */
	public native @Cast("clandmark::fl_double_t") float getScore();

}



// #endif // _FLANDMARK_H__


// Parsed from CFeaturePool.h

/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014, 2015 Michal Uricar
 * Copyright (C) 2014, 2015 Michal Uricar
 */

// #include "CFeatures.h"

// #include <vector>

// #ifndef _CFEATUREPOOL_H__
// #define _CFEATUREPOOL_H__

/**
 * \brief The CFeaturePool class
 */
@Namespace("clandmark") @NoOffset public static class CFeaturePool extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CFeaturePool(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public CFeaturePool(long size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(long size);
    @Override public CFeaturePool position(long position) {
        return (CFeaturePool)super.position(position);
    }


	/**
	 * \brief CFeaturePool
	 */
	public CFeaturePool() { super((Pointer)null); allocate(); }
	private native void allocate();

	/**
	 * \brief CFeaturePool
	 * @param width
	 * @param height
	 */
	public CFeaturePool(int width, int height, @Cast("unsigned char*") BytePointer nf/*=0x0*/) { super((Pointer)null); allocate(width, height, nf); }
	private native void allocate(int width, int height, @Cast("unsigned char*") BytePointer nf/*=0x0*/);
	public CFeaturePool(int width, int height) { super((Pointer)null); allocate(width, height); }
	private native void allocate(int width, int height);
	public CFeaturePool(int width, int height, @Cast("unsigned char*") ByteBuffer nf/*=0x0*/) { super((Pointer)null); allocate(width, height, nf); }
	private native void allocate(int width, int height, @Cast("unsigned char*") ByteBuffer nf/*=0x0*/);
	public CFeaturePool(int width, int height, @Cast("unsigned char*") byte[] nf/*=0x0*/) { super((Pointer)null); allocate(width, height, nf); }
	private native void allocate(int width, int height, @Cast("unsigned char*") byte[] nf/*=0x0*/);

	/**
	 *
	 */

	/**
	 * \brief getFeaturesFromPool
	 * @param index
	 * @return
	 */
	public native CFeatures getFeaturesFromPool(@Cast("unsigned int") int index);

	/**
	 * \brief addFeaturesToPool
	 * @param features
	 */
	public native void addFeaturesToPool(CFeatures features);

	/**
	 * \brief updateNFmipmap
	 * @param nf
	 */
	public native void updateNFmipmap(int width, int height, @Cast("unsigned char*const") BytePointer nf);
	public native void updateNFmipmap(int width, int height, @Cast("unsigned char*const") ByteBuffer nf);
	public native void updateNFmipmap(int width, int height, @Cast("unsigned char*const") byte[] nf);

	/**
	 * \brief setNFmipmap
	 * @param mipmap
	 */
	public native void setNFmipmap(@Cast("unsigned char*const") BytePointer mipmap);
	public native void setNFmipmap(@Cast("unsigned char*const") ByteBuffer mipmap);
	public native void setNFmipmap(@Cast("unsigned char*const") byte[] mipmap);

	/**
	 * \brief getCumulativeWidths
	 * @return
	 */
	public native IntPointer getCumulativeWidths();

	/**
	 * \brief getPyramidLevels
	 * @return
	 */
	public native int getPyramidLevels();

	/**
	 * \brief getWidth
	 * @return
	 */
	public native int getWidth();

	/**
	 * \brief getHeight
	 * @return
	 */
	public native int getHeight();

	/**
	 * \brief computeFeatures
	 */
	public native void computeFeatures();

	/**
	 * \brief updateFeaturesRaw
	 * @param data
	 */
	public native void updateFeaturesRaw(int index, Pointer data);

}



// #endif // _CFEATUREPOOL_H__


// Parsed from CFeatures.h

/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014, 2015 Michal Uricar
 * Copyright (C) 2014, 2015 Michal Uricar
 */

// #ifndef _CFEATURES_H__
// #define _CFEATURES_H__

//#include "CImg.h"

// #include "msvc-compat.h"

/**
 * \brief The CFeatures class
 */
@Namespace("clandmark") @NoOffset public static class CFeatures extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CFeatures(Pointer p) { super(p); }


	/**
	 * \brief CFeatures
	 */

	/**
	 * \brief CFeatures
	 * @param width
	 * @param height
	 */

	/**
	 * \brief ~CFeatures
	 */

	/**
	 * \brief compute
	 */
	public native void compute();

	/**
	 * \brief getFeatures
	 * @return
	 */
	public native Pointer getFeatures();

	/**
	 * \brief setFeatures
	 * @param features
	 */
	public native void setFeatures(CFeatures features);

	/**
	 * \brief setFeaturesRaw
	 * @param data
	 */
	public native void setFeaturesRaw(Pointer data);

	/**
	 * \brief setNFmipmap
	 * @param nfMipmap
	 */
	public native void setNFmipmap( @Cast("unsigned char*") BytePointer nfMipmap);
	public native void setNFmipmap( @Cast("unsigned char*") ByteBuffer nfMipmap);
	public native void setNFmipmap( @Cast("unsigned char*") byte[] nfMipmap);

	/** Normalized frame mipmap */
	public native @Cast("unsigned char*") BytePointer NFmipmap(); public native CFeatures NFmipmap(BytePointer NFmipmap);
	/** */
	@MemberGetter public native int kWidth();
	/** */
	@MemberGetter public native int kHeight();
	/** */
	@MemberGetter public native int kLevels();
	/** */
	public native IntPointer cumWidths(); public native CFeatures cumWidths(IntPointer cumWidths);
}



// #endif // _CFEATURES_H__


}
