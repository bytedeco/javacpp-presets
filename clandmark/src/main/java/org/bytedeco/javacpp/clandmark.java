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
	public native void detect(ByteCImg inputImage, IntPointer boundingBox, @Cast("clandmark::fl_double_t*const") FloatPointer ground_truth/*=0*/);
	public native void detect(ByteCImg inputImage, IntPointer boundingBox);
	public native void detect(ByteCImg inputImage, IntBuffer boundingBox, @Cast("clandmark::fl_double_t*const") FloatPointer ground_truth/*=0*/);
	public native void detect(ByteCImg inputImage, IntBuffer boundingBox);
	public native void detect(ByteCImg inputImage, int[] boundingBox, @Cast("clandmark::fl_double_t*const") FloatPointer ground_truth/*=0*/);
	public native void detect(ByteCImg inputImage, int[] boundingBox);

	/**
	 * \brief detect_optimized
	 * @param inputImage
	 * @param boundingBox
	 * @param ground_truth
	 */
	public native void detect_optimized(ByteCImg inputImage, IntPointer boundingBox, @Cast("clandmark::fl_double_t*const") FloatPointer ground_truth/*=0*/);
	public native void detect_optimized(ByteCImg inputImage, IntPointer boundingBox);
	public native void detect_optimized(ByteCImg inputImage, IntBuffer boundingBox, @Cast("clandmark::fl_double_t*const") FloatPointer ground_truth/*=0*/);
	public native void detect_optimized(ByteCImg inputImage, IntBuffer boundingBox);
	public native void detect_optimized(ByteCImg inputImage, int[] boundingBox, @Cast("clandmark::fl_double_t*const") FloatPointer ground_truth/*=0*/);
	public native void detect_optimized(ByteCImg inputImage, int[] boundingBox);

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
	public native void detect_mirrored(ByteCImg inputImage, IntPointer boundingBox, @Cast("clandmark::fl_double_t*const") FloatPointer ground_truth/*=0*/);
	public native void detect_mirrored(ByteCImg inputImage, IntPointer boundingBox);
	public native void detect_mirrored(ByteCImg inputImage, IntBuffer boundingBox, @Cast("clandmark::fl_double_t*const") FloatPointer ground_truth/*=0*/);
	public native void detect_mirrored(ByteCImg inputImage, IntBuffer boundingBox);
	public native void detect_mirrored(ByteCImg inputImage, int[] boundingBox, @Cast("clandmark::fl_double_t*const") FloatPointer ground_truth/*=0*/);
	public native void detect_mirrored(ByteCImg inputImage, int[] boundingBox);

	/**
	 * \brief detect
	 * @param inputImage	normalized image frame
	 * @param ground_truth
	 */
	public native void detect_base(ByteCImg inputImage, IntPointer ground_truth/*=0*/);
	public native void detect_base(ByteCImg inputImage);
	public native void detect_base(ByteCImg inputImage, IntBuffer ground_truth/*=0*/);
	public native void detect_base(ByteCImg inputImage, int[] ground_truth/*=0*/);

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
	public native void detect_base_optimized(ByteCImg inputImage, IntPointer ground_truth/*=0*/);
	public native void detect_base_optimized(ByteCImg inputImage);
	public native void detect_base_optimized(ByteCImg inputImage, IntBuffer ground_truth/*=0*/);
	public native void detect_base_optimized(ByteCImg inputImage, int[] ground_truth/*=0*/);

	/**
	 * \brief nodemax_base
	 * @param inputImage
	 * @param ground_truth
	 */
	public native void nodemax_base(ByteCImg inputImage, IntPointer ground_truth/*=0*/);
	public native void nodemax_base(ByteCImg inputImage);
	public native void nodemax_base(ByteCImg inputImage, IntBuffer ground_truth/*=0*/);
	public native void nodemax_base(ByteCImg inputImage, int[] ground_truth/*=0*/);

	/**
	 * \brief getFeatures
	 * @param inputImage
	 * @param boundingBox
	 * @return
	 */
	public native @Cast("clandmark::fl_double_t*") FloatPointer getFeatures(ByteCImg inputImage, IntPointer boundingBox, IntPointer configuration);
	public native @Cast("clandmark::fl_double_t*") FloatPointer getFeatures(ByteCImg inputImage, IntBuffer boundingBox, IntBuffer configuration);
	public native @Cast("clandmark::fl_double_t*") FloatPointer getFeatures(ByteCImg inputImage, int[] boundingBox, int[] configuration);

	/**
	 * \brief getFeatures_base
	 * @param nf
	 * @param configuration
	 * @return
	 */
	public native @Cast("clandmark::fl_double_t*") FloatPointer getFeatures_base(ByteCImg nf, IntPointer configuration);
	public native @Cast("clandmark::fl_double_t*") FloatPointer getFeatures_base(ByteCImg nf, IntBuffer configuration);
	public native @Cast("clandmark::fl_double_t*") FloatPointer getFeatures_base(ByteCImg nf, int[] configuration);

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
	public native @Cast("clandmark::fl_double_t*") FloatPointer getPsiNodes_base(ByteCImg nf, IntPointer configuration);
	public native @Cast("clandmark::fl_double_t*") FloatPointer getPsiNodes_base(ByteCImg nf, IntBuffer configuration);
	public native @Cast("clandmark::fl_double_t*") FloatPointer getPsiNodes_base(ByteCImg nf, int[] configuration);

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
	public native ByteCImg getNF();

	/**
	 * \brief getNF
	 * @param img
	 * @param bbox
	 * @return
	 */
	public native ByteCImg getNF(ByteCImg img, IntPointer bbox, @Cast("clandmark::fl_double_t*const") FloatPointer ground_truth/*=0*/);
	public native ByteCImg getNF(ByteCImg img, IntPointer bbox);
	public native ByteCImg getNF(ByteCImg img, IntBuffer bbox, @Cast("clandmark::fl_double_t*const") FloatPointer ground_truth/*=0*/);
	public native ByteCImg getNF(ByteCImg img, IntBuffer bbox);
	public native ByteCImg getNF(ByteCImg img, int[] bbox, @Cast("clandmark::fl_double_t*const") FloatPointer ground_truth/*=0*/);
	public native ByteCImg getNF(ByteCImg img, int[] bbox);

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
	public CFeaturePool(int width, int height, byte nf/*=0x0*/) { super((Pointer)null); allocate(width, height, nf); }
	private native void allocate(int width, int height, byte nf/*=0x0*/);
	public CFeaturePool(int width, int height) { super((Pointer)null); allocate(width, height); }
	private native void allocate(int width, int height);

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
	public native void updateNFmipmap(int width, int height, byte nf);

	/**
	 * \brief setNFmipmap
	 * @param mipmap
	 */
	public native void setNFmipmap(byte mipmap);

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
	public native void setNFmipmap( byte nfMipmap);

	/** Normalized frame mipmap */
	public native byte NFmipmap(); public native CFeatures NFmipmap(byte NFmipmap);
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


// Parsed from CImg.h

/*
 #
 #  File            : CImg.h
 #                    ( C++ header file )
 #
 #  Description     : The C++ Template Image Processing Toolkit.
 #                    This file is the main component of the CImg Library project.
 #                    ( http://cimg.sourceforge.net )
 #
 #  Project manager : David Tschumperle.
 #                    ( http://tschumperle.users.greyc.fr/ )
 #
 #                    A complete list of contributors is available in file 'README.txt'
 #                    distributed within the CImg package.
 #
 #  Licenses        : This file is 'dual-licensed', you have to choose one
 #                    of the two licenses below to apply.
 #
 #                    CeCILL-C
 #                    The CeCILL-C license is close to the GNU LGPL.
 #                    ( http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html )
 #
 #                or  CeCILL v2.0
 #                    The CeCILL license is compatible with the GNU GPL.
 #                    ( http://www.cecill.info/licences/Licence_CeCILL_V2-en.html )
 #
 #  This software is governed either by the CeCILL or the CeCILL-C license
 #  under French law and abiding by the rules of distribution of free software.
 #  You can  use, modify and or redistribute the software under the terms of
 #  the CeCILL or CeCILL-C licenses as circulated by CEA, CNRS and INRIA
 #  at the following URL: "http://www.cecill.info".
 #
 #  As a counterpart to the access to the source code and  rights to copy,
 #  modify and redistribute granted by the license, users are provided only
 #  with a limited warranty  and the software's author,  the holder of the
 #  economic rights,  and the successive licensors  have only  limited
 #  liability.
 #
 #  In this respect, the user's attention is drawn to the risks associated
 #  with loading,  using,  modifying and/or developing or reproducing the
 #  software by the user in light of its specific status of free software,
 #  that may mean  that it is complicated to manipulate,  and  that  also
 #  therefore means  that it is reserved for developers  and  experienced
 #  professionals having in-depth computer knowledge. Users are therefore
 #  encouraged to load and test the software's suitability as regards their
 #  requirements in conditions enabling the security of their systems and/or
 #  data to be ensured and,  more generally, to use and operate it in the
 #  same conditions as regards security.
 #
 #  The fact that you are presently reading this means that you have had
 #  knowledge of the CeCILL and CeCILL-C licenses and that you accept its terms.
 #
*/

// Set version number of the library.
// #ifndef cimg_version
public static final int cimg_version = 156;

/*-----------------------------------------------------------
 #
 # Test and possibly auto-set CImg configuration variables
 # and include required headers.
 #
 # If you find that the default configuration variables are
 # not adapted to your system, you can override their values
 # before including the header file "CImg.h"
 # (use the #define directive).
 #
 ------------------------------------------------------------*/

// Include standard C++ headers.
// This is the minimal set of required headers to make CImg-based codes compile.
// #include <cstdio>
// #include <cstdlib>
// #include <cstdarg>
// #include <cstring>
// #include <cmath>
// #include <ctime>
// #include <exception>

// Detect/configure OS variables.
//
// Define 'cimg_OS' to: '0' for an unknown OS (will try to minize library dependencies).
//                      '1' for a Unix-like OS (Linux, Solaris, BSD, MacOSX, Irix, ...).
//                      '2' for Microsoft Windows.
//                      (auto-detection is performed if 'cimg_OS' is not set by the user).
// #ifndef cimg_OS
// #if defined(unix)        || defined(__unix)      || defined(__unix__)
//  || defined(linux)       || defined(__linux)     || defined(__linux__)
//  || defined(sun)         || defined(__sun)
//  || defined(BSD)         || defined(__OpenBSD__) || defined(__NetBSD__)
//  || defined(__FreeBSD__) || defined __DragonFly__
//  || defined(sgi)         || defined(__sgi)
//  || defined(__MACOSX__)  || defined(__APPLE__)
//  || defined(__CYGWIN__)
public static final int cimg_OS = 1;
// #elif defined(_MSC_VER) || defined(WIN32)  || defined(_WIN32) || defined(__WIN32__)
//    || defined(WIN64)    || defined(_WIN64) || defined(__WIN64__)
// #else
// #endif
// #elif !(cimg_OS==0 || cimg_OS==1 || cimg_OS==2)
// #error CImg Library: Invalid configuration variable 'cimg_OS'.
// #error (correct values are '0 = unknown OS', '1 = Unix-like OS', '2 = Microsoft Windows').
// #endif

// Disable silly warnings on some Microsoft VC++ compilers.
// #ifdef _MSC_VER
// #pragma warning(push)
// #pragma warning(disable:4311)
// #pragma warning(disable:4312)
// #pragma warning(disable:4800)
// #pragma warning(disable:4804)
// #pragma warning(disable:4996)
public static final int _CRT_SECURE_NO_DEPRECATE = 1;
public static final int _CRT_NONSTDC_NO_DEPRECATE = 1;
// #endif

// Include OS-specific headers.
// #if cimg_OS==1
// #include <sys/types.h>
// #include <sys/time.h>
// #include <unistd.h>
// #elif cimg_OS==2
// #ifndef NOMINMAX
// #define NOMINMAX
// #endif
// #include <windows.h>
// #ifndef _WIN32_IE
public static final int _WIN32_IE = 0x0400;
// #endif
// #include <shlobj.h>
// #include <process.h>
// #include <io.h>
public static final int cimg_snprintf = _snprintf;
public static final int cimg_vsnprintf = _vsnprintf;
// #endif
// #ifndef cimg_snprintf
// #include <stdio.h>
// #endif

// Configure filename separator.
//
// Filename separator is set by default to '/', except for Windows where it is '\'.
// #ifndef cimg_file_separator
// #if cimg_OS==2
public static final int cimg_file_separator = '\\';
// #else
// #endif
// #endif

// Configure verbosity of output messages.
//
// Define 'cimg_verbosity' to: '0' to hide library messages (quiet mode).
//                             '1' to output library messages on the console.
//                             '2' to output library messages on a basic dialog window (default behavior).
//                             '3' to do as '1' + add extra warnings (may slow down the code!).
//                             '4' to do as '2' + add extra warnings (may slow down the code!).
//
// Define 'cimg_strict_warnings' to replace warning messages by exception throwns.
//
// Define 'cimg_use_vt100' to allow output of color messages on VT100-compatible terminals.
// #ifndef cimg_verbosity
// #elif !(cimg_verbosity==0 || cimg_verbosity==1 || cimg_verbosity==2 || cimg_verbosity==3 || cimg_verbosity==4)
// #error CImg Library: Configuration variable 'cimg_verbosity' is badly defined.
// #error (should be { 0=quiet | 1=console | 2=dialog | 3=console+warnings | 4=dialog+warnings }).
// #endif

// Configure display framework.
//
// Define 'cimg_display' to: '0' to disable display capabilities.
//                           '1' to use the X-Window framework (X11).
//                           '2' to use the Microsoft GDI32 framework.
// #ifndef cimg_display
// #if cimg_OS==0
// #elif cimg_OS==1
// #if defined(__MACOSX__) || defined(__APPLE__)
// #else
// #endif
// #elif cimg_OS==2
// #endif
// #elif !(cimg_display==0 || cimg_display==1 || cimg_display==2)
// #error CImg Library: Configuration variable 'cimg_display' is badly defined.
// #error (should be { 0=none | 1=X-Window (X11) | 2=Microsoft GDI32 }).
// #endif

// Include display-specific headers.
// #if cimg_display==1
// #include <X11/Xlib.h>
// #include <X11/Xutil.h>
// #include <X11/keysym.h>
// #include <pthread.h>
// #ifdef cimg_use_xshm
// #include <sys/ipc.h>
// #include <sys/shm.h>
// #include <X11/extensions/XShm.h>
// #endif
// #ifdef cimg_use_xrandr
// #include <X11/extensions/Xrandr.h>
// #endif
// #endif
// #ifndef cimg_appname
public static final String cimg_appname = "CImg";
// #endif

// Configure OpenMP support.
// (http://www.openmp.org)
//
// Define 'cimg_use_openmp' to enable OpenMP support.
//
// OpenMP directives may be used in a (very) few CImg functions to get
// advantages of multi-core CPUs.
// #ifdef cimg_use_openmp
// #include "omp.h"
// #define _cimg_static
// #else
// #define _cimg_static static
// #endif

// Configure OpenCV support.
// (http://opencv.willowgarage.com/wiki/)
//
// Define 'cimg_use_opencv' to enable OpenCV support.
//
// OpenCV library may be used to access images from cameras
// (see method 'CImg<T>::load_camera()').
// #ifdef cimg_use_opencv
// #ifdef True
// #undef True
// #define _cimg_redefine_True
// #endif
// #ifdef False
// #undef False
// #define _cimg_redefine_False
// #endif
// #include <cstddef>
// #include "cv.h"
// #include "highgui.h"
// #endif

// Configure LibPNG support.
// (http://www.libpng.org)
//
// Define 'cimg_use_png' to enable LibPNG support.
//
// PNG library may be used to get a native support of '.png' files.
// (see methods 'CImg<T>::{load,save}_png()'.
// #ifdef cimg_use_png
// #include "png.h"
// #endif

// Configure LibJPEG support.
// (http://en.wikipedia.org/wiki/Libjpeg)
//
// Define 'cimg_use_jpeg' to enable LibJPEG support.
//
// JPEG library may be used to get a native support of '.jpg' files.
// (see methods 'CImg<T>::{load,save}_jpeg()').
// #ifdef cimg_use_jpeg
// #include "jpeglib.h"
// #include "setjmp.h"
// #endif

// Configure LibTIFF support.
// (http://www.libtiff.org)
//
// Define 'cimg_use_tiff' to enable LibTIFF support.
//
// TIFF library may be used to get a native support of '.tif' files.
// (see methods 'CImg[List]<T>::{load,save}_tiff()').
// #ifdef cimg_use_tiff
// #include "tiffio.h"
// #endif

// Configure LibMINC2 support.
// (http://en.wikibooks.org/wiki/MINC/Reference/MINC2.0_File_Format_Reference)
//
// Define 'cimg_use_minc2' to enable LibMINC2 support.
//
// MINC2 library may be used to get a native support of '.mnc' files.
// (see methods 'CImg<T>::{load,save}_minc2()').
// #ifdef cimg_use_minc2
// #include "minc_io_simple_volume.h"
// #include "minc_1_simple.h"
// #include "minc_1_simple_rw.h"
// #endif

// Configure FFMPEG support.
// (http://www.ffmpeg.org)
//
// Define 'cimg_use_ffmpeg' to enable FFMPEG lib support.
//
// Avcodec and Avformat libraries from FFMPEG may be used
// to get a native support of various video file formats.
// (see methods 'CImg[List]<T>::load_ffmpeg()').
// #ifdef cimg_use_ffmpeg
// #if (defined(_STDINT_H) || defined(_STDINT_H_)) && !defined(UINT64_C)
// #warning "__STDC_CONSTANT_MACROS has to be defined before including <stdint.h>, this file will probably not compile."
// #endif
// #ifndef __STDC_CONSTANT_MACROS
// #define __STDC_CONSTANT_MACROS // ...or stdint.h wont' define UINT64_C, needed by libavutil
// #endif
// #include "avformat.h"
// #include "avcodec.h"
// #include "swscale.h"
// #endif

// Configure Zlib support.
// (http://www.zlib.net)
//
// Define 'cimg_use_zlib' to enable Zlib support.
//
// Zlib library may be used to allow compressed data in '.cimgz' files
// (see methods 'CImg[List]<T>::{load,save}_cimg()').
// #ifdef cimg_use_zlib
// #include "zlib.h"
// #endif

// Configure Magick++ support.
// (http://www.imagemagick.org/Magick++)
//
// Define 'cimg_use_magick' to enable Magick++ support.
//
// Magick++ library may be used to get a native support of various image file formats.
// (see methods 'CImg<T>::{load,save}()').
// #ifdef cimg_use_magick
// #include "Magick++.h"
// #endif

// Configure FFTW3 support.
// (http://www.fftw.org)
//
// Define 'cimg_use_fftw3' to enable libFFTW3 support.
//
// FFTW3 library may be used to efficiently compute the Fast Fourier Transform
// of image data, without restriction on the image size.
// (see method 'CImg[List]<T>::FFT()').
// #ifdef cimg_use_fftw3
// #include "fftw3.h"
// #endif

// Configure LibBoard support.
// (http://libboard.sourceforge.net/)
//
// Define 'cimg_use_board' to enable Board support.
//
// Board library may be used to draw 3d objects in vector-graphics canvas
// that can be saved as '.ps' or '.svg' files afterwards.
// (see method 'CImg<T>::draw_object3d()').
// #ifdef cimg_use_board
// #ifdef None
// #undef None
// #define _cimg_redefine_None
// #endif
// #include "Board.h"
// #endif

// Configure OpenEXR support.
// (http://www.openexr.com/)
//
// Define 'cimg_use_openexr' to enable OpenEXR support.
//
// OpenEXR library may be used to get a native support of '.exr' files.
// (see methods 'CImg<T>::{load,save}_exr()').
// #ifdef cimg_use_openexr
// #include "ImfRgbaFile.h"
// #include "ImfInputFile.h"
// #include "ImfChannelList.h"
// #include "ImfMatrixAttribute.h"
// #include "ImfArray.h"
// #endif

// Lapack configuration.
// (http://www.netlib.org/lapack)
//
// Define 'cimg_use_lapack' to enable LAPACK support.
//
// Lapack library may be used in several CImg methods to speed up
// matrix computations (eigenvalues, inverse, ...).
// #ifdef cimg_use_lapack
  public static native void sgetrf_(IntPointer arg0, IntPointer arg1, FloatPointer arg2, IntPointer arg3, IntPointer arg4, IntPointer arg5);
  public static native void sgetrf_(IntBuffer arg0, IntBuffer arg1, FloatBuffer arg2, IntBuffer arg3, IntBuffer arg4, IntBuffer arg5);
  public static native void sgetrf_(int[] arg0, int[] arg1, float[] arg2, int[] arg3, int[] arg4, int[] arg5);
  public static native void sgetri_(IntPointer arg0, FloatPointer arg1, IntPointer arg2, IntPointer arg3, FloatPointer arg4, IntPointer arg5, IntPointer arg6);
  public static native void sgetri_(IntBuffer arg0, FloatBuffer arg1, IntBuffer arg2, IntBuffer arg3, FloatBuffer arg4, IntBuffer arg5, IntBuffer arg6);
  public static native void sgetri_(int[] arg0, float[] arg1, int[] arg2, int[] arg3, float[] arg4, int[] arg5, int[] arg6);
  public static native void sgetrs_(@Cast("char*") BytePointer arg0, IntPointer arg1, IntPointer arg2, FloatPointer arg3, IntPointer arg4, IntPointer arg5, FloatPointer arg6, IntPointer arg7, IntPointer arg8);
  public static native void sgetrs_(@Cast("char*") ByteBuffer arg0, IntBuffer arg1, IntBuffer arg2, FloatBuffer arg3, IntBuffer arg4, IntBuffer arg5, FloatBuffer arg6, IntBuffer arg7, IntBuffer arg8);
  public static native void sgetrs_(@Cast("char*") byte[] arg0, int[] arg1, int[] arg2, float[] arg3, int[] arg4, int[] arg5, float[] arg6, int[] arg7, int[] arg8);
  public static native void sgesvd_(@Cast("char*") BytePointer arg0, @Cast("char*") BytePointer arg1, IntPointer arg2, IntPointer arg3, FloatPointer arg4, IntPointer arg5, FloatPointer arg6, FloatPointer arg7, IntPointer arg8, FloatPointer arg9, IntPointer arg10, FloatPointer arg11, IntPointer arg12, IntPointer arg13);
  public static native void sgesvd_(@Cast("char*") ByteBuffer arg0, @Cast("char*") ByteBuffer arg1, IntBuffer arg2, IntBuffer arg3, FloatBuffer arg4, IntBuffer arg5, FloatBuffer arg6, FloatBuffer arg7, IntBuffer arg8, FloatBuffer arg9, IntBuffer arg10, FloatBuffer arg11, IntBuffer arg12, IntBuffer arg13);
  public static native void sgesvd_(@Cast("char*") byte[] arg0, @Cast("char*") byte[] arg1, int[] arg2, int[] arg3, float[] arg4, int[] arg5, float[] arg6, float[] arg7, int[] arg8, float[] arg9, int[] arg10, float[] arg11, int[] arg12, int[] arg13);
  public static native void ssyev_(@Cast("char*") BytePointer arg0, @Cast("char*") BytePointer arg1, IntPointer arg2, FloatPointer arg3, IntPointer arg4, FloatPointer arg5, FloatPointer arg6, IntPointer arg7, IntPointer arg8);
  public static native void ssyev_(@Cast("char*") ByteBuffer arg0, @Cast("char*") ByteBuffer arg1, IntBuffer arg2, FloatBuffer arg3, IntBuffer arg4, FloatBuffer arg5, FloatBuffer arg6, IntBuffer arg7, IntBuffer arg8);
  public static native void ssyev_(@Cast("char*") byte[] arg0, @Cast("char*") byte[] arg1, int[] arg2, float[] arg3, int[] arg4, float[] arg5, float[] arg6, int[] arg7, int[] arg8);
  public static native void dgetrf_(IntPointer arg0, IntPointer arg1, DoublePointer arg2, IntPointer arg3, IntPointer arg4, IntPointer arg5);
  public static native void dgetrf_(IntBuffer arg0, IntBuffer arg1, DoubleBuffer arg2, IntBuffer arg3, IntBuffer arg4, IntBuffer arg5);
  public static native void dgetrf_(int[] arg0, int[] arg1, double[] arg2, int[] arg3, int[] arg4, int[] arg5);
  public static native void dgetri_(IntPointer arg0, DoublePointer arg1, IntPointer arg2, IntPointer arg3, DoublePointer arg4, IntPointer arg5, IntPointer arg6);
  public static native void dgetri_(IntBuffer arg0, DoubleBuffer arg1, IntBuffer arg2, IntBuffer arg3, DoubleBuffer arg4, IntBuffer arg5, IntBuffer arg6);
  public static native void dgetri_(int[] arg0, double[] arg1, int[] arg2, int[] arg3, double[] arg4, int[] arg5, int[] arg6);
  public static native void dgetrs_(@Cast("char*") BytePointer arg0, IntPointer arg1, IntPointer arg2, DoublePointer arg3, IntPointer arg4, IntPointer arg5, DoublePointer arg6, IntPointer arg7, IntPointer arg8);
  public static native void dgetrs_(@Cast("char*") ByteBuffer arg0, IntBuffer arg1, IntBuffer arg2, DoubleBuffer arg3, IntBuffer arg4, IntBuffer arg5, DoubleBuffer arg6, IntBuffer arg7, IntBuffer arg8);
  public static native void dgetrs_(@Cast("char*") byte[] arg0, int[] arg1, int[] arg2, double[] arg3, int[] arg4, int[] arg5, double[] arg6, int[] arg7, int[] arg8);
  public static native void dgesvd_(@Cast("char*") BytePointer arg0, @Cast("char*") BytePointer arg1, IntPointer arg2, IntPointer arg3, DoublePointer arg4, IntPointer arg5, DoublePointer arg6, DoublePointer arg7, IntPointer arg8, DoublePointer arg9, IntPointer arg10, DoublePointer arg11, IntPointer arg12, IntPointer arg13);
  public static native void dgesvd_(@Cast("char*") ByteBuffer arg0, @Cast("char*") ByteBuffer arg1, IntBuffer arg2, IntBuffer arg3, DoubleBuffer arg4, IntBuffer arg5, DoubleBuffer arg6, DoubleBuffer arg7, IntBuffer arg8, DoubleBuffer arg9, IntBuffer arg10, DoubleBuffer arg11, IntBuffer arg12, IntBuffer arg13);
  public static native void dgesvd_(@Cast("char*") byte[] arg0, @Cast("char*") byte[] arg1, int[] arg2, int[] arg3, double[] arg4, int[] arg5, double[] arg6, double[] arg7, int[] arg8, double[] arg9, int[] arg10, double[] arg11, int[] arg12, int[] arg13);
  public static native void dsyev_(@Cast("char*") BytePointer arg0, @Cast("char*") BytePointer arg1, IntPointer arg2, DoublePointer arg3, IntPointer arg4, DoublePointer arg5, DoublePointer arg6, IntPointer arg7, IntPointer arg8);
  public static native void dsyev_(@Cast("char*") ByteBuffer arg0, @Cast("char*") ByteBuffer arg1, IntBuffer arg2, DoubleBuffer arg3, IntBuffer arg4, DoubleBuffer arg5, DoubleBuffer arg6, IntBuffer arg7, IntBuffer arg8);
  public static native void dsyev_(@Cast("char*") byte[] arg0, @Cast("char*") byte[] arg1, int[] arg2, double[] arg3, int[] arg4, double[] arg5, double[] arg6, int[] arg7, int[] arg8);
  public static native void dgels_(@Cast("char*") BytePointer arg0, IntPointer arg1,IntPointer arg2,IntPointer arg3,DoublePointer arg4,IntPointer arg5,DoublePointer arg6,IntPointer arg7,DoublePointer arg8,IntPointer arg9,IntPointer arg10);
  public static native void dgels_(@Cast("char*") ByteBuffer arg0, IntBuffer arg1,IntBuffer arg2,IntBuffer arg3,DoubleBuffer arg4,IntBuffer arg5,DoubleBuffer arg6,IntBuffer arg7,DoubleBuffer arg8,IntBuffer arg9,IntBuffer arg10);
  public static native void dgels_(@Cast("char*") byte[] arg0, int[] arg1,int[] arg2,int[] arg3,double[] arg4,int[] arg5,double[] arg6,int[] arg7,double[] arg8,int[] arg9,int[] arg10);
  public static native void sgels_(@Cast("char*") BytePointer arg0, IntPointer arg1,IntPointer arg2,IntPointer arg3,FloatPointer arg4,IntPointer arg5,FloatPointer arg6,IntPointer arg7,FloatPointer arg8,IntPointer arg9,IntPointer arg10);
  public static native void sgels_(@Cast("char*") ByteBuffer arg0, IntBuffer arg1,IntBuffer arg2,IntBuffer arg3,FloatBuffer arg4,IntBuffer arg5,FloatBuffer arg6,IntBuffer arg7,FloatBuffer arg8,IntBuffer arg9,IntBuffer arg10);
  public static native void sgels_(@Cast("char*") byte[] arg0, int[] arg1,int[] arg2,int[] arg3,float[] arg4,int[] arg5,float[] arg6,int[] arg7,float[] arg8,int[] arg9,int[] arg10);
// #endif

// Check if min/max/PI macros are defined.
//
// CImg does not compile if macros 'min', 'max' or 'PI' are defined,
// because it redefines functions min(), max() and const variable PI in the cimg:: namespace.
// so it '#undef' these macros if necessary, and restore them to reasonable
// values at the end of this file.
// #ifdef min
// #undef min
// #define _cimg_redefine_min
// #endif
// #ifdef max
// #undef max
// #define _cimg_redefine_max
// #endif
// #ifdef PI
// #undef PI
// #define _cimg_redefine_PI
// #endif

// Define 'cimg_library' namespace suffix.
//
// You may want to add a suffix to the 'cimg_library' namespace, for instance if you need to work
// with several versions of the library at the same time.
// #ifdef cimg_namespace_suffix
// #define __cimg_library_suffixed(s) cimg_library_##s
// #define _cimg_library_suffixed(s) __cimg_library_suffixed(s)
public static native @MemberGetter int cimg_library_suffixed();
public static final int cimg_library_suffixed = cimg_library_suffixed();
// #else
// #endif

/*------------------------------------------------------------------------------
  #
  # Define user-friendly macros.
  #
  # These CImg macros are prefixed by 'cimg_' and can be used safely in your own
  # code. They are useful to parse command line options, or to write image loops.
  #
  ------------------------------------------------------------------------------*/

// Macros to define program usage, and retrieve command line arguments.
// #define cimg_usage(usage) cimg_library_suffixed::cimg::option((char*)0,argc,argv,(char*)0,usage,false)
// #define cimg_help(str) cimg_library_suffixed::cimg::option((char*)0,argc,argv,str,(char*)0)
// #define cimg_option(name,defaut,usage) cimg_library_suffixed::cimg::option(name,argc,argv,defaut,usage)
// #define cimg_argument(pos) cimg_library_suffixed::cimg::argument(pos,argc,argv)
// #define cimg_argument1(pos,s0) cimg_library_suffixed::cimg::argument(pos,argc,argv,1,s0)
// #define cimg_argument2(pos,s0,s1) cimg_library_suffixed::cimg::argument(pos,argc,argv,2,s0,s1)
// #define cimg_argument3(pos,s0,s1,s2) cimg_library_suffixed::cimg::argument(pos,argc,argv,3,s0,s1,s2)
// #define cimg_argument4(pos,s0,s1,s2,s3) cimg_library_suffixed::cimg::argument(pos,argc,argv,4,s0,s1,s2,s3)
// #define cimg_argument5(pos,s0,s1,s2,s3,s4) cimg_library_suffixed::cimg::argument(pos,argc,argv,5,s0,s1,s2,s3,s4)
// #define cimg_argument6(pos,s0,s1,s2,s3,s4,s5) cimg_library_suffixed::cimg::argument(pos,argc,argv,6,s0,s1,s2,s3,s4,s5)
// #define cimg_argument7(pos,s0,s1,s2,s3,s4,s5,s6) cimg_library_suffixed::cimg::argument(pos,argc,argv,7,s0,s1,s2,s3,s4,s5,s6)
// #define cimg_argument8(pos,s0,s1,s2,s3,s4,s5,s6,s7) cimg_library_suffixed::cimg::argument(pos,argc,argv,8,s0,s1,s2,s3,s4,s5,s6,s7)
// #define cimg_argument9(pos,s0,s1,s2,s3,s4,s5,s6,s7,s8) cimg_library_suffixed::cimg::argument(pos,argc,argv,9,s0,s1,s2,s3,s4,s5,s6,s7,s8)

// Macros to define and manipulate local neighborhoods.
// #define CImg_2x2(I,T) T I[4];
//                       T& I##cc = I[0]; T& I##nc = I[1];
//                       T& I##cn = I[2]; T& I##nn = I[3];
//                       I##cc = I##nc =
//                       I##cn = I##nn = 0

// #define CImg_3x3(I,T) T I[9];
//                       T& I##pp = I[0]; T& I##cp = I[1]; T& I##np = I[2];
//                       T& I##pc = I[3]; T& I##cc = I[4]; T& I##nc = I[5];
//                       T& I##pn = I[6]; T& I##cn = I[7]; T& I##nn = I[8];
//                       I##pp = I##cp = I##np =
//                       I##pc = I##cc = I##nc =
//                       I##pn = I##cn = I##nn = 0

// #define CImg_4x4(I,T) T I[16];
//                       T& I##pp = I[0]; T& I##cp = I[1]; T& I##np = I[2]; T& I##ap = I[3];
//                       T& I##pc = I[4]; T& I##cc = I[5]; T& I##nc = I[6]; T& I##ac = I[7];
//                       T& I##pn = I[8]; T& I##cn = I[9]; T& I##nn = I[10]; T& I##an = I[11];
//                       T& I##pa = I[12]; T& I##ca = I[13]; T& I##na = I[14]; T& I##aa = I[15];
//                       I##pp = I##cp = I##np = I##ap =
//                       I##pc = I##cc = I##nc = I##ac =
//                       I##pn = I##cn = I##nn = I##an =
//                       I##pa = I##ca = I##na = I##aa = 0

// #define CImg_5x5(I,T) T I[25];
//                       T& I##bb = I[0]; T& I##pb = I[1]; T& I##cb = I[2]; T& I##nb = I[3]; T& I##ab = I[4];
//                       T& I##bp = I[5]; T& I##pp = I[6]; T& I##cp = I[7]; T& I##np = I[8]; T& I##ap = I[9];
//                       T& I##bc = I[10]; T& I##pc = I[11]; T& I##cc = I[12]; T& I##nc = I[13]; T& I##ac = I[14];
//                       T& I##bn = I[15]; T& I##pn = I[16]; T& I##cn = I[17]; T& I##nn = I[18]; T& I##an = I[19];
//                       T& I##ba = I[20]; T& I##pa = I[21]; T& I##ca = I[22]; T& I##na = I[23]; T& I##aa = I[24];
//                       I##bb = I##pb = I##cb = I##nb = I##ab =
//                       I##bp = I##pp = I##cp = I##np = I##ap =
//                       I##bc = I##pc = I##cc = I##nc = I##ac =
//                       I##bn = I##pn = I##cn = I##nn = I##an =
//                       I##ba = I##pa = I##ca = I##na = I##aa = 0

// #define CImg_2x2x2(I,T) T I[8];
//                       T& I##ccc = I[0]; T& I##ncc = I[1];
//                       T& I##cnc = I[2]; T& I##nnc = I[3];
//                       T& I##ccn = I[4]; T& I##ncn = I[5];
//                       T& I##cnn = I[6]; T& I##nnn = I[7];
//                       I##ccc = I##ncc =
//                       I##cnc = I##nnc =
//                       I##ccn = I##ncn =
//                       I##cnn = I##nnn = 0

// #define CImg_3x3x3(I,T) T I[27];
//                       T& I##ppp = I[0]; T& I##cpp = I[1]; T& I##npp = I[2];
//                       T& I##pcp = I[3]; T& I##ccp = I[4]; T& I##ncp = I[5];
//                       T& I##pnp = I[6]; T& I##cnp = I[7]; T& I##nnp = I[8];
//                       T& I##ppc = I[9]; T& I##cpc = I[10]; T& I##npc = I[11];
//                       T& I##pcc = I[12]; T& I##ccc = I[13]; T& I##ncc = I[14];
//                       T& I##pnc = I[15]; T& I##cnc = I[16]; T& I##nnc = I[17];
//                       T& I##ppn = I[18]; T& I##cpn = I[19]; T& I##npn = I[20];
//                       T& I##pcn = I[21]; T& I##ccn = I[22]; T& I##ncn = I[23];
//                       T& I##pnn = I[24]; T& I##cnn = I[25]; T& I##nnn = I[26];
//                       I##ppp = I##cpp = I##npp =
//                       I##pcp = I##ccp = I##ncp =
//                       I##pnp = I##cnp = I##nnp =
//                       I##ppc = I##cpc = I##npc =
//                       I##pcc = I##ccc = I##ncc =
//                       I##pnc = I##cnc = I##nnc =
//                       I##ppn = I##cpn = I##npn =
//                       I##pcn = I##ccn = I##ncn =
//                       I##pnn = I##cnn = I##nnn = 0

// #define cimg_get2x2(img,x,y,z,c,I,T)
//   I[0] = (T)(img)(x,y,z,c), I[1] = (T)(img)(_n1##x,y,z,c), I[2] = (T)(img)(x,_n1##y,z,c), I[3] = (T)(img)(_n1##x,_n1##y,z,c)

// #define cimg_get3x3(img,x,y,z,c,I,T)
//   I[0] = (T)(img)(_p1##x,_p1##y,z,c), I[1] = (T)(img)(x,_p1##y,z,c), I[2] = (T)(img)(_n1##x,_p1##y,z,c), I[3] = (T)(img)(_p1##x,y,z,c),
//   I[4] = (T)(img)(x,y,z,c), I[5] = (T)(img)(_n1##x,y,z,c), I[6] = (T)(img)(_p1##x,_n1##y,z,c), I[7] = (T)(img)(x,_n1##y,z,c),
//   I[8] = (T)(img)(_n1##x,_n1##y,z,c)

// #define cimg_get4x4(img,x,y,z,c,I,T)
//   I[0] = (T)(img)(_p1##x,_p1##y,z,c), I[1] = (T)(img)(x,_p1##y,z,c), I[2] = (T)(img)(_n1##x,_p1##y,z,c), I[3] = (T)(img)(_n2##x,_p1##y,z,c),
//   I[4] = (T)(img)(_p1##x,y,z,c), I[5] = (T)(img)(x,y,z,c), I[6] = (T)(img)(_n1##x,y,z,c), I[7] = (T)(img)(_n2##x,y,z,c),
//   I[8] = (T)(img)(_p1##x,_n1##y,z,c), I[9] = (T)(img)(x,_n1##y,z,c), I[10] = (T)(img)(_n1##x,_n1##y,z,c), I[11] = (T)(img)(_n2##x,_n1##y,z,c),
//   I[12] = (T)(img)(_p1##x,_n2##y,z,c), I[13] = (T)(img)(x,_n2##y,z,c), I[14] = (T)(img)(_n1##x,_n2##y,z,c), I[15] = (T)(img)(_n2##x,_n2##y,z,c)

// #define cimg_get5x5(img,x,y,z,c,I,T)
//   I[0] = (T)(img)(_p2##x,_p2##y,z,c), I[1] = (T)(img)(_p1##x,_p2##y,z,c), I[2] = (T)(img)(x,_p2##y,z,c), I[3] = (T)(img)(_n1##x,_p2##y,z,c),
//   I[4] = (T)(img)(_n2##x,_p2##y,z,c), I[5] = (T)(img)(_p2##x,_p1##y,z,c), I[6] = (T)(img)(_p1##x,_p1##y,z,c), I[7] = (T)(img)(x,_p1##y,z,c),
//   I[8] = (T)(img)(_n1##x,_p1##y,z,c), I[9] = (T)(img)(_n2##x,_p1##y,z,c), I[10] = (T)(img)(_p2##x,y,z,c), I[11] = (T)(img)(_p1##x,y,z,c),
//   I[12] = (T)(img)(x,y,z,c), I[13] = (T)(img)(_n1##x,y,z,c), I[14] = (T)(img)(_n2##x,y,z,c), I[15] = (T)(img)(_p2##x,_n1##y,z,c),
//   I[16] = (T)(img)(_p1##x,_n1##y,z,c), I[17] = (T)(img)(x,_n1##y,z,c), I[18] = (T)(img)(_n1##x,_n1##y,z,c), I[19] = (T)(img)(_n2##x,_n1##y,z,c),
//   I[20] = (T)(img)(_p2##x,_n2##y,z,c), I[21] = (T)(img)(_p1##x,_n2##y,z,c), I[22] = (T)(img)(x,_n2##y,z,c), I[23] = (T)(img)(_n1##x,_n2##y,z,c),
//   I[24] = (T)(img)(_n2##x,_n2##y,z,c)

// #define cimg_get6x6(img,x,y,z,c,I,T)
//  I[0] = (T)(img)(_p2##x,_p2##y,z,c), I[1] = (T)(img)(_p1##x,_p2##y,z,c), I[2] = (T)(img)(x,_p2##y,z,c), I[3] = (T)(img)(_n1##x,_p2##y,z,c),
//  I[4] = (T)(img)(_n2##x,_p2##y,z,c), I[5] = (T)(img)(_n3##x,_p2##y,z,c), I[6] = (T)(img)(_p2##x,_p1##y,z,c), I[7] = (T)(img)(_p1##x,_p1##y,z,c),
//  I[8] = (T)(img)(x,_p1##y,z,c), I[9] = (T)(img)(_n1##x,_p1##y,z,c), I[10] = (T)(img)(_n2##x,_p1##y,z,c), I[11] = (T)(img)(_n3##x,_p1##y,z,c),
//  I[12] = (T)(img)(_p2##x,y,z,c), I[13] = (T)(img)(_p1##x,y,z,c), I[14] = (T)(img)(x,y,z,c), I[15] = (T)(img)(_n1##x,y,z,c),
//  I[16] = (T)(img)(_n2##x,y,z,c), I[17] = (T)(img)(_n3##x,y,z,c), I[18] = (T)(img)(_p2##x,_n1##y,z,c), I[19] = (T)(img)(_p1##x,_n1##y,z,c),
//  I[20] = (T)(img)(x,_n1##y,z,c), I[21] = (T)(img)(_n1##x,_n1##y,z,c), I[22] = (T)(img)(_n2##x,_n1##y,z,c), I[23] = (T)(img)(_n3##x,_n1##y,z,c),
//  I[24] = (T)(img)(_p2##x,_n2##y,z,c), I[25] = (T)(img)(_p1##x,_n2##y,z,c), I[26] = (T)(img)(x,_n2##y,z,c), I[27] = (T)(img)(_n1##x,_n2##y,z,c),
//  I[28] = (T)(img)(_n2##x,_n2##y,z,c), I[29] = (T)(img)(_n3##x,_n2##y,z,c), I[30] = (T)(img)(_p2##x,_n3##y,z,c), I[31] = (T)(img)(_p1##x,_n3##y,z,c),
//  I[32] = (T)(img)(x,_n3##y,z,c), I[33] = (T)(img)(_n1##x,_n3##y,z,c), I[34] = (T)(img)(_n2##x,_n3##y,z,c), I[35] = (T)(img)(_n3##x,_n3##y,z,c)

// #define cimg_get7x7(img,x,y,z,c,I,T)
//  I[0] = (T)(img)(_p3##x,_p3##y,z,c), I[1] = (T)(img)(_p2##x,_p3##y,z,c), I[2] = (T)(img)(_p1##x,_p3##y,z,c), I[3] = (T)(img)(x,_p3##y,z,c),
//  I[4] = (T)(img)(_n1##x,_p3##y,z,c), I[5] = (T)(img)(_n2##x,_p3##y,z,c), I[6] = (T)(img)(_n3##x,_p3##y,z,c), I[7] = (T)(img)(_p3##x,_p2##y,z,c),
//  I[8] = (T)(img)(_p2##x,_p2##y,z,c), I[9] = (T)(img)(_p1##x,_p2##y,z,c), I[10] = (T)(img)(x,_p2##y,z,c), I[11] = (T)(img)(_n1##x,_p2##y,z,c),
//  I[12] = (T)(img)(_n2##x,_p2##y,z,c), I[13] = (T)(img)(_n3##x,_p2##y,z,c), I[14] = (T)(img)(_p3##x,_p1##y,z,c), I[15] = (T)(img)(_p2##x,_p1##y,z,c),
//  I[16] = (T)(img)(_p1##x,_p1##y,z,c), I[17] = (T)(img)(x,_p1##y,z,c), I[18] = (T)(img)(_n1##x,_p1##y,z,c), I[19] = (T)(img)(_n2##x,_p1##y,z,c),
//  I[20] = (T)(img)(_n3##x,_p1##y,z,c), I[21] = (T)(img)(_p3##x,y,z,c), I[22] = (T)(img)(_p2##x,y,z,c), I[23] = (T)(img)(_p1##x,y,z,c),
//  I[24] = (T)(img)(x,y,z,c), I[25] = (T)(img)(_n1##x,y,z,c), I[26] = (T)(img)(_n2##x,y,z,c), I[27] = (T)(img)(_n3##x,y,z,c),
//  I[28] = (T)(img)(_p3##x,_n1##y,z,c), I[29] = (T)(img)(_p2##x,_n1##y,z,c), I[30] = (T)(img)(_p1##x,_n1##y,z,c), I[31] = (T)(img)(x,_n1##y,z,c),
//  I[32] = (T)(img)(_n1##x,_n1##y,z,c), I[33] = (T)(img)(_n2##x,_n1##y,z,c), I[34] = (T)(img)(_n3##x,_n1##y,z,c), I[35] = (T)(img)(_p3##x,_n2##y,z,c),
//  I[36] = (T)(img)(_p2##x,_n2##y,z,c), I[37] = (T)(img)(_p1##x,_n2##y,z,c), I[38] = (T)(img)(x,_n2##y,z,c), I[39] = (T)(img)(_n1##x,_n2##y,z,c),
//  I[40] = (T)(img)(_n2##x,_n2##y,z,c), I[41] = (T)(img)(_n3##x,_n2##y,z,c), I[42] = (T)(img)(_p3##x,_n3##y,z,c), I[43] = (T)(img)(_p2##x,_n3##y,z,c),
//  I[44] = (T)(img)(_p1##x,_n3##y,z,c), I[45] = (T)(img)(x,_n3##y,z,c), I[46] = (T)(img)(_n1##x,_n3##y,z,c), I[47] = (T)(img)(_n2##x,_n3##y,z,c),
//  I[48] = (T)(img)(_n3##x,_n3##y,z,c)

// #define cimg_get8x8(img,x,y,z,c,I,T)
//  I[0] = (T)(img)(_p3##x,_p3##y,z,c), I[1] = (T)(img)(_p2##x,_p3##y,z,c), I[2] = (T)(img)(_p1##x,_p3##y,z,c), I[3] = (T)(img)(x,_p3##y,z,c),
//  I[4] = (T)(img)(_n1##x,_p3##y,z,c), I[5] = (T)(img)(_n2##x,_p3##y,z,c), I[6] = (T)(img)(_n3##x,_p3##y,z,c), I[7] = (T)(img)(_n4##x,_p3##y,z,c),
//  I[8] = (T)(img)(_p3##x,_p2##y,z,c), I[9] = (T)(img)(_p2##x,_p2##y,z,c), I[10] = (T)(img)(_p1##x,_p2##y,z,c), I[11] = (T)(img)(x,_p2##y,z,c),
//  I[12] = (T)(img)(_n1##x,_p2##y,z,c), I[13] = (T)(img)(_n2##x,_p2##y,z,c), I[14] = (T)(img)(_n3##x,_p2##y,z,c), I[15] = (T)(img)(_n4##x,_p2##y,z,c),
//  I[16] = (T)(img)(_p3##x,_p1##y,z,c), I[17] = (T)(img)(_p2##x,_p1##y,z,c), I[18] = (T)(img)(_p1##x,_p1##y,z,c), I[19] = (T)(img)(x,_p1##y,z,c),
//  I[20] = (T)(img)(_n1##x,_p1##y,z,c), I[21] = (T)(img)(_n2##x,_p1##y,z,c), I[22] = (T)(img)(_n3##x,_p1##y,z,c), I[23] = (T)(img)(_n4##x,_p1##y,z,c),
//  I[24] = (T)(img)(_p3##x,y,z,c), I[25] = (T)(img)(_p2##x,y,z,c), I[26] = (T)(img)(_p1##x,y,z,c), I[27] = (T)(img)(x,y,z,c),
//  I[28] = (T)(img)(_n1##x,y,z,c), I[29] = (T)(img)(_n2##x,y,z,c), I[30] = (T)(img)(_n3##x,y,z,c), I[31] = (T)(img)(_n4##x,y,z,c),
//  I[32] = (T)(img)(_p3##x,_n1##y,z,c), I[33] = (T)(img)(_p2##x,_n1##y,z,c), I[34] = (T)(img)(_p1##x,_n1##y,z,c), I[35] = (T)(img)(x,_n1##y,z,c),
//  I[36] = (T)(img)(_n1##x,_n1##y,z,c), I[37] = (T)(img)(_n2##x,_n1##y,z,c), I[38] = (T)(img)(_n3##x,_n1##y,z,c), I[39] = (T)(img)(_n4##x,_n1##y,z,c),
//  I[40] = (T)(img)(_p3##x,_n2##y,z,c), I[41] = (T)(img)(_p2##x,_n2##y,z,c), I[42] = (T)(img)(_p1##x,_n2##y,z,c), I[43] = (T)(img)(x,_n2##y,z,c),
//  I[44] = (T)(img)(_n1##x,_n2##y,z,c), I[45] = (T)(img)(_n2##x,_n2##y,z,c), I[46] = (T)(img)(_n3##x,_n2##y,z,c), I[47] = (T)(img)(_n4##x,_n2##y,z,c),
//  I[48] = (T)(img)(_p3##x,_n3##y,z,c), I[49] = (T)(img)(_p2##x,_n3##y,z,c), I[50] = (T)(img)(_p1##x,_n3##y,z,c), I[51] = (T)(img)(x,_n3##y,z,c),
//  I[52] = (T)(img)(_n1##x,_n3##y,z,c), I[53] = (T)(img)(_n2##x,_n3##y,z,c), I[54] = (T)(img)(_n3##x,_n3##y,z,c), I[55] = (T)(img)(_n4##x,_n3##y,z,c),
//  I[56] = (T)(img)(_p3##x,_n4##y,z,c), I[57] = (T)(img)(_p2##x,_n4##y,z,c), I[58] = (T)(img)(_p1##x,_n4##y,z,c), I[59] = (T)(img)(x,_n4##y,z,c),
//  I[60] = (T)(img)(_n1##x,_n4##y,z,c), I[61] = (T)(img)(_n2##x,_n4##y,z,c), I[62] = (T)(img)(_n3##x,_n4##y,z,c), I[63] = (T)(img)(_n4##x,_n4##y,z,c);

// #define cimg_get9x9(img,x,y,z,c,I,T)
//  I[0] = (T)(img)(_p4##x,_p4##y,z,c), I[1] = (T)(img)(_p3##x,_p4##y,z,c), I[2] = (T)(img)(_p2##x,_p4##y,z,c), I[3] = (T)(img)(_p1##x,_p4##y,z,c),
//  I[4] = (T)(img)(x,_p4##y,z,c), I[5] = (T)(img)(_n1##x,_p4##y,z,c), I[6] = (T)(img)(_n2##x,_p4##y,z,c), I[7] = (T)(img)(_n3##x,_p4##y,z,c),
//  I[8] = (T)(img)(_n4##x,_p4##y,z,c), I[9] = (T)(img)(_p4##x,_p3##y,z,c), I[10] = (T)(img)(_p3##x,_p3##y,z,c), I[11] = (T)(img)(_p2##x,_p3##y,z,c),
//  I[12] = (T)(img)(_p1##x,_p3##y,z,c), I[13] = (T)(img)(x,_p3##y,z,c), I[14] = (T)(img)(_n1##x,_p3##y,z,c), I[15] = (T)(img)(_n2##x,_p3##y,z,c),
//  I[16] = (T)(img)(_n3##x,_p3##y,z,c), I[17] = (T)(img)(_n4##x,_p3##y,z,c), I[18] = (T)(img)(_p4##x,_p2##y,z,c), I[19] = (T)(img)(_p3##x,_p2##y,z,c),
//  I[20] = (T)(img)(_p2##x,_p2##y,z,c), I[21] = (T)(img)(_p1##x,_p2##y,z,c), I[22] = (T)(img)(x,_p2##y,z,c), I[23] = (T)(img)(_n1##x,_p2##y,z,c),
//  I[24] = (T)(img)(_n2##x,_p2##y,z,c), I[25] = (T)(img)(_n3##x,_p2##y,z,c), I[26] = (T)(img)(_n4##x,_p2##y,z,c), I[27] = (T)(img)(_p4##x,_p1##y,z,c),
//  I[28] = (T)(img)(_p3##x,_p1##y,z,c), I[29] = (T)(img)(_p2##x,_p1##y,z,c), I[30] = (T)(img)(_p1##x,_p1##y,z,c), I[31] = (T)(img)(x,_p1##y,z,c),
//  I[32] = (T)(img)(_n1##x,_p1##y,z,c), I[33] = (T)(img)(_n2##x,_p1##y,z,c), I[34] = (T)(img)(_n3##x,_p1##y,z,c), I[35] = (T)(img)(_n4##x,_p1##y,z,c),
//  I[36] = (T)(img)(_p4##x,y,z,c), I[37] = (T)(img)(_p3##x,y,z,c), I[38] = (T)(img)(_p2##x,y,z,c), I[39] = (T)(img)(_p1##x,y,z,c),
//  I[40] = (T)(img)(x,y,z,c), I[41] = (T)(img)(_n1##x,y,z,c), I[42] = (T)(img)(_n2##x,y,z,c), I[43] = (T)(img)(_n3##x,y,z,c),
//  I[44] = (T)(img)(_n4##x,y,z,c), I[45] = (T)(img)(_p4##x,_n1##y,z,c), I[46] = (T)(img)(_p3##x,_n1##y,z,c), I[47] = (T)(img)(_p2##x,_n1##y,z,c),
//  I[48] = (T)(img)(_p1##x,_n1##y,z,c), I[49] = (T)(img)(x,_n1##y,z,c), I[50] = (T)(img)(_n1##x,_n1##y,z,c), I[51] = (T)(img)(_n2##x,_n1##y,z,c),
//  I[52] = (T)(img)(_n3##x,_n1##y,z,c), I[53] = (T)(img)(_n4##x,_n1##y,z,c), I[54] = (T)(img)(_p4##x,_n2##y,z,c), I[55] = (T)(img)(_p3##x,_n2##y,z,c),
//  I[56] = (T)(img)(_p2##x,_n2##y,z,c), I[57] = (T)(img)(_p1##x,_n2##y,z,c), I[58] = (T)(img)(x,_n2##y,z,c), I[59] = (T)(img)(_n1##x,_n2##y,z,c),
//  I[60] = (T)(img)(_n2##x,_n2##y,z,c), I[61] = (T)(img)(_n3##x,_n2##y,z,c), I[62] = (T)(img)(_n4##x,_n2##y,z,c), I[63] = (T)(img)(_p4##x,_n3##y,z,c),
//  I[64] = (T)(img)(_p3##x,_n3##y,z,c), I[65] = (T)(img)(_p2##x,_n3##y,z,c), I[66] = (T)(img)(_p1##x,_n3##y,z,c), I[67] = (T)(img)(x,_n3##y,z,c),
//  I[68] = (T)(img)(_n1##x,_n3##y,z,c), I[69] = (T)(img)(_n2##x,_n3##y,z,c), I[70] = (T)(img)(_n3##x,_n3##y,z,c), I[71] = (T)(img)(_n4##x,_n3##y,z,c),
//  I[72] = (T)(img)(_p4##x,_n4##y,z,c), I[73] = (T)(img)(_p3##x,_n4##y,z,c), I[74] = (T)(img)(_p2##x,_n4##y,z,c), I[75] = (T)(img)(_p1##x,_n4##y,z,c),
//  I[76] = (T)(img)(x,_n4##y,z,c), I[77] = (T)(img)(_n1##x,_n4##y,z,c), I[78] = (T)(img)(_n2##x,_n4##y,z,c), I[79] = (T)(img)(_n3##x,_n4##y,z,c),
//  I[80] = (T)(img)(_n4##x,_n4##y,z,c)

// #define cimg_get2x2x2(img,x,y,z,c,I,T)
//   I[0] = (T)(img)(x,y,z,c), I[1] = (T)(img)(_n1##x,y,z,c), I[2] = (T)(img)(x,_n1##y,z,c), I[3] = (T)(img)(_n1##x,_n1##y,z,c),
//   I[4] = (T)(img)(x,y,_n1##z,c), I[5] = (T)(img)(_n1##x,y,_n1##z,c), I[6] = (T)(img)(x,_n1##y,_n1##z,c), I[7] = (T)(img)(_n1##x,_n1##y,_n1##z,c)

// #define cimg_get3x3x3(img,x,y,z,c,I,T)
//   I[0] = (T)(img)(_p1##x,_p1##y,_p1##z,c), I[1] = (T)(img)(x,_p1##y,_p1##z,c), I[2] = (T)(img)(_n1##x,_p1##y,_p1##z,c),
//   I[3] = (T)(img)(_p1##x,y,_p1##z,c), I[4] = (T)(img)(x,y,_p1##z,c), I[5] = (T)(img)(_n1##x,y,_p1##z,c),
//   I[6] = (T)(img)(_p1##x,_n1##y,_p1##z,c), I[7] = (T)(img)(x,_n1##y,_p1##z,c), I[8] = (T)(img)(_n1##x,_n1##y,_p1##z,c),
//   I[9] = (T)(img)(_p1##x,_p1##y,z,c), I[10] = (T)(img)(x,_p1##y,z,c), I[11] = (T)(img)(_n1##x,_p1##y,z,c),
//   I[12] = (T)(img)(_p1##x,y,z,c), I[13] = (T)(img)(x,y,z,c), I[14] = (T)(img)(_n1##x,y,z,c),
//   I[15] = (T)(img)(_p1##x,_n1##y,z,c), I[16] = (T)(img)(x,_n1##y,z,c), I[17] = (T)(img)(_n1##x,_n1##y,z,c),
//   I[18] = (T)(img)(_p1##x,_p1##y,_n1##z,c), I[19] = (T)(img)(x,_p1##y,_n1##z,c), I[20] = (T)(img)(_n1##x,_p1##y,_n1##z,c),
//   I[21] = (T)(img)(_p1##x,y,_n1##z,c), I[22] = (T)(img)(x,y,_n1##z,c), I[23] = (T)(img)(_n1##x,y,_n1##z,c),
//   I[24] = (T)(img)(_p1##x,_n1##y,_n1##z,c), I[25] = (T)(img)(x,_n1##y,_n1##z,c), I[26] = (T)(img)(_n1##x,_n1##y,_n1##z,c)

// Macros to perform various image loops.
//
// These macros are simpler to use than loops with C++ iterators.
// #define cimg_for(img,ptrs,T_ptrs) for (T_ptrs *ptrs = (img)._data, *_max##ptrs = (img)._data + (img).size(); ptrs<_max##ptrs; ++ptrs)
// #define cimg_rof(img,ptrs,T_ptrs) for (T_ptrs *ptrs = (img)._data + (img).size(); (ptrs--)>(img)._data; )
// #define cimg_foroff(img,off) for (unsigned long off = 0, _max##off = (img).size(); off<_max##off; ++off)

// #define cimg_for1(bound,i) for (int i = 0; i<(int)(bound); ++i)
// #define cimg_forX(img,x) cimg_for1((img)._width,x)
// #define cimg_forY(img,y) cimg_for1((img)._height,y)
// #define cimg_forZ(img,z) cimg_for1((img)._depth,z)
// #define cimg_forC(img,c) cimg_for1((img)._spectrum,c)
// #define cimg_forXY(img,x,y) cimg_forY(img,y) cimg_forX(img,x)
// #define cimg_forXZ(img,x,z) cimg_forZ(img,z) cimg_forX(img,x)
// #define cimg_forYZ(img,y,z) cimg_forZ(img,z) cimg_forY(img,y)
// #define cimg_forXC(img,x,c) cimg_forC(img,c) cimg_forX(img,x)
// #define cimg_forYC(img,y,c) cimg_forC(img,c) cimg_forY(img,y)
// #define cimg_forZC(img,z,c) cimg_forC(img,c) cimg_forZ(img,z)
// #define cimg_forXYZ(img,x,y,z) cimg_forZ(img,z) cimg_forXY(img,x,y)
// #define cimg_forXYC(img,x,y,c) cimg_forC(img,c) cimg_forXY(img,x,y)
// #define cimg_forXZC(img,x,z,c) cimg_forC(img,c) cimg_forXZ(img,x,z)
// #define cimg_forYZC(img,y,z,c) cimg_forC(img,c) cimg_forYZ(img,y,z)
// #define cimg_forXYZC(img,x,y,z,c) cimg_forC(img,c) cimg_forXYZ(img,x,y,z)

// #define cimg_rof1(bound,i) for (int i = (int)(bound)-1; i>=0; --i)
// #define cimg_rofX(img,x) cimg_rof1((img)._width,x)
// #define cimg_rofY(img,y) cimg_rof1((img)._height,y)
// #define cimg_rofZ(img,z) cimg_rof1((img)._depth,z)
// #define cimg_rofC(img,c) cimg_rof1((img)._spectrum,c)
// #define cimg_rofXY(img,x,y) cimg_rofY(img,y) cimg_rofX(img,x)
// #define cimg_rofXZ(img,x,z) cimg_rofZ(img,z) cimg_rofX(img,x)
// #define cimg_rofYZ(img,y,z) cimg_rofZ(img,z) cimg_rofY(img,y)
// #define cimg_rofXC(img,x,c) cimg_rofC(img,c) cimg_rofX(img,x)
// #define cimg_rofYC(img,y,c) cimg_rofC(img,c) cimg_rofY(img,y)
// #define cimg_rofZC(img,z,c) cimg_rofC(img,c) cimg_rofZ(img,z)
// #define cimg_rofXYZ(img,x,y,z) cimg_rofZ(img,z) cimg_rofXY(img,x,y)
// #define cimg_rofXYC(img,x,y,c) cimg_rofC(img,c) cimg_rofXY(img,x,y)
// #define cimg_rofXZC(img,x,z,c) cimg_rofC(img,c) cimg_rofXZ(img,x,z)
// #define cimg_rofYZC(img,y,z,c) cimg_rofC(img,c) cimg_rofYZ(img,y,z)
// #define cimg_rofXYZC(img,x,y,z,c) cimg_rofC(img,c) cimg_rofXYZ(img,x,y,z)

// #define cimg_for_in1(bound,i0,i1,i)
//  for (int i = (int)(i0)<0?0:(int)(i0), _max##i = (int)(i1)<(int)(bound)?(int)(i1):(int)(bound)-1; i<=_max##i; ++i)
// #define cimg_for_inX(img,x0,x1,x) cimg_for_in1((img)._width,x0,x1,x)
// #define cimg_for_inY(img,y0,y1,y) cimg_for_in1((img)._height,y0,y1,y)
// #define cimg_for_inZ(img,z0,z1,z) cimg_for_in1((img)._depth,z0,z1,z)
// #define cimg_for_inC(img,c0,c1,c) cimg_for_in1((img)._spectrum,c0,c1,c)
// #define cimg_for_inXY(img,x0,y0,x1,y1,x,y) cimg_for_inY(img,y0,y1,y) cimg_for_inX(img,x0,x1,x)
// #define cimg_for_inXZ(img,x0,z0,x1,z1,x,z) cimg_for_inZ(img,z0,z1,z) cimg_for_inX(img,x0,x1,x)
// #define cimg_for_inXC(img,x0,c0,x1,c1,x,c) cimg_for_inC(img,c0,c1,c) cimg_for_inX(img,x0,x1,x)
// #define cimg_for_inYZ(img,y0,z0,y1,z1,y,z) cimg_for_inZ(img,x0,z1,z) cimg_for_inY(img,y0,y1,y)
// #define cimg_for_inYC(img,y0,c0,y1,c1,y,c) cimg_for_inC(img,c0,c1,c) cimg_for_inY(img,y0,y1,y)
// #define cimg_for_inZC(img,z0,c0,z1,c1,z,c) cimg_for_inC(img,c0,c1,c) cimg_for_inZ(img,z0,z1,z)
// #define cimg_for_inXYZ(img,x0,y0,z0,x1,y1,z1,x,y,z) cimg_for_inZ(img,z0,z1,z) cimg_for_inXY(img,x0,y0,x1,y1,x,y)
// #define cimg_for_inXYC(img,x0,y0,c0,x1,y1,c1,x,y,c) cimg_for_inC(img,c0,c1,c) cimg_for_inXY(img,x0,y0,x1,y1,x,y)
// #define cimg_for_inXZC(img,x0,z0,c0,x1,z1,c1,x,z,c) cimg_for_inC(img,c0,c1,c) cimg_for_inXZ(img,x0,z0,x1,z1,x,z)
// #define cimg_for_inYZC(img,y0,z0,c0,y1,z1,c1,y,z,c) cimg_for_inC(img,c0,c1,c) cimg_for_inYZ(img,y0,z0,y1,z1,y,z)
// #define cimg_for_inXYZC(img,x0,y0,z0,c0,x1,y1,z1,c1,x,y,z,c) cimg_for_inC(img,c0,c1,c) cimg_for_inXYZ(img,x0,y0,z0,x1,y1,z1,x,y,z)
// #define cimg_for_insideX(img,x,n) cimg_for_inX(img,n,(img)._width-1-(n),x)
// #define cimg_for_insideY(img,y,n) cimg_for_inY(img,n,(img)._height-1-(n),y)
// #define cimg_for_insideZ(img,z,n) cimg_for_inZ(img,n,(img)._depth-1-(n),z)
// #define cimg_for_insideC(img,c,n) cimg_for_inC(img,n,(img)._spectrum-1-(n),c)
// #define cimg_for_insideXY(img,x,y,n) cimg_for_inXY(img,n,n,(img)._width-1-(n),(img)._height-1-(n),x,y)
// #define cimg_for_insideXYZ(img,x,y,z,n) cimg_for_inXYZ(img,n,n,n,(img)._width-1-(n),(img)._height-1-(n),(img)._depth-1-(n),x,y,z)
// #define cimg_for_insideXYZC(img,x,y,z,c,n) cimg_for_inXYZ(img,n,n,n,(img)._width-1-(n),(img)._height-1-(n),(img)._depth-1-(n),x,y,z)

// #define cimg_for_out1(boundi,i0,i1,i)
//  for (int i = (int)(i0)>0?0:(int)(i1)+1; i<(int)(boundi); ++i, i = i==(int)(i0)?(int)(i1)+1:i)
// #define cimg_for_out2(boundi,boundj,i0,j0,i1,j1,i,j)
//  for (int j = 0; j<(int)(boundj); ++j)
//  for (int _n1j = (int)(j<(int)(j0) || j>(int)(j1)), i = _n1j?0:(int)(i0)>0?0:(int)(i1)+1; i<(int)(boundi);
//   ++i, i = _n1j?i:(i==(int)(i0)?(int)(i1)+1:i))
// #define cimg_for_out3(boundi,boundj,boundk,i0,j0,k0,i1,j1,k1,i,j,k)
//  for (int k = 0; k<(int)(boundk); ++k)
//  for (int _n1k = (int)(k<(int)(k0) || k>(int)(k1)), j = 0; j<(int)(boundj); ++j)
//  for (int _n1j = (int)(j<(int)(j0) || j>(int)(j1)), i = _n1j || _n1k?0:(int)(i0)>0?0:(int)(i1)+1; i<(int)(boundi);
//   ++i, i = _n1j || _n1k?i:(i==(int)(i0)?(int)(i1)+1:i))
// #define cimg_for_out4(boundi,boundj,boundk,boundl,i0,j0,k0,l0,i1,j1,k1,l1,i,j,k,l)
//  for (int l = 0; l<(int)(boundl); ++L)
//  for (int _n1l = (int)(l<(int)(l0) || l>(int)(l1)), k = 0; k<(int)(boundk); ++k)
//  for (int _n1k = (int)(k<(int)(k0) || k>(int)(k1)), j = 0; j<(int)(boundj); ++j)
//  for (int _n1j = (int)(j<(int)(j0) || j>(int)(j1)), i = _n1j || _n1k || _n1l?0:(int)(i0)>0?0:(int)(i1)+1; i<(int)(boundi);
//   ++i, i = _n1j || _n1k || _n1l?i:(i==(int)(i0)?(int)(i1)+1:i))
// #define cimg_for_outX(img,x0,x1,x) cimg_for_out1((img)._width,x0,x1,x)
// #define cimg_for_outY(img,y0,y1,y) cimg_for_out1((img)._height,y0,y1,y)
// #define cimg_for_outZ(img,z0,z1,z) cimg_for_out1((img)._depth,z0,z1,z)
// #define cimg_for_outC(img,c0,c1,c) cimg_for_out1((img)._spectrum,c0,c1,c)
// #define cimg_for_outXY(img,x0,y0,x1,y1,x,y) cimg_for_out2((img)._width,(img)._height,x0,y0,x1,y1,x,y)
// #define cimg_for_outXZ(img,x0,z0,x1,z1,x,z) cimg_for_out2((img)._width,(img)._depth,x0,z0,x1,z1,x,z)
// #define cimg_for_outXC(img,x0,c0,x1,c1,x,c) cimg_for_out2((img)._width,(img)._spectrum,x0,c0,x1,c1,x,c)
// #define cimg_for_outYZ(img,y0,z0,y1,z1,y,z) cimg_for_out2((img)._height,(img)._depth,y0,z0,y1,z1,y,z)
// #define cimg_for_outYC(img,y0,c0,y1,c1,y,c) cimg_for_out2((img)._height,(img)._spectrum,y0,c0,y1,c1,y,c)
// #define cimg_for_outZC(img,z0,c0,z1,c1,z,c) cimg_for_out2((img)._depth,(img)._spectrum,z0,c0,z1,c1,z,c)
// #define cimg_for_outXYZ(img,x0,y0,z0,x1,y1,z1,x,y,z) cimg_for_out3((img)._width,(img)._height,(img)._depth,x0,y0,z0,x1,y1,z1,x,y,z)
// #define cimg_for_outXYC(img,x0,y0,c0,x1,y1,c1,x,y,c) cimg_for_out3((img)._width,(img)._height,(img)._spectrum,x0,y0,c0,x1,y1,c1,x,y,c)
// #define cimg_for_outXZC(img,x0,z0,c0,x1,z1,c1,x,z,c) cimg_for_out3((img)._width,(img)._depth,(img)._spectrum,x0,z0,c0,x1,z1,c1,x,z,c)
// #define cimg_for_outYZC(img,y0,z0,c0,y1,z1,c1,y,z,c) cimg_for_out3((img)._height,(img)._depth,(img)._spectrum,y0,z0,c0,y1,z1,c1,y,z,c)
// #define cimg_for_outXYZC(img,x0,y0,z0,c0,x1,y1,z1,c1,x,y,z,c)
//  cimg_for_out4((img)._width,(img)._height,(img)._depth,(img)._spectrum,x0,y0,z0,c0,x1,y1,z1,c1,x,y,z,c)
// #define cimg_for_borderX(img,x,n) cimg_for_outX(img,n,(img)._width-1-(n),x)
// #define cimg_for_borderY(img,y,n) cimg_for_outY(img,n,(img)._height-1-(n),y)
// #define cimg_for_borderZ(img,z,n) cimg_for_outZ(img,n,(img)._depth-1-(n),z)
// #define cimg_for_borderC(img,c,n) cimg_for_outC(img,n,(img)._spectrum-1-(n),c)
// #define cimg_for_borderXY(img,x,y,n) cimg_for_outXY(img,n,n,(img)._width-1-(n),(img)._height-1-(n),x,y)
// #define cimg_for_borderXYZ(img,x,y,z,n) cimg_for_outXYZ(img,n,n,n,(img)._width-1-(n),(img)._height-1-(n),(img)._depth-1-(n),x,y,z)
// #define cimg_for_borderXYZC(img,x,y,z,c,n)
//  cimg_for_outXYZC(img,n,n,n,n,(img)._width-1-(n),(img)._height-1-(n),(img)._depth-1-(n),(img)._spectrum-1-(n),x,y,z,c)

// #define cimg_for_spiralXY(img,x,y)
//  for (int x = 0, y = 0, _n1##x = 1, _n1##y = (img).width()*(img).height(); _n1##y;
//       --_n1##y, _n1##x+=(_n1##x>>2)-((!(_n1##x&3)?--y:((_n1##x&3)==1?(img)._width-1-++x:((_n1##x&3)==2?(img)._height-1-++y:--x))))?0:1)

// #define cimg_for_lineXY(x,y,x0,y0,x1,y1)
//  for (int x = (int)(x0), y = (int)(y0), _sx = 1, _sy = 1, _steep = 0,
//       _dx=(x1)>(x0)?(int)(x1)-(int)(x0):(_sx=-1,(int)(x0)-(int)(x1)),
//       _dy=(y1)>(y0)?(int)(y1)-(int)(y0):(_sy=-1,(int)(y0)-(int)(y1)),
//       _counter = _dx,
//       _err = _dx>_dy?(_dy>>1):((_steep=1),(_counter=_dy),(_dx>>1));
//       _counter>=0;
//       --_counter, x+=_steep?
//       (y+=_sy,(_err-=_dx)<0?_err+=_dy,_sx:0):
//       (y+=(_err-=_dy)<0?_err+=_dx,_sy:0,_sx))

// #define cimg_for2(bound,i)
//  for (int i = 0, _n1##i = 1>=(bound)?(int)(bound)-1:1;
//       _n1##i<(int)(bound) || i==--_n1##i;
//       ++i, ++_n1##i)
// #define cimg_for2X(img,x) cimg_for2((img)._width,x)
// #define cimg_for2Y(img,y) cimg_for2((img)._height,y)
// #define cimg_for2Z(img,z) cimg_for2((img)._depth,z)
// #define cimg_for2C(img,c) cimg_for2((img)._spectrum,c)
// #define cimg_for2XY(img,x,y) cimg_for2Y(img,y) cimg_for2X(img,x)
// #define cimg_for2XZ(img,x,z) cimg_for2Z(img,z) cimg_for2X(img,x)
// #define cimg_for2XC(img,x,c) cimg_for2C(img,c) cimg_for2X(img,x)
// #define cimg_for2YZ(img,y,z) cimg_for2Z(img,z) cimg_for2Y(img,y)
// #define cimg_for2YC(img,y,c) cimg_for2C(img,c) cimg_for2Y(img,y)
// #define cimg_for2ZC(img,z,c) cimg_for2C(img,c) cimg_for2Z(img,z)
// #define cimg_for2XYZ(img,x,y,z) cimg_for2Z(img,z) cimg_for2XY(img,x,y)
// #define cimg_for2XZC(img,x,z,c) cimg_for2C(img,c) cimg_for2XZ(img,x,z)
// #define cimg_for2YZC(img,y,z,c) cimg_for2C(img,c) cimg_for2YZ(img,y,z)
// #define cimg_for2XYZC(img,x,y,z,c) cimg_for2C(img,c) cimg_for2XYZ(img,x,y,z)

// #define cimg_for_in2(bound,i0,i1,i)
//  for (int i = (int)(i0)<0?0:(int)(i0),
//       _n1##i = i+1>=(int)(bound)?(int)(bound)-1:i+1;
//       i<=(int)(i1) && (_n1##i<(int)(bound) || i==--_n1##i);
//       ++i, ++_n1##i)
// #define cimg_for_in2X(img,x0,x1,x) cimg_for_in2((img)._width,x0,x1,x)
// #define cimg_for_in2Y(img,y0,y1,y) cimg_for_in2((img)._height,y0,y1,y)
// #define cimg_for_in2Z(img,z0,z1,z) cimg_for_in2((img)._depth,z0,z1,z)
// #define cimg_for_in2C(img,c0,c1,c) cimg_for_in2((img)._spectrum,c0,c1,c)
// #define cimg_for_in2XY(img,x0,y0,x1,y1,x,y) cimg_for_in2Y(img,y0,y1,y) cimg_for_in2X(img,x0,x1,x)
// #define cimg_for_in2XZ(img,x0,z0,x1,z1,x,z) cimg_for_in2Z(img,z0,z1,z) cimg_for_in2X(img,x0,x1,x)
// #define cimg_for_in2XC(img,x0,c0,x1,c1,x,c) cimg_for_in2C(img,c0,c1,c) cimg_for_in2X(img,x0,x1,x)
// #define cimg_for_in2YZ(img,y0,z0,y1,z1,y,z) cimg_for_in2Z(img,z0,z1,z) cimg_for_in2Y(img,y0,y1,y)
// #define cimg_for_in2YC(img,y0,c0,y1,c1,y,c) cimg_for_in2C(img,c0,c1,c) cimg_for_in2Y(img,y0,y1,y)
// #define cimg_for_in2ZC(img,z0,c0,z1,c1,z,c) cimg_for_in2C(img,c0,c1,c) cimg_for_in2Z(img,z0,z1,z)
// #define cimg_for_in2XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z) cimg_for_in2Z(img,z0,z1,z) cimg_for_in2XY(img,x0,y0,x1,y1,x,y)
// #define cimg_for_in2XZC(img,x0,z0,c0,x1,y1,c1,x,z,c) cimg_for_in2C(img,c0,c1,c) cimg_for_in2XZ(img,x0,y0,x1,y1,x,z)
// #define cimg_for_in2YZC(img,y0,z0,c0,y1,z1,c1,y,z,c) cimg_for_in2C(img,c0,c1,c) cimg_for_in2YZ(img,y0,z0,y1,z1,y,z)
// #define cimg_for_in2XYZC(img,x0,y0,z0,c0,x1,y1,z1,c1,x,y,z,c) cimg_for_in2C(img,c0,c1,c) cimg_for_in2XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z)

// #define cimg_for3(bound,i)
//  for (int i = 0, _p1##i = 0,
//       _n1##i = 1>=(bound)?(int)(bound)-1:1;
//       _n1##i<(int)(bound) || i==--_n1##i;
//       _p1##i = i++, ++_n1##i)
// #define cimg_for3X(img,x) cimg_for3((img)._width,x)
// #define cimg_for3Y(img,y) cimg_for3((img)._height,y)
// #define cimg_for3Z(img,z) cimg_for3((img)._depth,z)
// #define cimg_for3C(img,c) cimg_for3((img)._spectrum,c)
// #define cimg_for3XY(img,x,y) cimg_for3Y(img,y) cimg_for3X(img,x)
// #define cimg_for3XZ(img,x,z) cimg_for3Z(img,z) cimg_for3X(img,x)
// #define cimg_for3XC(img,x,c) cimg_for3C(img,c) cimg_for3X(img,x)
// #define cimg_for3YZ(img,y,z) cimg_for3Z(img,z) cimg_for3Y(img,y)
// #define cimg_for3YC(img,y,c) cimg_for3C(img,c) cimg_for3Y(img,y)
// #define cimg_for3ZC(img,z,c) cimg_for3C(img,c) cimg_for3Z(img,z)
// #define cimg_for3XYZ(img,x,y,z) cimg_for3Z(img,z) cimg_for3XY(img,x,y)
// #define cimg_for3XZC(img,x,z,c) cimg_for3C(img,c) cimg_for3XZ(img,x,z)
// #define cimg_for3YZC(img,y,z,c) cimg_for3C(img,c) cimg_for3YZ(img,y,z)
// #define cimg_for3XYZC(img,x,y,z,c) cimg_for3C(img,c) cimg_for3XYZ(img,x,y,z)

// #define cimg_for_in3(bound,i0,i1,i)
//  for (int i = (int)(i0)<0?0:(int)(i0),
//       _p1##i = i-1<0?0:i-1,
//       _n1##i = i+1>=(int)(bound)?(int)(bound)-1:i+1;
//       i<=(int)(i1) && (_n1##i<(int)(bound) || i==--_n1##i);
//       _p1##i = i++, ++_n1##i)
// #define cimg_for_in3X(img,x0,x1,x) cimg_for_in3((img)._width,x0,x1,x)
// #define cimg_for_in3Y(img,y0,y1,y) cimg_for_in3((img)._height,y0,y1,y)
// #define cimg_for_in3Z(img,z0,z1,z) cimg_for_in3((img)._depth,z0,z1,z)
// #define cimg_for_in3C(img,c0,c1,c) cimg_for_in3((img)._spectrum,c0,c1,c)
// #define cimg_for_in3XY(img,x0,y0,x1,y1,x,y) cimg_for_in3Y(img,y0,y1,y) cimg_for_in3X(img,x0,x1,x)
// #define cimg_for_in3XZ(img,x0,z0,x1,z1,x,z) cimg_for_in3Z(img,z0,z1,z) cimg_for_in3X(img,x0,x1,x)
// #define cimg_for_in3XC(img,x0,c0,x1,c1,x,c) cimg_for_in3C(img,c0,c1,c) cimg_for_in3X(img,x0,x1,x)
// #define cimg_for_in3YZ(img,y0,z0,y1,z1,y,z) cimg_for_in3Z(img,z0,z1,z) cimg_for_in3Y(img,y0,y1,y)
// #define cimg_for_in3YC(img,y0,c0,y1,c1,y,c) cimg_for_in3C(img,c0,c1,c) cimg_for_in3Y(img,y0,y1,y)
// #define cimg_for_in3ZC(img,z0,c0,z1,c1,z,c) cimg_for_in3C(img,c0,c1,c) cimg_for_in3Z(img,z0,z1,z)
// #define cimg_for_in3XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z) cimg_for_in3Z(img,z0,z1,z) cimg_for_in3XY(img,x0,y0,x1,y1,x,y)
// #define cimg_for_in3XZC(img,x0,z0,c0,x1,y1,c1,x,z,c) cimg_for_in3C(img,c0,c1,c) cimg_for_in3XZ(img,x0,y0,x1,y1,x,z)
// #define cimg_for_in3YZC(img,y0,z0,c0,y1,z1,c1,y,z,c) cimg_for_in3C(img,c0,c1,c) cimg_for_in3YZ(img,y0,z0,y1,z1,y,z)
// #define cimg_for_in3XYZC(img,x0,y0,z0,c0,x1,y1,z1,c1,x,y,z,c) cimg_for_in3C(img,c0,c1,c) cimg_for_in3XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z)

// #define cimg_for4(bound,i)
//  for (int i = 0, _p1##i = 0, _n1##i = 1>=(bound)?(int)(bound)-1:1,
//       _n2##i = 2>=(bound)?(int)(bound)-1:2;
//       _n2##i<(int)(bound) || _n1##i==--_n2##i || i==(_n2##i = --_n1##i);
//       _p1##i = i++, ++_n1##i, ++_n2##i)
// #define cimg_for4X(img,x) cimg_for4((img)._width,x)
// #define cimg_for4Y(img,y) cimg_for4((img)._height,y)
// #define cimg_for4Z(img,z) cimg_for4((img)._depth,z)
// #define cimg_for4C(img,c) cimg_for4((img)._spectrum,c)
// #define cimg_for4XY(img,x,y) cimg_for4Y(img,y) cimg_for4X(img,x)
// #define cimg_for4XZ(img,x,z) cimg_for4Z(img,z) cimg_for4X(img,x)
// #define cimg_for4XC(img,x,c) cimg_for4C(img,c) cimg_for4X(img,x)
// #define cimg_for4YZ(img,y,z) cimg_for4Z(img,z) cimg_for4Y(img,y)
// #define cimg_for4YC(img,y,c) cimg_for4C(img,c) cimg_for4Y(img,y)
// #define cimg_for4ZC(img,z,c) cimg_for4C(img,c) cimg_for4Z(img,z)
// #define cimg_for4XYZ(img,x,y,z) cimg_for4Z(img,z) cimg_for4XY(img,x,y)
// #define cimg_for4XZC(img,x,z,c) cimg_for4C(img,c) cimg_for4XZ(img,x,z)
// #define cimg_for4YZC(img,y,z,c) cimg_for4C(img,c) cimg_for4YZ(img,y,z)
// #define cimg_for4XYZC(img,x,y,z,c) cimg_for4C(img,c) cimg_for4XYZ(img,x,y,z)

// #define cimg_for_in4(bound,i0,i1,i)
//  for (int i = (int)(i0)<0?0:(int)(i0),
//       _p1##i = i-1<0?0:i-1,
//       _n1##i = i+1>=(int)(bound)?(int)(bound)-1:i+1,
//       _n2##i = i+2>=(int)(bound)?(int)(bound)-1:i+2;
//       i<=(int)(i1) && (_n2##i<(int)(bound) || _n1##i==--_n2##i || i==(_n2##i = --_n1##i));
//       _p1##i = i++, ++_n1##i, ++_n2##i)
// #define cimg_for_in4X(img,x0,x1,x) cimg_for_in4((img)._width,x0,x1,x)
// #define cimg_for_in4Y(img,y0,y1,y) cimg_for_in4((img)._height,y0,y1,y)
// #define cimg_for_in4Z(img,z0,z1,z) cimg_for_in4((img)._depth,z0,z1,z)
// #define cimg_for_in4C(img,c0,c1,c) cimg_for_in4((img)._spectrum,c0,c1,c)
// #define cimg_for_in4XY(img,x0,y0,x1,y1,x,y) cimg_for_in4Y(img,y0,y1,y) cimg_for_in4X(img,x0,x1,x)
// #define cimg_for_in4XZ(img,x0,z0,x1,z1,x,z) cimg_for_in4Z(img,z0,z1,z) cimg_for_in4X(img,x0,x1,x)
// #define cimg_for_in4XC(img,x0,c0,x1,c1,x,c) cimg_for_in4C(img,c0,c1,c) cimg_for_in4X(img,x0,x1,x)
// #define cimg_for_in4YZ(img,y0,z0,y1,z1,y,z) cimg_for_in4Z(img,z0,z1,z) cimg_for_in4Y(img,y0,y1,y)
// #define cimg_for_in4YC(img,y0,c0,y1,c1,y,c) cimg_for_in4C(img,c0,c1,c) cimg_for_in4Y(img,y0,y1,y)
// #define cimg_for_in4ZC(img,z0,c0,z1,c1,z,c) cimg_for_in4C(img,c0,c1,c) cimg_for_in4Z(img,z0,z1,z)
// #define cimg_for_in4XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z) cimg_for_in4Z(img,z0,z1,z) cimg_for_in4XY(img,x0,y0,x1,y1,x,y)
// #define cimg_for_in4XZC(img,x0,z0,c0,x1,y1,c1,x,z,c) cimg_for_in4C(img,c0,c1,c) cimg_for_in4XZ(img,x0,y0,x1,y1,x,z)
// #define cimg_for_in4YZC(img,y0,z0,c0,y1,z1,c1,y,z,c) cimg_for_in4C(img,c0,c1,c) cimg_for_in4YZ(img,y0,z0,y1,z1,y,z)
// #define cimg_for_in4XYZC(img,x0,y0,z0,c0,x1,y1,z1,c1,x,y,z,c) cimg_for_in4C(img,c0,c1,c) cimg_for_in4XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z)

// #define cimg_for5(bound,i)
//  for (int i = 0, _p2##i = 0, _p1##i = 0,
//       _n1##i = 1>=(bound)?(int)(bound)-1:1,
//       _n2##i = 2>=(bound)?(int)(bound)-1:2;
//       _n2##i<(int)(bound) || _n1##i==--_n2##i || i==(_n2##i = --_n1##i);
//       _p2##i = _p1##i, _p1##i = i++, ++_n1##i, ++_n2##i)
// #define cimg_for5X(img,x) cimg_for5((img)._width,x)
// #define cimg_for5Y(img,y) cimg_for5((img)._height,y)
// #define cimg_for5Z(img,z) cimg_for5((img)._depth,z)
// #define cimg_for5C(img,c) cimg_for5((img)._spectrum,c)
// #define cimg_for5XY(img,x,y) cimg_for5Y(img,y) cimg_for5X(img,x)
// #define cimg_for5XZ(img,x,z) cimg_for5Z(img,z) cimg_for5X(img,x)
// #define cimg_for5XC(img,x,c) cimg_for5C(img,c) cimg_for5X(img,x)
// #define cimg_for5YZ(img,y,z) cimg_for5Z(img,z) cimg_for5Y(img,y)
// #define cimg_for5YC(img,y,c) cimg_for5C(img,c) cimg_for5Y(img,y)
// #define cimg_for5ZC(img,z,c) cimg_for5C(img,c) cimg_for5Z(img,z)
// #define cimg_for5XYZ(img,x,y,z) cimg_for5Z(img,z) cimg_for5XY(img,x,y)
// #define cimg_for5XZC(img,x,z,c) cimg_for5C(img,c) cimg_for5XZ(img,x,z)
// #define cimg_for5YZC(img,y,z,c) cimg_for5C(img,c) cimg_for5YZ(img,y,z)
// #define cimg_for5XYZC(img,x,y,z,c) cimg_for5C(img,c) cimg_for5XYZ(img,x,y,z)

// #define cimg_for_in5(bound,i0,i1,i)
//  for (int i = (int)(i0)<0?0:(int)(i0),
//       _p2##i = i-2<0?0:i-2,
//       _p1##i = i-1<0?0:i-1,
//       _n1##i = i+1>=(int)(bound)?(int)(bound)-1:i+1,
//       _n2##i = i+2>=(int)(bound)?(int)(bound)-1:i+2;
//       i<=(int)(i1) && (_n2##i<(int)(bound) || _n1##i==--_n2##i || i==(_n2##i = --_n1##i));
//       _p2##i = _p1##i, _p1##i = i++, ++_n1##i, ++_n2##i)
// #define cimg_for_in5X(img,x0,x1,x) cimg_for_in5((img)._width,x0,x1,x)
// #define cimg_for_in5Y(img,y0,y1,y) cimg_for_in5((img)._height,y0,y1,y)
// #define cimg_for_in5Z(img,z0,z1,z) cimg_for_in5((img)._depth,z0,z1,z)
// #define cimg_for_in5C(img,c0,c1,c) cimg_for_in5((img)._spectrum,c0,c1,c)
// #define cimg_for_in5XY(img,x0,y0,x1,y1,x,y) cimg_for_in5Y(img,y0,y1,y) cimg_for_in5X(img,x0,x1,x)
// #define cimg_for_in5XZ(img,x0,z0,x1,z1,x,z) cimg_for_in5Z(img,z0,z1,z) cimg_for_in5X(img,x0,x1,x)
// #define cimg_for_in5XC(img,x0,c0,x1,c1,x,c) cimg_for_in5C(img,c0,c1,c) cimg_for_in5X(img,x0,x1,x)
// #define cimg_for_in5YZ(img,y0,z0,y1,z1,y,z) cimg_for_in5Z(img,z0,z1,z) cimg_for_in5Y(img,y0,y1,y)
// #define cimg_for_in5YC(img,y0,c0,y1,c1,y,c) cimg_for_in5C(img,c0,c1,c) cimg_for_in5Y(img,y0,y1,y)
// #define cimg_for_in5ZC(img,z0,c0,z1,c1,z,c) cimg_for_in5C(img,c0,c1,c) cimg_for_in5Z(img,z0,z1,z)
// #define cimg_for_in5XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z) cimg_for_in5Z(img,z0,z1,z) cimg_for_in5XY(img,x0,y0,x1,y1,x,y)
// #define cimg_for_in5XZC(img,x0,z0,c0,x1,y1,c1,x,z,c) cimg_for_in5C(img,c0,c1,c) cimg_for_in5XZ(img,x0,y0,x1,y1,x,z)
// #define cimg_for_in5YZC(img,y0,z0,c0,y1,z1,c1,y,z,c) cimg_for_in5C(img,c0,c1,c) cimg_for_in5YZ(img,y0,z0,y1,z1,y,z)
// #define cimg_for_in5XYZC(img,x0,y0,z0,c0,x1,y1,z1,c1,x,y,z,c) cimg_for_in5C(img,c0,c1,c) cimg_for_in5XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z)

// #define cimg_for6(bound,i)
//  for (int i = 0, _p2##i = 0, _p1##i = 0,
//       _n1##i = 1>=(bound)?(int)(bound)-1:1,
//       _n2##i = 2>=(bound)?(int)(bound)-1:2,
//       _n3##i = 3>=(bound)?(int)(bound)-1:3;
//       _n3##i<(int)(bound) || _n2##i==--_n3##i || _n1##i==--_n2##i || i==(_n3##i = _n2##i = --_n1##i);
//       _p2##i = _p1##i, _p1##i = i++, ++_n1##i, ++_n2##i, ++_n3##i)
// #define cimg_for6X(img,x) cimg_for6((img)._width,x)
// #define cimg_for6Y(img,y) cimg_for6((img)._height,y)
// #define cimg_for6Z(img,z) cimg_for6((img)._depth,z)
// #define cimg_for6C(img,c) cimg_for6((img)._spectrum,c)
// #define cimg_for6XY(img,x,y) cimg_for6Y(img,y) cimg_for6X(img,x)
// #define cimg_for6XZ(img,x,z) cimg_for6Z(img,z) cimg_for6X(img,x)
// #define cimg_for6XC(img,x,c) cimg_for6C(img,c) cimg_for6X(img,x)
// #define cimg_for6YZ(img,y,z) cimg_for6Z(img,z) cimg_for6Y(img,y)
// #define cimg_for6YC(img,y,c) cimg_for6C(img,c) cimg_for6Y(img,y)
// #define cimg_for6ZC(img,z,c) cimg_for6C(img,c) cimg_for6Z(img,z)
// #define cimg_for6XYZ(img,x,y,z) cimg_for6Z(img,z) cimg_for6XY(img,x,y)
// #define cimg_for6XZC(img,x,z,c) cimg_for6C(img,c) cimg_for6XZ(img,x,z)
// #define cimg_for6YZC(img,y,z,c) cimg_for6C(img,c) cimg_for6YZ(img,y,z)
// #define cimg_for6XYZC(img,x,y,z,c) cimg_for6C(img,c) cimg_for6XYZ(img,x,y,z)

// #define cimg_for_in6(bound,i0,i1,i)
//  for (int i = (int)(i0)<0?0:(int)(i0),
//       _p2##i = i-2<0?0:i-2,
//       _p1##i = i-1<0?0:i-1,
//       _n1##i = i+1>=(int)(bound)?(int)(bound)-1:i+1,
//       _n2##i = i+2>=(int)(bound)?(int)(bound)-1:i+2,
//       _n3##i = i+3>=(int)(bound)?(int)(bound)-1:i+3;
//       i<=(int)(i1) && (_n3##i<(int)(bound) || _n2##i==--_n3##i || _n1##i==--_n2##i || i==(_n3##i = _n2##i = --_n1##i));
//       _p2##i = _p1##i, _p1##i = i++, ++_n1##i, ++_n2##i, ++_n3##i)
// #define cimg_for_in6X(img,x0,x1,x) cimg_for_in6((img)._width,x0,x1,x)
// #define cimg_for_in6Y(img,y0,y1,y) cimg_for_in6((img)._height,y0,y1,y)
// #define cimg_for_in6Z(img,z0,z1,z) cimg_for_in6((img)._depth,z0,z1,z)
// #define cimg_for_in6C(img,c0,c1,c) cimg_for_in6((img)._spectrum,c0,c1,c)
// #define cimg_for_in6XY(img,x0,y0,x1,y1,x,y) cimg_for_in6Y(img,y0,y1,y) cimg_for_in6X(img,x0,x1,x)
// #define cimg_for_in6XZ(img,x0,z0,x1,z1,x,z) cimg_for_in6Z(img,z0,z1,z) cimg_for_in6X(img,x0,x1,x)
// #define cimg_for_in6XC(img,x0,c0,x1,c1,x,c) cimg_for_in6C(img,c0,c1,c) cimg_for_in6X(img,x0,x1,x)
// #define cimg_for_in6YZ(img,y0,z0,y1,z1,y,z) cimg_for_in6Z(img,z0,z1,z) cimg_for_in6Y(img,y0,y1,y)
// #define cimg_for_in6YC(img,y0,c0,y1,c1,y,c) cimg_for_in6C(img,c0,c1,c) cimg_for_in6Y(img,y0,y1,y)
// #define cimg_for_in6ZC(img,z0,c0,z1,c1,z,c) cimg_for_in6C(img,c0,c1,c) cimg_for_in6Z(img,z0,z1,z)
// #define cimg_for_in6XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z) cimg_for_in6Z(img,z0,z1,z) cimg_for_in6XY(img,x0,y0,x1,y1,x,y)
// #define cimg_for_in6XZC(img,x0,z0,c0,x1,y1,c1,x,z,c) cimg_for_in6C(img,c0,c1,c) cimg_for_in6XZ(img,x0,y0,x1,y1,x,z)
// #define cimg_for_in6YZC(img,y0,z0,c0,y1,z1,c1,y,z,c) cimg_for_in6C(img,c0,c1,c) cimg_for_in6YZ(img,y0,z0,y1,z1,y,z)
// #define cimg_for_in6XYZC(img,x0,y0,z0,c0,x1,y1,z1,c1,x,y,z,c) cimg_for_in6C(img,c0,c1,c) cimg_for_in6XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z)

// #define cimg_for7(bound,i)
//  for (int i = 0, _p3##i = 0, _p2##i = 0, _p1##i = 0,
//       _n1##i = 1>=(bound)?(int)(bound)-1:1,
//       _n2##i = 2>=(bound)?(int)(bound)-1:2,
//       _n3##i = 3>=(bound)?(int)(bound)-1:3;
//       _n3##i<(int)(bound) || _n2##i==--_n3##i || _n1##i==--_n2##i || i==(_n3##i = _n2##i = --_n1##i);
//       _p3##i = _p2##i, _p2##i = _p1##i, _p1##i = i++, ++_n1##i, ++_n2##i, ++_n3##i)
// #define cimg_for7X(img,x) cimg_for7((img)._width,x)
// #define cimg_for7Y(img,y) cimg_for7((img)._height,y)
// #define cimg_for7Z(img,z) cimg_for7((img)._depth,z)
// #define cimg_for7C(img,c) cimg_for7((img)._spectrum,c)
// #define cimg_for7XY(img,x,y) cimg_for7Y(img,y) cimg_for7X(img,x)
// #define cimg_for7XZ(img,x,z) cimg_for7Z(img,z) cimg_for7X(img,x)
// #define cimg_for7XC(img,x,c) cimg_for7C(img,c) cimg_for7X(img,x)
// #define cimg_for7YZ(img,y,z) cimg_for7Z(img,z) cimg_for7Y(img,y)
// #define cimg_for7YC(img,y,c) cimg_for7C(img,c) cimg_for7Y(img,y)
// #define cimg_for7ZC(img,z,c) cimg_for7C(img,c) cimg_for7Z(img,z)
// #define cimg_for7XYZ(img,x,y,z) cimg_for7Z(img,z) cimg_for7XY(img,x,y)
// #define cimg_for7XZC(img,x,z,c) cimg_for7C(img,c) cimg_for7XZ(img,x,z)
// #define cimg_for7YZC(img,y,z,c) cimg_for7C(img,c) cimg_for7YZ(img,y,z)
// #define cimg_for7XYZC(img,x,y,z,c) cimg_for7C(img,c) cimg_for7XYZ(img,x,y,z)

// #define cimg_for_in7(bound,i0,i1,i)
//  for (int i = (int)(i0)<0?0:(int)(i0),
//       _p3##i = i-3<0?0:i-3,
//       _p2##i = i-2<0?0:i-2,
//       _p1##i = i-1<0?0:i-1,
//       _n1##i = i+1>=(int)(bound)?(int)(bound)-1:i+1,
//       _n2##i = i+2>=(int)(bound)?(int)(bound)-1:i+2,
//       _n3##i = i+3>=(int)(bound)?(int)(bound)-1:i+3;
//       i<=(int)(i1) && (_n3##i<(int)(bound) || _n2##i==--_n3##i || _n1##i==--_n2##i || i==(_n3##i = _n2##i = --_n1##i));
//       _p3##i = _p2##i, _p2##i = _p1##i, _p1##i = i++, ++_n1##i, ++_n2##i, ++_n3##i)
// #define cimg_for_in7X(img,x0,x1,x) cimg_for_in7((img)._width,x0,x1,x)
// #define cimg_for_in7Y(img,y0,y1,y) cimg_for_in7((img)._height,y0,y1,y)
// #define cimg_for_in7Z(img,z0,z1,z) cimg_for_in7((img)._depth,z0,z1,z)
// #define cimg_for_in7C(img,c0,c1,c) cimg_for_in7((img)._spectrum,c0,c1,c)
// #define cimg_for_in7XY(img,x0,y0,x1,y1,x,y) cimg_for_in7Y(img,y0,y1,y) cimg_for_in7X(img,x0,x1,x)
// #define cimg_for_in7XZ(img,x0,z0,x1,z1,x,z) cimg_for_in7Z(img,z0,z1,z) cimg_for_in7X(img,x0,x1,x)
// #define cimg_for_in7XC(img,x0,c0,x1,c1,x,c) cimg_for_in7C(img,c0,c1,c) cimg_for_in7X(img,x0,x1,x)
// #define cimg_for_in7YZ(img,y0,z0,y1,z1,y,z) cimg_for_in7Z(img,z0,z1,z) cimg_for_in7Y(img,y0,y1,y)
// #define cimg_for_in7YC(img,y0,c0,y1,c1,y,c) cimg_for_in7C(img,c0,c1,c) cimg_for_in7Y(img,y0,y1,y)
// #define cimg_for_in7ZC(img,z0,c0,z1,c1,z,c) cimg_for_in7C(img,c0,c1,c) cimg_for_in7Z(img,z0,z1,z)
// #define cimg_for_in7XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z) cimg_for_in7Z(img,z0,z1,z) cimg_for_in7XY(img,x0,y0,x1,y1,x,y)
// #define cimg_for_in7XZC(img,x0,z0,c0,x1,y1,c1,x,z,c) cimg_for_in7C(img,c0,c1,c) cimg_for_in7XZ(img,x0,y0,x1,y1,x,z)
// #define cimg_for_in7YZC(img,y0,z0,c0,y1,z1,c1,y,z,c) cimg_for_in7C(img,c0,c1,c) cimg_for_in7YZ(img,y0,z0,y1,z1,y,z)
// #define cimg_for_in7XYZC(img,x0,y0,z0,c0,x1,y1,z1,c1,x,y,z,c) cimg_for_in7C(img,c0,c1,c) cimg_for_in7XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z)

// #define cimg_for8(bound,i)
//  for (int i = 0, _p3##i = 0, _p2##i = 0, _p1##i = 0,
//       _n1##i = 1>=(bound)?(int)(bound)-1:1,
//       _n2##i = 2>=(bound)?(int)(bound)-1:2,
//       _n3##i = 3>=(bound)?(int)(bound)-1:3,
//       _n4##i = 4>=(bound)?(int)(bound)-1:4;
//       _n4##i<(int)(bound) || _n3##i==--_n4##i || _n2##i==--_n3##i || _n1##i==--_n2##i ||
//       i==(_n4##i = _n3##i = _n2##i = --_n1##i);
//       _p3##i = _p2##i, _p2##i = _p1##i, _p1##i = i++, ++_n1##i, ++_n2##i, ++_n3##i, ++_n4##i)
// #define cimg_for8X(img,x) cimg_for8((img)._width,x)
// #define cimg_for8Y(img,y) cimg_for8((img)._height,y)
// #define cimg_for8Z(img,z) cimg_for8((img)._depth,z)
// #define cimg_for8C(img,c) cimg_for8((img)._spectrum,c)
// #define cimg_for8XY(img,x,y) cimg_for8Y(img,y) cimg_for8X(img,x)
// #define cimg_for8XZ(img,x,z) cimg_for8Z(img,z) cimg_for8X(img,x)
// #define cimg_for8XC(img,x,c) cimg_for8C(img,c) cimg_for8X(img,x)
// #define cimg_for8YZ(img,y,z) cimg_for8Z(img,z) cimg_for8Y(img,y)
// #define cimg_for8YC(img,y,c) cimg_for8C(img,c) cimg_for8Y(img,y)
// #define cimg_for8ZC(img,z,c) cimg_for8C(img,c) cimg_for8Z(img,z)
// #define cimg_for8XYZ(img,x,y,z) cimg_for8Z(img,z) cimg_for8XY(img,x,y)
// #define cimg_for8XZC(img,x,z,c) cimg_for8C(img,c) cimg_for8XZ(img,x,z)
// #define cimg_for8YZC(img,y,z,c) cimg_for8C(img,c) cimg_for8YZ(img,y,z)
// #define cimg_for8XYZC(img,x,y,z,c) cimg_for8C(img,c) cimg_for8XYZ(img,x,y,z)

// #define cimg_for_in8(bound,i0,i1,i)
//  for (int i = (int)(i0)<0?0:(int)(i0),
//       _p3##i = i-3<0?0:i-3,
//       _p2##i = i-2<0?0:i-2,
//       _p1##i = i-1<0?0:i-1,
//       _n1##i = i+1>=(int)(bound)?(int)(bound)-1:i+1,
//       _n2##i = i+2>=(int)(bound)?(int)(bound)-1:i+2,
//       _n3##i = i+3>=(int)(bound)?(int)(bound)-1:i+3,
//       _n4##i = i+4>=(int)(bound)?(int)(bound)-1:i+4;
//       i<=(int)(i1) && (_n4##i<(int)(bound) || _n3##i==--_n4##i || _n2##i==--_n3##i || _n1##i==--_n2##i ||
//       i==(_n4##i = _n3##i = _n2##i = --_n1##i));
//       _p3##i = _p2##i, _p2##i = _p1##i, _p1##i = i++, ++_n1##i, ++_n2##i, ++_n3##i, ++_n4##i)
// #define cimg_for_in8X(img,x0,x1,x) cimg_for_in8((img)._width,x0,x1,x)
// #define cimg_for_in8Y(img,y0,y1,y) cimg_for_in8((img)._height,y0,y1,y)
// #define cimg_for_in8Z(img,z0,z1,z) cimg_for_in8((img)._depth,z0,z1,z)
// #define cimg_for_in8C(img,c0,c1,c) cimg_for_in8((img)._spectrum,c0,c1,c)
// #define cimg_for_in8XY(img,x0,y0,x1,y1,x,y) cimg_for_in8Y(img,y0,y1,y) cimg_for_in8X(img,x0,x1,x)
// #define cimg_for_in8XZ(img,x0,z0,x1,z1,x,z) cimg_for_in8Z(img,z0,z1,z) cimg_for_in8X(img,x0,x1,x)
// #define cimg_for_in8XC(img,x0,c0,x1,c1,x,c) cimg_for_in8C(img,c0,c1,c) cimg_for_in8X(img,x0,x1,x)
// #define cimg_for_in8YZ(img,y0,z0,y1,z1,y,z) cimg_for_in8Z(img,z0,z1,z) cimg_for_in8Y(img,y0,y1,y)
// #define cimg_for_in8YC(img,y0,c0,y1,c1,y,c) cimg_for_in8C(img,c0,c1,c) cimg_for_in8Y(img,y0,y1,y)
// #define cimg_for_in8ZC(img,z0,c0,z1,c1,z,c) cimg_for_in8C(img,c0,c1,c) cimg_for_in8Z(img,z0,z1,z)
// #define cimg_for_in8XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z) cimg_for_in8Z(img,z0,z1,z) cimg_for_in8XY(img,x0,y0,x1,y1,x,y)
// #define cimg_for_in8XZC(img,x0,z0,c0,x1,y1,c1,x,z,c) cimg_for_in8C(img,c0,c1,c) cimg_for_in8XZ(img,x0,y0,x1,y1,x,z)
// #define cimg_for_in8YZC(img,y0,z0,c0,y1,z1,c1,y,z,c) cimg_for_in8C(img,c0,c1,c) cimg_for_in8YZ(img,y0,z0,y1,z1,y,z)
// #define cimg_for_in8XYZC(img,x0,y0,z0,c0,x1,y1,z1,c1,x,y,z,c) cimg_for_in8C(img,c0,c1,c) cimg_for_in8XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z)

// #define cimg_for9(bound,i)
//   for (int i = 0, _p4##i = 0, _p3##i = 0, _p2##i = 0, _p1##i = 0,
//        _n1##i = 1>=(int)(bound)?(int)(bound)-1:1,
//        _n2##i = 2>=(int)(bound)?(int)(bound)-1:2,
//        _n3##i = 3>=(int)(bound)?(int)(bound)-1:3,
//        _n4##i = 4>=(int)(bound)?(int)(bound)-1:4;
//        _n4##i<(int)(bound) || _n3##i==--_n4##i || _n2##i==--_n3##i || _n1##i==--_n2##i ||
//        i==(_n4##i = _n3##i = _n2##i = --_n1##i);
//        _p4##i = _p3##i, _p3##i = _p2##i, _p2##i = _p1##i, _p1##i = i++, ++_n1##i, ++_n2##i, ++_n3##i, ++_n4##i)
// #define cimg_for9X(img,x) cimg_for9((img)._width,x)
// #define cimg_for9Y(img,y) cimg_for9((img)._height,y)
// #define cimg_for9Z(img,z) cimg_for9((img)._depth,z)
// #define cimg_for9C(img,c) cimg_for9((img)._spectrum,c)
// #define cimg_for9XY(img,x,y) cimg_for9Y(img,y) cimg_for9X(img,x)
// #define cimg_for9XZ(img,x,z) cimg_for9Z(img,z) cimg_for9X(img,x)
// #define cimg_for9XC(img,x,c) cimg_for9C(img,c) cimg_for9X(img,x)
// #define cimg_for9YZ(img,y,z) cimg_for9Z(img,z) cimg_for9Y(img,y)
// #define cimg_for9YC(img,y,c) cimg_for9C(img,c) cimg_for9Y(img,y)
// #define cimg_for9ZC(img,z,c) cimg_for9C(img,c) cimg_for9Z(img,z)
// #define cimg_for9XYZ(img,x,y,z) cimg_for9Z(img,z) cimg_for9XY(img,x,y)
// #define cimg_for9XZC(img,x,z,c) cimg_for9C(img,c) cimg_for9XZ(img,x,z)
// #define cimg_for9YZC(img,y,z,c) cimg_for9C(img,c) cimg_for9YZ(img,y,z)
// #define cimg_for9XYZC(img,x,y,z,c) cimg_for9C(img,c) cimg_for9XYZ(img,x,y,z)

// #define cimg_for_in9(bound,i0,i1,i)
//   for (int i = (int)(i0)<0?0:(int)(i0),
//        _p4##i = i-4<0?0:i-4,
//        _p3##i = i-3<0?0:i-3,
//        _p2##i = i-2<0?0:i-2,
//        _p1##i = i-1<0?0:i-1,
//        _n1##i = i+1>=(int)(bound)?(int)(bound)-1:i+1,
//        _n2##i = i+2>=(int)(bound)?(int)(bound)-1:i+2,
//        _n3##i = i+3>=(int)(bound)?(int)(bound)-1:i+3,
//        _n4##i = i+4>=(int)(bound)?(int)(bound)-1:i+4;
//        i<=(int)(i1) && (_n4##i<(int)(bound) || _n3##i==--_n4##i || _n2##i==--_n3##i || _n1##i==--_n2##i ||
//        i==(_n4##i = _n3##i = _n2##i = --_n1##i));
//        _p4##i = _p3##i, _p3##i = _p2##i, _p2##i = _p1##i, _p1##i = i++, ++_n1##i, ++_n2##i, ++_n3##i, ++_n4##i)
// #define cimg_for_in9X(img,x0,x1,x) cimg_for_in9((img)._width,x0,x1,x)
// #define cimg_for_in9Y(img,y0,y1,y) cimg_for_in9((img)._height,y0,y1,y)
// #define cimg_for_in9Z(img,z0,z1,z) cimg_for_in9((img)._depth,z0,z1,z)
// #define cimg_for_in9C(img,c0,c1,c) cimg_for_in9((img)._spectrum,c0,c1,c)
// #define cimg_for_in9XY(img,x0,y0,x1,y1,x,y) cimg_for_in9Y(img,y0,y1,y) cimg_for_in9X(img,x0,x1,x)
// #define cimg_for_in9XZ(img,x0,z0,x1,z1,x,z) cimg_for_in9Z(img,z0,z1,z) cimg_for_in9X(img,x0,x1,x)
// #define cimg_for_in9XC(img,x0,c0,x1,c1,x,c) cimg_for_in9C(img,c0,c1,c) cimg_for_in9X(img,x0,x1,x)
// #define cimg_for_in9YZ(img,y0,z0,y1,z1,y,z) cimg_for_in9Z(img,z0,z1,z) cimg_for_in9Y(img,y0,y1,y)
// #define cimg_for_in9YC(img,y0,c0,y1,c1,y,c) cimg_for_in9C(img,c0,c1,c) cimg_for_in9Y(img,y0,y1,y)
// #define cimg_for_in9ZC(img,z0,c0,z1,c1,z,c) cimg_for_in9C(img,c0,c1,c) cimg_for_in9Z(img,z0,z1,z)
// #define cimg_for_in9XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z) cimg_for_in9Z(img,z0,z1,z) cimg_for_in9XY(img,x0,y0,x1,y1,x,y)
// #define cimg_for_in9XZC(img,x0,z0,c0,x1,y1,c1,x,z,c) cimg_for_in9C(img,c0,c1,c) cimg_for_in9XZ(img,x0,y0,x1,y1,x,z)
// #define cimg_for_in9YZC(img,y0,z0,c0,y1,z1,c1,y,z,c) cimg_for_in9C(img,c0,c1,c) cimg_for_in9YZ(img,y0,z0,y1,z1,y,z)
// #define cimg_for_in9XYZC(img,x0,y0,z0,c0,x1,y1,z1,c1,x,y,z,c) cimg_for_in9C(img,c0,c1,c) cimg_for_in9XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z)

// #define cimg_for2x2(img,x,y,z,c,I,T)
//   cimg_for2((img)._height,y) for (int x = 0,
//    _n1##x = (int)(
//    (I[0] = (T)(img)(0,y,z,c)),
//    (I[2] = (T)(img)(0,_n1##y,z,c)),
//    1>=(img)._width?(img).width()-1:1);
//    (_n1##x<(img).width() && (
//    (I[1] = (T)(img)(_n1##x,y,z,c)),
//    (I[3] = (T)(img)(_n1##x,_n1##y,z,c)),1)) ||
//    x==--_n1##x;
//    I[0] = I[1],
//    I[2] = I[3],
//    ++x, ++_n1##x)

// #define cimg_for_in2x2(img,x0,y0,x1,y1,x,y,z,c,I,T)
//   cimg_for_in2((img)._height,y0,y1,y) for (int x = (int)(x0)<0?0:(int)(x0),
//    _n1##x = (int)(
//    (I[0] = (T)(img)(x,y,z,c)),
//    (I[2] = (T)(img)(x,_n1##y,z,c)),
//    x+1>=(int)(img)._width?(img).width()-1:x+1);
//    x<=(int)(x1) && ((_n1##x<(img).width() && (
//    (I[1] = (T)(img)(_n1##x,y,z,c)),
//    (I[3] = (T)(img)(_n1##x,_n1##y,z,c)),1)) ||
//    x==--_n1##x);
//    I[0] = I[1],
//    I[2] = I[3],
//    ++x, ++_n1##x)

// #define cimg_for3x3(img,x,y,z,c,I,T)
//   cimg_for3((img)._height,y) for (int x = 0,
//    _p1##x = 0,
//    _n1##x = (int)(
//    (I[0] = I[1] = (T)(img)(_p1##x,_p1##y,z,c)),
//    (I[3] = I[4] = (T)(img)(0,y,z,c)),
//    (I[6] = I[7] = (T)(img)(0,_n1##y,z,c)),
//    1>=(img)._width?(img).width()-1:1);
//    (_n1##x<(img).width() && (
//    (I[2] = (T)(img)(_n1##x,_p1##y,z,c)),
//    (I[5] = (T)(img)(_n1##x,y,z,c)),
//    (I[8] = (T)(img)(_n1##x,_n1##y,z,c)),1)) ||
//    x==--_n1##x;
//    I[0] = I[1], I[1] = I[2],
//    I[3] = I[4], I[4] = I[5],
//    I[6] = I[7], I[7] = I[8],
//    _p1##x = x++, ++_n1##x)

// #define cimg_for_in3x3(img,x0,y0,x1,y1,x,y,z,c,I,T)
//   cimg_for_in3((img)._height,y0,y1,y) for (int x = (int)(x0)<0?0:(int)(x0),
//    _p1##x = x-1<0?0:x-1,
//    _n1##x = (int)(
//    (I[0] = (T)(img)(_p1##x,_p1##y,z,c)),
//    (I[3] = (T)(img)(_p1##x,y,z,c)),
//    (I[6] = (T)(img)(_p1##x,_n1##y,z,c)),
//    (I[1] = (T)(img)(x,_p1##y,z,c)),
//    (I[4] = (T)(img)(x,y,z,c)),
//    (I[7] = (T)(img)(x,_n1##y,z,c)),
//    x+1>=(int)(img)._width?(img).width()-1:x+1);
//    x<=(int)(x1) && ((_n1##x<(img).width() && (
//    (I[2] = (T)(img)(_n1##x,_p1##y,z,c)),
//    (I[5] = (T)(img)(_n1##x,y,z,c)),
//    (I[8] = (T)(img)(_n1##x,_n1##y,z,c)),1)) ||
//    x==--_n1##x);
//    I[0] = I[1], I[1] = I[2],
//    I[3] = I[4], I[4] = I[5],
//    I[6] = I[7], I[7] = I[8],
//    _p1##x = x++, ++_n1##x)

// #define cimg_for4x4(img,x,y,z,c,I,T)
//   cimg_for4((img)._height,y) for (int x = 0,
//    _p1##x = 0,
//    _n1##x = 1>=(img)._width?(img).width()-1:1,
//    _n2##x = (int)(
//    (I[0] = I[1] = (T)(img)(_p1##x,_p1##y,z,c)),
//    (I[4] = I[5] = (T)(img)(0,y,z,c)),
//    (I[8] = I[9] = (T)(img)(0,_n1##y,z,c)),
//    (I[12] = I[13] = (T)(img)(0,_n2##y,z,c)),
//    (I[2] = (T)(img)(_n1##x,_p1##y,z,c)),
//    (I[6] = (T)(img)(_n1##x,y,z,c)),
//    (I[10] = (T)(img)(_n1##x,_n1##y,z,c)),
//    (I[14] = (T)(img)(_n1##x,_n2##y,z,c)),
//    2>=(img)._width?(img).width()-1:2);
//    (_n2##x<(img).width() && (
//    (I[3] = (T)(img)(_n2##x,_p1##y,z,c)),
//    (I[7] = (T)(img)(_n2##x,y,z,c)),
//    (I[11] = (T)(img)(_n2##x,_n1##y,z,c)),
//    (I[15] = (T)(img)(_n2##x,_n2##y,z,c)),1)) ||
//    _n1##x==--_n2##x || x==(_n2##x = --_n1##x);
//    I[0] = I[1], I[1] = I[2], I[2] = I[3],
//    I[4] = I[5], I[5] = I[6], I[6] = I[7],
//    I[8] = I[9], I[9] = I[10], I[10] = I[11],
//    I[12] = I[13], I[13] = I[14], I[14] = I[15],
//    _p1##x = x++, ++_n1##x, ++_n2##x)

// #define cimg_for_in4x4(img,x0,y0,x1,y1,x,y,z,c,I,T)
//   cimg_for_in4((img)._height,y0,y1,y) for (int x = (int)(x0)<0?0:(int)(x0),
//    _p1##x = x-1<0?0:x-1,
//    _n1##x = x+1>=(int)(img)._width?(img).width()-1:x+1,
//    _n2##x = (int)(
//    (I[0] = (T)(img)(_p1##x,_p1##y,z,c)),
//    (I[4] = (T)(img)(_p1##x,y,z,c)),
//    (I[8] = (T)(img)(_p1##x,_n1##y,z,c)),
//    (I[12] = (T)(img)(_p1##x,_n2##y,z,c)),
//    (I[1] = (T)(img)(x,_p1##y,z,c)),
//    (I[5] = (T)(img)(x,y,z,c)),
//    (I[9] = (T)(img)(x,_n1##y,z,c)),
//    (I[13] = (T)(img)(x,_n2##y,z,c)),
//    (I[2] = (T)(img)(_n1##x,_p1##y,z,c)),
//    (I[6] = (T)(img)(_n1##x,y,z,c)),
//    (I[10] = (T)(img)(_n1##x,_n1##y,z,c)),
//    (I[14] = (T)(img)(_n1##x,_n2##y,z,c)),
//    x+2>=(int)(img)._width?(img).width()-1:x+2);
//    x<=(int)(x1) && ((_n2##x<(img).width() && (
//    (I[3] = (T)(img)(_n2##x,_p1##y,z,c)),
//    (I[7] = (T)(img)(_n2##x,y,z,c)),
//    (I[11] = (T)(img)(_n2##x,_n1##y,z,c)),
//    (I[15] = (T)(img)(_n2##x,_n2##y,z,c)),1)) ||
//    _n1##x==--_n2##x || x==(_n2##x = --_n1##x));
//    I[0] = I[1], I[1] = I[2], I[2] = I[3],
//    I[4] = I[5], I[5] = I[6], I[6] = I[7],
//    I[8] = I[9], I[9] = I[10], I[10] = I[11],
//    I[12] = I[13], I[13] = I[14], I[14] = I[15],
//    _p1##x = x++, ++_n1##x, ++_n2##x)

// #define cimg_for5x5(img,x,y,z,c,I,T)
//  cimg_for5((img)._height,y) for (int x = 0,
//    _p2##x = 0, _p1##x = 0,
//    _n1##x = 1>=(img)._width?(img).width()-1:1,
//    _n2##x = (int)(
//    (I[0] = I[1] = I[2] = (T)(img)(_p2##x,_p2##y,z,c)),
//    (I[5] = I[6] = I[7] = (T)(img)(0,_p1##y,z,c)),
//    (I[10] = I[11] = I[12] = (T)(img)(0,y,z,c)),
//    (I[15] = I[16] = I[17] = (T)(img)(0,_n1##y,z,c)),
//    (I[20] = I[21] = I[22] = (T)(img)(0,_n2##y,z,c)),
//    (I[3] = (T)(img)(_n1##x,_p2##y,z,c)),
//    (I[8] = (T)(img)(_n1##x,_p1##y,z,c)),
//    (I[13] = (T)(img)(_n1##x,y,z,c)),
//    (I[18] = (T)(img)(_n1##x,_n1##y,z,c)),
//    (I[23] = (T)(img)(_n1##x,_n2##y,z,c)),
//    2>=(img)._width?(img).width()-1:2);
//    (_n2##x<(img).width() && (
//    (I[4] = (T)(img)(_n2##x,_p2##y,z,c)),
//    (I[9] = (T)(img)(_n2##x,_p1##y,z,c)),
//    (I[14] = (T)(img)(_n2##x,y,z,c)),
//    (I[19] = (T)(img)(_n2##x,_n1##y,z,c)),
//    (I[24] = (T)(img)(_n2##x,_n2##y,z,c)),1)) ||
//    _n1##x==--_n2##x || x==(_n2##x = --_n1##x);
//    I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4],
//    I[5] = I[6], I[6] = I[7], I[7] = I[8], I[8] = I[9],
//    I[10] = I[11], I[11] = I[12], I[12] = I[13], I[13] = I[14],
//    I[15] = I[16], I[16] = I[17], I[17] = I[18], I[18] = I[19],
//    I[20] = I[21], I[21] = I[22], I[22] = I[23], I[23] = I[24],
//    _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x)

// #define cimg_for_in5x5(img,x0,y0,x1,y1,x,y,z,c,I,T)
//  cimg_for_in5((img)._height,y0,y1,y) for (int x = (int)(x0)<0?0:(int)(x0),
//    _p2##x = x-2<0?0:x-2,
//    _p1##x = x-1<0?0:x-1,
//    _n1##x = x+1>=(int)(img)._width?(img).width()-1:x+1,
//    _n2##x = (int)(
//    (I[0] = (T)(img)(_p2##x,_p2##y,z,c)),
//    (I[5] = (T)(img)(_p2##x,_p1##y,z,c)),
//    (I[10] = (T)(img)(_p2##x,y,z,c)),
//    (I[15] = (T)(img)(_p2##x,_n1##y,z,c)),
//    (I[20] = (T)(img)(_p2##x,_n2##y,z,c)),
//    (I[1] = (T)(img)(_p1##x,_p2##y,z,c)),
//    (I[6] = (T)(img)(_p1##x,_p1##y,z,c)),
//    (I[11] = (T)(img)(_p1##x,y,z,c)),
//    (I[16] = (T)(img)(_p1##x,_n1##y,z,c)),
//    (I[21] = (T)(img)(_p1##x,_n2##y,z,c)),
//    (I[2] = (T)(img)(x,_p2##y,z,c)),
//    (I[7] = (T)(img)(x,_p1##y,z,c)),
//    (I[12] = (T)(img)(x,y,z,c)),
//    (I[17] = (T)(img)(x,_n1##y,z,c)),
//    (I[22] = (T)(img)(x,_n2##y,z,c)),
//    (I[3] = (T)(img)(_n1##x,_p2##y,z,c)),
//    (I[8] = (T)(img)(_n1##x,_p1##y,z,c)),
//    (I[13] = (T)(img)(_n1##x,y,z,c)),
//    (I[18] = (T)(img)(_n1##x,_n1##y,z,c)),
//    (I[23] = (T)(img)(_n1##x,_n2##y,z,c)),
//    x+2>=(int)(img)._width?(img).width()-1:x+2);
//    x<=(int)(x1) && ((_n2##x<(img).width() && (
//    (I[4] = (T)(img)(_n2##x,_p2##y,z,c)),
//    (I[9] = (T)(img)(_n2##x,_p1##y,z,c)),
//    (I[14] = (T)(img)(_n2##x,y,z,c)),
//    (I[19] = (T)(img)(_n2##x,_n1##y,z,c)),
//    (I[24] = (T)(img)(_n2##x,_n2##y,z,c)),1)) ||
//    _n1##x==--_n2##x || x==(_n2##x = --_n1##x));
//    I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4],
//    I[5] = I[6], I[6] = I[7], I[7] = I[8], I[8] = I[9],
//    I[10] = I[11], I[11] = I[12], I[12] = I[13], I[13] = I[14],
//    I[15] = I[16], I[16] = I[17], I[17] = I[18], I[18] = I[19],
//    I[20] = I[21], I[21] = I[22], I[22] = I[23], I[23] = I[24],
//    _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x)

// #define cimg_for6x6(img,x,y,z,c,I,T)
//  cimg_for6((img)._height,y) for (int x = 0,
//    _p2##x = 0, _p1##x = 0,
//    _n1##x = 1>=(img)._width?(img).width()-1:1,
//    _n2##x = 2>=(img)._width?(img).width()-1:2,
//    _n3##x = (int)(
//    (I[0] = I[1] = I[2] = (T)(img)(_p2##x,_p2##y,z,c)),
//    (I[6] = I[7] = I[8] = (T)(img)(0,_p1##y,z,c)),
//    (I[12] = I[13] = I[14] = (T)(img)(0,y,z,c)),
//    (I[18] = I[19] = I[20] = (T)(img)(0,_n1##y,z,c)),
//    (I[24] = I[25] = I[26] = (T)(img)(0,_n2##y,z,c)),
//    (I[30] = I[31] = I[32] = (T)(img)(0,_n3##y,z,c)),
//    (I[3] = (T)(img)(_n1##x,_p2##y,z,c)),
//    (I[9] = (T)(img)(_n1##x,_p1##y,z,c)),
//    (I[15] = (T)(img)(_n1##x,y,z,c)),
//    (I[21] = (T)(img)(_n1##x,_n1##y,z,c)),
//    (I[27] = (T)(img)(_n1##x,_n2##y,z,c)),
//    (I[33] = (T)(img)(_n1##x,_n3##y,z,c)),
//    (I[4] = (T)(img)(_n2##x,_p2##y,z,c)),
//    (I[10] = (T)(img)(_n2##x,_p1##y,z,c)),
//    (I[16] = (T)(img)(_n2##x,y,z,c)),
//    (I[22] = (T)(img)(_n2##x,_n1##y,z,c)),
//    (I[28] = (T)(img)(_n2##x,_n2##y,z,c)),
//    (I[34] = (T)(img)(_n2##x,_n3##y,z,c)),
//    3>=(img)._width?(img).width()-1:3);
//    (_n3##x<(img).width() && (
//    (I[5] = (T)(img)(_n3##x,_p2##y,z,c)),
//    (I[11] = (T)(img)(_n3##x,_p1##y,z,c)),
//    (I[17] = (T)(img)(_n3##x,y,z,c)),
//    (I[23] = (T)(img)(_n3##x,_n1##y,z,c)),
//    (I[29] = (T)(img)(_n3##x,_n2##y,z,c)),
//    (I[35] = (T)(img)(_n3##x,_n3##y,z,c)),1)) ||
//    _n2##x==--_n3##x || _n1##x==--_n2##x || x==(_n3## x = _n2##x = --_n1##x);
//    I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4], I[4] = I[5],
//    I[6] = I[7], I[7] = I[8], I[8] = I[9], I[9] = I[10], I[10] = I[11],
//    I[12] = I[13], I[13] = I[14], I[14] = I[15], I[15] = I[16], I[16] = I[17],
//    I[18] = I[19], I[19] = I[20], I[20] = I[21], I[21] = I[22], I[22] = I[23],
//    I[24] = I[25], I[25] = I[26], I[26] = I[27], I[27] = I[28], I[28] = I[29],
//    I[30] = I[31], I[31] = I[32], I[32] = I[33], I[33] = I[34], I[34] = I[35],
//    _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x, ++_n3##x)

// #define cimg_for_in6x6(img,x0,y0,x1,y1,x,y,z,c,I,T)
//   cimg_for_in6((img)._height,y0,y1,y) for (int x = (int)(x0)<0?0:(int)x0,
//    _p2##x = x-2<0?0:x-2,
//    _p1##x = x-1<0?0:x-1,
//    _n1##x = x+1>=(int)(img)._width?(img).width()-1:x+1,
//    _n2##x = x+2>=(int)(img)._width?(img).width()-1:x+2,
//    _n3##x = (int)(
//    (I[0] = (T)(img)(_p2##x,_p2##y,z,c)),
//    (I[6] = (T)(img)(_p2##x,_p1##y,z,c)),
//    (I[12] = (T)(img)(_p2##x,y,z,c)),
//    (I[18] = (T)(img)(_p2##x,_n1##y,z,c)),
//    (I[24] = (T)(img)(_p2##x,_n2##y,z,c)),
//    (I[30] = (T)(img)(_p2##x,_n3##y,z,c)),
//    (I[1] = (T)(img)(_p1##x,_p2##y,z,c)),
//    (I[7] = (T)(img)(_p1##x,_p1##y,z,c)),
//    (I[13] = (T)(img)(_p1##x,y,z,c)),
//    (I[19] = (T)(img)(_p1##x,_n1##y,z,c)),
//    (I[25] = (T)(img)(_p1##x,_n2##y,z,c)),
//    (I[31] = (T)(img)(_p1##x,_n3##y,z,c)),
//    (I[2] = (T)(img)(x,_p2##y,z,c)),
//    (I[8] = (T)(img)(x,_p1##y,z,c)),
//    (I[14] = (T)(img)(x,y,z,c)),
//    (I[20] = (T)(img)(x,_n1##y,z,c)),
//    (I[26] = (T)(img)(x,_n2##y,z,c)),
//    (I[32] = (T)(img)(x,_n3##y,z,c)),
//    (I[3] = (T)(img)(_n1##x,_p2##y,z,c)),
//    (I[9] = (T)(img)(_n1##x,_p1##y,z,c)),
//    (I[15] = (T)(img)(_n1##x,y,z,c)),
//    (I[21] = (T)(img)(_n1##x,_n1##y,z,c)),
//    (I[27] = (T)(img)(_n1##x,_n2##y,z,c)),
//    (I[33] = (T)(img)(_n1##x,_n3##y,z,c)),
//    (I[4] = (T)(img)(_n2##x,_p2##y,z,c)),
//    (I[10] = (T)(img)(_n2##x,_p1##y,z,c)),
//    (I[16] = (T)(img)(_n2##x,y,z,c)),
//    (I[22] = (T)(img)(_n2##x,_n1##y,z,c)),
//    (I[28] = (T)(img)(_n2##x,_n2##y,z,c)),
//    (I[34] = (T)(img)(_n2##x,_n3##y,z,c)),
//    x+3>=(int)(img)._width?(img).width()-1:x+3);
//    x<=(int)(x1) && ((_n3##x<(img).width() && (
//    (I[5] = (T)(img)(_n3##x,_p2##y,z,c)),
//    (I[11] = (T)(img)(_n3##x,_p1##y,z,c)),
//    (I[17] = (T)(img)(_n3##x,y,z,c)),
//    (I[23] = (T)(img)(_n3##x,_n1##y,z,c)),
//    (I[29] = (T)(img)(_n3##x,_n2##y,z,c)),
//    (I[35] = (T)(img)(_n3##x,_n3##y,z,c)),1)) ||
//    _n2##x==--_n3##x || _n1##x==--_n2##x || x==(_n3## x = _n2##x = --_n1##x));
//    I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4], I[4] = I[5],
//    I[6] = I[7], I[7] = I[8], I[8] = I[9], I[9] = I[10], I[10] = I[11],
//    I[12] = I[13], I[13] = I[14], I[14] = I[15], I[15] = I[16], I[16] = I[17],
//    I[18] = I[19], I[19] = I[20], I[20] = I[21], I[21] = I[22], I[22] = I[23],
//    I[24] = I[25], I[25] = I[26], I[26] = I[27], I[27] = I[28], I[28] = I[29],
//    I[30] = I[31], I[31] = I[32], I[32] = I[33], I[33] = I[34], I[34] = I[35],
//    _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x, ++_n3##x)

// #define cimg_for7x7(img,x,y,z,c,I,T)
//   cimg_for7((img)._height,y) for (int x = 0,
//    _p3##x = 0, _p2##x = 0, _p1##x = 0,
//    _n1##x = 1>=(img)._width?(img).width()-1:1,
//    _n2##x = 2>=(img)._width?(img).width()-1:2,
//    _n3##x = (int)(
//    (I[0] = I[1] = I[2] = I[3] = (T)(img)(_p3##x,_p3##y,z,c)),
//    (I[7] = I[8] = I[9] = I[10] = (T)(img)(0,_p2##y,z,c)),
//    (I[14] = I[15] = I[16] = I[17] = (T)(img)(0,_p1##y,z,c)),
//    (I[21] = I[22] = I[23] = I[24] = (T)(img)(0,y,z,c)),
//    (I[28] = I[29] = I[30] = I[31] = (T)(img)(0,_n1##y,z,c)),
//    (I[35] = I[36] = I[37] = I[38] = (T)(img)(0,_n2##y,z,c)),
//    (I[42] = I[43] = I[44] = I[45] = (T)(img)(0,_n3##y,z,c)),
//    (I[4] = (T)(img)(_n1##x,_p3##y,z,c)),
//    (I[11] = (T)(img)(_n1##x,_p2##y,z,c)),
//    (I[18] = (T)(img)(_n1##x,_p1##y,z,c)),
//    (I[25] = (T)(img)(_n1##x,y,z,c)),
//    (I[32] = (T)(img)(_n1##x,_n1##y,z,c)),
//    (I[39] = (T)(img)(_n1##x,_n2##y,z,c)),
//    (I[46] = (T)(img)(_n1##x,_n3##y,z,c)),
//    (I[5] = (T)(img)(_n2##x,_p3##y,z,c)),
//    (I[12] = (T)(img)(_n2##x,_p2##y,z,c)),
//    (I[19] = (T)(img)(_n2##x,_p1##y,z,c)),
//    (I[26] = (T)(img)(_n2##x,y,z,c)),
//    (I[33] = (T)(img)(_n2##x,_n1##y,z,c)),
//    (I[40] = (T)(img)(_n2##x,_n2##y,z,c)),
//    (I[47] = (T)(img)(_n2##x,_n3##y,z,c)),
//    3>=(img)._width?(img).width()-1:3);
//    (_n3##x<(img).width() && (
//    (I[6] = (T)(img)(_n3##x,_p3##y,z,c)),
//    (I[13] = (T)(img)(_n3##x,_p2##y,z,c)),
//    (I[20] = (T)(img)(_n3##x,_p1##y,z,c)),
//    (I[27] = (T)(img)(_n3##x,y,z,c)),
//    (I[34] = (T)(img)(_n3##x,_n1##y,z,c)),
//    (I[41] = (T)(img)(_n3##x,_n2##y,z,c)),
//    (I[48] = (T)(img)(_n3##x,_n3##y,z,c)),1)) ||
//    _n2##x==--_n3##x || _n1##x==--_n2##x || x==(_n3##x = _n2##x = --_n1##x);
//    I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4], I[4] = I[5], I[5] = I[6],
//    I[7] = I[8], I[8] = I[9], I[9] = I[10], I[10] = I[11], I[11] = I[12], I[12] = I[13],
//    I[14] = I[15], I[15] = I[16], I[16] = I[17], I[17] = I[18], I[18] = I[19], I[19] = I[20],
//    I[21] = I[22], I[22] = I[23], I[23] = I[24], I[24] = I[25], I[25] = I[26], I[26] = I[27],
//    I[28] = I[29], I[29] = I[30], I[30] = I[31], I[31] = I[32], I[32] = I[33], I[33] = I[34],
//    I[35] = I[36], I[36] = I[37], I[37] = I[38], I[38] = I[39], I[39] = I[40], I[40] = I[41],
//    I[42] = I[43], I[43] = I[44], I[44] = I[45], I[45] = I[46], I[46] = I[47], I[47] = I[48],
//    _p3##x = _p2##x, _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x, ++_n3##x)

// #define cimg_for_in7x7(img,x0,y0,x1,y1,x,y,z,c,I,T)
//   cimg_for_in7((img)._height,y0,y1,y) for (int x = (int)(x0)<0?0:(int)(x0),
//    _p3##x = x-3<0?0:x-3,
//    _p2##x = x-2<0?0:x-2,
//    _p1##x = x-1<0?0:x-1,
//    _n1##x = x+1>=(int)(img)._width?(img).width()-1:x+1,
//    _n2##x = x+2>=(int)(img)._width?(img).width()-1:x+2,
//    _n3##x = (int)(
//    (I[0] = (T)(img)(_p3##x,_p3##y,z,c)),
//    (I[7] = (T)(img)(_p3##x,_p2##y,z,c)),
//    (I[14] = (T)(img)(_p3##x,_p1##y,z,c)),
//    (I[21] = (T)(img)(_p3##x,y,z,c)),
//    (I[28] = (T)(img)(_p3##x,_n1##y,z,c)),
//    (I[35] = (T)(img)(_p3##x,_n2##y,z,c)),
//    (I[42] = (T)(img)(_p3##x,_n3##y,z,c)),
//    (I[1] = (T)(img)(_p2##x,_p3##y,z,c)),
//    (I[8] = (T)(img)(_p2##x,_p2##y,z,c)),
//    (I[15] = (T)(img)(_p2##x,_p1##y,z,c)),
//    (I[22] = (T)(img)(_p2##x,y,z,c)),
//    (I[29] = (T)(img)(_p2##x,_n1##y,z,c)),
//    (I[36] = (T)(img)(_p2##x,_n2##y,z,c)),
//    (I[43] = (T)(img)(_p2##x,_n3##y,z,c)),
//    (I[2] = (T)(img)(_p1##x,_p3##y,z,c)),
//    (I[9] = (T)(img)(_p1##x,_p2##y,z,c)),
//    (I[16] = (T)(img)(_p1##x,_p1##y,z,c)),
//    (I[23] = (T)(img)(_p1##x,y,z,c)),
//    (I[30] = (T)(img)(_p1##x,_n1##y,z,c)),
//    (I[37] = (T)(img)(_p1##x,_n2##y,z,c)),
//    (I[44] = (T)(img)(_p1##x,_n3##y,z,c)),
//    (I[3] = (T)(img)(x,_p3##y,z,c)),
//    (I[10] = (T)(img)(x,_p2##y,z,c)),
//    (I[17] = (T)(img)(x,_p1##y,z,c)),
//    (I[24] = (T)(img)(x,y,z,c)),
//    (I[31] = (T)(img)(x,_n1##y,z,c)),
//    (I[38] = (T)(img)(x,_n2##y,z,c)),
//    (I[45] = (T)(img)(x,_n3##y,z,c)),
//    (I[4] = (T)(img)(_n1##x,_p3##y,z,c)),
//    (I[11] = (T)(img)(_n1##x,_p2##y,z,c)),
//    (I[18] = (T)(img)(_n1##x,_p1##y,z,c)),
//    (I[25] = (T)(img)(_n1##x,y,z,c)),
//    (I[32] = (T)(img)(_n1##x,_n1##y,z,c)),
//    (I[39] = (T)(img)(_n1##x,_n2##y,z,c)),
//    (I[46] = (T)(img)(_n1##x,_n3##y,z,c)),
//    (I[5] = (T)(img)(_n2##x,_p3##y,z,c)),
//    (I[12] = (T)(img)(_n2##x,_p2##y,z,c)),
//    (I[19] = (T)(img)(_n2##x,_p1##y,z,c)),
//    (I[26] = (T)(img)(_n2##x,y,z,c)),
//    (I[33] = (T)(img)(_n2##x,_n1##y,z,c)),
//    (I[40] = (T)(img)(_n2##x,_n2##y,z,c)),
//    (I[47] = (T)(img)(_n2##x,_n3##y,z,c)),
//    x+3>=(int)(img)._width?(img).width()-1:x+3);
//    x<=(int)(x1) && ((_n3##x<(img).width() && (
//    (I[6] = (T)(img)(_n3##x,_p3##y,z,c)),
//    (I[13] = (T)(img)(_n3##x,_p2##y,z,c)),
//    (I[20] = (T)(img)(_n3##x,_p1##y,z,c)),
//    (I[27] = (T)(img)(_n3##x,y,z,c)),
//    (I[34] = (T)(img)(_n3##x,_n1##y,z,c)),
//    (I[41] = (T)(img)(_n3##x,_n2##y,z,c)),
//    (I[48] = (T)(img)(_n3##x,_n3##y,z,c)),1)) ||
//    _n2##x==--_n3##x || _n1##x==--_n2##x || x==(_n3##x = _n2##x = --_n1##x));
//    I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4], I[4] = I[5], I[5] = I[6],
//    I[7] = I[8], I[8] = I[9], I[9] = I[10], I[10] = I[11], I[11] = I[12], I[12] = I[13],
//    I[14] = I[15], I[15] = I[16], I[16] = I[17], I[17] = I[18], I[18] = I[19], I[19] = I[20],
//    I[21] = I[22], I[22] = I[23], I[23] = I[24], I[24] = I[25], I[25] = I[26], I[26] = I[27],
//    I[28] = I[29], I[29] = I[30], I[30] = I[31], I[31] = I[32], I[32] = I[33], I[33] = I[34],
//    I[35] = I[36], I[36] = I[37], I[37] = I[38], I[38] = I[39], I[39] = I[40], I[40] = I[41],
//    I[42] = I[43], I[43] = I[44], I[44] = I[45], I[45] = I[46], I[46] = I[47], I[47] = I[48],
//    _p3##x = _p2##x, _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x, ++_n3##x)

// #define cimg_for8x8(img,x,y,z,c,I,T)
//   cimg_for8((img)._height,y) for (int x = 0,
//    _p3##x = 0, _p2##x = 0, _p1##x = 0,
//    _n1##x = 1>=((img)._width)?(img).width()-1:1,
//    _n2##x = 2>=((img)._width)?(img).width()-1:2,
//    _n3##x = 3>=((img)._width)?(img).width()-1:3,
//    _n4##x = (int)(
//    (I[0] = I[1] = I[2] = I[3] = (T)(img)(_p3##x,_p3##y,z,c)),
//    (I[8] = I[9] = I[10] = I[11] = (T)(img)(0,_p2##y,z,c)),
//    (I[16] = I[17] = I[18] = I[19] = (T)(img)(0,_p1##y,z,c)),
//    (I[24] = I[25] = I[26] = I[27] = (T)(img)(0,y,z,c)),
//    (I[32] = I[33] = I[34] = I[35] = (T)(img)(0,_n1##y,z,c)),
//    (I[40] = I[41] = I[42] = I[43] = (T)(img)(0,_n2##y,z,c)),
//    (I[48] = I[49] = I[50] = I[51] = (T)(img)(0,_n3##y,z,c)),
//    (I[56] = I[57] = I[58] = I[59] = (T)(img)(0,_n4##y,z,c)),
//    (I[4] = (T)(img)(_n1##x,_p3##y,z,c)),
//    (I[12] = (T)(img)(_n1##x,_p2##y,z,c)),
//    (I[20] = (T)(img)(_n1##x,_p1##y,z,c)),
//    (I[28] = (T)(img)(_n1##x,y,z,c)),
//    (I[36] = (T)(img)(_n1##x,_n1##y,z,c)),
//    (I[44] = (T)(img)(_n1##x,_n2##y,z,c)),
//    (I[52] = (T)(img)(_n1##x,_n3##y,z,c)),
//    (I[60] = (T)(img)(_n1##x,_n4##y,z,c)),
//    (I[5] = (T)(img)(_n2##x,_p3##y,z,c)),
//    (I[13] = (T)(img)(_n2##x,_p2##y,z,c)),
//    (I[21] = (T)(img)(_n2##x,_p1##y,z,c)),
//    (I[29] = (T)(img)(_n2##x,y,z,c)),
//    (I[37] = (T)(img)(_n2##x,_n1##y,z,c)),
//    (I[45] = (T)(img)(_n2##x,_n2##y,z,c)),
//    (I[53] = (T)(img)(_n2##x,_n3##y,z,c)),
//    (I[61] = (T)(img)(_n2##x,_n4##y,z,c)),
//    (I[6] = (T)(img)(_n3##x,_p3##y,z,c)),
//    (I[14] = (T)(img)(_n3##x,_p2##y,z,c)),
//    (I[22] = (T)(img)(_n3##x,_p1##y,z,c)),
//    (I[30] = (T)(img)(_n3##x,y,z,c)),
//    (I[38] = (T)(img)(_n3##x,_n1##y,z,c)),
//    (I[46] = (T)(img)(_n3##x,_n2##y,z,c)),
//    (I[54] = (T)(img)(_n3##x,_n3##y,z,c)),
//    (I[62] = (T)(img)(_n3##x,_n4##y,z,c)),
//    4>=((img)._width)?(img).width()-1:4);
//    (_n4##x<(img).width() && (
//    (I[7] = (T)(img)(_n4##x,_p3##y,z,c)),
//    (I[15] = (T)(img)(_n4##x,_p2##y,z,c)),
//    (I[23] = (T)(img)(_n4##x,_p1##y,z,c)),
//    (I[31] = (T)(img)(_n4##x,y,z,c)),
//    (I[39] = (T)(img)(_n4##x,_n1##y,z,c)),
//    (I[47] = (T)(img)(_n4##x,_n2##y,z,c)),
//    (I[55] = (T)(img)(_n4##x,_n3##y,z,c)),
//    (I[63] = (T)(img)(_n4##x,_n4##y,z,c)),1)) ||
//    _n3##x==--_n4##x || _n2##x==--_n3##x || _n1##x==--_n2##x || x==(_n4##x = _n3##x = _n2##x = --_n1##x);
//    I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4], I[4] = I[5], I[5] = I[6], I[6] = I[7],
//    I[8] = I[9], I[9] = I[10], I[10] = I[11], I[11] = I[12], I[12] = I[13], I[13] = I[14], I[14] = I[15],
//    I[16] = I[17], I[17] = I[18], I[18] = I[19], I[19] = I[20], I[20] = I[21], I[21] = I[22], I[22] = I[23],
//    I[24] = I[25], I[25] = I[26], I[26] = I[27], I[27] = I[28], I[28] = I[29], I[29] = I[30], I[30] = I[31],
//    I[32] = I[33], I[33] = I[34], I[34] = I[35], I[35] = I[36], I[36] = I[37], I[37] = I[38], I[38] = I[39],
//    I[40] = I[41], I[41] = I[42], I[42] = I[43], I[43] = I[44], I[44] = I[45], I[45] = I[46], I[46] = I[47],
//    I[48] = I[49], I[49] = I[50], I[50] = I[51], I[51] = I[52], I[52] = I[53], I[53] = I[54], I[54] = I[55],
//    I[56] = I[57], I[57] = I[58], I[58] = I[59], I[59] = I[60], I[60] = I[61], I[61] = I[62], I[62] = I[63],
//    _p3##x = _p2##x, _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x, ++_n3##x, ++_n4##x)

// #define cimg_for_in8x8(img,x0,y0,x1,y1,x,y,z,c,I,T)
//   cimg_for_in8((img)._height,y0,y1,y) for (int x = (int)(x0)<0?0:(int)(x0),
//    _p3##x = x-3<0?0:x-3,
//    _p2##x = x-2<0?0:x-2,
//    _p1##x = x-1<0?0:x-1,
//    _n1##x = x+1>=(img).width()?(img).width()-1:x+1,
//    _n2##x = x+2>=(img).width()?(img).width()-1:x+2,
//    _n3##x = x+3>=(img).width()?(img).width()-1:x+3,
//    _n4##x = (int)(
//    (I[0] = (T)(img)(_p3##x,_p3##y,z,c)),
//    (I[8] = (T)(img)(_p3##x,_p2##y,z,c)),
//    (I[16] = (T)(img)(_p3##x,_p1##y,z,c)),
//    (I[24] = (T)(img)(_p3##x,y,z,c)),
//    (I[32] = (T)(img)(_p3##x,_n1##y,z,c)),
//    (I[40] = (T)(img)(_p3##x,_n2##y,z,c)),
//    (I[48] = (T)(img)(_p3##x,_n3##y,z,c)),
//    (I[56] = (T)(img)(_p3##x,_n4##y,z,c)),
//    (I[1] = (T)(img)(_p2##x,_p3##y,z,c)),
//    (I[9] = (T)(img)(_p2##x,_p2##y,z,c)),
//    (I[17] = (T)(img)(_p2##x,_p1##y,z,c)),
//    (I[25] = (T)(img)(_p2##x,y,z,c)),
//    (I[33] = (T)(img)(_p2##x,_n1##y,z,c)),
//    (I[41] = (T)(img)(_p2##x,_n2##y,z,c)),
//    (I[49] = (T)(img)(_p2##x,_n3##y,z,c)),
//    (I[57] = (T)(img)(_p2##x,_n4##y,z,c)),
//    (I[2] = (T)(img)(_p1##x,_p3##y,z,c)),
//    (I[10] = (T)(img)(_p1##x,_p2##y,z,c)),
//    (I[18] = (T)(img)(_p1##x,_p1##y,z,c)),
//    (I[26] = (T)(img)(_p1##x,y,z,c)),
//    (I[34] = (T)(img)(_p1##x,_n1##y,z,c)),
//    (I[42] = (T)(img)(_p1##x,_n2##y,z,c)),
//    (I[50] = (T)(img)(_p1##x,_n3##y,z,c)),
//    (I[58] = (T)(img)(_p1##x,_n4##y,z,c)),
//    (I[3] = (T)(img)(x,_p3##y,z,c)),
//    (I[11] = (T)(img)(x,_p2##y,z,c)),
//    (I[19] = (T)(img)(x,_p1##y,z,c)),
//    (I[27] = (T)(img)(x,y,z,c)),
//    (I[35] = (T)(img)(x,_n1##y,z,c)),
//    (I[43] = (T)(img)(x,_n2##y,z,c)),
//    (I[51] = (T)(img)(x,_n3##y,z,c)),
//    (I[59] = (T)(img)(x,_n4##y,z,c)),
//    (I[4] = (T)(img)(_n1##x,_p3##y,z,c)),
//    (I[12] = (T)(img)(_n1##x,_p2##y,z,c)),
//    (I[20] = (T)(img)(_n1##x,_p1##y,z,c)),
//    (I[28] = (T)(img)(_n1##x,y,z,c)),
//    (I[36] = (T)(img)(_n1##x,_n1##y,z,c)),
//    (I[44] = (T)(img)(_n1##x,_n2##y,z,c)),
//    (I[52] = (T)(img)(_n1##x,_n3##y,z,c)),
//    (I[60] = (T)(img)(_n1##x,_n4##y,z,c)),
//    (I[5] = (T)(img)(_n2##x,_p3##y,z,c)),
//    (I[13] = (T)(img)(_n2##x,_p2##y,z,c)),
//    (I[21] = (T)(img)(_n2##x,_p1##y,z,c)),
//    (I[29] = (T)(img)(_n2##x,y,z,c)),
//    (I[37] = (T)(img)(_n2##x,_n1##y,z,c)),
//    (I[45] = (T)(img)(_n2##x,_n2##y,z,c)),
//    (I[53] = (T)(img)(_n2##x,_n3##y,z,c)),
//    (I[61] = (T)(img)(_n2##x,_n4##y,z,c)),
//    (I[6] = (T)(img)(_n3##x,_p3##y,z,c)),
//    (I[14] = (T)(img)(_n3##x,_p2##y,z,c)),
//    (I[22] = (T)(img)(_n3##x,_p1##y,z,c)),
//    (I[30] = (T)(img)(_n3##x,y,z,c)),
//    (I[38] = (T)(img)(_n3##x,_n1##y,z,c)),
//    (I[46] = (T)(img)(_n3##x,_n2##y,z,c)),
//    (I[54] = (T)(img)(_n3##x,_n3##y,z,c)),
//    (I[62] = (T)(img)(_n3##x,_n4##y,z,c)),
//    x+4>=(img).width()?(img).width()-1:x+4);
//    x<=(int)(x1) && ((_n4##x<(img).width() && (
//    (I[7] = (T)(img)(_n4##x,_p3##y,z,c)),
//    (I[15] = (T)(img)(_n4##x,_p2##y,z,c)),
//    (I[23] = (T)(img)(_n4##x,_p1##y,z,c)),
//    (I[31] = (T)(img)(_n4##x,y,z,c)),
//    (I[39] = (T)(img)(_n4##x,_n1##y,z,c)),
//    (I[47] = (T)(img)(_n4##x,_n2##y,z,c)),
//    (I[55] = (T)(img)(_n4##x,_n3##y,z,c)),
//    (I[63] = (T)(img)(_n4##x,_n4##y,z,c)),1)) ||
//    _n3##x==--_n4##x || _n2##x==--_n3##x || _n1##x==--_n2##x || x==(_n4##x = _n3##x = _n2##x = --_n1##x));
//    I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4], I[4] = I[5], I[5] = I[6], I[6] = I[7],
//    I[8] = I[9], I[9] = I[10], I[10] = I[11], I[11] = I[12], I[12] = I[13], I[13] = I[14], I[14] = I[15],
//    I[16] = I[17], I[17] = I[18], I[18] = I[19], I[19] = I[20], I[20] = I[21], I[21] = I[22], I[22] = I[23],
//    I[24] = I[25], I[25] = I[26], I[26] = I[27], I[27] = I[28], I[28] = I[29], I[29] = I[30], I[30] = I[31],
//    I[32] = I[33], I[33] = I[34], I[34] = I[35], I[35] = I[36], I[36] = I[37], I[37] = I[38], I[38] = I[39],
//    I[40] = I[41], I[41] = I[42], I[42] = I[43], I[43] = I[44], I[44] = I[45], I[45] = I[46], I[46] = I[47],
//    I[48] = I[49], I[49] = I[50], I[50] = I[51], I[51] = I[52], I[52] = I[53], I[53] = I[54], I[54] = I[55],
//    I[56] = I[57], I[57] = I[58], I[58] = I[59], I[59] = I[60], I[60] = I[61], I[61] = I[62], I[62] = I[63],
//    _p3##x = _p2##x, _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x, ++_n3##x, ++_n4##x)

// #define cimg_for9x9(img,x,y,z,c,I,T)
//   cimg_for9((img)._height,y) for (int x = 0,
//    _p4##x = 0, _p3##x = 0, _p2##x = 0, _p1##x = 0,
//    _n1##x = 1>=((img)._width)?(img).width()-1:1,
//    _n2##x = 2>=((img)._width)?(img).width()-1:2,
//    _n3##x = 3>=((img)._width)?(img).width()-1:3,
//    _n4##x = (int)(
//    (I[0] = I[1] = I[2] = I[3] = I[4] = (T)(img)(_p4##x,_p4##y,z,c)),
//    (I[9] = I[10] = I[11] = I[12] = I[13] = (T)(img)(0,_p3##y,z,c)),
//    (I[18] = I[19] = I[20] = I[21] = I[22] = (T)(img)(0,_p2##y,z,c)),
//    (I[27] = I[28] = I[29] = I[30] = I[31] = (T)(img)(0,_p1##y,z,c)),
//    (I[36] = I[37] = I[38] = I[39] = I[40] = (T)(img)(0,y,z,c)),
//    (I[45] = I[46] = I[47] = I[48] = I[49] = (T)(img)(0,_n1##y,z,c)),
//    (I[54] = I[55] = I[56] = I[57] = I[58] = (T)(img)(0,_n2##y,z,c)),
//    (I[63] = I[64] = I[65] = I[66] = I[67] = (T)(img)(0,_n3##y,z,c)),
//    (I[72] = I[73] = I[74] = I[75] = I[76] = (T)(img)(0,_n4##y,z,c)),
//    (I[5] = (T)(img)(_n1##x,_p4##y,z,c)),
//    (I[14] = (T)(img)(_n1##x,_p3##y,z,c)),
//    (I[23] = (T)(img)(_n1##x,_p2##y,z,c)),
//    (I[32] = (T)(img)(_n1##x,_p1##y,z,c)),
//    (I[41] = (T)(img)(_n1##x,y,z,c)),
//    (I[50] = (T)(img)(_n1##x,_n1##y,z,c)),
//    (I[59] = (T)(img)(_n1##x,_n2##y,z,c)),
//    (I[68] = (T)(img)(_n1##x,_n3##y,z,c)),
//    (I[77] = (T)(img)(_n1##x,_n4##y,z,c)),
//    (I[6] = (T)(img)(_n2##x,_p4##y,z,c)),
//    (I[15] = (T)(img)(_n2##x,_p3##y,z,c)),
//    (I[24] = (T)(img)(_n2##x,_p2##y,z,c)),
//    (I[33] = (T)(img)(_n2##x,_p1##y,z,c)),
//    (I[42] = (T)(img)(_n2##x,y,z,c)),
//    (I[51] = (T)(img)(_n2##x,_n1##y,z,c)),
//    (I[60] = (T)(img)(_n2##x,_n2##y,z,c)),
//    (I[69] = (T)(img)(_n2##x,_n3##y,z,c)),
//    (I[78] = (T)(img)(_n2##x,_n4##y,z,c)),
//    (I[7] = (T)(img)(_n3##x,_p4##y,z,c)),
//    (I[16] = (T)(img)(_n3##x,_p3##y,z,c)),
//    (I[25] = (T)(img)(_n3##x,_p2##y,z,c)),
//    (I[34] = (T)(img)(_n3##x,_p1##y,z,c)),
//    (I[43] = (T)(img)(_n3##x,y,z,c)),
//    (I[52] = (T)(img)(_n3##x,_n1##y,z,c)),
//    (I[61] = (T)(img)(_n3##x,_n2##y,z,c)),
//    (I[70] = (T)(img)(_n3##x,_n3##y,z,c)),
//    (I[79] = (T)(img)(_n3##x,_n4##y,z,c)),
//    4>=((img)._width)?(img).width()-1:4);
//    (_n4##x<(img).width() && (
//    (I[8] = (T)(img)(_n4##x,_p4##y,z,c)),
//    (I[17] = (T)(img)(_n4##x,_p3##y,z,c)),
//    (I[26] = (T)(img)(_n4##x,_p2##y,z,c)),
//    (I[35] = (T)(img)(_n4##x,_p1##y,z,c)),
//    (I[44] = (T)(img)(_n4##x,y,z,c)),
//    (I[53] = (T)(img)(_n4##x,_n1##y,z,c)),
//    (I[62] = (T)(img)(_n4##x,_n2##y,z,c)),
//    (I[71] = (T)(img)(_n4##x,_n3##y,z,c)),
//    (I[80] = (T)(img)(_n4##x,_n4##y,z,c)),1)) ||
//    _n3##x==--_n4##x || _n2##x==--_n3##x || _n1##x==--_n2##x || x==(_n4##x = _n3##x = _n2##x = --_n1##x);
//    I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4], I[4] = I[5], I[5] = I[6], I[6] = I[7], I[7] = I[8],
//    I[9] = I[10], I[10] = I[11], I[11] = I[12], I[12] = I[13], I[13] = I[14], I[14] = I[15], I[15] = I[16], I[16] = I[17],
//    I[18] = I[19], I[19] = I[20], I[20] = I[21], I[21] = I[22], I[22] = I[23], I[23] = I[24], I[24] = I[25], I[25] = I[26],
//    I[27] = I[28], I[28] = I[29], I[29] = I[30], I[30] = I[31], I[31] = I[32], I[32] = I[33], I[33] = I[34], I[34] = I[35],
//    I[36] = I[37], I[37] = I[38], I[38] = I[39], I[39] = I[40], I[40] = I[41], I[41] = I[42], I[42] = I[43], I[43] = I[44],
//    I[45] = I[46], I[46] = I[47], I[47] = I[48], I[48] = I[49], I[49] = I[50], I[50] = I[51], I[51] = I[52], I[52] = I[53],
//    I[54] = I[55], I[55] = I[56], I[56] = I[57], I[57] = I[58], I[58] = I[59], I[59] = I[60], I[60] = I[61], I[61] = I[62],
//    I[63] = I[64], I[64] = I[65], I[65] = I[66], I[66] = I[67], I[67] = I[68], I[68] = I[69], I[69] = I[70], I[70] = I[71],
//    I[72] = I[73], I[73] = I[74], I[74] = I[75], I[75] = I[76], I[76] = I[77], I[77] = I[78], I[78] = I[79], I[79] = I[80],
//    _p4##x = _p3##x, _p3##x = _p2##x, _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x, ++_n3##x, ++_n4##x)

// #define cimg_for_in9x9(img,x0,y0,x1,y1,x,y,z,c,I,T)
//   cimg_for_in9((img)._height,y0,y1,y) for (int x = (int)(x0)<0?0:(int)(x0),
//    _p4##x = x-4<0?0:x-4,
//    _p3##x = x-3<0?0:x-3,
//    _p2##x = x-2<0?0:x-2,
//    _p1##x = x-1<0?0:x-1,
//    _n1##x = x+1>=(img).width()?(img).width()-1:x+1,
//    _n2##x = x+2>=(img).width()?(img).width()-1:x+2,
//    _n3##x = x+3>=(img).width()?(img).width()-1:x+3,
//    _n4##x = (int)(
//    (I[0] = (T)(img)(_p4##x,_p4##y,z,c)),
//    (I[9] = (T)(img)(_p4##x,_p3##y,z,c)),
//    (I[18] = (T)(img)(_p4##x,_p2##y,z,c)),
//    (I[27] = (T)(img)(_p4##x,_p1##y,z,c)),
//    (I[36] = (T)(img)(_p4##x,y,z,c)),
//    (I[45] = (T)(img)(_p4##x,_n1##y,z,c)),
//    (I[54] = (T)(img)(_p4##x,_n2##y,z,c)),
//    (I[63] = (T)(img)(_p4##x,_n3##y,z,c)),
//    (I[72] = (T)(img)(_p4##x,_n4##y,z,c)),
//    (I[1] = (T)(img)(_p3##x,_p4##y,z,c)),
//    (I[10] = (T)(img)(_p3##x,_p3##y,z,c)),
//    (I[19] = (T)(img)(_p3##x,_p2##y,z,c)),
//    (I[28] = (T)(img)(_p3##x,_p1##y,z,c)),
//    (I[37] = (T)(img)(_p3##x,y,z,c)),
//    (I[46] = (T)(img)(_p3##x,_n1##y,z,c)),
//    (I[55] = (T)(img)(_p3##x,_n2##y,z,c)),
//    (I[64] = (T)(img)(_p3##x,_n3##y,z,c)),
//    (I[73] = (T)(img)(_p3##x,_n4##y,z,c)),
//    (I[2] = (T)(img)(_p2##x,_p4##y,z,c)),
//    (I[11] = (T)(img)(_p2##x,_p3##y,z,c)),
//    (I[20] = (T)(img)(_p2##x,_p2##y,z,c)),
//    (I[29] = (T)(img)(_p2##x,_p1##y,z,c)),
//    (I[38] = (T)(img)(_p2##x,y,z,c)),
//    (I[47] = (T)(img)(_p2##x,_n1##y,z,c)),
//    (I[56] = (T)(img)(_p2##x,_n2##y,z,c)),
//    (I[65] = (T)(img)(_p2##x,_n3##y,z,c)),
//    (I[74] = (T)(img)(_p2##x,_n4##y,z,c)),
//    (I[3] = (T)(img)(_p1##x,_p4##y,z,c)),
//    (I[12] = (T)(img)(_p1##x,_p3##y,z,c)),
//    (I[21] = (T)(img)(_p1##x,_p2##y,z,c)),
//    (I[30] = (T)(img)(_p1##x,_p1##y,z,c)),
//    (I[39] = (T)(img)(_p1##x,y,z,c)),
//    (I[48] = (T)(img)(_p1##x,_n1##y,z,c)),
//    (I[57] = (T)(img)(_p1##x,_n2##y,z,c)),
//    (I[66] = (T)(img)(_p1##x,_n3##y,z,c)),
//    (I[75] = (T)(img)(_p1##x,_n4##y,z,c)),
//    (I[4] = (T)(img)(x,_p4##y,z,c)),
//    (I[13] = (T)(img)(x,_p3##y,z,c)),
//    (I[22] = (T)(img)(x,_p2##y,z,c)),
//    (I[31] = (T)(img)(x,_p1##y,z,c)),
//    (I[40] = (T)(img)(x,y,z,c)),
//    (I[49] = (T)(img)(x,_n1##y,z,c)),
//    (I[58] = (T)(img)(x,_n2##y,z,c)),
//    (I[67] = (T)(img)(x,_n3##y,z,c)),
//    (I[76] = (T)(img)(x,_n4##y,z,c)),
//    (I[5] = (T)(img)(_n1##x,_p4##y,z,c)),
//    (I[14] = (T)(img)(_n1##x,_p3##y,z,c)),
//    (I[23] = (T)(img)(_n1##x,_p2##y,z,c)),
//    (I[32] = (T)(img)(_n1##x,_p1##y,z,c)),
//    (I[41] = (T)(img)(_n1##x,y,z,c)),
//    (I[50] = (T)(img)(_n1##x,_n1##y,z,c)),
//    (I[59] = (T)(img)(_n1##x,_n2##y,z,c)),
//    (I[68] = (T)(img)(_n1##x,_n3##y,z,c)),
//    (I[77] = (T)(img)(_n1##x,_n4##y,z,c)),
//    (I[6] = (T)(img)(_n2##x,_p4##y,z,c)),
//    (I[15] = (T)(img)(_n2##x,_p3##y,z,c)),
//    (I[24] = (T)(img)(_n2##x,_p2##y,z,c)),
//    (I[33] = (T)(img)(_n2##x,_p1##y,z,c)),
//    (I[42] = (T)(img)(_n2##x,y,z,c)),
//    (I[51] = (T)(img)(_n2##x,_n1##y,z,c)),
//    (I[60] = (T)(img)(_n2##x,_n2##y,z,c)),
//    (I[69] = (T)(img)(_n2##x,_n3##y,z,c)),
//    (I[78] = (T)(img)(_n2##x,_n4##y,z,c)),
//    (I[7] = (T)(img)(_n3##x,_p4##y,z,c)),
//    (I[16] = (T)(img)(_n3##x,_p3##y,z,c)),
//    (I[25] = (T)(img)(_n3##x,_p2##y,z,c)),
//    (I[34] = (T)(img)(_n3##x,_p1##y,z,c)),
//    (I[43] = (T)(img)(_n3##x,y,z,c)),
//    (I[52] = (T)(img)(_n3##x,_n1##y,z,c)),
//    (I[61] = (T)(img)(_n3##x,_n2##y,z,c)),
//    (I[70] = (T)(img)(_n3##x,_n3##y,z,c)),
//    (I[79] = (T)(img)(_n3##x,_n4##y,z,c)),
//    x+4>=(img).width()?(img).width()-1:x+4);
//    x<=(int)(x1) && ((_n4##x<(img).width() && (
//    (I[8] = (T)(img)(_n4##x,_p4##y,z,c)),
//    (I[17] = (T)(img)(_n4##x,_p3##y,z,c)),
//    (I[26] = (T)(img)(_n4##x,_p2##y,z,c)),
//    (I[35] = (T)(img)(_n4##x,_p1##y,z,c)),
//    (I[44] = (T)(img)(_n4##x,y,z,c)),
//    (I[53] = (T)(img)(_n4##x,_n1##y,z,c)),
//    (I[62] = (T)(img)(_n4##x,_n2##y,z,c)),
//    (I[71] = (T)(img)(_n4##x,_n3##y,z,c)),
//    (I[80] = (T)(img)(_n4##x,_n4##y,z,c)),1)) ||
//    _n3##x==--_n4##x || _n2##x==--_n3##x || _n1##x==--_n2##x || x==(_n4##x = _n3##x = _n2##x = --_n1##x));
//    I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4], I[4] = I[5], I[5] = I[6], I[6] = I[7], I[7] = I[8],
//    I[9] = I[10], I[10] = I[11], I[11] = I[12], I[12] = I[13], I[13] = I[14], I[14] = I[15], I[15] = I[16], I[16] = I[17],
//    I[18] = I[19], I[19] = I[20], I[20] = I[21], I[21] = I[22], I[22] = I[23], I[23] = I[24], I[24] = I[25], I[25] = I[26],
//    I[27] = I[28], I[28] = I[29], I[29] = I[30], I[30] = I[31], I[31] = I[32], I[32] = I[33], I[33] = I[34], I[34] = I[35],
//    I[36] = I[37], I[37] = I[38], I[38] = I[39], I[39] = I[40], I[40] = I[41], I[41] = I[42], I[42] = I[43], I[43] = I[44],
//    I[45] = I[46], I[46] = I[47], I[47] = I[48], I[48] = I[49], I[49] = I[50], I[50] = I[51], I[51] = I[52], I[52] = I[53],
//    I[54] = I[55], I[55] = I[56], I[56] = I[57], I[57] = I[58], I[58] = I[59], I[59] = I[60], I[60] = I[61], I[61] = I[62],
//    I[63] = I[64], I[64] = I[65], I[65] = I[66], I[66] = I[67], I[67] = I[68], I[68] = I[69], I[69] = I[70], I[70] = I[71],
//    I[72] = I[73], I[73] = I[74], I[74] = I[75], I[75] = I[76], I[76] = I[77], I[77] = I[78], I[78] = I[79], I[79] = I[80],
//    _p4##x = _p3##x, _p3##x = _p2##x, _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x, ++_n3##x, ++_n4##x)

// #define cimg_for2x2x2(img,x,y,z,c,I,T)
//  cimg_for2((img)._depth,z) cimg_for2((img)._height,y) for (int x = 0,
//    _n1##x = (int)(
//    (I[0] = (T)(img)(0,y,z,c)),
//    (I[2] = (T)(img)(0,_n1##y,z,c)),
//    (I[4] = (T)(img)(0,y,_n1##z,c)),
//    (I[6] = (T)(img)(0,_n1##y,_n1##z,c)),
//    1>=(img)._width?(img).width()-1:1);
//    (_n1##x<(img).width() && (
//    (I[1] = (T)(img)(_n1##x,y,z,c)),
//    (I[3] = (T)(img)(_n1##x,_n1##y,z,c)),
//    (I[5] = (T)(img)(_n1##x,y,_n1##z,c)),
//    (I[7] = (T)(img)(_n1##x,_n1##y,_n1##z,c)),1)) ||
//    x==--_n1##x;
//    I[0] = I[1], I[2] = I[3], I[4] = I[5], I[6] = I[7],
//    ++x, ++_n1##x)

// #define cimg_for_in2x2x2(img,x0,y0,z0,x1,y1,z1,x,y,z,c,I,T)
//  cimg_for_in2((img)._depth,z0,z1,z) cimg_for_in2((img)._height,y0,y1,y) for (int x = (int)(x0)<0?0:(int)(x0),
//    _n1##x = (int)(
//    (I[0] = (T)(img)(x,y,z,c)),
//    (I[2] = (T)(img)(x,_n1##y,z,c)),
//    (I[4] = (T)(img)(x,y,_n1##z,c)),
//    (I[6] = (T)(img)(x,_n1##y,_n1##z,c)),
//    x+1>=(int)(img)._width?(img).width()-1:x+1);
//    x<=(int)(x1) && ((_n1##x<(img).width() && (
//    (I[1] = (T)(img)(_n1##x,y,z,c)),
//    (I[3] = (T)(img)(_n1##x,_n1##y,z,c)),
//    (I[5] = (T)(img)(_n1##x,y,_n1##z,c)),
//    (I[7] = (T)(img)(_n1##x,_n1##y,_n1##z,c)),1)) ||
//    x==--_n1##x);
//    I[0] = I[1], I[2] = I[3], I[4] = I[5], I[6] = I[7],
//    ++x, ++_n1##x)

// #define cimg_for3x3x3(img,x,y,z,c,I,T)
//  cimg_for3((img)._depth,z) cimg_for3((img)._height,y) for (int x = 0,
//    _p1##x = 0,
//    _n1##x = (int)(
//    (I[0] = I[1] = (T)(img)(_p1##x,_p1##y,_p1##z,c)),
//    (I[3] = I[4] = (T)(img)(0,y,_p1##z,c)),
//    (I[6] = I[7] = (T)(img)(0,_n1##y,_p1##z,c)),
//    (I[9] = I[10] = (T)(img)(0,_p1##y,z,c)),
//    (I[12] = I[13] = (T)(img)(0,y,z,c)),
//    (I[15] = I[16] = (T)(img)(0,_n1##y,z,c)),
//    (I[18] = I[19] = (T)(img)(0,_p1##y,_n1##z,c)),
//    (I[21] = I[22] = (T)(img)(0,y,_n1##z,c)),
//    (I[24] = I[25] = (T)(img)(0,_n1##y,_n1##z,c)),
//    1>=(img)._width?(img).width()-1:1);
//    (_n1##x<(img).width() && (
//    (I[2] = (T)(img)(_n1##x,_p1##y,_p1##z,c)),
//    (I[5] = (T)(img)(_n1##x,y,_p1##z,c)),
//    (I[8] = (T)(img)(_n1##x,_n1##y,_p1##z,c)),
//    (I[11] = (T)(img)(_n1##x,_p1##y,z,c)),
//    (I[14] = (T)(img)(_n1##x,y,z,c)),
//    (I[17] = (T)(img)(_n1##x,_n1##y,z,c)),
//    (I[20] = (T)(img)(_n1##x,_p1##y,_n1##z,c)),
//    (I[23] = (T)(img)(_n1##x,y,_n1##z,c)),
//    (I[26] = (T)(img)(_n1##x,_n1##y,_n1##z,c)),1)) ||
//    x==--_n1##x;
//    I[0] = I[1], I[1] = I[2], I[3] = I[4], I[4] = I[5], I[6] = I[7], I[7] = I[8],
//    I[9] = I[10], I[10] = I[11], I[12] = I[13], I[13] = I[14], I[15] = I[16], I[16] = I[17],
//    I[18] = I[19], I[19] = I[20], I[21] = I[22], I[22] = I[23], I[24] = I[25], I[25] = I[26],
//    _p1##x = x++, ++_n1##x)

// #define cimg_for_in3x3x3(img,x0,y0,z0,x1,y1,z1,x,y,z,c,I,T)
//  cimg_for_in3((img)._depth,z0,z1,z) cimg_for_in3((img)._height,y0,y1,y) for (int x = (int)(x0)<0?0:(int)(x0),
//    _p1##x = x-1<0?0:x-1,
//    _n1##x = (int)(
//    (I[0] = (T)(img)(_p1##x,_p1##y,_p1##z,c)),
//    (I[3] = (T)(img)(_p1##x,y,_p1##z,c)),
//    (I[6] = (T)(img)(_p1##x,_n1##y,_p1##z,c)),
//    (I[9] = (T)(img)(_p1##x,_p1##y,z,c)),
//    (I[12] = (T)(img)(_p1##x,y,z,c)),
//    (I[15] = (T)(img)(_p1##x,_n1##y,z,c)),
//    (I[18] = (T)(img)(_p1##x,_p1##y,_n1##z,c)),
//    (I[21] = (T)(img)(_p1##x,y,_n1##z,c)),
//    (I[24] = (T)(img)(_p1##x,_n1##y,_n1##z,c)),
//    (I[1] = (T)(img)(x,_p1##y,_p1##z,c)),
//    (I[4] = (T)(img)(x,y,_p1##z,c)),
//    (I[7] = (T)(img)(x,_n1##y,_p1##z,c)),
//    (I[10] = (T)(img)(x,_p1##y,z,c)),
//    (I[13] = (T)(img)(x,y,z,c)),
//    (I[16] = (T)(img)(x,_n1##y,z,c)),
//    (I[19] = (T)(img)(x,_p1##y,_n1##z,c)),
//    (I[22] = (T)(img)(x,y,_n1##z,c)),
//    (I[25] = (T)(img)(x,_n1##y,_n1##z,c)),
//    x+1>=(int)(img)._width?(img).width()-1:x+1);
//    x<=(int)(x1) && ((_n1##x<(img).width() && (
//    (I[2] = (T)(img)(_n1##x,_p1##y,_p1##z,c)),
//    (I[5] = (T)(img)(_n1##x,y,_p1##z,c)),
//    (I[8] = (T)(img)(_n1##x,_n1##y,_p1##z,c)),
//    (I[11] = (T)(img)(_n1##x,_p1##y,z,c)),
//    (I[14] = (T)(img)(_n1##x,y,z,c)),
//    (I[17] = (T)(img)(_n1##x,_n1##y,z,c)),
//    (I[20] = (T)(img)(_n1##x,_p1##y,_n1##z,c)),
//    (I[23] = (T)(img)(_n1##x,y,_n1##z,c)),
//    (I[26] = (T)(img)(_n1##x,_n1##y,_n1##z,c)),1)) ||
//    x==--_n1##x);
//    I[0] = I[1], I[1] = I[2], I[3] = I[4], I[4] = I[5], I[6] = I[7], I[7] = I[8],
//    I[9] = I[10], I[10] = I[11], I[12] = I[13], I[13] = I[14], I[15] = I[16], I[16] = I[17],
//    I[18] = I[19], I[19] = I[20], I[21] = I[22], I[22] = I[23], I[24] = I[25], I[25] = I[26],
//    _p1##x = x++, ++_n1##x)

// #define cimglist_for(list,l) for (int l = 0; l<(int)(list)._width; ++L)
// #define cimglist_for_in(list,l0,l1,l)
//   for (int l = (int)(l0)<0?0:(int)(l0), _max##l = (unsigned int)l1<(list)._width?(int)(l1):(int)(list)._width-1; l<=_max##l; ++L)

// #define cimglist_apply(list,fn) cimglist_for(list,__##fn) (list)[__##fn].fn

// Macros used to display error messages when exceptions are thrown.
// You should not use these macros is your own code.
public static final String _cimgdisplay_instance =";
public static final String cimgdisplay_instance = _width + , + _height + , + _normalization + , + _title + ? + '\"' + : + '[' + , + _title + ? + _title + : + "untitled" + , + _title + ? + '\"' + : + ']';
public static final String _cimg_instance =";
public static final String cimg_instance = _width + , + _height + , + _depth + , + _spectrum + , + _data + , + _is_shared + ? + "" + : + "non-" + , + pixel_type + ( + );
public static final String _cimglist_instance =";
public static native @MemberGetter int cimglist_instance();
public static final int cimglist_instance = cimglist_instance();

/*------------------------------------------------
 #
 #
 #  Define cimg_library:: namespace
 #
 #
 -------------------------------------------------*/
/** Contains <i>all classes and functions</i> of the \CImg library.
/**
   This namespace is defined to avoid functions and class names collisions
   that could happen with the inclusion of other C++ header files.
   Anyway, it should not happen often and you should reasonnably start most of your
   \CImg-based programs with
   <pre>{@code
   #include "CImg.h"
   using namespace cimg_library;
   }</pre>
   to simplify the declaration of \CImg Library objects afterwards.
**/

  // Declare the four classes of the CImg Library.

  // Declare cimg:: namespace.
  // This is an uncomplete namespace definition here. It only contains some
  // necessary stuffs to ensure a correct declaration order of the classes and functions
  // defined afterwards.

    // Define ascii sequences for colored terminal output.
// #ifdef cimg_use_vt100
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native byte t_normal(int i);
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const char*") BytePointer t_normal();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native byte t_black(int i);
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const char*") BytePointer t_black();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native byte t_red(int i);
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const char*") BytePointer t_red();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native byte t_green(int i);
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const char*") BytePointer t_green();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native byte t_yellow(int i);
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const char*") BytePointer t_yellow();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native byte t_blue(int i);
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const char*") BytePointer t_blue();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native byte t_magenta(int i);
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const char*") BytePointer t_magenta();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native byte t_cyan(int i);
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const char*") BytePointer t_cyan();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native byte t_white(int i);
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const char*") BytePointer t_white();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native byte t_bold(int i);
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const char*") BytePointer t_bold();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native byte t_underscore(int i);
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const char*") BytePointer t_underscore();
// #else
// #endif

    @Namespace("cimg_library_suffixed::cimg") public static native FILE output(FILE file/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native FILE output();
    @Namespace("cimg_library_suffixed::cimg") public static native void info();

    /** Avoid warning messages due to unused parameters. Do nothing actually. */

    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("unsigned int*") @ByRef IntPointer _exception_mode(@Cast("const unsigned int") int value, @Cast("const bool") boolean is_set);

    /** Set current \CImg exception mode.
    /**
       The way error messages are handled by \CImg can be changed dynamically, using this function.
       @param mode Desired exception mode. Possible values are:
       - \c 0: Hide library messages (quiet mode).
       - \c 1: Print library messages on the console.
       - \c 2: Display library messages on a dialog window (default behavior).
       - \c 3: Do as \c 1 + add extra debug warnings (slow down the code!).
       - \c 4: Do as \c 2 + add extra debug warnings (slow down the code!).
     **/
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("unsigned int*") @ByRef IntPointer exception_mode(@Cast("const unsigned int") int mode);

    /** Return current \CImg exception mode.
    /**
       \note By default, return the value of configuration macro \c cimg_verbosity
    **/
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("unsigned int*") @ByRef IntPointer exception_mode();

    @Namespace("cimg_library_suffixed::cimg") public static native int dialog(@Cast("const char*") BytePointer title, @Cast("const char*") BytePointer msg, @Cast("const char*") BytePointer button1_label/*="OK"*/,
                          @Cast("const char*") BytePointer button2_label/*=0*/, @Cast("const char*") BytePointer button3_label/*=0*/,
                          @Cast("const char*") BytePointer button4_label/*=0*/, @Cast("const char*") BytePointer button5_label/*=0*/,
                          @Cast("const char*") BytePointer button6_label/*=0*/, @Cast("const bool") boolean centering/*=false*/);
    @Namespace("cimg_library_suffixed::cimg") public static native int dialog(@Cast("const char*") BytePointer title, @Cast("const char*") BytePointer msg);
    @Namespace("cimg_library_suffixed::cimg") public static native int dialog(String title, String msg, String button1_label/*="OK"*/,
                          String button2_label/*=0*/, String button3_label/*=0*/,
                          String button4_label/*=0*/, String button5_label/*=0*/,
                          String button6_label/*=0*/, @Cast("const bool") boolean centering/*=false*/);
    @Namespace("cimg_library_suffixed::cimg") public static native int dialog(String title, String msg);

    @Namespace("cimg_library_suffixed::cimg") public static native double eval(@Cast("const char*") BytePointer expression, double x/*=0*/, double y/*=0*/, double z/*=0*/, double c/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native double eval(@Cast("const char*") BytePointer expression);
    @Namespace("cimg_library_suffixed::cimg") public static native double eval(String expression, double x/*=0*/, double y/*=0*/, double z/*=0*/, double c/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native double eval(String expression);
  

  /*---------------------------------------
    #
    # Define the CImgException structures
    #
    --------------------------------------*/
  /** Instances of \c CImgException are thrown when errors are encountered in a \CImg function call.
  /**
     \par Overview
      <p>
      CImgException is the base class of all exceptions thrown by \CImg.
      CImgException is never thrown itself. Derived classes that specify the type of errord are thrown instead.
      These derived classes can be:
      <p>
      - \b CImgArgumentException: Thrown when one argument of a called \CImg function is invalid.
      This is probably one of the most thrown exception by \CImg.
      For instance, the following example throws a \c CImgArgumentException:
      <pre>{@code
      CImg<float> img(100,100,1,3); // Define a 100x100 color image with float-valued pixels.
      img.mirror('e');              // Try to mirror image along the (non-existing) 'e'-axis.
      }</pre>
      <p>
      - \b CImgDisplayException: Thrown when something went wrong during the display of images in CImgDisplay instances.
      <p>
      - \b CImgInstanceException: Thrown when an instance associated to a called \CImg method does not fit
      the function requirements. For instance, the following example throws a \c CImgInstanceException:
      <pre>{@code
      const CImg<float> img;           // Define an empty image.
      const float value = img.at(0);   // Try to read first pixel value (does not exist).
      }</pre>
      <p>
      - \b CImgIOException: Thrown when an error occured when trying to load or save image files.
      This happens when trying to read files that do not exist or with invalid formats.
      For instance, the following example throws a \c CImgIOException:
      <pre>{@code
      const CImg<float> img("missing_file.jpg");  // Try to load a file that does not exist.
      }</pre>
      <p>
      - \b CImgWarningException: Thrown only if configuration macro \c cimg_strict_warnings is set, and
      when a \CImg function has to display a warning message (see cimg::warn()).
      <p>
      It is not recommended to throw CImgException instances by yourself, since they are expected to be thrown only by \CImg.
      When an error occurs in a library function call, \CImg may display error messages on the screen or on the
      standard output, depending on the current \CImg exception mode.
      The \CImg exception mode can be get and set by functions cimg::exception_mode() and cimg::exception_mode(unsigned int).
      <p>
      \par Exceptions handling
      <p>
      In all cases, when an error occurs in \CImg, an instance of the corresponding exception class is thrown.
      This may lead the program to break (this is the default behavior), but you can bypass this behavior by handling the exceptions by yourself,
      using a usual <tt>try { ... } catch () { ... }</tt> bloc, as in the following example:
      <pre>{@code
      #define "CImg.h"
      using namespace cimg_library;
      int main() {
        cimg::exception_mode(0);                                    // Enable quiet exception mode.
        try {
          ...                                                       // Here, do what you want to stress the CImg library.
        } catch (CImgException &e) {                                // You succeeded: something went wrong!
          std::fprintf(stderr,"CImg Library Error: %s",e.what());   // Display your custom error message.
          ...                                                       // Do what you want now to save the ship!
          }
        }
      }</pre>
  **/
  @Namespace("cimg_library_suffixed") @NoOffset public static class CImgException extends Pointer {
      static { Loader.load(); }
      /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
      public CImgException(Pointer p) { super(p); }
      /** Native array allocator. Access with {@link Pointer#position(long)}. */
      public CImgException(long size) { super((Pointer)null); allocateArray(size); }
      private native void allocateArray(long size);
      @Override public CImgException position(long position) {
          return (CImgException)super.position(position);
      }
  
// #define _cimg_exception_err(etype,disp_flag)
//   std::va_list ap; va_start(ap,format); cimg_vsnprintf(_message,sizeof(_message),format,ap); va_end(ap);
//   if (cimg::exception_mode()) {
//     std::fprintf(cimg::output(),"\n%s[CImg] *** %s ***%s %s\n",cimg::t_red,etype,cimg::t_normal,_message);
//     if (cimg_display && disp_flag && !(cimg::exception_mode()%2)) try { cimg::dialog(etype,_message,"Abort"); } catch (CImgException&) {}
//     if (cimg::exception_mode()>=3) cimg_library_suffixed::cimg::info();
//   }

    public native @Cast("char") byte _message(int i); public native CImgException _message(int i, byte _message);
    @MemberGetter public native @Cast("char*") BytePointer _message();
    public CImgException() { super((Pointer)null); allocate(); }
    private native void allocate();
    public CImgException(@Cast("const char*") BytePointer format) { super((Pointer)null); allocate(format); }
    private native void allocate(@Cast("const char*") BytePointer format);
    public CImgException(String format) { super((Pointer)null); allocate(format); }
    private native void allocate(String format);
    /** Return a C-string containing the error message associated to the thrown exception. */
    public native @Cast("const char*") BytePointer what();
  }

  // The CImgInstanceException class is used to throw an exception related
  // to an invalid instance encountered in a library function call.
  @Namespace("cimg_library_suffixed") public static class CImgInstanceException extends CImgException {
      static { Loader.load(); }
      /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
      public CImgInstanceException(Pointer p) { super(p); }
  
    public CImgInstanceException(@Cast("const char*") BytePointer format) { super((Pointer)null); allocate(format); }
    private native void allocate(@Cast("const char*") BytePointer format);
    public CImgInstanceException(String format) { super((Pointer)null); allocate(format); }
    private native void allocate(String format);
  }

  // The CImgArgumentException class is used to throw an exception related
  // to invalid arguments encountered in a library function call.
  @Namespace("cimg_library_suffixed") public static class CImgArgumentException extends CImgException {
      static { Loader.load(); }
      /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
      public CImgArgumentException(Pointer p) { super(p); }
  
    public CImgArgumentException(@Cast("const char*") BytePointer format) { super((Pointer)null); allocate(format); }
    private native void allocate(@Cast("const char*") BytePointer format);
    public CImgArgumentException(String format) { super((Pointer)null); allocate(format); }
    private native void allocate(String format);
  }

  // The CImgIOException class is used to throw an exception related
  // to input/output file problems encountered in a library function call.
  @Namespace("cimg_library_suffixed") public static class CImgIOException extends CImgException {
      static { Loader.load(); }
      /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
      public CImgIOException(Pointer p) { super(p); }
  
    public CImgIOException(@Cast("const char*") BytePointer format) { super((Pointer)null); allocate(format); }
    private native void allocate(@Cast("const char*") BytePointer format);
    public CImgIOException(String format) { super((Pointer)null); allocate(format); }
    private native void allocate(String format);
  }

  // The CImgDisplayException class is used to throw an exception related
  // to display problems encountered in a library function call.
  @Namespace("cimg_library_suffixed") public static class CImgDisplayException extends CImgException {
      static { Loader.load(); }
      /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
      public CImgDisplayException(Pointer p) { super(p); }
  
    public CImgDisplayException(@Cast("const char*") BytePointer format) { super((Pointer)null); allocate(format); }
    private native void allocate(@Cast("const char*") BytePointer format);
    public CImgDisplayException(String format) { super((Pointer)null); allocate(format); }
    private native void allocate(String format);
  }

  // The CImgWarningException class is used to throw an exception for warnings
  // encountered in a library function call.
  @Namespace("cimg_library_suffixed") public static class CImgWarningException extends CImgException {
      static { Loader.load(); }
      /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
      public CImgWarningException(Pointer p) { super(p); }
  
    public CImgWarningException(@Cast("const char*") BytePointer format) { super((Pointer)null); allocate(format); }
    private native void allocate(@Cast("const char*") BytePointer format);
    public CImgWarningException(String format) { super((Pointer)null); allocate(format); }
    private native void allocate(String format);
  }

  /*-------------------------------------
    #
    # Define cimg:: namespace
    #
    -----------------------------------*/
  /** Contains \a low-level functions and variables of the \CImg Library.
  /**
     Most of the functions and variables within this namespace are used by the \CImg library for low-level operations.
     You may use them to access specific const values or environment variables internally used by \CImg.
     \warning Never write <tt>using namespace cimg_library::cimg;</tt> in your source code. Lot of functions in the
     <tt>cimg:: namespace</tt> have the same names as standard C functions that may be defined in the global namespace <tt>::</tt>.
  **/

    // Define traits that will be used to determine the best data type to work in CImg functions.
    //

    @Name("cimg_library_suffixed::cimg::type<bool>") public static class type extends Pointer {
        static { Loader.load(); }
        /** Default native constructor. */
        public type() { super((Pointer)null); allocate(); }
        /** Native array allocator. Access with {@link Pointer#position(long)}. */
        public type(long size) { super((Pointer)null); allocateArray(size); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public type(Pointer p) { super(p); }
        private native void allocate();
        private native void allocateArray(long size);
        @Override public type position(long position) {
            return (type)super.position(position);
        }
    
      public static native @Cast("const char*") BytePointer string();
      public static native @Cast("bool") boolean is_float();
      public static native @Cast("bool") boolean is_inf(@Cast("const bool") boolean arg0);
      public static native @Cast("bool") boolean is_nan(@Cast("const bool") boolean arg0);
      public static native @Cast("bool") boolean min();
      public static native @Cast("bool") boolean max();
      public static native @Cast("bool") boolean inf();
      public static native @Cast("bool") boolean is_inf();
      public static native @Cast("bool") boolean cut(double val);
      public static native @Cast("const char*") BytePointer format();
      public static native @Cast("const char*") BytePointer format(@Cast("const bool") boolean val);
    }
    @Name("cimg_library_suffixed::cimg::superset<bool,unsigned char>") public static class superset extends Pointer {
        static { Loader.load(); }
        /** Default native constructor. */
        public superset() { super((Pointer)null); allocate(); }
        /** Native array allocator. Access with {@link Pointer#position(long)}. */
        public superset(long size) { super((Pointer)null); allocateArray(size); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public superset(Pointer p) { super(p); }
        private native void allocate();
        private native void allocateArray(long size);
        @Override public superset position(long position) {
            return (superset)super.position(position);
        }
     }

public static final int _cimg_Tt =type;
public static final int _cimg_Tfloat =type;
public static final int _cimg_Ttfloat =type;
public static final int _cimg_Ttdouble =type;

    // Define variables used internally by CImg.
// #if cimg_display==1
    @Namespace("cimg_library_suffixed::cimg") @NoOffset public static class X11_info extends Pointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public X11_info(Pointer p) { super(p); }
        /** Native array allocator. Access with {@link Pointer#position(long)}. */
        public X11_info(long size) { super((Pointer)null); allocateArray(size); }
        private native void allocateArray(long size);
        @Override public X11_info position(long position) {
            return (X11_info)super.position(position);
        }
    
      public native @ByRef volatile nb_wins(); public native X11_info nb_wins(volatile nb_wins);
      public native pthread_t event_thread(); public native X11_info event_thread(pthread_t event_thread);
      public native CImgDisplay wins(int i); public native X11_info wins(int i, CImgDisplay wins);
      @MemberGetter public native @Cast("cimg_library_suffixed::CImgDisplay**") PointerPointer wins();
      public native Display display(); public native X11_info display(Display display);
      public native @Cast("unsigned int") int nb_bits(); public native X11_info nb_bits(int nb_bits);
      public native @Cast("bool") boolean is_blue_first(); public native X11_info is_blue_first(boolean is_blue_first);
      public native @Cast("bool") boolean is_shm_enabled(); public native X11_info is_shm_enabled(boolean is_shm_enabled);
      public native @Cast("bool") boolean byte_order(); public native X11_info byte_order(boolean byte_order);
// #ifdef cimg_use_xrandr
      public native XRRScreenSize resolutions(); public native X11_info resolutions(XRRScreenSize resolutions);
      public native @ByRef Rotation curr_rotation(); public native X11_info curr_rotation(Rotation curr_rotation);
      public native @Cast("unsigned int") int curr_resolution(); public native X11_info curr_resolution(int curr_resolution);
      public native @Cast("unsigned int") int nb_resolutions(); public native X11_info nb_resolutions(int nb_resolutions);
// #endif
      public X11_info() { super((Pointer)null); allocate(); }
      private native void allocate();
    }
// #if defined(cimg_module)
    @Namespace("cimg_library_suffixed::cimg") public static native @ByRef X11_info X11_attr();
// #elif defined(cimg_main)
// #else
// #endif

// #elif cimg_display==2
    @Namespace("cimg_library_suffixed::cimg") @NoOffset public static class Win32_info extends Pointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public Win32_info(Pointer p) { super(p); }
        /** Native array allocator. Access with {@link Pointer#position(long)}. */
        public Win32_info(long size) { super((Pointer)null); allocateArray(size); }
        private native void allocateArray(long size);
        @Override public Win32_info position(long position) {
            return (Win32_info)super.position(position);
        }
    
      public native @ByRef HANDLE wait_event(); public native Win32_info wait_event(HANDLE wait_event);
      public Win32_info() { super((Pointer)null); allocate(); }
      private native void allocate();
    }
// #if defined(cimg_module)
    @Namespace("cimg_library_suffixed::cimg") public static native @ByRef Win32_info Win32_attr();
// #elif defined(cimg_main)
// #else
// #endif
// #endif

// #if defined(cimg_use_magick)
    @Namespace("cimg_library_suffixed::cimg") public static class Magick_info extends Pointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public Magick_info(Pointer p) { super(p); }
        /** Native array allocator. Access with {@link Pointer#position(long)}. */
        public Magick_info(long size) { super((Pointer)null); allocateArray(size); }
        private native void allocateArray(long size);
        @Override public Magick_info position(long position) {
            return (Magick_info)super.position(position);
        }
    
      public Magick_info() { super((Pointer)null); allocate(); }
      private native void allocate();
    }
// #endif

// #if cimg_display==1
    // Define keycodes for X11-based graphical systems.
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyESC();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyF1();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyF2();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyF3();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyF4();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyF5();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyF6();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyF7();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyF8();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyF9();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyF10();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyF11();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyF12();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyPAUSE();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int key1();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int key2();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int key3();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int key4();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int key5();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int key6();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int key7();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int key8();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int key9();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int key0();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyBACKSPACE();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyINSERT();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyHOME();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyPAGEUP();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyTAB();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyQ();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyW();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyE();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyR();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyT();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyY();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyU();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyI();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyO();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyP();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyDELETE();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyEND();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyPAGEDOWN();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyCAPSLOCK();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyA();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyS();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyD();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyF();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyG();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyH();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyJ();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyK();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyL();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyENTER();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keySHIFTLEFT();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyZ();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyX();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyC();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyV();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyB();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyN();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyM();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keySHIFTRIGHT();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyARROWUP();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyCTRLLEFT();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyAPPLEFT();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyALT();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keySPACE();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyALTGR();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyAPPRIGHT();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyMENU();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyCTRLRIGHT();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyARROWLEFT();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyARROWDOWN();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyARROWRIGHT();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyPAD0();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyPAD1();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyPAD2();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyPAD3();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyPAD4();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyPAD5();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyPAD6();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyPAD7();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyPAD8();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyPAD9();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyPADADD();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyPADSUB();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyPADMUL();
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int keyPADDIV();

// #elif cimg_display==2
    // Define keycodes for Windows.

// #else
    // Define random keycodes when no display is available.
    // (should rarely be used then!).
// #endif

    /** Value of the mathematical constant PI */
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native double PI();

    // Define a 10x13 font (small size).
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int font10x13(int i);
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int*") IntPointer font10x13();

    // Define a 12x24 font (normal size).
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int font12x24(int i);
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int*") IntPointer font12x24();

    // Define a 16x32 font (large size).
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int font16x32(int i);
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int*") IntPointer font16x32();

    // Define a 29x57 font (extra large size).
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int") int font29x57(int i);
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native @Cast("const unsigned int*") IntPointer font29x57();

    // Define a 40x38 'danger' color logo (used by cimg::dialog()).
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native byte logo40x38(int i);
    @Namespace("cimg_library_suffixed::cimg") @MemberGetter public static native byte logo40x38();

    /** Get/set default output stream for the \CImg library messages.
    /**
       @param file Desired output stream. Set to \c 0 to get the currently used output stream only.
       @return Currently used output stream.
    **/

    /** Display a warning message on the default output stream.
    /**
       @param format C-string containing the format of the message, as with <tt>std::printf()</tt>.
       \note If configuration macro \c cimg_strict_warnings is set, this function throws a \c CImgWarningException instead.
       \warning As the first argument is a format string, it is highly recommended to write
       <pre>{@code
       cimg::warn("%s",warning_message);
       }</pre>
       instead of
       <pre>{@code
       cimg::warn(warning_message);
       }</pre>
       if \c warning_message can be arbitrary, to prevent nasty memory access.
    **/
    @Namespace("cimg_library_suffixed::cimg") public static native void warn(@Cast("const char*") BytePointer format);
    @Namespace("cimg_library_suffixed::cimg") public static native void warn(String format);

    // Execute an external system command.
    /**
       @param command C-string containing the command line to execute.
       @param module_name Module name.
       @return Status value of the executed command, whose meaning is OS-dependent.
       \note This function is similar to <tt>std::system()</tt>
       but it does not open an extra console windows
       on Windows-based systems.
    **/
    @Namespace("cimg_library_suffixed::cimg") public static native int system(@Cast("const char*") BytePointer command, @Cast("const char*") BytePointer module_name/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native int system(@Cast("const char*") BytePointer command);
    @Namespace("cimg_library_suffixed::cimg") public static native int system(String command, String module_name/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native int system(String command);

    /** Return a reference to a temporary variable of type T. */

    /** Exchange values of variables \c a and \c b. */

    /** Exchange values of variables (\c a1,\c a2) and (\c b1,\c b2). */

    /** Exchange values of variables (\c a1,\c a2,\c a3) and (\c b1,\c b2,\c b3). */

    /** Exchange values of variables (\c a1,\c a2,...,\c a4) and (\c b1,\c b2,...,\c b4). */

    /** Exchange values of variables (\c a1,\c a2,...,\c a5) and (\c b1,\c b2,...,\c b5). */

    /** Exchange values of variables (\c a1,\c a2,...,\c a6) and (\c b1,\c b2,...,\c b6). */

    /** Exchange values of variables (\c a1,\c a2,...,\c a7) and (\c b1,\c b2,...,\c b7). */

    /** Exchange values of variables (\c a1,\c a2,...,\c a8) and (\c b1,\c b2,...,\c b8). */

    /** Return the endianness of the current architecture.
    /**
       @return \c false for <i>Little Endian</i> or \c true for <i>Big Endian</i>.
    **/
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("bool") boolean endianness();

    /** Reverse endianness of all elements in a memory buffer.
    /**
       @param [in,out] buffer Memory buffer whose endianness must be reversed.
       @param size Number of buffer elements to reverse.
    **/

    /** Reverse endianness of a single variable.
    /**
       @param [in,out] a Variable to reverse.
       @return Reference to reversed variable.
    **/

    // Conversion functions to get more precision when trying to store unsigned ints values as floats.
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("unsigned int") int float2uint(float f);

    @Namespace("cimg_library_suffixed::cimg") public static native float uint2float(@Cast("const unsigned int") int u);

    /** Return the value of a system timer, with a millisecond precision.
    /**
       \note The timer does not necessarily starts from \c 0.
    **/
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("unsigned long") long time();

    // Implement a tic/toc mechanism to display elapsed time of algorithms.
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("unsigned long") long tictoc(@Cast("const bool") boolean is_tic);

    /** Start tic/toc timer for time measurement between code instructions.
    /**
       @return Current value of the timer (same value as time()).
    **/
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("unsigned long") long tic();

    /** End tic/toc timer and displays elapsed time from last call to tic().
    /**
       @return Time elapsed (in ms) since last call to tic().
    **/
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("unsigned long") long toc();

    /** Sleep for a given numbers of milliseconds.
    /**
       @param milliseconds Number of milliseconds to wait for.
       \note This function frees the CPU ressources during the sleeping time.
       It can be used to temporize your program properly, without wasting CPU time.
    **/
    @Namespace("cimg_library_suffixed::cimg") public static native void sleep(@Cast("const unsigned int") int milliseconds);

    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("unsigned int") int _wait(@Cast("const unsigned int") int milliseconds, @Cast("unsigned long*") @ByRef CLongPointer timer);

    /** Wait for a given number of milliseconds since the last call to wait().
    /**
       @param milliseconds Number of milliseconds to wait for.
       @return Number of milliseconds elapsed since the last call to wait().
       \note Same as sleep() with a waiting time computed with regard to the last call
       of wait(). It may be used to temporize your program properly, without wasting CPU time.
    **/
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("unsigned int") int wait(@Cast("const unsigned int") int milliseconds);

    // Random number generators.
    // CImg may use its own Random Number Generator (RNG) if configuration macro 'cimg_use_rng' is set.
    // Use it for instance when you have to deal with concurrent threads trying to call std::srand()
    // at the same time!
// #ifdef cimg_use_rng

    // Use a custom RNG.
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("unsigned int") int _rand(@Cast("const unsigned int") int seed/*=0*/, @Cast("const bool") boolean set_seed/*=false*/);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("unsigned int") int _rand();

    @Namespace("cimg_library_suffixed::cimg") public static native void srand();

    @Namespace("cimg_library_suffixed::cimg") public static native void srand(@Cast("const unsigned int") int seed);

    @Namespace("cimg_library_suffixed::cimg") public static native double rand();

// #else

    // Use the system RNG.

    /** Return a random variable between [0,1] with respect to an uniform distribution.
    /**
    **/
// #endif

    /** Return a random variable between [-1,1] with respect to an uniform distribution.
    /**
    **/
    @Namespace("cimg_library_suffixed::cimg") public static native double crand();

    /** Return a random variable following a gaussian distribution and a standard deviation of 1.
    /**
    **/
    @Namespace("cimg_library_suffixed::cimg") public static native double grand();

    /** Return a random variable following a Poisson distribution of parameter z.
    /**
    **/
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("unsigned int") int prand(double z);

    /** Bitwise-rotate value on the left. */

    @Namespace("cimg_library_suffixed::cimg") public static native float rol(float a, @Cast("const unsigned int") int n/*=1*/);
    @Namespace("cimg_library_suffixed::cimg") public static native float rol(float a);

    @Namespace("cimg_library_suffixed::cimg") public static native double rol(double a, @Cast("const unsigned int") int n/*=1*/);
    @Namespace("cimg_library_suffixed::cimg") public static native double rol(double a);

    /** Bitwise-rotate value on the right. */

    @Namespace("cimg_library_suffixed::cimg") public static native float ror(float a, @Cast("const unsigned int") int n/*=1*/);
    @Namespace("cimg_library_suffixed::cimg") public static native float ror(float a);

    @Namespace("cimg_library_suffixed::cimg") public static native double ror(double a, @Cast("const unsigned int") int n/*=1*/);
    @Namespace("cimg_library_suffixed::cimg") public static native double ror(double a);

    /** Return absolute value of a value. */
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("bool") boolean abs(@Cast("const bool") boolean a);
    @Namespace("cimg_library_suffixed::cimg") public static native byte abs(byte a);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("unsigned short") short abs(@Cast("const unsigned short") short a);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("unsigned int") int abs(@Cast("const unsigned int") int a);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("unsigned long") long abs(@Cast("const unsigned long") long a);
    @Namespace("cimg_library_suffixed::cimg") public static native double abs(double a);
    @Namespace("cimg_library_suffixed::cimg") public static native float abs(float a);

    /** Return square of a value. */

    /** Return <tt>1 + log_10(x)</tt> of a value \c x. */
    @Namespace("cimg_library_suffixed::cimg") public static native int xln(int x);

    /** Return the minimum between two values. */

    /** Return the minimum between three values. */

    /** Return the minimum between four values. */

    /** Return the maximum between two values. */

    /** Return the maximum between three values. */

    /** Return the maximum between four values. */

    /** Return the sign of a value. */

    /** Return the nearest power of 2 higher than given value. */

    /** Return the sinc of a given value. */
    @Namespace("cimg_library_suffixed::cimg") public static native double sinc(double x);

    /** Return the modulo of a value.
    /**
       @param x Input value.
       @param m Modulo value.
       \note This modulo function accepts negative and floating-points modulo numbers, as well as variables of any type.
    **/
    @Namespace("cimg_library_suffixed::cimg") public static native int mod(@Cast("const bool") boolean x, @Cast("const bool") boolean m);
    @Namespace("cimg_library_suffixed::cimg") public static native int mod(byte x, byte m);
    @Namespace("cimg_library_suffixed::cimg") public static native int mod(short x, short m);
    @Namespace("cimg_library_suffixed::cimg") public static native int mod(int x, int m);
    @Namespace("cimg_library_suffixed::cimg") public static native int mod(long x, long m);

    /** Return the min-mod of two values.
    /**
       \note <i>minmod(\p a,\p b)</i> is defined to be:
       - <i>minmod(\p a,\p b) = min(\p a,\p b)</i>, if \p a and \p b have the same sign.
       - <i>minmod(\p a,\p b) = 0</i>, if \p a and \p b have different signs.
    **/

    /** Return base-2 logarithm of a value. */
    @Namespace("cimg_library_suffixed::cimg") public static native double log2(double x);

    /** Return rounded value.
    /**
       @param x Value to be rounded.
       @param y Rounding precision.
       @param rounding_type Type of rounding operation (\c 0 = nearest, \c -1 = backward, \c 1 = forward).
       @return Rounded value, having the same type as input value \c x.
    **/

    @Namespace("cimg_library_suffixed::cimg") public static native double _pythagore(double a, double b);

    /** Convert ascii character to lower case. */
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("char") byte uncase(byte x);

    /** Convert C-string to lower case. */
    @Namespace("cimg_library_suffixed::cimg") public static native void uncase(@Cast("char*const") BytePointer str);
    @Namespace("cimg_library_suffixed::cimg") public static native void uncase(@Cast("char*const") ByteBuffer str);
    @Namespace("cimg_library_suffixed::cimg") public static native void uncase(@Cast("char*const") byte[] str);

    /** Read value in a C-string.
    /**
       @param str C-string containing the float value to read.
       @return Read value.
       \note Same as <tt>std::atof()</tt> extended to manage the retrieval of fractions from C-strings, as in <em>"1/2"</em>.
    **/
    @Namespace("cimg_library_suffixed::cimg") public static native double atof(@Cast("const char*") BytePointer str);
    @Namespace("cimg_library_suffixed::cimg") public static native double atof(String str);

    /** Compare the first \p l characters of two C-strings, ignoring the case.
    /**
       @param str1 C-string.
       @param str2 C-string.
       @param l Number of characters to compare.
       @return \c 0 if the two strings are equal, something else otherwise.
       \note This function has to be defined since it is not provided by all C++-compilers (not ANSI).
    **/
    @Namespace("cimg_library_suffixed::cimg") public static native int strncasecmp(@Cast("const char*") BytePointer str1, @Cast("const char*") BytePointer str2, int l);
    @Namespace("cimg_library_suffixed::cimg") public static native int strncasecmp(String str1, String str2, int l);

    /** Compare two C-strings, ignoring the case.
    /**
       @param str1 C-string.
       @param str2 C-string.
       @return \c 0 if the two strings are equal, something else otherwise.
       \note This function has to be defined since it is not provided by all C++-compilers (not ANSI).
    **/
    @Namespace("cimg_library_suffixed::cimg") public static native int strcasecmp(@Cast("const char*") BytePointer str1, @Cast("const char*") BytePointer str2);
    @Namespace("cimg_library_suffixed::cimg") public static native int strcasecmp(String str1, String str2);

    /** Remove delimiters on the start and/or end of a C-string.
    /**
       @param [in,out] str C-string to work with (modified at output).
       @param delimiter Delimiter character code to remove.
       @param is_symmetric Tells if the removal is done only if delimiters are symmetric (both at the beginning and the end of \c s).
       @param is_iterative Tells if the removal is done if several iterations are possible.
       @return \c true if delimiters have been removed, \c false otherwise.
   **/
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("bool") boolean strpare(@Cast("char*const") BytePointer str, byte delimiter/*=' '*/, @Cast("const bool") boolean is_symmetric/*=false*/, @Cast("const bool") boolean is_iterative/*=false*/);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("bool") boolean strpare(@Cast("char*const") BytePointer str);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("bool") boolean strpare(@Cast("char*const") ByteBuffer str, byte delimiter/*=' '*/, @Cast("const bool") boolean is_symmetric/*=false*/, @Cast("const bool") boolean is_iterative/*=false*/);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("bool") boolean strpare(@Cast("char*const") ByteBuffer str);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("bool") boolean strpare(@Cast("char*const") byte[] str, byte delimiter/*=' '*/, @Cast("const bool") boolean is_symmetric/*=false*/, @Cast("const bool") boolean is_iterative/*=false*/);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("bool") boolean strpare(@Cast("char*const") byte[] str);

    /** Replace escape sequences in C-strings by their binary ascii values.
    /**
       @param [in,out] str C-string to work with (modified at output).
     **/
    @Namespace("cimg_library_suffixed::cimg") public static native void strunescape(@Cast("char*const") BytePointer str);
    @Namespace("cimg_library_suffixed::cimg") public static native void strunescape(@Cast("char*const") ByteBuffer str);
    @Namespace("cimg_library_suffixed::cimg") public static native void strunescape(@Cast("char*const") byte[] str);

    // Return a temporary string describing the size of a memory buffer.
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("const char*") BytePointer strbuffersize(@Cast("const unsigned long") long size);

    /** Return the basename of a filename. */
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("const char*") BytePointer basename(@Cast("const char*") BytePointer str);
    @Namespace("cimg_library_suffixed::cimg") public static native String basename(String str);

    // Return a random filename.
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("const char*") BytePointer filenamerand();

    // Convert filename as a Windows-style filename (short path name).
    @Namespace("cimg_library_suffixed::cimg") public static native void winformat_string(@Cast("char*const") BytePointer str);
    @Namespace("cimg_library_suffixed::cimg") public static native void winformat_string(@Cast("char*const") ByteBuffer str);
    @Namespace("cimg_library_suffixed::cimg") public static native void winformat_string(@Cast("char*const") byte[] str);

    /** Open a file.
    /**
       @param path Path of the filename to open.
       @param mode C-string describing the opening mode.
       @return Opened file.
       \note Same as <tt>std::fopen()</tt> but throw a \c CImgIOException when
       the specified file cannot be opened, instead of returning \c 0.
    **/
    @Namespace("cimg_library_suffixed::cimg") public static native FILE fopen(@Cast("const char*") BytePointer path, @Cast("const char*") BytePointer mode);
    @Namespace("cimg_library_suffixed::cimg") public static native FILE fopen(String path, String mode);

    /** Close a file.
    /**
       @param file File to close.
       @return \c 0 if file has been closed properly, something else otherwise.
       \note Same as <tt>std::fclose()</tt> but display a warning message if
       the file has not been closed properly.
    **/
    @Namespace("cimg_library_suffixed::cimg") public static native int fclose(FILE file);

    /** Get/set path to store temporary files.
    /**
       @param user_path Specified path, or \c 0 to get the path currently used.
       @param reinit_path Force path to be recalculated (may take some time).
       @return Path where temporary files can be saved.
    **/
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("const char*") BytePointer temporary_path(@Cast("const char*") BytePointer user_path/*=0*/, @Cast("const bool") boolean reinit_path/*=false*/);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("const char*") BytePointer temporary_path();
    @Namespace("cimg_library_suffixed::cimg") public static native String temporary_path(String user_path/*=0*/, @Cast("const bool") boolean reinit_path/*=false*/);

    /** Get/set path to the <i>Program Files/</i> directory (Windows only).
    /**
       @param user_path Specified path, or \c 0 to get the path currently used.
       @param reinit_path Force path to be recalculated (may take some time).
       @return Path containing the program files.
    **/
// #if cimg_OS==2 */
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("const char*") BytePointer programfiles_path(@Cast("const char*") BytePointer user_path/*=0*/, @Cast("const bool") boolean reinit_path/*=false*/);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("const char*") BytePointer programfiles_path();
    @Namespace("cimg_library_suffixed::cimg") public static native String programfiles_path(String user_path/*=0*/, @Cast("const bool") boolean reinit_path/*=false*/);
// #endif

    /** Get/set path to the ImageMagick's \c convert binary.
    /**
       @param user_path Specified path, or \c 0 to get the path currently used.
       @param reinit_path Force path to be recalculated (may take some time).
       @return Path containing the \c convert binary.
    **/
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("const char*") BytePointer imagemagick_path(@Cast("const char*") BytePointer user_path/*=0*/, @Cast("const bool") boolean reinit_path/*=false*/);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("const char*") BytePointer imagemagick_path();
    @Namespace("cimg_library_suffixed::cimg") public static native String imagemagick_path(String user_path/*=0*/, @Cast("const bool") boolean reinit_path/*=false*/);

    /** Get/set path to the GraphicsMagick's \c gm binary.
    /**
       @param user_path Specified path, or \c 0 to get the path currently used.
       @param reinit_path Force path to be recalculated (may take some time).
       @return Path containing the \c gm binary.
    **/
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("const char*") BytePointer graphicsmagick_path(@Cast("const char*") BytePointer user_path/*=0*/, @Cast("const bool") boolean reinit_path/*=false*/);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("const char*") BytePointer graphicsmagick_path();
    @Namespace("cimg_library_suffixed::cimg") public static native String graphicsmagick_path(String user_path/*=0*/, @Cast("const bool") boolean reinit_path/*=false*/);

    /** Get/set path to the XMedcon's \c medcon binary.
    /**
       @param user_path Specified path, or \c 0 to get the path currently used.
       @param reinit_path Force path to be recalculated (may take some time).
       @return Path containing the \c medcon binary.
    **/
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("const char*") BytePointer medcon_path(@Cast("const char*") BytePointer user_path/*=0*/, @Cast("const bool") boolean reinit_path/*=false*/);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("const char*") BytePointer medcon_path();
    @Namespace("cimg_library_suffixed::cimg") public static native String medcon_path(String user_path/*=0*/, @Cast("const bool") boolean reinit_path/*=false*/);

    /** Get/set path to the FFMPEG's \c ffmpeg binary.
    /**
       @param user_path Specified path, or \c 0 to get the path currently used.
       @param reinit_path Force path to be recalculated (may take some time).
       @return Path containing the \c ffmpeg binary.
    **/
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("const char*") BytePointer ffmpeg_path(@Cast("const char*") BytePointer user_path/*=0*/, @Cast("const bool") boolean reinit_path/*=false*/);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("const char*") BytePointer ffmpeg_path();
    @Namespace("cimg_library_suffixed::cimg") public static native String ffmpeg_path(String user_path/*=0*/, @Cast("const bool") boolean reinit_path/*=false*/);

    /** Get/set path to the \c gzip binary.
    /**
       @param user_path Specified path, or \c 0 to get the path currently used.
       @param reinit_path Force path to be recalculated (may take some time).
       @return Path containing the \c gzip binary.
    **/
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("const char*") BytePointer gzip_path(@Cast("const char*") BytePointer user_path/*=0*/, @Cast("const bool") boolean reinit_path/*=false*/);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("const char*") BytePointer gzip_path();
    @Namespace("cimg_library_suffixed::cimg") public static native String gzip_path(String user_path/*=0*/, @Cast("const bool") boolean reinit_path/*=false*/);

    /** Get/set path to the \c gzip binary.
    /**
       @param user_path Specified path, or \c 0 to get the path currently used.
       @param reinit_path Force path to be recalculated (may take some time).
       @return Path containing the \c gunzip binary.
    **/
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("const char*") BytePointer gunzip_path(@Cast("const char*") BytePointer user_path/*=0*/, @Cast("const bool") boolean reinit_path/*=false*/);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("const char*") BytePointer gunzip_path();
    @Namespace("cimg_library_suffixed::cimg") public static native String gunzip_path(String user_path/*=0*/, @Cast("const bool") boolean reinit_path/*=false*/);

    /** Get/set path to the \c dcraw binary.
    /**
       @param user_path Specified path, or \c 0 to get the path currently used.
       @param reinit_path Force path to be recalculated (may take some time).
       @return Path containing the \c dcraw binary.
    **/
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("const char*") BytePointer dcraw_path(@Cast("const char*") BytePointer user_path/*=0*/, @Cast("const bool") boolean reinit_path/*=false*/);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("const char*") BytePointer dcraw_path();
    @Namespace("cimg_library_suffixed::cimg") public static native String dcraw_path(String user_path/*=0*/, @Cast("const bool") boolean reinit_path/*=false*/);

    /** Get/set path to the \c wget binary.
    /**
       @param user_path Specified path, or \c 0 to get the path currently used.
       @param reinit_path Force path to be recalculated (may take some time).
       @return Path containing the \c wget binary.
    **/
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("const char*") BytePointer wget_path(@Cast("const char*") BytePointer user_path/*=0*/, @Cast("const bool") boolean reinit_path/*=false*/);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("const char*") BytePointer wget_path();
    @Namespace("cimg_library_suffixed::cimg") public static native String wget_path(String user_path/*=0*/, @Cast("const bool") boolean reinit_path/*=false*/);

    /** Get/set path to the \c curl binary.
    /**
       @param user_path Specified path, or \c 0 to get the path currently used.
       @param reinit_path Force path to be recalculated (may take some time).
       @return Path containing the \c curl binary.
    **/
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("const char*") BytePointer curl_path(@Cast("const char*") BytePointer user_path/*=0*/, @Cast("const bool") boolean reinit_path/*=false*/);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("const char*") BytePointer curl_path();
    @Namespace("cimg_library_suffixed::cimg") public static native String curl_path(String user_path/*=0*/, @Cast("const bool") boolean reinit_path/*=false*/);

    /** Split filename into two C-strings \c body and \c extension. */
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("const char*") BytePointer split_filename(@Cast("const char*") BytePointer filename, @Cast("char*const") BytePointer body/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("const char*") BytePointer split_filename(@Cast("const char*") BytePointer filename);
    @Namespace("cimg_library_suffixed::cimg") public static native String split_filename(String filename, @Cast("char*const") ByteBuffer body/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native String split_filename(String filename);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("const char*") BytePointer split_filename(@Cast("const char*") BytePointer filename, @Cast("char*const") byte[] body/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native String split_filename(String filename, @Cast("char*const") BytePointer body/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("const char*") BytePointer split_filename(@Cast("const char*") BytePointer filename, @Cast("char*const") ByteBuffer body/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native String split_filename(String filename, @Cast("char*const") byte[] body/*=0*/);

    /** Generate a numbered version of a filename. */
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("char*") BytePointer number_filename(@Cast("const char*") BytePointer filename, int number, @Cast("const unsigned int") int n, @Cast("char*const") BytePointer str);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("char*") ByteBuffer number_filename(String filename, int number, @Cast("const unsigned int") int n, @Cast("char*const") ByteBuffer str);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("char*") byte[] number_filename(@Cast("const char*") BytePointer filename, int number, @Cast("const unsigned int") int n, @Cast("char*const") byte[] str);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("char*") BytePointer number_filename(String filename, int number, @Cast("const unsigned int") int n, @Cast("char*const") BytePointer str);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("char*") ByteBuffer number_filename(@Cast("const char*") BytePointer filename, int number, @Cast("const unsigned int") int n, @Cast("char*const") ByteBuffer str);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("char*") byte[] number_filename(String filename, int number, @Cast("const unsigned int") int n, @Cast("char*const") byte[] str);

    /** Try to guess format from an image file.
    /**
       @param file Input file (can be \c 0 if \c filename is set).
       @param filename Filename, as a C-string (can be \c 0 if \c file is set).
       @return C-string containing the guessed file format, or \c 0 if nothing has been guessed.
     **/
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("const char*") BytePointer file_type(FILE file, @Cast("const char*") BytePointer filename);
    @Namespace("cimg_library_suffixed::cimg") public static native String file_type(FILE file, String filename);

    /** Read data from file.
    /**
       @param [out] ptr Pointer to memory buffer that will contain the binary data read from file.
       @param nmemb Number of elements to read.
       @param stream File to read data from.
       @return Number of read elements.
       \note Same as <tt>std::fread()</tt> but may display warning message if all elements could not be read.
    **/

    /** Write data to file.
    /**
       @param ptr Pointer to memory buffer containing the binary data to write on file.
       @param nmemb Number of elements to write.
       @param [out] stream File to write data on.
       @return Number of written elements.
       \note Similar to <tt>std::fwrite</tt> but may display warning messages if all elements could not be written.
    **/

    /** Create an empty file.
    /**
       @param file Input file (can be \c 0 if \c filename is set).
       @param filename Filename, as a C-string (can be \c 0 if \c file is set).
    **/
    @Namespace("cimg_library_suffixed::cimg") public static native void fempty(FILE file, @Cast("const char*") BytePointer filename);
    @Namespace("cimg_library_suffixed::cimg") public static native void fempty(FILE file, String filename);

    /** Load file from network as a local temporary file.
    /**
       @param filename Filename, as a C-string.
       @param [out] filename_local C-string containing the path to a local copy of \c filename.
       @return Value of \c filename_local.
       \note Use external binaries \c wget or \c curl to perform. You must have one of these tools installed
       to be able to use this function.
    **/
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("char*") BytePointer load_network_external(@Cast("const char*") BytePointer filename, @Cast("char*const") BytePointer filename_local);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("char*") ByteBuffer load_network_external(String filename, @Cast("char*const") ByteBuffer filename_local);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("char*") byte[] load_network_external(@Cast("const char*") BytePointer filename, @Cast("char*const") byte[] filename_local);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("char*") BytePointer load_network_external(String filename, @Cast("char*const") BytePointer filename_local);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("char*") ByteBuffer load_network_external(@Cast("const char*") BytePointer filename, @Cast("char*const") ByteBuffer filename_local);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("char*") byte[] load_network_external(String filename, @Cast("char*const") byte[] filename_local);

    /** Return options specified on the command line. */
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("const char*") BytePointer option(@Cast("const char*") BytePointer name, int argc, @Cast("const char*const*const") PointerPointer argv,
                                  @Cast("const char*") BytePointer defaut, @Cast("const char*") BytePointer usage, @Cast("const bool") boolean reset_static);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("const char*") BytePointer option(@Cast("const char*") BytePointer name, int argc, @Cast("const char*const*const") @ByPtrPtr BytePointer argv,
                                  @Cast("const char*") BytePointer defaut, @Cast("const char*") BytePointer usage, @Cast("const bool") boolean reset_static);
    @Namespace("cimg_library_suffixed::cimg") public static native String option(String name, int argc, @Cast("const char*const*const") @ByPtrPtr ByteBuffer argv,
                                  String defaut, String usage, @Cast("const bool") boolean reset_static);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("const char*") BytePointer option(@Cast("const char*") BytePointer name, int argc, @Cast("const char*const*const") @ByPtrPtr byte[] argv,
                                  @Cast("const char*") BytePointer defaut, @Cast("const char*") BytePointer usage, @Cast("const bool") boolean reset_static);
    @Namespace("cimg_library_suffixed::cimg") public static native String option(String name, int argc, @Cast("const char*const*const") @ByPtrPtr BytePointer argv,
                                  String defaut, String usage, @Cast("const bool") boolean reset_static);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("const char*") BytePointer option(@Cast("const char*") BytePointer name, int argc, @Cast("const char*const*const") @ByPtrPtr ByteBuffer argv,
                                  @Cast("const char*") BytePointer defaut, @Cast("const char*") BytePointer usage, @Cast("const bool") boolean reset_static);
    @Namespace("cimg_library_suffixed::cimg") public static native String option(String name, int argc, @Cast("const char*const*const") @ByPtrPtr byte[] argv,
                                  String defaut, String usage, @Cast("const bool") boolean reset_static);

    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("const char*") BytePointer option(@Cast("const char*") BytePointer name, int argc, @Cast("const char*const*const") PointerPointer argv,
                                  @Cast("const char*") BytePointer defaut, @Cast("const char*") BytePointer usage/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("const char*") BytePointer option(@Cast("const char*") BytePointer name, int argc, @Cast("const char*const*const") @ByPtrPtr BytePointer argv,
                                  @Cast("const char*") BytePointer defaut);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("const char*") BytePointer option(@Cast("const char*") BytePointer name, int argc, @Cast("const char*const*const") @ByPtrPtr BytePointer argv,
                                  @Cast("const char*") BytePointer defaut, @Cast("const char*") BytePointer usage/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native String option(String name, int argc, @Cast("const char*const*const") @ByPtrPtr ByteBuffer argv,
                                  String defaut, String usage/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native String option(String name, int argc, @Cast("const char*const*const") @ByPtrPtr ByteBuffer argv,
                                  String defaut);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("const char*") BytePointer option(@Cast("const char*") BytePointer name, int argc, @Cast("const char*const*const") @ByPtrPtr byte[] argv,
                                  @Cast("const char*") BytePointer defaut, @Cast("const char*") BytePointer usage/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("const char*") BytePointer option(@Cast("const char*") BytePointer name, int argc, @Cast("const char*const*const") @ByPtrPtr byte[] argv,
                                  @Cast("const char*") BytePointer defaut);
    @Namespace("cimg_library_suffixed::cimg") public static native String option(String name, int argc, @Cast("const char*const*const") @ByPtrPtr BytePointer argv,
                                  String defaut, String usage/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native String option(String name, int argc, @Cast("const char*const*const") @ByPtrPtr BytePointer argv,
                                  String defaut);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("const char*") BytePointer option(@Cast("const char*") BytePointer name, int argc, @Cast("const char*const*const") @ByPtrPtr ByteBuffer argv,
                                  @Cast("const char*") BytePointer defaut, @Cast("const char*") BytePointer usage/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("const char*") BytePointer option(@Cast("const char*") BytePointer name, int argc, @Cast("const char*const*const") @ByPtrPtr ByteBuffer argv,
                                  @Cast("const char*") BytePointer defaut);
    @Namespace("cimg_library_suffixed::cimg") public static native String option(String name, int argc, @Cast("const char*const*const") @ByPtrPtr byte[] argv,
                                  String defaut, String usage/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native String option(String name, int argc, @Cast("const char*const*const") @ByPtrPtr byte[] argv,
                                  String defaut);

    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("bool") boolean option(@Cast("const char*") BytePointer name, int argc, @Cast("const char*const*const") PointerPointer argv,
                           @Cast("const bool") boolean defaut, @Cast("const char*") BytePointer usage/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("bool") boolean option(@Cast("const char*") BytePointer name, int argc, @Cast("const char*const*const") @ByPtrPtr BytePointer argv,
                           @Cast("const bool") boolean defaut);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("bool") boolean option(@Cast("const char*") BytePointer name, int argc, @Cast("const char*const*const") @ByPtrPtr BytePointer argv,
                           @Cast("const bool") boolean defaut, @Cast("const char*") BytePointer usage/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("bool") boolean option(String name, int argc, @Cast("const char*const*const") @ByPtrPtr ByteBuffer argv,
                           @Cast("const bool") boolean defaut, String usage/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("bool") boolean option(String name, int argc, @Cast("const char*const*const") @ByPtrPtr ByteBuffer argv,
                           @Cast("const bool") boolean defaut);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("bool") boolean option(@Cast("const char*") BytePointer name, int argc, @Cast("const char*const*const") @ByPtrPtr byte[] argv,
                           @Cast("const bool") boolean defaut, @Cast("const char*") BytePointer usage/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("bool") boolean option(@Cast("const char*") BytePointer name, int argc, @Cast("const char*const*const") @ByPtrPtr byte[] argv,
                           @Cast("const bool") boolean defaut);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("bool") boolean option(String name, int argc, @Cast("const char*const*const") @ByPtrPtr BytePointer argv,
                           @Cast("const bool") boolean defaut, String usage/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("bool") boolean option(String name, int argc, @Cast("const char*const*const") @ByPtrPtr BytePointer argv,
                           @Cast("const bool") boolean defaut);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("bool") boolean option(@Cast("const char*") BytePointer name, int argc, @Cast("const char*const*const") @ByPtrPtr ByteBuffer argv,
                           @Cast("const bool") boolean defaut, @Cast("const char*") BytePointer usage/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("bool") boolean option(@Cast("const char*") BytePointer name, int argc, @Cast("const char*const*const") @ByPtrPtr ByteBuffer argv,
                           @Cast("const bool") boolean defaut);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("bool") boolean option(String name, int argc, @Cast("const char*const*const") @ByPtrPtr byte[] argv,
                           @Cast("const bool") boolean defaut, String usage/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("bool") boolean option(String name, int argc, @Cast("const char*const*const") @ByPtrPtr byte[] argv,
                           @Cast("const bool") boolean defaut);

    @Namespace("cimg_library_suffixed::cimg") public static native int option(@Cast("const char*") BytePointer name, int argc, @Cast("const char*const*const") PointerPointer argv,
                          int defaut, @Cast("const char*") BytePointer usage/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native int option(@Cast("const char*") BytePointer name, int argc, @Cast("const char*const*const") @ByPtrPtr BytePointer argv,
                          int defaut);
    @Namespace("cimg_library_suffixed::cimg") public static native int option(@Cast("const char*") BytePointer name, int argc, @Cast("const char*const*const") @ByPtrPtr BytePointer argv,
                          int defaut, @Cast("const char*") BytePointer usage/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native int option(String name, int argc, @Cast("const char*const*const") @ByPtrPtr ByteBuffer argv,
                          int defaut, String usage/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native int option(String name, int argc, @Cast("const char*const*const") @ByPtrPtr ByteBuffer argv,
                          int defaut);
    @Namespace("cimg_library_suffixed::cimg") public static native int option(@Cast("const char*") BytePointer name, int argc, @Cast("const char*const*const") @ByPtrPtr byte[] argv,
                          int defaut, @Cast("const char*") BytePointer usage/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native int option(@Cast("const char*") BytePointer name, int argc, @Cast("const char*const*const") @ByPtrPtr byte[] argv,
                          int defaut);
    @Namespace("cimg_library_suffixed::cimg") public static native int option(String name, int argc, @Cast("const char*const*const") @ByPtrPtr BytePointer argv,
                          int defaut, String usage/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native int option(String name, int argc, @Cast("const char*const*const") @ByPtrPtr BytePointer argv,
                          int defaut);
    @Namespace("cimg_library_suffixed::cimg") public static native int option(@Cast("const char*") BytePointer name, int argc, @Cast("const char*const*const") @ByPtrPtr ByteBuffer argv,
                          int defaut, @Cast("const char*") BytePointer usage/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native int option(@Cast("const char*") BytePointer name, int argc, @Cast("const char*const*const") @ByPtrPtr ByteBuffer argv,
                          int defaut);
    @Namespace("cimg_library_suffixed::cimg") public static native int option(String name, int argc, @Cast("const char*const*const") @ByPtrPtr byte[] argv,
                          int defaut, String usage/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native int option(String name, int argc, @Cast("const char*const*const") @ByPtrPtr byte[] argv,
                          int defaut);

    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("char") byte option(@Cast("const char*") BytePointer name, int argc, @Cast("const char*const*const") PointerPointer argv,
                           byte defaut, @Cast("const char*") BytePointer usage/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("char") byte option(@Cast("const char*") BytePointer name, int argc, @Cast("const char*const*const") @ByPtrPtr BytePointer argv,
                           byte defaut);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("char") byte option(@Cast("const char*") BytePointer name, int argc, @Cast("const char*const*const") @ByPtrPtr BytePointer argv,
                           byte defaut, @Cast("const char*") BytePointer usage/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("char") byte option(String name, int argc, @Cast("const char*const*const") @ByPtrPtr ByteBuffer argv,
                           byte defaut, String usage/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("char") byte option(String name, int argc, @Cast("const char*const*const") @ByPtrPtr ByteBuffer argv,
                           byte defaut);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("char") byte option(@Cast("const char*") BytePointer name, int argc, @Cast("const char*const*const") @ByPtrPtr byte[] argv,
                           byte defaut, @Cast("const char*") BytePointer usage/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("char") byte option(@Cast("const char*") BytePointer name, int argc, @Cast("const char*const*const") @ByPtrPtr byte[] argv,
                           byte defaut);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("char") byte option(String name, int argc, @Cast("const char*const*const") @ByPtrPtr BytePointer argv,
                           byte defaut, String usage/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("char") byte option(String name, int argc, @Cast("const char*const*const") @ByPtrPtr BytePointer argv,
                           byte defaut);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("char") byte option(@Cast("const char*") BytePointer name, int argc, @Cast("const char*const*const") @ByPtrPtr ByteBuffer argv,
                           byte defaut, @Cast("const char*") BytePointer usage/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("char") byte option(@Cast("const char*") BytePointer name, int argc, @Cast("const char*const*const") @ByPtrPtr ByteBuffer argv,
                           byte defaut);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("char") byte option(String name, int argc, @Cast("const char*const*const") @ByPtrPtr byte[] argv,
                           byte defaut, String usage/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("char") byte option(String name, int argc, @Cast("const char*const*const") @ByPtrPtr byte[] argv,
                           byte defaut);

    @Namespace("cimg_library_suffixed::cimg") public static native float option(@Cast("const char*") BytePointer name, int argc, @Cast("const char*const*const") PointerPointer argv,
                            float defaut, @Cast("const char*") BytePointer usage/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native float option(@Cast("const char*") BytePointer name, int argc, @Cast("const char*const*const") @ByPtrPtr BytePointer argv,
                            float defaut);
    @Namespace("cimg_library_suffixed::cimg") public static native float option(@Cast("const char*") BytePointer name, int argc, @Cast("const char*const*const") @ByPtrPtr BytePointer argv,
                            float defaut, @Cast("const char*") BytePointer usage/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native float option(String name, int argc, @Cast("const char*const*const") @ByPtrPtr ByteBuffer argv,
                            float defaut, String usage/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native float option(String name, int argc, @Cast("const char*const*const") @ByPtrPtr ByteBuffer argv,
                            float defaut);
    @Namespace("cimg_library_suffixed::cimg") public static native float option(@Cast("const char*") BytePointer name, int argc, @Cast("const char*const*const") @ByPtrPtr byte[] argv,
                            float defaut, @Cast("const char*") BytePointer usage/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native float option(@Cast("const char*") BytePointer name, int argc, @Cast("const char*const*const") @ByPtrPtr byte[] argv,
                            float defaut);
    @Namespace("cimg_library_suffixed::cimg") public static native float option(String name, int argc, @Cast("const char*const*const") @ByPtrPtr BytePointer argv,
                            float defaut, String usage/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native float option(String name, int argc, @Cast("const char*const*const") @ByPtrPtr BytePointer argv,
                            float defaut);
    @Namespace("cimg_library_suffixed::cimg") public static native float option(@Cast("const char*") BytePointer name, int argc, @Cast("const char*const*const") @ByPtrPtr ByteBuffer argv,
                            float defaut, @Cast("const char*") BytePointer usage/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native float option(@Cast("const char*") BytePointer name, int argc, @Cast("const char*const*const") @ByPtrPtr ByteBuffer argv,
                            float defaut);
    @Namespace("cimg_library_suffixed::cimg") public static native float option(String name, int argc, @Cast("const char*const*const") @ByPtrPtr byte[] argv,
                            float defaut, String usage/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native float option(String name, int argc, @Cast("const char*const*const") @ByPtrPtr byte[] argv,
                            float defaut);

    @Namespace("cimg_library_suffixed::cimg") public static native double option(@Cast("const char*") BytePointer name, int argc, @Cast("const char*const*const") PointerPointer argv,
                             double defaut, @Cast("const char*") BytePointer usage/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native double option(@Cast("const char*") BytePointer name, int argc, @Cast("const char*const*const") @ByPtrPtr BytePointer argv,
                             double defaut);
    @Namespace("cimg_library_suffixed::cimg") public static native double option(@Cast("const char*") BytePointer name, int argc, @Cast("const char*const*const") @ByPtrPtr BytePointer argv,
                             double defaut, @Cast("const char*") BytePointer usage/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native double option(String name, int argc, @Cast("const char*const*const") @ByPtrPtr ByteBuffer argv,
                             double defaut, String usage/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native double option(String name, int argc, @Cast("const char*const*const") @ByPtrPtr ByteBuffer argv,
                             double defaut);
    @Namespace("cimg_library_suffixed::cimg") public static native double option(@Cast("const char*") BytePointer name, int argc, @Cast("const char*const*const") @ByPtrPtr byte[] argv,
                             double defaut, @Cast("const char*") BytePointer usage/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native double option(@Cast("const char*") BytePointer name, int argc, @Cast("const char*const*const") @ByPtrPtr byte[] argv,
                             double defaut);
    @Namespace("cimg_library_suffixed::cimg") public static native double option(String name, int argc, @Cast("const char*const*const") @ByPtrPtr BytePointer argv,
                             double defaut, String usage/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native double option(String name, int argc, @Cast("const char*const*const") @ByPtrPtr BytePointer argv,
                             double defaut);
    @Namespace("cimg_library_suffixed::cimg") public static native double option(@Cast("const char*") BytePointer name, int argc, @Cast("const char*const*const") @ByPtrPtr ByteBuffer argv,
                             double defaut, @Cast("const char*") BytePointer usage/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native double option(@Cast("const char*") BytePointer name, int argc, @Cast("const char*const*const") @ByPtrPtr ByteBuffer argv,
                             double defaut);
    @Namespace("cimg_library_suffixed::cimg") public static native double option(String name, int argc, @Cast("const char*const*const") @ByPtrPtr byte[] argv,
                             double defaut, String usage/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native double option(String name, int argc, @Cast("const char*const*const") @ByPtrPtr byte[] argv,
                             double defaut);

    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("const char*") BytePointer argument(@Cast("const unsigned int") int nb, int argc, @Cast("const char*const*const") PointerPointer argv, @Cast("const unsigned int") int nb_singles/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("const char*") BytePointer argument(@Cast("const unsigned int") int nb, int argc, @Cast("const char*const*const") @ByPtrPtr BytePointer argv);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("const char*") BytePointer argument(@Cast("const unsigned int") int nb, int argc, @Cast("const char*const*const") @ByPtrPtr BytePointer argv, @Cast("const unsigned int") int nb_singles/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native String argument(@Cast("const unsigned int") int nb, int argc, @Cast("const char*const*const") @ByPtrPtr ByteBuffer argv, @Cast("const unsigned int") int nb_singles/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native String argument(@Cast("const unsigned int") int nb, int argc, @Cast("const char*const*const") @ByPtrPtr ByteBuffer argv);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("const char*") BytePointer argument(@Cast("const unsigned int") int nb, int argc, @Cast("const char*const*const") @ByPtrPtr byte[] argv, @Cast("const unsigned int") int nb_singles/*=0*/);
    @Namespace("cimg_library_suffixed::cimg") public static native @Cast("const char*") BytePointer argument(@Cast("const unsigned int") int nb, int argc, @Cast("const char*const*const") @ByPtrPtr byte[] argv);

    /** Print informations about \CImg environement variables.
    /**
       \note Output is done on the default output stream.
    **/

    // Declare LAPACK function signatures if LAPACK support is enabled.
// #ifdef cimg_use_lapack

    @Namespace("cimg_library_suffixed::cimg") public static native void getrf(@ByRef IntPointer N, FloatPointer lapA, IntPointer IPIV, @ByRef IntPointer INFO);
    @Namespace("cimg_library_suffixed::cimg") public static native void getrf(@ByRef IntBuffer N, FloatBuffer lapA, IntBuffer IPIV, @ByRef IntBuffer INFO);
    @Namespace("cimg_library_suffixed::cimg") public static native void getrf(@ByRef int[] N, float[] lapA, int[] IPIV, @ByRef int[] INFO);

    @Namespace("cimg_library_suffixed::cimg") public static native void getri(@ByRef IntPointer N, FloatPointer lapA, IntPointer IPIV, FloatPointer WORK, @ByRef IntPointer LWORK, @ByRef IntPointer INFO);
    @Namespace("cimg_library_suffixed::cimg") public static native void getri(@ByRef IntBuffer N, FloatBuffer lapA, IntBuffer IPIV, FloatBuffer WORK, @ByRef IntBuffer LWORK, @ByRef IntBuffer INFO);
    @Namespace("cimg_library_suffixed::cimg") public static native void getri(@ByRef int[] N, float[] lapA, int[] IPIV, float[] WORK, @ByRef int[] LWORK, @ByRef int[] INFO);

    @Namespace("cimg_library_suffixed::cimg") public static native void gesvd(@Cast("char*") @ByRef BytePointer JOB, @ByRef IntPointer M, @ByRef IntPointer N, FloatPointer lapA, @ByRef IntPointer MN,
                          FloatPointer lapS, FloatPointer lapU, FloatPointer lapV, FloatPointer WORK, @ByRef IntPointer LWORK, @ByRef IntPointer INFO);
    @Namespace("cimg_library_suffixed::cimg") public static native void gesvd(@Cast("char*") @ByRef ByteBuffer JOB, @ByRef IntBuffer M, @ByRef IntBuffer N, FloatBuffer lapA, @ByRef IntBuffer MN,
                          FloatBuffer lapS, FloatBuffer lapU, FloatBuffer lapV, FloatBuffer WORK, @ByRef IntBuffer LWORK, @ByRef IntBuffer INFO);
    @Namespace("cimg_library_suffixed::cimg") public static native void gesvd(@Cast("char*") @ByRef byte[] JOB, @ByRef int[] M, @ByRef int[] N, float[] lapA, @ByRef int[] MN,
                          float[] lapS, float[] lapU, float[] lapV, float[] WORK, @ByRef int[] LWORK, @ByRef int[] INFO);

    @Namespace("cimg_library_suffixed::cimg") public static native void getrs(@Cast("char*") @ByRef BytePointer TRANS, @ByRef IntPointer N, FloatPointer lapA, IntPointer IPIV, FloatPointer lapB, @ByRef IntPointer INFO);
    @Namespace("cimg_library_suffixed::cimg") public static native void getrs(@Cast("char*") @ByRef ByteBuffer TRANS, @ByRef IntBuffer N, FloatBuffer lapA, IntBuffer IPIV, FloatBuffer lapB, @ByRef IntBuffer INFO);
    @Namespace("cimg_library_suffixed::cimg") public static native void getrs(@Cast("char*") @ByRef byte[] TRANS, @ByRef int[] N, float[] lapA, int[] IPIV, float[] lapB, @ByRef int[] INFO);

    @Namespace("cimg_library_suffixed::cimg") public static native void syev(@Cast("char*") @ByRef BytePointer JOB, @Cast("char*") @ByRef BytePointer UPLO, @ByRef IntPointer N, FloatPointer lapA, FloatPointer lapW, FloatPointer WORK, @ByRef IntPointer LWORK, @ByRef IntPointer INFO);
    @Namespace("cimg_library_suffixed::cimg") public static native void syev(@Cast("char*") @ByRef ByteBuffer JOB, @Cast("char*") @ByRef ByteBuffer UPLO, @ByRef IntBuffer N, FloatBuffer lapA, FloatBuffer lapW, FloatBuffer WORK, @ByRef IntBuffer LWORK, @ByRef IntBuffer INFO);
    @Namespace("cimg_library_suffixed::cimg") public static native void syev(@Cast("char*") @ByRef byte[] JOB, @Cast("char*") @ByRef byte[] UPLO, @ByRef int[] N, float[] lapA, float[] lapW, float[] WORK, @ByRef int[] LWORK, @ByRef int[] INFO);

    @Namespace("cimg_library_suffixed::cimg") public static native void sgels(@Cast("char*") @ByRef BytePointer TRANS, @ByRef IntPointer M, @ByRef IntPointer N, @ByRef IntPointer NRHS, FloatPointer lapA, @ByRef IntPointer LDA,
                          FloatPointer lapB, @ByRef IntPointer LDB, FloatPointer WORK, @ByRef IntPointer LWORK, @ByRef IntPointer INFO);
    @Namespace("cimg_library_suffixed::cimg") public static native void sgels(@Cast("char*") @ByRef ByteBuffer TRANS, @ByRef IntBuffer M, @ByRef IntBuffer N, @ByRef IntBuffer NRHS, FloatBuffer lapA, @ByRef IntBuffer LDA,
                          FloatBuffer lapB, @ByRef IntBuffer LDB, FloatBuffer WORK, @ByRef IntBuffer LWORK, @ByRef IntBuffer INFO);
    @Namespace("cimg_library_suffixed::cimg") public static native void sgels(@Cast("char*") @ByRef byte[] TRANS, @ByRef int[] M, @ByRef int[] N, @ByRef int[] NRHS, float[] lapA, @ByRef int[] LDA,
                          float[] lapB, @ByRef int[] LDB, float[] WORK, @ByRef int[] LWORK, @ByRef int[] INFO);

// #endif

    // End of the 'cimg' namespace
  

  /*------------------------------------------------
   #
   #
   #   Definition of mathematical operators and
   #   external functions.
   #
   #
   -------------------------------------------------*/

// #define _cimg_create_ext_operators(typ)
//   template<typename T>
//   inline CImg<typename cimg::superset<T,typ>::type> operator+(const typ val, const CImg<T>& img) {
//     return img + val;
//   }
//   template<typename T>
//   inline CImg<typename cimg::superset<T,typ>::type> operator-(const typ val, const CImg<T>& img) {
//     typedef typename cimg::superset<T,typ>::type Tt;
//     return CImg<Tt>(img._width,img._height,img._depth,img._spectrum,val)-=img;
//   }
//   template<typename T>
//   inline CImg<typename cimg::superset<T,typ>::type> operator*(const typ val, const CImg<T>& img) {
//     return img*val;
//   }
//   template<typename T>
//   inline CImg<typename cimg::superset<T,typ>::type> operator/(const typ val, const CImg<T>& img) {
//     return val*img.get_invert();
//   }
//   template<typename T>
//   inline CImg<typename cimg::superset<T,typ>::type> operator&(const typ val, const CImg<T>& img) {
//     return img & val;
//   }
//   template<typename T>
//   inline CImg<typename cimg::superset<T,typ>::type> operator|(const typ val, const CImg<T>& img) {
//     return img | val;
//   }
//   template<typename T>
//   inline CImg<typename cimg::superset<T,typ>::type> operator^(const typ val, const CImg<T>& img) {
//     return img ^ val;
//   }
//   template<typename T>
//   inline bool operator==(const typ val, const CImg<T>& img) {
//     return img == val;
//   }
//   template<typename T>
//   inline bool operator!=(const typ val, const CImg<T>& img) {
//     return img != val;
//   }

  /*-----------------------------------
   #
   # Define the CImgDisplay structure
   #
   ----------------------------------*/
  /** Allow to create windows, display images on them and manage user events (keyboard, mouse and windows events).
  /**
     CImgDisplay methods rely on a low-level graphic library to perform: it can be either \b X-Window (X11, for Unix-based systems)
     or \b GDI32 (for Windows-based systems).
     If both libraries are missing, CImgDisplay will not be able to display images on screen, and will enter a minimal mode
     where warning messages will be outputed each time the program is trying to call one of the CImgDisplay method.
     <p>
     The configuration variable \c cimg_display tells about the graphic library used.
     It is set automatically by \CImg when one of these graphic libraries has been detected.
     But, you can override its value if necessary. Valid choices are:
     - 0: Disable display capabilities.
     - 1: Use \b X-Window (X11) library.
     - 2: Use \b GDI32 library.
     <p>
     Remember to link your program against \b X11 or \b GDI32 libraries if you use CImgDisplay.
  **/
  @Namespace("cimg_library_suffixed") @NoOffset public static class CImgDisplay extends Pointer {
      static { Loader.load(); }
      /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
      public CImgDisplay(Pointer p) { super(p); }
      /** Native array allocator. Access with {@link Pointer#position(long)}. */
      public CImgDisplay(long size) { super((Pointer)null); allocateArray(size); }
      private native void allocateArray(long size);
      @Override public CImgDisplay position(long position) {
          return (CImgDisplay)super.position(position);
      }
  
    public native @Cast("unsigned long") long _timer(); public native CImgDisplay _timer(long _timer);
    public native @Cast("unsigned long") long _fps_frames(); public native CImgDisplay _fps_frames(long _fps_frames);
    public native @Cast("unsigned long") long _fps_timer(); public native CImgDisplay _fps_timer(long _fps_timer);
    public native @Cast("unsigned int") int _width(); public native CImgDisplay _width(int _width);
    public native @Cast("unsigned int") int _height(); public native CImgDisplay _height(int _height);
    public native @Cast("unsigned int") int _normalization(); public native CImgDisplay _normalization(int _normalization);
    public native float _fps_fps(); public native CImgDisplay _fps_fps(float _fps_fps);
    public native float _min(); public native CImgDisplay _min(float _min);
    public native float _max(); public native CImgDisplay _max(float _max);
    public native @Cast("bool") boolean _is_fullscreen(); public native CImgDisplay _is_fullscreen(boolean _is_fullscreen);
    public native @Cast("char*") BytePointer _title(); public native CImgDisplay _title(BytePointer _title);
    public native @ByRef volatile _window_width(); public native CImgDisplay _window_width(volatile _window_width);
    public native @ByRef volatile _window_height(); public native CImgDisplay _window_height(volatile _window_height);
    public native @ByRef volatile _button(); public native CImgDisplay _button(volatile _button);
    public native @ByRef volatile _keys(int i); public native CImgDisplay _keys(int i, volatile _keys);
    @MemberGetter public native volatile _keys();
    public native @ByRef volatile _released_keys(int i); public native CImgDisplay _released_keys(int i, volatile _released_keys);
    @MemberGetter public native volatile _released_keys();
    public native @ByRef volatile _window_x(); public native CImgDisplay _window_x(volatile _window_x);
    public native @ByRef volatile _window_y(); public native CImgDisplay _window_y(volatile _window_y);
    public native @ByRef volatile _mouse_x(); public native CImgDisplay _mouse_x(volatile _mouse_x);
    public native @ByRef volatile _mouse_y(); public native CImgDisplay _mouse_y(volatile _mouse_y);
    public native @ByRef volatile _wheel(); public native CImgDisplay _wheel(volatile _wheel);
    public native @ByRef volatile _is_closed(); public native CImgDisplay _is_closed(volatile _is_closed);
    public native @ByRef volatile _is_resized(); public native CImgDisplay _is_resized(volatile _is_resized);
    public native @ByRef volatile _is_moved(); public native CImgDisplay _is_moved(volatile _is_moved);
    public native @ByRef volatile _is_event(); public native CImgDisplay _is_event(volatile _is_event);
    public native @ByRef volatile _is_keyESC(); public native CImgDisplay _is_keyESC(volatile _is_keyESC);
    public native @ByRef volatile _is_keyF1(); public native CImgDisplay _is_keyF1(volatile _is_keyF1);
    public native @ByRef volatile _is_keyF2(); public native CImgDisplay _is_keyF2(volatile _is_keyF2);
    public native @ByRef volatile _is_keyF3(); public native CImgDisplay _is_keyF3(volatile _is_keyF3);
    public native @ByRef volatile _is_keyF4(); public native CImgDisplay _is_keyF4(volatile _is_keyF4);
    public native @ByRef volatile _is_keyF5(); public native CImgDisplay _is_keyF5(volatile _is_keyF5);
    public native @ByRef volatile _is_keyF6(); public native CImgDisplay _is_keyF6(volatile _is_keyF6);
    public native @ByRef volatile _is_keyF7(); public native CImgDisplay _is_keyF7(volatile _is_keyF7);
    public native @ByRef volatile _is_keyF8(); public native CImgDisplay _is_keyF8(volatile _is_keyF8);
    public native @ByRef volatile _is_keyF9(); public native CImgDisplay _is_keyF9(volatile _is_keyF9);
    public native @ByRef volatile _is_keyF10(); public native CImgDisplay _is_keyF10(volatile _is_keyF10);
    public native @ByRef volatile _is_keyF11(); public native CImgDisplay _is_keyF11(volatile _is_keyF11);
    public native @ByRef volatile _is_keyF12(); public native CImgDisplay _is_keyF12(volatile _is_keyF12);
    public native @ByRef volatile _is_keyPAUSE(); public native CImgDisplay _is_keyPAUSE(volatile _is_keyPAUSE);
    public native @ByRef volatile _is_key1(); public native CImgDisplay _is_key1(volatile _is_key1);
    public native @ByRef volatile _is_key2(); public native CImgDisplay _is_key2(volatile _is_key2);
    public native @ByRef volatile _is_key3(); public native CImgDisplay _is_key3(volatile _is_key3);
    public native @ByRef volatile _is_key4(); public native CImgDisplay _is_key4(volatile _is_key4);
    public native @ByRef volatile _is_key5(); public native CImgDisplay _is_key5(volatile _is_key5);
    public native @ByRef volatile _is_key6(); public native CImgDisplay _is_key6(volatile _is_key6);
    public native @ByRef volatile _is_key7(); public native CImgDisplay _is_key7(volatile _is_key7);
    public native @ByRef volatile _is_key8(); public native CImgDisplay _is_key8(volatile _is_key8);
    public native @ByRef volatile _is_key9(); public native CImgDisplay _is_key9(volatile _is_key9);
    public native @ByRef volatile _is_key0(); public native CImgDisplay _is_key0(volatile _is_key0);
    public native @ByRef volatile _is_keyBACKSPACE(); public native CImgDisplay _is_keyBACKSPACE(volatile _is_keyBACKSPACE);
    public native @ByRef volatile _is_keyINSERT(); public native CImgDisplay _is_keyINSERT(volatile _is_keyINSERT);
    public native @ByRef volatile _is_keyHOME(); public native CImgDisplay _is_keyHOME(volatile _is_keyHOME);
    public native @ByRef volatile _is_keyPAGEUP(); public native CImgDisplay _is_keyPAGEUP(volatile _is_keyPAGEUP);
    public native @ByRef volatile _is_keyTAB(); public native CImgDisplay _is_keyTAB(volatile _is_keyTAB);
    public native @ByRef volatile _is_keyQ(); public native CImgDisplay _is_keyQ(volatile _is_keyQ);
    public native @ByRef volatile _is_keyW(); public native CImgDisplay _is_keyW(volatile _is_keyW);
    public native @ByRef volatile _is_keyE(); public native CImgDisplay _is_keyE(volatile _is_keyE);
    public native @ByRef volatile _is_keyR(); public native CImgDisplay _is_keyR(volatile _is_keyR);
    public native @ByRef volatile _is_keyT(); public native CImgDisplay _is_keyT(volatile _is_keyT);
    public native @ByRef volatile _is_keyY(); public native CImgDisplay _is_keyY(volatile _is_keyY);
    public native @ByRef volatile _is_keyU(); public native CImgDisplay _is_keyU(volatile _is_keyU);
    public native @ByRef volatile _is_keyI(); public native CImgDisplay _is_keyI(volatile _is_keyI);
    public native @ByRef volatile _is_keyO(); public native CImgDisplay _is_keyO(volatile _is_keyO);
    public native @ByRef volatile _is_keyP(); public native CImgDisplay _is_keyP(volatile _is_keyP);
    public native @ByRef volatile _is_keyDELETE(); public native CImgDisplay _is_keyDELETE(volatile _is_keyDELETE);
    public native @ByRef volatile _is_keyEND(); public native CImgDisplay _is_keyEND(volatile _is_keyEND);
    public native @ByRef volatile _is_keyPAGEDOWN(); public native CImgDisplay _is_keyPAGEDOWN(volatile _is_keyPAGEDOWN);
    public native @ByRef volatile _is_keyCAPSLOCK(); public native CImgDisplay _is_keyCAPSLOCK(volatile _is_keyCAPSLOCK);
    public native @ByRef volatile _is_keyA(); public native CImgDisplay _is_keyA(volatile _is_keyA);
    public native @ByRef volatile _is_keyS(); public native CImgDisplay _is_keyS(volatile _is_keyS);
    public native @ByRef volatile _is_keyD(); public native CImgDisplay _is_keyD(volatile _is_keyD);
    public native @ByRef volatile _is_keyF(); public native CImgDisplay _is_keyF(volatile _is_keyF);
    public native @ByRef volatile _is_keyG(); public native CImgDisplay _is_keyG(volatile _is_keyG);
    public native @ByRef volatile _is_keyH(); public native CImgDisplay _is_keyH(volatile _is_keyH);
    public native @ByRef volatile _is_keyJ(); public native CImgDisplay _is_keyJ(volatile _is_keyJ);
    public native @ByRef volatile _is_keyK(); public native CImgDisplay _is_keyK(volatile _is_keyK);
    public native @ByRef volatile _is_keyL(); public native CImgDisplay _is_keyL(volatile _is_keyL);
    public native @ByRef volatile _is_keyENTER(); public native CImgDisplay _is_keyENTER(volatile _is_keyENTER);
    public native @ByRef volatile _is_keySHIFTLEFT(); public native CImgDisplay _is_keySHIFTLEFT(volatile _is_keySHIFTLEFT);
    public native @ByRef volatile _is_keyZ(); public native CImgDisplay _is_keyZ(volatile _is_keyZ);
    public native @ByRef volatile _is_keyX(); public native CImgDisplay _is_keyX(volatile _is_keyX);
    public native @ByRef volatile _is_keyC(); public native CImgDisplay _is_keyC(volatile _is_keyC);
    public native @ByRef volatile _is_keyV(); public native CImgDisplay _is_keyV(volatile _is_keyV);
    public native @ByRef volatile _is_keyB(); public native CImgDisplay _is_keyB(volatile _is_keyB);
    public native @ByRef volatile _is_keyN(); public native CImgDisplay _is_keyN(volatile _is_keyN);
    public native @ByRef volatile _is_keyM(); public native CImgDisplay _is_keyM(volatile _is_keyM);
    public native @ByRef volatile _is_keySHIFTRIGHT(); public native CImgDisplay _is_keySHIFTRIGHT(volatile _is_keySHIFTRIGHT);
    public native @ByRef volatile _is_keyARROWUP(); public native CImgDisplay _is_keyARROWUP(volatile _is_keyARROWUP);
    public native @ByRef volatile _is_keyCTRLLEFT(); public native CImgDisplay _is_keyCTRLLEFT(volatile _is_keyCTRLLEFT);
    public native @ByRef volatile _is_keyAPPLEFT(); public native CImgDisplay _is_keyAPPLEFT(volatile _is_keyAPPLEFT);
    public native @ByRef volatile _is_keyALT(); public native CImgDisplay _is_keyALT(volatile _is_keyALT);
    public native @ByRef volatile _is_keySPACE(); public native CImgDisplay _is_keySPACE(volatile _is_keySPACE);
    public native @ByRef volatile _is_keyALTGR(); public native CImgDisplay _is_keyALTGR(volatile _is_keyALTGR);
    public native @ByRef volatile _is_keyAPPRIGHT(); public native CImgDisplay _is_keyAPPRIGHT(volatile _is_keyAPPRIGHT);
    public native @ByRef volatile _is_keyMENU(); public native CImgDisplay _is_keyMENU(volatile _is_keyMENU);
    public native @ByRef volatile _is_keyCTRLRIGHT(); public native CImgDisplay _is_keyCTRLRIGHT(volatile _is_keyCTRLRIGHT);
    public native @ByRef volatile _is_keyARROWLEFT(); public native CImgDisplay _is_keyARROWLEFT(volatile _is_keyARROWLEFT);
    public native @ByRef volatile _is_keyARROWDOWN(); public native CImgDisplay _is_keyARROWDOWN(volatile _is_keyARROWDOWN);
    public native @ByRef volatile _is_keyARROWRIGHT(); public native CImgDisplay _is_keyARROWRIGHT(volatile _is_keyARROWRIGHT);
    public native @ByRef volatile _is_keyPAD0(); public native CImgDisplay _is_keyPAD0(volatile _is_keyPAD0);
    public native @ByRef volatile _is_keyPAD1(); public native CImgDisplay _is_keyPAD1(volatile _is_keyPAD1);
    public native @ByRef volatile _is_keyPAD2(); public native CImgDisplay _is_keyPAD2(volatile _is_keyPAD2);
    public native @ByRef volatile _is_keyPAD3(); public native CImgDisplay _is_keyPAD3(volatile _is_keyPAD3);
    public native @ByRef volatile _is_keyPAD4(); public native CImgDisplay _is_keyPAD4(volatile _is_keyPAD4);
    public native @ByRef volatile _is_keyPAD5(); public native CImgDisplay _is_keyPAD5(volatile _is_keyPAD5);
    public native @ByRef volatile _is_keyPAD6(); public native CImgDisplay _is_keyPAD6(volatile _is_keyPAD6);
    public native @ByRef volatile _is_keyPAD7(); public native CImgDisplay _is_keyPAD7(volatile _is_keyPAD7);
    public native @ByRef volatile _is_keyPAD8(); public native CImgDisplay _is_keyPAD8(volatile _is_keyPAD8);
    public native @ByRef volatile _is_keyPAD9(); public native CImgDisplay _is_keyPAD9(volatile _is_keyPAD9);
    public native @ByRef volatile _is_keyPADADD(); public native CImgDisplay _is_keyPADADD(volatile _is_keyPADADD);
    public native @ByRef volatile _is_keyPADSUB(); public native CImgDisplay _is_keyPADSUB(volatile _is_keyPADSUB);
    public native @ByRef volatile _is_keyPADMUL(); public native CImgDisplay _is_keyPADMUL(volatile _is_keyPADMUL);
    public native @ByRef volatile _is_keyPADDIV(); public native CImgDisplay _is_keyPADDIV(volatile _is_keyPADDIV);

    //@}
    //---------------------------
    //
    /** \name Plugins */
    //@{
    //---------------------------

// #ifdef cimgdisplay_plugin
// #include cimgdisplay_plugin
// #endif
// #ifdef cimgdisplay_plugin1
// #include cimgdisplay_plugin1
// #endif
// #ifdef cimgdisplay_plugin2
// #include cimgdisplay_plugin2
// #endif
// #ifdef cimgdisplay_plugin3
// #include cimgdisplay_plugin3
// #endif
// #ifdef cimgdisplay_plugin4
// #include cimgdisplay_plugin4
// #endif
// #ifdef cimgdisplay_plugin5
// #include cimgdisplay_plugin5
// #endif
// #ifdef cimgdisplay_plugin6
// #include cimgdisplay_plugin6
// #endif
// #ifdef cimgdisplay_plugin7
// #include cimgdisplay_plugin7
// #endif
// #ifdef cimgdisplay_plugin8
// #include cimgdisplay_plugin8
// #endif

    //@}
    //--------------------------------------------------------
    //
    /** \name Constructors / Destructor / Instance Management */
    //@{
    //--------------------------------------------------------

    /** Destructor.
    /**
       \note If the associated window is visible on the screen, it is closed by the call to the destructor.
    **/

    /** Construct an empty display.
    /**
       \note Constructing an empty CImgDisplay instance does not make a window appearing on the screen, until
       display of valid data is performed.
       \par Example
       <pre>{@code
       CImgDisplay disp;  // Does actually nothing.
       ...
       disp.display(img); // Construct new window and display image in it.
       }</pre>
    **/
    public CImgDisplay() { super((Pointer)null); allocate(); }
    private native void allocate();

    /** Construct a display with specified dimensions.
    /** @param width Window width.
        @param height Window height.
        @param title Window title.
        @param normalization Normalization type (<tt>0</tt>=none, <tt>1</tt>=always, <tt>2</tt>=once, <tt>3</tt>=pixel type-dependent, see normalization()).
        @param is_fullscreen Tells if fullscreen mode is enabled.
        @param is_closed Tells if associated window is initially visible or not.
        \note A black background is initially displayed on the associated window.
    **/
    public CImgDisplay(@Cast("const unsigned int") int width, @Cast("const unsigned int") int height,
                    @Cast("const char*") BytePointer title/*=0*/, @Cast("const unsigned int") int normalization/*=3*/,
                    @Cast("const bool") boolean is_fullscreen/*=false*/, @Cast("const bool") boolean is_closed/*=false*/) { super((Pointer)null); allocate(width, height, title, normalization, is_fullscreen, is_closed); }
    private native void allocate(@Cast("const unsigned int") int width, @Cast("const unsigned int") int height,
                    @Cast("const char*") BytePointer title/*=0*/, @Cast("const unsigned int") int normalization/*=3*/,
                    @Cast("const bool") boolean is_fullscreen/*=false*/, @Cast("const bool") boolean is_closed/*=false*/);
    public CImgDisplay(@Cast("const unsigned int") int width, @Cast("const unsigned int") int height) { super((Pointer)null); allocate(width, height); }
    private native void allocate(@Cast("const unsigned int") int width, @Cast("const unsigned int") int height);
    public CImgDisplay(@Cast("const unsigned int") int width, @Cast("const unsigned int") int height,
                    String title/*=0*/, @Cast("const unsigned int") int normalization/*=3*/,
                    @Cast("const bool") boolean is_fullscreen/*=false*/, @Cast("const bool") boolean is_closed/*=false*/) { super((Pointer)null); allocate(width, height, title, normalization, is_fullscreen, is_closed); }
    private native void allocate(@Cast("const unsigned int") int width, @Cast("const unsigned int") int height,
                    String title/*=0*/, @Cast("const unsigned int") int normalization/*=3*/,
                    @Cast("const bool") boolean is_fullscreen/*=false*/, @Cast("const bool") boolean is_closed/*=false*/);

    /** Construct a display from an image.
    /** @param img Image used as a model to create the window.
        @param title Window title.
        @param normalization Normalization type (<tt>0</tt>=none, <tt>1</tt>=always, <tt>2</tt>=once, <tt>3</tt>=pixel type-dependent, see normalization()).
        @param is_fullscreen Tells if fullscreen mode is enabled.
        @param is_closed Tells if associated window is initially visible or not.
        \note The pixels of the input image are initially displayed on the associated window.
    **/

    /** Construct a display from an image list.
    /** @param list The images list to display.
        @param title Window title.
        @param normalization Normalization type (<tt>0</tt>=none, <tt>1</tt>=always, <tt>2</tt>=once, <tt>3</tt>=pixel type-dependent, see normalization()).
        @param is_fullscreen Tells if fullscreen mode is enabled.
        @param is_closed Tells if associated window is initially visible or not.
        \note All images of the list, appended along the X-axis, are initially displayed on the associated window.
    **/

    /** Construct a display as a copy of an existing one.
    /**
        @param disp Display instance to copy.
        \note The pixel buffer of the input window is initially displayed on the associated window.
    **/
    public CImgDisplay(@Const @ByRef CImgDisplay disp) { super((Pointer)null); allocate(disp); }
    private native void allocate(@Const @ByRef CImgDisplay disp);

// #if cimg_display==0

    public static native void _no_display_exception();

    /** Destructor - Empty constructor \inplace.
    /**
       \note Replace the current instance by an empty display.
    **/
    public native @ByRef CImgDisplay assign();

    /** Construct a display with specified dimensions \inplace.
    /**
    **/
    public native @ByRef CImgDisplay assign(@Cast("const unsigned int") int width, @Cast("const unsigned int") int height,
                            @Cast("const char*") BytePointer title/*=0*/, @Cast("const unsigned int") int normalization/*=3*/,
                            @Cast("const bool") boolean is_fullscreen/*=false*/, @Cast("const bool") boolean is_closed/*=false*/);
    public native @ByRef CImgDisplay assign(@Cast("const unsigned int") int width, @Cast("const unsigned int") int height);
    public native @ByRef CImgDisplay assign(@Cast("const unsigned int") int width, @Cast("const unsigned int") int height,
                            String title/*=0*/, @Cast("const unsigned int") int normalization/*=3*/,
                            @Cast("const bool") boolean is_fullscreen/*=false*/, @Cast("const bool") boolean is_closed/*=false*/);

    /** Construct a display from an image \inplace.
    /**
    **/

    /** Construct a display from an image list \inplace.
    /**
    **/

    /** Construct a display as a copy of another one \inplace.
    /**
    **/
    public native @ByRef CImgDisplay assign(@Const @ByRef CImgDisplay disp);

// #endif

    /** Return a reference to an empty display.
    /**
       \note Can be useful for writing function prototypes where one of the argument (of type CImgDisplay&)
       must have a default value.
       \par Example
       <pre>{@code
       void foo(CImgDisplay& disp=CImgDisplay::empty());
       }</pre>
    **/
    public static native @ByRef CImgDisplay empty();

// #define cimg_fitscreen(dx,dy,dz) CImgDisplay::_fitscreen(dx,dy,dz,128,-85,false),CImgDisplay::_fitscreen(dx,dy,dz,128,-85,true)
    public static native @Cast("unsigned int") int _fitscreen(@Cast("const unsigned int") int dx, @Cast("const unsigned int") int dy, @Cast("const unsigned int") int dz,
                                       int dmin, int dmax,@Cast("const bool") boolean return_y);

    //@}
    //------------------------------------------
    //
    /** \name Overloaded Operators */
    //@{
    //------------------------------------------

    /** Display image on associated window.
    /**
       \note <tt>disp = img</tt> is equivalent to <tt>disp.display(img)</tt>.
    **/

    /** Display list of images on associated window.
    /**
       \note <tt>disp = list</tt> is equivalent to <tt>disp.display(list)</tt>.
    **/

    /** Construct a display as a copy of another one \inplace.
    /**
       \note Equivalent to assign(const CImgDisplay&).
     **/
    public native @ByRef @Name("operator =") CImgDisplay put(@Const @ByRef CImgDisplay disp);

    /** Return \c false if display is empty, \c true otherwise.
    /**
       \note <tt>if (disp) { ... }</tt> is equivalent to <tt>if (!disp.is_empty()) { ... }</tt>.
    **/
    public native @Name("operator bool") boolean asBoolean();

    //@}
    //------------------------------------------
    //
    /** \name Instance Checking */
    //@{
    //------------------------------------------

    /** Return \c true if display is empty, \c false otherwise.
    /**
    **/
    public native @Cast("bool") boolean is_empty();

    /** Return \c true if display is closed (i.e. not visible on the screen), \c false otherwise.
    /**
       \note
       - When a user physically closes the associated window, the display is set to closed.
       - A closed display is not destroyed. Its associated window can be show again on the screen using show().
    **/
    public native @Cast("bool") boolean is_closed();

    /** Return \c true if associated window has been resized on the screen, \c false otherwise.
    /**
    **/
    public native @Cast("bool") boolean is_resized();

    /** Return \c true if associated window has been moved on the screen, \c false otherwise.
    /**
    **/
    public native @Cast("bool") boolean is_moved();

    /** Return \c true if any event has occured on the associated window, \c false otherwise.
    /**
    **/
    public native @Cast("bool") boolean is_event();

    /** Return \c true if current display is in fullscreen mode, \c false otherwise.
    /**
    **/
    public native @Cast("bool") boolean is_fullscreen();

    /** Return \c true if any key is being pressed on the associated window, \c false otherwise.
    /**
       \note The methods below do the same only for specific keys.
    **/
    public native @Cast("bool") boolean is_key();

    /** Return \c true if key specified by given keycode is being pressed on the associated window, \c false otherwise.
    /**
       @param keycode Keycode to test.
       \note Keycode constants are defined in the cimg namespace and are architecture-dependent. Use them to ensure
       your code stay portable (see cimg::keyESC).
       \par Example
       <pre>{@code
       CImgDisplay disp(400,400);
       while (!disp.is_closed()) {
         if (disp.key(cimg::keyTAB)) { ... }  // Equivalent to 'if (disp.is_keyTAB())'.
         disp.wait();
       }
       }</pre>
    **/
    public native @Cast("bool") boolean is_key(@Cast("const unsigned int") int keycode);

    /** Return \c true if key specified by given keycode is being pressed on the associated window, \c false otherwise.
    /**
       @param keycode C-string containing the keycode label of the key to test.
       \note Use it when the key you want to test can be dynamically set by the user.
       \par Example
       <pre>{@code
       CImgDisplay disp(400,400);
       const char *const keycode = "TAB";
       while (!disp.is_closed()) {
         if (disp.is_key(keycode)) { ... }  // Equivalent to 'if (disp.is_keyTAB())'.
         disp.wait();
       }
       }</pre>
    **/
    public native @Cast("bool") boolean is_key(@Cast("const char*") BytePointer keycode);
    public native @Cast("bool") boolean is_key(String keycode);

    /** Return \c true if specified key sequence has been typed on the associated window, \c false otherwise.
    /**
       @param keycodes_sequence Buffer of keycodes to test.
       @param length Number of keys in the \c keycodes_sequence buffer.
       @param remove_sequence Tells if the key sequence must be removed from the key history, if found.
       \note Keycode constants are defined in the cimg namespace and are architecture-dependent. Use them to ensure
       your code stay portable (see cimg::keyESC).
       \par Example
       <pre>{@code
       CImgDisplay disp(400,400);
       const unsigned int key_seq[] = { cimg::keyCTRLLEFT, cimg::keyD };
       while (!disp.is_closed()) {
         if (disp.is_key_sequence(key_seq,2)) { ... }  // Test for the 'CTRL+D' keyboard event.
         disp.wait();
       }
       }</pre>
    **/
    public native @Cast("bool") boolean is_key_sequence(@Cast("const unsigned int*const") IntPointer keycodes_sequence, @Cast("const unsigned int") int length, @Cast("const bool") boolean remove_sequence/*=false*/);
    public native @Cast("bool") boolean is_key_sequence(@Cast("const unsigned int*const") IntPointer keycodes_sequence, @Cast("const unsigned int") int length);
    public native @Cast("bool") boolean is_key_sequence(@Cast("const unsigned int*const") IntBuffer keycodes_sequence, @Cast("const unsigned int") int length, @Cast("const bool") boolean remove_sequence/*=false*/);
    public native @Cast("bool") boolean is_key_sequence(@Cast("const unsigned int*const") IntBuffer keycodes_sequence, @Cast("const unsigned int") int length);
    public native @Cast("bool") boolean is_key_sequence(@Cast("const unsigned int*const") int[] keycodes_sequence, @Cast("const unsigned int") int length, @Cast("const bool") boolean remove_sequence/*=false*/);
    public native @Cast("bool") boolean is_key_sequence(@Cast("const unsigned int*const") int[] keycodes_sequence, @Cast("const unsigned int") int length);

// #define _cimg_iskey_def(k)
//     bool is_key##k() const {
//       return _is_key##k;
//     }

    /** Return \c true if the \c ESC key is being pressed on the associated window, \c false otherwise.
    /**
       \note Similar methods exist for all keys managed by \CImg (see cimg::keyESC).
    **/
    public native @Cast("bool") boolean is_keyESC(); public native @Cast("bool") boolean is_keyF1(); public native @Cast("bool") boolean is_keyF2(); public native @Cast("bool") boolean is_keyF3();
    public native @Cast("bool") boolean is_keyF4(); public native @Cast("bool") boolean is_keyF5(); public native @Cast("bool") boolean is_keyF6(); public native @Cast("bool") boolean is_keyF7();
    public native @Cast("bool") boolean is_keyF8(); public native @Cast("bool") boolean is_keyF9(); public native @Cast("bool") boolean is_keyF10(); public native @Cast("bool") boolean is_keyF11();
    public native @Cast("bool") boolean is_keyF12(); public native @Cast("bool") boolean is_keyPAUSE(); public native @Cast("bool") boolean is_key1(); public native @Cast("bool") boolean is_key2();
    public native @Cast("bool") boolean is_key3(); public native @Cast("bool") boolean is_key4(); public native @Cast("bool") boolean is_key5(); public native @Cast("bool") boolean is_key6();
    public native @Cast("bool") boolean is_key7(); public native @Cast("bool") boolean is_key8(); public native @Cast("bool") boolean is_key9(); public native @Cast("bool") boolean is_key0();
    public native @Cast("bool") boolean is_keyBACKSPACE(); public native @Cast("bool") boolean is_keyINSERT(); public native @Cast("bool") boolean is_keyHOME();
    public native @Cast("bool") boolean is_keyPAGEUP(); public native @Cast("bool") boolean is_keyTAB(); public native @Cast("bool") boolean is_keyQ(); public native @Cast("bool") boolean is_keyW();
    public native @Cast("bool") boolean is_keyE(); public native @Cast("bool") boolean is_keyR(); public native @Cast("bool") boolean is_keyT(); public native @Cast("bool") boolean is_keyY();
    public native @Cast("bool") boolean is_keyU(); public native @Cast("bool") boolean is_keyI(); public native @Cast("bool") boolean is_keyO(); public native @Cast("bool") boolean is_keyP();
    public native @Cast("bool") boolean is_keyDELETE(); public native @Cast("bool") boolean is_keyEND(); public native @Cast("bool") boolean is_keyPAGEDOWN();
    public native @Cast("bool") boolean is_keyCAPSLOCK(); public native @Cast("bool") boolean is_keyA(); public native @Cast("bool") boolean is_keyS(); public native @Cast("bool") boolean is_keyD();
    public native @Cast("bool") boolean is_keyF(); public native @Cast("bool") boolean is_keyG(); public native @Cast("bool") boolean is_keyH(); public native @Cast("bool") boolean is_keyJ();
    public native @Cast("bool") boolean is_keyK(); public native @Cast("bool") boolean is_keyL(); public native @Cast("bool") boolean is_keyENTER();
    public native @Cast("bool") boolean is_keySHIFTLEFT(); public native @Cast("bool") boolean is_keyZ(); public native @Cast("bool") boolean is_keyX(); public native @Cast("bool") boolean is_keyC();
    public native @Cast("bool") boolean is_keyV(); public native @Cast("bool") boolean is_keyB(); public native @Cast("bool") boolean is_keyN(); public native @Cast("bool") boolean is_keyM();
    public native @Cast("bool") boolean is_keySHIFTRIGHT(); public native @Cast("bool") boolean is_keyARROWUP(); public native @Cast("bool") boolean is_keyCTRLLEFT();
    public native @Cast("bool") boolean is_keyAPPLEFT(); public native @Cast("bool") boolean is_keyALT(); public native @Cast("bool") boolean is_keySPACE(); public native @Cast("bool") boolean is_keyALTGR();
    public native @Cast("bool") boolean is_keyAPPRIGHT(); public native @Cast("bool") boolean is_keyMENU(); public native @Cast("bool") boolean is_keyCTRLRIGHT();
    public native @Cast("bool") boolean is_keyARROWLEFT(); public native @Cast("bool") boolean is_keyARROWDOWN(); public native @Cast("bool") boolean is_keyARROWRIGHT();
    public native @Cast("bool") boolean is_keyPAD0(); public native @Cast("bool") boolean is_keyPAD1(); public native @Cast("bool") boolean is_keyPAD2();
    public native @Cast("bool") boolean is_keyPAD3(); public native @Cast("bool") boolean is_keyPAD4(); public native @Cast("bool") boolean is_keyPAD5();
    public native @Cast("bool") boolean is_keyPAD6(); public native @Cast("bool") boolean is_keyPAD7(); public native @Cast("bool") boolean is_keyPAD8();
    public native @Cast("bool") boolean is_keyPAD9(); public native @Cast("bool") boolean is_keyPADADD(); public native @Cast("bool") boolean is_keyPADSUB();
    public native @Cast("bool") boolean is_keyPADMUL(); public native @Cast("bool") boolean is_keyPADDIV();

    //@}
    //------------------------------------------
    //
    /** \name Instance Characteristics */
    //@{
    //------------------------------------------

// #if cimg_display==0

    /** Return width of the screen (current resolution along the X-axis).
    /**
    **/
    public static native int screen_width();

    /** Return height of the screen (current resolution along the Y-axis).
    /**
    **/
    public static native int screen_height();

// #endif

    /** Return display width.
    /**
       \note The width of the display (i.e. the width of the pixel data buffer associated to the CImgDisplay instance)
       may be different from the actual width of the associated window.
    **/
    public native int width();

    /** Return display height.
    /**
       \note The height of the display (i.e. the height of the pixel data buffer associated to the CImgDisplay instance)
       may be different from the actual height of the associated window.
    **/
    public native int height();

    /** Return normalization type of the display.
    /**
       The normalization type tells about how the values of an input image are normalized by the CImgDisplay to be correctly displayed.
       The range of values for pixels displayed on screen is <tt>[0,255]</tt>. If the range of values of the data to display
       is different, a normalization may be required for displaying the data in a correct way.
       The normalization type can be one of:
       - \c 0: Value normalization is disabled. It is then assumed that all input data to be displayed by the CImgDisplay instance
       have values in range <tt>[0,255]</tt>.
       - \c 1: Value normalization is always performed (this is the default behavior).
       Before displaying an input image, its values will be (virtually) stretched
       in range <tt>[0,255]</tt>, so that the contrast of the displayed pixels will be maximum.
       Use this mode for images whose minimum and maximum values are not prescribed to known values (e.g. float-valued images).
       Note that when normalized versions of images are computed for display purposes, the actual values of these images are not modified.
       - \c 2: Value normalization is performed once (on the first image display), then the same normalization coefficients are kept for
       next displayed frames.
       - \c 3: Value normalization depends on the pixel type of the data to display. For integer pixel types, the normalization
       is done regarding the minimum/maximum values of the type (no normalization occurs then for <tt>unsigned char</tt>).
       For float-valued pixel types, the normalization is done regarding the minimum/maximum value of the image data instead.
    <p>
    **/
    public native @Cast("unsigned int") int normalization();

    /** Return title of the associated window as a C-string.
    /**
       \note Window title may be not visible, depending on the used window manager or if the current display is in fullscreen mode.
    **/
    public native @Cast("const char*") BytePointer title();

    /** Return width of the associated window.
    /**
       \note The width of the display (i.e. the width of the pixel data buffer associated to the CImgDisplay instance)
       may be different from the actual width of the associated window.
    **/
    public native int window_width();

    /** Return height of the associated window.
    /**
       \note The height of the display (i.e. the height of the pixel data buffer associated to the CImgDisplay instance)
       may be different from the actual height of the associated window.
    **/
    public native int window_height();

    /** Return X-coordinate of the associated window.
    /**
       \note The returned coordinate corresponds to the location of the upper-left corner of the associated window.
    **/
    public native int window_x();

    /** Return Y-coordinate of the associated window.
    /**
       \note The returned coordinate corresponds to the location of the upper-left corner of the associated window.
    **/
    public native int window_y();

    /** Return X-coordinate of the mouse pointer.
    /**
       \note
       - If the mouse pointer is outside window area, \c -1 is returned.
       - Otherwise, the returned value is in the range [0,width()-1].
    **/
    public native int mouse_x();

    /** Return Y-coordinate of the mouse pointer.
    /**
       \note
       - If the mouse pointer is outside window area, \c -1 is returned.
       - Otherwise, the returned value is in the range [0,height()-1].
    **/
    public native int mouse_y();

    /** Return current state of the mouse buttons.
    /**
       \note Three mouse buttons can be managed. If one button is pressed, its corresponding bit in the returned value is set:
       - bit \c 0 (value \c 0x1): State of the left mouse button.
       - bit \c 1 (value \c 0x2): State of the right mouse button.
       - bit \c 2 (value \c 0x4): State of the middle mouse button.
       <p>
       Several bits can be activated if more than one button are pressed at the same time.
       \par Example
       <pre>{@code
       CImgDisplay disp(400,400);
       while (!disp.is_closed()) {
         if (disp.button()&1) { // Left button clicked.
           ...
         }
         if (disp.button()&2) { // Right button clicked.
           ...
         }
         if (disp.button()&4) { // Middle button clicked.
           ...
         }
         disp.wait();
       }
       }</pre>
    **/
    public native @Cast("unsigned int") int button();

    /** Return current state of the mouse wheel.
    /**
       \note
       - The returned value can be positive or negative depending on whether the mouse wheel has been scrolled forward or backward.
       - Scrolling the wheel forward add \c 1 to the wheel value.
       - Scrolling the wheel backward substract \c 1 to the wheel value.
       - The returned value cumulates the number of forward of backward scrolls since the creation of the display, or since the
       last reset of the wheel value (using set_wheel()). It is strongly recommended to quickly reset the wheel counter
       when an action has been performed regarding the current wheel value. Otherwise, the returned wheel value may be for instance \c 0
       despite the fact that many scrolls have been done (as many in forward as in backward directions).
       \par Example
       <pre>{@code
       CImgDisplay disp(400,400);
       while (!disp.is_closed()) {
         if (disp.wheel()) {
           int counter = disp.wheel();  // Read the state of the mouse wheel.
           ...                          // Do what you want with 'counter'.
           disp.set_wheel();            // Reset the wheel value to 0.
         }
         disp.wait();
       }
       }</pre>
    **/
    public native int wheel();

    /** Return one entry from the pressed keys history.
    /**
       @param pos Indice to read from the pressed keys history (indice \c 0 corresponds to latest entry).
       @return Keycode of a pressed key or \c 0 for a released key.
       \note
       - Each CImgDisplay stores a history of the pressed keys in a buffer of size \c 128. When a new key is pressed,
       its keycode is stored in the pressed keys history. When a key is released, \c 0 is put instead.
       This means that up to the 64 last pressed keys may be read from the pressed keys history.
       When a new value is stored, the pressed keys history is shifted so that the latest entry is always
       stored at position \c 0.
       - Keycode constants are defined in the cimg namespace and are architecture-dependent. Use them to ensure
       your code stay portable (see cimg::keyESC).
    **/
    public native @Cast("unsigned int") int key(@Cast("const unsigned int") int pos/*=0*/);
    public native @Cast("unsigned int") int key();

    /** Return one entry from the released keys history.
    /**
       @param pos Indice to read from the released keys history (indice \c 0 corresponds to latest entry).
       @return Keycode of a released key or \c 0 for a pressed key.
       \note
       - Each CImgDisplay stores a history of the released keys in a buffer of size \c 128. When a new key is released,
       its keycode is stored in the pressed keys history. When a key is pressed, \c 0 is put instead.
       This means that up to the 64 last released keys may be read from the released keys history.
       When a new value is stored, the released keys history is shifted so that the latest entry is always
       stored at position \c 0.
       - Keycode constants are defined in the cimg namespace and are architecture-dependent. Use them to ensure
       your code stay portable (see cimg::keyESC).
    **/
    public native @Cast("unsigned int") int released_key(@Cast("const unsigned int") int pos/*=0*/);
    public native @Cast("unsigned int") int released_key();

    /** Return keycode corresponding to the specified string.
    /**
       \note Keycode constants are defined in the cimg namespace and are architecture-dependent. Use them to ensure
       your code stay portable (see cimg::keyESC).
       \par Example
       <pre>{@code
       const unsigned int keyTAB = CImgDisplay::keycode("TAB");  // Return cimg::keyTAB.
       }</pre>
    **/
    public static native @Cast("unsigned int") int keycode(@Cast("const char*") BytePointer keycode);
    public static native @Cast("unsigned int") int keycode(String keycode);

    /** Return the current refresh rate, in frames per second.
    /**
       \note Returns a significant value when the current instance is used to display successive frames.
       It measures the delay between successive calls to frames_per_second().
    **/
    public native float frames_per_second();

    //@}
    //---------------------------------------
    //
    /** \name Window Manipulation */
    //@{
    //---------------------------------------

// #if cimg_display==0

    /** Display image on associated window.
    /**
       @param img Input image to display.
       \note This method returns immediately.
    **/

// #endif

    /** Display list of images on associated window.
    /**
       @param list List of images to display.
       @param axis Axis used to append the images along, for the visualization (can be \c x, \c y, \c z or \c c).
       @param align Relative position of aligned images when displaying lists with images of different sizes
       (\c 0 for upper-left, \c 0.5 for centering and \c 1 for lower-right).
       \note This method returns immediately.
    **/

// #if cimg_display==0

    /** Show (closed) associated window on the screen.
    /**
       \note
       - Force the associated window of a display to be visible on the screen, even if it has been closed before.
       - Using show() on a visible display does nothing.
    **/
    public native @ByRef CImgDisplay show();

    /** Close (visible) associated window and make it disappear from the screen.
    /**
       \note
       - A closed display only means the associated window is not visible anymore. This does not mean the display has been destroyed.
       Use show() to make the associated window reappear.
       - Using close() on a closed display does nothing.
    **/
    public native @ByRef @Name("close") CImgDisplay _close();

    /** Move associated window to a new location.
    /**
       @param pos_x X-coordinate of the new window location.
       @param pos_y Y-coordinate of the new window location.
       \note Depending on the window manager behavior, this method may not succeed (no exceptions are thrown nevertheless).
    **/
    public native @ByRef CImgDisplay move(int pos_x, int pos_y);

// #endif

    /** Resize display to the size of the associated window.
    /**
       @param force_redraw Tells if the previous window content must be updated and refreshed as well.
       \note
       - Calling this method ensures that width() and window_width() become equal, as well as height() and window_height().
       - The associated window is also resized to specified dimensions.
    **/
    public native @ByRef CImgDisplay resize(@Cast("const bool") boolean force_redraw/*=true*/);
    public native @ByRef CImgDisplay resize();

// #if cimg_display==0

    /** Resize display to the specified size.
    /**
       @param width Requested display width.
       @param height Requested display height.
       @param force_redraw Tells if the previous window content must be updated and refreshed as well.
       \note The associated window is also resized to specified dimensions.
    **/
    public native @ByRef CImgDisplay resize(int width, int height, @Cast("const bool") boolean force_redraw/*=true*/);
    public native @ByRef CImgDisplay resize(int width, int height);

// #endif

    /** Resize display to the size of an input image.
    /**
       @param img Input image to take size from.
       @param force_redraw Tells if the previous window content must be resized and updated as well.
       \note
       - Calling this method ensures that width() and <tt>img.width()</tt> become equal, as well as height() and <tt>img.height()</tt>.
       - The associated window is also resized to specified dimensions.
    **/

    /** Resize display to the size of another CImgDisplay instance.
    /**
       @param disp Input display to take size from.
       @param force_redraw Tells if the previous window content must be resized and updated as well.
       \note
       - Calling this method ensures that width() and <tt>disp.width()</tt> become equal, as well as height() and <tt>disp.height()</tt>.
       - The associated window is also resized to specified dimensions.
    **/
    public native @ByRef CImgDisplay resize(@Const @ByRef CImgDisplay disp, @Cast("const bool") boolean force_redraw/*=true*/);
    public native @ByRef CImgDisplay resize(@Const @ByRef CImgDisplay disp);

    // [internal] Render pixel buffer with size (wd,hd) from source buffer of size (ws,hs).

    /** Set normalization type.
    /**
       @param normalization New normalization mode.
    **/
    public native @ByRef CImgDisplay set_normalization(@Cast("const unsigned int") int normalization);

// #if cimg_display==0

    /** Set title of the associated window.
    /**
       @param format C-string containing the format of the title, as with <tt>std::printf()</tt>.
       \warning As the first argument is a format string, it is highly recommended to write
       <pre>{@code
       disp.set_title("%s",window_title);
       }</pre>
       instead of
       <pre>{@code
       disp.set_title(window_title);
       }</pre>
       if \c window_title can be arbitrary, to prevent nasty memory access.
    **/
    public native @ByRef CImgDisplay set_title(@Cast("const char*") BytePointer format);
    public native @ByRef CImgDisplay set_title(String format);

// #endif

    /** Enable or disable fullscreen mode.
    /**
       @param is_fullscreen Tells is the fullscreen mode must be activated or not.
       @param force_redraw Tells if the previous window content must be displayed as well.
       \note
       - When the fullscreen mode is enabled, the associated window fills the entire screen but the size of the current display
       is not modified.
       - The screen resolution may be switched to fit the associated window size and ensure it appears the largest as possible.
       For X-Window (X11) users, the configuration flag \c cimg_use_xrandr has to be set to allow the screen resolution change
       (requires the X11 extensions to be enabled).
    **/
    public native @ByRef CImgDisplay set_fullscreen(@Cast("const bool") boolean is_fullscreen, @Cast("const bool") boolean force_redraw/*=true*/);
    public native @ByRef CImgDisplay set_fullscreen(@Cast("const bool") boolean is_fullscreen);

// #if cimg_display==0

    /** Toggle fullscreen mode.
    /**
       @param force_redraw Tells if the previous window content must be displayed as well.
       \note Enable fullscreen mode if it was not enabled, and disable it otherwise.
    **/
    public native @ByRef CImgDisplay toggle_fullscreen(@Cast("const bool") boolean force_redraw/*=true*/);
    public native @ByRef CImgDisplay toggle_fullscreen();

    /** Show mouse pointer.
    /**
       \note Depending on the window manager behavior, this method may not succeed (no exceptions are thrown nevertheless).
    **/
    public native @ByRef CImgDisplay show_mouse();

    /** Hide mouse pointer.
    /**
       \note Depending on the window manager behavior, this method may not succeed (no exceptions are thrown nevertheless).
    **/
    public native @ByRef CImgDisplay hide_mouse();

    /** Move mouse pointer to a specified location.
    /**
       \note Depending on the window manager behavior, this method may not succeed (no exceptions are thrown nevertheless).
    **/
    public native @ByRef CImgDisplay set_mouse(int pos_x, int pos_y);

// #endif

    /** Simulate a mouse button release event.
    /**
       \note All mouse buttons are considered released at the same time.
    **/
    public native @ByRef CImgDisplay set_button();

    /** Simulate a mouse button press or release event.
    /**
       @param button Buttons event code, where each button is associated to a single bit.
       @param is_pressed Tells if the mouse button is considered as pressed or released.
    **/
    public native @ByRef CImgDisplay set_button(@Cast("const unsigned int") int button, @Cast("const bool") boolean is_pressed/*=true*/);
    public native @ByRef CImgDisplay set_button(@Cast("const unsigned int") int button);

    /** Flush all mouse wheel events.
    /**
       \note Make wheel() to return \c 0, if called afterwards.
    **/
    public native @ByRef CImgDisplay set_wheel();

    /** Simulate a wheel event.
    /**
       @param amplitude Amplitude of the wheel scrolling to simulate.
       \note Make wheel() to return \c amplitude, if called afterwards.
    **/
    public native @ByRef CImgDisplay set_wheel(int amplitude);

    /** Flush all key events.
    /**
       \note Make key() to return \c 0, if called afterwards.
    **/
    public native @ByRef CImgDisplay set_key();

    /** Simulate a keyboard press/release event.
    /**
       @param keycode Keycode of the associated key.
       @param is_pressed Tells if the key is considered as pressed or released.
       \note Keycode constants are defined in the cimg namespace and are architecture-dependent. Use them to ensure
       your code stay portable (see cimg::keyESC).
    **/
    public native @ByRef CImgDisplay set_key(@Cast("const unsigned int") int keycode, @Cast("const bool") boolean is_pressed/*=true*/);
    public native @ByRef CImgDisplay set_key(@Cast("const unsigned int") int keycode);

    /** Flush all display events.
    /**
       \note Remove all passed events from the current display.
    **/
    public native @ByRef CImgDisplay flush();

    /** Wait for any user event occuring on the current display. */
    public native @ByRef CImgDisplay wait();

    /** Wait for a given number of milliseconds since the last call to wait().
    /**
       @param milliseconds Number of milliseconds to wait for.
       \note Similar to cimg::wait().
    **/
    public native @ByRef CImgDisplay wait(@Cast("const unsigned int") int milliseconds);

    /** Wait for any event occuring on the display \c disp1. */
    public static native void wait(@ByRef CImgDisplay disp1);

    /** Wait for any event occuring either on the display \c disp1 or \c disp2. */
    public static native void wait(@ByRef CImgDisplay disp1, @ByRef CImgDisplay disp2);

    /** Wait for any event occuring either on the display \c disp1, \c disp2 or \c disp3. */
    public static native void wait(@ByRef CImgDisplay disp1, @ByRef CImgDisplay disp2, @ByRef CImgDisplay disp3);

    /** Wait for any event occuring either on the display \c disp1, \c disp2, \c disp3 or \c disp4. */
    public static native void wait(@ByRef CImgDisplay disp1, @ByRef CImgDisplay disp2, @ByRef CImgDisplay disp3, @ByRef CImgDisplay disp4);

    /** Wait for any event occuring either on the display \c disp1, \c disp2, \c disp3, \c disp4 or \c disp5. */
    public static native void wait(@ByRef CImgDisplay disp1, @ByRef CImgDisplay disp2, @ByRef CImgDisplay disp3, @ByRef CImgDisplay disp4, @ByRef CImgDisplay disp5);

    /** Wait for any event occuring either on the display \c disp1, \c disp2, \c disp3, \c disp4, ... \c disp6. */
    public static native void wait(@ByRef CImgDisplay disp1, @ByRef CImgDisplay disp2, @ByRef CImgDisplay disp3, @ByRef CImgDisplay disp4, @ByRef CImgDisplay disp5,
                         @ByRef CImgDisplay disp6);

    /** Wait for any event occuring either on the display \c disp1, \c disp2, \c disp3, \c disp4, ... \c disp7. */
    public static native void wait(@ByRef CImgDisplay disp1, @ByRef CImgDisplay disp2, @ByRef CImgDisplay disp3, @ByRef CImgDisplay disp4, @ByRef CImgDisplay disp5,
                         @ByRef CImgDisplay disp6, @ByRef CImgDisplay disp7);

    /** Wait for any event occuring either on the display \c disp1, \c disp2, \c disp3, \c disp4, ... \c disp8. */
    public static native void wait(@ByRef CImgDisplay disp1, @ByRef CImgDisplay disp2, @ByRef CImgDisplay disp3, @ByRef CImgDisplay disp4, @ByRef CImgDisplay disp5,
                         @ByRef CImgDisplay disp6, @ByRef CImgDisplay disp7, @ByRef CImgDisplay disp8);

    /** Wait for any event occuring either on the display \c disp1, \c disp2, \c disp3, \c disp4, ... \c disp9. */
    public static native void wait(@ByRef CImgDisplay disp1, @ByRef CImgDisplay disp2, @ByRef CImgDisplay disp3, @ByRef CImgDisplay disp4, @ByRef CImgDisplay disp5,
                         @ByRef CImgDisplay disp6, @ByRef CImgDisplay disp7, @ByRef CImgDisplay disp8, @ByRef CImgDisplay disp9);

    /** Wait for any event occuring either on the display \c disp1, \c disp2, \c disp3, \c disp4, ... \c disp10. */
    public static native void wait(@ByRef CImgDisplay disp1, @ByRef CImgDisplay disp2, @ByRef CImgDisplay disp3, @ByRef CImgDisplay disp4, @ByRef CImgDisplay disp5,
                         @ByRef CImgDisplay disp6, @ByRef CImgDisplay disp7, @ByRef CImgDisplay disp8, @ByRef CImgDisplay disp9, @ByRef CImgDisplay disp10);

// #if cimg_display==0

    /** Wait for any window event occuring in any opened CImgDisplay. */
    public static native void wait_all();

    /** Render image into internal display buffer.
    /**
       @param img Input image data to render.
       \note
       - Convert image data representation into the internal display buffer (architecture-dependent structure).
       - The content of the associated window is not modified, until paint() is called.
       - Should not be used for common CImgDisplay uses, since display() is more useful.
    **/

    /** Paint internal display buffer on associated window.
    /**
       \note
       - Update the content of the associated window with the internal display buffer, e.g. after a render() call.
       - Should not be used for common CImgDisplay uses, since display() is more useful.
    **/
    public native @ByRef CImgDisplay paint();

    /** Take a snapshot of the associated window content.
    /**
       @param [out] img Output snapshot. Can be empty on input.
    **/
// #endif

    // X11-based implementation
    //--------------------------
// #if cimg_display==1

    public native @ByRef Atom _wm_window_atom(); public native CImgDisplay _wm_window_atom(Atom _wm_window_atom);
    public native @ByRef Atom _wm_protocol_atom(); public native CImgDisplay _wm_protocol_atom(Atom _wm_protocol_atom);
    public native @ByRef Window _window(); public native CImgDisplay _window(Window _window);
    public native @ByRef Window _background_window(); public native CImgDisplay _background_window(Window _background_window);
    public native @ByRef Colormap _colormap(); public native CImgDisplay _colormap(Colormap _colormap);
    public native XImage _image(); public native CImgDisplay _image(XImage _image);
    public native Pointer _data(); public native CImgDisplay _data(Pointer _data);
// #ifdef cimg_use_xshm
    public native XShmSegmentInfo _shminfo(); public native CImgDisplay _shminfo(XShmSegmentInfo _shminfo);
// #endif

    public native void _handle_events(@Const XEvent pevent);

    public static native Pointer _events_thread(Pointer arg0);

    public native void _set_colormap(@ByRef Colormap _colormap, @Cast("const unsigned int") int dim);

    public native void _map_window();

    public native void _paint(@Cast("const bool") boolean wait_expose/*=true*/);
    public native void _paint();

    public native void _init_fullscreen();

    public native void _desinit_fullscreen();

    public static native int _assign_xshm(Display dpy, XErrorEvent error);

    public native void _assign(@Cast("const unsigned int") int dimw, @Cast("const unsigned int") int dimh, @Cast("const char*") BytePointer ptitle/*=0*/,
                     @Cast("const unsigned int") int normalization_type/*=3*/,
                     @Cast("const bool") boolean fullscreen_flag/*=false*/, @Cast("const bool") boolean closed_flag/*=false*/);
    public native void _assign(@Cast("const unsigned int") int dimw, @Cast("const unsigned int") int dimh);
    public native void _assign(@Cast("const unsigned int") int dimw, @Cast("const unsigned int") int dimh, String ptitle/*=0*/,
                     @Cast("const unsigned int") int normalization_type/*=3*/,
                     @Cast("const bool") boolean fullscreen_flag/*=false*/, @Cast("const bool") boolean closed_flag/*=false*/);

    public native @ByRef CImgDisplay paint(@Cast("const bool") boolean wait_expose/*=true*/);

    // Windows-based implementation.
    //-------------------------------
// #elif cimg_display==2

    public native @Cast("bool") boolean _is_mouse_tracked(); public native CImgDisplay _is_mouse_tracked(boolean _is_mouse_tracked);
    public native @Cast("bool") boolean _is_cursor_visible(); public native CImgDisplay _is_cursor_visible(boolean _is_cursor_visible);
    public native @ByRef HANDLE _thread(); public native CImgDisplay _thread(HANDLE _thread);
    public native @ByRef HANDLE _is_created(); public native CImgDisplay _is_created(HANDLE _is_created);
    public native @ByRef HANDLE _mutex(); public native CImgDisplay _mutex(HANDLE _mutex);
    public native @ByRef CLIENTCREATESTRUCT _ccs(); public native CImgDisplay _ccs(CLIENTCREATESTRUCT _ccs);
    public native @ByRef DEVMODE _curr_mode(); public native CImgDisplay _curr_mode(DEVMODE _curr_mode);
    public native @ByRef BITMAPINFO _bmi(); public native CImgDisplay _bmi(BITMAPINFO _bmi);
    public native @ByRef HDC _hdc(); public native CImgDisplay _hdc(HDC _hdc);

    public static native @ByVal LRESULT _handle_events(@ByVal HWND window,@ByVal UINT msg,@ByVal WPARAM wParam,@ByVal LPARAM lParam);

    public native @ByRef CImgDisplay _update_window_pos();
// #endif

    //@}
  }

  /*
   #--------------------------------------
   #
   #
   #
   # Definition of the CImg<T> structure
   #
   #
   #
   #--------------------------------------
   */

  /** Class representing an image (up to 4 dimensions wide), each pixel being of type \c T.
  /**
     This is the main class of the %CImg Library. It declares and constructs
     an image, allows access to its pixel values, and is able to perform various image operations.
     <p>
     \par Image representation
     <p>
     A %CImg image is defined as an instance of the container \c CImg<T>, which contains a regular grid of pixels,
     each pixel value being of type \c T. The image grid can have up to 4 dimensions: width, height, depth
     and number of channels.
     Usually, the three first dimensions are used to describe spatial coordinates <tt>(x,y,z)</tt>, while the number of channels
     is rather used as a vector-valued dimension (it may describe the R,G,B color channels for instance).
     If you need a fifth dimension, you can use image lists \c CImgList<T> rather than simple images \c CImg<T>.
     <p>
     Thus, the \c CImg<T> class is able to represent volumetric images of vector-valued pixels,
     as well as images with less dimensions (1d scalar signal, 2d color images, ...).
     Most member functions of the class CImg<\c T> are designed to handle this maximum case of (3+1) dimensions.
     <p>
     Concerning the pixel value type \c T:
     fully supported template types are the basic C++ types: <tt>unsigned char, char, short, unsigned int, int,
     unsigned long, long, float, double, ... </tt>.
     Typically, fast image display can be done using <tt>CImg<unsigned char></tt> images,
     while complex image processing algorithms may be rather coded using <tt>CImg<float></tt> or <tt>CImg<double></tt>
     images that have floating-point pixel values. The default value for the template T is \c float.
     Using your own template types may be possible. However, you will certainly have to define the complete set
     of arithmetic and logical operators for your class.
     <p>
     \par Image structure
     <p>
     The \c CImg<T> structure contains \e six fields:
     - \c _width defines the number of \a columns of the image (size along the X-axis).
     - \c _height defines the number of \a rows of the image (size along the Y-axis).
     - \c _depth defines the number of \a slices of the image (size along the Z-axis).
     - \c _spectrum defines the number of \a channels of the image (size along the C-axis).
     - \c _data defines a \a pointer to the \a pixel \a data (of type \c T).
     - \c _is_shared is a boolean that tells if the memory buffer \c data is shared with
       another image.
     <p>
     You can access these fields publicly although it is recommended to use the dedicated functions
     width(), height(), depth(), spectrum() and ptr() to do so.
     Image dimensions are not limited to a specific range (as long as you got enough available memory).
     A value of \e 1 usually means that the corresponding dimension is \a flat.
     If one of the dimensions is \e 0, or if the data pointer is null, the image is considered as \e empty.
     Empty images should not contain any pixel data and thus, will not be processed by CImg member functions
     (a CImgInstanceException will be thrown instead).
     Pixel data are stored in memory, in a non interlaced mode (See \ref cimg_storage).
     <p>
     \par Image declaration and construction
     <p>
     Declaring an image can be done by using one of the several available constructors.
     Here is a list of the most used:
     <p>
     - Construct images from arbitrary dimensions:
         - <tt>CImg<char> img;</tt> declares an empty image.
         - <tt>CImg<unsigned char> img(128,128);</tt> declares a 128x128 greyscale image with
         \c unsigned \c char pixel values.
         - <tt>CImg<double> img(3,3);</tt> declares a 3x3 matrix with \c double coefficients.
         - <tt>CImg<unsigned char> img(256,256,1,3);</tt> declares a 256x256x1x3 (color) image
         (colors are stored as an image with three channels).
         - <tt>CImg<double> img(128,128,128);</tt> declares a 128x128x128 volumetric and greyscale image
         (with \c double pixel values).
         - <tt>CImg<> img(128,128,128,3);</tt> declares a 128x128x128 volumetric color image
         (with \c float pixels, which is the default value of the template parameter \c T).
         - \b Note: images pixels are <b>not automatically initialized to 0</b>. You may use the function \c fill() to
         do it, or use the specific constructor taking 5 parameters like this:
         <tt>CImg<> img(128,128,128,3,0);</tt> declares a 128x128x128 volumetric color image with all pixel values to 0.
     <p>
     - Construct images from filenames:
         - <tt>CImg<unsigned char> img("image.jpg");</tt> reads a JPEG color image from the file "image.jpg".
         - <tt>CImg<float> img("analyze.hdr");</tt> reads a volumetric image (ANALYZE7.5 format) from the file "analyze.hdr".
         - \b Note: You need to install <a href="http://www.imagemagick.org">ImageMagick</a>
         to be able to read common compressed image formats (JPG,PNG, ...) (See \ref cimg_files_io).
     <p>
     - Construct images from C-style arrays:
         - <tt>CImg<int> img(data_buffer,256,256);</tt> constructs a 256x256 greyscale image from a \c int* buffer
         \c data_buffer (of size 256x256=65536).
         - <tt>CImg<unsigned char> img(data_buffer,256,256,1,3,false);</tt> constructs a 256x256 color image
         from a \c unsigned \c char* buffer \c data_buffer (where R,G,B channels follow each others).
         - <tt>CImg<unsigned char> img(data_buffer,256,256,1,3,true);</tt> constructs a 256x256 color image
         from a \c unsigned \c char* buffer \c data_buffer (where R,G,B channels are multiplexed).
         <p>
         The complete list of constructors can be found <a href="#constructors">here</a>.
     <p>
     \par Most useful functions
     <p>
     The \c CImg<T> class contains a lot of functions that operates on images.
     Some of the most useful are:
     <p>
     - operator()(): allows to access or write pixel values.
     - display(): displays the image in a new window.
  **/

  /*
   #-----------------------------------------
   #
   #
   #
   # Definition of the CImgList<T> structure
   #
   #
   #
   #------------------------------------------
   */
  /** Represent a list of images CImg<T>. */ // struct CImgList<T> { ...

  /*
  #---------------------------------------------
  #
   # Completion of previously declared functions
   #
   #----------------------------------------------
  */

  /** Display a simple dialog box, and wait for the user's response.
  /**
     @param title Title of the dialog window.
     @param msg Main message displayed inside the dialog window.
     @param button1_label Label of the 1st button.
     @param button2_label Label of the 2nd button (\c 0 to hide button).
     @param button3_label Label of the 3rd button (\c 0 to hide button).
     @param button4_label Label of the 4th button (\c 0 to hide button).
     @param button5_label Label of the 5th button (\c 0 to hide button).
     @param button6_label Label of the 6th button (\c 0 to hide button).
     @param logo Image logo displayed at the left of the main message.
     @param is_centered Tells if the dialog window must be centered on the screen.
     @return Indice of clicked button (from \c 0 to \c 5), or \c -1 if the dialog window has been closed by the user.
     \note
     - Up to 6 buttons can be defined in the dialog window.
     - The function returns when a user clicked one of the button or closed the dialog window.
     - If a button text is set to 0, the corresponding button (and the followings) will not appear in the dialog box. At least one button must be specified.
  **/

  /** Display a simple dialog box, and wait for the user's response \specialization. */

  /** Evaluate math expression.
  /**
     @param expression C-string describing the formula to evaluate.
     @param x Value of the pre-defined variable \c x.
     @param y Value of the pre-defined variable \c y.
     @param z Value of the pre-defined variable \c z.
     @param c Value of the pre-defined variable \c c.
     @return Result of the formula evaluation.
     \note Set \c expression to \c 0 to keep evaluating the last specified \c expression.
     \par Example
     <pre>{@code
     const double
       res1 = cimg::eval("cos(x)^2+sin(y)^2",2,2),  // will return '1'.
       res2 = cimg::eval(0,1,1);                    // will return '1' too.
     }</pre>
  **/

  // End of cimg:: namespace


  // End of cimg_library:: namespace


/** Short alias name. */

// #ifdef _cimg_redefine_False
// #endif
// #ifdef _cimg_redefine_True
// #endif
// #ifdef _cimg_redefine_None
// #endif
// #ifdef _cimg_redefine_min
// #endif
// #ifdef _cimg_redefine_max
// #endif
// #ifdef _cimg_redefine_PI
// #endif

// #endif
// Local Variables:
// mode: c++
// End:


}
