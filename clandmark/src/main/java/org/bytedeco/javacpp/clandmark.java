// Targeted by JavaCPP version 1.2-SNAPSHOT

package org.bytedeco.javacpp;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

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
	public native @ByRef fl_double_t overall(); public native Timings overall(fl_double_t overall);
	/** */
	public native @ByRef fl_double_t normalizedFrame(); public native Timings normalizedFrame(fl_double_t normalizedFrame);
	/** */
	public native @ByRef fl_double_t features(); public native Timings features(fl_double_t features);
	/** */
	public native @ByRef fl_double_t maxsum(); public native Timings maxsum(fl_double_t maxsum);
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
			@ByVal fl_double_t base_window_margin_x,
			@ByVal fl_double_t base_window_margin_y
		) { super((Pointer)null); allocate(landmarksCount, edgesCount, base_window_width, base_window_height, base_window_margin_x, base_window_margin_y); }
	private native void allocate(
			int landmarksCount,
			int edgesCount,
			int base_window_width,
			int base_window_height,
			@ByVal fl_double_t base_window_margin_x,
			@ByVal fl_double_t base_window_margin_y
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
				  @ByVal fl_double_t base_window_margin_x,
				  @ByVal fl_double_t base_window_margin_y);

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
							@ByVal fl_double_t base_window_margin_x,
							@ByVal fl_double_t base_window_margin_y);

	/**
	 * \brief ~CLandmark destructor
	 */

	/**
	 * \brief Function detects landmarks within given bounding box in a given image.
	 * @param inputImage	Input image
	 * @param boundingBox	Bounding box (format: [min_x, min_y, max_x, max_y]) of object of interest (i.e. axis aligned)
	 * @param ground_truth
	 */
	public native void detect(cimg_library::CImg<unsigned char> inputImage, IntPointer boundingBox, fl_double_t ground_truth/*=0*/);
	public native void detect(cimg_library::CImg<unsigned char> inputImage, IntPointer boundingBox);
	public native void detect(cimg_library::CImg<unsigned char> inputImage, IntBuffer boundingBox, fl_double_t ground_truth/*=0*/);
	public native void detect(cimg_library::CImg<unsigned char> inputImage, IntBuffer boundingBox);
	public native void detect(cimg_library::CImg<unsigned char> inputImage, int[] boundingBox, fl_double_t ground_truth/*=0*/);
	public native void detect(cimg_library::CImg<unsigned char> inputImage, int[] boundingBox);

	/**
	 * \brief detect_optimized
	 * @param inputImage
	 * @param boundingBox
	 * @param ground_truth
	 */
	public native void detect_optimized(cimg_library::CImg<unsigned char> inputImage, IntPointer boundingBox, fl_double_t ground_truth/*=0*/);
	public native void detect_optimized(cimg_library::CImg<unsigned char> inputImage, IntPointer boundingBox);
	public native void detect_optimized(cimg_library::CImg<unsigned char> inputImage, IntBuffer boundingBox, fl_double_t ground_truth/*=0*/);
	public native void detect_optimized(cimg_library::CImg<unsigned char> inputImage, IntBuffer boundingBox);
	public native void detect_optimized(cimg_library::CImg<unsigned char> inputImage, int[] boundingBox, fl_double_t ground_truth/*=0*/);
	public native void detect_optimized(cimg_library::CImg<unsigned char> inputImage, int[] boundingBox);

	/**
	 * \brief detect_optimizedFromPool
	 * @param ground_truth
	 */
	public native void detect_optimizedFromPool(IntPointer boundingBox, fl_double_t ground_truth/*=0*/);
	public native void detect_optimizedFromPool(IntPointer boundingBox);
	public native void detect_optimizedFromPool(IntBuffer boundingBox, fl_double_t ground_truth/*=0*/);
	public native void detect_optimizedFromPool(IntBuffer boundingBox);
	public native void detect_optimizedFromPool(int[] boundingBox, fl_double_t ground_truth/*=0*/);
	public native void detect_optimizedFromPool(int[] boundingBox);

	/**
	 * \brief detect_mirrored
	 * @param inputImage
	 * @param boundingBox
	 * @param ground_truth
	 */
	public native void detect_mirrored(cimg_library::CImg<unsigned char> inputImage, IntPointer boundingBox, fl_double_t ground_truth/*=0*/);
	public native void detect_mirrored(cimg_library::CImg<unsigned char> inputImage, IntPointer boundingBox);
	public native void detect_mirrored(cimg_library::CImg<unsigned char> inputImage, IntBuffer boundingBox, fl_double_t ground_truth/*=0*/);
	public native void detect_mirrored(cimg_library::CImg<unsigned char> inputImage, IntBuffer boundingBox);
	public native void detect_mirrored(cimg_library::CImg<unsigned char> inputImage, int[] boundingBox, fl_double_t ground_truth/*=0*/);
	public native void detect_mirrored(cimg_library::CImg<unsigned char> inputImage, int[] boundingBox);

	/**
	 * \brief detect
	 * @param inputImage	normalized image frame
	 * @param ground_truth
	 */
	public native void detect_base(cimg_library::CImg<unsigned char> inputImage, IntPointer ground_truth/*=0*/);
	public native void detect_base(cimg_library::CImg<unsigned char> inputImage);
	public native void detect_base(cimg_library::CImg<unsigned char> inputImage, IntBuffer ground_truth/*=0*/);
	public native void detect_base(cimg_library::CImg<unsigned char> inputImage, int[] ground_truth/*=0*/);

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
	public native void detect_base_optimized(cimg_library::CImg<unsigned char> inputImage, IntPointer ground_truth/*=0*/);
	public native void detect_base_optimized(cimg_library::CImg<unsigned char> inputImage);
	public native void detect_base_optimized(cimg_library::CImg<unsigned char> inputImage, IntBuffer ground_truth/*=0*/);
	public native void detect_base_optimized(cimg_library::CImg<unsigned char> inputImage, int[] ground_truth/*=0*/);

	/**
	 * \brief nodemax_base
	 * @param inputImage
	 * @param ground_truth
	 */
	public native void nodemax_base(cimg_library::CImg<unsigned char> inputImage, IntPointer ground_truth/*=0*/);
	public native void nodemax_base(cimg_library::CImg<unsigned char> inputImage);
	public native void nodemax_base(cimg_library::CImg<unsigned char> inputImage, IntBuffer ground_truth/*=0*/);
	public native void nodemax_base(cimg_library::CImg<unsigned char> inputImage, int[] ground_truth/*=0*/);

	/**
	 * \brief getFeatures
	 * @param inputImage
	 * @param boundingBox
	 * @return
	 */
	public native fl_double_t getFeatures(cimg_library::CImg<unsigned char> inputImage, IntPointer boundingBox, IntPointer configuration);
	public native fl_double_t getFeatures(cimg_library::CImg<unsigned char> inputImage, IntBuffer boundingBox, IntBuffer configuration);
	public native fl_double_t getFeatures(cimg_library::CImg<unsigned char> inputImage, int[] boundingBox, int[] configuration);

	/**
	 * \brief getFeatures_base
	 * @param nf
	 * @param configuration
	 * @return
	 */
	public native fl_double_t getFeatures_base(cimg_library::CImg<unsigned char> nf, IntPointer configuration);
	public native fl_double_t getFeatures_base(cimg_library::CImg<unsigned char> nf, IntBuffer configuration);
	public native fl_double_t getFeatures_base(cimg_library::CImg<unsigned char> nf, int[] configuration);

	/**
	 * \brief getFeatures_base_optimized
	 * @param configuration
	 * @return
	 */
	public native fl_double_t getFeatures_base_optimized(IntPointer configuration);
	public native fl_double_t getFeatures_base_optimized(IntBuffer configuration);
	public native fl_double_t getFeatures_base_optimized(int[] configuration);

	/**
	 * \brief getFeatures
	 * @param configuration
	 * @return
	 */
	public native fl_double_t getFeatures(IntPointer configuration);
	public native fl_double_t getFeatures(IntBuffer configuration);
	public native fl_double_t getFeatures(int[] configuration);

	/**
	 * \brief getPsiNodes_base
	 * @param nf
	 * @param configuration
	 * @return
	 */
	public native fl_double_t getPsiNodes_base(cimg_library::CImg<unsigned char> nf, IntPointer configuration);
	public native fl_double_t getPsiNodes_base(cimg_library::CImg<unsigned char> nf, IntBuffer configuration);
	public native fl_double_t getPsiNodes_base(cimg_library::CImg<unsigned char> nf, int[] configuration);

	/**
	 * \brief getPsiNodes
	 * @param configuration
	 * @return
	 */
	public native fl_double_t getPsiNodes(IntPointer configuration);
	public native fl_double_t getPsiNodes(IntBuffer configuration);
	public native fl_double_t getPsiNodes(int[] configuration);

	/**
	 * \brief setNormalizationFactor
	 * @param factor
	 */
	public native void setNormalizationFactor(@ByVal fl_double_t factor);

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
	public native fl_double_t getLandmarks();

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
	public native void setW(fl_double_t input_w);

	/**
	 * \brief getW
	 * @return joint weight vector w, allocates memory, does not care about its freeing!
	 */
	public native fl_double_t getW();

	/**
	 * \brief setNodesW
	 * @param input_w
	 */
	public native void setNodesW(fl_double_t input_w);

	/**
	 * \brief getQvalues
	 * @return
	 */
	public native fl_double_t getQvalues();

	/**
	 * \brief getGvalues
	 * @return
	 */
	public native fl_double_t getGvalues();

	/**
	 * \brief getLossValues
	 * @return
	 */
	public native fl_double_t getLossValues(IntPointer position);
	public native fl_double_t getLossValues(IntBuffer position);
	public native fl_double_t getLossValues(int[] position);

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
	public native fl_double_t getH();

	/**
	 * \brief getHinv
	 * @return
	 */
	public native fl_double_t getHinv();

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
	public native @Cast("fl_double_t**") @StdVector PointerPointer getIntermediateResults();

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
			@ByVal(nullValue = "1.2") fl_double_t base_window_margin_x/*=1.2*/,
			@ByVal(nullValue = "1.2") fl_double_t base_window_margin_y/*=1.2*/
		) { super((Pointer)null); allocate(landmarksCount, edgesCount, base_window_width, base_window_height, base_window_margin_x, base_window_margin_y); }
	private native void allocate(
			int landmarksCount/*=8*/,
			int edgesCount/*=7*/,
			int base_window_width/*=40*/,
			int base_window_height/*=40*/,
			@ByVal(nullValue = "1.2") fl_double_t base_window_margin_x/*=1.2*/,
			@ByVal(nullValue = "1.2") fl_double_t base_window_margin_y/*=1.2*/
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
	public native cimg_library::CImg<unsigned char> getNF();

	/**
	 * \brief getNF
	 * @param img
	 * @param bbox
	 * @return
	 */
	public native cimg_library::CImg<unsigned char> getNF(cimg_library::CImg<unsigned char> img, IntPointer bbox, fl_double_t ground_truth/*=0*/);
	public native cimg_library::CImg<unsigned char> getNF(cimg_library::CImg<unsigned char> img, IntPointer bbox);
	public native cimg_library::CImg<unsigned char> getNF(cimg_library::CImg<unsigned char> img, IntBuffer bbox, fl_double_t ground_truth/*=0*/);
	public native cimg_library::CImg<unsigned char> getNF(cimg_library::CImg<unsigned char> img, IntBuffer bbox);
	public native cimg_library::CImg<unsigned char> getNF(cimg_library::CImg<unsigned char> img, int[] bbox, fl_double_t ground_truth/*=0*/);
	public native cimg_library::CImg<unsigned char> getNF(cimg_library::CImg<unsigned char> img, int[] bbox);

	/**
	 * \brief getGroundTruthNF
	 * @return
	 */
	public native IntPointer getGroundTruthNF();

	/**
	 * \brief getGroundTruth
	 * @return
	 */
	public native fl_double_t getGroundTruth();

	/**
	 * \brief getNormalizationFactor
	 * @return
	 */
	public native @ByVal fl_double_t getNormalizationFactor();

	/**
	 * \brief setLossTables
	 * @param loss_data
	 * @param landmark_id
	 */
	public native void setLossTable(fl_double_t loss_data, int landmark_id);

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
	public native @ByVal fl_double_t getScore();

}



// #endif // _FLANDMARK_H__


}
