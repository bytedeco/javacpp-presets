/*
 * Copyright (C) 2013 Samuel Audet
 *
 * This file is part of JavaCPP.
 *
 * JavaCPP is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version (subject to the "Classpath" exception
 * as provided in the LICENSE.txt file that accompanied this code).
 *
 * JavaCPP is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with JavaCPP.  If not, see <http://www.gnu.org/licenses/>.
 */

package com.googlecode.javacpp.presets;

import com.googlecode.javacpp.Parser;
import com.googlecode.javacpp.annotation.Platform;
import com.googlecode.javacpp.annotation.Properties;

/**
 *
 * @author Samuel Audet
 */
@Properties(inherit={opencv_calib3d.class, opencv_video.class}, target="com.googlecode.javacpp.opencv_legacy", value={
    @Platform(include={"<opencv2/legacy/compat.hpp>", "<opencv2/legacy/legacy.hpp>"}, link="opencv_legacy@.2.4"),
    @Platform(value="windows", link="opencv_legacy248") })
public class opencv_legacy implements Parser.InfoMapper {
    public void map(Parser.InfoMap infoMap) {
        new opencv_calib3d().map(infoMap);
        infoMap.put(new Parser.Info("CvLCMEdge", "CvLCMNode", "CvBGStatModel").complete(true))
               .put(new Parser.Info("cvCreateImageData", "cvReleaseImageData", "cvmAlloc", "cvmFree", "cvmAllocArray", "cvmFreeArray").genericArgs("void", "CvArr*"))
               .put(new Parser.Info("cvSetImageData").genericArgs("void", "CvArr*", "void*", "int"))
               .put(new Parser.Info("cvGetImageRawData").genericArgs("void", "CvArr*", "uchar**", "int*", "CvSize*"))
               .put(new Parser.Info("cvIntegralImage", "cvMultiplyAccMask").genericArgs("void", "CvArr*", "CvArr*", "CvArr*", "CvArr*"))
               .put(new Parser.Info("cvMatchContours").genericArgs("double", "void*", "void*", "int", "double"))
               .put(new Parser.Info("cvUpdateMHIByTime").genericArgs("void", "CvArr*", "CvArr*", "double", "double"))
               .put(new Parser.Info("cvAccMask", "cvSquareAccMask").genericArgs("void", "CvArr*", "CvArr*", "CvArr*"))
               .put(new Parser.Info("cvRunningAvgMask").genericArgs("void", "CvArr*", "CvArr*", "CvArr*", "double"))
               .put(new Parser.Info("cvSetHistThresh").genericArgs("void", "CvHistogram*", "float**", "int"))
               .put(new Parser.Info("cvCalcHistMask").genericArgs("void", "IplImage**", "CvArr*", "CvHistogram*", "int"))
               .put(new Parser.Info("cvCvtPixToPlane", "cvCvtPlaneToPix").genericArgs("void", "CvArr*", "CvArr*", "CvArr*", "CvArr*", "CvArr*"))
               .put(new Parser.Info("cvPseudoInv").genericArgs("double", "CvArr*", "CvArr*"))
               .put(new Parser.Info("cvContourMoments").genericArgs("void", "CvArr*", "CvMoments*"))
               .put(new Parser.Info("cvGetPtrAt").genericArgs("void*", "CvArr*", "int", "int"))
               .put(new Parser.Info("cvGetAt").genericArgs("CvScalar", "CvArr*", "int", "int"))
               .put(new Parser.Info("cvSetAt").genericArgs("void", "CvArr*", "CvScalar", "int", "int"))
               .put(new Parser.Info("cvMeanMask").genericArgs("double", "CvArr*", "CvArr*"))
               .put(new Parser.Info("cvMean_StdDevMask").genericArgs("void", "CvArr*", "CvArr*", "double*", "double*"))
               .put(new Parser.Info("cvNormMask").genericArgs("double", "CvArr*", "CvArr*", "CvArr*", "int"))
               .put(new Parser.Info("cvRemoveMemoryManager").genericArgs("void", "CvAllocFunc*", "CvFreeFunc*", "void*"))
               .put(new Parser.Info("cvCopyImage").genericArgs("void", "CvArr*", "CvArr*"))
               .put(new Parser.Info("cvReleaseMatHeader").genericArgs("void", "CvMat**"))
               .put(new Parser.Info("CV_IS_SET_ELEM_EXISTS").genericArgs("bool", "CvSetElem*"))
               .put(new Parser.Info("cvMake2DPoints", "cvMake3DPoints", "cvConvertPointsHomogenious").genericArgs("void", "CvMat*", "CvMat*"))
               .put(new Parser.Info("cvWarpPerspectiveQMatrix").genericArgs("CvMat*", "CvPoint2D32f*", "CvPoint2D32f*", "CvMat*"))
               .put(new Parser.Info("CV_STEREO_GC_OCCLUDED").translate(false));
    }
}
