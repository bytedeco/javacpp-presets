/*
 * Copyright (C) 2013,2014 Samuel Audet
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

package org.bytedeco.javacpp.presets;

import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Samuel Audet
 */
@Properties(inherit={opencv_calib3d.class, opencv_features2d.class, opencv_video.class, opencv_ml.class}, value={
    @Platform(include={"<opencv2/legacy/blobtrack.hpp>", "<opencv2/legacy/compat.hpp>", "<opencv2/legacy/legacy.hpp>"}, link="opencv_legacy@.2.4"),
    @Platform(value="windows", link="opencv_legacy2410") },
        target="org.bytedeco.javacpp.opencv_legacy", helper="org.bytedeco.javacpp.helper.opencv_legacy")
public class opencv_legacy implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("cv_stricmp", "cv_strnicmp", "strdup", "stricmp", "cv_stricmp", "cv_strnicmp").cppTypes())
               .put(new Info("cvCreateImageData", "cvReleaseImageData", "cvmAlloc", "cvmFree", "cvmAllocArray", "cvmFreeArray").cppTypes("void", "CvArr*"))
               .put(new Info("cvSetImageData").cppTypes("void", "CvArr*", "void*", "int"))
               .put(new Info("cvGetImageRawData").cppTypes("void", "CvArr*", "uchar**", "int*", "CvSize*"))
               .put(new Info("cvIntegralImage", "cvMultiplyAccMask").cppTypes("void", "CvArr*", "CvArr*", "CvArr*", "CvArr*"))
               .put(new Info("cvMatchContours").cppTypes("double", "void*", "void*", "int", "double"))
               .put(new Info("cvUpdateMHIByTime").cppTypes("void", "CvArr*", "CvArr*", "double", "double"))
               .put(new Info("cvAccMask", "cvSquareAccMask").cppTypes("void", "CvArr*", "CvArr*", "CvArr*"))
               .put(new Info("cvRunningAvgMask").cppTypes("void", "CvArr*", "CvArr*", "CvArr*", "double"))
               .put(new Info("cvSetHistThresh").cppTypes("void", "CvHistogram*", "float**", "int"))
               .put(new Info("cvCalcHistMask").cppTypes("void", "IplImage**", "CvArr*", "CvHistogram*", "int"))
               .put(new Info("cvCvtPixToPlane", "cvCvtPlaneToPix").cppTypes("void", "CvArr*", "CvArr*", "CvArr*", "CvArr*", "CvArr*"))
               .put(new Info("cvPseudoInv").cppTypes("double", "CvArr*", "CvArr*"))
               .put(new Info("cvContourMoments").cppTypes("void", "CvArr*", "CvMoments*"))
               .put(new Info("cvGetPtrAt").cppTypes("void*", "CvArr*", "int", "int"))
               .put(new Info("cvGetAt").cppTypes("CvScalar", "CvArr*", "int", "int"))
               .put(new Info("cvSetAt").cppTypes("void", "CvArr*", "CvScalar", "int", "int"))
               .put(new Info("cvMeanMask").cppTypes("double", "CvArr*", "CvArr*"))
               .put(new Info("cvMean_StdDevMask").cppTypes("void", "CvArr*", "CvArr*", "double*", "double*"))
               .put(new Info("cvNormMask").cppTypes("double", "CvArr*", "CvArr*", "CvArr*", "int"))
               .put(new Info("cvRemoveMemoryManager").cppTypes("void", "CvAllocFunc", "CvFreeFunc", "void*"))
               .put(new Info("cvCopyImage").cppTypes("void", "CvArr*", "CvArr*"))
               .put(new Info("cvReleaseMatHeader").cppTypes("void", "CvMat**"))
               .put(new Info("cvQueryHistValue_1D").cppTypes("float", "CvHistogram*", "int"))
               .put(new Info("cvQueryHistValue_2D").cppTypes("float", "CvHistogram*", "int", "int"))
               .put(new Info("cvQueryHistValue_3D").cppTypes("float", "CvHistogram*", "int", "int", "int"))
               .put(new Info("cvQueryHistValue_nD").cppTypes("float", "CvHistogram*", "int*"))
               .put(new Info("cvGetHistValue_1D").cppTypes("void*", "CvHistogram*", "int"))
               .put(new Info("cvGetHistValue_2D").cppTypes("void*", "CvHistogram*", "int", "int"))
               .put(new Info("cvGetHistValue_3D").cppTypes("void*", "CvHistogram*", "int", "int", "int"))
               .put(new Info("cvGetHistValue_nD").cppTypes("void*", "CvHistogram*", "int*"))
               .put(new Info("CV_IS_SET_ELEM_EXISTS").cppTypes("bool", "CvSetElem*"))
               .put(new Info("cvMake2DPoints", "cvMake3DPoints", "cvConvertPointsHomogenious").cppTypes("void", "CvMat*", "CvMat*"))
               .put(new Info("cvWarpPerspectiveQMatrix").cppTypes("CvMat*", "CvPoint2D32f*", "CvPoint2D32f*", "CvMat*"))
               .put(new Info("CvImgObsInfo").base("AbstractCvImgObsInfo"))
               .put(new Info("CvEHMM").base("AbstractCvEHMM"))
               .put(new Info("CvGLCM").base("AbstractCvGLCM"))
               .put(new Info("CvFaceTracker").base("AbstractCvFaceTracker"))
               .put(new Info("CvConDensation").base("AbstractCvConDensation"))
               .put(new Info("CvBGStatModel").base("AbstractCvBGStatModel"))
               .put(new Info("CvBGCodeBookModel").base("AbstractCvBGCodeBookModel"))
               .put(new Info("CvImageDrawer", "cvCreateBlobTrackerMS1", "cvCreateBlobTrackerMS2",
                             "cvCreateBlobTrackerMS1ByList", "cvCreateBlobTrackerLHR", "cvCreateBlobTrackerLHRS",
                             "cvCreateTracks_One", "cvCreateTracks_Same", "cvCreateTracks_AreaErr",
                             "cvCreateProbS", "cvCreateProbMG", "cvCreateProbMG2", "cvCreateProbHist", "cvCreateProb", "cvMSERParams", "cvExtractMSER",
                             "cvCalcContoursCorrespondence", "cvMorphContours", "cvFindFace", "cvPostBoostingFindFace",
                             "cv::RTreeClassifier::safeSignatureAlloc", "cv::OneWayDescriptorBase::ConvertDescriptorsArrayToTree",
                             "cv::PlanarObjectDetector::getModelROI", "cv::PlanarObjectDetector::getDetector", "cv::PlanarObjectDetector::getClassifier").skip())
               .put(new Info("CV_STEREO_GC_OCCLUDED").translate(false));
    }
}
