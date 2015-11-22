// Targeted by JavaCPP version 1.2-SNAPSHOT

package org.bytedeco.javacpp;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.opencv_core.*;

public class opencv_ml extends org.bytedeco.javacpp.helper.opencv_ml {
    static { Loader.load(); }

// Parsed from <opencv2/ml.hpp>

/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (C) 2014, Itseez Inc, all rights reserved.
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

// #ifndef __OPENCV_ML_HPP__
// #define __OPENCV_ML_HPP__

// #ifdef __cplusplus
// #  include "opencv2/core.hpp"
// #endif

// #ifdef __cplusplus

// #include <float.h>
// #include <map>
// #include <iostream>

/**
  \defgroup ml Machine Learning
  <p>
  The Machine Learning Library (MLL) is a set of classes and functions for statistical
  classification, regression, and clustering of data.
  <p>
  Most of the classification and regression algorithms are implemented as C++ classes. As the
  algorithms have different sets of features (like an ability to handle missing measurements or
  categorical input variables), there is a little common ground between the classes. This common
  ground is defined by the class cv::ml::StatModel that all the other ML classes are derived from.
  <p>
  See detailed overview here: \ref ml_intro.
 */

/** \addtogroup ml
 *  \{
<p>
/** \brief Variable types */
/** enum cv::ml::VariableTypes */
public static final int
    /** same as VAR_ORDERED */
    VAR_NUMERICAL    = 0,
    /** ordered variables */
    VAR_ORDERED      = 0,
    /** categorical variables */
    VAR_CATEGORICAL  = 1;

/** \brief %Error types */
/** enum cv::ml::ErrorTypes */
public static final int
    TEST_ERROR = 0,
    TRAIN_ERROR = 1;

/** \brief Sample types */
/** enum cv::ml::SampleTypes */
public static final int
    /** each training sample is a row of samples */
    ROW_SAMPLE = 0,
    /** each training sample occupies a column of samples */
    COL_SAMPLE = 1;

/** \brief The structure represents the logarithmic grid range of statmodel parameters.
<p>
It is used for optimizing statmodel accuracy by varying model parameters, the accuracy estimate
being computed by cross-validation.
 */
@Namespace("cv::ml") @NoOffset public static class ParamGrid extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ParamGrid(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public ParamGrid(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public ParamGrid position(int position) {
        return (ParamGrid)super.position(position);
    }

    /** \brief Default constructor */
    public ParamGrid() { super((Pointer)null); allocate(); }
    private native void allocate();
    /** \brief Constructor with parameters */
    public ParamGrid(double _minVal, double _maxVal, double _logStep) { super((Pointer)null); allocate(_minVal, _maxVal, _logStep); }
    private native void allocate(double _minVal, double _maxVal, double _logStep);

    /** Minimum value of the statmodel parameter. Default value is 0. */
    public native double minVal(); public native ParamGrid minVal(double minVal);
    /** Maximum value of the statmodel parameter. Default value is 0. */
    public native double maxVal(); public native ParamGrid maxVal(double maxVal);
    /** \brief Logarithmic step for iterating the statmodel parameter.
    <p>
    The grid determines the following iteration sequence of the statmodel parameter values:
    \f[(minVal, minVal*step, minVal*{step}^2, \dots,  minVal*{logStep}^n),\f]
    where \f$n\f$ is the maximal index satisfying
    \f[\texttt{minVal} * \texttt{logStep} ^n <  \texttt{maxVal}\f]
    The grid is logarithmic, so logStep must always be greater then 1. Default value is 1.
    */
    public native double logStep(); public native ParamGrid logStep(double logStep);
}

/** \brief Class encapsulating training data.
<p>
Please note that the class only specifies the interface of training data, but not implementation.
All the statistical model classes in _ml_ module accepts Ptr\<TrainData\> as parameter. In other
words, you can create your own class derived from TrainData and pass smart pointer to the instance
of this class into StatModel::train.
<p>
\sa \ref ml_intro_data
 */
@Namespace("cv::ml") public static class TrainData extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TrainData(Pointer p) { super(p); }

    public static native float missingValue();

    public native int getLayout();
    public native int getNTrainSamples();
    public native int getNTestSamples();
    public native int getNSamples();
    public native int getNVars();
    public native int getNAllVars();

    public native void getSample(@ByVal Mat varIdx, int sidx, FloatPointer buf);
    public native void getSample(@ByVal Mat varIdx, int sidx, FloatBuffer buf);
    public native void getSample(@ByVal Mat varIdx, int sidx, float[] buf);
    public native @ByVal Mat getSamples();
    public native @ByVal Mat getMissing();

    /** \brief Returns matrix of train samples
    <p>
    @param layout The requested layout. If it's different from the initial one, the matrix is
        transposed. See ml::SampleTypes.
    @param compressSamples if true, the function returns only the training samples (specified by
        sampleIdx)
    @param compressVars if true, the function returns the shorter training samples, containing only
        the active variables.
    <p>
    In current implementation the function tries to avoid physical data copying and returns the
    matrix stored inside TrainData (unless the transposition or compression is needed).
     */
    public native @ByVal Mat getTrainSamples(int layout/*=cv::ml::ROW_SAMPLE*/,
                                    @Cast("bool") boolean compressSamples/*=true*/,
                                    @Cast("bool") boolean compressVars/*=true*/);
    public native @ByVal Mat getTrainSamples();

    /** \brief Returns the vector of responses
    <p>
    The function returns ordered or the original categorical responses. Usually it's used in
    regression algorithms.
     */
    public native @ByVal Mat getTrainResponses();

    /** \brief Returns the vector of normalized categorical responses
    <p>
    The function returns vector of responses. Each response is integer from {@code 0} to {@code <number of
    classes>-1}. The actual label value can be retrieved then from the class label vector, see
    TrainData::getClassLabels.
     */
    public native @ByVal Mat getTrainNormCatResponses();
    public native @ByVal Mat getTestResponses();
    public native @ByVal Mat getTestNormCatResponses();
    public native @ByVal Mat getResponses();
    public native @ByVal Mat getNormCatResponses();
    public native @ByVal Mat getSampleWeights();
    public native @ByVal Mat getTrainSampleWeights();
    public native @ByVal Mat getTestSampleWeights();
    public native @ByVal Mat getVarIdx();
    public native @ByVal Mat getVarType();
    public native int getResponseType();
    public native @ByVal Mat getTrainSampleIdx();
    public native @ByVal Mat getTestSampleIdx();
    public native void getValues(int vi, @ByVal Mat sidx, FloatPointer values);
    public native void getValues(int vi, @ByVal Mat sidx, FloatBuffer values);
    public native void getValues(int vi, @ByVal Mat sidx, float[] values);
    public native void getNormCatValues(int vi, @ByVal Mat sidx, IntPointer values);
    public native void getNormCatValues(int vi, @ByVal Mat sidx, IntBuffer values);
    public native void getNormCatValues(int vi, @ByVal Mat sidx, int[] values);
    public native @ByVal Mat getDefaultSubstValues();

    public native int getCatCount(int vi);

    /** \brief Returns the vector of class labels
    <p>
    The function returns vector of unique labels occurred in the responses.
     */
    public native @ByVal Mat getClassLabels();

    public native @ByVal Mat getCatOfs();
    public native @ByVal Mat getCatMap();

    /** \brief Splits the training data into the training and test parts
    \sa TrainData::setTrainTestSplitRatio
     */
    public native void setTrainTestSplit(int count, @Cast("bool") boolean shuffle/*=true*/);
    public native void setTrainTestSplit(int count);

    /** \brief Splits the training data into the training and test parts
    <p>
    The function selects a subset of specified relative size and then returns it as the training
    set. If the function is not called, all the data is used for training. Please, note that for
    each of TrainData::getTrain\* there is corresponding TrainData::getTest\*, so that the test
    subset can be retrieved and processed as well.
    \sa TrainData::setTrainTestSplit
     */
    public native void setTrainTestSplitRatio(double ratio, @Cast("bool") boolean shuffle/*=true*/);
    public native void setTrainTestSplitRatio(double ratio);
    public native void shuffleTrainTest();

    public static native @ByVal Mat getSubVector(@Const @ByRef Mat vec, @Const @ByRef Mat idx);

    /** \brief Reads the dataset from a .csv file and returns the ready-to-use training data.
    <p>
    @param filename The input file name
    @param headerLineCount The number of lines in the beginning to skip; besides the header, the
        function also skips empty lines and lines staring with {@code #}
    @param responseStartIdx Index of the first output variable. If -1, the function considers the
        last variable as the response
    @param responseEndIdx Index of the last output variable + 1. If -1, then there is single
        response variable at responseStartIdx.
    @param varTypeSpec The optional text string that specifies the variables' types. It has the
        format {@code ord[n1-n2,n3,n4-n5,...]cat[n6,n7-n8,...]}. That is, variables from {@code n1 to n2}
        (inclusive range), {@code n3}, {@code n4 to n5} ... are considered ordered and {@code n6}, {@code n7 to n8} ... are
        considered as categorical. The range {@code [n1..n2] + [n3] + [n4..n5] + ... + [n6] + [n7..n8]}
        should cover all the variables. If varTypeSpec is not specified, then algorithm uses the
        following rules:
        - all input variables are considered ordered by default. If some column contains has non-
          numerical values, e.g. 'apple', 'pear', 'apple', 'apple', 'mango', the corresponding
          variable is considered categorical.
        - if there are several output variables, they are all considered as ordered. Error is
          reported when non-numerical values are used.
        - if there is a single output variable, then if its values are non-numerical or are all
          integers, then it's considered categorical. Otherwise, it's considered ordered.
    @param delimiter The character used to separate values in each line.
    @param missch The character used to specify missing measurements. It should not be a digit.
        Although it's a non-numerical value, it surely does not affect the decision of whether the
        variable ordered or categorical.
     */
    public static native @Ptr TrainData loadFromCSV(@Str BytePointer filename,
                                          int headerLineCount,
                                          int responseStartIdx/*=-1*/,
                                          int responseEndIdx/*=-1*/,
                                          @Str BytePointer varTypeSpec/*=cv::String()*/,
                                          @Cast("char") byte delimiter/*=','*/,
                                          @Cast("char") byte missch/*='?'*/);
    public static native @Ptr TrainData loadFromCSV(@Str BytePointer filename,
                                          int headerLineCount);
    public static native @Ptr TrainData loadFromCSV(@Str String filename,
                                          int headerLineCount,
                                          int responseStartIdx/*=-1*/,
                                          int responseEndIdx/*=-1*/,
                                          @Str String varTypeSpec/*=cv::String()*/,
                                          @Cast("char") byte delimiter/*=','*/,
                                          @Cast("char") byte missch/*='?'*/);
    public static native @Ptr TrainData loadFromCSV(@Str String filename,
                                          int headerLineCount);

    /** \brief Creates training data from in-memory arrays.
    <p>
    @param samples matrix of samples. It should have CV_32F type.
    @param layout see ml::SampleTypes.
    @param responses matrix of responses. If the responses are scalar, they should be stored as a
        single row or as a single column. The matrix should have type CV_32F or CV_32S (in the
        former case the responses are considered as ordered by default; in the latter case - as
        categorical)
    @param varIdx vector specifying which variables to use for training. It can be an integer vector
        (CV_32S) containing 0-based variable indices or byte vector (CV_8U) containing a mask of
        active variables.
    @param sampleIdx vector specifying which samples to use for training. It can be an integer
        vector (CV_32S) containing 0-based sample indices or byte vector (CV_8U) containing a mask
        of training samples.
    @param sampleWeights optional vector with weights for each sample. It should have CV_32F type.
    @param varType optional vector of type CV_8U and size {@code <number_of_variables_in_samples> +
        <number_of_variables_in_responses>}, containing types of each input and output variable. See
        ml::VariableTypes.
     */
    public static native @Ptr TrainData create(@ByVal Mat samples, int layout, @ByVal Mat responses,
                                     @ByVal(nullValue = "cv::noArray()") Mat varIdx/*=cv::noArray()*/, @ByVal(nullValue = "cv::noArray()") Mat sampleIdx/*=cv::noArray()*/,
                                     @ByVal(nullValue = "cv::noArray()") Mat sampleWeights/*=cv::noArray()*/, @ByVal(nullValue = "cv::noArray()") Mat varType/*=cv::noArray()*/);
    public static native @Ptr TrainData create(@ByVal Mat samples, int layout, @ByVal Mat responses);
}

/** \brief Base class for statistical models in OpenCV ML.
 */
@Namespace("cv::ml") public static class StatModel extends AbstractStatModel {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public StatModel(Pointer p) { super(p); }

    /** Predict options */
    /** enum cv::ml::StatModel::Flags */
    public static final int
        UPDATE_MODEL = 1,
        /** makes the method return the raw results (the sum), not the class label */
        RAW_OUTPUT= 1,
        COMPRESSED_INPUT= 2,
        PREPROCESSED_INPUT= 4;

    /** \brief Returns the number of variables in training samples */
    public native int getVarCount();

    public native @Cast("bool") boolean empty();

    /** \brief Returns true if the model is trained */
    public native @Cast("bool") boolean isTrained();
    /** \brief Returns true if the model is classifier */
    public native @Cast("bool") boolean isClassifier();

    /** \brief Trains the statistical model
    <p>
    @param trainData training data that can be loaded from file using TrainData::loadFromCSV or
        created with TrainData::create.
    @param flags optional flags, depending on the model. Some of the models can be updated with the
        new training samples, not completely overwritten (such as NormalBayesClassifier or ANN_MLP).
     */
    public native @Cast("bool") boolean train( @Ptr TrainData trainData, int flags/*=0*/ );
    public native @Cast("bool") boolean train( @Ptr TrainData trainData );

    /** \brief Trains the statistical model
    <p>
    @param samples training samples
    @param layout See ml::SampleTypes.
    @param responses vector of responses associated with the training samples.
    */
    public native @Cast("bool") boolean train( @ByVal Mat samples, int layout, @ByVal Mat responses );

    /** \brief Computes error on the training or test dataset
    <p>
    @param data the training data
    @param test if true, the error is computed over the test subset of the data, otherwise it's
        computed over the training subset of the data. Please note that if you loaded a completely
        different dataset to evaluate already trained classifier, you will probably want not to set
        the test subset at all with TrainData::setTrainTestSplitRatio and specify test=false, so
        that the error is computed for the whole new set. Yes, this sounds a bit confusing.
    @param resp the optional output responses.
    <p>
    The method uses StatModel::predict to compute the error. For regression models the error is
    computed as RMS, for classifiers - as a percent of missclassified samples (0%-100%).
     */
    public native float calcError( @Ptr TrainData data, @Cast("bool") boolean test, @ByVal Mat resp );

    /** \brief Predicts response(s) for the provided sample(s)
    <p>
    @param samples The input samples, floating-point matrix
    @param results The optional output matrix of results.
    @param flags The optional flags, model-dependent. See cv::ml::StatModel::Flags.
     */
    public native float predict( @ByVal Mat samples, @ByVal(nullValue = "cv::noArray()") Mat results/*=cv::noArray()*/, int flags/*=0*/ );
    public native float predict( @ByVal Mat samples );

    /** \brief Create and train model with default parameters
    <p>
    The class must implement static {@code create()} method with no parameters or with all default parameter values
    */
}

/****************************************************************************************\
*                                 Normal Bayes Classifier                                *
\****************************************************************************************/

/** \brief Bayes classifier for normally distributed data.
<p>
\sa \ref ml_intro_bayes
 */
@Namespace("cv::ml") public static class NormalBayesClassifier extends StatModel {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public NormalBayesClassifier(Pointer p) { super(p); }

    /** \brief Predicts the response for sample(s).
    <p>
    The method estimates the most probable classes for input vectors. Input vectors (one or more)
    are stored as rows of the matrix inputs. In case of multiple input vectors, there should be one
    output vector outputs. The predicted class for a single input vector is returned by the method.
    The vector outputProbs contains the output probabilities corresponding to each element of
    result.
     */
    public native float predictProb( @ByVal Mat inputs, @ByVal Mat outputs,
                                   @ByVal Mat outputProbs, int flags/*=0*/ );
    public native float predictProb( @ByVal Mat inputs, @ByVal Mat outputs,
                                   @ByVal Mat outputProbs );

    /** Creates empty model
    Use StatModel::train to train the model after creation. */
    public static native @Ptr NormalBayesClassifier create();
}

/****************************************************************************************\
*                          K-Nearest Neighbour Classifier                                *
\****************************************************************************************/

/** \brief The class implements K-Nearest Neighbors model
<p>
\sa \ref ml_intro_knn
 */
@Namespace("cv::ml") public static class KNearest extends StatModel {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public KNearest(Pointer p) { super(p); }


    /** Default number of neighbors to use in predict method. */
    /** @see setDefaultK */
    public native int getDefaultK();
    /** \copybrief getDefaultK @see getDefaultK */
    public native void setDefaultK(int val);

    /** Whether classification or regression model should be trained. */
    /** @see setIsClassifier */
    public native @Cast("bool") boolean getIsClassifier();
    /** \copybrief getIsClassifier @see getIsClassifier */
    public native void setIsClassifier(@Cast("bool") boolean val);

    /** Parameter for KDTree implementation. */
    /** @see setEmax */
    public native int getEmax();
    /** \copybrief getEmax @see getEmax */
    public native void setEmax(int val);

    /** %Algorithm type, one of KNearest::Types. */
    /** @see setAlgorithmType */
    public native int getAlgorithmType();
    /** \copybrief getAlgorithmType @see getAlgorithmType */
    public native void setAlgorithmType(int val);

    /** \brief Finds the neighbors and predicts responses for input vectors.
    <p>
    @param samples Input samples stored by rows. It is a single-precision floating-point matrix of
        {@code <number_of_samples> * k} size.
    @param k Number of used nearest neighbors. Should be greater than 1.
    @param results Vector with results of prediction (regression or classification) for each input
        sample. It is a single-precision floating-point vector with {@code <number_of_samples>} elements.
    @param neighborResponses Optional output values for corresponding neighbors. It is a single-
        precision floating-point matrix of {@code <number_of_samples> * k} size.
    @param dist Optional output distances from the input vectors to the corresponding neighbors. It
        is a single-precision floating-point matrix of {@code <number_of_samples> * k} size.
    <p>
    For each input vector (a row of the matrix samples), the method finds the k nearest neighbors.
    In case of regression, the predicted result is a mean value of the particular vector's neighbor
    responses. In case of classification, the class is determined by voting.
    <p>
    For each input vector, the neighbors are sorted by their distances to the vector.
    <p>
    In case of C++ interface you can use output pointers to empty matrices and the function will
    allocate memory itself.
    <p>
    If only a single input vector is passed, all output matrices are optional and the predicted
    value is returned by the method.
    <p>
    The function is parallelized with the TBB library.
     */
    public native float findNearest( @ByVal Mat samples, int k,
                                   @ByVal Mat results,
                                   @ByVal(nullValue = "cv::noArray()") Mat neighborResponses/*=cv::noArray()*/,
                                   @ByVal(nullValue = "cv::noArray()") Mat dist/*=cv::noArray()*/ );
    public native float findNearest( @ByVal Mat samples, int k,
                                   @ByVal Mat results );

    /** \brief Implementations of KNearest algorithm
       */
    /** enum cv::ml::KNearest::Types */
    public static final int
        BRUTE_FORCE= 1,
        KDTREE= 2;

    /** \brief Creates the empty model
    <p>
    The static method creates empty %KNearest classifier. It should be then trained using StatModel::train method.
     */
    public static native @Ptr KNearest create();
}

/****************************************************************************************\
*                                   Support Vector Machines                              *
\****************************************************************************************/

/** \brief Support Vector Machines.
<p>
\sa \ref ml_intro_svm
 */
@Namespace("cv::ml") public static class SVM extends StatModel {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public SVM(Pointer p) { super(p); }


    public static class Kernel extends Algorithm {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public Kernel(Pointer p) { super(p); }
    
        public native int getType();
        public native void calc( int vcount, int n, @Const FloatPointer vecs, @Const FloatPointer another, FloatPointer results );
        public native void calc( int vcount, int n, @Const FloatBuffer vecs, @Const FloatBuffer another, FloatBuffer results );
        public native void calc( int vcount, int n, @Const float[] vecs, @Const float[] another, float[] results );
    }

    /** Type of a %SVM formulation.
    See SVM::Types. Default value is SVM::C_SVC. */
    /** @see setType */
    public native int getType();
    /** \copybrief getType @see getType */
    public native void setType(int val);

    /** Parameter \f$\gamma\f$ of a kernel function.
    For SVM::POLY, SVM::RBF, SVM::SIGMOID or SVM::CHI2. Default value is 1. */
    /** @see setGamma */
    public native double getGamma();
    /** \copybrief getGamma @see getGamma */
    public native void setGamma(double val);

    /** Parameter _coef0_ of a kernel function.
    For SVM::POLY or SVM::SIGMOID. Default value is 0.*/
    /** @see setCoef0 */
    public native double getCoef0();
    /** \copybrief getCoef0 @see getCoef0 */
    public native void setCoef0(double val);

    /** Parameter _degree_ of a kernel function.
    For SVM::POLY. Default value is 0. */
    /** @see setDegree */
    public native double getDegree();
    /** \copybrief getDegree @see getDegree */
    public native void setDegree(double val);

    /** Parameter _C_ of a %SVM optimization problem.
    For SVM::C_SVC, SVM::EPS_SVR or SVM::NU_SVR. Default value is 0. */
    /** @see setC */
    public native double getC();
    /** \copybrief getC @see getC */
    public native void setC(double val);

    /** Parameter \f$\nu\f$ of a %SVM optimization problem.
    For SVM::NU_SVC, SVM::ONE_CLASS or SVM::NU_SVR. Default value is 0. */
    /** @see setNu */
    public native double getNu();
    /** \copybrief getNu @see getNu */
    public native void setNu(double val);

    /** Parameter \f$\epsilon\f$ of a %SVM optimization problem.
    For SVM::EPS_SVR. Default value is 0. */
    /** @see setP */
    public native double getP();
    /** \copybrief getP @see getP */
    public native void setP(double val);

    /** Optional weights in the SVM::C_SVC problem, assigned to particular classes.
    They are multiplied by _C_ so the parameter _C_ of class _i_ becomes {@code classWeights(i) * C}. Thus
    these weights affect the misclassification penalty for different classes. The larger weight,
    the larger penalty on misclassification of data from the corresponding class. Default value is
    empty Mat. */
    /** @see setClassWeights */
    public native @ByVal Mat getClassWeights();
    /** \copybrief getClassWeights @see getClassWeights */
    public native void setClassWeights(@Const @ByRef Mat val);

    /** Termination criteria of the iterative %SVM training procedure which solves a partial
    case of constrained quadratic optimization problem.
    You can specify tolerance and/or the maximum number of iterations. Default value is
    {@code TermCriteria( TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, FLT_EPSILON )}; */
    /** @see setTermCriteria */
    public native @ByVal TermCriteria getTermCriteria();
    /** \copybrief getTermCriteria @see getTermCriteria */
    public native void setTermCriteria(@Const @ByRef TermCriteria val);

    /** Type of a %SVM kernel.
    See SVM::KernelTypes. Default value is SVM::RBF. */
    public native int getKernelType();

    /** Initialize with one of predefined kernels.
    See SVM::KernelTypes. */
    public native void setKernel(int kernelType);

    /** Initialize with custom kernel.
    See SVM::Kernel class for implementation details */
    public native void setCustomKernel(@Ptr Kernel _kernel);

    /** %SVM type */
    /** enum cv::ml::SVM::Types */
    public static final int
        /** C-Support Vector Classification. n-class classification (n \f$\geq\f$ 2), allows
        imperfect separation of classes with penalty multiplier C for outliers. */
        C_SVC= 100,
        /** \f$\nu\f$-Support Vector Classification. n-class classification with possible
        imperfect separation. Parameter \f$\nu\f$ (in the range 0..1, the larger the value, the smoother
        the decision boundary) is used instead of C. */
        NU_SVC= 101,
        /** Distribution Estimation (One-class %SVM). All the training data are from
        the same class, %SVM builds a boundary that separates the class from the rest of the feature
        space. */
        ONE_CLASS= 102,
        /** \f$\epsilon\f$-Support Vector Regression. The distance between feature vectors
        from the training set and the fitting hyper-plane must be less than p. For outliers the
        penalty multiplier C is used. */
        EPS_SVR= 103,
        /** \f$\nu\f$-Support Vector Regression. \f$\nu\f$ is used instead of p.
        See \cite LibSVM for details. */
        NU_SVR= 104;

    /** \brief %SVM kernel type
    <p>
    A comparison of different kernels on the following 2D test case with four classes. Four
    SVM::C_SVC SVMs have been trained (one against rest) with auto_train. Evaluation on three
    different kernels (SVM::CHI2, SVM::INTER, SVM::RBF). The color depicts the class with max score.
    Bright means max-score \> 0, dark means max-score \< 0.
    ![image](pics/SVM_Comparison.png)
    */
    /** enum cv::ml::SVM::KernelTypes */
    public static final int
        /** Returned by SVM::getKernelType in case when custom kernel has been set */
        CUSTOM= -1,
        /** Linear kernel. No mapping is done, linear discrimination (or regression) is
        done in the original feature space. It is the fastest option. \f$K(x_i, x_j) = x_i^T x_j\f$. */
        LINEAR= 0,
        /** Polynomial kernel:
        \f$K(x_i, x_j) = (\gamma x_i^T x_j + coef0)^{degree}, \gamma > 0\f$. */
        POLY= 1,
        /** Radial basis function (RBF), a good choice in most cases.
        \f$K(x_i, x_j) = e^{-\gamma ||x_i - x_j||^2}, \gamma > 0\f$. */
        RBF= 2,
        /** Sigmoid kernel: \f$K(x_i, x_j) = \tanh(\gamma x_i^T x_j + coef0)\f$. */
        SIGMOID= 3,
        /** Exponential Chi2 kernel, similar to the RBF kernel:
        \f$K(x_i, x_j) = e^{-\gamma \chi^2(x_i,x_j)}, \chi^2(x_i,x_j) = (x_i-x_j)^2/(x_i+x_j), \gamma > 0\f$. */
        CHI2= 4,
        /** Histogram intersection kernel. A fast kernel. \f$K(x_i, x_j) = min(x_i,x_j)\f$. */
        INTER= 5;

    /** %SVM params type */
    /** enum cv::ml::SVM::ParamTypes */
    public static final int
        C= 0,
        GAMMA= 1,
        P= 2,
        NU= 3,
        COEF= 4,
        DEGREE= 5;

    /** \brief Trains an %SVM with optimal parameters.
    <p>
    @param data the training data that can be constructed using TrainData::create or
        TrainData::loadFromCSV.
    @param kFold Cross-validation parameter. The training set is divided into kFold subsets. One
        subset is used to test the model, the others form the train set. So, the %SVM algorithm is
        executed kFold times.
    @param Cgrid grid for C
    @param gammaGrid grid for gamma
    @param pGrid grid for p
    @param nuGrid grid for nu
    @param coeffGrid grid for coeff
    @param degreeGrid grid for degree
    @param balanced If true and the problem is 2-class classification then the method creates more
        balanced cross-validation subsets that is proportions between classes in subsets are close
        to such proportion in the whole train dataset.
    <p>
    The method trains the %SVM model automatically by choosing the optimal parameters C, gamma, p,
    nu, coef0, degree. Parameters are considered optimal when the cross-validation
    estimate of the test set error is minimal.
    <p>
    If there is no need to optimize a parameter, the corresponding grid step should be set to any
    value less than or equal to 1. For example, to avoid optimization in gamma, set {@code gammaGrid.step
    = 0}, {@code gammaGrid.minVal}, {@code gamma_grid.maxVal} as arbitrary numbers. In this case, the value
    {@code Gamma} is taken for gamma.
    <p>
    And, finally, if the optimization in a parameter is required but the corresponding grid is
    unknown, you may call the function SVM::getDefaultGrid. To generate a grid, for example, for
    gamma, call {@code SVM::getDefaultGrid(SVM::GAMMA)}.
    <p>
    This function works for the classification (SVM::C_SVC or SVM::NU_SVC) as well as for the
    regression (SVM::EPS_SVR or SVM::NU_SVR). If it is SVM::ONE_CLASS, no optimization is made and
    the usual %SVM with parameters specified in params is executed.
     */
    public native @Cast("bool") boolean trainAuto( @Ptr TrainData data, int kFold/*=10*/,
                        @ByVal(nullValue = "cv::ml::SVM::getDefaultGrid(cv::ml::SVM::C)") ParamGrid Cgrid/*=cv::ml::SVM::getDefaultGrid(cv::ml::SVM::C)*/,
                        @ByVal(nullValue = "cv::ml::SVM::getDefaultGrid(cv::ml::SVM::GAMMA)") ParamGrid gammaGrid/*=cv::ml::SVM::getDefaultGrid(cv::ml::SVM::GAMMA)*/,
                        @ByVal(nullValue = "cv::ml::SVM::getDefaultGrid(cv::ml::SVM::P)") ParamGrid pGrid/*=cv::ml::SVM::getDefaultGrid(cv::ml::SVM::P)*/,
                        @ByVal(nullValue = "cv::ml::SVM::getDefaultGrid(cv::ml::SVM::NU)") ParamGrid nuGrid/*=cv::ml::SVM::getDefaultGrid(cv::ml::SVM::NU)*/,
                        @ByVal(nullValue = "cv::ml::SVM::getDefaultGrid(cv::ml::SVM::COEF)") ParamGrid coeffGrid/*=cv::ml::SVM::getDefaultGrid(cv::ml::SVM::COEF)*/,
                        @ByVal(nullValue = "cv::ml::SVM::getDefaultGrid(cv::ml::SVM::DEGREE)") ParamGrid degreeGrid/*=cv::ml::SVM::getDefaultGrid(cv::ml::SVM::DEGREE)*/,
                        @Cast("bool") boolean balanced/*=false*/);
    public native @Cast("bool") boolean trainAuto( @Ptr TrainData data);

    /** \brief Retrieves all the support vectors
    <p>
    The method returns all the support vector as floating-point matrix, where support vectors are
    stored as matrix rows.
     */
    public native @ByVal Mat getSupportVectors();

    /** \brief Retrieves the decision function
    <p>
    @param i the index of the decision function. If the problem solved is regression, 1-class or
        2-class classification, then there will be just one decision function and the index should
        always be 0. Otherwise, in the case of N-class classification, there will be \f$N(N-1)/2\f$
        decision functions.
    @param alpha the optional output vector for weights, corresponding to different support vectors.
        In the case of linear %SVM all the alpha's will be 1's.
    @param svidx the optional output vector of indices of support vectors within the matrix of
        support vectors (which can be retrieved by SVM::getSupportVectors). In the case of linear
        %SVM each decision function consists of a single "compressed" support vector.
    <p>
    The method returns rho parameter of the decision function, a scalar subtracted from the weighted
    sum of kernel responses.
     */
    public native double getDecisionFunction(int i, @ByVal Mat alpha, @ByVal Mat svidx);

    /** \brief Generates a grid for %SVM parameters.
    <p>
    @param param_id %SVM parameters IDs that must be one of the SVM::ParamTypes. The grid is
    generated for the parameter with this ID.
    <p>
    The function generates a grid for the specified parameter of the %SVM algorithm. The grid may be
    passed to the function SVM::trainAuto.
     */
    public static native @ByVal ParamGrid getDefaultGrid( int param_id );

    /** Creates empty model.
    Use StatModel::train to train the model. Since %SVM has several parameters, you may want to
    find the best parameters for your problem, it can be done with SVM::trainAuto. */
    public static native @Ptr SVM create();
}

/****************************************************************************************\
*                              Expectation - Maximization                                *
\****************************************************************************************/

/** \brief The class implements the Expectation Maximization algorithm.
<p>
\sa \ref ml_intro_em
 */
@Namespace("cv::ml") public static class EM extends StatModel {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public EM(Pointer p) { super(p); }

    /** Type of covariation matrices */
    /** enum cv::ml::EM::Types */
    public static final int
        /** A scaled identity matrix \f$\mu_k * I\f$. There is the only
        parameter \f$\mu_k\f$ to be estimated for each matrix. The option may be used in special cases,
        when the constraint is relevant, or as a first step in the optimization (for example in case
        when the data is preprocessed with PCA). The results of such preliminary estimation may be
        passed again to the optimization procedure, this time with
        covMatType=EM::COV_MAT_DIAGONAL. */
        COV_MAT_SPHERICAL= 0,
        /** A diagonal matrix with positive diagonal elements. The number of
        free parameters is d for each matrix. This is most commonly used option yielding good
        estimation results. */
        COV_MAT_DIAGONAL= 1,
        /** A symmetric positively defined matrix. The number of free
        parameters in each matrix is about \f$d^2/2\f$. It is not recommended to use this option, unless
        there is pretty accurate initial estimation of the parameters and/or a huge number of
        training samples. */
        COV_MAT_GENERIC= 2,
        COV_MAT_DEFAULT= COV_MAT_DIAGONAL;

    /** Default parameters */
    /** enum cv::ml::EM:: */
    public static final int DEFAULT_NCLUSTERS= 5, DEFAULT_MAX_ITERS= 100;

    /** The initial step */
    /** enum cv::ml::EM:: */
    public static final int START_E_STEP= 1, START_M_STEP= 2, START_AUTO_STEP= 0;

    /** The number of mixture components in the Gaussian mixture model.
    Default value of the parameter is EM::DEFAULT_NCLUSTERS=5. Some of %EM implementation could
    determine the optimal number of mixtures within a specified value range, but that is not the
    case in ML yet. */
    /** @see setClustersNumber */
    public native int getClustersNumber();
    /** \copybrief getClustersNumber @see getClustersNumber */
    public native void setClustersNumber(int val);

    /** Constraint on covariance matrices which defines type of matrices.
    See EM::Types. */
    /** @see setCovarianceMatrixType */
    public native int getCovarianceMatrixType();
    /** \copybrief getCovarianceMatrixType @see getCovarianceMatrixType */
    public native void setCovarianceMatrixType(int val);

    /** The termination criteria of the %EM algorithm.
    The %EM algorithm can be terminated by the number of iterations termCrit.maxCount (number of
    M-steps) or when relative change of likelihood logarithm is less than termCrit.epsilon. Default
    maximum number of iterations is EM::DEFAULT_MAX_ITERS=100. */
    /** @see setTermCriteria */
    public native @ByVal TermCriteria getTermCriteria();
    /** \copybrief getTermCriteria @see getTermCriteria */
    public native void setTermCriteria(@Const @ByRef TermCriteria val);

    /** \brief Returns weights of the mixtures
    <p>
    Returns vector with the number of elements equal to the number of mixtures.
     */
    public native @ByVal Mat getWeights();
    /** \brief Returns the cluster centers (means of the Gaussian mixture)
    <p>
    Returns matrix with the number of rows equal to the number of mixtures and number of columns
    equal to the space dimensionality.
     */
    public native @ByVal Mat getMeans();
    /** \brief Returns covariation matrices
    <p>
    Returns vector of covariation matrices. Number of matrices is the number of gaussian mixtures,
    each matrix is a square floating-point matrix NxN, where N is the space dimensionality.
     */
    public native void getCovs(@ByRef MatVector covs);

    /** \brief Returns a likelihood logarithm value and an index of the most probable mixture component
    for the given sample.
    <p>
    @param sample A sample for classification. It should be a one-channel matrix of
        \f$1 \times dims\f$ or \f$dims \times 1\f$ size.
    @param probs Optional output matrix that contains posterior probabilities of each component
        given the sample. It has \f$1 \times nclusters\f$ size and CV_64FC1 type.
    <p>
    The method returns a two-element double vector. Zero element is a likelihood logarithm value for
    the sample. First element is an index of the most probable mixture component for the given
    sample.
     */
    public native @ByVal Point2d predict2(@ByVal Mat sample, @ByVal Mat probs);

    /** \brief Estimate the Gaussian mixture parameters from a samples set.
    <p>
    This variation starts with Expectation step. Initial values of the model parameters will be
    estimated by the k-means algorithm.
    <p>
    Unlike many of the ML models, %EM is an unsupervised learning algorithm and it does not take
    responses (class labels or function values) as input. Instead, it computes the *Maximum
    Likelihood Estimate* of the Gaussian mixture parameters from an input sample set, stores all the
    parameters inside the structure: \f$p_{i,k}\f$ in probs, \f$a_k\f$ in means , \f$S_k\f$ in
    covs[k], \f$\pi_k\f$ in weights , and optionally computes the output "class label" for each
    sample: \f$\texttt{labels}_i=\texttt{arg max}_k(p_{i,k}), i=1..N\f$ (indices of the most
    probable mixture component for each sample).
    <p>
    The trained model can be used further for prediction, just like any other classifier. The
    trained model is similar to the NormalBayesClassifier.
    <p>
    @param samples Samples from which the Gaussian mixture model will be estimated. It should be a
        one-channel matrix, each row of which is a sample. If the matrix does not have CV_64F type
        it will be converted to the inner matrix of such type for the further computing.
    @param logLikelihoods The optional output matrix that contains a likelihood logarithm value for
        each sample. It has \f$nsamples \times 1\f$ size and CV_64FC1 type.
    @param labels The optional output "class label" for each sample:
        \f$\texttt{labels}_i=\texttt{arg max}_k(p_{i,k}), i=1..N\f$ (indices of the most probable
        mixture component for each sample). It has \f$nsamples \times 1\f$ size and CV_32SC1 type.
    @param probs The optional output matrix that contains posterior probabilities of each Gaussian
        mixture component given the each sample. It has \f$nsamples \times nclusters\f$ size and
        CV_64FC1 type.
     */
    public native @Cast("bool") boolean trainEM(@ByVal Mat samples,
                             @ByVal(nullValue = "cv::noArray()") Mat logLikelihoods/*=cv::noArray()*/,
                             @ByVal(nullValue = "cv::noArray()") Mat labels/*=cv::noArray()*/,
                             @ByVal(nullValue = "cv::noArray()") Mat probs/*=cv::noArray()*/);
    public native @Cast("bool") boolean trainEM(@ByVal Mat samples);

    /** \brief Estimate the Gaussian mixture parameters from a samples set.
    <p>
    This variation starts with Expectation step. You need to provide initial means \f$a_k\f$ of
    mixture components. Optionally you can pass initial weights \f$\pi_k\f$ and covariance matrices
    \f$S_k\f$ of mixture components.
    <p>
    @param samples Samples from which the Gaussian mixture model will be estimated. It should be a
        one-channel matrix, each row of which is a sample. If the matrix does not have CV_64F type
        it will be converted to the inner matrix of such type for the further computing.
    @param means0 Initial means \f$a_k\f$ of mixture components. It is a one-channel matrix of
        \f$nclusters \times dims\f$ size. If the matrix does not have CV_64F type it will be
        converted to the inner matrix of such type for the further computing.
    @param covs0 The vector of initial covariance matrices \f$S_k\f$ of mixture components. Each of
        covariance matrices is a one-channel matrix of \f$dims \times dims\f$ size. If the matrices
        do not have CV_64F type they will be converted to the inner matrices of such type for the
        further computing.
    @param weights0 Initial weights \f$\pi_k\f$ of mixture components. It should be a one-channel
        floating-point matrix with \f$1 \times nclusters\f$ or \f$nclusters \times 1\f$ size.
    @param logLikelihoods The optional output matrix that contains a likelihood logarithm value for
        each sample. It has \f$nsamples \times 1\f$ size and CV_64FC1 type.
    @param labels The optional output "class label" for each sample:
        \f$\texttt{labels}_i=\texttt{arg max}_k(p_{i,k}), i=1..N\f$ (indices of the most probable
        mixture component for each sample). It has \f$nsamples \times 1\f$ size and CV_32SC1 type.
    @param probs The optional output matrix that contains posterior probabilities of each Gaussian
        mixture component given the each sample. It has \f$nsamples \times nclusters\f$ size and
        CV_64FC1 type.
    */
    public native @Cast("bool") boolean trainE(@ByVal Mat samples, @ByVal Mat means0,
                            @ByVal(nullValue = "cv::noArray()") Mat covs0/*=cv::noArray()*/,
                            @ByVal(nullValue = "cv::noArray()") Mat weights0/*=cv::noArray()*/,
                            @ByVal(nullValue = "cv::noArray()") Mat logLikelihoods/*=cv::noArray()*/,
                            @ByVal(nullValue = "cv::noArray()") Mat labels/*=cv::noArray()*/,
                            @ByVal(nullValue = "cv::noArray()") Mat probs/*=cv::noArray()*/);
    public native @Cast("bool") boolean trainE(@ByVal Mat samples, @ByVal Mat means0);

    /** \brief Estimate the Gaussian mixture parameters from a samples set.
    <p>
    This variation starts with Maximization step. You need to provide initial probabilities
    \f$p_{i,k}\f$ to use this option.
    <p>
    @param samples Samples from which the Gaussian mixture model will be estimated. It should be a
        one-channel matrix, each row of which is a sample. If the matrix does not have CV_64F type
        it will be converted to the inner matrix of such type for the further computing.
    @param probs0
    @param logLikelihoods The optional output matrix that contains a likelihood logarithm value for
        each sample. It has \f$nsamples \times 1\f$ size and CV_64FC1 type.
    @param labels The optional output "class label" for each sample:
        \f$\texttt{labels}_i=\texttt{arg max}_k(p_{i,k}), i=1..N\f$ (indices of the most probable
        mixture component for each sample). It has \f$nsamples \times 1\f$ size and CV_32SC1 type.
    @param probs The optional output matrix that contains posterior probabilities of each Gaussian
        mixture component given the each sample. It has \f$nsamples \times nclusters\f$ size and
        CV_64FC1 type.
    */
    public native @Cast("bool") boolean trainM(@ByVal Mat samples, @ByVal Mat probs0,
                            @ByVal(nullValue = "cv::noArray()") Mat logLikelihoods/*=cv::noArray()*/,
                            @ByVal(nullValue = "cv::noArray()") Mat labels/*=cv::noArray()*/,
                            @ByVal(nullValue = "cv::noArray()") Mat probs/*=cv::noArray()*/);
    public native @Cast("bool") boolean trainM(@ByVal Mat samples, @ByVal Mat probs0);

    /** Creates empty %EM model.
    The model should be trained then using StatModel::train(traindata, flags) method. Alternatively, you
    can use one of the EM::train\* methods or load it from file using Algorithm::load\<EM\>(filename).
     */
    public static native @Ptr EM create();
}

/****************************************************************************************\
*                                      Decision Tree                                     *
\****************************************************************************************/

/** \brief The class represents a single decision tree or a collection of decision trees.
<p>
The current public interface of the class allows user to train only a single decision tree, however
the class is capable of storing multiple decision trees and using them for prediction (by summing
responses or using a voting schemes), and the derived from DTrees classes (such as RTrees and Boost)
use this capability to implement decision tree ensembles.
<p>
\sa \ref ml_intro_trees
*/
@Namespace("cv::ml") public static class DTrees extends StatModel {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DTrees(Pointer p) { super(p); }

    /** Predict options */
    /** enum cv::ml::DTrees::Flags */
    public static final int PREDICT_AUTO= 0, PREDICT_SUM= (1<<8), PREDICT_MAX_VOTE= (2<<8), PREDICT_MASK= (3<<8);

    /** Cluster possible values of a categorical variable into K\<=maxCategories clusters to
    find a suboptimal split.
    If a discrete variable, on which the training procedure tries to make a split, takes more than
    maxCategories values, the precise best subset estimation may take a very long time because the
    algorithm is exponential. Instead, many decision trees engines (including our implementation)
    try to find sub-optimal split in this case by clustering all the samples into maxCategories
    clusters that is some categories are merged together. The clustering is applied only in n \>
    2-class classification problems for categorical variables with N \> max_categories possible
    values. In case of regression and 2-class classification the optimal split can be found
    efficiently without employing clustering, thus the parameter is not used in these cases.
    Default value is 10.*/
    /** @see setMaxCategories */
    public native int getMaxCategories();
    /** \copybrief getMaxCategories @see getMaxCategories */
    public native void setMaxCategories(int val);

    /** The maximum possible depth of the tree.
    That is the training algorithms attempts to split a node while its depth is less than maxDepth.
    The root node has zero depth. The actual depth may be smaller if the other termination criteria
    are met (see the outline of the training procedure \ref ml_intro_trees "here"), and/or if the
    tree is pruned. Default value is INT_MAX.*/
    /** @see setMaxDepth */
    public native int getMaxDepth();
    /** \copybrief getMaxDepth @see getMaxDepth */
    public native void setMaxDepth(int val);

    /** If the number of samples in a node is less than this parameter then the node will not be split.
    <p>
    Default value is 10.*/
    /** @see setMinSampleCount */
    public native int getMinSampleCount();
    /** \copybrief getMinSampleCount @see getMinSampleCount */
    public native void setMinSampleCount(int val);

    /** If CVFolds \> 1 then algorithms prunes the built decision tree using K-fold
    cross-validation procedure where K is equal to CVFolds.
    Default value is 10.*/
    /** @see setCVFolds */
    public native int getCVFolds();
    /** \copybrief getCVFolds @see getCVFolds */
    public native void setCVFolds(int val);

    /** If true then surrogate splits will be built.
    These splits allow to work with missing data and compute variable importance correctly.
    Default value is false.
    \note currently it's not implemented.*/
    /** @see setUseSurrogates */
    public native @Cast("bool") boolean getUseSurrogates();
    /** \copybrief getUseSurrogates @see getUseSurrogates */
    public native void setUseSurrogates(@Cast("bool") boolean val);

    /** If true then a pruning will be harsher.
    This will make a tree more compact and more resistant to the training data noise but a bit less
    accurate. Default value is true.*/
    /** @see setUse1SERule */
    public native @Cast("bool") boolean getUse1SERule();
    /** \copybrief getUse1SERule @see getUse1SERule */
    public native void setUse1SERule(@Cast("bool") boolean val);

    /** If true then pruned branches are physically removed from the tree.
    Otherwise they are retained and it is possible to get results from the original unpruned (or
    pruned less aggressively) tree. Default value is true.*/
    /** @see setTruncatePrunedTree */
    public native @Cast("bool") boolean getTruncatePrunedTree();
    /** \copybrief getTruncatePrunedTree @see getTruncatePrunedTree */
    public native void setTruncatePrunedTree(@Cast("bool") boolean val);

    /** Termination criteria for regression trees.
    If all absolute differences between an estimated value in a node and values of train samples
    in this node are less than this parameter then the node will not be split further. Default
    value is 0.01f*/
    /** @see setRegressionAccuracy */
    public native float getRegressionAccuracy();
    /** \copybrief getRegressionAccuracy @see getRegressionAccuracy */
    public native void setRegressionAccuracy(float val);

    /** \brief The array of a priori class probabilities, sorted by the class label value.
    <p>
    The parameter can be used to tune the decision tree preferences toward a certain class. For
    example, if you want to detect some rare anomaly occurrence, the training base will likely
    contain much more normal cases than anomalies, so a very good classification performance
    will be achieved just by considering every case as normal. To avoid this, the priors can be
    specified, where the anomaly probability is artificially increased (up to 0.5 or even
    greater), so the weight of the misclassified anomalies becomes much bigger, and the tree is
    adjusted properly.
    <p>
    You can also think about this parameter as weights of prediction categories which determine
    relative weights that you give to misclassification. That is, if the weight of the first
    category is 1 and the weight of the second category is 10, then each mistake in predicting
    the second category is equivalent to making 10 mistakes in predicting the first category.
    Default value is empty Mat.*/
    /** @see setPriors */
    public native @ByVal Mat getPriors();
    /** \copybrief getPriors @see getPriors */
    public native void setPriors(@Const @ByRef Mat val);

    /** \brief The class represents a decision tree node.
     */
    @NoOffset public static class Node extends Pointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public Node(Pointer p) { super(p); }
        /** Native array allocator. Access with {@link Pointer#position(int)}. */
        public Node(int size) { super((Pointer)null); allocateArray(size); }
        private native void allocateArray(int size);
        @Override public Node position(int position) {
            return (Node)super.position(position);
        }
    
        public Node() { super((Pointer)null); allocate(); }
        private native void allocate();
        /** Value at the node: a class label in case of classification or estimated
         *  function value in case of regression. */
        public native double value(); public native Node value(double value);
        /** Class index normalized to 0..class_count-1 range and assigned to the
         *  node. It is used internally in classification trees and tree ensembles. */
        public native int classIdx(); public native Node classIdx(int classIdx);
        /** Index of the parent node */
        public native int parent(); public native Node parent(int parent);
        /** Index of the left child node */
        public native int left(); public native Node left(int left);
        /** Index of right child node */
        public native int right(); public native Node right(int right);
        /** Default direction where to go (-1: left or +1: right). It helps in the
         *  case of missing values. */
        public native int defaultDir(); public native Node defaultDir(int defaultDir);
        /** Index of the first split */
        public native int split(); public native Node split(int split);
    }

    /** \brief The class represents split in a decision tree.
     */
    @NoOffset public static class Split extends Pointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public Split(Pointer p) { super(p); }
        /** Native array allocator. Access with {@link Pointer#position(int)}. */
        public Split(int size) { super((Pointer)null); allocateArray(size); }
        private native void allocateArray(int size);
        @Override public Split position(int position) {
            return (Split)super.position(position);
        }
    
        public Split() { super((Pointer)null); allocate(); }
        private native void allocate();
        /** Index of variable on which the split is created. */
        public native int varIdx(); public native Split varIdx(int varIdx);
        /** If true, then the inverse split rule is used (i.e. left and right
         *  branches are exchanged in the rule expressions below). */
        public native @Cast("bool") boolean inversed(); public native Split inversed(boolean inversed);
        /** The split quality, a positive number. It is used to choose the best split. */
        public native float quality(); public native Split quality(float quality);
        /** Index of the next split in the list of splits for the node */
        public native int next(); public native Split next(int next);
        /** The threshold value in case of split on an ordered variable.
                              The rule is:
                              <pre>{@code {.none}
                              if var_value < c
                                then next_node <- left
                                else next_node <- right
                              }</pre> */
        public native float c(); public native Split c(float c);
        /** Offset of the bitset used by the split on a categorical variable.
                                    The rule is:
                                    <pre>{@code {.none}
                                    if bitset[var_value] == 1
                                        then next_node <- left
                                        else next_node <- right
                                    }</pre> */
        public native int subsetOfs(); public native Split subsetOfs(int subsetOfs);
    }

    /** \brief Returns indices of root nodes
    */
    public native @StdVector IntPointer getRoots();
    /** \brief Returns all the nodes
    <p>
    all the node indices are indices in the returned vector
     */
    public native @StdVector Node getNodes();
    /** \brief Returns all the splits
    <p>
    all the split indices are indices in the returned vector
     */
    public native @StdVector Split getSplits();
    /** \brief Returns all the bitsets for categorical splits
    <p>
    Split::subsetOfs is an offset in the returned vector
     */
    public native @StdVector IntPointer getSubsets();

    /** \brief Creates the empty model
    <p>
    The static method creates empty decision tree with the specified parameters. It should be then
    trained using train method (see StatModel::train). Alternatively, you can load the model from
    file using Algorithm::load\<DTrees\>(filename).
     */
    public static native @Ptr DTrees create();
}

/****************************************************************************************\
*                                   Random Trees Classifier                              *
\****************************************************************************************/

/** \brief The class implements the random forest predictor.
<p>
\sa \ref ml_intro_rtrees
 */
@Namespace("cv::ml") public static class RTrees extends DTrees {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public RTrees(Pointer p) { super(p); }


    /** If true then variable importance will be calculated and then it can be retrieved by RTrees::getVarImportance.
    Default value is false.*/
    /** @see setCalculateVarImportance */
    public native @Cast("bool") boolean getCalculateVarImportance();
    /** \copybrief getCalculateVarImportance @see getCalculateVarImportance */
    public native void setCalculateVarImportance(@Cast("bool") boolean val);

    /** The size of the randomly selected subset of features at each tree node and that are used
    to find the best split(s).
    If you set it to 0 then the size will be set to the square root of the total number of
    features. Default value is 0.*/
    /** @see setActiveVarCount */
    public native int getActiveVarCount();
    /** \copybrief getActiveVarCount @see getActiveVarCount */
    public native void setActiveVarCount(int val);

    /** The termination criteria that specifies when the training algorithm stops.
    Either when the specified number of trees is trained and added to the ensemble or when
    sufficient accuracy (measured as OOB error) is achieved. Typically the more trees you have the
    better the accuracy. However, the improvement in accuracy generally diminishes and asymptotes
    pass a certain number of trees. Also to keep in mind, the number of tree increases the
    prediction time linearly. Default value is TermCriteria(TermCriteria::MAX_ITERS +
    TermCriteria::EPS, 50, 0.1)*/
    /** @see setTermCriteria */
    public native @ByVal TermCriteria getTermCriteria();
    /** \copybrief getTermCriteria @see getTermCriteria */
    public native void setTermCriteria(@Const @ByRef TermCriteria val);

    /** Returns the variable importance array.
    The method returns the variable importance vector, computed at the training stage when
    CalculateVarImportance is set to true. If this flag was set to false, the empty matrix is
    returned.
     */
    public native @ByVal Mat getVarImportance();

    /** Creates the empty model.
    Use StatModel::train to train the model, StatModel::train to create and train the model,
    Algorithm::load to load the pre-trained model.
     */
    public static native @Ptr RTrees create();
}

/****************************************************************************************\
*                                   Boosted tree classifier                              *
\****************************************************************************************/

/** \brief Boosted tree classifier derived from DTrees
<p>
\sa \ref ml_intro_boost
 */
@Namespace("cv::ml") public static class Boost extends DTrees {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Boost(Pointer p) { super(p); }

    /** Type of the boosting algorithm.
    See Boost::Types. Default value is Boost::REAL. */
    /** @see setBoostType */
    public native int getBoostType();
    /** \copybrief getBoostType @see getBoostType */
    public native void setBoostType(int val);

    /** The number of weak classifiers.
    Default value is 100. */
    /** @see setWeakCount */
    public native int getWeakCount();
    /** \copybrief getWeakCount @see getWeakCount */
    public native void setWeakCount(int val);

    /** A threshold between 0 and 1 used to save computational time.
    Samples with summary weight \f$\leq 1 - weight_trim_rate\f$ do not participate in the *next*
    iteration of training. Set this parameter to 0 to turn off this functionality. Default value is 0.95.*/
    /** @see setWeightTrimRate */
    public native double getWeightTrimRate();
    /** \copybrief getWeightTrimRate @see getWeightTrimRate */
    public native void setWeightTrimRate(double val);

    /** Boosting type.
    Gentle AdaBoost and Real AdaBoost are often the preferable choices. */
    /** enum cv::ml::Boost::Types */
    public static final int
        /** Discrete AdaBoost. */
        DISCRETE= 0,
        /** Real AdaBoost. It is a technique that utilizes confidence-rated predictions
 *  and works well with categorical data. */
        REAL= 1,
        /** LogitBoost. It can produce good regression fits. */
        LOGIT= 2,
        /** Gentle AdaBoost. It puts less weight on outlier data points and for that
 * reason is often good with regression data. */
        GENTLE= 3;

    /** Creates the empty model.
    Use StatModel::train to train the model, Algorithm::load\<Boost\>(filename) to load the pre-trained model. */
    public static native @Ptr Boost create();
}

/****************************************************************************************\
*                                   Gradient Boosted Trees                               *
\****************************************************************************************/

/*class CV_EXPORTS_W GBTrees : public DTrees
{
public:
    struct CV_EXPORTS_W_MAP Params : public DTrees::Params
    {
        CV_PROP_RW int weakCount;
        CV_PROP_RW int lossFunctionType;
        CV_PROP_RW float subsamplePortion;
        CV_PROP_RW float shrinkage;

        Params();
        Params( int lossFunctionType, int weakCount, float shrinkage,
                float subsamplePortion, int maxDepth, bool useSurrogates );
    };

    enum {SQUARED_LOSS=0, ABSOLUTE_LOSS, HUBER_LOSS=3, DEVIANCE_LOSS};

    virtual void setK(int k) = 0;

    virtual float predictSerial( InputArray samples,
                                 OutputArray weakResponses, int flags) const = 0;

    static Ptr<GBTrees> create(const Params& p);
};*/

/****************************************************************************************\
*                              Artificial Neural Networks (ANN)                          *
\****************************************************************************************/

/////////////////////////////////// Multi-Layer Perceptrons //////////////////////////////

/** \brief Artificial Neural Networks - Multi-Layer Perceptrons.
<p>
Unlike many other models in ML that are constructed and trained at once, in the MLP model these
steps are separated. First, a network with the specified topology is created using the non-default
constructor or the method ANN_MLP::create. All the weights are set to zeros. Then, the network is
trained using a set of input and output vectors. The training procedure can be repeated more than
once, that is, the weights can be adjusted based on the new training data.
<p>
Additional flags for StatModel::train are available: ANN_MLP::TrainFlags.
<p>
\sa \ref ml_intro_ann
 */
@Namespace("cv::ml") public static class ANN_MLP extends StatModel {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ANN_MLP(Pointer p) { super(p); }

    /** Available training methods */
    /** enum cv::ml::ANN_MLP::TrainingMethods */
    public static final int
        /** The back-propagation algorithm. */
        BACKPROP= 0,
        /** The RPROP algorithm. See \cite RPROP93 for details. */
        RPROP= 1;

    /** Sets training method and common parameters.
    @param method Default value is ANN_MLP::RPROP. See ANN_MLP::TrainingMethods.
    @param param1 passed to setRpropDW0 for ANN_MLP::RPROP and to setBackpropWeightScale for ANN_MLP::BACKPROP
    @param param2 passed to setRpropDWMin for ANN_MLP::RPROP and to setBackpropMomentumScale for ANN_MLP::BACKPROP.
    */
    public native void setTrainMethod(int method, double param1/*=0*/, double param2/*=0*/);
    public native void setTrainMethod(int method);

    /** Returns current training method */
    public native int getTrainMethod();

    /** Initialize the activation function for each neuron.
    Currently the default and the only fully supported activation function is ANN_MLP::SIGMOID_SYM.
    @param type The type of activation function. See ANN_MLP::ActivationFunctions.
    @param param1 The first parameter of the activation function, \f$\alpha\f$. Default value is 0.
    @param param2 The second parameter of the activation function, \f$\beta\f$. Default value is 0.
    */
    public native void setActivationFunction(int type, double param1/*=0*/, double param2/*=0*/);
    public native void setActivationFunction(int type);

    /**  Integer vector specifying the number of neurons in each layer including the input and output layers.
    The very first element specifies the number of elements in the input layer.
    The last element - number of elements in the output layer. Default value is empty Mat.
    \sa getLayerSizes */
    public native void setLayerSizes(@ByVal Mat _layer_sizes);

    /**  Integer vector specifying the number of neurons in each layer including the input and output layers.
    The very first element specifies the number of elements in the input layer.
    The last element - number of elements in the output layer.
    \sa setLayerSizes */
    public native @ByVal Mat getLayerSizes();

    /** Termination criteria of the training algorithm.
    You can specify the maximum number of iterations (maxCount) and/or how much the error could
    change between the iterations to make the algorithm continue (epsilon). Default value is
    TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 0.01).*/
    /** @see setTermCriteria */
    public native @ByVal TermCriteria getTermCriteria();
    /** \copybrief getTermCriteria @see getTermCriteria */
    public native void setTermCriteria(@ByVal TermCriteria val);

    /** BPROP: Strength of the weight gradient term.
    The recommended value is about 0.1. Default value is 0.1.*/
    /** @see setBackpropWeightScale */
    public native double getBackpropWeightScale();
    /** \copybrief getBackpropWeightScale @see getBackpropWeightScale */
    public native void setBackpropWeightScale(double val);

    /** BPROP: Strength of the momentum term (the difference between weights on the 2 previous iterations).
    This parameter provides some inertia to smooth the random fluctuations of the weights. It can
    vary from 0 (the feature is disabled) to 1 and beyond. The value 0.1 or so is good enough.
    Default value is 0.1.*/
    /** @see setBackpropMomentumScale */
    public native double getBackpropMomentumScale();
    /** \copybrief getBackpropMomentumScale @see getBackpropMomentumScale */
    public native void setBackpropMomentumScale(double val);

    /** RPROP: Initial value \f$\Delta_0\f$ of update-values \f$\Delta_{ij}\f$.
    Default value is 0.1.*/
    /** @see setRpropDW0 */
    public native double getRpropDW0();
    /** \copybrief getRpropDW0 @see getRpropDW0 */
    public native void setRpropDW0(double val);

    /** RPROP: Increase factor \f$\eta^+\f$.
    It must be \>1. Default value is 1.2.*/
    /** @see setRpropDWPlus */
    public native double getRpropDWPlus();
    /** \copybrief getRpropDWPlus @see getRpropDWPlus */
    public native void setRpropDWPlus(double val);

    /** RPROP: Decrease factor \f$\eta^-\f$.
    It must be \<1. Default value is 0.5.*/
    /** @see setRpropDWMinus */
    public native double getRpropDWMinus();
    /** \copybrief getRpropDWMinus @see getRpropDWMinus */
    public native void setRpropDWMinus(double val);

    /** RPROP: Update-values lower limit \f$\Delta_{min}\f$.
    It must be positive. Default value is FLT_EPSILON.*/
    /** @see setRpropDWMin */
    public native double getRpropDWMin();
    /** \copybrief getRpropDWMin @see getRpropDWMin */
    public native void setRpropDWMin(double val);

    /** RPROP: Update-values upper limit \f$\Delta_{max}\f$.
    It must be \>1. Default value is 50.*/
    /** @see setRpropDWMax */
    public native double getRpropDWMax();
    /** \copybrief getRpropDWMax @see getRpropDWMax */
    public native void setRpropDWMax(double val);

    /** possible activation functions */
    /** enum cv::ml::ANN_MLP::ActivationFunctions */
    public static final int
        /** Identity function: \f$f(x)=x\f$ */
        IDENTITY = 0,
        /** Symmetrical sigmoid: \f$f(x)=\beta*(1-e^{-\alpha x})/(1+e^{-\alpha x}\f$
        \note
        If you are using the default sigmoid activation function with the default parameter values
        fparam1=0 and fparam2=0 then the function used is y = 1.7159\*tanh(2/3 \* x), so the output
        will range from [-1.7159, 1.7159], instead of [0,1].*/
        SIGMOID_SYM = 1,
        /** Gaussian function: \f$f(x)=\beta e^{-\alpha x*x}\f$ */
        GAUSSIAN = 2;

    /** Train options */
    /** enum cv::ml::ANN_MLP::TrainFlags */
    public static final int
        /** Update the network weights, rather than compute them from scratch. In the latter case
        the weights are initialized using the Nguyen-Widrow algorithm. */
        UPDATE_WEIGHTS = 1,
        /** Do not normalize the input vectors. If this flag is not set, the training algorithm
        normalizes each input feature independently, shifting its mean value to 0 and making the
        standard deviation equal to 1. If the network is assumed to be updated frequently, the new
        training data could be much different from original one. In this case, you should take care
        of proper normalization. */
        NO_INPUT_SCALE = 2,
        /** Do not normalize the output vectors. If the flag is not set, the training algorithm
        normalizes each output feature independently, by transforming it to the certain range
        depending on the used activation function. */
        NO_OUTPUT_SCALE = 4;

    public native @ByVal Mat getWeights(int layerIdx);

    /** \brief Creates empty model
    <p>
    Use StatModel::train to train the model, Algorithm::load\<ANN_MLP\>(filename) to load the pre-trained model.
    Note that the train method has optional flags: ANN_MLP::TrainFlags.
     */
    public static native @Ptr ANN_MLP create();
}

/****************************************************************************************\
*                           Logistic Regression                                          *
\****************************************************************************************/

/** \brief Implements Logistic Regression classifier.
<p>
\sa \ref ml_intro_lr
 */
@Namespace("cv::ml") public static class LogisticRegression extends StatModel {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public LogisticRegression(Pointer p) { super(p); }


    /** Learning rate. */
    /** @see setLearningRate */
    public native double getLearningRate();
    /** \copybrief getLearningRate @see getLearningRate */
    public native void setLearningRate(double val);

    /** Number of iterations. */
    /** @see setIterations */
    public native int getIterations();
    /** \copybrief getIterations @see getIterations */
    public native void setIterations(int val);

    /** Kind of regularization to be applied. See LogisticRegression::RegKinds. */
    /** @see setRegularization */
    public native int getRegularization();
    /** \copybrief getRegularization @see getRegularization */
    public native void setRegularization(int val);

    /** Kind of training method used. See LogisticRegression::Methods. */
    /** @see setTrainMethod */
    public native int getTrainMethod();
    /** \copybrief getTrainMethod @see getTrainMethod */
    public native void setTrainMethod(int val);

    /** Specifies the number of training samples taken in each step of Mini-Batch Gradient
    Descent. Will only be used if using LogisticRegression::MINI_BATCH training algorithm. It
    has to take values less than the total number of training samples. */
    /** @see setMiniBatchSize */
    public native int getMiniBatchSize();
    /** \copybrief getMiniBatchSize @see getMiniBatchSize */
    public native void setMiniBatchSize(int val);

    /** Termination criteria of the algorithm. */
    /** @see setTermCriteria */
    public native @ByVal TermCriteria getTermCriteria();
    /** \copybrief getTermCriteria @see getTermCriteria */
    public native void setTermCriteria(@ByVal TermCriteria val);

    /** Regularization kinds */
    /** enum cv::ml::LogisticRegression::RegKinds */
    public static final int
        /** Regularization disabled */
        REG_DISABLE = -1,
        /** %L1 norm */
        REG_L1 = 0,
        /** %L2 norm */
        REG_L2 = 1;

    /** Training methods */
    /** enum cv::ml::LogisticRegression::Methods */
    public static final int
        BATCH = 0,
        /** Set MiniBatchSize to a positive integer when using this method. */
        MINI_BATCH = 1;

    /** \brief Predicts responses for input samples and returns a float type.
    <p>
    @param samples The input data for the prediction algorithm. Matrix [m x n], where each row
        contains variables (features) of one object being classified. Should have data type CV_32F.
    @param results Predicted labels as a column matrix of type CV_32S.
    @param flags Not used.
     */
    public native float predict( @ByVal Mat samples, @ByVal(nullValue = "cv::noArray()") Mat results/*=cv::noArray()*/, int flags/*=0*/ );
    public native float predict( @ByVal Mat samples );

    /** \brief This function returns the trained paramters arranged across rows.
    <p>
    For a two class classifcation problem, it returns a row matrix. It returns learnt paramters of
    the Logistic Regression as a matrix of type CV_32F.
     */
    public native @ByVal Mat get_learnt_thetas();

    /** \brief Creates empty model.
    <p>
    Creates Logistic Regression model with parameters given.
     */
    public static native @Ptr LogisticRegression create();
}

/****************************************************************************************\
*                           Auxilary functions declarations                              *
\****************************************************************************************/

/** \brief Generates _sample_ from multivariate normal distribution
<p>
@param mean an average row vector
@param cov symmetric covariation matrix
@param nsamples returned samples count
@param samples returned samples array
*/
@Namespace("cv::ml") public static native void randMVNormal( @ByVal Mat mean, @ByVal Mat cov, int nsamples, @ByVal Mat samples);

/** \brief Generates sample from gaussian mixture distribution */


/** \brief Creates test set */
@Namespace("cv::ml") public static native void createConcentricSpheresTestSet( int nsamples, int nfeatures, int nclasses,
                                                @ByVal Mat samples, @ByVal Mat responses);

/** \} ml */




// #endif // __cplusplus
// #endif // __OPENCV_ML_HPP__

/* End of file. */


}
