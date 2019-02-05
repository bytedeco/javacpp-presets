package org.bytedeco.opencv.opencv_ml;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.Name;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.opencv.opencv_core.Algorithm;

import static org.bytedeco.opencv.global.opencv_core.*;

@Properties(inherit = org.bytedeco.opencv.presets.opencv_ml.class)
@Name("cv::ml::StatModel") public abstract class AbstractStatModel extends Algorithm {
    static { Loader.load(); }
    public AbstractStatModel(Pointer p) { super(p); }

    public static native @Ptr @Name("load<cv::ml::NormalBayesClassifier>") NormalBayesClassifier loadNormalBayesClassifier(@Str BytePointer filename, @Str BytePointer objname);
    public static native @Ptr @Name("load<cv::ml::NormalBayesClassifier>") NormalBayesClassifier loadNormalBayesClassifier(@Str String filename, @Str String objname);
    public static native @Ptr @Name("load<cv::ml::KNearest>") KNearest loadKNearest(@Str BytePointer filename, @Str BytePointer objname);
    public static native @Ptr @Name("load<cv::ml::KNearest>") KNearest loadKNearest(@Str String filename, @Str String objname);
    public static native @Ptr @Name("load<cv::ml::SVM>") SVM loadSVM(@Str BytePointer filename, @Str BytePointer objname);
    public static native @Ptr @Name("load<cv::ml::SVM>") SVM loadSVM(@Str String filename, @Str String objname);
    public static native @Ptr @Name("load<cv::ml::EM>") EM loadEM(@Str BytePointer filename, @Str BytePointer objname);
    public static native @Ptr @Name("load<cv::ml::EM>") EM loadEM(@Str String filename, @Str String objname);
    public static native @Ptr @Name("load<cv::ml::DTrees>") DTrees loadDTrees(@Str BytePointer filename, @Str BytePointer objname);
    public static native @Ptr @Name("load<cv::ml::DTrees>") DTrees loadDTrees(@Str String filename, @Str String objname);
    public static native @Ptr @Name("load<cv::ml::RTrees>") RTrees loadRTrees(@Str BytePointer filename, @Str BytePointer objname);
    public static native @Ptr @Name("load<cv::ml::RTrees>") RTrees loadRTrees(@Str String filename, @Str String objname);
    public static native @Ptr @Name("load<cv::ml::Boost>") Boost loadBoost(@Str BytePointer filename, @Str BytePointer objname);
    public static native @Ptr @Name("load<cv::ml::Boost>") Boost loadBoost(@Str String filename, @Str String objname);
    public static native @Ptr @Name("load<cv::ml::ANN_MLP>") ANN_MLP loadANN_MLP(@Str BytePointer filename, @Str BytePointer objname);
    public static native @Ptr @Name("load<cv::ml::ANN_MLP>") ANN_MLP loadANN_MLP(@Str String filename, @Str String objname);
    public static native @Ptr @Name("load<cv::ml::LogisticRegression>") LogisticRegression loadLogisticRegression(@Str BytePointer filename, @Str BytePointer objname);
    public static native @Ptr @Name("load<cv::ml::LogisticRegression>") LogisticRegression loadLogisticRegression(@Str String filename, @Str String objname);
}
