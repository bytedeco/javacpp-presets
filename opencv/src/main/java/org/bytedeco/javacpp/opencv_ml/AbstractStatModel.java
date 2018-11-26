package org.bytedeco.javacpp.opencv_ml;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Name;
import org.bytedeco.javacpp.opencv_core.Algorithm;
import org.bytedeco.javacpp.opencv_core.opencv_core_presets;

@Name("cv::ml::StatModel") public abstract class AbstractStatModel extends Algorithm {
    static { Loader.load(); }
    public AbstractStatModel(Pointer p) { super(p); }

    public static native @opencv_core_presets.Ptr
    @Name("load<cv::ml::NormalBayesClassifier>") NormalBayesClassifier loadNormalBayesClassifier(@opencv_core_presets.Str BytePointer filename, @opencv_core_presets.Str BytePointer objname);
    public static native @opencv_core_presets.Ptr
    @Name("load<cv::ml::NormalBayesClassifier>") NormalBayesClassifier loadNormalBayesClassifier(@opencv_core_presets.Str String filename, @opencv_core_presets.Str String objname);
    public static native @opencv_core_presets.Ptr
    @Name("load<cv::ml::KNearest>") KNearest loadKNearest(@opencv_core_presets.Str BytePointer filename, @opencv_core_presets.Str BytePointer objname);
    public static native @opencv_core_presets.Ptr
    @Name("load<cv::ml::KNearest>") KNearest loadKNearest(@opencv_core_presets.Str String filename, @opencv_core_presets.Str String objname);
    public static native @opencv_core_presets.Ptr
    @Name("load<cv::ml::SVM>") SVM loadSVM(@opencv_core_presets.Str BytePointer filename, @opencv_core_presets.Str BytePointer objname);
    public static native @opencv_core_presets.Ptr
    @Name("load<cv::ml::SVM>") SVM loadSVM(@opencv_core_presets.Str String filename, @opencv_core_presets.Str String objname);
    public static native @opencv_core_presets.Ptr
    @Name("load<cv::ml::EM>") EM loadEM(@opencv_core_presets.Str BytePointer filename, @opencv_core_presets.Str BytePointer objname);
    public static native @opencv_core_presets.Ptr
    @Name("load<cv::ml::EM>") EM loadEM(@opencv_core_presets.Str String filename, @opencv_core_presets.Str String objname);
    public static native @opencv_core_presets.Ptr
    @Name("load<cv::ml::DTrees>") DTrees loadDTrees(@opencv_core_presets.Str BytePointer filename, @opencv_core_presets.Str BytePointer objname);
    public static native @opencv_core_presets.Ptr
    @Name("load<cv::ml::DTrees>") DTrees loadDTrees(@opencv_core_presets.Str String filename, @opencv_core_presets.Str String objname);
    public static native @opencv_core_presets.Ptr
    @Name("load<cv::ml::RTrees>") RTrees loadRTrees(@opencv_core_presets.Str BytePointer filename, @opencv_core_presets.Str BytePointer objname);
    public static native @opencv_core_presets.Ptr
    @Name("load<cv::ml::RTrees>") RTrees loadRTrees(@opencv_core_presets.Str String filename, @opencv_core_presets.Str String objname);
    public static native @opencv_core_presets.Ptr
    @Name("load<cv::ml::Boost>") Boost loadBoost(@opencv_core_presets.Str BytePointer filename, @opencv_core_presets.Str BytePointer objname);
    public static native @opencv_core_presets.Ptr
    @Name("load<cv::ml::Boost>") Boost loadBoost(@opencv_core_presets.Str String filename, @opencv_core_presets.Str String objname);
    public static native @opencv_core_presets.Ptr
    @Name("load<cv::ml::ANN_MLP>") ANN_MLP loadANN_MLP(@opencv_core_presets.Str BytePointer filename, @opencv_core_presets.Str BytePointer objname);
    public static native @opencv_core_presets.Ptr
    @Name("load<cv::ml::ANN_MLP>") ANN_MLP loadANN_MLP(@opencv_core_presets.Str String filename, @opencv_core_presets.Str String objname);
    public static native @opencv_core_presets.Ptr
    @Name("load<cv::ml::LogisticRegression>") LogisticRegression loadLogisticRegression(@opencv_core_presets.Str BytePointer filename, @opencv_core_presets.Str BytePointer objname);
    public static native @opencv_core_presets.Ptr
    @Name("load<cv::ml::LogisticRegression>") LogisticRegression loadLogisticRegression(@opencv_core_presets.Str String filename, @opencv_core_presets.Str String objname);
}
