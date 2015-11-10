/*
 * Copyright (C) 2015 Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.bytedeco.javacpp.helper;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Name;

// required by javac to resolve circular dependencies
import org.bytedeco.javacpp.opencv_core.Algorithm;
import org.bytedeco.javacpp.opencv_ml.ANN_MLP;
import org.bytedeco.javacpp.opencv_ml.Boost;
import org.bytedeco.javacpp.opencv_ml.DTrees;
import org.bytedeco.javacpp.opencv_ml.EM;
import org.bytedeco.javacpp.opencv_ml.KNearest;
import org.bytedeco.javacpp.opencv_ml.LogisticRegression;
import org.bytedeco.javacpp.opencv_ml.NormalBayesClassifier;
import org.bytedeco.javacpp.opencv_ml.RTrees;
import org.bytedeco.javacpp.opencv_ml.SVM;
import org.bytedeco.javacpp.presets.opencv_core.Ptr;
import org.bytedeco.javacpp.presets.opencv_core.Str;

public class opencv_ml extends org.bytedeco.javacpp.presets.opencv_ml {

    @Name("cv::ml::StatModel") public static abstract class AbstractStatModel extends Algorithm {
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
}
