/*
 * Copyright (C) 2014-2023 Samuel Audet
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

package org.bytedeco.tesseract.presets;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Cast;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.annotation.MemberGetter;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

import org.bytedeco.leptonica.presets.leptonica;

/**
 *
 * @author Samuel Audet
 */
@Properties(target = "org.bytedeco.tesseract", global = "org.bytedeco.tesseract.global.tesseract", inherit = leptonica.class, value = {
    @Platform(define = "TESS_CAPI_INCLUDE_BASEAPI", include = {"tesseract/export.h", /*"tesseract/osdetect.h",*/ "tesseract/unichar.h",
        "tesseract/version.h", "tesseract/publictypes.h", "tesseract/pageiterator.h", "tesseract/ocrclass.h", "tesseract/ltrresultiterator.h",
        "tesseract/renderer.h", "tesseract/resultiterator.h", "tesseract/baseapi.h", "tesseract/capi.h", "locale.h"},
        compiler = "cpp11", link = "tesseract@.5.3.3"/*, resource = {"include", "lib"}*/),
    @Platform(value = "android", link = "tesseract"),
    @Platform(value = "windows", link = "tesseract53", preload = "libtesseract53") })
public class tesseract implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "tesseract"); }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info("locale.h").skip())
               .put(new Info("__NATIVE__", "ultoa", "snprintf", "vsnprintf", "SIGNED",
                             "TESS_API", "TESS_LOCAL", "_TESS_FILE_BASENAME_", "TESS_CALL").cppTypes().annotations().cppText(""))
               .put(new Info("STRING_IS_PROTECTED").define(false))
               .put(new Info("BOOL").cast().valueTypes("boolean").pointerTypes("BoolPointer").define())
               .put(new Info("__cplusplus", "TESS_CAPI_INCLUDE_BASEAPI").define())
               .put(new Info("std::string").annotations("@StdString").valueTypes("BytePointer", "String").pointerTypes("@Cast({\"char*\", \"std::string*\"}) BytePointer"))
               .put(new Info("std::vector<char>").pointerTypes("ByteVector").define())
               .put(new Info("std::vector<std::string>").pointerTypes("StringVector").define())
               .put(new Info("std::chrono::steady_clock::time_point").cast().pointerTypes("Pointer"))
               .put(new Info("MIN_INT32").javaText("public static final int MIN_INT32 = 0x80000000;"))

               .put(new Info("Pix").pointerTypes("PIX").skip())
               .put(new Info("Pta").pointerTypes("PTA").skip())
               .put(new Info("Box").pointerTypes("BOX").skip())
               .put(new Info("Pixa").pointerTypes("PIXA").skip())
               .put(new Info("Boxa").pointerTypes("BOXA").skip())

               .put(new Info("std::vector<std::vector<std::pair<const char*,float> > >").pointerTypes("StringFloatPairVectorVector").define())

               .put(new Info("TessResultCallback1<bool,int>").pointerTypes("DeleteCallback").define().virtualize())

               .put(new Info("TessCallback1<char>").pointerTypes("CharClearCallback").define().virtualize())
               .put(new Info("TessResultCallback2<bool,const char&,const char&>").pointerTypes("CharCompareCallback").define().virtualize())
               .put(new Info("TessResultCallback2<bool,FILE*,const char&>").pointerTypes("CharWriteCallback").define().virtualize())
               .put(new Info("TessResultCallback2<bool,tesseract::TFile*,char*>").pointerTypes("CharReadCallback").define().virtualize())
               .put(new Info("GenericVector<char>").pointerTypes("CharGenericVector").define())

               .put(new Info("TessCallback1<STRING>").pointerTypes("StringClearCallback").define().virtualize())
               .put(new Info("TessResultCallback2<bool,const STRING&,const STRING&>").pointerTypes("StringCompareCallback").define().virtualize())
               .put(new Info("TessResultCallback2<bool,FILE*,const STRING&>").pointerTypes("StringWriteCallback").define().virtualize())
               .put(new Info("TessResultCallback2<bool,tesseract::TFile*,STRING*>").pointerTypes("StringReadCallback").define().virtualize())
               .put(new Info("GenericVector<STRING>").pointerTypes("StringGenericVector").define())
               .put(new Info("GenericVector<STRING>::WithinBounds").skip())

               .put(new Info("TessCallback1<int>").pointerTypes("IntClearCallback").define().virtualize())
               .put(new Info("TessResultCallback2<bool,const int&,const int&>").pointerTypes("IntCompareCallback").define().virtualize())
               .put(new Info("TessResultCallback2<bool,FILE*,const int&>").pointerTypes("IntWriteCallback").define().virtualize())
               .put(new Info("TessResultCallback2<bool,tesseract::TFile*,int*>").pointerTypes("IntReadCallback").define().virtualize())
               .put(new Info("GenericVector<int>").pointerTypes("IntGenericVector").define())
               .put(new Info("GenericVector<StrongScriptDirection>").cast().pointerTypes("IntGenericVector"))
               .put(new Info("GenericVectorEqEq<int>").pointerTypes("IntGenericVectorEqEq").define())

               .put(new Info("GenericVector<char>::delete_data_pointers", "GenericVector<char>::SerializeClasses", "GenericVector<char>::DeSerializeClasses",
                             "GenericVector<int>::delete_data_pointers", "GenericVector<int>::SerializeClasses", "GenericVector<int>::DeSerializeClasses",
                             "GenericVector<char>::SkipDeSerialize", "GenericVector<char>::SkipDeSerializeClasses",
                             "GenericVector<int>::SkipDeSerialize", "GenericVector<int>::SkipDeSerializeClasses",
                             "GenericVector<STRING>::contains_index", "GenericVector<STRING>::delete_data_pointers", "GenericVector<STRING>::binary_search",
                             "GenericVector<STRING>::bool_binary_search", "GenericVector<STRING>::choose_nth_item", "GenericVector<STRING>::dot_product",
                             "GenericVector<STRING>::sort", "GenericVectorEqEq<int>::GenericVectorEqEq<int>(int)").skip())

               .put(new Info("TessCallback3<const UNICHARSET&,int,PAGE_RES*>").pointerTypes("TruthCallback3").define().virtualize())
               .put(new Info("TessCallback4<const UNICHARSET&,int,tesseract::PageIterator*,Pix*>").pointerTypes("TruthCallback4").define().virtualize())

               .put(new Info("list_rec").cppText("#define list_rec LIST"))
               .put(new Info("INT_FEATURE_ARRAY").valueTypes("INT_FEATURE_STRUCT"))

               .put(new Info("tesseract::ImageThresholder()").javaText(""))

               .put(new Info("TessBaseAPISetFillLatticeFunc", "TessBaseGetBlockTextOrientations", "TessBaseAPIInit", "TessCallbackUtils_::FailIsRepeatable", "LPBLOB", "kPolyBlockNames",
                             "tesseract::TessBaseAPI::SetFillLatticeFunc", "tesseract::ImageThresholder::GetPixRectGrey", "tesseract::ImageThresholder::GetPixRect",
                             "tesseract::ImageThresholder::SetRectangle", "tesseract::ImageThresholder::SetImage", "tesseract::ImageThresholder::IsEmpty", "tesseract::HOcrEscape",
                             "tesseract::ResultIterator::kComplexWord", "tesseract::ResultIterator::kMinorRunEnd", "tesseract::ResultIterator::kMinorRunStart",
                             "UNICHAR::utf8_step", "UNICHAR::utf8_str", "UNICHAR::first_uni", "UNICHAR::UNICHAR(char*, int)", "UNICHAR::UNICHAR(int)").skip());
    }

    public static native @MemberGetter int LC_ALL();
    public static native @MemberGetter int LC_COLLATE();
    public static native @MemberGetter int LC_CTYPE();
    public static native @MemberGetter int LC_MONETARY();
    public static native @MemberGetter int LC_NUMERIC();
    public static native @MemberGetter int LC_TIME();

    public static native String setlocale(int category, String locale);
    public static native @Cast("char*") BytePointer setlocale(int category, @Cast("const char*") BytePointer locale);
}
