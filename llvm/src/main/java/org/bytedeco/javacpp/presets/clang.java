/*
 * Copyright (C) 2015-2016 Samuel Audet
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

package org.bytedeco.javacpp.presets;

import org.bytedeco.javacpp.annotation.*;
import org.bytedeco.javacpp.tools.*;

@Properties(inherit = LLVM.class, target = "org.bytedeco.javacpp.clang", value = @Platform(value = {"linux-x86", "macosx"},
    include = {"<clang-c/Platform.h>", "<clang-c/CXErrorCode.h>", "<clang-c/CXString.h>", "<clang-c/CXCompilationDatabase.h>",
               "<clang-c/BuildSystem.h>", "<clang-c/Index.h>", "<clang-c/Documentation.h>"},
    compiler = "cpp11", link = "clang"))
public class clang implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("CINDEX_LINKAGE", "CINDEX_VERSION_STRING").cppTypes().annotations())
               .put(new Info("CINDEX_DEPRECATED").cppTypes().annotations("@Deprecated"))
               .put(new Info("__has_feature(blocks)").define(false))

               .put(new Info("CXVirtualFileOverlayImpl").pointerTypes("CXVirtualFileOverlay"))
               .put(new Info("CXModuleMapDescriptorImpl").pointerTypes("CXModuleMapDescriptor"))
               .put(new Info("CXTranslationUnitImpl").pointerTypes("CXTranslationUnit"))
               .put(new Info("CXCursorSetImpl").pointerTypes("CXCursorSet"))

               .put(new Info("CXVirtualFileOverlay").valueTypes("CXVirtualFileOverlay").pointerTypes("@ByPtrPtr CXVirtualFileOverlay", "@Cast(\"CXVirtualFileOverlay*\") PointerPointer"))
               .put(new Info("CXModuleMapDescriptor").valueTypes("CXModuleMapDescriptor").pointerTypes("@ByPtrPtr CXModuleMapDescriptor", "@Cast(\"CXModuleMapDescriptor*\") PointerPointer"))
               .put(new Info("CXTranslationUnit").valueTypes("CXTranslationUnit").pointerTypes("@ByPtrPtr CXTranslationUnit", "@Cast(\"CXTranslationUnit*\") PointerPointer"))
               .put(new Info("CXCursorSet").valueTypes("CXCursorSet").pointerTypes("@ByPtrPtr CXCursorSet", "@Cast(\"CXCursorSet*\") PointerPointer"));
    }
}
