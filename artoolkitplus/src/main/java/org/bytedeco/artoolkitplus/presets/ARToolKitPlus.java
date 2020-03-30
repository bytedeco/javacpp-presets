/*
 * Copyright (C) 2013-2020 Samuel Audet
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
 *
 *
 * WARNING: ARToolKitPlus itself is covered by the full GPLv3.
 * If your program uses this class, it will become bound to that license.
 */

package org.bytedeco.artoolkitplus.presets;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.presets.javacpp;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Samuel Audet
 */
@Properties(inherit=javacpp.class, target="org.bytedeco.artoolkitplus", global="org.bytedeco.artoolkitplus.global.ARToolKitPlus", value={
    @Platform(include={"ARToolKitPlus_plus.h", "<ARToolKitPlus/ARToolKitPlus.h>", "<ARToolKitPlus/config.h>", "<ARToolKitPlus/ar.h>",
        "<ARToolKitPlus/arMulti.h>", "<ARToolKitPlus/matrix.h>", "<ARToolKitPlus/vector.h>", "<ARToolKitPlus/Camera.h>",
        "<ARToolKitPlus/extra/BCH.h>", "<ARToolKitPlus/extra/Hull.h>", "<ARToolKitPlus/extra/rpp.h>",
        "<ARToolKitPlus/Tracker.h>", "<ARToolKitPlus/TrackerMultiMarker.h>", "<ARToolKitPlus/TrackerSingleMarker.h>",
        "<ARToolKitPlus/arBitFieldPattern.h>", "<ARToolKitPlus/arGetInitRot2Sub.h>"}, define="AR_STATIC", link="ARToolKitPlus"),
    @Platform(value="windows-x86", includepath="C:/Program Files (x86)/ARToolKitPlus/include/",
        linkpath="C:/Program Files (x86)/ARToolKitPlus/lib/"),
    @Platform(value="windows-x86_64", includepath="C:/Program Files/ARToolKitPlus/include/",
        linkpath="C:/Program Files/ARToolKitPlus/lib/") })
public class ARToolKitPlus implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "artoolkitplus"); }

    public void map(InfoMap infoMap) {
          infoMap.put(new Info("AR_EXPORT").cppTypes().annotations())
                 .put(new Info("defined(_MSC_VER) || defined(_WIN32_WCE)").define(false))
                 .put(new Info("ARToolKitPlus::IDPATTERN").cast().valueTypes("long").pointerTypes("LongPointer", "LongBuffer", "long[]"))
                 .put(new Info("ARFloat").cast().valueTypes("float").pointerTypes("FloatPointer", "FloatBuffer", "float[]"))
                 .put(new Info("ARToolKitPlus::_64bits").cast().valueTypes("long").pointerTypes("LongPointer", "LongBuffer", "long[]"))
                 .put(new Info("rpp_vec").cast().valueTypes("DoublePointer").pointerTypes("PointerPointer"))
                 .put(new Info("rpp_mat").valueTypes("@Cast(\"double(*)[3]\") DoublePointer").pointerTypes("@Cast(\"double(*)[3][3]\") PointerPointer"));
    }
}
