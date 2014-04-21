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
 *
 *
 * WARNING: ARToolKitPlus itself is covered by the full GPLv3.
 * If your program uses this class, it will become bound to that license.
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
@Properties(target="org.bytedeco.javacpp.ARToolKitPlus", value={
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
