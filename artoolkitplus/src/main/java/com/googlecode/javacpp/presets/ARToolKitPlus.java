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
@Properties(target="com.googlecode.javacpp.ARToolKitPlus", value={
    @Platform(include={"<ARToolKitPlus/ARToolKitPlus.h>", "<ARToolKitPlus/config.h>", "<ARToolKitPlus/ar.h>", 
        "<ARToolKitPlus/arMulti.h>", "<ARToolKitPlus/matrix.h>", "<ARToolKitPlus/vector.h>", "<ARToolKitPlus/Camera.h>",
        "<ARToolKitPlus/extra/BCH.h>", "<ARToolKitPlus/extra/Hull.h>", "<ARToolKitPlus/extra/rpp.h>", 
        "<ARToolKitPlus/Tracker.h>", "<ARToolKitPlus/TrackerMultiMarker.h>", "<ARToolKitPlus/TrackerSingleMarker.h>",
        "<ARToolKitPlus/arBitFieldPattern.h>", "<ARToolKitPlus/arGetInitRot2Sub.h>"},  link="ARToolKitPlus"),
    @Platform(value="windows-x86", includepath="C:/Program Files (x86)/ARToolKitPlus/include/",
        linkpath="C:/Program Files (x86)/ARToolKitPlus/lib/"),
    @Platform(value="windows-x86_64", includepath="C:/Program Files/ARToolKitPlus/include/",
        linkpath="C:/Program Files/ARToolKitPlus/lib/") })
public class ARToolKitPlus implements Parser.InfoMapper {
    public void map(Parser.InfoMap infoMap) {
          infoMap.put(new Parser.Info("AR_EXPORT").genericTypes().annotations())
                 .put(new Parser.Info("ARMat").opaque(false))
                 .put(new Parser.Info("std::vector<CornerPoint>", "CornerPoints").pointerTypes("@StdVector CornerPoint"))
                 .put(new Parser.Info("std::vector<int>").valueTypes("@StdVector int[]"))
                 .put(new Parser.Info("ARFloat").valueTypes("float").pointerTypes("FloatPointer", "FloatBuffer", "float[]").cast(true))
                 .put(new Parser.Info("ARToolKitPlus::_64bits").valueTypes("long").pointerTypes("LongPointer", "LongBuffer", "long[]").cast(true))
                 .put(new Parser.Info("rpp_vec").valueTypes("DoublePointer").pointerTypes("PointerPointer").cast(true))
                 .put(new Parser.Info("rpp_mat").valueTypes("@Cast(\"double(*)[3]\") DoublePointer").pointerTypes("@Cast(\"double(*)[3][3]\") PointerPointer"))
                 .put(new Parser.Info("defined(_MSC_VER) || defined(_WIN32_WCE)").define(false));
    }
}
