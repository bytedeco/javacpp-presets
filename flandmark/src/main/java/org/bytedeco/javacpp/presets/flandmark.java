/*
 * Copyright (C) 2014 Samuel Audet, Jarek Sacha
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
 * @author Jarek Sacha
 */
@Properties(target = "org.bytedeco.javacpp.flandmark", value = {
        @Platform(value = "windows", preload = {"msvcr100", "msvcp100"},
                link = {"opencv_core249", "opencv_imgproc249", "flandmark_static"},
                include = {"<flandmark_detector.h>"}),
        @Platform(value = "windows-x86",
                includepath = {"../../opencv/cppbuild/windows-x86/include/opencv/",
                        "../../opencv/cppbuild/windows-x86/include/opencv2/"},
                linkpath = {"../../opencv/cppbuild/windows-x86/lib/"},
                preloadpath = {"../../opencv/cppbuild/windows-x86/bin/",
                        "C:/Program Files (x86)/Microsoft Visual Studio 10.0/VC/redist/x86/Microsoft.VC100.CRT/"}),
        @Platform(value = "windows-x86_64",
                includepath = {"../../opencv/cppbuild/windows-x86_64/include/opencv/",
                        "../../opencv/cppbuild/windows-x86_64/include/opencv2/"},
                linkpath = {"../../opencv/cppbuild/windows-x86_64/lib/"},
                preloadpath = {"../../opencv/cppbuild/windows-x86_64/bin/",
                        "C:/Program Files (x86)/Microsoft Visual Studio 10.0/VC/redist/x64/Microsoft.VC100.CRT/"})})
public class flandmark implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("flandmark_detector.cpp").skip())
                .put(new Info().javaText("import static org.bytedeco.javacpp.opencv_core.*;"));
    }
}
