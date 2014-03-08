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
@Properties(target="com.googlecode.javacpp.videoInputLib", value={
    @Platform(value="windows", include={"<videoInput.h>", "<videoInput.cpp>"},
        includepath={"../videoInput-update2013/videoInputSrcAndDemos/libs/videoInput/",
                     "../videoInput-update2013/videoInputSrcAndDemos/libs/DShow/Include/"},
        link={"ole32", "oleaut32", "amstrmid", "strmiids", "uuid"}) })
public class videoInputLib implements Parser.InfoMapper {
    public void map(Parser.InfoMap infoMap) {
          infoMap.put(new Parser.Info("videoInput.cpp").skip(true))
                 .put(new Parser.Info("_WIN32_WINNT").cppTypes().define(false))
                 .put(new Parser.Info("std::vector<std::string>").pointerTypes("StringVector").define(true))
                 .put(new Parser.Info("GUID").cast(true).pointerTypes("Pointer"))
                 .put(new Parser.Info("long", "unsigned long").cast(true).valueTypes("int").pointerTypes("IntPointer", "IntBuffer", "int[]"));
    }
}
