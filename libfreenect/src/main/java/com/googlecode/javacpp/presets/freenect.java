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
@Properties(target="com.googlecode.javacpp.freenect", value={
    @Platform(include={"<libfreenect.h>", "<libfreenect-registration.h>", "<libfreenect-audio.h>", "<libfreenect_sync.h>"},
        includepath={"/usr/local/include/libfreenect/", "/opt/local/include/libfreenect/", "/usr/include/libfreenect/"},
        link={"freenect@0.2", "freenect_sync@0.2"}),
    @Platform(value="windows-x86", includepath="C:/Program Files (x86)/libfreenect/include/libfreenect/",
        linkpath="C:/Program Files (x86)/libfreenect/lib/"),
    @Platform(value="windows-x86_64", includepath="C:/Program Files/libfreenect/include/libfreenect/",
        linkpath="C:/Program Files/libfreenect/lib/") })
public class freenect implements Parser.InfoMapper {
    public void map(Parser.InfoMap infoMap) {
        infoMap.put(new Parser.Info("FREENECTAPI").genericTypes())
               .put(new Parser.Info("freenect_device_attributes").forwardDeclared(true))
               .put(new Parser.Info("timeval").pointerTypes("Pointer").cast(true));
    }
}
