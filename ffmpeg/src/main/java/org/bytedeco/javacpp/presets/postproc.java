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
@Properties(inherit=avutil.class, target="org.bytedeco.javacpp.postproc", value={
    @Platform(cinclude="<libpostproc/postprocess.h>", link="postproc@.52"),
    @Platform(value="windows", link="postproc-52") })
public class postproc implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("QP_STORE_T").cppTypes().valueTypes("byte").pointerTypes("BytePointer"))
               .put(new Info("LIBPOSTPROC_VERSION_INT < (52<<16)").define(false));
    }
}
