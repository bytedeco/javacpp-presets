/*
 * Copyright (C) 2013,2014,2015 Samuel Audet
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
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Samuel Audet
 */
@Properties(inherit = opencv_imgproc.class, value = {
    @Platform(include = {"<opencv2/photo/photo_c.h>", "<opencv2/photo.hpp>", "<opencv2/photo/cuda.hpp>"},
              link = "opencv_photo@.3.0", preload = "opencv_cuda@.3.0"),
    @Platform(value = "windows", link = "opencv_photo300", preload = "opencv_cuda300")},
        target = "org.bytedeco.javacpp.opencv_photo")
public class opencv_photo implements InfoMapper {
    public void map(InfoMap infoMap) {
    }
}
