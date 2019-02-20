/*
 * Copyright (C) 2015-2018 Samuel Audet
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

package org.bytedeco.cuda.presets;

import org.bytedeco.javacpp.annotation.NoException;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Samuel Audet
 */
@Properties(inherit = nppc.class, value = {
    @Platform(include = {"<nppi.h>", "<nppi_support_functions.h>", "<nppi_data_exchange_and_initialization.h>",
            "<nppi_arithmetic_and_logical_operations.h>", "<nppi_color_conversion.h>",
            "<nppi_threshold_and_compare_operations.h>", "<nppi_morphological_operations.h>",
            "<nppi_filtering_functions.h>", "<nppi_statistics_functions.h>",
            "<nppi_linear_transforms.h>", "<nppi_geometry_transforms.h>",
            "<nppi_compression_functions.h>", "<nppi_computer_vision.h>"},
            link = {"nppial@.10.0", "nppicc@.10.0", "nppicom@.10.0", "nppidei@.10.0", "nppif@.10.0",
                    "nppig@.10.0", "nppim@.10.0", "nppist@.10.0", "nppisu@.10.0", "nppitc@.10.0"}),
    @Platform(value = "windows-x86_64",
            preload = {"nppial64_100", "nppicc64_100", "nppicom64_100", "nppidei64_100", "nppif64_100",
                       "nppig64_100", "nppim64_100", "nppist64_100", "nppisu64_100", "nppitc64_100"})},
        target = "org.bytedeco.cuda.nppi", global = "org.bytedeco.cuda.global.nppi")
@NoException
public class nppi implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("nppiHLSToBGR_8u_AC4R", "nppiNormRelInfGetBufferHostSize_32s_C1R", "nppiSub_32s_C4IRSfs", "nppiSub_32s_C4RSfs").skip());
    }
}
