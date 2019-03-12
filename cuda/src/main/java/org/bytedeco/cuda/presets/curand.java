/*
 * Copyright (C) 2015-2019 Samuel Audet
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
@Properties(inherit = cudart.class, value = {
    @Platform(include = "<curand.h>", link = "curand@.10"),
    @Platform(value = "windows-x86_64", preload = "curand64_10")},
        target = "org.bytedeco.cuda.curand", global = "org.bytedeco.cuda.global.curand")
@NoException
public class curand implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("CURANDAPI").cppTypes().annotations().cppText(""))
               .put(new Info("curandGenerateBinomial", "curandGenerateBinomialMethod").skip())
               .put(new Info("curandGenerator_t").valueTypes("curandGenerator_st").pointerTypes("@ByPtrPtr curandGenerator_st"))
               .put(new Info("curandDistribution_t").valueTypes("curandDistribution_st").pointerTypes("@ByPtrPtr curandDistribution_st"))
               .put(new Info("curandDistributionM2Shift_t").valueTypes("curandDistributionM2Shift_st").pointerTypes("@ByPtrPtr curandDistributionM2Shift_st"))
               .put(new Info("curandHistogramM2_t").valueTypes("curandHistogramM2_st").pointerTypes("@ByPtrPtr curandHistogramM2_st"))
               .put(new Info("curandHistogramM2K_t").valueTypes("curandHistogramM2K_st").pointerTypes("@ByPtrPtr curandHistogramM2K_st"))
               .put(new Info("curandHistogramM2V_t").valueTypes("curandHistogramM2V_st").pointerTypes("@ByPtrPtr curandHistogramM2V_st"))
               .put(new Info("curandDiscreteDistribution_t").valueTypes("curandDiscreteDistribution_st").pointerTypes("@ByPtrPtr curandDiscreteDistribution_st"));
    }
}
