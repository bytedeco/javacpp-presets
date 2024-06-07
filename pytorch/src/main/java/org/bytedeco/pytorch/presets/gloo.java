/*
 * Copyright (C) 2023 Hervé Guillemet
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
package org.bytedeco.pytorch.presets;

import org.bytedeco.javacpp.ClassProperties;
import org.bytedeco.javacpp.LoadEnabled;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.*;

import static org.bytedeco.pytorch.presets.torch.template;

/**
 * @author Hervé Guillemet
 */
@Properties(
    inherit = torch.class,
    value = {
        @Platform(
            library = "jnitorch"
        )
    },
    target = "org.bytedeco.pytorch.gloo",
    global = "org.bytedeco.pytorch.global.gloo"
)
public class gloo implements LoadEnabled, InfoMapper {

    @Override
    public void init(ClassProperties properties) {
        torch.initIncludes(getClass(), properties);
    }

    @Override
    public void map(InfoMap infoMap) {

        infoMap
            .put(new Info().javaText("import org.bytedeco.pytorch.chrono.*;"))
        ;

        //// Instantiation of class templates.
        infoMap
            .put(new Info("gloo::ReductionFunction<float>").pointerTypes("ReductionFunctionFloat"))
            .put(new Info("gloo::ReductionFunction<int>").pointerTypes("ReductionFunctionInt"))
        ;

        //// Hopefully will skip only the initializers, not the fields:
        infoMap
            .put(new Info("ReductionFunction<T>::sum").skip())
            .put(new Info("ReductionFunction<T>::product").skip())
            .put(new Info("ReductionFunction<T>::min").skip())
            .put(new Info("ReductionFunction<T>::max").skip())
        ;

        //// Renaming to avoid clashes
        infoMap
            .put(new Info("gloo::transport::Context").pointerTypes("TransportContext"))
        ;

        infoMap
            .put(new Info("__CUDA_ARCH__").define(false))
        ;

        infoMap.put(new Info("gloo::kOnDeviceThreshold").javaText("public static final long kOnDeviceThreshold = 256 * 1024;"));

        new torch.PointerInfo("gloo::Context").makeShared(infoMap);
        new torch.PointerInfo("gloo::transport::Context").javaBaseName("TransportContext").makeShared(infoMap);
        new torch.PointerInfo("gloo::transport::Device").makeShared(infoMap);

        //// Unsure if instantiating these templates could have any interest
        //// for a use from Pytorch
        infoMap
            .put(new Info("gloo::sum", "gloo::product", "gloo::max", "gloo::min").skip())
        ;
    }
}
