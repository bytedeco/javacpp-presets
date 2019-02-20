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
@Properties(inherit = cudart.class, value = {
    @Platform(define = {"CUBLASAPI", "CUBLAS_V2_H_"}, include = {"<cublas_api.h>", "<cublas.h>", "<cublasXt.h>"}, link = "cublas@.10.0"),
    @Platform(value = "windows-x86_64", preload = "cublas64_100")},
        target = "org.bytedeco.cuda.cublas", global = "org.bytedeco.cuda.global.cublas")
@NoException
public class cublas implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("CUBLASAPI", "CUBLASWINAPI").cppTypes().annotations().cppText(""))
               .put(new Info("cublasHandle_t").valueTypes("cublasContext").pointerTypes("@ByPtrPtr cublasContext"))
               .put(new Info("cublasStatus").cppTypes())
               .put(new Info("cublasXtHandle_t").valueTypes("cublasXtContext").pointerTypes("@ByPtrPtr cublasXtContext"));
    }
}
