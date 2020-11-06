/*
 * Copyright (C) 2015-2020 Samuel Audet
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
@Properties(inherit = {cublas.class, cusparse.class}, value = {
    @Platform(include = {"<cusolver_common.h>", "<cusolverDn.h>", "<cusolverMg.h>", "<cusolverRf.h>", "cusolverSp.h"},
        link = {"cusolver@.11", "cusolverMg@.11"}),
    @Platform(value = "windows-x86_64", preload = {"cusolver64_11", "cusolverMg64_11"})},
        target = "org.bytedeco.cuda.cusolver", global = "org.bytedeco.cuda.global.cusolver")
@NoException
public class cusolver implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("CUDENSEAPI", "CRFWINAPI", "CUSOLVERAPI", "CUSOLVER_CPP_VERSION").cppTypes().annotations().cppText(""))
               .put(new Info("cusolverDnCunmtr_bufferSize", "cusolverDnDormtr_bufferSize", "cusolverDnZunmtr_bufferSize", "cusolverDnZunmtr",
                             "cusolverDnSormtr_bufferSize", "cusolverDnDormtr", "cusolverDnCunmtr", "cusolverDnSormtr").skip())
               .put(new Info("cusolverDnHandle_t").valueTypes("cusolverDnContext").pointerTypes("@ByPtrPtr cusolverDnContext"))
               .put(new Info("cusolverDnParams_t").valueTypes("cusolverDnParams").pointerTypes("@ByPtrPtr cusolverDnParams"))
               .put(new Info("syevjInfo_t").valueTypes("syevjInfo").pointerTypes("@ByPtrPtr syevjInfo"))
               .put(new Info("gesvdjInfo_t").valueTypes("gesvdjInfo").pointerTypes("@ByPtrPtr gesvdjInfo"))
               .put(new Info("cusolverDnIRSParams_t").valueTypes("cusolverDnIRSParams").pointerTypes("@ByPtrPtr cusolverDnIRSParams"))
               .put(new Info("cusolverDnIRSInfos_t").valueTypes("cusolverDnIRSInfos").pointerTypes("@ByPtrPtr cusolverDnIRSInfos"))
               .put(new Info("cusolverMgHandle_t").valueTypes("cusolverMgContext").pointerTypes("@ByPtrPtr cusolverMgContext"))
               .put(new Info("cusolverRfHandle_t").valueTypes("cusolverRfCommon").pointerTypes("@ByPtrPtr cusolverRfCommon"))
               .put(new Info("cusolverSpHandle_t").valueTypes("cusolverSpContext").pointerTypes("@ByPtrPtr cusolverSpContext"))
               .put(new Info("csrqrInfo_t").valueTypes("csrqrInfo").pointerTypes("@ByPtrPtr csrqrInfo"));
    }
}
