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
    @Platform(include = "<cusparse.h>", link = "cusparse@.10.0"),
    @Platform(value = "windows-x86_64", preload = "cusparse64_100")},
        target = "org.bytedeco.cuda.cusparse", global = "org.bytedeco.cuda.global.cusparse")
@NoException
public class cusparse implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("CUSPARSEAPI").cppTypes().annotations().cppText(""))
               .put(new Info("cusparseHandle_t").valueTypes("cusparseContext").pointerTypes("@ByPtrPtr cusparseContext"))
               .put(new Info("cusparseMatDescr_t").valueTypes("cusparseMatDescr").pointerTypes("@ByPtrPtr cusparseMatDescr"))
               .put(new Info("cusparseSolveAnalysisInfo_t").valueTypes("cusparseSolveAnalysisInfo").pointerTypes("@ByPtrPtr cusparseSolveAnalysisInfo"))
               .put(new Info("csrsv2Info_t").valueTypes("csrsv2Info").pointerTypes("@ByPtrPtr csrsv2Info"))
               .put(new Info("csrsm2Info_t").valueTypes("csrsm2Info").pointerTypes("@ByPtrPtr csrsm2Info"))
               .put(new Info("bsrsv2Info_t").valueTypes("bsrsv2Info").pointerTypes("@ByPtrPtr bsrsv2Info"))
               .put(new Info("bsrsm2Info_t").valueTypes("bsrsm2Info").pointerTypes("@ByPtrPtr bsrsm2Info"))
               .put(new Info("csric02Info_t").valueTypes("csric02Info").pointerTypes("@ByPtrPtr csric02Info"))
               .put(new Info("bsric02Info_t").valueTypes("bsric02Info").pointerTypes("@ByPtrPtr bsric02Info"))
               .put(new Info("csrilu02Info_t").valueTypes("csrilu02Info").pointerTypes("@ByPtrPtr csrilu02Info"))
               .put(new Info("bsrilu02Info_t").valueTypes("bsrilu02Info").pointerTypes("@ByPtrPtr bsrilu02Info"))
               .put(new Info("cusparseHybMat_t").valueTypes("cusparseHybMat").pointerTypes("@ByPtrPtr cusparseHybMat"))
               .put(new Info("csrgemm2Info_t").valueTypes("csrgemm2Info").pointerTypes("@ByPtrPtr csrgemm2Info"))
               .put(new Info("csru2csrInfo_t").valueTypes("csru2csrInfo").pointerTypes("@ByPtrPtr csru2csrInfo"))
               .put(new Info("cusparseColorInfo_t").valueTypes("cusparseColorInfo").pointerTypes("@ByPtrPtr cusparseColorInfo"))
               .put(new Info("pruneInfo_t").valueTypes("pruneInfo").pointerTypes("@ByPtrPtr pruneInfo"));
    }
}
