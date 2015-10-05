/*
 * Copyright (C) 2014-2015 Samuel Audet
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
@Properties(target="org.bytedeco.javacpp.lept", value={
    @Platform(include={"leptonica/alltypes.h", "leptonica/environ.h", "leptonica/array.h",
        "leptonica/stack.h", "leptonica/imageio.h", "leptonica/morph.h", "leptonica/pix.h",
        "leptonica/allheaders_min.h"}, link="lept@.4"),
    @Platform(value="android", link="lept"),
    @Platform(value="windows", link="liblept", preload={"libwinpthread-1", "libgcc_s_dw2-1", "libgcc_s_seh-1", "libstdc++-6", "liblept-4"}),
    @Platform(value="windows-x86", preloadpath="C:/msys64/mingw32/bin/"),
    @Platform(value="windows-x86_64", preloadpath="C:/msys64/mingw64/bin/") })
public class lept implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("LEPT_DLL", "LIBJP2K_HEADER", "L_END_LIST").cppTypes().annotations())
               .put(new Info("L_TIMER").skip())
               .put(new Info("L_WallTimer").skip())
               .put(new Info("L_WALLTIMER").skip())
               .put(new Info("Numa").pointerTypes("NUMA"))
               .put(new Info("Numaa").skip())
               .put(new Info("NUMAA").skip())
               .put(new Info("Numa2d").skip())
               .put(new Info("NUMA2D").skip())
               .put(new Info("NumaHash").skip())
               .put(new Info("NUMAHASH").skip())
               .put(new Info("L_Dna").skip())
               .put(new Info("L_DNA").skip())
               .put(new Info("L_Dnaa").skip())
               .put(new Info("L_DNAA").skip())
               .put(new Info("Sarray").skip())
               .put(new Info("SARRAY").skip())
               .put(new Info("L_Bytea").skip())
               .put(new Info("L_BYTEA").skip())
               .put(new Info("L_Stack").pointerTypes("L_STACK"))
               .put(new Info("L_Compressed_Data").pointerTypes("L_COMP_DATA"))
               .put(new Info("L_Pdf_Data").skip())
               .put(new Info("L_PDF_DATA").skip())
               .put(new Info("Sel").pointerTypes("SEL"))
               .put(new Info("Sela").skip())
               .put(new Info("SELA").skip())
               .put(new Info("L_Kernel").skip())
               .put(new Info("L_KERNEL").skip())
               .put(new Info("Pix").pointerTypes("PIX"))
               .put(new Info("PixColormap").pointerTypes("PIXCMAP"))
               .put(new Info("RGBA_Quad").skip())
               .put(new Info("RGBA_QUAD").skip())
               .put(new Info("Pixa").pointerTypes("PIXA"))
               .put(new Info("Pixaa").pointerTypes("PIXAA"))
               .put(new Info("Box").pointerTypes("BOX"))
               .put(new Info("Boxa").pointerTypes("BOXA"))
               .put(new Info("Boxaa").pointerTypes("BOXAA"))
               .put(new Info("Pta").pointerTypes("PTA"))
               .put(new Info("Ptaa").skip())
               .put(new Info("PTAA").skip())
               .put(new Info("Pixacc").skip())
               .put(new Info("PIXACC").skip())
               .put(new Info("PixTiling").skip())
               .put(new Info("PIXTILING").skip())
               .put(new Info("FPix").skip())
               .put(new Info("FPIX").skip())
               .put(new Info("FPixa").skip())
               .put(new Info("FPIXA").skip())
               .put(new Info("DPix").skip())
               .put(new Info("DPIX").skip())
               .put(new Info("PixComp").skip())
               .put(new Info("PIXC").skip())
               .put(new Info("PixaComp").skip())
               .put(new Info("PIXAC").skip());
    }
}
