/*
 * Copyright (C) 2014 Samuel Audet
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
 *
 *
 * Permission was received from Point Grey Research, Inc. to disclose the
 * information released by the application of this preset under the GPL,
 * as long as it is distributed as part of a substantially larger package,
 * and not as a standalone wrapper to the FlyCapture library.
 *
 */

package org.bytedeco.javacpp.presets;

import java.nio.ByteBuffer;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Samuel Audet
 */
@Properties(target="org.bytedeco.javacpp.PGRFlyCapture", value={
    @Platform(value="windows", link="PGRFlyCapture", preload="FlyCapture2",
        include={"<windows.h>", "<PGRFlyCapture.h>", "<PGRFlyCapturePlus.h>","<PGRFlyCaptureMessaging.h>"},
        includepath={"C:/Program Files/Point Grey Research/PGR FlyCapture/include/",
                     "C:/Program Files/Point Grey Research/FlyCapture2/include/FC1/"}),
    @Platform(value="windows-x86",
        linkpath   ={"C:/Program Files/Point Grey Research/PGR FlyCapture/lib/",
                     "C:/Program Files/Point Grey Research/FlyCapture2/lib/FC1/",
                     "C:/Program Files (x86)/Point Grey Research/PGR FlyCapture/lib/",
                     "C:/Program Files (x86)/Point Grey Research/FlyCapture2/lib/FC1/"},
        preloadpath={"C:/Program Files/Point Grey Research/PGR FlyCapture/bin/",
                     "C:/Program Files/Point Grey Research/FlyCapture2/bin/",
                     "C:/Program Files/Point Grey Research/FlyCapture2/bin/FC1/",
                     "C:/Program Files (x86)/Point Grey Research/PGR FlyCapture/bin/",
                     "C:/Program Files (x86)/Point Grey Research/FlyCapture2/bin/",
                     "C:/Program Files (x86)/Point Grey Research/FlyCapture2/bin/FC1/"}),
    @Platform(value="windows-x86_64",
        linkpath   ={"C:/Program Files/Point Grey Research/PGR FlyCapture/lib64/",
                     "C:/Program Files/Point Grey Research/FlyCapture2/lib64/FC1/" },
        preloadpath={"C:/Program Files/Point Grey Research/PGR FlyCapture/bin64/",
                     "C:/Program Files/Point Grey Research/FlyCapture2/bin64/",
                     "C:/Program Files/Point Grey Research/FlyCapture2/bin64/FC1/"}) })
public class PGRFlyCapture implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("windows.h").skip())
               .put(new Info("PGRFLYCAPTURE_API", "PGRFLYCAPTURE_CALL_CONVEN").cppTypes().annotations().cppText(""))
               .put(new Info("FlyCaptureContext").valueTypes("FlyCaptureContext")
                       .pointerTypes("@Cast(\"FlyCaptureContext*\") @ByPtrPtr FlyCaptureContext"))
               .put(new Info("FlyCaptureCallback").valueTypes("FlyCaptureCallback")
                       .pointerTypes("@Cast(\"FlyCaptureCallback*\") @ByPtrPtr FlyCaptureCallback"))
               .put(new Info("FlyCaptureImage").base("AbstractFlyCaptureImage"))
               .put(new Info("flycaptureInitializeNotify", "flycaptureLockNextEvent", "flycaptureUnlockEvent").skip())
               .put(new Info("OVERLAPPED").cast().pointerTypes("Pointer"))
               .put(new Info("long", "unsigned long", "ULONG").cast().valueTypes("int")
                       .pointerTypes("IntPointer", "IntBuffer", "int[]"));
    }

    public static abstract class AbstractFlyCaptureImage extends Pointer {
        public AbstractFlyCaptureImage() { }
        public AbstractFlyCaptureImage(Pointer p) { super(p); }

        public abstract int iRows();
        public abstract int iCols();
        public abstract int iRowInc();
        public abstract BytePointer pData();

        public ByteBuffer getByteBuffer() {
            return pData().capacity(iRowInc()*iRows()).asByteBuffer();
        }
    }
}
