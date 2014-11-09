/*
 * Copyright (C) 2014 Samuel Audet
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
