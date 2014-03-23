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
 * ****** IMPORTANT: Some functions are commented out to cover the
 * ****** common API from the FlyCapture SDK version 1.7 to 2.x.
 */

package com.googlecode.javacpp.presets;

import com.googlecode.javacpp.BytePointer;
import com.googlecode.javacpp.Parser;
import com.googlecode.javacpp.Pointer;
import com.googlecode.javacpp.annotation.Platform;
import com.googlecode.javacpp.annotation.Properties;
import java.nio.ByteBuffer;

/**
 *
 * @author Samuel Audet
 */
@Properties(target="com.googlecode.javacpp.PGRFlyCapture", value={
    @Platform(value="windows", link="PGRFlyCapture", preload="FlyCapture2",
        include={"<windows.h>", "<PGRFlyCapture.h>", "<PGRFlyCapturePlus.h>","<PGRFlyCaptureMessaging.h>"},
        includepath={"C:/Program Files/Point Grey Research/PGR FlyCapture/include/",
                     "C:/Program Files/Point Grey Research/FlyCapture2/include/FC1/"}),
    @Platform(value="windows-x86",
        linkpath   ={"C:/Program Files/Point Grey Research/PGR FlyCapture/lib/",
                     "C:/Program Files/Point Grey Research/FlyCapture2/lib/FC1/" },
        preloadpath={"C:/Program Files/Point Grey Research/PGR FlyCapture/bin/",
                     "C:/Program Files/Point Grey Research/FlyCapture2/bin/",
                     "C:/Program Files/Point Grey Research/FlyCapture2/bin/FC1/"}),
    @Platform(value="windows-x86_64",
        linkpath   ={"C:/Program Files/Point Grey Research/PGR FlyCapture/lib64/",
                     "C:/Program Files/Point Grey Research/FlyCapture2/lib64/FC1/" },
        preloadpath={"C:/Program Files/Point Grey Research/PGR FlyCapture/bin64/",
                     "C:/Program Files/Point Grey Research/FlyCapture2/bin64/",
                     "C:/Program Files/Point Grey Research/FlyCapture2/bin64/FC1/"}) })
public class PGRFlyCapture implements Parser.InfoMapper {
    public void map(Parser.InfoMap infoMap) {
          infoMap.put(new Parser.Info("PGRFLYCAPTURE_API", "PGRFLYCAPTURE_CALL_CONVEN").cppTypes().annotations())
                 .put(new Parser.Info("FlyCaptureContext").valueTypes("FlyCaptureContext").pointerTypes("@Cast(\"FlyCaptureContext*\") @ByPtrPtr FlyCaptureContext"))
                 .put(new Parser.Info("FlyCaptureImage").base("AbstractFlyCaptureImage"))
                 .put(new Parser.Info("FlyCaptureInfoEx::iInitialized", "FlyCaptureDriverInfo",
                                      "flycaptureGetDriverInfo", "flycaptureInitializeFromSerialNumberPlus",
                                      "flycaptureInitializeNotify", "flycaptureLockNextEvent", "flycaptureUnlockEvent").skip(true))
                 .put(new Parser.Info("OVERLAPPED").cast(true).pointerTypes("Pointer"))
                 .put(new Parser.Info("long", "unsigned long", "ULONG").cast(true).valueTypes("int").pointerTypes("IntPointer", "IntBuffer", "int[]"));
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
