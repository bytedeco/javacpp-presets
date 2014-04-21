/*
 * Copyright (C) 2013,2014 Samuel Audet
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
 */

package org.bytedeco.javacpp.presets;

import org.bytedeco.javacpp.Loader;
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
@Properties(target="org.bytedeco.javacpp.freenect", value={
    @Platform(not="android", include={"<libfreenect.h>", "<libfreenect_registration.h>", "<libfreenect_audio.h>", "<libfreenect_sync.h>"},
        includepath={"/usr/local/include/libfreenect/", "/opt/local/include/libfreenect/", "/usr/include/libfreenect/"},
        link={"freenect@0.4", "freenect_sync@0.4"}, preload="libusb-1.0"),
    @Platform(value="windows", include={"<libfreenect.h>", "<libfreenect_registration.h>", "<libfreenect_sync.h>"},
        link={"freenect", "freenect_sync", "pthreadVC2"}),
    @Platform(value="windows-x86", includepath="C:/Program Files (x86)/libfreenect/include/libfreenect/",
        linkpath={"C:/pthreads-w32-2-9-1-release/Pre-built.2/lib/x86", "C:/Program Files (x86)/libfreenect/lib/"},
        preload="libusb0_x86", preloadpath="C:/libusb-win32-bin-1.2.6.0/bin/x86/"),
    @Platform(value="windows-x86_64", includepath="C:/Program Files/libfreenect/include/libfreenect/",
        linkpath={"C:/pthreads-w32-2-9-1-release/Pre-built.2/lib/x64", "C:/Program Files/libfreenect/lib/"},
        preload="libusb0", preloadpath="C:/libusb-win32-bin-1.2.6.0/bin/amd64/") })
public class freenect implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("FREENECTAPI").cppTypes().annotations());
    }

    public static class timeval extends Pointer {
        static { Loader.load(); }
        public timeval() { allocate(); }
        public timeval(int size) { allocateArray(size); }
        public timeval(Pointer p) { super(p); }
        private native void allocate();
        private native void allocateArray(int size);

        public native long tv_sec();  public native timeval tv_sec (long tv_sec);
        public native long tv_usec(); public native timeval tv_usec(long tv_usec);
    }
}
