/*
 * Copyright (C) 2013-2020 Samuel Audet
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

package org.bytedeco.libfreenect.presets;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.NoException;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.presets.javacpp;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Samuel Audet
 */
@Properties(inherit = javacpp.class, target = "org.bytedeco.libfreenect", global = "org.bytedeco.libfreenect.global.freenect", value = {
    @Platform(not = "android", include = {"<libfreenect/libfreenect.h>", "<libfreenect/libfreenect_registration.h>",
                                        /*"<libfreenect/libfreenect_audio.h>",*/ "<libfreenect/libfreenect_sync.h>"},
        link = {"freenect@0.5", "freenect_sync@0.5"}),
    @Platform(value = "macosx-x86_64", preload = "usb-1.0@.0", preloadpath = "/usr/local/lib/"),
    @Platform(value = "windows", link = {"freenect", "freenect_sync", "pthreadVC2"}),
    @Platform(value = "windows-x86",    preload = "libusb0_x86"),
    @Platform(value = "windows-x86_64", preload = "libusb0") })
@NoException
public class freenect implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "libfreenect"); }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info("FREENECTAPI", "FREENECTAPI_SYNC").cppTypes().annotations());
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
