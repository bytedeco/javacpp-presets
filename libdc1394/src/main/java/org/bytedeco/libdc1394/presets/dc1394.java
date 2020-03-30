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

package org.bytedeco.libdc1394.presets;

import java.nio.ByteBuffer;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Cast;
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
@Properties(inherit = javacpp.class, target = "org.bytedeco.libdc1394", global = "org.bytedeco.libdc1394.global.dc1394", value = {
    @Platform(not = "android", include = {"<poll.h>", "<dc1394/dc1394.h>", "<dc1394/types.h>", "<dc1394/log.h>",
        "<dc1394/camera.h>", "<dc1394/control.h>", "<dc1394/capture.h>", "<dc1394/conversions.h>", "<dc1394/format7.h>",
        "<dc1394/iso.h>", "<dc1394/register.h>", "<dc1394/video.h>", "<dc1394/utils.h>"}, link = "dc1394@.25"),
    @Platform(value = "macosx", preload = "usb-1.0@.0", preloadpath = "/usr/local/lib/"),
    @Platform(value = "windows", include = {"<dc1394/dc1394.h>", "<dc1394/types.h>", "<dc1394/log.h>",
        "<dc1394/camera.h>", "<dc1394/control.h>", "<dc1394/capture.h>", "<dc1394/conversions.h>", "<dc1394/format7.h>",
        "<dc1394/iso.h>", "<dc1394/register.h>", "<dc1394/video.h>", "<dc1394/utils.h>"},
        preload = {"libdc1394-25", "1394camera", "libusb-1.0"}) })
@NoException
public class dc1394 implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "libdc1394"); }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info("poll.h").skip())
               .put(new Info("restrict").cppTypes())
               .put(new Info("YUV2RGB", "RGB2YUV").cppTypes("void", "int", "int", "int", "int&", "int&", "int&"))
               .put(new Info("dc1394video_frame_t").base("dc1394video_frame_t_abstract"));
    }

    public static final short
            POLLIN         = 0x001,
            POLLPRI        = 0x002,
            POLLOUT        = 0x004,
            POLLMSG        = 0x400,
            POLLREMOVE     = 0x1000,
            POLLRDHUP      = 0x2000,
            POLLERR        = 0x008,
            POLLHUP        = 0x010,
            POLLNVAL       = 0x020;

    @Platform(not = "windows") public static class pollfd extends Pointer {
        static { Loader.load(); }
        public pollfd() { allocate(); }
        public pollfd(long size) { allocateArray(size); }
        public pollfd(Pointer p) { super(p); }
        private native void allocate();
        private native void allocateArray(long size);

        @Override public pollfd position(long position) {
            return (pollfd)super.position(position);
        }

        public native int   fd();      public native pollfd fd     (int   fd);
        public native short events();  public native pollfd events (short fd);
        public native short revents(); public native pollfd revents(short fd);
    }

    @Platform(not = "windows") public native static int poll(pollfd fds, @Cast("nfds_t") long nfds, int timeout);

    public static abstract class dc1394video_frame_t_abstract extends Pointer {
        public dc1394video_frame_t_abstract() { }
        public dc1394video_frame_t_abstract(Pointer p) { super(p); }

        public abstract BytePointer image();
        public abstract long total_bytes();
        public ByteBuffer getByteBuffer() { return image().capacity((int)total_bytes()).asByteBuffer(); }
    }
}
