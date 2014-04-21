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

import java.nio.ByteBuffer;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Cast;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Samuel Audet
 */
@Properties(target="org.bytedeco.javacpp.dc1394", value={
    @Platform(value={"linux", "macosx"}, include={"<poll.h>", "<dc1394/dc1394.h>", "<dc1394/types.h>", "<dc1394/log.h>",
        "<dc1394/camera.h>", "<dc1394/control.h>", "<dc1394/capture.h>", "<dc1394/conversions.h>", "<dc1394/format7.h>",
        "<dc1394/iso.h>", "<dc1394/register.h>", "<dc1394/video.h>", "<dc1394/utils.h>"}, link="dc1394@.22", preload="libusb-1.0") })
public class dc1394 implements InfoMapper {
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

    public static class pollfd extends Pointer {
        static { Loader.load(); }
        public pollfd() { allocate(); }
        public pollfd(int size) { allocateArray(size); }
        public pollfd(Pointer p) { super(p); }
        private native void allocate();
        private native void allocateArray(int size);

        @Override public pollfd position(int position) {
            return (pollfd)super.position(position);
        }

        public native int   fd();      public native pollfd fd     (int   fd);
        public native short events();  public native pollfd events (short fd);
        public native short revents(); public native pollfd revents(short fd);
    }

    public native static int poll(pollfd fds, @Cast("nfds_t") long nfds, int timeout);

    public static abstract class dc1394video_frame_t_abstract extends Pointer {
        public dc1394video_frame_t_abstract() { }
        public dc1394video_frame_t_abstract(Pointer p) { super(p); }

        public abstract BytePointer image();
        public abstract long total_bytes();
        public ByteBuffer getByteBuffer() { return image().capacity((int)total_bytes()).asByteBuffer(); }
    }
}
