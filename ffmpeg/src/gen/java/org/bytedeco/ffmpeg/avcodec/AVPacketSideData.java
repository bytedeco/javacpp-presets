// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.ffmpeg.avcodec;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import org.bytedeco.ffmpeg.avutil.*;
import static org.bytedeco.ffmpeg.global.avutil.*;
import org.bytedeco.ffmpeg.swresample.*;
import static org.bytedeco.ffmpeg.global.swresample.*;

import static org.bytedeco.ffmpeg.global.avcodec.*;
 //DEPRECATED
// #endif

/**
 * This structure stores auxiliary information for decoding, presenting, or
 * otherwise processing the coded stream. It is typically exported by demuxers
 * and encoders and can be fed to decoders and muxers either in a per packet
 * basis, or as global side data (applying to the entire coded stream).
 *
 * Global side data is handled as follows:
 * - During demuxing, it may be exported through
 *   \ref AVStream.codecpar.side_data "AVStream's codec parameters", which can
 *   then be passed as input to decoders through the
 *   \ref AVCodecContext.coded_side_data "decoder context's side data", for
 *   initialization.
 * - For muxing, it can be fed through \ref AVStream.codecpar.side_data
 *   "AVStream's codec parameters", typically  the output of encoders through
 *   the \ref AVCodecContext.coded_side_data "encoder context's side data", for
 *   initialization.
 *
 * Packet specific side data is handled as follows:
 * - During demuxing, it may be exported through \ref AVPacket.side_data
 *   "AVPacket's side data", which can then be passed as input to decoders.
 * - For muxing, it can be fed through \ref AVPacket.side_data "AVPacket's
 *   side data", typically the output of encoders.
 *
 * Different modules may accept or export different types of side data
 * depending on media type and codec. Refer to \ref AVPacketSideDataType for a
 * list of defined types and where they may be found or used.
 */
@Properties(inherit = org.bytedeco.ffmpeg.presets.avcodec.class)
public class AVPacketSideData extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public AVPacketSideData() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public AVPacketSideData(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public AVPacketSideData(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public AVPacketSideData position(long position) {
        return (AVPacketSideData)super.position(position);
    }
    @Override public AVPacketSideData getPointer(long i) {
        return new AVPacketSideData((Pointer)this).offsetAddress(i);
    }

    public native @Cast("uint8_t*") BytePointer data(); public native AVPacketSideData data(BytePointer setter);
    public native @Cast("size_t") long size(); public native AVPacketSideData size(long setter);
    public native @Cast("AVPacketSideDataType") int type(); public native AVPacketSideData type(int setter);
}
