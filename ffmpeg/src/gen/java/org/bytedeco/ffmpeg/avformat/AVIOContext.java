// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.ffmpeg.avformat;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import org.bytedeco.ffmpeg.avutil.*;
import static org.bytedeco.ffmpeg.global.avutil.*;
import org.bytedeco.ffmpeg.swresample.*;
import static org.bytedeco.ffmpeg.global.swresample.*;
import org.bytedeco.ffmpeg.avcodec.*;
import static org.bytedeco.ffmpeg.global.avcodec.*;

import static org.bytedeco.ffmpeg.global.avformat.*;


/**
 * Bytestream IO Context.
 * New public fields can be added with minor version bumps.
 * Removal, reordering and changes to existing public fields require
 * a major version bump.
 * sizeof(AVIOContext) must not be used outside libav*.
 *
 * \note None of the function pointers in AVIOContext should be called
 *       directly, they should only be set by the client application
 *       when implementing custom I/O. Normally these are set to the
 *       function pointers specified in avio_alloc_context()
 */
@Properties(inherit = org.bytedeco.ffmpeg.presets.avformat.class)
public class AVIOContext extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public AVIOContext() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public AVIOContext(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public AVIOContext(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public AVIOContext position(long position) {
        return (AVIOContext)super.position(position);
    }
    @Override public AVIOContext getPointer(long i) {
        return new AVIOContext((Pointer)this).offsetAddress(i);
    }

    /**
     * A class for private options.
     *
     * If this AVIOContext is created by avio_open2(), av_class is set and
     * passes the options down to protocols.
     *
     * If this AVIOContext is manually allocated, then av_class may be set by
     * the caller.
     *
     * warning -- this field can be NULL, be sure to not pass this AVIOContext
     * to any av_opt_* functions in that case.
     */
    public native @Const AVClass av_class(); public native AVIOContext av_class(AVClass setter);

    /*
     * The following shows the relationship between buffer, buf_ptr,
     * buf_ptr_max, buf_end, buf_size, and pos, when reading and when writing
     * (since AVIOContext is used for both):
     *
     **********************************************************************************
     *                                   READING
     **********************************************************************************
     *
     *                            |              buffer_size              |
     *                            |---------------------------------------|
     *                            |                                       |
     *
     *                         buffer          buf_ptr       buf_end
     *                            +---------------+-----------------------+
     *                            |/ / / / / / / /|/ / / / / / /|         |
     *  read buffer:              |/ / consumed / | to be read /|         |
     *                            |/ / / / / / / /|/ / / / / / /|         |
     *                            +---------------+-----------------------+
     *
     *                                                         pos
     *              +-------------------------------------------+-----------------+
     *  input file: |                                           |                 |
     *              +-------------------------------------------+-----------------+
     *
     *
     **********************************************************************************
     *                                   WRITING
     **********************************************************************************
     *
     *                             |          buffer_size                 |
     *                             |--------------------------------------|
     *                             |                                      |
     *
     *                                                buf_ptr_max
     *                          buffer                 (buf_ptr)       buf_end
     *                             +-----------------------+--------------+
     *                             |/ / / / / / / / / / / /|              |
     *  write buffer:              | / / to be flushed / / |              |
     *                             |/ / / / / / / / / / / /|              |
     *                             +-----------------------+--------------+
     *                               buf_ptr can be in this
     *                               due to a backward seek
     *
     *                            pos
     *               +-------------+----------------------------------------------+
     *  output file: |             |                                              |
     *               +-------------+----------------------------------------------+
     *
     */
    /** Start of the buffer. */
    public native @Cast("unsigned char*") BytePointer buffer(); public native AVIOContext buffer(BytePointer setter);
    /** Maximum buffer size */
    public native int buffer_size(); public native AVIOContext buffer_size(int setter);
    /** Current position in the buffer */
    public native @Cast("unsigned char*") BytePointer buf_ptr(); public native AVIOContext buf_ptr(BytePointer setter);
    /** End of the data, may be less than
                                     buffer+buffer_size if the read function returned
                                     less data than requested, e.g. for streams where
                                     no more data has been received yet. */
    public native @Cast("unsigned char*") BytePointer buf_end(); public native AVIOContext buf_end(BytePointer setter);
    /** A private pointer, passed to the read/write/seek/...
                                     functions. */
    public native Pointer opaque(); public native AVIOContext opaque(Pointer setter);
    public static class Read_packet_Pointer_BytePointer_int extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    Read_packet_Pointer_BytePointer_int(Pointer p) { super(p); }
        protected Read_packet_Pointer_BytePointer_int() { allocate(); }
        private native void allocate();
        public native int call(Pointer opaque, @Cast("uint8_t*") BytePointer buf, int buf_size);
    }
    public native Read_packet_Pointer_BytePointer_int read_packet(); public native AVIOContext read_packet(Read_packet_Pointer_BytePointer_int setter);
    public static class Write_packet_Pointer_BytePointer_int extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    Write_packet_Pointer_BytePointer_int(Pointer p) { super(p); }
        protected Write_packet_Pointer_BytePointer_int() { allocate(); }
        private native void allocate();
        public native int call(Pointer opaque, @Cast("const uint8_t*") BytePointer buf, int buf_size);
    }
    public native Write_packet_Pointer_BytePointer_int write_packet(); public native AVIOContext write_packet(Write_packet_Pointer_BytePointer_int setter);
    public static class Seek_Pointer_long_int extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    Seek_Pointer_long_int(Pointer p) { super(p); }
        protected Seek_Pointer_long_int() { allocate(); }
        private native void allocate();
        public native @Cast("int64_t") long call(Pointer opaque, @Cast("int64_t") long offset, int whence);
    }
    public native Seek_Pointer_long_int seek(); public native AVIOContext seek(Seek_Pointer_long_int setter);
    /** position in the file of the current buffer */
    public native @Cast("int64_t") long pos(); public native AVIOContext pos(long setter);
    /** true if was unable to read due to error or eof */
    public native int eof_reached(); public native AVIOContext eof_reached(int setter);
    /** contains the error code or 0 if no error happened */
    public native int error(); public native AVIOContext error(int setter);
    /** true if open for writing */
    public native int write_flag(); public native AVIOContext write_flag(int setter);
    public native int max_packet_size(); public native AVIOContext max_packet_size(int setter);
    /** Try to buffer at least this amount of data
                                     before flushing it. */
    public native int min_packet_size(); public native AVIOContext min_packet_size(int setter);
    public native @Cast("unsigned long") long checksum(); public native AVIOContext checksum(long setter);
    public native @Cast("unsigned char*") BytePointer checksum_ptr(); public native AVIOContext checksum_ptr(BytePointer setter);
    public static class Update_checksum_long_BytePointer_int extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    Update_checksum_long_BytePointer_int(Pointer p) { super(p); }
        protected Update_checksum_long_BytePointer_int() { allocate(); }
        private native void allocate();
        public native @Cast("unsigned long") long call(@Cast("unsigned long") long checksum, @Cast("const uint8_t*") BytePointer buf, @Cast("unsigned int") int size);
    }
    public native Update_checksum_long_BytePointer_int update_checksum(); public native AVIOContext update_checksum(Update_checksum_long_BytePointer_int setter);
    /**
     * Pause or resume playback for network streaming protocols - e.g. MMS.
     */
    public static class Read_pause_Pointer_int extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    Read_pause_Pointer_int(Pointer p) { super(p); }
        protected Read_pause_Pointer_int() { allocate(); }
        private native void allocate();
        public native int call(Pointer opaque, int pause);
    }
    public native Read_pause_Pointer_int read_pause(); public native AVIOContext read_pause(Read_pause_Pointer_int setter);
    /**
     * Seek to a given timestamp in stream with the specified stream_index.
     * Needed for some network streaming protocols which don't support seeking
     * to byte position.
     */
    public static class Read_seek_Pointer_int_long_int extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    Read_seek_Pointer_int_long_int(Pointer p) { super(p); }
        protected Read_seek_Pointer_int_long_int() { allocate(); }
        private native void allocate();
        public native @Cast("int64_t") long call(Pointer opaque, int stream_index,
                             @Cast("int64_t") long timestamp, int flags);
    }
    public native Read_seek_Pointer_int_long_int read_seek(); public native AVIOContext read_seek(Read_seek_Pointer_int_long_int setter);
    /**
     * A combination of AVIO_SEEKABLE_ flags or 0 when the stream is not seekable.
     */
    public native int seekable(); public native AVIOContext seekable(int setter);

    /**
     * avio_read and avio_write should if possible be satisfied directly
     * instead of going through a buffer, and avio_seek will always
     * call the underlying seek function directly.
     */
    public native int direct(); public native AVIOContext direct(int setter);

    /**
     * ',' separated list of allowed protocols.
     */
    public native @Cast("const char*") BytePointer protocol_whitelist(); public native AVIOContext protocol_whitelist(BytePointer setter);

    /**
     * ',' separated list of disallowed protocols.
     */
    public native @Cast("const char*") BytePointer protocol_blacklist(); public native AVIOContext protocol_blacklist(BytePointer setter);

    /**
     * A callback that is used instead of write_packet.
     */
    public static class Write_data_type_Pointer_BytePointer_int_int_long extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    Write_data_type_Pointer_BytePointer_int_int_long(Pointer p) { super(p); }
        protected Write_data_type_Pointer_BytePointer_int_int_long() { allocate(); }
        private native void allocate();
        public native int call(Pointer opaque, @Cast("const uint8_t*") BytePointer buf, int buf_size,
                               @Cast("AVIODataMarkerType") int type, @Cast("int64_t") long time);
    }
    public native Write_data_type_Pointer_BytePointer_int_int_long write_data_type(); public native AVIOContext write_data_type(Write_data_type_Pointer_BytePointer_int_int_long setter);
    /**
     * If set, don't call write_data_type separately for AVIO_DATA_MARKER_BOUNDARY_POINT,
     * but ignore them and treat them as AVIO_DATA_MARKER_UNKNOWN (to avoid needlessly
     * small chunks of data returned from the callback).
     */
    public native int ignore_boundary_point(); public native AVIOContext ignore_boundary_point(int setter);

    /**
     * Maximum reached position before a backward seek in the write buffer,
     * used keeping track of already written data for a later flush.
     */
    public native @Cast("unsigned char*") BytePointer buf_ptr_max(); public native AVIOContext buf_ptr_max(BytePointer setter);

    /**
     * Read-only statistic of bytes read for this AVIOContext.
     */
    public native @Cast("int64_t") long bytes_read(); public native AVIOContext bytes_read(long setter);

    /**
     * Read-only statistic of bytes written for this AVIOContext.
     */
    public native @Cast("int64_t") long bytes_written(); public native AVIOContext bytes_written(long setter);
}
