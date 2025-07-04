// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.ffmpeg.avutil;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.ffmpeg.global.avutil.*;


/**
 * \addtogroup lavu_audio
 * \{
 *
 * \defgroup lavu_audiofifo Audio FIFO Buffer
 * \{
 */

/**
 * Context for an Audio FIFO Buffer.
 *
 * - Operates at the sample level rather than the byte level.
 * - Supports multiple channels with either planar or packed sample format.
 * - Automatic reallocation when writing to a full buffer.
 */
@Opaque @Properties(inherit = org.bytedeco.ffmpeg.presets.avutil.class)
public class AVAudioFifo extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public AVAudioFifo() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public AVAudioFifo(Pointer p) { super(p); }
}
