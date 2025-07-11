// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.ffmpeg.avutil;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.ffmpeg.global.avutil.*;


/**
 * An AVChannelLayout holds information about the channel layout of audio data.
 *
 * A channel layout here is defined as a set of channels ordered in a specific
 * way (unless the channel order is AV_CHANNEL_ORDER_UNSPEC, in which case an
 * AVChannelLayout carries only the channel count).
 * All orders may be treated as if they were AV_CHANNEL_ORDER_UNSPEC by
 * ignoring everything but the channel count, as long as av_channel_layout_check()
 * considers they are valid.
 *
 * Unlike most structures in FFmpeg, sizeof(AVChannelLayout) is a part of the
 * public ABI and may be used by the caller. E.g. it may be allocated on stack
 * or embedded in caller-defined structs.
 *
 * AVChannelLayout can be initialized as follows:
 * - default initialization with {0}, followed by setting all used fields
 *   correctly;
 * - by assigning one of the predefined AV_CHANNEL_LAYOUT_* initializers;
 * - with a constructor function, such as av_channel_layout_default(),
 *   av_channel_layout_from_mask() or av_channel_layout_from_string().
 *
 * The channel layout must be unitialized with av_channel_layout_uninit()
 *
 * Copying an AVChannelLayout via assigning is forbidden,
 * av_channel_layout_copy() must be used instead (and its return value should
 * be checked)
 *
 * No new fields may be added to it without a major version bump, except for
 * new elements of the union fitting in sizeof(uint64_t).
 */
@Properties(inherit = org.bytedeco.ffmpeg.presets.avutil.class)
public class AVChannelLayout extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public AVChannelLayout() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public AVChannelLayout(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public AVChannelLayout(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public AVChannelLayout position(long position) {
        return (AVChannelLayout)super.position(position);
    }
    @Override public AVChannelLayout getPointer(long i) {
        return new AVChannelLayout((Pointer)this).offsetAddress(i);
    }

    /**
     * Channel order used in this layout.
     * This is a mandatory field.
     */
    public native @Cast("AVChannelOrder") int order(); public native AVChannelLayout order(int setter);

    /**
     * Number of channels in this layout. Mandatory field.
     */
    public native int nb_channels(); public native AVChannelLayout nb_channels(int setter);

    /**
     * Details about which channels are present in this layout.
     * For AV_CHANNEL_ORDER_UNSPEC, this field is undefined and must not be
     * used.
     */
        /**
         * This member must be used for AV_CHANNEL_ORDER_NATIVE, and may be used
         * for AV_CHANNEL_ORDER_AMBISONIC to signal non-diegetic channels.
         * It is a bitmask, where the position of each set bit means that the
         * AVChannel with the corresponding value is present.
         *
         * I.e. when (mask & (1 << AV_CHAN_FOO)) is non-zero, then AV_CHAN_FOO
         * is present in the layout. Otherwise it is not present.
         *
         * \note when a channel layout using a bitmask is constructed or
         * modified manually (i.e.  not using any of the av_channel_layout_*
         * functions), the code doing it must ensure that the number of set bits
         * is equal to nb_channels.
         */
        @Name("u.mask") public native @Cast("uint64_t") long u_mask(); public native AVChannelLayout u_mask(long setter);
        /**
         * This member must be used when the channel order is
         * AV_CHANNEL_ORDER_CUSTOM. It is a nb_channels-sized array, with each
         * element signalling the presence of the AVChannel with the
         * corresponding value in map[i].id.
         *
         * I.e. when map[i].id is equal to AV_CHAN_FOO, then AV_CH_FOO is the
         * i-th channel in the audio data.
         *
         * When map[i].id is in the range between AV_CHAN_AMBISONIC_BASE and
         * AV_CHAN_AMBISONIC_END (inclusive), the channel contains an ambisonic
         * component with ACN index (as defined above)
         * n = map[i].id - AV_CHAN_AMBISONIC_BASE.
         *
         * map[i].name may be filled with a 0-terminated string, in which case
         * it will be used for the purpose of identifying the channel with the
         * convenience functions below. Otherise it must be zeroed.
         */
        @Name("u.map") public native AVChannelCustom u_map(); public native AVChannelLayout u_map(AVChannelCustom setter);

    /**
     * For some private data of the user.
     */
    public native Pointer opaque(); public native AVChannelLayout opaque(Pointer setter);
}
