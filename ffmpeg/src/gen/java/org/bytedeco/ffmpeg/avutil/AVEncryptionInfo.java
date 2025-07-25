// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.ffmpeg.avutil;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.ffmpeg.global.avutil.*;


/**
 * This describes encryption info for a packet.  This contains frame-specific
 * info for how to decrypt the packet before passing it to the decoder.
 *
 * The size of this struct is not part of the public ABI.
 */
@Properties(inherit = org.bytedeco.ffmpeg.presets.avutil.class)
public class AVEncryptionInfo extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public AVEncryptionInfo() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public AVEncryptionInfo(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public AVEncryptionInfo(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public AVEncryptionInfo position(long position) {
        return (AVEncryptionInfo)super.position(position);
    }
    @Override public AVEncryptionInfo getPointer(long i) {
        return new AVEncryptionInfo((Pointer)this).offsetAddress(i);
    }

    /** The fourcc encryption scheme, in big-endian byte order. */
    public native @Cast("uint32_t") int scheme(); public native AVEncryptionInfo scheme(int setter);

    /**
     * Only used for pattern encryption.  This is the number of 16-byte blocks
     * that are encrypted.
     */
    public native @Cast("uint32_t") int crypt_byte_block(); public native AVEncryptionInfo crypt_byte_block(int setter);

    /**
     * Only used for pattern encryption.  This is the number of 16-byte blocks
     * that are clear.
     */
    public native @Cast("uint32_t") int skip_byte_block(); public native AVEncryptionInfo skip_byte_block(int setter);

    /**
     * The ID of the key used to encrypt the packet.  This should always be
     * 16 bytes long, but may be changed in the future.
     */
    public native @Cast("uint8_t*") BytePointer key_id(); public native AVEncryptionInfo key_id(BytePointer setter);
    public native @Cast("uint32_t") int key_id_size(); public native AVEncryptionInfo key_id_size(int setter);

    /**
     * The initialization vector.  This may have been zero-filled to be the
     * correct block size.  This should always be 16 bytes long, but may be
     * changed in the future.
     */
    public native @Cast("uint8_t*") BytePointer iv(); public native AVEncryptionInfo iv(BytePointer setter);
    public native @Cast("uint32_t") int iv_size(); public native AVEncryptionInfo iv_size(int setter);

    /**
     * An array of subsample encryption info specifying how parts of the sample
     * are encrypted.  If there are no subsamples, then the whole sample is
     * encrypted.
     */
    public native AVSubsampleEncryptionInfo subsamples(); public native AVEncryptionInfo subsamples(AVSubsampleEncryptionInfo setter);
    public native @Cast("uint32_t") int subsample_count(); public native AVEncryptionInfo subsample_count(int setter);
}
