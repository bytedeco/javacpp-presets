package org.bytedeco.shakapackager.functions;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;
import org.bytedeco.shakapackager.EncryptionParams;

@Properties(inherit = org.bytedeco.shakapackager.presets.packager.class)
public class StringEncryptedStreamAttributes_EncryptionParams_EncryptedStreamAttributes extends FunctionPointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public    StringEncryptedStreamAttributes_EncryptionParams_EncryptedStreamAttributes(Pointer p) { super(p); }
    protected StringEncryptedStreamAttributes_EncryptionParams_EncryptedStreamAttributes() { allocate(); }
    private native void allocate();
    public native @StdString BytePointer call(@Const @ByRef EncryptionParams.EncryptedStreamAttributes arg0);
}


