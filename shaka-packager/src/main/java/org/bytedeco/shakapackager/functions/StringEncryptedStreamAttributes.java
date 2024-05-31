package org.bytedeco.shakapackager.functions;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;
import org.bytedeco.shakapackager.EncryptionParams;

@NoOffset @Name("std::function<std::string(const shaka::EncryptionParams::EncryptedStreamAttributes&)>") @Properties(inherit = org.bytedeco.shakapackager.presets.packager.class)
public class StringEncryptedStreamAttributes extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public StringEncryptedStreamAttributes(Pointer p) { super(p); }
    public StringEncryptedStreamAttributes(StringEncryptedStreamAttributes_EncryptionParams_EncryptedStreamAttributes value) { this(); put(value); }
    public StringEncryptedStreamAttributes()       { allocate();  }
    private native void allocate();
    public native @Name("operator =") @ByRef StringEncryptedStreamAttributes put(@ByRef StringEncryptedStreamAttributes x);

    public native @Name("operator =") @ByRef StringEncryptedStreamAttributes put(@ByRef StringEncryptedStreamAttributes_EncryptionParams_EncryptedStreamAttributes value);
    public native @Name("operator ()") @Const  @StdString BytePointer call(@Const @ByRef EncryptionParams.EncryptedStreamAttributes arg0);
}
