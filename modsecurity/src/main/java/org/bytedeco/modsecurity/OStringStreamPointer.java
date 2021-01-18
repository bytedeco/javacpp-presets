package org.bytedeco.modsecurity;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.ByRef;
import org.bytedeco.javacpp.annotation.Name;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.annotation.StdString;

@Name("std::ostringstream") @Properties(inherit = org.bytedeco.modsecurity.presets.modsecurity.class)
public class OStringStreamPointer extends Pointer {
    static { Loader.load(); }

    public OStringStreamPointer() {
        allocate();
    }

    public OStringStreamPointer(Pointer p) {
        super(p);
    }

    private native void allocate();

    public native @Name("operator =") @ByRef OStringStreamPointer put(@ByRef OStringStreamPointer x);

    public native void str(@StdString BytePointer value);
    public native @StdString BytePointer str();
}
