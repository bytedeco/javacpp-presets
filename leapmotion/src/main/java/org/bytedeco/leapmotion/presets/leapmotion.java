package org.bytedeco.leapmotion.presets;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;
import org.bytedeco.javacpp.tools.*;

@Properties(
    value = {
        @Platform(
                value = "windows",
                include = "LeapC.h",
                link = "LeapC")
    },
    target = "org.bytedeco.leapmotion",
    global = "org.bytedeco.leapmotion.global.leapmotion"
)
public class leapmotion implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "leapmotion"); }

    public void map(InfoMap infoMap) {
//  https://github.com/bytedeco/javacpp/wiki/Mapping-Recipes#specifying-names-to-use-in-java

        infoMap.put(new Info("LEAP_EXPORT","LEAP_CALL","LEAP_STATIC_ASSERT","static_assert").cppTypes().annotations())
                
                .put(new Info("_LEAP_CONNECTION").pointerTypes("LEAP_CONNECTION"))
                .put(new Info("LEAP_CONNECTION").valueTypes("LEAP_CONNECTION").pointerTypes("@Cast(\"LEAP_CONNECTION*\") PointerPointer", "@ByPtrPtr LEAP_CONNECTION"))
                
                .put(new Info("_LEAP_DEVICE").pointerTypes("LEAP_DEVICE"))
                .put(new Info("LEAP_DEVICE").valueTypes("LEAP_DEVICE").pointerTypes("@Cast(\"LEAP_DEVICE*\") PointerPointer", "@ByPtrPtr LEAP_DEVICE"))

                .put(new Info("_LEAP_CLOCK_REBASER").pointerTypes("LEAP_CLOCK_REBASER"))
                .put(new Info("LEAP_CLOCK_REBASER").valueTypes("LEAP_CLOCK_REBASER").pointerTypes("@Cast(\"LEAP_CLOCK_REBASER*\") PointerPointer", "@ByPtrPtr LEAP_CLOCK_REBASER"))

                .put(new Info("_LEAP_RECORDING").pointerTypes("LEAP_RECORDING"))
                .put(new Info("LEAP_RECORDING").valueTypes("LEAP_RECORDING").pointerTypes("@Cast(\"LEAP_RECORDING*\") PointerPointer", "@ByPtrPtr LEAP_RECORDING"))

                .put(new Info("_LEAP_CALIBRATION").pointerTypes("LEAP_CALIBRATION"))
                .put(new Info("LEAP_CALIBRATION").valueTypes("LEAP_CALIBRATION").pointerTypes("@Cast(\"LEAP_CALIBRATION*\") PointerPointer", "@ByPtrPtr LEAP_CALIBRATION"));
            
            //    .put(new Info("FAR").cppText("#define FAR"))
            //    .put(new Info("OF").cppText("#define OF(args) args"))
            //    .put(new Info("Z_ARG").cppText("#define Z_ARG(args) args"))
            //    .put(new Info("Byte", "Bytef", "charf").cast().valueTypes("byte").pointerTypes("BytePointer"))
            //    .put(new Info("uInt", "uIntf").cast().valueTypes("int").pointerTypes("IntPointer"))
            //    .put(new Info("uLong", "uLongf", "z_crc_t", "z_off_t", "z_size_t").cast().valueTypes("long").pointerTypes("CLongPointer"))
            //    .put(new Info("z_off64_t").cast().valueTypes("long").pointerTypes("LongPointer"))
            //    .put(new Info("voidp", "voidpc", "voidpf").valueTypes("Pointer"))
            //    .put(new Info("gzFile_s").pointerTypes("gzFile"))
            //    .put(new Info("gzFile").valueTypes("gzFile"))
            //    .put(new Info("Z_LARGE64", "!defined(ZLIB_INTERNAL) && defined(Z_WANT64)").define(false))
            //    .put(new Info("inflateGetDictionary", "gzopen_w", "gzvprintf").skip());
    }
}
