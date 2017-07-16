package org.bytedeco.javacpp.presets;

import org.bytedeco.javacpp.annotation.*;
import org.bytedeco.javacpp.tools.*;

@Properties(target="org.bytedeco.javacpp.accelerateblas", value={@Platform(include={"cblas.h","clapack.h"},
        includepath = {"/System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework/Headers/"},
        preload = {"BLAS","LAPACK"},
        preloadpath = {"/System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework/vecLib/"})})
public class accelerateblas implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("CBLAS_INDEX").cppTypes().annotations());
    }
}