package org.bytedeco.javacpp.presets;

import org.bytedeco.javacpp.annotation.*;
import org.bytedeco.javacpp.tools.*;

@Properties(target="org.bytedeco.javacpp.accelerate", value={@Platform(include={"cblas.h","clapack.h"},
        includepath = {"/System/Library/Frameworks/Accelerate.framework/versions/A/Frameworks/vecLib.framework/Versions/A/Headers/"},
        preload = {"BLAS","LAPACK"},
        preloadpath = {"/System/Library/Frameworks/Accelerate.framework/versions/A/Frameworks/vecLib.framework/Versions/A/"})})
public class blas implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("CBLAS_INDEX").cppTypes().annotations());
    }
}