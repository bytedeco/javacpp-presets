package org.bytedeco.kvazaar.presets;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;
import org.bytedeco.javacpp.tools.*;

@Properties(
    value = {
        @Platform(include = "<kvazaar.h>", link = "kvazaar#"),
        @Platform(value = "windows", preload = {"libkvazaar"})
    },
    target = "org.bytedeco.kvazaar",
    global = "org.bytedeco.kvazaar.global.kvazaar"
)
public class kvazaar implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "kvazaar"); }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info("FAR").cppText("#define FAR"));
    }
}