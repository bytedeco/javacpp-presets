package org.bytedeco.helloworld.presets;

import org.bytedeco.javacpp.annotation.*;
import org.bytedeco.javacpp.tools.*;

@Properties(
        target = "org.bytedeco.helloworld",
        global = "org.bytedeco.helloworld.global.helloworld",
        value = {
            @Platform(
                    value = {"linux-x86", "linux-x86_64", "windows-x86", "windows-x86_64"},
                    include = "helloworld.h",
                    resource = {"include", "lib"}
            ),
            @Platform(value = {"linux-x86", "linux-x86_64"}, link = "libhelloworld"),
            @Platform(value = {"windows-x86", "windows-x86_64"}, link = "libhelloworld")
        }
)
public class helloworld implements InfoMapper {

    @Override
    public void map(InfoMap infoMap) {
    }
}
