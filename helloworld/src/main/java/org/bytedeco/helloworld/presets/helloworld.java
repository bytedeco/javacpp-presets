package org.bytedeco.helloworld.presets;

import org.bytedeco.javacpp.annotation.*;
import org.bytedeco.javacpp.tools.*;

@Properties(
        target = "org.bytedeco.helloworld",
        global = "org.bytedeco.helloworld.global.helloworld",
        value = {
            @Platform(
                    value = {
                        "linux-x86",
                        "linux-x86_64",
                        "windows-x86",
                        "windows-x86_64"
                    },
                    include = "helloworld.h",
                    link = "helloworld"
            )
        }
)
public class helloworld implements InfoMapper {

    @Override
    public void map(InfoMap infoMap) {
    }
}
