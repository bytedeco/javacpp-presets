package org.bytedeco.libecl.presets;

import org.bytedeco.javacpp.ClassProperties;
import org.bytedeco.javacpp.LoadEnabled;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.*;
import org.bytedeco.javacpp.tools.*;

@Properties(
        target = "org.bytedeco.libecl",
        global = "org.bytedeco.libecl.global.libecl",
        value = {
            @Platform(
                    value = {
                        "linux-x86",
                        "linux-x86_64",
                        "macosx-x86_64",
                        "windows-x86",
                        "windows-x86_64"
                    },
                    link = "ecl"
            )
        }
)
public class libecl implements InfoMapper, LoadEnabled {

    static {
        Loader.checkVersion("org.bytedeco", "libecl");
    }

    @Override
    public void init(ClassProperties properties) {
    }

    @Override
    public void map(InfoMap infoMap) {
    }
}
