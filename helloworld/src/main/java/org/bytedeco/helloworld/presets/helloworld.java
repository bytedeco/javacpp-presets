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
                "macosx-x86_64",
                "windows-x86",
                "windows-x86_64"
            },
            include = "helloworld.h",
            link = "helloworld",
            preload = "libhelloworld-0"
        )
    }
)
public class helloworld implements InfoMapper {

    @Override
    public void map(InfoMap infoMap) {
        //    struct Person {
        //        char firstname[100];
        //        char lastname[100];
        //    };
        infoMap.put(new Info("Person").pointerTypes("PersonTypePtr"));
        //
        //    typedef struct Person PersonType;
        infoMap.put(new Info("PersonType").pointerTypes("PersonTypePtr"));
        //
        //
        //    typedef struct Person * PersonTypePtr;
        infoMap.put(new Info("PersonTypePtr").valueTypes("PersonTypePtr"));
    }
}
