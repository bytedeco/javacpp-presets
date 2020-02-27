package org.bytedeco.hyperscan.presets;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;


@Properties(
    value = {
        @Platform(
                value = {"linux-x86_64", "macosx-x86_64", "windows-x86_64"},
                compiler = "cpp11",
                include = {"hs/hs_common.h", "hs/hs_compile.h","hs/hs_runtime.h", "hs/hs.h"},
                link = { "hs@.5", "hs_runtime@.5" })
    },
    target = "org.bytedeco.hyperscan",
    global = "org.bytedeco.hyperscan.global.hyperscan"
)
public class hyperscan implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "hyperscan"); }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info("HS_CDECL").cppTypes().annotations());
    }
}
