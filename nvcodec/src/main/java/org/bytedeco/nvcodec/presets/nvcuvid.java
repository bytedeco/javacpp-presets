package org.bytedeco.nvcodec.presets;

import org.bytedeco.javacpp.ClassProperties;
import org.bytedeco.javacpp.LoadEnabled;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

import java.util.List;

import org.bytedeco.cuda.presets.cudart;

/**
 *
 * @author StaticDefault
 */
@Properties(
    inherit = cudart.class,
    value = {
        @Platform(
            value = {"linux-x86_64", "windows-x86_64"},
            compiler = "cpp11",
            include = {"cuviddec.h", "nvcuvid.h"},
            link = {"nvcuvid"}
        ),
        @Platform(
            value = "linux-x86_64",
            includepath = {"/usr/include/x86_64-linux-gnu/", "/usr/local/videocodecsdk/Interface/"},
            linkpath = {"/usr/lib/x86_64-linux-gnu/", "/usr/local/videocodecsdk/Lib/linux/stubs/x86_64/"}
        ),
        @Platform(
            value = "windows-x86_64",
            includepath = "C:/Program Files/NVIDIA GPU Computing Toolkit/VideoCodecSDK/Interface/",
            linkpath = "C:/Program Files/NVIDIA GPU Computing Toolkit/VideoCodecSDK/Lib/x64/"
        )
    },
    target = "org.bytedeco.nvcodec.nvcuvid",
    global = "org.bytedeco.nvcodec.global.nvcuvid"
)
public class nvcuvid implements LoadEnabled, InfoMapper {
    @Override
    public void init(ClassProperties properties) {
        String platform = properties.getProperty("platform");
        List<String> preloads = properties.get("platform.preload");
        List<String> resources = properties.get("platform.preloadresource");

        if (!Loader.isLoadLibraries()) {
            return;
        }
        int i = 0;
        String[] libs = {"cudart"};

        for (String lib : libs) {
            if (platform.startsWith("linux")) {
                lib += lib.equals("cudart") ? "@.11.0" : lib.equals("nvrtc") ? "@.11.1" : "@.11";
            } else if (platform.startsWith("windows")) {
                lib += lib.equals("cudart") ? "64_110" : lib.equals("nvrtc") ? "64_111_0" : "64_11";
            } else {
                continue;
            }
            if (!preloads.contains(lib)) {
                preloads.add(i++, lib);
            }
        }
        if (i > 0) {
            resources.add("/org/bytedeco/cuda/");
        }
    }

    @Override
    public void map(InfoMap infoMap) {
        infoMap.put(new Info().enumerate())
               .put(new Info("cuviddec.h").linePatterns("#define cuvidMapVideoFrame.*", "#define cuvidUnmapVideoFrame.*").skip())

               .put(new Info("NV_ENC_DEPRECATED").cppText("#define NV_ENC_DEPRECATED deprecated").cppTypes())
               .put(new Info("deprecated").annotations("@Deprecated"))

               .put(new Info("std::size_t").cast().valueTypes("long").pointerTypes("LongPointer", "LongBuffer", "long[]"))
               .put(new Info("const char").pointerTypes("String", "@Cast(\"const char*\") BytePointer"))

               .put(new Info("CUvideoctxlock").valueTypes("_CUcontextlock_st").pointerTypes("@ByPtrPtr _CUcontextlock_st"))
               .put(new Info("_CUVIDPICPARAMS").valueTypes("CUVIDPICPARAMS"))

               .put(new Info("defined(__CUVID_DEVPTR64)").define(true))
               .put(new Info("defined(__CUVID_DEVPTR64) && !defined(__CUVID_INTERNAL)").define(true))
               .put(new Info("!defined(__CUVID_DEVPTR64) || defined(__CUVID_INTERNAL)").define(false));
    }
}

