package org.bytedeco.javacpp.presets;

import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;


/**
 * Generate JavaCpp code for Annoy
 * https://github.com/spotify/annoy/tree/master/src
 */
@Properties(target="org.bytedeco.javacpp.annoy", value={@Platform(include={
        "/mnt/workspace/justice-data/source_code/src/main/cpp/annoylib.h",
        "/mnt/workspace/justice-data/source_code/src/main/cpp/kissrandom.h"
},
        link="z@.1")
})
public class annoy_preset implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap
                .put(new Info("AnnoyIndexInterface<int,float>").pointerTypes("AnnoyIndexInterface"))
                .put(new Info("AnnoyIndex<int,float,Euclidean,Kiss64Random>").pointerTypes("AnnoyIndexEuclidean").base("AnnoyIndexInterface"))
                .put(new Info("AnnoyIndex<int,float,Angular,Kiss64Random>").pointerTypes("AnnoyIndexAngular").base("AnnoyIndexInterface"))
                .put(new Info("ANNOY_NODE_ATTRIBUTE").cppTypes().annotations())
        ;
    }
}
