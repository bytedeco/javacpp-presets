package org.bytedeco.javacpp.presets;

import org.bytedeco.javacpp.annotation.*;
import org.bytedeco.javacpp.tools.*;
import org.bytedeco.javacpp.*;

/**
 *
 * @author Charles Rose
 */
@Properties(
    value = {
        @Platform(
            value = {"linux-x86_64", "macosx-x86_64"}, 
            link = "raw@.16",
            include = {
                "libraw/libraw.h",
                "libraw/libraw_alloc.h",
                "libraw/libraw_const.h",
                "libraw/libraw_datastream.h",
                "libraw/libraw_types.h"
            }
        )
    },
    target = "org.bytedeco.javacpp.libraw"
)
public class libraw implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("rawProcessor"))
               .put(new Info("INT64").cppTypes("long"))
               .put(new Info("DllDef", "progress_callback", "LibRaw_progress", "dng_stream", "libraw_internal_data_t", "libraw_memmgr", "LibRaw::set_exifparser_handler", "LibRaw::set_memerror_handler", "LibRaw::set_dataerror_handler", "LibRaw::phase_one_subtract_black").skip())
               .put(new Info(
                    "WIN32", 
                    "_WIN32", 
                    "defined(_WIN32) && !defined(__MINGW32__) && defined(_MSC_VER) && (_MSC_VER > 1310)", 
                    "__linux__"
                ).define(false));
    }

}
