package org.bytedeco.leapmotion.presets;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;
import org.bytedeco.javacpp.tools.*;

@Properties(
    value = {
        @Platform(
                value = "windows",
                include = "LeapC.h",
                link = "LeapC")
    },
    target = "org.bytedeco.leapmotion",
    global = "org.bytedeco.leapmotion.global.leapmotion"
)
public class leapmotion implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "leapmotion"); }

    public void map(InfoMap infoMap) {

        infoMap.put(new Info("LEAP_EXPORT","LEAP_CALL","LEAP_STATIC_ASSERT","static_assert").cppTypes().annotations())
                
                .put(new Info("_LEAP_CONNECTION").pointerTypes("LEAP_CONNECTION"))
                .put(new Info("LEAP_CONNECTION").valueTypes("LEAP_CONNECTION").pointerTypes("@Cast(\"LEAP_CONNECTION*\") _LEAP_CONNECTION", "@ByPtrPtr LEAP_CONNECTION"))
                
                .put(new Info("_LEAP_DEVICE").pointerTypes("LEAP_DEVICE"))
                .put(new Info("LEAP_DEVICE").valueTypes("LEAP_DEVICE").pointerTypes("@Cast(\"LEAP_DEVICE*\") _LEAP_DEVICE", "@ByPtrPtr LEAP_DEVICE"))

                .put(new Info("_LEAP_CLOCK_REBASER").pointerTypes("LEAP_CLOCK_REBASER"))
                .put(new Info("LEAP_CLOCK_REBASER").valueTypes("LEAP_CLOCK_REBASER").pointerTypes("@Cast(\"LEAP_CLOCK_REBASER*\") _LEAP_CLOCK_REBASER", "@ByPtrPtr LEAP_CLOCK_REBASER"))

                .put(new Info("_LEAP_RECORDING").pointerTypes("LEAP_RECORDING"))
                .put(new Info("LEAP_RECORDING").valueTypes("LEAP_RECORDING").pointerTypes("@Cast(\"LEAP_RECORDING*\") _LEAP_RECORDING", "@ByPtrPtr LEAP_RECORDING"))

                .put(new Info("_LEAP_CALIBRATION").pointerTypes("LEAP_CALIBRATION"))
                .put(new Info("LEAP_CALIBRATION").valueTypes("LEAP_CALIBRATION").pointerTypes("@Cast(\"LEAP_CALIBRATION*\") _LEAP_CALIBRATION", "@ByPtrPtr LEAP_CALIBRATION"));
            
                // .put(new Info("_LEAP_DISTORTION_MATRIX").pointerTypes("LEAP_DISTORTION_MATRIX"));
                // .put(new Info("LEAP_DISTORTION_MATRIX").valueTypes("distortion_matrix").pointerTypes("@Cast(\"LEAP_DISTORTION_MATRIX*\") PointerPointer", "@ByPtrPtr distortion_matrix"));
    }
}