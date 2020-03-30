/*
 * Copyright (C) 2014-2020 Samuel Audet, Jarek Sacha
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package org.bytedeco.spinnaker.presets;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.presets.javacpp;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 * Wrapper for Point Grey Spinnaker_C library (the C API v.1).
 *
 * @author Jarek Sacha
 */
@Properties(inherit = javacpp.class,
            target = "org.bytedeco.spinnaker.Spinnaker_C",
            global = "org.bytedeco.spinnaker.global.Spinnaker_C", value = {
        @Platform(value = {"linux-x86_64", "linux-arm64", "windows"},
                include = {"<SpinnakerPlatformC.h>",
                        "<SpinnakerDefsC.h>",
                        "<CameraDefsC.h>",
                        "<ChunkDataDefC.h>",
                        "<SpinnakerGenApiDefsC.h>",
                        "<SpinnakerGenApiC.h>",
                        "<SpinnakerC.h>",
                        "<SpinVideoC.h>",
                        "<QuickSpinDefsC.h>",
                        "<QuickSpinC.h>",
                        "<TransportLayerDefsC.h>",
                        "<TransportLayerDeviceC.h>",
                        "<TransportLayerInterfaceC.h>",
                        "<TransportLayerStreamC.h>",
                        "<TransportLayerSystemC.h>",
                },
                link = {"SpinVideo_C@.1", "Spinnaker_C@.1"},
                linkpath = "/usr/lib/",
                includepath = "/usr/include/spinnaker/spinc/"),
        @Platform(value = "windows", link = {"SpinnakerC_v140", "Spinnaker_v140", "SpinVideoC_v140", "SpinVideo_v140"},
                includepath = {"C:/Program Files/Spinnaker/include/spinc/", "C:/Program Files/Point Grey Research/Spinnaker/include/spinc/",
                               "C:/Program Files (x86)/Spinnaker/include/spinc/", "C:/Program Files (x86)/Point Grey Research/Spinnaker/include/spinc/"}),
        @Platform(value = "windows-x86",
                linkpath    = {"C:/Program Files/Spinnaker/lib/vs2015/", "C:/Program Files/Point Grey Research/Spinnaker/lib/vs2015/",
                               "C:/Program Files (x86)/Spinnaker/lib/vs2015/", "C:/Program Files (x86)/Point Grey Research/Spinnaker/lib/vs2015/"},
                preloadpath = {"C:/Program Files/Spinnaker/bin/vs2015/", "C:/Program Files/Point Grey Research/Spinnaker/bin/vs2015/",
                               "C:/Program Files (x86)/Spinnaker/bin/vs2015/", "C:/Program Files (x86)/Point Grey Research/Spinnaker/bin/vs2015/"}),
        @Platform(value = "windows-x86_64",
                linkpath    = {"C:/Program Files/Spinnaker/lib64/vs2015/", "C:/Program Files/Point Grey Research/Spinnaker/lib64/vs2015/"},
                preloadpath = {"C:/Program Files/Spinnaker/bin64/vs2015/", "C:/Program Files/Point Grey Research/Spinnaker/bin64/vs2015/"})})
public class Spinnaker_C implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "spinnaker"); }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info().enumerate())
                .put(new Info("SPINC_CALLTYPE").cppTypes().annotations().cppText(""))
                .put(new Info("SPINC_NO_DECLSPEC_STATEMENTS", "EXTERN_C").define())
                .put(new Info("SPINNAKERC_API").cppTypes("_spinError").cppText("enum _spinError").define())
                // Skip to avoid linker errors,
                // somehow JavaCPP did not generate wrapper for 'SPINNAKERC_API spinRegisterSetEx(...)'
                .put(new Info("spinRegisterSetEx").skip())
                // Skip deprecation macro, as it is causing parsing error in javacpp
                .put(new Info("SPINNAKERC_API_DEPRECATED").skip())
                // Avoid: jniSpinnaker_C.obj : error LNK2001: unresolved external symbol __imp_spinCameraForceIP
                .put(new Info("spinCameraForceIP").skip())
        ;
    }
}
