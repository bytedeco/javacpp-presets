/*
 * Copyright (C) 2020 Samuel Audet
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
 */

package org.bytedeco.arrow.presets;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Samuel Audet
 */
@Properties(
    inherit = arrow.class,
    value = {
        @Platform(
            not = "windows",
            include = {
                "plasma/compat.h",
                "plasma/common.h",
                "plasma/client.h",
                "plasma/events.h",
                "plasma/test_util.h",
            },
            link = "plasma@.200"
        ),
    },
    target = "org.bytedeco.plasma",
    global = "org.bytedeco.arrow.global.plasma"
)
public class plasma implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "plasma"); }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info("__APPLE__", "PLASMA_CUDA").define(false))
               .put(new Info("std::hash<plasma::UniqueID>").pointerTypes("UniqueIDHash"))
               .put(new Info("plasma::EventLoop::FileCallback", "plasma::EventLoop::TimerCallback").cast().pointerTypes("Pointer"))
               .put(new Info("plasma::plasma_config").javaText("@Namespace(\"plasma\") @MemberGetter public static native @Const PlasmaStoreInfo plasma_config();"))
        ;
    }
}
