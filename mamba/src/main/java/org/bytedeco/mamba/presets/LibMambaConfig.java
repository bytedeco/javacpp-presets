/*
 * Copyright (C) 2016-2020 Samuel Audet, Mark Kittisopikul
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

package org.bytedeco.mamba.presets;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;
import org.bytedeco.javacpp.tools.*;
import org.bytedeco.javacpp.presets.javacpp;

/**
 *
 * @author Mark Kittisopikul
 */
@Properties(inherit = javacpp.class, target = "org.bytedeco.mamba", global = "org.bytedeco.mamba.global.mamba",
    value = @Platform(
        include={"<mamba/api/c_api.h>", "<mamba/api/list.hpp>"},
        includepath = {"include/mamba/api"}
        //includepath = {"../../../cppbuild/linux-x86_64/include/mamba/api"}
        //linkpath = {"../build/libmamba/","/groups/scicompsoft/home/kittisopikulm/mambaforge-pypy3/envs/mambadev/lib"},
        //link = {"mamba", "stdc++"}
    )
)
public class LibMambaConfig implements InfoMapper {
    public void map(InfoMap infoMap) {
    }
}
