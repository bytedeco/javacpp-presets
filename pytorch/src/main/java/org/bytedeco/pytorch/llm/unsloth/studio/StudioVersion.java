/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
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
package org.bytedeco.pytorch.llm.unsloth.studio;

/**
 * Version constants for the pure-Java Unsloth Studio surface.
 *
 * <p>Independent of upstream Python Unsloth Studio packaging; bumped when the
 * Java product API changes incompatibly.
 */
public final class StudioVersion {

    public static final String VERSION = "1.0.0-beta";
    public static final String CODENAME = "studio-java";
    /** Upstream studio tree this port targets behaviourally. */
    public static final String UPSTREAM_REF = "unslothai/unsloth/studio@main";

    private StudioVersion() {}

    public static String version() {
        return VERSION;
    }

    public static String full() {
        return VERSION + " (" + CODENAME + "; aligns " + UPSTREAM_REF + ")";
    }
}
