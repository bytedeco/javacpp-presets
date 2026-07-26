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
package org.bytedeco.pytorch.utils.tqdm;

/**
 * ANSI colors for tqdm progress bars (inspired by bytedeco/storch-tqdm {@code ProgressBarColor}).
 */
public enum ProgressBarColor {
    NONE("", ""),
    BLACK("[30m", "[0m"),
    RED("[31m", "[0m"),
    GREEN("[32m", "[0m"),
    YELLOW("[33m", "[0m"),
    BLUE("[34m", "[0m"),
    MAGENTA("[35m", "[0m"),
    CYAN("[36m", "[0m"),
    WHITE("[37m", "[0m"),
    BRIGHT_BLACK("[90m", "[0m"),
    BRIGHT_RED("[91m", "[0m"),
    BRIGHT_GREEN("[92m", "[0m"),
    BRIGHT_YELLOW("[93m", "[0m"),
    BRIGHT_BLUE("[94m", "[0m"),
    BRIGHT_MAGENTA("[95m", "[0m"),
    BRIGHT_CYAN("[96m", "[0m"),
    BRIGHT_WHITE("[97m", "[0m");

    private final String on;
    private final String off;

    ProgressBarColor(String on, String off) {
        this.on = on;
        this.off = off;
    }

    public String apply(String text) {
        if (this == NONE || text == null || text.isEmpty()) {
            return text == null ? "" : text;
        }
        return on + text + off;
    }

    public String on() {
        return on;
    }

    public String off() {
        return off;
    }

    /** Parse common colour names (case-insensitive); unknown → {@link #NONE}. */
    public static ProgressBarColor fromName(String name) {
        if (name == null || name.isEmpty()) {
            return NONE;
        }
        String n = name.trim().toLowerCase().replace('-', '_').replace(' ', '_');
        for (ProgressBarColor c : values()) {
            if (c.name().equalsIgnoreCase(n)) {
                return c;
            }
        }
        return NONE;
    }
}
