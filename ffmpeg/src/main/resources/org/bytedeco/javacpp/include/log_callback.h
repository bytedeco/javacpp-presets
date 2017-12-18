/*
 * Copyright (C) 2015 Samuel Audet
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

#include <libavutil/log.h>

typedef void (*LogCallback)(int level, const char* msg);

static LogCallback logCallback;

void log_callback(void* ptr, int level, const char* fmt, va_list vl) {
    static int print_prefix = 1;
    char line[1024];

    if ((level & 0xff) > av_log_get_level()) {
        return;
    }

    av_log_format_line(ptr, level, fmt, vl, line, sizeof(line), &print_prefix);
    logCallback(level, line);
}

void setLogCallback(LogCallback lc) {
    av_log_set_callback(log_callback);
    logCallback = lc;
}
