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
package org.bytedeco.pytorch.utils.ffmpeg;

/**
 * Unchecked exception wrapping FFmpeg error codes.
 *
 * <p>PyAV equivalent: {@code av.error.FFmpegError} / {@code av.error.ValueError}.
 */
public class FFmpegException extends RuntimeException {

    private final int errorCode;

    public FFmpegException(String message) {
        super(message);
        this.errorCode = -1;
    }

    public FFmpegException(String message, int errorCode) {
        super(message + " (error " + errorCode + (errorCode < 0 ? ": " + errorMessage(errorCode) : "") + ")");
        this.errorCode = errorCode;
    }

    public FFmpegException(String message, Throwable cause) {
        super(message, cause);
        this.errorCode = -1;
    }

    public int errorCode() {
        return errorCode;
    }

    /** Make a descriptive message from an FFmpeg negative error code via {@code av_strerror}. */
    public static String errorMessage(int code) {
        try {
            FFmpegNative.load();
            return FFmpegNative.errorString(code);
        } catch (Throwable t) {
            return "FFmpeg error " + code;
        }
    }
}
