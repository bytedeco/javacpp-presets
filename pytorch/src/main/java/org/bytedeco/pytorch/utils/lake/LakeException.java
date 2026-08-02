/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet, mullerhai
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
package org.bytedeco.pytorch.utils.lake;

/**
 * Unchecked failure from lake adapters (connection, commit, stream load, catalog).
 */
public class LakeException extends RuntimeException {

    private final LakeFormat format;
    private final String operation;

    public LakeException(String message) {
        this(null, null, message, null);
    }

    public LakeException(String message, Throwable cause) {
        this(null, null, message, cause);
    }

    public LakeException(LakeFormat format, String operation, String message) {
        this(format, operation, message, null);
    }

    public LakeException(LakeFormat format, String operation, String message, Throwable cause) {
        super(buildMessage(format, operation, message), cause);
        this.format = format;
        this.operation = operation;
    }

    public LakeFormat format() {
        return format;
    }

    public String operation() {
        return operation;
    }

    private static String buildMessage(LakeFormat format, String operation, String message) {
        StringBuilder sb = new StringBuilder();
        if (format != null) sb.append('[').append(format).append("] ");
        if (operation != null && !operation.isBlank()) sb.append(operation).append(": ");
        if (message != null) sb.append(message);
        return sb.toString();
    }
}
