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
package org.bytedeco.pytorch.llm.unsloth.studio.util;

/**
 * Validation helpers aligned with upstream Unsloth Studio request limits
 * ({@code studio/backend/models/training.py} / {@code export.py}).
 */
public final class Validate {

    public static final int MAX_BATCH_SIZE = 4096;
    public static final int MAX_GRAD_ACCUM = 4096;
    public static final int MAX_STEPS = 1_000_000;
    public static final int MAX_EPOCHS = 1000;
    public static final int MAX_SEQ_LENGTH = 2_000_000;
    public static final double MAX_LR_VALUE = 1.0;
    public static final int MAX_LORA_R = 16_384;
    public static final int MAX_LORA_ALPHA = 32_768;
    public static final int MIN_VISION_IMAGE_SIZE = 256;
    public static final int MAX_VISION_IMAGE_SIZE = 2048;
    public static final long MAX_DATASET_SLICE_INDEX = 1_000_000_000L;
    public static final int MAX_PATH_COMPONENT = 255;
    public static final int MAX_CHAT_TEMPLATE_BYTES = 512 * 1024;
    public static final int MAX_PROJECT_NAME = 80;

    private Validate() {}

    public static void requireNonBlank(String name, String value) {
        if (value == null || value.isBlank()) {
            throw new StudioValidationException(name + " must not be blank");
        }
    }

    public static void requireNonNull(String name, Object value) {
        if (value == null) {
            throw new StudioValidationException(name + " is required");
        }
    }

    public static int batchSize(int v) {
        if (v < 1 || v > MAX_BATCH_SIZE) {
            throw new StudioValidationException(
                    "batch_size must be in [1, " + MAX_BATCH_SIZE + "], got " + v);
        }
        return v;
    }

    public static int gradAccum(int v) {
        if (v < 1 || v > MAX_GRAD_ACCUM) {
            throw new StudioValidationException(
                    "gradient_accumulation_steps must be in [1, " + MAX_GRAD_ACCUM + "], got " + v);
        }
        return v;
    }

    public static int maxSteps(int v) {
        if (v < 1 || v > MAX_STEPS) {
            throw new StudioValidationException(
                    "max_steps must be in [1, " + MAX_STEPS + "], got " + v);
        }
        return v;
    }

    public static double epochs(double v) {
        if (v <= 0 || v > MAX_EPOCHS) {
            throw new StudioValidationException(
                    "num_train_epochs must be in (0, " + MAX_EPOCHS + "], got " + v);
        }
        return v;
    }

    public static int maxSeqLength(int v) {
        if (v < 1 || v > MAX_SEQ_LENGTH) {
            throw new StudioValidationException(
                    "max_seq_length must be in [1, " + MAX_SEQ_LENGTH + "], got " + v);
        }
        return v;
    }

    public static double learningRate(double v) {
        if (!(v > 0.0) || v >= MAX_LR_VALUE) {
            throw new StudioValidationException(
                    "learning_rate must be in (0, 1.0), got " + v + "; typical range is 1e-6 .. 1e-3");
        }
        return v;
    }

    public static double parseLearningRate(Object raw) {
        if (raw == null) {
            throw new StudioValidationException("learning_rate is required");
        }
        if (raw instanceof Boolean) {
            throw new StudioValidationException("learning_rate must be a number, not a bool");
        }
        final double lr;
        try {
            lr = raw instanceof Number ? ((Number) raw).doubleValue() : Double.parseDouble(String.valueOf(raw));
        } catch (NumberFormatException e) {
            throw new StudioValidationException("learning_rate must be parseable as float (got " + raw + ")");
        }
        return learningRate(lr);
    }

    public static int loraR(int v) {
        if (v < 1 || v > MAX_LORA_R) {
            throw new StudioValidationException("lora_r must be in [1, " + MAX_LORA_R + "], got " + v);
        }
        return v;
    }

    public static int loraAlpha(int v) {
        if (v < 1 || v > MAX_LORA_ALPHA) {
            throw new StudioValidationException(
                    "lora_alpha must be in [1, " + MAX_LORA_ALPHA + "], got " + v);
        }
        return v;
    }

    public static Integer visionImageSize(Integer v) {
        if (v == null) {
            return null;
        }
        if (v < MIN_VISION_IMAGE_SIZE || v > MAX_VISION_IMAGE_SIZE) {
            throw new StudioValidationException(
                    "vision_image_size must be in [" + MIN_VISION_IMAGE_SIZE + ", "
                            + MAX_VISION_IMAGE_SIZE + "] or null, got " + v);
        }
        return v;
    }

    public static long datasetSliceIndex(long v) {
        if (v < 0 || v > MAX_DATASET_SLICE_INDEX) {
            throw new StudioValidationException(
                    "dataset slice index must be in [0, " + MAX_DATASET_SLICE_INDEX + "], got " + v);
        }
        return v;
    }

    /**
     * Validate a user-supplied save directory. Mirrors upstream export path rules:
     * no null bytes, no control chars, no {@code ..} segments, component length ≤ 255.
     */
    public static String saveDirectory(String value) {
        requireNonBlank("save_directory", value);
        String raw = value.trim();
        if (raw.indexOf('\0') >= 0) {
            throw new StudioValidationException("save_directory may not contain null bytes");
        }
        for (int i = 0; i < raw.length(); i++) {
            char c = raw.charAt(i);
            if (c == '\r' || c == '\n') {
                throw new StudioValidationException("save_directory may not contain control characters");
            }
        }
        String normalized = raw.replace('\\', '/');
        String[] parts = normalized.split("/");
        for (String part : parts) {
            if (part.isEmpty() || ".".equals(part)) {
                continue;
            }
            if ("..".equals(part)) {
                throw new StudioValidationException("save_directory may not contain '..' segments");
            }
            if (part.length() > MAX_PATH_COMPONENT) {
                throw new StudioValidationException(
                        "save_directory path components must be <= " + MAX_PATH_COMPONENT + " characters");
            }
        }
        return raw;
    }

    public static String projectName(String value) {
        if (value == null || value.isBlank()) {
            return null;
        }
        String t = value.trim();
        if (t.length() > MAX_PROJECT_NAME) {
            throw new StudioValidationException(
                    "project_name max length is " + MAX_PROJECT_NAME + ", got " + t.length());
        }
        return t;
    }

    public static String chatTemplateOverride(String value) {
        if (value == null) {
            return null;
        }
        if (value.isBlank()) {
            return null;
        }
        if (value.length() > MAX_CHAT_TEMPLATE_BYTES) {
            throw new StudioValidationException(
                    "Chat template exceeds the " + MAX_CHAT_TEMPLATE_BYTES + "-byte limit.");
        }
        byte[] utf8 = value.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        if (utf8.length > MAX_CHAT_TEMPLATE_BYTES) {
            throw new StudioValidationException(
                    "Chat template exceeds the " + MAX_CHAT_TEMPLATE_BYTES + "-byte limit.");
        }
        return value;
    }

    public static int range(String name, int v, int min, int max) {
        if (v < min || v > max) {
            throw new StudioValidationException(name + " must be in [" + min + ", " + max + "], got " + v);
        }
        return v;
    }

    public static double range(String name, double v, double min, double maxExclusive) {
        if (!(v >= min) || !(v < maxExclusive)) {
            throw new StudioValidationException(
                    name + " must be in [" + min + ", " + maxExclusive + "), got " + v);
        }
        return v;
    }
}
