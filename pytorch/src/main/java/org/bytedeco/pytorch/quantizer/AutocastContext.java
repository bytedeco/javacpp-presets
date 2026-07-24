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
package org.bytedeco.pytorch.quantizer;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * Java port of Python {@code torch.autocast} / {@code torch.amp.autocast_mode.autocast}.
 *
 * <p>There is no public C++ RAII class for autocast (unlike {@code c10::AutoGradMode});
 * Python implements the context manager on top of the free functions in
 * {@code ATen/autocast_mode.h}. This class mirrors that enter/exit logic for try-with-resources.
 *
 * <pre>{@code
 * try (AutocastContext ac = new AutocastContext(DeviceType.CUDA)) {
 *     // forward + loss under mixed precision
 * }
 * // exit before backward()
 * }</pre>
 *
 * <p>State is thread-local, same as the C++ / Python APIs.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class AutocastContext implements AutoCloseable {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private final DeviceType deviceType;
    private final ScalarType fastDtype;
    private final boolean enabled;
    private final boolean cacheEnabled;

    private boolean prevEnabled;
    private ScalarType prevFastDtype;
    private boolean prevCacheEnabled;
    private boolean entered;

    /**
     * Enable autocast on {@code deviceType} with the device's default autocast dtype
     * (fp16 for CUDA, bf16 for CPU — see {@link #get_autocast_dtype}).
     */
    public AutocastContext(DeviceType deviceType) {
        this(deviceType, null, true, true);
    }

    /** Enable/disable autocast with the device default dtype. */
    public AutocastContext(DeviceType deviceType, boolean enabled) {
        this(deviceType, null, enabled, true);
    }

    /**
     * Full constructor, matching Python
     * {@code torch.autocast(device_type, dtype=None, enabled=True, cache_enabled=True)}.
     *
     * @param deviceType   target device (CUDA, CPU, MPS, XPU, …)
     * @param dtype        ops dtype under autocast; {@code null} → {@link #get_autocast_dtype}
     * @param enabled      whether autocast is active in this region
     * @param cacheEnabled whether the autocast weight cache is enabled
     */
    public AutocastContext(DeviceType deviceType, ScalarType dtype,
                           boolean enabled, boolean cacheEnabled) {
        if (deviceType == null) {
            throw new IllegalArgumentException("deviceType must not be null");
        }
        if (!is_autocast_available(deviceType)) {
            throw new RuntimeException(
                "User specified an unsupported autocast device_type '" + deviceType + "'");
        }
        this.deviceType = deviceType;
        this.fastDtype = dtype != null ? dtype : get_autocast_dtype(deviceType);
        this.enabled = enabled;
        this.cacheEnabled = cacheEnabled;
    }

    /** Convenience: CUDA autocast (default dtype fp16). */
    public static AutocastContext cuda() {
        return new AutocastContext(DeviceType.CUDA);
    }

    /** Convenience: CUDA autocast with explicit dtype. */
    public static AutocastContext cuda(ScalarType dtype) {
        return new AutocastContext(DeviceType.CUDA, dtype, true, true);
    }

    /** Convenience: CPU autocast (default dtype bf16). */
    public static AutocastContext cpu() {
        return new AutocastContext(DeviceType.CPU);
    }

    /** Convenience: CPU autocast with explicit dtype. */
    public static AutocastContext cpu(ScalarType dtype) {
        return new AutocastContext(DeviceType.CPU, dtype, true, true);
    }

    public DeviceType deviceType() { return deviceType; }
    public ScalarType fastDtype() { return fastDtype; }
    public boolean enabled() { return enabled; }
    public boolean cacheEnabled() { return cacheEnabled; }

    /**
     * Enter the autocast region (Python {@code __enter__}).
     * Also called automatically when used as try-with-resources if you call {@link #enter()} first,
     * or use {@link #open()} factory.
     */
    public AutocastContext enter() {
        if (entered) {
            throw new IllegalStateException("AutocastContext already entered");
        }
        prevCacheEnabled = is_autocast_cache_enabled();
        prevEnabled = is_autocast_enabled(deviceType);
        prevFastDtype = get_autocast_dtype(deviceType);
        set_autocast_enabled(deviceType, enabled);
        set_autocast_dtype(deviceType, fastDtype);
        increment_nesting();
        set_autocast_cache_enabled(cacheEnabled);
        entered = true;
        return this;
    }

    /**
     * Factory that constructs and enters in one step for try-with-resources:
     * <pre>{@code try (AutocastContext ac = AutocastContext.open(DeviceType.CUDA)) { ... }}</pre>
     */
    public static AutocastContext open(DeviceType deviceType) {
        return new AutocastContext(deviceType).enter();
    }

    public static AutocastContext open(DeviceType deviceType, ScalarType dtype) {
        return new AutocastContext(deviceType, dtype, true, true).enter();
    }

    public static AutocastContext open(DeviceType deviceType, ScalarType dtype,
                                       boolean enabled, boolean cacheEnabled) {
        return new AutocastContext(deviceType, dtype, enabled, cacheEnabled).enter();
    }

    /** Exit the autocast region (Python {@code __exit__}). Idempotent. */
    @Override
    public void close() {
        if (!entered) {
            return;
        }
        // Drop the cache when we exit to a nesting level outside any autocast instance.
        if (decrement_nesting() == 0) {
            clear_cache();
        }
        set_autocast_enabled(deviceType, prevEnabled);
        set_autocast_dtype(deviceType, prevFastDtype);
        set_autocast_cache_enabled(prevCacheEnabled);
        entered = false;
    }
}
