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

package org.bytedeco.pytorch.llm.unsloth.studio.hardware;

import org.bytedeco.pytorch.llm.unsloth.studio.model.HardwareProfile;

import java.lang.management.ManagementFactory;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Cross-platform device probe (CPU / CUDA / MPS / ROCm hints).
 * Uses JVM + optional libtorch CUDA query without requiring a GPU at compile time.
 */
public final class DeviceProbe {

    private DeviceProbe() {}

    public static HardwareProfile probe() {
        String os = System.getProperty("os.name", "unknown");
        String arch = System.getProperty("os.arch", "unknown");
        int cores = Runtime.getRuntime().availableProcessors();
        long memMb = -1;
        try {
            com.sun.management.OperatingSystemMXBean osb =
                    (com.sun.management.OperatingSystemMXBean) ManagementFactory.getOperatingSystemMXBean();
            memMb = osb.getTotalMemorySize() / (1024L * 1024L);
        } catch (Throwable t) {
            memMb = Runtime.getRuntime().maxMemory() / (1024L * 1024L);
        }

        boolean cuda = false;
        boolean mps = false;
        boolean rocm = false;
        List<HardwareProfile.GpuDevice> gpus = new ArrayList<>();

        try {
            // Reflective / soft probe — never fail hard if native not loaded.
            Class<?> torch = Class.forName("org.bytedeco.pytorch.global.torch");
            try {
                Object cudaMod = Class.forName("org.bytedeco.pytorch.global.torch$cuda").getField("instance").get(null);
            } catch (Throwable ignored) {}
            try {
                boolean avail = (Boolean) Class.forName("org.bytedeco.pytorch.global.torch")
                        .getMethod("cuda_is_available").invoke(null);
                cuda = avail;
            } catch (Throwable t1) {
                try {
                    // alternate: torch.cuda.is_available via Device
                    cuda = Boolean.getBoolean("jnitorch.cuda")
                            || System.getenv("CUDA_VISIBLE_DEVICES") != null
                            || FilesExists("/usr/local/cuda")
                            || FilesExists("/usr/lib/wsl/lib");
                } catch (Throwable ignored) {}
            }
        } catch (Throwable ignored) {
            cuda = System.getenv("CUDA_VISIBLE_DEVICES") != null;
        }

        String osLower = os.toLowerCase(Locale.ROOT);
        if (osLower.contains("mac")) {
            // Apple Silicon may expose MPS via metal; we report capability hint only.
            mps = arch.toLowerCase(Locale.ROOT).contains("aarch64")
                    || arch.toLowerCase(Locale.ROOT).contains("arm");
        }
        String rocmPath = System.getenv("ROCM_PATH");
        rocm = rocmPath != null && !rocmPath.isBlank();

        if (cuda) {
            int count = 1;
            try {
                String vis = System.getenv("CUDA_VISIBLE_DEVICES");
                if (vis != null && !vis.isBlank() && !vis.equals("-1")) {
                    count = Math.max(1, vis.split(",").length);
                }
            } catch (Throwable ignored) {}
            for (int i = 0; i < count; i++) {
                gpus.add(new HardwareProfile.GpuDevice(i, "CUDA GPU " + i, estimateCudaVramMb(i), "cuda"));
            }
        } else if (rocm) {
            gpus.add(new HardwareProfile.GpuDevice(0, "ROCm GPU 0", 0, "rocm"));
        } else if (mps) {
            gpus.add(new HardwareProfile.GpuDevice(0, "Apple MPS", memMb > 0 ? memMb / 2 : 0, "mps"));
        }

        String rec = "cpu";
        if (cuda) rec = "cuda";
        else if (rocm) rec = "cuda"; // torch HIP uses cuda device API in many builds
        else if (mps) rec = "mps";

        return new HardwareProfile(os, arch, cores, memMb, cuda, mps, rocm, gpus, rec);
    }

    private static long estimateCudaVramMb(int index) {
        // Best-effort; real VRAM comes from nvidia-smi if present.
        try {
            Process p = new ProcessBuilder("nvidia-smi",
                    "--query-gpu=memory.total",
                    "--format=csv,noheader,nounits")
                    .redirectErrorStream(true)
                    .start();
            String out = new String(p.getInputStream().readAllBytes());
            p.waitFor();
            String[] lines = out.trim().split("\\R");
            if (index < lines.length) {
                return Long.parseLong(lines[index].trim());
            }
            if (lines.length > 0 && !lines[0].isBlank()) {
                return Long.parseLong(lines[0].trim());
            }
        } catch (Throwable ignored) {}
        return 0;
    }

    private static boolean FilesExists(String path) {
        try {
            return java.nio.file.Files.exists(java.nio.file.Path.of(path));
        } catch (Throwable t) {
            return false;
        }
    }
}
