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
package org.bytedeco.pytorch.llm.llamafactory.hparams;

import java.util.LinkedHashMap;
import java.util.Map;

/** Merge / export args (LLaMA-Factory export section). */
public final class ExportArgs {
    private final String exportDir;
    private final int exportSize;
    private final String exportDevice;
    private final String exportDtype;
    private final boolean exportLegacyFormat;
    private final String exportHubModelId;
    private final boolean mergeAdapters;

    private ExportArgs(Builder b) {
        this.exportDir = b.exportDir == null ? "export" : b.exportDir;
        this.exportSize = b.exportSize;
        this.exportDevice = b.exportDevice == null ? "cpu" : b.exportDevice;
        this.exportDtype = b.exportDtype == null ? "float16" : b.exportDtype;
        this.exportLegacyFormat = b.exportLegacyFormat;
        this.exportHubModelId = b.exportHubModelId;
        this.mergeAdapters = b.mergeAdapters;
    }

    public String exportDir() { return exportDir; }
    public int exportSize() { return exportSize; }
    public String exportDevice() { return exportDevice; }
    public String exportDtype() { return exportDtype; }
    public boolean exportLegacyFormat() { return exportLegacyFormat; }
    public String exportHubModelId() { return exportHubModelId; }
    public boolean mergeAdapters() { return mergeAdapters; }

    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>();
        HparamsMaps.put(m, "export_dir", exportDir);
        HparamsMaps.put(m, "export_size", exportSize);
        HparamsMaps.put(m, "export_device", exportDevice);
        HparamsMaps.put(m, "export_dtype", exportDtype);
        HparamsMaps.put(m, "export_legacy_format", exportLegacyFormat);
        HparamsMaps.put(m, "export_hub_model_id", exportHubModelId);
        HparamsMaps.put(m, "merge_adapters", mergeAdapters);
        return m;
    }

    public static ExportArgs defaults() { return builder().build(); }

    public static ExportArgs fromMap(Map<String, ?> m) {
        if (m == null || m.isEmpty()) return defaults();
        Builder b = builder();
        b.exportDir(HparamsMaps.str(m, b.exportDir, "export_dir"));
        b.exportSize(HparamsMaps.integer(m, b.exportSize, "export_size"));
        b.exportDevice(HparamsMaps.str(m, b.exportDevice, "export_device"));
        b.exportDtype(HparamsMaps.str(m, b.exportDtype, "export_dtype"));
        b.exportLegacyFormat(HparamsMaps.bool(m, b.exportLegacyFormat, "export_legacy_format"));
        b.exportHubModelId(HparamsMaps.strOrNull(m, "export_hub_model_id"));
        b.mergeAdapters(HparamsMaps.bool(m, b.mergeAdapters, "merge_adapters"));
        return b.build();
    }

    public static Builder builder() { return new Builder(); }

    public static final class Builder {
        private String exportDir = "export";
        private int exportSize = 5;
        private String exportDevice = "cpu";
        private String exportDtype = "float16";
        private boolean exportLegacyFormat;
        private String exportHubModelId;
        private boolean mergeAdapters = true;

        public Builder exportDir(String v) { this.exportDir = v; return this; }
        public Builder exportSize(int v) { this.exportSize = v; return this; }
        public Builder exportDevice(String v) { this.exportDevice = v; return this; }
        public Builder exportDtype(String v) { this.exportDtype = v; return this; }
        public Builder exportLegacyFormat(boolean v) { this.exportLegacyFormat = v; return this; }
        public Builder exportHubModelId(String v) { this.exportHubModelId = v; return this; }
        public Builder mergeAdapters(boolean v) { this.mergeAdapters = v; return this; }
        public ExportArgs build() { return new ExportArgs(this); }
    }
}
