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

import java.util.Collections;
import java.util.EnumSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Logical lake table handle: identity, schema, partition, location, properties.
 */
public final class LakeTable {

    private final LakeFormat format;
    private final String namespaceName;
    private final String name;
    private final LakeSchema schema;
    private final PartitionSpec partitionSpec;
    private final String location;
    private final Map<String, String> properties;
    private final Set<LakeCapabilities> capabilities;
    private final Long currentSnapshotId;

    private LakeTable(Builder b) {
        this.format = Objects.requireNonNull(b.format, "format");
        this.namespaceName = b.namespaceName == null ? "" : b.namespaceName;
        this.name = Objects.requireNonNull(b.name, "name");
        this.schema = Objects.requireNonNull(b.schema, "schema");
        this.partitionSpec = b.partitionSpec;
        this.location = b.location;
        this.properties = Collections.unmodifiableMap(new LinkedHashMap<>(b.properties));
        this.capabilities = b.capabilities.isEmpty()
                ? Set.of()
                : Collections.unmodifiableSet(EnumSet.copyOf(b.capabilities));
        this.currentSnapshotId = b.currentSnapshotId;
    }

    public static Builder builder(LakeFormat format, String name, LakeSchema schema) {
        return new Builder(format, name, schema);
    }

    public LakeFormat format() { return format; }
    public String namespaceName() { return namespaceName; }
    public String name() { return name; }
    public String fullName() {
        return namespaceName == null || namespaceName.isEmpty() ? name : namespaceName + "." + name;
    }
    public LakeSchema schema() { return schema; }
    public PartitionSpec partitionSpec() { return partitionSpec; }
    public String location() { return location; }
    public Map<String, String> properties() { return properties; }
    public Set<LakeCapabilities> capabilities() { return capabilities; }
    public Long currentSnapshotId() { return currentSnapshotId; }

    public boolean supports(LakeCapabilities c) {
        return capabilities.contains(c);
    }

    public static final class Builder {
        private final LakeFormat format;
        private String namespaceName;
        private final String name;
        private final LakeSchema schema;
        private PartitionSpec partitionSpec;
        private String location;
        private final Map<String, String> properties = new LinkedHashMap<>();
        private final EnumSet<LakeCapabilities> capabilities = EnumSet.noneOf(LakeCapabilities.class);
        private Long currentSnapshotId;

        private Builder(LakeFormat format, String name, LakeSchema schema) {
            this.format = format;
            this.name = name;
            this.schema = schema;
        }

        public Builder namespaceName(String ns) { this.namespaceName = ns; return this; }
        public Builder partitionSpec(PartitionSpec spec) { this.partitionSpec = spec; return this; }
        public Builder location(String loc) { this.location = loc; return this; }
        public Builder property(String k, String v) {
            if (k != null && v != null) properties.put(k, v);
            return this;
        }
        public Builder properties(Map<String, String> m) {
            if (m != null) properties.putAll(m);
            return this;
        }
        public Builder capability(LakeCapabilities c) {
            if (c != null) capabilities.add(c);
            return this;
        }
        public Builder capabilities(LakeCapabilities... caps) {
            if (caps != null) for (LakeCapabilities c : caps) capability(c);
            return this;
        }
        public Builder currentSnapshotId(Long id) { this.currentSnapshotId = id; return this; }

        public LakeTable build() {
            return new LakeTable(this);
        }
    }

    @Override
    public String toString() {
        return "LakeTable{" + format + " " + fullName()
                + " fields=" + schema.size()
                + (location != null ? " loc=" + location : "")
                + "}";
    }
}
