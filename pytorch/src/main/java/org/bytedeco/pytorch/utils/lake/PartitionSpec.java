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
 * Per-table partition specification.
 *
 * <p>Used for prune, compaction, and write-time partitioning control.</p>
 */
public class PartitionSpec {
    private final String[] identityColumns;
    private final String[] timeTruncate;   // e.g. "hour", "day"
    private final int[] bucketColumns;     // hash bucket count per column

    private PartitionSpec(Builder builder) {
        this.identityColumns = builder.identityColumns;
        this.timeTruncate = builder.timeTruncate;
        this.bucketColumns = builder.bucketColumns;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private String[] identityColumns = new String[0];
        private String[] timeTruncate = new String[0];
        private int[] bucketColumns = new int[0];

        public Builder identityColumns(String... columns) {
            this.identityColumns = columns;
            return this;
        }

        public Builder timeTruncate(String... trunc) {
            this.timeTruncate = trunc;
            return this;
        }

        public Builder bucketColumns(int... buckets) {
            this.bucketColumns = buckets;
            return this;
        }

        public PartitionSpec build() {
            return new PartitionSpec(this);
        }
    }

    public String[] identityColumns() {
        return identityColumns;
    }

    public String[] timeTruncate() {
        return timeTruncate;
    }

    public int[] bucketColumns() {
        return bucketColumns;
    }

    @Override
    public String toString() {
        return "PartitionSpec{identity=" + java.util.Arrays.toString(identityColumns) +
               ", timeTruncate=" + java.util.Arrays.toString(timeTruncate) +
               ", bucket=" + java.util.Arrays.toString(bucketColumns) + "}";
    }
}
