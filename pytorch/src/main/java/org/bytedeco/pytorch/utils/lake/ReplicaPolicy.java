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

import java.util.Objects;

/**
 * Read-side replica / consistency policy.
 *
 * <p>For Doris this maps to FE/BE routing hints and session consistency.
 * For open table formats (Iceberg/Hudi/Paimon) there is no classic HDFS replica
 * API — {@link Mode#ANY} is the default and storage-layer redundancy is assumed.
 * The policy still documents intent for multi-path / multi-region readers.</p>
 */
public final class ReplicaPolicy {

    public enum Mode {
        /** Prefer primary / leader replica when the engine exposes one. */
        PRIMARY,
        /** Any healthy replica (default for lakes). */
        ANY,
        /** Prefer data local to the caller (rack / AZ / host affinity when available). */
        LOCAL_PREFERRED
    }

    public enum ReadConsistency {
        /** Strong / latest committed (Doris default point query). */
        STRONG,
        /** Eventual — may read slightly stale (replica lag). */
        EVENTUAL,
        /** Snapshot isolation at a known snapshot/version id. */
        SNAPSHOT
    }

    private final Mode mode;
    private final int minReplicas;
    private final ReadConsistency consistency;
    private final Long snapshotId;

    private ReplicaPolicy(Builder b) {
        this.mode = b.mode;
        this.minReplicas = b.minReplicas;
        this.consistency = b.consistency;
        this.snapshotId = b.snapshotId;
    }

    public static ReplicaPolicy defaults() {
        return builder().build();
    }

    public static Builder builder() {
        return new Builder();
    }

    public Mode mode() { return mode; }
    public int minReplicas() { return minReplicas; }
    public ReadConsistency consistency() { return consistency; }
    public Long snapshotId() { return snapshotId; }

    public static final class Builder {
        private Mode mode = Mode.ANY;
        private int minReplicas = 1;
        private ReadConsistency consistency = ReadConsistency.STRONG;
        private Long snapshotId;

        public Builder mode(Mode mode) {
            this.mode = Objects.requireNonNull(mode);
            return this;
        }

        public Builder minReplicas(int n) {
            this.minReplicas = Math.max(1, n);
            return this;
        }

        public Builder consistency(ReadConsistency c) {
            this.consistency = Objects.requireNonNull(c);
            return this;
        }

        public Builder snapshotId(Long id) {
            this.snapshotId = id;
            return this;
        }

        public ReplicaPolicy build() {
            return new ReplicaPolicy(this);
        }
    }
}
