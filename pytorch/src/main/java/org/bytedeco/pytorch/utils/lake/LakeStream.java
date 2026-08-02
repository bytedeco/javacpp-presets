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

import org.bytedeco.pytorch.dataframe.DataFrame;

import java.time.Duration;
import java.util.Iterator;
import java.util.Objects;
import java.util.function.Consumer;

/**
 * Continuous micro-batch lake → {@link DataFrame} stream for online training / feature join.
 *
 * <p>API mirrors {@code utils.kafka.KafkaStream}:</p>
 * <pre>{@code
 * try (LakeStream stream = catalog.stream("db", "events")) {
 *     stream.batchRows(4096).forEachBatch(df -> {
 *         // feature assemble / train step
 *         stream.commit();
 *     });
 * }
 * }</pre>
 */
public interface LakeStream extends AutoCloseable, Iterable<DataFrame> {

    LakeStream batchRows(int batchRows);

    LakeStream idleStop(Duration idle);

    LakeStream maxBatches(long maxBatches);

    /**
     * Advance the read watermark (snapshot id, JDBC cursor, file offset).
     * Semantics are engine-specific; at-least-once unless the adapter documents stronger.
     */
    void commit() throws LakeException;

    /** Cooperative stop of a blocking forEachBatch loop. */
    void stop();

    boolean isStopped();

    /**
     * Pull next batch or {@code null} when idle-stop / end-of-snapshot reached.
     */
    DataFrame poll() throws LakeException;

    default void forEachBatch(Consumer<DataFrame> consumer) throws LakeException {
        Objects.requireNonNull(consumer, "consumer");
        while (!isStopped()) {
            DataFrame df = poll();
            if (df == null) break;
            consumer.accept(df);
        }
    }

    default void forEachBatch(int batchRows, Consumer<DataFrame> consumer) throws LakeException {
        batchRows(batchRows);
        forEachBatch(consumer);
    }

    @Override
    default Iterator<DataFrame> iterator() {
        return new Iterator<>() {
            private DataFrame next;
            private boolean primed;
            private boolean done;

            private void prime() {
                if (primed || done) return;
                try {
                    next = poll();
                    if (next == null) done = true;
                } catch (LakeException e) {
                    done = true;
                    throw e;
                }
                primed = true;
            }

            @Override
            public boolean hasNext() {
                prime();
                return !done && next != null;
            }

            @Override
            public DataFrame next() {
                prime();
                if (done || next == null) throw new java.util.NoSuchElementException();
                DataFrame cur = next;
                next = null;
                primed = false;
                return cur;
            }
        };
    }

    @Override
    void close();
}
