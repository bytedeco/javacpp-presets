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
package org.bytedeco.pytorch.llm.ktransformers.inference;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Multi-concurrency generate runtime (upstream "Support Multi-concurrency").
 *
 * <p>Serializes model forward with a fair lock while allowing up to
 * {@code concurrency} in-flight requests. Mini models share one Module; real
 * multi-replica serving would swap the pipeline factory.
 */
public final class MultiConcurrencyRuntime implements AutoCloseable {

    private final PrefillDecodePipeline pipeline;
    private final int concurrency;
    private final Semaphore slots;
    private final Object modelLock = new Object();
    private final ExecutorService executor;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    public MultiConcurrencyRuntime(PrefillDecodePipeline pipeline, int concurrency) {
        this.pipeline = Objects.requireNonNull(pipeline, "pipeline");
        this.concurrency = Math.max(1, concurrency);
        this.slots = new Semaphore(this.concurrency, true);
        this.executor = Executors.newFixedThreadPool(this.concurrency, r -> {
            Thread t = new Thread(r, "kt-infer-" + System.identityHashCode(this));
            t.setDaemon(true);
            return t;
        });
    }

    public int concurrency() { return concurrency; }

    public KtGenerateOutput generate(KtGenerateRequest req) throws InterruptedException {
        ensureOpen();
        slots.acquire();
        try {
            synchronized (modelLock) {
                return pipeline.generate(req);
            }
        } finally {
            slots.release();
        }
    }

    public Future<KtGenerateOutput> submit(KtGenerateRequest req) {
        ensureOpen();
        return executor.submit(() -> generate(req));
    }

    public List<KtGenerateOutput> generateAll(List<KtGenerateRequest> reqs, long timeoutMs)
            throws InterruptedException, ExecutionException, TimeoutException {
        Objects.requireNonNull(reqs, "reqs");
        List<Future<KtGenerateOutput>> futures = new ArrayList<>(reqs.size());
        for (KtGenerateRequest r : reqs) {
            futures.add(submit(r));
        }
        List<KtGenerateOutput> out = new ArrayList<>(reqs.size());
        for (Future<KtGenerateOutput> f : futures) {
            out.add(timeoutMs > 0 ? f.get(timeoutMs, TimeUnit.MILLISECONDS) : f.get());
        }
        return out;
    }

    private void ensureOpen() {
        if (closed.get()) {
            throw new IllegalStateException("MultiConcurrencyRuntime closed");
        }
    }

    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) return;
        executor.shutdownNow();
        try {
            executor.awaitTermination(5, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}
