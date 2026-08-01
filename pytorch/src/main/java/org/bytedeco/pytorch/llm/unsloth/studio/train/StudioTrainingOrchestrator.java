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

package org.bytedeco.pytorch.llm.unsloth.studio.train;

import org.bytedeco.pytorch.llm.unsloth.studio.StudioOptions;
import org.bytedeco.pytorch.llm.unsloth.studio.model.TrainingProgressEvent;
import org.bytedeco.pytorch.llm.unsloth.studio.model.TrainingRunRecord;
import org.bytedeco.pytorch.llm.unsloth.studio.model.TrainingStartRequest;
import org.bytedeco.pytorch.llm.unsloth.studio.model.TrainingType;
import org.bytedeco.pytorch.llm.unsloth.studio.observe.MetricsSink;
import org.bytedeco.pytorch.llm.unsloth.studio.observe.TrainingMetrics;
import org.bytedeco.pytorch.llm.unsloth.studio.util.IdGen;
import org.bytedeco.pytorch.llm.unsloth.studio.util.StudioPaths;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;

/**
 * Async training job manager: start / stop / await / resume metadata / progress.
 */
public final class StudioTrainingOrchestrator implements AutoCloseable {

    private final StudioOptions options;
    private final TrainingRunStore store;
    private final TrainingProgressBus bus;
    private final LoraQloraTrainer loraTrainer;
    private final RlTrainingFacade rlFacade;
    private final ExecutorService executor;
    private final Map<String, AtomicBoolean> stopFlags = new ConcurrentHashMap<>();
    private final Map<String, Future<?>> futures = new ConcurrentHashMap<>();
    private final List<MetricsSink> sinks;

    public StudioTrainingOrchestrator(StudioOptions options, TrainingRunStore store,
                                      TrainingProgressBus bus, List<MetricsSink> sinks) {
        this.options = Objects.requireNonNull(options);
        this.store = Objects.requireNonNull(store);
        this.bus = Objects.requireNonNull(bus);
        this.loraTrainer = new LoraQloraTrainer(bus);
        this.rlFacade = new RlTrainingFacade(bus);
        this.executor = Executors.newCachedThreadPool(r -> {
            Thread t = new Thread(r, "studio-train");
            t.setDaemon(true);
            return t;
        });
        this.sinks = sinks != null ? List.copyOf(sinks) : List.of();
        bus.subscribeAll(this::onProgress);
    }

    public TrainingProgressBus bus() { return bus; }
    public TrainingRunStore store() { return store; }

    public String start(TrainingStartRequest request) {
        String runId = IdGen.runId();
        Path out = request.outputDir()
                .map(Path::of)
                .orElse(StudioPaths.runDir(options, runId));
        try { StudioPaths.mkdirs(out); } catch (Exception ignored) {}

        TrainingRunRecord rec = TrainingRunRecord.builder()
                .runId(runId)
                .projectName(request.projectName().orElse(null))
                .request(request)
                .status(TrainingRunRecord.Status.QUEUED)
                .outputDir(out)
                .build();
        store.put(rec);
        AtomicBoolean stop = new AtomicBoolean(false);
        stopFlags.put(runId, stop);

        Future<?> fut = executor.submit(() -> {
            store.update(rec.toBuilder().status(TrainingRunRecord.Status.RUNNING).build());
            try {
                LoraQloraTrainer.Result result;
                if (request.trainingType() == TrainingType.REINFORCEMENT_LEARNING) {
                    result = rlFacade.train(runId, request, out, stop::get);
                } else {
                    result = loraTrainer.train(runId, request, out, stop::get);
                }
                TrainingRunRecord.Status st = stop.get()
                        ? TrainingRunRecord.Status.CANCELLED
                        : TrainingRunRecord.Status.COMPLETED;
                store.update(rec.toBuilder()
                        .status(st)
                        .globalStep(result.steps)
                        .lastLoss(result.lastLoss)
                        .finishedAtMs(System.currentTimeMillis())
                        .lastMetrics(Map.of(
                                "loss", result.lastLoss,
                                "trainable_params", (double) result.trainableParams,
                                "total_params", (double) result.totalParams))
                        .build());
            } catch (Throwable t) {
                bus.publish(TrainingProgressEvent.builder()
                        .runId(runId)
                        .phase(TrainingProgressEvent.Phase.FAILED)
                        .message(t.getMessage())
                        .build());
                store.update(rec.toBuilder()
                        .status(TrainingRunRecord.Status.FAILED)
                        .error(String.valueOf(t.getMessage()))
                        .finishedAtMs(System.currentTimeMillis())
                        .build());
            } finally {
                futures.remove(runId);
                stopFlags.remove(runId);
            }
        });
        futures.put(runId, fut);
        return runId;
    }

    public void stop(String runId) {
        AtomicBoolean f = stopFlags.get(runId);
        if (f != null) f.set(true);
        store.get(runId).ifPresent(r ->
                store.update(r.toBuilder().status(TrainingRunRecord.Status.CANCELLED).build()));
    }

    public void await(String runId) throws Exception {
        Future<?> f = futures.get(runId);
        if (f != null) f.get();
    }

    public void await(String runId, long timeoutMs) throws Exception {
        Future<?> f = futures.get(runId);
        if (f != null) f.get(timeoutMs, java.util.concurrent.TimeUnit.MILLISECONDS);
    }

    public Optional<TrainingRunRecord> run(String runId) {
        return store.get(runId);
    }

    public List<TrainingRunRecord> list() {
        return store.list();
    }

    public void onProgress(String runId, Consumer<TrainingProgressEvent> listener) {
        bus.subscribe(runId, listener::accept);
    }

    public String resume(String runId) {
        TrainingRunRecord prev = store.get(runId).orElseThrow(() ->
                new IllegalArgumentException("Unknown run: " + runId));
        if (prev.request() == null) {
            throw new IllegalStateException("Cannot resume run without request metadata: " + runId);
        }
        // New run id, same request, continue from stored step in message
        TrainingStartRequest req = prev.request();
        int remaining = Math.max(1, req.maxSteps() - prev.globalStep());
        TrainingStartRequest cont = TrainingStartRequest.builder()
                .modelName(req.modelName())
                .projectName(req.projectName().orElse(null))
                .trainingType(req.trainingType())
                .loadIn4bit(req.loadIn4bit())
                .loadIn8bit(req.loadIn8bit())
                .maxSeqLength(req.maxSeqLength())
                .loraR(req.loraR())
                .loraAlpha(req.loraAlpha())
                .loraDropout(req.loraDropout())
                .targetModules(req.targetModules())
                .learningRate(req.learningRate())
                .batchSize(req.batchSize())
                .gradientAccumulationSteps(req.gradientAccumulationSteps())
                .maxSteps(remaining)
                .dataset(req.dataset().orElse(null))
                .datasetPath(req.datasetPath().orElse(null))
                .seed(req.seed())
                .gradientCheckpointing(req.gradientCheckpointing())
                .outputDir(prev.outputDir() != null ? prev.outputDir().toString() : null)
                .build();
        return start(cont);
    }

    private void onProgress(TrainingProgressEvent ev) {
        TrainingMetrics metrics = TrainingMetrics.from(ev);
        for (MetricsSink sink : sinks) {
            try { sink.record(metrics); } catch (Throwable ignored) {}
        }
    }

    @Override
    public void close() {
        for (AtomicBoolean f : stopFlags.values()) f.set(true);
        executor.shutdownNow();
    }
}
