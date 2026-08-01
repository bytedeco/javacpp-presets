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

package org.bytedeco.pytorch.llm.llamacpp;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Runtime configuration for in-process and process-server backends.
 * Field names intentionally mirror llama.cpp CLI / server flags where practical.
 */
public final class LlamaRuntimeConfig {
    private final Path modelPath;
    private final LlamaBackend backend;
    private final int nCtx;
    private final int nBatch;
    private final int nUbBatch;
    private final int nThreads;
    private final int nGpuLayers;
    private final boolean offloadKqv;
    private final boolean offloadMoeExperts;
    private final List<Integer> tensorSplit;
    private final int mainGpu;
    private final boolean flashAttn;
    private final String cacheTypeK;
    private final String cacheTypeV;
    private final Path llamaServerBin;
    private final Path llamaCliBin;
    private final String serverHost;
    private final int serverPort;
    private final long serverStartTimeoutMs;
    private final String chatTemplate;
    private final boolean verbose;
    private final Map<String, String> extraServerArgs;
    private final boolean useMmap;
    private final boolean useMlock;

    private LlamaRuntimeConfig(Builder b) {
        this.modelPath = Objects.requireNonNull(b.modelPath, "modelPath");
        this.backend = b.backend != null ? b.backend : LlamaBackend.AUTO;
        this.nCtx = Math.max(8, b.nCtx);
        this.nBatch = Math.max(1, b.nBatch);
        this.nUbBatch = Math.max(1, b.nUbBatch);
        this.nThreads = b.nThreads > 0 ? b.nThreads : Math.max(1, Runtime.getRuntime().availableProcessors() / 2);
        this.nGpuLayers = b.nGpuLayers;
        this.offloadKqv = b.offloadKqv;
        this.offloadMoeExperts = b.offloadMoeExperts;
        this.tensorSplit = List.copyOf(b.tensorSplit);
        this.mainGpu = Math.max(0, b.mainGpu);
        this.flashAttn = b.flashAttn;
        this.cacheTypeK = b.cacheTypeK;
        this.cacheTypeV = b.cacheTypeV;
        this.llamaServerBin = b.llamaServerBin;
        this.llamaCliBin = b.llamaCliBin;
        this.serverHost = b.serverHost != null ? b.serverHost : "127.0.0.1";
        this.serverPort = b.serverPort;
        this.serverStartTimeoutMs = Math.max(1000L, b.serverStartTimeoutMs);
        this.chatTemplate = b.chatTemplate;
        this.verbose = b.verbose;
        this.extraServerArgs = Map.copyOf(b.extraServerArgs);
        this.useMmap = b.useMmap;
        this.useMlock = b.useMlock;
    }

    public static Builder builder() { return new Builder(); }

    public Path modelPath() { return modelPath; }
    public LlamaBackend backend() { return backend; }
    public int nCtx() { return nCtx; }
    public int nBatch() { return nBatch; }
    public int nUbBatch() { return nUbBatch; }
    public int nThreads() { return nThreads; }
    public int nGpuLayers() { return nGpuLayers; }
    public boolean offloadKqv() { return offloadKqv; }
    public boolean offloadMoeExperts() { return offloadMoeExperts; }
    public List<Integer> tensorSplit() { return tensorSplit; }
    public int mainGpu() { return mainGpu; }
    public boolean flashAttn() { return flashAttn; }
    public Optional<String> cacheTypeK() { return Optional.ofNullable(cacheTypeK); }
    public Optional<String> cacheTypeV() { return Optional.ofNullable(cacheTypeV); }
    public Optional<Path> llamaServerBin() { return Optional.ofNullable(llamaServerBin); }
    public Optional<Path> llamaCliBin() { return Optional.ofNullable(llamaCliBin); }
    public String serverHost() { return serverHost; }
    public int serverPort() { return serverPort; }
    public long serverStartTimeoutMs() { return serverStartTimeoutMs; }
    public Optional<String> chatTemplate() { return Optional.ofNullable(chatTemplate); }
    public boolean verbose() { return verbose; }
    public Map<String, String> extraServerArgs() { return extraServerArgs; }
    public boolean useMmap() { return useMmap; }
    public boolean useMlock() { return useMlock; }

    /** CLI args for {@code llama-server} (without binary path). */
    public List<String> toServerArgList(int boundPort) {
        List<String> args = new ArrayList<>();
        args.add("-m"); args.add(modelPath.toAbsolutePath().toString());
        args.add("--host"); args.add(serverHost);
        args.add("--port"); args.add(String.valueOf(boundPort > 0 ? boundPort : Math.max(0, serverPort)));
        args.add("-c"); args.add(String.valueOf(nCtx));
        args.add("-b"); args.add(String.valueOf(nBatch));
        args.add("-t"); args.add(String.valueOf(nThreads));
        if (nGpuLayers != 0) {
            args.add("-ngl"); args.add(String.valueOf(nGpuLayers));
        }
        if (flashAttn) args.add("-fa");
        if (offloadMoeExperts) {
            args.add("-ot"); args.add("exps=CPU"); // common pattern; host may override
        }
        if (!tensorSplit.isEmpty()) {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < tensorSplit.size(); i++) {
                if (i > 0) sb.append(',');
                sb.append(tensorSplit.get(i));
            }
            args.add("-ts"); args.add(sb.toString());
        }
        if (mainGpu > 0) {
            args.add("-mg"); args.add(String.valueOf(mainGpu));
        }
        if (cacheTypeK != null && !cacheTypeK.isBlank()) {
            args.add("-ctk"); args.add(cacheTypeK);
        }
        if (cacheTypeV != null && !cacheTypeV.isBlank()) {
            args.add("-ctv"); args.add(cacheTypeV);
        }
        if (!useMmap) args.add("--no-mmap");
        if (useMlock) args.add("--mlock");
        if (chatTemplate != null && !chatTemplate.isBlank()) {
            args.add("--chat-template"); args.add(chatTemplate);
        }
        for (Map.Entry<String, String> e : extraServerArgs.entrySet()) {
            args.add(e.getKey());
            if (e.getValue() != null && !e.getValue().isEmpty()) args.add(e.getValue());
        }
        return args;
    }

    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("model_path", modelPath.toString());
        m.put("backend", backend.name());
        m.put("n_ctx", nCtx);
        m.put("n_batch", nBatch);
        m.put("n_threads", nThreads);
        m.put("n_gpu_layers", nGpuLayers);
        m.put("offload_moe_experts", offloadMoeExperts);
        m.put("server_host", serverHost);
        m.put("server_port", serverPort);
        m.put("flash_attn", flashAttn);
        return m;
    }

    public static final class Builder {
        private Path modelPath;
        private LlamaBackend backend = LlamaBackend.AUTO;
        private int nCtx = 2048;
        private int nBatch = 512;
        private int nUbBatch = 512;
        private int nThreads = 0;
        private int nGpuLayers = 0;
        private boolean offloadKqv = true;
        private boolean offloadMoeExperts = false;
        private List<Integer> tensorSplit = List.of();
        private int mainGpu = 0;
        private boolean flashAttn = false;
        private String cacheTypeK;
        private String cacheTypeV;
        private Path llamaServerBin;
        private Path llamaCliBin;
        private String serverHost = "127.0.0.1";
        private int serverPort = 8080;
        private long serverStartTimeoutMs = 60_000L;
        private String chatTemplate;
        private boolean verbose = false;
        private Map<String, String> extraServerArgs = Map.of();
        private boolean useMmap = true;
        private boolean useMlock = false;

        public Builder modelPath(Path v) { this.modelPath = v; return this; }
        public Builder modelPath(String v) { this.modelPath = Path.of(v); return this; }
        public Builder backend(LlamaBackend v) { this.backend = v; return this; }
        public Builder nCtx(int v) { this.nCtx = v; return this; }
        public Builder nBatch(int v) { this.nBatch = v; return this; }
        public Builder nUbBatch(int v) { this.nUbBatch = v; return this; }
        public Builder nThreads(int v) { this.nThreads = v; return this; }
        public Builder nGpuLayers(int v) { this.nGpuLayers = v; return this; }
        public Builder offloadKqv(boolean v) { this.offloadKqv = v; return this; }
        public Builder offloadMoeExperts(boolean v) { this.offloadMoeExperts = v; return this; }
        public Builder tensorSplit(List<Integer> v) { this.tensorSplit = v != null ? new ArrayList<>(v) : List.of(); return this; }
        public Builder mainGpu(int v) { this.mainGpu = v; return this; }
        public Builder flashAttn(boolean v) { this.flashAttn = v; return this; }
        public Builder cacheTypeK(String v) { this.cacheTypeK = v; return this; }
        public Builder cacheTypeV(String v) { this.cacheTypeV = v; return this; }
        public Builder cacheTypeKv(String v) { this.cacheTypeK = v; this.cacheTypeV = v; return this; }
        public Builder llamaServerBin(Path v) { this.llamaServerBin = v; return this; }
        public Builder llamaCliBin(Path v) { this.llamaCliBin = v; return this; }
        public Builder serverHost(String v) { this.serverHost = v; return this; }
        public Builder serverPort(int v) { this.serverPort = v; return this; }
        public Builder serverStartTimeoutMs(long v) { this.serverStartTimeoutMs = v; return this; }
        public Builder chatTemplate(String v) { this.chatTemplate = v; return this; }
        public Builder verbose(boolean v) { this.verbose = v; return this; }
        public Builder extraServerArgs(Map<String, String> v) { this.extraServerArgs = v != null ? v : Map.of(); return this; }
        public Builder useMmap(boolean v) { this.useMmap = v; return this; }
        public Builder useMlock(boolean v) { this.useMlock = v; return this; }

        /** Apply Studio GGUF hardware controls if present on classpath. */
        public Builder fromStudioHardware(Object controls) {
            if (controls == null) return this;
            try {
                Object ngl = controls.getClass().getMethod("nGpuLayers").invoke(controls);
                if (ngl instanceof Number n) this.nGpuLayers = n.intValue();
                Object moe = controls.getClass().getMethod("offloadMoeExperts").invoke(controls);
                if (moe instanceof Boolean b) this.offloadMoeExperts = b;
                Object gpus = controls.getClass().getMethod("gpuIds").invoke(controls);
                if (gpus instanceof List<?> list) {
                    List<Integer> ids = new ArrayList<>();
                    for (Object o : list) if (o instanceof Number n) ids.add(n.intValue());
                    this.tensorSplit = ids;
                }
                Object fa = controls.getClass().getMethod("flashAttn").invoke(controls);
                if (fa instanceof Boolean b) this.flashAttn = b;
                Object th = controls.getClass().getMethod("threads").invoke(controls);
                if (th instanceof Number n && n.intValue() > 0) this.nThreads = n.intValue();
                try {
                    Object ck = controls.getClass().getMethod("cacheTypeK").invoke(controls);
                    if (ck instanceof java.util.Optional<?> opt && opt.isPresent()) {
                        this.cacheTypeK = String.valueOf(opt.get());
                    }
                } catch (NoSuchMethodException ignored) {}
            } catch (ReflectiveOperationException ignored) {}
            return this;
        }

        public LlamaRuntimeConfig build() { return new LlamaRuntimeConfig(this); }
    }
}
