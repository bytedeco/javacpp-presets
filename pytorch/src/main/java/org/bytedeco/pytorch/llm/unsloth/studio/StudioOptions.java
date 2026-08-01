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
package org.bytedeco.pytorch.llm.unsloth.studio;

import java.nio.file.Path;
import java.util.Objects;
import java.util.Optional;

/**
 * Immutable global options for {@link UnslothStudio}.
 */
public final class StudioOptions {

    private final Path dataRoot;
    private final Path cacheRoot;
    private final boolean enableApi;
    private final int apiPort;
    private final String apiBindHost;
    private final boolean enableBoard;
    private final int boardPort;
    private final boolean enableMcp;
    private final boolean enableAuth;
    private final String hfToken;
    private final String defaultDevice;
    private final boolean allowCodeExecution;
    private final boolean allowRemoteCode;
    private final int maxParallelGenerations;
    private final boolean tensorBoardSink;
    private final Path tensorBoardLogDir;

    private StudioOptions(Builder b) {
        this.dataRoot = Objects.requireNonNull(b.dataRoot, "dataRoot");
        this.cacheRoot = b.cacheRoot != null ? b.cacheRoot : b.dataRoot.resolve("cache");
        this.enableApi = b.enableApi;
        this.apiPort = b.apiPort;
        this.apiBindHost = b.apiBindHost != null ? b.apiBindHost : "127.0.0.1";
        this.enableBoard = b.enableBoard;
        this.boardPort = b.boardPort;
        this.enableMcp = b.enableMcp;
        this.enableAuth = b.enableAuth;
        this.hfToken = b.hfToken;
        this.defaultDevice = b.defaultDevice != null ? b.defaultDevice : "auto";
        this.allowCodeExecution = b.allowCodeExecution;
        this.allowRemoteCode = b.allowRemoteCode;
        this.maxParallelGenerations = Math.max(1, b.maxParallelGenerations);
        this.tensorBoardSink = b.tensorBoardSink;
        this.tensorBoardLogDir = b.tensorBoardLogDir != null
                ? b.tensorBoardLogDir
                : this.dataRoot.resolve("tb");
    }

    public static Builder builder() {
        return new Builder();
    }

    public static StudioOptions defaults() {
        return builder().build();
    }

    public Path dataRoot() { return dataRoot; }
    public Path cacheRoot() { return cacheRoot; }
    public boolean enableApi() { return enableApi; }
    public int apiPort() { return apiPort; }
    public String apiBindHost() { return apiBindHost; }
    public boolean enableBoard() { return enableBoard; }
    public int boardPort() { return boardPort; }
    public boolean enableMcp() { return enableMcp; }
    public boolean enableAuth() { return enableAuth; }
    public Optional<String> hfToken() { return Optional.ofNullable(hfToken); }
    public String defaultDevice() { return defaultDevice; }
    public boolean allowCodeExecution() { return allowCodeExecution; }
    public boolean allowRemoteCode() { return allowRemoteCode; }
    public int maxParallelGenerations() { return maxParallelGenerations; }
    public boolean tensorBoardSink() { return tensorBoardSink; }
    public Path tensorBoardLogDir() { return tensorBoardLogDir; }

    public Path runsDir() { return dataRoot.resolve("runs"); }
    public Path modelsDir() { return dataRoot.resolve("models"); }
    public Path datasetsDir() { return dataRoot.resolve("datasets"); }
    public Path exportsDir() { return dataRoot.resolve("exports"); }
    public Path recipesDir() { return dataRoot.resolve("recipes"); }

    public Builder toBuilder() {
        return builder()
                .dataRoot(dataRoot)
                .cacheRoot(cacheRoot)
                .enableApi(enableApi)
                .apiPort(apiPort)
                .apiBindHost(apiBindHost)
                .enableBoard(enableBoard)
                .boardPort(boardPort)
                .enableMcp(enableMcp)
                .enableAuth(enableAuth)
                .hfToken(hfToken)
                .defaultDevice(defaultDevice)
                .allowCodeExecution(allowCodeExecution)
                .allowRemoteCode(allowRemoteCode)
                .maxParallelGenerations(maxParallelGenerations)
                .tensorBoardSink(tensorBoardSink)
                .tensorBoardLogDir(tensorBoardLogDir);
    }

    public static final class Builder {
        private Path dataRoot = Path.of("studio-data");
        private Path cacheRoot;
        private boolean enableApi = false;
        private int apiPort = 8000;
        private String apiBindHost = "127.0.0.1";
        private boolean enableBoard = false;
        private int boardPort = 8001;
        private boolean enableMcp = false;
        private boolean enableAuth = false;
        private String hfToken;
        private String defaultDevice = "auto";
        private boolean allowCodeExecution = false;
        private boolean allowRemoteCode = false;
        private int maxParallelGenerations = 4;
        private boolean tensorBoardSink = false;
        private Path tensorBoardLogDir;

        public Builder dataRoot(Path dataRoot) { this.dataRoot = dataRoot; return this; }
        public Builder cacheRoot(Path cacheRoot) { this.cacheRoot = cacheRoot; return this; }
        public Builder enableApi(boolean enableApi) { this.enableApi = enableApi; return this; }
        public Builder apiPort(int apiPort) { this.apiPort = apiPort; return this; }
        public Builder apiBindHost(String apiBindHost) { this.apiBindHost = apiBindHost; return this; }
        public Builder enableBoard(boolean enableBoard) { this.enableBoard = enableBoard; return this; }
        public Builder boardPort(int boardPort) { this.boardPort = boardPort; return this; }
        public Builder enableMcp(boolean enableMcp) { this.enableMcp = enableMcp; return this; }
        public Builder enableAuth(boolean enableAuth) { this.enableAuth = enableAuth; return this; }
        public Builder hfToken(String hfToken) { this.hfToken = hfToken; return this; }
        public Builder defaultDevice(String defaultDevice) { this.defaultDevice = defaultDevice; return this; }
        public Builder allowCodeExecution(boolean v) { this.allowCodeExecution = v; return this; }
        public Builder allowRemoteCode(boolean v) { this.allowRemoteCode = v; return this; }
        public Builder maxParallelGenerations(int n) { this.maxParallelGenerations = n; return this; }
        public Builder tensorBoardSink(boolean v) { this.tensorBoardSink = v; return this; }
        public Builder tensorBoardLogDir(Path p) { this.tensorBoardLogDir = p; return this; }

        public StudioOptions build() {
            return new StudioOptions(this);
        }
    }
}
