package org.bytedeco.pytorch.serving.tritonserver;

import com.google.gson.Gson;
import org.bytedeco.pytorch.serving.tritonserver.enums.TritonLogFormat;
import org.bytedeco.pytorch.serving.tritonserver.enums.ModelControlMode;
import org.bytedeco.pytorch.serving.tritonserver.enums.RateLimitMode;
import org.bytedeco.pytorch.serving.tritonserver.exceptions.TritonInvalidArgumentException;
import org.bytedeco.tritonserver.tritonserver.TRITONSERVER_ServerOptions;
import org.bytedeco.pytorch.serving.tritonserver.internal.NativeError;

import java.util.*;

import static org.bytedeco.tritonserver.global.tritonserver.*;

/**
 * Server configuration options.
 *
 * <p>Corresponds to Python {@code tritonserver.Options}. Defaults match the
 * Python dataclass unless noted. Validated when {@link TServer#start} builds
 * native options via {@link #createNativeOptions()}.
 */
public final class TritonOption {
    private static final Gson GSON = new Gson();

    private List<String> modelRepository = new ArrayList<>();
    private String serverId = "triton";
    private ModelControlMode modelControlMode = ModelControlMode.NONE;
    private List<String> startupModels = new ArrayList<>();
    private boolean strictModelConfig = false;
    private RateLimitMode rateLimiterMode = RateLimitMode.OFF;
    private List<RateLimiterResource> rateLimiterResources = new ArrayList<>();
    private long pinnedMemoryPoolSize = 1L << 28;
    private Map<Integer, Long> cudaMemoryPoolSizes = new LinkedHashMap<>();
    private Map<String, Map<String, Object>> cacheConfig = new LinkedHashMap<>();
    private String cacheDirectory = "/opt/tritonserver/caches";
    private double minSupportedComputeCapability = 6.0;
    private boolean exitOnError = true;
    private boolean strictReadiness = true;
    private int exitTimeout = 30;
    private int bufferManagerThreadCount = 0;
    private int modelLoadThreadCount = 4;
    private boolean modelNamespacing = false;
    private String logFile;
    private boolean logInfo = false;
    private boolean logWarn = false;
    private boolean logError = false;
    private TritonLogFormat tritonLogFormat = TritonLogFormat.DEFAULT;
    private int logVerbose = 0;
    private boolean metrics = true;
    private boolean gpuMetrics = true;
    private boolean cpuMetrics = true;
    private int metricsInterval = 2000;
    private String backendDirectory = "/opt/tritonserver/backends";
    private String repoAgentDirectory = "/opt/tritonserver/repoagents";
    private List<ModelLoadDeviceLimit> modelLoadDeviceLimits = new ArrayList<>();
    private Map<String, Map<String, String>> backendConfiguration = new LinkedHashMap<>();
    private Map<String, Map<String, String>> hostPolicies = new LinkedHashMap<>();
    private Map<String, Map<String, String>> metricsConfiguration = new LinkedHashMap<>();

    public TritonOption() {}

    public TritonOption(TritonOption other) {
        Objects.requireNonNull(other, "other");
        this.modelRepository = new ArrayList<>(other.modelRepository);
        this.serverId = other.serverId;
        this.modelControlMode = other.modelControlMode;
        this.startupModels = new ArrayList<>(other.startupModels);
        this.strictModelConfig = other.strictModelConfig;
        this.rateLimiterMode = other.rateLimiterMode;
        this.rateLimiterResources = new ArrayList<>(other.rateLimiterResources);
        this.pinnedMemoryPoolSize = other.pinnedMemoryPoolSize;
        this.cudaMemoryPoolSizes = new LinkedHashMap<>(other.cudaMemoryPoolSizes);
        this.cacheConfig = deepCopyMapMap(other.cacheConfig);
        this.cacheDirectory = other.cacheDirectory;
        this.minSupportedComputeCapability = other.minSupportedComputeCapability;
        this.exitOnError = other.exitOnError;
        this.strictReadiness = other.strictReadiness;
        this.exitTimeout = other.exitTimeout;
        this.bufferManagerThreadCount = other.bufferManagerThreadCount;
        this.modelLoadThreadCount = other.modelLoadThreadCount;
        this.modelNamespacing = other.modelNamespacing;
        this.logFile = other.logFile;
        this.logInfo = other.logInfo;
        this.logWarn = other.logWarn;
        this.logError = other.logError;
        this.tritonLogFormat = other.tritonLogFormat;
        this.logVerbose = other.logVerbose;
        this.metrics = other.metrics;
        this.gpuMetrics = other.gpuMetrics;
        this.cpuMetrics = other.cpuMetrics;
        this.metricsInterval = other.metricsInterval;
        this.backendDirectory = other.backendDirectory;
        this.repoAgentDirectory = other.repoAgentDirectory;
        this.modelLoadDeviceLimits = new ArrayList<>(other.modelLoadDeviceLimits);
        this.backendConfiguration = deepCopyStringMap(other.backendConfiguration);
        this.hostPolicies = deepCopyStringMap(other.hostPolicies);
        this.metricsConfiguration = deepCopyStringMap(other.metricsConfiguration);
    }

    public static Builder builder() {
        return new Builder();
    }

    // ---- getters / fluent setters ----

    public List<String> modelRepository() {
        return Collections.unmodifiableList(modelRepository);
    }

    public TritonOption modelRepository(String path) {
        this.modelRepository = new ArrayList<>();
        if (path != null) {
            this.modelRepository.add(path);
        }
        return this;
    }

    public TritonOption modelRepository(List<String> paths) {
        this.modelRepository = paths == null ? new ArrayList<>() : new ArrayList<>(paths);
        return this;
    }

    public TritonOption addModelRepository(String path) {
        if (path != null) {
            this.modelRepository.add(path);
        }
        return this;
    }

    public String serverId() {
        return serverId;
    }

    public TritonOption serverId(String serverId) {
        this.serverId = serverId;
        return this;
    }

    public ModelControlMode modelControlMode() {
        return modelControlMode;
    }

    public TritonOption modelControlMode(ModelControlMode mode) {
        this.modelControlMode = Objects.requireNonNull(mode);
        return this;
    }

    public List<String> startupModels() {
        return Collections.unmodifiableList(startupModels);
    }

    public TritonOption startupModels(List<String> models) {
        this.startupModels = models == null ? new ArrayList<>() : new ArrayList<>(models);
        return this;
    }

    public boolean strictModelConfig() {
        return strictModelConfig;
    }

    public TritonOption strictModelConfig(boolean v) {
        this.strictModelConfig = v;
        return this;
    }

    public RateLimitMode rateLimiterMode() {
        return rateLimiterMode;
    }

    public TritonOption rateLimiterMode(RateLimitMode mode) {
        this.rateLimiterMode = Objects.requireNonNull(mode);
        return this;
    }

    public List<RateLimiterResource> rateLimiterResources() {
        return Collections.unmodifiableList(rateLimiterResources);
    }

    public TritonOption rateLimiterResources(List<RateLimiterResource> resources) {
        this.rateLimiterResources = resources == null ? new ArrayList<>() : new ArrayList<>(resources);
        return this;
    }

    public long pinnedMemoryPoolSize() {
        return pinnedMemoryPoolSize;
    }

    public TritonOption pinnedMemoryPoolSize(long size) {
        this.pinnedMemoryPoolSize = size;
        return this;
    }

    public Map<Integer, Long> cudaMemoryPoolSizes() {
        return Collections.unmodifiableMap(cudaMemoryPoolSizes);
    }

    public TritonOption cudaMemoryPoolSizes(Map<Integer, Long> sizes) {
        this.cudaMemoryPoolSizes = sizes == null ? new LinkedHashMap<>() : new LinkedHashMap<>(sizes);
        return this;
    }

    public Map<String, Map<String, Object>> cacheConfig() {
        return Collections.unmodifiableMap(cacheConfig);
    }

    public TritonOption cacheConfig(Map<String, Map<String, Object>> config) {
        this.cacheConfig = config == null ? new LinkedHashMap<>() : deepCopyMapMap(config);
        return this;
    }

    public String cacheDirectory() {
        return cacheDirectory;
    }

    public TritonOption cacheDirectory(String dir) {
        this.cacheDirectory = dir;
        return this;
    }

    public double minSupportedComputeCapability() {
        return minSupportedComputeCapability;
    }

    public TritonOption minSupportedComputeCapability(double v) {
        this.minSupportedComputeCapability = v;
        return this;
    }

    public boolean exitOnError() {
        return exitOnError;
    }

    public TritonOption exitOnError(boolean v) {
        this.exitOnError = v;
        return this;
    }

    public boolean strictReadiness() {
        return strictReadiness;
    }

    public TritonOption strictReadiness(boolean v) {
        this.strictReadiness = v;
        return this;
    }

    public int exitTimeout() {
        return exitTimeout;
    }

    public TritonOption exitTimeout(int seconds) {
        this.exitTimeout = seconds;
        return this;
    }

    public int bufferManagerThreadCount() {
        return bufferManagerThreadCount;
    }

    public TritonOption bufferManagerThreadCount(int n) {
        this.bufferManagerThreadCount = n;
        return this;
    }

    public int modelLoadThreadCount() {
        return modelLoadThreadCount;
    }

    public TritonOption modelLoadThreadCount(int n) {
        this.modelLoadThreadCount = n;
        return this;
    }

    public boolean modelNamespacing() {
        return modelNamespacing;
    }

    public TritonOption modelNamespacing(boolean v) {
        this.modelNamespacing = v;
        return this;
    }

    public String logFile() {
        return logFile;
    }

    public TritonOption logFile(String path) {
        this.logFile = path;
        return this;
    }

    public boolean logInfo() {
        return logInfo;
    }

    public TritonOption logInfo(boolean v) {
        this.logInfo = v;
        return this;
    }

    public boolean logWarn() {
        return logWarn;
    }

    public TritonOption logWarn(boolean v) {
        this.logWarn = v;
        return this;
    }

    public boolean logError() {
        return logError;
    }

    public TritonOption logError(boolean v) {
        this.logError = v;
        return this;
    }

    public TritonLogFormat logFormat() {
        return tritonLogFormat;
    }

    public TritonOption logFormat(TritonLogFormat format) {
        this.tritonLogFormat = Objects.requireNonNull(format);
        return this;
    }

    public int logVerbose() {
        return logVerbose;
    }

    public TritonOption logVerbose(int level) {
        this.logVerbose = level;
        return this;
    }

    public boolean metrics() {
        return metrics;
    }

    public TritonOption metrics(boolean v) {
        this.metrics = v;
        return this;
    }

    public boolean gpuMetrics() {
        return gpuMetrics;
    }

    public TritonOption gpuMetrics(boolean v) {
        this.gpuMetrics = v;
        return this;
    }

    public boolean cpuMetrics() {
        return cpuMetrics;
    }

    public TritonOption cpuMetrics(boolean v) {
        this.cpuMetrics = v;
        return this;
    }

    public int metricsInterval() {
        return metricsInterval;
    }

    public TritonOption metricsInterval(int ms) {
        this.metricsInterval = ms;
        return this;
    }

    public String backendDirectory() {
        return backendDirectory;
    }

    public TritonOption backendDirectory(String dir) {
        this.backendDirectory = dir;
        return this;
    }

    public String repoAgentDirectory() {
        return repoAgentDirectory;
    }

    public TritonOption repoAgentDirectory(String dir) {
        this.repoAgentDirectory = dir;
        return this;
    }

    public List<ModelLoadDeviceLimit> modelLoadDeviceLimits() {
        return Collections.unmodifiableList(modelLoadDeviceLimits);
    }

    public TritonOption modelLoadDeviceLimits(List<ModelLoadDeviceLimit> limits) {
        this.modelLoadDeviceLimits = limits == null ? new ArrayList<>() : new ArrayList<>(limits);
        return this;
    }

    public Map<String, Map<String, String>> backendConfiguration() {
        return Collections.unmodifiableMap(backendConfiguration);
    }

    public TritonOption backendConfiguration(Map<String, Map<String, String>> config) {
        this.backendConfiguration = config == null ? new LinkedHashMap<>() : deepCopyStringMap(config);
        return this;
    }

    public Map<String, Map<String, String>> hostPolicies() {
        return Collections.unmodifiableMap(hostPolicies);
    }

    public TritonOption hostPolicies(Map<String, Map<String, String>> policies) {
        this.hostPolicies = policies == null ? new LinkedHashMap<>() : deepCopyStringMap(policies);
        return this;
    }

    public Map<String, Map<String, String>> metricsConfiguration() {
        return Collections.unmodifiableMap(metricsConfiguration);
    }

    public TritonOption metricsConfiguration(Map<String, Map<String, String>> config) {
        this.metricsConfiguration = config == null ? new LinkedHashMap<>() : deepCopyStringMap(config);
        return this;
    }

    /** Aligns with Python {@code Options._cascade_log_levels}. */
    void cascadeLogLevels() {
        if (logVerbose > 0) {
            logInfo = true;
        }
        if (logInfo) {
            logWarn = true;
        }
        if (logWarn) {
            logError = true;
        }
    }

    /**
     * Build native {@link TRITONSERVER_ServerOptions}.
     *
     * <p>Caller owns the returned object and must delete it (Server.start does).
     * Mirrors Python {@code Options._create_tritonserver_server_options}.
     */
    TRITONSERVER_ServerOptions createNativeOptions() {
        if (modelRepository == null || modelRepository.isEmpty()) {
            throw new TritonInvalidArgumentException("Model repository must be specified.");
        }

        TRITONSERVER_ServerOptions options = new TRITONSERVER_ServerOptions((org.bytedeco.javacpp.Pointer) null);
        NativeError.check(TRITONSERVER_ServerOptionsNew(options), "creating server options");

        try {
            NativeError.check(TRITONSERVER_ServerOptionsSetServerId(options, serverId), "setting server id");

            for (String path : modelRepository) {
                NativeError.check(
                        TRITONSERVER_ServerOptionsSetModelRepositoryPath(options, path),
                        "setting model repository path");
            }

            NativeError.check(
                    TRITONSERVER_ServerOptionsSetModelControlMode(options, modelControlMode.code()),
                    "setting model control mode");

            for (String model : startupModels) {
                NativeError.check(
                        TRITONSERVER_ServerOptionsSetStartupModel(options, model),
                        "setting startup model");
            }

            NativeError.check(
                    TRITONSERVER_ServerOptionsSetStrictModelConfig(options, strictModelConfig),
                    "setting strict model config");
            NativeError.check(
                    TRITONSERVER_ServerOptionsSetRateLimiterMode(options, rateLimiterMode.code()),
                    "setting rate limiter mode");

            for (RateLimiterResource r : rateLimiterResources) {
                NativeError.check(
                        TRITONSERVER_ServerOptionsAddRateLimiterResource(
                                options, r.name(), r.count(), r.device()),
                        "adding rate limiter resource");
            }

            NativeError.check(
                    TRITONSERVER_ServerOptionsSetPinnedMemoryPoolByteSize(options, pinnedMemoryPoolSize),
                    "setting pinned memory pool size");

            for (Map.Entry<Integer, Long> e : cudaMemoryPoolSizes.entrySet()) {
                NativeError.check(
                        TRITONSERVER_ServerOptionsSetCudaMemoryPoolByteSize(
                                options, e.getKey(), e.getValue()),
                        "setting cuda memory pool size");
            }

            for (Map.Entry<String, Map<String, Object>> e : cacheConfig.entrySet()) {
                String json = GSON.toJson(e.getValue());
                NativeError.check(
                        TRITONSERVER_ServerOptionsSetCacheConfig(options, e.getKey(), json),
                        "setting cache config");
            }

            NativeError.check(
                    TRITONSERVER_ServerOptionsSetCacheDirectory(options, cacheDirectory),
                    "setting cache directory");
            NativeError.check(
                    TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability(
                            options, minSupportedComputeCapability),
                    "setting min compute capability");
            NativeError.check(
                    TRITONSERVER_ServerOptionsSetExitOnError(options, exitOnError),
                    "setting exit on error");
            NativeError.check(
                    TRITONSERVER_ServerOptionsSetStrictReadiness(options, strictReadiness),
                    "setting strict readiness");
            NativeError.check(
                    TRITONSERVER_ServerOptionsSetExitTimeout(options, exitTimeout),
                    "setting exit timeout");
            NativeError.check(
                    TRITONSERVER_ServerOptionsSetBufferManagerThreadCount(
                            options, bufferManagerThreadCount),
                    "setting buffer manager thread count");
            NativeError.check(
                    TRITONSERVER_ServerOptionsSetModelLoadThreadCount(options, modelLoadThreadCount),
                    "setting model load thread count");
            NativeError.check(
                    TRITONSERVER_ServerOptionsSetModelNamespacing(options, modelNamespacing),
                    "setting model namespacing");

            cascadeLogLevels();
            if (logFile != null) {
                NativeError.check(
                        TRITONSERVER_ServerOptionsSetLogFile(options, logFile),
                        "setting log file");
            }
            NativeError.check(TRITONSERVER_ServerOptionsSetLogInfo(options, logInfo), "setting log info");
            NativeError.check(TRITONSERVER_ServerOptionsSetLogWarn(options, logWarn), "setting log warn");
            NativeError.check(TRITONSERVER_ServerOptionsSetLogError(options, logError), "setting log error");
            NativeError.check(
                    TRITONSERVER_ServerOptionsSetLogFormat(options, tritonLogFormat.code()),
                    "setting log format");
            NativeError.check(
                    TRITONSERVER_ServerOptionsSetLogVerbose(options, logVerbose),
                    "setting log verbose");

            NativeError.check(TRITONSERVER_ServerOptionsSetMetrics(options, metrics), "setting metrics");
            NativeError.check(
                    TRITONSERVER_ServerOptionsSetCpuMetrics(options, cpuMetrics), "setting cpu metrics");
            NativeError.check(
                    TRITONSERVER_ServerOptionsSetGpuMetrics(options, gpuMetrics), "setting gpu metrics");
            NativeError.check(
                    TRITONSERVER_ServerOptionsSetMetricsInterval(options, metricsInterval),
                    "setting metrics interval");
            NativeError.check(
                    TRITONSERVER_ServerOptionsSetBackendDirectory(options, backendDirectory),
                    "setting backend directory");
            NativeError.check(
                    TRITONSERVER_ServerOptionsSetRepoAgentDirectory(options, repoAgentDirectory),
                    "setting repo agent directory");

            for (ModelLoadDeviceLimit limit : modelLoadDeviceLimits) {
                NativeError.check(
                        TRITONSERVER_ServerOptionsSetModelLoadDeviceLimit(
                                options, limit.kind().code(), limit.device(), limit.fraction()),
                        "setting model load device limit");
            }

            for (Map.Entry<String, Map<String, String>> policy : hostPolicies.entrySet()) {
                for (Map.Entry<String, String> setting : policy.getValue().entrySet()) {
                    NativeError.check(
                            TRITONSERVER_ServerOptionsSetHostPolicy(
                                    options, policy.getKey(), setting.getKey(), setting.getValue()),
                            "setting host policy");
                }
            }

            for (Map.Entry<String, Map<String, String>> cfg : metricsConfiguration.entrySet()) {
                for (Map.Entry<String, String> setting : cfg.getValue().entrySet()) {
                    NativeError.check(
                            TRITONSERVER_ServerOptionsSetMetricsConfig(
                                    options, cfg.getKey(), setting.getKey(), setting.getValue()),
                            "setting metrics config");
                }
            }

            for (Map.Entry<String, Map<String, String>> backend : backendConfiguration.entrySet()) {
                for (Map.Entry<String, String> setting : backend.getValue().entrySet()) {
                    NativeError.check(
                            TRITONSERVER_ServerOptionsSetBackendConfig(
                                    options, backend.getKey(), setting.getKey(), setting.getValue()),
                            "setting backend config");
                }
            }

            return options;
        } catch (RuntimeException ex) {
            TRITONSERVER_ServerOptionsDelete(options);
            throw ex;
        }
    }

    private static Map<String, Map<String, Object>> deepCopyMapMap(Map<String, Map<String, Object>> src) {
        Map<String, Map<String, Object>> out = new LinkedHashMap<>();
        for (Map.Entry<String, Map<String, Object>> e : src.entrySet()) {
            out.put(e.getKey(), e.getValue() == null ? null : new LinkedHashMap<>(e.getValue()));
        }
        return out;
    }

    private static Map<String, Map<String, String>> deepCopyStringMap(Map<String, Map<String, String>> src) {
        Map<String, Map<String, String>> out = new LinkedHashMap<>();
        for (Map.Entry<String, Map<String, String>> e : src.entrySet()) {
            out.put(e.getKey(), e.getValue() == null ? null : new LinkedHashMap<>(e.getValue()));
        }
        return out;
    }

    public static final class Builder {
        private final TritonOption tritonOptions = new TritonOption();

        public Builder modelRepository(String path) {
            tritonOptions.modelRepository(path);
            return this;
        }

        public Builder modelRepository(List<String> paths) {
            tritonOptions.modelRepository(paths);
            return this;
        }

        public Builder serverId(String id) {
            tritonOptions.serverId(id);
            return this;
        }

        public Builder modelControlMode(ModelControlMode mode) {
            tritonOptions.modelControlMode(mode);
            return this;
        }

        public Builder startupModels(List<String> models) {
            tritonOptions.startupModels(models);
            return this;
        }

        public Builder strictModelConfig(boolean v) {
            tritonOptions.strictModelConfig(v);
            return this;
        }

        public Builder logVerbose(int level) {
            tritonOptions.logVerbose(level);
            return this;
        }

        public Builder logInfo(boolean v) {
            tritonOptions.logInfo(v);
            return this;
        }

        public Builder backendDirectory(String dir) {
            tritonOptions.backendDirectory(dir);
            return this;
        }

        public Builder repoAgentDirectory(String dir) {
            tritonOptions.repoAgentDirectory(dir);
            return this;
        }

        public Builder metrics(boolean v) {
            tritonOptions.metrics(v);
            return this;
        }

        public Builder exitTimeout(int seconds) {
            tritonOptions.exitTimeout(seconds);
            return this;
        }

        public TritonOption build() {
            return new TritonOption(tritonOptions);
        }
    }
}
