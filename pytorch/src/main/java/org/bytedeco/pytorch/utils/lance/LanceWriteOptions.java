package org.bytedeco.pytorch.utils.lance;

import org.lance.WriteParams;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Options for writing a DataFrame as an official Lance dataset
 * ({@code org.lance:lance-core}).
 *
 * <pre>{@code
 * df.writeLance("clips.lance", LanceWriteOptions.overwrite()
 *     .maxRowsPerFile(1_000_000)
 *     .stableRowIds(true));
 * }</pre>
 */
public final class LanceWriteOptions {

    private WriteParams.WriteMode mode = WriteParams.WriteMode.CREATE;
    private Integer maxRowsPerFile;
    private Integer maxRowsPerGroup;
    private Long maxBytesPerFile;
    private Boolean stableRowIds;
    private Boolean enableV2ManifestPaths;
    private String dataStorageVersion;
    private Map<String, String> storageOptions = new LinkedHashMap<>();

    public static LanceWriteOptions defaults() {
        return new LanceWriteOptions();
    }

    public static LanceWriteOptions create() {
        return new LanceWriteOptions().mode(WriteParams.WriteMode.CREATE);
    }

    public static LanceWriteOptions overwrite() {
        return new LanceWriteOptions().mode(WriteParams.WriteMode.OVERWRITE);
    }

    public static LanceWriteOptions append() {
        return new LanceWriteOptions().mode(WriteParams.WriteMode.APPEND);
    }

    public LanceWriteOptions mode(WriteParams.WriteMode mode) {
        this.mode = mode == null ? WriteParams.WriteMode.CREATE : mode;
        return this;
    }

    public LanceWriteOptions mode(String mode) {
        if (mode == null || mode.isBlank()) return this;
        String m = mode.trim().toUpperCase();
        return switch (m) {
            case "APPEND" -> mode(WriteParams.WriteMode.APPEND);
            case "OVERWRITE" -> mode(WriteParams.WriteMode.OVERWRITE);
            default -> mode(WriteParams.WriteMode.CREATE);
        };
    }

    public LanceWriteOptions maxRowsPerFile(int v) {
        this.maxRowsPerFile = v;
        return this;
    }

    public LanceWriteOptions maxRowsPerGroup(int v) {
        this.maxRowsPerGroup = v;
        return this;
    }

    public LanceWriteOptions maxBytesPerFile(long v) {
        this.maxBytesPerFile = v;
        return this;
    }

    public LanceWriteOptions stableRowIds(boolean v) {
        this.stableRowIds = v;
        return this;
    }

    public LanceWriteOptions enableV2ManifestPaths(boolean v) {
        this.enableV2ManifestPaths = v;
        return this;
    }

    public LanceWriteOptions dataStorageVersion(String v) {
        this.dataStorageVersion = v;
        return this;
    }

    public LanceWriteOptions storageOption(String key, String value) {
        Objects.requireNonNull(key, "key");
        if (value == null) this.storageOptions.remove(key);
        else this.storageOptions.put(key, value);
        return this;
    }

    public LanceWriteOptions storageOptions(Map<String, String> opts) {
        this.storageOptions = opts == null
            ? new LinkedHashMap<>()
            : new LinkedHashMap<>(opts);
        return this;
    }

    public WriteParams.WriteMode mode() { return mode; }
    public Integer maxRowsPerFile() { return maxRowsPerFile; }
    public Integer maxRowsPerGroup() { return maxRowsPerGroup; }
    public Long maxBytesPerFile() { return maxBytesPerFile; }
    public Boolean stableRowIds() { return stableRowIds; }
    public Boolean enableV2ManifestPaths() { return enableV2ManifestPaths; }
    public String dataStorageVersion() { return dataStorageVersion; }
    public Map<String, String> storageOptions() {
        return Collections.unmodifiableMap(storageOptions);
    }

    /** Build native {@link WriteParams}. */
    public WriteParams toWriteParams() {
        WriteParams.Builder b = new WriteParams.Builder().withMode(mode);
        if (maxRowsPerFile != null) b.withMaxRowsPerFile(maxRowsPerFile);
        if (maxRowsPerGroup != null) b.withMaxRowsPerGroup(maxRowsPerGroup);
        if (maxBytesPerFile != null) b.withMaxBytesPerFile(maxBytesPerFile);
        if (stableRowIds != null) b.withEnableStableRowIds(stableRowIds);
        if (enableV2ManifestPaths != null) b.withEnableV2ManifestPaths(enableV2ManifestPaths);
        if (dataStorageVersion != null && !dataStorageVersion.isBlank()) {
            b.withDataStorageVersion(dataStorageVersion);
        }
        if (!storageOptions.isEmpty()) {
            b.withStorageOptions(new LinkedHashMap<>(storageOptions));
        }
        return b.build();
    }
}
