package org.bytedeco.pytorch.utils.minio;

import io.minio.DisableObjectLegalHoldArgs;
import io.minio.EnableObjectLegalHoldArgs;
import io.minio.GetObjectRetentionArgs;
import io.minio.IsObjectLegalHoldEnabledArgs;
import io.minio.ListObjectsArgs;
import io.minio.ObjectWriteResponse;
import io.minio.Result;
import io.minio.SetObjectRetentionArgs;
import io.minio.StatObjectArgs;
import io.minio.StatObjectResponse;
import io.minio.messages.Item;
import io.minio.messages.Retention;
import io.minio.messages.RetentionMode;

import java.time.ZonedDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * Object versioning, retention, and legal-hold helpers.
 *
 * <pre>{@code
 * MinioVersion v = m.versionApi();
 * ObjectWriteResponse w = v.putVersion("b", "k", data, opts);
 * List<VersionInfo> versions = v.listVersions("b", "k");
 * v.setRetention("b", "k", RetentionMode.GOVERNANCE, ZonedDateTime.now().plusDays(30));
 * v.enableLegalHold("b", "k");
 * v.deleteVersion("b", "k", versionId);
 * }</pre>
 */
public final class MinioVersion {

    private final Minio minio;

    MinioVersion(Minio minio) {
        this.minio = Objects.requireNonNull(minio, "minio");
    }

    public Minio minio() {
        return minio;
    }

    /** Put object (version id returned when bucket versioning is enabled). */
    public ObjectWriteResponse putVersion(String bucket, String objectKey, byte[] data, MinioOptions opts) {
        MinioOptions o = (opts == null ? MinioOptions.defaults() : opts).toBuilder()
                .bucket(bucket)
                .objectKey(objectKey)
                .build();
        return minio.putBytes(data, o);
    }

    public ObjectWriteResponse putVersion(String objectKey, byte[] data, MinioOptions opts) {
        String bucket = opts != null && opts.bucket() != null ? opts.bucket() : minio.options().bucket();
        if (bucket == null) {
            throw new MinioException("bucket required", null, "putVersion", null, objectKey);
        }
        return putVersion(bucket, objectKey, data, opts);
    }

    public void deleteVersion(String bucket, String objectKey, String versionId) {
        minio.removeObjectVersion(bucket, objectKey, versionId);
    }

    /**
     * List object versions under prefix (requires bucket versioning).
     * Uses {@code listObjects(...).includeVersions(true)}.
     */
    public List<VersionInfo> listVersions(String bucket, String prefix) {
        minio.ensureOpen();
        List<VersionInfo> out = new ArrayList<>();
        try {
            ListObjectsArgs.Builder b = ListObjectsArgs.builder()
                    .bucket(bucket)
                    .recursive(true)
                    .includeVersions(true);
            if (prefix != null && !prefix.isEmpty()) b.prefix(prefix);
            for (Result<Item> r : minio.raw().listObjects(b.build())) {
                Item item = r.get();
                if (item == null) continue;
                out.add(VersionInfo.from(item));
            }
            return out;
        } catch (Exception e) {
            throw MinioException.wrap("listVersions", bucket, prefix, e);
        }
    }

    public List<VersionInfo> listVersions(String bucket, String objectKey, boolean exactKeyOnly) {
        List<VersionInfo> all = listVersions(bucket, objectKey);
        if (!exactKeyOnly) return all;
        List<VersionInfo> filtered = new ArrayList<>();
        for (VersionInfo v : all) {
            if (objectKey.equals(v.objectName())) filtered.add(v);
        }
        return filtered;
    }

    public StatObjectResponse getAttributes(String bucket, String objectKey, String versionId) {
        minio.ensureOpen();
        try {
            StatObjectArgs.Builder b = StatObjectArgs.builder().bucket(bucket).object(objectKey);
            if (versionId != null) b.versionId(versionId);
            return minio.raw().statObject(b.build());
        } catch (Exception e) {
            throw MinioException.wrap("getAttributes", bucket, objectKey, e);
        }
    }

    // ── retention ────────────────────────────────────────────────────────────

    public void setRetention(String bucket, String objectKey, RetentionMode mode, ZonedDateTime retainUntil) {
        setRetention(bucket, objectKey, null, mode, retainUntil, false);
    }

    public void setRetention(String bucket, String objectKey, String versionId,
                             RetentionMode mode, ZonedDateTime retainUntil, boolean bypassGovernance) {
        minio.ensureOpen();
        try {
            Retention retention = new Retention(mode, retainUntil);
            SetObjectRetentionArgs.Builder b = SetObjectRetentionArgs.builder()
                    .bucket(bucket)
                    .object(objectKey)
                    .config(retention)
                    .bypassGovernanceMode(bypassGovernance);
            if (versionId != null) b.versionId(versionId);
            minio.raw().setObjectRetention(b.build());
        } catch (Exception e) {
            throw MinioException.wrap("setObjectRetention", bucket, objectKey, e);
        }
    }

    public void setRetentionDays(String bucket, String objectKey, int days, String mode) {
        RetentionMode m = parseMode(mode);
        setRetention(bucket, objectKey, m, ZonedDateTime.now().plusDays(Math.max(1, days)));
    }

    public Retention getRetention(String bucket, String objectKey) {
        return getRetention(bucket, objectKey, null);
    }

    public Retention getRetention(String bucket, String objectKey, String versionId) {
        minio.ensureOpen();
        try {
            GetObjectRetentionArgs.Builder b = GetObjectRetentionArgs.builder()
                    .bucket(bucket)
                    .object(objectKey);
            if (versionId != null) b.versionId(versionId);
            return minio.raw().getObjectRetention(b.build());
        } catch (Exception e) {
            throw MinioException.wrap("getObjectRetention", bucket, objectKey, e);
        }
    }

    // ── legal hold ───────────────────────────────────────────────────────────

    public void enableLegalHold(String bucket, String objectKey) {
        enableLegalHold(bucket, objectKey, null);
    }

    public void enableLegalHold(String bucket, String objectKey, String versionId) {
        minio.ensureOpen();
        try {
            EnableObjectLegalHoldArgs.Builder b = EnableObjectLegalHoldArgs.builder()
                    .bucket(bucket)
                    .object(objectKey);
            if (versionId != null) b.versionId(versionId);
            minio.raw().enableObjectLegalHold(b.build());
        } catch (Exception e) {
            throw MinioException.wrap("enableObjectLegalHold", bucket, objectKey, e);
        }
    }

    public void disableLegalHold(String bucket, String objectKey) {
        disableLegalHold(bucket, objectKey, null);
    }

    public void disableLegalHold(String bucket, String objectKey, String versionId) {
        minio.ensureOpen();
        try {
            DisableObjectLegalHoldArgs.Builder b = DisableObjectLegalHoldArgs.builder()
                    .bucket(bucket)
                    .object(objectKey);
            if (versionId != null) b.versionId(versionId);
            minio.raw().disableObjectLegalHold(b.build());
        } catch (Exception e) {
            throw MinioException.wrap("disableObjectLegalHold", bucket, objectKey, e);
        }
    }

    public boolean isLegalHoldEnabled(String bucket, String objectKey) {
        return isLegalHoldEnabled(bucket, objectKey, null);
    }

    public boolean isLegalHoldEnabled(String bucket, String objectKey, String versionId) {
        minio.ensureOpen();
        try {
            IsObjectLegalHoldEnabledArgs.Builder b = IsObjectLegalHoldEnabledArgs.builder()
                    .bucket(bucket)
                    .object(objectKey);
            if (versionId != null) b.versionId(versionId);
            return minio.raw().isObjectLegalHoldEnabled(b.build());
        } catch (Exception e) {
            throw MinioException.wrap("isObjectLegalHoldEnabled", bucket, objectKey, e);
        }
    }

    private static RetentionMode parseMode(String mode) {
        if (mode == null || mode.isBlank()) return RetentionMode.GOVERNANCE;
        String s = mode.trim().toUpperCase(Locale.ROOT);
        if (s.startsWith("COMP")) return RetentionMode.COMPLIANCE;
        return RetentionMode.GOVERNANCE;
    }

    /** One version row from listObjects(includeVersions=true). */
    public record VersionInfo(
            String objectName,
            String versionId,
            boolean isLatest,
            boolean isDeleteMarker,
            long size,
            String etag,
            ZonedDateTime lastModified
    ) {
        static VersionInfo from(Item item) {
            return new VersionInfo(
                    item.objectName(),
                    item.versionId(),
                    item.isLatest(),
                    item.isDeleteMarker(),
                    item.size(),
                    item.etag(),
                    item.lastModified()
            );
        }
    }
}
