package org.bytedeco.pytorch.utils.minio;

import io.minio.BucketExistsArgs;
import io.minio.DeleteBucketCorsArgs;
import io.minio.DeleteBucketEncryptionArgs;
import io.minio.DeleteBucketLifecycleArgs;
import io.minio.DeleteBucketNotificationArgs;
import io.minio.DeleteBucketPolicyArgs;
import io.minio.DeleteBucketReplicationArgs;
import io.minio.DeleteBucketTagsArgs;
import io.minio.GetBucketCorsArgs;
import io.minio.GetBucketEncryptionArgs;
import io.minio.GetBucketLifecycleArgs;
import io.minio.GetBucketNotificationArgs;
import io.minio.GetBucketPolicyArgs;
import io.minio.GetBucketReplicationArgs;
import io.minio.GetBucketTagsArgs;
import io.minio.GetBucketVersioningArgs;
import io.minio.ListenBucketNotificationArgs;
import io.minio.MakeBucketArgs;
import io.minio.RemoveBucketArgs;
import io.minio.Result;
import io.minio.SetBucketCorsArgs;
import io.minio.SetBucketEncryptionArgs;
import io.minio.SetBucketLifecycleArgs;
import io.minio.SetBucketNotificationArgs;
import io.minio.SetBucketPolicyArgs;
import io.minio.SetBucketReplicationArgs;
import io.minio.SetBucketTagsArgs;
import io.minio.SetBucketVersioningArgs;
import io.minio.messages.CORSConfiguration;
import io.minio.messages.LifecycleConfiguration;
import io.minio.messages.NotificationConfiguration;
import io.minio.messages.NotificationRecords;
import io.minio.messages.ReplicationConfiguration;
import io.minio.messages.SseConfiguration;
import io.minio.messages.Tags;
import io.minio.messages.VersioningConfiguration;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

/**
 * Bucket administration API wrapping MinIO 9.0.3 bucket operations.
 *
 * <pre>{@code
 * MinioBucket b = m.bucketApi();
 * b.create("datasets");
 * b.setVersioning("datasets", true);
 * b.setPolicy("datasets", policyJson);
 * b.setTags("datasets", Map.of("team", "ml"));
 * }</pre>
 */
public final class MinioBucket {

    private final Minio minio;

    MinioBucket(Minio minio) {
        this.minio = Objects.requireNonNull(minio, "minio");
    }

    public Minio minio() {
        return minio;
    }

    // ── lifecycle: create / delete / exists / list ───────────────────────────

    public void create(String bucket) {
        create(bucket, minio.options().region(), false);
    }

    public void create(String bucket, String region, boolean objectLock) {
        minio.ensureOpen();
        try {
            MakeBucketArgs.Builder b = MakeBucketArgs.builder().bucket(bucket);
            if (region != null && !region.isBlank()) b.region(region);
            if (objectLock) b.objectLock(true);
            minio.raw().makeBucket(b.build());
        } catch (Exception e) {
            String code = MinioException.mapErrorCode(e);
            if ("BucketAlreadyOwnedByYou".equals(code) || "BucketAlreadyExists".equals(code)) return;
            throw MinioException.wrap("createBucket", bucket, null, e);
        }
    }

    /** Alias of {@link #create(String)}. */
    public void makeBucket(String bucket) {
        create(bucket);
    }

    public void delete(String bucket) {
        minio.ensureOpen();
        try {
            minio.raw().removeBucket(RemoveBucketArgs.builder().bucket(bucket).build());
        } catch (Exception e) {
            throw MinioException.wrap("deleteBucket", bucket, null, e);
        }
    }

    /** Alias of {@link #delete(String)}. */
    public void removeBucket(String bucket) {
        delete(bucket);
    }

    public boolean exists(String bucket) {
        minio.ensureOpen();
        try {
            return minio.raw().bucketExists(BucketExistsArgs.builder().bucket(bucket).build());
        } catch (Exception e) {
            throw MinioException.wrap("bucketExists", bucket, null, e);
        }
    }

    public void ensure(String bucket) {
        if (!exists(bucket)) create(bucket);
    }

    public List<String> list() {
        return minio.listBuckets();
    }

    /**
     * Best-effort location: returns configured client region (MinIO SDK exposes
     * getBucketLocation on BaseS3Client async path; sync client uses region hint).
     */
    public String getLocation(String bucket) {
        // Prefer versioning call as connectivity probe; region from options.
        if (!exists(bucket)) {
            throw new MinioException("bucket not found: " + bucket, null, "getLocation", bucket, null, "NoSuchBucket");
        }
        String region = minio.options().region();
        return region == null || region.isBlank() ? "us-east-1" : region;
    }

    // ── versioning ───────────────────────────────────────────────────────────

    public void setVersioning(String bucket, boolean enabled) {
        minio.ensureOpen();
        try {
            VersioningConfiguration.Status status = enabled
                    ? VersioningConfiguration.Status.ENABLED
                    : VersioningConfiguration.Status.SUSPENDED;
            VersioningConfiguration cfg = new VersioningConfiguration(status, null, null, null);
            minio.raw().setBucketVersioning(SetBucketVersioningArgs.builder()
                    .bucket(bucket)
                    .config(cfg)
                    .build());
        } catch (Exception e) {
            throw MinioException.wrap("setBucketVersioning", bucket, null, e);
        }
    }

    public VersioningConfiguration getVersioning(String bucket) {
        minio.ensureOpen();
        try {
            return minio.raw().getBucketVersioning(GetBucketVersioningArgs.builder()
                    .bucket(bucket)
                    .build());
        } catch (Exception e) {
            throw MinioException.wrap("getBucketVersioning", bucket, null, e);
        }
    }

    public boolean isVersioningEnabled(String bucket) {
        VersioningConfiguration cfg = getVersioning(bucket);
        return cfg != null && cfg.status() == VersioningConfiguration.Status.ENABLED;
    }

    // ── policy ───────────────────────────────────────────────────────────────

    public void setPolicy(String bucket, String policyJson) {
        minio.ensureOpen();
        try {
            minio.raw().setBucketPolicy(SetBucketPolicyArgs.builder()
                    .bucket(bucket)
                    .config(policyJson)
                    .build());
        } catch (Exception e) {
            throw MinioException.wrap("setBucketPolicy", bucket, null, e);
        }
    }

    public String getPolicy(String bucket) {
        minio.ensureOpen();
        try {
            return minio.raw().getBucketPolicy(GetBucketPolicyArgs.builder().bucket(bucket).build());
        } catch (Exception e) {
            throw MinioException.wrap("getBucketPolicy", bucket, null, e);
        }
    }

    public void deletePolicy(String bucket) {
        minio.ensureOpen();
        try {
            minio.raw().deleteBucketPolicy(DeleteBucketPolicyArgs.builder().bucket(bucket).build());
        } catch (Exception e) {
            throw MinioException.wrap("deleteBucketPolicy", bucket, null, e);
        }
    }

    // ── tags ────────────────────────────────────────���────────────────────────

    public void setTags(String bucket, Map<String, String> tags) {
        minio.ensureOpen();
        try {
            minio.raw().setBucketTags(SetBucketTagsArgs.builder()
                    .bucket(bucket)
                    .tags(tags == null ? Map.of() : tags)
                    .build());
        } catch (Exception e) {
            throw MinioException.wrap("setBucketTags", bucket, null, e);
        }
    }

    public Map<String, String> getTags(String bucket) {
        minio.ensureOpen();
        try {
            Tags t = minio.raw().getBucketTags(GetBucketTagsArgs.builder().bucket(bucket).build());
            return t == null || t.get() == null ? Map.of() : new LinkedHashMap<>(t.get());
        } catch (Exception e) {
            throw MinioException.wrap("getBucketTags", bucket, null, e);
        }
    }

    public void deleteTags(String bucket) {
        minio.ensureOpen();
        try {
            minio.raw().deleteBucketTags(DeleteBucketTagsArgs.builder().bucket(bucket).build());
        } catch (Exception e) {
            throw MinioException.wrap("deleteBucketTags", bucket, null, e);
        }
    }

    // ── lifecycle ────────────────────────────────────────────────────────────

    public void setLifecycle(String bucket, LifecycleConfiguration config) {
        minio.ensureOpen();
        try {
            minio.raw().setBucketLifecycle(SetBucketLifecycleArgs.builder()
                    .bucket(bucket)
                    .config(config)
                    .build());
        } catch (Exception e) {
            throw MinioException.wrap("setBucketLifecycle", bucket, null, e);
        }
    }

    public LifecycleConfiguration getLifecycle(String bucket) {
        minio.ensureOpen();
        try {
            return minio.raw().getBucketLifecycle(GetBucketLifecycleArgs.builder().bucket(bucket).build());
        } catch (Exception e) {
            throw MinioException.wrap("getBucketLifecycle", bucket, null, e);
        }
    }

    public void deleteLifecycle(String bucket) {
        minio.ensureOpen();
        try {
            minio.raw().deleteBucketLifecycle(DeleteBucketLifecycleArgs.builder().bucket(bucket).build());
        } catch (Exception e) {
            throw MinioException.wrap("deleteBucketLifecycle", bucket, null, e);
        }
    }

    // ── CORS ─────────────────────────────────────────────────────────────────

    public void setCors(String bucket, CORSConfiguration config) {
        minio.ensureOpen();
        try {
            minio.raw().setBucketCors(SetBucketCorsArgs.builder().bucket(bucket).config(config).build());
        } catch (Exception e) {
            throw MinioException.wrap("setBucketCors", bucket, null, e);
        }
    }

    public CORSConfiguration getCors(String bucket) {
        minio.ensureOpen();
        try {
            return minio.raw().getBucketCors(GetBucketCorsArgs.builder().bucket(bucket).build());
        } catch (Exception e) {
            throw MinioException.wrap("getBucketCors", bucket, null, e);
        }
    }

    public void deleteCors(String bucket) {
        minio.ensureOpen();
        try {
            minio.raw().deleteBucketCors(DeleteBucketCorsArgs.builder().bucket(bucket).build());
        } catch (Exception e) {
            throw MinioException.wrap("deleteBucketCors", bucket, null, e);
        }
    }

    // ── encryption ───────────────────────────────────────────────────────────

    public void setEncryption(String bucket, SseConfiguration config) {
        minio.ensureOpen();
        try {
            minio.raw().setBucketEncryption(SetBucketEncryptionArgs.builder()
                    .bucket(bucket)
                    .config(config)
                    .build());
        } catch (Exception e) {
            throw MinioException.wrap("setBucketEncryption", bucket, null, e);
        }
    }

    public SseConfiguration getEncryption(String bucket) {
        minio.ensureOpen();
        try {
            return minio.raw().getBucketEncryption(GetBucketEncryptionArgs.builder().bucket(bucket).build());
        } catch (Exception e) {
            throw MinioException.wrap("getBucketEncryption", bucket, null, e);
        }
    }

    public void deleteEncryption(String bucket) {
        minio.ensureOpen();
        try {
            minio.raw().deleteBucketEncryption(DeleteBucketEncryptionArgs.builder().bucket(bucket).build());
        } catch (Exception e) {
            throw MinioException.wrap("deleteBucketEncryption", bucket, null, e);
        }
    }

    // ── replication ──────────────────────────────────────────────────────────

    public void setReplication(String bucket, ReplicationConfiguration config) {
        minio.ensureOpen();
        try {
            minio.raw().setBucketReplication(SetBucketReplicationArgs.builder()
                    .bucket(bucket)
                    .config(config)
                    .build());
        } catch (Exception e) {
            throw MinioException.wrap("setBucketReplication", bucket, null, e);
        }
    }

    public ReplicationConfiguration getReplication(String bucket) {
        minio.ensureOpen();
        try {
            return minio.raw().getBucketReplication(GetBucketReplicationArgs.builder().bucket(bucket).build());
        } catch (Exception e) {
            throw MinioException.wrap("getBucketReplication", bucket, null, e);
        }
    }

    public void deleteReplication(String bucket) {
        minio.ensureOpen();
        try {
            minio.raw().deleteBucketReplication(DeleteBucketReplicationArgs.builder().bucket(bucket).build());
        } catch (Exception e) {
            throw MinioException.wrap("deleteBucketReplication", bucket, null, e);
        }
    }

    // ── notification ─────────────────────────────────────────────────────────

    public void setNotification(String bucket, NotificationConfiguration config) {
        minio.ensureOpen();
        try {
            minio.raw().setBucketNotification(SetBucketNotificationArgs.builder()
                    .bucket(bucket)
                    .config(config)
                    .build());
        } catch (Exception e) {
            throw MinioException.wrap("setBucketNotification", bucket, null, e);
        }
    }

    public NotificationConfiguration getNotification(String bucket) {
        minio.ensureOpen();
        try {
            return minio.raw().getBucketNotification(GetBucketNotificationArgs.builder().bucket(bucket).build());
        } catch (Exception e) {
            throw MinioException.wrap("getBucketNotification", bucket, null, e);
        }
    }

    public void deleteNotification(String bucket) {
        minio.ensureOpen();
        try {
            minio.raw().deleteBucketNotification(DeleteBucketNotificationArgs.builder().bucket(bucket).build());
        } catch (Exception e) {
            throw MinioException.wrap("deleteBucketNotification", bucket, null, e);
        }
    }

    /**
     * Listen for bucket notification events. Caller must close the iterator.
     *
     * @param events e.g. {@code new String[]{"s3:ObjectCreated:*", "s3:ObjectRemoved:*"}}
     */
    public io.minio.CloseableIterator<Result<NotificationRecords>> listenNotification(
            String bucket, String prefix, String suffix, String[] events) {
        minio.ensureOpen();
        try {
            ListenBucketNotificationArgs.Builder b = ListenBucketNotificationArgs.builder()
                    .bucket(bucket)
                    .events(events == null ? new String[]{"s3:ObjectCreated:*"} : events);
            if (prefix != null) b.prefix(prefix);
            if (suffix != null) b.suffix(suffix);
            return minio.raw().listenBucketNotification(b.build());
        } catch (Exception e) {
            throw MinioException.wrap("listenBucketNotification", bucket, null, e);
        }
    }
}
