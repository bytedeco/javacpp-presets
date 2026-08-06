package dataframe;

import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.utils.minio.Minio;
import org.bytedeco.pytorch.utils.minio.MinioOptions;

import java.util.List;

/**
 * Minimal smoke test: connect to MinIO, list objects in a bucket and read the
 * first non-directory object into a DataFrame.
 *
 * Endpoint / credentials per user request:
 *  - endpoint: http://192.168.113.244:9001
 *  - accessKey: minioadmin
 *  - secretKey: minioadmin
 *  - bucket: recsys-models
 */
public class BenchmarkDataFrameMinioRead {
    public static void main(String[] args) throws Exception {
        final String endpoint = "http://192.168.113.244:9000";
        final String access = "minioadmin";
        final String secret = "minioadmin";
        final String bucket = "recsys-models";

        System.out.println("=== BenchmarkDataFrameMinioRead ===");
        // optional prefix from args
        String prefixArg = (args != null && args.length > 0) ? args[0] : "";
        String bucketName = bucket;
        String prefix = prefixArg == null ? "" : prefixArg;
        // tolerate bucket passed with a path (e.g. "bucket/prefix/...")
        if ((prefix == null || prefix.isBlank()) && bucketName != null && bucketName.contains("/")) {
            int idx = bucketName.indexOf('/');
            prefix = bucketName.substring(idx + 1);
            bucketName = bucketName.substring(0, idx);
            System.out.println("Interpreting bucket as bucket/prefix: bucket=" + bucketName + " prefix=" + prefix);
        }

        try (Minio m = Minio.connect(endpoint, access, secret)) {
            System.out.println("Connected to MinIO: " + m.endpoint());
            List<Minio.MinioObjectInfo> items = m.listObjects(bucketName, prefix == null ? "" : prefix, true);
            if (items == null || items.isEmpty()) {
                System.err.println("No objects found in bucket: " + bucketName + " prefix=" + prefix);
                System.exit(2);
            }

            // collect candidate object keys (skip directories and obvious placeholders)
            java.util.List<String> candidates = new java.util.ArrayList<>();
            for (Minio.MinioObjectInfo it : items) {
                if (it == null) continue;
                String name = it.objectName();
                if (name == null || name.isBlank()) continue;
                if (it.isDir()) continue;
                if (name.endsWith("/")) continue;
                if (name.endsWith(".keep") || name.endsWith(".empty") || name.equals(".keep")) continue;
                candidates.add(name);
            }

            if (candidates.isEmpty()) {
                System.err.println("No candidate objects to read in bucket: " + bucketName + " prefix=" + prefix);
                System.exit(3);
            }

            // Read all candidate objects and vertically concatenate their DataFrames (deep traversal)
            List<DataFrame> dfs = new java.util.ArrayList<>();
            for (String key : candidates) {
                System.out.println("Attempt reading: " + bucketName + "/" + key);
                try {
                    DataFrame df = DataFrame.readMinio(m, MinioOptions.bucket(bucketName, key));
                    df.show();
                    System.out.println("Read DataFrame rows=" + df.rowCount() + ", cols=" + df.columnCount());
                    if (df.rowCount() == 0 && df.columnCount() == 0) {
                        System.out.println("Empty DataFrame for object, skipping");
                        continue;
                    }
                    dfs.add(df);
                } catch (Throwable e) {
                    System.err.println("Failed to read object " + key + ": " + e.getMessage());
                }
            }

            if (dfs.isEmpty()) {
                System.err.println("No readable DataFrame objects found in bucket: " + bucketName + " prefix=" + prefix);
                System.exit(4);
            }

            DataFrame combined = dfs.size() == 1 ? dfs.get(0) : DataFrame.vstack(dfs);
            System.out.println("Combined DataFrame rows=" + combined.rowCount() + ", cols=" + combined.columnCount());
            int n = Math.min(5, combined.rowCount());
            if (n > 0) {
                System.out.println("First " + n + " rows:");
                for (int i = 0; i < n; i++) System.out.println(org.bytedeco.pytorch.utils.orm.dataframe.DataFrameMapper.toMap(combined, i));
            } else {
                System.out.println("No rows in combined DataFrame; columns: " + combined.columnCount());
            }

        } catch (Throwable t) {
            t.printStackTrace(System.err);
            System.err.println("Failed to read from MinIO: " + t.getMessage());
            System.exit(1);
        }
    }
}
