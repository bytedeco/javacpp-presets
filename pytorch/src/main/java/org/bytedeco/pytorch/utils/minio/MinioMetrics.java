package org.bytedeco.pytorch.utils.minio;

import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.concurrent.atomic.LongAdder;

/**
 * Lightweight in-process counters for MinIO put / get / list / delete paths.
 * No external metrics backend — callers may scrape {@link #snapshot()}.
 */
public final class MinioMetrics {

    private final LongAdder putCount = new LongAdder();
    private final LongAdder putFailed = new LongAdder();
    private final LongAdder putBytes = new LongAdder();
    private final LongAdder getCount = new LongAdder();
    private final LongAdder getFailed = new LongAdder();
    private final LongAdder getBytes = new LongAdder();
    private final LongAdder deleteCount = new LongAdder();
    private final LongAdder deleteFailed = new LongAdder();
    private final LongAdder listCount = new LongAdder();
    private final LongAdder copyCount = new LongAdder();
    private final LongAdder multipartParts = new LongAdder();
    private final DoubleAdder putLatencyMs = new DoubleAdder();
    private final AtomicLong putLatencyCount = new AtomicLong();
    private final DoubleAdder getLatencyMs = new DoubleAdder();
    private final AtomicLong getLatencyCount = new AtomicLong();
    private final long startedAtMs = System.currentTimeMillis();

    public void recordPut(long bytes, double latencyMs, boolean ok) {
        if (ok) {
            putCount.increment();
            putBytes.add(Math.max(0L, bytes));
        } else {
            putFailed.increment();
        }
        if (latencyMs >= 0) {
            putLatencyMs.add(latencyMs);
            putLatencyCount.incrementAndGet();
        }
    }

    public void recordPut(long bytes, boolean ok) {
        recordPut(bytes, -1, ok);
    }

    public void recordGet(long bytes, double latencyMs, boolean ok) {
        if (ok) {
            getCount.increment();
            getBytes.add(Math.max(0L, bytes));
        } else {
            getFailed.increment();
        }
        if (latencyMs >= 0) {
            getLatencyMs.add(latencyMs);
            getLatencyCount.incrementAndGet();
        }
    }

    public void recordGet(long bytes, boolean ok) {
        recordGet(bytes, -1, ok);
    }

    public void recordDelete(boolean ok) {
        if (ok) deleteCount.increment();
        else deleteFailed.increment();
    }

    public void recordList(int items, boolean ok) {
        if (ok) listCount.add(Math.max(0, items));
    }

    public void recordCopy(boolean ok) {
        if (ok) copyCount.increment();
    }

    public void recordMultipartPart(long bytes) {
        multipartParts.increment();
        putBytes.add(Math.max(0L, bytes));
    }

    public long putCount() { return putCount.sum(); }
    public long putFailed() { return putFailed.sum(); }
    public long putBytes() { return putBytes.sum(); }
    public long getCount() { return getCount.sum(); }
    public long getFailed() { return getFailed.sum(); }
    public long getBytes() { return getBytes.sum(); }
    public long deleteCount() { return deleteCount.sum(); }
    public long deleteFailed() { return deleteFailed.sum(); }
    public long listCount() { return listCount.sum(); }
    public long copyCount() { return copyCount.sum(); }
    public long multipartParts() { return multipartParts.sum(); }

    public double avgPutLatencyMs() {
        long n = putLatencyCount.get();
        if (n <= 0) return 0.0;
        return putLatencyMs.sum() / n;
    }

    public double avgGetLatencyMs() {
        long n = getLatencyCount.get();
        if (n <= 0) return 0.0;
        return getLatencyMs.sum() / n;
    }

    public double putQps() {
        double sec = Math.max(0.001, (System.currentTimeMillis() - startedAtMs) / 1000.0);
        return putCount.sum() / sec;
    }

    public double getQps() {
        double sec = Math.max(0.001, (System.currentTimeMillis() - startedAtMs) / 1000.0);
        return getCount.sum() / sec;
    }

    public double putThroughputMBps() {
        double sec = Math.max(0.001, (System.currentTimeMillis() - startedAtMs) / 1000.0);
        return (putBytes.sum() / (1024.0 * 1024.0)) / sec;
    }

    public double getThroughputMBps() {
        double sec = Math.max(0.001, (System.currentTimeMillis() - startedAtMs) / 1000.0);
        return (getBytes.sum() / (1024.0 * 1024.0)) / sec;
    }

    public Snapshot snapshot() {
        return new Snapshot(
                putCount.sum(),
                putFailed.sum(),
                putBytes.sum(),
                getCount.sum(),
                getFailed.sum(),
                getBytes.sum(),
                deleteCount.sum(),
                deleteFailed.sum(),
                listCount.sum(),
                copyCount.sum(),
                multipartParts.sum(),
                avgPutLatencyMs(),
                avgGetLatencyMs(),
                putQps(),
                getQps(),
                putThroughputMBps(),
                getThroughputMBps(),
                System.currentTimeMillis() - startedAtMs
        );
    }

    public void reset() {
        putCount.reset();
        putFailed.reset();
        putBytes.reset();
        getCount.reset();
        getFailed.reset();
        getBytes.reset();
        deleteCount.reset();
        deleteFailed.reset();
        listCount.reset();
        copyCount.reset();
        multipartParts.reset();
        putLatencyMs.reset();
        putLatencyCount.set(0);
        getLatencyMs.reset();
        getLatencyCount.set(0);
    }

    @Override
    public String toString() {
        return snapshot().toString();
    }

    /** Immutable point-in-time view. */
    public record Snapshot(
            long putCount,
            long putFailed,
            long putBytes,
            long getCount,
            long getFailed,
            long getBytes,
            long deleteCount,
            long deleteFailed,
            long listCount,
            long copyCount,
            long multipartParts,
            double avgPutLatencyMs,
            double avgGetLatencyMs,
            double putQps,
            double getQps,
            double putThroughputMBps,
            double getThroughputMBps,
            long uptimeMs
    ) {
        @Override
        public String toString() {
            return "MinioMetrics{put=" + putCount
                    + ", putFailed=" + putFailed
                    + ", get=" + getCount
                    + ", getFailed=" + getFailed
                    + ", delete=" + deleteCount
                    + ", list=" + listCount
                    + ", copy=" + copyCount
                    + ", parts=" + multipartParts
                    + ", putQps=" + String.format("%.1f", putQps)
                    + ", getQps=" + String.format("%.1f", getQps)
                    + ", putMB/s=" + String.format("%.2f", putThroughputMBps)
                    + ", getMB/s=" + String.format("%.2f", getThroughputMBps)
                    + ", avgPutMs=" + String.format("%.2f", avgPutLatencyMs)
                    + ", avgGetMs=" + String.format("%.2f", avgGetLatencyMs)
                    + ", uptimeMs=" + uptimeMs
                    + '}';
        }
    }
}
