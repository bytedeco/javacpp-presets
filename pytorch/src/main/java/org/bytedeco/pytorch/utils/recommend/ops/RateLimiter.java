/*
 * Rate limiting / flow control for recommendation APIs.
 *
 * Patterns (Sentinel / Guava / Envoy local rate limit):
 *   - Token bucket (smooth QPS)
 *   - Sliding window counter
 *   - Per-user / per-scene quotas
 *   - System adaptive limit under degradation
 */
package org.bytedeco.pytorch.utils.recommend.ops;

import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/** Token-bucket and sliding-window rate limiters. */
public final class RateLimiter {

    private RateLimiter() {}

    /** Thread-safe token bucket. */
    public static final class TokenBucket {
        private final double capacity;
        private final double refillPerSecond;
        private double tokens;
        private long lastRefillNs;
        private final Object lock = new Object();

        public TokenBucket(double capacity, double refillPerSecond) {
            if (capacity <= 0 || refillPerSecond <= 0) {
                throw new IllegalArgumentException("capacity/refill must be > 0");
            }
            this.capacity = capacity;
            this.refillPerSecond = refillPerSecond;
            this.tokens = capacity;
            this.lastRefillNs = System.nanoTime();
        }

        public boolean tryAcquire() {
            return tryAcquire(1.0);
        }

        public boolean tryAcquire(double permits) {
            if (permits <= 0) return true;
            synchronized (lock) {
                refill();
                if (tokens >= permits) {
                    tokens -= permits;
                    return true;
                }
                return false;
            }
        }

        public double available() {
            synchronized (lock) {
                refill();
                return tokens;
            }
        }

        private void refill() {
            long now = System.nanoTime();
            double elapsedSec = (now - lastRefillNs) / 1_000_000_000.0;
            if (elapsedSec > 0) {
                tokens = Math.min(capacity, tokens + elapsedSec * refillPerSecond);
                lastRefillNs = now;
            }
        }
    }

    /** Fixed/sliding window counter limiter (approx sliding via two buckets). */
    public static final class SlidingWindow {
        private final long windowMs;
        private final long maxPermits;
        private final AtomicLong currentWindowStart = new AtomicLong();
        private final AtomicLong currentCount = new AtomicLong();
        private final AtomicLong previousCount = new AtomicLong();

        public SlidingWindow(long maxPermits, long windowMs) {
            if (maxPermits <= 0 || windowMs <= 0) {
                throw new IllegalArgumentException("invalid window config");
            }
            this.maxPermits = maxPermits;
            this.windowMs = windowMs;
            this.currentWindowStart.set(System.currentTimeMillis());
        }

        public boolean tryAcquire() {
            long now = System.currentTimeMillis();
            long start = currentWindowStart.get();
            if (now - start >= windowMs) {
                if (currentWindowStart.compareAndSet(start, now)) {
                    previousCount.set(currentCount.getAndSet(0));
                }
            }
            start = currentWindowStart.get();
            double weight = 1.0 - ((now - start) / (double) windowMs);
            if (weight < 0) weight = 0;
            if (weight > 1) weight = 1;
            double estimated = previousCount.get() * weight + currentCount.get();
            if (estimated >= maxPermits) {
                return false;
            }
            currentCount.incrementAndGet();
            return true;
        }
    }

    /** Per-key (user/scene) quota manager. */
    public static final class KeyedLimiter {
        private final double capacity;
        private final double refillPerSecond;
        private final ConcurrentHashMap<String, TokenBucket> buckets = new ConcurrentHashMap<>();

        public KeyedLimiter(double capacity, double refillPerSecond) {
            this.capacity = capacity;
            this.refillPerSecond = refillPerSecond;
        }

        public boolean tryAcquire(String key) {
            Objects.requireNonNull(key, "key");
            TokenBucket b = buckets.computeIfAbsent(key,
                    k -> new TokenBucket(capacity, refillPerSecond));
            return b.tryAcquire();
        }

        public void clear() {
            buckets.clear();
        }

        public int keyCount() {
            return buckets.size();
        }
    }

    /**
     * Adaptive limiter: multiplies base QPS by a factor derived from
     * degradation level / CPU (Sentinel system adaptive style, simplified).
     */
    public static final class AdaptiveLimiter {
        private final TokenBucket bucket;
        private final double baseQps;
        private volatile double factor = 1.0;

        public AdaptiveLimiter(double baseQps) {
            this.baseQps = baseQps;
            this.bucket = new TokenBucket(baseQps, baseQps);
        }

        public void setFactor(double factor) {
            this.factor = Math.max(0.05, Math.min(1.0, factor));
        }

        /** Map degradation severity 0..4 to factor. */
        public void fromDegradationSeverity(int severity) {
            switch (severity) {
                case 0:
                    setFactor(1.0);
                    break;
                case 1:
                    setFactor(0.8);
                    break;
                case 2:
                    setFactor(0.5);
                    break;
                case 3:
                    setFactor(0.2);
                    break;
                default:
                    setFactor(0.05);
                    break;
            }
        }

        public boolean tryAcquire() {
            // Probabilistic thin: allow with probability = factor when bucket has tokens.
            if (!bucket.tryAcquire()) {
                return false;
            }
            if (factor >= 1.0) return true;
            return Math.random() <= factor;
        }

        public double effectiveQps() {
            return baseQps * factor;
        }
    }
}
