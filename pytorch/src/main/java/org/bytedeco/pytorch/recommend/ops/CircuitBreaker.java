/*
 * Circuit breaker (Netflix Hystrix / resilience4j style) for downstream
 * dependencies of the ranking service: feature store, ANN, fine-rank model
 * server, user profile service, etc.
 *
 * States: CLOSED -> OPEN -> HALF_OPEN -> CLOSED
 */
package org.bytedeco.pytorch.recommend.ops;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Supplier;

/** Circuit breaker for a named dependency. */
public final class CircuitBreaker {

    public enum State {
        CLOSED,
        OPEN,
        HALF_OPEN
    }

    public static final class Config {
        public final int failureThreshold;
        public final int successThresholdHalfOpen;
        public final long openDurationMs;
        public final double failureRateThreshold;
        public final int minimumCalls;

        public Config(
                int failureThreshold,
                int successThresholdHalfOpen,
                long openDurationMs,
                double failureRateThreshold,
                int minimumCalls) {
            this.failureThreshold = failureThreshold;
            this.successThresholdHalfOpen = successThresholdHalfOpen;
            this.openDurationMs = openDurationMs;
            this.failureRateThreshold = failureRateThreshold;
            this.minimumCalls = minimumCalls;
        }

        public static Config defaults() {
            return new Config(20, 3, 30_000L, 0.5, 20);
        }
    }

    private final String name;
    private final Config config;
    private final AtomicReference<State> state = new AtomicReference<>(State.CLOSED);
    private final AtomicLong consecutiveFailures = new AtomicLong();
    private final AtomicLong consecutiveSuccesses = new AtomicLong();
    private final AtomicLong openedAtMs = new AtomicLong();
    private final AtomicLong windowFailures = new AtomicLong();
    private final AtomicLong windowCalls = new AtomicLong();

    public CircuitBreaker(String name) {
        this(name, Config.defaults());
    }

    public CircuitBreaker(String name, Config config) {
        this.name = Objects.requireNonNull(name);
        this.config = Objects.requireNonNull(config);
    }

    public String name() {
        return name;
    }

    public State state() {
        tryHalfOpenTransition();
        return state.get();
    }

    public boolean allowRequest() {
        State s = state();
        if (s == State.CLOSED) return true;
        if (s == State.HALF_OPEN) return true; // limited probe; caller may still throttle
        return false; // OPEN
    }

    public <T> T execute(Supplier<T> supplier, Supplier<T> fallback) {
        if (!allowRequest()) {
            return fallback.get();
        }
        try {
            T result = supplier.get();
            onSuccess();
            return result;
        } catch (RuntimeException ex) {
            onFailure();
            return fallback.get();
        }
    }

    public void onSuccess() {
        windowCalls.incrementAndGet();
        consecutiveFailures.set(0);
        State s = state.get();
        if (s == State.HALF_OPEN) {
            long succ = consecutiveSuccesses.incrementAndGet();
            if (succ >= config.successThresholdHalfOpen) {
                state.set(State.CLOSED);
                consecutiveSuccesses.set(0);
                windowFailures.set(0);
                windowCalls.set(0);
            }
        }
    }

    public void onFailure() {
        windowCalls.incrementAndGet();
        windowFailures.incrementAndGet();
        long fails = consecutiveFailures.incrementAndGet();
        consecutiveSuccesses.set(0);
        State s = state.get();
        if (s == State.HALF_OPEN) {
            tripOpen();
            return;
        }
        if (s == State.CLOSED) {
            long calls = windowCalls.get();
            if (fails >= config.failureThreshold
                    || (calls >= config.minimumCalls
                    && windowFailures.get() / (double) calls >= config.failureRateThreshold)) {
                tripOpen();
            }
        }
    }

    private void tripOpen() {
        state.set(State.OPEN);
        openedAtMs.set(System.currentTimeMillis());
        consecutiveSuccesses.set(0);
    }

    private void tryHalfOpenTransition() {
        if (state.get() != State.OPEN) return;
        long opened = openedAtMs.get();
        if (System.currentTimeMillis() - opened >= config.openDurationMs) {
            state.compareAndSet(State.OPEN, State.HALF_OPEN);
        }
    }

    public void reset() {
        state.set(State.CLOSED);
        consecutiveFailures.set(0);
        consecutiveSuccesses.set(0);
        windowFailures.set(0);
        windowCalls.set(0);
    }

    @Override
    public String toString() {
        return "CircuitBreaker{name=" + name + ", state=" + state()
                + ", fails=" + consecutiveFailures.get() + "}";
    }
}
