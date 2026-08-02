package org.bytedeco.pytorch.serving.tritonserver;

import org.bytedeco.pytorch.serving.tritonserver.internal.InferCallbacks;

import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import org.bytedeco.pytorch.serving.tritonserver.exceptions.*;
/**
 * Blocking iterator over inference responses.
 *
 * <p>Corresponds to Python {@code ResponseIterator}: yields
 * {@link TritonInferenceResponse} until a final response (or error) is observed.
 * When {@code raiseOnError} is true, a response-level error is thrown as
 * {@link TritonException} instead of being returned.
 */
public final class TRTResponseIterator implements Iterator<TritonInferenceResponse>, Iterable<TritonInferenceResponse>, AutoCloseable {
    private final InferCallbacks.InferSink sink;
    private final boolean raiseOnError;
    private final AtomicBoolean closed = new AtomicBoolean(false);
    private TritonInferenceResponse pending;
    private boolean exhausted;

    public TRTResponseIterator(InferCallbacks.InferSink sink, boolean raiseOnError) {
        this.sink = Objects.requireNonNull(sink, "sink");
        this.raiseOnError = raiseOnError;
    }

    @Override
    public Iterator<TritonInferenceResponse> iterator() {
        return this;
    }

    @Override
    public boolean hasNext() {
        if (exhausted || closed.get()) {
            return false;
        }
        if (pending != null) {
            return true;
        }
        pending = takeNext();
        if (pending == null || pending.isSentinel()) {
            exhausted = true;
            pending = null;
            return false;
        }
        return true;
    }

    @Override
    public TritonInferenceResponse next() {
        if (!hasNext()) {
            throw new NoSuchElementException("no more inference responses");
        }
        TritonInferenceResponse r = pending;
        pending = null;
        if (r.isFinal()) {
            exhausted = true;
        }
        if (raiseOnError && r.hasError()) {
            throw r.error();
        }
        if (sink.callbackError != null && r.hasError() == false) {
            // Surface async callback failures even if response parsed ok earlier.
            Throwable t = sink.callbackError;
            if (t instanceof RuntimeException re) {
                throw re;
            }
            throw new TritonInternalException("infer callback failed: " + t.getMessage(), t);
        }
        return r;
    }

    /**
     * Cancel outstanding work if supported. MVP documents that cancel is best-effort
     * and may be a no-op depending on backend/binding support.
     */
    public void cancel() {
        // C API cancel is not universally exposed on all bindings; mark closed so
        // the iterator stops consuming. In-flight native work may still complete.
        close();
    }

    @Override
    public void close() {
        closed.set(true);
        exhausted = true;
        pending = null;
        // Drain queue to help GC of tensors; ignore content.
        BlockingQueue<TritonInferenceResponse> q = sink.queue();
        q.clear();
    }

    private TritonInferenceResponse takeNext() {
        BlockingQueue<TritonInferenceResponse> q = sink.queue();
        try {
            while (!closed.get()) {
                if (sink.isFinished() && q.isEmpty()) {
                    return null;
                }
                TritonInferenceResponse r = q.poll(100, TimeUnit.MILLISECONDS);
                if (r != null) {
                    return r;
                }
                if (sink.callbackError != null && q.isEmpty() && sink.isFinished()) {
                    throw new TritonInternalException(
                            "infer failed in callback: " + sink.callbackError.getMessage(),
                            sink.callbackError);
                }
            }
            return null;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new TritonInternalException("interrupted while waiting for inference response", e);
        }
    }
}
