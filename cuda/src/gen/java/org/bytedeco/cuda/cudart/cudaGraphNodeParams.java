// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.cuda.cudart;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.cuda.global.cudart.*;


/**
 * Graph node parameters.  See ::cudaGraphAddNode.
 */
@Properties(inherit = org.bytedeco.cuda.presets.cudart.class)
public class cudaGraphNodeParams extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public cudaGraphNodeParams(Pointer p) { super(p); }

    /** Type of the node */
    public native @Cast("cudaGraphNodeType") int type(); public native cudaGraphNodeParams type(int setter);
    /** Reserved.  Must be zero. */
    public native int reserved0(int i); public native cudaGraphNodeParams reserved0(int i, int setter);
    @MemberGetter public native IntPointer reserved0();
        /** Padding. Unused bytes must be zero. */
        public native long reserved1(int i); public native cudaGraphNodeParams reserved1(int i, long setter);
        @MemberGetter public native LongPointer reserved1();
        /** Kernel node parameters. */
        public native @ByRef cudaKernelNodeParamsV2 kernel(); public native cudaGraphNodeParams kernel(cudaKernelNodeParamsV2 setter);
        /** Memcpy node parameters. */
        public native @ByRef cudaMemcpyNodeParams memcpy(); public native cudaGraphNodeParams memcpy(cudaMemcpyNodeParams setter);
        /** Memset node parameters. */
        public native @ByRef cudaMemsetParamsV2 memset(); public native cudaGraphNodeParams memset(cudaMemsetParamsV2 setter);
        /** Host node parameters. */
        public native @ByRef cudaHostNodeParamsV2 host(); public native cudaGraphNodeParams host(cudaHostNodeParamsV2 setter);
        /** Child graph node parameters. */
        public native @ByRef cudaChildGraphNodeParams graph(); public native cudaGraphNodeParams graph(cudaChildGraphNodeParams setter);
        /** Event wait node parameters. */
        public native @ByRef cudaEventWaitNodeParams eventWait(); public native cudaGraphNodeParams eventWait(cudaEventWaitNodeParams setter);
        /** Event record node parameters. */
        public native @ByRef cudaEventRecordNodeParams eventRecord(); public native cudaGraphNodeParams eventRecord(cudaEventRecordNodeParams setter);
        /** External semaphore signal node parameters. */
        public native @ByRef cudaExternalSemaphoreSignalNodeParamsV2 extSemSignal(); public native cudaGraphNodeParams extSemSignal(cudaExternalSemaphoreSignalNodeParamsV2 setter);
        /** External semaphore wait node parameters. */
        public native @ByRef cudaExternalSemaphoreWaitNodeParamsV2 extSemWait(); public native cudaGraphNodeParams extSemWait(cudaExternalSemaphoreWaitNodeParamsV2 setter);
        /** Memory allocation node parameters. */
        public native @ByRef cudaMemAllocNodeParamsV2 alloc(); public native cudaGraphNodeParams alloc(cudaMemAllocNodeParamsV2 setter);
        /** Memory free node parameters. */
        public native @ByRef @Name("free") cudaMemFreeNodeParams _free(); public native cudaGraphNodeParams _free(cudaMemFreeNodeParams setter);
        /** Conditional node parameters. */
        public native @ByRef cudaConditionalNodeParams conditional(); public native cudaGraphNodeParams conditional(cudaConditionalNodeParams setter);

    /** Reserved bytes. Must be zero. */
    public native long reserved2(); public native cudaGraphNodeParams reserved2(long setter);
}
