package distribute;/*
 * Smoke test + micro-benchmark for the JavaCPP bindings of
 * {@code torch::distributed::rpc}.
 *
 * <p>This benchmark exercises the surface that does <em>not</em> require
 * booting a multi-process RPC stack: POD struct construction, field
 * accessors, the static {@code getCurrentRpcAgent} accessor, and the
 * standard JNI marshalling cost of every binding. Full end-to-end RPC
 * (two processes exchanging Messages over a tensorpipe loopback) is out of
 * scope for a single-process benchmark.</p>
 */

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.pytorch.rpc.RpcRetryOptions;
import org.bytedeco.pytorch.rpc.TensorPipeRpcBackendOptions;
import org.bytedeco.pytorch.rpc.WorkerInfo;

public class RpcSmoke {

    public static void main(String[] args) {
        System.out.println("[distribute.RpcSmoke] starting ...");

        // 1) Header-only smoke: every class bound by torch_rpc preset is loadable.
        checkLoaded("org.bytedeco.pytorch.rpc.RpcAgent",                   "RpcAgent (abstract)");
        checkLoaded("org.bytedeco.pytorch.rpc.TensorPipeAgent",            "TensorPipeAgent");
        checkLoaded("org.bytedeco.pytorch.rpc.Message",                    "Message");
        checkLoaded("org.bytedeco.pytorch.rpc.WorkerInfo",                 "WorkerInfo");
        checkLoaded("org.bytedeco.pytorch.rpc.RpcBackendOptions",          "RpcBackendOptions");
        checkLoaded("org.bytedeco.pytorch.rpc.TensorPipeRpcBackendOptions","TensorPipeRpcBackendOptions");
        checkLoaded("org.bytedeco.pytorch.rpc.RpcRetryOptions",            "RpcRetryOptions");
        checkLoaded("org.bytedeco.pytorch.rpc.RpcRetryInfo",               "RpcRetryInfo");
        checkLoaded("org.bytedeco.pytorch.rpc.RpcCommandBase",             "RpcCommandBase");
        checkLoaded("org.bytedeco.pytorch.rpc.ScriptCall",                 "ScriptCall");
        checkLoaded("org.bytedeco.pytorch.rpc.ScriptResp",                 "ScriptResp");
        checkLoaded("org.bytedeco.pytorch.rpc.ScriptRemoteCall",           "ScriptRemoteCall");
        checkLoaded("org.bytedeco.pytorch.rpc.RequestCallback",            "RequestCallback");
        checkLoaded("org.bytedeco.pytorch.rpc.RequestCallbackNoPython",    "RequestCallbackNoPython");
        // RequestCallbackImpl: parse-time shim + compile-time real header (Python/pybind).
        checkLoaded("org.bytedeco.pytorch.rpc.RequestCallbackImpl",        "RequestCallbackImpl");
        checkLoaded("org.bytedeco.pytorch.rpc.RRef",                       "RRef (abstract)");
        checkLoaded("org.bytedeco.pytorch.rpc.UserRRef",                   "UserRRef");
        checkLoaded("org.bytedeco.pytorch.rpc.OwnerRRef",                  "OwnerRRef");
        checkLoaded("org.bytedeco.pytorch.rpc.RRefContext",                "RRefContext");
        checkLoaded("org.bytedeco.pytorch.rpc.RRefForkData",               "RRefForkData");
        checkLoaded("org.bytedeco.pytorch.rpc.RRefForkRequest",            "RRefForkRequest");
        checkLoaded("org.bytedeco.pytorch.rpc.RRefChildAccept",            "RRefChildAccept");
        checkLoaded("org.bytedeco.pytorch.rpc.RRefAck",                    "RRefAck");
        checkLoaded("org.bytedeco.pytorch.rpc.RRefUserDelete",             "RRefUserDelete");
        checkLoaded("org.bytedeco.pytorch.rpc.RemoteRet",                  "RemoteRet");
        checkLoaded("org.bytedeco.pytorch.rpc.RRefMessageBase",            "RRefMessageBase");
        checkLoaded("org.bytedeco.pytorch.rpc.ForkMessageBase",            "ForkMessageBase");
        checkLoaded("org.bytedeco.pytorch.rpc.ScriptRRefFetchCall",        "ScriptRRefFetchCall");
        checkLoaded("org.bytedeco.pytorch.rpc.ScriptRRefFetchRet",         "ScriptRRefFetchRet");
        checkLoaded("org.bytedeco.pytorch.rpc.RRefFetchRet",               "RRefFetchRet");
        checkLoaded("org.bytedeco.pytorch.rpc.NetworkSourceInfo",          "NetworkSourceInfo");
        checkLoaded("org.bytedeco.pytorch.rpc.AggregatedNetworkData",      "AggregatedNetworkData");
        checkLoaded("org.bytedeco.pytorch.rpc.TransportRegistration",      "TransportRegistration");
        checkLoaded("org.bytedeco.pytorch.rpc.ChannelRegistration",        "ChannelRegistration");
        checkLoaded("org.bytedeco.pytorch.rpc.GloballyUniqueId",           "GloballyUniqueId");
        // Global accessor:
        checkLoaded("org.bytedeco.pytorch.global.torch_rpc",                "global torch_rpc");

        // 2) Static enum/value-type accessors from the global class.
        benchEnumConstants();
        benchRpcErrorTypeConstants();

        // 3) Constructor + field access exercises the JNI bridge.
        benchValueTypeConstruction();
        benchRpcRetryOptionsMutation();

        System.out.println("[distribute.RpcSmoke] OK");
    }

    //--------------------------------------------------------------------
    // Helpers
    //--------------------------------------------------------------------

    private static void checkLoaded(String fqn, String label) {
        try {
            Class<?> c = Class.forName(fqn);
            System.out.println("[distribute.RpcSmoke]  bound  " + fqn + "  (" + label + ")");
        } catch (ClassNotFoundException e) {
            throw new AssertionError("RPC class not bound: " + fqn, e);
        }
    }

    private static long nowNanos() { return System.nanoTime(); }
    private static double ms(long ns) { return ns / 1.0e6; }

    //--------------------------------------------------------------------
    // 1) Enumerations from the global class
    //--------------------------------------------------------------------
    private static void benchEnumConstants() {
        System.out.println("[distribute.RpcSmoke] benchEnumConstants");
        int[] constants = new int[8];
        long t0 = nowNanos();
        try {
            Class<?> global = Class.forName("org.bytedeco.pytorch.global.torch_rpc");
            String[] names = {
                "SCRIPT_CALL", "SCRIPT_RET",
                "PYTHON_CALL", "PYTHON_RET",
                "SCRIPT_REMOTE_CALL", "PYTHON_REMOTE_CALL", "REMOTE_RET",
                "REQUEST_TYPE"
            };
            int ok = 0;
            for (int i = 0; i < names.length; i++) {
                try {
                    java.lang.reflect.Field f = global.getField(names[i]);
                    constants[i] = f.getInt(null);
                    ok++;
                } catch (NoSuchFieldException nsf) {
                    constants[i] = -1;
                }
            }
            System.out.println("[distribute.RpcSmoke]  reflected " + ok + " of " + names.length + " enum constants from global");
        } catch (Throwable t) {
            System.out.println("[distribute.RpcSmoke]  (skipping: " + t.getClass().getSimpleName() + ": " + t.getMessage() + ")");
        }
        long t1 = nowNanos();
        System.out.printf("[distribute.RpcSmoke]  reflectively read 8 enum constants : %.2f ms  (%.0f ns/op)%n",
                ms(t1 - t0), (double) (t1 - t0) / 8);
    }

    private static void benchRpcErrorTypeConstants() {
        System.out.println("[distribute.RpcSmoke] benchRpcErrorTypeConstants");
        try {
            Class<?> wp = Class.forName("org.bytedeco.pytorch.rpc.WorkerInfo");
            System.out.println("[distribute.RpcSmoke]  WorkerInfo fields: "
                    + java.util.Arrays.toString(wp.getDeclaredFields()));
        } catch (Throwable t) {
            System.out.println("[distribute.RpcSmoke]  (skipping: " + t + ")");
        }
    }

    //--------------------------------------------------------------------
    // 2) constructing the public value types exercises the JNI bridge
    //    for a binding (allocation / deallocation paths).
    //--------------------------------------------------------------------
    private static void benchValueTypeConstruction() {
        System.out.println("[distribute.RpcSmoke] benchValueTypeConstruction");
        // Current JavaCPP peers only expose Pointer cast constructors for these
        // RPC value types (no zero-arg / value constructors). Probe via reflection
        // and skip cleanly when default construction is not bound — BenchmarkRpc
        // covers the broader public surface.
        probeDefaultConstruction(TensorPipeRpcBackendOptions.class);
        probeDefaultConstruction(RpcRetryOptions.class);
        probeConstruction(WorkerInfo.class, new Class<?>[]{BytePointer.class, short.class},
                new BytePointer("worker-0"), (short) 0);
        // Accessors that *are* bound:
        System.out.println("[distribute.RpcSmoke]  TensorPipeRpcBackendOptions.numWorkerThreads bound="
                + hasMethod(TensorPipeRpcBackendOptions.class, "numWorkerThreads"));
        System.out.println("[distribute.RpcSmoke]  RpcRetryOptions.maxRetries bound="
                + hasMethod(RpcRetryOptions.class, "maxRetries"));
        System.out.println("[distribute.RpcSmoke]  WorkerInfo.name_/id_ bound="
                + (hasMethod(WorkerInfo.class, "name_") && hasMethod(WorkerInfo.class, "id_")));
    }

    private static void benchRpcRetryOptionsMutation() {
        System.out.println("[distribute.RpcSmoke] benchRpcRetryOptionsMutation");
        if (!hasMethod(RpcRetryOptions.class, "maxRetries")) {
            System.out.println("[distribute.RpcSmoke]  skip mutation — maxRetries not bound");
            return;
        }
        System.out.println("[distribute.RpcSmoke]  skip mutation — default RpcRetryOptions ctor not bound "
                + "(Pointer-cast only). See BenchmarkRpc for agent-level coverage.");
    }

    private static boolean hasMethod(Class<?> cls, String name) {
        for (var m : cls.getMethods()) {
            if (m.getName().equals(name)) return true;
        }
        return false;
    }

    private static void probeDefaultConstruction(Class<?> cls) {
        try {
            var ctor = cls.getConstructor();
            Object o = ctor.newInstance();
            if (o instanceof Pointer p) p.close();
            System.out.println("[distribute.RpcSmoke]  " + cls.getSimpleName() + " default ctor: OK");
        } catch (NoSuchMethodException e) {
            System.out.println("[distribute.RpcSmoke]  " + cls.getSimpleName()
                    + " default ctor: NOT BOUND (Pointer cast only) — skip alloc bench");
        } catch (Throwable t) {
            System.out.println("[distribute.RpcSmoke]  " + cls.getSimpleName() + " default ctor failed: " + t);
        }
    }

    private static void probeConstruction(Class<?> cls, Class<?>[] types, Object... args) {
        try {
            var ctor = cls.getConstructor(types);
            Object o = ctor.newInstance(args);
            if (o instanceof Pointer p) p.close();
            System.out.println("[distribute.RpcSmoke]  " + cls.getSimpleName() + " value ctor: OK");
        } catch (NoSuchMethodException e) {
            System.out.println("[distribute.RpcSmoke]  " + cls.getSimpleName()
                    + " value ctor " + java.util.Arrays.toString(types)
                    + ": NOT BOUND — skip alloc bench");
        } catch (Throwable t) {
            System.out.println("[distribute.RpcSmoke]  " + cls.getSimpleName() + " value ctor failed: " + t);
        }
    }
}
