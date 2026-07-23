/*
 * Copyright (C) 2026 Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.presets;

import org.bytedeco.javacpp.ClassProperties;
import org.bytedeco.javacpp.LoadEnabled;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;
import org.bytedeco.pytorch.presets.torch.PointerInfo;

/**
 * JavaCPP preset binding the {@code torch::distributed::rpc} C++ module
 * (PyTorch's RPC framework).
 *
 * <p>This preset inherits from {@link torch} so all the parentInfo mappings
 * (intrusive_ptr, std::optional, std::vector, IValue, ...) are inherited, then
 * layers the RPC-specific bindings.</p>
 *
 * <p>Header {@code torch/csrc/distributed/rpc/rpc.h} and the {@code python_*}
 * family (which require a live Python interpreter via pybind11) are excluded:
 * they are pybind glue, not native C++ APIs.</p>
 *
 * @author Hervé Guillemet
 */
@Properties(
    inherit = torch.class,
    value = @Platform(
        value = {"linux", "macosx", "windows"},
        compiler = "cpp20",
        include = {
            // Ordered so headers with fewer dependencies come first.
            //
            // NOTE: we deliberately do NOT include the two headers that hold
            // only free functions with compound std types (unordered_map /
            // pair / vector of vector / vector of intrusive_ptr) — those
            // signatures don't compose cleanly with the parent's pointer
            // types and the free functions are out of scope of the RPC
            // binding anyway:
            //   * torch/csrc/distributed/rpc/utils.h
            //   * torch/csrc/distributed/rpc/agent_utils.h
            "torch/csrc/distributed/rpc/types.h",
            "torch/csrc/distributed/rpc/rpc_command_base.h",
            "torch/csrc/distributed/rpc/message.h",
            "torch/csrc/distributed/rpc/request_callback.h",
            "torch/csrc/distributed/rpc/request_callback_no_python.h",
            "torch/csrc/distributed/rpc/request_callback_impl.h",
            "torch/csrc/distributed/rpc/rref_proto.h",
            "torch/csrc/distributed/rpc/rref_impl.h",
            "torch/csrc/distributed/rpc/rref_context.h",
            "torch/csrc/distributed/rpc/script_call.h",
            "torch/csrc/distributed/rpc/script_remote_call.h",
            "torch/csrc/distributed/rpc/script_resp.h",
            "torch/csrc/distributed/rpc/rpc_agent.h",
            "torch/csrc/distributed/rpc/tensorpipe_agent.h",
        },
        link = { "c10", "torch", "torch_cpu" }
    ),
    target = "org.bytedeco.pytorch.rpc",
    global = "org.bytedeco.pytorch.global.torch_rpc"
)
public class torch_rpc implements LoadEnabled, InfoMapper {

    @Override
    public void init(ClassProperties properties) {
        // Deliberately do NOT call torch.initIncludes(...): that would replace
        // platform.include with the contents of torch_include.h, which has
        // every torch RPC header commented out ("Not on Windows"). The
        // @Platform.include list declared on this preset is the source of
        // truth for the headers to parse for RPC.
        //
        // torch.sharedMap(...) is still applied during map(...) below so the
        // parent InfoMappings (intrusive_ptr, std::optional, IValue, ...) are
        // available while parsing.
    }

    @Override
    public void map(InfoMap infoMap) {
        torch.sharedMap(infoMap);

        //--- Cross-package pointer types -----------------------------------------
        // Rpc class hierarchies (RRef, RRefContext, ...) reference c10::TypePtr
        // and other parent-package types. Torch.sharedMap(...) doesn't include
        // the full torch.map(); add the entries the parent torch preset applies
        // for these types so the parser can resolve them. The torch preset
        // declares TypePtr as an inner class of `Type` (Type.TypePtr) — qualify
        // it explicitly so the rpc-package generated files can resolve it.
        infoMap
            .put(new Info("c10::Type::SingletonOrSharedTypePtr<c10::Type>",
                          "c10::TypePtr", "c10::Type::TypePtr", "at::TypePtr",
                          "torch::jit::TypeAttr::ConstructorType",
                          "torch::jit::TypeAttr::ValueType")
                    .pointerTypes("Type.TypePtr"))
        ;

        //--- Scalar id types -------------------------------------------------------
        infoMap
            .put(new Info("torch::distributed::rpc::worker_id_t").valueTypes("short").pointerTypes("ShortPointer"))
            .put(new Info("torch::distributed::rpc::local_id_t").valueTypes("long").pointerTypes("LongPointer"))
            .put(new Info("torch::distributed::rpc::GloballyUniqueId").purify().pointerTypes("GloballyUniqueId"))

            //--- Enumerations ------------------------------------------------------
            .put(new Info("torch::distributed::rpc::RPCErrorType").enumerate().valueTypes("RPCErrorType"))
            .put(new Info("torch::distributed::rpc::MessageTypeFlags").enumerate().valueTypes("MessageTypeFlags"))
            // NOTE: torch::distributed::rpc::MessageType is fully overridden via
            // .javaText() below (search for "MessageType: parser emits bare
            // REQUEST_TYPE"). Do NOT add an .enumerate() entry here — the
            // InfoMap returns the FIRST matching Info on lookup, and a bare
            // .enumerate() entry would shadow the javaText override and the
            // parser would emit its own (broken) MessageType enum.

            //--- POD structs -------------------------------------------------------
            .put(new Info("torch::distributed::rpc::RpcBackendOptions").purify().pointerTypes("RpcBackendOptions"))
            .put(new Info("torch::distributed::rpc::TensorPipeRpcBackendOptions").purify().pointerTypes("TensorPipeRpcBackendOptions"))
            // The field `deviceMaps` is `std::unordered_map<std::string, DeviceMap>`,
            // where DeviceMap is an alias inside rpc:: - skip these accessors.
            .put(new Info("torch::distributed::rpc::TensorPipeRpcBackendOptions::deviceMaps").skip())
            .put(new Info("torch::distributed::rpc::RpcRetryOptions").purify().pointerTypes("RpcRetryOptions"))
            .put(new Info("torch::distributed::rpc::RpcRetryInfo").purify().pointerTypes("RpcRetryInfo"))
            .put(new Info("torch::distributed::rpc::WorkerInfo").purify().pointerTypes("WorkerInfo"))
            .put(new Info("torch::distributed::rpc::RegisterWorkerInfoOnce").pointerTypes("RegisterWorkerInfoOnce").skip())
            .put(new Info("torch::distributed::rpc::RRefForkData").purify().pointerTypes("RRefForkData"))
            .put(new Info("torch::distributed::rpc::NetworkSourceInfo").purify().pointerTypes("NetworkSourceInfo"))
            .put(new Info("torch::distributed::rpc::AggregatedNetworkData").purify().pointerTypes("AggregatedNetworkData"))
            .put(new Info("torch::distributed::rpc::TransportRegistration").purify().pointerTypes("TransportRegistration"))
            .put(new Info("torch::distributed::rpc::ChannelRegistration").purify().pointerTypes("ChannelRegistration"))

            //--- Python-coupled bits: out of scope for the native binding ----------
            .put(new Info("torch::distributed::rpc::JitRRefPickleGuard").skip())
            .put(new Info("torch::distributed::rpc::enableJitRRefPickle",
                          "torch::distributed::rpc::disableJitRRefPickle",
                          "torch::distributed::rpc::getAllowJitRRefPickle").skip())
            .put(new Info("torch::distributed::rpc::PythonCall",
                          "torch::distributed::rpc::PythonResp",
                          "torch::distributed::rpc::PythonRemoteCall").skip())
            .put(new Info("torch::distributed::rpc::PythonRRefFetchCall",
                          "torch::distributed::rpc::PythonRRefFetchRet").skip())

            //--- MessageType: parser emits bare REQUEST_TYPE / RESPONSE_TYPE
            //    which aren't imported in torch_rpc's global class. Provide a
            //    hand-written enum that uses the qualified MessageTypeFlags.X.
            //--- MessageType -------------------------------------------------
            //    The parser translates `MessageTypeFlags::REQUEST_TYPE` to
            //    `MessageTypeFlags.REQUEST_TYPE` but stops short of adding
            //    `.ordinal()` — so the generated constructor args are bare
            //    `REQUEST_TYPE` identifiers that don't exist in scope.
            //    Skip the parser-generated MessageType enum and inject a
            //    hand-written one via the null-keyed InfoList emit path.
            .put(new Info("torch::distributed::rpc::MessageType").skip())
            .put(new Info((String) null).javaText(
                "@Namespace(\"torch::distributed::rpc\") public enum MessageType {\n"
                + "  SCRIPT_CALL(MessageTypeFlags.REQUEST_TYPE.ordinal()),\n"
                + "  SCRIPT_RET(MessageTypeFlags.RESPONSE_TYPE.ordinal()),\n"
                + "  PYTHON_CALL(MessageTypeFlags.REQUEST_TYPE.ordinal()),\n"
                + "  PYTHON_RET(MessageTypeFlags.RESPONSE_TYPE.ordinal()),\n"
                + "  SCRIPT_REMOTE_CALL(MessageTypeFlags.REQUEST_TYPE.ordinal()),\n"
                + "  PYTHON_REMOTE_CALL(MessageTypeFlags.REQUEST_TYPE.ordinal()),\n"
                + "  REMOTE_RET(MessageTypeFlags.RESPONSE_TYPE.ordinal()),\n"
                + "  SCRIPT_RREF_FETCH_CALL(MessageTypeFlags.REQUEST_TYPE.ordinal()),\n"
                + "  PYTHON_RREF_FETCH_CALL(MessageTypeFlags.REQUEST_TYPE.ordinal()),\n"
                + "  SCRIPT_RREF_FETCH_RET(MessageTypeFlags.RESPONSE_TYPE.ordinal()),\n"
                + "  PYTHON_RREF_FETCH_RET(MessageTypeFlags.RESPONSE_TYPE.ordinal()),\n"
                + "  RREF_USER_DELETE(MessageTypeFlags.REQUEST_TYPE.ordinal()),\n"
                + "  RREF_FORK_REQUEST(MessageTypeFlags.REQUEST_TYPE.ordinal()),\n"
                + "  RREF_CHILD_ACCEPT(MessageTypeFlags.REQUEST_TYPE.ordinal()),\n"
                + "  RREF_ACK(MessageTypeFlags.RESPONSE_TYPE.ordinal()),\n"
                + "  FORWARD_AUTOGRAD_REQ(MessageTypeFlags.REQUEST_TYPE.ordinal()),\n"
                + "  FORWARD_AUTOGRAD_RESP(MessageTypeFlags.RESPONSE_TYPE.ordinal()),\n"
                + "  BACKWARD_AUTOGRAD_REQ(MessageTypeFlags.REQUEST_TYPE.ordinal()),\n"
                + "  BACKWARD_AUTOGRAD_RESP(MessageTypeFlags.RESPONSE_TYPE.ordinal()),\n"
                + "  CLEANUP_AUTOGRAD_CONTEXT_REQ(MessageTypeFlags.REQUEST_TYPE.ordinal()),\n"
                + "  CLEANUP_AUTOGRAD_CONTEXT_RESP(MessageTypeFlags.RESPONSE_TYPE.ordinal()),\n"
                + "  RUN_WITH_PROFILING_REQ(MessageTypeFlags.REQUEST_TYPE.ordinal()),\n"
                + "  RUN_WITH_PROFILING_RESP(MessageTypeFlags.RESPONSE_TYPE.ordinal()),\n"
                + "  RREF_BACKWARD_REQ(MessageTypeFlags.REQUEST_TYPE.ordinal()),\n"
                + "  RREF_BACKWARD_RESP(MessageTypeFlags.RESPONSE_TYPE.ordinal()),\n"
                + "  EXCEPTION(MessageTypeFlags.RESPONSE_TYPE.ordinal()),\n"
                + "  UNKNOWN(0x3c);\n"
                + "  public final int value;\n"
                + "  private MessageType(int v) { this.value = v; }\n"
                + "  private MessageType(MessageType e) { this.value = e.value; }\n"
                + "  public MessageType intern() { for (MessageType e : values()) if (e.value == value) return e; return this; }\n"
                + "  @Override public String toString() { return intern().name(); }\n"
                + "}"
            ))

            // RequestCallbackImpl::runPythonFunction references pybind11::object
            // (the `object` parameter); pybind is out of scope here.
            .put(new Info("torch::distributed::rpc::RequestCallbackImpl::runPythonFunction").skip())
        ;

        //--- Skip transitive free functions from utils.h / agent_utils.h --------
        // rref_context.h and tensorpipe_utils.h pull in utils.h / agent_utils.h
        // transitively, so we cannot just drop the @Platform.include entries.
        // Both headers only declare free functions (no classes / types), so
        // line-skip every line and nothing breaks downstream.
        infoMap
            .put(new Info("agent_utils.h").linePatterns(".*").skip())
            .put(new Info("utils.h").linePatterns(".*").skip())
        ;

        //--- getCurrentRpcAgent / setCurrentRpcAgent (free function, std::shared_ptr<RpcAgent>)
        infoMap.put(new Info("torch::distributed::rpc::getCurrentRpcAgent").javaText(
            "public static native @Name(\"torch::distributed::rpc::getCurrentRpcAgent\") "
                + "@SharedPtr @Cast(\"std::shared_ptr<torch::distributed::rpc::RpcAgent>\") RpcAgent getCurrentRpcAgent();"
        ));
        infoMap.put(new Info("torch::distributed::rpc::setCurrentRpcAgent").javaText(
            "public static native @Name(\"torch::distributed::rpc::setCurrentRpcAgent\") "
                + "void setCurrentRpcAgent(@SharedPtr @Cast(\"std::shared_ptr<torch::distributed::rpc::RpcAgent>\") RpcAgent agent);"
        ));

        //--- Message + RpcCommandBase family -------------------------------------
        infoMap
            .put(new Info("torch::distributed::rpc::Message").purify().pointerTypes("Message"))
            .put(new Info("torch::distributed::rpc::Message::Message").skip())   // No public ctors exposed
            .put(new Info("torch::distributed::rpc::createExceptionResponse",
                          "torch::distributed::rpc::createUserExceptionResponse").skip())
            .put(new Info("torch::distributed::rpc::Message::id",
                          "torch::distributed::rpc::Message::type",
                          "torch::distributed::rpc::Message::status",
                          "torch::distributed::rpc::Message::isShutdown",
                          "torch::distributed::rpc::Message::senderId",
                          "torch::distributed::rpc::Message::tensors",
                          "torch::distributed::rpc::Message::pickle",
                          "torch::distributed::rpc::Message::meta",
                          "torch::distributed::rpc::Message::markCompleted",
                          "torch::distributed::rpc::Message::markException",
                          // withStorages returns std::tuple of intrusive_ptr + weak_intrusive_ptr
                          // vector which we don't bind.
                          "torch::distributed::rpc::Message::withStorages").skip())

            .put(new Info("torch::distributed::rpc::RpcCommandBase")
                    .purify().pointerTypes("RpcCommandBase").virtualize())
            .put(new Info("torch::distributed::rpc::ScriptCall").purify().pointerTypes("ScriptCall"))
            .put(new Info("torch::distributed::rpc::ScriptRemoteCall").purify().pointerTypes("ScriptRemoteCall"))
            .put(new Info("torch::distributed::rpc::ScriptResp").purify().pointerTypes("ScriptResp"))
            .put(new Info("torch::distributed::rpc::RRefMessageBase").purify().pointerTypes("RRefMessageBase"))
            .put(new Info("torch::distributed::rpc::ForkMessageBase").purify().pointerTypes("ForkMessageBase"))
            // fromMessage returns std::pair<RRefId, ForkId>; pair of custom RPC
            // types not exposed, so skip the helper.
            .put(new Info("torch::distributed::rpc::ForkMessageBase::fromMessage").skip())
            .put(new Info("torch::distributed::rpc::RRefForkRequest").purify().pointerTypes("RRefForkRequest"))
            .put(new Info("torch::distributed::rpc::RRefChildAccept").purify().pointerTypes("RRefChildAccept"))
            .put(new Info("torch::distributed::rpc::RRefAck").purify().pointerTypes("RRefAck"))
            .put(new Info("torch::distributed::rpc::RRefUserDelete").purify().pointerTypes("RRefUserDelete"))
            .put(new Info("torch::distributed::rpc::RemoteRet").purify().pointerTypes("RemoteRet"))
            .put(new Info("torch::distributed::rpc::ScriptRRefFetchCall").purify().pointerTypes("ScriptRRefFetchCall"))
            .put(new Info("torch::distributed::rpc::ScriptRRefFetchRet").purify().pointerTypes("ScriptRRefFetchRet"))
            .put(new Info("torch::distributed::rpc::RRefFetchRet").purify().pointerTypes("RRefFetchRet"))
        ;

        //--- Request callback ----------------------------------------------------
        infoMap
            .put(new Info("torch::distributed::rpc::RequestCallback")
                    .purify().pointerTypes("RequestCallback").virtualize())
            .put(new Info("torch::distributed::rpc::RequestCallback::processMessage").virtualize())
            .put(new Info("torch::distributed::rpc::RequestCallbackNoPython")
                    .purify().pointerTypes("RequestCallbackNoPython").virtualize())
            .put(new Info("torch::distributed::rpc::RequestCallbackImpl").purify().pointerTypes("RequestCallbackImpl"))
            .put(new Info("torch::distributed::rpc::RequestCallback::operator ()").skip())
        ;

        //--- RRef runtime --------------------------------------------------------
        infoMap
            .put(new Info("c10::RRefInterface").purify().pointerTypes("RRefInterface").virtualize())
            .put(new Info("torch::distributed::rpc::RRef").purify().pointerTypes("RRef").virtualize())
            .put(new Info("torch::distributed::rpc::UserRRef").purify().pointerTypes("UserRRef").virtualize())
            .put(new Info("torch::distributed::rpc::OwnerRRef").purify().pointerTypes("OwnerRRef").virtualize())
            .put(new Info("torch::distributed::rpc::RRef::operator <<").skip())
            // Skip the constructors / factories that take c10::TypePtr — the
            // Java-side equivalent goes through Module factories in torch,
            // and resolving Type.TypePtr across packages is awkward.
            .put(new Info("torch::distributed::rpc::UserRRef::UserRRef").skip())
            .put(new Info("torch::distributed::rpc::OwnerRRef::OwnerRRef").skip())
            // RRef::type() returns TypePtr (cross-package type, see above).
            .put(new Info("torch::distributed::rpc::RRef::type").skip())
            // RRefContext::getOrCreateRRef / createUserRRef / createOwnerRRef /
            // getOrCreateOwnerRRef take TypePtr and produce intrusive_ptr<RRef>.
            .put(new Info("torch::distributed::rpc::RRefContext::getOrCreateRRef").skip())
            .put(new Info("torch::distributed::rpc::RRefContext::createUserRRef").skip())
            .put(new Info("torch::distributed::rpc::RRefContext::getOrCreateOwnerRRef").skip())
            .put(new Info("torch::distributed::rpc::RRefContext::createOwnerRRef").skip())
            .put(new Info("torch::distributed::rpc::RRefContext").purify().pointerTypes("RRefContext").virtualize())
            // destroyInstance returns std::vector<c10::intrusive_ptr<RRef>>; we
            // don't bind vector-of-intrusive_ptr combinations, skip the helper.
            // Match the parser's normalized key (templates stripped, const stripped).
            .put(new Info("torch::distributed::rpc::RRefContext::destroyInstance(bool)").skip())

            // withStorages is a free function (inline in message.h) returning
            // std::tuple<intrusive_ptr<Message>, vector<weak_intrusive_ptr<StorageImpl>>>.
            // Not worth binding vector-of-weak_intrusive_ptr in the parent's
            // pointer types — skip the helper.
            .put(new Info("torch::distributed::rpc::withStorages(c10::intrusive_ptr<torch::distributed::rpc::Message>)").skip())
        ;

        //--- RpcAgent hierarchy --------------------------------------------------
        infoMap
            .put(new Info("torch::distributed::rpc::RpcAgent").purify().pointerTypes("RpcAgent").virtualize())
            .put(new Info("torch::distributed::rpc::RpcAgent::RpcAgent").skip())     // protected
            .put(new Info("torch::distributed::rpc::RpcAgent::send").virtualize())
            .put(new Info("torch::distributed::rpc::RpcAgent::getWorkerInfo",
                          "torch::distributed::rpc::RpcAgent::getWorkerInfoByName",
                          "torch::distributed::rpc::RpcAgent::getWorkerInfos",
                          "torch::distributed::rpc::RpcAgent::join",
                          "torch::distributed::rpc::RpcAgent::sync",
                          "torch::distributed::rpc::RpcAgent::startImpl",
                          "torch::distributed::rpc::RpcAgent::shutdownImpl",
                          "torch::distributed::rpc::RpcAgent::getMetrics",
                          "torch::distributed::rpc::RpcAgent::getDebugInfo",
                          "torch::distributed::rpc::RpcAgent::getDeviceMap",
                          "torch::distributed::rpc::RpcAgent::getDevices",
                          "torch::distributed::rpc::RpcAgent::addGilWaitTime").virtualize())

            .put(new Info("torch::distributed::rpc::TensorPipeAgent")
                    .purify().pointerTypes("TensorPipeAgent").virtualize())
            .put(new Info("torch::distributed::rpc::TensorPipeAgent::TensorPipeAgent").skip())
            // updateGroupMembership uses `std::unordered_map<std::string, rpc::DeviceMap>`
            // which we don't bind.
            .put(new Info("torch::distributed::rpc::TensorPipeAgent::updateGroupMembership").skip())
            // pipeRead/pipeWrite templates are unsafe to bind generically.
            .put(new Info("torch::distributed::rpc::TensorPipeAgent::pipeRead",
                          "torch::distributed::rpc::TensorPipeAgent::pipeWrite",
                          "torch::distributed::rpc::TensorPipeAgent::usingUnixPipe",
                          "torch::distributed::rpc::TensorPipeAgent::usingShmPipe",
                          "torch::distributed::rpc::TensorPipeAgent::usingTransport",
                          "torch::distributed::rpc::TensorPipeAgent::getTransportName",
                          "torch::distributed::rpc::TensorPipeAgent::getChannelName",
                          "torch::distributed::rpc::TensorPipeAgent::getSrcOrDstRank").skip())
        ;

        //--- intrusive_ptr<...> for RPC classes ---------------------------------
        new PointerInfo("torch::distributed::rpc::RpcCommandBase")
                .javaBaseName("RpcCommandBase").makeIntrusive(infoMap);
        new PointerInfo("torch::distributed::rpc::Message").makeIntrusive(infoMap);

        //--- Skip tensorpipe internals (heavy template metaprogramming) ---------
        infoMap
            .put(new Info("tensorpipe::Error",
                          "tensorpipe::Listener",
                          "tensorpipe::Pipe",
                          "tensorpipe::Context",
                          "tensorpipe::Context::join",
                          "tensorpipe::Context::listen",
                          "tensorpipe::Context::connect",
                          "tensorpipe::Context::registerTransport",
                          "tensorpipe::Context::registerChannel").skip())
        ;
    }
}
