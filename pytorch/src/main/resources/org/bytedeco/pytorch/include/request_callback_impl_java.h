/*
 * Parse-time shim for torch::distributed::rpc::RequestCallbackImpl.
 *
 * The real header (request_callback_impl.h) pulls in
 * torch/csrc/jit/python/pybind.h → Python.h + pybind11 + THP/DynamicTypes.
 * That is fine for the JNI compile unit (which has Python include paths),
 * but during JavaCPP *parse* we only need the class layout and method
 * signatures that do not take pybind11::object.
 *
 * runPythonFunction(const py::object&, ...) is intentionally omitted here
 * and skipped in InfoMap — it cannot be called from pure Java without a
 * live CPython interpreter and a mapped py::object type.
 *
 * The real request_callback_impl.h is still listed in @Platform.include
 * so jnitorch_rpc.cpp gets a complete definition at compile time.
 */
#pragma once

#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/request_callback_no_python.h>
#include <torch/csrc/distributed/rpc/rpc_command_base.h>

namespace torch::distributed::rpc {

class TORCH_API RequestCallbackImpl : public RequestCallbackNoPython {
 public:
  std::unique_ptr<RpcCommandBase> deserializePythonRpcCommand(
      std::unique_ptr<RpcCommandBase> rpc,
      const MessageType& messageType) const override;

  c10::intrusive_ptr<JitFuture> processPythonCall(
      RpcCommandBase& rpc,
      const std::vector<c10::Stream>& streams) const override;

  c10::intrusive_ptr<JitFuture> processScriptCall(
      RpcCommandBase& rpc,
      const std::vector<c10::Stream>& streams) const override;

  c10::intrusive_ptr<JitFuture> processScriptRemoteCall(
      RpcCommandBase& rpc,
      const std::vector<c10::Stream>& streams) const override;

  c10::intrusive_ptr<JitFuture> processPythonRemoteCall(
      RpcCommandBase& rpc,
      const std::vector<c10::Stream>& streams) const override;

  c10::intrusive_ptr<JitFuture> processPythonRRefFetchCall(
      RpcCommandBase& rpc) const override;

  void handleRRefDelete(c10::intrusive_ptr<RRef>& rref) const override;

  c10::intrusive_ptr<JitFuture> processRpcWithErrors(
      RpcCommandBase& rpc,
      const MessageType& messageType,
      const std::vector<c10::Stream>& streams) const override;

  bool cudaAvailable() const override;

  c10::intrusive_ptr<JitFuture> processRRefBackward(
      RpcCommandBase& rpc) const override;

  c10::intrusive_ptr<JitFuture> runJitFunction(
      const c10::QualifiedName& name,
      std::vector<at::IValue>& stack,
      const std::vector<c10::Stream>& streams,
      bool isAsyncExecution) const;

  // runPythonFunction(const py::object&, ...) — declared only in the real
  // header; skipped from JavaCPP mapping (pybind11::object).
};

} // namespace torch::distributed::rpc
