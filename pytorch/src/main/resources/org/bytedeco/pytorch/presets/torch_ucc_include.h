// Parse list for torch_ucc preset (c10d::ProcessGroupUCC).
// Requires -DUSE_C10D_UCC and <ucc/api/ucc.h> on the include path
// (e.g. cppbuild/deps/install/include from a UCC source checkout).
#include "torch/csrc/distributed/c10d/UCCUtils.hpp"
#include "torch/csrc/distributed/c10d/ProcessGroupUCC.hpp"
