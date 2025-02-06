#include "xla/service/gpu/transforms/send_recv_decomposer.h"

#include <memory>

#include <gtest/gtest.h>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

using SendRecvDecomposerTest = HloTestBase;

TEST_F(SendRecvDecomposerTest, TestSendRecvDecompose) {
  constexpr absl::string_view kHloString = R"(
HloModule jit_g, entry_computation_layout={()->f32[2]{0}}, num_partitions=4

ENTRY main.19_spmd {
  constant = f32[] constant(1)
  broadcast.0 = f32[2]{0} broadcast(constant), dimensions={}
  copy.2 = f32[2]{0} copy(broadcast.0)
  copy.3 = f32[2]{0} copy(copy.2)
  custom-call.10 = f32[2]{0} custom-call(copy.3, copy.3), custom_call_target="xla.gpu.send", api_version=API_VERSION_TYPED_FFI, backend_config={perm = dense<[[2, 1], [0, 3]]> : tensor<2x2xi64>}
  custom-call.13 = f32[2]{0} custom-call(custom-call.10), custom_call_target="xla.gpu.recv", api_version=API_VERSION_TYPED_FFI, backend_config={perm = dense<[[2, 1], [0, 3]]> : tensor<2x2xi64>}
  custom-call.14 = f32[2]{0} custom-call(custom-call.13), custom_call_target="xla.gpu.zeros", api_version=API_VERSION_TYPED_FFI
  add.2 = f32[2]{0} add(copy.3, custom-call.14)
  custom-call.17 = f32[2]{0} custom-call(add.2, add.2), custom_call_target="xla.gpu.send", api_version=API_VERSION_TYPED_FFI, backend_config={perm = dense<[[1, 0], [3, 2]]> : tensor<2x2xi64>}
  custom-call.18 = f32[2]{0} custom-call(custom-call.17), custom_call_target="xla.gpu.recv", api_version=API_VERSION_TYPED_FFI, backend_config={perm = dense<[[1, 0], [3, 2]]> : tensor<2x2xi64>}
  add.3 = f32[2]{0} add(custom-call.13, custom-call.18)
  copy.4 = f32[2]{0} copy(add.3)
  ROOT copy.5 = f32[2]{0} copy(copy.4)
} // main.19_spmd
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  SendRecvDecomposer pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);
}

}  // namespace
}  // namespace xla::gpu
