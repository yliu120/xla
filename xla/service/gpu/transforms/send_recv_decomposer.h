#ifndef XLA_SERVICE_GPU_TRANSFORMS_SEND_RECV_DECOMPOSER_H_
#define XLA_SERVICE_GPU_TRANSFORMS_SEND_RECV_DECOMPOSER_H_

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla::gpu {

class SendRecvDecomposer : public HloModulePass {
 public:
  SendRecvDecomposer() = default;
  absl::string_view name() const override { return "send-recv-decomposer"; }

  // Decomposes send/recv instructions into send/send-done and recv/recv-done
  // instructions.
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_TRANSFORMS_SEND_RECV_DECOMPOSER_H_
