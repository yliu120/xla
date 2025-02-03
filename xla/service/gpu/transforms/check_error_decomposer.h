#ifndef XLA_SERVICE_GPU_TRANSFORMS_CHECK_ERROR_DECOMPOSER_H_
#define XLA_SERVICE_GPU_TRANSFORMS_CHECK_ERROR_DECOMPOSER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla::gpu {

// Decomposes error checking custom calls.
class CheckErrorDecomposer : public HloModulePass {
 public:
  absl::string_view name() const override { return "check_error_decomposer"; }
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_TRANSFORMS_CHECK_ERROR_DECOMPOSER_H_
