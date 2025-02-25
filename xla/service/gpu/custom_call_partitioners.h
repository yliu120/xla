#ifndef XLA_SERVICE_GPU_CUSTOM_CALL_PARTITIONERS_H_
#define XLA_SERVICE_GPU_CUSTOM_CALL_PARTITIONERS_H_

#include <optional>
#include <vector>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/service/custom_call_sharding_helper.h"
#include "xla/service/spmd/spmd_partitioner.h"

namespace xla::gpu::spmd {

constexpr char kSendCustomCall[] = "xla.gpu.send";
constexpr char kRecvCustomCall[] = "xla.gpu.recv";
constexpr char kFollowMeCustomCall[] = "FollowMe";

// Custom-call partitioner GPU send/recv custom calls.
class PassThroughPartitioner : public CustomCallPartitioner {
 public:
  bool IsCustomCallShardable(const HloInstruction* instruction) const override {
    return true;
  }

  absl::Status Partition(xla::spmd::SpmdPartitioningVisitor* partitioner,
                         HloInstruction* hlo) const override {
    // This seems to be enough as of now for testing.
    // In the future, we may need to be careful here to make sure send and
    // recv are partitioned in the same way, otherwise the result will be
    // incorrect.
    // if (hlo->custom_call_target() == kAfterAllCustomCall) {
    //   std::vector<HloInstruction*> new_operands;
    //   for (HloInstruction* operand : hlo->operands()) {
    //     new_operands.push_back(partitioner->GetPartitionedHlo(operand).hlo());
    //   }
    //   partitioner->SetPartitionedHlo(hlo, [&] {
    //     return partitioner->builder()->AddInstruction(
    //         hlo->CloneWithNewOperands(hlo->shape(), new_operands));
    //   });
    //   return absl::OkStatus();
    // }
    return partitioner->HandleElementwise(hlo);
  }

  // This allows replicated sharding on custom-call op to pass checks at spmd
  // partitioner preprocess stage.
  bool CanSideEffectingHaveReplicatedSharding() const override { return true; }

  // Run through this custom call for both forward and backwnard propagation.
  std::optional<HloSharding> InferShardingFromOperands(
      const HloInstruction* instruction) const override {
    if (instruction->operand(0)->has_sharding()) {
      return instruction->operand(0)->sharding();
    }
    return std::nullopt;
  }
};

}  // namespace xla::gpu::spmd

#endif  // XLA_SERVICE_GPU_CUSTOM_CALL_PARTITIONERS_H_
