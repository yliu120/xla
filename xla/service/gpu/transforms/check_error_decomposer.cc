#include "xla/service/gpu/transforms/check_error_decomposer.h"

#include <cstdint>

#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/literal_util.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/default/statusor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

constexpr absl::string_view kCheckErrorCustomCall = "__xla_check_error";

struct DecomposerContext {
  // Caches the reduction computation for logical or.
  HloComputation* reduce_or = nullptr;
};

std::string WriteHloMetadataToOpaque(const HloInstruction* target) {
  std::string hlo_info = absl::StrFormat("%s (%s:%d)", target->name(),
                                         target->metadata().source_file(),
                                         target->metadata().source_line());

  mlir::MLIRContext context;
  mlir::Builder builder(&context);
  llvm::SmallVector<mlir::NamedAttribute, 1> ffi_attributes;
  ffi_attributes.push_back(
      builder.getNamedAttr("metadata", builder.getStringAttr(hlo_info)));
  auto dictionary_attr = builder.getDictionaryAttr(ffi_attributes);
  std::string result;
  llvm::raw_string_ostream sstream(result);
  dictionary_attr.print(sstream);
  sstream.flush();
  return result;
}

HloComputation* ReductionComputation(HloModule* module) {
  HloComputation::Builder builder("reduce_logical_or");
  const Shape bool_shape = ShapeUtil::MakeShape(PRED, {});
  HloInstruction* lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, bool_shape, "lhs"));
  HloInstruction* rhs = builder.AddInstruction(
      HloInstruction::CreateParameter(1, bool_shape, "rhs"));
  builder.AddInstruction(
      HloInstruction::CreateBinary(bool_shape, HloOpcode::kOr, lhs, rhs));

  // Build and return the computation object
  return module->AddEmbeddedComputation(builder.Build());
}

const HloInstruction* FindOriginalTarget(const HloInstruction* to_check) {
  if (to_check->opcode() != HloOpcode::kGetTupleElement) {
    return to_check;
  }
  return to_check->operand(0);
}

absl::StatusOr<bool> DecomposeCheckErrorCustomCall(
    HloCustomCallInstruction* instr, DecomposerContext* context) {
  CHECK_EQ(instr->operand_count(), 2)
      << "check error custom call should have 2 operands (value, pred).";
  auto* target = instr->mutable_operand(0);
  auto* pred = instr->mutable_operand(1);
  auto* parent_comp = instr->parent();

  auto* init_value = parent_comp->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  Shape scalar_pred = ShapeUtil::MakeScalarShape(PRED);
  if (context->reduce_or == nullptr) {
    context->reduce_or = ReductionComputation(parent_comp->parent());
  }
  llvm::SmallVector<int64_t, 4> reduction_dims;
  reduction_dims.resize(pred->shape().dimensions_size());
  absl::c_iota(reduction_dims, 0);
  auto* reduce = parent_comp->AddInstruction(HloInstruction::CreateReduce(
      scalar_pred, pred, init_value,
      /*dimensions_to_reduce*/ reduction_dims, context->reduce_or));
  auto* custom_call =
      parent_comp->AddInstruction(HloInstruction::CreateCustomCall(
          scalar_pred, {reduce}, kCheckErrorCustomCall,
          WriteHloMetadataToOpaque(FindOriginalTarget(target)),
          CustomCallApiVersion::API_VERSION_TYPED_FFI));
  custom_call->set_output_to_operand_aliasing({{{}, {0, {}}}});

  // Chains the custom call into the graph naturally.
  auto* after_all = parent_comp->AddInstruction(
      HloInstruction::CreateAfterAll({custom_call}));
  auto* add_dependency = parent_comp->AddInstruction(
      HloInstruction::CreateAddDependency(target, after_all));
  TF_RETURN_IF_ERROR(instr->ReplaceAllUsesWith(add_dependency));
  return true;
}

}  // namespace

absl::StatusOr<bool> CheckErrorDecomposer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(2, "Before decomposing:\n" + module->ToString());
  bool changed = false;

  DecomposerContext context;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    for (HloInstruction* instruction : computation->MakeInstructionPostOrder()) {
      if (!instruction->IsCustomCall(kCheckErrorCustomCall)) {
        continue;
      }

      auto* custom_call = Cast<HloCustomCallInstruction>(instruction); 
      VLOG(2) << "Decomposing custom call: " << custom_call->ToShortString();
      TF_ASSIGN_OR_RETURN(bool is_decomposed,
                          DecomposeCheckErrorCustomCall(custom_call, &context));
      changed |= is_decomposed;
    }
  }

  if (changed) {
    HloDCE dce;
    TF_RETURN_IF_ERROR(dce.Run(module).status());
  }
  XLA_VLOG_LINES(2, "After decomposing:\n" + module->ToString());
  return changed;
}

}  // namespace xla::gpu
