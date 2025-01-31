#include "xla/service/gpu/transforms/nan_detector.h"

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/literal_util.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

constexpr absl::string_view kNanDetectionCustomCall = "__xla_report_nan";

struct NanDetectionContext {
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

// Adds instructions for nan detection.
absl::Status DetectNan(HloInstruction* target, NanDetectionContext* context) {
  CHECK(!target->shape().IsTuple());
  HloComputation* comp = target->parent();

  auto original_users = target->users();
  auto* compare = comp->AddInstruction(HloInstruction::CreateCompare(
      ShapeUtil::MakeShape(PRED, target->shape().dimensions()), target, target,
      Comparison::Direction::kNe));
  auto* init_value = comp->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  Shape scalar_pred = ShapeUtil::MakeScalarShape(PRED);

  if (context->reduce_or == nullptr) {
    context->reduce_or = ReductionComputation(comp->parent());
  }
  llvm::SmallVector<int64_t, 4> reduction_dims;
  reduction_dims.resize(compare->shape().dimensions_size());
  absl::c_iota(reduction_dims, 0);
  auto* reduce = comp->AddInstruction(HloInstruction::CreateReduce(
      scalar_pred, compare, init_value,
      /*dimensions_to_reduce*/ reduction_dims, context->reduce_or));
  auto* custom_call = comp->AddInstruction(HloInstruction::CreateCustomCall(
      scalar_pred, {reduce}, kNanDetectionCustomCall,
      WriteHloMetadataToOpaque(target),
      CustomCallApiVersion::API_VERSION_TYPED_FFI));
  custom_call->set_output_to_operand_aliasing({{{}, {0, {}}}});
  auto* after_all =
      comp->AddInstruction(HloInstruction::CreateAfterAll({custom_call}));
  auto* add_dependency = comp->AddInstruction(
      HloInstruction::CreateAddDependency(target, after_all));

  TF_RETURN_IF_ERROR(target->ReplaceUsesWith(original_users, add_dependency));
  return absl::OkStatus();
}

class NanDetectionVisitor : public DfsHloRewriteVisitor {
 public:
  absl::Status HandleWhile(HloInstruction* instr) override {
    NanDetectionVisitor visitor;
    visitor.context_ = this->context_;
    TF_RETURN_IF_ERROR(instr->while_body()->Accept(&visitor));
    if (visitor.changed()) {
      MarkAsChanged();
    }
    return absl::OkStatus();
  }
  absl::Status HandleCustomCall(HloInstruction* instr) override {
    return MaybeDetectNan(instr);
  }
  absl::Status HandleDot(HloInstruction* instr) override {
    return MaybeDetectNan(instr);
  }
  absl::Status HandleFusion(HloInstruction* instr) override {
    return MaybeDetectNan(instr);
  }
  absl::Status HandleGetTupleElement(HloInstruction* instr) override {
    if (gte_.contains(instr)) {
      return MaybeDetectNan(instr);
    }
    return absl::OkStatus();
  }
  absl::Status HandleReduce(HloInstruction* instr) override {
    return MaybeDetectNan(instr);
  }
  absl::Status HandleElementwiseUnary(HloInstruction* instr) override {
    switch (instr->opcode()) {
      case HloOpcode::kExp:
      case HloOpcode::kLog:
      case HloOpcode::kLog1p:
        return MaybeDetectNan(instr);
      default:
        return DefaultAction(instr);
    }
  }
  absl::Status HandleElementwiseBinary(HloInstruction* instr) override {
    if (instr->opcode() == HloOpcode::kDivide) {
      return MaybeDetectNan(instr);
    }
    return DefaultAction(instr);
  }

 private:
  absl::Status MaybeDetectNan(HloInstruction* instr) {
    if (instr->shape().IsTuple()) {
      // Delays examine tuple output to get-tuple-elements.
      for (auto* user : instr->users()) {
        if (user->opcode() == HloOpcode::kGetTupleElement &&
            !user->shape().IsTuple()) {
          gte_.insert(user);
        }
      }
      return absl::OkStatus();
    }

    if (!primitive_util::IsFloatingPointType(instr->shape().element_type())) {
      return absl::OkStatus();
    }

    TF_RETURN_IF_ERROR(DetectNan(instr, &context_));
    MarkAsChanged();
    return absl::OkStatus();
  }

  absl::flat_hash_set<HloInstruction*> gte_;
  NanDetectionContext context_;
};

}  // namespace

absl::StatusOr<bool> NanDetector::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(2, "Before nan detection:\n" + module->ToString());
  bool changed = false;

  // Recursively adding nan detection ops.
  NanDetectionVisitor visitor;
  TF_RETURN_IF_ERROR(module->entry_computation()->Accept(&visitor));

  XLA_VLOG_LINES(2, "After nan detection:\n" + module->ToString());
  return visitor.changed();
}

}  // namespace xla::gpu
