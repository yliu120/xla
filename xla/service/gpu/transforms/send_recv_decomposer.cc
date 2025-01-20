#include "xla/service/gpu/transforms/send_recv_decomposer.h"

#include <cstdint>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/strings/str_cat.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/literal_util.h"
#include "xla/literal.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"

namespace xla::gpu {
namespace {

constexpr absl::string_view kSendCustomCallName = "xla.gpu.send";
constexpr absl::string_view kRecvCustomCallName = "xla.gpu.recv";
constexpr absl::string_view kPermAttrName = "perm";

// TODO: Uses upstream class after rebasing.
using SourceTargetPairs = std::vector<std::pair<int64_t, int64_t>>;

struct DecomposeContext {
  // Cache an or computation for reduction added when decomposing recv.
  HloComputation* or_computation;
};

std::string SourceTargetPairsString(
    const SourceTargetPairs& source_target_pairs) {
  auto formatter = absl::PairFormatter(
      [](std::string* out, int64_t value) { absl::StrAppend(out, "{", value); },
      ",",
      [](std::string* out, int64_t value) {
        absl::StrAppend(out, value, "}");
      });
  const std::string pairs_str =
      absl::StrJoin(source_target_pairs, ",", formatter);
  return absl::StrCat("{", pairs_str, "}");
}

SourceTargetPairs ParseSourceTargetMapFromBackendConfig(
    absl::string_view backend_config) {
  std::vector<std::pair<int64_t, int64_t>> source_target_pairs;
  CHECK(!backend_config.empty())
      << "Backend config empty in xla.gpu.send/recv custom call.";
  mlir::MLIRContext context;
  mlir::Attribute attr = mlir::parseAttribute(backend_config, &context);
  auto dict = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(attr);
  CHECK(dict != nullptr)
      << "Backend config was set as an MLIR dict attr in JAX.";
  CHECK(dict.contains(kPermAttrName))
      << "Backend config does not contain the perm attribute.";

  auto array_attr =
      dict.get(kPermAttrName).dyn_cast<mlir::DenseIntElementsAttr>();
  CHECK(array_attr != nullptr)
      << "Perm attribute is not a dense int elements attr.";
  auto shape = array_attr.getShapedType().getShape();
  // Perm array needs to have a shape of [][2]
  CHECK(shape.size() == 2 && shape[1] == 2);
  for (int64_t i = 0; i < shape[0]; i++) {
    auto source_value = array_attr.getValues<mlir::IntegerAttr>()[i * 2];
    auto target_value = array_attr.getValues<mlir::IntegerAttr>()[i * 2 + 1];
    source_target_pairs.push_back(
        std::make_pair(source_value.getValue().getSExtValue(),
                       target_value.getValue().getSExtValue()));
  }
  return source_target_pairs;
}

FrontendAttributes MakeSendRecvFrontendAttributes(
    const FrontendAttributes& old_attributes,
    const SourceTargetPairs& source_target_pairs) {
  std::string source_target_pairs_str =
      SourceTargetPairsString(source_target_pairs);
  xla::FrontendAttributes attributes;
  attributes.mutable_map()->insert(old_attributes.map().begin(),
                                   old_attributes.map().end());
  (*attributes.mutable_map())[kSendRecvSourceTargetPairsAttr] =
      source_target_pairs_str;
  return attributes;
}

HloComputation* CreateOrComputation(absl::string_view name, HloModule* module) {
  auto builder = HloComputation::Builder(name);
  Shape bool_shape = ShapeUtil::MakeShape(PRED, {});
  auto lfs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, bool_shape, "x"));
  auto rhs = builder.AddInstruction(
      HloInstruction::CreateParameter(1, bool_shape, "y"));
  builder.AddInstruction(
      HloInstruction::CreateBinary(bool_shape, HloOpcode::kOr, lfs, rhs));
  return module->AddEmbeddedComputation(builder.Build());
}

absl::Status DecomposeSend(HloCustomCallInstruction* custom_call,
                           int64_t& next_channel_id) {
  HloComputation* comp = custom_call->parent();

  auto* token = comp->AddInstruction(HloInstruction::CreateToken());
  auto* send = comp->AddInstruction(
      HloInstruction::CreateSend(custom_call->mutable_operand(0), token,
                                 next_channel_id, /*is_host_transfer=*/false));
  SourceTargetPairs source_target_pairs = ParseSourceTargetMapFromBackendConfig(
      custom_call->raw_backend_config_string());
  send->set_frontend_attributes(MakeSendRecvFrontendAttributes(
      custom_call->frontend_attributes(), source_target_pairs));
  auto* send_done = comp->AddInstruction(HloInstruction::CreateSendDone(
      send, next_channel_id++, /*is_host_transfer=*/false));
  auto* add_dependency =
      comp->AddInstruction(HloInstruction::CreateAddDependency(
          custom_call->mutable_operand(0), send_done));

  TF_RETURN_IF_ERROR(custom_call->ReplaceAllUsesWith(add_dependency));
  TF_RETURN_IF_ERROR(comp->RemoveInstruction(custom_call));
  return absl::OkStatus();
}

absl::Status DecomposeRecv(HloCustomCallInstruction* custom_call,
                           int64_t& next_channel_id,
                           DecomposeContext* context) {
  HloComputation* comp = custom_call->parent();

  auto* token = comp->AddInstruction(HloInstruction::CreateToken());
  auto* recv = comp->AddInstruction(
      HloInstruction::CreateRecv(custom_call->shape(), token, next_channel_id,
                                 /*is_host_transfer=*/false));

  SourceTargetPairs source_target_pairs = ParseSourceTargetMapFromBackendConfig(
      custom_call->raw_backend_config_string());
  recv->set_frontend_attributes(MakeSendRecvFrontendAttributes(
      custom_call->frontend_attributes(), source_target_pairs));
  auto* recv_done = comp->AddInstruction(HloInstruction::CreateRecvDone(
      recv, next_channel_id++, /*is_host_transfer=*/false));
  HloInstruction* recv_data =
      comp->AddInstruction(HloInstruction::CreateGetTupleElement(recv_done, 0),
                           absl::StrCat(custom_call->name(), "-recv-data"));

  // If the device is not in the source list, output the original output.
  std::vector<int32_t> source_devices;
  for (const auto& source_target_pair : source_target_pairs) {
    source_devices.push_back(static_cast<int32_t>(source_target_pair.second));
  }
  Literal sources_literal = LiteralUtil::CreateR1<int32_t>(source_devices);
  auto* constant = comp->AddInstruction(
      HloInstruction::CreateConstant(std::move(sources_literal)));
  auto* partition_id =
      comp->AddInstruction(HloInstruction::CreatePartitionId());
  auto* convert = comp->AddInstruction(HloInstruction::CreateConvert(
      ShapeUtil::MakeScalarShape(S32), partition_id));
  auto* broadcast = comp->AddInstruction(HloInstruction::CreateBroadcast(
      constant->shape(), convert, /*broadcast_dimensions=*/{}));
  auto* compare = comp->AddInstruction(HloInstruction::CreateCompare(
      ShapeUtil::MakeShape(PRED, broadcast->shape().dimensions()), broadcast,
      constant, ComparisonDirection::kEq));

  if (context->or_computation == nullptr) {
    context->or_computation = CreateOrComputation(
        absl::StrCat("decomposed_recv", "_or"), comp->parent());
  }
  auto* init_value = comp->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  auto* reduce = comp->AddInstruction(HloInstruction::CreateReduce(
      ShapeUtil::MakeScalarShape(PRED), compare, /*init_value=*/init_value,
      /*dimensions_to_reduce=*/{0},
      /*reduce_computation=*/context->or_computation));
  auto* select = comp->AddInstruction(HloInstruction::CreateTernary(
      recv_data->shape(), HloOpcode::kSelect, reduce, recv_data,
      custom_call->mutable_operand(0)));

  TF_RETURN_IF_ERROR(custom_call->ReplaceAllUsesWith(select));
  TF_RETURN_IF_ERROR(comp->RemoveInstruction(custom_call));
  return absl::OkStatus();
}
}  // namespace

absl::StatusOr<bool> SendRecvDecomposer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool modified = false;
  int64_t next_channel_id = hlo_query::NextChannelId(*module);
  auto context = std::make_unique<DecomposeContext>();
  for (HloComputation* computation : module->MakeComputationPostOrder()) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() != HloOpcode::kCustomCall) {
        continue;
      }

      auto* custom_call = Cast<HloCustomCallInstruction>(instruction);
      if (custom_call->custom_call_target() == kSendCustomCallName) {
        TF_RETURN_IF_ERROR(DecomposeSend(custom_call, next_channel_id));
        modified = true;
      } else if (custom_call->custom_call_target() == kRecvCustomCallName) {
        TF_RETURN_IF_ERROR(
            DecomposeRecv(custom_call, next_channel_id, context.get()));
        modified = true;
      }
    }
  }

  if (modified) {
    HloDCE dce;
    TF_RETURN_IF_ERROR(dce.Run(module).status());
  }
  return modified;
}
}  // namespace xla::gpu
