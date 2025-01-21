#include "xla/service/gpu/transforms/send_recv_decomposer.h"

#include <cstdint>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/strings/str_cat.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/tsl/platform/errors.h"

namespace xla::gpu {
namespace {

constexpr absl::string_view kSendCustomCallName = "xla.gpu.send";
constexpr absl::string_view kRecvCustomCallName = "xla.gpu.recv";
constexpr absl::string_view kPermAttrName = "perm";

// TODO: Uses upstream class after rebasing.
using SourceTargetPairs = std::vector<std::pair<int64_t, int64_t>>;

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
                           int64_t& next_channel_id) {
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

  TF_RETURN_IF_ERROR(custom_call->ReplaceAllUsesWith(recv_data));
  TF_RETURN_IF_ERROR(comp->RemoveInstruction(custom_call));
  return absl::OkStatus();
}
}  // namespace

absl::StatusOr<bool> SendRecvDecomposer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool modified = false;
  int64_t next_channel_id = hlo_query::NextChannelId(*module);
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
        TF_RETURN_IF_ERROR(DecomposeRecv(custom_call, next_channel_id));
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
