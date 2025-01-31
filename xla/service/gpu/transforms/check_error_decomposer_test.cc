#include "xla/service/gpu/transforms/check_error_decomposer.h"

#include <optional>

#include "absl/strings/string_view.h"
#include "xla/service/layout_assignment.h"
#include "xla/tests/hlo_test_base.h"

namespace xla::gpu {
namespace {

class CheckErrorDecomposerTest : public HloTestBase {
 public:
  CheckErrorDecomposerTest()
      : HloTestBase(/*verifier_layout_sensitive=*/true,
                    /*allow_mixed_precision_in_hlo_verifier=*/true,
                    LayoutAssignment::InstructionCanChangeLayout) {}
  void CheckErrorDecomposerRewrite(absl::string_view hlo,
                                   std::optional<absl::string_view> expected) {
    RunAndFilecheckHloRewrite(hlo, CheckErrorDecomposer{}, expected);
  }
};

TEST_F(CheckErrorDecomposerTest, DecomposeCheckErrorCustomCall) {
  constexpr absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  param_0 = f32[128,128]{1,0} parameter(0)
  param_1 = f32[128,128]{1,0} parameter(1)
  div = f32[128,128]{1,0} divide(param_0, param_1)
  compare = pred[128,128]{1,0} compare(div, div), direction=NE
  ROOT custom-call = f32[128,128]{1,0} custom-call(div, compare), custom_call_target="__xla_check_error", api_version=API_VERSION_TYPED_FFI
})";

  CheckErrorDecomposerRewrite(hlo_string, R"(
  CHECK: %reduce_logical_or (lhs: pred[], rhs: pred[]) -> pred[] {
  CHECK: %div = f32[128,128]{1,0} divide(%param_0, %param_1)
  CHECK: %compare = pred[128,128]{1,0} compare(%div, %div), direction=NE
  CHECK: %constant = pred[] constant(false)
  CHECK: %reduce = pred[] reduce(%compare, %constant), dimensions={0,1}, to_apply=%reduce_logical_or
  CHECK: %custom-call.1 = pred[] custom-call(%reduce), custom_call_target="__xla_check_error",
  CHECK: %after-all = token[] after-all(%custom-call.1)
  CHECK: ROOT %add-dependency = f32[128,128]{1,0} add-dependency(%div, %after-all)
  })");
}

}  // namespace
}  // namespace xla::gpu
