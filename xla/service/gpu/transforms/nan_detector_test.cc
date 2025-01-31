#include "xla/service/gpu/transforms/nan_detector.h"

#include <memory>
#include <optional>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/layout_assignment.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

class NanDetectorTest : public HloTestBase {
 public:
  NanDetectorTest()
      : HloTestBase(/*verifier_layout_sensitive=*/true,
                    /*allow_mixed_precision_in_hlo_verifier=*/true,
                    LayoutAssignment::InstructionCanChangeLayout) {}
  void CheckNanDetectionRewrite(absl::string_view hlo,
                                std::optional<absl::string_view> expected) {
    RunAndFilecheckHloRewrite(hlo, NanDetector{}, expected);
  }
};

TEST_F(NanDetectorTest, AppendNanDetectionCalls) {
  constexpr absl::string_view hlo_string = R"(
HloModule main
max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}
add_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT add = f32[] add(arg_0, arg_1)
}
ENTRY main {
  param_0 = f32[127,125]{1,0} parameter(0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f32[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = f32[127,125]{1,0} subtract(param_0, broadcast)
})";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();

  EXPECT_TRUE(NanDetector().Run(module.get()).value());
  LOG(ERROR) << module->ToString();
}

}  // namespace
}  // namespace xla::gpu
