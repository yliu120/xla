#include <array>

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"  // IWYU pragma: keep
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/gpus/cuda/include/driver_types.h"
#endif

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream.h"
#include "xla/xla_data.pb.h"
#include "xla/tsl/platform/errors.h"

#if GOOGLE_CUDA
#define gpuSuccess cudaSuccess
#define gpuMemcpyAsync cudaMemcpyAsync
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#endif

namespace xla {
namespace {

// Fails loud when the input pred becomes true, which indicates NANs detected.
// An output buffer is used here for the sake of wiring this custom call
// naturally into the graph. The value of the output is unset.
static absl::Status ReportNan(
    se::Stream* stream, ffi::Buffer<PrimitiveType::PRED> src,
    absl::string_view metadata,
    ffi::Result<ffi::Buffer<PrimitiveType::PRED>> unused) {
  se::DeviceMemoryBase src_mem = src.device_memory();
  bool is_nan = false;
  std::array<bool, 1> host_buffer;
  auto memcpy_status = stream->MemcpyD2H(se::DeviceMemory<bool>(src_mem),
                                         absl::MakeSpan(host_buffer));
  if (!memcpy_status.ok()) {
    return absl::InternalError(
        "Unable to memcpy bool from device to host in report nan.");
  }
  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
  if (host_buffer[0]) {
    return absl::DataLossError(absl::StrFormat("NAN found in %s.", metadata));
  }
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kReportNan, ReportNan,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Arg<ffi::Buffer<PrimitiveType::PRED>>()
                           .Attr<absl::string_view>("metadata")
                           .Ret<ffi::Buffer<PrimitiveType::PRED>>(),  // dst
                       {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_report_nan", "CUDA",
                         kReportNan);

}  // namespace
}  // namespace xla
