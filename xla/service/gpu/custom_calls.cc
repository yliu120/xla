#include <memory>
#include "xla/tsl/platform/default/statusor.h"
#include "xla/tsl/platform/statusor.h"

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
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/xla_data.pb.h"
#include "tsl/profiler/lib/traceme.h"

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

using ::tsl::profiler::TraceMe;

struct CheckErrorContext {
  // All custom call invocations will sequentially use this buffer.
  std::unique_ptr<se::MemoryAllocation> predicate;
};

CheckErrorContext* GetCheckErrorContext() {
  static CheckErrorContext* check_error_context = []() {
    auto context = std::make_unique<CheckErrorContext>();
    return context.release();
  }();
  return check_error_context;
}

// Fails loud when the input pred becomes true, which indicates errors detected.
// An output buffer is used here for the sake of wiring this custom call
// naturally into the graph. The value of the output is unset.
static absl::Status CheckErrorAndReport(
    se::Stream* stream, ffi::Buffer<PrimitiveType::PRED> src,
    absl::string_view metadata,
    ffi::Result<ffi::Buffer<PrimitiveType::PRED>> unused) {
  se::DeviceMemoryBase src_mem = src.device_memory();
  tsl::profiler::TraceMe trace("CheckError");
  VLOG(2) << "Trying to check error for  " << metadata;

  // Lazily allocates host memory.
  // TODO(yunlongl): Finds an elegant way to move this into FFI stages.
  auto* context = GetCheckErrorContext();
  if (context->predicate == nullptr) {
    TF_ASSIGN_OR_RETURN(context->predicate,
                        stream->parent()->HostMemoryAllocate(sizeof(bool)));
    *reinterpret_cast<bool*>(context->predicate->opaque()) = false;
  }

  // Uses pinned memory to accelerate memcpy D2H.
  auto memcpy_status = stream->MemcpyD2H(
      se::DeviceMemory<bool>(src_mem),
      absl::MakeSpan(reinterpret_cast<bool*>(context->predicate->opaque()),
                     context->predicate->size()));
  if (!memcpy_status.ok()) {
    return absl::InternalError(
        absl::StrFormat("Unable to memcpy bool from device to host: %s",
                        memcpy_status.message()));
  }
  TF_RETURN_IF_ERROR(stream->DoHostCallbackWithStatus(
      [pred = context->predicate.get(), metadata = std::string(metadata),
       device_ordinal = stream->parent()->device_ordinal()]() {
        VLOG(2) << "Report error for " << metadata;
        bool is_error = *reinterpret_cast<bool*>(pred->opaque());
        // Unfortunately there is no way to propagate errors in host callbacks.
        if (is_error) {
          return absl::DataLossError(absl::StrFormat(
              "[Device %d] Error found in %s.", device_ordinal, metadata));
        }
        return absl::OkStatus();
      }));
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kCheckErrorAndReport, CheckErrorAndReport,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Arg<ffi::Buffer<PrimitiveType::PRED>>()
                           .Attr<absl::string_view>("metadata")
                           .Ret<ffi::Buffer<PrimitiveType::PRED>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_check_error", "CUDA",
                         kCheckErrorAndReport);

}  // namespace
}  // namespace xla
