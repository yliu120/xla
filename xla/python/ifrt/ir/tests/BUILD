load("//xla:lit.bzl", "enforce_glob", "lit_test_suite")
load("//xla:xla.bzl", "xla_cc_binary", "xla_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

lit_test_suite(
    name = "all_tests",
    srcs = enforce_glob(
        [
            "ifrt_duplicated_callee_elimination.mlir",
            "ifrt_merge_reshards.mlir",
            "ifrt_outline_atom_program_to_module.mlir",
            "ifrt_verify_donation.mlir",
            "ifrt_verify_sharding_specified.mlir",
            "spmd_expansion.mlir",
            "spmd_interface_verification.mlir",
            "verify_array.mlir",
            "verify_assemble.mlir",
            "verify_attrs.mlir",
            "verify_call.mlir",
            "verify_call_loaded_executable.mlir",
            "verify_copy_arrays.mlir",
            "verify_disassemble.mlir",
            "verify_loaded_executable.mlir",
            "verify_remap_arrays.mlir",
            "verify_reshard.mlir",
        ],
        include = [
            "*.mlir",
        ],
    ),
    cfg = "//xla:lit.cfg.py",
    tools = [
        ":ifrt-opt",
        "@llvm-project//llvm:FileCheck",
    ],
)

xla_cc_binary(
    name = "ifrt-opt",
    srcs = ["ifrt-opt.cc"],
    deps = [
        "//xla/mlir_hlo:hlo_dialect_registration",
        "//xla/python/ifrt/ir",
        "//xla/python/ifrt/ir/transforms:built_in_spmd_expansions",
        "//xla/python/ifrt/ir/transforms:passes",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirOptLib",
    ],
)

cc_library(
    name = "executable_impl_test_base",
    testonly = True,
    srcs = ["executable_impl_test_base.cc"],
    hdrs = ["executable_impl_test_base.h"],
    visibility = ["//xla/python/ifrt:friends"],
    deps = [
        "//xla:status_macros",
        "//xla/mlir/utils:error_util",
        "//xla/mlir_hlo:hlo_dialect_registration",
        "//xla/python/ifrt",
        "//xla/python/ifrt:test_util",
        "//xla/python/ifrt/ir",
        "//xla/python/ifrt/ir:sharding_param",
        "//xla/python/ifrt/ir/transforms:built_in_spmd_expansions",
        "//xla/tsl/concurrency:ref_count",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "executable_impl_test_lib",
    testonly = True,
    srcs = ["executable_impl_test_lib.cc"],
    visibility = ["//xla/python/ifrt:friends"],
    deps = [
        ":executable_impl_test_base",
        "//xla/pjrt:pjrt_executable",
        "//xla/python/ifrt",
        "//xla/python/ifrt:test_util",
        "//xla/python/ifrt/hlo:hlo_program",
        "//xla/python/ifrt/ir:ifrt_ir_program",
        "//xla/python/ifrt/ir:sharding_param",
        "//xla/python/pjrt_ifrt:xla_ifrt",
        "//xla/service:computation_placer_hdr",
        "//xla/tsl/concurrency:ref_count",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@tsl//tsl/platform:status_matchers",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/platform:test",
    ],
    alwayslink = True,
)

xla_cc_test(
    name = "executable_test_no_impl",
    srcs = [],
    deps = [
        ":executable_impl_test_lib",
        "//xla/python/ifrt:no_impl_test_main",
    ],
)
