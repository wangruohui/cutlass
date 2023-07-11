/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief GEMM Grouped Example.

    This workload computes a batch of GEMM operations with distinct problem sizes. Pointers to matrices
    in Global Memory are passed to the kernel in array (also held in Global Memory). Similarly,
    leading dimensions and problem sizes are stored in arrays in GMEM.

    This differs from "Batched Array" GEMM because the size of each GEMM problem in the Grouped GEMM
    concept may be distinct.

    This benchmark program initializes a workspace with random problem sizes for a given number of
    groups. Command line options enable overriding M, N, and/or K dimensions with uniform values to
    model problems more similar to the traditional batched GEMM.

    Additionally, problem sizes are collected and binned to compute the same problem as a series of
    conventional batched GEMMs (setup for this problem is not timed). This demonstrates the performance
    enhancement achieved by implementing a specialized grouped GEMM kernel.

    Examples:

      # Runs a grouped GEMM with 100 random problem sizes
      $ ./examples/24_gemm_grouped/24_gemm_grouped --groups=100

      # Runs a grouped GEMM with 100 random problem sizes (with GEMM-K dimension equal to 1024)
      $ ./examples/24_gemm_grouped/24_gemm_grouped --groups=100 --k=1024 --verbose=true

      # Runs a grouped GEMM that is equivalent to a batched GEMM
      $ ./examples/24_gemm_grouped/24_gemm_grouped --groups=100 --m=2048 --n=1024 --k=1024 --verbose=true

      # Execute Grouped GEMM and profile with NSight
      $ nv-nsight-cu-cli ./examples/24_gemm_grouped/24_gemm_grouped --m=256 --n=256 --k=256 --verbose=true \
                                                                    --iterations=1 --reference-check=false

*/

/////////////////////////////////////////////////////////////////////////////////////////////////

#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <unordered_map>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/device/gemm_universal.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm_complex.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Result structure
struct Result {

  double runtime_ms;
  double initialization_time_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  //
  // Methods
  //

  Result(
    double runtime_ms = 0,
    double initialization_time_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess
  ):
    runtime_ms(runtime_ms), initialization_time_ms(initialization_time_ms), gflops(gflops),
    status(status), error(error), passed(true) { }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Hash function for cutlass::gemm::GemmCoord
struct HashGemmCoord {
  size_t operator()(cutlass::gemm::GemmCoord const &problem) const {
    std::hash<int> hasher;
    return (hasher(problem.m() * 3)) ^ (hasher(1 + problem.n() * 5)) ^ (hasher(2 + problem.k() * 7));
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;
  bool error;
  bool reference_check;
  bool profile_initialization;
  bool sort_problems;

  std::vector<cutlass::gemm::GemmCoord> problem_sizes;

  // problem size bins
  std::unordered_map<
    cutlass::gemm::GemmCoord,
    std::vector<int32_t>,
    HashGemmCoord> problem_bins;

  int alignment;
  int problem_count;
  int iterations;
  int cuda_streams;
  bool verbose;
  float alpha;
  float beta;
  std::string benchmark_path;

  std::string   output_tag;
  std::ofstream output_file;

  using GroupScheduleMode = cutlass::gemm::kernel::GroupScheduleMode;
  std::vector<GroupScheduleMode> scheduler_modes;

  std::unordered_map<std::string, GroupScheduleMode>
    str_to_scheduler_mode = {
      {"kDeviceOnly", GroupScheduleMode::kDeviceOnly},
      {"kHostPrecompute", GroupScheduleMode::kHostPrecompute}
    };

  struct GroupScheduleModeHash {
    size_t operator()(GroupScheduleMode m) const {
      return static_cast<size_t>(m);
    }
  };

  std::unordered_map<GroupScheduleMode, std::string, GroupScheduleModeHash>
    scheduler_mode_to_str = {
      {GroupScheduleMode::kDeviceOnly, "kDeviceOnly"},
      {GroupScheduleMode::kHostPrecompute, "kHostPrecompute"}
    };

  std::vector<GroupScheduleMode> all_scheduler_modes = {GroupScheduleMode::kDeviceOnly, GroupScheduleMode::kHostPrecompute};

  //
  // Methods
  //

  Options():
    help(false),
    error(false),
    alignment(8),
    reference_check(true),
    profile_initialization(false),
    sort_problems(false),
    problem_count(15),
    iterations(20),
    cuda_streams(0),
    verbose(false),
    alpha(1),
    beta(),
    scheduler_modes({GroupScheduleMode::kDeviceOnly})
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("alignment", alignment, 8);
    cmd.get_cmd_line_argument("groups", problem_count, 15);
    cmd.get_cmd_line_argument("alpha", alpha, 1.0f);
    cmd.get_cmd_line_argument("beta", beta, 0.0f);
    cmd.get_cmd_line_argument("iterations", iterations, 20);
    cmd.get_cmd_line_argument("streams", cuda_streams, 0);
    cmd.get_cmd_line_argument("verbose", verbose, false);
    cmd.get_cmd_line_argument("reference-check", reference_check, true);
    cmd.get_cmd_line_argument("profile-initialization", profile_initialization, false);
    cmd.get_cmd_line_argument("sort-problems", sort_problems, false);
    cmd.get_cmd_line_argument("benchmark", benchmark_path);

    std::vector<std::string> scheduler_mode_strs;
    cmd.get_cmd_line_arguments("scheduler-modes", scheduler_mode_strs);

    if (!scheduler_mode_strs.empty()) {
      scheduler_modes.clear();
      if (scheduler_mode_strs.size() == 1 && scheduler_mode_strs[0] == "all") {
        scheduler_modes = all_scheduler_modes;
      } else {
        for (std::string precomp_str : scheduler_mode_strs) {
          auto it = str_to_scheduler_mode.find(precomp_str);
          if (it != str_to_scheduler_mode.end()) {
            scheduler_modes.push_back(it->second);
          } else if (precomp_str == "all") {
            std::cerr << "Flag --scheduler-modes=all must not contain other scheduler modes in list." << std::endl;
            error = true;
            return;
          } else {
            std::cerr << "Unrecognized scheduler mode '" << precomp_str << "'" << std::endl;
            error = true;
            return;
          }
        }
      }
    }

    std::string output_path;
    cmd.get_cmd_line_argument("tag", output_tag);
    cmd.get_cmd_line_argument("output_file", output_path);

    if (!output_path.empty()) {

      std::ios_base::openmode open_mode = std::ios_base::out;

      std::ifstream input_file(output_path.c_str());

      if (input_file.good()) {
        open_mode = std::ios_base::app;
        input_file.close();
      }

      output_file.open(output_path.c_str(), open_mode);

      if (output_file.good() && open_mode != std::ios_base::app) {
        output_file << "Tag,Provider,Kind,Groups,Runtime,GFLOPs\n";
      }
    }

    // Decide how to initialize the problems
    if (!benchmark_path.empty()) {
      if (!benchmark_problems()) {
        error = true;
        problem_sizes.clear();
        return;
      }
    }
    else {
      randomize_problems(cmd);
    }

    // Post-process the problem sizes
    bin_problems();
  }

  void randomize_problems(cutlass::CommandLine &cmd) {

    //
    // For now, randomly choose the problem sizes.
    //

    int cmd_line_m = -1;
    int cmd_line_n = -1;
    int cmd_line_k = -1;

    cmd.get_cmd_line_argument("m", cmd_line_m);
    cmd.get_cmd_line_argument("n", cmd_line_n);
    cmd.get_cmd_line_argument("k", cmd_line_k);

    problem_sizes.reserve(problem_count);

    for (int i = 0; i < problem_count; ++i) {

      int m = cmd_line_m;
      int n = cmd_line_n;
      int k = cmd_line_k;

      if (m < 1) {
        m = alignment * ((rand() % 256) + 1);
      }

      if (n < 1) {
        n = alignment * ((rand() % 256) + 1);
      }

      if (k < 1) {
        k = alignment * ((rand() % 256) + 1);
      }

      cutlass::gemm::GemmCoord problem(m, n, k);

      problem_sizes.push_back(problem);
    }
  }

  /// Load a benchmark
  bool benchmark_problems() {
    std::ifstream file(benchmark_path);
    if (!file.good()) {
      return false;
    }

    while (file.good()) {

      int idx = -1;
      std::string extent_str;

      file >> idx >> extent_str;

      if (idx < 0 || extent_str.empty()) {
        break;
      }

      cutlass::gemm::GemmCoord extent;
      std::vector<std::string> tokens;

      cutlass::CommandLine::tokenize(tokens, extent_str, 'x');

      for (int i = 0; i < int(tokens.size()); ++i) {
        int x = std::atoi(tokens.at(i).c_str());

        // round up
        if (x % alignment) {
          x += (alignment - (x % alignment));
        }

        extent.at(i) = x;
      }

      if (extent.product()) {
        problem_sizes.push_back(extent);
      }
    }

    return true;
  }

  /// Post processes the problems
  void bin_problems() {

    problem_bins.clear();

    problem_count = int(problem_sizes.size());

    //
    // Insert the problem sizes into a sorted container class. This is *NOT* necessary
    // to run the CUTLASS kernel, but it enables the execution of cublas's batched GEMM.
    //
    for (int i = 0; i < int(problem_sizes.size()); ++i) {
      auto it = problem_bins.find(problem_sizes.at(i));
      if (it == problem_bins.end()) {
        problem_bins.insert({problem_sizes.at(i), std::vector<int32_t>({i}) });
      }
      else {
        it->second.push_back(i);
      }
    }
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "24_gemm_grouped\n\n"
      << "  This example profiles the performance of a 'grouped' GEMM kernel. This is similar to batched GEMM\n"
      << "  in that multiple, independent GEMMs are computed by one grid launch. It differs in that each\n"
      << "  'group' may compute a unique problem size. Problem sizes and pointers to matrices are both stored\n"
      << "  in device Global Memory and loaded by the kernel.\n\n"
      << "Options:\n\n"
      << "  --help                           If specified, displays this usage statement.\n\n"
      << "  --benchmark=<str>                Executes a benchmark problem size.\n"
      << "  --output_file=<str>              Path to a CSV file to output results. If it exists already, results are appended.\n"
      << "  --tag=<str>                      String tag to prepend to the CSV file.\n"
      << "  --groups=<int>                   Number of individual GEMM problems (default: --groups=15)\n"
      << "  --m=<int>                        Sets the M dimension for all groups. Otherwise, it is selected randomly\n"
      << "  --n=<int>                        Sets the N dimension for all groups. Otherwise, it is selected randomly\n"
      << "  --k=<int>                        Sets the K dimension for all groups. Otherwise, it is selected randomly\n"
      << "  --alpha=<f32>                    Epilogue scalar alpha (real part)\n"
      << "  --beta=<f32>                     Epilogue scalar beta (real part)\n"
      << "  --scheduler-modes=<str>          List of scheduler modes to be profile for grouped GEMM scheduler (default: --scheduler_modes=kDeviceOnly)\n"
      << "  --iterations=<int>               Number of profiling iterations to perform.\n"
      << "  --reference-check=<bool>         If true, performs reference check.\n"
      << "  --verbose=<bool>                 If true, prints problem sizes and batching structure.\n"
      << "  --profile-initialization=<bool>  If true, profiles the device-level kernel's initialization.\n"
      << "  --sort-problems=<bool>           If true, sorts problem sizes in descending order of GEMM-K dimension.\n";

    out << "\n\nExamples:\n\n"

      << "# Runs a grouped GEMM with 100 random problem sizes\n"
      << "$ ./examples/24_gemm_grouped/24_gemm_grouped --groups=100\n\n"

      << "# Runs a grouped GEMM with 100 random problem sizes (with GEMM-K dimension equal to 1024)\n"
      << "$ ./examples/24_gemm_grouped/24_gemm_grouped --groups=100 --k=1024 --verbose=true\n\n"

      << "# Runs a grouped GEMM that is equivalent to a batched GEMM\n"
      << "$ ./examples/24_gemm_grouped/24_gemm_grouped --groups=100 --m=2048 --n=1024 --k=1024 --verbose=true\n\n"

      << "# Runs a grouped GEMM with each different scheduler mode\n"
      << "$ ./examples/24_gemm_grouped/24_gemm_grouped --scheduler-modes=all\n\n"

      << "# Runs a grouped GEMM with each different scheduler mode and profiles host-side initialization time\n"
      << "$ ./examples/24_gemm_grouped/24_gemm_grouped --scheduler-modes=all --profile-initialization=true\n\n"

      << "# Runs a grouped GEMM problem given an externally supplied benchmark file. This is a text file in which\n"
      << "# Each line contains a unique group index and an MxNxK triple indicating problemsize.\n"
      << "#\n"
      << "# For example, assume the following are the contents of 'problems.txt'\n"
      << "#\n"
      << "# 0 1024x256x520\n"
      << "# 1 520x264x1024\n"
      << "# 2 96x48x1024\n"
      << "#\n"
      << "$ ./examples/24_gemm_grouped/24_gemm_grouped --benchmark=problems.txt\n\n"

      << "# Execute Grouped GEMM and profile with NSight\n"
      << "$ nv-nsight-cu-cli ./examples/24_gemm_grouped/24_gemm_grouped --m=256 --n=256 --k=256 --verbose=true --iterations=1 --reference-check=false\n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const {

    // Number of real-valued multiply-adds
    int64_t fmas = int64_t();

    for (auto const & problem : problem_sizes) {
      fmas += problem.product();
    }

    // Two flops per multiply-add
    return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////


#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/// Executes a grouped kernel and measures runtime
int impl(std::vector<torch::Tensor> const &as,
            std::vector<torch::Tensor> const &bs,
            std::vector<torch::Tensor> const &cs,
            std::vector<cutlass::gemm::GemmCoord> problem_sizes) {

  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementOutput = cutlass::half_t;

  // using ElementA = float;
  // using ElementB = float;
  // using ElementOutput = float;

  using ElementAccumulator = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;

  // cutlass::DeviceAllocation<cutlass::gemm::GemmCoord> problem_sizes_device;
  using GroupScheduleMode = cutlass::gemm::kernel::GroupScheduleMode;

  using GemmKernel = cutlass::gemm::kernel::DefaultGemmGrouped<
    ElementA,
    LayoutA,
    cutlass::ComplexTransform::kNone,
    8,
    ElementB,
    LayoutB,
    cutlass::ComplexTransform::kNone,
    8,
    ElementOutput, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator, ElementAccumulator>,
    // NOTE: Threadblock swizzling is currently not supported by CUTLASS's grouped kernels.
    // This parameter is passed in at present to match the APIs of other kernels. The parameter
    // is unused within the kernel.
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    4,
    GroupScheduleMode::kDeviceOnly>::GemmKernel;

  // using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;
  using Gemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;


  // profile
  // Result result;
  int result = -1;

  int problem_count = problem_sizes.size();
  int threadblock_count = Gemm::sufficient(problem_sizes.data(), problem_count);

  // Early exit
  if (!threadblock_count) {
    std::cout << "Active CUDA device lacks hardware resources to run CUTLASS Grouped GEMM kernel." << std::endl;
    return result;
  }

  // result.passed = false;

  // allocate();

  // construct()
  using ElementC = ElementOutput;

  cutlass::DeviceAllocation<cutlass::gemm::GemmCoord> problem_sizes_device;

  // std::vector<int64_t> offset_A;
  // std::vector<int64_t> offset_B;
  // std::vector<int64_t> offset_C;
  // std::vector<int64_t> offset_D;

  std::vector<int64_t> lda_host;
  std::vector<int64_t> ldb_host;
  std::vector<int64_t> ldc_host;
  // std::vector<int64_t> ldd_host;

  cutlass::DeviceAllocation<int64_t> lda;
  cutlass::DeviceAllocation<int64_t> ldb;
  cutlass::DeviceAllocation<int64_t> ldc;
  // cutlass::DeviceAllocation<int64_t> ldd;

  // cutlass::DeviceAllocation<ElementA> block_A;
  // cutlass::DeviceAllocation<ElementB> block_B;
  // cutlass::DeviceAllocation<ElementC> block_C;
  // cutlass::DeviceAllocation<ElementC> block_D;

  cutlass::DeviceAllocation<ElementA *> ptr_A;
  cutlass::DeviceAllocation<ElementB *> ptr_B;
  cutlass::DeviceAllocation<ElementC *> ptr_C;
  // cutlass::DeviceAllocation<ElementC *> ptr_D;

  // allocation
  // int64_t total_elements_A = 0;
  // int64_t total_elements_B = 0;
  // int64_t total_elements_C = 0;
  // int64_t total_elements_D = 0;

  lda_host.resize(problem_count);
  ldb_host.resize(problem_count);
  ldc_host.resize(problem_count);
  // ldd_host.resize(problem_count);

  for (int32_t i = 0; i < problem_count; ++i) {

    auto problem = problem_sizes.at(i);

    lda_host.at(i) = LayoutA::packed({problem.m(), problem.k()}).stride(0);
    ldb_host.at(i) = LayoutB::packed({problem.k(), problem.n()}).stride(0);
    ldc_host.at(i) = LayoutC::packed({problem.m(), problem.n()}).stride(0);
    // ldd_host.at(i) = LayoutC::packed({problem.m(), problem.n()}).stride(0);

    // offset_A.push_back(total_elements_A);
    // offset_B.push_back(total_elements_B);
    // offset_C.push_back(total_elements_C);
    // offset_D.push_back(total_elements_D);

    // int64_t elements_A = problem.m() * problem.k();
    // int64_t elements_B = problem.k() * problem.n();
    // int64_t elements_C = problem.m() * problem.n();
    // int64_t elements_D = problem.m() * problem.n();

    // total_elements_A += elements_A;
    // total_elements_B += elements_B;
    // total_elements_C += elements_C;
    // total_elements_D += elements_D;
  }

  lda.reset(problem_count);
  ldb.reset(problem_count);
  ldc.reset(problem_count);
  // ldd.reset(problem_count);

  // block_A.reset(total_elements_A);
  // block_B.reset(total_elements_B);
  // block_C.reset(total_elements_C);
  // block_D.reset(total_elements_D);


  // omit sort_problems()

  // initialize()
  problem_sizes_device.reset(problem_count);
  problem_sizes_device.copy_from_host(problem_sizes.data());

  lda.copy_from_host(lda_host.data());
  ldb.copy_from_host(ldb_host.data());
  ldc.copy_from_host(ldc_host.data());
  // ldd.copy_from_host(ldd_host.data());

  //
  // Assign pointers
  //

  std::vector<ElementA *> ptr_A_host(problem_count);
  std::vector<ElementB *> ptr_B_host(problem_count);
  std::vector<ElementC *> ptr_C_host(problem_count);
  // std::vector<ElementC *> ptr_D_host(problem_count);

  for (int32_t i = 0; i < problem_count; ++i) {
    ptr_A_host.at(i) = (ElementA*) as[i].data_ptr();
    ptr_B_host.at(i) = (ElementB*) bs[i].data_ptr();
    ptr_C_host.at(i) = (ElementC*) cs[i].data_ptr();
    // ptr_D_host.at(i) = (ElementC*) cs[i].data_ptr();
  }

  ptr_A.reset(problem_count);
  ptr_A.copy_from_host(ptr_A_host.data());

  ptr_B.reset(problem_count);
  ptr_B.copy_from_host(ptr_B_host.data());

  ptr_C.reset(problem_count);
  ptr_C.copy_from_host(ptr_C_host.data());

  // ptr_D.reset(problem_count);
  // ptr_D.copy_from_host(ptr_D_host.data());

  // Configure the GEMM arguments
  //alpha =1, beta=0
  Gemm::EpilogueOutputOp::Params epilogue_op(1.0, 0.0);

  // Configure GEMM arguments
  Gemm::Arguments args(
    problem_sizes_device.get(),
    problem_count,
    threadblock_count,
    epilogue_op,
    ptr_A.get(),
    ptr_B.get(),
    ptr_C.get(),
    ptr_C.get(),
    lda.get(),
    ldb.get(),
    ldc.get(),
    ldc.get(),
    problem_sizes.data()
  );

  // Initialize the GEMM object
  Gemm gemm;

  size_t workspace_size = gemm.get_workspace_size(args);
  cutlass::DeviceAllocation<uint8_t> workspace(workspace_size);

  auto status = gemm.initialize(args, workspace.get());

  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Failed to initialize CUTLASS Grouped GEMM kernel." << std::endl;
    return -1;
  }

  // Run the grouped GEMM object
  status = gemm();  // .run?

  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Failed to run CUTLASS Grouped GEMM kernel." << std::endl;
    return -1;
  }

  // Wait for completion
  // auto error = cudaDeviceSynchronize();

  // if (error != cudaSuccess)  {
  //   std::cerr << "Kernel execution error: " << cudaGetErrorString(error);
  //   return -1;
  // }

  return 0;
}


/// Executes a grouped kernel and measures runtime
Result impl_host(std::vector<torch::Tensor> const &as,
            std::vector<torch::Tensor> const &bs,
            std::vector<torch::Tensor> const &cs,
            std::vector<cutlass::gemm::GemmCoord> problem_sizes) {

  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementOutput = cutlass::half_t;

  // using ElementA = float;
  // using ElementB = float;
  // using ElementOutput = float;

  using ElementAccumulator = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;

  // cutlass::DeviceAllocation<cutlass::gemm::GemmCoord> problem_sizes_device;
  using GroupScheduleMode = cutlass::gemm::kernel::GroupScheduleMode;

  using GemmKernel = cutlass::gemm::kernel::DefaultGemmGrouped<
    ElementA,
    LayoutA,
    cutlass::ComplexTransform::kNone,
    8,
    ElementB,
    LayoutB,
    cutlass::ComplexTransform::kNone,
    8,
    ElementOutput, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator, ElementAccumulator>,
    // NOTE: Threadblock swizzling is currently not supported by CUTLASS's grouped kernels.
    // This parameter is passed in at present to match the APIs of other kernels. The parameter
    // is unused within the kernel.
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    4,
    GroupScheduleMode::kHostPrecompute>::GemmKernel;

  // using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;
  using Gemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;


  // profile
  Result result;

  int problem_count = problem_sizes.size();
  int threadblock_count = Gemm::sufficient(problem_sizes.data(), problem_count);

  // Early exit
  if (!threadblock_count) {
    std::cout << "Active CUDA device lacks hardware resources to run CUTLASS Grouped GEMM kernel." << std::endl;
    return result;
  }

  result.passed = false;

  // allocate();

  // construct()
  using ElementC = ElementOutput;

  cutlass::DeviceAllocation<cutlass::gemm::GemmCoord> problem_sizes_device;

  // std::vector<int64_t> offset_A;
  // std::vector<int64_t> offset_B;
  // std::vector<int64_t> offset_C;
  // std::vector<int64_t> offset_D;

  std::vector<int64_t> lda_host;
  std::vector<int64_t> ldb_host;
  std::vector<int64_t> ldc_host;
  std::vector<int64_t> ldd_host;

  // cutlass::DeviceAllocation<int64_t> lda;
  // cutlass::DeviceAllocation<int64_t> ldb;
  // cutlass::DeviceAllocation<int64_t> ldc;
  // cutlass::DeviceAllocation<int64_t> ldd;

  // cutlass::DeviceAllocation<ElementA> block_A;
  // cutlass::DeviceAllocation<ElementB> block_B;
  // cutlass::DeviceAllocation<ElementC> block_C;
  // cutlass::DeviceAllocation<ElementC> block_D;

  // cutlass::DeviceAllocation<ElementA *> ptr_A;
  // cutlass::DeviceAllocation<ElementB *> ptr_B;
  // cutlass::DeviceAllocation<ElementC *> ptr_C;
  // cutlass::DeviceAllocation<ElementC *> ptr_D;

  // allocation
  // int64_t total_elements_A = 0;
  // int64_t total_elements_B = 0;
  // int64_t total_elements_C = 0;
  // int64_t total_elements_D = 0;

  lda_host.resize(problem_count);
  ldb_host.resize(problem_count);
  ldc_host.resize(problem_count);
  ldd_host.resize(problem_count);

  for (int32_t i = 0; i < problem_count; ++i) {

    auto problem = problem_sizes.at(i);

    lda_host.at(i) = LayoutA::packed({problem.m(), problem.k()}).stride(0);
    ldb_host.at(i) = LayoutB::packed({problem.k(), problem.n()}).stride(0);
    ldc_host.at(i) = LayoutC::packed({problem.m(), problem.n()}).stride(0);
    ldd_host.at(i) = LayoutC::packed({problem.m(), problem.n()}).stride(0);

    // offset_A.push_back(total_elements_A);
    // offset_B.push_back(total_elements_B);
    // offset_C.push_back(total_elements_C);
    // offset_D.push_back(total_elements_D);

    // int64_t elements_A = problem.m() * problem.k();
    // int64_t elements_B = problem.k() * problem.n();
    // int64_t elements_C = problem.m() * problem.n();
    // int64_t elements_D = problem.m() * problem.n();

    // total_elements_A += elements_A;
    // total_elements_B += elements_B;
    // total_elements_C += elements_C;
    // total_elements_D += elements_D;
  }

  // lda.reset(problem_count);
  // ldb.reset(problem_count);
  // ldc.reset(problem_count);
  // ldd.reset(problem_count);

  // block_A.reset(total_elements_A);
  // block_B.reset(total_elements_B);
  // block_C.reset(total_elements_C);
  // block_D.reset(total_elements_D);


  // omit sort_problems()

  // initialize()
  // problem_sizes_device.reset(problem_count);
  // problem_sizes_device.copy_from_host(problem_sizes.data());

  // lda.copy_from_host(lda_host.data());
  // ldb.copy_from_host(ldb_host.data());
  // ldc.copy_from_host(ldc_host.data());
  // ldd.copy_from_host(ldd_host.data());

  //
  // Assign pointers
  //

  std::vector<ElementA *> ptr_A_host(problem_count);
  std::vector<ElementB *> ptr_B_host(problem_count);
  std::vector<ElementC *> ptr_C_host(problem_count);
  std::vector<ElementC *> ptr_D_host(problem_count);

  for (int32_t i = 0; i < problem_count; ++i) {
    ptr_A_host.at(i) = (ElementA*) as[i].data_ptr();
    ptr_B_host.at(i) = (ElementB*) bs[i].data_ptr();
    ptr_C_host.at(i) = (ElementC*) cs[i].data_ptr();
    ptr_D_host.at(i) = (ElementC*) cs[i].data_ptr();
  }

  // ptr_A.reset(problem_count);
  // ptr_A.copy_from_host(ptr_A_host.data());

  // ptr_B.reset(problem_count);
  // ptr_B.copy_from_host(ptr_B_host.data());

  // ptr_C.reset(problem_count);
  // ptr_C.copy_from_host(ptr_C_host.data());

  // ptr_D.reset(problem_count);
  // ptr_D.copy_from_host(ptr_D_host.data());

  // Configure the GEMM arguments
  //alpha =1, beta=0
  Gemm::EpilogueOutputOp::Params epilogue_op(1.0, 0.0);

  // Configure GEMM arguments
  Gemm::Arguments args(
    problem_sizes_device.get(),
    problem_count,
    threadblock_count,
    epilogue_op,
    ptr_A_host.data(),
    ptr_B_host.data(),
    ptr_C_host.data(),
    ptr_D_host.data(),
    lda_host.data(),
    ldb_host.data(),
    ldc_host.data(),
    ldd_host.data(),
    problem_sizes.data()
  );

  // Initialize the GEMM object
  Gemm gemm;

  size_t workspace_size = gemm.get_workspace_size(args);
  cutlass::DeviceAllocation<uint8_t> workspace(workspace_size);

  result.status = gemm.initialize(args, workspace.get());

  if (result.status != cutlass::Status::kSuccess) {
    std::cerr << "Failed to initialize CUTLASS Grouped GEMM kernel." << std::endl;
    return result;
  }

  // Run the grouped GEMM object
  result.status = gemm.run();

  if (result.status != cutlass::Status::kSuccess) {
    std::cerr << "Failed to run CUTLASS Grouped GEMM kernel." << std::endl;
    return result;
  }

  // Wait for completion
  result.error = cudaDeviceSynchronize();

  if (result.error != cudaSuccess)  {
    std::cerr << "Kernel execution error: " << cudaGetErrorString(result.error);
    return result;
  }

  return result;
}


int grouped_gemm(std::vector<torch::Tensor> as, std::vector<torch::Tensor> bs,
                 std::vector<torch::Tensor> cs) {
  TORCH_CHECK(as.size() == bs.size());
  TORCH_CHECK(as.size() == cs.size());

  std::vector<cutlass::gemm::GemmCoord> problem_sizes;

  // create problem sizes
  for (int i = 0; i < as.size(); ++i) {
    int m = as[i].size(0);
    int k = as[i].size(1);
    int k2 = bs[i].size(0);
    int n = bs[i].size(1);

    TORCH_CHECK(k == k2);

    cutlass::gemm::GemmCoord problem(m, n, k);

    problem_sizes.push_back(problem);
  }

  // options.bin_problems();  // not sure effectiveness...

  int result = impl(as, bs, cs, problem_sizes);

  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("grouped_gemm", &grouped_gemm, "Grouped Gemm forward (CUDA)");
}