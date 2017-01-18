#include "logging_executor.hpp"

#include <agency/agency.hpp>
#include <agency/cuda.hpp>

int main()
{
  auto exec = make_logging_executor(agency::cuda::grid_executor(), [] __host__ __device__ (const agency::cuda::grid_executor&, executor_operation which)
  {
#ifndef __CUDA_ARCH__
    std::cout << which << std::endl;
#endif
  });

  agency::bulk_invoke(agency::par(2).on(exec), [] __host__ __device__ (agency::parallel_agent& self)
  {
    printf("agent %lu\n", self.rank());
  });

  return 0;
}

