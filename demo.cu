#include "logging_executor.hpp"
#include "annotating_executor.hpp"

#include <agency/agency.hpp>
#include <agency/cuda.hpp>

int main()
{
  auto exec = annotate(agency::cuda::grid_executor(), "hello world");

  agency::bulk_invoke(agency::par(2).on(exec), [] __host__ __device__ (agency::parallel_agent& self)
  {
    printf("agent %lu\n", self.rank());
  });

  return 0;
}

