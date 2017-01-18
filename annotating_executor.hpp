#pragma once

#include "logging_executor.hpp"
#include <agency/cuda.hpp>
#include <string>

#include <nvToolsExt.h>

class annotator
{
  private:
    std::string annotation_;

    __host__ __device__
    void annotate_bulk_then(const agency::cuda::device_id& device) const
    {
    }

  public:
    annotator(const std::string& annotation)
      : annotation_(annotation)
    {}

    __host__ __device__
    void operator()(agency::cuda::grid_executor& self, executor_operation which) const
    {
#ifndef __CUDA_ARCH__
      std::cout << "annotator::operator(): " << annotation_ << ": " << which << std::endl;
#endif

      if(which == executor_operation::bulk_then_execute)
      {
        annotate_bulk_then(self.device());
      }
      else
      {
        // ignore other operations for now
      }
    }

    __host__ __device__
    void after(agency::cuda::grid_executor& self, executor_operation which) const
    {
#ifndef __CUDA_ARCH__
      std::cout << "after" << std::endl;
#endif
    }
};

using annotating_executor = logging_executor<agency::cuda::grid_executor, annotator>;

annotating_executor annotate(const agency::cuda::grid_executor& exec, const std::string& annotation = "hello world")
{
  return annotating_executor(exec, annotator(annotation));
}

