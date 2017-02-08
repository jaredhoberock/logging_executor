/* Copyright (c) 2014, NVIDIA CORPORATION. All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <cstdio>
#include <iostream>
#include <agency/agency.hpp>
#include <agency/cuda.hpp>

#include "annotating_executor.hpp"
#include "async_copy.hpp"

void init_host_data( int n, double * x )
{
  annotator("init_host_data", blue);

  for(int i=0; i<n; ++i)
  {
    x[i] = i;
  }
}

void init_data(int n, double* x, double* x_d, double* y_d)
{
  auto policy = annotate(agency::cuda::par, "init_data", yellow);

  auto copy_finished = experimental::async_copy(policy, x, x + n, x_d);

  auto init_finished = agency::bulk_async(policy(n), [=] __device__ (agency::parallel_agent& self)
  {
    y_d[self.index()] = n - self.index();
  });
  
  copy_finished.wait();
  init_finished.wait();
}

void daxpy(int n, double a, double* x, double* y)
{
  auto policy = annotate(agency::cuda::par, "daxpy", magenta);

  agency::bulk_invoke(policy(n), [=] __device__ (agency::parallel_agent& self)
  {
    int i = self.index();
    y[i] = a*x[i] + y[i];
  });
}

void check_results(int n, double correctvalue, double* x_d)
{
  auto policy = annotate(agency::cuda::par, "check_results", cyan);

  agency::bulk_invoke(policy(n), [=] __device__ (agency::parallel_agent& self)
  {
    int i = self.index();

    if(x_d[i] != correctvalue)
    {
      printf("ERROR at index = %d, expected = %f, actual: %f\n",i,correctvalue,x_d[i]);
    }
  });
}

void run_test(int n)
{
  {
    annotator("run_test", green);

    double* x_d;
    double* y_d;
    cudaSetDevice(0);

    std::vector<double> x(n);

    cudaMalloc((void**)&x_d,n*sizeof(double));
    cudaMalloc((void**)&y_d,n*sizeof(double));
    
    init_host_data(n, x.data());
    
    init_data(n,x.data(),x_d,y_d);
    
    daxpy(n,1.0,x_d,y_d);
    
    check_results(n, n, y_d);
    
    cudaFree(y_d);
    cudaFree(x_d);
    cudaDeviceSynchronize();
  }

  std::cout << "OK" << std::endl;
}

int main()
{
  int n = 1<<22;
  run_test(n);
  return 0;
}
