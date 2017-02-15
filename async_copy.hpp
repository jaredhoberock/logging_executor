#pragma once

#include <agency/cuda.hpp>
#include <agency/cuda/experimental.hpp>
#include <agency/detail/type_traits.hpp>

namespace experimental
{
namespace detail
{


// this is the implementation of async_copy() which simply calls
// async_copy() via ADL
template<class ExecutionPolicy, class... Args>
auto adl_async_copy(ExecutionPolicy&& policy, Args&&... args) ->
  decltype(async_copy(std::forward<ExecutionPolicy>(policy), std::forward<Args>(args)...))
{
  return async_copy(std::forward<ExecutionPolicy>(policy), std::forward<Args>(args)...);
}


// this is the default implementation of async_copy()
template<class ExecutionPolicy, class T>
agency::cuda::async_future<void> default_async_copy(ExecutionPolicy&& policy, const T* first, const T* last, T* result)
{
  // create a stream for the copy
  // XXX should reach into policy's executor for its device_id or a ready future, preferably
  agency::cuda::detail::stream stream;

  // execute the copy
  agency::cuda::detail::throw_on_error(cudaMemcpyAsync(result, first, (last - first) * sizeof(T), cudaMemcpyDefault, stream.native_handle()), "cuda::async_copy(): cudaMemcpyAsync()");

  // get a future corresponding to the completion of the copy
  return agency::cuda::experimental::make_async_future(stream.native_handle());
}


template<class ExecutionPolicy, class... Args>
struct has_async_copy_free_function_impl
{
  template<class ExecutionPolicy1,
           class = decltype(
             async_copy(std::declval<ExecutionPolicy1>(), std::declval<Args>()...)
           )>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<ExecutionPolicy>(0));
};


} // end detail


template<class ExecutionPolicy, class... Args>
using has_async_copy_free_function = typename detail::has_async_copy_free_function_impl<ExecutionPolicy,Args...>::type;


template<class ExecutionPolicy, class... Args,
         __AGENCY_REQUIRES(
           has_async_copy_free_function<ExecutionPolicy,Args...>::value
         )>
auto async_copy(ExecutionPolicy&& policy, Args&&... args) ->
  decltype(detail::adl_async_copy(std::forward<ExecutionPolicy>(policy), std::forward<Args>(args)...))
{
  return detail::adl_async_copy(std::forward<ExecutionPolicy>(policy), std::forward<Args>(args)...);
} // end async_copy()


template<class ExecutionPolicy, class... Args,
         __AGENCY_REQUIRES(
           !has_async_copy_free_function<ExecutionPolicy,Args...>::value
         )>
auto async_copy(ExecutionPolicy&& policy, Args&&... args) ->
  decltype(detail::default_async_copy(std::forward<ExecutionPolicy>(policy), std::forward<Args>(args)...))
{
  return detail::default_async_copy(std::forward<ExecutionPolicy>(policy), std::forward<Args>(args)...);
} // end async_copy()


} // end experimental

