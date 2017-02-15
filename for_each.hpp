#pragma once

#include <agency/cuda.hpp>
#include <agency/cuda/experimental.hpp>
#include <agency/detail/type_traits.hpp>
#include "bulk_invoke.hpp"


namespace experimental
{
namespace detail
{


// this is the implementation of for_each() which simply calls
// for_each() via ADL
template<class ExecutionPolicy, class Iterator, class Function>
void adl_for_each(ExecutionPolicy&& policy, Iterator first, Iterator last, Function f)
{
  return for_each(std::forward<ExecutionPolicy>(policy), first, last, f);
}


// this is the default implementation of for_each()
template<class ExecutionPolicy, class Iterator, class Function>
void default_for_each(ExecutionPolicy&& policy, Iterator first, Iterator last, Function f)
{
  using policy_type = typename std::decay<ExecutionPolicy>::type;
  using execution_agent_type = typename policy_type::execution_agent_type;

  experimental::bulk_invoke(policy(last - first), [=] __host__ __device__ (execution_agent_type& self)
  {
    Iterator i = first + self.index();
    f(*i);
  });
}


template<class ExecutionPolicy, class Iterator, class Function>
struct has_for_each_free_function_impl
{
  template<class ExecutionPolicy1,
           class = decltype(
             for_each(std::declval<ExecutionPolicy1>(), std::declval<Iterator>(), std::declval<Iterator>(), std::declval<Function>())
           )>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<ExecutionPolicy>(0));
};


} // end detail


template<class ExecutionPolicy, class Iterator, class Function>
using has_for_each_free_function = typename detail::has_for_each_free_function_impl<ExecutionPolicy,Iterator,Function>::type;


template<class ExecutionPolicy, class Iterator, class Function,
         __AGENCY_REQUIRES(
           has_for_each_free_function<ExecutionPolicy,Iterator,Function>::value
         )>
void for_each(ExecutionPolicy&& policy, Iterator first, Iterator last, Function f)
{
  return detail::adl_for_each(std::forward<ExecutionPolicy>(policy), first, last, f);
}


template<class ExecutionPolicy, class Iterator, class Function,
         __AGENCY_REQUIRES(
           !has_for_each_free_function<ExecutionPolicy,Iterator,Function>::value
         )>
void for_each(ExecutionPolicy&& policy, Iterator first, Iterator last, Function f)
{
  return detail::default_for_each(std::forward<ExecutionPolicy>(policy), first, last, f);
}


} // end experimental

