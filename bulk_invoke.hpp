#pragma once

#include <agency/agency.hpp>

namespace experimental
{
namespace detail
{


// this is the implementation of bulk_invoke() which simply calls
// bulk_invoke() via ADL
template<class ExecutionPolicy, class... Args>
auto adl_bulk_invoke(ExecutionPolicy&& policy, Args&&... args) ->
  decltype(bulk_invoke(std::forward<ExecutionPolicy>(policy), std::forward<Args>(args)...))
{
  return bulk_invoke(std::forward<ExecutionPolicy>(policy), std::forward<Args>(args)...);
}


// this is the default implementation of bulk_invoke() which simply calls
// agency::bulk_invoke()
template<class ExecutionPolicy, class... Args>
auto default_bulk_invoke(ExecutionPolicy&& policy, Args&&... args) ->
  decltype(agency::bulk_invoke(std::forward<ExecutionPolicy>(policy), std::forward<Args>(args)...))
{
  return agency::bulk_invoke(std::forward<ExecutionPolicy>(policy), std::forward<Args>(args)...);
}


template<class ExecutionPolicy, class... Args>
struct has_bulk_invoke_free_function_impl
{
  template<class ExecutionPolicy1,
           class = decltype(
             bulk_invoke(std::declval<ExecutionPolicy1>(), std::declval<Args>()...)
           )>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<ExecutionPolicy>(0));
};


} // end detail


template<class ExecutionPolicy, class... Args>
using has_bulk_invoke_free_function = typename detail::has_bulk_invoke_free_function_impl<ExecutionPolicy,Args...>::type;


template<class ExecutionPolicy, class... Args,
         __AGENCY_REQUIRES(
           has_bulk_invoke_free_function<ExecutionPolicy,Args...>::value
         )>
auto bulk_invoke(ExecutionPolicy&& policy, Args&&... args) ->
  decltype(detail::adl_bulk_invoke(std::forward<ExecutionPolicy>(policy), std::forward<Args>(args)...))
{
  return detail::adl_bulk_invoke(std::forward<ExecutionPolicy>(policy), std::forward<Args>(args)...);
} // end bulk_invoke()


template<class ExecutionPolicy, class... Args,
         __AGENCY_REQUIRES(
           !has_bulk_invoke_free_function<ExecutionPolicy,Args...>::value
         )>
auto bulk_invoke(ExecutionPolicy&& policy, Args&&... args) ->
  decltype(detail::default_bulk_invoke(std::forward<ExecutionPolicy>(policy), std::forward<Args>(args)...))
{
  return detail::default_bulk_invoke(std::forward<ExecutionPolicy>(policy), std::forward<Args>(args)...);
} // end bulk_invoke()


} // end experimental

