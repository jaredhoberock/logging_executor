#pragma once

#include <agency/agency.hpp>

namespace experimental
{
namespace detail
{


// this is the implementation of bulk_invoke() which simply calls
// bulk_invoke() via ADL
template<class ExecutionPolicy, class... Args>
auto bulk_invoke_adl_implementation(ExecutionPolicy&& policy, Args&&... args) ->
  decltype(bulk_invoke(std::forward<ExecutionPolicy>(policy), std::forward<Args>(args)...))
{
  return bulk_invoke(std::forward<ExecutionPolicy>(policy), std::forward<Args>(args)...);
}


// this is the default implementation of bulk_invoke() which simply calls
// agency::bulk_invoke()
template<class ExecutionPolicy, class... Args>
auto bulk_invoke_default_implementation(ExecutionPolicy&& policy, Args&&... args) ->
  decltype(agency::bulk_invoke(std::forward<ExecutionPolicy>(policy), std::forward<Args>(args)...))
{
  return agency::bulk_invoke(std::forward<ExecutionPolicy>(policy), std::forward<Args>(args)...);
}


template<class ExecutionPolicy, class... Args>
struct can_adl_bulk_invoke_impl
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

template<class ExecutionPolicy, class... Args>
using can_adl_bulk_invoke = typename can_adl_bulk_invoke_impl<ExecutionPolicy,Args...>::type;


} // end detail


template<class ExecutionPolicy, class... Args,
         __AGENCY_REQUIRES(
           detail::can_adl_bulk_invoke<ExecutionPolicy,Args...>::value
         )>
auto bulk_invoke(ExecutionPolicy&& policy, Args&&... args) ->
  decltype(detail::bulk_invoke_adl_implementation(std::forward<ExecutionPolicy>(policy), std::forward<Args>(args)...))
{
  return detail::bulk_invoke_adl_implementation(std::forward<ExecutionPolicy>(policy), std::forward<Args>(args)...);
} // end bulk_invoke()


template<class ExecutionPolicy, class... Args,
         __AGENCY_REQUIRES(
           !detail::can_adl_bulk_invoke<ExecutionPolicy,Args...>::value
         )>
auto bulk_invoke(ExecutionPolicy&& policy, Args&&... args) ->
  decltype(detail::bulk_invoke_default_implementation(std::forward<ExecutionPolicy>(policy), std::forward<Args>(args)...))
{
  return detail::bulk_invoke_default_implementation(std::forward<ExecutionPolicy>(policy), std::forward<Args>(args)...);
} // end bulk_invoke()


} // end experimental

