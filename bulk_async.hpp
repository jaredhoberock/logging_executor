#pragma once

#include <agency/agency.hpp>

namespace experimental
{
namespace detail
{


// this is the implementation of bulk_async() which simply calls
// bulk_async() via ADL
template<class ExecutionPolicy, class... Args>
auto bulk_async_adl_implementation(ExecutionPolicy&& policy, Args&&... args) ->
  decltype(bulk_async(std::forward<ExecutionPolicy>(policy), std::forward<Args>(args)...))
{
  return bulk_async(std::forward<ExecutionPolicy>(policy), std::forward<Args>(args)...);
}


// this is the default implementation of bulk_async() which simply calls
// agency::bulk_async()
template<class ExecutionPolicy, class... Args>
auto bulk_async_default_implementation(ExecutionPolicy&& policy, Args&&... args) ->
  decltype(agency::bulk_async(std::forward<ExecutionPolicy>(policy), std::forward<Args>(args)...))
{
  return agency::bulk_async(std::forward<ExecutionPolicy>(policy), std::forward<Args>(args)...);
}


template<class ExecutionPolicy, class... Args>
struct can_adl_bulk_async_impl
{
  template<class ExecutionPolicy1,
           class = decltype(
             bulk_async(std::declval<ExecutionPolicy1>(), std::declval<Args>()...)
           )>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<ExecutionPolicy>(0));
};

template<class ExecutionPolicy, class... Args>
using can_adl_bulk_async = typename can_adl_bulk_async_impl<ExecutionPolicy,Args...>::type;


} // end detail


template<class ExecutionPolicy, class... Args,
         __AGENCY_REQUIRES(
           detail::can_adl_bulk_async<ExecutionPolicy,Args...>::value
         )>
auto bulk_async(ExecutionPolicy&& policy, Args&&... args) ->
  decltype(detail::bulk_async_adl_implementation(std::forward<ExecutionPolicy>(policy), std::forward<Args>(args)...))
{
  return detail::bulk_async_adl_implementation(std::forward<ExecutionPolicy>(policy), std::forward<Args>(args)...);
} // end bulk_async()


template<class ExecutionPolicy, class... Args,
         __AGENCY_REQUIRES(
           !detail::can_adl_bulk_async<ExecutionPolicy,Args...>::value
         )>
auto bulk_async(ExecutionPolicy&& policy, Args&&... args) ->
  decltype(detail::bulk_async_default_implementation(std::forward<ExecutionPolicy>(policy), std::forward<Args>(args)...))
{
  return detail::bulk_async_default_implementation(std::forward<ExecutionPolicy>(policy), std::forward<Args>(args)...);
} // end bulk_async()


} // end experimental

