#pragma once

#include "library.hpp"
#include "async_copy.hpp"
#include "bulk_async.hpp"
#include "bulk_invoke.hpp"
#include "for_each.hpp"
#include "logging_executor.hpp"

template<control_structure which, size_t enabled>
using control_structure_is_enabled = std::integral_constant<bool, bool(size_t(which) & enabled)>;

template<size_t enabled>
using async_copy_is_enabled = control_structure_is_enabled<control_structure::async_copy, enabled>;

template<size_t enabled>
using bulk_async_is_enabled = control_structure_is_enabled<control_structure::bulk_async, enabled>;

template<size_t enabled>
using bulk_invoke_is_enabled = control_structure_is_enabled<control_structure::bulk_invoke, enabled>;

template<size_t enabled>
using for_each_is_enabled = control_structure_is_enabled<control_structure::for_each, enabled>;


// we inherit from ExecutionPolicy's ParentPolicy to make the ParentPolicy's customization points available
// we will enable ExecutionPolicy's customization points one by one based on the enabled bitset below
template<class ExecutionPolicy, size_t enabled, class ParentPolicy = void>
class disabled_execution_policy : public ParentPolicy
{
  public:
    ExecutionPolicy& base()
    {
      return *reinterpret_cast<ExecutionPolicy*>(this);
    }

    const ExecutionPolicy& base() const
    {
      return *reinterpret_cast<const ExecutionPolicy*>(this);
    }

    // make various members of ExecutionPolicy available
    using execution_agent_type = typename ExecutionPolicy::execution_agent_type;
    using executor_type = typename ExecutionPolicy::executor_type;
    using param_type = typename ExecutionPolicy::param_type;

    // forward operator() to the base
    template<class... Args>
    auto operator()(Args&&... args) ->
      decltype(base()(std::forward<Args>(args)...))
    {
      return base()(std::forward<Args>(args)...);
    }

    executor_type& executor() const
    {
      return base().executor();
    }

    const param_type& param() const
    {
      return base().param();
    }

  private:
    // this member ensures sizeof(disabled_execution_policy) >= sizeof(ExecutionPolicy)
    char padding_[sizeof(ExecutionPolicy) - sizeof(ParentPolicy)];
};


// there is no declared ParentPolicy, so just privately inherit from ExecutionPolicy
template<class ExecutionPolicy, size_t enabled>
class disabled_execution_policy<ExecutionPolicy,enabled> : private ExecutionPolicy
{
  public:
    ExecutionPolicy& base()
    {
      return *this;
    }

    const ExecutionPolicy& base() const
    {
      return *this;
    }

    // make various members of ExecutionPolicy public
    using execution_agent_type = typename ExecutionPolicy::execution_agent_type;
    using executor_type = typename ExecutionPolicy::executor_type;
    using param_type = typename ExecutionPolicy::param_type;

    using ExecutionPolicy::operator();
    using ExecutionPolicy::executor;
    using ExecutionPolicy::param;
};


template<control_structure which, class ExecutionPolicy>
disabled_execution_policy<ExecutionPolicy, ~size_t(which)>& remove_overload(ExecutionPolicy& policy)
{
  return *reinterpret_cast<disabled_execution_policy<ExecutionPolicy, ~size_t(which)>*>(&policy);
}


template<class ParentPolicy, control_structure which, class ExecutionPolicy>
disabled_execution_policy<ExecutionPolicy, ~size_t(which), ParentPolicy>& prefer_overload(ExecutionPolicy& policy)
{
  return *reinterpret_cast<disabled_execution_policy<ExecutionPolicy, ~size_t(which), ParentPolicy>*>(&policy);
}


// define customization points for disabled_execution_policy below
// a customization point exists for disabled_execution_policy<ExecutionPolicy> it is enabled and if ExecutionPolicy has that customization point


template<class ExecutionPolicy, size_t enabled, class ParentPolicy, class... Args,
         __AGENCY_REQUIRES(async_copy_is_enabled<enabled>::value),
         __AGENCY_REQUIRES(experimental::has_async_copy_free_function<ExecutionPolicy&,Args&&...>::value)
        >
auto async_copy(disabled_execution_policy<ExecutionPolicy,enabled,ParentPolicy>& policy, Args&&... args) ->
  decltype(async_copy(policy.base(), std::forward<Args>(args)...))
{
  return async_copy(policy.base(), std::forward<Args>(args)...);
}


template<class ExecutionPolicy, size_t enabled, class ParentPolicy, class Function, class... Args,
         __AGENCY_REQUIRES(bulk_async_is_enabled<enabled>::value),
         __AGENCY_REQUIRES(experimental::has_bulk_async_free_function<ExecutionPolicy&,Function,Args&&...>::value)
        >
auto bulk_async(disabled_execution_policy<ExecutionPolicy,enabled,ParentPolicy>& policy, Function f, Args&&... args) ->
  decltype(bulk_async(policy.base(), f, std::forward<Args>(args)...))
{
  return bulk_async(policy.base(), f, std::forward<Args>(args)...);
}


template<class ExecutionPolicy, size_t enabled, class ParentPolicy, class Function, class... Args,
         __AGENCY_REQUIRES(bulk_invoke_is_enabled<enabled>::value),
         __AGENCY_REQUIRES(experimental::has_bulk_invoke_free_function<ExecutionPolicy&,Function,Args&&...>::value)
        >
auto bulk_invoke(disabled_execution_policy<ExecutionPolicy,enabled,ParentPolicy>& policy, Function f, Args&&... args) ->
  decltype(bulk_invoke(policy.base(), f, std::forward<Args>(args)...))
{
  return bulk_invoke(policy.base(), f, std::forward<Args>(args)...);
}


template<class ExecutionPolicy, size_t enabled, class ParentPolicy, class Iterator, class Function, 
         __AGENCY_REQUIRES(for_each_is_enabled<enabled>::value),
         __AGENCY_REQUIRES(experimental::has_for_each_free_function<ExecutionPolicy&,Iterator,Function>::value)
        >
void for_each(disabled_execution_policy<ExecutionPolicy,enabled,ParentPolicy>& policy, Iterator first, Iterator last, Function f)
{
  for_each(policy.base(), first, last, f);
}

