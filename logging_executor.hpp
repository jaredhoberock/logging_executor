#pragma once

#include <agency/agency.hpp>

#include <iostream>
#include <utility>

enum class executor_operation
{
  async_execute,
  bulk_async_execute,
  bulk_sync_execute,
  bulk_then_execute,
  future_cast,
  make_ready_future,
  max_shape_dimensions,
  sync_execute,
  then_execute,
  unit_shape
};

std::ostream& operator<<(std::ostream& os, const executor_operation& op)
{
  switch(op)
  {
    case executor_operation::async_execute:
    {
      os << "async_execute";
      break;
    }

    case executor_operation::bulk_async_execute:
    {
      os << "bulk_async_execute";
      break;
    }

    case executor_operation::bulk_sync_execute:
    {
      os << "bulk_sync_execute";
      break;
    }

    case executor_operation::bulk_then_execute:
    {
      os << "bulk_then_execute";
      break;
    }

    case executor_operation::future_cast:
    {
      os << "future_cast";
      break;
    }

    case executor_operation::make_ready_future:
    {
      os << "make_ready_future";
      break;
    }

    case executor_operation::max_shape_dimensions:
    {
      os << "max_shape_dimensions";
      break;
    }

    case executor_operation::sync_execute:
    {
      os << "sync_execute";
      break;
    }

    case executor_operation::then_execute:
    {
      os << "then_execute";
      break;
    }

    case executor_operation::unit_shape:
    {
      os << "unit_shape";
      break;
    }
  }

  return os;
}


// this function invokes logging_function.after(args...) if that expression is well-formed

// well-formed case
template<class LoggingFunction, class... Args>
auto invoke_after_if(LoggingFunction& logging_function, Args&&... args) ->
  decltype(logging_function.after(std::forward<Args>(args)...))
{
  return logging_function.after(std::forward<Args>(args)...);
}

// ill-formed case
template<class... Args>
void invoke_after_if(Args&&...)
{
  // do nothing
}


// logging_executor only has an executor operation member function if the base executor also has that function
// because we only want to log native operations on the base executor
// we might also want a version of this that has every member function so that we can trace the path through an adaptation
// XXX should make .after() an optional member of LoggingFunction
template<class Executor, class LoggingFunction>
struct logging_executor
{
  using base_executor_type = Executor;

  using execution_category = agency::executor_execution_category_t<base_executor_type>; 

  using shape_type = agency::executor_shape_t<base_executor_type>;

  template<class T>
  using future = agency::executor_future_t<base_executor_type,T>;

  base_executor_type base_executor_;

  // XXX consider passing the result of before_(...) as a parameter to after_(...)
  LoggingFunction logging_function_;

  logging_executor(const Executor& exec, LoggingFunction logging_function = LoggingFunction())
    : base_executor_(exec),
      logging_function_(logging_function)
  {}

  const base_executor_type& base_executor() const
  {
    return base_executor_;
  }

  // Note the purpose of the BaseExecutor = base_executor_type stuff
  // in these templates is to introduce a deduced type for the condition given to __AGENCY_REQUIRES()

  template<class Function,
           class BaseExecutor = base_executor_type,
           __AGENCY_REQUIRES(agency::is_asynchronous_executor<BaseExecutor>::value)>
  future<agency::detail::result_of_t<Function()>>
    async_execute(Function&& f)
  {
    logging_function_(base_executor_, executor_operation::async_execute);

    auto result = base_executor_.async_execute(std::forward<Function>(f));

    invoke_after_if(logging_function_, base_executor_, executor_operation::async_execute);

    return std::move(result);
  }

  template<class Function, class ResultFactory, class... Factories,
           class BaseExecutor = base_executor_type,
           __AGENCY_REQUIRES(agency::is_bulk_asynchronous_executor<BaseExecutor>::value)>
  agency::detail::result_of_t<ResultFactory()>
    bulk_async_execute(Function f, shape_type shape, ResultFactory result_factory, Factories... shared_factories)
  {
    logging_function_(base_executor_, executor_operation::bulk_async_execute);

    auto result = base_executor_.bulk_async_execute(f, shape, result_factory, shared_factories...);

    invoke_after_if(logging_function_, base_executor_, executor_operation::bulk_async_execute);

    return std::move(result);
  }

  template<class Function, class ResultFactory, class... Factories,
           class BaseExecutor = base_executor_type,
           __AGENCY_REQUIRES(agency::is_bulk_synchronous_executor<BaseExecutor>::value)>
  agency::detail::result_of_t<ResultFactory()>
    bulk_sync_execute(Function f, shape_type shape, ResultFactory result_factory, Factories... shared_factories)
  {
    logging_function_(base_executor_, executor_operation::bulk_sync_execute);

    auto result = base_executor_.bulk_sync_execute(f, shape, result_factory, shared_factories...);

    invoke_after_if(logging_function_, base_executor_, executor_operation::bulk_sync_execute);

    return std::move(result);
  }

  template<class Function, class Future, class ResultFactory, class... Factories,
           class BaseExecutor = base_executor_type,
           __AGENCY_REQUIRES(agency::is_bulk_continuation_executor<BaseExecutor>::value)>
  future<agency::detail::result_of_t<ResultFactory()>>
    bulk_then_execute(Function f, shape_type shape, Future& predecessor, ResultFactory result_factory, Factories... shared_factories)
  {
    logging_function_(base_executor_, executor_operation::bulk_then_execute);

    auto result = base_executor_.bulk_then_execute(f, shape, predecessor, result_factory, shared_factories...);

    invoke_after_if(logging_function_, base_executor_, executor_operation::bulk_then_execute);

    return std::move(result);
  }

  template<class T, class Future,
           class BaseExecutor = base_executor_type,
           __AGENCY_REQUIRES(agency::detail::has_future_cast<T,BaseExecutor,Future>::value)>
  future<T> future_cast(Future& fut)
  {
    logging_function_(base_executor_, executor_operation::future_cast);

    auto result = base_executor_.template future_cast<T>(fut);

    invoke_after_if(logging_function_, base_executor_, executor_operation::future_cast);

    return std::move(result);
  }

  template<class T, class... Args,
           class BaseExecutor = base_executor_type,
           __AGENCY_REQUIRES(agency::detail::has_make_ready_future<BaseExecutor,T,Args...>::value)>
  future<T> make_ready_future(Args&&... args)
  {
    logging_function_(base_executor_, executor_operation::make_ready_future);

    auto result = base_executor_.template make_ready_future<T>(std::forward<Args>(args)...);

    invoke_after_if(logging_function_, base_executor_, executor_operation::make_ready_future);

    return std::move(result);
  }

  template<class BaseExecutor = base_executor_type,
           __AGENCY_REQUIRES(agency::detail::has_max_shape_dimensions<BaseExecutor,shape_type>::value)>
  shape_type max_shape_dimensions() const
  {
    logging_function_(base_executor_, executor_operation::max_shape_dimensions);

    auto result = base_executor_.max_shape_dimensions();

    invoke_after_if(logging_function_, base_executor_, executor_operation::max_shape_dimensions);

    return std::move(result);
  }

  template<class Function,
           class BaseExecutor = base_executor_type,
           __AGENCY_REQUIRES(agency::is_synchronous_executor<BaseExecutor>::value)>
  agency::detail::result_of_t<Function()>
    sync_execute(Function&& f)
  {
    logging_function_(base_executor_, executor_operation::sync_execute);

    auto result = base_executor_.sync_execute(std::forward<Function>(f));

    invoke_after_if(logging_function_, base_executor_, executor_operation::sync_execute);

    return std::move(result);
  }

  template<class Function, class Future,
           class BaseExecutor = base_executor_type,
           __AGENCY_REQUIRES(agency::is_continuation_executor<BaseExecutor>::value)>
  auto then_execute(Function&& f, Future& predecessor) ->
    decltype(agency::then_execute(base_executor_, std::forward<Function>(f), predecessor))
  {
    logging_function_(base_executor_, executor_operation::then_execute);

    auto result = base_executor_.then_execute(std::forward<Function>(f), predecessor);

    invoke_after_if(logging_function_, base_executor_, executor_operation::then_execute);

    return std::move(result);
  }

  template<class BaseExecutor = base_executor_type,
           __AGENCY_REQUIRES(agency::detail::has_unit_shape<BaseExecutor,shape_type>::value)>
  shape_type unit_shape() const
  {
    logging_function_(base_executor_, executor_operation::unit_shape);

    auto result = base_executor_.unit_shape();

    invoke_after_if(logging_function_, base_executor_, executor_operation::unit_shape);

    return std::move(result);
  }
};

template<class Executor, class LoggingFunction>
logging_executor<Executor,LoggingFunction> make_logging_executor(const Executor& exec, LoggingFunction logging_function)
{
  return logging_executor<Executor,LoggingFunction>(exec, logging_function);
}

