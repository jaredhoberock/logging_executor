#include <agency/agency.hpp>
#include <agency/cuda.hpp>

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

template<class Executor, class LoggingFunction>
struct logging_executor
{
  using base_executor = Executor;

  using shape_type = agency::executor_shape_t<base_executor>;

  template<class T>
  using future = agency::executor_future_t<base_executor,T>;

  base_executor base_executor_;
  LoggingFunction logging_function_;

  logging_executor(const Executor& exec, LoggingFunction logging_function = LoggingFunction())
    : base_executor_(exec),
      logging_function_(logging_function)
  {}

  template<class Function>
  auto async_execute(Function&& f) ->
    decltype(agency::async_execute(base_executor_, std::forward<Function>(f)))
  {
    logging_function_(executor_operation::async_execute);

    return agency::async_execute(base_executor_, std::forward<Function>(f));
  }

  template<class Function, class ResultFactory, class... Factories>
  auto bulk_async_execute(Function f, shape_type shape, ResultFactory result_factory, Factories... shared_factories) ->
    decltype(agency::bulk_async_execute(base_executor_, f, shape, result_factory, shared_factories...))
  {
    logging_function_(executor_operation::bulk_async_execute);

    return agency::bulk_async_execute(base_executor_, f, shape, result_factory, shared_factories...);
  }

  template<class Function, class ResultFactory, class... Factories>
  auto bulk_sync_execute(Function f, shape_type shape, ResultFactory result_factory, Factories... shared_factories) ->
    decltype(agency::bulk_sync_execute(base_executor_, f, shape, result_factory, shared_factories...))
  {
    logging_function_(executor_operation::bulk_sync_execute);

    return agency::bulk_sync_execute(base_executor_, f, shape, result_factory, shared_factories...);
  }

  template<class Function, class Future, class ResultFactory, class... Factories>
  auto bulk_then_execute(Function f, shape_type shape, Future& predecessor, ResultFactory result_factory, Factories... shared_factories) ->
    decltype(agency::bulk_then_execute(base_executor_, f, shape, predecessor, result_factory, shared_factories...))
  {
    logging_function_(executor_operation::bulk_then_execute);

    return agency::bulk_then_execute(base_executor_, f, shape, predecessor, result_factory, shared_factories...);
  }

  template<class T, class Future>
  future<T> future_cast(Future& fut)
  {
    logging_function_(executor_operation::future_cast);

    return agency::future_cast<T>(base_executor_, fut);
  }

  template<class T, class... Args>
  future<T> make_ready_future(Args&&... args)
  {
    logging_function_(executor_operation::make_ready_future);

    return agency::make_ready_future<T>(base_executor_, std::forward<Args>(args)...);
  }

  shape_type max_shape_dimensions()
  {
    logging_function_(executor_operation::max_shape_dimensions);

    return agency::max_shape_dimensions(base_executor_);
  }

  template<class Function>
  auto sync_execute(Function&& f) ->
    decltype(agency::sync_execute(base_executor_, std::forward<Function>(f)))
  {
    logging_function_(executor_operation::sync_execute);

    return agency::sync_execute(base_executor_, std::forward<Function>(f));
  }

  template<class Function, class Future>
  auto then_execute(Function&& f, Future& predecessor) ->
    decltype(agency::then_execute(base_executor_, std::forward<Function>(f), predecessor))
  {
    logging_function_(executor_operation::then_execute);

    return agency::then_execute(base_executor_, std::forward<Function>(f), predecessor);
  }

  shape_type unit_shape()
  {
    logging_function_(executor_operation::unit_shape);

    return agency::unit_shape(base_executor_);
  }
};

template<class Executor, class Function>
logging_executor<Executor,Function> make_logging_executor(const Executor& exec, Function logger)
{
  return logging_executor<Executor,Function>(exec, logger);
}

int main()
{
  auto exec = make_logging_executor(agency::sequenced_executor(), [](executor_operation which)
  {
    std::cout << which << std::endl;
  });

  exec.sync_execute([]
  {
    std::cout << "function called" << std::endl;
  });

  return 0;
}

