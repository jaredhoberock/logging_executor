#pragma once

#include "logging_executor.hpp"
#include "async_copy.hpp"
#include "bulk_async.hpp"
#include "bulk_invoke.hpp"
#include <agency/cuda.hpp>
#include <string>

#include <nvToolsExt.h>

using color = std::uint32_t;

constexpr color green{0x0000ff00};
constexpr color blue{0x000000ff};
constexpr color yellow{0x00ffff00};
constexpr color magenta{0x00ff00ff};
constexpr color cyan{0x0000ffff};
constexpr color red{0x00ff0000};
constexpr color white{0x00ffffff};

class annotator
{
  private:
    std::string name_;
    color color_;
    size_t recursion_depth_;

    static void nvtx_range_push_ex(const std::string& message, const color& c)
    {
      nvtxEventAttributes_t eventAttrib = {0};
      eventAttrib.version = NVTX_VERSION;
      eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
      eventAttrib.colorType = NVTX_COLOR_ARGB;
      eventAttrib.color = c;
      eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
      eventAttrib.message.ascii = message.c_str();
      nvtxRangePushEx(&eventAttrib);
    }

    static void nvtx_range_pop()
    {
      nvtxRangePop();
    }

    void mark_executor_operation(executor_operation which) const
    {
      std::string message = indent() + to_string(which);
      std::cout << message << std::endl;

      std::string mark = name_ + ": " + to_string(which);
      nvtx_range_push_ex(mark, color_);
    }

    void mark_control_structure(control_structure which) const
    {
      std::string message = indent() + to_string(which);
      std::cout << message << std::endl;

      std::string mark = name_ + ": " + to_string(which);
      nvtx_range_push_ex(mark, color_);
    }

    std::string indent() const
    {
      return std::string().insert(0, 2 * recursion_depth_, ' ');
    }

  public:
    annotator(const std::string& name = std::string(), color c = green, size_t recursion_depth = 0)
      : name_(name), color_(c), recursion_depth_(recursion_depth)
    {
      std::string announce = name_;

      std::cout << announce << std::endl;

      descend();
    }

    // define copy constructor to silence __host__ __device__ warnings
    annotator(const annotator&) = default;

    // define destructor to silence __host__ __device__ warnings
    ~annotator()
    {
      ascend();
    }

    void descend()
    {
      ++recursion_depth_;
    }

    void ascend()
    {
      --recursion_depth_;
    }

    annotator new_descend() const
    {
      return annotator(name_, color_, recursion_depth_ + 1);
    }

    template<class Executor>
    void operator()(const Executor&, executor_operation which) const
    {
      mark_executor_operation(which);
    }

    template<class ExecutionPolicy>
    void operator()(const ExecutionPolicy&, control_structure which) const
    {
      mark_control_structure(which);
    }

    template<class Executor>
    void after(const Executor&, executor_operation which) const
    {
      nvtx_range_pop();
    }

    template<class ExecutionPolicy>
    void after(const ExecutionPolicy&, control_structure which) const
    {
      nvtx_range_pop();
    }
};


template<class Executor>
using annotating_executor = logging_executor<Executor, annotator>;

template<class Executor,
         __AGENCY_REQUIRES(
           agency::is_executor<Executor>::value
         )>
annotating_executor<Executor> annotate(const Executor& exec, const std::string& name, const color& c)
{
  return annotating_executor<Executor>(exec, annotator(name, c));
}


template<class ExecutionPolicy>
class annotating_execution_policy :
  public agency::basic_execution_policy<
    typename ExecutionPolicy::execution_agent_type,
    annotating_executor<typename ExecutionPolicy::executor_type>,
    annotating_execution_policy<ExecutionPolicy>
  >
{
  private:
    using super_t = agency::basic_execution_policy<
      typename ExecutionPolicy::execution_agent_type,
      annotating_executor<typename ExecutionPolicy::executor_type>,
      annotating_execution_policy<ExecutionPolicy>
    >;

  public:
    // inherit constructors
    using super_t::super_t;

    /// \brief The type of the adapted execution policy.
    using base_execution_policy_type = ExecutionPolicy;

    /// \brief This constructor takes an unannotated "base" policy and adapts it into a newly constructed annotating_execution_policy.
    /// \param policy An unannotated, base execution policy to annotate.
    /// \param name The name for the annotation.
    annotating_execution_policy(const base_execution_policy_type& policy, const std::string& name, const color& c)
      : super_t(policy.param(), annotate(policy.executor(), name, c))
    {}

    base_execution_policy_type base_execution_policy() const
    {
      return base_execution_policy_type(this->param(), this->executor().base_executor());
    }

    ::annotator& annotator()
    {
      return this->executor().logging_function();
    }

    // this class invokes the annotator in its constructor
    // and its destructor with the given token
    template<class Token>
    class annotated_scope
    {
      private:
        ::annotator annotator_;
        ExecutionPolicy policy_;
        Token token_;

      public:
        annotated_scope(annotating_execution_policy& annotated_policy, Token token)
          : annotator_(annotated_policy.annotator()), policy_(annotated_policy.base_execution_policy()), token_(token)
        {
          // update the recursion depth of the annotator
          // XXX should really happen inside the annotator's constructor somehow
          //annotator_.descend();

          // invoke the annotator
          annotator_(policy_, token_);
        }

        ~annotated_scope()
        {
          // invoke the annotator's .after() method
          annotator_.after(policy_, token_);
        }
    };

    template<class Token>
    annotated_scope<Token> annotate_scope(Token token)
    {
      return annotated_scope<Token>(*this, token);
    }
};


/// \brief Returns an annotating_execution_policy based on the given execution policy and annotation name.
template<class ExecutionPolicy,
         __AGENCY_REQUIRES(
           agency::is_execution_policy<ExecutionPolicy>::value
         )>
annotating_execution_policy<ExecutionPolicy> annotate(const ExecutionPolicy& policy, const std::string& name, const color& c = green)
{
  return annotating_execution_policy<ExecutionPolicy>(policy, name, c);
}


template<class ExecutionPolicy, class... Args>
auto async_copy(annotating_execution_policy<ExecutionPolicy>& policy, Args&&... args) ->
  decltype(experimental::async_copy(policy.base_execution_policy(), std::forward<Args>(args)...))
{
  // get the fancy annotating executor out of the policy
  auto annotating_executor = policy.executor();

  // increment the annotating executor's recursion depth
  annotating_executor.logging_function().descend();

  // unwrap the annotating execution policy and bind it to the annotating_executor
  // to ensure that executor operations still receive annotations
  auto unwrapped_policy = policy.base_execution_policy().on(annotating_executor);

  // annotate this scope
  auto scope = policy.annotate_scope(control_structure::async_copy);

  // call async_copy with the unwrapped policy
  // this will call the normal, unannotated version of async_copy
  // when async_copy uses executor operations, they will be annotated because the fancy annotated_executor
  // will be used
  return experimental::async_copy(unwrapped_policy, std::forward<Args>(args)...);
}


template<class ExecutionPolicy, class Function, class... Args>
auto bulk_async(annotating_execution_policy<ExecutionPolicy>& policy, Function f, Args&&... args) ->
  decltype(experimental::bulk_async(policy.base_execution_policy(), f, std::forward<Args>(args)...))
{
  // get the fancy annotating executor out of the policy
  auto annotating_executor = policy.executor();

  // increment the annotating executor's recursion depth
  annotating_executor.logging_function().descend();

  // unwrap the annotating execution policy and bind it to the annotating_executor
  // to ensure that executor operations still receive annotations
  auto unwrapped_policy = policy.base_execution_policy().on(annotating_executor);

  // annotate this scope
  auto scope = policy.annotate_scope(control_structure::bulk_invoke);

  // call bulk_async with the unwrapped policy
  // this will call the normal, unannotated version of bulk_async
  // when bulk_async uses executor operations, they will be annotated because the fancy annotated_executor
  // will be used
  return experimental::bulk_async(unwrapped_policy, f, std::forward<Args>(args)...);
}


template<class ExecutionPolicy, class Function, class... Args>
auto bulk_async(annotating_execution_policy<ExecutionPolicy>&& policy, Function f, Args&&... args) ->
  decltype(agency::bulk_async(policy.base_execution_policy(), f, std::forward<Args>(args)...))
{
  return ::bulk_async(policy, f, std::forward<Args>(args)...);
}


template<class ExecutionPolicy, class Function, class... Args>
auto bulk_invoke(annotating_execution_policy<ExecutionPolicy>& policy, Function f, Args&&... args) ->
  decltype(agency::bulk_invoke(policy.base_execution_policy(), f, std::forward<Args>(args)...))
{
  //// annotate this scope
  //auto scope = policy.annotate_scope(control_structure::bulk_invoke);

  //// call bulk_invoke() with the annotated scope's policy
  //return experimental::bulk_invoke(scope.policy(), f, std::forward<Args>(args)...);

  // get the fancy annotating executor out of the policy
  auto annotating_executor = policy.executor();

  // increment the annotating executor's recursion depth
  annotating_executor.logging_function().descend();

  // unwrap the annotating execution policy and bind it to the annotating_executor
  // to ensure that executor operations still receive annotations
  auto unwrapped_policy = policy.base_execution_policy().on(annotating_executor);

  // annotate this scope
  auto scope = policy.annotate_scope(control_structure::bulk_invoke);

  // call bulk_async with the unwrapped policy
  // this will call the normal, unannotated version of bulk_async
  // when bulk_async uses executor operations, they will be annotated because the fancy annotated_executor
  // will be used
  return experimental::bulk_invoke(unwrapped_policy, f, std::forward<Args>(args)...);
}

template<class ExecutionPolicy, class Function, class... Args>
auto bulk_invoke(annotating_execution_policy<ExecutionPolicy>&& policy, Function f, Args&&... args) ->
  decltype(agency::bulk_invoke(policy.base_execution_policy(), f, std::forward<Args>(args)...))
{
  return ::bulk_invoke(policy, f, std::forward<Args>(args)...);
}

