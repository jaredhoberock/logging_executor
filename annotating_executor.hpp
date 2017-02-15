#pragma once

#include "logging_executor.hpp"
#include "async_copy.hpp"
#include "bulk_async.hpp"
#include "bulk_invoke.hpp"
#include "for_each.hpp"
#include "disabled_execution_policy.hpp"
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
    size_t depth_;

    inline static void nvtx_range_push_ex(const std::string& message, const color& c)
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

    inline static void nvtx_range_pop()
    {
      nvtxRangePop();
    }

    template<class Token>
    inline void annotate(const Token& t) const
    {
      std::string indentation = std::string().insert(0, 2 * depth_, ' ');

      std::string message = indentation + to_string(t);
      std::cout << message << std::endl;

      std::string mark = name_ + ": " + to_string(t);
      nvtx_range_push_ex(mark, color_);
    }

    
    inline annotator(const std::string& name, color c, size_t depth)
      : name_(name), color_(c), depth_(depth)
    {}

  public:
    inline annotator(const std::string& name = std::string(), color c = green)
      : annotator(name, c, 0)
    {}

    // define copy constructor to silence __host__ __device__ warnings
    inline annotator(const annotator&) = default;

    // define destructor to silence __host__ __device__ warnings
    inline ~annotator() = default;

    inline void descend()
    {
      depth_++;
    }

    inline void ascend()
    {
      depth_--;
    }

    inline size_t depth() const
    {
      return depth_;
    }

    template<class Executor>
    void operator()(const Executor&, executor_operation which) const
    {
      annotate(which);
    }

    template<class ExecutionPolicy>
    void operator()(const ExecutionPolicy&, control_structure which) const
    {
      annotate(which);
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

template<class Executor,
         __AGENCY_REQUIRES(
           agency::is_executor<Executor>::value
         )>
annotating_executor<Executor> annotate(const Executor& exec, const annotator& a)
{
  return annotating_executor<Executor>(exec, a);
}


// XXX it would probably be simpler if annotating_execution_policy just derived from ExecutionPolicy
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

    annotating_execution_policy(const base_execution_policy_type& policy, const annotator& ann)
      : super_t(policy.param(), annotate(policy.executor(), ann))
    {}

    /// \brief This constructor takes an unannotated "base" policy and adapts it into a newly constructed annotating_execution_policy.
    /// \param policy An unannotated, base execution policy to annotate.
    /// \param name The name for the annotation.
    annotating_execution_policy(const base_execution_policy_type& policy, const std::string& name, const color& c)
      : annotating_execution_policy(policy, annotator(name, c))
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
        annotating_execution_policy& policy_;
        Token token_;

      public:
        annotated_scope(annotating_execution_policy& annotated_policy, Token token)
          : policy_(annotated_policy), token_(token)
        {
          // invoke the policy's annotator on the base policy
          policy_.annotator()(policy_.base_execution_policy(), token_);

          // "descend" the annotator into this scope
          policy_.annotator().descend();
        }

        ~annotated_scope()
        {
          // "ascend" the annotator out of this scope
          policy_.annotator().ascend();

          // invoke the annotator's .after() method on the base policy
          policy_.annotator().after(policy_.base_execution_policy(), token_);
        }

        // this returns an ExecutionPolicy to use "within" this scope
        annotating_execution_policy& policy()
        {
          return policy_;
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
  // make an annotator
  annotator ann(name, c);

  // "announce" the annotation
  std::string indentation = std::string().insert(0, 2 * ann.depth(), ' ');
  std::string announcement = indentation + name;

  std::cout << announcement << std::endl;

  // "descend" the annotator before returning a policy
  ann.descend();

  return annotating_execution_policy<ExecutionPolicy>(policy, ann);
}


template<class ExecutionPolicy, class... Args>
auto async_copy(annotating_execution_policy<ExecutionPolicy>& policy, Args&&... args) ->
  decltype(experimental::async_copy(policy.base_execution_policy(), std::forward<Args>(args)...))
{
  // annotate this scope
  auto scope = policy.annotate_scope(control_structure::async_copy);

  // when recursing, prefer ExecutionPolicy's async_copy overload
  auto& recurse_policy = prefer_overload<ExecutionPolicy,control_structure::async_copy>(scope.policy());

  // call async_copy() with the annotated scope's policy
  return experimental::async_copy(recurse_policy, std::forward<Args>(args)...);
}


template<class ExecutionPolicy, class Function, class... Args>
auto bulk_async(annotating_execution_policy<ExecutionPolicy>& policy, Function f, Args&&... args) ->
  decltype(experimental::bulk_async(policy.base_execution_policy(), f, std::forward<Args>(args)...))
{
  // annotate this scope
  auto scope = policy.annotate_scope(control_structure::bulk_async);

  // when recursing, prefer ExecutionPolicy's bulk_async overload
  auto& recurse_policy = prefer_overload<ExecutionPolicy,control_structure::bulk_async>(scope.policy());

  return experimental::bulk_async(recurse_policy, f, std::forward<Args>(args)...);
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
  // annotate this scope
  auto scope = policy.annotate_scope(control_structure::bulk_invoke);

  // when recursing, prefer ExecutionPolicy's bulk_invoke overload
  auto& recurse_policy = prefer_overload<ExecutionPolicy,control_structure::bulk_invoke>(scope.policy());

  return experimental::bulk_invoke(recurse_policy, f, std::forward<Args>(args)...);
}


template<class ExecutionPolicy, class Function, class... Args>
auto bulk_invoke(annotating_execution_policy<ExecutionPolicy>&& policy, Function f, Args&&... args) ->
  decltype(agency::bulk_invoke(policy.base_execution_policy(), f, std::forward<Args>(args)...))
{
  return ::bulk_invoke(policy, f, std::forward<Args>(args)...);
}


template<class ExecutionPolicy, class Iterator, class Function>
void for_each(annotating_execution_policy<ExecutionPolicy>& policy, Iterator first, Iterator last, Function f)
{
  // annotate this scope
  auto scope = policy.annotate_scope(control_structure::for_each);

  // when recursing, prefer ExecutionPolicy's for_each overload
  auto& recurse_policy = prefer_overload<ExecutionPolicy,control_structure::for_each>(scope.policy());

  return experimental::for_each(recurse_policy, first, last, f);
}


template<class ExecutionPolicy, class Iterator, class Function>
void for_each(annotating_execution_policy<ExecutionPolicy>&& policy, Iterator first, Iterator last, Function f)
{
  return ::for_each(policy, first, last, f);
}

