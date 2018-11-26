/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_core_Concurrency_h
#define INCLUDED_ml_core_Concurrency_h

#include <core/ImportExport.h>

#include <transwarp/transwarp.h>

#include <functional>
#include <thread>

namespace ml {
namespace core {
using transwarp::executor;
using transwarp::task;
using static_thread_pool = transwarp::detail::thread_pool;

//! Setup the global default executor for async.
//!
//! \note This is not thread safe as the intention is that it is invoked once at
//! the beginning of main.
//! \note If this is called with threads = 0 it defaults threads to using calling
//! std::thread::hardware_concurrency to size the thread pool.
CORE_EXPORT
void startDefaultAsyncExecutor(std::size_t threads = 0);

//! The global default thread pool.
//!
//! This gets the default parallel executor set by startDefaultAsyncExecutor.
//! If this hasn't been started execution happens serially in the same thread.
CORE_EXPORT
executor& defaultAsyncExecutor();

//! Get the default async executor thread pool size.
CORE_EXPORT
std::size_t defaultAsyncThreadPoolSize();

//! An "overload" of async which uses a specified executor.
template<typename FUNCTION, typename... ARGS>
std::shared_ptr<task<std::result_of_t<std::decay_t<FUNCTION>(std::decay_t<ARGS>...)>>>
async(executor& exec, FUNCTION&& f, ARGS&&... args) {
    // Note g stores copies of the arguments in the pack, which are moved into place
    // if possible, so this is safe to invoke in the context of a scheduled task.
    auto g = std::bind(std::forward<FUNCTION>(f), std::forward<ARGS>(args)...);

    // Create and schedule the task for the supplied executor.
    auto task = transwarp::make_task(transwarp::root, "async_task", g);
    task->schedule(exec);

    return task;
}

//! Wait for \p task to finish if there is one.
template<typename RESULT>
void await(const std::shared_ptr<task<RESULT>>& task) {
    if (task != nullptr) {
        task->wait();
    }
}

//! Wait for a collection of tasks to finish.
template<typename RESULT>
void await(const std::vector<std::shared_ptr<task<RESULT>>>& tasks) {
    for (const auto& task : tasks) {
        await(task);
    }
}

//! Run \p f in parallel using async.
//!
//! This executes \p f on each index in the range [\p start, \p end) using the
//! default async executor.
//!
//! \param[in] start The first index for which to execute \p f.
//! \param[in] end The end of the indices for which to execute \p f.
//! \param[in,out] f The function to execute on each index. This expected to
//! implement the std::function<void(std::size_t)> contract.
template<typename FUNCTION>
std::vector<FUNCTION> parallel_for_each(std::size_t start, std::size_t end, FUNCTION&& f) {

    if (end <= start) {
        return {std::forward<FUNCTION>(f)};
    }

    std::size_t threads{std::min(defaultAsyncThreadPoolSize(), end - start)};

    if (threads == 0) {
        for (std::size_t i = start; i < end; ++i) {
            f(i);
        }
        return {std::forward<FUNCTION>(f)};
    }

    std::vector<FUNCTION> functions(threads, std::forward<FUNCTION>(f));

    std::vector<task<std::result_of_t<std::decay_t<FUNCTION>()>>> tasks;

    for (std::size_t offset = 0; offset < threads; ++offset, ++start) {
        auto& g = functions[offset];
        async(defaultAsyncExecutor(),
              [&g, threads](std::size_t start_, std::size_t end_) { 
                  for (std::size_t j = start_; j < end_; j += threads) {
                      g(j);
                  }
              }, start, end);
    }

    await(tasks);

    return functions;
}

//! Run \p f in parallel using async.
//!
//! This executes \p f on each index in the range [\p start, \p end) using the
//! default async executor.
//!
//! \param[in] start The first index for which to execute \p f.
//! \param[in] end The end of the indices for which to execute \p f.
//! \param[in,out] f The function to execute on each index. This expected to
//! implement the std::function<void(std::size_t)> contract.
template<typename ITR, typename FUNCTION>
std::vector<FUNCTION> parallel_for_each(ITR start, ITR end, FUNCTION&& f) {

    std::size_t size{std::distance(start, end)};
    if (size == 0) {
        return {std::forward<FUNCTION>(f)};
    }

    std::size_t threads{std::min(defaultAsyncThreadPoolSize(), size)};

    if (threads == 0) {
        return {std::for_each(start, end, std::forward<FUNCTION>(f))};
    }

    std::vector<FUNCTION> functions(threads, std::forward<FUNCTION>(f));

    std::vector<task<std::result_of_t<std::decay_t<FUNCTION>()>>> tasks;

    for (std::size_t offset = 0; offset < threads; ++offset, ++start) {
        auto& g = functions[offset];
        async(defaultAsyncExecutor(),
              [&g, threads, offset, size](ITR start_) {
                  std::size_t i{offset};
                  auto increment = [&](ITR j) {
                      if (i < size) {
                          std::advance(j, threads);
                      }
                  };
                  for (ITR j = start_; i < size; i += threads, advance(j)) {
                      g(*j);
                  }
              }, start);
    }

    await(tasks);

    return functions;
}

namespace concurrency_detail {

template<typename FUNCTION, typename BOUND_STATE>
struct SFunctionWithBoundState {
    template<typename... ARGS>
    std::result_of_t<std::decay_t<FUNCTION>(std::decay_t<ARGS>...)>
    operator()(ARGS&& ...args) const {
        m_Function(s_FunctionState, std::forward<ARGS>(args)...);
    }
    FUNCTION s_Function;
    BOUND_STATE s_FunctionState;
};
}

//!
template<typename FUNCTION, typename STATE>
auto bind_retrievable_state(FUNCTION&& function, STATE&& state) {
    return concurrency_detail::SFunctionWithBoundState<FUNCTION, STATE>(std::forward<FUNCTION>(function),
                                                                        std::forward<STATE>(state));
}
}
}

#endif // INCLUDED_ml_core_Concurrency_h
