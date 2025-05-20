/**
 *  @file   fork_union.hpp
 *  @brief  Minimalistic C++ thread-pool designed for SIMT-style 'Fork-Join' parallelism.
 *  @author Ash Vardanian
 *  @date   May 2, 2025
 */
#pragma once
#include <memory>  // `std::allocator`
#include <thread>  // `std::thread`
#include <atomic>  // `std::atomic`
#include <cstddef> // `std::max_align_t`
#include <cassert> // `assert`

#define FORK_UNION_VERSION_MAJOR 0
#define FORK_UNION_VERSION_MINOR 2
#define FORK_UNION_VERSION_PATCH 2

/**
 *  On C++17 and later we can detect misuse of lambdas that are not properly annotated.
 *  On C++20 and later we can use concepts for cleaner compile-time checks.
 */
#define _FU_DETECT_CPP_20 (__cplusplus >= 202002L)
#define _FU_DETECT_CPP_17 (__cplusplus >= 201703L)
#if _FU_DETECT_CPP_17
#include <type_traits> // `std::is_nothrow_invocable_r`
#endif

#if _FU_DETECT_CPP_17
#define _FU_MAYBE_UNUSED [[maybe_unused]]
#else
#if defined(__GNUC__) || defined(__clang__)
#define _FU_MAYBE_UNUSED __attribute__((unused))
#elif defined(_MSC_VER)
#define _FU_MAYBE_UNUSED __pragma(warning(suppress : 4100))
#else
#define _FU_MAYBE_UNUSED
#endif
#endif

namespace ashvardanian {

/**
 *  @brief Minimalistic STL-based non-resizable thread-pool for simultaneous blocking tasks.
 *
 *  Note, that for N-wide parallelism, it initiates (N-1) threads, and the current caller thread
 *  is also executing tasks. The number of threads is fixed and cannot be changed.
 *
 *  It avoids heap allocations and expensive abstractions like `std::function`, `std::future`,
 *  `std::promise`, `std::condition_variable`, etc. It only uses `std::thread` and `std::atomic`,
 *  benefiting from the fact, that those are standardized since C++11.
 *
 *  Note, that "stopping" isn't done at a sub-task granularity level. So if you are submitting
 *  a huge task in a "dynamic" eager mode, it's up to you to abrupt it early with extra logic.
 *
 *  Most operations are performed with a "weak" memory model, to be able to leverage in-hardware
 *  support for atomic fence-less operations on Arm and Power architectures. Most atomic counters
 *  use the "acquire-release" model, and some going further to "relaxed" model.
 *  @see https://en.cppreference.com/w/cpp/atomic/memory_order#Release-Acquire_ordering
 */
template <typename allocator_type_ = std::allocator<std::byte>, std::size_t alignment_ = alignof(std::max_align_t)>
class fork_union {
  public:
    using allocator_t = allocator_type_;
    using thread_index_t = std::size_t;
    using task_index_t = std::size_t;

    /**
     *  @brief A POD-type describing a certain part of a parallel task.
     *
     *  User callbacks in a simple case receive just a single integer identifying the part
     *  of the task to be executed. For more advanced use-cases, when the user needs to know
     *  the thread ID (likely to use some thread-local storage), the callbacks should receive
     *  a `task_t` argument, and unpack the thread ID from it.
     */
    struct task_t {
        thread_index_t thread_index {0};
        task_index_t task_index {0};

        inline operator std::size_t() const noexcept { return task_index; }
    };

    using punned_task_context_t = void const *;
    using trampoline_pointer_t = void (*)(punned_task_context_t, task_t);

    struct c_callback_t {
        trampoline_pointer_t callable {nullptr};
        punned_task_context_t context {nullptr};

        inline void operator()(task_t task) const noexcept { callable(context, task); }
    };

  private:
    // Thread-pool-specific variables:
    allocator_t allocator_ {};
    std::thread *workers_ {nullptr};
    thread_index_t total_threads_ {0};
    alignas(alignment_) std::atomic<bool> stop_ {false};

    // Task-specific variables:
    punned_task_context_t task_lambda_pointer_ {nullptr};    // ? Pointer to the users lambda
    trampoline_pointer_t task_trampoline_pointer_ {nullptr}; // ? Calls the lambda
    task_index_t task_parts_count_ {0};
    alignas(alignment_) std::atomic<task_index_t> task_parts_remaining_ {0};
    alignas(alignment_) std::atomic<task_index_t> task_parts_passed_ {0}; // ? Only used in dynamic mode
    alignas(alignment_) std::atomic<std::size_t> task_generation_ {0};

  public:
    fork_union(fork_union &&) = delete;
    fork_union(fork_union const &) = delete;
    fork_union &operator=(fork_union &&) = delete;
    fork_union &operator=(fork_union const &) = delete;

    fork_union(allocator_t const &alloc = {}) noexcept : allocator_(alloc) {}
    ~fork_union() noexcept { stop_and_reset(); }
    std::size_t thread_count() const noexcept { return total_threads_; }

    /**
     *  @brief Creates a thread-pool with the given number of threads.
     *  @note This is the de-facto @b constructor, and can only be called once.
     *  @param[in] planned_threads The number of threads to be used. Should be larger than one.
     *  @retval false if the number of threads is zero or the "workers" allocation failed.
     *  @retval true if the thread-pool was created successfully, started, and is ready to use.
     */
    bool try_spawn(std::size_t const planned_threads) noexcept {
        if (planned_threads == 0) return false; // ! Can't have zero threads working on something
        if (planned_threads == 1) return true;  // ! The current thread will always be used
        if (total_threads_ != 0) return false;  // ! Already initialized

        // Allocate the thread pool
        thread_index_t const worker_threads = planned_threads - 1;
        std::thread *const workers = allocator_.allocate(worker_threads);
        if (workers == nullptr) return false; // ! Allocation failed

        // Initialize the thread pool can fail for all kinds of reasons,
        // that the `std::thread` documentation describes as "implementation-defined".
        // https://en.cppreference.com/w/cpp/thread/thread/thread
        for (thread_index_t i = 0; i < worker_threads; ++i) {
            try {
                new (&workers[i]) std::thread([this, i] { _worker_loop(i + 1); });
            }
            catch (...) {
                for (thread_index_t j = 0; j < i; ++j) workers[j].~thread();
                allocator_.deallocate(workers, worker_threads);
                return false; // ! Thread creation failed
            }
        }

        // If all went well, we can store the thread-pool and start using it
        workers_ = workers;
        total_threads_ = planned_threads;
        return true;
    }

    /**
     *  @brief Stops all threads and deallocates the thread-pool.
     *  @note Can only be called from the main thread, and can't have any tasks in-flight.
     */
    void stop_and_reset() noexcept {
        if (total_threads_ == 0) return;                                    // ? Uninitialized
        if (total_threads_ == 1) return;                                    // ? No worker threads to join
        assert(task_parts_remaining_.load(std::memory_order_seq_cst) == 0); // ! No tasks must be running

        // Stop all threads and wait for them to finish
        _reset_task();
        stop_.store(true, std::memory_order_release);

        thread_index_t const worker_threads = total_threads_ - 1;
        for (thread_index_t i = 0; i != worker_threads; ++i) {
            workers_[i].join();    // ? Wait for the thread to finish
            workers_[i].~thread(); // ? Call destructor
        }

        // Deallocate the thread pool
        allocator_.deallocate(workers_, worker_threads);

        // Prepare for future spawns
        total_threads_ = 0;
        workers_ = nullptr;
        stop_.store(false, std::memory_order_relaxed);
    }

    /**
     *  @brief Distributes @p (n) similar tasks into between threads.
     *  @param[in] n The number of times to call the @p function.
     *  @param[in] function The callback, receiving @b `task_t` or the task index as an argument.
     *
     *  Is designed for a "balanced" workload, where all threads have roughly the same amount of work.
     *  @sa `for_each_dynamic` for a more dynamic workload.
     *  The @p function is called @p (n) times, and each thread receives a slice of consecutive tasks.
     *  @sa `for_each_slice` if you prefer to receive workload slices over individual indices.
     */
    template <typename function_type_>
    void for_each_static(task_index_t const n, function_type_ const &function) noexcept {

#if _FU_DETECT_CPP_17
        // ? Exception handling and aggregating return values drastically increases code complexity
        static_assert((std::is_nothrow_invocable_r<void, function_type_, task_t>::value ||
                       std::is_nothrow_invocable_r<void, function_type_, task_index_t>::value),
                      "The callback must be invocable with a `task_t` or a `task_index_t` argument");
#endif

        for_each_slice(n, [function](task_t start_task, task_index_t count) noexcept {
            for (task_index_t i = 0; i < count; ++i)
                function(task_t {start_task.thread_index, start_task.task_index + i});
        });
    }

    /**
     *  @brief Splits a range of @p (n) tasks into consecutive chunks for each thread.
     *  @param[in] n The total length of the range to split between threads.
     *  @param[in] function The callback, receiving @b `task_t` or an unsigned integer and the slice length.
     */
    template <typename function_type_>
    void for_each_slice(task_index_t const n, function_type_ const &function) noexcept {

#if _FU_DETECT_CPP_17
        // ? Exception handling and aggregating return values drastically increases code complexity
        static_assert((std::is_nothrow_invocable_r<void, function_type_, task_t, task_index_t>::value ||
                       std::is_nothrow_invocable_r<void, function_type_, task_index_t, task_index_t>::value),
                      "The callback must be invocable with a `task_t` or a `task_index_t` argument");
#endif

        // No need to slice the workload if we only have one thread
        assert(total_threads_ != 0 && "Thread pool not initialized");
        if (total_threads_ == 1 || n == 1) return function(task_t {0, 0}, n);
        if (n == 0) return;

        // Divide and round-up the workload size per thread
        task_index_t const n_per_thread = (n + total_threads_ - 1) / total_threads_;
        for_each_thread([n, n_per_thread, function](thread_index_t thread_index) noexcept {
            task_index_t const begin = (std::min)(thread_index * n_per_thread, n);
            task_index_t const count = (std::min)(begin + n_per_thread, n) - begin;
            function(task_t {thread_index, begin}, count);
        });
    }

    /**
     *  @brief Executes a function in parallel on the current and all worker threads.
     *  @param[in] function The callback, receiving the thread index as an argument.
     */
    template <typename function_type_>
    void for_each_thread(function_type_ const &function) noexcept {
        if (total_threads_ == 1) return function(0);

#if _FU_DETECT_CPP_17
        // ? Exception handling and aggregating return values drastically increases code complexity
        static_assert(std::is_nothrow_invocable_r<void, function_type_, task_index_t>::value,
                      "The callback must be invocable with a `task_index_t` argument");
#endif

        // Store closure address
        task_lambda_pointer_ = std::addressof(function);
        task_trampoline_pointer_ = &_call_lambda_from_thread<function_type_>;
        task_parts_count_ = total_threads_;
        task_parts_remaining_.store(total_threads_ - 1, std::memory_order_relaxed);
        task_generation_.fetch_add(1, std::memory_order_release); // ? Wake up sleepers

        // Execute on the current thread
        function(0);

        // Wait until all threads are done
        while (task_parts_remaining_.load(std::memory_order_acquire)) std::this_thread::yield();

        // An optional reset of the task variables for debuggability
        _reset_task();
    }

    /**
     *  @brief Executes uneven tasks on all threads, greedying for work.
     *  @param[in] n The number of times to call the @p function.
     *  @param[in] function The callback, receiving the `task_t` or the task index as an argument.
     *  @sa `for_each_static` for a more "balanced" evenly-splittable workload.
     */
    template <typename function_type_>
    void for_each_dynamic(task_index_t const n, function_type_ const &function) noexcept {

#if _FU_DETECT_CPP_17
        // ? Exception handling and aggregating return values drastically increases code complexity
        static_assert((std::is_nothrow_invocable_r<void, function_type_, task_t>::value ||
                       std::is_nothrow_invocable_r<void, function_type_, task_index_t>::value),
                      "The callback must be invocable with a `task_t` or a `task_index_t` argument");
#endif

        // If there is just one thread, all work is done on the current thread
        if (total_threads_ == 1) {
            for (task_index_t i = 0; i < n; ++i) function(task_t {0, i});
            return;
        }

        // No need to slice the work if there is just one task
        if (n == 0) return;
        if (n == 1) return function(task_t {0, 0});

        // Store closure address
        task_lambda_pointer_ = std::addressof(function);
        task_trampoline_pointer_ = &_call_lambda_from_thread<function_type_>;
        task_parts_count_ = n;
        task_parts_passed_.store(0, std::memory_order_relaxed);
        task_parts_remaining_.store(n, std::memory_order_relaxed);
        task_generation_.fetch_add(1, std::memory_order_release); // ? Wake up sleepers

        // Execute on the current thread
        _worker_loop_for_dynamic_tasks(0);

        // We may be in the in-flight position, where the current thread is already receiving
        // tasks beyond the `task_parts_count_` index, but the worker threads are still executing
        // tasks and haven't decremented the `task_parts_remaining_` variable yet.
        while (task_parts_remaining_.load(std::memory_order_acquire)) std::this_thread::yield();

        // Optional reset of the task variables for debuggability
        _reset_task();
    }

  private:
    void _reset_task() noexcept {
        task_parts_count_ = 0;
        task_lambda_pointer_ = nullptr;
        task_trampoline_pointer_ = nullptr;
    }

    /**
     *  @brief A trampoline function that is used to call the user-defined lambda.
     *  @param[in] punned_lambda_pointer The pointer to the user-defined lambda.
     *  @param[in] thread_index The index of the thread that is executing this function.
     */
    template <typename function_type_>
    static void _call_lambda_from_thread(punned_task_context_t punned_lambda_pointer, task_t task) noexcept {
        auto const &lambda_object = *static_cast<function_type_ const *>(punned_lambda_pointer);
        lambda_object(task);
    }

    /**
     *  @brief The worker thread loop that is called by each of `this->workers_`.
     *  @param[in] thread_index The index of the thread that is executing this function.
     */
    void _worker_loop(thread_index_t const thread_index) noexcept {
        std::size_t last_task_generation = 0;

        while (true) {
            // Wait for either: a new ticket or a stop flag
            std::size_t new_task_generation;
            bool wants_to_stop;
            while ((new_task_generation = task_generation_.load(std::memory_order_acquire)) == last_task_generation &&
                   (wants_to_stop = stop_.load(std::memory_order_acquire)) == false)
                std::this_thread::yield();

            if (wants_to_stop) return;

            // Check if we are operating in the "dynamic" eager mode or a balanced "static" mode
            bool const is_static = task_parts_count_ == total_threads_;
            if (is_static && task_parts_count_) {
                task_trampoline_pointer_(task_lambda_pointer_, {thread_index, thread_index});
                // ! The decrement must come after the task is executed
                _FU_MAYBE_UNUSED task_index_t const before_decrement =
                    task_parts_remaining_.fetch_sub(1, std::memory_order_acq_rel);
                assert(before_decrement > 0 && "We can't be here if there are no worker threads");
            }
            else { _worker_loop_for_dynamic_tasks(thread_index); }
            last_task_generation = new_task_generation;
        }
    }

    void _worker_loop_for_dynamic_tasks(task_index_t const thread_index) noexcept {
        // Unlike the thread-balanced mode, we need to keep track of the number of passed tasks.
        // The traditional way to achieve that is to use the same single atomic `task_parts_remaining_`
        // variable, but using `compare_exchange_weak` interfaces. It's much more expensive on modern
        // CPUs, so we use an additional atomic variable and have 2x atomic increments of 2x atomic variables,
        // instead of 1x atomic load and 1x atomic CAS on 1x atomic variable (in optimistic low-contention case).
        while (true) {
            // The relative order of executed tasks doesn't matter here, so we can use relaxed memory order.
            task_index_t const new_task_index = task_parts_passed_.fetch_add(1, std::memory_order_relaxed);
            if (new_task_index >= task_parts_count_) break;
            task_trampoline_pointer_(task_lambda_pointer_, {thread_index, new_task_index});
            // ! The decrement must come after the task is executed
            _FU_MAYBE_UNUSED task_index_t const before_decrement =
                task_parts_remaining_.fetch_sub(1, std::memory_order_acq_rel);
            assert(before_decrement > 0 && "We can't be here if there are no tasks left");
        }
    }
};

using fork_union_t = fork_union<std::allocator<std::thread>>;

} // namespace ashvardanian
