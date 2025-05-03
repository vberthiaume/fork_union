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
 */
template <typename allocator_type_ = std::allocator<std::byte>>
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
    using trampoline_pointer_t = void (*)(punned_task_context_t, task_t const &);

    struct c_thread_callback_t {
        trampoline_pointer_t callable {nullptr};
        punned_task_context_t context {nullptr};
    };

  private:
    // Thread-pool-specific variables:
    allocator_t allocator_ {};
    std::thread *workers_ {nullptr};
    thread_index_t total_threads_ {0};
    alignas(std::max_align_t) std::atomic<bool> stop_ {false};

    // Task-specific variables:
    punned_task_context_t task_lambda_pointer_ {nullptr};    // ? Pointer to the users lambda
    trampoline_pointer_t task_trampoline_pointer_ {nullptr}; // ? Calls the lambda
    task_index_t task_parts_count_ {0};
    alignas(std::max_align_t) std::atomic<task_index_t> task_parts_remaining_ {0};
    alignas(std::max_align_t) std::atomic<task_index_t> task_parts_passed_ {0}; // ? Only used in eager mode
    alignas(std::max_align_t) std::atomic<std::size_t> task_generation_ {0};

  public:
    fork_union(allocator_t const &alloc = {}) noexcept : allocator_(alloc) {}
    fork_union(fork_union &&) = delete;
    fork_union(fork_union const &) = delete;
    fork_union &operator=(fork_union &&) = delete;
    fork_union &operator=(fork_union const &) = delete;

    /**
     *  @brief Creates a thread-pool with the given number of threads.
     *  @note This is the de-facto @b constructor, and can only be called once.
     *  @param[in] planned_threads The number of threads to be used. Should be larger than one.
     *  @retval false if the number of threads is zero or the "workers" allocation failed.
     *  @retval true if the thread-pool was created successfully, started, and is ready to use.
     */
    bool try_fork(std::size_t const planned_threads) noexcept {
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

    ~fork_union() noexcept {
        if (total_threads_ == 0) return;                                    // ? Uninitialized
        if (total_threads_ == 1) return;                                    // ? No worker threads to join
        assert(task_parts_remaining_.load(std::memory_order_seq_cst) == 0); // ! No tasks must be running

        // Stop all threads and wait for them to finish
        _reset_task();
        stop_.store(true, std::memory_order_seq_cst);

        thread_index_t const worker_threads = total_threads_ - 1;
        for (thread_index_t i = 0; i != worker_threads; ++i) {
            workers_[i].join();    // ? Wait for the thread to finish
            workers_[i].~thread(); // ? Call destructor
        }

        // Deallocate the thread pool
        allocator_.deallocate(workers_, worker_threads);

        // We don't have to reset the following variables, but it's a good practice
        total_threads_ = 0;
        workers_ = nullptr;
    }

    /**
     *  @brief  Executes a function in parallel on the current and all worker threads,
     *          calling it @p (n) times, slicing the workloads size @p (n) into continuous
     *          chunks allocated to each thread.
     */
    template <typename function_type_>
    void for_each(task_index_t const n, function_type_ const &function) noexcept {
        for_each_range(n, [function](task_t start_task, task_index_t count) noexcept {
            for (task_index_t i = 0; i < count; ++i)
                function(task_t {start_task.thread_index, start_task.task_index + i});
        });
    }

    /**
     *  @brief  Executes a function in parallel on the current and all worker threads,
     *          calling it @b (k) times, slicing the workloads size @p (n) into continuous
     *          @b (k) continuous chunks.
     */
    template <typename function_type_>
    void for_each_range(task_index_t const n, function_type_ const &function) noexcept {

        // No need to slice the workload if we only have one thread
        assert(total_threads_ != 0 && "Thread pool not initialized");
        if (total_threads_ == 1) return function(task_t {0, 0}, n);

        // Divide and round-up the workload size per thread
        task_index_t const n_per_thread = (n + total_threads_ - 1) / total_threads_;
        for_each_thread([n, n_per_thread, function](thread_index_t thread_index) noexcept {
            task_index_t const begin = thread_index * n_per_thread;
            task_index_t const count = std::min(n, begin + n_per_thread) - begin;
            function(task_t {thread_index, begin}, count);
        });
    }

    /**
     *  @brief Executes a function in parallel on the current and all worker threads.
     *  @param[in] function The function to be executed, receiving the thread index as an argument.
     */
    template <typename function_type_>
    void for_each_thread(function_type_ const &function) noexcept {
        if (total_threads_ == 1) return function(0);

        // Store closure address
        task_lambda_pointer_ = std::addressof(function);
        task_trampoline_pointer_ = &_call_lambda_from_thread<function_type_>;
        task_parts_count_ = total_threads_;
        task_parts_remaining_.store(total_threads_ - 1, std::memory_order_seq_cst);
        task_generation_.fetch_add(1, std::memory_order_seq_cst); // ? Wake up sleepers

        function(0); // Execute on the current thread
        while (task_parts_remaining_.load(std::memory_order_seq_cst)) std::this_thread::yield();
        _reset_task();
    }

    /**
     *  @brief Executes a function in parallel on the current and all worker threads,
     *         calling it @p (n) times, in no particular order, stealing the workload
     *         as soon as any more work is available.
     */
    template <typename function_type_>
    void eager(task_index_t const n, function_type_ const &function) noexcept {
        // If there is just one thread, all work is done on the current thread
        if (total_threads_ == 1) {
            for (task_index_t i = 0; i < n; ++i) function(i);
            return;
        }

        // Store closure address
        task_lambda_pointer_ = std::addressof(function);
        task_trampoline_pointer_ = &_call_lambda_from_thread<function_type_>;
        task_parts_count_ = n;
        task_parts_passed_.store(0, std::memory_order_seq_cst);
        task_parts_remaining_.store(n, std::memory_order_seq_cst);
        task_generation_.fetch_add(1, std::memory_order_seq_cst); // ? Wake up sleepers

        _worker_loop_for_eager_task(0); // Execute on the current thread
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
    static void _call_lambda_from_thread(void const *punned_lambda_pointer, task_t const &task) noexcept {
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
            while ((new_task_generation = task_generation_.load(std::memory_order_seq_cst)) == last_task_generation &&
                   (wants_to_stop = stop_.load(std::memory_order_seq_cst)) == false)
                std::this_thread::yield();

            if (wants_to_stop) return;

            // Check if we are operating in the "eager" or a more-balanced "parallel" mode
            bool const one_part_per_thread = task_parts_count_ == total_threads_;
            if (one_part_per_thread && task_parts_count_) {
                task_trampoline_pointer_(task_lambda_pointer_, {thread_index, thread_index});
                // ! The decrement must come after the task is executed
                task_index_t const before_decrement = task_parts_remaining_.fetch_sub(1, std::memory_order_seq_cst);
                assert(before_decrement > 0 && "We can't be here if there are no worker threads");
            }
            else { _worker_loop_for_eager_task(thread_index); }
            last_task_generation = new_task_generation;
        }
    }

    void _worker_loop_for_eager_task(task_index_t const thread_index) noexcept {
        // Unlike the thread-balanced mode, we need to keep track of the number of passed tasks.
        // The traditional way to achieve that is to use the same single atomic `task_parts_remaining_`
        // variable, but using `compare_exchange_weak` interfaces. It's much more expensive on modern
        // CPUs, so we use an additional atomic variable and have 2x atomic increments of 2x atomic variables,
        // instead of 1x atomic load and 1x atomic CAS on 1x atomic variable (in optimistic low-contention case).
        while (true) {
            task_index_t const new_task_index = task_parts_passed_.fetch_add(1, std::memory_order_seq_cst);
            if (new_task_index >= task_parts_count_) break;
            task_trampoline_pointer_(task_lambda_pointer_, {thread_index, new_task_index});
            // ! The decrement must come after the task is executed
            task_index_t const before_decrement = task_parts_remaining_.fetch_sub(1, std::memory_order_seq_cst);
            assert(before_decrement > 0 && "We can't be here if there are no tasks left");
        }
    }
};

using fork_union_t = fork_union<std::allocator<std::thread>>;

} // namespace ashvardanian
