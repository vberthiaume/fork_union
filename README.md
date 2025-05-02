# Fork Union

`fork_union` is a thread-pool for "Fork-Join" [SIMT-style](https://en.wikipedia.org/wiki/Single_instruction,_multiple_threads) parallelism in modern C++.
It's quite different from most open-source C++ thread-pool implementations, generally designed around a `std::queue` of `std::function` tasks, synchronized by a `std::mutex`.
Wrapping tasks into `std::function` is expensive, so is growing the `std::queue` and locking the `std::mutex` under contention.
When you can avoid it - you should.

![`fork_union` banner](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/fork_union.jpg?raw=true)

The most common alternative for that is using OpenMP, but it's not great for fine-grained parallelism, when different pieces of your application logic need to work on different sets of threads.
This is where `fork_union` comes in with a minimalistic STL implementation of a thread-pool, avoiding dynamic memory allocations on the hot path, and prioritizing lock-free user-space atomics to [system calls](https://en.wikipedia.org/wiki/System_call).

## Usage

To integrate into your project, either just copy the `fork_union.hpp` file into your project, add a Git submodule, or CMake:

```cmake
FetchContent_Declare(
    fork_union
    GIT_REPOSITORY
    https://github.com/ashvardanian/fork_union
)
FetchContent_MakeAvailable(fork_union)
```

Then, in your C++ code:

```cpp
#include <fork_union.hpp>

namespace fu = ashvardanian::fork_union;

int main() {
    fu::thread_pool pool(std::thread::hardware_concurrency());

    // Dispatch a callback to each thread in the pool
    pool.parallel([](std::size_t thread_index) {
        std::printf("Hello from thread %zu\n", thread_index);
    });

    // Execute 1000 tasks in parallel, expecting them to have comparable runtimes
    // and mostly co-locating subsequent tasks on the same thread. Analogous to:
    //
    //      #pragma omp parallel for schedule(static)
    //      for (int i = 0; i < 1000; ++i) { ... }
    //
    // You can also think about it as a shortcut for the `for_each_slice` + `for`.
    pool.for_each(1000, [](std::size_t task_index) {
        std::printf("Running task %zu of 3\n", task_index + 1);
    });
    pool.for_each_slice(1000, [](std::size_t first_index, std::size_t last_index) {
        std::printf("Running slice [%zu, %zu)\n", first_index, last_index);
    });

    // Like `for_each`, but each thread greedily steals tasks, without waiting for  
    // the others or expecting individual tasks to have same runtimes. Analogous to:
    //
    //      #pragma omp parallel for schedule(dynamic, 1)
    //      for (int i = 0; i < 3; ++i) { ... }
    pool.eager(3, [](std::size_t task_index) {
        std::printf("Running eager task %zu of 1000\n", task_index + 1);
    });
    return 0;
}
```

That's it.

## Why Not Use $𝑋$

### `std::mutex`

Unlike the `std::atomic`, the `std::mutex` is a system call, and it can be expensive to acquire and release.
It's implementations generally have 2 executable paths:

- the fast path, where the mutex is not contended, where it first tries to grab the mutex via a compare-and-swap operation, and if it succeeds, it returns immediately.
- the slow path, where the mutex is contended, and it has to go through the kernel to block the thread until the mutex is available.

On Linux the latter translates to a ["futex" syscall](https://en.wikipedia.org/wiki/Futex), which is expensive.

### Tasks, Queues, Conditional Variables, and Futures

C++ has rich functionality for concurrent applications, like `std::future`, `std::packaged_task`, `std::function`, `std::queue`, `std::conditional_variable`, and so on.
Most of those, I believe, aren't unusable in Big-Data applications, where you always operate in memory-constrained environments:

- The idea of raising a `std::bad_alloc` exception, when there is no memory left, and just hoping that someone up the call stack will catch it is simply not a great design idea for any Systems Engineering.
- The threat of having to synchronize ~200 physical CPU cores across 2-8 sockets, and potentially dozens of [NUMA](https://en.wikipedia.org/wiki/Non-uniform_memory_access) nodes around a shared global memory allocator, practically means you can't have predictable performance.

As we focus on a simpler ~~concurrency~~ parallelism model, we can avoid the complexity of allocating shared states, wrapping callbacks into some heap-allocated "tasks", and a lot of other boilerplate.
Less work - more performance.

### Other Thread Pools

There are several popular thread-pool implementations in C++:

- [`progschj/ThreadPool`](https://github.com/progschj/ThreadPool) - 8.3 K stars
- [`bshoshany/thread-pool`](https://github.com/bshoshany/thread-pool) - 2.5 K stars
- [`vit-vit/CTPL`](https://github.com/vit-vit/CTPL) - 1.9 K stars
- [`mtrebi/thread-pool`](https://github.com/mtrebi/thread-pool) - 1.2 K stars

They are all seemingly designed around non-SIMT-style parallelism, where you have a queue of tasks, and a pool of threads that pick up tasks from the queue.
