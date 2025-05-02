# Fork Union

The __`fork_union`__ library is a thread-pool for "Fork-Join" [SIMT-style](https://en.wikipedia.org/wiki/Single_instruction,_multiple_threads) parallelism in modern C++.
It's quite different from most open-source C++ thread-pool implementations, generally designed around a `std::queue` of `std::function` tasks, synchronized by a `std::mutex`.
Wrapping tasks into `std::function` is expensive, so is growing the `std::queue` and locking the `std::mutex` under contention.
When you can avoid it - you should.
OpenMP-like use-cases are the perfect example of that!

![`fork_union` banner](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/fork_union.jpg?raw=true)

OpenMP, however, isn't great for fine-grained parallelism, when different pieces of your application logic need to work on different sets of threads.
This is where __`fork_union`__ comes in with a minimalistic STL implementation of a thread-pool, avoiding dynamic memory allocations and exceptions on the hot path, and prioritizing lock-free user-space "atomics" to [system calls](https://en.wikipedia.org/wiki/System_call).

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
#include <fork_union.hpp>   // `fork_union_t`
#include <cstdio>           // `stderr`
#include <cstdlib>          // `EXIT_SUCCESS`

namespace av = ashvardanian;

int main() {
    av::fork_union_t pool;
    if (!pool.try_fork(std::thread::hardware_concurrency())) {
        std::fprintf(stderr, "Failed to fork the threads\n");
        return EXIT_FAILURE;
    }

    // Dispatch a callback to each thread in the pool
    pool.parallel([](std::size_t thread_index) noexcept {
        std::printf("Hello from thread %zu\n", thread_index);
    });

    // Execute 1000 tasks in parallel, expecting them to have comparable runtimes
    // and mostly co-locating subsequent tasks on the same thread. Analogous to:
    //
    //      #pragma omp parallel for schedule(static)
    //      for (int i = 0; i < 1000; ++i) { ... }
    //
    // You can also think about it as a shortcut for the `for_each_slice` + `for`.
    pool.for_each(1000, [](std::size_t task_index) noexcept {
        std::printf("Running task %zu of 3\n", task_index + 1);
    });
    pool.for_each_slice(1000, [](std::size_t first_index, std::size_t last_index) noexcept {
        std::printf("Running slice [%zu, %zu)\n", first_index, last_index);
    });

    // Like `for_each`, but each thread greedily steals tasks, without waiting for  
    // the others or expecting individual tasks to have same runtimes. Analogous to:
    //
    //      #pragma omp parallel for schedule(dynamic, 1)
    //      for (int i = 0; i < 3; ++i) { ... }
    pool.eager(3, [](std::size_t task_index) noexcept {
        std::printf("Running eager task %zu of 1000\n", task_index + 1);
    });
    return EXIT_SUCCESS;
}
```

That's it.

## Why Not Use $𝑋$

There are many other thread-pool implementations, that are more feature-rich, but have different limitations and design goals:

- [`progschj/ThreadPool`](https://github.com/progschj/ThreadPool) ![https://github.com/progschj/ThreadPool](https://img.shields.io/github/stars/progschj/ThreadPool)
- [`bshoshany/thread-pool`](https://github.com/bshoshany/thread-pool) ![https://github.com/bshoshany/thread-pool](https://img.shields.io/github/stars/bshoshany/thread-pool)
- [`vit-vit/CTPL`](https://github.com/vit-vit/CTPL) ![https://github.com/vit-vit/CTPL](https://img.shields.io/github/stars/vit-vit/CTPL)
- [`mtrebi/thread-pool`](https://github.com/mtrebi/thread-pool) ![https://github.com/mtrebi/thread-pool](https://img.shields.io/github/stars/mtrebi/thread-pool)

Those are not designed for the same OpenMP-like use-cases as __`fork_union`__.
Instead, they primarily focus on task-queueing, that requires a lot more work.

### Locks and Mutexes

Unlike the `std::atomic`, the `std::mutex` is a system call, and it can be expensive to acquire and release.
It's implementations generally have 2 executable paths:

- the fast path, where the mutex is not contended, where it first tries to grab the mutex via a compare-and-swap operation, and if it succeeds, it returns immediately.
- the slow path, where the mutex is contended, and it has to go through the kernel to block the thread until the mutex is available.

On Linux the latter translates to a ["futex" syscall](https://en.wikipedia.org/wiki/Futex), which is expensive.

### Memory Allocations

C++ has rich functionality for concurrent applications, like `std::future`, `std::packaged_task`, `std::function`, `std::queue`, `std::conditional_variable`, and so on.
Most of those, I believe, aren't unusable in Big-Data applications, where you always operate in memory-constrained environments:

- The idea of raising a `std::bad_alloc` exception, when there is no memory left, and just hoping that someone up the call stack will catch it is simply not a great design idea for any Systems Engineering.
- The threat of having to synchronize ~200 physical CPU cores across 2-8 sockets, and potentially dozens of [NUMA](https://en.wikipedia.org/wiki/Non-uniform_memory_access) nodes around a shared global memory allocator, practically means you can't have predictable performance.

As we focus on a simpler ~~concurrency~~ parallelism model, we can avoid the complexity of allocating shared states, wrapping callbacks into some heap-allocated "tasks", and a lot of other boilerplate.
Less work - more performance.

## Testing

To run the tests, use CMake:

```bash
cmake -B build_release
cmake --build build_release --config Release     
build_release/fork_union_test
```

For debug builds, consider using the VS Code debugger presets or the following commands:

```bash
cmake -B build_debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build_debug --config Debug
build_debug/fork_union_test
```
