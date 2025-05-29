# Fork Union 🍴

The __`fork_union`__ library is a thread-pool for "Fork-Join" [SIMT-style](https://en.wikipedia.org/wiki/Single_instruction,_multiple_threads) parallelism for Rust and C++.
It's quite different from most open-source thread-pool implementations, generally designed around heap-allocated "queues of tasks", synchronized by a "mutex".
In C++, wrapping tasks into a `std::function` is expensive, as is growing the `std::queue` and locking the `std::mutex` under contention.
Same for Rust.
When you can avoid it - you should.
OpenMP-like use-cases are the perfect example of that!

![`fork_union` banner](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/fork_union.jpg?raw=true)

OpenMP, however, isn't great for fine-grained parallelism, when different pieces of your application logic need to work on different sets of threads.
This is where __`fork_union`__ comes in with a minimalistic STL implementation of a thread-pool, avoiding dynamic memory allocations and exceptions on the hot path, and prioritizing lock-free and [CAS](https://en.wikipedia.org/wiki/Compare-and-swap)-free user-space "atomics" to [system calls](https://en.wikipedia.org/wiki/System_call).

## Usage

The __`fork_union`__ is dead-simple!
There is no nested parallelism, exception-handling, or "futures promises".
The thread pool has just one core API - `broadcast` to launch a callback on each thread.
The higher-level API for index-addressable tasks are:

- `for_n` - for individual evenly-sized tasks.
- `for_n_dynamic` - for individual unevenly-sized tasks.
- `for_slices` - for slices of evenly-sized tasks.

Both are available in C++ and Rust.

### Usage in Rust

A minimal example may look like this:

```rs
use fork_union as fu;
let pool = fu::spawn(2);
pool.broadcast(|thread_index| {
    println!("Hello from thread # {}", thread_index + 1);
});
```

Higher-level APIs distribute tasks across the threads in the pool:

```rs
fu::for_n(pool, 100, |prong| {
    println!("Running task {} on thread # {}",
        prong.task_index + 1, prong.thread_index + 1);
});
fu::for_slices(pool, 100, |prong, count| {
    println!("Running slice [{}, {}) on thread # {}",
        prong.task_index, prong.task_index + count, prong.thread_index + 1);
});
fu::for_n_dynamic(pool, 100, |prong| {
    println!("Running task {} on thread # {}",
        prong.task_index + 1, prong.thread_index + 1);
});
```

A safer `try_spawn_in` interface is recommended, using the Allocator API.
A more realistic example may look like this:

```rs
#![feature(allocator_api)]
use std::thread;
use std::error::Error;
use std::alloc::Global;
use fork_union as fu;

fn heavy_math(_: usize) {}

fn main() -> Result<(), Box<dyn Error>> {
    let pool = fu::ThreadPool::try_spawn(4)?;
    let pool = fu::ThreadPool::try_spawn_in(4, Global)?;
    let pool = fu::ThreadPool::try_named_spawn("heavy-math", 4)?;
    let pool = fu::ThreadPool::try_named_spawn_in("heavy-math", 4, Global)?;
    fu::for_n_dynamic(pool, 400, |prong| {
        heavy_math(prong.1);
    });
    Ok(())
}
```

### Usage in C++

To integrate into your C++ project, either just copy the `include/fork_union.hpp` file into your project, add a Git submodule, or CMake.
For a Git submodule, run:

```bash
git submodule add https://github.com/ashvardanian/fork_union.git extern/fork_union
```

Alternatively, using CMake:

```cmake
FetchContent_Declare(
    fork_union
    GIT_REPOSITORY
    https://github.com/ashvardanian/fork_union
)
FetchContent_MakeAvailable(fork_union)
target_link_libraries(your_target PRIVATE fork_union::fork_union)
```

Then, include the header in your C++ code:

```cpp
#include <fork_union.hpp>   // `thread_pool_t`
#include <cstdio>           // `stderr`
#include <cstdlib>          // `EXIT_SUCCESS`

namespace fu = ashvardanian::fork_union;

int main() {
    fu::thread_pool_t pool;
    if (!pool.try_spawn(std::thread::hardware_concurrency())) {
        std::fprintf(stderr, "Failed to fork the threads\n");
        return EXIT_FAILURE;
    }

    // Dispatch a callback to each thread in the pool
    pool.broadcast([&](std::size_t thread_index) noexcept {
        std::printf("Hello from thread # %zu (of %zu)\n", thread_index + 1, pool.count_threads());
    });

    // Execute 1000 tasks in parallel, expecting them to have comparable runtimes
    // and mostly co-locating subsequent tasks on the same thread. Analogous to:
    //
    //      #pragma omp parallel for schedule(static)
    //      for (int i = 0; i < 1000; ++i) { ... }
    //
    // You can also think about it as a shortcut for the `for_slices` + `for`.
    fu::for_n(pool, 1000, [](std::size_t task_index) noexcept {
        std::printf("Running task %zu of 3\n", task_index + 1);
    });
    fu::for_slices(pool, 1000, [](std::size_t first_index, std::size_t count) noexcept {
        std::printf("Running slice [%zu, %zu)\n", first_index, first_index + count);
    });

    // Like `for_n`, but each thread greedily steals tasks, without waiting for  
    // the others or expecting individual tasks to have same runtimes. Analogous to:
    //
    //      #pragma omp parallel for schedule(dynamic, 1)
    //      for (int i = 0; i < 3; ++i) { ... }
    fu::for_n_dynamic(pool, 3, [](std::size_t task_index) noexcept {
        std::printf("Running dynamic task %zu of 1000\n", task_index + 1);
    });
    return EXIT_SUCCESS;
}
```

That's it.

## Why Not Use $𝑋$

There are many other thread-pool implementations, that are more feature-rich, but have different limitations and design goals.

- Modern C++: [`taskflow/taskflow`](https://github.com/taskflow/taskflow), [`progschj/ThreadPool`](https://github.com/progschj/ThreadPool), [`bshoshany/thread-pool`](https://github.com/bshoshany/thread-pool)
- Traditional C++: [`vit-vit/CTPL`](https://github.com/vit-vit/CTPL), [`mtrebi/thread-pool`](https://github.com/mtrebi/thread-pool)
- Rust: [`tokio-rs/tokio`](https://github.com/tokio-rs/tokio), [`rayon-rs/rayon`](https://github.com/rayon-rs/rayon), [`smol-rs/smol`](https://github.com/smol-rs/smol)

Those are not designed for the same OpenMP-like use-cases as __`fork_union`__.
Instead, they primarily focus on task queueing, which requires a lot more work.

### Locks and Mutexes

Unlike the `std::atomic`, the `std::mutex` is a system call, and it can be expensive to acquire and release.
Its implementations generally have 2 executable paths:

- the fast path, where the mutex is not contended, where it first tries to grab the mutex via a compare-and-swap operation, and if it succeeds, it returns immediately.
- the slow path, where the mutex is contended, and it has to go through the kernel to block the thread until the mutex is available.

On Linux, the latter translates to a ["futex" syscall](https://en.wikipedia.org/wiki/Futex), which is expensive.

### Memory Allocations

C++ has rich functionality for concurrent applications, like `std::future`, `std::packaged_task`, `std::function`, `std::queue`, `std::conditional_variable`, and so on.
Most of those, I believe, aren't unusable in Big-Data applications, where you always operate in memory-constrained environments:

- The idea of raising a `std::bad_alloc` exception, when there is no memory left, and just hoping that someone up the call stack will catch it is simply not a great design idea for any Systems Engineering.
- The threat of having to synchronize ~200 physical CPU cores across 2-8 sockets, and potentially dozens of [NUMA](https://en.wikipedia.org/wiki/Non-uniform_memory_access) nodes around a shared global memory allocator, practically means you can't have predictable performance.

As we focus on a simpler ~~concurrency~~ parallelism model, we can avoid the complexity of allocating shared states, wrapping callbacks into some heap-allocated "tasks", and a lot of other boilerplate.
Less work - more performance.

### Atomics and CAS

Once you get to the lowest-level primitives on concurrency you end up with the `std::atomic` and a small set of hardware-supported atomic instructions.
Hardware implements it differently:

- x86 is built around the "Total Store Order" (TSO) [memory consistency model](https://en.wikipedia.org/wiki/Memory_ordering) and provides `LOCK` variants of the `ADD` and `CMPXCHG`, which act as full-blown "fences" - no loads or stores can be reordered across it.
- Arm, on the other hand, has a "weak" memory model, and provides a set of atomic instructions that are not fences, that match C++ concurrency model, offering `acquire`, `release`, and `acq_rel` variants of each atomic instruction—such as `LDADD`, `STADD`, and `CAS` - which allow precise control over visibility and ordering, especially with the introduction of "Large System Extension" (LSE) instructions in Armv8.1.

In practice, a locked atomic on x86 requires the cache line in the Exclusive state in the requester's L1 cache.
This will incur a coherence transaction (Read-for-Ownership) if some other core had the line.
Both Intel and AMD handle this similarly.

It makes [Arm and Power much more suitable for lock-free programming](https://arangodb.com/2021/02/cpp-memory-model-migrating-from-x86-to-arm/) and concurrent data structures, but some observations hold for both platforms.
Most importantly, "Compare and Swap" (CAS) is a very expensive operation, and should be avoided at all costs.

On x86, for example, the `LOCK ADD` [can easily take 50 CPU cycles](https://travisdowns.github.io/blog/2020/07/06/concurrency-costs), being 50x slower than a regular `ADD` instruction, but still easily 5-10x faster than a `LOCK CMPXCHG` instruction.
Once the contention rises, the gap naturally widens, and is further amplified by the increased "failure" rate of the CAS operation, when the value being compared has already changed.
That's why for the "dynamic" mode, we resort to using an additional atomic variable as opposed to more typical CAS-based implementations.

### Alignment

Assuming a thread-pool is a heavy object anyway, nobody will care if it's a bit larger than expected.
That allows us to over-align the internal counters to `std::hardware_destructive_interference_size` to avoid false sharing.
In that case, even on x86, where the entire cache will be exclusively owned by a single thread, in eager mode, we end up effectively "pipelining" the execution, where one thread may be incrementing the "in-flight" counter, while the other is decrementing the "remaining" counter, and others are executing the loop body in-between.

## Performance

One of the most common parallel workloads is the N-body simulation ¹.
An implementation is available in both C++ and Rust in `scripts/nbody.cpp` and `scripts/nbody.rs` respectively.
Both are extremely light-weight and involve little logic outside of number-crunching, so both can be easily profiled with `time` and introspected with `perf` Linux tools. 

---

C++ benchmarking results for $N=128$ bodies and $I=1e6$ iterations:

| Machine        | OpenMP (D) | OpenMP (S) | Fork Union (D) | Fork Union (S) |
| :------------- | ---------: | ---------: | -------------: | -------------: |
| 16x Intel SPR  |      20.3s |      16.0s |          18.1s |          10.3s |
| 12x Apple M2   |          ? |   1m:16.7s |     1m:30.3s ² |     1m:40.7s ² |
| 96x Graviton 4 |      32.2s |      20.8s |          39.8s |          26.0s |

Rust benchmarking results for $N=128$ bodies and $I=1e6$ iterations:

| Machine        | Rayon (D) | Rayon (S) | Fork Union (D) | Fork Union (S) |
| :------------- | --------: | --------: | -------------: | -------------: |
| 16x Intel SPR  |     51.4s |     38.1s |          15.9s |           9.8s |
| 12x Apple M2   |  3m:23.5s |   2m:0.6s |        4m:8.4s |       1m:20.8s |
| 96x Graviton 4 |  2m:13.9s |  1m:35.6s |          18.9s |          10.1s |

> ¹ Another common workload is "Parallel Reductions" covered in a separate [repository](https://github.com/ashvardanian/ParallelReductionsBenchmark).
> ² When a combination of performance and efficiency cores is used, dynamic stealing may be more efficient than static slicing.

## Safety & Logic

There are only 3 core atomic variables in this thread-pool, and some of them are practically optional.
Let's call every invocation of a `for_*` API - a "fork", and every exit from it a "join".

| Variable          | Users Perspective            | Internal Usage                        |
| :---------------- | :--------------------------- | :------------------------------------ |
| `stop`            | Stop the entire thread-pool  | Tells workers when to exit the loop   |
| `fork_generation` | "Forks" called since init    | Tells workers to wake up on new forks |
| `threads_to_sync` | Threads not joined this fork | Tells main thread when workers finish |

__Why don't we need atomics for "total_threads"?__
The only way to change the number of threads is to `stop_and_reset` the entire thread-pool and then `try_spawn` it again.
Either of those operations can only be called from one thread at a time and never coincides with any running tasks.
That's ensured by the `stop`.

__Why don't we need atomics for a "job pointer"?__
A new task can only be submitted from one thread, that updates the number of parts for each new fork.
During that update, the workers are asleep, spinning on old values of `fork_generation` and `stop`.
They only wake up and access the new value once `fork_generation` increments, ensuring safety.

__How do we deal with overflows and `SIZE_MAX`-sized tasks?__
The library entirely avoids saturating multiplication and only uses one saturating addition in "release" builds.
To test the consistency of arithmetic, the C++ template class can be instantiated with a custom `index_t`, such as `std::uint8_t` or `std::uint16_t`.
In the former case, no more than 255 threads can operate and no more than 255 tasks can be addressed, allowing us to easily test every weird corner case of [0:255] threads competing for [0:255] tasks.

## Testing and Benchmarking

To run the C++ tests, use CMake:

```bash
cmake -B build_release -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_COMPILER=clang++-15
cmake --build build_release --config Release
build_release/scripts/fork_union_test_cpp20
```

For C++ debug builds, consider using the VS Code debugger presets or the following commands:

```bash
cmake --build build_debug --config Debug
build_debug/scripts/fork_union_test_cpp20
```

To build with an alternative compiler, like LLVM Clang, use the following command:

```bash
sudo apt-get install libomp-15-dev clang++-15 # OpenMP version must match Clang
cmake -B build_debug -D CMAKE_BUILD_TYPE=Debug -D CMAKE_CXX_COMPILER=clang++-15
cmake --build build_debug --config Debug
build_debug/scripts/fork_union_test_cpp20
```

For Rust, use the following command:

```bash
rustup toolchain install nightly
cargo +nightly test --release
```
