#include <cstdio>    // `std::printf`, `std::fprintf`
#include <cstdlib>   // `EXIT_FAILURE`, `EXIT_SUCCESS`
#include <vector>    // `std::vector`
#include <algorithm> // `std::sort`

#include <fork_union.hpp>

namespace av = ashvardanian;

using test_func_t = bool() noexcept;

struct test_t {
    char const *name;
    test_func_t *function;
};

static bool test_try_spawn_success() noexcept {
    av::fork_union_t pool;
    auto const count_threads = std::thread::hardware_concurrency();
    if (!pool.try_spawn(count_threads)) return false;
    return true;
}

static bool test_try_spawn_zero() noexcept {
    av::fork_union_t pool;
    return !pool.try_spawn(0u);
}

/** @brief Make sure that `for_each_thread` is called from each thread. */
static bool test_for_each_thread() noexcept {
    auto const count_threads = std::thread::hardware_concurrency();
    std::vector<char> visited(count_threads, 0);
    {
        av::fork_union_t pool;
        if (!pool.try_spawn(count_threads)) return false;
        pool.for_each_thread([&](std::size_t const thread_index) noexcept { visited[thread_index] = 1; });
    }
    for (size_t i = 0; i < count_threads; ++i)
        if (!visited[i]) return false;
    return true;
}

/** @brief Make sure that `for_each_static` is called the right number of times with the right task IDs. */
static bool test_for_each_static() noexcept {
    constexpr std::size_t expected_parts = 10'000'000;

    std::vector<std::size_t> visited(expected_parts, 0);
    std::atomic<std::size_t> counter = 0;
    {
        av::fork_union_t pool;
        auto const count_threads = std::thread::hardware_concurrency();
        if (!pool.try_spawn(count_threads)) return false;
        pool.for_each_static(expected_parts, [&](std::size_t const task_index) noexcept {
            // ? Relax the memory order, as we don't care about the order of the results, will sort 'em later
            std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
            visited[count_populated] = task_index;
        });
    }

    // Make sure that all task IDs are unique and form the full range of [0, `expected_parts`).
    std::sort(visited.begin(), visited.end());
    return counter.load() == expected_parts && std::adjacent_find(visited.begin(), visited.end()) == visited.end() &&
           std::is_sorted(visited.begin(), visited.end()) && visited.front() == (0) &&
           visited.back() == (expected_parts - 1);
}

/** @brief Make sure that `for_each_dynamic` is called the right number of times with the right task IDs. */
static bool test_for_each_dynamic() noexcept {
    constexpr std::size_t expected_parts = 10'000'000;

    av::fork_union_t pool;
    auto const count_threads = std::thread::hardware_concurrency();
    if (!pool.try_spawn(count_threads)) return false;
    std::vector<std::size_t> visited(expected_parts, 0);
    std::atomic<std::size_t> counter = 0;
    pool.for_each_dynamic(expected_parts, [&](std::size_t const task_index) noexcept {
        // ? Relax the memory order, as we don't care about the order of the results, will sort 'em later
        std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
        visited[count_populated] = task_index;
    });

    // Make sure that all task IDs are unique and form the full range of [0, `expected_parts`).
    std::sort(visited.begin(), visited.end());
    return counter.load() == expected_parts && std::adjacent_find(visited.begin(), visited.end()) == visited.end() &&
           std::is_sorted(visited.begin(), visited.end()) && visited.front() == (0) &&
           visited.back() == (expected_parts - 1);
}

/** @brief Stress-tests the implementation by oversubscribing the number of threads. */
static bool test_oversubscribed_unbalanced_threads() noexcept {
    constexpr std::size_t expected_parts = 10'000'000;
    constexpr std::size_t oversubscription = 7;

    av::fork_union_t pool;
    auto const count_threads = std::thread::hardware_concurrency() * oversubscription;
    if (!pool.try_spawn(count_threads)) return false;
    std::vector<std::size_t> visited(expected_parts, 0);
    std::atomic<std::size_t> counter = 0;
    thread_local volatile std::size_t some_local_work = 0;
    pool.for_each_dynamic(expected_parts, [&](std::size_t const task_index) noexcept {
        // Perform some weird amount of work, that is not very different between consecutive tasks.
        for (std::size_t i = 0; i != task_index % oversubscription; ++i) some_local_work += i * i;

        // ? Relax the memory order, as we don't care about the order of the results, will sort 'em later
        std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
        visited[count_populated] = task_index;
    });

    // Make sure that all task IDs are unique and form the full range of [0, `expected_parts`).
    std::sort(visited.begin(), visited.end());
    return counter.load() == expected_parts && std::adjacent_find(visited.begin(), visited.end()) == visited.end() &&
           std::is_sorted(visited.begin(), visited.end()) && visited.front() == (0) &&
           visited.back() == (expected_parts - 1);
}

int main() {
    test_t const tests[] = {
        {"`try_spawn` Success", test_try_spawn_success},                                       //
        {"`try_spawn` Zero", test_try_spawn_zero},                                             //
        {"`for_each_thread` Dispatch", test_for_each_thread},                                  //
        {"`for_each_static` Static Scheduling", test_for_each_static},                         //
        {"`for_each_dynamic` Dynamic Scheduling", test_for_each_dynamic},                      //
        {"`for_each_dynamic` Oversubscribed Threads", test_oversubscribed_unbalanced_threads}, //
    };

    std::size_t const total = sizeof(tests) / sizeof(tests[0]);
    std::size_t failed = 0;
    for (std::size_t i = 0; i < total; ++i) {
        std::printf("Running %s... ", tests[i].name);
        bool const ok = tests[i].function();
        if (ok) { std::printf("PASS\n"); }
        else { std::printf("FAIL\n"); }
        failed += !ok;
    }

    if (failed > 0) {
        std::fprintf(stderr, "%zu/%zu tests failed\n", failed, total);
        return EXIT_FAILURE;
    }

    std::printf("All %zu tests passed\n", total);
    return EXIT_SUCCESS;
}
