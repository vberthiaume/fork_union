/**
 *  @brief Demo app: N-Body simulation with Fork Union and OpenMP.
 *  @author Ash Vardanian
 *  @file nbody.cpp
 *
 *  To control the script, several environment variables are used:
 *
 *  - `NBODY_COUNT` - number of bodies in the simulation (default: number of threads).
 *  - `NBODY_ITERATIONS` - number of iterations to run the simulation (default: 1000).
 *  - `NBODY_BACKEND` - backend to use for the simulation (default: `fork_union_static`).
 *  - `NBODY_THREADS` - number of threads to use for the simulation (default: number of hardware threads).
 *
 *  The backends include: `fork_union_static`, `fork_union_dynamic`, `openmp_static`, and `openmp_dynamic`.
 *  To compile and run:
 *
 *  @code{.sh}
 *  cmake -B build_release -D CMAKE_BUILD_TYPE=Release
 *  cmake --build build_release --config Release
 *  NBODY_COUNT=128 NBODY_THREADS=$(nproc) build_release/scripts/fork_union_nbody
 *  @endcode
 *
 *  The default profiling scheme is to 1M iterations for 128 particles on each backend:
 *
 *  @code{.sh}
 *  time NBODY_COUNT=128 NBODY_THREADS=$(nproc) NBODY_ITERATIONS=1000000 \
 *      NBODY_BACKEND=openmp_static build_release/scripts/fork_union_nbody
 *  time NBODY_COUNT=128 NBODY_THREADS=$(nproc) NBODY_ITERATIONS=1000000 \
 *      NBODY_BACKEND=openmp_dynamic build_release/scripts/fork_union_nbody
 *  time NBODY_COUNT=128 NBODY_THREADS=$(nproc) NBODY_ITERATIONS=1000000 \
 *      NBODY_BACKEND=fork_union_static build_release/scripts/fork_union_nbody
 *  time NBODY_COUNT=128 NBODY_THREADS=$(nproc) NBODY_ITERATIONS=1000000 \
 *      NBODY_BACKEND=fork_union_dynamic build_release/scripts/fork_union_nbody
 *  @endcode
 */
#include <vector> // `std::vector`
#include <random> // `std::random_device`, `std::uniform_real_distribution`
#include <thread> // `std::thread::hardware_concurrency`
#include <span>   // `std::span`
#include <bit>    // `std::bit_cast`

#if 0 //defined(_OPENMP)
#include <omp.h>
#endif

#include <fork_union.hpp>
#include <string>

namespace fun = ashvardanian::fork_union;

#if defined(__GNUC__) || defined(__clang__)
#define _FU_RESTRICT __restrict__
#else
#define _FU_RESTRICT
#endif

#pragma region - Shared Logic

static constexpr float g_const = 6.674e-11;
static constexpr float dt_const = 0.01;
static constexpr float softening_const = 1e-9;

struct vector3_t {
    float x, y, z;

    inline vector3_t &operator+=(vector3_t const &other) noexcept {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }
};

struct body_t {
    vector3_t position;
    vector3_t velocity;
    float mass;
};

inline float fast_rsqrt(float x) noexcept {
    std::uint32_t i = std::bit_cast<std::uint32_t>(x);
    i = 0x5f3759df - (i >> 1);
    float y = std::bit_cast<float>(i);
    float x2 = x * 0.5f;
    y = y * (1.5f - x2 * y * y);
    return y;
}

inline vector3_t gravitational_force(body_t const &bi, body_t const &bj) noexcept {
    float dx = bj.position.x - bi.position.x;
    float dy = bj.position.y - bi.position.y;
    float dz = bj.position.z - bi.position.z;
    float l2_squared = dx * dx + dy * dy + dz * dz + softening_const;
    float l2_reciprocal = fast_rsqrt(l2_squared);
    float l2_cube_reciprocal = l2_reciprocal * l2_reciprocal * l2_reciprocal;
    float mag = g_const * bi.mass * bj.mass * l2_cube_reciprocal;
    return {mag * dx, mag * dy, mag * dz};
}

inline void apply_force(body_t &bi, vector3_t const &f) noexcept {
    bi.velocity.x += f.x / bi.mass * dt_const;
    bi.velocity.y += f.y / bi.mass * dt_const;
    bi.velocity.z += f.z / bi.mass * dt_const;
    bi.position.x += bi.velocity.x * dt_const;
    bi.position.y += bi.velocity.y * dt_const;
    bi.position.z += bi.velocity.z * dt_const;
}

#pragma endregion - Shared Logic

#pragma region - Backends

void iteration_openmp_static(body_t *_FU_RESTRICT bodies, vector3_t *_FU_RESTRICT forces, std::size_t n) noexcept {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        vector3_t f {0.0, 0.0, 0.0};
        for (std::size_t j = 0; j < n; ++j) f += gravitational_force(bodies[i], bodies[j]);
        forces[i] = f;
    }
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) apply_force(bodies[i], forces[i]);
#endif
}

void iteration_openmp_dynamic(body_t *_FU_RESTRICT bodies, vector3_t *_FU_RESTRICT forces, std::size_t n) noexcept {
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t i = 0; i < n; ++i) {
        vector3_t f {0.0, 0.0, 0.0};
        for (std::size_t j = 0; j < n; ++j) f += gravitational_force(bodies[i], bodies[j]);
        forces[i] = f;
    }
#pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t i = 0; i < n; ++i) apply_force(bodies[i], forces[i]);
#endif
}

void iteration_fork_union_static(fun::thread_pool_t &pool, body_t *_FU_RESTRICT bodies, vector3_t *_FU_RESTRICT forces,
                                 std::size_t n) noexcept {
    for_n(pool, n, [=](std::size_t i) noexcept {
        vector3_t f {0.0, 0.0, 0.0};
        for (std::size_t j = 0; j < n; ++j) f += gravitational_force(bodies[i], bodies[j]);
        forces[i] = f;
    });
    for_n(pool, n, [=](std::size_t i) noexcept { apply_force(bodies[i], forces[i]); });
}

void iteration_fork_union_dynamic(fun::thread_pool_t &pool, body_t *_FU_RESTRICT bodies, vector3_t *_FU_RESTRICT forces,
                                  std::size_t n) noexcept {
    for_n_dynamic(pool, n, [=](std::size_t i) noexcept {
        vector3_t f {0.0, 0.0, 0.0};
        for (std::size_t j = 0; j < n; ++j) f += gravitational_force(bodies[i], bodies[j]);
        forces[i] = f;
    });
    for_n_dynamic(pool, n, [=](std::size_t i) noexcept { apply_force(bodies[i], forces[i]); });
}

#pragma endregion - Backends

int main() {
    // Read env vars
    std::size_t n = std::stoul(std::getenv("NBODY_COUNT") ? std::getenv ("NBODY_COUNT") : "0");
    std::size_t const iterations = std::stoul(std::getenv("NBODY_ITERATIONS") ? std::getenv ("NBODY_ITERATIONS") : "1000");
    std::string_view const backend = std::getenv("NBODY_BACKEND") ? std::getenv("NBODY_BACKEND") : "fork_union_static";
    std::size_t threads = std::stoul(std::getenv("NBODY_THREADS") ? std::getenv ("NBODY_THREADS") : "0");
    if (threads == 0) threads = std::thread::hardware_concurrency();
    if (n == 0) n = threads;

    // Prepare bodies and forces - 2 memory allocations
    std::vector<body_t> bodies(n);
    std::vector<vector3_t> forces(n);

    // Random generators are quite slow, but let's hope this doesn't take too long
    std::uniform_real_distribution<float> coordinate_distribution(0.0, 1.0);
    std::uniform_real_distribution<float> mass_distribution(1e20, 1e25);
    std::random_device random_device;
    std::mt19937 random_gen(random_device());
    for (std::size_t i = 0; i < n; ++i) {
        bodies[i].position.x = coordinate_distribution(random_gen);
        bodies[i].position.y = coordinate_distribution(random_gen);
        bodies[i].position.z = coordinate_distribution(random_gen);
        bodies[i].velocity.x = coordinate_distribution(random_gen);
        bodies[i].velocity.y = coordinate_distribution(random_gen);
        bodies[i].velocity.z = coordinate_distribution(random_gen);
        bodies[i].mass = mass_distribution(random_gen);
    }

#if defined(_OPENMP)
    omp_set_num_threads(static_cast<int>(threads));
    if (backend == "openmp_static") {
        for (std::size_t i = 0; i < iterations; ++i) iteration_openmp_static(bodies.data(), forces.data(), n);
        return EXIT_SUCCESS;
    }
    if (backend == "openmp_dynamic") {
        for (std::size_t i = 0; i < iterations; ++i) iteration_openmp_dynamic(bodies.data(), forces.data(), n);
        return EXIT_SUCCESS;
    }
#endif

    // Every other configuration uses Fork Union
    fun::thread_pool_t pool;
    if (!pool.try_spawn(threads)) {
        std::fprintf(stderr, "Failed to spawn thread pool\n");
        return EXIT_FAILURE;
    }

    if (backend == "fork_union_static") {
        for (std::size_t i = 0; i < iterations; ++i) iteration_fork_union_static(pool, bodies.data(), forces.data(), n);
        return EXIT_SUCCESS;
    }
    if (backend == "fork_union_dynamic") {
        for (std::size_t i = 0; i < iterations; ++i)
            iteration_fork_union_dynamic(pool, bodies.data(), forces.data(), n);
        return EXIT_SUCCESS;
    }

    std::fprintf(stderr, "Unsupported backend: %s\n", backend.data());
    return EXIT_FAILURE;
}
