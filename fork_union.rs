#![feature(allocator_api)]
use std::alloc::{Allocator, Global};
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

/// Describes a portion of work executed on a specific thread.
#[derive(Copy, Clone)]
pub struct Task {
    pub thread_index: usize,
    pub task_index: usize,
}

type Trampoline = unsafe fn(*const (), Task);

#[repr(align(64))]
struct Inner {
    stop: AtomicBool,
    task_context: AtomicPtr<()>,
    task_trampoline: AtomicUsize,
    task_parts_count: AtomicUsize,
    task_parts_remaining: AtomicUsize,
    task_parts_passed: AtomicUsize,
    task_generation: AtomicUsize,
    total_threads: usize,
}

impl Inner {
    fn new(total_threads: usize) -> Self {
        Self {
            stop: AtomicBool::new(false),
            task_context: AtomicPtr::new(ptr::null_mut()),
            task_trampoline: AtomicUsize::new(0),
            task_parts_count: AtomicUsize::new(0),
            task_parts_remaining: AtomicUsize::new(0),
            task_parts_passed: AtomicUsize::new(0),
            task_generation: AtomicUsize::new(0),
            total_threads,
        }
    }

    fn reset_task(&self) {
        self.task_parts_count.store(0, Ordering::Relaxed);
        self.task_context.store(ptr::null_mut(), Ordering::Relaxed);
        self.task_trampoline.store(0, Ordering::Relaxed);
    }

    fn trampoline(&self) -> Trampoline {
        unsafe { std::mem::transmute(self.task_trampoline.load(Ordering::Acquire)) }
    }

    fn context(&self) -> *const () {
        self.task_context.load(Ordering::Acquire)
    }
}

/// Minimalistic non-resizable thread-pool.
pub struct ForkUnion<A: Allocator + Clone = Global> {
    inner: Arc<Inner>,
    workers: Vec<JoinHandle<()>, A>,
}

impl<A: Allocator + Clone> ForkUnion<A> {
    /// Creates the pool with the desired number of threads using a custom allocator.
    pub fn try_spawn_in(planned_threads: usize, alloc: A) -> Option<Self> {
        if planned_threads == 0 {
            return None;
        }
        if planned_threads == 1 {
            let inner = Arc::new(Inner::new(1));
            return Some(Self {
                inner,
                workers: Vec::new_in(alloc),
            });
        }
        let inner = Arc::new(Inner::new(planned_threads));
        let mut workers = Vec::try_with_capacity_in(planned_threads - 1, alloc.clone()).ok()?;
        for i in 0..planned_threads - 1 {
            let inner_cloned = Arc::clone(&inner);
            workers.push(thread::spawn(move || worker_loop(inner_cloned, i + 1)));
        }
        Some(Self { inner, workers })
    }

    /// Returns the number of threads in the pool.
    pub fn thread_count(&self) -> usize {
        self.inner.total_threads
    }

    /// Stops all threads and waits for them to finish.
    pub fn stop_and_reset(&mut self) {
        if self.inner.total_threads <= 1 {
            return;
        }
        assert!(self.inner.task_parts_remaining.load(Ordering::SeqCst) == 0);
        self.inner.reset_task();
        self.inner.stop.store(true, Ordering::Release);
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
        self.inner.stop.store(false, Ordering::Relaxed);
    }

    /// Executes a function on each thread of the pool.
    pub fn for_each_thread<F>(&self, function: F)
    where
        F: Fn(usize) + Sync,
    {
        if self.inner.total_threads == 1 {
            function(0);
            return;
        }
        let ctx = &function as *const F as *const ();
        self.inner
            .task_context
            .store(ctx as *mut (), Ordering::Relaxed);
        self.inner
            .task_trampoline
            .store(call_thread::<F> as usize, Ordering::Relaxed);
        self.inner
            .task_parts_count
            .store(self.inner.total_threads, Ordering::Relaxed);
        self.inner
            .task_parts_remaining
            .store(self.inner.total_threads - 1, Ordering::Relaxed);
        self.inner.task_generation.fetch_add(1, Ordering::Release);
        function(0);
        while self.inner.task_parts_remaining.load(Ordering::Acquire) != 0 {
            thread::yield_now();
        }
        self.inner.reset_task();
    }

    /// Executes evenly sized tasks on each thread.
    pub fn for_each_slice<F>(&self, n: usize, function: F)
    where
        F: Fn(usize, usize) + Sync,
    {
        assert!(self.inner.total_threads != 0, "Thread pool not initialized");
        if self.inner.total_threads == 1 || n == 1 {
            function(0, n);
            return;
        }
        if n == 0 {
            return;
        }
        let per_thread = (n + self.inner.total_threads - 1) / self.inner.total_threads;
        self.for_each_thread(|thread_index| {
            let begin = std::cmp::min(thread_index * per_thread, n);
            let count = std::cmp::min(begin + per_thread, n) - begin;
            function(begin, count);
        });
    }

    /// Executes `n` balanced tasks in parallel.
    pub fn for_each_static<F>(&self, n: usize, function: F)
    where
        F: Fn(usize) + Sync,
    {
        self.for_each_slice(n, |start, count| {
            for i in 0..count {
                function(start + i);
            }
        });
    }

    /// Executes `n` uneven tasks, greedily stealing work.
    pub fn for_each_dynamic<F>(&self, n: usize, function: F)
    where
        F: Fn(usize) + Sync,
    {
        if self.inner.total_threads == 1 {
            for i in 0..n {
                function(i);
            }
            return;
        }
        if n == 0 {
            return;
        }
        if n == 1 {
            function(0);
            return;
        }
        let ctx = &function as *const F as *const ();
        self.inner
            .task_context
            .store(ctx as *mut (), Ordering::Relaxed);
        self.inner
            .task_trampoline
            .store(call_index::<F> as usize, Ordering::Relaxed);
        self.inner.task_parts_count.store(n, Ordering::Relaxed);
        self.inner.task_parts_passed.store(0, Ordering::Relaxed);
        self.inner.task_parts_remaining.store(n, Ordering::Relaxed);
        self.inner.task_generation.fetch_add(1, Ordering::Release);
        dynamic_loop(&self.inner, 0);
        while self.inner.task_parts_remaining.load(Ordering::Acquire) != 0 {
            thread::yield_now();
        }
        self.inner.reset_task();
    }
}

impl<A: Allocator + Clone> Drop for ForkUnion<A> {
    fn drop(&mut self) {
        self.stop_and_reset();
    }
}

impl ForkUnion<Global> {
    /// Creates the pool with the desired number of threads using the global allocator.
    pub fn try_spawn(planned_threads: usize) -> Option<Self> {
        Self::try_spawn_in(planned_threads, Global)
    }
}

unsafe fn call_thread<F: Fn(usize)>(ctx: *const (), task: Task) {
    let f = &*(ctx as *const F);
    f(task.thread_index);
}

unsafe fn call_index<F: Fn(usize)>(ctx: *const (), task: Task) {
    let f = &*(ctx as *const F);
    f(task.task_index);
}

fn worker_loop(inner: Arc<Inner>, thread_index: usize) {
    let mut last_generation = 0usize;
    loop {
        let mut new_generation;
        while {
            new_generation = inner.task_generation.load(Ordering::Acquire);
            new_generation == last_generation && !inner.stop.load(Ordering::Acquire)
        } {
            thread::yield_now();
        }
        if inner.stop.load(Ordering::Acquire) {
            return;
        }
        let is_static = inner.task_parts_count.load(Ordering::Acquire) == inner.total_threads;
        if is_static && inner.task_parts_count.load(Ordering::Acquire) > 0 {
            let trampoline = inner.trampoline();
            let context = inner.context();
            unsafe {
                trampoline(
                    context,
                    Task {
                        thread_index,
                        task_index: thread_index,
                    },
                );
            }
            inner.task_parts_remaining.fetch_sub(1, Ordering::AcqRel);
        } else {
            dynamic_loop(&inner, thread_index);
        }
        last_generation = new_generation;
    }
}

fn dynamic_loop(inner: &Arc<Inner>, thread_index: usize) {
    let trampoline = inner.trampoline();
    let context = inner.context();
    loop {
        let idx = inner.task_parts_passed.fetch_add(1, Ordering::Relaxed);
        if idx >= inner.task_parts_count.load(Ordering::Acquire) {
            break;
        }
        unsafe {
            trampoline(
                context,
                Task {
                    thread_index,
                    task_index: idx,
                },
            );
        }
        inner.task_parts_remaining.fetch_sub(1, Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::Global;
    use std::sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    };

    #[inline]
    fn hw_threads() -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    }

    #[test]
    fn spawn_with_global() {
        let pool = ForkUnion::try_spawn(2).expect("spawn");
        assert_eq!(pool.thread_count(), 2);
    }

    #[test]
    fn spawn_with_allocator() {
        let pool: ForkUnion<Global> = ForkUnion::try_spawn_in(2, Global).expect("spawn");
        assert_eq!(pool.thread_count(), 2);
    }

    #[test]
    fn try_spawn_success() {
        let pool = ForkUnion::try_spawn(hw_threads());
        assert!(pool.is_some());
    }

    #[test]
    fn try_spawn_zero_threads() {
        assert!(ForkUnion::try_spawn(0).is_none());
    }

    #[test]
    fn for_each_thread_dispatch() {
        let count_threads = hw_threads();
        let pool = ForkUnion::try_spawn(count_threads).expect("spawn");

        let visited = Arc::new(
            (0..count_threads)
                .map(|_| AtomicBool::new(false))
                .collect::<Vec<_>>(),
        );
        let visited_ref = Arc::clone(&visited);

        pool.for_each_thread(move |thread_index| {
            visited_ref[thread_index].store(true, Ordering::Relaxed);
        });

        for flag in visited.iter() {
            assert!(
                flag.load(Ordering::Relaxed),
                "thread never reached the callback"
            );
        }
    }

    #[test]
    fn for_each_static_uncomfortable_input_size() {
        let count_threads = hw_threads();
        let pool = ForkUnion::try_spawn(count_threads).expect("spawn");

        for input_size in 0..count_threads {
            let out_of_bounds = AtomicBool::new(false);
            pool.for_each_static(input_size, |task_index| {
                if task_index >= count_threads {
                    out_of_bounds.store(true, Ordering::Relaxed);
                }
            });
            assert!(
                !out_of_bounds.load(Ordering::Relaxed),
                "task_index exceeded thread_count at n = {input_size}"
            );
        }
    }

    #[test]
    fn for_each_static_static_scheduling() {
        const EXPECTED_PARTS: usize = 10_000_000;
        let pool = ForkUnion::try_spawn(hw_threads()).expect("spawn");

        let visited = Arc::new(
            (0..EXPECTED_PARTS)
                .map(|_| AtomicBool::new(false))
                .collect::<Vec<_>>(),
        );
        let duplicate = Arc::new(AtomicBool::new(false));
        let visited_ref = Arc::clone(&visited);
        let duplicate_ref = Arc::clone(&duplicate);

        pool.for_each_static(EXPECTED_PARTS, move |task_index| {
            if visited_ref[task_index].swap(true, Ordering::Relaxed) {
                duplicate_ref.store(true, Ordering::Relaxed);
            }
        });

        assert!(
            !duplicate.load(Ordering::Relaxed),
            "static scheduling produced duplicate task IDs"
        );
        for flag in visited.iter() {
            assert!(flag.load(Ordering::Relaxed));
        }
    }

    #[test]
    fn for_each_dynamic_dynamic_scheduling() {
        const EXPECTED_PARTS: usize = 10_000_000;
        let pool = ForkUnion::try_spawn(hw_threads()).expect("spawn");

        let visited = Arc::new(
            (0..EXPECTED_PARTS)
                .map(|_| AtomicBool::new(false))
                .collect::<Vec<_>>(),
        );
        let duplicate = Arc::new(AtomicBool::new(false));
        let visited_ref = Arc::clone(&visited);
        let duplicate_ref = Arc::clone(&duplicate);

        pool.for_each_dynamic(EXPECTED_PARTS, move |task_index| {
            if visited_ref[task_index].swap(true, Ordering::Relaxed) {
                duplicate_ref.store(true, Ordering::Relaxed);
            }
        });

        assert!(
            !duplicate.load(Ordering::Relaxed),
            "dynamic scheduling produced duplicate task IDs"
        );
        for flag in visited.iter() {
            assert!(flag.load(Ordering::Relaxed));
        }
    }

    #[test]
    fn oversubscribed_unbalanced_threads() {
        const EXPECTED_PARTS: usize = 10_000_000;
        const OVERSUBSCRIPTION: usize = 7;
        let threads = hw_threads() * OVERSUBSCRIPTION;
        let pool = ForkUnion::try_spawn(threads).expect("spawn");

        let visited = Arc::new(
            (0..EXPECTED_PARTS)
                .map(|_| AtomicBool::new(false))
                .collect::<Vec<_>>(),
        );
        let duplicate = Arc::new(AtomicBool::new(false));
        let visited_ref = Arc::clone(&visited);
        let duplicate_ref = Arc::clone(&duplicate);

        thread_local! { static LOCAL_WORK: std::cell::Cell<usize> = std::cell::Cell::new(0); }

        pool.for_each_dynamic(EXPECTED_PARTS, move |task_index| {
            // Mildly unbalanced CPU burn
            LOCAL_WORK.with(|cell| {
                let mut acc = cell.get();
                for i in 0..task_index % OVERSUBSCRIPTION {
                    acc = acc.wrapping_add(i * i);
                }
                cell.set(acc);
            });

            if visited_ref[task_index].swap(true, Ordering::Relaxed) {
                duplicate_ref.store(true, Ordering::Relaxed);
            }
        });

        assert!(
            !duplicate.load(Ordering::Relaxed),
            "oversubscribed run produced duplicate task IDs"
        );
        for flag in visited.iter() {
            assert!(flag.load(Ordering::Relaxed));
        }
    }

    #[test]
    fn c_function_pointer_compatibility() {
        static TASK_COUNTER: AtomicUsize = AtomicUsize::new(0);
        const EXPECTED_PARTS: usize = 10_000_000;

        fn tally(_: usize) {
            TASK_COUNTER.fetch_add(1, Ordering::Relaxed);
        }

        TASK_COUNTER.store(0, Ordering::Relaxed);
        let pool = ForkUnion::try_spawn(hw_threads()).expect("spawn");
        pool.for_each_dynamic(EXPECTED_PARTS, tally as fn(usize));

        assert_eq!(
            TASK_COUNTER.load(Ordering::Relaxed),
            EXPECTED_PARTS,
            "function-pointer callback executed the wrong number of times"
        );
    }
}
