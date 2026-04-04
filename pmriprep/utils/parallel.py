"""Utilities for parallel worker process configuration."""


def limit_worker_threads() -> None:
    """Restrict BLAS/OpenMP to 1 thread in pool workers.

    Per-voxel Bayesian fitting (VI, MAP) uses ProcessPoolExecutor where
    each worker handles small 1D convolutions (~50-100 points). OMP
    threading within each worker adds no benefit and causes severe
    contention when all workers compete for cores.

    Uses threadpoolctl to modify the thread count at runtime via the
    BLAS library's API, which works even after fork (unlike os.environ
    which is only read at library load time).
    """
    from threadpoolctl import threadpool_limits

    threadpool_limits(limits=1, user_api="blas")
    threadpool_limits(limits=1, user_api="openmp")
