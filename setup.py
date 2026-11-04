"""Build the platform-native perfusion numerical kernels."""

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class ConservativeBuildExt(build_ext):
    """Compile with standard floating-point arithmetic on supported compilers."""

    def build_extensions(self) -> None:
        """Disable fast-math and multiply-add contraction for numerical parity."""
        flags = (["/O2", "/fp:strict"] if self.compiler.compiler_type == "msvc"
                 else ["-O3", "-fno-fast-math", "-ffp-contract=off"])
        for extension in self.extensions:
            extension.extra_compile_args = flags + (
                ["-std=c11"] if extension.name.endswith("_native_trf")
                and self.compiler.compiler_type != "msvc" else []
            )
        super().build_extensions()


setup(
    ext_modules=[Extension(
        "pmriprep.models.dsc.signal_cpi._convolution",
        ["pmriprep/models/dsc/signal_cpi/_convolution.c"],
    ), Extension(
        "pmriprep.models.dce._native_trf",
        ["pmriprep/models/dce/_trf_module.c"],
        depends=[f"pmriprep/models/dce/_trf_{part}.c"
                 for part in ("solver", "evaluator", "trust", "recurrence")],
    )],
    cmdclass={"build_ext": ConservativeBuildExt},
)
