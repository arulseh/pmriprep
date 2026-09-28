"""Build the platform-native CPI numerical kernel."""

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class ConservativeBuildExt(build_ext):
    """Compile with standard floating-point arithmetic on supported compilers."""

    def build_extensions(self) -> None:
        """Disable fast-math and multiply-add contraction for numerical parity."""
        flags = (["/O2", "/fp:strict"] if self.compiler.compiler_type == "msvc"
                 else ["-O3", "-fno-fast-math", "-ffp-contract=off"])
        for extension in self.extensions:
            extension.extra_compile_args = flags
        super().build_extensions()


setup(
    ext_modules=[Extension(
        "pmriprep.models.dsc.signal_cpi._convolution",
        ["pmriprep/models/dsc/signal_cpi/_convolution.c"],
    )],
    cmdclass={"build_ext": ConservativeBuildExt},
)
