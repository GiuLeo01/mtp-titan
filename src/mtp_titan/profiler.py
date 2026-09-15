from dataclasses import dataclass

import torch

from torchtitan.tools.profiler import Profiler


class MtpProfiler(Profiler):

    @dataclass(kw_only=True, slots=True)
    class Config(Profiler.Config):
        pass

    def build_torch_profiler(self, **kwargs):
        original_profile = torch.profiler.profile

        def profile_with_memory(*args, **profile_kwargs):
            profile_kwargs["profile_memory"] = True
            return original_profile(*args, **profile_kwargs)

        torch.profiler.profile = profile_with_memory
        try:
            return super().build_torch_profiler(**kwargs)
        finally:
            torch.profiler.profile = original_profile
