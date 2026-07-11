from __future__ import annotations

import warnings
from typing import Any, Mapping, Sequence


class ModelFlopsEstimator:
    """Estimate per-sample model FLOPs from the first real forward pass."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.flops: int | None = None
        self._warning: str | None = None
        self._handles: list[Any] = []
        self._profiler: Any = None
        self._active = False
        self._batch_size = 1
        self._torch: Any = None
        self._install_hooks()

    def close(self) -> None:
        self._stop_active_profiler()
        self._remove_hooks()

    def pop_warning(self) -> str | None:
        warning = self._warning
        self._warning = None
        return warning

    def _install_hooks(self) -> None:
        if self.model is None:
            return
        register_pre_hook = getattr(self.model, "register_forward_pre_hook", None)
        register_hook = getattr(self.model, "register_forward_hook", None)
        if not callable(register_pre_hook) or not callable(register_hook):
            return
        try:
            self._torch = __import__("torch")
            if not hasattr(self._torch, "profiler"):
                self._set_warning(
                    "torch profiler is not available; model flops unavailable"
                )
                return
            self._handles.append(register_pre_hook(self._pre_forward_hook))
            try:
                self._handles.append(
                    register_hook(self._post_forward_hook, always_call=True)
                )
            except TypeError:
                self._handles.append(register_hook(self._post_forward_hook))
        except Exception as error:
            self._set_warning(f"model flops profiler setup failed: {error}")
            self._remove_hooks()

    def _pre_forward_hook(self, module: Any, args: tuple[Any, ...]) -> None:
        del module
        if self.flops is not None:
            return
        if self._active:
            self._stop_active_profiler()
        try:
            activities = [self._torch.profiler.ProfilerActivity.CPU]
            if _contains_cuda_tensor(args):
                activities.append(self._torch.profiler.ProfilerActivity.CUDA)
            self._batch_size = max(1, _infer_batch_size(args) or 1)
            profiler_kwargs = {
                "activities": activities,
                "with_flops": True,
                "record_shapes": False,
                "profile_memory": False,
                "acc_events": True,
            }
            try:
                self._profiler = self._torch.profiler.profile(**profiler_kwargs)
            except TypeError:
                profiler_kwargs.pop("acc_events")
                self._profiler = self._torch.profiler.profile(**profiler_kwargs)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*Profiler clears events.*",
                    category=UserWarning,
                )
                self._profiler.start()
            self._active = True
        except Exception as error:
            self._set_warning(f"model flops profiling failed to start: {error}")
            self._remove_hooks()

    def _post_forward_hook(
        self,
        module: Any,
        args: tuple[Any, ...],
        output: Any,
    ) -> None:
        del module, args, output
        if not self._active:
            return
        try:
            self._profiler.stop()
            total_flops = 0
            for event in self._profiler.key_averages():
                total_flops += int(getattr(event, "flops", 0) or 0)
            if total_flops > 0:
                self.flops = max(1, int(round(total_flops / self._batch_size)))
            else:
                self._set_warning(
                    "torch profiler did not report model flops; value unavailable"
                )
        except Exception as error:
            self._set_warning(f"model flops profiling failed to finish: {error}")
        finally:
            self._active = False
            self._profiler = None
            self._remove_hooks()

    def _stop_active_profiler(self) -> None:
        if not self._active or self._profiler is None:
            return
        try:
            self._profiler.stop()
        except Exception:
            pass
        finally:
            self._active = False
            self._profiler = None

    def _remove_hooks(self) -> None:
        while self._handles:
            handle = self._handles.pop()
            try:
                handle.remove()
            except Exception:
                pass

    def _set_warning(self, message: str) -> None:
        if self._warning is None:
            self._warning = message


def _infer_batch_size(args: tuple[Any, ...]) -> int | None:
    if not args:
        return None
    if len(args) == 1:
        return _infer_batch_container(args[0])
    return _infer_batch_container(args[0])


def _infer_batch_container(value: Any) -> int | None:
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            if len(shape) > 0:
                return int(shape[0])
        except Exception:
            return None
    if isinstance(value, Mapping):
        for key in ("inputs", "images", "image", "x", "tensor", "data"):
            if key in value:
                batch_size = _infer_batch_container(value[key])
                if batch_size is not None:
                    return batch_size
        for item in value.values():
            batch_size = _infer_batch_container(item)
            if batch_size is not None:
                return batch_size
        return None
    if isinstance(value, list):
        return len(value) or None
    if isinstance(value, tuple):
        if len(value) == 1:
            return _infer_batch_container(value[0])
        return _infer_batch_container(value[0]) or len(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return len(value) or None
    return None


def _contains_cuda_tensor(value: Any) -> bool:
    if bool(getattr(value, "is_cuda", False)):
        return True
    if isinstance(value, Mapping):
        return any(_contains_cuda_tensor(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_cuda_tensor(item) for item in value)
    return False
