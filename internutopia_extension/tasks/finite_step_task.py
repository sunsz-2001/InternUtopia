from internutopia.core.scene.scene import IScene
from internutopia.core.task import BaseTask
from internutopia_extension.configs.tasks.finite_step_task import FiniteStepTaskCfg


@BaseTask.register('FiniteStepTask')
class FiniteStepTask(BaseTask):
    """
    A task that can be terminated either manually or automatically after a specified number of steps.

    By default, auto-stop is DISABLED - the task will run indefinitely until manually stopped.
    This allows for interactive exploration and debugging.

    Termination modes:
    1. Manual: Call request_stop() to stop at any time
    2. Auto (max_steps): Enable with enable_auto_stop(), stops after max_steps
    3. Custom: Subclass and override _check_custom_termination_conditions()
    """

    def __init__(self, config: FiniteStepTaskCfg, scene: IScene):
        super().__init__(config, scene)

        # Step counter
        self._step_count = 0
        self._max_steps = config.max_steps

        # Termination control flags
        self._auto_stop_enabled = False  # Default: disabled, run indefinitely
        self._manual_stop_requested = False

    # ==================== Termination Control API ====================

    def enable_auto_stop(self) -> None:
        """
        Enable automatic termination after max_steps.
        Call this if you want the task to automatically stop after a certain number of steps.
        """
        self._auto_stop_enabled = True
        print(f"[{self.name}] Auto-stop enabled (will stop after {self._max_steps} steps)")

    def disable_auto_stop(self) -> None:
        """
        Disable automatic termination.
        The task will run indefinitely until manually stopped.
        """
        self._auto_stop_enabled = False
        print(f"[{self.name}] Auto-stop disabled (task will run indefinitely)")

    def is_auto_stop_enabled(self) -> bool:
        """Check if auto-stop is currently enabled."""
        return self._auto_stop_enabled

    def request_stop(self) -> None:
        """
        Manually request the task to stop.
        The task will be marked as done on the next is_done() check.
        """
        self._manual_stop_requested = True
        print(f"[{self.name}] Manual stop requested (current step: {self._step_count}/{self._max_steps})")

    def is_stop_requested(self) -> bool:
        """Check if manual stop has been requested."""
        return self._manual_stop_requested

    # ==================== Step Counter API ====================

    def get_current_step(self) -> int:
        """Get the current step count."""
        return self._step_count

    def get_max_steps(self) -> int:
        """Get the maximum number of steps (if auto-stop is enabled)."""
        return self._max_steps

    def set_max_steps(self, max_steps: int) -> None:
        """
        Update the maximum number of steps.
        Useful for dynamically adjusting termination criteria.
        """
        self._max_steps = max_steps
        print(f"[{self.name}] Max steps updated to {max_steps}")

    def reset_step_counter(self) -> None:
        """Reset the step counter to 0."""
        self._step_count = 0
        print(f"[{self.name}] Step counter reset")

    # ==================== Termination Logic ====================

    def is_done(self) -> bool:
        """
        Check if the task should terminate.

        Termination conditions (checked in order):
        1. Manual stop requested via request_stop()
        2. Auto-stop enabled AND step count exceeds max_steps
        3. Custom termination conditions (override _check_custom_termination_conditions())

        Returns:
            bool: True if the task should stop, False otherwise
        """
        self._step_count += 1

        # Priority 1: Manual stop
        if self._manual_stop_requested:
            return True

        # Priority 2: Auto-stop (if enabled)
        if self._auto_stop_enabled and self._step_count > self._max_steps:
            print(f"[{self.name}] Auto-stop triggered: {self._step_count} > {self._max_steps}")
            return True

        # Priority 3: Custom conditions (for subclasses)
        if self._check_custom_termination_conditions():
            return True

        return False

    def _check_custom_termination_conditions(self) -> bool:
        """
        Hook for subclasses to define custom termination conditions.

        Example use cases:
        - Stop when robot reaches a goal position
        - Stop when a metric threshold is reached
        - Stop on collision detection

        Returns:
            bool: True if custom termination condition is met, False otherwise
        """
        return False
