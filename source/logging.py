import logging
from time import time


class Heartbeat:
    interval_seconds: float
    last_heartbeat: float
    last_step: float | None
    ema_step_time: float | None
    ema: float

    def __init__(self, interval_seconds: float, ema: float = 0.99) -> None:
        self.interval_seconds = interval_seconds
        self.last_heartbeat = time()
        self.last_step = None
        self.ema_step_time = None
        self.ema = ema

    def step(self, step: int) -> None:
        # Print a heartbeat if enough time has passed.
        current = time()
        if current - self.last_heartbeat >= self.interval_seconds:
            self.last_heartbeat = current
            rate = ""
            if self.ema_step_time is not None:
                rate = f" (EMA rate: {1 / self.ema_step_time:.2f} it/s)"
            logging.info(f"Entering step {step}{rate}")

        # Update the EMA rate.
        if self.last_step is not None:
            step_time = current - self.last_step
            if self.ema_step_time is None:
                self.ema_step_time = step_time
            else:
                self.ema_step_time = (self.ema_step_time * self.ema) + (
                    step_time * (1 - self.ema)
                )

        # Don't record the first step.
        if step > 0:
            self.last_step = current
