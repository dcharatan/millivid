import logging
import signal
from contextlib import contextmanager


class PreemptionException(Exception):
    pass


class PreemptionManager:
    def __init__(self) -> None:
        self.preempted = False
        self.should_delay_preemption = False
        signal.signal(signal.SIGINT, self._handle_preemption)
        signal.signal(signal.SIGTERM, self._handle_preemption)

    def _handle_preemption(self, sig, frame):
        logging.info("Preemption signal received.")
        self.preempted = True
        if self.should_delay_preemption:
            logging.info("Delaying PreemptionException (inside atomic block).")
        else:
            logging.info("Raising PreemptionException immediately.")
            raise PreemptionException()

    @contextmanager
    def atomic(self):
        """Delay preemption on this process until the end of the block. Note that this
        does not synchronize distributed processes.
        """

        self.should_delay_preemption = True
        try:
            yield
        finally:
            self.should_delay_preemption = False
            if self.preempted:
                logging.info("Raising PreemptionException after atomic block.")
                raise PreemptionException()
