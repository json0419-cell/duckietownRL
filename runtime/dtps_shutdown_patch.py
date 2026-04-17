from __future__ import annotations


def apply_dtps_shutdown_patch() -> bool:
    """Make DTPS subscriber shutdown block until unsubscribe completes.

    The upstream implementation fires unsubscribe() asynchronously and returns
    immediately. When the env is torn down quickly between training segments,
    asyncio reports "Task was destroyed but it is pending!" for leftover DTPS
    subscription tasks. This patch keeps the behavior local to this project.
    """

    try:
        from duckietown.sdk.middleware.dtps.components import GenericDTPSSubscriber
    except Exception:
        return False

    if getattr(GenericDTPSSubscriber, "_duckietownrl_shutdown_patch", False):
        return True

    original_stop = GenericDTPSSubscriber._stop

    def _patched_stop(self):
        subscription = getattr(self, "_subscription", None)
        if subscription is not None:
            self._subscription = None
            try:
                self._connector.arun(subscription.unsubscribe(), block=True)
            except Exception:
                try:
                    self._connector.arun(subscription.unsubscribe(), block=False)
                except Exception:
                    pass
        return original_stop(self)

    GenericDTPSSubscriber._stop = _patched_stop
    GenericDTPSSubscriber._duckietownrl_shutdown_patch = True
    return True
