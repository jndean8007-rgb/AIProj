from collections import OrderedDict
from itertools import islice

from torch_llm.inference.request_state import RequestState


class ContinuousBatchScheduler:
    """FIFO admission queue. Call from the runtime's owning thread."""

    def __init__(self, max_requests: int):
        if not isinstance(max_requests, int) or isinstance(max_requests, bool) or max_requests <= 0:
            raise ValueError("max_requests must be a positive integer")
        self.max_requests = max_requests
        self.waiting_requests: OrderedDict[int, RequestState] = OrderedDict()
        self.active_requests: OrderedDict[int, RequestState] = OrderedDict()

    def submit(self, request: RequestState):
        if request.request_id in self.waiting_requests or request.request_id in self.active_requests:
            raise ValueError(f"Duplicate request ID: {request.request_id}")
        if request.finished or request.generated_tokens:
            raise ValueError("Only new requests can be submitted")
        self.waiting_requests[request.request_id] = request

    def admission_candidates(self, limit: int | None = None) -> list[tuple[int, RequestState]]:
        space = self.max_requests - len(self.active_requests)
        if limit is not None:
            if limit < 0:
                raise ValueError("limit must be nonnegative")
            space = min(space, limit)
        return list(islice(self.waiting_requests.items(), space))

    def get_active_requests(self):
        return self.active_requests.copy()

    def admit_request(self, request_id: int) -> RequestState:
        if len(self.active_requests) >= self.max_requests:
            raise RuntimeError("The scheduler has no free request slots")
        request = self.waiting_requests.pop(request_id)
        self.active_requests[request_id] = request
        return request

    def remove(self, request_id: int) -> RequestState:
        request = self.active_requests.pop(request_id)
        request.finished = True
        return request

    def has_pending_requests(self) -> bool:
        return bool(self.waiting_requests or self.active_requests)
