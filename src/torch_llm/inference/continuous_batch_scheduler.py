from collections import OrderedDict

class ContinuousBatchScheduler:
    def __init__(
            self,
            max_requests,
    ):
        self.waiting_requests = OrderedDict()
        self.active_requests = OrderedDict()

        self.max_requests = max_requests

    def submit(self, request):
        self.waiting_requests[request.request_id] = request

    def admission_candidates(self):
        space = self.max_requests - len(self.active_requests)
        return [(request_id, request) for request_id, request in list(self.waiting_requests.items())[:space]]

    def get_active_requests(self):
        return self.active_requests

    def admit_request(self, request_id, cache_slot):
        assert request_id not in self.active_requests
        self.active_requests[request_id] = self.waiting_requests.pop(request_id)

        self.active_requests[request_id].cache_slot = cache_slot

    def remove(self, request_id):
        self.active_requests.pop(request_id)

    '''def admit_waiting(self):
        for request, request_id in self.waiting_requests:
            if len(self.active_requests) < self.max_requests:
                self.active_requests[request_id] = request
                self.waiting_requests.pop(request_id)
            else:
                return'''
### request state vs request ids