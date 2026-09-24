from collections import OrderedDict

class ContinuousBatchScheduler:
    def __init__(
            self,
            max_requests,
    ):
        self.waiting_requests = OrderedDict()
        self.active_requests = OrderedDict()

        self.max_requests = max_requests

    def submit(self, request, request_id):
        self.waiting_requests[request_id] = request

    def get_active_requests(self):
        return self.active_requests

    def remove_finished(self, request_id):
        self.active_requests.pop(request_id)

    def admit_waiting(self):
        for request, request_id in self.waiting_requests:
            if len(self.active_requests) < self.max_requests:
                self.active_requests[request_id] = request
                self.waiting_requests.pop(request_id)
            else:
                return
### request state vs request ids