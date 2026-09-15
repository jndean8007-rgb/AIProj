
class OutputHandler:
    def __init__(self, ): #something wil be ehre later
        pass # later will be more complicated and require configs, metadata, etc

    def handle_output(self, batch, active_stream_indices):
        # simple initially
        for new_token, active_stream in zip(batch.token_ids, active_stream_indices):
            print(f"Stream {active_stream} -> {new_token}")

