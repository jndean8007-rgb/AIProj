
class Registry:
    def __init__(self, label):
        self.registry = {}
        self.label = label

    def register(self, name):
        if name in self.registry:
            raise ValueError("Name must not already be in registry")
        def decorate(cls):
            self.registry[name] = cls
            return cls
        return decorate

    def get(self, name):
        if name not in self.registry:
            raise ValueError("Name must be in registry")
        return self.registry.get(name)

    def names(self):
        return list(self.registry.keys())

MIXERS = Registry("MIXERS")
FFNS = Registry("FFNS")
RESIDUALS = Registry("RESIDUALS")
TOKEN_MEMORIES = Registry("TOKEN_MEMORIES")
OUTPUT_HEADS = Registry("OUTPUT_HEADS")
