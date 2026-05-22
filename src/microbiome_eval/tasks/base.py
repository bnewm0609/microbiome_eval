from pathlib import Path
PROJ_PATH = Path(__file__).parent.parent.parent.parent

class BaseTask:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def add_arguments(parser):
        pass

    def get_prompts(self):
        raise NotImplementedError

    def evaluate_responses(self, results):
        raise NotImplementedError
