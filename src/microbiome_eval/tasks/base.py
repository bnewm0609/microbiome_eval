PROJ_PATH = Path(__file__).parent.parent.parent

class BaseTask:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def add_arguments(parser):
        pass

    def get_prompts(self):
        raise NotImplementedError

    def evaluate_results(self, results):
        raise NotImplementedError
