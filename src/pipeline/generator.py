from models.vlm import OpenRouterVLM


class InitialGenerator:
    def __init__(self):
        self.vlm = OpenRouterVLM()

    def generate(self, image_path, question):

        # TEMPORARY TEST RESPONSE
        # We are intentionally introducing unsupported assumptions
        # to test whether the downstream pipeline detects and corrects them.

        return """
The image shows a disabled patient working in a woodworking workshop.
He appears to be receiving medical treatment and needs assistance.
The other man is a trained professional helping him.
The patient seems uncomfortable and dependent on the other worker.
"""