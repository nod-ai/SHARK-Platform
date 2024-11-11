from dataclasses import dataclass


@dataclass
class SGLangBenchmarkArgs:
    num_prompt: int
    base_url: str
    tokenizer: str
    request_rate: int
    backend: str = "shortfin"

    def __repr__(self):
        return (
            f"Backend: {self.backend}\n"
            f"Base URL: {self.base_url}\n"
            f"Num Prompt: {self.num_prompt}\n"
            f"Tokenizer: {self.tokenizer}\n"
            f"Request Rate: {self.request_rate}"
        )
