"""
A base policy class 
"""

class Policy:
    def __init__(self) -> None:
        pass
    def gradient(self):
        raise NotImplementedError()
