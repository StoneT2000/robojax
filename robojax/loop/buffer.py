class Buffer:
    def __init__(self, buffer_size: int, n_envs: int) -> None:
        pass


class JaxBuffer(Buffer):
    def __init__(self, buffer_size: int, n_envs: int) -> None:
        super().__init__(buffer_size, n_envs)
