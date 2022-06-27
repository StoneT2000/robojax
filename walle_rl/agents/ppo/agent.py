from walle_rl.architecture.ac.core import ActorCritic
class PPO:

    def __init__(
        self,
        ac: ActorCritic,
        env: gym.Env,
        num_envs: int,
        observation_space,
        action_space,
        steps_per_epoch: int = 10000,
        train_iters: int = 80,
        gamma: float = 0.99,
        gae_lambda: float = 0.97,
        clip_ratio: float = 0.2,
        ent_coef: float = 0.0,
        pi_coef: float = 1.0,
        vf_coef: float = 1.0,
        dapg_lambda: float = 0.1,
        dapg_damping: float = 0.99,
        # max_grad_norm: float = 0.5, # TODO
        # use_sde: bool = False,
        # sde_sample_freq: int = -1,
        target_kl: Optional[float] = 0.01,        # create_eval_env: bool = False,
        # verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "cpu",

    ) -> None: