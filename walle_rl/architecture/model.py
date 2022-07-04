from typing import Any, Callable, Optional, Tuple, TypeVar, Union

from chex import Array, ArrayTree, PRNGKey
from flax import struct
import flax.linen as nn
import jax
import optax
import jax.numpy as jnp

T = TypeVar("T", bound=nn.Module)

@struct.dataclass
class Model(struct.PyTreeNode):
    """
    model class that holds the model parameters and training state. Provides wrapped functions to execute forward passes in OOP style. 
    
    calling Model.create returns the original nn.Module but additional functions

    expects all parameters to optimized by a single optimizer
    """
    model: T = struct.field(pytree_node=False)
    params: ArrayTree
    apply_fn: Callable = struct.field(pytree_node=False)
    optimizer: Optional[optax.GradientTransformation] = struct.field(pytree_node=False)
    opt_state: Optional[optax.OptState] = None
    step: int = 0
    @classmethod
    def create(cls,
        model: T,
        key: PRNGKey,
        sample_input: Any,
        optimizer: Optional[optax.GradientTransformation] = None
    ) -> Union[T, 'Model']:
        model_vars = model.init(key, sample_input)
        opt_state = None
        if optimizer is not None:
            opt_state = optimizer.init(model_vars)
        return cls(model=model, params=model_vars, opt_state=opt_state, apply_fn=model.apply, optimizer=optimizer)
    def __call__(self, *args, **kwargs):
        return self.apply_fn(self.params, *args, **kwargs)

    def apply_gradient(self, grads):
        updates, updated_opt_state = self.optimizer.update(grads, self.opt_state, self.params)
        updated_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=updated_params,
            opt_state=updated_opt_state
        )
    def __getattribute__(self, name: str) -> Any:
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            # if attribute is another module, can we scope it?
            attr = self.model.__getattribute__(name)
            return attr

