import equinox as eqx
import jax
from equinox._module import BoundMethod
from jax.tree_util import GetAttrKey


def find_param_by_name(module: eqx.Module, name: str) -> list | None:
    """Find a leaf parameter in an Equinox module by name.

    Args:
        module: The Equinox module to search in.
        name: The name of the parameter to find.

    Returns:
        The parameter if found, None otherwise.
    """
    leaves_with_paths = jax.tree_util.tree_leaves_with_path(module)

    leaves = []
    for path, leaf in leaves_with_paths:
        if path and not isinstance(leaf, BoundMethod):
            last_key = path[-1]
            if isinstance(last_key, GetAttrKey) and last_key.name == name:
                leaves.append(leaf)

    return leaves if len(leaves) > 0 else None
