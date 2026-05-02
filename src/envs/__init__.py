from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
import sys
import os
from os.path import dirname, abspath

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)

if sys.platform == "linux":
    project_root = dirname(dirname(dirname(abspath(__file__))))
    os.environ.setdefault("SC2PATH",
                          os.path.join(project_root, "3rdparty", "StarCraftII"))
