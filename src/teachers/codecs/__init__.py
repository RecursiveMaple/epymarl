from .coop_reaching_codec import CoopReachingCodec
from .lb_foraging_codec import LBForagingCodec
from .predator_prey_codec import PredatorPreyCodec

REGISTRY = {}

REGISTRY["CooperativeReaching-v0"] = CoopReachingCodec
REGISTRY["LevelBasedForaging-v3"] = LBForagingCodec
REGISTRY["PredatorPrey-v0"] = PredatorPreyCodec
