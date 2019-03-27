from models.vvnet import *
from models.sscnet import *
from models.vvnetae import *
from models.vvnetrp import *
from models.network import *

NETWORK = dict()
for local_var in list(globals().keys()):
    if not local_var.startswith('VVNet') and not local_var.startswith('SSCNet'):
        continue
    NETWORK[local_var] = globals()[local_var]
