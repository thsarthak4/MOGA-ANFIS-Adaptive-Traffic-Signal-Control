from simulator import *
from controllers import *

for s in (0.5, 0.8, 1.0, 1.2):
    print(s, simulate(FixedTime(), Scenario("t", s), 1))