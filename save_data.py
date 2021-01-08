from singleton_classes import ProbDescription
from error_func_RK2_with_post_projection import error_RK2_with_post_projection
import json

# taylor vortex
# ---------------
probDescription = ProbDescription(N=[32,32],L=[1,1],μ =1e-3,dt = 0.005)

levels = 5        # total number of refinements

rx = 1
ry = 1
rt = 2

dts = [probDescription.get_dt()/rt**i for i in range(0,levels)]
n=5
tend = dts[0]*n
timesteps = [n*rt**i for i in range(0,levels) ]


metadata={  "simulationName":"taylor_vortex",
            "endTime": tend,
            "refinementsNumber": levels,
            "refinementFactor":rt,
            "timesteps":dts,
            "domainSize":[1,1],
            "resolution":[32,32]
          }

save_to = "./taylor_vortex_pressure_data"

# save the metadata to a json file:
with open(save_to+"/metadata.json",'w') as file:
    json.dump(metadata,file)

count = 0
for dt, nsteps in zip(dts, timesteps):
    probDescription.set_dt(dt)

    # taylor vortex
    # ---------------
    e, divs, _, phi = error_RK2_with_post_projection(steps=nsteps, name='heun', guess=None, project=[1],post_projection=True,save_to=save_to,refNum=count)
    count+=1