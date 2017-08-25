# space-wrappers
General purpose environment wrappers for openai gym.

## List of currently implemented Wrappers
* FlattenedActionWrapper
* DiscretizedActionWrapper
* RescaledActionWrapper
* FlattenedObservationWrapper
* DiscretizedObservationWrapper
* RescaledObservationWrapper


## Usage Example
Suppose you want to train a (D)DQN agent for an environment
with continuous actions. Since DQN implementations typically
expect to produce a single discrete action, the action space
has to both be discretized and flattened, as demonstrated
in the code below.  
```python
import gym
import space_wrappers
# An environment with a continuous action space.
# We first turn it into a MultiDiscrete, and then into
# a flat discrete action space.
env = gym.make("LunarLanderContinuous-v2")
wrapped = space_wrappers.DiscretizedActionWrapper(env, 3)
wrapped = space_wrappers.FlattenedActionWrapper(wrapped)

# this is now a single integer
print(wrapped.action_space.sample())
```


## TODO
- [ ] Documentation
- [ ] Handle Tuple spaces
- [ ] More Sanity checks
- [ ] Image transformations (resample, resize, ...)

