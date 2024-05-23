# Systems

As shown in the [getting started](./getting_started.md) disucssion, attpc_engine really has two separate subsystems: the [kinematics](./kinematics/index.md) sampler and the [detector](./detector/index.md) effects. These two systems are (mostly) distinct and are two separate modules rolled into one package.

## Why are they separate

There are a couple of reasons why they're separated. First, they don't really need to know much about each other. The only real information leakage is the target material (what gas you're using and how much volume you allow the reaction vertex to travel through). So in that sense, they really shouldn't contact each other!

But more importantly, separating them allows you to save intermediate steps. Consider the following:

Say you are testing the detection effciency for a reaction where you already have some data. You generate the kinematics, and you need a lot of samples so it takes forever, hours or maybe even a day. Then you run it through the detector system and see that oh no you forgot to set the electric field correctly! If the systems were directly coupled, you would have to rerun everything, including the kinematics! With them separated, you only need to rerun the detector system, saving you time and effort. 