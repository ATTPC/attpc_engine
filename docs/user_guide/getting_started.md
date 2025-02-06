# Getting Started

Once attpc_engine is installed it's time to setup our simulation project. Below is an outline of what we'd expect your project structure to look like

```txt
|---my_sim
|   |---generate_kinematics.py
|   |---apply_detector.py
|   |---target.json
|   |---.venv
|   |---output
|   |   |---kinematics
|   |   |---detector
```

You have a folder (in this case called `my_sim`) with two Python files (`generate_kinematics.py` and `apply_detector.py`) a virtual environment with attpc_engine installed (`.venv`) and an output directory with a folder for kinematics and detector effects. Note that you may want the output to be stored on a remote disk or other location as simulation files can be big at times (> 1 GB).

Now lets talk about generating and sampling a kinematics phase space!

## Sampling Kinematics

Below is an example script for running a kinematics sample, which in our example project we would write in `my_sim/generate_kinematics.py`

```python
from attpc_engine.kinematics import (
    KinematicsPipeline,
    KinematicsTargetMaterial,
    ExcitationGaussian,
    PolarUniform,
    run_kinematics_pipeline,
    Reaction,
)
from attpc_engine import nuclear_map
from spyral_utils.nuclear.target import load_target, GasTarget
from pathlib import Path
import numpy as np

output_path = Path("./output/kinematics/c16dd_d2_300Torr_184MeV.h5")
target_path = Path("./target.json")

target = load_target(target_path, nuclear_map)
# Check that our target loaded...
if not isinstance(target, GasTarget):
    raise Exception(f"Could not load target data from {target_path}!")

nevents = 10000

beam_energy = 184.131 # MeV

pipeline = KinematicsPipeline(
    [
        Reaction(
            target=nuclear_map.get_data(1, 2), # deuteron
            projectile=nuclear_map.get_data(6, 16), # 16C
            ejectile=nuclear_map.get_data(1, 2), # deuteron
        )
    ],
    [ExcitationGaussian(0.0, 0.001)], # No width to ground state
    [PolarUniform(0.0, np.pi)], # Full angular range 0 deg to 180 deg
    beam_energy=184.131, # MeV
    target_material=KinematicsTargetMaterial(
        material=target, z_range=(0.0, 1.0), rho_sigma=0.007
    ),
)

def main():
    run_kinematics_pipeline(pipeline, nevents, output_path)

if __name__ == "__main__":
    main()
```

First we import all of our pieces from the attpc_engine library and its dependencies. We also import the Python standard library Path object to handle our file paths.

We then start to define our kinematics configuration. First we define the output path to be an HDF5 file in our output directory. 
We also load a spyral-utils [gas target](https://attpc.github.io/spyral-utils/api/nuclear/target) from a file path.

```python
target = load_target(target_path, nuclear_map)
# Check that our target loaded...
if not isinstance(target, GasTarget):
    raise Exception(f"Could not load target data from {target_path}!")
```

We ask for 10000 events to be sampled, and set our beam energy to about 184 MeV.

```python
nevents = 10000

beam_energy = 184.131 # MeV
```
 
Now were ready to define our kinematics Pipeline. The first argument of the Pipeline is a list of steps, where each step is either a Reaction or Deacy. The first element of the list must *always* be a Reaction, and all subsequent steps are *always* Decays. The residual of the previous step is *always* the parent of the next step. The Pipeline will attempt to validate this information for you. We also must define a list of ExcitationDistribution objects. These describe the state in the residual that is populated by each Reaction/Decay. There is exactly *one* distribution per step. There are two types of predefined ExcitationDistributions (ExcitationGaussian and ExcitationUniform), but others can be implemented by implementing the ExcitationDistribution protocol. Similarly, we use PolarUniform to define a uniform polar angle distribution from 0 degrees to 180 degrees in the center of mass frame (technically, this is from -1 to 1 in cos(polar)). We also define our KinematicsTargetMaterial, which includes our gas target, as well as the allowed space within the target for our reaction vertex (range in z in meters and standard deviation of cylindrical &rho; in meters).

```python
pipeline = KinematicsPipeline(
    # Define a reaction chain (here just one simple scattering reaction)
    [
        Reaction(
            target=nuclear_map.get_data(1, 2), # deuteron
            projectile=nuclear_map.get_data(6, 16), # 16C
            ejectile=nuclear_map.get_data(1, 2), # deuteron
        )
    ],
    # Define our excitations
    [ExcitationGaussian(0.0, 0.0)], # No width to ground state
    [PolarUniform(0.0, np.pi)], # Full angular range 0 deg to 180 deg
    beam_energy=184.131, # MeV
    target_material=KinematicsTargetMaterial(
        material=target, z_range=(0.0, 1.0), rho_sigma=0.007
    ),
)
```

Now we can setup a main function wrapping launching the pipeline

```python
 def main():
    run_kinematics_pipeline(pipeline, nevents, output_path)
```

And finally, we setup the script to call main when run

```python
if __name__ == "__main__":
    main()
```

That's it! This script will then sample 10000 events from the kinematic phase space of ${}^{16}C(d,d')$ and write the data out to an HDF5 file in our output directory.

## Applying the Detector

Below is an example script for running a kinematics sample, which in our example project we would write in `my_sim/apply_detector.py`

```python
from attpc_engine.detector import (
    DetectorParams,
    ElectronicsParams,
    PadParams,
    Config,
    run_simulation,
    SpyralWriter,
)

from attpc_engine import nuclear_map
from spyral_utils.nuclear.target import TargetData, GasTarget
from pathlib import Path

input_path = Path("./output/kinematics/c16dd_d2_300Torr_184MeV.h5")
output_path = Path("./output/detector/")


target_path = Path("./target.json")

gas = load_target(target_path, nuclear_map)
# Check that our target loaded...
if not isinstance(gas, GasTarget):
    raise Exception(f"Could not load target data from {target_path}!")

detector = DetectorParams(
    length=1.0,
    efield=45000.0,
    bfield=2.85,
    mpgd_gain=175000,
    gas_target=gas,
    diffusion=0.277,
    fano_factor=0.2,
    w_value=34.0,
)

electronics = ElectronicsParams(
    clock_freq=6.25,
    amp_gain=900,
    shaping_time=1000,
    micromegas_edge=10,
    windows_edge=560,
    adc_threshold=10,
)

pads = PadParams()

config = Config(detector, electronics, pads)
writer = SpyralWriter(output_path, config, 5_000)

def main():
    run_simulation(
        config,
        input_path,
        writer,
    )

if __name__ == "__main__":
    main()
```

Just like in the kinematics script, we start off by importing a whole bunch of code. Next we define our kinematics input (which is the output of the kinematics script) and an output path. Note that for the output path we simply specify a directory; this is because our writer will handle breaking up the output data into reasonably sized files.

```python
input_path = Path("output/kinematics/c16dd_d2_300Torr_184MeV.h5")
output_path = Path("output/detector/run_0001.h5")
```

Then we define the same gas target that we used in the kinematics pipeline

```python
gas = load_target(target_path, nuclear_map)
# Check that our target loaded...
if not isinstance(gas, GasTarget):
    raise Exception(f"Could not load target data from {target_path}!")
```

and finally we begin to define the detector specific configuration, which is ultimately stored in a Config object.

```python
detector = DetectorParams(
    length=1.0,
    efield=45000.0,
    bfield=2.85,
    mpgd_gain=175000,
    gas_target=gas,
    diffusion=0.277,
    fano_factor=0.2,
    w_value=34.0,
)

electronics = ElectronicsParams(
    clock_freq=6.25,
    amp_gain=900,
    shaping_time=1000,
    micromegas_edge=10,
    windows_edge=560,
    adc_threshold=10,
)

pads = PadParams()

config = Config(detector, electronics, pads)
```

Note that by not passing any arguments to `PadParams` we are using the default pad description that is bundled with the package. See the [detector](./detector/index.md) guide for more details. For the output, we create a SpyralWriter object. This will take in the simulation data and convert it to match the format expected by the Spyral analysis. Note the final argument of the Writer; this is the maximum size of an individual file in events (here we've specified 5,000 events). The writer will then split our output up into many files, which will help later when trying to analyze the data with a framework like Spyral.

```python
writer = SpyralWriter(output_path, config, 5_000)
```

Then, just like in the kinematics script we set up a main function and set it to be run when the script is processed

```python
def main():
    run_simulation(
        config,
        input_path,
        writer,
    )

if __name__ == "__main__":
    main()
```

And just like that, we can now take our kinematic samples and apply detector effects!

## More Details

See the discussion of the systems [here](systems.md)
