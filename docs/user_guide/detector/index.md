# Detector system

The detector system applies AT-TPC specific detector and electronics effects to every 
event in the input kinematics file, generated from the kinematics system, to simulate 
how they would look in the AT-TPC. It is a Monte-Carlo simulation that aims to capture 
the overarching behavior of the detector and its electronics, not mimic their exact 
behavior which is horrendously complicated and not completely understood. The detector 
system code is based on code developed by Daniel Bazin originally in Igor Pro and 
includes effects pointed out by Adam Anthony in his [thesis](https://ezproxy.msu.edu/login?url=https://www.proquest.com/pqdtglobal1/dissertations-theses/fission-lead-region/docview/2855740534/sem-2?accountid=12598).

## Core steps of detector system

The detector system code simulates an event by going through a series of steps to 
ultimately produce its point cloud. For each event, the detector system:

1. Generate the trajectory of the each nucleus in the exit channel (or the nuclei specified by the user) of the event by solving its equation of motion in the AT-TPC with a fine grained-timestep
2. Determines how many electrons are created at each point of each nucleus' trajectory
3. Converts the z coordinate of each point in each trajectory to a GET electronics sampling-frequency based Time Bucket using the electron drift velocity
3. Transports each point of the nucleus' trajectory to the pad plane, applying diffusion if requested, to identify the sensing pad for each timestep
5. Form the point cloud from all of the trajectories, and label each point with which nucleus produced it
4. Write the point cloud and labels to disk

We can now describe each step in greater detail.

## Track simulation

The AT-TPC contains an inherent electric field parallel to the beam axis. It is also 
usually placed inside a solenoid that provides a magnetic field in the same direction. 
This means that a charged particle in the AT-TPC has an equation of motion given by the
relativistic Lorentz force

$$ 
\frac{d\pmb{p}}{dt} = q (\pmb{E} + \pmb{v} \times \pmb{B})
$$

where $\pmb{p}$ is the relativistic momentum, $\pmb{v}$ is the three velocity, $\pmb{E}$
is the electric field, and $\pmb{B}$ is the magnetic field. To prevent the ODE solver 
from calculating either very large or very small numbers, we actually divide both 
sides of this equation by the nucleus' rest mass and thus solve this differential 
equation for $\gamma \beta$. 

The equation of motion is subject to the physical boundaries of the detector and the 
condition that the nucleus has more than 1 eV of kinetic energy. Solutions are provided 
by the solver at time steps of 1e-10 s.

## Electron creation

Using $\gamma \beta$ provided by the ODE solver at each time step, the number of 
electrons created through ionization of the gas by the nucleus can be calculated. 
$\gamma \beta$ is used to find $\beta$ through manipulation of the $\gamma$ factor 
equation. $\gamma \beta$ is divided by $\beta$ to uncouple $\gamma$ which is then used 
in the relativistic kinetic energy formula to find the kinetic energy of the nucleus at
that point. The difference in energy between two successive points in the trajectory is
defined as the energy lost by the nucleus at the point at the later time.

With knowledge of the energy lost by the nucleus at the points in its trajectory, the 
electrons made can be found. The input W-value is the average energy needed to create 
an electron-ion pair in the gas, thus the energy lost at each point is divided by this 
value to find the electrons made. However, ionization is part statistical, so this 
number is taken as the mean of a normal distribution that is then sampled to find the 
actual number of electrons at each point. We say part statistical because it has been 
experimentally shown that the amount of electrons made does not follow purely 
statistical calculations. Therefore, the input Fano factor of the gas is a 
multiplicative constant that modifies the mean number of electrons made at each point 
in an attempt to capture the non-statistical nature of electron-ion pair creation in 
the gas.

Only an integer amount of electrons can be made. Trunction is applied and points with 
less than one electron made are removed from the track.

## Convert z position to time buckets

The GET electronics records information in discrete time intervals called time buckets. 
The trajectory of the nucleus returned from the ODE solver is described by three 
Cartesian coordinates. The z coordinate of each point then must be converted to time 
buckets. The formula to convert the z coordinate $z_{coord}$ to time buckets $z_{tb}$ 
is given by 

$$ 
z_{tb} = \frac{l - z_{coord}}{v_{e}} + m_{tb}
$$

where $l$ is the length of the AT-TPC active volume, $v_{e}$ is the electron drift 
velocity, and $m_{tb}$ is the micromegas time bucket. 

Two important points are made about this equation. First, $z_{coord}$ is subtracted 
from $l$. This is because the coordinate system used has the AT-TPC window at $z=0$ and
the micromegas at $z=l$ and time buckets are recorded with respect to the micromegas 
(electrons drift towards it). Second, the resulting time bucket has $m_{tb}$ added to 
it. This is due to the fact that the micromegas edge of the detector does not 
necessarily correspond to a time bucket of zero. A similar statement can be made for 
the window edge of the detector; it does not necessarily correspond to the maximum 
recordable time bucket.

## Point cloud construction

A point cloud is an Nx3 matrix, where each row corresponds to a point containing the 
following information: sensing pad number, detection time in GET Time Buckets, and the 
number of electrons detected. Point cloud construction primarily deals with determining
the first two from the calculated points in the nucleus' trajectory. There are two ways
to create a point cloud, and we start by describing the simple case of no transverse 
diffusion.

When the transverse diffusion is specified to be zero, point clouds are constructed 
from what we call "point transport" where the trajectory is projected onto the pad 
plane and the pad that each point hits is found from a lookup table. In this scheme, 
the electrons are not spread and hence the number of electrons that hit each pad is 
trivially the number of electrons at each point of the trajectory.

When the transverse diffusion is non-zero, point clouds are constructed from the more 
involved "transverse transport" scheme. The electrons made at each point are smeared 
from a bivariate normal distribution over the pad plane. The standard deviation 
$\sigma$ of the bivariate distribution is the same for both the x and y directions and 
is given by

$$
\sigma = \sqrt{2 D v_{e} t / E}
$$

where D is the diffusion coefficient, $t$ is the average electron drift time 
($z_{tb}$), and $E$ is the magnitude of the electric field. See 
[Peisert and Sauli](https://cds.cern.ch/record/154069?ln=en) for the derivation of this
equation. The smearing is done via a discretized grid and the same lookup table is used
to find which pad each pixel hits.

Despite the transport schema chosen, each point in the cloud has its corresponding time
bucket. This time bucket is truncated to an integer from the exact time bucket found in
the previous step because the GET system records in discrete time samples.

In this simulation, we only allow for transverse diffusion. Although longitudinal 
diffusion exists, it is not included. The reason is that it was found in early 
iterations of this simulation to not matter. Only for large longitudinal diffusion 
values noticeable effects were observed.

## Point cloud labels

Each point cloud also has an associated length N label array of per-point labels. These
labels indicate which nucleus produced the specific point in the cloud. The labels are 
given by the index of the nucleus in the kinematics list. For example, in a simple one 
step reaction a(b,c)d a=0, b=1, c=2, d=3. In a two step, a(b,c)d->e+f, e=4, f=5, and so
on for more complex scenarios. These labels are particularly useful for evaluating the
performance of machine learning methods like clustering in downstream analyses.

## Which nuclei get simulated in the detector simulation

By default, the detector simulation considers all nuclei in the exit channel. That is,
in a two step reaction a(b,c)d->e+f, the nuclei c, e, f are included in the detector
simulation and used to generate a point cloud. However, this is not always desireable. 
In some usecases, only on particle is interesting and all others simply generate extra
data. As such, the `run_simulation` function contains an optional keyword argument 
`indices` which can be used to specify the indices of the nuclei to include in the 
detector simulation. Indices follow the same convention as the kinematics simulation,
i.e. for our earlier example two step reaction a=0, b=1, c=2, d=3, e=4, and f=5.

## Why Point clouds

It is worth elaborating why the detector system outputs point clouds instead of raw 
traces. Using raw traces would allow for the full testing of AT-TPC analysis beginning 
with the signal analysis. As with any Monte Carlo simulation, attpc_engine needs enough
samples to converge upon a meaningful result. To this end, it is often desireable to
simulate a large numbber (more than 100,000) of events in the AT-TPC. Traces require 
such a large memory footprint, both on disk and in memory, that they are prohibitive to
simulating enough samples to reach convergence. Additionally, methods which effectively
mimic the True response of the AT-TPC pad plane and electronics are often dubious, 
relying on approximations that are hard to validate.

## Writing the Point cloud to disk

The final step of the process is to write the simulated detector response and labels to disk. 
attpc_engine aims to be analysis agnostic, and as such does not fomrally define the 
format of the output. Instead, attpc_engine provides a `SimulationWriter` 
[protocol](https://typing.readthedocs.io/en/latest/spec/protocol.html#protocols) class.
Users can define their own writer, so long as it implements the methods required by the 
`SimulationWriter` protocol.

### Use with Spyral

For convience, attpc_engine includes a `SpyralWriter`, which will output a HDF5 file 
formatted to mimic [Spyral](https://attpc.github.io/Spyral), the AT-TPC group-supported
analysis framework. `SpyralWriter` also includes the useful feature of splitting the 
output into multiple files based on number of events, which can help take advantage of 
parallel analysis pipelines. Note that attpc_engine does not claim to provide values in
the integral or amplitude fields of a Spyral point cloud that will match real data; 
`SpyralWriter` use the [theoretical response function](https://www.sciencedirect.com/science/article/pii/S0168900216309408?casa_token=DAWbhPvX49MAAAAA:Zen7nFs-pgG9wu__g0rrzus01B1pa7ZYspbz_KOnn-dZyhXNglEqtgywFKwYrvKKsIwEY10n49XV)
of the GET electronics to make an attempt to approximate the amplitude and integral, 
but this should be regarded as dubious at best.

Given that attpc_engine outputs point clouds, the results should not be run through the
PointcloudPhase, rather starting directly with the ClusterPhase. Note that because 
Spyral expects trace data, one will need to "fake" the trace datapath. The simplest 
way to do this is to point the trace path to the Pointcloud directory of the workspace.
Just be sure not to run the PointcloudPhase in this case!

Along with the normal point cloud data, the writer also writes the associated labels
under the same "cloud" group. Each label array is named "label_#" where # is the event
number for that label array.
