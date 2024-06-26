# Detector system

The detector system applies detector and electronics effects to every event in the input kinematics file, generated from the kinematics system, to simulate how they would look in the AT-TPC. It is a Monte-Carlo simulation that aims to capture the overarching behavior of the detector and its electronics, not mimic their exact behavior which is horrendously complicated and not completely understood. The detector system code is based on code developed by Daniel Bazin originally in Igor Pro and includes effects pointed out by Adam Anthony in his [thesis](https://ezproxy.msu.edu/login?url=https://www.proquest.com/pqdtglobal1/dissertations-theses/fission-lead-region/docview/2855740534/sem-2?accountid=12598).

# Output

The detector system outputs a HDF5 file with a point cloud for each event and some IC related attributes. This ensures that the output can be directly fed into Spyral for analysis. Note that this means that the point cloud construction phase of Spyral should not be run! The HDF5 file generated from the detector system simulation should begin the Spyral analysis with clustering.

It is worth elaborating why the detector system outputs point clouds instead of raw traces. Using raw traces would allow for the full testing of the Spyral analysis beginning with the point cloud reconstruction phase. The main problem with traces is that they simply take up too much disk space. The objective of a Monte-Carlo simulation is to simulate a phenomena enough times that statistical error is essentially removed and whatever is observed is due to the phenomena itself. To this end, it is often desireable to simulate hundreds of thousands of events in the AT-TPC. To prevent egregiously large trace files, the point cloud reconstruction can be cooked into the simulation to produce orders of magnitude smaller point cloud files. To maintain speed, the point clouds are simulated directly rather than simulating the traces and applying Spyral's point cloud construction phase to them. Almost certainly analysis effects due to the point cloud construciton phase are lost because of this. However, the decrease in file size using point clouds is too great to ignore.

# Core steps of detector system

The detector system code simulates an event by going through a series of steps to ultimately produce its point cloud. For each nucleus in the exit channel of an event, the detector system:

1. Simulates the track of the nucleus by solving its equation of motion in the AT-TPC
2. Determines how many electrons are created at each point of the nucleus' track
3. Converts the z coordinate of each point to the corresponding time bucket
3. Transports each point of the nucleus' track to the pad plane, applying diffusion if necessary, to construct its point cloud
4. Converts the point cloud to Spyral format

We can now describe each step in greater detail.

# Track simulation

The AT-TPC contains an inherent electric field parallel to the beam axis. It is also usually placed inside a solenoid that provides a magnetic field in the same direction. This means that a charged particle in the AT-TPC has an equation of motion given by the relativistic Lorentz force

$$ 
\frac{d\pmb{p}}{dt} = q (\pmb{E} + \pmb{v} \times \pmb{B})
$$

where $\pmb{p}$ is the relativistic momentum, $\pmb{v}$ is the three velocity, $\pmb{E}$ is the electric field, and $\pmb{B}$ is the magnetic field. To prevent the ODE solver from calculating either very large or very small numbers, we actually divide both sides of this equation by the nucleus' rest mass and thus solve this differential equation for $\gamma \beta$. 

The equation of motion is subject to the physical boundaries of the detector and the condition that the nucleus has more than 1 eV of kinetic energy. Solutions are provided by the solver at time steps of 1e-10 s.

# Electron creation

Using $\gamma \beta$ provided by the ODE solver at each time step, the number of electrons created through ionization of the gas by the nucleus can be calculated. $\gamma \beta$ is used to find $\beta$ through manipulation of the $\gamma$ factor equation. $\gamma \beta$ is divided by $\beta$ to uncouple $\gamma$ which is then used in the relativistic kinetic energy formula to find the kinetic energy of the nucleus at that point. The difference in energy between two successive points in the trajectory is defined as the energy lost by the nucleus at the point at the later time.

With knowledge of the energy lost by the nucleus at the points in its trajectory, the electrons made can be found. The input W-value is the average energy needed to create an electron-ion pair in the gas, thus the energy lost at each point is divided by this value to find the electrons made. However, ionization is part statistical, so this number is taken as the mean of a normal distribution that is then sampled to find the actual number of electrons at each point. We say part statistical because it has been experimentally shown that the amount of electrons made does not follow purely statistical calculations. Therefore, the input Fano factor of the gas is a multiplicative constant that modifies the mean number of electrons made at each point in an attempt to capture the non-statistical nature of electron-ion pair creation in the gas.

Only an integer amount of electrons can be made. Trunction is applied and points with less than one electron made are removed from the track.

# Convert z position to time buckets

The GET electronics records information in discrete time intervals called time buckets. The trajectory of the nucleus returned from the ODE solver is described by three Cartesian coordinates. The z coordinate of each point then must be converted to time buckets. The formula to convert the z coordinate $z_{coord}$ to time buckets $z_{tb}$ is given by 

$$ 
z_{tb} = \frac{l - z_{coord}}{v_{e}} + m_{tb}
$$

where $l$ is the length of the AT-TPC active volume, $v_{e}$ is the electron drift velocity, and $m_{tb}$ is the micromegas time bucket. 

Two important points are made about this equation. First, $z_{coord}$ is subtracted from $l$. This is because the coordinate system used has the AT-TPC window at $z=0$ and the micromegas at $z=l$ and time buckets are recorded with respect to the micromegas (electrons drift towards it). Second, the resulting time bucket has $m_{tb}$ added to it. This is due to the fact that the micromegas edge of the detector does not necessarily correspond to a time bucket of zero. A similar statement can be made for the window edge of the detector; it does not necessarily correspond to the maximum recordable time bucket.

# Point cloud construction

Each point in the point cloud has three pieces of information recorded: the pad it belongs to, how many electrons it contains, and its time bucket. Point cloud construction primarily deals with determining the first two from the calculated points in the nucleus' trajectory. There are two ways to construct a point cloud, and we start by describing the simple case of no transverse diffusion.

When the transverse diffusion is specified to be zero, point clouds are constructed from what we call "point transport" where the trajectory is projected onto the pad plane and the pad that each point hits is found from a lookup table. In this scheme, the electrons are not spread and hence the number of electrons that hit each pad is trivially the number of electrons at each point of the trajectory.

When the transverse diffusion is non-zero, point clouds are constructed from the more involved "transverse transport" scheme. The electrons made at each point are smeared from a bivariate normal distribution over the pad plane. The standard deviation $\sigma$ of the bivariate distribution is the same for both the x and y directions and is given by

$$
\sigma = \sqrt{2 D v_{e} t / E}
$$

where D is the diffusion coefficient, $t$ is the average electron drift time ($z_{tb}$), and $E$ is the magnitude of the electric field. See [Peisert and Sauli](https://cds.cern.ch/record/154069?ln=en) for the derivation of this equation. The smearing is done via a discretized grid and the same lookup table is used to find which pad each pixel hits.

Despite the transport schema chosen, each point in the cloud has its corresponding time bucket. This time bucket is truncated to an integer from the exact time bucket found in the previous step because the GET system records in discrete time samples.

In this simulation, we only allow for transverse diffusion. Although longitudinal diffusion exists, it is not included. The reason is that it was found in early iterations of this simulation to not matter. Only for large longitudinal diffusion values noticeable effects were observed.

# Format the point cloud

The final step of the detector simulation is to convert the simulated point cloud to Spyral format. We will avoid discussing what Spyral exactly expects since it is written in the Spyral documentation. We want to point out, though, that the defined SimulationWriter class allows for the user to easily implement a custom converter to write the simulated data to any format, or include any extra information, they may want. This is highly relevant for individuals who have tweaked Spyral. For example, take someone who has changed base Spyral to write point clouds that inlcude extra information. In order for the simulation to work with their code, they will have to write a custom converter that adds the extra fields to the simulated point clouds.

The only Spyral point cloud fields that require special consideration are the amplitude and integral of the point. Recall that the simulation has only recorded the number of electrons at each point. In real data, the amplitude and integral refer to the electrical signal induced by the drifting electrons on a pad. It is from such a signal that a real point cloud is created. This is to say that in order to find these two parameters from the number of electrons created at a point, we need to know the response of the electronics. At the present time, the simulation uses the theoretically derived response function provided by the chip manufacturer to estimate these quantities.