# Detector system

The detector system applies detector and electronics effects to every event in the input kinematics file, generated from the kinematics system, to simulate how they would look in the AT-TPC. It is a Monte-Carlo simulation that aims to capture the overarching behavior of the detector and its electronics, not mimic their exact behavior which is horrendously complicated and not completely understood. The detector system code is based on code developed by Daniel Bazin originally in Igor Pro and includes effects pointed out by Adam Anthony in his [thesis](https://ezproxy.msu.edu/login?url=https://www.proquest.com/pqdtglobal1/dissertations-theses/fission-lead-region/docview/2855740534/sem-2?accountid=12598).

# Output

The detector system outputs a HDF5 file with a point cloud for each event and some IC related attributes. This ensures that the output can be directly fed into Spyral for analysis. Note that this means that the point cloud construction phase of Spyral should not be run! The HDF5 file generated from the detector system simulation should begin the Spyral analysis with clustering.

It is worth elaborating why the detector system outputs point clouds instead of raw traces. Using raw traces would allow for the full testing of the Spyral analysis beginning with the point cloud reconstruction phase. The main problem with traces is that they simply take up too much data. The objective of a Monte-Carlo simulation is to simulate a phenomena enough times that statistical error is essentially removed and whatever is observed is due to the phenomena itself. To this end, it is often desireable to simulate hundreds of thousands of events in the AT-TPC. To prevent egregiously large trace files, the point cloud reconstruction can be cooked into the simulation to produce orders of magnitude smaller point cloud files. To maintain speed, the point clouds are simulated directly rather than simulating the traces and applying Spyral's point cloud construction phase to them. Almost certainly analysis effects due to the point cloud construciton phase are lost because of this. However, the decrease in file size using point clouds is too great to ignore.

# Core steps of detector system

The detector system code simulates an event by going through a series of steps to ultimately produce its point cloud. For each nucleus in the exit channel of an event, the detector system:

1. Simulates the track of the nucleus by solving its equation of motion in the AT-TPC
2. Determines how many electrons are created at each point of the nucleus' track
3. Transports each point of the nucleus' track to the pad plane, applying diffusion if necessary, to construct its point cloud

We can now describe each step in greater detail.

# Track simulation

The AT-TPC contains an inherent electric field parallel to the beam axis. It is also usually placed inside a solenoid that provides a magnetic field in the same direction. This means that a charged particle in the AT-TPC has an equation of motion given by the relativistic Lorentz force

$$ 
\frac{d\pmb{p}}{dt}=q (\pmb{E} + \pmb{v} \times \pmb{B})
$$

where $\pmb{p}$ is the relativistic momentum, $\pmb{v}$ is the three velocity, $\pmb{E}$ is the electric field, and $\pmb{B}$ is the magnetic field. To prevent the ODE solver from calculating either very large or very small numbers, we actually divide both sides of this equation by the nucleus' rest mass and thus solve this differential equation for $\gamma \beta$. It is subject to the physical boundaries of the detector and the condition that the nucleus has more than 1 eV of kinetic energy.

# Electron-ion pair creation

ss