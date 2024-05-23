# Kinematics Sampling

The kinematics system is focused on sampling the phase space allowed by a given reaction configuration. This involves specifying a reaction system and the number of samples needed, as well as the shape of the underlying distributions to be sampled from. Below we'll talk about each of these ideas and cover how to specify them in attpc_engine.

## Reactions

The core underlying model of the kinematics is that of a [reaction](../../api/kinematics/reaction.md). In attpc_engine there are two types of reactions. 

The first is called Reaction, which is of the kind $a(b, c)d$ where you have an incoming projectile (sometimes referred to as the beam) incident upon a target particle. The nuclei react and produce two products, one of which we refer to as an ejectile (this is typically the $c$ in the reaction equation) and the other we refer to as the residual (typically the $d$ in the above equation). In attpc_engine you define a reaction by providing the projectile, target, and ejectile nuclei. The residual is calculated for you based on the other inputs. 

The second is called a Decay, which is of the kind $a \rightarrow b+c$ where the parent nucleus, $a$, decays into two products $b$ and $c$. In attpc_engine you specify a Decay by the parent and one of the decay products. The other product is calculated for you.

Tying these two together is the concept of a reaction chain. Some experiments investigate the following: $a(b,c)d\rightarrow e+f$ where you have a primary reaction resulting in a residual that then undergoes a decay. In attpc_engine you can simulate this by providing a Reaction, followed by a Decay. The residual of the Reaction should then match the parent of the Decay. This will be validated for you at runtime.

In attpc_engine, reaction chains must *always* start with a Reaction, which can *only* be followed by Deacys. However, as long as it is energetically allowed, you can chain as many Decays as you would like.

## Sampling

With a reaction specified, we now need to describe how to sample from the reaction. In particular, there are a few key concepts: the beam (projectile) energy, the reaction angle, the propogation of previous steps, and the excitation of residual nuclei. Each of these plays a role in defining and sampling the phase space.

### The Beam Energy

In AT-TPC experiments, the detector often samples a range of beam energies due to the active target. That is, as the beam travels through the gas, it loses energy until it finally undergoes a reaction. To simulate this effect, you can specify a target material and the range of z-positions over which the reaction can take place (see [here](../../api/kinematics/pipeline.md)). The simulation will then uniformly sample along that z-range and calculate the beam energy for a given z. As an additional effect, you can specify a standard deviation in cylindrical &rho; over which to sample a beam offset (simulate a beam spot) in a normal distribution. The beam is assumed to be uniformly distributed in cylindrical &theta;. Note that the &rho; sampling has no effect on the beam energy. The beam is assumed to be travelling parallel to the beam axis.

### Reaction Angles

With a beam energy specified, the next step is to sample the reaction angles. This means sampling both spherical &theta; and &phi; in the center-of-mass frame of the incoming system (the target + projectile in the case of a Reaction, parent in the case of a Decay). The polar angle &theta; is uniformly distributed in cos(&theta;) from [-1, 1], while the azimuthal angle is uniformly distributed in the range [0, 2&pi;]. With the initial energy of the system and the outgoing angles, the system is completely described and the energies of the products can be found using simple conservation of momentum and energy. In a Reaction the angles of the ejectile is sampled. In a Decay the angles of the input product are sampled.

### Propogation through the Chain

In the above we we're mainly disucssing a single step. Howeve, you may be wondering how this affects subsequent steps, that is how we propogate the result of sampling the Reaction to the Decay. This is done by taking the resulting kinematics of the residual from the Reaction and passing them to the parent of the Decay. You can think of it as setting the "beam energy" for the Decay.

### Excitations

In many reactions we're not interested in just the ground state of a nucleus, but also it's excited states (which may Decay!). In attpc_engine an excited state may be specified for a residual in a Reaction or a non-input residual (the one that's calculated for you) in a Decay. Excitations can be specified as either a uniform distribution or Gaussian distribution by default, but this is an extensible feature (see the reference [here](../../api/kinematics/excitation.md)) so more exotic distributions can be implemented. Excitations can be tricky however, when it comes to allowed energies. In particular, if an excitation is near the energy threshold