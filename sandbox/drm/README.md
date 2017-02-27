# Dynamic Representational Modeling (DRM)

End-to-end training of neural systems.

Biological constraints + modularity + gradient descent

## TO DO

* make iterator work seamlessly
* also make the links between populations neural networks; this also
allows implementation of synaptic delays. We actually need this! If we
have a mechanism to generate delays then models with cycles become acyclic!
In contrast, the readout mechanism can be instantaneous
All of this does require that we sample at very high rates... Maybe we
can allow the connection to learn the delay. Any >0 delay will do...
* Can we use DRM to reverse engineer circuits in systems neuroscience? Think Ganguli, Bethge, etc.
* Allow each population to connect to multiple outputs
* Add documentation!
* First problem to fix: how to implement a learnable or chosen conduction delay? tapped delay lines?
