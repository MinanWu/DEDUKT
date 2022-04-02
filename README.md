This is my fork of DEDUKT (DistributED coda K-mer CounTer), a distributed-memory parallel *K*-mer counter with NVIDIA GPU support developed by the PASSIONLab at Lawrence Berkeley National Laboratory and UC Berkeley.

So far, it consists only of slight modifications to the build files that were necessary to allow DEDUKT to be built and installed again on NERSC’s Cori supercomputer in 2022.

More details can be found in the [README.md file for the original version of DEDUKT](https://github.com/PASSIONLab/DEDUKT/blob/master/README.md).



## Building and Installing on NERSC’s Cori

First, load the required modules by running the following commands:

```
module load cgpu
module load cuda/10.2.89
module load nvhpc
module load openmpi
```

Then, run the install script as usual:

```
sh install.sh
```