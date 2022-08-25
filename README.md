# hyperfinemoments

An implementation of the hyperfine moment fitting method (lindoy and Manolopoulos, Phys. Rev. Lett. 120, 220604) for simulating the real time quantum dynamics of isotropic central spin Hamiltonians:
$$ \hat{H} = \omega_z \hat{S}_z + \sum_{i=1}^N a_i \hat{\boldsymbol{S}} \cdot \hat{\boldsymbol{I}}_i $$
with an initial infinite temperature nuclear spin bath.

This method approximates the Hamiltonian of this system as
$$ \hat{H} = \omega_z \hat{S}_z + \sum_{b=1}^M a_{b} \sum_{i=1}^{N_b} \hat{\boldsymbol{S}} \cdot \hat{\boldsymbol{I}}_{bi} $$
where we have have selected $M$ sets of $N_b$ nuclear spins that interact with the central spin with the same hyperfine coupling constant.  Where the total number of nuclear spins is the same as in the original problem, and the modified hyperfine coupling constants are chosen so that we reproduce the first $M+1$ moments of the original hyperfine distribution.

This approximate Hamiltonian has a large degree of symmetry, which is exploited by this code to lead allow for practical simulations of this approximate Hamiltonian even for moderately large values of $M$.  By increasing $M$ the dynamics arising from this Hamiltonian approaches that for the full Hamiltonian, and often can be converged with $M \ll N$.  For large spin systems where each hyperfine coupling constant scales as $1/sqrt(N)$, the dynamics converges incredibly rapidly with $M$, so this approach can be quite efficient for very large systems for which the full Hilbert space dimension is impractically large.

The hyperfinemoments program expects a string specifying the path to the input file.  A series of example input files are included in the inputs directory.

## Dependencies
External:
    - Required: [RapidJSON](https://rapidjson.org/) - input file parsing
    - Optional: [ChaiScript](https://chaiscript.com/) - necessary fro ChaiScript function based specification of hyperfine coupling constants
    - Optional: [ChaiScript_Extras](https://github.com/ChaiScript/ChaiScript_Extras) - necessary fro ChaiScript function based specification of hyperfine coupling constants

Submodules (automatically downloaded if the respository is cloned using git clone --recurse-submodules https://github.com/lpjpl/hyperfinemoments.git):
    - [linalg](https://github.com/lpjpl/linalgt) - wrapper for the BLAS and LAPACK libraries
    - [utils_cpp](https://github.com/lpjpl/utils_cpp) - input file parsing/exception handling/maths utility functions

# Compile Instructions
This code requires cmake version 3.11 in order to compile. From hyperfine moment base directory (${hyperfine_base}) run:
```console
mkdir build
cd build
cmake ../
make
```

This builds the executable as ${hyperfine_base}/build/src/hyperfinemoment

The cmake build command will scan the directory (${hyperfine_base}/external) for any of the required external libraries, and use them if present.  If they aren't it will attempt to download the github repositories and make them available using the cmake FetchContent package.

ChaiScript functionality can be disabled by include  -DUSE_CHAI=Off in the cmake command.

BLAS versions can be selected by setting the -DCMAKE_BLA_VENDOR variable in the cmake commmand (see [cmake FindBlas](https://cmake.org/cmake/help/latest/module/FindBLAS.html) for details).

   
    

