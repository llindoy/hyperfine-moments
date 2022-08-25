#ifndef CENTRAL_SPIN_HPP
#define CENTRAL_SPIN_HPP

#include "MomentFitting.hpp"
#include "HomogeneousCoupling.hpp"
#include "system_hamiltonian.hpp"

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <random>
#include <array>
#include <vector>


#ifdef USE_CHAISCRIPT
#include <chaiscript/chaiscript.hpp>
#include <chaiscript/chaiscript_stdlib.hpp>
#include "chaiscript/extras/math.hpp"
#endif

#include <krylov_integrator.hpp>
#include <io/input_wrapper.hpp>

template <typename T> 
void welford_variance_update(std::array<T, 2>& curr, const T& val, size_t count)
{
    T delta = val - curr[0];
    curr[0] += delta/count;
    T delta2 = val - curr[0];
    curr[1] += delta*delta2;
}

template <typename T, typename IObj> 
void central_spin(const IObj& doc)
{
    using complex_type = linalg::complex<T>;
    try
    {
        //read in the system parameters
        std::vector<T> A;

        if(IOWRAPPER::has_member(doc["system"], "a"))
        {
            CALL_AND_HANDLE(IOWRAPPER::load<std::vector<T>>(doc["system"], "a", A), "Failed to load hyperfine coupling constants.");
        }
#ifdef USE_CHAISCRIPT
        else if(IOWRAPPER::has_member(doc["system"], "a(i,n)"))
        {
            std::string Afunc;
            CALL_AND_HANDLE(IOWRAPPER::load<std::string>(doc["system"], "a(i,n)", Afunc), "Failed to load hyperfine coupling constants.");
            size_t Ns;
            CALL_AND_HANDLE(IOWRAPPER::load<uint64_t>(doc["system"], "n", Ns), "Failed to load hyperfine coupling constants.");
            T tau;
            CALL_AND_HANDLE(IOWRAPPER::load<T>(doc["system"], "tau", tau), "Failed to load hyperfine coupling constants.");

            std::string fstr = std::string("var A = fun(i, N){") + Afunc + std::string(";}");
            chaiscript::ChaiScript chai;
            auto mathlib = chaiscript::extras::math::bootstrap();
            chai.add(mathlib);
            
            auto Af = chai.eval<std::function<T(size_t, size_t)>>(fstr.c_str());  

            A.resize(Ns);
            for(size_t i = 0; i < Ns; ++i)
            {
               A[i] = Af((i+1), Ns)/tau;
            }
        }
#endif
        else
        {
            RAISE_EXCEPTION("Failed to read in hyperfine coupling constants.");
        }


        //the frequency of the zeeman term
        T w;
        CALL_AND_HANDLE(IOWRAPPER::load<T>(doc["system"], "omega", w), "Failed to load hyperfine coupling constants.");


        /*
         * Read in the simulation parameters
         */
        //the cumulative weight tolerance to determine which of the symmetry blocks are included in the calculation
        T ctol;
        CALL_AND_HANDLE(IOWRAPPER::load<T>(doc, "ctol", ctol), "Failed to load hyperfine coupling constants.");
        ASSERT(ctol >= 0 && ctol < 1,  "ctol out of bounds.");

        //the number of spin-groups that will be used for the hyperfine fitting algorithm
        uint64_t M;
        CALL_AND_HANDLE(IOWRAPPER::load<uint64_t>(doc, "m", M), "Failed to load hyperfine coupling constants.");
        ASSERT(M > 0, "Must have at least one symmetrised hyperfine.");

        linalg::vector<complex_type> psi0;  
        CALL_AND_HANDLE(IOWRAPPER::load<linalg::vector<complex_type>>(doc, "psi0", psi0), "Failed to load in the initial central spin wavefunction.");

        uint64_t ntraj;
        CALL_AND_HANDLE(IOWRAPPER::load<uint64_t>(doc, "ntraj", ntraj), "Failed to load the number of trajectories to run.");

        T dt;
        CALL_AND_HANDLE(IOWRAPPER::load<T>(doc, "dt", dt), "Failed to load timestep.");
        ASSERT(dt > 0, "Timestep must be greater than zero.");
        T tmax;
        CALL_AND_HANDLE(IOWRAPPER::load<T>(doc, "tmax", tmax), "Failed to load maximum time.");
        ASSERT(tmax > 0, "Integration time must be greater than 0.");
        size_t Nsteps = size_t(tmax/dt) + 1;

        bool has_outfile = false;
        std::string ofname;
        CALL_AND_HANDLE(has_outfile = IOWRAPPER::load_optional<std::string>(doc, "outfile", ofname), "Failed to load output file name.");

        bool print_block_correlations = false;
        CALL_AND_HANDLE(IOWRAPPER::load_optional<bool>(doc, "printblockcorrelations", print_block_correlations), "Failed to load whether or not to read in block correlations.");

        bool sample_blocks = false;
        CALL_AND_HANDLE(IOWRAPPER::load_optional<bool>(doc, "sampleblocks", sample_blocks), "Failed to load whether or not to stochastically sample blocks.");

        uint64_t seed = 0;
        CALL_AND_HANDLE(IOWRAPPER::load_optional<uint64_t>(doc, "seed", seed), "Failed to load seed.");


        uint64_t minntraj = 0;
        CALL_AND_HANDLE(IOWRAPPER::load_optional<uint64_t>(doc, "minntraj", minntraj), "Failed to load minntraj.");


        uint64_t krylov_dim = 4;
        CALL_AND_HANDLE(IOWRAPPER::load_optional<uint64_t>(doc, "krylovdim", krylov_dim), "Failed to load krylov subspace dimension.");
        ASSERT(krylov_dim > 1, "Krylov dimension must be greater than 1.");

        T krylov_tol = 1e-8;
        CALL_AND_HANDLE(IOWRAPPER::load_optional<T>(doc, "krylovtol", krylov_tol), "Failed to load krylov subspace tolerance.");
        ASSERT(krylov_tol > 0, "Krylov tolerance must be greater than zero.");



        //read in the operators that we want to evaluate the observables of
        std::vector<linalg::matrix<complex_type>> ops;

        ASSERT(IOWRAPPER::has_member(doc, "ops"), "Failed to read in the observable operators.");
        //if we just have a single matrix then we just load it
        if(IOWRAPPER::is_type<linalg::matrix<complex_type>>(doc, "ops"))
        {
            ops.resize(1);
            CALL_AND_HANDLE(IOWRAPPER::load<linalg::matrix<complex_type>>(doc, "ops", ops[0]), "Failed to load in observable operator matrix.");
        }
        else if(IOWRAPPER::is_array(doc, "ops"))
        {
            ops.resize(IOWRAPPER::size(doc, "ops"));
            for(size_t i = 0; i < ops.size(); ++i)
            {
                CALL_AND_HANDLE(IOWRAPPER::load<linalg::matrix<complex_type>>(doc["ops"], i, ops[i]), "Failed to load in observable operator matrix.");
            }
        }
        else
        {
            RAISE_EXCEPTION("Could not figure out how to load the operators.");
        }

        //setup the arrays for storing the expectation values
        std::vector<linalg::vector<T>> vals(ops.size());
        for(size_t i = 0; i < vals.size(); ++i)
        {
            vals[i].resize(Nsteps+1);
            vals[i].fill_zeros();
        }

        std::vector<linalg::vector<T>> sems(ops.size());
        for(size_t i = 0; i < sems.size(); ++i)
        {
            sems[i].resize(Nsteps+1);
            sems[i].fill_zeros();
        }
    
        /* Now we can set up the simulations */
    
        //determine the optimal distribution of hyperfines and number of nuclear spins per block
        //based on splitting it into M blocks of equivalent hyperfines
        auto val = Moments::fit(A, M);
        
        std::vector<T> Adisc(M);
        for(size_t i = 0; i < M; ++i){
            Adisc[i] = std::get<0>(val[i]);
        }

        //now given the blocks of various numbers of nuclear spins (with homogeneous coupling for all of those hyperfines)
        //perform a direct sum decomposition to work out simplified hamiltonians.
        std::vector<std::vector<T> > weights(M);

        size_t nblocks = 1;
        for(size_t i=0; i<M; ++i)
        {
            size_t N = std::get<1>(val[i]);
            std::vector<uint64_t> nnuc(N,2);                                            //create an array corresponding to having the correct number of spin halfs
            HomogeneousCoupling<T>::direct_sum_decompose(nnuc, weights[i]);             //now perform the direct sum decomposition returning the weights for each term /(2J+1)
            for(size_t j=0; j<weights[i].size(); ++j){weights[i][j]*= ((j+1)/2.0);}     //scale the weights so that they are correct
            nblocks *= (N&1)?((N+1)/2):(N/2+1);
        }



        //now we construct the weights and the information about the simulations that we wish to perform
        std::vector<std::tuple<uint64_t, T, std::vector<size_t> > > simulation_info(nblocks, std::make_tuple(1, 1.0, std::vector<size_t>(M, 0)));
        for(size_t i=0; i<nblocks; ++i){
            int64_t ind = i;
            std::get<1>(simulation_info[i]) = 1.0;
            for(size_t k=0; k<M; ++k){
                size_t N = std::get<1>(val[k]);
                size_t n = ((N&1)?((N+1)/2):(N/2+1));
                size_t m = ind%n;  
                ind = (ind-m)/n;
                std::get<0>(simulation_info[i]) *= (N&1)?(2*(m+1)):(2*m+1);
                std::get<1>(simulation_info[i]) *= weights[k][(N&1)?(2*m+1):(2*m)];
                std::get<2>(simulation_info[i])[k] = (N&1)?(2*m+1):(2*m);
            }
        }

        //sort the simulation parameters based on decreasing weights
        std::sort(simulation_info.begin(), simulation_info.end(), 
            []( const std::tuple<uint64_t, T, std::vector<size_t> >& a, 
                const std::tuple<uint64_t, T, std::vector<size_t> >& b) -> bool
                {
                    return std::get<1>(a) > std::get<1>(b);
                });


        std::vector<T> cumulative_weights;  cumulative_weights.reserve(simulation_info.size());
        T sum_weight = 0;
        for(size_t i = 0; i < simulation_info.size(); ++i)
        {
            sum_weight += std::get<1>(simulation_info[i]);
            cumulative_weights.push_back(sum_weight);

            if(sum_weight > 1.0-ctol)
            {
                break;
            }
        }

        //now rescale the weights for each of the blocks so that we are sampling correctly from the set we are including
        for(size_t i = 0; i < cumulative_weights.size(); ++i)
        {
            cumulative_weights[i] /= sum_weight;
        }
        std::vector<T> traj_weighting(cumulative_weights.size());

        std::mt19937 rng(seed);

        //generate how many of each of the trajectories we will run
        std::vector<size_t> n_to_run(cumulative_weights.size());     std::fill(n_to_run.begin(), n_to_run.end(), 0);

        size_t nall_trajectories = ntraj;
        if(!sample_blocks)
            {
            for(size_t i = 0;  i < n_to_run.size(); ++i)
            {
                n_to_run[i] = minntraj;
                traj_weighting[i] = std::get<1>(simulation_info[i])/(minntraj > 0 ? minntraj : 1);
            }

            nall_trajectories = n_to_run.size()*minntraj > ntraj ? n_to_run.size()*minntraj : ntraj;
            for(size_t tid = n_to_run.size()*minntraj; tid < ntraj; ++tid)
            {
                size_t index = static_cast<size_t>(std::max_element(traj_weighting.begin(), traj_weighting.end())-traj_weighting.begin());
                ++n_to_run[index];
                traj_weighting[index] = std::get<1>(simulation_info[index])/n_to_run[index];
            }
            
            sum_weight = 0;
            for(size_t i = 0;  i < n_to_run.size(); ++i)
            {
                if(n_to_run[i] > 0)
                {
                    sum_weight += std::get<1>(simulation_info[i]);
                }
            }

            for(size_t i = 0;  i < n_to_run.size(); ++i)
            {
                traj_weighting[i] /= sum_weight;
            }
        }
        else
        {
            std::uniform_real_distribution<T> dist(0, 1);
            for(size_t i = 0; i < ntraj; ++i)
            {
                T rand = dist(rng);

                std::ptrdiff_t ind = ((std::upper_bound(cumulative_weights.begin(), cumulative_weights.end(), rand)) - cumulative_weights.begin());
                ASSERT(ind >= 0 && ind < cumulative_weights.size(), "Invalid index found.");
                ++n_to_run[ind];
                
            }

            for(size_t index = 0; index < n_to_run.size(); ++index)
            {
                if(n_to_run[index] != 0)
                {
                    traj_weighting[index] = 1.0/ntraj;
                }
                else
                {
                    traj_weighting[index] = 0.0;
                }
            }
        }

        size_t trajs_run = 0;
        //now we can actually go ahead and run the dynamics
        std::cout << std::setprecision(14);

        #pragma omp parallel default(shared)
        {
            std::vector<std::vector<std::array<T, 2>>> vals_local(ops.size());
            for(size_t i = 0; i < vals_local.size(); ++i)
            {
                vals_local[i].resize(Nsteps+1);
                for(size_t j = 0; j < Nsteps+1; ++j)
                {
                    vals_local[i][j] = {{0,0}};
                }
            }

            #pragma omp for schedule(dynamic, 1)
            for(size_t bi = 0; bi < n_to_run.size(); ++bi)
            {        
                if(n_to_run[bi] != 0)
                {
                    std::vector<size_t> Ns(std::get<2>(simulation_info[bi]));    //Ns currently contains 2I
                    for(size_t j = 0; j < Ns.size(); ++j){++Ns[j];}             //so make it contain 2I+1

                    linalg::csr_matrix<complex_type> H;


                    size_t nhilb = system_setup<T>::nhilb(Ns);
                    system_setup<T>::construct(w, Adisc, Ns, H);

                    T weight = std::get<1>(simulation_info[bi]);
    
                    bool compute_variance = true;
                    T traj_weight = traj_weighting[bi];
                    size_t ntrajectories = n_to_run[bi];
                    if(n_to_run[bi]*2 >= nhilb)
                    {
                        compute_variance = false;
                        ntrajectories = nhilb/2;
                        traj_weight *= (1.0*n_to_run[bi])/ntrajectories;
                    }
                    //if we are only set to run a single trajectory then we will actually run to to ensure that we can get an estimate of the variance.
                    if(n_to_run[bi] == 1 && (n_to_run[bi]*2 < nhilb))
                    {
                        ntrajectories = 2;
                        traj_weight *= 0.5;
                    }

                    std::cerr << "Block " << bi << ":  Nhilb: " << nhilb << ", Weight: " << weight << std::endl;
                    std::cerr << "Trajectory: " << std::endl;

                    linalg::vector<complex_type> psi(nhilb);
                    linalg::vector<complex_type> psi_I(nhilb/2);

                    //set up the krylov subspace integrator
                    utils::krylov_integrator<complex_type> integ(krylov_dim, nhilb, krylov_tol);
                    complex_type ni(0, -1);

                    for(size_t i = 0; i < vals_local.size(); ++i)
                    {
                        for(size_t j = 0; j < Nsteps+1; ++j)
                        {
                            vals_local[i][j] = {{0,0}};
                        }
                    }
        
                    for(size_t traj = 0; traj < ntrajectories; ++traj)
                    {
                        std::cerr << "\t" << traj+1 << " of " << ntrajectories << std::endl;
                        if(n_to_run[bi]*2 < nhilb)
                        {
                            //setup the initial wavefunction
                            system_setup<T>::sample_SU_N(psi_I, rng);

                            for(size_t Ihilb = 0; Ihilb < nhilb/2; ++Ihilb)
                            {
                                psi(Ihilb) = psi0(0) * psi_I(Ihilb);
                                psi(Ihilb + nhilb/2) = psi0(1) * psi_I(Ihilb);
                            }
                        }
                        else
                        {
                            psi.fill_zeros();   
                            psi(traj) = psi0(0);
                            psi(traj+nhilb/2.0) = psi0(1);
                        }
                        
                        for(size_t iops = 0; iops < ops.size(); ++iops)
                        {
                            T ival = system_setup<T>::evaluate_system_expectation_value(psi, ops[iops]);
                            if(!compute_variance)
                            {
                                vals_local[iops][0][0] += ival/ntrajectories;
                            }
                            else
                            {
                                welford_variance_update(vals_local[iops][0], ival, traj+1);
                            }
                        }

                        for(size_t it = 0; it < Nsteps; ++it)
                        {
                            CALL_AND_HANDLE(integ(psi, dt, ni, H), "Failed to apply Krylov subspace integrator.");
                            for(size_t iops = 0; iops < ops.size(); ++iops)
                            {
                                T ival = system_setup<T>::evaluate_system_expectation_value(psi, ops[iops]);
                                if(!compute_variance)
                                {
                                    vals_local[iops][it+1][0] += ival/ntrajectories;
                                }
                                else
                                {
                                    welford_variance_update(vals_local[iops][it+1], ival, traj+1);
                                }
                            }
                        }
                        //vals local contains the means of the trajectories

                        for(size_t it = 0; it <= Nsteps; ++it)
                        {
                            for(size_t iops = 0; iops < ops.size(); ++iops)
                            {
                                if(compute_variance)
                                {
                                    vals_local[iops][it][1] = std::sqrt(vals_local[iops][it][1]/(ntrajectories-1.0))/std::sqrt(ntrajectories*1.0);
                                }
                            }
                        }
                    }

                    #pragma omp critical
                    {
                        trajs_run += n_to_run[bi];
                        std::cerr << "Trajectories completed: " << trajs_run << " of " << nall_trajectories << std::endl;
                        if(print_block_correlations)
                        {
                            std::cout << "Block: " << bi << std::endl;
                            for(size_t it = 0; it <= Nsteps; ++it)
                            {
                                T t = it * dt;
                                std::cout << t << " ";
                                for(size_t iops = 0; iops < ops.size(); ++iops)
                                {
                                    T ival = vals_local[iops][it][0]*ntrajectories;
                                    T sem = vals_local[iops][it][1]*ntrajectories;
                                    vals[iops][it] += ival*traj_weight;
                                    sems[iops][it] = std::sqrt(sems[iops][it]*sems[iops][it] + sem*sem*traj_weight*traj_weight);
                                    std::cout << ival/ntrajectories << " " << sem/ntrajectories << " ";
                                }
                                std::cout << std::endl;
                            }
                        }
                        else
                        {
                            for(size_t it = 0; it <= Nsteps; ++it)
                            {
                                for(size_t iops = 0; iops < ops.size(); ++iops)
                                {
                                    T ival = vals_local[iops][it][0]*ntrajectories;
                                    T sem = vals_local[iops][it][1]*ntrajectories;
                                    vals[iops][it] += ival*traj_weight;
                                    sems[iops][it] = std::sqrt(sems[iops][it]*sems[iops][it] + sem*sem*traj_weight*traj_weight);
                                }
                            }
                        }
                    }
                }
            }
        }
        
        bool output_to_standard_out = true;
        if(has_outfile)
        {
            std::ofstream ofs(ofname.c_str());
            ofs << std::setprecision(14);
            if(ofs.is_open())
            {   
                output_to_standard_out = false;
                ofs << "t ";
                for(size_t iops = 0; iops < ops.size(); ++iops)
                {
                    ofs << "op" << iops+1 << "sem(op" << iops+1 << ") ";
                }
                ofs << std::endl;
                for(size_t it = 0; it <= Nsteps; ++it)
                {
                    T t = it * dt;
                    ofs << t << " ";
                    for(size_t iops = 0; iops < ops.size(); ++iops)
                    {
                        ofs << vals[iops](it) << " " << sems[iops][it] << " ";
                    }
                    ofs << std::endl;
                }
            }
        }

        if(output_to_standard_out)
        {
            std::cout << "t ";
            for(size_t iops = 0; iops < ops.size(); ++iops)
            {
                std::cout << "op" << iops+1 << "sem(op" << iops+1 << ") ";
            }
            std::cout << std::endl;
            for(size_t it = 0; it <= Nsteps; ++it)
            {   
                T t = it * dt;
                std::cout << t << " ";
                for(size_t iops = 0; iops < ops.size(); ++iops)
                {
                    std::cout << vals[iops][it] << " " << sems[iops][it] << " ";
                }
                std::cout << std::endl;
            }
        }
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to run central spin dynamics.");
    }
}


#endif

