#ifndef MOMENT_FITTING_HPP
#define MOMENT_FITTING_HPP

#include <vector>
#include <utility>
#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iostream>

#include <linalg/linalg.hpp>

extern "C" {
extern int dgesv_(int*, int*, double*, int*, int*, double*, int*, int*);
}

#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

class Moments{
public:

    static inline std::vector<std::pair<double, size_t> > fit(const linalg::vector<double>& Avec, size_t M, double tol = 1e-12)
    {
        std::vector<double> avec(Avec.size());
        for(size_t i = 0; i < Avec.size(); ++i)
        {
            avec[i] = Avec(i);
        }
        return fit(avec, M, tol);
    }

    //a function which takes in a vector of hyperfines Avec and fits a set of M hyperfines with various numbers of spins 
    //corresponding to each element of the set so that the Mth moment is correctly reproduced.
    static inline std::vector<std::pair<double, size_t> > fit(const std::vector<double>& Avec, size_t M, double tol = 1e-12){
        if(M > Avec.size()) throw std::runtime_error("Unable to solve the specified moment problem.  User asked for more points to be fit than there are values in the original distribution.");
        std::vector<std::pair<double, size_t> > Ares(M);

        //if the input vector size is the same size as M we are done.
        if(Avec.size() == M){for(size_t i=0; i<M; ++i){Ares[i] = std::make_pair(Avec[i], 1);}}
        else if(M == 1){
            double Amean = 0.0;
            for(size_t i=0; i<Avec.size(); ++i){
                Amean += Avec[i]*Avec[i];
            }
            Amean = sqrt(Amean/Avec.size());
            Ares[0] = std::make_pair(Amean, Avec.size());
        }
        else{

            //set up the arrays needed for constructing a first guess of the weights and hyperfines using a discrete stieltjes procedure
            linalg::vector<double> Ai(Avec.size());  
            for(size_t i =0; i < Avec.size(); ++i){Ai[i] = Avec[i];}

            linalg::vector<double> wi(Avec.size());  wi.fill_value(1.0);

            linalg::vector<double> Ar(M);            Ar.fill_zeros();
            linalg::vector<double> wr(M);            wr.fill_zeros();

            
            //now we perform the actual discrete stieltjes procedure
            discrete_stieltjes(wi.buffer(), Ai.buffer(), Avec.size(), wr.buffer(), Ar.buffer(), M);
            //and scale the weights so that they sum to the number of input hyperfines
            for(size_t i=0; i<M; ++i){wr[i]*=Avec.size();}

            //now that we have found a rough guess with non-integer weights we will now go ahead and find the optimal integer weights.  These weights will be selected so
            //that they have the correct total weight and so that the least squares fit of the hyperfines to these new weights will have the smallest error in the next M higher 
            //order moments.  
            linalg::vector<double> Aint(M);     Aint.fill_zeros();
            linalg::vector<uint64_t> Nint(M);   Nint.fill_zeros();
            CALL_AND_HANDLE(optimal_integer_weights(wr.buffer(), Ar.buffer(), M, Avec.size(), tol, Nint.buffer(), Aint.buffer()), "Failed to compute optimal integer weights.");

            for(size_t i=0; i<M; ++i){Ares[i] = std::make_pair(Aint[i], Nint[i]);}
        }
        return Ares;
    }


    static inline uint64_t count_set_bits(uint64_t v){
        #define CHAR_BIT 8
        v = v - ((v >> 1) & (uint64_t)~(uint64_t)0/3);
        v = (v & (uint64_t)~(uint64_t)0/15*3) + ((v >> 2) & (uint64_t)~(uint64_t)0/15*3);
        v = (v + (v >> 4)) & (uint64_t)~(uint64_t)0/255*15;   
        return (uint64_t)(v * ((uint64_t)~(uint64_t)0/255)) >> (sizeof(uint64_t) - 1) * CHAR_BIT;
    }
protected:
    //A function which computes the roots of a system of equations using newton's method.  This requires the user
    //to specify a function for computing the function and the transpose of its Jacobian.
    template <typename Func, typename ... Args>
    static inline void find_roots_newton(double* x, double * F, double * J, int64_t M, double tol, Func func, Args&&... args){
        double eps = func(x, F, J, M, std::forward<Args>(args)...);
        int mm = M; int INFO = 0; int NRHS=1;
        linalg::vector<int> IPIV(M);

        size_t niters = 0;
        while(eps > tol){
            dgesv_(&mm, &NRHS, J, &mm, IPIV.buffer(), F, &mm, &INFO);
            if(INFO != 0){throw std::runtime_error("Unable to find roots using newton's method. There was an issue when inverting the system of equations");}

            for(size_t k=0; k<M; ++k){x[k] -= F[k];}
            eps = func(x, F, J, M, std::forward<Args>(args)...);
            ++niters;
            if(niters == 1000 || std::isnan(eps))
            {
                throw std::runtime_error("failed to converge");
            }
        }

    }


    //a function for searching through all plausible integer number of spins that are consistent with the computed weights and selecting 
    //that which (upon refitting of the hyperfines) minimises the error in the M moments from (M - 2M) that it does not exactly fit (but 
    //which are exactly fit by the approximate scheme).
    static inline size_t optimal_integer_weights(double* wd, double* ad, int64_t M, int64_t N, double tol, size_t* Nint, double* Aint)
    {
        double* atemp = new double[M];
        size_t* ntemp = new size_t[M];
        size_t* nlower = new size_t[M];

        double* J = new double[M*M];    //the jacobian matrix needed for the newton's method
        double* F = new double[M];      //the difference between the exact and approximate moments

        double mindiff = -1;

        //generate the starting condition for our fitting to 
        int64_t ntot = 0;   
        for(size_t i=0; i<M; ++i){
            nlower[i] = static_cast<size_t>(wd[i]);      //this rounds all of the wd's to the integer smaller than them (assuming they are all positive)
            ntot += nlower[i];
        }

        size_t nvalid = 0;

        uint64_t nterms = (uint64_t(1) << M);
        //now we iterate over all possible 
        for(uint64_t i=0; i < nterms; ++i){
            //check to see if the total number of spins using this integer will be correct (equal to N)
            if(ntot + count_set_bits(i) == N){
                //create the state of interest 
                for(uint64_t k=0; k<M; ++k){
                    ntemp[k] = nlower[k] + ((i & (1 << k)) >> k); 
                    atemp[k] = ad[k];
                }

                bool attempt_compare = true;
                //now we perform newton's method to refit the hyperfines so that the first N moments are exactly reproduced
                try{find_roots_newton(atemp, F, J, M, tol, 
                    [](double * _x, double* _F, double* _J, int64_t _M, size_t* _N, double* _w, double* _a) -> double{
                        double abs_F = 0.0;
                        double abs_x = 0.0;
                        
                        for(int64_t _k=1; _k<_M+1; ++_k){
                            _F[_k-1] = 0.0;
                            for(size_t _i=0; _i<_M; ++_i){
                                double t1 = _N[_i]*pow(_x[_i],_k-1);
                                _F[_k-1] += t1*_x[_i];
                                _J[_i*_M+(_k-1)] = _k*t1;
                                _F[_k-1] -= _w[_i]*pow(_a[_i],_k);
                            }
                            abs_x += std::abs(_x[_k-1]);
                            abs_F += std::abs(_F[_k-1]);
                        }
                        return abs_F/abs_x;
                    }, ntemp, wd, ad);
                }
                catch(const std::exception & ex){attempt_compare = false;}

                if(attempt_compare)
                {
                    ++nvalid;
                    double mom_diffs = 0.0;
                    for(size_t k=M; k<2*M; ++k){
                        for(size_t _i=0; _i<M; ++_i){
                            mom_diffs += ntemp[_i]*pow(atemp[_i], k) - wd[_i]*pow(ad[_i], k);
                        }
                    }
                    mom_diffs = std::abs(mom_diffs);
                    if(!std::isnan(mom_diffs)){
                        //if M is large then we break on the first of these to give us a solution
                        if(mom_diffs < mindiff || mindiff < 0){
                            mindiff = mom_diffs;
                            for(uint64_t k=0; k<M; ++k){
                                Nint[k] = ntemp[k]; 
                                Aint[k] = atemp[k];
                            }
                        }
                    }
                }
            }
        }

        if(nvalid == 0){throw std::runtime_error("Failed to find a valid configuration of the spin-system.  The Newton refinement step likely failed to find an integer solution.  This happens for high order moments due to the ill-conditioning of the resultant Jacobian.");}

        delete[] atemp;
        delete[] ntemp;
        delete[] nlower;
        delete[] F;
        delete[] J;
        return nvalid;
    }

    //computes (a^2 + b^2)^(1/2) without overflow or underflow
    static inline double pythag(double a, double b){
        double absa = std::abs(a);
        double absb = std::abs(b);
        if(absa > absb) return absa*sqrt(1.0+(absb*absb)/(absa*absa));
        else return (absb == 0.0 ? 0.0 : absb*sqrt(1.0+(absa*absa)/(absb*absb)));
    }

    //computes the eigenvalues and eigenvectors of a real symmetric tridiagonal
    //matrix.  The d array contains the diagonal column and e the offdiagonals.
    //Initially the v matrix must be initialised to the identity matrix.
    //This is a slight variant of the routine in numerical recipes
    static inline void tqli(double * d, double * e, int n, double* z, int ldv){
        size_t nv = std::min(n, ldv);
	    int64_t m,l,iter,i,k;
        double s,r,p,g,f,dd,c,b;    
    
        for (i=2;i<=n;i++) e[i-2]=e[i-1]; /* Convenient to renumber the elements of e. */
        e[n-1]=0.0;
        for (l=1;l<=n;l++) {
            iter=0;
            do {
                for (m=l;m<=n-1;m++) { /* Look for a single small subdiagonal element to split the matrix. */
                    dd=fabs(d[m-1])+fabs(d[m]);
                    if (static_cast<double>(std::abs(e[m-1])+dd) == dd) break;
                }
                if (m != l) {
                    if (iter++ == 30) throw std::runtime_error("Too many iterations in tqli");
                    g=(d[l]-d[l-1])/(2.0*e[l-1]); /* Form shift. */
                    r=pythag(g,1.0);
                    g=d[m-1]-d[l-1]+e[l-1]/(g+SIGN(r,g)); /* This is dm - ks. */
                    s=c=1.0;
                    p=0.0;
                    for (i=m-1;i>=l;i--) { /* A plane rotation as in the original QL, followed by Givens */
                        f=s*e[i-1];          /* rotations to restore tridiagonal form.                     */
                        b=c*e[i-1];
                        e[i]=(r=pythag(f,g));
                        if (r == 0.0) { /* Recover from underflow. */
                            d[i] -= p;
                            e[m-1]=0.0;
                            break;
                        }
                        s=f/r;
                        c=g/r;
                        g=d[i]-p;
                        r=(d[i-1]-g)*s+2.0*c*b;
                        d[i]=g+(p=s*r);
                        g=c*r-b;
                        /* Next loop can be omitted if eigenvectors not wanted */
                        for (k=1;k<=nv;k++) { /* Form eigenvectors. */
                            f=z[(k-1)*n+i];
                            z[(k-1)*n+i]=s*z[(k-1)*n+i-1]+c*f;
                            z[(k-1)*n+i-1]=c*z[(k-1)*n+i-1]-s*f;
                        }
                    }
                    if (r == 0.0 && i >= l) continue;
                    d[l-1] -= p;
                    e[l-1]=g;
                    e[m-1]=0.0;
                }
            } while (m != l);
        }
        for(i=0; i<n; ++i){
            int64_t _k =i;
            for(int64_t j=i+1; j<n; ++j){
                if(d[j] < d[_k]) _k=j;
            }
            if(_k != i){
                double swp = d[_k];
                d[_k] = d[i];
                d[i] = swp;
                for(size_t _l = 0; _l<nv; ++_l){
                    swp = z[_l*n+_k];
                    z[_l*n+_k] = z[_l*n+i];
                    z[_l*n+i] = swp;
                }
            }
        }
    }
    
    //uses a discrete stieltjes procedure to construct an n-point contracted quadrature rule from an np-point 
    //primitive quadrature rule with the same (non-negative) weight function.
    static inline void discrete_stieltjes(double* w, double * x, size_t np, double * wi, double * xi, size_t m)
    {
        linalg::vector<double> p(m*np);
        linalg::vector<double> q(np);
        linalg::vector<double> v(m);
        double qq = 0.0;
    
        for(int64_t i=0; i<np; ++i){q[i] = sqrt(w[i]);}
        for(int64_t k=0; k<m; ++k){
            qq = 0.0;
            for(int64_t i=0; i<np; ++i){qq += q[i]*q[i];}
            qq = sqrt(qq);
            if (qq == 0.0){throw std::runtime_error("Failed to construct the contracted quadrature rule qq==0");}
            for(int64_t i=0; i<np; ++i){
                p[k*np+i] = q[i]/qq;
                q[i] = x[i]*p[k*np+i];
            }
            for(int64_t j=k; j>=0; --j){
                double pq = 0.0;
                for(int64_t i=0; i<np; ++i){pq += p[j*np+i]*q[i];}
                if(j == k){
                    v[k] = 0.0;
                    wi[k] = qq;
                    xi[k] = pq;
                }
                for(int64_t i=0; i<np; ++i){q[i] -= p[j*np+i]*pq;}
            }
        
        }
        double weight = w[0];
        v[0] = 1.0;
        int ldv = 1;

        try{tqli(xi, wi, m, v.buffer(), ldv);}
        catch(const std::exception& ex){
            std::cerr << ex.what() << std::endl;
            throw std::runtime_error("Failed to diagonalise three term recurrence relation");
        }
        for(size_t j=0; j<m; ++j){wi[j] = weight*weight*v[j]*v[j];}
    }

};



#endif  //MOMENT_FITTING_HPP//

