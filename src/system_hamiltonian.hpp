#ifndef SYSTEM_HAMILTONIAN_HPP
#define SYSTEM_HAMILTONIAN_HPP

#include <linalg/linalg.hpp>
#include <vector>
#include <array>
#include <random>

template <typename T> 
class system_setup
{
public:
    using complex_type = linalg::complex<T>;
public:
    static void sample_SU_N(linalg::vector<complex_type>& A, std::mt19937& rng)
    {
        std::normal_distribution<T> dist(0, 1);
        for(size_t i = 0; i < A.size(); ++i)
        {
            A(i) = complex_type(dist(rng), dist(rng));
        }

        T len = std::sqrt(linalg::real(linalg::dot_product(linalg::conj(A), A)));
        for(size_t i = 0; i < A.size(); ++i)
        {
            A(i) /= len;
        }
    }

    static size_t nhilb(const std::vector<size_t>& n)
    {
        size_t nhilb = 2;
        for(size_t i = 0; i<n.size(); ++i)
        {
            nhilb *= n[i];
        }
        return nhilb;
    }

    static T evaluate_system_expectation_value(const linalg::vector<complex_type>& psi, const linalg::matrix<complex_type>& op)
    {
        complex_type accum(0,0);
        size_t nshilb = op.size(0);
        size_t nIhilb = psi.size()/nshilb;

        for(size_t i = 0;  i<nshilb; ++i)
        {
            for(size_t k=0; k < nshilb; ++k)
            {   
                for(size_t j=0; j < nIhilb; ++j)
                {
                    accum += linalg::conj(psi(i*nIhilb+j))*op(i, k)*psi(k*nIhilb+j);
                }
            }
        }
        return linalg::real(accum);
    }

    template <typename U>
    static void construct(const T& w, const std::vector<T>& a, const std::vector<size_t>& n, linalg::csr_matrix<U>& H)
    {
        ASSERT(n.size() == a.size(), "a and n must have the same size.");
        size_t N = n.size();
        size_t nhilb = 2;
        size_t nIhilb = 1;

        //set up the striding arrays
        std::vector<size_t> stride(N);      std::fill(stride.begin(), stride.end(), 1);
        for(size_t i = 0; i<N; ++i)
        {
            nhilb *= n[i];
            nIhilb *= n[i];
        
            for(size_t j=0; j<i; ++j)
            {
                stride[j] *= n[i];
            }
        }

        //set up the values of the off-diagonal spin operators
        linalg::vector<linalg::vector<T>> m_Sms(N);
        linalg::vector<linalg::vector<T>> m_Szs(N);
        for(size_t i = 0; i < N; ++i)
        {
            if(n[i] > 1)
            {
                m_Sms[i].resize(n[i]-1);
                for(size_t j = 0; j < m_Sms[i].size(); ++j)
                {
                    T S = (n[i]-1.0) / 2.0;
    
                    T m = S - (j+1);
                    m_Sms[i][j] = sqrt(S*(S+1) - m*(m+1.0));
                }
            }

            m_Szs[i].resize(n[i]);
            for(size_t j = 0; j < m_Szs[i].size(); ++j)
            {
                T S = (n[i]-1.0) / 2.0;
                T m = S - j;
                m_Szs[i][j] = m;
            }
        }

        size_t nnz = nhilb;         //the diagonal terms.
        
        //now add on the hyperfine coupling terms
        for(size_t i = 0; i < N; ++i)
        {
            nnz += ((nhilb * (n[i]-1))/ n[i] );
        }

        CALL_AND_HANDLE(H.resize(nnz, nhilb, nhilb), "Failed to resize Hamiltonian matrix.");

        std::vector<size_t> state(N+1);     std::fill(state.begin(), state.end(), 0);
        std::vector<size_t> ldims(N+1);
        ldims[0] = 2;
        for(size_t i = 0; i < N; ++i)
        {
            ldims[i+1] = n[i];
        }

        auto rowptr = H.rowptr();
        auto colind = H.colind();
        auto buffer = H.buffer();

        size_t counter = 0;
        rowptr[0] = 0;
        for(size_t r = 0; r < nhilb; ++r)
        {
            //first attempt to add on the S- block
            if(state[0] == 1)
            {
                size_t c = r-nIhilb;

                //now we want to iterate over the hyperfines in reverse order as we want to add the elements with increasing column index
                for(size_t _spi = 0; _spi < N; ++_spi)
                {
                    size_t spi = (N - (_spi+1));
                    if(n[spi] > 1)
                    {
                        if(state[spi+1]+1 != n[spi])
                        {
                            T v = a[spi]/2.0 * m_Sms[spi][state[spi+1]];
                            buffer[counter] = v;
                            colind[counter] = c + stride[spi];
                            ++counter;
                        }
                    }
                }
            }

            //then the diagonals
            {
                T Sz = (state[0] == 0 ? 0.5 : -0.5);
                T v =  Sz * w;
                for(size_t spi = 0; spi < N; ++spi)
                {
                    T Iz = m_Szs[spi][state[spi+1]];
                    v += a[spi]*Sz*Iz;
                }
                buffer[counter] = v;
                colind[counter] = r;
                ++counter;
            }
            
            //and finally the S+ block
            if(state[0] == 0)
            {
                size_t c = r+nIhilb;

                //now we want to iterate over the hyperfines in the forward order as we want these to be sorted with increasing column index
                for(size_t spi = 0; spi < N; ++spi)
                {
                    if(n[spi] > 1)
                    {
                        if(state[spi+1] != 0 )
                        {
                            T v = a[spi]/2.0 * m_Sms[spi][state[spi+1]-1];
                            buffer[counter] = v;
                            colind[counter] = c - stride[spi];
                            ++counter;
                        }
                    }
                }
            }
            rowptr[r+1] = counter;
            advance_state(ldims, state);
        }
    }

    static bool advance_state(const std::vector<size_t>& n, std::vector<size_t>& state)
    {
        size_t counter = state.size() - 1;

        bool continue_updating = true;
        while(continue_updating)
        {
            ++state[counter];
            if(state[counter] == n[counter])
            {
                if(counter == 0)
                {
                    return true;
                }
                state[counter] = 0;
                counter = counter-1;
            }
            else
            {
                continue_updating = false;
            }
        }
        return false;
    }

public:

protected:

};

#endif

