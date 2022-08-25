#ifndef HOMOGENEOUS_COUPLING_HPP
#define HOMOGENEOUS_COUPLING_HPP

template <typename T> 
class HomogeneousCoupling
{
protected:
    static inline void add_direct_sum_decompose(uint64_t nnuc, const std::vector<T>& in, std::vector<T>& out){
        bool zeros = std::all_of(in.begin(), in.end(), [](T i) { return fabs(i)<1e-14; }); 
        std::fill_n(out.begin(), out.size(), 0);
        if(zeros){out[nnuc-1] = 1;}
        else{
            for(uint64_t i=0; i<in.size()-1; ++i){
                uint64_t start = ((i+1) < nnuc) ? (nnuc-(i+1)) : ((i+1)-nnuc);
                uint64_t end = (i+nnuc);
                for(uint64_t j = start; j<end; j+=2){
                    out[j] += in[i];
                }
            }
            for(uint64_t i=0; i<in.size(); ++i){
                out[i] /= nnuc;
            }
        }
    }

public:
    //Construct the weight associated with each element of the direct sum representation 
    static inline void direct_sum_decompose(const std::vector<uint64_t>& Nnuc, std::vector<T>& res){
        //first we determine the maximum possible value of J that the total spin angular momenta can take
        uint64_t max_J = 0;
        for(auto nnuc : Nnuc){max_J += (nnuc-1);}

        std::vector<T> temp;

        //now we allocate a vector (and a temporary helper vector) so that it can hold enough integers to store the number of each of the terms present
        res.resize(max_J+1, 0);
        temp.resize(max_J+1, 0);

        for(auto nnuc : Nnuc){
            add_direct_sum_decompose(nnuc, res, temp);
            std::swap(res, temp);
        }
    }

};


#endif //HOMOGENEOUS_COUPLING_HPP//

