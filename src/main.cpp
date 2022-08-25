#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <tuple>
#include <algorithm>
#include <fstream>

#include "central_spin.hpp"

int main(int argc, char* argv[]){
    if(argc < 2)
    {
        std::cerr << argv[0] << " " << "<input file>" << std::endl;
        return 1;
    }

    try
    {
        using complex_type = linalg::complex<double>;
        std::ifstream ifs(argv[1]);
        if(!ifs.is_open())
        {
            std::cerr << "Could not open input file." << std::endl;
            return 1;
        }
        using IObj = IOWRAPPER::input_base;

        IObj doc {};
        IOWRAPPER::parse_stream(doc, ifs);

        ASSERT(IOWRAPPER::has_member(doc, "system"), "Failed to read in system information.");
        ASSERT(IOWRAPPER::is_object(doc, "system"), "The system information is not an object.");

        std::string type;
        CALL_AND_HANDLE(IOWRAPPER::load<std::string>(doc["system"], "type", type), "Failed to load in the system type.");
        io::remove_whitespace_and_to_lower(type);

        if(type == std::string("centralspin"))
        {
            CALL_AND_RETHROW(central_spin<double>(doc));
        }
        else
        {
            RAISE_EXCEPTION("Failed to determine system type.");
        }
    
    }   
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        std::cerr << "Failed to perform spin dynamics using hyperfine moment-fitting method." << std::endl;
        return 1;
    }
    
}
