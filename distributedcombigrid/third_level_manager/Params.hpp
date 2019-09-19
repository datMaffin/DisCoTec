#include <boost/property_tree/ini_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/algorithm/string.hpp>
#include <string>
#include "sgpp/distributedcombigrid/third_level/NetworkUtils.hpp"

class Params
{
  private:
    u_int port_           = 0;
    u_int numSystems_     = 0;
    bool  loadedFromFile_ = false;

  public:
    Params();

    void loadFromFile(const std::string& filename);

    bool  areLoaded()     const;
    u_int getPort()       const;
    u_int getNumSystems() const;
};
