/*
 * LinearLoadModel.hpp
 *
 *  Created on: Oct 9, 2013
 *      Author: heenemo
 */

#ifndef LINEARLOADMODEL_HPP_
#define LINEARLOADMODEL_HPP_

#include "sgpp/distributedcombigrid/utils/LevelVector.hpp"
#include "sgpp/distributedcombigrid/utils/Types.hpp"
#include "sgpp/distributedcombigrid/loadmodel/LoadModel.hpp"

namespace combigrid {

class LinearLoadModel: public LoadModel {
 public:
  LinearLoadModel() = default;

  ~LinearLoadModel() = default;

  virtual real eval(const LevelVector& l) const;
};

} /* namespace combigrid */
#endif /* LINEARLOADMODEL_HPP_ */