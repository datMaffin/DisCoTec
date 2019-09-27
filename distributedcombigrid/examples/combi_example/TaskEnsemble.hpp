/*
 * TaskExample.hpp
 *
 *  Created on: Sep 25, 2015
 *      Author: heenemo
 */

#ifndef TASKEXAMPLE_HPP_
#define TASKEXAMPLE_HPP_

#include "sgpp/distributedcombigrid/fullgrid/DistributedFullGridEnsemble.hpp"
#include "sgpp/distributedcombigrid/task/Task.hpp"

namespace combigrid {

class TaskExample : public Task {
 public:
  /* if the constructor of the base task class is not sufficient we can provide an
   * own implementation. here, we add dt, nsteps, and p as a new parameters.
   */
  TaskExample(DimType dim, LevelVector& l, std::vector<bool>& boundary, real coeff,
              LoadModel* loadModel, real dt, size_t nsteps, IndexVector p = IndexVector(0),
              FaultCriterion* faultCrit = (new StaticFaults({0, IndexVector(0), IndexVector(0)})))
      : Task(dim, l, boundary, coeff, loadModel, faultCrit),
        dt_(dt),
        nsteps_(nsteps),
        p_(p),
        initialized_(false),
        stepsTotal_(0),
        dfgEnsemble_(nullptr) {}

  void init(CommunicatorType lcomm,
            std::vector<IndexVector> decomposition = std::vector<IndexVector>()) {
    assert(!initialized_);
    assert(dfgEnsemble_ == nullptr);

    int lrank;
    MPI_Comm_rank(lcomm, &lrank);

    /* create distributed full grid. we try to find a balanced ratio between
     * the number of grid points and the number of processes per dimension
     * by this very simple algorithm. to keep things simple we require powers
     * of two for the number of processes here. */
    int np;
    MPI_Comm_size(lcomm, &np);

    // check if power of two
    if (!((np > 0) && ((np & (~np + 1)) == np)))
      assert(false && "number of processes not power of two");

    DimType dim = this->getDim();
    IndexVector p(dim, 1);
    const LevelVector& l = this->getLevelVector();

    if (p_.size() == 0) {
      // compute domain decomposition
      IndexType prod_p(1);

      while (prod_p != static_cast<IndexType>(np)) {
        DimType dimMaxRatio = 0;
        real maxRatio = 0.0;

        for (DimType k = 0; k < dim; ++k) {
          real ratio = std::pow(2.0, l[k]) / p[k];

          if (ratio > maxRatio) {
            maxRatio = ratio;
            dimMaxRatio = k;
          }
        }

        p[dimMaxRatio] *= 2;
        prod_p = 1;

        for (DimType k = 0; k < dim; ++k) prod_p *= p[k];
      }
    } else {
      p = p_;
    }

    if (lrank == 0) {
      std::cout << "init task " << this->getID() << " with l = " << this->getLevelVector()
                << " and p = " << p << std::endl;
    }

    // create local subgrid on each process
    dfgEnsemble_ = new DFGEnsemble(dim, l, lcomm, this->getBoundary(), p);

    /* loop over local subgrids and set initial values */
    for (int i = 0; i< dfgEnsemble_->getNumFullGrids(); ++i){
      auto& dfg = dfgEnsemble_->getDFG(i);
      std::vector<CombiDataType>& elements = dfg.getElementVector();

      for (size_t i = 0; i < elements.size(); ++i) {
        IndexType globalLinearIndex = dfg.getGlobalLinearIndex(i);
        std::vector<real> globalCoords(dim);
        dfg.getCoordsGlobal(globalLinearIndex, globalCoords);
        elements[i] = TaskExample::myfunction(globalCoords, 0.0);
      }
    }

    initialized_ = true;
  }

  /* this is were the application code kicks in and all the magic happens.
   * do whatever you have to do, but make sure that your application uses
   * only lcomm or a subset of it as communicator.
   * important: don't forget to set the isFinished flag at the end of the computation.
   */
  void run(CommunicatorType lcomm) {
    assert(initialized_);

    // TODO if your Example uses another data structure, you need to copy
    // the data from elements to that data structure

    /* pseudo timestepping to demonstrate the behaviour of your typical
     * time-dependent simulation problem. */
    // TODO replace by your time time stepping algorithm

    for (size_t step = stepsTotal_; step < stepsTotal_ + nsteps_; ++step) {
      real time = step * dt_;

      for (int i = 0; i< dfgEnsemble_->getNumFullGrids(); ++i){
        auto& dfg = dfgEnsemble_->getDFG(i);
        std::vector<CombiDataType>& elements = dfg.getElementVector();

        for (size_t i = 0; i < elements.size(); ++i) {
          IndexType globalLinearIndex = dfg.getGlobalLinearIndex(i);
          std::vector<real> globalCoords(this->getDim());
          dfg.getCoordsGlobal(globalLinearIndex, globalCoords);
          elements[i] = TaskExample::myfunction(globalCoords, time);
        }
      }

      MPI_Barrier(lcomm);
    }

    stepsTotal_ += nsteps_;

    // TODO if your Example uses another data structure, you need to copy
    // the data from that data structure to elements/dfg_

    this->setFinished(true);
  }

  DistributedFullGrid<CombiDataType>& getDistributedFullGrid(int n = 0) { return dfgEnsemble_->getDFG(n); }

  /* this function evaluates the combination solution on a given full grid.
   * here, a full grid representation of your task's solution has to be created
   * on the process of lcomm with the rank r.
   * typically this would require gathering your (in whatever way) distributed
   * solution on one process and then converting it to the full grid representation.
   * the DistributedFullGrid class offers a convenient function to do this.
   */
  void getFullGrid(FullGrid<CombiDataType>& fg, RankType r, CommunicatorType lcomm, int n = 0) {
    assert(fg.getLevels() == getDistributedFullGrid(n).getLevels());

    getDistributedFullGrid(n).gatherFullGrid(fg, r);
  }

  // return the number of grids; here it is the number of grids in ensemble
  size_t getNumGrids() override { return powerOfTwo[dim_]; }

  static real myfunction(std::vector<real>& coords, real t) {
    real u = std::cos(M_PI * t);

    for (size_t d = 0; d < coords.size(); ++d) u *= std::cos(2.0 * M_PI * coords[d]);

    return u;

    /*
    double res = 1.0;
    for (size_t i = 0; i < coords.size(); ++i) {
      res *= -4.0 * coords[i] * (coords[i] - 1);
    }


    return res;
    */
  }

  void setZero() {}

  ~TaskExample() {
    if (dfgEnsemble_ != nullptr) delete dfgEnsemble_;
  }

 protected:
  /* if there are local variables that have to be initialized at construction
   * you have to do it here. the worker processes will create the task using
   * this constructor before overwriting the variables that are set by the
   * manager. here we need to set the initialized variable to make sure it is
   * set to false. */
  TaskExample() : initialized_(false), stepsTotal_(1), dfgEnsemble_(nullptr) {}

 private:
  friend class boost::serialization::access;

  // new variables that are set by manager. need to be added to serialize
  real dt_;
  size_t nsteps_;
  IndexVector p_;

  // pure local variables that exist only on the worker processes
  bool initialized_;
  size_t stepsTotal_;
  // DistributedFullGrid<CombiDataType>* dfg_;
  DFGEnsemble* dfgEnsemble_;

  /**
   * The serialize function has to be extended by the new member variables.
   * However this concerns only member variables that need to be exchanged
   * between manager and workers. We do not need to add "local" member variables
   * that are only needed on either manager or worker processes.
   * For serialization of the parent class members, the class must be
   * registered with the BOOST_CLASS_EXPORT macro.
   */
  template <class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    // handles serialization of base class
    ar& boost::serialization::base_object<Task>(*this);

    // add our new variables
    ar& dt_;
    ar& nsteps_;
    ar& p_;
  }
};

}  // namespace combigrid

#endif /* TASKEXAMPLE_HPP_ */
