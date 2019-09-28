#define BOOST_TEST_DYN_LINK
#include <mpi.h>
#include <vector>
#include <set>
#include <boost/test/unit_test.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/serialization/export.hpp>

// compulsory includes for basic functionality
#include "sgpp/distributedcombigrid/task/Task.hpp"
#include "sgpp/distributedcombigrid/utils/Types.hpp"
#include "sgpp/distributedcombigrid/combischeme/CombiMinMaxScheme.hpp"
#include "sgpp/distributedcombigrid/fullgrid/FullGrid.hpp"
#include "sgpp/distributedcombigrid/loadmodel/LinearLoadModel.hpp"
#include "sgpp/distributedcombigrid/loadmodel/LearningLoadModel.hpp"
#include "sgpp/distributedcombigrid/manager/CombiParameters.hpp"
#include "sgpp/distributedcombigrid/manager/ProcessGroupManager.hpp"
#include "sgpp/distributedcombigrid/manager/ProcessGroupWorker.hpp"
#include "sgpp/distributedcombigrid/manager/ProcessManager.hpp"
#include "sgpp/distributedcombigrid/fault_tolerance/LPOptimizationInterpolation.hpp"
#include "sgpp/distributedcombigrid/fault_tolerance/FaultCriterion.hpp"
#include "sgpp/distributedcombigrid/fault_tolerance/StaticFaults.hpp"
#include "sgpp/distributedcombigrid/fault_tolerance/WeibullFaults.hpp"
#include "test_helper.hpp"
#include "sgpp/distributedcombigrid/utils/Config.hpp"

using namespace combigrid;

/* functor for exact solution */
class TestFn {
 public:
  // function value
  double operator()(std::vector<double>& coords, double t) {
    double exponent = 0;
    for (DimType d = 0; d < coords.size(); ++d) {
      coords[d] = std::fmod(1.0 + std::fmod(coords[d] - t, 1.0), 1.0);
      exponent -= std::pow(coords[d] - 0.5, 2);
    }
    return std::exp(exponent * 100.0) * 2;
  }
};

/* simple task class to sequentialy solve the 2D advection equation with
 * periodic boundary conditions using the finite difference and explicit Euler
 * methods */
class TaskAdvectionFDM : public combigrid::Task {
 public:
  TaskAdvectionFDM(LevelVector& l, std::vector<bool>& boundary, real coeff, LoadModel* loadModel,
                   real dt, size_t nsteps)
      : Task(2, l, boundary, coeff, 
      loadModel), dt_(dt), nsteps_(nsteps) {}

  void init(CommunicatorType lcomm,
            std::vector<IndexVector> decomposition = std::vector<IndexVector>()) {
    // only use one process per group
    IndexVector p(getDim(), 1);

    dfg_ =
        new DistributedFullGrid<CombiDataType>(getDim(), getLevelVector(), lcomm, getBoundary(), p);
    phi_.resize(dfg_->getNrElements());

    for (IndexType li = 0; li < dfg_->getNrElements(); ++li) {
      std::vector<double> coords(getDim());
      dfg_->getCoordsGlobal(li, coords);

      double exponent = 0;
      for (DimType d = 0; d < getDim(); ++d) {
        exponent -= std::pow(coords[d] - 0.5, 2);
      }
      dfg_->getData()[li] = std::exp(exponent * 100.0) * 2;
    }
  }

  void run(CommunicatorType lcomm) {
    // velocity vector
    std::vector<CombiDataType> u(getDim());
    u[0] = 1;
    u[1] = 1;

    // gradient of phi
    std::vector<CombiDataType> dphi(getDim());

    IndexType l0 = dfg_->length(0);
    IndexType l1 = dfg_->length(1);
    double h0 = 1.0 / (double)l0;
    double h1 = 1.0 / (double)l1;

    for (size_t i = 0; i < nsteps_; ++i) {
      phi_.swap(dfg_->getElementVector());

      for (IndexType li = 0; li < dfg_->getNrElements(); ++li) {

        IndexVector ai(getDim());
        dfg_->getGlobalVectorIndex(li, ai);

        // west neighbor
        IndexVector wi = ai;
        wi[0] = (l0 + wi[0] - 1) % l0;
        IndexType lwi = dfg_->getGlobalLinearIndex(wi);

        // south neighbor
        IndexVector si = ai;
        si[1] = (l1 + si[1] - 1) % l1;
        IndexType lsi = dfg_->getGlobalLinearIndex(si);

        // calculate gradient of phi with backward differential quotient
        dphi[0] = (phi_[li] - phi_[lwi]) / h0;
        dphi[1] = (phi_[li] - phi_[lsi]) / h1;

        CombiDataType u_dot_dphi = u[0] * dphi[0] + u[1] * dphi[1];
        dfg_->getData()[li] = phi_[li] - u_dot_dphi * dt_;
      }
    }

    setFinished(true);
  }

  void getFullGrid(FullGrid<CombiDataType>& fg, RankType r, CommunicatorType lcomm, int n = 0) {
    dfg_->gatherFullGrid(fg, r);
  }

  DistributedFullGrid<CombiDataType>& getDistributedFullGrid(int n = 0) { return *dfg_; }

  void setZero() {}

  ~TaskAdvectionFDM() {
    if (dfg_ != NULL) delete dfg_;
  }

 protected:
  TaskAdvectionFDM() : dfg_(NULL) {}

 private:
  friend class boost::serialization::access;

  DistributedFullGrid<CombiDataType>* dfg_;
  real dt_;
  size_t nsteps_;
  std::vector<CombiDataType> phi_;

  template <class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    // ar& boost::serialization::make_nvp(
    // BOOST_PP_STRINGIZE(*this),boost::serialization::base_object<Task>(*this));
    ar& boost::serialization::base_object<Task>(*this);
    ar& dt_;
    ar& nsteps_;
  }
};

// this is necessary for correct function of task serialization
BOOST_CLASS_EXPORT(TaskAdvectionFDM)
BOOST_CLASS_EXPORT(StaticFaults)
BOOST_CLASS_EXPORT(WeibullFaults)

BOOST_CLASS_EXPORT(FaultCriterion)


void checkFtolerance(double l0err, double l2err, int nfaults) {
  
  int size = 7;
  BOOST_REQUIRE(TestHelper::checkNumMPIProcsAvailable(size));

  CommunicatorType comm = TestHelper::getComm(size);
  if (comm == MPI_COMM_NULL){ return; }

  combigrid::Stats::initialize();

  size_t ngroup = 6;
  size_t nprocs = 1;
  FaultsInfo faultsInfo;
  faultsInfo.numFaults_ = nfaults;

  theMPISystem()->initWorldReusable(comm, ngroup, nprocs);
   
  WORLD_MANAGER_EXCLUSIVE_SECTION {
    ProcessGroupManagerContainer pgroups;
    for (size_t i = 0; i < ngroup; ++i) {
      int pgroupRootID(boost::numeric_cast<int>(i));
      pgroups.emplace_back(std::make_shared<ProcessGroupManager>(pgroupRootID));
    }

    DimType dim = 2;
    LevelVector lmin(dim, 3);
    LevelVector lmax(dim, 6), leval(dim, 6);

    // choose dt according to CFL condition
    combigrid::real dt = 0.0001;
    size_t nsteps = 100;
    size_t ncombi = 100;
    std::vector<bool> boundary(dim, true);
  

    CombiMinMaxScheme combischeme(dim, lmin, lmax);
    combischeme.createAdaptiveCombischeme();
    combischeme.makeFaultTolerant();
    std::vector<LevelVector> levels = combischeme.getCombiSpaces();
    std::vector<combigrid::real> coeffs = combischeme.getCoeffs();

    BOOST_REQUIRE(true); //if things go wrong weirdly, see where things go wrong

    /* create load model */
#ifdef TIMING
    std::unique_ptr<LoadModel> loadmodel = std::unique_ptr<LearningLoadModel>(new LearningLoadModel(levels));
#else // TIMING
    std::unique_ptr<LoadModel> loadmodel = std::unique_ptr<LinearLoadModel>(new LinearLoadModel());
#endif //def TIMING

    /*IndexType checkProcs = 1;
    for (auto k : p)
      checkProcs *= k;
    assert(checkProcs == IndexType(nprocs));*/

    /* create Tasks */
   TaskContainer tasks;
    std::vector<int> taskIDs;
    for (size_t i = 0; i < levels.size(); i++) {
      //create FaultCriterion
      //FaultCriterion *faultCrit;
      //create fault criterion
      //if(faultsInfo.numFaults_ < 0){ //use random distributed faults
        //if numFaults is smallerthan 0 we use the absolute value
        //as lambda value for the weibull distribution
        //faultCrit = new WeibullFaults(0.7, abs(faultsInfo.numFaults_), ncombi, true);
      //}
      //else{ //use predefined static number and timing of faults
        //if numFaults = 0 there are no faults
        //faultCrit = new StaticFaults(faultsInfo);
      //}
     Task* t = new TaskAdvectionFDM(levels[i], boundary, coeffs[i], loadmodel.get(), dt, nsteps);
      tasks.push_back(t);
      taskIDs.push_back(t->getID());
    }

    /* create combi parameters */
    CombiParameters params(dim, lmin, lmax, boundary, levels, coeffs, taskIDs, ncombi, 1);
    //params.setParallelization(p);

    /* create Manager with process groups */
    ProcessManager manager( pgroups, tasks, params, std::move(loadmodel) );

    /* send combi parameters to workers */
    manager.updateCombiParameters();

    /* distribute task according to load model and start computation for
     * the first time */
    bool success = manager.runfirst();

    for (size_t i = 1; i < ncombi; ++i){
      if ( !success ) {
        std::cout << "failed group detected at combi iteration " << i-1<< std::endl;
//        manager.recover();

        std::vector<int> faultsID;

        //vector with pointers to managers of failed groups
        std::vector< ProcessGroupManagerID> groupFaults;
        manager.getGroupFaultIDs(faultsID, groupFaults);

        /* call optimization code to find new coefficients */
        const std::string prob_name = "interpolation based optimization";
        std::vector<int> redistributeFaultsID, recomputeFaultsID;
        manager.recomputeOptimumCoefficients(prob_name, faultsID,
                                             redistributeFaultsID, recomputeFaultsID);

        f//or ( auto id : redistributeFaultsID ) {
          //TaskAdvectionFDM* tmp = static_cast<TaskAdvectionFDM*>(manager.getTask(id));
          //tmp->setStepsTotal(i*nsteps);
        //}

        //for ( auto id : recomputeFaultsID ) {
         // TaskAdvectionFDM* tmp = static_cast<TaskAdvectionFDM*>(manager.getTask(id));
          //tmp->setStepsTotal((i-1)*nsteps);
        //}
        /* recover communicators*/
        bool failedRecovery = manager.recoverCommunicators(groupFaults);


        if(failedRecovery){
          //if the process groups could not be restored distribute tasks to other groups
          std::cout << "Redistribute groups \n";
          manager.redistribute(redistributeFaultsID);
        }
        else{
          //if process groups could be restored reinitialize restored process group (keep the original tasks)
          std::cout << "Reinitializing groups \n";
          manager.reInitializeGroup(groupFaults,recomputeFaultsID);
        }
        /* if some tasks have to be recomputed, do so
         * allowing recomputation reduces the overhead that would be needed
         * for finding a scheme that avoids all failed tasks*/
        if(!recomputeFaultsID.empty()){
          std::cout << "sending tasks for recompute \n";
          manager.recompute(recomputeFaultsID,failedRecovery,groupFaults); //toDO handle faults in recompute
        }
        std::cout << "updateing Combination Parameters \n";
        //needs to be after reInitialization!
        /* communicate new combination scheme*/
        manager.updateCombiParameters();

      }

      /* combine solution */
      manager.combine();
              
      if ( !success ){
        /* restore combischeme to its original state
         * and send new combiParameters to all surviving groups */
        manager.restoreCombischeme();
        manager.updateCombiParameters();
      }

      /* run tasks for next time interval */
      success = manager.runnext();
    }

    //manager.parallelEval( leval, filename, 0 );
// evaluate solution
    FullGrid<CombiDataType> fg_eval(dim, leval, boundary);
    manager.gridEval(fg_eval);

    // exact solution
    TestFn f;
    FullGrid<CombiDataType> fg_exact(dim, leval, boundary);
    fg_exact.createFullGrid();
    for (IndexType li = 0; li < fg_exact.getNrElements(); ++li) {
      std::vector<double> coords(dim);
      fg_exact.getCoords(li, coords);
      fg_exact.getData()[li] = f(coords, (double)((1 + ncombi) * nsteps) * dt);
    }


    // calculate error
    fg_exact.add(fg_eval, -1);
    printf("LP Norm: %f\n", fg_exact.getlpNorm(0));
    printf("LP Norm2: %f\n", fg_exact.getlpNorm(2));
    // results recorded previously
    BOOST_CHECK(fabs( fg_exact.getlpNorm(0) - l0err) < TestHelper::higherTolerance);
    BOOST_CHECK(fabs( fg_exact.getlpNorm(2) - l2err) < TestHelper::higherTolerance);

    /* send exit signal to workers in order to enable a clean program termination */
    manager.exit();
  }

else {
    ProcessGroupWorker pgroup;
    SignalType signal = -1;
    while (signal != EXIT) signal = pgroup.wait();
  }
  
  combigrid::Stats::finalize();
  MPI_Barrier(comm);
}

BOOST_AUTO_TEST_SUITE(ftolerance)

BOOST_AUTO_TEST_CASE(test_1, * boost::unit_test::tolerance(TestHelper::tolerance) * boost::unit_test::timeout(40)) {
  checkFtolerance(1.547297, 11.322462,1);
}


BOOST_AUTO_TEST_SUITE_END()
