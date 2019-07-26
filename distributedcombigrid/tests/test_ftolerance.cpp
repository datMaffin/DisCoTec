#define BOOST_TEST_DYN_LINK
#include "mpi.h"
#include <vector>
#include <set>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/serialization/export.hpp>

// compulsory includes for basic functionality
#include "sgpp/distributedcombigrid/task/Task.hpp"
#include "sgpp/distributedcombigrid/utils/Types.hpp"
#include "sgpp/distributedcombigrid/combischeme/CombiMinMaxScheme.hpp"
#include "sgpp/distributedcombigrid/fullgrid/FullGrid.hpp"
#include "sgpp/distributedcombigrid/loadmodel/LinearLoadModel.hpp"
#include "sgpp/distributedcombigrid/manager/CombiParameters.hpp"
#include "sgpp/distributedcombigrid/manager/ProcessGroupManager.hpp"
#include "sgpp/distributedcombigrid/manager/ProcessGroupWorker.hpp"
#include "sgpp/distributedcombigrid/manager/ProcessManager.hpp"
#include "sgpp/distributedcombigrid/fault_tolerance/LPOptimizationInterpolation.hpp"
#include "sgpp/distributedcombigrid/mpi_fault_simulator/MPI-FT.h"
#include "sgpp/distributedcombigrid/fault_tolerance/FaultCriterion.hpp"
#include "sgpp/distributedcombigrid/fault_tolerance/StaticFaults.hpp"
#include "sgpp/distributedcombigrid/fault_tolerance/WeibullFaults.hpp"

// include user specific task. this is the interface to your application
#include "TaskExample.hpp"

#include "HelperFunctions.hpp"
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

using namespace combigrid;

// this is necessary for correct function of task serialization
BOOST_CLASS_EXPORT(TaskExample)
BOOST_CLASS_EXPORT(StaticFaults)
BOOST_CLASS_EXPORT(WeibullFaults)

BOOST_CLASS_EXPORT(FaultCriterion)


void check_faultTolerance(bool useCombine, bool useFG, double l0err, double l2err,size_t num_faults ) {
  
  /* number of process groups and number of processes per group */
  size_t ngroup = 2;
  size_t nprocs = 2;

  DimType dim = 2;
  LevelVector lmin(dim,3), lmax(dim,6), leval(dim,4);
  IndexVector p(1,2);
  size_t ncombi = 1;
  size_t nsteps = 1;
  combigrid::real dt = 0.01;
  FaultsInfo faultsInfo = num_faults;

  std::vector<bool> boundary(dim,true);

  theMPISystem()->init( ngroup, nprocs );

  // manager code
  if ( theMPISystem()->getWorldRank() == theMPISystem()->getManagerRankWorld() ) {
    /* create an abstraction of the process groups for the manager's view
     * a pgroup is identified by the ID in gcomm
     */
    ProcessGroupManagerContainer pgroups;
    for (size_t i = 0; i < ngroup; ++i) {
      int pgroupRootID(i);
      pgroups.emplace_back(
          std::make_shared< ProcessGroupManager > ( pgroupRootID )
                          );
    }


    /* create load model */
    LoadModel* loadmodel = new LinearLoadModel();

    IndexType checkProcs = 1;
    for (auto k : p)
      checkProcs *= k;
    assert(checkProcs == IndexType(nprocs));

    CombiMinMaxScheme combischeme(dim, lmin, lmax);
    combischeme.createAdaptiveCombischeme();
    combischeme.makeFaultTolerant();
    std::vector<LevelVector> levels = combischeme.getCombiSpaces();
    std::vector<combigrid::real> coeffs = combischeme.getCoeffs();

    /* print info of the combination scheme */
    std::cout << "CombiScheme: " << std::endl;
    std::cout << combischeme;

    /* create Tasks */
    TaskContainer tasks;
    std::vector<int> taskIDs;

    for (size_t i = 0; i < levels.size(); i++) {
      //create FaultCriterion
      FaultCriterion *faultCrit;
      //create fault criterion
      if(faultsInfo.numFaults_ < 0){ //use random distributed faults
        //if numFaults is smallerthan 0 we use the absolute value
        //as lambda value for the weibull distribution
        faultCrit = new WeibullFaults(0.7, abs(faultsInfo.numFaults_), ncombi, true);
      }
      else{ //use predefined static number and timing of faults
        //if numFaults = 0 there are no faults
        faultCrit = new StaticFaults(faultsInfo);
      }
      Task* t = new TaskAdvectionFDM(levels[i], boundary, coeffs[i], loadmodel.get(), dt, nsteps);
      tasks.push_back(t);
      taskIDs.push_back( t->getID() );
    }

    /* create combi parameters */
    CombiParameters params(dim, lmin, lmax, boundary, levels, coeffs, taskIDs, ncombi, 1);
    params.setParallelization(p);

    /* create Manager with process groups */
    ProcessManager manager( pgroups, tasks, params );

    /* send combi parameters to workers */
    manager.updateCombiParameters();

    /* distribute task according to load model and start computation for
     * the first time */
    bool success = manager.runfirst();

    for (size_t i = 0; i < ncombi; ++i){

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

        for ( auto id : redistributeFaultsID ) {
          TaskExample* tmp = static_cast<TaskExample*>(manager.getTask(id));
          tmp->setStepsTotal(i*nsteps);
        }

        for ( auto id : recomputeFaultsID ) {
          TaskExample* tmp = static_cast<TaskExample*>(manager.getTask(id));
          tmp->setStepsTotal((i-1)*nsteps);
        }
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

    std::string filename("out/solution_" + std::to_string(ncombi) + ".dat" );
    manager.parallelEval( leval, filename, 0 );
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
    BOOST_CHECK(abs( fg_exact.getlpNorm(0) - l0err) < TestHelper::higherTolerance);
    BOOST_CHECK(abs( fg_exact.getlpNorm(2) - l2err) < TestHelper::higherTolerance);
    /* send exit signal to workers in order to enable a clean program termination */
    manager.exit();
  }

// worker code
  else {
    // create abstraction of the process group from the worker's view
    ProcessGroupWorker pgroup;

    // wait for instructions from manager
    SignalType signal = -1;
    while (signal != EXIT)
      signal = pgroup.wait();
  }

  if( ENABLE_FT ){
    WORLD_MANAGER_EXCLUSIVE_SECTION{
      std::cout << "Program finished successfully" << std::endl;
      std::cout << "To avoid problems with hanging killed processes, we exit with "
                << "MPI_Abort()" << std::endl;
      MPI_Abort( MPI_COMM_WORLD, 0 );
    }
  }

  return 0;
}
	
BOOST_AUTO_TEST_SUITE(ftolerance)

BOOST_AUTO_TEST_CASE(test_1, * boost::unit_test::tolerance(TestHelper::tolerance) * boost::unit_test::timeout(80)) {
  // use recombination

  check_faultTolerance(true, false, 2.977406, 42.028659,1);
}

/* BOOST_AUTO_TEST_CASE(test_2, * boost::unit_test::tolerance(TestHelper::tolerance) * boost::unit_test::timeout(60)) {
  // don't use recombination
  check_faultTolerance(false, false, 1.65104, 12.46828,100);
}

BOOST_AUTO_TEST_CASE(test_3, * boost::unit_test::tolerance(TestHelper::tolerance) * boost::unit_test::timeout(80)) {
  // calculate solution on fullgrid
  check_faultTolerance(false, true, 1.51188, 10.97143,100);
  MPI_Barrier(MPI_COMM_WORLD);
}

BOOST_AUTO_TEST_CASE(test_4, * boost::unit_test::tolerance(TestHelper::tolerance) * boost::unit_test::timeout(40)) {
  // use recombination
  check_faultTolerance(true, false, 0.083211, 0.473448,0);
}

BOOST_AUTO_TEST_CASE(test_5, * boost::unit_test::tolerance(TestHelper::tolerance) * boost::unit_test::timeout(60)) {
  // don't use recombination
  check_faultTolerance(false, false, 0.083211, 0.473448,0);
}

BOOST_AUTO_TEST_CASE(test_6, * boost::unit_test::tolerance(TestHelper::tolerance) * boost::unit_test::timeout(80)) {
  // calculate solution on fullgrid
  check_faultTolerance(false, true, 0.060058, 0.347316,0);
  MPI_Barrier(MPI_COMM_WORLD);

}*/

BOOST_AUTO_TEST_SUITE_END()

