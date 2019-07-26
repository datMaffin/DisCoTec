#define BOOST_TEST_DYN_LINK
#include <mpi.h>
#include <vector>
#include <set>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/serialization/export.hpp>
#include<boost/test/unit_test.hpp>

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
//#include "sgpp/distributedcombigrid/fault_tolerance/LPOptimizationInterpolation.hpp"
#include "sgpp/distributedcombigrid/mpi_fault_simulator/MPI-FT.h"
#include "sgpp/distributedcombigrid/fault_tolerance/FaultCriterion.hpp"
#include "sgpp/distributedcombigrid/fault_tolerance/StaticFaults.hpp"
#include "sgpp/distributedcombigrid/fault_tolerance/WeibullFaults.hpp"

// include user specific task. this is the interface to your application
#include "test_helper.hpp"
#include "sgpp/distributedcombigrid/fault_tolerance/TaskExample.hpp"

#include "sgpp/distributedcombigrid/fault_tolerance/HelperFunctions.hpp"

//,using namespace combigrid;
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

// this is necessary for correct function of task serialization

BOOST_CLASS_EXPORT(TaskExample)
//#ifdef ENABLE_FT
BOOST_CLASS_EXPORT(StaticFaults)
BOOST_CLASS_EXPORT(WeibullFaults)
//#endif
BOOST_CLASS_EXPORT(FaultCriterion)


void check_faultTolerance(bool useCombine, bool useFG, double l0err, double l2err,size_t num_faults ) {
    //int Sim_FT_MPI_Init(int *argc, char ***argv);

   int size = useFG ? 2 : 7;
  BOOST_REQUIRE(TestHelper::checkNumMPIProcsAvailable(size));

  CommunicatorType comm = TestHelper::getComm(size);
  if (comm == MPI_COMM_NULL){ return; }

  combigrid::Stats::initialize();

  size_t ngroup = 2;
  size_t nprocs = 2;
    DimType dim = 2;
    IndexVector p = (1,2);
    LevelVector lmin(dim, 3);
    LevelVector lmax(dim, 6), leval(dim, 4);

    // choose dt according to CFL condition
    combigrid::real dt = 0.01;
    FaultsInfo faultsInfo;
    faultsInfo.numFaults_ = num_faults;
 
    size_t nsteps = 1;
    size_t ncombi = 1;
    std::vector<bool> boundary(dim, true);

  theMPISystem()->initWorldReusable(comm, ngroup, nprocs);

  WORLD_MANAGER_EXCLUSIVE_SECTION {
    ProcessGroupManagerContainer pgroups;
    for (size_t i = 0; i < ngroup; ++i) {
      int pgroupRootID(i);
      pgroups.emplace_back(std::make_shared<ProcessGroupManager>(pgroupRootID));
    }

    //std::unique_ptr<LoadModel> loadmodel = std::unique_ptr<LinearLoadModel>(new LinearLoadModel());
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

    //BOOST_REQUIRE(true); //if things go wrong weirdly, see where things go wrong

//#ifdef TIMING
  //  std::unique_ptr<LoadModel> loadmodel = std::unique_ptr<LearningLoadModel>(new LearningLoadModel(levels));
//#else // TIMING
    
//#endif //def TIMING


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
        std::cout << " NUM OF FAULTSS:       " << faultsInfo.numFaults_;
        faultCrit = new WeibullFaults(0.7, abs(faultsInfo.numFaults_), ncombi, true);
      }
      else{ //use predefined static number and timing of faults
        //if numFaults = 0 there are no faults
	std::cout << " NUM OF FAULTSS:  " << faultsInfo.numFaults_ << std::endl;
        faultCrit = new StaticFaults(faultsInfo);
      }
      Task* t = new TaskExample(dim, levels[i], boundary, coeffs[i],loadmodel, dt, nsteps, p, faultCrit);
      tasks.push_back(t);
      taskIDs.push_back( t->getID() );
    }

    /* create combi parameters */
    CombiParameters params(dim, lmin, lmax, boundary, levels, coeffs, taskIDs, ncombi, 1);
    params.setParallelization(p);

    /* create Manager with process groups */
    ProcessManager manager( pgroups, tasks, params, std::unique_ptr<LoadModel>(loadmodel));

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

  //simft::Sim_FT_MPI_Finalize();
    combigrid::Stats::finalize();
      MPI_Barrier(comm);

  //return 0;
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

