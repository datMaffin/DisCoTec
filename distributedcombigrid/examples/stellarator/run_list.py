import os
from subprocess import call, check_output
import math
import sys
from ConfigParser import SafeConfigParser
import numpy as np

# global parameters
#mpicmd = "aprun"
procpernode = 24
submitJobs = True

#appdir = "./scaling5_test"
#executable = "advection-example-faults"
parfile = "ctparam"


# walltime has to be given in minutes
def writeJobscript( filename, jobname, nodes, walltime, cmd ):   
    # header
    PPN = procpernode
    header  = "#!/bin/bash\n"
    header += "##Name of job\n"
    header += "#PBS -N " + str(jobname) + "\n"
    header += "##number of nodes and procs per node\n"
    header += "#PBS -l nodes=" + str(nodes) + ":ppn=24\n"
    header += "##Maximum execution time\n"
    header += "#PBS -l walltime=" + walltime + "\n"
    
    if( nodes > 2731 ):
        header += "#PBS -q nolimit \n"
    
    # body
    body = "cd $PBS_O_WORKDIR \n"
    #body += "source ../xprt.bat \n"
    body += cmd + "\n"
    
    # write to file
    content = header + "\n" + body
    file = open(filename,'w')
    file.write(content)
    file.close()

def setctparam( parfile, lmin, lmax, leval, leval2, p, ngroups,\
                nprocs, ncombi, nsteps, dt ):
    # read parameter file
    parser = SafeConfigParser()
    parser.read( parfile )
       
    #update config
    parser.set( 'ct', 'lmin', '%s %s %s %s %s %s' % tuple(lmin) )
    parser.set( 'ct', 'lmax', '%s %s %s %s %s %s' % tuple(lmax) )
    parser.set( 'ct', 'leval', '%s %s %s %s %s %s' % tuple(leval) )
    parser.set( 'ct', 'leval2', '%s %s %s %s %s %s' % tuple(leval2) )
    parser.set( 'ct', 'p', '%s %s %s %s %s %s' % tuple(p)  )
    parser.set( 'manager', 'ngroup', str(ngroups) )
    parser.set( 'manager', 'nprocs', str(nprocs) )
    parser.set( 'ct', 'ncombi', str(ncombi)  )
    parser.set( 'application', 'nsteps', str(nsteps)  )
    parser.set( 'application', 'dt', str(dt)  )
    
    # set paths for plot files
    workdir = os.getcwd()
    plotfile1 = workdir + '/ginstance/plot.dat'
    plotfile2 = workdir + '/ginstance/plot2.dat'
    parser.set( 'ct', 'fg_file_path', plotfile1 )
    parser.set( 'ct', 'fg_file_path2', plotfile2 )
    
    cfgfile = open( parfile,'w')
    parser.write(cfgfile)
    cfgfile.close()


def createExperiment( basename, lmin, lmax, leval, leval2, \
                      p, ngroup, nprocs, ncombi, nsteps, dt, walltime ):   
    # make file name
    nsteps_total = ncombi*nsteps
         
    dirname = basename 
    dirname += "_" + '%s%s%s%s%s%s' % tuple(lmin)
    dirname += "_" + '%s%s%s%s%s%s' % tuple(lmax)
    dirname += "_" + str(nsteps_total)
    dirname += "_" + str(nsteps)
    dirname += "_" + str(dt)
     
    print dirname
        
    # create dir, copy files and go into dir
    path = "./" + dirname 
    call( ['mkdir', path] )
    call( ['cp', '-r', 'template', path ] )
    call( ['cp', 'manager', path ] )
    call( ['cp', 'preproc.py', path ] )
    call( ['cp', 'run.sh', path ] )
    call( ['cp', './ctparam', path ] )
    os.chdir( path )
    
    #write ctparam
    setctparam( parfile, lmin, lmax, leval, leval2, p, ngroup,\
                nprocs, ncombi, nsteps, dt )
    
    # start job
    if submitJobs:
        # write jobscript etc
        jobname = dirname
        nodes = int( np.ceil(ngroup*nprocs / float(procpernode)) + 1) 
        cmd = "./run.sh"
        writeJobscript( "./job.sub", jobname, nodes, walltime, cmd )  
        
        #call( ["qsub", "./job.sub"] )
    else:
        call( ["./run.sh"] )
    
    # leave dir
    os.chdir( '..' )


if __name__ == "__main__":
    #read filename from args
    filename = sys.argv[1]
    
    #loop over lines in file
    for line in open(filename):
        par = line.split()
        
        # skip empty lines
        if len(par) == 0:
            continue
        
        basename    = par[0]
        dim         = par[1] 
        
        lmin        = [ int(x) for x in par[2:8] ]
        lmax        = [ int(x) for x in par[8:14] ]
        leval       = [ int(x) for x in par[14:20] ]
        leval2      = [ int(x) for x in par[20:26] ]
        p           = [ int(x) for x in par[26:32] ]
        
        ngroup      = int(par[32])
        nprocs      = int(par[33])     
        
        ncombi      = int(par[34])
        nsteps      = int(par[35])
        dt          = float(par[36])
        
        walltime_minutes    = int(par[37])
        
        walltime_hours = walltime_minutes / 60
        walltime_minutes = walltime_minutes % 60
        
        walltime=  str(walltime_hours) + ":" + str(walltime_minutes) + ":00"
        
        print basename, lmin, lmax, leval, leval2, p, ngroup,\
                nprocs, ncombi, nsteps, dt
        
        createExperiment( basename, lmin, lmax, leval, leval2, \
                          p, ngroup, nprocs, ncombi, nsteps, dt, walltime )