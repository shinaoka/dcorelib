# Copyright (c) 2013 Commissariat à l'énergie atomique et aux énergies alternatives (CEA)
# Copyright (c) 2013 Centre national de la recherche scientifique (CNRS)
# Copyright (c) 2019-2020 Simons Foundation
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You may obtain a copy of the License at
#     https:#www.gnu.org/licenses/gpl-3.0.txt
#
# Authors: Olivier Parcollet, Nils Wentzell
# Modified by Hiroshi Shinaoka
import os
  
_mpi_imported = "OMPI_COMM_WORLD_RANK" in os.environ or "PMI_RANK" in os.environ

def is_on():
    return _mpi_imported

if _mpi_imported:
    from ._mpi import *
else:
    from ._no_mpi import *

import os,sys,datetime
myprint_err = lambda x : sys.stderr.write("%s\n"%x)
myprint_out = lambda x : sys.stdout.write("%s\n"%x)

if rank==0 and _mpi_imported:
    myprint_err ("Starting run with %s MPI rank(s) at : %s"%(size,str(datetime.datetime.now())))

master =0

All_Nodes_report = False

def bcast(x, root = 0):
    return world.bcast(x, root = root)

def send(val, dest):
    world.send(val, dest = dest)
    #print "node ", rank, " has sent ", val, " to node ", dest

def recv(source = 0):
    r = world.recv(source = source)
    #print "node ", rank, " receives ", r, " from node ", source
    return r

def barrier() : 
    world.barrier()

# temporary patch
# need to change the code for all all_reduce to use this layer....
def all_reduce(WORLD, x, F) : 
    return world.allreduce(x)


Verbosity_Level_Report_Max = 1
def report(*x,**opt):
    """ print on the master, do nothing on the nodes """
    try : 
        if opt['Verbosity_Level'] >  Verbosity_Level_Report_Max : return
    except :
        pass
    if All_Nodes_report and rank!=0 :
          for y in x:
              open('report_node%s'%rank,'a').write(str(y) + '\n') #             print y
              
    if rank==0 :
        myprint,myflush = (myprint_err,sys.stderr.flush)  if 'stderr' in opt and opt['stderr'] else (myprint_out,sys.stdout.flush)   
        for y in x:
          myprint( y)  # open('report','a').write(str(y) + '\n') #             print y
          myflush() # be sure to flush the buffer!  
        
def slice_inf(imin,imax) : 
  j=(imax - imin + 1)//size
  i= imax - imin + 1 - size*j
  return imin + rank*(j+1)  if  rank<=i-1 else imin + rank*j + i

def slice_sup(imin,imax) :
  j=(imax - imin + 1)//size
  i= imax - imin + 1 - size*j;
  return imin + (rank+1)*(j+1) -1  if  rank<=i-1  else imin + (rank+1)*j  + i - 1

def slice_array(A) : 
    """Given an array A, it returns a VIEW of a slice over the first dim on the node"""
    imax = A.shape[0]-1
    return A[slice_inf(0,imax):slice_sup(0,imax)+1] # +1 due to the slice convention

def is_master_node(): return rank == master

HostNames = {}
def master_gets_host_names():
    """ find the host name of all the nodes """
    from socket import gethostname
    global HostNames
    HostNames[rank] = gethostname()
    if is_master_node() :
      for proc in range (1,size) :
        HostNames[proc],status = recv(proc)
    else:
      send(gethostname(),master)
    barrier()
    assert len(HostNames)==size," internal pb MPI module"
    
    if is_master_node() :
      print("Hostnames : ")
      for u,host in list(HostNames.items()) :
        print("Node %d  on machine %s"%(u,host))


