import numpy as np
import dataclasses
from typing import Mapping # For type hinting the dict later
#some defaults  for plotting:
MARKERSIZE=50 
LINEWIDTH=1.5
LINEWIDTHOFUNITCELLBDRY=2.5
DEFAULT_COLORS=['b','r','k']

MAX_ADJACENT_CELLS=1 # the number of cells taken into account 
"""
When considering a periodic structure we only take one unit cell into account. 
So you shouldn't have edges crossing multiple unit cell- so we limit to edges crossing a nigle unit cell.
"""
EIGENVALUE_TOLERANCE =1e-10  # Tolerance for eigenvalue convergence
# Define standard lattice vectors for hexagonal symmetry
v1=np.array([np.sqrt(3)/2,0.5])  # The standrad lattice vectors of triangualar lattice of graphene 
v2=np.array([np.sqrt(3)/2,-0.5])
reciprocal_constant= 4.0*np.pi/np.sqrt(3) #reciprocal lattice constant
k1=np.array([0.5,np.sqrt(3)/2])*reciprocal_constant  #the standrad reciprocal lattice vectors of triangualar lattice of graphene 
k2=np.array([0.5,-np.sqrt(3)/2])*reciprocal_constant
attemp_num=5 #the number of attempt to converge by jitter
K_POINT_DUAL=(1/3,1/3) # the K point if the lattice is the dual lattice
K_POINT_REG=(1/3,-1/3) # the K point if the lattice is regular 


@dataclasses.dataclass(frozen=True) # frozen=True makes instances immutable, like your MappingProxyType
class simulation_parameters: 
    a:  int= 5 # int, co-prime to b, a<b, pararmter for the twist
    b: int= 1# int, co-prime to a, a<b, pararmter for the twist
    unit_cell_radius_factor: int =3 # Scaling factor for the plotted unit cell radius, (positive  integer)
    unit_cell_flag: bool=True # a flag whether to plot a unit cell or the entire graph 
    interlayer_dist_threshold: float= 1.0 # the distance of neighbors between graphs( positive float)
    min_band: int= 1 #the index of minimal band (positive int)
    max_band: int=3 #the index of maximal band (positive int)
    num_of_points: int=30  #the number of smapleing points for the band (positive int)
    inter_graph_weight: float=1.0 # the weight in the laplacian of edges between sub-graphs (float)
    intra_graph_weight: float=1.0# the weight in the laplacian of edges in the graph  (float)
    k_min: float=-0.1 #the minimal k to plot in natural unit (float)
    k_max: float=0.1 #the maximal k to plot in natural unit (float)
    K_flag: bool= True #a flag to know if to plot around the K point or around the orgin 
