"""
Poisson Equation Solver with Dirichlet boundary condtion in 2D using DOLFINx
============================================================================

Problem statement:
    -∇²u = f        in Ω = [0,1] x [0,1]
     u  = uD        on ∂Ω                 

Let the exact solution:
    u_exact(x, y) = 1 + x^2 + 2*y^2. 
Then,
    f  = -6                          
    uD = 1 + x^2 + 2*y^2             

The discrete variational problem:
    Find uh in Vh, finite dimensional trial space for all v in V_hat, finite
    dimensional test space such that
            a(uh, v) = L(v) 
    where a(uh, v) = ∫ ∇uh . ∇v dx, L(v) = ∫ f v dx 

Method:
    - Domain discretized into an 8x8 grid of quadrilateral cells.
    - Solved with continuous Lagrange (Q1) finite elements on V.
    - Linear system assembled and solved directly via LU factorization (PETSc).

Verification:
    - uex is computed on a higher-order space V2 (Lagrange degree 2) so that
      it is a more accurate representation of the true solution, against
      which we measure the L2 error of uh.
    - error_max compares uh directly to uD's dof values (valid only because
      both live on the same function space V, so their dofs line up 1-to-1).

Visualization:
    - First plot: the raw mesh geometry (no solution data), just to see the
      discretization.
    - Second plot: the actual solution uh, using plot.vtk_mesh(V) so that
      each mesh point corresponds exactly to one degree of freedom of V,
      keeping uh.x.array aligned with the plotted points.
"""
from mpi4py import MPI 
from dolfinx import mesh, fem, default_scalar_type, plot
from dolfinx.fem.petsc import LinearProblem
import numpy 
import ufl
import pyvista

# Domain
domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, mesh.CellType.quadrilateral)

# Function space
V = fem.functionspace(domain, ("Lagrange", 1))

# Dirichlet boundary function
uD = fem.Function(V)
# Interpolation: Initiated to evaluate values of the exact solution
# at mesh points
uD.interpolate(lambda x: 1 + x[0]**2 + 2*x[1]**2 )

tdim = domain.topology.dim # dimension of the domain
fdim = tdim - 1 # facet dimension: outer boundary dimension
domain.topology.create_connectivity(fdim, tdim) # connecting fdim to tdim
boundary_facets = mesh.exterior_facet_indices(domain.topology) # getting boundary facets, edges in our case(2D)

# Boundary degrees of freedom
boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(uD, boundary_dofs) # Boundary condition

# Using common space V for trial/test function
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

f = fem.Constant(domain, default_scalar_type(-6)) # Source term f = -6.

# Variational formulation of the PDE
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx 

# Linear system
problem = LinearProblem(
    a,
    L,
    bcs=[bc],
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options_prefix="Poisson",
)
uh = problem.solve() # approximate solution uh

V2 = fem.functionspace(domain, ("Lagrange", 2)) # Exact solution space
uex = fem.Function(V2, name="u_exact")
uex.interpolate(lambda x: 1 + x[0] ** 2 + 2 * x[1] ** 2)

# L2 error of (uh - uex)
L2_error = fem.form(ufl.inner(uh - uex, uh - uex) * ufl.dx)
error_local = fem.assemble_scalar(L2_error) # error over the cells on the local process
error_L2 = numpy.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM)) # error over the domain

# Since uD and uh has the same dofs(they are in the same function space V), calculating max error in all dofs
error_max = numpy.max(numpy.abs(uD.x.array - uh.x.array))
vertex_max = domain.comm.allreduce(error_max, op=MPI.MAX)
if domain.comm.rank == 0: 
    print(f"Error_L2 : {error_L2:.2e}")
    print(f"Error_max : {error_max:.2e}")

# Mesh visualization
domain.topology.create_connectivity(tdim, tdim)
topology, cell_types, geometry = plot.vtk_mesh(domain, tdim) # converting to mesh format pyvista understands 
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True, style="surface")
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    print("Error plotting")


# Solution(uh) visualization
u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V) # creating mesh from function space V
u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry) # Grid
u_grid.point_data["u"] = uh.x.array.real # giving point data to grid of uh
u_grid.set_active_scalars("u")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, show_edges=True)
u_plotter.view_xy()
if not pyvista.OFF_SCREEN:
    u_plotter.show()
