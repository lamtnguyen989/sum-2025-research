/* deal.II imports */
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/numerics/derivative_approximation.h>
#include <deal.II/meshworker/mesh_loop.h>

/* General cpp headers */
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cmath>

using namespace dealii;

template <int dim>
class InitialCondition
{

};

template <int dim>
class InviscidBurgers
{
    public:
        InviscidBurgers();
        void run();
    private:
        void make_grid();
        void setup_system();
        void assemble_system();
        void solve();
    
    // Systems components
    Triangulation<dim+1>    triangulation;
    FE_DGQ<dim+1>           fe;
    DoFHandler<dim+1>       dof_handler;

    // System parameters
    const double x_min, x_max;
    const_double t_min, t_max;
    unsigned int space_cells;
    unsigned_int time_cells;
};

