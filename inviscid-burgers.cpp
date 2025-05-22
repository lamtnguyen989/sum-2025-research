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

/* 
    Define the initial condition as u(x,0) = sin(pi * x)
*/
template <int dim>
class InitialCondition : public Function<dim>
{
    public:
        InitialCondition() : Function<dim>() {};
        virtual void value(const Point<dim> &p, const unsigned int component = 0) const override
        {
            return std::sin(numbers::PI * p[0]);
        }
};

/*
    Define the Dirichlet Boundary Condition on spatial interval [a,b] as u(a,t) = u(b,t) = 0
*/
template <int dim>
class BoundaryCondition : public Function<dim>
{
    public:
        BoundaryCondition() : Function<dim>() {};
        virtual void value(const Point<dim> &p, const unsigned int component) const override
        {
            return 0.0;
        }
};


/* ----------------------------------------------------------------------------------------------- */
template <int dim>
class InviscidBurgersDG
{
    public:
        InviscidBurgersDG(const unsigned int deg);
        void run();
    private:
        void make_grid();
        void setup_system();
        void assemble_system();
        void solve();
        double burgers_flux(double u);
        double numerical_flux (double u_plus, double u_minus, double lambda);
    
    // Systems components
    Triangulation<dim+1>    triangulation;
    FE_DGQ<dim+1>           fe;
    DoFHandler<dim+1>       dof_handler;

    // System parameters
    const double x_min, x_max;      // 1D spatial interval
    const double t_min, t_max;      // Time evolution
    unsigned int spatial_cells;     // Number of cells in spatial domain (most likely used as parameter for Grid's repetitions)
    unsigned int time_cells;        // Number of cells for spatial direction (again parameter for Grid's repetitions)
    unsigned int degree;            // degree of the solving system
    double current_time;            // Time-tracking (hopefully can use this for intermediate solutions processing)
};


template <int dim>
InviscidBurgersDG<dim>::InviscidBurgersDG(const unsigned int deg):
    degree(deg),
    fe(degree),
    dof_handler(triangulation),
    x_min = -1.0 , x_max = 1.0,
    t_min = 0.0 , t_max = 1.0,
    spatial_cells = 64,
    time_cells = 32,
    current_time = 0.0
{}


// GridGenerator::subdivided_hyper_rectangle
template <int dim>
void InviscidBurgersDG<dim>::make_grid()
{
    // Initializing repetiton in space-time direction (2nd argument)
    std::vector<unsigned int> repetitions(dim + 1);
    repetitions[0] = spatial_cells;
    repetitions[1] = time_cells;

    // Diagonally opposite points needed to define the grid (3rd & 4th argument)
    Point<dim+1> p1;
    p1[0] = x_min;
    p1[1] = t_min;

    Point<dim+1> p2;
    p2[0] = x_max;
    p2[1] = t_max;

    // Setting up subdivided_hyper_rectangle grid 
    bool colorized = false; // Colorized param (default=false)
    GridGenerator::subdivided_hyper_rectangle(triangulation, repetitions, p1, p2, colorized);
}

template <int dim>
void InviscidBurgersDG<dim>::setup_system()
{

}

template <int dim>
void InviscidBurgersDG<dim>::assemble_system()
{

}

template <int dim>
double InviscidBurgersDG<dim>::burgers_flux(double u)
{
    return 0.5*(u * u);
}

template <int dim>
double InviscidBurgersDG<dim>::numerical_flux(double u_plus, double u_minus, double lambda)
{
    return 0.5*(burgers_flux(u_minus) + burgers_flux(u_plus) - lambda*(u_plus - u_minus));
}

int main()
{

    return 0;
}