/* deal.II imports */
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>
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

        virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
        {
            return std::sin(numbers::PI * p[0]);
        }
};

/*
    Define the BoundaryValues of the rectangular grid [a,b] x [0,T]:
        - Initial Condition: sin(pi * x)
        - Boundary condition (Dirichlet): u(a,t) = u(b,t) = 0
*/
template <int dim>
class BoundaryValues : public Function<dim>
{
    public:
        BoundaryValues() : Function<dim>() {};
        virtual double value(const Point<dim> &p, const unsigned int component) const override
        {
            const double epsilon = 1e-10;   // Error needed to be consider a boundary point
            // Boundary condtion 
            if (std::abs(p[0] - 1.0) < epsilon || std::abs(p[0] + 1.0) < epsilon)
            {
                return 0.0;
            }
            
            // Initial condition
            if (std::abs(p[1]) < epsilon)
            {
                return std::sin(numbers::PI * p[0]);
            }

            // These should never be call for points outside of the boundary (the below line should never run)
            return 0.0;
        }
};

/*
    Structs for MeshWorker
*/
template<int spacetime_dim>     // Note that spacetime_dim is essentially dim+1
struct ScratchData
{
    // Data fields (Finite Elements)
    FEValues<spacetime_dim>             fe_values;
    FEInterfaceValues<spacetime_dim>    fe_face_values;  

    // Default constructor
    ScratchData(const MappingQ1<spacetime_dim> &mapping,
                const FiniteElement<spacetime_dim> &fe,
                const Quadrature<spacetime_dim> &quadrature,
                const Quadrature<spacetime_dim - 1> &face_quadrature,
                const UpdateFlags update_flags = update_values | update_gradients |
                                                update_quadrature_points | update_JxW_values,
                const UpdateFlags face_update_flags = update_values | update_gradients | update_normal_vectors |
                                                    update_quadrature_points | update_JxW_values):
        fe_values(mapping, fe, quadrature, update_flags),
        fe_face_values(mapping, fe, face_quadrature, face_update_flags)
    {}
    
    // Copy constructor (needed for MeshWorker)
    ScratchData(const ScratchData<spacetime_dim> &scratch_data)
        : fe_values(scratch_data.fe_values.get_mapping(),
                scratch_data.fe_values.get_fe(),
                scratch_data.fe_values.get_quadrature(),
                scratch_data.fe_values.get_update_flags())
        , fe_face_values(scratch_data.fe_values.get_mapping(),
                    scratch_data.fe_values.get_fe(),
                    scratch_data.fe_values.get_quadrature(),
                    scratch_data.fe_values.get_update_flags())
    {}
};


struct CopyDataFace
{
    FullMatrix<double>                      cell_matrix;
    std::vector<types::global_dof_index>    joint_dof_indices;
};


struct CopyData
{
    FullMatrix<double>                      cell_matrix;
    Vector<double>                          cell_residual;
    std::vector<types::global_dof_index>    local_dof_indices;
    std::vector<CopyDataFace>               face_data;

    template<class Iterator>
    void reinit(const Iterator &cell, unsigned int dofs_per_cell)
    {
        cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
        cell_residual.reinit(dofs_per_cell);

        local_dof_indices.resize(dofs_per_cell);
        cell->get_dof_indices(local_dof_indices);
    }
};

/* 
    Inviscid Burgers solver using Discontinuous Galerkin (DG) method
 */
template <int dim>
class InviscidBurgersDG
{
    public:
        InviscidBurgersDG(unsigned int deg);
        void run();
    private:
        void make_grid();
        void setup_system();
        void assemble_system();
        void solve();
        void output_data();
        void initial_condition();
        double burgers_flux(double u);
        double numerical_flux (double u_plus, double u_minus);
    
    // Systems components and data
    Triangulation<dim+1>    triangulation;
    const FE_DGQ<dim+1>     fe;
    const MappingQ1<dim+1>  mapping;
    const QGauss<dim+1>     quadrature;
    const QGauss<dim>       face_quadrature;
    DoFHandler<dim+1>       dof_handler;
    SparsityPattern         sparsity_pattern;
    SparseMatrix<double>    system_matrix;
    Vector<double>          residual;   // right_hand_side for many code in the tutorials
    Vector<double>          solution;


    // System parameters
    const double x_min, x_max;      // 1D spatial interval
    const double t_min, t_max;      // Time evolution
    unsigned int spatial_cells;     // Number of cells in spatial domain (most likely used as parameter for Grid's repetitions)
    unsigned int time_cells;        // Number of cells for spatial direction (again parameter for Grid's repetitions)
    unsigned int degree;            // degree of the solving system
    double current_time;            // Time-tracking (hopefully can use this for intermediate solutions processing)
};


// Constructor
template <int dim>
InviscidBurgersDG<dim>::InviscidBurgersDG(unsigned int deg):
    degree(deg),
    fe(degree),
    dof_handler(triangulation),
    x_min(-1.0) , x_max(1.0),
    t_min(0.0) , t_max(1.0),
    spatial_cells(64),
    time_cells(32),
    current_time(0.0),
    quadrature(fe.tensor_degree() + 1),
    face_quadrature(fe.tensor_degree() + 1)
{}

/*
    Flux calculations
*/
template <int dim>
double InviscidBurgersDG<dim>::burgers_flux(double u)
{
    return 0.5*(u * u);
}

template <int dim>
double InviscidBurgersDG<dim>::numerical_flux(double u_plus, double u_minus)
{
    double lambda = std::max(std::abs(u_plus), std::abs(u_minus));
    return 0.5*(burgers_flux(u_minus) + burgers_flux(u_plus) - lambda*(u_plus - u_minus));
}

/*
    Initial condition for Burgers
*/
template <int dim>
void InviscidBurgersDG<dim>::initial_condition()
{
    AffineConstraints constraints;  // Empty constraints

    // Projecting the initial condition 
    VectorTools::project(dof_handler,
                        constraints,
                        QGauss<dim+1>(degree+2),
                        InitialCondition<dim+1>(),
                        solution);
}


/*
    Making the grid with GridGenerator::subdivided_hyper_rectangle
*/
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
    bool colorized = false; // Colorized param
    GridGenerator::subdivided_hyper_rectangle(triangulation, repetitions, p1, p2, colorized);
}

/*
    setup_sysem() identical to Step 12 tutorial
*/
template <int dim>
void InviscidBurgersDG<dim>::setup_system()
{
    dof_handler.distribute_dofs(fe);

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
    residual.reinit(dof_handler.n_dofs());
}

/*
    Assembling the system matrix
*/
template <int dim>
void InviscidBurgersDG<dim>::assemble_system()
{
    using Iterator = typename DoFHandler<dim>::active_cell_iterator;
    const BoundaryValues<dim> boundary_function;

    // Cell worker
    const auto cell_worker = [&](const Iterator &cell, ScratchData<dim+1> &scratch_data, CopyData &copy_data)
    {
        const unsigned int n_dofs = scratch_data.fe_values.get_fe().n_dofs_per_cell();
        copy_data.reinit(cell, n_dofs);
        scratch_data.fe_values.reinit(cell);

        const auto &q_points = scratch_data.fe_values.get_quadrature_points();
        const FEValues<dim+1> &fe_v = scratch_data.fe_values;
        
        // volume terms
        const std::vector<double> &JxW  = fe_v.get_JxW_values();

        // Get the u-values (solution) for the flux calculations
        unsigned int n_q_pts = fe_v.n_quadrature_points;
        std::vector<double> u_values(n_q_pts);
        fe_v.get_function_values(solution, u_values);

        // Integrate over volume cells
        for (unsigned int p = 0; p < n_q_pts; p++)
        {
            for (int i = 0; i < n_dofs; i++)
            {
                for (int j = 0; j < n_dofs; j++)
                {
                    // Integrand: v * ∂u/∂t * dxdt
                    copy_data.cell_matrix(i,j) += fe_v.shape_value(i,p) * fe_v.shape_grad(i,p)[1] * JxW[p];

                    // Integrand: - (1/2*u^2 * ∂u/∂x * dxdt)
                    copy_data.cell_matrix(i,j) -= 0.5*burgers_flux(u_values[p]) * fe_v.shape_grad(i,p)[0] * JxW[p];
                }
            }
        }
    };

    const auto boundary_worker = [&](const Iterator &cell, 
                                    const unsigned int face_num, 
                                    ScratchData<dim+1> &scratch_data, 
                                    CopyData &copy_data)
    {
        scratch_data.fe_interface_values.reinit(cell, face_num);
        const FEFaceValues<dim+1> &fe_face = scratch_data.fe_face_values;

        const auto &q_points = fe_face.get_quadrature_points();
        const unsigned int n_facet_dofs = fe_face.get_fe().n_dofs_per_cell();
        const std::vector<double> &JxW = fe_face.get_fe().n_dofs_per_cell();

        // u-values
        unsigned int n_q_pts = fe_face.n_quadrature_points();
        std::vector<double> u_values(n_q_pts);
        fe_face.get_function_values(solution, u_values);

        // Handling boundary points (TODO)
        for (unsigned int p = 0; p < q_points.size(); p++)
        {
            // Applying the BC/IC
            double boundary_value = boundary_function.value(q_points[p]);

            // Get the coordinate and normal vector of the point
            const Point<dim+1> pt = fe_face.quadrature_point(p);  
            const Tensor<1, dim+1> normal = fe_face.normal_vector(p);

            // Integrating the residual at the boundary
            double flux = numerical_flux(u_values[p], boundary_value);
            for (int i = 0; i < n_facet_dofs; i++)
            {
                for (int j = 0; j < n_facet_dofs; j++)
                {
                    // Integrand: v * numericalflux(u_plus, u_minus) * ds
                    copy_data.cell_matrix(i,j) += fe_face.shape_value(i,p) * numerical_flux(fe_face.shape_value(j,p), boundary_value) * JxW[p];
                }
            }
        }
    };

    // Face worker
    const auto face_worker = [&](const Iterator     &cell,
                                 const unsigned int &f,
                                 const unsigned int &sf,
                                 const Iterator     &ncell,
                                 const unsigned int &nf,
                                 const unsigned int &nsf,
                                 ScratchData<dim>   &scratch_data,
                                 CopyData           &copy_data)
    {
        // TODO
    };

    // Copier
    const AffineConstraints<double> constraints;
    const auto copier = [&](const CopyData &c)
    {
        constraints.distribute_local_to_global(c.cell_matrix,
                                            c.cell_residual,
                                            c.local_dof_indices,
                                            system_matrix,
                                            residual);
        for (const auto &cdf : c.face_data)
        {
            constraints.distribute_local_to_global(cdf.cell_matrix,
                                                cdf.joint_dof_indices,
                                                system_matrix);
        }
    };

    // MeshWorker Loop
    ScratchData<dim+1>  scratch_data(mapping, fe, quadrature, face_quadrature);
    CopyData            copy_data;
    MeshWorker::mesh_loop(dof_handler.begin_active(), 
                        dof_handler.end(), 
                        cell_worker,
                        copier,
                        scratch_data,
                        copy_data,
                        MeshWorker::assemble_own_cells | MeshWorker::assemble_boundary_faces |MeshWorker::assemble_own_interior_faces_once,
                        boundary_worker,
                        face_worker);
}

// Solve
template<int dim>
void InviscidBurgersDG<dim>::solve()
{
    
}

template <int dim>
void InviscidBurgersDG<dim>::output_data()
{

}

template <int dim>
void InviscidBurgersDG<dim>::run()
{

}

/*
    main()
*/
int main()
{
    try
    {
        const unsigned int degree = 1;
        InviscidBurgersDG<1> ivBurgersDG(degree);
        ivBurgersDG.run();
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
            << "----------------------------------------------------"
            << std::endl;
        std::cerr << "Exception on processing: " << std::endl
            << exc.what() << std::endl
            << "Aborting!" << std::endl
            << "----------------------------------------------------"
            << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl << std::endl
            << "----------------------------------------------------"
            << std::endl;
        std::cerr << "Unknown exception!" << std::endl
            << "Aborting!" << std::endl
            << "----------------------------------------------------"
            << std::endl;
        return 1;
    }
    return 0;
}