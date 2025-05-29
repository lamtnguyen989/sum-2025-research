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

        virtual double value(const Point<dim> &p, const unsigned int component=0) const override
        {
            (void) component;   //  This is here to shut up the compiler
            return std::sin(numbers::PI * p[0]);
        }
};

/*
    Define the Dirichlet Boundary Condition on spatial interval [a,b] as u(a,t) = u(b,t) = 0
*/
template <int dim>
class BoundaryValues : public Function<dim>
{
    public:
        BoundaryValues() : Function<dim>() {};
        virtual double value(const Point<dim> &p, const unsigned int component=0) const override
        {
            (void) component;   // Shut up the compiler
            const double epsilon = 1e-10;    // Error needed to be consider a boundary point
            // Boundary condtion 
            if (std::abs(p[0] - 1.0) < epsilon || std::abs(p[0] + 1.0) < epsilon)
            {
                return 0.0;
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
    FEInterfaceValues<spacetime_dim>    fe_interface_values;  

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
        fe_interface_values(mapping, fe, face_quadrature, face_update_flags)
    {}
    
    // Copy constructor (needed for MeshWorker)
    ScratchData(const ScratchData<spacetime_dim> &scratch_data)
        : fe_values(scratch_data.fe_values.get_mapping(),
                scratch_data.fe_values.get_fe(),
                scratch_data.fe_values.get_quadrature(),
                scratch_data.fe_values.get_update_flags())
        , fe_interface_values(scratch_data.fe_interface_values.get_mapping(),
                    scratch_data.fe_interface_values.get_fe(),
                    scratch_data.fe_interface_values.get_quadrature(),
                    scratch_data.fe_interface_values.get_update_flags())
    {}
};


struct CopyDataFace
{
    FullMatrix<double>                      cell_matrix;
    Vector<double>                          cell_residual;
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
        InviscidBurgersDG(const unsigned int deg);
        void run();
    private:
        void make_grid();
        void setup_system();
        void assemble_system();
        void solve();
        void output_data();
        void initial_condition();
        double burgers_flux(double u);
        double numerical_flux(double u_plus, double u_minus);

    // System parameters
    unsigned int degree;            // degree of the solving system
    const double x_min, x_max;      // 1D spatial interval
    const double t_min, t_max;      // Time evolution
    unsigned int spatial_cells;     // Number of cells in spatial domain (most likely used as parameter for Grid's repetitions)
    unsigned int time_cells;        // Number of cells for spatial direction (again parameter for Grid's repetitions)
    double current_time;            // Time-tracking (hopefully can use this for intermediate solutions processing)

    // Systems components and data
    Triangulation<dim+1>    triangulation;
    DoFHandler<dim+1>       dof_handler;
    const FE_DGQ<dim+1>     fe;
    const MappingQ1<dim+1>  mapping;
    const QGauss<dim+1>     quadrature;
    const QGauss<dim>       face_quadrature;
    SparsityPattern         sparsity_pattern;
    SparseMatrix<double>    system_matrix;
    Vector<double>          residual;   // right_hand_side for many code in the tutorials
    Vector<double>          solution;
};


// Constructor
template <int dim>
InviscidBurgersDG<dim>::InviscidBurgersDG(const unsigned int deg):
    degree(deg),
    x_min(-1.0) , x_max(1.0),
    t_min(0.0) , t_max(1.0),
    spatial_cells(64),
    time_cells(32),
    current_time(0.0),
    triangulation(),
    dof_handler(triangulation),
    fe(degree),
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
    AffineConstraints<double> constraints;
    constraints.clear();
    constraints.close();

    // Projecting the initial condition 
    VectorTools::project(dof_handler,
                        constraints,
                        quadrature,
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
    using Iterator = typename DoFHandler<dim+1>::active_cell_iterator;
    const BoundaryValues<dim+1> boundary_function;

    // Cell worker (Volume integrations)
    const auto cell_worker = [&](const Iterator &cell, ScratchData<dim+1> &scratch_data, CopyData &copy_data)
    {
        const unsigned int n_dofs = scratch_data.fe_values.get_fe().n_dofs_per_cell();
        copy_data.reinit(cell, n_dofs);
        scratch_data.fe_values.reinit(cell);


        const FEValues<dim+1> &fe_v = scratch_data.fe_values;
        
        // Infinitesmal volume terms
        const std::vector<double> &JxW  = fe_v.get_JxW_values();

        // Get the u-values (solution) for the flux calculations
        unsigned int n_q_pts = fe_v.n_quadrature_points;
        std::vector<double> u_values(n_q_pts);
        fe_v.get_function_values(solution, u_values);

        // Integrate over volume cells
        for (unsigned int p = 0; p < n_q_pts; p++)
        {
            const double u = u_values[p];

            for (unsigned int i = 0; i < n_dofs; i++)
            {
                // Computing test functions and both spacetime derivatives at the quadrature point
                const double v_i = fe_v.shape_value(i,p);           // i-th Test function value
                const double dv_i_dx = fe_v.shape_grad(i, p)[0];    // Spatial derivative
                const double dv_i_dt = fe_v.shape_grad(i, p)[1];    // Time derivative

                // Cell volume residual: integrating (v_i * ∂u/∂t * dxdt) - (1/2 * u^2 * ∂v_i/∂x * dxdt)
                copy_data.cell_residual(i) += v_i * (u*dv_i_dt) * JxW[p];
                copy_data.cell_residual(i) -= burgers_flux(u) * dv_i_dx * JxW[p];

                // Cell Jacobian computation: ∂F(u)/∂u
                // The integrand at the (i,j) grid point is: (v_i * ∂v_j/∂t * dxdt) - (u * v_i * ∂v_i/∂x * dxdt)
                for (unsigned int j = 0; j < n_dofs; j++)
                {
                    const double v_j = fe_v.shape_value(j,p);             // j-th test fuction value 
                    const double dv_j_dt = fe_v.shape_grad(j, p)[1];    // time derivative of j-th test fucntion   

                    copy_data.cell_matrix(i,j) += v_i * dv_j_dt * JxW[p];
                    copy_data.cell_matrix(i,j) -= u * v_j * dv_i_dx * JxW[p];
                }
            }
        }
    };

    // Boundary worker (Applying BC/ICs)
    const auto boundary_worker = [&](const Iterator &cell, 
                                    const unsigned int face_number, 
                                    ScratchData<dim+1> &scratch_data, 
                                    CopyData &copy_data)
    {
        // Getting the FEFaceValue of the cell from the (reinitialized) FEInterfaceValues
        scratch_data.fe_interface_values.reinit(cell, face_number);
        const FEFaceValuesBase<dim+1> &fe_face = scratch_data.fe_interface_values.get_fe_face_values(0);

        // Getting the number of quadrature points and collecting all volume terms on the face integration
        const unsigned int n_q_pts = fe_face.n_quadrature_points;
        const std::vector<double> &JxW = fe_face.get_JxW_values();

        // Computing u_minus values
        std::vector<double> u_minus(n_q_pts);
        fe_face.get_function_values(solution, u_minus);

        // Integrating the boundary
        for (unsigned int p = 0; p < n_q_pts; p++)
        {
            // Skipping integration at initial condition points
            Point<dim+1> point = fe_face.quadrature_point(p);
            if (std::abs(point[1] - t_min) < 1e-10) 
                continue;

            // Enforcing Boundary condition via the numerical flux
            const double u_plus = boundary_function.value(point);
            const double lambda = std::max(std::abs(u_plus), std::abs(u_minus[p]));
            const double flux = numerical_flux(u_plus, u_minus[p]);

            // Face residual and Jacobians calculations
            // Note that Jacobian is obtained by acting [∂/∂u_minus] on the boundary residual
            for (unsigned int i = 0; i < fe_face.dofs_per_cell; i++)
            {
                // Residual integrand: numerical_flux * v_i * dxdt
                const double v_i = fe_face.shape_value(i, p);
                copy_data.cell_residual(i) -= flux * v_i * JxW[p];

                for (unsigned int j = 0; j < fe_face.dofs_per_cell; j++)
                {
                    // Jacobian integrand: v_i * u_minus/2*(1-lambda) * v_j * ds
                    const double v_j = fe_face.shape_value(j, p);
                    copy_data.cell_matrix(i,j) -= v_i * u_minus[p]/2 * (1-lambda) * v_j * JxW[p];
                }
            }
        }
    };

    // Face worker (interaction between neighboring cells)
    const auto face_worker = [&](const Iterator     &cell,
                                const unsigned int  &f,
                                const unsigned int  &sf,
                                const Iterator      &ncell,
                                const unsigned int  &nf,
                                const unsigned int  &nsf,
                                ScratchData<dim+1>  &scratch_data,
                                CopyData            &copy_data)
    {
        // Getting the FEInterfaceValues object and reinitialize it
        FEInterfaceValues<dim+1> &fe_iv = scratch_data.fe_interface_values;
        fe_iv.reinit(cell, f, sf, ncell, nf, nsf);

        // Getting number of quadrature points and volume elements as usual
        const unsigned int n_q_pts = fe_iv.n_quadrature_points;
        const std::vector<double> &JxW = fe_iv.get_JxW_values();

        // Computing the u_plus and u_minus values
        std::vector<double> u_minus(n_q_pts), u_plus(n_q_pts);
        fe_iv.get_fe_face_values(0).get_function_values(solution, u_minus);
        fe_iv.get_fe_face_values(1).get_function_values(solution, u_plus);  

        // Setup copy_data for the faces
        copy_data.face_data.emplace_back();
        CopyDataFace &copy_data_face = copy_data.face_data.back();

        // dofs on faces
        const unsigned int n_dofs_face = fe_iv.n_current_interface_dofs();
        copy_data_face.joint_dof_indices = fe_iv.get_interface_dof_indices();

        // Reinitialize matrix and residuals
        copy_data_face.cell_matrix.reinit(n_dofs_face, n_dofs_face);
        copy_data_face.cell_residual.reinit(n_dofs_face);
        
        // Residual and Jacobian calculations for current and neighboring cells
        for (unsigned int p = 0; p < n_q_pts; p++)
        {
            const double num_flux = numerical_flux(u_plus[p], u_minus[p]);
            const double lambda = std::max(std::abs(u_plus[p]), std::abs(u_minus[p]));

            // Numerical flux derivatives
            const double dflux_du_minus = 0.5 * (u_minus[p] - lambda);
            const double dflux_du_plus = 0.5 * (u_plus[p] - lambda);

            for (unsigned int i = 0; i < n_dofs_face; i++)
            {
                // Residual integration: [v_i] * numerical_flux * ds; where [] is the jump operator
                // For some reason jump_in_shape_values() method does not seems to work (aparently )
                const double v_i_minus = fe_iv.shape_value(false, i, p);
                const double v_i_plus = fe_iv.shape_value(true, i, p);
                const double v_i_jump = v_i_minus - v_i_plus;

                copy_data_face.cell_residual[i] += num_flux * v_i_jump * JxW[p];

                for (unsigned int j = 0; j < n_dofs_face; j++)
                {
                    // Jacobian integration: [v_i] * (∂f*/∂u_minus * v_j_minus + ∂f*/∂u_plus * v_j_plus) * ds
                    const double v_j_minus = fe_iv.shape_value(false, j, p);
                    const double v_j_plus = fe_iv.shape_value(true, j, p);
                    copy_data_face.cell_matrix(i,j) += v_i_jump * (dflux_du_minus*v_j_minus + dflux_du_plus*v_j_plus) * JxW[p];
                }
            }
        }
    };

    // Copier (assembling local to global data)
    const auto copier = [&](const CopyData &c) 
    {
        AffineConstraints<double> constraints;
        constraints.distribute_local_to_global(c.cell_matrix,
                                            c.cell_residual,
                                            c.local_dof_indices,
                                            system_matrix,
                                            residual);
 
        for (const auto &cdf : c.face_data)
        {
            constraints.distribute_local_to_global(cdf.cell_matrix,
                                                cdf.cell_residual,
                                                cdf.joint_dof_indices,
                                                system_matrix,
                                                residual);
        }
    };

    // MeshWorker Loop
    ScratchData<dim+1>  scratch_data(mapping, fe, quadrature, face_quadrature);
    CopyData            copy_data;

    MeshWorker::mesh_loop(dof_handler.begin_active(), dof_handler.end(),
                        cell_worker,
                        copier,
                        scratch_data,
                        copy_data,
                        MeshWorker::assemble_own_cells | MeshWorker::assemble_boundary_faces | MeshWorker::assemble_own_interior_faces_once,
                        boundary_worker,
                        face_worker);
}

// Solve using Newton-Raphson iteration
template<int dim>
void InviscidBurgersDG<dim>::solve()
{
    const double epsilon = 1e-10;       // Tolerance for residual norm
    const unsigned int max_iter = 20;   // Max Newton-Raphson iteration

    for (unsigned int k = 0; k < max_iter; k++)
    {
        assemble_system();
        const double residual_norm = residual.l2_norm();
        std::cout << "Iteration: " << k+1 << ", residual norm: " << residual_norm << std::endl;

        if (residual_norm < epsilon)
        {
            std::cout << "Solver converged!" << std::endl;
            break;
        }

        SparseDirectUMFPACK solver;
        solver.initialize(system_matrix);
        Vector<double> update(residual.size());
        Vector<double> neg_residual = residual;
        neg_residual *= -1.0;

        solver.vmult(update, neg_residual);

        solution += update;
    }
}

// Output to .vtu file
template <int dim>
void InviscidBurgersDG<dim>::output_data()
{
    DataOut<dim+1> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "u");
    data_out.build_patches();
    
    // Output to .vtu
    std::ofstream output("burgers_solution.vtu");
    data_out.write_vtu(output);
}

template <int dim>
void InviscidBurgersDG<dim>::run()
{
    std::cout << "Running with deal.II version: " << DEAL_II_PACKAGE_VERSION << std::endl;

    make_grid();

    std::cout << "Grid made with: " << triangulation.n_active_cells() << " active cells." << std::endl;

    setup_system();
    std::cout << "System done setting up." << std::endl;

    initial_condition();
    std::cout << "Initial condition applied." << std::endl;

    std::cout << "Starting solving system" << std::endl;

    solve();

    output_data();
    std::cout << "Data written." << std::endl;
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