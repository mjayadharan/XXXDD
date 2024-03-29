/* ---------------------------------------------------------------------
 * Declaration of MixedStokesProblemDD class template
 * ---------------------------------------------------------------------
 *
 * Author: Manu Jayadharan, Northwestern University, 2024.
 * based on the Eldar Khattatov's Elasticity DD implementation from 2017.
 */

#ifndef ELASTICITY_MFEDD_ELASTICITY_MFEDD_H
#define ELASTICITY_MFEDD_ELASTICITY_MFEDD_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <cstdlib>
#include <fstream>
#include <iostream>

#include "projector.h"

namespace dd_stokes
{
  using namespace dealii;

  // Mixed Elasticity Domain Decomposition class template
  template <int dim>
  class MixedStokesProblemDD
  {
  public:
    MixedStokesProblemDD(const unsigned int degree,
                         const unsigned int mortar_flag   = 0,
                         const unsigned int mortar_degree = 0);

    void
    run(const unsigned int                            refine,
        const std::vector<std::vector<unsigned int>> &reps,
        double                                        tol,
        std::string                                   name,
        unsigned int                                  maxiter,
        unsigned int                                  quad_degree = 11);


  private:
    MPI_Comm   mpi_communicator;
    MPI_Status mpi_status;

    Projector::Projector<dim> P_coarse2fine;
    Projector::Projector<dim> P_fine2coarse;

    void
    make_grid_and_dofs();

    void
    assemble_system();

    void
    get_interface_dofs();

    void
    assemble_rhs_star(FEFaceValues<dim> &fe_face_values);

    void
    solve_bar();

    void
    solve_star();

    void
    compute_multiscale_basis();

    void
    local_cg(const unsigned int &maxiter);

    double
    compute_interface_error(Function<dim> &exact_solution);

    void
    compute_errors(const unsigned int &cycle);

    void
    output_results(const unsigned int &cycle,
                   const unsigned int &refine,
                   const std::string & name);

    //For implementing GMRES
    void
	givens_rotation(double v1, double v2, double &cs, double &sn);

    void
	apply_givens_rotation(std::vector<double> &h, std::vector<double> &cs, std::vector<double> &sn,
    							unsigned int k_iteration);

    void
	back_solve(std::vector<std::vector<double>> H, std::vector<double> beta, std::vector<double> &y);

    void
    local_gmres(const unsigned int &maxiter);
    double vect_norm(std::vector<double> v);
    //just to test the local_gmres algorithm
      void
  	testing_gmres(const unsigned int &maxiter);


    unsigned int       gmres_iteration;
    // Number of subdomains in the computational domain
    std::vector<unsigned int> n_domains;

    // FE degree and DD parameters
    const unsigned int degree;
    const unsigned int mortar_degree;
    const unsigned int mortar_flag;
    unsigned int       cg_iteration;
    double             tolerance;
    unsigned int       qdegree;

    // Neighbors and interface information
    std::vector<int>                       neighbors;
    std::vector<unsigned int>              faces_on_interface;
    std::vector<unsigned int>              faces_on_interface_mortar;
    std::vector<std::vector<unsigned int>> interface_dofs;

    unsigned long n_stress_interface;

    // Subdomain coordinates (assuming logically rectangular blocks)
    Point<dim> p1;
    Point<dim> p2;

    // Fine triangulation
    Triangulation<dim> triangulation;
    FESystem<dim>      fe;
    DoFHandler<dim>    dof_handler;

    // Mortar triangulation
    Triangulation<dim> triangulation_mortar;
    FESystem<dim>      fe_mortar;
    DoFHandler<dim>    dof_handler_mortar;

    // Star and bar problem data structures
    BlockSparsityPattern      sparsity_pattern;
    BlockSparseMatrix<double> system_matrix;
    SparseDirectUMFPACK       A_direct;

    BlockVector<double> solution_bar_elast;
    BlockVector<double> solution_star_elast;
    BlockVector<double> solution;
    BlockVector<double> system_rhs_bar_elast;
    BlockVector<double> system_rhs_star_elast;
    BlockVector<double> interface_fe_function;

    // Mortar data structures
    BlockVector<double>              interface_fe_function_mortar;
    BlockVector<double>              solution_bar_mortar;
    BlockVector<double>              solution_star_mortar;
    std::vector<BlockVector<double>> multiscale_basis;

    // Output extra
    ConditionalOStream pcout;
    ConvergenceTable   convergence_table;
    TimerOutput        computing_timer;




  };
} // namespace dd_stokes

#endif // ELASTICITY_MFEDD_ELASTICITY_MFEDD_H
