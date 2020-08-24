#include <iostream>
#include <vector>
#include <string>
#include <time.h>
#include <stdlib.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/base/timer.h>
#include <deal.II/grid/manifold_lib.h>

#include <galerkin_tools/solver_wrapper.h>
#include <incremental_fe/scalar_functionals/omega_lib.h>
#include <incremental_fe/scalar_functionals/psi_lib.h>
#include <incremental_fe/scalar_functionals/linear_material_00.h>
#include <incremental_fe/fe_model.h>

using namespace std;
using namespace dealii;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;

template <unsigned int spacedim>
class AppliedPotential : public Function<spacedim>
{
private:

	const double
	mu;

	const double
	t_1;

public:

	AppliedPotential(	const double mu,
						const double t_1)
	:
	Function<spacedim>(),
	mu(mu),
	t_1(t_1)
	{
	}

	double
	value(	const Point<spacedim>&	/*p*/,
			const unsigned int		/*component=0*/)
	const
	{
		const double t = this->get_time();
		if(t < t_1)
			return mu * (t / t_1);
		else
			return mu;
	}
};

//post-processor computing a * c + b, where c is a scalar field and a and b are constant coefficients
template<unsigned int spacedim>
class PostprocessorConcentration : public DataPostprocessorScalar<spacedim>
{
	const double
	a;

	const double
	b;

	const unsigned int
	global_component_index_c;

public:
	PostprocessorConcentration(	const string&		name,
								const unsigned int	global_component_index_c,
								const double		a,
								const double		b)
	:
	DataPostprocessorScalar<spacedim>(name, update_values),
	a(a),
	b(b),
	global_component_index_c(global_component_index_c)
	{
	}

	void
	evaluate_vector_field(	const DataPostprocessorInputs::Vector<spacedim>&	input_data,
							vector<Vector<double>>&								computed_quantities)
	const
	{

		for(unsigned int dataset = 0; dataset < input_data.solution_values.size(); ++dataset)
		{
			computed_quantities[dataset][0] = a * input_data.solution_values[dataset][global_component_index_c] + b;
		}
	}
};

int main()
{

	srand(time(NULL));

	const unsigned int spacedim = 2;

/**************
 * parameters *
 **************/

	// normalization quantities
	const double c_norm = 1.0;																		// mol/m^3
	const double R_norm = 8.31446261815324;															// J/mol/K
	const double T_norm = 298.0;																	// K
	const double L_norm = 0.001;																	// m
	const double D_norm = 1.0e-12;																	// m^2/s

	// model geometry and mesh
	const double L = 0.001/L_norm;																	// geometric dimension of model
	const unsigned int N_refinements = 2;															// number of global refinements

	// material parameters
	const double c_0 = 1.0/c_norm;																	// initial concentration of diffusing species (mol/m^3)
	const double c_v_0 = 1.0/c_norm;																// initial concentration of vacancies (mol/m^3)
	const double mu_0 = 0.0;																		// reference chemical potential of diffusing species divided by RT
	const double mu_v_0 = 0.0;																		// reference chemical potential of vacancies divided by RT
	const double D = 1e-12 / D_norm;																// Dissipation coefficient (m^2/s)
	const double RT = (8.31446261815324 * 298.0) / (R_norm * T_norm);								// gas constant times temperature (J/mol)

	// times and loading
	const double t_1 = 0.1 * (L_norm * L_norm / D_norm) * D_norm / L_norm / L_norm;					// total time of computation (s)
	const double U = -1.0*(R_norm * T_norm) / (R_norm * T_norm);									// applied potential difference (J/mol)

	// numerical parameters
	const unsigned int degree = 1;																	// degree of finite element approximation
	const double alpha = 0.5;																		// time integration parameter
	const unsigned int method = 1;																	// numerical method
																									// (0: Miehe's method, 1: alpha-family, 2: modified alpha-family)
	MappingQGeneric<spacedim, spacedim> mapping_domain(1);											// FE mapping on domain
	MappingQGeneric<spacedim-1, spacedim> mapping_interface(1);										// FE mapping on interfaces
	const unsigned int n_subdivisions = 1;															// number of element subdivisions in output

	// refinements in time
	const unsigned int steps_min = 1;																// minimum number of time steps until final time
	const unsigned int max_ref_t = 15;																// maximum refinements in time
	const unsigned int steps_max = steps_min * (unsigned int)(pow(2., (double)max_ref_t) + 0.5);	// maximum number of time steps until final time

	// further parameters
	const string variant_string = "_" + Utilities::to_string(alpha)
								+ "_" + Utilities::to_string(method);
	const string file_name_res	= "results/results_alpha" + variant_string + ".dat";				// file where results are stored
	const string file_name_ref	= "results/results_ref.dat";										// file where reference solution is stored
	const bool read_reference_from_file = false;														// determines whether reference solution is read from file or computed
																									// (of course, it must be computed at least once ...)
	const unsigned int solver_sym = 0;																// solver for method != 1: 0 - PARDISO, 1 - MA57, else - UMFPACK
	const unsigned int solver_unsym = 1;															// solver for method == 1: 0 - PARDISO, else - UMFPACK

	// set up global data object for information transfer between different places and request predictor corrector algorithm if necessary
	GlobalDataIncrementalFE<spacedim> global_data;
	if(method == 2)
		global_data.set_predictor_corrector();

/**********************
 * independent fields *
 **********************/

	IndependentField<spacedim, spacedim> I("I", FE_RaviartThomas<spacedim>(degree), {0});				// flux
	IndependentField<spacedim, spacedim> delta_c("delta_c", FE_DGQ<spacedim>(degree), 1, {0});			// change in species concentration
	IndependentField<spacedim, spacedim> mu("mu", FE_DGQ<spacedim>(degree), 1, {0});					// Lagrangian multiplier (chemical potential)

/********************
 * dependent fields *
 ********************/

	// flux components
	DependentField<spacedim, spacedim> I_x("I_x");
	DependentField<spacedim, spacedim> I_y("I_y");
	DependentField<spacedim, spacedim> I_z("I_z");
	I_x.add_term(1.0, I, 0);
	I_y.add_term(1.0, I, 1);
	if(spacedim == 3)
		I_z.add_term(1.0, I, 2);

	// divergence of flux
	DependentField<spacedim, spacedim> div_I("div_I");
	div_I.add_term(1.0, I, 0, 0);
	div_I.add_term(1.0, I, 1, 1);
	if(spacedim == 3)
		div_I.add_term(1.0, I, 2, 2);

	// concentration of diffusing species
	DependentField<spacedim, spacedim> c("c");
	c.add_term(1.0, delta_c);
	c.add_term(c_0);

	// concentration of vacancies
	DependentField<spacedim, spacedim> c_v("c_v");
	c_v.add_term(-1.0, delta_c);
	c_v.add_term(c_v_0);

	// chemical potential
	DependentField<spacedim, spacedim> mu_("mu");
	mu_.add_term(1.0, mu);

	// components of I on surfaces
	DependentField<spacedim-1, spacedim> I_x_s("I_x_s");
	DependentField<spacedim-1, spacedim> I_y_s("I_y_s");
	DependentField<spacedim-1, spacedim> I_z_s("I_z_s");
	I_x_s.add_term(1.0, I, 0, InterfaceSide::minus);
	I_y_s.add_term(1.0, I, 1, InterfaceSide::minus);
	if(spacedim == 3)
		I_z_s.add_term(1.0, I, 2, InterfaceSide::minus);

/********
 * grid *
 ********/

	// domain mesh
	Triangulation<spacedim> tria_domain_, tria_domain;
	Point<spacedim> p_1(-L*0.5, 0.0), p_2(L*0.5, L*0.5);
	vector<unsigned int> subdivisions(spacedim, 2);
	subdivisions[0] = 4;
	GridGenerator::subdivided_hyper_rectangle(tria_domain_, subdivisions, p_1, p_2);
 	std::set<typename Triangulation<spacedim>::active_cell_iterator> cells_to_remove;

 	// define material id's
	for(auto& cell : tria_domain_.active_cell_iterators())
	{
		cell->set_material_id(0);
		if(cell->center().distance(Point<spacedim>()) < 0.25 * L)
			cells_to_remove.insert(cell);
	}
	GridGenerator::create_triangulation_with_removed_cells (tria_domain_, cells_to_remove, tria_domain);
	for(auto& cell : tria_domain.active_cell_iterators())
	{
		for(unsigned int v = 0; v < GeometryInfo<spacedim>::vertices_per_cell; ++v)
		{
			if( fabs( cell->vertex(v).distance(Point<spacedim>()) - 0.25 * L * sqrt(2) ) < 1e-8 )
				cell->vertex(v) = cell->vertex(v) / sqrt(2);
		}
	}

	// triangulation system
	TriangulationSystem<spacedim> tria_system(tria_domain);

	// define interfaces
	tria_domain.set_all_manifold_ids(1);
	tria_domain.set_all_manifold_ids_on_boundary(0);
	for(const auto& cell : tria_domain.cell_iterators_on_level(0))
	{
		for(unsigned int face = 0; face < GeometryInfo<spacedim>::faces_per_cell; ++face)
		{
			if(cell->face(face)->at_boundary())
			{
				if(cell->face(face)->center()[0] < -0.5*L + 1e-12)
				{
					tria_system.add_interface_cell(cell, face, 0);
				}
				else if(cell->face(face)->center()[0] > 0.5*L - 1e-12)
				{
					tria_system.add_interface_cell(cell, face, 1);
				}
				else if(cell->face(face)->center()[1] < 1e-12)
				{
					tria_system.add_interface_cell(cell, face, 1);
				}
				else if(cell->face(face)->center()[1] > 0.5*L - 1e-12)
				{
					tria_system.add_interface_cell(cell, face, 1);
				}
				else
				{
					tria_system.add_interface_cell(cell, face, 1);
					cell->face(face)->set_all_manifold_ids(2);
				}
			}
		}
	}

	SphericalManifold<spacedim> spherical_manifold_domain;
	SphericalManifold<spacedim-1, spacedim> spherical_manifold_interface;
	tria_system.set_interface_manifold(2,spherical_manifold_interface);
	TransfiniteInterpolationManifold<spacedim> transfinite_interpolation_manifold;
	tria_domain.set_manifold (2, spherical_manifold_domain);
	transfinite_interpolation_manifold.initialize(tria_domain);
	tria_domain.set_manifold (1, transfinite_interpolation_manifold);

	tria_system.close();
	tria_domain.refine_global(N_refinements);

/*************
 * potential *
 *************/

	// chemical potential of diffusing species
	PsiChemical00<spacedim> psi_c(	{c},
									{0},
									QGauss<spacedim>(degree + 1),
									global_data,
									RT, c_0, mu_0,
									alpha,
									0.0);

	// chemical potential of vacancies
	PsiChemical00<spacedim> psi_c_v({c_v},
									{0},
									QGauss<spacedim>(degree + 1),
									global_data,
									RT, c_v_0, mu_v_0,
									alpha,
									0.0);

	// dissipation
	OmegaFluxDissipation00<spacedim> delta(	{I_x, I_y, I_z, c},
											{0},
											QGauss<spacedim>(degree + 2),	// very important to integrate here with degree + 2, otherwise stability threshold is dt ~ h^4 for alpha < 0.5
											global_data,
											D/RT,
											method,
											alpha);

	// Lagrangian multiplier term for incorporation of constraint between rates of c and I
	OmegaDivergenceConstraint00<spacedim> constraint_c_I(	{div_I, c, mu_},
															{0},
															QGauss<spacedim>(degree + 1),
															global_data,
															method,
															alpha);

	// Power term
	AppliedPotential<spacedim> applied_potential_fun(U, t_1);
	OmegaFluxPower00<spacedim> applied_potential( 	{I_x_s, I_y_s, I_z_s},
													{0},
													QGauss<spacedim-1>(degree + 1),
													global_data,
													applied_potential_fun,
													method,
													alpha);

	TotalPotentialContribution<spacedim> psi_c_tpc(psi_c);
	TotalPotentialContribution<spacedim> psi_c_v_tpc(psi_c_v);
	TotalPotentialContribution<spacedim> delta_tpc(delta);
	TotalPotentialContribution<spacedim> constraint_c_I_tpc(constraint_c_I);
	TotalPotentialContribution<spacedim> applied_potential_tpc(applied_potential);

 	TotalPotential<spacedim> total_potential;
 	total_potential.add_total_potential_contribution(psi_c_tpc);
 	total_potential.add_total_potential_contribution(psi_c_v_tpc);
	total_potential.add_total_potential_contribution(delta_tpc);
	total_potential.add_total_potential_contribution(constraint_c_I_tpc);
	total_potential.add_total_potential_contribution(applied_potential_tpc);

	Constraints<spacedim> constraints;

/***************************************************
 * set up finite element model and do computations *
 ***************************************************/

	BlockSolverWrapperPARDISO solver_wrapper_pardiso;
	if((method != 1) || (alpha == 0.0))
		solver_wrapper_pardiso.matrix_type = 2;
	else
		solver_wrapper_pardiso.matrix_type = 0;
	BlockSolverWrapperUMFPACK2 solver_wrapper_umfpack;
	BlockSolverWrapperMA57 solver_wrapper_ma57;

	SolverWrapper<Vector<double>, BlockVector<double>, TwoBlockMatrix<SparseMatrix<double>>, TwoBlockSparsityPattern>* solver_wrapper;
	if((method != 1) || (alpha == 0.0))
	{
		switch(solver_sym)
		{
			case 0:
				solver_wrapper = &solver_wrapper_pardiso;
				cout << "Selected PARDISO as solver" << endl;
				if(alpha == 0.0)
					solver_wrapper_pardiso.res_max = 1e16;
				break;
			case 1:
				solver_wrapper = &solver_wrapper_ma57;
				cout << "Selected MA57 as solver" << endl;
				break;
			default:
				solver_wrapper = &solver_wrapper_umfpack;
				cout << "Selected UMFPACK as solver" << endl;
		}
	}
	else
	{
		switch(solver_unsym)
		{
			case 0:
				solver_wrapper = &solver_wrapper_pardiso;
				cout << "Selected PARDISO as solver" << endl;
				break;
			default:
				solver_wrapper = &solver_wrapper_umfpack;
				cout << "Selected UMFPACK as solver" << endl;
		}
	}
	solver_wrapper_pardiso.analyze = 1;
	solver_wrapper_umfpack.analyze = 1;
	solver_wrapper_ma57.analyze = 1;
	global_data.set_compute_sparsity_pattern(1);
	global_data.set_max_iter(15);
	global_data.set_threshold_residual(1e-15);
	global_data.set_perform_line_search(false);

// first generate reference solution for comparison

	FEModel<spacedim, Vector<double>, BlockVector<double>, GalerkinTools::TwoBlockMatrix<SparseMatrix<double>>> fe_model_reference(total_potential, tria_system, mapping_domain, mapping_interface, global_data, constraints, *solver_wrapper);

	// manually generate zero normal flux constraints as this is not automized for Raviart-Thomas elements
	vector<unsigned int> dof_indices_local_global;
	const unsigned int global_component_index_I = fe_model_reference.get_assembly_helper().get_u_omega_global_component_index(I);
	AffineConstraints<double> custom_constraints;
	for(const auto& cell : fe_model_reference.get_assembly_helper().get_dof_handler_system().interface_active_iterators())
	{
		if(cell.refinement_case == InterfaceRefinementCase::at_boundary)
		{
			if(cell.interface_cell->material_id() == 1)
			{
				const auto& domain_cell = cell.domain_cell_minus;
				const unsigned int face = cell.face_minus;
				dof_indices_local_global.resize(domain_cell->get_fe().n_dofs_per_cell());
				domain_cell.get_dof_indices(dof_indices_local_global);
				for(unsigned int m = 0; m < domain_cell->get_fe().n_dofs_per_face(); ++m)
				{
					const unsigned int local_index = domain_cell->get_fe().face_to_cell_index(m, face, domain_cell->face_orientation(face), domain_cell->face_flip(face), domain_cell->face_rotation(face));
					const auto& nonzero_components = domain_cell->get_fe().get_nonzero_components(local_index);
					for(unsigned int d = 0; d < spacedim; ++d)
					{
						if(nonzero_components[global_component_index_I + d])
						{
							const unsigned int global_index = dof_indices_local_global[local_index];
							custom_constraints.add_line(dof_indices_local_global[local_index]);
							break;
						}
					}
				}
			}
		}
	}
	custom_constraints.close();

	// post processing quantities
	PostprocessorConcentration<spacedim> pp_c("c", fe_model_reference.get_assembly_helper().get_u_omega_global_component_index(delta_c), 1.0, c_0);
	fe_model_reference.attach_data_postprocessor_domain(pp_c);
	PostprocessorConcentration<spacedim> pp_c_v("c_v", fe_model_reference.get_assembly_helper().get_u_omega_global_component_index(delta_c), -1.0, c_v_0);
	fe_model_reference.attach_data_postprocessor_domain(pp_c_v);

	if(read_reference_from_file)
	{
		fe_model_reference.read_solution_from_file(file_name_ref);
	}
	else
	{
		for(unsigned int step = 0; step < steps_max; ++step)
		{
			cout << "time step " << step + 1 <<" of " << steps_max << endl;
			if(fe_model_reference.do_time_step(t_1*(step + 1.0)/(double)steps_max, custom_constraints) < 0)
			{
				global_data.print_error_messages();
				cout << "FATAL ERROR!" << endl;
				return 1;
			}
			//fe_model_reference.write_output_independent_fields("results/output_files/domain" + variant_string + "_n" + Utilities::to_string(max_ref_t), "results/output_files/interface" + variant_string + "_n" + Utilities::to_string(max_ref_t), n_subdivisions);
		}
		fe_model_reference.write_output_independent_fields("results/output_files/domain" + variant_string + "_n" + Utilities::to_string(max_ref_t), "results/output_files/interface" + variant_string + "_n" + Utilities::to_string(max_ref_t), n_subdivisions);
		global_data.print_error_messages();
		fe_model_reference.write_solution_to_file(file_name_ref);
	}

// now compute with different step widths and compare with reference solution

	FILE* printout = fopen (file_name_res.c_str(),"w");
	unsigned int steps = steps_min;
	for(unsigned int refinement_step = 0; refinement_step < max_ref_t; ++refinement_step)
	{
		global_data.reinit();
		solver_wrapper_pardiso.analyze = 1;
		solver_wrapper_umfpack.analyze = 1;
		solver_wrapper_ma57.analyze = 1;
		global_data.set_compute_sparsity_pattern(1);
		global_data.set_max_iter(15);
		global_data.set_threshold_residual(1e-15);
		global_data.set_perform_line_search(false);

		double solve_time = 0.0;
		FEModel<spacedim, Vector<double>, BlockVector<double>, GalerkinTools::TwoBlockMatrix<SparseMatrix<double>>> fe_model(total_potential, tria_system, mapping_domain, mapping_interface, global_data, constraints, *solver_wrapper);

		Timer timer;
		timer.start ();

		bool error = false;
		for(unsigned int step = 0; step < steps; ++step)
		{
			cout << "time step " << step+1 <<" of " << steps << " (refinement cycle " << refinement_step << ")" << endl;
			if(fe_model.do_time_step(t_1*(step + 1.0)/(double)steps, custom_constraints) < 0)
			{
				cout << "FATAL ERROR!" << endl;
				global_data.print_error_messages();
				error = true;
				break;
			}
			solve_time += fe_model.get_solve_time_last_step();
			//fe_model.write_output_independent_fields("results/output_files/domain" + variant_string + "_n" + Utilities::to_string(refinement_step), "results/output_files/interface" + variant_string + "_n" + Utilities::to_string(refinement_step), n_subdivisions);
		}
		if(!error)
			fe_model.write_output_independent_fields("results/output_files/domain" + variant_string + "_n" + Utilities::to_string(refinement_step), "results/output_files/interface" + variant_string + "_n" + Utilities::to_string(refinement_step), n_subdivisions);

		timer.stop ();
		cout << "Elapsed CPU time step: " << timer.cpu_time() << " seconds." << endl;
		cout << "Elapsed wall time step: " << timer.wall_time() << " seconds." << endl;

		global_data.print_error_messages();
		const double dt = t_1 / (double)steps;
		ComponentMask cm_domain(spacedim + 2, false);
		cm_domain.set(fe_model_reference.get_assembly_helper().get_u_omega_global_component_index(delta_c), true);
		double d = 1e16;
		if(!error)
			d = fe_model_reference.compute_distance_to_other_solution(fe_model, QGaussLobatto<spacedim>(degree+1), QGaussLobatto<spacedim-1>(degree+1), VectorTools::NormType::Linfty_norm, cm_domain).first;
		fprintf(printout, "%- 1.16e %- 1.16e %- 1.16e%- 1.16e \n", dt, d, timer.wall_time(), solve_time);
		steps *= 2;
	}
	fclose(printout);
}
