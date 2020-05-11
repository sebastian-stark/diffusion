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
	const unsigned int N_e = 10;																	// number of elements

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
	const double alpha = 1.0;																		// time integration parameter
	const unsigned int method = 0;																	// numerical method
																									// (0: Miehe's method, 1: alpha-family, 2: modified alpha-family)
	MappingQGeneric<spacedim, spacedim> mapping_domain(1);											// FE mapping on domain
	MappingQGeneric<spacedim-1, spacedim> mapping_interface(1);										// FE mapping on interfaces
	const unsigned int n_subdivisions = 2;															// number of element subdivisions in output


	// refinements in time
	const unsigned int steps_min = 1;																// minimum number of time steps until final time
	const unsigned int max_ref_t = 18;																// maximum refinements in time
	const unsigned int steps_max = steps_min * (unsigned int)(pow(2., (double)max_ref_t) + 0.5);	// maximum number of time steps until final time

	// further parameters
	const string variant_string = "_" + Utilities::to_string(alpha)
								+ "_" + Utilities::to_string(method);
	const string file_name_res	= "results/results_alpha" + variant_string + ".dat";				// file where results are stored
	const string file_name_ref	= "results/results_ref.dat";										// file where reference solution is stored
	const bool read_reference_from_file = true;														// determines whether reference solution is read from file or computed
																									// (of course, it must be computed at least once ...)

	// set up global data object for information transfer between different places and request predictor corrector algorithm if necessary
	GlobalDataIncrementalFE<spacedim> global_data;
	if(method == 2)
		global_data.set_predictor_corrector();

/**********************
 * independent fields *
 **********************/

	IndependentField<spacedim, spacedim> I("I", FE_RaviartThomas<spacedim>(degree), {0});				// flux
//	IndependentField<spacedim, spacedim> I("I", FE_Q<spacedim>(degree+1), spacedim, {0});				// flux
	IndependentField<spacedim, spacedim> delta_c("delta_c", FE_DGQ<spacedim>(degree), 1, {0});			// change in species concentration
	IndependentField<spacedim, spacedim> mu("mu", FE_DGQ<spacedim>(degree), 1, {0});					// Lagrangian multiplier (chemical potential)
	IndependentField<spacedim-1, spacedim> mu_s("mu_s", FE_DGQ<spacedim-1, spacedim>(degree), 1, {1});	// Lagrangian multiplier enforcing zero normal flux
//	IndependentField<spacedim-1, spacedim> mu_s("mu_s", FE_Q<spacedim-1, spacedim>(degree+1), 1, {1});	// Lagrangian multiplier enforcing zero normal flux

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

	// Lagrangian multiplier for normal flux boundary condition
	DependentField<spacedim-1, spacedim> mu_s_("mu_s");
	mu_s_.add_term(1.0, mu_s);

/********
 * grid *
 ********/

	// domain mesh
	Triangulation<spacedim> tria_domain;
	Point<spacedim> p_1, p_2;
	for(unsigned int m = 0; m < spacedim; ++m)
	{
		p_1[m] = -0.5 * L;
		p_2[m] = 0.5 * L;
	}
	vector<unsigned int> subdivisions(spacedim, 1);
	subdivisions[0] = N_e;
	GridGenerator::subdivided_hyper_rectangle(tria_domain, subdivisions, p_1, p_2);

	// define material id's
	for(const auto& cell : tria_domain.cell_iterators_on_level(0))
		cell->set_material_id(0);

	// triangulation system
	TriangulationSystem<spacedim> tria_system(tria_domain);

	// define interfaces
	for(const auto& cell : tria_domain.cell_iterators_on_level(0))
	{
		for(unsigned int face = 0; face < GeometryInfo<spacedim>::faces_per_cell; ++face)
		{
			if(cell->face(face)->at_boundary())
			{
				if(cell->face(face)->center()[0] < -0.5*L + 1e-12)
					tria_system.add_interface_cell(cell, face, 0);
				else
					tria_system.add_interface_cell(cell, face, 1);
			}
		}
	}
	tria_system.close();

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

	// Lagrangian multiplier term for incorporation of zero normal flux constraint
	OmegaZeroNormalFlux00<spacedim> zero_normal_flux( 	{I_x_s, I_y_s, I_z_s, mu_s_},
														{1},
														QGauss<spacedim-1>(degree + 1),
														global_data,
														method,
														alpha);

	TotalPotentialContribution<spacedim> psi_c_tpc(psi_c);
	TotalPotentialContribution<spacedim> psi_c_v_tpc(psi_c_v);
	TotalPotentialContribution<spacedim> delta_tpc(delta);
	TotalPotentialContribution<spacedim> constraint_c_I_tpc(constraint_c_I);
	TotalPotentialContribution<spacedim> applied_potential_tpc(applied_potential);
	TotalPotentialContribution<spacedim> zero_normal_flux_tpc(zero_normal_flux);

 	TotalPotential<spacedim> total_potential;
 	total_potential.add_total_potential_contribution(psi_c_tpc);
 	total_potential.add_total_potential_contribution(psi_c_v_tpc);
	total_potential.add_total_potential_contribution(delta_tpc);
	total_potential.add_total_potential_contribution(constraint_c_I_tpc);
	total_potential.add_total_potential_contribution(applied_potential_tpc);
	total_potential.add_total_potential_contribution(zero_normal_flux_tpc);

	Constraints<spacedim> constraints;

/***************************************************
 * set up finite element model and do computations *
 ***************************************************/

// first generate reference solution for comparison
	BlockSolverWrapperUMFPACK solver_wrapper;
	FEModel<spacedim, Vector<double>, BlockVector<double>, GalerkinTools::TwoBlockMatrix<SparseMatrix<double>>> fe_model_reference(total_potential, tria_system, mapping_domain, mapping_interface, global_data, constraints, solver_wrapper);

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
			if(fe_model_reference.do_time_step(t_1*(step + 1.0)/(double)steps_max) < 0)
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

		FEModel<spacedim, Vector<double>, BlockVector<double>, GalerkinTools::TwoBlockMatrix<SparseMatrix<double>>> fe_model(total_potential, tria_system, mapping_domain, mapping_interface, global_data, constraints, solver_wrapper);

		Timer timer;
		timer.start ();

		bool error = false;
		for(unsigned int step = 0; step < steps; ++step)
		{
			cout << "time step " << step+1 <<" of " << steps << " (refinement cycle " << refinement_step << ")" << endl;
			if(fe_model.do_time_step(t_1*(step + 1.0)/(double)steps) < 0)
			{
				cout << "FATAL ERROR!" << endl;
				global_data.print_error_messages();
				error = true;
				break;
			}
			//fe_model.write_output_independent_fields("results/output_files/domain" + variant_string + "_n" + Utilities::to_string(refinement_step), "results/output_files/interface" + variant_string + "_n" + Utilities::to_string(refinement_step));
		}
		if(!error)
			fe_model.write_output_independent_fields("results/output_files/domain" + variant_string + "_n" + Utilities::to_string(refinement_step), "results/output_files/interface" + variant_string + "_n" + Utilities::to_string(refinement_step));

		timer.stop ();
		cout << "Elapsed CPU time step: " << timer.cpu_time() << " seconds." << endl;
		cout << "Elapsed wall time step: " << timer.wall_time() << " seconds." << endl;

		global_data.print_error_messages();
		const double dt = t_1 / (double)steps;
		ComponentMask cm_domain(spacedim + 2, false), cm_interface(1, false);
		cm_domain.set(fe_model_reference.get_assembly_helper().get_u_omega_global_component_index(delta_c), true);
		double d = 1e16;
		if(!error)
			d = fe_model_reference.compute_distance_to_other_solution(fe_model, QGauss<spacedim>(3), QGauss<spacedim-1>(3), VectorTools::NormType::Linfty_norm, cm_domain, cm_interface).first;
		fprintf(printout, "%- 1.16e %- 1.16e %- 1.16e\n", dt, d, timer.cpu_time());
		steps *= 2;
	}
	fclose(printout);
}
