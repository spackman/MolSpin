/////////////////////////////////////////////////////////////////////////
// TaskStaticHSDirectSpectra implementation (RunSection module) by Luca Gerhards
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include <iostream>
#include "TaskStaticHSDirectSpectra.h"
#include "Transition.h"
#include "Operator.h"
#include "Settings.h"
#include "State.h"
#include "SpinSpace.h"
#include "SpinSystem.h"
#include "ObjectParser.h"
#include "Spin.h"
#include "Interaction.h"
#include "Pulse.h"
#include <cmath>
#include <iomanip> // std::setprecision
#ifdef _OPENMP
#include <omp.h>
#endif

namespace RunSection
{
	namespace
	{
		double TraceSparseDense(const arma::sp_cx_mat &A, const arma::cx_mat &B)
		{
			arma::cx_double sum = arma::cx_double(0.0, 0.0);
			for (auto it = A.begin(); it != A.end(); ++it)
			{
				sum += (*it) * B(it.col(), it.row());
			}
			return std::real(sum);
		}
	}

	// -----------------------------------------------------
	// TaskStaticHSDirectSpectra Constructors and Destructor
	// -----------------------------------------------------
	TaskStaticHSDirectSpectra::TaskStaticHSDirectSpectra(const MSDParser::ObjectParser &_parser, const RunSection &_runsection) : BasicTask(_parser, _runsection), timestep(1.0), totaltime(1.0e+4), reactionOperators(SpinAPI::ReactionOperatorType::Haberkorn)
	{
	}

	TaskStaticHSDirectSpectra::~TaskStaticHSDirectSpectra()
	{
	}
	// -----------------------------------------------------
	// TaskStaticHSDirectSpectra protected methods
	// -----------------------------------------------------
	bool TaskStaticHSDirectSpectra::RunLocal()
	{
		this->Log() << "Running method StaticHS_Direct_Yields." << std::endl;

		// If this is the first step, write first part of header to the data file
		if (this->RunSettings()->CurrentStep() == 1)
		{
			this->WriteHeader(this->Data());
		}

		// Loop through all SpinSystems
		auto systems = this->SpinSystems();
		for (auto i = systems.cbegin(); i != systems.cend(); i++) // iteration through all spin systems, in this case (or usually), this is one
		{
			// Count the number of nuclear spins
			int nucspins = 0;
			std::vector<int> SpinNumbers;
			for (auto l = (*i)->spins_cbegin(); l != (*i)->spins_cend(); l++)
			{
				std::string spintype;
				(*l)->Properties()->Get("type", spintype);
				if (spintype != "electron")
				{
					nucspins += 1;
				}
				if (spintype == "electron")
				{
					// Throws an error if the spins are not spin 1/2
					if ((*l)->Multiplicity() != 2)
					{
						std::cout << (*l)->Multiplicity() << std::endl;
						this->Log() << "SkippingSpin System \"" << (*i)->Name() << "\" as electron spins have the wrong multiplicity." << std::endl;
						std::cout << "# ERROR: electron spins have to be spin 1/2! Skipping the SpinSystem." << std::endl;
						return 1;
					}
				}
			}

			this->Log() << "\nStarting with SpinSystem \"" << (*i)->Name() << "\"." << std::endl;

			// Obtain a SpinSpace to describe the system
			SpinAPI::SpinSpace space(*(*i));
			space.UseSuperoperatorSpace(false);
			space.SetReactionOperatorType(this->reactionOperators);

			std::string InitialState;
			arma::cx_mat InitialStateVector;
			if(this->Properties()->Get("initialstate", InitialState))
			{
				// Set up states for time-propagation
				arma::cx_mat TaskInitialStateVector(4, 1);
				std::string InitialStateLower;

				// Convert the string to lowercase for case-insensitive comparison
				InitialStateLower.resize(InitialState.size());
				std::transform(InitialState.begin(), InitialState.end(), InitialStateLower.begin(), ::tolower);

				if (InitialStateLower == "singlet")
				{
					arma::cx_mat SingletState(4, 1);
					SingletState(0) = 0.0;
					SingletState(1) = 1.0 / sqrt(2);
					SingletState(2) = -1.0 / sqrt(2);
					SingletState(3) = 0.0;
					TaskInitialStateVector = SingletState;
					this->Log() << "Singlet initial state." << std::endl;
				}
				else if (InitialStateLower == "tripletminus")
				{
					arma::cx_mat TripletMinusState(4, 1);
					TripletMinusState(0) = 0.0;
					TripletMinusState(1) = 0.0;
					TripletMinusState(2) = 0.0;
					TripletMinusState(3) = 1.0;
					TaskInitialStateVector = TripletMinusState;
					this->Log() << "Triplet minus initial state." << std::endl;
				}
				else if (InitialStateLower == "tripletzero")
				{
					arma::cx_mat TripletZeroState(4, 1);
					TripletZeroState(0) = 0.0;
					TripletZeroState(1) = 1.0 / sqrt(2);
					TripletZeroState(2) = 1.0 / sqrt(2);
					TripletZeroState(3) = 0.0;
					TaskInitialStateVector = TripletZeroState;
					this->Log() << "Triplet zero initial state." << std::endl;
				}
				else if (InitialStateLower == "tripletplus")
				{
					arma::cx_mat TripletPlusState(4, 1);
					TripletPlusState(0) = 1.0;
					TripletPlusState(1) = 0.0;
					TripletPlusState(2) = 0.0;
					TripletPlusState(3) = 0.0;
					TaskInitialStateVector = TripletPlusState;
					this->Log() << "Triplet plus initial state." << std::endl;
				}
				else
				{
					std::cout << "# ERROR: Invalid initial state value! It is set to a Singlet state." << std::endl;
					this->Log() << "Initial state is undefined. Setting it to a Singlet state" << std::endl;
					arma::cx_mat SingletState(4, 1);
					SingletState(0) = 0.0;
					SingletState(1) = 1.0 / sqrt(2);
					SingletState(2) = -1.0 / sqrt(2);
					SingletState(3) = 0.0;
					TaskInitialStateVector = SingletState;
				}
				InitialStateVector = TaskInitialStateVector;
			}
			else
			{
				// Make sure we have an initial state
				auto initial_states = (*i)->InitialState();
				if (initial_states.size() < 1)
				{
					this->Log() << "Skipping SpinSystem \"" << (*i)->Name() << "\" as no initial state was specified." << std::endl;
					continue;
				}

				arma::cx_vec tmp_InitialStateVector;

				for (auto j = initial_states.cbegin(); j != initial_states.cend(); j++)
				{
					if (!space.GetStateSubSpace(*j, tmp_InitialStateVector))
					{
						this->Log() << "Failed to obtain projection matrix onto state \"" << (*j)->Name() << "\", initial state of SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
						continue;
					}
				}

				InitialStateVector = arma::reshape(tmp_InitialStateVector, tmp_InitialStateVector.n_elem, 1);
			}

			int Z = space.SpaceDimensions() / InitialStateVector.n_rows; // Size of the nuclear spin subspace
			std::cout << "# Hilbert Space Size " << InitialStateVector.n_rows * Z << " x " << InitialStateVector.n_rows * Z << std::endl;
			this->Log() << "Hilbert Space Size " << InitialStateVector.n_rows * Z << " x " << InitialStateVector.n_rows * Z << std::endl;
			this->Log() << "Size of Nuclear Spin Subspace " << Z << std::endl;

			arma::cx_mat B;
			B.zeros(Z * InitialStateVector.n_rows, Z);

			for (int it = 0; it < Z; it++)
			{
				arma::colvec temp(Z);
				temp(it) = 1;
				B.col(it) = arma::kron(InitialStateVector, temp);
			}

			// Get Information about the polarization of choice
			bool CIDSP = false;
			if (!this->Properties()->Get("cidsp", CIDSP))
			{
				this->Log() << "Failed to obtain an input for a CIDSP" << std::endl;
			}

			// Get projectors of interest of the spectrum
			arma::sp_cx_mat Iprojx;
			arma::sp_cx_mat Iprojy;
			arma::sp_cx_mat Iprojz;

			std::vector<std::string> spinList;
			int m;

			// Check transitions, rates and projection operators
			auto transitions = (*i)->Transitions();
			arma::sp_cx_mat P;
			int num_transitions = 0;

			int projection_counter = 0;
			std::map<int, arma::sp_cx_mat> Operators;
			std::vector<arma::sp_cx_mat> OperatorsSparse;
			std::vector<arma::cx_mat> OperatorsDense;
			arma::vec rates(1, 1);

			// Getting the projection operators
			if (this->Properties()->GetList("spinlist", spinList, ','))
			{
				for (auto l = (*i)->spins_cbegin(); l != (*i)->spins_cend(); l++)
				{
					for (m = 0; m < (int)spinList.size(); m++)
					{
						if ((*l)->Name() == spinList[m])
						{
							if (!space.CreateOperator(arma::conv_to<arma::sp_cx_mat>::from((*l)->Sx()), (*l), Iprojx))
							{
								return false;
							}
								if (!space.CreateOperator(arma::conv_to<arma::sp_cx_mat>::from((*l)->Sy()), (*l), Iprojy))
							{
								return false;
							}
								if (!space.CreateOperator(arma::conv_to<arma::sp_cx_mat>::from((*l)->Sz()), (*l), Iprojz))
							{
								return false;
							}

							if (CIDSP == true)
							{
								// Gather rates and operators
								for (auto j = transitions.cbegin(); j != transitions.cend(); j++)
								{
									if ((*j)->SourceState() == nullptr)
										continue;
									if (!space.GetState((*j)->SourceState(), P))
									{
										std::cout << "# ERROR: Could not obtain projection matrix!" << std::endl;
										this->Log() << "Failed to obtain projection matrix onto state \"" << (*j)->Name() << "\" of SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
										return 1;
									}
									if (num_transitions != 0)
									{
										rates.insert_rows(num_transitions, 1);
									}

									Operators[projection_counter] = (*j)->Rate() * Iprojx * P;
									projection_counter += 1;
									Operators[projection_counter] = (*j)->Rate() * Iprojy * P;
									projection_counter += 1;
									Operators[projection_counter] = (*j)->Rate() * Iprojz * P;
									projection_counter += 1;

									rates(num_transitions) = (*j)->Rate();
									num_transitions++;
								}
							}
							else
							{
								// Gather rates and operators
								Operators[projection_counter] = Iprojx;
								projection_counter += 1;
								Operators[projection_counter] = Iprojy;
								projection_counter += 1;
								Operators[projection_counter] = Iprojz;
								projection_counter += 1;
							
								for (auto j = transitions.cbegin(); j != transitions.cend(); j++)
								{
									if ((*j)->SourceState() == nullptr)
										continue;

									if (num_transitions != 0)
									{
										rates.insert_rows(num_transitions, 1);
									}

									rates(num_transitions) = (*j)->Rate();
									num_transitions++;
								}
							}
						}	
					}
				}
			}

			OperatorsSparse.resize(projection_counter);
			double total_nnz = 0.0;
			double total_size = 0.0;
			for (const auto &entry : Operators)
			{
				OperatorsSparse[entry.first] = entry.second;
				total_nnz += static_cast<double>(entry.second.n_nonzero);
				total_size += static_cast<double>(entry.second.n_rows) * entry.second.n_cols;
			}
			bool use_sparse_ops = (total_size > 0.0) && ((total_nnz / total_size) < 0.1);
			if (!use_sparse_ops)
			{
				OperatorsDense.resize(projection_counter);
				for (const auto &entry : Operators)
				{
					OperatorsDense[entry.first] = arma::cx_mat(entry.second);
				}
			}
			else if (projection_counter > 0)
			{
				this->Log() << "Using sparse operators for expectation values." << std::endl;
			}

			// Get the Hamiltonian
			arma::sp_cx_mat K;
			K.zeros(InitialStateVector.n_rows * Z, InitialStateVector.n_rows * Z);

			for (auto j = transitions.cbegin(); j != transitions.cend(); j++)
			{
				if ((*j)->SourceState() == nullptr)
					continue;
				space.GetState((*j)->SourceState(), P);
				K += (*j)->Rate() / 2 * P;
			}

			arma::cx_mat Binitial = B;

			// Setting or calculating total time.
			double totaltime = this->totaltime;
			double inputTotaltime = 0.0;
			if (this->Properties()->Get("totaltime", inputTotaltime))
			{
				if (std::isfinite(inputTotaltime) && inputTotaltime >= 0.0)
				{
					totaltime = inputTotaltime;
				}
				else
				{
					this->Log() << "# ERROR: invalid total time!" << std::endl;
					return false;
				}
			}

			// Setting timestep
			double dt = this->timestep;
			double inputTimestep = 0.0;
			if (this->Properties()->Get("timestep", inputTimestep))
			{
				if (std::isfinite(inputTimestep) && inputTimestep > 0.0)
				{
					dt = inputTimestep;
				}
				else
				{
					std::cout << "# WARNING: undefined timestep, using by default 0.1 ns!" << std::endl;
					this->Log() << "# WARNING: undefined timestep, using by default 0.1 ns!" << std::endl;
					dt = 0.1;
				}
			}

			this->Log() << "Time step is chosen as " << dt << " ns." << std::endl;

			// Number of time propagation steps
			int num_steps = std::ceil(totaltime / dt);
			this->Log() << "Number of time propagation steps: " << num_steps << "." << std::endl;
			if (num_steps == 0)
			{
				num_steps = 1;
				this->Log() << "change Number of propagation steps to: " << num_steps << " in order to propagate one step." << std::endl;
			}

			// Choose Propagation Method and other parameters
			std::string propmethod;
			this->Properties()->Get("propagationmethod", propmethod);

			std::string precision;
			this->Properties()->Get("precision", precision);

			int krylovsize;
			this->Properties()->Get("krylovsize", krylovsize);

			double krylovtol;
			this->Properties()->Get("krylovtol", krylovtol);

			if (propmethod == "autoexpm")
			{
				this->Log() << "Autoexpm is chosen as the propagation method." << std::endl;
				if (precision == "double")
				{
					this->Log() << "Double precision is chosen for the autoexpm method." << std::endl;
				}
				else if (precision == "single")
				{
					this->Log() << "Single precision is chosen for the autoexpm method." << std::endl;
				}
				else if (precision == "half")
				{
					this->Log() << "Half precision is chosen for the autoexpm method." << std::endl;
				}
				else
				{
					std::cout << "# ERROR: undefined precision. Using single digit precision!" << std::endl;
					this->Log() << "No precision for autoexpm method was defined. Using single digit precision." << std::endl;
					precision = "single";
				}
			}
			else if (propmethod == "krylov")
			{
				if (krylovsize > 0)
				{
					this->Log() << "Krylov basis size is chosen as " << krylovsize << "." << std::endl;
					if (krylovtol > 0)
					{
						this->Log() << "Tolerance for krylov propagation is chosen as " << krylovtol << "." << std::endl;
					}
					else
					{
						std::cout << "# ERROR: undefined tolerance for krylov subspace propagation! Using the default of 1e-16." << std::endl;
						this->Log() << "Undefined tolerance for the krylov subspace. Using the default of 1e-16." << std::endl;
						krylovtol = 1e-16;
					}
				}
				else
				{
					std::cout << "# ERROR: undefined size of the krylov subspace! Using the default size of 16." << std::endl;
					this->Log() << "Undefined size of the krylov subspace. Using the default size of 16." << std::endl;
					krylovsize = 16;
					if (krylovtol > 0)
					{
						this->Log() << "Tolerance for krylov propagation is chosen as " << krylovtol << "." << std::endl;
					}
					else
					{
						std::cout << "# ERROR: undefined tolerance for krylov subspace propagation! Using the default of 1e-16." << std::endl;
						this->Log() << "Undefined tolerance for the krylov subspace. Using the default of 1e-16." << std::endl;
						krylovtol = 1e-16;
					}
				}
			}
			else
			{
				std::cout << "# WARNING: Undefined propagation method, using normal exponential method." << std::endl; // autoexpm with single accuracy!" << std::endl;
				this->Log() << "WARNING: Undefined propagation method, using normal exponential method." << std::endl;	 // autoexpm with single accuracy." << std::endl;
				propmethod = "normal";
			}

			// Powder averaging options (shared keywords with superspace powder task)
			std::string Method = "timeevo";
			if (!this->Properties()->Get("method", Method))
			{
				this->Log() << "Failed to obtain an input for a Method. Please specify method = timeinf or method = timeevo. Using timeevo by default." << std::endl;
				Method = "timeevo";
			}
			bool method_timeevo = Method.compare("timeevo") == 0;
			bool method_timeinf = Method.compare("timeinf") == 0;
			if (!method_timeevo && !method_timeinf)
			{
				this->Log() << "Method \"" << Method << "\" is not supported for Hilbert space spectra. Using timeevo." << std::endl;
				Method = "timeevo";
				method_timeevo = true;
				method_timeinf = false;
			}

			// Read if the result should be integrated or not.
			bool integration = false;
			if (!this->Properties()->Get("integration", integration))
			{
				this->Log() << "Failed to obtain an input for an integtation. Plese use integration = true/false. Using integration = false by default. " << std::endl;
			}

			// Read integrationwindow from the input file
			std::string Integrationwindow;
			if (!this->Properties()->Get("integrationtimeframe", Integrationwindow))
			{
				this->Log() << "Failed to obtain an input for a integrationtimeframe. Please choose integrationtimeframe = pulse / freeevo / full. Using freeevo propagation evolution window by default" << std::endl;
				Integrationwindow = "freeevo";
			}
			this->Log() << "Timewindow for the propagation integration: " << Integrationwindow << std::endl;

			// Read printtimeframe from the input file
			std::string Timewindow;
			if (!this->Properties()->Get("printtimeframe", Timewindow))
			{
				this->Log() << "Failed to obtain an input for a printtimeframe. Please choose printtimeframe =  pulse / freeevo / full. Using full propagation evolution window by default" << std::endl;
				Timewindow = "full";
			}
			this->Log() << "Timewindow for the propagation printing: " << Timewindow << std::endl;

			bool print_pulses = (Timewindow.compare("freeevo") != 0);
			bool print_freeevo = (Timewindow.compare("pulse") != 0);
			bool integrate_pulses = integration && (Integrationwindow.compare("freeevo") != 0);
			bool integrate_freeevo = integration && (Integrationwindow.compare("pulse") != 0);

			int numPoints = 0;
			bool hasPowderPoints = this->Properties()->Get("powdersamplingpoints", numPoints);
			if (!hasPowderPoints)
			{
				this->Log() << "No powdersamplingpoints provided. Powder averaging is disabled by default." << std::endl;
				numPoints = 0;
			}
			if (numPoints < 1)
			{
				numPoints = 0;
				if (hasPowderPoints)
				{
					this->Log() << "Powder averaging disabled (powdersamplingpoints <= 0)." << std::endl;
				}
			}

			std::vector<std::tuple<double, double, double>> grid;
			if (numPoints > 0)
			{
				if (!this->CreateUniformGrid(numPoints, grid))
				{
					this->Log() << "Failed to obtain a powder grid." << std::endl;
				}
				else if (numPoints > 1)
				{
					this->Log() << "Using powder averaging with " << numPoints << " orientations." << std::endl;
				}
			}
			else
			{
				grid.push_back({0.0, 0.0, 1.0});
			}

			std::vector<std::string> HamiltonianH0list;
			std::vector<std::string> HamiltonianH1list;
			bool hasH0list = this->Properties()->GetList("hamiltonianh0list", HamiltonianH0list, ',');
			bool hasH1list = this->Properties()->GetList("hamiltonianh1list", HamiltonianH1list, ',');

			// Read a pulse sequence from the input
			std::vector<std::tuple<std::string, double>> Pulsesequence;
			bool hasPulseSequence = this->Properties()->GetPulseSequence("pulsesequence", Pulsesequence);
			if (hasPulseSequence)
			{
				this->Log() << "Pulsesequence" << std::endl;
			}

			std::vector<double> pulse_times;
			std::vector<double> pulse_dts;
			bool has_pulse_output = false;
			bool pulse_has_initial_step = false;
			double pulse_total_time = 0.0;
			if (print_pulses && hasPulseSequence)
			{
				double current_time = 0.0;
				bool include_initial_step = true;

				for (const auto &seq : Pulsesequence)
				{
					std::string pulse_name = std::get<0>(seq);
					double timerelaxation = std::get<1>(seq);

					SpinAPI::pulse_ptr pulse_ptr = nullptr;
					for (auto pulse = (*i)->pulses_cbegin(); pulse < (*i)->pulses_cend(); pulse++)
					{
						if ((*pulse)->Name().compare(pulse_name) == 0)
						{
							pulse_ptr = *pulse;
							break;
						}
					}

					if (pulse_ptr == nullptr)
					{
						this->Log() << "Pulse \"" << pulse_name << "\" was not found in SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
						continue;
					}

					double pulse_dt = pulse_ptr->Timestep();
					if (!std::isfinite(pulse_dt) || pulse_dt <= 0.0)
					{
						this->Log() << "Invalid timestep for pulse \"" << pulse_name << "\". Skipping pulse timeline generation." << std::endl;
						continue;
					}

					if (pulse_ptr->Type() == SpinAPI::PulseType::LongPulseStaticField || pulse_ptr->Type() == SpinAPI::PulseType::LongPulse)
					{
						unsigned int steps = static_cast<unsigned int>(std::abs(pulse_ptr->Pulsetime() / pulse_dt));

						if (include_initial_step)
						{
							pulse_times.push_back(current_time);
							pulse_dts.push_back(0.0);
							include_initial_step = false;
						}

						for (unsigned int n = 1; n <= steps; ++n)
						{
							current_time += pulse_dt;
							pulse_times.push_back(current_time);
							pulse_dts.push_back(pulse_dt);
						}
					}

					if (timerelaxation != 0.0)
					{
						unsigned int steps = static_cast<unsigned int>(std::abs(timerelaxation / pulse_dt));
						for (unsigned int n = 1; n <= steps; ++n)
						{
							current_time += pulse_dt;
							pulse_times.push_back(current_time);
							pulse_dts.push_back(pulse_dt);
						}
					}
				}

				pulse_total_time = current_time;
				has_pulse_output = !pulse_times.empty();
				pulse_has_initial_step = has_pulse_output && (pulse_dts.front() == 0.0);
			}

			arma::vec time;
			if (method_timeevo)
			{
				time.set_size(num_steps);
				for (int k = 0; k < num_steps; ++k)
				{
					time(k) = k * dt;
				}
			}

			arma::mat ExptValues; // reused for timeevo
			if (method_timeevo)
			{
				ExptValues.zeros(num_steps, projection_counter);
			}
			arma::cx_mat rho_integrated;
			if (method_timeinf)
			{
				int dim = InitialStateVector.n_rows * Z;
				rho_integrated.zeros(dim, dim);
			}

			size_t grid_size = grid.size();
			int nthreads = 1;
#ifdef _OPENMP
			nthreads = omp_get_max_threads();
#endif

			std::vector<arma::mat> ExptValuesPartial;
			if (method_timeevo)
			{
				ExptValuesPartial.resize(nthreads);
				for (auto &m : ExptValuesPartial)
				{
					m.zeros(num_steps, projection_counter);
				}
			}

			std::vector<arma::mat> ExptValuesPulsePartial;
			if (has_pulse_output)
			{
				ExptValuesPulsePartial.resize(nthreads);
				for (auto &m : ExptValuesPulsePartial)
				{
					m.zeros(pulse_times.size(), projection_counter);
				}
			}

			std::vector<arma::cx_mat> rho_integrated_partial;
			if (method_timeinf)
			{
				int dim = InitialStateVector.n_rows * Z;
				rho_integrated_partial.resize(nthreads);
				for (auto &m : rho_integrated_partial)
				{
					m.zeros(dim, dim);
				}
			}

			arma::cx_mat Iden_dense;
			if (method_timeinf)
			{
				int dim = InitialStateVector.n_rows * Z;
				Iden_dense = arma::eye<arma::cx_mat>(dim, dim);
			}

			SpinAPI::SpinSpace base_space(space);
			base_space.SetReactionOperatorType(this->reactionOperators);
			base_space.UseSuperoperatorSpace(false);

			std::vector<SpinAPI::SpinSpace> spaces;
			spaces.resize(nthreads);
			for (int t = 0; t < nthreads; ++t)
			{
				spaces[t] = base_space;
			}

#pragma omp parallel for schedule(static) if (grid_size > 1)
			for (size_t grid_num = 0; grid_num < grid_size; ++grid_num)
			{
				int tid = 0;
#ifdef _OPENMP
				tid = omp_get_thread_num();
#endif
				SpinAPI::SpinSpace &space_thread = spaces[tid];

				const auto &grid_point = grid[grid_num];
				double theta, phi, weight;
				std::tie(theta, phi, weight) = grid_point;

				arma::mat Rot_mat = arma::eye<arma::mat>(3, 3);
				double gamma = 0.0;
				if (!this->CreateRotationMatrix(gamma, theta, phi, Rot_mat))
				{
					this->Log() << "Failed to obtain rotation matrix for powder orientation." << std::endl;
				}

				arma::sp_cx_mat H;
				if (hasH0list)
				{
					arma::sp_cx_mat H0;
					if (!space_thread.BaseHamiltonianRotated(HamiltonianH0list, Rot_mat, H0))
					{
						this->Log() << "Failed to obtain rotated Hamiltonian for SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
						continue;
					}
					H = H0;

					if (hasH1list)
					{
						arma::sp_cx_mat H1;
						if (!space_thread.ThermalHamiltonian(HamiltonianH1list, H1))
						{
							this->Log() << "Failed to obtain Hamiltonian H1 in Hilbert Space." << std::endl;
							continue;
						}

						H += H1;
					}
				}
				else
				{
					if (!space_thread.Hamiltonian(H))
					{
						this->Log() << "Failed to obtain the Hamiltonian in Hilbert Space." << std::endl;
						std::cout << "# ERROR: Failed to obtain the Hamiltonian!" << std::endl;
						continue;
					}
				}

				arma::cx_mat B = Binitial;

				auto record_expectation = [&](arma::mat &target, size_t row_index, const arma::cx_mat &state) {
					arma::cx_mat state_conj = arma::conj(state);
					for (int idx = 0; idx < projection_counter; ++idx)
					{
						arma::cx_mat OB;
						if (use_sparse_ops)
						{
							OB = OperatorsSparse[idx] * state;
						}
						else
						{
							OB = OperatorsDense[idx] * state;
						}
						double abs_trace = std::real(arma::accu(state_conj % OB));
						target(row_index, idx) = abs_trace / Z;
					}
				};

				size_t pulse_step_index = 0;
				arma::mat ExptValuesPulseOrientation;
				if (has_pulse_output)
				{
					ExptValuesPulseOrientation.zeros(pulse_times.size(), projection_counter);
					if (pulse_has_initial_step)
					{
						record_expectation(ExptValuesPulseOrientation, pulse_step_index, B);
						pulse_step_index = 1;
					}
				}

				// Get pulses and pulse the system for this orientation
				arma::sp_cx_mat A = arma::cx_double(0.0, -1.0) * H - K;

				if (hasPulseSequence)
				{
					// Loop through all pulse sequences
					for (const auto &seq : Pulsesequence)
					{
						// Write which pulse in pulsesequence is calculating now
						if (grid_num == 0)
						{
							this->Log() << std::get<0>(seq) << ", " << std::get<1>(seq) << std::endl;
						}

						// Save the parameters from the input as variables
						std::string pulse_name = std::get<0>(seq);
						double timerelaxation = std::get<1>(seq);

						for (auto pulse = (*i)->pulses_cbegin(); pulse < (*i)->pulses_cend(); pulse++)
						{
							if ((*pulse)->Name().compare(pulse_name) == 0)
							{

								// Apply a pulse to our density vector
								if ((*pulse)->Type() == SpinAPI::PulseType::InstantPulse)
								{
									// Create a Pulse operator in HS; only one side of exponentials as we only propagate wavevectors
									arma::sp_cx_mat pulse_operator;
									if (!space_thread.PulseOperatorOnStatevector((*pulse), pulse_operator))
									{
										this->Log() << "Failed to create a pulse operator in SS." << std::endl;
										continue;
									}

									// Take a step, "first" is propagator and "second" is current state
									B = pulse_operator * B;
								}
								else if ((*pulse)->Type() == SpinAPI::PulseType::LongPulseStaticField)
								{

									// Create a Pulse operator in HS; only one side of exponentials as we only propagate wavevectors
									arma::sp_cx_mat pulse_operator;
									if (!space_thread.PulseOperatorOnStatevector((*pulse), pulse_operator))
									{
										this->Log() << "Failed to create a pulse operator in SS." << std::endl;
										continue;
									}

									// Create array containing a propagator and the current state of each system
									std::pair<arma::sp_cx_mat, arma::cx_mat> G;

									// Get the propagator and put it into the array together with the initial state
									arma::sp_cx_mat A_sp = arma::conv_to<arma::sp_cx_mat>::from(arma::expmat(arma::conv_to<arma::cx_mat>::from((A + (arma::cx_double(0.0, -1.0) * pulse_operator)) * (*pulse)->Timestep())));
									G = std::pair<arma::sp_cx_mat, arma::cx_mat>(A_sp, B);

									unsigned int steps = static_cast<unsigned int>(std::abs((*pulse)->Pulsetime() / (*pulse)->Timestep()));
									for (unsigned int n = 1; n <= steps; n++)
									{
										// Take a step, "first" is propagator and "second" is current state
										B = G.first * G.second;

										// Get the new current state vector matrix
										G.second = B;

										if (has_pulse_output && pulse_step_index < ExptValuesPulseOrientation.n_rows)
										{
											record_expectation(ExptValuesPulseOrientation, pulse_step_index, B);
											++pulse_step_index;
										}
									}
								}
								else if ((*pulse)->Type() == SpinAPI::PulseType::LongPulse)
								{
									// Create a Pulse operator in SS
									arma::sp_cx_mat pulse_operator;
									if (!space_thread.PulseOperatorOnStatevector((*pulse), pulse_operator))
									{
										this->Log() << "Failed to create a pulse operator in SS." << std::endl;
										continue;
									}

									// Create array containing a propagator and the current state of each system
									std::pair<arma::sp_cx_mat, arma::cx_mat> G;

									// Get the propagator and put it into the array together with the initial state
									arma::sp_cx_mat A_sp = arma::conv_to<arma::sp_cx_mat>::from(arma::expmat(arma::conv_to<arma::cx_mat>::from((A + (arma::cx_double(0.0, -1.0) * pulse_operator * std::cos((*pulse)->Frequency() * (*pulse)->Timestep()))) * (*pulse)->Timestep())));
									G = std::pair<arma::sp_cx_mat, arma::cx_mat>(A_sp, B);

									unsigned int steps = static_cast<unsigned int>(std::abs((*pulse)->Pulsetime() / (*pulse)->Timestep()));
									for (unsigned int n = 1; n <= steps; n++)
									{
										// Take a step, "first" is propagator and "second" is current state
										B = G.first * G.second;

										// Get the new current state density vector
										G.second = B;

										if (has_pulse_output && pulse_step_index < ExptValuesPulseOrientation.n_rows)
										{
											record_expectation(ExptValuesPulseOrientation, pulse_step_index, B);
											++pulse_step_index;
										}
									}
								}
								else
								{
									this->Log() << "Not implemented yet, sorry." << std::endl;
								}

								// Get the system relax during the time

								// Create array containing a propagator and the current state of each system
								std::pair<arma::sp_cx_mat, arma::cx_mat> G;
								arma::sp_cx_mat A_sp = arma::conv_to<arma::sp_cx_mat>::from(arma::expmat(arma::conv_to<arma::cx_mat>::from(A * (*pulse)->Timestep())));
								// Get the propagator and put it into the array together with the initial state
								G = std::pair<arma::sp_cx_mat, arma::cx_mat>(A_sp, B);

								unsigned int steps = static_cast<unsigned int>(std::abs(timerelaxation / (*pulse)->Timestep()));
								for (unsigned int n = 1; n <= steps; n++)
								{
									// Take a step, "first" is propagator and "second" is current state
									B = G.first * G.second;

									// Get the new current state density vector
									G.second = B;

									if (has_pulse_output && pulse_step_index < ExptValuesPulseOrientation.n_rows)
									{
										record_expectation(ExptValuesPulseOrientation, pulse_step_index, B);
										++pulse_step_index;
									}
								}
							}
						}
					}
				}

				if (has_pulse_output && pulse_step_index != ExptValuesPulseOrientation.n_rows)
				{
					if (grid_num == 0)
					{
						this->Log() << "Warning: Pulse output step count mismatch. Expected " << ExptValuesPulseOrientation.n_rows
									<< ", recorded " << pulse_step_index << "." << std::endl;
					}
				}

				arma::mat ExptValuesOrientation;
				if (method_timeevo)
					ExptValuesOrientation.zeros(num_steps, projection_counter);

				// Propagate the system in time using the specified method
				if (method_timeevo && propmethod == "autoexpm")
				{
					arma::mat M; // used for variable estimation
					arma::sp_cx_mat H_prop = H - arma::cx_double(0.0, 1.0) * K;

					for (int k = 0; k < num_steps; k++)
					{
						arma::cx_mat Bconj = arma::conj(B);
						// Calculate the expected values for each transition operator
						for (int idx = 0; idx < projection_counter; idx++)
						{
							arma::cx_mat OB;
							if (use_sparse_ops)
							{
								OB = OperatorsSparse[idx] * B;
							}
							else
							{
								OB = OperatorsDense[idx] * B;
							}
							double abs_trace = std::real(arma::accu(Bconj % OB));
							double expected_value = abs_trace / Z;
							ExptValuesOrientation(k, idx) = expected_value;
						}

						// Update B using the Higham propagator
						B = space_thread.HighamProp(H_prop, B, -arma::cx_double(0.0, 1.0) * dt, precision, M);
					}
				}
				else if (method_timeevo && propmethod == "krylov")
				{
					arma::sp_cx_mat H_prop = H - arma::cx_double(0.0, 1.0) * K;

					for (int itr = 0; itr < Z; itr++)
					{
						arma::cx_vec prop_state = B.col(itr);

						// Calculate the expected values for each transition operator
						for (int idx = 0; idx < projection_counter; idx++)
						{
							double result = std::abs(arma::cdot(prop_state, Operators[idx] * prop_state));
							ExptValuesOrientation(0, idx) += result;
						}

						arma::cx_mat Hessen(krylovsize, krylovsize, arma::fill::zeros); // Upper Hessenberg matrix

						arma::cx_mat KryBasis(InitialStateVector.n_rows * Z, krylovsize, arma::fill::zeros); // Orthogonal krylov subspace

						KryBasis.col(0) = prop_state / norm(prop_state);

						double h_mplusone_m;
						space_thread.ArnoldiProcess(H_prop, prop_state, KryBasis, Hessen, krylovsize, h_mplusone_m);

						arma::cx_colvec e1;
						e1.zeros(krylovsize);
						e1(0) = 1;
						arma::cx_colvec ek;
						ek.zeros(krylovsize);
						ek(krylovsize - 1) = 1;

						arma::cx_vec cx = arma::expmat(Hessen * dt) * e1;

						prop_state = norm(prop_state) * KryBasis * cx;

						int k = 1;

						while (k < num_steps)
						{
							// Calculate the expected values for each transition operator
							for (int idx = 0; idx < projection_counter; idx++)
							{
								double result = std::abs(arma::cdot(prop_state, Operators[idx] * prop_state));
								ExptValuesOrientation(k, idx) += result;
							}

							Hessen.zeros(krylovsize, krylovsize);
							KryBasis.zeros(InitialStateVector.n_rows * Z, krylovsize);

							KryBasis.col(0) = prop_state / norm(prop_state);

							space_thread.ArnoldiProcess(H_prop, prop_state, KryBasis, Hessen, krylovsize, h_mplusone_m);
							cx = arma::expmat(Hessen * dt) * e1;

							// Update the state using Krylov Subspace propagator
							prop_state = norm(prop_state) * KryBasis * cx;
							k++;
						}
					}

					ExptValuesOrientation /= Z;
				}
				else if (method_timeevo)
				{
					if (grid_num == 0)
					{
						this->Log() << "Using robust matrix exponential propagator for time-independent Hamiltonian." << std::endl;
					}

					// Include the recombination operator K
					arma::sp_cx_mat H_total = arma::cx_double(0.0, -1.0) * H - K;

					// Precompute the matrix exponential for the entire time step
					arma::cx_mat exp_H = arma::expmat(arma::cx_mat(H_total) * dt);

					// Propagate B
					for (int k = 0; k < num_steps; ++k)
					{
						arma::cx_mat Bconj = arma::conj(B);
						// Calculate the expected values for each transition operator
						for (int idx = 0; idx < projection_counter; ++idx)
						{
							arma::cx_mat OB;
							if (use_sparse_ops)
							{
								OB = OperatorsSparse[idx] * B;
							}
							else
							{
								OB = OperatorsDense[idx] * B;
							}
							double abs_trace = std::real(arma::accu(Bconj % OB));
							double expected_value = abs_trace / Z;
							ExptValuesOrientation(k, idx) = expected_value;
						}

						B = exp_H * B;
					}
				}

				if (has_pulse_output)
					ExptValuesPulsePartial[tid] += weight * ExptValuesPulseOrientation;

				if (method_timeevo)
					ExptValuesPartial[tid] += weight * ExptValuesOrientation;
				if (method_timeinf)
				{
					// Compute integrated density matrix in Hilbert space via Sylvester/Lyapunov:
					// A_state X + X A_state^† = -rho0, with A_state = -i H - K
					arma::cx_mat rho0mat = B * B.st();
					arma::cx_mat A_dense = -arma::cx_double(0.0, 1.0) * arma::cx_mat(H) - arma::cx_mat(K);
					arma::cx_mat A_star = arma::conj(A_dense);

					// Solve (A_state^* ⊗ I + I ⊗ A_state) vec(X) = -vec(rho0)
					// This corresponds to A_state X + X A_state^† = -rho0.
					arma::cx_mat L = arma::kron(A_star, Iden_dense) + arma::kron(Iden_dense, A_dense);
					arma::cx_vec rhs = arma::vectorise(-rho0mat);
					arma::cx_vec sol = arma::solve(L, rhs);
					if (sol.is_empty())
					{
						this->Log() << "Failed to solve timeinf Lyapunov equation in Hilbert space." << std::endl;
						continue;
					}
					arma::cx_mat X = arma::reshape(sol, rho0mat.n_rows, rho0mat.n_cols);

					rho_integrated_partial[tid] += weight * X;
				}
			}

			if (method_timeevo)
			{
				for (auto &m : ExptValuesPartial)
				{
					ExptValues += m;
				}
			}
			arma::mat ExptValuesPulse;
			if (has_pulse_output)
			{
				ExptValuesPulse.zeros(pulse_times.size(), projection_counter);
				for (auto &m : ExptValuesPulsePartial)
				{
					ExptValuesPulse += m;
				}
			}
			if (method_timeinf)
			{
				for (auto &m : rho_integrated_partial)
				{
					rho_integrated += m;
				}
			}

			double time_offset = print_pulses ? pulse_total_time : 0.0;

			if (has_pulse_output && print_pulses)
			{
				if (integrate_pulses)
				{
					this->Log() << "Writing integrated polarisation during pulse sequence." << std::endl;
				}

				arma::mat integrated_pulse;
				if (integrate_pulses)
				{
					integrated_pulse.zeros(pulse_times.size(), projection_counter);
					for (size_t k = 1; k < pulse_times.size(); ++k)
					{
						double dt_pulse = pulse_dts[k];
						for (int idx = 0; idx < projection_counter; ++idx)
						{
							integrated_pulse(k, idx) = integrated_pulse(k - 1, idx) + dt_pulse * (ExptValuesPulse(k - 1, idx) + ExptValuesPulse(k, idx)) / 2.0;
						}
					}
					if (!pulse_times.empty())
					{
						integrated_pulse.row(0) = ExptValuesPulse.row(0);
					}
				}

				for (size_t k = 0; k < pulse_times.size(); ++k)
				{
					this->Data() << this->RunSettings()->CurrentStep() << " ";
					this->Data() << std::setprecision(12) << pulse_times[k] << " ";
					this->WriteStandardOutput(this->Data());

					for (int idx = 0; idx < projection_counter; ++idx)
					{
						if (integrate_pulses)
						{
							this->Data() << " " << integrated_pulse(k, idx);
						}
						else
						{
							this->Data() << " " << ExptValuesPulse(k, idx);
						}
					}
					this->Data() << std::endl;
				}
			}

			if (method_timeinf)
			{
				if (print_freeevo)
				{
					this->Log() << "Writing time-integrated (time -> inf) polarisation." << std::endl;

					this->Data() << this->RunSettings()->CurrentStep() << " ";
					this->Data() << "inf"
								 << " ";
					this->WriteStandardOutput(this->Data());

					for (int idx = 0; idx < projection_counter; idx++)
					{
						double val = use_sparse_ops ? (TraceSparseDense(OperatorsSparse[idx], rho_integrated) / Z)
													: (std::real(arma::trace(OperatorsDense[idx] * rho_integrated)) / Z);
						this->Data() << std::setprecision(12) << val << " ";
					}
					this->Data() << std::endl;
				}
			}
			else if (method_timeevo && print_freeevo)
			{
				if (integrate_freeevo)
				{
					this->Log() << "Writing integrated polarisation over time." << std::endl;

					arma::mat integrated;
					integrated.zeros(num_steps, projection_counter);

					for (int k = 1; k < num_steps; ++k)
					{
						for (int idx = 0; idx < projection_counter; ++idx)
						{
							integrated(k, idx) = integrated(k - 1, idx) + dt * (ExptValues(k - 1, idx) + ExptValues(k, idx)) / 2.0;
						}
					}

					if (num_steps > 0)
					{
						integrated.row(0) = ExptValues.row(0);
					}

					for (int k = 0; k < num_steps; k++)
					{
						this->Data() << this->RunSettings()->CurrentStep() << " ";
						this->Data() << std::setprecision(12) << time_offset + time(k) << " ";
						this->WriteStandardOutput(this->Data());

						for (int idx = 0; idx < projection_counter; idx++)
						{
							this->Data() << " " << integrated(k, idx);
						}
						this->Data() << std::endl;
					}
				}
				else
				{
					for (int k = 0; k < num_steps; k++)
					{
						// Write results
						this->Data() << this->RunSettings()->CurrentStep() << " ";
						this->Data() << std::setprecision(12) << time_offset + time(k) << " ";
						this->WriteStandardOutput(this->Data());

						for (int idx = 0; idx < projection_counter; idx++)
						{
							this->Data() << " " << ExptValues(k, idx);
						}
						this->Data() << std::endl;
					}
				}
			}

			this->Log() << "\nDone with SpinSystem \"" << (*i)->Name() << "\"" << std::endl;
		}
		this->Data() << std::endl;
		return true;
	}

	// Writes the header of the data file (but can also be passed to other streams)
	void TaskStaticHSDirectSpectra::WriteHeader(std::ostream &_stream)
	{
		_stream << "Step ";
		_stream << "Time ";
		this->WriteStandardOutputHeader(_stream);

		std::vector<std::string> spinList;
		bool CIDSP = false;
		int m;

		// Get header for each spin system
		auto systems = this->SpinSystems();
		for (auto i = systems.cbegin(); i != systems.cend(); i++)
		{

			if (this->Properties()->GetList("spinlist", spinList, ','))
			{
				for (auto l = (*i)->spins_cbegin(); l != (*i)->spins_cend(); l++)
				{
					std::string spintype;

					(*l)->Properties()->Get("type", spintype);

					for (m = 0; m < (int)spinList.size(); m++)
					{

						if ((*l)->Name() == spinList[m])
						{
							// Yields are written per transition
							// bool CIDSP = false;
							if (this->Properties()->Get("cidsp", CIDSP) && CIDSP == true)
							{
								// Write each transition name
								auto transitions = (*i)->Transitions();
								for (auto j = transitions.cbegin(); j != transitions.cend(); j++)
								{
									_stream << (*i)->Name() << "." << (*l)->Name() << "." << (*j)->Name() << ".yield"
											<< ".Ix ";
									_stream << (*i)->Name() << "." << (*l)->Name() << "." << (*j)->Name() << ".yield"
											<< ".Iy ";
									_stream << (*i)->Name() << "." << (*l)->Name() << "." << (*j)->Name() << ".yield"
											<< ".Iz ";
								}
							}
							else
							{
								// Write each state name
								auto states = (*i)->States();
								_stream << (*i)->Name() << "." << (*l)->Name() << ".Ix ";
								_stream << (*i)->Name() << "." << (*l)->Name() << ".Iy ";
								_stream << (*i)->Name() << "." << (*l)->Name() << ".Iz ";
							}
						}
					}
				}
			}
		}
		_stream << std::endl;
	}

	// Validation
	bool TaskStaticHSDirectSpectra::Validate()
	{
		// Get the reacton operator type
		std::string str;
		if (this->Properties()->Get("reactionoperators", str))
		{
			if (str.compare("haberkorn") == 0)
			{
				this->reactionOperators = SpinAPI::ReactionOperatorType::Haberkorn;
				this->Log() << "Setting reaction operator type to Haberkorn." << std::endl;
			}
			else if (str.compare("lindblad") == 0)
			{
				this->reactionOperators = SpinAPI::ReactionOperatorType::Lindblad;
				this->Log() << "Setting reaction operator type to Lindblad." << std::endl;
			}
			else
			{
				this->Log() << "Warning: Unknown reaction operator type specified. Using default reaction operators." << std::endl;
			}
		}

		return true;
	}

	bool TaskStaticHSDirectSpectra::CreateRotationMatrix(double &_alpha, double &_beta, double &_gamma, arma::mat &_R) const
	{
		arma::mat R1 = {
			{std::cos(_alpha), -std::sin(_alpha), 0.0},
			{std::sin(_alpha), std::cos(_alpha), 0.0},
			{0.0, 0.0, 1.0}};

		arma::mat R2 = {
			{std::cos(_beta), 0.0, std::sin(_beta)},
			{0.0, 1.0, 0.0},
			{-std::sin(_beta), 0.0, std::cos(_beta)}};

		arma::mat R3 = {
			{std::cos(_gamma), -std::sin(_gamma), 0.0},
			{std::sin(_gamma), std::cos(_gamma), 0.0},
			{0.0, 0.0, 1.0}};

		_R = R1 * R2 * R3;

		return true;
	}

	bool TaskStaticHSDirectSpectra::CreateUniformGrid(int &_Npoints, std::vector<std::tuple<double, double, double>> &_uniformGrid) const
	{
		std::vector<double> theta(_Npoints);
		std::vector<double> phi(_Npoints);
		std::vector<double> weight(_Npoints);

		_uniformGrid.resize(_Npoints);

		const double golden = arma::datum::pi * (1.0 + std::sqrt(5.0)); // not standard golden angle

		for (int i = 0; i < _Npoints; ++i)
		{
			double index = static_cast<double>(i) + 0.5;

			theta[i] = std::acos(1.0 - index / _Npoints);							// hemisphere
			phi[i] = golden * index;												// hemisphere
			weight[i] = std::sin(theta[i]) * 2 * arma::datum::pi / _Npoints; // 2 * pi for hemisphere
			_uniformGrid[i] = {theta[i], phi[i], weight[i]};
		}

		return true;
	}

}
