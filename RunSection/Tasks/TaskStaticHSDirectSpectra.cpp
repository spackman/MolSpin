/////////////////////////////////////////////////////////////////////////
// TaskStaticHSDirectSpectra implementation (RunSection module) by Luca Gerhards
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include <algorithm>
#include <cctype>
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
		enum class OutputComponent
		{
			X,
			Y,
			Z,
			P,
			M
		};

		struct SpinOperatorSet
		{
			SpinAPI::spin_ptr spin;
			arma::sp_cx_mat Sx;
			arma::sp_cx_mat Sy;
			arma::sp_cx_mat Sz;
			arma::sp_cx_mat Sp;
			arma::sp_cx_mat Sm;
		};

		struct OutputOpDescriptor
		{
			size_t spin_index = 0;
			OutputComponent component = OutputComponent::X;
			arma::sp_cx_mat projector;
			double scale = 1.0;
			bool use_projector = false;
		};

		arma::mat PassiveZXZRotation(const arma::vec &framelist)
		{
			arma::mat RFrame = arma::eye<arma::mat>(3, 3);
			double a = (framelist.n_elem >= 1) ? framelist(0) : 0.0;
			double b = (framelist.n_elem >= 2) ? framelist(1) : 0.0;
			double g = (framelist.n_elem >= 3) ? framelist(2) : 0.0;

			// Passive ZXZ Euler rotation: R = Rz(gamma) * Ry(beta) * Rz(alpha)
			const double ca = std::cos(a), sa = std::sin(a);
			const double cb = std::cos(b), sb = std::sin(b);
			const double cg = std::cos(g), sg = std::sin(g);

			arma::mat Ra = {{ca, sa, 0.0}, {-sa, ca, 0.0}, {0.0, 0.0, 1.0}};
			arma::mat Rb = {{cb, 0.0, -sb}, {0.0, 1.0, 0.0}, {sb, 0.0, cb}};
			arma::mat Rg = {{cg, sg, 0.0}, {-sg, cg, 0.0}, {0.0, 0.0, 1.0}};
			RFrame = Rg * Rb * Ra;
			return RFrame;
		}

		double TraceSparseDense(const arma::sp_cx_mat &A, const arma::cx_mat &B)
		{
			arma::cx_double sum = arma::cx_double(0.0, 0.0);
			for (auto it = A.begin(); it != A.end(); ++it)
			{
				sum += (*it) * B(it.col(), it.row());
			}
			return std::real(sum);
		}

		double TraceDenseTransposed(const arma::cx_mat &A, const arma::cx_mat &B_t)
		{
			return std::real(arma::accu(A % B_t));
		}

		bool ResolveFieldInteraction(const SpinAPI::system_ptr &system, const std::string &fieldInteractionName, SpinAPI::interaction_ptr &fieldInteraction)
		{
			fieldInteraction = nullptr;
			if (system == nullptr)
				return false;

			if (!fieldInteractionName.empty())
				fieldInteraction = system->interactions_find(fieldInteractionName);

			if (fieldInteraction == nullptr)
			{
				for (auto inter = system->interactions_cbegin(); inter != system->interactions_cend(); inter++)
				{
					std::string type;
					if ((*inter)->Properties()->Get("type", type))
					{
						std::transform(type.begin(), type.end(), type.begin(), [](unsigned char c)
									   { return static_cast<char>(std::tolower(c)); });
						if (type == "zeeman")
						{
							fieldInteraction = (*inter);
							break;
						}
					}
				}
			}

			return (fieldInteraction != nullptr);
		}

		bool ResolveSpinsByName(const SpinAPI::system_ptr &system, const std::vector<std::string> &names, std::vector<SpinAPI::spin_ptr> &spins)
		{
			spins.clear();
			if (system == nullptr)
				return false;
			for (const auto &name : names)
			{
				auto spin = system->spins_find(name);
				if (spin == nullptr)
					return false;
				bool exists = false;
				for (const auto &existing : spins)
				{
					if (existing == spin)
					{
						exists = true;
						break;
					}
				}
				if (!exists)
					spins.push_back(spin);
			}
			return true;
		}

		[[maybe_unused]] void SecularizeHamiltonian(arma::sp_cx_mat &H, const arma::vec &mz_diag, double tol)
		{
			if (H.n_rows != H.n_cols || H.n_rows != mz_diag.n_elem)
				return;

			if (!std::isfinite(tol) || tol < 0.0)
				tol = 0.0;

			arma::sp_cx_mat filtered = arma::zeros<arma::sp_cx_mat>(H.n_rows, H.n_cols);
			for (auto it = H.begin(); it != H.end(); ++it)
			{
				if (std::abs(mz_diag(it.row()) - mz_diag(it.col())) <= tol)
					filtered(it.row(), it.col()) = (*it);
			}
			H = std::move(filtered);
		}

		void SecularizeHamiltonianEigenbasis(arma::sp_cx_mat &H, const arma::vec &energies, const arma::cx_mat &eigvec, double tol)
		{
			if (H.n_rows != H.n_cols || H.n_rows != energies.n_elem || eigvec.n_rows != H.n_rows || eigvec.n_cols != H.n_rows)
				return;

			if (!std::isfinite(tol) || tol < 0.0)
				tol = 0.0;

			arma::cx_mat H_dense = arma::cx_mat(H);
			H_dense = 0.5 * (H_dense + H_dense.st());
			const arma::cx_mat Udag = arma::trans(arma::conj(eigvec));
			arma::cx_mat H_eig = Udag * H_dense * eigvec;

			const arma::uword dim = energies.n_elem;
			for (arma::uword r = 0; r < dim; ++r)
			{
				for (arma::uword c = 0; c < dim; ++c)
				{
					if (std::abs(energies(r) - energies(c)) > tol)
						H_eig(r, c) = arma::cx_double(0.0, 0.0);
				}
			}

			H_dense = eigvec * H_eig * Udag;
			H_dense = 0.5 * (H_dense + H_dense.st());
			H = arma::sp_cx_mat(H_dense);
		}
	}

	// -----------------------------------------------------
	// TaskStaticHSDirectSpectra Constructors and Destructor
	// -----------------------------------------------------
	TaskStaticHSDirectSpectra::TaskStaticHSDirectSpectra(const MSDParser::ObjectParser &_parser, const RunSection &_runsection)
		: BasicTask(_parser, _runsection),
		  timestep(1.0),
		  totaltime(1.0e+4),
		  mwFrequencyGHz(0.0),
		  secularTolerance(1e-6),
		  powderGammaPoints(1),
		  powderFullSphere(true),
		  fullTensorRotation(true),
		  rwaEnabled(false),
		  secularizeInteractions(false),
		  fieldInteractionName(""),
		  rwaSpinNames(),
		  reactionOperators(SpinAPI::ReactionOperatorType::Haberkorn)
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
		this->Log() << "Running task StaticHS-Direct-Spectra." << std::endl;

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
						this->Log() << "Skipping SpinSystem \"" << (*i)->Name()
									<< "\" because electron spins must be spin 1/2 (multiplicity 2). Found multiplicity "
									<< (*l)->Multiplicity() << "." << std::endl;
						return 1;
					}
				}
			}

			this->Log() << "\nStarting with SpinSystem \"" << (*i)->Name() << "\"." << std::endl;

			// Obtain a SpinSpace to describe the system
			SpinAPI::SpinSpace space(*(*i));
			space.UseSuperoperatorSpace(false);

			this->Properties()->Get("powderfullsphere", this->powderFullSphere);
			this->Properties()->Get("fulltensorrotation", this->fullTensorRotation);

			space.UseFullTensorRotation(this->fullTensorRotation);
			space.SetReactionOperatorType(this->reactionOperators);
			if (this->fullTensorRotation)
			{
				this->Log() << "Full tensor rotation enabled (off-diagonal terms retained)." << std::endl;
			}

			std::string InitialState;
			arma::cx_mat InitialStateVector;
			if (this->Properties()->Get("initialstate", InitialState))
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
					this->Log() << "Invalid initial state value \"" << InitialState << "\". Using Singlet state." << std::endl;
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
				this->Log() << "Failed to obtain input for CIDSP. Using default false." << std::endl;
			}

			// Get projectors of interest of the spectrum
			std::vector<std::string> spinList;
			const bool hasSpinList = this->Properties()->GetList("spinlist", spinList, ',');
			int m;

			// Check transitions, rates and projection operators
			auto transitions = (*i)->Transitions();
			arma::sp_cx_mat P;

			std::vector<SpinOperatorSet> detect_ops;
			std::vector<arma::mat> detect_g_frame_base;
			std::vector<OutputOpDescriptor> output_desc;

			const bool rwa_enabled = this->rwaEnabled;
			const bool secularize = this->secularizeInteractions;
			const bool need_field_interaction = hasSpinList || rwa_enabled || secularize;

			SpinAPI::interaction_ptr fieldInteraction = nullptr;
			bool have_field_interaction = false;
			arma::mat RFrame = arma::eye<arma::mat>(3, 3);
			arma::vec Bvec;
			double Bmag = 0.0;
			double mu_prefactor = 1.0;
			if (need_field_interaction)
			{
				have_field_interaction = ResolveFieldInteraction((*i), this->fieldInteractionName, fieldInteraction);
				if (!have_field_interaction)
				{
					this->Log() << "Failed to resolve Zeeman interaction in SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
				}
				else
				{
					RFrame = PassiveZXZRotation(fieldInteraction->Framelist());
					Bvec = fieldInteraction->Field();
					if (Bvec.n_elem == 3)
					{
						Bmag = arma::norm(Bvec);
					}
					mu_prefactor = fieldInteraction->Prefactor();
					if (fieldInteraction->AddCommonPrefactor())
						mu_prefactor *= 8.79410005e+1;
				}
			}

			// Getting the projection operators
			if (hasSpinList)
			{
				for (auto l = (*i)->spins_cbegin(); l != (*i)->spins_cend(); l++)
				{
					for (m = 0; m < (int)spinList.size(); m++)
					{
						if ((*l)->Name() == spinList[m])
						{
							SpinOperatorSet ops;
							ops.spin = (*l);
							if (!space.CreateOperator(arma::conv_to<arma::sp_cx_mat>::from((*l)->Sx()), (*l), ops.Sx))
								return false;
							if (!space.CreateOperator(arma::conv_to<arma::sp_cx_mat>::from((*l)->Sy()), (*l), ops.Sy))
								return false;
							if (!space.CreateOperator(arma::conv_to<arma::sp_cx_mat>::from((*l)->Sz()), (*l), ops.Sz))
								return false;
							if (!space.CreateOperator(arma::conv_to<arma::sp_cx_mat>::from((*l)->Sp()), (*l), ops.Sp))
								return false;
							if (!space.CreateOperator(arma::conv_to<arma::sp_cx_mat>::from((*l)->Sm()), (*l), ops.Sm))
								return false;

							detect_ops.push_back(ops);
							arma::mat g_frame = arma::eye<arma::mat>(3, 3);
							if (have_field_interaction && fieldInteraction != nullptr)
							{
								arma::mat g_base = arma::conv_to<arma::mat>::from((*l)->GetTensor().LabFrame());
								if (fieldInteraction->IgnoreTensors())
									g_base = arma::eye<arma::mat>(3, 3);
								g_frame = RFrame * g_base * RFrame.t();
							}
							detect_g_frame_base.push_back(g_frame);

							const size_t spin_index = detect_ops.size() - 1;
							if (CIDSP == true)
							{
								// Gather rates and operators
								for (auto j = transitions.cbegin(); j != transitions.cend(); j++)
								{
									if ((*j)->SourceState() == nullptr)
										continue;
									if (!space.GetState((*j)->SourceState(), P))
									{
										this->Log() << "Failed to obtain projection matrix onto state \"" << (*j)->Name() << "\" of SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
										return false;
									}

									OutputOpDescriptor desc;
									desc.spin_index = spin_index;
									desc.use_projector = true;
									desc.projector = P;
									desc.scale = (*j)->Rate();
									desc.component = OutputComponent::X;
									output_desc.push_back(desc);
									desc.component = OutputComponent::Y;
									output_desc.push_back(desc);
									desc.component = OutputComponent::Z;
									output_desc.push_back(desc);
									desc.component = OutputComponent::P;
									output_desc.push_back(desc);
									desc.component = OutputComponent::M;
									output_desc.push_back(desc);
								}
							}
							else
							{
								OutputOpDescriptor desc;
								desc.spin_index = spin_index;
								desc.use_projector = false;
								desc.scale = 1.0;
								desc.component = OutputComponent::X;
								output_desc.push_back(desc);
								desc.component = OutputComponent::Y;
								output_desc.push_back(desc);
								desc.component = OutputComponent::Z;
								output_desc.push_back(desc);
								desc.component = OutputComponent::P;
								output_desc.push_back(desc);
								desc.component = OutputComponent::M;
								output_desc.push_back(desc);
							}
						}
					}
				}
			}

			const int projection_counter = static_cast<int>(output_desc.size());

			const bool have_valid_field = have_field_interaction && (Bvec.n_elem == 3) && std::isfinite(Bmag) && (Bmag > 0.0);
			std::vector<SpinAPI::spin_ptr> rwaSpins;
			std::vector<SpinOperatorSet> rwa_ops;
			std::vector<arma::mat> rwa_g_frame_base;
			std::vector<std::string> zeelist;

			if (rwa_enabled || secularize)
			{
				if (!have_field_interaction || fieldInteraction == nullptr)
				{
					this->Log() << "Rotating-frame handling requires a Zeeman interaction in SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
					if (rwa_enabled)
						continue;
				}
				else
				{
					if (!this->rwaSpinNames.empty())
					{
						if (!ResolveSpinsByName((*i), this->rwaSpinNames, rwaSpins))
						{
							this->Log() << "Failed to resolve rwaspins list in SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
						}
					}
					else if (!fieldInteraction->Group1().empty())
					{
						rwaSpins = fieldInteraction->Group1();
					}
					else if (!spinList.empty())
					{
						if (!ResolveSpinsByName((*i), spinList, rwaSpins))
						{
							this->Log() << "Failed to resolve spinlist for rotating-frame handling in SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
						}
					}
					else
					{
						auto allSpins = (*i)->Spins();
						rwaSpins.assign(allSpins.begin(), allSpins.end());
					}

					if (rwaSpins.empty())
					{
						this->Log() << "No spins available for rotating-frame handling in SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
						if (rwa_enabled)
							continue;
					}
					else
					{
						zeelist.push_back(fieldInteraction->Name());
						for (const auto &spin : rwaSpins)
						{
							SpinOperatorSet ops;
							ops.spin = spin;
							if (!space.CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(spin->Sx()), spin, ops.Sx))
							{
								this->Log() << "Failed to build Sx operator for rotating-frame spin \"" << spin->Name() << "\"." << std::endl;
								rwaSpins.clear();
								break;
							}
							if (!space.CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(spin->Sy()), spin, ops.Sy))
							{
								this->Log() << "Failed to build Sy operator for rotating-frame spin \"" << spin->Name() << "\"." << std::endl;
								rwaSpins.clear();
								break;
							}
							if (!space.CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(spin->Sz()), spin, ops.Sz))
							{
								this->Log() << "Failed to build Sz operator for rotating-frame spin \"" << spin->Name() << "\"." << std::endl;
								rwaSpins.clear();
								break;
							}
							rwa_ops.push_back(ops);

							arma::mat g_frame = arma::eye<arma::mat>(3, 3);
							arma::mat g_base = arma::conv_to<arma::mat>::from(spin->GetTensor().LabFrame());
							if (fieldInteraction->IgnoreTensors())
								g_base = arma::eye<arma::mat>(3, 3);
							g_frame = RFrame * g_base * RFrame.t();
							rwa_g_frame_base.push_back(g_frame);
						}
					}
				}

				if ((rwa_enabled || secularize) && !have_valid_field)
				{
					this->Log() << "Rotating-frame handling requires a valid field vector in SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
					if (rwa_enabled)
						continue;
				}
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

			SpinAPI::HilbertRelaxationCache relaxation_cache;
			bool use_density_matrix = false;
			for (auto j = (*i)->operators_cbegin(); j != (*i)->operators_cend(); j++)
			{
				if (space.RelaxationOperator((*j), relaxation_cache))
				{
					use_density_matrix = true;
					this->Log() << "Added relaxation operator \"" << (*j)->Name() << "\" to Hilbert-space propagation.\n";
				}
			}
			if (use_density_matrix)
			{
				this->Log() << "Relaxation operators detected. Using density-matrix propagation in Hilbert space." << std::endl;
			}

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
					this->Log() << "WARNING: Undefined timestep. Using default 0.1 ns." << std::endl;
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
				this->Log() << "Changing number of propagation steps to: " << num_steps << " in order to propagate one step." << std::endl;
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
					this->Log() << "Undefined precision for autoexpm method. Using single precision." << std::endl;
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
						this->Log() << "Undefined tolerance for the krylov subspace. Using the default of 1e-16." << std::endl;
						krylovtol = 1e-16;
					}
				}
				else
				{
					this->Log() << "Undefined size of the krylov subspace. Using the default size of 16." << std::endl;
					krylovsize = 16;
					if (krylovtol > 0)
					{
						this->Log() << "Tolerance for krylov propagation is chosen as " << krylovtol << "." << std::endl;
					}
					else
					{
						this->Log() << "Undefined tolerance for the krylov subspace. Using the default of 1e-16." << std::endl;
						krylovtol = 1e-16;
					}
				}
			}
			else if (propmethod == "rk4" || propmethod == "explicit")
			{
				this->Log() << "Explicit RK4 is chosen as the propagation method." << std::endl;
			}
			else
			{
				this->Log() << "WARNING: Undefined propagation method. Using normal exponential method." << std::endl;
				propmethod = "normal";
			}

			bool relax_use_split_expm = false;
			arma::cx_mat K_dense;
			if (use_density_matrix)
			{
				relax_use_split_expm = (propmethod != "rk4" && propmethod != "explicit");
				if (relax_use_split_expm)
				{
					K_dense = arma::cx_mat(K);
					this->Log() << "Relaxation operators active; using split-exponential propagation (Hamiltonian expm + RK4 relaxation)." << std::endl;
					if (propmethod != "normal")
					{
						this->Log() << "Note: propagationmethod is ignored for relaxation; use propagationmethod = rk4 to force explicit RK4." << std::endl;
					}
				}
				else
				{
					this->Log() << "Relaxation operators active; using explicit RK4 density-matrix propagation." << std::endl;
				}
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
				this->Log() << "Failed to obtain input for integration. Please use integration = true/false. Using integration = false by default." << std::endl;
			}

			// Read integrationwindow from the input file
			std::string Integrationwindow;
			if (!this->Properties()->Get("integrationtimeframe", Integrationwindow))
			{
				this->Log() << "Failed to obtain input for integrationtimeframe. Please choose integrationtimeframe = pulse / freeevo / full. Using freeevo by default." << std::endl;
				Integrationwindow = "freeevo";
			}
			this->Log() << "Timewindow for the propagation integration: " << Integrationwindow << std::endl;

			// Read printtimeframe from the input file
			std::string Timewindow;
			if (!this->Properties()->Get("printtimeframe", Timewindow))
			{
				this->Log() << "Failed to obtain input for printtimeframe. Please choose printtimeframe = pulse / freeevo / full. Using full by default." << std::endl;
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
				if (this->powderGammaPoints > 1)
				{
					this->Log() << "Sampling gamma with " << this->powderGammaPoints << " points per orientation." << std::endl;
				}
				if (this->powderFullSphere)
				{
					this->Log() << "Using full-sphere powder grid." << std::endl;
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
			std::vector<std::string> HamiltonianAllList;
			if (!hasH0list)
			{
				for (auto inter = (*i)->interactions_cbegin(); inter != (*i)->interactions_cend(); ++inter)
				{
					HamiltonianAllList.push_back((*inter)->Name());
				}
			}

			// Read a pulse sequence from the input
			std::vector<std::tuple<std::string, double>> Pulsesequence;
			bool hasPulseSequence = this->Properties()->GetPulseSequence("pulsesequence", Pulsesequence);
			if (hasPulseSequence)
			{
				this->Log() << "Pulse sequence:" << std::endl;
			}

			double omega_mw = 0.0;
			if (rwa_enabled)
			{
				if (this->mwFrequencyGHz > 0.0)
				{
					omega_mw = 2.0 * arma::datum::pi * this->mwFrequencyGHz;
				}
				else
				{
					for (auto pulse = (*i)->pulses_cbegin(); pulse < (*i)->pulses_cend(); pulse++)
					{
						if ((*pulse)->Type() == SpinAPI::PulseType::LongPulse)
						{
							double freq = (*pulse)->Frequency();
							if (std::isfinite(freq) && freq > 0.0)
							{
								omega_mw = freq;
								this->Log() << "Using pulse frequency " << freq << " rad/ns for rotating-frame shift." << std::endl;
								break;
							}
						}
					}
				}

				if (!(omega_mw > 0.0))
				{
					this->Log() << "Rotating-wave approximation requires mwfrequency (rad/ns) or a LongPulse frequency." << std::endl;
					return false;
				}
				this->Log() << "Rotating-wave approximation enabled (omega_mw = " << omega_mw << " rad/ns)." << std::endl;
				if (secularize)
				{
					this->Log() << "Secularizing interactions with energy tolerance " << this->secularTolerance << " rad/ns." << std::endl;
				}
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
			arma::vec ExptValuesTimeinf;
			if (method_timeinf)
			{
				ExptValuesTimeinf.zeros(projection_counter);
			}

			size_t grid_size = grid.size();
			const int gamma_points = (grid_size > 1) ? std::max(1, this->powderGammaPoints) : 1;
			const double gamma_weight = 1.0 / static_cast<double>(gamma_points);
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

			std::vector<arma::vec> ExptValuesTimeinfPartial;
			if (method_timeinf)
			{
				ExptValuesTimeinfPartial.resize(nthreads);
				for (auto &m : ExptValuesTimeinfPartial)
				{
					m.zeros(projection_counter);
				}
			}

			arma::cx_mat Iden_dense;
			if (method_timeinf)
			{
				int dim = InitialStateVector.n_rows * Z;
				Iden_dense = arma::eye<arma::cx_mat>(dim, dim);
			}

			arma::cx_mat relaxation_super;
			if (method_timeinf && use_density_matrix)
			{
				space.RelaxationSuperoperatorHilbert(relaxation_cache, relaxation_super);
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

				const double base_weight = weight;

				for (int gamma_idx = 0; gamma_idx < gamma_points; ++gamma_idx)
				{
					double gamma = 0.0;
					if (gamma_points > 1)
					{
						gamma = 2.0 * arma::datum::pi * (static_cast<double>(gamma_idx) + 0.5) / static_cast<double>(gamma_points);
					}
					double weight = base_weight * gamma_weight;

					arma::mat Rot_mat = arma::eye<arma::mat>(3, 3);
					if (!this->CreateRotationMatrix(phi, theta, gamma, Rot_mat))
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
					}
					else
					{
						if (!space_thread.BaseHamiltonianRotated(HamiltonianAllList, Rot_mat, H))
						{
							this->Log() << "Failed to obtain rotated Hamiltonian for SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
							continue;
						}
					}

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

					const arma::mat Rpowder = Rot_mat.t();
					std::vector<arma::sp_cx_mat> OperatorsSparseLocal;
					std::vector<arma::cx_mat> OperatorsDenseLocal;
					bool use_sparse_ops = false;

					if (projection_counter > 0)
					{
						const arma::cx_double imag_unit(0.0, 1.0);
						std::vector<arma::sp_cx_mat> mu_x(detect_ops.size());
						std::vector<arma::sp_cx_mat> mu_y(detect_ops.size());
						std::vector<arma::sp_cx_mat> mu_z(detect_ops.size());
						std::vector<arma::sp_cx_mat> mu_p(detect_ops.size());
						std::vector<arma::sp_cx_mat> mu_m(detect_ops.size());
						bool tensor_dim_ok = true;

						for (size_t spin_idx = 0; spin_idx < detect_ops.size(); ++spin_idx)
						{
							arma::mat g = Rpowder * detect_g_frame_base[spin_idx] * Rpowder.t();
							if (!this->fullTensorRotation)
								g = g % arma::eye<arma::mat>(3, 3);

							const auto &ops = detect_ops[spin_idx];
							arma::sp_cx_mat mux = g(0, 0) * ops.Sx + g(1, 0) * ops.Sy + g(2, 0) * ops.Sz;
							arma::sp_cx_mat muy = g(0, 1) * ops.Sx + g(1, 1) * ops.Sy + g(2, 1) * ops.Sz;
							arma::sp_cx_mat muz = g(0, 2) * ops.Sx + g(1, 2) * ops.Sy + g(2, 2) * ops.Sz;
							if (mu_prefactor != 1.0)
							{
								mux *= mu_prefactor;
								muy *= mu_prefactor;
								muz *= mu_prefactor;
							}
							if (mux.n_rows != H.n_rows || mux.n_cols != H.n_cols)
							{
								tensor_dim_ok = false;
								break;
							}
							mu_x[spin_idx] = std::move(mux);
							mu_y[spin_idx] = std::move(muy);
							mu_z[spin_idx] = std::move(muz);
							mu_p[spin_idx] = mu_x[spin_idx] + imag_unit * mu_y[spin_idx];
							mu_m[spin_idx] = mu_x[spin_idx] - imag_unit * mu_y[spin_idx];
						}

						if (!tensor_dim_ok)
						{
							this->Log() << "Magnetic moment operator size mismatch in SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
							continue;
						}

						OperatorsSparseLocal.resize(projection_counter);
						for (int op_idx = 0; op_idx < projection_counter; ++op_idx)
						{
							const auto &desc = output_desc[op_idx];
							arma::sp_cx_mat op;
							switch (desc.component)
							{
							case OutputComponent::X:
								op = mu_x[desc.spin_index];
								break;
							case OutputComponent::Y:
								op = mu_y[desc.spin_index];
								break;
							case OutputComponent::Z:
								op = mu_z[desc.spin_index];
								break;
							case OutputComponent::P:
								op = mu_p[desc.spin_index];
								break;
							case OutputComponent::M:
								op = mu_m[desc.spin_index];
								break;
							}

							if (desc.scale != 1.0)
								op *= desc.scale;
							if (desc.use_projector)
								op = op * desc.projector;
							OperatorsSparseLocal[op_idx] = std::move(op);
						}

						double total_nnz = 0.0;
						double total_size = 0.0;
						for (const auto &op : OperatorsSparseLocal)
						{
							total_nnz += static_cast<double>(op.n_nonzero);
							total_size += static_cast<double>(op.n_rows) * op.n_cols;
						}
						use_sparse_ops = (total_size > 0.0) && ((total_nnz / total_size) < 0.1);
						if (!use_sparse_ops)
						{
							OperatorsDenseLocal.resize(projection_counter);
							for (int op_idx = 0; op_idx < projection_counter; ++op_idx)
							{
								OperatorsDenseLocal[op_idx] = arma::cx_mat(OperatorsSparseLocal[op_idx]);
							}
						}
						else if (grid_num == 0 && gamma_idx == 0)
						{
							this->Log() << "Using sparse operators for expectation values." << std::endl;
						}
					}

					if (rwa_enabled || secularize)
					{
						if (secularize && !zeelist.empty())
						{
							arma::sp_cx_mat Hz_sp;
							if (!space_thread.BaseHamiltonianRotated(zeelist, Rot_mat, Hz_sp))
							{
								this->Log() << "Failed to obtain Zeeman Hamiltonian for secularization in SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
							}
							else
							{
								arma::cx_mat Hz_dense = arma::cx_mat(Hz_sp);
								Hz_dense = 0.5 * (Hz_dense + Hz_dense.st());
								arma::vec eigval;
								arma::cx_mat eigvec;
								if (!arma::eig_sym(eigval, eigvec, Hz_dense))
								{
									this->Log() << "Failed to diagonalize Zeeman Hamiltonian for secularization in SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
								}
								else
								{
									SecularizeHamiltonianEigenbasis(H, eigval, eigvec, this->secularTolerance);
								}
							}
						}

						if (rwa_enabled)
						{
							arma::sp_cx_mat S_eff_total = arma::zeros<arma::sp_cx_mat>(H.n_rows, H.n_cols);
							bool have_frame = false;
							for (size_t idx = 0; idx < rwa_ops.size(); ++idx)
							{
								arma::mat g = Rpowder * rwa_g_frame_base[idx] * Rpowder.t();
								if (!this->fullTensorRotation)
									g = g % arma::eye<arma::mat>(3, 3);

								arma::vec Beff = g * Bvec;
								double geff = arma::norm(Beff);
								if (!std::isfinite(geff) || geff <= 0.0)
									continue;

								const double bx = Beff(0) / geff;
								const double by = Beff(1) / geff;
								const double bz = Beff(2) / geff;
								S_eff_total += bx * rwa_ops[idx].Sx + by * rwa_ops[idx].Sy + bz * rwa_ops[idx].Sz;
								have_frame = true;
							}

							if (!have_frame)
							{
								this->Log() << "Rotating-frame operator could not be constructed for SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
								continue;
							}
							if (S_eff_total.n_rows != H.n_rows || S_eff_total.n_cols != H.n_cols)
							{
								this->Log() << "Rotating-frame operator size mismatch in SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
								continue;
							}

							H -= omega_mw * S_eff_total;
						}
					}

					if (use_density_matrix)
					{
						arma::cx_mat rho = Binitial * Binitial.st();
						int dim = static_cast<int>(rho.n_rows);

						arma::cx_mat work_left(dim, dim, arma::fill::zeros);
						arma::cx_mat work_right(dim, dim, arma::fill::zeros);
						arma::cx_mat relax(dim, dim, arma::fill::zeros);
						arma::cx_mat k1(dim, dim, arma::fill::zeros);
						arma::cx_mat k2(dim, dim, arma::fill::zeros);
						arma::cx_mat k3(dim, dim, arma::fill::zeros);
						arma::cx_mat k4(dim, dim, arma::fill::zeros);
						arma::cx_mat tmp_state(dim, dim, arma::fill::zeros);
						arma::cx_mat rk_accum(dim, dim, arma::fill::zeros);

						bool use_dense_H = false;
						arma::cx_mat H_dense;
						double H_density = 0.0;
						if (relax_use_split_expm)
						{
							H_dense = arma::cx_mat(H);
							use_dense_H = true;
						}
						else if (H.n_rows > 0 && H.n_cols > 0)
						{
							H_density = static_cast<double>(H.n_nonzero) / (static_cast<double>(H.n_rows) * static_cast<double>(H.n_cols));
							if (H_density > 0.15)
							{
								H_dense = arma::cx_mat(H);
								use_dense_H = true;
							}
						}

						const arma::cx_double imag_unit(0.0, 1.0);

						auto record_expectation_rho = [&](arma::mat &target, size_t row_index, const arma::cx_mat &state)
						{
							arma::cx_mat state_t = state.t();
							for (int idx = 0; idx < projection_counter; ++idx)
							{
								double val = use_sparse_ops ? (TraceSparseDense(OperatorsSparseLocal[idx], state) / Z)
															: (TraceDenseTransposed(OperatorsDenseLocal[idx], state_t) / Z);
								target(row_index, idx) = val;
							}
						};

						auto drho = [&](const arma::cx_mat &state, const arma::sp_cx_mat &H_total, const arma::cx_mat *H_dense_ptr, arma::cx_mat &out)
						{
							if (H_dense_ptr != nullptr)
							{
								work_left = (*H_dense_ptr) * state;
								work_right = state * (*H_dense_ptr);
							}
							else
							{
								work_left = H_total * state;
								work_right = state * H_total;
							}
							out = -imag_unit * (work_left - work_right);
							work_left = K * state;
							work_right = state * K;
							out -= (work_left + work_right);
							space_thread.ApplyRelaxationHilbert(relaxation_cache, state, relax);
							out += relax;
						};

						auto rk4_step = [&](arma::cx_mat &state, const arma::sp_cx_mat &H_total, const arma::cx_mat *H_dense_ptr, double step_dt)
						{
							drho(state, H_total, H_dense_ptr, k1);
							tmp_state = state;
							tmp_state += (0.5 * step_dt) * k1;
							drho(tmp_state, H_total, H_dense_ptr, k2);
							tmp_state = state;
							tmp_state += (0.5 * step_dt) * k2;
							drho(tmp_state, H_total, H_dense_ptr, k3);
							tmp_state = state;
							tmp_state += step_dt * k3;
							drho(tmp_state, H_total, H_dense_ptr, k4);
							rk_accum = k1;
							rk_accum += 2.0 * k2;
							rk_accum += 2.0 * k3;
							rk_accum += k4;
							state += (step_dt / 6.0) * rk_accum;
						};

						auto relax_rhs = [&](const arma::cx_mat &state, arma::cx_mat &out)
						{
							space_thread.ApplyRelaxationHilbert(relaxation_cache, state, out);
						};

						auto rk4_relax_step = [&](arma::cx_mat &state, double step_dt)
						{
							relax_rhs(state, k1);
							tmp_state = state;
							tmp_state += (0.5 * step_dt) * k1;
							relax_rhs(tmp_state, k2);
							tmp_state = state;
							tmp_state += (0.5 * step_dt) * k2;
							relax_rhs(tmp_state, k3);
							tmp_state = state;
							tmp_state += step_dt * k3;
							relax_rhs(tmp_state, k4);
							rk_accum = k1;
							rk_accum += 2.0 * k2;
							rk_accum += 2.0 * k3;
							rk_accum += k4;
							state += (step_dt / 6.0) * rk_accum;
						};

						auto build_unitary_half = [&](const arma::cx_mat &H_total_dense, double step_dt, arma::cx_mat &U_half, arma::cx_mat &U_half_st)
						{
							arma::cx_mat A_dense = -imag_unit * H_total_dense - K_dense;
							U_half = arma::expmat(A_dense * (0.5 * step_dt));
							U_half_st = U_half.st();
						};

						auto apply_unitary_half = [&](arma::cx_mat &state, const arma::cx_mat &U_half, const arma::cx_mat &U_half_st)
						{
							work_left = U_half * state;
							state = work_left * U_half_st;
						};

						auto split_step = [&](arma::cx_mat &state, const arma::cx_mat &U_half, const arma::cx_mat &U_half_st, double step_dt)
						{
							apply_unitary_half(state, U_half, U_half_st);
							rk4_relax_step(state, step_dt);
							apply_unitary_half(state, U_half, U_half_st);
						};

						size_t pulse_step_index = 0;
						arma::mat ExptValuesPulseOrientation;
						if (has_pulse_output)
						{
							ExptValuesPulseOrientation.zeros(pulse_times.size(), projection_counter);
							if (pulse_has_initial_step)
							{
								record_expectation_rho(ExptValuesPulseOrientation, pulse_step_index, rho);
								pulse_step_index = 1;
							}
						}

						if (hasPulseSequence)
						{
							for (const auto &seq : Pulsesequence)
							{
								if (grid_num == 0 && gamma_idx == 0)
								{
									this->Log() << std::get<0>(seq) << ", " << std::get<1>(seq) << std::endl;
								}

								std::string pulse_name = std::get<0>(seq);
								double timerelaxation = std::get<1>(seq);

								for (auto pulse = (*i)->pulses_cbegin(); pulse < (*i)->pulses_cend(); pulse++)
								{
									if ((*pulse)->Name().compare(pulse_name) == 0)
									{
										double pulse_dt = (*pulse)->Timestep();
										if (!std::isfinite(pulse_dt) || pulse_dt <= 0.0)
										{
											this->Log() << "Invalid timestep for pulse \"" << (*pulse)->Name() << "\". Skipping pulse propagation." << std::endl;
											continue;
										}

										if ((*pulse)->Type() == SpinAPI::PulseType::InstantPulse)
										{
											arma::sp_cx_mat pulse_operator;
											if (!space_thread.PulseOperatorOnStatevector((*pulse), pulse_operator))
											{
												this->Log() << "Failed to create a pulse operator in HS." << std::endl;
												continue;
											}
											arma::cx_mat U = arma::cx_mat(pulse_operator);
											rho = U * rho * U.st();
										}
										else if ((*pulse)->Type() == SpinAPI::PulseType::LongPulseStaticField)
										{
											arma::sp_cx_mat pulse_operator;
											if (!space_thread.PulseOperatorOnStatevector((*pulse), pulse_operator))
											{
												this->Log() << "Failed to create a pulse operator in HS." << std::endl;
												continue;
											}

											unsigned int steps = static_cast<unsigned int>(std::abs((*pulse)->Pulsetime() / pulse_dt));
											if (relax_use_split_expm)
											{
												arma::cx_mat H_pulse_dense = H_dense + arma::cx_mat(pulse_operator);
												arma::cx_mat U_half;
												arma::cx_mat U_half_st;
												build_unitary_half(H_pulse_dense, pulse_dt, U_half, U_half_st);
												for (unsigned int n = 1; n <= steps; ++n)
												{
													split_step(rho, U_half, U_half_st, pulse_dt);

													if (has_pulse_output && pulse_step_index < ExptValuesPulseOrientation.n_rows)
													{
														record_expectation_rho(ExptValuesPulseOrientation, pulse_step_index, rho);
														++pulse_step_index;
													}
												}
											}
											else if (use_dense_H)
											{
												arma::cx_mat H_pulse_dense = H_dense + arma::cx_mat(pulse_operator);
												for (unsigned int n = 1; n <= steps; ++n)
												{
													rk4_step(rho, H, &H_pulse_dense, pulse_dt);

													if (has_pulse_output && pulse_step_index < ExptValuesPulseOrientation.n_rows)
													{
														record_expectation_rho(ExptValuesPulseOrientation, pulse_step_index, rho);
														++pulse_step_index;
													}
												}
											}
											else
											{
												arma::sp_cx_mat H_pulse = H + pulse_operator;
												for (unsigned int n = 1; n <= steps; ++n)
												{
													rk4_step(rho, H_pulse, nullptr, pulse_dt);

													if (has_pulse_output && pulse_step_index < ExptValuesPulseOrientation.n_rows)
													{
														record_expectation_rho(ExptValuesPulseOrientation, pulse_step_index, rho);
														++pulse_step_index;
													}
												}
											}
										}
										else if ((*pulse)->Type() == SpinAPI::PulseType::LongPulse)
										{
											arma::sp_cx_mat pulse_operator;
											if (!space_thread.PulseOperatorOnStatevector((*pulse), pulse_operator))
											{
												this->Log() << "Failed to create a pulse operator in HS." << std::endl;
												continue;
											}

											double pulse_factor = 1.0;
											if (rwa_enabled)
											{
												pulse_factor = 0.5;
											}
											else
											{
												pulse_factor = std::cos((*pulse)->Frequency() * pulse_dt);
											}
											unsigned int steps = static_cast<unsigned int>(std::abs((*pulse)->Pulsetime() / pulse_dt));
											if (relax_use_split_expm)
											{
												arma::cx_mat H_pulse_dense = H_dense + arma::cx_mat(pulse_operator) * pulse_factor;
												arma::cx_mat U_half;
												arma::cx_mat U_half_st;
												build_unitary_half(H_pulse_dense, pulse_dt, U_half, U_half_st);
												for (unsigned int n = 1; n <= steps; ++n)
												{
													split_step(rho, U_half, U_half_st, pulse_dt);

													if (has_pulse_output && pulse_step_index < ExptValuesPulseOrientation.n_rows)
													{
														record_expectation_rho(ExptValuesPulseOrientation, pulse_step_index, rho);
														++pulse_step_index;
													}
												}
											}
											else if (use_dense_H)
											{
												arma::cx_mat H_pulse_dense = H_dense + arma::cx_mat(pulse_operator) * pulse_factor;
												for (unsigned int n = 1; n <= steps; ++n)
												{
													rk4_step(rho, H, &H_pulse_dense, pulse_dt);

													if (has_pulse_output && pulse_step_index < ExptValuesPulseOrientation.n_rows)
													{
														record_expectation_rho(ExptValuesPulseOrientation, pulse_step_index, rho);
														++pulse_step_index;
													}
												}
											}
											else
											{
												arma::sp_cx_mat H_pulse = H + pulse_operator * pulse_factor;
												for (unsigned int n = 1; n <= steps; ++n)
												{
													rk4_step(rho, H_pulse, nullptr, pulse_dt);

													if (has_pulse_output && pulse_step_index < ExptValuesPulseOrientation.n_rows)
													{
														record_expectation_rho(ExptValuesPulseOrientation, pulse_step_index, rho);
														++pulse_step_index;
													}
												}
											}
										}
										else
										{
											this->Log() << "Not implemented yet, sorry." << std::endl;
										}

										unsigned int relax_steps = static_cast<unsigned int>(std::abs(timerelaxation / pulse_dt));
										if (relax_steps > 0)
										{
											if (relax_use_split_expm)
											{
												arma::cx_mat U_half;
												arma::cx_mat U_half_st;
												build_unitary_half(H_dense, pulse_dt, U_half, U_half_st);
												for (unsigned int n = 1; n <= relax_steps; ++n)
												{
													split_step(rho, U_half, U_half_st, pulse_dt);

													if (has_pulse_output && pulse_step_index < ExptValuesPulseOrientation.n_rows)
													{
														record_expectation_rho(ExptValuesPulseOrientation, pulse_step_index, rho);
														++pulse_step_index;
													}
												}
											}
											else
											{
												for (unsigned int n = 1; n <= relax_steps; ++n)
												{
													rk4_step(rho, H, use_dense_H ? &H_dense : nullptr, pulse_dt);

													if (has_pulse_output && pulse_step_index < ExptValuesPulseOrientation.n_rows)
													{
														record_expectation_rho(ExptValuesPulseOrientation, pulse_step_index, rho);
														++pulse_step_index;
													}
												}
											}
										}
									}
								}
							}
						}

						if (has_pulse_output && pulse_step_index != ExptValuesPulseOrientation.n_rows)
						{
							if (grid_num == 0 && gamma_idx == 0)
							{
								this->Log() << "Warning: Pulse output step count mismatch. Expected " << ExptValuesPulseOrientation.n_rows
											<< ", recorded " << pulse_step_index << "." << std::endl;
							}
						}

						arma::mat ExptValuesOrientation;
						if (method_timeevo)
						{
							ExptValuesOrientation.zeros(num_steps, projection_counter);
							if (relax_use_split_expm)
							{
								arma::cx_mat U_half;
								arma::cx_mat U_half_st;
								build_unitary_half(H_dense, dt, U_half, U_half_st);
								for (int k = 0; k < num_steps; ++k)
								{
									record_expectation_rho(ExptValuesOrientation, k, rho);
									split_step(rho, U_half, U_half_st, dt);
								}
							}
							else
							{
								for (int k = 0; k < num_steps; ++k)
								{
									record_expectation_rho(ExptValuesOrientation, k, rho);
									rk4_step(rho, H, use_dense_H ? &H_dense : nullptr, dt);
								}
							}
						}

						if (method_timeinf)
						{
							arma::cx_mat rho0mat = rho;
							arma::cx_mat A_dense = -arma::cx_double(0.0, 1.0) * arma::cx_mat(H) - arma::cx_mat(K);
							arma::cx_mat A_star = arma::conj(A_dense);

							arma::cx_mat L = arma::kron(A_star, Iden_dense) + arma::kron(Iden_dense, A_dense);
							if (!relaxation_super.is_empty())
							{
								L += relaxation_super;
							}
							arma::cx_vec rhs = arma::vectorise(-rho0mat);
							arma::cx_vec sol = arma::solve(L, rhs);
							if (sol.is_empty())
							{
								this->Log() << "Failed to solve timeinf Lyapunov equation in Hilbert space." << std::endl;
								continue;
							}
							arma::cx_mat X = arma::reshape(sol, rho0mat.n_rows, rho0mat.n_cols);
							arma::cx_mat X_t = X.t();

							arma::vec &acc = ExptValuesTimeinfPartial[tid];
							for (int idx = 0; idx < projection_counter; ++idx)
							{
								double val = use_sparse_ops ? (TraceSparseDense(OperatorsSparseLocal[idx], X) / Z)
															: (TraceDenseTransposed(OperatorsDenseLocal[idx], X_t) / Z);
								acc(idx) += weight * val;
							}
						}

						if (has_pulse_output)
							ExptValuesPulsePartial[tid] += weight * ExptValuesPulseOrientation;

						if (method_timeevo)
							ExptValuesPartial[tid] += weight * ExptValuesOrientation;

						continue;
					}

					arma::cx_mat B = Binitial;

					auto record_expectation = [&](arma::mat &target, size_t row_index, const arma::cx_mat &state)
					{
						arma::cx_mat state_conj = arma::conj(state);
						for (int idx = 0; idx < projection_counter; ++idx)
						{
							arma::cx_mat OB;
							if (use_sparse_ops)
							{
								OB = OperatorsSparseLocal[idx] * state;
							}
							else
							{
								OB = OperatorsDenseLocal[idx] * state;
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
							if (grid_num == 0 && gamma_idx == 0)
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
											this->Log() << "Failed to create a pulse operator in HS." << std::endl;
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
											this->Log() << "Failed to create a pulse operator in HS." << std::endl;
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
											this->Log() << "Failed to create a pulse operator in HS." << std::endl;
											continue;
										}

										// Create array containing a propagator and the current state of each system
										std::pair<arma::sp_cx_mat, arma::cx_mat> G;

										double pulse_factor = 1.0;
										if (rwa_enabled)
										{
											pulse_factor = 0.5;
										}
										else
										{
											pulse_factor = std::cos((*pulse)->Frequency() * (*pulse)->Timestep());
										}

										// Get the propagator and put it into the array together with the initial state
										arma::sp_cx_mat A_sp = arma::conv_to<arma::sp_cx_mat>::from(
											arma::expmat(arma::conv_to<arma::cx_mat>::from((A + (arma::cx_double(0.0, -1.0) * pulse_operator * pulse_factor)) * (*pulse)->Timestep())));
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
						if (grid_num == 0 && gamma_idx == 0)
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
									OB = OperatorsSparseLocal[idx] * B;
								}
								else
								{
									OB = OperatorsDenseLocal[idx] * B;
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
								arma::cx_vec tmp;
								if (use_sparse_ops)
								{
									tmp = OperatorsSparseLocal[idx] * prop_state;
								}
								else
								{
									tmp = OperatorsDenseLocal[idx] * prop_state;
								}
								double result = std::abs(arma::cdot(prop_state, tmp));
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
									arma::cx_vec tmp;
									if (use_sparse_ops)
									{
										tmp = OperatorsSparseLocal[idx] * prop_state;
									}
									else
									{
										tmp = OperatorsDenseLocal[idx] * prop_state;
									}
									double result = std::abs(arma::cdot(prop_state, tmp));
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
						if (grid_num == 0 && gamma_idx == 0)
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
									OB = OperatorsSparseLocal[idx] * B;
								}
								else
								{
									OB = OperatorsDenseLocal[idx] * B;
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
						arma::cx_mat X_t = X.t();

						arma::vec &acc = ExptValuesTimeinfPartial[tid];
						for (int idx = 0; idx < projection_counter; ++idx)
						{
							double val = use_sparse_ops ? (TraceSparseDense(OperatorsSparseLocal[idx], X) / Z)
														: (TraceDenseTransposed(OperatorsDenseLocal[idx], X_t) / Z);
							acc(idx) += weight * val;
						}
					}
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
				for (auto &m : ExptValuesTimeinfPartial)
				{
					ExptValuesTimeinf += m;
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
						this->Data() << std::setprecision(12) << ExptValuesTimeinf(idx) << " ";
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
									_stream << (*i)->Name() << "." << (*l)->Name() << "." << (*j)->Name() << ".yield"
											<< ".Ip ";
									_stream << (*i)->Name() << "." << (*l)->Name() << "." << (*j)->Name() << ".yield"
											<< ".Im ";
								}
							}
							else
							{
								// Write each state name
								auto states = (*i)->States();
								_stream << (*i)->Name() << "." << (*l)->Name() << ".Ix ";
								_stream << (*i)->Name() << "." << (*l)->Name() << ".Iy ";
								_stream << (*i)->Name() << "." << (*l)->Name() << ".Iz ";
								_stream << (*i)->Name() << "." << (*l)->Name() << ".Ip ";
								_stream << (*i)->Name() << "." << (*l)->Name() << ".Im ";
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

		if (!this->Properties()->Get("powdergammapoints", this->powderGammaPoints))
		{
			this->Properties()->Get("powdergammastps", this->powderGammaPoints);
		}
		if (this->powderGammaPoints < 1)
		{
			this->powderGammaPoints = 1;
		}

		this->Properties()->Get("powderfullsphere", this->powderFullSphere);
		this->Properties()->Get("fulltensorrotation", this->fullTensorRotation);

		this->rwaEnabled = false;
		this->Properties()->Get("rwa", this->rwaEnabled);

		this->secularizeInteractions = false;
		if (!this->Properties()->Get("secularize", this->secularizeInteractions))
			this->Properties()->Get("secular", this->secularizeInteractions);

		this->secularTolerance = 1e-6;
		if (!this->Properties()->Get("seculartolerance", this->secularTolerance))
			this->Properties()->Get("seculartol", this->secularTolerance);

		this->mwFrequencyGHz = 0.0;
		bool hasFrequency = this->Properties()->Get("mwfrequency", this->mwFrequencyGHz);
		if (!hasFrequency)
			hasFrequency = this->Properties()->Get("frequency", this->mwFrequencyGHz);
		if (!hasFrequency)
			this->Properties()->Get("rffrequency", this->mwFrequencyGHz);

		this->fieldInteractionName.clear();
		this->Properties()->Get("fieldinteraction", this->fieldInteractionName);

		this->rwaSpinNames.clear();
		this->Properties()->GetList("rwaspins", this->rwaSpinNames, ',');

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

			if (this->powderFullSphere)
			{
				theta[i] = std::acos(1.0 - 2.0 * index / _Npoints);
				phi[i] = golden * index;
				weight[i] = 4 * arma::datum::pi / _Npoints;
			}
			else
			{
				theta[i] = std::acos(1.0 - index / _Npoints); // hemisphere
				phi[i] = golden * index;					  // hemisphere
				weight[i] = 2 * arma::datum::pi / _Npoints;
			}
			_uniformGrid[i] = {theta[i], phi[i], weight[i]};
		}

		return true;
	}

}
