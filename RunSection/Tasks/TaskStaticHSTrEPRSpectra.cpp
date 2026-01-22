/////////////////////////////////////////////////////////////////////////
// TaskStaticHSTrEPRSpectra implementation (RunSection module)
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

#include "ActionAddVector.h"
#include "TaskStaticHSTrEPRSpectra.h"
#include "ObjectParser.h"
#include "Settings.h"
#include "Spin.h"
#include "SpinSpace.h"
#include "SpinSystem.h"
#include "State.h"
#include "Interaction.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace RunSection
{
	namespace
	{
		std::string ToLower(std::string value)
		{
			std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
			return value;
		}

		int SopheGridPointCount(int gridSize, int nOctants, bool closedPhi)
		{
			if (gridSize < 1)
				return 0;
			if (nOctants == -1)
				return 1;
			if (nOctants == 0)
				return gridSize;

			int nOct = (nOctants == 8) ? 4 : nOctants;
			int nOrient = gridSize + nOct * gridSize * (gridSize - 1) / 2;
			if (!closedPhi)
				nOrient -= (gridSize - 1);

			if (nOctants == 8)
			{
				int nPhi = nOct * (gridSize - 1) + 1;
				nOrient += (nOrient - nPhi + 1);
			}

			return nOrient;
		}
	}

	// -----------------------------------------------------
	// TaskStaticHSTrEPRSpectra Constructors and Destructor
	// -----------------------------------------------------
	TaskStaticHSTrEPRSpectra::TaskStaticHSTrEPRSpectra(const MSDParser::ObjectParser &_parser, const RunSection &_runsection)
		: BasicTask(_parser, _runsection),
		  mwFrequencyGHz(0.0),
		  linewidth_mT(0.0),
		  linewidthFad_mT(0.0),
		  linewidthDonor_mT(0.0),
		  lineshape("gaussian"),
		  powderGridType("fibonacci"),
		  powderGridSymmetry("D2h"),
		  powderGridSize(0),
		  powdersamplingpoints(0),
		  powderGammaPoints(1),
		  powderFullSphere(true),
		  fullTensorRotation(true),
		  detectSpinNames(),
		  electron1Name(""),
		  electron2Name(""),
		  fieldInteractionName(""),
		  initialStateName(""),
		  hamiltonianH0list()
	{
	}

	TaskStaticHSTrEPRSpectra::~TaskStaticHSTrEPRSpectra()
	{
	}

	// -----------------------------------------------------
	// TaskStaticHSTrEPRSpectra protected methods
	// -----------------------------------------------------
	bool TaskStaticHSTrEPRSpectra::RunLocal()
	{
		this->Log() << "Running task StaticHS-TrEPR-Spectra." << std::endl;

		if (this->RunSettings()->CurrentStep() == 1)
		{
			this->WriteHeader(this->Data());
		}

		// Excitation angular frequency (rad/ns)
		const double omega_mw = 2.0 * arma::datum::pi * this->mwFrequencyGHz;

		// Loop through all SpinSystems
		auto systems = this->SpinSystems();
		for (auto sysIt = systems.cbegin(); sysIt != systems.cend(); sysIt++)
		{
			this->Log() << "\nStarting with SpinSystem \"" << (*sysIt)->Name() << "\"." << std::endl;

			SpinAPI::SpinSpace space(*(*sysIt));
			space.UseSuperoperatorSpace(false);
			space.UseFullTensorRotation(this->fullTensorRotation);
			const arma::uword spaceDim = space.HilbertSpaceDimensions();
			if (this->fullTensorRotation)
			{
				this->Log() << "Full tensor rotation enabled (off-diagonal terms retained)." << std::endl;
			}

			// Build list of interactions to include in H0
			std::vector<std::string> h0list = this->hamiltonianH0list;
			if (h0list.empty())
			{
				for (const auto &interaction : (*sysIt)->Interactions())
				{
					if (!SpinAPI::IsStatic(*interaction))
						continue;
					h0list.push_back(interaction->Name());
				}
			}

			if (h0list.empty())
			{
				this->Log() << "No interactions specified for Hamiltonian H0 in SpinSystem \"" << (*sysIt)->Name() << "\". Skipping." << std::endl;
				continue;
			}

			// Determine Zeeman interaction used for B and for dH/dB (Jacobian)
			SpinAPI::interaction_ptr fieldInteraction = nullptr;
			if (!this->ResolveFieldInteraction((*sysIt), fieldInteraction))
			{
				this->Log() << "No Zeeman interaction found in SpinSystem \"" << (*sysIt)->Name() << "\". Need a Zeeman interaction for field->frequency mapping." << std::endl;
				continue;
			}

			std::vector<SpinAPI::spin_ptr> detectSpins;
			std::vector<std::string> detectSpinNames;
			if (!this->ResolveDetectionSpins((*sysIt), fieldInteraction, detectSpins, detectSpinNames))
			{
				this->Log() << "Failed to resolve detection spins in SpinSystem \"" << (*sysIt)->Name() << "\"." << std::endl;
				continue;
			}
			if (detectSpins.empty())
			{
				this->Log() << "No detection spins available in SpinSystem \"" << (*sysIt)->Name() << "\"." << std::endl;
				continue;
			}
			this->Log() << "Using " << detectSpins.size() << " detection spins in SpinSystem \"" << (*sysIt)->Name() << "\"." << std::endl;

			arma::vec Bvec = fieldInteraction->Field();
			if (Bvec.n_elem != 3)
			{
				this->Log() << "Zeeman interaction \"" << fieldInteraction->Name() << "\" does not provide a 3-vector field." << std::endl;
				continue;
			}
			const double Bmag = arma::norm(Bvec);
			if (!std::isfinite(Bmag) || Bmag <= 0.0)
			{
				this->Log() << "Zeeman field magnitude is invalid (" << Bmag << ")." << std::endl;
				continue;
			}
			const double field_mT = 1.0e3 * Bmag;

			auto cacheIt = this->spectrumCache.find((*sysIt)->Name());
			if (this->RunSettings()->CurrentStep() == 1 && cacheIt == this->spectrumCache.end())
			{
				arma::vec field0;
				arma::vec fieldStep;
				if (this->GetLinearFieldSweep((*sysIt), fieldInteraction, field0, fieldStep))
				{
					SpectrumCache cache;
					if (this->BuildCachedSpectrum((*sysIt), fieldInteraction, field0, fieldStep, cache))
					{
						this->spectrumCache.emplace((*sysIt)->Name(), std::move(cache));
						cacheIt = this->spectrumCache.find((*sysIt)->Name());
					}
				}
			}
			if (cacheIt != this->spectrumCache.end())
			{
				const auto &cache = cacheIt->second;
				const auto idx = static_cast<size_t>(this->RunSettings()->CurrentStep() - 1);
				if (idx < cache.field_mT.size())
				{
					this->Data() << this->RunSettings()->CurrentStep() << " ";
					this->Data() << this->RunSettings()->Time() << " ";
					this->WriteStandardOutput(this->Data());
					this->Data() << cache.field_mT[idx] << " "
								<< cache.total_x[idx] << " " << cache.total_y[idx] << " " << cache.total_perp[idx] << " "
								<< cache.cross_x[idx] << " " << cache.cross_y[idx] << " ";
					for (size_t i = 0; i < cache.spin_names.size(); ++i)
					{
						this->Data() << cache.spin_x[i][idx] << " " << cache.spin_y[i][idx] << " " << cache.spin_perp[i][idx] << " "
									<< cache.spin_p[i][idx] << " " << cache.spin_m[i][idx] << " ";
					}
					this->Data() << std::endl;
					continue;
				}
			}

			// Build initial density matrix
			arma::cx_mat rho0;
			bool hasInitialState = false;
			if (!this->initialStateName.empty())
			{
				auto state = (*sysIt)->states_find(this->initialStateName);
				if (state == nullptr)
				{
					this->Log() << "Initial state \"" << this->initialStateName << "\" not found in SpinSystem \"" << (*sysIt)->Name() << "\"." << std::endl;
				}
				else
				{
					if (space.GetState(state, rho0))
						hasInitialState = true;
				}
			}
			if (!hasInitialState)
			{
				auto initial_states = (*sysIt)->InitialState();
				if (initial_states.empty())
				{
					this->Log() << "Skipping SpinSystem \"" << (*sysIt)->Name() << "\" as no initial state was specified." << std::endl;
					continue;
				}

				for (auto state = initial_states.cbegin(); state != initial_states.cend(); state++)
				{
					arma::cx_mat tmp;
					if (!space.GetState(*state, tmp))
					{
						this->Log() << "Failed to obtain projection matrix onto state \"" << (*state)->Name() << "\" of SpinSystem \"" << (*sysIt)->Name() << "\"." << std::endl;
						continue;
					}

					if (!hasInitialState)
					{
						rho0 = tmp;
						hasInitialState = true;
					}
					else
					{
						rho0 += tmp;
					}
				}
			}

			if (!hasInitialState)
			{
				this->Log() << "Failed to construct initial state for SpinSystem \"" << (*sysIt)->Name() << "\"." << std::endl;
				continue;
			}
			rho0 /= arma::trace(rho0);

			// Embed *bare* spin operators (Sx,Sy,Sz) into the full Hilbert space.
			// Build magnetic dipole operators from orientation-rotated tensors to match H0's frame.
			std::vector<arma::cx_mat> Sx_list(detectSpins.size());
			std::vector<arma::cx_mat> Sy_list(detectSpins.size());
			std::vector<arma::cx_mat> Sz_list(detectSpins.size());
			bool operators_ok = true;
			for (size_t i = 0; i < detectSpins.size(); ++i)
			{
				auto spin = detectSpins[i];
				if (!space.CreateOperator(arma::conv_to<arma::cx_mat>::from(spin->Sx()), spin, Sx_list[i]) ||
					!space.CreateOperator(arma::conv_to<arma::cx_mat>::from(spin->Sy()), spin, Sy_list[i]) ||
					!space.CreateOperator(arma::conv_to<arma::cx_mat>::from(spin->Sz()), spin, Sz_list[i]))
				{
					this->Log() << "Failed to build bare spin operators for spin \"" << spin->Name() << "\"." << std::endl;
					operators_ok = false;
					break;
				}
			}
			if (!operators_ok)
				continue;

			// Build powder grid (theta,phi) and optional gamma sampling.
			int numPoints = this->powdersamplingpoints;
			std::vector<std::tuple<double, double, double>> grid;
			const bool useSopheGrid = (this->powderGridType == "sophe");
			if (useSopheGrid)
			{
				int gridSize = this->powderGridSize;
				if (gridSize < 2)
				{
					double maxPhi = 0.0;
					bool closedPhi = false;
					int nOctants = 0;
					if (numPoints > 1 && this->SopheGridParams(this->powderGridSymmetry, maxPhi, closedPhi, nOctants))
					{
						int bestSize = 0;
						int bestDiff = std::numeric_limits<int>::max();
						for (int candidate = 2; candidate <= 200; ++candidate)
						{
							int count = SopheGridPointCount(candidate, nOctants, closedPhi);
							int diff = std::abs(count - numPoints);
							if (diff < bestDiff)
							{
								bestDiff = diff;
								bestSize = candidate;
								if (diff == 0)
									break;
							}
						}
						if (bestSize > 0)
							gridSize = bestSize;
					}
					if (gridSize < 2)
						gridSize = 19;
				}

				if (!this->CreateSopheGrid(gridSize, this->powderGridSymmetry, grid))
				{
					this->Log() << "Failed to obtain SOPHE grid for powder averaging." << std::endl;
					continue;
				}
				numPoints = static_cast<int>(grid.size());
				this->Log() << "Using SOPHE grid (" << this->powderGridSymmetry << ", GridSize=" << gridSize << ") with " << numPoints << " orientations." << std::endl;
			}
			else if (numPoints > 1)
			{
				if (!this->CreateUniformGrid(numPoints, grid))
				{
					this->Log() << "Failed to obtain a uniform grid for powder averaging." << std::endl;
					continue;
				}
				this->Log() << "Using powder averaging with " << numPoints << " orientations." << std::endl;
				if (this->powderFullSphere)
					this->Log() << "Using full-sphere powder grid." << std::endl;
			}
			else
			{
				grid.clear();
				grid.emplace_back(0.0, 0.0, 1.0);
				numPoints = 1;
			}
			const int gamma_points = (numPoints > 1) ? std::max(1, this->powderGammaPoints) : 1;
			const double gamma_weight = useSopheGrid ? (2.0 * arma::datum::pi / static_cast<double>(gamma_points))
														: (1.0 / static_cast<double>(gamma_points));
			if (this->powderGammaPoints > 1)
				this->Log() << "Sampling gamma with " << this->powderGammaPoints << " points per orientation." << std::endl;

			// Field-domain linewidth (FWHM, mT). Apply the lineshape in field units.
			const double lwB_mT = std::abs(this->linewidth_mT);

			// Precompute interaction-frame rotation for the Zeeman interaction (g-tensor frame).
			// IMPORTANT: Interaction framelists are interpreted as passive ZXZ Euler angles.
			// SpinSpace::InteractionOperatorRotated() uses the same convention internally.
			arma::mat RFrame = arma::eye<arma::mat>(3, 3);
			{
				auto fr = fieldInteraction->Framelist();
				double a = (fr.n_elem >= 1) ? fr(0) : 0.0;
				double b = (fr.n_elem >= 2) ? fr(1) : 0.0;
				double g = (fr.n_elem >= 3) ? fr(2) : 0.0;

				// Passive ZXZ Euler rotation: R = Rz(gamma) * Ry(beta) * Rz(alpha)
				const double ca = std::cos(a), sa = std::sin(a);
				const double cb = std::cos(b), sb = std::sin(b);
				const double cg = std::cos(g), sg = std::sin(g);

				arma::mat Ra = {{ca, sa, 0.0}, {-sa, ca, 0.0}, {0.0, 0.0, 1.0}};
				arma::mat Rb = {{cb, 0.0, -sb}, {0.0, 1.0, 0.0}, {sb, 0.0, cb}};
				arma::mat Rg = {{cg, sg, 0.0}, {-sg, cg, 0.0}, {0.0, 0.0, 1.0}};
				RFrame = Rg * Rb * Ra;
			}

			// Base tensors (as specified on spins)
			std::vector<arma::mat> g_frame_base(detectSpins.size());
			for (size_t i = 0; i < detectSpins.size(); ++i)
			{
				arma::mat g_base = arma::conv_to<arma::mat>::from(detectSpins[i]->GetTensor().LabFrame());
				if (fieldInteraction->IgnoreTensors())
					g_base = arma::eye<arma::mat>(3, 3);
				g_frame_base[i] = RFrame * g_base * RFrame.t();
			}

			double mu_prefactor = fieldInteraction->Prefactor();
			if (fieldInteraction->AddCommonPrefactor())
				mu_prefactor *= 8.79410005e+1;

			// Accumulators
			double total_x = 0.0;
			double total_y = 0.0;
			double total_perp = 0.0;
			double cross_x = 0.0;
			double cross_y = 0.0;
			std::vector<double> spin_x(detectSpins.size(), 0.0);
			std::vector<double> spin_y(detectSpins.size(), 0.0);
			std::vector<double> spin_perp(detectSpins.size(), 0.0);
			std::vector<double> spin_p(detectSpins.size(), 0.0);
			std::vector<double> spin_m(detectSpins.size(), 0.0);

			// For dH/dB we need the Zeeman Hamiltonian only (rotated per orientation)
			std::vector<std::string> zeelist;
			zeelist.push_back(fieldInteraction->Name());

			const arma::cx_double I(0.0, 1.0);
			const size_t spin_count = detectSpins.size();

			auto accumulate_grid = [&](int grid_num,
									   double &acc_total_x, double &acc_total_y, double &acc_total_perp,
									   double &acc_cross_x, double &acc_cross_y,
									   std::vector<double> &acc_spin_x, std::vector<double> &acc_spin_y,
									   std::vector<double> &acc_spin_perp, std::vector<double> &acc_spin_p,
									   std::vector<double> &acc_spin_m)
			{
				auto [theta, phi, w_solid] = grid[grid_num];
				const double base_weight = w_solid;

				for (int gamma_idx = 0; gamma_idx < gamma_points; ++gamma_idx)
				{
					double gamma = 0.0;
					if (gamma_points > 1)
						gamma = 2.0 * arma::datum::pi * (static_cast<double>(gamma_idx) + 0.5) / static_cast<double>(gamma_points);

					const double w = base_weight * gamma_weight;

					arma::mat Rot;
					if (!this->CreateRotationMatrix(phi, theta, gamma, Rot))
						continue;

					// Rot is an ACTIVE rotation. SpinSpace::InteractionOperatorRotated() will transpose it internally
					// to obtain the PASSIVE tensor rotation used for anisotropic couplings. To keep magnetic dipole operators
					// consistent with the Hamiltonian orientation, we use the same PASSIVE matrix here.
					const arma::mat Rpowder = Rot.t();

					// Build rotated base Hamiltonian
					arma::sp_cx_mat H0_sp;
					if (!space.BaseHamiltonianRotated(h0list, Rot, H0_sp))
						continue;

					arma::cx_mat H0 = arma::cx_mat(H0_sp);
					arma::vec eigval;
					arma::cx_mat eigvec;
					if (!arma::eig_sym(eigval, eigvec, H0))
						continue;

					const arma::cx_mat Udag = arma::trans(arma::conj(eigvec));
					const arma::cx_mat rho_eig = Udag * rho0 * eigvec;

					// Zeeman-only rotated Hamiltonian -> dH/dB magnitude for field-to-energy Jacobian.
					arma::sp_cx_mat Hz_sp;
					if (!space.BaseHamiltonianRotated(zeelist, Rot, Hz_sp))
						continue;
					arma::cx_mat dHdB = arma::cx_mat(Hz_sp) / Bmag; // rad/ns/T

					std::vector<arma::cx_mat> mux_list(spin_count);
					std::vector<arma::cx_mat> muy_list(spin_count);
					arma::cx_mat muxT = arma::zeros<arma::cx_mat>(spaceDim, spaceDim);
					arma::cx_mat muyT = arma::zeros<arma::cx_mat>(spaceDim, spaceDim);
					bool tensor_dim_ok = true;

					for (size_t i = 0; i < spin_count; ++i)
					{
						arma::mat g = Rpowder * g_frame_base[i] * Rpowder.t();
						if (!this->fullTensorRotation)
							g = g % arma::eye<arma::mat>(3, 3);

						arma::cx_mat mux = g(0, 0) * Sx_list[i] + g(1, 0) * Sy_list[i] + g(2, 0) * Sz_list[i];
						arma::cx_mat muy = g(0, 1) * Sx_list[i] + g(1, 1) * Sy_list[i] + g(2, 1) * Sz_list[i];

						if (mu_prefactor != 1.0)
						{
							mux *= mu_prefactor;
							muy *= mu_prefactor;
						}

						if (mux.n_rows != spaceDim || mux.n_cols != spaceDim || muy.n_rows != spaceDim || muy.n_cols != spaceDim)
							tensor_dim_ok = false;
						mux_list[i] = mux;
						muy_list[i] = muy;
						muxT += mux;
						muyT += muy;
					}
					if (!tensor_dim_ok)
						continue;

					// Transform mu operators into eigenbasis
					arma::cx_mat muxT_eig = Udag * muxT * eigvec;
					arma::cx_mat muyT_eig = Udag * muyT * eigvec;
					std::vector<arma::cx_mat> mux_eig(spin_count);
					std::vector<arma::cx_mat> muy_eig(spin_count);
					for (size_t i = 0; i < spin_count; ++i)
					{
						mux_eig[i] = Udag * mux_list[i] * eigvec;
						muy_eig[i] = Udag * muy_list[i] * eigvec;
					}

					const arma::uword dim = eigval.n_elem;
					if (dim != spaceDim)
						continue;

					double loc_total_x = 0.0;
					double loc_total_y = 0.0;
					double loc_total_perp = 0.0;
					double loc_cross_x = 0.0;
					double loc_cross_y = 0.0;
					std::vector<double> loc_spin_x(spin_count, 0.0);
					std::vector<double> loc_spin_y(spin_count, 0.0);
					std::vector<double> loc_spin_perp(spin_count, 0.0);
					std::vector<double> loc_spin_p(spin_count, 0.0);
					std::vector<double> loc_spin_m(spin_count, 0.0);

					for (arma::uword m = 0; m < dim; ++m)
					{
						const double rho_mm = std::real(rho_eig(m, m));
						for (arma::uword n = m + 1; n < dim; ++n)
						{
							const double rho_nn = std::real(rho_eig(n, n));
							const double population = rho_mm - rho_nn;
							if (std::abs(population) < 1e-15)
								continue;

							const double deltaOmega = (eigval(n) - eigval(m)) - omega_mw; // rad/ns
							const arma::cx_vec Um = eigvec.col(m);
							const arma::cx_vec Vn = eigvec.col(n);
							const arma::cx_double domega_dB_c = arma::cdot(Vn - Um, dHdB * (Vn + Um)); // rad/ns/T
							const double abs_domega_dB = std::abs(std::real(domega_dB_c));
							if (!std::isfinite(abs_domega_dB) || abs_domega_dB < 1e-15)
								continue;

							const double dBdE = 1.0 / abs_domega_dB; // T / (rad/ns)
							// Jacobian safeguard: the 1/g factor (dB/dE) can diverge when
							// d(E_n-E_m)/dB \approx 0 (near avoided crossings / degeneracies).
							// Skip these transitions to avoid unphysical spikes.
							if (dBdE > 1e5)
								continue;

							const double deltaB = deltaOmega * dBdE;  // T
							const double deltaB_mT = 1.0e3 * deltaB;

							const double L = this->LineshapeValue(deltaB_mT, lwB_mT);
							if (L == 0.0)
								continue;

							const double wField = dBdE * L;

							const double ITx = std::norm(muxT_eig(m, n));
							const double ITy = std::norm(muyT_eig(m, n));
							double I_sum_x = 0.0;
							double I_sum_y = 0.0;

							for (size_t i = 0; i < spin_count; ++i)
							{
								const arma::cx_double muix = mux_eig[i](m, n);
								const arma::cx_double muiy = muy_eig[i](m, n);
								const double Iix = std::norm(muix);
								const double Iiy = std::norm(muiy);

								I_sum_x += Iix;
								I_sum_y += Iiy;

								loc_spin_x[i] += population * Iix * wField;
								loc_spin_y[i] += population * Iiy * wField;
								loc_spin_perp[i] += population * 0.5 * (Iix + Iiy) * wField;

								const arma::cx_double mup = muix + I * muiy;
								const arma::cx_double mum = muix - I * muiy;
								loc_spin_p[i] += population * std::norm(mup) * wField;
								loc_spin_m[i] += population * std::norm(mum) * wField;
							}

							const double ICx = ITx - I_sum_x;
							const double ICy = ITy - I_sum_y;

							loc_total_x += population * ITx * wField;
							loc_total_y += population * ITy * wField;
							loc_total_perp += population * 0.5 * (ITx + ITy) * wField;
							loc_cross_x += population * ICx * wField;
							loc_cross_y += population * ICy * wField;
						}
					}

					acc_total_x += w * loc_total_x;
					acc_total_y += w * loc_total_y;
					acc_total_perp += w * loc_total_perp;
					acc_cross_x += w * loc_cross_x;
					acc_cross_y += w * loc_cross_y;

					for (size_t i = 0; i < spin_count; ++i)
					{
						acc_spin_x[i] += w * loc_spin_x[i];
						acc_spin_y[i] += w * loc_spin_y[i];
						acc_spin_perp[i] += w * loc_spin_perp[i];
						acc_spin_p[i] += w * loc_spin_p[i];
						acc_spin_m[i] += w * loc_spin_m[i];
					}
				}
			};

#ifdef _OPENMP
#pragma omp parallel
			{
				double local_total_x = 0.0;
				double local_total_y = 0.0;
				double local_total_perp = 0.0;
				double local_cross_x = 0.0;
				double local_cross_y = 0.0;
				std::vector<double> local_spin_x(spin_count, 0.0);
				std::vector<double> local_spin_y(spin_count, 0.0);
				std::vector<double> local_spin_perp(spin_count, 0.0);
				std::vector<double> local_spin_p(spin_count, 0.0);
				std::vector<double> local_spin_m(spin_count, 0.0);

#pragma omp for
				for (int grid_num = 0; grid_num < numPoints; ++grid_num)
				{
					accumulate_grid(grid_num, local_total_x, local_total_y, local_total_perp, local_cross_x, local_cross_y,
									local_spin_x, local_spin_y, local_spin_perp, local_spin_p, local_spin_m);
				}

#pragma omp critical
				{
					total_x += local_total_x;
					total_y += local_total_y;
					total_perp += local_total_perp;
					cross_x += local_cross_x;
					cross_y += local_cross_y;
					for (size_t i = 0; i < spin_count; ++i)
					{
						spin_x[i] += local_spin_x[i];
						spin_y[i] += local_spin_y[i];
						spin_perp[i] += local_spin_perp[i];
						spin_p[i] += local_spin_p[i];
						spin_m[i] += local_spin_m[i];
					}
				}
			}
#else
			for (int grid_num = 0; grid_num < numPoints; ++grid_num)
			{
				accumulate_grid(grid_num, total_x, total_y, total_perp, cross_x, cross_y,
								spin_x, spin_y, spin_perp, spin_p, spin_m);
			}
#endif

			// Output
			this->Data() << this->RunSettings()->CurrentStep() << " ";
			this->Data() << this->RunSettings()->Time() << " ";
			this->WriteStandardOutput(this->Data());
			this->Data() << field_mT << " "
						<< total_x << " " << total_y << " " << total_perp << " "
						<< cross_x << " " << cross_y << " ";
			for (size_t i = 0; i < detectSpinNames.size(); ++i)
			{
				this->Data() << spin_x[i] << " " << spin_y[i] << " " << spin_perp[i] << " "
							<< spin_p[i] << " " << spin_m[i] << " ";
			}
			this->Data() << std::endl;
		}

		return true;
	}

	double TaskStaticHSTrEPRSpectra::LineshapeValue(double _delta, double _fwhm) const
	{
		if (!std::isfinite(_delta) || !std::isfinite(_fwhm))
			return 0.0;

		if (_fwhm <= 0.0)
		{
			return (std::abs(_delta) < 1e-12) ? 1.0 : 0.0;
		}

		const double x = _delta / _fwhm;
		if (this->lineshape == "lorentzian")
		{
			const double gamma = 0.5 * _fwhm;
			return (1.0 / arma::datum::pi) * (gamma / (_delta * _delta + gamma * gamma));
		}

		const double pref = std::sqrt(4.0 * std::log(2.0) / arma::datum::pi) / _fwhm;
		return pref * std::exp(-4.0 * std::log(2.0) * x * x);
	}

	double TaskStaticHSTrEPRSpectra::LinewidthToOmega(double _fwhm_mT, double _giso) const
	{
		const double muB_over_hbar = 8.79410005e+1; // rad / ns / T
		return std::abs(_fwhm_mT) * 1.0e-3 * muB_over_hbar * std::abs(_giso);
	}

	bool TaskStaticHSTrEPRSpectra::CreateRotationMatrix(double &_alpha, double &_beta, double &_gamma, arma::mat &_R) const
	{
		// IMPORTANT: This must return an *ACTIVE* rotation matrix.
		// SpinSpace::InteractionOperatorRotated() interprets the supplied powder matrix as ACTIVE and
		// internally transposes it to obtain the PASSIVE tensor-rotation used for coupling tensors.
		//
		// To align with the passive ZXZ Euler convention, construct the ACTIVE
		// inverse rotation such that R.t() = Rz(gamma) * Ry(beta) * Rz(alpha).
		const double ca = std::cos(_alpha), sa = std::sin(_alpha);
		const double cb = std::cos(_beta), sb = std::sin(_beta);
		const double cg = std::cos(_gamma), sg = std::sin(_gamma);

		// Active inverse rotation (negative angles)
		arma::mat R1 = {
			{ca, -sa, 0.0},
			{sa, ca, 0.0},
			{0.0, 0.0, 1.0}};

		arma::mat R2 = {
			{cb, 0.0, sb},
			{0.0, 1.0, 0.0},
			{-sb, 0.0, cb}};

		arma::mat R3 = {
			{cg, -sg, 0.0},
			{sg, cg, 0.0},
			{0.0, 0.0, 1.0}};

		_R = R1 * R2 * R3;
		return true;
	}

	bool TaskStaticHSTrEPRSpectra::SopheGridParams(const std::string &_symmetry, double &_maxPhi, bool &_closedPhi, int &_nOctants) const
	{
		std::string sym = ToLower(_symmetry);
		if (sym == "c1")
		{
			_maxPhi = 2.0 * arma::datum::pi;
			_closedPhi = false;
			_nOctants = 8;
		}
		else if (sym == "ci")
		{
			_maxPhi = 2.0 * arma::datum::pi;
			_closedPhi = false;
			_nOctants = 4;
		}
		else if (sym == "c2h")
		{
			_maxPhi = arma::datum::pi;
			_closedPhi = false;
			_nOctants = 2;
		}
		else if (sym == "s6")
		{
			_maxPhi = 2.0 * arma::datum::pi / 3.0;
			_closedPhi = false;
			_nOctants = 2;
		}
		else if (sym == "c4h")
		{
			_maxPhi = arma::datum::pi / 2.0;
			_closedPhi = false;
			_nOctants = 1;
		}
		else if (sym == "c6h")
		{
			_maxPhi = arma::datum::pi / 3.0;
			_closedPhi = false;
			_nOctants = 1;
		}
		else if (sym == "d2h")
		{
			_maxPhi = arma::datum::pi / 2.0;
			_closedPhi = true;
			_nOctants = 1;
		}
		else if (sym == "th")
		{
			_maxPhi = arma::datum::pi / 2.0;
			_closedPhi = true;
			_nOctants = 1;
		}
		else if (sym == "d3d")
		{
			_maxPhi = arma::datum::pi / 3.0;
			_closedPhi = true;
			_nOctants = 1;
		}
		else if (sym == "d4h")
		{
			_maxPhi = arma::datum::pi / 4.0;
			_closedPhi = true;
			_nOctants = 1;
		}
		else if (sym == "oh")
		{
			_maxPhi = arma::datum::pi / 4.0;
			_closedPhi = true;
			_nOctants = 1;
		}
		else if (sym == "d6h")
		{
			_maxPhi = arma::datum::pi / 6.0;
			_closedPhi = true;
			_nOctants = 1;
		}
		else if (sym == "dinfh")
		{
			_maxPhi = 0.0;
			_closedPhi = true;
			_nOctants = 0;
		}
		else if (sym == "o3")
		{
			_maxPhi = 0.0;
			_closedPhi = true;
			_nOctants = -1;
		}
		else
		{
			return false;
		}

		return true;
	}

	bool TaskStaticHSTrEPRSpectra::CreateSopheGrid(int _gridSize, const std::string &_symmetry, std::vector<std::tuple<double, double, double>> &_grid) const
	{
		_grid.clear();
		if (_gridSize < 1)
			return false;

		double maxPhi = 0.0;
		bool closedPhi = false;
		int nOctants = 0;
		if (!this->SopheGridParams(_symmetry, maxPhi, closedPhi, nOctants))
			return false;

		if (nOctants == -1)
		{
			_grid.emplace_back(0.0, 0.0, 4.0 * arma::datum::pi);
			return true;
		}

		if (nOctants == 0)
		{
			if (_gridSize < 2)
				return false;
			const double dtheta = (arma::datum::pi / 2.0) / static_cast<double>(_gridSize - 1);
			std::vector<double> boundaries;
			boundaries.reserve(static_cast<size_t>(_gridSize + 1));
			boundaries.push_back(0.0);
			for (int i = 0; i < _gridSize - 1; ++i)
				boundaries.push_back(dtheta * (0.5 + static_cast<double>(i)));
			boundaries.push_back(arma::datum::pi / 2.0);

			_grid.reserve(static_cast<size_t>(_gridSize));
			for (int i = 0; i < _gridSize; ++i)
			{
				const double theta = dtheta * static_cast<double>(i);
				const double w = -2.0 * (2.0 * arma::datum::pi) * (std::cos(boundaries[i + 1]) - std::cos(boundaries[i]));
				_grid.emplace_back(theta, 0.0, w);
			}

			return true;
		}

		if (_gridSize < 2)
			return false;

		const int nOct = (nOctants == 8) ? 4 : nOctants;
		const double dtheta = (arma::datum::pi / 2.0) / static_cast<double>(_gridSize - 1);
		const double sindth2 = std::sin(dtheta / 2.0);
		const double w0 = closedPhi ? 0.5 : 1.0;

		const int nOrientations = _gridSize + nOct * _gridSize * (_gridSize - 1) / 2;
		std::vector<double> phi(static_cast<size_t>(nOrientations), 0.0);
		std::vector<double> theta(static_cast<size_t>(nOrientations), 0.0);
		std::vector<double> weights(static_cast<size_t>(nOrientations), 0.0);

		phi[0] = 0.0;
		theta[0] = 0.0;
		weights[0] = maxPhi * (1.0 - std::cos(dtheta / 2.0));

		int start = 1;
		for (int iSlice = 2; iSlice <= _gridSize - 1; ++iSlice)
		{
			const int nPhi = nOct * (iSlice - 1) + 1;
			const double dPhi = maxPhi / static_cast<double>(nPhi - 1);
			for (int j = 0; j < nPhi; ++j)
			{
				const int idx = start + j;
				double w = 2.0 * std::sin((iSlice - 1) * dtheta) * sindth2 * dPhi;
				if (j == 0)
					w *= w0;
				else if (j == nPhi - 1)
					w *= 0.5;
				weights[idx] = w;
				phi[idx] = dPhi * static_cast<double>(j);
				theta[idx] = dtheta * static_cast<double>(iSlice - 1);
			}
			start += nPhi;
		}

		const int nPhiEq = nOct * (_gridSize - 1) + 1;
		const double dPhiEq = maxPhi / static_cast<double>(nPhiEq - 1);
		for (int j = 0; j < nPhiEq; ++j)
		{
			const int idx = start + j;
			double w = sindth2 * dPhiEq;
			if (j == 0)
				w *= w0;
			else if (j == nPhiEq - 1)
				w *= 0.5;
			weights[idx] = w;
			phi[idx] = dPhiEq * static_cast<double>(j);
			theta[idx] = arma::datum::pi / 2.0;
		}

		if (!closedPhi)
		{
			std::vector<int> rmv;
			rmv.reserve(static_cast<size_t>(_gridSize - 1));
			int csum = 0;
			for (int i = 1; i <= _gridSize - 1; ++i)
			{
				csum += nOct * i + 1;
				rmv.push_back(csum);
			}

			std::vector<double> phi2;
			std::vector<double> theta2;
			std::vector<double> weights2;
			phi2.reserve(phi.size() - rmv.size());
			theta2.reserve(theta.size() - rmv.size());
			weights2.reserve(weights.size() - rmv.size());

			size_t rmv_pos = 0;
			for (size_t idx = 0; idx < phi.size(); ++idx)
			{
				if (rmv_pos < rmv.size() && static_cast<int>(idx) == rmv[rmv_pos])
				{
					++rmv_pos;
					continue;
				}
				phi2.push_back(phi[idx]);
				theta2.push_back(theta[idx]);
				weights2.push_back(weights[idx]);
			}

			phi.swap(phi2);
			theta.swap(theta2);
			weights.swap(weights2);
		}

		if (nOctants == 8)
		{
			const int nPhi = nPhiEq;
			const int N = static_cast<int>(theta.size());
			int start_idx = N - nPhi;
			if (start_idx < 0)
				start_idx = 0;

			std::vector<double> phi_add;
			std::vector<double> theta_add;
			std::vector<double> weights_add;
			phi_add.reserve(static_cast<size_t>(start_idx + 1));
			theta_add.reserve(static_cast<size_t>(start_idx + 1));
			weights_add.reserve(static_cast<size_t>(start_idx + 1));

			for (int i = start_idx; i >= 0; --i)
			{
				weights[i] *= 0.5;
				phi_add.push_back(phi[i]);
				theta_add.push_back(arma::datum::pi - theta[i]);
				weights_add.push_back(weights[i]);
			}

			phi.insert(phi.end(), phi_add.begin(), phi_add.end());
			theta.insert(theta.end(), theta_add.begin(), theta_add.end());
			weights.insert(weights.end(), weights_add.begin(), weights_add.end());
		}

		const double scale = 2.0 * (2.0 * arma::datum::pi / maxPhi);
		for (auto &w : weights)
			w *= scale;

		_grid.reserve(phi.size());
		for (size_t i = 0; i < phi.size(); ++i)
			_grid.emplace_back(theta[i], phi[i], weights[i]);

		return true;
	}

	bool TaskStaticHSTrEPRSpectra::CreateUniformGrid(int &_Npoints, std::vector<std::tuple<double, double, double>> &_uniformGrid) const
	{
		std::vector<double> theta(_Npoints);
		std::vector<double> phi(_Npoints);
		std::vector<double> weight(_Npoints);

		_uniformGrid.resize(_Npoints);

		const double golden = arma::datum::pi * (1.0 + std::sqrt(5.0));

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
				theta[i] = std::acos(1.0 - index / _Npoints);
				phi[i] = golden * index;
				weight[i] = 2 * arma::datum::pi / _Npoints;
			}
			_uniformGrid[i] = {theta[i], phi[i], weight[i]};
		}

		return true;
	}

	bool TaskStaticHSTrEPRSpectra::ResolveFieldInteraction(const SpinAPI::system_ptr &_system, SpinAPI::interaction_ptr &_fieldInteraction) const
	{
		_fieldInteraction = nullptr;
		if (_system == nullptr)
			return false;

		if (!this->fieldInteractionName.empty())
			_fieldInteraction = _system->interactions_find(this->fieldInteractionName);

		if (_fieldInteraction == nullptr)
		{
			for (auto inter = _system->interactions_cbegin(); inter != _system->interactions_cend(); inter++)
			{
				std::string type;
				if ((*inter)->Properties()->Get("type", type))
				{
					type = ToLower(type);
					if (type == "zeeman")
					{
						_fieldInteraction = (*inter);
						break;
					}
				}
			}
		}

		return (_fieldInteraction != nullptr);
	}

	bool TaskStaticHSTrEPRSpectra::ResolveDetectionSpins(const SpinAPI::system_ptr &_system, const SpinAPI::interaction_ptr &_fieldInteraction,
														std::vector<SpinAPI::spin_ptr> &_spins, std::vector<std::string> &_spinNames) const
	{
		_spins.clear();
		_spinNames.clear();
		if (_system == nullptr)
			return false;

		auto add_spin = [&](const SpinAPI::spin_ptr &spin) -> bool {
			if (spin == nullptr)
				return false;
			for (const auto &existing : _spins)
			{
				if (existing == spin)
					return true;
			}
			_spins.push_back(spin);
			return true;
		};

		if (!this->detectSpinNames.empty())
		{
			for (const auto &name : this->detectSpinNames)
			{
				auto spin = _system->spins_find(name);
				if (spin == nullptr)
					return false;
				add_spin(spin);
			}
		}
		else if (!this->electron1Name.empty() || !this->electron2Name.empty())
		{
			if (!this->electron1Name.empty())
			{
				auto spin = _system->spins_find(this->electron1Name);
				if (spin == nullptr)
					return false;
				add_spin(spin);
			}
			if (!this->electron2Name.empty())
			{
				auto spin = _system->spins_find(this->electron2Name);
				if (spin == nullptr)
					return false;
				add_spin(spin);
			}
		}
		else if (_fieldInteraction != nullptr)
		{
			auto group = _fieldInteraction->Group1();
			for (const auto &spin : group)
			{
				add_spin(spin);
			}
		}

		if (_spins.empty())
		{
			auto allSpins = _system->Spins();
			for (const auto &spin : allSpins)
				add_spin(spin);
		}

		if (_spins.empty())
			return false;

		for (const auto &spin : _spins)
			_spinNames.push_back(spin->Name());

		return true;
	}

	bool TaskStaticHSTrEPRSpectra::GetLinearFieldSweep(const SpinAPI::system_ptr &_system, const SpinAPI::interaction_ptr &_fieldInteraction, arma::vec &_field0, arma::vec &_fieldStep) const
	{
		if (_system == nullptr || _fieldInteraction == nullptr)
			return false;

		_field0 = _fieldInteraction->Field();
		if (_field0.n_elem != 3 || !_field0.is_finite())
			return false;

		const std::string target = _system->Name() + "." + _fieldInteraction->Name() + ".field";
		bool found = false;
		arma::vec direction;
		double value = 0.0;

		for (const auto &action : this->Actions())
		{
			auto add = std::dynamic_pointer_cast<ActionAddVector>(action);
			if (!add)
				return false;

			std::string targetName;
			if (!add->GetProperties()->Get("vector", targetName))
				add->GetProperties()->Get("actionvector", targetName);

			if (targetName != target)
				return false;

			if (found)
				return false;

			if (!add->GetProperties()->Get("direction", direction))
				return false;
			if (direction.n_elem != 3 || !direction.is_finite())
				return false;

			direction = arma::normalise(direction);
			value = add->Value();

			if (add->Period() != 1 || add->First() != 1)
				return false;
			if (add->Last() != 0 && add->Last() < this->RunSettings()->Steps())
				return false;

			found = true;
		}

		if (!found)
			return false;

		_fieldStep = value * direction;
		if (arma::norm(_field0) > 0.0)
		{
			arma::vec dir0 = arma::normalise(_field0);
			if (arma::norm(arma::cross(dir0, direction)) > 1e-6)
				return false;
		}

		return true;
	}

	bool TaskStaticHSTrEPRSpectra::BuildCachedSpectrum(const SpinAPI::system_ptr &_system, const SpinAPI::interaction_ptr &_fieldInteraction, const arma::vec &_field0, const arma::vec &_fieldStep, SpectrumCache &_cache)
	{
		if (_system == nullptr || _fieldInteraction == nullptr)
			return false;

		const unsigned int steps = this->RunSettings()->Steps();
		if (steps < 2)
			return false;

		if (_field0.n_elem != 3 || !_field0.is_finite())
			return false;

		std::vector<double> field_T(steps, 0.0);
		_cache.field_mT.assign(steps, 0.0);
		_cache.total_x.assign(steps, 0.0);
		_cache.total_y.assign(steps, 0.0);
		_cache.total_perp.assign(steps, 0.0);
		_cache.cross_x.assign(steps, 0.0);
		_cache.cross_y.assign(steps, 0.0);
		_cache.steps = steps;

		arma::vec Bvec = _field0;
		for (unsigned int i = 0; i < steps; ++i)
		{
			const double Bmag = arma::norm(Bvec);
			if (!std::isfinite(Bmag) || Bmag <= 0.0)
				return false;
			field_T[i] = Bmag;
			_cache.field_mT[i] = 1.0e3 * Bmag;
			Bvec += _fieldStep;
		}

		const double dBstep = (steps > 1) ? (field_T[1] - field_T[0]) : 0.0;
		const double dBabs = std::abs(dBstep);
		const double omega_mw = 2.0 * arma::datum::pi * this->mwFrequencyGHz;

		SpinAPI::SpinSpace space(*_system);
		space.UseSuperoperatorSpace(false);
		space.UseFullTensorRotation(this->fullTensorRotation);
		const arma::uword spaceDim = space.HilbertSpaceDimensions();

		std::vector<std::string> h0list = this->hamiltonianH0list;
		if (h0list.empty())
		{
			for (const auto &interaction : _system->Interactions())
			{
				if (!SpinAPI::IsStatic(*interaction))
					continue;
				h0list.push_back(interaction->Name());
			}
		}

		if (h0list.empty())
			return false;

		std::vector<std::string> h0list_noB;
		h0list_noB.reserve(h0list.size());
		for (const auto &name : h0list)
		{
			if (name != _fieldInteraction->Name())
				h0list_noB.push_back(name);
		}

		std::vector<SpinAPI::spin_ptr> detectSpins;
		std::vector<std::string> detectSpinNames;
		if (!this->ResolveDetectionSpins(_system, _fieldInteraction, detectSpins, detectSpinNames))
			return false;
		if (detectSpins.empty())
			return false;
		_cache.spin_names = detectSpinNames;
		_cache.spin_x.assign(detectSpins.size(), std::vector<double>(steps, 0.0));
		_cache.spin_y.assign(detectSpins.size(), std::vector<double>(steps, 0.0));
		_cache.spin_perp.assign(detectSpins.size(), std::vector<double>(steps, 0.0));
		_cache.spin_p.assign(detectSpins.size(), std::vector<double>(steps, 0.0));
		_cache.spin_m.assign(detectSpins.size(), std::vector<double>(steps, 0.0));

		arma::cx_mat rho0;
		bool hasInitialState = false;
		if (!this->initialStateName.empty())
		{
			auto state = _system->states_find(this->initialStateName);
			if (state != nullptr && space.GetState(state, rho0))
				hasInitialState = true;
		}
		if (!hasInitialState)
		{
			auto initial_states = _system->InitialState();
			if (initial_states.empty())
				return false;

			for (auto state = initial_states.cbegin(); state != initial_states.cend(); state++)
			{
				arma::cx_mat tmp;
				if (!space.GetState(*state, tmp))
					continue;

				if (!hasInitialState)
				{
					rho0 = tmp;
					hasInitialState = true;
				}
				else
				{
					rho0 += tmp;
				}
			}
		}

		if (!hasInitialState)
			return false;
		rho0 /= arma::trace(rho0);

		std::vector<arma::cx_mat> Sx_list(detectSpins.size());
		std::vector<arma::cx_mat> Sy_list(detectSpins.size());
		std::vector<arma::cx_mat> Sz_list(detectSpins.size());
		for (size_t i = 0; i < detectSpins.size(); ++i)
		{
			auto spin = detectSpins[i];
			if (!space.CreateOperator(arma::conv_to<arma::cx_mat>::from(spin->Sx()), spin, Sx_list[i]) ||
				!space.CreateOperator(arma::conv_to<arma::cx_mat>::from(spin->Sy()), spin, Sy_list[i]) ||
				!space.CreateOperator(arma::conv_to<arma::cx_mat>::from(spin->Sz()), spin, Sz_list[i]))
			{
				return false;
			}
		}

		int numPoints = this->powdersamplingpoints;
		std::vector<std::tuple<double, double, double>> grid;
		const bool useSopheGrid = (this->powderGridType == "sophe");
		if (useSopheGrid)
		{
			int gridSize = this->powderGridSize;
			if (gridSize < 2)
			{
				double maxPhi = 0.0;
				bool closedPhi = false;
				int nOctants = 0;
				if (numPoints > 1 && this->SopheGridParams(this->powderGridSymmetry, maxPhi, closedPhi, nOctants))
				{
					int bestSize = 0;
					int bestDiff = std::numeric_limits<int>::max();
					for (int candidate = 2; candidate <= 200; ++candidate)
					{
						int count = SopheGridPointCount(candidate, nOctants, closedPhi);
						int diff = std::abs(count - numPoints);
						if (diff < bestDiff)
						{
							bestDiff = diff;
							bestSize = candidate;
							if (diff == 0)
								break;
						}
					}
					if (bestSize > 0)
						gridSize = bestSize;
				}
				if (gridSize < 2)
					gridSize = 19;
			}

			if (!this->CreateSopheGrid(gridSize, this->powderGridSymmetry, grid))
				return false;
			numPoints = static_cast<int>(grid.size());
		}
		else if (numPoints > 1)
		{
			if (!this->CreateUniformGrid(numPoints, grid))
				return false;
		}
		else
		{
			grid.clear();
			grid.emplace_back(0.0, 0.0, 1.0);
			numPoints = 1;
		}
		const int gamma_points = (numPoints > 1) ? std::max(1, this->powderGammaPoints) : 1;
		const double gamma_weight = useSopheGrid ? (2.0 * arma::datum::pi / static_cast<double>(gamma_points))
													: (1.0 / static_cast<double>(gamma_points));

		const double lwB_mT = std::abs(this->linewidth_mT);
		const double lwB_T = lwB_mT * 1.0e-3;
		const double lineWindow = (lwB_T > 0.0 && dBabs > 0.0) ? 6.0 * lwB_T : 0.0;

		arma::mat RFrame = arma::eye<arma::mat>(3, 3);
		{
			auto fr = _fieldInteraction->Framelist();
			double a = (fr.n_elem >= 1) ? fr(0) : 0.0;
			double b = (fr.n_elem >= 2) ? fr(1) : 0.0;
			double g = (fr.n_elem >= 3) ? fr(2) : 0.0;

			const double ca = std::cos(a), sa = std::sin(a);
			const double cb = std::cos(b), sb = std::sin(b);
			const double cg = std::cos(g), sg = std::sin(g);

			arma::mat Ra = {{ca, sa, 0.0}, {-sa, ca, 0.0}, {0.0, 0.0, 1.0}};
			arma::mat Rb = {{cb, 0.0, -sb}, {0.0, 1.0, 0.0}, {sb, 0.0, cb}};
			arma::mat Rg = {{cg, sg, 0.0}, {-sg, cg, 0.0}, {0.0, 0.0, 1.0}};
			RFrame = Rg * Rb * Ra;
		}

		std::vector<arma::mat> g_frame_base(detectSpins.size());
		for (size_t i = 0; i < detectSpins.size(); ++i)
		{
			arma::mat g_base = arma::conv_to<arma::mat>::from(detectSpins[i]->GetTensor().LabFrame());
			if (_fieldInteraction->IgnoreTensors())
				g_base = arma::eye<arma::mat>(3, 3);
			g_frame_base[i] = RFrame * g_base * RFrame.t();
		}

		double mu_prefactor = _fieldInteraction->Prefactor();
		if (_fieldInteraction->AddCommonPrefactor())
			mu_prefactor *= 8.79410005e+1;

		std::vector<std::string> zeelist;
		zeelist.push_back(_fieldInteraction->Name());

		const arma::uword dim = space.HilbertSpaceDimensions();
		std::vector<std::pair<arma::uword, arma::uword>> transitions;
		transitions.reserve(static_cast<size_t>(dim) * static_cast<size_t>(dim - 1) / 2);
		for (arma::uword m = 0; m < dim; ++m)
		{
			for (arma::uword n = m + 1; n < dim; ++n)
				transitions.emplace_back(m, n);
		}

		const arma::cx_double I(0.0, 1.0);
		const size_t spin_count = detectSpins.size();

		for (int grid_num = 0; grid_num < numPoints; ++grid_num)
		{
			auto [theta, phi, w_solid] = grid[grid_num];
			const double base_weight = w_solid;

			for (int gamma_idx = 0; gamma_idx < gamma_points; ++gamma_idx)
			{
				double gamma = 0.0;
				if (gamma_points > 1)
					gamma = 2.0 * arma::datum::pi * (static_cast<double>(gamma_idx) + 0.5) / static_cast<double>(gamma_points);

				const double w = base_weight * gamma_weight;

				arma::mat Rot;
				if (!this->CreateRotationMatrix(phi, theta, gamma, Rot))
					continue;

				const arma::mat Rpowder = Rot.t();

				arma::sp_cx_mat Hstatic_sp;
				if (h0list_noB.empty())
				{
					Hstatic_sp = arma::zeros<arma::sp_cx_mat>(dim, dim);
				}
				else if (!space.BaseHamiltonianRotated(h0list_noB, Rot, Hstatic_sp))
				{
					continue;
				}
				arma::cx_mat Hstatic = arma::cx_mat(Hstatic_sp);

				arma::sp_cx_mat Hz_sp;
				if (!space.BaseHamiltonianRotated(zeelist, Rot, Hz_sp))
					continue;
				arma::cx_mat dHdB = arma::cx_mat(Hz_sp) / field_T[0];

				std::vector<arma::cx_mat> mux_list(spin_count);
				std::vector<arma::cx_mat> muy_list(spin_count);
				bool tensor_dim_ok = true;
				for (size_t i = 0; i < spin_count; ++i)
				{
					arma::mat g = Rpowder * g_frame_base[i] * Rpowder.t();
					if (!this->fullTensorRotation)
						g = g % arma::eye<arma::mat>(3, 3);

					arma::cx_mat mux = g(0, 0) * Sx_list[i] + g(1, 0) * Sy_list[i] + g(2, 0) * Sz_list[i];
					arma::cx_mat muy = g(0, 1) * Sx_list[i] + g(1, 1) * Sy_list[i] + g(2, 1) * Sz_list[i];

					if (mu_prefactor != 1.0)
					{
						mux *= mu_prefactor;
						muy *= mu_prefactor;
					}

					if (mux.n_rows != spaceDim || mux.n_cols != spaceDim || muy.n_rows != spaceDim || muy.n_cols != spaceDim)
						tensor_dim_ok = false;
					mux_list[i] = mux;
					muy_list[i] = muy;
				}
				if (!tensor_dim_ok)
					continue;

				std::vector<double> prev_delta(transitions.size(), 0.0);
				arma::cx_mat prev_eigvec;
				bool have_prev = false;

				for (unsigned int step = 0; step < steps; ++step)
				{
					arma::cx_mat H = Hstatic + field_T[step] * dHdB;
					arma::vec eigval;
					arma::cx_mat eigvec;
					if (!arma::eig_sym(eigval, eigvec, H))
						continue;

					std::vector<double> curr_delta(transitions.size(), 0.0);

					for (size_t t = 0; t < transitions.size(); ++t)
					{
						const auto [m, n] = transitions[t];
						curr_delta[t] = (eigval(n) - eigval(m)) - omega_mw;
					}

					if (!have_prev)
					{
						prev_delta = curr_delta;
						prev_eigvec = eigvec;
						have_prev = true;
						continue;
					}

					for (size_t t = 0; t < transitions.size(); ++t)
					{
						const double d1 = prev_delta[t];
						const double d2 = curr_delta[t];
						if (d1 == 0.0 && d2 == 0.0)
							continue;
						if (d1 == 0.0 || d2 == 0.0 || (d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0))
						{
							const double denom = (d1 - d2);
							if (std::abs(denom) < 1e-15)
								continue;
							const double tfrac = d1 / denom;
							const double Bres = field_T[step - 1] + tfrac * (field_T[step] - field_T[step - 1]);

							const bool use_prev = std::abs(d1) <= std::abs(d2);
							const arma::cx_mat &eigvec_use = use_prev ? prev_eigvec : eigvec;

							const auto [m, n] = transitions[t];
							const arma::cx_vec Um = eigvec_use.col(m);
							const arma::cx_vec Vn = eigvec_use.col(n);

							const double population = std::real(arma::cdot(Um, rho0 * Um)) - std::real(arma::cdot(Vn, rho0 * Vn));
							if (std::abs(population) < 1e-15)
								continue;

							const arma::cx_double domega_dB_c = arma::cdot(Vn - Um, dHdB * (Vn + Um));
							const double abs_domega_dB = std::abs(std::real(domega_dB_c));
							if (!std::isfinite(abs_domega_dB) || abs_domega_dB < 1e-15)
								continue;
							const double dBdE = 1.0 / abs_domega_dB;
							if (dBdE > 1e5)
								continue;

							std::vector<double> amp_spin_x(spin_count, 0.0);
							std::vector<double> amp_spin_y(spin_count, 0.0);
							std::vector<double> amp_spin_perp(spin_count, 0.0);
							std::vector<double> amp_spin_p(spin_count, 0.0);
							std::vector<double> amp_spin_m(spin_count, 0.0);

							arma::cx_double muTx(0.0, 0.0);
							arma::cx_double muTy(0.0, 0.0);
							double I_sum_x = 0.0;
							double I_sum_y = 0.0;

							for (size_t i = 0; i < spin_count; ++i)
							{
								const arma::cx_double muix = arma::cdot(Um, mux_list[i] * Vn);
								const arma::cx_double muiy = arma::cdot(Um, muy_list[i] * Vn);
								muTx += muix;
								muTy += muiy;

								const double Iix = std::norm(muix);
								const double Iiy = std::norm(muiy);
								I_sum_x += Iix;
								I_sum_y += Iiy;

								amp_spin_x[i] = population * Iix * dBdE;
								amp_spin_y[i] = population * Iiy * dBdE;
								amp_spin_perp[i] = population * 0.5 * (Iix + Iiy) * dBdE;

								const arma::cx_double mup = muix + I * muiy;
								const arma::cx_double mum = muix - I * muiy;
								amp_spin_p[i] = population * std::norm(mup) * dBdE;
								amp_spin_m[i] = population * std::norm(mum) * dBdE;
							}

							const double ITx = std::norm(muTx);
							const double ITy = std::norm(muTy);
							const double ICx = ITx - I_sum_x;
							const double ICy = ITy - I_sum_y;

							const double amp_total_x = population * ITx * dBdE;
							const double amp_total_y = population * ITy * dBdE;
							const double amp_total_perp = population * 0.5 * (ITx + ITy) * dBdE;
							const double amp_crossx = population * ICx * dBdE;
							const double amp_crossy = population * ICy * dBdE;

							if (lwB_mT <= 0.0 || dBabs == 0.0)
							{
								size_t idx = 0;
								if (dBstep != 0.0)
								{
									const double pos = (Bres - field_T[0]) / dBstep;
									idx = static_cast<size_t>(std::llround(pos));
								}

								if (idx < steps)
								{
									_cache.total_x[idx] += w * amp_total_x;
									_cache.total_y[idx] += w * amp_total_y;
									_cache.total_perp[idx] += w * amp_total_perp;
									_cache.cross_x[idx] += w * amp_crossx;
									_cache.cross_y[idx] += w * amp_crossy;

									for (size_t i = 0; i < spin_count; ++i)
									{
										_cache.spin_x[i][idx] += w * amp_spin_x[i];
										_cache.spin_y[i][idx] += w * amp_spin_y[i];
										_cache.spin_perp[i][idx] += w * amp_spin_perp[i];
										_cache.spin_p[i][idx] += w * amp_spin_p[i];
										_cache.spin_m[i][idx] += w * amp_spin_m[i];
									}
								}
							}
							else
							{
								size_t start = 0;
								size_t end = steps - 1;
								if (lineWindow > 0.0 && dBabs > 0.0)
								{
									const int half = static_cast<int>(std::ceil(lineWindow / dBabs));
									const double pos = (Bres - field_T[0]) / dBstep;
									const int center = static_cast<int>(std::llround(pos));
									start = static_cast<size_t>(std::max(0, center - half));
									end = static_cast<size_t>(std::min(static_cast<int>(steps) - 1, center + half));
								}

								for (size_t j = start; j <= end; ++j)
								{
									const double deltaB_mT = (field_T[j] - Bres) * 1.0e3;
									const double L = this->LineshapeValue(deltaB_mT, lwB_mT);
									if (L == 0.0)
										continue;

									const double weight = w * L;
									_cache.total_x[j] += weight * amp_total_x;
									_cache.total_y[j] += weight * amp_total_y;
									_cache.total_perp[j] += weight * amp_total_perp;
									_cache.cross_x[j] += weight * amp_crossx;
									_cache.cross_y[j] += weight * amp_crossy;

									for (size_t i = 0; i < spin_count; ++i)
									{
										_cache.spin_x[i][j] += weight * amp_spin_x[i];
										_cache.spin_y[i][j] += weight * amp_spin_y[i];
										_cache.spin_perp[i][j] += weight * amp_spin_perp[i];
										_cache.spin_p[i][j] += weight * amp_spin_p[i];
										_cache.spin_m[i][j] += weight * amp_spin_m[i];
									}
								}
							}
						}
					}

					prev_delta.swap(curr_delta);
					prev_eigvec = eigvec;
				}
			}
		}

		return true;
	}

	void TaskStaticHSTrEPRSpectra::WriteHeader(std::ostream &_stream)
	{
		_stream << "Step ";
		_stream << "Time ";
		this->WriteStandardOutputHeader(_stream);

		auto systems = this->SpinSystems();
		for (auto i = systems.cbegin(); i != systems.cend(); i++)
		{
			SpinAPI::interaction_ptr fieldInteraction = nullptr;
			std::vector<SpinAPI::spin_ptr> detectSpins;
			std::vector<std::string> detectSpinNames;
			this->ResolveFieldInteraction((*i), fieldInteraction);
			if (!this->ResolveDetectionSpins((*i), fieldInteraction, detectSpins, detectSpinNames))
			{
				detectSpinNames.clear();
			}

			_stream << (*i)->Name() << ".Field_mT ";
			_stream << (*i)->Name() << ".Total_x ";
			_stream << (*i)->Name() << ".Total_y ";
			_stream << (*i)->Name() << ".Total_perp ";
			_stream << (*i)->Name() << ".Cross_x ";
			_stream << (*i)->Name() << ".Cross_y ";

			for (const auto &spinName : detectSpinNames)
			{
				_stream << (*i)->Name() << "." << spinName << "_x ";
				_stream << (*i)->Name() << "." << spinName << "_y ";
				_stream << (*i)->Name() << "." << spinName << "_perp ";
				_stream << (*i)->Name() << "." << spinName << "_p ";
				_stream << (*i)->Name() << "." << spinName << "_m ";
			}
		}

		_stream << std::endl;
	}

	bool TaskStaticHSTrEPRSpectra::Validate()
	{
		bool hasFrequency = this->Properties()->Get("mwfrequency", this->mwFrequencyGHz);
		if (!hasFrequency)
			hasFrequency = this->Properties()->Get("frequency", this->mwFrequencyGHz);
		if (!hasFrequency)
			hasFrequency = this->Properties()->Get("rffrequency", this->mwFrequencyGHz);
		if (!hasFrequency)
		{
			this->Log() << "Failed to obtain mwfrequency/frequency. Using frequency = 0 by default." << std::endl;
		}

		const bool hasLinewidth = this->Properties()->Get("linewidth", this->linewidth_mT);
		const bool hasLegacyFad = this->Properties()->Get("linewidth_fad", this->linewidthFad_mT);
		const bool hasLegacyDonor = this->Properties()->Get("linewidth_donor", this->linewidthDonor_mT);
		if (!hasLinewidth)
		{
			if (!hasLegacyFad && !hasLegacyDonor)
			{
				this->Log() << "Failed to obtain linewidth. Using linewidth = 0 by default." << std::endl;
			}
			this->linewidth_mT = 0.5 * (std::abs(this->linewidthFad_mT) + std::abs(this->linewidthDonor_mT));
		}

		if (this->Properties()->Get("lineshape", this->lineshape))
		{
			this->lineshape = ToLower(this->lineshape);
		}
		else
		{
			this->lineshape = "gaussian";
		}

		if (!this->Properties()->Get("powdersamplingpoints", this->powdersamplingpoints))
		{
			this->powdersamplingpoints = 0;
		}

		if (this->Properties()->Get("powdergridtype", this->powderGridType))
		{
			this->powderGridType = ToLower(this->powderGridType);
		}
		else
		{
			this->powderGridType = "fibonacci";
		}
		this->Properties()->Get("powdergridsymmetry", this->powderGridSymmetry);
		if (!this->Properties()->Get("powdergridsize", this->powderGridSize))
		{
			this->powderGridSize = 0;
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

		this->detectSpinNames.clear();
		this->Properties()->GetList("detectspins", this->detectSpinNames, ',');
		this->Properties()->Get("electron1", this->electron1Name);
		this->Properties()->Get("electron2", this->electron2Name);

		this->Properties()->Get("fieldinteraction", this->fieldInteractionName);
		this->Properties()->Get("initialstate", this->initialStateName);

		if (this->Properties()->GetList("hamiltonianh0list", this->hamiltonianH0list, ','))
		{
			this->Log() << "HamiltonianH0list = [";
			for (size_t j = 0; j < this->hamiltonianH0list.size(); j++)
			{
				this->Log() << this->hamiltonianH0list[j];
				if (j < this->hamiltonianH0list.size() - 1)
					this->Log() << ", ";
			}
			this->Log() << "]" << std::endl;
		}

		return true;
	}
}
