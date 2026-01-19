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
	}

	// -----------------------------------------------------
	// TaskStaticHSTrEPRSpectra Constructors and Destructor
	// -----------------------------------------------------
	TaskStaticHSTrEPRSpectra::TaskStaticHSTrEPRSpectra(const MSDParser::ObjectParser &_parser, const RunSection &_runsection)
		: BasicTask(_parser, _runsection),
		  mwFrequencyGHz(0.0),
		  linewidthFad_mT(0.0),
		  linewidthDonor_mT(0.0),
		  lineshape("gaussian"),
		  powdersamplingpoints(0),
		  powderGammaPoints(1),
		  powderFullSphere(true),
		  fullTensorRotation(true),
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

		// Microwave angular frequency (rad/ns)
		const double omega_mw = 2.0 * arma::datum::pi * this->mwFrequencyGHz;

		// Loop through all SpinSystems
		auto systems = this->SpinSystems();
		for (auto sysIt = systems.cbegin(); sysIt != systems.cend(); sysIt++)
		{
			this->Log() << "\nStarting with SpinSystem \"" << (*sysIt)->Name() << "\"." << std::endl;

			SpinAPI::SpinSpace space(*(*sysIt));
			space.UseSuperoperatorSpace(false);
			space.UseFullTensorRotation(this->fullTensorRotation);
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

			// Find electron spins
			auto electron1 = (*sysIt)->spins_find(this->electron1Name);
			auto electron2 = (*sysIt)->spins_find(this->electron2Name);
			if (electron1 == nullptr || electron2 == nullptr)
			{
				this->Log() << "Failed to find electron spins \"" << this->electron1Name << "\" and/or \"" << this->electron2Name << "\" in SpinSystem \"" << (*sysIt)->Name() << "\"." << std::endl;
				continue;
			}

			std::string electron1Type;
			std::string electron2Type;
			electron1->Properties()->Get("type", electron1Type);
			electron2->Properties()->Get("type", electron2Type);
			if (electron1Type != "electron" || electron2Type != "electron")
			{
				this->Log() << "Spin \"" << electron1->Name() << "\" or \"" << electron2->Name() << "\" is not of type electron." << std::endl;
				continue;
			}

			// Determine Zeeman interaction used for B and for dH/dB (Jacobian)
			SpinAPI::interaction_ptr fieldInteraction = nullptr;
			if (!this->fieldInteractionName.empty())
			{
				fieldInteraction = (*sysIt)->interactions_find(this->fieldInteractionName);
			}
			if (fieldInteraction == nullptr)
			{
				for (auto inter = (*sysIt)->interactions_cbegin(); inter != (*sysIt)->interactions_cend(); inter++)
				{
					std::string type;
					if ((*inter)->Properties()->Get("type", type))
					{
						type = ToLower(type);
						if (type == "zeeman")
						{
							fieldInteraction = (*inter);
							break;
						}
					}
				}
			}
			if (fieldInteraction == nullptr)
			{
				this->Log() << "No Zeeman interaction found in SpinSystem \"" << (*sysIt)->Name() << "\". Need a Zeeman interaction for field->frequency mapping." << std::endl;
				continue;
			}

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

			// Embed *bare* electron spin operators (Sx,Sy,Sz) into the full Hilbert space.
			// We deliberately build mu operators from rotated g-tensors per orientation (pepper-equivalent).
			arma::cx_mat Sx1, Sy1, Sz1;
			arma::cx_mat Sx2, Sy2, Sz2;
			if (!space.CreateOperator(arma::conv_to<arma::cx_mat>::from(electron1->Sx()), electron1, Sx1) ||
				!space.CreateOperator(arma::conv_to<arma::cx_mat>::from(electron1->Sy()), electron1, Sy1) ||
				!space.CreateOperator(arma::conv_to<arma::cx_mat>::from(electron1->Sz()), electron1, Sz1))
			{
				this->Log() << "Failed to build bare spin operators for electron \"" << electron1->Name() << "\"." << std::endl;
				continue;
			}
			if (!space.CreateOperator(arma::conv_to<arma::cx_mat>::from(electron2->Sx()), electron2, Sx2) ||
				!space.CreateOperator(arma::conv_to<arma::cx_mat>::from(electron2->Sy()), electron2, Sy2) ||
				!space.CreateOperator(arma::conv_to<arma::cx_mat>::from(electron2->Sz()), electron2, Sz2))
			{
				this->Log() << "Failed to build bare spin operators for electron \"" << electron2->Name() << "\"." << std::endl;
				continue;
			}

			// Build powder grid (theta,phi) and optional gamma sampling.
			int numPoints = this->powdersamplingpoints;
			std::vector<std::tuple<double, double, double>> grid;
			if (numPoints > 1)
			{
				if (!this->CreateUniformGrid(numPoints, grid))
				{
					this->Log() << "Failed to obtain a uniform grid for powder averaging." << std::endl;
					continue;
				}
				this->Log() << "Using powder averaging with " << numPoints << " orientations." << std::endl;
				if (this->powderGammaPoints > 1)
					this->Log() << "Sampling gamma with " << this->powderGammaPoints << " points per orientation." << std::endl;
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
			const double gamma_weight = 1.0 / static_cast<double>(gamma_points);

			// Field-domain linewidth (FWHM, Tesla). Pepper broadens in field, not in frequency.
			// If linewidth_fad and linewidth_donor differ, we use their mean as a single experimental broadening.
			const double lwB_mT = 0.5 * (std::abs(this->linewidthFad_mT) + std::abs(this->linewidthDonor_mT));
			const double lwB_T = lwB_mT * 1.0e-3;

			// Precompute interaction-frame rotation for the Zeeman interaction (g-tensor frame).
			// IMPORTANT: Interaction framelists must be interpreted with EasySpin's erot ZXZ PASSIVE convention.
			// SpinSpace::InteractionOperatorRotated() uses the same convention internally.
			arma::mat RFrame = arma::eye<arma::mat>(3, 3);
			{
				auto fr = fieldInteraction->Framelist();
				double a = (fr.n_elem >= 1) ? fr(0) : 0.0;
				double b = (fr.n_elem >= 2) ? fr(1) : 0.0;
				double g = (fr.n_elem >= 3) ? fr(2) : 0.0;

				// EasySpin erot.m matrix (passive ZXZ): R = Rz(g)*Ry(b)*Rz(a)
				const double ca = std::cos(a), sa = std::sin(a);
				const double cb = std::cos(b), sb = std::sin(b);
				const double cg = std::cos(g), sg = std::sin(g);

				arma::mat Ra = {{ca, sa, 0.0}, {-sa, ca, 0.0}, {0.0, 0.0, 1.0}};
				arma::mat Rb = {{cb, 0.0, -sb}, {0.0, 1.0, 0.0}, {sb, 0.0, cb}};
				arma::mat Rg = {{cg, sg, 0.0}, {-sg, cg, 0.0}, {0.0, 0.0, 1.0}};
				RFrame = Rg * Rb * Ra;
			}

			// Base g-tensors (as specified on spins)
			arma::mat g1_base = arma::conv_to<arma::mat>::from(electron1->GetTensor().LabFrame());
			arma::mat g2_base = arma::conv_to<arma::mat>::from(electron2->GetTensor().LabFrame());

			// Accumulators
			double total_x = 0.0;
			double total_y = 0.0;
			double total_perp = 0.0;
			double fadx = 0.0;
			double donorx = 0.0;
			double fady = 0.0;
			double donory = 0.0;
			double crossx = 0.0;
			double crossy = 0.0;
			double fadp = 0.0;
			double fadm = 0.0;
			double donorp = 0.0;
			double donorm = 0.0;

			// For dH/dB we need the Zeeman Hamiltonian only (rotated per orientation)
			std::vector<std::string> zeelist;
			zeelist.push_back(fieldInteraction->Name());

			const arma::cx_double I(0.0, 1.0);

#ifdef _OPENMP
#pragma omp parallel for reduction(+ : total_x, total_y, total_perp, fadx, donorx, fady, donory, crossx, crossy, fadp, fadm, donorp, donorm)
#endif
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

					// Rot is an ACTIVE rotation. SpinSpace::InteractionOperatorRotated() will transpose it internally
					// to obtain the PASSIVE tensor rotation used for anisotropic couplings. To keep microwave operators
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

					// Zeeman-only rotated Hamiltonian -> dH/dB magnitude for Jacobian (pepper dBdE mapping)
					arma::sp_cx_mat Hz_sp;
					if (!space.BaseHamiltonianRotated(zeelist, Rot, Hz_sp))
						continue;
					arma::cx_mat dHdB = arma::cx_mat(Hz_sp) / Bmag; // rad/ns/T
					arma::cx_mat dHdB_eig = Udag * dHdB * eigvec;
					arma::vec dHdB_diag = arma::real(arma::diagvec(dHdB_eig));

					// Rotate g-tensors to interaction frame, then to lab using the PASSIVE powder rotation (Rpowder).
					arma::mat g1 = RFrame * g1_base * RFrame.t();
					arma::mat g2 = RFrame * g2_base * RFrame.t();
					g1 = Rpowder * g1 * Rpowder.t();
					g2 = Rpowder * g2 * Rpowder.t();
					if (!this->fullTensorRotation)
					{
						g1 = g1 % arma::eye<arma::mat>(3, 3);
						g2 = g2 % arma::eye<arma::mat>(3, 3);
					}

					// mu_j = sum_i g_{i,j} S_i  (j = x,y in lab)
					arma::cx_mat mux1 = g1(0, 0) * Sx1 + g1(1, 0) * Sy1 + g1(2, 0) * Sz1;
					arma::cx_mat muy1 = g1(0, 1) * Sx1 + g1(1, 1) * Sy1 + g1(2, 1) * Sz1;
					arma::cx_mat mux2 = g2(0, 0) * Sx2 + g2(1, 0) * Sy2 + g2(2, 0) * Sz2;
					arma::cx_mat muy2 = g2(0, 1) * Sx2 + g2(1, 1) * Sy2 + g2(2, 1) * Sz2;

					arma::cx_mat muxT = mux1 + mux2;
					arma::cx_mat muyT = muy1 + muy2;

					arma::cx_mat mup1 = mux1 + I * muy1;
					arma::cx_mat mum1 = mux1 - I * muy1;
					arma::cx_mat mup2 = mux2 + I * muy2;
					arma::cx_mat mum2 = mux2 - I * muy2;

					// Transform mu operators into eigenbasis
					arma::cx_mat mux1_eig = Udag * mux1 * eigvec;
					arma::cx_mat mux2_eig = Udag * mux2 * eigvec;
					arma::cx_mat muy1_eig = Udag * muy1 * eigvec;
					arma::cx_mat muy2_eig = Udag * muy2 * eigvec;
					arma::cx_mat muxT_eig = Udag * muxT * eigvec;
					arma::cx_mat muyT_eig = Udag * muyT * eigvec;
					arma::cx_mat mup1_eig = Udag * mup1 * eigvec;
					arma::cx_mat mum1_eig = Udag * mum1 * eigvec;
					arma::cx_mat mup2_eig = Udag * mup2 * eigvec;
					arma::cx_mat mum2_eig = Udag * mum2 * eigvec;

					const arma::uword dim = eigval.n_elem;

					double loc_total_x = 0.0;
					double loc_total_y = 0.0;
					double loc_total_perp = 0.0;
					double loc_fadx = 0.0;
					double loc_donorx = 0.0;
					double loc_fady = 0.0;
					double loc_donory = 0.0;
					double loc_crossx = 0.0;
					double loc_crossy = 0.0;
					double loc_fadp = 0.0;
					double loc_fadm = 0.0;
					double loc_donorp = 0.0;
					double loc_donorm = 0.0;

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
							const double domega_dB = dHdB_diag(n) - dHdB_diag(m); // rad/ns/T
							const double abs_domega_dB = std::abs(domega_dB);
							if (!std::isfinite(abs_domega_dB) || abs_domega_dB < 1e-15)
								continue;

							const double dBdE = 1.0 / abs_domega_dB; // T / (rad/ns)
							// EasySpin/pepper safeguard: the 1/g factor (dB/dE) can diverge when
							// d(E_n-E_m)/dB \approx 0 (near avoided crossings / degeneracies).
							// Pepper aborts if this gets too large; we emulate that behavior by
							// skipping those transitions to avoid unphysical spikes.
							if (dBdE > 1e5)
								continue;

							const double deltaB = deltaOmega * dBdE;  // T

							const double L = this->LineshapeValue(deltaB, lwB_T);
							if (L == 0.0)
								continue;

							const double wField = dBdE * L;

							// x-channel
							const arma::cx_double mu1x = mux1_eig(m, n);
							const arma::cx_double mu2x = mux2_eig(m, n);
							const arma::cx_double muTx = muxT_eig(m, n);
							const double I1x = std::norm(mu1x);
							const double I2x = std::norm(mu2x);
							const double ITx = std::norm(muTx);
							const double ICx = 2.0 * std::real(mu1x * std::conj(mu2x));

							// y-channel
							const arma::cx_double mu1y = muy1_eig(m, n);
							const arma::cx_double mu2y = muy2_eig(m, n);
							const arma::cx_double muTy = muyT_eig(m, n);
							const double I1y = std::norm(mu1y);
							const double I2y = std::norm(mu2y);
							const double ITy = std::norm(muTy);
							const double ICy = 2.0 * std::real(mu1y * std::conj(mu2y));

							loc_total_x += population * ITx * wField;
							loc_total_y += population * ITy * wField;
							loc_total_perp += population * 0.5 * (ITx + ITy) * wField;

							loc_fadx += population * I1x * wField;
							loc_donorx += population * I2x * wField;
							loc_crossx += population * ICx * wField;

							loc_fady += population * I1y * wField;
							loc_donory += population * I2y * wField;
							loc_crossy += population * ICy * wField;

							// circular components (for diagnostics)
							loc_fadp += population * std::norm(mup1_eig(m, n)) * wField;
							loc_fadm += population * std::norm(mum1_eig(m, n)) * wField;
							loc_donorp += population * std::norm(mup2_eig(m, n)) * wField;
							loc_donorm += population * std::norm(mum2_eig(m, n)) * wField;
						}
					}

					total_x += w * loc_total_x;
					total_y += w * loc_total_y;
					total_perp += w * loc_total_perp;
					fadx += w * loc_fadx;
					donorx += w * loc_donorx;
					fady += w * loc_fady;
					donory += w * loc_donory;
					crossx += w * loc_crossx;
					crossy += w * loc_crossy;
					fadp += w * loc_fadp;
					fadm += w * loc_fadm;
					donorp += w * loc_donorp;
					donorm += w * loc_donorm;
				}
			}

			// Output
			this->Data() << this->RunSettings()->CurrentStep() << " ";
			this->Data() << this->RunSettings()->Time() << " ";
			this->WriteStandardOutput(this->Data());
			this->Data() << field_mT << " "
						<< total_x << " " << total_y << " " << total_perp << " "
						<< fadx << " " << donorx << " " << crossx << " "
						<< fady << " " << donory << " " << crossy << " "
						<< fadp << " " << fadm << " " << donorp << " " << donorm
						<< std::endl;
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
			return 1.0 / (1.0 + 4.0 * x * x);
		}

		return std::exp(-4.0 * std::log(2.0) * x * x);
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
		// This implementation matches the convention used in other HS powder tasks in MolSpin:
		// R = Rz(alpha) * Ry(beta) * Rz(gamma) with
		// Rz = [[c,-s,0],[s,c,0],[0,0,1]] and Ry = [[c,0,s],[0,1,0],[-s,0,c]].
		const double ca = std::cos(_alpha), sa = std::sin(_alpha);
		const double cb = std::cos(_beta), sb = std::sin(_beta);
		const double cg = std::cos(_gamma), sg = std::sin(_gamma);

		arma::mat R1 = {
			{ca, -sa, 0.0},
			{sa,  ca, 0.0},
			{0.0, 0.0, 1.0}};

		arma::mat R2 = {
			{cb, 0.0, sb},
			{0.0, 1.0, 0.0},
			{-sb, 0.0, cb}};

		arma::mat R3 = {
			{cg, -sg, 0.0},
			{sg,  cg, 0.0},
			{0.0, 0.0, 1.0}};

		_R = R1 * R2 * R3;
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

	void TaskStaticHSTrEPRSpectra::WriteHeader(std::ostream &_stream)
	{
		_stream << "Step ";
		_stream << "Time ";
		this->WriteStandardOutputHeader(_stream);

		auto systems = this->SpinSystems();
		for (auto i = systems.cbegin(); i != systems.cend(); i++)
		{
			_stream << (*i)->Name() << ".Field_mT ";
			_stream << (*i)->Name() << ".Total_x ";
			_stream << (*i)->Name() << ".Total_y ";
			_stream << (*i)->Name() << ".Total_perp ";

			_stream << (*i)->Name() << ".FADx ";
			_stream << (*i)->Name() << ".Donorx ";
			_stream << (*i)->Name() << ".Cross_x ";

			_stream << (*i)->Name() << ".FADy ";
			_stream << (*i)->Name() << ".Donory ";
			_stream << (*i)->Name() << ".Cross_y ";

			_stream << (*i)->Name() << ".FADp ";
			_stream << (*i)->Name() << ".FADm ";
			_stream << (*i)->Name() << ".Donorp ";
			_stream << (*i)->Name() << ".Donorm ";
		}

		_stream << std::endl;
	}

	bool TaskStaticHSTrEPRSpectra::Validate()
	{
		if (!this->Properties()->Get("mwfrequency", this->mwFrequencyGHz))
		{
			this->Log() << "Failed to obtain mwfrequency. Using mwfrequency = 0 by default." << std::endl;
		}

		if (!this->Properties()->Get("linewidth_fad", this->linewidthFad_mT))
		{
			this->Log() << "Failed to obtain linewidth_fad. Using linewidth_fad = 0 by default." << std::endl;
		}

		if (!this->Properties()->Get("linewidth_donor", this->linewidthDonor_mT))
		{
			this->Log() << "Failed to obtain linewidth_donor. Using linewidth_donor = 0 by default." << std::endl;
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

		if (!this->Properties()->Get("electron1", this->electron1Name))
		{
			this->Log() << "Failed to obtain electron1 name. Please specify electron1 = <spin>;" << std::endl;
			return false;
		}

		if (!this->Properties()->Get("electron2", this->electron2Name))
		{
			this->Log() << "Failed to obtain electron2 name. Please specify electron2 = <spin>;" << std::endl;
			return false;
		}

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
