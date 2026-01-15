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

		const double omega_mw = 2.0 * arma::datum::pi * this->mwFrequencyGHz;

		// Loop through all SpinSystems
		auto systems = this->SpinSystems();
		for (auto i = systems.cbegin(); i != systems.cend(); i++)
		{
			this->Log() << "\nStarting with SpinSystem \"" << (*i)->Name() << "\"." << std::endl;

			SpinAPI::SpinSpace space(*(*i));
			space.UseSuperoperatorSpace(false);

			// Build list of interactions to include in H0
			std::vector<std::string> h0list = this->hamiltonianH0list;
			if (h0list.empty())
			{
				for (const auto &interaction : (*i)->Interactions())
				{
					if (!SpinAPI::IsStatic(*interaction))
						continue;
					h0list.push_back(interaction->Name());
				}
			}

			if (h0list.empty())
			{
				this->Log() << "No interactions specified for Hamiltonian H0 in SpinSystem \"" << (*i)->Name() << "\". Skipping." << std::endl;
				continue;
			}

			// Find electron spins
			auto electron1 = (*i)->spins_find(this->electron1Name);
			auto electron2 = (*i)->spins_find(this->electron2Name);
			if (electron1 == nullptr || electron2 == nullptr)
			{
				this->Log() << "Failed to find electron spins \"" << this->electron1Name << "\" and/or \"" << this->electron2Name << "\" in SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
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

			// Build initial density matrix
			arma::cx_mat rho0;
			bool hasInitialState = false;

			if (!this->initialStateName.empty())
			{
				auto state = (*i)->states_find(this->initialStateName);
				if (state == nullptr)
				{
					this->Log() << "Initial state \"" << this->initialStateName << "\" not found in SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
				}
				else
				{
					if (space.GetState(state, rho0))
					{
						hasInitialState = true;
					}
				}
			}

			if (!hasInitialState)
			{
				auto initial_states = (*i)->InitialState();
				if (initial_states.empty())
				{
					this->Log() << "Skipping SpinSystem \"" << (*i)->Name() << "\" as no initial state was specified." << std::endl;
					continue;
				}

				for (auto state = initial_states.cbegin(); state != initial_states.cend(); state++)
				{
					arma::cx_mat tmp;
					if (!space.GetState(*state, tmp))
					{
						this->Log() << "Failed to obtain projection matrix onto state \"" << (*state)->Name() << "\" of SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
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
				this->Log() << "Failed to construct initial state for SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
				continue;
			}

			rho0 /= arma::trace(rho0);

			// Build magnetization operators in the Hilbert space
			arma::cx_mat Mx1;
			arma::cx_mat Mx2;
			arma::cx_mat Mp1;
			arma::cx_mat Mm1;
			arma::cx_mat Mp2;
			arma::cx_mat Mm2;
			if (!space.CreateOperator(arma::conv_to<arma::cx_mat>::from(electron1->Tx()), electron1, Mx1))
			{
				this->Log() << "Failed to build magnetization operator for electron \"" << electron1->Name() << "\"." << std::endl;
				continue;
			}
			if (!space.CreateOperator(arma::conv_to<arma::cx_mat>::from(electron2->Tx()), electron2, Mx2))
			{
				this->Log() << "Failed to build magnetization operator for electron \"" << electron2->Name() << "\"." << std::endl;
				continue;
			}
			if (!space.CreateOperator(arma::conv_to<arma::cx_mat>::from(electron1->Sp()), electron1, Mp1))
			{
				this->Log() << "Failed to build S+ operator for electron \"" << electron1->Name() << "\"." << std::endl;
				continue;
			}
			if (!space.CreateOperator(arma::conv_to<arma::cx_mat>::from(electron1->Sm()), electron1, Mm1))
			{
				this->Log() << "Failed to build S- operator for electron \"" << electron1->Name() << "\"." << std::endl;
				continue;
			}
			if (!space.CreateOperator(arma::conv_to<arma::cx_mat>::from(electron2->Sp()), electron2, Mp2))
			{
				this->Log() << "Failed to build S+ operator for electron \"" << electron2->Name() << "\"." << std::endl;
				continue;
			}
			if (!space.CreateOperator(arma::conv_to<arma::cx_mat>::from(electron2->Sm()), electron2, Mm2))
			{
				this->Log() << "Failed to build S- operator for electron \"" << electron2->Name() << "\"." << std::endl;
				continue;
			}

			double giso_fad = electron1->GetTensor().Isotropic();
			double giso_donor = electron2->GetTensor().Isotropic();
			if (!std::isfinite(giso_fad) || giso_fad == 0.0)
				giso_fad = 2.0023;
			if (!std::isfinite(giso_donor) || giso_donor == 0.0)
				giso_donor = 2.0023;

			const double linewidth_fad = this->LinewidthToOmega(this->linewidthFad_mT, giso_fad);
			const double linewidth_donor = this->LinewidthToOmega(this->linewidthDonor_mT, giso_donor);

			// Construct grid
			int numPoints = this->powdersamplingpoints;
			std::vector<std::tuple<double, double, double>> grid;
			if (numPoints > 1)
			{
				if (!this->CreateUniformGrid(numPoints, grid))
				{
					this->Log() << "Failed to obtain a uniform grid for powder averaging." << std::endl;
					continue;
				}
			}
			else
			{
				grid.clear();
				grid.emplace_back(0.0, 0.0, 1.0);
				numPoints = 1;
			}

			double total_intensity = 0.0;
			double fad_intensity = 0.0;
			double donor_intensity = 0.0;
			double fadp_intensity = 0.0;
			double fadm_intensity = 0.0;
			double donorp_intensity = 0.0;
			double donorm_intensity = 0.0;

			#ifdef _OPENMP
			#pragma omp parallel for reduction(+ : total_intensity, fad_intensity, donor_intensity, fadp_intensity, fadm_intensity, donorp_intensity, donorm_intensity)
			#endif
			for (int grid_num = 0; grid_num < numPoints; ++grid_num)
			{
				auto [theta, phi, weight] = grid[grid_num];

				arma::mat Rot_mat;
				double gamma = 0.0;
				if (!this->CreateRotationMatrix(gamma, theta, phi, Rot_mat))
				{
					continue;
				}

				arma::sp_cx_mat H0_sp;
				if (!space.BaseHamiltonianRotated(h0list, Rot_mat, H0_sp))
				{
					continue;
				}

				arma::cx_mat H = arma::cx_mat(H0_sp);
				arma::vec eigval;
				arma::cx_mat eigvec;
				if (!arma::eig_sym(eigval, eigvec, H))
				{
					continue;
				}

				arma::cx_mat Udag = arma::trans(arma::conj(eigvec));
				arma::cx_mat rho_eig = Udag * rho0 * eigvec;
				arma::cx_mat Mx1_eig = Udag * Mx1 * eigvec;
				arma::cx_mat Mx2_eig = Udag * Mx2 * eigvec;
				arma::cx_mat Mp1_eig = Udag * Mp1 * eigvec;
				arma::cx_mat Mm1_eig = Udag * Mm1 * eigvec;
				arma::cx_mat Mp2_eig = Udag * Mp2 * eigvec;
				arma::cx_mat Mm2_eig = Udag * Mm2 * eigvec;

				double fad_local = 0.0;
				double donor_local = 0.0;
				double fadp_local = 0.0;
				double fadm_local = 0.0;
				double donorp_local = 0.0;
				double donorm_local = 0.0;
				const arma::uword dim = eigval.n_elem;

				for (arma::uword m = 0; m < dim; ++m)
				{
					const double rho_mm = std::real(rho_eig(m, m));
					for (arma::uword n = m + 1; n < dim; ++n)
					{
						const double rho_nn = std::real(rho_eig(n, n));
						const double population = rho_mm - rho_nn;
						const double delta = (eigval(n) - eigval(m)) - omega_mw;

						const double mx1 = std::norm(Mx1_eig(m, n));
						const double mx2 = std::norm(Mx2_eig(m, n));
						const double mp1 = std::norm(Mp1_eig(m, n));
						const double mm1 = std::norm(Mm1_eig(m, n));
						const double mp2 = std::norm(Mp2_eig(m, n));
						const double mm2 = std::norm(Mm2_eig(m, n));

						fad_local += population * mx1 * this->LineshapeValue(delta, linewidth_fad);
						donor_local += population * mx2 * this->LineshapeValue(delta, linewidth_donor);
						fadp_local += population * mp1 * this->LineshapeValue(delta, linewidth_fad);
						fadm_local += population * mm1 * this->LineshapeValue(delta, linewidth_fad);
						donorp_local += population * mp2 * this->LineshapeValue(delta, linewidth_donor);
						donorm_local += population * mm2 * this->LineshapeValue(delta, linewidth_donor);
					}
				}

				fad_intensity += weight * fad_local;
				donor_intensity += weight * donor_local;
				fadp_intensity += weight * fadp_local;
				fadm_intensity += weight * fadm_local;
				donorp_intensity += weight * donorp_local;
				donorm_intensity += weight * donorm_local;
				total_intensity += weight * (fad_local + donor_local);
			}

			// Determine field strength for output
			double field_mT = 0.0;
			SpinAPI::interaction_ptr fieldInteraction = nullptr;
			if (!this->fieldInteractionName.empty())
			{
				fieldInteraction = (*i)->interactions_find(this->fieldInteractionName);
			}

			if (fieldInteraction == nullptr)
			{
				for (auto inter = (*i)->interactions_cbegin(); inter != (*i)->interactions_cend(); inter++)
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

			if (fieldInteraction != nullptr)
			{
				arma::vec field = fieldInteraction->Field();
				if (field.n_elem == 3)
				{
					if (std::abs(field(0)) < 1e-12 && std::abs(field(1)) < 1e-12)
					{
						field_mT = 1.0e3 * field(2);
					}
					else
					{
						field_mT = 1.0e3 * arma::norm(field);
					}
				}
			}

			this->Data() << this->RunSettings()->CurrentStep() << " ";
			this->Data() << this->RunSettings()->Time() << " ";
			this->WriteStandardOutput(this->Data());
			this->Data() << field_mT << " " << total_intensity << " " << fad_intensity << " " << donor_intensity
						 << " " << fadp_intensity << " " << fadm_intensity << " " << donorp_intensity << " " << donorm_intensity << std::endl;
		}

		this->Data() << std::endl;
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

			theta[i] = std::acos(1.0 - index / _Npoints);
			phi[i] = golden * index;
			weight[i] = std::sin(theta[i]) * 2 * arma::datum::pi / _Npoints;
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
			_stream << (*i)->Name() << ".Total ";
			_stream << (*i)->Name() << ".FAD ";
			_stream << (*i)->Name() << ".Donor ";
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
