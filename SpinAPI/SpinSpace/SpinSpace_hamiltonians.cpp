/////////////////////////////////////////////////////////////////////////
// SpinSpace class (SpinAPI Module)
// ------------------
// This source files contains methods to create matrix representations of
// interactions, and thus to generate Hamiltonians.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
namespace SpinAPI
{
	// -----------------------------------------------------
	// Hamiltonian representations in the space
	// -----------------------------------------------------
	// Returns the matrix representation of the Interaction object on the spin space
	bool SpinSpace::InteractionOperator(const interaction_ptr &_interaction, arma::cx_mat &_out) const
	{
		// Make sure the interaction is valid
		if (_interaction == nullptr)
			return false;

		// Create temporary matrix to hold the result
		arma::cx_mat tmp = arma::zeros<arma::cx_mat>(this->HilbertSpaceDimensions(), this->HilbertSpaceDimensions());

		// Get the interaction tensor
		auto ATensor = _interaction->CouplingTensor();

		if (_interaction->Type() == InteractionType::SingleSpin)
		{
			// Get the field at the current time or trajectory step
			arma::vec field;
			field = _interaction->Field();

			// Obtain the list of spins interacting with the field, and define matrices to hold the magnetic moment operators
			auto spinlist = _interaction->Group1();
			arma::cx_mat Sx;
			arma::cx_mat Sy;
			arma::cx_mat Sz;

			// Fill the matrix with the sum of all the interactions (i.e. between spin magnetic moment and fields)
			for (auto i = spinlist.cbegin(); i != spinlist.cend(); i++)
			{
				if (_interaction->IgnoreTensors())
				{
					this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Sx()), (*i), Sx);
					this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Sy()), (*i), Sy);
					this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Sz()), (*i), Sz);
				}
				else
				{
					this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Tx()), (*i), Sx);
					this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Ty()), (*i), Sy);
					this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Tz()), (*i), Sz);
				}

				if (ATensor != nullptr && !IsIsotropic(*ATensor))
				{
					// Use the tensor to calculate the product S * A * B
					auto A = ATensor->LabFrame();

					tmp += Sx * field(0) * A(0, 0) + Sx * field(1) * A(0, 1) + Sx * field(2) * A(0, 2);
					tmp += Sy * field(0) * A(1, 0) + Sy * field(1) * A(1, 1) + Sy * field(2) * A(1, 2);
					tmp += Sz * field(0) * A(2, 0) + Sz * field(1) * A(2, 1) + Sz * field(2) * A(2, 2);
				}
				else
				{
					tmp += Sx * field(0) + Sy * field(1) + Sz * field(2);
				}
			}
		}
		else if (_interaction->Type() == InteractionType::DoubleSpin)
		{
			// Obtain lists of interacting spins, coupling tensor, and define matrices to hold the magnetic moment operators
			auto spins1 = _interaction->Group1();
			auto spins2 = _interaction->Group2();
			arma::cx_mat S1x;
			arma::cx_mat S1y;
			arma::cx_mat S1z;
			arma::cx_mat S2x;
			arma::cx_mat S2y;
			arma::cx_mat S2z;

			// Fill the matrix with the sum of all the interactions
			for (auto i = spins1.cbegin(); i != spins1.cend(); i++)
			{
				for (auto j = spins2.cbegin(); j != spins2.cend(); j++)
				{
					// Obtain the magnetic moment operators within the Hilbert space
					if (_interaction->IgnoreTensors())
					{
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Sx()), (*i), S1x);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Sy()), (*i), S1y);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Sz()), (*i), S1z);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*j)->Sx()), (*j), S2x);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*j)->Sy()), (*j), S2y);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*j)->Sz()), (*j), S2z);
					}
					else
					{
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Tx()), (*i), S1x);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Ty()), (*i), S1y);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Tz()), (*i), S1z);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*j)->Tx()), (*j), S2x);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*j)->Ty()), (*j), S2y);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*j)->Tz()), (*j), S2z);
					}

					if (ATensor != nullptr && !IsIsotropic(*ATensor))
					{
						// Use the tensor to calculate the product S_1 * A * S_2
						auto A = ATensor->LabFrame();

						tmp += S1x * S2x * A(0, 0) + S1x * S2y * A(0, 1) + S1x * S2z * A(0, 2);
						tmp += S1y * S2x * A(1, 0) + S1y * S2y * A(1, 1) + S1y * S2z * A(1, 2);
						tmp += S1z * S2x * A(2, 0) + S1z * S2y * A(2, 1) + S1z * S2z * A(2, 2);
					}
					else
					{
						// If there is no interaction tensor or if the tensor is isotropic, just take the dot product
						tmp += S1x * S2x + S1y * S2y + S1z * S2z;
					}
				}
			}
		}
		else if (_interaction->Type() == InteractionType::Exchange)
		{
			// Obtain lists of interacting spins, coupling tensor, and define matrices to hold the magnetic moment operators
			auto spins1 = _interaction->Group1();
			auto spins2 = _interaction->Group2();
			arma::cx_mat S1x;
			arma::cx_mat S1y;
			arma::cx_mat S1z;
			arma::cx_mat S2x;
			arma::cx_mat S2y;
			arma::cx_mat S2z;

			// Fill the matrix with the sum of all the interactions
			for (auto i = spins1.cbegin(); i != spins1.cend(); i++)
			{
				for (auto j = spins2.cbegin(); j != spins2.cend(); j++)
				{
					// Obtain the magnetic moment operators within the Hilbert space
					if (_interaction->IgnoreTensors())
					{
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Sx()), (*i), S1x);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Sy()), (*i), S1y);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Sz()), (*i), S1z);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*j)->Sx()), (*j), S2x);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*j)->Sy()), (*j), S2y);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*j)->Sz()), (*j), S2z);
					}
					else
					{
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Tx()), (*i), S1x);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Ty()), (*i), S1y);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Tz()), (*i), S1z);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*j)->Tx()), (*j), S2x);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*j)->Ty()), (*j), S2y);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*j)->Tz()), (*j), S2z);
					}
					// TODO: This is not correct when the user uses a non isotropic exchange which he will not when being smart.
					if (ATensor != nullptr && !IsIsotropic(*ATensor))
					{
						// Use the tensor to calculate the product S_1 * A * S_2
						auto A = ATensor->LabFrame();
						tmp += 2.0 * (S1x * S2x * A(0, 0) + S1x * S2y * A(0, 1) + S1x * S2z * A(0, 2));
						tmp += 2.0 * (S1y * S2x * A(1, 0) + S1y * S2y * A(1, 1) + S1y * S2z * A(1, 2));
						tmp += 2.0 * (S1z * S2x * A(2, 0) + S1z * S2y * A(2, 1) + S1z * S2z * A(2, 2));

						arma::cx_mat half = 0.5 * arma::eye<arma::cx_mat>(this->HilbertSpaceDimensions(), this->HilbertSpaceDimensions());

						tmp += half;
					}
					else
					{
						// TODO: Be careful with the sign here!!! Usually you would use a minus sign to at the whole interaction.
						//  If there is no interaction tensor or if the tensor is isotropic, just take the dot product
						tmp += 2.0 * (S1x * S2x + S1y * S2y + S1z * S2z);
						arma::cx_mat half = 0.5 * arma::eye<arma::cx_mat>(this->HilbertSpaceDimensions(), this->HilbertSpaceDimensions());

						tmp += half;
					}
				}
			}
		}
		else if (_interaction->Type() == InteractionType::Zfs)
		{
			// Obtain lists of interacting spins, coupling tensor, and define matrices to hold the magnetic moment operators
			auto spins1 = _interaction->Group1();
			arma::cx_mat Sx;
			arma::cx_mat Sy;
			arma::cx_mat Sz;

			// Fill the matrix with the sum of all the interactions
			for (auto i = spins1.cbegin(); i != spins1.cend(); i++)
			{
				// Obtain the magnetic moment operators within the Hilbert space
				if (_interaction->IgnoreTensors())
				{
					this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Sx()), (*i), Sx);
					this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Sy()), (*i), Sy);
					this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Sz()), (*i), Sz);
				}
				else
				{
					this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Tx()), (*i), Sx);
					this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Ty()), (*i), Sy);
					this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Tz()), (*i), Sz);
				}

				// Get D and E value
				double D = _interaction->Dvalue();
				double E = _interaction->Evalue();

				if (D == 0.0 && E == 0.0)
				{
					std::cout << "D or E value for zero-field splitting was not found." << std::endl;
				}
				else
				{
					// Calculate Zfs interaction
					tmp = D * (Sz * Sz - ((1.00 / 3.00) * (*i)->S() * ((*i)->S() + 1))) + E * (Sx * Sx - Sy * Sy);
				}
			}
		}
		else if (_interaction->Type() == InteractionType::SemiClassicalField)
		{
			// Obtain lists of interacting spins, coupling tensor, and define matrices to hold the magnetic moment operators
			auto spins1 = _interaction->Group1();

			//  Grab amplitude and orientation parameters
			const double   B0   = _interaction->Hfiamplitude();     // Tesla
			const int   n    = _interaction->Orientations(); // averaging grid

			// Build Sx, Sy, Sz for *each* electron in Group1
			arma::cx_mat Sx, Sy, Sz;
			for (auto i = spins1.cbegin(); i != spins1.cend(); i++)
			{
				// Obtain the magnetic moment operators within the Hilbert space
				if (_interaction->IgnoreTensors())
				{
					this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Sx()), (*i), Sx);
					this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Sy()), (*i), Sy);
					this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Sz()), (*i), Sz);
				}
				else
				{
					this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Tx()), (*i), Sx);
					this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Ty()), (*i), Sy);
					this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Tz()), (*i), Sz);
				}

				// Semi-classical average  (weight = ½ sinθ Δθ)
				for (int k = 0; k < n; ++k)
				{
					double theta      = M_PI * (k + 0.5) / n;
					double weight = 0.5 * std::sin(theta) * (M_PI / n);

					// Local field components in the *molecule* frame
					double Bx = B0 * std::sin(theta);
					double Bz = B0 * std::cos(theta);

					tmp += weight * (Bx * Sx + Bz * Sz);   // Sy-component = 0 by symmetry
				}
			}
		}
		else
		{
			// The interaction type was not recognized
			return false;
		}

		// Multiply by the isotropic value, if the tensor was isotropic
		if (ATensor != nullptr && IsIsotropic(*ATensor))
			tmp *= ATensor->Isotropic();

		// Multiply with the given prefactor (or 1.0 if none was specified)
		tmp *= _interaction->Prefactor();

		// Multiply by common prefactor (bohr magneton / hbar)
		if (_interaction->AddCommonPrefactor())
			tmp *= 8.79410005e+1;

		// Check whether we want a superspace or Hilbert space result
		if (this->useSuperspace)
		{
			arma::cx_mat lhs;
			arma::cx_mat rhs;
			auto result = this->SuperoperatorFromLeftOperator(tmp, lhs);
			result &= this->SuperoperatorFromRightOperator(tmp, rhs);
			if (result)
				_out = lhs - rhs;
			else
				return false;
		}
		else
		{
			// We already have the result in the Hilbert space
			_out = tmp;
		}

		return true;
	}

	// Returns the matrix representation of the Interaction object on the spins space (sparse matrix version)
	bool SpinSpace::InteractionOperator(const interaction_ptr &_interaction, arma::sp_cx_mat &_out) const
	{
		// Make sure the interaction is valid
		if (_interaction == nullptr)
			return false;

		// Create temporary matrix to hold the result
		arma::sp_cx_mat tmp = arma::zeros<arma::sp_cx_mat>(this->HilbertSpaceDimensions(), this->HilbertSpaceDimensions());

		// Get the interaction tensor
		auto ATensor = _interaction->CouplingTensor();

		if (_interaction->Type() == InteractionType::SingleSpin)
		{
			// Get the field at the current time or trajectory step
			arma::vec field;
			field = _interaction->Field();

			// Obtain the list of spins interacting with the field, and define matrices to hold the magnetic moment operators
			auto spinlist = _interaction->Group1();
			arma::sp_cx_mat Sx;
			arma::sp_cx_mat Sy;
			arma::sp_cx_mat Sz;

			// Fill the matrix with the sum of all the interactions (i.e. between spin magnetic moment and fields)
			for (auto i = spinlist.cbegin(); i != spinlist.cend(); i++)
			{
				if (_interaction->IgnoreTensors())
				{
					this->CreateOperator((*i)->Sx(), (*i), Sx);
					this->CreateOperator((*i)->Sy(), (*i), Sy);
					this->CreateOperator((*i)->Sz(), (*i), Sz);
				}
				else
				{
					this->CreateOperator((*i)->Tx(), (*i), Sx);
					this->CreateOperator((*i)->Ty(), (*i), Sy);
					this->CreateOperator((*i)->Tz(), (*i), Sz);
				}

				if (ATensor != nullptr && !IsIsotropic(*ATensor))
				{
					// Use the tensor to calculate the product S * A * B
					auto A = ATensor->LabFrame();
					tmp += Sx * field(0) * A(0, 0) + Sx * field(1) * A(0, 1) + Sx * field(2) * A(0, 2);
					tmp += Sy * field(0) * A(1, 0) + Sy * field(1) * A(1, 1) + Sy * field(2) * A(1, 2);
					tmp += Sz * field(0) * A(2, 0) + Sz * field(1) * A(2, 1) + Sz * field(2) * A(2, 2);
				}
				else
				{
					tmp += Sx * field(0) + Sy * field(1) + Sz * field(2);
				}
			}
		}
		else if (_interaction->Type() == InteractionType::DoubleSpin)
		{
			// Obtain lists of interacting spins, coupling tensor, and define matrices to hold the magnetic moment operators
			auto spins1 = _interaction->Group1();
			auto spins2 = _interaction->Group2();
			arma::sp_cx_mat S1x;
			arma::sp_cx_mat S1y;
			arma::sp_cx_mat S1z;
			arma::sp_cx_mat S2x;
			arma::sp_cx_mat S2y;
			arma::sp_cx_mat S2z;

			// Fill the matrix with the sum of all the interactions
			for (auto i = spins1.cbegin(); i != spins1.cend(); i++)
			{
				for (auto j = spins2.cbegin(); j != spins2.cend(); j++)
				{
					// Obtain the magnetic moment operators within the Hilbert space
					if (_interaction->IgnoreTensors())
					{

						this->CreateOperator((*i)->Sx(), (*i), S1x);
						this->CreateOperator((*i)->Sy(), (*i), S1y);
						this->CreateOperator((*i)->Sz(), (*i), S1z);
						this->CreateOperator((*j)->Sx(), (*j), S2x);
						this->CreateOperator((*j)->Sy(), (*j), S2y);
						this->CreateOperator((*j)->Sz(), (*j), S2z);
					}
					else
					{

						this->CreateOperator((*i)->Tx(), (*i), S1x);
						this->CreateOperator((*i)->Ty(), (*i), S1y);
						this->CreateOperator((*i)->Tz(), (*i), S1z);
						this->CreateOperator((*j)->Tx(), (*j), S2x);
						this->CreateOperator((*j)->Ty(), (*j), S2y);
						this->CreateOperator((*j)->Tz(), (*j), S2z);
					}

					if (ATensor != nullptr && !IsIsotropic(*ATensor))
					{
						// Use the tensor to calculate the product S_1 * A * S_2
						auto A = ATensor->LabFrame();
						tmp += S1x * S2x * A(0, 0) + S1x * S2y * A(0, 1) + S1x * S2z * A(0, 2);
						tmp += S1y * S2x * A(1, 0) + S1y * S2y * A(1, 1) + S1y * S2z * A(1, 2);
						tmp += S1z * S2x * A(2, 0) + S1z * S2y * A(2, 1) + S1z * S2z * A(2, 2);
					}
					else
					{
						// If there is no interaction tensor or if the tensor is isotropic, just take the dot product
						tmp += S1x * S2x + S1y * S2y + S1z * S2z;
					}
				}
			}
		}
		else if (_interaction->Type() == InteractionType::Exchange)
		{
			// Obtain lists of interacting spins, coupling tensor, and define matrices to hold the magnetic moment operators
			auto spins1 = _interaction->Group1();
			auto spins2 = _interaction->Group2();
			arma::sp_cx_mat S1x;
			arma::sp_cx_mat S1y;
			arma::sp_cx_mat S1z;
			arma::sp_cx_mat S2x;
			arma::sp_cx_mat S2y;
			arma::sp_cx_mat S2z;

			// Fill the matrix with the sum of all the interactions
			for (auto i = spins1.cbegin(); i != spins1.cend(); i++)
			{
				for (auto j = spins2.cbegin(); j != spins2.cend(); j++)
				{
					// Obtain the magnetic moment operators within the Hilbert space
					if (_interaction->IgnoreTensors())
					{

						this->CreateOperator((*i)->Sx(), (*i), S1x);
						this->CreateOperator((*i)->Sy(), (*i), S1y);
						this->CreateOperator((*i)->Sz(), (*i), S1z);
						this->CreateOperator((*j)->Sx(), (*j), S2x);
						this->CreateOperator((*j)->Sy(), (*j), S2y);
						this->CreateOperator((*j)->Sz(), (*j), S2z);
					}
					else
					{

						this->CreateOperator((*i)->Tx(), (*i), S1x);
						this->CreateOperator((*i)->Ty(), (*i), S1y);
						this->CreateOperator((*i)->Tz(), (*i), S1z);
						this->CreateOperator((*j)->Tx(), (*j), S2x);
						this->CreateOperator((*j)->Ty(), (*j), S2y);
						this->CreateOperator((*j)->Tz(), (*j), S2z);
					}

					// TODO: This is not correct when the user uses a non isotropic exchange which he will not when being smart.
					if (ATensor != nullptr && !IsIsotropic(*ATensor))
					{
						// Use the tensor to calculate the product S_1 * A * S_2
						auto A = ATensor->LabFrame();
						tmp += 2.0 * (S1x * S2x * A(0, 0) + S1x * S2y * A(0, 1) + S1x * S2z * A(0, 2));
						tmp += 2.0 * (S1y * S2x * A(1, 0) + S1y * S2y * A(1, 1) + S1y * S2z * A(1, 2));
						tmp += 2.0 * (S1z * S2x * A(2, 0) + S1z * S2y * A(2, 1) + S1z * S2z * A(2, 2));

						arma::sp_cx_mat half = 0.5 * arma::eye<arma::sp_cx_mat>(this->HilbertSpaceDimensions(), this->HilbertSpaceDimensions());

						tmp(arma::span::all, arma::span::all) += half;
					}
					else
					{
						// TODO: Be carfeul with the sign here!!! Usually you would use a minus sign to at the whol interaction.
						//  If there is no interaction tensor or if the tensor is isotropic, just take the dot product
						tmp += 2.0 * (S1x * S2x + S1y * S2y + S1z * S2z);

						arma::sp_cx_mat half = 0.5 * arma::eye<arma::sp_cx_mat>(this->HilbertSpaceDimensions(), this->HilbertSpaceDimensions());

						tmp(arma::span::all, arma::span::all) += half;
					}
				}
			}
		}
		else if (_interaction->Type() == InteractionType::Zfs)
		{
			// Obtain lists of interacting spins, coupling tensor, and define matrices to hold the magnetic moment operators
			auto spinlist = _interaction->Group1();
			// Build Sx, Sy, Sz for *each* electron in Group1
			arma::sp_cx_mat Sx;
			arma::sp_cx_mat Sy;
			arma::sp_cx_mat Sz;

			// Fill the matrix with the sum of all the interactions (i.e. between spin magnetic moment and fields)
			for (auto i = spinlist.cbegin(); i != spinlist.cend(); i++)
			{
				if (_interaction->IgnoreTensors())
				{
					this->CreateOperator((*i)->Sx(), (*i), Sx);
					this->CreateOperator((*i)->Sy(), (*i), Sy);
					this->CreateOperator((*i)->Sz(), (*i), Sz);
				}
				else
				{
					this->CreateOperator((*i)->Tx(), (*i), Sx);
					this->CreateOperator((*i)->Ty(), (*i), Sy);
					this->CreateOperator((*i)->Tz(), (*i), Sz);
				}

				// Get D and E value
				double D = _interaction->Dvalue();
				double E = _interaction->Evalue();

				if (D == 0.0 && E == 0.0)
				{
					std::cout << "D or E value for zero-field splitting was not found." << std::endl;
				}
				else
				{
					// Calculate Zfs interaction
					tmp = D * (Sz * Sz - ((1.00 / 3.00) * (*i)->S() * ((*i)->S() + 1))) + E * (Sx * Sx - Sy * Sy);
				}
			}
		}
		else if (_interaction->Type() == InteractionType::SemiClassicalField)
		{
			// Obtain lists of interacting spins, coupling tensor, and define matrices to hold the magnetic moment operators
			auto spinlist = _interaction->Group1();

			//  Grab amplitude and orientation parameters
			const double   B0   = _interaction->Hfiamplitude();     // Tesla
			const int   n    = _interaction->Orientations(); // averaging grid

			// Build Sx, Sy, Sz for *each* electron in Group1
			arma::sp_cx_mat Sx;
			arma::sp_cx_mat Sy;
			arma::sp_cx_mat Sz;

			// Fill the matrix with the sum of all the interactions (i.e. between spin magnetic moment and fields)
			for (auto i = spinlist.cbegin(); i != spinlist.cend(); i++)
			{
				if (_interaction->IgnoreTensors())
				{
					this->CreateOperator((*i)->Sx(), (*i), Sx);
					this->CreateOperator((*i)->Sy(), (*i), Sy);
					this->CreateOperator((*i)->Sz(), (*i), Sz);
				}
				else
				{
					this->CreateOperator((*i)->Tx(), (*i), Sx);
					this->CreateOperator((*i)->Ty(), (*i), Sy);
					this->CreateOperator((*i)->Tz(), (*i), Sz);
				}

				// Semi-classical average  (weight = ½ sinθ Δθ)
				for (int k = 0; k < n; ++k)
				{
					double theta      = M_PI * (k + 0.5) / n;
					double weight = 0.5 * std::sin(theta) * (M_PI / n);

					// Local field components in the *molecule* frame
					double Bx = B0 * std::sin(theta);
					double Bz = B0 * std::cos(theta);

					tmp += weight * (Bx * Sx + Bz * Sz);  // Sy-component = 0 by symmetry
				}
			}
		}
		else
		{
			// The interaction type was not recognized
			return false;
		}

		// Multiply by the isotropic value, if the tensor was isotropic
		if (ATensor != nullptr && IsIsotropic(*ATensor))
			tmp *= ATensor->Isotropic();

		// Multiply with the given prefactor (or 1.0 if none was specified)
		tmp *= _interaction->Prefactor();

		// Multiply by common prefactor. TODO: Consider which system of units to use
		if (_interaction->AddCommonPrefactor())
			tmp *= 8.79410005e+1;

		// Check whether we want a superspace or Hilbert space result
		if (this->useSuperspace)
		{
			arma::sp_cx_mat lhs;
			arma::sp_cx_mat rhs;
			auto result = this->SuperoperatorFromLeftOperator(tmp, lhs);
			result &= this->SuperoperatorFromRightOperator(tmp, rhs);
			if (result)
				_out = lhs - rhs;
			else
				return false;
		}
		else
		{
			// We already have the result in the dense matrix to the Hamiltonian at the given time or trajectorye Hilbert space
			_out = tmp;
		}

		return true;
	}

	// Returns the matrix representation of the Interaction object on the spins space (sparse matrix version). Singlespin and double spin interaction are rotated with interactionframe and additionally to the point on the sphere. ZFS and Semiclassical field are currently not modified in any way
	bool SpinSpace::InteractionOperatorRotated(const interaction_ptr &_interaction, arma::mat &_rotationmatrix, arma::sp_cx_mat &_out) const
		{
			// Make sure the interaction is valid
			if (_interaction == nullptr)
				return false;

			// Create temporary matrix to hold the result
			arma::sp_cx_mat tmp = arma::zeros<arma::sp_cx_mat>(this->HilbertSpaceDimensions(), this->HilbertSpaceDimensions());

			// Get the interaction tensor
			auto ATensor = _interaction->CouplingTensor();

			// IMPORTANT:
			// The powder-rotation matrix produced in TaskStaticHSDirectSpectra / TaskStaticHSTrEPRSpectra
			// is an *active* rotation (see CreateRotationMatrix: cos,-sin; sin,cos). For tensor rotation
			// in the EasySpin/pepper sense we need the corresponding *passive* matrix.
			// Therefore we transpose here.
			const arma::mat Rpowder = _rotationmatrix.t();

			// Interaction-frame rotation from the interaction framelist.
			// We interpret framelist Euler angles with EasySpin's erot ZXZ passive convention.
			arma::mat RFrame = arma::eye<arma::mat>(3, 3);
			{
				auto fr = _interaction->Framelist();
				double a = (fr.n_elem >= 1) ? fr(0) : 0.0;
				double b = (fr.n_elem >= 2) ? fr(1) : 0.0;
				double g = (fr.n_elem >= 3) ? fr(2) : 0.0;

				// EasySpin erot.m matrix (passive). See EasySpin docs/source.
				const double ca = std::cos(a), sa = std::sin(a);
				const double cb = std::cos(b), sb = std::sin(b);
				const double cg = std::cos(g), sg = std::sin(g);

				arma::mat Ra = {{ca, sa, 0.0}, {-sa, ca, 0.0}, {0.0, 0.0, 1.0}};
				arma::mat Rb = {{cb, 0.0, -sb}, {0.0, 1.0, 0.0}, {sb, 0.0, cb}};
				arma::mat Rg = {{cg, sg, 0.0}, {-sg, cg, 0.0}, {0.0, 0.0, 1.0}};
				RFrame = Rg * Rb * Ra;
			}

			auto RotateTensorFrameAndPowder = [&](const arma::mat &A) -> arma::mat {
				arma::mat Af = RFrame * A * RFrame.t();
				arma::mat Al = Rpowder * Af * Rpowder.t();
				if (!this->useFullTensorRotation)
				{
					// keep only diagonal elements (legacy behavior)
					Al = Al % arma::eye<arma::mat>(3, 3);
				}
				return Al;
			};

			if (_interaction->Type() == InteractionType::SingleSpin)
			{
				// Get the field (lab frame)
				arma::vec field = _interaction->Field();

				// Obtain the list of spins interacting with the field
				auto spinlist = _interaction->Group1();
				arma::sp_cx_mat Sx, Sy, Sz;

				for (auto i = spinlist.cbegin(); i != spinlist.cend(); i++)
				{
					this->CreateOperator((*i)->Sx(), (*i), Sx);
					this->CreateOperator((*i)->Sy(), (*i), Sy);
					this->CreateOperator((*i)->Sz(), (*i), Sz);

					// Spin tensor (typically g-tensor for electron Zeeman)
					arma::mat G = arma::conv_to<arma::mat>::from((*i)->GetTensor().LabFrame());
					if (_interaction->IgnoreTensors())
					{
						// match InteractionOperator(): ignore spin tensors completely
						G = arma::eye<arma::mat>(3, 3);
					}

					// Apply interaction frame and powder orientation
					G = RotateTensorFrameAndPowder(G);

					// H = S · (G · B)
					tmp += Sx * (field(0) * G(0, 0) + field(1) * G(0, 1) + field(2) * G(0, 2));
					tmp += Sy * (field(0) * G(1, 0) + field(1) * G(1, 1) + field(2) * G(1, 2));
					tmp += Sz * (field(0) * G(2, 0) + field(1) * G(2, 1) + field(2) * G(2, 2));
				}
			}
			else if (_interaction->Type() == InteractionType::DoubleSpin)
			{
				auto spins1 = _interaction->Group1();
				auto spins2 = _interaction->Group2();

				arma::sp_cx_mat S1x, S1y, S1z;
				arma::sp_cx_mat S2x, S2y, S2z;

				for (auto i = spins1.cbegin(); i != spins1.cend(); i++)
				{
					for (auto j = spins2.cbegin(); j != spins2.cend(); j++)
					{
						// Match InteractionOperator(): use S operators or T operators depending on IgnoreTensors
						if (_interaction->IgnoreTensors())
						{
							this->CreateOperator((*i)->Sx(), (*i), S1x);
							this->CreateOperator((*i)->Sy(), (*i), S1y);
							this->CreateOperator((*i)->Sz(), (*i), S1z);
							this->CreateOperator((*j)->Sx(), (*j), S2x);
							this->CreateOperator((*j)->Sy(), (*j), S2y);
							this->CreateOperator((*j)->Sz(), (*j), S2z);
						}
						else
						{
							this->CreateOperator((*i)->Tx(), (*i), S1x);
							this->CreateOperator((*i)->Ty(), (*i), S1y);
							this->CreateOperator((*i)->Tz(), (*i), S1z);
							this->CreateOperator((*j)->Tx(), (*j), S2x);
							this->CreateOperator((*j)->Ty(), (*j), S2y);
							this->CreateOperator((*j)->Tz(), (*j), S2z);
						}

						if (ATensor != nullptr && !IsIsotropic(*ATensor))
						{
							arma::mat A = arma::conv_to<arma::mat>::from(ATensor->LabFrame());
							A = RotateTensorFrameAndPowder(A);

							// S1 · A · S2
							tmp += S1x * S2x * A(0, 0) + S1x * S2y * A(0, 1) + S1x * S2z * A(0, 2);
							tmp += S1y * S2x * A(1, 0) + S1y * S2y * A(1, 1) + S1y * S2z * A(1, 2);
							tmp += S1z * S2x * A(2, 0) + S1z * S2y * A(2, 1) + S1z * S2z * A(2, 2);
						}
						else
						{
							// isotropic dot product
							tmp += S1x * S2x + S1y * S2y + S1z * S2z;
						}
					}
				}
			}
			else if (_interaction->Type() == InteractionType::Exchange)
			{
				auto spins1 = _interaction->Group1();
				auto spins2 = _interaction->Group2();
				arma::sp_cx_mat S1x, S1y, S1z;
				arma::sp_cx_mat S2x, S2y, S2z;

				for (auto i = spins1.cbegin(); i != spins1.cend(); i++)
				{
					for (auto j = spins2.cbegin(); j != spins2.cend(); j++)
					{
						if (_interaction->IgnoreTensors())
						{
							this->CreateOperator((*i)->Sx(), (*i), S1x);
							this->CreateOperator((*i)->Sy(), (*i), S1y);
							this->CreateOperator((*i)->Sz(), (*i), S1z);
							this->CreateOperator((*j)->Sx(), (*j), S2x);
							this->CreateOperator((*j)->Sy(), (*j), S2y);
							this->CreateOperator((*j)->Sz(), (*j), S2z);
						}
						else
						{
							this->CreateOperator((*i)->Tx(), (*i), S1x);
							this->CreateOperator((*i)->Ty(), (*i), S1y);
							this->CreateOperator((*i)->Tz(), (*i), S1z);
							this->CreateOperator((*j)->Tx(), (*j), S2x);
							this->CreateOperator((*j)->Ty(), (*j), S2y);
							this->CreateOperator((*j)->Tz(), (*j), S2z);
						}

						if (ATensor != nullptr && !IsIsotropic(*ATensor))
						{
							arma::mat A = arma::conv_to<arma::mat>::from(ATensor->LabFrame());
							A = RotateTensorFrameAndPowder(A);
							// factor 2 and +1/2 as in original implementation
							tmp += 2.0 * (S1x * S2x * A(0, 0) + S1x * S2y * A(0, 1) + S1x * S2z * A(0, 2));
							tmp += 2.0 * (S1y * S2x * A(1, 0) + S1y * S2y * A(1, 1) + S1y * S2z * A(1, 2));
							tmp += 2.0 * (S1z * S2x * A(2, 0) + S1z * S2y * A(2, 1) + S1z * S2z * A(2, 2));
							arma::sp_cx_mat half = 0.5 * arma::eye<arma::sp_cx_mat>(this->HilbertSpaceDimensions(), this->HilbertSpaceDimensions());
							tmp(arma::span::all, arma::span::all) += half;
						}
						else
						{
							tmp += 2.0 * (S1x * S2x + S1y * S2y + S1z * S2z);
							arma::sp_cx_mat half = 0.5 * arma::eye<arma::sp_cx_mat>(this->HilbertSpaceDimensions(), this->HilbertSpaceDimensions());
							tmp(arma::span::all, arma::span::all) += half;
						}
					}
				}
			}
			else if (_interaction->Type() == InteractionType::Zfs)
			{
				// ZFS not modified here (legacy)
				auto spins1 = _interaction->Group1();
				arma::sp_cx_mat Sx, Sy, Sz;
				for (auto i = spins1.cbegin(); i != spins1.cend(); i++)
				{
					this->CreateOperator((*i)->Sx(), (*i), Sx);
					this->CreateOperator((*i)->Sy(), (*i), Sy);
					this->CreateOperator((*i)->Sz(), (*i), Sz);

					double D = _interaction->Dvalue();
					double E = _interaction->Evalue();
					if (D == 0.0 && E == 0.0)
					{
						std::cout << "D or E value for zero-field splitting was not found." << std::endl;
					}
					else
					{
						tmp = D * (Sz * Sz - ((1.00 / 3.00) * (*i)->S() * ((*i)->S() + 1))) + E * (Sx * Sx - Sy * Sy);
					}
				}
			}
			else if (_interaction->Type() == InteractionType::SemiClassicalField)
			{
				// unchanged
				auto spins1 = _interaction->Group1();
				const double B0 = _interaction->Hfiamplitude();
				const int n = _interaction->Orientations();

				arma::sp_cx_mat Sx, Sy, Sz;
				for (auto i = spins1.cbegin(); i != spins1.cend(); i++)
				{
					this->CreateOperator((*i)->Sx(), (*i), Sx);
					this->CreateOperator((*i)->Sy(), (*i), Sy);
					this->CreateOperator((*i)->Sz(), (*i), Sz);

					for (int k = 0; k < n; ++k)
					{
						double theta = M_PI * (k + 0.5) / n;
						double weight = 0.5 * std::sin(theta) * (M_PI / n);
						double Bx = B0 * std::sin(theta);
						double Bz = B0 * std::cos(theta);
						tmp += weight * (Bx * Sx + Bz * Sz);
					}
				}
			}
			else
			{
				// The interaction type was not recognized
				return false;
			}

			// Multiply by the isotropic value, if the tensor was isotropic
			if (ATensor != nullptr && IsIsotropic(*ATensor))
				tmp *= ATensor->Isotropic();

			// Multiply with the given prefactor (or 1.0 if none was specified)
			tmp *= _interaction->Prefactor();

			// Multiply by common prefactor. TODO: Consider which system of units to use
			if (_interaction->AddCommonPrefactor())
				tmp *= 8.79410005e+1;

			// Check whether we want a superspace or Hilbert space result
			if (this->useSuperspace)
			{
				arma::cx_mat tmpDense = arma::cx_mat(tmp);
				arma::cx_mat lhs;
				arma::cx_mat rhs;
				auto result = this->SuperoperatorFromLeftOperator(tmpDense, lhs);
				result &= this->SuperoperatorFromRightOperator(tmpDense, rhs);
				if (result)
					_out = arma::sp_cx_mat(lhs - rhs);
				else
					return false;
			}
			else
			{
				_out = tmp;
			}

			return true;
		}


	// Sets the dense matrix to the Hamiltonian at the given time or trajectory step
	bool SpinSpace::Hamiltonian(arma::cx_mat &_out) const
	{
		// If we don't have any interactions, the Hamiltonian is zero
		if (this->interactions.size() < 1)
		{
			_out = arma::zeros<arma::cx_mat>(this->SpaceDimensions(), this->SpaceDimensions());
			return true;
		}

		// Get the first interaction contribution
		auto i = this->interactions.cbegin();
		arma::cx_mat tmp;
		arma::cx_mat result;
		if (!this->InteractionOperator((*i), result))
			return false;

		// We have already used the first interaction
		i++;

		// Loop through the rest
		for (; i != this->interactions.cend(); i++)
		{
			// Attempt to get the matrix representing the Interaction object in the spin space
			if (!this->InteractionOperator((*i), tmp))
				return false;
			result += tmp;
		}

		_out = result;
		return true;
	}

	// Sets the sparse matrix to the Hamiltonian at the given time or trajectory step
	bool SpinSpace::Hamiltonian(arma::sp_cx_mat &_out) const
	{
		// If we don't have any interactions, the Hamiltonian is zero
		if (this->interactions.size() < 1)
		{
			_out = arma::sp_cx_mat(this->SpaceDimensions(), this->SpaceDimensions());
			return true;
		}

		// Get the first interaction contribution
		auto i = this->interactions.cbegin();
		arma::sp_cx_mat tmp;
		arma::sp_cx_mat result;
		if (!this->InteractionOperator((*i), result))
			return false;

		// We have already used the first interaction
		i++;

		// Loop through the rest
		for (; i != this->interactions.cend(); i++)
		{
			// Attempt to get the matrix representing the Interaction object in the spin space
			if (!this->InteractionOperator((*i), tmp))
				return false;
			result += tmp;
		}

		_out = result;
		return true;
	}

	// Sets the dense matrix to the part of the Hamiltonian that is independent of time or trajectory step
	bool SpinSpace::StaticHamiltonian(arma::cx_mat &_out) const
	{
		// If we don't have any interactions, the Hamiltonian is zero
		arma::cx_mat result = arma::zeros<arma::cx_mat>(this->SpaceDimensions(), this->SpaceDimensions());
		if (this->interactions.size() < 1)
		{
			_out = result;
			return true;
		}

		arma::cx_mat tmp;
		for (auto i = this->interactions.cbegin(); i != this->interactions.cend(); i++)
		{
			// Skip any dynamic interactions (time-dependent or with a trajectory)
			if (!IsStatic(*(*i)))
				continue;

			// Attempt to get the matrix representing the Interaction object in the spin space
			if (!this->InteractionOperator((*i), tmp))
				return false;

			result += tmp;
		}

		_out = result;
		return true;
	}

	// Sets the sparse matrix to the part of the Hamiltonian that is independent of time or trajectory step
	bool SpinSpace::StaticHamiltonian(arma::sp_cx_mat &_out) const
	{
		// If we don't have any interactions, the Hamiltonian is zero
		arma::sp_cx_mat result = arma::sp_cx_mat(this->SpaceDimensions(), this->SpaceDimensions());
		if (this->interactions.size() < 1)
		{
			_out = result;
			return true;
		}

		arma::sp_cx_mat tmp;
		for (auto i = this->interactions.cbegin(); i != this->interactions.cend(); i++)
		{
			// Skip any dynamic interactions (time-dependent or with a trajectory)
			if (!IsStatic(*(*i)))
				continue;

			// Attempt to get the matrix representing the Interaction object in the spin space
			if (!this->InteractionOperator((*i), tmp))
				return false;

			result += tmp;
		}

		_out = result;
		return true;
	}

	// Sets the dense matrix to the part of the Hamiltonian that is dependent on time and/or trajectory step
	bool SpinSpace::DynamicHamiltonian(arma::cx_mat &_out) const
	{
		// If we don't have any interactions, the Hamiltonian is zero
		arma::cx_mat result = arma::zeros<arma::cx_mat>(this->SpaceDimensions(), this->SpaceDimensions());
		if (this->interactions.size() < 1)
		{
			_out = result;
			return true;
		}

		arma::cx_mat tmp;
		for (auto i = this->interactions.cbegin(); i != this->interactions.cend(); i++)
		{
			// Skip static interactions
			if (IsStatic(*(*i)))
				continue;

			// Attempt to get the matrix representing the Interaction object in the spin space
			if (!this->InteractionOperator((*i), tmp))
				return false;

			result += tmp;
		}

		_out = result;
		return true;
	}

	// Sets the sparse matrix to the part of the Hamiltonian that is dependent on time and/or trajectory step
	bool SpinSpace::DynamicHamiltonian(arma::sp_cx_mat &_out) const
	{
		// If we don't have any interactions, the Hamiltonian is zero
		arma::sp_cx_mat result = arma::sp_cx_mat(this->SpaceDimensions(), this->SpaceDimensions());
		if (this->interactions.size() < 1)
		{
			_out = result;
			return true;
		}

		arma::sp_cx_mat tmp;
		for (auto i = this->interactions.cbegin(); i != this->interactions.cend(); i++)
		{
			// Skip static interactions
			if (IsStatic(*(*i)))
				continue;

			// Attempt to get the matrix representing the Interaction object in the spin space
			if (!this->InteractionOperator((*i), tmp))
				return false;

			result += tmp;
		}

		_out = result;
		return true;
	}

	// Sets the dense matrix to the part of the Hamiltonian that is used in the thermal equilibrium (without Zeeman terms)
	bool SpinSpace::ThermalHamiltonian(std::vector<std::string> thermalhamiltonian_list, arma::cx_mat &_out) const
	{
		// If we don't have any interactions, the Hamiltonian is zero
		arma::cx_mat result = arma::zeros<arma::cx_mat>(this->SpaceDimensions(), this->SpaceDimensions());
		if (this->interactions.size() < 1)
		{
			_out = result;
			return true;
		}

		arma::cx_mat tmp;
		for (auto i = this->interactions.cbegin(); i != this->interactions.cend(); i++)
		{
			// Skip any dynamic interactions (time-dependent or with a trajectory)
			if (!IsStatic(*(*i)))
				continue;

			// Check if the interaction name is in the thermalhamiltonian_list
			if (std::find(thermalhamiltonian_list.begin(), thermalhamiltonian_list.end(), (*i)->Name()) != thermalhamiltonian_list.end())
			{
				// Attempt to get the matrix representing the Interaction object in the spin space
				if (!this->InteractionOperator((*i), tmp))
					return false;

				result += tmp;
			}
		}

		_out = result;
		return true;
	}

	// Sets the sparce matrix to the part of the Hamiltonian that is used in the thermal equilibrium (without Zeeman terms)
	bool SpinSpace::ThermalHamiltonian(std::vector<std::string> thermalhamiltonian_list, arma::sp_cx_mat &_out) const
	{
		// If we don't have any interactions, the Hamiltonian is zero
		arma::cx_mat result = arma::zeros<arma::cx_mat>(this->SpaceDimensions(), this->SpaceDimensions());
		if (this->interactions.size() < 1)
		{
			_out = result;
			return true;
		}

		arma::cx_mat tmp;
		for (auto i = this->interactions.cbegin(); i != this->interactions.cend(); i++)
		{
			// Skip any dynamic interactions (time-dependent or with a trajectory)
			if (!IsStatic(*(*i)))
				continue;

			// Check if the interaction name is in the thermalhamiltonian_list
			if (std::find(thermalhamiltonian_list.begin(), thermalhamiltonian_list.end(), (*i)->Name()) != thermalhamiltonian_list.end())
			{
				// Attempt to get the matrix representing the Interaction object in the spin space
				if (!this->InteractionOperator((*i), tmp))
					return false;

				result += tmp;
			}
		}

		_out = result;
		return true;
	}

	// Sets the rotated sparce matrix to the part of the Hamiltonian that is used in the base Hamiltonina H0 (without additional mw field)
	bool SpinSpace::BaseHamiltonianRotated(std::vector<std::string> basehamiltonian_list, arma::mat rotmatrix, arma::sp_cx_mat &_out) const
		{
			// If we don't have any interactions, the Hamiltonian is zero
			arma::sp_cx_mat result = arma::sp_cx_mat(this->SpaceDimensions(), this->SpaceDimensions());
			if (this->interactions.size() < 1)
			{
				_out = result;
				return true;
			}

			arma::sp_cx_mat tmp;
			for (auto i = this->interactions.cbegin(); i != this->interactions.cend(); i++)
			{
				// Skip any dynamic interactions (time-dependent or with a trajectory)
				if (!IsStatic(*(*i)))
					continue;

				// Check if the interaction name is in the basehamiltonian_list
				if (std::find(basehamiltonian_list.begin(), basehamiltonian_list.end(), (*i)->Name()) != basehamiltonian_list.end())
				{
					// Attempt to get the matrix representing the Interaction object in the spin space
					if (!this->InteractionOperatorRotated((*i), rotmatrix, tmp))
						return false;

					result += tmp;
				}
			}

			_out = result;
			return true;
		}


}
