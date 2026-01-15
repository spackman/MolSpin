#ifndef MOD_RunSection_TaskStaticHSTrEPRSpectra
#define MOD_RunSection_TaskStaticHSTrEPRSpectra

#include "BasicTask.h"
#include "SpinSpace.h"
#include <tuple>

namespace RunSection
{
	class TaskStaticHSTrEPRSpectra : public BasicTask
	{
	private:
		double mwFrequencyGHz;
		double linewidthFad_mT;
		double linewidthDonor_mT;
		std::string lineshape;
		int powdersamplingpoints;
		std::string electron1Name;
		std::string electron2Name;
		std::string fieldInteractionName;
		std::string initialStateName;
		std::vector<std::string> hamiltonianH0list;

		double LineshapeValue(double _delta, double _fwhm) const;
		double LinewidthToOmega(double _fwhm_mT, double _giso) const;
		bool CreateRotationMatrix(double &_alpha, double &_beta, double &_gamma, arma::mat &_R) const;
		bool CreateUniformGrid(int &_Npoints, std::vector<std::tuple<double, double, double>> &_uniformGrid) const;
		void WriteHeader(std::ostream &_stream);

	protected:
		bool RunLocal() override;
		bool Validate() override;

	public:
		TaskStaticHSTrEPRSpectra(const MSDParser::ObjectParser &, const RunSection &);
		~TaskStaticHSTrEPRSpectra();
	};
}

#endif
