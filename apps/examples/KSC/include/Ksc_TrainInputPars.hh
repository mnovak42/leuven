
#ifndef KSC_TRAININPUTPARS_HH
#define KSC_TRAININPUTPARS_HH

#include "cxxopts.hh"

#include <iostream>
#include <string>

template <typename DTYPE, typename INP_DTYPE>
class Ksc_TrainInputPars {
public:

  Ksc_TrainInputPars() : fOptions(nullptr) {
    DefOpts();
  }
 ~Ksc_TrainInputPars() {
   if (fOptions) delete fOptions;
  }

public:
  //
  cxxopts::Options*      fOptions;
  //
  // --- training data set input
  size_t                 fTheTrDataNumber;
  size_t                 fTheTrDataDimension;
  std::string            fTheTrDataFile;
  // --- clustering (required)
  size_t                 fTheClusterNumber;
  INP_DTYPE              fTheClusterRBFKernelPar;
  // --- clustering (optional)
  int                    fTheClusterEncodingScheme;
  size_t                 fTheClusterEvalOutlierThreshold;
  DTYPE                  fTheClusterEvalWBalance;
  size_t                 fTheClusterLevel;
  std::string            fTheClusterResDataFile;
  std::string            fTheClusterResFile;
  //
  size_t                 fTheVerbosityLevel;
  size_t                 fTheNumBLASThreads;
  bool                   fUseGPU;

  friend std::ostream& operator<<(std::ostream& os, const Ksc_TrainInputPars& p) {
     os << "\n ===============================================================\n"
        << "\n KSC: Training (with defaults for optionals):\n\n"
        << "  ------ Training data set related: \n"
        << "  trDataNumber               = " << p.fTheTrDataNumber         << "\n"
        << "  trDataDimension            = " << p.fTheTrDataDimension      << "\n"
        << "  trDataFile                 = " << p.fTheTrDataFile           << "\n\n"
        << "  ------ Clustering related: \n"
        << "  clNumber                   = " << p.fTheClusterNumber        << "\n"
        << "  clRBFKernelPar             = " << p.fTheClusterRBFKernelPar  << "\n"
        << "  clEncodingScheme(BAS=2)    = " << p.fTheClusterEncodingScheme<< "\n"
        << "  clEvalOutlierThrs(0)       = " << p.fTheClusterEvalOutlierThreshold<< "\n"
        << "  clEvalWBalance(0.2)        = " << p.fTheClusterEvalWBalance  << "\n"
        << "  clResFile(CRes.dat)        = " << p.fTheClusterResFile       << "\n"
        << "  clResDataFile(CData.dat)   = " << p.fTheClusterResDataFile   << "\n"
        << "  clLevel(1)                 = " << p.fTheClusterLevel         << "\n\n"
        << "  ------ Other, optional parameters: \n"
        << "  verbosityLevel(2)          = " << p.fTheVerbosityLevel       << "\n"
        << "  numBLASThreads(4)          = " << p.fTheNumBLASThreads       << "\n"
        << "  useGPU                     = " << p.fUseGPU                  << "\n"
        << "\n ===============================================================\n";
    return os;
  }



  void DefOpts() {

    const std::string description =
"\n Application that Trains a KSC model using a 1D RBF kernel on the\n\
 given Training Data set by using the provided parameters.\n\n";

    if (fOptions) delete fOptions;
    fOptions = new cxxopts::Options("KSC Training: ./Ksc_Train ", description);

    fOptions->add_options("Training data set [REQUIRED]")
     ("trDataNumber"   , "(size_t) Number of training data.",         cxxopts::value<size_t>())
     ("trDataDimension", "(size_t) Dimension of the training data.",  cxxopts::value<size_t>())
     ("trDataFile"     , "(string) File name of the training data.",  cxxopts::value<std::string>())
    ;

    fOptions->add_options("Clustering [REQUIRED]")
     ("clNumber"         , "(size_t)    Number of required cluster.",                  cxxopts::value<size_t>())
     ("clRBFKernelPar"   , "(INP_DTYPE) RBF kernel parameter to be used in the KSC.",  cxxopts::value<INP_DTYPE>())
    ;
    fOptions->add_options("Clustering [OPTIONAL]")
     ("clEncodingScheme" , "(string) KSC cluster membership encoding scheme (BLF, AMS or BAS).",                     cxxopts::value<std::string>()->default_value("BAS"))
     ("clEvalOutlierThrs", "(size_t) clusters with cardinality below this are considered to be outliers with zero contibution to quality measure (0).", cxxopts::value<std::size_t>()->default_value("0"))
     ("clEvalWBalance"   , "(DTYPE)  Weight to give to the balance term in the model evaluation (must be in [0,1]).",        cxxopts::value<DTYPE>()->default_value("0.2"))
     ("clResFile"        , "(string) The result of clustering the training data is written into this file (if clLevel>0).",  cxxopts::value<std::string>()->default_value("CRes.dat"))
     ("clResDataFile"    , "(string) The reordered training data is written into this file (if clLevel>0).",                 cxxopts::value<std::string>()->default_value("CData.dat"))
     ("clLevel"          , "(size_t) KSC clustering level: 0 - clustering only; 1 - additional membership strength (AMS, BAS); 2 - membership strength for all clusters (only with AMS).",  cxxopts::value<size_t>()->default_value("1"))
    ;

    fOptions->add_options("Others [OPTIONAL]")
     ("verbosityLevel", "(size_t) Verbosity level.",                              cxxopts::value<size_t>()->default_value("2"))
     ("numBLASThreads", "(size_t) Number of threads to be used in BLAS/LAPACK.",  cxxopts::value<size_t>()->default_value("4"))
     ("useGPU"        , "(bool)   Use GPU in the training (only if `leuven` was built with -DUSE_CUBLAS).")
     ("h,help"        , "(flag)   Print usage and available parameters")
    ;
  //  std::cerr<< fOptions->help({"", "Training data set [REQUIRED]"}) << std::endl;
  }



  int GetOpts(int argc, char **argv) {
    // parse args
    try {
      auto result = fOptions->parse(argc, argv);

      // help
      if (result.count("help")>0) {
         std::cout << fOptions->help({"",
                         "Training data set [REQUIRED]",
                         "Clustering [REQUIRED]",
                         "Clustering [OPTIONAL]",
                         "Others [OPTIONAL]"
                       })
                   << std::endl;
        exit(0);
      }

      // --- training data set related:
      if (result.count("trDataNumber")>0) {
        fTheTrDataNumber = result["trDataNumber"].as<size_t>();
      } else {
        throw cxxopts::OptionException("  '--trDataNumber' is a required argument");
      }
      if (result.count("trDataDimension")>0) {
        fTheTrDataDimension = result["trDataDimension"].as<size_t>();
      } else {
        throw cxxopts::OptionException("  '--trDataDimension' is a required argument");
      }
      if (result.count("trDataFile")>0) {
        fTheTrDataFile = result["trDataFile"].as<std::string>();
      } else {
        throw cxxopts::OptionException("  '--trDataFile' is a required argument");
      }

      // --- clustering related (required):
      if (result.count("clNumber")>0) {
        fTheClusterNumber = result["clNumber"].as<size_t>();
      } else {
        throw cxxopts::OptionException("  '--clNumber' is a required argument");
      }
      if (result.count("clRBFKernelPar")>0) {
        fTheClusterRBFKernelPar = result["clRBFKernelPar"].as<INP_DTYPE>();
      } else {
        throw cxxopts::OptionException("  '--clRBFKernelPar' is a required argument");
      }
      // --- clustering related (optional):
      std::string str = result["clEncodingScheme"].as<std::string>();
      if (str.compare("BLF")==0) {
        fTheClusterEncodingScheme = 0;
      } else if (str.compare("AMS")==0) {
        fTheClusterEncodingScheme = 1;
      } else if (str.compare("BAS")==0) {
        fTheClusterEncodingScheme = 2;
      } else {
        throw cxxopts::OptionException("  '--clEncodingSchem' unknown argument value");
      }
      fTheClusterEvalOutlierThreshold = std::max(std::size_t(0), result["clEvalOutlierThrs"].as<std::size_t>());
      fTheClusterEvalWBalance         = std::max(0.            , std::min(1.            , result["clEvalWBalance"].as<DTYPE>()));
      fTheClusterResFile              = result["clResFile"].as<std::string>();
      fTheClusterResDataFile          = result["clResDataFile"].as<std::string>();
      size_t cLevel = result["clLevel"].as<size_t>();
      if (cLevel<3) {
        fTheClusterLevel = cLevel;
      } else {
        throw cxxopts::OptionException("  '--clLevel' unknown argument value");
      }

      // --- Other, optionals (i.e. with default):
      fTheVerbosityLevel = result["verbosityLevel"].as<size_t>();
      fTheNumBLASThreads = result["numBLASThreads"].as<size_t>();
      fUseGPU            = result.count("useGPU")>0 ? true : false;

    } catch (const cxxopts::OptionException& oe) {
        std::cerr << "\n*** Wrong input argument (see usage below): \n"
                  << oe.what()
                  << "\n------------------------------------------------ \n"
                  << fOptions->help({"",
                                     "Training data set [REQUIRED]",
                                     "Clustering [REQUIRED]",
                                     "Clustering [OPTIONAL]",
                                     "Others [OPTIONAL]"
                                    })
                  << std::endl;
        return -1;
    }
    return 0;
  }
};


#endif