
#ifndef KSCICHOL_TESTINPUTPARS_HH
#define KSCICHOL_TESTINPUTPARS_HH

#include "cxxopts.hh"

#include <iostream>
#include <string>

template <typename DTYPE, typename INP_DTYPE>
class KscIchol_TestInputPars {
public:

  KscIchol_TestInputPars() : fOptions(nullptr) {
    DefOpts();
  }
 ~KscIchol_TestInputPars() {
   if (fOptions) delete fOptions;
  }

public:
  //
  cxxopts::Options*      fOptions;
  //
  // --- incomplete Cholesky (required)
  DTYPE                  fTheIcholTolError;
  size_t                 fTheIcholMaxRank;
  std::vector<INP_DTYPE> fTheIcholRBFKernelPar;
  // --- incomplete Cholesky (optional)
  std::string            fTheIcholRedSetFile;
  std::string            fTheIcholPermVectFile;

  // --- training data set input
  size_t                 fTheTrDataNumber;
  size_t                 fTheTrDataDimension;
  std::string            fTheTrDataFile;
  // --- test data set input
  size_t                 fTheTestDataNumber;
  std::string            fTheTestDataFile;
  // --- clustering (required)
  size_t                 fTheClusterNumber;
  INP_DTYPE              fTheClusterRBFKernelPar;
  // --- clustering (optional)
  int                    fTheClusterEncodingScheme;
  size_t                 fTheClusterEvalOutlierThreshold;
  DTYPE                  fTheClusterEvalWBalance;
  size_t                 fTheClusterLevel;
  std::string            fTheClusterResFile;
  //
  size_t                 fTheVerbosityLevel;
  size_t                 fTheNumBLASThreads;
  bool                   fUseGPU;

  friend std::ostream& operator<<(std::ostream& os, const KscIchol_TestInputPars& p) {
     os << "\n ===============================================================\n"
        << "\n Ksc Training & Testing Input Parameters (with defaults for optionals):\n\n"
        << "  ------ Cholesky decomposition related: \n"
        << "  icholTolError              = " << p.fTheIcholTolError        << "\n"
        << "  icholMaxRank               = " << p.fTheIcholMaxRank         << "\n"
        //<< "  icholRBFKernelPar          = " << p.fTheIcholRBFKernelPar    << "\n\n"
        << "  icholRBFKernelPar          = ";
        size_t nIcholPars = p.fTheIcholRBFKernelPar.size();
        if (nIcholPars == 1) {
           os << p.fTheIcholRBFKernelPar[0] << "\n";
        } else if (nIcholPars == 2) {
           os << "{" << p.fTheIcholRBFKernelPar[0] << ", " << p.fTheIcholRBFKernelPar[1]
              << "}  --> " << nIcholPars << " number of parameters. \n";
        } else if (nIcholPars == 3) {
           os << "{" << p.fTheIcholRBFKernelPar[0] << ", " << p.fTheIcholRBFKernelPar[1]
              << ", "<< p.fTheIcholRBFKernelPar[2]
              << "}  --> " << nIcholPars << " number of parameters. \n";
        } else {
           os << "{" << p.fTheIcholRBFKernelPar[0] << ", " << p.fTheIcholRBFKernelPar[1]
              << ", ..., " << p.fTheIcholRBFKernelPar[nIcholPars-1]
              << "}  --> " << nIcholPars << " number of parameters. \n";
        }
        if (!p.fTheIcholRedSetFile.empty())
          os << "  icholRedSetFile            = " << p.fTheIcholRedSetFile   << "\n";
        if (!p.fTheIcholPermVectFile.empty())  
          os << "  icholPermVectFile          = " << p.fTheIcholPermVectFile   << "\n";
     os << "\n"      
        << "  ------ Training data set related: \n"
        << "  trDataNumber               = " << p.fTheTrDataNumber         << "\n"
        << "  trDataDimension            = " << p.fTheTrDataDimension      << "\n"
        << "  trDataFile                 = " << p.fTheTrDataFile           << "\n\n"
        << "  ------ Test data set related: \n"
        << "  tstDataNumber              = " << p.fTheTestDataNumber       << "\n"
        << "  tstDataFile                = " << p.fTheTestDataFile         << "\n\n"
        << "  ------ Clustering related: \n"
        << "  clNumber                   = " << p.fTheClusterNumber        << "\n"
        << "  clRBFKernelPar             = " << p.fTheClusterRBFKernelPar  << "\n"
        << "  clEncodingScheme(BAS=2)    = " << p.fTheClusterEncodingScheme<< "\n"
        << "  clEvalOutlierThrs(0)       = " << p.fTheClusterEvalOutlierThreshold<< "\n"
        << "  clEvalWBalance(0.2)        = " << p.fTheClusterEvalWBalance  << "\n"
        << "  clResFile(CRes.dat)        = " << p.fTheClusterResFile       << "\n"
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
  "\n   Application that trains a sparse KSC model using a 1D RBF kernel on the given\n\
   training data set and applies the trained model to cluster a given test data set.\n\
   \n\
   The sparsity is achived with the combination of approximating the training set\n\
   kernel matrix by its incomplete Cholesky factorisation (i.e. incomplete QR \n\
   decomposition of the training data feature map) and using the reduced set method.\n\n";

    if (fOptions) delete fOptions;
    fOptions = new cxxopts::Options("KSC Training & Testing Application", description);

    // add argument that are related to the incomplete Cholesky factorisation of
    // the training data set
    fOptions->add_options("Cholesky decomposition [REQUIRED]")
     ("icholTolError"     , "(double)    Tolerated approximate error in the inc. Cholesky decomposition.",     cxxopts::value<double>())
     ("icholMaxRank"      , "(size_t)    Maximum number of data to select in the inc. Cholesky decomposition.", cxxopts::value<size_t>())
     ("icholRBFKernelPar" , "(INP_DTYPE) RBF kernel parameter to be used in the inc. Cholesky decomposition (scalar or vector).",  cxxopts::value< std::vector<INP_DTYPE> >())
    ;

    fOptions->add_options("Cholesky decomposition [OPTIONAL]")
     ("icholRedSetFile"   , "(string) The reduced set data is written into this file (if given).",                   cxxopts::value<std::string>()->default_value(""))
     ("icholPermVectFile" , "(string) The permutations (done during the ICD) is written into this file (if given).", cxxopts::value<std::string>()->default_value(""))
    ;

    fOptions->add_options("Training data set [REQUIRED]")
     ("trDataNumber"   , "(size_t) Number of training data.",         cxxopts::value<size_t>())
     ("trDataDimension", "(size_t) Dimension of the training data.",  cxxopts::value<size_t>())
     ("trDataFile"     , "(string) File name of the training data.",  cxxopts::value<std::string>())
    ;

    fOptions->add_options("Test data set [REQUIRED]")
     ("tstDataNumber"   , "(size_t) Number of test data.",         cxxopts::value<size_t>())
     ("tstDataFile"     , "(string) File name of the test data.",  cxxopts::value<std::string>())
    ;

    fOptions->add_options("Clustering [REQUIRED]")
     ("clNumber"         , "(size_t)    Number of required cluster.",                  cxxopts::value<size_t>())
     ("clRBFKernelPar"   , "(INP_DTYPE) RBF kernel parameter to be used in the KSC.",  cxxopts::value<INP_DTYPE>())
    ;
    fOptions->add_options("Clustering [OPTIONAL]")
     ("clEncodingScheme" , "(string) KSC cluster membership encoding scheme (BLF, AMS or BAS).",                        cxxopts::value<std::string>()->default_value("BAS"))
     ("clEvalOutlierThrs", "(size_t) clusters with cardinality below this are considered to be outliers with zero contibution to quality measure (0).", cxxopts::value<std::size_t>()->default_value("0"))
     ("clEvalWBalance"   , "(DTYPE)  Weight to give to the balance term in the model evaluation (must be in [0,1]).",   cxxopts::value<DTYPE>()->default_value("0.2"))
     ("clResFile"        , "(string) The result of clustering the test data is written into this file (if clLevel>0).", cxxopts::value<std::string>()->default_value("CRes.dat"))
     ("clLevel"          , "(size_t) KSC clustering level: 0 - clustering only; 1 - additional membership strength (AMS, BAS); 2 - membership strength for all clusters (only with AMS).",  cxxopts::value<size_t>()->default_value("1"))
    ;

    fOptions->add_options("Others [OPTIONAL]")
     ("verbosityLevel", "(size_t) Verbosity level.",                              cxxopts::value<size_t>()->default_value("2"))
     ("numBLASThreads", "(size_t) Number of threads to be used in BLAS/LAPACK.",  cxxopts::value<size_t>()->default_value("4"))
     ("useGPU"        , "(bool)   Use GPU in the training (only if `leuven` was built with -DUSE_CUBLAS).")
     ("h,help"        , "(flag)   Print usage and available parameters")
    ;
  //  std::cerr<< fOptions->help({"", "Cholesky decomposition [REQUIRED]", "Training data set [REQUIRED]"}) << std::endl;
  }



  int GetOpts(int argc, char **argv) {
    // parse args
    try {
      auto result = fOptions->parse(argc, argv);
      // help
      if (result.count("help")>0) {
         std::cout << fOptions->help({"",
                         "Cholesky decomposition [REQUIRED]",
                         "Cholesky decomposition [OPTIONAL]",
                         "Training data set [REQUIRED]",
                         "Test data set [REQUIRED]",
                         "Clustering [REQUIRED]",
                         "Clustering [OPTIONAL]",
                         "Others [OPTIONAL]"
                       })
                   << std::endl;
        exit(0);
      }
      // --- incomplete Cholesky related:
      if (result.count("icholTolError")>0) {
        fTheIcholTolError = result["icholTolError"].as<double>();
      } else {
        throw cxxopts::OptionException("  '--icholTolError' is a required argument");
      }
      if (result.count("icholMaxRank")>0) {
        fTheIcholMaxRank = result["icholMaxRank"].as<size_t>();
      } else {
        throw cxxopts::OptionException("  '--icholMaxRank' is a required argument");
      }
      if (result.count("icholRBFKernelPar")>0) {
        fTheIcholRBFKernelPar = result["icholRBFKernelPar"].as< std::vector<INP_DTYPE> >();
      } else {
        throw cxxopts::OptionException("  '--icholRBFKernelPar' is a required argument");
      }
      fTheIcholRedSetFile   = result["icholRedSetFile"].as<std::string>();
      fTheIcholPermVectFile = result["icholPermVectFile"].as<std::string>();

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

      // --- test data set related:
      if (result.count("tstDataNumber")>0) {
        fTheTestDataNumber = result["tstDataNumber"].as<size_t>();
      } else {
        throw cxxopts::OptionException("  '--tstDataNumber' is a required argument");
      }
      if (result.count("tstDataFile")>0) {
        fTheTestDataFile = result["tstDataFile"].as<std::string>();
      } else {
        throw cxxopts::OptionException("  '--tstDataFile' is a required argument");
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
                                     "Cholesky decomposition [REQUIRED]",
                                     "Cholesky decomposition [OPTIONAL]",
                                     "Training data set [REQUIRED]",
                                     "Test data set [REQUIRED]",
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
