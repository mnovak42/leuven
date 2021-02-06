
#ifndef KSC_TUNEINPUTPARS_HH
#define KSC_TUNEINPUTPARS_HH

#include "cxxopts.hh"

#include <iostream>
#include <string>

template <typename DTYPE, typename INP_DTYPE>
class Ksc_TuneInputPars {
public:

  Ksc_TuneInputPars() : fOptions(nullptr) {
    DefOpts();
  }
 ~Ksc_TuneInputPars() {
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
  // --- validation data set input
  size_t                 fTheValDataNumber;
  std::string            fTheValDataFile;
  // --- clustering (optional)
  int                    fTheClusterEncodingScheme;
  size_t                 fTheClusterEvalOutlierThreshold;
  DTYPE                  fTheClusterEvalWBalance;
  // --- tuning: cluster number and kernle parametrs
  size_t                 fTheMinClusterNumber;
  size_t                 fTheMaxClusterNumber;
  std::vector<INP_DTYPE> fTheKernelParameters;
  //
  size_t                 fTheVerbosityLevel;
  size_t                 fTheNumBLASThreads;
  std::string            fTheResFile;
  bool                   fUseGPU;

  friend std::ostream& operator<<(std::ostream& os, const Ksc_TuneInputPars& p) {
     os << "\n ===============================================================\n"
        << "\n KSC: Hyper Parameter Tuning (with defaults for optionals):\n\n"
        << "  ------ Training data set related: \n"
        << "  trDataNumber               = " << p.fTheTrDataNumber         << "\n"
        << "  trDataDimension            = " << p.fTheTrDataDimension      << "\n"
        << "  trDataFile                 = " << p.fTheTrDataFile           << "\n\n"
        << "  ------ Validation data set related: \n"
        << "  valDataNumber              = " << p.fTheValDataNumber        << "\n"
        << "  valDataFile                = " << p.fTheValDataFile          << "\n\n"
        << "  ------ Tuning related: \n"
        << "  minClusterNumber           = " << p.fTheMinClusterNumber     << "\n"
        << "  maxClusterNumber           = " << p.fTheMaxClusterNumber     << "\n"
        << "  kernelPrameters            = ";
        size_t nKerPars = p.fTheKernelParameters.size();
        if (nKerPars > 3) {
           os << "{" << p.fTheKernelParameters[0] << ", " << p.fTheKernelParameters[1]
              << ", ..., " << p.fTheKernelParameters[nKerPars-1]
              << "}  --> " << nKerPars << " number of parameters. \n\n";
        } else {
          os << "{" << p.fTheKernelParameters[0] << ", " << p.fTheKernelParameters[1]
             << ", "<< p.fTheKernelParameters[2]
             << "}  --> " << nKerPars << " number of parameters. \n\n";
        }
     os << "  ------ Clustering related: \n"
        << "  clEncodingScheme(BAS=2)    = " << p.fTheClusterEncodingScheme      << "\n"
        << "  clEvalOutlierThrs(0)       = " << p.fTheClusterEvalOutlierThreshold<< "\n"
        << "  clEvalWBalance(0.2)        = " << p.fTheClusterEvalWBalance        << "\n\n"
        << "  ------ Other, optional parameters: \n"
        << "  verbosityLevel(2)          = " << p.fTheVerbosityLevel       << "\n"
        << "  numBLASThreads(4)          = " << p.fTheNumBLASThreads       << "\n"
        << "  resFile(TuningRes)         = " << p.fTheResFile              << "\n"
        << "  useGPU                     = " << p.fUseGPU                  << "\n"
        << "\n ===============================================================\n";
    return os;
  }



  void DefOpts() {

    const std::string description =
"\n Application that tunes a KSC model using a 1D RBF kernel to find the optimal\n\
 values of the `kernel parameter` (bandwidth) and `clusters number` hyper\n\
 parameters.\n\n\
 The application trains a KSC model at each point of a 2D 'kernel parameter' -\n\
 'cluster number' grid (defined by the input arguments) on the given training \n\
 data set. Each of these model is applied on the validation data set and the \n\
 corresponding model evaluation criterion is computed. The 2D point, giving the\n\
 best value of the model evaluation, is reported and all values over the 2D \n\
 grid are written to the output file. The optimal kernel parameter and cluster \n\
 number parameters then can be determined by inspecting these values.\n\n";

    if (fOptions) delete fOptions;
    fOptions = new cxxopts::Options("KSC: Hyper Parameter Tuning", description);

    fOptions->add_options("Training data set [REQUIRED]")
     ("trDataNumber"   , "(size_t) Number of training data.",         cxxopts::value<size_t>())
     ("trDataDimension", "(size_t) Dimension of the training data.",  cxxopts::value<size_t>())
     ("trDataFile"     , "(string) File name of the training data.",  cxxopts::value<std::string>())
    ;

    fOptions->add_options("Validation data set [REQUIRED]")
     ("valDataNumber"   , "(size_t) Number of validation data.",         cxxopts::value<size_t>())
     ("valDataFile"     , "(string) File name of the validation data.",  cxxopts::value<std::string>())
    ;

    fOptions->add_options("Tuning [REQUIRED]")
     ("minClusterNumber", "(size_t) Minimum cluster number for grid search.",  cxxopts::value<size_t>())
     ("maxClusterNumber", "(size_t) Maximum cluster numebr for grid search.",  cxxopts::value<size_t>())
     ("kernelParameters", "(vector of INP_DTYPE) List of RBF kernel parameters for grid search (separated by ',')",  cxxopts::value< std::vector<INP_DTYPE> >())
    ;

    fOptions->add_options("Clustering [OPTIONAL]")
     ("clEncodingScheme" , "(string) KSC cluster membership encoding scheme (BLF, AMS or BAS).",                           cxxopts::value<std::string>()->default_value("BAS"))
     ("clEvalOutlierThrs", "(size_t) clusters with cardinality below this are considered to be outliers with zero contibution to quality measure (0).", cxxopts::value<std::size_t>()->default_value("0"))
     ("clEvalWBalance"   , "(DTYPE)  Weight to give to the balance term in the model evaluation (must be in [0,1]).",      cxxopts::value<DTYPE>()->default_value("0.2"))
    ;

    fOptions->add_options("Others [OPTIONAL]")
     ("verbosityLevel"  , "(size_t) Verbosity level.",                              cxxopts::value<size_t>()->default_value("2"))
     ("numBLASThreads"  , "(size_t) Number of threads to be used in BLAS/LAPACK.",  cxxopts::value<size_t>()->default_value("4"))
     ("resFile"         , "(string) The result of tuning the KSC model is written into this file.",  cxxopts::value<std::string>()->default_value("TuningRes.dat"))
     ("useGPU"          , "(bool)   Use GPU in the training (only if `leuven` was built with -DUSE_CUBLAS).")
     ("h,help"          , "(flag)   Print usage and available parameters")
    ;
  }



  int GetOpts(int argc, char **argv) {
    // parse args
    try {
      auto result = fOptions->parse(argc, argv);
      // help
      if (result.count("help")>0) {
         std::cout << fOptions->help({"",
                         "Training data set [REQUIRED]",
                         "Validation data set [REQUIRED]",
                         "Tuning [REQUIRED]",
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

      // --- validation data set related:
      if (result.count("valDataNumber")>0) {
        fTheValDataNumber = result["valDataNumber"].as<size_t>();
      } else {
        throw cxxopts::OptionException("  '--valDataNumber' is a required argument");
      }
      if (result.count("valDataFile")>0) {
        fTheValDataFile = result["valDataFile"].as<std::string>();
      } else {
        throw cxxopts::OptionException("  '--valDataFile' is a required argument");
      }

      // --- tuning related (required):
      if (result.count("minClusterNumber")>0) {
        fTheMinClusterNumber = result["minClusterNumber"].as<size_t>();
      } else {
        throw cxxopts::OptionException("  '--minClusterNumber' is a required argument");
      }
      if (result.count("maxClusterNumber")>0) {
        fTheMaxClusterNumber = result["maxClusterNumber"].as<size_t>();
      } else {
        throw cxxopts::OptionException("  '--maxClusterNumber' is a required argument");
      }
      if (fTheMinClusterNumber>fTheMaxClusterNumber) {
        throw cxxopts::OptionException("  '--maxClusterNumber' must be > than '--minClusterNumber' !");
      }
      if (result.count("kernelParameters")>0) {
        fTheKernelParameters = result["kernelParameters"].as< std::vector<INP_DTYPE> >();
      } else {
        throw cxxopts::OptionException("  '--kernelParameters' is a required argument");
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

      // --- Other, optionals (i.e. with default):
      fTheVerbosityLevel = result["verbosityLevel"].as<size_t>();
      fTheNumBLASThreads = result["numBLASThreads"].as<size_t>();
      fTheResFile        = result["resFile"].as<std::string>();
      // remove extension from fTheResFile
      fTheResFile = fTheResFile.substr(0, fTheResFile.find_last_of("."));
      fUseGPU     = result.count("useGPU")>0 ? true : false;

    } catch (const cxxopts::OptionException& oe) {
        std::cerr << "\n*** Wrong input argument (see usage below): \n"
                  << oe.what()
                  << "\n------------------------------------------------ \n"
                  << fOptions->help({"",
                                     "Training data set [REQUIRED]",
                                     "Validation data set [REQUIRED]",
                                     "Tuning [REQUIRED]",
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
