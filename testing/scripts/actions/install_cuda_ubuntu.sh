## 
## THIS SCRIPT IS DEVELOPED BY PETER HEYWOOD AND TAKEN FROM 
## https://github.com/ptheywood/cuda-cmake-github-actions
##

# @todo - better / more robust parsing of inputs from env vars.
## -------------------
## Constants
## -------------------

# @todo - apt repos/known supported versions?

# @todo - GCC support matrix?

# List of sub-packages to install.
# @todo - pass this in from outside the script? 
# @todo - check the specified subpackages exist via apt pre-install?  apt-rdepends cuda-9-0 | grep "^cuda-"?

# Ideally choose from the list of meta-packages to minimise variance between cuda versions (although it does change too)
CUDA_PACKAGES_IN=(
    "command-line-tools"
    "libraries-dev"
)

## -------------------
## Bash functions
## -------------------
# returns 0 (true) if a >= b
function version_ge() {
    [ "$#" != "2" ] && echo "${FUNCNAME[0]} requires exactly 2 arguments." && exit 1
    [ "$(printf '%s\n' "$@" | sort -V | head -n 1)" == "$2" ]
}
# returns 0 (true) if a > b
function version_gt() {
    [ "$#" != "2" ] && echo "${FUNCNAME[0]} requires exactly 2 arguments." && exit 1
    [ "$1" = "$2" ] && return 1 || version_ge $1 $2
}
# returns 0 (true) if a <= b
function version_le() {
    [ "$#" != "2" ] && echo "${FUNCNAME[0]} requires exactly 2 arguments." && exit 1
    [ "$(printf '%s\n' "$@" | sort -V | head -n 1)" == "$1" ]
}
# returns 0 (true) if a < b
function version_lt() {
    [ "$#" != "2" ] && echo "${FUNCNAME[0]} requires exactly 2 arguments." && exit 1
    [ "$1" = "$2" ] && return 1 || version_le $1 $2
}

## -------------------
## Select CUDA version
## -------------------

# Get the cuda version from the environment as $cuda.
CUDA_VERSION_MAJOR_MINOR=${cuda}

# Split the version.
# We (might/probably) don't know PATCH at this point - it depends which version gets installed.
CUDA_MAJOR=$(echo "${CUDA_VERSION_MAJOR_MINOR}" | cut -d. -f1)
CUDA_MINOR=$(echo "${CUDA_VERSION_MAJOR_MINOR}" | cut -d. -f2)
CUDA_PATCH=$(echo "${CUDA_VERSION_MAJOR_MINOR}" | cut -d. -f3)
# use lsb_release to find the OS.
UBUNTU_VERSION=$(lsb_release -sr)
UBUNTU_VERSION="${UBUNTU_VERSION//.}"

echo "CUDA_MAJOR: ${CUDA_MAJOR}"
echo "CUDA_MINOR: ${CUDA_MINOR}"
echo "CUDA_PATCH: ${CUDA_PATCH}"
# echo "UBUNTU_NAME: ${UBUNTU_NAME}"
echo "UBUNTU_VERSION: ${UBUNTU_VERSION}"

# If we don't know the CUDA_MAJOR or MINOR, error.
if [ -z "${CUDA_MAJOR}" ] ; then
    echo "Error: Unknown CUDA Major version. Aborting."
    exit 1
fi
if [ -z "${CUDA_MINOR}" ] ; then
    echo "Error: Unknown CUDA Minor version. Aborting."
    exit 1
fi
# If we don't know the Ubuntu version, error.
if [ -z ${UBUNTU_VERSION} ]; then
    echo "Error: Unknown Ubuntu version. Aborting."
    exit 1
fi


## ---------------------------
## GCC studio support check?
## ---------------------------

# @todo

## -------------------------------
## Select CUDA packages to install
## -------------------------------
CUDA_PACKAGES=""
for package in "${CUDA_PACKAGES_IN[@]}"
do : 
    # @todo This is not perfect. Should probably provide a separate list for diff versions
    # cuda-compiler-X-Y if CUDA >= 9.1 else cuda-nvcc-X-Y
    if [[ "${package}" == "nvcc" ]] && version_ge "$CUDA_VERSION_MAJOR_MINOR" "9.1" ; then
        package="compiler"
    elif [[ "${package}" == "compiler" ]] && version_lt "$CUDA_VERSION_MAJOR_MINOR" "9.1" ; then
        package="nvcc"
    fi
    # Build the full package name and append to the string.
    CUDA_PACKAGES+=" cuda-${package}-${CUDA_MAJOR}-${CUDA_MINOR}"
done
echo "CUDA_PACKAGES ${CUDA_PACKAGES}"

## -----------------
## Prepare to install
## -----------------

PIN_FILENAME="cuda-ubuntu${UBUNTU_VERSION}.pin"
PIN_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VERSION}/x86_64/${PIN_FILENAME}"
APT_KEY_URL="http://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VERSION}/x86_64/7fa2af80.pub"
REPO_URL="http://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VERSION}/x86_64/"

echo "PIN_FILENAME ${PIN_FILENAME}"
echo "PIN_URL ${PIN_URL}"
echo "APT_KEY_URL ${APT_KEY_URL}"

## -----------------
## Install
## -----------------
echo "Adding CUDA Repository"
wget ${PIN_URL}
sudo mv ${PIN_FILENAME} /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys ${APT_KEY_URL}
sudo add-apt-repository "deb ${REPO_URL} /"
sudo apt-get update

echo "Installing CUDA packages ${CUDA_PACKAGES}"
sudo apt-get -y install ${CUDA_PACKAGES}

if [[ $? -ne 0 ]]; then
    echo "CUDA Installation Error."
    exit 1
fi
## -----------------
## Set environment vars / vars to be propagated
## -----------------

CUDA_PATH=/usr/local/cuda-${CUDA_MAJOR}.${CUDA_MINOR}
echo "CUDA_PATH=${CUDA_PATH}"
export CUDA_PATH=${CUDA_PATH}


# Quick test. @temp
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib:$LD_LIBRARY_PATH"
nvcc -V
