#!/bin/bash
set -x
set -e

target=$(hostname --fqdn)

# Set internet proxy for summit and ascent. Psana2 needs this access.
if [[ ${target} = *".crusher."* || ${target} = *".frontier."* || ${target} = *".summit."* || ${target} = *".ascent."* ]]; then
    export all_proxy=socks://proxy.ccs.ornl.gov:3128/
    export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
    export http_proxy=http://proxy.ccs.ornl.gov:3128/
    export https_proxy=http://proxy.ccs.ornl.gov:3128/
    export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'
fi


# Set job submisson command
if [[ ${target} = *"summit"* || ${target} = *"ascent"* ]]; then
    export SKOPI_TEST_LAUNCHER="jsrun -n1 -a1 -g1"
elif [[ ${target} = *"perlmutter"* || ${target} = *"frontier"* ]]; then
    export SKOPI_TEST_LAUNCHER="srun -n1 -G1"
fi


# Pick up all the tests
export USE_CUPY=1
$SKOPI_TEST_LAUNCHER pytest tests/test_diffraction.py
$SKOPI_TEST_LAUNCHER pytest beam/tests
$SKOPI_TEST_LAUNCHER pytest detector/tests
$SKOPI_TEST_LAUNCHER pytest geometry/tests
