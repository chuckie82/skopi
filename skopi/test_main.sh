#!/bin/bash
set -x
set -e
    

# Set internet proxy for summit and ascent. Psana2 needs this access.
if [[ $(hostname --fqdn) = *".crusher."* || $(hostname --fqdn) = *".frontier."* || $(hostname --fqdn) = *".summit."* || $(hostname --fqdn) = *".ascent."* ]]; then
    export all_proxy=socks://proxy.ccs.ornl.gov:3128/
    export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
    export http_proxy=http://proxy.ccs.ornl.gov:3128/
    export https_proxy=http://proxy.ccs.ornl.gov:3128/
    export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'
fi


# Pick up all the tests
jsrun -n1 -g1 pytest -s tests/test_diffraction.py
jsrun -n1 -g1 pytest beam/tests
jsrun -n1 -g1 pytest detector/tests
jsrun -n1 -g1 pytest -s geometry/tests
export USE_CUPY=1
jsrun -n1 -g1 pytest -s tests/test_diffraction.py
jsrun -n1 -g1 pytest beam/tests
jsrun -n1 -g1 pytest detector/tests
jsrun -n1 -g1 pytest -s geometry/tests
