
YELLOW='\033[1;33m'
RED='\033[1;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No color

pushd .
cd src

python compare_gradcam_shap_generation_time.py

popd