
YELLOW='\033[1;33m'
RED='\033[1;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No color

pushd .
cd src

python plot_tsne_dataset.py

popd