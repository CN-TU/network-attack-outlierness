
VECTORS=AGM CAIA CAIAConAGM Cisco Consensus OptOut TA
PYTHON=python3

all: ${VECTORS} forward-search.out

forward_search.out: CAIAConAGM.csv
	${PYTHON} -u find_vector.py > forward_search.out

%_data.npy: %.csv
	${PYTHON} od.py $* pre
	
%_results.csv: %_data.npy
	mkdir -p scores hist histlog results stats 
	${PYTHON} od.py $* nopre HBOS
	${PYTHON} od.py $* nopre LOF
	${PYTHON} od.py $* nopre KNN
	${PYTHON} od.py $* nopre IF
	${PYTHON} od.py $* nopre SDO
	
%_tuneres.out: %.csv
	${PYTHON} -u tune.py $* > $*_tuneres.out

${VECTORS}: %: %_results.csv %_tuneres.out
