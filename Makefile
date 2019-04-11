
VECTORS=AGM CAIA CAIAConAGM Cisco Consensus OptOut TA
PYTHON=python3

all: ${VECTORS} forward_search.out

forward_search.out: CAIAConAGM.csv
	${PYTHON} -u find_vector.py > forward_search.out

%_data.npy: %.csv
	${PYTHON} od.py $* pre
	
results/%_results.csv: %_data.npy
	mkdir -p scores hist histlog results stats 
	${PYTHON} od.py $* nopre HBOS
	${PYTHON} od.py $* nopre LOF
	${PYTHON} od.py $* nopre KNN
	${PYTHON} od.py $* nopre IF
	${PYTHON} od.py $* nopre SDO
	
tuneres/%_tuneres.out: %.csv
	mkdir -p tuneres
	${PYTHON} -u tune.py $* > tuneres/$*_tuneres.out

${VECTORS}: %: results/%_results.csv tuneres/%_tuneres.out
