
VECTORS=AGM CAIA CAIAConAGM Cisco Consensus OptOut TA
PYTHON=python3

RESULTFILES=$(addprefix results/, %_HBOS_results.csv %_LOF_results.csv %_KNN_results.csv %_IF_results.csv %_SDO_results.csv) 
 
all: ${VECTORS} forward_search.out

${VECTORS}: %: ${RESULTFILES} tuneres/%_tuneres.out

%_data.npy: %.csv
	${PYTHON} od.py $* pre
	
${RESULTFILES}: %_data.npy
	mkdir -p scores hist histlog results stats 
	${PYTHON} od.py $* nopre HBOS
	${PYTHON} od.py $* nopre LOF
	${PYTHON} od.py $* nopre KNN
	${PYTHON} od.py $* nopre IF
	${PYTHON} od.py $* nopre SDO
		
tuneres/%_tuneres.out: %.csv
	mkdir -p tuneres
	${PYTHON} -u tune.py $* > tuneres/$*_tuneres.out
	
forward_search.out: CAIAConAGM.csv
	${PYTHON} -u find_vector.py > forward_search.out

.PHONY: all ${VECTORS}
.SECONDARY: 
