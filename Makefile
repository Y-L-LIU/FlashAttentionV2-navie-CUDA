flash_atten:src/kernel.cu
	nvcc -o flashatten src/kernel.cu --expt-relaxed-constexpr -arch sm_75 -O3

debug:
	nvcc -o flashatten_debug src/kernel.cu --expt-relaxed-constexpr -arch sm_75 -g  -G 

baseline:src/baseline.cu
	nvcc -o baseline src/baseline.cu --expt-relaxed-constexpr -arch sm_75