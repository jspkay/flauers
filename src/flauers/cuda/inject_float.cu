// TODO: include stdlib.h to provide explicit definitions of int32_t and others


/**************** FLOAT32 FUNCTIONS *************** */
extern "C" __device__ int
bitflip_float32(
	float* output,
	float input,
	int bitstring){

	int to_inject = *( (int*) &input );
	to_inject = to_inject ^ bistring;
	*output = *( (float*) &to_inject);
	return 0;
}
 
 
extern "C" __device__ int
sa0_float32(
	float* output,
	float input,
	int bitstring ){

	int to_inject = *( (int*) &input );
	to_inject = to_inject & ~bistring;
	*output = *( (double*) &to_inject);
	return 0;
}

extern "C" __device__ int
sa1_float32(
	float* output,
	float input,
	int bitstring ){

	int to_inject = *( (int*) &input );
	to_inject = to_inject | bistring;
	*output = *( (double*) &to_inject);
	return 0;
}

/**************** FLOAT64 FUNCTIONS *************** */
extern "C" __device__ int
bitflip_float64(
	double *output,
	double input,
	int bitstring ){
	
	long to_inject = *( (long*) &input );
	to_inject = to_inject ^ bistring;
	*output = *( (double*) &to_inject);
	return 0;
}
