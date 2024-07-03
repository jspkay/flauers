// TODO: include stdlib.h to provide explicit definitions of int32_t and others

extern "C" __device__ int
inject_int8(
	int* output,
	int input,
	int bitstring, 
	int type ){

	if(type == 0)      *output &= ~bitstring; // sa0
	else if(type == 1) *output |=  bitstring;  // sa1
	else if(type == 2) *output ^=  bitstring;  // bit flip
	else 			   *output  =  bitstring; // bizantine

	return 0;
} 

extern "C" __device__ int
inject_int32(
	int* output,
	int input,
	int bitstring, 
	int type ){

	if(type == 0)      *output &= ~bitstring; // sa0
	else if(type == 1) *output |=  bitstring;  // sa1
	else if(type == 2) *output ^=  bitstring;  // bit flip
	else 			   *output  =  bitstring; // bizantine

	return 0;
} 

/**************** FLOAT32 *************** */
extern "C" __device__ int
inject_float32(
	float* output,
	float input,
	int bitstring, 
	int type ){

	int to_inject = *( (int*) &input );

	if(type == 0)      to_inject &= ~bitstring; // sa0
	else if(type == 1) to_inject |=  bitstring;  // sa1
	else if(type == 2) to_inject ^=  bitstring;  // bit flip
	else 				to_inject =  bitstring; // bizantine

	*output = *( (float*) &to_inject);
	return 0;
} 

/**************** FLOAT32 *************** */
extern "C" __device__ int
inject_float64(
	double* output,
	double input,
	long bitstring, 
	int type ){

	int to_inject = *( (long*) &input );

	if(type == 0)      to_inject &= ~bitstring; // sa0
	else if(type == 1) to_inject |=  bitstring;  // sa1
	else if(type == 2) to_inject ^=  bitstring;  // bit flip
	else 				to_inject =  bitstring; // bizantine

	*output = *( (double*) &to_inject);
	return 0;
}
