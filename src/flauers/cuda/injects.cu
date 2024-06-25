// TODO: include stdlib.h to provide explicit definitions of int32_t and others
//#include <cuda/std/cstdlib>

extern "C" __device__ int
inject_int8(
	int8_t* output,
	int8_t input,
	int8_t bitstring, 
	int8_t type ){

	if(type == 0)      *output &= ~bitstring; // sa0
	else if(type == 1) *output |=  bistring;  // sa1
	else if(type == 2) *output ^=  bistring;  // bit flip
	else 			   *output  =  bistring; // bizantine

	return 0;
} 

extern "C" __device__ int
inject_int32(
	int* output,
	int input,
	int bitstring, 
	int type ){

	if(type == 0)      *output &= ~bitstring; // sa0
	else if(type == 1) *output |=  bistring;  // sa1
	else if(type == 2) *output ^=  bistring;  // bit flip
	else 			   *output  =  bistring; // bizantine

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
	else if(type == 1) to_inject |=  bistring;  // sa1
	else if(type == 2) to_inject ^=  bistring;  // bit flip
	else 				to_inject =  bistring; // bizantine

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
	else if(type == 1) to_inject |=  bistring;  // sa1
	else if(type == 2) to_inject ^=  bistring;  // bit flip
	else 				to_inject =  bistring; // bizantine

	*output = *( (double*) &to_inject);
	return 0;
}