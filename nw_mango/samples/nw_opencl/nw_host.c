/* 			Embedded Systems - a.y. 2019/2020	           *
 * 		Luca Gambarotto - mat. 928094 - c.p. 10502632		   *
 
 * The present application is a porting for mangolibs of the needle-wunsch *
 * algorithm implementation for opencl taken from the rodinia benchmark    *
 * suite.								   *
 * All the modifications done to run the application using mangolibs are   *
 * highlighted in the following code using the FIXME annotation            *
 * For more information about the original version of the software visit   *
 * http://rodinia.cs.virginia.edu/doku.php?id=needleman-wunsch             *

/* This application implements the Needleman-Wunsch algorithm, used in     *
 * bioinformatics to search for similitudies in peptide sequences. This is *
 * particular useful to develop vaccines against viruses or to study       *
 * autoimmunity phenomena.                                                 *
 * The main idea behind this algorithm is to divide the long sequences to  *
 * be studied into a series of smaller problems, each one individually     *
 * computable, and then combine the sub-result to obtain the full result   *
 * Additional details about this algorithm can be found at the following   *
 * link: https://www.cs.sjsu.edu/~aid/cs152/NeedlemanWunsch.pdf            */



/* FIXME: removed BLOCK_SIZE definition, the ported sample is able to      *
 * adapt its execution according to the sequence length. Despite this fact,*
 * to keep the new implementation adherent to the original code the        *
 * accepted sequences are supposed to be of a length multiple of 16        */

#define LIMIT -999

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <sys/time.h>


/* FIXME: These library headers have been added in substitution of        *
 * standard OpenCL headers to make the application call the reimplemented *
 * OCl functions provided by mango                                        */
#include "CL/cl_mango.h"
#include "CL/cl_types.h"


/* blosum62 is the substitution matrix chosen for this implementation of  *
 * the algorithm. To adapt the sample to a different execution matrix it  *
 * is sufficient to change the following lines of code                    */
int blosum62[24][24] = {
{ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4},
{-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4},
{-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4},
{-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
{-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4},
{-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4},
{-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4},
{-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4},
{-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4},
{-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4},
{-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4},
{-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4},
{-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
{ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4},
{ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4},
{-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4},
{-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4},
{ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4},
{-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4},
{-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1}
};

static cl_context	    context;
static cl_command_queue cmd_queue;
static cl_device_type   device_type;
// FIXME: added a deviceId for mango
static cl_device_id		device_id;
static cl_int           num_devices;
static int err;

// FIXME: mango data structure;
struct cl_mango_data mango_data;


/* FIXME: this function has been completely rewritten to follow mango     *
 * initialization process. The aim of this part of the code is to         *
 * populate the context and the cmq_que parameters accordingly to the     *
 * operations of mango.						          *
 * This function can be taken as it is also for future implementations    *
 * of applications running on mango.                                      */
static int initialize(int use_gpu)
{
	printf("Initializing OpenCL device...");
	cl_uint dev_cnt = 0;
	// This function save in the dev_cnt variable the number of
	// platforms found on the system
	clGetPlatformIDs(0, 0, &dev_cnt);
	if(dev_cnt == 0){
		printf("Error: no OpenCL platform found");
		return -1;
	}
	printf("OpenCL platforms found: n=%d\n", dev_cnt);
	
	cl_platform_id platform_ids[100];
	
	// Here all the ids of the dev_cnt platforms found previously
	// found on the systems are stored in the platform_ids array
	clGetPlatformIDs(dev_cnt, platform_ids, NULL);
	for(unsigned int id = 0; id < dev_cnt; ++id){
		printf("OpenCL platform %d: %s\n", id, platform_ids[id]->name);
	}
	
	// The device id selected from the first available platform found
	// by mango is stored in the device_id variable
	err = clGetDeviceIDs(platform_ids[0], use_gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
	if(err != CL_SUCCESS){
		printf("Error: failed to create a device group!\n");
		return -1;
	}
	
	// Initialization of the data structures required by mango
	mango_data.recipe = "test_manga";
	mango_data.application_name = "nw_sample";
	
	// Creation of the context using the dedive previously selected
	context = clCreateContext(0, 1, &device_id, NULL, &mango_data, &err);	
	if(!context){
		printf("Error: failed to create a compute context!\n");
		return -1;
	}
	
	// Creation of the related command queue to send commands to the device
	cmd_queue = clCreateCommandQueue(context, device_id, 0, &err);
	if(!cmd_queue){
		printf("Error: failed to create a command queue!\n");
		return -1;
	}

	return 0;
}

/* FIXME: variables that were not significant for mango have been removed      *
 * This function can be easily reused for different mangolib sw implementations*/
static int shutdown()
{
	/*If not yet released, all the resources are released*/
	if( cmd_queue ) clReleaseCommandQueue( cmd_queue );
	if( context ) clReleaseContext( context );

	// reset all variables
	cmd_queue = 0;
	context = 0;
	num_devices = 0;
	device_type = 0;

	return 0;
}

/* This function returns the biggest of the 3 numbers passed as arguments     */	
int maximum( int a, int b, int c)
{

	int k;
	if( a <= b )
	  k = b;
	else 
	  k = a;
	if( k <=c )
	  return(c);
	else
	  return(k);
}

/* In case of non-compliant parameters in the argv array, this function shows *
 * on screen the correct format for the parameters.			      */
void usage(int argc, char **argv)
{
	fprintf(stderr, "Usage: %s <max_rows/max_cols> <penalty> \n", argv[0]);
	fprintf(stderr, "\t<dimension>  - x and y dimensions\n");
	fprintf(stderr, "\t<penalty> - penalty(positive integer)\n");
	exit(1);
}


double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

int main(int argc, char **argv){

    int max_rows, max_cols, penalty;

	if (argc == 3)
	{
		max_rows = atoi(argv[1]);
		max_cols = atoi(argv[1]);
		penalty = atoi(argv[2]);
	}
    else{
	     usage(argc, argv);
    }
	
	/* For comparability reasons this limitation has been kept also *
	 * in the mango version of the sample, but theoretically it     *
	 * should be able to manage sequence of random lenght           */
	if(atoi(argv[1])%16!=0){
	fprintf(stderr,"The dimension values must be a multiple of 16\n");
	exit(1);
	}

	max_rows = max_rows + 1;
	max_cols = max_cols + 1;
	
	
	/*Initialize the three matrices needed:			       *
	 * - reference: keep the scoring of each pairs of elements of  *
	 *   the two sequences. These scoring are computed using the   *
	 *   substitution matrix (in this case the blosum62 one)       *
	 * - input_itemsets: keep the two sequences generated in a 
	 *   random way. After the reference matrix is completely      *
	 *   filled, the content of this matrix cleared to prepare for *
	 *   the computation of the traceback matrix		       *
	 * - output_itemsets: this matrix will contain the traceback   *
	 *   matrix, computed using the external kernel		       */
	int *reference;
	int *input_itemsets;
	int *output_itemsets;
	
	reference = (int *)malloc( max_rows * max_cols * sizeof(int) );
    	input_itemsets = (int *)malloc( max_rows * max_cols * sizeof(int) );
	output_itemsets = (int *)malloc( max_rows * max_cols * sizeof(int) );
	
	srand(time(NULL));
	
	/* input_itemset initialization (the two sequences are randomly *
	 * generated with values from 0 to 10				*/
	for (int i = 0 ; i < max_cols; i++){
		for (int j = 0 ; j < max_rows; j++){
			input_itemsets[i*max_cols+j] = 0;
		}
	}

	for( int i=1; i< max_rows ; i++){    //initialize the cols
			input_itemsets[i*max_cols] = rand() % 10 + 1;
	}
	
   	for( int j=1; j< max_cols ; j++){    //initialize the rows
			input_itemsets[j] = rand() % 10 + 1;
	}
	
	/* refence is filled with the scorings taken from the blusum62 matrix */
	for (int i = 1 ; i < max_cols; i++){
		for (int j = 1 ; j < max_rows; j++){
		reference[i*max_cols+j] = blosum62[input_itemsets[i*max_cols]][input_itemsets[j]];
		}
	}
	
	/* input_itemsets is prepared for the traceback computation */
    	for( int i = 1; i< max_rows ; i++)
       		input_itemsets[i*max_cols] = -i * penalty;
	for( int j = 1; j< max_cols ; j++)
       		input_itemsets[j] = -j * penalty;



	int use_gpu = 1;
	// OpenCL initialization
	if(initialize(use_gpu)) return -1;
	
	/* FIXME: the path string contains the path to the pre-compiled *
	 * kernel to be executed					*/
	char* path = "/opt/mango/usr/local/share/nw_kernel/nw_opencl_dev";
	/* FIXME: the mango opencl program is created as a "normal"     *
	 * opencl program, loading it from the binary specified in the  *
	 * path string							*/
	cl_program program;
	program = clCreateProgramWithBinary(context, 1, &device_id, NULL, (const unsigned char **) & path, NULL, &err);
	if(!program){
		printf("ERROR: Failed to create compute program!\n");
		return -1;
	}
	/* the program is built  and the kernel is created as in any    *
         * other opencl program						*/
	err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	if(err != CL_SUCCESS){
		size_t len;
		char buffer[2048];
		printf("ERROR: failed to build program executable!\n");
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		printf("%s\n", buffer);
		exit(1);
	}

	cl_kernel kernel;
	kernel = clCreateKernel(program, "kernel", &err);
	if(!kernel || err != CL_SUCCESS){
		printf("ERROR: failed to create compute kernel!\n");
		exit(1);
	}
		
	
	/* Three buffers are created to pass the input_itemsets, the reference and the *
	 * output_itemsets matrices to the kernel executor 			       */
	cl_mem input_itemsets_d;
	cl_mem output_itemsets_d;
	cl_mem reference_d;
	
	input_itemsets_d = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, max_cols * max_rows * sizeof(int), input_itemsets, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer input_item_set (size:%d) => %d\n", max_cols * max_rows, err); return -1;}
	reference_d		 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, max_cols * max_rows * sizeof(int), reference, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer reference (size:%d) => %d\n", max_cols * max_rows, err); return -1;}
	output_itemsets_d = clCreateBuffer(context, CL_MEM_READ_WRITE, max_cols * max_rows * sizeof(int), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer output_item_set (size:%d) => %d\n", max_cols * max_rows, err); return -1;}
	
	if(!input_itemsets_d || !reference_d || !output_itemsets_d){
		printf("ERROR: failed to allocate buffers!\n");
		exit(1);
	}

	/* FIXME: unlike standard opencl implementations, input and output buffers must *
	 * also be stored in 2 arrays (one for the input and one for the output buf.)   *
	 * to be passed to the clEnqueueNDRangeKernel. In this specific case the two 	*
	 * input buffers are the input-itemsets and the reference buffers, while the    *
	 * output one, supposed to store the traceback matrix computed by the kernel,   *
	 * is the output_itemsets one						        */
	cl_mem in_buffers[2], out_buffers[1];
	in_buffers[0] = input_itemsets_d;
	in_buffers[1] = reference_d;
	out_buffers[0] = output_itemsets_d;

	/* Kernel parameters setup */
	int cols = max_cols;
	int p = penalty;
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&reference_d);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&input_itemsets_d);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&output_itemsets_d);
	err |= clSetKernelArg(kernel, 3, sizeof(int), (void*)&cols);
	err |= clSetKernelArg(kernel, 4, sizeof(int), (void*)&p);

	if(err!=CL_SUCCESS){
		printf("ERROR: Failed to set kernel arguments!\n");
		exit(1);
	}

	/* FIXME: the clEnqueueNDRangeKernel have a different sintax w.r.t standard  *
	 * implementations of opencl. The two arrays containing the input and output *
	 * buffers must be passed to this function as arguments			     */
	err = clEnqueueNDRangeKernel(cmd_queue, kernel,
					reinterpret_cast<cl_mem *>(&in_buffers), 2,
					reinterpret_cast<cl_mem *>(&out_buffers), 1,
					0, NULL, NULL);

	if(err != CL_SUCCESS){
		printf("ERROR: Failed to execute kernel!\n");
		exit(1);
	}

	/* Enqueue the write-back of the content of the output_itemsets buffer in the *
	 * matrix on the host, as like as in standard implementations of opencl	      */
	err = clEnqueueReadBuffer(cmd_queue, output_itemsets_d, CL_TRUE, 0, max_cols * max_rows * sizeof(int), output_itemsets, 0, NULL, NULL);

	if(err != CL_SUCCESS){
		printf("ERROR: Failed to read traceback matrix from device memory!\n");
		exit(1);
	}

	/* FIXME: Instead of using the clFinish function, in mango implementation we  *
	 * only need to use clStartComputation to actually start executing the        *
	 * commands we enqueued above in the cmd_queue. clFinish is not supposed to   *
	 * be called explicitely in the host code				      */
	clStartComputation(cmd_queue);


	/* Only to check the correctness of the execution, the sample will write out  *
	 * the result (the three matrices and the actual traceback) in a file called  *
	 * "results.txt, in the same folder of the sample binary		      */
	FILE *fpo = fopen("result.txt","w");
	fprintf(fpo, "print traceback value GPU:\n\n");
	
	fprintf(fpo, "input_itemsets matrix:\n");
	for(int i = 0; i<cols; i++){
		for(int q = 0; q<cols; q++){
			fprintf(fpo, "%4d ", input_itemsets[i * cols + q]);		
		}
		fprintf(fpo, "\n\n");	
	}

	fprintf(fpo, "reference matrix:\n");
	for(int i = 0; i<cols; i++){
		for(int q = 0; q<cols; q++){
			fprintf(fpo, "%4d ", reference[i * cols + q]);		
		}
		fprintf(fpo, "\n\n");	
	}
    	
	fprintf(fpo, "output_itemsets matrix:\n");
	for(int i = 0; i<cols; i++){
		for(int q = 0; q<cols; q++){
			fprintf(fpo, "%4d ", output_itemsets[i * cols + q]);		
		}
		fprintf(fpo, "\n\n");	
	}
	
	fprintf(fpo, "Traceback:\n");
	for (int i = max_rows - 2,  j = max_rows - 2; i>=0, j>=0;){
		int nw, n, w, traceback;
		if ( i == max_rows - 2 && j == max_rows - 2 )
			fprintf(fpo, "%d ", output_itemsets[ i * max_cols + j]); //print the first element
		if ( i == 0 && j == 0 )
           break;
		if ( i > 0 && j > 0 ){
			nw = output_itemsets[(i - 1) * max_cols + j - 1];
		    w  = output_itemsets[ i * max_cols + j - 1 ];
            n  = output_itemsets[(i - 1) * max_cols + j];
		}
		else if ( i == 0 ){
		    nw = n = LIMIT;
		    w  = output_itemsets[ i * max_cols + j - 1 ];
		}
		else if ( j == 0 ){
		    nw = w = LIMIT;
            n  = output_itemsets[(i - 1) * max_cols + j];
		}
		else{
		}

		int new_nw, new_w, new_n;
		new_nw = nw + reference[i * max_cols + j];
		new_w = w - penalty;
		new_n = n - penalty;
		
		traceback = maximum(new_nw, new_w, new_n);
		if(traceback == new_nw)
			traceback = nw;
		if(traceback == new_w)
			traceback = w;
		if(traceback == new_n)
            traceback = n;
			
		fprintf(fpo, "%d ", traceback);

		if(traceback == nw )
		{i--; j--; continue;}

        else if(traceback == w )
		{j--; continue;}

        else if(traceback == n )
		{i--; continue;}

		else
		;
	}
	
	fclose(fpo);


	printf("*********************************\n** NW_OPENCL: Computation Done **\n*********************************\n");
	
	// OpenCL shutdown
	if(shutdown()) return -1;

	/* Memory buffers are released as in std opencl implementations */
	clReleaseMemObject(input_itemsets_d);
	clReleaseMemObject(output_itemsets_d);
	clReleaseMemObject(reference_d);

	free(reference);
	free(input_itemsets);
	free(output_itemsets);
	
}

