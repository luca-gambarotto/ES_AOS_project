/* 			Embedded Systems - a.y. 2019/2020	           *
 * 		Luca Gambarotto - mat. 928094 - c.p. 10502632		   *
 
 * The present application is a porting for mangolibs of the needle-wunsch *
 * kernel for opencl taken from the rodinia benchmark suite.   		   *
 * All the modifications done to run the application using mangolibs are   *
 * highlighted in the following code using the FIXME annotation            *
 * For more information about the original version of the software visit   *
 * http://rodinia.cs.virginia.edu/doku.php?id=needleman-wunsch             */


#include "dev/mango_hn.h"
#include "dev/debug.h"
#include <stdlib.h>

/* Support function, not declared with mango_kernel pragma and therefore  *
 * not recognized as a kernel function					  */

int maximum( int a,
		 int b,
		 int c){
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


/* FIXME: in mango all the kernel functions are marked with the following *
 * pragma. */
#pragma mango_kernel
void kernel_function(int *reference, int *input_itemsets, int *output_itemsets, int cols, int penalty, mango_event_t e) {


	/* Create a score array to store the traceback matrix *
	 * during the computation phase			      */

	int* score = malloc(cols*cols*sizeof(int));	

	for (int r=0;r<cols;r++) {
		for (int c=0;c<cols;c++) {
			score[r * cols + c] = input_itemsets[r*cols+c];
		}
	}	
	
	/* Fill the upper left triangular part of the score matrix according to the NW algorithm */
 	for(int m=0; m<(cols-1); m++){
		for(int tx=0; tx<=m; tx++){
			int x = tx + 1;
			int y = m - tx + 1;
			score[y * cols + x] = maximum(
				score[(y-1) * cols + (x-1)] + reference[(y) * cols + (x)],
				score[y*cols + (x-1)] - penalty,
				score[(y-1)*cols + x] - penalty);
		}	
	}


	/* Fill the lower right triangular part of the score matrix according to the NW algorithm */	
	for(int m=(cols-3); m>=0; m--){
		for(int tx=0; tx<=m; tx++){
			int x = tx + cols - 1 - m;
			int y = cols - 1 - tx;
			score[y * cols + x] = maximum(
				score[(y-1) * cols + (x-1)] + reference[(y-1) * cols + (x-1)],
				score[y*cols + (x-1)] - penalty,
				score[(y-1)*cols + x] - penalty); 1;
		}	
	}

	/* Copy back the results of the computation in the output_itemsets buffer */
	for (int r=0;r<cols;r++) {
		for (int c=0;c<cols;c++) {
			output_itemsets[r * cols + c] = score[r*cols+c];
		}
	}

	free(score);

	/* FIXME: add whis function to synchronize the write back of the buffer */
	mango_write_synchronization(&e, 1);

	return;
}


