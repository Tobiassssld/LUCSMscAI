// include standard headers
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <limits.h>

// TODO: YOUR "branched" in-place sort implementation
// NOTE: Depending on your choice of sort algorithm / implementation,
//       you only need to use/implement *either* this "in-place"
//       *or* the alternative "out-of-place" variant, but *not* both.
void sort_branched_inplace(int N, int *data_array) {
    for (int i = 0; i < N - 1; i++) {
        for (int j = 0; j < N - i - 1; j++) {
            if (data_array[j] > data_array[j + 1]) {  // 条件分支
                int temp = data_array[j];
                data_array[j] = data_array[j + 1];
                data_array[j + 1] = temp;
            }
        }
    }
}


// The remaining code can be used as-is; i.e., no need for any changes.

// main program
int main( int argc , char ** argv ) {

	// variables for timing
	struct timeval before , after;
	long long int usec = 0;

	// check number command line arguments
	if ( argc != 3 ) {
		// incorrect number of command line arguments
		fprintf(
		    stderr ,
		    "need two arguments\n"
		);
		// exit with error status
		return( 1 );
	}

	// read first command line argument:
	// number of data values
	int N = atoi( argv[ 1 ] );
	// sanity check: echo argument value
	fprintf(
		stderr ,
		"N = %d\n" ,
		N
	);

	// read second command line argument:
	// (absolute/relative) path to input data file
	char *input_data_file_location = strdup( argv[ 2 ] );
	// sanity check: echo argument value
	fprintf(
		stderr ,
		"input_data_file_location = '%s'\n" ,
		input_data_file_location
	);

	// allocate memory for data array
	int *data_array = malloc( N * sizeof(int) );
	if ( data_array == NULL ) {
		// report error
		fprintf(
			stderr ,
			"malloc failed for data_array (%zu bytes)\n" ,
			N * sizeof(int)
		);
		// free allocated memory
		free( input_data_file_location );
		// exit with error status
		return( 1 );
	}

	// open input data file for reading
	FILE *input_data_file = fopen( input_data_file_location , "r" );
	if ( input_data_file == NULL ) {
		// report error
		perror( "fopen failed" );
		// free allocated memory
		free( data_array );
		free( input_data_file_location );
		// exit with error status
		return( 1 );
	}

	// load content of input data file in to data array;
	// measure and report time it takes
	gettimeofday( &before , NULL );
	for ( int i = 0 ; i < N ; i++ ) {
		int e = fscanf(
			input_data_file ,
			"%d\n" ,
			data_array + i
		);
		if ( e == EOF ) {
			// report error
			perror( "fscanf: premature EOF" );
			// close input data file
			fclose( input_data_file );
			// free allocated memory
			free( data_array );
			free( input_data_file_location );
			// exit with error status
			return( 1 );
		}
		if ( e < 1 ) {
			// report error
			perror( "fscanf failed" );
			// close input data file
			fclose( input_data_file );
			// free allocated memory
			free( data_array );
			free( input_data_file_location );
			// exit with error status
			return( 1 );
		}
	}
	gettimeofday( &after , NULL );
	usec = (long long int)(after.tv_sec - before.tv_sec) * 1000000
			   + (after.tv_usec - before.tv_usec);
	fprintf(
		stderr ,
		"load data: %9lld usec ; data_array[ 0 ] = %10d , data_array[ %d ] = %10d , data_array[ %d ] = %10d\n" ,
		usec , data_array[ 0 ] , N / 2 , data_array[ N / 2 ] , N - 1 , data_array[ N - 1 ]
	);

	// close input data file
	fclose( input_data_file );

	// sort data array in-place;
	// measure the time it takes
	gettimeofday( &before , NULL );
	sort_branched_inplace( N , data_array );
	gettimeofday( &after , NULL );

	// Check whether result is indeed sorted;
	// *required* but *not sufficient* criteria for correctness!
	for ( int i = 1 ; i < N ; i++ ) {
		if ( data_array[ i - 1 ] > data_array[ i ] ) {
			// report error
			fprintf(
				stderr ,
				"result not sorted:\n data_array[ %d ] == %10d\n data_array[ %d ] == %10d\n" ,
				i - 1 , data_array[ i - 1 ] , i , data_array[ i ]
			);
			// free allocated memory
			free( data_array );
			free( input_data_file_location );
			// exit with error status
			return( 1 );
		}
	}

	// calculate and report sorting time
	usec = (long long int)(after.tv_sec - before.tv_sec) * 1000000
			   + (after.tv_usec - before.tv_usec);
	fprintf(
		stderr ,
		"sort data: %9lld usec ; data_array[ 0 ] = %10d , data_array[ %d ] = %10d , data_array[ %d ] = %10d\n" ,
		usec , data_array[ 0 ] , N / 2 , data_array[ N / 2 ] , N - 1 , data_array[ N - 1 ]
	);

	// output sorted data array to console (stdout)
	// (one value per line)
	for ( int i = 0 ; i < N ; i++ ) {
		printf(
			"%d\n" ,
			data_array[ i ]
		);
	}

	// free allocated memory
	free( data_array );
	free( input_data_file_location );

	// exit without error status
	return( 0 );
}
