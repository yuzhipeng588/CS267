#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <vector>
#include "common.h"
//
//  benchmarking program
//
void traverse_vec(std::vector<int>* , int , int , int , particle_t* ,double *,double *,int *,int);
int main( int argc, char **argv ){    
    int navg,nabsavg=0;
    double davg,dmin, absmin=1.0, absavg=0.0;

    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        printf( "-s <filename> to specify a summary file name\n" );
        printf( "-no turns off all correctness checks and particle output\n");
        return 0;
    }
    
    int n = read_int( argc, argv, "-n", 10000 );

    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );
    
    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname ? fopen ( sumname, "a" ) : NULL;
    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    set_size( n );
    init_particles( n, particles );
    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );
    int length =(int)ceil(sqrt(n*0.0005));
    int num = (int)ceil(sqrt(5*n));
    std::vector<int>* vectors=new std::vector<int>[num*num];
    for( int i=0; i < n; i++ ){
                vectors[(int)(particles[i].y/length*num)+(int)(particles[i].x/length)].push_back(i);
    }	
    for( int step = 0; step < NSTEPS; step++ )
    {
	navg = 0;
        davg = 0.0;
	dmin = 1.0;
        //
        //  compute forces
        //
        for( int i = 0; i < n; i++ )
        {
            int p_x = (int)(particles[i].x/length*num);
	    int p_y = (int)(particles[i].y/length*num);
            particles[i].ax = particles[i].ay = 0;
	    traverse_vec(vectors,p_x-1,p_y-1,length,particles,&dmin,&davg,&navg,i);
	    traverse_vec(vectors,p_x-1,p_y,length,particles,&dmin,&davg,&navg,i);
	    traverse_vec(vectors,p_x-1,p_y+1,length,particles,&dmin,&davg,&navg,i);
	    traverse_vec(vectors,p_x,p_y-1,length,particles,&dmin,&davg,&navg,i);
	    traverse_vec(vectors,p_x,p_y,length,particles,&dmin,&davg,&navg,i);
	    traverse_vec(vectors,p_x,p_y+1,length,particles,&dmin,&davg,&navg,i);
   	    traverse_vec(vectors,p_x+1,p_y-1,length,particles,&dmin,&davg,&navg,i);
	    traverse_vec(vectors,p_x+1,p_y,length,particles,&dmin,&davg,&navg,i);
	    traverse_vec(vectors,p_x+1,p_y+1,length,particles,&dmin,&davg,&navg,i);	
	    if(dmin<0.4)
		printf("x = %d, y = %d ",particles[i].x,particles[i].y);
	}
        //
        //  move particles
        //
	delete[] vectors;
        vectors = new std::vector<int>[num*num];
        for( int i = 0; i < n; i++ ){
            move( particles[i] );
            vectors[(int)(particles[i].y/length*num)+(int)(particles[i].x/length)].push_back(i);
        }
        if( find_option( argc, argv, "-no" ) == -1 )
        {
          //
          // Computing statistical data
          //
          if (navg) {
            absavg +=  davg/navg;
            nabsavg++;
          }
          if (dmin < absmin) absmin = dmin;
		
          //
          //  save if necessary
          //
          if( fsave && (step%SAVEFREQ) == 0 )
              save( fsave, n, particles );
        }
    }
    simulation_time = read_timer( ) - simulation_time;
    
    printf( "n = %d, simulation time = %g seconds", n, simulation_time);

    if( find_option( argc, argv, "-no" ) == -1 )
    {
      if (nabsavg) absavg /= nabsavg;
    // 
    //  -The minimum distance absmin between 2 particles during the run of the simulation
    //  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
    //  -A simulation where particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
    //
    //  -The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
    //
    printf( ", absmin = %lf, absavg = %lf", absmin, absavg);
    if (absmin < 0.4) printf ("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
    if (absavg < 0.8) printf ("\nThe average distance is below 0.8 meaning that most particles are not interacting");
    }
    printf("\n");     

    //
    // Printing summary data
    //
    if( fsum) 
        fprintf(fsum,"%d %g\n",n,simulation_time);
 
    //
    // Clearing space
    //
    if( fsum )
        fclose( fsum );    
    free( particles );
    if( fsave )
        fclose( fsave );
    
    return 0;
}
void traverse_vec(std::vector<int>* vectors, int p_x, int p_y, int length, particle_t* particles,double *dmin,double* davg, int* navg,int i){
if(p_x>=0&&p_x<length&&p_y>=0&&p_y<length){
                for(std::vector<int>::iterator it = vectors[p_y*length+p_x].begin(); it != vectors[p_y*length+p_x].end(); ++it) {
                         int part_ = *it;
			 apply_force( particles[i], particles[part_],dmin,davg,navg);
                }
            }
}
