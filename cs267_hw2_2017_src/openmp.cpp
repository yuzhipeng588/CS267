#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "common.h"
#include "omp.h"
#include <vector>
#include <algorithm>
//
//  benchmarking program
//
void traverse_vec(std::vector<int>* , int , int , int , particle_t* ,double *,double *,int *,int);
int main( int argc, char **argv )
{   
    int navg,nabsavg=0,numthreads; 
    double dmin, absmin=1.0,davg,absavg=0.0;
	
    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        printf( "-s <filename> to specify a summary file name\n" ); 
        printf( "-no turns off all correctness checks and particle output\n");   
        return 0;
    }

    int n = read_int( argc, argv, "-n", 1000 );
    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );

    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname ? fopen ( sumname, "a" ) : NULL;      

    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    set_size( n );
    init_particles( n, particles );
   
    int length =(int)ceil(sqrt(n*0.0005));
    int num = (int)ceil(sqrt(5*n));
    std::vector<int>* vectors=new std::vector<int>[num*num];
    for( int i=0; i < n; i++ ){
                vectors[(int)(particles[i].y/length*num)+(int)(particles[i].x/length)].push_back(i);
    }
    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );
   // #pragma omp parallel shared(vectors,length,num) private(dmin) 
   // {
    numthreads = omp_get_num_threads();
    for( int step = 0; step < NSTEPS; step++ )
    {
        navg = 0;
        davg = 0.0;
	dmin = 1.0;
        //
        //  compute all forces
        //
	#pragma omp parallel shared(vectors,length,num) private(dmin) 
	{
        #pragma omp for reduction (+:navg) reduction(+:davg)
        for(int i = 0; i < n; i++ ) {
            int p_x = (int)(particles[i].x/length*num);
            int p_y = (int)(particles[i].y/length*num);
            particles[i].ax = particles[i].ay = 0;
//	    opm_set_lock (lock l0);
            traverse_vec(vectors,p_x-1,p_y-1,num,particles,&dmin,&davg,&navg,i);
            traverse_vec(vectors,p_x-1,p_y,num,particles,&dmin,&davg,&navg,i);
            traverse_vec(vectors,p_x-1,p_y+1,num,particles,&dmin,&davg,&navg,i);
            traverse_vec(vectors,p_x,p_y-1,num,particles,&dmin,&davg,&navg,i);
            traverse_vec(vectors,p_x,p_y,num,particles,&dmin,&davg,&navg,i);
            traverse_vec(vectors,p_x,p_y+1,num,particles,&dmin,&davg,&navg,i);
            traverse_vec(vectors,p_x+1,p_y-1,num,particles,&dmin,&davg,&navg,i);
            traverse_vec(vectors,p_x+1,p_y,num,particles,&dmin,&davg,&navg,i);
            traverse_vec(vectors,p_x+1,p_y+1,num,particles,&dmin,&davg,&navg,i);
//	    opm_unset_lock (lock l0);
        }
	
        //
        //  move particles
        //
       // delete[] vectors;
       // vectors = new std::vector<int>[num*num];
        #pragma omp for
        for( int i = 0; i < n; i++ ){
	    int num_subset_old = (int)(particles[i].y/length*num)+(int)(particles[i].x/length);
            move( particles[i] );
	    int num_subset_new = (int)(particles[i].y/length*num)+(int)(particles[i].x/length);
	    if(num_subset_old!=num_subset_new){
 //		 opm_set_lock (lock l1);
		 std::vector<int>::iterator position = std::find(vectors[num_subset_old].begin(),vectors[num_subset_old].end(),i);
		 if(position != vectors[num_subset_old].end()){
			 vectors[num_subset_old].erase(position);
           		 vectors[num_subset_new].push_back(i);
		 }
//		 opm_unset_lock (lock l1);
	    }
        }  

        if( find_option( argc, argv, "-no" ) == -1 ) 
        {
          //
          //  compute statistical data
          //
          #pragma omp master
          if (navg) { 
            absavg += davg/navg;
            nabsavg++;
          }

          #pragma omp critical
	  if (dmin < absmin) absmin = dmin; 
		
          //
          //  save if necessary
          //
          #pragma omp master
          if( fsave && (step%SAVEFREQ) == 0 )
              save( fsave, n, particles );
        }
        }
    }
    simulation_time = read_timer( ) - simulation_time;
    printf( "n = %d,threads = %d, simulation time = %g seconds", n,numthreads, simulation_time);

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
        fprintf(fsum,"%d %d %g\n",n,numthreads,simulation_time);

    //
    // Clearing space
    //
    if( fsum )
        fclose( fsum );
    delete[] vectors;
    free( particles );
    if( fsave )
        fclose( fsave );
    
    return 0;
}
void traverse_vec(std::vector<int>* vectors, int p_x, int p_y, int num, particle_t* particles,double *dmin,double* davg, int* navg,int i){
	    if(p_x>=0&&p_x<num&&p_y>=0&&p_y<num){
		if(vectors[p_y*num+p_x].size()==0)return;
//		opm_set_lock(lock l);
                for(std::vector<int>::iterator it = vectors[p_y*num+p_x].begin(); it != vectors[p_y*num+p_x].end(); ++it) {
                         int part_ = *it;
                         apply_force( particles[i], particles[part_],dmin,davg,navg);
                }
//		opm_unset_lock(lock l);
            }
}
