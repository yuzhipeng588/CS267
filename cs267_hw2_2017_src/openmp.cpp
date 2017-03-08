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
    omp_lock_t* lock=new omp_lock_t[num*num];
    #pragma omp for schedule(dynamic) nowait
    for(int i=0;i<num*num;i++)
        omp_init_lock(&(lock[i]));
    #pragma omp for schedule(dynamic)
    for( int i=0; i < n; i++ ){
                vectors[(int)(particles[i].y/length*num)*num+(int)(particles[i].x/length*num)].push_back(i);
    }
    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );
//    #pragma omp parallel shared(particles,vectors,lock,absmin) private(dmin) 
//    {
    numthreads = omp_get_num_threads();
//    printf("%d",NSTEPS);
    for( int step = 0; step < NSTEPS; step++ )
    {
        navg = 0;
        davg = 0.0;
	dmin = 1.0;
        //
        //  compute all forces
        //
    #pragma omp parallel shared(particles,vectors,lock,absmin) private(dmin) 
    {

        #pragma omp for schedule(dynamic) reduction (+:navg) reduction(+:davg)
        for(int i = 0; i < n; i++ ) {
            int p_x = (int)(particles[i].x/length*num);
            int p_y = (int)(particles[i].y/length*num);
            particles[i].ax = particles[i].ay = 0;
            traverse_vec(vectors,p_x-1,p_y-1,num,particles,&dmin,&davg,&navg,i);
 	    traverse_vec(vectors,p_x-1,p_y,num,particles,&dmin,&davg,&navg,i);
	    traverse_vec(vectors,p_x-1,p_y+1,num,particles,&dmin,&davg,&navg,i);
	    traverse_vec(vectors,p_x,p_y-1,num,particles,&dmin,&davg,&navg,i);
	    traverse_vec(vectors,p_x,p_y,num,particles,&dmin,&davg,&navg,i);
	    traverse_vec(vectors,p_x,p_y+1,num,particles,&dmin,&davg,&navg,i);
	    traverse_vec(vectors,p_x+1,p_y-1,num,particles,&dmin,&davg,&navg,i);
	    traverse_vec(vectors,p_x+1,p_y,num,particles,&dmin,&davg,&navg,i);
	    traverse_vec(vectors,p_x+1,p_y+1,num,particles,&dmin,&davg,&navg,i);
	}

/*
        #pragma omp for schedule(dynamic) reduction (+:navg) reduction(+:davg)
        for(int i = 0; i < n; i++ ) {
	    #pragma omp parallel
	    {
            int p_x = (int)(particles[i].x/length*num);
            int p_y = (int)(particles[i].y/length*num);
            particles[i].ax = particles[i].ay = 0;
	    
	    for(int j=p_x-1;j<=p_x+1;j++){
		for(int k=p_y-1;k<=p_y+1;k++)
                    traverse_vec(vectors,j,k,num,particles,&dmin,&davg,&navg,i);
            }
	    }
        }
*/	
        //
        //  move particles
        //

        #pragma omp for schedule(dynamic) nowait 
        for( int i = 0; i < n; i++ ){
	    int num_subset_old = (int)(particles[i].y/length*num)*num+(int)(particles[i].x/length*num);
            move( particles[i] );
	    int num_subset_new = (int)(particles[i].y/length*num)*num+(int)(particles[i].x/length*num);
//	    #pragma omp critical
	    if(num_subset_old!=num_subset_new){
		 omp_set_lock (&(lock[num_subset_old]));
		 std::vector<int>::iterator position = std::find(vectors[num_subset_old].begin(),vectors[num_subset_old].end(),i);
		 vectors[num_subset_old].erase(position);
		 omp_unset_lock(&(lock[num_subset_old]));

		 omp_set_lock (&(lock[num_subset_new]));
           	 vectors[num_subset_new].push_back(i);
		 omp_unset_lock(&(lock[num_subset_new]));
		 
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
	  if (dmin < absmin&&dmin!=0) absmin = dmin; 
		
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
    delete[] lock;
    delete[] vectors;
    free( particles );
    if( fsave )
        fclose( fsave );
    
    return 0;
}
void traverse_vec(std::vector<int>* vectors, int p_x, int p_y, int num, particle_t* particles,double *dmin,double* davg, int* navg,int i){
	    if(p_x>=0&&p_x<num&&p_y>=0&&p_y<num){
                for(std::vector<int>::iterator it = vectors[p_y*num+p_x].begin(); it != vectors[p_y*num+p_x].end(); ++it) {
                         int part_ = *it;
                         apply_force( particles[i], particles[part_],dmin,davg,navg);
                }
            }
}
