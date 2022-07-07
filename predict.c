#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <gsl/gsl_math.h>

#include "allvars.h"
#include "proto.h"

// PM: squared distance between a particle and (cenx, ceny, cenz)
float sink_dr2(int i) {
  float dx = cenx - P[i].Pos[0];
  float dy = ceny - P[i].Pos[1];
  float dz = cenz - P[i].Pos[2];
 
  float boxsize = 125.;
  if (abs(dx) > boxsize / 2.){
    dx = boxsize - dx;
  }

  if (abs(dy) > boxsize / 2.){
    dy = boxsize - dy;
  }

  if (abs(dz) > boxsize / 2.){
    dz = boxsize - dz;
  }  
  return dx*dx + dy*dy + dz*dz;
}

// PM: return the index of the particle with the minimum potential within some
// fixed comoving distance of (cenx, ceny, cenz).
int sink_min_potential_index(int type, float r) {
  int min_idx = -1;
  
  float r2 = r*r;
  
  int i;
  for (i = 0; i < NumPart; i++) {
	if ((P[i].Type != type && P[i].ID != id0) ||
		sink_dr2(i) > r2) continue;
	
	if (min_idx == -1 || P[i].Potential < P[min_idx].Potential) {
	  min_idx = i;
	}
  }
  
  return min_idx;
}

// PM move the sink particle to the position of a nearby
void adjust_sink_position(int i) {
  int j = sink_min_potential_index(0, All.SinkSearchRadius);
  if (j == -1) return;
  
  int dim;
  for (dim = 0; dim < 3; dim++) {
	P[i].Pos[dim] = P[j].Pos[dim];
  }
  
  cenx = P[i].Pos[0];
  ceny = P[i].Pos[1];
  cenz = P[i].Pos[2];
}

// PM computes the mean velcoity around the sink particle
void sink_mean_velocity(int type, float r, float *out) {
  int sum = 0;
  int dim;
  for (dim = 0; dim < 3; dim++) {
	out[dim] = 0;
  }

  float r2 = r*r;

  int i;
  for (i = 0; i < NumPart; i++) {
	if ((P[i].Type != type && P[i].ID != id0) ||
		sink_dr2(i) > r2) continue;
	
	for (dim = 0; dim < 3; dim++) {
	  out[dim] += P[i].Vel[dim];
	}
	sum++;
  }

  for (dim = 0; dim < 3; dim++) {
	out[dim] /= sum;
  }
}

// PM makes the sink particle's velcoity equal to the mean velocity of the
// surrounding particles.
void adjust_sink_velocity(int i) {
  float mean_v[3];

  sink_mean_velocity(0, All.SinkSearchRadius, mean_v);
  
  int dim;
  for (dim = 0; dim < 3; dim++) {
	P[i].Vel[dim] = mean_v[dim];
  }
}


/*! \file predict.c
 *  \brief drift particles by a small time interval
 *
 *  This function contains code to implement a drift operation on all the
 *  particles, which represents one part of the leapfrog integration scheme.
 */


/*! This function drifts all particles from the current time to the future:
 *  time0 - > time1
 *
 *  If there is no explicit tree construction in the following timestep, the
 *  tree nodes are also drifted and updated accordingly. Note: For periodic
 *  boundary conditions, the mapping of coordinates onto the interval
 *  [0,All.BoxSize] is only done before the domain decomposition, or for
 *  outputs to snapshot files.  This simplifies dynamic tree updates, and
 *  allows the domain decomposition to be carried out only every once in a
 *  while.
 */
void move_particles(int time0, int time1)
{
  int i, j, task;
  double dt_drift, dt_gravkick, dt_hydrokick, dt_entr;
  double t0, t1;


  t0 = second();

  if(All.ComovingIntegrationOn)
    {
      dt_drift = get_drift_factor(time0, time1);
      dt_gravkick = get_gravkick_factor(time0, time1);
      dt_hydrokick = get_hydrokick_factor(time0, time1);
    }
  else
    {
      dt_drift = dt_gravkick = dt_hydrokick = (time1 - time0) * All.Timebase_interval;
    }


  int itask=0;
  int otask=0;
  for(i = 0; i < NumPart; i++){
	// DY update disk center
	if(P[i].ID==id0){

	  printf("Update disk center, a,m,h,pos,vphys:  "
			 "%f %f %f   %f %f %f   %f %f %f\n",
			 All.Time,P[i].Mass,All.ForceSoftening[P[i].Type],
			 P[i].Pos[0],P[i].Pos[1],P[i].Pos[2],
			 P[i].Vel[0]/sqrt(All.Time),
			 P[i].Vel[1]/sqrt(All.Time),
			 P[i].Vel[2]/sqrt(All.Time));

	  // PM: if enough time has passed, move the sink particle on top of a
	  // nearby HR particle with the most negative potential.

	  // PM: We can implement this  if things start getting slow.
	  //if (All.Ti_Current > prev_sink_adjustment + SINK_ADJUSTMENT_TIME) {
	  adjust_sink_position(i);
	  adjust_sink_velocity(i);
	  //}

	  itask=ThisTask;
	  cenx=P[i].Pos[0];
	  ceny=P[i].Pos[1];
	  cenz=P[i].Pos[2];
	}
  }
  // Need to call on all tasks! 
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Allreduce(&itask, &otask, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Bcast(&cenx, 1, MPI_FLOAT, otask, MPI_COMM_WORLD);
  MPI_Bcast(&ceny, 1, MPI_FLOAT, otask, MPI_COMM_WORLD);
  MPI_Bcast(&cenz, 1, MPI_FLOAT, otask, MPI_COMM_WORLD);
  //printf("###### check center: thistask %d itaks %d %f %f %f\n",ThisTask,otask,cenx,ceny,cenz);

  for(i = 0; i < NumPart; i++)
    {
      for(j = 0; j < 3; j++)
	P[i].Pos[j] += P[i].Vel[j] * dt_drift;

      if(P[i].Type == 0)
	{
#ifdef PMGRID
	  for(j = 0; j < 3; j++)
	    SphP[i].VelPred[j] +=
	      (P[i].GravAccel[j] + P[i].GravPM[j]) * dt_gravkick + SphP[i].HydroAccel[j] * dt_hydrokick;
#else
	  for(j = 0; j < 3; j++)
	    SphP[i].VelPred[j] += P[i].GravAccel[j] * dt_gravkick + SphP[i].HydroAccel[j] * dt_hydrokick;
#endif
	  SphP[i].Density *= exp(-SphP[i].DivVel * dt_drift);
	  SphP[i].Hsml *= exp(0.333333333333 * SphP[i].DivVel * dt_drift);

	  if(SphP[i].Hsml < All.MinGasHsml)
	    SphP[i].Hsml = All.MinGasHsml;

	  dt_entr = (time1 - (P[i].Ti_begstep + P[i].Ti_endstep) / 2) * All.Timebase_interval;

	  SphP[i].Pressure = (SphP[i].Entropy + SphP[i].DtEntropy * dt_entr) * pow(SphP[i].Density, GAMMA);
	}
    }

  /* if domain-decomp and tree are not going to be reconstructed, update dynamically.  */
  if(All.NumForcesSinceLastDomainDecomp < All.TotNumPart * All.TreeDomainUpdateFrequency)
    {
      for(i = 0; i < Numnodestree; i++)
	for(j = 0; j < 3; j++)
	  Nodes[All.MaxPart + i].u.d.s[j] += Extnodes[All.MaxPart + i].vs[j] * dt_drift;

      force_update_len();

      force_update_pseudoparticles();
    }

  t1 = second();

  All.CPU_Predict += timediff(t0, t1);
}

/*! This function makes sure that all particle coordinates (Pos) are
 *  periodically mapped onto the interval [0, BoxSize].  After this function
 *  has been called, a new domain decomposition should be done, which will
 *  also force a new tree construction.
 */
#ifdef PERIODIC
void do_box_wrapping(void)
{
  int i, j;
  double boxsize[3];

  for(j = 0; j < 3; j++)
    boxsize[j] = All.BoxSize;

#ifdef LONG_X
  boxsize[0] *= LONG_X;
#endif
#ifdef LONG_Y
  boxsize[1] *= LONG_Y;
#endif
#ifdef LONG_Z
  boxsize[2] *= LONG_Z;
#endif

  for(i = 0; i < NumPart; i++)
    for(j = 0; j < 3; j++)
      {
	while(P[i].Pos[j] < 0)
	  P[i].Pos[j] += boxsize[j];

	while(P[i].Pos[j] >= boxsize[j])
	  P[i].Pos[j] -= boxsize[j];
      }
}
#endif
