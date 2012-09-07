//C++ includes
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <ctime>
#include <cmath>

#include <cuda_runtime.h>

//my includes
#include "lbm_utils.h"
#include "lattice_vars.h"
#include "vtk_lib.h"
#include "lbm_kernels.h"

using namespace std;

int main(int argc, char * argv[]){

  //read parameters
  int Num_ts;
  int Nx;
  int Ny;
  int Nz;
  int numSpd;
  float u_bc;
  float rho_lbm;
  float omega;
  int dynamics;
  int ts_rep_freq;
  int plot_freq;

  float * omega_op;

  //read the data
  ifstream input_params("params.lbm",ios::in);
  input_params >> Num_ts;
  input_params >> Nx;
  input_params >> Ny;
  input_params >> Nz;
  input_params >> numSpd;
  input_params >> u_bc;
  input_params >> rho_lbm;
  input_params >> omega;
  input_params >> dynamics;
  input_params >> ts_rep_freq;
  input_params >> plot_freq;
  input_params.close();

  float * ex;
  float * ey;
  float * ez;
  float * w;
  int * bb_spd;
  float * M = new float[numSpd*numSpd];
  
     
  if(dynamics==3){
    ifstream omega_op("M.lbm",ios::in);
    for(int rows=0;rows<numSpd;rows++){
      for(int cols=0;cols<numSpd;cols++){
	omega_op >> M[rows*numSpd+cols];
      }
    }
  }
  
  const int nnodes=Nx*Ny*Nz;

  //get appropriate lattice parameters.  Need these so you can 
  //initialize the lattice (at least...)

  switch (numSpd){
    //lattice parameters are defined in lattice_vars.h
  case (15):
    ex = ex15;
    ey = ey15;
    ez = ez15;
    w = w15;
    bb_spd = bb15;
    //M = M15;
    break;

  case(19):
    ex = ex19;
    ey = ey19;
    ez = ez19;
    w = w19;
    bb_spd=bb19;
    //M = M19;
    break;

  case(27):
    ex = ex27;
    ey = ey27;
    ez = ez27;
    w = w27;
    bb_spd = bb27;
    //M = NULL; //none implemented for 27-speed model yet.
    break;

  }

  float * fIn_h = new float[nnodes*numSpd];

  // arrays to hold output data to write into VTK files
  float * rho_h = new float[nnodes];
  float * ux_h = new float[nnodes];
  float * uy_h = new float[nnodes];
  float * uz_h = new float[nnodes];

  // arrays to hold geometry data
  float * X = new float[nnodes];
  float * Y = new float[nnodes];
  float * Z = new float[nnodes];

  gen_XYZ_rectLattice(X,Y,Z,Nx,Ny,Nz);

  initialize_lattice(fIn_h,w,rho_lbm,nnodes,numSpd);

  //visualization stuff...
  string densityFileStub("density");
  string velocityFileStub("velocity");
  string fileSuffix(".vtk");
  stringstream ts_ind;
  string ts_ind_str;
  int vtk_ts=0;
  string fileName;
  string dataNameDensity("density");
  string dataNameVelocity("velocity");
  int dims[3];
  dims[0]=Nx; dims[1]=Ny; dims[2]=Nz;
  float origin[3];
  origin[0]=0.; origin[1]=0.; origin[2]=0.;
  float spacing[3];
  spacing[0]=1.; spacing[1]=1.;spacing[2]=1.;

  //declare GPU variables
  float * fIn_d;
  float * fOut_d;
  float * rho_d;
  float * ux_d;
  float * uy_d;
  float * uz_d;

  float * M_d;

  //allocate GPU data
  cudaMalloc((void**)&fIn_d,nnodes*numSpd*sizeof(float));
  cudaMalloc((void**)&fOut_d,nnodes*numSpd*sizeof(float));
  cudaMalloc((void**)&rho_d,nnodes*sizeof(float));
  cudaMalloc((void**)&ux_d,nnodes*sizeof(float));
  cudaMalloc((void**)&uy_d,nnodes*sizeof(float));
  cudaMalloc((void**)&uz_d,nnodes*sizeof(float));
  cudaMalloc((void**)&M_d,numSpd*numSpd*sizeof(float));

  //copy initialized data to the GPU
  cudaMemcpy(fIn_d,fIn_h,nnodes*numSpd*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(M_d,M,numSpd*numSpd*sizeof(float),cudaMemcpyHostToDevice);
  //ready to start time stepping
  for(int ts=0;ts<Num_ts;ts++){
   
    if((ts+1)%ts_rep_freq==0){
      cout << "Executing time step number " << ts+1 << endl;
    }

    switch (dynamics) {
    case 1: //LBGK

      switch (numSpd){

      case 15:
	if((ts%2)==0){

	  //even step D3Q15 LBGK
	  ldc_D3Q15_LBGK_ts_cuda(fOut_d,fIn_d,rho_d,ux_d,uy_d,uz_d,
				 u_bc,omega,Nx,Ny,Nz);

	}else{

	  //odd step D3Q15 LBGK
	  ldc_D3Q15_LBGK_ts_cuda(fIn_d,fOut_d,rho_d,ux_d,uy_d,uz_d,
				 u_bc,omega,Nx,Ny,Nz);
	}
	break;

      case 19:
	if((ts%2)==0){

	  //even step D3Q19 LBGK
	  ldc_D3Q19_LBGK_ts_cuda(fOut_d,fIn_d,rho_d,ux_d,uy_d,uz_d,
				 u_bc,omega,Nx,Ny,Nz);
	}else{

	  //odd step D3Q19 LBGK
	  ldc_D3Q19_LBGK_ts_cuda(fIn_d,fOut_d,rho_d,ux_d,uy_d,uz_d,
				 u_bc,omega,Nx,Ny,Nz);
	}
	break;
      }

      break;


    case 3: //MRT
      switch (numSpd){

      case 15:
	if((ts%2)==0){

	  //even step D3Q15 MRT

	}else{

	  //odd step D3Q15 MRT

	}
	break;

      case 19:
	if((ts%2)==0){

	  //even step D3Q19 MRT
	  ldc_D3Q19_MRT_ts_cuda(fOut_d,fIn_d,rho_d,ux_d,uy_d,uz_d,
				 u_bc,M_d,Nx,Ny,Nz);
	}else{

	  //odd step D3Q19 MRT
	  ldc_D3Q19_MRT_ts_cuda(fIn_d,fOut_d,rho_d,ux_d,uy_d,uz_d,
				 u_bc,M_d,Nx,Ny,Nz);
	}
	break;
      }
    }



    //plot logic...

    if(((ts+1)%plot_freq) == 0){

      //get data from the GPU
      cudaMemcpy(rho_h,rho_d,nnodes*sizeof(float),cudaMemcpyDeviceToHost);
      cudaMemcpy(ux_h,ux_d,nnodes*sizeof(float),cudaMemcpyDeviceToHost);
      cudaMemcpy(uy_h,uy_d,nnodes*sizeof(float),cudaMemcpyDeviceToHost);
      cudaMemcpy(uz_h,uz_d,nnodes*sizeof(float),cudaMemcpyDeviceToHost);

      //write data to VTK file
      ts_ind << vtk_ts;
      fileName=densityFileStub+ts_ind.str()+fileSuffix;
      SaveVTKImageData_ascii(rho_h,fileName,dataNameDensity,origin,
			     spacing,dims);

      fileName=velocityFileStub+ts_ind.str()+fileSuffix;
      SaveVTKStructuredGridVectorAndMagnitude_ascii(ux_h,uy_h,uz_h,
						    X,Y,Z,
						    fileName,
						    dataNameVelocity,
						    dims);

      ts_ind.str("");
      vtk_ts+=1;

    }


  }//for(int ts=0...


  //free GPU and CPU data
  cudaFree(fIn_d);
  cudaFree(fOut_d);
  cudaFree(rho_d);
  cudaFree(ux_d);
  cudaFree(uy_d);
  cudaFree(uz_d);
  cudaFree(M_d);


  delete [] M;
  delete [] fIn_h;
  delete [] rho_h;
  delete [] ux_h;
  delete [] uy_h;
  delete [] uz_h;
  delete [] X;
  delete [] Y;
  delete [] Z;


}
