#include <iostream>

using namespace std;

//generate the lid-node list for 3D lbm routines.
extern "C"
void gen_ldc_lnl(int * lnl, int Nx, int Ny, int Nz){

  int tid;
  //#pragma omp parallel for
  for(int Z = 0; Z<Nz;Z++){
    for(int Y = 0; Y<Ny; Y++){
      for(int X=0; X<Nx;X++){
	tid=X+Y*Nx+Z*Nx*Ny;

	if(X==0){
	  if(!((Y==0)||(Y==(Ny-1))||(Z==0)||(Z==(Nz-1)))) {lnl[tid]=1;}
	}else{
	  lnl[tid]=0;
	}
      }
    }
  }

}

extern "C"
void gen_ldc_snl(int * snl, int Nx, int Ny, int Nz){
  int tid;
  for(int Z = 0; Z<Nz;Z++){
    for(int Y = 0; Y<Ny; Y++){
      for(int X=0;X<Nx;X++){
	tid = X+Y*Nx+Z*Nx*Ny;

	if(((Y==0)||(Y==(Ny-1))||(Z==0)||(Z==(Nz-1))||(X==(Nx-1)))){
	  snl[tid]=1;
	}else{
	  snl[tid]=0;
	}
      }
    }
  }
}

extern "C"
void gen_XYZ_rectLattice(float * X, float * Y, float * Z, const int Nx,
			 const int Ny, const int Nz){
  int tid;
  for(int z=0;z<Nz;z++){
    for(int y=0;y<Ny;y++){
      for(int x=0;x<Nx;x++){
	tid=x+y*Nx+z*Nx*Ny;
	X[tid]=x;Y[tid]=y;Z[tid]=z;
      }
    }
  }

}

// void print_array(const float * A, const int nrows, const int ncols){

//   cout << endl;
//   for(int i=0;i<nrows;i++){
//     for(int j=0;j<ncols;j++){
//       cout << A[i*ncols+j];
//       if(j<(ncols-1)) 
// 	cout << ", ";
//     }
//     cout << endl;
//   }
//   cout << endl;

// }

extern "C"
void initialize_lattice(float * f, const float * w,float rho, 
			const int nnodes, const int nspd){

  for(int spd=0;spd<nspd;spd++){
    for(int nd=0;nd<nnodes;nd++){
      f[nnodes*spd+nd]=w[spd]*rho;
    }
  }

}

void initialize_lattice_t(float * f, const float * w, float rho,
			  const int nnodes, const int nspd){

  for(int nd=0;nd<nnodes;nd++){
    for(int spd=0;spd<nspd;spd++){
      f[nd*nspd+spd]=w[spd]*rho;
    }
  }

}
