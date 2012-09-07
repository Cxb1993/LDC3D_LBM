#ifndef LBM_KERNELS_H
#define LBM_KERNELS_H


void ldc_D3Q15_LBGK_ts_cuda(float * fOut, const float * fIn,
			    float * rho, float * ux, float * uy,
			    float * uz, const float u_bc,
			    const float omega, const int Nx,
			    const int Ny, const int Nz);

void ldc_D3Q19_LBGK_ts_cuda(float * fOut, const float * fIn,
			    float * rho, float * ux, float * uy,
			    float * uz, const float u_bc,
			    const float omega, const int Nx,
			    const int Ny, const int Nz);

void ldc_D3Q19_MRT_ts_cuda(float * fOut, const float * fIn,
			   float * rho, float * ux, float * uy,
			   float * uz, const float u_bc,
			   const float * omega_op, const int Nx,
			   const int Ny, const int Nz);

#endif
