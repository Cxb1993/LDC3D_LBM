#include "lbm_kernels.h"

#define TPB 96

__global__ void ldc_D3Q19_MRT_ts(float * fOut, const float * fIn, 
				  float * rho_d,
				  float * ux_d, float * uy_d, float * uz_d,
				  const float u_bc, const float * M,
				  const int Nx, const int Ny,
				  const int Nz){
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  int nnodes=Nx*Ny*Nz;
  if(tid<(nnodes)){
    float f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18;
    float cu;
    float w;

    __shared__ float omega[19][19];

    //load the data into the registers
    f0=fIn[tid]; f1=fIn[nnodes+tid];
    f2=fIn[2*nnodes+tid]; f3=fIn[3*nnodes+tid];
    f4=fIn[4*nnodes+tid]; f5=fIn[5*nnodes+tid];
    f6=fIn[6*nnodes+tid]; f7=fIn[7*nnodes+tid];
    f8=fIn[8*nnodes+tid]; f9=fIn[9*nnodes+tid];
    f10=fIn[10*nnodes+tid]; f11=fIn[11*nnodes+tid];
    f12=fIn[12*nnodes+tid]; f13=fIn[13*nnodes+tid];
    f14=fIn[14*nnodes+tid]; f15=fIn[15*nnodes+tid];
    f16=fIn[16*nnodes+tid]; f17=fIn[17*nnodes+tid];
    f18=fIn[18*nnodes+tid];

    //load omega into shared memory.  Remember omega is in row-major order...
    for(int om_bit = threadIdx.x;om_bit<(19*19);om_bit+=blockDim.x){
      int row = om_bit/19;
      int col = om_bit - row*19;
      omega[row][col]=*(M+om_bit);

    }



    //compute density and velocity
    float rho = f0+f1+f2+f3+f4+f5+f6+f7+f8+f9+f10+f11+f12+f13+f14+f15+f16+f17+f18;
    float ux=f1-f2+f7-f8+f9-f10+f11-f12+f13-f14; ux/=rho;
    float uy=f3-f4+f7+f8-f9-f10+f15-f16+f17-f18; uy/=rho;
    float uz=f5-f6+f11+f12-f13-f14+f15+f16-f17-f18; uz/=rho;

    int Z = tid/(Nx*Ny);
    int Y = (tid - Z*Nx*Ny)/Nx;
    int X = tid - Z*Nx*Ny - Y*Nx;

    if((X==0)&&(!((Y==0)||(Y==(Ny-1))||(Z==0)||(Z==(Nz-1))))){
      //apply velocity boundary condition here...
      //u_bc is the prescribed y velocity

      //speed 1 (1,0,0) w=1./18.
      w = 1./18.;
      cu = 3.*(-ux);
      f1+=w*rho*cu;

      //speed 2 (-1,0,0) 
      cu=3.*(-1.)*(-ux);
      f2+=w*rho*cu;

      //speed 3 (0,1,0)
      cu = 3.*(u_bc-uy);
      f3+=w*rho*cu;

      //speed 4 (0,-1,0)
      cu = 3.*(-1.)*(u_bc-uy);
      f4+=w*rho*cu;

      //speed 5 (0,0,1)
      cu = 3.*(-uz);
      f5+=w*rho*cu;

      //speed 6 (0,0,-1)
      cu = 3.*(-1.)*(-uz);
      f6+=w*rho*cu;

      w = 1./36.;
      //speed 7 (1,1,0)
      cu = 3.*((-ux)+(u_bc-uy));
      f7+=w*rho*cu;

      //speed 8 ( -1,1,0)
      cu = 3.*((-1.)*(-ux) + (u_bc-uy));
      f8+=w*rho*cu;

      //speed 9 (1,-1,0)
      cu = 3.*((-ux) -(u_bc-uy));
      f9+=w*rho*cu;

      //speed 10 (-1,-1,0)
      cu = 3.*(-(-ux) -(u_bc-uy));
      f10+=w*rho*cu;

      //speed 11 (1,0,1)
      cu = 3.*((-ux)+(-uz));
      f11+=w*rho*cu;

      //speed 12 (-1,0,1)
      cu = 3.*(ux -uz);
      f12+=w*rho*cu;

      //speed 13 (1,0,-1)
      cu = 3.*(-ux + uz);
      f13+=w*rho*cu;

      //speed 14 (-1,0,-1)
      cu = 3.*(ux+uz);
      f14+=w*rho*cu;

      //speed 15 ( 0,1,1)
      cu = 3.*((u_bc-uy)-uz);
      f15+=w*rho*cu;

      //speed 16 (0,-1,1)
      cu = 3.*(-(u_bc-uy)-uz);
      f16+=w*rho*cu;

      //speed 17 (0,1,-1)
      cu = 3.*((u_bc-uy)+uz);
      f17+=w*rho*cu;

      //speed 18 (0,-1,-1)
      cu = 3.*((uy-u_bc)+uz);
      f18+=w*rho*cu;

      ux=0.; uy =u_bc; uz = 0.;

    }//if(lnl[tid]==1)...

    __syncthreads();
    //if(snl[tid]==1){
    if(((Y==0)||(Y==(Ny-1))||(Z==0)||(Z==(Nz-1))||(X==(Nx-1)))){
      //bounce back
      ux=0.; uy=0.; uz=0.;

      
      //1 -- 2
      cu=f1;f1=f2;f2=cu;
      // 3 -- 4
      cu=f3;f3=f4;f4=cu;
      //5--6
      cu=f5;f5=f6;f6=cu;
      //7--10
      cu=f7;f7=f10;f10=cu;
      //8--9
      cu=f8;f8=f9;f9=cu;
      //11-14
      cu=f11;f11=f14;f14=cu;
      //12-13
      cu=f12;f12=f13;f13=cu;
      //15-18
      cu=f15;f15=f18;f18=cu;
      //16-17
      cu=f16;f16=f17;f17=cu;

    }else{

   
     
      float fe0,fe1,fe2,fe3,fe4,fe5,fe6,fe7,fe8,fe9,fe10,fe11,fe12,fe13,fe14,fe15,fe16,fe17,fe18;

      //speed 0, ex=ey=ez=0, w=1/3
      fe0=rho*(1./3.)*(1.-1.5*(ux*ux+uy*uy+uz*uz));
   
      //speed 1, ex=1, ey=ez=0, w=1/18
      cu = 3.*(1.*ux);
      fe1=rho*(1./18.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
      //speed 2, ex=-1, ey=ez=0
      cu=3.*(-1.*ux);
      fe2=rho*(1./18.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
      //speed 3 (0,1,0)
      cu=3.*(uy);
 
      fe3=rho*(1./18.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
      //speed 4 (0,-1,0)
      cu = 3.*(-uy);
      fe4=rho*(1./18.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
      //speed 5 (0,0,1)
      cu = 3.*(uz);
      fe5=rho*(1./18.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
      //speed 6 (0,0,-1)
      cu = 3.*(-uz);
      fe6=rho*(1./18.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
      //speed 7 (1,1,0)  w= 1/36
      cu = 3.*(ux+uy);
      fe7=rho*(1./36.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
      //speed 8 (-1,1,0)
      cu = 3.*(-ux+uy);
      fe8=rho*(1./36.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
      //speed 9 (1,-1,0)
      cu=3.*(ux-uy);
      fe9=rho*(1./36.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
      //speed 10 (-1,-1,0)
      cu = 3.*(-ux-uy);
      fe10=rho*(1./36.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
      //speed 11 (1,0,1)
      cu = 3.*(ux+uz);
      fe11=rho*(1./36.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
      //speed 12 (-1,0,1)
      cu = 3.*(-ux+uz);
      fe12=rho*(1./36.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
      //speed 13 (1,0,-1)
      cu = 3.*(ux-uz);
      fe13=rho*(1./36.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
      //speed 14 (-1,0,-1)
      cu=3.*(-ux-uz);
      fe14=rho*(1./36.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
      //speed 15 (0,1,1)
      cu=3.*(uy+uz);
      fe15=rho*(1./36.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
      //speed 16 (0,-1,1)
      cu=3.*(-uy+uz);
      fe16=rho*(1./36.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
      //speed 17 (0,1,-1)
      cu=3.*(uy-uz);
      fe17=rho*(1./36.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
      //speed 18 (0,-1,-1)
      cu=3.*(-uy-uz);
      fe18=rho*(1./36.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));



      //collect into non-equilibrium parts
      fe0=f0-fe0;
      fe1=f1-fe1;
      fe2=f2-fe2;
      fe3=f3-fe3;
      fe4=f4-fe4;
      fe5=f5-fe5;
      fe6=f6-fe6;
      fe7=f7-fe7;
      fe8=f8-fe8;
      fe9=f9-fe9;
      fe10=f10-fe10;
      fe11=f11-fe11;
      fe12=f12-fe12;
      fe13=f13-fe13;
      fe14=f14-fe14;
      fe15=f15-fe15;
      fe16=f16-fe16;
      fe17=f17-fe17;
      fe18=f18-fe18;


  

      f0=f0-(fe0*omega[0][0]+fe1*omega[1][0]+fe2*omega[2][0]+fe3*omega[3][0]+fe4*omega[4][0]+fe5*omega[5][0]+fe6*omega[6][0]+fe7*omega[7][0]+fe8*omega[8][0]+fe9*omega[9][0]+fe10*omega[10][0]+fe11*omega[11][0]+fe12*omega[12][0]+fe13*omega[13][0]+fe14*omega[14][0]+fe15*omega[15][0]+fe16*omega[16][0]+fe17*omega[17][0]+fe18*omega[18][0]);

      f1=f1-(fe0*omega[0][1]+fe1*omega[1][1]+fe2*omega[2][1]+fe3*omega[3][1]+fe4*omega[4][1]+fe5*omega[5][1]+fe6*omega[6][1]+fe7*omega[7][1]+fe8*omega[8][1]+fe9*omega[9][1]+fe10*omega[10][1]+fe11*omega[11][1]+fe12*omega[12][1]+fe13*omega[13][1]+fe14*omega[14][1]+fe15*omega[15][1]+fe16*omega[16][1]+fe17*omega[17][1]+fe18*omega[18][1]);

      f2=f2-(fe0*omega[0][2]+fe1*omega[1][2]+fe2*omega[2][2]+fe3*omega[3][2]+fe4*omega[4][2]+fe5*omega[5][2]+fe6*omega[6][2]+fe7*omega[7][2]+fe8*omega[8][2]+fe9*omega[9][2]+fe10*omega[10][2]+fe11*omega[11][2]+fe12*omega[12][2]+fe13*omega[13][2]+fe14*omega[14][2]+fe15*omega[15][2]+fe16*omega[16][2]+fe17*omega[17][2]+fe18*omega[18][2]);

      f3=f3-(fe0*omega[0][3]+fe1*omega[1][3]+fe2*omega[2][3]+fe3*omega[3][3]+fe4*omega[4][3]+fe5*omega[5][3]+fe6*omega[6][3]+fe7*omega[7][3]+fe8*omega[8][3]+fe9*omega[9][3]+fe10*omega[10][3]+fe11*omega[11][3]+fe12*omega[12][3]+fe13*omega[13][3]+fe14*omega[14][3]+fe15*omega[15][3]+fe16*omega[16][3]+fe17*omega[17][3]+fe18*omega[18][3]);

      f4=f4-(fe0*omega[0][4]+fe1*omega[1][4]+fe2*omega[2][4]+fe3*omega[3][4]+fe4*omega[4][4]+fe5*omega[5][4]+fe6*omega[6][4]+fe7*omega[7][4]+fe8*omega[8][4]+fe9*omega[9][4]+fe10*omega[10][4]+fe11*omega[11][4]+fe12*omega[12][4]+fe13*omega[13][4]+fe14*omega[14][4]+fe15*omega[15][4]+fe16*omega[16][4]+fe17*omega[17][4]+fe18*omega[18][4]);

      f5=f5-(fe0*omega[0][5]+fe1*omega[1][5]+fe2*omega[2][5]+fe3*omega[3][5]+fe4*omega[4][5]+fe5*omega[5][5]+fe6*omega[6][5]+fe7*omega[7][5]+fe8*omega[8][5]+fe9*omega[9][5]+fe10*omega[10][5]+fe11*omega[11][5]+fe12*omega[12][5]+fe13*omega[13][5]+fe14*omega[14][5]+fe15*omega[15][5]+fe16*omega[16][5]+fe17*omega[17][5]+fe18*omega[18][5]);

      f6=f6-(fe0*omega[0][6]+fe1*omega[1][6]+fe2*omega[2][6]+fe3*omega[3][6]+fe4*omega[4][6]+fe5*omega[5][6]+fe6*omega[6][6]+fe7*omega[7][6]+fe8*omega[8][6]+fe9*omega[9][6]+fe10*omega[10][6]+fe11*omega[11][6]+fe12*omega[12][6]+fe13*omega[13][6]+fe14*omega[14][6]+fe15*omega[15][6]+fe16*omega[16][6]+fe17*omega[17][6]+fe18*omega[18][6]);

      f7=f7-(fe0*omega[0][7]+fe1*omega[1][7]+fe2*omega[2][7]+fe3*omega[3][7]+fe4*omega[4][7]+fe5*omega[5][7]+fe6*omega[6][7]+fe7*omega[7][7]+fe8*omega[8][7]+fe9*omega[9][7]+fe10*omega[10][7]+fe11*omega[11][7]+fe12*omega[12][7]+fe13*omega[13][7]+fe14*omega[14][7]+fe15*omega[15][7]+fe16*omega[16][7]+fe17*omega[17][7]+fe18*omega[18][7]);

      f8=f8-(fe0*omega[0][8]+fe1*omega[1][8]+fe2*omega[2][8]+fe3*omega[3][8]+fe4*omega[4][8]+fe5*omega[5][8]+fe6*omega[6][8]+fe7*omega[7][8]+fe8*omega[8][8]+fe9*omega[9][8]+fe10*omega[10][8]+fe11*omega[11][8]+fe12*omega[12][8]+fe13*omega[13][8]+fe14*omega[14][8]+fe15*omega[15][8]+fe16*omega[16][8]+fe17*omega[17][8]+fe18*omega[18][8]);

      f9=f9-(fe0*omega[0][9]+fe1*omega[1][9]+fe2*omega[2][9]+fe3*omega[3][9]+fe4*omega[4][9]+fe5*omega[5][9]+fe6*omega[6][9]+fe7*omega[7][9]+fe8*omega[8][9]+fe9*omega[9][9]+fe10*omega[10][9]+fe11*omega[11][9]+fe12*omega[12][9]+fe13*omega[13][9]+fe14*omega[14][9]+fe15*omega[15][9]+fe16*omega[16][9]+fe17*omega[17][9]+fe18*omega[18][9]);

      f10=f10-(fe0*omega[0][10]+fe1*omega[1][10]+fe2*omega[2][10]+fe3*omega[3][10]+fe4*omega[4][10]+fe5*omega[5][10]+fe6*omega[6][10]+fe7*omega[7][10]+fe8*omega[8][10]+fe9*omega[9][10]+fe10*omega[10][10]+fe11*omega[11][10]+fe12*omega[12][10]+fe13*omega[13][10]+fe14*omega[14][10]+fe15*omega[15][10]+fe16*omega[16][10]+fe17*omega[17][10]+fe18*omega[18][10]);

      f11=f11-(fe0*omega[0][11]+fe1*omega[1][11]+fe2*omega[2][11]+fe3*omega[3][11]+fe4*omega[4][11]+fe5*omega[5][11]+fe6*omega[6][11]+fe7*omega[7][11]+fe8*omega[8][11]+fe9*omega[9][11]+fe10*omega[10][11]+fe11*omega[11][11]+fe12*omega[12][11]+fe13*omega[13][11]+fe14*omega[14][11]+fe15*omega[15][11]+fe16*omega[16][11]+fe17*omega[17][11]+fe18*omega[18][11]);

      f12=f12-(fe0*omega[0][12]+fe1*omega[1][12]+fe2*omega[2][12]+fe3*omega[3][12]+fe4*omega[4][12]+fe5*omega[5][12]+fe6*omega[6][12]+fe7*omega[7][12]+fe8*omega[8][12]+fe9*omega[9][12]+fe10*omega[10][12]+fe11*omega[11][12]+fe12*omega[12][12]+fe13*omega[13][12]+fe14*omega[14][12]+fe15*omega[15][12]+fe16*omega[16][12]+fe17*omega[17][12]+fe18*omega[18][12]);

      f13=f13-(fe0*omega[0][13]+fe1*omega[1][13]+fe2*omega[2][13]+fe3*omega[3][13]+fe4*omega[4][13]+fe5*omega[5][13]+fe6*omega[6][13]+fe7*omega[7][13]+fe8*omega[8][13]+fe9*omega[9][13]+fe10*omega[10][13]+fe11*omega[11][13]+fe12*omega[12][13]+fe13*omega[13][13]+fe14*omega[14][13]+fe15*omega[15][13]+fe16*omega[16][13]+fe17*omega[17][13]+fe18*omega[18][13]);

      f14=f14-(fe0*omega[0][14]+fe1*omega[1][14]+fe2*omega[2][14]+fe3*omega[3][14]+fe4*omega[4][14]+fe5*omega[5][14]+fe6*omega[6][14]+fe7*omega[7][14]+fe8*omega[8][14]+fe9*omega[9][14]+fe10*omega[10][14]+fe11*omega[11][14]+fe12*omega[12][14]+fe13*omega[13][14]+fe14*omega[14][14]+fe15*omega[15][14]+fe16*omega[16][14]+fe17*omega[17][14]+fe18*omega[18][14]);

      f15=f15-(fe0*omega[0][15]+fe1*omega[1][15]+fe2*omega[2][15]+fe3*omega[3][15]+fe4*omega[4][15]+fe5*omega[5][15]+fe6*omega[6][15]+fe7*omega[7][15]+fe8*omega[8][15]+fe9*omega[9][15]+fe10*omega[10][15]+fe11*omega[11][15]+fe12*omega[12][15]+fe13*omega[13][15]+fe14*omega[14][15]+fe15*omega[15][15]+fe16*omega[16][15]+fe17*omega[17][15]+fe18*omega[18][15]);

      f16=f16-(fe0*omega[0][16]+fe1*omega[1][16]+fe2*omega[2][16]+fe3*omega[3][16]+fe4*omega[4][16]+fe5*omega[5][16]+fe6*omega[6][16]+fe7*omega[7][16]+fe8*omega[8][16]+fe9*omega[9][16]+fe10*omega[10][16]+fe11*omega[11][16]+fe12*omega[12][16]+fe13*omega[13][16]+fe14*omega[14][16]+fe15*omega[15][16]+fe16*omega[16][16]+fe17*omega[17][16]+fe18*omega[18][16]);

      f17=f17-(fe0*omega[0][17]+fe1*omega[1][17]+fe2*omega[2][17]+fe3*omega[3][17]+fe4*omega[4][17]+fe5*omega[5][17]+fe6*omega[6][17]+fe7*omega[7][17]+fe8*omega[8][17]+fe9*omega[9][17]+fe10*omega[10][17]+fe11*omega[11][17]+fe12*omega[12][17]+fe13*omega[13][17]+fe14*omega[14][17]+fe15*omega[15][17]+fe16*omega[16][17]+fe17*omega[17][17]+fe18*omega[18][17]);

      f18=f18-(fe0*omega[0][18]+fe1*omega[1][18]+fe2*omega[2][18]+fe3*omega[3][18]+fe4*omega[4][18]+fe5*omega[5][18]+fe6*omega[6][18]+fe7*omega[7][18]+fe8*omega[8][18]+fe9*omega[9][18]+fe10*omega[10][18]+fe11*omega[11][18]+fe12*omega[12][18]+fe13*omega[13][18]+fe14*omega[14][18]+fe15*omega[15][18]+fe16*omega[16][18]+fe17*omega[17][18]+fe18*omega[18][18]);

    }
    //everyone relaxes towards equilibrium
    // f0=f0-omega*(f0-fe0);
    // f1=f1-omega*(f1-fe1);
    // f2=f2-omega*(f2-fe2);
    // f3=f3-omega*(f3-fe3);
    // f4=f4-omega*(f4-fe4);
    // f5=f5-omega*(f5-fe5);
    // f6=f6-omega*(f6-fe6);
    // f7=f7-omega*(f7-fe7);
    // f8=f8-omega*(f8-fe8);
    // f9=f9-omega*(f9-fe9);
    // f10=f10-omega*(f10-fe10);
    // f11=f11-omega*(f11-fe11);
    // f12=f12-omega*(f12-fe12);
    // f13=f13-omega*(f13-fe13);
    // f14=f14-omega*(f14-fe14);
    // f15=f15-omega*(f15-fe15);
    // f16=f16-omega*(f16-fe16);
    // f17=f17-omega*(f17-fe17);
    // f18=f18-omega*(f18-fe18);


    rho_d[tid]=rho; ux_d[tid]=ux; uy_d[tid]=uy; uz_d[tid]=uz;
    //relax


    //now, streaming...
    int X_t,Y_t,Z_t,tid_t;

    //speed 0 (0,0,0)
    fOut[tid]=f0;
    //stream(fOut,f0,0,X,Y,Z,0,0,0,Nx,Ny,Nz);

    //speed 1 (1,0,0)
    X_t=X+1;Y_t=Y; Z_t=Z;
    if(X_t==Nx) X_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[nnodes+tid_t]=f1;
    
    //speed 2 (-1,0,0)
    X_t=X-1; Y_t=Y; Z_t=Z;
    if(X_t<0)X_t=Nx-1;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[2*nnodes+tid_t]=f2;

    //speed 3 (0,1,0)
    X_t=X; Y_t=Y+1; Z_t=Z;
    if(Y_t==Ny)Y_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    //tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[3*nnodes+tid_t]=f3;
    //speed 4 ( 0,-1,0)
    X_t=X; Y_t=Y-1; Z_t=Z;
    if(Y_t<0)Y_t=Ny-1;

    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[4*nnodes+tid_t]=f4; 
    //speed 5 ( 0,0,1)
    X_t=X;Y_t=Y;Z_t=Z+1;
    if(Z_t==Nz)Z_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[5*nnodes+tid_t]=f5;
    //speed 6 (0,0,-1)
    X_t=X; Y_t=Y;Z_t=Z-1;
    if(Z_t<0)Z_t=Nz-1;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[6*nnodes+tid_t]=f6;
    //speed 7 (1,1,0)
    X_t=X+1;Y_t=Y+1;Z_t=Z;
    if(X_t==Nx)X_t=0;
    if(Y_t==Ny)Y_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[7*nnodes+tid_t]=f7;
    //speed 8 (-1,1,0)
    X_t=X-1;Y_t=Y+1;Z_t=Z;
    if(X_t<0)X_t=Nx-1;
    if(Y_t==Ny)Y_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[8*nnodes+tid_t]=f8;
    //speed 9 (1,-1,0)
    X_t=X+1;Y_t=Y-1;Z_t=Z;
    if(X_t==Nx)X_t=0;
    if(Y_t<0)Y_t=Ny-1;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[9*nnodes+tid_t]=f9;
    //speed 10 (-1,-1,0)
    X_t=X-1;Y_t=Y-1;Z_t=Z;
    if(X_t<0)X_t=Nx-1;
    if(Y_t<0)Y_t=Ny-1;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[10*nnodes+tid_t]=f10;
    //speed 11 (1,0,1)
    X_t=X+1;Y_t=Y;Z_t=Z+1;
    if(X_t==Nx)X_t=0;
    if(Z_t==Nz)Z_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[11*nnodes+tid_t]=f11;
    //speed 12 (-1,0,1)
    X_t=X-1;Y_t=Y;Z_t=Z+1;
    if(X_t<0)X_t=Nx-1;
    if(Z_t==Nz)Z_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[12*nnodes+tid_t]=f12;
    //speed 13 (1,0,-1)
    X_t=X+1;Y_t=Y;Z_t=Z-1;
    if(X_t==Nx)X_t=0;
    if(Z_t<0)Z_t=Nz-1;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[13*nnodes+tid_t]=f13;
    //speed 14 (-1,0,-1)
    X_t=X-1;Y_t=Y;Z_t=Z-1;
    if(X_t<0)X_t=Nx-1;
    if(Z_t<0)Z_t=Nz-1;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[14*nnodes+tid_t]=f14;
    //speed 15 (0,1,1)
    X_t=X;Y_t=Y+1;Z_t=Z+1;
    if(Y_t==Ny)Y_t=0;
    if(Z_t==Nz)Z_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[15*nnodes+tid_t]=f15;
    //speed 16 (0,-1,1)
    X_t=X;Y_t=Y-1;Z_t=Z+1;
    if(Y_t<0)Y_t=Ny-1;
    if(Z_t==Nz)Z_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[16*nnodes+tid_t]=f16;

    //speed 17 (0,1,-1)
    X_t=X;Y_t=Y+1;Z_t=Z-1;
    if(Y_t==Ny)Y_t=0;
    if(Z_t<0)Z_t=Nz-1;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[17*nnodes+tid_t]=f17;


    //speed 18 ( 0,-1,-1)
    X_t=X;Y_t=Y-1;Z_t=Z-1;
    if(Y_t<0)Y_t=Ny-1;
    if(Z_t<0)Z_t=Nz-1;

    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[18*nnodes+tid_t]=f18;
  }
}



__global__ void ldc_D3Q19_LBGK_ts(float * fOut, const float * fIn, 
				  float * rho_d,
				  float * ux_d, float * uy_d, float * uz_d,
				  const float u_bc, const float omega,
				  const int Nx, const int Ny,
				  const int Nz){
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  int nnodes=Nx*Ny*Nz;
  if(tid<(nnodes)){
    float f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18;
    float cu;
    float w;
    //load the data into the registers
    f0=fIn[tid]; f1=fIn[nnodes+tid];
    f2=fIn[2*nnodes+tid]; f3=fIn[3*nnodes+tid];
    f4=fIn[4*nnodes+tid]; f5=fIn[5*nnodes+tid];
    f6=fIn[6*nnodes+tid]; f7=fIn[7*nnodes+tid];
    f8=fIn[8*nnodes+tid]; f9=fIn[9*nnodes+tid];
    f10=fIn[10*nnodes+tid]; f11=fIn[11*nnodes+tid];
    f12=fIn[12*nnodes+tid]; f13=fIn[13*nnodes+tid];
    f14=fIn[14*nnodes+tid]; f15=fIn[15*nnodes+tid];
    f16=fIn[16*nnodes+tid]; f17=fIn[17*nnodes+tid];
    f18=fIn[18*nnodes+tid];
    //compute density and velocity
    float rho = f0+f1+f2+f3+f4+f5+f6+f7+f8+f9+f10+f11+f12+f13+f14+f15+f16+f17+f18;
    float ux=f1-f2+f7-f8+f9-f10+f11-f12+f13-f14; ux/=rho;
    float uy=f3-f4+f7+f8-f9-f10+f15-f16+f17-f18; uy/=rho;
    float uz=f5-f6+f11+f12-f13-f14+f15+f16-f17-f18; uz/=rho;

    int Z = tid/(Nx*Ny);
    int Y = (tid - Z*Nx*Ny)/Nx;
    int X = tid - Z*Nx*Ny - Y*Nx;

    if((X==0)&&(!((Y==0)||(Y==(Ny-1))||(Z==0)||(Z==(Nz-1))))){
      //apply velocity boundary condition here...
      //u_bc is the prescribed y velocity

      //speed 1 (1,0,0) w=1./18.
      w = 1./18.;
      cu = 3.*(-ux);
      f1+=w*rho*cu;

      //speed 2 (-1,0,0) 
      cu=3.*(-1.)*(-ux);
      f2+=w*rho*cu;

      //speed 3 (0,1,0)
      cu = 3.*(u_bc-uy);
      f3+=w*rho*cu;

      //speed 4 (0,-1,0)
      cu = 3.*(-1.)*(u_bc-uy);
      f4+=w*rho*cu;

      //speed 5 (0,0,1)
      cu = 3.*(-uz);
      f5+=w*rho*cu;

      //speed 6 (0,0,-1)
      cu = 3.*(-1.)*(-uz);
      f6+=w*rho*cu;

      w = 1./36.;
      //speed 7 (1,1,0)
      cu = 3.*((-ux)+(u_bc-uy));
      f7+=w*rho*cu;

      //speed 8 ( -1,1,0)
      cu = 3.*((-1.)*(-ux) + (u_bc-uy));
      f8+=w*rho*cu;

      //speed 9 (1,-1,0)
      cu = 3.*((-ux) -(u_bc-uy));
      f9+=w*rho*cu;

      //speed 10 (-1,-1,0)
      cu = 3.*(-(-ux) -(u_bc-uy));
      f10+=w*rho*cu;

      //speed 11 (1,0,1)
      cu = 3.*((-ux)+(-uz));
      f11+=w*rho*cu;

      //speed 12 (-1,0,1)
      cu = 3.*(ux -uz);
      f12+=w*rho*cu;

      //speed 13 (1,0,-1)
      cu = 3.*(-ux + uz);
      f13+=w*rho*cu;

      //speed 14 (-1,0,-1)
      cu = 3.*(ux+uz);
      f14+=w*rho*cu;

      //speed 15 ( 0,1,1)
      cu = 3.*((u_bc-uy)-uz);
      f15+=w*rho*cu;

      //speed 16 (0,-1,1)
      cu = 3.*(-(u_bc-uy)-uz);
      f16+=w*rho*cu;

      //speed 17 (0,1,-1)
      cu = 3.*((u_bc-uy)+uz);
      f17+=w*rho*cu;

      //speed 18 (0,-1,-1)
      cu = 3.*((uy-u_bc)+uz);
      f18+=w*rho*cu;

      ux=0.; uy =u_bc; uz = 0.;

    }//if(lnl[tid]==1)...

   
    //if(snl[tid]==1){
    if(((Y==0)||(Y==(Ny-1))||(Z==0)||(Z==(Nz-1))||(X==(Nx-1)))){
      //bounce back
      ux=0.; uy=0.; uz=0.;

      
      //1 -- 2
      cu=f1;f1=f2;f2=cu;
      // 3 -- 4
      cu=f3;f3=f4;f4=cu;
      //5--6
      cu=f5;f5=f6;f6=cu;
      //7--10
      cu=f7;f7=f10;f10=cu;
      //8--9
      cu=f8;f8=f9;f9=cu;
      //11-14
      cu=f11;f11=f14;f14=cu;
      //12-13
      cu=f12;f12=f13;f13=cu;
      //15-18
      cu=f15;f15=f18;f18=cu;
      //16-17
      cu=f16;f16=f17;f17=cu;

    }

    rho_d[tid]=rho; ux_d[tid]=ux; uy_d[tid]=uy; uz_d[tid]=uz;
    //relax
     
    float fe0,fe1,fe2,fe3,fe4,fe5,fe6,fe7,fe8,fe9,fe10,fe11,fe12,fe13,fe14,fe15,fe16,fe17,fe18;

    //speed 0, ex=ey=ez=0, w=1/3
    fe0=rho*(1./3.)*(1.-1.5*(ux*ux+uy*uy+uz*uz));
   
    //speed 1, ex=1, ey=ez=0, w=1/18
    cu = 3.*(1.*ux);
    fe1=rho*(1./18.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
    //speed 2, ex=-1, ey=ez=0
    cu=3.*(-1.*ux);
    fe2=rho*(1./18.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
    //speed 3 (0,1,0)
    cu=3.*(uy);
 
    fe3=rho*(1./18.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
    //speed 4 (0,-1,0)
    cu = 3.*(-uy);
    fe4=rho*(1./18.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
    //speed 5 (0,0,1)
    cu = 3.*(uz);
    fe5=rho*(1./18.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
    //speed 6 (0,0,-1)
    cu = 3.*(-uz);
    fe6=rho*(1./18.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
    //speed 7 (1,1,0)  w= 1/36
    cu = 3.*(ux+uy);
    fe7=rho*(1./36.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
    //speed 8 (-1,1,0)
    cu = 3.*(-ux+uy);
    fe8=rho*(1./36.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
    //speed 9 (1,-1,0)
    cu=3.*(ux-uy);
    fe9=rho*(1./36.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
    //speed 10 (-1,-1,0)
    cu = 3.*(-ux-uy);
    fe10=rho*(1./36.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
    //speed 11 (1,0,1)
    cu = 3.*(ux+uz);
    fe11=rho*(1./36.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
    //speed 12 (-1,0,1)
    cu = 3.*(-ux+uz);
    fe12=rho*(1./36.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
    //speed 13 (1,0,-1)
    cu = 3.*(ux-uz);
    fe13=rho*(1./36.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
    //speed 14 (-1,0,-1)
    cu=3.*(-ux-uz);
    fe14=rho*(1./36.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
    //speed 15 (0,1,1)
    cu=3.*(uy+uz);
    fe15=rho*(1./36.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
    //speed 16 (0,-1,1)
    cu=3.*(-uy+uz);
    fe16=rho*(1./36.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
    //speed 17 (0,1,-1)
    cu=3.*(uy-uz);
    fe17=rho*(1./36.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
    //speed 18 (0,-1,-1)
    cu=3.*(-uy-uz);
    fe18=rho*(1./36.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));



    //everyone relaxes towards equilibrium
    f0=f0-omega*(f0-fe0);
    f1=f1-omega*(f1-fe1);
    f2=f2-omega*(f2-fe2);
    f3=f3-omega*(f3-fe3);
    f4=f4-omega*(f4-fe4);
    f5=f5-omega*(f5-fe5);
    f6=f6-omega*(f6-fe6);
    f7=f7-omega*(f7-fe7);
    f8=f8-omega*(f8-fe8);
    f9=f9-omega*(f9-fe9);
    f10=f10-omega*(f10-fe10);
    f11=f11-omega*(f11-fe11);
    f12=f12-omega*(f12-fe12);
    f13=f13-omega*(f13-fe13);
    f14=f14-omega*(f14-fe14);
    f15=f15-omega*(f15-fe15);
    f16=f16-omega*(f16-fe16);
    f17=f17-omega*(f17-fe17);
    f18=f18-omega*(f18-fe18);



   

   

    //now, streaming...
    int X_t,Y_t,Z_t,tid_t;

    //speed 0 (0,0,0)
    fOut[tid]=f0;
    //stream(fOut,f0,0,X,Y,Z,0,0,0,Nx,Ny,Nz);

    //speed 1 (1,0,0)
    X_t=X+1;Y_t=Y; Z_t=Z;
    if(X_t==Nx) X_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[nnodes+tid_t]=f1;
    
    //speed 2 (-1,0,0)
    X_t=X-1; Y_t=Y; Z_t=Z;
    if(X_t<0)X_t=Nx-1;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[2*nnodes+tid_t]=f2;

    //speed 3 (0,1,0)
    X_t=X; Y_t=Y+1; Z_t=Z;
    if(Y_t==Ny)Y_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    //tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[3*nnodes+tid_t]=f3;
    //speed 4 ( 0,-1,0)
    X_t=X; Y_t=Y-1; Z_t=Z;
    if(Y_t<0)Y_t=Ny-1;

    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[4*nnodes+tid_t]=f4; 
    //speed 5 ( 0,0,1)
    X_t=X;Y_t=Y;Z_t=Z+1;
    if(Z_t==Nz)Z_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[5*nnodes+tid_t]=f5;
    //speed 6 (0,0,-1)
    X_t=X; Y_t=Y;Z_t=Z-1;
    if(Z_t<0)Z_t=Nz-1;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[6*nnodes+tid_t]=f6;
    //speed 7 (1,1,0)
    X_t=X+1;Y_t=Y+1;Z_t=Z;
    if(X_t==Nx)X_t=0;
    if(Y_t==Ny)Y_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[7*nnodes+tid_t]=f7;
    //speed 8 (-1,1,0)
    X_t=X-1;Y_t=Y+1;Z_t=Z;
    if(X_t<0)X_t=Nx-1;
    if(Y_t==Ny)Y_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[8*nnodes+tid_t]=f8;
    //speed 9 (1,-1,0)
    X_t=X+1;Y_t=Y-1;Z_t=Z;
    if(X_t==Nx)X_t=0;
    if(Y_t<0)Y_t=Ny-1;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[9*nnodes+tid_t]=f9;
    //speed 10 (-1,-1,0)
    X_t=X-1;Y_t=Y-1;Z_t=Z;
    if(X_t<0)X_t=Nx-1;
    if(Y_t<0)Y_t=Ny-1;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[10*nnodes+tid_t]=f10;
    //speed 11 (1,0,1)
    X_t=X+1;Y_t=Y;Z_t=Z+1;
    if(X_t==Nx)X_t=0;
    if(Z_t==Nz)Z_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[11*nnodes+tid_t]=f11;
    //speed 12 (-1,0,1)
    X_t=X-1;Y_t=Y;Z_t=Z+1;
    if(X_t<0)X_t=Nx-1;
    if(Z_t==Nz)Z_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[12*nnodes+tid_t]=f12;
    //speed 13 (1,0,-1)
    X_t=X+1;Y_t=Y;Z_t=Z-1;
    if(X_t==Nx)X_t=0;
    if(Z_t<0)Z_t=Nz-1;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[13*nnodes+tid_t]=f13;
    //speed 14 (-1,0,-1)
    X_t=X-1;Y_t=Y;Z_t=Z-1;
    if(X_t<0)X_t=Nx-1;
    if(Z_t<0)Z_t=Nz-1;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[14*nnodes+tid_t]=f14;
    //speed 15 (0,1,1)
    X_t=X;Y_t=Y+1;Z_t=Z+1;
    if(Y_t==Ny)Y_t=0;
    if(Z_t==Nz)Z_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[15*nnodes+tid_t]=f15;
    //speed 16 (0,-1,1)
    X_t=X;Y_t=Y-1;Z_t=Z+1;
    if(Y_t<0)Y_t=Ny-1;
    if(Z_t==Nz)Z_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[16*nnodes+tid_t]=f16;

    //speed 17 (0,1,-1)
    X_t=X;Y_t=Y+1;Z_t=Z-1;
    if(Y_t==Ny)Y_t=0;
    if(Z_t<0)Z_t=Nz-1;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[17*nnodes+tid_t]=f17;


    //speed 18 ( 0,-1,-1)
    X_t=X;Y_t=Y-1;Z_t=Z-1;
    if(Y_t<0)Y_t=Ny-1;
    if(Z_t<0)Z_t=Nz-1;

    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[18*nnodes+tid_t]=f18;
  }
}


__global__ void ldc_D3Q15_LBGK_ts(float * fOut, const float * fIn,float * rho_d,
				 float * ux_d, float * uy_d, float * uz_d,
				  const float u_bc, const float omega,
				  const int Nx, const int Ny,
				 const int Nz){

  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  int Z = tid/(Nx*Ny);
  int Y = (tid - Z*Nx*Ny)/Nx;
  int X = tid - Z*Nx*Ny - Y*Nx;
  if((X<Nx)&&(Y<Ny)&&(Z<Nz)){
    //int tid=X+Y*Nx+Z*Nx*Ny;
    float f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14;
    float cu;
    //load the data into registers
    f0=fIn[tid]; f1=fIn[Nx*Ny*Nz+tid];
    f2=fIn[2*Nx*Ny*Nz+tid]; f3=fIn[3*Nx*Ny*Nz+tid];
    f4=fIn[4*Nx*Ny*Nz+tid]; f5=fIn[5*Nx*Ny*Nz+tid];
    f6=fIn[6*Nx*Ny*Nz+tid]; f7=fIn[7*Nx*Ny*Nz+tid];
    f8=fIn[8*Nx*Ny*Nz+tid]; f9=fIn[9*Nx*Ny*Nz+tid];
    f10=fIn[10*Nx*Ny*Nz+tid]; f11=fIn[11*Nx*Ny*Nz+tid];
    f12=fIn[12*Nx*Ny*Nz+tid]; f13=fIn[13*Nx*Ny*Nz+tid];
    f14=fIn[14*Nx*Ny*Nz+tid];

    //compute density
    float rho = f0+f1+f2+f3+f4+f5+f6+f7+f8+f9+f10+f11+f12+f13+f14;
    float ux=f1-f2+f7-f8+f9-f10+f11-f12+f13-f14; ux/=rho;
    float uy=f3-f4+f7+f8-f9-f10+f11+f12-f13-f14; uy/=rho;
    float uz=f5-f6+f7+f8+f9+f10-f11-f12-f13-f14; uz/=rho;

    

    //if it's a lid node, update
    //if(lnl[tid]==1){
    if((X==0)&&(!((Y==0)||(Y==(Ny-1))||(Z==0)||(Z==(Nz-1))))){

      //speed 1 ex=1 ey=ez=0. w=1./9.
      cu=3.*(1.)*(-ux);
      f1+=(1./9.)*rho*cu;

      //speed 2 ex=-1 ey=ez=0. w=1./9.
      cu=3.*(-1.)*(-ux);
      f2+=(1./9.)*rho*cu;

      //speed 3 ey=1; ex=ez=0; w=1./9.
      cu=3.*(1.)*(u_bc-uy);
      f3+=(1./9.)*rho*cu;

      //speed 4 ey=-1; ex=ez=0; w=1./9.
      cu=3.*(-1.)*(u_bc-uy);
      f4+=(1./9.)*rho*cu;

      //speed 5 ex=ey=0; ez=1; w=1./9.
      cu=3.*(1.)*(-uz);
      f5+=(1./9.)*rho*cu;

      //speed 6 ex=ey=0; ez=-1; w=1./9.
      cu=3.*(-1.)*(-uz);
      f6+=(1./9.)*rho*cu;

      //speed 7 ex=ey=ez=1; w=1./72.
      cu=3.*((1.)*-ux+(1.)*(u_bc-uy)+(1.)*-uz);
      f7+=(1./72.)*rho*cu;

      //speed 8 ex=-1 ey=ez=1; w=1./72.
      cu=3.*((-1.)*-ux+(1.)*(u_bc-uy)+(1.)*-uz);
      f8+=(1./72.)*rho*cu;

      //speed 9 ex=1 ey=-1 ez=1
      cu=3.0*((1.)*-ux+(-1.)*(u_bc-uy)+(1.)*-uz);
      f9+=(1./72.)*rho*cu;

      //speed 10 ex=-1 ey=-1 ez=1
      cu=3.0*((-1.)*-ux+(-1.)*(u_bc-uy)+(1.)*-uz);
      f10+=(1./72.)*rho*cu;

      //speed 11 ex=1 ey=1 ez=-1
      cu=3.0*((1.)*-ux +(1.)*(u_bc-uy)+(-1.)*-uz);
      f11+=(1./72.)*rho*cu;

      //speed 12 ex=-1 ey=1 ez=-1
      cu=3.0*((-1.)*-ux+(1.)*(u_bc-uy)+(-1.)*-uz);
      f12+=(1./72.)*rho*cu;

      //speed 13 ex=1 ey=-1 ez=-1 w=1./72.
      cu=3.0*((1.)*-ux+(-1.)*(u_bc-uy)+(-1.)*-uz);
      f13+=(1./72.)*rho*cu;
      
      //speed 14 ex=ey=ez=-1 w=1./72.
      cu=3.0*((-1.)*-ux + (-1.)*(u_bc-uy) +(-1.)*-uz);
      f14+=(1./72.)*rho*cu;

      ux=0.; uy=u_bc; uz=0.;

    }//if(lnl[tid]==1)...

   
    //if(snl[tid]==1){
    if(((Y==0)||(Y==(Ny-1))||(Z==0)||(Z==(Nz-1))||(X==(Nx-1)))){
      ux=0.; uy=0.; uz=0.;
      // 1--2
      cu=f2; f2=f1; f1=cu;
      //3--4
      cu=f4; f4=f3; f3=cu;
      //5--6
      cu=f6; f6=f5; f5=cu;
      //7--14
      cu=f14; f14=f7; f7=cu;
      //8--13
      cu=f13; f13=f8; f8=cu;
      //9--12
      cu=f12; f12=f9; f9=cu;
      //10--11
      cu=f11; f11=f10; f10=cu;

    }else{

     

      //relax
      //speed 0 ex=ey=ez=0 w=2./9.
      float fEq;
      fEq=rho*(2./9.)*(1.-1.5*(ux*ux+uy*uy+uz*uz));
      f0=f0-omega*(f0-fEq);

      //speed 1 ex=1 ey=ez=0 w=1./9.
      cu=3.*(1.*ux);
      fEq=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
    		       1.5*(ux*ux+uy*uy+uz*uz));
      f1=f1-omega*(f1-fEq);

      //speed 2 ex=-1 ey=ez=0 w=1./9.
      cu=3.*((-1.)*ux);
      fEq=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
    		       1.5*(ux*ux+uy*uy+uz*uz));
      f2=f2-omega*(f2-fEq);

      //speed 3 ex=0 ey=1 ez=0 w=1./9.
      cu=3.*(1.*uy);
      fEq=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
    		       1.5*(ux*ux+uy*uy+uz*uz));
      f3=f3-omega*(f3-fEq);

      //speed 4 ex=0 ey=-1 ez=0 w=1./9.
      cu=3.*(-1.*uy);
      fEq=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
    		       1.5*(ux*ux+uy*uy+uz*uz));
      f4=f4-omega*(f4-fEq);

      //speed 5 ex=ey=0 ez=1 w=1./9.
      cu=3.*(1.*uz);
      fEq=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
    		       1.5*(ux*ux+uy*uy+uz*uz));
      f5=f5-omega*(f5-fEq);

      //speed 6 ex=ey=0 ez=-1 w=1./9.
      cu=3.*(-1.*uz);
      fEq=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
    		       1.5*(ux*ux+uy*uy+uz*uz));
      f6=f6-omega*(f6-fEq);

      //speed 7 ex=ey=ez=1 w=1./72.
      cu=3.*(ux+uy+uz);
      fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
    			1.5*(ux*ux+uy*uy+uz*uz));
      f7=f7-omega*(f7-fEq);

      //speed 8 ex=-1 ey=ez=1 w=1./72.
      cu=3.*(-ux+uy+uz);
      fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
    			1.5*(ux*ux+uy*uy+uz*uz));
      f8=f8-omega*(f8-fEq);

      //speed 9 ex=1 ey=-1 ez=1 w=1./72.
      cu=3.*(ux-uy+uz);
      fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
    			1.5*(ux*ux+uy*uy+uz*uz));
      f9=f9-omega*(f9-fEq);

      //speed 10 ex=-1 ey=-1 ez=1 w=1/72
      cu=3.*(-ux-uy+uz);
      fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
    			1.5*(ux*ux+uy*uy+uz*uz));
      f10=f10-omega*(f10-fEq);

      //speed 11 ex=1 ey=1 ez=-1 w=1/72
      cu=3.*(ux+uy-uz);
      fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
    			1.5*(ux*ux+uy*uy+uz*uz));
      f11=f11-omega*(f11-fEq);

      //speed 12 ex=-1 ey=1 ez=-1 w=1/72
      cu=3.*(-ux+uy-uz);
      fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
    			1.5*(ux*ux+uy*uy+uz*uz));
      f12=f12-omega*(f12-fEq);

      //speed 13 ex=1 ey=ez=-1 w=1/72
      cu=3.*(ux-uy-uz);
      fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
    			1.5*(ux*ux+uy*uy+uz*uz));
      f13=f13-omega*(f13-fEq);

      //speed 14 ex=ey=ez=-1 w=1/72
      cu=3.*(-ux-uy-uz);
      fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
    			1.5*(ux*ux+uy*uy+uz*uz));
      f14=f14-omega*(f14-fEq);
    }
    ux_d[tid]=ux; uy_d[tid]=uy; uz_d[tid]=uz; rho_d[tid]=rho;


    int X_t, Y_t, Z_t;
    int tid_t;

    //speed 0 ex=ey=ez=0
    fOut[tid]=f0;

    //speed 1 ex=1 ey=ez=0
    X_t=X+1; Y_t=Y; Z_t=Z;
    if(X_t==Nx) X_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[Nx*Ny*Nz+tid_t]=f1;

    //speed 2 ex=-1 ey=ez=0;
    X_t=X-1; Y_t=Y; Z_t=Z;
    if(X_t<0) X_t=(Nx-1);
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[2*Nx*Ny*Nz+tid_t]=f2;

    //speed 3 ex=0 ey=1 ez=0
    X_t=X; Y_t=Y+1; Z_t=Z;
    if(Y_t==Ny) Y_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[3*Nx*Ny*Nz+tid_t]=f3;

    //speed 4 ex=0 ey=-1 ez=0
    X_t=X; Y_t=Y-1; Z_t=Z;
    if(Y_t<0) Y_t=(Ny-1);
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[4*Nx*Ny*Nz+tid_t]=f4;

    //speed 5 ex=ey=0 ez=1
    X_t=X; Y_t=Y; Z_t=Z+1;
    if(Z_t==Nz) Z_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[5*Nx*Ny*Nz+tid_t]=f5;

    //speed 6 ex=ey=0 ez=-1
    X_t=X; Y_t=Y; Z_t=Z-1;
    if(Z_t<0) Z_t=(Nz-1);
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[6*Nx*Ny*Nz+tid_t]=f6;

    //speed 7 ex=ey=ez=1
    X_t=X+1; Y_t=Y+1; Z_t=Z+1;
    if(X_t==Nx) X_t=0;
    if(Y_t==Ny) Y_t=0;
    if(Z_t==Nz) Z_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[7*Nx*Ny*Nz+tid_t]=f7;

    //speed 8 ex=-1 ey=1 ez=1
    X_t=X-1; Y_t=Y+1; Z_t=Z+1;
    if(X_t<0) X_t=(Nx-1);
    if(Y_t==Ny) Y_t=0;
    if(Z_t==Nz) Z_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[8*Nx*Ny*Nz+tid_t]=f8;

    //speed 9 ex=1 ey=-1 ez=1
    X_t=X+1; Y_t=Y-1; Z_t=Z+1;
    if(X_t==Nx) X_t=0;
    if(Y_t<0) Y_t=(Ny-1);
    if(Z_t==Nz) Z_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[9*Nx*Ny*Nz+tid_t]=f9;

    //speed 10 ex=-1 ey=-1 ez=1
    X_t=X-1; Y_t=Y-1; Z_t=Z+1;
    if(X_t<0) X_t=(Nx-1);
    if(Y_t<0) Y_t=(Ny-1);
    if(Z_t==Nz) Z_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[10*Nx*Ny*Nz+tid_t]=f10;

    //speed 11 ex=1 ey=1 ez=-1
    X_t=X+1; Y_t=Y+1; Z_t=Z-1;
    if(X_t==Nx) X_t=0;
    if(Y_t==Ny) Y_t=0;
    if(Z_t<0) Z_t=(Nz-1);
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[11*Nx*Ny*Nz+tid_t]=f11;

    //speed 12 ex=-1 ey=1 ez=-1
    X_t=X-1; Y_t=Y+1; Z_t=Z-1;
    if(X_t<0) X_t=(Nx-1);
    if(Y_t==Ny) Y_t=0;
    if(Z_t<0) Z_t=(Nz-1);
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[12*Nx*Ny*Nz+tid_t]=f12;

    //speed 13 ex=1 ey=-1 ez=-1
    X_t=X+1; Y_t=Y-1; Z_t=Z-1;
    if(X_t==Nx) X_t=0;
    if(Y_t<0) Y_t=(Ny-1);
    if(Z_t<0) Z_t=(Nz-1);
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[13*Nx*Ny*Nz+tid_t]=f13;

    //speed 14 ex=ey=ez=-1
    X_t=X-1; Y_t=Y-1; Z_t=Z-1;
    if(X_t<0) X_t=(Nx-1);
    if(Y_t<0) Y_t=(Ny-1);
    if(Z_t<0) Z_t=(Nz-1);
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[14*Nx*Ny*Nz+tid_t]=f14;



  }//if(X<Nx...


}







void ldc_D3Q15_LBGK_ts_cuda(float * fOut, const float * fIn, 
			    float * rho, float * ux, float * uy, float * uz,
			    const float u_bc, const float omega,
			    const int Nx, const int Ny, const int Nz){

  dim3 BLOCKS(TPB,1,1);
  dim3 GRIDS((Nx*Ny*Nz+TPB-1)/TPB,1,1);
  ldc_D3Q15_LBGK_ts<<<GRIDS,BLOCKS>>>(fOut,fIn,rho,ux,uy,uz,
				      u_bc,omega,Nx,Ny,Nz);

}


void ldc_D3Q19_LBGK_ts_cuda(float * fOut, const float * fIn, 
			    float * rho, float * ux, float * uy, float * uz,
			    const float u_bc, const float omega,
			    const int Nx, const int Ny, const int Nz){

  dim3 BLOCKS(TPB,1,1);
  dim3 GRIDS((Nx*Ny*Nz+TPB-1)/TPB,1,1);
  ldc_D3Q19_LBGK_ts<<<GRIDS,BLOCKS>>>(fOut,fIn,rho,ux,uy,uz,
				      u_bc,omega,Nx,Ny,Nz);

}

void ldc_D3Q19_MRT_ts_cuda(float * fOut, const float * fIn, 
			    float * rho, float * ux, float * uy, float * uz,
			    const float u_bc, const float* omega_op,
			    const int Nx, const int Ny, const int Nz){

  dim3 BLOCKS(TPB,1,1);
  dim3 GRIDS((Nx*Ny*Nz+TPB-1)/TPB,1,1);
  ldc_D3Q19_MRT_ts<<<GRIDS,BLOCKS>>>(fOut,fIn,rho,ux,uy,uz,
				      u_bc,omega_op,Nx,Ny,Nz);

}
