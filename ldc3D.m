% ldc3D.m

clear
clc
close('all');

lattice_selection=1;
% 1 = D3Q15
% 2 = D3Q19


dynamics = 1;
% 1 = LBGK
% 3 = MRT

Num_ts = 10000;
ts_rep_freq = 1000;
plot_freq = Num_ts+1;

Re = 1000;
dt = 2.51953e-4;
Ny_divs = 96;

Lx_p = 1;
Ly_p = 1;
Lz_p = 1;

fluid = 4;

switch fluid
    case 1
        rho_p = 1260;
        nu_p = 1.49/rho_p;
        
    case 2
        rho_p = 965.3;
        nu_p = 0.06/rho_p;
        
    case 3
        rho_p = 1000;
        nu_p = 1e-3/rho_p;
        
    case 4
        rho_p = 1000;
        nu_p = 0.001;
        
end

Lo = Ly_p;
Uo = nu_p*Re/Lo;
To = Lo/Uo;
Uavg = Uo;

Ld = 1; Td = 1; Ud = (To/Lo)*Uavg;
nu_d = 1/Re;

dx = 1/(Ny_divs-1);
u_lbm = (dt/dx)*Ud;
nu_lbm=(dt/(dx^2))*nu_d;
omega = get_BGK_Omega(nu_lbm);

u_conv_fact = (dt/dx)*(To/Lo);
t_conv_fact = (dt*To);
l_conv_fact = dx*Lo;
p_conv_fact = ((l_conv_fact/t_conv_fact)^2)*(1/3); %<-- this should work within the fluid...


rho_lbm = rho_p;
rho_out = rho_lbm;

Ny = ceil((Ny_divs-1)*(Ly_p/Lo))+1;
Nx = ceil((Ny_divs-1)*(Lx_p/Lo))+1;
Nz = ceil((Ny_divs-1)*(Lz_p/Lo))+1;

switch lattice_selection
    
    case 1
        [w,ex,ey,ez,bb_spd]=D3Q15_lattice_parameters();
        lattice = 'D3Q15';
    case 2
        [w,ex,ey,ez,bb_spd]=D3Q19_lattice_parameters();
        lattice = 'D3Q19';
    case 3
        [w,ex,ey,ez,bb_spd]=D3Q27_lattice_parameters();
        lattice = 'D3Q27';
        
end

switch dynamics
    
 
    case 3 % MRT
      
        M = getMomentMatrix(lattice);
        S = getEwMatrixMRT(lattice,omega);
        omega_op = M\(S*M);
        
        
end

nnodes=Nx*Ny*Nz;
numSpd = length(w);

fprintf('Number of Lattice-points = %d.\n',nnodes);
fprintf('Number of time-steps = %d. \n',Num_ts);
fprintf('LBM viscosity = %g. \n',nu_lbm);
fprintf('LBM relaxation parameter (omega) = %g. \n',omega);
fprintf('LBM flow Mach number = %g. \n',u_lbm);
fprintf('dx*dx = %g.\n',dx*dx);
fprintf('dt = %g.\n',dt);

input_string = sprintf('Do you wish to continue? [Y/n] \n');

run_dec = input(input_string,'s');

if ((run_dec ~= 'n') && (run_dec ~= 'N'))
    
    fprintf('Ok! Cross your fingers!! \n');
    
    
    s = system('rm *.lbm');
    params = fopen('params.lbm','w');
    fprintf(params,'%d \n',Num_ts);
    fprintf(params, '%d \n',Nx);
    fprintf(params, '%d \n',Ny);
    fprintf(params, '%d \n',Nz);
    fprintf(params, '%d \n',numSpd);
    fprintf(params, '%f \n',u_lbm);
    fprintf(params, '%f \n',rho_lbm);
    fprintf(params, '%f \n',omega);
    fprintf(params, '%d \n',dynamics);
    fprintf(params, '%d \n',ts_rep_freq);
    fprintf(params, '%d \n',plot_freq);
    
    if(dynamics==3)
        save('M.lbm','omega_op','-ascii');
    end
    
    tic;
    system('./clbm_ldc3D');
    run_time = toc;
    fprintf('Lattice Point Updates per second = %g. \n',Nx*Ny*Nz*Num_ts/run_time);
    
else
    fprintf('Run aborted.  Better luck next time!\n');
end