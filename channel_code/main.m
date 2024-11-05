clear all
close all

N=144; % 12*12 ,RIS
M=64; % BS

L=5;
P=5;

f=10e9; % system frequency 10GHz
lambda=3e8/f; % antenna wavelength
d=lambda/2; % inter-element spacing for virtual channel representation
sigma_2=1; % noise power
P_max=1; % transmit power




realization=1;
for reali=1:realization
    alpha=1/sqrt(2)*(normrnd(0,1,L,1)+1i*normrnd(0,1,L,1)); % path gains
    phi_t=unifrnd(-1,1,L,1); % BS path DoA  phi_t=cos(phi_true)
    phi_r=unifrnd(-1,1,L,1); % RIS path AoA azimuth   phi_r=sin(phi_true)sin(theta)
    theta_r=unifrnd(-1,1,L,1); % RIS path AoA elevation  theta_r=cos(theta)
    H=zeros(N,M);

    for l=1:L
        H=H+sqrt(N*M/L)*alpha(l)*kron(array_response_far(sqrt(N),phi_r(l)),array_response_far(sqrt(N),theta_r(l)))*array_response_far(M,phi_t(l))';
    end
    beta=1/sqrt(2)*(normrnd(0,1,P,1)+1i*normrnd(0,1,P,1)); % path gains
    x=unifrnd(1,60,P,1);
    y=unifrnd(1,50,P,1);
    z=unifrnd(-25,25,P,1);
    h=zeros(N,1);
    for p=1:P
        h=h+sqrt(N/P)*beta(p)*array_response_near(N,x(p),y(p),z(p),lambda,d);
    end

    noise=1/sqrt(2)*(normrnd(0,1,M,1)+1i*normrnd(0,1,M,1));
    %% exhaustive search
    x_grid=1:0.5:25;
    y_grid=1:0.5:25;
    z_grid=-25:0.5:25;
    for xx=1:length(x_grid)
        for yy=1:length(y_grid)
            for zz=1:length(z_grid)
                c=sqrt(N)*array_response_near(N,x_grid(xx),y_grid(yy),z_grid(zz),lambda,d);
                C_w=c*c';
                gain_record(xx,yy,zz)=abs(array_response_far(M,phi_t(1))'*H'*diag(h)*C_w*diag(h')*H*array_response_far(M,phi_t(1)))^2;
            end
        end
    end

    %% Bayesian optimization-based beam training


    % y_echo=H'*diag(h)*V*diag(h')*H
end