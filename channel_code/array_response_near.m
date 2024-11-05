function c = array_response_near(N,x,y,z,lambda,d)


NY=sqrt(N);
NZ=NY;
for ny=1:NY
    for nz=1:NZ
        R(ny,nz)=norm([0,(ny-1)*d,(nz-1)*d]-[x,y,z]);
        C(ny,nz)=1/sqrt(N)*exp(1i*2*pi/lambda*R(ny,nz));
    end
end

c=reshape(C,N,1);

end

