function a = array_response_far(N,theta)

a=zeros(N,1); 
 
for n=1:N
a(n)=1/sqrt(N)*exp(1i*pi*(n-1)*theta);
end

end

