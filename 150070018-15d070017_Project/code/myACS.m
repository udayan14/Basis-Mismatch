%% Implementation of Alternating Convex Search problem

function [x,theta] = myACS(y,phi,N,Q)
alpha = 0.1;
beta = 0.1;
TOL = 1e-5;
rel_tol = 0.001;
theta = zeros(Q*N/2,1);
t = 0;
dict = getDict(theta,N,Q);
Mul = phi*dict;
lambda = alpha * norm((Mul)' * y,Inf);
[x,~]=l1_ls(Mul,y,lambda,rel_tol,true);
prev = norm(y - phi * getDict(theta,N,Q) * x).^2 + lambda * norm(x,1);  %stopping criterion

while true
    dict = getDict(theta,N,Q);
    Mul = phi*dict;
    lambda = alpha * norm((Mul)' * y,Inf);
    [x,~]=l1_ls(Mul,y,lambda,rel_tol,true);
    k = beta * norm(x);
    S = find(abs(x)>=k);
    for i = S'
        f = @(in) minimizer(y,phi,getDict(theta,N,Q),x,i,in,Q,N); 
        estimate = fminbnd(f,-1/(2*Q*N),1/(2*Q*N));
        theta(i) = estimate;
    end
    curr = norm(y - phi * getDict(theta,N,Q) * x).^2 + lambda * norm(x,1);
    if (abs((prev-curr)/prev) < TOL) || t>100
        break
    end
    prev = curr;
    t = t+1;
end
end

function dict = getDict(theta,N,Q)
    arr = 0:N-1;
    dict = zeros(N,Q*N);
    for i = 1 : Q*N
        if i <= Q*N/2
            dict(:,i) = cos(2*pi*arr*((i-1)/(Q*N) + theta(i)))*(2/N)^0.5;
        else
            dict(:,i) = -1 * sin(2*pi*arr*((Q*N-i)/(Q*N)-theta(Q*N-i+1)))*(2/N)^0.5;
        end
    end
end

function val = minimizer(y,phi,dict,x,i,theta_val,Q,N)
    arr = 0:N-1;   
    
    if i <= Q*N/2
        dict(:,i) = cos(2*pi*arr*((i-1)/(Q*N) + theta_val))*(2/N)^0.5;
    else
        dict(:,i) = -sin(2*pi*arr*((Q*N-i)/(Q*N)-theta_val))*(2/N)^0.5;
    end
    val = norm(y-phi*dict*x).^2;
end