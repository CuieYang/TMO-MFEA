function [PV,DV,call_evaluations] = VariableClustering(Population,ObjValue,nSel,nPer,L,U,dim,M,opt,fname,task,call_evaluations)
% Detect the kind of each decision variable
[N,D] = size(Population);
ND    = T_ENS(ObjValue,1) == 1;
fmin  = min(ObjValue(ND,:),[],1);
fmax  = max(ObjValue(ND,:),[],1);

%% Calculate the proper values of each decision variable
Angle  = zeros(D,nSel);
RMSE   = zeros(D,nSel);
Sample = randi(N,1,nSel);
for i = 1 : D
    % Generate several random solutions by perturbing the i-th dimension
    Decs      = repmat(Population(Sample,:),nPer,1);
    Decs(:,i) = rand(size(Decs,1),1);
    nObjValue = [];
    n_size = size(Decs,1);  
    for k=1:n_size
        xtemp=Decs(k,1:dim);
        x=L+xtemp.*(U-L);
        [objs,~]=benchmark(x,M,opt,fname,dim,task);
        nObjValue = [nObjValue;objs];
    end
    for j = 1 : nSel
        % Normalize the objective values of the current perturbed solutions
        Points = nObjValue(j:nSel:end,:);
        Points = (Points-repmat(fmin,size(Points,1),1))./repmat(fmax-fmin,size(Points,1),1);
        Points = Points - repmat(mean(Points,1),nPer,1);
        % Calculate the direction vector of the determining line
        [~,~,V] = svd(Points);
        Vector  = V(:,1)'./norm(V(:,1)');
        % Calculate the root mean square error
        error = zeros(1,nPer);
        for k = 1 : nPer
            error(k) = norm(Points(k,:)-sum(Points(k,:).*Vector)*Vector);
        end
        RMSE(i,j) = sqrt(sum(error.^2));
        % Calculate the angle between the line and the hyperplane
        normal     = ones(1,size(Vector,2));
        sine       = abs(sum(Vector.*normal,2))./norm(Vector)./norm(normal);
        Angle(i,j) = real(asin(sine)/pi*180);
    end
end
call_evaluations = call_evaluations+n_size*D;
%% Detect the kind of each decision variable
VariableKind = (mean(RMSE,2)<1e-2)';
result       = kmeans(Angle,2)';
if any(result(VariableKind)==1) && any(result(VariableKind)==2)
    if mean(mean(Angle(result==1&VariableKind,:))) > mean(mean(Angle(result==2&VariableKind,:)))
        VariableKind = VariableKind & result==1;
    else
        VariableKind = VariableKind & result==2;
    end
end
PV = find(~VariableKind);
DV = find(VariableKind);
end