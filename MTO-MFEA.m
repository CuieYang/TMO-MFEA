% This trial version of the Multi-Objective Multifactorial Evolutionary Algorithm (MO-MFEA: which is based on NSGA-II) has been
% developed to handle only two "continuous" multiobjective problems (or tasks) at a time.
% Each problem may comprise upto three objective functions.
% Generalization to many-tasking can be done based on this framework.
function data_MOMFEA=MOMFEA_v01(fname,pop,rmp,gen,muc,mum,reps,maxFEs,numRecords)
gen = ceil(maxFEs/pop);
no_of_tasks=2;
if mod(pop,2)~=0
    pop=pop+1;
end
pop1=pop/2;
pop2=pop1;
fnceval_calls = zeros(1,reps);
EvBestFitness = zeros(no_of_tasks*reps,numRecords);    % best fitness found
TotalEvaluations=zeros(reps,gen);               % total number of task evaluations so fer
bestobjRecord = Inf(reps, numRecords, no_of_tasks);

for rep=1:reps
    
    [L1,U1,L2,U2,M1,M2,opt1,opt2,dim1,dim2] = Initiate(fname);
    dim=max([dim1,dim2]);
    for i=1:pop
        population(i)=Chromosome;
        population(i)=initialize(population(i),dim);
        if i<=pop1
            population(i).skill_factor=1;
        else
            population(i).skill_factor=2;
        end
    end
    for i=1:pop
        population(i)=evaluate(population(i),L1,U1,L2,U2,dim1,dim2,M1,M2,opt1,opt2,fname);
    end
    fnceval_calls(rep)=fnceval_calls(rep) + pop;
    TotalEvaluations(rep,1)=fnceval_calls(rep);
    
    population_T1=population([population.skill_factor]==1);
    population_T2=population([population.skill_factor]==2);
    no_of_objs_T1 = length(population_T1(1).objs_T1);
    no_of_objs_T2 = length(population_T2(1).objs_T2);
    [population_T1,frontnumbers]=SolutionComparison.nondominatedsort(population_T1,pop1,no_of_objs_T1);
    [population_T1,~]=SolutionComparison.diversity(population_T1,frontnumbers,pop1,no_of_objs_T1);
    [population_T2,frontnumbers]=SolutionComparison.nondominatedsort(population_T2,pop2,no_of_objs_T2);
    [population_T2,~]=SolutionComparison.diversity(population_T2,frontnumbers,pop2,no_of_objs_T2);
    
    E_pop_T1 = population_T1([population_T1.front]==1);
    E_pop_T2 = population_T2([population_T2.front]==1);
    
    T1_data=[];
    T2_data=[];
    for i=1:length(E_pop_T1)
        T1_data = [T1_data;E_pop_T1(i).objs_T1];
    end
    for i=1:length(E_pop_T2)
        T2_data = [T2_data;E_pop_T2(i).objs_T2];
    end
    IGD = Evalution(T1_data,T2_data,fname);
    for i = 1:no_of_tasks
        EvBestFitness(i+2*(rep-1),:)= IGD(i);
        bestobjRecord(rep, :, i) = IGD(i);
    end
    population(1:pop1) = population_T1;
    population(pop1+1:pop) = population_T2;
    population1=[];
    population2=[];
    fitness1=[];
    fitness2=[];
    for i=1:pop1
        population1=[population1;population(i).rnvec(1:dim1)];
        fitness1 = [fitness1;population(i).objs_T1];
    end
    for i=1:pop2
        population2=[population2;population(i+pop1).rnvec(1:dim2)];
        fitness2 = [fitness2;population(i+pop1).objs_T2];
    end
    nSel        = 5;        % number of selected solutions in variable clustering
    nPer        = 10;       % number of perturbations in variable clustering
    [PV1,DV1,fnceval_calls(rep)] = VariableClustering(population1,fitness1,nSel,nPer,L1,U1,dim1,M1,opt1,fname,1,fnceval_calls(rep));
    [PV2,DV2,fnceval_calls(rep)] = VariableClustering(population2,fitness2,nSel,nPer,L2,U2,dim2,M2,opt2,fname,2,fnceval_calls(rep));

    XC_dim1 = length(PV1);  % diversity related
    XC_dim2 = length(PV2);
    XC_dim = max(XC_dim1,XC_dim2);
    XS_dim1 = length(DV1);   % convergence-related
    XS_dim2 = length(DV2);
    XS_dim = max(XS_dim1,XS_dim2);
    
    population1 = population1(:,DV1);  
    population2 = population2(:,DV2);
    for i=1:pop1
        population(i).XSnvec = population(i).rnvec(1:XS_dim);   
        population(i).XCnvec = population(i).rnvec(1:XC_dim);
    end
    for i=1+pop1:pop
        population(i).XSnvec = population(i).rnvec(1:XS_dim);
        population(i).XCnvec = population(i).rnvec(1:XC_dim);
    end
    

    for generation=1:gen
        rndlist=randperm(pop);
        population=population(rndlist);
        for i = 1:pop % Performing binary tournament selection to create parent pool
            parent(i)=Chromosome();
            p1=1+round(rand(1)*(pop-1));
            p2=1+round(rand(1)*(pop-1));
            if population(p1).rank < population(p2).rank
                parent(i) = population(p1);
            elseif population(p1).rank == population(p2).rank
                if rand(1) <= 0.5
                    parent(i) = population(p1);
                else
                    parent(i) = population(p2);
                end
            else
                parent(i) = population(p2);
            end
        end
        count=1;
        grouprandperform = randperm(XS_dim);
        Dgroup1 =grouprandperform(1:(floor(XS_dim/2)));
        Dgroup2 =grouprandperform(1+(floor(XS_dim/2)):end);
%                 Dgroup1 = 1:XS_dim;
        Dparent1 = randperm(pop);
        Dparent2 = randperm(pop);
        Pparent =  randperm(pop);
        for i=1:2:pop-1 % Create offspring population via mutation and crossover
            child(count)=Chromosome;
            child(count+1)=Chromosome;
            p1=Dparent1(i);
            p2=Dparent1(i+1);
            if parent(p1).skill_factor==parent(p2).skill_factor || rand(1)<rmp
                [child(count).XSnvec(Dgroup1),child(count+1).XSnvec(Dgroup1)]=Evolve.crossover(parent(p1).XSnvec(Dgroup1),parent(p2).XSnvec(Dgroup1),muc,length(Dgroup1));
                child(count).XSnvec(Dgroup1) = Evolve.mutate(child(count).XSnvec(Dgroup1),mum,1/dim,length(Dgroup1));
                child(count+1).XSnvec(Dgroup1)=Evolve.mutate(child(count+1).XSnvec(Dgroup1),mum,1/dim,length(Dgroup1));
                sf1 = round(rand(1));
                sf2 = round(rand(1));
                if sf1==1
                    child(count).skill_factor=parent(p1).skill_factor;
                else
                    child(count).skill_factor=parent(p2).skill_factor;
                end
                if sf2==1
                    child(count+1).skill_factor=parent(p2).skill_factor;
                else
                    child(count+1).skill_factor=parent(p1).skill_factor;
                end
            else
                child(count).XSnvec(Dgroup1) = Evolve.mutate(parent(p1).XSnvec(Dgroup1),mum,1,length(Dgroup1));
                child(count+1).XSnvec(Dgroup1)=Evolve.mutate(parent(p2).XSnvec(Dgroup1),mum,1,length(Dgroup1));
                child(count).skill_factor=parent(p1).skill_factor;
                child(count+1).skill_factor=parent(p2).skill_factor;
            end
            
            p1=Dparent2(i);
            p2=Dparent2(i+1);
            if parent(p1).skill_factor==parent(p2).skill_factor || rand(1)<rmp
                [child(count).XSnvec(Dgroup2),child(count+1).XSnvec(Dgroup2)]=Evolve.crossover(parent(p1).XSnvec(Dgroup2),parent(p2).XSnvec(Dgroup2),muc,length(Dgroup2));
                child(count).XSnvec(Dgroup2) = Evolve.mutate(child(count).XSnvec(Dgroup2),mum,1/dim,length(Dgroup2));
                child(count+1).XSnvec(Dgroup2)=Evolve.mutate(child(count+1).XSnvec(Dgroup2),mum,1/dim,length(Dgroup2));
            else
                child(count).XSnvec(Dgroup2) = Evolve.mutate(parent(p1).XSnvec(Dgroup2),mum,1,length(Dgroup2));
                child(count+1).XSnvec(Dgroup2)=Evolve.mutate(parent(p2).XSnvec(Dgroup2),mum,1,length(Dgroup2));
            end
            p1=Pparent(i);
            p2=Pparent(i+1);
            [child(count).XCnvec,child(count+1).XCnvec]=Evolve.crossover(parent(p1).XCnvec,parent(p2).XCnvec,muc,XC_dim);
            child(count).XCnvec = Evolve.mutate(child(count).XCnvec,mum,1,XC_dim);
            child(count+1).XCnvec=Evolve.mutate(child(count+1).XCnvec,mum,1,XC_dim);
            count=count+2;
        end
        for i=1:pop
            xs = child(i).XSnvec;
            xc = child(i).XCnvec;
            if child(i).skill_factor == 1
                child(i).rnvec(PV1) = xc(1:XC_dim1);
                child(i).rnvec(DV1) = xs(1:XS_dim1);
            else
                child(i).rnvec(PV2) = xc(1:XC_dim2);
                child(i).rnvec(DV2) = xs(1:XS_dim2);
            end
        end

        for i=1:pop
            child(i)=evaluate(child(i),L1,U1,L2,U2,dim1,dim2,M1,M2,opt1,opt2,fname);
        end
        fnceval_calls(rep)=fnceval_calls(rep) + pop;
        TotalEvaluations(rep,generation)=fnceval_calls(rep);
        population=reset(population,pop);
        intpopulation(1:pop)=population;
        intpopulation(pop+1:2*pop)=child;
        intpopulation_T1=intpopulation([intpopulation.skill_factor]==1);
        intpopulation_T2=intpopulation([intpopulation.skill_factor]==2);
        T1_pop=length(intpopulation_T1);
        T2_pop=length(intpopulation_T2);
        [intpopulation_T1,frontnumbers]=SolutionComparison.nondominatedsort(intpopulation_T1,T1_pop,no_of_objs_T1);
        [intpopulation_T1,~]=SolutionComparison.diversity(intpopulation_T1,frontnumbers,T1_pop,no_of_objs_T1);
        [intpopulation_T2,frontnumbers]=SolutionComparison.nondominatedsort(intpopulation_T2,T2_pop,no_of_objs_T2);
        [intpopulation_T2,~]=SolutionComparison.diversity(intpopulation_T2,frontnumbers,T2_pop,no_of_objs_T2);
        population(1:pop1) = intpopulation_T1(1:pop1);
        population(pop1+1:pop) = intpopulation_T2(1:pop2);

        E_pop_T1 = intpopulation_T1([intpopulation_T1.front]==1);
        E_pop_T2 = intpopulation_T2([intpopulation_T2.front]==1);
        T1_data=[];
        T2_data=[];
        for i=1:length(E_pop_T1)
            T1_data = [T1_data;E_pop_T1(i).objs_T1];
        end
        for i=1:length(E_pop_T2)
            T2_data = [T2_data;E_pop_T2(i).objs_T2];
        end
        IGD = Evalution(T1_data,T2_data,fname);
        for i=1:no_of_tasks
            
            rindex = floor(fnceval_calls(rep)/(maxFEs/numRecords))+1;
            if rindex <=numRecords
                EvBestFitness(i+2*(rep-1),rindex) = IGD(i);
                bestobjRecord(rep, rindex, i) = IGD(i);
            end
        end
        if(fnceval_calls(rep) >= maxFEs)
            break;
        end
    end
end
data_MOMFEA.bestobjRecord = bestobjRecord;
data_MOMFEA.EvBestFitness = EvBestFitness;

end