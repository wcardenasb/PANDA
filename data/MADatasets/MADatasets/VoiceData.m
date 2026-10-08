% index for the Voice dataset
clear all; close all; clc;
load('Voice.mat')
N = length(y);
Nk = 15;
Ntr = round(N*0.7);
idxtr = zeros(Ntr, Nk);
idxte = zeros(N-Ntr, Nk);
for i = 1:Nk
    aux = randperm(N)-1;
    idxtr(:,i) = aux(1:Ntr); 
    idxte(:,i) = aux(Ntr+1:end); 
end

save('Voice.mat', 'idxtr', 'idxte', '-append');

