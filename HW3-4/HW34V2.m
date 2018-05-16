clear all; clc;

load('Prior_2.mat'); load('Alpha.mat'); load('blkzz.mat');
load('TrainingSamplesDCT_subsets_8.mat');

pc = 0.1919;
pg = 1-pc;
%-------------Calculate ML Mean of both classes-----------------%

[cheetah_samples, dim] = size(D4_FG); [grass_samples, dim] = size(D4_BG);
cheetah_muML = sum(D4_FG)/cheetah_samples;
grass_muML = sum(D4_BG)/grass_samples;

%-------------Calculate ML Covariance of both classes-----------------%

cheetah_covML = zeros(64,64);
grass_covML = zeros(64,64);

for i = 1:cheetah_samples
    temp1 = (D4_FG(i,:) - cheetah_muML)';
    temp2 = (D4_FG(i,:) - cheetah_muML);
    cheetah_covML = cheetah_covML+(temp1*temp2);
end
cheetah_covML = cheetah_covML/cheetah_samples;

for i = 1:grass_samples
    temp1 = (D4_BG(i,:) - grass_muML)';
    temp2 = (D4_BG(i,:) - grass_muML);
    grass_covML = grass_covML+(temp1*temp2);
end
grass_covML = grass_covML/grass_samples;

%-------------BDRs-----------------%

mask = imread('cheetah_mask.bmp');
Ch_mask = double(mask)/255;

Ch_image = imread('cheetah.bmp');
Ch_image = im2double(Ch_image);
[x,y] = size(Ch_image);

a = alpha;

ML_err(1:9)=0;
MAP_err(1:9)=0;
Pred_err(1:9)=0;

for k=1:length(a)
    
    % update and initialize our variables
    idx = 1;
    E_prior = diag(W0)*a(k);
    
    cheetah_mun = E_prior*inv(E_prior+cheetah_covML/cheetah_samples)*...
        cheetah_muML'+cheetah_covML*inv(E_prior+cheetah_covML/cheetah_samples)*...
        mu0_FG'/cheetah_samples;
    
    grass_mun = E_prior*inv(E_prior+grass_covML/grass_samples)*grass_muML'+...
        grass_covML*inv(E_prior+grass_covML/grass_samples)*mu0_BG'/grass_samples;
    
    En_cheetah = E_prior*inv(E_prior+cheetah_covML/cheetah_samples)*...
        cheetah_covML/cheetah_samples;
    
    En_grass = E_prior*inv(E_prior+grass_covML/grass_samples)*grass_covML/grass_samples;
    
    mask_ML = zeros(x,y);
    mask_MAP = zeros(x,y);
    mask_Pred = zeros(x,y);
    
    map_errors=0;
    ml_errors=0;
    pred_errors = 0;
    det_map = 0; det_ml = 0; det_pred = 0;
    fal_map = 0; fal_ml = 0; fal_pred = 0;
    for i = 1:x-7
        for j = 1:y-7
            if Bayes(blkzz(idx,:),cheetah_mun',En_cheetah+cheetah_covML,64,pc)>Bayes(blkzz(idx,:),grass_mun', En_grass+grass_covML,64,pg)
                mask_Pred(i,j) = 1;
            end
            if Bayes(blkzz(idx,:),cheetah_mun',cheetah_covML,64,pc)>Bayes(blkzz(idx,:),grass_mun', grass_covML,64,pg)
                mask_MAP(i,j) = 1;
            end
            if Bayes(blkzz(idx,:),cheetah_muML,cheetah_covML,64,pc)>Bayes(blkzz(idx,:),grass_muML, grass_covML,64,pg)
                mask_ML(i,j)=1;
            end
            idx = idx+1;
        end
    end
    
%     for r = 1:255
%         for s = 1:270
%             if Ch_mask(r,s)~=mask_Pred(r,s)
%                 pred_errors=pred_errors+1;
%             end
%             if Ch_mask(r,s)~=mask_MAP(r,s)
%                 map_errors=map_errors+1;
%             end
%             if Ch_mask(r,s)~=mask_ML(r,s)
%                 ml_errors=ml_errors+1;
%             end
%         end
%     end
%     
    for u = 1:255
        for v = 1:270
            if Ch_mask(u,v) == 1 && mask_Pred(u,v) == 1
                det_pred = det_pred+1;
            end
            if Ch_mask(u,v) == 0 && mask_Pred(u,v) == 1
                fal_pred = fal_pred+1;
            end
            if Ch_mask(u,v) == 1 && mask_ML(u,v) == 1
                det_ml = det_ml+1;
            end
            if Ch_mask(u,v) == 0 && mask_ML(u,v) == 1
                fal_ml = fal_ml+1;
            end
            if Ch_mask(u,v) == 1 && mask_MAP(u,v) == 1
                det_map = det_map+1;
            end
            if Ch_mask(u,v) ==0 && mask_MAP(u,v) == 1
                fal_map = fal_map+1;
            end
        end
    end
    Pred_err(k) = (1-det_pred/(255*270*pc))*pc+pg*(fal_pred/(255*270*pg));
    ML_err(k) = (1-det_ml/(255*270*pc))*pc+pg*(fal_ml/(255*270*pg));
    MAP_err(k) = (1-det_map/(255*270*pc))*pc+pg*(fal_map/(255*270*pg));
    
%     ML_err(k) = ml_errors/(255*270);
%     MAP_err(k) = map_errors/(255*270);
%     Pred_err(k) = pred_errors/(255*270);
end

figure;
semilogx(a, ML_err, '-.r', 'LineWidth', 1.5), hold on
semilogx(a, Pred_err, '-ob', 'LineWidth', 1.5), hold on
semilogx(a, MAP_err, '--xg', 'LineWidth', 1.5)
xlabel('\alpha'), ylabel('Probability of Error')
title('Probability of Error vs. \alpha')
legend('ML','BPE', 'MAP')

    

    
    
    
    

function bdr = Bayes(x, mu, E, d, p)

bdr = -0.5*(x-mu)*inv(E)*(x-mu)'-0.5*log((2*pi)^(d)*det(E))+log(p);

end

