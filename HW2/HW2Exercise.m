clear all, clc

load('TrainingSamplesDCT_8_new.mat');
zig=load('Zig-Zag Pattern.txt');
zig=zig+1;

sizeFG = numel(TrainsampleDCT_FG);
sizeBG = numel(TrainsampleDCT_BG);
total = sizeFG + sizeBG;

pc = 250/(250+1053);
pg = 1 - pc;
train_fg = [TrainsampleDCT_FG(:,1) TrainsampleDCT_FG(:,3)...
    TrainsampleDCT_FG(:,10) TrainsampleDCT_FG(:,38)...
    TrainsampleDCT_FG(:,39) TrainsampleDCT_FG(:,51) TrainsampleDCT_FG(:,57)...
    TrainsampleDCT_FG(:,58)];
train_bg = [TrainsampleDCT_BG(:,1) TrainsampleDCT_BG(:,3)...
    TrainsampleDCT_BG(:,10) TrainsampleDCT_BG(:,38)...
    TrainsampleDCT_BG(:,39) TrainsampleDCT_BG(:,51) TrainsampleDCT_BG(:,57)...
    TrainsampleDCT_BG(:,58)];

mu_f8 = mean(train_fg)';
mu_b8 = mean(train_bg)';
covar_f8 = cov(train_fg);
covar_b8 = cov(train_bg);

sigma_c64(64,64)= 0;
sigma_c8(8,8) = 0;
for s = 1:250
    tmp1 = (TrainsampleDCT_FG(s,:)-mean(TrainsampleDCT_FG));
    tmp2 = tmp1'*tmp1;
    sigma_c64 = sigma_c64+tmp2;
    tmp11 = train_fg(s,:)-mean(train_fg);
    tmp22 = tmp11'*tmp11;
    sigma_c8 = sigma_c8 + tmp22;
end
sigma_c64 = sigma_c64*1/250;
sigma_c8 = sigma_c8*1/250;

sigma_g64(64,64) = 0;
sigma_g8(8,8) = 0;
for s = 1:1053
    tmp1 = (TrainsampleDCT_BG(s,:)-mean(TrainsampleDCT_BG));
    tmp2 = tmp1'*tmp1;
    sigma_g64 = sigma_g64+tmp2;
    tmp11 = train_bg(s,:)-mean(train_bg);
    tmp22 = tmp11'*tmp11;
    sigma_g8 = sigma_g8 + tmp22;
end
sigma_g64 = sigma_g64*1/1053;
sigma_g8 = sigma_g8*1/1053;

% histogram('Categories', {'Cheetah', 'Grass'}, 'BinCounts',...
%     [sizeFG/total sizeBG/total]);
% xlabel('Class'), ylabel('Proportion'), title('Histogram of Priors')
% 
% cov_fg=cov(TrainsampleDCT_FG);
% cov_bg=cov(TrainsampleDCT_BG);
j = 1;
% [1 3 4 59 60 62 63 64]
% for i = 1:64
%     mu_fg = mean(TrainsampleDCT_FG(:,i));
%     mu_bg = mean(TrainsampleDCT_BG(:,i));
%     min_x = min(min(TrainsampleDCT_FG(:,i)), min(TrainsampleDCT_BG(:,i)));
%     max_x = max(max(TrainsampleDCT_FG(:,i)), max(TrainsampleDCT_BG(:,i)));
% %     min_x=min(TrainsampleDCT_FG(:,i));
% %     max_x=max(TrainsampleDCT_FG(:,i));
%     norm_fg = normpdf(min_x:0.0001:max_x,mu_fg, sqrt(sigma_c64(i,i)));
%     norm_bg = normpdf(min_x:0.0001:max_x,mu_bg, sqrt(sigma_g64(i,i)));
%     fig_num = ceil(i/8);
%     
%     figure(fig_num);
%     subplot(4,2,mod(i-1,8)+1), plot(min_x:0.0001:max_x,norm_fg,'-r'), hold on,...
%         plot(min_x:0.0001:max_x,norm_bg, 'b');
%     xlabel(['X' num2str(i) ' DCT Value']),...
%         title(['P(X' num2str(i) '|Cheetah) vs P(X' num2str(i) '|Grass)'])
%     legend('Cheetah', 'Grass');
%     j=j+1;
% end
% mod(i-1,8)+1
% % 
Ch = imread('cheetah.bmp');
Ch = im2double(Ch);
[x,y] = size(Ch);

count = 1;
for i=1:x-7
    for j=1:y-7
        blk = Ch(i:i+7,j:j+7);
        blkdct2 = dct2(blk);
        blkdct2_reordered(zig)=blkdct2;
        blkzz(count,:) = blkdct2_reordered;
        count = count + 1;
    end
end
% % % 
% % % % 
% % % % % 64-dimensional Gaussians
% % % % mu_b = mean(TrainsampleDCT_BG)';
% % % % mu_f = mean(TrainsampleDCT_FG)';
% % % % covar_b = cov(TrainsampleDCT_BG);
% % % % covar_f = cov(TrainsampleDCT_FG);
% % % % 
% % % % 
% % % % idx=1;
% % % % Seg(255,270)=0;
% % % % for i=1:x-7
% % % %     for j=1:y-7
% % % %         if mahal(blkzz(idx,:)',mu_f,covar_f,0.1919,64) < ...
% % % %                 mahal(blkzz(idx,:)',mu_b,covar_b,0.8081,64)
% % % %             Seg(i,j) = 1;
% % % %         else
% % % %             Seg(i,j) = 0;
% % % %         end
% % % %         idx = idx + 1;
% % % %     end
% % % % end
% % % % 
% % % 
% % % 
% % % %--------------------8-dimensional Gaussian--------------------------%
% % % 
% Try 1, 3, 12, 21, 32, 39, 48, 57
mu_b = mean(TrainsampleDCT_BG)';
mu_f = mean(TrainsampleDCT_FG)';
covar_b = cov(TrainsampleDCT_BG);
covar_f = cov(TrainsampleDCT_FG);


idx=1;
Seg8(255,270)=0;
Seg64(255,270)=0;
for i=1:x-7
    for j=1:y-7
        blk8dim = [blkzz(idx,1) blkzz(idx,3) blkzz(idx,10) blkzz(idx,38)...
            blkzz(idx,39) blkzz(idx,51) blkzz(idx,57) blkzz(idx,58)];
        if BDR(blk8dim',mu_f8,sigma_c8,pc,8) > ...
                BDR(blk8dim',mu_b8,sigma_g8,pg,8)
            Seg8(i,j) = 1;
        end
        if BDR(blkzz(idx,:)',mu_f,sigma_c64,pc,64) > ...
                BDR(blkzz(idx,:)',mu_b,sigma_g64,pg,64)
            Seg64(i,j) = 1;
        end
             
        idx = idx + 1;
    end
end


figure;
imagesc(Seg8)
colormap(gray(255))

figure;
imagesc(Seg64);
colormap(gray(255))


mask = imread('cheetah_mask.bmp');
Ch_adj = double(mask)/255;

figure;
C = imfuse(Seg8,Ch_adj,'blend','Scaling','joint');
imshow(C)

errors8=0;
errors64=0;
for x = 1:255
    for y=1:270
        if Ch_adj(x,y)~=Seg8(x,y)
            errors8=errors8+1;
        end
        if Ch_adj(x,y)~=Seg64(x,y)
            errors64=errors64+1;
        end
    end
end


% p_err64 = errors64/(255*270)*100;
% error64 = ['64-Dimensional Error: ' num2str(p_err64) '%'];
% disp(error64);

det8 = 0;
fal8=0;
det64 = 0;
fal64=0;
for x = 1:255
    for y = 1:270
        if Ch_adj(x,y) == 1 && Seg8(x,y) == 1
            det8 = det8+1;
        end
        if Ch_adj(x,y) == 0 && Seg8(x,y) == 1
            fal8 = fal8+1;
        end
        if Ch_adj(x,y) == 1 && Seg64(x,y) == 1
            det64 = det64+1;
        end
        if Ch_adj(x,y) == 0 && Seg64(x,y) == 1
            fal64 = fal64+1;
        end
    end
end


% Output our results (errors, detection, etc...)
p_err8 = errors8/(255*270)*100;
error8 = ['8-Dimensional Error Rate: ' num2str(p_err8) '%'];
disp(error8);
detection_rate8 = det8/(255*270*pc);
false_alarm8 = fal8 / (255*270*pg);
f_rate8 = ['8-Dimensional False Alarm Rate: ' num2str(false_alarm8)];
drate8 = ['8-Dimensional Detection Rate: ' num2str(detection_rate8)];
disp(drate8);
disp(f_rate8);
pr_error8 = (1-detection_rate8)*pc+pg*false_alarm8;
p_error8 = ['8-Dimensional Probability of Error: ' num2str(pr_error8)];
disp(p_error8);

disp('------------------------------------------------------------');
p_err64 = errors64/(255*270)*100;
error64 = ['64-Dimensional Error Rate: ' num2str(p_err64) '%'];
disp(error64);
detection_rate64 = det64/(255*270*pc);
false_alarm64 = fal64 / (255*270*pg);
f_rate64 = ['64-Dimensional False Alarm Rate: ' num2str(false_alarm64)];
drate64 = ['64-Dimensional Detection Rate: ' num2str(detection_rate64)];
disp(drate64);
disp(f_rate64);
pr_error64 = (1-detection_rate64)*pc+pg*false_alarm64;
p_error64 = ['64-Dimensional Probability of Error: ' num2str(pr_error64)];
disp(p_error64);

function mahalanobis_distance = mahal(x,mu,E,P,d)
% calculates mahalanobis distcance between x and mu of class distribution.
% x and mu are column vectors
d = -0.5*(x-mu)'*E^(-1)*(x-mu);
alpha = -0.5*log((2*pi)^d)*det(E)+2*log(P);

% d = (x-mu)'*inv(E)*(x-mu);
% alpha = log10(2*pi)^(d/2)*det(E)-2*log10(P);
mahalanobis_distance = d+alpha;
end

function bayes = BDR(x, mu, E, p, d)
bayes = 1/sqrt((2*pi)^d*det(E))*exp(-0.5*(x-mu)'*inv(E)*(x-mu))*p;
end







